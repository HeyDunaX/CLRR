"""CPU correctness for the exact comparator wrappers and checkpoint exporter."""
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from transformers import (MBartConfig, M2M100Config, MT5Config, T5Config,
                          AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments)
from experiment_runner import engine as worker


def base_model(backbone):
    common = dict(vocab_size=96, d_model=32, pad_token_id=0, eos_token_id=2,
                  decoder_start_token_id=2)
    if backbone in ('mbart-large-50', 'nllb-200'):
        cls = MBartConfig if backbone == 'mbart-large-50' else M2M100Config
        config = cls(**common, encoder_layers=6, decoder_layers=2, encoder_attention_heads=4,
                     decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
                     max_position_embeddings=64, dropout=0., attention_dropout=0., activation_dropout=0.,
                     encoder_layerdrop=0., decoder_layerdrop=0.)
    else:
        cls = MT5Config if backbone == 'mt5-small' else T5Config
        config = cls(**common, d_ff=64, d_kv=8, num_heads=4, num_layers=8,
                     num_decoder_layers=2, dropout_rate=0., tie_word_embeddings=True)
    return AutoModelForSeq2SeqLM.from_config(config)


def main():
    worker.validate_environment(gpu=False)
    torch.set_num_threads(2)
    results = []
    with tempfile.TemporaryDirectory(prefix='controls-cpu-') as tmp:
        for spec in worker.plan()['runs']:
            torch.manual_seed(42)
            model, stats = worker.wrap_model(base_model(spec['backbone']), spec)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            assert stats['trainable_params'] > 0
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout): module.p = 0.
            model.train()
            ids = torch.randint(3, 90, (4, 12))
            labels = torch.randint(3, 90, (4, 10)); labels[:, -2:] = -100
            batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': labels}
            model.zero_grad()
            loss = model(**batch).loss
            assert torch.isfinite(loss)
            loss.backward()
            grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
            assert any(g is not None and g.abs().sum() > 0 for _, g in grads)
            assert all(g is None or torch.isfinite(g).all() for _, g in grads)
            if spec['method'] == 'middle-alignment':
                model.zero_grad()
                source = model._extract_middle_rep(ids, torch.ones_like(ids))
                target_ids = labels.masked_fill(labels == -100, 0)
                target = model._extract_middle_rep(target_ids, (labels != -100).long())
                source.retain_grad(); target.retain_grad()
                logits = torch.nn.functional.normalize(source, dim=-1) @ torch.nn.functional.normalize(target, dim=-1).T / spec['temperature']
                index = torch.arange(4)
                align = .5 * (torch.nn.functional.cross_entropy(logits, index) + torch.nn.functional.cross_entropy(logits.T, index))
                align.backward()
                assert source.grad.abs().sum() > 0 and target.grad.abs().sum() > 0
                with torch.no_grad():
                    ce = model.base_model(**batch).loss
                    total = model(**batch).loss
                torch.testing.assert_close(total, ce + spec['alignment_weight'] * align.detach())
            # Trainer must scale the CE+auxiliary microbatch mean exactly once.
            folder = Path(tmp) / spec['id']
            trainer = worker.ControlTrainer(model=model,
                args=Seq2SeqTrainingArguments(output_dir=str(folder), use_cpu=True, report_to=[],
                    gradient_accumulation_steps=4, save_safetensors=False), generation_dtype='fp32')
            assert trainer.model_accepts_loss_kwargs is False
            model.zero_grad()
            trainer.compute_loss(model, dict(batch), num_items_in_batch=torch.tensor(999)).backward()
            first = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad()
            model(**batch).loss.backward()
            for n, p in model.named_parameters():
                if n in first: torch.testing.assert_close(p.grad, first[n])
            model.eval()
            with torch.no_grad(): expected = model(**batch).logits.clone()
            trainer._save(str(folder))
            trainer.state.best_model_checkpoint = str(folder)
            with torch.no_grad():
                next(p for p in model.parameters() if p.requires_grad).add_(.01)
            trainer._load_best_model()
            with torch.no_grad(): torch.testing.assert_close(model(**batch).logits, expected, rtol=1e-5, atol=1e-6)
            if (folder / 'adapter_config.json').exists():
                base = base_model(spec['backbone'])
                # PEFT adapters do not contain frozen base weights: reload the exact base.
                original = model.get_base_model()
                if spec['method'] == 'strong-lora-b':
                    # Use a clean model with original frozen state, then adapter restores saved modules.
                    state = original.state_dict()
                    clean = {}
                    for k, v in state.items():
                        if '.lora_' in k or '.modules_to_save.' in k: continue
                        clean[k.replace('.base_layer.', '.').replace('.original_module.', '.')] = v
                    base.load_state_dict(clean, strict=True)
                else:
                    clean = {k.replace('.base_layer.', '.'): v for k, v in original.state_dict().items() if '.lora_' not in k}
                    base.load_state_dict(clean, strict=True)
                restored = worker.PeftModel.from_pretrained(base, str(folder))
                if spec['method'] == 'strong-lora-b':
                    core = restored.get_base_model(); weight = core.model.shared.modules_to_save['default'].weight
                    for module in (core.model.encoder.embed_tokens, core.model.decoder.embed_tokens, core.lm_head):
                        assert torch.equal(module.modules_to_save['default'].weight, weight)
                        module.modules_to_save['default'].weight = weight
            else:
                base, info = AutoModelForSeq2SeqLM.from_pretrained(folder, output_loading_info=True)
                assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
                restored, _ = worker.wrap_model(base, spec)
            restored.eval()
            with torch.no_grad(): actual = restored(**batch).logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            results.append({'id': spec['id'], 'passed': True, 'finite_gradients': True,
                'save_reload_logits_match': True, 'accumulation_loss_contract': True,
                'trainable_params': stats['trainable_params']})
            print('CPU_GATE_PASS', spec['id'], flush=True)
            del trainer, model, restored
    worker.write(worker.OUT / 'CPU_GATE.json', {'passed': True, 'cases': results,
        'bundle_sha256': worker.sha256(worker.OUT / 'bundle_manifest.json')})
    print('CPU_ALL_PASS', len(results), flush=True)


if __name__ == '__main__': main()
