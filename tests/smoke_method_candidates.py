"""Correctness gates for the four experimental methods; no downloaded weights."""
import copy
import json
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from transformers import MBartConfig, MBartForConditionalGeneration, Seq2SeqTrainingArguments
from experiment_runner import engine as worker
from method_candidates.centered_lsr import centered_cosine_loss
from method_candidates.precache import PrecacheResidualRewire
from amis_rewire.modeling import CrossLayerResidualRewire


def tiny():
    return MBartForConditionalGeneration(MBartConfig(vocab_size=96, d_model=32,
        encoder_layers=6, decoder_layers=2, encoder_attention_heads=4, decoder_attention_heads=4,
        encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=2, decoder_start_token_id=2,
        dropout=0., attention_dropout=0., activation_dropout=0., encoder_layerdrop=0., decoder_layerdrop=0.))


def main():
    torch.set_num_threads(2)
    worker.validate_environment(gpu=False)
    source = torch.randn(4, 12, requires_grad=True)
    target = torch.randn(4, 12, requires_grad=True)
    loss = centered_cosine_loss(source, target)
    loss.backward()
    assert source.grad.abs().sum() > 0 and target.grad is None
    s = source.detach().clone().requires_grad_()
    t = target.detach()
    manual = (1 - torch.nn.functional.cosine_similarity(s-s.mean(0), t-t.mean(0), dim=-1, eps=1e-6)).mean()
    manual.backward()
    torch.testing.assert_close(manual.detach(), loss.detach())
    torch.testing.assert_close(s.grad, source.grad)
    torch.testing.assert_close(centered_cosine_loss(s+10, t-8), loss.detach(), atol=2e-6, rtol=1e-5)
    for count in (1, 4):
        z = torch.ones(count, 12, requires_grad=True)
        degenerate = centered_cosine_loss(z, z)
        degenerate.backward()
        assert torch.isfinite(degenerate) and torch.isfinite(z.grad).all()
    # Direct donor formula + detach; layer 4 (zero-based) distinguishes pre/post.
    for cls in (PrecacheResidualRewire, CrossLayerResidualRewire):
        model = cls(tiny(), distance=2, strength=.1)
        raw = [torch.full((2, 3, 32), float(i+1), requires_grad=True) for i in range(6)]
        outputs = []
        with model._fresh_cache():
            for i, h in enumerate(raw):
                out = model._make_hook('encoder', i)(None, (), (h,))[0]
                expected = h if i < 2 else h + .1 * (raw[i-2].detach() if cls is PrecacheResidualRewire else outputs[i-2].detach())
                torch.testing.assert_close(out, expected)
                assert not model._cache['encoder'][i].requires_grad
                outputs.append(out)
            outputs[-1].sum().backward()
            assert raw[-1].grad is not None and all(h.grad is None for h in raw[:-1])
    cases = []
    with tempfile.TemporaryDirectory(prefix='candidates-gate-') as temp:
        for spec in worker.plan()['runs']:
            torch.manual_seed(42)
            base = tiny()
            reference = copy.deepcopy(base)
            model, meta = worker.wrap_model(base, spec)
            assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())
            batch = {'input_ids': torch.randint(3, 90, (4, 12)), 'attention_mask': torch.ones(4, 12, dtype=torch.long),
                     'labels': torch.randint(3, 90, (4, 10))}
            batch['attention_mask'][:, -2:] = 0
            batch['labels'][:, -2:] = -100
            model.train()
            out = model(**batch)
            out.loss.backward()
            plain = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            assert plain and all(torch.isfinite(g).all() for g in plain.values())
            model.zero_grad()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            checkpointed = model(**batch)
            torch.testing.assert_close(checkpointed.loss, out.loss)
            checkpointed.loss.backward()
            for n, p in model.named_parameters():
                if n in plain: torch.testing.assert_close(p.grad, plain[n], rtol=2e-4, atol=2e-6)
            # Trainer accumulation must divide each microbatch exactly once.
            folder = Path(temp) / spec['id']
            trainer = worker.ControlTrainer(model=model, args=Seq2SeqTrainingArguments(
                output_dir=str(folder), use_cpu=True, report_to=[], gradient_accumulation_steps=4,
                save_safetensors=False), generation_dtype='fp32')
            assert trainer.model_accepts_loss_kwargs is False
            trainer.current_gradient_accumulation_steps = 4
            model.zero_grad()
            returned = trainer.training_step(model, dict(batch), num_items_in_batch=torch.tensor(999))
            torch.testing.assert_close(returned, out.loss.detach()/4, rtol=2e-4, atol=2e-6)
            for n, p in model.named_parameters():
                if n in plain: torch.testing.assert_close(p.grad, plain[n]/4, rtol=2e-4, atol=2e-6)
            model.eval()
            with torch.no_grad(): expected = model(**batch).logits.clone()
            trainer._save(str(folder))
            trainer.state.best_model_checkpoint = str(folder)
            with torch.no_grad(): next(model.parameters()).add_(.1)
            trainer._load_best_model()
            with torch.no_grad(): torch.testing.assert_close(model(**batch).logits, expected)
            restored = MBartForConditionalGeneration.from_pretrained(folder)
            restored, _ = worker.wrap_model(restored, spec)
            restored.eval()
            with torch.no_grad(): torch.testing.assert_close(restored(**batch).logits, expected)
            cases.append({'id': spec['id'], 'checkpoint_gradient_match': True, 'accumulation4_match': True,
                          'save_reload_match': True, 'finite_gradients': True, 'extra_parameters': 0})
    worker.write(worker.OUT / 'CPU_GATE.json', {'passed': True, 'cases': cases,
        'bundle_sha256': worker.sha256(worker.OUT / 'bundle_manifest.json'),
        'centered_formula_gradient_match': True, 'target_stop_gradient': True,
        'precache_formula_detach_match': True, 'degenerate_batch_finite': True})
    print('CPU_GATE_PASS', json.dumps(cases))


if __name__ == '__main__': main()
