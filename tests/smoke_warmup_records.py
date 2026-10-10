"""CPU/GPU smoke for production warmup schedules, snapshots and reload."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import MBartConfig, MBartForConditionalGeneration, Seq2SeqTrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from amis_rewire.evaluation import PrecisionSeq2SeqTrainer
from amis_rewire.train import RoutingWarmupCallback
from amis_rewire.scientific_records import ScientificRecordsCallback
from types import SimpleNamespace
from transformers.trainer_callback import TrainerState


def main():
    torch.set_num_threads(2)
    cuda = '--cuda' in sys.argv
    config = MBartConfig(vocab_size=32, d_model=16, encoder_layers=3, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_position_embeddings=32, pad_token_id=1, eos_token_id=2,
        bos_token_id=0, decoder_start_token_id=2, dropout=0, attention_dropout=0)
    torch.manual_seed(42)
    raw = MBartForConditionalGeneration(config)
    x = torch.tensor([[4, 5, 2]])
    labels = torch.tensor([[6, 7, 2]])
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        for full in (False, True):
            base = copy.deepcopy(raw)
            rewire = CrossLayerResidualRewire(base, distance=2, strength=.1, stack='encoder')
            model = JEPAGuidedSeq2SeqLM(rewire, jepa_weight=.1) if full else rewire
            folder = Path(temporary) / str(full)
            args = Seq2SeqTrainingArguments(output_dir=str(folder), report_to=[], save_safetensors=False, use_cpu=not cuda)
            trainer = PrecisionSeq2SeqTrainer(model=model, args=args, eval_rewire_strength=.1)
            assert trainer._find_rewire(raw) is None
            assert trainer._find_rewire(model) is rewire
            callback = RoutingWarmupCallback(.1, .2)
            state = TrainerState(max_steps=20, global_step=0, epoch=1, log_history=[{'eval_loss': 2.}],
                                    best_model_checkpoint='checkpoint-4', best_metric=1.)
            control = SimpleNamespace(should_training_stop=False)
            callback.on_train_begin(args, state, control, model=model)
            records = ScientificRecordsCallback(folder, .1, .1 if full else 0, .2, None, '', 'tiny')
            for step in range(5):
                state.global_step = step
                callback.on_step_begin(args, state, control, model=model)
                records.on_step_begin(args, state, control, model=model)
                assert abs(rewire.strength - .1 * min(1, step / 4)) < 1e-12
            # alpha0 matches clean raw, alpha.1 changes logits; cache is fresh for each call.
            model.eval()
            device = next(model.parameters()).device
            raw.to(device)
            rewire.strength = 0
            with torch.no_grad():
                routed0 = rewire(input_ids=x.to(device), attention_mask=x.ne(1).to(device), labels=labels.to(device)).logits.cpu()
                reference = raw.eval()(input_ids=x.to(device), attention_mask=x.ne(1).to(device), labels=labels.to(device)).logits.cpu()
            assert torch.equal(routed0, reference)
            rewire.strength = .1
            with torch.no_grad():
                routed1 = rewire(input_ids=x.to(device), attention_mask=x.ne(1).to(device), labels=labels.to(device)).logits.cpu()
            assert not torch.equal(routed1, reference)
            if cuda:
                model.cuda().train()
                model.gradient_checkpointing_enable()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=x.cuda(), attention_mask=x.ne(1).cuda(), labels=labels.cuda()).loss
                loss.backward()
                assert torch.isfinite(loss) and any(p.grad is not None for p in base.parameters())
            state.global_step = 4
            trainer._save(str(folder / 'checkpoint-4'))
            records.on_save(args, state, control)
            assert (folder / 'checkpoint-4/training_args.bin').exists()
            assert (folder / 'trajectory/epoch-01-step-4.zip').exists()
            records.on_train_end(args, state, control)
            assert json.loads((folder / 'eval_loss_trace.json').read_text())
            from transformers import AutoModelForSeq2SeqLM
            restored, info = AutoModelForSeq2SeqLM.from_pretrained(folder / 'checkpoint-4', output_loading_info=True)
            assert not any(info[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
            assert all(torch.equal(v.cpu(), restored.state_dict()[k]) for k, v in base.state_dict().items())
            # Trainer restore must return the exact saved weights after an actual mutation.
            trainer.state.best_model_checkpoint = str(folder / 'checkpoint-4')
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1)
            trainer._load_best_model()
            assert all(torch.equal(v.cpu(), restored.state_dict()[k]) for k, v in base.state_dict().items())
            results.append({'full': full, 'cuda_bf16': cuda, 'schedule': 'PASS', 'alpha0_raw': 'PASS',
                            'snapshot': 'PASS', 'strict_reload': 'PASS', 'best_reload': 'PASS'})
    out = Path(sys.argv[sys.argv.index('--result-dir')+1]) if '--result-dir' in sys.argv else ROOT / 'outputs_rebuttal/amis_routing_warmup_20261004_v2'
    out.mkdir(parents=True, exist_ok=True)
    (out / ('SMOKE_GPU_PASS.json' if cuda else 'SMOKE_CPU_PASS.json')).write_text(json.dumps(results, indent=2))
    print('WARMUP_SMOKE_PASS', json.dumps(results))


if __name__ == '__main__':
    main()
