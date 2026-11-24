"""CPU gate for the frozen 2x2 Lexical Fork Preference (`lexical_fork/v1`) pilot bundle."""
import gc
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / 'src'), str(ROOT / 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import MBartForConditionalGeneration

from experiment_runner import engine as worker
from methods.lexical_fork.v1 import LexicalForkCollator, annotate_window_fork_counts
from test_lexical_fork_training import (
    LexicalForkTrainingGate,
    _AlphabeticForkTokenizer,
    _sample_selected_map,
    tiny_batch,
    tiny_mbart,
)


def main():
    torch.set_num_threads(2)
    worker.validate_environment(gpu=False)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(LexicalForkTrainingGate)
    )
    if not result.wasSuccessful():
        raise RuntimeError('Lexical Fork CPU gate failed')

    tok = _AlphabeticForkTokenizer()
    cases = []
    for spec in worker.plan()['runs']:
        torch.manual_seed(spec['seed'])
        run_spec = dict(spec, max_source_length=32, max_target_length=32, rollout_batch_size=8)
        model, metadata = worker.wrap_model(tiny_mbart(dropout=0.1, seed=spec['seed']), run_spec)
        if hasattr(model, 'set_tokenizer'):
            model.set_tokenizer(tok)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

        collator = LexicalForkCollator(tok, model, pad_to_multiple_of=8)
        raw_feats = [
            dict(
                input_ids=[4, 8 + i * 2, 10 + i * 2, 2],
                attention_mask=[1, 1, 1, 1],
                labels=[5, 12 + i * 2, 14 + i * 2, 2],
                row_id=i,
            )
            for i in range(4)
        ]
        batch = collator(raw_feats)
        sel_map = _sample_selected_map((0, 1))
        k_w = annotate_window_fork_counts([batch], sel_map)
        assert k_w == 2
        model.set_epoch_fork_state(2, float(run_spec.get('lambda_epoch2', 0.05)), sel_map)

        model.train()
        model(**batch).loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.last_diagnostics['window_selected'] == 2

        eval_sample = tiny_batch(batch_size=2, src_len=8, tgt_len=7, seed=spec['seed'])
        model.eval()
        with torch.no_grad():
            expected = model(
                input_ids=eval_sample['input_ids'],
                attention_mask=eval_sample['attention_mask'],
                labels=eval_sample['labels'],
            ).logits

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, state_dict=model.state_dict())
            base = MBartForConditionalGeneration.from_pretrained(Path(directory))
            restored, _ = worker.wrap_model(base, run_spec)
            restored.verify_checkpoint_metadata(directory)
            restored.restore_fork_state(directory)
            assert restored.active_epoch_idx == 2
            assert restored.selected_map == sel_map
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    restored(
                        input_ids=eval_sample['input_ids'],
                        attention_mask=eval_sample['attention_mask'],
                        labels=eval_sample['labels'],
                    ).logits,
                    expected,
                )
            del restored, base
            gc.collect()

        cases.append(dict(
            id=spec['id'],
            passed=True,
            finite_gradients=True,
            save_reload_logits_match=True,
            fork_state_restored=True,
            parameters=metadata,
        ))

    worker.write(worker.OUT / 'CPU_GATE.json', dict(
        passed=True,
        cases=cases,
        scientific_tests=result.testsRun,
        novelty_certified=False,
        bundle_sha256=worker.sha256(worker.OUT / 'bundle_manifest.json'),
    ))


if __name__ == '__main__':
    main()
