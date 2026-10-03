"""Checks for decoder padding, provenance, precision, and gradient logging repairs."""
import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/revalidation'))
from amis_rewire.metrics import build_compute_metrics, generation_metrics, score_prediction_csv
from amis_rewire.evaluation import generation_precision
from run_no_stop_gradient import GradientNormTracker


class RepairTests(unittest.TestCase):
    def test_ignored_prediction_ids_use_tokenizer_pad(self):
        class Tokenizer:
            pad_token_id = 7
            def batch_decode(self, ids, skip_special_tokens=True):
                # Token 0 is ordinary text for this tokenizer, not padding.
                return [' '.join('bad' if i == 0 else 'good' for i in row if i != 7)
                        for row in np.asarray(ids)]
        actual = build_compute_metrics(Tokenizer(), ['good good'], '13a')(
            (np.array([[1, 2, -100]]), np.array([[99, 99, -100]])))
        self.assertEqual(actual, generation_metrics(['good good'], ['good good'], '13a'))

    def test_saved_target_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            reference, prediction = [Path(folder) / x for x in ('ref.csv', 'pred.csv')]
            for path, fields, row in ((reference, ['source', 'target'], ['s', 'correct']),
                                     (prediction, ['source', 'target', 'prediction'], ['s', 'wrong', 'p'])):
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream); writer.writerow(fields); writer.writerow(row)
            with self.assertRaisesRegex(ValueError, 'target mismatch'):
                score_prediction_csv(prediction, reference, '13a')

    def test_precision_context_restores_forward_and_tf32_after_error(self):
        model = torch.nn.Linear(2, 2)
        original = model.forward
        wrapped = lambda *a, **k: original(*a, **k)
        model._original_forward = original
        model.forward = wrapped
        previous = torch.backends.cuda.matmul.allow_tf32
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with generation_precision(model, 'fp32'):
                self.assertEqual(model.forward, original)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                raise RuntimeError('probe')
        self.assertIs(model.forward, wrapped)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, previous)
        with self.assertRaises(ValueError):
            with generation_precision(model, 'fp23'):
                pass

    def test_gradient_tracker_uses_logged_norm_after_gradients_are_cleared(self):
        tracker = GradientNormTracker()
        tracker.on_log(None, SimpleNamespace(global_step=3, epoch=.5), None, logs={'loss': 1})
        self.assertEqual(tracker.trace, [])
        tracker.on_log(None, SimpleNamespace(global_step=3, epoch=.5), None, logs={'grad_norm': 2.5})
        self.assertEqual(tracker.trace[0]['total_grad_norm'], 2.5)
        self.assertEqual(tracker.trace[0]['measurement'], 'trainer_logged_pre_clip_global_norm')

    def test_nllb_import_with_installed_src_package_layout(self):
        from nllb_suite import modeling_nllb
        self.assertTrue(callable(modeling_nllb.load_nllb_model))

    def test_comparator_checkpoint_preserves_base_weights_and_legacy_load(self):
        from transformers import T5Config, T5ForConditionalGeneration
        from comparative_baselines.middle_align_acl2025.model import MiddleAlignMT5
        config = T5Config(vocab_size=16, d_model=8, d_ff=16, num_layers=2,
                          num_decoder_layers=1, num_heads=2,
                          pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
        for cls in (MiddleAlignMT5,):
            wrapper = cls(T5ForConditionalGeneration(config))
            state = wrapper.state_dict()
            self.assertFalse(any(key.startswith('base_model.') for key in state))
            restored = T5ForConditionalGeneration(config)
            restored.load_state_dict(state, strict=True)
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]))
            legacy = {'base_model.' + key: value for key, value in state.items()}
            self.assertEqual(wrapper.load_state_dict(legacy, strict=True).missing_keys, [])


if __name__ == '__main__':
    unittest.main()
