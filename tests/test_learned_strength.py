"""Check donor replay, detached edges, zero initialization and persisted scalars."""
import copy
import tempfile
import unittest

import torch
from transformers import MBartForConditionalGeneration

from test_alignment_residual import native, batch
from methods.registry import wrap_method
from methods.learned_strength import METHODS, STATE_KEY


def model_for(method, core=None):
    return wrap_method(native() if core is None else core,
        dict(method=method, distance=2, strength=0., jepa_weight=.1 if METHODS[method] else 0.))


def route(model, method):
    return model.base_model if METHODS[method] else model


class StrengthGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_forward_generation_and_gradients(self):
        torch.manual_seed(42)
        base = native().eval()
        model = model_for('learned-strength-clrr-v1', copy.deepcopy(base)).eval()
        torch.testing.assert_close(base(**batch()).logits, model(**batch()).logits, rtol=0, atol=0)
        inputs = {k: v for k, v in batch().items() if k != 'labels'}
        with torch.no_grad():
            self.assertTrue(torch.equal(base.generate(**inputs, max_new_tokens=4),
                                        model.generate(**inputs, max_new_tokens=4)))
        model.train()(**batch()).loss.backward()
        self.assertGreater(model.route_strengths.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.route_strengths.grad).all())
        self.assertGreater(model.get_encoder().layers[0].fc1.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.extra_parameter_count(), 2)

    def test_skip_donor_is_detached(self):
        model = model_for('learned-strength-clrr-v1')
        donor = torch.randn(2, 3, 16, requires_grad=True)
        recipient = torch.randn_like(donor, requires_grad=True)
        model._cache['encoder'][0] = donor
        model.route_strengths.data.fill_(.2)
        model._make_hook('encoder', 2)(None, (), recipient).sum().backward()
        self.assertIsNone(donor.grad)
        self.assertIsNotNone(recipient.grad)

    def test_checkpoint_replay_full_lsr_and_accumulation(self):
        for method in METHODS:
            torch.manual_seed(42)
            plain = model_for(method)
            route(plain, method).route_strengths.data.fill_(.12)
            # Build independently to avoid copying hooks bound to another model.
            replay = model_for(method)
            replay.load_state_dict(plain.state_dict())
            replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            plain.train(); replay.train()
            for model in (plain, replay):
                for _ in range(2):
                    (model(**batch()).loss / 2).backward()
            left, right = dict(plain.named_parameters()), dict(replay.named_parameters())
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assertEqual(left[key].grad is None, right[key].grad is None, key)
                if left[key].grad is not None:
                    torch.testing.assert_close(left[key].grad, right[key].grad,
                        rtol=3e-4, atol=2e-6, msg=key)
            self.assertEqual(route(replay, method)._checkpoint_context, {})

    def test_save_load_and_no_cross_forward_cache(self):
        for method in METHODS:
            model = model_for(method).eval()
            route(model, method).route_strengths.data.copy_(torch.tensor([.12, -.07]))
            expected = model(**batch()).logits.detach()
            model(**{k: v[:1] for k, v in batch().items()})
            torch.testing.assert_close(model(**batch()).logits, expected)
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                base, info = MBartForConditionalGeneration.from_pretrained(folder, output_loading_info=True)
                self.assertEqual(info['unexpected_keys'], [STATE_KEY])
                restored = model_for(method, base).eval()
                route(restored, method).restore_scalars(folder)
                torch.testing.assert_close(restored(**batch()).logits, expected, rtol=0, atol=0)
                self.assertEqual(set(restored.state_dict()), set(model.state_dict()))


if __name__ == '__main__':
    unittest.main()
