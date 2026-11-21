"""CPU correctness gate on a real tiny mBART; no dataset training/download."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from transformers import MBartConfig, MBartForConditionalGeneration
from methods.registry import wrap_method
from methods.residual_effect.v1.lsr import paired_contrastive_loss


def backbone():
    config = MBartConfig(vocab_size=40, d_model=16, encoder_layers=4, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_position_embeddings=32, dropout=.1,
        attention_dropout=.1, activation_dropout=.1, encoder_layerdrop=0.,
        decoder_layerdrop=0., pad_token_id=1, bos_token_id=0, eos_token_id=2,
        decoder_start_token_id=2, use_cache=False)
    config._attn_implementation = 'eager'
    return MBartForConditionalGeneration(config)


def batch():
    return dict(input_ids=torch.tensor([[0, 5, 6, 2, 1], [0, 7, 8, 9, 2]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
        labels=torch.tensor([[0, 10, 11, 2, -100], [0, 12, 13, 14, 2]]))


class EffectGate(unittest.TestCase):
    def test_singleton_evaluation_keeps_logits_and_training_guard(self):
        torch.set_num_threads(1)
        single = {key: value[:1] for key, value in batch().items()}
        for method in ('ordinary-lsr-v1', 'ordinary-full-v1', 'effect-clrr-v1', 'effect-full-v1'):
            with self.subTest(method=method):
                model = wrap_method(backbone(), dict(method=method, alignment_weight=.1, temperature=.1))
                model.eval()
                with torch.no_grad():
                    self.assertEqual(model.auxiliary_loss(**single).item(), 0.)
                    result = model(**single)
                    self.assertTrue(torch.isfinite(result.loss))
                    decoder = model.prepare_decoder_input_ids_from_labels(single['labels'])
                    logits = model(input_ids=single['input_ids'], attention_mask=single['attention_mask'],
                                   decoder_input_ids=decoder).logits
                    torch.testing.assert_close(result.logits, logits)
                model.train()
                with self.assertRaises(ValueError):
                    model.auxiliary_loss(**single)

    def test_formula_and_target_detach(self):
        s = torch.eye(4, requires_grad=True)
        t = torch.eye(4, requires_grad=True)
        loss = paired_contrastive_loss(s, t, .1)
        expected = torch.log(torch.exp(torch.tensor(10.)) + 3) - 10
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-4)
        self.assertLess(loss, paired_contrastive_loss(s, t.roll(1, 0), .1))
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertTrue(torch.isfinite(s.grad).all())
        with self.assertRaises(ValueError):
            paired_contrastive_loss(s[:1], t[:1], .1)

    def test_all_branches_checkpoint_and_restore(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        initial = backbone().state_dict()
        for method in ('ordinary-lsr-v1', 'ordinary-full-v1', 'effect-clrr-v1', 'effect-full-v1'):
            with self.subTest(method=method):
                core = backbone(); core.load_state_dict(initial)
                model = wrap_method(core, dict(method=method, alignment_weight=.1, temperature=.1))
                model.train(); model.enable_input_require_grads()
                flags = [m.training for m in model.modules()]
                rng = torch.random.get_rng_state().clone()
                aux = model.auxiliary_loss(**batch())
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertEqual(flags, [m.training for m in model.modules()])
                aux.backward()
                self.assertGreater(core.model.encoder.layers[0].fc1.weight.grad.norm(), 0)
                self.assertIsNone(core.model.decoder.layers[0].fc1.weight.grad)
                self.assertTrue(torch.isfinite(aux))
                model.zero_grad(set_to_none=True)
                torch.manual_seed(123)
                plain = model(**batch()).loss
                plain.backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
                torch.manual_seed(123)
                checkpointed = model(**batch()).loss
                checkpointed.backward()
                torch.testing.assert_close(checkpointed, plain)
                for n, p in model.named_parameters():
                    if n in grads:
                        torch.testing.assert_close(p.grad, grads[n], atol=2e-6, rtol=2e-4)
                self.assertEqual(model.extra_parameter_count(), 0)
                self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in core.parameters()))
                self.assertEqual(model.content_mask(batch()['input_ids'], batch()['attention_mask']).tolist(),
                    [[False, True, True, False, False], [False, True, True, True, False]])
                model.eval()
                with torch.no_grad():
                    reference = model(**batch()).logits
                    inference = model(input_ids=batch()['input_ids'], attention_mask=batch()['attention_mask'],
                        decoder_input_ids=core.prepare_decoder_input_ids_from_labels(batch()['labels'])).logits
                torch.testing.assert_close(reference, inference)
                with tempfile.TemporaryDirectory() as folder:
                    model.save_pretrained(folder)
                    loaded = MBartForConditionalGeneration.from_pretrained(folder)
                    rebuilt = wrap_method(loaded, dict(method=method, alignment_weight=.1, temperature=.1)).eval()
                    with torch.no_grad():
                        torch.testing.assert_close(reference, rebuilt(**batch()).logits)

    def test_zero_effect_and_exception_restore(self):
        model = wrap_method(backbone(), dict(method='effect-clrr-v1', alignment_weight=.1, temperature=.1))
        model.train()
        model.base_model.strength = 0.
        loss = model.auxiliary_loss(**batch())
        torch.testing.assert_close(loss, torch.log(torch.tensor(2.)))
        self.assertTrue(torch.isfinite(loss))
        model.base_model.strength = .1
        flags = [m.training for m in model.modules()]
        with self.assertRaises(RuntimeError):
            with model.deterministic_view(False):
                raise RuntimeError('injected')
        self.assertEqual(model.base_model.strength, .1)
        self.assertEqual(flags, [m.training for m in model.modules()])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EffectGate))
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs/experiments/amis_residual_effect_preflight'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'CPU_GATE.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),
        tests=result.testsRun, torch=torch.__version__, device='cpu', training_started=False), indent=2), encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
