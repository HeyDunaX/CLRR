"""CPU-only formula, mask, margin, RNG and full tied-backbone VJP tests."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location(
    'goc_probe', Path(__file__).resolve().parents[1]/'scripts/analysis/probe_goc_zero_step.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class GOCZeroStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def samples(self, dtype=torch.float64):
        torch.manual_seed(132)
        return (torch.randn(7, 11, dtype=dtype), torch.randn(7, 11, dtype=dtype),
                torch.tensor([0, 3, -100, 10, 2, 1, -100]), torch.tensor([0, 1, 0, 1, 1, 0, 1]))

    def test_analytic_gradients_vs_autograd_fp64_and_fp32(self):
        for dtype, tolerance in ((torch.float64, 2e-14), (torch.float32, 3e-7)):
            a, b, y, rand = self.samples(dtype)
            expected = p.logit_gradients(a, b, y, rand)
            for method in p.METHODS:
                with self.subTest(dtype=dtype, method=method):
                    x, z = a.clone().requires_grad_(), b.clone().requires_grad_()
                    actual = torch.autograd.grad(p.scalar_loss(x, z, y, method, rand), (x, z))
                    for g, target in zip(actual, expected[method]):
                        torch.testing.assert_close(g, target/y.ne(-100).sum(), rtol=tolerance, atol=tolerance)

    def test_ties_swap_and_identical_views(self):
        # Equal gold probabilities with different non-gold distributions.
        a = torch.tensor([[.5, .4, .1]], dtype=torch.float64).log()
        b = torch.tensor([[.5, .1, .4]], dtype=torch.float64).log()
        y = torch.tensor([0])
        g = p.logit_gradients(a, b, y)
        torch.testing.assert_close(g['GOC2.50'][0], g['SYM2.50'][0])
        torch.testing.assert_close(g['GOC2.50'][1], g['SYM2.50'][1])
        a, b, y, _ = self.samples()
        forward, reverse = p.logit_gradients(a, b, y), p.logit_gradients(b, a, y)
        for method in ('W', 'AC', 'GOC1.25', 'GOC2.50', 'SYM2.50', 'INV2.50'):
            torch.testing.assert_close(forward[method][0], reverse[method][1])
            torch.testing.assert_close(forward[method][1], reverse[method][0])
        same = p.logit_gradients(a, a, y)
        for method in p.METHODS[1:]:
            for value in same[method]:
                self.assertEqual(float(value.abs().max()), 0)

    def test_mask_and_chunking_and_scaling(self):
        a, b, y, rand = self.samples()
        full = p.logit_gradients(a, b, y, rand)
        for method in p.METHODS:
            self.assertEqual(float(full[method][0][y == -100].abs().sum()), 0)
            chunks = [p.logit_gradients(a[i:i+2], b[i:i+2], y[i:i+2], rand[i:i+2])[method]
                      for i in range(0, len(y), 2)]
            for view in (0, 1):
                torch.testing.assert_close(full[method][view], torch.cat([c[view] for c in chunks]))
        for view in (0, 1):
            torch.testing.assert_close(full['GOC2.50'][view], 2*full['GOC1.25'][view])
        padded = p.logit_gradients(a, b, torch.full_like(y, -100))
        for value in padded.values():
            self.assertEqual(float(value[0].abs().sum()+value[1].abs().sum()), 0)

    def test_margin_direction_finite_difference_and_competitor_tie(self):
        z = torch.tensor([[2., 1., 0.], [0., 2., 2.]], dtype=torch.float64)
        g = torch.tensor([[.3, .7, -.5], [0., .5, -.25]], dtype=torch.float64)
        y = torch.tensor([0, 0])
        def margin(x):
            return x[:, 0] - x[:, 1:].max(-1).values
        eps = 1e-6
        expected = (margin(z-eps*g)-margin(z))/eps
        torch.testing.assert_close(p.margin_direction(z, g, y), expected, atol=1e-9, rtol=1e-9)
        self.assertEqual(float(p.margin_direction(z, g, torch.tensor([-100, -100])).abs().sum()), 0)

    def test_entropy_derivative_sign(self):
        a, b, y, rand = self.samples()
        # Direct autograd checks -common entropy direction, independent of helper algebra.
        common = p.logit_gradients(a, b, y)['COMMON'][0]
        x = a.clone().requires_grad_()
        logp = x.log_softmax(-1)
        entropy = -(logp.exp()*logp).sum(-1)
        gh = torch.autograd.grad(entropy.sum(), x)[0]
        reference = -(gh*common).sum(-1)
        pa, la = a.softmax(-1), a.log_softmax(-1)
        analytic = (pa*(la+entropy.detach().unsqueeze(-1))*common).sum(-1)
        torch.testing.assert_close(reference, analytic, atol=1e-14, rtol=1e-14)

    def test_exact_rng_replay_and_separate_teacher_rng(self):
        torch.manual_seed(0)
        layer = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(.4)).train()
        x = torch.ones(4, 8)
        def evaluate(seed):
            p.set_seed(seed)
            return layer(x)
        s1, s2 = p.seed_views(2, 4, 1)
        torch.testing.assert_close(evaluate(s1), evaluate(s1), rtol=0, atol=0)
        self.assertGreater(float((evaluate(s1)-evaluate(s2)).abs().max()), 0)
        self.assertEqual(len({s for split in range(2) for b in range(32) for r in range(8)
                              for s in p.seed_views(b, r, split)}), 2*32*8*2)

    def test_chunked_full_backbone_vjp_matches_native_tied_model(self):
        from transformers import MBartConfig, MBartForConditionalGeneration
        cfg = MBartConfig(vocab_size=19, d_model=8, encoder_layers=1, decoder_layers=1,
                          encoder_attention_heads=2, decoder_attention_heads=2,
                          encoder_ffn_dim=16, decoder_ffn_dim=16, max_position_embeddings=32,
                          pad_token_id=1, decoder_start_token_id=2, dropout=.2, attention_dropout=.1,
                          activation_dropout=.1, encoder_layerdrop=0., decoder_layerdrop=0.)
        cfg._attn_implementation = 'sdpa'
        torch.manual_seed(101)
        model = MBartForConditionalGeneration(cfg).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        batch = dict(input_ids=torch.tensor([[4, 5, 2], [6, 2, 1]]),
                     attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
                     decoder_input_ids=torch.tensor([[2, 4, 5], [2, 3, 1]]))
        labels = torch.tensor([[4, 5, 2], [3, 2, -100]])
        seeds = p.seed_views(0, 0)
        original = p.parameter_hash(model)
        for method in p.B_METHODS:
            model.zero_grad(set_to_none=True)
            z = []
            for seed in seeds:
                p.set_seed(seed)
                z.append(model(**batch, use_cache=False).logits.reshape(-1, cfg.vocab_size))
            p.scalar_loss(*z, labels.reshape(-1), method).backward()
            native = {k: v.grad.detach().clone() for k, v in model.named_parameters()}
            with patch.object(p, 'check_budget'):
                actual, errors = p.head_backbone_gradient(model, batch, labels, seeds, method, 0.)
            self.assertEqual(errors, [0., 0.])
            for name in native:
                torch.testing.assert_close(actual[name], native[name], atol=2e-6, rtol=3e-4,
                                           msg=f'{method}: {name}')
        self.assertEqual(original, p.parameter_hash(model))

    def test_gold_logit_push_does_not_guarantee_margin_improvement(self):
        teacher = torch.tensor([[.40, .59, .01]], dtype=torch.float64).log()
        student = torch.tensor([[.30, .35, .35]], dtype=torch.float64).log()
        y = torch.tensor([0])
        g1, g2 = p.logit_gradients(teacher, student, y)['GOC2.50']
        self.assertEqual(float(g1.abs().sum()), 0.)
        self.assertGreater(float(-g2[0, 0]), 0.)
        self.assertLess(float(p.margin_direction(student, g2, y)[0]), 0.)

    def test_random_teacher_expectation_equals_deterministic_sym_sg(self):
        a, b, y, _ = self.samples()
        zero = p.logit_gradients(a, b, y, torch.zeros_like(y))['RAND2.50']
        one = p.logit_gradients(a, b, y, torch.ones_like(y))['RAND2.50']
        sym = p.logit_gradients(a, b, y)['SYM2.50']
        for v in (0, 1):
            torch.testing.assert_close((zero[v]+one[v])/2, sym[v])

    def test_paired_sentence_summary_no_special_double_count(self):
        records = []
        for pair in range(8):
            records.append(dict(split='validation', row_id=0, position=0, pair=pair,
                                token_id=2, special=True, frequency_group='special', strong=True,
                                case2_margin=float(pair < 3), measurement=float(pair)))
        summary = p.summarize_part_a(records, {})['groups']
        self.assertEqual(summary['validation/special/all']['observations'], 8)
        self.assertEqual(summary['validation/special/all']['case2_occurrence_histogram'], {'3': 1})
        self.assertEqual(summary['validation/special/all']['metrics']['measurement']['mean'], 3.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
