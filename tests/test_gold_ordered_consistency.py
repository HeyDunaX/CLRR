"""Scientific, gradient, replay, and persistence gates for Gold-Ordered Consistency (GOC)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / 'src'), str(ROOT / 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration

from test_alignment_residual import native, batch
from methods.paired_route.v1.model import paired_loss
from methods.decoupled_consistency.v1.model import consistency_loss
from methods.gold_ordered_consistency import GoldOrderedConsistencyMT, METHODS, goc_loss
from methods.gold_ordered_consistency.v1.validation import validate_recipe


def _analytic_aux_gradients(z1, z2, labels, mode, beta):
    z1, z2 = z1.float(), z2.float()
    valid = labels.ne(-100)
    safe = labels.masked_fill(~valid, 0).unsqueeze(-1)
    a, b = z1.log_softmax(-1), z2.log_softmax(-1)
    p, q = a.exp(), b.exp()
    ag, bg = a.gather(-1, safe).squeeze(-1), b.gather(-1, safe).squeeze(-1)
    winner = torch.where(ag > bg, 1., torch.where(ag < bg, 0., .5)).unsqueeze(-1)
    delta = a - b
    d = p - q
    if mode == 'rdrop':
        kl_pq = (p * delta).sum(-1, keepdim=True)
        kl_qp = -(q * delta).sum(-1, keepdim=True)
        g1 = .5 * beta * (p * (delta - kl_pq) + p - q)
        g2 = .5 * beta * (q * (-delta - kl_qp) + q - p)
    elif mode == 'goc':
        g1 = beta * (1. - winner) * d
        g2 = -beta * winner * d
    elif mode == 'sym_sg':
        g1 = .5 * beta * d
        g2 = -.5 * beta * d
    elif mode == 'inv':
        g1 = beta * winner * d
        g2 = -beta * (1. - winner) * d
    else:
        raise ValueError(mode)
    mask = valid.unsqueeze(-1)
    denom = valid.sum().to(z1.dtype)
    return (g1 * mask) / denom, (g2 * mask) / denom


class GoldOrderedConsistencyGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_losses_and_gradients_all_modes_and_weights(self):
        torch.manual_seed(2026)
        a = torch.randn(2, 4, 17, requires_grad=True)
        b = torch.randn(2, 4, 17, requires_grad=True)
        labels = torch.tensor([[3, 4, 2, -100], [5, 1, 2, 6]])
        for mode in ('goc', 'sym_sg', 'inv', 'rdrop'):
            for beta in (1.25, 2.5):
                with self.subTest(mode=mode, beta=beta):
                    a.grad = b.grad = None
                    total, ce, kl, stats = goc_loss(a, b, labels, mode=mode, beta=beta, chunk_size=2)
                    total_full, ce_full, kl_full, _ = goc_loss(a, b, labels, mode=mode, beta=beta, chunk_size=99)
                    torch.testing.assert_close(total, total_full)
                    torch.testing.assert_close(ce, ce_full)
                    torch.testing.assert_close(kl, kl_full)
                    self.assertEqual(stats['valid_tokens'], 7)
                    self.assertEqual(stats['tie_fraction'], 0.)
                    if mode == 'rdrop':
                        ref_total, ref_ce, ref_kl = paired_loss(a, b, labels, beta, chunk_size=2)
                        torch.testing.assert_close(total, ref_total)
                        torch.testing.assert_close(ce, ref_ce)
                        torch.testing.assert_close(kl, ref_kl)
                        if beta == 1.25:
                            w_total, w_ce, w_gold, w_rest = consistency_loss(a, b, labels, 'full', 1.25, 2)
                            torch.testing.assert_close(total, w_total)
                            torch.testing.assert_close(ce, w_ce)
                            torch.testing.assert_close(kl, w_gold + w_rest)
                    (beta * kl).backward()
                    g1_exp, g2_exp = _analytic_aux_gradients(a.detach(), b.detach(), labels, mode, beta)
                    torch.testing.assert_close(a.grad, g1_exp, atol=2e-7, rtol=2e-5)
                    torch.testing.assert_close(b.grad, g2_exp, atol=2e-7, rtol=2e-5)
                    self.assertEqual(float(a.grad[0, 3].abs().sum() + b.grad[0, 3].abs().sum()), 0.)

    def test_ties_swap_symmetry_and_zero_at_identical_views(self):
        # Exact tie on gold (index 0) with different non-gold distributions.
        a = torch.tensor([[[.5, .4, .1]]], dtype=torch.float32).log().requires_grad_(True)
        b = torch.tensor([[[.5, .1, .4]]], dtype=torch.float32).log().requires_grad_(True)
        y = torch.tensor([[0]])
        for beta in (1.25, 2.5):
            a.grad = b.grad = None
            loss_goc, _, kl_goc, stats = goc_loss(a, b, y, mode='goc', beta=beta)
            loss_goc.backward()
            g_goc = (a.grad.clone(), b.grad.clone())
            a.grad = b.grad = None
            loss_sym, _, kl_sym, _ = goc_loss(a, b, y, mode='sym_sg', beta=beta)
            loss_sym.backward()
            g_sym = (a.grad.clone(), b.grad.clone())
            a.grad = b.grad = None
            loss_inv, _, kl_inv, _ = goc_loss(a, b, y, mode='inv', beta=beta)
            self.assertEqual(stats['tie_fraction'], 1.0)
            torch.testing.assert_close(loss_goc, loss_sym)
            torch.testing.assert_close(loss_inv, loss_sym)
            torch.testing.assert_close(kl_goc, kl_sym)
            torch.testing.assert_close(kl_inv, kl_sym)
            for x, z in zip(g_goc, g_sym):
                torch.testing.assert_close(x, z)
        # View permutation equivariance.
        torch.manual_seed(77)
        x = torch.randn(2, 3, 11, requires_grad=True)
        z = torch.randn(2, 3, 11, requires_grad=True)
        labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
        for mode in ('goc', 'sym_sg', 'inv', 'rdrop'):
            x.grad = z.grad = None
            l_ab, _, k_ab, _ = goc_loss(x, z, labels, mode=mode, beta=2.5)
            l_ab.backward()
            gab = (x.grad.clone(), z.grad.clone())
            x.grad = z.grad = None
            l_ba, _, k_ba, _ = goc_loss(z, x, labels, mode=mode, beta=2.5)
            l_ba.backward()
            gba = (x.grad.clone(), z.grad.clone())
            torch.testing.assert_close(l_ab, l_ba)
            torch.testing.assert_close(k_ab, k_ba)
            torch.testing.assert_close(gab[0], gba[0])
            torch.testing.assert_close(gab[1], gba[1])
            _, _, k_same, stats_same = goc_loss(x, x, labels, mode=mode, beta=2.5)
            self.assertEqual(float(k_same.item()), 0.)
            self.assertEqual(stats_same['tie_fraction'], 1.0)

    def test_teacher_stop_gradient_property_on_winner_and_loser(self):
        # View 1 has higher gold probability (0.7 > 0.2) at token 0.
        a = torch.tensor([[[.7, .2, .1]]], dtype=torch.float32).log().requires_grad_(True)
        b = torch.tensor([[[.2, .5, .3]]], dtype=torch.float32).log().requires_grad_(True)
        y = torch.tensor([[0]])
        _, _, kl_goc, stats = goc_loss(a, b, y, mode='goc', beta=2.5)
        self.assertEqual(stats['winner_view1_fraction'], 1.0)
        (2.5 * kl_goc).backward()
        ga, gb = a.grad.clone(), b.grad.clone()
        # Teacher view 1 gets zero auxiliary logit gradient; student view 2 gold logit is pushed up (negative grad).
        self.assertEqual(float(ga.abs().max()), 0.)
        self.assertLess(float(gb[0, 0, 0]), 0.)
        # Inverted mode reverses teacher and student.
        a.grad = b.grad = None
        _, _, kl_inv, _ = goc_loss(a, b, y, mode='inv', beta=2.5)
        (2.5 * kl_inv).backward()
        ga_inv, gb_inv = a.grad.clone(), b.grad.clone()
        self.assertEqual(float(gb_inv.abs().max()), 0.)
        self.assertGreater(float(ga_inv[0, 0, 0]), 0.)

    def test_checkpoint_replay_dropout_accumulation(self):
        for method in METHODS:
            for beta in (1.25, 2.5):
                with self.subTest(method=method, beta=beta):
                    torch.manual_seed(142)
                    base = native(.15)
                    plain = GoldOrderedConsistencyMT(copy.deepcopy(base), method=method, beta=beta).train()
                    checked = GoldOrderedConsistencyMT(copy.deepcopy(base), method=method, beta=beta).train()
                    checked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
                    for model in (plain, checked):
                        torch.manual_seed(177)
                        for _ in range(2):
                            (model(**batch()).loss / 2).backward()
                        self.assertGreater(model.last_diagnostics['kl'], 0.)
                        self.assertFalse(model._checkpoint_context)
                        self.assertTrue(all(not c for c in model._cache.values()))
                    count = 0
                    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
                        self.assertEqual(p.grad is None, q.grad is None, name)
                        if p.grad is not None:
                            torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=3e-6, msg=name)
                            self.assertTrue(torch.isfinite(p.grad).all())
                            count += 1
                    self.assertGreater(count, 40)

    def test_native_inference_persistence_and_recipe_validation(self):
        core = native().eval()
        data = batch()
        inputs = {k: v for k, v in data.items() if k != 'labels'}
        for method in METHODS:
            model = GoldOrderedConsistencyMT(copy.deepcopy(core), method=method, beta=2.5).eval()
            self.assertEqual(model.extra_parameter_count(), 0)
            torch.testing.assert_close(model(**data).logits, core(**data).logits, rtol=0, atol=0)
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                restored = GoldOrderedConsistencyMT(
                    MBartForConditionalGeneration.from_pretrained(folder), method=method, beta=2.5).eval()
                restored.verify_checkpoint_metadata(folder)
                torch.testing.assert_close(restored(**data).logits, core(**data).logits, rtol=0, atol=0)
                with torch.no_grad():
                    self.assertTrue(torch.equal(
                        model.generate(**inputs, max_new_tokens=4, use_cache=True),
                        restored.generate(**inputs, max_new_tokens=4, use_cache=False)))
                with self.assertRaises(ValueError):
                    GoldOrderedConsistencyMT(native(), method=method, beta=1.25).verify_checkpoint_metadata(folder)
        with self.assertRaises(ValueError):
            goc_loss(torch.randn(1, 2, 5), torch.randn(1, 2, 5), torch.tensor([[-100, -100]]))
        with self.assertRaises(ValueError):
            GoldOrderedConsistencyMT(native(), method='unknown-v1', beta=1.25)


if __name__ == '__main__':
    unittest.main()
