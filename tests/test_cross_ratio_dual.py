"""Scientific and engineering unit tests for CR-Dual (cross_ratio_dual/v1)."""
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / 'src'), str(ROOT / 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from transformers import MBartConfig, MBartForConditionalGeneration

from methods.cross_ratio_dual.v1 import (
    CrossRatioDualMT,
    METHODS,
    build_length_matched_involution,
    chunked_seq_logprob_and_ce,
    cross_ratio_huber_loss,
    unidirectional_contrast_loss,
)
from methods.cross_ratio_dual.v1.validation import validate_recipe


def tiny_mbart(dropout=0.15, seed=7):
    torch.manual_seed(seed)
    cfg = MBartConfig(
        vocab_size=64,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=32,
        dropout=dropout,
        attention_dropout=dropout,
        activation_dropout=dropout,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        forced_bos_token_id=5,
    )
    return MBartForConditionalGeneration(cfg)


def tiny_batch(batch_size=4, src_len=9, tgt_len=8, seed=11):
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(6, 60, (batch_size, src_len), generator=gen)
    input_ids[:, 0] = 4  # src_lang
    attention_mask = torch.ones_like(input_ids)
    # Give distinct lengths and trailing pads
    for i in range(batch_size):
        valid_s = max(4, src_len - (i % 3))
        input_ids[i, valid_s - 1] = 2  # eos
        if valid_s < src_len:
            input_ids[i, valid_s:] = 1  # pad
            attention_mask[i, valid_s:] = 0

    labels = torch.randint(6, 60, (batch_size, tgt_len), generator=gen)
    labels[:, 0] = 5  # tgt_lang
    for i in range(batch_size):
        valid_t = max(4, tgt_len - ((i + 1) % 3))
        labels[i, valid_t - 1] = 2  # eos
        if valid_t < tgt_len:
            labels[i, valid_t:] = -100
    return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class CrossRatioDualGate(unittest.TestCase):
    def test_exact_arnold_press_invariance_fp64(self):
        torch.manual_seed(42)
        n = 8
        energy = torch.randn(n, n, dtype=torch.float64) + 2.5 * torch.eye(n, dtype=torch.float64)
        energy = energy + torch.randn(1, n, dtype=torch.float64) * 2.0  # strong target prior
        p_xy = F.softmax(energy.reshape(-1), dim=0).reshape(n, n)
        log_py_x = torch.log(p_xy / p_xy.sum(dim=1, keepdim=True))
        log_px_y = torch.log(p_xy / p_xy.sum(dim=0, keepdim=True))

        idx_i = torch.tensor([0, 2, 4, 6])
        idx_j = torch.tensor([1, 3, 5, 7])
        pi = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])

        lp_f_diag = log_py_x[torch.arange(n), torch.arange(n)]
        lp_f_off = log_py_x[pi, torch.arange(n)]
        lp_r_diag = log_px_y[torch.arange(n), torch.arange(n)]
        lp_r_off = log_px_y[torch.arange(n), pi]
        len_x = torch.tensor([25., 29., 18., 22., 35., 31., 14., 16.], dtype=torch.float64)
        len_y = torch.tensor([9., 11., 10., 12., 15., 13., 7., 8.], dtype=torch.float64)

        loss, stats = cross_ratio_huber_loss(
            lp_f_diag, lp_f_off, lp_r_diag, lp_r_off, len_x, len_y, idx_i, idx_j, delta=1.0
        )
        self.assertLess(float(loss), 1e-24)
        self.assertLess(stats['abs_cr_violation'], 1e-13)

    def test_length_matched_involution_even_odd_and_singleton(self):
        # Even batch
        src_l = torch.tensor([20, 5, 19, 6, 12, 11])
        tgt_l = torch.tensor([10, 4, 9, 5, 8, 7])
        idx_i, idx_j, pi = build_length_matched_involution(src_l, tgt_l)
        self.assertEqual(int(idx_i.numel()), 3)
        self.assertTrue(torch.equal(pi[pi], torch.arange(6)))
        self.assertTrue((pi != torch.arange(6)).all())

        # Odd batch: 1 trailing element excluded cleanly
        idx_i5, idx_j5, pi5 = build_length_matched_involution(src_l[:5], tgt_l[:5])
        self.assertEqual(int(idx_i5.numel()), 2)
        self.assertTrue(torch.equal(pi5[pi5], torch.arange(5)))
        paired = torch.cat([idx_i5, idx_j5])
        self.assertTrue((pi5[paired] != paired).all())

        # Singleton batch: gracefully 0 pairs
        idx_i1, idx_j1, pi1 = build_length_matched_involution(src_l[:1], tgt_l[:1])
        self.assertEqual(int(idx_i1.numel()), 0)
        self.assertEqual(int(pi1[0]), 0)

    def test_chunked_logprob_exactness_and_zero_pad_grad(self):
        torch.manual_seed(3)
        logits = torch.randn(4, 7, 32, requires_grad=True)
        labels = torch.randint(0, 32, (4, 7))
        labels[0, 5:] = -100
        labels[2, 4:] = -100

        seq_lp, seq_len, mean_ce = chunked_seq_logprob_and_ce(logits, labels, chunk_size=5)

        # Reference unchunked FP32
        ref_lp = F.log_softmax(logits.detach().float(), dim=-1)
        mask = labels.ne(-100)
        ref_tok = ref_lp.gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1) * mask
        torch.testing.assert_close(seq_lp, ref_tok.sum(dim=1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(mean_ce, -ref_tok.sum() / mask.sum(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(seq_len, mask.sum(dim=1).float())

        # Perturbing masked logits has zero effect & zero gradient
        logits_pert = logits.detach().clone()
        logits_pert[0, 5:] += 500.0
        logits_pert[2, 4:] -= 500.0
        seq_lp2, _, mean_ce2 = chunked_seq_logprob_and_ce(logits_pert, labels, chunk_size=5)
        torch.testing.assert_close(seq_lp, seq_lp2, rtol=0.0, atol=0.0)
        torch.testing.assert_close(mean_ce, mean_ce2, rtol=0.0, atol=0.0)

        (seq_lp.sum() + mean_ce).backward()
        self.assertEqual(float(logits.grad[0, 5:].abs().max()), 0.0)
        self.assertEqual(float(logits.grad[2, 4:].abs().max()), 0.0)

    def test_reverse_batch_construction_exact(self):
        model = CrossRatioDualMT(tiny_mbart(0.0), method='cross-ratio-dual-v1')
        b = tiny_batch(batch_size=4)
        rev_ids, rev_mask, rev_labels = model._build_reverse_batch(
            b['input_ids'], b['attention_mask'], b['labels']
        )
        pad_id = model.config.pad_token_id
        self.assertTrue(torch.equal(rev_mask, b['labels'].ne(-100).long()))
        self.assertTrue(torch.equal(rev_ids[rev_mask == 1], b['labels'][b['labels'] != -100]))
        self.assertTrue((rev_ids[rev_mask == 0] == pad_id).all())
        self.assertTrue(torch.equal(rev_labels[b['attention_mask'] == 1], b['input_ids'][b['attention_mask'] == 1]))
        self.assertTrue((rev_labels[b['attention_mask'] == 0] == -100).all())

    def test_paired_rng_identical_subnetwork_under_active_dropout(self):
        # Freeze encoder (no per-row encoder dropout difference) while keeping active 0.25
        # decoder dropout: Paired-RNG MUST yield identical diagonal & off-diagonal logits (M == 0)!
        model = CrossRatioDualMT(tiny_mbart(dropout=0.25, seed=5), method='cross-ratio-dual-v1', paired_rng=True)
        model.train()
        model.get_encoder().eval()
        b = tiny_batch(batch_size=4)
        b['input_ids'] = b['input_ids'][:1].repeat(4, 1)
        b['attention_mask'] = b['attention_mask'][:1].repeat(4, 1)
        b['labels'] = b['labels'][:1].repeat(4, 1)
        out = model(**b)
        self.assertTrue(torch.isfinite(out.loss))
        self.assertAlmostEqual(model.last_diagnostics['mean_m_fwd'], 0.0, places=6)
        self.assertAlmostEqual(model.last_diagnostics['mean_m_rev'], 0.0, places=6)
        self.assertAlmostEqual(model.last_diagnostics['aux_loss'], 0.0, places=6)

    def test_gradient_checkpointing_exact_parity_with_paired_rng(self):
        # Verifies NotebookLM Round B hazard: reentrant checkpointing + Paired-RNG must NOT corrupt gradients!
        b = tiny_batch(batch_size=4, seed=21)
        base_a = tiny_mbart(dropout=0.15, seed=13)
        base_b = tiny_mbart(dropout=0.15, seed=13)
        base_b.load_state_dict(base_a.state_dict())

        m_nockpt = CrossRatioDualMT(base_a, method='cross-ratio-dual-v1', cr_lambda=0.25, paired_rng=True)
        m_ckpt = CrossRatioDualMT(base_b, method='cross-ratio-dual-v1', cr_lambda=0.25, paired_rng=True)
        m_ckpt.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

        m_nockpt.train()
        m_ckpt.train()
        torch.manual_seed(999)
        loss_a = m_nockpt(**b).loss
        loss_a.backward()

        torch.manual_seed(999)
        loss_b = m_ckpt(**b).loss
        loss_b.backward()

        torch.testing.assert_close(loss_b, loss_a, rtol=1e-5, atol=1e-6)
        for (na, pa), (nb, pb) in zip(m_nockpt.named_parameters(), m_ckpt.named_parameters()):
            self.assertEqual(na, nb)
            self.assertIsNotNone(pa.grad, na)
            self.assertIsNotNone(pb.grad, nb)
            torch.testing.assert_close(pb.grad, pa.grad, rtol=1e-4, atol=1e-5)

    def test_shuffled_control_differs_from_exact_cr_dual(self):
        # Direct functional verification on distinct pair cross-ratios:
        # Pair 0 has exact match M_fwd=10, M_rev=10; Pair 1 has exact match M_fwd=30, M_rev=30.
        # Exact CR-Dual loss == 0.0, whereas Shuffled-Reverse-Pair matches (10 vs 30) and (30 vs 10) >> 0!
        idx_i = torch.tensor([0, 2])
        idx_j = torch.tensor([1, 3])
        lp_f_diag = torch.tensor([-5.0, -5.0, -5.0, -5.0])
        lp_f_off = torch.tensor([-10.0, -10.0, -20.0, -20.0])  # M_fwd = [10, 30]
        lp_r_diag = torch.tensor([-5.0, -5.0, -5.0, -5.0])
        lp_r_off = torch.tensor([-10.0, -10.0, -20.0, -20.0])  # M_rev = [10, 30]
        len_x = torch.tensor([10.0, 10.0, 10.0, 10.0])
        len_y = torch.tensor([10.0, 10.0, 10.0, 10.0])
        loss_exact, _ = cross_ratio_huber_loss(
            lp_f_diag, lp_f_off, lp_r_diag, lp_r_off, len_x, len_y, idx_i, idx_j, delta=1.0, shuffle_reverse=False
        )
        loss_shuf, _ = cross_ratio_huber_loss(
            lp_f_diag, lp_f_off, lp_r_diag, lp_r_off, len_x, len_y, idx_i, idx_j, delta=1.0, shuffle_reverse=True
        )
        self.assertAlmostEqual(float(loss_exact), 0.0, places=6)
        self.assertGreater(float(loss_shuf), 0.4)

    def test_warmup_schedule_and_gpu_gate_activation(self):
        model = CrossRatioDualMT(tiny_mbart(0.0), method='cross-ratio-dual-v1', cr_lambda=0.25, cr_warmup_fraction=0.2)
        self.assertAlmostEqual(model.current_lambda(), 0.25)
        model.set_training_progress(0, 720)
        self.assertAlmostEqual(model.current_lambda(), 0.0)
        model.set_training_progress(72, 720)
        self.assertAlmostEqual(model.current_lambda(), 0.125)
        model.set_training_progress(144, 720)
        self.assertAlmostEqual(model.current_lambda(), 0.25)
        model.set_training_progress(500, 720)
        self.assertAlmostEqual(model.current_lambda(), 0.25)
        # 2-step GPU gate activates full lambda on step 1
        model.set_training_progress(1, 2)
        self.assertAlmostEqual(model.current_lambda(), 0.25)

    def test_all_four_modes_train_eval_save_reload_and_zero_extra_params(self):
        b = tiny_batch(batch_size=4, seed=41)
        for method in METHODS:
            base = tiny_mbart(dropout=0.1, seed=19)
            n_base = sum(p.numel() for p in base.parameters())
            kwargs = dict(alpha_rev=0.0, cr_lambda=0.25) if method == 'unidirectional-contrast-v1' else (
                dict(alpha_rev=0.5, cr_lambda=0.0) if method == 'bidirectional-ce-v1' else
                dict(alpha_rev=0.5, cr_lambda=0.25)
            )
            model = CrossRatioDualMT(base, method=method, **kwargs)
            self.assertEqual(sum(p.numel() for p in model.parameters()), n_base)
            model.train()
            out = model(**b)
            self.assertTrue(torch.isfinite(out.loss))
            out.loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

            # Eval / native inference parity
            model.eval()
            with torch.no_grad():
                expected = model(**b).logits
                native_out = base(**b).logits
                torch.testing.assert_close(expected, native_out, rtol=0.0, atol=0.0)
                gen_c = model.generate(input_ids=b['input_ids'][:2], attention_mask=b['attention_mask'][:2],
                                       num_beams=2, max_new_tokens=5, use_cache=True)
                gen_u = model.generate(input_ids=b['input_ids'][:2], attention_mask=b['attention_mask'][:2],
                                       num_beams=2, max_new_tokens=5, use_cache=False)
                self.assertTrue(torch.equal(gen_c, gen_u))

            with tempfile.TemporaryDirectory() as tmp:
                model.save_pretrained(tmp, state_dict=model.state_dict())
                reloaded_base = MBartForConditionalGeneration.from_pretrained(Path(tmp))
                reloaded = CrossRatioDualMT(reloaded_base, method=method, **kwargs)
                reloaded.verify_checkpoint_metadata(tmp)
                reloaded.eval()
                with torch.no_grad():
                    torch.testing.assert_close(reloaded(**b).logits, expected)
                # Tampered config must fail verification
                cfg_path = Path(tmp) / 'cross_ratio_dual_config.json'
                bad = json.loads(cfg_path.read_text(encoding='utf-8'))
                bad['cr_delta'] = 99.0
                cfg_path.write_text(json.dumps(bad), encoding='utf-8')
                with self.assertRaises(ValueError):
                    reloaded.verify_checkpoint_metadata(tmp)
                del reloaded, reloaded_base
                gc.collect()

    def test_recipe_validation_guards(self):
        cfg_path = ROOT / 'configs/experiments/amis_cr_dual_pilot_a100.json'
        if not cfg_path.exists():
            return
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        for run in cfg['runs']:
            validate_recipe(cfg, run, ROOT)
        # Shuffled with microbatch < 4 must fail
        bad_run = dict(next(r for r in cfg['runs'] if r['method'] == 'cross-ratio-shuffled-v1'))
        bad_run['microbatch'] = 2
        with self.assertRaises(ValueError):
            validate_recipe(cfg, bad_run, ROOT)


if __name__ == '__main__':
    unittest.main()
