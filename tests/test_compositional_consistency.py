"""Scientific and engineering unit tests for CoCC (compositional_consistency/v1)."""
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
from transformers import (
    M2M100Config,
    M2M100ForConditionalGeneration,
    MBartConfig,
    MBartForConditionalGeneration,
)

from methods.compositional_consistency.v1 import (
    CompositionalConsistencyMT,
    METHODS,
    build_composite_batch_s1,
    chunked_aligned_kl_loss,
    chunked_ce_loss,
)
from methods.compositional_consistency.v1.validation import validate_recipe


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
        max_position_embeddings=64,
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


def tiny_m2m100(dropout=0.1, seed=9):
    torch.manual_seed(seed)
    cfg = M2M100Config(
        vocab_size=64,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=64,
        dropout=dropout,
        attention_dropout=dropout,
        activation_dropout=dropout,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
    )
    return M2M100ForConditionalGeneration(cfg)


def tiny_batch(batch_size=4, src_len=9, tgt_len=8, seed=11):
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(6, 60, (batch_size, src_len), generator=gen)
    input_ids[:, 0] = 4  # src_lang
    attention_mask = torch.ones_like(input_ids)
    for i in range(batch_size):
        valid_s = max(5, src_len - (i % 3))
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


class CompositionalConsistencyGate(unittest.TestCase):
    def test_fp64_exact_aligned_gradients_and_zero_masks(self):
        torch.manual_seed(0)
        vocab = 11
        # Two rows of lengths 5 ([9, 3, 4, 5, 2]) and 4 ([9, 6, 7, 2, -100])
        input_ids = torch.tensor([[8, 10, 4, 2, 1], [8, 5, 6, 2, 1]])
        attn_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
        labels = torch.tensor([[9, 3, 4, 5, 2], [9, 6, 7, 2, -100]])

        # Force deterministic (0, 1) pair
        torch.manual_seed(1)
        comp = build_composite_batch_s1(
            input_ids, attn_mask, labels, pad_token_id=1, decoder_start_token_id=2,
            max_source_length=32, max_target_length=32, mask_y_j1=True, layout='s1',
        )
        self.assertIsNotNone(comp)

        z_iso_base = torch.randn(2, 5, vocab, dtype=torch.float64)
        z_comp_base = torch.randn(1, comp['labels'].shape[1], vocab, dtype=torch.float64)

        for direction in ('sym_sg', 'iso_teacher'):
            zi = z_iso_base.clone().requires_grad_(True)
            zc = z_comp_base.clone().requires_grad_(True)
            kl = chunked_aligned_kl_loss(
                zi, zc, comp['iso_flat_idx'], comp['comp_flat_idx'], direction=direction, chunk_size=2
            )
            kl.backward()

            p_i = F.softmax(zi.detach().reshape(-1, vocab), dim=-1)
            p_c = F.softmax(zc.detach().reshape(-1, vocab), dim=-1)
            exp_gi = torch.zeros(2 * 5, vocab, dtype=torch.float64)
            exp_gc = torch.zeros(comp['labels'].shape[1], vocab, dtype=torch.float64)
            n_tok = int(comp['iso_flat_idx'].numel())
            self.assertEqual(n_tok, 6)  # 4 from seg i + 2 from seg j (excluding y_{j,1})

            w_i = (0.5 / n_tok) if direction == 'sym_sg' else 0.0
            w_c = (0.5 / n_tok) if direction == 'sym_sg' else (1.0 / n_tok)
            for ii, ci in zip(comp['iso_flat_idx'].tolist(), comp['comp_flat_idx'].tolist()):
                exp_gi[ii] += w_i * (p_i[ii] - p_c[ci])
                exp_gc[ci] += w_c * (p_c[ci] - p_i[ii])

            gi = torch.zeros_like(zi) if zi.grad is None else zi.grad
            self.assertLess(float((gi.reshape(-1, vocab) - exp_gi).abs().max()), 1e-12)
            self.assertLess(float((zc.grad.reshape(-1, vocab) - exp_gc).abs().max()), 1e-12)
            # Zero gradient on isolated pad position (row 1, col 4) and internal <tgt_lang>
            self.assertEqual(float(gi[1, 4].abs().max()), 0.0)
            sep_pos = 5 if comp['pairs'][0] == (0, 1) else 4
            self.assertEqual(float(zc.grad[0, sep_pos].abs().max()), 0.0)
            self.assertEqual(float(zc.grad[0, sep_pos + 1].abs().max()), 0.0)  # y_{j,1} masked
            if direction == 'iso_teacher':
                self.assertIsNone(zi.grad)

    def test_s1_layout_decoder_input_ids_and_token_parity_mbart(self):
        base = tiny_mbart(0.0)
        b = tiny_batch(batch_size=4, src_len=9, tgt_len=8, seed=17)
        torch.manual_seed(5)
        comp = build_composite_batch_s1(
            b['input_ids'], b['attention_mask'], b['labels'],
            pad_token_id=1, decoder_start_token_id=2,
            max_source_length=32, max_target_length=32, mask_y_j1=True, layout='s1',
        )
        self.assertIsNotNone(comp)
        self.assertEqual(len(comp['pairs']), 2)

        # Verify bit-exact parity with MBart.prepare_decoder_input_ids_from_labels BEFORE masking internal tgt_lang
        raw_labels = comp['labels'].clone()
        for r, (i, j) in enumerate(comp['pairs']):
            sy_i = int(b['labels'][i].ne(-100).sum().item())
            sy_j = int(b['labels'][j].ne(-100).sum().item())
            self.assertEqual(int(comp['labels'][r, sy_i - 1]), 2)  # seg 1 EOS preserved
            self.assertEqual(int(comp['labels'][r, sy_i]), -100)   # internal tgt_lang masked
            self.assertEqual(int(comp['labels'][r, sy_i + sy_j - 1]), 2)  # seg 2 EOS preserved
            # Check prefix seen right before y_{j,1} equals [..., </s>_i, <tgt_lang>]
            self.assertEqual(int(comp['decoder_input_ids'][r, sy_i]), 2)
            self.assertEqual(int(comp['decoder_input_ids'][r, sy_i + 1]), 5)
            raw_labels[r, sy_i] = 5
        expected_dec = base.prepare_decoder_input_ids_from_labels(raw_labels)
        self.assertTrue(torch.equal(comp['decoder_input_ids'], expected_dec))

        # Every aligned token pair points to identical non-(-100) target token IDs
        iso_toks = b['labels'].reshape(-1)[comp['iso_flat_idx']]
        comp_toks = comp['labels'].reshape(-1)[comp['comp_flat_idx']]
        self.assertTrue((iso_toks != -100).all())
        self.assertTrue(torch.equal(iso_toks, comp_toks))

    def test_m2m100_nllb_compatibility(self):
        from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right as m2m_shift

        base = tiny_m2m100(dropout=0.1, seed=3)
        b = tiny_batch(batch_size=4, src_len=8, tgt_len=7, seed=19)
        torch.manual_seed(8)
        comp = build_composite_batch_s1(
            b['input_ids'], b['attention_mask'], b['labels'],
            pad_token_id=1, decoder_start_token_id=2,
            max_source_length=32, max_target_length=32, mask_y_j1=True, layout='s1',
        )
        raw_labels = comp['labels'].clone()
        for r, (i, _) in enumerate(comp['pairs']):
            sy_i = int(b['labels'][i].ne(-100).sum().item())
            raw_labels[r, sy_i] = 5
        expected_dec = m2m_shift(raw_labels, pad_token_id=1, decoder_start_token_id=2)
        self.assertTrue(torch.equal(comp['decoder_input_ids'], expected_dec))

        model = CompositionalConsistencyMT(base, method='cocc-sym-v1', beta=1.25, max_source_length=32, max_target_length=32)
        model.train()
        out = model(**b)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            expected = model(**b).logits
            native = base(**b).logits
            torch.testing.assert_close(expected, native, rtol=0.0, atol=0.0)

    def test_length_guarded_pairing_odd_singleton_and_oversized_fallback(self):
        # Odd batch (B=5 -> 2 pairs)
        b5 = tiny_batch(batch_size=5, src_len=9, tgt_len=8, seed=23)
        comp5 = build_composite_batch_s1(
            b5['input_ids'], b5['attention_mask'], b5['labels'], 1, 2,
            max_source_length=32, max_target_length=32,
        )
        self.assertIsNotNone(comp5)
        self.assertEqual(len(comp5['pairs']), 2)

        # Singleton (B=1) and oversized (max_target_length=4): graceful fallback across all 4 modes
        b1 = tiny_batch(batch_size=1, src_len=9, tgt_len=8, seed=23)
        b2 = tiny_batch(batch_size=2, src_len=9, tgt_len=8, seed=23)
        for method in METHODS:
            beta = 0.0 if method == 'concat-ce-v1' else 1.25
            model = CompositionalConsistencyMT(
                tiny_mbart(0.1), method=method, beta=beta, max_source_length=64, max_target_length=8
            )
            model.train()
            out1 = model(**b1)
            self.assertTrue(torch.isfinite(out1.loss))
            self.assertEqual(model.last_diagnostics['num_pairs'], 0)
            out1.loss.backward()

            model.zero_grad(set_to_none=True)
            out2 = model(**b2)
            self.assertTrue(torch.isfinite(out2.loss))
            self.assertEqual(model.last_diagnostics['num_pairs'], 0)
            out2.loss.backward()

    def test_chunked_ce_loss_exactness_and_zero_pad_grad(self):
        torch.manual_seed(4)
        logits = torch.randn(4, 7, 32, requires_grad=True)
        labels = torch.randint(0, 32, (4, 7))
        labels[0, 5:] = -100
        labels[3, 4:] = -100
        ce = chunked_ce_loss(logits, labels, chunk_size=3)
        ref = F.cross_entropy(logits.detach().float().reshape(-1, 32), labels.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(ce, ref, rtol=1e-5, atol=1e-6)

        logits_pert = logits.detach().clone()
        logits_pert[0, 5:] += 300.0
        ce2 = chunked_ce_loss(logits_pert, labels, chunk_size=3)
        torch.testing.assert_close(ce, ce2, rtol=0.0, atol=0.0)

        ce.backward()
        self.assertEqual(float(logits.grad[0, 5:].abs().max()), 0.0)
        self.assertEqual(float(logits.grad[3, 4:].abs().max()), 0.0)

    def test_reentrant_checkpointing_exact_parity_all_modes(self):
        b = tiny_batch(batch_size=4, src_len=9, tgt_len=8, seed=29)
        for method in METHODS:
            beta = 0.0 if method == 'concat-ce-v1' else 1.25
            base_a = tiny_mbart(dropout=0.15, seed=13)
            base_b = tiny_mbart(dropout=0.15, seed=13)
            base_b.load_state_dict(base_a.state_dict())

            m_no = CompositionalConsistencyMT(
                base_a, method=method, beta=beta, max_source_length=32, max_target_length=32, logit_chunk_tokens=4
            )
            m_ck = CompositionalConsistencyMT(
                base_b, method=method, beta=beta, max_source_length=32, max_target_length=32, logit_chunk_tokens=4
            )
            m_ck.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            m_no.train()
            m_ck.train()

            torch.manual_seed(777)
            la = m_no(**b).loss
            la.backward()

            torch.manual_seed(777)
            lb = m_ck(**b).loss
            lb.backward()

            torch.testing.assert_close(lb, la, rtol=1e-5, atol=1e-6)
            for (na, pa), (nb, pb) in zip(m_no.named_parameters(), m_ck.named_parameters()):
                self.assertEqual(na, nb)
                self.assertIsNotNone(pa.grad, na)
                self.assertIsNotNone(pb.grad, nb)
                torch.testing.assert_close(pb.grad, pa.grad, rtol=1e-4, atol=1e-5)

    def test_deterministic_resume_via_torch_rng(self):
        b = tiny_batch(batch_size=6, src_len=9, tgt_len=8, seed=31)
        model = CompositionalConsistencyMT(
            tiny_mbart(dropout=0.15, seed=5), method='cocc-sym-v1', beta=1.25, max_source_length=32, max_target_length=32
        )
        model.train()
        torch.manual_seed(4242)
        state = torch.get_rng_state()
        l1 = model(**b).loss.detach()
        d1 = dict(model.last_diagnostics)

        torch.set_rng_state(state)
        l2 = model(**b).loss.detach()
        d2 = dict(model.last_diagnostics)
        torch.testing.assert_close(l1, l2, rtol=0.0, atol=0.0)
        self.assertEqual(d1, d2)

    def test_all_modes_eval_native_generate_and_save_reload(self):
        b = tiny_batch(batch_size=4, src_len=9, tgt_len=8, seed=41)
        for method in METHODS:
            beta = 0.0 if method == 'concat-ce-v1' else 1.25
            base = tiny_mbart(dropout=0.1, seed=19)
            n_base = sum(p.numel() for p in base.parameters())
            model = CompositionalConsistencyMT(
                base, method=method, beta=beta, max_source_length=32, max_target_length=32
            )
            self.assertEqual(model.extra_parameter_count(), 0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), n_base)

            model.eval()
            with torch.no_grad():
                expected = model(**b).logits
                native = base(**b).logits
                torch.testing.assert_close(expected, native, rtol=0.0, atol=0.0)
                gen_c = model.generate(
                    input_ids=b['input_ids'][:2], attention_mask=b['attention_mask'][:2],
                    num_beams=2, max_new_tokens=5, use_cache=True,
                )
                gen_u = model.generate(
                    input_ids=b['input_ids'][:2], attention_mask=b['attention_mask'][:2],
                    num_beams=2, max_new_tokens=5, use_cache=False,
                )
                self.assertTrue(torch.equal(gen_c, gen_u))

            with tempfile.TemporaryDirectory() as tmp:
                model.save_pretrained(tmp, state_dict=model.state_dict())
                reloaded_base = MBartForConditionalGeneration.from_pretrained(Path(tmp))
                reloaded = CompositionalConsistencyMT(
                    reloaded_base, method=method, beta=beta, max_source_length=32, max_target_length=32
                )
                reloaded.verify_checkpoint_metadata(tmp)
                reloaded.eval()
                with torch.no_grad():
                    torch.testing.assert_close(reloaded(**b).logits, expected)
                cfg_path = Path(tmp) / 'compositional_consistency_config.json'
                bad = json.loads(cfg_path.read_text(encoding='utf-8'))
                bad['layout'] = 's0'
                cfg_path.write_text(json.dumps(bad), encoding='utf-8')
                with self.assertRaises(ValueError):
                    reloaded.verify_checkpoint_metadata(tmp)
                del reloaded, reloaded_base
                gc.collect()

    def test_recipe_validation_guards(self):
        cfg_path = ROOT / 'configs/experiments/amis_cocc_pilot_a100.json'
        self.assertTrue(cfg_path.is_file())
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        for run in cfg['runs']:
            validate_recipe(cfg, run, ROOT)

        cocc_run = next(r for r in cfg['runs'] if r['method'] == 'cocc-sym-v1')
        for field, bad_val in (('mask_y_j1', False), ('layout', 's0'), ('consistency_beta', 0.0)):
            bad_run = dict(cocc_run)
            bad_run[field] = bad_val
            with self.assertRaises(ValueError):
                validate_recipe(cfg, bad_run, ROOT)

        bad_cfg = json.loads(json.dumps(cfg))
        bad_cfg['design_review']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            validate_recipe(bad_cfg, cocc_run, ROOT)


if __name__ == '__main__':
    unittest.main()
