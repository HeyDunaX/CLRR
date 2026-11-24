"""Scientific and engineering unit tests for Off-EOS Stopping (off_eos_stopping/v1)."""
import gc
import json
import math
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

from methods.off_eos_stopping.v1 import (
    METHODS,
    OffEosStoppingMT,
    detect_repetition_horizon,
    extract_eos_margins,
    huber_one_sided,
    off_eos_stop_loss,
    select_chrf_stop_set,
)
from methods.off_eos_stopping.v1.validation import validate_recipe


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


class _DigitPairTokenizer:
    """Minimal deterministic tokenizer guaranteeing active chrF overlap in small tests."""
    all_special_ids = [0, 1, 2, 4, 5]

    @staticmethod
    def decode(ids, skip_special_tokens=True):
        skip = {0, 1, 2, 4, 5} if skip_special_tokens else set()
        # Map every lexical ID to a shared digit root + id mod 10 so chrF > 0 cleanly
        return ' '.join(f'w{int(x) % 10}' for x in ids if int(x) not in skip)


class OffEosStoppingGate(unittest.TestCase):
    def test_fp64_exact_stop_loss_gradients_and_zero_suffix(self):
        margin = torch.tensor([-0.7, 0.35, -0.12, 0.42, -0.09, 1.3], dtype=torch.float64, requires_grad=True)
        feasible = [2, 3]
        delta = 0.1
        tau = 0.25

        loss = off_eos_stop_loss(margin, feasible, delta=delta, tau=tau)
        loss.backward()

        costs = []
        derivs = []
        m_det = margin.detach()
        for k in feasible:
            a = F.relu(m_det[:k] + delta)
            b = F.relu(delta - m_det[k])
            costs.append(huber_one_sided(m_det[:k] + delta).sum() + huber_one_sided(delta - m_det[k]))
            d = torch.zeros_like(m_det)
            d[:k] = a.clamp(max=1.0)
            d[k] = -b.clamp(max=1.0)
            derivs.append(d)

        c = torch.stack(costs)
        weights = (-c / tau).softmax(dim=0)
        expected_grad = weights @ torch.stack(derivs)
        self.assertLess(float((margin.grad - expected_grad).abs().max()), 1e-12)
        # Suffix tokens beyond max(feasible) must receive identically zero gradient.
        self.assertEqual(float(margin.grad[4:].abs().max()), 0.0)

        # Verify singleton length-only equivalence: soft-min over [k] == C_k exactly
        m_single = m_det.clone().requires_grad_(True)
        loss_single = off_eos_stop_loss(m_single, [3], delta=delta, tau=tau)
        self.assertLess(float((loss_single.detach() - costs[1]).abs()), 1e-12)

    def test_extract_eos_margins_zero_special_and_non_winner_grads(self):
        torch.manual_seed(3)
        logits = torch.tensor([
            [[0.5, -2.0, 1.1, 0.4, 2.5, 0.2, 1.8, -0.3],
             [0.1,  3.0, -0.5, 1.2, 4.0, -1.0, 0.7, 2.2]],
        ], dtype=torch.float64, requires_grad=True)
        # Special tokens: pad=1, eos=2, bos=0, src=4, tgt=5; lexical winners should be col 6 (step 0) and col 7 (step 1).
        margins = extract_eos_margins(logits, eos_token_id=2, non_lexical_ids=(0, 1, 2, 4, 5))
        expected_m = torch.tensor([[1.1 - 1.8, -0.5 - 2.2]], dtype=torch.float64)
        self.assertLess(float((margins.detach() - expected_m).abs().max()), 1e-12)

        margins.sum().backward()
        # Verify grads are +1 on eos(2), -1 on argmax lexical (6 at step 0, 7 at step 1), and 0 everywhere else
        self.assertEqual(float(logits.grad[0, 0, 2]), 1.0)
        self.assertEqual(float(logits.grad[0, 0, 6]), -1.0)
        self.assertEqual(float(logits.grad[0, 1, 2]), 1.0)
        self.assertEqual(float(logits.grad[0, 1, 7]), -1.0)
        for col in (0, 1, 3, 4, 5):
            self.assertEqual(float(logits.grad[0, :, col].abs().max()), 0.0)

    def test_rollout_vs_pass2_logits_exact_alignment(self):
        for builder, method in (
            (lambda: tiny_mbart(dropout=0.0, seed=13), 'off-eos-chrf-v1'),
            (lambda: tiny_m2m100(dropout=0.0, seed=17), 'off-eos-length-v1'),
        ):
            base = builder()
            base.eval()
            model = OffEosStoppingMT(
                base, method=method, extra_past_eos=4, max_rollout_cap=16, max_target_length=32
            )
            model.eval()
            b = tiny_batch(batch_size=2, src_len=8, tgt_len=6, seed=21)
            _, dec_start_id, forced_bos_id, non_lex_ids = model._resolve_special_ids()
            non_lex_set = set(non_lex_ids)
            _, gold_lens = model._extract_gold_body(b['labels'], non_lex_set)

            # Run deterministic KV-cached rollout via wrapper helper
            traj_tokens, _, _ = model._greedy_rollout_past_eos(b['input_ids'], b['attention_mask'], gold_lens)
            prefix = [dec_start_id, forced_bos_id] if forced_bos_id is not None else [dec_start_id]

            # Build uncached Pass 2 decoder inputs and verify step-by-step logit alignment with KV cache
            pad_id = int(base.config.pad_token_id)
            vocab = int(base.config.vocab_size)
            non_lex_mask = torch.zeros(vocab, dtype=torch.bool)
            for tid in non_lex_ids:
                non_lex_mask[tid] = True

            for row_idx in range(2):
                seq = prefix + traj_tokens[row_idx]
                dec_full = torch.tensor([seq], dtype=torch.long)
                inp = b['input_ids'][row_idx:row_idx + 1]
                att = b['attention_mask'][row_idx:row_idx + 1]
                with torch.no_grad():
                    z_pass2 = model._forward_logits(inp, att, dec_full)[:, len(prefix) - 1:, :]
                    enc = base.get_encoder()(input_ids=inp, attention_mask=att, return_dict=True)
                    step_in = torch.tensor([prefix], dtype=torch.long)
                    pkv = None
                    step_logits_list = []
                    for tok in traj_tokens[row_idx]:
                        out = base(
                            encoder_outputs=enc, attention_mask=att,
                            decoder_input_ids=step_in, past_key_values=pkv,
                            use_cache=True, return_dict=True,
                        )
                        pkv = out.past_key_values
                        lg = out.logits[:, -1:, :]
                        step_logits_list.append(lg)
                        self.assertEqual(int(lg[0, 0].masked_fill(non_lex_mask, float('-inf')).argmax().item()), tok)
                        step_in = torch.tensor([[tok]], dtype=torch.long)
                    z_kv = torch.cat(step_logits_list, dim=1)
                    diff = float((z_pass2[:, :len(traj_tokens[row_idx]), :] - z_kv).abs().max().item())
                    self.assertLess(diff, 1e-4)

    def test_detect_repetition_horizon_and_chrf_stop_set(self):
        # 1-gram triple post-EOS triggers cutoff cleanly before 3rd repeat
        ln, hit = detect_repetition_horizon([10, 11, 12, 13, 13, 13, 14], native_eos_step=3)
        self.assertTrue(hit)
        self.assertEqual(ln, 5)

        # 2-gram cycle post-EOS triggers cutoff cleanly
        ln2, hit2 = detect_repetition_horizon([10, 11, 12, 13, 14, 13, 14], native_eos_step=3)
        self.assertTrue(hit2)
        self.assertEqual(ln2, 6)

        # Global 4-repeat before EOS triggers safety cut while respecting min_keep=1
        ln3, hit3 = detect_repetition_horizon([9, 9, 9, 9], native_eos_step=None, min_keep=1)
        self.assertTrue(hit3)
        self.assertEqual(ln3, 3)

        # Clean trajectory remains untouched
        ln4, hit4 = detect_repetition_horizon([10, 11, 12, 13, 14], native_eos_step=3)
        self.assertFalse(hit4)
        self.assertEqual(ln4, 5)

        # chrF stop set selects near-peak band and safely skips 0-score trajectories
        tok_map = {10: 'wo ', 11: 'ai ', 12: 'ni', 13: ' zzz'}
        dec = lambda ids: ''.join(tok_map[i] for i in ids)
        fset, best_sc, best_k = select_chrf_stop_set([10, 11, 12, 13], 'wo ai ni', dec, chrf_band=0.5)
        self.assertGreater(best_sc, 90.0)
        self.assertEqual(best_k, 3)
        self.assertEqual(fset, [3])

        fempty, zsc, zbest = select_chrf_stop_set([13, 13], 'wo ai ni', dec, chrf_band=0.5)
        self.assertEqual(fempty, [])
        self.assertEqual(zsc, 0.0)
        self.assertIsNone(zbest)

    def test_reentrant_checkpointing_eval_save_reload_and_warmup_both_modes(self):
        b = tiny_batch(batch_size=4, src_len=9, tgt_len=8, seed=29)
        for method in METHODS:
            base_a = tiny_mbart(dropout=0.15, seed=13)
            base_b = tiny_mbart(dropout=0.15, seed=13)
            base_b.load_state_dict(base_a.state_dict())

            tok = _DigitPairTokenizer()
            m_no = OffEosStoppingMT(
                base_a, method=method, stop_weight=0.10, extra_past_eos=4,
                max_rollout_cap=16, stop_warmup_fraction=0.10, max_target_length=32, tokenizer=tok,
            )
            m_ck = OffEosStoppingMT(
                base_b, method=method, stop_weight=0.10, extra_past_eos=4,
                max_rollout_cap=16, stop_warmup_fraction=0.10, max_target_length=32, tokenizer=tok,
            )
            m_ck.enable_input_require_grads()
            m_ck.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            m_no.train()
            m_ck.train()
            m_no.set_training_progress(2, 100)
            m_ck.set_training_progress(2, 100)
            self.assertAlmostEqual(m_no.effective_weight(), 0.03, places=6)
            m_no.set_training_progress(50, 100)
            m_ck.set_training_progress(50, 100)
            self.assertAlmostEqual(m_no.effective_weight(), 0.10, places=6)

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

            # Eval forward + beam generate must match unwrapped base model identically
            m_no.eval()
            with torch.no_grad():
                expected = m_no(**b).logits
                native = base_a(**b).logits
                torch.testing.assert_close(expected, native, rtol=0.0, atol=0.0)
                gen_c = m_no.generate(
                    input_ids=b['input_ids'][:2], attention_mask=b['attention_mask'][:2],
                    num_beams=2, max_new_tokens=5, use_cache=True,
                )
                gen_u = m_no.generate(
                    input_ids=b['input_ids'][:2], attention_mask=b['attention_mask'][:2],
                    num_beams=2, max_new_tokens=5, use_cache=False,
                )
                self.assertTrue(torch.equal(gen_c, gen_u))

            with tempfile.TemporaryDirectory() as tmp:
                m_no.save_pretrained(tmp, state_dict=m_no.state_dict())
                reloaded_base = MBartForConditionalGeneration.from_pretrained(Path(tmp))
                reloaded = OffEosStoppingMT(
                    reloaded_base, method=method, stop_weight=0.10, extra_past_eos=4,
                    max_rollout_cap=16, stop_warmup_fraction=0.10, max_target_length=32, tokenizer=tok,
                )
                reloaded.verify_checkpoint_metadata(tmp)
                reloaded.eval()
                with torch.no_grad():
                    torch.testing.assert_close(reloaded(**b).logits, expected)
                cfg_path = Path(tmp) / 'off_eos_stopping_config.json'
                bad = json.loads(cfg_path.read_text(encoding='utf-8'))
                bad['stop_tau'] = 0.99
                cfg_path.write_text(json.dumps(bad), encoding='utf-8')
                with self.assertRaises(ValueError):
                    reloaded.verify_checkpoint_metadata(tmp)
                del reloaded, reloaded_base
                gc.collect()

    def test_m2m100_nllb_compatibility(self):
        b = tiny_batch(batch_size=3, src_len=8, tgt_len=7, seed=19)
        for method in METHODS:
            base = tiny_m2m100(dropout=0.1, seed=3)
            model = OffEosStoppingMT(
                base, method=method, stop_weight=0.10, extra_past_eos=4,
                max_rollout_cap=16, max_target_length=32, tokenizer=_DigitPairTokenizer(),
            )
            model.train()
            out = model(**b)
            self.assertTrue(torch.isfinite(out.loss))
            out.loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(**b).logits, base(**b).logits, rtol=0.0, atol=0.0)

    def test_deterministic_resume_via_torch_rng(self):
        b = tiny_batch(batch_size=4, src_len=9, tgt_len=8, seed=31)
        model = OffEosStoppingMT(
            tiny_mbart(dropout=0.15, seed=5),
            method='off-eos-chrf-v1',
            stop_weight=0.10,
            extra_past_eos=4,
            max_rollout_cap=16,
            max_target_length=32,
            tokenizer=_DigitPairTokenizer(),
        )
        model.train()
        model.set_training_progress(1, 20)
        torch.manual_seed(4242)
        state = torch.get_rng_state()
        l1 = model(**b).loss.detach()
        d1 = dict(model.last_diagnostics)

        torch.set_rng_state(state)
        l2 = model(**b).loss.detach()
        d2 = dict(model.last_diagnostics)
        torch.testing.assert_close(l1, l2, rtol=0.0, atol=0.0)
        self.assertEqual(d1, d2)

    def test_recipe_validation_guards(self):
        cfg_path = ROOT / 'configs/experiments/amis_off_eos_pilot_a100.json'
        self.assertTrue(cfg_path.is_file())
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        for run in cfg['runs']:
            validate_recipe(cfg, run, ROOT)

        chrf_run = next(r for r in cfg['runs'] if r['method'] == 'off-eos-chrf-v1')
        for field, bad_val in (('stop_weight', 0.0), ('stop_margin', -0.1), ('strength', 0.1), ('extra_past_eos', 0)):
            bad_run = dict(chrf_run)
            bad_run[field] = bad_val
            with self.assertRaises(ValueError):
                validate_recipe(cfg, bad_run, ROOT)

        bad_cfg = json.loads(json.dumps(cfg))
        bad_cfg['post_training_test'] = {'authorized': True}
        with self.assertRaises(ValueError):
            validate_recipe(bad_cfg, chrf_run, ROOT)

        bad_cfg2 = json.loads(json.dumps(cfg))
        bad_cfg2['design_review']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            validate_recipe(bad_cfg2, chrf_run, ROOT)


if __name__ == '__main__':
    unittest.main()
