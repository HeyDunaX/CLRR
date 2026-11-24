"""Scientific and engineering unit tests for 2x2 Lexical Fork Preference (`lexical_fork/v1`)."""
import copy
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
from datasets import Dataset
from transformers import (
    M2M100Config,
    M2M100ForConditionalGeneration,
    MBartConfig,
    MBartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    set_seed,
)

from experiment_runner.engine import ControlTrainer
from methods.lexical_fork.v1 import (
    METHODS,
    LexicalForkCollator,
    LexicalForkEpochCallback,
    LexicalForkMT,
    annotate_window_fork_counts,
    build_epoch_fork_table,
    build_lexical_boundary_masks,
    hash_prefix_index,
    hash_selection_key,
    select_balanced_epoch_forks,
)
from methods.lexical_fork.v1.validation import validate_recipe


class _AlphabeticForkTokenizer:
    """Minimal SentencePiece-like tokenizer exposing BOW (`▁`) and continuation tokens."""

    pad_token_id = 1
    eos_token_id = 2
    bos_token_id = 0
    unk_token_id = 3
    all_special_ids = [0, 1, 2, 3, 4, 5]
    padding_side = 'right'
    model_input_names = ['input_ids', 'attention_mask']
    src_lang = 'amis_XX'
    tgt_lang = 'zh_CN'
    lang_code_to_id = {'amis_XX': 4, 'zh_CN': 5}
    vocab_size = 64

    _TOKENS = ['<s>', '<pad>', '</s>', '<unk>', 'amis_XX', 'zh_CN'] + [
        (f'\u2581w{chr(97 + ((i - 6) // 2) % 26)}' if i % 2 == 0 else f'c{chr(97 + ((i - 7) // 2) % 26)}')
        for i in range(6, 64)
    ]

    def __len__(self):
        return self.vocab_size

    def convert_ids_to_tokens(self, tid):
        if isinstance(tid, list):
            return [self._TOKENS[int(x)] for x in tid]
        return self._TOKENS[int(tid)]

    def _encode_one(self, text, max_length=32):
        words = [w for w in str(text).strip().split() if w] or ['alpha']
        body = [6 + (sum(ord(c) for c in w) % 28) * 2 for w in words]
        ids = [4] + body[: max(1, max_length - 2)] + [2]
        return ids

    def __call__(self, texts, return_tensors=None, padding=False, truncation=True, max_length=32, **kwargs):
        if isinstance(texts, str):
            ids = self._encode_one(texts, max_length=max_length)
            enc = {'input_ids': ids, 'attention_mask': [1] * len(ids)}
            return self.pad([enc], padding=False, return_tensors=return_tensors) if return_tensors else enc
        encoded = [
            {'input_ids': ids, 'attention_mask': [1] * len(ids)}
            for ids in (self._encode_one(t, max_length=max_length) for t in texts)
        ]
        if padding or return_tensors:
            return self.pad(encoded, padding=True, return_tensors=return_tensors)
        return {'input_ids': [x['input_ids'] for x in encoded], 'attention_mask': [x['attention_mask'] for x in encoded]}

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None, **kwargs):
        seqs = [list(x['input_ids']) for x in encoded_inputs]
        masks = [list(x.get('attention_mask', [1] * len(s))) for x, s in zip(encoded_inputs, seqs)]
        target_len = max(len(s) for s in seqs) if padding else len(seqs[0])
        if max_length is not None:
            target_len = max(target_len, int(max_length))
        if pad_to_multiple_of is not None and int(pad_to_multiple_of) > 1:
            m = int(pad_to_multiple_of)
            target_len = ((target_len + m - 1) // m) * m
        out_ids = [s + [self.pad_token_id] * (target_len - len(s)) for s in seqs]
        out_mask = [mk + [0] * (target_len - len(mk)) for mk in masks]
        result = {'input_ids': out_ids, 'attention_mask': out_mask}
        if 'labels' in encoded_inputs[0]:
            lbls = [list(x['labels']) for x in encoded_inputs]
            result['labels'] = [l + [-100] * (target_len - len(l)) for l in lbls]
        if return_tensors == 'pt':
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        return result

    def decode(self, ids, skip_special_tokens=True):
        skip = set(self.all_special_ids) if skip_special_tokens else set()
        out = []
        for x in ids:
            tid = int(x)
            if tid in skip or tid < 0 or tid >= self.vocab_size:
                continue
            piece = self._TOKENS[tid]
            if piece.startswith('\u2581'):
                out.append(' ' + piece[1:])
            else:
                out.append(piece)
        return ''.join(out).strip()

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def save_pretrained(self, save_directory, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)


def tiny_mbart(dropout=0.0, seed=7):
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


def tiny_m2m100(dropout=0.0, seed=9):
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


def tiny_batch(batch_size=4, src_len=9, tgt_len=8, seed=11, row_start=0):
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(6, 60, (batch_size, src_len), generator=gen)
    input_ids[:, 0] = 4
    attention_mask = torch.ones_like(input_ids)
    for i in range(batch_size):
        valid_s = max(5, src_len - (i % 3))
        input_ids[i, valid_s - 1] = 2
        if valid_s < src_len:
            input_ids[i, valid_s:] = 1
            attention_mask[i, valid_s:] = 0

    labels = torch.randint(6, 60, (batch_size, tgt_len), generator=gen)
    labels[:, 0] = 5
    for i in range(batch_size):
        valid_t = max(4, tgt_len - ((i + 1) % 3))
        labels[i, valid_t - 1] = 2
        if valid_t < tgt_len:
            labels[i, valid_t:] = -100
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        row_ids=torch.arange(row_start, row_start + batch_size, dtype=torch.long),
    )


def _sample_selected_map(row_ids=(0, 1, 3, 4, 5, 7)):
    prefixes = [[], [8], [10, 12], [14], [16, 18, 20], [22, 24]]
    out = {}
    for k, rid in enumerate(row_ids):
        h = prefixes[k % len(prefixes)]
        out[int(rid)] = dict(
            j=len(h),
            h=h,
            a=8 + (k * 2) % 20,
            b=10 + (k * 2) % 20,
            sign=1 if (k % 2 == 0) else -1,
            margin=0.25 + 0.1 * k,
            stratum=0,
        )
    return out


class LexicalForkTrainingGate(unittest.TestCase):
    def test_ga4_window_normalization_and_empty_window(self):
        tok = _AlphabeticForkTokenizer()
        # Disable dropout via config=0 so single-batch CE + GA=4 microbatch CE match identically.
        model_ga = LexicalForkMT(
            tiny_mbart(dropout=0.0, seed=17),
            method='lexical-fork-2x2-v1',
            tokenizer=tok,
        )
        model_ref = LexicalForkMT(
            copy.deepcopy(model_ga.base_model),
            method='lexical-fork-2x2-v1',
            tokenizer=tok,
        )
        # Keep identical length/pad structure across 4 microbatches of size 2 each (counts: 2, 0, 1, 3 => K_w=6)
        full = tiny_batch(batch_size=8, src_len=8, tgt_len=7, seed=21, row_start=0)
        selected_rows = (0, 1, 4, 6, 7)
        # Wait: ensure exact counts (2, 0, 1, 3): mb0=[0,1]->2, mb1=[2,3]->0, mb2=[4,5]->1 ([4]), mb3=[6,7]->wait need 3 in mb3!
        # Adjust microbatches to uneven sizes [2, 2, 2, 3] or keep size=3 per microbatch:
        full = tiny_batch(batch_size=12, src_len=8, tgt_len=7, seed=21, row_start=0)
        # mb0=[0,1,2]->2 (0,1), mb1=[3,4,5]->0, mb2=[6,7,8]->1 (7), mb3=[9,10,11]->3 (9,10,11)
        sel_map = _sample_selected_map(row_ids=(0, 1, 7, 9, 10, 11))
        model_ga.set_epoch_fork_state(2, 0.10, sel_map)
        model_ref.set_epoch_fork_state(2, 0.10, sel_map)

        mbs = [
            {k: v[s:e] for k, v in full.items()}
            for s, e in ((0, 3), (3, 6), (6, 9), (9, 12))
        ]
        k_w = annotate_window_fork_counts(mbs, sel_map)
        self.assertEqual(k_w, 6)
        self.assertEqual([b['fork_window_count'] for b in mbs], [6, 6, 6, 6])
        self.assertEqual([b['fork_window_gas'] for b in mbs], [4, 4, 4, 4])

        model_ga.train()
        model_ga.zero_grad()
        seen_mb_counts = []
        for mb in mbs:
            out = model_ga(**mb)
            seen_mb_counts.append(model_ga.last_diagnostics['mb_selected'])
            (out.loss / 4.0).backward()
        self.assertEqual(seen_mb_counts, [2, 0, 1, 3])

        model_ref.train()
        model_ref.zero_grad()
        # Exact reference: mean CE over the 4 microbatch CE losses + (0.10 / 6) * sum(softplus) over all 6 forks
        ce_terms = []
        fork_losses = []
        _, _, prefix = (1, 2, [2, 5])
        for mb in mbs:
            dec_gold = model_ref.prepare_decoder_input_ids_from_labels(mb['labels'])
            z_gold = model_ref.base_model(
                input_ids=mb['input_ids'],
                attention_mask=mb['attention_mask'],
                decoder_input_ids=dec_gold,
                use_cache=False,
                return_dict=True,
            ).logits
            valid = mb['labels'] != -100
            ce_mb = F.cross_entropy(z_gold.view(-1, z_gold.shape[-1]), mb['labels'].view(-1), ignore_index=-100)
            ce_terms.append(ce_mb)
            for idx, rid in enumerate(mb['row_ids'].tolist()):
                if int(rid) not in sel_map:
                    continue
                entry = sel_map[int(rid)]
                dec_f = torch.tensor([prefix + list(entry['h'])], dtype=torch.long)
                z_f = model_ref.base_model(
                    input_ids=mb['input_ids'][idx:idx + 1],
                    attention_mask=mb['attention_mask'][idx:idx + 1],
                    decoder_input_ids=dec_f,
                    use_cache=False,
                    return_dict=True,
                ).logits[0, -1, :].float()
                sp = F.softplus(-float(entry['sign']) * (z_f[entry['b']] - z_f[entry['a']]))
                fork_losses.append(sp)
        ref_loss = torch.stack(ce_terms).mean() + 0.10 * torch.stack(fork_losses).sum() / 6.0
        ref_loss.backward()

        max_diff = max(
            float((p_ga.grad - p_ref.grad).abs().max())
            for p_ga, p_ref in zip(model_ga.parameters(), model_ref.parameters())
            if p_ref.grad is not None
        )
        self.assertLess(max_diff, 1e-5)

        # Empty window (0, 0, 0, 0) must yield zero fork gradient and match CE-only gradient exactly.
        empty_mbs = [
            {k: v[s:e] for k, v in tiny_batch(batch_size=8, src_len=8, tgt_len=7, seed=31, row_start=100).items()}
            for s, e in ((0, 2), (2, 4), (4, 6), (6, 8))
        ]
        self.assertEqual(annotate_window_fork_counts(empty_mbs, sel_map), 0)
        model_ga.zero_grad()
        for mb in empty_mbs:
            out = model_ga(**mb)
            self.assertEqual(model_ga.last_diagnostics['fork_window_scaled'], 0.0)
            (out.loss / 4.0).backward()

        model_ce_only = LexicalForkMT(copy.deepcopy(model_ga.base_model), method='lexical-fork-2x2-v1', tokenizer=tok)
        model_ce_only.set_epoch_fork_state(1, 0.0, {})
        model_ce_only.train()
        model_ce_only.zero_grad()
        annotate_window_fork_counts(empty_mbs, {})
        for mb in empty_mbs:
            (model_ce_only(**mb).loss / 4.0).backward()

        empty_diff = max(
            float((p_a.grad - p_b.grad).abs().max())
            for p_a, p_b in zip(model_ga.parameters(), model_ce_only.parameters())
            if p_b.grad is not None
        )
        self.assertLess(empty_diff, 1e-6)

    def test_5stratum_margin_balanced_selection_and_cutoff(self):
        # 50 rows across margins [0.05..2.50]; heavy natural positives to test 2x2 quota matching
        scored = []
        for i in range(50):
            s_2x2 = 1 if (i % 5 < 2) else (-1 if (i % 5 < 4) else 0)
            s_nat = 1 if (i % 2 == 0) else -1
            scored.append(
                dict(
                    row_id=i,
                    j=1,
                    h=[8],
                    a=10,
                    b=12,
                    margin=0.05 * (i + 1),
                    s_2x2=s_2x2,
                    s_nat=s_nat,
                )
            )
        # First check threshold cutoff: raw_pos_2x2 == 20 < 32 disables active selection.
        sel_off, stats_off = select_balanced_epoch_forks(scored, mode='2x2', seed=42, epoch_idx=2, num_strata=5, min_pos_2x2=32)
        self.assertFalse(stats_off['active'])
        self.assertEqual(stats_off['deactivated_reason'], 'raw_pos_2x2_below_threshold')
        self.assertEqual(sel_off, {})

        # With lowered threshold, both 2x2 and natural must match pos == neg == quota per stratum.
        sel_2x2, st_2x2 = select_balanced_epoch_forks(scored, mode='2x2', seed=42, epoch_idx=2, num_strata=5, min_pos_2x2=10)
        sel_nat, st_nat = select_balanced_epoch_forks(scored, mode='natural', seed=42, epoch_idx=2, num_strata=5, min_pos_2x2=10)
        self.assertTrue(st_2x2['active'])
        self.assertEqual(st_2x2['selected_pos'], st_2x2['selected_neg'])
        self.assertEqual(st_nat['selected_pos'], st_nat['selected_neg'])
        for g_2x2, g_nat in zip(st_2x2['strata'], st_nat['strata']):
            expected_q = min(g_2x2['raw_pos_2x2'], g_2x2['raw_neg_2x2'])
            self.assertEqual(g_2x2['selected_each_sign'], expected_q)
            self.assertLessEqual(g_nat['selected_each_sign'], expected_q)
            s = g_2x2['stratum']
            self.assertEqual(
                sum(1 for v in sel_2x2.values() if v['stratum'] == s and v['sign'] == 1),
                sum(1 for v in sel_2x2.values() if v['stratum'] == s and v['sign'] == -1),
            )
            self.assertEqual(
                sum(1 for v in sel_nat.values() if v['stratum'] == s and v['sign'] == 1),
                sum(1 for v in sel_nat.values() if v['stratum'] == s and v['sign'] == -1),
            )

    def test_blake2b_deterministic_prefix_and_selection_hashing(self):
        elig = [0, 2, 4, 7, 9, 11]
        p1 = hash_prefix_index(42, 2, 123, elig)
        p2 = hash_prefix_index(42, 2, 123, elig)
        self.assertEqual(p1, p2)
        self.assertIn(p1, elig)

        seen_across = {hash_prefix_index(42, ep, r, elig) for ep in range(1, 5) for r in range(25)}
        self.assertGreater(len(seen_across), 1)

        k1 = hash_selection_key(42, 2, '2x2', 1, 77)
        k2 = hash_selection_key(42, 2, '2x2', 1, 77)
        k3 = hash_selection_key(42, 2, 'natural', 1, 77)
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)

    def test_epoch1_pure_ce_exact_parity(self):
        tok = _AlphabeticForkTokenizer()
        m_nat = LexicalForkMT(tiny_mbart(dropout=0.0, seed=5), method='lexical-fork-natural-v1', tokenizer=tok)
        m_2x2 = LexicalForkMT(copy.deepcopy(m_nat.base_model), method='lexical-fork-2x2-v1', tokenizer=tok)
        m_nat.set_epoch_fork_state(1, 0.0, {})
        m_2x2.set_epoch_fork_state(1, 0.0, _sample_selected_map((0, 1)))  # ignored when active_lambda==0
        m_nat.train()
        m_2x2.train()

        batch = tiny_batch(batch_size=4, src_len=8, tgt_len=7, seed=9)
        o_nat = m_nat(**batch)
        o_2x2 = m_2x2(**batch)
        self.assertEqual(float(o_nat.loss.detach()), float(o_2x2.loss.detach()))
        o_nat.loss.backward()
        o_2x2.loss.backward()
        max_diff = max(
            float((p1.grad - p2.grad).abs().max())
            for p1, p2 in zip(m_nat.parameters(), m_2x2.parameters())
            if p1.grad is not None
        )
        self.assertEqual(max_diff, 0.0)

    def test_right_padding_invariance_reentrant_ckpt_and_eval_beam4_parity(self):
        tok = _AlphabeticForkTokenizer()
        base = tiny_mbart(dropout=0.0, seed=13)
        m_batched = LexicalForkMT(base, method='lexical-fork-2x2-v1', tokenizer=tok)
        m_single = LexicalForkMT(copy.deepcopy(base), method='lexical-fork-2x2-v1', tokenizer=tok)
        # Rows 0 and 1 have different prefix lengths h (0 vs 3) -> right-padding inside dec_fork
        sel_map = {
            0: dict(j=0, h=[], a=8, b=10, sign=1, margin=0.3, stratum=0),
            1: dict(j=3, h=[12, 14, 16], a=18, b=20, sign=-1, margin=0.4, stratum=0),
        }
        batch = tiny_batch(batch_size=2, src_len=8, tgt_len=7, seed=19, row_start=0)
        batch['fork_window_count'] = 2
        batch['fork_window_gas'] = 1

        m_batched.set_epoch_fork_state(2, 0.1, sel_map)
        m_batched.train()
        m_batched.zero_grad()
        m_batched(**batch).loss.backward()

        m_single.set_epoch_fork_state(2, 0.1, sel_map)
        m_single.train()
        m_single.zero_grad()
        for idx in (0, 1):
            sub = {
                'input_ids': batch['input_ids'][idx:idx + 1],
                'attention_mask': batch['attention_mask'][idx:idx + 1],
                'labels': batch['labels'][idx:idx + 1],
                'row_ids': batch['row_ids'][idx:idx + 1],
                'fork_window_count': 2,
                'fork_window_gas': 2,
            }
            # Note: scale single-row CE by 0.5 to match 2-row batch CE; fork uses window k_w=2, g_w=2 -> weight 0.1/2
            _, _, prefix = (1, 2, [2, 5])
            dec_gold = m_single.prepare_decoder_input_ids_from_labels(sub['labels'])
            z_g = m_single.base_model(input_ids=sub['input_ids'], attention_mask=sub['attention_mask'], decoder_input_ids=dec_gold, use_cache=False, return_dict=True).logits
            # Direct CE on 2-row full batch already verified above; compare direct LexicalForkMT forward on identical tokens
            # Wait: check CE invariance too? We sum unpadded individual fork losses directly:
            ce_row = F.cross_entropy(z_g.view(-1, z_g.shape[-1]), sub['labels'].view(-1), ignore_index=-100)
            entry = sel_map[idx]
            z_f = m_single.base_model(
                input_ids=sub['input_ids'],
                attention_mask=sub['attention_mask'],
                decoder_input_ids=torch.tensor([prefix + entry['h']], dtype=torch.long),
                use_cache=False,
                return_dict=True,
            ).logits[0, -1, :].float()
            sp = F.softplus(-float(entry['sign']) * (z_f[entry['b']] - z_f[entry['a']]))
            valid_toks = int((batch['labels'] != -100).sum())
            row_toks = int((sub['labels'] != -100).sum())
            ((row_toks / valid_toks) * ce_row + 0.1 * 0.5 * sp).backward()

        pad_diff = max(
            float((p1.grad - p2.grad).abs().max())
            for p1, p2 in zip(m_batched.parameters(), m_single.parameters())
            if p1.grad is not None
        )
        self.assertLess(pad_diff, 1e-5)

        # Reentrant gradient checkpointing parity + build_epoch_fork_table compatibility
        m_ckpt = LexicalForkMT(copy.deepcopy(base), method='lexical-fork-2x2-v1', tokenizer=tok)
        m_ckpt.set_epoch_fork_state(2, 0.1, sel_map)
        m_ckpt.enable_input_require_grads()
        m_ckpt.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        m_ckpt.train()
        m_ckpt.zero_grad()
        m_ckpt(**batch).loss.backward()
        ckpt_diff = max(
            float((p1.grad - p2.grad).abs().max())
            for p1, p2 in zip(m_batched.parameters(), m_ckpt.parameters())
            if p1.grad is not None
        )
        self.assertLess(ckpt_diff, 1e-5)

        _, ep_stats = build_epoch_fork_table(
            m_ckpt,
            tok,
            ['alpha beta', 'gamma delta'],
            ['alpha beta', 'gamma delta'],
            mode='2x2',
            seed=42,
            epoch_idx=2,
            min_pos_2x2=1,
            max_source_length=16,
            max_target_length=16,
        )
        self.assertTrue(m_ckpt.training)
        self.assertEqual(ep_stats['eligible_forks'], 2)

        # eval() and beam-4 generate() bit-exact parity + 0 extra parameters
        self.assertEqual(m_batched.extra_parameter_count(), 0)
        m_batched.eval()
        raw_ref = copy.deepcopy(m_batched.base_model).eval()
        with torch.no_grad():
            torch.testing.assert_close(
                m_batched(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels']).logits,
                raw_ref(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels']).logits,
            )
            gen_wrap = m_batched.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], num_beams=4, max_new_tokens=6)
            gen_raw = raw_ref.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], num_beams=4, max_new_tokens=6)
            self.assertTrue(torch.equal(gen_wrap, gen_raw))

    def test_save_reload_metadata_and_2step_control_trainer_resume_parity(self):
        tok = _AlphabeticForkTokenizer()
        sources = [f'w{i % 4} w{(i + 1) % 4} w{(i + 2) % 4}' for i in range(8)]
        refs = [f't{i % 3} t{(i + 1) % 3}' for i in range(8)]
        ds = Dataset.from_dict(
            {
                'input_ids': [tok._encode_one(s, max_length=12) for s in sources],
                'attention_mask': [[1] * len(tok._encode_one(s, max_length=12)) for s in sources],
                'labels': [[5, 8 + (i % 10), 10 + (i % 10), 2] for i in range(8)],
                'row_id': list(range(8)),
            }
        )
        probe = tiny_batch(batch_size=2, src_len=8, tgt_len=6, seed=41)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            full_dir = root / 'full'
            set_seed(42)
            model = LexicalForkMT(
                tiny_mbart(dropout=0.1, seed=42),
                method='lexical-fork-2x2-v1',
                seed=42,
                tokenizer=tok,
                max_source_length=16,
                max_target_length=16,
            )
            args = Seq2SeqTrainingArguments(
                output_dir=str(full_dir),
                seed=42,
                data_seed=42,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                max_steps=2,
                learning_rate=5e-4,
                logging_steps=1,
                save_strategy='steps',
                save_steps=1,
                eval_strategy='no',
                use_cpu=True,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                save_safetensors=False,
                restore_callback_states_from_checkpoint=True,
                remove_unused_columns=False,
                report_to=[],
            )
            cb = LexicalForkEpochCallback(full_dir, sources, refs, tok, seed=42, row_ids=list(range(8)), gate_mode=True)
            trainer = ControlTrainer(
                model=model,
                args=args,
                processing_class=tok,
                data_collator=LexicalForkCollator(tok, model, pad_to_multiple_of=8),
                train_dataset=ds,
                generation_dtype='fp32',
                callbacks=[cb],
            )
            trainer.train()
            self.assertEqual(trainer.state.global_step, 2)
            self.assertTrue(bool(model.selected_map))
            expected_selected_map = copy.deepcopy(model.selected_map)

            model.eval()
            with torch.no_grad():
                expected_logits = model(
                    input_ids=probe['input_ids'],
                    attention_mask=probe['attention_mask'],
                    labels=probe['labels'],
                ).logits.detach().cpu()

            ckpt1 = full_dir / 'checkpoint-1'
            self.assertTrue((ckpt1 / 'lexical_fork_config.json').is_file())
            self.assertTrue((ckpt1 / 'lexical_fork_state.json').is_file())

            # Reload from checkpoint-1, verify metadata mismatch detection, and resume step 2 identically
            set_seed(42)
            base_res = MBartForConditionalGeneration.from_pretrained(ckpt1)
            resumed = LexicalForkMT(
                base_res,
                method='lexical-fork-2x2-v1',
                seed=42,
                tokenizer=tok,
                max_source_length=16,
                max_target_length=16,
            )
            resumed.verify_checkpoint_metadata(ckpt1)
            resumed.restore_fork_state(ckpt1)
            self.assertEqual(resumed.active_epoch_idx, 2)
            self.assertEqual(resumed.selected_map, expected_selected_map)

            bad_wrapper = LexicalForkMT(base_res, method='lexical-fork-2x2-v1', seed=99, tokenizer=tok)
            with self.assertRaises(ValueError):
                bad_wrapper.verify_checkpoint_metadata(ckpt1)

            res_dir = root / 'resumed'
            res_args = copy.deepcopy(args)
            res_args.output_dir = str(res_dir)
            res_cb = LexicalForkEpochCallback(res_dir, sources, refs, tok, seed=42, row_ids=list(range(8)), gate_mode=True)
            resumed_trainer = ControlTrainer(
                model=resumed,
                args=res_args,
                processing_class=tok,
                data_collator=LexicalForkCollator(tok, resumed, pad_to_multiple_of=8),
                train_dataset=ds,
                generation_dtype='fp32',
                callbacks=[res_cb],
            )
            resumed_trainer.train(resume_from_checkpoint=str(ckpt1))
            self.assertEqual(resumed_trainer.state.global_step, 2)
            self.assertEqual(resumed.selected_map, expected_selected_map)
            resumed.eval()
            with torch.no_grad():
                resumed_logits = resumed(
                    input_ids=probe['input_ids'],
                    attention_mask=probe['attention_mask'],
                    labels=probe['labels'],
                ).logits.detach().cpu()
            torch.testing.assert_close(resumed_logits, expected_logits, rtol=1e-5, atol=2e-5)

            del trainer, resumed_trainer, model, resumed
            gc.collect()

    def test_m2m100_compatibility_and_validate_recipe_guards(self):
        tok = _AlphabeticForkTokenizer()
        m2m = LexicalForkMT(tiny_m2m100(dropout=0.0, seed=9), method='lexical-fork-2x2-v1', tokenizer=tok)
        bow, cont = build_lexical_boundary_masks(tok, 64, extra_special_ids=[0, 1, 2, 3, 4, 5])
        self.assertGreater(int(bow.sum()), 5)
        self.assertGreater(int(cont.sum()), 5)

        m2m.set_epoch_fork_state(2, 0.05, _sample_selected_map((0, 1)))
        m2m.train()
        batch = tiny_batch(batch_size=2, src_len=8, tgt_len=7, seed=55, row_start=0)
        batch['fork_window_count'] = 2
        batch['fork_window_gas'] = 1
        loss = m2m(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m2m.parameters()))

        cfg = json.loads((ROOT / 'configs/experiments/amis_lexical_fork_pilot_a100.json').read_text(encoding='utf-8'))
        for run in cfg['runs']:
            validate_recipe(cfg, run, ROOT)

        bad_lam = dict(cfg['runs'][1], lambda_epoch2=-0.01)
        with self.assertRaises(ValueError):
            validate_recipe(cfg, bad_lam, ROOT)

        bad_min = dict(cfg['runs'][1], min_pos_2x2=0)
        with self.assertRaises(ValueError):
            validate_recipe(cfg, bad_min, ROOT)

        bad_cfg = copy.deepcopy(cfg)
        bad_cfg['post_training_test'] = {'authorized': True}
        with self.assertRaises(ValueError):
            validate_recipe(bad_cfg, cfg['runs'][0], ROOT)


if __name__ == '__main__':
    unittest.main()
