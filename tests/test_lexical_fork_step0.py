"""CPU unit tests for Round 5 Lexical Fork Step-0 diagnostic helpers."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / 'src'), str(ROOT / 'scripts/analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU, CHRF
from transformers import MBartConfig, MBartForConditionalGeneration

import step0_lexical_fork_a100 as lf


class _MockSPMTokenizer:
    """Minimal SentencePiece-like tokenizer with mixed BOW, continuation, digits, punct, and special IDs."""

    pad_token_id = 1
    eos_token_id = 2
    bos_token_id = 0
    unk_token_id = 3
    all_special_ids = [0, 1, 2, 3, 4, 5]

    def __init__(self):
        # Vocab size 32
        self._pieces = {
            0: '<s>',
            1: '<pad>',
            2: '</s>',
            3: '<unk>',
            4: 'tl_XX',
            5: 'zh_CN',
            6: '\u2581',       # pure space marker -> excluded
            7: '.',            # punctuation -> excluded
            8: '\u2581123',    # digits only -> excluded
            9: '\u2581我',     # BOW content
            10: '\u2581你',    # BOW content
            11: '\u2581他',    # BOW content
            12: '爱',          # continuation content
            13: '喜',          # continuation content
            14: '欢',          # continuation content
            15: '\u2581苹果',  # BOW content
            16: '\u2581香蕉',  # BOW content
            17: '\u2581猫',    # BOW content
            18: '\u2581狗',    # BOW content
            19: '儿',          # continuation content
            20: '们',          # continuation content
            21: '\u2581好',    # BOW content
            22: '\u2581坏',    # BOW content
            23: '\u2581吃',    # BOW content
            24: '\u2581喝',    # BOW content
            25: '了',          # continuation content
            26: '着',          # continuation content
            27: '\u2581很',    # BOW content
            28: '\u2581真',    # BOW content
            29: '\u2581天',    # BOW content
            30: '气',          # continuation content
            31: '!',           # punctuation -> excluded
        }
        self._text = {
            k: v.replace('\u2581', ' ') for k, v in self._pieces.items()
        }

    def convert_ids_to_tokens(self, tid):
        return self._pieces.get(int(tid))

    def decode(self, ids, skip_special_tokens=True):
        skip = set(self.all_special_ids) if skip_special_tokens else set()
        return ''.join(self._text[int(t)] for t in ids if int(t) not in skip).strip()

    def __call__(self, texts, return_tensors='pt', padding=True, truncation=True, max_length=32):
        # Deterministically turn each string into a short [tl_XX, ..., </s>] source sequence
        seqs = []
        for s in texts:
            ids = [4] + [9 + (ord(ch) % 20) for ch in s[:6]] + [2]
            seqs.append(ids[:max_length])
        max_l = max(len(x) for x in seqs)
        inp = torch.full((len(seqs), max_l), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), max_l), dtype=torch.long)
        for i, seq in enumerate(seqs):
            inp[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, :len(seq)] = 1
        return {'input_ids': inp, 'attention_mask': mask}


def _tiny_mbart(vocab_size=32, seed=13):
    torch.manual_seed(seed)
    cfg = MBartConfig(
        vocab_size=vocab_size,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        forced_bos_token_id=5,
    )
    model = MBartForConditionalGeneration(cfg)
    model.eval()
    return model


class LexicalForkStep0Gate(unittest.TestCase):
    def test_lexical_boundary_masks(self):
        tok = _MockSPMTokenizer()
        bow, cont = lf.build_lexical_boundary_masks(tok, 32)
        for excluded in (0, 1, 2, 3, 4, 5, 6, 7, 8, 31):
            self.assertFalse(bool(bow[excluded]))
            self.assertFalse(bool(cont[excluded]))
        for bow_id in (9, 10, 11, 15, 16, 17, 18, 21, 22, 23, 24, 27, 28, 29):
            self.assertTrue(bool(bow[bow_id]))
            self.assertFalse(bool(cont[bow_id]))
        for cont_id in (12, 13, 14, 19, 20, 25, 26, 30):
            self.assertFalse(bool(bow[cont_id]))
            self.assertTrue(bool(cont[cont_id]))

    def test_bucketed_branch_rollout_exact_parity_with_single_example(self):
        tok = _MockSPMTokenizer()
        model = _tiny_mbart(vocab_size=32, seed=7)
        # Encourage occasional EOS so trajectories terminate cleanly
        with torch.no_grad():
            model.final_logits_bias[0, 9:18] += 0.4
            model.final_logits_bias[0, 2] += 0.2
        bow, cont = lf.build_lexical_boundary_masks(tok, 32)
        sources = ['alpha one', 'beta two three', 'gamma', 'delta four']
        device = torch.device('cpu')

        nat = lf.greedy_native_with_rivals(
            model, tok, sources, bow, cont, device, batch_size=2, max_new_tokens=10
        )
        # Compare native greedy tokens against single-example unbatched forward decoding
        for r_idx, src in enumerate(sources):
            enc = tok([src])
            prefix = [2, 5]
            seq = []
            for step in range(10):
                dec = torch.tensor([prefix + seq], dtype=torch.long)
                with torch.no_grad():
                    logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], decoder_input_ids=dec).logits[0, -1, :]
                top1 = int(logits.argmax().item())
                if top1 == 2:
                    break
                seq.append(top1)
            self.assertEqual(nat[r_idx]['native_tokens'], seq)

        forks = lf.select_fork_positions(nat, policy='random_prefix', seed=42)
        self.assertGreaterEqual(len(forks), 1)
        tails = lf.greedy_branch_rollouts(model, tok, sources, nat, forks, device, batch_size=2, max_target_length=16)

        for f in forks:
            i, j, b_tok = f['row_idx'], f['j'], f['b']
            enc = tok([sources[i]])
            tail_single = []
            cur_pref = [2, 5] + nat[i]['native_tokens'][:j] + [b_tok]
            for _ in range(16 - len(cur_pref)):
                dec = torch.tensor([cur_pref + tail_single], dtype=torch.long)
                with torch.no_grad():
                    logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], decoder_input_ids=dec).logits[0, -1, :]
                nt = int(logits.argmax().item())
                if nt == 2:
                    break
                tail_single.append(nt)
            self.assertEqual(tails[(i, j, b_tok)], tail_single)

    def test_corpus_sufficient_stats_exact_parity_and_6way_sign(self):
        # Craft 6 rows where row 0 strictly gains (+1), row 1 strictly loses (-1), row 2 disagrees across tails (0).
        tok = _MockSPMTokenizer()
        refs = ['我 爱 苹果', '我 吃 香蕉', '今天 天气 很 好', '他 喜欢 猫', '你 喝 茶', '我 爱 猫']
        native_rows = [
            dict(row_idx=0, native_tokens=[9, 13, 16], rivals=[10, 12, 15], margins=[0.1, 0.1, 0.1], eligible_positions=[1], native_text=tok.decode([9, 13, 16])),
            dict(row_idx=1, native_tokens=[9, 23, 16], rivals=[10, 24, 15], margins=[0.2, 0.2, 0.2], eligible_positions=[1], native_text=tok.decode([9, 23, 16])),
            dict(row_idx=2, native_tokens=[29, 30, 27, 22], rivals=[29, 12, 27, 21], margins=[0.1, 0.1, 0.1, 0.1], eligible_positions=[3], native_text=tok.decode([29, 30, 27, 22])),
            dict(row_idx=3, native_tokens=[11, 13, 14, 17], rivals=[-1]*4, margins=[1.0]*4, eligible_positions=[], native_text=tok.decode([11, 13, 14, 17])),
            dict(row_idx=4, native_tokens=[10, 24], rivals=[-1]*2, margins=[1.0]*2, eligible_positions=[], native_text=tok.decode([10, 24])),
            dict(row_idx=5, native_tokens=[9, 12, 17], rivals=[-1]*3, margins=[1.0]*3, eligible_positions=[], native_text=tok.decode([9, 12, 17])),
        ]
        # Row 0: h=[9], a=13('喜'), u_a=[16](' 香蕉'); b=12('爱'), u_b=[15](' 苹果') -> all 4 comparisons favor b=12!
        # Row 1: h=[9], a=23('吃'), u_a=[16](' 香蕉'); b=24('喝'), u_b=[17](' 猫') -> ref is '我 吃 香蕉' so a=23 wins everywhere (-1)
        # Row 2: h=[29,30,27], a=22('坏'), u_a=[]; b=21('好'), u_b=[22](' 坏') -> u_b destroys b on h b u_b vs h a u_b
        forks = [
            dict(row_idx=0, j=1, a=13, b=12, margin=0.1),
            dict(row_idx=1, j=1, a=23, b=24, margin=0.2),
            dict(row_idx=2, j=3, a=22, b=21, margin=0.1),
        ]
        tails = {
            (0, 1, 12): [15],
            (1, 1, 24): [17],
            (2, 3, 21): [22],
        }
        summary, recs = lf.evaluate_lexical_forks(tok, refs, native_rows, forks, tails, bleu_tok='zh', n_draws=20)
        self.assertEqual(recs[0]['s_6way'], 1)
        self.assertEqual(recs[1]['s_6way'], -1)
        self.assertEqual(recs[2]['s_6way'], 0)
        self.assertEqual(summary['counts']['pos_2x2'], 1)
        self.assertEqual(summary['counts']['neg_2x2'], 1)
        self.assertEqual(summary['counts']['abstain_2x2'], 1)

        # Verify exact corpus BLEU/chrF0 parity with direct sacrebleu.corpus_score
        edited_1tok = [tok.decode([9, 12, 16])] + [r['native_text'] for r in native_rows[1:]]
        direct_b = BLEU(tokenize='zh').corpus_score(edited_1tok, [refs]).score
        direct_c = CHRF(word_order=0).corpus_score(edited_1tok, [refs]).score
        self.assertAlmostEqual(summary['metrics']['oracle_one_token_2x2']['bleu'], round(direct_b, 4), places=4)
        self.assertAlmostEqual(summary['metrics']['oracle_one_token_2x2']['chrf0'], round(direct_c, 4), places=4)

    def test_hybrid_nll_exact_arithmetic(self):
        tok = _MockSPMTokenizer()
        model = _tiny_mbart(vocab_size=32, seed=5)
        sources = ['alpha']
        native_rows = [dict(row_idx=0, native_tokens=[9, 13, 16, 21])]
        fork_records = [dict(row_idx=0, j=1, a=13, b=12, u_a=[16, 21], u_b=[15], s_6way=1)]
        res = lf.measure_hybrid_nll(model, tok, sources, native_rows, fork_records, torch.device('cpu'), batch_size=2, max_target_length=32)

        # Verify direct cross_entropy match for h b u_a minus h a u_a
        enc = tok(sources)
        with torch.no_grad():
            def seq_stats(piv, tail):
                tail_eos = tail + [2]
                dec = torch.tensor([[2, 5, 9, piv] + tail_eos[:-1]], dtype=torch.long)
                logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], decoder_input_ids=dec).logits[0, 3:3 + len(tail_eos), :]
                nlls = F.cross_entropy(logits.float(), torch.tensor(tail_eos), reduction='none')
                return float(nlls.mean()), float(nlls[0])
            m0, j0 = seq_stats(13, [16, 21])
            m1, j1 = seq_stats(12, [16, 21])
            m2, j2 = seq_stats(12, [15])
            m3, j3 = seq_stats(13, [15])

        self.assertAlmostEqual(res['pos_2x2']['delta_nll_ua_mean'], round(m1 - m0, 4), places=4)
        self.assertAlmostEqual(res['pos_2x2']['delta_nll_ub_mean'], round(m3 - m2, 4), places=4)
        self.assertAlmostEqual(res['pos_2x2']['junction_delta_ua_mean'], round(j1 - j0, 4), places=4)
        self.assertAlmostEqual(res['pos_2x2']['junction_delta_ub_mean'], round(j3 - j2, 4), places=4)

    def test_fp64_lexical_fork_loss_gradient(self):
        z = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float64)
        active = s.ne(0)
        diff = z[:, 1] - z[:, 0]
        lam = 0.25
        loss = (lam / 4.0) * (active.to(torch.float64) * F.softplus(-s * diff)).sum()
        loss.backward()

        sig = torch.sigmoid(-s * diff.detach())
        exp_dzb = (lam / 4.0) * torch.where(active, -s * sig, torch.zeros_like(s))
        self.assertLess(float((z.grad[:, 1] - exp_dzb).abs().max()), 1e-12)
        self.assertLess(float((z.grad[:, 0] + exp_dzb).abs().max()), 1e-12)
        self.assertEqual(float(z.grad[2].abs().max()), 0.0)

    def test_replica_order_invariance_on_tiny_mbart(self):
        tok = _MockSPMTokenizer()
        model = _tiny_mbart(vocab_size=32, seed=19)
        with torch.no_grad():
            model.final_logits_bias[0, 9:18] += 0.4
            model.final_logits_bias[0, 2] += 0.2
        bow, cont = lf.build_lexical_boundary_masks(tok, 32)
        sources = ['one a', 'two b', 'three c', 'four d']
        refs = ['我 爱 苹果', '我 吃 香蕉', '他 喜欢 猫', '你 喝 茶']
        dev = torch.device('cpu')

        nat = lf.greedy_native_with_rivals(model, tok, sources, bow, cont, dev, batch_size=2, max_new_tokens=10)
        forks = lf.select_fork_positions(nat, policy='random_prefix', seed=42)
        self.assertGreaterEqual(len(forks), 1)
        tails = lf.greedy_branch_rollouts(model, tok, sources, nat, forks, dev, batch_size=2, max_target_length=16)
        _, recs = lf.evaluate_lexical_forks(tok, refs, nat, forks, tails, bleu_tok='zh', n_draws=10)
        rep = lf.measure_replica_disagreement(
            model, tok, sources, refs, bow, cont, nat, recs, dev,
            bleu_tok='zh', n_rows=4, batch_size_rep2=1, max_source_length=32, max_target_length=16, max_new_tokens=10,
        )
        self.assertEqual(rep['forks_checked'], len(forks))
        self.assertEqual(rep['native_token_disagreement_rate'], 0.0)
        self.assertEqual(rep['sign_disagreement_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()
