"""Descriptive prediction strata and leave-one-row-out influence; no new tests or selection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('predictions-dir', 'references', 'train', 'design', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    parser.add_argument('--flat-layout', action='store_true', help='Run folders directly under predictions-dir')
    parser.add_argument('--grounding', type=Path, help='Optional frozen four-cell validation probe workspace')
    args = parser.parse_args()
    raw = pd.read_csv(args.references, dtype=str, keep_default_na=False)
    train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    design = json.loads(args.design.read_text())
    refs = raw.target.str.strip().tolist()
    ids = sorted({ident for pair in design['pairs'] for ident in pair})
    metrics = {'BLEU': BLEU(tokenize=design['bleu_tokenizer']),
               'chrF++': CHRF(word_order=2), 'chrF0': CHRF(word_order=0)}
    systems, stats, totals, provenance = {}, {}, {}, {}
    for ident in ids:
        folder = args.predictions_dir / ident if args.flat_layout else args.predictions_dir / 'runs' / ident
        path = folder / f'{args.split}_predictions.csv'
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert frame[['source', 'target']].equals(raw[['source', 'target']]), ident
        if args.split == 'test':
            receipt = json.loads((folder / 'COMPLETE.json').read_text())
            assert receipt['status'] == 'COMPLETE_VERIFIED' and receipt['test_backup']['verified']
        else:
            receipt = json.loads((folder / 'VAL_COMPLETE.json').read_text())
            meta = json.loads((folder / 'validation_metrics.json').read_text())
            assert receipt['status'] == 'VALIDATION_COMPLETE_VERIFIED' and receipt['backup']['verified']
            assert meta['beams'] == 1 and meta['generation_precision'] == 'fp32'
            assert meta['raw_reference_sha256'] == hashlib.sha256(args.references.read_bytes()).hexdigest()
        systems[ident] = frame.prediction.str.strip().tolist()
        provenance[ident] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name, metric in metrics.items():
            values = np.asarray(metric._extract_corpus_statistics(systems[ident], [refs]), dtype=np.float64)
            stats[ident, name], totals[ident, name] = values, values.sum(0)
            score = metric._compute_score_from_stats(totals[ident, name]).score
            key = {'BLEU': 'bleu', 'chrF++': 'chrf++', 'chrF0': 'chrf0'}[name]
            assert abs(score - receipt['scores'][key]) < 1e-8
    source_words = [s.casefold().split() for s in raw.source]
    frequency = Counter(word for source in train.source for word in source.casefold().split())
    source_length = np.asarray([len(words) for words in source_words])
    target_length = np.asarray([len(''.join(s.split())) for s in refs])
    rare_fraction = np.asarray([sum(frequency[w] <= 1 for w in words) / max(1, len(words)) for words in source_words])
    axes = {'source_words': source_length, 'target_characters': target_length,
            'rare_source_fraction': rare_fraction}
    cuts, groups = {}, {}
    for axis, values in axes.items():
        boundaries = np.quantile(values, [.25, .5, .75])
        cuts[axis] = boundaries.tolist()
        bins = np.searchsorted(boundaries, values, side='left')
        for q in range(4):
            indices = np.flatnonzero(bins == q)
            if len(indices):
                groups[f'{axis}_Q{q+1}'] = indices
    training_sources = {' '.join(s.casefold().split()) for s in train.source}
    overlap = np.asarray([' '.join(s.casefold().split()) in training_sources for s in raw.source])
    for label, mask in [('source_seen_in_train', overlap), ('source_unseen_in_train', ~overlap)]:
        if mask.any():
            groups[label] = np.flatnonzero(mask)
    args.output.mkdir(parents=True, exist_ok=True)
    system_diagnostics = {}
    def ratio(matches, count):
        return np.divide(matches, count, out=np.zeros_like(matches, dtype=float), where=count != 0)
    for ident, predictions in systems.items():
        lengths = np.asarray([len(''.join(s.split())) for s in predictions])
        ratios = lengths / np.maximum(1, target_length)
        unigram = stats[ident, 'chrF0'][:, :3]
        word_bigram = stats[ident, 'chrF++'][:, -3:].sum(0)
        repeated = [max(Counter(s[i:i+3] for i in range(len(s)-2)).values(), default=0) >= 3
                    for s in [''.join(p.split()) for p in predictions]]
        system_diagnostics[ident] = dict(empty=int((lengths == 0).sum()),
            exact_reference=sum(p == r for p, r in zip(predictions, refs)),
            mean_length_ratio=float(ratios.mean()), median_length_ratio=float(np.median(ratios)),
            short_ratio_lt_0_5=int((ratios < .5).sum()), long_ratio_gt_1_5=int((ratios > 1.5).sum()),
            repeated_character_trigram_ge3=int(sum(repeated)),
            char_unigram_precision_mean=float(ratio(unigram[:,2], unigram[:,0]).mean()),
            char_unigram_recall_mean=float(ratio(unigram[:,2], unigram[:,1]).mean()),
            word_bigram_hyp_ref_match=word_bigram.tolist())
    grounding = {}
    if args.grounding:
        assert args.split == 'validation', 'Grounding join is validation-only'
        grounding_plan = json.loads((args.grounding / 'PLAN.json').read_text())
        assert grounding_plan['validation_sha256'] == hashlib.sha256(args.references.read_bytes()).hexdigest()
        for ident in ids:
            folder = args.predictions_dir / ident if args.flat_layout else args.predictions_dir / 'runs' / ident
            val_receipt = json.loads((folder / 'VAL_COMPLETE.json').read_text())
            assert val_receipt['archive_sha256'] == grounding_plan['selections'][ident]['archive_sha256']
            g = pd.read_csv(args.grounding / ident / 'per_sentence.csv').sort_values('row')
            receipt = json.loads((args.grounding / ident / 'RESULT.json').read_text())
            assert hashlib.sha256((args.grounding / ident / 'per_sentence.csv').read_bytes()).hexdigest() == receipt['csv_sha256']
            assert g.row.tolist() == list(range(len(raw)))
            grounding[ident] = g
    strata, influence, sentence, examples, associations = [], [], [], [], []
    for control, candidate in design['pairs']:
        a, b = systems[control], systems[candidate]
        deltas = np.asarray([metrics['chrF0'].sentence_score(y, [r]).score -
                             metrics['chrF0'].sentence_score(x, [r]).score for x, y, r in zip(a, b, refs)])
        for i, delta in enumerate(deltas):
            ca = stats[control, 'chrF0'][i, :3]; cb = stats[candidate, 'chrF0'][i, :3]
            sentence.append({'control': control, 'candidate': candidate, 'row': i,
                             'sentence_chrf0_delta': float(delta), 'source_words': int(source_length[i]),
                             'target_characters': int(target_length[i]), 'rare_source_fraction': float(rare_fraction[i]),
                             'identical_prediction': a[i] == b[i],
                             'char_unigram_precision_delta': float(cb[2]/max(1,cb[0])-ca[2]/max(1,ca[0])),
                             'char_unigram_recall_delta': float(cb[2]/max(1,cb[1])-ca[2]/max(1,ca[1])),
                             'control_length_ratio': len(''.join(a[i].split())) / max(1, target_length[i]),
                             'candidate_length_ratio': len(''.join(b[i].split())) / max(1, target_length[i])})
            if grounding:
                branch_a = design['native_branches'][control]; branch_b = design['native_branches'][candidate]
                sentence[-1].update(
                    native_CE_benefit=float(grounding[control].iloc[i][branch_a+'_match']-grounding[candidate].iloc[i][branch_b+'_match']),
                    native_gap_delta=float(grounding[candidate].iloc[i]['g_'+branch_b]-grounding[control].iloc[i]['g_'+branch_a]))
        joined = pd.DataFrame([r for r in sentence if r['control'] == control and r['candidate'] == candidate])
        signs = np.where(deltas > 1e-8, 'gain', np.where(deltas < -1e-8, 'loss', 'tie'))
        entry = dict(control=control, candidate=candidate,
                     identical_predictions=int(sum(x == y for x,y in zip(a,b))),
                     sentence_chrf0_counts={s: int((signs == s).sum()) for s in ['gain','loss','tie']},
                     mean_sentence_chrf0_delta=float(deltas.mean()))
        if grounding:
            for column in ['native_CE_benefit', 'native_gap_delta']:
                corr = joined[column].rank().corr(joined.sentence_chrf0_delta.rank())
                entry[column+'_spearman'] = None if pd.isna(corr) else float(corr)
                for label, mask in [('up', joined[column] > 1e-4), ('down', joined[column] < -1e-4),
                                    ('near_zero', joined[column].abs() <= 1e-4)]:
                    entry[column+'_'+label] = {s: int(((signs == s) & mask).sum()) for s in ['gain','loss','tie']}
        associations.append(entry)
        for label, order in [('largest_gain', np.argsort(-deltas)), ('largest_loss', np.argsort(deltas))]:
            for i in order[:10]:
                examples.append({'control': control, 'candidate': candidate, 'category': label, 'row': int(i),
                                 'sentence_chrf0_delta': float(deltas[i]), 'source': raw.source.iloc[i],
                                 'target': refs[i], 'control_prediction': a[i], 'candidate_prediction': b[i]})
        for name, metric in metrics.items():
            observed = metric._compute_score_from_stats(totals[candidate, name]).score - metric._compute_score_from_stats(totals[control, name]).score
            leave_out = np.asarray([metric._compute_score_from_stats(totals[candidate, name] - stats[candidate, name][i]).score -
                                    metric._compute_score_from_stats(totals[control, name] - stats[control, name][i]).score for i in range(len(refs))])
            idx = int(np.argmax(np.abs(leave_out - observed)))
            retained = [j for j in range(len(refs)) if j != idx]
            direct = metric.corpus_score([b[j] for j in retained], [[refs[j] for j in retained]]).score - metric.corpus_score([a[j] for j in retained], [[refs[j] for j in retained]]).score
            assert abs(direct - leave_out[idx]) < 1e-8
            influence.append({'control': control, 'candidate': candidate, 'metric': name, 'full_delta': observed,
                              'leave_one_out_min': float(leave_out.min()), 'leave_one_out_max': float(leave_out.max()),
                              'largest_influence_row': idx, 'max_abs_delta_change': float(abs(leave_out[idx] - observed)),
                              'sign_changes': int(np.sum((leave_out > 0) != (observed > 0))), 'direct_replay_error': abs(direct - leave_out[idx])})
            for label, indices in groups.items():
                scores = [metric._compute_score_from_stats(stats[ident, name][indices].sum(0)).score for ident in (control, candidate)]
                strata.append({'control': control, 'candidate': candidate, 'group': label, 'rows': len(indices),
                               'metric': name, 'control_score': scores[0], 'candidate_score': scores[1], 'delta': scores[1] - scores[0]})
    for filename, records in [('strata.csv', strata), ('influence.csv', influence),
                              ('sentence_deltas.csv', sentence), ('examples.csv', examples)]:
        pd.DataFrame(records).to_csv(args.output / filename, index=False)
    report = {'status': 'PASS', 'rows': len(refs), 'systems': len(systems), 'pairs': len(design['pairs']),
              'split': args.split, 'generation_reused': True,
              'quartile_boundaries': cuts, 'train_source_overlap_rows': int(overlap.sum()),
              'provenance': provenance, 'reference_sha256': hashlib.sha256(args.references.read_bytes()).hexdigest(),
              'scope': 'Post-hoc descriptive strata. No significance testing, checkpoint reselection or removal from primary test. '
                       'Length and rarity flags are automatic descriptions, not verified linguistic error categories. '
                       'Subset corpus BLEU deltas are not additive contributions to full-corpus BLEU.'}
    (args.output / 'PREDICTION_ANALYSIS.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'SYSTEM_DIAGNOSTICS.json').write_text(json.dumps(system_diagnostics, indent=2), encoding='utf-8')
    (args.output / 'GENERATION_ASSOCIATION.json').write_text(json.dumps(associations, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
