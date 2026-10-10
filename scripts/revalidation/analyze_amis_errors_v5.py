"""Descriptive analysis of verified saved predictions; never selects settings."""
import csv
import hashlib
import json
from pathlib import Path

import sacrebleu
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
FAMILIES = ['baseline', 'original_full_detached', 'matched_lsr', 'tuned_lsr']
BINS = {'<=5': lambda n: n <= 5, '6-10': lambda n: 6 <= n <= 10, '>10': lambda n: n > 10}
CHRF0 = CHRF(word_order=0)
CHRFPP = CHRF(word_order=2)


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(rows, gold):
    assert len(rows) == len(gold), (len(rows), len(gold))
    for i, (row, ref) in enumerate(zip(rows, gold)):
        assert int(row['index']) == i, (i, row['index'])
        assert row['source'] == ref['source'] and row['target'] == ref['target'], i


def metrics(preds, refs):
    if not preds:
        return {'rows': 0}
    chars = lambda text: sum(not c.isspace() for c in text)
    ref_chars = sum(map(chars, refs))
    support = [sum(CHRFPP._extract_reference_info([ref])['ref_ngrams'][0][7].values()) for ref in refs]
    return {'rows': len(preds), 'BLEUzh': sacrebleu.corpus_bleu(preds, [refs], tokenize='zh').score,
            'chrF++': CHRFPP.corpus_score(preds, [refs]).score,
            'chrF0': CHRF0.corpus_score(preds, [refs]).score,
            'prediction_reference_nonwhitespace_character_ratio': sum(map(chars, preds)) / ref_chars if ref_chars else None,
            'empty_prediction_rate': sum(not p.strip() for p in preds) / len(preds),
            'reference_word_bigram_count': sum(support),
            'reference_rows_with_word_bigrams': sum(n > 0 for n in support)}


def summarize(rows, column='prediction'):
    result = {'all': metrics([r[column] for r in rows], [r['target'] for r in rows]), 'source_word_bins': {}}
    for label, condition in BINS.items():
        subset = [r for r in rows if condition(len(r['source'].split()))]
        result['source_word_bins'][label] = metrics([r[column] for r in subset], [r['target'] for r in subset])
    return result


def compare(full, other, full_col='prediction', other_col='prediction'):
    results = {}
    for label, condition in [('all', lambda n: True), *BINS.items()]:
        pairs = [(a, b) for a, b in zip(full, other) if condition(len(a['source'].split()))]
        refs = [a['target'] for a, _ in pairs]
        fp, op = [a[full_col] for a, _ in pairs], [b[other_col] for _, b in pairs]
        fm, om = metrics(fp, refs), metrics(op, refs)
        differences = [CHRF0.sentence_score(a, [r]).score - CHRF0.sentence_score(b, [r]).score for a, b, r in zip(fp, op, refs)]
        results[label] = {'rows': len(pairs), 'corpus_delta_full_minus_comparator':
                         {k: fm[k] - om[k] for k in ['BLEUzh', 'chrF++', 'chrF0']} if pairs else {},
                         'sentence_chrF0_descriptive_only': {'win': sum(d > 1e-10 for d in differences),
                             'loss': sum(d < -1e-10 for d in differences), 'tie': sum(abs(d) <= 1e-10 for d in differences)}}
    return results


def main():
    gold = read(ROOT / 'data_processed/amis_mandarin/test.csv')
    val_gold = read(ROOT / 'data_processed/amis_mandarin/validation.csv')
    assert len(gold) == 575
    score_rows = read(ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004/test_scores_all_runs.csv')
    report = {'scope': 'Descriptive only; test is not used for hyperparameter selection.',
              'source_word_bins_defined_before_analysis': ['<=5', '6-10', '>10'],
              'source_word_definition': 'whitespace split of raw source',
              'metric_implementation': 'native SacreBLEU corpus BLEU tokenize=zh, CHRF word_order=2 and 0',
              'sacrebleu_version': sacrebleu.__version__, 'test_decode': 'existing goldtest beam4',
              'test_gold_sha256': sha(ROOT / 'data_processed/amis_mandarin/test.csv'),
              'test': {}, 'test_comparisons': {}, 'validation': {}, 'validation_comparisons': {},
              'v5_prediction_coverage': {'available': [], 'missing': []}}
    loaded = {}
    inventory = json.loads((ROOT / 'outputs_rebuttal/amis_followup_20261004_v2/validation_inventory_21.json').read_text())
    for family in FAMILIES:
        for seed in [42, 43, 44]:
            selected = [r for r in score_rows if r['family'] == family and int(r['seed']) == seed]
            assert len(selected) == 1
            entry = selected[0]
            path = ROOT / entry['prediction_file']
            assert sha(path) == entry['prediction_sha256'], str(path)
            rows = read(path)
            verify(rows, gold)
            key = f'{family}_seed{seed}'
            loaded[(family, seed)] = rows
            report['test'][key] = {'prediction_file': entry['prediction_file'], 'verified_sha256': sha(path), **summarize(rows)}
            actual = report['test'][key]['all']
            assert all(abs(actual[a] - float(entry[b])) < 1e-8 for a, b in [('BLEUzh', 'BLEU'), ('chrF++', 'chrF++'), ('chrF0', 'chrF0')]), key
    for seed in [42, 43, 44]:
        for comparator in ['baseline', 'matched_lsr', 'tuned_lsr']:
            report['test_comparisons'][f'full_minus_{comparator}_seed{seed}'] = compare(loaded[('original_full_detached', seed)], loaded[(comparator, seed)])
    val = {}
    for entry in inventory:
        family, seed = entry['family'], entry['seed']
        if family not in FAMILIES:
            continue
        model_id = entry['id']
        for mode in (['val_main_b1', 'val_main_b4', 'test_main_b1'] if family in ['original_full_detached', 'tuned_lsr'] else ['val_main_b1']):
            path = OUT / 'runs' / model_id / mode / 'predictions.csv'
            if family == 'baseline' and mode == 'val_main_b1' and not path.exists():
                path = OUT / 'baseline_archive_audit_v3' / model_id / 'validation_predictions.csv'
            rel = str(path.relative_to(ROOT))
            report['v5_prediction_coverage']['available' if path.exists() else 'missing'].append(rel)
            if not path.exists():
                continue
            rows = read(path)
            verify(rows, gold if mode.startswith('test') else val_gold)
            report['validation' if mode.startswith('val') else 'test'][f'v5_{family}_seed{seed}_{mode}'] = {'prediction_file': rel, 'observed_sha256': sha(path), **summarize(rows)}
            if mode == 'val_main_b1':
                val[(family, seed)] = rows
        if family == 'original_full_detached':
            path = ROOT / 'outputs_rebuttal/amis_followup_20261004_v4/route_control' / model_id / 'route_predictions_comparison.csv'
            if path.exists():
                rows = read(path)
                verify(rows, val_gold)
                report['validation'][f'v4_full_seed{seed}'] = {'prediction_file': str(path.relative_to(ROOT)), 'observed_sha256': sha(path),
                    'route_on': summarize(rows, 'prediction_route_on'), 'route_off': summarize(rows, 'prediction_route_off')}
                report['validation_comparisons'][f'v4_route_on_minus_off_seed{seed}'] = compare(rows, rows, 'prediction_route_on', 'prediction_route_off')
    for seed in [42, 43, 44]:
        for comparator in ['baseline', 'matched_lsr', 'tuned_lsr']:
            if ('original_full_detached', seed) in val and (comparator, seed) in val:
                report['validation_comparisons'][f'v5_full_minus_{comparator}_seed{seed}'] = compare(val[('original_full_detached', seed)], val[(comparator, seed)])
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'error_analysis.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'artifact': str(target), 'verified_test_runs': len(loaded), 'v5_available': len(report['v5_prediction_coverage']['available']),
                      'test_full_minus_tuned': {str(seed): report['test_comparisons'][f'full_minus_tuned_lsr_seed{seed}']['all'] for seed in [42, 43, 44]}}, indent=2))


if __name__ == '__main__':
    main()
