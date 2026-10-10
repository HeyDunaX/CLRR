"""Independent scoring audit; writes a new receipt, preserves original results."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import sacrebleu

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((OUT / 'final_evaluation_manifest.json').read_text(encoding='utf-8'))
test_path = ROOT / manifest['protocol']['test_dataset']
test = rows(test_path)
assert sha(test_path) == manifest['protocol']['test_sha256']
refs = [r['target'] for r in test]
assert len(refs) == 575
records = rows(OUT / 'test_scores_all_runs.csv')
assert len(records) == 18
audit = []
systems = {}
for record in records:
    path = ROOT / record['prediction_file']
    predictions = rows(path)
    assert len(predictions) == len(test)
    assert sha(path) == record['prediction_sha256']
    assert all(p['source'] == t['source'] and p['target'] == t['target'] for p, t in zip(predictions, test))
    assert [int(p['index']) for p in predictions] == list(range(575))
    hyps = [p['prediction'] for p in predictions]
    scores = {
        'BLEU': sacrebleu.corpus_bleu(hyps, [refs], tokenize='zh').score,
        'chrF++': sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
        'chrF0': sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score,
    }
    error = max(abs(scores[key] - float(record[key])) for key in scores)
    assert error < 1e-8, (record, scores)
    systems[record['family'], int(record['seed'])] = hyps
    audit.append({'family': record['family'], 'seed': int(record['seed']), 'max_score_error': error, 'scores': scores})

metric = sacrebleu.metrics.CHRF(word_order=2)
reference_word_bigram_rows = [i for i, stat in enumerate(metric._extract_corpus_statistics(refs, [refs])) if stat[-2] > 0]
# Native sentence statistics record hypothesis/reference/match per n-gram order.
word_bigram_stats = {}
leave_one_out_deltas = {}
for seed in [42, 43, 44]:
    for family in ['original_full_detached', 'tuned_lsr']:
        hyps = systems[family, seed]
        stats = metric._extract_corpus_statistics(hyps, [refs])
        word_bigram_stats[f'{family}:{seed}'] = {str(i): stats[i][-3:] for i in reference_word_bigram_rows}
    full, tuned = systems['original_full_detached', seed], systems['tuned_lsr', seed]
    for excluded in [None, 170]:
        indices = [i for i in range(575) if i != excluded]
        rs = [refs[i] for i in indices]
        gap = sacrebleu.corpus_chrf([full[i] for i in indices], [rs], word_order=2).score - sacrebleu.corpus_chrf([tuned[i] for i in indices], [rs], word_order=2).score
        leave_one_out_deltas[f'{seed}:excluded={excluded}'] = gap

bootstrap = {}
for size in [18, 30]:
    table = rows(OUT / f'test_paired_bootstrap{size}.csv')
    assert len(table) == size
    bootstrap[str(size)] = {
        'pairs': sorted({(r['challenger'], r['control']) for r in table}),
        'significant_holm_05': sum(float(r[f'p_holm{size}']) < .05 for r in table),
        'unadjusted_ci_excludes_zero': sum(float(r['ci_low']) > 0 or float(r['ci_high']) < 0 for r in table),
    }

receipt = {
    'sacrebleu_version': sacrebleu.__version__,
    'prediction_sets_verified': len(audit), 'raw_test_sha256_verified': sha(test_path),
    'alignment_and_prediction_hashes_verified': True, 'score_audit': audit,
    'reference_word_bigram_rows': reference_word_bigram_rows,
    'word_bigram_stats_hyp_ref_match': word_bigram_stats,
    'full_minus_tuned_chrfpp_leave_one_out': leave_one_out_deltas,
    'bootstrap_file_review': bootstrap,
    'notes': ['Leave-one-out is a diagnostic; primary test scores remain unchanged.',
              'Current secondary30 pairs differ from handoff section D declared secondary30; diagnostic Full lambda .03 is absent from final matrix.',
              'No training, generation, HF access or original artifact modifications in this audit.'],
}
(OUT / 'final_review_20261004.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items() if key != 'score_audit'}, indent=2, ensure_ascii=False))
