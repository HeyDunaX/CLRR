"""Audit all twelve frozen C outputs without changing primary scores or test rows."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs_rebuttal/amis_confirmation_20261003'
OUT = BASE / 'metric_influence'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert sacrebleu.__version__ == '2.6.0', sacrebleu.__version__
    assert not (OUT/'COMPLETE.json').exists(), 'Audit already complete; inspect existing outputs.'
    reference = ROOT/'data_processed/amis_mandarin/test.csv'
    raw = read(reference)
    refs = [r['target'].strip() for r in raw]
    metrics = {0: CHRF(word_order=0), 2: CHRF(word_order=2)}
    scores, systems, stats, totals, sources = [], {}, {}, {}, {}
    for row in read(BASE/'seed_scores.csv'):
        seed, method = int(row['seed']), row['method']
        path = BASE/'runs'/f'mbart-amis-{method}-seed{seed}'/'test_predictions.csv'
        assert digest(path) == row['prediction_sha256'], path
        records = read(path)
        assert len(records) == len(raw) == 575
        assert [r['source'] for r in records] == [r['source'] for r in raw]
        assert [r['target'].strip() for r in records] == refs
        predictions = [r['prediction'].strip() for r in records]
        systems[seed, method] = predictions
        values = {}
        for order, metric in metrics.items():
            key = seed, method, order
            stats[key] = np.asarray(metric._extract_corpus_statistics(predictions, [refs]), dtype='float64')
            totals[key] = stats[key].sum(axis=0)
            values[order] = metric.corpus_score(predictions, [refs]).score
            assert abs(metric._compute_score_from_stats(totals[key]).score-values[order]) < 1e-10
        assert abs(values[2]-float(row['chrF++'])) < 1e-8, (seed, method)
        scores.append(dict(seed=seed, method=method, chrf0=values[0], chrfpp=values[2]))
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    assert len(systems) == 12
    summaries, details, checks = [], [], []
    pairs = [('baseline', 'full'), ('lsr', 'full'), ('baseline', 'clrr'), ('clrr', 'full')]
    lookup = {(r['seed'], r['method']): r for r in scores}
    for seed in (42, 43, 44):
        for baseline, challenger in pairs:
            for order, metric in metrics.items():
                a, b = (seed, baseline, order), (seed, challenger, order)
                metric_name = 'chrf0' if order == 0 else 'chrfpp'
                delta = lookup[seed, challenger][metric_name]-lookup[seed, baseline][metric_name]
                leave_out = np.asarray([
                    metric._compute_score_from_stats(totals[b]-stats[b][i]).score
                    - metric._compute_score_from_stats(totals[a]-stats[a][i]).score
                    for i in range(len(refs))])
                low, high = int(leave_out.argmin()), int(leave_out.argmax())
                summaries.append(dict(seed=seed, baseline=baseline, challenger=challenger,
                    metric=metric_name, full_test_delta=delta,
                    leave_one_out_min_delta=float(leave_out[low]), leave_one_out_max_delta=float(leave_out[high]),
                    removed_index_at_min_delta=low, removed_index_at_max_delta=high,
                    sign_changes_count=int((np.sign(leave_out) != np.sign(delta)).sum()), rows=len(refs)))
                for i, value in enumerate(leave_out):
                    details.append(dict(seed=seed, baseline=baseline, challenger=challenger,
                        metric=metric_name, removed_index=i, leave_one_out_delta=float(value),
                        delta_reduction=float(delta-value)))
                for index in sorted({low, high}):
                    short_refs = refs[:index]+refs[index+1:]
                    direct = []
                    for method in (baseline, challenger):
                        predictions = systems[seed, method]
                        direct.append(metric.corpus_score(predictions[:index]+predictions[index+1:], [short_refs]).score)
                    error = abs(direct[1]-direct[0]-leave_out[index])
                    assert error < 1e-10, (seed, baseline, challenger, order, index)
                    checks.append(dict(seed=seed, baseline=baseline, challenger=challenger,
                        word_order=order, removed_index=index, direct_corpus_error=float(error)))
    OUT.mkdir(exist_ok=True)
    write('scores.csv', scores)
    write('influence_summary.csv', summaries)
    write('influence_all_rows.csv', details)
    receipt = dict(scope='Exploratory C character and influence audit; complete primary test retained',
        rows=575, systems=12, comparisons=24, source_predictions=sources,
        reference_path=reference.relative_to(ROOT).as_posix(), reference_sha256=digest(reference),
        seed_scores_sha256=digest(BASE/'seed_scores.csv'), script_sha256=digest(Path(__file__)),
        sacrebleu_version=sacrebleu.__version__, metric_signatures={str(k):str(v.get_signature()) for k,v in metrics.items()},
        direct_api_checks=checks, primary_chrfpp_reproduced=True,
        no_new_significance_tests=True, no_training=True)
    (OUT/'protocol.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(systems=12, comparisons=24, rows=575)), encoding='utf-8')
    for row in summaries:
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
