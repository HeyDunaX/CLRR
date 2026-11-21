"""Verify and summarize a frozen diagnostic, with sentences as resampling units."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sacrebleu.metrics import BLEU, CHRF


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correlation(x, y):
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranked_association(frame, predictor, outcome):
    columns = [predictor, outcome, 'entropy', 'prefix_length', 'target_characters']
    matrix = frame[columns].to_numpy(float)
    ranks = np.column_stack([rankdata(matrix[:, i]) for i in range(matrix.shape[1])])
    def compute(data):
        if len(data) < 9: return None, None
        x, y = data[:, 0], data[:, 1]
        controls = np.column_stack([np.ones(len(data)), data[:, 2:]])
        rx = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
        ry = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
        return correlation(x, y), correlation(rx, ry)
    rho, partial = compute(ranks)
    rng = np.random.default_rng(20261115)
    samples = [compute(ranks[rng.integers(0, len(ranks), len(ranks))]) for _ in range(2000)]
    result = dict(rows=len(frame), spearman=rho, partial_rank=partial,
        controls=columns[2:], bootstrap_samples=2000,
        caveat='Descriptive bootstrap of sentence ranks after averaging observed events; not a pre-registered test or causal estimate')
    for i, name in enumerate(('spearman_ci95', 'partial_rank_ci95')):
        values = [s[i] for s in samples if s[i] is not None]
        result[name] = np.quantile(values, [.025, .975]).tolist() if values else None
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    plan = json.loads((out / 'PLAN.json').read_text())
    complete = json.loads((out / 'COMPLETE.json').read_text())
    assert complete['passed'] and complete['plan_sha256'] == digest(out / 'PLAN.json')
    assert complete['optimizer_steps'] == 0 and not complete['test_used']
    assert json.loads((out / 'CPU_GATE.json').read_text())['passed']
    raw = pd.read_csv(Path('data_processed') / plan['dataset'] / 'validation.csv', keep_default_na=False)
    assert digest(Path('data_processed') / plan['dataset'] / 'validation.csv') == plan['data']['validation']['sha256']
    metrics = {'chrf++': CHRF(word_order=2), 'chrf0': CHRF(word_order=0),
        'bleu': BLEU(tokenize='zh', effective_order=True)}
    summaries = {}
    for checkpoint in plan['checkpoints']:
        ident = checkpoint['id']; folder = out / ident
        result = json.loads((folder / 'RESULT.json').read_text())
        assert result['checkpoint'] == checkpoint and result['plan_sha256'] == digest(out / 'PLAN.json')
        assert result['csv_sha256'] == digest(folder / 'decisions.csv')
        gate = json.loads((folder / 'GPU_GATE.json').read_text())
        assert gate['passed'] and gate['no_gradients'] and gate['native_replay']
        frame = pd.read_csv(folder / 'decisions.csv', keep_default_na=False)
        assert len(frame) == len(plan['validation_rows']) * len(plan['dropout_seeds'])
        assert not frame.duplicated(['row', 'pair']).any()
        assert sorted(frame.row.unique()) == sorted(plan['validation_rows'])
        for _, item in frame.iterrows():
            reference = raw.iloc[item.row]
            assert item.source == reference.source and item.target == reference.target
            if item.has_lexical_disagreement:
                assert float(item.action_a) != float(item.action_b)
                assert float(item.prefix_length) == float(item.position)+1
                for name, scorer in metrics.items():
                    for label in ('a', 'b'):
                        replay = scorer.sentence_score(item['branch_'+label], [item.target]).score
                        assert abs(replay-float(item[label+'_'+name])) < 1e-9
                    gap = abs(float(item['a_'+name])-float(item['b_'+name]))
                    assert abs(gap-float(item['abs_gap_'+name])) < 1e-9
        events = frame[frame.has_lexical_disagreement].copy()
        for c in ('same_decoded_branches', 'branch_a_at_length_limit', 'branch_b_at_length_limit'):
            assert events[c].isin([True, False, 'True', 'False']).all()
            events[c] = events[c].map(lambda x: x is True or x == 'True')
        numeric = ['entropy', 'prefix_length', 'target_characters', 'symmetric_kl', 'js',
            'abs_gap_chrf0', 'abs_gap_chrf++', 'abs_gap_bleu']
        for c in numeric + [label+'_'+metric for label in ('a', 'b') for metric in metrics]:
            events[c] = pd.to_numeric(events[c])
        sentence = events.groupby('row')[numeric].mean()
        sentence.to_csv(folder / 'sentence_event_means.csv')
        summary = dict(sentences=len(plan['validation_rows']), pairs=len(plan['dropout_seeds']),
            events=len(events), sentences_with_events=len(sentence),
            fraction_pairs_with_disagreement=float(frame.has_lexical_disagreement.mean()),
            no_event_not_assigned_zero_gap=True, conditional_event_means=sentence.mean().to_dict(),
            same_decoded_branches=int(events.same_decoded_branches.sum()),
            metric_direction_disagreement=int((((events['a_chrf0']-events['b_chrf0']) *
                (events['a_bleu']-events['b_bleu'])) < 0).sum()),
            at_length_limit=int(events.branch_a_at_length_limit.sum()+events.branch_b_at_length_limit.sum()),
            association={p: ranked_association(sentence, p, 'abs_gap_chrf0')
                for p in ('symmetric_kl', 'js')})
        summaries[ident] = summary
    report = dict(summaries=summaries, optimizer_steps=0, test_used=False,
        scientific_admission='NO_AUTOMATIC_TRAINING_OR_NOVELTY_ADMISSION',
        limitations=['32 repeatedly used validation sentences; one training seed',
            'First lexical disagreement selection depends on checkpoint and dropout pair; no matched-position comparison between models',
            'Single-reference surface metrics cannot distinguish valid paraphrases from errors',
            'Greedy rollout costs depend on continuation policy; not optimal action values',
            'Full-prefix train-mode forwards resample masks across token computations, not coherent cached stochastic rollouts',
            'Conditional event associations omit no-event pairs; sentence averages have differing event coverage',
            'No intervention on training, no diagnosis that exposure bias causes prior failures'])
    (out / 'ANALYSIS.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Frozen decision-branch diagnostic', '',
        'Exploratory; no training or test. Single-reference scores are proxies, not semantic labels.', '',
        '| Checkpoint | Events / pairs | Sentences with events | Mean abs chrF0 gap* | KL–gap partial rank (95% CI) |',
        '|---|---:|---:|---:|---|']
    for name, s in summaries.items():
        a = s['association']['symmetric_kl']
        lines.append(f"| {name} | {s['events']}/{s['sentences']*s['pairs']} | {s['sentences_with_events']} | {s['conditional_event_means']['abs_gap_chrf0']:.4f} | {a['partial_rank']} {a['partial_rank_ci95']} |")
    lines += ['', '*Mean across sentence event means, conditional on an event; not a corpus metric improvement.', '', '## Limits', '']
    lines += ['- '+x for x in report['limitations']]
    lines += ['', '## Decision', '', 'No method or training admitted by this diagnostic alone.']
    (out / 'ANALYSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
