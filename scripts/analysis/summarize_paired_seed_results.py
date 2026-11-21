"""Summarize fixed paired seed contrasts without treating sentences as seeds."""
import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path


def seed_contrasts(scores, mapping):
    rows = []
    for control in ('LearnedDonor', 'Baseline', 'M0', 'Donor', 'Local'):
        for metric in ('BLEU', 'chrF++', 'chrF0'):
            deltas = [scores[ids['C0']][metric] - scores[ids[control]][metric]
                      for ids in mapping.values()]
            observed = abs(statistics.mean(deltas))
            null = [abs(statistics.mean(s * d for s, d in zip(signs, deltas)))
                    for signs in itertools.product((-1, 1), repeat=len(deltas))]
            rows.append(dict(control=control, candidate='C0', metric=metric,
                deltas=deltas, mean=statistics.mean(deltas), sd=statistics.stdev(deltas),
                positive_seeds=sum(d > 0 for d in deltas),
                p_seed_exact=sum(v >= observed - 1e-12 for v in null) / len(null)))
    running = 0.
    for rank, index in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]['p_seed_exact'])):
        running = max(running, min(1., (len(rows) - rank) * rows[index]['p_seed_exact']))
        rows[index]['p_seed_holm'] = running
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', required=True)
    parser.add_argument('--design', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation-statistics')
    parser.add_argument('--predictions-dir', help='Optional per-system CSV diagnostics; never changes metrics')
    args = parser.parse_args()
    report = json.loads(Path(args.statistics).read_text(encoding='utf-8'))
    design = json.loads(Path(args.design).read_text(encoding='utf-8'))
    mapping = design['seed_systems']
    assert list(mapping) == ['42', '43', '44'], 'This development block froze three seeds'
    expected = {ident for ids in mapping.values() for ident in ids.values()}
    assert set(report['scores']) == expected and report['holm_family_size'] == 45
    scores = report['scores']
    comparisons = seed_contrasts(scores, mapping)
    means = {arm: {metric: dict(mean=statistics.mean(scores[ids[arm]][metric] for ids in mapping.values()),
                               sd=statistics.stdev(scores[ids[arm]][metric] for ids in mapping.values()))
                    for metric in ('BLEU', 'chrF++', 'chrF0')}
             for arm in next(iter(mapping.values()))}
    interactions = {metric: [scores[ids['C0']][metric] - scores[ids['Donor']][metric]
                            - scores[ids['Local']][metric] + scores[ids['M0']][metric]
                            for ids in mapping.values()]
                    for metric in ('BLEU', 'chrF++', 'chrF0')}
    gates = None
    rank_diagnostics = {}
    if args.validation_statistics:
        val = json.loads(Path(args.validation_statistics).read_text(encoding='utf-8'))
        assert val['split'] == 'validation'
        contrasts = seed_contrasts(val['scores'], mapping)
        baseline = {r['metric']: r for r in contrasts if r['control'] == 'Baseline'}
        controls = [r for r in contrasts if r['control'] != 'Baseline' and r['metric'] == 'BLEU']
        gates = dict(mean_C0_minus_baseline_BLEU_at_least_030=baseline['BLEU']['mean'] >= .30,
            mean_C0_minus_baseline_chrf0_nonnegative=baseline['chrF0']['mean'] >= 0,
            each_control_mean_BLEU_positive=all(r['mean'] > 0 for r in controls),
            each_control_at_least_two_positive_seeds=all(r['positive_seeds'] >= 2 for r in controls),
            validation_contrasts=contrasts)
        if report['split'] == 'test':
            def ranks(values):
                return [statistics.mean(i + 1 for i, v in enumerate(sorted(values)) if v == value)
                        for value in values]
            for seed, ids in mapping.items():
                vr = ranks([val['scores'][ident]['chrF++'] for ident in ids.values()])
                tr = ranks([scores[ident]['BLEU'] for ident in ids.values()])
                rank_diagnostics[seed] = statistics.correlation(vr, tr)
    prediction_diagnostics = []
    if args.predictions_dir:
        from sacrebleu.metrics import CHRF
        word_metric = CHRF(word_order=2)
        for seed, ids in mapping.items():
            for arm, ident in ids.items():
                with (Path(args.predictions_dir) / 'runs' / ident / f"{report['split']}_predictions.csv").open(encoding='utf-8', newline='') as f:
                    frame = list(csv.DictReader(f))
                assert len(frame) == report['rows']
                predictions = [r['prediction'].strip() for r in frame]
                targets = [r['target'].strip() for r in frame]
                chars = lambda text: len(''.join(text.split()))
                corpus_stats = [sum(values) for values in zip(*word_metric._extract_corpus_statistics(predictions, [targets]))]
                prediction_diagnostics.append(dict(seed=seed, arm=arm, id=ident,
                    exact_matches=sum(p == t for p, t in zip(predictions, targets)),
                    empty_outputs=sum(not p for p in predictions),
                    literal_unk_outputs=sum('<unk>' in p for p in predictions),
                    character_length_ratio=sum(map(chars, predictions)) / sum(map(chars, targets)),
                    word_unigram_hyp_reference_matches=corpus_stats[18:21],
                    word_bigram_hyp_reference_matches=corpus_stats[21:24]))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    summary = dict(split=report['split'], rows=report['rows'], means=means,
        seed_comparisons=comparisons, interactions_descriptive=interactions,
        sentence_holm_family_size=45, sentence_significant_count=sum(r['significant_holm'] for r in report['tests']),
        seed_holm_family_size=15, validation_gates=gates, scope=design['scope'],
        selection_rank_descriptive=rank_diagnostics, prediction_diagnostics=prediction_diagnostics,
        warning='Three development seeds; exact two-sided sign-flip minimum p=.25. Seed exchangeability is an assumption. Sentence bootstrap excludes training variance. Neither nonsignificance nor a narrow observed gap establishes equivalence.')
    (out / 'SEED_SUMMARY.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out / 'seed_scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'arm', 'id', 'BLEU', 'chrF++', 'chrF0']); writer.writeheader()
        for seed, ids in mapping.items():
            for arm, ident in ids.items(): writer.writerow(dict(seed=seed, arm=arm, id=ident, **scores[ident]))
    lines = ['# Paired seed summary', '', '## Material Passport', '',
        f"VERIFIED: metric replay and native sentence-bootstrap p-values, {report['rows']} {report['split']} sentences. ANALYZED: seed effects and practical gates.",
        '', design['scope'], '', '| Arm | BLEU mean ± SD | chrF++ mean ± SD | chrF0 mean ± SD |',
        '|---|---:|---:|---:|']
    for arm, metrics in means.items():
        lines.append('| ' + arm + ' | ' + ' | '.join(f"{v['mean']:.4f} ± {v['sd']:.4f}" for v in metrics.values()) + ' |')
    lines += ['', '## Paired training seed effects', '',
        'Deltas are C0 minus control in seed order 42,43,44. Holm15 below concerns seed-level tests; the separate sentence-level family is Holm45.', '',
        '| Control | Metric | Seed deltas | Mean delta | SD | Wins | Exact seed p | Seed Holm15 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['control']} | {r['metric']} | {', '.join(f'{v:+.4f}' for v in r['deltas'])} | {r['mean']:+.4f} | {r['sd']:.4f} | {r['positive_seeds']}/3 | {r['p_seed_exact']:.3f} | {r['p_seed_holm']:.3f} |")
    lines += ['', '## Interaction (descriptive)', '', 'C0 − Donor − Local + M0; no interaction p-value added to Holm45.']
    for metric, values in interactions.items():
        lines.append(f"- {metric}: {values}; mean={statistics.mean(values):+.4f}, SD={statistics.stdev(values):.4f}.")
    if gates:
        lines += ['', '## Previously frozen validation gates', '', 'These gates use validation; test does not replace validation selection.']
        lines.extend(f'- {key}: {value}' for key, value in gates.items() if key != 'validation_contrasts')
    if rank_diagnostics:
        lines += ['', '## Validation/test rank agreement (descriptive)', '',
            'Within each seed, Spearman rank correlation across six arms: validation chrF++ versus test BLEU. Six arms per seed, no added significance test or mechanistic inference.']
        lines.extend(f'- Seed {seed}: {value:+.4f}.' for seed, value in rank_diagnostics.items())
    if prediction_diagnostics:
        lines += ['', '## Prediction diagnostics (descriptive)', '',
            '| Seed | Arm | Exact matches | Empty | Literal unk | Character length ratio | Word-bigram matches / reference count |',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in prediction_diagnostics:
            w = row['word_bigram_hyp_reference_matches']
            lines.append(f"| {row['seed']} | {row['arm']} | {row['exact_matches']} | {row['empty_outputs']} | {row['literal_unk_outputs']} | {row['character_length_ratio']:.4f} | {w[2]}/{w[1]} |")
        lines += ['', 'Word statistics are native SacreBLEU CHRF(word_order=2) corpus counts, not additional significance tests. Sparse word-bigram support can make chrF++ sensitive to a single matching word bigram; chrF0 omits this component.']
    lines += ['', f"Sentence bootstrap Holm45 significant: {summary['sentence_significant_count']}/45.", '', summary['warning'], '',
        '## Statistical fallacy scan (11/11 checked)', '',
        '1. Simpson: show all seeds and mean; mixed directions remain visible.',
        '2. Ecological: corpus scores do not imply every sentence improves.',
        '3. Berkson: validation-selected checkpoints; selected development candidates limit generalization.',
        '4. Collider: no post-test filtering of checkpoints or examples.',
        '5. Base rates: diagnostic prevalence reasoning not applicable.',
        '6. Regression to mean: C0 chosen after pilots; seed42 observed before this replication.',
        '7. Survivorship: all frozen seeds/arms reported, including negative deltas.',
        '8. Look elsewhere: report full Holm45 family and separate Holm15 seed tests; historic families unchanged.',
        '9. Forking paths: Amis test has prior exposure; exploratory characterization, not independent confirmation.',
        '10. Causality: controlled recipe comparisons support measured differences, not unique mechanism or novelty.',
        '11. Reverse causality: training precedes evaluation; no reverse direction claimed.', '',
        'Percentile sentence CIs and SacreBLEU centered absolute-bootstrap p-values are different procedures and need not invert each other. CIs are marginal, not simultaneous.']
    (out / 'SEED_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'summary': str(out / 'SEED_SUMMARY.md'), 'significant_sentence_tests': summary['sentence_significant_count'], 'validation_gates': {k:v for k,v in (gates or {}).items() if k != 'validation_contrasts'}}, indent=2))


if __name__ == '__main__':
    main()
