"""CPU trajectory and surface-error audit of frozen C0/baseline checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def chars(value):
    return ''.join(value.split())


def repeated_fraction(value, width=3):
    value = chars(value)
    grams = [value[i:i + width] for i in range(max(0, len(value) - width + 1))]
    return (len(grams) - len(set(grams))) / max(1, len(grams))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch-metadata', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    design = read(args.predictions / 'statistics_design.json')
    repo = read(args.predictions / 'PLAN.json')['repo']
    metrics = {'bleu': BLEU(tokenize='zh'), 'chrf++': CHRF(word_order=2), 'chrf0': CHRF(word_order=0)}
    trajectories, selected, surfaces, strata, examples, specs = [], [], [], [], [], []
    provenance = {}
    for seed, arms in design['seed_systems'].items():
        frames = {}
        for arm in ('Baseline', 'C0'):
            ident = arms[arm]
            complete = read(args.predictions / 'runs' / ident / 'COMPLETE.json')
            lock = complete['selection']
            prefix = lock['archive_path'].rsplit('/', 1)[0]
            folder = out / 'inputs' / ident
            folder.mkdir(parents=True, exist_ok=True)
            files = {}
            for name in ('final_trainer_state.json', 'selected_trainer_state.json', 'run_metadata.json',
                         'validation_predictions.csv', 'validation_metrics.json'):
                local = folder / name
                if not local.exists():
                    assert args.fetch_metadata, f'Missing {local}; opt in to small metadata download'
                    url = f'https://huggingface.co/{repo}/resolve/{lock["revision"]}/{quote(prefix + "/" + name)}'
                    with urlopen(url, timeout=60) as response:
                        data = response.read(2_000_001)
                    assert len(data) <= 2_000_000, 'Metadata-only size limit'
                    local.write_bytes(data)
                files[name] = hashlib.sha256(local.read_bytes()).hexdigest()
            provenance[ident] = dict(selection=lock, file_sha256=files)
            final, chosen, meta = [read(folder / x) for x in
                                    ('final_trainer_state.json', 'selected_trainer_state.json', 'run_metadata.json')]
            spec = meta['run']
            assert int(spec['seed']) == int(seed)
            specs.append(dict(id=ident, arm=arm, seed=int(seed), spec=spec, selection=lock,
                              validation_predictions=(folder / 'validation_predictions.csv').as_posix()))
            events = [e for e in final['log_history'] if 'eval_bleu' in e]
            assert len(events) == 20, (ident, len(events))
            best = max(events, key=lambda e: e['eval_bleu'])
            selected_step = meta['selected_step']
            assert best['step'] == selected_step, (ident, 'Not best validation BLEU')
            for e in events:
                trajectories.append(dict(id=ident, arm=arm, seed=int(seed), epoch=e['epoch'], step=e['step'],
                    selected=e['step'] == selected_step, **{k: v for k, v in e.items() if k.startswith('eval_')}))
            record = dict(id=ident, arm=arm, seed=int(seed), selected_epoch=meta['selected_epoch'],
                selected_step=selected_step, best_bleu=best['eval_bleu'], final_bleu=events[-1]['eval_bleu'],
                best_chrf_epoch=max(events, key=lambda e: e['eval_chrf++'])['epoch'],
                best_to_final_bleu=events[-1]['eval_bleu'] - best['eval_bleu'],
                final_epoch=final['epoch'], selected_state_step=chosen['global_step'])
            assert chosen['global_step'] == selected_step
            selected.append(record)
            frames[arm] = {}
            for split in ('validation', 'test'):
                path = folder / 'validation_predictions.csv' if split == 'validation' else args.predictions / 'runs' / ident / 'test_predictions.csv'
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                frames[arm][split] = frame
                ref, hyp = frame.target.str.strip().tolist(), frame.prediction.str.strip().tolist()
                scores = {k: m.corpus_score(hyp, [ref]).score for k, m in metrics.items()}
                expected = read(folder / 'validation_metrics.json') if split == 'validation' else complete
                expected = expected.get('scores', expected)
                for key in metrics:
                    if key in expected:
                        assert abs(expected[key] - scores[key]) < 1e-7
                ratios = [len(chars(h)) / max(1, len(chars(t))) for h, t in zip(hyp, ref)]
                surfaces.append(dict(id=ident, arm=arm, seed=int(seed), split=split, rows=len(frame), **scores,
                    corpus_length_ratio=sum(map(len, map(chars, hyp))) / sum(map(len, map(chars, ref))),
                    exact=sum(h == t for h, t in zip(hyp, ref)), empty=sum(not h for h in hyp),
                    short_half_reference=sum(r < .5 for r in ratios), long_twice_reference=sum(r > 2 for r in ratios),
                    mean_repeated_char_trigram_fraction=float(np.mean([repeated_fraction(h) for h in hyp]))))
        for split in ('validation', 'test'):
            base, cand = frames['Baseline'][split], frames['C0'][split]
            assert base[['source', 'target']].equals(cand[['source', 'target']])
            lens = base.target.map(lambda x: len(chars(x)))
            for low, high, label in [(0, 10, 'target_chars_0_10'), (11, 25, 'target_chars_11_25'),
                                     (26, 50, 'target_chars_26_50'), (51, 10**9, 'target_chars_51_plus')]:
                mask = lens.between(low, high)
                if not mask.any():
                    continue
                ref = [base.loc[mask, 'target'].str.strip().tolist()]
                for name, metric in metrics.items():
                    b = metric.corpus_score(base.loc[mask, 'prediction'].str.strip().tolist(), ref).score
                    c = metric.corpus_score(cand.loc[mask, 'prediction'].str.strip().tolist(), ref).score
                    strata.append(dict(seed=int(seed), split=split, stratum=label, rows=int(mask.sum()),
                                       metric=name, baseline=b, candidate=c, delta=c-b))
            for i, (b, c) in enumerate(zip(base.itertuples(), cand.itertuples())):
                # Character metric ranks examples; it does not diagnose semantic omissions.
                sb = metrics['chrf0'].sentence_score(b.prediction.strip(), [b.target.strip()]).score
                sc = metrics['chrf0'].sentence_score(c.prediction.strip(), [c.target.strip()]).score
                examples.append(dict(seed=int(seed), split=split, row=i, source=b.source, target=b.target,
                    baseline=b.prediction, candidate=c.prediction, delta_sentence_chrf0=sc-sb,
                    baseline_chars=len(chars(b.prediction)), candidate_chars=len(chars(c.prediction))))
    for filename, rows in [('trajectory.csv', trajectories), ('selection.csv', selected),
                           ('surface_metrics.csv', surfaces), ('length_strata.csv', strata), ('paired_examples.csv', examples)]:
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    # Balanced examples for all seeds and splits; no test-derived method selection.
    frame = pd.DataFrame(examples)
    sample = []
    for _, group in frame.groupby(['seed', 'split']):
        order = group.sort_values('delta_sentence_chrf0')
        sample.extend(order.head(5).to_dict('records') + order.tail(5).to_dict('records'))
    pd.DataFrame(sample).to_csv(out / 'examples_for_review.csv', index=False)
    plan = dict(status='CPU_ANALYZED', runs=specs, provenance=provenance,
                validation_only_for_method_decisions=True, optimizer_steps=0,
                caution='Surface proxies and reference-based scores cannot identify semantic omissions or a causal mechanism.')
    (out / 'PLAN.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# C0 seed diagnostics', '', '## Material Passport', '',
             'CPU replay of six frozen systems. Post-hoc exploratory diagnostics; no new training or test-based selection.', '',
             '| Seed | Arm | Selected epoch | Best val BLEU | Final val BLEU | Best chrF++ epoch |',
             '|---|---|---:|---:|---:|---:|']
    for x in selected:
        lines.append(f"| {x['seed']} | {x['arm']} | {x['selected_epoch']:.0f} | {x['best_bleu']:.4f} | {x['final_bleu']:.4f} | {x['best_chrf_epoch']:.0f} |")
    lines += ['', '| Seed | Split | Delta BLEU | Delta chrF0 | Baseline short/long | C0 short/long |',
              '|---|---|---:|---:|---|---|']
    for seed in (42, 43, 44):
        for split in ('validation', 'test'):
            b, c = [next(x for x in surfaces if x['seed']==seed and x['split']==split and x['arm']==arm) for arm in ('Baseline','C0')]
            lines.append(f"| {seed} | {split} | {c['bleu']-b['bleu']:+.4f} | {c['chrf0']-b['chrf0']:+.4f} | {b['short_half_reference']}/{b['long_twice_reference']} | {c['short_half_reference']}/{c['long_twice_reference']} |")
    lines += ['', 'Short = less than half reference character length; long = more than twice. These are surface flags, not verified omissions or hallucinations.',
              'Length strata are descriptive corpus subsets; no new p-value family. All examples retained, balanced tails provided for manual review.',
              'A validation/test direction reversal does not establish overfitting to training or identify why a seed fails. GPU interventions are frozen-model sensitivity, not retrained ablation.',
              'Statistical safeguards: all three seeds retained; no pooling of sentences as seeds; no selective significance; test is previously exposed; no equivalence or causal claim from nonsignificance.']
    (out / 'CPU_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
