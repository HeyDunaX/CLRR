"""Build descriptive mechanism tables after independent artifact verification."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs_rebuttal/amis_followup_20261004_v4'


def main():
    audit = json.loads((BASE / 'independent_audit.json').read_text())
    assert audit['status'] == 'VERIFIED'
    families = defaultdict(list)
    by_family_seed = {}
    for path in (BASE / 'probes/runs').glob('*/probe_results.json'):
        record = json.loads(path.read_text())
        layers = record['layer_spectra_and_alignment']
        grads = record['average_gradient_by_layer_group']['all_parameters']
        item = {'seed': record['seed'], 'block12_source_rank': layers['layer_12']['source_effective_rank_singular'],
                'final_source_rank': layers['layer_final']['source_effective_rank_singular'],
                'final_energy_rank': layers['layer_final']['source_effective_rank_cov'],
                'final_alignment': layers['layer_final']['paired_cosine_mean'],
                'gradient_cosine': grads['mean_gradient_cosine'], 'weighted_lsr_ce_ratio': grads['mean_lsr_to_ce_ratio']}
        families[record['family']].append(item)
        by_family_seed[(record['family'], record['seed'])] = item
    means = {}
    for family, items in families.items():
        means[family] = {key: {'mean': float(np.mean([m[key] for m in items])),
                               'sample_std': float(np.std([m[key] for m in items], ddof=1))}
                         for key in items[0] if key != 'seed'}
    comparisons = {}
    for left, right in [('clrr_only_detached', 'baseline'), ('original_full_detached', 'matched_lsr'),
                        ('full_detached_l003', 'tuned_lsr'), ('original_full_detached', 'tuned_lsr')]:
        comparisons[left + '_minus_' + right] = [
            {'seed': seed, **{key: by_family_seed[left, seed][key] - by_family_seed[right, seed][key]
                            for key in ['block12_source_rank', 'final_source_rank', 'final_energy_rank', 'final_alignment']}}
            for seed in [42, 43, 44]]
    residuals = {}
    for path in (BASE / 'probes/runs').glob('*/residuals.csv'):
        if path.stat().st_size <= 5:
            continue
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        residuals[path.parent.name] = {phase: {
            'rows': sum(r['phase'] == phase for r in rows),
            'ratio_mean': float(np.mean([float(r['residual_to_current_rms']) for r in rows if r['phase'] == phase])),
            'ratio_max': max(float(r['residual_to_current_rms']) for r in rows if r['phase'] == phase)}
            for phase in ['source', 'target']}
    negative = defaultdict(lambda: [0, 0])
    for path in (BASE / 'probes/runs').glob('*/probe_results.json'):
        m = json.loads(path.read_text())
        for b in m['gradient_batches']:
            for name, g in b['groups'].items():
                if name.startswith('encoder_layer_') and g['gradient_cosine'] is not None:
                    negative[m['family']][0] += g['gradient_cosine'] < 0
                    negative[m['family']][1] += 1
    routes = audit['route_checks']
    report = {'family_means': means, 'per_seed': dict(families), 'paired_rank_deltas': comparisons,
              'residuals': residuals, 'negative_layer_batch_cosines': dict(negative),
              'route_delta_bleu_mean': float(np.mean([r['delta_bleu'] for r in routes])),
              'route_delta_bleu_sample_std': float(np.std([r['delta_bleu'] for r in routes], ddof=1)),
              'route_changed_mean_percent': float(np.mean([r['changed'] for r in routes]) / 576 * 100),
              'limits': ['Validation best checkpoints only; no new test quality measurement.',
                         'On/off uses routing-trained weights, not independently trained matched controls.',
                         'Pooled endpoint rank and alignment do not establish causal anti-collapse or improved translation.',
                         'Gradient norms do not equal Adam update magnitudes; endpoint means do not rule out temporal conflicts.',
                         'Counterfactual lambda=0.1 for baseline/CLRR-only is not their training objective.']}
    (BASE / 'descriptive_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'family_means': means, 'negative_layer_batch_cosines': dict(negative),
                      'route_delta_bleu_mean': report['route_delta_bleu_mean'], 'residual_max': max(x[p]['ratio_max'] for x in residuals.values() for p in x)}, indent=2))


if __name__ == '__main__':
    main()
