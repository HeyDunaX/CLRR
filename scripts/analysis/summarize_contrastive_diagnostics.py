"""Summarize frozen validation probes; never load weights or change training results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def audit_verified_cohort(root):
    """Replay small, previously measured artifacts; never download model weights."""
    receipt = json.loads((root / 'DIAGNOSTICS_VERIFICATION_LOCAL.json').read_text(encoding='utf-8'))
    public = receipt['public_artifacts_verified']
    if receipt['status'] != 'PASS' or not public['verified']:
        raise ValueError('An independently verified input receipt is required')
    rows, sources, batches = [], {}, None
    for family in ('baseline', 'nce', 'p2', 'p3'):
        for seed in (42, 43, 44):
            ident = f'{family}-s{seed}'
            path = root / 'diagnostics' / ident / 'RESULT.json'
            raw = path.read_bytes()
            matches = [v for k, v in public['files'].items() if k.endswith(f'/diagnostics/{ident}/RESULT.json')]
            digest = hashlib.sha256(raw).hexdigest()
            if len(matches) != 1 or matches[0]['sha256'] != digest or matches[0]['size'] != len(raw):
                raise ValueError(f'Unverified artifact: {ident}')
            result = json.loads(raw)
            if result['optimizer_steps'] != 0 or not result['parameter_versions_unchanged']:
                raise ValueError(f'Probe mutated parameters: {ident}')
            current = [g['indices'] for g in result['gradients']]
            if batches is None:
                batches = current
            if current != batches or len(current) != 4:
                raise ValueError(f'Gradient batches differ: {ident}')
            sources[ident] = {'sha256': digest, 'selection': result['selection'],
                              'method': result['method'], 'actual_weights': result['actual_objective_weights'],
                              'probe_weights': result['gradient_probe_weights'],
                              'counterfactual_probes': result['counterfactual_probes']}
            groups = {'all_parameters': ['all_parameters'],
                      **{name: [f'encoder_layer{i}' for i in range(start, start + 4)]
                         for name, start in [('early_0_3', 0), ('middle_4_7', 4), ('late_8_11', 8)]}}
            values = {group: [] for group in groups}
            for g in result['gradients']:
                if g['actual_wrapper_loss_error'] >= 2e-6:
                    raise ValueError(f'Loss decomposition mismatch: {ident}')
                for v in g['pairs']['CE_LSR'].values():
                    a, b, dot = v['ce_norm_squared'], v['lsr_norm_squared'], v['dot_product']
                    if not all(math.isfinite(x) for x in (a, b, dot)) or min(a, b) < 0:
                        raise ValueError(f'Invalid gradient scalar: {ident}')
                    for key, expected in [('ce_gradient_norm', math.sqrt(a)),
                                          ('lsr_gradient_norm', math.sqrt(b))]:
                        if not math.isclose(v[key], expected, rel_tol=1e-7, abs_tol=1e-10):
                            raise ValueError(f'Norm replay mismatch: {ident}')
                    if a and b and not math.isclose(v['gradient_cosine'], dot / math.sqrt(a*b),
                                                   rel_tol=1e-7, abs_tol=1e-10):
                        raise ValueError(f'Cosine replay mismatch: {ident}')
                for group, names in groups.items():
                    parts = [g['pairs']['CE_LSR'][name] for name in names]
                    a = sum(v['ce_norm_squared'] for v in parts)
                    b = sum(v['lsr_norm_squared'] for v in parts)
                    dot = sum(v['dot_product'] for v in parts)
                    if not a or not b:
                        raise ValueError(f'Undefined combined cosine: {ident}/{group}')
                    values[group].append((dot / math.sqrt(a*b), math.sqrt(b/a)))
            for group, vals in values.items():
                rows.append({'id': ident, 'seed': seed, 'group': group,
                             'actual_lsr_weight': result['actual_objective_weights']['LSR'],
                             'lsr_probe_weight': result['gradient_probe_weights']['LSR'],
                             'mean_cosine': mean(v[0] for v in vals),
                             'negative_batches': sum(v[0] < 0 for v in vals),
                             'batches': len(vals), 'mean_norm_ratio': mean(v[1] for v in vals)})
    out = root / 'lsr_compatibility'
    out.mkdir(exist_ok=True)
    report = {'status': 'PASS', 'models': len(sources), 'optimizer_steps': 0,
              'upstream_repo': public['repo'], 'upstream_revision': public['revision'],
              'gradient_batches': batches, 'sources': sources, 'rows': rows,
              'limitations': ['Selected checkpoints only; four repeated validation batches.',
                              'Norm ratios are instantaneous, not accumulated Adam updates.',
                              'Zero actual LSR weight means a counterfactual LSR probe.',
                              'Region cosine combines disjoint gradient coordinates before averaging batches.',
                              'No training-time causal claim or independent-seed significance test.']}
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with (out / 'SUMMARY.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# CE–LSR compatibility replay', '',
             'Hash/norm/cosine/loss-decomposition and same-batch checks PASS. No new GPU measurement.', '',
             '| Checkpoint | Actual LSR | Mean cosine | Negative batches | Mean weighted LSR/CE norm |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['group'] == 'all_parameters':
            lines.append(f"| {row['id']} | {row['actual_lsr_weight']} | {row['mean_cosine']:.4f} | "
                         f"{row['negative_batches']}/{row['batches']} | {row['mean_norm_ratio']:.5f} |")
    lines += ['', '## Interpretation', '',
              'These selected-checkpoint probes do not show pervasive negative CE–LSR gradients. '
              'They do not justify conflict surgery or increasing lambda solely from a small norm ratio.', '',
              *[f'- {s}' for s in report['limitations']], '',
              f"Input revision: `{public['revision']}`. Per-file hashes and checkpoint provenance: RESULT.json."]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'models': len(sources), 'output': str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--verified-cohort', action='store_true',
                        help='Replay the 12-model verified cohort without unpacking remote_artifacts.json')
    args = parser.parse_args()
    root = args.input.resolve()
    if args.verified_cohort:
        audit_verified_cohort(root)
        return
    bundle = json.loads((root / 'remote_artifacts.json').read_text(encoding='utf-8-sig'))
    for name, artifact in bundle.items():
        destination = (root / name).resolve()
        assert destination.is_relative_to(root)
        content = artifact['content'].encode('utf-8')
        assert hashlib.sha256(content).hexdigest() == artifact['sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    plan, complete = read('PLAN.json'), read('COMPLETE.json')
    assert complete['status'] == 'COMPLETE' and complete['optimizer_steps'] == 0
    assert complete['worker_sha256'] == plan['worker_sha256']
    summary = {}
    for ident in plan['ids']:
        result = read(f'{ident}/RESULT.json')
        assert result['parameter_versions_unchanged'] and result['optimizer_steps'] == 0
        assert result['selection'] == plan['selection'][ident]
        assert len(result['gradients']) == len(plan['gradient_batches'])
        assert [g['indices'] for g in result['gradients']] == plan['gradient_batches']
        assert max(g['actual_wrapper_loss_error'] for g in result['gradients']) < 2e-6
        geometry = result['geometry']
        item = {
            'method': result['method'], 'actual_weights': result['actual_objective_weights'],
            'counterfactual_probes': result['counterfactual_probes'],
            'retrieval': geometry['retrieval'],
            'rank': {s: {k: v for k, v in geometry[s].items() if k != 'singular_values'}
                     for s in ('source', 'target')},
            'losses_mean': {k: mean(g[k] for g in result['gradients'])
                            for k in ('ce', 'lsr', 'infonce')},
            'gradient_pairs': {},
            'residual_ratio_mean_by_layer': result['residual_ratio_mean_by_layer'],
            'peak_allocated_MiB': result['peak_allocated_MiB'],
        }
        for pair in ('CE_LSR', 'CE_InfoNCE', 'LSR_InfoNCE'):
            item['gradient_pairs'][pair] = {}
            for group in result['gradients'][0]['pairs'][pair]:
                values = [g['pairs'][pair][group] for g in result['gradients']]
                item['gradient_pairs'][pair][group] = {
                    'first_norm_mean': mean(v['ce_gradient_norm'] for v in values),
                    'second_norm_mean': mean(v['lsr_gradient_norm'] for v in values),
                    'second_to_first_ratio_mean': mean(v['lsr_to_ce_ratio'] for v in values)
                                                 if all(v['lsr_to_ce_ratio'] is not None for v in values) else None,
                    'cosine_mean': mean(v['gradient_cosine'] for v in values)
                                   if all(v['gradient_cosine'] is not None for v in values) else None,
                    'negative_cosine_batches': sum(v['gradient_cosine'] is not None and
                                                  v['gradient_cosine'] < 0 for v in values),
                }
        summary[ident] = item
    (root / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    receipt = {'status': 'PASS', 'artifacts': {k: v['sha256'] for k, v in bundle.items()},
               'ids': plan['ids'], 'validation_rows': len(plan['indices']),
               'gradient_batches': len(plan['gradient_batches']), 'optimizer_steps': 0,
               'worker_sha256': plan['worker_sha256'], 'temporary_weights_removed': complete['temporary_weights_removed']}
    (root / 'VERIFICATION.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    for ident, item in summary.items():
        print(ident, json.dumps({k: v for k, v in item.items()
                                if k != 'gradient_pairs' and k != 'residual_ratio_mean_by_layer'}))
        for pair in item['gradient_pairs']:
            print(pair, json.dumps(item['gradient_pairs'][pair]['all_parameters']))
        print('residual source', {k: v for k, v in item['residual_ratio_mean_by_layer'].items()
                                  if k.startswith('source_')})


if __name__ == '__main__':
    main()
