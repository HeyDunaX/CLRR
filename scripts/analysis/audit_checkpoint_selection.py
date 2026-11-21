"""Audit existing validation histories and selected checkpoints; never load weights."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSON snapshots keyed by run ID')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    snapshots = json.loads(args.input.read_text(encoding='utf-8'))
    records = {}
    for ident, snapshot in snapshots.items():
        history = sorted(snapshot['evaluations'], key=lambda row: row['step'])
        selected = snapshot['selected']
        best_step = selected.get('best_global_step') or selected['global_step']
        chosen = next(row for row in history if row['step'] == best_step)
        chrf_best = max(history, key=lambda row: row['eval_chrf++'])
        bleu_best = max(history, key=lambda row: row['eval_bleu'])
        scores = snapshot['validation_metrics']['scores']
        assert chrf_best['step'] == best_step, (ident, 'Selected step is not earliest chrF++ maximum')
        assert abs(chosen['eval_chrf++'] - selected['best_metric']) < 1e-10
        assert abs(chosen['eval_chrf++'] - scores['chrf++']) < 1e-10
        assert abs(chosen['eval_bleu'] - scores['bleu']) < 1e-10
        completed = snapshot['final']
        assert completed['global_step'] >= best_step
        correlation = pd.Series([r['eval_bleu'] for r in history]).rank().corr(
            pd.Series([r['eval_chrf++'] for r in history]).rank())
        records[ident] = {
            'method': snapshot['metadata']['run']['method'],
            'selected_epoch': chosen['epoch'], 'selected_step': best_step,
            'selected_bleu': scores['bleu'], 'selected_chrfpp': scores['chrf++'],
            'selected_chrf0': scores['chrf0'],
            'max_validation_bleu_epoch': bleu_best['epoch'],
            'max_validation_bleu_step': bleu_best['step'],
            'max_validation_bleu': bleu_best['eval_bleu'],
            'bleu_gap_to_selected': bleu_best['eval_bleu'] - scores['bleu'],
            'chrfpp_at_bleu_max': bleu_best['eval_chrf++'],
            'completed_epoch': completed['epoch'], 'completed_step': completed['global_step'],
            'evaluated_epochs': len(history), 'bleu_chrfpp_rank_correlation': float(correlation),
            'chrf0_available_across_epochs': all('eval_chrf0' in row for row in history),
            'selection_correct': True,
        }
        if all('eval_chrf0' in row for row in history):
            chrf0_best = max(history, key=lambda row: row['eval_chrf0'])
            assert abs(chosen['eval_chrf0'] - scores['chrf0']) < 1e-10
            records[ident].update(
                max_validation_chrf0_epoch=chrf0_best['epoch'],
                max_validation_chrf0_step=chrf0_best['step'],
                max_validation_chrf0=chrf0_best['eval_chrf0'],
                chrf0_gap_to_selected=chrf0_best['eval_chrf0'] - scores['chrf0'])
    report = {'status': 'SELECTION_AUDIT_PASS', 'runs': records,
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'limits': 'Validation history only. No checkpoint reselection or test evaluation. '
                        'Historical chrF0 cannot be reconstructed if it was not logged.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'SELECTION_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame.from_dict(records, orient='index').to_csv(args.output / 'selection_audit.csv', index_label='id')
    lines = ['# Validation checkpoint selection audit', '',
             '| Run | Selected epoch | Selected BLEU | Epoch max val BLEU | Max val BLEU | Gap | Completed epoch |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for ident, row in records.items():
        lines.append(f"| {ident} | {row['selected_epoch']:g} | {row['selected_bleu']:.4f} | "
                     f"{row['max_validation_bleu_epoch']:g} | {row['max_validation_bleu']:.4f} | "
                     f"{row['bleu_gap_to_selected']:+.4f} | {row['completed_epoch']:g} |")
    lines += ['', report['limits'], '', 'All selected steps match the earliest maximum validation chrF++.']
    (args.output / 'SELECTION_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
