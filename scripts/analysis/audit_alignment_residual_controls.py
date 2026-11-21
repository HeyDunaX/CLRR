"""Prepare hash-verified, small frozen artifacts for the CE-controls analysis."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--probes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fetch-missing', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'PASS', 'runs': {}, 'probes': {}}
    baseline = read(args.pilot / 'runs/R1/run_metadata.json')
    baseline_args = None
    protocol_keys = ('epochs', 'patience', 'warmup_ratio', 'optim', 'scheduler',
                     'weight_decay', 'bf16', 'fp16', 'generation_precision',
                     'gradient_checkpointing', 'loss_reduction', 'selection')
    run_keys = ('model_revision', 'seed', 'data_seed', 'learning_rate',
                'microbatch', 'accumulation', 'max_source_length',
                'max_target_length', 'src_lang', 'tgt_lang')
    for ident in ('R1', 'AR_C0', 'CE_DONOR', 'CE_LOCAL'):
        source = (args.controls if ident.startswith('CE_') else args.pilot) / 'runs' / ident
        receipt = read(source / 'VAL_COMPLETE.json')
        assert receipt['status'] == 'VALIDATION_COMPLETE_VERIFIED'
        assert receipt['backup']['verified']
        target = args.output / 'inputs/runs' / ident
        target.mkdir(parents=True, exist_ok=True)
        (target / 'VAL_COMPLETE.json').write_bytes((source / 'VAL_COMPLETE.json').read_bytes())
        files = {}
        names = ('run_metadata.json', 'validation_predictions.csv', 'validation_metrics.json',
                 'resolved_training_arguments.json', 'selected_trainer_state.json',
                 'final_trainer_state.json')
        for name in names:
            remote, expected = next((k, v) for k, v in receipt['backup']['files'].items()
                                    if k.endswith('/' + name))
            local = source / name
            if not local.exists() and (target / name).exists():
                local = target / name
            if local.exists():
                data = local.read_bytes()
                if sha(data) != expected['sha256']:
                    data = data.replace(b'\r\n', b'\n')
            else:
                assert args.fetch_missing, f'Missing {local}; opt in to small-file fetching'
                assert expected['size'] < 1_000_000, 'Never download weights'
                url = ('https://huggingface.co/' + receipt['backup']['repo'] + '/resolve/'
                       + receipt['backup']['revision'] + '/' + quote(remote))
                with urlopen(url, timeout=60) as response:
                    data = response.read(1_000_001)
            assert len(data) == expected['size'] and sha(data) == expected['sha256'], local
            (target / name).write_bytes(data)
            files[name] = expected
        meta = read(target / 'run_metadata.json')
        for key in protocol_keys:
            assert meta['protocol'][key] == baseline['protocol'][key], (ident, key)
        for key in run_keys:
            assert meta['run'][key] == baseline['run'][key], (ident, key)
        assert meta['data'] == baseline['data'], ident
        actual_args = read(target / 'resolved_training_arguments.json')
        if ident == 'R1':
            baseline_args = actual_args
        assert {k: v for k, v in actual_args.items() if k not in ('output_dir', 'logging_dir')} == {
            k: v for k, v in baseline_args.items() if k not in ('output_dir', 'logging_dir')}, ident
        final = read(target / 'final_trainer_state.json')
        events = [e for e in final['log_history'] if 'eval_chrf++' in e]
        assert events, (ident, 'No validation events')
        best = max(events, key=lambda e: e['eval_chrf++'])
        assert best['step'] == meta['selected_step'], (ident, 'Selection mismatch')
        assert abs(best['eval_chrf++'] - receipt['scores']['chrf++']) < 1e-6
        assert meta['archive_sha256'] == receipt['archive_sha256']
        report['runs'][ident] = {
            'scores': receipt['scores'], 'selected_epoch': meta['selected_epoch'],
            'completed_epoch': meta['completed_epoch'], 'selected_step': meta['selected_step'],
            'archive_sha256': receipt['archive_sha256'], 'revision': receipt['backup']['revision'],
            'files': files,
        }
    for ident in ('AR_C0', 'AR_C1', 'AR_C2'):
        probe = read(args.probes / ident / 'RESULT.json')
        original = read(args.pilot / 'runs' / ident / 'VAL_COMPLETE.json')
        assert probe['status'] == 'PASS' and probe['optimizer_steps'] == 0
        assert probe['parameter_versions_unchanged'] and probe['validation_rows'] == 576
        assert probe['archive_sha256'] == original['archive_sha256']
        assert probe['checkpoint_revision'] == original['backup']['revision']
        worker = Path(__file__).with_name('measure_alignment_residual_checkpoint.py')
        assert probe['worker_sha256'] == sha(worker.read_bytes())
        summary = {'archive_sha256': probe['archive_sha256'], 'scores': probe['scores']}
        summary['residual_norm_ranges'] = {}
        for key in ('correction_over_current', 'residual_over_current'):
            values = [v for layer in probe['residual_norms'] for v in layer[key]]
            summary['residual_norm_ranges'][key] = {'min': min(values), 'max': max(values)}
        if probe['gradients']:
            summary['gradient_means'] = {}
            for group in ('module', 'backbone'):
                items = [batch['groups'][group] for batch in probe['gradients']]
                summary['gradient_means'][group] = {
                    key: (sum(x[key] for x in items) / len(items)
                          if all(x[key] is not None for x in items) else None)
                    for key in ('weighted_lsr_over_ce', 'cosine')}
        report['probes'][ident] = summary
    c0 = read(args.probes / 'AR_C0/RESULT.json')
    frame = pd.read_csv(args.probes / 'AR_C0/interventions.csv', keep_default_na=False)
    predictions = pd.read_csv(args.output / 'inputs/runs/AR_C0/validation_predictions.csv',
                              keep_default_na=False)
    assert len(frame) == 4 * len(predictions)
    for condition, scores in c0['scores'].items():
        rows = frame[frame.condition == condition].sort_values('row')
        assert rows.row.tolist() == list(range(len(predictions)))
        for name, metric in (('bleu', BLEU(tokenize='zh')), ('chrf++', CHRF(word_order=2))):
            score = metric.corpus_score(rows.prediction.str.strip().tolist(),
                                       [predictions.target.str.strip().tolist()]).score
            assert abs(score - scores[name]) < 1e-8, (condition, name)
        assert abs(rows.nll_sum.sum() / rows.tokens.sum() - scores['token_ce']) < 1e-7
        if condition == 'all-on':
            assert rows.prediction.tolist() == predictions.prediction.tolist()
    c1 = read(args.probes / 'AR_C1/RESULT.json')
    c2 = read(args.probes / 'AR_C2/RESULT.json')
    assert [x['rows'] for x in c1['gradients']] == [x['rows'] for x in c2['gradients']]
    assert all(x['groups']['backbone']['weighted_lsr_norm'] == 0 for x in c2['gradients'])
    design = {'draws': 10000, 'seed': 42, 'bleu_tokenizer': 'zh',
              'pairs': [['CE_DONOR', 'AR_C0'], ['CE_LOCAL', 'AR_C0'], ['R1', 'CE_LOCAL']],
              'scope': 'Post-hoc development: 576 reused validation rows, one training seed. '
                       'Holm9 covers 3 contrasts x 3 metrics, as recorded before this analysis. '
                       'Sentence uncertainty excludes training randomness and method selection.'}
    (args.output / 'statistics_design.json').write_text(json.dumps(design, indent=2), encoding='utf-8')
    (args.output / 'AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'runs': list(report['runs']),
                      'probes': list(report['probes']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
