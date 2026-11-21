"""Prepare reproducible, validation-only audit inputs for the four-arm RTX pilot."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    lock = read(config['protocol_lock']['path'])
    assert hashlib.sha256(Path(config['protocol_lock']['path']).read_bytes()).hexdigest() == config['protocol_lock']['sha256']
    sources = {'R1': args.baseline, **{r['id']: args.cohort / 'runs' / r['id'] for r in config['runs']}}
    audit = {'status': 'PASS', 'scope': 'Validation only; no weights downloaded; no new training.', 'runs': {}}
    for ident, folder in sources.items():
        receipt, meta = read(folder / 'VAL_COMPLETE.json'), read(folder / 'run_metadata.json')
        assert receipt['status'] == 'VALIDATION_COMPLETE_VERIFIED' and receipt['backup']['verified']
        assert meta['data'] == config['data']
        assert meta['test_used_for_selection'] is False
        for key, value in lock['run'].items():
            assert meta['run'][key] == value, (ident, key)
        for key, value in lock['protocol'].items():
            if ident == 'R1' and key == 'attention_implementation':
                continue  # Historical default independently established in backend_probe.json.
            assert meta['protocol'][key] == value, (ident, key)
        for key, value in meta['runtime'].items():
            assert value == config['runtime'][key], (ident, key)
        dest = args.output / 'inputs' / 'runs' / ident
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('VAL_COMPLETE.json', 'run_metadata.json', 'validation_metrics.json', 'validation_predictions.csv'):
            if name != 'VAL_COMPLETE.json':
                remote = meta['run']['remote_prefix'] + '/' + name
                expected = receipt['backup']['files'][remote]
                content = (folder / name).read_bytes()
                # The SSH text collector converts LF to CRLF on Windows. Verify
                # that this is the only byte change; retain original local files.
                if hashlib.sha256(content).hexdigest() != expected['sha256']:
                    content = content.replace(b'\r\n', b'\n')
                assert len(content) == expected['size'], (ident, name)
                assert hashlib.sha256(content).hexdigest() == expected['sha256'], (ident, name)
            shutil.copy2(folder / name, dest / name)
        fetched = []
        if ident != 'R1':
            for name in ('loss_components.jsonl', 'final_trainer_state.json', 'selected_trainer_state.json', 'resolved_training_arguments.json', 'training_receipt.json'):
                remote = meta['run']['remote_prefix'] + '/' + name
                expected = receipt['backup']['files'][remote]
                url = f"https://huggingface.co/{receipt['backup']['repo']}/resolve/{receipt['backup']['revision']}/{remote}"
                target = dest / name
                content = target.read_bytes() if target.exists() else urlopen(url, timeout=60).read()
                assert len(content) == expected['size']
                assert hashlib.sha256(content).hexdigest() == expected['sha256']
                target.write_bytes(content)
                fetched.append(name)
            state = read(dest / 'final_trainer_state.json')
            resolved = read(dest / 'resolved_training_arguments.json')
            for key, value in dict(per_device_train_batch_size=4, gradient_accumulation_steps=32,
                num_train_epochs=20, learning_rate=5e-5, seed=42, data_seed=42, bf16=False, fp16=False,
                gradient_checkpointing=True, max_grad_norm=1.0).items():
                assert resolved[key] == value, (ident, key)
            assert read(args.cohort / 'gates' / ident / 'PASS.json')['passed'] is True
            evaluations = [row for row in state['log_history'] if 'eval_chrf++' in row]
            assert evaluations, list(state['log_history'][-1])
            best = max(evaluations, key=lambda row: row['eval_chrf++'])
            assert best['step'] == meta['selected_step']
            assert abs(best['eval_chrf++'] - receipt['scores']['chrf++']) < 1e-8
        audit['runs'][ident] = dict(selected_epoch=meta['selected_epoch'], completed_epoch=meta['completed_epoch'],
            archive_sha256=receipt['archive_sha256'], backup_revision=receipt['backup']['revision'],
            scores=receipt['scores'], fetched_small_files=fetched)
    probe = Path('outputs/experiments/amis_depth_replication_rtx/baseline_reuse_audit/backend_probe.json')
    audit['transport_note'] = 'Local JSON/CSV collector uses Windows CRLF; HF SHA256 matches after LF restoration only. No content edits.'
    audit['historical_backend_evidence'] = read(probe)
    assert all(item['resolved_backend'] == 'sdpa' for item in audit['historical_backend_evidence']['cached_configs'])
    (args.output / 'AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    designs = {
        'locked': [['AR_C1', 'AR_C2'], ['AR_C0', 'AR_C2'], ['AR_C3', 'AR_C2'], ['R1', 'AR_C2']],
        'exploratory': [['R1', 'AR_C0'], ['AR_C0', 'AR_C1'], ['AR_C0', 'AR_C2'], ['AR_C3', 'AR_C2']],
    }
    for label, pairs in designs.items():
        design = dict(draws=10000, seed=42, bleu_tokenizer='zh', pairs=pairs,
            scope=f'{label} family of 12 tests; reused validation, one frozen seed. Sentence uncertainty excludes training randomness and selection bias. Exploratory family is post hoc and is not confirmatory.')
        (args.output / f'{label}_design.json').write_text(json.dumps(design, indent=2), encoding='utf-8')
    losses = {}
    for ident in sources:
        if ident == 'R1':
            continue
        dest = args.output / 'inputs' / 'runs' / ident
        # Evaluation/save callbacks repeat the last microbatch diagnostic. These
        # are snapshots, never epoch means or full-dataset training losses.
        df = pd.read_json(dest / 'loss_components.jsonl', lines=True).drop_duplicates('step', keep='first')
        losses[ident] = dict(unique_step_snapshots=len(df),
            first10_median=df.head(10)[['ce', 'lsr', 'weighted_lsr']].median().to_dict(),
            last10_median=df.tail(10)[['ce', 'lsr', 'weighted_lsr']].median().to_dict(),
            gradient_vectors_logged=False,
            eval_loss_scope='CE + weighted LSR for C1/C2/C3; CE for C0. Not directly comparable CE across arms.')
    (args.output / 'LOSS_SUMMARY.json').write_text(json.dumps(losses, indent=2), encoding='utf-8')
    if all((args.output / f'{label}_statistics' / 'paired_bootstrap.json').exists() for label in designs):
        tests = {}
        for label in designs:
            for row in read(args.output / f'{label}_statistics' / 'paired_bootstrap.json')['tests']:
                tests[(row['control'], row['candidate'], row['metric'])] = dict(row)
        rows = sorted(tests.values(), key=lambda row: row['p_raw'])
        running = 0.
        for rank, row in enumerate(rows):
            running = max(running, min(1., (len(rows)-rank)*row['p_raw']))
            row['p_holm_all_unique'] = running
        (args.output / 'ALL_COMPARISONS_HOLM.json').write_text(json.dumps(dict(
            family_size=len(rows), scope='Post-hoc sensitivity across all unique comparisons; locked family unchanged.',
            tests=rows), indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
