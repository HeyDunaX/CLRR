"""Independent public artifact replay and selection lock for approved controls."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
import pandas as pd
import requests
from huggingface_hub import HfApi
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT / 'scratch'))
from amis_rewire.metrics import generation_metrics, metric_protocol
from experiment_runner.colab import get_hf_token
OUT = ROOT / 'outputs_rebuttal/amis_standardized_controls'
REPO = 'FiveC/amis-rewire-checkpoints'


def get(path, revision, required=True):
    response = requests.get(f'https://huggingface.co/{REPO}/resolve/{revision}/{path}', timeout=60)
    if response.status_code == 404 and not required: return None
    response.raise_for_status()
    return response.content


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def replay(predictions, raw, reported):
    frame = pd.read_csv(io.BytesIO(predictions), dtype=str, keep_default_na=False)
    assert len(frame) == len(raw)
    assert frame[['source', 'target']].equals(raw[['source', 'target']])
    scores = generation_metrics(frame['prediction'].tolist(), raw['target'].tolist(), tokenize='zh')
    scores['chrf0'] = CHRF(word_order=0).corpus_score(frame['prediction'].str.strip().tolist(), [raw['target'].str.strip().tolist()]).score
    assert all(abs(scores[k] - reported[k]) < 1e-9 for k in scores)
    return scores


def main():
    global OUT
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=['validation', 'test', 'publish-plan'], required=True)
    p.add_argument('--supplement', action='store_true', help='Verify the approved two Strong mBART runs')
    a = p.parse_args()
    if a.supplement:
        OUT = ROOT / 'outputs_rebuttal/amis_standardized_strong_mbart'
    api = HfApi(token=get_hf_token()); assert not api.model_info(REPO).private
    config = json.loads((OUT / 'PLAN.json').read_text())
    if a.phase == 'publish-plan':
        names = ['PLAN.json', 'bundle_manifest.json', 'CPU_GATE.json']
        if (OUT / 'A06_execution_assignment.json').exists():
            names.append('A06_execution_assignment.json')
        for name in names:
            api.upload_file(repo_id=REPO, path_in_repo=f'registry/plans/{OUT.name}/{name}', path_or_fileobj=str(OUT / name))
        print('APPROVED_PLAN_AND_CPU_GATE_PUBLIC', flush=True); return
    current = api.model_info(REPO).sha
    split = a.phase
    raw_path = ROOT / 'data_processed/amis_mandarin' / f'{split}.csv'
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == config['data'][split]['sha256']
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    verified, pending = {}, []
    for spec in config['runs']:
        prefix, ident = spec['remote_prefix'], spec['id']
        data = get(prefix + ('/VAL_COMPLETE.json' if split == 'validation' else '/COMPLETE.json'), current, required=False)
        if data is None:
            pending.append(ident); continue
        receipt = json.loads(data)
        assert receipt['id'] == ident
        backup = receipt['backup'] if split == 'validation' else receipt['test_backup']
        assert backup['verified'] and backup['repo'] == REPO
        entries = {e.path: e for e in api.get_paths_info(REPO, list(backup['files']), revision=backup['revision'])}
        folder = OUT / 'runs' / ident; folder.mkdir(parents=True, exist_ok=True)
        for path, proof in backup['files'].items():
            entry = entries[path]; assert entry.size == proof['size']
            if entry.lfs: assert entry.lfs.sha256 == proof['sha256']
            if not path.endswith('/best.zip'):
                small = get(path, backup['revision'])
                assert hashlib.sha256(small).hexdigest() == proof['sha256']
                if not entry.lfs:
                    assert hashlib.sha1(b'blob ' + str(len(small)).encode() + b'\0' + small).hexdigest() == entry.blob_id
                (folder / path.rsplit('/', 1)[-1]).write_bytes(small)
        predictions = (folder / f'{split}_predictions.csv').read_bytes()
        scores = replay(predictions, raw, receipt['scores'])
        metrics = json.loads((folder / f'{split}_metrics.json').read_text())
        assert metrics['protocol'] == metric_protocol('zh') and metrics['tf32_generation'] is False
        assert metrics['generation_precision'] == 'fp32'
        assert metrics['beams'] == (spec['validation_beams'] if split == 'validation' else 4)
        if split == 'validation':
            meta = json.loads((folder / 'run_metadata.json').read_text())
            args = json.loads((folder / 'resolved_training_arguments.json').read_text())
            assert meta['run'] == spec and meta['data'] == config['data']
            assert meta['protocol'] == config['protocol'] and not meta['test_used_for_selection']
            for field, expected in [('num_train_epochs', 20), ('gradient_accumulation_steps', spec['accumulation']),
                ('per_device_train_batch_size', spec['microbatch']), ('learning_rate', spec['learning_rate']),
                ('generation_num_beams', spec['validation_beams']), ('bf16', True), ('metric_for_best_model', 'eval_chrf++')]:
                assert args[field] == expected, (ident, field, args[field], expected)
            assert meta['stop_state']['patience'] == 10 and meta['save_reload_logits_match']
            assert meta['completed_epoch'] <= 20 and meta['selected_epoch'] <= meta['completed_epoch']
            weight = entries[prefix + '/best.zip']; assert weight.lfs.sha256 == receipt['archive_sha256']
            verified[ident] = {'revision': backup['revision'], 'archive_path': prefix + '/best.zip',
                'archive_sha256': receipt['archive_sha256'], 'validation_receipt_sha256': hashlib.sha256(data).hexdigest(),
                'selected_step': meta['selected_step'], 'scores': scores, 'selected_epoch': meta['selected_epoch'],
                'completed_epoch': meta['completed_epoch'], 'stop_reason': meta['stop_state']['stop_reason']}
            (folder / 'VAL_COMPLETE.json').write_bytes(data)
        else:
            assert receipt['status'] == 'COMPLETE_VERIFIED'
            lock_path = ROOT / 'outputs_rebuttal/amis_standardized_controls/selection_lock_14.json'
            lock = json.loads(lock_path.read_text(encoding='utf-8'))
            assert lock['status'] == 'ALL_14_VALIDATION_VERIFIED' and len(lock['runs']) == 14
            assert receipt['selection'] == lock['runs'][ident]
            assert receipt['joint_selection_lock_sha256'] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
            verified[ident] = {'scores': scores, 'revision': backup['revision']}
            (folder / 'COMPLETE.json').write_bytes(data)
        print('PUBLIC_REPLAY_PASS', ident, split, json.dumps(scores), flush=True)
    count = len(config['runs'])
    report = {'status': f'ALL_{count}_VALIDATION_VERIFIED' if split == 'validation' and not pending else
              (f'ALL_{count}_TEST_VERIFIED' if split == 'test' and not pending else 'PARTIAL_VERIFIED'),
              'phase': split, 'runs': verified, 'pending': pending, 'public_revision': current}
    write(OUT / f'{split}_verification.json', report)
    if not pending and split == 'validation':
        write(OUT / 'selection_lock.json', report)
        api.upload_file(repo_id=REPO, path_in_repo=f'registry/plans/{OUT.name}/selection_lock.json', path_or_fileobj=str(OUT / 'selection_lock.json'))
        print('SELECTION_LOCK_PUBLIC', flush=True)
    if not pending and split == 'test':
        api.upload_file(repo_id=REPO, path_in_repo=f'registry/plans/{OUT.name}/test_verification.json', path_or_fileobj=str(OUT / 'test_verification.json'))
    print('VERIFICATION_STATUS', report['status'], 'done', len(verified), 'pending', pending, flush=True)


if __name__ == '__main__': main()
