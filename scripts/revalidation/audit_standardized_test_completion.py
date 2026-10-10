"""Audit all 14 test artifacts, public index, documentation and completed queues."""
import hashlib
import json
import sys
from pathlib import Path
import requests
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner.colab import get_hf_token, call_pilot
REPO = 'FiveC/amis-rewire-checkpoints'
OUT = ROOT / 'outputs_rebuttal/amis_standardized_test_14'


def get(path, revision):
    response = requests.get(f'https://huggingface.co/{REPO}/resolve/{revision}/{path}', timeout=60)
    response.raise_for_status()
    return response.content


def main():
    api = HfApi(token=get_hf_token())
    info = api.model_info(REPO)
    assert not info.private
    index = json.loads(get('registry/checkpoint_index.json', info.sha))
    indexed = {r['path']: r for r in index['checkpoints']}
    lock_path = ROOT / 'outputs_rebuttal/amis_standardized_controls/selection_lock_14.json'
    lock = json.loads(lock_path.read_text())
    assert get('registry/plans/amis_standardized_controls/selection_lock_14.json', info.sha) == lock_path.read_bytes()
    verified = {}
    for cohort, count in [('amis_standardized_controls', 12), ('amis_standardized_strong_mbart', 2)]:
        folder = ROOT / 'outputs_rebuttal' / cohort
        report_path = folder / 'test_verification.json'
        report = json.loads(report_path.read_text())
        assert report['status'] == f'ALL_{count}_TEST_VERIFIED' and not report['pending']
        assert get(f'registry/plans/{cohort}/test_verification.json', info.sha) == report_path.read_bytes()
        specs = {r['id']: r for r in json.loads((folder / 'PLAN.json').read_text())['runs']}
        for ident, result in report['runs'].items():
            spec = specs[ident]
            receipt_path = folder / 'runs' / ident / 'COMPLETE.json'
            assert get(spec['remote_prefix'] + '/COMPLETE.json', info.sha) == receipt_path.read_bytes()
            receipt = json.loads(receipt_path.read_text())
            assert receipt['selection'] == lock['runs'][ident]
            assert receipt['joint_selection_lock_sha256'] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
            assert receipt['independent_metric_replay'] and receipt['test_backup']['verified']
            row = indexed[lock['runs'][ident]['archive_path']]
            assert row['test_verified'] and row['sha256'] == lock['runs'][ident]['archive_sha256']
            verified[ident] = result
    assert len(verified) == 14 and set(verified) == set(lock['runs'])
    for relative in ['docs/experiments/EXPERIMENTS_UPDATED.md', 'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md']:
        doc = (ROOT / relative).read_text(encoding='utf-8')
        section = doc.split('<!-- standardized-es10-test-results -->')[1].split('<!-- end-standardized-es10-test-results -->')[0]
        for ident, result in verified.items():
            row = next(line for line in section.splitlines() if line.startswith(f'| {ident} |'))
            for key in ['bleu', 'chrf++', 'chrf0']:
                assert f"{result['scores'][key]:.4f}" in row
    for account in [1, 2]:
        state = json.loads((OUT / f'status/account{account}.json').read_text())
        assert not state['process_alive'] and state['queue']['status'] == 'TEST_QUEUE_COMPLETE'
        assert state['queue']['assigned'] == state['queue']['completed']
        assert set(state['queue']['completed']).issubset(verified)
    usage = {}
    for account in [1, 2, 3]:
        code, stdout, stderr = call_pilot(str(account), ['usage'])
        assert code == 0, stderr
        assert 'Active assignments: 0' in stdout and 'Usage rate: 0.00/hr' in stdout
        usage[str(account)] = stdout.strip()
    for cohort in ['amis_standardized_controls', 'amis_standardized_strong_mbart']:
        assert not list((ROOT / 'outputs_rebuttal' / cohort / 'runs').rglob('*.zip'))
        assert not list((ROOT / 'outputs_rebuttal' / cohort / 'runs').rglob('*.safetensors'))
    audit = {'status': 'ALL_14_TEST_COMPLETION_AUDIT_PASS', 'public_revision': info.sha,
             'verified': verified, 'usage': usage, 'test_accounts': [1, 2], 'weights_downloaded_to_pc': False}
    path = OUT / 'completion_audit.json'
    path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    for name in ['PLAN.json', 'completion_audit.json']:
        api.upload_file(repo_id=REPO, path_in_repo=f'registry/plans/{OUT.name}/{name}', path_or_fileobj=str(OUT / name))
    print(audit['status'], len(verified), flush=True)


if __name__ == '__main__':
    main()
