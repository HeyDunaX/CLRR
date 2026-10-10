"""Publish a joint selection barrier only after independent replay of all 14 runs."""
import hashlib
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner.colab import get_hf_token


def main():
    combined, proofs = {}, {}
    for name, expected in [('amis_standardized_controls', 12), ('amis_standardized_strong_mbart', 2)]:
        folder = ROOT / 'outputs_rebuttal' / name
        path = folder / 'validation_verification.json'
        report = json.loads(path.read_text())
        plan = json.loads((folder / 'PLAN.json').read_text())
        assert report['status'] == f'ALL_{expected}_VALIDATION_VERIFIED'
        assert not report['pending'] and len(report['runs']) == expected
        assert set(report['runs']) == {r['id'] for r in plan['runs']}
        assert not set(combined).intersection(report['runs'])
        combined.update(report['runs'])
        proofs[name] = {'verification_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'plan_sha256': hashlib.sha256((folder / 'PLAN.json').read_bytes()).hexdigest()}
    assert len(combined) == 14
    lock = {'status': 'ALL_14_VALIDATION_VERIFIED', 'runs': combined,
            'cohort_proofs': proofs, 'test_used_for_selection': False}
    out = ROOT / 'outputs_rebuttal/amis_standardized_controls/selection_lock_14.json'
    out.write_text(json.dumps(lock, indent=2), encoding='utf-8')
    api = HfApi(token=get_hf_token())
    assert not api.model_info('FiveC/amis-rewire-checkpoints').private
    result = api.upload_file(repo_id='FiveC/amis-rewire-checkpoints',
        path_in_repo='registry/plans/amis_standardized_controls/selection_lock_14.json',
        path_or_fileobj=str(out))
    print('JOINT_14_SELECTIONS_LOCKED', result.oid)


if __name__ == '__main__':
    main()
