"""Evaluate explicitly authorized, validation-selected RTX checkpoints using the shared engine."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner.integrity import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(args, out):
    from huggingface_hub import HfApi
    assert args.authorize_test, 'Explicit test authorization is required'
    assert not out.exists(), 'Use the existing frozen evaluation; do not overwrite'
    cohorts = [ROOT / name for name in args.cohort]
    plan = copy.deepcopy(read(cohorts[0] / 'PLAN.json'))
    plan['runs'] = []
    plan['protocol']['test_barrier'] = 'User-authorized evaluation of frozen validation-selected checkpoints'
    plan['protocol']['minimum_free_gpu_mib'] = args.minimum_free_gpu_mib
    plan['test_authorization'] = {'requested': True, 'selection': 'Existing validation best, unchanged',
        'scope': 'Exploratory test characterization; prior development/test exposure is retained, not independent confirmation'}
    api = HfApi(token=False)
    info = api.model_info(plan['repo']); assert not info.private
    locked, provenance = {}, {}
    for cohort in cohorts:
        parent = read(cohort / 'PLAN.json')
        assert parent['data'] == plan['data'] and parent['runtime'] == plan['runtime']
        for spec in parent['runs']:
            if args.ids and spec['id'] not in args.ids:
                continue
            ident = spec['id']; assert ident not in locked
            marker = read(cohort / 'runs' / ident / 'VAL_COMPLETE.json')
            assert marker['spec'] == spec and marker['status'] == 'VALIDATION_COMPLETE_VERIFIED'
            assert marker['backup']['verified']
            path = spec['remote_prefix'] + '/best.zip'
            entry = api.get_paths_info(plan['repo'], [path], revision=info.sha)[0]
            assert entry.lfs.sha256 == marker['archive_sha256']
            locked[ident] = {'revision': info.sha, 'archive_path': path,
                             'archive_sha256': marker['archive_sha256']}
            provenance[ident] = {'validation_receipt': str((cohort / 'runs' / ident / 'VAL_COMPLETE.json').relative_to(ROOT)),
                                 'receipt_sha256': sha256(cohort / 'runs' / ident / 'VAL_COMPLETE.json')}
            plan['runs'].append(spec)
    assert plan['runs'] and (not args.ids or set(locked) == set(args.ids))
    plan['test_authorization']['training_seeds'] = sorted({s['seed'] for s in plan['runs']})
    out.mkdir(parents=True)
    plan['status'] = 'FROZEN_TEST_ONLY'
    (out / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    files = list((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve(), out / 'PLAN.json']
    files += [ROOT / 'data_processed' / plan['dataset'] / (s + '.csv') for s in plan['data']]
    manifest = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in files}
    (out / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    lock = {'status': 'ALL_VALIDATION_VERIFIED', 'runs': locked, 'plan_sha256': sha256(out / 'PLAN.json'),
            'test_used_for_selection': False, 'validation_provenance': provenance}
    (out / 'selection_lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    print('FROZEN_TEST_CHECKPOINTS', list(locked), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'queue', 'run'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cohort', action='append', default=[])
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--id')
    parser.add_argument('--authorize-test', action='store_true')
    parser.add_argument('--minimum-free-gpu-mib', type=int, default=16000)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve(); assert out.is_relative_to(ROOT / 'outputs/experiments')
    if args.stage == 'prepare':
        prepare(args, out); return
    plan = read(out / 'PLAN.json')
    assert plan['test_authorization']['requested']
    if args.stage == 'queue':
        for spec in plan['runs']:
            if (out / 'runs' / spec['id'] / 'COMPLETE.json').exists():
                assert read(out / 'runs' / spec['id'] / 'COMPLETE.json')['test_backup']['verified']
                continue
            while True:
                free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                                   '--format=csv,noheader,nounits'], text=True).strip())
                if free >= plan['protocol']['minimum_free_gpu_mib']:
                    break
                print('WAIT_FREE_VRAM', spec['id'], free, flush=True); time.sleep(60)
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--stage', 'run',
                            '--output', str(out.relative_to(ROOT)), '--id', spec['id']], cwd=ROOT, check=True)
        print('FROZEN_TEST_QUEUE_COMPLETE', flush=True)
    else:
        os.environ.setdefault('HF_TOKEN', (ROOT / '.secrets/hf_token').read_text().strip())
        from experiment_runner import engine
        engine.configure(out)
        spec = next(s for s in plan['runs'] if s['id'] == args.id)
        engine.test_run(spec)


if __name__ == '__main__':
    main()
