"""Authorized post-training test worker; launcher must hold the project lock."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from experiment_runner.integrity import sha256

COHORT = 'outputs/experiments/amis_two_method_round1_rtx'
OUTPUT = 'outputs/experiments/amis_two_method_round1_rtx_test'
DISPATCH = 'outputs/experiments/amis_two_method_round1_rtx_test_dispatch'


def main():
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    dispatch = ROOT/DISPATCH
    def write(record):
        temp=dispatch/'status.tmp'
        temp.write_text(json.dumps(record,indent=2),encoding='utf-8')
        temp.replace(dispatch/'status.json')
    try:
        state=json.loads((ROOT/COHORT/'TWO_METHOD_QUEUE.json').read_text())
        assert state['status']=='COMPLETE', 'Training queue incomplete/failed; do not overlap'
        assert state['completed'], 'No completed models to test'
        assert set(state['completed']) <= {'AC_S42','DRM_S42'}
        for name,digest in json.loads((ROOT/COHORT/'bundle_manifest.json').read_text()).items():
            assert sha256(ROOT/name)==digest, ('Training source changed',name)
        os.environ.setdefault('HF_TOKEN',(ROOT/'.secrets/hf_token').read_text().strip())
        evaluator=ROOT/'scripts/analysis/evaluate_frozen_rtx.py'
        recipe=json.loads((dispatch/'recipe.json').read_text())
        assert sha256(evaluator)==recipe['evaluator_sha256']
        assert sha256(Path(__file__))==recipe['worker_sha256']
        ids=state['completed']
        write(dict(status='PREPARING_TEST',ids=ids,training_unchanged=True,test_for_selection=False))
        output=ROOT/OUTPUT
        if not output.exists():
            subprocess.run([sys.executable,'-u',str(evaluator),'--stage','prepare',
                '--cohort',COHORT,'--output',OUTPUT,'--ids',*ids,
                '--authorize-test','--minimum-free-gpu-mib','20000'],check=True,cwd=ROOT)
        else:
            locked=json.loads((output/'selection_lock.json').read_text())
            assert set(locked['runs'])==set(ids), 'Existing test selection mismatch'
        write(dict(status='TEST_RUNNING',ids=ids,beams=4,rows=575,precision='fp32',
            test_for_selection=False,interpretation='Exploratory, not independent confirmation'))
        subprocess.run([sys.executable,'-u',str(evaluator),'--stage','queue','--output',OUTPUT],
            check=True,cwd=ROOT)
        results={}
        for ident in ids:
            receipt=json.loads((output/'runs'/ident/'COMPLETE.json').read_text())
            assert receipt['status']=='COMPLETE_VERIFIED' and receipt['test_backup']['verified']
            results[ident]=receipt['scores']
        write(dict(status='COMPLETE_VERIFIED',results=results,
            test_for_selection=False,interpretation='Exploratory test characterization'))
        print('TWO_METHOD_TEST_COMPLETE',json.dumps(results),flush=True)
    except Exception as error:
        write(dict(status='FAILED',error=str(error),training_unchanged=True))
        raise


if __name__=='__main__':
    main()
