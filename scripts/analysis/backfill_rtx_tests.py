"""Evaluate inventoried missing RTX tests; no training or selector changes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from experiment_runner.integrity import sha256, verify_remote_files


def main(manifest_path):
    from huggingface_hub import HfApi
    assert ROOT==Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    manifest=json.loads(manifest_path.read_text())
    folder=manifest_path.parent
    def write(name,value):
        p=folder/name;tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)
    for name,digest in manifest['evaluation_sources'].items():
        assert sha256(ROOT/name)==digest, ('Evaluation source changed',name)
    prior=json.loads((ROOT/manifest['after_dispatch']/'status.json').read_text())
    assert prior['status'] in ('COMPLETE_VERIFIED','FAILED'), 'Earlier AC/DRM test queue still active'
    os.environ.setdefault('HF_TOKEN',(ROOT/'.secrets/hf_token').read_text().strip())
    api=HfApi(token=os.environ['HF_TOKEN']);repo=manifest['repo']
    assert not api.model_info(repo).private
    state=dict(status='RUNNING',completed=[],skipped=[],failed=[],backed_up=[],
        prior_ac_drm_dispatch_status=prior['status'])
    write('status.json',state)
    for job in manifest['backup_only']:
        try:
            source=ROOT/job['source']
            completion=json.loads((source/'COMPLETE.json').read_text())
            assert completion['selection']['archive_sha256']==job['archive_sha256']
            metric=json.loads((source/'test_metrics.json').read_text())
            assert metric['beams']==4 and metric['generation_precision']=='fp32'
            assert metric['raw_reference_sha256']==job['test_sha256']
            import pandas as pd
            from amis_rewire.metrics import generation_metrics
            from sacrebleu.metrics import CHRF
            frame=pd.read_csv(source/'test_predictions.csv',dtype=str,keep_default_na=False)
            refs=pd.read_csv(ROOT/'data_processed'/job['dataset']/'test.csv',dtype=str,keep_default_na=False)
            assert sha256(ROOT/'data_processed'/job['dataset']/'test.csv')==job['test_sha256']
            assert len(frame)==len(refs)==job['rows']
            assert frame['source'].tolist()==refs['source'].tolist() and frame['target'].tolist()==refs['target'].tolist()
            scores=generation_metrics(frame['prediction'].tolist(),refs['target'].tolist(),tokenize=job['bleu_tokenizer'])
            scores['chrf0']=CHRF(word_order=0).corpus_score(frame['prediction'].tolist(),[refs['target'].str.strip().tolist()]).score
            assert all(abs(scores[k]-metric['scores'][k])<1e-10 for k in scores)
            current=api.get_paths_info(repo,[job['prefix']+'/best.zip'])[0]
            assert current.lfs.sha256==job['archive_sha256']
            files={job['prefix']+'/'+name:source/name for name in ('test_metrics.json','test_predictions.csv')}
            assert not api.file_exists(repo,job['prefix']+'/test_metrics.json'), 'Test already uploaded; inspect before overwrite'
            for remote,local in files.items():
                api.upload_file(repo_id=repo,path_in_repo=remote,path_or_fileobj=str(local))
            backup=verify_remote_files(api,repo,files)
            receipt=dict(status='COMPLETE_VERIFIED',id=job['id'],scores=scores,
                selection=completion['selection'],test_backup=backup,independent_metric_replay=True,
                recovered_existing_test=True,original_completion_sha256=sha256(source/'COMPLETE.json'))
            dest=folder/(job['id']+'_recovery.json');dest.write_text(json.dumps(receipt,indent=2))
            api.upload_file(repo_id=repo,path_in_repo=job['prefix']+'/COMPLETE.json',path_or_fileobj=str(dest))
            verify_remote_files(api,repo,{job['prefix']+'/COMPLETE.json':dest})
            state['backed_up'].append(job['id'])
        except Exception as error:
            state['failed'].append(dict(id=job['id'],stage='BACKUP_ONLY',error=str(error)))
        write('status.json',state)
    evaluator=ROOT/'scripts/analysis/evaluate_frozen_rtx.py'
    for job in manifest['jobs']:
        state.update(current=job['key'],stage='TEST')
        write('status.json',state)
        try:
            assert shutil.disk_usage(ROOT).free>=15*2**30, 'Insufficient disk for transient checkpoint restore'
            current=api.get_paths_info(repo,[job['prefix']+'/best.zip'])[0]
            assert current.lfs.sha256==job['archive_sha256'], 'Checkpoint changed since inventory'
            output=ROOT/job['output']
            marker=output/'runs'/job['id']/'COMPLETE.json'
            if marker.exists():
                old=json.loads(marker.read_text())
                assert old['selection']['archive_sha256']==job['archive_sha256'] and old['test_backup']['verified']
                state['skipped'].append(job['key']);continue
            # A concurrent public upload must be inspected, never overwritten.
            assert not api.file_exists(repo,job['prefix']+'/test_metrics.json'), 'Public test appeared after inventory'
            logs=folder/'logs';logs.mkdir(exist_ok=True)
            with (logs/(job['key']+'.log')).open('a') as log:
                if not output.exists():
                    subprocess.run([sys.executable,'-u',str(evaluator),'--stage','prepare',
                        '--cohort',job['cohort'],'--output',job['output'],'--ids',job['id'],
                        '--authorize-test','--minimum-free-gpu-mib','20000'],
                        check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                lock=json.loads((output/'selection_lock.json').read_text())
                assert lock['runs'][job['id']]['archive_sha256']==job['archive_sha256']
                subprocess.run([sys.executable,'-u',str(evaluator),'--stage','queue','--output',job['output']],
                    check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            receipt=json.loads(marker.read_text())
            assert receipt['test_backup']['verified']
            state['completed'].append(dict(key=job['key'],id=job['id'],scores=receipt['scores']))
        except Exception as error:
            state['failed'].append(dict(key=job['key'],stage='TEST',error=str(error)))
        write('status.json',state)
    state.update(status='COMPLETE_WITH_ERRORS' if state['failed'] else 'COMPLETE_VERIFIED',stage='FINISHED')
    write('status.json',state)
    print('RTX_TEST_BACKFILL_FINISHED',json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True,type=Path)
    main(parser.parse_args().manifest)
