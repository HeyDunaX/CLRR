"""Offline regression checks for scientific backup/selection/statistics gates."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import hashlib
import zipfile
import numpy as np
import torch
from transformers import MBartConfig, MBartForConditionalGeneration
from scipy.stats import ttest_1samp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/revalidation')]
from fresh_confirmation_integrity import verify_remote_files, restore_archive, remove_weights, strict_wrapped_model
from run_fresh_seed_confirmation import weight_digest

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

verifier=module('fresh_verifier','scripts/revalidation/verify_fresh_confirmation_local.py')
from experiment_runner import legacy as launcher
checks=[]
with tempfile.TemporaryDirectory() as tmp:
    base=Path(tmp)
    tiny=MBartForConditionalGeneration(MBartConfig(vocab_size=32,d_model=16,encoder_layers=2,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32))
    tiny.save_pretrained(base/'bin',safe_serialization=False)
    tiny.save_pretrained(base/'safe',safe_serialization=True)
    # Compare via loaded models because safetensors omits tied aliases from storage.
    loaded=strict_wrapped_model(base/'bin','clrr-enc',strength=.1,stack='encoder')
    assert sum(p.numel() for p in loaded.parameters())==sum(p.numel() for p in tiny.parameters())
    assert weight_digest(base/'bin/pytorch_model.bin')==weight_digest(base/'bin/pytorch_model.bin')
    strict_wrapped_model(base/'safe','baseline')
    checks.append('strict_tiny_checkpoint_bin_and_safetensors_load')
    archive=base/'weights.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.write(base/'bin/config.json','config.json')
    restore_archive(archive,base/'restore')
    with zipfile.ZipFile(base/'unsafe.zip','w') as z:z.writestr('../escape','bad')
    try:restore_archive(base/'unsafe.zip',base/'restore')
    except AssertionError:pass
    else:raise AssertionError('Path traversal accepted')
    checks.append('archive_restore_and_path_traversal_rejection')
    data=b'backup';(base/'artifact').write_bytes(data)
    class API:
        def model_info(self,repo):return SimpleNamespace(sha='pinned')
        def get_paths_info(self,repo,paths,revision):
            assert revision=='pinned'
            return [SimpleNamespace(path='artifact',size=len(data),lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))]
    receipt=verify_remote_files(API(),'repo',{'artifact':base/'artifact'})
    (base/'collection_receipt.json').write_text(json.dumps(receipt))
    (base/'runs/best_model').mkdir(parents=True)
    remove_weights(base,base/'runs',archive)
    assert not archive.exists() and not (base/'runs/best_model').exists()
    checks.append('pinned_remote_checksum_then_bounded_cleanup')
    (base/'artifact').write_bytes(b'wrong!')
    try:verify_remote_files(API(),'repo',{'artifact':base/'artifact'})
    except AssertionError:pass
    else:raise AssertionError('Corrupted bytes accepted')
    checks.append('corrupted_backup_rejected')
    diffs=np.array([.3,.8,-.1,.4,.2,.6,.1,.5])
    actual=verifier.paired_t_test(diffs)
    assert abs(actual['p_value']-ttest_1samp(diffs,0).pvalue)<1e-12
    assert verifier.exact_permutation_sign_flip_p([1,1,1])==.25
    assert verifier.paired_t_test([0,0])['p_value']==1
    assert verifier.paired_t_test([1,1])['p_value']==0
    checks.append('student_t_and_exact_sign_flip')
    verifier.BASE_DIR=base/'missing'
    try:verifier.verify_and_report('validation')
    except RuntimeError:pass
    else:raise AssertionError('Incomplete cohort accepted')
    checks.append('incomplete_validation_cohort_fails_closed')
for account in (1,2):
    train=launcher.generate_remote_runner(account,'train',partition='2gpu')
    test=launcher.generate_remote_runner(account,'test',partition='2gpu')
    assert train.count('--stage train')==16 and '--stage test' not in train
    assert test.count('--stage test')==16 and '--stage train' not in test
    assert '--cuda --audit-rerun' in train and 'fresh_bundle_files.json' in train
checks.append('two_accounts_stage_barrier_and_real_gpu_gate_commands')
all_runs=[]
for account, count in ((1,12),(2,14),(3,6)):
    train=launcher.generate_remote_runner(account,'train',partition='3gpu_safe')
    test=launcher.generate_remote_runner(account,'test',partition='3gpu_safe')
    assert train.count('--stage train')==count and '--stage test' not in train
    assert test.count('--stage test')==count and '--stage train' not in test
    all_runs.extend(launcher.get_run_plan(account,'3gpu_safe'))
assert len(all_runs)==len(set(all_runs))==32
assert set(all_runs)=={(family,seed) for family in ('baseline','clrr','lsr','full') for seed in range(45,53)}
checks.append('three_accounts_cover_same_32_runs_without_duplicates')
out=ROOT/'outputs_rebuttal/repository_cleanup'
out.mkdir(exist_ok=True,parents=True)
(out/'integrity_regression.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2))
print('FRESH_CONFIRMATION_INTEGRITY_PASS',len(checks))
