"""Matched old-method Turkish pilot: 12 runs, patience10; train/test barrier."""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
import torch
import transformers
import accelerate
from transformers import TrainerCallback, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from amis_rewire import train
from amis_rewire.evaluation import PrecisionSeq2SeqTrainer
from amis_rewire.metrics import safe_decode_inputs, generation_metrics, metric_protocol
from sacrebleu.metrics import CHRF
from fresh_confirmation_integrity import sha256, weight_file, verify_remote_files, restore_archive, remove_weights, strict_wrapped_model

FAMILY_CONFIGS = {
    'baseline': {'method': 'baseline', 'alpha': 0.0, 'lambda': 0.0, 'repo': 'FiveC/turkish-old-method-pilot-baseline-20261007'},
    'middle': {'method': 'middle-align', 'alpha': 0.0, 'lambda': 0.10, 'repo': 'FiveC/turkish-old-method-pilot-middle-20261007'},
    'lsr':      {'method': 'jepa',     'alpha': 0.0,  'lambda': 0.10, 'repo': 'FiveC/turkish-old-method-pilot-lsr-20261007'},
    'full':     {'method': 'jepa-clrr-enc', 'alpha': 0.10, 'lambda': 0.10, 'repo': 'FiveC/turkish-old-method-pilot-full-20261007'}
}

PREFIX = 'turkish_old_method_pilot_es10_20261007'

def weight_digest(path):
    if Path(path).suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        h.update(name.encode())
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(memoryview(t.contiguous().numpy()).cast('B'))
    del state
    gc.collect()
    return h.hexdigest()

def upload_metadata_folder(api, repo, folder, remote, required=False):
    import time
    for attempt in range(3 if required else 1):
        try:
            api.upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=remote, repo_type='model')
            return True
        except Exception as error:
            # Never print authorization header or token
            print(f'METADATA_UPLOAD_DEFERRED {type(error).__name__} {remote}', flush=True)
            if required and attempt < 2:
                time.sleep(30)
    if required:
        raise RuntimeError(f'Required final metadata upload failed for {remote}')
    return False

def evaluate_frozen_test(folder, family, repo, run_name, api):
    """Post-selection test evaluation strictly using the selected best checkpoint."""
    import pandas as pd
    gc.collect()
    torch.cuda.empty_cache()
    
    cfg = FAMILY_CONFIGS[family]
    best_dir = folder / 'best_model'
    tokenizer = AutoTokenizer.from_pretrained(best_dir, use_fast=True)
    model = strict_wrapped_model(
        str(best_dir),
        method=cfg['method'],
        distance=2,
        strength=cfg['alpha'],
        stack='encoder',
        jepa_weight=cfg['lambda']
    )
    train.configure_mbart(tokenizer, model, 'tr_TR', 'en_XX')
    model.config.use_cache = False
    
    prep = argparse.Namespace(source_prefix='', max_source_length=256, max_target_length=256)
    data = ROOT / 'data_processed/opus100_turkish_english_20k'
    dataset = train.tokenize_dataset(train.load_split(data, 'test'), tokenizer, prep)
    
    args = Seq2SeqTrainingArguments(
        output_dir=str(folder / 'test_staging'),
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=256,
        bf16=True,
        fp16=False,
        tf32=False,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False
    )
    trainer = PrecisionSeq2SeqTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        generation_dtype='fp32',
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    )
    
    result = trainer.predict(dataset)
    prediction = result.predictions[0] if isinstance(result.predictions, tuple) else result.predictions
    decoded = [s.strip() for s in tokenizer.batch_decode(safe_decode_inputs(prediction, tokenizer.pad_token_id), skip_special_tokens=True)]
    
    raw = pd.read_csv(data / 'test.csv', dtype=str, keep_default_na=False)
    assert len(decoded) == len(raw) == 2000, f"Expected 2000 test samples, got {len(decoded)}"
    
    frame = raw.copy()
    frame['prediction'] = decoded
    predictions_csv = folder / 'test_predictions.csv'
    frame.to_csv(predictions_csv, index=False)
    
    scores = generation_metrics(decoded, raw['target'].tolist(), tokenize='13a')
    scores['chrf0'] = CHRF(word_order=0).corpus_score(decoded, [raw['target'].str.strip().tolist()]).score
    metadata = {
        'selected_checkpoint_only': True,
        'selection_split': 'validation',
        'test_used_for_selection': False,
        'beams': 4,
        'batch_size': 8,
        'generation_precision': 'fp32',
        'tf32_generation': False,
        'raw_test_sha256': hashlib.sha256((data / 'test.csv').read_bytes()).hexdigest(),
        'protocol': metric_protocol('13a'),
        **{f'test_{k}': v for k, v in scores.items()}
    }
    metrics_json = folder / 'test_metrics.json'
    metrics_json.write_text(json.dumps(metadata, indent=2))
    
    # Upload test artifacts
    for p in (metrics_json, predictions_csv):
        assert train.safe_hf_upload(api, p, repo, f'{PREFIX}/{run_name}/{p.name}')
        
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return metadata

class SelectionBackupCallback(TrainerCallback):
    def __init__(self, out_dir, run_name, repo, api):
        self.out_dir = out_dir
        self.run_name = run_name
        self.repo = repo
        self.api = api

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        staged = self.out_dir / 'selection_staging' / f'checkpoint-{state.global_step}'
        staged.mkdir(parents=True, exist_ok=True)
        for name in ('trainer_state.json', 'training_receipt.json'):
            if (checkpoint / name).exists():
                shutil.copy2(checkpoint / name, staged / name)
        uploaded = upload_metadata_folder(self.api, self.repo, staged, f'{PREFIX}/{self.run_name}/selection/checkpoint-{state.global_step}')
        (staged / 'upload_status.json').write_text(json.dumps({'uploaded': uploaded, 'step': state.global_step}))
        state.save_to_json(str(self.out_dir / 'latest_trainer_state.json'))

    def on_train_end(self, args, state, control, **kwargs):
        state.save_to_json(str(self.out_dir / 'final_trainer_state.json'))
        staged = self.out_dir / 'selected_provenance'
        staged.mkdir(exist_ok=True)
        for name in ('trainer_state.json', 'training_receipt.json'):
            shutil.copy2(Path(state.best_model_checkpoint) / name, staged / name)
        upload_metadata_folder(self.api, self.repo, staged, f'{PREFIX}/{self.run_name}/selected_provenance', required=True)

def run_worker(family, seed, stage):
    cfg = FAMILY_CONFIGS[family]
    repo = cfg['repo']
    run_name = f"mbart-tr-en-{family}-seed{seed}-es10"
    out_dir = ROOT / f"outputs_rebuttal/{PREFIX}/{family}/seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment checks
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA with BF16 support is required.'
    assert 'A100' in torch.cuda.get_device_name(0), 'A100 required'
    for split, record in json.loads((ROOT/'data_processed/opus100_turkish_english_20k/manifest.json').read_text())['splits'].items():
        assert sha256(ROOT/f'data_processed/opus100_turkish_english_20k/{split}.csv')==record['sha256']
    manifest = json.loads((ROOT / 'fresh_bundle_files.json').read_text())
    assert all(sha256(ROOT / name) == digest for name, digest in manifest.items()), 'Bundle contents changed'
    assert transformers.__version__=='4.57.6' and accelerate.__version__=='1.15.0'
    assert torch.__version__.split('+')[0]=='2.6.0'
    gate_root = ROOT / f'outputs_rebuttal/{PREFIX}/gpu_gate'
    gate_receipt = json.loads((gate_root/'GATE_COMPLETE.json').read_text())
    assert gate_receipt['bundle_manifest_sha256'] == sha256(ROOT/'fresh_bundle_files.json')
    assert 'A100' in gate_receipt['gpu'] and gate_receipt['passed']
    fullsize = json.loads((ROOT/f'outputs_rebuttal/{PREFIX}/fullsize_gate/FULLSIZE_PASS.json').read_text())
    assert fullsize['status']=='FULLSIZE_GPU_PASS' and fullsize['batch']==32 and fullsize['accumulation']==4
    for gate_name in ('optimization_matrix.json', 'rerun_lifecycle_smoke.json'):
        gate = json.loads((gate_root / gate_name).read_text())
        # Regression gate uses an object; lifecycle gate a list of four branches.
        cases = gate if isinstance(gate, list) else gate.get('cases', gate.get('results', []))
        expected = 16 if gate_name == 'optimization_matrix.json' else 4
        assert len(cases)==expected and all(c.get('passed', False) for c in cases), gate_name
        assert all(c.get('cuda_bf16', c.get('cuda_bf16_training', False)) for c in cases), 'Gate was not run on GPU'
        actual={c.get('family', c.get('method')) for c in cases}
        expected_families={'baseline','middle','lsr','full'} if gate_name=='optimization_matrix.json' else {'baseline','middle-align','jepa','jepa-clrr-enc'}
        assert actual==expected_families, 'Gate must include actual Middle branch'
    token = os.environ.get('HF_TOKEN')
    assert token, "HF_TOKEN environment variable is required."
    api = HfApi(token=token)
    api.create_repo(repo, repo_type='model', private=True, exist_ok=True)
    
    if stage == 'test':
        print(f"\n=======================================================")
        print(f">>> STARTING TEST EVALUATION FOR {family.upper()} SEED {seed}")
        print(f"=======================================================")
        assert (out_dir / 'VAL_COMPLETE.json').exists(), f"Validation stage not complete for {family} seed {seed}!"
        assert not (out_dir / 'COMPLETE.json').exists(), f"Test stage already complete for {family} seed {seed}!"
        
        val_receipt = json.loads((out_dir / 'VAL_COMPLETE.json').read_text())
        lock = json.loads((ROOT / f'outputs_rebuttal/{PREFIX}/selection_lock.json').read_text())
        if lock['status'] == 'ALL_4_VALIDATION_VERIFIED':
            expected = {f'mbart-tr-en-{f}-seed42-es10' for f in ('baseline','middle','lsr','full')}
            assert seed == 42 and set(lock['runs']) == expected
        else:
            assert lock['status'] == 'ALL_12_VALIDATION_VERIFIED' and len(lock['runs']) == 12
        assert lock['runs'][run_name]['weight_digest'] == val_receipt['selected_and_exported_weight_digest']
        assert lock['runs'][run_name]['validation_receipt_sha256'] == sha256(out_dir/'VAL_COMPLETE.json')
        collected = json.loads((out_dir / 'collection_receipt.json').read_text())
        assert collected['verified']
        folder = out_dir / 'runs' / run_name
        archive = out_dir / 'backups' / run_name / f"{run_name}-best.zip"
        
        # Verify archive
        if not archive.exists():
            from huggingface_hub import hf_hub_download
            remote_archive = val_receipt['archive_path']
            restore_cache = out_dir/'hf_restore'
            downloaded = hf_hub_download(repo_id=repo, filename=remote_archive, revision=collected['revision'], token=token, local_dir=restore_cache)
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, archive)
            assert restore_cache.resolve().is_relative_to(out_dir.resolve())
            shutil.rmtree(restore_cache)
        with archive.open('rb') as f:
            archive_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        assert archive_sha == val_receipt['archive_sha256'], "Archive SHA256 mismatch!"
        
        best_model_dir = folder / 'best_model'
        if not (best_model_dir / 'config.json').exists():
            restore_archive(archive, best_model_dir)
        assert weight_digest(weight_file(best_model_dir)) == val_receipt['selected_and_exported_weight_digest']
                
        test_meta = evaluate_frozen_test(folder, family, repo, run_name, api)
        
        val_receipt.update({
            'status': 'COMPLETE',
            'post_selection_test_bleu': test_meta['test_bleu'],
            'post_selection_test_chrfpp': test_meta['test_chrf++'],
            'post_selection_test_chrf0': test_meta['test_chrf0']
        })
        (out_dir / 'COMPLETE.json').write_text(json.dumps(val_receipt, indent=2))
        
        staged = out_dir / 'final_provenance'
        staged.mkdir(exist_ok=True)
        shutil.copy2(out_dir / 'COMPLETE.json', staged / 'COMPLETE.json')
        upload_metadata_folder(api, repo, staged, f'{PREFIX}/{run_name}/provenance', required=True)
        remote_files = {f'{PREFIX}/{run_name}/provenance/COMPLETE.json': out_dir/'COMPLETE.json',
                        f'{PREFIX}/{run_name}/test_metrics.json': folder/'test_metrics.json',
                        f'{PREFIX}/{run_name}/test_predictions.csv': folder/'test_predictions.csv'}
        verified_test = verify_remote_files(api, repo, remote_files)
        verified_test['archive_verification'] = collected
        verified_test['files'] = {**collected['files'], **verified_test['files']}
        (out_dir/'collection_receipt.json').write_text(json.dumps(verified_test, indent=2))
        remove_weights(out_dir, folder, archive)
        print(f"TEST STAGE COMPLETED: {json.dumps(val_receipt)}", flush=True)
        return

    # STAGE: TRAIN
    print(f"\n=======================================================")
    print(f">>> STARTING TRAINING RUN: {family.upper()} SEED {seed}")
    print(f"=======================================================")
    
    # Overwrite & namespace check
    existing_files = api.list_repo_files(repo)
    assert not any(n.startswith(f"{PREFIX}/{run_name}/") for n in existing_files), f"Run {run_name} already exists on {repo}!"
    assert not (out_dir / 'VAL_COMPLETE.json').exists(), f"VAL_COMPLETE.json already exists in {out_dir}!"
    
    # Intercept PrecisionSeq2SeqTrainer to install SelectionBackupCallback and verify gates
    orig_trainer = train.PrecisionSeq2SeqTrainer
    def trainer_factory(*args, **kwargs):
        kwargs['callbacks'].append(SelectionBackupCallback(out_dir, run_name, repo, api))
        trainer = orig_trainer(*args, **kwargs)
        if family != 'baseline':
            assert trainer.model_accepts_loss_kwargs is False, "Wrapped methods must have model_accepts_loss_kwargs=False"
        assert trainer.args.gradient_accumulation_steps == 4
        assert trainer.args.bf16 and trainer.args.gradient_checkpointing
        assert trainer.args.metric_for_best_model == 'eval_chrf++'
        assert trainer.args.generation_num_beams == 1
        
        (out_dir / 'trainer_gate.json').write_text(json.dumps({
            'family': family, 'seed': seed,
            'accepts_loss_kwargs': trainer.model_accepts_loss_kwargs,
            'gradient_accumulation_steps': 4,
            'bf16': True, 'gradient_checkpointing': True,
            'selection_metric': 'eval_chrf++',
            'eval_beams': 1
        }, indent=2))
        return trainer
        
    train.PrecisionSeq2SeqTrainer = trainer_factory
    
    # Build CLI arguments for production engine
    sys.argv = [
        'train_worker',
        '--model', 'mbart-large-50',
        '--model-name', 'facebook/mbart-large-50-many-to-many-mmt',
        '--model-revision', 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30',
        '--method', cfg['method'],
        '--rewire-distance', '2',
        '--rewire-stack', 'encoder',
        '--rewire-strength', str(cfg['alpha']),
        '--jepa-weight', str(cfg['lambda']),
        '--middle-layer-idx', '6',
        '--alignment-temperature', '0.1',
        '--routing-warmup-ratio', '0.0',
        '--data-dir', str(ROOT / 'data_processed/opus100_turkish_english_20k'),
        '--output-dir', str(out_dir / 'runs'),
        '--backup-dir', str(out_dir / 'backups'),
        '--run-name', run_name,
        '--seed', str(seed),
        '--learning-rate', '5e-5',
        '--weight-decay', '0.0',
        '--warmup-ratio', '0.06',
        '--num-train-epochs', '20',
        '--early-stopping-patience', '10',
        '--per-device-train-batch-size', '32',
        '--gradient-accumulation-steps', '4',
        '--per-device-eval-batch-size', '8',
        '--max-source-length', '256',
        '--max-target-length', '256',
        '--eval-beams', '1',
        '--num-beams', '4',
        '--generation-precision', 'fp32',
        '--src-lang', 'tr_TR',
        '--tgt-lang', 'en_XX',
        '--bleu-tokenizer', '13a',
        '--no-fp16',
        '--bf16',
        '--gradient-checkpointing',
        '--no-auto-resume',
        '--validation-only',
        '--logging-steps', '10',
        '--dataloader-num-workers', '2',
        '--save-total-limit', '2',
        '--hf-backup-repo', repo,
        '--hf-backup-prefix', PREFIX
    ]
    (out_dir / 'command.json').write_text(json.dumps(sys.argv, indent=2))
    
    # Execute production training
    train.main()
    
    # Post-training verification
    folder = out_dir / 'runs' / run_name
    metrics = json.loads((folder / 'metrics.json').read_text())
    final_state = json.loads((out_dir / 'final_trainer_state.json').read_text())
    assert final_state['best_model_checkpoint'] == metrics['best_model_checkpoint'], "Best model checkpoint mismatch!"
    
    # Verify weight digest of selected vs exported
    best_weights_file = folder / 'best_model/pytorch_model.bin'
    if not best_weights_file.exists():
        best_weights_file = folder / 'best_model/model.safetensors'
    assert best_weights_file.exists(), f"Best model weights not found in {folder / 'best_model'}"
    best_digest = weight_digest(best_weights_file)
    assert best_digest == weight_digest(weight_file(Path(final_state['best_model_checkpoint']))), 'Selected/exported weights differ'
    
    # Check backup archive
    archive_file = out_dir / 'backups' / run_name / f"{run_name}-best.zip"
    assert archive_file.exists(), f"Backup archive not found: {archive_file}"
    with archive_file.open('rb') as f:
        archive_sha = hashlib.file_digest(f, 'sha256').hexdigest()
        
    val_receipt = {
        'status': 'VALIDATION_COMPLETE',
        'family': family,
        'seed': seed,
        'run_name': run_name,
        'selected_checkpoint': final_state['best_model_checkpoint'],
        'best_metric_chrfpp': final_state['best_metric'],
        'validation_bleu': metrics.get('eval_bleu'),
        'validation_chrf0': metrics.get('eval_chrf0'),
        'completed_epochs': metrics['completed_epochs'],
        'hf_repo': repo,
        'hf_prefix': PREFIX,
        'selected_and_exported_weight_digest': best_digest,
        'archive_sha256': archive_sha,
        'archive_size_bytes': archive_file.stat().st_size
    }
    val_receipt['archive_path'] = f'{PREFIX}/{run_name}/{archive_file.name}'
    val_receipt['bundle_files'] = manifest
    
    # Staging provenance
    staged = out_dir / 'final_provenance'
    staged.mkdir(exist_ok=True)
    for p in list(out_dir.glob('*.json')) + [folder / 'training_receipt.json']:
        if p.exists():
            shutil.copy2(p, staged / p.name)
            
    (out_dir / 'VAL_COMPLETE.json').write_text(json.dumps(val_receipt, indent=2))
    shutil.copy2(out_dir / 'VAL_COMPLETE.json', staged / 'VAL_COMPLETE.json')
    upload_metadata_folder(api, repo, staged, f'{PREFIX}/{run_name}/provenance', required=True)
    remote_files = {val_receipt['archive_path']: archive_file}
    for path in staged.iterdir():
        if path.is_file(): remote_files[f'{PREFIX}/{run_name}/provenance/{path.name}'] = path
    for name in ('metrics.json', 'validation_predictions.csv'):
        remote_files[f'{PREFIX}/{run_name}/{name}'] = folder/name
    for name in ('trainer_state.json', 'training_receipt.json'):
        remote_files[f'{PREFIX}/{run_name}/selected_provenance/{name}'] = out_dir/'selected_provenance'/name
    collection = verify_remote_files(api, repo, remote_files)
    assert collection['files'][val_receipt['archive_path']]['sha256']==archive_sha
    (out_dir/'collection_receipt.json').write_text(json.dumps(collection, indent=2))
    remove_weights(out_dir, folder, archive_file)
    
    # Local disk cleanup: intermediate checkpoints deleted, keeping best_model and archive
    for ckpt_dir in (out_dir / 'runs' / run_name).glob('checkpoint-*'):
        if ckpt_dir.is_dir():
            print(f"Cleaning intermediate checkpoint to save disk: {ckpt_dir.name}")
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            
    print(f"VALIDATION STAGE COMPLETE: {json.dumps(val_receipt)}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('baseline', 'middle', 'clrr', 'lsr', 'full'), required=True)
    parser.add_argument('--seed', type=int, choices=range(42, 45), required=True)
    parser.add_argument('--stage', choices=('train', 'test'), default='train')
    args = parser.parse_args()
    run_worker(args.family, args.seed, args.stage)
