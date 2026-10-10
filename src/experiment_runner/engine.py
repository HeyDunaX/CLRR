"""Approved ES10 controls: pinned recipe, gates, verified public backup, test barrier."""
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT / 'scripts/revalidation'))
import numpy as np
import pandas as pd
import torch
import transformers
import accelerate
import peft
import sacrebleu
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                         Seq2SeqTrainingArguments, EarlyStoppingCallback, TrainerCallback, set_seed)
from peft import PeftModel
from amis_rewire.evaluation import PrecisionSeq2SeqTrainer, generation_precision
from amis_rewire.metrics import build_compute_metrics, generation_metrics, metric_protocol, safe_decode_inputs
from comparative_baselines.middle_align_acl2025.model import MiddleAlignMT5
from peft_baselines.bitfit_adapter import apply_bitfit_to_model
from peft_baselines.lora_adapter import apply_lora_to_model
from .integrity import sha256, verify_remote_files, restore_archive
from methods.registry import ALIASES, CANDIDATES, ORIGINAL_ALIASES, wrap_method

OUT = ROOT / 'outputs_rebuttal/amis_standardized_controls'
DATA = ROOT / 'data_processed/amis_mandarin'
REPO = 'FiveC/amis-rewire-checkpoints'


def configure(output):
    """Select one frozen workspace in this process; training code is shared."""
    global OUT, DATA, REPO
    OUT = Path(output).resolve()
    configuration = plan()
    DATA = ROOT / 'data_processed' / configuration['dataset']
    REPO = configuration['repo']


def protocol():
    return plan()['protocol']


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(path)


def plan():
    return json.loads((OUT / 'PLAN.json').read_text(encoding='utf-8'))


def validate_environment(gpu=True):
    versions = {'torch': torch.__version__.split('+')[0], 'transformers': transformers.__version__,
                'accelerate': accelerate.__version__, 'peft': peft.__version__, 'sacrebleu': sacrebleu.__version__}
    for name, version in versions.items():
        assert version == plan()['runtime'][name], (name, version, plan()['runtime'][name])
    for name, digest in json.loads((OUT / 'bundle_manifest.json').read_text()).items():
        assert sha256(ROOT / name) == digest, ('Bundle changed', name)
    for split, record in plan()['data'].items():
        assert sha256(DATA / f'{split}.csv') == record['sha256']
    if gpu:
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        assert 'A100' in torch.cuda.get_device_name(0)
        assert 35 < torch.cuda.get_device_properties(0).total_memory / 2**30 < 45
    return versions


def wrap_model(base, spec):
    method = ORIGINAL_ALIASES.get(spec['method'], spec['method'])
    if method == 'middle-alignment':
        model = MiddleAlignMT5(base, middle_layer_idx=spec['middle_layer_idx'],
                               temperature=spec['temperature'], align_weight=spec['alignment_weight'],
                               pad_token_id=base.config.pad_token_id or 0)
        metadata = {'symmetric_InfoNCE': True, 'source_target_gradients': True,
                    'layer_hidden_state_index': spec['middle_layer_idx'], 'includes_embedding_index_zero': True,
                    'temperature': spec['temperature'], 'weight': spec['alignment_weight']}
    elif method == 'bitfit':
        model, metadata = apply_bitfit_to_model(base)
        assert all('bias' in n for n, p in model.named_parameters() if p.requires_grad)
    elif method in ('narrow-lora', 'strong-lora-a', 'strong-lora-b'):
        narrow = method == 'narrow-lora'
        model, metadata = apply_lora_to_model(base, r=8 if narrow else 16,
            lora_alpha=16 if narrow else 32, lora_dropout=.05,
            target_modules=['q_proj', 'v_proj'] if narrow else ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'],
            modules_to_save=['shared', 'embed_tokens', 'lm_head'] if method == 'strong-lora-b' else None)
        if method != 'strong-lora-b':
            assert not any('embed' in n and p.requires_grad for n, p in model.named_parameters())
    elif ALIASES.get(method, method) in CANDIDATES or method in ('original-clrr', 'original-lsr', 'original-full'):
        model = wrap_method(base, spec)
        metadata = {'candidate': method, 'definition': spec.get('definition', method),
                    'extra_parameters': 0}
    else:
        assert method == 'baseline'
        model, metadata = base, {'loss': 'CE only'}
    model.config.use_cache = False
    # Reentrant checkpointing matches the historical engine. Frozen-embedding
    # methods need grad-bearing embedding outputs without unfreezing weights.
    model.enable_input_require_grads()
    if method in ('bitfit', 'narrow-lora', 'strong-lora-a', 'strong-lora-b'):
        core = model.get_base_model() if isinstance(model, PeftModel) else model
        # mBART/M2M100 share the weight but call separate encoder/decoder modules.
        # The generic shared-module hook alone is never invoked on these paths.
        def require_output_grad(module, inputs, output):
            output.requires_grad_(True)
        for stack in (core.get_encoder(), core.get_decoder()):
            stack.embed_tokens.register_forward_hook(require_output_grad)
    metadata['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata['allocated_params'] = sum(p.numel() for p in model.parameters())
    metadata['loss_reduction'] = 'microbatch_mean'
    return model, metadata


def configure_languages(model, tokenizer, spec):
    if spec['src_lang']:
        tokenizer.src_lang, tokenizer.tgt_lang = spec['src_lang'], spec['tgt_lang']
        target = tokenizer.convert_tokens_to_ids(spec['tgt_lang'])
        assert target != tokenizer.unk_token_id
        model.config.forced_bos_token_id = target
        model.generation_config.forced_bos_token_id = target
        if spec['backbone'] == 'mbart-large-50':
            model.config.decoder_start_token_id = tokenizer.eos_token_id
            model.generation_config.decoder_start_token_id = tokenizer.eos_token_id


def load_model(spec, folder=None):
    if folder and (Path(folder) / 'adapter_config.json').exists():
        base = AutoModelForSeq2SeqLM.from_pretrained(spec['model_id'], revision=spec['model_revision'])
        model = PeftModel.from_pretrained(base, str(folder), is_trainable=False)
        if spec['method'] == 'strong-lora-b':
            core = model.get_base_model()
            weight = core.model.shared.modules_to_save['default'].weight
            for module in (core.model.encoder.embed_tokens, core.model.decoder.embed_tokens, core.lm_head):
                assert torch.equal(module.modules_to_save['default'].weight, weight)
                module.modules_to_save['default'].weight = weight
        metadata = {'restored_adapter': True}
    else:
        base, info = AutoModelForSeq2SeqLM.from_pretrained(str(folder) if folder else spec['model_id'],
            revision=None if folder else spec['model_revision'], output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
        model, metadata = wrap_model(base, spec)
    tokenizer = AutoTokenizer.from_pretrained(str(folder) if folder else spec['model_id'],
                                             revision=None if folder else spec['model_revision'], use_fast=True)
    configure_languages(model, tokenizer, spec)
    model.config.use_cache = False
    return model, tokenizer, metadata


def dataset(split, tokenizer, spec):
    raw = pd.read_csv(DATA / f'{split}.csv', dtype=str, keep_default_na=False)
    def tokenize(batch):
        inputs = tokenizer([spec['source_prefix'] + s for s in batch['source']],
                           max_length=spec['max_source_length'], truncation=True)
        inputs['labels'] = tokenizer(text_target=batch['target'], max_length=spec['max_target_length'], truncation=True)['input_ids']
        return inputs
    return raw, Dataset.from_pandas(raw, preserve_index=False).map(tokenize, batched=True, remove_columns=['source', 'target'])


class ControlTrainer(PrecisionSeq2SeqTrainer):
    """Save adapters as adapters and Middle weights without wrapper key prefixes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def _save(self, output_dir=None, state_dict=None):
        folder = Path(output_dir or self.args.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        model.save_pretrained(str(folder), state_dict=state_dict or model.state_dict(), safe_serialization=False)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(folder))
        torch.save(self.args, folder / 'training_args.bin')
        receipt = Path(self.args.output_dir) / 'training_receipt.json'
        if receipt.exists() and receipt.resolve() != (folder / receipt.name).resolve():
            shutil.copy2(receipt, folder / receipt.name)


class Progress(TrainerCallback):
    def __init__(self, folder, ident, early=None):
        self.folder, self.ident, self.early = folder, ident, early

    def on_log(self, args, state, control, logs=None, **kwargs):
        write(self.folder / 'progress.json', {'id': self.ident, 'status': 'TRAINING',
              'epoch': state.epoch, 'step': state.global_step, 'max_steps': state.max_steps,
              'best_metric': state.best_metric, 'updated_unix': time.time(), 'logs': logs,
              'patience_counter': self.early.early_stopping_patience_counter if self.early else None})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.on_log(args, state, control, logs=metrics)
        print('EPOCH_VALIDATION', self.ident, state.epoch, json.dumps(metrics), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        state.save_to_json(str(self.folder / 'final_trainer_state.json'))
        write(self.folder / 'stop_state.json', {'completed_epoch': state.epoch, 'completed_step': state.global_step,
            'max_steps': state.max_steps, 'best_step': state.best_global_step, 'best_metric': state.best_metric,
            'best_checkpoint': state.best_model_checkpoint, 'patience': protocol()['patience'],
            'patience_counter': self.early.early_stopping_patience_counter if self.early else None,
            'stop_reason': 'epoch_cap' if state.global_step >= state.max_steps else 'early_stopping'})


def arguments(folder, spec, gate=False):
    recipe = protocol()
    return Seq2SeqTrainingArguments(output_dir=str(folder), seed=spec['seed'], data_seed=spec['seed'],
        num_train_epochs=recipe['epochs'], max_steps=2 if gate else -1, learning_rate=spec['learning_rate'],
        weight_decay=recipe['weight_decay'], warmup_ratio=recipe['warmup_ratio'],
        lr_scheduler_type=recipe['scheduler'], optim=recipe['optim'],
        adam_beta1=recipe['adam_beta1'], adam_beta2=recipe['adam_beta2'],
        adam_epsilon=recipe['adam_epsilon'], max_grad_norm=recipe['max_grad_norm'],
        per_device_train_batch_size=spec['microbatch'], gradient_accumulation_steps=spec['accumulation'],
        per_device_eval_batch_size=recipe['eval_batch'], eval_strategy='no' if gate else 'epoch',
        save_strategy='steps' if gate else 'epoch', save_steps=1, save_total_limit=recipe['save_limit'],
        logging_steps=1 if gate else 10, predict_with_generate=True,
        generation_num_beams=spec['validation_beams'], generation_max_length=spec['max_target_length'],
        load_best_model_at_end=not gate, metric_for_best_model=None if gate else 'eval_chrf++',
        greater_is_better=True, bf16=recipe['bf16'], fp16=False, tf32=recipe['tf32_training'],
        gradient_checkpointing=recipe['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant': recipe['gradient_checkpointing_use_reentrant']},
        dataloader_num_workers=recipe.get('dataloader_num_workers', 2), dataloader_pin_memory=True, save_safetensors=False,
        restore_callback_states_from_checkpoint=True, remove_unused_columns=False, report_to=[])


def predict(trainer, ds, tokenizer, raw, name, folder, beams):
    trainer.args.generation_num_beams = beams
    bleu_tokenizer = protocol()['bleu_tokenizer']
    trainer.compute_metrics = build_compute_metrics(tokenizer, raw['target'].tolist(), tokenize=bleu_tokenizer)
    result = trainer.predict(ds, metric_key_prefix=name)
    ids = result.predictions[0] if isinstance(result.predictions, tuple) else result.predictions
    decoded = [s.strip() for s in tokenizer.batch_decode(safe_decode_inputs(ids, tokenizer.pad_token_id), skip_special_tokens=True)]
    assert len(decoded) == len(raw)
    scores = generation_metrics(decoded, raw['target'].tolist(), tokenize=bleu_tokenizer)
    scores['chrf0'] = sacrebleu.metrics.CHRF(word_order=0).corpus_score(decoded, [raw['target'].str.strip().tolist()]).score
    frame = raw.copy(); frame['prediction'] = decoded
    frame.to_csv(folder / f'{name}_predictions.csv', index=False)
    write(folder / f'{name}_metrics.json', {'scores': scores, 'protocol': metric_protocol(bleu_tokenizer),
          'beams': beams, 'generation_precision': 'fp32', 'tf32_generation': False,
          'raw_reference_sha256': sha256(DATA / f'{name}.csv')})
    return scores


def probe_logits(model, batch):
    model.eval()
    with torch.no_grad(), generation_precision(model, 'fp32'):
        logits = model(**batch).logits
        return logits[:, :2, :128].detach().float().cpu()


def gpu_gate(spec):
    validate_environment()
    set_seed(spec['seed'])
    folder = OUT / 'gates' / spec['id']
    assert not (folder / 'PASS.json').exists(), 'Gate already exists; verify bundle before rerun'
    folder.mkdir(parents=True, exist_ok=True)
    model, tokenizer, meta = load_model(spec)
    raw, ds = dataset('train', tokenizer, spec)
    # Include long examples and use the exact training microbatch/accumulation.
    indices = sorted(range(len(raw)), key=lambda i: len(raw.iloc[i]['source']) + len(raw.iloc[i]['target']), reverse=True)
    sample = ds.select(indices[:spec['microbatch'] * spec['accumulation'] * 2])
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    trainer = ControlTrainer(model=model, args=arguments(folder, spec, gate=True), processing_class=tokenizer,
                             data_collator=collator, train_dataset=sample, generation_dtype='fp32')
    started = time.time()
    trainer.train()
    assert trainer.state.global_step == 2
    assert np.isfinite(trainer.state.log_history[-1]['train_loss'])
    assert (folder / 'checkpoint-2/optimizer.pt').exists() and (folder / 'checkpoint-2/rng_state.pth').exists()
    trainer.state.best_model_checkpoint = str(folder / 'checkpoint-2')
    trainer._load_best_model()
    best = folder / 'export'
    trainer.save_model(str(best))
    probe = collator([sample[i] for i in range(2)])
    probe = {k: v.to('cuda') for k, v in probe.items()}
    expected = probe_logits(model, probe)
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    restored, _, _ = load_model(spec, best); restored.to('cuda')
    actual = probe_logits(restored, probe)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
    receipt = {'passed': True, 'id': spec['id'], 'gpu': torch.cuda.get_device_name(0),
               'bf16': True, 'checkpointing': True, 'microbatch': spec['microbatch'],
               'accumulation': spec['accumulation'], 'steps': 2, 'save_reload_logits_match': True,
               'trainer_best_checkpoint_reload': True,
               'optimizer_rng_present': True, 'seconds': time.time() - started, 'parameters': meta,
               'max_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
               'bundle_sha256': sha256(OUT / 'bundle_manifest.json')}
    write(folder / 'PASS.json', receipt)
    del restored, probe; gc.collect(); torch.cuda.empty_cache()
    for target in [best, *folder.glob('checkpoint-*')]:
        assert target.resolve().is_relative_to(folder.resolve())
        shutil.rmtree(target)
    print('GPU_GATE_PASS', spec['id'], json.dumps(receipt), flush=True)


def upload_and_verify(folder, spec, names):
    api = HfApi(token=os.environ['HF_TOKEN'])
    assert not api.model_info(REPO).private
    files = {spec['remote_prefix'] + '/' + n: folder / n for n in names}
    for remote, local in files.items():
        api.upload_file(repo_id=REPO, path_in_repo=remote, path_or_fileobj=str(local))
    return verify_remote_files(api, REPO, files)


def train_run(spec, resume=None):
    versions = validate_environment()
    gate = json.loads((OUT / 'gates' / spec['id'] / 'PASS.json').read_text())
    assert gate['passed'] and gate['bundle_sha256'] == sha256(OUT / 'bundle_manifest.json')
    cpu = json.loads((OUT / 'CPU_GATE.json').read_text())
    assert cpu['passed'] and cpu['bundle_sha256'] == sha256(OUT / 'bundle_manifest.json')
    folder = OUT / 'runs' / spec['id']; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'VAL_COMPLETE.json').exists():
        raise RuntimeError('Run already validated; do not retrain or overwrite')
    api = HfApi(token=os.environ['HF_TOKEN'])
    if not resume:
        assert not any(folder.glob('checkpoint-*')), 'Explicit --resume is required'
        assert not api.file_exists(REPO, spec['remote_prefix'] + '/best.zip'), 'Remote target exists; do not overwrite'
    else:
        assert Path(resume).resolve().is_relative_to(folder.resolve())
        assert all((Path(resume) / n).exists() for n in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'])
    set_seed(spec['seed'])
    model, tokenizer, meta = load_model(spec)
    raw_val, val_ds = dataset('validation', tokenizer, spec)
    _, train_ds = dataset('train', tokenizer, spec)
    metadata = {'run': spec, 'protocol': plan()['protocol'], 'data': plan()['data'], 'method': meta,
                'runtime': versions, 'gpu': torch.cuda.get_device_name(0),
                'bundle_sha256': sha256(OUT / 'bundle_manifest.json'), 'git_head': plan()['code_git_head'],
                'test_used_for_selection': False, 'status': 'TRAINING', 'started_unix': time.time()}
    write(folder / 'run_metadata.json', metadata)
    write(folder / 'training_receipt.json', metadata)
    early = EarlyStoppingCallback(early_stopping_patience=protocol()['patience'])
    trainer = ControlTrainer(model=model, args=arguments(folder, spec), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8),
        train_dataset=train_ds, eval_dataset=val_ds, generation_dtype='fp32',
        compute_metrics=build_compute_metrics(tokenizer, raw_val['target'].tolist(), tokenize=protocol()['bleu_tokenizer']),
        callbacks=[early, Progress(folder, spec['id'], early)])
    write(folder / 'resolved_training_arguments.json', trainer.args.to_dict())
    trainer.train(resume_from_checkpoint=resume)
    trainer.state.save_to_json(str(folder / 'final_trainer_state.json'))
    best_checkpoint = Path(trainer.state.best_model_checkpoint)
    shutil.copy2(best_checkpoint / 'trainer_state.json', folder / 'selected_trainer_state.json')
    scores = predict(trainer, val_ds, tokenizer, raw_val, 'validation', folder, spec['validation_beams'])
    assert abs(scores['chrf++'] - trainer.state.best_metric) < 1e-6, (scores, trainer.state.best_metric)
    export = folder / 'best_model'; trainer.save_model(str(export))
    probe = trainer.data_collator([val_ds[i] for i in range(2)])
    probe = {k: v.to('cuda') for k, v in probe.items()}
    expected = probe_logits(model, probe)
    final_state = json.loads((folder / 'final_trainer_state.json').read_text())
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    restored, _, _ = load_model(spec, export); restored.to('cuda')
    torch.testing.assert_close(probe_logits(restored, probe), expected, rtol=1e-5, atol=2e-5)
    del restored, probe; gc.collect(); torch.cuda.empty_cache()
    metadata.update(status='VALIDATION_SELECTED', selected_epoch=json.loads((folder / 'selected_trainer_state.json').read_text())['epoch'],
                    selected_step=int(best_checkpoint.name.split('-')[-1]),
                    completed_epoch=final_state['epoch'], completed_step=final_state['global_step'],
                    stop_state=json.loads((folder / 'stop_state.json').read_text()),
                    selected_validation=scores, completed_unix=time.time(), save_reload_logits_match=True)
    write(folder / 'run_metadata.json', metadata)
    with zipfile.ZipFile(folder / 'best.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in export.rglob('*'):
            if path.is_file(): archive.write(path, path.relative_to(export).as_posix())
    metadata['archive_sha256'] = sha256(folder / 'best.zip')
    write(folder / 'run_metadata.json', metadata)
    names = ['best.zip', 'run_metadata.json', 'training_receipt.json', 'resolved_training_arguments.json',
             'validation_metrics.json', 'validation_predictions.csv', 'final_trainer_state.json',
             'selected_trainer_state.json', 'stop_state.json']
    try:
        receipt = upload_and_verify(folder, spec, names)
    except Exception:
        write(folder / 'UPLOAD_PENDING.json', {'id': spec['id'], 'weights_retained': True})
        raise
    write(folder / 'collection_receipt.json', receipt)
    complete = {'status': 'VALIDATION_COMPLETE_VERIFIED', 'id': spec['id'], 'spec': spec,
                'scores': scores, 'archive_sha256': metadata['archive_sha256'], 'backup': receipt,
                'run_metadata_sha256': sha256(folder / 'run_metadata.json')}
    write(folder / 'VAL_COMPLETE.json', complete)
    upload_and_verify(folder, spec, ['VAL_COMPLETE.json', 'collection_receipt.json'])
    for target in [export, folder / 'best.zip', *folder.glob('checkpoint-*')]:
        assert target.resolve().is_relative_to(folder.resolve())
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    write(folder / 'progress.json', {'id': spec['id'], 'status': 'VALIDATION_COMPLETE_VERIFIED',
          'scores': scores, 'updated_unix': time.time()})
    print('VAL_COMPLETE_VERIFIED', spec['id'], json.dumps(scores), flush=True)


def test_run(spec):
    assert not protocol()['test_barrier'].startswith('No test'), 'Validation-only configuration'
    validate_environment()
    lock = json.loads((OUT / 'selection_lock.json').read_text())
    assert lock['status'] in ('ALL_VALIDATION_VERIFIED', 'ALL_12_VALIDATION_VERIFIED')
    assert set(lock['runs']) == {run['id'] for run in plan()['runs']}
    assert lock.get('test_used_for_selection', False) is False
    if 'plan_sha256' in lock:
        assert lock['plan_sha256'] == sha256(OUT / 'PLAN.json')
    folder = OUT / 'runs' / spec['id']; folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'COMPLETE.json').exists()
    selected = lock['runs'][spec['id']]
    archive = Path(hf_hub_download(REPO, spec['remote_prefix'] + '/best.zip', revision=selected['revision'], token=os.environ['HF_TOKEN'], local_dir=folder / 'restore'))
    assert sha256(archive) == selected['archive_sha256']
    export = folder / 'best_model'; restore_archive(archive, export)
    set_seed(spec['seed'])
    model, tokenizer, _ = load_model(spec, export)
    raw, ds = dataset('test', tokenizer, spec)
    args = arguments(folder / 'test_stage', spec)
    args.gradient_checkpointing = False
    args.load_best_model_at_end = False
    trainer = ControlTrainer(model=model, args=args, processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8), generation_dtype='fp32')
    scores = predict(trainer, ds, tokenizer, raw, 'test', folder, protocol()['test_beams'])
    frame = pd.read_csv(folder / 'test_predictions.csv', dtype=str, keep_default_na=False)
    recomputed = generation_metrics(frame['prediction'].tolist(), frame['target'].tolist(), tokenize=protocol()['bleu_tokenizer'])
    assert all(abs(recomputed[k] - scores[k]) < 1e-10 for k in recomputed)
    receipt = upload_and_verify(folder, spec, ['test_metrics.json', 'test_predictions.csv'])
    write(folder / 'COMPLETE.json', {'status': 'COMPLETE_VERIFIED', 'id': spec['id'],
          'scores': scores, 'test_backup': receipt, 'selection': selected, 'independent_metric_replay': True})
    upload_and_verify(folder, spec, ['COMPLETE.json'])
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    for target in (export, folder / 'restore'):
        assert target.resolve().is_relative_to(folder.resolve())
        shutil.rmtree(target)
    print('COMPLETE_VERIFIED', spec['id'], json.dumps(scores), flush=True)


def main():
    raise SystemExit('Use scripts/experiment.py with --config; historical runs require their pinned source bundle.')
