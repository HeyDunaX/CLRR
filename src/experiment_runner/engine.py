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
from methods.registry import TWO_METHODS, ALIASES, CANDIDATES, ORIGINAL_ALIASES, TOKEN_METHODS, CONTRASTIVE_METHODS, INTERACTION_METHODS, SEQUENCE_METHODS, DEPTH_METHODS, wrap_method
from methods.decoder_residual_bridge.v1 import DecoderResidualBridge, METHODS as DRB_METHODS
from methods.alignment_residual import METHODS as ALIGNMENT_RESIDUAL_METHODS, model_class as residual_model_class
from methods.difference_preserving.v1 import METHODS as DIFFERENCE_METHODS
from methods.difference_preserving.v2.model import METHODS as RELATIONAL_METHODS
from methods.transition_lsr import METHODS as TRANSITION_METHODS
from methods.learned_strength import METHODS as STRENGTH_METHODS, STATE_KEY as STRENGTH_STATE_KEY
from methods.paired_route import METHODS as PAIRED_ROUTE_METHODS
from methods.decoupled_consistency import METHODS as CONSISTENCY_METHODS
from methods.prefix_consistency import METHODS as PREFIX_METHODS
from methods.continuation_consistency import METHODS as CMC_METHODS
from methods.corrective_order import METHODS as ORDER_METHODS
from methods.virtual_corrective_order.v1.model import METHODS as VIRTUAL_ORDER_METHODS
from methods.gold_ordered_consistency import METHODS as GOC_METHODS

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
        assert torch.cuda.is_available()
        if protocol().get('hardware_profile', 'a100-standard') == 'rtx6000-fp32':
            approved = Path('/workspace/ura/thanhdat/projects/CLRR').resolve()
            assert ROOT.resolve() == approved and Path(sys.prefix).resolve().is_relative_to(approved / 'conda')
            assert torch.cuda.get_device_name(0) == 'Quadro RTX 6000'
            assert torch.cuda.mem_get_info()[0] >= protocol()['minimum_free_gpu_mib'] * 2**20
        else:
            assert torch.cuda.is_bf16_supported() and 'A100' in torch.cuda.get_device_name(0)
            assert 35 < torch.cuda.get_device_properties(0).total_memory / 2**30 < 45
    return versions


def wait_for_gpu():
    """Resource admission only; never alter batch/precision or other processes."""
    if protocol().get('hardware_profile') != 'rtx6000-fp32' or not protocol().get('wait_for_gpu', False):
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Quadro RTX 6000'
    threshold = protocol()['minimum_free_gpu_mib']
    started = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] // 2**20
        if free >= threshold:
            write(OUT / 'RESOURCE_ADMISSION.json', dict(status='READY', free_mib=free,
                required_mib=threshold, waited_seconds=time.time()-started, updated_unix=time.time()))
            return
        state = dict(status='WAITING_FOR_GPU_MEMORY', free_mib=free, required_mib=threshold,
            waited_seconds=time.time()-started, updated_unix=time.time())
        write(OUT / 'RESOURCE_ADMISSION.json', state)
        print('RESOURCE_WAIT', json.dumps(state), flush=True)
        time.sleep(30)


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
    elif ALIASES.get(method, method) in CANDIDATES or method in TOKEN_METHODS | CONTRASTIVE_METHODS | INTERACTION_METHODS | SEQUENCE_METHODS | DEPTH_METHODS.keys() | ALIGNMENT_RESIDUAL_METHODS.keys() | DRB_METHODS.keys() | DIFFERENCE_METHODS.keys() | RELATIONAL_METHODS.keys() | TRANSITION_METHODS.keys() | STRENGTH_METHODS.keys() | PAIRED_ROUTE_METHODS.keys() | CONSISTENCY_METHODS.keys() | PREFIX_METHODS.keys() | CMC_METHODS.keys() | TWO_METHODS | VIRTUAL_ORDER_METHODS | ORDER_METHODS.keys() | GOC_METHODS.keys() or method in ('original-clrr', 'original-lsr', 'original-full'):
        model = wrap_method(base, spec)
        metadata = {'candidate': method, 'definition': spec.get('definition', method),
                    'extra_parameters': 0}
        if method in STRENGTH_METHODS:
            route = model.base_model if STRENGTH_METHODS[method] else model
            metadata.update(extra_parameters=route.extra_parameter_count(), learned_strength=route.method_metadata())
        if method in PAIRED_ROUTE_METHODS:
            metadata['paired_route'] = model.method_metadata()
        if method in CONSISTENCY_METHODS:
            metadata['decoupled_consistency'] = model.method_metadata()
        if method in TWO_METHODS:
            metadata['two_method_round1'] = model.method_metadata()
        if method in GOC_METHODS:
            metadata['gold_ordered_consistency'] = model.method_metadata()
        if method in CMC_METHODS:
            metadata['continuation_consistency'] = model.method_metadata()
        if method in PREFIX_METHODS:
            metadata['prefix_consistency'] = model.method_metadata()
        if method in TRANSITION_METHODS:
            metadata['transition_objective'] = model.method_metadata()
        if method in RELATIONAL_METHODS:
            metadata['relational_objective'] = model.method_metadata()
        if method in DIFFERENCE_METHODS:
            metadata['difference_objective'] = model.method_metadata()
            metadata['pairing_control'] = DIFFERENCE_METHODS[method][0]
        if method in DRB_METHODS:
            metadata.update(extra_parameters=model.extra_parameter_count(), drb=model.method_metadata())
        if method in ALIGNMENT_RESIDUAL_METHODS:
            metadata.update(extra_parameters=model.extra_parameter_count(),
                            alignment_residual=model.method_metadata())
        if method in DEPTH_METHODS:
            metadata.update(extra_parameters=model.extra_parameter_count(),
                donor_layer=model.donor_layer, decoder_layer=model.decoder_layer,
                correction_mode=DEPTH_METHODS[method], donor_value_scaling='detached-per-source-Frobenius',
                attention_implementation=model.config._attn_implementation)
        if method in INTERACTION_METHODS:
            metadata['interaction_objective'] = model.method_metadata()
        if method in SEQUENCE_METHODS:
            metadata['sequence_objective'] = model.method_metadata()
    else:
        assert method == 'baseline'
        model, metadata = base, {'loss': 'CE only'}
    if method in VIRTUAL_ORDER_METHODS:
        metadata['virtual_order'] = model.method_metadata()
    if method in ORDER_METHODS:
        from methods.corrective_order.v1.data import apply_calibration
        apply_calibration(model, OUT, spec)
        metadata['corrective_order'] = model.method_metadata()
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
    if spec['method'] in DIFFERENCE_METHODS.keys() | RELATIONAL_METHODS.keys():
        model.special_token_ids = tuple(sorted(set(tokenizer.all_special_ids)))
    if spec['method'] in DRB_METHODS:
        model.special_ids = sorted(tokenizer.all_special_ids)
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
    reload_selected = folder is not None
    if folder is None and spec['method'] in SEQUENCE_METHODS:
        from .sequence import initialization_folder
        folder = initialization_folder()
    if folder and spec['method'] in DRB_METHODS:
        model = DecoderResidualBridge.from_pretrained(folder)
        for key, value in {'method':spec['method'], 'rank':spec['adapter_rank'],
                'module_seed':spec['module_seed'], 'donor_layer':spec['donor_layer'],
                'decoder_layer':spec['decoder_layer']}.items():
            assert model.method_metadata()[key] == value, ('DRB checkpoint recipe mismatch', key)
        assert model.config._attn_implementation == protocol()['attention_implementation']
        metadata = {'restored_drb':True, 'extra_parameters':model.extra_parameter_count()}
    elif folder and spec['method'] in ALIGNMENT_RESIDUAL_METHODS:
        model = residual_model_class(spec['method']).from_pretrained(folder)
        assert model.config._attn_implementation == protocol()['attention_implementation']
        for key, value in {'method': spec['method'], 'rank': spec['adapter_rank'],
                'module_seed': spec['module_seed'], 'distance': spec['distance'],
                'strength': spec['strength'], 'jepa_weight': spec['jepa_weight']}.items():
            assert model.method_metadata()[key] == value, ('Checkpoint recipe mismatch', key)
        metadata = {'restored_alignment_residual': True, 'extra_parameters': model.extra_parameter_count()}
    elif folder and spec['method'] in DEPTH_METHODS:
        from methods.depth_difference.v2 import ValueScaledDepthDifference
        model = ValueScaledDepthDifference.from_pretrained(folder)
        assert model.config._attn_implementation == protocol()['attention_implementation'], 'Depth checkpoint backend differs from locked recipe'
        assert model.method_id == spec['method'] and model.donor_layer == spec['donor_layer'] and model.decoder_layer == spec['decoder_layer']
        metadata = {'restored_depth': True, 'extra_parameters': model.extra_parameter_count()}
    elif folder and (Path(folder) / 'adapter_config.json').exists():
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
        backend = {'attn_implementation': protocol()['attention_implementation']} if protocol().get('attention_implementation') else {}
        base, info = AutoModelForSeq2SeqLM.from_pretrained(str(folder) if folder else spec['model_id'],
            revision=None if folder else spec['model_revision'], output_loading_info=True, **backend)
        if folder and spec['method'] in STRENGTH_METHODS:
            assert info.get('unexpected_keys') == [STRENGTH_STATE_KEY], info
            info['unexpected_keys'] = []
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
        model, metadata = wrap_model(base, spec)
        if folder and spec['method'] in STRENGTH_METHODS:
            route = model.base_model if STRENGTH_METHODS[spec['method']] else model
            route.restore_scalars(folder)
        if reload_selected and spec['method'] in SEQUENCE_METHODS:
            saved = json.loads((Path(folder) / 'sequence_risk_config.json').read_text())
            assert saved == model.method_metadata(), 'Sequence checkpoint/recipe mismatch'
        if folder and spec['method'] in INTERACTION_METHODS:
            saved = json.loads((Path(folder) / 'interaction_config.json').read_text())
            assert saved == model.method_metadata(), 'Interaction checkpoint/recipe mismatch'
        if folder and spec['method'] in INTERACTION_METHODS:
            saved = json.loads((Path(folder) / 'interaction_config.json').read_text())
            assert saved == model.method_metadata(), 'Interaction checkpoint/recipe mismatch'
    if folder and spec['method'] in PAIRED_ROUTE_METHODS.keys() | CONSISTENCY_METHODS.keys() | PREFIX_METHODS.keys() | CMC_METHODS.keys() | TWO_METHODS | VIRTUAL_ORDER_METHODS | ORDER_METHODS.keys() | GOC_METHODS.keys():
        model.verify_checkpoint_metadata(folder)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(folder) if folder else spec['model_id'],
                                                  revision=None if folder else spec['model_revision'], use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(folder) if folder else spec['model_id'],
                                                  revision=None if folder else spec['model_revision'], use_fast=False)
    configure_languages(model, tokenizer, spec)
    if spec['method'] in TRANSITION_METHODS:
        if folder:
            model.verify_checkpoint_metadata(folder)
        metadata['transition_objective'] = model.method_metadata()
    if spec['method'] in RELATIONAL_METHODS:
        if folder:
            model.verify_checkpoint_metadata(folder)
        metadata['relational_objective'] = model.method_metadata()
    if spec['method'] in DIFFERENCE_METHODS:
        # Pooling IDs must be configured before validating the persisted recipe.
        if folder:
            model.verify_checkpoint_metadata(folder)
        metadata['difference_objective'] = model.method_metadata()
    if spec['method'] in DRB_METHODS:
        metadata['drb'] = model.method_metadata()
    if spec['method'] in TOKEN_METHODS | CONTRASTIVE_METHODS:
        model.special_token_ids = tuple(sorted(set(tokenizer.all_special_ids)))
        metadata['special_token_ids'] = list(model.special_token_ids)
    if spec['method'] in CONTRASTIVE_METHODS:
        metadata['alignment_objective'] = {'mode': model.mode, 'weight': model.alignment_weight,
            'temperature': model.temperature, 'target_detached': True, 'auxiliary_dropout': False,
            'negative_pool': 'microbatch only', 'jepa_weight': model.jepa_weight}
    if spec['method'] in TOKEN_METHODS:
        metadata['token_objective'] = {'epsilon': .10, 'iterations': 200, 'weight': .10,
            'target_detached': True, 'plan_detached': True, 'reduction': 'mean_per_sentence'}
    model.config.use_cache = False
    actual_attention = model.config._attn_implementation
    if protocol().get('attention_implementation'):
        assert actual_attention == protocol()['attention_implementation'], 'Loaded attention backend differs from recipe'
    metadata['attention_implementation'] = actual_attention
    return model, tokenizer, metadata


def dataset(split, tokenizer, spec, technical_peers=False):
    raw = pd.read_csv(DATA / f'{split}.csv', dtype=str, keep_default_na=False)
    def tokenize(batch):
        inputs = tokenizer([spec['source_prefix'] + s for s in batch['source']],
                           max_length=spec['max_source_length'], truncation=True)
        inputs['labels'] = tokenizer(text_target=batch['target'], max_length=spec['max_target_length'], truncation=True)['input_ids']
        return inputs
    encoded = Dataset.from_pandas(raw, preserve_index=False).map(tokenize, batched=True, remove_columns=['source', 'target'])
    if split == 'train' and spec['method'] in ORDER_METHODS:
        from methods.corrective_order.v1.data import add_probes
        encoded = add_probes(encoded, spec['seed'])
    if split == 'train' and spec['method'] in DIFFERENCE_METHODS:
        if technical_peers:
            # Arbitrary train-only peers stress code, not semantic correctness.
            if len(raw) < 2:
                raise ValueError('Technical fixture requires at least two rows')
            peers = {i: (i + 1) % len(raw) for i in range(len(raw))}
        else:
            from methods.difference_preserving.v1.data import read_pairs
            pairing = plan()['pairing']
            pair_file = ROOT / pairing['manifest']
            if sha256(pair_file) != pairing['sha256']:
                raise ValueError('Reviewed pair manifest changed')
            peers = read_pairs(pair_file, DATA / 'train.csv', raw.to_dict('records'),
                               DIFFERENCE_METHODS[spec['method']][0], spec['seed'])
        for key in ('input_ids', 'attention_mask', 'labels'):
            values = encoded[key]
            encoded = encoded.add_column('peer_' + key, [values[peers.get(i, i)] for i in range(len(raw))])
        encoded = encoded.add_column('pair_eligible', [i in peers for i in range(len(raw))])
    if split == 'train' and spec['method'] in SEQUENCE_METHODS and spec['risk_weight'] > 0:
        from .sequence import read_pool
        records = read_pool(tokenizer, spec)
        assert len(records) == len(raw)
        encoded = encoded.add_column('candidate_labels', [r['labels'] for r in records])
        encoded = encoded.add_column('candidate_costs', [r['costs'] for r in records])
    return raw, encoded


def collator(model, tokenizer, spec):
    if spec['method'] in ORDER_METHODS:
        from methods.corrective_order.v1.data import ProbeCollator
        return ProbeCollator(tokenizer, model)
    if spec['method'] in DIFFERENCE_METHODS:
        from methods.difference_preserving.v1.data import DifferenceCollator
        return DifferenceCollator(tokenizer, model, pad_to_multiple_of=8)
    if spec['method'] in SEQUENCE_METHODS:
        from methods.sequence_risk.v1.data import SequenceRiskCollator
        return SequenceRiskCollator(tokenizer, model, pad_to_multiple_of=8)
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)


class ControlTrainer(PrecisionSeq2SeqTrainer):
    """Save adapters as adapters and Middle weights without wrapper key prefixes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        if getattr(model, 'method', None) in VIRTUAL_ORDER_METHODS:
            started = time.perf_counter()
            result = super().training_step(model, inputs, num_items_in_batch)
            with (Path(self.args.output_dir) / 'microbatch_diagnostics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=self.state.global_step,
                    epoch=self.state.epoch, seconds=time.perf_counter()-started,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    **model.last_diagnostics)) + '\n')
            return result
        if getattr(model, 'method', None) in ORDER_METHODS:
            from methods.corrective_order.v1.telemetry import record_training_step
            return record_training_step(self, super().training_step, model, inputs, num_items_in_batch)
        return super().training_step(model, inputs, num_items_in_batch)

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
        self.started_unix = time.time()

    @staticmethod
    def _bar(current, total, width=20):
        if not total or total <= 0:
            return '[' + ('-' * width) + ']'
        ratio = max(0., min(1., float(current) / float(total)))
        filled = int(round(ratio * width))
        return '[' + ('#' * filled) + ('-' * (width - filled)) + f'] {ratio*100:5.1f}%'

    def on_log(self, args, state, control, logs=None, **kwargs):
        diagnostics = getattr(kwargs.get('model'), 'last_diagnostics', None)
        if diagnostics:
            with (self.folder / 'loss_components.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=state.global_step, epoch=state.epoch,
                    scope='last training microbatch; not epoch mean', **diagnostics)) + '\n')
        elapsed = max(1e-6, time.time() - self.started_unix)
        eta = ((elapsed / state.global_step) * max(0, state.max_steps - state.global_step)
               if state.global_step and state.max_steps else None)
        bar = self._bar(state.global_step, state.max_steps)
        write(self.folder / 'progress.json', {'id': self.ident, 'status': 'TRAINING',
              'progress_bar': bar, 'epoch': state.epoch, 'step': state.global_step,
              'max_steps': state.max_steps, 'elapsed_seconds': round(elapsed, 1),
              'eta_seconds': round(eta, 1) if eta is not None else None,
              'best_metric': state.best_metric, 'updated_unix': time.time(), 'logs': logs,
              'patience_counter': self.early.early_stopping_patience_counter if self.early else None})
        if logs and 'loss' in logs:
            ep = f"{state.epoch:.2f}" if state.epoch is not None else '?'
            best = f"{state.best_metric:.4f}" if state.best_metric is not None else 'n/a'
            eta_str = f"{eta/60:.1f}m" if eta is not None else '?'
            print(f"PROGRESS {self.ident:<16} {bar} step {state.global_step}/{state.max_steps} "
                  f"(ep {ep}) loss={logs['loss']:.4f} best={best} ETA={eta_str}", flush=True)

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
        load_best_model_at_end=not gate, metric_for_best_model=None if gate else recipe.get('selection_metric', 'eval_chrf++'),
        greater_is_better=True, bf16=recipe['bf16'], fp16=recipe.get('fp16', False), tf32=recipe['tf32_training'],
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


def gpu_gate(spec, technical_only=False):
    if technical_only and spec['method'] not in DIFFERENCE_METHODS:
        raise ValueError('Technical-only gate supports difference prototypes only')
    if spec['method'] in DIFFERENCE_METHODS and not technical_only:
        from methods.difference_preserving.v1.data import require_scientific_admission
        require_scientific_admission(plan())
    validate_environment()
    set_seed(spec['seed'])
    folder = OUT / ('technical_gates' if technical_only else 'gates') / spec['id']
    pass_name = 'TECHNICAL_PASS.json' if technical_only else 'PASS.json'
    assert not (folder / pass_name).exists(), 'Gate already exists; verify bundle before rerun'
    folder.mkdir(parents=True, exist_ok=True)
    model, tokenizer, meta = load_model(spec)
    raw, ds = dataset('train', tokenizer, spec, technical_peers=technical_only)
    # Include long examples and use the exact training microbatch/accumulation.
    indices = sorted(range(len(raw)), key=lambda i: len(raw.iloc[i]['source']) + len(raw.iloc[i]['target']), reverse=True)
    if spec['method'] in DIFFERENCE_METHODS:
        # Include reviewed pairs in the gate sample for sparse-pair corpora.
        # This changes gate data only, never the real training sampler.
        eligible = ds['pair_eligible']
        paired = [i for i in indices if eligible[i]]
        if not paired:
            raise ValueError('GPU gate requires reviewed eligible train pairs')
        indices = paired + [i for i in indices if not eligible[i]]
    sample = ds.select(indices[:spec['microbatch'] * spec['accumulation'] * 2])
    data_collator = collator(model, tokenizer, spec)
    callbacks = []
    if spec['method'] in ORDER_METHODS:
        from methods.corrective_order.v1.telemetry import TrainingTelemetry
        callbacks.append(TrainingTelemetry(folder))
    trainer = ControlTrainer(model=model, args=arguments(folder, spec, gate=True), processing_class=tokenizer,
                             data_collator=data_collator, train_dataset=sample, generation_dtype='fp32',
                             callbacks=callbacks)
    started = time.time()
    trainer.train()
    assert trainer.state.global_step == 2
    paired_checks = {}
    if spec['method'] in VIRTUAL_ORDER_METHODS:
        diagnostic = dict(model.last_diagnostics)
        assert diagnostic['order'] > 0 and np.isfinite(diagnostic['weighted_order'])
        assert diagnostic['virtual_steps_committed'] == 0
        paired_checks = dict(virtual_order=True, diagnostics=diagnostic)
    if spec['method'] in ORDER_METHODS:
        diagnostic = dict(model.last_diagnostics)
        assert np.isfinite(diagnostic['ce']) and np.isfinite(diagnostic['auxiliary'])
        assert diagnostic['virtual_steps_committed'] == 0
        if model.mode != 'ce':
            assert diagnostic['auxiliary'] > 0
        assert not model.base_model.is_gradient_checkpointing
        paired_checks = dict(corrective_order=True, diagnostics=diagnostic,
            all_parameters_virtual=True, full_higher_order_gradient=True,
            whole_view_checkpointing=model.view_checkpointing, saved_tensors=model.offload+'-offload')
    if spec['method'] in PAIRED_ROUTE_METHODS or spec['method'] in CONSISTENCY_METHODS or spec['method'] in PREFIX_METHODS or spec['method'] in CMC_METHODS or spec['method'] in TWO_METHODS or spec['method'] in GOC_METHODS:
        diagnostic = dict(model.last_diagnostics)
        assert np.isfinite(diagnostic['ce']) and np.isfinite(diagnostic['kl'])
        assert diagnostic['checkpoint_context_empty'] and not model._checkpoint_context
        if spec['method'] in TWO_METHODS or spec['method'] in CONSISTENCY_METHODS or spec['method'] in PREFIX_METHODS or spec['method'] in CMC_METHODS or spec['method'] in GOC_METHODS or PAIRED_ROUTE_METHODS[spec['method']][1]:
            if spec['method'] == 'continuation-control-variate-v2':
                assert diagnostic['dense_kl'] > 0, 'Dense consistency must be active'
                # The unbiased control-variate estimate may be negative.
            else:
                assert diagnostic['kl'] > 0, 'Consistency must be active on real dropout views'
        if spec['method'] in GOC_METHODS:
            assert 0. <= diagnostic['winner_view1_fraction'] <= 1.
            assert 0. <= diagnostic['tie_fraction'] <= 1.
        if spec['method'] in CMC_METHODS:
            assert diagnostic['channel_separation'] > 1e-6, 'Continuation channel is degenerate'
            assert 0 < diagnostic['selected_mass'] <= 1
        paired_checks = dict(paired_route_active=True, diagnostics=diagnostic)
        if spec['method'] in PREFIX_METHODS:
            assert np.isfinite(diagnostic['path'])
            if diagnostic['mode'] != 'disabled':
                assert diagnostic['path'] > 0 and diagnostic['path_weight'] == .10
    if spec['method'] in STRENGTH_METHODS:
        route = model.base_model if STRENGTH_METHODS[spec['method']] else model
        assert torch.isfinite(route.route_strengths).all()
        assert route.route_strengths.detach().abs().sum() > 0, 'Route scalars did not learn'
    relational_checks = {}
    if spec['method'] in TRANSITION_METHODS:
        diagnostic = dict(model.last_diagnostics)
        assert diagnostic['edges'] == 10 and diagnostic['source_requires_grad']
        assert not diagnostic['target_requires_grad']
        assert np.isfinite(diagnostic['auxiliary_loss']) and diagnostic['auxiliary_loss'] > 0
        assert diagnostic['source_norm_min'] > 1e-6 and diagnostic['target_norm_min'] > 1e-6
        relational_checks = dict(transition_auxiliary_active=True, diagnostics=diagnostic)
    if spec['method'] in RELATIONAL_METHODS:
        diagnostic = dict(model.last_diagnostics)
        assert diagnostic['valid_pairs'] >= 2, 'Real gate has insufficient relational pairs'
        assert np.isfinite(diagnostic['relational_loss']) and diagnostic['relational_loss'] > 0
        relational_checks = dict(auxiliary_active=True, diagnostics=diagnostic)
    assert np.isfinite(trainer.state.log_history[-1]['train_loss'])
    assert (folder / 'checkpoint-2/optimizer.pt').exists() and (folder / 'checkpoint-2/rng_state.pth').exists()
    trainer.state.best_model_checkpoint = str(folder / 'checkpoint-2')
    trainer._load_best_model()
    best = folder / 'export'
    trainer.save_model(str(best))
    probe = data_collator([sample[i] for i in range(2)])
    probe = {k: v.to('cuda') for k, v in probe.items()}
    expected = probe_logits(model, probe)
    depth_checks = {}
    if spec['method'] == 'decision-residual-memory-v1':
        # Exercise the actual Trainer validation route, including collated
        # labels, before admitting a full DRM training run.
        _, generated, labels = trainer.prediction_step(model,probe,False,
            num_beams=1,max_new_tokens=8)
        assert generated is not None and labels is not None
        depth_checks['trainer_generation_with_labels'] = True
    reload_source = best
    if protocol().get('retain_secondary_chrf', False):
        from .selection import SecondaryBestCheckpoint
        trainer.state.log_history.append(dict(step=2, **{'eval_chrf++': 1.0}))
        SecondaryBestCheckpoint(folder).on_save(trainer.args, trainer.state, trainer.control)
        trainer.state.log_history.pop()
        reload_source = folder / 'secondary_reload'
        restore_archive(folder / 'best_chrf.zip', reload_source)
        depth_checks['secondary_archive_restore'] = True
    if spec['method'] in DEPTH_METHODS or spec['method'] in ALIGNMENT_RESIDUAL_METHODS or spec['method'] in DRB_METHODS or spec['method'] in DIFFERENCE_METHODS or spec['method'] in RELATIONAL_METHODS or spec['method'] in TRANSITION_METHODS or spec['method'] in STRENGTH_METHODS or spec['method'] in PAIRED_ROUTE_METHODS or spec['method'] in CONSISTENCY_METHODS or spec['method'] in PREFIX_METHODS or spec['method'] in CMC_METHODS or spec['method'] in TWO_METHODS or spec['method'] in ORDER_METHODS or spec['method'] in VIRTUAL_ORDER_METHODS or spec['method'] in GOC_METHODS:
        inputs = {key: probe[key] for key in ('input_ids','attention_mask')}
        with torch.no_grad(), generation_precision(model,'fp32'):
            cached = model.generate(**inputs,num_beams=4,max_new_tokens=8,use_cache=True)
            uncached = model.generate(**inputs,num_beams=4,max_new_tokens=8,use_cache=False)
        assert torch.equal(cached,uncached), 'Depth-v2 cached generation mismatch'
        depth_checks['beam4_prefix_replay_requests_match' if spec['method'] == 'decision-residual-memory-v1' else 'beam4_cache_matches'] = True
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    restored, _, _ = load_model(spec, reload_source); restored.to('cuda')
    actual = probe_logits(restored, probe)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
    if spec['method'] == 'depth-difference-value-scaled-v2' or spec['method'] in ALIGNMENT_RESIDUAL_METHODS or spec['method'] in DRB_METHODS or spec['method'] in DIFFERENCE_METHODS or spec['method'] in RELATIONAL_METHODS or spec['method'] in TRANSITION_METHODS or spec['method'] in STRENGTH_METHODS or spec['method'] in PAIRED_ROUTE_METHODS or spec['method'] in CONSISTENCY_METHODS or spec['method'] in PREFIX_METHODS or spec['method'] in CMC_METHODS or spec['method'] in TWO_METHODS or spec['method'] in ORDER_METHODS or spec['method'] in VIRTUAL_ORDER_METHODS or spec['method'] in GOC_METHODS:
        # Shared Trainer resume path: reload step1 weights plus optimizer/scheduler/RNG,
        # repeat step2, and compare against uninterrupted step2.
        del restored; gc.collect(); torch.cuda.empty_cache()
        set_seed(spec['seed'])
        resumed, _, _ = load_model(spec, folder/'checkpoint-1')
        resumed_args = arguments(folder/'resume_check',spec,gate=True)
        resumed_trainer = ControlTrainer(model=resumed,args=resumed_args,processing_class=tokenizer,
            data_collator=collator(resumed,tokenizer,spec),train_dataset=sample,generation_dtype='fp32')
        resumed_trainer.train(resume_from_checkpoint=str(folder/'checkpoint-1'))
        assert resumed_trainer.state.global_step == 2
        torch.testing.assert_close(probe_logits(resumed,probe),expected,rtol=1e-5,atol=2e-5)
        depth_checks['optimizer_scheduler_rng_resume_matches'] = True
        del resumed_trainer,resumed;gc.collect();torch.cuda.empty_cache()
        restored = None
    receipt = {'passed': True, 'id': spec['id'], 'gpu': torch.cuda.get_device_name(0),
               'technical_only': technical_only,
               'semantic_admission': False if technical_only else None,
               'peer_policy': 'arbitrary cyclic train-row fixture' if technical_only else 'recipe',
               'bf16': protocol()['bf16'], 'fp16': protocol().get('fp16', False),
               'checkpointing': protocol()['gradient_checkpointing'], 'microbatch': spec['microbatch'],
               'accumulation': spec['accumulation'], 'steps': 2, 'save_reload_logits_match': True,
               'trainer_best_checkpoint_reload': True,
               'optimizer_rng_present': True, 'seconds': time.time() - started, 'parameters': meta,
               'max_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
               'bundle_sha256': sha256(OUT / 'bundle_manifest.json')}
    if depth_checks: receipt['depth_checks'] = depth_checks
    if relational_checks: receipt['relational_checks'] = relational_checks
    if paired_checks: receipt['paired_checks'] = paired_checks
    write(folder / pass_name, receipt)
    del restored, probe; gc.collect(); torch.cuda.empty_cache()
    for target in [best, *folder.glob('secondary_reload'), *folder.glob('checkpoint-*'), *folder.glob('resume_check')]:
        assert target.resolve().is_relative_to(folder.resolve())
        shutil.rmtree(target)
    if (folder / 'best_chrf.zip').exists():
        (folder / 'best_chrf.zip').unlink()
    print('GPU_GATE_PASS', spec['id'], json.dumps(receipt), flush=True)


def upload_and_verify(folder, spec, names):
    api = HfApi(token=os.environ['HF_TOKEN'])
    assert not api.model_info(REPO).private
    files = {spec['remote_prefix'] + '/' + n: folder / n for n in names}
    for remote, local in files.items():
        api.upload_file(repo_id=REPO, path_in_repo=remote, path_or_fileobj=str(local))
    return verify_remote_files(api, REPO, files)


def train_run(spec, resume=None):
    if spec['method'] in GOC_METHODS:
        from methods.gold_ordered_consistency.v1.validation import require_admission
        require_admission(plan(), ROOT)
    if spec['method'] in VIRTUAL_ORDER_METHODS:
        record = plan()['virtual_order_calibration']
        path = ROOT / record['path']
        assert sha256(path) == record['sha256']
        calibration = json.loads(path.read_text())
        assert calibration['passed'] and calibration['optimizer_steps'] == 0
        assert not calibration['test_used'] and not calibration['validation_used']
        assert spec['order_coefficient'] == calibration['coefficient']
    if spec['method'] in ORDER_METHODS:
        from methods.corrective_order.v1.data import require_admission
        require_admission(OUT, plan())
    if spec['method'] in TWO_METHODS:
        admission = json.loads((OUT/'two_method_probe'/spec['id']/'ADMISSION.json').read_text())
        assert admission['passed'] and admission['bundle_sha256'] == sha256(OUT/'bundle_manifest.json')
        assert admission['optimizer_steps'] == 0 and not admission['test_used']
        queue = json.loads((OUT/'TWO_METHOD_GATES_COMPLETE.json').read_text())
        assert queue['bundle_sha256'] == sha256(OUT/'bundle_manifest.json')
        assert spec['id'] in queue['admitted']
    if spec['method'] in CMC_METHODS:
        if spec['method'] in ('continuation-control-variate-v2','continuation-sampled-direct-v2'):
            record=plan()['estimator_admission']
            path=ROOT/record['path']
            assert sha256(path)==record['sha256'], 'Estimator diagnostic changed'
            admission=json.loads(path.read_text())
            assert admission['passed'] and admission['optimizer_steps']==0 and not admission['test_used']
        else:
            admission = json.loads((OUT/'continuation_probe/ADMISSION.json').read_text())
            assert admission['passed'] and admission['bundle_sha256'] == sha256(OUT/'bundle_manifest.json')
        for other in plan()['runs']:
            check = json.loads((OUT/'gates'/other['id']/'PASS.json').read_text())
            assert check['passed'] and check['bundle_sha256'] == sha256(OUT/'bundle_manifest.json')
    if plan().get('prefix_preparation'):
        raise ValueError('Scientific preparation bundle cannot start long training')
    if spec['method'] in PREFIX_METHODS:
        admission = plan()['prefix_admission']
        for key in ('diagnostic', 'calibration'):
            record = admission[key]
            path = ROOT / record['path']
            assert sha256(path) == record['sha256'], 'Prefix admission receipt changed'
            receipt = json.loads(path.read_text())
            assert receipt['passed']
        if spec['method'] == 'calibrated-consistency-v1':
            assert spec['consistency_beta'] == receipt['beta_cal']
    if spec['method'] in DIFFERENCE_METHODS:
        from methods.difference_preserving.v1.data import require_scientific_admission
        require_scientific_admission(plan())
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
    if resume and spec['method'] in DIFFERENCE_METHODS.keys() | RELATIONAL_METHODS.keys() | TRANSITION_METHODS.keys() | PAIRED_ROUTE_METHODS.keys() | CONSISTENCY_METHODS.keys() | PREFIX_METHODS.keys() | CMC_METHODS.keys() | TWO_METHODS | VIRTUAL_ORDER_METHODS | ORDER_METHODS.keys() | GOC_METHODS.keys():
        model.verify_checkpoint_metadata(resume)
    raw_val, val_ds = dataset('validation', tokenizer, spec)
    _, train_ds = dataset('train', tokenizer, spec)
    metadata = {'run': spec, 'protocol': plan()['protocol'], 'data': plan()['data'], 'method': meta,
                'runtime': versions, 'gpu': torch.cuda.get_device_name(0),
                'bundle_sha256': sha256(OUT / 'bundle_manifest.json'), 'git_head': plan()['code_git_head'],
                'test_used_for_selection': False, 'status': 'TRAINING', 'started_unix': time.time()}
    write(folder / 'run_metadata.json', metadata)
    write(folder / 'training_receipt.json', metadata)
    early = (EarlyStoppingCallback(early_stopping_patience=protocol()['patience'])
             if protocol().get('early_stopping_enabled', True) else None)
    callbacks = ([early] if early else []) + [Progress(folder, spec['id'], early)]
    if spec['method'] in ORDER_METHODS:
        from methods.corrective_order.v1.telemetry import TrainingTelemetry
        callbacks.append(TrainingTelemetry(folder))
    if protocol().get('retain_secondary_chrf', False):
        from .selection import SecondaryBestCheckpoint
        callbacks.append(SecondaryBestCheckpoint(folder))
    trainer = ControlTrainer(model=model, args=arguments(folder, spec), processing_class=tokenizer,
        data_collator=collator(model, tokenizer, spec),
        train_dataset=train_ds, eval_dataset=val_ds, generation_dtype='fp32',
        compute_metrics=build_compute_metrics(tokenizer, raw_val['target'].tolist(), tokenize=protocol()['bleu_tokenizer'],
            include_chrf0=plan().get('log_validation_chrf0', False)),
        callbacks=callbacks)
    write(folder / 'resolved_training_arguments.json', trainer.args.to_dict())
    try:
        trainer.train(resume_from_checkpoint=resume)
    except Exception as error:
        if spec['method'] in ORDER_METHODS or spec['method'] in VIRTUAL_ORDER_METHODS:
            import traceback
            from methods.corrective_order.v1.telemetry import append
            failure = dict(status='TRAINING_FAILED', id=spec['id'],
                step=trainer.state.global_step, epoch=trainer.state.epoch,
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            append(folder, 'training_events.jsonl', failure)
            write(folder/'training_failure.json', failure)
            trainer.state.save_to_json(str(folder/'failed_trainer_state.json'))
        raise
    trainer.state.save_to_json(str(folder / 'final_trainer_state.json'))
    best_checkpoint = Path(trainer.state.best_model_checkpoint)
    shutil.copy2(best_checkpoint / 'trainer_state.json', folder / 'selected_trainer_state.json')
    scores = predict(trainer, val_ds, tokenizer, raw_val, 'validation', folder, spec['validation_beams'])
    selection_key = protocol().get('selection_metric', 'eval_chrf++').removeprefix('eval_')
    assert abs(scores[selection_key] - trainer.state.best_metric) < 1e-6, (scores, trainer.state.best_metric)
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
    if protocol().get('retain_secondary_chrf', False):
        secondary = json.loads((folder / 'secondary_selection.json').read_text())
        assert secondary['metric'] == 'eval_chrf++'
        secondary['archive_sha256'] = sha256(folder / 'best_chrf.zip')
        write(folder / 'secondary_selection.json', secondary)
        names += ['best_chrf.zip', 'secondary_selection.json']
    if spec['method'] in ORDER_METHODS:
        shutil.copy2(OUT / 'order_preparation/CALIBRATION.json', folder / 'order_calibration.json')
        names.append('order_calibration.json')
        names += [n for n in ('microbatch_diagnostics.jsonl', 'microbatch_progress.json',
                              'training_events.jsonl', 'training_failure.json') if (folder/n).exists()]
    if spec['method'] in VIRTUAL_ORDER_METHODS:
        shutil.copy2(ROOT / plan()['virtual_order_calibration']['path'], folder / 'order_calibration.json')
        names.append('order_calibration.json')
        names += [n for n in ('microbatch_diagnostics.jsonl','training_failure.json','training_events.jsonl')
                  if (folder/n).exists()]
    if (folder / 'loss_components.jsonl').exists():
        names.append('loss_components.jsonl')
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
    targets = [export, folder / 'best.zip', *folder.glob('checkpoint-*')]
    if protocol().get('retain_secondary_chrf', False):
        targets.append(folder / 'best_chrf.zip')
    for target in targets:
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
    args.eval_strategy = 'no'
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
