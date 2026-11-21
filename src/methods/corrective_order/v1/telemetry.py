"""Append-only diagnostics for every microbatch and optimizer/validation event."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import torch
from transformers import TrainerCallback


def append(folder, name, record):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(updated_unix=time.time(), **record)
    with (folder/name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, default=str)+'\n')


def resources(folder):
    data = dict(disk_free_gib=shutil.disk_usage(folder).free/2**30)
    if Path('/proc/meminfo').exists():
        data['host_available_gib'] = next(int(s.split()[1])/2**20 for s in
            Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if torch.cuda.is_available():
        data.update(gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
            gpu_reserved_gib=torch.cuda.memory_reserved()/2**30,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    return data


def record_training_step(trainer, base_step, model, inputs, num_items_in_batch):
    folder = Path(trainer.args.output_dir)
    trainer._order_microbatch = getattr(trainer, '_order_microbatch', 0)+1
    record = dict(step_before_update=trainer.state.global_step,
        microbatch_in_session=trainer._order_microbatch,
        epoch=trainer.state.epoch, session=getattr(trainer, '_order_session', None),
        input_shape=list(inputs['input_ids'].shape),
        supervised_tokens=int(inputs['labels'].ne(-100).sum()),
        batch_sha256=hashlib.sha256(b''.join(inputs[k].detach().cpu().contiguous().numpy().tobytes()
            for k in ('input_ids', 'labels'))).hexdigest())
    if record['session'] is None:
        trainer._order_session = record['session'] = str(time.time_ns())
    started = time.time()
    folder.mkdir(parents=True, exist_ok=True)
    progress = folder/'microbatch_progress.json'
    progress.write_text(json.dumps(dict(status='FORWARD_BACKWARD', started_unix=started, **record)))
    try:
        result = base_step(model, inputs, num_items_in_batch)
    except BaseException as error:
        failure = dict(record, status='FAILED', error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), seconds=time.time()-started)
        append(folder, 'training_events.jsonl', failure)
        (folder/'training_failure.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        raise
    diagnostic_model = trainer.accelerator.unwrap_model(model, keep_torch_compile=False)
    diagnostic = dict(getattr(diagnostic_model, 'last_diagnostics', {}))
    ce, weighted = diagnostic.get('ce', 0), diagnostic.get('weighted_auxiliary', 0)
    record.update(status='BACKWARD_COMPLETE', seconds=time.time()-started,
        diagnostics=diagnostic, weighted_auxiliary_to_ce=weighted/max(abs(ce), 1e-30),
        resources=resources(folder), scope='one microbatch; unscaled objective components')
    append(folder, 'microbatch_diagnostics.jsonl', record)
    progress.write_text(json.dumps(record, default=str), encoding='utf-8')
    return result


class TrainingTelemetry(TrainerCallback):
    def __init__(self, folder):
        self.folder = Path(folder)

    def event(self, kind, state, **data):
        append(self.folder, 'training_events.jsonl', dict(event=kind, step=state.global_step,
            epoch=state.epoch, **data))

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        with torch.no_grad():
            norms = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
            norm = float(torch.stack(norms).norm()) if norms else 0.
        self.event('PRE_OPTIMIZER', state, gradient_norm_after_clipping=norm,
            learning_rates=[g['lr'] for g in optimizer.param_groups], resources=resources(self.folder))

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.event('TRAINER_LOG', state, logs=logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.event('VALIDATION', state, metrics=metrics, best_metric=state.best_metric,
            best_checkpoint=state.best_model_checkpoint)

    def on_save(self, args, state, control, **kwargs):
        self.event('CHECKPOINT', state, checkpoint=f'checkpoint-{state.global_step}',
            best_metric=state.best_metric, best_checkpoint=state.best_model_checkpoint)
