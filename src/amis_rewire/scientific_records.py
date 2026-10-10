"""Persist the declared routing schedule and replayable trajectory weights."""
import hashlib
import json
import os
import zipfile
from pathlib import Path

from transformers import TrainerCallback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


class ScientificRecordsCallback(TrainerCallback):
    def __init__(self, folder, alpha, weight, warmup_ratio, repo, prefix, run_name):
        self.folder = Path(folder)
        self.alpha = alpha
        self.weight = weight
        self.warmup_ratio = warmup_ratio
        self.repo = repo
        self.prefix = prefix.strip('/')
        self.run_name = run_name
        self.api = None
        if repo:
            from huggingface_hub import HfApi, get_token
            self.api = HfApi(token=get_token())

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        from .train import RoutingWarmupCallback
        rewire = RoutingWarmupCallback()._find_rewire(model)
        alpha = rewire.strength if rewire is not None else 0.0
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / 'routing_schedule.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'optimizer_step_before_update': state.global_step,
                                'alpha_train': alpha, 'lambda': self.weight,
                                'alpha_eval': self.alpha, 'planned_steps': state.max_steps}) + '\n')

    def on_save(self, args, state, control, **kwargs):
        save(self.folder / 'eval_loss_trace.json', [r for r in state.log_history if 'eval_loss' in r])
        save(self.folder / 'training_history.json', state.log_history)
        save(self.folder / 'selection_receipt.json', {'step': state.global_step, 'epoch': state.epoch,
             'best_checkpoint': state.best_model_checkpoint, 'best_metric': state.best_metric,
             'metric': 'raw-reference validation chrF++',
             'eval_beams': args.generation_num_beams, 'alpha_eval': self.alpha})
        epoch = round(state.epoch)
        # Fixed early / end-of-warmup / midpoint and actual last epoch, never test selected.
        if epoch not in (1, 4, 10) and not control.should_training_stop and state.global_step < state.max_steps:
            return
        source = self.folder / f'checkpoint-{state.global_step}'
        destination = self.folder / 'trajectory'
        destination.mkdir(exist_ok=True)
        archive = destination / f'epoch-{epoch:02d}-step-{state.global_step}.zip'
        names = [p for p in source.iterdir() if p.is_file() and
                 p.name not in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth')]
        assert any(p.name in ('pytorch_model.bin', 'model.safetensors') for p in names), source
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
            for p in names:
                z.write(p, p.name)
        receipt = {'epoch': state.epoch, 'global_step': state.global_step, 'archive': archive.name,
                   'archive_bytes': archive.stat().st_size, 'sha256': digest(archive),
                   'alpha_eval': self.alpha, 'lambda': self.weight,
                   'alpha_train_last_update': self.alpha * min(1.0, (state.global_step - 1) / max(1, int(state.max_steps * self.warmup_ratio))) if self.warmup_ratio else self.alpha,
                   'purpose': 'diagnostic weights; optimizer excluded; not resumable training checkpoint'}
        if self.api:
            remote = '/'.join((self.prefix, self.run_name, 'trajectory', archive.name))
            info = self.api.upload_file(path_or_fileobj=str(archive), path_in_repo=remote,
                                       repo_id=self.repo, repo_type='model')
            receipt.update(hf_repo=self.repo, hf_path=remote, hf_revision=info.oid)
            entries = self.api.get_paths_info(self.repo, [remote], revision=info.oid, repo_type='model')
            assert len(entries) == 1 and entries[0].size == receipt['archive_bytes']
            assert entries[0].lfs and entries[0].lfs.sha256 == receipt['sha256'], 'HF archive checksum mismatch'
        save(archive.with_suffix('.json'), receipt)

    def on_train_end(self, args, state, control, **kwargs):
        state.save_to_json(str(self.folder / 'final_trainer_state.json'))
        save(self.folder / 'training_history.json', state.log_history)
        save(self.folder / 'eval_loss_trace.json', [r for r in state.log_history if 'eval_loss' in r])
