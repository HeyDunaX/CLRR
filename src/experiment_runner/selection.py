"""Keep a second validation-selected model without retaining optimizer copies."""
import json
import math
import zipfile
from pathlib import Path

from transformers import TrainerCallback


class SecondaryBestCheckpoint(TrainerCallback):
    """Snapshot the earliest best secondary metric before checkpoint rotation."""

    def __init__(self, folder, metric='eval_chrf++'):
        self.folder = Path(folder)
        self.metric = metric
        self.receipt = self.folder / 'secondary_selection.json'
        self.best = json.loads(self.receipt.read_text()) if self.receipt.exists() else None

    def on_save(self, args, state, control, **kwargs):
        metrics = next((r for r in reversed(state.log_history)
                        if r.get('step') == state.global_step and self.metric in r), None)
        if metrics is None:
            return
        score = float(metrics[self.metric])
        assert math.isfinite(score), metrics
        if self.best is not None and score <= self.best['score']:
            return
        source = self.folder / f'checkpoint-{state.global_step}'
        files = [p for p in source.iterdir() if p.is_file() and p.name not in
                 ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'scaler.pt')]
        assert any(p.name in ('pytorch_model.bin', 'model.safetensors') for p in files)
        target = self.folder / 'best_chrf.zip'
        record = dict(metric=self.metric, score=score, step=state.global_step,
                      epoch=state.epoch, metrics=metrics, ties='earliest best',
                      archive=target.name, resumable=False)
        temporary = target.with_suffix('.pending.zip')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED) as archive:
            for p in files:
                archive.write(p, p.name)
            archive.writestr('secondary_selection.json', json.dumps(record, indent=2))
        temporary.replace(target)
        self.best = record
        temporary = self.receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.best, indent=2), encoding='utf-8')
        temporary.replace(self.receipt)
