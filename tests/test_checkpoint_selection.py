"""Secondary selection survives rolling deletion and preserves earliest ties."""
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from experiment_runner.selection import SecondaryBestCheckpoint


class CheckpointSelectionGate(unittest.TestCase):
    def test_rotation_ties_restart_and_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            callback = SecondaryBestCheckpoint(folder)
            for step, score in [(1, 2.), (2, 1.), (3, 2.), (4, 3.)]:
                checkpoint = folder / f'checkpoint-{step}'
                checkpoint.mkdir()
                (checkpoint / 'pytorch_model.bin').write_bytes(bytes([step]))
                (checkpoint / 'config.json').write_text('{}')
                (checkpoint / 'optimizer.pt').write_text('excluded')
                state = SimpleNamespace(global_step=step, epoch=float(step),
                    log_history=[dict(step=step, **{'eval_chrf++': score})])
                callback.on_save(None, state, None)
                expected_step = 1 if step < 4 else 4
                with zipfile.ZipFile(folder / 'best_chrf.zip') as archive:
                    self.assertEqual(archive.read('pytorch_model.bin'), bytes([expected_step]))
                    self.assertNotIn('optimizer.pt', archive.namelist())
                    self.assertEqual(json.loads(archive.read('secondary_selection.json'))['step'], expected_step)
                shutil.rmtree(checkpoint)
                callback = SecondaryBestCheckpoint(folder)
                self.assertEqual(callback.best['step'], expected_step)

    def test_no_evaluation_never_selects_training_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = SecondaryBestCheckpoint(directory)
            callback.on_save(None, SimpleNamespace(global_step=1, epoch=1.,
                log_history=[{'step': 1, 'loss': .1}]), None)
            self.assertFalse((Path(directory) / 'best_chrf.zip').exists())

    def test_nonfinite_metric_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = SecondaryBestCheckpoint(directory)
            with self.assertRaises(AssertionError):
                callback.on_save(None, SimpleNamespace(global_step=1, epoch=1.,
                    log_history=[dict(step=1, **{'eval_chrf++': float('nan')})]), None)
