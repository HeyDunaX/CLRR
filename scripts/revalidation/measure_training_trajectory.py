"""Run the existing trainer with observational probes that restore training RNG/state."""
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/revalidation'))
from amis_rewire import train
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from measure_checkpoint_mechanisms import (LayerMeasurements, extract_representations,
    spectrum_diagnostics, measure_gradients, read_csv, save_json)


class TrajectoryCallback(train.ConsoleMetricsCallback):
    def __init__(self, folder, cli):
        self.folder, self.cli = folder, cli

    def on_save(self, args, state, control, model=None, processing_class=None, tokenizer=None, **kwargs):
        epoch = int(round(state.epoch))
        late = control.should_training_stop or state.global_step >= state.max_steps
        if epoch not in (1, 5) and not late:
            return
        destination = self.folder / f'epoch-{epoch:02d}-step-{state.global_step}'
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / 'DONE.json').exists(): return
        print('TRAJECTORY_PROBE_START', epoch, state.global_step, flush=True)
        tokenizer = processing_class or tokenizer
        assert tokenizer is not None
        rows = read_csv(Path(self.cli.data_dir) / 'validation.csv')
        # on_save is after optimizer zero_grad; probes may not discard pending gradients.
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())
        mode, use_cache = model.training, model.config.use_cache
        raw = model
        rewired = None
        while isinstance(raw, (CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM)):
            if isinstance(raw, CrossLayerResidualRewire): rewired = raw
            raw = raw.base_model
        checkpointing = raw.is_gradient_checkpointing
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        collector = None
        try:
            model.gradient_checkpointing_disable()
            model.config.use_cache = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model.eval()
            collector = LayerMeasurements(model, rewired)
            representations = extract_representations(model, tokenizer, rows, 256, collector)
            collector.close()
            collapse = {phase: spectrum_diagnostics(value) for phase, value in representations.items()}
            save_json(destination / 'collapse.json', collapse)
            residual = [r['residual_to_current_rms'] for r in collector.rows if r['phase'] == 'source' and r['layer'] >= 3]
            save_json(destination / 'residual_summary.json', {'mean': float(np.mean(residual)),
                'max': float(np.max(residual)), 'validation_rows': len(rows), 'layers': list(range(3,13))})
            import csv
            with (destination / 'residual_per_example.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(collector.rows[0]))
                writer.writeheader(); writer.writerows(collector.rows)
            method = {'baseline': 'baseline', 'clrr-enc': 'clrr', 'jepa': 'lsr', 'jepa-clrr-enc': 'full'}[self.cli.method]
            gradients = measure_gradients(model, tokenizer, rows, 256, method)
            save_json(destination / 'gradients.json', gradients)
            save_json(destination / 'DONE.json', {'epoch': epoch, 'step': state.global_step,
                'method': method, 'training_seed': self.cli.seed,
                'roles': [*(['early'] if epoch == 1 else []), *(['middle_fixed_epoch5'] if epoch == 5 else []),
                          *(['late_actual_training_endpoint'] if late else [])],
                'split': 'validation', 'selection_independent': True,
                'auxiliary_counterfactual': method in ('baseline', 'clrr'),
                'scope': 'observational trajectory; fixed validation rows/batches; not causal attribution'})
        finally:
            if collector is not None: collector.close()
            model.zero_grad(set_to_none=True)
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
            model.config.use_cache = use_cache
            model.train(mode)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
            random.setstate(python_rng); np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
            torch.cuda.empty_cache()
        print('TRAJECTORY_PROBE_COMPLETE', epoch, state.global_step, flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--probe-output-dir', required=True)
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    cli = train.parse_args()
    base_callback = TrajectoryCallback
    train.ConsoleMetricsCallback = lambda: base_callback(Path(own.probe_output_dir), cli)
    train.main()


if __name__ == '__main__': main()
