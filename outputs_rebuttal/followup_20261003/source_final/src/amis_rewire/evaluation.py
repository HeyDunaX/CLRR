"""Explicit generation precision shared by wrapped and unwrapped models."""
from contextlib import contextmanager
import torch
from transformers import Seq2SeqTrainer


@contextmanager
def generation_precision(model, precision):
    """Suspend Accelerate forward autocast only during generation."""
    restored = []
    for module in model.modules():
        # Inspect own attributes: wrappers delegate __getattr__ to their base.
        if '_original_forward' in module.__dict__:
            restored.append((module, module.forward))
            module.forward = module.__dict__['_original_forward']
    device = next(model.parameters()).device.type
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        if precision == 'fp32':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=precision == 'bf16' and device == 'cuda'):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
        for module, forward in restored:
            module.forward = forward


class PrecisionSeq2SeqTrainer(Seq2SeqTrainer):
    """Use one declared generation precision for all experiment methods."""
    def __init__(self, *args, generation_dtype='fp32', **kwargs):
        self.generation_dtype = generation_dtype
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        original_generate = model.generate
        had_override = 'generate' in model.__dict__

        def generate(*args, **kwargs):
            with generation_precision(model, self.generation_dtype):
                return original_generate(*args, **kwargs)

        model.generate = generate
        try:
            return super().prediction_step(model, inputs, prediction_loss_only,
                                           ignore_keys=ignore_keys, **gen_kwargs)
        finally:
            if had_override:
                model.generate = original_generate
            else:
                del model.generate
