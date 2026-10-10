"""Explicit generation precision shared by wrapped and unwrapped models."""
from contextlib import contextmanager
import torch
from transformers import Seq2SeqTrainer


@contextmanager
def generation_precision(model, precision):
    """Suspend Accelerate forward autocast only during generation."""
    if precision not in ('fp32', 'bf16'):
        raise ValueError(f'Unsupported generation precision: {precision}')
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
    def __init__(self, *args, generation_dtype='fp32', eval_rewire_strength=None, **kwargs):
        if generation_dtype not in ('fp32', 'bf16'):
            raise ValueError(f'Unsupported generation precision: {generation_dtype}')
        self.generation_dtype = generation_dtype
        self.eval_rewire_strength = eval_rewire_strength
        super().__init__(*args, **kwargs)

    def _find_rewire(self, model):
        from .modeling import CrossLayerResidualRewire
        m = model
        visited = set()
        while m is not None and id(m) not in visited:
            visited.add(id(m))
            if isinstance(m, CrossLayerResidualRewire):
                return m
            m = getattr(m, 'base_model', None)
        return None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.eval_rewire_strength is not None:
            rewire = self._find_rewire(self.model)
            if rewire is not None:
                old_strength = rewire.strength
                rewire.strength = self.eval_rewire_strength
                try:
                    return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
                finally:
                    rewire.strength = old_strength
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        if self.eval_rewire_strength is not None:
            rewire = self._find_rewire(self.model)
            if rewire is not None:
                old_strength = rewire.strength
                rewire.strength = self.eval_rewire_strength
                try:
                    return super().predict(test_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
                finally:
                    rewire.strength = old_strength
        return super().predict(test_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

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

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        import os
        from transformers.modeling_utils import PreTrainedModel
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        m = self.model
        while not isinstance(m, PreTrainedModel) and hasattr(m, "base_model"):
            m = m.base_model
        if isinstance(m, PreTrainedModel):
            m.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
        else:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        if getattr(self, "processing_class", None) is not None:
            self.processing_class.save_pretrained(output_dir)
        elif getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        from pathlib import Path
        import shutil
        receipt = Path(self.args.output_dir) / "training_receipt.json"
        target = Path(output_dir) / receipt.name
        if receipt.exists() and receipt.resolve() != target.resolve():
            shutil.copyfile(receipt, target)
