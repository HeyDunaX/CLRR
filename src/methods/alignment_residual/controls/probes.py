"""Temporary source-forward interventions; no parameter or v1 source edits."""
from contextlib import contextmanager
import torch


@contextmanager
def residual_intervention(model, condition, content_mask, seed=42, statistics=None):
    if condition not in ('all-on', 'module-off', 'donor-off', 'donor-shuffled'):
        raise ValueError(condition)
    encoder = model.get_encoder()
    handles, current, shifts, forwards = [], {}, {}, []
    generator = torch.Generator().manual_seed(seed)
    for key, module in encoder.residual_modules.items():
        index = int(key)
        layer = encoder.layers[index]
        original = layer.forward
        def capture(*args, index=index, original=original, **kwargs):
            output = original(*args, **kwargs)
            current[index] = output[0]
            return output
        # ResidualEncoder calls block.forward directly inside its checkpoint
        # proxy, so ordinary layer forward hooks would never run.
        forwards.append((layer, original))
        layer.forward = capture

        def before(_module, inputs, index=index):
            value, = inputs
            if condition in ('all-on', 'module-off'):
                return
            hidden = current[index]
            donor = value + hidden
            replacement = torch.zeros_like(donor)
            if condition == 'donor-shuffled':
                replacement = donor.clone()
                for row in range(len(donor)):
                    ids = content_mask[row].nonzero().flatten()
                    perm = torch.randperm(len(ids), generator=generator).to(ids.device)
                    replacement[row, ids] = donor[row, ids[perm]]
            shifts[index] = replacement - donor
            return (replacement - hidden,)

        def after(_module, inputs, output, index=index):
            if statistics is not None:
                hidden = current[index].detach()
                donor = inputs[0].detach() + hidden
                weight = content_mask[..., None].to(hidden.dtype)
                def norm(value):
                    return ((value.float().square()*weight).sum((1,2)) /
                        (weight.sum((1,2))*value.shape[-1]).clamp_min(1)).sqrt()
                denominator = norm(hidden).clamp_min(1e-12)
                statistics.append(dict(layer=index+1,
                    correction_over_current=(norm(output.detach())/denominator).cpu().tolist(),
                    residual_over_current=(norm(encoder.residual_strength*(donor+output.detach()))/denominator).cpu().tolist()))
            if condition == 'module-off':
                return torch.zeros_like(output)
            if condition in ('donor-off', 'donor-shuffled'):
                # Original encoder adds donor+correction. Compensate its fixed
                # donor term as well as changing the module's donor input.
                return output + shifts.pop(index)
        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for layer, original in forwards:
            layer.forward = original
