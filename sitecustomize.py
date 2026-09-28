import torch

# Fix PyTorch 2.6 default weights_only=True breaking Hugging Face Trainer checkpoint resuming (rng_state.pth)
_orig_load = torch.load

def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)

torch.load = _patched_load

try:
    from numpy._core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([_reconstruct])
except Exception:
    pass
