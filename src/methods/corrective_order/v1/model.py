"""Functional corrective-order consistency and its matched mathematical controls.

Virtual SGD is an auxiliary map, not a description of the real AdamW optimizer.
Every trainable backbone parameter participates. No virtual update is committed.
"""
from contextlib import contextmanager
from contextlib import nullcontext
import json
import math
from pathlib import Path
import weakref

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput

METHODS = {
    'corrective-order-ce-v1': 'ce',
    'corrective-order-rdrop-v1': 'rdrop',
    'corrective-order-consistency-v1': 'order',
    'corrective-order-curvature-v1': 'curvature',
}


@contextmanager
def offload_saved_tensors():
    """Exact CPU offload with a host-RAM reserve; never exhaust a shared host."""
    cache = {}
    def pack(tensor):
        if not tensor.is_cuda:
            return tensor.device, tensor.detach()
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor._version)
        cached = cache.get(key)
        if cached is not None and cached[0]() is tensor:
            return tensor.device, cached[1]
        if tensor.numel() * tensor.element_size() >= 2**20 and Path('/proc/meminfo').exists():
            available = next(int(line.split()[1])*1024 for line in
                Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
            if available < 6*2**30 + tensor.numel()*tensor.element_size():
                raise RuntimeError('HOST_RAM_ADMISSION_FAILED: preserve at least 6 GiB for other users')
        cpu = tensor.detach().to('cpu')
        cache[key] = (weakref.ref(tensor), cpu)
        return tensor.device, cpu
    def unpack(saved):
        device, tensor = saved
        return tensor.to(device)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def symmetric_kl(left, right, labels, chunk=16):
    """Full-vocabulary token mean, double log-prob arithmetic, padding excluded."""
    if left.shape != right.shape or left.shape[:-1] != labels.shape:
        raise ValueError('Expected matching [batch,time,vocab] logits and labels')
    valid = labels.reshape(-1).ne(-100)
    count = valid.sum()
    if count == 0:
        raise ValueError('No supervised probe tokens')
    left, right = left.flatten(0, 1), right.flatten(0, 1)
    total = left.new_zeros((), dtype=torch.float64)
    def reduce(a, b, mask):
        a, b = F.log_softmax(a.double(), -1), F.log_softmax(b.double(), -1)
        return (.5 * ((a.exp() - b.exp()) * (a - b)).sum(-1) * mask).sum()
    for start in range(0, len(valid), chunk):
        args = (left[start:start+chunk], right[start:start+chunk], valid[start:start+chunk])
        total = total + (checkpoint(reduce, *args, use_reentrant=False)
                         if torch.is_grad_enabled() else reduce(*args))
    return total / count


def compose_paths(parameters, loss, eta, seeds):
    """loss(state,seed) -> scalar. Exact finite maps, fully differentiable."""
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError('eta must be finite and positive')
    a, b = seeds
    la, lb = loss(parameters, a), loss(parameters, b)
    def step(state, objective):
        gradients = torch.autograd.grad(objective, tuple(state.values()),
                                        create_graph=True, retain_graph=True)
        return {key: value - eta * gradient
                for (key, value), gradient in zip(state.items(), gradients)}
    pa, pb = step(parameters, la), step(parameters, lb)
    pab = step(pa, loss(pa, b))
    pba = step(pb, loss(pb, a))
    return (la + lb) / 2, pab, pba, pa, pb


@contextmanager
def replay(module, seed, training):
    """Explicit mode/RNG context is recreated INSIDE checkpoint recomputation."""
    flags = [(m, m.training) for m in module.modules()]
    devices = sorted({p.device.index for p in module.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices), sdpa_kernel(SDPBackend.MATH):
        try:
            torch.manual_seed(int(seed))
            # Dropout OFF for probes, even if invoked later during backward.
            module.train(training)
            yield
        finally:
            for child, flag in flags:
                child.training = flag


class CorrectiveOrderMT(nn.Module):
    accepts_loss_kwargs = False

    def __init__(self, base_model, method, eta=.01, coefficient=1., multiplier=1., beta=1.25,
                 offload='cpu'):
        super().__init__()
        self.base_model = base_model
        self.method, self.mode = method, METHODS[method]
        self.eta, self.coefficient = float(eta), float(coefficient)
        self.multiplier, self.beta = float(multiplier), float(beta)
        if offload not in ('cpu', 'disk'):
            raise ValueError('Unknown exact saved-tensor storage')
        self.offload = offload
        self.view_checkpointing = False
        self.last_diagnostics = {}
        if not all(math.isfinite(v) and v >= 0 for v in (coefficient, multiplier, beta)):
            raise ValueError('Invalid objective coefficients')
        if eta <= 0 or not math.isfinite(eta):
            raise ValueError('Invalid virtual step')

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        if (gradient_checkpointing_kwargs or {}).get('use_reentrant', False):
            raise ValueError('Exact virtual SGD requires non-reentrant checkpointing')
        # Per-layer closures would replay after functional_call has restored the
        # real parameters. Checkpoint the WHOLE functional call with explicit
        # parameter tensor inputs instead, including all virtual states.
        self.base_model.gradient_checkpointing_disable()
        self.view_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()
        self.view_checkpointing = False

    def _evaluate(self, state, batch, seed, training, scalar=False):
        names = tuple(state)
        inputs = {k: v for k, v in batch.items()
                  if k in ('input_ids', 'attention_mask', 'labels', 'decoder_input_ids', 'decoder_attention_mask')}
        def run(*values):
            with replay(self.base_model, seed, training):
                out = functional_call(self.base_model, dict(zip(names, values)), (),
                    dict(**inputs, use_cache=False, return_dict=True), tie_weights=True)
                return out.loss if scalar else out.logits
        if self.view_checkpointing and torch.is_grad_enabled():
            return checkpoint(run, *state.values(), use_reentrant=False, preserve_rng_state=True)
        return run(*state.values())

    def objective(self, batch, probe=None, seeds=None, auxiliary=True):
        state = {k: p for k, p in self.base_model.named_parameters() if p.requires_grad}
        if not state or any(not p.requires_grad for p in self.base_model.parameters()):
            raise ValueError('This version requires the full trainable backbone')
        seeds = tuple(seeds) if seeds is not None else tuple(torch.randint(0, 2**31-1, (2,)).tolist())
        diagnostics = dict(mode=self.mode, views=2, eta=self.eta, dropout_seeds=list(seeds),
            coefficient=self.coefficient * self.multiplier,
            virtual_steps_committed=0, inner_gradients_detached=False,
            virtual_parameters=sum(p.numel() for p in state.values()))
        if self.mode in ('ce', 'rdrop') or not auxiliary:
            if self.mode == 'rdrop' and auxiliary:
                left = self._evaluate(state, batch, seeds[0], True)
                right = self._evaluate(state, batch, seeds[1], True)
                ce = .5 * (F.cross_entropy(left.flatten(0, 1), batch['labels'].flatten()) +
                           F.cross_entropy(right.flatten(0, 1), batch['labels'].flatten()))
                penalty = symmetric_kl(left, right, batch['labels'])
            else:
                ce = .5 * sum(self._evaluate(state, batch, seed, True, scalar=True) for seed in seeds)
                penalty = ce * 0
            weight = self.beta if self.mode == 'rdrop' else 0.
        else:
            if probe is None:
                raise ValueError('Order objective requires an independently sampled TRAIN probe')
            ce, pab, pba, pa, pb = compose_paths(state,
                lambda p, s: self._evaluate(p, batch, s, True, scalar=True), self.eta, seeds)
            with torch.no_grad():
                distance = sum((pab[k].detach().double() - pba[k].detach().double()).square().sum()
                               for k in state).sqrt()
            diagnostics['parameter_order_distance'] = float(distance)
            if self.mode == 'curvature':
                # Symmetric second-order displacement vs simultaneous first-
                # order drift. NOT pab vs midpoint (that merely rescales order).
                left_state = {k: (pab[k] + pba[k]) / 2 for k in state}
                right_state = {k: pa[k] + pb[k] - state[k] for k in state}
            else:
                left_state, right_state = pab, pba
            left = self._evaluate(left_state, probe, 0, False)
            right = self._evaluate(right_state, probe, 0, False)
            raw = symmetric_kl(left, right, probe['labels'])
            penalty = raw / self.eta**4
            weight = self.coefficient * self.multiplier
            diagnostics['raw_probe_skl'] = float(raw.detach())
            del pab, pba, pa, pb, left_state, right_state
        loss = ce + weight * penalty
        diagnostics.update(ce=float(ce.detach()), auxiliary=float(penalty.detach()),
                           weighted_auxiliary=float((weight * penalty).detach()))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite corrective-order objective')
        self.last_diagnostics = diagnostics
        return loss, ce, penalty

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        probe = {k.removeprefix('probe_'): kwargs.pop(k)
                 for k in list(kwargs) if k.startswith('probe_')}
        batch = dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        batch = {k: v for k, v in batch.items() if v is not None}
        if not self.training or labels is None:
            with sdpa_kernel(SDPBackend.MATH):
                return self.base_model(**batch)
        # Offload saved autograd tensors, including functional checkpoint inputs,
        # without changing arithmetic, batch, parameter subspace or gradients.
        context = self.offload_context() if input_ids.is_cuda else nullcontext()
        with context:
            loss, _, _ = self.objective(batch, probe or None)
        return Seq2SeqLMOutput(loss=loss)

    def offload_context(self):
        if self.offload == 'disk':
            from .storage import disk_saved_tensors
            root = Path('/workspace/ura/thanhdat/projects/CLRR')
            if not Path(__file__).resolve().is_relative_to(root):
                raise ValueError('RTX disk spill is restricted to the CLRR project')
            return disk_saved_tensors(root/'tmp/corrective-order-autograd')
        return offload_saved_tensors()

    def generate(self, *args, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return self.base_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return self.base_model.load_state_dict(state_dict, strict=strict, **kwargs)

    def method_metadata(self):
        return dict(version=1, method=self.method, mode=self.mode, eta=self.eta,
            coefficient=self.coefficient, multiplier=self.multiplier, beta=self.beta,
            trainable_subspace='ALL backbone parameters', virtual_optimizer='unclipped SGD',
            real_optimizer='recipe AdamW', virtual_steps_committed=0, stop_gradient=False,
            masks='two native masks; exact replay across paths', probe='TRAIN only; dropout OFF',
            penalty='full-vocabulary token-mean symmetric KL / eta^4',
            checkpointing='whole-functional-call non-reentrant', sdpa_kernel='MATH',
            saved_tensors=self.offload,
            extra_parameters=0, original_clrr=False, original_lsr=False)

    def save_pretrained(self, folder, state_dict=None, **kwargs):
        if state_dict is not None:
            state_dict = {k.removeprefix('base_model.'): v for k, v in state_dict.items()}
        self.base_model.save_pretrained(folder, state_dict=state_dict, **kwargs)
        Path(folder, 'corrective_order_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'corrective_order_config.json').read_text())
        if saved != self.method_metadata():
            raise ValueError('Corrective-order checkpoint definition changed')
