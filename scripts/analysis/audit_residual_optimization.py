"""Audit C0 initialization, gradient flow and checkpoint replay; no training run."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import torch
from methods.alignment_residual.v1.module import ResidualBottleneck
from methods.alignment_residual.v1 import AlignmentResidualMT
from test_alignment_residual import native, batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info()
        assert free >= 4 * 1024**3, 'Insufficient free VRAM; do not disturb other jobs'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checks = {}
    module = ResidualBottleneck(16, 4, torch.Generator().manual_seed(10042)).to(device).double()
    value = torch.randn(2, 4, 16, device=device, dtype=torch.float64, requires_grad=True)
    probe = torch.randn_like(value)
    loss = (.1 * module(value) * probe).sum()
    loss.backward()
    checks['zero_init_output'] = bool(module(value).eq(0).all())
    checks['up_gradient_nonzero_at_init'] = float(module.up.grad.norm()) > 0
    checks['down_gradient_zero_at_init_expected'] = float(module.down.grad.norm()) == 0
    checks['input_gradient_zero_at_init_expected'] = float(value.grad.norm()) == 0
    initial_up_gradient = float(module.up.grad.norm())
    optimizer = torch.optim.AdamW(module.parameters(), lr=5e-5, weight_decay=0.)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (.1 * module(value) * probe).sum().backward()
    checks['down_gradient_nonzero_after_up_update'] = float(module.down.grad.norm()) > 0
    # Independent analytical and finite-difference checks at nonzero weights.
    module.zero_grad(set_to_none=True)
    (.1 * module(value) * probe).sum().backward()
    analytic = float(module.up.grad[0, 0])
    with torch.no_grad():
        original = module.up[0, 0].item()
        eps = 1e-6
        module.up[0, 0] = original + eps
        positive = float((.1 * module(value) * probe).sum())
        module.up[0, 0] = original - eps
        negative = float((.1 * module(value) * probe).sum())
        module.up[0, 0] = original
    finite = (positive - negative) / (2 * eps)
    checks['finite_difference_up'] = abs(analytic - finite) < 1e-8
    gradients = []
    for alpha in (.1, 1.):
        module.zero_grad(set_to_none=True)
        (alpha * module(value) * probe).sum().backward()
        gradients.append(module.up.grad.detach().clone())
    checks['alpha_gradient_scaling'] = torch.allclose(gradients[1], 10 * gradients[0], atol=1e-10, rtol=1e-10)
    # This scaling is a local derivative, not an Adam update or quality prediction.
    data = {k: v.to(device) for k, v in batch().items()}
    torch.manual_seed(42)
    model = AlignmentResidualMT(native(.15), 'alignment-residual-capacity-v1', rank=4, jepa_weight=0.).to(device)
    replay = copy.deepcopy(model)
    replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    results = []
    for item in (model, replay):
        torch.manual_seed(123)
        result = item(**data)
        result.loss.backward()
        results.append((result.logits.detach(), {n: p.grad.detach().clone() for n, p in item.named_parameters() if p.grad is not None}))
    checks['reentrant_forward'] = torch.allclose(results[0][0], results[1][0], atol=2e-5, rtol=2e-4)
    checks['reentrant_gradient_keys'] = results[0][1].keys() == results[1][1].keys()
    checks['reentrant_gradients'] = all(torch.allclose(g, results[1][1][n], atol=5e-5, rtol=2e-4) for n, g in results[0][1].items())
    checks['all_up_layers_receive_ce_gradient'] = all(float(m.up.grad.norm()) > 0 for m in model.get_encoder().residual_modules.values())
    checks['backbone_receives_ce_gradient'] = any(p.grad is not None and float(p.grad.norm()) > 0 for n, p in model.named_parameters() if '.residual_modules.' not in n)
    report = dict(passed=all(checks.values()), checks=checks, device=str(device), torch=torch.__version__,
                  initial_up_gradient_norm=initial_up_gradient,
                  scope='synthetic correctness audit; one module optimizer step, no dataset training; not evidence of optimization quality on trained checkpoints')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
