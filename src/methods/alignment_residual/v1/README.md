# Alignment residual v1 (development)

Specification: `docs/research/CLRR_LSR_ADAPTER_FORMULATION.md`.
This is a full fine-tuning experiment with small added modules, not PEFT.
Novelty is unresolved; gradient confinement itself is prior art.

Four stable method IDs:

| ID | Fixed donor | LSR gradient |
|---|---|---|
| alignment-residual-capacity-v1 | yes | none; CE only |
| alignment-residual-fullgrad-v1 | yes | backbone + module |
| alignment-residual-confined-v1 | yes | module only |
| alignment-local-confined-v1 | none | module only |

At encoder layers 3–12: `h = u + .1*(q*sg(h[l-2]) + A(q*sg(h[l-2])-u))`.
`A(v)=up(GELU(down(RMS(v))))`; no biases/dropout, rank64, zero-up init.
Source auxiliary replay uses the CE source RNG. A parameter-constant encoder
view blocks auxiliary weight gradients while preserving block input Jacobians.
Each block has its own reentrant checkpoint closure and all donors are local
to that forward graph. Target encoding is no-grad with normal training dropout.
The capacity arm also consumes the target forward to match future dropout RNG.

This encoder loop is pinned to Transformers4.57.6 mBART, LayerDrop0, SDPA.
No changes to the original CLRR/LSR implementations. Custom checkpoints include
the entire wrapper and `alignment_residual_config.json`; restore with
`AlignmentResidualMT.from_pretrained`, not generic `AutoModel`.

Entry point remains `scripts/experiment.py --config ...`; see the experiment
runbook for build, CPU checks, remote GPU gates, and subsequent training.
