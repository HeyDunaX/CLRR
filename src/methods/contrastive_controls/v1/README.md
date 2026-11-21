# Contrastive controls v1

Matched ablations of P3 (`ordinary-full-v1`). Reuse the existing ordinary
InfoNCE wrapper, tokenizer masks and shared experiment engine.

| ID | Objective | Encoder route |
|---|---|---|
| `nce-only-v1` | CE + .1 ordinary InfoNCE | None |
| `clrr-nce-v1` | CE + .1 ordinary InfoNCE | Original detached CLRR |
| `attached-ordinary-full-v1` | CE + .1 original LSR + .1 ordinary InfoNCE | Source-attached CLRR |

Every route uses distance2, strength.1, post-rewire donors. Target branches
remain stop-gradient, including in the source-attached control. No extra
parameters; negative pool remains microbatch4 with temperature.1.

## Attached route and checkpointing

The attached control keeps the donor in the autograd graph. On Transformers
4.57.6 mBART, each checkpointed block runs without the route hook, then adds
the donor **outside** the checkpoint operation. This gives the donor an
explicit gradient path and keeps backward recompute independent of mutable
cache state. Auxiliary dropout-off views and evaluation use the normal hook.

Forward values match detached P3 for identical weights. Checkpointing on/off
must preserve gradients, including LSR and InfoNCE source paths. The original
CLRR and P3 implementations are unchanged; this is not a new novelty claim.

Gate: `conda run -n clrr python tests/test_contrastive_controls.py`.
Recipe: `configs/experiments/amis_contrastive_controls_rtx.json`.
