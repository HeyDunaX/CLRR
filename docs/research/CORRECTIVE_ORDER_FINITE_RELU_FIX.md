# Corrective-order: correction of numerical admission on ReLU

## Decision and scope

This amends the **admission protocol and virtual step**, not the six loss definitions.
It is not a successful training result or a novelty claim. Failed smooth-scaling
receipts remain unchanged. The new cohort is `amis_corrective_order_finite_relu_rtx`.

The original backbone, full parameter space, native dropout, exact unrolled
derivatives, probe sampling, FP32, B4/GA32, LR5e-5, ES10 validation BLEU and
six-arm controls remain fixed. No GELU replacement or detached inner gradients.

## Evidence and interpretation

- Pinned mBART config has `activation_function=relu`, with zero layerdrop.
  Receipt: `outputs/experiments/amis_corrective_order_small_step_rtx/backbone_numerical_audit.json`.
- Identical-mask full-backbone replay gave exactly zero endpoint difference and
  probe SKL. Receipt: same output, `order_preparation/NULL_REPLAY.json`.
- The prior numerical sweep did not pass its declared smooth-scaling gate.
  Receipt: same output, `order_preparation/NUMERICAL_PROGRESS.json`.
- `relu_counterexample.json` in the same output is a reproducible FP64
  counterexample: endpoint exponents 1.5 and mean-KL exponents approximately 3
  at the resolved finite scales, despite exact replay and correct composition.
  It disproves mandatory smooth scaling for all ReLU problems; it does not prove
  that rounding has zero contribution in full mBART.

For `ell_a=sum(ReLU(theta))`, `ell_b=-sum(theta)`, the two compositions differ
by eta for coordinates in `(-eta,0)`. With nonzero density at zero, the affected
fraction scales with eta. Endpoint L2 then scales as eta^(3/2), and independent
binary-probe mean KL as eta^3. On a fixed finite lattice this statement concerns
scales above lattice spacing, not the limit below the nearest kink.

Therefore a mandatory eta^2 endpoint / eta^4 KL criterion imports a smoothness
assumption that this architecture does not meet globally. The finite update maps
and piecewise autodiff objective remain defined away from their kink surfaces.
We make **no** claim of a verified smooth Lie-bracket limit or optimizer-step equivalence.

## Revised checks before any long training

1. CPU algebra, outer finite differences, offload equivalence, masking, exact
   checkpoint/resume, and audit/telemetry noninterference must pass.
2. Direct full-mBART TRAIN-only audit records fc1 sign changes under the same
   replay mask before/after a virtual update. Require observed crossings for
   this boundary explanation; report all 24 layers, exclude padded positions.
   A positive count is supporting evidence, not causal isolation from rounding.
3. Identical-mask null replay must produce exactly zero SKL and endpoint distance.
4. Use a **single a-priori eta=5e-5**, numerically equal to the fixed peak LR.
   This is a reproducibility convention, not a claim that virtual SGD matches
   AdamW displacement. No eta selection using validation or test.
5. Require finite positive CE, endpoint difference, raw SKL and normalized loss.
   Calibrate the original median 0.10 auxiliary/CE gradient-norm ratio on exactly
   three fixed TRAIN batches for order and symmetric-curvature separately;
   require finite positive outer gradient norms and unchanged base weights.
6. All six real Trainer GPU gates (2 optimizer steps, GA32, save/load/resume)
   must PASS on the same frozen bundle before any long training.

Keep `SKL/eta^4` in the formula: at fixed eta it is a scalar normalization;
the existing train-only coefficient absorbs its scale. We do not reinterpret
the normalization as evidence of eta^4 asymptotics. Small-step resolution and
gradient drift remain risks to investigate from logs, not guarantees of benefit.

## Training logs

- `microbatch_progress.json`: current forward/backward or completed microbatch.
- `microbatch_diagnostics.jsonl`: every completed microbatch, dropout seeds,
  data fingerprint (not raw text), CE/raw SKL/normalized and weighted loss,
  endpoint distance, time and available RAM/VRAM/disk.
- `training_events.jsonl`: optimizer events, gradient norm **after clipping**,
  LR, Trainer logs (including preclip norm when supplied), validation and saves.
- `training_failure.json`: exception and complete traceback for microbatch failure.
- Existing recipe/source hashes, run metadata, Trainer state, optimizer/scheduler
  and RNG checkpoint state remain available. Successful run upload includes all
  diagnostic logs; failed-run logs remain on RTX for retrieval. We do not claim
  to measure separate CE/auxiliary gradient norms at every training step.

## Literature and independent critique

Primary references read: [PyTorch autograd, non-differentiable functions](https://docs.pytorch.org/docs/2.6/notes/autograd.html)
and [mBART public config](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/raw/main/config.json).
Pinned runtime config was independently read from RTX's existing model cache.

Theory-only NotebookLM debate query `836e0bbcbdfb` is saved as
`outputs/experiments/amis_corrective_order_small_step_rtx/notebook_relu_gate_debate.json`.
The counterexample was verified independently in code. The answer is **not** a
proof or reliable literature citation: its claims that gradient update maps are
continuous, KL is coercive, all ReLU losses have piecewise-constant gradients,
and a universal slope interval can gate all ReLU networks are false or unsupported.
Those recommendations were rejected. Its caution about SGD vs AdamW and
fixed initial gradient calibration is retained as a limitation to measure.
