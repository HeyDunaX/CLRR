# CMC estimator audit — conditional diagnostic before training

Status: AUTHOR AUTHORIZED; frozen-checkpoint diagnostic first. No claim of new
ACL novelty. Control variates are established statistical machinery (REBAR,
https://arxiv.org/abs/1703.07370). This investigates a CMC implementation
hypothesis, not a new semantic objective. Existing scores and recipes stay intact.

## Hypothesis and estimators

Let D_t be direct symmetric KL of the same two native dropout views, and C_t
the existing detached hybrid-channel continuation symmetric KL. Old estimator:
C_J. Candidate: mean_t D_t + C_J - D_J, with J uniform among supervised tokens.
For a batch, preserve length-weighted sampling and the total valid-token divisor.
Expected surrogate gradient is unchanged; variance reduction is not guaranteed.
Identity channel must recover dense W for each sampling outcome. Do not clamp
negative estimates: that changes the expectation.

## Locked diagnostic (before GPU result)

- Script: `scripts/analysis/analyze_cmc_gradient_estimator.py`.
- Fixed pinned W42 archive from `configs/experiments/amis_continuation_consistency_rtx.json`.
- TRAIN only; no validation or test; zero optimizer steps.
- Four short TRAIN sentences: scan a fixed random permutation (seed271828),
  accept target token length5–10 and source length<=64. Include EOS/language
  tokens exactly as the existing supervised mask. This intentionally narrow
  subset cannot establish variance on long sequences or training microbatch4.
- Two fixed dropout seeds19421/19422, giving eight conditions. Within each
  condition, enumerate ALL positions using the SAME two graphs and detached
  channels. All trainable parameters, not only logits or selected layers.
- Existing RTX FP32, SDPA, TF32 off, reentrant gradient checkpointing. Diagnostic
  microbatch1 is for conditional variance measurement, not a training change.
- Compute trace variance of g_C(J) and g_C(J)-g_D(J). Adding the exactly computed
  dense gradient changes the mean but not conditional sampling variance.
- Operational admission, locked before result: pooled new/old trace variance
  <=0.8 AND reduction in at least6/8 conditions. Otherwise HOLD, no new training.
  This is an exploratory engineering screen, not statistical significance.
- CPU gate: enumerate six sampling outcomes for unequal sentence lengths2/3;
  verify loss/gradient expectation and identity-channel gradients; include a
  counterexample where variance increases.
- GPU>=20000MiB free, exclusive project lock. Never terminate other processes.
- No public upload of this diagnostic payload by default. Only small JSON/logs
  return locally. Transient restored W copy is removed after completion; the
  pinned source checkpoint remains on HF.

## Conditional next stage

If the diagnostic passes, two new Amis seed42 validation-only pilot arms may
be prepared: plain unscaled sampled direct SKL, and control-variate CMC. They
require CPU/GPU correctness gates before training. Preserve the locked training
recipe (FP32/SDPA, microbatch4, accumulation32, LR5e-5, max20epochs, ES10,
validation BLEU beam1). Reuse W/CMC only after provenance matching. No test
selection, automatic sweep, or ACL novelty claim follows from a variance PASS.

Pilot screening locked before training: primary CV-CMC minus reused W42
validation BLEU >=+0.30; CV must also beat the plain sampled-direct arm and
old CMC42. One-seed screening is exploratory. Report all arms even on failure.
Both new arms execute the same continuation probes to match their compute;
sampled-direct does not use their result in its loss. Dense direct-KL overhead
in CV is additional and must be reported. Training starts from the same pinned
pretrained mBART, NOT the W checkpoint used in the frozen diagnostic.

Implementation: `src/methods/continuation_consistency/v2/` keeps v1 model code
unchanged. Tests cover identity-channel equivalence to dense W including all
parameter gradients, reentrant checkpoint gradient replay, and native inference
after save/reload. The independent CPU audit covers unequal sentence lengths
and expectation across every sampling outcome.

## Commands

```powershell
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 scripts/analysis/analyze_cmc_gradient_estimator.py --cpu
```

Remote, inside `/workspace/ura/thanhdat/projects/CLRR/`, under project lock:

```bash
source environment.sh
python -u scripts/analysis/analyze_cmc_gradient_estimator.py
```

Launch/result receipts: `outputs/experiments/amis_cmc_estimator_gradient_audit/`.
