# Learned-strength CLRR: checkpoint diagnosis

Status: COMPLETE, 2026-11-10. Exploratory Amis seed42 validation diagnosis.
No training, optimizer update, test evaluation, or recipe change was performed.

## 1. What the pilot established

| Selected model | Validation BLEU | chrF++ | BLEU minus baseline |
|---|---:|---:|---:|
| Reused RTX baseline | 17.6823 | 14.7030 | — |
| P1 learned-strength CLRR + CE | 16.6360 | 14.1009 | -1.0463 |
| P2 learned-strength CLRR + original cosine LSR(.1) | 17.3459 | 14.4554 | -0.3364 |

Both candidates completed 20 epochs/720 steps and failed the prospective
baseline +0.30 BLEU/nondecreasing chrF++ advancement gate. Adding LSR to P1
improved BLEU by 0.7099, but this does not establish an interaction without
a matched original LSR-only arm. Historical baseline SDPA provenance was
reconstructed rather than captured contemporaneously; retain that caveat.

## 2. Measurements on the selected checkpoints

Both exact public HF archives were retrieved at their recorded revisions and
SHA256 checked. The trained numerical source hashes were checked. FP32,
SDPA, TF32 disabled, eval mode, dropout and gradient checkpointing disabled.
Rows: `default_rng(42).permutation(576)[:64]`, shared across both models.
Gradient probes use the first 16 selected rows in four batches of four.

| Measurement | P1 | P2 |
|---|---:|---:|
| Mean absolute learned scalar | 0.00336943 | 0.00282396 |
| Mean route injection / recipient norm | 0.24125% | 0.17294% |
| Range of layer mean injection ratios | 0.00291–0.44721% | 0.00095–0.36234% |
| Route ON mean CE | 3.08839494 | 2.92986489 |
| Route OFF mean CE | 3.08955460 | 2.92993202 |
| CE OFF minus ON (nats/token) | +0.00115966 | +0.00006713 |
| Descriptive bootstrap 95% CI | [-0.00357465, +0.00602082] | [-0.00228562, +0.00261189] |

Injection is the nonpadding-token Frobenius norm of `a_l * donor` divided by
the recipient output norm immediately before rewiring, computed per sentence
and layer, then averaged. Layers are not independent observations. CE is the
macro mean of sentence mean target-token NLL from the routed bare model;
it excludes the auxiliary loss. OFF replaces only route hooks with identity.
The ON logits were reproduced exactly after restoring hooks.

CI: paired sentence bootstrap, 10,000 resamples, RNG42, conditional on these
64 rows and selected checkpoints, unadjusted and exploratory. Both contain
zero. This is not a generated-BLEU ablation or evidence about retraining
without CLRR.

### Learned coefficients (recipient encoder layer, one-based)

| Layer | P1 | P2 |
|---|---:|---:|
| 3 | +0.00146756 | +0.00001486 |
| 4 | +0.00354353 | +0.00214686 |
| 5 | +0.00544485 | +0.00410521 |
| 6 | +0.00590795 | +0.00477036 |
| 7 | +0.00551248 | +0.00385372 |
| 8 | +0.00520523 | +0.00378019 |
| 9 | +0.00378497 | +0.00158563 |
| 10 | +0.00227488 | -0.00051120 |
| 11 | +0.00040892 | -0.00313003 |
| 12 | -0.00014390 | -0.00434152 |

The route is weak at the selected checkpoints. This does not prove that
learning rate caused underlearning: small strengths can also be preferred by
optimization. Negative upper-layer coefficients do not justify deleting
layers or imposing signs. Weak final routes do not imply that the trained
backbone equals the baseline or that routes had no training-time effect.

## 3. Correctness and optimizer checks

- Explicit injection formula matched hook outputs; OFF matched zero scalars
  in the tiny CPU check; ON restoration was exact.
- Scalars were restored from the exact selected checkpoints, remained
  trainable, and occurred once in the reconstructed optimizer: LR 5e-5,
  weight decay 0. All four scalar gradient norms per model were finite and
  nonzero. No registration/frozen-parameter/save-load defect was identified
  within these checks.
- No optimizer step was taken; parameter versions and scalar values stayed
  unchanged. Eval-mode gradients are not training-mode accumulated updates.
- Recorded training preclip global gradient norms exceeded max_grad_norm=1
  in 57/72 sampled P1 entries and 59/72 P2 entries. These are sampled global
  norms, not every step or scalar-specific clipping. Without full optimizer
  history they cannot reconstruct cumulative Adam scalar updates.
- First diagnostic launch failed while constructing Trainer without an
  evaluation dataset. Only diagnostic argument construction was fixed to
  disable evaluation; the successful worker SHA is recorded below. The
  failure did not affect the completed training or checkpoints.

## 4. Original LSR-only control inventory

Public canonical HF inventory revision:
`3df24282f7f38eadd747b0774f17eb75de6e2813` (4,816 files, 39 metadata JSONs read).

| Existing control | Evidence | Matches current original-LSR RTX control? |
|---|---|---|
| Original cosine LSR, 11 training receipts (seeds42–52) | BF16, microbatch32, accumulation4, ES4 | No: current recipe FP32/b4/ga32/ES10 |
| ordinary-lsr-v1, RTX seeds42–44 | InfoNCE objective, content masking, chrF++ selector | No: different objective and selector |
| Historical original LSR folders | Some weights/metrics exist, sufficient matched training metadata not identified | Not certified |

No metadata-certified original cosine LSR-only control matching the current
RTX recipe was identified. This is not proof that no such checkpoint exists
elsewhere. Older results remain valid within their documented cohorts.
No old score was changed; no test predictions were fetched for this audit.

## 5. Decision and next step

1. Keep the learned-strength recipe at NO-GO for expansion. Do not open a
   scalar-LR/layer/lambda sweep based solely on this probe.
2. The concrete missing comparison is **one original cosine LSR-only run,
   Amis/mBART seed42**, on the locked RTX FP32/SDPA/b4/ga32/LR5e-5,
   ES10/max20, validation BLEU beam1 recipe, lambda=.1. First reuse a checkpoint
   if sufficient matching metadata is subsequently found. This control was
   **not launched** in this diagnosis.
3. Compare that control with the reused baseline and P2 on validation. If
   LSR-only matches/exceeds P2, CLRR's added value remains unsupported and a
   stronger CLRR module is not justified by these numbers. If P2 exceeds it,
   that is a development hypothesis requiring replication, not an ACL claim.
4. Any next design must explain a measured failure and undergo prior-work
   review, correctness and GPU gates before a new pilot. Learned scalar
   shortcuts alone are not established as a novel contribution.

## 6. Provenance and storage

Worker: `scripts/analysis/diagnose_learned_strength.py`.
Successful SHA256:
`1e583b35dad06d7194381324d30446608dd66436062a7341e905eb3beda1623b`.

P1 archive revision `44893834914b087b2a18efb866bb4d0434629b41`, SHA256
`ac750509be6b11ab85206cd0c7fc297a6675c1ac6569c27283cd35372e2251bb`.
P2 archive revision `ff121c099d0b0bd31abc79590764fc285fd601bd`, SHA256
`a8beeea61eb9ea6eb2bc18bbd81ffb624633036cf3c5416710b1014de3fdc7e4`.
Repo: `FiveC/amis-rewire-checkpoints`; the original checkpoint paths and
receipts remain unchanged.

Small local evidence: `outputs/experiments/learned_strength_diagnosis/`,
including DESIGN, COMPLETE, SUMMARY, per-run RESULT, control/log audits and
REMOTE_FINAL_CHECK. No checkpoint weights were downloaded to Windows.
Remote temporary archives/models removed; final independent inspection found
no remaining weight files in the diagnostic output, PID422668 absent,
23,952 MiB GPU free / 0% utilization and 70.89 GiB disk free.
