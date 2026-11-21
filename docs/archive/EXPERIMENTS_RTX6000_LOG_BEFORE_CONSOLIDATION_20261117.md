## DRM validation integration repair — queued, 2026-11-17

Author requested fixing DRM and completing all missing RTX tests. The failed original cohort remains unchanged as historical evidence; this is a new DRM-only frozen execution, with the identical original protocol/run spec and canonical HF destination. AC is not retrained.

**Fixes:** remove scoring-only `labels` from generation kwargs (no reference tokens enter decoding); populate the `Seq2SeqLMOutput` loss mapping so Trainer can read `outputs['loss']`. No change to DRM's decision-memory formula, CE/SKL objective, initialization, data, seed42, LR5e-5, B4×GA32, FP32/SDPA, 20epochs/ES10, or validation-BLEU beam1 selection.

Regression reproduced the original `unused model_kwargs: labels` error before the fix and then exposed the missing loss mapping. All **9 CPU correctness tests PASS** on the repaired frozen bundle, including actual `PrecisionSeq2SeqTrainer.predict` with collated labels/decoder inputs, greedy and beam4, reference-label independence, checkpoint replay, gradient accumulation, save/reload, and no inference gradient side effects. The full-model GPU gate now also exercises the Trainer prediction route with labels. CPU PASS is not a GPU gate or completed training result.

- Config: `configs/experiments/amis_drm_validation_fix_rtx.json`.
- Frozen output: `outputs/experiments/amis_drm_validation_fix_rtx/`.
- Bundle SHA256: `20268e725987c66e3a04dec02e585e5a8868b2c49a6e00578971fc28b558b99c`.
- Manifest SHA256 / CPU receipt: `d20743c76a7dcf3288ac9b268a1d39b6e02e8a23d6d87c3992173396f69a1538`.
- Detached dispatcher PID **3479703**, verified alive, **WAITING_PRIOR_TESTS**.
- Dispatch receipts/log: `outputs/experiments/amis_drm_validation_fix_dispatch/`.

The repaired bundle is staged only; it will not replace source while the immutable historical test queue or AC-only test follower is active. After both are terminal, the dispatcher acquires the project lock, verifies the old source and staged bundle hashes, deploys, reruns CPU tests and frozen TRAIN scientific admission, then the expanded full-model GPU gate. Long training runs only if these gates PASS, followed by validation-selected checkpoint upload/verification and authorized test beam4. The prior failed DRM run saved no resumable checkpoint, so training restarts from the original pretrained initialization. No other user's process is stopped and resource admission retains the20,000MiB free-VRAM threshold.

**Test audit snapshot:** 27/44 historical missing tests completed, 0 failures;49 previously verified tests skipped and token R3/R4 recovered tests backed up. AC has its separate follower after the44-test queue; DRM's test follows successful repaired training.

| Completed test | BLEU | chrF (`word_order=0`) | chrF++ |
|---|---:|---:|---:|
| Baseline SDPA seed42 |19.392604|18.091695|13.888931|
| Baseline SDPA seed43 |19.553961|18.004826|13.865648|
| Baseline SDPA seed44 |20.245322|18.597458|18.885684|
| Middle seed42 |19.375310|17.856526|18.306136|

Baseline test metrics/receipts were verified against pinned HF artifacts in `outputs/experiments/rtx_missing_test_inventory/hf_test_audit.json`; these are reused, not newly inferred scores. Middle's new verified test is in `outputs/experiments/rtx_test_backfill/amis_middle_comparator_rtx__MIDDLE_S42/` on RTX and the public checkpoint prefix. Selectors/backend provenance remain separate; no significance or independent-confirmation claim is made. Existing historical score tables are unchanged.

---

## RTX missing-test backfill — active, 2026-11-17

**Current status supersedes the AC/DRM queued status below.** Completed primary validation-selected checkpoints were audited against local receipts and public HF artifacts. No retraining or checkpoint reselection is authorized by this backfill.

| Audit outcome | Count | Action |
|---|---:|---|
| Previously verified tests |49|Skip; checkpoint SHA matched|
| Missing historical primary-checkpoint tests |44|Detached serial test queue PID3454963|
| Token-LSR / Token-Full local tests |2|Uploaded and verified; no GPU rerun|
| Completed AC checkpoint |1|Independent test follower PID3456273, after current project lock|
| Retired Turkish primary checkpoint |1|Excluded, as previously requested|

The 97-record inventory covers the primary completed checkpoint receipts found in the audited RTX cohorts. Four alternative secondary chrF-selected archives are outside this primary-checkpoint backfill. See [full inventory](../../outputs/experiments/rtx_test_backfill_dispatch/INVENTORY.md).

**Progress verified:** W seed42 test completed, BLEU21.163097347093107, chrF++19.55528067762161, chrF0 19.459829990378026. W seed43 is running, W seed44 follows. These test results are exploratory characterization, not fresh confirmation; selector/backend differences remain recorded per run. Do not pool all 45 tests as one matched comparison.

Each test uses its original validation-selected checkpoint, fixed archive SHA and recorded protocol. Metrics/predictions are publicly backed up under the corresponding canonical HF checkpoint prefix; restored weights are removed only after verification. The two dispatchers share the project lock, preventing overlap with each other; resource admission waits for sufficient VRAM without stopping other users.

**DRM integration failure:** original training stopped at step36, first epoch validation, because Hugging Face generation rejected the `labels` keyword passed through the custom generation wrapper. This is an evaluation integration error, not a completed efficacy result. No DRM epoch checkpoint was saved. The original AC/DRM training and dependent test dispatchers are FAILED. AC is complete and has been queued independently; DRM requires a separately verified integration fix before any restart. Frozen source files are unchanged while backfill runs.

Receipts: `outputs/experiments/rtx_test_backfill_dispatch/` (manifest, launch, status snapshot, DRM traceback), `outputs/experiments/amis_two_method_round1_rtx_ac_test_dispatch/`. Test results: `outputs/experiments/rtx_test_backfill/` and `outputs/experiments/amis_two_method_round1_rtx_ac_test/` on RTX. Only small result artifacts are retained locally; weights are not downloaded to the workstation.

---

## AC / DRM test follow-up — queued, 2026-11-17

Author explicitly requested adding test evaluation for completed models.
Detached post-training dispatcher PID **3451373**, verified alive and waiting
on the project training lock; DRM training remains unchanged and running.
The dispatcher waits for the entire original queue to complete, then evaluates
AC and DRM in sequence using their **unchanged validation-BLEU-selected**
checkpoints. It does not run test concurrently with DRM training.

Protocol: full575 Amis test sentences, beam4, FP32/SDPA, original references and
existing BLEU/chrF++ metric implementation. Test results are exploratory; they
are not used for checkpoint selection or independent confirmation. Each test
result/predictions is uploaded under its existing canonical public HF checkpoint
prefix and verified before transient restored weights are removed.

- Worker: `scripts/analysis/after_two_method_test.py` (new; training bundle unchanged).
- Evaluator: existing `scripts/analysis/evaluate_frozen_rtx.py`, SHA256 `a319c986b1bfa83723ad1d4bc9cd4c4fe41355350ef69da46f35e3733cd6e5aa`.
- Dispatch receipts/status/log: `outputs/experiments/amis_two_method_round1_rtx_test_dispatch/`.
- Test outputs after selection locks are prepared: `outputs/experiments/amis_two_method_round1_rtx_test/`.
- No retraining, no additional model families, no automatic significance claim.

---

## AC / DRM independent pilot — launched 2026-11-17

Author approved two independent Amis→Mandarin mBART seed42 pilots after the
review in [two-method design](../research/TWO_METHOD_DESIGN_ROUND1.md).
Eight CPU correctness tests PASS locally. Detached RTX queue PID **3418304**;
initial free GPU memory **23952 MiB**, project disk about191 GiB. This is a
**probe/gate launch, not a completed training result**.

| ID | Method | Training admission | HF destination below `checkpoints/amis_mandarin/mbart-large-50/` |
|---|---|---|---|
|AC_S42|antisymmetric-consistency-v1|Frozen full-parameter gradient screen, then GPU gate|`antisymmetric-consistency-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`|
|DRM_S42|decision-residual-memory-v1|Frozen gold-NLL preservation and latency screen, then GPU gate|`decision-residual-memory-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`|

Same original mBART initialization as W; no warm start from W. Pinned trained W
is used only by frozen TRAIN probes. Recipe retains FP32 SDPA, LR5e-5, B4×GA32,
20epochs, ES10 selected by original-reference validation BLEU beam1. No test.
DRM explicitly uses full-prefix beam-local replay with KV caching disabled;
its extra latency is measured, not claimed compute-matched.

Configuration: `configs/experiments/amis_two_method_round1_rtx.json`.
Output: `outputs/experiments/amis_two_method_round1_rtx/`.
Bundle SHA256: `0ef40ec4f5672c3c9a886badfa3fe0ee7477852ab14f8a0a357bf0e955445aad`.
Canonical public repository: `FiveC/amis-rewire-checkpoints`. Completed model
weights are removed from RTX only after public upload/read-back SHA verification;
failed uploads keep weights. Failed admission skips the candidate and records
why. Neither novelty nor efficacy is established by passing a gate.

```powershell
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_two_method_round1_rtx.json --account 1
```

---

> Latest follow-up (2026-11-15): [PLC failure analysis](../research/PLC_FAILURE_ANALYSIS.md)
> contains the completed no-training checkpoint audit. New results are in the
> post-hoc audit entry below; historical experiment numbers are preserved.

## Source-response frozen probe — COMPLETE, 2026-11-14

No new training. Baseline/B(KL2.5)/W(KL1.25) seed42 measured on first64 Amis
validation rows,4 fixed dropout pairs and2 wrong-source pairings. FP32 SDPA,
TF32 off, teacher forcing, no optimizer or test. CPU checks and all3 GPU gates
PASS (strict checkpoint loading, exact RNG replay, different masks, no grads).
Queue exited0; all transient weights removed. Final free VRAM23952MiB.
Collected16 small artifacts with SHA verification; all summary means replayed
from1024 CSV records/checkpoint within1e-10. No weights downloaded locally.

| Lexical diagnostic | Baseline | B KL2.5 | W KL1.25 |
|---|---:|---:|---:|
| Ordinary dropout symmetric KL |0.53734|0.28508|0.36922|
| Gold logp correct-minus-wrong source |3.82411|3.56874|3.83621|
| Matched CE |3.55714|3.31033|3.45383|
| Gold-minus-competitor margin |-0.51002|-0.52944|-0.45422|
| Teacher-forced top1 |0.49509|0.50208|0.49717|

Compatible with a regularization-strength tradeoff, not a causal mechanism:
W has source gap greater than B on52/64 rows, dropout KL greater on59/64;
but its CE is worse than B and top1 slightly lower. A larger source gap does
not imply better correct-source likelihood: W correct/wrong CE both worsen
relative to B. Shuffled negatives and confidence scaling confound grounding.
Existing generation/diagnostic sentence-delta correlations are near zero
(|Spearman rho|<=0.090 on64 rows); no simple generation explanation established.
W remains tuned R-Drop, not a new method or significance claim.

Raw q=softmax(z_correct-z_wrong) gives89-93% mass to tokens whose ordinary
matched p<1e-5. Tail mass92.65-93.82% must be contextualized: train-count<=1
IDs occupy98.8978% of vocabulary, so raw tail mass alone is NOT collapse or
pathological enrichment evidence. The rejected response-KL candidate remains
NO-GO for the previously documented prior overlap and uninformative minimum;
this diagnostic does not train that loss.

Full verified report:
`outputs/experiments/amis_consistency_source_response_rtx/ANALYSIS.md`.
Plan SHA256:0920cc8eba608d0ced6bcdea37e1782efec1add8f37419af3ea3960a2f93d110.
Next: design a falsifiable source-conditioned consistency RQ and audit against
R-Drop/DDR/ALD, CBMI and CKD BEFORE any method implementation or pilot. Current
probe alone does not justify another sweep. No test, Ede or A100 opened.

## W/B replication COMPLETE — validation only (2026-11-14)

All four new runs completed20epochs/720steps. HF backup verified for all four;
24 small JSON/CSV artifacts collected, original576 validation references aligned
and 8 corpus metrics replayed within1e-10. No checkpoint weights downloaded.

| Seed | Baseline BLEU | B beta2.5 | W beta1.25 | W minus B |
|---|---:|---:|---:|---:|
| 42 | 17.6823 | 17.9107 | 18.2724 | +0.3617 |
| 43 | 16.7345 | 17.5757 | 18.3958 | +0.8200 |
| 44 | 16.2089 | 18.3274 | 18.1881 | -0.1393 |
| Mean | 16.8753 | 17.9380 | 18.2854 | +0.3475 |

W-minus-baseline mean BLEU +1.4102, positive3/3; W-minus-B positive2/3. W mean chrF++ 15.1272, baseline 14.2180, B 14.8973.
Practical development gate PASS; not statistical significance or novelty proof.
W44 is slightly below B44, so do not claim uniform W superiority. Baseline
backend provenance caveat remains. Seed42 guided this development follow-up.
No test, Ede, new seeds or A100 launched; queue exited normally. Full small
summary: outputs/experiments/amis_consistency_strength_w_replication_rtx_suite/SUMMARY.json.

## W/B strength replication — QUEUE RUNNING (2026-11-13)

After seed42 W improved, author requested continuing. Four new runs only:
W_S43/W_S44 (beta1.25) and B_S43/B_S44 (beta2.5), Amis/mBART, same fixed
FP32 SDPA B4 GA32 ES10 max20 validation-BLEU recipe. No test or transfer.
Detached suite PID **1830468**, both remote CPU gates PASS (10 tests total),
first GPU gate observed at step0/2. All four GPU gates precede long training.
Public HF canonical seed43/44 paths; verified backup before weight cleanup.
Baseline43/44 reused after full-trajectory audit: BLEU and chrF++ select the same
saved checkpoints, full20epochs, matching runtime/data/optimizer. Historical SDPA
backend evidence is reconstructed, not a contemporaneous receipt. No baseline
rewrites or retraining. Details/gates: `docs/research/CONSISTENCY_STRENGTH_REPLICATION.md`.
W is a reduced-strength R-Drop control, not a new ACL-method novelty claim.

## G/R/W COMPLETE — validation seed42, NO-GO for split components (2026-11-13)

All three runs completed 20 epochs / 720 steps, not early stopping. All best
checkpoints uploaded to FiveC/amis-rewire-checkpoints with verified hashes.
Queue completed and GPU idle. No test was run.

| Recipe | Validation BLEU | chrF++ |
|---|---:|---:|
| Baseline reused | 17.6823 | 14.7030 |
| B full consistency, beta2.5 | 17.9107 | 14.8427 |
| G gold-only, beta2.5 | 16.8007 | 14.1567 |
| R non-gold conditional, beta2.5 | 17.2483 | 14.4396 |
| W full consistency, beta1.25 | **18.2724** | **15.1728** |

G/R fail the predeclared gate; do not automatically replicate them. W gains
+0.5901 BLEU / +0.4698 chrF++ over baseline and about +0.3617 BLEU over B.
This supports investigating consistency strength, not a novel gold/rest split.
Single-seed validation development evidence only; no statistical or test claim.
Further training is not automatically authorized by this result.

## G/R/W queue launched on RTX (2026-11-13)

Disk blocker resolved after author-approved ViNID cleanup: about 209.2 GiB free.
Frozen bundle unchanged: `87f961c90fbeb999c75327b7456231cb34ce406b9f9e8eb9c4ffa1caa9362a70`.
Queue PID **1684260**, detached under authorized CLRR project; remote CPU gate
**PASS (5 tests)**. Assigned `G_S42`, `R_S42`, `W_S42`. Next: pinned baseline/B
frozen diagnostics, then ALL three GPU gates, then sequential long training.
Long training and GPU gates not yet observed at this status check. No test stage.
The BLOCKED DISK entry below records the earlier state; launch receipt is
`outputs/experiments/amis_decoupled_consistency_rtx/launch_rtx.json`.

## G/R/W consistency pilot — CPU PASS, BLOCKED DISK (2026-11-13)

Recipe: `configs/experiments/amis_decoupled_consistency_rtx.json`.
Implementation: `src/methods/decoupled_consistency/v1/`.
Design: `../research/POST_PAIRED_ROUTE_METHOD_PLAN.md`.
Receipt: `outputs/experiments/amis_decoupled_consistency_rtx/IMPLEMENTATION_STATUS.json`.

| Run | Dataset / model / seed | Objective | Status |
|---|---|---|---|
| G_S42 | Amis Mandarin / mBART-50 / 42 | mean CE +2.5 gold/rest binary KL | CPU PASS; no GPU gate/train |
| R_S42 | same | mean CE +2.5 weighted conditional non-gold KL | CPU PASS; no GPU gate/train |
| W_S42 | same | mean CE +1.25 full symmetric KL | CPU PASS; no GPU gate/train |

Five new scientific tests PASS; five original paired-route regression tests PASS.
Frozen bundle and CPU_GATE.json prepared. No new validation/test scores exist.
RTX read-only admission: 23952 MiB free VRAM, idle, project lock free; disk only
3.89 GiB. Queue requires 40 GiB free, therefore NOT deployed or launched.
Do not delete other users' files or change precision/batch to bypass admission.
After space is available: frozen baseline/B diagnostics (64 fixed validation rows,
4 dropout seeds; logit gradients only), all 3 GPU gates, then G/R/W sequentially.
Same pinned A-D protocol: FP32 SDPA TF32off B4 GA32 ES10/max20 validation BLEU.
Reuses baseline/A/B evidence; no old scores modified. No test/replication/transfer.
These are attributed engineering ablations of R-Drop/DKD, not novelty certification.

## Paired native/CLRR engineering pilot — COMPLETE / NO-GO (2026-11-10)

Author approved four Amis/mBART seed42 arms after the prior-art/design review.
Recipe: `configs/experiments/amis_paired_route_rtx.json`.
Design: `../research/CLRR_RECOVERY_DESIGN.md`.
Separate implementation: `src/methods/paired_route/v1/model.py`.

| Run | Views | Objective | Latest verified state |
|---|---|---|---|
| A_NATIVE_CE_S42 | OFF/OFF | Mean two-view CE | COMPLETE20epochs; val BLEU16.8619 / chrF++14.2623; HF verified |
| B_NATIVE_RDROP_S42 | OFF/OFF | Mean CE +2.5 symmetric KL | COMPLETE20epochs; val BLEU17.9107 / chrF++14.8427; HF verified |
| C_PAIRED_CLRR_CE_S42 | OFF/ON | Mean two-view CE | COMPLETE20epochs; val BLEU16.6620 / chrF++14.0011; HF verified |
| D_PAIRED_CLRR_RDROP_S42 | OFF/ON | Mean CE +2.5 symmetric KL | COMPLETE20epochs; val BLEU17.3386 / chrF++14.3730; HF verified |

Five CPU correctness tests PASS. Exact full-vocabulary KL with token-chunk
recomputation; backbone per-layer reentrant checkpointing retained and gradient
replay tested with dropout/accumulation. Numerical/data/baseline protocol lock
retained: FP32/SDPA, TF32off, b4/ga32, LR5e-5, ES10/max20,
original-reference validation BLEU beam1. No LSR, test, new seeds or transfer.
Consistency attributed to R-Drop; engineering pilot, no novelty certification.

Detached queue PID478144 in `/workspace/ura/thanhdat/projects/CLRR/`;
output `outputs/experiments/amis_paired_route_rtx`.
Bundle SHA256 `cd3fddc5b687c22bfcc2d5d91d379247346b089ff6960222fa76c17f2f048030`.
Before launch: GPU free23952MiB, disk free100.71GiB, project lock idle.
Queue runs remote CPU then all four GPU gates; training starts only after all
PASS. Failure stops queue. Sequential A–D; no other users' processes touched.
Each completed checkpoint is uploaded and hash-verified before weight cleanup.
Public HF prefix per method:
`checkpoints/amis_mandarin/mbart-large-50/<paired-method-id>/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`
in `FiveC/amis-rewire-checkpoints`. No scores available at launch.

Verified follow-up: remote CPU gate and all4 GPU gates PASS with the exact same
frozen bundle. Each passed save/reload logits, Trainer best-checkpoint reload,
beam4 cached/uncached generation and optimizer/scheduler/RNG resume equivalence.
Peak allocated GPU memory about14.38GiB per gate. A has entered actual long
training (720-step cap); observed steps1–2, not just a technical gate. Queue alive.
Small immutable receipts copied locally under this cohort's `gates/*/PASS.json`.

Final monitor: queue exited normally with `QUEUE_COMPLETE`, all4 runs complete.
All reached epoch cap20/720steps, not early stopping. Selected A/C/D epoch17
(step612), B epoch18 (step648). Public backup receipts match all4 completion
receipts. CPU replay576/576 original-reference validation predictions per arm
matches recorded BLEU/chrF++;12 raw JSON/CSV files fetched from pinned public
revisions and SHA256-verified. No weights downloaded locally, no test opened.

| Final comparison | Delta BLEU | Delta chrF++ |
|---|---:|---:|
| B minus reused baseline | +0.2284 | +0.1397 |
| C minus A | -0.1999 | -0.2612 |
| D minus B (primary) | -0.5721 | -0.4697 |
| D minus reused baseline | -0.3437 | -0.3300 |

Primary D fails the prospective advancement gate. NO-GO for expanding this
paired-route recipe; no automatic test, additional seeds or Ê Đê transfer.
B supports an exploratory consistency signal, not a CLRR contribution.
These are validation seed42 comparisons, not significance or equivalence claims.
Summary: `outputs/experiments/amis_paired_route_rtx/PILOT_SUMMARY.json`.
C backup revision `e1b1539664b1187def812cb4141b1b516b9cfe04`;
D backup revision `94d1b87cf8b480fc097e6f17c116efa693e53132`.

## Original cosine LSR-only control — COMPLETE (2026-11-10)

LSR_COSINE_S42 finished20epochs/720steps (epoch cap, not early stopping),
selected epoch19/step684 by validation BLEU. CPU/GPU gates PASS. Public HF
backup hash-verified; queue exited, no remaining gate/training weights found.
Only small evidence files collected locally; no test opened.

| Model | Validation BLEU | Validation chrF++ |
|---|---:|---:|
| Reused baseline | 17.6823 | 14.7030 |
| Original cosine LSR-only | 17.2418 | 14.4559 |
| P1 learned CLRR + CE | 16.6360 | 14.1009 |
| P2 learned CLRR + original LSR | 17.3459 | 14.4554 |

LSR-only minus baseline: -0.4405 BLEU/-0.2471 chrF++.
P2 minus LSR-only: +0.1041 BLEU/-0.0005 chrF++.
The matched control does not show a substantial CLRR addition in this seed;
P2 remains below baseline. No significance or interaction claim follows from
one exploratory validation seed. Keep learned-strength expansion NO-GO;
do not automatically open test, more seeds, or parameter sweeps.
Recipe: `configs/experiments/amis_original_lsr_control_rtx.json`.
Evidence: `outputs/experiments/amis_original_lsr_control_rtx/runs/LSR_COSINE_S42/`.

## Original cosine LSR-only control — QUEUE LAUNCHED (2026-11-10)

Author approved the single missing matched control after checkpoint diagnosis.
Recipe: `configs/experiments/amis_original_lsr_control_rtx.json`.
Run `LSR_COSINE_S42`: Amis/mBART, original stop-gradient cosine LSR lambda=.1,
no CLRR, seed/data_seed42. Data/runtime/protocol lock and critical numerical
source hashes match the completed learned-strength cohort. FP32/SDPA,
b4/ga32, LR5e-5, ES10/max20, original-reference validation BLEU beam1.
CPU gate PASS locally and on RTX; exact GPU gate PASS (2 optimizer steps,
save/reload and Trainer best-checkpoint reload verified, peak11.99GiB).
Detached queue PID426253 alive and automatically proceeds to long training. No test or replication queued.
Prelaunch GPU free23952MiB; disk free70.89GiB; project lock free.
Bundle SHA256 `e8496913aa7119bc730e78600c411f50cbb35b4908df20a772feb78217bd9138`.
Public HF destination:
`checkpoints/amis_mandarin/mbart-large-50/lsr/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`
in `FiveC/amis-rewire-checkpoints`. Worker uploads/validates backup hashes
before removing trained weights. Results pending; compare validation with
baseline and learned-strength P2, retaining exploratory one-seed scope.

## Follow-up diagnosis COMPLETE (2026-11-10)

Selected checkpoint probes (64 shared validation rows, zero optimizer steps)
found weak routes: mean injection/recipient norm P1 **0.24125%**, P2
**0.17294%**. Route OFF-minus-ON CE = +0.00115966 / +0.00006713 nats/token;
both exploratory sentence-bootstrap 95% CIs include zero. Scalars have finite
nonzero gradients and appear once in the reconstructed optimizer; no defect
was identified by these checks. This does not prove a learning-rate cause.
Inventory did not identify a metadata-certified original cosine LSR-only
control matching RTX FP32/b4/ga32/ES10/BLEU selection. Existing ordinary-lsr
RTX runs use InfoNCE and cannot substitute. Next missing comparison is one
matched original LSR-only seed42 control; **not launched**. No test opened,
old scores unchanged, temporary diagnostic weights cleaned, queue exited.
Full scope, caveats, coefficients and provenance:
[Learned-strength diagnosis](../research/LEARNED_STRENGTH_DIAGNOSIS.md).

## Current result — learned-strength CLRR COMPLETE / NO-GO (2026-11-10)

Both GPU gates PASS; both runs completed20epochs/720steps (epoch cap, no
early stop). Queue COMPLETE and no longer alive; GPU free23952MiB, util0%.
Independent CPU replay on all576 original validation rows reproduced BLEUzh
and native chrF++ exactly; source/reference ordering matched the original CSV.

| Model | Validation BLEU | Validation chrF++ | Selected epoch | Delta BLEU vs baseline |
|---|---:|---:|---:|---:|
| Reused baseline | 17.6823 | 14.7030 | 19 | — |
| P1 learned-strength CLRR + CE | 16.6360 | 14.1009 | 18 | -1.0463 |
| P2 learned-strength CLRR + original LSR | 17.3459 | 14.4554 | 19 | -0.3364 |

P2-P1 = +0.7099 BLEU/+0.3544 chrF++; P2-original Full = +0.2304 BLEU,
but neither candidate passes the prospective baseline+.30 BLEU/nondecreasing
chrF++ gate. **NO-GO expansion**, no new seeds/test/Ede/A100 triggered.
Single-seed exploratory development results; no statistical confirmation.
This rejects advancement of this recipe, not every possible learned route.
HF SHA backup verified: P1 revision44893834914b087b2a18efb866bb4d0434629b41,
P2 revisionff121c099d0b0bd31abc79590764fc285fd601bd. Remote inspection confirmed
no remaining best archives/models/checkpoint directories in either run.
Only small artifacts collected locally. Replay: `outputs/experiments/amis_learned_strength_rtx/VALIDATION_SCREEN.json`.

### Launch provenance (completed)

Author approved two new Amis/mBART seed42 runs: P1 learned-strength CLRR + CE,
P2 the same route + original LSR(.1). Ten scalar strengths start at zero;
all other numerical recipe fields match the pinned RTX baseline protocol.
CPU correctness 4/4 PASS. Detached RTX queue PID **379839** launched with
stage `gate-train`: CPU replay, both exact GPU gates, then sequential training.
Launch does not mean GPU gates have passed or long training has started.
Bundle SHA256: `df4273e8058d78e088e69933cb277fa4a6b64aaadba63ce00a281dff5e83e2e1`.
Prelaunch GPU free23952MiB/total24576MiB; disk free70.9GiB; CLRR lock free.
Recipe: `configs/experiments/amis_learned_strength_rtx.json`.
Output: `outputs/experiments/amis_learned_strength_rtx`.
Public HF: `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/{learned-strength-clrr-v1,learned-strength-full-v1}/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`.
Backup is pending until training finishes; delete weights only after SHA verified.
No test, further seeds, Ede or A100 stage in this queue. This is an authorized
engineering control, not a new standalone novelty claim. Earlier Transition
NO-GO remains unchanged; its expansion HOLD does not apply to these two newly
approved runs. See [design and gates](../research/LEARNED_STRENGTH_PILOT.md).

## Previous result — Transition LSR engineering pilot COMPLETE (2026-11-10)

Follow-up audit **COMPLETE (2026-11-10)**: 63 local validation receipts, five
20-epoch trajectories, five pinned checkpoint geometry/gradient probes (zero
optimizer steps), and 576-row validation strata/leave-one-out replay PASS.
Transition Full edge cosine matched/shuffled = 0.983797/0.981994; baseline =
0.968193/0.964215. Endpoint centered cosine: 0.245356 versus baseline 0.275199.
No severe endpoint rank collapse shown in this 64-example probe; no evidence
here for auxiliary gradient domination. Route dependence remains, without
baseline superiority. No additional training/test/replication admitted.
Details and next-stage gates: [TRANSITION_FOLLOWUP_DIAGNOSIS.md](../research/TRANSITION_FOLLOWUP_DIAGNOSIS.md).


Both Amis seed42 runs completed 20 epochs / 720 steps (epoch cap, not early
stopping). CPU5/5 and both GPU gates PASS. Protocol remained FP32/SDPA,
B4/GA32/LR5e-5/ES10/max20, original-reference validation BLEU selection beam1.

| Model | Validation BLEU | Validation chrF++ | Selected epoch |
|---|---:|---:|---:|
| Reused baseline | 17.6823 | 14.7030 | 19 |
| Transition LSR-only | 17.0833 | 14.2826 | 20 |
| Original CLRR + Transition LSR | 17.5220 | 14.6387 | 17 |

Full minus Transition LSR: +0.4387 BLEU; Full minus baseline: -0.1604 BLEU
and -0.0643 chrF++. Advancement gate FAIL: do not open replication, test,
Ede or A100 from this pilot. These are exploratory single-seed validation
results, not evidence of statistical significance or a test-set gain.

Both checkpoints have SHA-verified HF backups in `FiveC/amis-rewire-checkpoints`,
under `checkpoints/amis_mandarin/mbart-large-50/{transition-lsr-v1,transition-full-v1}/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`.
Backup revisions: LSR `4282e56df20c4fbe28da93a6322e208597fa9eb9`,
Full `33ea4df8bcb503132feb0249cf0bf87cf7a60686`.
Queue completed; GPU observed idle. Only small artifacts retrieved locally.
Raw metadata/prediction bytes were independently SHA-verified, parsed metadata
matched monitor copies, and both metrics recomputed exactly against original CSV.
Monitor text reserialization changes byte hashes; this was a transfer-format
issue, not changed predictions or scores.

Receipt: `outputs/experiments/amis_transition_lsr_rtx/PILOT_SUMMARY.json`.
Author-approved bounded engineering pilot superseded the earlier execution HOLD;
FDD overlap and historical novelty assessment below remain historical evidence.

---

## Current novelty verdict — Transition LSR v1: NO-GO

Focused review identified FDD (ACL2025), Section3.3 Eq7: cosine alignment
of changes across selected depths. The proposed bilingual pooled-state variant
is not identical, but the existing evidence does not establish a sufficiently
substantive new contribution. Provisional design admission is superseded.
See `docs/research/TRANSITION_LSR_NOVELTY_VERDICT.md` for exact differences.
No GPU gate or training started. CPU5/5 PASS is technical evidence only.
Preserve frozen code/recipe and historical results; no automatic retry.

---

## Current candidate — Transition LSR v1: PREPARED, LAUNCH BLOCKED (2026-11-09)

Author requested implementing the two-arm proposal and an RTX queue. Design and
primary-prior comparison are in `docs/research/TRANSITION_LSR_DESIGN.md`.
Code lives separately in `src/methods/transition_lsr/v1/`; frozen recipe:
`configs/experiments/amis_transition_lsr_rtx.json`.

- Amis/mBART seed42: TRANSITION_LSR_S42 then TRANSITION_FULL_S42.
- CPU scientific gate: 5/5 PASS, both cases PASS; original method/data untouched.
- FP32/SDPA/B4/GA32/LR5e-5/ES10/max20, validation BLEU beam1; no test stage.
- RTX read-only check: 23952/24576MiB free, GPU0%, project lock free, disk71GiB.
- **Automatic approval review REJECTED launch. No remote GPU gate or training
  was started.** Publication-level novelty remains unresolved; do not turn
  `novelty_certified` true or retry via another transport to bypass rejection.
- Full automatic queue command is prepared, but awaits an explicit author
  decision on admitting this bounded exploratory pilot, or stronger novelty
  evidence/redesign satisfying the existing gate. No automatic retry.
- Planned public paths (NOT uploaded):
  `checkpoints/amis_mandarin/mbart-large-50/{transition-lsr-v1,transition-full-v1}/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`.
  Existing runner verifies SHA backup before deleting transient weights.
- Receipts: `outputs/experiments/amis_transition_lsr_rtx/CPU_GATE.json`,
  `LAUNCH_BLOCKED.json`. No new translation scores exist.

Earlier STOPPED banner below concerns relational-v2; that queue stays stopped.

---

## CURRENT STATUS — STOPPED FOR NOVELTY REVIEW (2026-11-09)

Author corrected the required sequence: design → related-work/novelty review →
only if sufficiently justified, smoke/correctness gates → only after all gates
PASS, long training. Translation quality cannot substitute for novelty review.
Do not run smoke, resume or launch a candidate whose novelty is UNESTABLISHED/HOLD.
A focused prior list or different implementation alone does not clear this gate.

Relational v2 queue PID170990 was stopped; all five queue processes exited.
Only A had begun long training. Existing A checkpoint-36 is preserved; B/C/D
have no long-training results. This interrupted run is not a completed result.
No other user's process was signalled. Previous launch/PASS entries below are
historical; they do not authorize resumption. Current candidate is NOT admitted.

Before proposing a candidate again, document primary-paper method sections,
formula/mechanism overlap, the concrete remaining contribution, evidence and
uncertainty. If the distinction is insufficient, reject/redesign before GPU use.

---

## Baseline train-pair discrimination — hoàn tất 09/11/2026

Forward-only, **không phải run train mới hay BLEU/chrF++**. Baseline seed42
chrF++-selected lịch sử, FP32/SDPA/TF32off. Data train gốc giữ nguyên; lexical
pairs dùng Unicode miner v2, chưa có semantic labels. Repo revision khóa:
`40fac65856b883e999bee74a45075d9c9d5dc7a5`; archive SHA xác minh trước forward.

| Dataset | Nhóm | Cặp | Source đúng được ưu tiên cả hai chiều | Mean symmetric source margin | Cặp reference-choice lỗi/tie ít nhất một chiều |
|---|---|---:|---:|---:|---:|
| Amis | all near | 165 | 165/165 | 1.9214 | 1 |
| Amis | min-index selected near | 109 | 109/109 | 1.8481 | 1 |
| Amis | generic | 193 | 193/193 | 6.4709 | 0 |
| Ê Đê | all near | 2679 | 2679/2679 | 2.8923 | 3 |
| Ê Đê | min-index selected near | 1044 | 1044/1044 | 2.7147 | 3 |
| Ê Đê | generic | 1461 | 1461/1461 | 8.2560 | 0 |

Đơn vị nats/content target token, bỏ mọi special IDs. Source margin đổi source
cho CÙNG reference; reference-choice đổi reference cho CÙNG source. Chưa xác
minh song ngữ nên cột cuối không tương đương số lỗi dịch gold. Bảng không phải
accuracy generation, kết quả tổng quát hóa hoặc matched BLEU-selected comparator.
Không đọc validation/test. Không kiểm định iid vì cặp dùng chung các câu và
chỉ một seed. Near/generic chung anchors nhưng không matched độ dài/độ khó.

Amis134.5s,1090 unique score cells,2.62GiB peak allocated; ÊĐê208.5s,10996cells,
4.65GiB. Queue PID44880 hoàn tất, dọn weights tạm, GPU về23952MiB free/0%.
Chỉ JSON/CSV nhỏ về PC, SHA/download và CSV margins/means đã kiểm chứng.
Reusable script `scripts/analysis/measure_train_pair_discrimination.py`;3score
unit tests PASS. Không thay score thực nghiệm lịch sử, không upload HF mới.

Artifacts: `outputs/experiments/difference_preserving_feasibility/baseline_train_contrasts/`.
Giải thích/next decision: `docs/research/DIFFERENCE_PRESERVING_PILOT.md`, mục16.
Kết quả chưa ủng hộ giả thuyết baseline mất khả năng phân biệt train contrasts;
không tự mở queue loss mới. Ưu tiên audit lỗi validation generation có sẵn.

## Difference-preserving LSR — GPU technical checks hoàn tất 09/11/2026

Đây là kiểm thử code trên model đầy đủ, **không phải kết quả chất lượng/pilot**.
R0–R3 dùng cyclic train peers làm fixture, không giả xác minh semantic pairs.

| Dataset | Cases | Kết quả | Peak allocated VRAM |
|---|---|---|---:|
| Amis–Mandarin | R0/R1/R2/R3, seed42 | 4/4 PASS | 14.41GiB |
| Ê Đê–Vietnamese | R0/R1/R2/R3, seed42 | 4/4 PASS | 15.16GiB |

Mỗi case2optimizersteps, B4/GA32, FP32/SDPA, checkpointing; save/reload,
beam4 cached/uncached và resume optimizer/scheduler/RNG đều đạt tolerance
rtol1e-5/atol2e-5 hiện có. Đã verify8receipt theo manifest, không đổi protocol
baseline. Checkpoint tạm đã dọn; queue tự kết thúc, GPU0%,23952MiB trống.
Không download weights hoặc upload checkpoint kỹ thuật lênHF.

Tokenizer: đối chiếu toàn bộ train nguồn/đích của hai dataset với tokenizer
checkpoint reload đều khớp IDs; không áp dụng regex fix. Đây là kiểm tra
tokenization, không phải đánh giá validation/test. Semantic/novelty vẫnHOLD,
chưa train dài. Receipts riêng `TECHNICAL_PASS.json` không thay normal gate.

Báo cáo: `outputs/experiments/amis_difference_preserving_gpu_technical_suite/REPORT.md`.
Chi tiết: [mục13 kế hoạch](../research/DIFFERENCE_PRESERVING_PILOT.md).

## Difference-preserving LSR — chuẩn bị 07/11/2026, chưa train

Code phiên bản riêng và recipes R0–R3 đã được chuẩn bị. Kiểm kê **train-only**:
Amis165cặp ứng viên/193trong4600câu (4,20%); Ê Đê2460cặp/1461trong15092câu
(9,68%). Đây là cặp gần nhau về chữ, **0cặp đã được xác minh nghĩa**.
CPU unit tests15/15 PASS (sau sửa metadata/resume) trên model nhỏ; không phải GPU gate hoặc kết quả dịch.
Novelty và semantic-pair gates HOLD; không mở training/test. RTX SSH trả502,
chưa đọc được VRAM. Chi tiết, code, recipe và artifacts trong
[DIFFERENCE_PRESERVING_PILOT.md](../research/DIFFERENCE_PRESERVING_PILOT.md).
Không thay các số thực nghiệm lịch sử bên dưới.

## DRB Amis — phân tích validation hoàn tất 06/11/2026

Đã verify hash predictions/JSON từ HF revision40fac65856b883e999bee74a45075d9c9d5dc7a5,
576câu và metrics khớp<1e-8; cả4run cùng recipe/runtime/bundle. SelectorBLEU
replay đúng, cả4train20epochs nên không phải dừng sớm làm mất checkpoint.

D3−D2: BLEU−0.4536, CI95%[−1.2668,+0.3607], p native0.1052, Holm3=0.3156.
chrF++−0.1030, CI[−0.5976,+0.3975], Holm3=0.4716. Pairedbootstrap10k,
seed10042; family3metrics, exploratory validation-only, không kiểm định xác nhận.
Không chứng minh querygate gây hại: D3 thắngD2 ở10/20epochs vàCIchứa0.
Không có predictionrỗng hoặc repetitionflag; meanlengthratioD2/D3~0.997;
strata không cho pattern giảm đơn điệu theo độ dài. Chưa xác định cơ chế nguyên nhân.

**NO-GO thăng hạng D3 theo practicalgate** vì BLEU thấp hơn cả3controls.
Không tự đưaD2 thành methodchính, không mởtest/seed/A100 hoặc sweep. GiữHFweights
và artifacts nhỏ. ÊĐê vẫnHOLD. Bước thiết kế sau phải dựa cơ chế/prior, không
quy lỗi chắc chắn cho detachment/gate/data từ các chỉ báo hiện có.

Báo cáo: `outputs/experiments/amis_decoder_residual_bridge_rtx_analysis/validation/VALIDATION_ANALYSIS.md`.

---

## DRB Amis — cả4runs đã train xong (xác minh HF ngày06/11/2026)

Kiểm tra repo public tại revision `40fac65856b883e999bee74a45075d9c9d5dc7a5`: cả4prefix có
best.zip, validation_metrics.json và stop_state.json. Cả4 hoàn tất20epochs/720steps;
không early-stop. Đây là validationbeam1, BLEUzh, chọn checkpoint theoBLEU.

| Run | BLEU | chrF++ | Best epoch | Completed epoch |
|---|---:|---:|---:|---:|
| D0_S42 | 16.7525 | 14.1727 | 19 | 20 |
| D1_S42 | 16.8277 | 14.1486 | 18 | 20 |
| D2_S42 | 17.1272 | 14.4005 | 20 | 20 |
| D3_S42 | 16.6736 | 14.2976 | 19 | 20 |

D2 cao nhất trong4arms; D3 không vượt các matched controls theo BLEU nên
không đạt gate so controls đã khóa. Baseline audit trước có17.682329BLEU;
chưa dùng đối chiếu đó làm kết luận cuối trước receipt/runtime replay. Không
mởtest hoặc trainbổsung. SSH RTX hiện lỗi502/timeout: chưa xác minh trực tiếp
queueexit, receipts SHAverification hoặc cleanup trênmáy. Sự tồn tại4archive
trênHF được xác nhận độc lập, không đồng nghĩa đã kiểm traSHAtoànbộ tại lượt này.
ÊĐê vẫnHOLD doresume gatefail; chưa cótraining ÊĐê trongqueue này.

Artifacts nhỏ: `outputs/experiments/amis_decoder_residual_bridge_rtx_training/hf_status.json`.

---

## DRB — mở train dài Amis ngày 06/11/2026

Theo yêu cầu tác giả cho train dài, tách admission theo dataset: **Amis có cả4
CPU/GPU gates PASS nên mở4runs D0→D1→D2→D3 seed42**; Ê Đê tiếp tục HOLD vì
resume gate chưa đạt. Đây là thay đổi thứ tự điều phối so với suite cũ yêu cầu
8gates trước mọi training; không đổi recipe, tolerance, dữ liệu hoặc score cũ.

Queue detached PID2001404, thư mục remote
`outputs/experiments/amis_decoder_residual_bridge_rtx_training/`.
VRAM lúc mở23540MiB free. Giữ nguyên FP32/SDPA, B4/GA32, LR5e-5,
validation BLEU zh/beam1, ES10, max20epochs. Không mở test/replication/A100.
Upload từng final archive vào prefixDRB đã khóa, SHA-verify rồi xóa weights tạm.

Ê Đê: weights liên tục so với resume sau2steps lệch tối đa4.0233e-7; loss
log trùng ở4chữ số, gradnorm lệch~9.54e-7. Đây là bằng chứng định lượng về
mức độ lệch, **chưa xác nhận nguyên nhân, không biến gate FAIL thành PASS**.
Không đổi backend/determinism hoặc nới tolerance để ép qua gate. Chưa train Ê Đê.

Launch receipt: `outputs/experiments/amis_decoder_residual_bridge_rtx_training/launch_rtx.json`.

---

## Cập nhật DRB 06/11/2026 — queue dừng tại GPU resume gate Ê Đê

Queue PID1886492 đã dừng; chưa mở training dài (0/8 models). Cả4 GPU gates
Amis PASS. Gate D0 Ê Đê thất bại khi so logits sau optimizer/scheduler/RNG resume
với chạy liên tục: 3/512 phần tử ngoài tolerance, max absolute difference
4.6133995056152344e-05; atol2e-05, rtol1e-05. Các gate D1–D3 Ê Đê chưa chạy.
GPU đã giải phóng:23540MiB free,util0% ở lần kiểm tra này.
Không thay tolerance/protocol, không tự đánh PASS, không suy đây là điểm chất
lượng phương pháp. Cần chẩn đoán tính tái lập resume trước khi mở lại queue.

---

## DRB v1 — triển khai 06/11/2026 (validation-only)

Tác giả duyệt tám pilots seed42: D0 local, D1 donor, D2 source-gated difference,
D3 decoder-query-gated difference, trên Amis rồi Ê Đê. Đây là CE-only full
fine-tuning với module196608parameters; không phải LSR hoặc PEFT.

- Code riêng: `src/methods/decoder_residual_bridge/v1/`.
- Config: `configs/experiments/{amis,rhade}_decoder_residual_bridge_rtx.json`.
- Hai bundle đã PASS CPU: 9 scientific tests và exporter/roundtrip cho cả bốn arms.
- RTX queue detached PID1886492 tại `outputs/experiments/amis_decoder_residual_bridge_rtx_suite/`.
- VRAM lúc admission23540MiB free; GPU gate D0 Amis đã vào bước optimizer.
- Tất cả8 GPU gates phải PASS trước bất kỳ training dài nào. Gate gồm2optimizer
  steps, exactB4/GA32, checkpoint/reload, beam4cache và optimizer/scheduler/RNGresume.
- Lock FP32/SDPA/TF32off, LR5e-5, max20epochs, ES10; **chọn best và early stopping
  bằng validation BLEU beam1**, zh cho Amis,13a cho Ê Đê. Không mở test/A100/seed khác.
- HF public `FiveC/amis-rewire-checkpoints/checkpoints/<dataset>/mbart-large-50/drb-<arm>-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu-rank64-donor6-dec12.../`.
  Upload và SHA-verify rồi mới xóa weights tạm; local chỉ giữ artifacts nhỏ.
- Baseline reuse: Amis đã có audit cùng BLEU-best epoch19, cần receipt runtime
  khi kết luận cuối. Ê Đê baseline cũ chọn chrF++ epoch17, BLEU-best epoch18
  (34.029051); **chưa được coi là baseline BLEU-selected**. Kiểm kê weights trước
  quyết định chạy bổ sung. Không tự mở baseline ngoài8runs.
- Chưa có điểm validation mới. Novelty và hiệu quả vẫn là giả thuyết cần kiểm chứng.

Chi tiết: `docs/research/DECODER_RESIDUAL_BRIDGE_PROPOSAL.md`.
Baseline audit: `outputs/audits/drb_baseline_reuse/REUSE_AUDIT.json`.

---

## Audit 06/11/2026 — CPU/CUDA PASS, chưa mở training mới

12 optimization checks trên CPU và 12 trên CUDA PASS; 12 tests C0/can thiệp PASS. JSON trong `outputs/experiments/rhade_residual_diagnostics/`. Đây là kiểm chứng tiny synthetic, không có model/data run mới và không thay số chất lượng. Novelty gate U1 orthogonal/U2 gated donor chưa đạt; xem phân tích/công thức và prior trong `docs/research/RESEARCH_INSIGHTS.md`.

---

### Chẩn đoán frozen C0 Ê Đê — hoàn tất 06/11/2026

1.000 câu validation, seed42, checkpoint và protocol giữ nguyên; 0 optimizer steps. All-on khớp chính xác predictions validation gốc. Không sinh thêm test để chọn thiết kế.

| Can thiệp inference | BLEU | ΔBLEU so all-on | ΔchrF++ |
|---|---:|---:|---:|
| All-on | 34.0381 | 0 | 0 |
| Tắt module học | 34.0316 | −0.0065 | −0.0174 |
| Tắt donor | 28.8044 | −5.2337 | −4.5247 |
| Shuffle donor | 28.7713 | −5.2668 | −5.0632 |

Mean theo câu của tỷ lệ norm correction đã nhân alpha / current là 0.0204–0.0419% theo layer3–12; toàn residual / current là 7.82–8.30%. Đây là độ lớn activation, không phải tỷ lệ đóng góp chất lượng hoặc cập nhật optimizer.

**Kết luận:** module học có sensitivity inference rất nhỏ trên checkpoint này; donor tạo phụ thuộc sau training. Cả Amis và Ê Đê chưa chứng minh C0 vượt baseline ổn định. Không suy bỏ donor khi train sẽ tốt, module vô dụng trong training, hoặc nguyên nhân chắc chắn do data. Không mở replication/A100 cho C0 hiện tại.

Artifacts nhỏ: `outputs/experiments/rhade_residual_diagnostics/{SUMMARY.json,REPORT.md,gpu/}`. Weights khôi phục tạm đã dọn theo CLEANUP receipt. Public diagnostics: `FiveC/amis-rewire-checkpoints/analysis/rhade_vietnamese/mbart-large-50/alignment-residual-diagnostics/seed-42`, revision `de794fd0b53fbc51b3f263204b4c193777357271`, 9 files verified. Không tải weights về Windows.


---

## Cập nhật 06/11/2026 — Ê Đê pilot và test đã hoàn tất

Cùng seed42, RTX FP32/SDPA, checkpoint chọn theo validation chrF++, beam1. Test beam4 trên 1.000 câu, BLEU 13a; original data và splits. Chỉ một seed, test đã tham gia development.

| Model | Validation BLEU | Validation chrF++ | Test BLEU | Test chrF++ |
|---|---:|---:|---:|---:|
| Baseline | 34.0232 | 50.4714 | 34.0402 | 50.0957 |
| C0 | 34.0381 | 50.5047 | 33.9203 | 50.0804 |
| LearnedDonor | 34.0729 | 50.4590 | 34.0843 | 50.1812 |
| M0 | 34.3728 | 50.8678 | 34.1761 | 50.1916 |

C0 không đạt gate validation đã định trước (+0.0149 BLEU so baseline; −0.0348 so learned donor; −0.3347 so M0), nên không tự mở replication. Test C0−baseline −0.1200 BLEU, CI95% [−1.0910,+0.5726]; paired bootstrap 10.000, p Holm9=1.0. Cả 9 phép so sánh chưa significant sau Holm. Không suy không có hiệu ứng hay đồng nhất mô hình; bootstrap câu không đo biến thiên giữa seed. Checkpoint và predictions public, báo cáo thống kê revision `2921e8b9dbc58e81f370265bea23d37a0d6a1941`.

---

## Trạng thái hiện hành — 31/10/2026

4selector reruns và test đã hoàn tất, upload verified. Cohort BLEU-selected riêng: C0 test19.6748±.9963 vs baseline19.7306±.4530; delta−.0558BLEU. chrF++18.6874vs15.5468, chrF018.4089vs18.2313. Holm45:2/45significant, gồm **C0 thua baseline seed44 BLEU** và C0 thắng Local seed42 chrF0. Không có gain so baseline significant sau correction. Các bảng chrF++ cũ phía dưới giữ nguyên, không overwrite.

CPU diagnostics6systems PASS: C0 validation BLEU hơn baseline cả3seeds +.4040/+.7841/+.8758; seed44 test giảm−1.7175 dù chọn đúng epoch16. Full details `outputs/experiments/amis_residual_seed_diagnostics/CPU_REPORT.md`; code args `scripts/analysis/analyze_residual_seed_failure.py`.



GPU diagnostics hoàn tất:3seeds×4conditions×576validation rows,0optimizer steps, all-on replay PASS. Tắt module ΔBLEU +.0101/−.0596/+.0050; tắt donor −3.0082/−2.9801/−3.3059; shuffle donor −2.7787/−2.9260/−2.9021. Mạng đã phụ thuộc donor; sensitivity correction lúc inference nhỏ. Không suy ra bỏ correction khi train sẽ tốt, cũng không đủ cơ sở tăng rank/strength. Báo cáo `outputs/experiments/amis_residual_seed_diagnostics/gpu/GPU_REPORT.md`; public artifacts SHA verified HF revision `b3ab39c228a1e49d2fa0f6c25c3d33d5a059bb8a`.

Tác giả đã duyệt và launch queue Ê Đê RTX PID3505181: BASELINE_S42→C0_S42→LEARNED_DONOR_S42→M0_S42. CPU local18tests PASS,4arm roundtrip PASS. Remote CPU rồi cả4GPU gates phải PASS mới train. Recipe `configs/experiments/rhade_alignment_residual_rtx.json`, lock `configs/protocols/rhade_rtx_sdpa_seed42.json`; source id_ID,target vi_VN, original data unchanged, FP32 SDPA B4×GA32 ES10/20epoch, selector chrF++ beam1. Baseline cũ thiếu receipt runtime backend nên giữ lịch sử, chạy baseline SDPA riêng. Validation-only, không tự mở test/replication/A100. Tình trạng gate/train thực tế xem queue.log và receipts, không coi launch là train đã bắt đầu.

---

## Lịch sử — 30/10/2026: mở4 selector reruns sau kiểm kê

Người dùng đã duyệt4 runs RTX chọn theo validation BLEU: C0 S42, learned-donor S42, M0 S42, donor S44. Giữ14 kết quả không đổi; train4 đủ20 epochs, lưu riêng best BLEU/chrF++ và dọn weights sau backup verified. Không giữ20 checkpoint. Recipe/receipt và lệnh theo dõi: [BLEU_SELECTION_RTX.md](../operations/BLEU_SELECTION_RTX.md). Các số test dưới đây vẫn là bảng chrF++-selected cũ, giữ nguyên; chưa có số mới.


Đã xong11 runs mới; test15 checkpoint và dùng lại3 baseline, đủ6 nhánh×3 seeds,575 câu/nhánh. Input/protocol/selection/hash audit PASS. Test beam4 FP32; recipe RTX SDPA B4×GA32 ES10 giữ nguyên. Baseline backend lịch sử được tái dựng từ stack/default, không có runtime receipt đầy đủ.

| Nhánh | Test BLEU mean ± SD |
|---|---:|
|Baseline|19.7306 ±0.4530|
|C0|19.6194 ±0.9454|
|M0|19.6614 ±0.1122|
|Local|19.7045 ±0.8619|
|Donor|19.2226 ±0.5602|
|LearnedDonor|19.2948 ±0.6639|

Validation practical gates đều PASS; test C0−baseline=−0.1112 BLEU, seed gaps +0.7663/+0.6175/−1.7175. Holm45:1/45 significant, là C0 kém baseline seed44 (p=.0269973). Chưa có improvement significant sau Holm. chrF++ +3.0682 dễ gây hiểu nhầm vì test chỉ có3 reference word-bigrams; chrF0 chỉ+.0711. Exact seed tests không có significance. Amis test đã có development exposure.

**Khuyến nghị HOLD nâng C0 thành phương pháp chính/A100**; không tự train thêm hoặc đổi selector. C0 chỉ vượt donor/learned-donor về mean, chưa vượt Local/M0. Báo cáo đầy đủ: `outputs/experiments/amis_alignment_residual_replication_rtx_test/ANALYSIS.md`; statistics, summary và INPUT_AUDIT cùng workspace. Public analysis prefix: `analysis/amis_mandarin/mbart-large-50/alignment-residual-replication/test` trong FiveC/amis-rewire-checkpoints; kiểm tra ANALYSIS_HF_UPLOAD.json để xác nhận backup.

Các phần dưới là lịch sử, giữ nguyên số và quyết định tại thời điểm ghi.


---

# Thí nghiệm RTX6000

## 29/10/2026 — Learned-donor seed42 hoàn tất

CE_LEARNED_DONOR validation BLEU **17.0231**, chrF++ **14.3745**, chrF0 **16.4864**; best epoch20, train đủ20 epoch, stop=epoch cap. C0 tương ứng **17.9412 / 14.9673 / 17.2061**. C0−learned-donor: BLEU **+0.9181**, chrF++ **+0.5927**, chrF0 **+0.7197**. Bootstrap paired 10.000/seed42 trên 576 câu validation: BLEU CI95% [−0.1382,+2.0449], p raw=.046795, Holm3=.102290; cả 3 phép so sánh không đạt p Holm<.05. Kết quả thăm dò một seed, validation đã dùng để phát triển; chưa xác nhận novelty hoặc hiệu quả ổn định.

Checkpoint public `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/alignment-learned-donor-ce-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-rank64/`, upload receipt verified revision `440c15907ad71dcc63939c47889e4f018cd45c6e`, archive SHA256 `6a8fb4aa1d4a86d99494536fc2ded33d9a2961e18b9d17ddc6a856c79f2fff2d`. Đã xác nhận không còn best.zip/best_model/checkpoint-* của run trên RTX. SHA metadata/predictions khớp receipt; protocol và metric replay PASS. Thống kê: `outputs/experiments/amis_learned_donor_ce_rtx/analysis/statistics/PAIRED_BOOTSTRAP.md`. Không đánh giá test.

## Follow-up — lịch sử triển khai learned-donor CE seed42

Một arm `CE_LEARNED_DONOR`, ID `alignment-learned-donor-ce-v1`, công thức
`u+.1[b+A(b)]`; C0 đối chiếu có sẵn dùng `A(b-u)`. CPU16/16scientific tests
PASS; GPU gate PASS (cache/reload/resume, peak14.299GiB); queue đã TRAINING,
PID2506609. Model thêm1.310.720params bằng C0. Cùng protocol lock
FP32/SDPA/LR5e-5/B4×GA32/20epochs/ES10, validation-only. Chưa có quality score.
Recipe `configs/experiments/amis_learned_donor_ce_rtx.json`; receipts và kết quả
tại `outputs/experiments/amis_learned_donor_ce_rtx/`. Không train lại C0 hoặc
hai CE controls; không mở block9/test/A100. Chi tiết và HF prefix ở
[runbook](../operations/ALIGNMENT_RESIDUAL_CE_CONTROLS_RTX.md).

## 29/10/2026 — Matched CE controls và probes đã kiểm chứng

Queue đã hoàn tất. SHA/size của metadata/predictions/state nhỏ tại pinned HF
revisions, resolved Trainer arguments, checkpoint selection và metric replay PASS.
Cả hai controls đủ20epochs, ES10; validation576/beam1/FP32/SDPA, seed42.

| System | BLEU | chrF++ | chrF0 | Best epoch |
|---|---:|---:|---:|---:|
| Baseline R1 reused |17.6823|14.7030|16.9547|19|
| C0 reused |17.9412|14.9673|17.2061|14|
| CE_DONOR, donor/zero correction + target dummy |17.0378|14.2944|16.3933|19|
| CE_LOCAL, local rank64/CE + target dummy |17.0637|14.2563|16.3762|17|

Holm9=3contrasts×3metrics, post-hoc development;10.000paired draws/RNG42:
C0−CE_DONOR BLEU+.9034, CI[−.1104,+1.9326], pHolm=.194581;
C0−CE_LOCAL+.8775, CI[−.2760,+2.0221], pHolm=.239176;
CE_LOCAL−baseline−.6186, CI[−1.7995,+.5569], pHolm=.242976.0/9significant.
C0−baseline+.2589 và thống kê cũ giữ nguyên ở audit pilot, không đổi family.

C0 module-off18.0245BLEU(+.0833); donor-off14.0572(−3.8840);
donor-shuffled14.9135(−3.0277). Probe0optimizer steps; inference intervention
không thay retrained ablation. C2 backbone LSR gradient bằng0 đúng policy,
quality vẫn NO-GO. Không suy gradient conflict từ ratio snapshot.

**HOLD mở block9/test/A100; chưa training mới.** Full report, novelty prior,
HF provenance và kế hoạch có điều kiện:
[ALIGNMENT_RESIDUAL_CONTROLS_REVIEW.md](../research/ALIGNMENT_RESIDUAL_CONTROLS_REVIEW.md).
JSON/CSV: `outputs/experiments/amis_alignment_residual_ce_controls_rtx/analysis/`.
Runbook/HF paths: [ALIGNMENT_RESIDUAL_CE_CONTROLS_RTX.md](../operations/ALIGNMENT_RESIDUAL_CE_CONTROLS_RTX.md).

## Đã hoàn tất và kiểm chứng — Alignment residual C0–C3 (28/10/2026)

Queue hoàn thành4/4, GPU gates4/4PASS, cả4HF backups verified. Metric replay
và checkpoint-selection audit PASS; tất cả đủ20epochs. **C2 quality NO-GO**
theo thresholds khóa trước. Chưa test/A100 hoặc train thêm.

Method mới ở `src/methods/alignment_residual/v1/`, không sửa method cũ.
Config `configs/experiments/amis_alignment_residual_rtx.json`; workspace riêng
`outputs/experiments/amis_alignment_residual_rtx/`. Đây là RTX development,
không điền vào bảng A100. Novelty HOLD.

| ID | Nội dung | Seed | Validation BLEU / chrF++ / chrF0 |
|---|---|---:|---|
| Baseline R1 reused | CE only, SDPA |42| 17.6823 / 14.7030 / 16.9547 |
| AR_C0 | CLRR + module, CE only (capacity control) |42| 17.9412 / 14.9673 / 17.2061 |
| AR_C1 | CLRR + module + LSR full-gradient |42| 16.9277 / 14.1528 / 16.2272 |
| AR_C2 | CLRR + module + LSR confined-gradient |42| 16.8765 / 14.2070 / 16.3203 |
| AR_C3 | Local module + LSR confined-gradient, q0 |42| 17.4056 / 14.5169 / 16.6518 |

Local8scientific tests PASS; bundle CPU4/4PASS; GPU receipts4/4PASS.
Lock SDPA/FP32/B4×GA32/20epochs/ES10 giữ nguyên; reference
baseline42 là R1 RTX đã có. HF tách method ID/seed42/`rtx6000-fp32-sdpa-b4-ga32-es10-rank64`.
Paired bootstrap10.000, RNG42, native SacreBLEU p replay PASS. Locked Holm12:
C2−baseline BLEU−.8058; C2−C0−1.0647; C2−C1−.0511; C2−C3−.5291.
0/12 significant. Post-hoc C0−baseline+.2589, CI95%[−.9107,+1.4384],
Holm p=.644336 trong exploratory12; không đủ chứng minh gain. Toàn18tests
duy nhất cũng không qua Holm. Best epochs C0/C1/C2/C3=14/15/16/18.
C1/C2 thấp hơn C0 cả4nhóm độ dài source; hướng gap không đổi khi bỏ từng câu.
Loss snapshots cho thấy C1 giảm LSR nhưng chất lượng thấp hơn C0; C2 không
phục hồi quality. Không suy ra destructive gradients từ scalar logs. No-GO
đã xác định, chưa cần GPU probe bổ sung để quyết định; nguyên nhân vẫn mở.
Report: [ALIGNMENT_RESIDUAL_PILOT_AUDIT.md](../research/ALIGNMENT_RESIDUAL_PILOT_AUDIT.md).
Artifacts nhỏ (<1MiB): `outputs/experiments/amis_alignment_residual_rtx/analysis/`.
Không sửa số baseline/method cũ. Chi tiết lệnh và checkpoint paths:
[ALIGNMENT_RESIDUAL_RTX.md](../operations/ALIGNMENT_RESIDUAL_RTX.md).

## Trạng thái hiện hành — P1 chạy lại theo baseline SDPA (28/10/2026)

**Kết quả đã kiểm chứng, validation576/beam1/FP32:**

| Seed | Baseline BLEU | P1 BLEU | ΔBLEU | Baseline chrF++ | P1 chrF++ | ΔchrF++ |
|---|---:|---:|---:|---:|---:|---:|
|42|17.6823|16.5836|−1.0987|14.7030|14.2458|−0.4572|
|43|16.7345|16.6308|−0.1037|14.2061|14.0466|−0.1595|
|44|16.2089|17.0428|+0.8339|13.7450|14.2107|+0.4657|
|Mean|16.8753|16.7524|−0.1228|14.2180|14.1677|−0.0503|

chrF0 mean16.3603→16.2730 (−0.0873). Thắng1/3seeds cả3metrics. Mọi model đủ20epochs/720steps; P1 chọn epoch16/15/20, không bị ES. CPU/GPU gates3/3PASS, backup3/3verified. Replay CSV khớp scores6/6; Trainer args chỉ khác output/log paths. Paired sentence bootstrap10000draws, RNG42, Holm9 (3seeds×3metrics):0/9significant, mọi95%CI chứa0. Không suy equivalence từ non-significance; không dùng eager screen seed42 để cộng gộp. **P1 chưa cho cải thiện ổn định theo baseline recipe đã khóa; HOLD mở rộng.** Chưa test.

Artifacts nhỏ: `outputs/experiments/amis_depth_p1_sdpa_rtx/analysis/{VALIDATION_REPORT.md,AUDIT.json,scores_verified.csv,paired_seed_deltas.csv,statistics/paired_bootstrap.json}`. Bảng số cũ được giữ nguyên; trạng thái launch/gate phía dưới là lịch sử.

Tác giả khóa cấu hình baseline RTX cũ và duyệt **3runs P1 seeds42/43/44**, dùng lại baseline cả3seeds. Queue eager baseline/P1 seeds43/44 trước đó đã dừng ở gate, chưa training dài. Không ghép số eager seed42 với cohort SDPA. Chi tiết hiện hành và lịch sử audit: `docs/operations/DEPTH_SCREEN_RTX.md`.

Recipe `configs/experiments/amis_depth_p1_sdpa_rtx.json`; lock `configs/protocols/amis_rtx_baseline_sdpa.json`: SDPA, FP32, LR5e-5, B4/GA32,20epochs/ES10, chọn earliest-best original validation chrF++ beam1; cùng model revision/data hashes/runtime baseline. Chỉ train P1, chưa test/Ê Đê. CPU tests SDPA10/10 và eager regression10/10PASS; status GPU/training phải đọc receipts mới, không suy từ bảng kế hoạch.

Queue mới đã launch detached PID `2058969`, bundle SHA256 `6132f653cbd13ae6e9ca6373663a11262b25ed6dba89e753d81e16e07259d0cb`. CPU bundle3/3PASS; snapshot đầu: P42 GPU gate step1 hữu hạn, chưa hoàn tất cả3gates/chưa train dài. Queue tự gate hết trước rồi train P42→P43→P44. Receipt và trạng thái mới: `outputs/experiments/amis_depth_p1_sdpa_rtx/`.

## Lịch sử 28/10/2026 — Baseline reuse: replication eager tạm giữ

Baseline seeds43/44 đã chạy trong `amis_contrastive_multiseed_rtx`, cùng RTX FP32/b4×GA32/ES10, đủ20epochs. Đã kiểm tra metadata HF: cả hai `VALIDATION_COMPLETE_VERIFIED`, backup verified; selected steps684/720. Checkpoint prefixes `checkpoints/amis_mandarin/mbart-large-50/baseline/seed-{43,44}/rtx6000-fp32-b4-ga32-es10/`.

Queue depth replication mới đã được giữ và dừng trước training dài sau khi tác giả nhắc reuse. Chưa được kết luận cần train lại baseline: recipe depth mới pin eager, metadata cũ không ghi backend nên cần xác minh actual implementation/equivalence. Giữ nguyên số và checkpoints cũ; không tự mở training trùng. Chi tiết [depth runbook](../operations/DEPTH_SCREEN_RTX.md).

## 26/10/2026 — R1 initialization audit hoàn tất

Tái sử dụng và xác minh predictions/metadata ở đúng pinned initialization commit (không tải weights/inference mới). R1: BLEU17.6823, chrF++14.7030, chrF0 16.9547. S0–S3 đều thấp hơn R1 trên cả3 metrics; toàn bộ12 CI95% paired bootstrap chứa0. Không thay scores hoặc selection rule cũ. RTX kiểm tra thấy23.542 MiB trống nhưng không dùng GPU. Không mở training; candidate learned-strength chỉ là known adapted control, novelty HOLD.

Chi tiết: [CONTINUATION_REVIEW_AND_NEXT_DESIGN.md](../research/CONTINUATION_REVIEW_AND_NEXT_DESIGN.md); artifacts `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/initialization_audit/`.

## 23/10/2026 — CPU diagnostics: sequence-risk continuation

Đã tái tính khớp 12 corpus scores của S0–S3 trên validation 576 câu; paired bootstrap 5.000 lần, toàn bộ 21 CI thăm dò chứa 0. S3−S1 BLEU −0.2927, CI95% [−1.1405, +0.5426]; S1−S0 +0.2790, CI95% [−0.4484, +1.0617]. Không có bằng chứng ổn định về gain, cũng chưa chứng minh harm/equivalence.

S1/S2/S3 best epoch 1; S0 best epoch 2 theo validation chrF++. Cả bốn hoàn tất 3 epochs (epoch cap, không phải early stopping); eval loss đều tăng epoch 1→3. Không thấy empty output hoặc lặp trigram ký tự >=3; các proxy độ dài không chứng minh lỗi bỏ sót ngữ nghĩa. Giữ NO_GO_CURRENT_RECIPE; chưa mở rộng training hay đổi trọng số dựa trên scalar loss snapshots.

Báo cáo: `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/cpu_diagnostics/REPORT.md`; code tái lập: `scripts/analysis/analyze_sequence_risk_cpu.py`. 12 file log nhỏ từ HF đã xác minh SHA256 theo commit receipt. Không tải weights, dùng GPU, train hoặc xem test. CI theo câu của một seed không thay cho multi-seed evidence.


## K?t qu? x?c minh 20/10/2026 ? sequence-risk pilot: NO-GO

?? ho?n t?t 4/4 runs tr?n RTX6000, seed42, c?ng baseline R1, 3 continuation epochs; validation 576 c?u, beam1. Ch?a ??nh gi? test. ??y l? continuation controls, kh?ng thay s? baseline g?c.

| Run | C?u h?nh | BLEU | chrF++ | chrF0 |
|---|---|---:|---:|---:|
| S0 | Baseline + CE | 17.2803 | 14.4668 | 16.6790 |
| S1 | Baseline + CE + risk | 17.5593 | 14.4865 | 16.7025 |
| S2 | CLRR + CE | 17.4191 | 14.3061 | 16.4921 |
| S3 | CLRR + CE + risk | 17.2666 | 14.2728 | 16.4245 |

**Kh?ng ??t c?c development gates:** S3?S1 BLEU ?0.2927 (chrF0 ?0.2780), S3?S0 BLEU ?0.0137, interaction BLEU ?0.4315. Baseline+risk t?ng +0.2790 BLEU so v?i continuation CE control, nh?ng chrF0 ch? +0.0235; ch?a c? b?ng ch?ng ? ngh?a th?ng k? ho?c ?? ?n ??nh nhi?u seed. K?t qu? kh?ng ?ng h? gh?p CLRR hi?n t?i v?i risk theo recipe n?y; kh?ng ch?ng minh m?i bi?n th? CLRR ??u v? ?ch.

C? 4 checkpoint ?? upload v? x?c minh hash tr?n `FiveC/amis-rewire-checkpoints`; receipts n?m trong `runs/S*/VAL_COMPLETE.json`. Queue k?t th?c `RTX_PIPELINE_COMPLETE`; kh?ng m? th?m training/test. Ngu?n s?: `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/PILOT_SUMMARY.json`. C?c tr?ng th?i running/hold b?n d??i l? l?ch s?.


## Hiện hành: Amis sequence-risk, shared train pool temperature1.3

Recipe mới `configs/experiments/amis_sequence_risk_diverse_pool_rtx.json`;
workspace cùng tên. T1.3 đã qua train-only probe256rows: singleton55.47%→
10.94%, noEOS/cap0; exactT1 replay8/8PASS. CPU25/25 và four-arm shared
gate PASS. Fullpool4600rows PASS:481singleton(10.46%),491equalcost(10.67%),
meanunique3.2152,noEOS/cap0. Queue PID1620106: **real-dataGPU4/4PASS,
S0 đang maintraining**, log1/108steps đã xác nhận. PublicRUN_DESIGN4folders và
pool/preflight đãverify revision5f9af89e10dbcafcd36add92f7ae8dd248f159f2;
chưa có model training hoàn tất. Old T1 failed attempt dưới đây giữ nguyên,
không resume/waive gate.

S0–S3 giữ cấu hình continuation và baselineR1 đã pin; checkpoint prefixes
mới thêm `-t1p3`. Không có model scores mới để điền ở thời điểm ghi mục này.
[Kế hoạch hiện hành §9](../research/METHOD_UPGRADE_DESIGN.md#9-tiếp-tục-sau-pool-gate-temperature13-namespace-mới).

## Hiện hành 18/10/2026: Amis sequence-risk continuation, bốn arms

**Trạng thái mới nhất: STOPPED BY POOL GATE.** Đủ4600train rows nhưng
2560singleton (55.6522%) và2572equal-cost (55.9130%) vượt gate<50%.
Queue đã kết thúc; chưa chạy real-data GPU gates hoặc train S0–S3. GPU
idle23542MiB free, không có checkpoint training mới. Các trạng thái “queued”
bên dưới là lịch launch, không phải training đã bắt đầu. Giữ nguyên recipe
đã khóa; chưa chọn thay đổi pool/threshold.

**Đã launch detached queue PID1505797**, CPU18/18 unit + shared gate4/4
PASS local/RTX. Full-shape GPU/AdamW preflight **PASS**, peak9.1199GiB,
B4/L256/K5; đang chuẩn bị train pool. Chưa có training model
hoàn tất hoặc số mới. Các mục Ê Đê/Turkish phía dưới là lịch sử, không phải
queue đang chạy. Xem trạng thái live trong workspace, không suy từ banner.

Recipe: `configs/experiments/amis_sequence_risk_continuation_rtx.json`.
Workspace: `outputs/experiments/amis_sequence_risk_continuation_rtx/`.
Shared R1 baseline pinned, FP32/B4/GA32/LR1e-5, seed42, cap3 continuation
epochs, patience10, best validation chrF++/beam1; không test. MRT là known
objective control; đây chưa phải phương pháp mới hoặc confirmation.

| ID | Nhánh | Validation BLEU | chrF++ | chrF0 | Trạng thái |
|---|---|---:|---:|---:|---|
| S0 | Baseline + CE continuation | — | — | — | Queued sau gates |
| S1 | Baseline + CE + sequence-risk | — | — | — | Queued sau gates |
| S2 | CLRR + CE continuation | — | — | — | Queued sau gates |
| S3 | CLRR + CE + sequence-risk | — | — | — | Queued sau gates |

Sequence loss/pool/GO và commands được khóa tại
[METHOD_UPGRADE_DESIGN.md §8](../research/METHOD_UPGRADE_DESIGN.md#8-recipe-đã-duyệt-triển-khai-trên-rtx6000--18102026).
Mỗi model upload vào kho public `FiveC/amis-rewire-checkpoints` với hậu tố
`rtx6000-fp32-b4-ga32-es10-continuation-risk-pilot`; verify backup trước
khi xóa weights RTX. Giữ nguyên tất cả scores/checkpoints benchmark cũ.

## Ê Đê hiện hành: đổi source proxy sang `id_ID→vi_VN`

Tác giả đã duyệt chạy sau khi tra cứu quan hệ Chamic–Malayic. Dùng recipe
`configs/experiments/rhade_contrastive_id_rtx.json`, workspace
`outputs/experiments/rhade_contrastive_id_rtx/`, bốn nhánh cùng sourceproxy.
Queue `tl_XX` PID370375 đã kết thúc, chưa train model dài; chỉ có2GPUgates PASS.
Bundle/receipts `tl_XX` giữ riêng; không reuse gates sang `id_ID`.
Giữ nguyên dataset bytes và toàn bộ recipe train/eval còn lại. Đây là prior
ngôn ngữ học, không phải chọn proxy theo validation/test score. Chi tiết ở
[plan Ê Đê](../research/EDE_SECOND_LANGUAGE_PLAN.md).
Các ghi chú `tl_XX` phía dưới là lịch sử preflight, đã được mục này thay thế.

**Launch hiện hành khoảng22:09, PID373554**. CPU4/4 và GPU4/4 PASS;
**22:17:51 Baseline train thật step5/2360, GPU97%**. Bốn model cùng
610.879.488trainable parameters. Gate receipts public verified revision
`2e0d0f4fac077c30595190053610d653fe26fd4c`. SourceID250032,
targetBOS250024 đã audit. Preflight8artifacts public verified revision
`87c0ff82962df078182b8e2254e9d56132ef41e6`. Bundle SHA
`bcaa8ef6c5b72bc3e4188ac1a630509fc77907f70b527c27795acf96d19b1446`.
Đường dẫn checkpoint cấu hình dùng hậu tố `src-id` để lưu rõ language policy.

## Hiện hành 15/10/2026: Turkish đã dừng, chuyển dataset thứ hai sang Ê Đê

**Ê Đê đã được duyệt chạy và launch RTX khoảng21:59**, PID370375, namespace
`outputs/experiments/rhade_contrastive_rtx/`. Baseline/Full cũ/NCE/P3 seed42;
CPU4/4 PASS local/remote, GPU Baseline PASS lúc22:02. Chờ tất cả4GPUgates PASS
mới train. Chung mBART pinned/FP32/LR5e-5/b4×GA32/ES10/cap20, sourceproxy
tl_XX/targetvi_VN. Test đủ1000rows chỉ sau khóa4validationselections; bootstrap
10k/Holm12 rồi upload/verify. [Plan và commands](../research/EDE_SECOND_LANGUAGE_PLAN.md).

Tác giả yêu cầu dừng Turkish; hai process groups training/post-test đã kết thúc.
Baseline Turkish dừng epoch12 theo ES10, chọn epoch2; validation BLEU20.8872,
chrF++39.5760. Checkpoint đã upload SHA/size verified, revision
`f0d78118747dfcdba57247a88d46079a1e275399`. NCE dừng giữa chừng, P3 chưa train;
chưa test cohort này. Goal theo dõi đã paused; không tự mở lại.

Dataset thứ hai hiện là `data_processed/rhade_vietnamese/`: nguyên văn các
split15.092/1.000/1.000, chỉ chuyển field/container. Đã bỏ thư mục processed
Turkish local theo yêu cầu; kết quả/checkpoints lịch sử vẫn giữ để truy vết.
[Plan Ê Đê](../research/EDE_SECOND_LANGUAGE_PLAN.md) đã duyệt4run seed42,
recipe `configs/experiments/rhade_contrastive_rtx.json` đã đóng băng và launch gates.
Các lệnh và trạng thái Turkish bên dưới là lịch sử, không phải queue đang chạy.

## Turkish: ba runs seed42 — khởi động 14/10/2026

**Cập nhật 15/10/2026, 18:39:** tác giả cho chạy tiếp sau khi tạm ngưng.
Đã xác minh bundle source không đổi, rồi SIGCONT đúng hai process groups
4058571/4060879; không restart hoặc đổi recipe. CPU3/3 và GPU3/3 gates PASS.
Queue đã chuyển sang Baseline seed42, tokenization đủ train20k/validation2000;
post-controller tiếp tục chờ train xong để khóa/test/bootstrap. Receipt RTX:
`outputs/experiments/turkish_contrastive_rtx/RESUME.json`. Thời gian GPU gate
P3 bao gồm khoảng tạm ngưng, không dùng làm số liệu so tốc độ.

Tác giả duyệt Baseline, NCE-only và P3 Full detached trên RTX. Đây là pilot
development để kiểm tra khả năng chuyển sang ngôn ngữ khác, chưa phải confirmation.

| Thứ tự | ID | Objective |
|---|---|---|
| 1 | `baseline-s42` | CE |
| 2 | `nce-s42` | CE + 0.10 InfoNCE |
| 3 | `p3-s42` | CE + 0.10 LSR + 0.10 InfoNCE, CLRR encoder d2/alpha0.10 |

- Recipe: `configs/experiments/turkish_contrastive_rtx.json`; workspace
  `outputs/experiments/turkish_contrastive_rtx/` trên PC và RTX.
- OPUS100 gốc: train20k, validation2000, test2000, giữ toàn bộ strings/thứ tự/
  duplicates. `data_processed/opus100_turkish_english_20k/`; không dùng v2.
- mBART-50 pinned revision, `tr_TR→en_XX`; cùng FP32/TF32off, LR5e-5,
  batch4×GA32, max length256, cap20/ES10, fused AdamW, warmup.06/linear,
  clip1/weight_decay0, checkpointing reentrant. Negative pool InfoNCE là
  microbatch4, không phải effective batch128. Không ghép số baseline A100 cũ.
- Chọn earliest-best validation chrF++ beam1; ghi thêm BLEU13a và chrF0.
  Tất cả ba selections upload/xác minh trước khi khóa và test beam4/FP32.
- Test tự động qua CLI chung `scripts/analysis/evaluate_frozen_rtx.py`.
  Sau đó `paired_checkpoint_statistics.py`: 10000 paired draws/RNG42,
  một family Holm9 = ba cặp × ba metrics. Primary endpoint P3−baseline BLEU;
  không coi significance một seed là độ ổn định giữa seeds.
- CPU3/3 PASS; model/loss source không đổi so với Amis cohort. Chỉ bundle
  builder thêm hai CLI test/statistics vào source archive để tái lập.
- Training/gates PID4058571; post-training controller PID4060879. Baseline
  GPU gate đã PASS, hai gates còn lại đang chạy tại thời điểm ghi. Chưa có số
  training/validation/test dài. Cả hai processes chạy độc lập với chat/SSH.
- Source bundle SHA256 `de236a82b94d700a100b68812f632310aadcd37381878803b907cf422cc4607d`.
  Preflight public SHA/size verified, revision `c883b7799f479908e0fc12222f972e8558be7147`.
- Checkpoints: `checkpoints/opus100_turkish_english_20k/mbart-large-50/`
  `{baseline,nce-only-v1,ordinary-full-v1}/seed-42/rtx6000-fp32-b4-ga32-es10.../`.
  Sau upload xác minh từng model, weights local được dọn. Test restores được
  dọn sau upload kết quả. Stats nằm trong `analysis/opus100_turkish_english_20k/`
  `mbart-large-50/contrastive-controls/seed-42/rtx6000-fp32/test/`.
- RTX shared load có thể tăng runtime. Không tắt host/process người khác;
  queue tự kết thúc sau test/statistics. Lỗi kỹ thuật/upload dừng pipeline,
  giữ weights chưa được backup.

Lệnh training/gates đã dùng:

```powershell
conda run -n clrr python scripts/experiment.py build --config configs/experiments/turkish_contrastive_rtx.json
conda run -n clrr python scripts/experiment.py cpu --config configs/experiments/turkish_contrastive_rtx.json
conda run -n clrr python scripts/experiment.py launch --config configs/experiments/turkish_contrastive_rtx.json --account 1
```

Không launch lại queue đang chạy. Lệnh theo dõi:

```powershell
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/turkish_contrastive_rtx.json --account 1
```

Các lệnh test/statistics được ghi sẵn trong `run_post_training.sh` của workspace,
đợi queue training hoàn tất rồi chạy tự động; đã kiểm tra Bash LF và syntax.

File này ghi các phép đo và pilot của CLRR chạy trên RTX6000. RTX pilot dùng để
sàng lọc phương pháp; kết quả chính của paper vẫn cần được xác nhận bằng các
run A100 cùng protocol.

## Queue12 seeds43/44 — được duyệt13/10/2026

Đã deploy queue nền PID3396045 gồm Baseline, NCE-only A, detached Full P3, LSR+NCE P2,
CLRR+NCE B và attached Full C, mỗi nhánh seeds43/44. Tổng12 training runs;
dùng lại seed42. Local CPU12/12 và metric regression5/5 PASS. Cùng recipe
RTX FP32/b4×GA32/LR5e-5/ES10/cap20; log thêm chrF0, selector vẫn chrF++.
Queue gate→train→validation→upload verified→cleanup, tự exit sau run12.
Remote CPU12/12 PASS, GPU gates đang chạy; preflight7files public SHA verified
ở revision `da7985a5bca0b21a09bc484c7eec644873b29ede`. RTX host được giữ hoạt động.
Không quality-stop giữa hai giai đoạn vì tác giả
đã duyệt đủ12; technical failure giữ weights và dừng. Chưa có scores mới.
Recipe `configs/experiments/amis_contrastive_multiseed_rtx.json`; trạng thái và
thứ tự đầy đủ tại [runbook](../research/CONTRASTIVE_CONTROLS_RTX_RUNBOOK.md).

## Queue đối chứng P3 A/B/C — 12/10/2026

**Kiểm tra không train hoàn tất13/10:** bootstrap10k, một family Holm18 mới:
0/18 đạt ý nghĩa; interaction+1.0614 BLEU có CI chứa0. Audit6 histories PASS;
valBLEU và chrF++ chọn cùng checkpoint, cả6 đủ20epochs. Frozen A diagnostic
576rows/4batch PASS, zero optimizer steps: centered retrieval68.58% S→T,
69.27% T→S; covariance rank119.96/120.06. P3 existing:70.66%/73.44%, rank
103.93/113.62. Chi tiết tại
[evidence report](../../outputs/experiments/amis_contrastive_evidence/EVIDENCE_REPORT.md).

**Test hoàn tất 13/10/2026 theo yêu cầu tác giả:** checkpoint vẫn chọn bằng
validation; đủ575 câu test gốc, beam4, FP32, TF32 off, batch2, lengths256.
Tính lại cả3 metrics từ predictions và đối chiếu source/target PASS. Ba bộ
test artifacts đã public trong đúng folder checkpoint, SHA/size verified tại
revision `c781e6bcaea2bb04af30ac9f6272bcadb2160e1d`. Queue kết thúc và weights
tải tạm trên RTX đã dọn. Máy local chỉ nhận9 JSON nhỏ.

| Nhánh | Test BLEU | Test chrF++ | Test chrF0 |
|---|---:|---:|---:|
| A NCE-only | 19.9879 | 19.0258 | 18.6971 |
| B CLRR + NCE, không LSR | 19.7927 | 18.7785 | 18.4922 |
| C P3, donor attached | 19.8905 | 18.8535 | 18.6160 |
| Baseline RTX reference đã có | 19.3926 | 13.8889 | 18.0917 |
| P3 donor detached reference đã có | 19.9158 | 18.8726 | 18.5920 |

A đứng đầu cả3 metrics trong ba nhánh mới. C−P3 chỉ−0.0253 BLEU; A−P3
chỉ+0.0721 BLEU, nên chưa có cơ sở kết luận donor detach tạo khác biệt ổn định
hoặc Full vượt NCE-only trên test. Thứ hạng validation và test khác nhau;
không chọn lại checkpoint hoặc chốt method dựa trên test này. Bootstrap mới
đã chạy và chưa đạt Holm18. Native chrF++ vẫn cần đọc cùng chrF0 vì word bigrams
Mandarin thưa. Test này phục vụ development sau khi đã xem validation.
Evidence: `outputs/experiments/amis_contrastive_controls_rtx_test/TEST_VERIFICATION.json`.

Đã duyệt và hoàn tất3 runs seed42: A NCE-only, B CLRR+NCE không LSR, C P3 với
donor CLRR attached. Recipe RTX FP32/LR5e-5/b4×GA32/ES10/cap20; validation-only.
CPU formula/checkpoint-gradient/export, shared Trainer gates và cả3 GPU gates
đều PASS trước train; peak GPU allocation12.08GiB. Cả3 chạy đủ20 epochs; chọn
epoch A19/B19/C18 theo chrF++ validation beam1. Checkpoint và artifacts đã
upload, SHA xác minh, local weights đã dọn; queue kết thúc sau C.

| Nhánh | BLEU val | chrF++ val | chrF0 val | So sánh chính |
|---|---:|---:|---:|---|
| A NCE-only | 17.0583 | 14.3556 | 16.4069 | Nền contrastive |
| B CLRR + NCE, không LSR | 16.8872 | 14.2274 | 16.3714 | So A: −0.1711/−0.1283/−0.0355 |
| C P3, donor attached | 17.8040 | 14.7318 | 16.9037 | So A: +0.7457/+0.3761/+0.4968; so B: +0.9168/+0.5044/+0.5323 |
| P3 reference, donor detached | 18.4584 | 15.1246 | 17.3609 | C−P3: −0.6544/−0.3928/−0.4572 |

Thứ tự số delta là BLEU/chrF++/chrF0. Kết quả một seed validation-only, không
có kiểm định thống kê và không dùng test. C thấp hơn P3 reference detached;
đây là tín hiệu để giữ donor detach trong candidate hiện tại. A/B/C và giải
thích đầy đủ xem [runbook](../research/CONTRASTIVE_CONTROLS_RTX_RUNBOOK.md).
Không có thí nghiệm nào khác được mở sau C.

## Chẩn đoán P3 hoàn tất — 12/10/2026

Không training mới. Đo frozen checkpoints R1/R2/P2/P3 seed42: toàn bộ576
validation pairs, cùng4microbatch4 gradient, FP32/dropout off/TF32 off.
Archive SHA, loss decomposition, finite gradients và không đổi parameters PASS;
optimizer_steps=0. Checkpoint tải tạm đã được xóa sau đo.

| Checkpoint | Raw retrieval S→T | Centered retrieval S→T | Covariance effective rank source/target |
|---|---:|---:|---:|
| Baseline R1 | 54.69% | 65.97% | 146.85 / 142.22 |
| Full cũ R2 | 55.56% | 66.49% | 141.03 / 139.75 |
| LSR + InfoNCE P2 | 69.27% | 69.44% | 101.24 / 100.46 |
| CLRR + LSR + InfoNCE P3 | 71.70% | 70.66% | 103.93 / 113.62 |

Retrieval không phải BLEU. P3 có tín hiệu phân biệt cặp dịch có ích nhưng không
claim anti-collapse hoặc novelty đã được xác nhận. Xem
[report và review prior](../research/P3_NOVELTY_AND_MECHANISM_REVIEW.md).
JSON/report public tại `analysis/amis_mandarin/mbart-large-50/p3-diagnostics/seed-42`
trong `FiveC/amis-rewire-checkpoints`, revision
`1d879e1531c98840e07e3f7fa812b15b53fe947c`;12 artifacts nhỏ đã verify SHA.
Các scores pilot/test và receipts không bị ghi đè.

## Token pilot — audit hoàn tất 11/10/2026

Pilot R1–R4 đã hoàn tất train, test và các phép đo không train. Những trạng thái
đang chạy ghi ngày10/10 ở dưới là nhật ký lịch sử. Cả4 runs đạt epoch20 với ES10;
checkpoint được chọn bằng validation chrF++ beam1, test beam4 FP32.

| Run | Phương pháp | Epoch được chọn | Validation BLEU | Test BLEU | Test chrF++ | Test chrF0 |
|---|---|---:|---:|---:|---:|---:|
| R1 | Baseline | 19 | 17.6823 | 19.3926 | 13.8889 | 18.0917 |
| R2 | Full gốc | 14 | 17.1156 | 18.7821 | 18.1317 | 17.6237 |
| R3 | Token-LSR v1 | 19 | 17.5811 | 19.0641 | 18.3830 | 17.9597 |
| R4 | Token-Full v1 | 18 | 16.7533 | 19.3822 | 18.5570 | 18.2176 |

Metadata, data hashes, source bundle, runtime và training arguments khớp giữa
bốn run (ngoại trừ output/log paths). Tính lại toàn bộ metrics validation khớp
CSV đã lưu. Checkpoint archive SHA và strict loading PASS; loss wrapper khớp
phân rã CE+auxiliary ở16/16 trường hợp; gradient hữu hạn và không có optimizer step.

Diagnostics dùng cùng128validation rows,4microbatch4, FP32/dropout off/TF32 off.
Effective covariance rank source: R1=74.380, R2=73.615, R3=69.174, R4=72.460;
không có bằng chứng collapse nghiêm trọng trong phép đo pooled này.
Can thiệp tắt rewiring tầng3–5/6–9/10–12 trên R2 và R4 cho các dấu khác nhau;
tất cảCI chứa0, không có phép so sánh đạt ý nghĩa sau Holm12.

**Lưu ý chrF++:** test chỉ có3reference word bigrams theo tokenization nội bộ;
R1 khớp0, R2/R3/R4 mỗi model khớp1. Mức tăng chrF++ của R4 không thể dùng
đơn độc làm bằng chứng chất lượng dịch tăng rộng rãi. Số gốc được giữ nguyên.

**Quyết định:** Token-Full v1 không vượt gate validation đã khóa; chưa mở thêm
seed hoặc A100. Kết luận này thuộc pilot RTX seed42, không thay thế đối sánh A100.
GPU trước và sau phép đo: free23542MiB, utilization0%; peak allocated tối đa8214MiB.
Bản tải/extract checkpoint tạm của cả4model đã xóa sau khi lưu kết quả, model
gốc còn ở HF revision `9885941b05d2e7d79d7ee694babf553660b402fa`.

Báo cáo: [Token pilot audit](../research/TOKEN_PILOT_AUDIT.md).
Artifacts nhỏ local và RTX: `outputs/experiments/amis_token_pilot_audit/`.
Mã tái lập: `scripts/analysis/audit_token_pilot.py --stage collect|cpu|lengths|launch|monitor|summarize`;
snapshot GPU thực sự đã chạy: `executed_worker.py`, SHA khớp `COMPLETE.json`.

## Đã chạy: kiểm tra effective rank

| Model | Dữ liệu | GPU | Câu validation | Precision | Effective rank |
|---|---|---|---:|---|---:|
| Baseline mBART seed42 | Amis→Mandarin | RTX6000 | 128 | FP32 | 73.096673 |
| Baseline mBART seed42 | Amis→Mandarin | A100 tham chiếu | 128 | FP32 | 73.096670 |

Chênh lệch RTX−A100 là **+0.0000032**. Checkpoint, các câu và batch32 giống
nhau; strict loading và SHA xác minh thành công. Phép đo dùng tối đa 2393 MiB
VRAM. Không có optimizer step.

Artifacts: `outputs_rebuttal/rtx_rank_smoke/COMPLETE.json` và `VERIFIED.json`.

## Token-LSR v1: pilot 4 runs được duyệt — 10/10/2026

**Trạng thái lúc17:01 ngày10/10/2026 (Asia/Saigon):** R1 hoàn thành, chọn
epoch19 sau20 epochs: BLEU17.6823, chrF++14.7030, chrF0 16.9547. `best.zip`
2,466,896,632 bytes cùng8 metadata/metrics artifacts public đã xác minh ở
revision `c431445b9dcc6765530270a5621b801a6162660d`; archive SHA256
`7fbb9e66fc10ac9b1d75a0caaec9dcc314db0d1d5f450329bce5f810b4eec00c`.
R2 đang train step220/720, epoch6.11, best validation chrF++11.9957; R3/R4 chờ.
Queue sống, RTX dùng khoảng93% GPU.
Chưa có BLEU/chrF của bốn runs mới. Queue độc lập với máy PC, chạy tuần tự trên
một GPU dùng chung; không bật Colab và không tác động process người khác.

| ID | Method | Seed | Câu hỏi |
|---|---|---:|---|
| R1 | Baseline | 42 | Mốc CE-only cùng recipe RTX |
| R2 | Full cũ | 42 | CLRR gốc + sentence LSR trên cùng recipe RTX |
| R3 | Token-LSR v1 | 42 | Token objective có ích không? |
| R4 | CLRR + Token-LSR v1 | 42 | CLRR bổ sung gì cho token objective? |

### Recipe đã khóa

- Recipe: `configs/experiments/amis_token_rtx.json`.
- Source: `src/methods/token_lsr/v1/lsr.py`; engine/CLI chung.
- Workspace: `/workspace/ura/thanhdat/projects/CLRR/`; conda `clrr`, cache/tmp/output nằm trong project.
- Quadro RTX6000 24GB; FP32, BF16/FP16/TF32 đều tắt; checkpointing reentrant.
- Batch4 × accumulation32 = effective128, giống nhau cả bốn runs. Recipe này
  khác A100 batch32/accumulation4/BF16; không ghép điểm hai GPU thành đối sánh.
- AdamW fused, LR5e-5, weight decay0, warmup0.06, linear scheduler, clip1.
- Epoch cap20, patience10; validation576 câu, beam1, raw-reference chrF++ chọn
  best checkpoint, ties chọn lần đầu. BLEU tokenizer zh, chrF0 báo kèm.
- Không đọc test để chọn/tune candidate; recipe khóa validation-only.
- mBART revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30` và CSV hashes trong recipe.
- Token loss: final encoder content tokens, bỏ all special IDs/pad;
  cosine cost, balanced entropic OT epsilon0.10/200 iterations; target và plan
  detach; mean per sentence; lambda0.10. Full giữ nguyên CLRR encoder d2/alpha0.10.
- Mỗi GPU stage yêu cầu free VRAM ít nhất20000MiB; queue dừng nếu gate/train/upload
  lỗi. Không tự sửa microbatch, lambda hoặc resume sau lỗi.

### Correctness và checkpoint

CPU đã PASS cả bốn nhánh: finite backward, accumulation32, save/load logits;
Token có padding invariance, target/plan detach, marginal error2.98e-8;
Token wrappers khớp gradients khi bật checkpointing. GPU gate của từng nhánh
chạy2 optimizer steps với actual batch/accumulation, rồi save/load checkpoint.
Cả bốn GPU gates đã PASS (peak allocated khoảng11.991GiB). Queue đã bắt đầu
R1 → R2 → R3 → R4; R1 validation checkpoint epoch19: BLEU17.6823, chrF++14.7030, chrF0 16.9547;
9 artifacts public đã xác minh receipt/SHA tại revision `c431445b9dcc6765530270a5621b801a6162660d`.
R2 đang train step220/720 (epoch6.11), best validation chrF++11.9957; R3/R4 chờ.
Chưa dùng test.
BLEU14.3339/chrF++12.5929 là metrics validation giữa training tại epoch6, không
phải số cuối. Training loss/gradient hữu hạn, không thấy OOM.
Tokenizer roundtrip khớp cả4600 train và576 validation; không có empty content.
Upload public preflight nhỏ đã verify thành công trước training.

Best checkpoint mỗi model được upload public và verify SHA/size trước khi
xóa weights training. Upload lỗi giữ checkpoint và dừng queue.

```text
FiveC/amis-rewire-checkpoints/
  checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/
    rtx6000-fp32-b4-ga32-es10/
```

PC chỉ nhận JSON/CSV nhỏ; outputs ở RTX:
`outputs/experiments/amis_token_rtx/`. Bundle source pin tại
`cache/bundles/amis_token_rtx.zip` trên RTX và `.cache/experiment_bundles/` trên PC.

### GO/NO-GO đã đề xuất và được dùng cho đợt này

Ngưỡng sàng lọc riêng của Token pilot thay cho ngưỡng pilot chung trước đây:

1. R4 hơn cả R1/R2 ít nhất+0.30 validation BLEU, chrF++/chrF0 không giảm: đáng
   kiểm tra thêm seeds.
2. R4 hơn R3 ít nhất+0.20 BLEU, chrF++/chrF0 không giảm: có tín hiệu đóng góp CLRR.
3. R3 tốt nhưng R4 không bổ sung: tiếp tục alignment, chưa bảo vệ claim CLRR.
4. Số nhỏ hoặc metrics trái chiều: chưa mở rộng. Cả hai không hơn controls:
   NO-GO cấu hình v1. Không tự động chạy seeds/A100/Turkish tiếp theo.

Một seed chỉ là development screening, chưa phải ý nghĩa thống kê hoặc bằng
chứng novelty. OT alignment có prior work; chi phí token probes không phải
translation scores.

### Lệnh theo dõi / bàn giao

Trên PC dùng conda clrr:

```bash
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_token_rtx.json --account 1
conda run -n clrr python scripts/experiment.py status --config configs/experiments/amis_token_rtx.json
```

Trên RTX:

```bash
source /workspace/ura/thanhdat/projects/CLRR/environment.sh
cd /workspace/ura/thanhdat/projects/CLRR
tail -n 25 outputs/experiments/amis_token_rtx/queue.log
```

Không launch lại queue đã có `launch_rtx.json`. Resume chỉ một ID chưa xong,
chỉ định checkpoint có optimizer/scheduler/RNG và dùng đúng bundle. Không sửa
source/config trong lúc queue đang chạy. Không tắt RTX shared sau chạy.

### X?c nh?n GPU gates ? 13/10/2026, 14:46 (Asia/Saigon)

- ?? 12/12 GPU gates PASS tr?n Quadro RTX 6000, c?ng bundle manifest `ffec529bd310aaeb3639c1699f10c7e7570a522a1d84f05c6fb26d80b8e640cb`.
- T?t c? receipts x?c nh?n save/reload logits kh?p, reload best checkpoint v? c? optimizer/RNG; FP32, microbatch 4, accumulation 32, checkpointing b?t. Peak allocated VRAM kho?ng 11.99?12.08 GiB trong tiny gates.
- Queue PID 3396045 ch?y ??c l?p (PPID 1, SID 3396045). Training d?i ??u ti?n ?? b?t ??u, log ??t step 2/720.
- Ng??i d?ng x?c nh?n ch? d?ng chat/t?t m?y c? nh?n; queue ti?p t?c to?n b? 12 runs, kh?ng t?t m?y RTX.

### Test ?a seed ???c duy?t ? 14/10/2026

Training 12/12 runs ho?n t?t l?c 05:14, t?t c? backups verified, weights tr?n RTX ?? d?n. T?c gi? cho ph?p test ng?y 14/10. Test workspace: `outputs/experiments/amis_contrastive_multiseed_rtx_test/`; worker d?ng engine test chung, c?ng bundle model g?c. 12 selections ?? kh?a t?i HF revision `b6e40b2d7166f16d5412c211f6465eb25b2027ba`, SHA256 t?ng archive kh?p VAL_COMPLETE. Queue PID 3867190, autonomous; ch?a c? k?t qu? test m?i t?i th?i ?i?m ghi.

Protocol: 575 c?u CSV g?c, beam4 FP32, BLEU zh, chrF++ native word_order2, chrF0 word_order0; kh?ng ch?n l?i checkpoint. Worker t? gh?p 6 checkpoints seed42 ?? test tr??c ?? ?? l?p mean/SD v? paired seed deltas cho seeds42/43/44. Thi?t k? bootstrap kh?a tr??c test m?i: 10000 shared paired sentence draws, RNG42; Holm90 = 10 c?p x 3 seeds x 3 metrics. N?m nh?nh so baseline v? n?m so s?nh: P3?NCE, P3?P2, P3?CLRR+NCE, Attached?P3, Attached?NCE. ??y l? exploratory development; n=3 exact two-sided seed signflip kh?ng th? ??t p<.05 (p t?i thi?u .25). Kh?ng coi sentence bootstrap l? b?ng ch?ng ?n ??nh gi?a seeds.

Artifacts test upload trong prefix checkpoint t?ng run; th?ng k? t?i `analysis/amis_mandarin/mbart-large-50/contrastive-controls/multiseed/test/`. Test queue t? d?ng sau upload x?c minh; kh?ng t?t RTX.

### K?t qu? test v? th?ng k? ?? x?c minh ? 14/10/2026

12/12 test runs m?i ho?n t?t; gh?p 6 checkpoint seed42 ?? test = 18 checkpoints, seeds42/43/44. To?n b? 575 c?u test g?c, beam4 FP32; test kh?ng d?ng ch?n checkpoint. Worker test/metrics d?ng shared engine; predictions order, 18 score replays v? Holm90 ?? ki?m ch?ng ??c l?p local PASS. RTX queue terminal QUEUE_COMPLETE, weights t?i test ?? d?n; GPU nh?n.

| Method | BLEU mean ? SD | chrF++ mean ? SD | chrF0 mean ? SD | BLEU ? baseline |
|---|---:|---:|---:|---:|
| Baseline | 19.7306 ? 0.4530 | 15.5468 ? 2.8916 | 18.2313 ? 0.3200 | +0.0000 |
| NCE-only | 19.6806 ? 0.2858 | 18.7104 ? 0.3443 | 18.3484 ? 0.3667 | -0.0501 |
| LSR + NCE | 19.2135 ? 0.1608 | 18.3675 ? 0.0153 | 17.9450 ? 0.0660 | -0.5172 |
| CLRR + NCE | 19.7895 ? 0.0329 | 18.7009 ? 0.0803 | 18.4312 ? 0.1154 | +0.0588 |
| Full detached (CLRR + LSR + NCE) | 19.9420 ? 0.6351 | 18.8099 ? 0.4091 | 18.5322 ? 0.4984 | +0.2114 |
| Full attached (CLRR + LSR + NCE) | 19.8749 ? 0.3366 | 18.7661 ? 0.2122 | 18.4795 ? 0.2926 | +0.1443 |

| BLEU contrast | Seed42 | Seed43 | Seed44 | Mean ? | Wins |
|---|---:|---:|---:|---:|---:|
| Full detached (CLRR + LSR + NCE) ? Baseline | +0.5232 | +1.0359 | -0.9249 | +0.2114 | 2/3 |
| Full attached (CLRR + LSR + NCE) ? Baseline | +0.4979 | +0.6495 | -0.7145 | +0.1443 | 2/3 |
| Full detached (CLRR + LSR + NCE) ? NCE-only | -0.0721 | +0.9588 | -0.1024 | +0.2614 | 1/3 |
| Full detached (CLRR + LSR + NCE) ? LSR + NCE | +0.8662 | +1.3700 | -0.0506 | +0.7285 | 2/3 |
| Full detached (CLRR + LSR + NCE) ? CLRR + NCE | +0.1232 | +0.7692 | -0.4347 | +0.1526 | 2/3 |
| Full attached (CLRR + LSR + NCE) ? Full detached (CLRR + LSR + NCE) | -0.0253 | -0.3864 | +0.2104 | -0.0671 | 1/3 |
| Full attached (CLRR + LSR + NCE) ? NCE-only | -0.0974 | +0.5725 | +0.1081 | +0.1944 | 2/3 |

Statistics: 10000 paired sentence bootstrap draws, RNG42. Single Holm family90 = 10 frozen pairs x 3 seeds x 3 metrics; all pairwise native SacreBLEU p-values matched independently. 0/90 pass corrected p<.05. Minimum raw p=.00639936; minimum Holm p=.57594241. P3?baseline BLEU: seed42 CI[-.7101,1.7748], seed43[-.1469,2.2581], seed44[-2.0390,.2074]; all three Holm p=1. Exact seed signflip P3?baseline raw p=.75, P3?NCE raw p=1; n=3 minimum attainable two-sided p=.25. Seed signflip values are descriptive raw p, not part of Holm90.

Interpretation: P3 mean BLEU leads this cohort by +.2114 over baseline, but only2/3 seeds win; P3?NCE wins1/3. Attached mean falls .0671 below detached. LSR+NCE falls below baseline3/3. Every candidate is below baseline in seed44. No demonstrated robust superiority, no basis to prefer attached based on validation alone. CLRR+NCE has small absolute BLEU SD=.0329, but baseline-relative advantage is not stable.

chrF++ has large baseline spread (13.8889,13.8656,18.8857), while chrF0 is steadier. Raw-reference replay PASS: this is not an identified scoring error. Sparse Mandarin word n-gram sensitivity was documented earlier; do not present the +3 chrF++ mean gap alone as robust quality improvement.

This remains exploratory development: seed42 test was previously seen, models/Amis test have been inspected. Sentence bootstrap conditions on frozen training runs and does not prove training-seed stability or novelty. Next scientific task is diagnose seed44 and the contribution of LSR; no additional training started here.

Artifacts: `outputs/experiments/amis_contrastive_multiseed_rtx_test/statistics/` and `TEST_VERIFICATION_LOCAL.json`. Public stats revision `2ef77e02ecd237de4b64ef064bd29929ab30154a`, path `analysis/amis_mandarin/mbart-large-50/contrastive-controls/multiseed/test/`; per-run test artifacts are inside each canonical checkpoint prefix. Local collection contains only JSON/CSV/Markdown; no weights.

### Nontraining audit completed ? 14/10/2026

18/18 validation selections PASS; all18ran20epochs. No early-stopping or selector explanation of the seed instability. Full44?baseline short sources?5words:?3.3693BLEU/178rows; whole-test loss remains negative under every single-row deletion. Only3/575refs contain native word bigrams; Full?baseline chrF++ is fragile to row170 (over4.6point influence in seeds42/43). Primary575rows and scores remain unchanged.

GPU probes:8new+4reused after exact batch/data/archive checks; same576validation rows/fourbatch4, FP32 eval, zero optimizer steps, parameter versions unchanged, loss decomposition PASS. Full CE?LSR cosine .0749/.1056/.0819 and CE?NCE .0558/.1179/.0737 across seeds42/43/44; no mean-negative layer groups. End-checkpoint probes do not prove training-time conflict is absent. Weighted LSR/CE norms?.27?.28%, NCE/CE?.82?.91%; instantaneous norms are not Adam update magnitudes.

Centered S?T retrieval mean: Baseline63.89%, NCE69.27%, LSR+NCE69.04%, Full70.08%. Source covariance effective rank mean:144.69/119.11/102.23/105.51. Alignment helps retrieval while BLEU gains are inconsistent. Reduced rank is compression, not proven harmful collapse. Full residual/current RMS?.070?.084 in layers3?12, no seed44 spike.

Conclusion: revisit alignment objective/detail preservation rather than increasing weights blindly or changing patience. No new method/training started. Full evidence: `outputs/experiments/amis_contrastive_multiseed_evidence/NONTRAINING_REPORT.md`. CPU public revision c7eed619c590c0932056ec0b18872223c21efdde; GPU public revision224d68f0fe7ed8bc4c21298858bc0169818a9427. Temporary weights cleaned; NPZ only retained on RTX.
# Frozen route-token validation audit — 2026-11-09

COMPLETE, RTX6000 conda clrr, FP32/SDPA/TF32off, token batch4 đúng historical
audit; no optimizer, no test, no checkpoint reselection.576validation rows,
5494content tokens/model. Same trained weights; disabled additional encoder
CLRR edges, retained every transformer block. Historical seed42 chrF++-selected
snapshots; this does not replace the BLEU-selector baseline or any test table.

| Model | Disabled edges | Repairs | Damages | Δcontent top1 (pp) | Mean Δgold margin |
|---|---|---:|---:|---:|---:|
| CLRR-only P1 |3–12|183|327|−2.6210|−0.367995|
| CLRR-only P1 |3–7|120|165|−0.8191|−0.086179|
| CLRR-only P1 |8–12|114|137|−0.4186|−0.084410|
| Full R2 |3–12|175|341|−3.0215|−0.372549|
| Full R2 |3–7|123|166|−0.7827|−0.087602|
| Full R2 |8–12|94|132|−0.6917|−0.065545|

Repair/damage compares each intervention to its model's native routes, NOT
baseline training. All6net changes negative. Descriptive10000sentence-bootstrap
CIs in `outputs/experiments/amis_route_token_interventions/REPORT.md`; no
confirmatory p-values or training-seed inference. CPU independent CSV replay PASS.
Original checkpoint revision e04195904361dcbdf36d4ae6c7f577ca37eb3256.
P1 archive SHA6456a549965925f424e557f24d5729e3565cb8595f24eeeafd99e4c62c2d597c;
R2 SHA8f06ca99b1eb9d8e5f0ed28f05b4097c0528c1338ba893d0ab85bdf08451e4e0.

Native token tables matched historical NLL/margin to<=3.6e-15 and all correct
flags exactly. B2 attempt failed numerical replay; corrected to original B4,
kept tolerance and failure logs. Zero backbone/layer1control errors. All original
token IDs hash matched; tokenizer warning did not trigger data/tokenizer changes.
15small files (~4.1MB) collected with SHA checks, no weights downloaded to PC;
temporary remote archives/models removed after successful result writing.
GPU23952MiB free/0%, queue ended. Source/provenance and failed-start reasons in
the output folder. Current decision: no fixed group-pruning repair pilot admitted.



## Relational v2 all-training screen — RTX queue launched 09/11/2026

Author approved four Amis seed42 models; recipe
`configs/experiments/amis_relational_screen_rtx.json`, separate implementation
`src/methods/difference_preserving/v2/`. Existing v1 recipes and numbers unchanged.

| ID | CLRR | CE auxiliary (.1) | Status at launch |
|---|---|---|---|
| A_S42 | off | adapted RKD-D distance | queued behind all gates |
| B_S42 | off | difference-direction cosine | queued behind all gates |
| C_S42 | original d2/strength.1 | adapted RKD-D distance | queued behind all gates |
| D_S42 | original d2/strength.1 | difference-direction cosine | queued behind all gates |

Local CPU correctness13/13 PASS; v1 regression19/19 PASS. Bundle ZIP SHA256
`93d03ae38a5f536b3da53ce542f38e83ba5964091fd08fc38c21965103051d0b`.
Queue PID170990, remote output `outputs/experiments/amis_relational_screen_rtx`,
suite `outputs/experiments/amis_relational_screen_rtx_suite`, within
`/workspace/ura/thanhdat/projects/CLRR/`. Launch reported23952MiB free VRAM.
Queue executes remote CPU, all four GPU gates, then sequential A→B→C→D.
A launch receipt does not mean GPU PASS or long training has begun.

Lock: original Amis4600/576/575 CSVs, FP32 SDPA B4/GA32 LR5e-5 ES10/max20,
validation BLEU beam1, pretrained initialization, no test. At most6 unordered
within-microbatch pairs; no minimal-pair labels. Novelty remains UNESTABLISHED.
Coverage/degeneracy diagnostics logged; λ.1 does not equalize gradient strengths.

Historical baseline may be reused for this exploratory comparison: completed20
epochs, best BLEU and chrF++ both epoch19/step684, validation BLEU17.6823291745,
chrF++14.7030053622. Selector was historically chrF++; original artifacts untouched.
SDPA evidence is reconstructed, not a contemporaneous runtime receipt. See
`outputs/experiments/relational_v2_baseline_audit/baseline_reuse.json` and
`docs/research/DIFFERENCE_PRESERVING_PILOT.md` §18 for pinned revision/SHA.
No unverified LSR baseline is used.

Storage per selected model: public `FiveC/amis-rewire-checkpoints`,
`checkpoints/amis_mandarin/mbart-large-50/<method-id>/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu/`.
Weights cleaned only after verified SHA backup. No translation-quality result yet.

Reproducibility backup verified: HF revision `10a6fe3d2a3c775bd3f3c256dec274b2d4778f5f`, prefix
`analysis/amis_mandarin/mbart-large-50/relational-v2-screen/seed-42/reproducibility`. Contains frozen source bundle, PLAN/config/protocol lock,
CPU gate and baseline reuse audit; all file SHA256 values verified after upload.
Receipt: `outputs/experiments/amis_relational_screen_rtx/reproducibility_upload.json`.


### Verified execution update — all gates PASS, long training active

CPU gate13/13 PASS; real GPU gates A/B/C/D all PASS on the same frozen bundle.
Gate durations176.65/176.63/180.78/183.26seconds; peak allocated memory about
14.357GiB each. All four verified active auxiliary loss, strict save/reload,
beam4 cache parity, and optimizer/scheduler/RNG resume. Each gate saw64
microbatches with384/384 valid pairs; this is gate coverage, not full-training
coverage or translation quality. Gate weights/checkpoints were cleaned.

Queue PID170990 has entered actual long training A_S42 (observed step8/720,
GPU96% utilization), then B_S42→C_S42→D_S42. No validation-quality result yet.
Current receipts: `outputs/experiments/amis_relational_screen_rtx/gates/*/PASS.json`,
`PREFLIGHT_SUMMARY.json`, `queue_validation_account1.json`.
An individual gate PASS does not imply the four long runs have completed.


## PLC v1 preparation — 2026-11-14

Author approved implementation and a conditional seed42 Amis pilot, including
a matched Middle Alignment comparator. This is RTX development, not A100 paper
confirmation. No test is opened. Code: `src/methods/prefix_consistency/v1/`;
workflow: `docs/operations/PREFIX_CONSISTENCY_RTX_RUNBOOK.md`.

Five CPU tests passed locally and on RTX. Frozen W42 was measured on all576
validation sentences with8 independent fixed dropout pairs. Supervised special
tokens were retained. RNG replay matched exactly; optimizer steps=0.

| Diagnostic | Sentence mean | Bootstrap95% CI |
|---|---:|---|
| P prefix |0.99115024|[0.93101650,1.05425849]|
| D diagonal |0.96132091|[0.91114443,1.01583050]|
| E endpoint |1.24374525|[1.16663703,1.32322901]|
| Cross=P−D |0.02982932|[0.00053590,0.06009497]|

Dropout pairs were averaged within each sentence before10000 sentence bootstrap
resamples. Cross passes the predeclared positive-lower-bound admission narrowly;
this is not evidence of better BLEU or a method novelty certificate.

Train-only calibration: fixed first64 training rows, microbatch4, pretrained
initialization, identical paired masks for the two gradient objectives, no
optimizer updates. beta_cal=2.563909025096167. This matches initial aggregate RMS
gradient norms, not directions or subsequent Adam updates.

Artifacts: `outputs/experiments/amis_prefix_consistency_preparation_rtx/`
(`prefix_preparation/DIAGNOSTIC.json`, `CALIBRATION.json`, `prefix_drift.csv`,
`LOCAL_VERIFICATION.json`). Preparation queue PID2224993 completed without training.

Conditional training inventory: P/D/E/Wcal seed42 plus matched RTX Middle42.
Baseline/W42 are reused; all new optimizer gates must pass before long training.
No new validation training scores are available at this preparation stage.

PLC execution update: independent local recomputation matched the diagnostic
CSV hash, bootstrap interval and calibration coefficient. Local frozen-bundle CPU
gates passed for all four PLC/control runs and Middle. Suite PID2228354 launched
on RTX, all GPU gates precede training. Baseline42 reuse is backed by its full
validation trajectory: both BLEU and chrF++ select step684/epoch19; SDPA backend
is historically reconstructed. W42 has contemporary SDPA and BLEU metadata.
No new training score is reported yet. Public diagnostic/calibration prefix:
`analysis/amis_mandarin/mbart-large-50/prefix-consistency/seed-42/preparation/`
(four files, read-back hashes verified).


## PLC RTX pilot completed — 2026-11-15

All five runs have VALIDATION_COMPLETE_VERIFIED and public HF backup receipts.
Both queues report QUEUE_COMPLETE; suite log ends SUITE_TRAINING_COMPLETE.
No test was evaluated. These are exploratory seed42 validation beam1 results.

| Model | BLEU | chrF++ | Delta BLEU vs baseline | Delta BLEU vs W |
|---|---:|---:|---:|---:|
| Baseline (reused) | 17.682329 | 14.703005 | +0.000000 | -0.590080 |
| W (reused) | 18.272410 | 15.172807 | +0.590080 | +0.000000 |
| P_PREFIX_S42 | 17.714812 | 14.845499 | +0.032482 | -0.557598 |
| D_DIAGONAL_S42 | 18.271456 | 15.179737 | +0.589127 | -0.000954 |
| E_ENDPOINT_S42 | 17.907733 | 14.970888 | +0.225404 | -0.364677 |
| WCAL_S42 | 17.241326 | 14.421296 | -0.441004 | -1.031084 |
| MIDDLE_S42 | 16.677522 | 13.937579 | -1.004807 | -1.594887 |

PLC fails its predeclared gate: P−W=-0.557598 (<+0.30), and P is below D/E.
P−baseline=+0.032482 and P−Middle=+1.037289 do not rescue the primary failure.
D is effectively tied with W (delta=-0.000954 BLEU); no significance claim follows.
The narrowly positive frozen-W cross-drift diagnostic did not translate into a
benefit from penalizing ordered prefixes at lambda0.10. This does not prove that
cross-position covariance is universally harmful or that other weights would work.
Wcal is lower than W by1.031084 BLEU; this initial gradient-norm match does not
preserve optimizer directions or trajectories. Do not infer that every stronger KL
coefficient fails from a single seed. Middle uses batch4 negatives and is not the
A100 batch32 comparator. Baseline backend retains its historical provenance caveat.

Decision: NO-GO for automatic PLC replication/A100 promotion or further sweep.
No new training/test launched. Preserve all results, including unsuccessful controls.

All five HF best.zip SHA256 metadata values independently match completion receipts;
no .pt/.bin/.safetensors/.zip files remain in the five remote run directories.
Only small JSON receipts were downloaded locally. Verification artifacts:
`outputs/experiments/amis_prefix_consistency_pilot_rtx_suite/HF_COMPLETION_VERIFICATION.json`
and `LATEST_COMPLETE_STATUS.json`.
# PLC post-hoc mechanism audit — 2026-11-15

No new training or test. P/D/E/Wcal/Middle score replay from pinned HF
predictions PASS; all20epochs completed. PLC−W validation BLEU−0.557598 has
descriptive post-selection paired-bootstrap95%CI[−1.567756,+0.365787].
Do not relabel this as statistically established harm or fresh confirmation.

Frozen W/P/D diagnostics completed on16fixed validation rows x2dropout pairs;
partial parameter gradients on four rows x three named blocks.0optimizer
steps, exact RNG replay, CPU analytic/autograd checks, target token-ID and
artifact SHA checks PASS. At W,39.665% token observations allow compensating
PLC gradients;21.229% low-probability tokens carry61.674% of PLC logit-gradient
energy. Partial block conflict alone does not explain failure: ordinary KL
also has negative cosines. No new method admitted and no pilot launched.

Report: [PLC_FAILURE_ANALYSIS.md](../research/PLC_FAILURE_ANALYSIS.md).
Separate artifacts: `outputs/experiments/amis_prefix_consistency_failure_analysis/`.
Temporary downloaded weights removed; all originals remain on HF.
Historical numbers below are preserved.



## SAC exact coupled-mask frozen probe — COMPLETE, 2026-11-15

No training or test. Two pinned existing seed42 checkpoints: baseline and W.
128 evenly spaced Amis validation sentences, two source derangements, two dropout
pairs. FP32 SDPA TF32 off; shared padded shapes and RNG replay across sources.
CPU checks and both native GPU gates PASS. Checkpoint/archive and collected CSV
hashes verified; all lexical summary means replay within1e-10. Transient restored
HF weights removed only inside this probe; GPU free23952MiB after completion.
Only small JSON/CSV/Markdown collected locally, approximately0.6MB.

| Lexical diagnostic | Baseline | W |
|---|---:|---:|
| C = I(M;S given V) |0.01173794|0.00877760|
| B = I(M;V given S) |0.11311421|0.08041024|
| Ratio mean C / mean B |0.10377072|0.10916025|
| Source information |0.40704737|0.40496691|
| Correct-minus-distractor gold logp |4.17587733|4.14305048|
| Absolute source-gap change across masks |1.45389747|1.23111944|

W absolute C lower25.2%; mask source-gap variability lower15.3%. W-minus-baseline
C=-0.00296034, approximate paired sentence-bootstrap95%CI[-0.00364783,-0.00230133].
Source-gap delta=-0.03282684 CI[-0.12790443,+0.06313675]; source-information delta
=-0.00208046 CI[-0.00737780,+0.00315333]. CI crossing0 is not proof of equivalence.
Do not claim selective removal of C: C/B ratio is slightly higher in W.

W C/error Spearman0.3130 falls to partial rank0.03326 after entropy and source/
target character-length adjustment; approximate sentence-bootstrap95%CI
[-0.17089,+0.24019]. This is descriptive, not causal identification. Conditioning
can overadjust or leave confounding; source-pair dependence and training-seed
uncertainty are not covered by these fixed-pairing intervals. Onlyseed42 and an
extensively reused validation set; teacher-forced errors are not generation BLEU.

Decision: SAC training admission remains HOLD; deprioritize three four-forward
pilots because ordinary W already exhibits the targeted stability proxy and no
separate benefit from further reducing C is demonstrated. Not proof SAC cannot
work. No new BLEU/test score, no method-quality claim, no automatic pilot launch.

Full artifacts: outputs/experiments/amis_source_attribution_frozen_rtx/ANALYSIS.md,
ANALYSIS.json, PLAN.json, COLLECTION.json and per-checkpoint CSV/GPU_GATE.json.
Reproduce: python scripts/analysis/operate_source_attribution_probe.py monitor
(or collect); python scripts/analysis/analyze_source_attribution.py --output
outputs/experiments/amis_source_attribution_frozen_rtx. Use conda clrr.

## 2026-11-15 — Frozen decision-branch probe and literature screen

Status: COMPLETE diagnostic; **no new method or training admitted**.

- Exact pinned baseline/W seed42 checkpoints reused from the SAC plan. 32 evenly
  spaced rows of 576 Amis validation sentences, two fixed dropout pairs per row,
  FP32 / SDPA / TF32 off. No optimizer or test access.
- Each checkpoint generates its own native greedy trajectory. At the first
  lexical argmax disagreement between two full-prefix dropout forwards, force
  each proposed action after the SAME prefix and complete with dropout-off
  greedy decoding. This is not a pair of cached stochastic decoding trajectories.
- CPU gate and both native GPU replay gates PASS. Exact source/reference rows,
  checkpoint provenance, CSV SHA and all branch sentence scores replay verified.
  11 remote JSON/CSV files, 62,632 bytes collected; no weights downloaded locally.
  Remote verified transient checkpoint copies removed; project lock free and
  GPU returned to 23,952 MiB free after completion.

| Descriptive quantity | Baseline S42 | W S42 |
|---|---:|---:|
| Pairs with at least one eligible lexical disagreement |54/64|52/64|
| Sentences with any measured event |29/32|29/32|
| Conditional mean absolute branch chrF0 gap |6.7825|8.2054|
| Conditional mean absolute branch chrF++ gap |6.2996|7.5253|
| Partial rank: local symmetric KL vs absolute chrF0 gap |−0.0825|−0.1398|
| Approximate sentence-bootstrap 95% interval |[−0.4197,0.2684]|[−0.5221,0.2773]|

Average events within sentences before associations; adjust for entropy, prefix
length and target character length. 2,000 descriptive fixed-rank sentence
bootstrap replicates. No-event pairs are missing for conditional gap analyses,
not zero-regret observations. Neither masks nor sentences are training seeds.

Interpretation: this small probe does not establish that larger local dropout
KL predicts larger downstream reference-relative branch costs. It does not
justify selecting training positions by KL or claiming a causal exposure-bias
failure. The larger conditional gap for W does NOT show W is worse: its generated
prefixes, chosen positions and event coverage differ. No rollout hit length cap.
chrF0 and sentence BLEU preferred opposite branches in 7 baseline and 4 W events;
one reference cannot establish which paraphrases are semantically valid.

Literature decisions:
- The proposal to sample model prefixes, branch actions, roll out and optimize
  fixed reference-relative local costs is substantially covered by SEARNN:
  https://www.jbalayrac.com/papers/searnn_paper.pdf . Dropout action sampling
  alone is not accepted as an adequate methodological distinction.
- R-Drop remains prior art and the successful W control, not our new method:
  https://papers.nips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf .
- Decoder dropout consistency plus an anti-language-model contrastive objective
  has prior in Section3.4 of What Works and Doesn't Work, A Deep Decoder for NMT:
  https://aclanthology.org/2022.findings-acl.39.pdf .
  Related goals are NOT proof every source-sensitive consistency objective is
  algebraically identical.
- Vocabulary hierarchies are a close prior requiring formula-level comparison:
  https://aclanthology.org/N19-1046.pdf . This is inter-layer hierarchical
  regularization, not proof every final-layer dropout grouping is identical.
- NotebookLM screen overclaimed complete collisions, confused Amis SOURCE
  resource scarcity with Mandarin TARGET grouping, and treated near-zero SAC
  association as contradiction of all source-neglect hypotheses. These claims
  were challenged and corrected; raw replies are retained as unverified research
  assistance, not authoritative novelty evidence. The correction also incorrectly
  describes this probe as independent stochastic trajectory divergence; actual
  implementation is native-prefix forwards, as stated above.

Decision: no blind KL-weighted/selective/on-policy pilot. No claim that all
candidate mechanisms are impossible, or that a novel useful objective is now
available. Further design must identify an intervention and substantive
difference from the closest prior; these diagnostics alone do not supply either.

Artifacts: outputs/experiments/amis_decision_branches_frozen_rtx/
PLAN.json, CPU_GATE.json, per-checkpoint GPU_GATE.json/RESULT.json/decisions.csv,
COLLECTION.json, ANALYSIS.json/md, LITERATURE_SCREEN.json and
LITERATURE_CORRECTIONS.json.

Reproduce (conda clrr):
```powershell
python scripts/analysis/operate_source_attribution_probe.py monitor --output outputs/experiments/amis_decision_branches_frozen_rtx
python scripts/analysis/operate_source_attribution_probe.py collect --output outputs/experiments/amis_decision_branches_frozen_rtx
python scripts/analysis/analyze_decision_branches.py --output outputs/experiments/amis_decision_branches_frozen_rtx
```



## CMC continuation consistency — RTX pilot queue (2026-11-15)

- Author authorized implementation and RTX pilots; no test stage.
- Design/formula/prior-work screen: `docs/research/CONTINUATION_CONSISTENCY_PILOT.md`.
- Recipe: `configs/experiments/amis_continuation_consistency_rtx.json`.
- Separate implementation: `src/methods/continuation_consistency/v1/`.
- Three new Amis seed42 runs: CMC_S42, NORM_S42, PERMUTED_S42.
- Reuse pinned W42 (validation BLEU18.2724095712), baseline42 (17.6823291745);
  no retraining of these controls. Middle remains a secondary existing RTX comparator.
- Protocol locked: FP32 SDPA, LR5e-5, micro4/GA32, cap20, ES10,
  validation BLEU beam1; original CSV references, no dataset modifications.
- Four CMC CPU test groups PASS locally and on RTX; five paired-route regression
  tests PASS locally. GPU gates and long-training results are NOT yet complete.
- Bundle ZIP SHA256: a3b92743013b3abf2fbe4c003a578068e01381b17328624592c87c33e0c9b5e8.
- Manifest SHA256: 867725ec181dda138900477d35b5c738512eba11cd5cd85f8aafccaa6824a49c.
- Detached queue PID2642063, remote output:
  `/workspace/ura/thanhdat/projects/CLRR/outputs/experiments/amis_continuation_consistency_rtx`.
- Initial monitoring: alive; CPU PASS; restoring pinned W for train-only channel
  admission (zero optimizer steps). All three GPU gates must pass before any long run.
- Scientific admission/success thresholds fixed in design before results; no
  semantic-equivalence or global novelty claim. NotebookLM raw screen retained at
  `outputs/experiments/amis_continuation_consistency_rtx/NOTEBOOKLM_SCREEN.json`.
- Checkpoints upload to canonical public FiveC/amis-rewire-checkpoints under
  checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/
  rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu; shared engine removes transient weights
  only after remote SHA verification.

Frozen-W train-only channel admission PASS: retained KL ratio0.6731495449, mean channel separation0.9494252521;32 train rows x2 dropout seeds,0 optimizer steps. This establishes a nontrivial channel, not translation gain. Receipt: outputs/experiments/amis_continuation_consistency_rtx/continuation_probe/ADMISSION.json. GPU gates are now running.


## CMC estimator frozen-gradient audit — 2026-11-16

TRAIN-only, frozen W42, zero optimizer steps. All-parameter conditional position-sampling variance: CV/old pooled ratio **0.083786773**, reduction **91.621%**, improved **8/8** conditions (four short TRAIN rows × two dropout seeds). CPU expectation/identity tests PASS; temporary restored checkpoint removed. Source and data hashes are pinned in `outputs/experiments/amis_cmc_estimator_gradient_audit/COMPLETE.json` and `SOURCE_MANIFEST.json`; full table in that folder's `REPORT.md`. This is not evidence of generation gains or the historical failure cause, and control variates are not a new statistical principle. Two validation-only seed42 pilot arms are authorized conditional on both GPU correctness gates; no new test or replication is authorized.


## Active pilot queue — 2026-11-16

- Recipe: `configs/experiments/amis_cmc_estimator_rtx.json`.
- Detached RTX PID: 3059993.
- IDs: SAMPLED_S42 (plain unscaled sampled SKL, compute-matched probes), then CV_S42 (control-variate CMC).
- Both GPU gates must PASS before either long training run. CPU gates run again remotely against the frozen bundle. Shell uses `set -euo pipefail`, so a failed gate stops the queue.
- Validation-only, no test or automatic replication. Max20epochs, ES10, validation BLEU beam1; unchanged pinned baseline protocol.
- Public checkpoint prefix: `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu/`. Methods: `continuation-sampled-direct-v2`, `continuation-control-variate-v2`. Existing verified backup-before-cleanup policy applies.
- Monitor: `python scripts/experiment.py monitor --config configs/experiments/amis_cmc_estimator_rtx.json --account 1` in conda clrr.

## CMC estimator — completed and CPU review (2026-11-17)

Current status supersedes the preceding active-queue entry: **QUEUE_COMPLETE**, both GPU gates and both validation runs completed. No test or replication opened. Both checkpoint backups have verified SHA receipts in `FiveC/amis-rewire-checkpoints`.

| Amis mBART seed42, original-reference validation beam1 | BLEU | chrF++ | Selected epoch | Completed epoch |
|---|---:|---:|---:|---:|
| Baseline, existing |17.682329|14.703005|existing|existing|
| W, existing dense weak full consistency |18.272410|15.172807|20|20|
| CMC, existing |15.540216|13.316294|18|20|
| Sampled direct SKL, SAMPLED_S42 |16.133234|13.699138|17|20|
| Control-variate CMC, CV_S42 |17.148787|14.439779|19|20|

- New artifacts: `outputs/experiments/amis_cmc_estimator_rtx/runs/{SAMPLED_S42,CV_S42}/VAL_COMPLETE.json`, `validation_metrics.json`, `validation_predictions.csv`, `run_metadata.json`, `stop_state.json`.
- SAMPLED backup revision: `b80c5383b4294c073caf471baa78a6cd42f766ce`; archive SHA256 `ea872658a65c9a11e52c3fc243b736841e4fe24e98c83767bdb05e27bdc9129a`.
- CV backup revision: `b297fea97a1b4cd4ea0a34c9c87b3ca1560310ab`; archive SHA256 `f25e2a4fa72406b216d95ec850dc43149714a66923bffe57a9db412a6e8c0821`.
- Canonical method folders: `continuation-sampled-direct-v2` and `continuation-control-variate-v2`, each under `checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu/`.

CPU review: `outputs/experiments/cmc_estimator_review/REPORT.md`, `ANALYSIS.json`, `INPUT_MANIFEST.json`, `REFERENCE_ERROR_AUDIT.md`, `PRIOR_DIRECTIONS.md`. Only small receipt-pinned artifacts fetched; no weights, GPU or test. All576 validation rows align and W/CMC/SAMPLED/CV replayed metrics match receipts within1e-8. Checked scientific recipe and recorded runtime fields match; micro4/GA32, FP32/SDPA, ES10, cap20, validation BLEU selector unchanged.

[MEASURED] CV−CMC +1.608571 BLEU; CV−SAMPLED +1.015554; CV−baseline −0.533542; CV−W −1.123622. Locked screen requires CV−W>=+0.30, hence **NO-GO** despite beating the other two controls. Both new runs reached epoch cap, not early stopping.

[MEASURED] Last20 logged optimizer-step grad norms average W1.083317, CMC3.533981, SAMPLED4.175565, CV2.008220. CMC/SAMPLED/CV exceed clip threshold1 at72/72 logged steps; W63/72. These are logged pre-clipping norms, not all steps or gradient variance. Last20 logged microbatch CE means W0.090466, CMC0.322632, SAMPLED0.375730, CV0.161463. Final validation CE W2.763629 versus CV2.705764, while W generation is better: CE alone does not establish generation quality. Do not change selector retrospectively.

[MEASURED] CV versus W sentence chrF0:196 better/192 worse/188 tied, yet mean sentence delta−0.859891 and corpusBLEU−1.123622. No empty outputs or repeated4-character substrings occurring>=3 in any of the five576-row files. Length flags are not semantic error labels. A16-row stratified manual reference-relative inspection finds meaning/aspect discrepancies in both outputs at16/16; even all8 metric-gain examples retain discrepancies. This is selected-sample evidence, not a corpus error rate or Amis expert adjudication.

[INFERENCE] Estimator dynamics plausibly contribute, but neither this one-seed comparison nor the earlier eight-condition frozen-W variance diagnostic establishes the unique failure cause. Clipping and Adam are nonlinear; pre-clipping unbiasedness is not update equivalence. [UNKNOWN] Whether exact dense CMC would beat W. Do not claim the continuation objective is proved harmful.

Prior-direction index now covers87 local shared-runner validation receipts, not all historical/HF runs. Separate language/backend/selector/compute cohorts are not pooled. Older ordinary Full seed42 has18.458409 BLEU in its own cohort; do not overstate W42 as universally best or say every older method failed.

Decision: preserve CV as an estimator diagnostic, do not expand training/test or sweep weights. Any new candidate needs a measured causal question and a check against `PRIOR_DIRECTIONS.md` before literature/novelty admission.

Reproduce in conda clrr:
```powershell
python outputs/experiments/cmc_estimator_review/analyze.py
python scripts/analysis/audit_method_evidence.py --output outputs/experiments/cmc_estimator_review/inventory
```

### CMC other-seed grouping, CPU follow-up (2026-11-17)

Artifacts: `outputs/experiments/cmc_estimator_review/shared_errors/`; reproduction: `python -X utf8 outputs/experiments/cmc_estimator_review/analyze_shared_errors.py` in conda clrr. Seven input hashes verified,576 validation rows aligned. W43/W44 define groups without W42/CV42 scores; this remains post-hoc reused-validation analysis, not an independent confirmation.

| Group | Rows | CV42−W42 mean sentence chrF0 |
|---|---:|---:|
| All |576|−0.859891|
| Both other W seeds below10 |185|+0.1233|
| Remaining |391|−1.3251|
| Other W seeds agree after whitespace removal |129|−1.6800|

Sensitivity thresholds5/10/15 and full feature associations retained in ANALYSIS.json. Removing largest absolute delta:−0.745311;5% trimmed mean:−0.309643. Do not substitute these sentence-level figures for corpus metrics. No new training/test/GPU, recipe change or candidate admission. See INTERPRETATION_AND_NEXT_STEP.md for limitations and the next proposed discriminating diagnostic.


## Frozen CMC CE-inclusive update geometry — 2026-11-17 (COMPLETE)

Artifacts: `outputs/experiments/amis_cmc_update_geometry_rtx/INTERPRETATION.md`, `ANALYSIS.json`, `DENSE.json`, `TRIALS.json`, `COMPLETE.json`, `COLLECTION.json`. Reproduce analysis with `python -X utf8 scripts/analysis/summarize_cmc_update_geometry.py` in conda clrr. Fixed W42 checkpoint,128 TRAIN rows,32 microbatches of4,8 conditional position-sampling repeats with identical replayed dropout. FP32/SDPA, no optimizer constructed/updates, parameters unchanged, no test/validation decoding. CPU/GPU gates PASS; native GPU VJP relative error0. Runtime942.20s, peak6819.07MiB allocated; temporary restored weights removed, only small JSON collected and SHA verified.

| Quantity | Result |
|---|---:|
| Dense W auxiliary cosine with CE |+0.395056|
| Dense CMC auxiliary cosine with CE |+0.361411|
| CE-inclusive W/CMC cosine |0.976959|
| Relative total-gradient change CMC versus W |22.4718%|
| Sampled CMC: RMS error / dense total norm; clipped |1.79784;7/8|
| CV: RMS error / dense total norm; clipped |0.776001;1/8|
| Sampled direct: RMS error / dense total norm; clipped |1.94545;8/8|

**[MEASURED]** Both dense auxiliaries align positively with CE on this batch; weighted auxiliary/CE norm ratios1.40717(W),1.21785(CMC). Sampling remains noisy after accumulation32. CV empirical conditional MSE decreases81.3695% versus sampled CMC here; this differs from the earlier91.621% variance result on four short rows. Dense W/CMC total norms0.756193/0.685585 lie below clip1, whereas sampled estimates frequently cross it.

**[LIMIT]** Eight repeats do not establish systematic clipping bias: distances of clipped MC means from clipped exact totals0.457718/0.171450/0.511076(CMC/CV/direct) include conditional Monte Carlo error (trace SE0.305463/0.184057/0.321143). No optimizer-state files exist in this pinned archive; historical Adam effects remain unmeasured. These results do not identify the causal training failure, population effects or dense-CMC translation quality.

**[DECISION]** No new method/training admitted. Avoid another sampler/CV-only proposal, and do not motivate gradient-conflict surgery or larger weights using an unsupported claim of CE conflict/negligible auxiliary gradients. Prior CV training still trails W; estimator improvement alone did not rescue it. New method design needs a discriminating explanation of actual translation failures. Full interpretation and commands are in the artifact report.
