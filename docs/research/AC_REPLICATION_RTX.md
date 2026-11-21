# AC replication on RTX

Author authorized two additional runs on 2026-11-17: Amis/mBART AC seeds43 and44, after the existing DRM training/test dispatcher completes successfully. Seed42 is reused, not retrained.

## Scope

Exploratory replication following inspection of seed42 test. AC failed the original validation promotion threshold versus W; this author-authorized follow-up does not retrospectively change that result. AC is an engineering candidate with limited novelty, not an established ACL contribution. The mechanism and prior/debate record remain in TWO_METHOD_DESIGN_ROUND1.md. Additional seeds cannot establish novelty by themselves.

## Frozen design

Same AC algorithm, beta1.25, pretrained revision, original data hashes, FP32/SDPA, B4 GA32, LR5e-5, cap20, ES10, validation BLEU beam1/earliest-best. Only seed/data_seed, run ID and matching canonical HF seed folder change. Baseline/W42-44 and AC42 are reused. No test stage or A100 stage is authorized by this recipe.

Both fresh IDs pass CPU, the existing fixed-W TRAIN mechanism probe and individual GPU gates before either long training run. The fixed-W probe intentionally retains seed42/four TRAIN rows; it is a method admission check, not an independent seed result. The probe runner now dispatches by method rather than the literal AC_S42 ID; no mathematical change.

The queue stages its bundle without altering active DRM source; it waits for COMPLETE_VERIFIED and project lock before deployment. It rejects existing HF destinations, verifies frozen source/data/runtime and runs sequentially. Upload each checkpoint to public FiveC/amis-rewire-checkpoints with readback/hash verification; remove transient weights only after verified backup using the shared engine. Failures preserve evidence and stop; no process belonging to another user is touched.

## Evaluation after completion

Report every seed and failure. Compare paired validation deltas versus W and baseline, mean and dispersion. Test42 is development evidence; no claim of independent confirmation. Future test and paired bootstrap/seed-level analysis must preserve fixed validation-selected checkpoints. R-Drop norm-matched/tuned controls and broader-language evidence remain necessary for a paper claim.
