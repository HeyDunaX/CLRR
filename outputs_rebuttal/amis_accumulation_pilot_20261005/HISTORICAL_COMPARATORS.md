# Historical controls for corrected accumulation pilot — 2026-10-05

Read-only local audit, no GPU/training/source edits. Native SacreBLEU2.6.0 recomputation of canonical validation/test CSVs passes, as do prediction hashes and current train/validation hashes against the historical confirmation manifest.

## Exact two controls

| Control | Available historical checkpoint run | Selected checkpoint from metrics | Native val BLEU / chrF++ / chrF0 | Native test BLEU / chrF++ / chrF0 |
|---|---|---|---|---|
| Matched LSR-only, lambda.10 | `mbart-amis-lsr_l010-seed42` | 576, epoch16 | 16.5688 / 14.1674 / 16.2731 | 19.7865 / 18.6797 / 18.3581 |
| Original fixed Full, alpha.10/lambda.10 | `mbart-amis-full_a010_l010-seed42-replica` | 720, epoch20 | 17.6437 / 14.7317 / 16.8752 | 19.9566 / 18.8878 / 18.6626 |

Both available controls were trained in `amis_configuration_confirmation_20261004`; model snapshot revision is `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`. Validation generation is FP32,TF32off,beam1,576 rows; test beam4,575 rows. Existing test hypotheses equal the corresponding preserved original lost-archive runs exactly after outer strip. This verifies test prediction equivalence, not a full-state recovery or fair corrected optimization.

**Do not use V5 inventory hardcoded selected steps432/468 for these two controls.** Their actual `metrics.json` and confirmation summary indicate576/720. Full logs/Trainer history for the two replicas were not found locally; selected suffixes are supported metadata, not a complete training chronology reconstruction.

## Recipe to retain for the two new runs

| Setting | Matched LSR-only | Full |
|---|---|---|
| Method | `jepa` | `jepa-clrr-enc` |
| Routing alpha | 0, no routing | .10 fixed |
| LSR lambda | .10 | .10 |
| Distance/stack | Inactive | d2 encoder |
| Seed | 42 | 42 |
| Train | LR5e-5, WD0, LRwarmup.06, microbatch32 ×accumulation4 | Same |
| Length/eval | Length256,evalbatch8,validation beam1,FP32generation | Same |
| Stopping | Max20epochs,patience4,raw-reference chrF++ selection | Same |
| Precision | BF16,gradient checkpointing; declared trainTF32 | Same |
| Languages | tl_XX→zh_CN,empty sourceprefix | Same |
| Routing warmup | 0 | **0: do not substitute warmup20%** |
| Restart | Fresh pretrained,unique corrected namespace,no auto-resume | Same |

The deliberate difference is corrected effective backward normalization, documented by root's correctness receipts. No architecture,lambda or LR change is required for this diagnostic pilot. If the corrected objective also changes token weighting, record that explicitly; historical CLI equality alone is insufficient to establish equivalence.

## Remaining limits

- Exact historical training dependency/GPU/runtime receipt was not found for these two replicas. Their source/model/data manifest exists; later inference runtime receipts do not prove the original training runtime.
- Missing original selected archives were replaced by independent retraining; no old optimizer should be resumed for corrected training.
- Old metrics remain valid historical weight measurements, but cannot be treated as the corrected cohort's matched controls after accumulation normalization changes.
- One seed pilot estimates sensitivity to the correction; it does not establish superiority or seed stability.
- Existing native test scores are recorded here for provenance only. The pilot should use its locked evaluation plan; do not tune candidate settings using historical test bins.

Machine-readable details, HF archive paths/revisions/hashes, original-loss run links and score-source hashes: `historical_comparators.json`. Extractor: `audit_historical_comparators.py`, executed in conda `clrr`.
