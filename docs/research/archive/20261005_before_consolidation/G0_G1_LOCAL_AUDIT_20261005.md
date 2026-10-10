# G0/G1 local audit — 2026-10-05

## Status and scope

**TECHNICAL_HOLD.** This audit reads existing artifacts and recomputes validation metrics; it does not train or change scientific records. The independent accumulation correctness check performed in this audit phase found wrapper/baseline scaling differences: see `outputs_rebuttal/amis_g0_g1_20261005/accumulation/receipt.json` and the root investigation. Equality of command-line hyperparameters is not certification of equivalent optimization. The observations below characterize existing checkpoints; they do not establish a fair method comparison until affected training cohorts are resolved.

All local Python ran in conda `clrr`. No token/launcher read, GPU action, historical file overwrite, or manuscript change was performed by this local audit.

## G0 — evidence and reuse

The corrected V5 inventory contains 15 controls in the five core families, seeds 42/43/44. Twelve FT/Full/matched-LSR/tuned-LSR validation prediction paths and SHA256 values pass the local hash audit. The three CLRR-only entries are explicitly historical and were not remeasured in V5. Their later FP32 beam1 replay is available in both warmup machine control receipts, copied into the new audit directory; do not interpret missing V5 paths as missing checkpoints.

| Family | Existing weights/predictions | Reuse limit |
|---|---|---|
| FT | Three HF archive identities; corrected V5 validation predictions | Baseline42 original best step is unknown; 216 was an unsupported historical label. Generation is verified, not original checkpoint selection chronology. |
| CLRR-only | Three archive identities; both warmup-machine native validation replay receipts | Preserve replay receipt as source for chrF0; original V5 inventory lacks chrF0. |
| Matched LSR .10 | Three identified archives and verified V5 predictions | Existing checkpoint description only while effective optimization mismatch is investigated. |
| Tuned LSR .03 | Three identified archives and verified V5 predictions; both-machine exact replay | Development-selected comparator; not globally optimal lambda and not independent confirmation. |
| Full .10/.10 detached | Three identified archives and verified V5 predictions; both-machine exact replay | Existing checkpoint description only while effective optimization mismatch is investigated. |

Archived 20261003 `modeling.py` AST confirms that both `CrossLayerResidualRewire.forward` and `JEPAGuidedSeq2SeqLM.forward` accept `**kwargs` and pop `num_items_in_batch`; archived evaluation code has no `accepts_loss_kwargs` occurrence. This corroborates the historical wrapper pattern. It alone does not prove the exact Trainer/runtime of every historical run. It must not be described as an Adam learning rate being multiplied by four.

Outputs: `control_inventory_audit.csv`, `control_runtime_replay_receipts.json`, `control_configuration_receipt_copy.json`, `archived_wrapper_ast_receipt.json`.

## D1 — warmup selection and stopping

Selection was native raw-reference validation chrF++, beam1, fixed evaluation alpha=.10. All six selected checkpoints equal the maximum logged chrF++ checkpoint and also the maximum logged BLEU checkpoint. All selected prediction CSVs match the canonical 576 validation source/target rows and COMPLETE hashes. Native SacreBLEU 2.6.0 recomputation reproduces the selected scores.

| Run | Best epoch | Endpoint | Best BLEU | Endpoint BLEU | Best−endpoint BLEU | Stopped early? |
|---|---:|---:|---:|---:|---:|---|
| Full seed42 | 20 | 20 | 17.7908 | 17.7908 | .0000 | No |
| Full seed43 | 20 | 20 | 16.2231 | 16.2231 | .0000 | No |
| Full seed44 | 18 | 20 | 17.0006 | 16.9565 | .0441 | No |
| CLRR-only seed42 | 18 | 20 | 16.6623 | 16.5244 | .1379 | No |
| CLRR-only seed43 | 8 | 12 | 14.4003 | 14.1922 | .2082 | Yes, consistent with patience4 |
| CLRR-only seed44 | 20 | 20 | 16.8898 | 16.8898 | .0000 | No |

Training alpha records pass `.10 * min(step_before_update / 144, 1)` for every recorded update, with evaluation alpha always .10. There are 720 planned optimizer steps. Five runs reach epoch20; CLRR43 reaches epoch12. This is not a finding that all runs stop prematurely.

Minimum validation CE occurs at epochs5–6 for all six runs, while generation-selected bests are mostly epochs18–20. Lower teacher-forced CE and better generation do not peak together. Do not change selection to CE after examining these results.

The selected-best warmup Full mean is 17.0048 BLEU, versus fixed Full 17.2135 and tuned LSR16.9942 on the same evaluation protocol. The warmup verdict cannot be rescued by correcting an endpoint/best mixup: Full42/43 already use endpoints; Full44's actual selected best is used. These differences remain conditional on the optimization correctness investigation.

**D2 is optional**, useful to characterize routing dependence of warmup best weights. It is not needed to re-establish selected-best selection correctness or the current warmup gate failure, and cannot resolve gradient accumulation fairness.

Outputs: `warmup_history_selection_audit.csv`, `audit_summary.json`.

## D4 — existing V5 error analysis

All prediction files referenced by V5 test/validation error analysis match their recorded SHA256 values. Bins are raw source whitespace word counts. They are descriptive, not morphology annotations; no new hypothesis is selected from test and then called independent confirmation.

| Comparison, mean across three seeds | Split/bin | Rows per seed | BLEU delta | chrF0 delta |
|---|---|---:|---:|---:|
| Full−FT | Test all | 575 | +.3800 | +.3263 |
| Full−matched LSR | Test all | 575 | +.2344 | +.1675 |
| Full−tuned LSR | Test all | 575 | −.1267 | −.1635 |
| Full−tuned LSR | Test <=5 | 178 | −.2760 | −.3157 |
| Full−tuned LSR | Test 6–10 | 368 | +.0232 | +.0091 |
| Full−tuned LSR | Test >10 | 29 | −1.0257 | −1.1509 |
| Full−tuned LSR | Validation >10 | 38 | +1.9546 | +1.3022 |

Long-bin test harm contrasts with validation benefit, and bins are small. There is no consistent basis here for claiming a long-sentence mechanism or choosing a length-specific upgrade. Test references contain only three word bigrams in three rows; chrF++ deltas are not evidence for morphology. The existing chrF0 sensitivity/influence audit remains necessary.

Outputs: `v5_error_prediction_hash_audit.csv`, `v5_error_bins_descriptive.csv`, `v5_error_comparisons_descriptive.csv`, `v5_error_comparison_means.csv`.

## Next decision

1. Resolve accumulation correctness and map affected cohorts before training upgrades or declaring superiority.
2. Keep old metrics as genuine measurements of old checkpoints, separately labeled by their recipe. Do not overwrite them or relabel them as corrected training.
3. D3 may describe the current target anchor and gradient behavior, but upgrade efficacy requires a fair corrected pilot.
4. No Turkish run and no new candidate training is authorized by this diagnostic report.

Reproduce only this local audit:

```powershell
conda run --no-capture-output -n clrr python -X utf8 outputs_rebuttal/amis_g0_g1_20261005/audit_local.py
```
