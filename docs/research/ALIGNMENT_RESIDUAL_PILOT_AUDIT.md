# Alignment residual pilot: kiểm chứng validation

## Material Passport

- Mode: experiment validation; Verification Status: VERIFIED for metric replay, input provenance and checkpoint selection; ANALYZED for interpretation.
- Dataset: Amis→Mandarin, 576 validation rows, seed42, RTX6000 FP32/SDPA.
- Source: frozen config `configs/experiments/amis_alignment_residual_rtx.json`; receipts and predictions under `outputs/experiments/amis_alignment_residual_rtx/`.
- Training data, method code, baseline lock and checkpoint selections remain frozen. No test evaluation or new training in this audit.

## Kết quả và quyết định

**C2 NO-GO theo quality thresholds đã khóa. C0 chỉ là tín hiệu development yếu; chưa đủ căn cứ chuyển sang A100 hoặc mở rộng training.**

| Model | BLEU | chrF++ | chrF0 | Best epoch | Completed epoch |
|---|---:|---:|---:|---:|---:|
| Baseline R1 | 17.6823 | 14.7030 | 16.9547 | 19 | 20 |
| C0: CLRR + module, CE | 17.9412 | 14.9673 | 17.2061 | 14 | 20 |
| C1: CLRR + module, full-gradient LSR | 16.9277 | 14.1528 | 16.2272 | 15 | 20 |
| C2: CLRR + module, confined LSR | 16.8765 | 14.2070 | 16.3203 | 16 | 20 |
| C3: local module, confined LSR, no donor | 17.4056 | 14.5169 | 16.6518 | 18 | 20 |

| Locked C2 contrast | Δ BLEU | Required Δ BLEU | Result |
|---|---:|---:|---|
| C2−baseline | −0.8058 | ≥+0.30 | FAIL |
| C2−C0 | −1.0647 | ≥+0.20 | FAIL |
| C2−C1, primary | −0.0511 | ≥+0.20 | FAIL |
| C2−C3 | −0.5291 | ≥+0.20 | FAIL |

The nondegradation requirements also fail: C2 chrF0 is lower than baseline/C0/C3, and C2 chrF++ is lower than baseline/C0. Token repair−damage was not measured; it is unnecessary to decide this NO-GO because the required quality conditions already fail. Do not mark that unmeasured condition PASS.

## Integrity checks

- All five systems use the same pretrained revision, source/target language IDs, data hashes, learning rate, microbatch4, accumulation32, 20 epochs, patience10, original-reference validation chrF++, beam1, FP32 and earliest-best selection.
- Torch2.6.0/Transformers4.57.6 and recorded runtime versions agree. Historical baseline metadata lacks an explicit attention field; the archived `baseline_reuse_audit/backend_probe.json` resolves its default to SDPA. This is reconstructed evidence, not a field added to the old run.
- Four new runs' resolved Trainer arguments match the critical frozen settings; four GPU PASS receipts are present. All four ran to epoch20. Best checkpoints selected at epochs14/15/16/18 agree with maximal recorded validation chrF++.
- Recomputed corpus BLEU/chrF++/chrF0 reproduce all five receipts within 1e−8. All CSV source/target rows match the original validation CSV in order.
- Small HF metadata/loss/state files were fetched at pinned upload revisions and checked against recorded SHA256. Local collected JSON/CSV have Windows CRLF; restoring LF alone exactly matches HF bytes. This transport difference does not change scores.
- No checkpoint weights were downloaded to Windows. Existing HF backup receipts were verified by the original upload process; this audit verifies fetched small files, not a new full download of archived weights.

## Paired bootstrap

10,000 paired sentence resamples, RNG42, corpus metrics; p-values independently matched to native SacreBLEU. The locked family remains four contrasts × three metrics =12 tests. Additional comparisons requested after training form a clearly post-hoc family12. A conservative sensitivity report also corrects all18 unique tests together; none passes .05.

| Comparison | Δ BLEU | Marginal 95% bootstrap CI | Native p | Holm p, stated family |
|---|---:|---|---:|---:|
| C2−C1 | −0.0511 | [−1.0470, +0.9299] | .365663 | .874713, locked12 |
| C2−C0 | −1.0647 | [−2.0663, −0.0913] | .017698 | .176982, locked12 |
| C2−C3 | −0.5291 | [−1.4534, +0.4146] | .107389 | .644336, locked12 |
| C2−baseline | −0.8058 | [−1.9769, +0.3430] | .073593 | .588741, locked12 |
| C0−baseline | +0.2589 | [−0.9107, +1.4384] | .230177 | .644336, exploratory12 |
| C1−C0 | −1.0135 | [−2.1669, +0.1095] | .040296 | .282072, exploratory12 |

CIs are marginal percentile intervals, not simultaneous intervals or inversions of the native SacreBLEU p-value procedure. A negative marginal CI is not evidence of Holm-adjusted significance. Sentence uncertainty conditions on one trained seed and reused validation; it does not account for checkpoint selection, repeated method development or training randomness. Nonsignificance does not establish equivalence.

## Độ dài câu và ảnh hưởng từng câu

| Source word count | Rows | C0−baseline BLEU | C1−C0 BLEU | C2−C0 BLEU | C2−C3 BLEU |
|---|---:|---:|---:|---:|---:|
| ≤5 | 174 | +0.7656 | −1.2061 | −0.3838 | −0.3403 |
| 6–7 | 193 | +0.5147 | −0.5676 | −1.5823 | −0.4081 |
| 8 | 90 | −0.3370 | −1.4812 | −1.1855 | −0.3541 |
| ≥9 | 119 | −0.0331 | −1.1386 | −0.9481 | −0.9541 |

C0's positive aggregate is concentrated in shorter sources; it does not improve every length group. C1/C2 fall below C0 across every source-length group, and C2 falls below C3 across every group. These are descriptive, post-hoc subsets, not additive contributions to whole-corpus BLEU or separately tested discoveries.

Removing any single validation row leaves C0−baseline BLEU positive: range +0.0827 to +0.3966. C1−C0, C2−C0 and C2−C3 stay negative under every single-row deletion. Therefore a single row does not explain the direction of these gaps; this does not establish robustness to resampling or new seeds. Zero exact normalized source overlaps with training were found; this is not a semantic leakage audit.

## Loss và điều có thể kết luận về cơ chế

Loss JSONL contains last-microbatch snapshots, including repeated entries when evaluation/save callbacks log; it contains no CE–LSR gradient vectors. Deduplicating by optimizer step leaves88 snapshots per run. The first10/last10 medians below are descriptive snapshot summaries, not epoch means.

| Run | First10 median LSR | Last10 median LSR | Last10 median CE |
|---|---:|---:|---:|
| C0 | 0 | 0 | .0887 |
| C1 | .4563 | .1825 | .1189 |
| C2 | .5915 | .6139 | .0977 |
| C3 | .5893 | .5808 | .1048 |

C1 reduces its logged alignment objective while obtaining worse generation metrics than C0. This supports a mismatch between reducing this surrogate and improving translation in this pilot. It does not prove destructive gradient conflict or harmful representation collapse.

C2's logged alignment loss does not fall like C1's, and confining auxiliary gradients to the module does not recover C0 quality. Possible explanations include limited module influence or difficulty optimizing this objective under confinement. The logs do not distinguish those explanations. Do not increase λ or module rank solely from these numbers.

C1/C2/C3 `eval_loss` includes weighted LSR; C0 includes CE only. Direct cross-arm comparison of those eval losses as if all were pure CE would be wrong. Training snapshot CE values also cannot replace a matched validation CE probe.

No new GPU probe is necessary for the quality NO-GO decision. Causal attribution remains unresolved: a future mechanism-specific question requires same-batch CE/LSR gradient measurements and module response ablations on frozen checkpoints. Endpoint probes cannot retrospectively establish training-wide conflicts. No such measurement is claimed here.

## Bước tiếp theo

1. Stop expansion of C2: no automatic test, extra seeds, λ sweep or A100 confirmation.
2. Keep C0 as an exploratory reference. Its +.259 BLEU is uncertain; it is CLRR plus added capacity, not a clean demonstration that the module improves CLRR-only. Audit an existing matched CLRR-only comparator before proposing another control run.
3. Before new method code/training, state one mechanism hypothesis and its distinguishing frozen-checkpoint measurement. If the question is LSR conflict, measure CE/LSR gradients on the module separately from the backbone, with the actual full/confined policies. If the question is useful donor information, intervene on donor/module separately and retain matched dropout/RNG.
4. Only after that evidence, choose a single candidate and predefine controls/thresholds. C0 seeds43/44 are an option for checking the weak capacity signal, not an already-authorized automatic expansion or a novelty claim. Baseline seeds must pass reuse audit.

Novelty stays HOLD; this audit neither resolves the CAFE overlap nor establishes a new contribution.

## Statistical interpretation checks: 11/11 reviewed

Simpson: length-group heterogeneity reported; no aggregate-to-subgroup generalization. Ecological inference: corpus effects are not claims about every sentence. Berkson/selection: repeatedly used validation and best-checkpoint selection limit inference. Collider: no outcome-conditioned subset used for primary statistics. Base rates: not a diagnostic classifier, not applicable. Regression to mean: seed42/C0 were not independently confirmed. Survivorship: all4 planned arms included and completed. Look-elsewhere: locked12 retained; exploratory12 labeled; union18 sensitivity supplied. Forking paths: no selector/threshold changed after results. Correlation/causation: no claim that gradient conflict or collapse caused quality loss. Reverse causality: alignment loss association is not used to identify causal direction. These checks do not remove the limitations.

## Artifacts and reproduction

Analysis folder: `outputs/experiments/amis_alignment_residual_rtx/analysis/` contains `AUDIT.json`, `LOSS_SUMMARY.json`, `ALL_COMPARISONS_HOLM.json`, `locked_statistics/`, `exploratory_statistics/` and `strata/`. All are small JSON/CSV/Markdown inputs/results.

Use Python in conda `clrr`. Prepare verified inputs with:

```bash
python scripts/analysis/audit_alignment_residual_pilot.py --cohort outputs/experiments/amis_alignment_residual_rtx --baseline outputs/experiments/amis_token_rtx/runs/R1 --config configs/experiments/amis_alignment_residual_rtx.json --output outputs/experiments/amis_alignment_residual_rtx/analysis
python scripts/analysis/paired_checkpoint_statistics.py --predictions-dir outputs/experiments/amis_alignment_residual_rtx/analysis/inputs --references data_processed/amis_mandarin/validation.csv --design outputs/experiments/amis_alignment_residual_rtx/analysis/locked_design.json --output outputs/experiments/amis_alignment_residual_rtx/analysis/locked_statistics --split validation
```

Repeat bootstrap with `exploratory_design.json` and `exploratory_statistics/`. Run the audit command again afterward to create the all-unique-comparison sensitivity summary. Strata use the existing `scripts/analysis/analyze_prediction_strata.py`, the same inputs, `--split validation`, and `--train data_processed/amis_mandarin/train.csv`.
