# Corrected controls — 05/10/2026

## Final status — checked 17:00, Asia/Saigon

All six trainings completed. Independent local prediction replay and remote archive/metadata checks passed. Colab 2 stopped at 16:04:35; Colab 1 stopped at 16:10:08, with explicit session termination receipts. The execution notes below describe launch-time status.

| Seed | CLRR-only test BLEU | CLRR-only test chrF++ | LSR lambda .03 validation chrF++ |
|---|---:|---:|---:|
| 42 | 19.223095 | 18.362492 | 13.906081 |
| 43 | 19.492997 | 18.531956 | 14.066061 |
| 44 | 19.345498 | 18.374130 | 14.178781 |

LSR lambda .03 mean validation chrF++ is **14.050307473361284**, below lambda .10 **14.221997785916189**, and lower at every matched seed. The validation decision selects **lambda .10**. Lambda .03 test predictions remain blind, as planned; this is a completed tuning experiment.

CLRR-only mean test BLEU is approximately **19.3539**, versus corrected baseline **19.3886**, LSR lambda .10 **19.3930**, and Full **20.1303**. CLRR alone does not improve mean BLEU here; Full exceeds CLRR at all three seeds and exceeds LSR lambda .10 at all three seeds. This supports investigating the combination, but does not establish statistically significant superiority or ACL readiness. Prior corrected Full-versus-LSR statistical caveats remain applicable.

All six model backups are in the two repositories listed below. Small artifacts and verification receipts are local under `outputs_rebuttal/amis_corrected_controls_20261005/`. Old experiment scores remain unchanged.

## Scope and purpose

Six authorized mBART Amis runs, seeds 42/43/44. This completes the corrected G0 controls before deciding whether to upgrade the method.

| GPU | Family | Question | Backup repository |
|---|---|---|---|
| Colab 1, A100 Standard 40GB | LSR-only, lambda .03 | Does Full beat an LSR control selected using validation? | FiveC/amis-corrected-lsr-l003-20261005 |
| Colab 2, A100 Standard 40GB | CLRR-only, alpha .10, lambda 0 | What does rewiring contribute without LSR? | FiveC/amis-corrected-clrr-only-20261005 |

Both jobs launched successfully, PIDs 3940 / 7642. Each runs GPU preflight before long training. Training progress and gate outcomes still require monitoring.

## Locked protocol

Same corrected source and data as the preceding cohort. Model revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; LR 5e-5, batch 32, accumulation 4, BF16, gradient checkpointing, lengths 256, maximum 20 epochs, early stopping patience 4, warmup .06, weight decay 0. Validation chrF++ with greedy decoding selects checkpoint; frozen test uses beam 4, FP32 generation, Chinese BLEU tokenization. No automatic resume.

Bundle SHA256: `e6889d3356e18d59b24700d843a59c3558d35a92206dd531e364a10fc4ff872b`. Per-file hashes are in `outputs_rebuttal/amis_corrected_controls_20261005/bundle_manifest.json`. Do not regenerate using the stale initial scaffolding script `scratch/build_corrected_controls.py`.

## Validation selection and test access

Train all three lambda .03 seeds with validation only. Compare their mean validation chrF++ with corrected lambda .10 scores: 14.190173628560752, 14.274377140832009, 14.201442588355809. Lock and upload `lsr_selection_decision.json` before test access. A strict validation win selects .03; a tie selects .10. Only test .03 if selected; otherwise keep all .03 test predictions blind.

CLRR-only has a fixed configuration and receives test evaluation after its validation checkpoint selection. Test results do not select a configuration.

## Preservation and completion

Each run must upload its selected model archive and selection provenance, verify HF checksums, collect small artifacts locally and replay metrics from predictions. Only then delete checkpoint weights on Colab. Keep lambda .03 model archives until the validation decision and any required test evaluations finish.

Stop each GPU independently after its job finishes and all required local and HF verification passes. Connection failure requires retrying monitoring, not restarting training. Old experiment numbers remain unchanged. This file currently records execution setup; no new scores or scientific conclusion are available yet.
