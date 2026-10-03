# Asháninka → Spanish: baseline and CLRR+LSR

## Protocol frozen before training

Dataset: AmericasNLP 2021 commit `d3f519c6b38299d8149311e65a369047350e6849`.
Splits: 3,883 train / 881 validation / 1,003 test. CSV hashes must match
`data_processed/ashaninka_spanish/manifest.json`.

| Setting | mBART baseline and CLRR+LSR | NLLB baseline and CLRR+LSR |
| --- | --- | --- |
| Backbone | `facebook/mbart-large-50-many-to-many-mmt` | `facebook/nllb-200-distilled-600M` |
| Source proxy / target | `es_XX` / `es_XX` | `spa_Latn` / `spa_Latn` |
| Seed / data seed | 42 / 42 | 42 / 42 |
| Learning rate | 5e-5 | 5e-5 |
| Optimizer | AdamW torch fused on CUDA | AdamW torch |
| Weight decay | **0.0**, matching actual previous trainer defaults | **0.0**, matching actual previous trainer defaults |
| Betas / epsilon | 0.9, 0.999 / 1e-8 | 0.9, 0.999 / 1e-8 |
| Warmup / scheduler | 6% / linear | 6% / linear |
| Device batch × accumulation | 4 × 32 = 128 | 16 × 8 = 128 |
| Eval batch | 8 | 16 |
| Max source / target length | 256 / 256 | 128 / 128 |
| Max epochs / patience | 20 / 4 | 20 / 4 |
| Validation / test beams | **1 / 4**, matching earlier mBART runs | **4 / 4**, matching earlier NLLB runs |
| Selection | highest validation chrF++ | highest validation chrF++ |
| Precision | BF16 autocast, TF32 | BF16 autocast, TF32 |
| Gradient checkpointing | enabled for both methods | disabled for both methods |
| Official test metrics | original CSV references, BLEU 13a, chrF++ word_order=2 | same |
| CLRR+LSR only | encoder, d=2, alpha=0.1, lambda=0.1 | same |

The final partial training batch can contain fewer than 128 examples. Fairness
means matched conditions within each baseline/CLRR pair; backbone-specific
lengths, validation beams and optimizer implementation are documented above.
This run is single-seed evidence, not a multi-seed average or a tuning sweep.
Test data is never used for selecting a checkpoint or changing hyperparameters.

Historical result values remain unchanged. The older documents' global claim
of weight_decay=0.01 does not match the trainer defaults inspected locally.
Any historical metric discrepancy is reported to the author; checkpoint-based
recomputation requires explicit author approval.

## Execution and environment

Use conda `clrr` for Python, tests and training. Colab: A100, no `--high-mem`.
Pin Transformers 4.57.6 and PEFT 0.21.0 to the verified local versions.

```powershell
conda run -n clrr python scripts/run_ashaninka_experiments.py --dry-run
conda run -n clrr python tests/smoke_ashaninka_training.py
conda run -n clrr python -u scripts/run_ashaninka_experiments.py --hf-backup-repo ""
```

Order: mBART baseline → mBART CLRR+LSR → NLLB baseline → NLLB CLRR+LSR.
An earlier `PAUSE_AFTER_LORA` marker does not stop this explicitly requested suite.
Existing output may be resumed only if its command, data and source hashes match.
Artifacts go to `results/ashaninka_spanish/`, summary to
`results/ashaninka_spanish_scores.csv`. Original-reference scores are verified
from each new run's predictions before writing `completed.json`.

## Monitoring and collection

Check the running process, log, GPU/VRAM, disk, latest epoch/checkpoint and errors
every **10 minutes**. Preserve the per-run manifests, logs, metrics, predictions,
best checkpoint archives and summary. Download and verify artifact hashes before
stopping Colab. Record failures and investigate them before restarting; do not
silently change either method's scientific protocol to obtain better results.
