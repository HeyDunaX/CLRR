# Asháninka → Spanish: mBART CLRR+LSR, patience 10

Authorized on 2026-10-02 after completing the original patience-4 runs.
This is an additional early-stopping sensitivity experiment. It does not replace
the original matched baseline/CLRR comparison.

Use the frozen configuration in `docs/ASHANINKA_RUNBOOK.md`, with **only
early-stopping patience changed from 4 to 10**. Maximum epochs remains 20.
Start from `facebook/mbart-large-50-many-to-many-mmt`, seed/data seed 42,
LR 5e-5, weight decay 0, effective batch 128, source/target limit 256,
validation/test beams 1/4, BF16, gradient checkpointing enabled.
CLRR encoder d=2, alpha=0.1; LSR lambda=0.1.
Select the checkpoint with highest validation chrF++; test references are the
original CSV, SacreBLEU 13a and chrF++ word_order=2.

```powershell
conda run -n clrr python scripts/run_ashaninka_experiments.py --runs mbart_clrr --early-stopping-patience 10 --num-train-epochs 20 --output-dir results/ashaninka_patience10/runs --hf-backup-repo= --dry-run
```

Actual training uses the same command without `--dry-run` in remote conda `clrr`.
Colab: A100 Standard, no `--high-mem`.

Output: `results/ashaninka_patience10/runs/mbart-ashaninka-es-clrr-enc/`.
Summary: `results/ashaninka_patience10/ashaninka_spanish_scores.csv`.
Checkpoint: `results/ashaninka_patience10/runs/backups/mbart-ashaninka-es-clrr-enc/mbart-ashaninka-es-clrr-enc-best.zip`.
HF backup: `FiveC/amis-rewire-checkpoints`, prefix
`ashaninka-es/2026-10-02/mbart-clrr-lsr-patience10/`.

Report progress in chat every 10 minutes while the chat is active. The background
monitor also saves status/log files every 10 minutes. On successful completion,
verify prediction rows, official scores, checkpoint archive and HF file sizes,
download reports and the upload receipt, then stop Colab. Retain the runtime on
failure for diagnosis. Do not change any historical result values.

The baseline with patience 4 cannot by itself establish a matched comparison
against this tuned CLRR run. Any later patience-10 baseline needs separate authorization.
