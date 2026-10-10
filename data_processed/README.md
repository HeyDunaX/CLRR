# Processed datasets

Use the existing conda environment `clrr` for Python, preparation, tests, and training.
Each dataset lives in `data_processed/<dataset_name>/`, with `train.csv`,
`validation.csv`, and `test.csv` containing exactly `source,target`.

| Dataset | Direction | Train | Validation | Test | Use |
| --- | --- | ---: | ---: | ---: | --- |
| `amis_mandarin` | Amis → Mandarin | 4,600 | 576 | 575 | Current manuscript and matched component study |
| `opus100_turkish_english_20k` | Turkish → English | 20,000 | 2,000 | 2,000 | Official OPUS-100 evaluation splits; retained 20k training sample; training not launched by this change |

Select explicitly with `--data-dir data_processed/<dataset_name>` and a separate
output/run name. Trainers default to `data_processed/amis_mandarin`.
Run in this repository, for example:

```powershell
conda run -n clrr python tests/smoke_translation_training.py
```

The Amis files retain their original bytes, split membership, and row order;
hashes and source provenance are recorded in `amis_mandarin/manifest.json`.
The paper uses mBART `tl_XX→zh_CN` and historical NLLB `zho_Hant→zho_Hant`.
See [metrics](../docs/experiments/METRICS.md) and [current experiments](../docs/experiments/EXPERIMENTS_UPDATED.md).

Turkish uses the official OPUS-100 validation/test: all 2,000 rows per split,
exact strings and upstream order, including duplicates. The existing 20k train
sample retains its bytes and order. Manifest pins upstream revision, hashes and
row IDs. Overlap is audited, not filtered: 17 exact stripped-source types overlap
between validation/test; no train/evaluation source overlap was found. This is
a simulated low-resource training subset, not the full OPUS-100 1M-pair training
benchmark. Use mBART `tr_TR→en_XX`, BLEU `13a` and native chrF++.
The filtered preparation v2 was retired at the author's request on 06/10/2026.
Old v2 outputs are historical and must not be silently reused for this protocol.
