# Processed datasets

Use the existing conda environment `clrr` for Python, preparation, tests, and training.
Each dataset lives in `data_processed/<dataset_name>/`, with `train.csv`,
`validation.csv`, and `test.csv` containing exactly `source,target`.

| Dataset | Direction | Train | Validation | Test | Use |
| --- | --- | ---: | ---: | ---: | --- |
| `amis_mandarin` | Amis → Mandarin | 4,600 | 576 | 575 | Current manuscript and matched component study |
| `rhade_vietnamese` | Ê Đê (Rhade) → Vietnamese | 15,092 | 1,000 | 1,000 | Second language dataset; original published splits, text and row order |

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

Ê Đê replaces Turkish as the second language dataset (author decision, 15/10/2026).
The local Turkish processed directory was removed; its existing checkpoint receipts
and historical results are retained for traceability and are not Ê Đê controls.

Ê Đê conversion maps only `translation.ede → source` and `translation.vi → target`.
No trimming, Unicode normalization, filtering, deduplication, resampling, split
changes or row reordering is performed. Parquet/CSV hashes and an audit-only
overlap report live in [the manifest](rhade_vietnamese/manifest.json).
Preparation and verification commands are in [the dataset README](rhade_vietnamese/README.md).
See [the proposed experiment plan](../docs/research/EDE_SECOND_LANGUAGE_PLAN.md)
for the approved language-token policy and four-run pilot, now launched through
the shared RTX gate/training queue (15/10/2026).
