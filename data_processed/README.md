# Processed datasets

Use the existing conda environment `clrr` for Python, preparation, tests, and training.
Each dataset lives in `data_processed/<dataset_name>/`, with `train.csv`,
`validation.csv`, and `test.csv` containing exactly `source,target`.

| Dataset | Direction | Train | Validation | Test | Use |
| --- | --- | ---: | ---: | ---: | --- |
| `amis_mandarin` | Amis → Mandarin | 4,600 | 576 | 575 | Current manuscript and matched component study |
| `opus100_turkish_english_20k_v2` | Turkish → English | 20,000 | 1,946 | 2,000 | Prepared; training stopped pending author decision |

Select explicitly with `--data-dir data_processed/<dataset_name>` and a separate
output/run name. Trainers default to `data_processed/amis_mandarin`.
Run in this repository, for example:

```powershell
conda run -n clrr python tests/smoke_translation_training.py
```

The Amis files retain their original bytes, split membership, and row order;
hashes and source provenance are recorded in `amis_mandarin/manifest.json`.
The paper uses mBART `tl_XX→zh_CN` and historical NLLB `zho_Hant→zho_Hant`.
See [metrics](../docs/METRICS.md) and [current experiments](../docs/EXPERIMENTS_UPDATED.md).

Turkish v2 has a fixed 20k training sample and source-disjoint splits. Its
manifest records the OPUS-100 revision, processing, and hashes. Use mBART
`tr_TR→en_XX`, BLEU `13a`, and the shared chrF++ protocol when explicitly resumed.
The earlier `opus100_turkish_english_20k` directory is a superseded preparation
record; select v2 explicitly. No Turkish trained-method result is in the paper.
