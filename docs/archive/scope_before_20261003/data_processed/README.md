# Processed parallel datasets

Each dataset has its own directory. Training loaders expect UTF-8 CSV files with
exactly two columns in this order: `source,target`.

| Dataset directory | Translation direction | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| `amis_mandarin` | Amis → Mandarin | 4,600 | 576 | 575 |
| `ashaninka_spanish` | Asháninka → Spanish | 3,883 | 881 | 1,003 |

```text
data_processed/
  amis_mandarin/
    train.csv
    validation.csv
    test.csv
    manifest.json
  ashaninka_spanish/
    train.csv
    validation.csv
    test.csv
    manifest.json
    raw/
```

Pass `--data-dir data_processed/<dataset_name>` to select a dataset. Existing
Amis training and analysis commands default to `data_processed/amis_mandarin`.
Use a separate output/run name for every dataset and method.

The existing Amis CSVs were moved from `data/processed` without changing any
bytes, row order, or split membership. Their original SHA-256 hashes are recorded
in `amis_mandarin/manifest.json`. Historical experiment artifacts retain their
original paths and hashes.

## Asháninka–Spanish

Official source: [AmericasNLP 2021](https://github.com/AmericasNLP/americasnlp2021).
Train/dev are in `data/ashaninka-spanish`; test is in `test_data`.
The preparation script pins commit `d3f519c6b38299d8149311e65a369047350e6849`.

Reproduce with the **conda clrr** environment from the repository root:

```powershell
conda run -n clrr python scripts/prepare_ashaninka_spanish.py
```

Files `*.cni` are the source; corresponding `*.es` files are the target.
Preprocessing strips outer whitespace, then removes empty pairs jointly while
preserving official split membership and row order. Original dev lines 72 and
123 have blank Spanish references: 883 raw pairs become 881 validation pairs.
No synthetic data is included. Raw downloads remain under `raw/`; the manifest
records URLs, hashes, dropped line numbers, duplicate counts, and cross-split
overlap. Exact source and exact pair overlap between splits are both zero.

Cite the shared task and the original corpus sources listed by its maintainers:

- [AmericasNLP 2021 Findings](https://aclanthology.org/2021.americasnlp-1.23/).
- [Ortega et al. (2020), Overcoming Resistance: The Normalization of an Amazonian Tribal Language](https://aclanthology.org/2020.loresmt-1.1/).
- Cushimariano Romano & Sebastián Q. (2008), *Diccionario Asháninka-Castellano*.
- Mihas (2011), *Añaāni katonkosatzi parenini, El idioma del alto Perené*.

## Training configuration status

The Asháninka runner explicitly selects Spanish (`es_XX` for mBART,
`spa_Latn` for NLLB) as target and as the source-language proxy; these models
have no Asháninka language token. Both methods within each backbone use the
same proxy, seed, optimizer, learning rate, batch size, and evaluation protocol.
Official test scores use original CSV references: BLEU `13a` and chrF++
`word_order=2`. Historical Amis results are preserved.

Validate commands and dataset hashes without training:

```powershell
conda run -n clrr python scripts/run_ashaninka_experiments.py --dry-run
```

The runner executes mBART baseline, mBART CLRR+LSR, NLLB baseline, then NLLB
CLRR+LSR. It verifies test scores against predictions and records completion
only after the best checkpoint archive exists. See
`docs/ASHANINKA_RUNBOOK.md` for the exact frozen protocol and monitoring.
