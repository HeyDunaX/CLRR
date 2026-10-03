# OPUS-100 Turkish → English, 20k v2

Separate dataset for the matched CLRR pilot; the previous dataset is preserved.

| Split | Rows | Unique pairs |
|---|---:|---:|
| Train | 20,000 | 20,000 |
| Validation | 1,946 | 1,946 |
| Test | 2,000 | 1,982 |

UTF-8 CSV schema: `source,target`. Official OPUS-100 test order is retained,
including duplicate pairs. Train/validation/test have zero exact stripped-source
overlap. Validation was deduplicated and test-source overlap removed. Train is
a new deterministic sample of eligible unique pairs, not the old 20k sample.

Source: `Helsinki-NLP/opus-100`, revision
`805090dc28bf78897da9641cdf08b61287580df9`. The manifest pins parquet hashes,
CSV hashes, sampling and upstream row IDs. This is simulated low-resource
fine-tuning; no claim of pretraining disjointness.

Reproduce in conda `clrr`:

```powershell
conda run -n clrr python scripts/prepare_turkish_followup.py
```

Use mBART `tr_TR → en_XX`, BLEU13a and raw chrF++6/2. See
`docs/FOLLOWUP_RUNBOOK_20261003.md` for matched experiment settings.
