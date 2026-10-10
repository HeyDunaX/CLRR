# OPUS-100 Turkish → English — 20k training subset

Official validation/test: 2000 rows each, exact strings/order, duplicates retained. No evaluation filtering or deduplication. Existing 20k training sample unchanged and verified against pinned official train.

CSV schema: `source,target`. Manifest records revision, hashes, upstream row IDs and overlap audit. This is a low-resource subset experiment, not the full 1M benchmark.

Run `conda run -n clrr python scripts/prepare_turkish_followup.py` to verify the sample and restore official evaluation rows. Preparation does not launch training.
