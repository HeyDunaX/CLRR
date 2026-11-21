# Ê Đê (Rhade) → Vietnamese

Source: [NIRVLab/rhade-vietnamese-mt](https://huggingface.co/datasets/NIRVLab/rhade-vietnamese-mt),
pinned revision `dfbbffa09652e34054ef676589067f2ef965633b`.
Upstream files, rather than the outdated dataset card counts, define this export.

| Split | Rows | Schema |
|---|---:|---|
| train | 15,092 | source,target |
| validation | 1,000 | source,target |
| test | 1,000 | source,target |

Only the container/field names change: `translation.ede → source`,
`translation.vi → target`. All decoded strings (including whitespace, punctuation,
case, line breaks and Unicode composition), rows, duplicates, order and split
membership are preserved. CSV quoting is standard serialization, not text repair.
There is no filtering, resampling, normalization or new split.

The exporter checks every CSV row against its pinned Parquet row after a CSV
read-back. [manifest.json](manifest.json) records upstream Parquet SHA-256, output
CSV SHA-256 and an exact overlap audit. A source can occur with different target
translations: 104 validation rows and 94 test rows have a source present in train;
no exact source–target pair is shared between splits. These rows are retained.

```bash
conda run -n clrr python scripts/prepare_hf_parallel.py --repo NIRVLab/rhade-vietnamese-mt --revision dfbbffa09652e34054ef676589067f2ef965633b --source ede --target vi --dataset rhade_vietnamese
conda run -n clrr python scripts/prepare_hf_parallel.py --repo NIRVLab/rhade-vietnamese-mt --revision dfbbffa09652e34054ef676589067f2ef965633b --source ede --target vi --dataset rhade_vietnamese --verify-only
```

Run from the repository root. Verification re-downloads the small Parquet files
into memory and checks all text/order and CSV bytes without changing outputs.
Existing differing CSVs are never overwritten implicitly.

The dataset card declares CC BY-NC 4.0 and requests citation of Tran (2025) and
the contributing resources. Preserve that attribution; this conversion does not
relicense the corpus. No data is uploaded by the preparation script.

Ê Đê has no native mBART language token. The approved experiment uses Indonesian
source proxy `id_ID` and target `vi_VN`, chosen from a linguistic prior on
15/10/2026 before model validation/test scores. The earlier `tl_XX` proposal was
retired after preflight only. All four methods use the same approved proxy.
Use Vietnamese BLEU `13a`, native chrF++ and chrF0; do not use Mandarin `zh` BLEU.

Training plan: [EDE_SECOND_LANGUAGE_PLAN.md](../../docs/research/EDE_SECOND_LANGUAGE_PLAN.md).
