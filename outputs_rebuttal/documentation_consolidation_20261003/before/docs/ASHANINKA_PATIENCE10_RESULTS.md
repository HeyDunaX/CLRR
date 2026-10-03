# Asháninka → Spanish: mBART CLRR+LSR patience-10 results

Completed 2026-10-02. Trained from pretrained mBART for 20 epochs, seed 42,
with the protocol in `ASHANINKA_PATIENCE10_RUNBOOK.md`.
The highest validation chrF++ checkpoint is epoch 14 (`checkpoint-434`).
The author also requested evaluation of fixed epoch 20 (`checkpoint-620`).

All test values below use 1,003 original CSV references, beam 4,
SacreBLEU tokenizer 13a and chrF++ word_order=2.

| Run / checkpoint selection | Patience | Test BLEU | Test chrF++ |
| --- | ---: | ---: | ---: |
| Original mBART baseline, selected by validation chrF++ | 4 | 4.008525491403604 | 20.853627641616395 |
| Original mBART CLRR+LSR, selected by validation chrF++ | 4 | 3.9670904856431672 | 20.699039452099623 |
| New mBART CLRR+LSR, selected by validation chrF++, epoch 14 | 10 | 3.796952612114073 | 20.975283238139824 |
| New mBART CLRR+LSR, fixed epoch 20 | 10 | 4.094883276845612 | 20.958774160834572 |

The new run's official validation-selected checkpoint remains epoch 14.
Epoch 20 has higher test BLEU but slightly lower test chrF++ than epoch 14.
Its differences from the original baseline are +0.086357785442008 BLEU and
+0.105146519218177 chrF++.

Interpretation: this is single-seed, additional checkpoint analysis. The original
baseline used patience 4, so comparison against the new patience-10 CLRR run
does not establish an advantage under matched stopping settings. Epoch 20 was
evaluated after the author's request; do not replace the validation-selected
main result based on test scores. This is a new training trajectory, so changes
from the earlier CLRR run cannot be attributed solely to patience.
Historical values have been retained.

Artifacts:

- Validation-selected metrics/predictions/manifests: `results/ashaninka_patience10/runs/mbart-ashaninka-es-clrr-enc/`.
- Epoch-20 metrics/predictions: `results/ashaninka_patience10/epoch20/mbart-ashaninka-es-clrr-enc-epoch20/`.
- Comparison: `results/ashaninka_patience10/best_vs_epoch20.csv`.
- HF model archives and reports: `FiveC/amis-rewire-checkpoints`, prefix `ashaninka-es/2026-10-02/mbart-clrr-lsr-patience10/`; fixed epoch 20 under `epoch20/`.
- HF receipts: `outputs_rebuttal/mbart_patience10_hf_uploaded.json` and `outputs_rebuttal/mbart_epoch20_hf_uploaded.json`.
- Colab shutdown receipt: `outputs_rebuttal/mbart_patience10_stopped.json`.

Both model archives uploaded successfully; each uploaded file's size was checked
against its remote source. Reports were saved locally before stopping Colab.
