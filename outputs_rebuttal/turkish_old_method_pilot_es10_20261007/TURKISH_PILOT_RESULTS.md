# Turkish→English — mBART-50, pilot seed 42

Official OPUS100 validation/test splits unchanged; 20,000 training pairs, 2,000 validation and 2,000 test pairs. All four selected checkpoints: epoch 2; training stopped at epoch 12 with patience 10. Test after four selections were locked: beam 4, FP32 generation, TF32 off, BLEU13a/native chrF++/chrF0. All test predictions and scores independently verified.

| Method | Seed | Test BLEU | Test chrF++ | Test chrF0 | ΔBLEU vs Baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 42 | 22.634203 | 41.549538 | 42.595475 | +0.000000 |
| middle | 42 | 22.519996 | 41.389460 | 42.427257 | -0.114208 |
| lsr | 42 | 22.200584 | 41.271311 | 42.305028 | -0.433619 |
| full | 42 | 22.570620 | 41.549304 | 42.616011 | -0.063583 |

## Full minus each comparator

| Comparator | ΔBLEU | ΔchrF++ | ΔchrF0 |
|---|---:|---:|---:|
| baseline | -0.063583 | -0.000233 | +0.020536 |
| middle | +0.050624 | +0.159845 | +0.188754 |
| lsr | +0.370036 | +0.277993 | +0.310983 |

## Interpretation

Full improves on LSR in all three metrics for this frozen seed. Full has almost the same chrF++ as Baseline, slightly lower BLEU, and slightly higher chrF0. The run does not show a clear Full advantage over Baseline. This is an exploratory one-seed result, not evidence of seed robustness or statistical significance. No method or checkpoint was tuned using these test results.

Checkpoint repo: https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/opus100_turkish_english_20k .
Storage rules: docs/operations/HUGGINGFACE_CHECKPOINT_STORAGE.md .
Results: public_test_results.json; per-family final collection_receipt.json.
