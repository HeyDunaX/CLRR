# Frozen candidate validation diagnostics

All six checkpoints strict-loaded; archive hashes verified; generation-on reproduces saved predictions after the original whitespace stripping where available. Raw decode CSV retains a few leading spaces; metrics match exactly. All scores independently replayed; zero optimizer steps.

Validation576, seed42, FP32/dropout off/TF32 off. Bootstrap is conditional on selected checkpoints; validation selection and seed randomness are not covered.

## Generation and geometry

| Model | BLEU | chrF++ | Source covariance effective rank | Target covariance effective rank | Matched cosine | Shuffled cosine | Centered matched cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD | 17.3557 | 14.4780 | 74.089 | 75.756 | 0.7968 | 0.7225 | 0.2907 |
| BASE | 16.2659 | 13.9015 | 73.097 | 75.259 | 0.3369 | 0.1975 | 0.2533 |
| U1C | 16.4181 | 13.9381 | 73.930 | 77.835 | 0.3278 | 0.1930 | 0.2458 |
| U1F | 17.3296 | 14.5519 | 74.005 | 75.511 | 0.7939 | 0.7185 | 0.2910 |
| U2L | 16.4203 | 13.9045 | 62.238 | 65.432 | 0.4758 | 0.2544 | 0.3941 |
| U2F | 17.0227 | 14.5080 | 67.499 | 71.106 | 0.4310 | 0.2356 | 0.3554 |

## Inference interventions

| Model | Group off (1-based) | Metric | Delta off−on | CI95 | Holm18 p |
|---|---|---|---:|---|---:|
| OLD | early_off | BLEU | -0.6159 | [-1.4135, +0.1806] | 0.5599 |
| OLD | middle_off | BLEU | -0.6904 | [-1.6003, +0.2477] | 0.5599 |
| OLD | late_off | BLEU | -0.7244 | [-1.3207, -0.1230] | 0.1581 |
| OLD | early_off | chrF++ | -0.4087 | [-0.9211, +0.0897] | 0.5599 |
| OLD | middle_off | chrF++ | -0.5711 | [-1.1237, -0.0119] | 0.2968 |
| OLD | late_off | chrF++ | -0.5001 | [-0.8432, -0.1608] | 0.0378 |
| U1F | early_off | BLEU | -0.1118 | [-0.9434, +0.7347] | 0.8375 |
| U1F | middle_off | BLEU | -0.7722 | [-1.7114, +0.1190] | 0.5378 |
| U1F | late_off | BLEU | -0.8394 | [-1.5572, -0.1241] | 0.1856 |
| U1F | early_off | chrF++ | -0.1580 | [-0.6973, +0.3714] | 0.8375 |
| U1F | middle_off | chrF++ | -0.6647 | [-1.3116, -0.0589] | 0.2968 |
| U1F | late_off | chrF++ | -0.5252 | [-0.9918, -0.0804] | 0.2475 |
| U2F | early_off | BLEU | -0.1691 | [-1.0841, +0.7033] | 0.8375 |
| U2F | middle_off | BLEU | +0.6065 | [-0.2124, +1.4449] | 0.5599 |
| U2F | late_off | BLEU | -0.3033 | [-1.1168, +0.4823] | 0.8375 |
| U2F | early_off | chrF++ | -0.2840 | [-0.9518, +0.3276] | 0.8375 |
| U2F | middle_off | chrF++ | -0.0240 | [-0.6419, +0.5754] | 0.8375 |
| U2F | late_off | chrF++ | -0.5115 | [-1.0918, +0.0274] | 0.4728 |

## CE–LSR and target anchor

| Model | Mean routed CE–LSR cosine | Mean raw CE–LSR cosine | Routed weighted-LSR/CE norm ratio | Routed/raw anchor cosine | Anchor relative L2 |
|---|---:|---:|---:|---:|---:|
| OLD | +0.01734 | +0.00863 | 0.003116 | 0.973202 | 0.245394 |
| U1F | +0.01473 | +0.00493 | 0.003258 | 0.975812 | 0.233563 |
| U2L | +0.09851 | +0.09851 | 0.006162 | 1.000000 | 0.000000 |
| U2F | +0.07827 | +0.06904 | 0.004095 | 0.945687 | 0.325520 |

## Layer diagnostics

| Model | Layer | Source residual/current RMS ratio | Source effective covariance rank | Routed CE–LSR cosine | Negative batches /4 |
|---|---:|---:|---:|---:|---:|
| OLD | 1 | 0.00000 | 73.910 | +0.01612 | 2/4 |
| OLD | 2 | 0.00000 | 73.709 | +0.02690 | 1/4 |
| OLD | 3 | 0.06866 | 70.920 | +0.04137 | 0/4 |
| OLD | 4 | 0.06997 | 72.262 | +0.05465 | 0/4 |
| OLD | 5 | 0.07344 | 74.014 | +0.05298 | 0/4 |
| OLD | 6 | 0.07618 | 74.215 | +0.07146 | 0/4 |
| OLD | 7 | 0.07937 | 73.529 | +0.05790 | 0/4 |
| OLD | 8 | 0.08239 | 72.758 | +0.03442 | 1/4 |
| OLD | 9 | 0.08332 | 71.235 | +0.02476 | 1/4 |
| OLD | 10 | 0.01767 | 22.702 | +0.03147 | 1/4 |
| OLD | 11 | 0.00830 | 3.658 | +0.04735 | 0/4 |
| OLD | 12 | 0.03147 | 2.564 | +0.05548 | 1/4 |
| BASE | 1 | 0.00000 | 74.551 | — | — |
| BASE | 2 | 0.00000 | 72.634 | — | — |
| BASE | 3 | 0.00000 | 68.731 | — | — |
| BASE | 4 | 0.00000 | 70.609 | — | — |
| BASE | 5 | 0.00000 | 71.896 | — | — |
| BASE | 6 | 0.00000 | 71.375 | — | — |
| BASE | 7 | 0.00000 | 70.410 | — | — |
| BASE | 8 | 0.00000 | 69.997 | — | — |
| BASE | 9 | 0.00000 | 68.416 | — | — |
| BASE | 10 | 0.00000 | 22.858 | — | — |
| BASE | 11 | 0.00000 | 2.758 | — | — |
| BASE | 12 | 0.00000 | 1.907 | — | — |
| U1C | 1 | 0.00000 | 74.379 | — | — |
| U1C | 2 | 0.00000 | 73.446 | — | — |
| U1C | 3 | 0.06786 | 70.239 | — | — |
| U1C | 4 | 0.07002 | 71.605 | — | — |
| U1C | 5 | 0.06931 | 72.968 | — | — |
| U1C | 6 | 0.07187 | 72.667 | — | — |
| U1C | 7 | 0.07503 | 72.125 | — | — |
| U1C | 8 | 0.07760 | 71.414 | — | — |
| U1C | 9 | 0.07924 | 69.545 | — | — |
| U1C | 10 | 0.01734 | 26.093 | — | — |
| U1C | 11 | 0.00786 | 3.874 | — | — |
| U1C | 12 | 0.03048 | 2.607 | — | — |
| U1F | 1 | 0.00000 | 74.029 | +0.00665 | 3/4 |
| U1F | 2 | 0.00000 | 73.485 | +0.01221 | 3/4 |
| U1F | 3 | 0.06812 | 70.954 | +0.02700 | 1/4 |
| U1F | 4 | 0.07024 | 71.888 | +0.04591 | 0/4 |
| U1F | 5 | 0.06936 | 73.813 | +0.05274 | 0/4 |
| U1F | 6 | 0.07191 | 73.962 | +0.06602 | 0/4 |
| U1F | 7 | 0.07501 | 73.271 | +0.06231 | 1/4 |
| U1F | 8 | 0.07792 | 72.569 | +0.02562 | 1/4 |
| U1F | 9 | 0.07872 | 71.021 | +0.01292 | 1/4 |
| U1F | 10 | 0.01627 | 21.261 | +0.00884 | 1/4 |
| U1F | 11 | 0.00763 | 3.503 | +0.02287 | 1/4 |
| U1F | 12 | 0.03152 | 2.455 | +0.04654 | 1/4 |
| U2L | 1 | 0.00000 | 74.832 | +0.19382 | 0/4 |
| U2L | 2 | 0.00000 | 72.852 | +0.21567 | 0/4 |
| U2L | 3 | 0.00000 | 69.163 | +0.21064 | 0/4 |
| U2L | 4 | 0.00000 | 69.841 | +0.20780 | 0/4 |
| U2L | 5 | 0.00000 | 71.445 | +0.21129 | 0/4 |
| U2L | 6 | 0.00000 | 71.278 | +0.21550 | 0/4 |
| U2L | 7 | 0.00000 | 70.771 | +0.21193 | 0/4 |
| U2L | 8 | 0.00000 | 69.347 | +0.15860 | 0/4 |
| U2L | 9 | 0.00000 | 66.080 | +0.14998 | 0/4 |
| U2L | 10 | 0.00000 | 19.786 | +0.13222 | 0/4 |
| U2L | 11 | 0.00000 | 3.057 | +0.09535 | 0/4 |
| U2L | 12 | 0.00000 | 2.206 | +0.10998 | 0/4 |
| U2F | 1 | 0.00000 | 74.230 | +0.19070 | 0/4 |
| U2F | 2 | 0.00000 | 73.652 | +0.21585 | 0/4 |
| U2F | 3 | 0.06863 | 71.084 | +0.21528 | 0/4 |
| U2F | 4 | 0.07048 | 72.270 | +0.22359 | 0/4 |
| U2F | 5 | 0.07369 | 73.709 | +0.22723 | 0/4 |
| U2F | 6 | 0.07613 | 73.648 | +0.22802 | 0/4 |
| U2F | 7 | 0.07847 | 72.923 | +0.20976 | 0/4 |
| U2F | 8 | 0.07961 | 71.689 | +0.17560 | 0/4 |
| U2F | 9 | 0.08112 | 69.073 | +0.14991 | 0/4 |
| U2F | 10 | 0.01879 | 25.474 | +0.13449 | 0/4 |
| U2F | 11 | 0.00875 | 4.318 | +0.12157 | 0/4 |
| U2F | 12 | 0.03096 | 3.023 | +0.09086 | 0/4 |

## Interpretation limits

- Dropout-off gradients are frozen-checkpoint diagnostics, not historical optimizer updates.
- Negative CE–LSR cosine indicates local directional conflict, not proven harm to generalization.
- Larger residual or lower rank alone does not establish a defect.
- Inference intervention gains do not guarantee a retraining gain.
- Raw target is a counterfactual, not an evaluated new translation method.
- No new training or test evaluation is authorized by this analysis.
