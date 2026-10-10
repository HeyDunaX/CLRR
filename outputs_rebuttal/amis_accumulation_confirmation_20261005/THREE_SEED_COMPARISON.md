# Corrected accumulation: three-seed descriptive comparison

All numbers are separate from original experiments. Baseline controls are audited and reused; means use seeds42/43/44 and sample SD (ddof1).

| Regime | Family | Seed | Val BLEU | Val chrF++ | Test BLEU | Test chrF++ |
|---|---|---:|---:|---:|---:|---:|
| baseline_audited | baseline | 42 | 16.2659 | 13.9015 | 19.7281 | 18.7138 |
| baseline_audited | baseline | 43 | 17.1513 | 14.2007 | 18.1378 | 13.1592 |
| baseline_audited | baseline | 44 | 17.9653 | 14.7987 | 20.3000 | 18.9017 |
| corrected | lsr | 42 | 16.4208 | 14.1902 | 19.0954 | 18.4345 |
| historical | lsr | 42 | 16.5688 | 14.1674 | 19.7865 | 18.6797 |
| corrected | lsr | 43 | 16.7381 | 14.2744 | 19.8193 | 18.6610 |
| historical | lsr | 43 | 16.4653 | 13.9452 | 19.4879 | 18.6488 |
| corrected | lsr | 44 | 16.7391 | 14.2014 | 19.2644 | 18.2922 |
| historical | lsr | 44 | 16.5950 | 14.0698 | 19.3284 | 18.3633 |
| corrected | full | 42 | 17.3557 | 14.4780 | 20.3213 | 19.0147 |
| historical | full | 42 | 17.6437 | 14.7317 | 19.9566 | 18.8878 |
| corrected | full | 43 | 17.2648 | 14.3271 | 20.1780 | 18.8568 |
| historical | full | 43 | 17.0645 | 14.4126 | 19.6123 | 18.4753 |
| corrected | full | 44 | 17.1153 | 14.2164 | 19.8916 | 18.6996 |
| historical | full | 44 | 16.9324 | 14.3483 | 19.7371 | 18.6065 |

## Mean ± sample SD

| Split | Metric | Regime | Family | Mean ± SD |
|---|---|---|---|---:|
| validation | bleu | baseline_audited | baseline | 17.1275 ± 0.8499 |
| validation | bleu | historical | lsr | 16.5430 ± 0.0686 |
| validation | bleu | historical | full | 17.2135 ± 0.3783 |
| validation | bleu | corrected | lsr | 16.6327 ± 0.1835 |
| validation | bleu | corrected | full | 17.2453 ± 0.1214 |
| validation | chrfpp | baseline_audited | baseline | 14.3003 ± 0.4568 |
| validation | chrfpp | historical | lsr | 14.0608 ± 0.1114 |
| validation | chrfpp | historical | full | 14.4975 ± 0.2053 |
| validation | chrfpp | corrected | lsr | 14.2220 ± 0.0457 |
| validation | chrfpp | corrected | full | 14.3405 ± 0.1313 |
| test | bleu | baseline_audited | baseline | 19.3886 ± 1.1203 |
| test | bleu | historical | lsr | 19.5343 ± 0.2325 |
| test | bleu | historical | full | 19.7687 ± 0.1743 |
| test | bleu | corrected | lsr | 19.3930 ± 0.3787 |
| test | bleu | corrected | full | 20.1303 ± 0.2188 |
| test | chrfpp | baseline_audited | baseline | 16.9249 ± 3.2626 |
| test | chrfpp | historical | lsr | 18.5639 ± 0.1745 |
| test | chrfpp | historical | full | 18.6565 ± 0.2107 |
| test | chrfpp | corrected | lsr | 18.4626 ± 0.1860 |
| test | chrfpp | corrected | full | 18.8570 ± 0.1575 |

## Paired seed differences

| Split | Metric | Contrast | Seed42 | Seed43 | Seed44 | Mean Δ | Positive seeds |
|---|---|---|---:|---:|---:|---:|---:|
| validation | bleu | corrected_full_minus_lsr | +0.9348 | +0.5267 | +0.3762 | +0.6126 | 3/3 |
| validation | bleu | corrected_full_minus_baseline | +1.0898 | +0.1136 | -0.8499 | +0.1178 | 2/3 |
| validation | bleu | corrected_lsr_minus_baseline | +0.1550 | -0.4132 | -1.2262 | -0.4948 | 1/3 |
| validation | bleu | lsr_corrected_minus_historical | -0.1479 | +0.2728 | +0.1441 | +0.0897 | 2/3 |
| validation | bleu | full_corrected_minus_historical | -0.2880 | +0.2003 | +0.1829 | +0.0317 | 2/3 |
| validation | chrfpp | corrected_full_minus_lsr | +0.2878 | +0.0527 | +0.0149 | +0.1185 | 3/3 |
| validation | chrfpp | corrected_full_minus_baseline | +0.5764 | +0.1263 | -0.5823 | +0.0401 | 2/3 |
| validation | chrfpp | corrected_lsr_minus_baseline | +0.2886 | +0.0736 | -0.5973 | -0.0783 | 2/3 |
| validation | chrfpp | lsr_corrected_minus_historical | +0.0228 | +0.3292 | +0.1317 | +0.1612 | 3/3 |
| validation | chrfpp | full_corrected_minus_historical | -0.2537 | -0.0855 | -0.1319 | -0.1571 | 0/3 |
| test | bleu | corrected_full_minus_lsr | +1.2259 | +0.3587 | +0.6273 | +0.7373 | 3/3 |
| test | bleu | corrected_full_minus_baseline | +0.5932 | +2.0402 | -0.4084 | +0.7417 | 2/3 |
| test | bleu | corrected_lsr_minus_baseline | -0.6326 | +1.6814 | -1.0356 | +0.0044 | 1/3 |
| test | bleu | lsr_corrected_minus_historical | -0.6911 | +0.3313 | -0.0640 | -0.1413 | 1/3 |
| test | bleu | full_corrected_minus_historical | +0.3647 | +0.5657 | +0.1545 | +0.3616 | 3/3 |
| test | chrfpp | corrected_full_minus_lsr | +0.5802 | +0.1958 | +0.4074 | +0.3945 | 3/3 |
| test | chrfpp | corrected_full_minus_baseline | +0.3009 | +5.6976 | -0.2020 | +1.9321 | 2/3 |
| test | chrfpp | corrected_lsr_minus_baseline | -0.2793 | +5.5018 | -0.6095 | +1.5377 | 1/3 |
| test | chrfpp | lsr_corrected_minus_historical | -0.2452 | +0.0122 | -0.0711 | -0.1014 | 1/3 |
| test | chrfpp | full_corrected_minus_historical | +0.1269 | +0.3814 | +0.0931 | +0.2005 | 3/3 |

These are descriptive comparisons, without p-values, equivalence, or paper-readiness claims. Test results do not choose hyperparameters.
