# Corrected accumulation — significance characterization

## Material Passport

Post-hoc analysis of frozen selected checkpoints, 575 test sentences, 10000 paired bootstrap draws, RNG42; all native SacreBLEU p-values independently matched. No training/GPU use or model selection.

| Contrast | Seed | Metric | Delta | Delta 95% CI | p native | p Holm12 | Holm12 < .05 |
|---|---:|---|---:|---|---:|---:|---|
| Full_minus_lsr | 42 | BLEU | +1.2259 | [+0.0830, +2.4318] | 0.021598 | 0.237576 | False |
| Full_minus_lsr | 42 | chrF++ | +0.5802 | [-0.1351, +1.3479] | 0.059594 | 0.595940 | False |
| Full_minus_baseline | 42 | BLEU | +0.5932 | [-0.5212, +1.7258] | 0.118488 | 0.863014 | False |
| Full_minus_baseline | 42 | chrF++ | +0.3009 | [-0.3818, +1.0148] | 0.147085 | 0.863014 | False |
| Full_minus_lsr | 43 | BLEU | +0.3587 | [-0.6129, +1.3013] | 0.161084 | 0.863014 | False |
| Full_minus_lsr | 43 | chrF++ | +0.1958 | [-0.3735, +0.7806] | 0.175182 | 0.863014 | False |
| Full_minus_baseline | 43 | BLEU | +2.0402 | [+0.7645, +3.3924] | 0.001700 | 0.020398 | True |
| Full_minus_baseline | 43 | chrF++ | +5.6976 | [+0.4675, +14.0281] | 0.106689 | 0.863014 | False |
| Full_minus_lsr | 44 | BLEU | +0.6273 | [-0.3914, +1.6352] | 0.095890 | 0.863014 | False |
| Full_minus_lsr | 44 | chrF++ | +0.4074 | [-0.2722, +1.1212] | 0.103590 | 0.863014 | False |
| Full_minus_baseline | 44 | BLEU | -0.4084 | [-1.6109, +0.7734] | 0.177682 | 0.863014 | False |
| Full_minus_baseline | 44 | chrF++ | -0.2020 | [-0.9263, +0.5263] | 0.204780 | 0.863014 | False |

## Mean delta: sentence uncertainty with all three seeds fixed

- Full_minus_lsr BLEU: mean +0.7373, conditional CI [+0.0403, +1.4557]. Same sentence draws across seeds; excludes training randomness.
- Full_minus_lsr chrF++: mean +0.3945, conditional CI [-0.0392, +0.8657]. Same sentence draws across seeds; excludes training randomness.
- Full_minus_baseline BLEU: mean +0.7417, conditional CI [-0.0265, +1.5182]. Same sentence draws across seeds; excludes training randomness.
- Full_minus_baseline chrF++: mean +1.9321, conditional CI [+0.0084, +4.7936]. Same sentence draws across seeds; excludes training randomness.

## Across training seeds

- Full_minus_lsr BLEU: exact two-sided sign-flip p=0.250, n=3, minimum attainable p=.25. Assumption: symmetric paired effects under null.
- Full_minus_lsr chrF++: exact two-sided sign-flip p=0.250, n=3, minimum attainable p=.25. Assumption: symmetric paired effects under null.
- Full_minus_baseline BLEU: exact two-sided sign-flip p=0.500, n=3, minimum attainable p=.25. Assumption: symmetric paired effects under null.
- Full_minus_baseline chrF++: exact two-sided sign-flip p=0.500, n=3, minimum attainable p=.25. Assumption: symmetric paired effects under null.

## Limits

- Sentence resampling conditional on frozen checkpoints; does not cover training randomness.
- Only three training seeds; exact two-sided seed sign-flip cannot attain p<0.05.
- Sentences assumed exchangeable; correlated documents may invalidate sentence-level precision.
- Exploratory analysis after inspecting test scores; no pre-registration claim.
- P-values do not establish novelty, mechanism, cross-language generalization or ACL acceptance.

Method: https://github.com/mjpost/sacrebleu and https://aclanthology.org/W04-3250/ .
