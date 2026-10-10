# Corrected accumulation pilot: old/new comparison

| Family | Old test BLEU | New test BLEU | Delta | Old chrF++ | New chrF++ | Delta | Changed hypotheses |
|---|---:|---:|---:|---:|---:|---:|---:|
| lsr | 19.786496 | 19.095440 | -0.691056 | 18.679745 | 18.434505 | -0.245239 | 442/575 |
| full | 19.956574 | 20.321295 | +0.364721 | 18.887811 | 19.014707 | +0.126896 | 400/575 |

## Full minus LSR

- validation: BLEU old +1.074889, corrected +0.934848; chrF++ old +0.564262, corrected +0.287807.
- test: BLEU old +0.170078, corrected +1.225855; chrF++ old +0.208066, corrected +0.580202.

## Interpretation boundaries

- These are native metrics of verified selected checkpoints; corrected scores are separate from old artifacts.
- No multi-seed significance or equivalence claim from this pilot. No variants chosen from test.
- Gradient-norm statistics cover logged steps only. Historical full trajectories/runtime receipts are incomplete.
