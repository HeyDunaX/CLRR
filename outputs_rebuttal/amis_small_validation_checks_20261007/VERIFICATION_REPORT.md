# Validation checks after fresh confirmation — 07/10/2026

Scope: 32 selected checkpoints; validation576 only; no training or new test generation. Pinned HF hashes and source hashes verified; all16routing-on predictions match saved validation exactly. Spectral ranks independently recomputed from saved singular spectra, not from raw pooled matrices.

## Alignment and common direction

|Family|Matched cos|Shuffled cos|Matched−shuffled|Centered matched|Centered matched−shuffled|Source common-mean energy|
|---|---:|---:|---:|---:|---:|---:|
|baseline|0.350414|0.204775|0.145639|0.264952|0.264581|0.424158|
|clrr|0.327271|0.187786|0.139485|0.253826|0.253473|0.398229|
|lsr|0.798194|0.709564|0.088629|0.334473|0.334049|0.704336|
|full|0.782713|0.700077|0.082636|0.303308|0.302915|0.695073|

## Inference interventions: condition minus all-on

|Family|Condition|ΔBLEU|95% seed Student-t CI|Wins|ΔchrF++|ΔchrF0|
|---|---|---:|---|---:|---:|---:|
|clrr|drop_10_11|-0.073839|[-0.404047, +0.256369]|4/8|-0.101613|-0.100934|
|full|drop_10|-0.080985|[-0.331639, +0.169669]|2/8|-0.073571|-0.074916|
|full|drop_10_11|-0.049780|[-0.374957, +0.275398]|4/8|-0.118199|-0.119711|
|full|drop_11|-0.083960|[-0.217798, +0.049878]|3/8|-0.104708|-0.104158|

Intervals are exploratory, unadjusted; paired training-seed descriptions on frozen weights. They do not establish gains from retraining. Matched-minus-shuffled evaluates discrimination beyond common direction; neither high raw cosine nor a low residual cosine proves collapse/conflict. Baseline/CLRR old LSR gradient probes are counterfactual, not their training objective.
