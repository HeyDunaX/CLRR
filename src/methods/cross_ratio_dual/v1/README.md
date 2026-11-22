# Cross-Ratio Dual Consistency (`cross_ratio_dual/v1`)

Implements **Marginal-Free Cross-Ratio Dual Consistency (`CR-Dual`)** and its three
pre-registered discriminating controls on many-to-many pretrained seq2seq models
(`mBART-50`, `NLLB-200`) with zero external language models, zero extra parameters,
and zero inference overhead.

## Method IDs
- `bidirectional-ce-v1`: `CE_fwd + alpha_rev * CE_rev` (Ding et al., EMNLP 2021 BiT control).
- `cross-ratio-dual-v1`: `BIT_CE + lambda(s) * Huber((M_fwd(i,j) - M_rev(i,j)) / L_ij)` with length-matched disjoint involution `pi`, permuted source padding mask, Paired-RNG shared dropout, and linear `lambda(s)` warmup.
- `cross-ratio-shuffled-v1`: Control 1 falsifier with identical passes/compute/Huber matching `M_fwd(i,j)/L_ij` against cyclically rolled `M_rev(k,l)/L_kl`.
- `unidirectional-contrast-v1`: Control 3 falsifier using unidirectional SCD-style off-diagonal margin on `X -> Y` only.

Design & audit record: `docs/research/ROUND3_METHOD_DESIGN_AUDIT.md`.
