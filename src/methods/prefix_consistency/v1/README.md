# Prefix Likelihood Consistency (PLC), exploratory v1

Two independent dropout views share gold prefixes. Let `a_t` be their gold
log-probability difference. Valid tokens include every label other than -100.
For each sentence of length T:

- P = mean_k ((sum_{t<=k} a_t)^2 / k).
- D = sum_t a_t^2 * (sum_{k>=t} 1/k) / T.
- E = (sum_t a_t)^2 / T.

The objective preserves W's CE + 1.25 symmetric full-vocabulary KL and adds
0.10 times the sentence mean of P (or control D/E). Wcal uses no path term and
a separately frozen training-only gradient-norm calibrated KL coefficient.
Both views receive gradients. No parameters or forward passes are added.
Inference uses the native backbone. FP32 score arithmetic and chunked
recomputation avoid retaining extra full-vocabulary log-softmax tensors.

See `docs/research/METHOD_REDESIGN_CANDIDATES.md` for scientific admission,
novelty scope and validation-only pilot criteria. A passing CPU/GPU gate is
not evidence that the method improves translation.
