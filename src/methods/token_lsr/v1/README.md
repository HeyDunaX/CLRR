# Token-LSR v1

Method IDs: `token-lsr-v1`, `token-full-v1`. Final encoder content tokens exclude
padding and all tokenizer special IDs. Cosine costs feed balanced entropic OT
with uniform marginals, epsilon=0.10, 200 iterations, marginal error <1e-3.
Plan and target are detached. Loss is CE + 0.10 times the mean per-sentence
transport-weighted cosine cost, not Sinkhorn divergence. No standalone
anti-collapse or syntax-preservation guarantee; no extra trainable parameters.

Full uses original encoder CLRR, distance2/strength0.10, post-rewiring detached
donors and independent source/target caches. Original methods remain separate.
The shared runner supplies tokenizer.all_special_ids on load and restore.

RTX pilot: configs/experiments/amis_token_rtx.json; four same-hardware runs,
FP32, microbatch4/accumulation32, effective128, ES10, validation-only.
