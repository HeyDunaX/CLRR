# U2 v2: value-input scale control

Experimental method family integrated with the configuration-driven runner.
Keys are unchanged. For each source example, over non-padding positions:

    c = stopgrad(sqrt(sum(H_top**2) / max(sum(H_donor**2), 1e-12)))
    V_donor = v_proj(c * stopgrad(H_donor))
    delta = (A_donor - A_top) @ (V_donor - V_top)

The scale is computed in FP32. Shared projections remain trainable; scale has no
gradient to the recipient. Main attention, zero-initialized gates, donor layer6
and decoder layer12 are inherited from v1. The epsilon is a zero-norm guard,
not a tuned hyperparameter. No centering, LayerNorm or additional learned scale.

Checkpoints use a distinct method ID; the v1 loader rejects v2 checkpoints.
Requires query-independent additive source masks with valid entries equal to0.
Per-example source scaling is compatible with native beam expansion and caches.

Motivation: frozen training64 audit. This does not establish novelty or quality.
Controls must share this scale convention when comparing interaction operators.

`depth-donor-addition-v2` is the simple depth-access control:
`C_out = native_top_context + gate_head * (A_donor @ V_donor)`.
It uses the same detached donor, value-input scaling, shared projections,
zero-initialized head gates and correction dropout convention as mixed U2.
This is an adapted control, not a reproduction of a published architecture.
