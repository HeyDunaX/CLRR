# Transition LSR v1 — design and pre-training review

## Decision (2026-11-09)

**DESIGN_REVIEW_PASS_FOR_BOUNDED_PILOT**, not a certified novelty or ACL claim.
The author authorized implementation, gates and an RTX queue for the proposed
two-arm design. The previous relational-v2 queue remains stopped; its HOLD is
not lifted. This is a different candidate reviewed before smoke/training.

The research question is whether bilingual supervision on the change between
the endpoints of an existing CLRR edge transfers better than final-state LSR.
It is a hypothesis, not an established explanation of earlier failures.

## Locked formula

For encoder layer l=3..12 (one-based), d=2, u_l is the block output BEFORE
the CLRR hook. The donor v_(l-d) is the POST-rewiring output of layer l-d.
v_l = u_l + .1 * sg(v_(l-d)) for the Full arm; v_l = u_l for LSR-only.
The first two layers have v_l=u_l. Pooling p is the original attention-masked
mean, including non-pad special tokens, with no tokenizer/data changes.

r_l(x) = p(u_l(x)) - sg(p(v_(l-d)(x)))
L_aux = mean over examples and l of
        1 - dot(normalize(r_l(x),eps=1e-6), normalize(sg(r_l(y)),eps=1e-6))
L = L_CE + .1 * L_aux. No original final-state LSR is added.

Both auxiliary passes use the shared evolving encoder with dropout disabled;
target is entirely no-grad. They execute before the unchanged stochastic CE
pass. This follows the deterministic auxiliary-view infrastructure used by
recent RTX candidates, but differs from original train-mode/dropout LSR.
Consequently gains cannot by themselves isolate the objective from that
auxiliary-view policy. Additional matched state-alignment controls would be
needed before making that causal claim. Donor detach is local to subtraction:
ordinary sequential encoder gradients remain. No extra parameters or negatives.

## Primary-paper method comparison BEFORE implementation

| Prior | Method section / objective | Overlap | Concrete distinction / limit |
|---|---|---|---|
| Middle Alignment, ACL 2025 | Representation-alignment method: pooled middle states, contrastive alignment and alternating task/alignment training | Parallel views and internal representations | Here pooled depth differences at existing encoder edges, no contrastive negatives or alternating optimizer. Generic middle alignment is not a new contribution. |
| TRepLiNa, 2025 | Section4.2 Eq1–3: token-state CKA at a selected layer plus REPINA | Layer-wise bilingual supervision | Neither CKA nor pretrained-reference anchoring is used here; supervision is paired directional change across two depths. Its published REPINA description has ambiguities; do not claim exact code reproduction. |
| Teacher-free Feature Distillation, ECCV2022 | Section3.2 Eq3: squared distance between shallow/deep features, deeper feature frozen | Zero-parameter, shared-network inter-layer supervision | Their objective makes depths agree within one input. Here two languages' depth differences agree, while depths need not approach each other. Stop-gradient itself is established prior. |
| FSP, CVPR2017 | Section3.2–3.3 Eq1–2: cross-layer feature-product matrices and teacher/student squared error | Teaching transformation/flow rather than endpoint state | Our vector subtraction/cosine and parallel shared-encoder inputs differ from feature-product matrices and teacher/student networks. The broad 'learn the process' rationale is established prior, not ours. Full PDF fetch was blocked; comparison uses indexed primary-paper Eq1–2 and official record, not an assertion of exhaustive review. |
| Existing P4/P5 in this repo | Encoder ON minus encoder OFF, then effect InfoNCE | Differences used as a learning target | Those are routing counterfactual effects; this is a within-pass depth difference, without ON/OFF passes or InfoNCE. |

Sources:
- https://aclanthology.org/2025.acl-long.778.pdf
- https://arxiv.org/html/2510.06249v1#S4.SS2
- https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860337.pdf
- https://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html

Non-equivalence: adding the same vector offset to both depths of a language
leaves its depth difference unchanged, but generally changes direct state
cosines and FSP products. This is an algebraic property, NOT evidence that
real language offsets are constant across depth. Shared-coordinate subtraction
is meaningful computationally but does not guarantee shared semantic changes.

The reviewed formulas do not directly instantiate this exact objective.
That supports a narrowly scoped pilot, not an exhaustive novelty guarantee.
Do not claim new cross-layer distillation, anti-collapse, syntax preservation,
or superiority from formula differences. Low norms, common directions and
rank collapse remain possible; eps stabilizes computation, not semantics.

## Two runs and stop policy

Amis/mBART seed42 only: TRANSITION_LSR_S42 then TRANSITION_FULL_S42.
Use the pinned amis_drb_bleu baseline protocol: FP32, SDPA, B4/GA32, LR5e-5,
20 epochs maximum, ES10, validation BLEU beam1, original CSV references.
Reuse baseline19th-epoch validation BLEU17.6823291745, chrF++14.7030053622;
its historical backend is reconstructed, not a contemporaneous runtime receipt.
Do not pool these exploratory RTX results with A100 confirmation cohorts.

Advance only if Full exceeds baseline by >=.30 validation BLEU AND exceeds
transition-LSR-only by >=.20 BLEU, without falling below baseline chrF++.
These are development decisions, not significance tests. Failure means stop:
no automatic layer/lambda sweep, test, Ede, replication or A100 launch.

Order: this review -> CPU correctness -> all real GPU gates -> long training.
CPU gates cover explicit hook formula, local donor/target detach, CE/RNG
invariance, padding, degenerate vectors, checkpoint recomputation and resume.
Real gates use exact B4/GA32/FP32/SDPA with optimizer updates, save/reload,
beam-cache comparison and resume replay. Any failure stops the queue.
Checkpoints go to the canonical public repo in distinct method/seed/profile
folders. Delete remote transient weights only after SHA-verified HF backup.
