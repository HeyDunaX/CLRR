# Transition LSR v1 — completed focused novelty verdict

## Decision: NO-GO as the next novel-method pilot

This supersedes the provisional DESIGN_REVIEW_PASS_FOR_BOUNDED_PILOT in
TRANSITION_LSR_DESIGN.md and its frozen bundle. Keep those records unchanged
for provenance. No GPU gate, training, or automatic retry is admitted.

## Newly identified closest primary prior

Gong et al., **Beyond Logits: Aligning Feature Dynamics for Effective Knowledge
Distillation**, ACL2025, pp23067–23077:
https://aclanthology.org/2025.acl-long.1125.pdf

Read Sections3.1–3.4, specifically Eq5 and Eq7 plus definitions immediately
below Eq7. FDD aligns cosine directions of differences between selected layer
representations. Its representations are log predictive distributions obtained
through teacher/student LM heads. It also includes trajectory and output losses.
Thus the central rationale of supervising changes across depth, and the
cosine-of-layer-differences operator, already exist in a directly relevant NLP
prior. A two-layer gap is not by itself a distinct principle: FDD selects depths.

## Exact comparison, without claiming identity

| Aspect | FDD | Proposed Transition LSR |
|---|---|---|
| Views | Distinct teacher/student models | Shared evolving encoder, parallel source/reference |
| Compared quantity | Differences in LM-head log distributions | Differences of pooled hidden states before recipient rewire / after donor rewire |
| Loss core | Cosine distance of depth differences | Cosine distance of depth differences |
| Gradient | Student learns from teacher | Target detached; explicit local source donor detach, normal sequential gradients retained |
| Routing | Not the proposed fixed CLRR edge injection | Original fixed CLRR, one arm without routing |
| Extra terms | Trajectory KD and output KD | Translation CE, no original endpoint LSR |

The implementations and complete objectives are not identical. However,
changing the application to bilingual encoder views, pooling and the detach
placement has not yet established a substantive new methodological contribution.
The constant-offset invariance argument also follows from subtraction, so it
cannot distinguish the proposed mechanism from generic delta alignment.
This is a novelty-strength judgment, not proof that every cross-lingual
extension of FDD is unpublishable or that this candidate would score poorly.

## Connection to empirical evidence

Existing results support investigating why alignment quality does not reliably
translate to translation quality. They do not show that matching bilingual
layer transitions is the missing mechanism. No diagnosis yet justifies the
particular two-depth supervision beyond that general hypothesis. Earlier
route-use evidence establishes co-adaptation, not a benefit over baseline.

Therefore the current two runs would mostly test a cross-lingual delta-alignment
adaptation, while the author explicitly requires a sufficiently justified
novelty contribution before spending GPU. Do not label that a novelty PASS.

## Execution state and next research decision

Five CPU tests passed: useful implementation correctness, not novelty evidence.
RTX was only queried read-only; no new GPU gate or long run was started.
Keep the original CLRR/LSR implementation and all old numbers unchanged.
Do not add a gate, projector, new loss, or layer sweep to manufacture distance
from FDD. Before proposing another pilot, write a concrete CLRR/LSR-specific
mechanism hypothesis from existing evidence and check its nearest complete
prior formula first. Any FDD-derived design must explicitly cite FDD and
identify what substantive claim is added and which matched control tests it.
