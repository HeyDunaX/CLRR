# Margin-envelope consistency: proposed method and prior audit

Date: 2026-11-15. Current status: HOLD / NOT ADMITTED FOR TRAINING after exact
gradient decomposition below. No model implementation, GPU job or training
launched. Internal descriptive name, not an established contribution.

## Current admission decision: exact decomposition weakens novelty

The author's follow-up was conditional: debate with NotebookLM and run IF okay.
The prior requirement to reject thin-novelty pilots remains applicable. The
earlier four-run table below is design history/proposal, not an active queue.

An additional formula audit found an exact gradient decomposition. With detached
common weights, delta(v)=m1(v)-m2(v), define

    S = 0.5 * sum_v w(v)*delta(v)^2
    H = -0.5 * sum_v stopgrad(w(v)*abs(delta(v)))*(m1(v)+m2(v))

For BOTH views, at the current logits:

    grad_z L_env = 0.5 * grad_z S + grad_z H.

Thus the proposed direction is exactly half symmetric margin matching plus
adaptive supervised margin sharpening. H is a stop-gradient SURROGATE; the
identity concerns local gradients, not scalar loss values, independently trained
controls, or identical full training trajectories. It does not demonstrate an
exact published predecessor or impossibility of empirical gains. It materially
weakens the proposed independent teaching-mechanism contribution, which already
had an incremental prior audit. No strong-novelty admission is justified.

Independent CPU autograd verification in conda clrr: float64,7rows,11classes,
random logits/weights, both gradient vectors, max absolute error0.0. Zero optimizer
steps, no dataset access. Follow-up NotebookLM query968130fd11a4 submitted with
the exact decomposition and the author's conditional-admission constraint.

Do not implement/launch four training arms merely because all ingredients can
be coded correctly. A later explicit authorization of a thin-novelty engineering
pilot would be a different research decision.

## Decision

Propose a competitor-specific, directed teaching rule between two dropout views.
This is an incremental algorithm candidate worth evaluating with explicit
controls, not a certified strong-novelty ACL method. Existing author restrictions
on novelty admission remain in force; this document does not authorize training.

The architectural cross-depth recovery idea was considered first and set aside:
the repository's DRB proposal already implements a closely related decoder read.
Do not repeat that experiment under a new name.

## Evidence and research question

Existing matched Amis development validation means, seeds42–44:
baseline16.875257; ordinary R-Drop W(beta1.25)18.285427, delta+1.410170.
W is PRIOR ART. Original CLRR/Full did not pass their eight-seed confirmation.
Prefix consistency P, its diagonal D and strength calibration did not establish
a better method than W. See RESEARCH_INSIGHTS.md and PLC_FAILURE_ANALYSIS.md.

The latest frozen branch probe does not establish a relationship between local
dropout KL and downstream branch cost. It cannot justify KL-based token selection
or a claim that exposure bias caused previous failures. No diagnostic established
that symmetric-KL gradient interference caused those failures either.

RQ: Does transferring the stronger reference-versus-competitor margin between
stochastic views improve generation beyond ordinary output KL, symmetric margin
matching, and an equally strong gold-confidence update?

## Formula

For each existing supervised target position with gold token y, run the unchanged
backbone twice with independent native dropout masks, obtaining logits z1,z2 and
probabilities p1,p2. Teacher forcing, labels and data are unchanged. Define for
every vocabulary item v != y:

    m_k(v) = z_k(y) - z_k(v)
    M(v) = stopgrad(max(m_1(v), m_2(v)))
    w(v) = stopgrad((p_1(v)+p_2(v)) /
                    sum_{u != y}(p_1(u)+p_2(u)))
    d_k(v) = relu(M(v)-m_k(v))
    L_env = 0.5 * sum_{k=1,2} sum_{v != y} w(v)*d_k(v)^2
    L = (CE(z1,y)+CE(z2,y))/2 + beta*L_env

Average both CE and auxiliary over the SAME labels != -100, including any
supervised language/EOS labels inherited from W. Weights use a stable log-space
normalization over all non-gold items; no top-k vocabulary truncation or
reference-dependent filtering. Candidate adds no parameters or inference work.
No original CLRR, LSR, target encoder, teacher model or rollouts in this version.
Its connection to the original project is task-directed alignment; this is a
replacement method, not a claim that original CLRR/LSR succeeded.

If margins are0.8 and0.2 for one competitor, only the second view is taught that
gap. Teacher identity may reverse for another competitor. If both margins are
negative, the weaker is moved toward the less-negative value: this does not make
either view correct. CE remains necessary.

## What can be proved, and what cannot

With detached w and M, let a_k(v)=w(v)*d_k(v), S_k=sum_v a_k(v).
The unweighted auxiliary logit gradient for view k is

    dL_env/dz_k(y) = -S_k
    dL_env/dz_k(v) = a_k(v) >= 0.

For an isolated direct gradient step on logits, every gold-versus-competitor
margin increases by eta*(S_k+a_k(v)). A competitor's stronger view has zero
gradient FROM THAT PAIR, but can receive a gold-coordinate gradient from other
competitors. The complete gradient sums to zero across vocabulary coordinates.
This property is elementary, not a new task-preservation theorem.

There is NO corresponding guarantee for shared backbone parameters, AdamW,
held-out prefixes, translation quality, calibration or future optimizer steps.
Even though the loss VALUE equals weighted symmetric margin MSE/2, its autodiff
direction differs because the winning target is detached. With positive weights,
the zero set is still p1=p2, exactly as ordinary R-Drop. Do not claim a relaxed
consistency zero set or semantic equivalence awareness.

Risks: selection of unusually large noisy margins, overconfidence, suppression
of legitimate alternative translations, and shared-parameter interference.
Unlike PLC there is no cumulative prefix sum or gold-probability inverse factor;
this is a formula distinction, not evidence those factors caused PLC failure.

## Primary-source and NotebookLM audit

| Prior read | Already established | Difference and boundary |
|---|---|---|
| [R-Drop](https://papers.nips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf), Eq2 | Two dropout passes, CE plus symmetric output KL | Candidate changes the teaching direction at each gold/competitor pair; two-pass consistency is inherited |
| [Refined Logit Distillation](https://arxiv.org/html/2408.07703v2), Section4 | Label-informed teacher correction, confidence matching and masking misleading class relations | Candidate does not mask classes above gold or swap/augment a teacher probability; label-safe distillation is not new |
| [Label Revision](https://www.comp.hkbu.edu.hk/~ymc/papers/journal/TCDS-2025_3559881-publication-version.pdf), Eqs7–9 | Revise teacher probabilities by mixing in the hard label | Candidate uses per-competitor logit margins from current stochastic views, not a revised fixed-teacher distribution |
| [TPKD](https://arxiv.org/html/2609.39338v1), Section5, Eq6 and AppendixB.1 | Preserve CE margin gains by projecting conditional-KD gradients onto a safe cone | Candidate regresses toward a detached cross-view margin envelope; no conditional-distribution target or nearest-cone projection. Task-preserving learning is prior art |

TPKD is a September2026 preprint; publication status is not assumed. Differences
in these inspected formulas do not establish absence of every possible prior or
an ACL-level contribution. In particular, simple margin regression, stop-gradient
teachers and winner selection are established ingredients. Remaining potential
contribution: competitor-specific stochastic teaching plus an empirically
supported explanation of when it helps. No first-safe-distillation claim.

NotebookLM notebook bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890:
initial audit query ee3341bb921c; follow-up eab6cef7440d after importing TPKD
source b358b94c-9187-4fb0-bfc9-28b3ea72ba16. Both completed. The follow-up
restricted sources to TPKD and R-Drop. Advisory assessment: incremental novelty,
no exact general-form identity identified in the checked formulas. This is not
a novelty certificate.

Corrections to raw assistance: initial zero-set distinction was false; candidate
has zero-sum logit gradients (mass conservation); a stronger view is frozen only
for a particular pair, not the entire vector. Moving a negative gold margin upward
does not itself sharpen an incorrect rival prediction. The statement that no
teacher choice can EVER yield the same TPKD vector is too strong: zero directions
and isolated coincidences exist. General update rules differ; do not assert
universal disjointness. Gold-relative margins also encode non-gold log-odds by
subtraction, so they do not literally discard all non-gold relational information;
the candidate simply does not explicitly fit a teacher's conditional distribution.

## Proposed seed42 pilot: four NEW models, not launched

Amis->Mandarin, unchanged CSVs, mBART-50; use the exact W RTX protocol/runtime
after a provenance audit. Existing lock: FP32, SDPA, TF32off, microbatch4, GA32,
LR5e-5, max20epochs, ES10, validation BLEU greedy/beam1, earliest tie.
No test-based selection. No changing baseline or old method scores.

| New run | Auxiliary | What it isolates |
|---|---|---|
| M | Directed envelope above | Main candidate |
| S | 0.5*sum_v w(v)*(m1(v)-m2(v))^2, no detached winner | Same numeric discrepancy with symmetric teaching |
| G | Adaptive weighted CE with SAME gold-logit auxiliary gradient as M | Rival-specific teaching versus mere confidence boosting |
| T | Two-view adaptation of TPKD, other dropout view as detached teacher in each direction | Closest task-preserving prior; not the original paper's full training reproduction |

G is defined without an additional threshold: for each view compute the same
detached S_k as M and set

    c_k = stopgrad(S_k / (1-p_k(y)))
    L_G_aux = sum_k c_k * CE(z_k,y).

Then gold-coordinate gradient=-S_k, competitor gradient=S_k*p_k(v)/(1-p_k(y)).
This matches the total gold pressure at the SAME logits, not the entire gradient
norm, optimizer update or later training trajectory. Compute the non-gold mass
stably in log space; no silent cap. Nonfinite values are technical HOLD.

Reuse existing baseline/W/two-pass CE/Middle/Strong LoRA only with matching
recipe/provenance; A100 numbers are not matched RTX controls. W is the primary
quality comparator, baseline is secondary. Do not rerun old CLRR/Full/LSR.

Before training, choose one coefficient per objective on a fixed first64 TRAIN
rows and four fixed mask pairs, matching initial RMS auxiliary logit-gradient
norm to W beta1.25. G inherits M's coefficient to preserve its exact gold-gradient
control; do not independently calibrate G. Record subset, seeds, gradients,
coefficients and hashes. Zero/nonfinite norms HOLD. This controls one initial
strength statistic only. No validation sweep. No coefficient is fixed yet.

## Sequence and decision criteria proposed before any pilot result

1. Scientific CPU checks: detached-winner gradients, crossing winners, both-wrong
   margins, zero set, translation invariance, zero-sum gradients, exact G control,
   numeric equivalence but gradient difference versus S, padding/special labels.
2. Tiny RTX gate: original dropout/RNG schedule, GA/checkpointing equivalence,
   finite optimizer steps, memory admission, strict reload/native generation.
3. Only with author admission, train four seed42 arms, validation only; log CE,
   BLEU/chrF++, reference rank, gold-confidence, entropy, margin-gap distributions,
   directional gradient norms and timing. Start from the same pretrained weights,
   not continuation from a selected best checkpoint.
4. Practical development screen proposed: M exceeds W by >=0.30 validation BLEU,
   exceeds S/G/T, and does not lose validation chrF++. These are development
   criteria, not statistical significance or evidence that margins caused gains.
   If not met, do not promote the mechanism or open a parameter sweep.
5. If promising, replicate M on seeds43/44 against existing matched W/baseline;
   test only after selection is locked. A one-seed result never establishes
   statistical stability. Paired sentence bootstrap and across-seed uncertainty
   answer different questions; no significance guarantee.
6. Transfer the locked recipe to Ede and final A100 confirmation only after the
   Amis decision. A credible ACL case requires replicated gains over strong
   controls, transfer, and a defensible mechanism; this card supplies none of
   those outcomes yet.
