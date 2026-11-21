# Token consistency budget: novelty audit

Date: 2026-11-14

## Follow-up: source-response consistency — rejected before implementation

After rejecting the gradient-budget candidate below, we considered stabilizing
the change in token predictions induced by switching the source. This follow-up
was audited before implementation, smoke tests or training.

For two dropout realizations m, use the same gold target prefix with matched
source x and a randomly mismatched source x_minus:

    d_m = log p_m(v | x, prefix) - log p_m(v | x_minus, prefix)
    q_m = softmax(d_m / T)
    L = mean CE on matched-source views + beta * symmetric_KL(q_1, q_2)

**Decision: NO-GO for training this formulation as the next main method.**
This is not proof that every detail of the complete algorithm was previously
published. It rejects a weakly motivated candidate with close prior art and an
unresolved failure mode.

### Closest prior, verified directly

[Comparative Knowledge Distillation](https://arxiv.org/pdf/2311.02253),
Section 3.2, Equations 1–4, defines softmax of vector differences between
sample representations, aligns teacher/student comparison distributions with
KL, and combines that objective with CE. The preprint explicitly includes
logits as representations; its published WACV 2025 version is indexed at
[CVF](https://openaccess.thecvf.com/content/WACV2025/html/Xu_Comparative_Knowledge_Distillation_WACV_2025_paper.html).

Algebraically, vocabulary-constant log partition terms cancel:

    softmax((log p_true - log p_wrong)/T)
      = softmax((z_true - z_wrong)/T).

Thus the relational transform itself is already present in CKD. Remaining
differences are shared-network dropout self-consistency, symmetric two-sided
gradients, and paired source conditions at a fixed autoregressive prefix.
These distinctions do not alone establish a strong new contribution.

Other checked primary sources:

| Source | Verified overlap / distinction |
|---|---|
| [Deep Decoder, Section 3.4](https://aclanthology.org/2022.findings-acl.39.pdf) | DDR matches two decoder output distributions with symmetric KL; ALD contrasts pooled decoder representations under different source corruption levels. Neither is the exact proposed difference-distribution objective. |
| [CBMI, Equations 7–13](https://aclanthology.org/2022.acl-long.169.pdf) | Translation/target-LM log quotient, then token/sentence CE weighting; different denominator and objective. |
| [CDCM, Equation 16 and KD objective](https://pmc.ncbi.nlm.nih.gov/articles/PMC12860726/) | Subtracts counterfactual teacher logits before soft-label distillation; separate teacher/student classification setting. |
| [Source-contrastive decoding, Equation 1](https://arxiv.org/html/2309.07098) | Subtracts probabilities inside the decoding log objective. It is NOT identical to subtracting log probabilities and is not this training regularizer. |

### Mathematical and evidential checks

- If both dropout views ignore the source, d_1=d_2=0, q_1=q_2 is uniform,
  and the auxiliary KL is zero. This is a minimum of the regularizer, not proof
  that the combined CE objective necessarily converges to source collapse.
- The logit cancellation above removes any direct inverse-probability
  singularity. Tiny denominator probabilities alone do not prove numerical
  instability or gradient explosion. Irrelevant tokens with large logit
  differences can nevertheless dominate q; whether that occurs needs data.
- Old positive source-grounding probes concern older models. They neither
  establish a missing-grounding bottleneck in W nor disprove every grounding
  problem. New W/B measurements would be necessary.
- Four source-conditioned decoder views increase compute relative to ordinary
  two-view R-Drop. Exact cost depends on encoder reuse and implementation;
  no measured timing or fixed speedup/slowdown is claimed.

### NotebookLM audit and corrections

Four public sources were added, followed by CKD. The author explicitly
authorized sending candidate formulas and unpublished results after automatic
review initially blocked that export. Candidate query `958846bc7b01` completed;
follow-up correction query `9c9d6621b053` also completed and identified CKD
Equations 1–3 as the closest relational transform.

The initial response conflated source-contrastive probability subtraction with
log-probability contrastive decoding, overstated gradient explosion and claimed
older grounding measurements disproved missing grounding. These statements are
rejected here. The follow-up corrected the first two; project grounding scope
is governed by the local receipts, not the assistant's generalization. ALD at
equal positive/negative similarity gives log(2), not its maximum. NotebookLM
is an aid to review, not the evidence for the final verdict.

### New CPU evidence: W versus B, all three development seeds

Reused the existing `scripts/analysis/analyze_prediction_strata.py` in conda
clrr on six saved prediction sets and all 576 aligned validation rows. Receipts,
reference hashes, FP32 beam1 settings and BLEU/chrF++/chrF0 replay checks PASS.
No checkpoint weights, GPU, generation, test, or new significance test used.
Temporary input copies were removed automatically.

| Seed | W−B corpus BLEU | Range after omitting each single row | Sentence chrF0 wins / losses / ties | W−B BLEU on longest-source quartile |
|---|---:|---:|---:|---:|
| 42 | +0.3617 | +0.1847 to +0.4545 | 197 / 200 / 179 | +0.8579 |
| 43 | +0.8200 | +0.6394 to +0.9171 | 187 / 211 / 178 | +1.9164 |
| 44 | −0.1393 | −0.3202 to −0.0117 | 173 / 215 / 188 | +0.6764 |

Longest-source quartile means >8 whitespace-delimited source words; quartile
cuts are 5/7/8. This post-hoc subset pattern is a diagnostic lead, not a
linguistic category, causal explanation, additive contribution to corpus BLEU,
or an independent replication. W does not improve most individual sentences
by sentence chrF0; that fact can coexist with a higher corpus BLEU. No single
omitted row changes the sign of any seed's full-corpus BLEU difference. This
does not establish a confidence interval or statistical significance.

Artifacts: `outputs/experiments/amis_consistency_strength_w_replication_rtx_suite/validation_error_audit/`.

### Next bounded step: diagnose existing models, no new training

Research question: does weaker output consistency preserve useful source
response better, or mainly alter prediction confidence/length? This question
is still open; the length pattern above does not answer it.

1. Restore only baseline/B/W seed42 checkpoints by their pinned revisions and
   archive hashes. Audit the existing baseline backend caveat; do not silently
   substitute another baseline.
2. Frozen validation-only probe: first 64 rows, fixed dropout seed pairs
   42001–42004, two fixed derangements of source rows avoiding identical source
   strings, same gold target prefixes. Record ordinary matched-source dropout
   KL, matched-minus-mismatched gold log probability, source-response variation
   across masks, gold-vs-strongest-competitor margin, and response-distribution
   entropy/tail mass. Define tail by TRAIN target-token counts, not validation
   labels or the candidate's own rankings. Keep source-EOS/language/pad tokens
   separate. Check actual vocabulary size from the pinned model config.
3. Use paired RNG states for matched/mismatched passes within each view and
   independent states across views; document that variable source lengths
   prevent a perfect semantic match of all encoder dropout positions. No
   optimizer, checkpoint reselection or assertion about training trajectory.
4. Report token-level quantities aggregated by sentence and then mask/pair;
   repeated masks are not independent training seeds. Join fixed rows to saved
   generation diagnostics descriptively, without threshold tuning on test.
5. If no reproducible useful source-response deficit is found, close this
   direction. If found, define a mechanism-specific candidate and repeat the
   immediate prior-art audit before code/smoke/training. Do not automatically
   train the rejected formula or open a lambda sweep.

The frozen probe above is a plan, NOT executed in this follow-up. W remains an
attributed tuned R-Drop control; existing CLRR/LSR results remain unchanged.

## Decision

**NO-GO as the next standalone method contribution in its current form.**
This is a research prioritization decision, not proof that the exact full
algorithm has previously been published. Do not open the proposed four-run
pilot on the strength of the earlier chat plan. No code or training was launched
for this candidate in this audit.

## Candidate audited

For each non-padding target token, let c be mean CE over two native dropout
views, k their full-vocabulary symmetric KL, and z their concatenated logits.

    g = gradient_z(c)
    h = gradient_z(k)
    a = stop_gradient(min(1.25, norm(g) / (norm(h) + epsilon)))
    loss = mean_valid_tokens(c + a*k)

Only the coefficient is detached. Both arguments of symmetric KL remain
differentiable. The proposal adds no inference module.

## Primary sources checked directly

| Source | Verified mechanism | Remaining distinction |
|---|---|---|
| [VQGAN official implementation](https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py) | `calculate_adaptive_weight` uses a gradient-norm ratio, clamp, and detach to balance reconstruction and adversarial losses | Candidate applies a related rule to CE/KL at each token's logits rather than a batch loss at decoder parameters |
| [Adaptive Gradient Balancing, Section IV-B](https://arxiv.org/html/2104.01889) | Bounds auxiliary adversarial output-gradient scale relative to supervised pixel-loss gradients, using moving statistics | Candidate uses instantaneous token norms and symmetric dropout consistency |
| [MetaBalance, Algorithm 1 and Strategy A](https://arxiv.org/html/2203.06801) | Rescales an auxiliary gradient by target/auxiliary norm ratio when auxiliary magnitude is too large | Candidate changes the unit of adaptation to token logits and the auxiliary objective to dropout KL |
| [GradNorm, Section 3](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf) | Learns task weights using shared-parameter gradient norms and relative training rates | Candidate has a direct coefficient without a separate weight optimizer |
| [Selective KD for NMT, Section 5](https://aclanthology.org/2021.acl-long.504.pdf) | Uses word CE to select tokens for distillation | Candidate continuously weights consistency rather than selecting a hard subset |
| [AdaKD, Sections 3.4-3.6](https://arxiv.org/html/2510.11615) | Adapts token selection and token temperatures using learning difficulty | Does not establish identity with the proposed norm-ratio rule, but token-adaptive distillation is existing prior art |

VQGAN's paper prints a gradient ratio in Equation 7; the exact norm/clamp/detach
implementation was checked in the authors' code rather than inferred from that
notation.

## Why the overlap matters

Ignoring numerical epsilon, define the original weighted auxiliary logit
gradient u = 1.25*h. The proposed contribution is:

    a*h = min(1, norm(g)/norm(u))*u

It preserves u when its magnitude is below the supervised budget, and otherwise
rescales it to norm(g). This is the same magnitude-capping operation as the
reduce-only strategy in MetaBalance, now applied independently in token-logit
space. The full algorithms differ in scope and computational graph; this is not
a claim that they produce identical parameter updates.

Consequently, neither gradient-ratio balancing nor the norm cap is a defensible
new primitive. Token-level application may be an adaptation worth testing as a
control, but presently lacks evidence supporting a standalone method claim.

## Empirical motivation versus missing evidence

Existing development results favor fixed full-KL weight 1.25 over 2.5 on mean
validation BLEU across seeds 42/43/44. Gold-only and rest-only pilot branches
failed. These observations motivate checking regularization strength; they do
not establish that token-specific norm caps improve generalization.

The frozen probe used 64 validation rows, 717 supervised tokens and four dropout
masks at final baseline/B checkpoints. Its recorded aggregate logit-gradient
cosines are positive. It does not demonstrate aggregate gradient conflict,
token-level harmful pressure, parameter-gradient dominance, or Adam update
dominance. Halving the coefficient in that probe is an analytic rescaling, not
a measurement on the trained W checkpoint.

## Consequences

- Keep W as a reduced-weight R-Drop comparator, not a new method claim.
- Withdraw the earlier four-run candidate pilot from the immediate plan.
- A future method must first pass a primary-source novelty review, including
  explicit comparison to the nearest algorithms; NotebookLM is a reviewer and
  retrieval aid, not a novelty certificate.
- Any later efficacy study of this cap must include tuned fixed consistency
  and controls for batch-wise strength and token assignment.
- No guarantee of translation improvement, statistical significance, or ACL
  acceptance follows from this audit.

## NotebookLM review provenance

Notebook: `bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`.
Fast literature search task: `b3020bce-3fb7-475b-a4b9-341ffad0b602`.
Six primary-source URLs above were added to the notebook for source-grounded
review. Unverified search hits were not imported wholesale.
The targeted review query is `ee4742f9f9a9`; its result is supplementary to the
direct source comparison above.

The query completed and recommended NO-GO as a standalone algorithmic
contribution, while distinguishing potential engineering utility from
originality. It did not identify exact identity with a complete prior algorithm.
Its response contained inline citation numbers but empty structured source and
reference mappings; therefore those citation numbers are not used as evidence
in this audit. Mechanism claims above were checked directly against primary
sources. The review covered VQGAN, AGB, GradNorm and AdaKD; the MetaBalance and
Selective KD comparisons were checked separately and their URLs also added to
the notebook. Claims in the generated review about a known R-Drop failure,
computational cheapness, mandatory benchmark choices, or certain reviewer
reactions are not treated as established findings.
