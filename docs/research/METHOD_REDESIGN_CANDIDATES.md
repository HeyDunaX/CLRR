## Proposed candidate: Prefix Likelihood Consistency (PLC) — 2026-11-14

Status: AUTHOR APPROVED IMPLEMENTATION AND CONDITIONAL RTX PILOT. Isolated v1
code and five CPU correctness tests passed; W42 scientific admission, train-only
calibration and real GPU optimizer/resume gates are required before training.
This is a proposed next research direction, not an established improvement or
certification of ACL novelty. The preceding source-assignment/projection and
RO screens are not active pilot requests. A preparation bundle cannot start
long training; pilot admission is recorded separately.

### 1. Evidence, hypothesis and contribution

Ordinary R-Drop W(beta1.25) is the positive development anchor: mean validation
BLEU18.2854 versus baseline16.8753 over seeds42–44. Its +1.4102 improvement is
not a new method contribution. CLRR/LSR have not shown a stable confirmation
advantage, and G gold-only consistency failed when used alone. PLC therefore
keeps W's full-vocabulary consistency instead of replacing it with gold-only
matching. No claim that CLRR/LSR are universally ineffective follows.

Hypothesis: covariance between dropout-induced log-likelihood changes at
different target positions can produce coherent prefix-score drift. A
token-separable regularizer does not explicitly distinguish equal-magnitude
changes that reinforce each other from changes that cancel. This phenomenon
has NOT yet been measured in W. Long-source gains in an earlier audit are a
post-hoc lead, not evidence for this target-position mechanism.

RQ1: Is ordered prefix likelihood drift present beyond tokenwise variance?
RQ2: Does constraining that drift improve generation beyond ordinary R-Drop,
the same diagonal weighting, endpoint-only matching and initial-strength control?
RQ3: Does the result replicate across seeds and a second language direction?

The possible contribution is a measured cross-position mechanism plus a simple
objective with discriminating controls. It is not the invention of dropout,
consistency, sequence likelihoods, or quadratic regularization. If only generic
extra regularization helps, do not make the proposed mechanism claim.

### 2. Exact objective and gradient contract

For each (x,y), run the SAME model twice with independent native dropout,
identical teacher-forced target inputs and original source/target IDs. Let T
count the exact existing W valid labels: labels!=-100, INCLUDING supervised
special tokens such as EOS and any supervised language token. Do not silently
remove a special-token position or change the W CE/KL reductions. This mask
was inspected in src/methods/decoupled_consistency/v1/model.py, which explicitly
records inclusion of supervised special tokens. Record the actual label IDs
in the implementation gate to verify tokenizer behavior.

    a_t = log p_theta,m1(y_t | x,y_<t) - log p_theta,m2(y_t | x,y_<t)
    S_k = sum_(t<=k) a_t
    L_P = (1/T) sum_(k=1..T) S_k^2/k
    L_PLC = L_W + lambda * mean_sentences(L_P)

L_W is the existing mean two-view CE plus1.25 times its existing full-vocabulary
SKL term, unchanged. Proposed first-pilot lambda=0.10 is a declared engineering
starting value, NOT theoretically optimal or selected from validation results.
No lambda sweep is included. Both views receive gradients; neither is detached.
Use FP32 log_softmax and cumulative sums; never multiply small probabilities.
Pack valid label positions per sentence so padding cannot change prefixes.

Gold log probabilities backpropagate to ALL logits:
d log p(y)/d z_v = 1[v=y]-p(v). This does not mean gold probabilities determine
the complete vocabulary distribution; keeping W preserves that additional
constraint. No extra encoder, teacher, trainable module, route, lexical table,
negative source, generated candidate or altered data is required. Additional
loss work/storage is O(B*T) using existing two-pass logits, relative to W;
actual wall time and peak memory remain to be measured. Inference stays native.

### 3. Algebra and limitations

    L_P = a^T K a
    K_st = (1/T) sum_(k=max(s,t)..T) 1/k
    trace(K)=1; K is positive definite for finite T>=1.

The diagonal control uses EXACTLY K_tt, so it retains the early-position
weighting. The difference is the signed cross term, not a uniformly stronger
penalty. Negative cross-products can make L_P smaller than its diagonal part.
For fixed(x,y), independent identically distributed dropout views give
E[a_s*a_t]=2 Cov_m(log p_m(y_s),log p_m(y_t)). This follows from independence
and equal means, not from assuming independent target positions.

For a=(1,-1), L_P=0.5, diagonal=1.0, endpoint=0.0. Endpoint matching can miss an
intermediate excursion. This example is algebra, NOT an observed model error.
CPU float64 checks in conda clrr verify the quadratic form, derivative2Ka,
trace/positive definiteness for T1/2/4/17, and non-gold logit gradients:
outputs/experiments/prefix_path_design/MATH_CHECKS.json.

L_P=0 iff the gold log-probability differences are all zero; other vocabulary
probabilities need not match. Arbitrarily extreme log-probability differences
can make the squared objective large; log_softmax avoids probability underflow
but does not bound gradients. Monitor finite values and gradient norms using
the unchanged shared clipping policy. Do not silently add clipping or change
lambda during a run. The loss can also encourage adjacent changes to cancel;
this is a genuine failure mode, constrained partly by unchanged W, not excluded
by positive definiteness. The kernel emphasizes earlier target positions;
diagonal controls are essential, and tokenization affects the path.

This is a gold-prefix SCORE discrepancy, not KL over sequence distributions,
an exposure-bias solution, a semantic-equivalence metric, a causal attribution,
or a guarantee of source grounding. R-Drop sums conditional divergences; it
does NOT assume target positions are statistically independent. Reduced path
variance alone is not sufficient: generation and CE may deteriorate.

### 4. Primary literature and NotebookLM audit

| Prior | Verified relation | Distinction to test, not a novelty certificate |
|---|---|---|
| [R-Drop, Eq2–4](https://arxiv.org/pdf/2106.14448) | Two stochastic views, CE and symmetric output KL | PLC adds an explicitly ordered quadratic of cumulative gold log-probability differences |
| [Fraternal Dropout, Eq1–3](https://arxiv.org/pdf/1711.00066) | Shared-weight dropout views, squared pre-softmax differences and variance interpretation | PLC does not invent variance consistency; it applies a prefix integration kernel to normalized gold log probabilities |
| [LR-Drop, Section3](https://aclanthology.org/2024.lrec-main.891.pdf) | Adds hidden-state and attention-matrix consistency | No hidden/attention matching in PLC; attention matching already involves relations between positions, so do not call all LR-Drop operations position-independent |
| [SeqKD, Section3.2](https://aclanthology.org/D16-1139.pdf) | Sequence-level distillation approximated with teacher beam outputs | PLC has no external teacher or replacement training targets; it evaluates shared gold-prefix scores |
| [CCL, Section3](https://aclanthology.org/2024.emnlp-main.768.pdf) | Adapts consistency-related training through difficulty/proficiency weights | PLC's proposed object is cross-position score accumulation, not a curriculum or sample-weight schedule |

NotebookLM queries79f9282a190b and4a7c296e29b1 completed. The latter restricted
the audit to actual R-Drop/Fraternal/LR-Drop/SeqKD sources newly supplied/read.
Neither audit identified this exact kernel in those sources. This is bounded
prior coverage, NOT global novelty verification. Additional web searches for
prefix/cumulative likelihood consistency did not establish absence of prior.
The expected-variance interpretation is already conceptually present in
Fraternal Dropout; only the particular ordered object and its empirical value
are proposed differences. A new name or positive pilot alone is insufficient.

Corrections to the first NotebookLM answer: gold log probabilities DO gradient
all logits; R-Drop makes no target-independence assumption; LR-Drop attention
matrices encode token-pair relations; PSD does not imply L_P>=L_diagonal.
Discard automated predictions of reviewer decisions or guaranteed novelty.

### 5. Prospective no-training diagnostic and admission

First inspect an existing W42 checkpoint, with fixed original576 validation
rows and8 independently seeded dropout pairs per row. Teacher forcing only,
no optimizer/test, exact checkpoint/RNG replay and masks. Compute per row/pair
L_P, L_D=sum_t K_tt*a_t^2, L_E=(sum_t a_t)^2/T, and C=L_P-L_D; keep only small
JSON/CSV. Existing source-response per_sentence CSVs aggregate tokens and
cannot reconstruct this ordered vector; do not pretend this diagnostic is
already available. Reuse its checkpoint provenance and shared loader.

Average pairs WITHIN each sentence before sentence-level uncertainty analysis;
do not treat8 masks as8 independent examples. Report mean C, C/mean L_D,
sentence-bootstrap95%CI and prespecified target-length strata. This quantifies
a score property, not generation causality or independent population evidence.
If the CI does not support positive net cross-position drift, HOLD this pilot:
the proposed positive-accumulation premise lacks support. Do not interpret an
inconclusive CI as proof of zero covariance or open a different kernel sweep.
If supported, finish source audit and technical gates before training.

### 6. Bounded pilot proposal: Amis to Mandarin, mBART50, seed42

Up to FOUR new models; none is authorized/launched by this document:

| ID | New run | Specific question |
|---|---|---|
| P | W +0.10 L_P | Does prefix accumulation regularization improve generation? |
| D | W +0.10 sum_t K_tt*a_t^2 | Is any gain merely the same positional weighting without cross terms? |
| E | W +0.10 (sum_t a_t)^2/T | Are intermediate prefixes useful beyond whole-sentence score matching? |
| Wcal | Ordinary R-Drop with fixed TRAIN-only calibrated beta | Is initial regularizer strength a sufficient alternative explanation? |

Reuse baseline, W and B only with the existing provenance caveats and matching
training/selection recipe; no blanket rerun. Wcal beta is computed ONCE at the
common pretrained initialization from64 fixed TRAIN rows and paired masks,
without updates. Let g_K,b=grad SKL_b and g_P,b=grad L_P,b over all trainable
parameters. Set beta_cal=sqrt(sum_b||1.25g_K,b+0.10g_P,b||^2 /
sum_b||g_K,b||^2). This matches the aggregate initial squared regularizer
gradient norm, not its direction or subsequent Adam trajectory. Zero/nonfinite
denominator means admission failure. Freeze/record beta before validation or
training, then restore pristine RNG/model state; never recalibrate by epoch.

All new arms start from the SAME pretrained backbone; no continuation from W
while comparing against full-training controls. Lock FP32/SDPA/TF32off, B4GA32,
LR5e-5, ES10/max20, original data and language IDs, validation BLEU beam1
selection and unchanged optimizer/scheduler/clipping. CPU/full-shape GPU gates
must cover both-view gradients, valid-label packing, exact lambda0=W behavior,
checkpointed accumulation/dropout, save/reload/generation and resume. Wcal
calibration is additional preparation, not training. No RTX job starts now.

Pilot practical promotion proposal: P must exceed W by at least0.30 validation
BLEU AND exceed D/E/Wcal; report all scores, CE and chrF++ without claiming
significance. The0.30 threshold is a declared engineering screen, not a power
calculation or universal effect-size rule. If D/E/Wcal matches or beats P, do
not attribute gains to ordered prefix accumulation. Mixed or tiny results HOLD
expansion instead of automatically tuning lambda or adding architecture.

Only if promoted: replicate P and the strongest new competing control on
seeds43/44 (FOUR further models), reusing matched W43/44. Treat this as
development, then decide a locked second-direction study on Ede-to-Vietnamese
with its own matched baseline/W/candidate. Independent confirmation and A100
paper numbers require a subsequent frozen design and sufficient seed-level
evidence; a three-seed pilot cannot certify reliable superiority. Existing
A100 comparator scores are preserved, but precision/selector mismatches must
be resolved before claiming fully matched comparisons or automatic reuse.

## Active design brief — 2026-11-14

Status: RESEARCH/DESIGN ONLY. No candidate is admitted to training by this
entry. The author asked us to screen weak drafts internally and return a
complete, worthwhile proposal rather than a sequence of rejected pilots.

### Evidence that the next proposal must explain

- Ordinary R-Drop W(beta1.25), seeds42–44, has mean validation BLEU18.2854
  versus baseline16.8753: +1.4102, positive3/3. These are development results,
  not independent confirmation or proof of a new method.
- W exceeds ordinary R-Drop B(beta2.5) by0.3475 mean BLEU, positive2/3.
  Changing consistency strength is a necessary control, not a novel method.
- CLRR/LSR confirmation did not reproduce the earlier development advantage.
  This does not establish source neglect, harmful anisotropy, or a universal
  failure of residual reuse. Existing probes must not be rerun under new names.

### Research target and admission requirements

Target RQ: which translation-relevant information should remain invariant
under stochastic model perturbations, beyond matching all token probabilities?
This is an open question, not an established defect of vocabulary consistency.

A final proposal must specify the representation/distribution being preserved,
why this object is relevant to generation, an explicit objective and gradient
paths, degenerate solutions, inference behavior, and its closest mathematical
prior. An unfamiliar name or a different normalization axis is insufficient.
It must also explain how to distinguish a specific mechanism from simply
weaker regularization, extra CE passes, or additional computation.

Candidate-specific controls should reuse the existing baseline and W only when
their data, initialization, training/selection recipe and artifact provenance
match. Existing A100 comparators remain unchanged; they are not automatically
matched controls for a new RTX recipe. No numerical quality gate is being
retroactively registered by this design brief.

The eventual sequence is: complete one candidate card + primary-source audit
and NotebookLM critique; CPU correctness; full-recipe GPU smoke; bounded
seed42 validation-only pilot with mechanism controls; seed replication and
second-language development if warranted; frozen method and A100 evaluation.
No test-based candidate selection and no automatic expansion to a large queue.

### Internal screen receipt (not a proposed pilot)

NotebookLM query cf13809a1873 completed for a source-assignment consistency
draft: s_m(i,j)=mean_t log p(y_it|x_j,y_i<t,m), r_m=softmax_j(s_m/tau),
with matched CE, correct-source classification and SKL(r_1,r_2). Its supervised
term is conditional InfoNCE and its consistency term is ordinary SKL on a new
categorical object. No standalone new primitive is claimed and this draft is
not selected for implementation or training.

Primary sources directly read in this screen:
[R-Drop](https://arxiv.org/pdf/2106.14448),
[CorefCL](https://aclanthology.org/2021.wmt-1.121.pdf), and
[Curriculum Consistency Learning](https://aclanthology.org/2024.emnlp-main.768.pdf).
CorefCL Section4.2 already contrasts conditional likelihoods under original
and corrupted context with a margin objective. CCL Section3 already adapts
consistency-related training by difficulty/proficiency. These are substantive
prior constraints, not proof that every source-assignment variant is identical.

NotebookLM is a critique aid, not a novelty authority. Its response contains
an inconsistent CKD author attribution and unsupported generalizations about
the causes/frequency of low-resource errors; those are not adopted. Uniform
source priors and unnormalized sequence likelihoods define a finite-candidate
Bayesian model; length normalization instead gives a tempered score, not a
calibrated natural-language source posterior. A uniform score has nonzero
classification loss but may still be stationary in a parametrization that
cannot distinguish sources. A full B-by-B pairing scores2B^2 sequences across
two views; batching means this is not necessarily2B^2 separate API/model calls.

No source, dataset, score, checkpoint, production recipe or remote process was
changed. Historical candidate cards below remain available for provenance.

## Historical design screen — source-conditioned consistency, 2026-11-14

Status: DESIGN SCREEN ONLY; NOT ADMITTED TO GPU GATES OR TRAINING.
Author requested a stronger contribution before training. The reference-only
RO/DW/RW decision below still applies. No production code, recipe, checkpoint,
dataset, comparator score or remote process was changed by this screen.

### Evidence boundary and research question

Ordinary R-Drop W(beta1.25) has mean validation BLEU18.2854 versus
baseline16.8753 over seeds42–44. W exceeds B(beta2.5) by0.3475 mean BLEU,
positive2/3 seeds. This motivates investigating consistency, but does NOT
establish that vocabulary KL suppresses valid translations or that source
neglect causes the remaining errors. Source-response probes already exist;
do not repeat them or claim their wrong-source gap proves grounding quality.

Candidate RQ: does preserving source-conditioned lexical evidence across
dropout views outperform preserving every target-token probability, beyond
merely weakening regularization? This RQ is a hypothesis, not a finding.

### Candidate formula (not an implementation specification)

Fit and freeze t(s|v) on TRAIN ONLY, where s is a source subword type and v a
target subword. For sentence x, use each distinct source type once. Define
A_x(s,v)=t(s|v) for types present in x, and
A_x(NULL,v)=1-sum_{s present in x}t(s|v). Unseen target types map to NULL.
Special output tokens have separate singleton bins and no lexical entries.
All columns are nonnegative and sum to1; no top-k renormalization or discarded
tail mass. How to estimate a reliable lexical table is UNRESOLVED; the formula
does not imply word/subword alignment has been solved on this small corpus.

For independently sampled dropout views at the SAME teacher-forced prefix:

    q_m = A_x p_m
    SKL(a,b) = [KL(a||b)+KL(b||a)]/2
    L = [CE(p_1,y)+CE(p_2,y)]/2 + beta * SKL(q_1,q_2)

No hidden-state alignment, rewiring, extra LM or inference-time change.
This is a proposed replacement regularizer, NOT a new version of CLRR/LSR.
Beta, token masks, estimation settings and train admission are NOT locked.

### Primary-literature comparison before any implementation

| Primary source | Already established | Difference remaining in this draft |
|---|---|---|
| Li et al., NAACL2019, Understanding and Improving Hidden Representations, section3.2, Figure3, Eq7 | Vocabulary probability aggregation followed by KL consistency in NMT | Their global Brown hierarchy and adjacent-layer classifiers differ from our sentence-source lexical map and same-output dropout views |
| Bhardwaj et al., COLING2022, KNOT, sections3–4, Eq1 | Semantic relations between output classes can enter a distribution-matching loss through OT cost | This draft uses a fixed stochastic pushforward and KL, not Sinkhorn OT or multi-teacher federated distillation |
| Chen et al., ACL2022, Masked Label Smoothing | Shared multilingual vocabulary makes indiscriminate output smoothing problematic; source-exclusive words can be masked | Their soft-label support mask is not source-sentence projection of two predictions |
| Gao et al., COLING2020, Unifying Input and Output Smoothing | Smoothing priors, strengths and token selection are existing NMT design dimensions | A different smoothing prior alone would not establish this draft's contribution |

Primary URLs read during this screen:
- https://aclanthology.org/N19-1046.pdf
- https://aclanthology.org/2022.coling-1.425.pdf
- https://aclanthology.org/2022.acl-short.74.pdf
- https://aclanthology.org/2020.coling-main.386/

Do not claim exact equivalence to SHR or KNOT. Equally, do not claim that
projection plus KL is new, or that a source-conditioned map alone clears
standalone novelty. NotebookLM source-backed review receipts are recorded
below when completed; its opinion cannot certify global novelty.

### Analytic rejection risks, checked before spending GPU time

1. Stochastic projection contracts KL by data processing. A smaller loss may
   reflect lost distinctions, not better translation evidence.
2. Exact counterexample evaluated in conda clrr (no model/data involved):
   p=(.6,.1,.3), r=(.1,.6,.3), group first2 coordinates. Full SKL=.8958797346;
   both projected distributions=(.7,.3), so projected SKL=0. This demonstrates
   an unobserved distinction, NOT that every such grouping harms translation.
3. Large NULL mass or similar lexical columns can make consistency vacuous.
   A source-insensitive predictor can also satisfy consistency; the loss
   provides no general grounding guarantee. CE does not remove this objection
   to a claimed mechanism.
4. Sharing a lexical explanation does not imply semantic equivalence:
   antonyms, tense and named-entity alternatives may be merged.
5. Permuting only ROW LABELS of A preserves KL exactly; it is NOT a valid
   random-map control. A useful control must disrupt target-column/source
   associations while preserving relevant sparsity/mass statistics.

### Admission decision and conditional evidence needed

HOLD / NO-GO FOR LONG TRAINING in this form. The proposal has identifiable
design differences, but neither a sufficiently supported mechanism nor a
demonstrated contribution beyond old projected consistency. Do not repair
this by stacking more losses or launching a coefficient sweep.

If this direction is reconsidered, first resolve lexical-table estimation,
coverage/NULL mass and whether genuinely different source associations survive
the map. These are train-only/data and frozen-checkpoint checks, not long runs.
Only after a stronger rationale is accepted would a pilot need: proposed map,
association-destroyed matched map, and ordinary full-KL with strength calibrated
on TRAIN gradients (calibration is not equivalent to matching Adam updates).
Existing W is the anchor; no blanket rerun of baseline/Middle/LoRA/BitFit.
No pilot budget or queue is authorized by this conditional list.

### Completed NotebookLM review and independent corrections

Notebook bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890:
- Query6de8746e370e completed, using12 sources including R-Drop, DKD, KNOT,
  MLS and smoothing. It recommends no training as formulated.
- Queryca9fae5cb023 completed using exactly the newly added SHR, KNOT and MLS
  primary sources: c8f2def7-0267-45fd-a221-73c594485862,
  6d2429bc-a4e4-4a47-8c57-b14c2702c5f8,
  d15d7a75-1704-440c-8d17-10d8d794796b. It explicitly cites SHR Eq7 and
  distinguishes sentence-source mapping from global hierarchy/cross-layer
  heads. It recommends against costly training now.

Independent corrections to its replies (do not adopt these assertions):
- KL value contraction does NOT alone establish a parameter-gradient norm
  bound or a scalar-equivalent regularizer. Gradient direction also changes.
- Output loss CAN backpropagate through cross-attention into the encoder;
  absence of a direct alignment target is not absence of encoder gradients.
- NULL dominance is a possible failure, not a measured property here.
- Shared lexical bins need not make opposite words exactly identical unless
  the relevant columns match; the numerical example is a constructed case.
- Neither a convex training Pareto frontier nor definite reviewer rejection
  follows from the cited evidence. Attention correlation is not by itself a
  causal grounding certificate. No theorem or specific metric is mandatory
  simply because NotebookLM said so.
- Its suggestion of10k–20k training steps on another benchmark is NOT admitted;
  it ignores the requested pre-training screen and existing Amis controls.

Final author-facing decision: one additional candidate screened on paper;
NO new model result and NO new training. Do not present this as the accepted
next method or as evidence that the overall research direction cannot work.

## Previous decision — increment reuse prototype, 2026-11-14

Status: AUTHOR CHOSE STRONGER-DESIGN-FIRST. RO/DW/RW NOT ADMITTED TO TRAINING.
CPU prototype retained for reference; no GPU gate or long training launched.
Standalone method novelty NOT cleared. This supersedes the earlier DESIGN ONLY
status. Original CLRR/LSR, W, baseline and A100 results are unchanged.

### Evidence and optimization target

Strongest reused RTX development anchor is ordinary R-Drop W, beta1.25:
mean validation BLEU18.2854 versus baseline16.8753, delta+1.4102 over seeds42–44,
positive3/3. It is an existing method, not our new contribution or an independent
confirmation result. CLRR-only has not shown a stable benefit. Neither global
collapse nor harmful residual duplication has been established as its cause.
Any new reuse mechanism must improve W, not merely improve plain baseline.

### Actual prototype

Code: src/methods/increment_reuse/v1/model.py (separate from original methods).
It is deliberately NOT registered in the production experiment runner.
For pre-LN mBART, u is the receiving layer state after attention and G includes
its native FFN pre-normalization/dropout. Donor d=output_k-input_k remains
differentiable. Reuse excludes padding and every tokenizer special token.

| Variant | Output | Question |
|---|---|---|
| RO | u+G(u+alpha*d) | Can FFN reading an early increment help? |
| DW | u+G(u)+alpha*d | Is direct residual writing sufficient? |
| RW | u+alpha*d+G(u+alpha*d) | Does combining read and write help? |

One scalar alpha starts at0; no extra FFN pass. Both dropout views use the
same variant and exact existing W loss: meanCE+1.25 symmetricKL, valid-token
mean per microbatch. There is no LSR or changed LR. The donor/receiver edge
is NOT selected; tiny-model tests use indices0→2 only as a correctness fixture.
Do not mistake fixture indices for a frozen12-layer pilot recipe.

### Literature decision before GPU

Primary sources read directly and added to NotebookLM:

- Hyper-Connections, ICLR2025, section2 Eqs2–5 and AppendixI/J:
  https://arxiv.org/pdf/2409.19606
- MUDDFormer, ICML2025, section2.3 Eqs7–8:
  https://arxiv.org/pdf/2502.12170
- Earlier reviewed AACL2026 inter-layer coordination AppendixA Eqs5–6/8:
  https://aclanthology.org/2026.findings-aacl.149.pdf
- Dynamic Layer Aggregation for NMT, Eqs3–9:
  https://arxiv.org/pdf/1902.05770

For conceptual state H=(u,H_k,H_(k-1)), HC read coefficients(1,alpha,-alpha),
carry preserving u, and branch write to the first stream realize local RO.
This demonstrates operator subsumption; it is NOT exact equivalence of our
whole pretrained model and the full learned multi-stream HC architecture.
MUDDFormer Eq7 separates attention Q/K/V/residual but its FFN still reads and
carries the same X_A. That implementation difference does not establish
standalone novelty. Finite-difference/Jacobian responses are not our invention.

NotebookLM query87efecb66e05 completed USING both primary sources
6d24f340-8cbd-483a-b07f-ec8824da9176 and
779d4e9b-4928-4770-885e-d92610fd24e2; references cite HC Eq2 and MUDD Eq7.
Previous querya8ec68622f00 had no retrieved sources and was not relied on as
a source-backed verdict. NotebookLM is a review assistant, not a novelty
certificate. Correct its reply: alpha addsONE parameter, W is a comparator
not a performance ceiling, and non-significance doesNOT prove a null or
falsify all cross-layer reuse. Only one fixed design can be rejected by its
own sufficiently informative evidence.

Conclusion: valid mechanism ablation, insufficient standalone architectural
novelty. Author explicitly chose "find a different direction with a clearer
contribution before training" on2026-11-14. RO/DW/RW is reference code only.
Do not run its3-arm screen, GPU gates or replication. Its conditional budget
below is historical/inactive. No further permission question is needed about
this choice. A new design must have a supported RQ and primary-paper/NotebookLM
comparison before implementation, smoke tests and any long-training admission.

### Completed correctness checks

Conda clrr, random tiny mBART CPU, tests/test_increment_reuse.py:

1. alpha0: bit-exact W loss/backbone gradients under identical dropout RNG;
   scalar gradient nonzero so the added route can start learning.
2. alpha0.2: plain versus reentrant-checkpointed backbone and scalar gradients
   agree across32 accumulated microbatches, both dropout passes; caches and
   temporary hooks clear. Donor is explicit checkpoint input, not detached.
3. Explicit RO/DW/RW FFN formulas match; masked positions have zero direct
   donor gradient.
4. Nonzero-alpha checkpoint roundtrip preserves logits and generated IDs.

Receipt: outputs/experiments/increment_reuse_preflight/CPU_GATE.json.
These checks do not establish full-size GPU fitness or translation quality.
Optimizer resume/shared-runner integration, frozen recipe, resource admission,
GPU gates and HF upload/restore gates remain pending.

RTX read-only check this session succeeded after sandbox elevation:
QuadroRTX6000 total24576MiB, used257MiB, free23952MiB (nvidia-smi snapshot).
No remote code was uploaded, GPU allocated or process stopped. This is a
snapshot, not a reservation or permission to bypass the scientific gate.

### Historical conditional budget — INACTIVE after author choice

| Stage | Data/model | New training | Decision |
|---|---|---|---|
| Contribution choice | Literature/above evidence |0| CLOSED: author selected stronger design, no RO/DW/RW training |
| Technical integration | Tiny mBART, then RTX full-backbone gate |0 long runs| All correctness/resume/provenance gates PASS |
| Engineering screen, ONLY if chosen | Amis→Mandarin/mBART, seed42 |3: RO/DW/RW| LockONEedge before results; reuse W/baseline |
| Replication, ONLY if screen passes | Same, seeds43/44 |4: RO+best direct control per seed| Compare paired RO−W and RO−control |
| Transfer, after recipe lock | Ede→Vietnamese |Plan later| No Turkish; separate proxy/data lock |
| Final paper | A100 |Plan later| Resolve final comparator selection/precision matching |

Unchanged RTX recipe if admitted: FP32/SDPA/TF32off, B4GA32, LR5e-5,
ES10/max20, validationBLEUbeam1, configs/protocols/amis_drb_bleu.json.
No test during search. Proposed practical screen threshold remains RO−W≥0.30
validationBLEU, RO above both controls, chrF++ drop≤0.20. These are development
choices, not statistical significance. Across-seed uncertainty and paired
sentence bootstrap answer different questions; neither proves novelty.

Existing14 A100 ES10 comparator numbers remain available. They have BF16/
TF32on and chrF++ checkpoint selection, so are not fully matched to RTX BLEU
pilots. Do not silently relabel them or rerun everything without an audit.

---

## Previous decision — substantive novelty and comparator reuse, 2026-11-14

No new candidate admitted and no GPU/training launched. Conditional-dropout
pilot E/D/Wcal remains HOLD under the author clarification. Two further paper
designs (decoder-Fisher alignment; bilingual token-transport layer updates)
were screened out before implementation following primary-paper and NotebookLM
review. They do not yet establish a sufficiently supported contribution.

Existing14 A100 ES10 comparator runs stay unchanged and reusable as recorded
benchmark results: mBART baseline/Middle/narrow LoRA/BitFit/Strong LoRA A/B;
NLLB baseline/Middle/Strong LoRA A/B/BitFit; mT5 Middle; ByT5 baseline/Middle.
Local frozen-plan audit confirms the same data hashes as RTX. A100 BF16/TF32-on,
validation chrF++ selection and some microbatches/seeds differ from current
RTX FP32/SDPA/TF32-off, validation BLEU selection. This does not invalidate
recorded scores, but prevents calling them fully matched RTX controls.
Current BLEU selection rule is NOT changed. No blanket comparator rerun.
A close new conceptual prior may require one targeted comparator despite
retaining all existing Middle/LoRA/BitFit results. W remains a strong reused
RTX development comparator, not a novel contribution.

Audit, exact rejected formulas, primary links, NotebookLM corrections and
reuse conditions: `outputs/experiments/method_redesign_audit/REVIEW.md`.
Machine-readable14-run local inventory: `COMPARATOR_REUSE_AUDIT.json` there.
NotebookLM query dc27ac1819af completed. Older plans below are historical.

---

# Hai ứng viên thiết kế sau kiểm chứng continuation

Ngày: 27/10/2026. Phạm vi: tổng hợp thực nghiệm, công thức, prior và prototype correctness.
**Hiện hành (27/10/2026): tác giả đã duyệt triển khai screen4runs sau audit**, với control donor addition mới. Xem [runbook hiện hành](../operations/DEPTH_SCREEN_RTX.md). Queue6 nhánh cũ vẫn HOLD, cả6 CPU/GPU gates PASS nhưng0model train dài; receipt `outputs/experiments/amis_depth_difference_v2_rtx/NOVELTY_HOLD.json`. §14 là quyết định lịch sử trước khi duyệt screen mới; novelty vẫn chưa xác nhận. Các bảng thực nghiệm và insight cũ giữ nguyên.

## 1. Quyết định hiện tại

| Hướng | Quyết định | Lý do |
|---|---|---|
| U1: bảo vệ margin của những token baseline dự đoán đúng | NO-GO làm đóng góp phương pháp chính | Động cơ phù hợp với probe, nhưng quá gần selective distillation và top-1 ranking KD |
| U2: residual attention dùng đồng thời chênh lệch attention và value giữa hai độ sâu | Technical gates PASS; training HOLD; NO-GO cho claim primitive mới | Có công thức và phép phân biệt cụ thể, nhưng toán tử lõi là trường hợp có ràng buộc của differential attention; chưa đủ cơ sở nhận là phương pháp mới cho ACL |

**Chưa có ứng viên nào vượt cả gate bằng chứng và gate novelty.** U2 là giả thuyết nghiên cứu đã được duyệt pilot để kiểm chứng, chưa phải phương pháp được chứng minh tốt hơn. Không đổi tên một biến thể đã có rồi xem đó là novelty.

## 2. Các hướng đã thử và điều không nên lặp lại

Các hàng thuộc những cohort/protocol khác nhau; không lấy hiệu số giữa hai hàng khác cohort làm gain của phương pháp.

| Hướng | Bằng chứng đã có | Quyết định thiết kế |
|---|---|---|
| Original CLRR + sentence LSR | Amis, 8 fresh seeds: Full−LSR BLEU −0.019809, thắng 3/8; Full−baseline −0.041468, thắng 4/8 | Không mở rộng nguyên recipe để tìm lại kết quả thuận lợi |
| Precache / centered alignment | Có validation pilot thuận lợi, chưa tương đương confirmation nhiều seed | Không trình bày như bằng chứng phương pháp đã ổn định |
| Token-LSR và Token-Full | RTX seed42 validation BLEU: baseline 17.6823; Full 17.1156; Token-LSR 17.5811; Token-Full 16.7533 | Alignment chi tiết hơn chưa tạo gain dịch trong pilot |
| Contrastive residual effect / P3 | Mean gain BLEU so baseline khoảng +0.2114 trên 3 seeds, thắng 2/3; không có kết quả vượt Holm trong family 90 phép đã báo cáo | Không đổi tên effect alignment thành ý tưởng mới; chưa mở rộng hàng loạt |
| Bỏ route theo nhóm layer | Can thiệp ở weights đã co-adapt chưa hỗ trợ một lựa chọn bỏ layer có lợi ổn định | Không suy từ inference ablation sang cam kết gain khi train lại |
| Generic learned route gate | Đã audit ReZero, Lexical Shortcuts và Value Residual Learning | Có thể là control; chưa đủ đóng góp mới |
| Selective KL / teacher protection | Đã xem xét trong thiết kế trước, có prior distillation sát | Không dùng việc đổi loss nhỏ làm cơ sở novelty độc lập |
| Sequence-risk continuation S0–S3 | Cả bốn best checkpoint thấp hơn initialization R1, kể cả CE thuần; CI theo câu chứa 0 | Dừng recipe continuation này; chưa quy nguyên nhân riêng cho risk hoặc CLRR |

Nguồn nội bộ: [thiết kế trước](METHOD_UPGRADE_DESIGN.md), [token audit](TOKEN_PILOT_AUDIT.md), [P3 review](P3_NOVELTY_AND_MECHANISM_REVIEW.md), [continuation audit](CONTINUATION_REVIEW_AND_NEXT_DESIGN.md).

## 3. Bằng chứng nào dẫn đến thiết kế mới?

| Quan sát | Giả thuyết hợp lý để xét | Chưa được kết luận |
|---|---|---|
| Full có content NLL tốt hơn baseline: 3.427548 so với 3.474846; top-1 lại thấp hơn: 49.6360% so với 50.4550% | Cần phân biệt cải thiện phân phối xác suất với cải thiện quyết định token | NLL tốt là nguyên nhân làm BLEU xấu |
| Full sửa 227 token baseline sai nhưng làm mất 272 token baseline đúng | Xem xét bảo vệ quyết định đã tốt, thay vì chỉ căn chỉnh hidden states | Teacher-forced token probe chứng minh lỗi khi autoregressive decoding |
| Route ON hữu ích hơn OFF với source thật ở weights đã co-adapt | Nghiên cứu cách decoder sử dụng thông tin qua độ sâu | Encoder đang bỏ qua source hoặc target fluency là toàn bộ nguyên nhân |
| CE continuation cũng giảm so với R1 đã chọn | Pilot mới cần tránh lẫn hiệu ứng continuation và hiệu ứng cơ chế | Đã chứng minh reset optimizer hay overfitting là nguyên nhân |

Nguồn: [token/decoding diagnosis](TOKEN_DECODING_DIAGNOSIS.md), [source grounding](SOURCE_GROUNDING_VALIDATION_REPORT.md), [initialization audit](CONTINUATION_REVIEW_AND_NEXT_DESIGN.md).

## 4. U1 — Bảo vệ margin của quyết định baseline

### 4.1 Công thức cụ thể

Teacher q là baseline đã train, đóng băng. Student p là candidate. Chỉ dùng token nội dung và nhãn **training**, không dùng validation/test để tạo nhãn bảo vệ.

Đặt C = {t: argmax q(t) = y(t)} và margin logit:

\[
m_p(t)=z_p(t,y_t)-\max_{v\ne y_t}z_p(t,v).
\]

\[
L_{protect}=\frac{1}{|C|}\sum_{t\in C}
\max\{0,\operatorname{sg}(m_q(t))-m_p(t)\},\qquad
L=L_{CE}+\lambda L_{protect}.
\]

Nếu C rỗng thì auxiliary loss bằng 0. Đây là margin preservation, không phải bảo đảm không quên ở mọi câu. Chưa chọn lambda vì hướng này chưa qua novelty gate.

### 4.2 Gradient, chi phí và rủi ro

- Teacher và phép chọn C không nhận gradient. Student nhận subgradient qua gold logit và đối thủ có logit lớn nhất khi hinge hoạt động.
- Cần teacher forward hoặc cache logits/margins training có provenance; inference chỉ cần student.
- Giữ margin của baseline có thể hạn chế student sửa phân phối theo hướng tốt hơn; teacher có thể đúng do tình cờ. Không có bảo đảm BLEU tăng.
- CE vốn đã khuyến khích gold top-1; phải chứng minh lợi ích ngoài việc đổi trọng số những token này.

### 4.3 Novelty verdict

[TIE-KD, ACL 2023](https://aclanthology.org/2023.acl-long.448/) đã đặt trọng tâm vào top-1 teacher và dùng hierarchical ranking loss. [Selective Knowledge Distillation, ACL 2021](https://aclanthology.org/2021.acl-long.504/) đã nghiên cứu chọn phần dữ liệu để distill. Công thức U1 không được khẳng định trùng hoàn toàn các paper này, nhưng đóng góp hiện mô tả quá gần những nguyên lý đó.

**Loại U1 khỏi vị trí phương pháp chính.** Không mở training chỉ để xem có gain rồi mới xử lý novelty.

## 5. U2 — Residual của hai chênh lệch tại decoder

### 5.1 Ý tưởng và phạm vi thay đổi

Thay vì cộng nguyên donor hidden state vào encoder, decoder được đọc một phần hiệu chỉnh chỉ tồn tại khi **cả cách phân bố attention lẫn values đều khác giữa hai độ sâu encoder**.

Đây là method family riêng: primary candidate dùng CE, không chèn CLRR gốc, LSR hoặc risk. Không gọi nó là Full cũ. Việc bỏ các auxiliary objectives ở candidate nhằm kiểm tra riêng cơ chế, không xóa lịch sử hoặc khẳng định chúng vô ích.

Định nghĩa ban đầu cho mBART-50: donor sau encoder layer 6, recipient là encoder output cuối mà cross-attention gốc sử dụng; chỉ sửa cross-attention decoder layer 12. Layer được đếm từ 1. Layer 6 là midpoint chọn trước để giới hạn thiết kế, **chưa được dữ liệu chứng minh tối ưu**. Không tự đổi layer sau khi thấy điểm.

### 5.2 Công thức từng head

Dùng đúng các projection Q/K/V hiện có của cross-attention decoder cuối. H_d là donor; H_L là input K/V của cross-attention gốc:

\[
Q=D W_Q+b_Q,\quad
K_d=\operatorname{sg}(H_d)W_K+b_K,\quad
V_d=\operatorname{sg}(H_d)W_V+b_V,
\]
\[
K_L=H_LW_K+b_K,\quad V_L=H_LW_V+b_V,
\]
\[
A_d=\operatorname{softmax}(QK_d^\top/\sqrt{d_h}+M),\quad
A_L=\operatorname{softmax}(QK_L^\top/\sqrt{d_h}+M),
\]
\[
\Delta C=(A_d-A_L)(V_d-V_L),\qquad
C'=\operatorname{Dropout}(A_L)V_L+g_h\Delta C,\quad g_h(0)=0.
\]

M là cùng source padding mask; query giữ nguyên causal decoder. Concatenate các head rồi dùng **một lần** output projection và bias gốc. Attention trong correction dùng xác suất trước dropout; nhánh baseline giữ native attention dropout. Không thêm dropout/RNG draw vào correction trong định nghĩa này.

Với checkpoint mBART có 16 heads, thêm 16 scalar g_h, không thêm ma trận projection. **Không còn parameter-neutral hoặc compute-neutral.** Zero initialization là kỹ thuật đã có prior, không phải claim novelty.

### 5.3 Tính chất kiểm được và rủi ro

- Correction bằng 0 nếu A_d = A_L **hoặc** V_d = V_L. Chênh lệch values đơn thuần hoặc attention đơn thuần không đủ kích hoạt nó.
- Khai triển: A_d V_d − A_d V_L − A_L V_d + A_L V_L. Đây là tương phản đại số bốn ô, không phải bằng chứng nhân quả hay chứng minh triệt nhiễu.
- Khi g = 0, hàm forward phải khớp baseline. Gate có thể nhận gradient CE ngay nếu correction có tín hiệu; không bảo đảm gradient khác 0 trên mọi batch.
- Stop-gradient chỉ đặt trên donor hidden ở nhánh phụ. Shared W_K/W_V vẫn nhận gradient; đường encoder tuần tự qua H_L vẫn truyền về layer donor. Đây chưa phải kết luận detach tốt hơn attach.
- Khi H_d và H_L gần nhau, correction có thể nhỏ bậc hai theo khoảng cách representation. Có nguy cơ tín hiệu quá yếu; không tự tăng scale để tìm gain.
- Đổi chỗ hai độ sâu đổi dấu cả hai hiệu nên tích giữ nguyên. Không được giải thích công thức là một hướng vận chuyển thông tin duy nhất.
- Thêm K/V donor cache cho một decoder layer, attention computation và hidden cache; phải đo latency/VRAM. Donor cùng token positions nhưng khác chuẩn hóa/phân phối qua độ sâu là rủi ro thực tế.
- Không dùng target encoder pass hoặc gold để gate lúc inference. Phải giữ incremental generation, beam cache reorder và mask chính xác.

### 5.4 Prior sát nhất: khác gì, chưa mới ở đâu?

| Prior | Phần đã có | Đối chiếu U2 |
|---|---|---|
| [Transparent Attention, EMNLP 2018](https://aclanthology.org/D18-1338.pdf), §2.4 | Decoder truy cập kết hợp nhiều lớp encoder | Đọc lower-layer states không mới; U2 chỉ khác ở residual tương phản thay weighted mixture |
| [Differential Transformer](https://arxiv.org/html/2410.05258v2), §2.1 | Toán tử (softmax(Q1K1ᵀ) − λ softmax(Q2K2ᵀ))V | Đặt hai query bằng nhau, hai keys từ hai độ sâu, λ = 1 và V = V_d − V_L sẽ biểu diễn correction U2. Vì vậy U2 không phải attention family mới ngoài toán tử này |
| [Value Residual Learning, ACL 2025](https://aclanthology.org/2025.acl-long.1375.pdf) | Residual/cross-layer information ở values và các đối chứng mixing | Chênh lệch value qua layer không đủ mới; phải xét kết hợp và ràng buộc cụ thể |
| [DEX, preprint](https://arxiv.org/html/2505.16333v1), §3.1 | Điều chỉnh pretrained attention output bằng nhánh trừ learnable | U2 lấy hai attention maps theo depth thay output projection contrast; sự khác vị trí này chưa tự chứng minh contribution đủ mạnh |
| [Maracani et al., CVPR 2025](https://arxiv.org/html/2503.16184v1), §4.3 Eq.(11) | Differential cross-attention trong decoder nhận diện văn bản | Chuyển differential attention sang cross-attention/decoder đã có prior. U2 khác ở ràng buộc hai độ sâu và value-difference, không được nhận novelty chỉ từ vị trí decoder |
| [Differential Transformer V2, báo cáo tác giả 2026](https://huggingface.co/blog/microsoft/diff-attn-v2) | Hai query heads dùng chung K/V; coefficient theo token/head | Learned gate và thay cấu trúc chia sẻ projection cũng đã được khai thác; U2 không được claim token/head gating là ý tưởng mới |

**Verdict: novelty HOLD.** Phần có thể nghiên cứu là ràng buộc dùng chung projections theo hai độ sâu và hai điều kiện triệt tiêu. Chưa tìm thấy công thức này trong những nguồn đã đọc không chứng minh không tồn tại prior. Chưa được claim ACL-level novelty, denoising, source grounding tốt hơn hay lexical preservation.

## 6. Những việc tiếp theo nếu tiếp tục U2

### 6.1 Trước hết: chọn đúng tiêu chuẩn đóng góp

Chỉ đáng mở prototype nếu chấp nhận câu hỏi nghiên cứu cụ thể: **ràng buộc đồng thời attention-difference và value-difference có giúp hơn các thành phần đơn lẻ khi fine-tune NMT không?** Nếu mục tiêu bắt buộc là một toán tử attention chưa có tiền lệ, U2 không đáp ứng; cần thiết kế khác, không mở pilot này.

### 6.2 Correctness gate trước mọi training

1. Chứng minh/test số: khai triển bốn ô, hai điều kiện correction bằng 0, gates 0 khớp logits và loss baseline trong eval/train với cùng RNG.
2. Finite-difference gradients cho gates và shared projections; xác minh stop-gradient đúng phạm vi và không chặn đường tuần tự.
3. Source padding, shift-right, causal query, precision và checkpoint recomputation không làm lẫn cache; không dùng thông tin target tương lai.
4. Full-sequence và incremental decoding khớp; beam cache reorder đúng; save/reload không mất donor/gates/recipe.
5. Xác nhận số parameters, compute, memory; ghi step-zero. Tiny GPU gate chỉ sau CPU gate, kiểm tra VRAM trước.

Các mục trên là checklist thiết kế; phạm vi đã thực hiện và giới hạn PASS cụ thể ở §9.

### 6.3 Pilot có điều kiện: 5 model mới seed42 khi dùng lại được baseline

Cập nhật sau audit đại số tại §8: thêm control **additive-only**, nên kế hoạch trước gồm 4 model mới là chưa đủ để tách riêng phần tương tác. Chưa mở các runs này.

| Arm | Correction trước scalar gate | Chứng minh gì? |
|---|---|---|
| Baseline | 0 | Mốc cùng recipe; ưu tiên dùng lại checkpoint hợp lệ |
| Ordinary depth difference | A_d V_d − A_L V_L | Kiểm tra chỉ đọc thêm một depth đã đủ chưa |
| Attention-only | (A_d − A_L)V_L | Phần khác biệt attention có đủ không? |
| Value-only | A_L(V_d − V_L) | Phần khác biệt values có đủ không? |
| Additive-only | (A_d − A_L)V_L + A_L(V_d − V_L) | Hai thành phần đơn lẻ cộng lại có đủ chưa, khi bỏ tích tương tác? |
| U2 | (A_d − A_L)(V_d − V_L) | Kết hợp có thêm lợi ích ngoài từng thành phần không? |

Năm arm mới dùng cùng donor, vị trí, shared projections, số gates, initialization và quy tắc dropout. Ordinary-depth correction là adapted control cụ thể, không được gọi là tái lập đầy đủ Transparent Attention. Attention-only là control gần differential attention với ràng buộc depth; không thay cho toàn bộ benchmark của paper đó. Số tham số bằng nhau không có nghĩa FLOPs/latency bằng nhau; phải báo compute thực tế.

Tất cả bắt đầu từ **pretrained gốc**, không continuation từ R1 đã hội tụ. ES10, max20 epochs và các cấu hình còn lại phải khóa theo recipe baseline phù hợp. Chỉ dùng lại baseline nếu data/hash, precision, batch/accumulation, optimizer/schedule, seed, library và checkpoint-selection protocol tương thích. Nếu không tương thích: nêu rõ khác biệt và xin chốt thêm baseline; không âm thầm thay số cũ. Số model mới là 5 nếu dùng lại baseline, 6 nếu phải mở baseline matched. Đây là ngân sách thiết kế có điều kiện, không phải ngân sách đã được duyệt chạy.

Đề nghị development endpoint trước khi chạy: validation chrF++ beam1; U2 phải hơn baseline và control tốt nhất ít nhất +0.30 chrF++, BLEU không giảm so với cả hai. Đây là ngưỡng quyết định phát triển đề xuất, chưa phải statistical confirmation. Không dùng test để chọn layer/gate/arm; Amis validation đã được xem nhiều nên không phải holdout sạch.

Nếu correctness fail: sửa implementation, không diễn giải điểm. Nếu quality gate fail: dừng U2, không sweep layer/strength liên tục. Nếu pass: khóa cấu hình rồi mới lập confirmation nhiều seeds và Ê Đê; chưa đủ kết luận từ seed42. Gain không tự giải quyết novelty; đóng góp phải thể hiện vì sao ràng buộc này cần thiết so với prior/control.

## 7. Kết luận để quyết định

1. Thực nghiệm hiện có loại được nhiều giải thích đơn giản: tăng alignment, thêm contrastive loss, tiếp tục train hay thêm scalar gate chưa đủ cứu phương pháp một cách ổn định.
2. U1 có động cơ thực nghiệm nhưng thiếu khoảng cách với prior, nên loại làm đóng góp chính.
3. U2 có câu hỏi cơ chế kiểm được và khác CLRR encoder cũ, nhưng vẫn nằm trong differential-attention design space. Chỉ ưu tiên nghiên cứu có điều kiện; **chưa có phương pháp mới đủ bằng chứng/novelty để mở training ngay**.

Không sửa bảng paper hoặc insight cũ theo giả thuyết trong tài liệu này. Các phép kiểm tại §8 dùng tensor tổng hợp trên CPU, không phải kết quả dịch. Prototype code và GPU preflight được thực hiện sau đó, báo cáo tại §9; chưa có training runs.

## 8. Audit tiếp theo — prior trực tiếp và toán tử CPU

### 8.1 Kết quả mới có ý nghĩa gì?

Đã đọc bổ sung §4.3 của [Maracani et al.](https://arxiv.org/html/2503.16184v1): differential cross-attention dùng hai query/key streams trong decoder đã có. U2 không trùng toàn bộ architecture đó, nhưng claim “đưa differential attention vào decoder” phải loại. Báo cáo [DIFF V2 của tác giả](https://huggingface.co/blog/microsoft/diff-attn-v2) cũng cho thấy thay gating và sharing scheme là prior cần đối chiếu.

Không có cơ sở đảm bảo novelty tuyệt đối từ tìm kiếm có phạm vi. Ngược lại, việc biểu diễn U2 bằng một attention primitive đã có cũng không tự chứng minh mọi inductive constraint của U2 đều không có giá trị nghiên cứu. Nội dung phải chứng minh, nếu tiếp tục, là **tác dụng của ràng buộc hai chênh lệch qua depth**, không phải phát minh phép trừ softmax.

### 8.2 Kiểm tra toán học thực hiện

Chạy trong conda `clrr` local, Torch 2.6.0, float64 CPU, seed42; Q có shape [2,3], K/V [4,3]. Dùng tensor ngẫu nhiên tổng hợp, không load model/data/weights. Artifact: [u2_operator_audit.json](../../outputs_rebuttal/method_design/u2_operator_audit.json).

**11/11 phép kiểm đại số và gradient PASS**; gate derivative ở ví dụ tổng hợp khác 0. Đây chưa phải model correctness gate PASS.

Đặt ΔA = A_d − A_L, ΔV = V_d − V_L. Phát hiện quan trọng cho thiết kế đối chứng:

\[
A_dV_d-A_LV_L
=\underbrace{\Delta A V_L}_{attention\ only}
+\underbrace{A_L\Delta V}_{value\ only}
+\underbrace{\Delta A\Delta V}_{mixed}.
\]

Do đó ordinary-depth control đã chứa mixed term. Cần additive-only control ΔA V_L + A_L ΔV để biết điều gì xảy ra khi bỏ riêng phần tương tác khỏi ordinary-depth correction. So U2 với hai thành phần riêng lẻ là chưa đủ.

| Kiểm tra | Kết quả | Giới hạn diễn giải |
|---|---|---|
| Khai triển bốn ô và decomposition trên | Sai số tối đa khoảng 2.3e−16 | Chỉ xác nhận đại số |
| U2 tương đương differential operator với value = ΔV, λ = 1 | Sai số khoảng 2e−16 | Không nhận là attention primitive mới |
| Gates 0 khớp baseline context | Khớp chính xác trên tensor tổng hợp | Chưa kiểm logits/dropout/cache của mBART |
| Gate derivative | 0.6333371907 trong ví dụ đã khóa | Không bảo đảm mọi batch có gradient hữu ích |
| Autograd finite-difference gradcheck | PASS cho Q, hai K và hai V | Chưa kiểm detach, checkpoint recomputation hoặc generation integration |
| Hai attention bằng nhau / hai values bằng nhau | Correction = 0 | Đúng hai điều kiện thiết kế |
| Chênh lệch values là constant trên source positions | Correction xấp xỉ 0 | U2 loại cả thông tin thay đổi toàn cục loại này; chưa biết có lợi hay hại |
| Chỉ một source position hợp lệ | Correction = 0 | Đây là một trường hợp nhánh phụ hoàn toàn không tác dụng |
| Depth differences ε và ε/2 | Norm ratio 3.99995, tiến tới 4 | Phù hợp tín hiệu bậc hai; không chứng minh tín hiệu trên checkpoint thật nhỏ |

### 8.3 Chốt bước kế tiếp và điểm dừng

- **GO cho prototype cơ chế có giới hạn**, nếu tiếp tục theo U2: chỉ module toán tử riêng + integration gates ở §6.2; chưa gắn tên phương pháp chính hoặc đổi baseline.
- **HOLD training và HOLD novelty claim**: không dùng các CPU PASS để bỏ qua thiếu bằng chứng về độ hữu ích qua depth hoặc để gọi U2 là đã novel.
- Prototype phải báo norm correction, gate gradients và chi phí ở pretrained/matched batch trước khi xin mở pilot. Dùng một donor/vị trí đã định, không quét layer.
- Nếu correction không hoạt động đúng hoặc không có tín hiệu ở phép đo đã khóa, dừng và báo; không tự thêm normalization/scale/loss để cứu candidate.
- Nếu prototype qua gates, trình lại ngân sách đã sửa: 5 runs mới (hoặc 6 nếu baseline cũ không matched). Chưa có lệnh train hoặc queue nào được mở.

Đây là quyết định tiếp tục kiểm chứng một inductive constraint có prior liên quan, không phải chứng nhận phương pháp đủ mới hoặc đủ mạnh cho ACL.

## 9. Prototype đã triển khai và preflight đã chạy

### 9.1 Phạm vi code

- Method riêng: `src/methods/depth_difference/v1/model.py`, class `DepthDifferencePrototype`.
- Không đăng ký vào `methods.registry`, không sửa trainer hoặc công thức các method cũ.
- Donor được truyền cùng memory trong mỗi forward; không giữ donor mutable trên module. Ordinary cross-attention layers chỉ đọc recipient; layer cuối cache cả hai K/V theo feature dimension để native beam reorder xử lý cùng nhau.
- Source/recipient output trả ra ngoài giữ hidden dimension gốc. Precomputed memory phải được tạo bởi `prototype.encode()`.
- Format riêng `prototype.safetensors` + config/recipe; loader riêng strict. Không tạo tên weights chuẩn để generic HF loader vô tình bỏ mất gates.
- Script dùng args: `scripts/analysis/preflight_depth_difference.py`. Đã đưa đúng code/gates vào project RTX `/workspace/ura/thanhdat/projects/CLRR/`.

### 9.2 Các gate đã PASS

CPU: `conda run -n clrr python tests/test_depth_difference.py`, **9/9 tests PASS** trên mBART nhỏ random. Bao gồm:

1. Gate0 khớp baseline loss/logits, RNG và gradients của backbone; đúng số tham số thêm.
2. Công thức attention, shared projection gradients, donor không nhận gradient bổ sung; head mask.
3. Checkpoint recomputation khớp loss và toàn bộ gradients cho cả reentrant và non-reentrant.
4. Full sequence khớp incremental decoding; K/V reorder khớp sau đổi thứ tự batch.
5. Greedy và beam4 khớp cache/no-cache; gate0 khớp generation baseline.
6. Source padding không ảnh hưởng output; thay target tương lai không ảnh hưởng logits trước đó.
7. Precomputed memory, nhiều forward khác shape/batch và output hidden states không lẫn donor.
8. Save/load strict giữ gates, logits, beam generation; native HF loader phải từ chối format này.
9. Các input ngoài phạm vi được kiểm tra phải bị từ chối rõ ràng.

Đây không phải kiểm chứng đầy đủ mọi input/backend. Chưa tích hợp Trainer/resume/distributed/AMP/static cache hoặc backbone khác.

GPU: dùng **pretrained mBART gốc đã cache**, không R1 fine-tuned, không download mới. Amis train.csv, hai hàng đầu, FP32/eval/dropout OFF/TF32 OFF, source proxy tl_XX → zh_CN, max_length256; source tensor [2,19], labels [2,12]. Encoder donor6, decoder12. Môi trường đúng prefix conda `clrr` trong project, Torch2.6.0+cu124, Transformers4.57.6.

| Phép đo | Kết quả | Ý nghĩa |
|---|---:|---|
| Parameters thêm | 16 | Một scalar/head, không parameter-neutral |
| Gate0 logits max absolute error so native | 8.7738e−5 | PASS `atol=1e−5, rtol=1e−5`; không tuyên bố bitwise equal trên GPU |
| Gate0 greedy so native | Khớp exact IDs | Trên batch này, max_new_tokens8 |
| Gate=.1 tạm thời, beam4 cached/no-cache | Khớp exact IDs | Đây là input kiểm tra cache, không phải strength được chọn hay model đã train |
| Norm gradient gate | 1.652568 | Có tín hiệu trên batch này, chưa chứng minh hướng hữu ích lâu dài |
| Norm gradient encoder layer1 fc1 | 6.253597 | Đường gradient tuần tự vẫn hoạt động |
| Peak allocated CUDA | 6,410.24 MiB (~6.26 GiB) | Chỉ preflight, không ước lượng bộ nhớ training đầy đủ |
| Optimizer steps / training runs | 0 / 0 | Không cập nhật weights |

Revision pretrained: `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`. Hash CSV training, worker và method code được lưu trong JSON. Nguồn artifact:

- `outputs_rebuttal/method_design/u2_tiny_cpu_preflight.json` (tiny random, không giải thích chất lượng).
- `outputs_rebuttal/method_design/u2_pretrained_gpu.json` (lần preflight đầu).
- `outputs_rebuttal/method_design/u2_pretrained_gpu_scale_audit.json` (lặp cùng batch, thêm instrumentation, công thức giữ nguyên).
- Trên RTX: `outputs/experiments/u2_preflight/`. Không tạo checkpoint hoặc upload weights mới.

### 9.3 Insight mới: nhánh có tín hiệu nhưng scale cần thận trọng

Trên đúng batch trên, trước output projection và trước nhân gate:

| Đại lượng | Norm |
|---|---:|
| Recipient hidden | 103.8964 |
| Donor hidden | 431.1977 |
| Recipient V | 206.6565 |
| Donor V | 878.7990 |
| Context baseline | 42.5665 |
| Correction U2 chưa nhân gate | 908.4974 |

Correction/base context norm = **21.3430**; từng head khoảng **3.58–98.12**. Donor hidden norm khoảng 4.15 lần recipient, donor V khoảng 4.25 lần recipient V. Trung bình L1 distance giữa hai attention distributions là 1.18389 (cùng source mask).

**Diễn giải:** rủi ro “correction quá nhỏ” ở tensor random không xuất hiện trên batch pretrained này. Có chênh lệch scale qua depth và correction lớn so với context gốc. Tuy nhiên denominator có thể nhỏ do weighted averaging/cancellation; các số này chưa chứng minh riêng scale donor gây ratio lớn hoặc sẽ gây bất ổn khi train. Mean/norm batch nhỏ có cả các query positions theo padding; không phải phân tích riêng rare tokens hay lỗi dịch.

Gate0 khiến phần thêm vào forward bằng0. Không suy “correction thô lớn21 lần” thành “model đang bị sửa21 lần”. Với gate=.1 đồng nhất, norm phần thêm vào bằng khoảng2.13 lần context trên batch này, nhưng .1 chỉ dùng thử cache; **không dùng nó làm giá trị khởi tạo hoặc giá trị train mặc định**. Gate học được có learning dynamics riêng; chưa đo optimizer trajectory.

**Quyết định:** prototype correctness PASS trong phạm vi trên; quality và novelty vẫn chưa PASS. Không tự thêm normalization, đổi layer hoặc áp strength CLRR cũ .1 để sửa scale, vì sẽ là biến thể khác cần định nghĩa và control riêng. Chưa mở 5/6 runs pilot.

### 9.4 Trạng thái máy sau preflight

Đã kết thúc tiến trình gate và trả GPU: nvidia-smi báo free23,542 MiB, used666 MiB, utilization0%. Không dừng máy chủ hoặc tiến trình người khác. Không đưa checkpoint trở lại ổ đĩa; chỉ thêm source và JSON nhỏ. Baseline/insight/experiment tables cũ giữ nguyên.

## 10. Frozen scale audit trên 64 câu training

Ngày 27/10/2026. Worker: `scripts/analysis/audit_depth_difference_scale.py`.
Artifact: `outputs_rebuttal/method_design/u2_scale_train64.json`; bản trên RTX: `outputs/experiments/u2_preflight/scale_train64.json`.

### 10.1 Protocol và phạm vi

- Lấy 64 hàng không hoàn lại bằng `random.Random(42).sample` từ Amis training CSV; lưu toàn bộ chỉ số hàng zero-based và SHA256 trong JSON. Không dùng validation/test.
- Cùng pretrained revision, donor6, decoder12, FP32/eval/eager/TF32 OFF như §9. Batch4, 16 batches, max_length256, tl_XX → zh_CN.
- Tất cả backbone parameters đóng băng; chỉ gate zero-init nhận backward CE để đo gradient, không optimizer, không cập nhật gate hoặc weights.
- Norm context/correction đo trước output projection và trước gate, chỉ trên **693 vị trí target có labels khác −100**. Norm hidden/value chỉ tính source có attention_mask=1. Các vị trí language/EOS được giữ theo labels; không gọi đây là phân tích riêng từ vựng.
- Với mỗi câu, hệ số chẩn đoán `c = ||H_top_valid||_F / ||H_donor_valid||_F`. Chiếu `c * H_donor` qua K hoặc V như bốn trường hợp bên dưới. Đây là can thiệp phân tích, không thay code phương pháp hoặc chọn cấu hình train.
- Các bias projection giữ nguyên. Vì thế hidden norm được cân chính xác nhưng projected V không buộc có norm bằng nhau.
- Pool tổng bình phương trên toàn bộ batch/vị trí/head rồi lấy căn; không lấy trung bình các tỷ lệ batch. Không coi 16 batch là 16 training seeds.

### 10.2 Kết quả

| Keys donor | Values donor | Norm correction | Norm context gốc | Correction/context | L1 attention trung bình | Donor V / top V |
|---|---|---:|---:|---:|---:|---:|
| Gốc | Gốc | 4935.9415 | 238.5596 | 20.6906 | 1.2688 | 4.4289 |
| Gốc | Cân scale | 1212.1338 | 238.5596 | 5.0811 | 1.2688 | 1.0792 |
| Cân scale | Gốc | 4536.8735 | 238.5596 | 19.0178 | 1.1390 | 4.4289 |
| Cân scale | Cân scale | 1126.8378 | 238.5596 | 4.7235 | 1.1390 | 1.0792 |

Hệ số c min/median/max = **0.22348 / 0.23915 / 0.26096**. Tỷ lệ correction/context theo head ở nhánh gốc khoảng **4.64–81.56**; cân cả K/V còn **0.69–19.59**. Context chuẩn có thể nhỏ do averaging/cancellation, nên lưu cả absolute norms để tránh diễn giải chỉ dựa vào denominator.

Gradient gate CE hữu hạn trên cả 16 batch; norm min/median/max = **0.50560 / 0.71196 / 0.89025**. Phần lớn head đổi dấu gradient giữa các batch; head index14 âm ở cả16 batch. Đây là tín hiệu đạo hàm cục bộ tại pretrained initialization, không dự đoán hướng cập nhật Adam hoặc gain dịch sau train.

### 10.3 Điều có thể kết luận và bước tiếp theo

1. Correction thô lớn không chỉ xuất hiện ở hai hàng đầu: trên mẫu training64 và sau khi bỏ padded queries, ratio vẫn khoảng20.69×.
2. Can thiệp scale values giảm rõ norm correction trên weights và query cố định; scale keys riêng có tác động nhỏ hơn trong phép đo này. Chưa phân tích toàn bộ training distribution hoặc trajectory.
3. Cân scale cả hai vẫn còn correction lớn ở một số heads. Không suy rằng chỉ normalization là đủ hoặc nhánh raw chắc chắn train hỏng. Gate0 vẫn khiến phần cộng vào forward bằng0.
4. **Chưa có bằng chứng BLEU/chrF++ tăng và novelty vẫn HOLD.** Không dùng scale audit làm kết quả chất lượng hoặc claim đóng góp mới.
5. Trước training dài: chốt candidate card chọn raw hay một biến thể kiểm soát scale được định nghĩa rõ; nếu chọn biến thể mới phải cập nhật công thức, controls và kiểm tra gradients/cache/save-load tương ứng. Sau đó mới tích hợp trainer và kiểm thử một vài optimizer steps, xác minh gates/updates hữu hạn và đo bộ nhớ theo đúng recipe. Không mở thêm grid layer/scale từ phép đo này.
6. Pilot phân biệt cơ chế vẫn cần các controls §8; nếu normalization được chọn, cùng quy tắc normalization phải áp cho các controls phù hợp. Không so nhánh mới được tinh chỉnh với controls có scale bất lợi rồi gán gain cho toán tử mixed.

### 10.4 Tái lập và tài nguyên

Trước chạy GPU free23,542 MiB; sau kết thúc cũng free23,542 MiB, used666 MiB, utilization0%. Dùng lock project và ngưỡng free16,000 MiB; không tác động tiến trình khác. Chỉ thêm worker và JSON nhỏ, không tải hoặc tạo checkpoint.

CPU kiểm tra reduction giữ đúng valid target và loại padded queries: PASS. Toàn bộ batch GPU có correction và gate gradient hữu hạn. Đây không thay thế bộ kiểm thử correctness của prototype.

Worker SHA256: `a46f4931ed24a9e1cac2f8cbfb7d8f8a1ab60539f7ca4140310767e06a3c931d`.
Method SHA256 không đổi: `ce4409dd9ac2184a7cf5ba760004138abd268a1cd19556e2d2e2ae7cff3b799a`.

Lệnh chạy từ project RTX, sau khi đặt các cache/temp environment variables theo runbook:

```bash
conda/envs/clrr/bin/python scripts/analysis/audit_depth_difference_scale.py \
  --data /workspace/ura/thanhdat/projects/CLRR/data_processed/amis_mandarin/train.csv \
  --cache-dir /workspace/ura/thanhdat/projects/CLRR/cache/huggingface/hub \
  --output /workspace/ura/thanhdat/projects/CLRR/outputs/experiments/u2_preflight/scale_train64.json \
  --rows 64 --batch-size 4 --seed 42
```

## 11. U2 v2: kiểm soát scale nhánh values và optimizer smoke

### 11.1 Candidate card được chốt cho bước kỹ thuật

Code riêng: `src/methods/depth_difference/v2/`, class `ValueScaledDepthDifference`.
Giữ `v1` làm biến thể raw. v1 chỉ thêm extension point cho attention class, donor-value input và method ID; 9 kiểm thử v1 vẫn PASS sau thay đổi. Không đổi công thức v1 hoặc các phương pháp cũ.

Với source mask m và donor layer6, recipient là output encoder cuối:

```text
c_b = stopgrad(sqrt(sum_valid_source(H_top^2)
                   / max(sum_valid_source(H_donor^2), 1e-12)))
K_d = k_proj(stopgrad(H_donor))                 # giữ keys gốc
V_d' = v_proj(c_b * stopgrad(H_donor))         # cân input values
delta = (A_d - A_top) @ (V_d' - V_top)
C_out = Dropout(A_top) @ V_top + gate_head * delta
gate_head(init) = 0
```

Scale tính FP32 theo từng câu, chỉ valid source, không centering/LayerNorm hoặc tham số học thêm. `1e-12` là guard cho norm bằng0, không phải hyperparameter đã tune. Scale detach cả hai phía: không tạo thêm gradient qua norm recipient. Q/K/V projections vẫn học chung. Chỉ decoder layer12 nhận correction; CE thuần, không CLRR/LSR cũ. Khi gate0, backbone vẫn giữ baseline forward trong tolerance đã kiểm thử; gate không bị khóa ở0 khi tối ưu.

**Lý do chọn:** §10 cho thấy scale values đóng góp lớn vào correction norm. Chọn thử một thay đổi có thể giải thích từ chẩn đoán. Không suy rằng đây là normalization tối ưu hoặc đóng góp novel; chưa chạy validation để chọn giữa các normalization.

### 11.2 Các đối chứng phải dùng cùng scale convention

Đặt `dA = A_d - A_top`, `dV = V_d' - V_top`:

| Nhánh pilot dự kiến | Correction trước gate | Câu hỏi |
|---|---|---|
| Baseline | 0 | Mốc chất lượng với cùng protocol |
| Ordinary depth difference | `A_d @ V_d' - A_top @ V_top` | U2 có hơn residual difference thông thường? |
| Attention-only | `dA @ V_top` | Có cần value difference? |
| Value-only | `A_top @ dV` | Có cần attention difference? |
| Additive-only | `dA @ V_top + A_top @ dV` | Mixed interaction có cần thiết? |
| U2 v2 mixed | `dA @ dV` | Kiểm chứng nhánh interaction |

Identity kiểm tra: ordinary = attention-only + value-only + mixed. Các nhánh dùng cùng shared projections, donor detach, layer, gate zero-init, scale convention và recipe train. Dùng cùng số gates cho các nhánh thêm correction. Không dùng normalization làm lợi thế riêng cho U2 rồi diễn giải gain thành tác dụng interaction.

Đây vẫn là **5 nhánh mới + baseline**. Chỉ tái dùng baseline nếu metadata/protocol khớp đầy đủ, nếu không cần6 runs. So raw-v1 với v2 sau training sẽ là run bổ sung nếu sau này muốn claim normalization; chưa có trong pilot và chưa được coi là phép ablation đã hoàn thành.

### 11.3 Kiểm thử đã thực hiện

`tests/test_depth_difference_v2.py`: **10/10 PASS** trên CPU, gồm công thức explicit, donor/scale detach, shared projection gradients, bỏ padded source khỏi scale, zero-donor hữu hạn, zero-gate baseline/RNG/gradients, checkpoint recomputation, incremental cache/reorder, beam generation, padding/causality, save–load và3 bước AdamW. Loader v1 từ chối checkpoint method ID v2.

GPU worker: `scripts/analysis/smoke_depth_difference_updates.py`.
Artifact: `outputs_rebuttal/method_design/u2_v2_optimizer_smoke.json` (chứa report và GPU-after). Trên RTX: `outputs/experiments/u2_preflight/v2_optimizer_smoke.json`.

| Cấu hình/kết quả | Giá trị |
|---|---|
| Pretrained | Cùng mBART revision §9; không dùng checkpoint fine-tuned |
| Data | 12 hàng training lấy mẫu cố định seed42, chỉ số và hashes trong JSON |
| Optimizer | AdamW, LR5e-5, weight_decay0, clip norm1, không scheduler |
| Steps / microbatch / accumulation | 3 / 1 / 4 |
| Precision / mode | FP32, TF32 OFF, train/dropout ON |
| Gradient checkpointing | Non-reentrant |
| Loss, gradients, weights hữu hạn | PASS cả3 steps |
| Norm gradient toàn model trước clip | 41.5047 → 25.5738 → 20.0860 |
| Max absolute gate sau step3 | 0.0001489563 |
| Norm cập nhật encoder layer1 fc1 | 0.2035016 |
| Peak allocated | 12,286.47 MiB (~12.00 GiB) |
| Peak reserved | 14,422 MiB (~14.08 GiB) |
| GPU free trước/sau | 23,542 / 23,542 MiB |

Field `gate_gradient` trong receipt được ghi **sau global clipping**, còn `preclip_gradient_norm` là norm trước clip. Không so magnitude hai field như cùng reduction. Ba steps dùng các câu khác nhau nên loss thay đổi không chứng minh hội tụ hoặc hơn baseline. Không có full-GPU cache/generation test riêng cho v2; các kiểm thử đó hiện là CPU.

Đây là optimizer smoke giới hạn3 steps, accumulation4 để kiểm tra kỹ thuật, **không phải recipe train chuẩn hóa**. Không dùng effective batch4 này để so số với baseline batch32, không suy tốc độ/VRAM đủ cho mọi câu dài256 hoặc training đầy đủ. Temporary updated weights bị hủy khi tiến trình kết thúc, không có model chất lượng cần giữ hoặc push; chỉ JSON nhỏ được lưu.

### 11.4 Bước tiếp theo và cổng dừng

1. Viết các correction controls ở bảng11.2 trong cùng module có args; kiểm tra identity và gradients bằng tensor nhỏ. Đóng băng candidate và control formulas trước pilot.
2. Tích hợp runner bằng method ID mới; bảo toàn recipe baseline, data hashes, tokenizer, batch/accumulation, scheduler, checkpoint selection và early stopping10. Chạy trainer smoke theo đúng recipe, gồm checkpoint resume và cached beam với v2 trên pretrained GPU. Không tái dùng memory estimate của smoke ngắn như gate bộ nhớ chính thức.
3. Chỉ khi các gate trên PASS mới chạy pilot seed42 validation-only; đánh giá theo protocol đã định trước. Đề xuất gate exploratory vẫn là validation chrF++ beam1 ít nhất +0.30 trên baseline và best control, BLEU không giảm; kết quả một seed không phải confirmation.
4. Nếu mixed không hơn best control thì không giữ claim interaction; nếu cả nhóm không hơn baseline thì dừng hướng này. Nếu hơn, kiểm chứng seed bổ sung trước mở test hoặc ngôn ngữ thứ2. Không đổi ngưỡng sau khi xem số.
5. Novelty vẫn HOLD: phép chuẩn hóa scale không tự tạo novelty; các prior differential attention/depth residual đã ghi ở §8 vẫn áp dụng. Chưa có căn cứ hứa phương pháp mới đủ ACL.

## 12. Queue pilot U2-v2 đã triển khai

### 12.1 Recipe và fairness

Người dùng duyệt triển khai queue RTX ngày27/10/2026. Config duy nhất: `configs/experiments/amis_depth_difference_v2_rtx.json`; entrypoint chung: `scripts/experiment.py`.

**6 runs** gồm baseline mới vì backend attention của baseline lịch sử chưa được pin trong receipt. Không kết luận số baseline cũ sai. Baseline cohort này dùng cùng eager backend với5 nhánh mới; không thay bảng số cũ bằng kết quả mới khi chưa tổng hợp.

| Thứ tự | ID | Method ID | Vai trò |
|---:|---|---|---|
| 1 | B0 | baseline | Baseline matched eager mới |
| 2 | P5 | depth-difference-value-scaled-v2 | Mixed interaction U2 |
| 3 | P1 | depth-ordinary-v2 | Ordinary context difference |
| 4 | P2 | depth-attention-v2 | Attention difference only |
| 5 | P3 | depth-value-v2 | Value difference only |
| 6 | P4 | depth-additive-v2 | Additive without mixed |

Cùng seed/data_seed42, pretrained mBART revision §9, Amis train4600/validation576, source tl_XX/target zh_CN, max source/target256; training microbatch4 × accumulation32 = **effective batch128**, FP32, BF16/FP16/TF32 OFF. AdamW torch fused LR5e-5, beta(.9,.999), epsilon1e-8, weight_decay0, warmup6%, linear scheduler, clip1, checkpointing reentrant. Tối đa20 epochs, patience10, chọn theo original-reference validation chrF++ greedy beam1; ties giữ earliest best. Eval batch2, pad multiple8. Không đánh giá test trong queue.

### 12.2 Gates và triển khai

- `tests/test_depth_difference_controls.py`:2 tests PASS, kiểm tra5 công thức/đạo hàm, identity ordinary=additive+mixed, zero-gate baseline, beam4 cache/no-cache và full-state mode-aware serialization.
- CPU gate runner PASS **B0/P5/P1/P2/P3/P4**, gồm accumulation loss contract và Trainer best-checkpoint reload; receipt `outputs/experiments/amis_depth_difference_v2_rtx/CPU_GATE.json`.
- Gate GPU mỗi nhánh:2 optimizer steps trên các hàng dài, cùng microbatch/accumulation của training, kiểm tra best checkpoint và export reload. Các nhánh depth thêm beam4 cache/no-cache; P5 thêm resume từ step1 với optimizer/scheduler/RNG và đối chiếu step2 liên tục. Gate lỗi thì queue dừng, không mở training.
- Queue PID lúc launch: `1726458`; chạy detached qua launcher chung, giữ `tmp/training.lock`. Không có queue CLRR active trước launch, VRAM free23,542 MiB; ổ `/workspace` còn133GiB theo `df -h`. Không can thiệp người khác.
- Bundle zip SHA256: `21f4e7b4ed76be475319f934ae51b71aaf47ae3d3b57f80d23f6cfd2e9b6fd8d`.
- Thư mục remote: `/workspace/ura/thanhdat/projects/CLRR/outputs/experiments/amis_depth_difference_v2_rtx/`; logs `queue.log`, script `run.sh`, PID `queue.pid`, gates `gates/<ID>/PASS.json`, training `runs/<ID>/`.

Đây là launch receipt; không dùng trạng thái launch để suy mọi gate đã PASS hoặc training đã bắt đầu. Trạng thái tiếp theo phải lấy từ log/receipts thực tế.

Snapshot sau launch: queue còn sống, **B0 GPU gate PASS**, gồm2 optimizer steps và Trainer/export reload,132.83 giây, peak allocated11.99GiB. Chưa có PASS receipts của5 nhánh depth tại snapshot này; training dài chưa bắt đầu. Receipt nhỏ local: `outputs/experiments/amis_depth_difference_v2_rtx/gate_launch_status.json`.

### 12.3 Checkpoint và điều kiện kết thúc

Repo public `FiveC/amis-rewire-checkpoints`, prefix theo mỗi method:

```text
checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/
  rtx6000-fp32-eager-b4-ga32-es10-depth-pilot/
```

Giữ archive `best.zip`, metadata, resolved args, validation predictions/metrics và trainer states nhỏ. v2 dùng method ID riêng trong `prototype_config.json`, gates nằm trong full wrapper state; load phải dùng đúng wrapper. Sau upload, runner xác minh remote SHA/receipts rồi mới xóa checkpoint/export/archive local. Upload hoặc verify lỗi thì giữ local và dừng queue. Không shutdown server RTX. Queue kết thúc sau6 runs validation; không tự mở test, thêm seeds hoặc ngôn ngữ khác.

Theo dõi từ Windows trong conda clrr:

```powershell
python scripts/experiment.py monitor --config configs/experiments/amis_depth_difference_v2_rtx.json --account 1
```

Quyết định exploratory theo §11.4; chưa có số pilot tại thời điểm launch. Không claim novelty hoặc cải thiện từ correctness gates.

## 13. Audit bổ sung related work sau câu hỏi của tác giả

Ngày27/10/2026. **Audit trước launch có phạm vi giới hạn, chưa đủ để xác nhận novelty.** Đã có6 nguồn gần trong §5.4 và prior ReZero; lần này đọc lại phần phương pháp/công thức và bổ sung Depth-Attention2026, NeuTRENO2023. Không gọi đây là đọc hết literature, systematic review đầy đủ hoặc chứng nhận không tồn tại prior trùng.

### 13.1 Nguồn gốc và đối chiếu công thức

| Nguồn primary, phần đã đối chiếu | Phần đã có trong literature | U2 khác ở đâu / hệ quả |
|---|---|---|
| [Differential Transformer, §2.1](https://arxiv.org/html/2410.05258v2) | Trừ hai softmax attention maps rồi nhân V | Primitive `(A_d-A_top) ΔV` nằm trong biểu thức differential attention khi chọn V=ΔV, λ=1. Không phải phát minh primitive mới. Đây là đối chiếu toán tử, không phải chứng minh toàn bộ kiến trúc/tham số hóa và training dynamics đồng nhất với paper |
| [Accurate Scene Text Recognition…, §4.3 Eq11](https://arxiv.org/html/2503.16184v1) | Differential cross-attention trong decoder đã có | Vị trí cross-attention/decoder tự nó không phải novelty. Bài này không dùng cặp encoder depths với value-difference như U2 |
| [Transparent Attention, §2.4 Eq1–2](https://aclanthology.org/D18-1338.pdf) | Decoder đọc tổ hợp học được của các encoder layer outputs | Quyền truy cập lower encoder layers đã có. U2 dùng signed mixed difference thay hidden-state weighted mixture |
| [Value Residual Learning, §3.1–3.3 Eq3–5](https://aclanthology.org/2025.acl-long.1375.pdf) | ResFormer trộn values từ layer đầu vào layers sau; SVFormer chia sẻ values | Cross-depth values đã có; cần phân biệt value-only với mixed. Không gọi value difference hoặc reuse values là novelty độc lập |
| [Depth-Attention, §3.2 Eq2–4, 2026](https://arxiv.org/html/2606.05014v2) | Query trộn values theo depth ở cùng token rồi dùng causal self-attention | **Nguồn thiếu trong bảng trước.** Không cùng công thức U2: depth softmax mixing trong decoder self-attention khác signed difference của hai encoder memories trong cross-attention. Tuy nhiên “depth information inside attention/value mixing” không thể claim mới |
| [DEX, §3.1 Eq3](https://arxiv.org/html/2505.16333v1) | Điều chỉnh output attention của pretrained model bằng projection trừ có hệ số học | Pretrained attention có nhánh correction đã có; U2 khác ở nguồn depth và ràng buộc interaction, không ở việc thêm correction đơn thuần |
| [Differential Transformer V2, báo cáo/code tác giả](https://huggingface.co/blog/microsoft/diff-attn-v2) | Shared K/V, thêm queries, coefficient token/head | Chia sẻ projections hoặc gating không đủ làm novelty. Đây là nguồn kỹ thuật tác giả, không tự ghi là paper peer-reviewed |
| [ReZero, Eq1](https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf) | Residual branch với scalar học được, khởi tạo0 | Gate zero-init là lựa chọn implementation có prior |
| [NeuTRENO, §3](https://proceedings.neurips.cc/paper_files/paper/2023/file/fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf) | Fidelity regularization chống over-smoothing dẫn tới nhánh hiệu chỉnh | Không claim “thêm correction để chống over-smoothing” là mới. U2 hiện cũng chưa có bằng chứng nó sửa over-smoothing gây hại trên Amis |

Phạm vi đọc: methodology/formula và các đoạn liên quan trực tiếp, không tuyên bố đã đọc từng trang mọi appendix hoặc chạy code tác giả. Tìm bổ sung các cụm `differential attention + value difference(s)`, `cross-layer + differential attention`, `attention maps + value difference` trên arXiv/OpenReview/ACL. Nhiều query không trả nguồn trực tiếp đúng công thức; điều đó không chứng minh không có prior. Chưa hoàn thành forward/backward citation tracing mọi nhánh.

### 13.2 Verdict và tác động tới queue

**Novelty HOLD.** Chưa thấy một nguồn đã kiểm tra có toàn bộ ràng buộc đúng như U2-v2; bằng chứng hiện tại cũng chưa đủ gọi U2 là phương pháp đủ novel cho ACL. Tách biệt ba điều: khác implementation, hữu ích thực nghiệm và đóng góp nghiên cứu đáng kể.

Đóng góp còn có thể kiểm chứng: ràng buộc mixed difference qua depth, correction triệt tiêu khi attention hoặc values bằng nhau, và tác dụng của ràng buộc đó trên low-resource NMT. Các điều kiện triệt tiêu là hệ quả đại số, không tự chứng minh có ích. Cân scale values được chọn từ diagnostic, chưa có novelty riêng.

Sáu runs §12 là **baseline + internal operator ablations**. Chúng có giá trị phân biệt cơ chế nhưng **không phải benchmark đầy đủ với prior**. Ordinary/P2/P3 là adapted controls, không gọi là tái lập Transparent Attention/Differential Transformer/ResFormer. Nếu pilot tốt, trước claim paper cần xác định prior/control sát nhất để so trực tiếp, mô tả adaptation và tài nguyên; không tự thêm runs vào queue hiện hành.

Quyết định đề xuất: giữ training dài trong lúc chốt claim và phạm vi prior; nếu tác giả chọn tiếp tục pilot thì xem đó là đầu tư exploratory với rủi ro novelty, chưa mở confirmation. Đã gửi câu hỏi cho tác giả; **chưa nhận câu trả lời tại snapshot này, chưa thực hiện thay đổi queue**. Snapshot kiểm tra: B0/P5/P1 có GPU PASS receipts, queue alive, chưa có queue training state. P5 PASS bao gồm resume và beam-cache checks theo gate code. Không dùng snapshot này suy trạng thái về sau.

Audit CLRR/LSR gốc là phạm vi khác; audit U2 này không xác nhận hoặc bác bỏ novelty của phương pháp cũ. Nếu quay lại CLRR/LSR cần đối chiếu công thức gốc với residual-rewiring và alignment/distillation prior riêng.

## 14. Quyết định sau khi tác giả đồng ý chốt audit trước training

### 14.1 Trạng thái thực thi đã xác minh

Ngày27/10/2026,18:43 giờ Việt Nam: cả **B0/P5/P1/P2/P3/P4 GPU gates PASS**, mỗi gate2 optimizer steps, cùng FP32/microbatch4/accumulation32/checkpointing của cohort. Các nhánh depth PASS beam4 cache/no-cache; P5 PASS optimizer/scheduler/RNG resume. Peak allocated các gate khoảng11.99GiB, riêng P5 có resume khoảng14.27GiB. Các bước này chỉ kiểm tra kỹ thuật, không đo chất lượng dịch.

Queue shell được giữ trước khi mở training, sau đó kết thúc riêng process group đã xác minh chứa script của project. `runs/` không có file; `ps` sau dừng không còn process trong group. **0/6 models đã train dài; không có checkpoint training cần cứu hoặc upload.** Giữ config, source bundle và receipts, không tự restart bằng launcher hiện tại. Snapshot cũ ở §12–13 chỉ mô tả lịch sử trước quyết định này.

### 14.2 Bổ sung hai nguồn NMT sát với động cơ thiết kế

| Nguồn / phạm vi đã đọc | Hệ quả cho U2 |
|---|---|
| [Lexical Shortcuts, WMT2019, §1–3](https://aclanthology.org/W19-5211.pdf) | Gated access tới lexical representations để giảm representation bottleneck đã có. Bài dùng embedding shortcuts trong self-attention; không phải mixed depth cross-attention U2. Không claim động cơ giữ thông tin tầng thấp hoặc gated shortcut tự nó mới |
| [Every Layer Counts, PBML2020, phần phương pháp MLMHA và so sánh aggregation](https://pure.ulster.ac.uk/ws/files/87417499/art_ampomah_et_al.pdf) | Decoder truy cập trực tiếp nhiều encoder outputs đã được khảo sát trong NMT. Không claim chỉ thêm decoder access tới tầng6 là đóng góp mới. So sánh phải phân biệt lợi ích có thêm memory với lợi ích toán tử mixed |

Đây là audit có giới hạn, đối chiếu phần phương pháp liên quan; chưa tái lập code tác giả, chưa hoàn tất citation tracing toàn bộ. Kết quả tìm kiếm không thấy trùng nguyên cấu hình không chứng minh độc quyền ý tưởng.

### 14.3 Claim nào giữ được, claim nào phải bỏ

Đặt `F(A,V)=AV`. Mixed correction là finite mixed difference:

```text
F(Ad,Vd)-F(Ad,Vt)-F(At,Vd)+F(At,Vt)
  = (Ad-At)(Vd-Vt).
```

Đây là identity của phép nhân song tuyến tính, không phải định lý attention mới. Nó triệt tiêu khi Ad=At hoặc Vd=Vt; vì hai attention maps có row sum1, correction cũng triệt tiêu nếu value difference giống nhau tại mọi vị trí source. Những tính chất này chưa chứng minh translation hữu ích. Scale v2 cân norm hidden input trước V projection, **không đảm bảo projected value norms bằng nhau**.

| Claim dự kiến | Verdict |
|---|---|
| Phát minh phép trừ attention maps, cross-attention differential, depth access, value reuse hoặc gate zero-init | NO-GO: có prior tương ứng ở §13–14 |
| Một cấu hình ràng buộc differential attention bằng paired encoder depths và mixed value difference | Mô tả implementation hợp lệ; chưa đủ chứng minh đóng góp ACL đáng kể |
| Mixed restriction tốt hơn các cách truy cập depth đơn giản trong low-resource NMT | RQ còn mở; cần kết quả, không phải insight đã được xác nhận |
| Sửa over-smoothing, khôi phục morphology, hoặc giải quyết nguyên nhân CLRR/LSR thất bại | Chưa có bằng chứng nhân quả; không claim |

### 14.4 RQ và đối chứng cần khóa trước khi mở lại

**RQ có thể bảo vệ:** Với cùng pretrained backbone và quyền truy cập hai encoder depths, ràng buộc đồng thời chênh lệch attention và values có cải thiện dịch hơn fusion thông thường hoặc chỉ chênh lệch một thành phần không?

Sáu nhánh hiện tại trả lời decomposition nội bộ khá rõ, nhưng chưa có **simple positive donor-context addition** `AtVt + g AdVd` hoặc prior-derived hidden mixture. P1 ordinary difference không đồng nhất positive addition khi dùng cùng gate: hệ số context baseline cũng bị thay đổi. Vì vậy không gọi P1 là toàn bộ đối sánh depth fusion.

Nếu tiếp tục hướng U2, cần chốt trên giấy một comparator đơn giản cho quyền truy cập donor và một adaptation sát prior trước; ghi rõ shared projections, detach, scaling, dropout, layer selection, tham số và chi phí. Comparator adapted không được đặt tên như đã tái lập đầy đủ paper gốc. Chưa thêm hoặc train run nào cho các comparator này; không tự tăng budget6 runs.

### 14.5 GO / NO-GO hiện tại

- **GO kỹ thuật:** code đã qua CPU/GPU gates của cohort, có thể nghiên cứu tiếp khi thiết kế được duyệt.
- **HOLD training dài:** novelty và đối chứng chưa đủ chốt; queue đã dừng thực tế.
- **NO-GO cho U2 làm đóng góp attention primitive mới:** công thức gần prior và là phép kết hợp các thành phần đã biết.
- **Chưa loại bỏ giả thuyết ứng dụng NMT:** nếu tác giả vẫn muốn khảo sát, coi là exploratory với rủi ro novelty; cần duyệt lại bảng runs sau khi khóa comparator. Không hứa một pilot tốt đủ sức nộp ACL.

Khuyến nghị hiện tại: không mở lại nguyên queue6 nhánh chỉ vì gates PASS. Chốt đóng góp muốn bảo vệ và comparator trước, giữ CLRR/LSR cùng insight cũ làm evidence lịch sử. Không đổi phương pháp hoặc thêm sweep khi chưa được tác giả đồng ý.
