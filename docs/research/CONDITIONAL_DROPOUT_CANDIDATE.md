# Conditional encoder-dropout consistency: candidate card

Date: 2026-11-14. Status: HOLD — author rejected a weak-novelty mechanism pilot.
No trainer, GPU gate or training launched.

## Current decision: do not launch

The author clarified after the proposed pilot that novelty must be strong
enough to justify a potential paper contribution BEFORE spending training
budget. This mask-coupling proposal does not meet that requirement. E/D/Wcal
are therefore on HOLD; the earlier pilot request is superseded. The proposed
budget and execution steps below are retained as design history, not an active
queue or permission to train. Do not promote this candidate merely because a
future score might improve. Next method proposals require a primary-source
novelty audit of their central contribution before pilot admission.

## 1. Decision and research question

Treat this as a **mechanism-oriented engineering probe**, not an established new
ACL method. Its exact encoder/decoder mask coupling was not identified in the
primary sources checked below. Shared stochasticity, component-specific
consistency and symmetric KL are already known. Absence from this checked set
does not establish global novelty.

**RQ:** With each view's dropout distribution fixed, does consistency under
encoder perturbations help low-resource translation more than consistency
under decoder or whole-model perturbations, beyond auxiliary-loss strength?

The possible contribution is an experimentally supported account of *where*
consistency helps, together with a useful recipe. A seed42 improvement alone
cannot establish that contribution or an ACL acceptance claim.

## 2. Evidence motivating the question

Existing Amis validation BLEU (development, not new results):

| Seed | Baseline | B: ordinary R-Drop, beta2.5 | W: ordinary R-Drop, beta1.25 |
|---|---:|---:|---:|
| 42 |17.682329|17.910697|18.272410|
| 43 |16.734501|17.575737|18.395755|
| 44 |16.208941|18.327450|18.188117|
| Mean |16.875257|17.937961|18.285427|

W is tuned R-Drop, not a new method. The frozen first64-validation-row probe
found lower ordinary dropout KL for B than W and different correct/wrong-source
responses. These are final-checkpoint associations. They do not establish
source neglect, a causal failure of decoder regularization, or a need for this
candidate. In particular, W's larger correct-minus-wrong source gap also came
with worse correct-source CE than B. Sentence-generation correlations were
near zero. Full verified evidence remains in
`outputs/experiments/amis_consistency_source_response_rtx/ANALYSIS.md`.

Earlier CLRR additions and gold-only/non-gold-only consistency did not beat W
in the corresponding development comparisons. Do not add them to this pilot
without independent evidence; do not rebrand ordinary R-Drop as CLRR/LSR.

## 3. Exact candidate E

For source x, teacher-forced target prefix h and gold token y:

    z1 = D_theta(E_theta(x; e1), h; d)
    z2 = D_theta(E_theta(x; e2), h; d)
    p1 = softmax(z1); p2 = softmax(z2)
    SKL(p1,p2) = (KL(p1||p2) + KL(p2||p1))/2
    L = [CE(z1,y) + CE(z2,y)]/2 + 1.25 * SKL(p1,p2)

Reduce over labels != -100, including supervised special tokens, exactly as W.
No temperature, stop-gradient, teacher, extra parameters or new loss weighting.
Both branches and both encoder/decoder parameter sets receive gradients.

- e1,e2 are independent encoder dropout realizations.
- d is the same **active stochastic decoder realization** in the two views.
- Decoder embedding, attention and FFN dropout masks are shared at corresponding
  operations, with normal dropout rates. Fresh realizations on subsequent steps.
- Every individual view has the original dropout marginal. Only the joint law
  across views changes. This does not halve per-view decoder variance.
- This does not switch decoder dropout off, freeze the decoder, or guarantee
  that the decoder is unaffected by the consistency gradient.
- Native deterministic inference, no rewiring or LSR.

Limitation: a model ignoring encoder outputs can make this auxiliary loss zero.
CE is still necessary; the construction does not guarantee source grounding.

## 4. Prior audit, checked directly and with NotebookLM

| Primary source | Relevant design | Consequence |
|---|---|---|
| [R-Drop, NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf), section2.2 | Two stochastic views, CE and symmetric output KL | Loss is inherited, not novel |
| [What Works and Doesn't Work, A Deep Decoder for NMT](https://aclanthology.org/2022.findings-acl.39.pdf), section3.4, Eq5 | DDR feeds the same encoder representation to two stochastic decoder passes | Swapped perturbation control below is prior-inspired; do not claim decoder consistency is new |
| [Layer-wise Regularized Dropout](https://aclanthology.org/2024.lrec-main.891.pdf), section3, Eqs3,6-8 | Hidden-state, attention and output agreement | Layer-specific consistency is known |
| [UnitY](https://aclanthology.org/2023.acl-long.872.pdf), AppendixB, Eq2 | R-Drop on decoders in a speech translation architecture | Decoder-localized R-Drop is known |
| [Consistency Models Made Easy](https://arxiv.org/html/2406.14548v1), section3.1 near Eq12 | Same dropout masks between different noise-level inputs | Shared masks to isolate a perturbation are known outside NMT |
| [DuetMatch](https://arxiv.org/pdf/2510.16146), sections3.2-3.3 and3.6, Eqs8-12,23-24 | Component perturbations, separately frozen/trainable branches, teacher updates | Component-specific dropout consistency is known; candidate has shared trainable weights, no frozen teacher/EMA |

Nearest NMT contrast is DDR's opposite allocation of stochasticity. Nearest
general design ingredients include shared masks and component perturbations.
The inspected formulations do not establish an exact match to E, but the
remaining distinction is modest. **No novelty guarantee; no standalone ACL
novelty PASS assigned.** Do not claim the first encoder consistency method.

NotebookLM notebook: bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890.
Initial query: d8de8bd45a24. Follow-up incorporating ECT and DuetMatch:
c9ca71500f59. Review is advisory; primary equations take precedence.
Rejected review claims: sharing masks halves marginal variance; decoder has no
consistency gradient; equality with one control proves a unique causal account.
Follow-up completed and rated standalone novelty low, mechanism probe feasible.
Its dismissal of ECT's mask sharing and covariance=1 claim are incorrect; see
`outputs/experiments/conditional_dropout_design/NOTEBOOKLM_REVIEW.md`.

## 5. Proposed experiment budget: three new seed42 models

This is a proposed budget, not a running queue or a training authorization receipt.

| Run | Data/backbone/seed | Encoder masks | Decoder masks | Purpose |
|---|---|---|---|---|
| E |Amis->Mandarin / mBART-50 /42|Independent|Shared, active|Candidate; beta1.25 |
| D |Same|One shared stochastic encoder output|Independent|DDR-style swapped control; beta1.25 |
| Wcal |Same|Independent|Independent|Ordinary R-Drop with one precomputed coefficient, to control initial auxiliary-gradient strength |

D is a DDR-style control, not a complete reproduction of the deep-decoder
paper, which also studies other architectural/training choices.

Reuse existing W, B, A (two-pass CE) and baseline only after matching model/data
revision, recipe, selector, backend, seed and output provenance. Historical
baseline backend reconstruction remains a caveat; do not call it a perfectly
contemporaneous control. No need to retrain all old methods for this pilot.

### Fixed calibration for Wcal, before any long training

Use the unmodified pretrained model revision below, first64 TRAIN rows, batch2,
dropout pair seeds42001-42004. This subset is not the validation diagnostic set.
Compute the mean per-batch L2 norm of the gradient of *unweighted* token-mean
SKL with respect to concatenated two-view logits, separately for E and ordinary W.
Freeze:

    beta_Wcal = 1.25 * mean_norm_E / mean_norm_W

Use the same supervised-token mask and reductions. Record batches, seeds,
per-batch norms, coefficient and hashes before training. Zero/nonfinite
denominator or coefficient is HOLD; do not silently substitute a value or grid.
One calibration, no validation-based coefficient search. This matches one
initial logit-gradient summary, not parameter gradients, later gradients or
Adam updates. It cannot remove every strength confound. If already numerically
equal to1.25 within relative1e-6, reuse W and record that only two models are needed.

## 6. Locked recipe and execution prerequisites

Inherit the exact protocol/run fields from
`configs/experiments/amis_consistency_strength_w_replication_rtx.json` and
`configs/protocols/amis_drb_bleu.json` (SHA256
bdca2050235d6e938bb4866cf68354ee4ec63f443298a90eed6df3bdea6ad926).

- mBART revision e30b6cb8eb0d43a0b73cab73c7676b9863223a30; source tl_XX, target zh_CN.
- Existing original CSV train4600/validation576/test575; unchanged references.
- FP32, SDPA, TF32 off, B4, accumulation32, effective batch128, LR5e-5.
- Max20 epochs; early stopping10; select original-reference validation BLEU,
  beam1 FP32, earliest tie. No new test, Ede or A100 in this pilot.
- Optimizer/scheduler, dropout rates, warmup, clipping and lengths inherited
  unchanged; runtime versions and all hashes must be recorded in final config.
- RTX work only inside `/workspace/ura/thanhdat/projects/CLRR`; conda clrr;
  check shared GPU availability first; do not terminate anyone else's process.

Before a training-capable queue can be admitted:

1. Implement separately under `src/methods/conditional_dropout/v1/`, preserving W.
2. CPU checks: marginal CE agreement by exact mask enumeration; shared-all KL0;
   encoder-only/decoder-only mask allocation; nonzero decoder gradients possible;
   independent masks recover ordinary W loss; padding/special-token reductions.
3. GPU checks on tiny mBART: inventory every stochastic site, including SDPA
   attention dropout and layerdrop if enabled; verify exact intended coupling,
   RNG advancement, accumulation and activation-checkpoint backward replay.
   Resetting a global seed before whole passes is NOT a sufficient implementation.
4. Same-mask input replay, strict save/load and deterministic generation checks;
   two optimizer steps only, no data/precision/backend changes to make gates pass.
5. Freeze training config and Wcal receipt, all gates PASS. If SDPA dropout
   coupling cannot be verified, HOLD instead of switching to eager or changing
   dropout probabilities. Report resource or correctness failures explicitly.

## 7. Decision rules, fixed before pilot outcomes

Primary development score: validation BLEU beam1. Secondary: validation chrF++.
Promote E only if seed42 BLEU exceeds the best compatible W/B/D/Wcal comparator
by at least0.30 and chrF++ is no lower than that BLEU-winning comparator.
Baseline and A are contextual checks, not sufficient targets to beat. Define
comparator ties by higher chrF++ then fixed order W,B,D,Wcal.

- E loses to W: stop this candidate, no expansion or post-hoc coefficient sweep.
- E beats W but not D/Wcal: useful result about dropout/strength, insufficient
  support for preferring encoder perturbations; no mechanism claim.
- E passes: replicate E and the strongest relevant control on seeds43/44 using
  the identical recipe; this is a separate next stage, not an automatic queue.
- Only after stable development evidence: freeze method and evaluate a second
  language pair under matched controls. Existing Amis/Ede development exposure
  must be disclosed; neither becomes an untouched holdout by using fresh seeds.
- Test/paired bootstrap comes after method lock. Sentence bootstrap does not
  replace between-training-seed uncertainty. No significance claims from this
  seed42 pilot and no repeated test-driven tuning.

## 8. Current state

Completed: mathematical specification, existing-evidence review, primary-paper
audit, two NotebookLM critiques, and five exact finite-mask CPU checks (PASS).
The CPU check verifies equal expected per-view CE, shared-all KL0, possible
decoder KL gradients, the source-insensitive auxiliary minimum, and absence
of a universal scalar equivalence to ordinary W. It is not a pretrained-model
smoke gate or evidence of translation improvement. Small artifacts live in
`outputs/experiments/conditional_dropout_design/`.
This card proposes a bounded mechanism experiment.
It does not establish a new successful method. No GPU/trainer implementation,
new queue, test evaluation or checkpoint upload has been performed for E/D/Wcal.
