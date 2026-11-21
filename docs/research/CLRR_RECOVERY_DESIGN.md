# CLRR recovery: evidence, related work and bounded next design

Date: 2026-11-10. Status: **FOUR-ARM ENGINEERING PILOT AUTHORIZED; CPU PASS;
GPU admission pending.** Author approved running A–D after the design review.

This document proposes an engineering refinement of original CLRR. It does not
certify novelty or claim an ACL-ready method. Existing scores, checkpoints,
recipes and data remain unchanged. RTX results are exploratory and separate
from the A100 paper cohort.

## 1. Decision from the evidence

The current original Full recipe has not demonstrated a reliable advantage
over baseline. Further unrestricted LSR weights, scalar strengths or layer
sweeps are not justified. Preserve original CLRR as the architectural hypothesis,
but test a different training objective: supervise both the native and rewired
paths using the translation target. Add an established output-consistency
regularizer only with its own matched control.

This targets a specific hypothesis: a useful rewired path may be learnable
without sacrificing the native path's translation decisions. It is a hypothesis,
not a diagnosis proved by the current probes. Shared weights do not guarantee
that the native path retains the old baseline's accuracy.

## 2. Evidence supporting and constraining the decision

Latest CPU inventory:
`outputs/experiments/method_recovery_review/INVENTORY.md` and `EVIDENCE.json`.
There are **66 completed local validation receipts and 3,548 history entries**.
These are not 66 independent models, nor an exhaustive HF/paper inventory.
Missing local receipts do not establish missing HF checkpoints.

### Recent Amis/mBART seed42 validation

| Method | BLEU | chrF++ | Interpretation |
|---|---:|---:|---|
| Reused baseline R1 | 17.6823 | 14.7030 | Reference, with historical runtime caveat below |
| Original CLRR-only | 16.6216 | 14.1252 | Fixed rewiring alone has not helped this seed |
| Original cosine LSR-only | 17.2418 | 14.4559 | Completed missing control; below baseline |
| Original Full | 17.1156 | 14.3450 | Adding CLRR does not improve the original LSR control here |
| Learned-strength CLRR + CE | 16.6360 | 14.1009 | Learned scalars do not rescue CE training |
| Learned-strength Full | 17.3459 | 14.4554 | Only +0.1041 BLEU vs LSR-only; chrF++ essentially unchanged |
| Transition LSR-only | 17.0833 | 14.2826 | New alignment objective alone below baseline |
| CLRR + Transition LSR | 17.5220 | 14.6387 | Better than original Full, still below baseline |

These rows are descriptive, not one newly randomized eight-arm comparison.
Original runs selected by chrF++; complete trajectories establish that their
selected checkpoints also maximize BLEU for the original rows above. Historical
SDPA is reconstructed rather than supported by a contemporaneous explicit
backend receipt. Do not erase this provenance limit or mix recipe groups.

Original LSR-only completed 20 epochs; original CLRR/Full and Transition runs
also have complete 20-epoch histories. ES patience is not the explanation for
their stopping time. This does not establish whether longer training would help.

### Wider evidence: what not to repeat blindly

| Evidence / family | Finding | Consequence |
|---|---|---|
| Original Full, fresh A100 seeds45–52 | Full−LSR mean −0.0198 BLEU, 3/8 positive; Full−baseline −0.0415, 4/8 positive | Current confirmation failed; neither significant superiority nor equivalence established |
| Original Full token probe, seed42 | Content NLL improves 3.474846→3.427548, but top1 falls 50.4550%→49.6360%; 227 repairs and 272 broken baseline-correct positions | Better likelihood/alignment need not improve translation decisions; teacher forcing is not causal evidence about generated BLEU |
| Route ON/OFF probes | Original trained routes can help their own model; pruning selected upper routes did not reliably improve quality | Co-adaptation is plausible; do not infer retraining after pruning must help |
| Learned-strength diagnosis | Injection norm about 0.17–0.24%; ON/OFF CE CIs include zero; optimizer/gradient registration checks pass | Weak route is observed; neither a proven bug nor proof that increasing strength/LR helps |
| Gradient and centered-cosine probes | Instantaneous auxiliary gradients do not establish domination; common direction exists but centered alignment remains | No supported claim that all alignment is fake or that collapse explains failure |
| Alignment-residual C0 replication | BLEU-selected 3-seed test mean delta −0.0558; deltas +0.9326,+0.6175,−1.7175 | A promising pilot did not yield stable improvement |
| Depth fusion and decoder residual bridge | Validation improvements insufficient against matched controls | Do not introduce another generic depth gate/adapter as if untested |
| Sequence-risk continuation | All four selected candidates below their starting baseline | No automatic longer continuation or optimizer-reset explanation |
| Relational / near-pair branch | Original paired-margin premise was not supported by initial source-discrimination probes; later training was interrupted | Not a completed negative training result; not a justified next default |

Detailed evidence stays in `TRANSITION_FOLLOWUP_DIAGNOSIS.md`,
`LEARNED_STRENGTH_DIAGNOSIS.md`, `CONTINUATION_REVIEW_AND_NEXT_DESIGN.md`,
and `../experiments/EXPERIMENTS_RTX6000.md`. Amis and Ê Đê test results have
already been inspected during development; neither is a pristine holdout now.

## 3. Related work audit and novelty boundary

The following papers' method sections were consulted, not just their titles.
This is a targeted prior-art check, not an exhaustive proof of absence.

| Prior | Relevant mechanism | Boundary for this proposal |
|---|---|---|
| [R-Drop, NeurIPS 2021, §2.1 and §3.1](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf) | Two dropout passes, supervised NLL and symmetric output KL | Two-view consistency and symmetric KL are established; must include R-Drop-style control |
| [BERT-fused NMT / Drop-Net, ICLR 2020, §4.2](https://arxiv.org/pdf/2002.06823) | Random use of conventional/BERT attention branches during training; expected combination at inference | Branch robustness is established; our deterministic OFF/ON pair is a design distinction, not proof of substantial novelty |
| [Prediction Difference Regularization, ACL 2022, §3–4](https://aclanthology.org/2022.acl-long.528.pdf) | Regularizes prediction changes between original and perturbed inputs | Prediction-level perturbation regularization is established; different perturbation alone does not establish novelty |
| [TIE-KD, ACL 2023, §3–4](https://aclanthology.org/2023.acl-long.448.pdf) | Examines token-ranking information in NMT distillation | Preserving top-ranked predictions is not a new standalone idea |
| [Selective KD, ACL 2021, §3–4](https://aclanthology.org/2021.acl-long.504.pdf) | Selectively applies teacher knowledge using sample criteria | A frozen teacher with confidence/margin selection would require this comparison, not an unsupported new-method claim |

Earlier repository audits also cover ReZero, layer aggregation, lexical shortcuts,
Ladder Side-Tuning and transition-based alignment. Those mechanisms are not
being reintroduced under new names here.

**Novelty decision:** allow this as an explicitly attributed engineering
optimization of CLRR; **do not admit it as a new standalone ACL contribution**.
The useful scientific question is whether rewiring adds benefit beyond a strong
output-consistency control. If it does not, the contribution cannot be rescued
by renaming the regularizer. Original CLRR's ownership is not evidence of
novelty relative to the literature.

## 4. Proposed objective: supervised native and rewired paths

For the same source x and teacher-forced target prefix y_<t, use shared weights
theta and independent ordinary dropout masks xi0, xi1:

    p0,t = P_theta(y_t | x, y_<t, route=OFF, xi0)
    p1,t = P_theta(y_t | x, y_<t, route=ON,  xi1)

ON uses the original encoder CLRR: distance2, strength0.1 and detached donors.
OFF disables those extra connections only. OFF is not a frozen baseline teacher.
No scalar gate, adapter, EMA network, negative examples or target encoder pass.

Let M be the target positions with labels != -100, including supervised language
and EOS tokens exactly as in baseline CE. Each CE averages over M:

    L_CE = ( CE(p0,y) + CE(p1,y) ) / 2
    L_cons = mean_{t in M} [ KL(p0,t || p1,t) + KL(p1,t || p0,t) ] / 2
    L = L_CE + beta * L_cons

KL sums over the full vocabulary before averaging target positions. Both
distributions receive gradients through both KL arguments; no detached teacher.
Temperature1. Padding is excluded. CE retains the baseline microbatch/token
reduction and accumulation convention in all four arms.

For the initial engineering pilot propose beta=2.5, fixed for both consistency
arms, with no sweep. This follows the relative weighting of R-Drop's reported
translation alpha5 when its sum of two CE terms is divided by two. It is a
normalization rationale, not a claim to reproduce their full training recipe,
nor evidence that this coefficient is optimal for Amis/mBART.

LSR is omitted in this first factorial test because its completed matched control
is below baseline and it would obscure the effect of paired supervision.
This is a controlled omission, not proof that LSR can never help. A later LSR
add-back would need a separately locked comparison after this family succeeds.

## 5. Four proposed new pilot runs

All: **Amis→Mandarin, mBART-50, RTX6000, seed/data_seed42, validation only**.
Fresh training from the same pretrained revision; no continuation from a selected
fine-tuned checkpoint. Labels below are local to this proposal.

| Run | Views | Objective | Role | Inference route |
|---|---|---|---|---|
| A | OFF / OFF | Mean two-pass CE | Controls two-view supervised training | OFF |
| B | OFF / OFF | Mean CE + beta symmetric KL | R-Drop-style matched comparator | OFF |
| C | OFF / ON | Mean two-pass CE | Isolates paired CLRR supervision | ON |
| D | OFF / ON | Mean CE + beta symmetric KL | Primary proposed engineering candidate | ON |

D versus B is the primary architectural comparison. C versus A measures
rewiring without consistency. B versus A and D versus C describe the
regularizer's contribution. Old baseline remains a reference; it is not replaced
or rerun by definition. A/B are necessary new objective controls, not duplicate
baseline runs. No best-of-four oracle, ensemble or per-sentence route selection.

Two full translation passes increase training cost; equal optimizer steps do not
mean equal FLOPs versus the old one-pass baseline. All four new arms share that
cost structure. Inference uses one pass and introduces no learned parameters.

## 6. Locked recipe and required implementation gates

Preserve the current RTX recipe: FP32, SDPA, TF32off, microbatch4,
accumulation32, LR5e-5, warmup0.06, weight_decay0, max_grad_norm1,
fused AdamW, source/target length256, ES10/max20epochs, original-reference
validation BLEU beam1 selector, unchanged data hashes and pretrained revision
from the existing frozen manifest. Source tl_XX, target zh_CN. No test selection.

There is already an **untrained paired-CE prototype** in
`src/methods/grounded_interaction/v1/model.py`. It is not a newly discovered
method and cannot be launched unchanged: it disables per-layer checkpointing
and substitutes whole-view non-reentrant checkpointing. That is a material
difference from the locked runner. Reuse only audited ideas/components, not
its runtime settings implicitly.

Before training:

1. Write a separate versioned method implementation and immutable four-arm
   recipe. Record the prior-art attribution and provisional engineering status.
2. CPU correctness: CE/KL masks and reductions, exact KL formula and gradients,
   KL zero when distributions coincide, routeOFF equivalence, detached-donor
   semantics, independent dropout/RNG replay and save/reload equivalence.
3. Verify multi-view backward with the locked checkpointing setting: route/cache
   state cannot leak across recomputation; every eligible layer receives the
   intended gradient. Compare checkpointed vs uncheckpointed gradients on a
   deterministic tiny model. Both KL arguments must remain differentiable.
4. Full-shape RTX admission and tiny gate for all arms: check free GPU memory
   first, two optimizer steps with actual accumulation, finite loss/gradients,
   save/reload and best-checkpoint selection. Do not disturb other users.
5. Full-vocabulary FP32 KL can be expensive for mBART's vocabulary. Do not
   silently reduce batch/precision, truncate vocabulary or switch backend to fit.
   If exact execution does not fit, HOLD; any memory implementation must be
   numerically checked before launch. Baseline numerical lock takes precedence.

Implementation: `src/methods/paired_route/v1/model.py`, separate from all old
methods. Five CPU correctness tests PASS, including full-reference KL value and
both-argument gradients, masked targets, dropout/accumulation gradient replay,
donor detach, route equivalence and checkpoint persistence. The implementation
retains per-layer reentrant backbone checkpointing and captures each encoder
invocation's donor/route for backward. Exact token-chunk loss recomputation
limits auxiliary memory without truncating the vocabulary. GPU gates pending.

## 7. Proposed advancement and stop rules

These are prospective practical thresholds, **not significance tests or
retrospectively achieved gates**. Freeze them with the recipe before launch.

- Primary D must exceed B by at least +0.20 validation BLEU and the reused
  baseline by at least +0.30, with chrF++ no lower than either comparator.
  A historical-baseline comparison still carries its provenance caveat.
- Inspect C−A, routeON/OFF token metrics and route norms as diagnostics, not
  alternate selectors. Improved KL or route norms alone do not pass the gate.
- If D fails, do not automatically promote whichever arm happens to score best,
  open test, increase beta or launch another grid. Record NO-GO for this primary
  design. A C-only follow-up would require an explicit new decision.
- If D passes, run only B/D on seeds43/44 next: **4 additional runs**. Require
  positive D−B on at least2/3 seeds, mean delta >=+0.20 BLEU and no mean chrF++
  loss; baseline comparisons must use the matching seed cohort. This is still
  development evidence, not proof of statistical significance.
- If replication passes, transfer the locked B/D comparison to Ê Đê, with
  dataset-specific language tokens from its existing manifest. A 3-seed transfer
  costs **6 runs**. It is transfer evidence on a development-exposed dataset.
- Only after those decisions lock the final evaluation and any A100 matched
  training. Report sentence paired bootstrap and seed-level uncertainty as
  different quantities; select multiplicity families prospectively.

Budget: **4 initial runs authorized after all GPU gates PASS.** Conditional replication4 and
transfer6 are not queued or automatically authorized by this document.

## 8. What would strengthen the paper?

- D beating B across seeds and datasets would support CLRR adding value beyond
  an established regularizer. It would not by itself establish theoretical novelty.
- D matching B would indicate that generic consistency may explain the gain;
  an ACL method claim centered on CLRR would remain weak.
- All arms below baseline would reject this recovery direction; preserve the
  scientific negative evidence and reconsider the research question instead of
  continuing unbounded optimization.

At present the evidence does not support promising that the original recipe
will become competitive. This plan tests one remaining grounded hypothesis
with explicit controls and a stopping rule.
