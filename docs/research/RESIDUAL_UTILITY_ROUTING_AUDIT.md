# Residual utility: validation evidence and routing design audit

## Material Passport

- Date: 2026-10-16 (Asia/Saigon).
- Scope: author-authorized steps 1–2, frozen-checkpoint measurement and candidate/prior audit.
- Status: measurements COMPLETE / independently replayed from CSV; candidate
  is NO-GO as a new main-method contribution in its present formulation.
- No optimizer steps, new model training, checkpoint reselection or test-based tuning.
- Interpretation and design authored with AI assistance; experimental evidence is linked below.

## 1. Question and frozen measurement plan

Does a fixed CLRR route help teacher-forced translation CE uniformly, or does
its utility vary by sentence/layer enough to motivate selective routing?
This is distinct from whether a learned gate can generalize or improve generation.

Worker: `scripts/analysis/measure_residual_utility.py`.
Remote workspace: `outputs/experiments/residual_utility_validation/`.

| Dataset | Existing checkpoints | Validation rows per checkpoint |
|---|---|---:|
| Amis–Mandarin | P1 CLRR-only, R1 baseline, R2 old Full | 576 |
| Rhade–Vietnamese | old Full seed42, baseline seed42 | 1,000 |

Selections are pinned by the existing `selection_lock.json` (archive revision
and SHA256). Baseline interventions add a counterfactual route to the same
baseline weights; they do not represent a trained CLRR model. All models use
the strict bare seq2seq backbone for logits, without LSR/InfoNCE loss wrappers.

For each checkpoint and every original validation row:

1. All routes ON, distance2/strength0.1, post-addition detached donors.
2. All routes OFF, same weights and teacher-forced target prefix.
3. One route OFF at each layer3–12, all other routes ON.

Removing one route also changes its downstream cached donors; this is part of
the intervention. Layerwise effects cannot be added to predict a combined mask.

For example i, let CE_i be summed target NLL divided by its non-pad target token
count. Keep language/EOS labels exactly as in training CE; also report corpus
token-weighted CE separately. Utility of the removed route is
`CE_i(condition) - CE_i(all-on)`; positive means the active route helped.
Record benefit >0.001 nats/token and harm <−0.001, with ties between them.
The threshold is a descriptive tolerance, not a significance gate.

Compute 10k sentence bootstrap descriptive mean-utility CIs. These are post-hoc,
unadjusted intervals, not simultaneous tests or confirmatory evidence.
Reference-dependent best-of-ON/OFF CE is an optimistic diagnostic ceiling only:
it is not an executable router, generated BLEU, or a guarantee of attainable gain.

Fixed exploratory strata: source whitespace length <=2 / 3–10 / >10; fraction
of source content tokenizer IDs with train frequency <=5: 0 / (0,.25] / >.25;
source UNK presence. No data filtering, text modification or new splits.

### Correctness and provenance

- Check free VRAM before every model; threshold16,000 MiB; respect project lock.
- Initial GPU availability23,530 MiB, utilization0%; all execution under
  `/workspace/ura/thanhdat/projects/CLRR/`, conda `clrr`.
- FP32, TF32 off, dropout off, checkpointing off, no generation or gradients.
- Strict missing/unexpected/mismatched-key checks; archive SHA; train/validation
  hashes; pinned library versions; original routing/helper source hashes.
- Repeat-ON logits, layer1-OFF negative control, OFF versus zero-strength CE;
  per-token CE reduction versus HF output.loss; parameter versions unchanged.
- The first attempt stopped before GPU work because the old Amis orchestration
  manifest differs from the current generic engine/registry/metrics files.
  The corrected measurement uses strict bare HF loading instead of that trainer.
  Original routing/helpers match the archived manifest; `configure_languages`
  AST matches the archived Amis bundle. Rhade frozen source manifest fully matches.
  Failed-start log is retained; no trained model or historical score was changed.
- Temporary downloaded weights are removed only after archive verification and
  successful result writing. Small JSON/CSV receipts remain.

## 2. Exact candidate examined, not implemented

The previous chat suggested token gates but proposed sequence-CE utility labels.
Those labels do not identify which source token helped. **Correct the initial
design to one gate per sentence and layer**, matching the intervention unit.
Token routing requires source-token interventions and cannot borrow these labels.

For layers l=3..12, d_l is the same detached post-rewiring distance2 donor:

\[
z_l=\operatorname{sg}([\operatorname{norm}(\operatorname{pool}(h_l));
                      \operatorname{norm}(\operatorname{pool}(d_l))]),\quad
g_l(x)=\sigma(w_l^T z_l+b_l),\quad
\widetilde h_{l,t}=h_{l,t}+0.1g_l(x)\operatorname{sg}(d_{l,t}).
\]

Pooling uses source content positions, omitting all special IDs and PAD.
Gate is shared over source positions within a sentence, not over the batch.
For d=1024 this adds10*(2d+1)=20,490 trainable parameters. This candidate is
**not parameter-neutral**. Gates forced0/1 must reproduce bare/original routes.
Provisional initialization w=0, b=log(.01/.99) starts near the baseline, not
exactly equal; this is an engineering choice, not an empirically optimal value.

### G0: indispensable learned-gate control

Train backbone + gates with the existing translation CE only. No LSR, InfoNCE,
KL, sparsity penalty, EMA teacher or additional data. This isolates a learned
route from its proposed utility supervision.

### G1: candidate with same-policy sequence utility

At a refresh step, choose one layer (cycle through3..12). In two deterministic
no-grad forwards with current weights, force only its gate to0 or1; all other
gates are recomputed using the same current policy in their respective branches.
Use the same source and target prefix and score the complete non-pad target:

\[
\delta_{i,l}=\operatorname{sg}(CE_i(g_l=0)-CE_i(g_l=1)),\quad
q_{i,l}=\mathbf1[\delta_{i,l}>0].
\]

Ignore supervision where |delta|<=.001 nats/token. For remaining examples:

\[
L=L_{CE}+0.1\,\mathbf1[s\bmod10=0]\,
       \operatorname{mean}_{i:|\delta_{i,l}|>.001}
          BCE(g_l(x_i),q_{i,l}).
\]

The event index s is an optimizer step; refresh for all microbatches belonging
to that step, with the same chosen layer. No inverse-frequency rescaling or
stale utility labels. An empty retained set contributes0. These constants are
provisional fixed design choices, not values justified by gradient ratios.
Gate-input features and utility targets are detached in the auxiliary path;
BCE updates gate weights only. CE continues updating backbone and gates.
Inference runs one source-only gated network and never reads target labels.
Teacher dropout flags, RNG and routing/cache state must be restored before CE.
Both G0/G1 must use the same gate architecture and initialization; a compute-
matched G0 runs the same no-grad probes with zero auxiliary coefficient.

This is not selective output KL: it supervises the routing decision. Nevertheless,
that distinction does not resolve the prior overlap listed next.

## 3. Primary-source novelty comparison (2026-10-16)

| Source read | Already present | Difference in this candidate | Decision |
|---|---|---|---|
| [Emelin et al., WMT2019, §2.2, Eqs5–12](https://aclanthology.org/W19-5211.pdf) | Learned gated lexical shortcuts in NMT | Detached distance2 donor rather than embedding K/V highway | Gating a shortcut in MT is not new |
| [GateSkip, §3.1–3.2, Eqs2–5](https://arxiv.org/html/2510.13876v2) | Sigmoid gates on residual branches, CE-based training | Cross-layer encoder donor; sentence gate; no execution budget | Generic learned residual gate is a control, not sufficient novelty |
| [Revisiting Auxiliary Losses for Conditional Depth Routing, arXiv preprint, §3.3 Eqs15–18 and §7](https://arxiv.org/html/2604.17228v1) | Counterfactual downstream CE utility supervises routing; discusses using the current policy in future work | Full target-sequence utility for a cross-layer donor rather than windowed decoder FFN utility; binary labels rather than utility/rank regression | Very close conceptual prior; same-policy rollout alone is not a new principle |
| [SkipCLM, §3.1](https://aclanthology.org/2025.naacl-srw.50.pdf) | Skip connections plus cross-language contrastive supervision | This candidate uses CE/gate supervision and no contrastive objective | Avoid claiming skip-based cross-language adaptation as new |

Official GateSkip code entry point: [author repository](https://github.com/Thiggel/GateSkip).
Formula assessment above is based on the paper, not a claim of reproducing that
code. This is a scoped novelty screen, not an exhaustive absence-of-prior proof.
The conditional-depth result is an arXiv preprint and is treated as such.

**Novelty verdict: NO-GO for promoting G0/G1 as the new main ACL method in this
form.** Applying an existing routing-supervision principle to CLRR/translation
may yield an adapted control or empirical contribution; it has not established
a distinct algorithmic contribution. Positive probe results cannot override this.

## 4. Decisions after the measurements

- Positive ON/OFF heterogeneity supports investigating selective routing at
  fixed weights, not that a source-only gate can learn the oracle choices.
- A tiny oracle CE margin weakens the immediate payoff argument for this
  candidate; it does not prove retraining or other routes cannot work.
- The proposed5 training runs remain unlaunched. A revised candidate must
  resolve novelty and attribution before a training card/config is approved.
- Keep all old method results, including negatives; retain the existing baseline
  controls with their actual protocol. No baseline retraining is needed for these
  measurements.

## 5. Results: all five checkpoints complete

All means below average per-sentence token CE (nats/token), not corpus BLEU.
Positive OFF−ON means the active route helped the frozen checkpoint. Baseline
ON is an artificial intervention, not a separately trained model.

| Dataset/checkpoint | CE all ON | CE all OFF | Mean OFF−ON | ON helps >.001 | ON harms >.001 | Reference-dependent oracle gain over best constant |
|---|---:|---:|---:|---:|---:|---:|
| Amis baseline R1 | 2.894823 | 2.804171 | −.090652 | 36.81% | 63.02% | .107079 |
| Amis CLRR-only P1 | 2.869134 | 3.071728 | +.202594 | 69.44% | 30.56% | .083753 |
| Amis old Full R2 | 2.763204 | 3.003039 | +.239835 | 70.31% | 29.69% | .063528 |
| Rhade baseline42 | 2.108859 | 2.082639 | −.026220 | 36.20% | 57.00% | .114805 |
| Rhade old Full42 | 2.109729 | 2.280262 | +.170533 | 71.90% | 26.70% | .084235 |

Fractions need not sum to100% because |delta|<=.001 is a tolerance/tie region.
Unadjusted descriptive bootstrap CI95 for mean OFF−ON, in the same row order:
[-.123266,−.057788], [.166386,.239956], [.205393,.274222],
[-.050887,−.000973], [.139721,.201936]. No multiplicity-adjusted significance
claim is made from these intervals.

### Findings that constrain the next design

1. **Heterogeneity exists at fixed weights.** Some sentences have lower CE ON,
   others OFF. The reference-dependent oracle margins motivate a question,
   not an achievable performance promise; no source-only predictor was fitted.
2. **Weights and route co-adapt.** All10 individual layer-removal mean utilities
   are positive in each of the three route-trained checkpoints. All10 are
   negative in both baselines with artificially inserted routes. This pattern
   is consistent with co-adaptation; it does not establish a general benefit
   from routing or identify the best architecture to retrain.
3. **No universal early/late-layer deletion rule.** Disabling any single route
   increases mean CE in these route-trained checkpoints. Per-layer effects
   vary across sentences, and cannot be added into a multi-layer mask.
4. **No universal rare-token/UNK rule.** In Rhade baseline, ON helps the <=2-word
   group on average (+.045677 OFF−ON, n326), but harms >10-word sources
   (−.066239, n492). In old Full it helps both (+.180144/+.188765).
   UNK rows favor ON more strongly in Full, but are harmed more strongly by
   inserted routes in baseline. These are selected-validation, unadjusted
   exploratory strata, not a learned/generalizable policy. Amis rare_fraction
   >.25 has only7 rows and <=2 words only3: do not generalize those strata.
5. **CE remains a surrogate.** Amis old Full has lower validation CE than
   baseline here yet its already reported RTX test BLEU is lower. Rhade old
   Full has slightly higher validation CE than baseline but slightly higher
   already reported test BLEU. No new test inference was performed. A utility-
   supervised gate still needs translation-quality evidence, not just CE gain.

### Verification and storage

- 3*576+2*1000=3,728 checkpoint/example pairs;12 conditions each=44,736 CE rows.
- All15 repeat/negative-control/zero-strength checks PASS (max error0).
- Strict archive/model loading, library versions, route/helper hashes and
  unchanged parameter versions PASS; optimizer_steps0 and gradients absent.
- CSV reductions, utility signs, fractions and oracle arithmetic independently
  replayed locally in conda `clrr`: **5/5 PASS**.
- Remote terminal marker: `RESIDUAL_UTILITY_QUEUE_COMPLETE`; process ended.
- Temporary downloaded model/zip files removed; local transfer only15 small
  JSON/CSV artifacts totaling4,133,269 bytes, with SHA256 checks.
- Authoritative artifacts: [measurement workspace](../../outputs/experiments/residual_utility_validation/),
  [independent verification](../../outputs/experiments/residual_utility_validation/LOCAL_VERIFICATION.json).
- Public backup locations (verification receipt: `HF_UPLOAD.json` in that workspace):
  [Amis analysis](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/analysis/amis_mandarin/mbart-large-50/residual-utility/seed-42/rtx6000-fp32)
  and [Rhade analysis](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/analysis/rhade_vietnamese/mbart-large-50/residual-utility/seed-42/rtx6000-fp32-src-id).

## 6. Combined decision after steps1–2

**Mechanistic motivation: qualified GO for studying sentence-dependent route
utility. Novelty: NO-GO for the generic learned/CE-supervised gate as the main
new method. Training: HOLD; zero new training runs opened.**

The evidence does not justify automatic layer deletion, a rare-token heuristic,
or claiming that gate training will recover the oracle margin. The next design
must add a substantively distinct mechanism/formulation and a decisive matched
control; otherwise this gate remains an adapted comparator. Do not launch the
previously proposed5 pilot runs on the strength of CE oracle heterogeneity alone.
