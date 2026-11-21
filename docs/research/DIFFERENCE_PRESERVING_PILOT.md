## CURRENT STATUS — STOPPED FOR NOVELTY REVIEW (2026-11-09)

Author corrected the required sequence: design → related-work/novelty review →
only if sufficiently justified, smoke/correctness gates → only after all gates
PASS, long training. Translation quality cannot substitute for novelty review.
Do not run smoke, resume or launch a candidate whose novelty is UNESTABLISHED/HOLD.
A focused prior list or different implementation alone does not clear this gate.

Relational v2 queue PID170990 was stopped; all five queue processes exited.
Only A had begun long training. Existing A checkpoint-36 is preserved; B/C/D
have no long-training results. This interrupted run is not a completed result.
No other user's process was signalled. Previous launch/PASS entries below are
historical; they do not authorize resumption. Current candidate is NOT admitted.

Before proposing a candidate again, document primary-paper method sections,
formula/mechanism overlap, the concrete remaining contribution, evidence and
uncertainty. If the distinction is insufficient, reject/redesign before GPU use.

---

# Difference-preserving LSR: implementation and scientific admission

**Current v2 execution, 2026-11-09:** author authorized the four-arm
all-training relational screen in section18. Wrapper/runner and CPU tests are
implemented; real GPU gates and training status are recorded separately.
The old reviewed-near-pair v1 HOLD remains unchanged. Novelty is unestablished.

**Decision update 2026-11-09 (section14): NO-GO for the current R0–R3 long-training
queue as a main-method candidate.** Technical checks remain8/8 PASS. The focused
prior audit does not establish a strong contribution; semantic correspondence
is unverified, and a new Unicode mining defect was identified for Ede. This is
a design-readiness decision, not a measured failure of an untrained method.
No training, recipe, source implementation or historical score is changed by
this review. Earlier HOLD statements below are historical context.

Date: 2026-11-07. Author authorized the staged plan. **Current status: local
prototype implemented; local preparation completed; semantic and novelty gates
HOLD. On November9 the author authorized RTX technical checks; all8 full-size
technical cases completed PASS (section13). No long training is authorized
by these technical receipts.** Correctness PASS must not be interpreted as
quality PASS. The earlier local-only preparation is recorded in section12.

## 1. What is implemented

Isolated package: `src/methods/difference_preserving/v1/`.
Historical CLRR/LSR implementations and stored scores are not edited.
The common runner recognizes four explicit method IDs and preserves the
existing optimizer, data order, CE microbatch reduction and generation path.

Let `z` be the final encoder content-token masked mean, without L2 normalization
before subtraction. All tokenizer special IDs are excluded from pooling.
For anchor/peer parallel pairs `(xi,yi)` and `(xj,yj)`:

```
dx = z(xi) - z(xj)
dy = stop_gradient(z(yi) - z(yj))
Ldiff = mean_valid[1 - cosine(dx, dy)]
L = LCE(original anchor microbatch) + 0.1 Ldiff
```

Target differences with norm <=1e-6 are excluded; a batch without valid pairs
returns zero auxiliary loss. Small source differences are not hidden by
masking. Their gradients can be large: there is no anti-collapse guarantee.
The target is the current shared encoder, not a fixed/EMA teacher.
Original LSR weight is zero. No added parameters or inference modules.
CLRR-on retains the original encoder route, distance2/strength0.1/detached
post-rewiring cache. The prototype does not silently use the manuscript's
embedding-donor boundary convention.

Auxiliary source/peer passes are differentiable; both target passes are detached.
Encoder dropout/checkpointing is disabled only inside auxiliary views, through
the existing deterministic-view context. CE retains its original settings and
RNG sequence. Up to four extra encoder passes are needed for eligible anchors:
this is not compute-free. Diagnostics distinguish eligible pairs, nonzero
target differences and small source differences.

## 2. Actual train-only feasibility results

Discovery rule: normalized source word sequences of length3..50 differ at
exactly one word position, with distinct nonempty original target strings.
Normalization is for discovery only; original CSV files remain unchanged.
No validation/test sentences or model weights were read for pair discovery.

| Dataset | Train rows | Lexical candidate pairs | Covered train rows | Coverage | Semantically verified |
|---|---:|---:|---:|---:|---:|
| Amis–Mandarin | 4600 | 165 | 193 | 4.20% | 0 |
| Ê Đê–Vietnamese | 15092 | 2460 | 1461 | 9.68% | 0 |

Artifacts: `outputs/experiments/difference_preserving_feasibility/<dataset>/`:
`PAIR_AUDIT.json`, `lexical_candidates.jsonl`, `PAIR_REVIEW.csv`,
`reviewed_pairs.json`. The review CSV contains up to100 deterministic sampled
pairs; labels are blank, not inferred. `reviewed_pairs.json` is deliberately
empty/PENDING. To authorize a specific pair, its review must identify the
meaning difference, translation correspondence and reviewer. A sample audit
does not automatically verify every unreviewed candidate.

These counts establish lexical availability, not usable semantic coverage.
Some Amis examples have nearly synonymous targets (hot/very hot), while others
change several target details. One changed source word is not proof of a single
grammatical contrast. A bilingual review is required before claiming that.
Low coverage is a risk, not a proof that this objective will fail.

## 3. Related-work audit and decision

This is a targeted audit of primary method sections, not an exhaustive novelty
certification. Some publisher PDF requests failed; accessible full-text versions
are listed below.

| Prior | Overlap | Actual distinction / unresolved issue |
|---|---|---|
| [RKD, CVPR2019, §3.2](https://arxiv.org/html/1904.05068) | Learning relations between examples | RKD's distance/angle distillation is not this direct bilingual difference-direction cosine. The general relational idea is prior art. |
| [CCPL, ECCV2022, §3.1 Eq2–3](https://arxiv.org/html/2207.04808) | Builds and aligns difference vectors | Uses image feature differences, MLP and InfoNCE. Difference-vector preservation is not a new primitive here. |
| [DiffCSE, NAACL2022, method](https://aclanthology.org/2022.naacl-main.311.pdf) | Sentence embeddings sensitive to edits | Uses generated edits and conditional token-change prediction. Sensitivity to meaning-changing edits alone is not a new claim. |

**Decision: HOLD main-method novelty and long training.** There is a concrete
implementation difference, but no established reason that selecting natural
near-pairs plus CLRR is a sufficient research contribution. Generic and candidate
arms below use the SAME proposed loss; R0/R2 must not be presented as exact RKD
reproductions. No new superiority or morphology claim is supported.

Next scientific work: verify a bilingual sample; identify whether the proposed
contrasts diagnose actual baseline/Full translation errors; then decide whether
a specific task-conditioned objective is needed beyond this generic prototype.
Do not mark novelty PASS merely because code differs or a score rises.

## 4. Four-arm pilot prepared, not launched

| ID | Method ID | CLRR | Peer selection |
|---|---|---|---|
| R0_S42 | difference-generic-v1 | off | deterministic generic control |
| R1_S42 | difference-candidate-v1 | off | reviewed natural near-pair |
| R2_S42 | difference-clrr-generic-v1 | on | deterministic generic control |
| R3_S42 | difference-clrr-candidate-v1 | on | reviewed natural near-pair |

Both selection modes use exactly the same eligible anchors. Candidate peers
are selected deterministically from reviewed pairs; generic peers exclude self,
the candidate peer and identical source/target strings. Original CE rows and
sampling remain unchanged. Token lengths/extra encoder FLOPs can differ and
must be measured rather than described as matched wall time.

Recipes:
`configs/experiments/amis_difference_preserving_rtx.json` and
`configs/experiments/rhade_difference_preserving_rtx.json`.
Each derives only from the corresponding locked protocol, not from a DRB recipe
with inherited candidate names. Native FP32/SDPA/TF32off, LR5e-5, B4/GA32,
max20/ES10, validation BLEU beam1, original references; no test.
The weight0.1 is inherited as a single fixed exploratory choice, not justified
by comparing raw gradient norms across different losses. Section12 now fixes
exploratory advancement thresholds in the local-review snapshots; this does
not clear scientific admission or authorize execution.

Primary mechanistic comparison R3−R1; selection comparisons R3−R2 and R1−R0.
Baseline superiority also requires a receipt-verified matching BLEU-selected
RTX baseline. No historical/A100/eager score is silently substituted.

The full staged ceiling remains4 Amis seed42 +8 Amis seeds43/44 +4 Ê Đê seed42
+8 Ê Đê seeds43/44 =24 new candidate/control trainings, only if earlier gates
pass; missing matching baseline runs would be additional. There are currently
zero new trained models from this proposal.

## 5. Reproduction and admission rules

Use the existing conda `clrr`, including on the authorized remote project.

```powershell
conda run -n clrr python -m unittest discover -s tests -p test_difference_preserving.py -v
conda run -n clrr python scripts/experiment.py build --config configs/experiments/amis_difference_preserving_rtx.json
conda run -n clrr python scripts/experiment.py cpu --config configs/experiments/amis_difference_preserving_rtx.json
```

Pair discovery, choose a fresh output directory (existing audits are protected):

```powershell
conda run -n clrr python scripts/analysis/audit_difference_pairs.py --dataset amis_mandarin --output outputs/experiments/difference_pairs_new_audit
```

The prepared recipe pins a PENDING pair manifest so CPU correctness can be
checked without inventing labels. `read_pairs` rejects it for real training data.
RTX launcher, GPU gate and training independently reject scientific HOLD.
Real GPU gates must include eligible examples, use B4/GA32 and perform optimizer,
export/reload, beam-cache and optimizer/scheduler/RNG-resume checks. CPU tests use
tiny random models and synthetic peers; they do not replace those GPU gates.

After genuine semantic/novelty review, prepare a NEW frozen recipe/workspace
with the reviewed pair manifest/hash, locked quality gates and audited baseline.
Do not edit existing frozen manifests/receipts to make them appear to pass.
Then use the common launch/monitor commands, not a new ad hoc training worker.

## 6. RTX connectivity

Read-only attempt on2026-11-07: sandbox denied Tailscale's named pipe; the retry
with elevated execution reached Tailscale but SSH returned502/dial failure to
100.90.53.12:22. VRAM is unknown. No remote gate/training was launched, no source
was deployed and no other user's process was touched. A connection failure does
not prove the remote machine is powered off.

## 7. What remains before any training

1. Resolve semantic-pair verification and scientifically sufficient novelty.
2. Complete frozen-checkpoint error/contrast diagnostics; no evidence that this
   particular loss fixes an observed translation failure has been established.
3. Verify the matching baseline receipt and lock practical pilot thresholds.
4. Restore RTX access, report free VRAM, and pass all GPU gates before training.

This work is preparation and feasibility analysis, not a successful method
upgrade. Existing experimental numbers and the paper are not rewritten.

## 8. Completed local verification

Both frozen bundles completed the shared `cpu` entry point:13 unique unit
tests PASS (repeated for each bundle), plus four synthetic R0–R3 cases per
dataset. Assertions verified that `rtx.launch`, `engine.gpu_gate` and
`engine.train_run` all reject scientific HOLD before attempting GPU training.
The existing two DRB configuration files still validate. No remote GPU PASS,
real optimizer training or generation-quality result is claimed.

Receipt:
`outputs/experiments/difference_preserving_feasibility/IMPLEMENTATION_VERIFICATION.json`.
Per-bundle receipts: `outputs/experiments/{amis,rhade}_difference_preserving_rtx/CPU_GATE.json`.

| Frozen bundle | SHA256 |
|---|---|
| Amis | f283efbe8e41575ce52024753ed2a1b454b63eb58987158b8b5d69857e0437d3 |
| Ê Đê | db89383fd9b761514ff4f7d5866bdecbc7ee39e90931711c92a6850968d60212 |


## 9. Corrective verification and dataset decision — 2026-11-07

Checkpoint metadata now pins the special-token exclusion list and optional
CLRR distance/strength/stack. Reload verification runs after tokenizer/language
configuration; explicit training resume validates the same metadata. A mismatch
fails rather than silently restoring a different objective.

15 unique difference tests PASS, including a real CPU ControlTrainer test:
accumulation4 + reentrant checkpointing, interrupted at optimizer step2 and
resumed to step3 with the same schedule. Final weights match uninterrupted
training bitwise for CLRR on/off. Five original-reference metric tests and three
checkpoint-selection tests also PASS. This does not replace full-size GPU gates.

New frozen CPU-verified preparations:
- `configs/experiments/amis_difference_preserving_rtx_preflight_v2.json`
- `configs/experiments/rhade_difference_preserving_rtx_preflight_v2.json`

The original preparation bundles and receipts are retained as historical;
their source hashes predate these fixes. Use preflight_v2 for the current code.
No GPU training has started. Latest read-only RTX retry again returned SSH502,
port22 timeout; VRAM remains unknown. Receipt:
`outputs/experiments/difference_preserving_feasibility/FIX_VERIFICATION.json`.

**All training rows remain in CE.** Peer discovery only determines availability
of the auxiliary difference loss; it is not a smaller translation training set.
No oversampling or filtering of original training data is introduced.
Before semantic review, lexical coverage implies about15.76% of random B4
microbatches contain an eligible Amis row versus33.46% for Ede (sampling without
replacement). These are analytic upper bounds from candidate rows, not measured
optimizer activity or semantic validity. Degenerate pooled differences can
further reduce actual auxiliary activity.

Prefer Ede for the next pair-feasibility review, because more lexical candidates
are available; this does not establish that Amis is unsuitable or Ede yields
better translation. Neither dataset currently has verified semantic pairs.

## 10. Related-work check and ACL generality requirement — 2026-11-07

Read the method/formula sections of the primary sources below, not only their
abstracts. This is a focused overlap audit, not an exhaustive novelty guarantee.

| Prior | What already exists | Difference of this prototype / implication |
|---|---|---|
| [RKD, CVPR2019](https://arxiv.org/html/1904.05068), sections3.2.1–3.2.2, Eq5–10 | Teacher/student normalized pair distances and triplet angles | Our loss matches corresponding bilingual difference directions. It is not an exact RKD reproduction, but preserving relations is established. |
| [CCPL, ECCV2022](https://arxiv.org/html/2207.04808), section3.1, Eq2–3 | Subtracted feature vectors, MLP normalization, InfoNCE between corresponding differences | Direct precedent for difference preservation; moving the principle into MT alone is not sufficient evidence of a strong contribution. |
| [DiffCSE, NAACL2022](https://aclanthology.org/2022.naacl-main.311.pdf), section3 | SimCSE plus conditional replaced-token detection from edited sentences | Sensitivity to meaning-changing edits has NLP prior; our bilingual direction loss is a different objective, not a new general motivation. |
| [MMKD, EMNLP2022](https://aclanthology.org/2022.emnlp-main.202.pdf), section3.4, Eq3–5 | Cross-lingual structure alignment via relational similarity distributions and KL | Cross-lingual relation preservation already exists. Our shared evolving encoder/no extra heads differs, but requires a substantive translation-specific justification. |
| [DAP,2023](https://arxiv.org/html/2305.09148), section3.3.2, Eq6–7 | Token reconstruction from the other language plus sentence ranking | Local information preservation alongside global alignment is established; retrieval results do not themselves establish generation benefits. |

CL2CM also appeared in the search as relevant cross-lingual relational KD;
its publisher full text was not fully verified here. Do not mark that review
complete or claim the search found no overlapping work.

**Decision:** retain scientific HOLD. Code correctness and a different equation
are insufficient to promote this prototype to the paper's main method.
The current one-word/whitespace-based miner is a feasibility instrument; it
cannot support a claim of language-independent applicability, especially for
scripts without explicit word boundaries. Two successful Austronesian source
languages would show transfer between those settings, not unrestricted typology.

Before choosing a final method:
1. Connect the proposed operation to a measured baseline translation failure.
2. Define a reusable training mechanism and pair/coverage policy without
   language-specific affix rules or manual per-language tuning. Any mechanism
   change needs a new candidate definition; do not silently alter this version.
3. Distinguish against nearest prior mathematically and with matched controls;
   R0–R3 alone do not rule out a generic relational-regularization explanation.
4. Lock pilot endpoints, cost budget and failure rules before validation runs.
   After a candidate passes development, verify multiple seeds and transfer to
   another language/backbone with matching baselines. Treat already-inspected
   datasets as development evidence rather than a fresh independent holdout.
5. Evaluate free generation BLEU/chrF++ and a targeted error analysis. Better
   embedding geometry alone does not establish better translation.

ACL remains the target; neither novelty clearance nor acceptance is guaranteed.
No new long training is opened while these decisions remain unresolved.

## 11. Grounding in the author's manuscript and verified evidence — 2026-11-07

Read `C:/Users/Lenovo/Downloads/53-2 (1).pdf` (15 pages), especially sections
3.2–3.3. PDF SHA256:
`e6564587846f3d8fbdd7b5d985804d5b1d219c2aeb3d246e5609f1fa8814c504`.
The foundation is detached cross-layer residual transport plus
reference-guided latent alignment. The target encoder shares the evolving
parameters; it is not a frozen pretrained teacher or EMA teacher. Historical
manuscript scores do not establish the new candidate's quality.

The manuscript permits an embedding donor at the first distance boundary;
the historical implementation hooks block outputs and starts its distance2
donor at block3. Preserve that implemented convention in matched experiments.
Do not silently make historical scores correspond to a different formula.

### Completed CPU evidence synthesis

Reused 12 already-measured checkpoints (baseline/NCE/P2/P3, seeds42–44), not
12 new training runs. Verified all12 RESULT.json byte counts and SHA256 against
the existing HF receipt at revision
`224d68f0fe7ed8bc4c21298858bc0169818a9427`. Replayed gradient norms/cosines,
loss-decomposition checks, unchanged-parameter flags and the same four batch
index lists across all12 models. Region summaries combine gradient dot
products and squared norms across disjoint coordinates before calculating
cosines, rather than averaging layer cosines as if they were one gradient.

Artifacts:
`outputs/experiments/amis_contrastive_multiseed_evidence/lsr_compatibility/`
contains `RESULT.json`, `SUMMARY.csv`, `REPORT.md`.

Reproduce in conda clrr:

```powershell
python scripts/analysis/summarize_contrastive_diagnostics.py --input outputs/experiments/amis_contrastive_multiseed_evidence --verified-cohort
```

| Probe family | Seeds | Mean CE–LSR cosine range across checkpoints | Negative batches, pooled count |
|---|---|---:|---:|
| Baseline, counterfactual LSR | 42–44 | 0.0676–0.0954 | 0/12 |
| NCE, counterfactual LSR | 42–44 | 0.0970–0.1245 | 1/12 |
| P2, actual LSR coefficient0.1 | 42–44 | 0.0947–0.1232 | 0/12 |
| P3, actual LSR coefficient0.1 | 42–44 | 0.0749–0.1056 | 0/12 |

The pooled counts are descriptive repeated batches, not independent seeds.
Both auxiliary objectives are present in P2/P3: a CE–LSR cosine is not the
whole optimizer direction. Probes use weighted0.1LSR, original nonpadding
pooling (including specials), selected checkpoints only. They do not show the
training trajectory or accumulated Adam update. No new GPU forward/backward
was run for this synthesis.

Verification: existing `test_contrastive_diagnostics.py` completed2/2 PASS
in conda clrr; cohort replay completed PASS for all12 inputs. These checks
validate diagnostic arithmetic/decomposition, not a new method's quality.

**Decision:** these observations do not support pervasive gradient conflict
as the explanation for old Full's failure. Do not choose gradient surgery or
increase lambda merely because the instantaneous norm ratio is small. They
also do not prove LSR harmless, useful, or causal. The distinction-preserving
hypothesis remains unconfirmed; it is not established by these gradients.

### Additional nearest prior

[Veitsman et al., MRL2026](https://aclanthology.org/2026.mrl-1.10/),
"Why Better Cross-Lingual Alignment Fails for Better Cross-Lingual Transfer:
Case of Encoders," already studies alignment/task gradient compatibility.
Read the full-text gradient methodology/results (sections3.2.1 and4.3).
Their experiments concern encoder-only XLM-R and classification/POS transfer;
our setting is encoder-decoder translation. That distinction motivates an MT
study but does not make the generic alignment-versus-task argument novel.

### Ordered continuation and stopping rules

| Stage | Actual work | Admission / result interpretation |
|---|---|---|
| A — completed | Read original CLRR/LSR definitions, audit nearest priors, replay12 existing gradient probes | Excludes an unsupported conflict-based redesign; no new quality claim |
| B — required before selection | Establish usable bilingual difference pairs and targeted translation distinctions; otherwise redesign the pairing mechanism with a new explicit candidate card | Whitespace one-word candidates alone do not establish semantic contrasts or generality. No fabricated review labels |
| C — required before pilot | Specify why the new objective differs substantively from relational KD/CCPL and why it should improve a measured MT error; lock controls and cost | Current R0–R3 prototype remains a research instrument until this is satisfied |
| D — technical gates | Restore RTX access, report free VRAM, then full-size forward/backward, accumulation, checkpointing and resume checks on the frozen bundle | CPU PASS cannot waive GPU or scientific gates; never disturb other users' jobs |
| E — conditional pilot | Four seed42 arms R0–R3 from section1, all original training rows retained, matched frozen recipe, selector validationBLEU/ES10 | Validation-only development; compare the candidate against its generic-pair control and CLRR-on/off interaction, plus protocol-matched existing baseline. No automatic test/A100 expansion |
| F — conditional confirmation | Replicate a pilot that passes its predeclared practical gates, then transfer with matched controls | A single seed, higher cosine, or a differently named equation is insufficient for an ACL contribution |

No additional method is selected merely to occupy the GPU. This is a staged
continuation of the author's CLRR/LSR research question, not a claim that the
current prototype has solved it. Amis and Ede already inspected remain
development settings. Existing paper scores and baseline protocol stay intact.

## 12. Local preparation completed; RTX explicitly deferred — 2026-11-07

### Complete train-only structure audit

The audit now describes every discovered candidate, rather than only the
100 sampled review rows. It reports normalized character edit spans and
unchanged raw references, never semantic approvals. This supplements, not
replaces, the original review sheet.

| Dataset | Candidate pairs | One target character edit span | Multiple target edit spans | Semantically verified |
|---|---:|---:|---:|---:|
| Amis | 165 | 69 | 96 | 0 |
| Ede | 2460 | 1457 | 1003 | 0 |

No target pair becomes identical after the audit's Unicode/case normalization.
Multiple spans can still represent one semantic change (e.g. Vietnamese
"chăn" to "ngô"); one span can still change several concepts. Do not use this
table to label correctness, filter data, rank languages or declare a successful
semantic audit. No training/validation/test CSV was modified. Only train rows
were used for this structure analysis.

Artifacts in `outputs/experiments/difference_preserving_feasibility/`:
`target_structure_amis/` and `target_structure_rhade/` each contain
`TARGET_STRUCTURE.jsonl`, `TARGET_STRUCTURE_AUDIT.json` and
`TARGET_STRUCTURE_REPORT.md`. Reusable command:

```powershell
conda run -n clrr python scripts/analysis/audit_difference_pairs.py --dataset rhade_vietnamese --target-structure --output outputs/experiments/difference_preserving_feasibility/new_target_structure_audit
```

**Unresolved:** neither spelling checks nor target-language reading verifies
the source-side meaning in Amis/Ede. A qualified bilingual review remains
necessary for the current semantic-pair candidate. The assistant has not
invented reviewer identities or marked these pairs PASS. This is the specific
reason step1 is not a completed semantic gate.

### Formula/gradient review

For an accepted nondegenerate pair, define `u=dx/||dx||`, `v=sg(dy/||dy||)`.
Above the epsilon clamp, its loss is `1-u·v` and its source derivatives are:

```
dL/dz(xi) = -(I - u u^T) v / ||dx||
dL/dz(xj) = -dL/dz(xi)
dL/dz(yi) = dL/dz(yj) = 0
```

Average across valid pairs and multiply by0.1 in the training objective.
Epsilon-clamped derivatives follow PyTorch cosine_similarity, not the
above unclamped formula. Source collapse can produce gradients proportional
to1/epsilon; finite does not mean well-conditioned. Pair swapping and adding
a shared offset to source or target vectors leave this objective unchanged.
This removes shared offsets from this loss, not necessarily harmful
information from the model. It supplies no theorem of anti-collapse,
translation improvement, or preservation of all semantic distinctions.

CLRR remains an optional unchanged residual route. LSR's reference-guided
principle is retained through differences; the original mean-cosine LSR term
is replaced, not secretly added as another tuned objective. Four arms can
test pair selection and CLRR interaction; they cannot establish that this
loss is better than original LSR unless an original-LSR checkpoint with
matching data/runtime/selector/seed is separately verified or rerun later.

### Closed related-work gap

Read the accessible [CL2CM full text](https://arxiv.org/html/2312.08984),
Methods, especially Eq14–15. It combines sentence/word relation scores and
distills softmax similarity distributions with KL from a cross-lingual network
to a vision-language network. Its target is retrieval rather than translation.
Our direct direction-cosine between bilingual differences, shared encoder,
and optional CLRR differ, but relational cross-lingual transfer is already
prior art. The publisher endpoint was blocked; arXivv2 was the full text read.
This closes the earlier unread-method gap, not the main-method novelty gate.

**Novelty conclusion:** technically distinct prototype; insufficient evidence
to present the current combination as a strong ACL method contribution.
Do not use successful code tests, an isolated positive seed, or the original
paper's claims as novelty clearance. No exhaustive priority guarantee is made.

### Locked pilot decisions, not permission to run

Prepared separate local-review snapshots for both datasets:

- `configs/experiments/amis_difference_preserving_local_review.json`
- `configs/experiments/rhade_difference_preserving_local_review.json`

They preserve existing FP32/SDPA/B4/GA32, LR5e-5, maximum20epochs,
ES10 and original-reference validationBLEU beam1. R0–R3 remain seed42;
all original CE rows remain. Scientific gates remain HOLD. The recorded
`execution_authorization` is a user constraint, not a new runtime guard;
no launch command has been issued. Existing admission guards still reject
these scientific HOLD snapshots.

| Decision | Rule fixed before new training |
|---|---|
| Primary pairing effect | R3−R2 validationBLEU >=+0.30 |
| Baseline improvement | R3−receipt-verified matching baseline validationBLEU >=+0.30 |
| CLRR contribution | R3−R1 validationBLEU >0 |
| Secondary guardrail | chrF++ delta >=0 for all three required comparisons |
| Missing baseline provenance | HOLD; no A100/eager/other-selector substitute |
| Numerical correctness | Finite losses/gradients, verified checkpoint reload/resume; eligible and valid-pair activity logged |
| Cost/coverage reporting | Report auxiliary-active batches, valid target differences, near-zero source differences, extra token/FLOP/runtime cost; same eligible anchors does not imply equal valid-pair counts or wall time |
| Pilot success | Advances only to replication review; does not establish statistical significance |
| Pilot failure | NO-GO current recipe; no automatic lambda sweep, dataset switch, or test inspection |

These are practical development thresholds, not power-derived significance
cutoffs. Seed42 comparisons alone cannot establish a seed-stable method.
Any future paired bootstrap resamples the original validation sentences;
10000replicates do not mean10000new examples. Report exploratory intervals
and every declared comparison. After replication, use paired seed deltas as
well as sentence resampling; three seeds have limited inferential resolution.
No automatic test, A100 or additional-seed queue is authorized by this file.

### Local checks and receipts

CPU tests rerun in conda clrr with CUDA hidden:15 difference-preserving tests,
5 original-reference metric tests and3 checkpoint-selection tests PASS.
This includes synthetic optimizer steps and interrupted/resumed tiny models;
it is not training a full-size translation model. No GPU or remote resource
was used. Original code paths, dataset hashes and historical score files
remain unchanged by this preparation.

Both local-review bundles were frozen and their source/configuration manifests
verified locally. Receipt and exact bundle hashes:
`outputs/experiments/difference_preserving_feasibility/LOCAL_PREPARATION.json`.
Frozen bundle validity applies to that source snapshot; source changes require
a fresh build, not overwriting earlier receipts. Full-size GPU gates are
deferred by the author's explicit instruction, not reported as PASS.

**Current next dependency:** actual bilingual-pair evidence and a defensible
method contribution. Everything that can be verified by string structure,
local synthetic tests and a focused prior audit has been prepared. No claim
is made that these checks substitute for the missing scientific evidence.

## 13. RTX technical-only validation — completed 2026-11-09

User subsequently requested deploying and completing technical checks. Prior
local-only instruction is superseded for these checks, not for long training.
Admission: Quadro RTX6000,23952MiB free,0% utilization,77.87GiB disk free;
project training lock available. No other user's process was stopped.

Shared CLI now exposes explicit `technical-suite` / `technical-gate` stages.
The latter uses arbitrary cyclic peers from original train rows, solely to
stress the auxiliary loss. It does not forge a reviewed-pair manifest or
change scientific HOLD. All auxiliary-eligible fixtures stress a denser path
than expected sparse real training. Their losses are not translation scores.

The test reuses full-size model loading, recipe FP32/SDPA/B4/GA32 and maximum
lengths. Each case runs2 optimizer steps, reloads the selected step2 checkpoint,
exports/reloads weights, checks beam4 cached versus uncached generation and
repeats step2 from saved step1 optimizer/scheduler/RNG. Existing tolerances
remain unchanged. Technical transient checkpoints are removed after success;
they are not research checkpoints and are not uploaded to HF.

Receipts are written exclusively to
`technical_gates/<id>/TECHNICAL_PASS.json`, never `gates/<id>/PASS.json`.
They cannot satisfy the normal training gate path. Normal `gpu_gate` and
`train_run` still reject scientific HOLD. Added a CPU regression test proving
this distinction; that test and the existing15 method tests PASS locally.

Configs:
- `configs/experiments/amis_difference_preserving_gpu_technical.json`
- `configs/experiments/rhade_difference_preserving_gpu_technical.json`

Remote suite:
`/workspace/ura/thanhdat/projects/CLRR/outputs/experiments/amis_difference_preserving_gpu_technical_suite/`.
Initial queue PID11153; local launch receipt in the corresponding output
directory records both frozen bundle hashes. Eight cases, R0–R3 for each
dataset, sequential under the project lock. No test-set evaluation, no
training continuation and no automatic replication follow the queue.

**Final verification:8/8 TECHNICAL_PASS receipts collected and matched to
the corresponding frozen source manifests.** The queue exited with
`TECHNICAL_SUITE_COMPLETE_NO_TRAINING`. No transient checkpoint/export/
resume directories or `.pt`/`.bin`/`.safetensors` files remain in the scoped
technical gate folders. Disk free77.85GiB; GPU0%,23952MiB free after exit.
No long training, held-out generation or upload was performed.

| Dataset | Cases | Result | Peak allocated GiB | Per-case gate seconds |
|---|---|---|---:|---:|
| Amis | R0–R3 | 4/4 PASS | 14.41 | 186.9–203.6 |
| Ede | R0–R3 | 4/4 PASS | 15.16 | 189.9–194.2 |

These times exclude some process/model startup overhead and are not full
training speed estimates. Save/reload/resume comparisons retained original
rtol1e-5/atol2e-5; GPU results are not claimed bitwise identical.
Verification/report:
`outputs/experiments/amis_difference_preserving_gpu_technical_suite/VERIFICATION.json`
and `REPORT.md`. No new translation-quality score is available. Scientific
pairing/novelty HOLD remains; no normal training-admission PASS was emitted.

Additional tokenizer audit: Transformers4.57.6 emits a Mistral regex warning
when reading these local mBART configs. The actual tokenizer is Unigram with
Metaspace, not a Mistral regex tokenizer. No regex correction was applied.
Source and target token IDs match on all4600 Amis and15092 Ede train rows
between the original tokenizer and a reloaded Amis checkpoint tokenizer,
configured separately for each language pair. Normalizer/pre-tokenizer/
post-processor/decoder/added-token structures match after language setup.
Vocabulary score serialization differs by at most3.55e-15 (equal after
float32 rounding), so raw backend JSON is not byte-identical. The initial
strict comparison failure is retained remotely as
`TOKENIZER_STRICT_BACKEND_COMPARE.json`; the refined scope/result is
`TOKENIZER_RELOAD_AUDIT.json`. This verifies observed train tokenization,
not universal equivalence for arbitrary strings or quality on held-out data.

## 14. Scientific decision review — 2026-11-09

Scope: author requested steps1–3 (related work, pair evidence, GO/NO-GO).
Read the isolated loss/model/data/pair code and both100-row train-pair review
samples. Recomputed train-only graph/tokenization audits in conda clrr.
No GPU, model forward, validation/test reading, remote deployment or training.
New small artifacts, preserving earlier receipts:
`outputs/experiments/difference_preserving_feasibility/novelty_decision_20261109/PAIR_IMPLEMENTATION_AUDIT.json`
and `UNICODE_MINING_AUDIT.json`.

### 14.1 What is inherited, different, and still unsupported

Primary full texts consulted again, including method equations:

| Prior | Established mechanism | Distinction of this implementation | Consequence |
|---|---|---|---|
| [RKD, CVPR2019, section3.2 Eq4–10](https://arxiv.org/html/1904.05068) | Transfers pair distances and triplet angles between teacher/student example representations | Direct cosine between source and target pair-difference vectors, with an evolving shared encoder | Not an exact RKD implementation; relational supervision itself is prior art |
| [CCPL, ECCV2022, section3.1 Eq2–3](https://arxiv.org/html/2207.04808) | Subtracts neighboring image feature vectors, projects/normalizes differences and contrasts matching differences with InfoNCE | Natural bilingual sentence peers; direct direction cosine without projector or negatives; optional CLRR | The difference-vector primitive and preservation idea cannot be claimed as new |
| [DiffCSE, NAACL2022, section3](https://arxiv.org/html/2204.10298) | Conditional replaced-token detection makes sentence embeddings informative about generated edits, alongside contrastive learning | No generated edits, generator or discriminator; natural parallel pairs and encoder difference alignment | Sensitivity to sentence edits is prior motivation, not this implementation's novel claim |
| [CL2CM, AAAI2024, Cross-lingual Knowledge Transfer Eq14–15](https://arxiv.org/html/2312.08984) | Distills cross-lingual sentence/word similarity relations into a cross-modal retrieval model with KL | No vision branch or relation-distribution KL; translation CE plus shared-encoder directional pair loss | Cross-lingual relational transfer already exists; the exact MT construction differs |

These are targeted comparisons, not an exhaustive proof of priority. Broader
searches for bilingual difference vectors and relational NMT did not provide
a verified exact duplicate in the inspected texts. Absence of a found duplicate
is not novelty clearance. A new search lead about token-relation graphs in an
IJCAI2024 KD paper is not treated as a verified exact overlap or new comparator.

The defensible prospective question is: **does reference-guided supervision of
local bilingual changes improve translation beyond point alignment and generic
relation regularization, and does CLRR add anything?** This is a research question,
not an established answer or automatically sufficient ACL contribution.
Combining existing ingredients or changing the task alone does not establish
the explanatory/empirical contribution needed here.

The loss is invariant to separate shared offsets and to positive global scaling
of source or target representations (away from epsilon). This follows directly
from subtraction and cosine; it is not a new theorem of semantic preservation.
It aligns directions, not their magnitudes, and does not prevent collapse.
Detached target parameters still evolve through CE and later training steps.
Consequently its targets are neither a stable teacher nor verified meanings.
No existing measurement demonstrates that this specific loss repairs a baseline
translation error. Earlier selected-checkpoint CE–LSR gradient probes did not
show pervasive conflict, so gradient-conflict repair is not a supported rationale.

### 14.2 Sample inspection: evidence and limits

Both existing deterministic100-pair CSV samples were inspected, not labeled as
bilingual expert reviews. The following row IDs are zero-based train indices.
These examples establish observable reference issues; source correctness stays
unverified. No failure percentage is estimated from these illustrative cases.

| Dataset / rows | Observation from original references | What this does and does not support |
|---|---|---|
| Amis432/2463 | Mandarin references describe very hot / hot weather | Could be degree or near-synonymy; not evidence of a large semantic contrast |
| Amis3183/3187 | Both references describe many summer typhoons with different wording | Surface difference need not be a meaningful semantic change |
| Amis554/2212 | References change occupation and gender wording | A single source token substitution does not isolate a single target feature |
| Amis3371/3435 | One reference ends at the fragment `傷口若不擦藥會` | Potential truncation/referential quality issue; cannot certify a clean semantic pair |
| Ede3160/11319 | `Em tôi đang uống rượu.` / `Bà tôi đang uống rượu.` | Clear target-side kinship contrast; Ede correspondence still needs verification |
| Ede13038/15073 | `Chúng ta phải sửa ớt.` / `Chúng ta phải sửa gỗ.` | Grammatically comparable but unusual target content; not proof the translations are wrong or natural semantic contrasts |
| Ede2707/11520 | Long near-duplicate religious references differ in wording, punctuation and clause detail | One lexical substitution does not guarantee a localized target edit |

**Verified bilingual semantic pairs remain0 for both datasets.** Do not turn
these observations into reviewed-pair PASS labels. A sample review cannot
authorize all unreviewed edges, and a target-language judgment cannot establish
source-to-target correspondence. No CSV data was edited or removed.

### 14.3 New implementation/coverage findings

| Quantity | Amis | Ede |
|---|---:|---:|
| Existing lexical candidate edges | 165 | 2460 |
| Covered original train rows | 193/4600 (4.20%) | 1461/15092 (9.68%) |
| Distinct edges used by current minimum-row-index peer rule, IF all candidates were admitted | 109 | 1031 |
| Candidate edges not selected under that hypothetical rule | 56 | 1429 |
| Maximum anchors selecting the same peer under that rule | 8 | 12 |
| Train rows whose lexical token sequence differs with combining marks preserved | 0 | 4277 |
| Old candidate edges failing the alternative mark-aware length/substitution rule | 0 | 84 |
| Alternative mark-aware candidate edges (diagnostic only) | 165 | 2679 |

`read_pairs` picks the smallest row-index peer for each anchor, not every
candidate edge and not a rotating schedule. Thus the existing candidate count
is not the number of distinct pairs this code would train on. Counts above
are conditional simulations, NOT a semantic admission or observed training.
Choosing a minimum index is deterministic but not evidence of an informative
contrast. Generic arms match eligible anchors, not pair difficulty or encoder FLOPs.

**Concrete mining defect:** Python's current `[^\\W_]` token pattern excludes
remaining Unicode combining marks after NFKC. For example Ede `Pô krơ̆ng`
becomes three lexical tokens (`pô`, `krơ`, `ng`) although preserving the mark
keeps two. An independent diagnostic using Unicode letter/number/mark groups
found2679 candidate edges:303 added and84 removed relative to2460. This is a
token-boundary/eligibility defect, not evidence of a defective underlying dataset.
The alternative is not silently promoted to the production rule. It has not
undergone linguistic word-segmentation validation and does not verify semantics.
Existing historical translation scores and tokenizer IDs are unaffected: this
new mining code has not been used for long training. Old receipts remain intact.

With uniform shuffled B4 batches and IF every original candidate-covered row
were eligible, probability of at least one eligible anchor is15.76% Amis and
33.46% Ede (`1 - product((N-K-i)/(N-i), i=0..3)`). This is a combinatorial
expectation, not a measured activity rate; real verified coverage is unknown.
Loss code averages eligible pairs within each microbatch, not all anchor rows.
One active anchor therefore has different per-anchor weighting than four active
anchors. Accumulation does not automatically change this into a global valid-pair
mean. Do not call0.1 an effective dataset-wide weight or change it from this audit.

### 14.4 Decision and next actions

| Gate | Decision | Reason |
|---|---|---|
| Numerical/software feasibility | PASS (existing8 technical cases) | Full-size forward/backward/reload/resume checked; not real-pair quality |
| Pair mining fidelity | NO-GO current Ede miner | Combining-mark token boundaries change eligibility |
| Semantic/localized correspondence | HOLD | No verified bilingual pairs; concrete reference-side concerns |
| Main-method novelty | NO-GO for current contribution claim | Distinct implementation, but borrowed relational/difference mechanism and no established task-specific gain/explanation |
| Current4-arm long-training queue | DO NOT LAUNCH | Not a scientifically ready main-method candidate; no newly trained model was judged to fail |

Recommended order, with no jobs launched by this review:

1. Correct the isolated miner's Unicode handling with regression cases; rerun
   train-only counts into a new receipt. Keep all original CSVs, CE sampling,
   and historical results unchanged. Make peer-selection coverage explicit;
   do not change min-index selection, loss reduction and sampling together.
2. Establish whether clean, useful bilingual contrasts actually exist. Have a
   competent bilingual reviewer record valid contrast, paraphrase/no meaningful
   contrast, uncertain, or reference-quality concern. Use the sample to judge
   feasibility, then authorize only actually reviewed pairs for this contract.
   If no reviewer is available, a separate explicitly heuristic lexical-pair
   exploratory contract is possible; it cannot be presented as verified semantic
   supervision or silently pass the current gates.
3. Before designing another upgrade, use frozen-baseline predictions on TRAIN
   contrasts only to ask whether the model misses the differing information.
   Report teacher-forced own-reference versus paired-reference margins in both
   directions, with length/tokenization caveats and generic-pair controls. Such
   margins are diagnostic, not semantic labels or a new training objective.
   Inventory available train predictions/checkpoints first; GPU cost would be
   forward-only if missing. No new test inspection is needed.
4. If the mechanism is supported and pair quality is sufficient, reconsider a
   clearly labeled seed42 development pilot with matched original LSR and
   relational controls plus optional CLRR. The prepared4arms alone cannot
   isolate benefit over original LSR, and are not RKD/CCPL reproductions.
   Reuse only receipt-verified matching baselines. Lock a fresh design before
   execution; preserve FP32/SDPA/B4/GA32/ES10/validationBLEU.
5. If these conditions fail, retire this prototype as a research instrument;
   design a task-conditioned alternative after auditing relevant prior methods.
   Do not merely add a gate/projector, increase lambda, or switch languages to
   obtain favorable scores. Replication/A100/test follow only a justified pilot.

This decision does not prove that CLRR/LSR are universally ineffective, does not
discard their earlier mechanistic insights, and does not promise ACL acceptance.
It rejects spending the current long-training budget on an insufficiently
justified main-method claim. No production code, frozen recipe or scientific
admission flag was modified to manufacture a PASS.

## 15. Code corrections after scientific review — 2026-11-09

Author requested fixing the findings. Fixed the isolated lexical tokenizer to
retain Unicode combining marks with stdlib Unicode categories; no new package
dependency, translation tokenizer change or dataset normalization on disk.
Added positive integer word-bound validation. New regression cases cover Ede
combining marks, NFC/NFD equivalence, apostrophes, punctuation and the false
three-word eligibility of a two-word source.

Fixed generic control sampling to exclude ALL reviewed neighbors of an anchor,
not only its selected minimum-index neighbor. Otherwise another accepted contrast
could re-enter the generic arm. Regression test checks a multi-neighbor anchor
over20 seeds and order-independent candidate selection using synthetic reviews.
Kept min-index candidate policy explicit; rotating peers or enumerating edges
would change the experiment rather than fix Unicode. No loss reduction, lambda,
CE sampler, optimizer or recipe change was made.

Reran both audits into new folders, preserving previous receipts:
`outputs/experiments/difference_preserving_feasibility/unicode_fix_v2/<dataset>/`.
Each includes new `PAIR_AUDIT.json`, `lexical_candidates.jsonl`, and blank
`PAIR_REVIEW.csv`. Audit now reports a named tokenizer policy and conditional
edge selection to avoid presenting every discovered edge as a trained pair.

| Dataset | Corrected candidates | Covered rows | Conditional selected undirected edges |
|---|---:|---:|---:|
| Amis | 165 | 193/4600 | 109 |
| Ede | 2679 | 1461/15092 | 1044 |

Counts are discovery/conditional behavior, not semantic verification. Coverage
did not increase despite more Ede edges. Both original train SHA256 hashes match
earlier audits. Source/target strings were not edited; validation/test not read.

Verification in conda clrr on CPU: **19 tests PASS**, including tiny-model
accumulation/checkpoint/resume, target stop-gradient, CE RNG, Unicode and control
exclusion regressions. These CPU tests do not certify a newly frozen GPU bundle.
The earlier8/8 technical results remain valid only for their recorded old source
snapshot; source changes require a fresh bundle and verification before launch.
No RTX job, HF upload, historical score edit or training launch was performed.

Remaining scientific dependencies cannot be fixed by code: actual bilingual
correspondence review and a supported novel contribution. Those statuses stay
unresolved. New candidate counts do not reverse section14's long-training decision.

## 16. Frozen-baseline train discrimination — completed 2026-11-09

Author authorized inventory and forward-only diagnostics. No new training,
generation or validation/test data access. Reusable args-based script:
`scripts/analysis/measure_train_pair_discrimination.py` (`--stage prepare/run`,
`--output`, `--minimum-free-gpu-mib`). Three scoring tests PASS on CPU: direct
log-probability/mask agreement, additive target-bias cancellation, and empty
content rejection. Native RTX conda clrr, FP32/SDPA/TF32off, diagnostic batch4,
original max lengths256. Full parameter gradients disabled, eval/inference mode.

Inventory: only the original pretrained mBART weights were present remotely;
no reusable train-pair probability output found. Restored each public baseline
sequentially to a scoped temporary directory; checked archive SHA256 against
its existing validation receipt. Both are historical chrF++-selected seed42
snapshots, **not substitutes for BLEU-selected pilot baselines**. Amis metadata
does not itself explicitly pin training attention backend; prior audits discuss
that history. This measurement explicitly uses SDPA and makes no new complete
training-protocol equivalence certification.

Repo `FiveC/amis-rewire-checkpoints`, pinned revision
`40fac65856b883e999bee74a45075d9c9d5dc7a5`:
- Amis: `checkpoints/amis_mandarin/mbart-large-50/baseline/seed-42/rtx6000-fp32-b4-ga32-es10/best.zip`, SHA256 `7fbb9e66fc10ac9b1d75a0caaec9dcc314db0d1d5f450329bce5f810b4eec00c`.
- Ede: `checkpoints/rhade_vietnamese/mbart-large-50/baseline/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-src-id/best.zip`, SHA256 `3443038509628c1773be7ed5f579570472c338a2eda0b099b4545cdcc67860c8`.

For original train pairs `(xi,yi),(xj,yj)`, let `sxy` be mean log probability
over target-content tokens (all special token IDs excluded). Score all four
source/reference combinations with ordinary teacher forcing:

```
m_i = s(xi,yi) - s(xj,yi)
m_j = s(xj,yj) - s(xi,yj)
m_sym = (m_i + m_j)/2
```

Each source margin holds the target fixed. The symmetric margin cancels
additive source-only and target-only score offsets, not every length/semantic
confound. Also retain reference-choice margins with the source fixed and a
secondary score including all supervised target labels. No BLEU is computed.
Generic pairs use the same candidate-covered anchors, exclude every candidate
neighbor, self and identical strings; they are not matched for topic/difficulty
or token length. No semantic labels inferred from scores. No pair had to be
excluded for token-identical/truncated/empty-content sequences in this run.

| Dataset / group | Pairs | Both source margins >0 | Mean symmetric margin (nats/content token) | Pairs with at least one nonpositive reference-choice margin |
|---|---:|---:|---:|---:|
| Amis all lexical near | 165 | 165/165 | 1.9214 | 1 |
| Amis min-index-selected near | 109 | 109/109 | 1.8481 | 1 |
| Amis generic | 193 | 193/193 | 6.4709 | 0 |
| Ede all lexical near | 2679 | 2679/2679 | 2.8923 | 3 |
| Ede min-index-selected near | 1044 | 1044/1044 | 2.7147 | 3 |
| Ede generic | 1461 | 1461/1461 | 8.2560 | 0 |

Do not interpret source discrimination as100% translation accuracy or perfect
reference choice. Amis reference-choice case3371/3435 includes the previously
flagged truncated-looking target. Ede cases270/13445,867/3900,1756/12093 include
different wording/pronouns; bilingual correctness is still unverified. Minimum
individual source margin:0.002693 Amis,0.013745 Ede. Small positive margins
are not proof of robustness. These dependent graph pairs and one seed do not
justify iid significance claims; no p-value is reported.

**Interpretation:** this probe supplies no source-ranking failures in the train
near-pair pool. Smaller near margins versus generic indicate harder contrasts,
not a demonstrated failure or a demonstrated benefit of the proposed objective.
Memorization is a plausible explanation of train success; this does not establish
generalization or prove the new loss cannot help. It weakens the current rationale
that these train contrasts expose a baseline inability needing this repair.
Do not open the difference-preserving pilot on that unsupported rationale.

Next justified work: inspect already available baseline VALIDATION generation
errors, identify a repeatable task failure, then compare the proposed repair
against close prior objectives. Do not treat all near pairs as semantic gold,
switch datasets to seek a positive result, or tune lambda from train margins.
The present review/diagnostic authorization did not launch that next analysis.

Outputs:
`outputs/experiments/difference_preserving_feasibility/baseline_train_contrasts/`:
`ANALYSIS.md`, `LOCAL_VERIFICATION.json`, `DOWNLOAD_VERIFICATION.json`, per-dataset
frozen `DESIGN.json`, `PAIRS.json`, `PAIR_SCORES.csv`, `RESULT.json`, review examples.
Local verification independently recomputed CSV margins/means, checked design,
script, pair-source and train hashes, and matched all downloaded artifact SHA256s.
Only small JSON/CSV downloaded to PC; no HF publication performed in this task.
Remote queue PID44880 completed with `TRAIN_PAIR_DIAGNOSTICS_COMPLETE_NO_TRAINING`.
Amis134.5s/2.62GiB peak allocated; Ede208.5s/4.65GiB. Restored archive/model files
removed after successful measurement; GPU returned to23952MiB free,0% utilization.
No other user's process was stopped and historical scores were not edited.

## 17. All-training relational screen: revised formulation, CPU only

Author requested continued method development with ongoing related-work review.
This section does not report new translation quality, change historical scores,
or clear the old v1 semantic-pair gate. No RTX training was launched here.

### 17.1 Corrected reasoning

Positive train source-ranking margins do not establish generalization and do
not falsify a regularizer. Conversely, a low representation loss is not proof
of better translation. Earlier HOLD decisions concern specific rationale and
readiness, not measured failure of every relational objective.

Do not call generic parallel-example pairs semantic minimal pairs. For the new
screen use all original training anchors, within their original CE microbatch;
build all unordered pairs after tokenization/truncation, excluding identical
source OR identical target content-ID sequences and empty-content rows.
No CSV edits, peer oversampling, semantic labels, validation mining or test use.
All anchors retain their ordinary CE. With B4 there are at most6 unordered pairs;
gradient accumulation32 does NOT turn that into a128-example relation pool.
Eligibility coverage and batches with fewer than2 valid pairs must be logged.
This design seeks full training eligibility, not a claim of100% valid pairs.

### 17.2 Explicit objectives and close prior

Let s_i and t_i be unnormalized content-masked mean encoder vectors of source
and reference. All target representations/normalizers are stop-gradient. Use
the same valid pair set P in each objective, and float32 reductions:

```
dx_ij = s_i - s_j
dy_ij = stop_gradient(t_i - t_j)
L_direction = mean_P [1 - cosine(dx_ij, dy_ij)]
mu_x = mean_P ||dx_ij||_2
mu_y = mean_P ||dy_ij||_2
L_distance = mean_P Huber(||dx_ij||_2 / max(mu_x,eps),
                          ||dy_ij||_2 / mu_y)
L_total = L_CE + lambda L_aux
```

Pairs with target norm<=1e-6 are excluded; small source norms are retained and
reported. No valid pairs gives differentiable zero. Source normalizer remains
differentiable. A denominator below epsilon is flagged; exactly collapsed
source vectors can be stationary for the norm-based control. There is no
anti-collapse or bounded-gradient theorem. Shared target encoder still evolves
through CE/source gradients, despite stop-gradient on the target view.

| Primary source read | Actual overlap | What this screen can establish |
|---|---|---|
| [RKD, CVPR2019, section3.2, Eq5–11](https://arxiv.org/html/1904.05068) | Mini-batch-normalized distances with Huber; angular relations; task loss plus relational loss | L_distance is an adapted RKD-D control, not a novel objective or reproduction of their fixed teacher/student experiments |
| [CCPL, ECCV2022, section3.1, Eq2–3](https://arxiv.org/html/2207.04808) | Corresponding feature-difference vectors, MLP and InfoNCE in image style transfer | Direct bilingual direction cosine differs operationally, but difference preservation itself is prior art |
| [HRKD, EMNLP2021, section2](https://aclanthology.org/2021.emnlp-main.250.pdf) | Hierarchical and domain relational distillation, domain graphs and compare-aggregate | Generic relational graphs or multiple layers cannot themselves be claimed new |

Method sections above were inspected; this is a focused audit, not exhaustive
novelty certification. Generic direction alignment remains a mechanism candidate,
not an established ACL contribution. Good scores alone will not establish novelty.

### 17.3 Two analytical findings verified on CPU

1. RKD-D with exactly ONE valid pair gives normalized distances1 and1: loss0
   regardless of the two raw distances. A one-peer, one-pair comparison would
   be uninformative. All-pairs B4 prevents this by construction only when
   multiple geometrically distinct valid pairs remain; log degeneracies.
2. Both objectives can be0 for the same rank-one geometry in source/target,
   including a very small positive common scale above epsilon. Therefore neither
   objective alone establishes non-collapse. This is a mathematical property,
   NOT evidence that any real trained checkpoint has collapsed.

A fixed90-degree rotation of only one view preserves RKD distances while
changing direction loss. This verifies different constraints, not superior
translation: whether enforcing the shared coordinate orientation helps is
precisely what remains to be tested.

Code: `src/methods/difference_preserving/v2/geometry.py`. Eight CPU tests in
`tests/test_relational_geometry_v2.py` cover translation/scale invariance,
rotation distinction, nonuniform distortion signal, the single-pair degeneracy,
stop-gradient and finite/scale-invariant source gradients, empty/collapsed
inputs, jointly impoverished geometry, and invalid input rejection.
No production wrapper, collator, registry or recipe was changed. CPU loss tests
do not certify accumulation, RNG, save/reload or generation of a future wrapper.

### 17.4 Bounded development screen, not yet admitted for training

| Arm | CLRR | Auxiliary | Purpose |
|---|---|---|---|
| A | off | adapted RKD-D | ordinary relational control |
| B | off | direct difference-direction cosine | test proposed constraint |
| C | original | adapted RKD-D | matched route/control interaction |
| D | original | direct difference-direction cosine | full candidate |

Amis seed42 first; lock existing RTX protocol, SDPA/FP32/TF32off, B4/GA32,
ES10/max20, validation BLEU beam1. lambda.1 is one exploratory setting, not
equal auxiliary strength across differently scaled losses or tuned comparison.
Pretrained initialization, CE order/dropout/reduction and optimizer stay locked.
Reuse only BLEU-selected baseline/LSR receipts with audited protocol equivalence;
the historical chrF++ diagnostic checkpoint is not automatically reusable.

Before launch: implement a separate v2 wrapper and content-ID eligibility,
freeze bundle/metadata and pass CE RNG, gradient accumulation/checkpointing,
save/reload and tiny real-model GPU gates. Auxiliary source/ref encoding uses
deterministic encoder views; two extra encoder passes can reuse within-batch
vectors instead of four source/peer passes. Runtime savings are unmeasured.

Exploratory advancement targets proposed BEFORE runs: candidate B or D at least
baseline+0.30 validation BLEU AND its route-matched distance control+0.20 BLEU,
without lower chrF++. D also needs B+0.20 BLEU to prioritize CLRR interaction.
These are investment thresholds, not significance tests or ACL requirements.
If B succeeds but D adds nothing, pursue the auxiliary objective without forcing
CLRR. If neither succeeds, stop this screen; no open-ended layer/lambda sweep.
If an arm advances, allocate equal validation tuning opportunity to close
controls before multiseed confirmation; do not claim superiority over tuned RKD
from this single-lambda screen. No test access in selection and no automatic
move to A100 or a second language solely to seek a favorable sign.


## 18. Authorized implementation and execution — 2026-11-09

The author approved all four section17 arms. Separate implementation:
`src/methods/difference_preserving/v2/{geometry,model}.py`, registry IDs
`relational-distance-v2`, `relational-direction-v2`,
`relational-clrr-distance-v2`, `relational-clrr-direction-v2`.
Recipe: `configs/experiments/amis_relational_screen_rtx.json`.
No old v1 semantic gate is bypassed; this new design makes no semantic-minimal-pair claim.
Admission is explicitly exploratory, with novelty UNESTABLISHED.

| ID | Method | CLRR | Auxiliary weight | Data/selection |
|---|---|---|---|---|
| A_S42 | adapted RKD-D control | off | .1 | Original Amis train; validation BLEU beam1 |
| B_S42 | difference-direction cosine | off | .1 | Same |
| C_S42 | adapted RKD-D control | original d2/strength.1 | .1 | Same |
| D_S42 | difference-direction cosine | original d2/strength.1 | .1 | Same |

All use seed42, FP32/SDPA, LR5e-5, B4/GA32, ES10/max20, original data hashes
and pinned mBART revision in `configs/protocols/amis_drb_bleu.json`.
No new train sampling, extra parameters, original LSR term, test, Ede or A100.
Four GPU gates must finish before the first long run. Gates exercise real
optimizer2steps, B4/GA32, reentrant checkpointing, active auxiliary loss,
strict selected-model reload, beam4 cached/uncached generation and step1 resume.
All training is sequential under the project training.lock. Resource admission
requires at least20000MiB free; other users' processes are untouched.

CPU tests cover all four wrappers: CE logits/RNG equality, nonzero finite
auxiliary gradients, no extra parameters, checkpoint replay gradients,
save/reload metadata, generation and exact Trainer accumulation/resume.
Eight geometry tests remain separate. GPU PASS does not establish novelty or
translation quality. Coverage counters are scoped since model load and logged
with last-microbatch diagnostics; they are not epoch-mean auxiliary losses.

### 18.1 Baseline reuse evidence (no retraining)

Audit: `outputs/experiments/relational_v2_baseline_audit/baseline_reuse.json`.
Pinned HF revision: `e04195904361dcbdf36d4ae6c7f577ca37eb3256`.
Archive: baseline/seed-42/rtx6000-fp32-b4-ga32-es10/best.zip under canonical
Amis mBART checkpoint hierarchy; SHA256
`7fbb9e66fc10ac9b1d75a0caaec9dcc314db0d1d5f450329bce5f810b4eec00c`.
The historical selector was chrF++, and its labels/receipts remain unchanged.
Its completed20-epoch validation trajectory shows BOTH BLEU and chrF++ select
epoch19/step684: BLEU17.6823291745, chrF++14.7030053622. Applying BLEU patience10
to that trajectory does not stop earlier. Same original CSV hashes, seed,
model revision, optimizer and batch/precision recipe. SDPA evidence is historical
stack/default reconstruction (saved backend_probe.json), not a contemporaneous
backend receipt. This limitation is preserved explicitly. Reuse is permitted for
this exploratory screen; it does not create a new independently trained baseline.
No LSR comparison is admitted without its own equivalence audit.

### 18.2 Operation and storage

Build and CPU gate locally in conda clrr, then launch via
`experiment_runner.rtx.launch_suite(['configs/experiments/amis_relational_screen_rtx.json'])`.
Queue lives exclusively in `/workspace/ura/thanhdat/projects/CLRR/`.
Small receipts come back through the existing monitor command. Each selected
checkpoint/validation artifacts uploads to public `FiveC/amis-rewire-checkpoints`:
`checkpoints/amis_mandarin/mbart-large-50/<method-id>/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu/`.
Cleanup of per-run weights occurs only after remote SHA verification; small
scientific receipts remain. No historical checkpoint is overwritten.
Progress and actual gate results must be read from the queue, not inferred from
this authorized plan. Section17.4 advancement thresholds remain locked.

Launch receipt: suite PID170990, initial free VRAM23952MiB; local CPU13/13 PASS.
Actual GPU gate/training progress is recorded in the recipe output receipts.

Reproducibility backup verified: HF revision `10a6fe3d2a3c775bd3f3c256dec274b2d4778f5f`, prefix
`analysis/amis_mandarin/mbart-large-50/relational-v2-screen/seed-42/reproducibility`. Contains frozen source bundle, PLAN/config/protocol lock,
CPU gate and baseline reuse audit; all file SHA256 values verified after upload.
Receipt: `outputs/experiments/amis_relational_screen_rtx/reproducibility_upload.json`.


### Verified execution update — all gates PASS, long training active

CPU gate13/13 PASS; real GPU gates A/B/C/D all PASS on the same frozen bundle.
Gate durations176.65/176.63/180.78/183.26seconds; peak allocated memory about
14.357GiB each. All four verified active auxiliary loss, strict save/reload,
beam4 cache parity, and optimizer/scheduler/RNG resume. Each gate saw64
microbatches with384/384 valid pairs; this is gate coverage, not full-training
coverage or translation quality. Gate weights/checkpoints were cleaned.

Queue PID170990 has entered actual long training A_S42 (observed step8/720,
GPU96% utilization), then B_S42→C_S42→D_S42. No validation-quality result yet.
Current receipts: `outputs/experiments/amis_relational_screen_rtx/gates/*/PASS.json`,
`PREFLIGHT_SUMMARY.json`, `queue_validation_account1.json`.
An individual gate PASS does not imply the four long runs have completed.
