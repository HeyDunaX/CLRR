# Transition LSR follow-up: evidence and decision record

Date: 2026-11-10. This record concerns the authorized follow-up after the two
Transition LSR runs. Existing experiment scores, CSV data and method recipes
remain unchanged. New measurements use validation only and zero optimizer steps.

## 1. Completed CPU audit

Reproducible inventory command, in conda `clrr`:

```powershell
python scripts/analysis/audit_method_evidence.py --output outputs/experiments/transition_followup_diagnosis
```

Current local shared-runner inventory: 63 completed validation receipts and
3,456 training/evaluation history entries. These are receipt counts, not 63
independent models or an exhaustive count of historical paper experiments.
Copied receipts in test-evaluation folders are excluded. Missing local receipts
do not establish that a checkpoint is missing on HF.

Artifacts: `VALIDATION_INVENTORY.csv`, `EVIDENCE.json`, `INVENTORY.md`, and
`TRAJECTORY_ANALYSIS.json` under the output directory above. Recipe fingerprints
include dataset hashes, runtime, model revision, selector, numerical settings,
batching and language tokens. A different fingerprint requires inspection; it
does not automatically prove an unfair comparison.

## 2. Observed Amis seed42 trajectories

| Model | Selected validation BLEU | Selected chrF++ | Best BLEU epoch | Completed epochs |
|---|---:|---:|---:|---:|
| Baseline R1 | 17.682329 | 14.703005 | 19 | 20 |
| Original CLRR-only P1 | 16.621620 | 14.125185 | 18 | 20 |
| Original Full R2 | 17.115572 | 14.345025 | 14 | 20 |
| Transition LSR-only | 17.083317 | 14.282589 | 20 | 20 |
| CLRR + Transition LSR | 17.521970 | 14.638739 | 17 | 20 |

Original runs used chrF++ selection; the stored full trajectories show that the
selected checkpoints in this five-model table also maximize BLEU on their own
completed trajectories. This establishes selector equivalence for these runs,
not for every historical run. Historical attention backend is reconstructed,
not backed by a contemporaneous explicit runtime receipt; preserve that limit.

- Transition Full minus original Full: **+0.406398 BLEU**.
- Transition Full minus baseline: **-0.160359 BLEU**, **-0.064267 chrF++**.
- Transition Full minus Transition LSR: **+0.438654 BLEU**.
- All five have complete 20-epoch trajectories. Early stopping cannot explain
  their stopping time. This does not prove that longer training would help.
- Transition LSR last-microbatch auxiliary samples: 0.813799 at step10 and
  0.014217 at step720. Transition Full: 0.817414 and 0.010469. They concern
  different examples; they are not epoch means or a controlled fixed-batch test.
- `eval_loss` is not a common CE measure across wrappers: original Full includes
  endpoint LSR during evaluation, whereas Transition adds its auxiliary only in
  training. Do not compare these logged values as identical CE.
- There is no original LSR-only RTX receipt in this five-model matched subset.
  Do not substitute an A100 result to fabricate a complete factorial comparison.

## 3. Frozen-checkpoint GPU measurements

Worker: `scripts/analysis/diagnose_transition_followup.py`.
Remote output: `/workspace/ura/thanhdat/projects/CLRR/outputs/experiments/transition_followup_gpu`.
The five checkpoints in section2 are restored one at a time from their pinned
HF revisions, with archive SHA verification and strict shared-engine loading.
Numerical method dependencies are verified against each original manifest.
The registry has acquired additional candidates; that alone does not change the
original method formula. Diagnostic glue has its own worker hash.

Fixed validation sample: 64 examples from NumPy seed42; four fixed microbatches
of four examples for gradients. The design and exact indices are saved before
measurement. FP32, SDPA where specified by the original recipe, TF32 off,
dropout off, no optimizer or parameter update. Project lock and a minimum of
20,000 MiB free VRAM protect other users' work.

Measurements:

1. Matched versus 100 shuffled source/target cosines, centered cosine, centered
   source effective rank and variance, both at the endpoint and on ten depth
   transitions. Rank is sample-limited by 64 examples.
2. Transition norms relative to recipient norms and the actual injected rewire
   contribution relative to the pre-rewire recipient, per edge.
3. CE and weighted-auxiliary gradient norms/cosines globally and by encoder
   layer, on the same four batches. These are eval-mode instantaneous gradients,
   not training-time Adam updates or causal proof of optimization failure.
4. CE with routing ON versus OFF on those batches. This tests dependence of an
   already co-adapted checkpoint; it does not predict a retrained ablation.

Each result asserts unchanged parameter versions. Temporary downloaded/extracted
weights are removed only after the pinned archive is verified and the probe
succeeds. Only small JSON/CSV/Markdown results are collected locally.

## 4. Primary related work checked before another candidate

- [FDD, ACL2025](https://aclanthology.org/2025.acl-long.1125.pdf), sections3.1–3.3,
  Eq7: cosine matching of differences across selected layer depths already
  exists. Its teacher/student log-prediction space differs from our bilingual
  hidden-state pooling; that distinction alone is not an established new
  contribution. Keep the present variant labelled engineering exploration.
- [Du et al., Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/html/1812.02224v2),
  sections3 and6: gradient-cosine-based weighting has prior art and its estimates
  have limitations. A single end-checkpoint gradient sign cannot establish
  that auxiliary learning is harmful throughout training.
- [PCGrad, NeurIPS2020](https://arxiv.org/html/2001.06782v4), section2.3:
  projecting conflicting gradients is existing methodology. Adding this to LSR
  would need attribution and evidence; it is not a new CLRR principle by itself.

This is a focused review of relevant alternatives, not a guarantee that all
related work has been exhausted or that a candidate is novel.

## 5. Conditional execution gates

The Transition pilot failed its previously locked advancement gate. Therefore
do not replicate it, run its test, transfer it to Ede/A100, or sweep layer/lambda.

The user's authorization to execute the full workflow applies to later steps
when their prerequisites are met; it does not turn a failed gate into a PASS.

| Next stage | Required evidence | Action if absent |
|---|---|---|
| One candidate formula | Specific mechanism supported by this audit and older ablations; one core change; matched control | Do not invent a module to fill the stage |
| Related-work admission | Formula-level comparison and modest defensible claim | Reject or redesign before smoke/training |
| Technical admission | CPU correctness, actual GPU forward/backward, checkpoint replay, unchanged baseline recipe | Fix implementation and rerun gates |
| Pilot admission | Frozen Amis seed42 recipe and validation-only endpoint, threshold and run budget set before training | Do not launch a sweep |
| Replication | Pilot quality gate PASS; no selection on test | Stop that candidate |
| Ede / final evaluation | Method locked, matched baseline provenance, explicit dataset recipe | No automatic expansion from a failed pilot |

GPU results and the resulting candidate decision are recorded below.

## 6. GPU measurements complete: five of five PASS

All five pinned checkpoint archives were verified. All measurements used the
same 64 validation examples and four batches, zero optimizer steps; unchanged
parameter versions and temporary-weight cleanup passed. Restored models report
SDPA. This does not replace missing historical contemporaneous backend receipts.

Layer-transition cosines below are means over ten edges, not ten independent
experiments. Centering retains a paired signal; do not call all alignment fake.

| Model | Transition matched cosine | Shuffled cosine | Centered matched cosine | Endpoint source rank (64 examples) |
|---|---:|---:|---:|---:|
| CLRR-only | 0.279251 | 0.270606 | 0.181767 | 58.2305 |
| Baseline | 0.968193 | 0.964215 | 0.375880 | 58.2747 |
| Original Full | 0.291795 | 0.283201 | 0.193411 | 58.1169 |
| Transition Full | 0.983797 | 0.981994 | 0.400323 | 58.3733 |
| Transition LSR | 0.976610 | 0.974031 | 0.387802 | 58.2106 |

### Supported conclusions

- Raw transition cosine is largely non-specific: baseline is 0.968193 matched
  versus 0.964215 shuffled; Transition Full reaches 0.983797 versus 0.981994.
  A large raw cosine alone is weak evidence of translation-pair preservation.
- Centered edge alignment remains nonzero (Transition Full 0.400323 versus
  baseline 0.375880). This does not support calling the entire objective fake.
- Endpoint centered alignment does not improve: Transition Full 0.245356 versus
  baseline 0.275199, even though edge raw cosine increases. Thus optimizing the
  edge proxy and preserving final translation representations can diverge.
- Endpoint source effective ranks are 58.12–58.37 on 64 examples. This sample
  does not show severe endpoint rank collapse, or prove absence of other forms
  of compression. Rank cannot be compared numerically to earlier 576-row ranks.
- Mean weighted-aux/CE gradient norm ratios: original Full 0.183971%,
  Transition LSR 0.032235%, Transition Full 0.021054%. All four global gradient
  cosines of Transition Full are positive (0.012659–0.034422). These end-checkpoint
  probes do not justify either gradient-dominance claims or blindly increasing
  lambda. Layer-local conflict and earlier training behavior remain distinct.
- Routing OFF raises mean microbatch CE by 0.046105 for CLRR-only, 0.242762
  for original Full, and 0.382429 for Transition Full. The trained models use
  the route; dependence does not imply beating a separately trained baseline.

## 7. Validation output analysis complete

Existing `analyze_prediction_strata.py` was reused with raw SHA-verified saved
predictions, original 576 validation rows and four predeclared-for-this-analysis
comparisons. This remains post-hoc development analysis. All corpus metrics
replay the stored receipts; no new generation, test or dataset edits.

- Transition Full versus baseline: highest rare-source-fraction quartile
  **-3.121507 BLEU (113 rows)**, longest-target quartile **-1.239172 (102 rows)**.
  Groups use whitespace frequency and character length, not verified morphology.
  Subset BLEU deltas are not additive and do not establish causal loss of detail.
- Leave-one-row-out BLEU difference stays negative: **[-0.306157, -0.031254]**.
  The negative corpus sign does not depend on a single validation row.
- Transition Full has no empty output or repeated-character-trigram flag.
  Mean nonspace length ratio is 0.997618. Character-unigram precision is
  0.510922 versus baseline 0.510545; recall 0.487329 versus 0.495103.
  This is a reference-relative coverage signal, not verified semantic omission.

## 8. Execution decision after the audit

**NO-GO expanding Transition v1. No new long training was opened.** The new
findings refine the diagnosis toward pair-specific, task-relevant preservation
of details; they do not identify a validated architectural fix. Existing endpoint
centering, token alignment, contrastive alignment, confined gradients, learned
donor modules and MRT controls already have historical experiments or prior
reviews. They must not be reintroduced under new names as untested novelty.

| Workflow stage | Outcome in this follow-up |
|---|---|
| Receipt/protocol inventory | DONE; 63 local validation receipts, historical coverage explicitly limited |
| Learning trajectories | DONE; five models, 20 epochs each; selector equivalence checked individually |
| Transition/endpoint norms, geometry and gradients | DONE; five checkpoints, identical batches, zero updates |
| CLRR dependence / output strata | DONE; descriptive intervention and validation analyses |
| Focused primary-prior review | DONE; FDD, auxiliary-gradient weighting, PCGrad; not exhaustive novelty certification |
| New candidate admission | NOT MET: no specific new intervention yet supported strongly enough to lock a new ACL method claim |
| New candidate smoke / pilot / replication / Ede / final test | NOT OPENED: dependent on candidate and quality gates; not falsely marked complete |

The next design should make a falsifiable prediction about preserving rare-source
or longer-target content beyond mere cosine improvement, retain CLRR/LSR as
controls, and be checked against lexical-shortcut, token-alignment and distillation
priors before code. No guarantee of a better method or ACL acceptance is inferred.
No further GPU measurement or full train is queued by this report.

## 9. Storage and completion

17 small analysis/source artifacts were uploaded and hash/size verified in public
`FiveC/amis-rewire-checkpoints`, revision `4f146a543b93a4359499bef77e51f19703c9aa4e`, prefix
`analysis/amis_mandarin/mbart-large-50/transition-lsr-followup/seed-42`. Local receipt: `outputs/experiments/transition_followup_diagnosis/HF_BACKUP.json`.
The public REPORT.md is the report snapshot before this upload receipt was appended.
All five temporary checkpoint restores were removed; project disk free70.9GiB.
Queue exited, GPU23952MiB free/0% at completion. No training queue remains from this follow-up.
