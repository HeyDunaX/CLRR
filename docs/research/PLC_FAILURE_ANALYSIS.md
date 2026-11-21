# PLC failure analysis and next-method requirements

Updated: 2026-11-15. Development evidence only; no new training or test evaluation.

## 1. Scope and decision

PLC failed its predeclared validation promotion gate. It is not demonstrated to
be statistically worse than W across training seeds. Do not promote PLC, sweep
its weights, or discard the broader consistency direction on this evidence.
The useful outcome of this audit is a diagnosis of objective geometry and
limits of the earlier mechanism claim, not a newly validated method.

RTX was used only for frozen-checkpoint diagnostics. No optimizer was created,
no checkpoint was changed, and no training process was started. Temporary
restored weights for W/P/D were removed after measurements; public originals
remain at their pinned verified revisions. Final RTX free VRAM: 23952 MiB.

## 2. Data, reproducibility and measurement checks

- All 576 original validation source/reference rows match every prediction CSV.
- BLEU/chrF++ replay agrees with receipts within 1e-10, from pinned HF revisions.
- All five new models completed epoch20/step720; this was not an ES10 stop.
- Best BLEU selected at epoch19 for P/D/E/Middle, epoch18 for Wcal; reused W at20.
- Independent analytic output-gradient checks agree with autograd on a toy
  masked/noncontiguous sequence. Frozen-block gradient extraction agrees with
  all-trainable extraction under reentrant checkpointing on a tiny mBART.
- On RTX, strict state loading, frozen bundle/data/runtime checks and exact
  dropout RNG replay passed. Paired views differ. No optimizer steps.
- A Transformers4.57.6 Mistral regex warning occurred when loading local mBART
  exports. Inspection of its warning branch shows a config-version-based
  detection path that can flag this non-Mistral local model. No regex correction
  was applied. All observed target token IDs on the16 fixed probe rows agree
  exactly with the original pinned mBART tokenizer, for each checkpoint.
  This check does not independently compare all source token IDs.
- An initial diagnostic-only backward failed because a hook on shared
  embeddings did not reach the two embedding modules actually called by
  mBART. Explicit encoder/decoder embedding hooks fixed it; CPU gradient
  equivalence passed before the successful rerun. This is not evidence of a
  defect in the completed training runs, which trained the full backbone.

## 3. Performance and uncertainty

All scores below are Amis validation, seed42, FP32 SDPA, beam1.
Bootstrap: 10000 paired sentence resamples, fixed seed20261115, exact corpus
sufficient statistics. Checkpoints were already selected on this validation
set: intervals are post-selection descriptive intervals, not independent
confirmation or training-seed uncertainty. No significance claim or adjusted
p-value is inferred from these intervals.

| Model | BLEU | chrF++ | BLEU delta vs W | Descriptive 95% interval |
|---|---:|---:|---:|---|
| W |18.272410|15.172807|0|—|
| PLC/P |17.714812|14.845499|−0.557598|[−1.567756,+0.365787]|
| D |18.271456|15.179737|−0.000954|[−0.804338,+0.806788]|
| E |17.907733|14.970888|−0.364677|[−1.370241,+0.622326]|
| Wcal |17.241326|14.421296|−1.031084|[−2.078390,−0.023411]|
| Middle RTX |16.677522|13.937579|−1.594887|[−2.744269,−0.500392]|

PLC trails W in19/20 same-epoch BLEU snapshots; these are correlated snapshots,
not20 independent trials. Validation CE bottoms at epoch6 for both W and PLC,
while generation BLEU continues improving much later. At epoch20 Wcal CE is
2.670941 vs W2.763629, yet Wcal BLEU is lower: optimizing teacher-forced CE
alone is not a valid method-selection rule. Do not change the BLEU selector.

### Error strata

Reference character length excludes whitespace. Quartile boundaries are9,11,14
characters; ties produce unequal group sizes. Groups were chosen post hoc and
are descriptive, not separate significance tests.

| Reference length | Sentences | PLC−W BLEU | PLC−W chrF++ |
|---|---:|---:|---:|
| <=9 |182|−0.507214|−0.302360|
| 10–11 |131|−0.860705|−0.620678|
| 12–14 |161|−1.245812|−0.725645|
| >14 |102|+0.315119|+0.324709|

Thus the simple story 'prefix coupling especially damages long sentences' is
not supported by these predictions. Both W and PLC have zero empty predictions
and zero character-trigram repetitions occurring at least three times. Their
total output/reference character ratios are0.966329 and0.959625. These proxies
cannot establish semantic adequacy or exclude hallucination.

Sentence chrF++: PLC improves175 rows, loses172, ties229;180 predictions are
identical. Counts do not determine corpus BLEU: magnitude and n-gram aggregation
matter. Examples sorted by sentence chrF++ are intentionally selected examples,
not prevalence estimates. For instance row13's reference/W mention crocodiles
where PLC outputs kites, indicating a lexical mismatch against this reference;
this is not evidence that every disagreement is a source-grounding error.

## 4. What was measured on the frozen checkpoints

W/P/D:16 validation rows evenly spaced by index, two dropout pairs each.
Output-gradient diagnostics use microbatch1, not a replay of training b4/GA32.
Parameter diagnostics use the first four selected rows0,38,76,115, one paired
mask each, and three named weights:

- `model.encoder.layers.11.fc2.weight`
- `model.decoder.layers.11.fc2.weight`
- `model.decoder.layers.11.encoder_attn.q_proj.weight`

Their12 row/block measurements are correlated, not12 independent samples or
seeds. They do not measure the whole-backbone gradient or historical Adam
moments. No claims about the early/middle training trajectory follow.

### A. Cross-token compensation is real

For a_t=log p1(y_t)−log p2(y_t), PLC is a^T K a. Its derivative is2Ka;
unlike D's2diag(K)a, this can have a sign opposite to a_t. Descending that
component can increase an individual discrepancy while reducing the aggregate
prefix objective. This is a property of the loss, not an implementation bug.

| Frozen model | Tokens where (dP/da_t)*a_t<0 | Token observations |
|---|---:|---:|
| W |39.665%|358|
| PLC |36.592%|358|
| D |36.034%|358|

These percentages describe the hypothetical PLC gradient on each checkpoint;
D was not trained with that PLC gradient. Repeated mask observations are not
independent sentences. Compensation does not itself prove harmful translation
updates; it invalidates the stronger interpretation that every token's
disagreement must decrease under this auxiliary loss.

### B. Small scalar loss does not imply a negligible gradient

On W, tokens with mean gold probability<.01 are21.229% of token observations
but account for61.674% of PLC's squared output-gradient norm (D:54.339%).
On PLC itself, the respective figures are20.670% and56.595% (D:44.060%).
Special-token energy is below1e-8 of the total on these probes: a special-token
dominance explanation is not supported here.

For close Bernoulli gold masses p,q=p+delta with |delta| small relative to p and
1-p, squared log-probability difference is approximately delta^2/p^2, while our
half-symmetric KL convention is approximately delta^2/[2p(1-p)]. Their local
ratio is2(1-p)/p. This explains a possible low-probability emphasis, not an
unbounded softmax-logit derivative, and not proof of empirical damage.

At W, median weighted output-gradient norm/CE is0.428 for0.1P,0.435 for0.1D,
and0.784 for1.25KL. Max0.1P/CE is3.152. At PLC, median0.1P/CE is0.717,
max3.641. The auxiliary gradient is not uniformly tiny. These are unaccumulated
logit-space norms and cannot be identified with Adam update sizes.

### C. There is partial parameter conflict, but it is not unique to PLC

Negative cosine against CE across the12 correlated row/block measurements:

| Checkpoint | KL | PLC hypothetical/actual | D hypothetical/actual |
|---|---:|---:|---:|
| W |6/12|5/12|1/12|
| PLC |6/12|3/12|0/12|
| D |6/12|5/12|0/12|

PLC has more negative components than D on this small probe, supporting a
mechanism hypothesis. But KL also conflicts despite W's better translation
score. Therefore 'any negative cosine is harmful' or 'gradient surgery will
fix PLC' is not justified. At W, median weighted partial parameter norm/CE is
0.340 for PLC vs0.190 for D. This differs from output-space ratios.

### D. The originally advertised cross-drift mechanism is not confirmed

| Checkpoint | Mean P | Mean D | Mean P−D |
|---|---:|---:|---:|
| W |1.303319|0.847778|0.455540|
| PLC |1.213520|0.717383|0.496137|
| D |1.254619|0.736155|0.518464|

PLC has slightly lower total P on these16 rows but a higher cross component
than W. This does not demonstrate successful selective suppression of the
cross-position quantity that motivated the design. The sample is small and
post hoc; do not call the differences statistically established.
Shuffling a's positions100 times changes mean P to1.510732/1.396237/1.408073
for W/P/D. This is a diagnostic permutation only, not a trained shuffled model
or proof that ordering causes a BLEU difference.

## 5. Ranked explanations and rejected overclaims

| Explanation | Evidence level | Decision |
|---|---|---|
| Wrong metrics/reference/checkpoint or premature stopping | Checks do not support it | Do not rerun to repair a score that already replays |
| Prefix loss permits cross-token compensation | Algebra and measured signs | Real design risk; remove the per-token stability claim |
| Raw log-score geometry emphasizes uncertain gold tokens | Local analysis and measured concentration | Supported diagnostic property; causal contribution remains open |
| PLC has negligible effect because scalar loss is small | Gradient measurements contradict it | Reject |
| Off-diagonal coupling may interfere with CE | More conflict than D on selected blocks | Plausible, not a proven sole cause |
| All negative gradient cosines should be removed | W's KL also conflicts | Reject as an automatic fix |
| Long sentences or special tokens explain the failure | Measured strata/energy do not support it | Reject for this probe |
| PLC is globally worse than W or all consistency is useless | One seed, CI includes0; W works on existing development seeds | Reject |

## 6. Consequences for the next method

Keep W and the existing A100 comparators. Do not train a new method merely
because it minimizes the prefix diagnostic. Do not promote D as novel because
its BLEU is close to W. Preserve all old scores.

Design requirements before another pilot:

1. State what translation error is being targeted beyond dropout disagreement;
   it must not be inferred from a positive covariance alone.
2. If a consistency term remains, explicitly audit probability geometry,
   cross-token compensation and weighted gradient scale. A small lambda is
   insufficient assurance. Any proposed guarantee must say which space it uses.
3. No generic gradient surgery or confidence weighting presented as a new ACL
   contribution. Compare equations to PCGrad, MetaBalance, R-Drop, Fraternal
   Dropout, LR-Drop and source-based adaptive MT before admission.
4. Require a mechanism-specific matched control, while reusing verified W,
   baseline and Middle runs; no blanket retraining of comparators.
5. Keep ES10/max20/BLEU selection and the locked RTX recipe. Fixed-batch
   diagnostic logging must be part of the next implementation before training:
   initialization/25%/50%/100%, fixed mask seeds, per-token probability/length
   groups, CE/KL/new-loss gradients on named blocks, and population scope.
6. Only after a substantive design/prior distinction passes: CPU algebra and
   replay, native full-recipe GPU gate, then a bounded seed42 pilot. Replication
   and A100 are conditional on the predeclared validation improvement gate.

No new method is admitted by this report. In particular, normalizing the old
prefix loss, taking its absolute differences, or applying PCGrad is not yet a
new paper contribution. This audit narrows the search; it does not promise a
high-scoring replacement or establish a complete causal training explanation.

## 7. Literature audit and source boundaries

NotebookLM source-restricted review completed with actual cited source IDs;
receipt: `outputs/experiments/amis_prefix_consistency_failure_analysis/LITERATURE_REVIEW_RECEIPT.json`.
Its earlier unrestricted response had no returned source-use receipts and
contained causal/threshold overclaims; these were not accepted as evidence.
Independent primary-paper checks:

- [R-Drop](https://arxiv.org/abs/2106.14448): ordinary two-view symmetric KL is prior.
- [Fraternal Dropout](https://arxiv.org/abs/1711.00066): squared paired prediction
  discrepancy/variance is prior; it is not the exact gold-prefix kernel.
- [LR-Drop](https://aclanthology.org/2024.lrec-main.891/): hidden-state and attention
  map consistency already includes relations between positions.
- [PCGrad](https://arxiv.org/html/2001.06782v3), sections2.2–2.3: negative cosine
  alone is not its complete harmful-interference hypothesis; projection is prior.
- [CBMI](https://aclanthology.org/2022.acl-long.169/): source/target-context-based
  token and sentence reweighting already has direct MT precedent.

## 8. Reproduction

Run in conda clrr; no commands below train a model:

```powershell
conda run -n clrr python scripts/analysis/analyze_prefix_failure.py check
conda run -n clrr python scripts/analysis/analyze_prefix_failure.py cpu
conda run -n clrr python scripts/analysis/analyze_prefix_failure.py summarize
```

GPU probe is RTX-project-only, requires the saved pinned `probe_MANIFEST.json`
and an available GPU/project lock. It is not a new experiment recipe. All
downloaded probe artifacts are JSON/CSV,283344 bytes across12 files, SHA checked.
Main machine-readable reports: `CPU_ANALYSIS.json`, `MECHANISM_SUMMARY.json`,
`TOKENIZER_CHECK.json`; per-model raw records and token CSVs are under `probe/`.
