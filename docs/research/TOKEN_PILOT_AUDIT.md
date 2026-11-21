# Token-LSR pilot audit — 11/10/2026

No training or optimizer steps. Same frozen 128 validation rows; four microbatch4 gradient probes. FP32, dropout/TF32 off. Public checkpoint revision pinned and archives SHA256 verified. Temporary checkpoint copies removed from RTX.

## Metadata and validation

All four runs reached epoch20 with patience10. Source bundle/runtime and resolved training arguments match except output/log paths. Recomputed validation metrics match saved values.

| Model | Selected epoch | Validation BLEU | chrF++ | chrF0 | Source effective covariance rank | Mean weighted LSR/CE grad ratio |
|---|---:|---:|---:|---:|---:|---:|
| R1 baseline | 19 | 17.6823 | 14.7030 | 16.9547 | 74.380 | 0.1317% |
| R2 full | 14 | 17.1156 | 14.3450 | 16.4809 | 73.615 | 0.2147% |
| R3 token-lsr-v1 | 19 | 17.5811 | 14.5050 | 16.5823 | 69.174 | 0.2280% |
| R4 token-full-v1 | 18 | 16.7533 | 14.0672 | 16.1561 | 72.460 | 0.1473% |

Baseline gradient ratio is a counterfactual token-loss probe, not a loss used by baseline training. Rank is calculated from squared singular values of centered pooled vectors (128 rows, rank limit127), not token rank.

## Loss and alignment probes

| Model | CE | Sentence LSR | Token LSR | Centered matched cosine | Weighted aux/CE gradient cosine |
|---|---:|---:|---:|---:|---:|
| R1 | 3.3212 | 0.6614 | 0.8325 | 0.2821 | 0.0851 |
| R2 | 2.9655 | 0.2473 | 0.6582 | 0.2882 | 0.0380 |
| R3 | 3.2076 | 0.3318 | 0.6342 | 0.3173 | 0.0703 |
| R4 | 3.1006 | 0.3681 | 0.6855 | 0.2898 | 0.0435 |

CE/loss means above use16 validation examples in4 fixed microbatches, not the whole validation corpus. Baseline auxiliary objectives are counterfactual probes. Actual trained wrappers agree exactly with independently decomposed CE+weighted auxiliary losses in all16 model/batch cases. All gradients are finite, auxiliary decoder gradients are absent, and parameter version counters are unchanged.

## Inference interventions on the locked validation subset

| Model | Disabled layers (1-based) | Delta BLEU | 95% paired bootstrap delta CI | Raw p | Holm12 p |
|---|---|---:|---|---:|---:|
| R2 | 3–5 | +0.9557 | [-0.4599, +2.3794] | 0.0804 | 0.9647 |
| R2 | 6–9 | -0.3237 | [-2.7685, +2.1345] | 0.2897 | 1.0000 |
| R2 | 10–12 | +0.7626 | [-0.8881, +2.5356] | 0.1420 | 1.0000 |
| R4 | 3–5 | -0.8338 | [-2.2573, +1.0448] | 0.1066 | 1.0000 |
| R4 | 6–9 | +0.5519 | [-1.4178, +3.0354] | 0.2256 | 1.0000 |
| R4 | 10–12 | -0.3534 | [-1.9335, +1.5971] | 0.2370 | 1.0000 |

## Metric interpretation

Existing test chrF++ uses only3 reference word bigrams across575 rows. Baseline matches0; R2/R3/R4 each match1. This sparse order strongly affects the +4.6681 chrF++ gap; it is not broad evidence of improved character fidelity. Existing official scores are unchanged. Word-order0 chrF is included as a diagnostic.

| Model | Existing test chrF0 | Diagnostic chrF+ (word-order1) | Existing test chrF++ |
|---|---:|---:|---:|
| R1 | 18.0917 | 15.8731 | 13.8889 |
| R2 | 17.6237 | 15.5445 | 18.1317 |
| R3 | 17.9597 | 15.8327 | 18.3830 |
| R4 | 18.2176 | 16.0288 | 18.5570 |

## Length and output checks

Validation maximum untruncated token length: source33, target31; neither reaches256. Retokenized predictions reach at most47 tokens, far below the cap. Original generated token IDs were not retained, so this is a screening proxy for generation truncation.
No empty validation translations. The predefined repeated-character-trigram heuristic flags2 of576 Token-Full outputs and0 for the other models; this is not a manual error rate. Twelve randomly selected validation examples are saved in QUALITATIVE_SAMPLE.csv.

## Interpretation

Token-LSR reduces the measured token objective compared with baseline, yet does not improve full-validation BLEU. Token-Full has a higher token loss and lower centered pooled matched cosine than Token-LSR in these probes. This supports investigating the interaction between alignment and rewiring; it does not establish causation or gold token-alignment accuracy.
No catastrophic rank collapse is apparent in the pooled representation probe. Source residual RMS ratios are approximately6.7–8.4% at layers3–9 and0.8–3.2% at layers10–12. Rewiring is active; low efficacy is not explained by an inactive hook.
All intervention CIs include0; none is significant after Holm12 correction. Opposite early-layer intervention signs across original Full and Token-Full do not support a universal early-layer removal rule.

## Decision boundaries

Token-Full fails the locked validation screening gate: its BLEU is below baseline, original Full and Token-LSR. No automatic extra seeds or A100 training. Changing selection from chrF++ to BLEU would select epoch19 for R4 and improve logged validation BLEU by only0.1114; it remains below baseline. This alternative is descriptive and does not replace the frozen selection.

One seed, selected endpoints, validation already used for checkpoint selection. Gradients are instantaneous, dropout disabled; not accumulated Adam updates. Inference intervention is not evidence that retraining the modified topology will improve.

Artifacts: CPU_AUDIT.json, ANALYSIS.json, R1–R4/RESULT.json, PLAN.json, QUALITATIVE_SAMPLE.csv, TOKEN_LENGTHS.json, TEST_METRIC_WORD_COUNTS.json, TRANSFER_SHA256.json and executed_worker.py.
