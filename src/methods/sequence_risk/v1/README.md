# Candidate-set risk: known control, numerical prototype

Status 18/10/2026: **8 numerical + 10 pipeline CPU gates PASS**. Integrated
with the shared runner, frozen initialization, train-only pool, GPU preflight,
checkpoint reload and public backup. GPU/run status is recorded in the recipe
workspace; CPU success alone does not authorize bypassing GPU gates.
This is a known objective control, not an ACL novelty claim.

Reference: [Shen et al., ACL 2016, §3.2, Eqs. 12–14](https://aclanthology.org/P16-1159.pdf).
For differentiable sequence **sum** log probabilities `s[b,k]` and fixed
reference-based costs `c[b,k]`:

```
q[b,k] = masked_softmax(alpha * s[b,k], over candidates)
risk = mean_b(sum_k(q[b,k] * stop_gradient(c[b,k])))
d risk / d s[b,k] = alpha/B * q[b,k] * (c[b,k] - E_q[c[b,:]])
```

There is no length normalization inside this function. Scoring must include
the candidate's EOS, exclude padding and use the established language-prefix
convention. Do not supply generation beam scores: they may include length
penalties/processors. Re-score candidate tokens under the current model with
gradients. Generation and metric computation remain outside autograd.

`valid` pads variable candidate counts. Deduplicate complete token sequences
before calling; singleton/equal-cost sets have no useful risk gradient. The
function detaches costs and averages sources equally. All four wrapper arms
use native CE and whole-view non-reentrant checkpointing; risk candidate
scoring uses dropout OFF and the same route used at inference. See `model.py`
and `data.py`. The loss function itself does not launch anything.

The finite candidate-set objective is standard. Offline cached pools, changing
the risk metric, or combining it with CLRR do not establish a novel method.
See [the design and conditional experiment plan](../../../../docs/research/METHOD_UPGRADE_DESIGN.md#7-quyết-định-tiếp-theo--18102026).

CPU reproduction, from repository root, in the existing `clrr` environment:

```powershell
conda run -n clrr python -X utf8 -m unittest discover -s tests -p test_sequence_risk.py -v
conda run -n clrr python -X utf8 -m unittest discover -s tests -p test_sequence_risk_pipeline.py -v
```

Approved recipe: `configs/experiments/amis_sequence_risk_continuation_rtx.json`.
Three continuation epochs (108 nominal optimizer steps), LR1e-5 in all arms,
patience10, FP32, B4×GA32, alpha.05, beta1 for risk arms/0 for CE arms,
score_chunk2. Four train-only samples plus gold, token dedup, shared frozen
baseline-generated pool; cost is 1-sentence_chrF0/100. These are fixed
development choices, not tuned optima. No test evaluation or parameter sweep.

Use `scripts/experiment.py inspect/build/cpu/launch/monitor --config <recipe>`;
launch uses `--account 1` on RTX. The detached queue runs CPU gate, full-shape
synthetic optimizer preflight, pool preparation, all four real-data GPU gates,
then sequential S0–S3 training. Each verified HF backup precedes local weight
cleanup. Failure stops the queue; it does not change batch/precision/config.
