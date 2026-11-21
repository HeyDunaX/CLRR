# Continuation-Marginal Consistency (CMC): locked RTX pilot

Status: author authorized implementation, gates and three Amis seed42 pilots.
This is development, with no test evaluation and no automatic replication/A100.

## Motivation and scope

Native two-view R-Drop W is the existing strongest reusable consistency control.
Gold/rest-only and prefix modifications did not establish an improvement over W.
CMC asks whether imposing agreement after a shared continuation channel preserves
useful current-token variation while retaining agreement about consequences.
This is a hypothesis, not an established cause of earlier failures.

Research question: does consequence-level dropout consistency improve translation
over current-token consistency, beyond merely reducing its gradient magnitude?

## Prior-work screen completed before implementation

NotebookLM notebook `bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`, completed query
`902f589ac093`, compared the proposed formula with R-Drop, TeaForN, SEARNN,
WASD, TOPD and RSD. The consulted sources did not explicitly specify this shared
detached continuation pushforward of two dropout distributions. This supports a
controlled pilot; it does not certify exhaustive novelty or an ACL contribution.
The KL chain-rule identity and data-processing inequality are established math.

Closest primary references:

- R-Drop: https://arxiv.org/abs/2106.14448 — symmetric KL on current-token outputs.
- TeaForN: https://aclanthology.org/2020.emnlp-main.702/ — multi-step teacher forcing.
- SEARNN: https://arxiv.org/abs/1706.04499 — learning with roll-in/roll-out costs.
- TOPD: https://arxiv.org/html/2606.00305v1
- WASD: https://arxiv.org/html/2610.07706v1
- RSD: https://arxiv.org/html/2610.33012v1

The continuation channel is not a semantic-equivalence oracle: a common next word
can follow prefixes with opposing meanings. Test quality and mechanism controls
must decide whether this representation is useful.

## Implemented formula (sparse hybrid approximation)

Two independent dropout passes give p1(a), p2(a) at the same gold prefix.
The channel K(u|a) is a third, dropout-OFF native forward, detached from autograd.
The ideal objective is mean two-view CE + beta SKL(p1 K, p2 K).

To avoid a vocabulary-sized number of decoder forwards, define A as the union of
top4 actions in each view plus the gold token, excluding EOS. For a in A, use K.
Each a outside A maps to its own distinct identity symbol; EOS therefore remains
a dedicated termination identity. These symbols are disjoint from future words.
This is a stochastic hybrid kernel with ALL probability mass preserved. No
shortlist renormalization and no collapsed 'other' bucket. Its SKL is exactly:

    SKL(sum_A p1(a) K(.|a), sum_A p2(a) K(.|a))
    + 0.5 sum_{a outside A} (p1(a)-p2(a)) (log p1(a)-log p2(a)).

The first arguments are sub-probabilities with their original masses. Detached
selection/channel does not prevent gradients through either view's probabilities.
No new parameters. Inference is the original native mBART forward/generation.

CE still uses every supervised target token. For the auxiliary loss, sample one
uniform supervised position per sentence and weight by sentence token count,
dividing by the total supervised tokens in that microbatch. This estimates the
same token-average auxiliary objective. Masks include supervised specials, as W.
Sampling forks/restores the active RNG after the two dropout forwards. Full RNG,
optimizer and scheduler restoration is tested by the shared resume gate.

## Three runs and controls

| ID | Method | Purpose |
|---|---|---|
| CMC_S42 | continuation-consistency-v1 | Shared consequence-level agreement |
| NORM_S42 | continuation-norm-control-v1 | Current-token SKL at the same sampled positions, scaled by detached ratio of CMC/current logit-gradient norms |
| PERMUTED_S42 | continuation-permuted-v1 | Rotate K rows within each position; same channel cost and shortlist, broken action-to-consequence mapping |

All runs perform the same channel probes. The norm control uses **per-position
local calibration**, rather than an initial fixed beta calibration: this is an
explicit engineering refinement of the proposed strength-matched R-Drop control.
It matches auxiliary gradient magnitude in logit space, not parameter or Adam
update space. It is not a claim of identical training dynamics.

Locked: beta1.25, top_k4, probe_chunk2; Amis4600/576, mBART50 pinned revision,
FP32 SDPA RTX6000, micro4/GA32, LR5e-5, 20 epochs, ES10, validation BLEU beam1,
original CSV references. No data or baseline recipe changes.

## Admission and success criteria fixed before results

1. CPU: explicit dense stochastic-kernel equivalence, gradients through both
   probabilities and none through K, data processing, identity equivalence,
   probe versus full native forward, dropout flags/RNG, checkpointing with
   accumulation, native inference, save/reload and recipe rejection.
2. Frozen W checkpoint: verify pinned archive SHA; 32 fixed sampled TRAIN rows,
   two fixed dropout seeds, zero optimizer steps. Require nonconstant channel
   (mean row L1 deviation >1e-6), positive current SKL, and aggregate retained
   CMC/current SKL ratio in [0,.99]. Otherwise HOLD; no long training.
   This measures nontriviality only, not preservation of semantic distinctions.
3. ALL three GPU gates must PASS before ANY long run: exact FP32/SDPA,
   micro4/GA32/reentrant checkpointing, two optimizer steps; save/reload,
   optimizer/scheduler/RNG resume agreement, native cached/uncached beam4,
   finite active auxiliary and nonconstant channel. Gate beam4 is a train-row
   correctness fixture, not test evaluation.
4. Primary pilot criterion: CMC validation BLEU >= reused W42 +0.30 and greater
   than both new controls. Report chrF++ and baseline/Middle as secondary. One
   seed is exploratory and cannot establish cross-seed statistical significance.
5. No automatic test, extra seeds, method switch or A100 queue. Review results.

Public checkpoint policy: canonical repo `FiveC/amis-rewire-checkpoints`, path
`checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`.
The shared engine uploads each completed model and verifies remote SHA before
deleting ONLY its own transient weights. Small diagnostic receipts remain.

## Commands

Run Python in conda clrr. Configuration is
`configs/experiments/amis_continuation_consistency_rtx.json`.

    python scripts/experiment.py build --config configs/experiments/amis_continuation_consistency_rtx.json
    python scripts/experiment.py cpu --config configs/experiments/amis_continuation_consistency_rtx.json
    python scripts/experiment.py launch --config configs/experiments/amis_continuation_consistency_rtx.json --account 1
    python scripts/experiment.py monitor --config configs/experiments/amis_continuation_consistency_rtx.json --account 1

The detached queue owns the project training lock. It runs CPU -> frozen-W probe
-> all GPU gates -> three serial training runs -> verified uploads. Any failed
stage stops the queue. All files/environments remain inside the authorized RTX
project `/workspace/ura/thanhdat/projects/CLRR`.
