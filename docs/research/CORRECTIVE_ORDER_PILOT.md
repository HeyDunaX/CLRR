# Corrective-order consistency: six-arm RTX pilot

## Scope and authorization

Author approved implementation and the RTX queue after reviewing the six-run
plan. This is development on Amis, seed42, validation-only. It is not an A100
paper cohort, a novelty certificate, or an independent holdout confirmation.
Historical methods and scores are unchanged. No test is authorized in this queue.

## Definition

On the same supervised TRAIN microbatch D, sample two native dropout masks a,b.
Replay each mask identically at every occurrence, including backward recompute.
Let ell_a,ell_b be valid-token mean CE, U_a(theta)=theta-eta grad ell_a(theta).
All backbone parameters participate; inner SGD is unclipped and differentiable.

    theta_ab = U_b(U_a(theta)); theta_ba = U_a(U_b(theta))
    R_order = SKL(p(theta_ab,Q),p(theta_ba,Q)) / eta^4
    L = (ell_a+ell_b)/2 + lambda R_order

Q is another TRAIN microbatch from a fixed randomized derangement shared by
every arm. These probes are not curated semantic pairs and do not change CSVs.
Dropout is OFF on Q. Decoder inputs use the gold TRAIN prefix. SKL is full-vocab,
averaged over labels != -100, with FP64 log-prob arithmetic, not top-k sampling.
The real optimizer is the common AdamW. No virtual optimizer state or update is
committed. Exact outer differentiation requires second derivatives, no detached
inner gradients, no restricted trainable layer subset, no first-order substitute.

## RQ and priors

RQ: does consistency of supervised corrective trajectories generalize better
than pre-update prediction consistency under stochastic subnetworks?

- R-Drop (2021): https://arxiv.org/abs/2106.14448
- Commute Your Domains (2025): https://arxiv.org/html/2501.15556
- The Geometry of Sequential Learning (2026): https://arxiv.org/html/2606.24993

The bracket expansion Hb ga - Ha gb is prior mathematics, not our contribution.
The candidate is a functional endpoint regularizer for same-label dropout paths.
Related-work review retains it as NEAR prior, not a claim that nobody did this.
NotebookLM notebook bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890 contains the earlier
theory audit. Subsequent design query f1b31319f3dd was read skeptically: its
proposed pab-vs-midpoint control was rejected (same bracket, just rescaled), its
minus sign in the midpoint expansion was corrected, and unsupported FLOP/RAM
estimates are not used as measured facts.

## Six arms (all seed42)

| ID | Objective | Purpose |
|---|---|---|
| R0_CE2_S42 | two-view mean CE | lambda=0 matched objective control |
| R1_RDROP_S42 | two-view mean CE +1.25 SKL before update | nearest literature comparator |
| R2_ORDER_S42 | main at calibrated lambda0 | central candidate |
| R3_ORDER_LOW_S42 | main at lambda0/3 | predeclared lower sensitivity |
| R4_ORDER_HIGH_S42 | main at 3 lambda0 | predeclared higher sensitivity |
| R5_CURVATURE_S42 | symmetric curvature penalty | order vs symmetric second-order effect |

For R5, theta_sym=(theta_ab+theta_ba)/2,
theta_par=theta-eta(ga+gb), R_sym=SKL(p(theta_sym,Q),p(theta_par,Q))/eta^4.
Its leading parameter difference is +eta^2(Hb ga+Ha gb)/2. This differs from
the antisymmetric bracket; it is not an isolated proof of causality if it loses.
Match its initial auxiliary gradient scale independently using TRAIN only.

## Numerical calibration (no optimizer steps)

Use three fixed TRAIN microbatches. On the first, examine eta=.003,.01,.03 with
fixed masks. Require adjacent positive finite R and ||delta theta||/eta^2 ratios
within [.5,2]. Choose the smaller eta in the first valid pair, otherwise STOP.
This is a declared numerical screen, not proof that order sensitivity is harmful.
For order and symmetric control separately, set lambda0 to the median of
0.10*||grad CE||/||grad R|| across three TRAIN batches. The 10% ratio is a fixed
engineering starting scale, not an established optimum. Freeze calibration with
bundle hash; no recentering from validation. All gates must PASS before any run.

## Fairness and runtime

Same pretrained mBART revision, CSV hashes, LR5e-5, FP32/TF32 OFF, B4/GA32,
max lengths256, AdamW fused, linear warmup .06, clip1, max20epochs, ES10,
original-reference validation BLEU beam1 and earliest-best tie policy.
Track BLEU, chrF0, chrF++; no test. Frozen lock:
configs/protocols/amis_corrective_order_rtx.json.

Explicit execution differences from old W: SDPA math kernel only; whole
functional-call non-reentrant checkpointing; saved tensors offloaded to CPU.
Old W is a historical reference, not the formal matched comparator. R1 reruns.
All six arms use these execution settings. CPU offload must preserve gradients;
host admission preserves at least 6GiB RAM, checks before large allocations.
No automatic microbatch, length, precision or parameter-subspace changes.
Per-step timing, peak GPU memory, calibration diagnostics and loss components
are retained. Equal optimizer-step protocol does NOT imply equal compute.
Compute-matched W is a follow-up if the method shows promise, not silently an
extra model here. No runtime forecast until measuring the actual GPU gate.

## Queue and artifacts

    conda run -n clrr python scripts/experiment.py build --config configs/experiments/amis_corrective_order_rtx.json
    conda run -n clrr python scripts/experiment.py cpu --config configs/experiments/amis_corrective_order_rtx.json
    conda run -n clrr python scripts/experiment.py launch --config configs/experiments/amis_corrective_order_rtx.json --account 1 --stage gate-train
    conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_corrective_order_rtx.json --account 1

Remote project ONLY /workspace/ura/thanhdat/projects/CLRR including conda/cache.
Queue: outputs/experiments/amis_corrective_order_rtx/queue.log.
Stages: remote CPU -> train-only calibration -> ALL six GPU gates -> six
validation training runs -> verified public backup -> delete backed-up weights.
One project lock, no other-user process manipulation, no shutdown of shared RTX.
An exception prevents subsequent stages; scientific numerical failures are not
silently bypassed. No model is called trained merely because gates ran steps.

HF repository FiveC/amis-rewire-checkpoints, beneath
checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/
rtx6000-fp32-sdpa-math-offload-b4-ga32-es10-select-bleu/<variant>/.
Each trained checkpoint gets native weights, method definition, validation CSV,
metrics, selected/completed step, resolved args and verified upload receipts.
Only small JSON/CSV/Markdown receipts are downloaded locally by monitor.
No A100 scores overwritten; no existing checkpoint target reused.

## Decision after pilot

Compare on validation to both CE2 and R-Drop, assess sensitivity and curvature
control, and report effect sizes plus compute rather than a seed42 significance
claim. Lock one candidate before seeds43/44; test only after explicit selection
lock. Existing Amis has already served development. No promise of acceptance or
causal mechanism solely from one high score.
