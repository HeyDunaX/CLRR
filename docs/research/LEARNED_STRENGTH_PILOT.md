# Learned-strength CLRR — approved engineering pilot

## Current decision (2026-11-10)

ENGINEERING_PILOT_AUTHORIZED_NOT_NOVELTY_CLAIM. The author approved two Amis
seed42 runs on RTX6000 after CPU and both GPU gates PASS. This approval supersedes
the training HOLD for this engineering control in CONTINUATION_REVIEW_AND_NEXT_DESIGN.md;
it does not change that document's negative verdict on standalone ACL novelty.

Prior review: ReZero (UAI2021, https://proceedings.mlr.press/v161/bachlechner21a/),
Lexical Shortcuts (WMT2019, https://aclanthology.org/W19-5211/), and
Value Residual Learning (ACL2025, https://aclanthology.org/2025.acl-long.1375/)
already cover zero-init residual scalars, gated NMT shortcuts and residual mixing.
Applying these ideas to CLRR is an engineering/mechanism pilot, not certified novelty.

## Formula and controls

For encoder layers l=3..12: h_l = F_l(h_(l-1)) + a_l * stopgrad(h_(l-2)).
Donors are post-rewire; ten unconstrained signed trainable scalars start at zero.
P1 uses CE. P2 uses CE + .1 original LSR, including the original train-mode
target encoder and eval-loss behavior. No Transition LSR, extra warmup, clipping,
layer selection, lambda sweep or separate scalar LR. All parameters use LR5e-5.
The pretrained mBART revision is the shared pinned revision, not a trained baseline.

Frozen Amis recipe: train4600/validation576; tl_XX->zh_CN; SDPA FP32/TF32off;
microbatch4/accumulation32; source/target256; ES10/max20; original-reference
validation BLEU greedy beam1, earliest best; epoch checkpoints only. Step-zero
is a correctness gate, not a candidate in checkpoint selection, matching the
existing baseline. No test stage, no new seed or Ede/A100 expansion authorized.

Reuse baseline validation BLEU17.682329174532335 and chrF++14.703005362215766.
Its complete trajectory established equivalent BLEU/chrF++ selection; historical
SDPA evidence is reconstructed rather than contemporaneously recorded. Old
baseline/CLRR/Full artifacts and scores are not overwritten.

## Prospective development gates

At least one candidate: validation BLEU >= baseline+.30 and chrF++ >= baseline.
Retain an LSR contribution hypothesis only if P2-P1 BLEU >=+.20.
Fail: stop without a gate/layer/lambda sweep. Pass only proposes seeds43/44,
then Ede; one seed is not statistical confirmation. Amis is development data.
These are new prospective thresholds; previous Transition NO-GO is unchanged.

## Correctness and execution

CPU: zero-strength logits/generation equal backbone; nonzero scalar gradients;
skip donor detached but sequential gradients preserved; gradient checkpointing
with two accumulated microbatches matches no-checkpoint gradients for both arms;
target pass cannot corrupt captured source donors; save/load persists scalars.
GPU: exact FP32/SDPA/b4/ga32/reentrant recipe, two optimizer steps, nonzero learned
scalars, Trainer best reload, exported reload, beam cache parity, resume parity.
All GPU gates precede training. Resource admission needs >=20000 MiB free;
never change another user's processes. Artifacts stay inside the CLRR project.
Checkpoint public prefixes are declared in the JSON recipe; delete remote
training weights only after verified SHA backup in FiveC/amis-rewire-checkpoints.

Reproduce using configs/experiments/amis_learned_strength_rtx.json:

```bash
python scripts/experiment.py build --config configs/experiments/amis_learned_strength_rtx.json
python scripts/experiment.py cpu --config configs/experiments/amis_learned_strength_rtx.json
python scripts/experiment.py launch --config configs/experiments/amis_learned_strength_rtx.json --account 1 --stage gate-train
```

Use conda clrr. The RTX queue runs detached and stops after both validation runs.
