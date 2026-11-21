# Learned-donor CE control v1

Method ID: `alignment-learned-donor-ce-v1`.

Matched control for C0: `u + .1 * [sg(h[l-2]) + A(sg(h[l-2]))]`.
Only A's input differs from C0's `A(sg(h[l-2]) - u)`. Rank64, zero-up
initialization, private module RNG, CE plus target dummy pass, and all baseline
RTX training settings are unchanged. Full backbone fine-tuning; not PEFT.

The isolated encoder loop deliberately preserves frozen v1 code. A source-level
test enforces that only the correction-input expression differs. Independent
forward, nonzero-gradient, dropout RNG, checkpointing and reload tests also run.
This is an adapted learned-skip comparator, not an exact SkipCLM reproduction
or a new method claim. Validation-only seed42 screening; no automatic test.

Recipe: `configs/experiments/amis_learned_donor_ce_rtx.json`.
Use `scripts/experiment.py build`, `cpu`, then `launch --account 1` as documented
in `docs/operations/ALIGNMENT_RESIDUAL_CE_CONTROLS_RTX.md`.
