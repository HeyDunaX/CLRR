# Route/source interaction v1 — prototype

Design, formulas, priors, risks and gate results:
[GROUNDED_INTERACTION_DESIGN_AND_PREFLIGHT.md](../../../../docs/research/GROUNDED_INTERACTION_DESIGN_AND_PREFLIGHT.md).

Stable IDs: `route-paired-ce-v1`, `route-source-contrast-v1`,
`route-source-interaction-v1`. Original CLRR routing is retained; inference
uses route ON. No LSR or added parameters. Whole-view non-reentrant
checkpointing is explicit and differs from the old block implementation.

Technical prototype only. Pilot training and main-method novelty remain HOLD.
No training recipe or queue has been launched.

Correctness (conda `clrr`):

```bash
python -m unittest discover -s tests -p test_grounded_interaction.py -v
python scripts/revalidation/smoke_grounded_interaction.py --device cpu --scale tiny --microbatch 4 --length 16 --accumulation 4 --output outputs/experiments/grounded_interaction_preflight/manual_cpu.json
```

The smoke command only uses synthetic random weights, forward/backward;
it takes no optimizer steps and refuses to overwrite an existing receipt.
Check shared GPU free VRAM before using `--device cuda` and keep all files,
environment and caches inside the approved project workspace.
