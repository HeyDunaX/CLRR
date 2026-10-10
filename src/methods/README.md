# Method source layout

Each method has one implementation here. `amis_rewire` retains the training,
data and evaluation pipeline; old modeling imports are compatibility shims.
`method_candidates` also re-exports implementations for existing workers.

```text
methods/
├── registry.py
├── common/helpers.py
├── original/
│   ├── clrr.py                 # Original detached, post-addition donor
│   ├── lsr.py                  # Original sentence-level cosine LSR
│   └── factory.py              # Existing load_model API and CLI method IDs
├── precache/v1/clrr.py         # Pre-addition detached donor (U1)
├── centered_lsr/v1/lsr.py      # Microbatch centered cosine (U2)
├── attached_source/v1/clrr.py  # Existing stop-gradient ablation control
└── token_lsr/v1/README.md      # Planned; no implementation yet
```

## Method IDs

`methods.registry.wrap_method(base, spec)` wraps an already loaded backbone.
Existing training CLI IDs and `load_model` arguments remain unchanged.

| Method ID | Existing candidate alias | Composition |
| --- | --- | --- |
| `baseline` | — | Backbone only |
| `original-clrr` | — | Original CLRR |
| `original-lsr` | — | Original LSR |
| `original-full` | — | Original CLRR + original LSR |
| `precache-clrr-v1` | `clrr-precache` | Precache CLRR |
| `precache-full-v1` | `full-precache` | Precache CLRR + original LSR |
| `centered-lsr-v1` | `lsr-centered` | Centered LSR |
| `centered-full-v1` | `full-centered` | Original CLRR + centered LSR |

The registry keeps candidate distance `2`, strength `.10` and LSR weight `.10`:
these are the existing frozen recipes. The original `load_model` factory still
accepts explicit configuration, including encoder/decoder routing.

## Scientific definitions and checkpoint loading

This reorganization preserves existing formulas. Precache donors are detached
before addition, but still come from the routed pass. Centered LSR uses FP32
masked sentence means, separate source/target microbatch means, epsilon `1e-6`,
and a detached target. Accumulation does not enlarge the centering batch.
A singleton batch has constant centered auxiliary loss. This variant does not
guarantee prevention of collapse. Target dropout and Full target routing remain.

Load backbone weights, then reconstruct the exact wrapper and settings from
the checkpoint run receipt. A plain backbone does not reproduce CLRR or LSR.
State-dict keys and old import paths are preserved. No scores are recomputed;
no trained checkpoint or result is moved by this source edit.

New training receipts hash the `methods` tree. Bundle builders collecting all
`src/**/*.py` include it automatically; explicit Turkish source inventories
also include it. Old receipts/manifests remain immutable: use their pinned source
bundle to resume an old run. New hashes are not substituted into old receipts.

## Adding the next method

Implement Token LSR under `token_lsr/v1/` when approved. Add its ID to the
registry after correctness checks; it is currently unavailable. Future changes
to a tried variant belong in `v2/`, leaving `v1/` reproducible.

Worker: `scripts/revalidation/run_method_candidates.py`.
Gates: `tests/smoke_method_candidates.py`, `tests/smoke_attached_routing.py`,
`tests/test_rerun_safety.py`.

## Refactor verification — 2026-10-10

Verified offline on CPU in conda `clrr`:

- Exact before/after loss, logits, parameter gradients, state-dict keys and
  beam-2 generation for original CLRR, LSR, Full, precache CLRR/Full,
  centered LSR/Full, and attached-source control (eight combinations).
- Four candidate gates: checkpoint gradients, accumulation=4, finite gradients,
  zero added parameters, saving/restoring and loading the best checkpoint.
- Attached-source forward/checkpoint gradient parity; ten resume/backup safety
  tests; old factory monkeypatch compatibility; both package import namespaces;
  NLLB imports and setuptools discovery of the new packages.

The candidate gate used a temporary manifest for the refactored source and
copied run specifications. Historical manifests and gate artifacts were not
overwritten. These CPU checks verify the source refactor, not model quality
or an A100 training run.
