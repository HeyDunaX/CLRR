# Pre-rerun provenance, selection and backup audit — 2026-10-05

## Scope

Read-only audit of `src/amis_rewire/{train,evaluation,scientific_records}.py`, the warmup worker, and existing metadata. No GPU or unit training was performed by this audit. Local Python ran in conda `clrr`; outputs are new under `outputs_rebuttal/amis_pre_rerun_20261005/`. Existing results and source files were not mutated by this auditor.

The root accumulation investigation is authoritative for numerical normalization. This report does not certify the root fix or assert that all historical runs used the current runtime. Source SHA256 values are frozen in `provenance_summary.json`; the findings below describe that inspected revision. Root fixes may supersede specific behaviors.

## Findings requiring a gate before rerun

| Finding | Evidence | Scientific/operational consequence | Required action |
|---|---|---|---|
| Historical wrapper/baseline accumulation mismatch | Root actual Trainer receipt and archived wrapper AST audit in G0/G1 | Existing checkpoint metrics remain valid measurements; matched CLI args alone do not make optimization fair | Resolve normalization and affected-cohort decisions before claiming superiority |
| Generic auto-resume is path-based, enabled by default | `train.py` CLI and `get_last_checkpoint(output_dir)` | A corrected run using an old folder can silently inherit incompatible optimizer/scheduler/history | Fresh unique corrected prefix, no auto-resume; explicit resume only with compatibility receipt |
| Per-epoch backup callback is a no-op | `CheckpointBackupCallback.on_save` returns without archive/upload despite recoverability docstring | Local resumable checkpoints exist, but remote per-epoch recovery is not provided by this callback | Do not promise per-epoch remote resumability; back up complete state if required |
| Generic final model upload can exhaust retries and return | `archive_best_model`, `safe_hf_upload` False branch only warns | Standalone CLI can finish with missing model ZIP on HF; metrics upload does not repair model archive | Fail closed on exhausted upload; verify remote size/SHA/revision before deleting local weights |
| Baseline best-step sources disagree | See selection table below | Replayed metric correctness cannot certify original best-selection chronology | Recover authoritative Trainer state/history; label unknown/conflicting rather than invent epoch |
| Scientific receipt labels beam1 regardless of CLI args | `ScientificRecordsCallback.on_save` fixed string | A run with eval_beams other than1 gets a misleading receipt | Write actual evalbeam/precision/protocol from args |

The corrected warmup v2 worker already uses fresh folders, `--no-auto-resume`, rejects incomplete folders, verifies final archive remote size/SHA and best reload/probes, uploads scientific records, then writes COMPLETE and deletes checkpoint folders. Its completed-run backup safety is stronger than the generic training CLI. Do not attribute the generic backup gaps to missing v2 final archives: those were verified.

## Baseline reuse: what metadata supports, and what it does not

Historical baseline seeds42/43/44 metadata agrees on LR5e-5, WD0, LRwarmup.06, train microbatch32, accumulation4, evaluation batch8, maximum lengths256,20epochs,patience4,BF16,gradient checkpointing, FP32 generation, evalbeam1/testbeam4, tl_XX→zh_CN. The model field pins the cache snapshot revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30` even though the later `model_revision` CLI field is absent. Original auto-resume is false.

This argument equality does not prove equivalent backward normalization, original runtime identity, batch ordering or exact original selection. Generation replay has independently passed and remains reusable as a measurement of those weights.

| Baseline | Historical `metrics.json` selected checkpoint | Corrected V5 inventory `best_step` | Conclusion |
|---|---|---:|---|
| Seed42 | Recovered selected best; original selected step unavailable | null | Unknown chronology; do not restore unsupported216 |
| Seed43 | checkpoint-720 | 252 | Conflicting metadata; neither suffix should be silently promoted to authoritative |
| Seed44 | checkpoint-612 | 504 | Conflicting metadata; require archive/history clarification |

V5 checkpoint metadata records model/tokenizer/generation config and hashes, plus exact prediction replay. These are strong inference receipts, but the displayed local metadata is not an original training runtime/selection receipt. Original confirmation metadata cannot be substituted for a freshly corrected training receipt just because test predictions reproduce.

### Minimum criteria before reusing FT as a corrected-cohort comparator

| Criterion | Minimum evidence |
|---|---|
| Dataset identity | Same train/validation/test SHA256, original references, no row changes |
| Initialization/tokenization | Same base model/tokenizer revision, language codes, prefixes, maximum lengths |
| Sampling | Same seed/data_seed, effective batch, accumulation, sampler/order and distributed size |
| Optimization | Same optimizer/scheduler/warmup/WD/clip; proven compatible CE normalization with the corrected wrappers |
| Runtime/precision | Pinned Trainer/PyTorch behavior, BF16/TF32/GC settings and optimizer implementation |
| Selection | Same predeclared raw-reference chrF++/beam/precision, epoch evaluation, patience/tie behavior, recoverable selected-best provenance |
| Weights and outputs | Exact archive identity, strict load audit, independent selected validation and test replay |
| Restart state | Fresh start, or exact full-state resume with source/config/data/runtime compatibility |

**Current decision: baseline reuse is conditional, not guaranteed.** If root fixes only wrapper loss-kwargs handling and verifies the native FT path is unchanged, old FT may be a useful historical comparator. Strong claims about a freshly controlled cohort still require the selection/runtime gaps above to be resolved or fresh baseline runs. There is no fixed rerun count before that decision.

## Schedule, seeding and best-selection behavior

| Code behavior | Assessment |
|---|---|
| `set_seed(seed)` before model construction; TrainingArguments seed=data_seed | Appropriate declared seed handling. This does not establish cross-runtime bitwise determinism. |
| Evaluation/save every epoch; `eval_chrf++`, greater-is-better, load_best_model_at_end | Consistent selection recipe. Warmup six-run audit confirms actual selected-best matches the logged metric maxima. |
| Warmup alpha uses optimizer global_step; evaluation/predict restores fixed alpha in finally | Correct for audited v2 fresh runs; resume train_begin resets alpha but next step_begin uses restored global_step, so no confirmed persistent restart-to-zero bug. |
| Scientific schedule JSONL appends | Suitable fresh runs. Resuming an interrupted step can repeat step records; analysis must not assume one unique line per update without a resume boundary receipt. |
| Milestone epochs1/4/10 | Correct predeclared milestones for v2 warmup20%. Generic end-of-warmup comment does not adapt to arbitrary warmup ratios. |
| Legacy fallback LRwarmup computes floor dataset/effective-batch steps | Conditional compatibility risk on runtimes without warmup_ratio; current pinned runtime has warmup_ratio, so not an observed current-run bug. |
| Duplicate `save_safetensors=False` keyword plus TypeError fallback | Redundant path: supported runtime first raises duplicate-keyword TypeError, then fallback creates arguments correctly. Not an unrecoverable error and not the source of the accumulation bug. Broad TypeError catch complicates diagnosis; one constructor call is preferable. |

## Backup and resume distinctions

| Artifact | What it contains | Can resume exact training? |
|---|---|---|
| Selected `best_model` / best ZIP | Selected weights/config/tokenizer/training_args | **No**, optimizer/scheduler/RNG not included |
| Scientific trajectory ZIP | Weights and metadata; optimizer/scheduler/RNG explicitly excluded | **No**, diagnostic forward/backward only |
| Local Trainer checkpoint directory | Model plus Trainer-managed optimizer/scheduler/RNG/state when normally saved | Potentially yes, only with runtime/code/config compatibility |
| Warmup COMPLETE + HF archive receipt | Verified remote selected weights and scientific records | Recovers completed-run evaluation; not exact continuation |

The generic `_save` delegates flattened wrapper state_dict to the underlying pretrained model and saves training_args; wrapper `load_state_dict` delegates back consistently. Existing best-reload tests support that inference save/load route. This audit finds no standalone reason to claim wrapper weights are missing. Future resume must still preserve the method configuration that reconstructs the parameter-free routing/LSR wrapper.

## Required preflight before expensive reruns

1. Root fixes and validates gradient normalization using actual Trainer, variable target lengths and accumulation, including incomplete accumulation boundaries where relevant.
2. Record corrected code SHA, exact runtime, data/model hashes and prospective selection protocol.
3. Use a unique corrected run namespace; reject existing unfinished folders unless explicitly compatible full-state recovery is selected.
4. Make final HF archive verification fail closed before COMPLETE/cleanup. Generic CLI and orchestration should have consistent guarantees.
5. Decide baseline reuse from evidence above, not from the expectation that baseline “must be unaffected.”
6. Keep old metrics labeled historical recipe; generate new results separately. Do not convert measurements of old weights into measurements of newly corrected training.

No GPU run or method upgrade is recommended solely by this provenance report. No Turkish experiment was started.

## Artifacts and reproduction

- `outputs_rebuttal/amis_pre_rerun_20261005/historical_configuration_inventory.json`: 32 existing configuration records with source hashes, selected-checkpoint fields and missing-key lists.
- `outputs_rebuttal/amis_pre_rerun_20261005/provenance_summary.json`: static source hashes, confirmed inspected behaviors and baseline selection conflicts.
- `outputs_rebuttal/amis_pre_rerun_20261005/provenance_audit.py`: read-only extractor, writes only the new directory.

```powershell
conda run --no-capture-output -n clrr python -X utf8 outputs_rebuttal/amis_pre_rerun_20261005/provenance_audit.py
```

Re-running after source fixes updates inspected source hashes to the new revision; historical argument records remain untouched. The written findings above are explicitly for the audited pre-repair behavior and should be supplemented by root fix receipts.
