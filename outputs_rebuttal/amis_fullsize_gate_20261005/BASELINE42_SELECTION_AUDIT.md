# Baseline seed42 original selection audit — 2026-10-05

## Decision

**Original selected step remains UNKNOWN.** The selected weights and their predictions are verified, but the available records do not reconstruct the completed run's selection chronology. Do not label checkpoint216 or504 as the final selected best.

For a fresh cohort with fully auditable selection, rerun **baseline seed42 separately from pretrained**, using the prospectively locked corrected recipe. Preserve the old seed42 scores as historical recovered-weight measurements. This is a recommendation, not a launched training run. It does not establish whether seeds43/44 can be reused under the corrected optimization contract; that is a separate gate.

## What was actually checked

| Evidence | Observation | What it establishes |
|---|---|---|
| Original local baseline42 log | Evaluation records through step504/epoch14; log cuts off during step540/epoch15 evaluation; no final selected-checkpoint JSON | Partial history only |
| Four surviving local trainer states | Steps216,252,468,504; best metrics11.7766,11.7965,13.1806,13.3450 | Selection as of those partial checkpoints, not after completed training |
| Last preserved trainer state504 | Best step504, epoch14, chrF++13.3450 | Best known at epoch14, not final selected archive identity |
| Recovery metadata | Explicitly says original selected step unavailable and late endpoint probe lost | The recovery did not reconstruct chronology |
| Recovery script | Loads archived selected weights, regenerates predictions; writes descriptive unknown best-step label | Inference recovery, not a training-state recovery |
| Selected ZIP receipt | Archive SHA256 `aaed1e8cf39b4f1ad6a5f4bef6fa04d44fbeb9fb2599e2649702e937822a10c8`, size2,270,664,334 bytes | Identity of the preserved selected-weight archive |
| HF repository metadata | Matching confirmation namespace contains the baseline42 ZIP; no matching small state/selection file found | The inspected namespace has weights, not the missing selection record |
| Remote ZIP central directory | Successfully inspected with HTTP206 and **65,536 bytes**; eight members, no `trainer_state.json`, optimizer, scheduler or RNG file | This selected archive cannot provide the missing Trainer chronology |

Remote ZIP members are weights, model config, tokenizer files, generation config and `training_args.bin`. Training arguments describe the declared recipe, not the selected global step or full validation history. No weights were downloaded by this audit. ZIP64 sentinel size fields are recorded as unknown in the receipt rather than mistaken for actual file sizes.

## Interpretation boundaries

- The old seed42 test BLEU/chrF++ and corrected native validation generation remain measurements of real preserved weights. The missing step does not make those predictions fabricated or numerically invalid.
- A recovered checkpoint's test-score agreement does not establish how its selection metric evolved over all20 epochs.
- The strongest surviving pre-incident state is not a valid replacement for the completed run's selected-best record.
- HF search was restricted to matching baseline42 files in the confirmation namespace. This is not proof that no external backup exists anywhere; the local recovery receipts explicitly document the loss, and the preserved selected ZIP lacks the needed record.
- No model was trained, no GPU used, no original source or scientific scores edited.

Machine-readable evidence: `baseline42_selection_trace.json`. Reproduction script: `trace_baseline42_selection.py` (local audit by default; `--hf` adds authorized metadata listing and a bounded ZIP directory HTTP Range read).
