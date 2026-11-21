# PLC v1: conditional RTX development pilot

Author approved implementation and conditional RTX execution on 2026-11-14.
This is an exploratory Amis→Mandarin mBART development experiment, not an
A100 paper result or a statistical confirmation. No test evaluation is queued.

## Implementation and recipe

- Implementation: `src/methods/prefix_consistency/v1/`.
- Shared training: `scripts/experiment.py` and `src/experiment_runner/engine.py`.
- Preparation: `scripts/analysis/prepare_prefix_consistency.py --config ...`.
- CPU tests: `tests/test_prefix_consistency.py` (five tests passed locally).
- Fixed RTX recipe: FP32, SDPA, TF32 off, microbatch4/accumulation32,
  LR5e-5, ES10, max20 epochs, validation BLEU beam1 selection, original CSV
  references, tl_XX→zh_CN. See the pinned protocol in each JSON recipe.
- All remote files stay under `/workspace/ura/thanhdat/projects/CLRR/`.

## Order and gates

1. Frozen preparation config:
   `configs/experiments/amis_prefix_consistency_preparation_rtx.json`.
   Queue started with PID2224993. Its generic launch receipt says
   `RTX_GATES_LAUNCHED`, but this config runs scientific preparation only:
   **zero optimizer updates and no long training**.
2. W42 diagnostic: all576 validation sentences ×8 fixed dropout pairs. Compute
   per-sentence P/D/E/cross=P−D. Average pairs within sentence, then bootstrap
   sentences10000 times. Positive lower95% bound on mean cross is required.
   Otherwise HOLD; do not open the pilot just because CPU tests passed.
3. If admitted, calibrate Wcal beta on fixed first64 training rows, batch4,
   at pretrained initialization, with identical masks for KL and combined
   gradients. No optimizer updates; restore fresh pretrained initialization
   for every actual training run. Freeze beta and admission receipt hashes
   in a separate training config before launching.
4. Every new run must pass CPU and real GPU two-step optimizer/save/load/resume
   gates. All GPU gates must finish before the suite opens long training.
5. Detached suite runs sequentially, uploads each selected checkpoint to the
   existing public HF repo, verifies backup, then uses existing safe cleanup.

## Conditional training inventory

| Run | Seed | Training objective | Role |
|---|---:|---|---|
| P |42|W +0.10 ordered prefix P|Candidate|
| D |42|W +0.10 diagonal D|Same diagonal weights; removes cross-position terms|
| E |42|W +0.10 endpoint E|Checks whether prefixes add value beyond final score|
| Wcal |42|Ordinary full-vocabulary R-Drop, calibrated beta|Initial aggregate gradient-norm control|
| Middle |42|Existing Middle Alignment, layer6, tau0.1, weight0.1|New matched RTX comparator; A100 number is not interchangeable|

Baseline and W42 are reused after provenance checks. No matching RTX Middle
run was found in local research/config inventory or remote run metadata;
the five-run inventory includes it to answer the author's comparison request.
Middle uses the same microbatch4 negative pool as this RTX pilot; it is not
an exact reproduction of the larger-microbatch A100 comparator.

Reference development validation BLEU: baseline42=17.682329; W42=18.272410.
Recheck their actual receipts when producing the final comparison table.
Do not substitute test scores or A100 numbers into that table.

## Decision after validation

P must exceed W by at least0.30 validation BLEU and beat D/E/Wcal to justify
conditional replication. Also report P−baseline and P−Middle, plus chrF++
as a secondary metric. One seed does not establish statistical significance.
Failure means report the result and stop; no automatic parameter sweep,
test opening, multi-seed expansion, Ê Đê run, or A100 launch.

## Storage

Public repo: `FiveC/amis-rewire-checkpoints`.
Checkpoint pattern:
`checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu/`.
Use method names prefix/diagonal/endpoint/calibrated-consistency-v1 and
middle-alignment. Never overwrite an existing best.zip. Keep only small
JSON/CSV/Markdown receipts locally; do not download model weights locally.


## Current execution receipt

Preparation completed and independently verified: CSV SHA256, sentence bootstrap
and beta calibration arithmetic agree (floating-point tolerance1e-12).
W42 cross95% lower bound=+0.0005359038, a narrow PASS.

Training configs are now frozen:

- `configs/experiments/amis_prefix_consistency_pilot_rtx.json` (P,D,E,Wcal).
- `configs/experiments/amis_middle_comparator_rtx.json` (Middle).

Both local CPU receipts PASS. Detached suite PID2228354, remote workspace
`outputs/experiments/amis_prefix_consistency_pilot_rtx_suite/`.
Its order is CPU checks → all five GPU optimizer gates → P,D,E,Wcal,Middle
training. Queue errors stop the chain; no failing gate is bypassed.

Launch command (already executed; do not launch a duplicate):

```powershell
conda run -n clrr python scripts/experiment.py launch-suite --configs configs/experiments/amis_prefix_consistency_pilot_rtx.json configs/experiments/amis_middle_comparator_rtx.json
```

Monitor either config with `scripts/experiment.py monitor --config ... --account 1`.
The four public preparation artifacts were uploaded and read-back hash verified:
`analysis/amis_mandarin/mbart-large-50/prefix-consistency/seed-42/preparation/`.

Baseline42's original selector was chrF++, but its full20-epoch validation log
independently shows the same earliest BLEU maximum at step684 (epoch19), so
checkpoint selection is equivalent on that completed trajectory. Its SDPA
backend is reconstructed from historical runtime, not a contemporaneous field.
W42 has an explicit SDPA/BLEU receipt. Configs retain these provenance details.


## Reuse policy reaffirmed by author

Before scheduling any additional RTX training, check both local/remote completed
receipts and HF backups, including results already removed from the RTX disk.
Reuse requires matching dataset hashes, backbone revision, method/formula and
weights, seed, optimizer/schedule, microbatch/accumulation, precision/backend,
ES/maxepochs and checkpoint selector (or documented trajectory equivalence).
A different folder name alone does not justify rerunning. Missing metadata is
not proof that no previous run exists; record uncertainty rather than relabel it.

Current five-run inventory was checked against RTX completed VAL_COMPLETE
receipts and the five method prefixes on HF. No completed matching RTX runs
were found. Middle has historical checkpoints, with the documented A100 ES10
metadata; that record does not replace the matched RTX comparator. Baseline/W
remain reused. Inventory receipt:
`outputs/experiments/amis_prefix_consistency_pilot_rtx_suite/REUSE_INVENTORY.json`.
Queue PID2228354 remains alive; P GPU gate passed, D gate in progress at audit.


## PLC RTX pilot completed — 2026-11-15

All five runs have VALIDATION_COMPLETE_VERIFIED and public HF backup receipts.
Both queues report QUEUE_COMPLETE; suite log ends SUITE_TRAINING_COMPLETE.
No test was evaluated. These are exploratory seed42 validation beam1 results.

| Model | BLEU | chrF++ | Delta BLEU vs baseline | Delta BLEU vs W |
|---|---:|---:|---:|---:|
| Baseline (reused) | 17.682329 | 14.703005 | +0.000000 | -0.590080 |
| W (reused) | 18.272410 | 15.172807 | +0.590080 | +0.000000 |
| P_PREFIX_S42 | 17.714812 | 14.845499 | +0.032482 | -0.557598 |
| D_DIAGONAL_S42 | 18.271456 | 15.179737 | +0.589127 | -0.000954 |
| E_ENDPOINT_S42 | 17.907733 | 14.970888 | +0.225404 | -0.364677 |
| WCAL_S42 | 17.241326 | 14.421296 | -0.441004 | -1.031084 |
| MIDDLE_S42 | 16.677522 | 13.937579 | -1.004807 | -1.594887 |

PLC fails its predeclared gate: P−W=-0.557598 (<+0.30), and P is below D/E.
P−baseline=+0.032482 and P−Middle=+1.037289 do not rescue the primary failure.
D is effectively tied with W (delta=-0.000954 BLEU); no significance claim follows.
The narrowly positive frozen-W cross-drift diagnostic did not translate into a
benefit from penalizing ordered prefixes at lambda0.10. This does not prove that
cross-position covariance is universally harmful or that other weights would work.
Wcal is lower than W by1.031084 BLEU; this initial gradient-norm match does not
preserve optimizer directions or trajectories. Do not infer that every stronger KL
coefficient fails from a single seed. Middle uses batch4 negatives and is not the
A100 batch32 comparator. Baseline backend retains its historical provenance caveat.

Decision: NO-GO for automatic PLC replication/A100 promotion or further sweep.
No new training/test launched. Preserve all results, including unsuccessful controls.

All five HF best.zip SHA256 metadata values independently match completion receipts;
no .pt/.bin/.safetensors/.zip files remain in the five remote run directories.
Only small JSON receipts were downloaded locally. Verification artifacts:
`outputs/experiments/amis_prefix_consistency_pilot_rtx_suite/HF_COMPLETION_VERIFICATION.json`
and `LATEST_COMPLETE_STATUS.json`.
