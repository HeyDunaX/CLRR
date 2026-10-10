> **Entrypoints sau khi gom code (10/10/2026):** các lệnh launcher/monitor/prepare riêng đã hoàn thành bên dưới là lịch sử. Run mới dùng [EXPERIMENT_WORKFLOW.md](EXPERIMENT_WORKFLOW.md) và `scripts/experiment.py` với args/config. Để replay run cũ cần bundle nguồn đã pin; không sửa hash receipt cũ. Số thực nghiệm và kế hoạch nghiên cứu bên dưới không đổi.

# Amis method candidates: seed42 validation pilot

## Authorized scope

Four models, mBART-50, Amis→Mandarin, Colab 1/2 A100 Standard 40GB, conda `clrr`.
No Turkish runs, no extra seeds, and no test evaluation in this pilot.

| Run | Method | Account | Question |
| --- | --- | --- | --- |
| U1C | CLRR-precache | 1 | Does using pre-addition donors help CLRR alone? |
| U1F | CLRR-precache + historical LSR | 1 | Does the donor change help Full? |
| U2L | Centered-LSR | 2 | Does centered sentence alignment improve validation translation? |
| U2F | Historical CLRR + Centered-LSR | 2 | Does centered alignment help Full? |

Each account executes gate→train for its first run, then gate→train for its second.
Queue failure stops that account; the other account can continue.

## Locked protocol and baseline fairness

Pinned mBART revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`, seed/data seed42,
LR5e-5, AdamW fused, weight decay0, linear scheduler, warmup0.06, clipping1,
microbatch32, accumulation4, 20 epochs maximum, patience10, lengths256/256,
BF16 training, TF32 training, reentrant checkpointing, eval batch8, workers0.
Source language proxy `tl_XX`, target `zh_CN`; no source prefix.
Checkpoint selection: original CSV validation references, chrF++, beam1,
FP32 generation with TF32 disabled; ties retain earliest best checkpoint.

Baseline seed42's receipt matches these training/generation settings and dataset
hashes. It reached epoch20/step720 and selected step720 (validation chrF++
13.901540564847089). Its original patience4 label is preserved. Because patience
did not truncate that run, raising it to10 does not extend the same trajectory
beyond the epoch20 cap. The reuse audit is embedded in `PLAN.json`.

Historical CLRR-only and LSR-only did stop with ES4. Comparisons against those
are exploratory, not fully matched ES10 comparisons. If needed, obtain approval
for at most their two corresponding ES10 control runs; they are not in this queue.
One seed cannot establish statistical superiority or ACL novelty.

## Code and verification

New methods: `src/method_candidates/`; historical methods are unchanged by this
pilot. `wrap_candidate` reconstructs the method from the checkpoint receipt.
The worker reuses the previously verified controls engine for training,
checkpoint export, strict reload, metric calculation, and public backup.

CPU gate checks donor formula/detach, centered formula and gradients,
target stop-gradient, degenerate-batch finiteness, checkpointing gradients,
Trainer accumulation4 scaling, and save/reload for all four methods.
Full-size GPU gate requires A10040GB, BF16, exact microbatch32/accumulation4,
two optimizer steps including long examples, optimizer/RNG checkpoint files,
and FP32 logits consistency after reloading. Training requires both gates PASS.

## Launch and monitor

Use the existing conda `clrr` environment for every command:

```powershell
conda run -n clrr python scripts/revalidation/prepare_method_candidates.py
conda run -n clrr python tests/smoke_method_candidates.py
conda run -n clrr python scripts/revalidation/launch_method_candidates.py --account all
conda run -n clrr python scripts/revalidation/monitor_method_candidates.py --account 1
conda run -n clrr python scripts/revalidation/monitor_method_candidates.py --account 2
```

Prepare is blocked after training/gates begin. Do not relaunch a live queue.
Do not modify a launched bundle. Progress and queue states live in
`outputs_rebuttal/amis_method_candidates/`; monitors collect small JSON only.
Remote workspace `/content/CLRR_candidates`, log `/content/method_candidates.log`.
Monitor queue, current epoch/step, validation metric, memory, and backup receipts.

## Public checkpoint storage

Each completed selected model is uploaded to public
`FiveC/amis-rewire-checkpoints`:

`checkpoints/amis_mandarin/mbart-large-50/<method-id>/seed-42/es10/`

Store `best.zip`, method/run metadata, resolved arguments, validation predictions
and metrics, final/selected trainer states, stop state, and backup receipts.
The archive includes the run specification in `training_receipt.json`.
Delete remote training weights only after public upload checksum verification;
upload failures retain weights and stop the queue. No checkpoint downloads to
the local machine during monitoring. Existing results are not overwritten.

## Interpretation

Screen using validation only. Improvements against baseline and historical Full
need consistent protocol, numerical checks, and later seed confirmation.
Centered alignment becoming numerically nicer alone is not a successful method.
These variants overlap prior literature; a good score does not prove novelty.

## Completed seed42 screen

All four validation runs are complete. Each model reached epoch20. Selected
validation chrF++: U1C13.938141 (epoch14), U1F14.551875 (epoch19), U2L13.904488
(epoch20), U2F14.508005 (epoch20). BLEU/chrF++/chrF0 were replayed from public
validation predictions; source/target order matched all576 validation rows.
Public LFS archive hashes matched receipts. Colab1 and Colab2 were stopped with
active assignments0 and usage rate0.00/hr.

Against the reused baseline42 validation chrF++13.901541, U1F is +0.650334,
U2F +0.606465, U1C +0.036600, and U2L +0.002947. Full-minus-component deltas
are +0.613734 chrF++ for U1F−U1C and +0.603518 for U2F−U2L. These are one-seed
validation results, not statistical evidence or test-set results. U1F is the
screening leader by0.043869 chrF++ over U2F; confirmation across seeds and a
locked test evaluation would be needed before choosing it as the final method.
