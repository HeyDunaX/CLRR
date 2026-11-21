# W vs B: bounded validation replication

## Decision and attribution — 2026-11-13

Author requested continuing after W improved on seed42. This is a new follow-up
decision made after seeing pilot validation, not a passed G/R confirmation gate.
G/R failed and remain stopped. W is full R-Drop consistency at beta1.25 versus
B beta2.5; neither coefficient nor splitting full KL is a new ACL contribution.
Prior-art rationale remains in POST_PAIRED_ROUTE_METHOD_PLAN.md. No new objective,
architecture, hidden-state loss, or coefficient search is introduced here.

RQ: does the validation advantage of reducing consistency strength repeat across
training seeds with the same model, optimizer and numerical/data recipe?

| Run | Dataset | Backbone | Seed | Objective |
|---|---|---|---:|---|
| W_S43 | Amis to Mandarin | mBART-50 | 43 | mean two-view CE +1.25 symmetric KL |
| W_S44 | same | same | 44 | same |
| B_S43 | same | same | 43 | mean two-view CE +2.5 symmetric KL |
| B_S44 | same | same | 44 | same |

All initialize from the same pinned pretrained revision, never seed42 trained
weights. LR5e-5, FP32/SDPA/TF32off, microbatch4/accumulation32, ES10/max20,
validation BLEU beam1, original CSV references; no test or new language. Two
recipes reuse existing validated implementations and one detached suite executes
ALL four GPU gates before training any of these four runs.

## Baseline reuse and provenance caveat

Existing baseline43/44 completed20epochs/720steps. Full validation histories show
BLEU and chrF++ choose the same saved checkpoint: seed43 step684 BLEU16.7345007067,
chrF++14.2060546640; seed44 step720 BLEU16.2089408794, chrF++13.7450014654.
Data hashes, model revision, runtime, optimizer and batching match the pinned
baseline recipe. Original attention backend was implicit; SDPA is reconstructed
from the historical runtime/cache probe, not a contemporaneous backend receipt.
Keep this limitation in all comparisons, as for baseline42. No baseline score or
checkpoint is overwritten. Machine-readable audit is embedded in both recipes.

Seed42 reused: baseline17.6823291745/14.7030053622, B17.9107/14.8427,
W18.2724095712/15.1728074921 (BLEU/chrF++). Exact B scores are read from its
verified VAL_COMPLETE.json. Seed42 has already influenced the follow-up design;
seeds43/44 are development replication, not an independent held-out task.

## Interpretation locked before seeds43/44 results

Report seedwise W-minus-B and W-minus-baseline, mean, SD and seed-level CI;
with n3, do not claim population significance from sentence resampling alone.
Practical continuation gate: across42/43/44 mean W-minus-baseline >=0.30 BLEU,
mean W-minus-B >=0.20 BLEU, positive deltas on at least2/3 seeds for each
comparison, and mean chrF++ not below either comparator. These are development
thresholds, not proof of significance or novelty. If met, propose a separate
transfer/method-design decision. Do not automatically launch test, Ede, A100,
extra seeds or lambda sweep. If not met, report instability and stop this queue.

## Execution and storage

Recipes: amis_consistency_strength_w_replication_rtx.json and
amis_consistency_strength_b_replication_rtx.json under configs/experiments.
Canonical public storage:
checkpoints/amis_mandarin/mbart-large-50/{weak-full-consistency-v1,paired-native-rdrop-v1}/seed-{43,44}/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu.
Do not overwrite seed42. Verify HF backup hashes before cleanup. Use only
/workspace/ura/thanhdat/projects/CLRR; admit >=20000MiB free GPU and >=40GiB
disk, never kill other processes. The detached suite exits after its four runs.
