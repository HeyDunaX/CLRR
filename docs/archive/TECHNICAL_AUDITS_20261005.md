# Technical audits — archived evidence

Resolution: accumulation normalization was fixed; baseline43/44 selected steps are720/612, and fresh baseline42 selected720. The old bundle passed its GPU/full-size gates. These receipts do NOT certify the new fresh-confirmation bundle: its GPU gate remains pending in the current handoff. Earlier HOLD/unknown/retrain recommendations below record intermediate audit states.

Current interpretation: [RESEARCH_INSIGHTS](../research/RESEARCH_INSIGHTS.md). Execution status: [current handoff](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md).

## Contents
- [G0_G1_LOCAL_AUDIT_20261005](#g0-g1-local-audit-20261005)
- [G0_G1_VERIFICATION_20261005](#g0-g1-verification-20261005)
- [PRE_RERUN_PROVENANCE_AUDIT_20261005](#pre-rerun-provenance-audit-20261005)
- [PRE_RERUN_VERIFICATION_20261005](#pre-rerun-verification-20261005)
- [GPU_PREFLIGHT_AND_BASELINE_GATE_20261005](#gpu-preflight-and-baseline-gate-20261005)
- [FULLSIZE_GATE_AND_BASELINE42_20261005](#fullsize-gate-and-baseline42-20261005)

---

<a id="g0-g1-local-audit-20261005"></a>

## G0_G1_LOCAL_AUDIT_20261005

Original source: `docs/archive/TECHNICAL_AUDITS_20261005.md#g0-g1-local-audit-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

# G0/G1 local audit — 2026-10-05

## Status and scope

**TECHNICAL_HOLD.** This audit reads existing artifacts and recomputes validation metrics; it does not train or change scientific records. The independent accumulation correctness check performed in this audit phase found wrapper/baseline scaling differences: see `outputs_rebuttal/amis_g0_g1_20261005/accumulation/receipt.json` and the root investigation. Equality of command-line hyperparameters is not certification of equivalent optimization. The observations below characterize existing checkpoints; they do not establish a fair method comparison until affected training cohorts are resolved.

All local Python ran in conda `clrr`. No token/launcher read, GPU action, historical file overwrite, or manuscript change was performed by this local audit.

## G0 — evidence and reuse

The corrected V5 inventory contains 15 controls in the five core families, seeds 42/43/44. Twelve FT/Full/matched-LSR/tuned-LSR validation prediction paths and SHA256 values pass the local hash audit. The three CLRR-only entries are explicitly historical and were not remeasured in V5. Their later FP32 beam1 replay is available in both warmup machine control receipts, copied into the new audit directory; do not interpret missing V5 paths as missing checkpoints.

| Family | Existing weights/predictions | Reuse limit |
|---|---|---|
| FT | Three HF archive identities; corrected V5 validation predictions | Baseline42 original best step is unknown; 216 was an unsupported historical label. Generation is verified, not original checkpoint selection chronology. |
| CLRR-only | Three archive identities; both warmup-machine native validation replay receipts | Preserve replay receipt as source for chrF0; original V5 inventory lacks chrF0. |
| Matched LSR .10 | Three identified archives and verified V5 predictions | Existing checkpoint description only while effective optimization mismatch is investigated. |
| Tuned LSR .03 | Three identified archives and verified V5 predictions; both-machine exact replay | Development-selected comparator; not globally optimal lambda and not independent confirmation. |
| Full .10/.10 detached | Three identified archives and verified V5 predictions; both-machine exact replay | Existing checkpoint description only while effective optimization mismatch is investigated. |

Archived 20261003 `modeling.py` AST confirms that both `CrossLayerResidualRewire.forward` and `JEPAGuidedSeq2SeqLM.forward` accept `**kwargs` and pop `num_items_in_batch`; archived evaluation code has no `accepts_loss_kwargs` occurrence. This corroborates the historical wrapper pattern. It alone does not prove the exact Trainer/runtime of every historical run. It must not be described as an Adam learning rate being multiplied by four.

Outputs: `control_inventory_audit.csv`, `control_runtime_replay_receipts.json`, `control_configuration_receipt_copy.json`, `archived_wrapper_ast_receipt.json`.

## D1 — warmup selection and stopping

Selection was native raw-reference validation chrF++, beam1, fixed evaluation alpha=.10. All six selected checkpoints equal the maximum logged chrF++ checkpoint and also the maximum logged BLEU checkpoint. All selected prediction CSVs match the canonical 576 validation source/target rows and COMPLETE hashes. Native SacreBLEU 2.6.0 recomputation reproduces the selected scores.

| Run | Best epoch | Endpoint | Best BLEU | Endpoint BLEU | Best−endpoint BLEU | Stopped early? |
|---|---:|---:|---:|---:|---:|---|
| Full seed42 | 20 | 20 | 17.7908 | 17.7908 | .0000 | No |
| Full seed43 | 20 | 20 | 16.2231 | 16.2231 | .0000 | No |
| Full seed44 | 18 | 20 | 17.0006 | 16.9565 | .0441 | No |
| CLRR-only seed42 | 18 | 20 | 16.6623 | 16.5244 | .1379 | No |
| CLRR-only seed43 | 8 | 12 | 14.4003 | 14.1922 | .2082 | Yes, consistent with patience4 |
| CLRR-only seed44 | 20 | 20 | 16.8898 | 16.8898 | .0000 | No |

Training alpha records pass `.10 * min(step_before_update / 144, 1)` for every recorded update, with evaluation alpha always .10. There are 720 planned optimizer steps. Five runs reach epoch20; CLRR43 reaches epoch12. This is not a finding that all runs stop prematurely.

Minimum validation CE occurs at epochs5–6 for all six runs, while generation-selected bests are mostly epochs18–20. Lower teacher-forced CE and better generation do not peak together. Do not change selection to CE after examining these results.

The selected-best warmup Full mean is 17.0048 BLEU, versus fixed Full 17.2135 and tuned LSR16.9942 on the same evaluation protocol. The warmup verdict cannot be rescued by correcting an endpoint/best mixup: Full42/43 already use endpoints; Full44's actual selected best is used. These differences remain conditional on the optimization correctness investigation.

**D2 is optional**, useful to characterize routing dependence of warmup best weights. It is not needed to re-establish selected-best selection correctness or the current warmup gate failure, and cannot resolve gradient accumulation fairness.

Outputs: `warmup_history_selection_audit.csv`, `audit_summary.json`.

## D4 — existing V5 error analysis

All prediction files referenced by V5 test/validation error analysis match their recorded SHA256 values. Bins are raw source whitespace word counts. They are descriptive, not morphology annotations; no new hypothesis is selected from test and then called independent confirmation.

| Comparison, mean across three seeds | Split/bin | Rows per seed | BLEU delta | chrF0 delta |
|---|---|---:|---:|---:|
| Full−FT | Test all | 575 | +.3800 | +.3263 |
| Full−matched LSR | Test all | 575 | +.2344 | +.1675 |
| Full−tuned LSR | Test all | 575 | −.1267 | −.1635 |
| Full−tuned LSR | Test <=5 | 178 | −.2760 | −.3157 |
| Full−tuned LSR | Test 6–10 | 368 | +.0232 | +.0091 |
| Full−tuned LSR | Test >10 | 29 | −1.0257 | −1.1509 |
| Full−tuned LSR | Validation >10 | 38 | +1.9546 | +1.3022 |

Long-bin test harm contrasts with validation benefit, and bins are small. There is no consistent basis here for claiming a long-sentence mechanism or choosing a length-specific upgrade. Test references contain only three word bigrams in three rows; chrF++ deltas are not evidence for morphology. The existing chrF0 sensitivity/influence audit remains necessary.

Outputs: `v5_error_prediction_hash_audit.csv`, `v5_error_bins_descriptive.csv`, `v5_error_comparisons_descriptive.csv`, `v5_error_comparison_means.csv`.

## Next decision

1. Resolve accumulation correctness and map affected cohorts before training upgrades or declaring superiority.
2. Keep old metrics as genuine measurements of old checkpoints, separately labeled by their recipe. Do not overwrite them or relabel them as corrected training.
3. D3 may describe the current target anchor and gradient behavior, but upgrade efficacy requires a fair corrected pilot.
4. No Turkish run and no new candidate training is authorized by this diagnostic report.

Reproduce only this local audit:

```powershell
conda run --no-capture-output -n clrr python -X utf8 outputs_rebuttal/amis_g0_g1_20261005/audit_local.py
```


---

<a id="g0-g1-verification-20261005"></a>

## G0_G1_VERIFICATION_20261005

Original source: `docs/archive/TECHNICAL_AUDITS_20261005.md#g0-g1-verification-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

# G0/G1 — kiểm chứng trước nâng cấp phương pháp

## Material Passport

- Ngày: 05/10/2026, Asia/Saigon.
- Workflow: academic-research-suite / experiment-agent, validate và diagnostic execution.
- Phạm vi: G0, D1, D3, D4; không train research models, không chạy Turkish.
- Verification Status: **TECHNICAL_HOLD cho so sánh training lịch sử; bản sửa accumulation qua synthetic regression.**
- Nguồn: [local audit](TECHNICAL_AUDITS_20261005.md#g0-g1-local-audit-20261005), [artifacts mới](../../outputs_rebuttal/amis_g0_g1_20261005), source hiện tại và archived AST, primary literature.

## 1. Lỗi thực đã phát hiện

Với Transformers 4.57.6, Trainer coi `forward(**kwargs)` là nhận loss kwargs trừ khi model khai báo khác. Hai wrappers `CrossLayerResidualRewire` và `JEPAGuidedSeq2SeqLM` bỏ `num_items_in_batch`, trả mean loss của microbatch. Trainer vì thế bỏ phép chia gradient accumulation khi token count được cung cấp. Baseline mBART không nhận kwargs này và được chia đúng.

Đã kiểm chứng bằng **một synthetic SGD update**, tiny mBART, FP32 CPU, dropout=0, bốn examples có số label tokens bằng nhau; so batch4/accum1 với batch1/accum4. Đây là correctness test, không train mô hình nghiên cứu.

| Family | Tỷ lệ norm update accum4/batch4 trước sửa | Sau sửa | Kết luận |
|---|---:|---:|---|
| Baseline | 1.0000 | 1.0000 | PASS |
| CLRR-only | 4.0000 | 1.0000 | Bug được tái hiện và sửa |
| LSR-only | 4.0000 | 1.0000 | Bug được tái hiện và sửa |
| Full | 4.0000 | 1.0000 | Bug được tái hiện và sửa |

Sau sửa, relative update differences dưới 3e-7. Receipts: [trước sửa](../../outputs_rebuttal/amis_g0_g1_20261005/accumulation/receipt.json), [sau sửa](../../outputs_rebuttal/amis_g0_g1_20261005/accumulation_fixed/receipt.json).

Sửa tối thiểu: khai báo `accepts_loss_kwargs = False` trên cả hai wrapper để Trainer chuẩn hóa mean loss theo số microbatches. Attached-source wrapper kế thừa CLRR cũng nhận khai báo này. Forward/inference, weights và metrics đã lưu không thay đổi.

### Giới hạn và impact

- **Không kết luận LR AdamW tăng 4 lần.** Scaling gradient có tác động khác với SGD vì Adam moments, epsilon, clipping và precision. Cần corrected-training cohort để biết quality thay đổi thế nào.
- Mean CE của từng microbatch được average theo accumulation. Với số label tokens khác nhau, không mặc định tương đương global token-weighted loss của một batch lớn; regression bằng equal-token examples cô lập lỗi factor4.
- Archived AST xác nhận historical wrappers có cùng pattern bỏ token count. Runtime/Trainer của từng cohort vẫn phải được truy vết; không tự gán bug cho mọi backbone và mọi PEFT run.
- Amis C/confirmation/detach/warmup dùng wrappers và accumulation4 dưới pinned runtime là các cohort cần đánh dấu affected/pending repair. FT baseline không mắc cơ chế lỗi này trong regression; khả năng reuse còn phụ thuộc initialization, shuffle, selection và recipe phù hợp.
- CLRR/LSR cùng bị scale trong test. So giữa hai wrapped methods vẫn bị ảnh hưởng bởi optimizer/clipping; không kết luận mọi relative ranking sai hoặc mọi ranking giữ nguyên.
- BLEU/chrF của outputs cũ vẫn là phép đo hợp lệ của đúng checkpoint cũ. Không đổi số hoặc gọi chúng là kết quả train đã sửa.
- Không tiếp tục G2 bằng code đã sửa rồi so với wrapped controls cũ như một matched study. Phải tạo cohort được chuẩn hóa cho affected branches trước.

## 2. D1/D4 — những gì đã xác nhận

- Sáu warmup chọn đúng max validation chrF++; đồng thời cũng là max BLEU trong history.
- Full best epochs 20/20/18; CLRR-only 18/8/20. Chỉ CLRR43 early-stop epoch12, best epoch8. Không có bằng chứng mọi runs bị stopping quá sớm.
- Routing alpha schedule đúng; evaluation alpha=.10. Selected-best records/hashes và native metrics khớp.
- Warmup không tốt hơn không phải do nhầm endpoint với best; interpretation quality vẫn bị giới hạn bởi lỗi optimization mới phát hiện.
- V5 length bins đảo dấu giữa validation/test, không đủ chọn một nâng cấp chuyên trị câu dài.
- D2 warmup routing-off không cần để xác nhận selection; chưa chạy vì không giải quyết technical hold hiện tại.

## 3. D3 — target-anchor counterfactual

Đã hoàn thành trên một A100 Standard, không high-mem. Ba fixed Full selected checkpoints tải bằng pinned HF revision, xác minh size/SHA256 và strict weight loading. Dùng cùng tám batches × bốn validation examples như V4.

Source giữ routing α=.10; target pass lần lượt routing .10 và raw .0, cùng source graph/CE, target detached và caches độc lập. Đo anchor cosine/L2, LSR loss và CE–LSR gradients. FP32/eval/dropout-off/TF32-off, không optimizer updates. Đây là counterfactual characterization của weights cũ, không chứng minh training gain của U1.

| Seed | Cosine anchor routed/raw | LSR routed | LSR raw | Cosine hai gradients LSR | CE–LSR cosine routed / raw |
|---|---:|---:|---:|---:|---|
| 42 | .97307 | .19972 | .20425 | .95928 | .04718 / .03044 |
| 43 | .97249 | .20201 | .20744 | .95617 | .05897 / .04180 |
| 44 | .97283 | .20542 | .21106 | .95970 | .05845 / .04239 |

Số là mean qua tám fixed batches cho từng checkpoint. Tắt target routing đổi anchor và gradient, nhưng routed anchor có LSR loss thấp hơn và CE–LSR cosine cao hơn trong cả ba seeds. Không có evidence từ probe này rằng raw anchor giảm global gradient conflict. Điều đó không chứng minh raw anchor kém về quality, nhưng cũng không đủ khóa U1 như một giải pháp đã có cơ sở mạnh. LSR loss thấp hơn không tự chứng minh translation tốt hơn.

Collection SHA256, cả ba archive size/SHA256 và cache/target-detach checks PASS. [Summary](../../outputs_rebuttal/amis_g0_g1_20261005/verification_summary.json). Regression bản sửa trên GPU pinned cũng PASS cả bốn families, relative update differences dưới 3e-7: [receipt](../../outputs_rebuttal/amis_g0_g1_20261005/accumulation_gpu_fixed/receipt.json). Không có research optimizer steps trong D3; synthetic regression thực hiện một tiny update mỗi điều kiện.

A100 đã tắt sau khi thu/audit kết quả: Colab CLI xác nhận `Session terminated`. Lần stop đầu gặp DNS lỗi, lần thử lại thành công. Chỉ dùng một GPU vì D3 ba checkpoints nhỏ chạy tuần tự đủ; không cấp runtime thứ hai.

## 4. G0 novelty — đối chiếu công thức gần

| Primary source | Công thức/cấu trúc đã đọc | Khác với CLRR hiện tại | Việc cần chứng minh |
|---|---|---|---|
| [SkipCLM, §3](https://aclanthology.org/2025.naacl-srw.50.pdf) | Skip learned linear+ReLU, scaling warmup; InfoNCE trên decoder XGLM | CLRR identity fixed/detached encoder, nhiều block, LSR shared-encoder cosine | Skip+alignment không đủ làm novelty; cần ích lợi của rule cụ thể |
| [Value Residual Learning, §3.2](https://aclanthology.org/2025.acl-long.1375.pdf) | ResFormer trộn first-layer value với current value trước attention; có constants và learned scalars | CLRR cộng hidden block output sau whole block, source cách d và detached | Không claim constant/zero-added routing tự nó chưa từng có; kiểm tra location/source/gradient choices |
| [ResiDual, §2.2](https://arxiv.org/html/2304.14802v1) | Hai residual streams, cuối stack cộng Post-LN output và normalized dual sum | CLRR không đổi backbone LN thành dual streams, cộng activation xa tại mỗi block | Không mượn collapse guarantees của ResiDual cho CLRR |

Đây là bounded formula audit, không phải chứng nhận novelty toàn bộ literature. Exact reproduction comparator cần official code/version và matching recipe trước khi triển khai. U1/U2 còn là giả thuyết; chưa khóa candidate khi technical hold chưa xử lý.

## 5. Statistical fallacy scan — 11/11 đã xét

| Loại | Kết luận trong phạm vi audit |
|---|---|
| Simpson's paradox | Các length bins và splits khác dấu; không gộp thành một cơ chế phổ quát |
| Ecological fallacy | Mean checkpoints không chứng minh lợi ích cho từng câu hoặc morphology |
| Berkson's paradox | Selected-best và tuned candidates là mẫu được chọn; không đại diện mọi cấu hình |
| Collider bias | Không regression điều kiện hóa quality vào rank/loss đã chọn; không làm causal inference từ selected checkpoints |
| Base-rate neglect | Không có classification/sensitivity/PPV claim; không áp dụng |
| Regression to mean | Không gọi một seed thắng hoặc tuned winner là cải thiện ổn định; cần cohort xác nhận |
| Survivorship bias | Sáu warmup đều được report, gồm early-stop seed43; metadata/weights thiếu ghi unknown |
| Look-elsewhere effect | Không tìm subgroup/metric thắng mới; giữ các statistical families lịch sử |
| Forking paths | Lỗi được ghi riêng, không đổi metric/selection hậu nghiệm; recipe đã sửa là cohort mới |
| Correlation vs causation | Counterfactual anchor là thay đổi tính toán, không chứng minh training translation gain |
| Reverse causality | Không suy geometry tạo quality hoặc quality tạo geometry từ các endpoint quan sát |

Không thực hiện kiểm định significance mới cho diagnostics; tám batches trong một checkpoint không phải tám independent training seeds.

## 6. Quyết định tiếp theo

1. D3 đã hoàn thành và lưu số riêng; không cần lặp warmup on/off để sửa selection.
2. Correction đã qua runtime GPU pinned; archived wrapper pattern được chứng nhận. Tiếp tục impact matrix theo từng historical cohort, không tự invalidate mọi PEFT run.
3. Chốt corrected control cohort trước khi train nâng cấp. Không dùng số tuned LSR cũ như mốc tối ưu của recipe đã sửa.
4. Sau đó mới đánh giá có cần U1/U2. Chưa có bằng chứng để kết luận phải bỏ CLRR hoặc LSR vì quality hiện tại.

Không train lại research models trong lượt kiểm tra này. Không đổi manuscript hoặc ghi đè historical experiment numbers.

### Cohort sửa lỗi đề xuất, chưa launch

Để có mốc sạch trước khi sửa phương pháp, có thể chạy lại năm cấu hình gốc FT / CLRR-only .10 / LSR-only .10 / LSR-only .03 / Full .10/.10, seeds42/43/44: **15 runs**, cùng code/runtime/selection đã khóa. Việc chạy lại FT giúp có full receipt của cohort mới; chỉ giảm số runs khi G0 chứng nhận reuse đủ. Đây là đề xuất scientific repair, không phải training đã được thực hiện trong lượt diagnostic này.

λ=.03 tiếp tục là một configuration đối sánh đã định trước, không mặc định vẫn là tuned optimum sau sửa lỗi. Re-tuning cần validation-only và budget công bố. Những thresholds cũ dựa trên wrapped controls chưa sửa phải được gắn lại với corrected controls trước pilot nâng cấp. Nếu corrected four-way study đã đủ tốt, có thể xác nhận nó trước khi cần đổi kiến trúc.


---

<a id="pre-rerun-provenance-audit-20261005"></a>

## PRE_RERUN_PROVENANCE_AUDIT_20261005

Original source: `docs/archive/TECHNICAL_AUDITS_20261005.md#pre-rerun-provenance-audit-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

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


---

<a id="pre-rerun-verification-20261005"></a>

## PRE_RERUN_VERIFICATION_20261005

Original source: `docs/archive/TECHNICAL_AUDITS_20261005.md#pre-rerun-verification-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

# Kiểm tra trước khi train lại mBART Amis — 05/10/2026

> **Cập nhật sau báo cáo:** tiny GPU gate đã PASS; provenance seeds43/44 đã xác minh best-step720/612. Xem [báo cáo GPU và baseline](TECHNICAL_AUDITS_20261005.md#gpu-preflight-and-baseline-gate-20261005). Các mục pending bên dưới ghi trạng thái trước lần kiểm tra bổ sung. Baseline42 selected-step vẫn chưa xác minh.

## Material Passport

- Scope: đường train mBART Amis FT / CLRR-only / LSR-only / Full sau sửa accumulation; không mở GPU hoặc train model nghiên cứu trong lượt này.
- Execution: mọi Python/test local trong conda `clrr`, tiny offline fixtures.
- Verification Status: **LOCAL_IMPLEMENTATION_PASS; historical baseline reuse còn có điều kiện; GPU deployment gate chưa chạy cho toàn bộ source mới.**
- Evidence: [artifacts](../../outputs_rebuttal/amis_pre_rerun_20261005), [audit provenance trước sửa](TECHNICAL_AUDITS_20261005.md#pre-rerun-provenance-audit-20261005), [G0/G1](TECHNICAL_AUDITS_20261005.md#g0-g1-verification-20261005).

## 1. Các kiểm tra đã thực hiện

| Kiểm tra | Phạm vi | Kết quả |
|---|---|---|
| Accumulation so với explicit microbatch-mean gradients | 4 families × GC on/off × SGD/AdamW-clipped =16 cases;5 examples, variable label-token counts, accumulation4, last group1 | PASS, max absolute gradient difference0 trong fixture deterministic |
| Checkpointed vs non-checkpointed gradients | Detached/attached routing + LSR, dropout.1 với RNG cố định | PASS, max differences cỡ3e-8; added-source gradient thực sự khác detached |
| Real CLI lifecycle | 4 families,2epochs, GC on, accumulation4, uneven final group, validation-only, scientific records | PASS; best_model khớp bitwise với live model sau best reload; checkpoint có training receipt; không tạo test predictions |
| Precision pipeline smoke | 4 families qua real CLI, FP32 generation và strict reload | PASS trên CPU |
| Warmup/snapshot/restore | CLRR và Full; alpha schedule, alpha0 identity, strict save/reload, actual weight mutation rồi best reload | PASS |
| Original-reference metric tests | 4 tests | PASS |
| Resume/backup safety tests | 10 tests: reject old code/changed recipe/changed data/missing state, reject overwrite, compatible receipt, validation-only không đọc test, HF failure/corruption/success | PASS, HF dùng mocks, không gọi mạng |

Các kiểm tra gradient variable-length so với **average microbatch means**, là semantics hiện tại của cả baseline và wrappers. Không claim tương đương một globally token-weighted giant batch khi mỗi microbatch có số tokens khác nhau.

Target pass của LSR dùng no_grad. Warning reentrant checkpoint `None of inputs requires_grad` xuất hiện ở target branch này trong tiny checks; source/parameter gradients đã được kiểm tra, không diễn giải warning đó thành mất toàn bộ training gradients.

Backward checkpoint recomputation có thể repopulate detached caches sau backward. Public forward/generate bắt đầu cache mới và xóa khi kết thúc. Next-forward isolation và gradient equivalence PASS; không nhầm yêu cầu cache rỗng ngay sau backward thành một bug optimization mới.

## 2. Các sửa bổ sung để tránh chạy sai hoặc mất checkpoint

### Resume và overwrite

- CLI mặc định không auto-resume.
- Trước load pretrained/Trainer, chặn output folder không rỗng nếu không explicit-compatible resume.
- Ghi `training_receipt.json`: args recipe, train/validation hashes (test chỉ nếu không validation-only), source hashes và Python/torch/Transformers/Accelerate versions.
- Resume yêu cầu receipt khớp và đủ model weights/config, optimizer, scheduler, Trainer state, RNG. Checkpoint cũ thiếu receipt bị từ chối; không nối optimization cũ với code đã sửa.
- Receipt được copy vào Trainer checkpoints và best_model để archive có provenance.

Compatibility tests xác minh gate bằng fixtures; không claim đã tái lập bitwise một interrupted full GPU training. Corrected cohort sẽ khởi đầu từ pretrained, không resume historical weights.

### Hugging Face backup

- Final best ZIP upload thất bại sau retries giờ raise, giữ local archive; không báo success mặc định.
- Thành công chỉ sau khi remote size và LFS SHA256 khớp local ZIP.
- Worker vẫn phải giữ local weights đến khi whole-run completion artifacts được xác minh. Một successful model upload không thay verification của metrics/predictions.
- Per-epoch generic backup callback vẫn không upload full optimizer/RNG. Scientific trajectory ZIPs không resumable. Nếu cần bảo vệ interrupted progress khi launch, phải chốt chính sách full-state backup riêng; không hứa remote resumability từ best ZIP.

### Metadata/setup

- Scientific selection receipt ghi actual evalbeam từ TrainingArguments, bỏ hardcoded beam1 label.
- Bỏ duplicate save_safetensors constructor và broad TypeError fallback. Giá trị save_safetensors=False giữ nguyên dưới runtime pinned; đây là sửa setup/diagnostics, không đổi optimizer objective.

## 3. Baseline cũ: chưa chốt reuse vô điều kiện

| Seed | Metadata hiện có | Giới hạn |
|---|---|---|
| 42 | Recovered selected weights; original selected step unknown | Không suy step216 hoặc selectedepoch giả |
| 43 | metrics ghi checkpoint720; V5 inventory ghi beststep252 | Hai nguồn không thống nhất selected-step chronology |
| 44 | metrics ghi checkpoint612; V5 inventory ghi beststep504 | Hai nguồn không thống nhất selected-step chronology |

FP32 predictions và scores đã replay đúng không tự chứng nhận chronological best selection. Các baseline không mắc cơ chế accumulation wrapper bug trong tiny actual Trainer tests. Tuy nhiên chưa đủ evidence để cam kết chắc chắn giảm cohort15→12 runs. Không kết luận weights hoặc BLEU của baseline sai chỉ vì metadata conflicts.

## 4. Gate trước dùng Colab units cho training dài

1. Khóa riêng namespace corrected cohort, HF prefix và code manifest, không dùng các worker/names warmup cũ.
2. Khóa recipe: mBART/model-tokenizer revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`, Amis split4600/576/575, tl_XX→zh_CN, LR5e-5, WD0, LRwarmup.06, batch32×accum4, evalbatch8, lengths256,20epochs,patience4, fixedα.10 khi routing bật, λ.10/.03 theo control, không routing warmup mới.
3. Train BF16/TF32-on/GC; validation generation FP32/TF32-off/beam1, native raw-reference chrF++ chọn best. Không dùng test để chọn configuration.
4. Khi mở A100 cho corrected cohort, kiểm tra actual versions/source/data hashes/HF write namespace và chạy **tiny GPU preflight** trên đúng bundle mới (accumulation+GC+save/reload). Không mở20epochs trước gate này.
5. Chốt baseline reuse hoặc thêm3FT runs sau provenance review. Nếu không recover chronology, report unknown hoặc fresh cohort; không tự điền selected step.
6. Chỉ launch corrected components sau gates; chưa train U1/U2 hoặc Turkish để né technical issue.

**Hiện không có lỗi gradient/selection/save-reload còn tái hiện trong các fixtures đã kiểm tra.** Đây không phải bảo đảm toàn bộ run lớn không thể lỗi mạng/OOM/runtime. GPU FP32 accumulation fix đã PASS ở lượt G0 trước; BF16+accumulation4+GC trên toàn bộ bundle mới vẫn là deployment gate, chưa được thực hiện trong lượt local này.

## 5. Lệnh tái lập kiểm tra local

```powershell
conda run --no-capture-output -n clrr python -X utf8 tests/smoke_rerun_optimization.py
conda run --no-capture-output -n clrr python -X utf8 tests/smoke_attached_routing.py
conda run --no-capture-output -n clrr python -X utf8 tests/smoke_precision_training.py --audit-rerun --result-dir outputs_rebuttal/amis_pre_rerun_20261005
conda run --no-capture-output -n clrr python -X utf8 tests/smoke_warmup_records.py --result-dir outputs_rebuttal/amis_pre_rerun_20261005
conda run --no-capture-output -n clrr python -X utf8 -m unittest discover -s tests -p test_rerun_safety.py
conda run --no-capture-output -n clrr python -X utf8 -m unittest discover -s tests -p test_original_reference_metrics.py
```

Các tests chỉ tạo tiny temporary weights hoặc receipts trong directory audit riêng; không sửa scientific scores cũ. File này không phải lệnh launch research training.


---

<a id="gpu-preflight-and-baseline-gate-20261005"></a>

## GPU_PREFLIGHT_AND_BASELINE_GATE_20261005

Original source: `docs/archive/TECHNICAL_AUDITS_20261005.md#gpu-preflight-and-baseline-gate-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

# Baseline provenance và GPU preflight — 05/10/2026

## Kết luận

**GPU_PREFLIGHT_PASS** cho bundle source hiện tại. A100 Standard đã tắt sau khi tải artifacts và xác minh hash. Chưa mở training dài.

Baseline43/44 đã giải quyết được mâu thuẫn selected-step: **720/612**, không phải252/504. Không có bằng chứng từ kiểm tra này rằng BLEU/chrF của baseline sai.

## 1. Baseline cũ

| Seed | Selected-step đã xác minh | Quyết định |
|---|---|---|
| 42 | Chưa tìm được original selected-step sau recovery | Giữ baseline như recovered selected weights; chưa chứng nhận chronology. Không suy diễn step216. |
| 43 | 720, epoch20 | Không cần train lại chỉ vì mâu thuẫn metadata252. Có thể dùng lại nếu giữ đúng data/runtime/recipe/protocol đã audit. |
| 44 | 612, epoch17 | Không cần train lại chỉ vì mâu thuẫn metadata504. Có thể dùng lại nếu giữ đúng data/runtime/recipe/protocol đã audit. |

Log gốc seeds43/44 khớp SHA trong snapshot. Final log, metrics.json và all_metrics.json đồng ý selected checkpoint; validation chronology cũng đạt maximum chrF++ tại các bước này. Giá trị252/504 đến từ hardcode trong `scripts/revalidation/audit_amis_followup_f0.py`, rồi được giữ ở V5 inventory. Đây là lỗi provenance metadata, không phải kết quả đo lại checkpoint.

Đã tạo riêng `outputs_rebuttal/amis_pre_rerun_20261005/validation_inventory_provenance_corrected.json`: chỉ sửa best-step43/44, giữ scores và lưu giá trị bị thay. Evidence: `baseline_step_trace.json`. File lịch sử không bị ghi đè.

**Chưa chốt reuse vô điều kiện cả ba seeds**: seed42 còn thiếu selected-step provenance. Muốn cohort có chronology đầy đủ thì phải tìm thêm primary selection record của42 hoặc chạy baseline42 theo protocol mới; không tự động train lại cả ba baseline chỉ vì lỗi inventory43/44.

## 2. GPU gate thực chạy

| Hạng mục | Kết quả |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, Standard; không yêu cầu high-mem |
| Runtime của tests | Python3.13.15, torch2.6.0+cu124, Transformers4.57.6, Accelerate1.15.0 |
| Bundle | 22 file source/tests; SHA trước upload, sau extract, sau tests và local hiện tại khớp |
| Gradient reference | 16 cases: baseline/CLRR/LSR/Full × checkpointing on/off × SGD/AdamW-clipped; BF16, accumulation4, unequal label counts, partial final group |
| Gradient comparison | PASS; max absolute difference0 trong fixture deterministic |
| Actual CLI lifecycle | Bốn nhánh, hai tiny epochs, BF16 + accumulation4 + checkpointing, validation-only; PASS |
| Best selection/save/reload | Best weights bitwise khớp live model sau reload; checkpoint có receipt; không sinh test predictions |
| Routing schedule và restore | CLRR/Full: schedule, alpha0 identity, snapshot, strict reload, best reload đều PASS |
| Kết thúc | Artifacts tải về và xác minh; CLI xác nhận Session terminated |

Reference là average microbatch means; không suy ra globally token-weighted giant batch khi số tokens khác nhau. Fixture dùng mBART nhỏ offline, không phải benchmark Amis đầy đủ hay kiểm tra VRAM ở batch32/length256. PASS cho phép đi tiếp gate triển khai; không chứng minh chất lượng phương pháp hoặc đảm bảo full-size run không OOM.

Warning checkpoint target branch không có requires_grad vẫn xuất hiện ở LSR no_grad target pass; gradients thực đã qua comparison. Không coi warning này là thất bại gradient toàn model.

Artifacts: `outputs_rebuttal/amis_pre_rerun_gpu_20261005/`, gồm `GPU_PASS.json`, `local_verification.json`, manifests, environment freeze, optimization/lifecycle/warmup receipts và full log.

## 3. Trước training dài

1. Dùng đúng source đã khóa ở bundle manifest; source thay đổi thì gate tương ứng phải được kiểm tra lại.
2. Giữ protocol dataset/model revision/optimizer/selection đã thống nhất; namespace corrected cohort riêng, không resume optimization cũ.
3. Chốt cách xử lý provenance baseline42. Baseline43/44 có thể reuse theo recipe cũ đã audit; scores không bị thay bởi sửa metadata.
4. Chốt backup full-state nếu muốn resume sau runtime mất: best ZIP không thay optimizer/scheduler/RNG backup.
5. Khi được mở cohort, chạy smoke kích thước thực trước rồi mới chạy nhiều seeds. Không dùng tiny PASS để cam kết đủ VRAM hoặc đủ sức ACL.


---

<a id="fullsize-gate-and-baseline42-20261005"></a>

## FULLSIZE_GATE_AND_BASELINE42_20261005

Original source: `docs/archive/TECHNICAL_AUDITS_20261005.md#fullsize-gate-and-baseline42-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

# Bước1–2: baseline42 và smoke kích thước thật — 05/10/2026

## Trạng thái

- **Bước2: FULLSIZE_GPU_PASS**, artifacts đã tải về và SHA khớp source local.
- **Bước1: VERIFIED**. Original selected-step42 không phục hồi được; fresh FT seed42 đã chạy đủ20 epochs, best step720/epoch20, test scores khớp hoàn toàn số cũ. Namespace riêng, scores cũ giữ nguyên. Chưa chạy12 runs phương pháp.
- **A100 đã tắt**, CLI xác nhận `Session terminated`.

## Kết quả baseline42 mới

| Đại lượng | Run mới | So với số cũ |
|---|---:|---|
| Completed epochs | 20 | Không early stopping |
| Selected step/epoch | 720 /20 | Đây là provenance của fresh run; không gán ngược cho archive cũ |
| Best validation chrF++ (beam1) | 13.901540564847089 | Khớp preserved weights đã replay |
| Test BLEU (beam4, FP32) | 19.728081433402032 | Khớp hoàn toàn |
| Test chrF++ (beam4, FP32) | 18.713838023321145 | Khớp hoàn toàn |

Bốn primary surviving checkpoints216/252/468/504: BLEU, chrF++ và loss đều khớp old/new, abs difference0. Test predictions đủ575 rows, source/target khớp raw CSV; tính lại metrics từ predictions bằng original references PASS. Source và cả ba data split hashes khớp receipt.

Best ZIP2,270,665,528 bytes, SHA256 `e2d5c884c8c2da270aa1bbeaa4e36990021f1559b5fca2926d05cca3f5b0e53c`. HF archive LFS SHA và size khớp; final Trainer state, selected Trainer state, metrics và predictions cũng khớp remote Git blob/LFS checksums. Không tải model weights xuống máy local.

Colab collection download trả kết quả chậm; đã khởi động thêm recovery small metadata từ HF. Gói Colab sau đó về đầy đủ, giữ được full console log. Verification cuối dùng `colab_zip` và kiểm tra archive/source/data/metrics, không coi việc gọi lệnh download là bằng chứng file đã về.

Artifacts baseline: `outputs_rebuttal/amis_corrected_baseline42_20261005/local_verification.json`, `collection_receipt.json`, `final_trainer_state.json`, full training log, metrics và predictions. Evidence immutable ZIP checksum ở local verification.

## Quyết định reuse cho corrected cohort

| Seed | Baseline dùng tiếp | Selection provenance |
|---|---|---|
| 42 | Fresh run `mbart-amis-baseline-seed42-corrected` | 720/epoch20, full fresh chronology, HF archive và metadata verified |
| 43 | Reuse preserved selected checkpoint theo protocol đã audit | 720/epoch20, hash-verified original log/metrics/history; inventory252 được sửa riêng |
| 44 | Reuse preserved selected checkpoint theo protocol đã audit | 612/epoch17, hash-verified original log/metrics/history; inventory504 được sửa riêng |

**GO cho baseline controls và triển khai cohort theo recipe đã khóa**, không cần train lại43/44 chỉ vì lỗi inventory. Fresh42 tái lập đúng các mốc old baseline; source accumulation fix áp dụng wrappers, baseline training đã qua actual Trainer checks. Điều kiện reuse: giữ data/model revision, optimizer, selection và inference protocol đã audit. Đổi recipe thì quyết định này phải được xem lại.

Manifest cho bước sau: **`outputs_rebuttal/amis_fullsize_gate_20261005/baseline_controls.json`** khóa selected-step và HF archive identity của cả ba seeds. Historical V5 inventory42 vẫn giữ best_step unknown; không dùng historical42 thay fresh42 trong corrected cohort.

**Chưa GO chất lượng phương pháp**: kết quả này hoàn thiện control/provenance, không chứng minh CLRR hoặc Full mạnh hơn LSR. Bước sau là12 corrected runs (CLRR-only, matched LSR.10, tuned LSR.03, Full ×seeds42/43/44), chưa launch trong lượt này.

## Bước1 — truy provenance

Đã kiểm tra local log, surviving partial Trainer states, recovery receipts, HF file metadata và ZIP central directory bằng HTTP Range64KB. Original log chỉ giữ eval đến504; partial state504 không chứng minh final selected-step. Selected ZIP không có trainer_state/optimizer/scheduler/RNG. Không gán216 hoặc504 cho final selected archive.

Evidence: `outputs_rebuttal/amis_fullsize_gate_20261005/BASELINE42_SELECTION_AUDIT.md` và `baseline42_selection_trace.json`.

Run mới đã hoàn tất: `mbart-amis-baseline-seed42-corrected`, HF repo `FiveC/amis-rewire-checkpoints`, prefix **`amis_corrected_baseline42_20261005`**. Khởi đầu từ pretrained pinned revision, không resume hoặc sửa archive cũ.

Recipe: mBART-50 `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; Amis4600/576/575; tl_XX→zh_CN; seed42; LR5e-5, WD0, warmup.06; batch32×accum4; evalbatch8; lengths256; BF16 + gradient checkpointing + fused AdamW; max20 epochs, patience4; validationbeam1, testbeam4, generationFP32; selection validationchrF++.

Sau mỗi checkpoint, callback upload Trainer state và training receipt riêng lên HF; final selected provenance được giữ riêng. Final best ZIP phải qua remote size và LFS SHA256 gate. Full optimizer/RNG chưa được backup remote; selection metadata không thay resumable state. Không claim có thể resume sau mất runtime từ selected best ZIP.

## Bước2 — kiểm tra thực thi

Sử dụng mBART-50 đầy đủ610,879,488 parameters, actual Amis train128 rows và validation8 rows. Mỗi family chạy2 optimizer steps (8 microbatches): một step với dynamic padding dữ liệu thật; một step stress với source/target padded256. Sequential sampling bảo đảm thứ tự hai điều kiện; đây là smoke, không phải training benchmark.

| Family | Actual batches peak allocated GiB | Stress256 peak allocated GiB | Stress256 peak reserved GiB | Kết quả |
|---|---:|---:|---:|---|
| Baseline | 7.30 | 33.45 | 36.38 | PASS |
| CLRR-only | 7.31 | 33.45 | 36.40 | PASS |
| LSR-only | 7.31 | 33.48 | 36.40 | PASS |
| Full | 7.33 | 33.48 | 36.40 | PASS |

Tất cả dùng batch32, accumulation4, BF16, checkpointing, fused AdamW, FP32 parameters và pinned pretrained revision. Loss/parameters finite; Trainer xác nhận model_accepts_loss_kwargs=False. Validation generation FP32, beam1, batch8 chạy được. Các tiny save/reload/gradient equivalence checks trước đó vẫn là bằng chứng riêng.

GPU: A100-SXM4-40GB Standard; không yêu cầu high-mem. Runtime: torch2.6.0+cu124, Transformers4.57.6, Accelerate1.15.0. Bundle11 source/script/data files: pre-upload, remote và local hashes khớp.

Artifacts: `outputs_rebuttal/amis_fullsize_gate_20261005/`: `FULLSIZE_PASS.json`, từng family.json, `local_verification.json`, source/data manifests, environment và log.

PASS xác nhận footprint đã thử. Stress là padding256, không phải128 examples có256 tokens thật; mask/value có thể làm thay đổi hành vi numeric. Không đảm bảo mọi batch hoặc inferencebeam4worstcase đều có footprint giống smoke. Quality/superiority cần cohort nhiều seeds; smoke không tạo score paper.

