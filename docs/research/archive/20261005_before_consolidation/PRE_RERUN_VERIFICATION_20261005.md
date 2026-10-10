# Kiểm tra trước khi train lại mBART Amis — 05/10/2026

> **Cập nhật sau báo cáo:** tiny GPU gate đã PASS; provenance seeds43/44 đã xác minh best-step720/612. Xem [báo cáo GPU và baseline](GPU_PREFLIGHT_AND_BASELINE_GATE_20261005.md). Các mục pending bên dưới ghi trạng thái trước lần kiểm tra bổ sung. Baseline42 selected-step vẫn chưa xác minh.

## Material Passport

- Scope: đường train mBART Amis FT / CLRR-only / LSR-only / Full sau sửa accumulation; không mở GPU hoặc train model nghiên cứu trong lượt này.
- Execution: mọi Python/test local trong conda `clrr`, tiny offline fixtures.
- Verification Status: **LOCAL_IMPLEMENTATION_PASS; historical baseline reuse còn có điều kiện; GPU deployment gate chưa chạy cho toàn bộ source mới.**
- Evidence: [artifacts](../../../../outputs_rebuttal/amis_pre_rerun_20261005/), [audit provenance trước sửa](PRE_RERUN_PROVENANCE_AUDIT_20261005.md), [G0/G1](G0_G1_VERIFICATION_20261005.md).

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
