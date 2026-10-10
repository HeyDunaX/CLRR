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
