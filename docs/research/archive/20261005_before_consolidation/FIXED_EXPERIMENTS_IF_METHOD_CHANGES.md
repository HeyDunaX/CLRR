# Các thực nghiệm giữ lại khi thay CLRR và/hoặc LSR

> **Cập nhật 05/10/2026:** G0 phát hiện lỗi accumulation ở wrappers CLRR/LSR, xem [verification](G0_G1_VERIFICATION_20261005.md). Các số dưới đây vẫn giữ làm records của checkpoints cũ. Không coi affected wrapped controls là matched controls của code đã sửa; cần corrected cohort trước khi so phương pháp mới.

Ngày lập: 05/10/2026.

Nguồn: [paper hiện tại](../../../paper/clrr_main.tex), [experiments đã cập nhật](../../../experiments/EXPERIMENTS_UPDATED.md), [kế hoạch dài hạn](ACL_LONG_TERM_EXPERIMENT_PLAN.md).

## 1. Quy tắc giữ số

**Giữ nguyên kết quả đã xác minh của đúng run/checkpoint, dataset, split và protocol.** Thay phương pháp không làm mất các baseline và đối sánh đã chạy. Kết quả mới được bổ sung riêng, không ghi đè kết quả cũ.

“Cố định” nghĩa là kết quả của thí nghiệm đã chạy được lưu làm mốc; không nghĩa là mọi số đều dùng được trong mọi bảng so sánh tương lai. Nếu đổi recipe, initialization, selection, seeds hoặc dataset, cần đối chứng tương ứng cho cohort mới. Nếu tìm thấy lỗi thật trong đo đạc, phải ghi nhận bản sửa và provenance, không bảo vệ số sai vì đã chốt.

Các bảng dưới đây là **bố cục đề xuất**, chưa thay đổi manuscript. Paper hiện tại có 10 bảng: 6 trong phần chính và 4 trong appendix.

## 2. Main Table 1 — mBART Amis: baseline và đối sánh chính

FT là mean ± sample standard deviation qua seeds 42/43/44 của matched suite C. Các đối sánh A là một checkpoint; nên tách hai panel nếu đặt cùng bảng. Tất cả là test Amis→Mandarin, nhưng A và C không có cùng lịch sử training/generation.

| Method | BLEU | chrF++ | Evidence | Vai trò khi thay phương pháp |
|---|---:|---:|---|---|
| Full fine-tuning | 19.39 ± 1.12 | 16.92 ± 3.26 | C, 3 seeds | Giữ baseline chính cho cohort hiện tại |
| Middle-Layer Alignment | 19.2893 | 18.4378 | A, một checkpoint | Giữ đối sánh alignment |
| Strong LoRA A | 10.2296 | 8.6319 | A, seed 42 | Giữ đối sánh adaptation |
| Strong LoRA B | 13.8066 | 10.3293 | A, seed 42 | Giữ đối sánh adaptation |
| Phương pháp cuối cùng | Chưa có | Chưa có | Cohort tương ứng | Bổ sung sau khi khóa phương pháp |

Mean FT đầy đủ: BLEU 19.3886, chrF++ 16.9249, chrF0 18.0452. Không lấy đối sánh một seed làm matched three-seed evidence. Nếu cần claim superiority trước LoRA/alignment với equal tuning budget, bổ sung matched runs; số lịch sử vẫn giữ.

## 3. Main Table 2 — baseline nhiều backbone

Đây là evidence B: các selected checkpoints lịch sử được regenerate dưới common FP32 inference. Mỗi hàng là một checkpoint, không phải mean nhiều seeds. Số làm tròn theo bảng `backbones` trong paper hiện tại.

| Backbone | Method | BLEU | chrF++ | chrF0 | Khi thay phương pháp |
|---|---|---:|---:|---:|---|
| mBART | Fine-tuning | 19.61 | 14.05 | 18.23 | Giữ |
| NLLB | Fine-tuning | 13.62 | 10.50 | 13.77 | Giữ |
| mT5 | Fine-tuning | 2.79 | 3.81 | 5.06 | Giữ |
| ByT5 | Fine-tuning | 7.49 | 8.30 | 9.65 | Giữ |
| Từng backbone | Phương pháp cuối cùng | Chưa có | Chưa có | Chưa có | Bổ sung |

Baseline mBART 19.61 ở B khác mean 19.39 ở C vì khác cohort. Không thay thế qua lại. B chuẩn hóa inference, không xóa khác biệt optimization và checkpoint selection lịch sử. Nếu phương pháp mới đổi recipe, giữ bảng này dưới nhãn historical và tạo bảng xác nhận với baseline phù hợp.

## 4. Main Table 3 — ablation phương pháp cuối cùng

| Hàng dự kiến | Số hiện tại có thể dùng lại? | Quy tắc |
|---|---|---|
| FT baseline | Có nếu cùng cohort/recipe | Giữ mốc đã xác minh |
| Phương pháp mới đầy đủ | Không | Chạy mới |
| Bỏ thành phần A của phương pháp mới | Không | Chạy ablation mới |
| Bỏ thành phần B của phương pháp mới | Không | Chạy ablation mới |
| Original CLRR / Original LSR / Original Full | Có dưới tên phiên bản cũ | Đối sánh tiền thân, không gọi là ablation của kiến trúc mới |

Nếu chỉ đổi CLRR và giữ nguyên LSR, LSR-only cũ có thể tiếp tục là control khi objective, λ và toàn bộ recipe phù hợp. Nếu đổi LSR, cần control cho LSR mới; giữ LSR cũ làm comparator riêng.

## 5. Appendix Table A1 — toàn bộ đối sánh độc lập với CLRR/LSR

Tất cả các hàng sau thuộc saved-prediction audit A. Các số có thể giữ khi bỏ cả CLRR và LSR. Vị trí main/appendix phụ thuộc phạm vi claim và chất lượng matching; không chọn vị trí chỉ dựa vào score có thuận lợi hay không.

| Backbone | Method | BLEU | chrF++ | Vị trí đề xuất |
|---|---|---:|---:|---|
| mBART | BitFit | 0.3463 | 2.2552 | Appendix |
| mBART | Narrow LoRA | 3.2365 | 4.4213 | Appendix |
| mBART | Strong LoRA A | 10.2296 | 8.6319 | Main Table 1 và chi tiết appendix |
| mBART | Strong LoRA B | 13.8066 | 10.3293 | Main Table 1 và chi tiết appendix |
| mBART | Middle-Layer Alignment | 19.2893 | 18.4378 | Main Table 1 và chi tiết appendix |
| NLLB | BitFit | 1.0750 | 2.9643 | Appendix |
| NLLB | Narrow LoRA | 3.5383 | 4.9407 | Appendix |
| NLLB | Strong LoRA A | 11.4625 | 9.5004 | Appendix hoặc bảng đối sánh NLLB |
| NLLB | Strong LoRA B | 9.8024 | 8.4010 | Appendix hoặc bảng đối sánh NLLB |
| NLLB | Middle-Layer Alignment | 13.9648 | 10.6596 | Appendix hoặc bảng đối sánh NLLB |
| mT5 | Middle-Layer Alignment | 4.6625 | 5.2103 | Appendix |
| ByT5 | Middle-Layer Alignment | 7.5660 | 8.3414 | Appendix |

Strong LoRA A/B đổi cả LR và embedding training; không diễn giải A–B thành hiệu ứng của riêng embeddings. Middle-Layer Alignment là adaptation shared-encoder của ý tưởng alignment, không phải exact reproduction toàn bộ protocol bài nguồn. Narrow LoRA/BitFit lịch sử không đủ để kết luận PEFT nói chung thất bại.

## 6. Appendix Table A2 — mốc phương pháp nguyên bản

Đây là kết quả test beam 4, mean qua seeds 42/43/44. Các hàng được giữ như phương pháp tiền thân nếu thay cả CLRR/LSR; không mặc định nằm trong main table.

| Method nguyên bản, mBART Amis | BLEU | chrF++ | chrF0 | Vai trò sau khi đổi |
|---|---:|---:|---:|---|
| CLRR-only, α=.10 | 19.5769 | 18.5172 | 18.2035 | Original routing comparator |
| LSR-only, λ=.10 | 19.5343 | 18.5639 | 18.2040 | Original alignment comparator |
| Tuned LSR-only, λ=.03 | 19.8954 | 17.2923 | 18.5350 | Đối sánh LSR cũ đã tune |
| Full detached, α=.10/λ=.10 | 19.7687 | 18.6565 | 18.3715 | Mốc phương pháp ban đầu |
| Full source-attached | 19.5658 | 18.5633 | 18.2638 | Ablation lịch sử |
| Full α=.10/λ=.03 | 19.0854 | 18.2324 | 17.8648 | Ablation lịch sử |

Warmup v2, α/λ screening, decoder/both-stack variants và interventions cũ giữ trong records phát triển, tách đúng split/protocol. Warmup hiện chỉ có validation; không ghép score warmup vào bảng test này.

## 7. Ánh xạ 10 bảng hiện tại khi đổi phương pháp

| Label bảng trong paper | Hiện tại | Hàng/kết quả giữ lại | Phần cần thay hoặc bổ sung |
|---|---|---|---|
| `means` | Main | FT; các nhánh cũ lưu riêng | Mean/std phương pháp mới |
| `seed_deltas` | Main | Bảng cũ trong lịch sử | Deltas mới; không dùng delta cũ cho method mới |
| `backbones` | Main | Các hàng FT | Các hàng phương pháp mới cùng protocol phù hợp |
| `historical_tests` | Main | Kiểm định lịch sử nguyên statistical family | Kiểm định riêng cho so sánh mới |
| `routing` | Main | On/off của đúng weights cũ | Intervention thích hợp với phương pháp mới |
| `mechanisms` | Main | Baseline probes có provenance; method probes cũ lưu riêng | Probes của phương pháp mới |
| `individual_seeds` | Appendix | FT từng seed; runs cũ lưu riêng | Runs mới với IDs rõ ràng |
| `fresh_tests` | Appendix | Family 24 tests cũ | Family mới định trước |
| `historical_a` | Appendix | Toàn bộ đối sánh A1 | Tách routing/LSR cũ thành tiền thân |
| `trajectory` | Appendix | Trajectory của phương pháp cũ | Đo mới nếu cần claim training dynamics |

P-value, CI, rank, residual, gradient và trajectory của CLRR/LSR cũ không được gán cho phương pháp mới. Những probes bị invalidated không dùng làm evidence hiện hành.

## 8. Những nền tảng khóa và những phần cần chạy thêm

| Thành phần | Quyết định |
|---|---|
| Amis train/validation/test, membership/order, raw references và manifest | Giữ nền dữ liệu đã audit: 4,600/576/575 |
| Metrics | Giữ protocol đã chứng nhận; BLEU zh, native chrF++, chrF0 diagnostic; không chọn lại metric theo score |
| FT cohort C seeds 42/43/44 | Giữ đối chứng development hiện tại |
| FT historical B trên 4 backbones | Giữ baseline lịch sử, không giả làm multi-seed confirmation |
| Đối sánh A1 | Giữ results; bổ sung matched HPO/seeds nếu claim cần |
| Seeds xác nhận mới | Chạy đối chứng cùng seeds; không ghép mean cũ khác seeds |
| Turkish và dataset mới | Cần baseline/đối sánh của chính dataset đó; số Amis không chuyển sang |
| Phương pháp mới đổi objective/recipe | Bổ sung controls tương ứng, giữ lịch sử nguyên vẹn |
| Efficiency benchmark | Đo lại method mới và baseline dưới cùng điều kiện |
| Asháninka / LayerSkip | Không đưa lại vào phạm vi paper |

## 9. Kết luận bàn giao

**Các đối chứng độc lập với phương pháp được giữ:** FT trên mBART/NLLB/mT5/ByT5; Strong LoRA A/B trên mBART/NLLB; Narrow LoRA và BitFit trên mBART/NLLB; Middle-Layer Alignment trên cả bốn backbones.

**CLRR-only, LSR-only và Full cũ giữ làm mốc tiền thân.** Nếu bỏ hoặc thay chúng, cập nhật phương pháp và ablation trong manuscript, không xóa scientific records hoặc thay tên phương pháp trên số cũ.

Không cần tự động train lại toàn bộ đối sánh. Chỉ bổ sung khi cohort/protocol hoặc claim mới đòi hỏi matching tốt hơn. File này không phải lệnh launch training.
