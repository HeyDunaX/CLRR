# Thực nghiệm tiếp theo — 03/10/2026

## Material Passport

- Skill: academic-research-suite, experiment-agent; thực thi và xác minh inline.
- Authorization: tác giả cho phép dọn file tạm và chạy các thí nghiệm cần thiết, GPU trên Colab, báo cả kết quả âm tính.
- Môi trường: Python conda `clrr`; Colab A100 Standard, không high-mem.
- Outputs mới: `outputs_rebuttal/followup_20261003/`; không overwrite kết quả lịch sử.
- Trạng thái: phần đánh giá lại/checkpoint diagnostics đã hoàn tất; Turkish PAUSED_BY_USER lúc08:05 ngày03/10/2026. Baseline lưu checkpoint epoch2/global_step314; ba methods còn lại chưa train. Không tự chạy tiếp Turkish.

## Phạm vi và thứ tự

1. Cleanup gói truyền/file tạm đã có artifacts xác minh: 5 file, có paths/bytes/SHA trong `cleanup_manifest.json`. Giữ dữ liệu, checkpoints, code và kết quả nghiên cứu; không xóa thay đổi working tree khác.
2. Probe precision bằng tiny mBART qua Accelerate và bốn wrapper: ghi dtype thật tại projection encoder. Không suy từ code review rằng historical run chắc chắn sai.
3. Đánh giá lại 12 trained checkpoints đã pin trong manifest, full original test, FP32/TF32 off, batch8/pad multiple8, beams4, historical max_length. Baseline/Full/LSR-only/CLRR-only dùng chung decoding trong mỗi family. Không train lại ở bước này.
4. Với năm Full checkpoints: inference routing-off alpha=0 trên **cùng trained weights**. Đo mức phụ thuộc vào skip ở inference; không gọi là LSR-only training ablation, không chứng minh training contribution.
5. Probe batch1 vs batch8 trên sentinel cho ba model còn lệch prediction cũ. Không sửa prediction lịch sử từ sentinel.
6. Chuẩn bị OPUS-100 Turkish→English20k v2, rồi pilot mBART bốn cấu hình seed42 khi correctness/precision gates đạt. Chọn/báo mọi model, không chọn dataset hay variants theo test.

## Protocol Turkish đã khóa trước chạy

Dataset `data_processed/opus100_turkish_english_20k_v2/` lấy từ Helsinki-NLP/opus-100 revision `805090dc28bf78897da9641cdf08b61287580df9`. Train20.000 **unique pairs** lấy mẫu mới xác định bằng NumPy RNG42 từ official train; dev1946 sau dedup/loại source trùng test; giữ official test2000/thứ tự và18duplicate excess pairs. Không sửa v1; không gọi test unique hoặc cam kết pretraining-disjoint.

| Method | CLI method | Alpha | Lambda |
|---|---|---:|---:|
| Baseline | baseline | 0 | 0 |
| CLRR-only | clrr-enc | 0.1 | 0 |
| LSR-only | jepa | 0 | 0.1 |
| Full | jepa-clrr-enc | 0.1 | 0.1 |

- Backbone pretrained mBART-50 pin revision trước load; cùng initial weights/tokenizer.
- Source `tr_TR`, target `en_XX`; encoder distance2.
- Full-model memory calibration tại batch32,length256 đã qua forward/backward/optimizer smoke, peak26.624GiB trên A10040GB; chọn batch32×accumulation4 chung cho cả nhóm. Calibrated weights là bản tạm, không dùng làm initial weights của pilot.
- Seed/data_seed42; LR5e−5, AdamW, weight_decay0, warmup_ratio0.06.
- Batch32 × accumulation4 =128 cho cả bốn methods; max source/target256; max20epochs, patience4, cùng validation chrF++ chọn best như pipeline trước. Microbatch4 vận hành thử bị ngưng trước epoch1 vì GPU chỉ dùng khoảng37%, ước lượng4giờ/20epochs/run; không dùng trial đó làm kết quả. Baseline restart từ seed42 với cấu hình mới, không resume ngang protocol.
- Train BF16 + gradient checkpointing; **validation/test generation explicit FP32** cho tất cả methods. Validation greedy beam1, final test beam4, eval batch8; cùng TF32 policy được ghi trong run configuration.
- BLEU13a là quality endpoint chính; raw corpus chrF++6/2 endpoint phụ, char-only chrF6/0 diagnostic. References nguyên CSV, không decode labels, không bỏ empty predictions.
- Pilot một seed là exploratory. Primary comparisons Full−Baseline và Full−LSR-only; bootstrap paired sentences không thay training-seed uncertainty. Giữ CLRR-only để phân rã đóng góp; báo tương tác thận trọng.
- Không tăng λ/alpha/patience riêng Full sau nhìn test. Confirmation nhiều seeds chỉ khi pilot đáng tiếp tục; nếu Full không có lợi ích thêm, báo rõ và chốt kế hoạch sửa method trước khi mở rộng GPU.

## Gates và điểm cần báo tác giả

- Loader strict, archive SHA/vocab IDs/data order đúng, generation dtype được probe xác nhận.
- Dataset sources disjoint; pinned provenance/hash/row IDs xác minh local và Colab.
- Trainer smoke backward/save/reload và FP32 generation qua wrapper phải đạt trước full train.
- Kết quả mới báo từng bước tốt/xấu. Nonsignificant không tự chứng minh equivalence; một pilot tốt không tự chứng minh generalization/multi-seed.
- Missing mBART CLRR-only best504: không dùng432 thay best. Ash Full diagnostic/reevaluation này là **patience4**, không gán sang epoch20/patience10.
- Kiểm tra tiến độ và báo chat tối thiểu mỗi10phút khi chạy; khi lỗi/hoàn tất báo ngay. Lưu đủ artifacts về local trước tắt Colab.

## Tiêu chí diễn giải cuối

- **Implementation ổn** khi các gates đúng; tách khỏi **chất lượng tốt hơn controls**.
- Nếu Full hơn Baseline nhưng gần LSR-only: lợi ích thêm của CLRR chưa đủ bằng chứng.
- Nếu Full kém controls trong protocol công bằng: giữ kết quả âm tính, chưa tăng chi phí confirmation tự động.
- Không kết luận đủ A* từ residual/rank hoặc một seed; cần contribution và robustness phù hợp claim.
