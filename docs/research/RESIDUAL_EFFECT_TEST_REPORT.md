# Residual-effect RTX — test và paired bootstrap

Ngày 12/10/2026. Đánh giá test được tác giả yêu cầu sau khi xem validation.
Đây là phân tích exploratory; test đã được xem, và không còn là xác nhận độc lập
cho một phương pháp được chọn tiếp dựa trên chính các số này.

## Protocol và xác minh

- Amis→Mandarin, 575 câu từ CSV test gốc; cùng thứ tự source/target cho cả7 runs.
- RTX6000, FP32, TF32 off, beam4, max_length256, eval_batch2. Không train lại.
- Checkpoint best validation chrF++ giữ nguyên; selection lock được tạo trước test.
- SacreBLEU2.6.0: BLEU tokenizer zh, chrF++ word_order2, chrF0 word_order0.
- Predictions, metrics và COMPLETE đã upload public vào đúng folder checkpoint
  trong `FiveC/amis-rewire-checkpoints`; checksum remote đã đối chiếu.
- Test queue đã kết thúc; weights tải tạm trên RTX đã được xóa sau xác minh.
  Local chỉ giữ CSV/JSON/Markdown nhỏ.

Remote artifacts được kiểm tra tại revision
`3298ef63fba5555f361370f45db178bfaa8e1b5f`.
Evidence: `outputs/experiments/amis_residual_effect_rtx_test/TEST_ARTIFACT_VERIFICATION.json`.

## Kết quả test

| ID | Nhánh | BLEU | chrF++ | chrF0 | Δ BLEU vs baseline |
|---|---|---:|---:|---:|---:|
| R1 | Baseline | 19.3926 | 13.8889 | 18.0917 | 0.0000 |
| R2 | Full cũ: CLRR + LSR | 18.7821 | 18.1317 | 17.6237 | −0.6105 |
| P1 | CLRR-only | 19.7806 | 18.7598 | 18.4717 | +0.3880 |
| P2 | LSR + ordinary InfoNCE | 19.0496 | 18.3541 | 17.8877 | −0.3430 |
| P3 | CLRR + LSR + ordinary InfoNCE | **19.9158** | **18.8726** | **18.5920** | **+0.5232** |
| P4 | CLRR + effect InfoNCE, không LSR | 19.2178 | 18.4420 | 18.0395 | −0.1748 |
| P5 | CLRR + LSR + effect InfoNCE | 19.2836 | 18.4591 | 18.1236 | −0.1090 |

Không chọn lại epoch theo test. P3 dùng checkpoint epoch16; P1 epoch18;
P2/P4/P5/baseline epoch19; Full cũ epoch14.

## Ý nghĩa thống kê

10.000 paired sentence bootstrap resamples, RNG42; mỗi resample có575 câu lấy
có hoàn lại. Cùng indices cho cả hệ thống; tính corpus metrics, không lấy trung
bình sentence BLEU. CI95 là percentile CI của chênh lệch điểm, chưa hiệu chỉnh
đồng thời các CI. p raw dùng native SacreBLEU paired bootstrap và được tính lại
độc lập bằng sufficient statistics, khớp dưới1e-12.

Thiết kế đã ghi trước khi xem điểm test: năm nhánh P1–P5 vs baseline; P3/P5 vs
Full cũ; P3 vs P2; P5 vs P3; P4 vs P1; P5 vs P4. Tổng11 cặp ×3 metrics =
**33 phép kiểm định trong một Holm family**, không tách lại theo kết quả.

| Candidate − control | Δ BLEU | CI95 của Δ | p raw | p Holm33 | Có ý nghĩa sau Holm? |
|---|---:|---|---:|---:|---|
| P1 − baseline | +0.3880 | [−0.8902, +1.7060] | 0.200380 | 1.000000 | Không |
| P2 − baseline | −0.3430 | [−1.6276, +0.9458] | 0.206979 | 1.000000 | Không |
| P3 − baseline | +0.5232 | [−0.7101, +1.7748] | 0.149385 | 1.000000 | Không |
| P4 − baseline | −0.1748 | [−1.3929, +1.0559] | 0.284372 | 1.000000 | Không |
| P5 − baseline | −0.1090 | [−1.3879, +1.2407] | 0.339666 | 1.000000 | Không |
| P3 − Full cũ | +1.1337 | [−0.0192, +2.2697] | 0.024398 | 0.756324 | Không |
| P5 − Full cũ | +0.5015 | [−0.4652, +1.4786] | 0.121988 | 1.000000 | Không |
| P3 − P2 | +0.8662 | [−0.3146, +2.0441] | 0.067593 | 1.000000 | Không |
| P5 − P3 | −0.6322 | [−1.8604, +0.6124] | 0.121688 | 1.000000 | Không |
| P4 − P1 | −0.5628 | [−1.4766, +0.3434] | 0.095790 | 1.000000 | Không |
| P5 − P4 | +0.0659 | [−0.9516, +1.1268] | 0.355764 | 1.000000 | Không |

**Không phép kiểm định nào trong33 tests đạt p_Holm<0.05.**
Toàn bộ33 hàng gồm chrF++/chrF0 tại
`outputs/experiments/amis_residual_effect_rtx_test/statistics/PAIRED_BOOTSTRAP.md`
và `paired_bootstrap.json`/`paired_bootstrap.csv` cùng thư mục.

P3 − baseline trên chrF0: +0.5003, CI[−0.4325,+1.4701], p raw0.119288,
p Holm1.000000. Trên chrF++: +4.9836 nhưng CI rất rộng
[−0.2042,+13.2635], p raw0.122888, p Holm1.000000. Không dùng chênh chrF++
lớn này để khẳng định cải thiện lớn và ổn định; báo cùng BLEU/chrF0 và kết quả
audit sparse word n-grams đã có.

Native SacreBLEU p-value và percentile delta CI là hai cách xây dựng khác nhau,
có thể khác kết luận gần ngưỡng: P3−Full cũ BLEU có p raw<.05 nhưng percentile
CI vẫn chứa0 rất nhẹ. Sau Holm33 kết luận là không có ý nghĩa; không chọn cách
báo thuận lợi hơn. P3−Full cũ chrF0 có CI[+0.1333,+1.8064], p raw0.011499,
nhưng p Holm0.379462 nên vẫn không đạt tiêu chuẩn đã chốt.

## Diễn giải cho phương pháp

P3 tiếp tục đứng đầu điểm số và tăng ở BLEU/chrF0 so baseline trên cả validation
và test. Tuy nhiên test gain chỉ+0.5232 BLEU và CI chứa0: **ứng viên có triển
vọng, chưa xác nhận vượt baseline**. Không gọi đây là bằng chứng ổn định qua seeds.

P1 CLRR-only từ thấp hơn baseline trên validation sang cao hơn trên test; do đó
không kết luận CLRR luôn có hại từ một validation seed. P4/P5 vẫn không vượt
baseline về BLEU. NO-GO của residual-effect P5 trên gate validation giữ nguyên;
test không được dùng để đảo gate hoặc đổi checkpoint.

Nếu tiếp tục P3, cần khóa phương pháp và kiểm tra novelty, rồi paired runs trên
các training seeds khác. Sentence bootstrap ở đây chỉ đo độ bất định theo câu
với checkpoint seed42 cố định; không đo biến thiên do training hoặc chứng minh
novelty/cơ chế/khả năng tổng quát. Không mở thêm training trong yêu cầu này.

## Sửa lỗi đánh giá và tái lập

1. `engine.test_run` đặt eval_strategy='no' khi tạo Trainer chỉ để predict;
   sửa lỗi thiếu eval_dataset của Transformers4.57.6. Không đổi recipe train.
2. Auxiliary InfoNCE bằng0 cho singleton **evaluation** batch cuối; không có
   negatives, không thay đổi logits. Training vẫn từ chối batch<2. Cả4 CPU
   correctness tests PASS, gồm singleton logits và kiểm tra gradient/train cũ.
3. Evaluation source snapshot và patch receipt được lưu trong workspace test;
   validation checkpoints/receipts cũ không bị sửa.

Worker: `scripts/analysis/evaluate_frozen_rtx.py`, stage prepare/queue/run;
cohort đầu vào `amis_residual_effect_rtx` và `amis_token_rtx`.
Output riêng `outputs/experiments/amis_residual_effect_rtx_test`.

Tính lại thống kê bằng conda clrr:

```bash
python scripts/analysis/paired_checkpoint_statistics.py --predictions-dir outputs/experiments/amis_residual_effect_rtx_test --references data_processed/amis_mandarin/test.csv --design outputs/experiments/amis_residual_effect_rtx_test/statistics_design.json --output outputs/experiments/amis_residual_effect_rtx_test/statistics
```

Phương pháp/library: [SacreBLEU paired significance tests](https://github.com/mjpost/sacrebleu#paired-significance-tests).
