# Residual-effect RTX pilot — validation đã xác minh

Ngày 12/10/2026. P1–P5 hoàn tất và queue kết thúc. Đây là development seed42,
validation 576 câu, beam1 FP32; chưa đánh giá test trong đợt này.

Sau báo cáo validation, tác giả đã yêu cầu đánh giá test ngày 12/10/2026.
Kết quả riêng tại [RESIDUAL_EFFECT_TEST_REPORT.md](RESIDUAL_EFFECT_TEST_REPORT.md);
checkpoint vẫn được chọn bằng validation, không chọn lại theo test.

## Protocol và bằng chứng

Đã tính lại BLEU zh, chrF++ và chrF0 từ CSV predictions của cả bảy runs, đối
chiếu source/target với CSV gốc: khớp số đã lưu với sai khác dưới 1e-8.
Best metric khớp selected trainer state và maximum validation chrF++ trong
final trainer history. Tất cả chạy đủ20 epoch; early stopping không cắt run nào.

Cấu hình thực tế cả bảy runs khớp: RTX6000, seed/data_seed42, LR5e-5,
microbatch4 × accumulation32, FP32, TF32 off, gradient checkpointing,
warmup0.06, weight_decay0, AdamW fused, linear scheduler, patience10,
chọn checkpoint bằng validation chrF++ beam1. Accumulation không tăng số
negatives của contrastive; các nhánh đều dùng cùng microbatch4.

Repo public: `FiveC/amis-rewire-checkpoints`. Đã kiểm tra metadata remote tại
revision `e04195904361dcbdf36d4ae6c7f577ca37eb3256`: kích thước và LFS SHA-256
checkpoint khớp receipts; các artifacts nhỏ khớp nội dung/hash local.
Không tải weights về máy local. Không còn best.zip, best_model hoặc checkpoint
epoch trong bảy thư mục run trên RTX sau cleanup được xác minh.

Evidence nhỏ: `outputs/experiments/amis_residual_effect_rtx/VALIDATION_VERIFICATION.json`.
Queue log có `GUARDED_QUEUE_COMPLETE`; queue/worker đã kết thúc.

## Bảng kết quả

| ID | Nhánh | Best epoch | BLEU | chrF++ | chrF0 | Δ BLEU vs baseline |
|---|---|---:|---:|---:|---:|---:|
| R1 | Baseline | 19 | 17.6823 | 14.7030 | 16.9547 | 0.0000 |
| R2 | Full cũ: CLRR + LSR | 14 | 17.1156 | 14.3450 | 16.4809 | −0.5668 |
| P1 | CLRR-only | 18 | 16.6216 | 14.1252 | 16.2232 | −1.0607 |
| P2 | LSR + ordinary InfoNCE | 19 | 17.2700 | 14.4825 | 16.5855 | −0.4124 |
| P3 | CLRR + LSR + ordinary InfoNCE | 16 | **18.4584** | **15.1246** | **17.3609** | **+0.7761** |
| P4 | CLRR + effect InfoNCE, không LSR | 19 | 15.7011 | 13.6663 | 15.6602 | −1.9812 |
| P5 | CLRR + LSR + effect InfoNCE | 19 | 17.1512 | 14.4008 | 16.5448 | −0.5311 |

chrF++ là metric chọn checkpoint; BLEU/chrF0 được báo tại chính checkpoint đó.
Native chrF++ cần diễn giải cùng vấn đề word bigrams thưa trong Mandarin đã
audit; bảng này vẫn giữ protocol đã khóa, không chọn lại checkpoint theo BLEU.

## Gate đã khóa và kết luận

Gate trong `configs/experiments/amis_residual_effect_rtx.json` áp dụng cho P5.

| Điều kiện | Kết quả | Quyết định |
|---|---:|---|
| P5 − baseline ≥ +0.30 BLEU | −0.5311 | FAIL |
| P5 − Full cũ ≥ +0.30 BLEU | +0.0356 | FAIL |
| P5 − P2 ≥ +0.20 BLEU | −0.1188 | FAIL |
| P5 − P3 ≥ +0.20 BLEU | −1.3072 | FAIL |
| chrF0 không giảm so baseline | −0.4099 | FAIL |

**NO-GO cho cấu hình residual-effect hiện tại.** Không mở rộng cấu hình P5
sang A100/multiple seeds theo gate của đợt này. Một seed chưa chứng minh effect
objective luôn thất bại, nhưng đủ để không mở rộng cấu hình đang thử.

P3 là tín hiệu đáng theo đuổi: hơn baseline +0.7761 BLEU/+0.4216 chrF++/
+0.4062 chrF0; hơn Full cũ +1.3428 BLEU, hơn P2 +1.1884 BLEU.
Các số này hỗ trợ một hướng kết hợp CLRR, LSR và ordinary contrastive trên
validation seed42. Chưa chứng minh ổn định theo seed hoặc ưu thế trên test.
Không chuyển novelty của effect objective sang P3: P3 dùng alignment thông
thường, cần đánh giá novelty riêng trước khi chọn làm đóng góp chính.

P4 − P1 = −0.9205 BLEU: thêm effect objective khi không có LSR chưa giúp CLRR.
P5 − P4 = +1.4501 BLEU: giữ LSR hữu ích trong hai nhánh effect của pilot này;
không có cơ sở bỏ LSR từ kết quả hiện có. P5 chỉ hơn Full cũ +0.0356 BLEU,
không đạt mức tăng có giá trị theo ngưỡng đã khóa.

## Bước tiếp theo để tác giả quyết định

1. Giữ P3 như candidate development; giữ P4/P5 và artifacts để truy vết NO-GO.
2. Trước train tiếp, rà lại novelty của CLRR + ordinary contrastive và xem
   điều gì vượt ngoài một tổ hợp objective đã biết. Pilot này không xác minh novelty.
3. Nếu chọn P3, khóa công thức/recipe rồi thiết kế paired confirmation với
   baseline và Full cũ ở các seeds khác; thêm P2 nếu cần xác minh đóng góp CLRR
   trong objective mới. Các đối chứng đã có chỉ tái sử dụng khi cùng seed/protocol.
4. Giữ test barrier trong giai đoạn chọn phương pháp. Chưa mở thêm training hoặc
   test evaluation tự động từ báo cáo này.

Không báo ý nghĩa thống kê từ một seed. Paired sentence bootstrap, nếu chạy,
chỉ phản ánh biến thiên câu của checkpoint này; không thay thế xác nhận theo seeds.

## Hugging Face

Các checkpoint nằm trong
`checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/<recipe>/`.
Recipe của baseline, Full cũ và CLRR-only: `rtx6000-fp32-b4-ga32-es10`.
Recipe P2–P5 thêm `-tau0.1-beta0.1`. Methods lần lượt:
`baseline`, `full`, `clrr`, `ordinary-lsr-v1`, `ordinary-full-v1`,
`effect-clrr-v1`, `effect-full-v1`.

P5 upload revision: `425406c35567dbe48bd672bbc6470ee4625097a1`.
Receipt đầy đủ của từng run có trong JSON verification; weights lưu trên HF,
còn local chỉ lưu evidence nhỏ.
