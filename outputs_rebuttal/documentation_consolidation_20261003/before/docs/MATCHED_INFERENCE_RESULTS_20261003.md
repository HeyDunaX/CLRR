# Kết quả đánh giá với precision thống nhất — 03/10/2026

## Material Passport

- Skill: academic-research-suite / experiment-agent, thực thi và xác minh inline.
- Status: VERIFIED cho 12 checkpoint và 5 inference routing-off; pilot Turkish đã dừng theo yêu cầu tác giả lúc08:05 ngày03/10, chưa có kết quả so sánh hoàn tất.
- GPU: A100 Standard, môi trường conda `clrr`; Torch2.6.0+cu124, Transformers4.57.6, Accelerate1.15.0, SacreBLEU2.6.0. Frozen requirements trong outputs.
- Weights: `FiveC/amis-rewire-checkpoints`, revision `63b0dbc4fab41a669ccb36ed2f22894868634b8a`, kiểm tra archive SHA và strict loading; cấu hình khôi phục từ paired baseline khi ZIP wrapper thiếu config, như lượt diagnostics trước.
- Dữ liệu: full original test, giữ thứ tự/reference/empty predictions; source/target CSV local và public có cùng parsed rows.
- Outputs mới: [scores](../outputs_rebuttal/followup_20261003/matched_inference_scores.csv), [bootstrap](../outputs_rebuttal/followup_20261003/matched_inference_bootstrap.csv), [protocol](../outputs_rebuttal/followup_20261003/analysis_protocol.json), predictions và receipt trong `reevaluation/`.
- 19 file kết quả/docs lịch sử vẫn khớp restoration SHA; không cập nhật các bảng cũ. Các số dưới thuộc một phiên bản đánh giá mới.

## 1. Vấn đề precision đã kiểm tra trực tiếp

Probe tiny mBART qua Accelerate BF16 cho thấy dtype tại attention projection:

| Method | Encoder mặc định | Decoder mặc định | Explicit FP32 |
|---|---|---|---|
| Baseline | FP32 | BF16 | cả hai FP32 |
| CLRR-only | FP32 | FP32 | cả hai FP32 |
| LSR-only | FP32 | FP32 | cả hai FP32 |
| Full | FP32 | FP32 | cả hai FP32 |

Nguyên nhân trong pipeline hiện tại: Accelerate bọc root `forward`, nhưng wrapper `.generate` chuyển sang base model; encoder được gọi riêng khi generation. Đây là xác minh **môi trường hiện tại**, không tự chứng minh dtype của mọi historical run.

Bản sửa thêm `--generation-precision fp32|bf16`, thống nhất autocast tại generation và tạm bỏ decorator root forward của Accelerate trong phạm vi này. Explicit FP32 tắt TF32 khi generate; sau đó phục hồi forward và TF32 cho training/loss evaluation. Không đổi công thức CLRR/LSR. Khi không truyền flag, CLI cũ giữ hành vi cũ. Cả bốn branches đã qua GPU BF16 train/backward/save/reload và CPU smoke.

## 2. Protocol đánh giá lại

FP32 weights và generation, TF32 off, eval/no_grad, beam4, batch8, pad multiple8. Max length256 cho mBART/mT5/ByT5,128 cho NLLB, theo manifest. Raw original references; corpus BLEUzh cho Amis,13a cho Asháninka; raw chrF++6/2, char-only chrF6/0 diagnostic. Không optimizer steps trong reevaluation.

Asháninka trong bảng là **Full patience4**, không phải selected epoch14/patience10 hoặc epoch20. Không đổi checkpoint-selection lịch sử; fixed saved checkpoints không tương đương với một fresh matched training protocol.

## 3. Điểm mới

| Dataset/backbone | Method | BLEU | chrF++ |
|---|---|---:|---:|
| Amis mBART | Baseline | 19.6144 | 14.0547 |
| Amis mBART | Full | 20.3896 | 19.0824 |
| Amis mBART | LSR-only | 19.7067 | 18.6169 |
| Amis NLLB | Baseline | 13.6186 | 10.4990 |
| Amis NLLB | Full | 14.2835 | 10.9179 |
| Amis mT5 | Baseline | 2.7887 | 3.8129 |
| Amis mT5 | Full | 4.5961 | 5.0425 |
| Amis mT5 | CLRR-only | 4.4393 | 5.1761 |
| Amis ByT5 | Baseline | 7.4856 | 8.2977 |
| Amis ByT5 | Full | 7.3160 | 8.0966 |
| Asháninka mBART | Baseline p4 | 3.9522 | 20.9190 |
| Asháninka mBART | Full p4 | 3.9072 | 20.7560 |

### Các thay đổi so với prediction lịch sử

| Model | Số câu đổi | ΔBLEU mới−cũ | ΔchrF++ mới−cũ |
|---|---:|---:|---:|
| mBART LSR-only | 15 | +0.0776 | +0.0416 |
| NLLB baseline | 80 | +0.1184 | +0.0602 |
| NLLB Full | 4 | +0.0016 | +0.0027 |
| ByT5 baseline | 44 | −0.0980 | −0.0080 |
| ByT5 Full | 7 | +0.0052 | +0.0011 |
| Asháninka baseline p4 | 313 | −0.0563 | +0.0654 |
| Asháninka Full p4 | 78 | −0.0599 | +0.0570 |

mBART baseline/Full và cả ba mT5 models khớp toàn bộ text predictions cũ. Các thay đổi không đủ để đảo kết luận chính. Vì môi trường/cache/padding/precision có thể khác lịch sử, không gán tất cả Δ cho precision hoặc tuyên bố historical predictions sai. Sentinel batch1 vs batch8 ở ba model còn lệch không tự giải quyết việc tái lập historical inference.

## 4. Kiểm định quality

10.000 paired bootstrap resamples, seed42; CI percentile95% của corpus-score delta. Family36 tests =12pairs×3metrics, Holm correction; exploratory cố định saved checkpoints. Một p BLEU được đối chiếu với native SacreBLEU PairedTest, khớp. Không thay training-seed uncertainty bằng sentence bootstrap.

| So sánh BLEU | Δ | CI95% | Holm p |
|---|---:|---|---:|
| mBART Full−Baseline | +0.7752 | [−0.4388,2.0441] | 1.0000 |
| mBART Full−LSR-only | +0.6830 | [−0.4806,1.8530] | 1.0000 |
| NLLB Full−Baseline | +0.6650 | [−0.3295,1.7156] | 1.0000 |
| mT5 Full−Baseline | +1.8074 | [1.1826,2.4746] | 0.0036 |
| mT5 Full−CLRR-only | +0.1567 | [−0.3769,0.7116] | 1.0000 |
| ByT5 Full−Baseline | −0.1696 | [−0.8869,0.5266] | 1.0000 |
| Asháninka Full−Baseline p4 | −0.0450 | [−0.5495,0.4557] | 1.0000 |

mT5 Full−Baseline vẫn có tín hiệu trên cả BLEU/chrF++/chrF0. Full−CLRR-only mT5 chưa rõ; chrF++ còn thấp hơn0.1336. mBART/NLLB positive point estimates chưa đủ chắc; mBART +5.0277chrF++ vẫn CI [−0.1872,13.2525], phù hợp cảnh báo word-bigram thưa từ audit trước. Không diễn giải nonsignificance là chứng minh bằng nhau.

## 5. Routing-off trên cùng trained Full weights

Chỉ đổi inference alpha0.1→0; tất cả weights/decoding giữ nguyên. Đây là **sensitivity của model đã train với CLRR**, không phải training ablation LSR-only.

| Model | Full BLEU | Routing-off BLEU | On−off | CI95% | Holm p |
|---|---:|---:|---:|---|---:|
| mBART Amis | 20.3896 | 16.4609 | +3.9287 | [2.7175,5.1948] | 0.0036 |
| NLLB Amis | 14.2835 | 12.9177 | +1.3659 | [0.3337,2.7009] | 0.4080 |
| mT5 Amis | 4.5961 | 4.1384 | +0.4577 | [0.0719,0.8816] | 0.4140 |
| ByT5 Amis | 7.3160 | 7.5166 | −0.2006 | [−0.7983,0.3788] | 1.0000 |
| mBART Asháninka | 3.9072 | 2.8312 | +1.0760 | [0.5619,1.5944] | 0.0056 |

**Tốt:** mBART Amis và Asháninka model thật sự phụ thuộc skip lúc inference theo kiểm định này. CLRR không phải một nhánh không được model dùng.

**Giới hạn:** bỏ thành phần mà model quen khi train tạo distribution shift. On−off dương không chứng minh train Full tốt hơn train LSR-only/Baseline, morphology preservation hay generalization. Asháninka phụ thuộc skip mà vẫn không hơn baseline là ví dụ rõ. ByT5 point estimate tốt hơn khi routing-off, nhưng uncertainty chưa đủ để khẳng định tắt routing luôn cải thiện.

## 6. Phương pháp hiện có đủ tốt không?

**Implementation có cơ sở để tiếp tục:** correctness tests, native gradient diagnostics và precision gates đã đạt; residual vừa phải, pooled spectrum không collapse về một hướng, routing có tác dụng đo được ở một số trained models.

**Quality contribution chưa đủ để bảo vệ claim rộng:** Full chưa vượt LSR-only mBART một cách chắc; mT5 Full chưa rõ lợi ích thêm so với CLRR-only; ByT5/Asháninka chưa có gain. Những phép đo này không biến bài thành bằng chứng multi-seed hoặc đủ cho method paper mạnh. Không tuyên bố universal anti-collapse hay universal improvement.

Pilot Turkish bốn configurations cùng protocol là bước tiếp theo để kiểm tra incremental CLRR contribution và chuyển sang target English. Nếu Full vẫn không hơn LSR-only/baseline có ý nghĩa thực tiễn, cần báo tác giả và tập trung cải thiện/thu hẹp phương pháp trước khi tốn thêm GPU cho confirmation. Nếu pilot tốt, vẫn cần matched seeds; một pilot không đủ chốt claim tổng quát.

## 7. Tình trạng Turkish và vận hành

- Dataset v2 đã chuẩn bị và hash/selected row IDs khớp giữa local và Colab:20k unique train/1946dev/2000official test.
- Không truncation ở dev/test tại max256; train chỉ1source và1target vượt256. Mean source/target train≈11.58/12.28tokens; số đầy đủ trong tokenization audit.
- Pretrained mBART pin `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`, native `tr_TR→en_XX`; zero-shot reference trước train, không dùng để chọn method.
- Microbatch4 thử vận hành dùng GPU thấp và dự kiến4giờ/20epochs/run, đã dừng trước epoch1; giữ trace riêng, không coi là result. Chọn một microbatch lớn hơn bằng memory calibration ở length256 cho Full, áp dụng chung cho cả bốn methods, global batch128 và hyperparameters giữ nguyên. Baseline restart từ seed42, không resume qua protocol.
- Lượt train đã dừng theo yêu cầu tác giả: baseline có checkpoint hoàn chỉnh gần nhất epoch2/global_step314, best validation chrF++39.8543. LSR-only/CLRR-only/Full chưa train. Zero-shot test20.5746BLEU/39.9016chrF++, không được so trực tiếp với validation greedy của baseline. Không có test score cuối cho trained Turkish baseline và không có so sánh phương pháp Turkish.
- Các kết quả trên **chưa bao gồm pilot Turkish**. Runbook và raw status nằm riêng; không có thay đổi số cũ. Không tiếp tục Turkish khi tác giả chưa yêu cầu.

## Kiểm tra cách suy luận

Đã phân biệt multiple comparisons, fixed-checkpoint vs seed variance, association vs cause, statistical vs practical effect, absence of significance vs equivalence, inference ablation vs training control, protocol versions và checkpoint-selection khác nhau. Không xóa negative results, chọn row có lợi, đổi λ từ test, hay gán dataset simulated20k là unseen/endangered-language adaptation. Không có gold morphology/semantic annotations để bảo vệ những claims đó từ các diagnostics này.
