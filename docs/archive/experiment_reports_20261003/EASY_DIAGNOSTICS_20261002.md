# Các phép đo ít tốn để kiểm chứng CLRR — 2026-10-02

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, thực hiện inline.
- Origin Mode: run + validate.
- Origin Date: 2026-10-02, Asia/Saigon.
- Verification Status: EXECUTED; kết quả có JSON/CSV, input hashes và kiểm tra API riêng.
- Version Label: clrr_easy_diagnostics_v1.
- Scope: CPU trong conda `clrr`; không train, không tải pretrained weights/checkpoint.
- Trọng số ngẫu nhiên chỉ dùng cho kiểm tra implementation. Đo chất lượng dùng predictions đã audit của checkpoint thực.
- Đây là báo cáo mới. Giữ nguyên số liệu/tài liệu cũ; 19 file trong restoration manifest vẫn khớp SHA-256.

## 1. Kết luận thực hành

**Có bằng chứng mới hữu ích để bảo vệ bài, nhưng chưa đủ chứng minh CLRR thắng ổn định qua training seeds.**

1. Implementation vượt qua các kiểm tra routing, detach, cache, zero-coefficient equivalence và gradient checkpointing trên mô hình nhỏ của bốn kiến trúc.
2. mBART Full có mức tăng trên ký tự thật, vượt cả baseline và LSR-only về point estimate. Dấu dương không phụ thuộc vào một câu test duy nhất.
3. mT5 có mức tăng chrF rõ và còn ý nghĩa sau hiệu chỉnh chung 50 phép so sánh hậu nghiệm.
4. mBART/NLLB chưa xác lập ưu thế thống kê; ByT5 và Asháninka chưa tạo thêm bằng chứng về tính tổng quát.
5. Chưa có lý do từ các kiểm tra này để kết luận cần sửa công thức CLRR ngay. Ưu tiên thí nghiệm đối chứng và multi-seed; kiểm tra gradient/norm trên checkpoint thực khi có weights phù hợp.

## 2. Kiểm tra implementation — TỐT trong phạm vi đã thử

Artifact: [correctness.json](../../../outputs_rebuttal/easy_diagnostics_20261002/correctness.json).

Môi trường: PyTorch 2.6.0+cu124, Transformers 4.57.6; thực thi CPU float32. Mỗi model nhỏ có 4 encoder layers, 2 decoder layers, hidden size 32; trọng số khởi tạo seed 42. Không optimizer step.

Kiến trúc: `MBartForConditionalGeneration`, `M2M100ForConditionalGeneration` dùng cho NLLB, `MT5ForConditionalGeneration`, và `T5ForConditionalGeneration` đại diện kiến trúc ByT5. Không dùng tokenizer hoặc weights pretrained của các model lớn.

| Phép kiểm tra | Kết quả |
|---|---|
| Routing đúng khoảng cách d=2 | PASS: đầu ra toy [1, 2, 3.1, 4.2] với alpha=0.1 |
| Gradient theo nhánh residual | PASS: nguồn skip không nhận gradient từ riêng cạnh skip; destination gradient = 1 |
| Tensor trong cache đã detach | PASS |
| alpha=0, lambda=0 so với baseline | Logits/loss/gradient khớp chính xác trên cả 4 kiến trúc |
| Số parameter khi thêm wrapper | Bằng baseline; không thêm parameter |
| Batch khác xen giữa hai lần gọi | Logits không đổi; cache rỗng sau forward |
| Beam generation lặp lại sau input khác | Giống hệt; thử num_beams=2 trên encoder CLRR |
| LSR source/target | Source nhận gradient auxiliary khác 0; target hidden không requires_grad |
| Checkpointing, dropout=0 | 32/32 so sánh PASS: 4 kiến trúc × 4 methods × 2 chế độ |
| Checkpointing, dropout=0.1 | 8/8 so sánh Full PASS: 4 kiến trúc × 2 chế độ |

Hai chế độ checkpointing là `use_reentrant=True/False`; so sánh toàn vector gradient của cùng weights/input/RNG. Sai khác L2 tương đối lớn nhất **1.61006724e-7**, dưới tolerance **1e-5**; logits và loss khớp.

Warning `None of the inputs have requires_grad=True` xuất hiện với reentrant checkpointing trong nhánh target no-grad; các phép so sánh gradient source vẫn PASS. Không lấy warning này làm bằng chứng mất gradient source.

**Giới hạn:** các kết quả không xác minh mọi cấu hình decoder/both-stack, mixed precision/GPU, layerdrop, phiên bản thư viện lịch sử hoặc quá trình optimization của checkpoint đã train. Các gradient norms trong JSON là của mô hình ngẫu nhiên; không dùng để giải thích hiệu quả của mBART/NLLB đã train.

## 3. ChrF trên ký tự — kết quả mới từ predictions thực

Artifacts: [character_chrf_bootstrap.csv](../../../outputs_rebuttal/easy_diagnostics_20261002/character_chrf_bootstrap.csv), [robustness_protocol.json](../../../outputs_rebuttal/easy_diagnostics_20261002/robustness_protocol.json).

Giữ reference gốc và đầy đủ test rows; chỉ strip khoảng trắng đầu/cuối. SacreBLEU 2.6.0: char_order=6, **word_order=0**, beta=2; các cấu hình còn lại theo protocol chuẩn. Đây là chrF diagnostic, **không thay thế chrF++/BLEU chính thức**.

Paired bootstrap 10,000 lần, seed 42, đơn vị câu; CI percentile 95% của chênh lệch corpus score. P-value theo implementation SacreBLEU; đã đối chiếu public `PairedTest` API trên mBART Full-vs-Baseline. Tất cả là kiểm tra hậu nghiệm, fixed checkpoints, một training seed.

| Dataset / backbone / so sánh | Delta chrF | CI 95% | p Holm chung 50 | Đánh giá |
|---|---:|---|---:|---|
| Amis mBART Full − Baseline | +0.6907 | [-0.2363, 1.6608] | 1.0000 | Tín hiệu dương; chưa đủ chắc |
| Amis mBART Full − LSR-only | +0.6681 | [-0.2182, 1.5563] | 1.0000 | Đóng góp CLRR cần xác nhận qua seeds |
| Amis mBART CLRR-only − Baseline | -0.9957 | [-1.8811, -0.1256] | 0.3696 | Point estimate xấu; chưa có ý nghĩa sau Holm |
| Amis mBART LSR-only − Baseline | +0.0226 | [-0.9934, 1.0638] | 1.0000 | Gần như ngang baseline trên ký tự |
| Amis NLLB Full − Baseline | +0.6349 | [-0.0498, 1.3291] | 0.8657 | Dương nhưng chưa đủ chắc |
| Amis mT5 Full − Baseline | +1.6108 | [1.2587, 1.9804] | 0.0050 | TỐT: bằng chứng rõ trên checkpoint này |
| Amis ByT5 Full − Baseline | -0.2163 | [-0.6192, 0.1534] | 1.0000 | Không hỗ trợ claim cải thiện mọi backbone |
| Asháninka mBART Full p10/best14 − Baseline p4 | +0.0337 | [-0.4550, 0.5177] | 1.0000 | Gần ngang; khác patience, không phải đối chứng hoàn toàn đồng nhất |

CSV có cả Holm cho 8 diagnostics mới và Holm kết hợp **42 kiểm tra audit cũ + 8 mới = 50**. Không chỉnh lại file audit cũ. Dùng family chung 50 khi mô tả bằng chứng bổ sung của đợt này. Không coi p nhỏ của một diagnostic hậu nghiệm là validation độc lập.

CI percentile và p-value centered absolute-difference của SacreBLEU không phải hai phép đảo tương đương; vì vậy NLLB có raw p=0.0338 nhưng CI vẫn chứa 0. Không lấy raw p này để tuyên bố thắng sau hiệu chỉnh.

### Insight mới có ích

Trong mBART, chrF ký tự của Baseline / CLRR-only / LSR-only / Full lần lượt là **18.2288 / 17.2331 / 18.2514 / 18.9195**.

Điều này làm rõ hai điểm:

- Mức tăng chrF++ lớn của LSR-only không đi kèm mức tăng ký tự tương ứng; cần thận trọng với word-bigram rất thưa trong tiếng Trung.
- Full vẫn có mức tăng ký tự trên cả Baseline và LSR-only. Vì vậy tín hiệu Full không hoàn toàn do thành phần word-bigram thưa. Chưa đủ để kết luận phối hợp CLRR+LSR tạo hiệu ứng ổn định hoặc cơ chế bảo toàn hình thái.

## 4. Ảnh hưởng từng câu — TỐT cho độ bền dấu của point estimate

Artifacts: [influence_summary.csv](../../../outputs_rebuttal/easy_diagnostics_20261002/influence_summary.csv), [influence_protocol.json](../../../outputs_rebuttal/easy_diagnostics_20261002/influence_protocol.json). Có bảng đầy đủ 4,600 dòng: 4 comparisons × 2 metrics × 575 lần bỏ một câu.

Mỗi lần chỉ bỏ một câu, tính lại corpus score; **kết quả chính vẫn giữ cả 575 câu**. Shortcut cộng/trừ sufficient statistics đã đối chiếu direct corpus API ở câu ảnh hưởng lớn nhất của cả 8 comparisons/metrics.

| So sánh, chrF word_order=0 | Delta đủ 575 câu | Min–max sau bỏ một câu bất kỳ | Số lần delta ≤ 0 |
|---|---:|---|---:|
| mBART Full − Baseline | +0.6907 | +0.5673 đến +0.7984 | 0/575 |
| mBART Full − LSR-only | +0.6681 | +0.5015 đến +0.8226 | 0/575 |
| NLLB Full − Baseline | +0.6349 | +0.5186 đến +0.7252 | 0/575 |
| mT5 Full − Baseline | +1.6108 | +1.5686 đến +1.6346 | 0/575 |

Với **chrF++**, mBART Full−Baseline dao động **+0.3844 đến +7.6908**; bỏ câu index 170 làm mức tăng giảm từ +5.0277 xuống +0.3844. Đây là nhạy cảm lớn về độ lớn; không phải công thức tính sai.

Với **chrF ký tự**, bỏ chính câu 170 vẫn còn **+0.5673**. Với Full−LSR-only, câu ảnh hưởng lớn nhất là index 65; bỏ câu đó vẫn còn **+0.5015**.

**Claim có thể dùng với phạm vi rõ:** “Trên các checkpoint Amis→Mandarin đã lưu, mức tăng chrF ký tự của CLRR+LSR trên mBART so với Baseline và LSR-only vẫn dương khi bỏ bất kỳ một câu test nào. Tuy nhiên CI bootstrap vẫn chứa 0; cần xác nhận qua training seeds.”

Leave-one-out không chứng minh mọi subgroup đều cải thiện, không thay thế bootstrap hoặc multi-seed, và không chứng minh bảo toàn affix.

## 5. Những phần còn cần checkpoint hoặc train

Không tìm thấy `.safetensors`/`pytorch_model.bin` của checkpoint đã train trong các thư mục kết quả/checkpoint local đã kiểm tra. Chưa tải archive lớn hoặc chạy suite diagnostics cũ có vấn đề về loader.

| Câu hỏi còn lại | Cần gì | Ưu tiên |
|---|---|---|
| Residual có quá lớn/đổi hướng sai trên model đã train? | Đúng checkpoint + forward đo RMS, tỷ lệ norm, cosine theo layer | Sau khi có weights/provenance |
| CE và weighted LSR có xung đột hoặc lệch scale? | Đúng weights + vài backward, log riêng hai gradient trước zero_grad/clip | Cần checkpoint hoặc instrumentation khi pilot |
| Hidden representation có collapse hoặc giữ morphology? | Checkpoint đúng; SVD đầy đủ/probe với nhãn hợp lệ và controls | Chưa được chứng minh bằng phép đo đợt này |
| CLRR thêm giá trị trên LSR-only qua seeds? | Các run có cùng protocol và matched seeds | Thiết yếu cho đóng góp riêng phương pháp |
| Tổng quát sang ngôn ngữ khác? | Turkish→English20k với Baseline / LSR-only / Full | Bước train tiếp theo hợp lý |

Loader phân tích hiện có đoán method từ tên thư mục và có fallback về pretrained model khi thiếu weights; không dùng kết quả từ loader đó làm bằng chứng nếu chưa xác minh đúng checkpoint/architecture/flags. Không chạy thêm phép đo cơ chế trên model ngẫu nhiên để thay thế checkpoint thực.

## 6. Bước tiếp theo: Turkish→English20k

**Chưa khởi động train trong đợt này.** Kế hoạch đầy đủ: [NEXT_STEPS_AUDITED_20261002.md](NEXT_STEPS_AUDITED_20261002.md).

Đề xuất chạy pilot mBART **Baseline → LSR-only → CLRR+LSR**, một seed, cùng training/evaluation/checkpoint selection protocol và budget. Nếu tiếp tục multi-seed thì dùng matched seeds cho cả ba; NLLB làm sau khi pilot có thông tin.

Trước pilot phải xử lý dataset hiện có: validation/test có **17 source trùng**, trong đó **14 exact pairs**; train có **399 duplicate-pair excess rows**. Chuẩn bị version mới có 20k train pairs duy nhất, validation không trùng test; giữ test chính thức và ghi rõ split/seed/provenance. Giữ dataset version cũ. Không dùng test để chọn alpha/lambda/LR.

Turkish nằm trong pretraining languages của mBART; gọi thí nghiệm này là **fine-tuning với ngân sách dữ liệu 20k**, không phải bằng chứng thích nghi sang ngôn ngữ hoàn toàn chưa thấy.

Nếu chạy Colab: A100 không high-mem; cập nhật chat mỗi 10 phút theo yêu cầu đã thống nhất.

## 7. Tái chạy và artifacts

Các script chỉ ghi vào `outputs_rebuttal/easy_diagnostics_20261002/`, không sửa historical scores/docs. Dùng Python của environment `clrr`:

```powershell
conda run -n clrr python -X utf8 scripts/revalidation/measure_clrr_correctness.py
conda run -n clrr python -X utf8 scripts/revalidation/measure_saved_output_robustness.py
conda run -n clrr python -X utf8 scripts/revalidation/measure_prediction_influence.py
```

- [verification_receipt.json](../../../outputs_rebuttal/easy_diagnostics_20261002/verification_receipt.json): assertions và kiểm tra SHA-256 19 file cũ.
- [output_summary.csv](../../../outputs_rebuttal/easy_diagnostics_20261002/output_summary.csv): exact matches, empty predictions, character length ratios; diagnostics phụ, không phải morphology metrics.
- Input chất lượng chính: [all_verified_translation_scores.csv](../../../outputs_rebuttal/metric_audit_20261002/full/all_verified_translation_scores.csv), 36 runs, và raw test.csv theo SHA-256 được lưu.
