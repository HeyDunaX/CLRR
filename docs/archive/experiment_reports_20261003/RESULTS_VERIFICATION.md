# Xác minh kết quả CLRR — 2026-10-02

## Kết luận

**Bài yếu đi ở hai luận điểm chính:** vượt full-tuning mBART/NLLB chưa có ý nghĩa thống kê, và bằng chứng cơ chế giữ hình thái/ổn định gradient/anti-collapse chưa đủ để kết luận. Không thể giữ claim mọi cải thiện p<0.001.

**Bằng chứng còn tốt:** CLRR+LSR vượt các cấu hình Strong LoRA A/B đã thử trên mBART/NLLB, còn có ý nghĩa sau Holm. mT5 encoder+LSR và decoder+LSR vượt baseline cũng còn có ý nghĩa. Đây là fixed-checkpoint, single-seed evidence; không chứng minh PEFT thất bại tổng quát hoặc hiệu quả nhất dưới cùng tuning budget.

## Phạm vi và provenance

Theo yêu cầu tác giả verify và sửa toàn bộ số trước khi train tiếp. Thực hiện validate inline theo academic-research-suite; không train, nạp weights, sinh lại bản dịch hoặc đổi checkpoint.

- **36 prediction outputs** đối chiếu count/order/source/target với original test.csv, giữ SHA-256 và full precision.
- **42 metric tests**: 21 paired system comparisons × BLEU/chrF++; 10.000 bootstrap samples, seed42.
- **4 p-value checks** với public SacreBLEU PairedTest API khớp; **3 API–CLI checks** trong audit ban đầu.
- **9.200 cosine rows**: 575 sentences × 8 layers × 2 systems; tái lập mean, không chạy encoder mới.
- Dataset/parameter/training-runtime metadata có nguồn riêng; không coi việc đọc metadata là rebuild model hoặc benchmark độc lập.
- Public Hub pinned revision `63b0dbc4fab41a669ccb36ed2f22894868634b8a`. NLLB small reports lấy bằng HTTP Range, không tải checkpoint archive 41GB.
- mT5 LSR-only, SVD, probing, no-sg test scores, sensitivity và latency **chưa được chứng nhận**. Coverage 36 không có nghĩa mọi claim lịch sử đều verified.

Nguồn chính: [METRICS.md](../../METRICS.md), [36 scores](../../../results/metrics_standardized_scores.csv), [translation audit](../../../outputs_rebuttal/metric_audit_20261002/full/translation_audit.json), [42 tests](../../../outputs_rebuttal/metric_audit_20261002/full/paired_bootstrap_standardized.csv), [bootstrap protocol](../../../outputs_rebuttal/metric_audit_20261002/full/bootstrap_protocol.json), [API checks](../../../outputs_rebuttal/metric_audit_20261002/full/bootstrap_api_checks.json).

## Những số dịch thay đổi đáng chú ý

| Số/bảng lịch sử | Trước | Sau raw-reference audit | Nhận xét |
| --- | ---: | ---: | --- |
| mBART baseline (paper rounded) | 19.6106 / 14.0538 | 19.6144 / 14.0547 | Rất nhỏ |
| mBART full (paper rounded) | 20.3927 / 19.0839 | 20.3896 / 19.0824 | Rất nhỏ |
| mBART LSR-only | 20.0828 / 16.0690 | 19.6291 / 18.5753 | Claim lợi ích routing giảm mạnh |
| mBART Middle-Align | 19.7243 / 15.5253 | 19.2893 / 18.4378 | Đối chứng mạnh hơn theo chrF++ |
| mBART CLRR-Dec+LSR | 20.3986 / 16.1939 | 19.9188 / 18.6999 | Encoder vẫn cao hơn, chưa chứng minh cơ chế |
| mBART Strong LoRA A | 10.4193 / 9.8008 | 10.2296 / 8.6319 | CLRR vẫn cao hơn |
| mBART Strong LoRA B | 14.0432 / 11.4901 | 13.8066 / 10.3293 | CLRR vẫn cao hơn |
| mT5 Middle-Align | 4.7498 / 5.9373 | 4.6625 / 5.2103 | Cao hơn decoder theo chrF++ |
| mBART LayerSkip | 19.2150 / 15.9569 | 18.7506 / 18.1079 | Historical comparison giữ trong audit |
| mT5 LayerSkip | 3.9721 / 5.2886 | 3.9292 / 4.5845 | Historical comparison giữ trong audit |

Các mốc “trước” là stored reports/docs, không phải score với cùng reference. mBART reference tokenizer roundtrip đổi **62/575** câu; raw metric JSON cũ được giữ để truy vết. Full precision trước/sau ở CSV audit, không dùng bảng làm nguồn tính lại.

NLLB tám outputs và năm Ash outputs tái lập theo protocol raw gốc; điểm chính không đổi. Historical ByT5 full **7.3108/8.0956**, baseline **7.5837/8.3057**: kết quả âm tính được giữ.

## Bootstrap mới: tốt ở đâu, chưa tốt ở đâu

Paired sentence indices dùng chung cho hai systems; metric tính corpus-level từ SacreBLEU sufficient statistics. CI là percentile95% của **signed delta**. p-value dùng SacreBLEU centered absolute-difference bootstrap; CI và p không là test inversion, có thể không khớp quyết định ở biên. Holm trên toàn42 tests; family lập trước computation nhưng là lựa chọn **post-hoc audit**, không phải preregistration.

| Đối sánh | Metric | Δ | 95% CI Δ | p raw | p Holm |
| --- | --- | ---: | --- | ---: | ---: |
| mBART: CLRR+LSR − Baseline | BLEU | +0.7752 | [-0.4388, 2.0441] | 0.0954 | 1.0000 |
| mBART: CLRR+LSR − Baseline | chrF++ | +5.0277 | [-0.1872, 13.2525] | 0.1214 | 1.0000 |
| mBART: CLRR+LSR − LSR-only | BLEU | +0.7606 | [-0.4086, 1.9237] | 0.0888 | 1.0000 |
| mBART: CLRR+LSR − LSR-only | chrF++ | +0.5071 | [-0.2316, 1.2522] | 0.0798 | 1.0000 |
| mBART: CLRR+LSR − Strong LoRA A | BLEU | +10.1601 | [8.2860, 12.1476] | 0.0001 | 0.0042 |
| mBART: CLRR+LSR − Strong LoRA A | chrF++ | +10.4505 | [5.0353, 18.8632] | 0.0002 | 0.0052 |
| mBART: CLRR+LSR − Strong LoRA B | BLEU | +6.5830 | [4.9815, 8.2314] | 0.0001 | 0.0042 |
| mBART: CLRR+LSR − Strong LoRA B | chrF++ | +8.7531 | [3.4417, 15.6069] | 0.0004 | 0.0096 |
| NLLB: CLRR+LSR − Baseline | BLEU | +0.7818 | [-0.2288, 1.7849] | 0.0573 | 1.0000 |
| NLLB: CLRR+LSR − Baseline | chrF++ | +0.4763 | [-0.0682, 1.0552] | 0.0466 | 0.8853 |
| NLLB: CLRR+LSR − Strong LoRA A | BLEU | +2.8195 | [1.4847, 4.1169] | 0.0001 | 0.0042 |
| NLLB: CLRR+LSR − Strong LoRA A | chrF++ | +1.4148 | [0.7611, 2.1693] | 0.0003 | 0.0075 |
| NLLB: CLRR+LSR − Strong LoRA B | BLEU | +4.4796 | [3.1097, 5.8786] | 0.0001 | 0.0042 |
| NLLB: CLRR+LSR − Strong LoRA B | chrF++ | +2.5142 | [1.8153, 3.3643] | 0.0001 | 0.0042 |
| mT5: CLRR-Dec+LSR − Baseline | BLEU | +2.0428 | [1.3530, 2.7629] | 0.0001 | 0.0042 |
| mT5: CLRR-Dec+LSR − Baseline | chrF++ | +1.3803 | [0.6403, 1.7663] | 0.0001 | 0.0042 |
| mT5: CLRR+LSR − Baseline | BLEU | +1.8074 | [1.1826, 2.4746] | 0.0001 | 0.0042 |
| mT5: CLRR+LSR − Baseline | chrF++ | +1.2296 | [0.5181, 1.5591] | 0.0001 | 0.0042 |
| Ash mBART: CLRR+LSR (patience 10, fixed epoch 20) − Baseline (patience 4) | BLEU | +0.0864 | [-0.3886, 0.5811] | 0.2619 | 1.0000 |
| Ash mBART: CLRR+LSR (patience 10, fixed epoch 20) − Baseline (patience 4) | chrF++ | +0.1051 | [-0.3802, 0.5653] | 0.2372 | 1.0000 |

- mBART full − baseline và full − LSR-only: CI cắt0, Holm p=1.0000; **chưa chứng minh superiority**.
- NLLB full − baseline: chrF++ raw p≈0.0466 nhưng Holm≈0.8853; không còn significance trong family đã khai báo.
- Full − Strong LoRA: các Holm p<0.01; tốt cho rebuttal giới hạn cấu hình đã thử. LR/HPO/trainable-subset khác nhau nên không suy thành capacity ceiling.
- mT5 full/decoder − baseline tốt; decoder − Middle không có ý nghĩa và chrF++ thấp hơn.
- Ash mọi đối sánh không có ý nghĩa; patience10/epoch20 khác protocol baseline, là post-hoc.
- Không dùng test outcomes để chọn lại model, LR, patience hoặc metric chính.

## chrF++ Mandarin: một câu gây độ nhạy lớn

Reference corpus có **3 word bigrams**, trong3/575 câu. mBART baseline match0; full và LSR-only match1 tại **index170 (0-based)**. Internal punctuation processing tạo word units dù phần lớn Mandarin không có whitespace; đây là metric chuẩn, không phải lỗi công thức.

- Full raw chrF++ gain **+5.0277**.
- Character-only chrF(w=0): **18.2288→18.9195**, gain **+0.6907**.
- Leave-one-out index170 chỉ là diagnostic: raw chrF++ **13.9655→14.3499**, gain **+0.3844**.
- Bảng chính vẫn giữ575 câu; không đổi sang TokenizerZh hay loại câu để tìm significance.
- ma-regex gain +8.8125 cũng không chứng minh hình thái, vì proxy regex và cùng tính nhạy metric.

Câu có provenance:
- Source: `Acaw aka han ko niradoman no mako hana!`
- Reference: `不要在我剛挑的水中舀水好不好!`
- Baseline: `不要在我剛挑的水中丟花!`
- CLRR+LSR: trùng reference.

Dịch tốt hơn ở câu này là quan sát hợp lệ, không phải kiểm định bảo toàn affix. [Components](../../../outputs_rebuttal/metric_audit_20261002/full/chrf_component_diagnostics.json).

## Diagnostics và các claims bị rút

| Hạng mục | Phát hiện | Trạng thái hiện hành |
| --- | --- | --- |
| Cosine mT5 | mean tầng cuối0.9519/0.8745; bảng cũ0.6277/0.6033 không khớp artifact | Aggregation verified, chưa xác minh checkpoint bằng forward mới |
| No-sg gradient | 18 norms đều0, layer maps rỗng; callback đọc sau zero_grad | Trace không hợp lệ; rút >1000/1200 và with-sg18.4±4.2/15–35 |
| Stop-grad đường đi | block cache bắt đầu L1, không embedding; L1→L12 distance11 | Abstract inter-block paths144, không233; không theorem ổn định tuyệt đối |
| SVD anti-collapse | Không có raw/full spectra/report gốc | Rút Effective Rank387.67/411.58 và kết luận chống collapse khỏi evidence |
| Affix probe F1 | Không có CSV 13 layers × 3 tasks | Rút bảng và kết luận distributed encoding khỏi evidence |
| Sensitivity sweep | Không có trial metadata/output; điểm default khớp test | Không gọi validation sweep hoặc optimum đã chứng minh |
| Latency/memory overhead | Không có measurements/repeats | Rút ≤0.4% latency, ≤2.2% train overhead |
| Qualitative examples | 5 câu research và 3 câu paper không khớp source test | Không dùng như saved real model predictions |
| mT5 LSR-only | Doc4.5422/5.0118 khác metadata4.9740/5.9426, không predictions | Không chọn một bộ số để chứng nhận |
| no-sg test BLEU/chrF++ | Chỉ summary, không prediction evidence | Không certified raw scores |

SVD centered575×1024 có rank tối đa574; script chỉ lưu50 singular values đầu không đủ tái tính full effective rank. Loader diagnostic có nhánh strict=False/fallback và mBART baseline/CLRR trỏ cùng checkpoint; cần audit/fix riêng trước forward. Không coi source diagnostics đã được chứng nhận toàn bộ.

## Dataset/parameter/runtime corrections

Amis train/val/test:4600/576/575. Whitespace source tokens:31.136/3.947/3.863; BMP Han target characters:45.438/5.972/5.643. Không phải các số42.180/5.314/5.286 và49.812/6.290/6.244 cũ. Không có empty source/target hay exact duplicate pairs, exact pair overlaps giữa splits=0.

Ash train/val/test:3883/881/1003. Train có23 duplicate pairs; cross-split exact pair overlap=0. Không dedup sau training; không suy thành near-duplicate/source-only leakage đã loại trừ.

mBART total610.879.488; BitFit train335.872 (0.0550%), added0; Narrow LoRA train/added1.179.648. NLLB total615.073.792; BitFit train333.824 (0.0543%), **không131.072/0.0213%**. CLRR thêm0 nhưng full-tuning.

NLLB StrongB train271.005.696:
- **43.4496%** / active adapted623.724.544 weights, gồm base+8.650.752 adapter.
- **30.5848%** / registered886.079.488 parameters, gồm frozen originals retained by modules_to_save.
Hai tỷ lệ phản ánh mẫu số khác nhau; không đổi metadata gốc để che khác biệt. Adapter architectural overhead8.650.752; frozen-copy storage overhead là đại lượng riêng.

Stored train-runtime totals: mBART StrongA/B93.4888/95.1309min; NLLB32.9175/40.4702min; mBART Middle17.6412min (**không34.1**). Full mBART31.8/32.5 và mT5 table12.4min không có report train-runtime chứng nhận. Không so run totals làm overhead benchmark với epochs/settings khác nhau.

## Source/report đã sửa và giới hạn

Shared raw-reference validation/test callback được áp dụng cho 5 entrypoints, rebind đúng split trước predict, reject count mismatch; CSV loaders giữ nguyên literal NA và string values. Regression tests **4/4**, import/protocol checks **5/5**, py_compile **13 files** pass. Gradient tracker dùng Trainer-logged global pre-clip norm; chưa rerun.

Ba runners comparative/PEFT/Strong LoRA mBART bỏ hardcoded scores và re-score outputs thay vì chép metrics lịch sử. Historical reference lookup chỉ chấp nhận test.csv đúng SHA-256; canonical CSV được phép sync trong repo. Các run totals/parameter counts không có evidence được để trống, không dùng 0 trainable để đại diện “zero added”.

Đồng bộ METRICS/EXPERIMENTS/RESEARCH_INSIGHTS/implementation_plan/README/PEFT reference và **hai clrr_main.tex**. Derived score CSVs sửa theo canonical36outputs; raw historical metrics/predictions và upload receipts giữ nguyên. Historical archive không phải nguồn score hiện hành. Snapshot trước chỉnh và manifest nằm trong `outputs_rebuttal/metric_audit_20261002/full/files_before/`.

Checkpoint lịch sử có thể chọn theo decoded validation labels; audit **không** tái chọn checkpoint. LaTeX chưa render PDF do không có compiler local; kiểm tra cấu trúc/reference riêng, không claim compile thành công.

## Kiểm tra suy luận nghiên cứu

Đã kiểm tra11 rủi ro suy luận trong phạm vi audit: Simpson/aggregation (corpus metrics không cộng tuyến tính); ecological inference (regex không gold); selection (epoch20 post-hoc); collider (chưa thiết kế causal conditioning); base-rate (prefix overlap và target bigram rất thưa); regression-to-mean (không kiểm định qua seed); survivorship (ByT5/Ash âm tính giữ lại); look-elsewhere (Holm42); forking paths (protocol raw đã khai báo, không metric shopping); correlation→causation (cosine/gains không chứng minh mechanism); reverse causation (chưa identification). Đây là rà soát giới hạn, không xác nhận mọi rủi ro đã được loại trừ.

**Quyết định tiếp theo cần dựa trên evidence đã sửa:** claim superiority so với full-tuning/cơ chế chưa đủ; claim hơn Strong LoRA có hỗ trợ trong cấu hình đã thử. Không mở thêm training trong bước xác minh này.

## Giao th?c c?a ri?ng b?o c?o m?i

Reference l? target g?c trong test.csv; ??ng count/order, ch? strip whitespace hai ??u; kh?ng decode reference v? kh?ng b? prediction r?ng. SacreBLEU 2.6.0, corpus BLEU case-sensitive, smooth exp, effective_order=False; tokenizer zh cho Mandarin, 13a cho Spanish. chrF++ char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False; kh?ng external tokenization. B?ng hi?n th?4dec; CSV audit gi? full precision.

## To?n b? 36 b? ?i?m m?i

| Dataset | Backbone | Method / checkpoint | N | BLEU | chrF++ chu?n |
| --- | --- | --- | ---: | ---: | ---: |
| amis_mandarin | mBART | Baseline | 575 | 19.6144 | 14.0547 |
| amis_mandarin | mBART | CLRR+LSR | 575 | 20.3896 | 19.0824 |
| amis_mandarin | mBART | CLRR-only | 575 | 18.4697 | 13.2443 |
| amis_mandarin | mBART | LSR-only | 575 | 19.6291 | 18.5753 |
| amis_mandarin | mBART | CLRR-Dec+LSR | 575 | 19.9188 | 18.6999 |
| amis_mandarin | mBART | BitFit | 575 | 0.3463 | 2.2552 |
| amis_mandarin | mBART | Narrow LoRA | 575 | 3.2365 | 4.4213 |
| amis_mandarin | mBART | Strong LoRA A | 575 | 10.2296 | 8.6319 |
| amis_mandarin | mBART | Strong LoRA B | 575 | 13.8066 | 10.3293 |
| amis_mandarin | NLLB | Strong LoRA A | 575 | 11.4625 | 9.5004 |
| amis_mandarin | NLLB | Strong LoRA B | 575 | 9.8024 | 8.4010 |
| amis_mandarin | mBART | Middle-Layer Alignment | 575 | 19.2893 | 18.4378 |
| amis_mandarin | NLLB | Baseline | 575 | 13.5001 | 10.4389 |
| amis_mandarin | NLLB | CLRR+LSR | 575 | 14.2820 | 10.9152 |
| amis_mandarin | NLLB | CLRR-Dec+LSR | 575 | 13.4194 | 10.4559 |
| amis_mandarin | NLLB | Middle-Layer Alignment | 575 | 13.9648 | 10.6596 |
| amis_mandarin | NLLB | BitFit | 575 | 1.0750 | 2.9643 |
| amis_mandarin | NLLB | Narrow LoRA | 575 | 3.5383 | 4.9407 |
| amis_mandarin | mT5 | Baseline | 575 | 2.7887 | 3.8129 |
| amis_mandarin | mT5 | CLRR-only | 575 | 4.4393 | 5.1761 |
| amis_mandarin | mT5 | CLRR+LSR | 575 | 4.5961 | 5.0425 |
| amis_mandarin | mT5 | CLRR-Dec+LSR | 575 | 4.8315 | 5.1933 |
| amis_mandarin | mT5 | CLRR-Both+LSR | 575 | 3.4459 | 4.5753 |
| amis_mandarin | mT5 | Middle-Layer Alignment | 575 | 4.6625 | 5.2103 |
| ashaninka_spanish | mBART | Baseline (patience 4) | 1003 | 4.0085 | 20.8536 |
| ashaninka_spanish | mBART | CLRR+LSR (patience 4) | 1003 | 3.9671 | 20.6990 |
| ashaninka_spanish | NLLB | Baseline (epoch 20) | 1003 | 3.8594 | 20.4176 |
| ashaninka_spanish | mBART | CLRR+LSR (patience 10, selected epoch 14) | 1003 | 3.7970 | 20.9753 |
| ashaninka_spanish | mBART | CLRR+LSR (patience 10, fixed epoch 20) | 1003 | 4.0949 | 20.9588 |
| amis_mandarin | ByT5 | Baseline | 575 | 7.5837 | 8.3057 |
| amis_mandarin | ByT5 | CLRR+LSR | 575 | 7.3108 | 8.0956 |
| amis_mandarin | ByT5 | CLRR-Dec+LSR | 575 | 7.4394 | 8.0630 |
| amis_mandarin | ByT5 | LayerSkip | 575 | 2.5533 | 4.8568 |
| amis_mandarin | ByT5 | Middle-Layer Alignment | 575 | 7.5660 | 8.3414 |
| amis_mandarin | mT5 | LayerSkip | 575 | 3.9292 | 4.5845 |
| amis_mandarin | mBART | LayerSkip | 575 | 18.7506 | 18.1079 |

## ??y ?? 42 ki?m ??nh m?i

| Dataset / backbone | Challenger ? reference | Metric | ? | 95% CI | p raw | p Holm |
| --- | --- | --- | ---: | --- | ---: | ---: |
| amis_mandarin / mBART | CLRR+LSR ? Baseline | BLEU | +0.7752 | [-0.4388, 2.0441] | 0.0954 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR ? Baseline | chrF++ | +5.0277 | [-0.1872, 13.2525] | 0.1214 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR ? LSR-only | BLEU | +0.7606 | [-0.4086, 1.9237] | 0.0888 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR ? LSR-only | chrF++ | +0.5071 | [-0.2316, 1.2522] | 0.0798 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR ? CLRR-only | BLEU | +1.9199 | [0.8183, 3.0558] | 0.0006 | 0.0138 |
| amis_mandarin / mBART | CLRR+LSR ? CLRR-only | chrF++ | +5.8381 | [0.7650, 14.0341] | 0.1000 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR ? Middle-Layer Alignment | BLEU | +1.1004 | [-0.0643, 2.2602] | 0.0292 | 0.6131 |
| amis_mandarin / mBART | CLRR+LSR ? Middle-Layer Alignment | chrF++ | +0.6446 | [-0.0528, 1.3661] | 0.0333 | 0.6659 |
| amis_mandarin / mBART | CLRR+LSR ? CLRR-Dec+LSR | BLEU | +0.4709 | [-0.5804, 1.5743] | 0.1480 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR ? CLRR-Dec+LSR | chrF++ | +0.3825 | [-0.2401, 1.0502] | 0.1040 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR ? BitFit | BLEU | +20.0434 | [18.0696, 22.0865] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR ? BitFit | chrF++ | +16.8272 | [11.3416, 25.0158] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR ? Narrow LoRA | BLEU | +17.1532 | [15.1766, 19.2210] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR ? Narrow LoRA | chrF++ | +14.6610 | [9.2155, 23.1350] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR ? Strong LoRA A | BLEU | +10.1601 | [8.2860, 12.1476] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR ? Strong LoRA A | chrF++ | +10.4505 | [5.0353, 18.8632] | 0.0002 | 0.0052 |
| amis_mandarin / mBART | CLRR+LSR ? Strong LoRA B | BLEU | +6.5830 | [4.9815, 8.2314] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR ? Strong LoRA B | chrF++ | +8.7531 | [3.4417, 15.6069] | 0.0004 | 0.0096 |
| amis_mandarin / NLLB | CLRR+LSR ? Baseline | BLEU | +0.7818 | [-0.2288, 1.7849] | 0.0573 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR ? Baseline | chrF++ | +0.4763 | [-0.0682, 1.0552] | 0.0466 | 0.8853 |
| amis_mandarin / NLLB | CLRR+LSR ? Middle-Layer Alignment | BLEU | +0.3172 | [-0.7610, 1.3798] | 0.1909 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR ? Middle-Layer Alignment | chrF++ | +0.2556 | [-0.3116, 0.8397] | 0.1408 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR ? Narrow LoRA | BLEU | +10.7437 | [9.2558, 12.2837] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR ? Narrow LoRA | chrF++ | +5.9745 | [5.2485, 7.2458] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR ? Strong LoRA A | BLEU | +2.8195 | [1.4847, 4.1169] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR ? Strong LoRA A | chrF++ | +1.4148 | [0.7611, 2.1693] | 0.0003 | 0.0075 |
| amis_mandarin / NLLB | CLRR+LSR ? Strong LoRA B | BLEU | +4.4796 | [3.1097, 5.8786] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR ? Strong LoRA B | chrF++ | +2.5142 | [1.8153, 3.3643] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR ? Baseline | BLEU | +2.0428 | [1.3530, 2.7629] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR ? Baseline | chrF++ | +1.3803 | [0.6403, 1.7663] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR ? Middle-Layer Alignment | BLEU | +0.1689 | [-0.3511, 0.7041] | 0.1813 | 1.0000 |
| amis_mandarin / mT5 | CLRR-Dec+LSR ? Middle-Layer Alignment | chrF++ | -0.0171 | [-0.2487, 0.2175] | 0.3433 | 1.0000 |
| amis_mandarin / mT5 | CLRR+LSR ? Baseline | BLEU | +1.8074 | [1.1826, 2.4746] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR+LSR ? Baseline | chrF++ | +1.2296 | [0.5181, 1.5591] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Both+LSR ? CLRR+LSR | BLEU | -1.1501 | [-1.7066, -0.6727] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Both+LSR ? CLRR+LSR | chrF++ | -0.4672 | [-0.7534, -0.2068] | 0.0013 | 0.0286 |
| ashaninka_spanish / mBART | CLRR+LSR (patience 4) ? Baseline (patience 4) | BLEU | -0.0414 | [-0.5238, 0.4389] | 0.3361 | 1.0000 |
| ashaninka_spanish / mBART | CLRR+LSR (patience 4) ? Baseline (patience 4) | chrF++ | -0.1546 | [-0.6556, 0.3387] | 0.1830 | 1.0000 |
| ashaninka_spanish / mBART | CLRR+LSR (patience 10, selected epoch 14) ? Baseline (patience 4) | BLEU | -0.2116 | [-0.7274, 0.2826] | 0.1501 | 1.0000 |
| ashaninka_spanish / mBART | CLRR+LSR (patience 10, selected epoch 14) ? Baseline (patience 4) | chrF++ | +0.1217 | [-0.3667, 0.5960] | 0.2204 | 1.0000 |
| ashaninka_spanish / mBART | CLRR+LSR (patience 10, fixed epoch 20) ? Baseline (patience 4) | BLEU | +0.0864 | [-0.3886, 0.5811] | 0.2619 | 1.0000 |
| ashaninka_spanish / mBART | CLRR+LSR (patience 10, fixed epoch 20) ? Baseline (patience 4) | chrF++ | +0.1051 | [-0.3802, 0.5653] | 0.2372 | 1.0000 |

## T?p con regex ? ?i?m m?i mBART

| T?p con | N | Baseline BLEU | CLRR+LSR BLEU | Baseline chrF++ | CLRR+LSR chrF++ | ? chrF++ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 575 | 19.6144 | 20.3896 | 14.0547 | 19.0824 | +5.0277 |
| mi | 195 | 17.8639 | 18.4293 | 14.9140 | 15.3779 | +0.4638 |
| ma | 246 | 18.2216 | 20.6057 | 13.2853 | 22.0979 | +8.8125 |
| pa | 127 | 19.1066 | 19.4780 | 15.1684 | 15.3053 | +0.1368 |
| Root | 161 | 21.1423 | 20.7916 | 15.1877 | 14.7532 | -0.4345 |

## Cosine mT5 ?? ??i chi?u aggregation

B?ng ??y ?? theo artifact [cosine_by_layer.csv](../../../results/analysis/cosine_by_layer.csv); mean t? 9.200 h?ng ?? t?i l?p.
