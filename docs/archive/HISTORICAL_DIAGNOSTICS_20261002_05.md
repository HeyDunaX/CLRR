# Historical diagnostics — scoped evidence

Historical checkpoints and probes only. These results do not substitute for corrected training controls or establish anti-collapse, morphological preservation, causal interaction, or robust superiority. Warmup v2 failed its historical validation gates. Asháninka/LayerSkip are excluded from the current paper scope; any mentions in preserved audit records are historical. The old insight snapshot is retained for unique diagnostic provenance, not as current interpretation.

Current interpretation: [RESEARCH_INSIGHTS](../research/RESEARCH_INSIGHTS.md). Execution status: [current handoff](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md).

## Contents
- [RESULTS_VERIFICATION](#results-verification)
- [EASY_DIAGNOSTICS_20261002](#easy-diagnostics-20261002)
- [CHECKPOINT_DIAGNOSTICS_20261002](#checkpoint-diagnostics-20261002)
- [MATCHED_INFERENCE_RESULTS_20261003](#matched-inference-results-20261003)
- [AMIS_WARMUP_V2_RESULTS](#amis-warmup-v2-results)
- [RESEARCH_INSIGHTS](#research-insights)
- [FIXED_EXPERIMENTS_IF_METHOD_CHANGES](#fixed-experiments-if-method-changes)

---

<a id="results-verification"></a>

## RESULTS_VERIFICATION

Original source: `docs/archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#results-verification`. Status: historical record; the summary at the top and the current handoff take precedence.

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

Nguồn chính: [METRICS.md](README.md), [36 scores](../../results/metrics_standardized_scores.csv), [translation audit](../../outputs_rebuttal/metric_audit_20261002/full/translation_audit.json), [42 tests](../../outputs_rebuttal/metric_audit_20261002/full/paired_bootstrap_standardized.csv), [bootstrap protocol](../../outputs_rebuttal/metric_audit_20261002/full/bootstrap_protocol.json), [API checks](../../outputs_rebuttal/metric_audit_20261002/full/bootstrap_api_checks.json).

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

Dịch tốt hơn ở câu này là quan sát hợp lệ, không phải kiểm định bảo toàn affix. [Components](../../outputs_rebuttal/metric_audit_20261002/full/chrf_component_diagnostics.json).

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

B?ng ??y ?? theo artifact [cosine_by_layer.csv](../../results/analysis/cosine_by_layer.csv); mean t? 9.200 h?ng ?? t?i l?p.


---

<a id="easy-diagnostics-20261002"></a>

## EASY_DIAGNOSTICS_20261002

Original source: `docs/archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#easy-diagnostics-20261002`. Status: historical record; the summary at the top and the current handoff take precedence.

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

Artifact: [correctness.json](../../outputs_rebuttal/easy_diagnostics_20261002/correctness.json).

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

Artifacts: [character_chrf_bootstrap.csv](../../outputs_rebuttal/easy_diagnostics_20261002/character_chrf_bootstrap.csv), [robustness_protocol.json](../../outputs_rebuttal/easy_diagnostics_20261002/robustness_protocol.json).

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

Artifacts: [influence_summary.csv](../../outputs_rebuttal/easy_diagnostics_20261002/influence_summary.csv), [influence_protocol.json](../../outputs_rebuttal/easy_diagnostics_20261002/influence_protocol.json). Có bảng đầy đủ 4,600 dòng: 4 comparisons × 2 metrics × 575 lần bỏ một câu.

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

**Chưa khởi động train trong đợt này.** Kế hoạch đầy đủ: [NEXT_STEPS_AUDITED_20261002.md](README.md).

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

- [verification_receipt.json](../../outputs_rebuttal/easy_diagnostics_20261002/verification_receipt.json): assertions và kiểm tra SHA-256 19 file cũ.
- [output_summary.csv](../../outputs_rebuttal/easy_diagnostics_20261002/output_summary.csv): exact matches, empty predictions, character length ratios; diagnostics phụ, không phải morphology metrics.
- Input chất lượng chính: [all_verified_translation_scores.csv](../../outputs_rebuttal/metric_audit_20261002/full/all_verified_translation_scores.csv), 36 runs, và raw test.csv theo SHA-256 được lưu.


---

<a id="checkpoint-diagnostics-20261002"></a>

## CHECKPOINT_DIAGNOSTICS_20261002

Original source: `docs/archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#checkpoint-diagnostics-20261002`. Status: historical record; the summary at the top and the current handoff take precedence.

# Chẩn đoán checkpoint CLRR — đo 02/10/2026, tổng hợp 03/10/2026

## Material Passport

- Trạng thái: hoàn tất đo và xác minh artifacts cho 12 checkpoint đã train; 0 optimizer steps.
- Nguồn weights: `FiveC/amis-rewire-checkpoints`, revision `63b0dbc4fab41a669ccb36ed2f22894868634b8a`. Archive, kích thước, SHA-256 và prediction đối chiếu của từng model nằm trong `protocol.json`.
- Máy đo: Colab A100 40 GB, shape Standard, không yêu cầu high-mem; Python conda `clrr`, Torch 2.6.0+cu124, Transformers 4.57.6. Phiên không còn hoạt động khi kiểm tra ngày 03/10.
- Dữ liệu: public GitHub `HeyDunaX/CLRR`, commit `db4c2322e11725db3aa6bc8cf4c4c812e11bf8e9`; các hàng CSV được kiểm tra trùng dữ liệu local. Không upload CSV local.
- Artifacts: [thư mục kết quả](../../outputs_rebuttal/checkpoint_diagnostics_20261002), [xác minh local](../../outputs_rebuttal/checkpoint_diagnostics_20261002/local_verification.json), [bảng tổng hợp](../../outputs_rebuttal/checkpoint_diagnostics_20261002/mechanism_summary.csv).
- Đã kiểm tra SHA-256 của archive 709,830,219 bytes và 113 file cloud; strict load weights, 96 batch gradient, chỉ số mẫu validation giống nhau giữa các method cùng dataset. 19 file lịch sử trong restoration manifest giữ nguyên SHA-256.
- Phạm vi: chẩn đoán quan sát tại checkpoint đã chọn; không phải đo tiến trình train, không phải can thiệp nhân quả, không phải chứng minh đa seed. Không thay BLEU/chrF hay sửa EXPERIMENTS/RESEARCH_INSIGHTS/TeX cũ.

## 1. Giao thức đo

**Residual:** hook trước và sau CLRR; kiểm tra `post = pre + 0.1 × skip`, distance 2, encoder. RMS tính trên vị trí hợp lệ và tất cả chiều hidden; ratio là RMS residual/RMS output lớp hiện tại trước rewiring. Bảng dưới là trung bình và cực đại qua các ví dụ nguồn và các lớp nhận skip, từ lớp 3 trở đi.

**Collapse:** toàn bộ 575 câu test Amis và 1,003 câu test Asháninka, nguồn và đích; eval, FP32, TF32 tắt. Mean pooling theo attention mask, gồm special tokens hợp lệ. SVD đầy đủ trên biểu diễn đã center, tính float64. Effective rank là `exp(H(s/sum(s)))`; khác effective rank của eigenvalues covariance. Giới hạn rank là `min(n−1,d)`. Raw mean pairwise cosine đo hướng chung; cosine cao không tự chứng minh collapse.

**Gradient:** 32 câu validation cố định, RNG 20261002, 8 batch × 4; train mode với dropout và seed 1000+batch_id, target stop-gradient giữ dropout train mode. BF16 autocast, parameters FP32, gradient chưa scale/clip; không optimizer. Dùng tensor CE gốc và graph auxiliary gốc (`total_loss − native_CE`) của wrapper cho Full/LSR-only. Ratio là trung bình ratio norm theo batch, không phải ratio của loss scalar. Nhóm encoder loại shared embeddings; các nhóm parameters không chồng lặp.

Baseline/CLRR-only không dùng LSR khi train: gradient auxiliary của chúng trong artifacts là **counterfactual λ=0.1**, không được diễn giải là lực tối ưu lịch sử. Không so absolute gradient norm giữa kiến trúc như một thước đo chất lượng.

## 2. Effective rank và mức tập trung biểu diễn

| Dataset/model | Method | Source e-rank | Rank limit | Target e-rank | Source pairwise cosine |
|---|---|---:|---:|---:|---:|
| Amis mBART | Baseline | 390.700 | 574 | 390.675 | 0.432 |
| Amis mBART | Full | 387.850 | 574 | 383.852 | 0.717 |
| Amis mBART | LSR-only | 388.194 | 574 | 381.721 | 0.707 |
| Amis NLLB | Baseline | 383.630 | 574 | 423.467 | 0.696 |
| Amis NLLB | Full | 387.674 | 574 | 411.582 | 0.646 |
| Amis mT5 | Baseline | 197.389 | 512 | 259.242 | 0.855 |
| Amis mT5 | Full | 222.676 | 512 | 256.212 | 0.836 |
| Amis mT5 | CLRR-only | 216.523 | 512 | 258.193 | 0.765 |
| Amis ByT5 | Baseline | 261.717 | 574 | 309.060 | 0.613 |
| Amis ByT5 | Full | 259.642 | 574 | 291.086 | 0.902 |
| Asháninka mBART | Baseline | 535.223 | 1002 | 600.183 | 0.621 |
| Asháninka mBART | Full, patience 4 | 535.568 | 1002 | 576.261 | 0.690 |

**Tốt:** các checkpoint Full giữ phổ nhiều chiều; không có dấu hiệu collapse về một hướng theo phép đo pooled này. NLLB Full tái lập chính xác hai số SVD cũ khi làm tròn: **387.67/411.58**. mT5 Full source rank tăng khoảng 12.8% so với baseline, CLRR-only cũng tăng: bằng chứng cơ chế đáng theo dõi cùng kết quả dịch mT5 đã audit.

**Chưa tốt:** Full không tăng effective rank nhất quán. mBART Full gần LSR-only, thấp hơn baseline; target rank Full thấp hơn baseline trên cả năm cặp dataset/backbone. ByT5 Full source cosine tăng mạnh 0.613→0.902, source rank hơi giảm. Asháninka source rank gần như không đổi. Không dùng các bảng này để claim “CLRR universally improves diversity”, “stop-gradient guarantees anti-collapse”, hay “preserves semantics”.

mT5 rank tăng là tương quan với kết quả dịch, chưa chứng minh tăng rank gây tăng chất lượng. So sánh target/source rank phải trong cùng backbone/protocol; không so rank thô giữa các hidden dimensions.

## 3. Residual norms và cân bằng CE–LSR

| Dataset/model, method | Residual/current mean | Max | Weighted LSR/CE gradient, toàn model | Encoder excl. shared | Mean gradient cosine | Batch cosine âm /8 |
|---|---:|---:|---:|---:|---:|---:|
| Amis mBART Full | 5.911% | 8.626% | 0.167% | 0.310% | 0.057 | 0 |
| Amis mBART LSR-only | 0% | 0% | 0.237% | 0.412% | 0.046 | 1 |
| Amis NLLB Full | 6.632% | 8.764% | 0.487% | 0.737% | 0.149 | 0 |
| Amis mT5 Full | 6.213% | 9.838% | 0.794% | 1.755% | 0.084 | 3 |
| Amis mT5 CLRR-only | 6.209% | 9.833% | inactive | inactive | counterfactual | counterfactual |
| Amis ByT5 Full | 4.338% | 9.305% | 1.184% | 1.795% | −0.124 | 5 |
| Asháninka mBART Full, p4 | 5.391% | 8.033% | 0.223% | 0.459% | 0.068 | 0 |

**Tốt:** trên các ví dụ/lớp đo, residual Full trung bình khoảng 4.3–6.6% của current output, cực đại dưới 10%; không thấy residual lấn át hoặc tăng vọt. Đây là mô tả checkpoint, không phải bảo đảm ổn định toàn bộ quá trình train hoặc mọi input.

**Cần điều tra:** weighted LSR gradient nhỏ so với CE tại checkpoint cuối được chọn. Điều này giúp định lượng λ=0.1 nhưng chưa đủ để tăng λ: không biết lực LSR ở giai đoạn đầu. ByT5 có conflict trên 5/8 batch, mean cosine âm; phù hợp với giả thuyết cần xem lại tương tác LSR/kiến trúc ở ByT5, chưa đủ để khẳng định đây là nguyên nhân giảm chất lượng. mT5 có 3/8 batch conflict dù mean cosine dương và chất lượng dịch tốt hơn; tránh claim LSR luôn đồng thuận CE.

Đo này thay thế việc dùng trace callback gradient bằng 0 sau `zero_grad` để suy luận. Nó không xác thực lại các trace lịch sử đó. Trong quá trình phát triển phép đo, CE tái dựng thủ công từ returned logits BF16 khác native CE tối đa 0.002555; **cả 12 checkpoint đã được đo lại bằng native objective**. Bản đầu giữ riêng `gradients_initial_reconstruction.json`, không dùng cho bảng cuối. Sai khác scalar auxiliary tái dựng/gốc tối đa khoảng 2.35e−7; gradient chính lấy từ graph gốc.

## 4. Đối chiếu generation: còn giới hạn tái lập

8 câu sentinel cố định/model; beams 4, max length theo manifest; so text với prediction lịch sử bất biến. Đây là kiểm tra mẫu nhỏ, không phải full-test re-evaluation.

| Checkpoint | BF16 khớp /8 | FP32 khớp /8 |
|---|---:|---:|
| mBART baseline | 8 | 8 |
| mBART Full | 8 | 8 |
| mBART LSR-only | 6 | 7 |
| NLLB baseline | 6 | 6 |
| NLLB Full | 8 | 8 |
| mT5 baseline | 6 | 8 |
| mT5 Full | 7 | 8 |
| mT5 CLRR-only | 7 | 8 |
| ByT5 baseline | 8 | 8 |
| ByT5 Full | 7 | 8 |
| Asháninka mBART baseline | 8 | 4 |
| Asháninka mBART Full p4 | 4 | 7 |

Precision thay đổi có thể đổi beam search: trực tiếp thấy ở mT5/ByT5/Asháninka. Tuy nhiên, **FP32 không giải quyết hết mismatch** của mBART LSR-only, NLLB baseline và Asháninka Full. Không kết luận “đúng hoàn toàn predictions lịch sử” hoặc “sai checkpoint” chỉ từ sentinel. Chi tiết câu thay đổi có trong `sentinel_comparison.csv`; archive SHA/strict loading/tokenizer vocab đã kiểm tra.

Code review cho thấy wrapper `.generate` gọi base model, có khả năng bypass autocast bọc root forward của Accelerate, trong khi baseline đi qua root. Đây là **giả thuyết về pipeline**, chưa xác minh precision thực thi của từng historical run. Trước thí nghiệm mới phải đặt cùng generation precision rõ ràng cho tất cả method, cùng tokenizer/language codes/beams/max length và lưu environment. Không thay điểm lịch sử từ 8 câu sentinel này. Muốn sửa điểm checkpoint cần chạy lại toàn test cùng protocol rồi lưu phiên bản kết quả mới.

## 5. Checkpoint loading và phần chưa đo

- Wrapper ZIP thiếu config được phục hồi từ **paired trained baseline archive**, ghi source/member/SHA trong provenance; strict loading không có missing/unexpected/mismatched weights.
- NLLB tokenizer cũ lưu `extra_special_tokens` dạng list: chuyển cách truyền tham số cho Transformers 4.57.6; kiểm tra toàn bộ vocab và token IDs không đổi. Không áp dụng thay regex tokenizer âm thầm.
- Giữ saved language codes: Amis mBART `tl_XX→zh_CN`, NLLB `zho_Hant→zho_Hant`, Asháninka mBART `es_XX→es_XX`. Đây là cấu hình lịch sử để tái đo, không phải language codes cho Turkish.
- **mBART CLRR-only best checkpoint 504 chưa có trong revision HF đã kiểm tra** (chỉ tìm được đến 432). Không dùng checkpoint 432 thay thế. Vì vậy chưa đủ 4 thành phần mBART cho phân tích cơ chế.
- Asháninka Full trong báo cáo là checkpoint patience 4 theo manifest; **không phải** run patience 10/epoch 20. Không gán kết quả cơ chế này cho epoch 20.
- Không có đo no-stop-gradient trên checkpoint đã train tương ứng; không suy luận nhân quả về stop-gradient. Không đo multi-seed, tốc độ/peak memory training hay Turkish trong lượt này.

## 6. Các kết luận dùng được và bước kế tiếp

1. **Bảo vệ được ở phạm vi đo:** CLRR residual có độ lớn vừa phải tại saved checkpoints; pooled representations chưa collapse về một hướng; gradient CE và weighted LSR đo được bằng objective thực tế. NLLB SVD cũ nay có artifacts để tái lập.
2. **Insight có ích:** mT5 có source effective rank tăng ở cả CLRR-only và Full; ByT5 có anisotropy tăng và CE–LSR conflict. Có thể báo cáo cơ chế phụ thuộc backbone, với giới hạn quan sát rõ ràng.
3. **Không bảo vệ được claim mạnh:** ưu thế diversity phổ quát, stop-gradient bảo đảm chống collapse, semantic preservation, CE–LSR luôn đồng thuận, hay ổn định xuyên suốt training. Residual nhỏ/không collapse không chứng minh BLEU tốt hơn.
4. **Trước Turkish:** thống nhất generation precision và ghi provenance; xử lý overlap/duplicate đã phát hiện ở dataset trong kế hoạch riêng; giữ budget 20k và dùng language codes đúng. Pilot baseline/CLRR-only/LSR-only/Full với training/selection/evaluation protocol bằng nhau, sau đó mới quyết định mở rộng seed và λ.
5. **Chưa cần sửa phương pháp ngay:** thiếu bằng chứng theo thời gian và can thiệp. Nếu cần cải thiện LSR, nên đo gradient tại các checkpoint sớm/giữa/cuối và kiểm tra ablation λ/LSR trên validation theo kế hoạch định trước; không chọn cấu hình bằng test và không tăng λ chỉ vì norm ở checkpoint cuối nhỏ.

## Tái lập và kết quả riêng

- Script đo: `scripts/revalidation/measure_checkpoint_mechanisms.py`; bản thực thi và SHA trong outputs.
- Tổng hợp local trong conda `clrr`: `python scripts/revalidation/summarize_checkpoint_mechanisms.py`.
- Raw: pooled representations từng lớp `.npz`, residual từng ví dụ `.csv`, full spectra `collapse.json`, native gradients `gradients.json`, loading provenance, BF16/FP32 sentinels, frozen requirements và logs.
- SHA archive: `66a9eb8b9ed530970476a0399a20ba66d5cf81f3457c02b2383ffaf61b9efac4`.
- Tất cả số mới ở thư mục/report riêng; không train Turkish và không sửa số cũ trong lượt này.


---

<a id="matched-inference-results-20261003"></a>

## MATCHED_INFERENCE_RESULTS_20261003

Original source: `docs/archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#matched-inference-results-20261003`. Status: historical record; the summary at the top and the current handoff take precedence.

# Kết quả đánh giá với precision thống nhất — 03/10/2026

## Material Passport

- Skill: academic-research-suite / experiment-agent, thực thi và xác minh inline.
- Status: VERIFIED cho 12 checkpoint và 5 inference routing-off; pilot Turkish đã dừng theo yêu cầu tác giả lúc08:05 ngày03/10, chưa có kết quả so sánh hoàn tất.
- GPU: A100 Standard, môi trường conda `clrr`; Torch2.6.0+cu124, Transformers4.57.6, Accelerate1.15.0, SacreBLEU2.6.0. Frozen requirements trong outputs.
- Weights: `FiveC/amis-rewire-checkpoints`, revision `63b0dbc4fab41a669ccb36ed2f22894868634b8a`, kiểm tra archive SHA và strict loading; cấu hình khôi phục từ paired baseline khi ZIP wrapper thiếu config, như lượt diagnostics trước.
- Dữ liệu: full original test, giữ thứ tự/reference/empty predictions; source/target CSV local và public có cùng parsed rows.
- Outputs mới: [scores](../../outputs_rebuttal/followup_20261003/matched_inference_scores.csv), [bootstrap](../../outputs_rebuttal/followup_20261003/matched_inference_bootstrap.csv), [protocol](../../outputs_rebuttal/followup_20261003/analysis_protocol.json), predictions và receipt trong `reevaluation/`.
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


---

<a id="amis-warmup-v2-results"></a>

## AMIS_WARMUP_V2_RESULTS

Original source: `docs/archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#amis-warmup-v2-results`. Status: historical record; the summary at the top and the current handoff take precedence.

# Amis routing warmup v2 — kết quả và quyết định

**Hoàn tất:** 04/10/2026, 21:48 (Asia/Saigon). **Quyết định: NO-GO cho schedule warmup 20%.**

Đủ 6/6 model, ba seeds 42/43/44 cho Full và CLRR-only, chạy song song trên hai A100 40GB Standard. Cả hai hàng đợi COMPLETE. Không chạy Turkish, không mở thêm grid, không đánh giá test cho ứng viên không qua gate. Các kết quả cũ được giữ nguyên; báo cáo này chỉ chứa v2.

## 1. Protocol và kiểm chứng

- mBART-50, Amis→Mandarin; α tăng 0→0.10 trong 144/720 optimizer steps, sau đó cố định. Evaluation luôn α=0.10. Full λ=0.10; CLRR-only λ=0.
- Giữ LR=5e-5, batch32×accum4, maximum20epochs, patience4 và protocol controls. BF16 training; FP32 generation, TF32 off. Model/tokenizer revision và code/data hashes được khóa trước training.
- Chọn best bằng validation **native chrF++**, beam1; số dưới đây là **validation**, không phải test hay số để thay bảng test của paper.
- Runtime isolated: torch2.6.0+cu124, transformers4.57.6, sacrebleu2.6.0. Hai máy qua smoke, replay controls và gate trước training. Namespace v1 đã invalidated; không trộn vào bảng.
- Sáu COMPLETE receipts đều có best reload PASS, trajectory probes PASS và HF best archive SHA256/size verification. Tải về 180 scientific text records; tính lại BLEU/chrF++/chrF0 độc lập từ predictions, kiểm tra hash và đúng576 hàng validation: **PASS cả sáu**. Probe JSON hashes cũng khớp receipts.
- Có24 snapshots (epoch1/4/10/actual final) và36 điều kiện probe: tách α evaluation cố định với α training cuối snapshot ở giai đoạn warmup. CLRR-only seed43 dừng epoch12 theo patience; các run còn lại train20epochs.

## 2. Số từng seed

| Nhánh warmup | Seed | BLEU | chrF++ | chrF0 | ΔBLEU so fixed cùng seed | ΔchrF0 so fixed |
|---|---:|---:|---:|---:|---:|---:|
| Full | 42 | 17.7908 | 14.9599 | 17.0269 | +0.1472 | +0.1517 |
| Full | 43 | 16.2231 | 13.8586 | 15.9132 | −0.8414 | −0.6746 |
| Full | 44 | 17.0006 | 14.3394 | 16.5026 | +0.0681 | −0.0101 |
| CLRR-only | 42 | 16.6623 | 13.9601 | 16.0884 | +0.3968 | +0.2400 |
| CLRR-only | 43 | 14.4003 | 12.5549 | 14.5335 | −2.5095 | −1.7136 |
| CLRR-only | 44 | 16.8898 | 14.2423 | 16.3599 | −0.3840 | −0.1595 |

## 3. Tổng hợp ba seeds và gate đã khóa

| Cấu hình | BLEU mean | chrF++ mean | chrF0 mean |
|---|---:|---:|---:|
| Fixed Full control | 17.2135 | 14.4975 | 16.6586 |
| **Warmup Full** | **17.0048** | **14.3859** | **16.4809** |
| Tuned LSR-only, λ=.03 | 16.9942 | 14.3170 | 16.4950 |
| Fixed CLRR-only control | 16.8164 | 14.0765 | 16.2050 |
| **Warmup CLRR-only** | **15.9842** | **13.5858** | **15.6606** |

Sample SD warmup Full: BLEU0.7839, chrF++0.5521, chrF00.5572. Warmup CLRR-only: BLEU1.3763, chrF++0.9038, chrF00.9855. Ba seeds là pilot nhỏ; không tuyên bố statistical significance từ các chênh lệch này.

Warmup Full so fixed Full: **−0.2087 BLEU, −0.1116 chrF++, −0.1777 chrF0**. So tuned LSR: **+0.0106 BLEU, +0.0689 chrF++, −0.0141 chrF0** — gần hòa, chưa có bằng chứng tăng ích ổn định.

**Cả năm điều kiện gate đều không đạt:** BLEU+0.30; chrF+++0.30; no-harm chrF0; ≥2/3 seeds cùng tăng BLEU vàchrF0 (chỉseed42 đạt); và cả ba mean vượt tunedLSR. Đây là gate nội bộ, không phải quy định ACL.

## 4. Insight từ đo cơ chế

### Warmup này chưa giải quyết vấn đề

Seed42 tăng nhẹ nhưngseed43 giảm ở cảhai nhánh; CLRR-only mean giảm0.8323BLEU. Full giảm ít hơnCLRR-only, phù hợp với khả năng LSR bù một phần suy giảm, nhưng chênh lệch hai nhánh không tự chứng minh nguyên nhân. Không chọnseed42 để quảng bá warmup, và không chuyển sang Turkish để cứu một cấu hình chưa qua gate Amis.

### Alignment tăng không đồng nghĩa dịch tốt hơn

Ở Full, paired cosine tại **encoder final LayerNorm** tăng từ khoảng0.44–0.46 (epoch1) lên0.7966–0.7977 (epoch20). CLRR-only ở endpoint thấp hơn rõ rệt. LSR có tác động tới representation alignment, nhưng Full warmup vẫn gần hòa tunedLSR về quality. Vì vậy cosine cao chỉ là bằng chứng cơ chế alignment, chưa là bằng chứng CLRR tăng chất lượng dịch.

### Chưa có dấu hiệu rank collapse tiến triển trong các snapshots đã đo

Full final-LN source covariance effective rank cuối training khoảng138.6–140.5, cao hơn đầutraining khoảng108–115; target rank cuối khoảng140.6–143.1. Không có trajectory giảm rank xuyên suốt để quy kết collapse là nguyên nhân chất lượng thấp. Đây không phải chứng minh không collapse ở mọi vị trí/token/không gian; số đo dùng pooled representations576 câu, centered spectrum và rank tối đa575.

### Không có bằng chứng LSR gradient lấn át CE trên toàn bộ parameter union

Full all-parameter weightedLSR/CE gradient norm ratio khoảng0.39–0.42% ởepoch1 và0.16–0.18% ởepoch20; cosine gradient cuối khoảng0.055–0.056. Không nên kết luận λ=.10 quá lớn hoặc CE–LSR xung đột mạnh toàn mô hình. Tác động encoder từnglayer cần đọc riêng, vì union gồm decoder và LSR không có gradientdecoder. CLRR-only được đo **LSR counterfactual λ=.10**, không phải loss thực tế đãtrain.

### Residual không phình dần rõ rệt

Full source residual/current RMS trung bình ởαeval=.10 khoảng5.76% đầutraining và5.89–5.90% cuối. Đại lượng này là trung bình chẩn đoán trên các layer córewiring; không phải bound từnglayer/câu. Chưa có bằng chứng residualnorm explosion để tự động normalize/rescale phương pháp.

### Có vấn đề generalization/selection cần xác minh trước nâng cấp

Native CE token-weighted trên **cùng32 validationexamples** xuống khoảng2.77–2.79 ởepoch4 rồi lên2.92–2.97 ởepoch20 trongFull. chrF++ vẫn có thể chọncheckpointmuộn (Full44 chọnepoch18). Đây cho thấy CE vàgenerationmetric có trajectory khác nhau; chưa đủ để đổi selectionmetric riêng choCLRR. Loggedtrainingloss không dùng để so trực tiếp vì khácaggregation vàauxiliaryloss.

Các gradient là **offline BF16 diagnostic**, TF32off, dropout có kiểm soát vàcheckpointingdisabled; không phải bản sao từngbackward củatraining. Snapshots cuối không luôn làselectedbest (Full44bestepoch18, snapshotcuốiepoch20). Không lấyendpointmechanism làm số đo trực tiếp củabest nếu khôngloadbest đểđo.

## 5. Việc tiếp theo có cơ sở

1. **Giữ fixed routing là đối chứng; loại warmup20% khỏi danh sách ứng viên ưu tiên.** NO-GO này áp dụng cho schedule vừa thử, chưa chứng minh mọi biến thểCLRR vô ích. Hiện chưa đủ incremental evidence để hứa claim vượt tunedLSR choACL.
2. Trước training mới, làm **validation-only matched inference ablation** trên đúng selectedbest Full/CLRR-only/tunedLSR của ba seeds: α0/.10, cùngweights, cùngbeam1 và diagnosticbeam4; đoquality/nativeCE vàmechanism tạibest. α0 tắt routing lúcinference là counterfactual có distribution shift; không gọi đó là mô hìnhtrainkhôngCLRR. Mục tiêu: phân biệt routing tạiinference, quá trìnhtraining và beam sensitivity.
3. Lấy **fixed-control epoch histories** nếu có và đối chiếu theoepoch/step vớiwarmup, nhất làseed43. Thiếuhistories/snapshots thì ghi rõ, không tạo bằng chứng thay thế bằng endpointweights.
4. Nếu routing tạibest làmquality giảm nhất quán nhưng alignment/rank vẫn ổn, ưu tiên **layer-selective routing** dựa vào residual/perlayer evidence. Nếu vấn đề tập trung optimization/seed sensitivity, thiết kế một thay đổi optimization có kiểm soát. Không tăng/giảmλ chỉ vì BLEUchưađẹp; đo hiện tại chưa chứng minh gradient imbalance.
5. Chỉ sau khi khóa **một giả thuyết và một thay đổi** mới chạy lại bốnnhánh matched Baseline/CLRR-only/LSR-only/Full, ≥3seeds, cùng budget và gate validation. Gatefail thì báo thực chất lợi ích, không mở grid vô hạn. Turkish đểsauquyếtđịnh củauser.

## 6. Nguồn và checkpoint

- Local nguồn mới: `outputs_rebuttal/amis_routing_warmup_20261004_v2/`.
- `independently_verified_validation_scores.csv`: sáusốchạyđãtínhlại.
- `independent_audit_summary.json`: means/SD và tấtcảgatebooleans.
- `mechanism_trajectory_summary.csv`:36điềukiệnprobe, táchαmode.
- `collected_records/runs/`: predictions, histories, selection, snapshot/HF/probe receipts.
- HF: `FiveC/amis-rewire-checkpoints`, prefix **`amis_routing_warmup_20261004_v2/<run>/`**; sáubestarchives và24trajectoryarchives ởnamespace riêng. Bestarchive vàtrajectoryarchive đượccheckSHA/size trước COMPLETE; trajectoryweights không chứaoptimizer, khôngphải resumabletrainingcheckpoint.

Không thay sốpaper/EXPERIMENTS cũ bằng những sốvalidation này.


---

<a id="research-insights"></a>

## RESEARCH_INSIGHTS

Original source: `docs/archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#research-insights`. Status: historical record; the summary at the top and the current handoff take precedence.

# CLRR — insight từ thực nghiệm đã xác minh

**HIỆN HÀNH — V5 đã đo và audit, 04/10/2026:** xem thí nghiệm Mục12 và bàn giao19.4–19.5.
42 điều kiện Full/LSR và các đối chứng baseline đã xác minh; A100 Standard đã tắt.

**Có lợi:** routing có tác dụng trên weights đã train; tắt từng nhóm layer không
tạo cải thiện trung bình. Cả ba baseline archives tái lập đủ575câu test cũ,
không có bằng chứng nhầm weights. Sửa nhãn beam của Baseline seed42 giúp
Full−Baseline validation mean đổi từ−.4498 thành+.0861BLEU, nhưng chỉ1/3seed dương.

**Không có lợi cho claim superiority:** bốn α khác và ba group ablations đều có
mean BLEU thấp hơnα.10/routing đầy đủ; chrF++/chrF0 giảm ở3/3seeds trong cả bảy
điều kiện. Không α nào qua guard không làm giảm các metric. Full−tuned LSR mean
BLEU: validation beam1+.2193/beam4+.2055; test beam1−.2168/beam4−.1267.
chrF0 test cũng âm ở cả hai beam. Chênh lệch không chỉ do beam; chưa xác định
overfitting/exposure bias là nguyên nhân. α.125/.15 giảm CE trên32câu cố định
nhưng làm điểm dịch giảm: CE subset và rank không thay cho đánh giá generation.

**Số sửa cần nhớ:** Baseline seed42 validation17.8735BLEU/14.9009chrF++ thực tế
là beam4; beam1 đúng16.2659/13.9015. Mean Baseline validation beam1=17.1275BLEU,
không phải17.6633. Test scores cũ giữ nguyên. Best step216 thiếu provenance:
chuyển sang unknown. Source recovery và F0 metadata đã sửa; lượt sau dùng
corrected V5 inventory riêng. Old CSV/inventory và các lần lỗi giữ để truy vết.

**Cơ chế chưa xác định:** V4 endpoints chưa cho thấy residual explosion hoặc
LSR domination.35probe lịch sử có gradient union/raw spectra tái tính đúng;
các thống kê mô tả rank tăng và relative auxiliary gradient giảm theo epoch.
Historical executable/runtime/language provenance còn thiếu, nên chưa coi đó
là chứng minh nhân quả hoặc xác minh đầy đủ trajectory training.

**Quyết định:** cần pilot thay đổi cách train để cải thiện quality/stability;
chưa đủ căn cứ bỏ CLRR hoặc sửaλ/normalization/gradient path tùy ý. Đề xuất
routing warmup cho Full và CLRR-only, mỗi nhánh ba seeds; controls và GO/NO-GO
ở19.5. **Chưa code/chưa chạy pilot; Turkish vẫn ngưng.** Chọn cấu hình bằng
validation đã khóa rồi confirm test và replication trên dữ liệu chưa xem.
Không hứa ACL hoặc chọn cấu hình bằng test.

**LẦN TRƯỚC — V4 đã thực thi và audit VERIFIED, 04/10/2026:** xem bàn giao
**Mục 18** và thực nghiệm **Mục 11**. Đã chạy A100 Standard, không high-mem;
21/21 endpoints,3/3 route-controls; tải artifacts/SHA và tắt Colab. CPU audit
trong `clrr` tái chấm/replay, tái SVD84endpoint matrices và kiểm tra gradient/
residual/provenance đều PASS. Không train model mới hoặc sửa test scores cũ.

**Có lợi cho phương pháp:** route-on tái lập đúng576predictions cũ ở cả3seeds;
on−off BLEU=+3.1060/+3.0783/+2.7187, mean **+2.9676**, validationbeam1.
Rút kết luận routing vô tác dụng. Đây là dependence của weights đã train với
routing, **không phải +2.97BLEU trước independently trained baseline/LSR**.

**Insight rank mới:** Fullλ0.10−matchedLSR có block12source singular rank tăng
15.83% (dương3/3), nhưng sau finalLN singular rank385.2893 so với386.3290
(âm3/3). Final energy entropy rank tăng5.21% (dương3/3). Không có một kết luận
“rank luôn tăng” theo mọi definition/site, không chứng minh chống collapse gây
ra quality gain. CLRR-only cũng có final singular rank thấp hơnbaseline3/3.

**Gradient/residual:** Fullλ0.10 weightedLSR/CEnormmean0.1570%, globalcos0.0516;
có13/288layer×batchcosinesâm, nên rút câu “không conflict” tuyệt đối. Residual
RMSratio max mọiroutingbranches/phase8.7783%; chưa thấy endpoint explosion,
không phải bảo đảm mọiphase vô hại. Không có căn cứ sửa loss/normalization ngay.

**Quality gate chưa đổi:** Full−matchedLSR+0.2344BLEU (3/3 dương), nhưng
Full−tunedLSR−0.1267; Fullλ0.03−tunedLSR−0.8100. Validation Full−tunedLSR
mean+0.2193BLEU trong inventory không chuyển thành test gain. Cần hiểu
generalization/selection; không dùng rank/alignment để thay translation quality.

**Tiếp theo đề xuất:** error analysis bằng predictions đã có; nếu cần,
validation-only inferenceαsweep12generations mới (3seeds,αchung), rồi routing
group intervention có điều kiện9generations. Đây là diagnostics/tuning mới,
chưa chạy; chỉ chọn pilot architecture/schedule khi có quan sát phân biệt,
không thêm λgrid hoặc mở Turkish tìm số thuận lợi. Xem18.5 để biết controls,
selection và phạm vi exploratory. Receipt:
`outputs_rebuttal/amis_followup_20261004_v4/independent_audit.json`.
Các block V3/V2 bên dưới là lịch sử và đã được thay thế trong phạm vi đo V4.

**Review V3 trước khi chạy V4, 04/10/2026 — xem bàn giao Mục 17.** CSV route-control
khớp phép chấm, nhưng route-on không tái lập validation trước đó (thấp hơn
1.6971–3.1833 BLEU). Source local và bundled worker thiếu `tl_XX/zh_CN`;
runtime tokenizer thực tế chưa được lưu để giải thích đầy đủ chênh lệch.
F4 đã sửa cách tháo hooks, nhưng chưa dùng “89% câu đổi” như evidence cùng
protocol. Chưa dùng rank layer12 cao hơn làm claim chống collapse: inputs cần
xác nhận, site trước final LayerNorm khác V2, covariance rank field thực ra là
participation ratio. Gradient accumulator bỏ decoder CE ở baseline/CLRR-only
(48 batches), nên global scope giữa controls chưa nhất quán.

**Quality đã xác minh giữ nguyên:** Full λ0.10 hơn matched LSR +0.2344 BLEU
(3/3 seeds), nhưng thấp hơn tuned LSR −0.1267; Full λ0.03 thấp hơn tuned LSR
−0.8100 (3/3 seeds). Có tín hiệu để thử cải tiến, chưa đạt gate mở rộng.
Kế hoạch: hai checkpoint sentinel kiểm tra protocol → 6 on/off generations
và 21 endpoint probes đã sửa, không train → khóa giả thuyết → pilot/confirmation
có controls phù hợp. Không mở grid hoặc Turkish trước gate. Receipt riêng:
`outputs_rebuttal/amis_followup_20261004_v2/v3_independent_review_20261004.json`.
Các block review V2/final-test bên dưới là lịch sử; ưu tiên Mục 17 khi bàn giao.

**Review follow-up mới nhất04/10/2026 — bàn giao mục16:** tái tính21test sets,
reference/order/SHA và scores khớp (maxerror≈1e-14); đúngsecondary30pairs/Holm,
nativeSacreBLEUp khớp choBLEU/chrF++ ởFullλ0.03−tunedLSRseed42.
Fullλ0.03 thua tunedLSR cả3seeds: mean−0.8100BLEU/−0.6702chrFcharacter.
Giảmλ đơn thuần chưa giúp; originalFull vẫn−0.1267BLEU trước tunedcontrol.
**Phát hiện F4 INVALID:** haiwrappers chungbase_model khiến hookα0.10 vẫn hoạt động
trongroute-off. Đã tái hiệnCPU; không dùng0câuđổi để kết luận routing vô tác dụng.
F3 có21endpoints/batchtables hợp cấu trúc nhưng thiếu per-layer gradients/ranks,
rawspectra/representations và runtime/loadingprovenance. Pooledfinalrank lớn không
chứng minh anti-collapse toànencoder. NO-GO mởrộng giữ nguyên theoquality;
chưa buộc chuyểnpaper sangmechanistic study hoặc đổi phươngpháp trước sửaF4.
Receipt riêng: `outputs_rebuttal/amis_followup_20261004_v2/independent_review_20261004.json`.
Các đoạn“secondarychưađủ” bên dưới là trước lầnreview này; familyđúng hiệnđãcó.

**Kết luận mới nhất04/10/2026 sau final test:** đã có17runs mới và18models test.
Review CPU độc lập trong conda `clrr` xác minh cả18prediction sets: raw575rows,
source/target/index, SHA256 và cả3metrics đều khớp (sai số<1e-8).
Xem [bàn giao mục13](../operations/CONFIGURATION_SEARCH_HANDOFF.md#13-review-độc-lập-sau-final-test--04102026)
và receipt `outputs_rebuttal/amis_final_evaluation_20261004/final_review_20261004.json`.

Full−baseline testmean+0.3800BLEU/+0.3263chrFcharacter;
Full−matchedLSR+0.2344BLEU(3/3seeds)/+0.1675chrFcharacter.
Nhưng Full−tunedLSRλ0.03 **−0.1267BLEU/−0.1635chrFcharacter** (thua2/3seeds).
Gain+1.3642chrF++ trước tunedLSR rất nhạy wordbigram: chỉ bỏ row170 để chẩn đoán
thì mean gap thành−0.1832; giữ nguyên mọi575rows cho score chính.
Không gọi tunedLSR seed43 collapse hoặc suy detach tối ưu từquality/sample std.
Không p-value primaryHolm18 nào đạt0.05; CI lưu là chưa điều chỉnh.
FilesecondaryHolm30 hiện dùng pairs khác kế hoạch, thiếu interactionλ0.03
và CLRR-only−baseline, nên cần hoàn tất đúng family ở artifacts mới.

**NO-GO mở rộng/claim superiority theo gate hiện hành**, chưa kết luận phương pháp vô ích.
Tiếp theo: hoàn tất controls không train → probe validation CE–LSR/residual/rank
→ chọn candidate có cơ sở và xác nhận trước mở rộng. Chưa chạy GPU/Turkish trong review.
Các cập nhật screening/confirmation dưới đây là lịch sử, không phải trạng thái cuối.

**Bàn giao chạy tiếp sau kiểm tra source:** xem mục14 của file bàn giao. Vòng kế
tiếp không train: chấm3Fullλ0.03, hoàn tất secondary family đúng30tests, probe21best
trên validation và route-on/offFull3seeds. Đã kiểm tra đủ35trajectory probesC cùng
8batches; global weightedLSR/CEnorm ở Full nhỏ và residual ratio max khoảng0.0868.
Chưa có căn cứ tăngλ hoặc normalization ngay. Các probe entries hiện cần adapter
cho λ/split/detach/output trước chạy; đây là kế hoạch, chưa mởGPU.

Ngày lập: 03/10/2026. Đi cùng [EXPERIMENTS_UPDATED.md](../experiments/EXPERIMENTS_UPDATED.md)
và [paper hiện hành](../paper/clrr_main.tex). File này diễn giải bằng chứng; sổ kết quả và
artifact gốc vẫn là nguồn số. Không có phép đo hoặc lượt training mới trong lần lập file này.

**Cập nhật04/10/2026 — screening D đã xong8/8:** xem [phân tích bàn giao, mục9](../operations/CONFIGURATION_SEARCH_HANDOFF.md#9-phân-tích-hiện-hành-và-quyết-định--04102026)
và receipt `outputs_rebuttal/amis_configuration_20261003_v1/screening_review_20261004.json`.
Tái tính8sets validation đạt; không có test mới hoặc training trong lượt phân tích này.
Best Full vẫn là gốc α0.10/λ0.10 (14.7317chrF++), không phải Full α0.20 mới (14.5738).
Best LSR λ0.03 đạt14.8587, baseline C seed42 đạt14.9009; Full chưa vượt hai mốc này.
Full−matchedLSR0.10 +0.5643chrF++, nhưng Full−bestLSR −0.1271;
gain matched-control chưa đủ bảo vệ contribution riêng qua tuning.
LSR0.03 đứng đầu8trial mới trong mọi576leave-one-out cả chrF/chrF++; ranking thấp hơn
có thể đổi nên không claim toàn ranking bền. Các điểm dừng/phân tích dưới đây là lịch sử.
Kế hoạch mới ưu tiên độ tin cậy theo tác giả: audit/reuse → phục hồi2replicas seed42
và xác nhận LSR0.03 seeds43/44 (4train) → interaction/detach controls (5train)
→ attached CLRR-only3train nếu architecture đó được chọn. Chưa khởi động GPU.
Hiện **chưa đủ vững cho claim CLRR cải thiện rõ**, nhưng chưa kết luận phải bỏ phương pháp.

**Sau confirmation6runs,04/10:** [bàn giao mục10](../operations/CONFIGURATION_SEARCH_HANDOFF.md)
đối chiếu Fullλ0.10 với tunedLSR0.03 đúng3seeds: mean+0.2193BLEU/+0.1805chrF++,
2/3seeds dương. Full hơn matchedLSR0.10+0.6705BLEU/+0.4368chrF++,3/3dương;
nhưng vẫn thua baseline validation mean−0.4498BLEU, BLEU giảm3/3seeds.
Đây là tín hiệu có lợi cho routing so với controls, chưa là superiority chắc chắn.
Fullλ0.03 gần hòa LSRλ0.03; sample std thấp ở3seeds không chứng minh representation stability.
Replicas42 khớp validation C, hỗ trợ metric reproducibility; không chứng minh tất định tuyệt đối.
Đã thấy4B2+2interaction, chưa thấy3no-detach runs hoặc final candidate test.
Giữλ0.10 làm selected Full để kiểm chứng tiếp; chưa có căn cứ đổi architecture hoặc tăng claim paper.

## 1. Bản cũ và phạm vi hiện hành

- [Insight cũ đã archive](README.md)
  được giữ nguyên để truy vết. Những nhận định lịch sử về bảo toàn morphology,
  chống collapse, gradient explosion hoặc PEFT overfitting không tự trở thành
  kết luận của bản mới.
- [Phiên bản insight sớm hơn](README.md)
  và [phân tích paper cũ](README.md) chỉ là lịch sử.
- Phạm vi hiện hành: Amis→Mandarin. Không đưa Asháninka hoặc LayerSkip trở lại.
  ByT5 vẫn thuộc các đối sánh được giữ trong paper hiện hành.
- [METRICS.md](../experiments/METRICS.md) quy định cách đo;
  [CLRR_GO_NOGO_PLAN.md](../research/ACL_LONG_TERM_EXPERIMENT_PLAN.md) quy định các gate phát triển.

## 2. Ba nhóm bằng chứng phải đọc riêng

| Nhóm | Đã có | Có thể trả lời | Giới hạn |
| --- | --- | --- | --- |
| A | 28 outputs lịch sử Amis, chấm lại saved predictions | Điểm của các cấu hình đã chạy với raw references | Training và inference lịch sử không matched |
| B | 10 selected checkpoints generate FP32 và 4 routing-off outputs | Điểm dưới inference chung; phụ thuộc routing của cùng weights | Không sửa khác biệt training/selection lịch sử |
| C | 12 lượt mBART, bốn nhánh × seeds 42/43/44; 35 trajectory probes | Đóng góp thành phần dưới protocol chung; hành vi trong training | Một direction/backbone, ba seed; endpoint không nhất thiết là best |

Không gộp A/B/C thành một mean/std hoặc kiểm định. Giữ các family Holm gốc
A42/B36/C24. Sentence bootstrap không đo biến thiên giữa các lần train.

## 3. Tín hiệu chính từ suite matched C

Nguồn: `outputs_rebuttal/amis_confirmation_20261003/seed_summary.csv`,
`paired_seed_deltas.csv`, `seed_scores.csv` và `paired_bootstrap24.csv`.

| Phương pháp | BLEU mean ± sample std | chrF++ mean ± sample std |
| --- | ---: | ---: |
| Baseline | 19.3886 ± 1.1203 | 16.9249 ± 3.2626 |
| CLRR-only | 19.5769 ± 0.6057 | 18.5172 ± 0.4136 |
| LSR-only | 19.5343 ± 0.2325 | 18.5639 ± 0.1745 |
| Full | 19.7687 ± 0.1743 | 18.6565 ± 0.2107 |

| So sánh | Mean ΔBLEU | Seed dương | Diễn giải hiện tại |
| --- | ---: | ---: | --- |
| Full−Baseline | +0.3800 | 2/3 | Gain trung bình dương, có seed giảm 0.5629 |
| Full−LSR-only | +0.2344 | 3/3 | Tín hiệu hữu ích nhất cho đóng góp routing trong tổ hợp |
| CLRR-only−Baseline | +0.1883 | 1/3 | Chưa chứng minh routing đứng riêng cải thiện ổn định |
| Full−CLRR-only | +0.1917 | 2/3 | Chưa chứng minh kết hợp luôn tốt hơn routing đơn |

**Có lợi:** Full có mean BLEU cao nhất và hơn LSR-only ở cả ba seed.
**Còn yếu:** không kiểm định nào trong Holm24 đạt p<0.05; ba seed chưa đủ
chứng minh tính ổn định tổng quát. Full có std nhỏ nhất là mô tả của ba lượt này,
không phải bằng chứng đã giảm variance của quá trình training.

Mean Full−Baseline chrF++ là +1.7316, phần lớn đến từ seed43 (+5.3162).
Mean Full−LSR-only chỉ +0.0926 chrF++, với hai seed dương.
Chưa diễn giải những số này thành cải thiện hình thái trước character/influence audit.

## 4. Historical results và cơ chế: điều gì được hỗ trợ?

| Quan sát | Có thể nói | Chưa thể nói |
| --- | --- | --- |
| mT5 Full−Baseline +1.8074 BLEU, Holm p=0.0036 ở B | Có gain trên các checkpoints lịch sử được đánh giá lại | Gain đã lặp qua seed/protocol matched; hữu ích trên mọi backbone |
| mBART routing on−off +3.9287 BLEU, Holm p=0.0036 ở B | Checkpoint Full đã sử dụng routing trong computation | CLRR hơn một LSR-only được train riêng 3.93 BLEU |
| ByT5 Full−Baseline −0.1696 BLEU, không significant | Có point estimate giảm ở backbone này | Phương pháp luôn có lợi hoặc ByT5 chắc chắn bị hại |
| Full weighted LSR/CE global gradient median khoảng 0.43–0.54% đầu và 0.15% cuối training C | Lực gradient tương đối của auxiliary objective giảm theo các mốc đo | Tăng lambda sẽ tự cải thiện dịch; gradient nhỏ đồng nghĩa không tác động |
| Pooled effective ranks còn cao; biến đổi khác nhau theo backbone | Chưa thấy global collapse trong các phép đo này | CLRR đã chứng minh chống collapse hoặc bảo toàn phụ tố |

CLRR thêm zero parameters nhưng vẫn full fine-tuning. Không gọi đó là giảm
optimizer memory hay tốc độ tốt hơn PEFT khi chưa benchmark cùng điều kiện.
Khác LR giữa các phương pháp không tự là bất công; khác tuning budgets/protocol
khiến các điểm LoRA lịch sử chưa đủ làm ranking tổng quát.

## 5. Rủi ro đo đạc cần giải quyết trước tối ưu

Historical mBART ở B có Full−Baseline chrF++ +5.0277, nhưng exploratory
leave-one-out bỏ row170 làm gap còn khoảng +0.3844. Character-only chrF trên
toàn test có gap +0.6907. Đây là sensitivity của metric trên Mandarin ít khoảng trắng;
không phải lý do xóa câu hoặc thay primary metric sau khi thấy test.

**Cập nhật:** influence audit của cả 12 outputs C đã hoàn tất ở mục9; phân tích
trước đó được giữ như bối cảnh, không suy từ B sang C.
Ba nhóm regex mi/ma/pa là proxy có overlap, không là gold morphology.

Chưa có matched training control cùng routing nhưng không detach, nên chưa
phân biệt lợi ích của activation reuse với lợi ích của stop-gradient.
Tiny-model correctness chỉ chứng minh hành vi implementation.

## 6. Đánh giá bài hiện tại

Quick review trong chat: soundness 3/5, excitement 2.5/5, overall 2.5/5
(Borderline Findings), confidence 3/5. Đây là nhận định của cùng trợ lý đã tham gia
sửa bài, **không phải panel độc lập hoặc dự báo acceptance đã hiệu chuẩn**.
Thang tham khảo: [ARR review form](https://aclrollingreview.org/reviewform).

Các vấn đề còn lại: đóng góp riêng nhỏ/chưa chắc chắn; thiếu control detach;
influence audit C chưa xong; novelty cần đối chiếu trực tiếp với nghiên cứu xuyên tầng,
như [Multi-layer Representation Fusion](https://aclanthology.org/C18-1255/)
và [Dense Information Flow](https://aclanthology.org/N18-1117/).
Những công trình này là đầu mối literature cần đọc sâu; không kết luận CLRR trùng chúng.
Một language pair không tự làm bài không hợp lệ; scope và sức mạnh claim phải phù hợp.

**Quyết định:** GO cho chẩn đoán rẻ; chưa đủ bằng chứng để nộp ACL main với kỳ vọng
một đóng góp phương pháp mạnh. Chưa có căn cứ phải bỏ CLRR hoặc làm lại toàn bộ phương pháp.

## 7. Bước tiếp theo đã chuẩn bị: character và influence audit C

### Đầu vào đã kiểm tra

- Đủ 12 `runs/mbart-amis-<method>-seed<seed>/test_predictions.csv` trong suite C.
- Mỗi file 575 rows; SHA-256 khớp `seed_scores.csv`; source đúng thứ tự và
  saved targets khớp raw targets sau strip ngoài chuỗi.
- Receipt: [input_manifest.json](../../outputs_rebuttal/amis_confirmation_20261003/metric_influence_preparation/input_manifest.json).
- CPU trong conda `clrr`; không cần checkpoint, Colab hoặc training.

### Phép đo cần thực hiện

1. Tính character-only chrF cho cả 12 outputs; đối chiếu full-test chrF++ với số đã ghi.
2. Với từng seed, đo Full−Baseline, Full−LSR, CLRR−Baseline và Full−CLRR.
3. Leave-one-row-out cho chrF và chrF++: mỗi row bỏ đúng một lần; báo toàn bộ
   khoảng biến thiên, row ảnh hưởng nhất, độ thay đổi gap và số lần đổi dấu.
4. Kiểm tra shortcut sufficient statistics bằng direct corpus scoring ở những
   row cực trị; giữ mọi câu trong kết quả chính và không tính p-value mới từ leave-one-out.

Script lịch sử `scripts/revalidation/measure_prediction_influence.py` đang trỏ
nhóm A và output cũ. **Cần bổ sung chế độ riêng cho C trước khi chạy**, kiểm tra
hash/reference alignment và ghi outputs vào thư mục C riêng; không chạy nguyên
script cũ rồi gọi là audit suite mới.

### Đầu ra và quyết định sau khi đo

- Artifact mới dự kiến: `outputs_rebuttal/amis_confirmation_20261003/metric_influence/`.
- Ghi số vào EXPERIMENTS_UPDATED, diễn giải vào file insight này. Paper chưa đổi
  cho đến khi kết quả được đối chiếu; kết quả cũ và raw artifacts vẫn giữ nguyên.
- Nếu BLEU/char-only cùng ủng hộ gain và influence không tập trung mạnh:
  có cơ sở chuyển sang chẩn đoán/validation tuning theo Gate1–2.
- Nếu gain chrF++ dựa vào rất ít câu hoặc metrics bất đồng: audit validation
  và lỗi dịch trước khi đổi trọng số. Không loại câu để làm gain đẹp hơn.
- Báo từng kết quả có lợi/bất lợi lên chat trước khi chuyển sang thí nghiệm khác.

**Trạng thái khi lập kế hoạch:** đã chuẩn bị đầu vào; chưa chạy audit mới, sweep trọng số,
control không detach hay Turkish. Trạng thái thực thi mới xem mục9.

## 8. Kế hoạch chạy tiếp sau review

[CLRR_GO_NOGO_PLAN.md](../research/ACL_LONG_TERM_EXPERIMENT_PLAN.md) đã được cập nhật thành lộ trình cụ thể:
audit không train → tuning Full/LSR cùng số trials → validation qua seed →
CLRR-only/no-detach controls → đánh giá Amis → mở rộng độc lập hoặc nâng cấp theo nguyên nhân.
Trần Amis20lượt mới nếu reuse hợp lệ, nâng cấp có điều kiện tối đa10lượt;
Turkish và NLLB mỗi bên pilot4, chỉ mở thành12 nếu qua gate tương ứng.
Không có kết quả mới trong lần cập nhật này. Các threshold là tiêu chí đầu tư dự án,
không là chuẩn acceptance hoặc bảo đảm phương pháp sẽ thắng.

## 9. Audit C đã xong và vòng D đã bắt đầu

**Điểm dừng mới:** tác giả yêu cầu xong Full α0.20/λ0.10/seed42 thì tạm ngưng.
Hướng dẫn và trạng thái cuối tại [CONFIGURATION_SEARCH_HANDOFF.md](../operations/CONFIGURATION_SEARCH_HANDOFF.md).
Trial α0.05/λ0.10/seed42 đã xong: validation chrF++14.4092, thấp hơn Full gốc14.7317.
Giảm routing chưa giúp ở trial này; chưa có kết luận từ cả screening hoặc test mới.
Validation chỉ có1word-bigram reference (row197); trial α0.05 match0, nên câu này không
tạo gain word-bigram cho trial đó. Bỏ row197 chỉ đổi chrF++ thành14.4440.
No-detach CPU gradient/checkpoint gate đã đạt; GPU/control training vẫn chưa chạy.

Trial α0.20/λ0.10/seed42 cũng đã xong20epochs, selected checkpoint720:
validation BLEU17.4586, chrF++14.5738, chrF16.7469. Tốt hơn α0.05 nhưng chưa vượt
α0.10 gốc (validation chrF++14.7317). Cả hai trial mới đều match0word-bigrams;
thứ hạng giữa hai trial giữ nguyên trên chrF và khi bỏ row197. Chưa có test mới.
**Insight tạm thời:** thay α theo hai hướng chưa giúp ở seed42; chưa đủ kết luận λ
hay contribution CLRR vì sáu screening trials và confirmation chưa chạy.
Đã tạm ngưng theo tác giả, backup cả hai checkpoints FiveC đã verify SHA/size,
metrics/predictions lưu local; chưa nâng cấp phương pháp.

Nguồn: `outputs_rebuttal/amis_confirmation_20261003/metric_influence/`.
Đủ12systems/575rows, chrF++ khớp số cũ <1e-8; shortcut leave-one-out được kiểm tra
với direct corpus scoring ở các cực trị, sai số <1e-10. Không sửa điểm cũ, không
xóa câu và không tạo kiểm định significance mới.

| So sánh character-only chrF | Seed42 | Seed43 | Seed44 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Full−Baseline | +0.2654 | +1.1001 | −0.3865 | +0.3263 |
| Full−LSR | +0.3046 | −0.0676 | +0.2655 | +0.1675 |

**Có lợi:** Full vẫn có mean character-only cao nhất và Full−LSR dương2/3seeds;
gap dương ở seed42/44 không đổi dấu khi bỏ bất kỳ một câu nào.
**Bất lợi:** không xác nhận lợi ích character-only ở3/3seeds. Seed43 Full−Baseline
chrF++ +5.3162 giảm còn +0.5803 khi bỏ row170; character-only gap +1.1001 còn
+0.8602. ChrF++ làm hiệu ứng seed43 trông lớn hơn mức character agreement.
Đây không là lỗi của primary metric hay bằng chứng phải loại câu; giữ full test.
Leave-one-out không chứng minh robustness khi bỏ nhiều câu hoặc qua training seeds.

**Hành động:** tiếp tục validation tuning có controls; chưa tăng claim trong paper.
Không dùng mean chrF++ +1.7316 làm bằng chứng gain rộng/ổn định hoặc morphology.

---

### Tổng kết Thực nghiệm Phase C2 (Attached Control) & Final Test Suite (04/10/2026)

1. **Hiệu ứng của Cơ chế Detach (Attached vs Detached):**
   - Ba seeds Full Source-Attached ($\alpha=0.10, \lambda=0.10$, `detach=False`) cho kết quả không đồng đều: seed42 giảm, seeds43/44 có cải thiện trên validation. Attached validation mean BLEU cao hơn detached khoảng0.1236, chrF++ thấp hơn0.0471.
   - Giữ detached theo validation chrF++ đã khóa. Quality và sample std chưa chứng minh `.detach()` tối ưu, bảo đảm ổn định biểu diễn hoặc tránh gradient interference; cần probes trực tiếp như bàn giao mục14.

2. **Đánh giá Test Checkpoints Cuối cùng (18 Models, 6 Nhánh, 3 Seeds):**
   - Test575rows, FP32, beam4, tokenizer `zh`, SacreBLEU2.6.0. Theo bàn giao, architecture khóa trước generation mới; testC đã được xem, không gọi là test chưa từng được biết.
   - **Độ phân tán quan sát:** Full có stdBLEU0.1743/chrF++0.2107 trên3seeds này. ChrF++ của baseline/tunedLSR nhạy wordbigram hiếm; không gọi giảmscore seed43 là collapse. TunedLSR seed43 vẫn hơn Full ở BLEU và chrF character-only.
   - **Tín hiệu gia tăng có tính lặp lại (Incremental Gain):** Full CLRR Detached vượt Matched LSR $\lambda=0.10$ ở **100% (3/3) các seeds** (+0.1701 ở seed 42, +0.1243 ở seed 43, +0.4087 ở seed 44; mean delta **+0.2344 $\pm$ 0.1527 BLEU**).
   - **Kiểm định và giới hạn:** không p-value primaryHolm18 nào đạt0.05. Holm điều chỉnh p-values, không mở rộng CI đang lưu; có2CI chưa điều chỉnh không chứa0. SecondaryHolm30 hiện khác pairs đã khóa và cần hoàn tất theo bàn giao mục14. Full thua tunedLSR mean0.1267BLEU/0.1635chrF character-only; chưa đủ claim superiority.


---

<a id="fixed-experiments-if-method-changes"></a>

## FIXED_EXPERIMENTS_IF_METHOD_CHANGES

Original source: `docs/archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#fixed-experiments-if-method-changes`. Status: historical record; the summary at the top and the current handoff take precedence.

# Các thực nghiệm giữ lại khi thay CLRR và/hoặc LSR

> **Cập nhật 05/10/2026:** G0 phát hiện lỗi accumulation ở wrappers CLRR/LSR, xem [verification](TECHNICAL_AUDITS_20261005.md#g0-g1-verification-20261005). Các số dưới đây vẫn giữ làm records của checkpoints cũ. Không coi affected wrapped controls là matched controls của code đã sửa; cần corrected cohort trước khi so phương pháp mới.

Ngày lập: 05/10/2026.

Nguồn: [paper hiện tại](../paper/clrr_main.tex), [experiments đã cập nhật](../experiments/EXPERIMENTS_UPDATED.md), [kế hoạch dài hạn](../research/ACL_LONG_TERM_EXPERIMENT_PLAN.md).

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

