# Giao thức và số liệu metrics chuẩn của CLRR

Ngày audit: **2026-10-02**. Nguồn số chính: [metrics_standardized_scores.csv](../results/metrics_standardized_scores.csv). 36 outputs đã xác minh; [audit đầy đủ](RESULTS_VERIFICATION.md) và [translation_audit.json](../outputs_rebuttal/metric_audit_20261002/full/translation_audit.json) lưu provenance/SHA. Audit ban đầu 29 outputs và CLI checks giữ tại [standardized_metrics.json](../outputs_rebuttal/metric_audit_20261002/standardized_metrics.json).

## 1. Giao thức chính duy nhất

- Reference là `target` nguyên bản trong `data_processed/<dataset>/test.csv`, đúng thứ tự và đủ số câu. Không tokenize/decode reference bằng tokenizer của mô hình.
- Hypothesis là predictions đã decode; chỉ bỏ whitespace ở hai đầu. Giữ nguyên punctuation, case và mọi prediction rỗng; không lọc câu khó hoặc đổi Unicode normalization.
- **BLEU:** SacreBLEU, corpus-level, case-sensitive, order 4, smoothing `exp`, `effective_order=False`; tokenizer `zh` cho Mandarin, `13a` cho Spanish/English.
- **chrF++ chuẩn:** `CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False)` trên văn bản gốc. Không dùng TokenizerZh, SentencePiece hay tách từ bên ngoài trước chrF++.
- Điểm ở thang 0–100; bảng hiển thị 4 chữ số thập phân, CSV/JSON giữ full precision. Không lấy trung bình sentence-level BLEU/chrF++ để thay corpus score.

Định nghĩa và reporting theo [SacreBLEU](https://github.com/mjpost/sacrebleu), [chrF++: words helping character n-grams](https://aclanthology.org/W17-4770/) và [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/). Cấu hình giữa các paper có thể khác; chỉ so điểm trực tiếp khi cùng test/reference và metric signature.

### Signatures đã dùng — SacreBLEU 2.6.0

```text
BLEU Mandarin: nrefs:1|case:mixed|eff:no|tok:zh|smooth:exp|version:2.6.0
BLEU Spanish/English: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF++: nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

Lệnh tương đương sau khi xuất reference gốc và predictions thành hai file UTF-8, mỗi dòng một câu:

```powershell
conda run -n clrr python -m sacrebleu reference.detok.txt -i prediction.detok.txt -m bleu chrf --tokenize zh --chrf-word-order 2 --format json --width 4
```

Đổi `--tokenize zh` thành `--tokenize 13a` cho Spanish/English. Option này dành cho BLEU; không tiền xử lý chrF++.

### Phân tích phụ

`chrF++ (TokenizerZh diagnostic)` là một phép đo khác: áp dụng TokenizerZh lên cả hypothesis và reference trước CHRF. TokenizerZh tách ký tự Hán và phần ngoài Hán; không phải morphological word segmentation. Không gọi số này là chrF++ chuẩn hoặc trộn với số chính. Không kết luận chrF++ chuẩn bị lỗi chỉ vì tiếng Trung thiếu khoảng trắng.

## 2. Amis → Mandarin — mBART

Backbone: `facebook/mbart-large-50-many-to-many-mmt`. Test 575 câu; mọi hàng dưới đây được tính lại từ predictions đã lưu, cùng reference gốc.

| Phương pháp | BLEU | chrF++ chuẩn |
| --- | ---: | ---: |
| Baseline | 19.6144 | 14.0547 |
| CLRR-only | 18.4697 | 13.2443 |
| LSR-only | 19.6291 | 18.5753 |
| CLRR+LSR | 20.3896 | 19.0824 |
| CLRR-Dec+LSR | 19.9188 | 18.6999 |
| Middle-Layer Alignment | 19.2893 | 18.4378 |
| BitFit | 0.3463 | 2.2552 |
| Narrow LoRA | 3.2365 | 4.4213 |
| Strong LoRA A | 10.2296 | 8.6319 |
| Strong LoRA B | 13.8066 | 10.3293 |

## 3. Amis → Mandarin — NLLB

Backbone: `facebook/nllb-200-distilled-600M`. Test 575 câu.

| Phương pháp | BLEU | chrF++ chuẩn |
| --- | ---: | ---: |
| Baseline | 13.5001 | 10.4389 |
| CLRR+LSR | 14.2820 | 10.9152 |
| CLRR-Dec+LSR | 13.4194 | 10.4559 |
| Middle-Layer Alignment | 13.9648 | 10.6596 |
| BitFit | 1.0750 | 2.9643 |
| Narrow LoRA | 3.5383 | 4.9407 |
| Strong LoRA A | 11.4625 | 9.5004 |
| Strong LoRA B | 9.8024 | 8.4010 |

Predictions của sáu lượt NLLB gốc được đọc riêng từ `nllb-200/nllb_results.zip` trên Hugging Face qua HTTP range: chỉ **815.959 bytes** được truyền từ archive **41.046.143.831 bytes**. Không tải toàn bộ checkpoint và không nạp trọng số. Receipt: [remote_nllb_reports/receipt.json](../outputs_rebuttal/metric_audit_20261002/remote_nllb_reports/receipt.json).

## 4. Amis → Mandarin — mT5-small

Test 575 câu. CLRR-only ở đây là encoder rewiring không LSR.

| Phương pháp | BLEU | chrF++ chuẩn |
| --- | ---: | ---: |
| Baseline | 2.7887 | 3.8129 |
| CLRR-only | 4.4393 | 5.1761 |
| CLRR+LSR | 4.5961 | 5.0425 |
| CLRR-Dec+LSR | 4.8315 | 5.1933 |
| CLRR-Both+LSR | 3.4459 | 4.5753 |
| Middle-Layer Alignment | 4.6625 | 5.2103 |

**LSR-only chưa xác minh:** không tìm thấy predictions tương ứng. Số `4.5422 / 5.0118` trong docs cũ không được chứng nhận; một metrics file trên HF ghi `4.973986 / 5.942590`, cũng chưa được đưa vào bảng chuẩn vì chưa có predictions để đối chiếu. Không tự chọn một trong hai bộ số.

## 5. Asháninka → Spanish

Test 1.003 câu. Các điểm dưới đây tái lập khớp metrics đã lưu; không cần sửa điểm.

| Backbone | Phương pháp / checkpoint | BLEU (13a) | chrF++ chuẩn |
| --- | --- | ---: | ---: |
| mBART | Baseline (patience 4) | 4.0085 | 20.8536 |
| mBART | CLRR+LSR (patience 4) | 3.9671 | 20.6990 |
| NLLB | Baseline (epoch 20) | 3.8594 | 20.4176 |
| mBART | CLRR+LSR (patience 10, selected epoch 14) | 3.7970 | 20.9753 |
| mBART | CLRR+LSR (patience 10, fixed epoch 20) | 4.0949 | 20.9588 |

Lượt patience 10 và Baseline patience 4 có quy tắc dừng khác nhau. Epoch 14 là best theo validation; epoch 20 là phân tích bổ sung theo yêu cầu tác giả, không có khác biệt có ý nghĩa trong bootstrap mới; chưa chứng minh generalization.

## 6. Những số đã sửa và nguyên nhân

Tokenizer mBART làm thay đổi **62/575 reference** khi encode/decode; không có reference bị truncate trong phép kiểm tra max length 256. Dùng cached tokenizer và reference roundtrip tái lập đúng các điểm Trainer của LSR-only, CLRR-Dec, Middle-Align, BitFit, Narrow LoRA và Strong LoRA A/B. Những điểm đó không được trộn với điểm dùng CSV gốc.

| mBART / phương pháp | BLEU cũ → chuẩn | chrF++ cũ → chuẩn |
| --- | ---: | ---: |
| LSR-only | 20.0828 → 19.6291 | 16.0690 → 18.5753 |
| CLRR-Dec+LSR | 20.3986 → 19.9188 | 16.1939 → 18.6999 |
| BitFit | 0.3511 → 0.3463 | 2.8464 → 2.2552 |
| Narrow LoRA | 3.3284 → 3.2365 | 5.1583 → 4.4213 |
| Strong LoRA A | 10.4193 → 10.2296 | 9.8008 → 8.6319 |
| Strong LoRA B | 14.0432 → 13.8066 | 11.4901 → 10.3293 |
| Middle-Layer Alignment | 19.7243 → 19.2893 | 15.5253 → 18.4378 |

Các số từng ghi trong docs `19.6106/14.0538` (Baseline) và `20.3927/19.0839` (CLRR+LSR) được đồng bộ thành **19.6144/14.0547** và **20.3896/19.0824** theo predictions hiện có. Các bảng `chrF(w=0)` và TokenizerZh trước đây cũng có số lệch artifact; diagnostic full precision nằm trong audit.

mT5 Middle-Align được sửa từ `4.7498/5.9373` thành **4.6625/5.2103**. CLRR-Dec là **4.8315/5.1933**; CLRR-Both là **3.4459/4.5753** theo saved predictions.

Các `metrics.json`, CSV lịch sử và biên nhận upload được giữ nguyên làm raw artifacts. Chỉ dùng CSV chuẩn ở đầu tài liệu cho bảng điểm hiện tại; không đọc trực tiếp điểm Trainer cũ để chép vào paper. Bản Markdown trước audit lưu tại `outputs_rebuttal/metric_audit_20261002/docs_before/`.

## 7. Kết quả còn tốt hay xấu?

| So sánh | Δ BLEU | Δ chrF++ chuẩn |
| --- | ---: | ---: |
| mBART CLRR+LSR − Baseline | +0.7752 | +5.0277 |
| mBART CLRR+LSR − LSR-only | +0.7606 | +0.5071 |
| mBART CLRR+LSR − Strong LoRA A | +10.1601 | +10.4505 |
| mBART CLRR+LSR − Strong LoRA B | +6.5830 | +8.7531 |
| NLLB CLRR+LSR − Baseline | +0.7818 | +0.4763 |

- **Tốt:** CLRR+LSR vẫn dẫn các đối chứng mBART/NLLB đã kiểm tra. Lợi ích trên Baseline vẫn dương ở cả BLEU và chrF++.
- **Bất lợi cho diễn giải cũ:** lợi ích raw chrF++ trên LSR-only là **+0.5071**, thay vì khoảng +3,01. LSR-only đã có phần lớn mức tăng raw chrF++ trên Baseline; chưa thể quy toàn bộ gain cho rewiring.
- **CLRR-only vẫn thấp hơn Baseline:** bằng chứng hiện tại hỗ trợ tổ hợp có LSR, chưa hỗ trợ CLRR luôn hữu ích độc lập.
- **Significance đã tính lại:** full − baseline mBART/NLLB và full − LSR-only không có ý nghĩa sau Holm; full − Strong LoRA A/B còn có ý nghĩa ở cả hai backbone. Một seed, chưa chứng minh bảo toàn hình thái; A/B đổi LR nên không tách riêng embeddings.

## 8. Tập con hình thái — cùng giao thức chuẩn

| Tập con | N | Baseline BLEU | CLRR+LSR BLEU | Baseline chrF++ | CLRR+LSR chrF++ | Δ chrF++ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 575 | 19.6144 | 20.3896 | 14.0547 | 19.0824 | +5.0277 |
| mi | 195 | 17.8639 | 18.4293 | 14.9140 | 15.3779 | +0.4638 |
| ma | 246 | 18.2216 | 20.6057 | 13.2853 | 22.0979 | +8.8125 |
| pa | 127 | 19.1066 | 19.4780 | 15.1684 | 15.3053 | +0.1368 |
| Root | 161 | 21.1423 | 20.7916 | 15.1877 | 14.7532 | -0.4345 |

Nhóm được xác định bằng các regex phụ tố hiện có trong script; không phải gold morphological annotations. Nhóm mi/ma/pa có thể giao nhau: **414** câu có ít nhất một prefix, **161** câu Root, **141** câu có từ hai prefix-category trở lên. **154** là số membership dư (`195+246+127−414`), không phải số câu đa phụ tố. Tổng marginal **568 không lớn hơn 575**.

Số BLEU nhóm mi- trước đây ghi `21.50 → 23.32` không khớp phép tính trên predictions hiện có; chuẩn là **17.8639 → 18.4293**. Root giảm chrF++ **-0.4345**; không gọi là “trong sai số thống kê” khi chưa có CI tương ứng.

## 9. Cosine — sửa bảng theo artifact đã lưu

Bảng ở §2.2 của `RESEARCH_INSIGHTS.md` được đồng bộ với [cosine_by_layer.csv](../results/analysis/cosine_by_layer.csv). Đã kiểm tra lại phép lấy mean từ [cosine_by_sentence.csv](../results/analysis/cosine_by_sentence.csv): 9.200 hàng, 575 câu/tầng/mô hình, sai số tổng hợp < 1e-12. Không chạy lại encoder hoặc nạp trọng số.

| Encoder layer | mT5 Baseline | mT5 CLRR-Enc | Δ (CLRR − Baseline) |
| ---: | ---: | ---: | ---: |
| 1 | 0.3319 | 0.3345 | +0.0026 |
| 2 | 0.4768 | 0.4800 | +0.0032 |
| 3 | 0.7131 | 0.6387 | -0.0745 |
| 4 | 0.7929 | 0.6539 | -0.1391 |
| 5 | 0.8602 | 0.7422 | -0.1180 |
| 6 | 0.9092 | 0.8143 | -0.0949 |
| 7 | 0.9271 | 0.8346 | -0.0926 |
| 8 | 0.9519 | 0.8745 | -0.0774 |

Tầng cuối từng ghi **0.6277 → 0.6033**; artifact đúng là **0.9519 → 0.8745** (Δ **-0.0774**). Đây là mean within-sentence pairwise token cosine của mT5, không phải metric dịch hay kiểm chứng affix. Padding/special tokens bị loại, self-pairs bị loại; lấy mean theo câu, source max length 256. Không so trực tiếp với cosine pooled-sentence hoặc phép đo có mask/aggregation khác.

## 10. Source metric đã sửa

Shared callback `src/amis_rewire/metrics.py::build_compute_metrics` decode **predictions only**, snapshot references gốc. Năm entrypoints amis_rewire/PEFT/NLLB/Middle-Align/LayerSkip bind validation.csv khi train, rebind test.csv trước predict; JSON báo evaluation protocol. Thiếu reference hoặc sai count gây lỗi thay vì tính điểm trên split sai. Bốn regression tests pass: không decode labels, giữ prediction rỗng, không bị mutation refs, phát hiện count/source-order mismatch và từ chối reference dataset khác. `run_no_stop_gradient.py` cũng dùng split tương ứng.

Ba runners tổng hợp comparative/PEFT/Strong LoRA mBART bỏ hardcoded scores: historical references đọc bảng canonical và kiểm tra SHA-256 của test.csv; new results tính lại từ saved predictions/CSV gốc. `.gitignore` cho phép đưa bảng canonical nhỏ vào repo để runner không thiếu nguồn điểm khi sync. Canonical table không thay predictions hoặc checkpoints.

Historical checkpoints vẫn được giữ nguyên: không retroactively chọn lại best checkpoint bằng metric validation mới. Test scores được re-score từ outputs đã lưu, không generate lại.

## 11. Paired bootstrap mới

[42 tests](../outputs_rebuttal/metric_audit_20261002/full/paired_bootstrap_standardized.csv), **10.000** samples, seed **42**, paired sentence indices; corpus-score statistics, không mean sentence scores. CI là percentile95% signed delta; p-value là centered absolute-difference bootstrap của SacreBLEU. Hai phép tính không phải inversion nên CI/p có thể khác nhau ở biên. Holm correction toàn bộ 42 tests; protocol xác định trước khi tính, nhưng lựa chọn family này là **post-hoc audit**, không phải preregistration. Bốn phép p-value đối chiếu public `PairedTest` API khớp.

mBART full − baseline: BLEU Δ+0.7752 CI[-0.4388,2.0441], raw p=0.0954; chrF++ Δ+5.0277 CI[-0.1872,13.2525], raw p=0.1214; Holm đều1.0000. Full − LSR-only raw p=0.0888/0.0798, Holm đều1.0000. NLLB full − baseline raw p=0.0573/0.0466, Holm1.0000/0.8853. Không gắn significance marker lên các đối sánh này.

CLRR+LSR − Strong LoRA A/B có Holm p<0.01 cho cả hai metrics và backbone. mT5 full/decoder − baseline cũng còn có ý nghĩa. Tất cả đối sánh Ash mBART với baseline không có ý nghĩa. Đây là uncertainty theo test sentences ở fixed checkpoints, không phải robustness qua training seeds; LR/HPO/stopping confounds vẫn tồn tại.

## 12. Mandarin chrF++ — độ nhạy word bigram

Reference có **3 word bigrams** trong **3/575 câu** theo internal tokenizer của CHRF. Baseline match0; full/LSR-only match1 ở index170. Đây là tính chất cấu hình metric chuẩn trên corpus này, không phải lỗi công thức.

Character-only chrF(w=0): baseline18.2288, full18.9195 (**+0.6907**). Diagnostic bỏ index170: chrF++ baseline13.9655, full14.3499 (**+0.3844**, thay vì +5.0277). **Bảng chính luôn giữ 575 câu**. Không chuyển metric sang TokenizerZh hoặc loại câu để có significance. Dữ liệu component ở [chrf_component_diagnostics.json](../outputs_rebuttal/metric_audit_20261002/full/chrf_component_diagnostics.json).

## 13. Coverage và số chưa chứng nhận

36 bộ predictions (bao gồm ByT5/LayerSkip lịch sử), 3 API–CLI metric checks, 4 bootstrap API checks và cosine mean 9.200 hàng đã kiểm tra. Cả hai `clrr_main.tex` đã cập nhật số và rút claims không đủ evidence. ByT5 full7.3108/8.0956 thấp hơn baseline7.5837/8.3057; không giấu kết quả âm tính.

**Chưa chứng nhận:** mT5 LSR-only; SVD/probeF1; no-sg test metric thiếu predictions; gradient cũ đo sai; sensitivity/validation sweep; latency/memory overhead; ví dụ định tính không khớp test. Những bảng/claims này bị rút khỏi kết luận hiện hành, snapshot giữ nguyên để truy vết. Không nạp weights hoặc train lại trong audit.

Metadata params/runtime kiểm tra từ reports, không rebuild model hoặc benchmark mới. NLLB StrongB train271.005.696:43.4496% trên active adapted623.724.544,30.5848% trên registered886.079.488 (gồm frozen storage copies). Không trộn hai mẫu số. Chi tiết và thay đổi [RESULTS_VERIFICATION.md](RESULTS_VERIFICATION.md).
