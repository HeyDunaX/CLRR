# CƠ SỞ LÝ THUYẾT, INSIGHT KHOA HỌC & LUẬN ĐIỂM BÀI BÁO (CLRR RESEARCH INSIGHTS)

> **Audit đầy đủ 2026-10-02:** 36 bộ predictions đã đối chiếu reference CSV gốc; 42 phép kiểm định paired bootstrap (10.000 mẫu, seed 42, Holm toàn bộ 42 tests). Xem [METRICS.md](METRICS.md), [RESULTS_VERIFICATION.md](RESULTS_VERIFICATION.md) và `results/metrics_standardized_scores.csv`. TokenizerZh chrF++ chỉ là diagnostic; checkpoint lịch sử được giữ nguyên.

> **Tài liệu lưu trữ cơ sở lý thuyết, minh chứng toán học, phân tích hình thái học định tính và luận điểm khoa học chuẩn bị cho bản thảo bài báo.**  
> *Đã tinh gọn, lược bỏ LayerSkip & ByT5, tập trung sâu sắc vào cơ chế bảo toàn hình thái chắp dính trên kiến trúc Seq2Seq đa ngữ và mở rộng nghiên cứu.*  
> *(Bản lưu trữ lịch sử chứa LayerSkip & ByT5 được bảo tồn tại: `docs/archive/RESEARCH_INSIGHTS_v1_with_layerskip_byt5.md`)*

---

## 1. Định Vị Học Thuật & Đặt Vấn Đề (Problem Statement)

* **Cặp ngôn ngữ nghiên cứu:** Tiếng Amis (Nam Đảo Formosan, ngôn ngữ có nguy cơ mai một theo UNESCO) $\to$ Tiếng Trung (Mandarin - Chinese, ngôn ngữ đơn lập phân tích).
* **Thách thức cốt lõi:**
  1. **Dữ liệu cực đoan:** Toàn bộ ngữ liệu chỉ có **5.751 cặp câu song ngữ**.
  2. **Bất đối xứng hình thái học sâu sắc:** Tiếng Amis mang tính đa tổng hợp (polysynthetic) và chắp dính (agglutinative) cao với hệ thống phụ tố dày đặc (*mi-*, *ma-*, *pa-*, *-en*, *-om-*), trong khi tiếng Trung là ngôn ngữ đơn lập không biến hình từ.
  3. **Giả thuyết over-smoothing:** cosine token mT5 tăng ở tầng sâu; chưa chứng minh phụ tố bị mất hay đây là nguyên nhân lỗi dịch.
  4. **Đối sánh PEFT:** cấu hình đã thử thấp hơn full-tuning/CLRR; chưa có bằng chứng nhân quả rằng thêm tham số gây overfitting.
* **Giải pháp CLRR:** Can thiệp thuần túy vào luồng truyền trạng thái ẩn ($d=2, \alpha=0.1, \text{stop\_gradient}$) kết hợp căn chỉnh tiềm ẩn LSR mà **hoàn toàn không thêm bất kỳ tham số huấn luyện nào ($\Delta\theta = 0$)**.

---

## 2. Chẩn đoán cosine token mT5 (Figure 2)

### 2.1. Phương pháp đo đạc định lượng
Đo lường bằng chỉ số **Average Pairwise Within-Sentence Token Cosine Similarity** qua từng tầng Encoder $l \in [1, 8]$ trên toàn bộ 575 câu tập Test:
$$\text{CosSim}(l) = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{h_{i,l}^\top h_{j,l}}{\|h_{i,l}\|_2 \|h_{j,l}\|_2}$$

### 2.2. Bảng số liệu thống kê chi tiết (`results/analysis/cosine_by_layer.csv`)

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

Audit đã đối chiếu mean từ `cosine_by_sentence.csv`: 575 câu/tầng, 9.200 hàng cho hai mô hình × 8 tầng; sai số so với CSV tổng hợp nhỏ hơn 1e-12. Bảng cũ ở tầng 2–8 không khớp artifact; tầng cuối được sửa từ 0.6277/0.6033 thành **0.9519/0.8745**, Δ = **-0.0774**.

Cosine được tính trên token không padding và không special token, loại self-pairs, rồi lấy trung bình theo câu; source bị truncate ở 256 token. Đây là mT5 CLRR-Enc, không phải phép đo mBART. Baseline tăng từ 0.3319 đến 0.9519; CLRR có cosine thấp hơn ở tầng 3–8, phù hợp với giảm mức đồng hướng token trong hai checkpoint đã đo. Chỉ số này chưa chứng minh giữ affix, tránh collapse hoặc nguyên nhân cải thiện dịch. Không nạp checkpoint hay chạy lại forward trong bước đồng bộ artifact này.

---

## 3. Stop-gradient — tính chất toán học và giới hạn bằng chứng

Với graph nén mỗi block thành một nút và nhánh đi 1 hoặc 2 tầng, số đường đi thỏa $P(q)=P(q-1)+P(q-2)$, $P(0)=P(1)=1$. Code cache **output block**, không cache embedding: L1→L12 có 11 khoảng cách, nên có **144** đường đi nếu không detach. Số **233** cũ đếm thêm một nút đầu không khớp implementation. Đây không phải số đường đi qua toàn bộ attention/residual nội bộ.

$\partial\,\mathrm{sg}(h)/\partial h=0$ loại đạo hàm qua nhánh CLRR được thêm. Forward vẫn đổi input của tầng sau nên Jacobian đường tuần tự có thể đổi; không suy ra gradient giống baseline, không nhiễu, hoặc hội tụ ổn định tuyệt đối.

Trace no-sg hiện có 18 giá trị norm bằng 0 vì callback cũ đọc **sau zero_grad**. Không có artifact cho norm >1000 hay baseline ổn định 15–35/18.4±4.2. Những khẳng định này được rút lại. Callback được sửa để lưu global pre-clip norm Trainer báo trong logs; chưa train lại.

## 4. So sánh Middle-Layer Alignment

mBART CLRR+LSR: **20.3896 BLEU / 19.0824 chrF++**; Middle-Layer Alignment: **19.2893 / 18.4378**. Chênh lệch không còn có ý nghĩa sau Holm (BLEU p=0.6131; chrF++ p=0.6659). mT5 decoder CLRR có BLEU cao hơn nhưng chrF++ thấp hơn Middle-Align; cả hai khác biệt không có ý nghĩa. Không dùng thứ hạng điểm làm bằng chứng giữ hình thái hoặc hạn chế bản chất của phương pháp đối chứng.

## 5. Insight Bóc Tách Thành Phần — sau audit metrics

| Phương pháp | BLEU | chrF++ chuẩn |
| --- | ---: | ---: |
| Baseline | 19.6144 | 14.0547 |
| CLRR-only | 18.4697 | 13.2443 |
| LSR-only | 19.6291 | 18.5753 |
| CLRR+LSR | 20.3896 | 19.0824 |

- CLRR+LSR − Baseline: **+0.7752 BLEU / +5.0277 chrF++ chuẩn**.
- CLRR+LSR − LSR-only: **+0.7606 BLEU / +0.5071 chrF++**. Chênh lệch raw chrF++ khoảng +3,01 trước đây trộn reference; chênh lệch chuẩn là +0.5071.
- LSR-only đã có mức tăng chrF++ **+4.5206** so với Baseline; không quy toàn bộ gain cho routing. CLRR-only thấp hơn Baseline.
- Full − LSR-only: raw p BLEU/chrF++ **0.0888 / 0.0798**, Holm đều 1.0000; CI đều cắt 0. Chưa chứng minh routing cải thiện trên LSR-only. Nhóm ma regex tăng **+8.8125 chrF++**, nhưng chịu ảnh hưởng word-bigram thưa và không phải gold morphology.

---

## 6. Ví dụ định tính — rút khỏi bằng chứng thực nghiệm

Năm câu minh họa trước đây không khớp nguồn nào trong test.csv (đối chiếu strip/lowercase), và không có provenance gắn với prediction thật. Bảng cũ được giữ ở snapshot audit; không dùng như kết quả dịch thật.

Ví dụ **có provenance** tại index 170 (0-based): source `Acaw aka han ko niradoman no mako hana!`; reference `不要在我剛挑的水中舀水好不好!`; baseline `不要在我剛挑的水中丟花!`; CLRR+LSR trùng reference. Đây là ví dụ dịch tốt hơn, không tự chứng minh bảo toàn affix.

## 7. Giới hạn đối sánh và mở rộng

CLRR không thêm tham số nhưng full-tuning backbone; đây không phải phương pháp cập nhật ít tham số như LoRA. Kết quả Strong LoRA đã thử thấp hơn CLRR; chưa có matched HPO/multi-seed để kết luận PEFT thất bại tổng quát.

Asháninka→Spanish đã chạy mBART baseline/CLRR và NLLB baseline. Chưa chứng minh ưu thế trên cặp thứ hai; ByT5 lịch sử còn giảm điểm. Giữ các kết quả bất lợi trong audit, không chọn bộ dữ liệu theo điểm test có lợi.

## 8. Kết Quả Chẩn Đoán Thực Nghiệm (Priority 2 — Rebuttal Diagnostics)

> Trạng thái sau audit 2026-10-02: overlap regex tái lập được; SVD/probing thiếu artifact, không được coi là đã chứng nhận.

### 8.1. Kiểm Toán Hình Thái Học (Morphological Overlap Audit)

Phân vùng **Venn disjoint** của 575 câu test:

| Nhóm | Số câu | % |
|------|--------|---|
| Chỉ *mi-* (Actor Voice) | 86 | 15.0% |
| Chỉ *ma-* (Patient/Stative) | 138 | 24.0% |
| Chỉ *pa-* (Causative) | 49 | 8.5% |
| *mi-* + *ma-* | 63 | 10.9% |
| *mi-* + *pa-* | 33 | 5.7% |
| *ma-* + *pa-* | 32 | 5.6% |
| *mi-* + *ma-* + *pa-* | 13 | 2.3% |
| **Root/Simple (control)** | **161** | **28.0%** |
| **Tổng disjoint** | **575** | **100%** |

**Sửa cách đếm:** tổng marginal 568 không vượt 575; nó vượt 414 câu affixed duy nhất. Có **141 câu** thuộc từ hai nhóm prefix trở lên; **154** là membership dư, không phải số câu đa phụ tố.

### 8.2. SVD anti-collapse — chưa xác minh

Không tìm được raw representations/full singular values hoặc báo cáo SVD gốc trong repo hay inventory Hub. Effective Rank **387.67 / 411.58**, cosine và variance trong bảng cũ chưa được chứng nhận; rút kết luận “LSR không collapse”. Snapshot giữ số cũ để truy vết. Nếu matrix 575×1024 được center theo câu, rank tối đa là 574; phải ghi rõ định nghĩa/normalization. Script chỉ lưu 50 singular values đầu không đủ tái lập full effective rank.

### 8.3. Affix probing — chưa xác minh

Không tìm được CSV probing gốc cho bảng 13 layers × 3 tasks. Rút bảng F1 khỏi bằng chứng hiện hành và rút diễn giải distributed encoding như kết luận thực nghiệm. Nếu khôi phục được các F1 cũ, chiều giảm ở tầng sâu cũng không trực tiếp ủng hộ giả thuyết CLRR giữ affix tốt hơn.

### 8.4. Trước khi dùng diagnostics

Cần artifact, checkpoint provenance, model-family loader đúng và phép đo có thể tái lập. Loader có nhánh fallback/strict=False và diagnostic mBART có nhánh dùng cùng checkpoint cho hai hệ thống; các đường này chưa được chứng nhận bằng forward mới. Không chạy thêm trong audit này.

## 9. Strong LoRA Suite (Run A) — Kết Quả Đo Lường & Luận Điểm Đối Sách Học Thuật

> **Ngày ghi nhận:** 01/10/2026  
> **Cấu hình Run A:** `mBART-large-50` (611M), LoRA All-Linear ($r=16, \alpha=32$, dropout 0.05), can thiệp toàn bộ 6 ma trận tuyến tính (`q_proj, k_proj, v_proj, out_proj, fc1, fc2`) trên cả 12 tầng Encoder & 12 tầng Decoder. Đóng băng token embeddings. LR $2 \times 10^{-4}$, warmup 0.06, batch $4 \times 32 = 128$, seed 42. Đã hoàn thành 20 epochs trên GPU NVIDIA A100. Checkpoint tốt nhất tại epoch 19 (`checkpoint-684`).

### 9.1. Bảng So Sánh Đối Chuẩn — reference gốc

| Phương pháp | BLEU | chrF++ chuẩn |
| --- | ---: | ---: |
| BitFit | 0.3463 | 2.2552 |
| Narrow LoRA | 3.2365 | 4.4213 |
| Strong LoRA A | 10.2296 | 8.6319 |
| Strong LoRA B | 13.8066 | 10.3293 |
| Baseline | 19.6144 | 14.0547 |
| Middle-Layer Alignment | 19.2893 | 18.4378 |
| CLRR+LSR | 20.3896 | 19.0824 |

### 9.2. Diễn giải sau audit

Strong LoRA A − Narrow LoRA: **+6.9931 BLEU / +4.2105 chrF++**. CLRR+LSR − Strong LoRA A/B: **+10.1601 / +6.5830 BLEU**. Run A/B đạt 10.2296 / 13.8066 BLEU; A/B đổi LR nên chưa tách được tác động mở khóa embeddings.

Các cấu hình Strong LoRA cải thiện rõ so với Narrow LoRA nhưng chưa chứng minh “tối ưu kịch trần”. CLRR thêm zero parameters nhưng vẫn full-tuning toàn bộ backbone. Chưa có multi-seed và HPO budget tương đương để kết luận PEFT thất bại tổng quát.

### 9.3. Cách dùng số trong rebuttal

Dùng cùng reference CSV gốc và signatures trong [METRICS.md](METRICS.md). Không trộn Trainer-reference scores với raw-reference scores; không dùng p-values cũ cho đối sánh đã đổi reference. Chi tiết số trước/sau được lưu trong audit.

---

## 10. NLLB-200 Strong LoRA Suite (Run A & B) — Khép Góc Đối Sách PEFT & Hiện Tượng Phân Kỳ Embedding

> **Ngày ghi nhận:** 02/10/2026  
> **Backbone:** `facebook/nllb-200-distilled-600M` (615M parameters, 12 enc / 12 dec, shared embeddings 256.206 tokens).  
> **Cấu hình Run A:** LoRA All-Linear ($r=16, \alpha=32$, dropout 0.05), can thiệp toàn bộ 6 ma trận tuyến tính (`q_proj, k_proj, v_proj, out_proj, fc1, fc2`) trên cả 24 blocks. Đóng băng token embeddings. LR $2 \times 10^{-4}$, Effective Batch 128 ($16 \times 8$), seed 42. Đã hoàn thành 20 epochs trên GPU NVIDIA A100.  
> **Cấu hình Run B:** All-Linear ($r=16, \alpha=32$) kết hợp unfreeze toàn bộ ma trận nhúng và đầu ra (`shared, embed_tokens, lm_head`, 262M tham số). LR $1 \times 10^{-4}$, Effective Batch 128, seed 42. Đã hoàn thành 20 epochs trên GPU NVIDIA A100.

### 10.1. Bảng So Sánh Đối Chuẩn NLLB — metrics chuẩn

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

### 10.2. Diễn giải sau audit

Toàn bộ tám phương pháp NLLB trong bảng đã được kiểm tra bằng predictions và reference gốc. CLRR+LSR − Baseline: **+0.7818 BLEU / +0.4763 chrF++**. CLRR+LSR − Strong LoRA A/B: **+2.8195 / +4.4796 BLEU**.

NLLB A/B có BLEU 11.4625 / 9.8024; hướng thay đổi khác mBART (10.2296 / 13.8066). Việc đổi đồng thời embeddings và LR không cho phép quy chênh lệch riêng cho overfitting hoặc dispersion. Đây là quan sát thực nghiệm, không phải chứng minh cơ chế.

## 11. Kết luận sau xác minh đầy đủ

**Tốt:** các khác biệt CLRR+LSR so với Strong LoRA A/B trên mBART/NLLB còn có ý nghĩa sau Holm; mT5 CLRR+LSR và decoder variant vượt baseline có ý nghĩa.

**Bất lợi:** mBART/NLLB full − baseline và mBART full − LSR-only không có ý nghĩa sau correction. chrF++ mBART +5.0277 rất nhạy với một câu: reference chỉ có 3 word bigrams; index 170 là bigram match duy nhất của full/LSR-only. chrF character-only chỉ tăng **+0.6907**. Bỏ index 170 *chỉ để chẩn đoán* làm delta chrF++ còn **+0.3844**; bảng chính vẫn giữ đủ 575 câu.

Asháninka không có đối sánh nào vượt baseline có ý nghĩa; fixed epoch 20/patience 10 là post-hoc, khác quy tắc baseline. ByT5 full thấp hơn baseline. Bài yếu đi ở luận điểm superiority so với full-tuning và cơ chế affix/stability/anti-collapse; so sánh Strong LoRA còn giá trị trong phạm vi cấu hình đã chạy.

36 bộ predictions, 42 tests bootstrap, 4 đối chiếu public API; mT5 LSR-only/SVD/probing/latency/sensitivity vẫn chưa xác minh. Chi tiết [RESULTS_VERIFICATION.md](RESULTS_VERIFICATION.md). Không train, đổi checkpoint hay sinh lại bản dịch.
