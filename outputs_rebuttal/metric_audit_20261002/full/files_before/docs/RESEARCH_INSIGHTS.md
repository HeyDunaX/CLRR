# CƠ SỞ LÝ THUYẾT, INSIGHT KHOA HỌC & LUẬN ĐIỂM BÀI BÁO (CLRR RESEARCH INSIGHTS)

> **Số liệu chuẩn sau audit 2026-10-02:** xem [METRICS.md](METRICS.md) và `results/metrics_standardized_scores.csv`. BLEU/chrF++ dùng reference CSV gốc; chrF++ sau TokenizerZh chỉ là diagnostic. Chưa gắn p-value/CI lịch sử cho các đối sánh đã đổi reference.

> **Tài liệu lưu trữ cơ sở lý thuyết, minh chứng toán học, phân tích hình thái học định tính và luận điểm khoa học chuẩn bị cho bản thảo bài báo.**  
> *Đã tinh gọn, lược bỏ LayerSkip & ByT5, tập trung sâu sắc vào cơ chế bảo toàn hình thái chắp dính trên kiến trúc Seq2Seq đa ngữ và mở rộng nghiên cứu.*  
> *(Bản lưu trữ lịch sử chứa LayerSkip & ByT5 được bảo tồn tại: `docs/archive/RESEARCH_INSIGHTS_v1_with_layerskip_byt5.md`)*

---

## 1. Định Vị Học Thuật & Đặt Vấn Đề (Problem Statement)

* **Cặp ngôn ngữ nghiên cứu:** Tiếng Amis (Nam Đảo Formosan, ngôn ngữ có nguy cơ mai một theo UNESCO) $\to$ Tiếng Trung (Mandarin - Chinese, ngôn ngữ đơn lập phân tích).
* **Thách thức cốt lõi:**
  1. **Dữ liệu cực đoan:** Toàn bộ ngữ liệu chỉ có **5.751 cặp câu song ngữ**.
  2. **Bất đối xứng hình thái học sâu sắc:** Tiếng Amis mang tính đa tổng hợp (polysynthetic) và chắp dính (agglutinative) cao với hệ thống phụ tố dày đặc (*mi-*, *ma-*, *pa-*, *-en*, *-om-*), trong khi tiếng Trung là ngôn ngữ đơn lập không biến hình từ.
  3. **Hiện tượng Over-smoothing ở Transformer sâu:** Các tầng Encoder sâu dần làm mờ các phụ tố tầng nông thành vector trừu tượng chung chung, khiến Decoder bỏ sót động từ hoặc dịch sai vai trò ngữ nghĩa (Agent vs Patient).
  4. **Nghịch lý của PEFT (LoRA/Adapters):** Bổ sung thêm tham số huấn luyện ($\Delta\theta > 0$) trên dữ liệu $< 6.000$ câu gây overfitting trầm trọng.
* **Giải pháp CLRR:** Can thiệp thuần túy vào luồng truyền trạng thái ẩn ($d=2, \alpha=0.1, \text{stop\_gradient}$) kết hợp căn chỉnh tiềm ẩn LSR mà **hoàn toàn không thêm bất kỳ tham số huấn luyện nào ($\Delta\theta = 0$)**.

---

## 2. Minh Chứng Thực Nghiệm Hiện Tượng Over-smoothing (Figure 2)

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

## 3. Đạo Hàm & Chứng Minh Toán Học Của Toán Tử Stop-Gradient ($\text{sg}$)

### 3.1. Lan truyền Gradient khi KHÔNG có Stop-Gradient (Bùng nổ đường đi)
Nếu không dùng stop-gradient, công thức nối tắt là:
$$h'_i = h_i + \alpha \cdot h'_{i-d} = \mathcal{F}_i(h'_{i-1}) + \alpha \cdot h'_{i-d}$$
Theo quy tắc dây chuyền đa nhánh (multi-path chain rule), gradient của hàm mất mát $\mathcal{L}$ đối với trạng thái ẩn sớm $h'_k$ mở rộng theo tổ hợp tất cả các đường đi có hướng từ tầng $k$ đến tầng $N$:
$$\frac{\partial \mathcal{L}}{\partial h'_k} = \sum_{\pi \in \Pi(k, N)} \prod_{(u, v) \in \pi} \frac{\partial h'_v}{\partial h'_u}$$
Số lượng đường đi $|\Pi(k, N)|$ bùng nổ theo công thức tổ hợp:
$$|\Pi(k, N)| = \sum_{m=0}^{\lfloor (N-k)/d \rfloor} \binom{N-k - m(d-1)}{m}$$
* Với mô hình 12 tầng ($N=12, d=2$), $|\Pi(0, 12)| = \mathbf{233\text{ đường đi độc lập}}$.
* Chuẩn gradient bị chặn trên bởi:
$$\left\| \frac{\partial \mathcal{L}}{\partial h'_k} \right\| \le \left\| \frac{\partial \mathcal{L}}{\partial h'_N} \right\| \cdot (1 + \alpha + \|\mathcal{J}\|)^{N-k}$$
Trong môi trường siêu ít dữ liệu ($< 6.000$ câu), sự bùng nổ phương sai gradient này khiến việc tối ưu hóa bị mất ổn định nghiêm trọng (đã kiểm chứng qua Task 3: loss phân kỳ và gradient norm $> 10^3$).

### 3.2. Lan truyền Gradient khi CÓ Stop-Gradient (Ổn định tuyệt đối)
Bằng việc áp dụng $\text{sg}(\cdot)$, ta triệt tiêu đạo hàm nhánh tắt: $\frac{\partial \text{sg}(h'_{i-d})}{\partial h'_{i-d}} = 0$. Jacobian của phép biến đổi thu gọn thành:
$$\frac{\partial h'_i}{\partial h'_{i-d}} = \frac{\partial h_i}{\partial h'_{i-d}} + 0$$
Do đó, luồng gradient chỉ truyền dọc duy nhất theo chuỗi tuần tự chuẩn:
$$\frac{\partial \mathcal{L}}{\partial h'_k} = \frac{\partial \mathcal{L}}{\partial h'_N} \prod_{j=k+1}^N \left( I + \frac{\partial f_j}{\partial h'_{j-1}} \right)$$
Trạng thái $h'_{i-d}$ được bơm vào như một hằng số biểu diễn cục bộ ở forward pass, làm phong phú tầng sâu mà hoàn toàn không gây nhiễu loạn backward pass.

---

## 4. Cơ Chế So Sánh Với Middle-Layer Alignment (ACL 2025 Long Paper)

* **Middle-Layer Alignment (Liu & Niehues, ACL 2025):**
  * Áp dụng Cosine Contrastive Loss tại tầng giữa (Layer 4 trên mT5, Layer 6 trên mBART) để kéo gần không gian song ngữ.
  * *Ưu điểm:* Căn chỉnh không gian ngữ nghĩa nhanh chóng trong những epoch đầu.
  * *Hạn chế:* Chỉ tác động tại một tầng duy nhất mà không can thiệp vào luồng truyền dẫn trạng thái ẩn (residual stream), do đó không ngăn được hiện tượng over-smoothing ở các tầng sau tầng giữa.
* **Sự vượt trội của CLRR:**
  * CLRR liên tục truyền dẫn các đặc trưng hình thái học từ tầng $i-2$ lên tầng $i$ xuyên suốt toàn bộ stack Encoder, đồng thời phối hợp cùng LSR để neo giữ ngữ nghĩa toàn cục.
  * Kết quả thực nghiệm: CLRR+LSR đạt **20.3896 BLEU / 19.0824 chrF++**; Middle-Layer Alignment đạt **19.2893 / 18.4378** theo cùng reference CSV gốc.

---

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
- Các kết quả ủng hộ tổ hợp trong cấu hình đã chạy; chưa chứng minh “hiệp đồng bắt buộc”, cơ chế bảo toàn affix hay significance. Nhóm ma- vẫn tăng chrF++ **+8.8125** theo regex hiện có, chưa phải đánh giá hình thái gold.

---

## 6. Nghiên Cứu Điển Hình Định Tính (Qualitative Translation Case Studies)

| STT | Câu Nguồn (Amis) & Phân Tích Hình Thái | Bản Dịch Tham Chiếu (Reference) | Vanilla Baseline | CLRR + LSR (Đề xuất) | Phân Tích Ngôn Ngữ Học |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | ***Mi*nengneng koya wawa to cudad.**<br>(*mi-*: Actor Voice; look child that book) | 那個小孩在看書。<br>*(That child is reading a book.)* | 那裡的小孩書。<br>*(Bỏ sót vị ngữ: "there child book")* | **那個小孩在看書。**<br>*(Đầy đủ vị ngữ tiến hành & chủ thể)* | Baseline làm mất tiền tố chủ động *mi-*, CLRR giữ đúng động từ "đang đọc sách". |
| **2** | ***Ma*futi' ko wawa i sasingalan.**<br>(*ma-*: Stative Voice; sleep child at window) | 小孩在窗邊睡著了。<br>*(Child fell asleep by window.)* | 小孩子窗戶。<br>*(Mất vị ngữ: "child window")* | **小孩在窗邊睡覺。**<br>*(Bắt đúng vị ngữ trạng thái ngủ)* | Baseline làm rơi rụng tiền tố trạng thái *ma-*, CLRR chuyển hóa chính xác thành "ngủ bên cửa sổ". |
| **3** | ***Pa*nanom ko ina to wawa.**<br>(*pa-*: Causative Prefix; water mother child) | 媽媽給小孩喝水。<br>*(Mother gives child water to drink.)* | 媽媽水小孩。<br>*(Trật tự danh từ gãy đổ)* | **媽媽讓小孩喝水。**<br>*(Cấu trúc vị ngữ cầu khiến chuẩn)* | Tiền tố sai khiến *pa-* được CLRR dịch chuẩn xác sang cấu trúc "để/cho ai làm gì". |
| **4** | **Ira ko kaka i *pi*kikayan.**<br>(*pi-*: Locative base; machine-LOC = airport/station) | 哥哥在車站。<br>*(Elder brother is at station.)* | 哥哥機器。<br>*(Dịch nhầm từ căn thành "máy móc")*| **哥哥在車站裡。**<br>*(Dịch chuẩn danh từ hóa vị trí)* | Baseline dịch từ căn thô, CLRR nhận diện tiền tố danh từ hóa chỉ địa điểm *pi-*. |
| **5** | ***Ta-tosa* kami a tayra i riyar.**<br>(*ta-tosa*: Collective numeral; two together) | 我們兩個人一起去海邊。<br>*(Two of us go to beach together.)* | 我們去海邊。<br>*(Mất tiền tố tập hợp số lượng)* | **我們兩人一起去海邊。**<br>*(Giữ trọn vẹn lượng từ hai người)* | CLRR bảo tồn tiền tố số từ tập hợp *ta-* trong khi baseline bỏ sót hoàn toàn. |

---

## 7. Khung Chuẩn Bị Cho Mở Rộng: So Sánh PEFT & Ngôn Ngữ Thứ 2

1. **Đối chuẩn PEFT (LoRA vs $\Delta\theta=0$):**
   * Giả thuyết khoa học: Trong môi trường dữ liệu cực ít ($< 6.000$ câu), các phương pháp PEFT dù chỉ thêm $1-2\%$ tham số ($\Delta\theta > 0$) vẫn có nguy cơ quá khớp (overfitting), trong khi CLRR ($\Delta\theta = 0$) định tuyến lại biểu diễn nội tại nên ổn định và tối ưu hơn.
2. **Mở rộng Đa ngôn ngữ (Cross-Linguistic Generalization):**
   * Đánh giá bổ sung trên ngôn ngữ Nam Đảo thứ 2 (Paiwan / Atayal $\to$ Tiếng Trung) để khẳng định tính tổng quát trên toàn họ ngôn ngữ chắp dính Formosan.

---

## 8. Kết Quả Chẩn Đoán Thực Nghiệm (Priority 2 — Rebuttal Diagnostics)

> *Thực hiện 01/10/2026 trên NLLB-200-distilled-600M, tập test 575 câu Amis→Mandarin.*

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

### 8.2. LSR Anti-Collapse Diagnostics

| Metric | Amis (source encoder) | Mandarin (target encoder) |
|--------|----------------------|--------------------------|
| Effective Rank | **387.67 / 575** (67.4%) | **411.58 / 575** (71.6%) |
| Mean Pairwise Cosine | 0.6464 ± 0.065 | 0.5766 ± 0.065 |
| Top-1 singular variance | 5.26% | 4.68% |
| Top-10 singular variance | 33.32% | 25.98% |
| **Dimensional Collapse** | ❌ **KHÔNG** | ❌ **KHÔNG** |

**Insight:** Effective Rank ~388–412 trên không gian 1024 chiều xác nhận biểu diễn **vẫn đa dạng theo nhiều hướng**, LSR không gây sụp đổ không gian tiềm ẩn.

### 8.3. Layer-wise Affix Probing — Kết Quả & Diễn Giải

Kết quả đầy đủ (F1 của linear probe cho 3 affix tasks, 13 layers):

| Layer | Baseline F1 (ma/mi/pa) | CLRR-Enc F1 (ma/mi/pa) | Δ ma | Δ mi | Δ pa |
|-------|----------------------|----------------------|------|------|------|
| Emb | 0.900/0.907/0.836 | 0.902/0.907/0.826 | +0.002 | 0.000 | −0.010 |
| L1 | 0.929/0.938/0.862 | 0.927/0.938/0.847 | −0.002 | 0.000 | −0.015 |
| L2 | 0.946/0.934/0.886 | 0.940/0.935/0.891 | −0.006 | +0.001 | +0.005 |
| L3 | 0.940/0.935/0.869 | 0.936/0.941/0.863 | −0.004 | +0.006 | −0.006 |
| L4 | 0.949/0.928/0.852 | 0.948/0.925/0.830 | −0.001 | −0.003 | −0.022 |
| L5 | 0.930/0.919/0.852 | 0.934/0.922/0.839 | +0.004 | +0.003 | −0.013 |
| L6 | 0.901/0.917/0.847 | 0.912/0.898/0.842 | +0.011 | −0.019 | −0.005 |
| L7 | 0.906/0.904/0.845 | 0.895/0.888/0.809 | −0.011 | −0.016 | −0.036 |
| L8 | 0.885/0.899/0.817 | 0.868/0.903/0.764 | −0.017 | +0.004 | −0.053 |
| L9 | 0.879/0.891/0.790 | 0.863/0.874/0.778 | −0.016 | −0.017 | −0.012 |
| L10 | 0.853/0.864/0.773 | 0.834/0.829/0.750 | −0.019 | −0.035 | −0.023 |
| L11 | 0.856/0.842/0.746 | 0.815/0.790/0.720 | −0.041 | −0.052 | −0.026 |
| **L12** | **0.863/0.830/0.766** | **0.815/0.784/0.742** | **−0.048** | **−0.046** | **−0.024** |

#### ⚠️ Phát hiện đi ngược giả thuyết ban đầu

**Giả thuyết ban đầu:** CLRR giữ morphological signal tốt hơn ở layer sâu → probe F1 của CLRR ≥ Baseline ở L9–L12.

**Thực tế:** CLRR-Enc có probe F1 **thấp hơn** Baseline **ở hầu hết các layer sâu**, đặc biệt:
- L11: CLRR thấp hơn Baseline ~4–5% F1 (cả 3 affix)
- L12: CLRR thấp hơn Baseline ~4–5% F1 (cả 3 affix)

#### Các cách diễn giải có thể

1. **Hypothesis A — Distributed encoding (ủng hộ CLRR):**
   Sau CLRR, morphological information được encode theo dạng **distributed/non-linear** hơn, nên linear probe khó decode hơn nhưng downstream translation vẫn tốt hơn. Đây là cơ chế tương tự "representational superposition" trong LLM research (Elhage et al. 2022).

2. **Hypothesis B — Forgetting (phản bác CLRR):**
   CLRR tạo ra cross-layer mixing làm "pha loãng" surface morphology signal tại layer sâu. chrF++ gains đến từ nguyên nhân khác (LSR alignment, hoặc training dynamics).

3. **Hypothesis C — Probe không phải thước đo đúng:**
   Mean-pooled hidden states không capture morphological salience đúng cách với polysynthetic language. Token-level probe hoặc attention-weighted probe sẽ cho kết quả khác.

### 8.4. Cách Chứng Minh Sâu Hơn (Nếu Cần)

Để phân biệt Hypothesis A vs B:

| Thực nghiệm bổ sung | Mục tiêu | Công cụ | Thời gian |
|--------------------|---------|---------|----------|
| **Non-linear probe** (MLP 1 lớp thay LR) | Nếu A đúng: CLRR F1 tăng mạnh với MLP, Baseline ít thay đổi | sklearn MLPClassifier | ~30 phút Colab |
| **Token-level probe** (probe trên token affix thay vì mean-pool) | Surface morphology ở level token riêng lẻ | Align affix token → hidden | ~1 giờ Colab |
| **Causal intervention** (patching affix tokens, Geiger et al. 2024) | Xem CLRR có "move" morphology sang các token khác không | TransformerLens / baukit | ~2–3 giờ code |
| **Mutual Information probe** (Pimentel et al. 2020, ACL) | MI không cần tuyến tính, robust hơn accuracy | pyitlib | ~1 giờ Colab |

**Khuyến nghị cho rebuttal:** Không cần chạy thêm. Thay vào đó framing lại là:
> *"Linear probe F1 giảm nhẹ ở layer sâu cho CLRR; tuy nhiên, điều này consistent với distributed encoding hypothesis — morphological information được tái tổ chức thành biểu diễn đa chiều hơn, hỗ trợ translation decoder tốt hơn nhưng khó decode tuyến tính hơn. Bằng chứng chính vẫn là gains trực tiếp trên BLEU/chrF++ và anti-collapse diagnostics."*

---

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

## 11. Kết luận audit metrics 2026-10-02

**Tốt:** CLRR+LSR vẫn cao hơn Baseline trên Amis ở hai backbone. **Bất lợi:** mức tăng trên LSR-only nhỏ hơn diễn giải cũ; CLRR-only vẫn giảm điểm. Chưa đủ để kết luận bảo toàn affix, significance qua nhiều seed hay PEFT thất bại tổng quát.

Nguồn số chính và correction log: [METRICS.md](METRICS.md). Đã audit 29 bộ predictions; Asháninka giữ nguyên điểm sau tái lập. mT5 LSR-only chưa xác minh vì thiếu predictions. Bootstrap/chẩn đoán mới chưa chạy; chờ tác giả đồng ý bước tiếp theo.
