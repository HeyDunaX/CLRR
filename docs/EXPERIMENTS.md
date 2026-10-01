# HỒ SƠ THỰC NGHIỆM & KẾT QUẢ CLRR (CLRR EXPERIMENTAL SUITE & METRICS)

> **Tài liệu lưu trữ chính thức toàn bộ số liệu thực nghiệm, bảng đối sánh, kiểm định bóc tách thành phần (Ablation), và đánh giá hình thái học.**  
> *Đã lược bỏ các đối chuẩn không phù hợp (LayerSkip & ByT5) để tập trung tối đa vào kiến trúc Seq2Seq chủ lực (`mBART-large-50` & `mT5-small`), sẵn sàng mở rộng thêm Method và Ngôn ngữ mới.*  
> *(Bản lưu trữ lịch sử chứa LayerSkip & ByT5 được bảo tồn tại: `docs/archive/EXPERIMENTS_v1_with_layerskip_byt5.md`)*

---

## 1. Dữ Liệu Song Ngữ & Giao Thức Công Bằng Tuyệt Đối (Fairness Protocol)

* **Ngữ liệu song ngữ chuẩn:** Amis (Pangcah) $\to$ Tiếng Trung (Mandarin - Chinese) từ ngữ liệu của *Zheng et al. (2022)*:
  * **Tổng cộng:** **5.751 cặp câu**
  * **Train:** **4.600 câu** (42.180 token Amis / 49.812 chữ Hán)
  * **Validation:** **576 câu** (5.314 token Amis / 6.290 chữ Hán)
  * **Test:** **575 câu** (5.286 token Amis / 6.244 chữ Hán)
* **Quy chuẩn siêu tham số cố định (Strict Fairness Protocol):**
  * **Random Seed:** Cố định 42 cho toàn bộ dataloader, model weights và CUDA runtime.
  * **Effective Batch Size:** 128 (Per-device batch size 16 $\times$ 8 gradient accumulation steps, hoặc 8 $\times$ 16).
  * **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$, weight decay 0.01).
  * **Learning Rate & Warmup:** $3 \times 10^{-4}$ cho `mT5-small`; $5 \times 10^{-5}$ cho `mBART-large-50`; Linear Warmup 0.06 qua 20 epochs (700-720 optimizer steps).
  * **Early Stopping:** Patience 4 epochs theo Validation chrF++.
  * **Precision:** Mixed Precision BF16 với TF32 trên GPU NVIDIA A100-SXM4-40GB.
  * **Độ đo chuẩn hóa:** 
    * **BLEU (zh):** `sacrebleu.corpus_bleu(preds, refs, tokenize='zh')`.
    * **chrF++ (raw, w=2):** `sacrebleu.corpus_chrf(preds, refs, word_order=2)` (Chỉ số chuẩn WMT/ACL).
    * **chrF++ (TokenizerZh):** Phân đoạn từ Hán ngữ để đánh giá tương thích.

---

## 2. Bảng Đối Sánh Chính (Main Comparative Matrix)

Đánh giá trên 575 câu Test Set (Beam Search size 4, length penalty 1.0):

| Backbone Architecture | Phương pháp | Nguồn trích dẫn | $\Delta\theta$ | BLEU (zh) | chrF++ (w=2) | chrF++ (TokenizerZh) | Vai trò & Ý nghĩa học thuật |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`facebook/mbart-large-50`**<br>(12 enc / 12 dec, 611M) | Standard Fine-Tuning | Baseline cơ sở | 0 | 19.6106 | 14.0538 | 22.7214 | Điểm tựa pre-trained mBART |
| | Middle-Layer Alignment | ACL 2025 Long | 0 | 19.7243 | 15.5253 | 22.8105 | Căn chỉnh tầng giữa (+1.47 chrF++) |
| | **CLRR-Dec + LSR (Ours)** | Đề xuất (Decoder) | 0 | **20.3986** | 16.1939$^\dagger$ | 23.1542 | Tối ưu hóa mạch sinh từ đích |
| | **CLRR-Enc + LSR (Ours)** | **Đề xuất chính (Enc)** | **0** | **20.3927** | **19.0839**$^\dagger$ | **23.5912** | **Đột phá hình thái học (+5.03 chrF++, $p < 0.001$)** |
| \midrule | | | | | | | |
| **`google/mt5-small`**<br>(8 enc / 8 dec, 300M) | Standard Fine-Tuning | Baseline cơ sở | 0 | 2.7887 | 3.8129 | 4.9521 | Điểm tựa pre-trained mT5 |
| | Middle-Layer Alignment | ACL 2025 Long | 0 | 4.7498 | **5.9373** | 6.8412 | Căn chỉnh tầng giữa (+1.96 BLEU) |
| | CLRR-Enc (Ours) | Đề xuất (Ablation) | 0 | 4.4393 | 5.1761 | 6.1204 | Nối tầng Encoder độc lập (+1.65 BLEU) |
| | CLRR-Enc + LSR (Ours) | Đề xuất (Encoder) | 0 | 4.5961$^\dagger$ | 5.0425$^\dagger$ | 6.0853 | Nối tầng Enc + Căn chỉnh tiềm ẩn |
| | **CLRR-Dec + LSR (Ours)** | **Đề xuất (Decoder)** | **0** | **4.8315**$^\dagger$ | 5.1873 | 6.2419 | **Đỉnh cao BLEU trên mT5 (+2.04 BLEU)** |

*Ghi chú:* $\dagger$ biểu thị cải thiện có ý nghĩa thống kê so với Baseline ($p_{\text{holm}} < 0.05$ qua 10.000 lần paired bootstrap resampling).

---

### 2.1. Bảng Đối Chuẩn Parameter-Efficient Fine-Tuning (PEFT Baselines on mBART-50)

Đánh giá thực nghiệm trên 575 câu Test Set (Beam Search size 4) nhằm giải quyết phản biện về tính hiệu quả tham số (Parameter-Efficiency):

| Backbone Architecture | Phương pháp | Nguồn trích dẫn | $\Delta\theta$ (Thêm mới) | Tham số huấn luyện (% Model) | BLEU (zh) ↑ | chrF++ (w=2) ↑ | Nhận định học thuật & Hiện tượng đo đạc |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`facebook/mbart-large-50`**<br>(12 enc / 12 dec, 611M) | **BitFit** (Bias-only) | Ben-Zaken et al. (ACL 2022) | **0** | 335.872 (0.0550%) | **0.3511** | **2.8464** | Đóng băng 99.95% backbone khiến mô hình không thể học biểu diễn ngôn ngữ unseen Amis |
| | **LoRA** ($r=8, \alpha=16$) | Hu et al. (ICLR 2022) | **+1.179.648** | 1.179.648 (0.1927%) | **3.3284** | **5.1583** | Chỉ bắt được vài từ vựng rời rạc; adapter 1.18M không đủ sức tái định hình không gian đa ngữ |
| | Standard Fine-Tuning | Official Baseline | 0 | 610.879.488 (100%) | 19.6106 | 14.0538 | Điểm tựa baseline chuẩn mBART |
| | Middle-Layer Alignment | Liu & Niehues (ACL 2025) | 0 | 610.879.488 (100%) | 19.7243 | 15.5253 | Căn chỉnh tầng giữa đơn lẻ (+1.47 chrF++) |
| | **CLRR-Enc + LSR (Ours)** | **Đề xuất chính** | **0** | **610.879.488 (Zero New Params)** | **20.3927** | **19.0839** | **Áp đảo hoàn toàn LoRA (+17.06 BLEU) và BitFit (+20.04 BLEU), chứng minh ưu thế tuyệt đối của $\Delta\theta=0$** |

---

### 2.2. Bảng Thực Nghiệm SOTA NLLB-200 (`facebook/nllb-200-distilled-600M`)

Đánh giá thực nghiệm trên 575 câu Test Set (Beam Search size 4) trên mô hình dịch đa ngôn ngữ chuyên biệt NLLB-200 (Phương án A — chạy các mô hình độc lập):

| Phương pháp | Nguồn trích dẫn | $\Delta\theta$ (Thêm mới) | Tham số huấn luyện (% Model) | BLEU (zh) ↑ | chrF++ (w=2) ↑ | chrF++ (Zh) ↑ | Nhận xét thực nghiệm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Vanilla Baseline** | Official Fine-Tuning | 0 | 614.9M (100%) | **13.5001** | **10.4389** | **18.0885** | Điểm tựa pre-trained NLLB-200 sau 20 epochs |
| **BitFit** (Bias-only) | Ben-Zaken et al. (ACL 2022) | 0 | 131.0K (0.0213%) | **1.0750** | **2.9643** | **5.7081** | Nghẽn biểu diễn nghiêm trọng do chỉ cập nhật bias |
| **LoRA** ($r=8, \alpha=16$) | Hu et al. (ICLR 2022) | +1.18M | 1.18M (0.1914%) | **3.5383** | **4.9407** | **9.1651** | Adapter 1.18M vượt trội BitFit nhưng tụt xa Baseline (-9.96 BLEU) |
| **Middle-Layer Alignment** | Liu & Niehues (ACL 2025) | 0 | 614.9M (100%) | **13.9648** | **10.6596** | **18.4448** | Căn chỉnh tầng giữa (+0.46 BLEU, +0.36 chrF++ Zh so với Baseline) |
| **CLRR-Enc + LSR (Ours)** | Đề xuất chính (Enc) | 0 | 614.9M (100%) | *Đang huấn luyện...* | *Đang huấn luyện...* | *Đang huấn luyện...* | Đang huấn luyện trên GPU A100 |
| **CLRR-Dec + LSR (Ours)** | Đề xuất (Dec) | 0 | 614.9M (100%) | *Xếp hàng* | *Xếp hàng* | *Xếp hàng* | Chạy sau CLRR-Enc |

---

## 3. Bảng Bóc Tách Thành Phần Độc Lập trên mBART-50 (Component Ablation Suite)

Nhằm làm sáng tỏ cơ chế đóng góp của từng thành phần:

| Cấu hình | Tham số can thiệp | $\Delta\theta$ | BLEU (zh) | chrF++ (w=2) | chrF (w=0) | chrF++ (TokenizerZh) | Đóng góp cơ chế |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Vanilla Baseline** | $\alpha=0, \lambda=0$ | 0 | 19.6106 | 14.0538 | 17.8421 | 22.7214 | Điểm tựa ban đầu |
| **CLRR-only (Ablation 1)** | $\alpha=0.1, \lambda=0$ | 0 | 18.4697 | 13.2443 | 17.2305 | 21.9045 | Nối tắt đơn lẻ; thiếu mỏ neo ngữ nghĩa trong không gian 611M |
| **LSR-only (Ablation 2)** | $\alpha=0, \lambda=0.1$ | 0 | **20.0828** | **16.0690** | **18.2514** | **22.7966** | Căn chỉnh tiềm ẩn; mỏ neo ngữ nghĩa (+0.47 BLEU / +2.02 chrF++) |
| **Full CLRR-Enc + LSR (Ours)** | $\alpha=0.1, \lambda=0.1$ | 0 | **20.3927** | **19.0839** | **19.8210** | **23.5912** | **Hiệp đồng hoàn hảo (Synergy): Đạt đỉnh cả BLEU và chrF++ (+5.03 chrF++, $p < 0.001$)** |

---

## 4. Bảng Bóc Tách Vị Trí Nối Tầng trên mT5-small (Rewiring Stack Ablation)

| Cấu hình | Vị trí Nối tắt (Rewire Stack) | $\lambda_{\text{align}}$ | BLEU (zh) | chrF++ (w=2) | Kết luận thực nghiệm |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Standard Baseline | Không | 0.0 | 2.7887 | 3.8129 | Điểm mốc chuẩn |
| CLRR-Enc | Chỉ Encoder | 0.0 | 4.4393 | 5.1761 | Nối tầng đơn lẻ tăng +1.65 BLEU, +1.36 chrF++ |
| LSR alone | Không | 0.1 | 4.5422 | 5.0118 | Căn chỉnh ngữ nghĩa tăng +1.75 BLEU |
| CLRR-Enc + LSR | Chỉ Encoder | 0.1 | 4.5961 | 5.0425 | Kết hợp trên Encoder |
| **CLRR-Dec + LSR** | **Chỉ Decoder** | **0.1** | **4.8315** | **5.1873** | **Tối ưu nhất cho mô hình sinh từ subword nhỏ** |
| CLRR-Both + LSR | Cả Encoder & Decoder | 0.1 | 3.4478 | 4.5752 | **Sụp đổ giao thoa (-1.15 BLEU, $p < 0.001$)** do làm hỏng cross-attention |

---

## 5. Kết Quả Kiểm Định Hình Thái Học & Tính Ổn Định

### 5.1. Phân Tích Tập Con Hình Thái Học Tiếng Amis (Morphological Breakdown)
Phân tích 575 câu test thành các tập con ngữ pháp dựa trên các tiền tố vị ngữ đặc trưng của tiếng Amis:

| Tập con Hình thái học | Số lượng câu | Vanilla chrF++ | CLRR chrF++ | Mức tăng ($\Delta$) | Ý nghĩa ngôn ngữ học |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Toàn bộ Test Set** | 575 | 14.05 | 19.08 | **+5.03** | Trung bình toàn cục |
| **Tiền tố *ma-* (Stative/Patient Voice)** | 246 | 13.29 | **22.10** | **+8.81** | **Bảo tồn vị ngữ trạng thái cực kỳ xuất sắc** |
| **Tiền tố *mi-* (Actor Voice)** | 195 | 14.91 | 15.38 | +0.47 | BLEU tăng từ 21.50 lên 23.32 (+1.82 BLEU) |
| **Tiền tố *pa-* (Causative)** | 127 | 15.17 | 15.31 | +0.14 | Giữ đúng quan hệ nhân quả |
| **Câu từ căn / Đơn giản (Root/No affix)**| 161 | 15.19 | 14.75 | -0.44 | Không đổi (trong khoảng sai số thống kê) |

### 5.2. Audit Hiện Tượng Tokenizer chrF++ trên mBART-50
* **Giải trình hiện tượng SacreBLEU:** chrF++ ($w=2$) mặc định dùng whitespace để tách từ. Tiếng Trung không có khoảng trắng nên word bigram = 0, kéo chrF++ raw xuống 14.05 (Vanilla) và 19.08 (CLRR).
* **Khi dùng phân đoạn từ tiếng Trung (`TokenizerZh`):**
  * Vanilla Baseline: **22.72 chrF++**
  * CLRR-Enc: **23.59 chrF++** (Đồng pha hoàn toàn với mức tăng BLEU 19.61 lên 20.39).

### 5.3. Minh Chứng Thực Nghiệm Stop-Gradient Tránh Bùng Nổ Gradient
* Huấn luyện `mT5-small` ở chế độ **`CLRR (no-sg)`** (bỏ `detach()`):
* Kết quả: Gradient norm bùng nổ vượt $10^3$ tại epoch 3-4, loss dao động mạnh và phân kỳ; trong khi `CLRR (with-sg)` giữ gradient norm ổn định hoàn toàn quanh mức 15–35.

---

## 6. Danh Mục Checkpoint Lưu Trữ trên Hugging Face Hub

Toàn bộ mô hình đã được lưu trữ an toàn tại repository chính thức:  
👉 **`FiveC/amis-rewire-checkpoints`**

| Checkpoint Path trên Hugging Face | Dung lượng | Trạng thái |
| :--- | :---: | :---: |
| `nllb-200/nllb-200-baseline-best.zip` | 6.84 GB | Đã tải lên |
| `nllb-200/nllb-200-bitfit-best.zip` | 1.46 GB | Đã tải lên |
| `nllb-200/nllb-200-lora-best.zip` | 18.9 MB | Đã tải lên |
| `nllb-200/nllb-200-middle_align-best.zip` | ~6.8 GB | Đã hoàn thành / Tải lên HF Hub |
| `revalidation_ablations/mbart-large-50-ami-cmn-lsr-only/mbart-large-50-ami-cmn-lsr-only-best.zip` | 2.27 GB | Đã tải lên |
| `revalidation_ablations/mbart-large-50-ami-cmn-clrr-only/mbart-large-50-ami-cmn-clrr-only-best.zip` | 2.27 GB | Đã tải lên |
| `outputs_revalidation.zip` (Metrics, predictions, log kiểm định) | ~15 MB | Đã tải lên |
| `outputs/` & `outputs_extra/` (Checkpoints mBART và mT5 gốc — lưu trên HF) | ~12 GB | Đã tải lên |

---

## 7. Khung Mở Rộng: Sẵn Sàng Tiếp Nhận Phương Pháp Mới & Ngôn Ngữ Thứ 2

Cấu trúc tài liệu này đã được chuẩn hóa để sẵn sàng tích hợp:
1. **Phương pháp mới:** Đối chuẩn PEFT (LoRA rank 8 / BitFit / Morphology-aware segmentation) trên mBART-50.
2. **Ngôn ngữ thứ 2:** Tập dữ liệu song ngữ mới (Paiwan / Atayal $\to$ Tiếng Trung) chứng minh tính tổng quát đa ngôn ngữ Nam Đảo.
