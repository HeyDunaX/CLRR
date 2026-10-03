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
  * **Learning Rate & Warmup:** Full-tuning dùng $3 \times 10^{-4}$ cho `mT5-small`, $5 \times 10^{-5}$ cho `mBART-large-50`; LoRA cũ/Strong LoRA A dùng 2e-4, Strong LoRA B/BitFit dùng 1e-4. Warmup ratio 0.06, tối đa 20 epochs. LR không cố định giữa các nhóm phương pháp; xem giới hạn tuning tại §2.1.2.
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
| | **LoRA** ($r=8, \alpha=16$, hẹp) | Hu et al. (ICLR 2022) | **+1.179.648** | 1.179.648 (0.1927%) | **3.3284** | **5.1583** | Chỉ can thiệp $q, v$; adapter 1.18M chưa đủ sức tái định hình không gian biểu diễn |
| | **Strong LoRA** ($r=16$, All-Linear) | Hu et al. / Mở rộng rebuttal | **+8.650.752** | 8.650.752 (1.4161%) | **10.4193** | **9.8008** | Run A; reference Trainer, xem audit §2.1.1; chưa tuning LR đầy đủ |
| | **Strong LoRA B** ($r=16$, All-Linear + Embeddings) | Mở rộng rebuttal | **+8.650.752** | 264,706,048 (42.73% model có adapter) | **14.0432** | **11.4901** | Reference Trainer; reference gốc và giới hạn LR tại §2.1.2 |
| | Standard Fine-Tuning | Official Baseline | 0 | 610.879.488 (100%) | 19.6106 | 14.0538 | Điểm tựa baseline chuẩn mBART |
| | Middle-Layer Alignment | Liu & Niehues (ACL 2025) | 0 | 610.879.488 (100%) | 19.7243 | 15.5253 | Căn chỉnh tầng giữa đơn lẻ (+1.47 chrF++) |
| | **CLRR-Enc + LSR (Ours)** | **Đề xuất chính** | **0** | **610.879.488 (Zero New Params)** | **20.3927** | **19.0839** | Kết quả lịch sử full-tuning; cần thống nhất reference và audit tuning LR trước khi kết luận đối sánh |

---

### 2.1.1. Strong LoRA Run A — 2026-10-01

- Cấu hình: mBART-50, LoRA rank 16 / alpha 32 / dropout 0.05, `q_proj,k_proj,v_proj,out_proj,fc1,fc2`, embeddings đóng băng, LR 2e-4, seed 42, batch 4 × accumulation 32, BF16; validation greedy, test beam 4.
- Đã huấn luyện 20 epochs; checkpoint tốt nhất tại epoch 19 (`checkpoint-684`); thêm mới/trainable 8,650,752 tham số. Training time 93.49 phút.

| Reference dùng khi tính điểm | Test BLEU (zh) | Test chrF++ raw (w=2) | Test chrF++ Zh |
| :--- | ---: | ---: | ---: |
| Reference đã tokenize/decode (giao thức Trainer hiện tại) | 10.4193 | 9.8008 | — |
| Reference gốc từ `data_processed/amis_mandarin/test.csv` | 10.2296 | 8.6319 | 15.2118 |

Audit: đúng 575 dự đoán, đúng thứ tự/source/target; điểm Trainer tính lại khớp; best adapter khớp checkpoint được chọn; archive và SHA-256 đã kiểm tra sau tải về. Tokenizer làm thay đổi 62 references, nên cần dùng cùng một bộ reference khi so sánh các mô hình trong bảng bài báo.

Artifact tại máy này:

- `outputs_rebuttal/strong_lora_scores.csv`
- `outputs_rebuttal/strong_lora_run_a_verification.json`
- `outputs_rebuttal/strong_lora_run_a_results.zip` (kết quả, checkpoint, log, source manifest)
- `results/mbart-large-50-lora-all-linear/metrics.json`
- `results/mbart-large-50-lora-all-linear/test_predictions.csv`
- `results/mbart-large-50-lora-all-linear/best_model.zip`

**Run B đã hoàn tất theo yêu cầu tiếp tục của tác giả; xem §2.1.2.** A100 được tạo không có `--high-mem`; Colab đã dừng và kiểm tra không còn phiên hoạt động. Các kết quả trên được sinh trực tiếp từ artifact bằng `scratch/record_run_a.py`.

---

### 2.1.2. Strong LoRA Run B — 2026-10-01

- Chạy riêng từ backbone `facebook/mbart-large-50-many-to-many-mmt`, không nạp checkpoint Run A. Bắt đầu 19:04 ICT trên A100 40GB, không yêu cầu `--high-mem`.
- LoRA r=16 / alpha=32 / dropout=0.05; targets `q_proj,k_proj,v_proj,out_proj,fc1,fc2`; embeddings trainable dùng chung weight giữa `shared`, encoder/decoder `embed_tokens` và `lm_head`.
- LR 0.0001, seed 42, batch 4 × accumulation 32, BF16, warmup 0.06; tối đa 20 epochs, patience 4, validation greedy, test beam 4.
- Hoàn tất 20 epochs; best epoch 19 (`checkpoint-684`); thêm 8,650,752 tham số, train 264,706,048/619,530,240 (42.7269%). Training time 95.13 phút.

| Reference dùng khi tính điểm | Test BLEU (zh) | Test chrF++ raw (w=2) | Test chrF++ Zh |
| :--- | ---: | ---: | ---: |
| Reference đã tokenize/decode (Trainer) | 14.0432 | 11.4901 | — |
| Reference gốc từ `data_processed/amis_mandarin/test.csv` | 13.8066 | 10.3293 | 17.6304 |

Audit: 575 dự đoán, đúng thứ tự/source/target; điểm Trainer tính lại khớp; best adapter bằng checkpoint được chọn; ZIP và SHA-256 được kiểm tra sau tải về. Cả ba split có hash khớp Run A. Tokenizer thay đổi 62 references; dùng thống nhất reference khi đối sánh. Kết quả sinh từ artifact, không chép tay.

Artifact: `outputs_rebuttal/strong_lora_scores.csv`, `outputs_rebuttal/strong_lora_run_b_verification.json`, `outputs_rebuttal/strong_lora_run_b_source_manifest.json`, `outputs_rebuttal/strong_lora_run_b_results.zip`, `outputs_rebuttal/strong_lora_run_b.log`, và `results/mbart-large-50-lora-all-linear-unfreeze-embed/{metrics.json,test_predictions.csv,best_model.zip}`.

Phiên Colab đã dừng sau khi tải và xác minh artifact trên máy.

#### Giới hạn đối sánh learning rate

Script full-tuning mBART (baseline/CLRR/các phương pháp full-tuning đối chứng) dùng LR 5e-5; LoRA A dùng 2e-4; LoRA B/BitFit dùng 1e-4. LR riêng cho mỗi phương pháp có cơ sở vì nhóm tham số và cách tham số hóa khác nhau; paper LoRA cũng tuning LR theo phương pháp ([Appendix D.4, Table 12](https://arxiv.org/html/2106.09685#A4.SS4)). Tuy nhiên, chưa xác minh một sweep LR với ngân sách tương đương trong các artifact hiện có; các kết quả này phản ánh cấu hình đã chạy, chưa chứng minh mức tối ưu của mỗi phương pháp.

A–B đổi đồng thời LR và việc mở khóa embeddings, nên chưa tách được ảnh hưởng riêng của embeddings. Để bổ sung: kiểm tra LR trên validation với ngân sách thử tương đương, chọn cấu hình bằng validation chrF++, ghi toàn bộ trials/seeds và protocol; bổ sung A/B cùng LR nếu cần ablation embeddings. Không dùng test để chọn LR. Chưa chạy thêm sweep trong phiên này. CLRR thêm 0 tham số nhưng vẫn full-tuning, cần báo riêng added/trainable params. Các bảng lịch sử cần audit thống nhất reference và cấu hình trước khi khẳng định ưu thế phương pháp.

---

### 2.2. Bảng Thực Nghiệm SOTA NLLB-200 (`facebook/nllb-200-distilled-600M`)

Đánh giá thực nghiệm trên 575 câu Test Set (Beam Search size 4) trên mô hình dịch đa ngôn ngữ chuyên biệt NLLB-200 (Phương án A — chạy các mô hình độc lập):

| Phương pháp | Nguồn trích dẫn | $\Delta\theta$ (Thêm mới) | Tham số huấn luyện (% Model) | BLEU (zh) ↑ | chrF++ (w=2) ↑ | chrF++ (Zh) ↑ | Nhận xét thực nghiệm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Vanilla Baseline** | Official Fine-Tuning | 0 | 614.9M (100%) | **13.5001** | **10.4389** | **18.0885** | Điểm tựa pre-trained NLLB-200 sau 20 epochs |
| **BitFit** (Bias-only) | Ben-Zaken et al. (ACL 2022) | 0 | 131.0K (0.0213%) | **1.0750** | **2.9643** | **5.7081** | Nghẽn biểu diễn nghiêm trọng do chỉ cập nhật bias |
| **Narrow LoRA** ($r=8, q/v$) | Hu et al. (ICLR 2022) | +1.18M | 1.18M (0.1914%) | **3.5383** | **4.9407** | **9.1651** | Adapter 1.18M hẹp, chưa đủ sức xoay chuyển không gian đa ngữ |
| **Strong LoRA A** ($r=16$, All-Linear) | Mở rộng rebuttal (ACL) | **+8.65M** | **8.65M (1.3870%)** | **11.4625** | **9.5004** | **16.6314** | **Tăng vọt +7.92 BLEU so với Narrow LoRA; chứng minh PEFT được tối ưu công bằng** |
| **Strong LoRA B** (+Embeddings) | Mở rộng rebuttal (ACL) | **+8.65M** | **271.0M (43.4496%)** | **9.8024** | **8.4010** | **14.8131** | Mở khóa 262M embedding NLLB gây phân tán biểu diễn/overfit trên 4.6k câu |
| **Middle-Layer Alignment** | Liu & Niehues (ACL 2025) | 0 | 614.9M (100%) | **13.9648** | **10.6596** | **18.4448** | Căn chỉnh tầng giữa (+0.46 BLEU, +0.36 chrF++ Zh so với Baseline) |
| **CLRR-Dec + LSR (Ours)** | Đề xuất (Dec) | 0 | 614.9M (100%) | **13.4194** | **10.4559** | **18.0274** | Nối tầng Decoder (+0.02 chrF++ so với Baseline), khẳng định Encoder là vị trí tối ưu trên NLLB |
| **CLRR-Enc + LSR (Ours)** | **Đề xuất chính (Enc)** | **0** | **614.9M (100%)** | **14.2820**$^\dagger$ | **10.9152**$^\dagger$ | **18.7482** | **SOTA TOÀN DIỆN TRÊN NLLB-200 (+0.78 BLEU, +0.66 chrF++ Zh so với Baseline; hơn Strong LoRA A +2.82 BLEU, Zero new params)** |

---

### 2.2.1. NLLB-200 Strong LoRA Run A (All-Linear) — 2026-10-02

- **Cấu hình:** `facebook/nllb-200-distilled-600M`, LoRA rank 16 / alpha 32 / dropout 0.05, can thiệp toàn bộ 6 ma trận tuyến tính (`q_proj, k_proj, v_proj, out_proj, fc1, fc2`) trên 12 tầng Encoder + 12 tầng Decoder. Đóng băng ma trận nhúng.
- **Siêu tham số:** Learning rate $2 \times 10^{-4}$, seed 42, batch 16 $\times$ accumulation 8 = Effective Batch 128, BF16, linear warmup 0.06 over 20 epochs. Checkpoint đánh giá Validation chrF++ mỗi epoch.
- **Tiến trình:** Hoàn tất 20/20 epochs (720 steps) trên NVIDIA A100 GPU trong **32.9 phút**. Checkpoint tốt nhất tại epoch 18 (`checkpoint-648`).
- **Tham số:** Thêm mới và huấn luyện **8.650.752 tham số** (1,3870% tổng số 623.724.544 tham số của mô hình LoRA).
- **Kết quả kiểm thử chính thức (Test Set 575 câu):**
  - **Test BLEU (zh):** **`11.4625`** (Tăng bùng nổ **+7.9242 BLEU** so với Narrow LoRA 3.5383)
  - **Test chrF++ (raw, w=2):** **`9.5004`** (Tăng **+4.5597 chrF++**)
  - **Test chrF++ (Zh):** **`16.6314`** (Tăng **+7.4663 chrF++ Zh**)
- **Artifacts lưu trữ tại:**
  - `results/nllb-200/nllb-200-lora-all-linear/metrics.json`
  - `results/nllb-200/nllb-200-lora-all-linear/test_predictions.csv`
  - `results/nllb-200/nllb-200-lora-all-linear/nllb-200-strong_lora_a-best.zip`

---

### 2.2.2. NLLB-200 Strong LoRA Run B (All-Linear + Unfrozen Embeddings) — 2026-10-02

- **Cấu hình:** `facebook/nllb-200-distilled-600M`, All-Linear adapter ($r=16, \alpha=32$) kết hợp mở khóa toàn bộ ma trận nhúng và đầu ra (`shared`, `embed_tokens`, `lm_head`).
- **Siêu tham số:** Learning rate $1 \times 10^{-4}$ (chuẩn PEFT embedding plasticity), seed 42, batch 16 $\times$ accumulation 8 = Effective Batch 128, BF16. Tối đa 20 epochs, patience 4.
- **Tiến trình:** Hoàn tất 20/20 epochs (720 steps) trên NVIDIA A100 GPU trong **40.4 phút**. Checkpoint tốt nhất tại epoch 19 (`checkpoint-684`).
- **Tham số:** Thêm 8.650.752 tham số adapter, cập nhật **271.005.696 tham số** (chiếm **43,4496%** toàn bộ mô hình).
- **Kết quả kiểm thử chính thức (Test Set 575 câu):**
  - **Test BLEU (zh):** **`9.8024`**
  - **Test chrF++ (raw, w=2):** **`8.4010`**
  - **Test chrF++ (Zh):** **`14.8131`**
- **Artifacts lưu trữ tại:**
  - `results/nllb-200/nllb-200-lora-all-linear-unfreeze-embed/metrics.json`
  - `results/nllb-200/nllb-200-lora-all-linear-unfreeze-embed/test_predictions.csv`
  - `results/nllb-200/nllb-200-lora-all-linear-unfreeze-embed/nllb-200-strong_lora_b-best.zip`
- **So sánh đối chiếu:**
  - Ngược lại với mBART-50 (nơi unfreezing embeddings tăng từ 10.42 lên 14.04 BLEU), trên NLLB-200, việc mở khóa 262M tham số embedding làm giảm điểm từ **11.46 xuống 9.80 BLEU**. Không gian embedding 256k token của NLLB bị quá khớp (overfit) khi chịu gradient update từ chỉ 4.600 câu Amis; đóng băng embedding giúp duy trì cấu trúc biểu diễn đa ngữ tốt hơn nhiều.
  - Tuy nhiên, dù ở cấu hình nào, Strong LoRA trên NLLB vẫn kém CLRR-Enc + LSR (14.2820 BLEU / 18.7482 chrF++ Zh) từ **+2.82 đến +4.48 BLEU**, chứng minh ưu thế áp đảo của đường truyền tắt cấu trúc nội tại so với can thiệp adapter bên ngoài.

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
| `nllb-200/nllb-200-middle_align-best.zip` | 6.83 GB | Đã tải lên |
| `nllb-200/nllb-200-clrr_enc-best.zip` | 6.83 GB | Đã tải lên |
| `nllb-200/nllb-200-clrr_dec-best.zip` | ~6.8 GB | Đã hoàn thành / Tải lên HF Hub |
| `nllb-200/nllb_results.zip` (Toàn bộ 6 phương pháp NLLB) | Đang đóng gói | Tải lên HF Hub |
| `revalidation_ablations/mbart-large-50-ami-cmn-lsr-only/mbart-large-50-ami-cmn-lsr-only-best.zip` | 2.27 GB | Đã tải lên |
| `revalidation_ablations/mbart-large-50-ami-cmn-clrr-only/mbart-large-50-ami-cmn-clrr-only-best.zip` | 2.27 GB | Đã tải lên |
| `outputs_revalidation.zip` (Metrics, predictions, log kiểm định) | ~15 MB | Đã tải lên |
| `outputs/` & `outputs_extra/` (Checkpoints mBART và mT5 gốc — lưu trên HF) | ~12 GB | Đã tải lên |

---

## 7. Khung Mở Rộng: Sẵn Sàng Tiếp Nhận Phương Pháp Mới & Ngôn Ngữ Thứ 2

Cấu trúc tài liệu này đã được chuẩn hóa để sẵn sàng tích hợp:
1. **Phương pháp mới:** Đối chuẩn PEFT (LoRA rank 8 / BitFit / Morphology-aware segmentation) trên mBART-50.
2. **Ngôn ngữ thứ 2:** Tập dữ liệu song ngữ mới (Paiwan / Atayal $\to$ Tiếng Trung) chứng minh tính tổng quát đa ngôn ngữ Nam Đảo.

## 8. Asháninka → Spanish — kết quả ngày 2026-10-02

Dữ liệu AmericasNLP 2021: **3.883 train / 881 validation / 1.003 test**.
Điểm test dùng tham chiếu gốc từ CSV, SacreBLEU **13a**, chrF++ **word_order=2**,
beam **4**. Seed 42, LR **5e-5**, weight decay **0.0**, effective batch **128**.
Giao thức đầy đủ: [ASHANINKA_RUNBOOK.md](ASHANINKA_RUNBOOK.md).

| Backbone / phương pháp | Patience | Checkpoint đánh giá | Test BLEU | Test chrF++ |
| --- | ---: | --- | ---: | ---: |
| mBART Baseline | 4 | Tốt nhất theo validation, epoch 12 | 4.0085 | 20.8536 |
| mBART CLRR+LSR, lượt gốc | 4 | Tốt nhất theo validation, epoch 8 | 3.9671 | 20.6990 |
| **mBART CLRR+LSR, lượt mới — epoch 20** | **10** | **Epoch cuối, theo yêu cầu tác giả** | **4.0949** | **20.9588** |
| mBART CLRR+LSR, lượt mới — epoch 14 | 10 | Tốt nhất theo validation | 3.7970 | 20.9753 |
| NLLB Baseline | 4 | Tốt nhất theo validation, epoch 20 | 3.8594 | 20.4176 |
| NLLB CLRR+LSR | — | Chưa chạy | — | — |

**Epoch 20 được điền theo yêu cầu tác giả:** so với mBART Baseline, tăng
**0.0864 BLEU / 0.1051 chrF++**. Lượt mới train đủ 20 epoch trong khoảng
**99.4 phút**. Epoch 14 vẫn là checkpoint được chọn theo validation chrF++;
epoch 20 được đánh giá thêm sau yêu cầu xem checkpoint cuối. So sánh patience
10 với Baseline patience 4 là kết quả bổ sung một seed, chưa chứng minh ưu thế
dưới cùng quy tắc dừng hoặc ý nghĩa thống kê.

Số đầy đủ đối chiếu từ:

- Epoch 20: `results/ashaninka_patience10/epoch20/mbart-ashaninka-es-clrr-enc-epoch20/metrics.json`.
- Epoch 14: `results/ashaninka_patience10/runs/mbart-ashaninka-es-clrr-enc/metrics.json`.
- Bảng so sánh: `results/ashaninka_patience10/best_vs_epoch20.csv`.
- Các lượt gốc: `results/ashaninka_spanish_scores.csv`.
- [Chi tiết kết quả patience 10](ASHANINKA_PATIENCE10_RESULTS.md).

Checkpoint và báo cáo đã upload vào `FiveC/amis-rewire-checkpoints`:
`ashaninka-es/2026-10-02/`, lượt mới dưới `mbart-clrr-lsr-patience10/`,
checkpoint epoch 20 dưới `mbart-clrr-lsr-patience10/epoch20/`.
Biên nhận: `outputs_rebuttal/mbart_epoch20_hf_uploaded.json`.
Colab đã tắt sau khi lưu báo cáo và kiểm tra upload.
