# HỒ SƠ THỰC NGHIỆM & KẾT QUẢ CLRR (CLRR EXPERIMENTAL SUITE & METRICS)

> **Số liệu chuẩn sau audit 2026-10-02:** xem [METRICS.md](METRICS.md) và `results/metrics_standardized_scores.csv`. BLEU/chrF++ dùng reference CSV gốc; chrF++ sau TokenizerZh chỉ là diagnostic. Chưa gắn p-value/CI lịch sử cho các đối sánh đã đổi reference.

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
    * **chrF++ chuẩn:** CHRF `char_order=6, word_order=2, beta=2` trên predictions và reference CSV gốc; không tiền xử lý TokenizerZh.
    * **chrF++ (TokenizerZh diagnostic):** chỉ dùng phân tích phụ; không thay chrF++ chuẩn hoặc gọi là phân đoạn từ hình thái.

---

## 2. Bảng Đối Sánh Chính (Main Comparative Matrix)

Đánh giá trên 575 câu Test Set (Beam Search size 4, length penalty 1.0):

| Backbone | Phương pháp | BLEU (zh) | chrF++ chuẩn |
| --- | --- | ---: | ---: |
| mBART | Baseline | 19.6144 | 14.0547 |
| mBART | Middle-Layer Alignment | 19.2893 | 18.4378 |
| mBART | CLRR-Dec+LSR | 19.9188 | 18.6999 |
| mBART | CLRR+LSR | 20.3896 | 19.0824 |
| mT5 | Baseline | 2.7887 | 3.8129 |
| mT5 | CLRR-only | 4.4393 | 5.1761 |
| mT5 | CLRR+LSR | 4.5961 | 5.0425 |
| mT5 | CLRR-Dec+LSR | 4.8315 | 5.1933 |
| mT5 | Middle-Layer Alignment | 4.6625 | 5.2103 |

*Ghi chú:* chưa gắn lại p-value/CI cho bảng chuẩn. Các dấu significance của bảng lịch sử không được chuyển sang đối sánh đã đổi reference.

---

### 2.1. Bảng Đối Chuẩn Parameter-Efficient Fine-Tuning (PEFT Baselines on mBART-50)

Đánh giá thực nghiệm trên 575 câu Test Set (Beam Search size 4) nhằm giải quyết phản biện về tính hiệu quả tham số (Parameter-Efficiency):

| Phương pháp | BLEU | chrF++ chuẩn |
| --- | ---: | ---: |
| BitFit | 0.3463 | 2.2552 |
| Narrow LoRA | 3.2365 | 4.4213 |
| Strong LoRA A | 10.2296 | 8.6319 |
| Strong LoRA B | 13.8066 | 10.3293 |
| Baseline | 19.6144 | 14.0547 |
| Middle-Layer Alignment | 19.2893 | 18.4378 |
| CLRR+LSR | 20.3896 | 19.0824 |

---

### 2.1.1. Strong LoRA Run A — 2026-10-01

- Cấu hình: mBART-50, LoRA rank 16 / alpha 32 / dropout 0.05, `q_proj,k_proj,v_proj,out_proj,fc1,fc2`, embeddings đóng băng, LR 2e-4, seed 42, batch 4 × accumulation 32, BF16; validation greedy, test beam 4.
- Đã huấn luyện 20 epochs; checkpoint tốt nhất tại epoch 19 (`checkpoint-684`); thêm mới/trainable 8,650,752 tham số. Training time 93.49 phút.

| Reference chính | Test BLEU (zh) | Test chrF++ chuẩn |
| --- | ---: | ---: |
| CSV gốc | 10.2296 | 8.6319 |

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

| Reference chính | Test BLEU (zh) | Test chrF++ chuẩn |
| --- | ---: | ---: |
| CSV gốc | 13.8066 | 10.3293 |

Audit: 575 dự đoán, đúng thứ tự/source/target; điểm Trainer tính lại khớp; best adapter bằng checkpoint được chọn; ZIP và SHA-256 được kiểm tra sau tải về. Cả ba split có hash khớp Run A. Tokenizer thay đổi 62 references; dùng thống nhất reference khi đối sánh. Kết quả sinh từ artifact, không chép tay.

Artifact: `outputs_rebuttal/strong_lora_scores.csv`, `outputs_rebuttal/strong_lora_run_b_verification.json`, `outputs_rebuttal/strong_lora_run_b_source_manifest.json`, `outputs_rebuttal/strong_lora_run_b_results.zip`, `outputs_rebuttal/strong_lora_run_b.log`, và `results/mbart-large-50-lora-all-linear-unfreeze-embed/{metrics.json,test_predictions.csv,best_model.zip}`.

Phiên Colab đã dừng sau khi tải và xác minh artifact trên máy.

#### Giới hạn đối sánh learning rate

Script full-tuning mBART (baseline/CLRR/các phương pháp full-tuning đối chứng) dùng LR 5e-5; LoRA A dùng 2e-4; LoRA B/BitFit dùng 1e-4. LR riêng cho mỗi phương pháp có cơ sở vì nhóm tham số và cách tham số hóa khác nhau; paper LoRA cũng tuning LR theo phương pháp ([Appendix D.4, Table 12](https://arxiv.org/html/2106.09685#A4.SS4)). Tuy nhiên, chưa xác minh một sweep LR với ngân sách tương đương trong các artifact hiện có; các kết quả này phản ánh cấu hình đã chạy, chưa chứng minh mức tối ưu của mỗi phương pháp.

A–B đổi đồng thời LR và việc mở khóa embeddings, nên chưa tách được ảnh hưởng riêng của embeddings. Để bổ sung: kiểm tra LR trên validation với ngân sách thử tương đương, chọn cấu hình bằng validation chrF++, ghi toàn bộ trials/seeds và protocol; bổ sung A/B cùng LR nếu cần ablation embeddings. Không dùng test để chọn LR. Chưa chạy thêm sweep trong phiên này. CLRR thêm 0 tham số nhưng vẫn full-tuning, cần báo riêng added/trainable params. Các bảng lịch sử cần audit thống nhất reference và cấu hình trước khi khẳng định ưu thế phương pháp.

---

### 2.2. Bảng Thực Nghiệm NLLB-200 (`facebook/nllb-200-distilled-600M`)

Đánh giá thực nghiệm trên 575 câu Test Set (Beam Search size 4) trên mô hình dịch đa ngôn ngữ chuyên biệt NLLB-200 (Phương án A — chạy các mô hình độc lập):

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

---

### 2.2.1. NLLB-200 Strong LoRA Run A (All-Linear) — 2026-10-02

- **Cấu hình:** `facebook/nllb-200-distilled-600M`, LoRA rank 16 / alpha 32 / dropout 0.05, can thiệp toàn bộ 6 ma trận tuyến tính (`q_proj, k_proj, v_proj, out_proj, fc1, fc2`) trên 12 tầng Encoder + 12 tầng Decoder. Đóng băng ma trận nhúng.
- **Siêu tham số:** Learning rate $2 \times 10^{-4}$, seed 42, batch 16 $\times$ accumulation 8 = Effective Batch 128, BF16, linear warmup 0.06 over 20 epochs. Checkpoint đánh giá Validation chrF++ mỗi epoch.
- **Tiến trình:** Hoàn tất 20/20 epochs (720 steps) trên NVIDIA A100 GPU trong **32.9 phút**. Checkpoint tốt nhất tại epoch 18 (`checkpoint-648`).
- **Tham số:** Thêm mới và huấn luyện **8.650.752 tham số** (1,3870% tổng số 623.724.544 tham số của mô hình LoRA).
- **Kết quả kiểm thử chính thức (Test Set 575 câu):**
  - **Test BLEU (zh):** **`11.4625`** (Tăng bùng nổ **+7.9242 BLEU** so với Narrow LoRA 3.5383)
  - **Test chrF++ (raw, w=2):** **`9.5004`** (Tăng **+4.5597 chrF++**)
  - **Test chrF++ (TokenizerZh diagnostic):** **`16.6314`** (Tăng **+7.4663 chrF++ Zh**)
- **Artifacts lưu trữ tại:**
  - `results/nllb-200/nllb-200-lora-all-linear/metrics.json`
  - `results/nllb-200/nllb-200-lora-all-linear/test_predictions.csv`
  - `results/nllb-200/nllb-200-lora-all-linear/nllb-200-strong_lora_a-best.zip`

---

### 2.2.2. NLLB-200 Strong LoRA Run B (All-Linear + Unfrozen Embeddings) — 2026-10-02

- **Cấu hình:** `facebook/nllb-200-distilled-600M`, All-Linear adapter ($r=16, \alpha=32$) kết hợp mở khóa toàn bộ ma trận nhúng và đầu ra (`shared`, `embed_tokens`, `lm_head`).
- **Siêu tham số:** Learning rate $1 \times 10^{-4}$ (cấu hình Run B đã chạy), seed 42, batch 16 $\times$ accumulation 8 = Effective Batch 128, BF16. Tối đa 20 epochs, patience 4.
- **Tiến trình:** Hoàn tất 20/20 epochs (720 steps) trên NVIDIA A100 GPU trong **40.4 phút**. Checkpoint tốt nhất tại epoch 19 (`checkpoint-684`).
- **Tham số:** Thêm 8.650.752 tham số adapter, cập nhật **271.005.696 tham số** (chiếm **43,4496%** toàn bộ mô hình).
- **Kết quả kiểm thử chính thức (Test Set 575 câu):**
  - **Test BLEU (zh):** **`9.8024`**
  - **Test chrF++ (raw, w=2):** **`8.4010`**
  - **Test chrF++ (TokenizerZh diagnostic):** **`14.8131`**
- **Artifacts lưu trữ tại:**
  - `results/nllb-200/nllb-200-lora-all-linear-unfreeze-embed/metrics.json`
  - `results/nllb-200/nllb-200-lora-all-linear-unfreeze-embed/test_predictions.csv`
  - `results/nllb-200/nllb-200-lora-all-linear-unfreeze-embed/nllb-200-strong_lora_b-best.zip`
- **So sánh đối chiếu:**
  - Ngược lại với mBART-50 (nơi Run A/B có BLEU 10.2296 / 13.8066), trên NLLB-200, việc mở khóa 262M tham số embedding làm giảm điểm từ **11.46 xuống 9.80 BLEU**. A/B đổi cả việc mở khóa embeddings và LR; chưa thể quy chênh lệch riêng cho embeddings hoặc kết luận cơ chế overfitting.
  - Tuy nhiên, dù ở cấu hình nào, Strong LoRA trên NLLB vẫn kém CLRR-Enc + LSR (14.2820 BLEU / 10.9152 chrF++ chuẩn) từ **+2.82 đến +4.48 BLEU**, cao hơn trong các cấu hình đã chạy; chưa có sweep tương đương hay multi-seed để kết luận tổng quát.

---

## 3. Bảng Bóc Tách Thành Phần Độc Lập trên mBART-50

| Phương pháp | BLEU | chrF++ chuẩn |
| --- | ---: | ---: |
| Baseline | 19.6144 | 14.0547 |
| CLRR-only | 18.4697 | 13.2443 |
| LSR-only | 19.6291 | 18.5753 |
| CLRR+LSR | 20.3896 | 19.0824 |

CLRR+LSR − LSR-only: **+0.7606 BLEU / +0.5071 chrF++**. LSR-only − Baseline: **+0.0146 BLEU / +4.5206 chrF++**. CLRR-only thấp hơn Baseline; các chênh lệch này chưa có bootstrap/multi-seed mới.

---

## 4. Bảng Bóc Tách Vị Trí Nối Tầng trên mT5-small

| Phương pháp | BLEU | chrF++ chuẩn |
| --- | ---: | ---: |
| Baseline | 2.7887 | 3.8129 |
| CLRR-only | 4.4393 | 5.1761 |
| CLRR+LSR | 4.5961 | 5.0425 |
| CLRR-Dec+LSR | 4.8315 | 5.1933 |
| CLRR-Both+LSR | 3.4459 | 4.5753 |

LSR-only: **chưa xác minh vì thiếu predictions**; không dùng số cũ 4.5422/5.0118 trong bảng hiện tại. CLRR-Dec có BLEU cao nhất trong các biến thể mT5 đã kiểm tra; Middle-Align có chrF++ 5.2103, cao hơn CLRR-Dec 5.1933. Không suy ra cơ chế cross-attention chỉ từ thứ hạng điểm.

---

## 5. Kết Quả Kiểm Định Hình Thái Học & Tính Ổn Định

### 5.1. Phân Tích Tập Con Hình Thái Học Tiếng Amis

| Tập con | N | Baseline BLEU | CLRR+LSR BLEU | Baseline chrF++ | CLRR+LSR chrF++ | Δ chrF++ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 575 | 19.6144 | 20.3896 | 14.0547 | 19.0824 | +5.0277 |
| mi | 195 | 17.8639 | 18.4293 | 14.9140 | 15.3779 | +0.4638 |
| ma | 246 | 18.2216 | 20.6057 | 13.2853 | 22.0979 | +8.8125 |
| pa | 127 | 19.1066 | 19.4780 | 15.1684 | 15.3053 | +0.1368 |
| Root | 161 | 21.1423 | 20.7916 | 15.1877 | 14.7532 | -0.4345 |

Phân nhóm theo regex hiện có, không phải gold morphology annotations. mi/ma/pa giao nhau; 414 câu affixed và 161 Root. 141 câu có từ hai nhóm prefix trở lên; 154 là membership dư. Không diễn giải n-gram gains như chứng minh nhân quả về bảo toàn hình thái.

### 5.2. Phân Biệt chrF++ Chuẩn và TokenizerZh Diagnostic

chrF++ chuẩn dùng văn bản gốc, gồm xử lý punctuation nội bộ của SacreBLEU. Thiếu whitespace không có nghĩa metric bị lỗi hay mọi word bigram đều bằng 0. TokenizerZh diagnostic của Baseline/CLRR+LSR là **22.7229 / 23.5869**; chỉ là phép đo phụ. Toàn bộ scores/signatures nằm trong [METRICS.md](METRICS.md).

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
| `nllb-200/nllb_results.zip` (Toàn bộ 6 phương pháp NLLB) | 41.046.143.831 bytes | Đã có trên HF; audit CSV bằng HTTP range |
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

## 9. Audit metrics ngày 2026-10-02

Đã kiểm tra 29 bộ predictions và 3 đối chiếu CLI/API. NLLB và Asháninka tái lập khớp điểm chính; mBART có reference roundtrip làm thay đổi 62/575 câu. Các bảng trên dùng số chuẩn; bảng sửa số đầy đủ và giới hạn: [METRICS.md](METRICS.md). Raw artifacts giữ nguyên; bản trước sửa lưu ở `outputs_rebuttal/metric_audit_20261002/docs_before/`.
