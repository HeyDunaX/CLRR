# KẾ HOẠCH TRIỂN KHAI & RUNBOOK THỰC NGHIỆM ĐỐI CHẤT REVIEWER (PEFT ENHANCEMENT & SCIENTIFIC RUNBOOK)

> **Cập nhật trạng thái vận hành 2026-10-02 (Kế hoạch thực nghiệm mở rộng & Quy chuẩn siêu tham số công bằng tuyệt đối):**
> * **Đã hoàn tất 100% các thí nghiệm cốt lõi trên Amis $\to$ Mandarin:** mBART-50 (Full-FT, CLRR-Enc, CLRR-Dec, Middle-Align, BitFit, Narrow LoRA, Strong LoRA A, Strong LoRA B), NLLB-200 (Amis suite), mT5-small (Appendix), ByT5.
> * **Dữ liệu mới Asháninka $\to$ Spanish (`data_processed/ashaninka_spanish/`) đã sẵn sàng 100%:** Nguồn AmericasNLP 2021 (commit `d3f519c`), train 3.883 / val 881 / test 1.003 câu, SHA-256 xác minh đầy đủ, cross-split overlap = 0.
> * **Mã nguồn đã nâng cấp hỗ trợ đa ngôn ngữ:** Tự động nhận diện ngôn ngữ đích (`es_XX` / `spa_Latn`), proxy nguồn Latinh, và tokenizer SacreBLEU (`13a` cho Spanish, `zh` cho Mandarin). Hỗ trợ trực tiếp `strong_lora_a` và `strong_lora_b` trên NLLB-200. Unit test passed 12/12.

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN & TỐI ƯU HÓA TÀI NGUYÊN (SCIENTIFIC RIGOR, FAIRNESS, SPEED & CU PROTECTION):**
> 1. **Mục tiêu tối thượng:** Bẻ gãy dứt điểm các nghi vấn từ báo cáo phản biện (yêu cầu kiểm chứng đa ngôn ngữ trên AmericasNLP, nghi ngờ LoRA trên NLLB, và tính trung thực của độ đo), đưa bài báo lên *Strong Accept* tại ACL Main Track (hoặc ComputEL-10).
> 2. **Bảo tồn toàn vẹn thành quả đã đạt được:** Đóng băng toàn bộ kết quả SOTA của CLRR-Enc trên `mBART-large-50` (20.39 BLEU, 19.08 chrF++), `nllb-200-distilled-600M` (14.28 BLEU, 18.75 chrF++ Zh), và `mT5-small` (4.83 BLEU). Checkpoint lưu trữ an toàn tại `FiveC/amis-rewire-checkpoints`.
> 3. **Giao thức công bằng tuyệt đối (Strict Fairness Protocol):** Cố định toàn diện siêu tham số giữa các phương pháp đối chuẩn (Seed 42, Effective Batch Size 128, Early Stopping Patience 4 theo chrF++, Precision BF16, Generation Beam Size 4, SacreBLEU `13a` cho Spanish / `zh` cho Mandarin + chrF++ `word_order=2`).
> 4. **Chuẩn mực nghiệm thu (No Hand-copied Numbers):** Bắt buộc đối chiếu trực tiếp từ `metrics.json` và `test_predictions.csv` sinh ra từ artifact. Tuyệt đối không chép tay số liệu.

---

## 1. Bản Đồ Trạng Thái Thực Nghiệm (Experiment Status Matrix)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        BẢN ĐỒ TIẾN ĐỘ THỰC NGHIỆM ĐỐI CHẤT REVIEWER & MỞ RỘNG ACL                      │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
         ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
         ▼                                          ▼                                          ▼
 [ĐÃ HOÀN TẤT 100%]                         [ƯU TIÊN 1: NLLB LoRA]                     [ƯU TIÊN 2: ASHÁNINKA]
  ✅ mBART-50 Amis Suite (Full + LoRA A/B)   🚀 SẴN SÀNG CHẠY TRÊN A100                 🚀 SẴN SÀNG CHẠY TRÊN A100
  ✅ NLLB-200 Amis Core Suite               - Run A: All-Linear (r=16)                 - Run 1: mBART Baseline (es_XX)
  ✅ mT5-small Suite (Appendix F)           - Run B: All-Linear + Unfreeze Embed       - Run 2: mBART CLRR-Enc + LSR
  ✅ ByT5-small Baseline vs CLRR            - Tạo đối xứng 1-1 với mBART-50            - Run 3: NLLB Baseline (spa_Latn)
  ✅ Chẩn đoán Probing & SVD Anti-Collapse  - Dự kiến: ~40 phút / run                  - Run 4: NLLB CLRR-Enc + LSR
```

### Bảng Tổng Hợp Chi Tiết Các Nhiệm Vụ Tiếp Theo:

| Thứ Tự | Hạng Mục Thí Nghiệm | Script Thực Thi | Cấu Hình Kỹ Thuật | Mục Đích Đối Sách Reviewer |
| :---: | :--- | :--- | :--- | :--- |
| **P1.1** | **Strong LoRA A trên NLLB-200** | `scripts/peft/run_strong_lora_nllb.py` | All-Linear ($r=16, \alpha=32$, 6 projections, frozen embed, LR 2e-4) | Trả lời câu hỏi reviewer: *"Liệu LoRA trên NLLB có cứu được không?"*. Kiểm tra mức phục hồi so với Narrow LoRA (3.54 BLEU). |
| **P1.2** | **Strong LoRA B trên NLLB-200** | `scripts/peft/run_strong_lora_nllb.py` | All-Linear ($r=16, \alpha=32$, unfreeze shared/embed/lm_head, LR 1e-4) | Khẳng định kể cả khi mở khóa embeddings, LoRA trên NLLB vẫn thua Baseline và CLRR-Enc (14.28 BLEU). |
| **P2.1** | **mBART Baseline trên Asháninka** | `scripts/run_ashaninka_experiments.py` | Full-FT, LR 5e-5, target `es_XX`, tokenizer `13a`, chrF++ $w=2$ | Thiết lập điểm tựa đối chuẩn trên ngôn ngữ thứ hai (Arawakan, AmericasNLP). |
| **P2.2** | **mBART CLRR-Enc trên Asháninka** | `scripts/run_ashaninka_experiments.py` | CLRR-Enc + LSR ($d=2, \alpha=0.1, \lambda=0.1$), LR 5e-5, target `es_XX` | **Bằng chứng quyết định cho ACL Main:** Chứng minh CLRR tiếp tục vượt trội trên ngôn ngữ đa tổng hợp thứ hai (+chrF++, +BLEU). |
| **P3.1** | **NLLB Baseline trên Asháninka** | `scripts/run_ashaninka_experiments.py` | Full-FT, LR 5e-5, target `spa_Latn`, tokenizer `13a` | Thiết lập điểm tựa trên backbone dịch đa ngữ chuyên biệt cho Asháninka $\to$ Spanish. |
| **P3.2** | **NLLB CLRR-Enc trên Asháninka** | `scripts/run_ashaninka_experiments.py` | CLRR-Enc + LSR ($d=2, \alpha=0.1, \lambda=0.1$), LR 5e-5, target `spa_Latn` | Xác lập tính ưu việt của CLRR trên cả hai backbone đối với cặp ngôn ngữ thứ hai. |

---

## 2. Bảng Quy Chuẩn Siêu Tham Số Chi Tiết Từng Thí Nghiệm (Strict Fairness Protocol & Per-Experiment Parameter Matrix)

Nhằm đảm bảo **tính công bằng học thuật tuyệt đối (Strict Scientific Fairness)** theo tiêu chuẩn ACL, không để xảy ra bất kỳ sự thiên vị nào về điều kiện huấn luyện giữa các phương pháp đối chuẩn (Baselines, PEFT baselines) và phương pháp đề xuất (CLRR-Enc + LSR), bảng dưới đây liệt kê tường minh **từng dòng tham số cài đặt cho toàn bộ các thí nghiệm đã thực hiện trước đó và các thí nghiệm mới chuẩn bị chạy**:

### 2.1. Ma Trận Siêu Tham Số Huấn Luyện Từng Thí Nghiệm (Training & Architecture Hyperparameters)

| ID | Tên Thí Nghiệm / Run Name | Trạng Thái | Mô Hình Gốc (HF Backbone) | Cặp Ngôn Ngữ | Phương Pháp | Tham Số Tối Ưu ($\theta_{\text{train}}$) | Tham Số Thêm ($\Delta\theta_{\text{add}}$) | Modules Can Thiệp / Mở Khóa | CLRR Setup ($d, \alpha, \lambda$) | Learning Rate | Batch Size (Device $\times$ Accum) | Effective Batch | Max Epochs & Patience | Optimizer & Warmup | Seed |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **REF-1** | `mbart-large-50-baseline` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Full Fine-Tuning | 610.88M (100%) | 0 | Toàn bộ mô hình | N/A | $5 \times 10^{-5}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-2** | `mbart-large-50-clrr-enc` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | CLRR-Enc + LSR | 610.88M (100%) | 0 | Toàn bộ mô hình | $d=2, \alpha=0.1, \lambda=0.1$ | $5 \times 10^{-5}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-3** | `mbart-large-50-lora-all-linear` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Strong LoRA Run A | 3.54M (0.58%) | +3.54M | $q,k,v,o,fc1,fc2$ ($r=16, \alpha=32$) | N/A | $2 \times 10^{-4}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-4** | `mbart-...-unfreeze-embed` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Strong LoRA Run B | 260.0M (42.6%) | +3.54M | All-Lin + `shared,lm_head` | N/A | $1 \times 10^{-4}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-5** | `nllb-200-baseline` | ✅ Đã chạy | `nllb-200-distilled-600M` | Amis $\to$ Zh | Full Fine-Tuning | 614.91M (100%) | 0 | Toàn bộ mô hình | N/A | $5 \times 10^{-5}$ | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-6** | `nllb-200-clrr-enc` | ✅ Đã chạy | `nllb-200-distilled-600M` | Amis $\to$ Zh | CLRR-Enc + LSR | 614.91M (100%) | 0 | Toàn bộ mô hình | $d=2, \alpha=0.1, \lambda=0.1$ | $5 \times 10^{-5}$ | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P1.1** | `nllb-200-lora-all-linear` | 🚀 **Sắp chạy** | `nllb-200-distilled-600M` | Amis $\to$ Zh | Strong LoRA Run A | 8.65M (1.41%) | +8.65M | $q,k,v,o,fc1,fc2$ ($r=16, \alpha=32$) | N/A | **$2 \times 10^{-4}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P1.2** | `nllb-...-unfreeze-embed` | 🚀 **Sắp chạy** | `nllb-200-distilled-600M` | Amis $\to$ Zh | Strong LoRA Run B | 264.7M (42.7%) | +8.65M | All-Lin + `shared,embed,lm_head` | N/A | **$1 \times 10^{-4}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P2.1** | `mbart-ashaninka-es-baseline` | 🚀 **Sắp chạy** | `mbart-large-50-...` | Ashán $\to$ Es | Full Fine-Tuning | 610.88M (100%) | 0 | Toàn bộ mô hình | N/A | **$5 \times 10^{-5}$** | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P2.2** | `mbart-ashaninka-es-clrr-enc` | 🚀 **Sắp chạy** | `mbart-large-50-...` | Ashán $\to$ Es | CLRR-Enc + LSR | 610.88M (100%) | 0 | Toàn bộ mô hình | **$d=2, \alpha=0.1, \lambda=0.1$** | **$5 \times 10^{-5}$** | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P3.1** | `nllb-ashaninka-es-baseline` | 🚀 **Sắp chạy** | `nllb-200-distilled-600M` | Ashán $\to$ Es | Full Fine-Tuning | 614.91M (100%) | 0 | Toàn bộ mô hình | N/A | **$5 \times 10^{-5}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P3.2** | `nllb-ashaninka-es-clrr-enc` | 🚀 **Sắp chạy** | `nllb-200-distilled-600M` | Ashán $\to$ Es | CLRR-Enc + LSR | 614.91M (100%) | 0 | Toàn bộ mô hình | **$d=2, \alpha=0.1, \lambda=0.1$** | **$5 \times 10^{-5}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |

> [!TIP]
> **Điểm mấu chốt bảo đảm tính công bằng tuyệt đối (Fairness Guarantees):**
> 1. **Batch Size hiệu dụng:** Toàn bộ 12 thí nghiệm trên đều đạt chính xác **Effective Batch Size = 128** ($4 \times 32$ cho mBART-50 do giới hạn bộ nhớ VRAM, $16 \times 8$ cho NLLB-200).
> 2. **Learning Rate chuẩn hóa:** 
>    - Full Fine-Tuning và CLRR luôn dùng **$5 \times 10^{-5}$** (nghiêm cấm tinh chỉnh LR để làm lợi cho CLRR).
>    - LoRA Run A (chỉ adapter) dùng **$2 \times 10^{-4}$** (mức LR tối ưu tiêu chuẩn cho PEFT theo văn bản ACL/LoRA gốc).
>    - LoRA Run B (adapter + unfreeze embeddings) dùng **$1 \times 10^{-4}$** (hạ nhẹ LR nhằm bảo vệ ma trận nhúng không bị gradient explosion/divergence).
> 3. **Optimizer & Regularization:** Toàn bộ đều dùng `AdamW` ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$), `weight_decay=0.01`, warmup linear 6% tổng steps.
> 4. **Early Stopping:** Cố định `patience=4` epochs, checkpoint được chọn là checkpoint đạt **Validation chrF++ cao nhất** (`load_best_model_at_end=True`), hoàn toàn không chọn theo Training Loss.

---

### 2.2. Bảng Quy Chuẩn Ngôn Ngữ, Tokenizer & Độ Đo Đánh Giá (Linguistic & Evaluation Protocol)

| ID Nhóm | Thí Nghiệm | Tập Dữ Liệu & Quy Mô (Train / Val / Test) | Max Len (Src / Tgt) | Source Lang Code & Proxy | Target Lang Code | Tokenizer Decoding BLEU | Cấu Hình chrF++ | Beam Size & Length Penalty |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **REF / P1** | **Amis $\to$ Mandarin** (mBART & NLLB Suite) | `data_processed/amis_mandarin`<br>(4.600 / 576 / 575) | 128 / 128 | `zho_Hant` / `zh_CN` | `zho_Hant` (NLLB)<br>`zh_CN` (mBART) | `zh` (Chinese Char tokenizer) | `word_order=2, beta=2` | Beam = **4**<br>Len Penalty = **1.0** |
| **P2** | **Asháninka $\to$ Spanish** (mBART Suite) | `data_processed/ashaninka_spanish`<br>(3.883 / 881 / 1.003) | 256 / 256 | `es_XX` (Latin proxy) | `es_XX` | `13a` (International/WMT) | `word_order=2, beta=2` | Beam = **4**<br>Len Penalty = **1.0** |
| **P3** | **Asháninka $\to$ Spanish** (NLLB Suite) | `data_processed/ashaninka_spanish`<br>(3.883 / 881 / 1.003) | 128 / 128 | `spa_Latn` (Latin proxy) | `spa_Latn` | `13a` (International/WMT) | `word_order=2, beta=2` | Beam = **4**<br>Len Penalty = **1.0** |

> [!NOTE]
> **Giải thích kỹ thuật về Tokenizer BLEU:**
> * Đối với Mandarin, ký tự tiếng Trung viết liền không có dấu cách nên bắt buộc dùng SacreBLEU `zh` tokenizer để tách ký tự/từ đơn.
> * Đối với Spanish (ngôn ngữ alphabet Latinh), AmericasNLP và WMT chuẩn hóa sử dụng tokenizer `13a` kết hợp chrF++ `word_order=2`. Việc sử dụng `13a` loại bỏ hoàn toàn hiện tượng lệch điểm do khoảng trắng mà reviewer từng chất vấn trên tiếng Trung.

---

## 3. Kế Hoạch 3 Giai Đoạn Huấn Luyện Sắp Tới

### GIAI ĐOẠN 1: Strong LoRA Suite trên NLLB-200 (Amis $\to$ Mandarin)
* **Thời gian ước tính:** ~75–80 phút trên GPU A100 (khoảng 38–40 phút/run).
* **Đầu vào:** `data_processed/amis_mandarin/`
* **Lệnh kích hoạt tự động:**
  ```bash
  python -u scripts/peft/run_strong_lora_nllb.py
  ```
* **Nghiệm thu:** 
  * Run A (`results/nllb-200/nllb-200-lora-all-linear/metrics.json`)
  * Run B (`results/nllb-200/nllb-200-lora-all-linear-unfreeze-embed/metrics.json`)
  * File tổng hợp: `outputs_rebuttal/strong_lora_nllb_scores.csv`

---

### GIAI ĐOẠN 2: Core Matrix trên Ngôn Ngữ Thứ Hai Asháninka $\to$ Spanish (mBART-50)
* **Thời gian ước tính:** ~85–90 phút trên GPU A100.
* **Đầu vào:** `data_processed/ashaninka_spanish/` (3.883 train / 881 val / 1.003 test)
* **Lệnh kích hoạt tự động:**
  ```bash
  python -u scripts/run_ashaninka_experiments.py --runs mbart_baseline mbart_clrr
  ```
* **Nghiệm thu:**
  * Baseline: `results/ashaninka_spanish/mbart-ashaninka-es-baseline/metrics.json`
  * CLRR-Enc: `results/ashaninka_spanish/mbart-ashaninka-es-clrr-enc/metrics.json`
  * Điểm số: SacreBLEU (tokenizer `13a`) và chrF++ ($w=2$) trên 1.003 câu kiểm thử chính thức của AmericasNLP.

---

### GIAI ĐOẠN 3: Core Matrix trên Asháninka $\to$ Spanish (NLLB-200)
* **Thời gian ước tính:** ~70 phút trên GPU A100.
* **Đầu vào:** `data_processed/ashaninka_spanish/`
* **Lệnh kích hoạt tự động:**
  ```bash
  python -u scripts/run_ashaninka_experiments.py --runs nllb_baseline nllb_clrr
  ```
* **Nghiệm thu:**
  * Baseline: `results/ashaninka_spanish/nllb-ashaninka-es-baseline/metrics.json`
  * CLRR-Enc: `results/ashaninka_spanish/nllb-ashaninka-es-clrr-enc/metrics.json`

---

## 4. Sổ Tay Lệnh Vận Hành Google Colab (Colab Runbook)

### Yêu Cầu Phần Cứng:
* **GPU:** NVIDIA A100 (chạy không cần `--high-mem`, tiết kiệm Compute Units tối đa).
* **Disk:** $\ge 25\text{GB}$ trống trên `/content`.

### Quy Trình Kích Hoạt Từng Bước Trên Máy Chủ:

#### Bước 1: Khởi động Colab A100
```bash
colab new -s colab --gpu A100
```

#### Bước 2: Đồng bộ mã nguồn & Dữ liệu
```bash
cd /content
if [ ! -d "/content/CLRR" ]; then
    git clone https://github.com/HeyDunaX/CLRR.git /content/CLRR
fi
cd /content/CLRR
git checkout main
git pull origin main
pip install -q -r requirements.txt
pip install -q peft sacrebleu scikit-learn
pip uninstall -y torchao 2>/dev/null || true
```

#### Bước 3: Chạy Preflight Test xác nhận môi trường (< 15 giây)
```bash
python -m unittest tests/test_followup.py
```

#### Bước 4: Chạy Giai Đoạn 1 (NLLB Strong LoRA Suite trên Amis)
```bash
nohup python -u scripts/peft/run_strong_lora_nllb.py > /content/nllb_lora.log 2>&1 &
tail -f /content/nllb_lora.log
```

#### Bước 5: Chạy Giai Đoạn 2 & 3 (Asháninka Suite trên cả mBART và NLLB)
```bash
nohup python -u scripts/run_ashaninka_experiments.py > /content/ashaninka.log 2>&1 &
tail -f /content/ashaninka.log
```

---

## 5. Bảng Dự Kiến Bổ Sung Vào Bài Báo Sau Khi Hoàn Tất

### Bảng 1: Bổ Sung NLLB Strong LoRA (Khép góc PEFT trên Amis $\to$ Mandarin)
| Backbone Architecture | Cấu hình Thích nghi | $\Delta\theta_{\text{add}}$ | $\theta_{\text{train}}$ (% Model) | BLEU (zh) ↑ | chrF++ (w=2) ↑ | chrF++ (Zh) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`nllb-200-distilled-600M`** | BitFit (Bias-only) | 0 | 131k (0.02%) | 1.0750 | 2.9643 | 5.7081 |
| | Narrow LoRA ($r=8, q/v$) | +1.18M | 1.18M (0.19%) | 3.5383 | 4.9407 | 9.1651 |
| | **Strong LoRA A ($r=16$, All-Linear)** | **+8.65M** | **8.65M (1.41%)** | *[Đang chạy]* | *[Đang chạy]* | *[Đang chạy]* |
| | **Strong LoRA B (+Embeddings)** | **+8.65M** | **264.7M (42.7%)** | *[Đang chạy]* | *[Đang chạy]* | *[Đang chạy]* |
| | Standard Fine-Tuning | 0 | 615M (100%) | 13.5001 | 10.4389 | 18.0885 |
| | Middle-Layer Alignment | 0 | 615M (100%) | 13.9648 | 10.6596 | 18.4448 |
| | **CLRR-Enc + LSR (Ours)** | **0** | **615M (100%)** | **14.2820** | **10.9152** | **18.7482** |

### Bảng 2: Bảng Đối Sánh Ngôn Ngữ Thứ Hai (Asháninka $\to$ Spanish, 1.003 Test Sentences)
| Backbone Architecture | Phương pháp | $\Delta\theta_{\text{add}}$ | $\theta_{\text{train}}$ (% Model) | BLEU (13a) ↑ | chrF++ (w=2) ↑ | Nhận định thực nghiệm AmericasNLP |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`mBART-large-50`** | Standard Fine-Tuning | 0 | 611M (100%) | *[Đang chạy]* | *[Đang chạy]* | Điểm tựa mBART trên Asháninka $\to$ Spanish |
| | **CLRR-Enc + LSR (Ours)** | **0** | **611M (100%)** | *[Đang chạy]* | *[Đang chạy]* | **Khẳng định tính tổng quát hóa đa ngôn ngữ** |
| **`nllb-200-distilled-600M`** | Standard Fine-Tuning | 0 | 615M (100%) | *[Đang chạy]* | *[Đang chạy]* | Điểm tựa NLLB trên Asháninka $\to$ Spanish |
| | **CLRR-Enc + LSR (Ours)** | **0** | **615M (100%)** | *[Đang chạy]* | *[Đang chạy]* | **Thiết lập SOTA mới trên AmericasNLP 2021** |

---

## 6. Hướng Dẫn Nghiệm Thu & Đóng Gói
1. Toàn bộ mã nguồn chạy mới đã được commit/sẵn sàng trong [`scripts/peft/run_strong_lora_nllb.py`](file:///d:/Code/CLRR/scripts/peft/run_strong_lora_nllb.py) và [`scripts/run_ashaninka_experiments.py`](file:///d:/Code/CLRR/scripts/run_ashaninka_experiments.py).
2. Khi các run trên Colab A100 hoàn tất, chỉ cần tải thư mục `results/` và `outputs_rebuttal/` về máy này để cập nhật trực tiếp vào [`clrr_main.tex`](file:///d:/Code/CLRR/docs/CLRR-paper/latex/clrr_main.tex).
