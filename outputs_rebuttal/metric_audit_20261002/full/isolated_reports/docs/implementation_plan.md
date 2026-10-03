# KẾ HOẠCH TRIỂN KHAI & RUNBOOK THỰC NGHIỆM ĐỐI CHẤT REVIEWER (PEFT ENHANCEMENT & SCIENTIFIC RUNBOOK)

> **Audit đầy đủ 2026-10-02:** 36 bộ predictions đã đối chiếu reference CSV gốc; 42 phép kiểm định paired bootstrap (10.000 mẫu, seed 42, Holm toàn bộ 42 tests). Xem [METRICS.md](METRICS.md), [RESULTS_VERIFICATION.md](RESULTS_VERIFICATION.md) và `results/metrics_standardized_scores.csv`. TokenizerZh chrF++ chỉ là diagnostic; checkpoint lịch sử được giữ nguyên.

> **Asháninka được tác giả cho phép tiếp tục ngày 2026-10-02:** giao thức chạy mới
> được chốt tại [ASHANINKA_RUNBOOK.md](ASHANINKA_RUNBOOK.md). Baseline/CLRR trong
> mỗi backbone dùng cùng cấu hình; weight decay thực tế là 0.0, mBART validation
> greedy/test beam 4, NLLB validation/test beam 4. Điểm test mới dùng reference CSV
> gốc. Tác giả đã cho phép verify và sửa số từ predictions đã lưu. Không nạp
> checkpoint, train lại hoặc đổi checkpoint trong audit; metadata gốc được bảo tồn.
>
> **Kết quả Asháninka cập nhật 2026-10-02:** mBART Baseline, mBART CLRR+LSR
> patience 4 và NLLB Baseline đã hoàn tất. Lượt mBART CLRR+LSR patience 10
> đã train đủ 20 epoch. Theo yêu cầu tác giả, Bảng 2 ghi thêm checkpoint
> **epoch 20: BLEU 4.0949 / chrF++ 20.9588**; checkpoint chọn theo validation
> là epoch 14 và được ghi riêng. NLLB CLRR chưa chạy. Các checkpoint đã backup
> Hugging Face và Colab đã tắt. Chi tiết: [ASHANINKA_PATIENCE10_RESULTS.md](ASHANINKA_PATIENCE10_RESULTS.md).

> **Cập nhật trạng thái vận hành 2026-10-02 (Kế hoạch thực nghiệm mở rộng & Cấu hình và giới hạn đối sánh):**
> * **Đã có outputs cho các thí nghiệm cốt lõi trên Amis $\to$ Mandarin:** mBART-50 (Full-FT, CLRR-Enc, CLRR-Dec, Middle-Align, BitFit, Narrow LoRA, Strong LoRA A, Strong LoRA B), NLLB-200 (Amis suite), mT5-small (Appendix), ByT5.
> * **Dữ liệu mới Asháninka $\to$ Spanish (`data_processed/ashaninka_spanish/`) đã sẵn sàng 100%:** Nguồn AmericasNLP 2021 (commit `d3f519c`), train 3.883 / val 881 / test 1.003 câu, SHA-256 xác minh đầy đủ, exact stripped source–target pair overlap giữa các split = 0; train có 23 duplicate pairs. Chưa audit near-duplicates.
> * **Mã nguồn đã nâng cấp hỗ trợ đa ngôn ngữ:** Tự động nhận diện ngôn ngữ đích (`es_XX` / `spa_Latn`), proxy nguồn Latinh, và tokenizer SacreBLEU (`13a` cho Spanish, `zh` cho Mandarin). Hỗ trợ trực tiếp `strong_lora_a` và `strong_lora_b` trên NLLB-200. 12/12 là ghi nhận lịch sử, không rerun toàn bộ trong audit này. Hai regression tests mới cho raw-reference callback đã pass.

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN & TỐI ƯU HÓA TÀI NGUYÊN (SCIENTIFIC RIGOR, FAIRNESS, SPEED & CU PROTECTION):**
> 1. **Mục tiêu:** kiểm chứng nghi vấn reviewer bằng outputs có provenance, giữ cả kết quả âm tính và giới hạn suy luận.
> 2. **Bảo tồn checkpoint, sửa số theo artifact:** mBART CLRR+LSR **20.3896 BLEU / 19.0824 chrF++ chuẩn**; NLLB CLRR+LSR **14.2820 / 10.9152**. TokenizerZh chrF++ không phải số chính. Checkpoint được giữ nguyên tại `FiveC/amis-rewire-checkpoints`; nguồn điểm: [METRICS.md](METRICS.md).
> 3. **Đối sánh có kiểm soát:** giữ dữ liệu, seed 42 và effective batch 128 trong suite mBART/NLLB; LR khác theo method. Ash patience10/epoch20 là bổ sung post-hoc. Chưa có ngân sách HPO bằng nhau hay multi-seed, không gọi công bằng tuyệt đối.
> 4. **Nguồn nghiệm thu:** `results/metrics_standardized_scores.csv` tính từ predictions và raw test.csv. `metrics.json` lịch sử có thể dùng decoded references; giữ làm provenance, không coi là metric chuẩn tự động.

---

## 1. Bản Đồ Trạng Thái Thực Nghiệm (Experiment Status Matrix)

| Hạng mục | Trạng thái sau audit |
| --- | --- |
| Amis mBART/NLLB Strong LoRA A/B | Đã chạy; raw scores tái lập |
| mT5/ByT5/LayerSkip lịch sử | Predictions có sẵn được audit; mT5 LSR-only chưa xác minh |
| SVD/probing | Thiếu artifacts, chưa chứng nhận |
| Ash mBART baseline/CLRR, NLLB baseline | Đã xong; không có chênh lệch có ý nghĩa |
| Ash NLLB CLRR | Chưa chạy; chờ kế hoạch tác giả |
| Vận hành | Colab đã tắt; audit không khởi động training |

### Bảng Tổng Hợp Chi Tiết Các Nhiệm Vụ Tiếp Theo:

| Thứ Tự | Hạng Mục Thí Nghiệm | Script Thực Thi | Cấu Hình Kỹ Thuật | Mục Đích Đối Sách Reviewer |
| :---: | :--- | :--- | :--- | :--- |
| **P1.1** | **Strong LoRA A trên NLLB-200** | `scripts/peft/run_strong_lora_nllb.py` | All-Linear ($r=16, \alpha=32$, 6 projections, frozen embed, LR 2e-4) | Trả lời câu hỏi reviewer: *"Liệu LoRA trên NLLB có cứu được không?"*. Kiểm tra mức phục hồi so với Narrow LoRA (3.54 BLEU). |
| **P1.2** | **Strong LoRA B trên NLLB-200** | `scripts/peft/run_strong_lora_nllb.py` | All-Linear ($r=16, \alpha=32$, unfreeze shared/embed/lm_head, LR 1e-4) | Khẳng định kể cả khi mở khóa embeddings, LoRA trên NLLB vẫn thua Baseline và CLRR-Enc (14.28 BLEU). |
| **P2.1** | **mBART Baseline trên Asháninka** | `scripts/run_ashaninka_experiments.py` | Full-FT, LR 5e-5, target `es_XX`, tokenizer `13a`, chrF++ $w=2$ | Thiết lập điểm tựa đối chuẩn trên ngôn ngữ thứ hai (Arawakan, AmericasNLP). |
| **P2.2** | **mBART CLRR-Enc trên Asháninka** | `scripts/run_ashaninka_experiments.py` | CLRR-Enc + LSR ($d=2, \alpha=0.1, \lambda=0.1$), LR 5e-5, target `es_XX` | Đã đo; chưa chứng minh CLRR vượt baseline trên ngôn ngữ thứ hai. |
| **P3.1** | **NLLB Baseline trên Asháninka** | `scripts/run_ashaninka_experiments.py` | Full-FT, LR 5e-5, target `spa_Latn`, tokenizer `13a` | Thiết lập điểm tựa trên backbone dịch đa ngữ chuyên biệt cho Asháninka $\to$ Spanish. |
| **P3.2** | **NLLB CLRR-Enc trên Asháninka** | `scripts/run_ashaninka_experiments.py` | CLRR-Enc + LSR ($d=2, \alpha=0.1, \lambda=0.1$), LR 5e-5, target `spa_Latn` | Kiểm tra khả năng mở rộng trên backbone thứ hai; chưa có kết quả. |

---

## 2. Cấu hình từng thí nghiệm và giới hạn fairness

Bảng ghi cấu hình đã chạy/dự kiến; không chứng minh optimality hoặc HPO budget bằng nhau. Raw metadata và launcher là nguồn xác minh từng cấu hình. Những mặc định không có metadata tương ứng không được chứng nhận như fact lịch sử.

### 2.1. Ma Trận Siêu Tham Số Huấn Luyện Từng Thí Nghiệm (Training & Architecture Hyperparameters)

| ID | Tên Thí Nghiệm / Run Name | Trạng Thái | Mô Hình Gốc (HF Backbone) | Cặp Ngôn Ngữ | Phương Pháp | Tham Số Tối Ưu ($\theta_{\text{train}}$) | Tham Số Thêm ($\Delta\theta_{\text{add}}$) | Modules Can Thiệp / Mở Khóa | CLRR Setup ($d, \alpha, \lambda$) | Learning Rate | Batch Size (Device $\times$ Accum) | Effective Batch | Max Epochs & Patience | Optimizer & Warmup | Seed |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **REF-1** | `mbart-large-50-baseline` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Full Fine-Tuning | 610.88M (100%) | 0 | Toàn bộ mô hình | N/A | $5 \times 10^{-5}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-2** | `mbart-large-50-clrr-enc` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | CLRR-Enc + LSR | 610.88M (100%) | 0 | Toàn bộ mô hình | $d=2, \alpha=0.1, \lambda=0.1$ | $5 \times 10^{-5}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-3** | `mbart-large-50-lora-all-linear` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Strong LoRA Run A | 8.65M (1.3963%) | +8.65M | $q,k,v,o,fc1,fc2$ ($r=16, \alpha=32$) | N/A | $2 \times 10^{-4}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-4** | `mbart-...-unfreeze-embed` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Strong LoRA Run B | 264.71M (42.7269%) | +8.65M | All-Lin + `shared,lm_head` | N/A | $1 \times 10^{-4}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-5** | `nllb-200-baseline` | ✅ Đã chạy | `nllb-200-distilled-600M` | Amis $\to$ Zh | Full Fine-Tuning | 615.073792M (100%) | 0 | Toàn bộ mô hình | N/A | $5 \times 10^{-5}$ | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-6** | `nllb-200-clrr-enc` | ✅ Đã chạy | `nllb-200-distilled-600M` | Amis $\to$ Zh | CLRR-Enc + LSR | 615.073792M (100%) | 0 | Toàn bộ mô hình | $d=2, \alpha=0.1, \lambda=0.1$ | $5 \times 10^{-5}$ | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P1.1** | `nllb-200-lora-all-linear` | ✅ **Đã hoàn thành** (BLEU: **11.46**, chrF++: **9.50**) | `nllb-200-distilled-600M` | Amis $\to$ Zh | Strong LoRA Run A | 8.65M (1.39%) | +8.65M | $q,k,v,o,fc1,fc2$ ($r=16, \alpha=32$) | N/A | **$2 \times 10^{-4}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P1.2** | `nllb-...-unfreeze-embed` | ✅ **Đã hoàn thành** (BLEU: **9.80**, chrF++: **8.40**) | `nllb-200-distilled-600M` | Amis $\to$ Zh | Strong LoRA Run B | 271.005696M (43.4496% active; 30.5848% registered) | +8.65M | All-Lin + `shared,embed,lm_head` | N/A | **$1 \times 10^{-4}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P2.1** | `mbart-ashaninka-es-baseline` | ✅ **Hoàn tất, đã backup HF** | `mbart-large-50-...` | Ashán $\to$ Es | Full Fine-Tuning | 610.88M (100%) | 0 | Toàn bộ mô hình | N/A | **$5 \times 10^{-5}$** | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P2.2** | `mbart-ashaninka-es-clrr-enc` | ✅ **Hoàn tất lượt gốc, đã backup HF** | `mbart-large-50-...` | Ashán $\to$ Es | CLRR-Enc + LSR | 610.88M (100%) | 0 | Toàn bộ mô hình | **$d=2, \alpha=0.1, \lambda=0.1$** | **$5 \times 10^{-5}$** | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P2.2 bổ sung** | `mbart-ashaninka-es-clrr-enc` dưới `ashaninka_patience10/runs/` | ✅ **Hoàn tất; đánh giá epoch 14 và 20, đã backup HF** | `mbart-large-50-...` | Ashán $\to$ Es | CLRR-Enc + LSR | 610.88M (100%) | 0 | Toàn bộ mô hình | **$d=2, \alpha=0.1, \lambda=0.1$** | **$5 \times 10^{-5}$** | $4 \times 32$ | **128** | 20 ep / 10 pat | AdamW, 6% linear | 42 |
| **P3.1** | `nllb-ashaninka-es-baseline` | ✅ **Hoàn tất, đã backup HF** | `nllb-200-distilled-600M` | Ashán $\to$ Es | Full Fine-Tuning | 615.073792M (100%) | 0 | Toàn bộ mô hình | N/A | **$5 \times 10^{-5}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P3.2** | `nllb-ashaninka-es-clrr-enc` | ⏸️ **Tạm dừng (Chờ lệnh)** | `nllb-200-distilled-600M` | Ashán $\to$ Es | CLRR-Enc + LSR | 615.073792M (100%) | 0 | Toàn bộ mô hình | **$d=2, \alpha=0.1, \lambda=0.1$** | **$5 \times 10^{-5}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |

> [!TIP]
> **Những biến kiểm soát và ngoại lệ:**
> 1. **Batch Size hiệu dụng:** Các run mBART/NLLB trong bảng đều đạt chính xác **Effective Batch Size = 128** ($4 \times 32$ cho mBART-50 do giới hạn bộ nhớ VRAM, $16 \times 8$ cho NLLB-200).
> 2. **Learning Rate chuẩn hóa:**
>    - Full Fine-Tuning và CLRR luôn dùng **$5 \times 10^{-5}$** (nghiêm cấm tinh chỉnh LR để làm lợi cho CLRR).
>    - LoRA Run A (chỉ adapter) dùng **$2 \times 10^{-4}$** (cấu hình đã chạy, chưa xác minh tối ưu qua sweep).
>    - LoRA Run B (adapter + unfreeze embeddings) dùng **$1 \times 10^{-4}$** (cấu hình đã chạy; chưa tách riêng tác động LR và embeddings).
> 3. **Optimizer & Regularization:** AdamW, warmup 6%; entrypoint mặc định và Asháninka thực tế weight decay 0.0. Không chứng nhận weight_decay=0.01 cho mọi run lịch sử.
> 4. **Early Stopping:** lượt gốc patience4, chọn best validation chrF++; Ash patience10 là ngoại lệ, epoch20 là checkpoint cố định post-hoc. Callback lịch sử có decoded-label references, audit không chọn lại checkpoint.

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
> * Với Spanish, giao thức của dự án dùng BLEU tokenizer `13a` và chrF++ `word_order=2`. Tokenizer này chỉ áp dụng cho BLEU; chrF++ chính dùng văn bản gốc. Khi so với paper/benchmark khác cần khớp signature, không mặc định AmericasNLP hoặc mọi WMT paper cùng cấu hình; xem [METRICS.md](METRICS.md).

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

### Bảng 2: Bảng Đối Sánh Ngôn Ngữ Thứ Hai (Asháninka $\to$ Spanish, 1.003 Test Sentences)
| Backbone Architecture | Phương pháp | $\Delta\theta_{\text{add}}$ | $\theta_{\text{train}}$ (% Model) | BLEU (13a) ↑ | chrF++ (w=2) ↑ | Nhận định thực nghiệm AmericasNLP |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`mBART-large-50`** | Standard Fine-Tuning | 0 | 611M (100%) | 4.0085 | 20.8536 | Patience 4, chọn checkpoint theo validation chrF++ |
| | **CLRR-Enc + LSR — epoch 20** | **0** | **611M (100%)** | **4.0949** | **20.9588** | Patience 10, checkpoint epoch cuối theo yêu cầu tác giả; +0.0864 BLEU / +0.1051 chrF++ so với Baseline |
| | CLRR-Enc + LSR — epoch 14 | 0 | 611M (100%) | 3.7970 | 20.9753 | Patience 10, checkpoint chọn theo validation chrF++ |
| | CLRR-Enc + LSR — lượt gốc | 0 | 611M (100%) | 3.9671 | 20.6990 | Patience 4, cùng quy tắc dừng với Baseline |
| **`nllb-200-distilled-600M`** | Standard Fine-Tuning | 0 | 615M (100%) | 3.8594 | 20.4176 | Đã hoàn tất 20 epoch, chọn checkpoint theo validation chrF++ |
| | **CLRR-Enc + LSR (Ours)** | **0** | **615M (100%)** | *[Chưa chạy]* | *[Chưa chạy]* | Đang chờ tác giả lên kế hoạch tiếp |

Kết quả epoch 20 được ghi theo yêu cầu tác giả sau khi xem thêm checkpoint.
Đây là phân tích bổ sung một seed; Baseline dùng patience 4 nên so sánh với
lượt CLRR patience 10 chưa khớp quy tắc dừng. Không suy ra ưu thế có ý nghĩa
thống kê hoặc thay checkpoint chọn theo validation bằng checkpoint chọn từ test.

---

## 6. Hướng Dẫn Nghiệm Thu & Đóng Gói
1. Toàn bộ mã nguồn chạy mới đã được commit/sẵn sàng trong [`scripts/peft/run_strong_lora_nllb.py`](file:///d:/Code/CLRR/scripts/peft/run_strong_lora_nllb.py) và [`scripts/run_ashaninka_experiments.py`](file:///d:/Code/CLRR/scripts/run_ashaninka_experiments.py).
2. Khi các run trên Colab A100 hoàn tất, chỉ cần tải thư mục `results/` và `outputs_rebuttal/` về máy này để cập nhật trực tiếp vào [`clrr_main.tex`](file:///d:/Code/CLRR/docs/CLRR-paper/latex/clrr_main.tex).

## 7. Nghiệm thu số trước đợt train mới

36 outputs/42 bootstrap tests đã kiểm tra; cả hai clrr_main.tex được đồng bộ. Full−baseline mBART/NLLB và full−LSR-only chưa có ý nghĩa; Strong LoRA comparisons còn có ý nghĩa. Diagnostics SVD/probing/stability/latency chưa đủ evidence. Xem [RESULTS_VERIFICATION.md](RESULTS_VERIFICATION.md). Các lệnh runbook chỉ là hướng dẫn, không được thực thi trong audit này.
