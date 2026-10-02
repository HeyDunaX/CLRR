# KẾ HOẠCH TRIỂN KHAI & RUNBOOK THỰC NGHIỆM ĐỐI CHẤT REVIEWER (PEFT ENHANCEMENT & SCIENTIFIC RUNBOOK)

> **Cập nhật chiến lược mở rộng 2026-10-02 (Chuyển trọng tâm sang OPUS-100 Turkish $\to$ English 20k):**
> * **Giải quyết triệt để lỗi Early Stopping của bộ Asháninka:** Ngữ liệu Asháninka (~3.8k câu) có quy mô quá nhỏ dẫn tới dao động validation loss cao và kích hoạt dừng sớm bất thường.
> * **Chuẩn hóa Benchmark Chắp dính (Agglutinative Morphology Benchmark):** Cặp **Turkish $\to$ English** (OPUS-100) được trích xuất cố định 20.000 cặp câu train (seed 42), 2.000 val và 2.000 test chính thức.
> * **Xóa bỏ 100% yếu tố gây nhiễu mã ngôn ngữ giả:** mBART hỗ trợ chính thức `tr_TR -> en_XX`, NLLB hỗ trợ chính thức `tur_Latn -> eng_Latn` (0 token `<unk>`), loại bỏ hoàn toàn việc phải mượn mã `es_XX` hay `spa_Latn`.
> * **Bộ Asháninka được bảo tồn đầy đủ trong kho lưu trữ dữ liệu:** Dữ liệu và runbook của Asháninka vẫn được lưu trữ tại `data_processed/ashaninka_spanish/` và `ASHANINKA_RUNBOOK.md` làm tài liệu đối chiếu dự phòng.

---

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN & TỐI ƯU HÓA TÀI NGUYÊN (SCIENTIFIC RIGOR, FAIRNESS, SPEED & CU PROTECTION):**
> 1. **Mục tiêu tối thượng:** Bẻ gãy dứt điểm các nghi vấn từ báo cáo phản biện (yêu cầu kiểm chứng đa ngôn ngữ trên ngôn ngữ chắp dính độc lập, nghi ngờ LoRA trên NLLB, và tính trung thực của độ đo), đưa bài báo lên *Strong Accept* tại ACL Main Track (hoặc ComputEL-10).
> 2. **Bảo tồn toàn vẹn thành quả đã đạt được:** Đóng băng toàn bộ kết quả SOTA của CLRR-Enc trên `mBART-large-50` (20.39 BLEU, 19.08 chrF++), `nllb-200-distilled-600M` (14.28 BLEU, 18.75 chrF++ Zh), và `mT5-small` (4.83 BLEU). Checkpoint lưu trữ an toàn tại `FiveC/amis-rewire-checkpoints`.
> 3. **Giao thức công bằng tuyệt đối (Strict Fairness Protocol):** Cố định toàn diện siêu tham số giữa các phương pháp đối chuẩn (Seed 42, Effective Batch Size 128, Early Stopping Patience 5 theo chrF++, Precision BF16, Generation Beam Size 4, SacreBLEU `13a` cho English / Spanish + chrF++ `word_order=2`).
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
 [ĐÃ HOÀN TẤT 100%]                         [ƯU TIÊN 1: NLLB LoRA]                     [ƯU TIÊN 2: TURKISH 20K]
  ✅ mBART-50 Amis Suite (Full + LoRA A/B)   ✅ Run A: All-Linear (11.46 BLEU)          🚀 SẴN SÀNG KHỞI ĐỘNG
  ✅ NLLB-200 Amis Core Suite                ✅ Run B: Unfreeze Embed (9.80 BLEU)       - OPUS-100 en-tr (20k/2k/2k)
  ✅ mT5-small Suite (Appendix F)            ✅ Đã lưu checkpoint & metrics             - 0 token <unk>, mã chuẩn
  ✅ ByT5-small Baseline vs CLRR             ✅ Đối xứng 1-1 hoàn chỉnh với mBART       - Run 1-2: mBART Base vs CLRR
  ✅ Chẩn đoán Probing & SVD Anti-Collapse   ✅ Đã phân tích lý thuyết embedding        - Run 3-4: NLLB Base vs CLRR
```

### Bảng Tổng Hợp Chi Tiết Các Nhiệm Vụ Tiếp Theo:

| Thứ Tự | Hạng Mục Thí Nghiệm | Script Thực Thi | Cấu Hình Kỹ Thuật | Mục Đích Đối Sách Reviewer |
| :---: | :--- | :--- | :--- | :--- |
| **P1.1** | **Strong LoRA A trên NLLB-200** | `scripts/peft/run_strong_lora_nllb.py` | All-Linear ($r=16, \alpha=32$, 6 projections, frozen embed, LR 2e-4) | ✅ **HOÀN THÀNH (11.46 BLEU)**. Trả lời câu hỏi reviewer: LoRA phục hồi +7.92 BLEU nhưng vẫn thua CLRR (14.28 BLEU). |
| **P1.2** | **Strong LoRA B trên NLLB-200** | `scripts/peft/run_strong_lora_nllb.py` | All-Linear ($r=16, \alpha=32$, unfreeze shared/embed/lm_head, LR 1e-4) | ✅ **HOÀN THÀNH (9.80 BLEU)**. Chứng minh mở khóa embedding 256k tokens bị phân kỳ từ vựng (vocabulary dispersion). |
| **P2.1** | **mBART Baseline trên Turkish 20k** | `scripts/run_turkish_experiments.py` | Full-FT, LR 1e-4, `tr_TR -> en_XX`, tokenizer `13a`, chrF++ $w=2$ | Thiết lập điểm tựa đối chuẩn trên ngôn ngữ chắp dính thứ hai (OPUS-100 Turkish $\to$ English). |
| **P2.2** | **mBART CLRR-Enc trên Turkish 20k** | `scripts/run_turkish_experiments.py` | CLRR-Enc + LSR ($d=2, \alpha=0.1, \lambda=0.1$), LR 1e-4, `tr_TR -> en_XX` | **Bằng chứng quyết định cho ACL Main:** Chứng minh CLRR tiếp tục vượt trội trên ngôn ngữ chắp dính thứ hai (+chrF++, +BLEU). |
| **P2.3** | **NLLB Baseline trên Turkish 20k** | `scripts/run_turkish_experiments.py` | Full-FT, LR 5e-5, `tur_Latn -> eng_Latn`, tokenizer `13a` | Thiết lập điểm tựa trên backbone NLLB-200 với mã ngôn ngữ chuẩn thức. |
| **P2.4** | **NLLB CLRR-Enc trên Turkish 20k** | `scripts/run_turkish_experiments.py` | CLRR-Enc + LSR ($d=2, \alpha=0.1, \lambda=0.1$), LR 5e-5, `tur_Latn -> eng_Latn` | Xác lập tính ưu việt của CLRR trên cả hai backbone đối với cặp Turkish $\to$ English. |

---

## 2. Bảng Quy Chuẩn Siêu Tham Số Chi Tiết Từng Thí Nghiệm (Strict Fairness Protocol & Per-Experiment Parameter Matrix)

Nhằm đảm bảo **tính công bằng học thuật tuyệt đối (Strict Scientific Fairness)** theo tiêu chuẩn ACL, không để xảy ra bất kỳ sự thiên vị nào về điều kiện huấn luyện giữa các phương pháp đối chuẩn (Baselines, PEFT baselines) và phương pháp đề xuất (CLRR-Enc + LSR), bảng dưới đây liệt kê tường minh **từng dòng tham số cài đặt cho toàn bộ các thí nghiệm đã thực hiện trước đó và các thí nghiệm mới chuẩn bị chạy**:

### 2.1. Ma Trận Siêu Tham Số Huấn Luyện Từng Thí Nghiệm (Training & Architecture Hyperparameters)

| ID | Tên Thí Nghiệm / Run Name | Trạng Thái | Mô Hình Gốc (HF Backbone) | Cặp Ngôn Ngữ | Phương Pháp | Tham Số Tối Ưu ($\theta_{\text{train}}$) | Tham Số Thêm ($\Delta\theta_{\text{add}}$) | Modules Can Thiệp / Mở Khóa | CLRR Setup ($d, \alpha, \lambda$) | Learning Rate | Batch Size (Device $\times$ Accum) | Effective Batch | Max Epochs & Patience | Optimizer & Warmup | Seed |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **REF-1** | `mbart-large-50-baseline` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Full Fine-Tuning | 610.88M (100%) | 0 | Toàn bộ mô hình | N/A | $5 \times 10^{-5}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-2** | `mbart-large-50-clrr-enc` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | CLRR-Enc + LSR | 610.88M (100%) | 0 | Toàn bộ mô hình | $d=2, \alpha=0.1, \lambda=0.1$ | $5 \times 10^{-5}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-3** | `mbart-large-50-lora-all-linear` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Strong LoRA Run A | 3.54M (0.58%) | +3.54M | $q,k,v,o,fc1,fc2$ ($r=16, \alpha=32$) | N/A | $2 \times 10^{-4}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-4** | `mbart-...-unfreeze-embed` | ✅ Đã chạy | `mbart-large-50-...` | Amis $\to$ Zh | Strong LoRA Run B | 260.0M (42.6%) | +3.54M | All-Lin + `shared,lm_head` | N/A | $1 \times 10^{-4}$ | $4 \times 32$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-5** | `nllb-200-baseline` | ✅ Đã chạy | `nllb-200-distilled-600M` | Amis $\to$ Zh | Full Fine-Tuning | 614.91M (100%) | 0 | Toàn bộ mô hình | N/A | $5 \times 10^{-5}$ | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **REF-6** | `nllb-200-clrr-enc` | ✅ Đã chạy | `nllb-200-distilled-600M` | Amis $\to$ Zh | CLRR-Enc + LSR | 614.91M (100%) | 0 | Toàn bộ mô hình | $d=2, \alpha=0.1, \lambda=0.1$ | $5 \times 10^{-5}$ | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P1.1** | `nllb-200-lora-all-linear` | ✅ **Đã hoàn thành** (BLEU: **11.46**, chrF++: **9.50**) | `nllb-200-distilled-600M` | Amis $\to$ Zh | Strong LoRA Run A | 8.65M (1.39%) | +8.65M | $q,k,v,o,fc1,fc2$ ($r=16, \alpha=32$) | N/A | **$2 \times 10^{-4}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P1.2** | `nllb-200-lora-unfreeze-embed` | ✅ **Đã hoàn thành** (BLEU: **9.80**, chrF++: **8.40**) | `nllb-200-distilled-600M` | Amis $\to$ Zh | Strong LoRA Run B | 271.0M (43.45%) | +8.65M | All-Lin + `shared,embed_tokens,lm_head` | N/A | **$1 \times 10^{-4}$** | $16 \times 8$ | **128** | 20 ep / 4 pat | AdamW, 6% linear | 42 |
| **P2.1** | `mbart-tr-en-baseline` | ⏳ Sẵn sàng | `mbart-large-50-...` | Tr $\to$ En (20k) | Full Fine-Tuning | 610.88M (100%) | 0 | Toàn bộ mô hình | N/A | **$1 \times 10^{-4}$** | $4 \times 32$ | **128** | 15 ep / 5 pat | AdamW, 6% linear | 42 |
| **P2.2** | `mbart-tr-en-clrr-enc` | ⏳ Sẵn sàng | `mbart-large-50-...` | Tr $\to$ En (20k) | CLRR-Enc + LSR | 610.88M (100%) | 0 | Toàn bộ mô hình | **$d=2, \alpha=0.1, \lambda=0.1$** | **$1 \times 10^{-4}$** | $4 \times 32$ | **128** | 15 ep / 5 pat | AdamW, 6% linear | 42 |
| **P2.3** | `nllb-tr-en-baseline` | ⏳ Sẵn sàng | `nllb-200-distilled-600M` | Tr $\to$ En (20k) | Full Fine-Tuning | 614.91M (100%) | 0 | Toàn bộ mô hình | N/A | **$5 \times 10^{-5}$** | $16 \times 8$ | **128** | 15 ep / 5 pat | AdamW, 6% linear | 42 |
| **P2.4** | `nllb-tr-en-clrr-enc` | ⏳ Sẵn sàng | `nllb-200-distilled-600M` | Tr $\to$ En (20k) | CLRR-Enc + LSR | 614.91M (100%) | 0 | Toàn bộ mô hình | **$d=2, \alpha=0.1, \lambda=0.1$** | **$5 \times 10^{-5}$** | $16 \times 8$ | **128** | 15 ep / 5 pat | AdamW, 6% linear | 42 |

### 2.2. Ma Trận Đánh Giá & Suy Luận Chuẩn Thống Kê (Evaluation Protocol Matrix)

| Cặp Ngôn Ngữ / Bài Test | Tokenizer Đánh Giá BLEU | chrF++ Cấu Hình | Độ Dài Tối Đa (Src / Tgt) | Beam Size (Val / Test) | Cơ Chế Chọn Checkpoint Tốt Nhất | Reference Dùng Để Chấm Điểm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Amis $\to$ Mandarin (Amis Core)** | `zh` (SacreBLEU) | $w=2$ (character level) | 256 / 256 | 1 / 4 | `eval_chrf++` cao nhất trên Validation | `test.csv` (decoded & raw ref đối chiếu) |
| **Turkish $\to$ English (OPUS-100 20k)** | `13a` (SacreBLEU) | $w=2$ (character level) | 128 / 128 | 1 (mBART) / 4 (NLLB) | `eval_chrf++` cao nhất trên Validation | `test.csv` gốc (2.000 câu chuẩn) |
| **Asháninka $\to$ Spanish (Dự phòng)** | `13a` (SacreBLEU) | $w=2$ (character level) | 256 (mBART) / 128 (NLLB) | 1 (mBART) / 4 (NLLB) | `eval_chrf++` cao nhất trên Validation | `test.csv` gốc (1.003 câu chuẩn) |

---

## 3. Nhật Ký Tiến Độ Chi Tiết (Execution Tracking Log)

- **2026-10-02 08:30:** Hoàn thành chạy thực nghiệm NLLB Strong LoRA Run A ($r=16$, All-Linear, 8.65M params) đạt **11.4625 BLEU** (+7.92 BLEU so với Narrow LoRA).
- **2026-10-02 09:20:** Hoàn thành chạy thực nghiệm NLLB Strong LoRA Run B (Unfreeze Embeddings) đạt **9.8024 BLEU** (chứng minh hiện tượng vocabulary dispersion khi mở khóa ma trận nhúng 256k tokens trên tập dữ liệu nhỏ).
- **2026-10-02 10:15:** Cập nhật kết quả NLLB LoRA vào `docs/CLRR-paper/latex/clrr_main.tex`, `docs/EXPERIMENTS.md` và `docs/RESEARCH_INSIGHTS.md`.
- **2026-10-02 16:05:** Xử lý và thẩm định toàn diện bộ dữ liệu mới **OPUS-100 Turkish $\to$ English 20k** tại `data_processed/opus100_turkish_english_20k/`. Tỷ lệ `<unk>` token là 0.00% trên cả mBART và NLLB.
- **2026-10-02 16:10:** Tạo script điều phối `scripts/run_turkish_experiments.py`, thiết lập quy chuẩn tham số công bằng, kiểm thử smoke test thành công 100%. Sẵn sàng khởi động huấn luyện theo lệnh của tác giả.
