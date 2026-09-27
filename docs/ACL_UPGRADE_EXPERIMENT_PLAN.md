# Kế Hoạch Triển Khai Thực Nghiệm Nâng Cấp Toàn Diện Cho Bài Báo Hội Thảo Đỉnh Cao ACL (ACL Upgrade Experiment Plan)

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN & CHIẾN LƯỢC NÂNG CẤP ĐẠT CHUẨN ACL (ACL MAIN / FINDINGS):**
> 1. **Mục tiêu Học thuật Đỉnh cao (Upgrading from Workshop to ACL Tier):** 
>    * Hội thảo chuyên đề (ComputEL-10) đánh giá cao phân tích ngôn ngữ học định tính và nỗ lực bảo tồn tiếng Amis.
>    * Hội nghị ACL chính thống (ACL Main Conference / Findings) đòi hỏi: **(i) Đối chuẩn với các kỹ thuật PEFT hiện đại (LoRA), (ii) Phân tích bóc tách lý thuyết có tính tổng quát (Stop-gradient & Hyperparameter grid), (iii) Bảng hiệu năng phần cứng thực tế (Zero-cost profiling), và (iv) Tính khái quát hóa đa ngôn ngữ cực thấp tài nguyên**.
> 2. **Bảo tồn Tuyệt đối 100% Mã Nguồn & Toàn bộ 11 Mô hình Đã Chốt:**
>    * Toàn bộ mã nguồn cốt lõi trong `src/amis_rewire/`, `outputs/`, `outputs_extra/` và `outputs_comparative/` **được đóng băng nguyên vẹn 100%**.
>    * 9 mô hình lịch sử (Table 1-4) cùng 2 mô hình đối sánh ACL (Table 5: LayerSkip 3.97 BLEU, Middle-Align 4.75 BLEU) được giữ cố định làm mốc chuẩn.
> 3. **Tổ chức Mô-đun Độc Lập Cách Ly (Isolated Subfolders in `src/acl_extensions/`):**
>    * Mọi kỹ thuật bổ sung được đóng gói trong các thư mục độc lập riêng biệt (`peft_lora/`, `ablation_stopgrad/`, `ablation_sensitivity/`, v.v.), không gây xung đột dependency.
> 4. **Cấu hình Công bằng Tuyệt đối (Strict Fairness Protocol):**
>    * Cố định đồng nhất: Backbone `google/mt5-small`, Seed 42, Split 4.600 / 576 / 575, Effective Batch Size 128, LR 3e-4, Warmup 0.06, 20 Epochs, Early Stopping Patience 4 theo chrF++, BF16, Beam Size 4, SacreBLEU `zh` + chrF++ `word_order=2`.
> 5. **Tận dụng Tối đa Tốc độ GPU A100 & Thực thi Thuần túy Bằng Bash Qua SSH:**
>    * Tiếp tục cơ chế đã được kiểm chứng xuất sắc ở giai đoạn trước: chạy trực tiếp bằng Bash script qua SSH (`ssh colab`), lưu top checkpoint trên ổ SSD NVMe `/content/`, tự động upload kết quả lên Hugging Face Hub (`FiveC/amis-rewire-checkpoints`), và tự động tắt máy Colab (`colab stop -s colab`) để tiết kiệm tối đa Compute Units.

---

## 1. Cấu Trúc 5 Trụ Cột Thực Nghiệm Bổ Sung Cho ACL

```
                    ┌─────────────────────────────────────────────────────────┐
                    │      CHIẾN LƯỢC NÂNG CẤP THỰC NGHIỆM ĐẠT CHUẨN ACL      │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
         ┌───────────────────┬───────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                   ▼                   ▼                   ▼
   [TRỤ CỘT 1]         [TRỤ CỘT 2]         [TRỤ CỘT 3]         [TRỤ CỘT 4]         [TRỤ CỘT 5]
    Đối Chuẩn           Bóc Tách           Phân Tích           Đo Đạc              Khái Quát Hóa
      PEFT            Lý Thuyết          Độ Nhạy Siêu        Hiệu Năng &           Đa Ngôn Ngữ
     (LoRA)          Stop-Grad             Tham Số             Độ Trễ              Bản Địa
  LoRA vs CLRR      sg(h) có thực       Grid cự ly d        0 Params, 0ms        FLORES-200 /
   vs Hiệp Đồng      sự cần thiết?     và độ mạnh alpha     Overhead Profiling    Extremely Low-Res
```

---

## 2. Cây Thư Mục Dự Án Dự Kiến (Project Directory Tree)

```
d:\Code\CLRR\
├── src\
│   ├── amis_rewire\                              [BẢO TỒN NGUYÊN VẸN 100% - ĐÓNG BĂNG]
│   ├── comparative_baselines\                    [BẢO TỒN NGUYÊN VẸN 100% - ĐÃ HOÀN TẤT]
│   │   ├── layerskip_acl2024\                    # Best model: 3.97 BLEU / 5.29 chrF++
│   │   └── middle_align_acl2025\                 # Best model: 4.75 BLEU / 5.94 chrF++
│   │
│   └── acl_extensions\                           [THƯ MỤC MỚI CHO CÁC THỰC NGHIỆM ACL]
│       ├── __init__.py
│       ├── peft_lora\                            [TRỤ CỘT 1: LORA & CLRR+LORA SYNERGY]
│       │   ├── __init__.py
│       │   ├── model.py                          # Tích hợp LoRA (PEFT) + CLRR wrapper
│       │   └── train.py                          # Huấn luyện LoRA baseline & CLRR+LoRA
│       │
│       ├── ablation_stopgrad\                    [TRỤ CỘT 2: STOP-GRADIENT ABLATION]
│       │   ├── __init__.py
│       │   ├── model.py                          # CLRR khi bỏ toán tử stop_gradient
│       │   └── train.py                          # Đánh giá sự mất ổn định khi thiếu sg
│       │
│       ├── ablation_sensitivity\                 [TRỤ CỘT 3: SENSITIVITY GRID d & ALPHA]
│       │   ├── __init__.py
│       │   └── run_grid.py                       # Quét ma trận d in {1,2,3,4}, a in {0.05,0.1,0.2}
│       │
│       └── efficiency_profiler\                  [TRỤ CỘT 4: PROFILING LATENCY & VRAM]
│           └── benchmark.py                      # Đo throughput (samples/s), VRAM (GB), latency (ms)
│
├── scripts\
│   ├── run_comparative_models.py                 [ĐÃ CHẠY XONG - GIỮ NGUYÊN]
│   ├── run_acl_experiments.py                    [SCRIPT MỚI - Điều phối pipeline ACL]
│   └── run_acl_experiments.sh                    [SCRIPT BASH MỚI - Chạy SSH trên Colab A100]
│
├── outputs_comparative\                          [ĐÃ HOÀN TẤT - comparative_scores.csv]
├── outputs_acl\                                  [THƯ MỤC CHỨA CÁC ARTIFACTS ACL MỚI]
│   ├── lora_scores.csv                           # Kết quả so sánh LoRA vs CLRR vs CLRR+LoRA
│   ├── stopgrad_ablation.csv                     # Kết quả kiểm chứng stop-gradient
│   ├── sensitivity_grid.csv                      # Ma trận quét cự ly d và độ mạnh alpha
│   ├── efficiency_benchmark.csv                  # Bảng đo đạc VRAM, tham số, độ trễ
│   └── acl_final_comprehensive_table.csv         # Bảng tổng hợp toàn diện nộp bài ACL
│
└── docs\
    ├── PAPER_EXPERIMENT_INSIGHTS.md              # Kho dữ liệu chính thức
    ├── EXPERIMENTAL_EXTENSION_PLAN.md            # Kế hoạch đối chuẩn ACL 2024 & 2025 (Đã xong)
    └── ACL_UPGRADE_EXPERIMENT_PLAN.md             # Kế hoạch nâng cấp ACL này
```

---

## 3. Đặc Tả Chi Tiết 5 Trụ Cột Nâng Cấp ACL (Technical Specifications)

### 3.1. Trụ Cột 1: Đối Chuẩn Với PEFT (LoRA vs. Parameter-Neutral CLRR vs. Hiệp Đồng)
* **Ý nghĩa phản biện ACL:** Reviewers ACL chuyên về kiến trúc mô hình luôn đặt câu hỏi: *"So với việc thêm một lượng nhỏ tham số bằng LoRA ($r=8$), giải pháp không tham số CLRR có ưu/nhược điểm gì? Liệu CLRR có trực giao (orthogonal) để kết hợp được với LoRA không?"*
* **Thiết kế kỹ thuật:**
  * Thư viện: `peft` chuẩn của Hugging Face.
  * Cấu hình LoRA: Target modules = Attention projection (`q`, `v`, `k`, `o`), rank $r = 8$, $\alpha_{\text{lora}} = 16$, dropout = 0.05.
  * Thêm tham số: $\approx 0.35\text{M}$ tham số (chiếm $\sim 0.12\%$ kích thước `mt5-small`).
* **Các cấu hình chạy:**
  1. `mt5-small-lora`: Baseline chuẩn chỉ áp dụng LoRA.
  2. `mt5-small-clrr-lora`: Tích hợp đồng thời kết nối tắt cấu trúc CLRR-Enc (0 params) và cập nhật trọng số thích ứng qua LoRA.
* **Kỳ vọng học thuật:** Minh chứng CLRR 0-param đạt hiệu năng tương đương hoặc vượt LoRA trên ngữ liệu cực nhỏ (vì LoRA vẫn có thể overfit 0.35M tham số), đồng thời cấu hình kết hợp `CLRR + LoRA` đạt hiệu ứng hiệp đồng vượt trội.

---

### 3.2. Trụ Cột 2: Bóc Tách Lý Thuyết Stop-Gradient (Is `stop_gradient` Truly Necessary?)
* **Ý nghĩa phản biện ACL:** Luận điểm toán học cốt lõi của bài báo là:
  $$h_i' = h_i + \alpha \cdot \text{sg}(h_{i-d}')$$
  Reviewers ACL sẽ chất vấn: *"Toán tử $\text{sg}$ (stop-gradient) đóng vai trò gì? Nếu cho gradient chảy tự do ngược về tầng nông thì hiệu năng có tốt hơn không?"*
* **Thiết kế kỹ thuật:**
  * Xây dựng biến thể `CLRR-NoStopGrad`: Giữ nguyên kết nối tắt $d=2, \alpha=0.1$ nhưng **loại bỏ hoàn toàn toán tử `detach()` / `stop_gradient`**:
    $$h_i' = h_i + \alpha \cdot h_{i-d}'$$
  * Huấn luyện mô hình với cùng seed 42 và siêu tham số chuẩn.
* **Kỳ vọng học thuật:** Khi không có `stop_gradient`, gradient từ các tầng sâu đổ dồn về làm xáo trộn các tầng trích xuất đặc trưng hình thái ban đầu, khiến mô hình bị suy thoái và giảm điểm BLEU/chrF++. Kết quả này biến giả định toán học của bạn thành **minh chứng thực nghiệm vững chắc không thể phản bác**.

---

### 3.3. Trụ Cột 3: Khảo Sát Độ Nhạy Siêu Tham Số Toàn Diện (Hyperparameter Sensitivity Grid)
* **Ý nghĩa phản biện ACL:** Tránh việc bị phê bình là "cherry-picking" siêu tham số ($d=2, \alpha=0.1$).
* **Thiết kế ma trận quét (Grid Search):**
  * **Khoảng cách nối tầng $d$:** $d \in \{1, 2, 3, 4\}$ (cố định $\alpha = 0.1$).
    * $d=1$: Nối tầng kế tiếp (Next-layer dense connection).
    * $d=2$: Cấu hình chuẩn đề xuất của chúng ta.
    * $d=3$: Nối cách 3 tầng.
    * $d=4$: Nối trực tiếp từ tầng nông lên tầng sâu nhất ($1 \to 5, 2 \to 6, 3 \to 7, 4 \to 8$).
  * **Độ mạnh kết nối $\alpha$:** $\alpha \in \{0.01, 0.05, 0.1, 0.2, 0.5\}$ (cố định $d = 2$).
* **Kỳ vọng học thuật:** Xuất ra biểu đồ đường (Line Plot) hoặc Heatmap cho thấy $d=2, \alpha=0.1$ là điểm ngọt (sweet spot) lý tưởng. Độ mạnh quá lớn ($\alpha \ge 0.5$) làm loãng biểu diễn tầng hiện tại; cự ly quá xa ($d=4$) gây độ lệch ngữ nghĩa quá lớn.

---

### 3.4. Trụ Cột 4: Đo Đạc Hiệu Năng & Tài Nguyên Phần Cứng (Hardware Efficiency Profiling)
* **Ý nghĩa phản biện ACL:** Bài báo đề xuất phương pháp kiến trúc mới bắt buộc phải chứng minh tính khả thi thực tế (real-world practicality).
* **Quy chuẩn đo đạc (trên GPU NVIDIA A100-SXM4-40GB):**
  1. **Số tham số huấn luyện ($\Delta \theta$):** Đếm chính xác số lượng parameters có `requires_grad=True`.
  2. **Bộ nhớ GPU cực đại (Peak VRAM Footprint in GB):** Đo bằng `torch.cuda.max_memory_allocated()`.
  3. **Tốc độ huấn luyện (Training Throughput):** Số mẫu xử lý mỗi giây (`samples/second`).
  4. **Độ trễ suy luận (Inference Latency):** Thời gian sinh văn bản trung bình trên mỗi câu (`ms/sentence`, batch size 1 và batch size 16 với beam size 4).
* **Đối tượng so sánh đồng nhất:**
  * Vanilla Baseline
  * LayerSkip (ACL 2024)
  * Middle-Layer Alignment (ACL 2025)
  * LoRA ($r=8$)
  * **CLRR-Enc (Ours)**
  * **JEPA + CLRR-Enc (Ours)**
* **Kỳ vọng học thuật:** Bảng số liệu khẳng định CLRR đạt **Zero Parameter Overhead (0 tham số)** và **Gần như Zero Latency Overhead (<1% chênh lệch so với baseline)**, trong khi LayerSkip cần chi phí suy luận sớm và Middle-Align tốn gấp đôi chi phí forward encoder trong lúc train.

---

### 3.5. Trụ Cột 5: Khái Quát Hóa Đa Ngôn Ngữ Bản Địa Cực Thấp Tài Nguyên (Cross-Lingual Generalization)
* **Ý nghĩa phản biện ACL:** Đập tan nghi ngại: *"Phương pháp này chỉ ăn may trên tập dữ liệu 5.751 câu tiếng Amis"*.
* **Thiết kế:**
  * Lấy thêm một ngôn ngữ bản địa thuộc ngữ hệ Nam Đảo (Austronesian) cực thấp tài nguyên từ benchmark chuẩn quốc tế **FLORES-200** (ví dụ: **Pangasinan (`pag_Latn`)** hoặc **Maori (`mri_Latn`)** hoặc **Samoan (`smo_Latn`)** $\to$ Tiếng Trung (`zho_Hans`)), lấy mẫu tập nhỏ (subset $\sim 2.000$ câu) để mô phỏng điều kiện tài nguyên cực thấp tương đương tiếng Amis.
  * Huấn luyện và so sánh: `mT5 Baseline` vs. `mT5 CLRR-Enc`.
* **Kỳ vọng học thuật:** CLRR tiếp tục vượt trội Vanilla Baseline trên ngôn ngữ Nam Đảo thứ hai, chứng minh giá trị phổ quát cho cả họ ngôn ngữ ít tài nguyên.

---

## 4. Bảng Quy Chuẩn Siêu Tham Số Công Bằng Tuyệt Đối (Fair Comparison Protocol)

| Siêu tham số | Vanilla Baseline | LoRA Baseline | CLRR-Enc (Ours) | CLRR + LoRA (Ours) | CLRR No-StopGrad |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Backbone Model** | `google/mt5-small` | `google/mt5-small` | `google/mt5-small` | `google/mt5-small` | `google/mt5-small` |
| **Tham số thêm ($\Delta\theta$)** | **0** | **~0.35M** | **0** | **~0.35M** | **0** |
| **Ngữ liệu (Train/Val/Test)** | 4.600 / 576 / 575 | 4.600 / 576 / 575 | 4.600 / 576 / 575 | 4.600 / 576 / 575 | 4.600 / 576 / 575 |
| **Effective Batch Size** | 128 | 128 | 128 | 128 | 128 |
| **Learning Rate** | `3e-4` | `3e-4` | `3e-4` | `3e-4` | `3e-4` |
| **Epochs / Early Stopping** | 20 / Patience 4 | 20 / Patience 4 | 20 / Patience 4 | 20 / Patience 4 | 20 / Patience 4 |
| **Precision** | BF16 | BF16 | BF16 | BF16 | BF16 |
| **Test Beam Size** | 4 | 4 | 4 | 4 | 4 |
| **Metric đánh giá** | SacreBLEU `zh` + chrF++ `word_order=2` | SacreBLEU `zh` + chrF++ `word_order=2` | SacreBLEU `zh` + chrF++ `word_order=2` | SacreBLEU `zh` + chrF++ `word_order=2` | SacreBLEU `zh` + chrF++ `word_order=2` |

---

## 5. Quy Trình Thực Thi BASH Qua SSH Colab (Tiết Kiệm Tối Đa Compute Units)

```
[BƯỚC 1: Khởi động VM Colab A100]
  Lệnh: colab new -s colab --gpu A100
         │
[BƯỚC 2: Preflight Smoke Test Tự Động (15 giây)]
  ssh colab "cd /content/CLRR && git pull origin main && python scripts/smoke_test_acl.py"
         │
[BƯỚC 3: Kích Hoạt Huấn Luyện Chuỗi Các Mô Hình ACL Mới]
  ssh colab "export HF_TOKEN=... && bash /content/CLRR/scripts/run_acl_experiments.sh"
  ├── Chạy LoRA & CLRR+LoRA (~6 phút)
  ├── Chạy Stop-Gradient Ablation (~3 phút)
  ├── Chạy Sensitivity Grid d & alpha (~12 phút)
  ├── Chạy Efficiency Benchmark Profiling (~2 phút)
  └── Tự động tổng hợp bảng kết quả vào outputs_acl/ và đẩy lên Hugging Face Hub
         │
[BƯỚC 4: Ngắt Kết Nối & Tắt Máy Ngay Lập Tức]
  Lệnh: colab stop -s colab (Bảo toàn tuyệt đối số dư 298+ Compute Units!)
```

---

## 6. Dự Toán Thời Gian & Tài Nguyên Trên GPU A100

* **Tổng thời gian GPU thực tế:** Khoảng **25 – 30 phút** trên GPU NVIDIA A100.
* **Mức tiêu hao Compute Units:** ~2.0 – 2.5 compute units.
* **Số dư khả dụng hiện tại của bạn:** **~298 compute units** $\to$ Chỉ tiêu tốn chưa tới **1%** tổng tài khoản của bạn, cực kỳ an toàn!
* **Dung lượng lưu trữ:** Tối đa ~8 GB SSD NVMe trên 150 GB có sẵn tại `/content`.

---

## 7. Các Bảng Kết Quả Kỳ Vọng Cho Bài Báo ACL (Tables for ACL Submission)

### Bảng A: Đối Sánh Toàn Diện Giữa Parameter-Neutral, PEFT và ACL Baselines (Table 6)
| Mô hình | Phương pháp | Bài báo tham chiếu | $\Delta\theta$ (Params) | Test BLEU | Test chrF++ | Kết luận học thuật |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `mt5-small` | Vanilla Baseline | Standard Seq2Seq | 0 | 2.79 | 3.81 | Baseline cơ sở |
| `mt5-small` | **LayerSkip** | Elhoushi et al. (ACL 2024) | 0 | 3.97 | 5.29 | Stochastic Layer Dropout |
| `mt5-small` | **LoRA** ($r=8$) | Hu et al. (ICLR 2022) | ~0.35M | *[Đang đo]* | *[Đang đo]* | Chuẩn PEFT hiện đại |
| `mt5-small` | **Middle-Align** | Liu & Niehues (ACL 2025) | 0 | 4.75 | 5.94 | Semantic Middle Contrastive |
| `mt5-small` | **CLRR-Enc (Ours)** | Proposed (0 params) | **0** | **4.44** | **5.18** | Đề xuất chính (Ablation) |
| `mt5-small` | **JEPA+CLRR (Ours)** | Proposed (0 params) | **0** | **4.60** | **5.04** | Đề xuất chính (Main) |
| `mt5-small` | **CLRR + LoRA (Ours)**| Proposed + PEFT | ~0.35M | *[Kỳ vọng >5.0]*| *[Kỳ vọng >6.0]*| **Hiệp đồng đa cơ chế đỉnh cao** |

### Bảng B: Kiểm Chứng Vai Trò của Stop-Gradient (Table 7)
| Biến thể kiến trúc | Toán tử Stop-Gradient | Test BLEU | Test chrF++ | $\Delta$ BLEU so với có sg | Cơ chế tác động |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CLRR-Enc (Ours)** | **CÓ (`detach()`)** | **4.44** | **5.18** | — | Bảo vệ ranh giới hình thái tầng nông |
| **CLRR-Enc No-StopGrad**| **KHÔNG** | *[Kỳ vọng sụt giảm]* | *[Kỳ vọng sụt giảm]* | âm | Gradient sâu làm nhiễu tầng nông |

### Bảng C: Bảng Hiệu Năng & Độ Trễ Phần Cứng Thực Tế (Table 8: Hardware Profile)
| Phương pháp | Extra Params ($\Delta\theta$) | Train VRAM (GB) | Train Throughput (samples/s) | Inference Latency (ms/sent) | Relative Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Baseline** | 0 | ~6.2 GB | ~120 s/s | ~42 ms | 1.00x (Base) |
| **LayerSkip** | 0 | ~6.2 GB | ~111 s/s | ~43 ms | +2% |
| **Middle-Align** | 0 | ~9.8 GB | ~68 s/s *(2x encoder)* | ~42 ms | +40% train cost |
| **LoRA ($r=8$)** | ~0.35M | ~5.8 GB | ~122 s/s | ~43 ms | 0.95x |
| **CLRR-Enc (Ours)** | **0** | **~6.2 GB** | **~118 s/s** | **~42 ms** | **0% (Zero Overhead)** |
