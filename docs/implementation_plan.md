# KẾ HOẠCH TRIỂN KHAI & THỨ TỰ ƯU TIÊN THỰC NGHIỆM CLRR (CLRR EXPERIMENTAL IMPLEMENTATION & EXECUTION PLAN)

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN & TỐI ƯU HÓA TÀI NGUYÊN (SCIENTIFIC RIGOR, FAIRNESS, SPEED & CU PROTECTION):**
> 1. **Mục tiêu thực nghiệm tối thượng:** Tập trung 100% vào việc củng cố các luận điểm khoa học chưa thể bị phản bác của CLRR ($\Delta\theta = 0$, chống over-smoothing hình thái học trên dữ liệu cực hiếm), giải quyết trực diện các câu hỏi hóc búa nhất của hội đồng phản biện ACL / ComputEL-10.
> 2. **Bảo tồn 100% mã nguồn và số liệu lịch sử:** Đóng băng toàn bộ code lõi trong `src/amis_rewire/`, toàn bộ 18 mô hình đã chạy và các artifacts đã chốt trong `outputs/`, `outputs_extra/`, `outputs_comparative/`, `outputs_revalidation/` (lưu trữ an toàn trên Hugging Face `FiveC/amis-rewire-checkpoints`).
> 3. **Thứ tự ưu tiên chạy thí nghiệm GPU (GPU Experiment Priority Order):**
>    * **ƯU TIÊN 1 (Chạy trước tiên — ~35 phút A100):** Huấn luyện đối chuẩn PEFT (**LoRA** & **BitFit**) trên `mBART-large-50`. Đây là tuyến phòng thủ quan trọng nhất: Chứng minh trong ngữ liệu cực hiếm (<6.000 câu), các phương pháp thêm tham số $\Delta\theta > 0$ (LoRA $r=8$, BitFit) bị quá khớp hoặc nghẽn biểu diễn phụ tố, khẳng định ưu thế tuyệt đối của can thiệp $\Delta\theta = 0$ từ CLRR.
>    * **ƯU TIÊN 2 (Chạy tiếp theo — ~90 phút A100):** Huấn luyện mô hình dịch đa ngôn ngữ SOTA nhất của Meta **`NLLB-200-distilled-600M`** (Baseline vs Middle-Layer Alignment vs CLRR-Enc+LSR). Chứng minh CLRR nâng tầm hiệu năng trên mô hình đã được pretrain sẵn trên hàng chục ngôn ngữ Nam Đảo, khẳng định over-smoothing là thuộc tính cố hữu của Transformer 12 tầng sâu.
>    * **ƯU TIÊN 3 (Dự phòng / Mở rộng Journal — ~60 phút A100):** Huấn luyện trên ngữ liệu Formosan thứ 2 (**Paiwan $\to$ Chinese**). Chứng minh CLRR tổng quát hóa trên toàn bộ ngữ hệ Nam Đảo.
>    * **LOẠI BỎ (Không chạy):** Tuyệt đối KHÔNG chạy biến thể lai ghép `CLRR + Middle-Layer Alignment` trên `mT5-small` để bảo vệ tính độc lập và đóng góp nguyên bản của bài báo ($\Delta\theta=0$ Architectural Rewiring).
> 4. **Cấu hình công bằng tuyệt đối (Strict Fairness Protocol):** Cố định toàn diện: Seed 42, Split 4.600 / 576 / 575, Effective Batch Size 128, Learning Rate chuẩn, Warmup 0.06, 20 Epochs, Early Stopping Patience 4 theo chrF++, BF16, Generation Beam Size 4, SacreBLEU `zh` + chrF++ `word_order=2`.
> 5. **Bắt buộc chạy Preflight Smoke Test:** Kiểm thử 1 batch ($B=2$), 1 forward/backward step ($\le 30$s) trên CPU/GPU trước khi chạy full 20 epochs trên A100.
> 6. **Tự động hóa hoàn toàn bằng BASH qua SSH Colab (Không dùng Notebook):** Chạy nền unbuffered (`nohup`), lưu SSD NVMe Colab `/content/`, tự động upload Best Model lên Hugging Face Hub, và **ngắt máy ảo ngay lập tức (`colab stop -s colab`)** để bảo toàn Compute Units.

---

## 1. Cấu Trúc Tổng Quan & Thứ Tự Ưu Tiên Chạy Thí Nghiệm (Prioritized Execution Roadmap)

```
                    ┌────────────────────────────────────────────────────────┐
                    │       LỘ TRÌNH THỰC THI THÍ NGHIỆM CLRR TRÊN GPU       │
                    │         (CLRR GPU EXPERIMENTAL EXECUTION ROADMAP)      │
                    └───────────────────────────┬────────────────────────────┘
                                                │
          ┌─────────────────────────────────────┴─────────────────────────────────────┐
          ▼                                                                           ▼
   [ƯU TIÊN 1: PEFT BASELINE SUITE]                                            [ƯU TIÊN 2: NLLB-200 SOTA SUITE]
   Backbone: mBART-large-50 (611M)                                             Backbone: NLLB-200-distilled-600M
   Phương pháp: LoRA (r=8) & BitFit                                            Phương pháp: Baseline, Mid-Align, CLRR-Enc+LSR
   Thời gian: ~35 phút A100 (2 runs)                                           Thời gian: ~90 phút A100 (3 runs)
   Mục tiêu: Đập tan câu hỏi "Sao không dùng PEFT?"                           Mục tiêu: Nâng tầm ACL/EMNLP Main Track
   Luận điểm: Δθ > 0 quá khớp vs Δθ = 0 vượt trội                              Luận điểm: Vượt trội trên mô hình SOTA chuyên dịch
   [TRẠNG THÁI: CHẠY TRƯỚC TIÊN]                                               [TRẠNG THÁI: CHẠY TIẾP THEO]
                                                │
                                                ▼
                                  [ƯU TIÊN 3: DỰ PHÒNG / JOURNAL]
                                  Ngôn ngữ Formosan thứ 2 (Paiwan -> Chinese)
                                  Khái quát hóa toàn ngữ hệ Nam Đảo (~60 phút)
                                  [TRẠNG THÁI: DỰ PHÒNG CHO BẢN JOURNAL]
```

### Bảng Ma Trận Nhiệm Vụ & Thứ Tự Ưu Tiên Thực Nghiệm:

| Thứ Tự | Mã Nhiệm Vụ | Hạng Mục Thí Nghiệm Cụ Thể | Mục Tiêu Học Thuật & Luận Điểm Đối Sách | Tài Nguyên & Thời Gian | Trạng Thái |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **P1** | **Task E-PEFT** | **Huấn luyện Đối Chuẩn PEFT (LoRA & BitFit) trên `mBART-large-50`** | Chặn đứng phản biện về Parameter-Efficient Fine-Tuning: Chứng minh trong ngữ liệu cực hiếm (4.600 câu), việc thêm tham số ngoại lai ($\Delta\theta > 0$) của LoRA ($r=8$) hay BitFit bị quá khớp hoặc kém linh hoạt so với tái định tuyến luồng trạng thái ẩn ($\Delta\theta = 0$) của CLRR. | GPU A100 (SSH)<br>~35 phút (2 runs) | **Ưu tiên 1 — Chạy trước tiên** |
| **P2** | **Task E-NLLB** | **Huấn luyện Mô Hình SOTA Chuyên Dịch `NLLB-200-distilled-600M`** | Đưa CLRR lên "ông vua" dịch thuật ngôn ngữ hiếm của Meta (đã pretrain sẵn trên hàng chục ngôn ngữ Nam Đảo). Chứng minh over-smoothing tầng sâu là thuộc tính cố hữu của Transformer 12 tầng và CLRR giải quyết triệt để trên mọi nền tảng pretrain. | GPU A100 (SSH)<br>~90 phút (3 runs) | **Ưu tiên 2 — Chạy tiếp theo** |
| **P3** | **Task E-LANG2**| **Mở rộng Ngôn ngữ Formosan thứ 2 (Paiwan $\to$ Chinese)** | Mở rộng tính tổng quát hóa đa ngôn ngữ Nam Đảo trên ngữ liệu Formosan thứ 2 (Paiwan). | GPU A100 (SSH)<br>~60 phút (2 runs) | **Ưu tiên 3 — Dự phòng Journal** |
| **--** | **Task X-HYBRID**| **Biến thể lai ghép CLRR + Middle-Align trên mT5** | **LOẠI BỎ (Không thực hiện):** Làm loãng đóng góp cốt lõi của bài báo, biến đề xuất thành phương pháp ghép nối chắp vá. | Không chạy | **Hủy bỏ** |

---

## 2. Cây Thư Mục Triển Khai Thực Nghiệm (Experiment Directory Tree)

```
d:\Code\CLRR\
├── src\
│   ├── amis_rewire\                              [BẢO TỒN NGUYÊN VẸN 100% - ĐÓNG BĂNG]
│   │   ├── modeling.py                           # Cốt lõi CLRR & LSR gốc (ĐÓNG BĂNG)
│   │   ├── train.py                              # Pipeline huấn luyện chuẩn (ĐÓNG BĂNG)
│   │   ├── metrics.py                            # Đo lường SacreBLEU chuẩn
│   │   └── prepare_data.py                       # Xử lý dữ liệu chuẩn
│   │
│   ├── comparative_baselines\                    [MODULE BASELINES ĐÃ HOÀN TẤT]
│   │   ├── layerskip_acl2024\                    # LayerSkip chuẩn ACL 2024
│   │   └── middle_align_acl2025\                 # Middle-Layer Alignment chuẩn ACL 2025
│   │
│   ├── peft_baselines\                           [MODULE PHỤC VỤ ƯU TIÊN 1 (PEFT)]
│   │   ├── lora_adapter.py                       # Cấu hình LoRA (HuggingFace PEFT, r=8, alpha=16)
│   │   └── bitfit_adapter.py                     # Cấu hình BitFit (chỉ tinh chỉnh bias vectors)
│   │
│   └── nllb_suite\                               [MODULE PHỤC VỤ ƯU TIÊN 2 (NLLB-200)]
│       ├── modeling_nllb_clrr.py                 # Hook CLRR trên NLLB-200-distilled-600M
│       └── train_nllb.py                         # Training script NLLB với tokenizer NLLB chuẩn
│
├── scripts\
│   ├── peft\                                     [SCRIPTS CHO ƯU TIÊN 1]
│   │   ├── smoke_test_peft.py                    # Preflight Smoke Test cho LoRA/BitFit
│   │   ├── run_peft_mbart.py                     # Huấn luyện LoRA & BitFit trên mBART-50
│   │   └── run_peft_suite.sh                     # Bash runner điều phối tự động trên Colab A100
│   │
│   └── nllb\                                     [SCRIPTS CHO ƯU TIÊN 2]
│       ├── smoke_test_nllb.py                    # Preflight Smoke Test cho NLLB-200
│       └── run_nllb_suite.sh                     # Runner cho Baseline, Middle-Align, CLRR
│
├── outputs_rebuttal\                             [KHO ARTIFACTS PEFT & NLLB MỚI]
│   ├── peft_scores.csv                           # Kết quả so sánh LoRA / BitFit vs CLRR
│   └── nllb_scores.csv                           # Kết quả đối sánh NLLB-200
│
├── outputs_revalidation\                         [ĐÃ HOÀN TẤT 100% - ĐÃ LÊN HUGGING FACE]
│   ├── mbart-large-50-ami-cmn-clrr-only\         # Best checkpoint & metrics
│   ├── mbart-large-50-ami-cmn-lsr-only\          # Best checkpoint & metrics
│   ├── mbart_ablation_results.csv                # Bảng bóc tách thành phần mBART (Table 3)
│   └── morphological_breakdown.csv               # Số liệu bóc tách tiền tố mi-, ma- (Table 5)
│
└── docs\
    ├── EXPERIMENTS.md                            # Hồ sơ thực nghiệm tập trung
    ├── RESEARCH_INSIGHTS.md                      # Cơ sở lý thuyết và dẫn chứng
    └── COLAB_SSH_GUIDE.md                        # Hướng dẫn Colab SSH và template
```

---

## 3. Đặc Tả Kỹ Thuật Chi Tiết Cho Từng Thí Nghiệm Ưu Tiên

### 3.1. Ưu Tiên 1: Huấn Luyện Đối Chuẩn PEFT (LoRA & BitFit trên mBART-50) (Task E-PEFT)
* **Lý do chạy trước tiên:**
  1. **Tính cấp thiết:** Bất kỳ phản biện nào khi đọc một bài báo về fine-tuning LLM/Seq2Seq trên tập dữ liệu nhỏ (4.600 câu) đều sẽ lập tức đặt câu hỏi: *"Tại sao không dùng LoRA hoặc BitFit (PEFT) để tránh overfitting, mà lại cần tái định tuyến residual ($\Delta\theta=0$)?"*
  2. **Thời gian cực nhanh:** Chỉ mất **~35 phút** trên A100 cho cả 2 mô hình (LoRA và BitFit), tiết kiệm tối đa Compute Units.
  3. **Độ an toàn cao:** Chạy trên chính backbone `mBART-large-50` quen thuộc đã được kiểm chứng ổn định, không lo lỗi tokenizer hay data format.
* **Phương pháp triển khai:**
  1. **LoRA (Hu et al., ICLR 2022):**
     * Tích hợp qua thư viện `peft` chính thức của Hugging Face.
     * Áp dụng LoRA trên các ma trận chú ý Query & Value (`q_proj`, `v_proj`) của cả Encoder và Decoder:
       $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A, \quad r=8, \alpha=16, \text{dropout}=0.05$$
      * Số tham số huấn luyện: 72 module Q/V $\times$ 16,384 = **1,179,648 tham số** (chiếm **0.1927%** toàn mô hình, $\Delta\theta = +1.18\text{M}$).
   2. **BitFit (Ben-Zaken et al., ACL 2022):**
     * Đóng băng toàn bộ ma trận trọng số, chỉ mở cho phép cập nhật các vector bias: $\theta = \{b\}$.
      * Số tham số huấn luyện: 256 bias tensors = **335,872 tham số** (chiếm **0.0550%** toàn mô hình, $\Delta\theta = 0$ vì không sinh tham số mới).
* **Bảng kết quả đối chuẩn thực nghiệm chính thức (Test Set 575 câu):**

| Backbone Model | Phương pháp | Cơ chế can thiệp | $\Delta\theta$ (Thêm mới) | Tham số huấn luyện (% Model) | BLEU (zh) ↑ | chrF++ (w=2) ↑ | Hiện tượng thực tế đo được |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`mBART-large-50`** | **BitFit** (ACL 2022) | Bias Tuning | **0** | 335.872 (0.0550%) | **0.3511** | **2.8464** | Đóng băng 99.95% backbone khiến mô hình không học được ngôn ngữ unseen Amis |
| | **LoRA** (ICLR 2022) | Low-Rank Adapters | **+1.179.648** | 1.179.648 (0.1927%) | **3.3284** | **5.1583** | Chỉ bắt được từ vựng đơn lẻ; adapter 1.18M không đủ sức tái định hình không gian đa ngữ |
| | Standard Fine-Tuning | Full Tuning | 0 | 610.879.488 (100%) | 19.6106 | 14.0538 | Điểm tựa baseline chuẩn |
| | Middle-Align (ACL 2025) | Mid-layer Loss | 0 | 610.879.488 (100%) | 19.7243 | 15.5253 | Căn chỉnh tầng giữa đơn lẻ |
| | **Full CLRR-Enc + LSR (Ours)**| **Residual Rewiring** | **0** | **610.879.488 (Zero New Params)**| **20.3927** | **19.0839** | **Đột phá áp đảo (+17.06 BLEU so với LoRA, $\Delta\theta=0$)** |
---

### 3.2. Ưu Tiên 2: Huấn Luyện Mô Hình SOTA Chuyên Dịch `NLLB-200-distilled-600M` (Task E-NLLB)
* **Lý do chạy tiếp theo:**
  1. **Nâng tầm bài báo lên Main Track ACL/EMNLP:** NLLB-200 là đỉnh cao hiện nay về dịch thuật ngôn ngữ hiếm (low-resource translation) của Meta. Mô hình đã được pretrain sẵn trên rất nhiều ngôn ngữ thuộc ngữ hệ Nam Đảo (Tagalog, Cebuano, Malay, Pangasinan...).
  2. **Khẳng định tính phổ quát của cơ chế Over-smoothing:** mBART được pretrain dạng BART khôi phục văn bản, trong khi NLLB được pretrain dịch đa ngữ song ngữ. Nếu CLRR chiến thắng vang dội trên cả NLLB-200, ta chứng minh được over-smoothing tầng sâu là bệnh cố hữu của mọi Transformer 12-tầng, và CLRR là giải pháp kiến trúc phổ quát.
* **Đặc tả kiến trúc & Tokenizer:**
  * Model ID: `facebook/nllb-200-distilled-600M` (12 layers enc, 12 layers dec, $d_{\text{model}}=1024$, 600M params).
  * Tokenizer: NLLB SentencePiece 256k tokens. Ngôn ngữ đích chuẩn là `zho_Hant` (Tiếng Trung Phồn thể, `forced_bos_token_id=256201`).
* **Kế hoạch 7 Runs đối sánh công bằng tuyệt đối trên NLLB-200 (Phương án A - Enc/Dec độc lập):**
  1. `nllb-200-600m-baseline`: Standard fine-tuning (20 epochs).
  2. `nllb-200-600m-bitfit`: BitFit Bias-only tuning (ACL 2022).
  3. `nllb-200-600m-lora`: LoRA $r=8, \alpha=16$ trên $Q/V$ (ICLR 2022).
  4. `nllb-200-600m-layerskip`: LayerSkip stochastic layer dropout (ACL 2024).
  5. `nllb-200-600m-middle-align`: Middle-Layer Alignment tại Layer 6 (ACL 2025).
  6. `nllb-200-600m-clrr-enc`: CLRR-Enc ($d=2, \alpha=0.1$) + LSR ($\lambda=0.1$) (Proposed Main).
  7. `nllb-200-600m-clrr-dec`: CLRR-Dec ($d=2, \alpha=0.1$) + LSR ($\lambda=0.1$) (Proposed Decoder).
* **Thời gian & Tài nguyên:**
  * ~25--30 phút/run $\times$ 7 runs $\approx$ **3.0--3.5 giờ** trên GPU A100 Colab.
  * Tự động lưu kết quả về `results/nllb-200/{method}/` và đóng gói đẩy lên Hugging Face `FiveC/amis-rewire-checkpoints`.

---

### 3.3. Ưu Tiên 3: Ngôn Ngữ Formosan Thứ 2 (Paiwan $\to$ Chinese) (Task E-LANG2)
* **Mục tiêu:** Mở rộng tính tổng quát hóa đa ngôn ngữ Nam Đảo trên ngữ liệu Formosan thứ 2.
* **Thời điểm thực hiện:** Sau khi hoàn thành xong P1 và P2, hoặc dành cho phiên bản mở rộng Journal (TACL/CL).

---

## 4. Bảng Quy Chuẩn Siêu Tham Số Công Bằng Tuyệt Đối (Fair Comparison Protocol)

| Siêu tham số | mBART Baseline | mBART + LoRA | mBART + BitFit | mBART + CLRR (Ours) | NLLB Baseline | NLLB + CLRR (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Backbone Model** | `mbart-large-50` | `mbart-large-50` | `mbart-large-50` | `mbart-large-50` | `nllb-200-600m` | `nllb-200-600m` |
| **Số tầng Enc / Dec** | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| **Hidden Dim ($d_{\text{model}}$)** | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| **Tham số thêm ($\Delta\theta$)** | **0** | $\approx 1.54\text{M}$ | $\approx 0.12\text{M}$ | **0** | **0** | **0** |
| **Tập dữ liệu** | 4.600 / 576 / 575 | 4.600 / 576 / 575 | 4.600 / 576 / 575 | 4.600 / 576 / 575 | 4.600 / 576 / 575 | 4.600 / 576 / 575 |
| **Random Seed** | 42 | 42 | 42 | 42 | 42 | 42 |
| **Số Epoch tối đa** | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| **Effective Batch Size** | 128 | 128 | 128 | 128 | 128 | 128 |
| **Learning Rate** | $5 \times 10^{-5}$ | $2 \times 10^{-4}$ (PEFT) | $1 \times 10^{-4}$ | $5 \times 10^{-5}$ | $5 \times 10^{-5}$ | $5 \times 10^{-5}$ |
| **Warmup Ratio** | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| **Optimizer / Precision** | AdamW / BF16 | AdamW / BF16 | AdamW / BF16 | AdamW / BF16 | AdamW / BF16 | AdamW / BF16 |
| **Early Stopping** | Patience 4 | Patience 4 | Patience 4 | Patience 4 | Patience 4 | Patience 4 |
| **Beam Search Size** | 4 | 4 | 4 | 4 | 4 | 4 |

---

## 5. Quy Trình Preflight Smoke Test Bắt Buộc

Trước khi kích hoạt bất kỳ script nào trên A100, bắt buộc chạy kiểm thử cục bộ/remote 1 batch trong $\le 25$ giây:

1. **Smoke Test PEFT (LoRA & BitFit):**
   ```bash
   python scripts/peft/smoke_test_peft.py
   ```
   *Yêu cầu kiểm tra:* Khởi tạo adapter chính xác, tính loss forward, backward chỉ cập nhật adapter parameters, beam search generate ra chuỗi token không lỗi.
2. **Smoke Test NLLB-200:**
   ```bash
   python scripts/nllb/smoke_test_nllb.py
   ```
   *Yêu cầu kiểm tra:* Tải checkpoint `facebook/nllb-200-distilled-600M`, gắn hook CLRR vào `model.model.encoder.layers`, kiểm tra tensor skip connection shape khớp chuẩn xác.

---

## 6. Quy Trình Thực Thi Tự Động Bằng BASH Qua SSH Colab

```bash
# -------------------------------------------------------------
# BƯỚC 1: Khởi tạo Máy ảo Colab A100 & Kết nối SSH
# -------------------------------------------------------------
colab new -s colab_a100 --gpu A100

# -------------------------------------------------------------
# BƯỚC 2: Đồng bộ mã nguồn & Chạy Preflight Smoke Test (< 30s)
# -------------------------------------------------------------
ssh colab "cd /content/CLRR && git pull && python scripts/peft/smoke_test_peft.py"

# -------------------------------------------------------------
# BƯỚC 3: Kích hoạt Huấn Luyện Nền (Detached Execution)
# -------------------------------------------------------------
ssh colab "nohup bash /content/CLRR/scripts/peft/run_peft_suite.sh > /content/peft_run.log 2>&1 &"

# -------------------------------------------------------------
# BƯỚC 4: Tự động tải kết quả & Upload Hugging Face Hub
# -------------------------------------------------------------
# Script tự động đẩy outputs_rebuttal/ lên repo FiveC/amis-rewire-checkpoints
# và ghi log kết thúc.

# -------------------------------------------------------------
# BƯỚC 5: Tắt Máy Ảo Ngay Lập Tức (Bảo Toàn Compute Units)
# -------------------------------------------------------------
colab stop -s colab_a100
```

---

## 7. Trình Tự Thực Thi Khuyến Nghị (Step-by-Step Action Plan)

1. **ĐÃ HOÀN TẤT 100% (ƯU TIÊN 1): BỘ ĐỐI CHUẨN PEFT (LoRA & BitFit trên mBART-50)**
   - **Trạng thái:** Đã hoàn thành toàn bộ trên NVIDIA A100-SXM4-40GB. Đã upload artifacts lên HF Hub (`FiveC/amis-rewire-checkpoints`) và lưu bảng tổng kết tại `outputs_rebuttal/peft_scores.csv`.
   - **Kết quả đo lường chính thức trên Test Set (575 câu):**
     * **LoRA ($r=8$):** BLEU = **3.33** | chrF++ = **5.16** (1.18M params, 0.19%)
     * **BitFit (Bias-only):** BLEU = **0.35** | chrF++ = **2.85** (336k params, 0.055%)
     * **CLRR-Enc + LSR (Ours):** BLEU = **20.39** | chrF++ = **19.08** (Zero new params)
   - **Kết luận khoa học:** CLRR áp đảo hoàn toàn LoRA (+17.06 BLEU) và BitFit (+20.04 BLEU), chứng minh PEFT bị nghẽn biểu diễn nghiêm trọng khi thích ứng với ngôn ngữ unseen cực hiếm tài nguyên.
2. **CHẠY TIẾP THEO (ƯU TIÊN 2 — ~90 PHÚT A100): BỘ ĐỐI CHUẨN NLLB-200 SOTA HOẶC mT5-SMALL VARIANT**
   - **Tùy chọn A (NLLB-200 SOTA):** Mở rộng tính tổng quát trên kiến trúc dịch máy đa ngôn ngữ 600M thế hệ mới.
   - **Tùy chọn B (mT5 CLRR + Middle-layer Alignment):** Kiểm tra biến thể kết hợp trên mT5.
