# Kế Hoạch Thực Nghiệm Nâng Cấp ACL: Chạy Phủ Toàn Diện Đa Kiến Trúc (Cross-Architecture Benchmarking) & Phân Tích Chuyên Sâu

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN & CHIẾN LƯỢC NÂNG CẤP ĐẠT CHUẨN ACL (ACL MAIN / FINDINGS):**
> 1. **Tính Cân Xứng Tuyệt Đối Giữa Các Backbone (Symmetrical Cross-Architecture Matrix):**
>    * Trong bài báo hiện tại, mô hình đề xuất của chúng ta (**CLRR**) đã được kiểm chứng trên cả **3 họ kiến trúc đại diện**:
>      * `google/mt5-small` (8 enc / 8 dec — Subword đa ngữ chuẩn).
>      * `facebook/mbart-large-50-many-to-many-mmt` (12 enc / 12 dec — Chuyên biệt dịch máy đa ngữ).
>      * `google/byt5-small` (12 enc / 4 dec — Byte-level không từ điển, kiến trúc bất đối xứng).
>    * **Khoảng trống cần bổ sung ngay cho ACL:** Hai phương pháp đối chuẩn gần đây (**LayerSkip - ACL 2024** và **Middle-Layer Alignment - ACL 2025**) mới chỉ được chạy trên `mt5-small`. Để bài báo đạt độ chặt chẽ tuyệt đối trước hội đồng phản biện ACL, **bắt buộc phải chạy phủ đủ LayerSkip và Middle-Align trên cả `mBART-50` và `ByT5-small`**.
> 2. **Bảo tồn Tuyệt đối 100% Mã Nguồn & Toàn bộ Kết quả Đã Chốt:**
>    * Tuyệt đối **KHÔNG chỉnh sửa bất kỳ file nào trong `src/amis_rewire/`** và không can thiệp vào các checkpoint/kết quả đã chốt (`outputs_extra/` và `outputs_comparative/`). Toàn bộ 11 điểm số chính thức hiện có được đóng băng vĩnh viễn.
> 3. **Tổ chức Thư Mục Độc Lập Cho Từng Backbone:**
>    * Mở rộng `src/comparative_baselines/` để hỗ trợ đa backbone (`--model-name-or-path`) một cách đồng nhất và tự động nhận diện kiến trúc (`t5`, `mbart`, `byt5`).
> 4. **Cấu hình Công bằng Tuyệt đối (Strict Fairness Protocol):**
>    * Cố định đồng nhất: Seed 42, Split 4.600 / 576 / 575, Effective Batch Size 128, LR chuẩn (`3e-4` cho mT5/ByT5, `5e-5` cho mBART-50), Warmup 0.06, 20 Epochs, Early Stopping Patience 4 theo chrF++, BF16, Test Beam Size 4, SacreBLEU `zh` + chrF++ `word_order=2`.
> 5. **Tận dụng Tối đa Tốc độ GPU A100 & Tự Động Hóa SSH BASH:**
>    * Thực thi thuần túy bằng Bash script qua SSH (`ssh colab`), lưu top 3 checkpoint trên ổ SSD NVMe `/content/`, nén và đẩy Best Model lên Hugging Face Hub (`FiveC/amis-rewire-checkpoints`), và tự động tắt máy Colab (`colab stop -s colab`) để bảo toàn số dư 298+ Compute Units.

---

## 1. Cấu Trúc Hai Giai Đoạn Nâng Cấp Toàn Diện Cho ACL

```
                        ┌─────────────────────────────────────────────────────────┐
                        │      CHIẾN LƯỢC NÂNG CẤP THỰC NGHIỆM ĐẠT CHUẨN ACL      │
                        └────────────────────────────┬────────────────────────────┘
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             ▼                                               ▼
                     [GIAI ĐOẠN 1: BẮT BUỘC]                        [GIAI ĐOẠN 2: CHUYÊN SÂU]
                   Chạy Phủ 2 Baseline ACL Trên                    Phân Tích Bóc Tách Lý Thuyết,
                   2 Backbone Còn Lại (mBART & ByT5)                Độ Trễ & Đối Chuẩn PEFT (LoRA)
                             │                                               │
             ┌───────────────┴───────────────┐               ┌───────────────┼───────────────┐
             ▼                               ▼               ▼               ▼               ▼
      [mBART-Large-50]                 [ByT5-Small]      [LoRA vs CLRR]  [Stop-Grad]   [Hardware Profiler]
       • LayerSkip                     • LayerSkip        • PEFT đối     • Kiểm chứng   • Zero-Overhead
       • Middle-Align (Layer 6)        • Middle-Align       chuẩn         toán học       (VRAM, Latency)
```

---

## 2. Cây Thư Mục Dự Án Dự Kiến (Project Directory Tree)

```
d:\Code\CLRR\
├── src\
│   ├── amis_rewire\                              [BẢO TỒN NGUYÊN VẸN 100% - ĐÓNG BĂNG]
│   │
│   └── comparative_baselines\                    [MỞ RỘNG HỖ TRỢ ĐA BACKBONE]
│       ├── __init__.py
│       ├── layerskip_acl2024\                    [HỖ TRỢ: mt5-small, mbart-large-50, byt5-small]
│       │   ├── __init__.py
│       │   ├── model.py                          # Hook phổ quát cho cả T5Block & MBartEncoderLayer
│       │   └── train.py                          # Hỗ trợ switch model backbone & configure_mbart
│       │
│       └── middle_align_acl2025\                 [HỖ TRỢ: mt5-small, mbart-large-50, byt5-small]
│           ├── __init__.py
│           ├── model.py                          # Tự động chọn middle layer (layer 4 cho mt5, layer 6 cho mbart/byt5)
│           └── train.py                          # Hỗ trợ switch model backbone & configure_mbart
│
├── scripts\
│   ├── smoke_test_comparative.py                 [Cập nhật kiểm thử 3 backbones]
│   ├── run_comparative_models.py                 [Thêm tham số --backbones mt5,mbart,byt5]
│   └── run_comparative_models.sh                 [Runner Bash qua SSH Colab A100]
│
├── outputs_comparative\                          [KHO KẾT QUẢ ĐỐI SÁNH ĐA BACKBONE]
│   ├── mt5_small\                                # ĐÃ HOÀN TẤT (LayerSkip: 3.97, Middle-Align: 4.75)
│   ├── mbart_large_50\                           # SẮP CHẠY (LayerSkip & Middle-Align trên mBART)
│   ├── byt5_small\                               # SẮP CHẠY (LayerSkip & Middle-Align trên ByT5)
│   └── full_comparative_matrix.csv               # BẢNG MA TRẬN ĐỐI SÁNH 3x4 HOÀN CHỈNH CHO ACL
│
└── docs\
    ├── PAPER_EXPERIMENT_INSIGHTS.md              # Kho dữ liệu chính thức
    └── ACL_UPGRADE_EXPERIMENT_PLAN.md             # Kế hoạch nâng cấp chi tiết này
```

---

## 3. Đặc Tả Kỹ Thuật Cho Từng Backbone Mới (Technical Specifications)

### 3.1. Backbone 2: `facebook/mbart-large-50-many-to-many-mmt` (12 Layers Encoder, 12 Layers Decoder)
* **Đặc trưng kiến trúc:**
  * Mô hình Seq2Seq chuyên biệt cho dịch máy đa ngôn ngữ (Pre-trained Multilingual NMT), kích thước ~610M tham số.
  * Cần cấu hình Tokenizer đặc thù: Tiếng Amis (chữ Latinh) dùng mã proxy `tl_XX` (Tagalog - cùng ngữ hệ Nam Đảo Formosan) hoặc `en_XX`; Tiếng Trung dùng mã `zh_CN` và đặt `forced_bos_token_id`.
  * Learning rate chuẩn: **`5e-5`** (đúng theo thiết lập chính thức của bài báo).
* **Triển khai LayerSkip trên mBART-50:**
  * Encoder gồm $L=12$ tầng (`model.encoder.layers`).
  * Tỉ lệ ngắt tầng lũy thừa:
    $$D(l) = e^{\frac{l \ln 2}{11}} - 1 \quad (l \in [0, 11])$$
    $p_0 = 0.0$ (tầng đầu luôn giữ), $p_{11} = p_{\max} = 0.2$.
* **Triển khai Middle-Layer Alignment trên mBART-50:**
  * Tổng số tầng Encoder $L=12$. Tầng giữa tối ưu được chọn là **Tầng 6** ($i=6$, trung tâm điểm của biểu diễn ngữ nghĩa trừu tượng).
  * Contrastive Loss căn chỉnh biểu diễn giữa câu Amis và câu tiếng Trung tại tầng 6 với nhiệt độ $\tau = 0.1, \lambda = 0.1$.

---

### 3.2. Backbone 3: `google/byt5-small` (12 Layers Encoder, 4 Layers Decoder — Byte-Level)
* **Đặc trưng kiến trúc:**
  * Mô hình xử lý cấp độ Byte (UTF-8, Token-free), không phụ thuộc vào từ vựng SentencePiece cố định.
  * Kiến trúc **bất đối xứng cao**: 12 tầng Encoder nhưng chỉ có 4 tầng Decoder.
  * Learning rate chuẩn: **`3e-4`**.
* **Triển khai LayerSkip trên ByT5-small:**
  * Encoder gồm $L=12$ tầng (`encoder.block`).
  * Áp dụng stochastic layer dropout trên 12 tầng Encoder ($p_{\max} = 0.2$).
* **Triển khai Middle-Layer Alignment trên ByT5-small:**
  * Tổng số tầng Encoder $L=12$. Tầng giữa được chọn là **Tầng 6** ($i=6$).
  * Đo lường khả năng căn chỉnh ngữ nghĩa trực tiếp từ chuỗi byte UTF-8 của tiếng Amis sang tiếng Trung.

---

## 4. Bảng Quy Chuẩn Siêu Tham Số Công Bằng Tuyệt Đối (Fair Comparison Protocol)

| Siêu tham số | `google/mt5-small` | `facebook/mbart-large-50` | `google/byt5-small` |
| :--- | :---: | :---: | :---: |
| **Loại Tokenizer** | SentencePiece Subword (250k) | mBART Subword (250k) | Raw UTF-8 Byte-level (256) |
| **Số tầng Encoder / Decoder** | 8 / 8 | 12 / 12 | 12 / 4 (Bất đối xứng) |
| **Middle Layer Index ($i$)** | **Layer 4** | **Layer 6** | **Layer 6** |
| **Tập dữ liệu** | 4.600 train / 576 val / 575 test | 4.600 train / 576 val / 575 test | 4.600 train / 576 val / 575 test |
| **Effective Batch Size** | 128 (16 x 8) | 128 (16 x 8) | 128 (16 x 8) |
| **Learning Rate** | `3e-4` | **`5e-5`** | `3e-4` |
| **Warmup Ratio** | 0.06 | 0.06 | 0.06 |
| **Epochs / Early Stopping** | 20 / Patience 4 (val chrF++) | 20 / Patience 4 (val chrF++) | 20 / Patience 4 (val chrF++) |
| **Precision** | BF16 | BF16 | BF16 |
| **Generation Beam Size** | 4 | 4 | 4 |
| **Metric đánh giá** | SacreBLEU `zh` + chrF++ `w=2` | SacreBLEU `zh` + chrF++ `w=2` | SacreBLEU `zh` + chrF++ `w=2` |

---

## 5. Quy Trình Preflight Smoke Test Bắt Buộc

Trước khi kích hoạt huấn luyện 20 epochs trên GPU A100:
1. Nạp đồng thời cả 3 backbone: `mt5-small`, `mbart-large-50`, `byt5-small`.
2. Kiểm tra forward pass 1 batch mẫu ($B=2$) cho cả `LayerSkip` và `MiddleAlign` trên từng backbone.
3. Kiểm tra backward pass (`loss.backward()`) đảm bảo gradient cập nhật đầy đủ.
4. Kiểm tra `generate(beam_size=4)` trên cả 3 tokenizer.
5. Thời gian chạy smoke test: **~40 giây**.

---

## 6. Cơ Chế Thực Thi Thuần Túy Bằng BASH Qua SSH Colab

```
[BƯỚC 1: Khởi tạo Máy ảo A100]
  Lệnh: colab new -s colab --gpu A100
         │
[BƯỚC 2: Đồng Bộ Code & Chạy Smoke Test 3 Backbone]
  ssh colab "cd /content/CLRR && git pull origin main && python scripts/smoke_test_comparative.py"
         │
[BƯỚC 3: Kích Hoạt Huấn Luyện 4 Mô Hình Bổ Sung Bằng BASH]
  ssh colab "export HF_TOKEN=... && bash /content/CLRR/scripts/run_comparative_models.sh --backbones mbart,byt5"
  ├── [Run 1] mBART-50 + LayerSkip (~15 phút) -> Push HF
  ├── [Run 2] mBART-50 + Middle-Align (~18 phút) -> Push HF
  ├── [Run 3] ByT5-small + LayerSkip (~12 phút) -> Push HF
  └── [Run 4] ByT5-small + Middle-Align (~14 phút) -> Push HF
         │
[BƯỚC 4: Tự Động Xuất Bảng Ma Trận Tổng Hợp 3x4 & Tắt Máy Ngay Lập Tức]
  Lệnh: colab stop -s colab (Bảo toàn tuyệt đối số dư 290+ Compute Units!)
```

---

## 7. Dự Toán Thời Gian & Tài Nguyên Trên GPU A100

* **Tổng thời gian chạy 4 mô hình:** Khoảng **55 – 60 phút** trên GPU NVIDIA A100-SXM4-40GB.
* **Mức tiêu hao Compute Units:** ~4.0 – 4.2 compute units.
* **Số dư khả dụng hiện tại của bạn:** **~298.5 compute units** $\to$ Sau khi chạy xong vẫn còn **hơn 294 compute units** (chỉ tiêu tốn ~1.4% số dư)!
* **Dung lượng lưu trữ:** Tối đa ~12 GB trên 150 GB SSD NVMe tại `/content`.

---

## 8. Bảng Kết Quả Kỳ Vọng Cho Bài Báo ACL (MA TRẬN ĐỐI SÁNH ĐA KIẾN TRÚC 3x4 HOÀN HẢO)

> [!TIP]
> Đây sẽ là **Bảng Trung Tâm (Table 1 / Table 5 chính thức)** trong bài báo nộp ACL. Nó bao quát toàn diện mọi khía cạnh: từ Subword nhỏ, Mô hình dịch lớn, đến Byte-level không từ điển:

| Backbone Kiến Trúc | Phương Pháp | Bài Báo Tham Chiếu | Extra Params | BLEU (zh) | chrF++ (w=2) | Phân Loại Học Thuật |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`mT5-small`** *(8-enc / 8-dec)* | Standard Fine-Tuning | Standard Baseline | 0 | 2.79 | 3.81 | Baseline cơ sở (Đã chốt) |
| | **LayerSkip** | Elhoushi et al. (**ACL 2024**) | 0 | **3.97** | **5.29** | Measured Baseline (Đã chốt) |
| | **Middle-Layer Align** | Liu & Niehues (**ACL 2025**) | 0 | **4.75** | **5.94** | Measured Baseline (Đã chốt) |
| | **CLRR-Enc** (Ours) | Proposed (Ablation) | **0** | **4.44** | **5.18** | **Đề xuất của bạn (Đã chốt)** |
| | **JEPA + CLRR-Enc** (Ours)| **Proposed (Main)** | **0** | **4.60** | **5.04** | **Đề xuất chính (Đã chốt)** |
| | **JEPA + CLRR-Dec** (Ours)| Proposed (Decoder-only) | **0** | **4.83** | **5.19** | **Đề xuất của bạn (Đã chốt)** |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`mBART-large-50`** *(12-enc / 12-dec)*| Standard Fine-Tuning | Translation Baseline | 0 | 19.61 | 14.05 | Baseline cơ sở (Đã chốt) |
| | **LayerSkip** | Elhoushi et al. (**ACL 2024**) | 0 | *[Sắp chạy]* | *[Sắp chạy]* | Đối chuẩn ACL trên NMT lớn |
| | **Middle-Layer Align** | Liu & Niehues (**ACL 2025**) | 0 | *[Sắp chạy]* | *[Sắp chạy]* | Đối chuẩn ACL trên NMT lớn |
| | **JEPA + CLRR-Enc** (Ours)| **Proposed (Main)** | **0** | **20.39** | **19.08** | **Đề xuất chính (+5.03 chrF++)** |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`ByT5-small`** *(12-enc / 4-dec)* | Standard Fine-Tuning | Byte-Level Baseline | 0 | 7.58 | 8.31 | Baseline cơ sở (Đã chốt) |
| | **LayerSkip** | Elhoushi et al. (**ACL 2024**) | 0 | *[Sắp chạy]* | *[Sắp chạy]* | Đối chuẩn ACL trên Byte-level |
| | **Middle-Layer Align** | Liu & Niehues (**ACL 2025**) | 0 | *[Sắp chạy]* | *[Sắp chạy]* | Đối chuẩn ACL trên Byte-level |
| | **JEPA + CLRR-Enc** (Ours)| **Proposed (Main)** | **0** | **7.31** | **8.10** | **Đề xuất chính (Đã chốt)** |
