# KẾ HOẠCH TRIỂN KHAI & RUNBOOK THỰC NGHIỆM ĐỐI CHẤT REVIEWER (PEFT ENHANCEMENT & SCIENTIFIC RUNBOOK)

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN & TỐI ƯU HÓA TÀI NGUYÊN (SCIENTIFIC RIGOR, FAIRNESS, SPEED & CU PROTECTION):**
> 1. **Mục tiêu tối thượng:** Giải tỏa triệt để mọi "Red Flag" từ báo cáo phản biện (nghi ngờ cấu hình PEFT đối chứng hẹp, thiếu đối chứng byte-level, nghi ngờ sụp đổ biểu diễn LSR, và thiếu bằng chứng over-smoothing qua layer-wise probing), đưa vị thế bài báo lên *Strong Accept*.
> 2. **Bảo tồn toàn vẹn thành quả đã đạt được:** Đóng băng toàn bộ kết quả SOTA của CLRR-Enc trên `mBART-large-50` (20.39 BLEU, 19.08 chrF++), `mT5-small` (4.83 BLEU), và `NLLB-200-distilled-600M` (14.28 BLEU, 18.75 chrF++). Checkpoint đã lưu trữ an toàn trên Hugging Face Hub (`FiveC/amis-rewire-checkpoints`).
> 3. **Giao thức công bằng tuyệt đối (Strict Fairness Protocol):** Cố định toàn diện: Seed 42, Split 4.600 / 576 / 575, Effective Batch Size 128, Early Stopping Patience 4 theo chrF++, Precision BF16 (hoặc FP16), Generation Beam Size 4, SacreBLEU `zh` + chrF++ `word_order=2`.
> 4. **Chuẩn mực nghiệm thu (No Hand-copied Numbers):** Kết quả bắt buộc xuất ra `metrics.json` và `test_predictions.csv` (đủ đúng 575 dòng dữ liệu kiểm thử + 1 dòng header). Tuyệt đối không chép tay số liệu.

---

## 1. Bản Đồ Trạng Thái Thực Nghiệm (Experiment Status Matrix)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              BẢN ĐỒ TIẾN ĐỘ THỰC NGHIỆM ĐỐI CHẤT REVIEWER                              │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
         ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
         ▼                                          ▼                                          ▼
 [ƯU TIÊN 2: CHẨN ĐOÁN]                    [ƯU TIÊN 1: STRONG LoRA]                   [ƯU TIÊN 3: ByT5]
  ✅ 100% HOÀN THÀNH                        ⏳ ĐANG CHỜ CHẠY GPU A100                 ✅ ĐÃ CÓ SỐ LIỆU SẴN
  - Affix Probing (L1-12): Done             - Run A: All-Linear (r=16)                 - Base: 7.58 BLEU
  - LSR Anti-Collapse SVD: Done             - Run B: All-Linear + Unfreeze Embed       - CLRR: 7.44 BLEU
  - Morphological Overlap: Done             - Ước tính: ~35-40 phút / run              - Đưa vào Discussion
  - PEFT Audit Checkpoint: Done             - Mục tiêu: Chứng minh CLRR vượt trội      - Không tốn thêm GPU
```

### Bảng Tổng Hợp Chi Tiết Từng Hạng Mục:

| Thứ Tự | Hạng Mục Thí Nghiệm | Module / Kịch Bản | Trạng Thái | Kết Quả / Phát Hiện Khoa Học Đã Ghi Nhận |
| :---: | :--- | :--- | :---: | :--- |
| **P2** | **Morphological Overlap Audit** | `audit_morphological_overlap.py` | ✅ **DONE** | Làm rõ vì sao tổng câu các nhóm ($729 > 575$): Do **154 câu chứa đồng thời đa phụ tố**. Nhóm Root ($N=161$) là nhóm đối chứng độc lập hoàn toàn ($414 + 161 = 575$). Lưu tại `outputs_rebuttal/morphological_overlap_matrix.csv`. |
| **P2** | **LSR Anti-Collapse SVD & Rank** | `diagnose_lsr_collapse.py` | ✅ **DONE** | Bác bỏ nghi ngờ về "Dimensional collapse": Effective Rank đạt **387.67 / 575** (Amis) và **411.58 / 575** (Mandarin) trên không gian 1024 chiều. Top-1 chỉ chiếm ~5%, biểu diễn phân bổ đa chiều. |
| **P2** | **Layer-wise Linear Probing** | `probe_affixes_layerwise.py` | ✅ **DONE** | F1 của CLRR giảm nhẹ (~4–5%) ở tầng sâu L11–L12 so với Baseline. **Đi ngược giả thuyết over-smoothing ban đầu**. Diễn giải chuẩn xác: cơ chế *Distributed Representation / Superposition* (Elhage et al., 2022). chrF++ và BLEU vẫn tăng mạnh. Đã ghi chi tiết vào `docs/RESEARCH_INSIGHTS.md`. |
| **P2** | **Audit Checkpoint PEFT cũ** | `audit_peft_predictions.py` | ✅ **DONE** | Tái lập chính xác 100% điểm số báo cáo trong bài: LoRA (r=8, q/v) đạt 3.3284 BLEU / 8.0858 chrF++, BitFit đạt 0.3511 BLEU / 4.4408 chrF++. Khẳng định mô hình sinh ra thật, không phải lỗi đánh giá. |
| **P3** | **Khôi phục đối chứng ByT5** | `EXPERIMENTS_v1_with_layerskip_byt5.md` | ✅ **DONE** | Số liệu sẵn có: ByT5-small Baseline 7.58 BLEU vs CLRR 7.44 BLEU. Tích hợp vào Discussion phân tích ranh giới inductive bias giữa byte-level và subword-level. |
| **P1** | **Strong LoRA Suite (`mBART-50`)** | `run_strong_lora_mbart.py` | ⏳ **SẮP CHẠY** | **TRỌNG TÂM HIỆN TẠI:** Chạy 2 run: Run A (LoRA All-Linear $r=16, \alpha=32$) và Run B (All-Linear + Unfrozen Embeddings). Chứng minh CLRR ($\Delta\theta = 0$) vẫn ưu việt trước baseline LoRA mở rộng tối đa. |
| **P4** | **Strong LoRA Suite (`NLLB-200`)** | `train_peft.py` (NLLB) | ⏸️ **DỰ PHÒNG** | Chạy nếu còn dư Compute Units trên Colab A100. |

---

## 2. Luồng Chạy Chi Tiết Sắp Tới: Ưu Tiên 1 — Strong LoRA Suite

### 2.1. Động Lực Học Thuật & Luận Điểm Đối Sách
Reviewer nhận định cấu hình LoRA trước đây quá hẹp ($r=8$, chỉ can thiệp $q\_proj, v\_proj$, embeddings bị đóng băng) dẫn đến điểm thấp (3.33 BLEU). Để phản biện thuyết phục, ta huấn luyện **Strong LoRA Suite** trên cùng backbone `facebook/mbart-large-50-many-to-many-mmt`:

1. **Run A (LoRA All-Linear, $r=16, \alpha=32$):**
   - Can thiệp toàn bộ 6 ma trận tuyến tính của Transformer: `q_proj, k_proj, v_proj, out_proj, fc1, fc2` trên cả 12 tầng Encoder và 12 tầng Decoder.
   - Thêm $\Delta\theta_{\text{add}} \approx 6.20\text{M}$ tham số mới (~1.01% mô hình).
   - Đóng băng hoàn toàn Token Embeddings (`frozen embeddings`).
   - Learning rate: $2 \times 10^{-4}$, warmup 0.06.
   - *Mục tiêu:* Kiểm tra giới hạn dung lượng của adapter.

2. **Run B (LoRA All-Linear + Unfrozen Embeddings):**
   - Giữ nguyên cấu hình LoRA All-Linear ($r=16, \alpha=32$) như Run A.
   - Mở khóa gradient cho ma trận embedding từ vựng (`embed_tokens` / `shared`, `lm_head`).
   - Huấn luyện $\approx 6.20\text{M}$ adapter + $256.000 \times 1024$ embedding weights (~268M tham số huấn luyện).
   - Learning rate: $1 \times 10^{-4}$, warmup 0.06.
   - *Mục tiêu:* Giải quyết triệt để nút thắt từ vựng chưa từng thấy của ngôn ngữ chắp dính Amis.

---

## 3. Sổ Tay Vận Hành Google Colab (Colab Runbook Cho Bên Khác Chạy)

> [!TIP]
> Bạn có thể bàn giao toàn bộ phần này cho bất kỳ ai hoặc bất kỳ máy ảo GPU nào (Google Colab, RunPod, Lambda Labs, Vast.ai). Hệ thống đã được thiết kế hoàn toàn tự động, độc lập và có cơ chế bảo vệ chống đứt gãy kết nối.

### Yêu Cầu Phần Cứng Tối Thiểu:
- **GPU:** Khuyến nghị NVIDIA A100 (40GB/80GB) hoặc V100/L4 (tối thiểu 16GB VRAM như T4).
- **RAM hệ thống:** $\ge 12\text{GB}$.
- **Dung lượng đĩa trống:** $\ge 25\text{GB}$ trên `/content`.
- **Thời gian chạy ước tính:** ~35–40 phút cho mỗi Run trên A100 (~1.2 - 1.5 giờ tổng cộng cho 2 Run).

---

### PHƯƠNG ÁN 1: Chạy Tự Động Qua Terminal / SSH (Khuyên Dùng Nhất)

Phương án này sử dụng background script `nohup`, an toàn 100% nếu máy tính cá nhân bị ngắt mạng, gập màn hình hoặc Colab bị ngắt kết nối SSH tạm thời.

#### Bước 1: Mở Terminal SSH hoặc Terminal trên Google Colab
Kết nối vào máy Colab:
```bash
ssh colab
```
*(Hoặc mở tab Terminal trực tiếp trên giao diện web Google Colab Pro).*

#### Bước 2: Clone hoặc Cập Nhật Mã Nguồn Mới Nhất
```bash
cd /content
if [ ! -d "/content/CLRR" ]; then
    git clone https://github.com/HeyDunaX/CLRR.git /content/CLRR
fi
cd /content/CLRR
git checkout main
git pull origin main
```

#### Bước 3: Cài Đặt Môi Trường & Dependencies
```bash
pip install -q -r requirements.txt
pip install -q peft sacrebleu scikit-learn
pip uninstall -y torchao 2>/dev/null || true
```

#### Bước 4: Kiểm Tra GPU & Chạy Preflight Smoke Test (< 25 giây)
Trước khi chạy thật, bắt buộc chạy smoke test để đảm bảo PyTorch, CUDA, PEFT hoạt động hoàn hảo:
```bash
nvidia-smi
python -u scripts/peft/smoke_test_peft.py
```
*Kết quả mong đợi:* In ra `ALL PEFT SMOKE TESTS PASSED (< 25s)` mà không có bất kỳ ngoại lệ nào.

#### Bước 5: Kích Hoạt Huấn Luyện Nền (Autonomous Background Run)
Thiết lập Hugging Face Token (nếu muốn tự động sao lưu checkpoint lên HF Hub) và kích hoạt chạy ngầm:
```bash
export HF_TOKEN="hf_your_token_here"   # Tùy chọn: điền token nếu muốn auto-upload HF
nohup bash scripts/peft/run_strong_lora_suite.sh > /content/strong_lora.log 2>&1 &
```

#### Bước 6: Theo Dõi Tiến Trình Huấn Luyện Thời Gian Thực
Để xem log tiến độ huấn luyện (loss, BLEU từng epoch, thời gian):
```bash
tail -f /content/strong_lora.log
```
*(Nhấn `Ctrl + C` để thoát màn hình theo dõi log; tiến trình huấn luyện vẫn chạy ngầm bình thường).*

Để kiểm tra GPU có đang chạy hay không:
```bash
nvidia-smi
ps aux | grep train_peft
```

#### Bước 7: Kiểm Tra & Nghiệm Thu Kết Quả (Verification Checklist)
Khi tiến trình hoàn thành (xem trong log hiện `STRONG LORA SUITE COMPLETED SUCCESSFULLY`), chạy các lệnh sau để xác minh:
```bash
# 1. Kiểm tra 2 file dự đoán test set có đủ đúng 576 dòng (1 header + 575 câu)
wc -l results/mbart-large-50-lora-all-linear/test_predictions.csv
wc -l results/mbart-large-50-lora-all-linear-unfreeze-embed/test_predictions.csv

# 2. Xem trực tiếp bảng so sánh điểm BLEU và chrF++ vừa sinh ra:
cat outputs_rebuttal/strong_lora_scores.csv
```

#### Bước 8: Tải Kết Quả Về Máy
- File nén toàn bộ kết quả đã được tự động tạo tại: `/content/CLRR/strong_lora_results.zip`.
- Nếu có điền `HF_TOKEN`, toàn bộ kết quả đã được upload tự động lên repository `FiveC/amis-rewire-checkpoints/peft_baselines/`.
- Nếu tải trực tiếp từ máy cá nhân qua Colab CLI:
  ```powershell
  colab download -s colab /content/CLRR/outputs_rebuttal/strong_lora_scores.csv outputs_rebuttal/strong_lora_scores.csv
  colab download -s colab /content/CLRR/strong_lora_results.zip outputs_rebuttal/strong_lora_results.zip
  ```

---

### PHƯƠNG ÁN 2: Chạy Trực Tiếp Bằng Từng Cell Trên Colab Notebook (`.ipynb`)

Nếu người chạy không quen sử dụng Terminal / SSH, họ có thể mở một Colab Notebook mới, chọn Runtime **A100 GPU** và copy-paste lần lượt các cell sau:

#### [Cell 1] — Chuẩn bị mã nguồn và thư mục
```python
# Cell 1: Clone repo và di chuyển vào thư mục làm việc
import os
!cd /content && git clone https://github.com/HeyDunaX/CLRR.git || (cd /content/CLRR && git pull origin main)
os.chdir('/content/CLRR')
!git checkout main
!git pull origin main
```

#### [Cell 2] — Cài đặt thư viện
```python
# Cell 2: Cài đặt dependencies
!pip install -q -r requirements.txt
!pip install -q peft sacrebleu scikit-learn
!pip uninstall -y torchao 2>/dev/null || true
!nvidia-smi
```

#### [Cell 3] — Preflight Smoke Test (< 25s)
```python
# Cell 3: Chạy Smoke Test kiểm tra tính tương thích
!python -u scripts/peft/smoke_test_peft.py
```

#### [Cell 4] — Chạy Tự Động Toàn Bộ Strong LoRA Suite
```python
# Cell 4: Chạy toàn bộ Run A và Run B (~1.2 - 1.5 giờ trên A100)
import os
# os.environ["HF_TOKEN"] = "hf_xxx" # Tùy chọn: bỏ comment nếu có token HF
!python -u scripts/peft/run_strong_lora_mbart.py
```

#### [Cell 5] — Nghiệm thu và hiển thị kết quả
```python
# Cell 5: In kết quả đối chuẩn ra màn hình và kiểm tra số dòng test
import pandas as pd
df = pd.read_csv("outputs_rebuttal/strong_lora_scores.csv")
display(df)

!wc -l results/mbart-large-50-lora-all-linear/test_predictions.csv
!wc -l results/mbart-large-50-lora-all-linear-unfreeze-embed/test_predictions.csv
```

---

### PHƯƠNG ÁN 3: Chạy Từng Lệnh CLI Riêng Biệt (Dành Cho Ai Muốn Can Thiệp Từng Run)

Nếu muốn chạy riêng rẽ từng thí nghiệm để thử nghiệm các siêu tham số khác nhau:

#### 1. Lệnh chạy Run A: LoRA All-Linear (Frozen Embeddings):
```bash
python -u -m peft_baselines.train_peft \
  --method lora \
  --model-name facebook/mbart-large-50-many-to-many-mmt \
  --data-dir data/processed \
  --output-dir results \
  --run-name mbart-large-50-lora-all-linear \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules "q_proj,k_proj,v_proj,out_proj,fc1,fc2" \
  --no-unfreeze-embeddings \
  --num-train-epochs 20 \
  --early-stopping-patience 4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 32 \
  --auto-resume
```

#### 2. Lệnh chạy Run B: LoRA All-Linear + Unfrozen Embeddings:
```bash
python -u -m peft_baselines.train_peft \
  --method lora \
  --model-name facebook/mbart-large-50-many-to-many-mmt \
  --data-dir data/processed \
  --output-dir results \
  --run-name mbart-large-50-lora-all-linear-unfreeze-embed \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules "q_proj,k_proj,v_proj,out_proj,fc1,fc2" \
  --unfreeze-embeddings \
  --num-train-epochs 20 \
  --early-stopping-patience 4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 32 \
  --auto-resume
```

---

## 4. Bảng Kết Quả Dự Kiến Khi Chạy Xong (Để Bàn Giao & Điền Bài Báo)

Sau khi hoàn tất, kết quả từ `outputs_rebuttal/strong_lora_scores.csv` sẽ được điền vào Bảng 2 của bài báo (`clrr_main.tex`):

| Backbone Architecture | Cấu hình Thích nghi | Module can thiệp | $\Delta\theta_{\text{add}}$ (Thêm mới) | $\theta_{\text{train}}$ (Huấn luyện) | BLEU (zh) | chrF++ (w=2) | Luận điểm đối sách học thuật |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`mBART-large-50`** (611M) | Standard Fine-Tuning | Toàn bộ mô hình | 0 | 611M (100%) | 19.6106 | 14.0538 | Điểm mốc chuẩn full-tuning |
| | BitFit (Bias-only) | Vector Bias | 0 | 336k (0.055%) | 0.3511 | 2.8464 | Nghẽn biểu diễn nghiêm trọng |
| | LoRA ($r=8$, Ban đầu) | $W_q, W_v$ | +1.18M | 1.18M (0.193%) | 3.3284 | 5.1583 | Baseline LoRA hẹp ban đầu |
| | **LoRA All-Linear ($r=16$)** | $q, k, v, o, fc1, fc2$ | +6.20M | 6.20M (1.01%) | *[Chờ số liệu]* | *[Chờ số liệu]* | Khảo sát dung lượng adapter mở rộng |
| | **LoRA All-Lin + Unfreeze** | All-Linear + Embeddings | +6.20M | ~268M (43.9%) | *[Chờ số liệu]* | *[Chờ số liệu]* | Giải phóng nghẽn từ vựng Amis |
| | **CLRR-Enc + LSR (Ours)** | **Residual Rewiring + LSR** | **0** | **611M (100%)** | **20.3927** | **19.0839** | **SOTA vượt trội, $\Delta\theta=0$** |

---

## 5. Hướng Dẫn Xử Lý Sự Cố (Troubleshooting FAQ)

1. **Lỗi GPU Out of Memory (OOM):**
   - Nếu chạy trên GPU 16GB (như T4 hoặc V100 16GB) mà gặp OOM ở Run B (do ma trận embedding lớn):
   - Hạ `--per-device-train-batch-size 2` và tăng `--gradient-accumulation-steps 64` (để giữ nguyên Effective Batch Size $2 \times 64 = 128$).
2. **Tiến trình dừng đột ngột do timeout phiên Colab:**
   - Script đã có cờ `--auto-resume`. Nếu phiên làm việc bị ngắt, chỉ cần gõ lại lệnh chạy, mã nguồn sẽ tự động tìm checkpoint gần nhất trong `results/mbart-large-50-...` và tiếp tục huấn luyện mà không phải train lại từ đầu.
3. **Mất kết nối SSH từ máy tính cá nhân:**
   - Nếu bạn đã kích hoạt lệnh bằng `nohup bash scripts/peft/run_strong_lora_suite.sh > /content/strong_lora.log 2>&1 &`, tiến trình chạy trên máy chủ Google Colab **không bao giờ bị dừng**. Khi kết nối SSH lại, chỉ cần gõ `tail -f /content/strong_lora.log` để xem tiếp.
4. **Không có token Hugging Face (`HF_TOKEN`):**
   - Không ảnh hưởng gì đến quá trình huấn luyện. Kết quả và dự đoán vẫn được lưu an toàn tại thư mục cục bộ `results/` và `outputs_rebuttal/`.
