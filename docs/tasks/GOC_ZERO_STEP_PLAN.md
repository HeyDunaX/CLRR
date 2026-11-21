# Kế hoạch bàn giao: Chẩn đoán 0 bước cho GOC trên A100 (`work/goc-zero-step`)

Ngày lập: **21/11/2026** (Local Windows → Server A100 JupyterLab).
Tài liệu nền: [ROUND2_GOC_IPAM_DESIGN_AUDIT.md](../research/ROUND2_GOC_IPAM_DESIGN_AUDIT.md), [JUPYTERLAB_A100_GUIDE.md](../operations/JUPYTERLAB_A100_GUIDE.md), manifest artifacts nhỏ tại [MANIFEST.json](artifacts/goc_zero_step/MANIFEST.json).

---

## 1. Mục tiêu, giả thuyết và phạm vi được phép

### Mục tiêu
Kiểm chứng **trực tiếp trên checkpoint `mBART-50` thật (`W_S42` cố định + pretrained `mBART-50` cố định)** với **0 optimizer steps** xem hiện tượng nền tảng của **GOC (`Gold-Ordered Asymmetric Dropout Consistency`)** có thực sự xảy ra ở mức đáng kể và lặp lại hay không — **trước khi quyết định có mở pilot training trên A100 hay dừng GOC**.

### Giả thuyết cần kiểm chứng (và bác bỏ nếu sai)
- **H_GOC:** Trên checkpoint thật `W_S42` dưới các cặp dropout độc lập:
  1. **Case 2 thực tế** (vị trí subword đích có bất đồng giữa 2 dropout views mà thành phần đối xứng `-g_sym` và/hoặc tổng gradient phụ `W` làm giảm logit gold hoặc giảm khoảng cách **gold-vs-strongest-competitor margin** $\Delta m_t = z(y_t) - \max_{v \ne y_t} z(v)$ trên view mạnh hơn $w_t$) xảy ra với tần suất và biên độ đáng kể, đặc biệt trên các **subword đích hiếm ($\le 10$ lần trong `train.csv`)**.
  2. Trên view yếu hơn $\ell_t$, hướng cập nhật của `GOC` ($p_{w_t} - p_{\ell_t}$) **thực sự tăng khoảng cách margin** $\Delta m_t$ (không chỉ tăng riêng $z(y_t)$ trong khi cũng kéo đối thủ mạnh nhất lên tương đương), và dấu hiệu này lặp lại ổn định qua **8 cặp dropout masks độc lập**.
  3. View thắng $w_t$ **không** chủ yếu thắng bằng cách đẩy khối lượng xác suất sang top-100 subword tần suất cao (bác bỏ rủi ro $w_t$ chỉ là nhiễu target-LM prior).
- **H_NULL (Bác bỏ / STOP):** Case 2 hiếm hoặc hiệu ứng lên gold-vs-competitor margin rất nhỏ / không lặp qua 8 cặp dropout, hoặc $p_{w_t}$ làm xấu đuôi phân phối / tăng khối lượng prior trên subword hiếm ⇒ **DỪNG hướng GOC**, không mở pilot training.

### Phạm vi được phép cho Server (TUÂN THỦ NGHIÊM NGẶT)
- **ĐƯỢC PHÉP:**
  1. Đọc tài liệu, xác minh commit, data SHA, checkpoint HF SHA, và kiểm tra môi trường `conda run -n clrr ...`.
  2. Viết script chẩn đoán 0 bước độc lập (ví dụ `scripts/analysis/probe_goc_zero_step.py`) và bộ unit test CPU (`tests/test_goc_zero_step.py`).
  3. Chạy **CPU tests** (kiểm đại số FP64/FP32, tie handling, mask `-100`, exact margin delta, exact RNG replay).
  4. Chạy **1 job GPU chẩn đoán 0 optimizer steps trên A100 40GB** (chế độ **FP32, TF32 off, SDPA** để khớp số học với checkpoint `W_S42` đã pin; 0 bước optimizer, không sửa weights; dọn weights tải tạm sau khi verify receipts).
  5. Ghi báo cáo định lượng + JSON/CSV nhỏ và push kết quả lên cùng nhánh `work/goc-zero-step`.
- **CHƯA ĐƯỢC PHÉP (TUYỆT ĐỐI KHÔNG CHẠY TRONG LƯỢT NÀY):**
  - **KHÔNG** viết hoặc chạy training dài (`GOC_S42`, `GOC_RAND_S42`, `GOC_INV_S42`, `W_S42`, `BASELINE_S42`).
  - **KHÔNG** đánh giá tập `test.csv`.
  - **KHÔNG** chạy IPAM / Self-PoE (đang ở trạng thái HOLD).
  - **KHÔNG** tự merge vào `main`, không sửa receipts lịch sử, không đưa weights/cache/logs lớn vào Git.

---

## 2. Môi trường thực thi & Lưu trữ trên Server

- **Máy chạy:** NVIDIA **A100 40 GB** trên JupyterLab (xác nhận quyết định tác giả ngày 21/11/2026: **chuyển hoàn toàn sang A100 cho chẩn đoán và các pilot sau này, không chạy RTX nữa**).
  - *Lưu ý minh bạch cohort:* Chẩn đoán 0 bước đo trên checkpoint `W_S42` (vốn train trên RTX FP32 SDPA) và pretrained `mBART-50` gốc; chạy phép đo frozen ở **FP32, TF32 off, SDPA** trên A100 để tránh lệch số học BF16/TF32 và **không** trộn số đo frozen này thành kết quả training cohort A100.
- **Thư mục làm việc:** `/workspace/long/CLRR`
- **Conda env:** `clrr` (mọi lệnh Python dùng `conda run -n clrr python -X utf8 ...`).
- **Nơi lưu cache / tmp / outputs:** Giữ hoàn toàn trong `/workspace/long/CLRR/.cache/` (hoặc biến `HF_HOME` / `TMPDIR` nằm trong `/workspace/long/CLRR`), outputs thô lưu tại `/workspace/long/CLRR/outputs/experiments/goc_zero_step_a100/` (ngoài Git). Sau khi xong và lưu receipts nhỏ, **dọn weights tạm** giải nén từ HF.

---

## 3. Dữ liệu, Checkpoints, Seeds & Công thức đã khóa

### 3.1. Dữ liệu (`data_processed/amis_mandarin/`)
Xác minh trước khi chạy:
- `train.csv`: **4.600** dòng, SHA256 `4e13d83f035a18c67e1b2e3990baff3aa92154a34b754a4d0cdef9a71d85b8e6`
- `validation.csv`: **576** dòng, SHA256 `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4`
- `test.csv`: **KHÔNG ĐỤNG ĐẾN**.
- Ngôn ngữ/Tokenizer: `src_lang = tl_XX`, `tgt_lang = zh_CN`, `max_source_length = 256`, `max_target_length = 256`, valid target mask `labels != -100` (giữ nguyên quy ước `src/methods/decoupled_consistency/v1/model.py`; báo cáo tách riêng **lexical subwords** vs **special tokens** EOS/lang_id nếu có trong `labels`).

### 3.2. Checkpoints cố định (0 optimizer steps)
1. **Checkpoint trained `W_S42` (trọng tâm đo cơ chế):**
   - HF repo: `FiveC/amis-rewire-checkpoints`
   - Revision đã pin: `de91462e792e66a92d51770b00cb674f4566b15d`
   - File: `checkpoints/amis_mandarin/mbart-large-50/weak-full-consistency-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu/best.zip`
   - Archive SHA256: `ebf1a2ebf080340e133be07cb902908d55f48f3165a4c973acfe358e0f457c74`
   - *Kiểm tra tính toàn vẹn bắt buộc:* Xác minh SHA256 của `best.zip` trước giải nén và xác nhận tổng tham số / reload strict khớp, không thay đổi weights sau job.
2. **Pretrained initialization `mBART-50` (đo tỷ số full-backbone gradient norm khởi tạo):**
   - Model ID: `facebook/mbart-large-50-many-to-many-mmt`
   - Revision đã pin: `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`

### 3.3. Mẫu dòng, Microbatch & Dropout Seeds cố định
- **Tập TRAIN (đo động học tối ưu & full-backbone gradient norms):**
  - Lấy **64 dòng TRAIN cố định** (rows `0..63`, tức 16 microbatches cỡ 4 — hoặc 2 microbatches cỡ 32; khóa rõ **microbatch = 4** để khớp hình học padding với `W_S42` và **ghi thêm** 3 batches cỡ 32 cho gradient-norm calibration tương ứng recipe A100 B32×GA4!).
- **Tập VALIDATION (đo hành vi trên dữ liệu không thuộc train, tách bảng riêng với TRAIN):**
  - Lấy **128 dòng VALIDATION cố định** (chọn đều chỉ số `row_id = int(i * 576 / 128)` cho `i = 0..127`, cố định và lưu danh sách `row_ids` vào receipt).
- **8 cặp dropout masks độc lập:**
  - Dùng `torch.manual_seed` / `torch.cuda.manual_seed_all` cố định theo từng `(batch_idx, pair_idx, view_idx)` với `pair_idx in 0..7`, `view_idx in {0, 1}` (công thức seed xác định ví dụ `base_seed = 20261121`, `seed = base_seed + batch_idx * 100 + pair_idx * 10 + view_idx`), bật `model.train()` để dùng đúng native dropout của `mBART-50`.
  - Bắt buộc kiểm tra: (a) hai views trong cùng một pair cho logits khác nhau (`max_abs_diff > 0`); (b) replay cùng seed cho kết quả khớp chính xác (`max_abs_diff == 0` hoặc `< 1e-6` với deterministic SDPA/FP32).

### 3.4. Công thức đo (trên cùng logits `z1, z2` của mỗi cặp dropout)
Với $p_1 = \operatorname{softmax}(z_1)$, $p_2 = \operatorname{softmax}(z_2)$ (FP32):
- **W (R-Drop $\beta_W = 1.25$):** $\mathcal{L}_{\text{aux}}^W = 1.25 \cdot \frac{1}{2}[\text{KL}(p_1\|p_2) + \text{KL}(p_2\|p_1)]$.
  - Phân rã gradient theo trục view: $g_{\text{anti}} = \frac{g_1 - g_2}{2}$, $g_{\text{sym}} = \frac{g_1 + g_2}{2}$ (lực descent là $-g$).
- **AC ($\beta = 1.25$):** Chỉ giữ $g_{\text{anti}}$ (bỏ $g_{\text{sym}}$).
- **GOC ($\beta_{\text{goc}} \in \{1.25, 2.50\}$):** $w_t = \arg\max_{v\in\{1,2\}} \log p_{v,t}(y_t)$, $\ell_t = 3 - w_t$ (hòa chia 0.5/0.5), $\mathcal{L}_{\text{aux}}^{\text{GOC}} = \beta_{\text{goc}} \cdot \text{KL}(\operatorname{sg}(p_{w_t, t}) \| p_{\ell_t, t})$.
- **GOC-SYM-SG (EDWARD-NMT Eq. 13 tất định) & GOC-RAND (chọn ngẫu nhiên 50/50):**
  - `SYM-SG`: $\frac{\beta_{\text{goc}}}{2}[\text{KL}(\operatorname{sg}(p_1)\|p_2) + \text{KL}(\operatorname{sg}(p_2)\|p_1)]$.
  - `RAND`: $\beta_{\text{goc}} \cdot \text{KL}(\operatorname{sg}(p_{r_t}) \| p_{3-r_t})$, $r_t \sim \text{Bern}(0.5)$ cố định theo seed.
- **GOC-INV:** $\beta_{\text{goc}} \cdot \text{KL}(\operatorname{sg}(p_{\ell_t, t}) \| p_{w_t, t})$ (đảo ngược).

---

## 4. Phép đo bắt buộc & Cách tổng hợp

Giữ tách biệt rõ **đơn vị subword đích (`mBART-50` token IDs)** khi phân tần suất từ `train.csv` (tính tần suất xuất hiện của từng target subword ID trong toàn bộ 4.600 câu `train.csv`, chia 3 nhóm: **Hiếm $\le 10$**, **Trung bình $11\text{–}100$**, **Phổ biến $>100$**, tách riêng special tokens). Không trộn lẫn subword ID với ký tự Hán.

### Phần A — Phép đo cấp token/logit trên `W_S42` (TRAIN 64 câu & VAL 128 câu, tách bảng; 8 cặp dropout/câu)
Báo cáo **cả hai mẫu số**: (i) **toàn bộ vị trí subword hợp lệ** và (ii) **nhóm bất đồng mạnh** ($|p_1(y_t) - p_2(y_t)| > 0.05$), đồng thời phân tầng theo 3 nhóm tần suất subword (`<=10`, `11-100`, `>100`):
1. **Tần suất Case 2 thật trên `W_S42`:**
   - Tỷ lệ vị trí có `-g_sym(y_t) < 0` (thành phần chung của SKL đẩy giảm logit gold).
   - Tỷ lệ vị trí có tổng lực descent của SKL trên view thắng `w_t` đẩy giảm logit gold (`-g_W_on_w(y_t) < 0`).
   - Tỷ lệ vị trí mà `-g_sym` làm giảm entropy (`dH/dt < 0`) trên `W_S42` thật (đối chiếu với con số mô phỏng 80.6%).
2. **Tác động lên khoảng cách gold-vs-strongest-competitor margin** $\Delta m_t = z(y_t) - \max_{v \ne y_t} z(v)$:
   - Với mỗi vị trí $t$, tính đạo hàm định hướng bậc 1 của margin theo từng luật cập nhật aux (`W` trên $w_t$, `W` trên $\ell_t$, `W` trung bình 2 views, `AC`, `GOC`, `SYM-SG`, `INV`):
     $$\dot{m}_t^{(\text{method})} = (-g_t^{(\text{method})}[y_t]) - (-g_t^{(\text{method})}[c_t^*]), \qquad c_t^* = \arg\max_{v \ne y_t} z_t[v].$$
   - Báo trung bình, trung vị, tỷ lệ dương ($\dot{m}_t > 0$) và độ ổn định (mean ± std qua 8 cặp dropout, kèm 95% bootstrap theo câu).
3. **Kiểm tra rủi ro $p_{w_t}$ là nhiễu prior:**
   - So sánh trên nhóm subword hiếm (`<=10`): (a) tổng khối lượng xác suất non-gold trên top-100 subword tần suất cao nhất của `train.csv` giữa $p_{w_t}$ và $p_{\ell_t}$; (b) entropy non-gold của $p_{w_t}$ vs $p_{\ell_t}$.
   - **Không** báo kết quả tầm thường $p_{w_t}(y_t) > p_{\ell_t}(y_t)$ làm bằng chứng cơ chế.

### Phần B — Phép đo Full-Backbone Parameter Gradient Norm (trên 3 batches TRAIN cố định, cùng dropout views)
Đo trên **cả `pretrained mBART-50` (init)** và **`W_S42` (trained)**, ở cùng microbatch/dropout views (báo cho cả `B=4` và `B=32`):
1. $\|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2$, $\|\nabla_\theta \mathcal{L}_{\text{aux}}^{W}\|_2$ ($\beta=1.25$), $\|\nabla_\theta \mathcal{L}_{\text{aux}}^{\text{GOC}}\|_2$ (tại $\beta=1.25$ và $\beta=2.50$), $\|\nabla_\theta \mathcal{L}_{\text{aux}}^{\text{SYM-SG}}\|_2$, $\|\nabla_\theta \mathcal{L}_{\text{aux}}^{\text{INV}}\|_2$.
2. Tỷ số $\|\nabla_\theta \mathcal{L}_{\text{aux}}^{\text{GOC}}\| / \|\nabla_\theta \mathcal{L}_{\text{aux}}^{W}\|$ và $\|\nabla_\theta \mathcal{L}_{\text{aux}}^{\text{GOC}}\| / \|\nabla_\theta \mathcal{L}_{\text{CE}}\|$ → xác định chính xác hệ số $\beta_{\text{goc}}$ nếu đi tiếp.
3. Cosine similarity giữa các parameter gradients: $\cos(\nabla_\theta L_{\text{aux}}^{\text{GOC}}, \nabla_\theta L_{\text{CE}})$, $\cos(\nabla_\theta L_{\text{aux}}^{W}, \nabla_\theta L_{\text{CE}})$, $\cos(\nabla_\theta L_{\text{aux}}^{\text{GOC}}, \nabla_\theta L_{\text{aux}}^{W})$, $\cos(\nabla_\theta L_{\text{aux}}^{\text{GOC}}, \nabla_\theta L_{\text{aux}}^{\text{INV}})$.

---

## 5. Ngưỡng quyết định (GO / HOLD / STOP) khóa trước

- **GO đề xuất pilot training (cần trình local duyệt trước khi chạy train):**
  1. Tất cả CPU tests + kiểm tra tính toàn vẹn checkpoint/RNG replay PASS.
  2. Trên `W_S42` thật (nhóm bất đồng mạnh & subword hiếm $\le 10$, lặp qua 8 cặp dropout): Case 2 (`-g_W_on_w` làm giảm margin $\dot{m}_t < 0$) xảy ra rõ rệt (không phải hiện tượng hiếm $<5\%$), VÀ `GOC` cho **lợi thế dương ròng trên margin $\dot{m}_t$** (vượt cả `W` và `SYM-SG`/`INV`) trên subword hiếm, VÀ $p_{w_t}$ không làm tăng khối lượng prior top-100 so với $p_{\ell_t}$.
  3. Gradient toàn backbone hữu hạn, hướng phân tách rõ khỏi `W` và `INV`, cho thấy mức $\beta_{\text{goc}}$ hợp lý (tỷ số norm với `W` trong khoảng hợp lý, không nổ hoặc triệt tiêu).
- **STOP (dừng hướng GOC ngay, không mở 3 run train):**
  - Case 2 quá hiếm hoặc hiệu ứng lên gold-vs-competitor margin quá nhỏ / không lặp qua 8 cặp dropout; HOẶC `GOC` không tăng margin hơn `SYM-SG` trên subword hiếm; HOẶC $p_{w_t}$ chủ yếu tăng khối lượng prior top-100 / gây lệch phân phối đuôi xấu hơn.
- **Giới hạn tài nguyên:** Peak VRAM < 32 GiB (A100 40GB), tổng thời gian chạy job 0-step dự kiến < 20 phút. Nếu OOM hoặc lỗi số học (NaN/Inf), dừng và báo lỗi nguyên trạng.

---

## 6. Ghi chú cho pha Pilot A100 tiếp theo (CHƯA CHẠY TRONG LƯỢT NÀY — chỉ lưu để đồng bộ)

Tác giả chọn hướng cho các pilot train sau này trên A100: **Phương án 1 (khớp recipe A100 BF16, B32×GA4, LR 5e-5, max 20 epochs, ES10) nhưng dùng selector `Val BLEU` (beam 1, FP32)**.
Kết quả audit trajectory toàn bộ 14 run A100 ES10 cũ (`docs/tasks/artifacts/goc_zero_step/a100_selector_AUDIT.json`):
- **0/14 run A100 ES10 bị dừng sớm do `chrF++`** (cả 14/14 chạy đủ 20/20 epochs).
- **8/14 run trùng chính xác best epoch** giữa `chrF++` và `BLEU` (`DiffBLEU = 0.0000`, gồm `A01`, `A03`, `A06`, `B01`, `B03`, `B04`, `C01`, `D02`).
- **6/14 run lệch nhẹ:** đáng chú ý nhất là **`A02 mBART Middle s42`** (Ep 16 chrF++ `16.4643` vs Ep 18 BLEU `16.6657`, lệch **`+0.2014` BLEU**); các run còn lại lệch rất nhỏ (`+0.0044` đến `+0.1292`).
- **`Baseline` A100 ES10 cũ (`A01`) là seed 46**, chưa có `Baseline_S42` ES10 và **chưa có `W_S42` trên A100**.
- Do đó, **nếu và chỉ nếu 0-step PASS** và tác giả duyệt mở pilot A100, các run nền tối thiểu để fair trên A100 seed 42 sẽ là `BASELINE_S42` + `W_S42` (+ `MIDDLE_S42` nếu muốn loại độ lệch +0.2014 của A02), còn lại (`LoRA A/B`, `Narrow LoRA`, `BitFit`, `NLLB`, `mT5`, `ByT5`) tái sử dụng từ A100 cũ.

---

## 7. Artifacts bắt buộc Server bàn giao lên nhánh `work/goc-zero-step`

Khi hoàn tất phạm vi được phép, server đẩy lên nhánh `work/goc-zero-step` (chỉ file code, test, báo cáo và JSON/CSV nhỏ — **không** đẩy `.ipynb_checkpoints/`, `codex_reply.md`, `setup-result.txt`, weights hay cache):
1. Code & test: `scripts/analysis/probe_goc_zero_step.py`, `tests/test_goc_zero_step.py`.
2. Báo cáo bàn giao: `docs/tasks/reports/GOC_ZERO_STEP_REPORT.md` (gồm lệnh đã chạy, runtime, versions, commit/data/checkpoint SHA, toàn bộ bảng số Phần A & B, giới hạn, và khuyến nghị GO/STOP).
3. Receipts nhỏ đã rà: `docs/tasks/reports/goc_zero_step_results.json` (và/hoặc CSV tóm tắt nhỏ < 500 KB) kèm SHA ánh xạ nếu có file gốc nằm trong `outputs/`.
