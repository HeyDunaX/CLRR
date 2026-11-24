# Thiết kế & Kiểm định Phương pháp Vòng 5 — `2×2 Cross-Tail Lexical Fork Preference` (`lexical_fork` v1)

Ngày: **24/11/2026**.  
Trạng thái: **EXPLORATORY SEED-42 A100 PILOT (AUTHORIZED AFTER STEP-0 DIAGNOSTIC)** — giữ nguyên ghi nhận minh bạch các cổng Step-0 trước khi chạy pilot thăm dò 2 run trên A100, chưa mở `test.csv`.

---

## 1. Biên bản Chẩn đoán Step-0 & Phạm vi Diễn giải Khoa học

Toàn bộ kết quả Step-0 (`0` optimizer steps, không mở `test.csv`) trên checkpoint công khai `BASELINE_S42` (`01a0b7a709ea5fcd24a95982f22c8150adaca4d5`, SHA `985f9fc49574`) lưu tại [STEP0_REPORT.json](../../outputs/experiments/round5_lexical_fork_step0/STEP0_REPORT.json):

### 1.1. Truy vết chính xác mốc Native Baseline (`0.0109` BLEU difference)
- **`17.5264` BLEU** (`16.7632` chrF0, `BP = 0.9591`, `LR = 0.9599`, `6346/6611` ký tự) trong [VAL_COMPLETE.json](../../docs/tasks/reports/goc_pilot_a100_summary.json) và [STEP0_REPORT.json](../../outputs/experiments/round5_lexical_fork_step0/STEP0_REPORT.json) (`argmax` toàn vocab FP32 greedy gốc của `BASELINE_S42`).
- **`17.5373` BLEU** (`16.7709` chrF0, `BP = 0.9593`, `LR = 0.9601`, `6347/6611` ký tự) trong chẩn đoán `off_eos_stopping` trước đó bắt nguồn từ đường rollout khóa token đặc biệt ngoài `lexical` trước khi kiểm tra `EOS` (lệch 1 ký tự trên tập val). Hai artifact dùng nguồn gốc tách biệt nên luôn báo cáo tách bạch theo artifact.

### 1.2. Kết quả Step-0 `amis_mandarin` (`576` câu validation, `BASELINE_S42` FP32 `TF32=off`)

| Chính sách / Hệ thống | Số câu đổi | Corpus BLEU (`zh`) | Corpus chrF0 | BP | LR (`sys/ref`) | $\Delta$ BLEU vs Native |
|---|---:|---:|---:|---:|---:|---:|
| **`native` (`BASELINE_S42`)** | 0 | **17.5264** | **16.7632** | 0.9591 | 0.9599 | `0.0000` |
| `matched_natural_only` (`500` draws phân tầng, 1 token) | 91 | 18.1689 | 17.1567 | 0.9604 | 0.9611 | `+0.6425` |
| `all_natural_pos_one_token` (mọi câu thuận tự nhiên) | 175 | 18.3975 | 17.2964 | 0.9605 | 0.9613 | `+0.8711` |
| **`oracle_one_token_2x2` (`random_prefix`)** | **91** | **18.4654** | **17.4070** | **0.9583** | **0.9592** | **`+0.9390`** |
| **`oracle_natural_branch_2x2` (`h b u_b`, `random_prefix`)** | **91** | **18.8783** | **17.7237** | **0.9572** | **0.9581** | **`+1.3519`** |

- **Ổn định & Kích hoạt:** `rho_active = 49.65%` (`286/576`, `91` pos / `195` neg); độ ổn định dấu trên 2 nửa corpus rời nhau **`100.0%`** (`286/286`, `91/91`); sai lệch replay đảo thứ tự batch **`0.0%` (`0/128`)**.
- **Ghi nhận minh bạch cổng FAIL ở Step-0 (`gate_off_support_nll_lt_050: FAIL`):**
  Trên tập `pos_2x2` (`n=91`, `random_prefix`), chuỗi ghép chéo có `delta_nll_pooled_mean = 1.7082` nats (tỷ số perplexity $\approx e^{1.7082} = 5.52$, median `0.9822`, tại điểm nối `junction_delta = 4.62–4.82` nats), vượt ngưỡng `0.50` nats đã đặt trước. Do đó:
  1. Giữ nguyên ghi nhận `gate_off_support_nll_lt_050: FAIL` trong hồ sơ Step-0.
  2. Quyết định mở pilot **là pilot thăm dò sau khi xem kết quả Step-0** để kiểm nghiệm giả thuyết *bộ lọc nhãn huấn luyện* (**không** tuyên bố phép ghép đo lường chính xác Controlled Direct Effect nhân quả không bị nhiễu ngữ pháp).
- **Quan sát mật độ nhãn trên `train_subset_576` ở epoch 19:** Tại checkpoint hội tụ (`train BLEU = 95.68`), chỉ còn `3/524` (`0.57%`) nhãn dương sửa lỗi so với `521` nhãn củng cố — chứng minh bắt buộc phải chạy pilot ngay từ **pretrained initialization (`mBART-50`)**, ghi log mật độ nhãn qua từng epoch và tự động tắt loss phụ khi cạn nhãn sửa.

---

## 2. Thiết kế Khóa cho 2 Run Pilot A100 (`Seed 42`)

### 2.1. Hai Run Khớp Tuyệt Đối (`amis_lexical_fork_pilot_a100.json`)
Protocol khóa [amis_a100_bleu.json](../protocols/amis_a100_bleu.json): `BF16/TF32` train, `B32xGA4`, `LR=5e-5`, `max_epochs=20`, `ES10`, validation `beam 1` FP32 `zh`, seed/data_seed `42`, `gradient_checkpointing_use_reentrant=True`, `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
1. **`CTRL_NATURAL_S42`** (`lexical-fork-natural-v1`) — Đối chứng khớp công thức, rollout, phân tầng, số cập nhật dấu cộng/trừ và lịch $\lambda_e$, chỉ lấy nhãn từ hiệu số chéo tự nhiên ($h b u_b$ vs $h a u_a$).
2. **`FORK_2X2_S42`** (`lexical-fork-2x2-v1`) — Ứng viên chính dùng quy tắc đồng thuận 6 hiệu số $2 \times 2$.

### 2.2. Sinh Nhãn Đóng Băng Đầu Mỗi Epoch & Bộ Lọc Khớp Đối Chứng
1. **Chọn prefix không nhìn reference (`random_prefix`):** Với mỗi câu train $i \in \{0..4599\}$, đầu mỗi epoch có loss phụ ($e \ge 2$), dùng băm tất định `blake2b(seed, epoch_idx, row_id)` để chọn đúng 1 vị trí nội dung hợp lệ (top-1 greedy $a$ có đối thủ tương thích $b$ cùng ranh giới SentencePiece `▁` vs phần tiếp nối).
2. **Nền corpus cố định suốt epoch (`FP64`, `BLEU zh` + `chrF0`):** Chạy greedy rollout trên **toàn bộ 4,600 câu train** ở chế độ `eval()`, `FP32`, `TF32=off` để tạo chuỗi $y^{\text{native}}_i$ và vector thống kê đủ $T_m = \sum_i t^{\text{native}}_{m,i}$, sau đó sinh nhánh thay thế $u_b$ không đệm theo bucket độ dài và chấm $S_m(v) = m(T_m - t^{\text{native}}_{m,i} + t^v_{m,i})$ cho 4 biến thể $hau_a, hbu_a, hau_b, hbu_b$ với ngưỡng hòa `1e-10`.
3. **Ngưỡng cạn tín hiệu (`raw_pos_2x2 < 32 => lambda_e = 0`):** Trên snapshot của từng run, đếm số câu đạt $s_{2\times 2} = +1$. Nếu `< 32`, đặt $\lambda_e = 0$ cho riêng epoch đó (vẫn ghi đầy đủ số thô và kiểm tra lại ở epoch kế tiếp).
4. **Cân bằng dấu & 5 tầng margin (`|p2_g|, |n2_g|`):** Sắp xếp ổn định toàn bộ fork theo `(margin, row_id)` và chia 5 tầng đều nhau (`np.array_split`). Trong mỗi tầng $g$, đặt hạn mức $q_g = \min(|p^{(2\times 2)}_g|, |n^{(2\times 2)}_g|)$.
   - **`FORK_2X2_S42`:** chọn $q_g$ câu dương và $q_g$ câu âm từ pool `2×2` theo thứ tự băm `blake2b(seed, epoch, '2x2', sign, row_id)`.
   - **`CTRL_NATURAL_S42`:** chọn đúng $q_g$ câu dương và $q_g$ câu âm từ pool `natural` theo băm `blake2b(seed, epoch, 'natural', sign, row_id)` (bảo đảm khớp chặt số lượng dấu `+` và `−` theo từng tầng margin trong snapshot của run đó; tự động kiểm chứng `|p_nat_g| >= q_g` và `|n_nat_g| >= q_g`).

### 2.3. Hàm Mục Tiêu, Lịch $\lambda_e$, Chuẩn hóa Cửa sổ Accumulation & Kiểm tra Epoch 1
- **Lịch $\lambda_e$:** Epoch 1: $\lambda_1 = 0.0$ (`CE` thuần); Epoch 2: $\lambda_2 = 0.05$; Epoch 3 trở đi: $\lambda_e = 0.10$ (nếu không bị khóa về `0` bởi ngưỡng `< 32`).
- **Chuẩn hóa trên toàn cửa sổ gradient accumulation ($G = 4$ microbatch):**
  Gọi $K_w = \sum_{m=1}^{G} |\mathcal{A}_m|$ là tổng số fork được chọn trong cả cửa sổ 4 microbatch tạo nên 1 bước optimizer (`K_w == 0 => fork_term = 0`). Do `ControlTrainer` (`accepts_loss_kwargs = False`) tự chia loss mỗi microbatch cho $G$, mô hình trả về ở microbatch $m$:
  $$\mathcal{L}_m = \mathcal{L}_{\text{CE}, m} + \lambda_e \, \frac{G}{K_w} \sum_{i \in \mathcal{A}_m} \log\!\big(1 + \exp[-s_i(z_{b_i} - z_{a_i})]\big),$$
  nên tổng gradient sau $G$ microbatch bằng chính xác $\frac{1}{G}\sum_{m=1}^{G}\nabla\mathcal{L}_{\text{CE},m} + \lambda_e \cdot \frac{1}{K_w}\sum_{i \in \text{window}} \nabla\log(1 + e^{-s_i(z_{b_i}-z_{a_i})})$.
- **Kiểm tra tích hợp Epoch 1 (`CE` parity):** Vì epoch 1 của cả 2 run đều dùng $\lambda_1 = 0$, loss từng step, số optimizer step và validation metrics/predictions cuối epoch 1 giữa `CTRL_NATURAL_S42` và `FORK_2X2_S42` phải khớp nhau.

---

## 3. Tiêu chí Quyết định Sau Pilot (`GO / NO-GO`)

Báo cáo đầy đủ **BLEU (`zh`), chrF0, chrF++, BP, LR**, phân bố nhãn thô vs thực dùng qua từng epoch, chênh lệch số cập nhật thực tế giữa hai run, và thời gian riêng cho **labeling / train / eval**:
1. **`GO_MULTISEED`:** `BLEU(FORK_2X2) - BLEU(CTRL_NATURAL) >= +0.30` VÀ `chrF0(FORK_2X2) > chrF0(CTRL_NATURAL)` VÀ `|LR(FORK_2X2) - LR(CTRL_NATURAL)| <= 0.01` VÀ `BLEU(FORK_2X2) > 17.5264` (`BASELINE_S42`) $\implies$ mở xác nhận đa seed & Ê Đê.
2. **`TRAIN_SIGNAL_EXHAUSTED`:** Loss phụ bị tắt gần như toàn bộ hoặc tổng fork thực dùng quá thưa không đủ tác động tối ưu hóa.
3. **`NO_GAIN_OVER_NATURAL_CONTROL`:** Hai nhánh có đủ tín hiệu hoạt động nhưng `FORK_2X2` ngang hoặc thua `CTRL_NATURAL` $\implies$ bác bỏ giá trị gia tăng của quy tắc hoán đổi `2×2`.
4. **`NO_GO`:** Không đạt ngưỡng `GO_MULTISEED` (kể cả khi `FORK_2X2` tụt dưới `BASELINE_S42`).
