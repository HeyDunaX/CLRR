# Thiết kế & Kiểm định Phương pháp Vòng 5 — `Off-EOS Trajectory Huber Margin Regularization` (`off_eos_stopping` v1)

Ngày: **24/11/2026**.  
Trạng thái: **LOCKED DESIGN FOR ROUND 5 STAGE-1 A100 PILOT** — triển khai sạch `off-eos-chrf-v1` (Candidate B) và đối chứng khớp `off-eos-length-v1` (Control B1), chưa mở `test.csv`.  
Bằng chứng thiết kế & kiểm toán Vòng 5 lưu tại:
- [PAIR_THEORY_ROUND_B.json](../../outputs/experiments/round5_method_design/PAIR_THEORY_ROUND_B.json) — Phản biện NotebookLM Vòng B (`PILOT có điều kiện` cho Candidate B, `HOLD` cho Candidate A).
- [PAIR_FINAL_CHECKS.json](../../outputs/experiments/round5_method_design/PAIR_FINAL_CHECKS.json) và [verify_pair_formulas.py](../../outputs/experiments/round5_method_design/verify_pair_formulas.py) — Kiểm toán đạo hàm FP64 (`5.55e-17` sai số tối đa, `0.0` gradient ngoài tập điểm dừng).
- [PAIR_STOP_SCREEN.json](../../outputs/experiments/round5_method_design/PAIR_STOP_SCREEN.json) — Sàng lọc cắt ngắn 1 chiều trên bản dịch RTX (`299/576` Amis tăng `> 1` chrF, trung bình `+3.33`; `115/999` Ê Đê tăng `> 1` chrF, trung bình `+1.01`).

---

## 1. Động cơ Tổng quát & Tính Độc lập

1. **Tổng quát 100% cho mọi bài toán NMT / Seq2Seq:** Không dùng luật OOV, hình thái học, từ điển hay đặc thù ngôn ngữ nào của tiếng Amis hay Ê Đê. Áp dụng nguyên vẹn cho mọi cặp ngôn ngữ (`amis_mandarin`, `rhade_vietnamese`, v.v.) và mọi kiến trúc encoder-decoder (`mBART-50`, `NLLB-200`).
2. **Độc lập hoàn toàn với `SYM-SG` (EDWARD-NMT 2026 Eq. 13):** Giữ nguyên `CE` chuẩn trên dữ liệu vàng (không dùng `stop-grad` R-Drop, 0 dữ liệu ngoài, 0 LM ngoài, 0 tham số thêm, 0 inference overhead).
3. **Chuyển hẳn sang quỹ đạo on-policy vượt qua `EOS`:** Thay vì tiếp tục điều chỉnh phân phối token trên teacher-forcing (vốn đã chạm trần trên >130 run trước), phương pháp can thiệp trực tiếp vào **quyết định dừng thời gian 1 chiều** trên chuỗi tham lam (greedy) do chính mô hình tự sinh vượt qua điểm `EOS` tự nhiên.

---

## 2. Công thức Toán học Khóa (`off_eos_stopping` v1)

Với mỗi microbatch huấn luyện gồm $B$ cặp câu $(\mathbf{x}_b, \mathbf{y}_b)$:

### 2.1. Lượt 1 — Teacher-Forced Cross-Entropy Chuẩn
Giữ nguyên hoàn toàn loss nền:
$$\mathcal{L}_{\text{CE}} = \text{chunked\_ce}(f_\theta(\mathbf{x}, \mathbf{y}_{\text{in}}), \mathbf{y}).$$

### 2.2. Rollout Greedy On-Policy Vượt `EOS` (`eval()`, `torch.no_grad()`, có KV-Cache)
Encoder chỉ chạy 1 lần trên $(\mathbf{x}, \text{mask})$. Decoder tự sinh ở chế độ `eval()` (tắt dropout để quỹ đạo tất định) bắt đầu từ tiền tố đặc trưng `[decoder_start_token_id, forced_bos_token_id]` (hoặc `[decoder_start_token_id]` nếu `forced_bos_token_id` là `None`). Ở mỗi bước $t = 0, 1, \dots$:
- Nếu chưa ghi nhận điểm dừng tự nhiên $t_{\text{eos}, b}$ và $\arg\max_v z_{b,t,v} = \text{EOS}$, ghi nhận $t_{\text{eos}, b} = t$.
- Ép chọn token từ vựng tiếp theo $\hat{y}_{b,t} = \arg\max_{v \in \mathcal{V}_{\text{lex}}} z_{b,t,v}$ (mask toàn bộ token đặc biệt $\mathcal{V}_{\text{special}}$ gồm `pad`, `eos`, `bos`, `decoder_start`, `forced_bos` và `tokenizer.all_special_ids`) để quỹ đạo đi tiếp qua điểm dừng tự nhiên tối đa `extra_past_eos = 12` bước sau $t_{\text{eos}, b}$, không vượt quá ngưỡng an toàn:
  $$T_{\text{cap}, b} = \min\big(L_{\max,\text{tgt}} - 2,\; T_{\text{roll\_cap}},\; \max(2, \lfloor 1.5 L_b^* \rceil + \text{extra\_past\_eos})\big), \qquad T_{\text{roll\_cap}} = 96,$$
  trong đó $L_b^*$ là số token từ vựng thân câu của nhãn vàng $\mathbf{y}_b$.
- **Bảo vệ chống lặp vô hạn (Anti-Repetition Horizon Guard):** Dọc theo quỹ đạo từ vựng $\hat{\mathbf{y}}_b$, nếu xuất hiện vòng lặp 1-gram (`w, w, w`) hoặc 2-gram (`a, b, a, b`) bắt đầu từ bước $i \ge \max(4, t_{\text{eos}, b})$, cắt bớt đuôi lặp tại $i$ (giữ tối thiểu $T_b \ge 1$) để tránh các cụm từ lặp kéo điểm `chrF` giả tạo ở phần bịa sau `EOS`.

### 2.3. Tập Điểm Dừng Chấp Nhận Được $F_b$ (Khác biệt Nhân quả giữa Method Chính & Control)
- **`off-eos-chrf-v1` (Candidate B — Utility-Band Stopping):**
  Tính $\text{chrF}_0$ cấp câu (`word_order=0`, chuẩn `sacrebleu`) cho mọi tiền tố $\hat{\mathbf{y}}_{b, 0:k}$ ($k = 1, \dots, T_b$) so với bản dịch tham chiếu tập train $r_b$:
  $$s_{b,k} = \text{chrF}_0(\hat{\mathbf{y}}_{b, 0:k}, r_b), \qquad s_b^* = \max_{1 \le k \le T_b} s_{b,k}.$$
  Đặt tập điểm dừng hợp lệ $F_b = \{\,k \in \{1..T_b\} : s_{b,k} \ge s_b^* - \eta_{\text{band}}\,\}$ với $\eta_{\text{band}} = 0.5$. Nếu $s_b^* \le 0$ (mô hình không khớp được ký tự nào với câu tham chiếu), bỏ qua mẫu $b$ khỏi loss dừng (`active_fraction` phản ánh trực tiếp tỷ lệ tham gia).
- **`off-eos-length-v1` (Control B1 — Length-Only Falsifier):**
  Cùng rollout, cùng ngân sách tính toán và cùng công thức Huber-margin, nhưng thay $F_b$ bằng tập 1 điểm neo đúng độ dài từ vựng vàng:
  $$F_b = \{\,\text{clamp}(L_b^*, 1, T_b)\,\}$$
  (không truy cập `chrF` hay nội dung từ vựng đích). So sánh trực diện `off-eos-chrf-v1` vs `off-eos-length-v1` tách riêng đóng góp của chất lượng nội dung tại điểm dừng khỏi việc khớp độ dài kiểu Ma et al. (NAACL 2019).

### 2.4. Lượt 2 (Có Gradient) & Loss Huber Soft-Min
Chỉ chạy forward trên các mẫu tích cực ($|F_b| > 0$) với chuỗi giải mã tự sinh `[prefix, \hat{y}_{b,0}, ..., \hat{y}_{b,T_b-1}]` (`use_cache=False`, an toàn dưới `gradient_checkpointing(use_reentrant=True)`). Cắt lát logits `z_traj[:, P-1:, :]` (với $P = |\text{prefix}|$) để vị trí $t \in \{0, \dots, T_b\}$ khớp 1-1 với bước dự đoán $\hat{y}_{b,t}$ ($t < T_b$) và quyết định kết thúc sau $\hat{y}_{b, T_b-1}$ ($t = T_b$).

Đặt margin quyết định dừng tại bước $t$:
$$m_t = z_{t,\text{EOS}} - \max_{v \in \mathcal{V}_{\text{lex}}} z_{t,v}.$$
Chi phí dừng tại bước $k \in F_b$ ($\delta = 0.1$):
$$C_k = \sum_{t=0}^{k-1} \rho(m_t + \delta) + \rho(\delta - m_k), \qquad \rho(u) = \begin{cases} \frac{1}{2} u_+^2 & u_+ < 1 \\ u_+ - \frac{1}{2} & u_+ \ge 1 \end{cases}, \quad u_+ = \max(0, u).$$
Loss dừng trên mỗi câu ($\tau = 0.25$) và hàm mục tiêu tổng:
$$\mathcal{L}_{\text{stop}, b} = -\tau \log\!\left(\frac{1}{|F_b|} \sum_{k \in F_b} e^{-C_k/\tau}\right), \qquad \mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{eff}}(s)\, \frac{1}{|\mathcal{A}|}\sum_{b \in \mathcal{A}} \mathcal{L}_{\text{stop}, b},$$
với `stop_weight = 0.10` và warmup tuyến tính `stop_warmup_fraction = 0.10` ($\lambda_{\text{eff}}(s) = \lambda \cdot \min(1, \frac{s + 1}{W})$, chủ động từ step đầu tiên).

### 2.5. Suy luận Tự nhiên (`eval` / `generate`)
Khi `not self.training`, wrapper chuyển thẳng toàn bộ lời gọi tới `base_model` gốc (0 tham số thêm, 0 phụ thuộc `chrF` hay rollout mở rộng lúc đánh giá/test).

---

## 3. Chẩn đoán 0-Step (`off-eos-diagnose`) & Stage-1 Pilot A100

### 3.1. Chẩn đoán Bước 0 Bắt buộc Trước Train Dài (`--step0-only`)
Không chạy bước tối ưu nào (`optimizer_steps = 0`, `test_split_opened = false`):
1. Tải checkpoint công khai `BASELINE_S42` (`01a0b7a709ea5fcd24a95982f22c8150adaca4d5`, SHA `985f9fc4`) trên `amis_mandarin` validation (576 câu), chạy rollout 2 chiều `extra_past_eos = 12` để đo tách biệt:
   - `native_greedy` BLEU / chrF0 / LR;
   - `truncate_only_oracle` ($k \le t_{\text{eos}}$) vs `extend_only_oracle` ($k \ge t_{\text{eos}}$) vs `two_sided_oracle` vs `length_only_oracle` (trần corpus BLEU + tỷ lệ câu kích hoạt anti-repetition guard).
2. Đo tỷ số chuẩn gradient `||grad L_stop||_2 / ||grad L_CE||_2` và benchmark thời gian thực tế (`seconds_rollout`, `seconds_chrf`, `seconds_pass12`, `estimated_minutes_per_epoch`) trên 4 microbatch train (`B=32`).

### 3.2. Thiết kế Stage-1 Pilot A100 (`configs/experiments/amis_off_eos_pilot_a100.json`)
Protocol khóa [amis_a100_bleu.json](../protocols/amis_a100_bleu.json) (`BF16/TF32` train, `B32xGA4`, `LR=5e-5`, `cap=20 epochs`, `ES=10`, validation `beam 1` FP32 `zh`, seed `42`, `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`):
- **Tái sử dụng mốc A100:** `BASELINE_S42` (`17.5264` BLEU, `16.7632` chrF0, `14.5382` chrF++, SHA `985f9fc4`) và `W_B125_S42` (`17.7069` BLEU, `17.0817` chrF0, `14.8852` chrF++, SHA `4df107c0`). *(Ghi nhận tham khảo thêm `GOC_SYM_B250_S42 = 18.5143`.)*
- **2 Run mới:**
  1. `OFF_EOS_LENGTH_S42` (`off-eos-length-v1`) — Control B1 khớp hoàn toàn rollout/compute/loss, chỉ chọn điểm dừng theo độ dài vàng.
  2. `OFF_EOS_CHRF_S42` (`off-eos-chrf-v1`) — Candidate B chính thức chọn tập điểm dừng $F_b$ theo `chrF0`.

### 3.3. Cổng GO / NO-GO Khóa Trước
1. **Gate 1 (Vượt Baseline):** `BLEU(CHRF) - BLEU(BASELINE) >= +0.70` (`>= 18.23`) VÀ `chrF0(CHRF) > 16.7632`.
2. **Gate 2 (Vượt R-Drop SOTA):** `BLEU(CHRF) - BLEU(W_B125) >= +0.50` (`>= 18.21`) VÀ `chrF0(CHRF) > 17.0817` VÀ `chrF++(CHRF) > 14.8852`.
3. **Gate 3 (Falsifier Nhân quả vs Control B1):** `BLEU(CHRF) - BLEU(LENGTH) >= +0.30` (chứng minh chất lượng nội dung `chrF` quyết định gain, không phải học độ dài kiểu Ma et al. 2019).
4. **Gate 4 (Hiệu chuẩn độ dài):** `LR(CHRF) ∈ [0.93, 1.05]` (chống chạy lố triệt tiêu `EOS` như NotebookLM Vòng B đã cảnh báo).
- **Phán quyết:** Đạt Gates 1–4 $\implies$ **`FULL_GO`** (chốt Candidate B làm method chính cho xác nhận 3 seeds, `rhade_vietnamese`, `NLLB-200`, mở test `beam 4`). Đạt Gates 3–4 và thắng Baseline/R-Drop dương rõ (`>= +0.20` trên `W_B125`) nhưng hụt `+0.50` $\implies$ **`PROMISING_MECHANISM_REVIEW`**. Ngược lại $\implies$ **`NO_GO`**.
