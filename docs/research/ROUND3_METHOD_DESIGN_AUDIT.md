# Thiết kế & Kiểm định phương pháp Vòng 3 — `CR-Dual` (Giữ) và Loại `SGR` / `AF-SCD` trước training

Ngày: **22/11/2026**.  
Trạng thái: **DESIGN & AUDIT ONLY** — chưa viết code training production, chưa chạy GPU gate, chưa train và chưa mở test set.  
Mọi kết quả thực nghiệm trích từ [báo cáo pilot A100 9-run GOC](../tasks/reports/GOC_PILOT_A100_REPORT.md) và [goc_pilot_a100_summary.json](../tasks/reports/goc_pilot_a100_summary.json). Toàn bộ bằng chứng toán học FP64 lưu tại `outputs/experiments/round3_method_design/MATH_CHECKS*.json`; nhật ký thêm nguồn PDF và phản biện NotebookLM lưu tại `outputs/experiments/round3_method_design/NOTEBOOKLM_ROUND*.json`.

---

## 1. Tóm tắt quyết định sau Audit Vòng 3 (`METHOD_DEVELOPMENT_HANDOFF.md` §4)

Mục tiêu vòng 3: thiết kế một phương pháp **đủ độ mới ở mức bài ACL Main** (không lặp lại tinh chỉnh nhỏ quanh R-Drop hay architectural rewiring vốn đã cạn dư địa), có **0 dữ liệu ngoài / 0 LM ngoài / 0 tham số thêm / 0 inference overhead**, và **tổng quát hóa được trên cả `amis_mandarin` lẫn `rhade_vietnamese`** (và nhiều backbone many-to-many như `mBART-50`, `NLLB-200`).

| Ứng viên | Cơ chế cốt lõi | Kiểm toán FP64 & Phản biện NotebookLM (2 vòng) | Quyết định |
|---|---|---|---|
| **Draft 1 — `SGR`** (*Source-Gated Prefix-Error Self-Recovery*) | Gây nhiễu prefix bằng top-1 non-gold + `KL(sg(p_clean) \|\| p_tilde)` có trọng số nguồn từ `roll(H(X), 1)` | NotebookLM Vòng A bác: **GẦN (tổ hợp kỹ thuật)** từ Guo et al. (ACL 2022 PD-R) + He et al. (EAMT 2024) + Lee et al. (EMNLP 2022 NPMI); thiếu ràng buộc chống over-recovery (He 2024 Eq. 3); mâu thuẫn cú pháp khi thay prefix. | **LOẠI TRƯỚC TRAIN (KILLED)** |
| **Draft 2 — `AF-SCD`** (*Adaptive-Plausibility Source-Contrastive Self-Distillation*) | Đưa Source-Contrastive Decoding (Sennrich 2024) thành teacher log-linear `exp((1+γ)log p - γ log p')` trong lúc train | Phản ví dụ FP64 (`MATH_CHECKS.json`): khi mô hình đoán sai đặc thù theo nguồn (`0.65`), `AF-SCD` khuếch đại lỗi lên **`0.946`** và đè gold (`0.05 → 0.018`). NotebookLM Vòng A chỉ ra kỳ dị chia `p'_t(v) → 0` và bất liên tục tại biên mask. | **LOẠI TRƯỚC TRAIN (KILLED)** |
| **Ứng viên giữ lại — `CR-Dual` (V1: `CR_HUBER_FULL`)** (*Marginal-Free Cross-Ratio Dual Consistency*) | Áp dụng định lý tương thích điều kiện **Arnold & Press (1989)** qua **4-point log cross-ratio** $M^{\to}_\theta(i,j) = M^{\leftarrow}_\theta(i,j)$ trong cùng mạng many-to-many để **triệt tiêu tuyệt đối cả hai LM biên $P(x), P(y)$** của Xia et al. (ICML 2017 DSL) | **Novelty: `CHƯA THẤY` trong NLP/NMT** trên toàn bộ 9 paper gốc đối chiếu (Xia 2017 DSL, Ding 2021 BiT, Zheng 2021 REDER, Sennrich 2024 SCD, Lee 2022 NPMI, Voita 2021, SIWCon, R-Drop, EDWARD-NMT). Thắng áp đảo mọi đối chứng trên bài kiểm tra tổng quát hóa neural 5-seed FP64 (`1.215` NLL vs `BiT 1.591`, `DSL 1.582`, `Shuffled 1.304`). NotebookLM Vòng B: **`ADMIT TO LOCKED CANDIDATE CARD`**. | **GIỮ LÀM ỨNG VIÊN DUY NHẤT (ADMITTED)** |

*(Hai biến thể phụ của `CR-Dual` là `V2: CR_HUBER_NO_OFFDIAG_GRAD` và `V3: CR_PAIR_JS_FULL` đều đã bị **LOẠI (KILLED)** sau kiểm toán Vòng A/B: V2 thoái hóa thành hệ số nhân CE cục bộ làm mất gradient cross-ratio; V3 bão hòa sigmoid sớm và hầu như không vượt `BiT`.)*

---

## 2. Candidate Card Khóa Trước — `CR-Dual` (Marginal-Free Cross-Ratio Dual Consistency)

### 2.1. Câu hỏi nghiên cứu (RQ)
**RQ:** Khi tinh chỉnh một mô hình dịch máy đa ngữ tiền huấn luyện hai chiều (`mBART-50` / `NLLB-200`) cho cặp ngôn ngữ bất đối xứng tài nguyên (`X_unseen → Y_pretrained`, như `Amis → Mandarin` và `Rhade → Vietnamese`), việc cưỡng chế **tính tương thích Bayes không cần LM biên** thông qua đẳng thức tỷ số chéo 4 điểm (*4-point log cross-ratio* $M^{\to}_\theta(i,j) = M^{\leftarrow}_\theta(i,j)$) giữa chiều thuận $P_\theta(y|x)$ và chiều nghịch $P_\theta(x|y)$ trong cùng một mạng có:
1. Vượt qua rõ rệt cả huấn luyện hai chiều thuần (`BiT`: $\mathcal{L}_{\text{CE}}^{\to} + \alpha_{\text{rev}}\mathcal{L}_{\text{CE}}^{\leftarrow}$) lẫn đối chứng mạnh nhất hiện có (`SYM-SG` / EDWARD-NMT Eq. 13),
2. Vượt qua **đối chứng bác bỏ ghép cặp tráo (`CR_SHUFFLED`: ghép $M^{\to}(i,j)$ với $M^{\leftarrow}(k,l)$ của cặp không liên quan)** và **đối chứng contrastive một chiều (`UNI_CONTRAST`)** để chứng minh phần lợi ích đến từ đúng liên kết tương thích Bayes theo từng cặp `(i, j)` chứ không chỉ từ `BiT`, thêm compute hay đẩy negative thuần túy,
3. Giảm ảo giác phụ thuộc target-LM prior và giữ được cải thiện nhất quán trên cả ngôn ngữ chắp dính OOV cao (`Amis`: 16.6% OOV) lẫn ngôn ngữ OOV thấp (`Rhade`: 1.0% OOV)?

### 2.2. Bằng chứng thực nghiệm khởi điểm (`[ĐO ĐƯỢC]`)
1. **Cohort A100 9-run (`Amis → Mandarin`, seed 42, validation 576 câu, `beam 1` FP32 — [GOC_PILOT_A100_REPORT.md](../tasks/reports/GOC_PILOT_A100_REPORT.md)):**
   - `BASELINE_S42`: Val BLEU **`17.5264`**, chrF0 `16.7632`, Val Loss `2.8913` (min Val Loss `2.5978` tại epoch 6 khi BLEU mới `13.37`).
   - `W_B125_S42` (R-Drop $\beta=1.25$): Val BLEU **`17.7069`**, chrF0 `17.0817`.
   - `GOC_B125_S42` / `GOC_B250_S42`: `18.0944` / `17.8097` $\implies$ **NO-GO** vì thua `GOC_SYM` (`-0.31` đến `-0.70` BLEU) và tụt recall ký tự Hán hiếm (`76/74` vs `83`).
   - `GOC_SYM_B250_S42` (`0.5/0.5` stop-grad KL = EDWARD-NMT Eq. 13): Val BLEU **`18.5143`** (`+0.99` vs Baseline, `+0.81` vs `W_B125`), chrF0 **`17.5405`**, Rare CJK `83/789 (10.52%)` — **đối chứng mạnh nhất hiện có** trên cohort A100, nhưng không thể lấy làm novelty độc lập vì trùng công thức EDWARD-NMT Eq. 13.
   - `GOC_INV_B250_S42`: Val BLEU sụp đổ xuống **`15.1466`** nhưng lại có **Val Loss thấp nhất toàn bảng (`2.4671`)** — chứng minh `[ĐO ĐƯỢC]` rằng teacher-forced Val Loss bị tách rời/nghịch tương quan với chất lượng sinh autoregressive trên Amis.
2. **Chẩn đoán lỗi xuyên ngôn ngữ (`analyze_a100_val_errors.py` & `check_datasets_oov.py`):**
   - `amis_mandarin` (4.600/576): **16.6% source word OOV** (84.2% câu validation chứa $\ge 1$ từ OOV); BLEU của `GOC_SYM_B250` rơi tự do từ `29.73` (`0% OOV`, $n=90$) $\to$ `18.50` (`0–25% OOV`, $n=386$) $\to$ `7.93` (`>25% OOV`, $n=100$); 38.0% câu đạt $<10$ BLEU với nhiều ảo giác trôi theo target LM tiếng Trung ngay cả trên từ đã gặp (ví dụ `miasik ko loma'` = quét nhà $\to$ `將家燒掉` = đốt nhà).
   - `rhade_vietnamese` (15.092/1.000): chỉ **1.0% source word OOV** (10.0% câu validation). Do đó, một mẹo chỉ xử lý OOV hình thái sẽ không chuyển tốt sang Ê Đê — phương pháp ACL cần đánh vào **nguyên lý bất đối xứng phân phối nguồn–đích tổng quát** (`X_unseen → Y_pretrained`).
3. **Lịch sử 100+ run của project ([RESEARCH_INSIGHTS.md](RESEARCH_INSIGHTS.md)):** Các phương pháp can thiệp kiến trúc / hidden-state (`CLRR`, `LSR`, `DRB`, `AlignmentResidual`, `Middle`) đều không vượt được regularization trực tiếp trên không gian xác suất đầu ra, trong khi các biến thể tinh chỉnh quanh R-Drop (`AC`, `PLC`, `CMC`, `VCO`, `GOC`) đều vướng overlap novelty hoặc thua `SYM-SG`.

### 2.3. Định lý, Công thức, Gradient & Chi phí

#### A. Định lý triệt tiêu phân phối biên (Arnold & Press, 1989 $\to$ Marginal-Free DSL)
Trong Dual Supervised Learning (Xia et al., ICML 2017, Eq. 3), điều kiện tương thích Bayes cho một cặp song ngữ $(x_i, y_i)$ là:
$$\log P(x_i) + \log P_\theta(y_i \mid x_i) = \log P(y_i) + \log P_\theta(x_i \mid y_i).$$
Bởi $P(x_i)$ và $P(y_i)$ chưa biết, Xia et al. phải ước lượng bằng các mô hình ngôn ngữ ngoài $\hat{P}(x), \hat{P}(y)$ — nguồn gây bias nặng trên ngôn ngữ cực ít tài nguyên chưa có trong pretraining (`Amis`, `Rhade`).

Xét **hai cặp song ngữ bất kỳ** $(x_i, y_i)$ và $(x_j, y_j)$ thuộc cùng một minibatch. Với mọi phân phối đồng thời hợp lệ $P(x, y) > 0$, định nghĩa **Forward Log Cross-Ratio** và **Reverse Log Cross-Ratio**:
$$M^{\to}_\theta(i, j) \triangleq \log P_\theta(y_i \mid x_i) - \log P_\theta(y_i \mid x_j) - \log P_\theta(y_j \mid x_i) + \log P_\theta(y_j \mid x_j),$$
$$M^{\leftarrow}_\theta(i, j) \triangleq \log P_\theta(x_i \mid y_i) - \log P_\theta(x_i \mid y_j) - \log P_\theta(x_j \mid y_i) + \log P_\theta(x_j \mid y_j),$$
trong đó $\log P_\theta(y \mid x) = \sum_{t=1}^{|y|} \log P_\theta(y_t \mid \langle\text{tgt\_lang}\rangle, y_{<t}, x)$ là **tổng log-xác suất chuỗi chính xác** trên toàn bộ token hợp lệ (`labels != -100`, bao gồm `</s>`, không áp label smoothing trong thành phần `M`).

**Định lý 1 (Tương thích không cần biên — Arnold & Press, 1989):**
1. **Triệt tiêu biên tuyệt đối:** Với mọi phân phối đồng thời $P(x,y) > 0$, cả 4 số hạng biên $\log P(x_i), \log P(x_j), \log P(y_i), \log P(y_j)$ tự triệt tiêu hoàn toàn:
   $$M^{\to}(i, j) = \log\frac{P(x_i, y_i) P(x_j, y_j)}{P(x_j, y_i) P(x_i, y_j)} = M^{\leftarrow}(i, j).$$
2. **Điều kiện cần và đủ:** $M^{\to}_\theta(i, j) = M^{\leftarrow}_\theta(i, j)$ với mọi cặp $(i, j)$ trên không gian chuỗi hữu hạn $\mathcal{X}\times\mathcal{Y}$ là điều kiện **cần và đủ** để tồn tại duy nhất (sai khác hằng số chuẩn hóa) hai phân phối biên $P(x), P(y)$ và phân phối đồng thời $P(x, y) = P(x)P_\theta(y|x) = P(y)P_\theta(x|y)$.
3. **Chuẩn hóa độ dài đối xứng giữ nguyên đẳng thức:** Vì tỷ lệ số subword $|x|/|y|$ khác 1 đáng kể (ví dụ Amis dài gấp ~2× chữ Hán), chia riêng từng chiều cho $|y|$ và $|x|$ sẽ **phá vỡ** đẳng thức Bayes (lệch tới `0.608` trong kiểm FP64). Thay vào đó, chia **cả hai vế cho cùng mẫu số đối xứng**:
   $$L_{ij} = \frac{1}{2}\big(|x_i| + |x_j| + |y_i| + |y_j|\big) \implies \tilde{M}^{\to}_\theta(i,j) = \frac{M^{\to}_\theta(i,j)}{L_{ij}}, \quad \tilde{M}^{\leftarrow}_\theta(i,j) = \frac{M^{\leftarrow}_\theta(i,j)}{L_{ij}}.$$
   Khi đó $\tilde{M}^{\to}(i,j) = \tilde{M}^{\leftarrow}(i,j)$ vẫn đúng **chính xác tuyệt đối** (`8.33e-17` FP64) và có thang độ lớn $O(1)$ nats/token.

#### B. Hàm mục tiêu khóa trước (`CR-Dual v1` = V1 `CR_HUBER_FULL` + Paired-RNG + Warmup)
Trong một mô hình many-to-many chia sẻ duy nhất (`mBART-50`: `tl_XX ↔ zh_CN` cho Amis, `id_ID ↔ vi_VN` cho Ê Đê):
1. **Ghép cặp rời nhau đồng độ dài (Length-Bucket Involution $\pi$):** Sắp xếp các mẫu trong microbatch (cỡ chẵn $B$) theo độ dài chuỗi nguồn/đích rồi ghép từng cặp liền kề $(0,1), (2,3), \dots, (B-2, B-1)$ để tạo phép hoán vị nghịch đảo rời nhau $\pi$ ($\pi(\pi(i)) = i, \pi(i) \neq i$). Việc ghép đồng độ dài giúp giảm thiểu padding thừa và tránh làm nhiễu cross-attention trên câu lệch chiều dài quá lớn.
2. **Tái sử dụng encoder & hoán vị mask padding chính xác:**
   - Chạy encoder đúng **2 lượt**: $H_X = \text{Enc}(X, m_X)$ và $H_Y = \text{Enc}(Y, m_Y)$.
   - Khi chạy lượt decoder off-diagonal, truyền **đồng thời** $H_X[\pi]$ và **source padding mask hoán vị $m_X[\pi]$** (tương tự $H_Y[\pi], m_Y[\pi]$) để cross-attention không bao giờ chú ý vào token `<pad>` của câu ghép.
3. **Paired-RNG Shared Dropout (Khắc phục nhiễu dropout 4 lượt từ Vòng A §2C):**
   - Lưu/khôi phục trạng thái RNG dropout (bên ngoài ranh giới checkpoint với `preserve_rng_state=True`, kiểm định bằng unit test gradient checkpointing chính xác) sao cho `Dec(H_X, m_X, Y)` và `Dec(H_X[pi], m_X[pi], Y)` dùng **cùng một mặt nạ dropout $\epsilon_f$**, và `Dec(H_Y, m_Y, X)` cùng `Dec(H_Y[pi], m_Y[pi], X)` dùng **cùng một mặt nạ dropout $\epsilon_r$**.
   - Nhờ đó, mỗi hiệu $\log P_{\epsilon_f}(y_i|x_i) - \log P_{\epsilon_f}(y_i|x_j)$ đo độ nhạy nguồn trên **cùng một mạng con tất định**, triệt tiêu phương sai dropout nội chiều trong $M^{\to}$ và $M^{\leftarrow}$.
4. **Hàm mất mát tổng tại bước $s$:**
   $$\mathcal{L}_{\text{CR-Dual}}(s) = \mathcal{L}_{\text{CE}}^{\to}(X \to Y) + \alpha_{\text{rev}}\,\mathcal{L}_{\text{CE}}^{\leftarrow}(Y \to X) + \lambda(s)\cdot \frac{1}{B/2}\sum_{\{i,j\}\in\pi} \psi_\delta\!\left(\frac{M^{\to}_\theta(i,j) - M^{\leftarrow}_\theta(i,j)}{L_{ij}}\right),$$
   trong đó:
   - $\mathcal{L}_{\text{CE}}^{\to}, \mathcal{L}_{\text{CE}}^{\leftarrow}$ là cross-entropy chuẩn có label smoothing `0.1` (khớp recipe A100 hiện tại);
   - $\alpha_{\text{rev}} = 0.5$ (mặc định an toàn cho cặp bất đối xứng `X_unseen → Y_pretrained` sau khi kiểm Ding et al. 2021 BiT; kiểm thêm độ nhạy với $\alpha_{\text{rev}}$ ở diagnostic);
   - $\psi_\delta$ là hàm Huber với $\delta = 1.0$ (bậc hai quanh 0, tuyến tính ở đuôi để chống nổ gradient);
   - $\lambda(s) = \lambda_{\max}\cdot \min(1, s / S_{\text{warm}})$ tăng tuyến tính từ $0 \to \lambda_{\max}$ trong **20% bước đầu** (epochs 0–4) để chiều nghịch $Y \to X$ ổn định trước khi ép ghép nối cross-ratio; giá trị $\lambda_{\max}$ được hiệu chỉnh định lượng ở Bước 0 sao cho $\lambda_{\max}\cdot \|\nabla_\theta \mathcal{L}_{\text{CR}}\|_2 / \|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2 \in [0.05, 0.15]$.

#### C. Chi phí tính toán & Inference
- **Tham số thêm:** `0`. **LM ngoài / dữ liệu ngoài:** `0`. **Inference overhead:** `0` (dịch chiều $X \to Y$ bằng backbone `mBART-50` gốc, `beam 1` val / `beam 4` test).
- **Lượt tính toán khi train:** **2 lượt Encoder + 4 lượt Decoder** (so với `BASELINE`: 1 Enc + 1 Dec; `R-Drop`/`SYM-SG`: 2 Enc + 2 Dec; `BiT`: 2 Enc + 2 Dec) $\implies$ ước tính ~**1.5× FLOPs** và ~**1.4–1.6× wall time** của `SYM-SG` (khoảng ~1.5–1.9 giờ/run trên A100 40 GB với BF16 + gradient checkpointing; xác minh thực tế ở GPU 2-step gate).

---

### 2.4. Kiểm chứng toán học FP64 (`[ĐO ĐƯỢC — Mô phỏng FP64]`)
Tất cả 4 nhóm kiểm tra lưu tại `outputs/experiments/round3_method_design/MATH_CHECKS_CRDUAL*.json`:
1. **Test 1 — Exact Marginal Cancellation & Length-Ratio Invariance (`MATH_CHECKS_CRDUAL.json`):**
   - Sai số tại phân phối đúng $P^*(x,y)$ với độ dài chênh lệch mạnh (`|x| ∈ [10,45]`, `|y| ∈ [5,22]`): $\max|M^{\to}-M^{\leftarrow}| = 2.66\times 10^{-15}$, chuẩn hóa đối xứng $L_{ij}$: $8.33\times 10^{-17}$.
   - Chuẩn hóa độ dài riêng hai chiều (bị lỗi): lệch tới **`0.608`**.
   - Xia et al. (2017) DSL với $\hat{P}(x)$ đều (mô phỏng low-resource LM misspecified): lệch trung bình **`0.621`** (tối đa `1.149`) ngay tại cặp mô hình dịch **hoàn hảo**!
2. **Test 2 — Nhạy cảm với Target-LM Hallucination (`MATH_CHECKS_CRDUAL.json`):**
   - Trộn target marginal $P^*(y)$ vào chiều thuận $P_\theta(y|x) = (1-\alpha)P^*(y|x) + \alpha P^*(y)$ với $\alpha \in \{0, 0.1, 0.25, 0.5, 0.8\}$: **100%** các cặp cho $M^{\to} < M^{\leftarrow}$, chênh lệch trung bình giảm đơn điệu `0.00 → -0.92 → -1.82 → -3.13 → -5.08`.
3. **Test 3 — Bẫy Overfitting trên bảng tham số tự do (`MATH_CHECKS_CRDUAL.json` — Phản ví dụ trung thực):**
   - Trên bảng lookup overfit nặng (`576` tham số cho `60` mẫu, 400 bước), `CR_HUBER_FULL` đẩy `min_offdiag_logp` xuống sâu hơn (`-23.85` vs `BiT -22.66`), và falsifier đảo dấu (`M_fwd = -M_rev`) lại vô tình giảm KL vì đóng vai trò confidence penalty. Phát hiện này buộc chúng tôi **không dùng Wrong-Sign làm falsifier chính**, mà thay bằng **`CR_SHUFFLED`** (Control 1)!
4. **Test 4 — Tổng quát hóa Neural 5-seed trên dữ liệu test chưa thấy (`label_smoothing=0.1` + pretrained target prior bias, `MATH_CHECKS_CRDUAL_GEN.json` & `MATH_CHECKS_CRDUAL_CONTROLS.json`):**

| Phương pháp (5 seeds trung bình, 2.000 mẫu test/seed) | Test NLL thuận ↓ | Test Acc thuận ↑ | Test NLL nghịch ↓ | Held-out CR Viol ↓ |
|---|---:|---:|---:|---:|
| `BASELINE_UNI` (CE thuận) | `1.6108` | `64.37%` | `3.2321` | `5.235` |
| `BIT_CE` (CE thuận + nghịch) | `1.5908` | `65.38%` | `1.4539` | `2.775` |
| `DSL_MISSPECIFIED_LM` (Xia 2017, $\hat{P}(x)$ lệch) | `1.5823` | `65.76%` | `1.4313` | `2.719` |
| `CTRL3_UNI_CONTRASTIVE_MARGIN` (SCD-style 1 chiều) | `1.6052` | `66.71%` | `3.2510` | `5.674` |
| `CR_PAIR_JS_FULL` (V3 — KILLED) | `1.5774` | `65.74%` | `1.4330` | `2.654` |
| `CR_HUBER_NO_OFFDIAG_GRAD` (V2 — KILLED) | `1.4349` | `67.89%` | `1.2801` | `2.494` |
| `CTRL1_SHUFFLED_REV_PAIR` ($M^{\to}(i,j) \approx M^{\leftarrow}(k,l)$) | `1.3044` | `68.27%` | `1.1920` | `1.196` |
| **`CR_HUBER_FULL` (`CR-Dual` V1 — ADMITTED)** | **`1.2148`** | **`69.95%`** | **`1.1079`** | `1.392` |
| `CR_WRONG_SIGN_FALSIFIER` ($M^{\to} = -M^{\leftarrow}$) | `1.7914` | `47.50%` | `1.7857` | `0.322` |

- **Ghi chú trung thực cực kỳ quan trọng từ bảng trên:** Trong bài kiểm tra tổng hợp này, `CTRL1_SHUFFLED_REV_PAIR` đã thu được **~76% mức giảm NLL** từ `BiT → CR_DUAL` (`1.591 → 1.304`) nhờ hiệu ứng truyền thang tương phản toàn cục giữa hai chiều, trong khi **liên kết đúng cặp `(i, j)` theo Arnold & Press** đóng góp thêm **~24% mức giảm NLL (`1.304 → 1.215`) và `+1.68%` accuracy (`68.27% → 69.95%`)**. Đây chính là lý do **`CR_SHUFFLED` là đối chứng bác bỏ bắt buộc số 1** trên `mBART-50` thật!

---

### 2.5. Audit Novelty & Đối chiếu Prior gần (đã mở đọc nguồn gốc + kiểm NotebookLM)

| Prior gần | Nguồn gốc đã đối chiếu | Phần gần / giao thoa | Khác biệt cơ chế của `CR-Dual` | Mức độ |
|---|---|---|---|---|
| **Dual Supervised Learning (DSL)** (Xia et al., ICML 2017) | `1707.00415` (§2.2, Eq. 3) | Ép tương thích Bayes giữa hai chiều dịch $P(y|x), P(x|y)$ | DSL tính trên từng cặp $(x_i, y_i)$ nên **bắt buộc ước lượng external LMs $\hat{P}(x), \hat{P}(y)$**. `CR-Dual` dùng 4-point cross-ratio để **triệt tiêu tuyệt đối cả hai biên**, 0 LM ngoài. | **GẦN** |
| **BiT** (Ding et al., EMNLP 2021) | `2021.emnlp-main.263.pdf` (§2.2 Eqs. 3–5, §3.3 Table 6) | Train đồng thời cả hai chiều $X\to Y$ và $Y\to X$ trong một mạng | BiT thuần là joint data-level MLE (`CE_fwd + CE_rev`), **không có bất kỳ ràng buộc tương thích xác suất hay off-diagonal nào** giữa hai chiều. | **GẦN** |
| **REDER** (Zheng et al., NeurIPS 2021) | `zheng2021_duplex_reder.pdf` (§2.3) | Mạng hai chiều khả nghịch có cycle consistency | Kiến trúc non-autoregressive RevNet, dùng cycle reconstruction giải mã; không phải đẳng thức cross-ratio teacher-forced trên autoregressive Transformer. | **GẦN** |
| **Mirror-Generative NMT** (Zheng et al., ICLR 2020) & **Dual Transfer Learning** (Wang et al., AAAI 2018) | OpenReview `HkxQRTNYPH`; AAAI 2018 `11999` | Kết hợp hai chiều dịch với LM biên / importance sampling cho dữ liệu đơn ngữ | Vẫn mô hình hóa rõ ràng LM biên $P(x), P(y)$ (hoặc xấp xỉ tổng bằng importance sampling) cho semi-supervised monolingual data; không dùng 4-point cross-ratio triệt tiêu biên trên dữ liệu song ngữ. | **GẦN** |
| **SCD** (Sennrich et al., EACL 2024) & **NPMI** (Lee et al., EMNLP 2022) | `2309.07098`; `2022.emnlp-main.547.pdf` | Đối sánh với nguồn nhiễu $x_{\text{rand}}$ hoặc hiệu chuẩn MI | SCD là thuật toán giải mã (tốn 2× chi phí inference, cần chỉnh tay $\lambda$); NPMI dùng external LM. Cả hai không có đẳng thức đại số đóng hai chiều. | **GẦN** |
| **R-Drop** (NeurIPS 2021) & **EDWARD-NMT** (2026) | `2106.14448v2.pdf`; `2026.findings-aacl.149.pdf` (§3.4 Eq. 13) | Nhất quán đầu ra trong lúc train | Chỉ là nhất quán **nội chiều** giữa hai dropout subnetworks trên cùng hướng $X\to Y$; hoàn toàn trực giao với tương thích Bayes **xuyên chiều** của `CR-Dual`. | **TRỰC GIAO** |
| **4-point Cross-Ratio Marginal-Free Dual Consistency trong NMT** | Arnold & Press (JASA 1989); Besag (1974) (thống kê không gian / bảng ngẫu nhiên) | Đẳng thức tương thích phân phối điều kiện cổ điển | **Chưa từng thấy chuyển hóa vào huấn luyện seq2seq/NMT để triệt tiêu LM biên của DSL** trong phạm vi toàn bộ truy vấn Web và 9 paper gốc đã kiểm. | **`CHƯA THẤY`** |

- **Kết luận novelty trung thực:** **`CHƯA THẤY` exact match trong NLP/NMT trong phạm vi đã đọc** *(không phải chứng nhận novelty toàn cầu)*. Đóng góp ở mức ACL Main dựa trên 3 trụ cột rõ ràng: (1) định lý Marginal-Free Cross-Ratio DSL giải quyết tận gốc bài toán misspecified external LMs của Xia et al. (2017) cho ngôn ngữ tài nguyên cực thấp; (2) cơ chế Paired-RNG + chuẩn hóa độ dài đối xứng $L_{ij}$ giữ nguyên đẳng thức Bayes trên Transformer autoregressive với 0 inference overhead; (3) bằng chứng thực nghiệm tách sạch phần lợi ích tương thích Bayes khỏi `BiT`, `CR_SHUFFLED`, `UNI_CONTRAST` và `SYM-SG` trên cả `Amis → Mandarin` và `Rhade → Vietnamese`.

---

### 2.6. Chẩn đoán Bước 0 Bắt Buộc (TRƯỚC KHI MỞ PILOT 4 RUN)

Để tránh lặp lại việc chạy 9 run tốn kém trước khi kiểm giả thuyết cốt lõi, **Bước 0** thực hiện trên `mBART-50` thật (`Amis ↔ Mandarin`, A100 BF16, ~30–40 phút GPU) bằng chính checkpoint **`BIT_CE_S42`** (Run #1 của pilot — **không lãng phí run nào!**):
1. Lưu nhẹ 3 snapshot trạng thái `BIT_CE_S42` tại **epochs 1, 4, 10** (chỉ đo diagnostic rồi xóa snapshot trung gian).
2. Trên cùng 100 cặp validation đồng độ dài `(i, j)` (và 100 cặp train):
   - **Đo 1 (Phân phối thực tế của $\tilde{M}^{\to}$ vs $\tilde{M}^{\leftarrow}$):** Xác nhận dấu và độ lớn của $\tilde{M}^{\to}(i,j) - \tilde{M}^{\leftarrow}(i,j)$ trên `mBART-50` thật (kiểm tra giả thuyết $\tilde{M}^{\to} < \tilde{M}^{\leftarrow}$ và tốc độ ổn định của chiều nghịch `zh_CN → tl_XX` ở epoch 4).
   - **Đo 2 (Hiệu quả Paired-RNG Shared Dropout):** Đo $\operatorname{Var}(\tilde{M}^{\to})$ qua 8 lần lặp dropout độc lập vs Paired-RNG trên cùng 100 cặp. **Ngưỡng chấp nhận:** Paired-RNG phải giảm phương sai $\ge 70\%$ và vượt qua bài test **checkpointed vs uncheckpointed gradient exactness (`< 1e-5` FP32)**.
   - **Đo 3 (Hiệu chỉnh định lượng $\lambda_{\max}$ không đoán mò):** Đo tỷ số chuẩn gradient toàn mạng $\rho_{\text{grad}} = \|\nabla_\theta \mathcal{L}_{\text{CR}}\|_2 / \|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2$ tại epoch 4; khóa duy nhất một giá trị $\lambda_{\max}$ sao cho $\lambda_{\max}\cdot \rho_{\text{grad}} \in [0.05, 0.15]$.
   - **Đo 4 (Cổng sớm cho `BIT_CE_S42`):** Quan sát xem `BIT_CE_S42` có bị phá hủy nghiêm trọng trên `Amis → Mandarin` (tụt $> 1.5$ BLEU dưới `BASELINE_S42 = 17.53`) không. Nếu `BIT_CE_S42` sụp đổ hoàn toàn ở $\alpha_{\text{rev}}=0.5$, phải chẩn đoán ngay nguyên nhân decoder capacity cannibalization trước khi chạy 3 run còn lại.

---

### 2.7. Bảng Pilot Khóa Trước (A100, Amis Seed 42, Validation-Only)

Giữ nguyên 100% protocol lock [amis_a100_bleu.json](../../configs/protocols/amis_a100_bleu.json) để **tái sử dụng trực tiếp** các run A100 vừa hoàn tất (`mBART-50` rev `e30b6cb8…`, BF16 + TF32 train, SDPA, reentrant checkpointing, LR `5e-5`, `B32×GA4`, max 20 epochs, ES10 theo **validation BLEU `beam 1` FP32 `zh`**, không mở test).

| Nhánh | Trạng thái | Công thức / Cấu hình | Vai trò khoa học |
|---|---|---|---|
| `BASELINE_S42` | **REUSE** (`985f9fc4…`) | $\mathcal{L}_{\text{CE}}^{\to}$ | Baseline 1 chiều (`Val BLEU = 17.5264`, `chrF0 = 16.7632`) |
| `W_B125_S42` | **REUSE** (`4df107c0…`) | R-Drop $\beta=1.25$ | R-Drop chuẩn (`Val BLEU = 17.7069`, `chrF0 = 17.0817`) |
| `GOC_SYM_B250_S42` | **REUSE** (`47f6f925…`) | EDWARD-NMT Eq. 13 (`SYM-SG`, $\beta=2.50$) | Đối chứng mạnh nhất hiện có (`Val BLEU = 18.5143`, `chrF0 = 17.5405`) |
| **`BIT_CE_S42`** | **NEW #1** | $\mathcal{L}_{\text{CE}}^{\to} + \alpha_{\text{rev}}\mathcal{L}_{\text{CE}}^{\leftarrow}$ ($\lambda=0$, $\alpha_{\text{rev}}=0.5$) | **Đối chứng BiT (Ding 2021)**: tách hiệu ứng huấn luyện hai chiều thuần + làm checkpoint Bước 0 |
| **`CR_DUAL_S42`** | **NEW #2** | `BIT_CE` + $\lambda(s)\cdot\text{Huber}(\frac{M^{\to}(i,j)-M^{\leftarrow}(i,j)}{L_{ij}})$ + Paired-RNG | **Ứng viên chính (`CR-Dual` V1)** |
| **`CR_SHUFFLED_S42`** | **NEW #3** | Y hệt `CR_DUAL_S42` (cùng compute/VRAM/Huber/Paired-RNG), nhưng ghép $M^{\to}(i,j)$ với $M^{\leftarrow}(k,l)$ tráo cặp $(k,l)\neq(i,j)$ | **Falsifier #1 (Bắt buộc)**: tách tương thích Bayes đúng cặp khỏi hiệu ứng truyền thang tương phản toàn cục / thêm compute |
| **`UNI_CONTRAST_S42`** | **NEW #4** | $\mathcal{L}_{\text{CE}}^{\to} + \lambda(s)\cdot\text{ReLU}(\gamma - \frac{\log P(y_i|x_i)-\log P(y_i|x_j)}{|y_i|})$ (1 chiều, dùng chung $H_X[\pi]$) | **Falsifier #2 (Bắt buộc)**: tách ghép nối hai chiều khỏi đẩy negative off-diagonal kiểu SCD một chiều |

- **Ngân sách tài nguyên A100 40 GB:** Chỉ **4 runs mới** (ước tính ~1.0–1.8 giờ/run, tổng **~6 giờ GPU A100**; peak VRAM dự kiến ~16–22 GiB < 40 GiB; backup HF & kiểm SHA trước khi dọn weights).
- **Cổng GO / NO-GO khóa trước (trên Validation BLEU `beam 1` FP32 `zh`, chrF0 & Rare CJK Recall — *đính chính gợi ý NLL của NotebookLM vì Val Loss đã đo là anti-correlated với BLEU trên Amis*):**
  1. **Technical Gate (Bắt buộc trước train):** CPU unit tests (exact Arnold-Press invariance `< 1e-12` FP64, symmetric $L_{ij}$, padding mask permutation `m_X[pi]` zero-grad on `-100`, Paired-RNG determinism, **checkpointed vs uncheckpointed gradient exactness `< 1e-5`**, save/reload, native inference) PASS + GPU 2-step optimizer/resume gate PASS.
  2. **Gate 1 — Mechanism vs BiT:** `CR_DUAL_S42 - BIT_CE_S42 >= +0.50` Val BLEU VÀ `chrF0(CR_DUAL) > chrF0(BIT_CE)`.
  3. **Gate 2 — Falsifier #1 (Exact Pairing):** `CR_DUAL_S42 - CR_SHUFFLED_S42 >= +0.30` Val BLEU (kèm kiểm tra paired sentence bootstrap 10.000 mẫu trên 576 câu validation + held-out CR violation giảm). *Nếu `CR_DUAL <= CR_SHUFFLED`: **NO-GO ngay**, rút lại tuyên bố tương thích Bayes Arnold-Press!*
  4. **Gate 3 — Falsifier #2 (Dual Coupling vs 1-Way Contrast):** `CR_DUAL_S42 - UNI_CONTRAST_S42 >= +0.40` Val BLEU.
  5. **Gate 4 — Quality Bar vs Strongest Comparator (`SYM-SG`):**
     - **FULL GO (Đủ điều kiện mở rộng sang bảng chính ACL):** Đạt Gates 1–3 **VÀ** `CR_DUAL_S42 >= 18.5143` Val BLEU (`>= GOC_SYM_B250_S42`, tức $\ge +0.99$ trên `BASELINE_S42`) cùng recall ký tự Hán hiếm $\ge 10.0\%$.
     - **CONDITIONAL ORTHOGONALITY GO:** Đạt Gates 1–3 VÀ `CR_DUAL_S42 >= 18.25` (vượt `BASELINE` $\ge +0.72$, vượt `W_B125` $\ge +0.54$), mở đúng **1 run kiểm tra tính trực giao** `CR_DUAL + SYM_SG` (phải đạt $\ge 18.85$, vượt `SYM_SG` ít nhất `+0.34` BLEU).
     - **NO-GO:** Hụt Gates 1–3 hoặc hụt cả hai mức Gate 4 $\implies$ **DỪNG**, lưu toàn bộ số âm vào registry, không mở test, không sweep hyperparameter ngẫu hứng.

### 2.8. Lộ trình lên bài ACL Main nếu `CR-Dual` đạt GO
1. **Xác nhận đa seed trên A100 (`amis_mandarin`, seeds 42, 43, 44):** Chạy đủ 3 seeds cho `BASELINE`, `SYM_SG`, `BIT_CE`, `CR_DUAL`, `CR_SHUFFLED`, `UNI_CONTRAST` (và `CR_DUAL + SYM_SG` nếu trực giao), đánh giá test `beam 4` với paired bootstrap 10.000 mẫu + hiệu chỉnh Holm-Bonferroni.
2. **Kiểm chứng chuyển ngôn ngữ chưa dùng để chọn method (`rhade_vietnamese`, `id_ID ↔ vi_VN`, 15.092/1.000/1.000, 3 seeds):** Chứng minh `CR-Dual` hoạt động ngay cả khi OOV nguồn chỉ `1.0%`.
3. **Kiểm chứng xuyên backbone (`NLLB-200-distilled-600M`):** Xác nhận cơ chế không phụ thuộc riêng `mBART-50`.
4. **Phân tích cơ chế cho bài báo:** So sánh trực tiếp với Xia et al. (2017) DSL (dùng external KenLM/GPT-2 làm $\hat{P}(x), \hat{P}(y)$ trên Amis để đo lỗi bias thực tế của DSL!), đo tỷ lệ giảm hallucination nguồn–đích, và phân tách minh bạch phần gain từ negative contrast (`BiT → CR_SHUFFLED`) so với phần gain từ exact Arnold-Press pairing (`CR_SHUFFLED → CR_DUAL`).

---

## 3. Nhật ký Truy vấn Web & Kiểm định NotebookLM (Mục 4 Pha 3–4)

### 3.1. Truy vấn Web đã thực hiện (17 truy vấn trên 2 cụm)
- **Cụm 1 — `SGR` & `AF-SCD` (7 truy vấn, đã loại):** exposure bias self-recovery NMT (`He 2021`, `He 2024`), prediction difference regularization (`Guo 2022`), NPMI/CBMI adaptive NMT (`Lee 2022`), source-contrastive decoding training (`Sennrich 2024`), multi-view subword regularization (`Wang 2021`).
- **Cụm 2 — `CR-Dual` / Marginal-Free Cross-Ratio Dual Learning (10 truy vấn):**
  1. `"Dual Supervised Learning" marginal language model eliminate OR marginal-free OR "cross-ratio" OR "odds ratio" NMT`
  2. `"Mirror-Generative Neural Machine Translation" ICLR 2020 Zheng arxiv pdf`
  3. `"Mirror-Generative Neural Machine Translation" site:arxiv.org`
  4. `"bidirectional" "neural machine translation" "low-resource" consistency OR "dual" 2023 OR 2024 OR 2025 ACL OR EMNLP`
  5. `"conditional compatibility" "cross-ratio" OR "odds ratio" "language model" OR "translation" OR "seq2seq"`
  6. `"log P(y_i|x_i)" "log P(y_i|x_j)" "log P(x_i|y_i)" "log P(x_i|y_j)" OR "cross-ratio" "dual"`
  7. `"Arnold" "Press" "compatible conditional" "neural" OR "machine translation" OR "dual learning" OR "generative"`
  8. `"Wang" "Dual Transfer Learning" "ICML 2019" OR "dual supervised learning" "without language model" OR "language model free"`
  9. `"Dual Transfer Learning for Neural Machine Translation with Marginal Distribution Regularization" Wang AAAI 2018 pdf`
  10. `"dual" "in-batch negative" OR "contrastive" "bidirectional" "machine translation" "log P(y|x)"`

### 3.2. NotebookLM (`bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`, conv `8ec5ca51-f2be-4cca-a640-c5b4035a99d5`)
- **Paper gốc mới thêm vào Notebook CLRR (đã tra trùng trước):**
  - `2ee085d3-8094-498e-8e39-d504e984562c` — Guo et al. (ACL 2022, PD-R)
  - `188ed5ca-9cf7-4ae9-8e79-7e5dbfc4e212` — He et al. (EMNLP 2021, Exposure Bias vs Self-Recovery)
  - `c8f69f73-8249-4906-a0f3-2d55f6041305` — Lee et al. (EMNLP 2022, NPMI Adaptive Training)
  - `98c942b1-7cd7-4ce6-8920-b1d2bdb7264b` — Xia et al. (ICML 2017, Dual Supervised Learning)
  - `3446c54f-eef5-42db-8ffa-5fc747012f96` — Ding et al. (EMNLP 2021, BiT)
  *(Đã xóa nguồn OpenReview bị chặn bot `e3c509b8-d162-41f1-a80a-713beec7ea55`.)*
- **Paper gốc có sẵn dùng đối chiếu trực tiếp trong `--source-ids`:** Zheng et al. 2021 REDER (`dda6d8b4…`), Sennrich et al. 2024 SCD (`67b0c77f…`), Voita et al. 2021 (`0f4ec420…`), SIWCon (`6f2c4813…`), R-Drop (`3f48a525…`), EDWARD-NMT 2026 (`7f03b33a…`).
- **Các điểm đã đính chính từ phản hồi NotebookLM (không chấp nhận mù quáng):**
  1. **Sửa lỗi đếm lượt tính toán Vòng A:** NotebookLM đếm nhầm thành "8 lượt decoder (4× BiT)"; đã chứng minh và được NotebookLM xác nhận lại ở Vòng B là **2 Encoder + 4 Decoder** (~1.5× R-Drop/SYM-SG, ~1.7–2× BiT) nhờ phép hoán vị rời nhau $\pi$.
  2. **Sửa gợi ý lịch warmup Vòng A §3:** NotebookLM đề xuất train thuần chiều thuận trước rồi mới bật đồng thời $\alpha_{\text{rev}}$ và $\lambda_{\text{CR}}$ (dẫn tới chiều nghịch chưa học khi CR bật); đã hiệu chỉnh đối chiếu Ding et al. (2021 BiT) thành train song song `CE_fwd + 0.5*CE_rev` từ bước 1 và chỉ tăng tuyến tính $\lambda(s): 0 \to \lambda_{\max}$ trong 20% bước đầu.
  3. **Sửa gợi ý đánh giá Vòng B (`beam 5` và `Delta NLL <= -0.15`):** Giữ nguyên khóa cohort A100 (`beam 1` FP32 validation selector) để tái sử dụng hợp lệ `BASELINE_S42 / W_B125_S42 / GOC_SYM_B250_S42`, và bỏ ràng buộc `Delta NLL <= -0.15` vì thực nghiệm 9-run A100 đã chứng minh Val Loss anti-correlated với BLEU trên Amis (`GOC_INV_B250` có Val Loss thấp nhất `2.4671` nhưng BLEU thấp nhất `15.1466`).
