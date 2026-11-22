# Thiết kế & Kiểm định phương pháp Vòng 4 — `CoCC` (*Compositional Context-Invariance Consistency*)

Ngày: **22/11/2026**.  
Trạng thái: **DESIGN & AUDIT ONLY** — chưa viết code training production, chưa tạo config experiment, chưa chạy GPU gate, chưa train và chưa mở test set.  
Mọi kết quả thực nghiệm A100 trích từ [CR_DUAL_PILOT_A100_REPORT.md](../tasks/reports/CR_DUAL_PILOT_A100_REPORT.md), [cr_dual_pilot_a100_summary.json](../tasks/reports/cr_dual_pilot_a100_summary.json), [GOC_PILOT_A100_REPORT.md](../tasks/reports/GOC_PILOT_A100_REPORT.md), [goc_pilot_a100_summary.json](../tasks/reports/goc_pilot_a100_summary.json) và [RESEARCH_INSIGHTS.md](RESEARCH_INSIGHTS.md). Toàn bộ bằng chứng toán học FP64, probe trên `mBART-50` đóng băng và thống kê subword pairing lưu tại [MATH_CHECKS_COCC.json](../../outputs/experiments/round4_method_design/MATH_CHECKS_COCC.json); nhật ký hai vòng phản biện NotebookLM lưu tại [NOTEBOOKLM_ROUND4_COCC_A.json](../../outputs/experiments/round4_method_design/NOTEBOOKLM_ROUND4_COCC_A.json) và [NOTEBOOKLM_ROUND4_COCC_B.json](../../outputs/experiments/round4_method_design/NOTEBOOKLM_ROUND4_COCC_B.json).

---

## 1. Tổng kết 5 Quy luật Thực nghiệm (>120 runs) & Các hướng đã loại ở Vòng 4

Sau 2 đợt pilot A100 gần nhất (`GOC` 9 runs + `CR-Dual` 4 runs, cùng protocol khóa [amis_a100_bleu.json](../../configs/protocols/amis_a100_bleu.json)) và toàn bộ lịch sử >100 runs trên RTX 6000, có **5 quy luật thực nghiệm `[ĐO ĐƯỢC]`** bắt buộc phải chi phối thiết kế Vòng 4:

1. **Quy luật 1 — Bắt buộc giữ chiều thuận 100% (`X → Y`):** Huấn luyện hai chiều kiểu BiT (`BIT_CE_S42 = CE_fwd + 0.5*CE_rev`) làm Val BLEU rớt thẳng từ `17.5264` (`BASELINE_S42`) xuống `16.4574` (**`-1.0690`** BLEU) ngay cả khi chưa thêm penalty nào, do decoder `mBART-50` phải chia dung lượng học sinh ngôn ngữ `Amis` chưa có trong pretraining (*decoder capacity cannibalization*).
2. **Quy luật 2 — Regularization trực tiếp trên phân phối xác suất đầu ra thắng can thiệp kiến trúc / hidden-state:** Toàn bộ các nhánh can thiệp hidden-state / rewiring (`CLRR`, `LSR`, `DRB`, `AlignmentResidual`, `Middle Alignment`) đều không vượt được regularization trực tiếp trên phân phối predictive ở đầu ra.
3. **Quy luật 3 — Chỉ dùng dữ liệu song ngữ sạch, tránh đẩy negative off-diagonal theo cặp câu:** Cả đối chứng đẩy negative một chiều (`UNI_CONTRAST_S42 = 16.7299`, `-0.7965` dưới Baseline) lẫn đối chứng hai chiều (`CR_DUAL_S42 = 16.1977`, `-1.3287`; `CR_SHUFFLED_S42 = 16.9163`, `-0.6101`) đều gây hại rõ rệt trên `mBART-50`.
4. **Quy luật 4 — Stop-gradient linear probability matching (`0.5 * (p1 - p2)`) hiệu quả hơn hẳn R-Drop và trọng số bất đối xứng:** Loại bỏ thành phần Reverse-KL Jacobian $0.5\beta p_v(\log(p_v/p_{\bar{v}}) - \text{KL})$ nâng `GOC_SYM_B250_S42` lên **`18.5143`** BLEU (`+0.9879` trên `BASELINE_S42`, `+1.1341` trên `W_B250_S42 = 17.3802`), đồng thời thắng luôn hai biến thể bất đối xứng `GOC` (`18.0944` / `17.8097`) và `GOC_INV` (`16.2213` / `15.1466`). Tuy nhiên bản thân `SYM-SG` trùng dạng với **EDWARD-NMT (2026) Eq. 13**, nên chỉ giữ làm **đối chứng A100 mạnh nhất hiện có (`18.5143`)**.
5. **Quy luật 5 — Lỗ hổng còn sót lại ngay trên `GOC_SYM_B250_S42` (`[ĐO ĐƯỢC]` từ chẩn đoán predictions A100):**
   - **Tụt Brevity Penalty (thiếu độ dài câu dịch):** `GOC_SYM_B250_S42` có `BP = 0.9384` (`sys/ref = 6216/6611` ký tự CJK), ngắn hơn rõ rệt so với `BASELINE_S42` (`BP = 0.9591`, `6346/6611`) và `W_B125_S42` (`BP = 0.9493`, `6284/6611`), dù precision n-gram cao nhất (`51.0/25.1/13.8/8.6`). Consistency chỉ qua dropout trên câu đơn cô lập làm mô hình thận trọng quá mức và dễ nuốt nội dung.
   - **Sụp đổ khi tỷ lệ từ OOV nguồn cao & ảo giác theo target LM:** Nhóm câu `>25% OOV nguồn` (`n=100`) của `GOC_SYM_B250` tụt xuống `7.93` BLEU (thấp hơn cả `BASELINE = 8.37`); `219/576` (`38.0%`) câu đạt `< 10` sentence BLEU với nhiều lỗi bỏ từ quen hoặc bịa câu tiếng Trung trôi theo prior (ví dụ các hàng `40`, `220`, `393`, `481` trong bộ chẩn đoán validation). Theo độ dài câu nguồn, nhóm `long_src >= 13 từ` (`n=7`, cỡ mẫu nhỏ) chỉ đạt `12.70` so với `18.82` ở `short_src <= 6 từ` (`n=270`).
   - **Giới hạn cấu trúc của consistency tại vị trí cố định:** Mọi biến thể R-Drop / `SYM-SG` chỉ khớp hai lượt dropout trên cùng một câu đơn đứng một mình tại cùng hệ tọa độ vị trí cố định, chưa bao giờ ép phân phối đầu ra bất biến trước **nhiễu ngữ cảnh chéo tương lai (`x_j`)**, **lịch sử tiền tố quá khứ (`y_i`)** hay **dịch chuyển chỉ số vị trí dài**.

### Sàng lọc hướng đi Vòng 4 qua tìm kiếm tài liệu (19 truy vấn Web + lịch sử dự án)

| Hướng đã xét | Bằng chứng tài liệu / dự án | Quyết định Vòng 4 |
|---|---|---|
| **Sharpness-Aware (`Functional-SAM`, `KL-SAM`, `GA-SAM`)** | Cụm công trình Optimization/NLP đã công bố rộng rãi (`Functional-SAM`, `KL-SAM` ICLR/OpenReview, `GA-SAM` cho NMT Transformers); rủi ro trùng cao nếu áp dụng lên NMT. | **LOẠI (Đông đúc)** |
| **Representation Self-Distillation / Layer Pruning (`Xu et al., ACL/Findings 2025`)** | Đã xuất bản công thức representation/rank-correlation self-distillation cho NMT ít tài nguyên (IWSLT); không giải quyết trực tiếp bất biến ngữ cảnh. | **LOẠI (Đông đúc)** |
| **EMA / Mean Teacher NMT** | Mẫu hình chuẩn đã phổ biến rộng làm bộ ổn định tối ưu hóa / teacher động; không tạo đóng góp cơ chế mới ở mức ACL Main. | **LOẠI (Yếu novelty)** |
| **DKD / Rest-consistency (`decoupled_consistency`) / GOC / PLC / CMC / VCO** | Đã triển khai và kiểm chứng thất bại trước `W` / `SYM-SG` trong chính repo CLRR. | **LOẠI (Đã kiểm)** |
| **Bidirectional (`BiT` / `CR-Dual`) & Off-diagonal Contrast (`UNI_CONTRAST`)** | Thua `-0.61` đến `-1.33` BLEU dưới `BASELINE_S42` trên A100 (`NO_GO` Vòng 3). | **ĐÓNG VĨNH VIỄN** |
| **`CoCC` (*Compositional Context-Invariance Consistency*)** | Khớp phân phối xác suất từng token giữa lượt câu đơn cô lập và lượt ghép cặp nguyên bản `S1` bằng stop-gradient (`CHƯA THẤY` trong 19 truy vấn + 8 paper gốc NotebookLM). | **GIỮ & KIỂM ĐỊNH (`ADMIT`)** |

---

## 2. Candidate Card Khóa Trước — `CoCC` (*Compositional Context-Invariance Consistency*)

### 2.1. Câu hỏi nghiên cứu (RQ)
**RQ:** Khi fine-tuning một chiều (`X → Y`) mô hình dịch đa ngữ tiền huấn luyện (`mBART-50` / `NLLB-200`) trong điều kiện ít tài nguyên, việc cưỡng chế **tính bất biến phân phối dự đoán theo từng token giữa ngữ cảnh câu đơn cô lập và ngữ cảnh ghép cặp nguyên bản có nhiễu tương lai + tiền tố quá khứ (`CoCC`, Layout S1)** — với **cùng ngân sách 2 lượt forward và tổng số token như `SYM-SG`**, 0 tham số thêm, 0 dữ liệu/LM ngoài và 0 inference overhead — có:
1. Vượt qua rõ rệt cả **tăng cường dữ liệu ghép câu thuần (`CONCAT_CE`, Nguyen et al. IWSLT 2021 `RAND`)** lẫn **việc chạy chồng `SYM-SG` nội-view không căn chỉnh xuyên-view (`CONCAT_SYMSG`, ngân sách 4 lượt)** để chứng minh lợi ích đến từ chính căn chỉnh bất biến `iso ↔ comp` theo token,
2. Vượt qua đối chứng câu đơn mạnh nhất hiện có **`GOC_SYM_B250_S42` (`18.5143` Val BLEU)** đồng thời phục hồi/giữ hiệu chuẩn độ dài & kết thúc `[EOS]` (`LR ∈ [0.93, 1.05]`),
3. Giữ được phần lợi ích ngay cả khi tắt dropout trên hai lượt KL (`COCC_PURE_CTX`: $\epsilon_1 = \epsilon_2 = 0$) ở Stage-2 và khái quát hóa tốt trên cả hai cặp ngôn ngữ có hình thái/chấm câu trái ngược (`Amis → Mandarin` với 16.77% OOV & 100% dấu kết câu nguồn vs `Rhade → Vietnamese` với 1.00% OOV & chỉ 54.67% dấu kết câu nguồn)?

### 2.2. Bằng chứng thực nghiệm khởi điểm (`[ĐO ĐƯỢC]`)

| Dữ liệu / Hệ thống (`Amis → Mandarin`, A100 Seed 42, Validation 576 câu, `beam 1` FP32) | Val BLEU ↑ | Val chrF0 ↑ | Val chrF++ ↑ | BP (`sys/ref`) | Ghi chú cơ chế |
|---|---:|---:|---:|---:|---|
| `BASELINE_S42` (`baseline`, `985f9fc4`) | `17.5264` | `16.7632` | `14.5382` | `0.9591` (`6346/6611`) | Tụt mạnh ở `high_OOV >25%` (`8.37`) và `long_src >=13` (`7.63`) |
| `W_B125_S42` (`rdrop-consistency-v1`, `4df107c0`) | `17.7069` | `17.0817` | `14.8852` | `0.9493` (`6284/6611`) | Reverse-KL Jacobian hạn chế mức tăng trước `SYM-SG` |
| `W_B250_S42` (`rdrop-consistency-v1`, `98fa0c08`) | `17.3802` | `16.8022` | `14.5969` | `0.9556` (`6324/6611`) | Tăng $\beta$ làm tụt `-0.33` BLEU dưới `W_B125` |
| **`GOC_SYM_B250_S42`** (`symmetric-stopgrad`, `47f6f925`) | **`18.5143`** | **`17.5405`** | **`15.3266`** | **`0.9384`** (`6216/6611`) | **Mạnh nhất A100**, `Rare CJK = 83/789 (10.52%)`, nhưng **BP tụt thấp nhất** & `>25% OOV` còn `7.93` |
| `BIT_CE_S42` (`bidirectional-ce-v1`, `55c51fddd989`) | `16.4574` | `15.8076` | `13.6946` | — | Decoder capacity cannibalization (`-1.0690`) |
| `CR_DUAL_S42` (`cross-ratio-dual-v1`, `6045bdddb08d`) | `16.1977` | `15.8281` | `13.7608` | — | Thua cả falsifier `CR_SHUFFLED` (`-0.7185`) |
| `CR_SHUFFLED_S42` (`cross-ratio-shuffled-v1`, `6609bb24a96c`) | `16.9163` | `16.4246` | `14.2490` | — | Falsifier xác nhận nhiễu từ chiều nghịch |
| `UNI_CONTRAST_S42` (`unidirectional-contrast-v1`, `1334bdb1552d`) | `16.7299` | `16.1342` | `13.9757` | — | Đẩy off-diagonal sequence negatives một chiều gây hại (`-0.7965`) |

- **Đối chiếu đặc tính hai tập dữ liệu (`[ĐO ĐƯỢC]` trong phiên này trên `train.csv`/`validation.csv`):**
  - `amis_mandarin` (`4,600` train / `576` val): nguồn dài trung bình `6.77` từ (`max = 19`), `100.0%` câu nguồn và `97.96%` câu đích train có dấu kết câu (`.!?。！？`); tập validation có **`16.77%` token OOV chuẩn hóa** (`17.64%` theo script `analyze_a100_val_errors.py`) và **`84.38%` (`486/576`) câu chứa $\ge 1$ từ OOV** (`0/576` câu nguồn trùng khít train).
  - `rhade_vietnamese` (`15,092` train / `1,000` val): nguồn dài trung bình `13.23` từ (`max = 197`), chỉ **`54.67%` câu nguồn train** (`75.13%` câu đích train) có dấu kết câu (pha trộn mục từ từ điển, câu văn và câu Kinh Thánh), và tỷ lệ OOV chuẩn hóa trên validation chỉ là **`1.00%` token / `10.10%` câu**. Vì vậy, thiết kế Vòng 4 tuyệt đối không được giả định câu nào cũng kết thúc bằng dấu chấm câu.

### 2.3. Công thức chính xác (`CoCC v1`), Layout S1, Boundary Mask `y_{j,1}`, Gradient & Chi phí

#### A. Ghép cặp rời nhau có bảo vệ độ dài (*Length-Guarded Random Pairing*)
Trong mỗi microbatch gồm $B$ cặp câu song ngữ $\{\,(\mathbf{x}_k, \mathbf{y}_k)\,\}_{k=0}^{B-1}$ đã chuẩn hóa theo layout gốc (`mBART-50` / `NLLB-200`):
$$\mathbf{x}_k = [\langle\text{src\_lang}\rangle,\; x_{k,1},\dots,x_{k,m_k},\; \langle/\text{s}\rangle], \qquad \mathbf{y}_k = [\langle\text{tgt\_lang}\rangle,\; y_{k,1},\dots,y_{k,n_k},\; \langle/\text{s}\rangle].$$
Mỗi bước huấn luyện, phân hoạch ngẫu nhiên các mẫu trong microbatch thành tối đa $\lfloor B/2 \rfloor$ cặp rời nhau $r = (i, j)$ (hoặc ghép greedy ngẫu nhiên trên hoán vị batch) thỏa mãn **ràng buộc không cắt cụt**:
$$|\mathbf{x}_i| + |\mathbf{x}_j| \le L_{\max,\text{src}} \quad \text{VÀ} \quad |\mathbf{y}_i| + |\mathbf{y}_j| \le L_{\max,\text{tgt}} \qquad (L_{\max} = 256).$$
Các mẫu lẻ hoặc vượt $L_{\max}$ khi ghép vẫn tham gia `CE_iso` ở Pass 1 như bình thường và tự động không tham gia tính `KL` ghép cặp (fallback an toàn, 0 mất mát dữ liệu). Trên dữ liệu thật, độ phủ ghép cặp đạt **`100.0%` trên Amis** và **`99.9417%` trên Rhade** (Mục 2.4).

#### B. Bố cục ghép nguyên bản **Layout S1** (Sửa tận gốc lỗi mất `[EOS]` của Layout S0)
Khác với bản nháp `S0` (cắt bỏ `</s>` giữa hai câu dẫn tới triệt tiêu `[EOS]` ở câu 1 và bùng nổ xác suất `[EOS]` sai ở ranh giới Rhade), **Layout S1** ghép nguyên vẹn hai chuỗi gốc ở mức token ID (tránh mọi sai lệch re-tokenization ở ranh giới chuỗi BPE/SentencePiece):
$$\mathbf{x}_{\text{comp}, r} = \mathbf{x}_i \parallel \mathbf{x}_j = \big[\langle\text{src\_lang}\rangle,\; \mathbf{x}_{i,\text{body}},\; \langle/\text{s}\rangle,\; \langle\text{src\_lang}\rangle,\; \mathbf{x}_{j,\text{body}},\; \langle/\text{s}\rangle\big],$$
$$\tilde{\mathbf{y}}_{\text{comp}, r} = \mathbf{y}_i \parallel \mathbf{y}_j = \big[\langle\text{tgt\_lang}\rangle,\; \mathbf{y}_{i,\text{body}},\; \langle/\text{s}\rangle,\; \langle\text{tgt\_lang}\rangle,\; \mathbf{y}_{j,\text{body}},\; \langle/\text{s}\rangle\big].$$

**Bất biến triển khai bắt buộc (`decoder_input_ids_comp` trước khi mask `-100`):**
- Gọi `shift_tokens_right(raw_y_comp, ...)` **TRƯỚC KHI** đặt mặt nạ `-100` lên `labels_comp` (lưu ý tương thích đa backbone: `mBART-50` dùng `shift_tokens_right(ids, pad_token_id)` đặt `decoder_start_token_id = eos_token_id`; `NLLB-200 / M2M100` dùng 3 tham số `shift_tokens_right(ids, pad_token_id, decoder_start_token_id)` — bắt buộc khẳng định trong unit test đa backbone).
- Nhờ thứ tự này, `decoder_input_ids_comp` có dạng $[\langle/\text{s}\rangle, \langle\text{tgt\_lang}\rangle, \mathbf{y}_{i,\text{body}}, \langle/\text{s}\rangle, \langle\text{tgt\_lang}\rangle, \mathbf{y}_{j,\text{body}}]$, nghĩa là ngay trước khi dự đoán $y_{j,1}$, decoder nhìn đúng tiền tố $[\dots, \langle/\text{s}\rangle, \langle\text{tgt\_lang}\rangle]$ khớp 1-1 với tiền tố khởi đầu câu đơn của $\mathbf{y}_j$ (không bao giờ chèn `<pad>` giả).
- Sau đó, đặt **`labels_comp[r, len_i] = -100`** cho đúng vị trí target $\langle\text{tgt\_lang}\rangle$ nội tại đầu Segment 2 (và `-100` trên padding cuối chuỗi). Cả $\langle/\text{s}\rangle_i$ (tại `u = len_i - 1`) lẫn $\langle/\text{s}\rangle_j$ (tại `u = len_i + len_j - 1`) đều **giữ nguyên giám sát trong $\mathcal{L}_{\text{CE, comp}}$**, hoàn toàn triệt tiêu nguy cơ *EOS suppression*.

#### C. Bản đồ căn chỉnh 1-1 $\mathcal{M}$ với Boundary Mask `mask_y_j1 = True`
Ký hiệu $l_i = |\mathbf{y}_i|$ và $l_j = |\mathbf{y}_j|$ (đều gồm $[\langle\text{tgt\_lang}\rangle, \text{body}, \langle/\text{s}\rangle]$). Trên mỗi cặp ghép $r = (i, j)$, tập cặp vị trí căn chỉnh $\mathcal{M}_r$ gồm đúng hai phân đoạn trực giao:
1. **Segment 1 (*Encoder & Cross-Attention Future-Distractor Invariance*):**
   $$(i,\; t) \longleftrightarrow (r,\; t) \qquad \forall\; t \in \{1,\; 2,\; \dots,\; l_i - 1\}.$$
   *(Bao gồm toàn bộ token nội dung $y_{i,1..n_i}$ VÀ token kết thúc $\langle/\text{s}\rangle_i$ của câu 1. Vì encoder self-attention là hai chiều trên $\mathbf{x}_i \parallel \mathbf{x}_j$, ép $p_{\text{comp}} \approx p_{\text{iso}}$ ở Segment 1 trực tiếp buộc bộ mã hóa & cross-attention miễn nhiễm với nhiễu từ nguồn tương lai $\mathbf{x}_j$, kể cả giữ xác suất kết thúc đúng lúc tại $\langle/\text{s}\rangle_i$.)*
2. **Segment 2 (*Past-Context & Positional-Shift Invariance*, có `mask_y_j1 = True`):**
   $$(j,\; t) \longleftrightarrow (r,\; l_i + t) \qquad \forall\; t \in \{2,\; 3,\; \dots,\; l_j - 1\}.$$
   *(Token đầu câu 2 $y_{j,1}$ tại $t=1 \leftrightarrow u=l_i+1$ **vẫn học qua $\mathcal{L}_{\text{CE, comp}}$**, nhưng được **loại khỏi KL** để không phạt mô hình ngôn ngữ decoder nối tiếp hợp lệ khi 45% câu nguồn trên ngữ liệu như Rhade thiếu dấu chấm câu! Từ $t \ge 2$ trở đi gồm cả nội dung $y_{j,2..n_j}$ và $\langle/\text{s}\rangle_j$, ngữ cảnh decoder bị chi phối bởi lịch sử nội câu của chính $\mathbf{y}_j$, cho phép kiểm tra thuần túy tính bất biến trước lịch sử câu quá khứ $(\mathbf{x}_i, \mathbf{y}_i)$ và dịch chuyển chỉ số vị trí dài $l_i + t$.)*

#### D. Hàm mục tiêu (`COCC_SYM` vs `COCC_ISO_TEACHER`), Gradient & Chi phí
- **Pass 1 (Isolated, batch $B$):** $\mathbf{z}_{\text{iso}} = f_\theta(\mathbf{x}_k, \mathbf{y}_{k,<t}; \epsilon_1)$, chuẩn xác suất $p_{\text{iso}} = \text{softmax}(\mathbf{z}_{\text{iso}})$ trên FP32, tính $\mathcal{L}_{\text{CE, iso}}$.
- **Pass 2 (Composite, batch $\le \lfloor B/2 \rfloor$, dài $\approx 2\times$):** $\mathbf{z}_{\text{comp}} = f_\theta(\mathbf{x}_{\text{comp}}, \mathbf{y}_{\text{comp},<u}; \epsilon_2)$, tính $\mathcal{L}_{\text{CE, comp}}$ (bỏ `-100`).
- **Hàm mất mát chính khóa trước (`COCC_SYM`, V1):**
  $$\mathcal{L}_{\text{CoCC-SYM}} = \frac{1}{2}\big(\mathcal{L}_{\text{CE, iso}} + \mathcal{L}_{\text{CE, comp}}\big) + \frac{\beta}{2|\mathcal{M}|}\sum_{(k,t)\leftrightarrow(r,u)\in\mathcal{M}}\Big[\mathbb{D}_{\text{KL}}\big(\text{sg}(p_{\text{iso}}(k,t)) \parallel p_{\text{comp}}(r,u)\big) + \mathbb{D}_{\text{KL}}\big(\text{sg}(p_{\text{comp}}(r,u)) \parallel p_{\text{iso}}(k,t)\big)\Big].$$
- **Biến thể đối chứng có hướng (`COCC_ISO_TEACHER`, V2):**
  $$\mathcal{L}_{\text{CoCC-V2}} = \frac{1}{2}\big(\mathcal{L}_{\text{CE, iso}} + \mathcal{L}_{\text{CE, comp}}\big) + \frac{\beta}{|\mathcal{M}|}\sum_{(k,t)\leftrightarrow(r,u)\in\mathcal{M}}\mathbb{D}_{\text{KL}}\big(\text{sg}(p_{\text{iso}}(k,t)) \parallel p_{\text{comp}}(r,u)\big).$$
- **Gradient giải tích trên cặp căn chỉnh $(k,t)\leftrightarrow(r,u)\in\mathcal{M}$:**
  $$\frac{\partial \mathcal{L}_{\text{KL}}^{\text{SYM}}}{\partial \mathbf{z}_{\text{iso}}(k,t)} = \frac{\beta}{2|\mathcal{M}|}\big(p_{\text{iso}}(k,t) - p_{\text{comp}}(r,u)\big), \qquad \frac{\partial \mathcal{L}_{\text{KL}}^{\text{SYM}}}{\partial \mathbf{z}_{\text{comp}}(r,u)} = \frac{\beta}{2|\mathcal{M}|}\big(p_{\text{comp}}(r,u) - p_{\text{iso}}(k,t)\big),$$
  trong khi dưới `COCC_ISO_TEACHER` thì $\frac{\partial \mathcal{L}_{\text{KL}}^{\text{V2}}}{\partial \mathbf{z}_{\text{iso}}} \equiv \mathbf{0}$ (cô lập triệt để vai trò cập nhật hai chiều trên $\mathbf{z}_{\text{iso}}$). Toàn bộ phép giảm KL tính trên FP32 theo từng đoạn `chunk_size=32` dùng `torch.utils.checkpoint` giống [goc_loss](../../src/methods/gold_ordered_consistency/v1/model.py).
- **Chi phí & Inference:**
  - **Tổng số token / bước:** Pass 1 có $B\cdot L$ token; Pass 2 có $\le (B/2)\cdot(2L) = B\cdot L$ token $\implies$ tổng $\le 2BL$ token, **tương đương đúng `1.0×` ngân sách forward/backward của `SYM-SG` và `R-Drop`**!
  - **VRAM ước tính:** ~15–18 GiB (dưới 40 GiB A100; khuyến nghị chạy kèm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
  - **Tham số thêm / Dữ liệu ngoài / Inference overhead:** **Tất cả bằng `0`** (suy luận chạy đúng `mBART-50`/`NLLB-200` gốc trên từng câu đơn).

---

### 2.4. Kiểm chứng toán học FP64 & Đo trực tiếp trên `mBART-50` đóng băng (`[ĐO ĐƯỢC]`)

Toàn bộ kết quả dưới đây trích nguyên vẹn từ [MATH_CHECKS_COCC.json](../../outputs/experiments/round4_method_design/MATH_CHECKS_COCC.json) đo trực tiếp trong phiên thiết kế:

#### A. Kiểm chứng độ chính xác Gradient & Mask trong FP64 (`task-1096`)

| Chế độ kiểm tra (`len_i=5, len_j=4`, 1 pad `-100`, 1 internal `<tgt_lang>` `-100`) | Số token KL $|\mathcal{M}|$ | $\max|\nabla z_{\text{iso}} - \nabla_{\text{exact}}|$ | $\max|\nabla z_{\text{comp}} - \nabla_{\text{exact}}|$ | Grad `<pad>` & `<tgt_lang>` | Grad tại $y_{j,1}$ |
|---|---:|---:|---:|---:|---:|
| `sym_sg` (`mask_y_j1=False`) | `7` | `6.94e-18` | `6.94e-18` | `0.0` | `0.01091` |
| **`sym_sg` (`mask_y_j1=True` — KHÓA)** | `6` | **`6.94e-18`** | **`6.94e-18`** | **`0.0`** | **`0.0`** |
| `iso_teacher` (`mask_y_j1=False`) | `7` | `0.0` (`grad is None`) | `1.39e-17` | `0.0` | `0.02182` |
| **`iso_teacher` (`mask_y_j1=True` — ABAL)** | `6` | **`0.0` (`grad is None`)** | **`1.39e-17`** | **`0.0`** | **`0.0`** |

- Đồng thời đã kiểm chữ ký `shift_tokens_right` trong `transformers 4.57.6`: `MBart` nhận `(input_ids, pad_token_id)`, còn `M2M100/NLLB` nhận `(input_ids, pad_token_id, decoder_start_token_id)`.

#### B. Đo trực tiếp `S0` (bỏ EOS giữa) vs `S1` (giữ nguyên EOS + `<lang>`) trên `mBART-50` đóng băng (`task-1079`, 16 cặp validation / nhóm)

| Ngữ liệu | Bố cục | `s1_body` KL `i→c` / `c→i` | `s1_eos` KL `i→c` | `s2_y1` KL `i→c` / `c→i` | `s2_int` KL `i→c` / `c→i` | Entropy `iso` / `comp` | `P_iso(eos)` vs `P_comp(eos)` *(vị trí check)* | `CE_s2` `iso` / `comp` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`amis_mandarin`** (100% nguồn có dấu câu) | `S0` | `0.3118` / `0.2832` | `null` | `1.1128` / `2.0817` | `0.5047` / `0.5351` | `6.4258` / `6.5496` | `0.0000` vs `0.0261` *(tại $y_{j,1}$)* | `5.2428` / `5.3224` |
| **`amis_mandarin`** | **`S1`** | **`0.2053` / `0.1925`** | **`1.4985`** | **`0.2413` / `0.2776`** | **`0.3913` / `0.4212`** | **`6.2035` / `6.3578`** | **`0.7593` vs `0.2016`** *(tại `eos_i`)* | `5.2428` / `4.8945` |
| **`rhade_unpunct`** (45% không chấm câu) | `S0` | `0.1525` / `0.1673` | `null` | `3.6757` / `5.7735` | `0.4908` / `0.6243` | `4.4482` / `4.3760` | `0.0001` vs **`0.1687`** *(va chạm EOS tại $y_{j,1}$)* | `5.2315` / **`5.9693`** |
| **`rhade_unpunct`** | **`S1`** | `0.1724` / `0.1839` | `0.7351` | **`2.2576` / `4.9574`** | **`0.5351` / `0.6912`** | **`4.4199` / `4.3506`** | `0.3089` vs `0.2490` *(tại `eos_i`)* | `5.2315` / **`5.2356`** |
| **`rhade_all`** (mẫu chung) | `S0` | `0.1415` / `0.1519` | `null` | `2.4078` / `4.0824` | `0.3354` / `0.3649` | `4.4998` / `4.5228` | `0.0001` vs **`0.2950`** | `4.8199` / `5.4732` |
| **`rhade_all`** | **`S1`** | `0.2226` / `0.2353` | `0.3360` | **`2.2299` / `5.5587`** | **`0.4405` / `0.4804`** | **`4.3172` / `4.1538`** | **`0.7531` vs `0.7091`** *(tại `eos_i`)* | `4.8199` / `5.2197` |

**4 kết luận rút trực tiếp từ dữ liệu thật `mBART-50` ở bảng trên:**
1. **`S1` giải quyết triệt để va chạm `[EOS]` tại biên $y_{j,1}$ và giữ nguyên hiệu chuẩn `[EOS]`:** Trên `Amis`, `S1` hạ `s2_y1` KL từ `1.1128 / 2.0817` xuống **`0.2413 / 0.2776`** (giảm **$7.5\times$** chiều `c2i`), đồng thời phơi bày đúng bệnh mất kết thúc ở Segment 1: `P_iso(eos) = 0.7593` rơi xuống `P_comp(eos) = 0.2016` (`s1_eos_kl = 1.4985`) khi có nguồn tương lai $\mathbf{x}_j$. Trên `Rhade`, `S0` làm `P_comp(eos)` tại $y_{j,1}$ vọt lên `0.1687–0.2950` và làm `ce_comp_s2` đội từ `5.2315 → 5.9693`; sang `S1`, `ce_comp_s2` tụt về đúng **`5.2356`** và xác suất `eos_i` phục hồi về **`0.7531` vs `0.7091`**.
2. **`mask_y_j1 = True` vẫn bắt buộc ngay cả dưới `S1`:** Trên `Rhade` không chấm câu, dù đã có `[eos, vi_VN]` làm vách ngăn, self-attention nhân quả tại $y_{j,1}$ vẫn đọc qua phần đuôi chưa dứt câu của $\mathbf{y}_i$, giữ `s2_y1_kl` cao gấp 4–7× phần thân (`2.2576 / 4.9574` vs `0.5351 / 0.6912`). Mask $y_{j,1}$ khỏi KL là thiết yếu cho độ an toàn đa ngôn ngữ.
3. **Bác bỏ thực nghiệm giả định Vòng A rằng $p_{\text{comp}}$ có entropy quá cao:** Entropy trung bình của `p_iso` và `p_comp` gần như tương đương (`6.2035` vs `6.3578` trên Amis; `4.3172` vs `4.1538` trên Rhade) và KL nội đoạn đối xứng chặt chẽ (`0.3913` vs `0.4212`; `0.4405` vs `0.4804`). Kết hợp việc `V2` có $\nabla_{z_{\text{iso}}}\mathcal{L}_{\text{KL}} \equiv 0$, giữ `COCC_SYM` làm phương án chính và `COCC_ISO_TEACHER` làm ablation là có căn cứ đo đạc vững chắc.

#### C. Độ phủ ghép cặp theo subword thực tế của `mBART-50` ở $B=32, L_{\max}=256$ (`task-1106`, 5 epochs mô phỏng)

| Tập Train | Số câu | Độ dài subword nguồn (Mean / p95 / p99 / Max) | Độ dài subword đích (Mean / p95 / p99 / Max) | Tỷ lệ ghép cặp `S1` `B=32` (Mean / Min) |
|---|---:|---|---|---|
| `amis_mandarin` | `4,600` | `15.81 / 23 / 28 / 39` | `11.22 / 16 / 20 / 30` | **`1.000000 / 1.000000`** |
| `rhade_vietnamese` | `15,092` | `34.66 / 103 / 135 / 256` | `21.19 / 60 / 78 / 241` | **`0.999417 / 0.999205`** |

---

### 2.5. Audit Novelty & Đối chiếu Prior gần (đã mở đọc nguồn gốc + kiểm NotebookLM)

| Prior gần | Nguồn gốc đã đối chiếu | Cơ chế của prior | Khác biệt cơ chế của `CoCC` | Mức độ |
|---|---|---|---|---|
| **Nguyen, Murray, & Chiang (IWSLT 2021)** | `2021.iwslt-1.33.pdf` (`a57b68a4…`, §2.1 `RAND`) | Ghép ngẫu nhiên `(x_i x_j, y_i y_j)` thành tập tăng cường và huấn luyện bằng **CE thuần** ($\beta=0$). | Không chạy song song hai lượt `iso` và `comp`, **không có căn chỉnh phân phối xác suất từng token** giữa câu đơn và câu ghép. (`CONCAT_CE` là đối chứng #1). | **GẦN** |
| **Lupo, Dinarelli, & Besacier (WMT 2022)** | `2022.wmt-1.77.pdf` (`fd9cf5b6…`, §3.1 Eq. 1) | *Context Discounting* giảm trọng số CE trên câu ngữ cảnh ($\text{CD}\cdot\mathcal{L}_{\text{ctx}} + \mathcal{L}_{\text{curr}}$) và dịch position embedding. | Chỉ chạy **một view ngữ cảnh**, không dùng câu đơn làm điểm neo phân phối và không đo bất biến cross-attention tương lai hay quá khứ. | **GẦN** |
| **Guo et al. (ACL 2022, PD-R)** | `2022.acl-long.528.pdf` (`2ee085d3…`, §2.2 Eq. 3, §4 Eq. 9) | *Prediction Difference Regularization*: gây nhiễu token tại chỗ (word-dropout/replace) trên **cùng một câu** và phạt sai khác dự đoán. | PD-R nhiễu token cục bộ trong câu đơn; `CoCC` giữ nguyên 100% câu sạch và kiểm tra **bất biến trước cấu trúc ghép cặp vĩ mô** (nhiễu $\mathbf{x}_j$ tương lai + lịch sử $\mathbf{y}_i$ quá khứ + dịch vị trí). | **GẦN** |
| **Wang et al. (NAACL 2021, MVR)** | `2021.naacl-main.40.pdf` (`e8c4cdd3…`, §4 Eq. 1) | Nhất quán xác suất giữa hai phân mảnh subword khác nhau của **cùng một câu nguồn**. | Không ghép hai câu song ngữ độc lập, không có căn chỉnh 1-1 trên tọa độ dịch chuyển giữa hai khung ngữ cảnh. | **GẦN** |
| **Liang et al. (NeurIPS 2021, R-Drop)** & **EDWARD-NMT (2026)** | `2106.14448v2.pdf` (`3f48a525…`); `2026.findings-aacl.149.pdf` (`7f03b33a…`, §3.4 Eq. 13) | Nhất quán hai dropout sub-models trên **cùng đầu vào giống hệt nhau** ($X \equiv X$) ở vị trí cố định. | `CoCC` cưỡng chế bất biến **xuyên độ dài và xuyên khung tọa độ ngữ cảnh** ($X_k$ cô lập vs $X_i \parallel X_j$ ghép cặp), gồm cả sửa lỗi mất `[EOS]` trước nguồn nhiễu tương lai $\mathbf{x}_j$. | **GẦN / BỔ TRỢ** |
| **Voita et al. (ACL 2021)** & **Sennrich et al. (EACL 2024, SCD)** | `2021.acl-long.91.pdf` (`0f4ec420…`); `arXiv:2309.07098` (`67b0c77f…`) | Phân tích đóng góp nguồn–đích bằng LRP / khử ảo giác lúc suy luận bằng source-contrastive decoding. | Voita là công cụ phân tích; SCD tốn 2× chi phí lúc giải mã; `CoCC` đóng gói cơ chế kháng nhiễu ngữ cảnh thành hàm train không tốn chi phí lúc test. | **LIÊN QUAN** |
| **Token-aligned Isolated-vs-Concatenated Distribution Invariance (`CoCC`)** | Tổng hợp toàn bộ 19 truy vấn Web + 8 paper gốc NotebookLM | Căn chỉnh phân phối stop-gradient 1-1 theo Layout S1 có `mask_y_j1` giữa lượt câu đơn và lượt ghép cặp rời nhau trên 2 view tổng token ngang bằng `R-Drop`. | **Chưa tìm thấy tiền lệ trùng trong phạm vi tài liệu đã đọc.** *(Ghi chú trung thực: không khẳng định chứng nhận novelty toàn cầu tuyệt đối.)* | **`CHƯA THẤY`** |

- **Ghi chú định danh cơ chế từ Vòng B (§1B):** Vì encoder Transformer dùng self-attention hai chiều trên $\mathbf{x}_i \parallel \mathbf{x}_j$, các biểu diễn ẩn $H(\mathbf{x}_i)$ đã bị ngữ cảnh hóa bởi câu nhiễu tương lai $\mathbf{x}_j$. Do đó trong bài báo phải định danh chính xác **Segment 1 là *Encoder & Cross-Attention Future-Distractor Invariance*** (không gọi là prefix invariance thuần túy), và **Segment 2 là *Past-Context & Positional-Shift Invariance***.

---

### 2.6. Chẩn đoán Bước 0 Bắt buộc trên A100 (0 optimizer steps, ~2 phút)

Trước khi chạy 4 run pilot dài, chạy một script chẩn đoán nhanh (0 bước tối ưu, không ghi đè weights) trên **cả pretrained `mBART-50` init lẫn checkpoint công khai đã train `GOC_SYM_B250_S42` (`47f6f925…`)**:
1. **Đo Context-Sensitivity trên mô hình đã hội tụ `GOC_SYM_B250_S42`:** Đo KL `S1` (`s1_body`, `s1_eos`, `s2_y1`, `s2_int`, `s2_eos`) và tỷ lệ mất `[EOS]` ở Segment 1 trên 128 cặp validation để kiểm xem checkpoint `SYM-SG` tốt nhất hiện tại có còn nhạy với nhiễu câu tương lai/quá khứ hay không (làm bằng chứng `0-step` cho động lực phương pháp).
2. **Hiệu chuẩn định lượng $\beta_{\text{CoCC}}$ tại khởi tạo `mBART-50`:** Trên 4 microbatch train cố định ($B=32$), đo tỷ số chuẩn gradient toàn backbone $\rho_{\text{grad}} = \|\nabla_\theta \mathcal{L}_{\text{KL}}\|_2 / \|\nabla_\theta \mathcal{L}_{\text{CE}}\|_2$ cho `ISO_SYM_SG` và `CoCC`:
   - Nếu $\rho_{\text{grad}}(\text{CoCC})$ nằm trong biên `1.5×` của $\rho_{\text{grad}}(\text{ISO\_SYM\_SG})$, **khóa cứng $\beta_{\text{CoCC}} = 2.50$** (khớp trực tiếp $\beta$ tốt nhất của `GOC_SYM_B250_S42`).
   - Ngược lại, **khóa duy nhất một giá trị $\beta_{\text{CoCC}} = \text{clip}(2.50 \cdot \rho_{\text{iso}} / \rho_{\text{CoCC}},\; 0.50,\; 2.50)$ làm tròn về bội `0.25`** **TRƯỚC bước tối ưu #1** (tuyệt đối không sweep $\beta$ sau khi thấy điểm validation!).

---

### 2.7. Bảng Pilot Khóa Trước (A100, Amis Seed 42, Validation-Only, Protocol `amis_a100_bleu.json`)

Giữ nguyên 100% protocol lock [amis_a100_bleu.json](../../configs/protocols/amis_a100_bleu.json) (`mBART-50` rev `e30b6cb8…`, A100 40GB BF16 + TF32 train, SDPA, reentrant gradient checkpointing, `LR = 5e-5`, `B32×GA4`, max 20 epochs, ES10 theo **validation BLEU `beam 1` FP32 `zh`**, không mở test):

| Nhánh | Trạng thái | Công thức / Cấu hình | Lượt forward / step | Vai trò khoa học |
|---|---|---|---:|---|
| `BASELINE_S42` | **REUSE** (`985f9fc4`) | $\mathcal{L}_{\text{CE, iso}}$ | `1×` | Baseline chuẩn (`Val BLEU = 17.5264`, `chrF0 = 16.7632`, `BP = 0.9591`) |
| `W_B125_S42` | **REUSE** (`4df107c0`) | R-Drop $\beta=1.25$ | `2×` | R-Drop chuẩn (`Val BLEU = 17.7069`, `chrF0 = 17.0817`, `BP = 0.9493`) |
| `GOC_SYM_B250_S42` | **REUSE** (`47f6f925`) | `ISO_SYM_SG` ($\beta=2.50$, EDWARD-NMT Eq. 13) | `2×` | **Đối chứng SOTA đơn câu** (`Val BLEU = 18.5143`, `chrF0 = 17.5405`, `BP = 0.9384`) |
| **`CONCAT_CE_S42`** | **NEW #1** | Layout S1, $0.5(\mathcal{L}_{\text{CE, iso}} + \mathcal{L}_{\text{CE, comp}})$, $\beta=0$ | `2×` (`2BL`) | **Falsifier #1 — Nguyen et al. (IWSLT 2021 `RAND`)**: tách hiệu ứng tăng cường dữ liệu ghép câu thuần khỏi KL consistency |
| **`CONCAT_SYMSG_S42`** | **NEW #2** | Layout S1, `CE` như trên + `0.5*SYM_SG(iso_1, iso_2) + 0.5*SYM_SG(comp_1, comp_2)`, **không căn chỉnh `iso ↔ comp`** | `4×` (`4BL`) *(hào phóng 2×)* | **Falsifier #2 — Chồng `SYM-SG` ngây thơ**: tách đóng góp của **căn chỉnh xuyên-view `iso ↔ comp`** khỏi việc chạy dropout `SYM-SG` nội-view trên câu ghép |
| **`COCC_SYM_S42`** | **NEW #3** | Layout S1 + `mask_y_j1=True` + **symmetric stop-grad `0.5/0.5`** ($\beta_{\text{CoCC}}$ khóa Bước 0) | `2×` (`2BL`) | **Ứng viên chính (`CoCC` V1)** — cùng ngân sách token/pass với `SYM-SG` và chỉ bằng `0.5×` lượt của `CONCAT_SYMSG`! |
| **`COCC_ISO_TEACHER_S42`** | **NEW #4** | Y hệt `COCC_SYM_S42`, đổi KL thành **một chiều** $\mathbb{D}_{\text{KL}}(\text{sg}(p_{\text{iso}})\parallel p_{\text{comp}})$ | `2×` (`2BL`) | **Ablation có hướng (V2)** — kiểm chứng có thể bác bỏ xem gradient trên $\mathbf{z}_{\text{iso}}$ có cần thiết không |

- **Ngân sách A100 40 GB Stage-1:** Đúng **4 runs mới** (~1.0–1.4 giờ/run, tổng **~5 giờ GPU A100**; khuyến nghị đặt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` khi chạy trên máy dùng chung GPU; backup HF & xác minh SHA trước khi dọn).
- **Cổng GO / NO-GO khóa trước (Validation BLEU `beam 1` FP32 `zh`, `chrF0`, `LR = sys_len/ref_len`, Rare CJK Recall):**
  1. **Technical Gate (Bắt buộc trước train):** Toàn bộ CPU unit tests PASS — gồm exact FP64 KL & gradient `< 1e-12`, `decoder_input_ids_comp` có `[..., </s>, <tgt_lang>]` trước `y_{j,1}` (không chèn `<pad>`), zero-grad tuyệt đối (`0.0`) tại `-100` padding, internal `<tgt_lang>`, và vị trí mask $y_{j,1}$, bất biến dưới `checkpoint(..., use_reentrant=True)` (`< 1e-5`), fallback khi không đủ cặp ghép, save/reload logits khớp, và native single-pass inference — CỘNG VỚI GPU 2-step optimizer/resume gate PASS cho cả 4 run mới.
  2. **Gate 1 (Falsifier tăng cường dữ liệu — bắt buộc):** `BLEU(COCC_SYM) - BLEU(CONCAT_CE) >= +0.50` VÀ `chrF0(COCC_SYM) > chrF0(CONCAT_CE)`.
  3. **Gate 2 (Falsifier chồng nội-view không căn chỉnh — bắt buộc):** `BLEU(COCC_SYM) - BLEU(CONCAT_SYMSG) >= +0.30` (thắng ngay cả khi `CONCAT_SYMSG` dùng gấp đôi số lượt forward!).
  4. **Gate 3 (Ngưỡng vượt SOTA câu đơn hiện tại — bắt buộc):** `BLEU(COCC_SYM) - BLEU(GOC_SYM_B250) >= +0.40` (tức **`COCC_SYM >= 18.91`** Val BLEU, hay `>= +1.39` trên `BASELINE_S42`) VÀ `chrF0(COCC_SYM) > 17.5405`.
  5. **Gate 4 (Hiệu chuẩn kết thúc `[EOS]` & Brevity):** `LR(COCC_SYM) ∈ [0.93, 1.05]` (`sys_len/ref_len`) — không bị sinh thừa (`LR > 1.05` do hỏng `[EOS]`) và không tụt hụt độ dài nặng hơn `GOC_SYM_B250` (`BP = 0.9384`).
  6. **Gate 5 (Kiểm định hướng stop-gradient):** Báo cáo minh bạch `COCC_SYM` vs `COCC_ISO_TEACHER`. Nếu `COCC_SYM` đạt Gates 1–4, giữ `COCC_SYM` làm phương pháp khóa. Nếu `COCC_ISO_TEACHER` đạt Gates 1–4 và vượt `COCC_SYM` $\ge +0.30$ BLEU, ghi nhận trung thực sự đảo chiều và bắt buộc kiểm tra trên seed xác nhận độc lập trước khi chốt công thức cho bảng chính.
  7. **Chính sách dừng NO-GO:** Hụt Gate 1, Gate 2, Gate 3 hoặc Gate 4 $\implies$ **`NO_GO` ngay lập tức**, lưu toàn bộ 4 runs vào registry, không mở test, không nhân seeds, không sweep hyperparameter ngẫu hứng.
- **Stage-2 Mechanism Isolation (CHỈ chạy nếu Stage-1 đạt GO trên Seed 42 — đúng 2 runs, không phải sweep):**
  1. **`COCC_PURE_CTX_S42`**: Đánh giá KL `iso ↔ comp` với dropout tạm tắt ở cả hai lượt (`model.eval()` trên nhánh KL giữ grad cho tham số, hoặc dropout mask $\epsilon_1 = \epsilon_2 = 0$), trong khi CE vẫn giữ dropout huấn luyện chuẩn — **tách sạch 100% đóng góp thuần của bất biến ngữ cảnh khỏi nhiễu dropout** (khắc phục lỗi kỹ thuật `fork_rng` của Vòng A!).
  2. **`COCC_S0_NOSEP_S42`**: Biến thể dùng bố cục `S0` bỏ `</s>` giữa câu + không mask `y_{j,1}` để xác nhận bằng chứng causal của thiết kế `S1 + mask_y_j1` qua training thật.

---

### 2.8. Lộ trình lên bài ACL Main nếu `CoCC` đạt GO

1. **Xác nhận đa seed trên A100 (`amis_mandarin`, seeds 42, 43, 44):** Chạy đủ 3 seeds ghép cặp cho `BASELINE`, `ISO_SYM_SG`, `CONCAT_CE`, `CONCAT_SYMSG`, `COCC_SYM` (và `COCC_PURE_CTX`), đánh giá test `beam 4` FP32 với paired bootstrap 10.000 mẫu + hiệu chỉnh Holm-Bonferroni (giữ ghi chú trung thực rằng `Amis` validation đã dùng trong quá trình phát triển nên kết quả `Amis` test là đồng bộ theo seed, còn `Rhade` là tập chưa đụng khi chọn method).
2. **Kiểm chứng chuyển ngôn ngữ thực thụ chưa dùng chọn method (`rhade_vietnamese`, `id_ID → vi_VN`, `15,092 / 1,000 / 1,000`, 3 seeds):** Chứng minh `CoCC` hoạt động mạnh ngay cả khi nguồn chỉ có `1.00%` OOV và `45.3%` câu nguồn không có dấu chấm câu nhờ `Layout S1 + mask_y_j1`.
3. **Kiểm chứng xuyên backbone (`NLLB-200-distilled-600M`):** Chứng minh tính tổng quát không phụ thuộc riêng `mBART-50` (dùng đúng helper `shift_tokens_right` 3-arg của M2M100/NLLB).
4. **Phân tích cơ chế cho bài báo ACL:** Phân tách định lượng đóng góp giữa Segment-1 *Future-Distractor Invariance* vs Segment-2 *Past-Context & Shift Invariance*, đường cong BLEU theo độ dài câu & tỷ lệ OOV, phục hồi Brevity Penalty & độ chính xác vị trí `[EOS]`, Recall ký tự hiếm CJK, và báo cáo chi phí Wall Time / FLOPs / Peak VRAM ngang hàng với `SYM-SG`.

---

## 3. Nhật ký Truy vấn Web & Kiểm định NotebookLM (Mục 4 Pha 3–4)

### 3.1. Truy vấn Web đã thực hiện (19 truy vấn trên 2 cụm)
- **Cụm 1 — Sàng lọc các nhánh Optimization / Self-Distillation / SAM / R-Drop Jacobian (7 truy vấn):**
  1. `"low-resource" "neural machine translation" "mBART" OR "NLLB" ACL 2024 OR ACL 2025 OR EMNLP 2024 OR EMNLP 2025 regularization OR training objective`
  2. `"stop-gradient" "KL" OR "consistency" "R-Drop" "low-resource" "machine translation" OR "NMT" 2024 OR 2025 OR 2026`
  3. `"self-distillation" OR "sharpness" OR "SAM" OR "weight averaging" "low-resource" "neural machine translation" ACL OR EMNLP OR ICLR 2024 2025`
  4. `("Functional SAM" OR "KL-SAM" OR "SAM" "R-Drop" OR "Sharpness-Aware" "KL divergence" "stop-gradient") "machine translation" OR "NMT" OR "NLP"`
  5. `"Sharpness-Aware Minimization" "neural machine translation" OR "NMT" OR "low-resource" ACL OR EMNLP OR ICLR OR NeurIPS`
  6. `("EMA" OR "exponential moving average" OR "self-distillation") "neural machine translation" "low-resource" ACL OR EMNLP 2023 OR 2024 OR 2025`
  7. `"R-Drop" "stop-gradient" OR "reverse KL" OR "Fisher" OR "Jacobian" "sharpening" OR "overconfidence" OR "mode-seeking"`
- **Cụm 2 — Kiểm tra trực tiếp độ mới cho `CoCC` (*Sentence Concatenation + Token-Aligned Distribution Invariance*, 12 truy vấn):**
  8. `"concatenation" "consistency" "KL" "neural machine translation" OR "NMT" "sentence"`
  9. `"sentence concatenation" OR "concatenating sentences" "low-resource" "neural machine translation" ACL OR EMNLP OR ICLR OR WMT`
  10. `"compositional" "consistency" "KL divergence" "machine translation" OR "seq2seq" OR "NMT"`
  11. `"P(y_i|x_i, x_j)" OR "context invariance" OR "distractor sentence" "consistency" "neural machine translation"`
  12. `"document-level" OR "multi-sentence" "consistency regularization" "single-sentence" "neural machine translation" ACL OR EMNLP`
  13. `"Data Augmentation by Concatenation for Low-Resource Translation" "Nguyen" arxiv pdf`
  14. `"Data Augmentation by Concatenation for Low-Resource Translation" "consistency" OR "KL" OR "R-Drop" OR "distillation"`
  15. `"Focused Concatenation" "Context-Aware" "Neural Machine Translation" pdf aclanthology OR arxiv`
  16. `"invariance" "concatenation" "KL" "translation" OR "seq2seq" OR "encoder-decoder"`
  17. `"multi-view" OR "consistency" "sentence concatenation" OR "concatenated" "machine translation" OR "NMT" 2022 OR 2023 OR 2024 OR 2025 OR 2026`
  18. `"Sentence Concatenation" ("KL divergence" OR "R-Drop" OR "self-distillation" OR "consistency loss") "machine translation" OR "NMT"`
  19. `"CONMT" AAAI "consistency" "machine translation" OR "SeqMix" "KL" "consistency" NMT`

### 3.2. Nguồn gốc NotebookLM (`bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`, conv `8ec5ca51-f2be-4cca-a640-c5b4035a99d5`)
- **2 paper gốc mới thêm vào Notebook CLRR (cùng 6 paper gốc có sẵn):**
  1. `a57b68a4-26fd-4e5d-9057-09de85598a54` — **Nguyen, Murray, & Chiang (IWSLT 2021)**, *Data Augmentation by Concatenation for Low-Resource Translation: A Mystery and a Solution* (`https://aclanthology.org/2021.iwslt-1.33.pdf`)
  2. `fd9cf5b6-1f95-4b61-9b64-85bc74ba1eef` — **Lupo, Dinarelli, & Besacier (WMT 2022)**, *Focused Concatenation for Context-Aware Neural Machine Translation* (`https://aclanthology.org/2022.wmt-1.77.pdf`)
  3. `3f48a525-0c39-494b-85cb-62067da08d96` — **Liang et al. (NeurIPS 2021)**, *R-Drop* (`2106.14448v2.pdf`)
  4. `7f03b33a-7333-4425-8c60-99efde611936` — **EDWARD-NMT (Findings AACL 2026)** (`https://aclanthology.org/2026.findings-aacl.149.pdf`)
  5. `2ee085d3-8094-498e-8e39-d504e984562c` — **Guo et al. (ACL 2022)**, *Prediction Difference Regularization (PD-R)* (`https://aclanthology.org/2022.acl-long.528.pdf`)
  6. `0f4ec420-4eb6-4a67-acf2-07ea1a3efa3e` — **Voita et al. (ACL 2021)**, *Analyzing the Source and Target Contributions to Predictions in NMT* (`https://aclanthology.org/2021.acl-long.91.pdf`)
  7. `e8c4cdd3-a413-4534-933d-0fe7a6ed2410` — **Wang et al. (NAACL 2021)**, *Multi-view Subword Regularization (MVR)* (`https://aclanthology.org/2021.naacl-main.40.pdf`)
  8. `67b0c77f-8a1f-4ab7-9259-f7ae9876aac7` — **Sennrich et al. (EACL 2024)**, *Source-Contrastive & Language-Contrastive Decoding (SCD)* (`https://arxiv.org/abs/2309.07098`)

### 3.3. Phán quyết NotebookLM & 4 Đính chính phê phán (không chấp nhận mù quáng)
- **Vòng A (`NOTEBOOKLM_ROUND4_COCC_A.json`):** `CONDITIONAL ADMISSION TO ROUND B REFINEMENT` — xác nhận **`CHƯA THẤY` trong tài liệu đã xuất bản** (không phải "chỉ là Concat + R-Drop"), nhưng chỉ ra đúng lỗ hổng va chạm ranh giới $y_{j,1}$ và nguy cơ mất `[EOS]` ở Layout S0.
- **Vòng B (`NOTEBOOKLM_ROUND4_COCC_B.json`):** **`ADMIT TO LOCKED CANDIDATE CARD`** sau khi chứng minh `Layout S1 + mask_y_j1 = True + decoder_input_ids_comp` trước mask `-100` bằng đo đạc trực tiếp trên `mBART-50` đóng băng và kiểm tra FP64.
- **4 điểm chủ động đính chính lại đề xuất của NotebookLM để giữ chuẩn khoa học cohort A100:**
  1. **Giữ locked validation selector `beam 1` FP32 thay cho `beam 5, lp=1.0` của Round B §3:** Protocol [amis_a100_bleu.json](../../configs/protocols/amis_a100_bleu.json) khóa `validation_beams = 1` để **tái sử dụng hợp lệ 100%** ba đối chứng `BASELINE_S42 (17.5264)`, `W_B125_S42 (17.7069)` và `GOC_SYM_B250_S42 (18.5143)` (beam 4 chỉ mở cho test sau khi qua gates).
  2. **Thay $\beta = 1.0$ tùy tiện của Round B §3 bằng hiệu chuẩn Step-0 neo theo `GOC_SYM_B250_S42` ($\beta=2.50$):** Đảm bảo so sánh công bằng về lực cập nhật gradient với đối chứng mạnh nhất hiện có (`2.50` mặc định nếu $\rho_{\text{grad}}$ trong biên `1.5×`, hoặc làm tròn trong `[0.50, 2.50]` trước Bước 1).
  3. **Sửa lỗi kỹ thuật `fork_rng` ở Round A §3 thành `COCC_PURE_CTX` (`dropout = 0` trên hai nhánh KL):** `fork_rng` không thể làm $\epsilon_1 = \epsilon_2$ cho activation dropout giữa hai tensor khác hình dạng $(B, L, d)$ và $(B/2, 2L, d)$; chỉ có đánh giá hai view KL với dropout tắt ($\epsilon_1 = \epsilon_2 = 0$) trong Stage-2 mới cô lập chính xác 100% bất biến ngữ cảnh khỏi nhiễu dropout.
  4. **Bác bỏ nhận định Vòng A §2B về `p_comp` entropy cao & giữ `COCC_SYM` làm ứng viên chính:** Số đo thật trên `mBART-50` cho thấy entropy nội đoạn của `p_iso` và `p_comp` ngang bằng (`6.20` vs `6.36` Amis; `4.32` vs `4.15` Rhade), còn biến thể một chiều `V2` làm $\nabla_{z_{\text{iso}}}\mathcal{L}_{\text{KL}} \equiv \mathbf{0}$ (kiểm chứng FP64 `6.94e-18`), xóa bỏ hoàn toàn regularization cho câu đơn — do đó `COCC_SYM` là ứng viên chính, còn `COCC_ISO_TEACHER` giữ vai trò ablation song song trong pilot.
