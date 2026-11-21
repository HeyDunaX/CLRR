# Thiết kế phương pháp vòng 2 — Audit GOC (giữ) và IPAM/Self-PoE (loại trước train)

Ngày: **21/11/2026**. Trạng thái: **DESIGN & AUDIT ONLY** — chưa viết trainer production, chưa chạy GPU gate, chưa train hoặc mở test.
Mọi con số thực nghiệm trích từ [báo cáo chẩn đoán Mục 3](../../outputs/experiments/sec3_diagnosis_vco_drm_ac_w/REPORT.md). Kiểm toán đại số FP64 lưu tại `outputs/experiments/round2_method_design/MATH_CHECKS.json`; nhật ký phản biện NotebookLM hai vòng lưu tại `outputs/experiments/round2_method_design/NOTEBOOKLM_ROUND_{A,B}.json`.

---

## 1. Tóm tắt quyết định sau audit (Mục 4)

| Hướng | Cơ chế đề xuất | Vai trò | Kết luận sau audit paper gốc + NotebookLM |
|---|---|---|---|
| **Hướng 1 — GOC** (`Gold-Ordered Asymmetric Dropout Consistency`) | Dùng hai lượt dropout của `W`, định hướng KL một chiều có `stop-gradient` từ view có xác suất gold cao hơn $w_t$ sang view yếu hơn $\ell_t$ | Khai thác kết quả mạnh nhất `W` và sửa lỗi sụp đổ từ hiếm / ưu tiên prior của `W`/`AC` | **GIỮ LÀM ỨNG VIÊN DUY NHẤT (GẦN).** Cơ chế chưa thấy trong R-Drop / Baziotis / EDWARD-NMT; cần qua chẩn đoán 0-step trên `W_S42` và pilot 3 runs có đối chứng `GOC-RAND`, `GOC-INV`. |
| **Hướng 2 — IPAM / Self-PoE** (`Internal Prior-Amplified Margin`) | Chạy thêm lượt decoder tắt cross-attention để lấy internal LM $z^{\text{ilm}}$, train $(1-\lambda)\text{CE}(z^{\text{full}}, y) + \lambda\text{CE}(z^{\text{full}} + \alpha \operatorname{sg}(z^{\text{ilm}}), y)$ | Cơ chế độc lập (1 lượt encoder, không R-Drop) chống target-LM prior dominance | **TẠM GÁC TRƯỚC TRAINING (PARKED / HOLD).** Chưa có bằng chứng nhánh tắt cross-attention tạo ra prior đáng tin trên checkpoint `mBART-50` của project (Meng et al. 2021 §5.5 cảnh báo AED ILM-PPL lên 796.7 khi tắt context; Baziotis et al. 2020 Bảng 4 cảnh báo POSTNORM dùng external LM trên De→En làm giảm BLEU 26.9→24.5; Clark et al. 2019 §5.1 cảnh báo dependent-bias over-compensation). |

Theo quy tắc Mục 4 (`METHOD_DEVELOPMENT_HANDOFF.md`), chỉ giữ **1 hướng qua audit (`GOC`)** thay vì ép giữ một hướng thứ hai yếu. Không nhận GOC đã được chứng nhận novelty toàn cầu hay đã có bằng chứng tăng BLEU.

---

## 2. Candidate Card — Hướng 1: Gold-Ordered Asymmetric Dropout Consistency (GOC)

### 2.1. Câu hỏi nghiên cứu (RQ)
**RQ:** Trong dịch máy ít tài nguyên trên mô hình đa ngữ tiền huấn luyện (`mBART-50`, Amis→Mandarin), R-Drop đối xứng (`W`) có làm mất từ vựng nội dung hiếm vì (i) view dropout yếu kéo tụt view giữ đúng tín hiệu nguồn và (ii) thành phần gradient đối xứng $\frac{g_1+g_2}{2}$ đè nén token gold dao động mạnh để đẩy các prior tần suất cao ổn định không? Định hướng nhất quán theo thứ tự xác suất gold ($w_t \to \ell_t$ với `stop-gradient` trên $w_t$) có cải thiện recall từ hiếm và BLEU vượt `W` và vượt đối chứng `GOC-RAND` (cùng `stop-gradient`, chọn teacher ngẫu nhiên) không?

### 2.2. Hiện tượng có nguồn & Nguyên nhân thử kiểm
- **[ĐO ĐƯỢC — Chẩn đoán Mục 3]** Trên Amis validation seed 42 (576 câu, `REPORT.md` §2–§6):
  - `W_S42` (R-Drop $\beta=1.25$) đạt **18.2724 Val BLEU** (Val CE `2.7636`, recall ký tự Hán hiếm $\le 10$ trong train: **11.41%** trên 789 lượt ký tự tham chiếu).
  - `AC_S42` (bỏ thành phần gradient chung $\frac{g_1+g_2}{2}$ của SKL) đạt **Val CE thấp nhất (`2.5813`)**, nhưng tụt **−0.6594 Val BLEU** (`17.6130`), tụt **−0.3981 mean Test BLEU** (3 seeds), giảm recall ký tự Hán hiếm từ **11.41% → 9.76%** (tương đương chênh lệch **~13 lượt ký tự trên 789 lượt tham chiếu** — là tín hiệu cụ thể để đặt câu hỏi, chưa đủ kết luận cơ chế toàn cục), và thay thế bằng ký tự có tần suất train cao hơn W (`+16.2`) trên 388 câu bất đồng.
  - `B_NATIVE_RDROP_S42` ($\beta=2.5$) và `Wcal` ($\beta=2.56$) cũng giảm Val BLEU (`17.9107` / `17.2413`) khi tăng hệ số SKL đối xứng.
- **[ĐO ĐƯỢC — Chỉ trên logits mô phỏng FP64, chưa đo trên `mBART-50` thật (`check_skl_sym_math.py` & `MATH_CHECKS.json`)]**
  - Phân tích chính xác gradient của symmetric KL trên 2 views $(z_1, z_2)$:
    $$g_{\text{anti}} = \frac{g_1 - g_2}{2}, \qquad g_{\text{sym}} = \frac{g_1 + g_2}{2} = \frac{1}{2}\Big[(p_1 - p_2)\log\frac{p_1}{p_2} - p_1\text{KL}(p_1\|p_2) - p_2\text{KL}(p_2\|p_1)\Big].$$
  - Trên 1.000 cặp logit mô phỏng ngẫu nhiên, `-g_sym` làm giảm entropy trong **80.6%** mẫu mô phỏng.
  - **Ví dụ dựng minh họa (Case 2):** Khi token gold dao động mạnh giữa hai views ($p_1(y)=0.653, p_2(y)=0.296$) trong khi một đối thủ ổn định hơn, trong không gian logits cả `-g_anti` (trên view 1: `-0.342`) lẫn `-g_sym` (`-0.0069`) đều có dấu giảm logit gold. **Hiện chưa biết Case 2 xảy ra với tần suất và độ lớn bao nhiêu trên `mBART-50` thật, hay có giải thích được các lỗi dịch của `W` và `AC` không.**
- **[PHỎNG ĐOÁN CẦN KIỂM]** Định hướng KL một chiều có `stop-gradient` từ view có xác suất gold cao hơn sang view yếu hơn (`sg(p_w) -> p_l`) có thể giảm lực kéo ngược trên token gold ở view yếu và tránh mất độ sắc nét như `AC`.

### 2.3. Công thức, nguồn tín hiệu, gradient, chi phí & inference
Cho cặp `(x, y)`, chạy hai lượt forward native của `mBART-50` với hai dropout masks độc lập (giữ nguyên valid-token mask `labels != -100` như `W`), thu được logits $z_{1,t}, z_{2,t}$ và phân phối $p_{1,t}, p_{2,t}$ tại mỗi vị trí subword $t \in \{1,\dots,T\}$:

1. **Định thứ tự theo gold log-likelihood tại từng token subword $t$:**
   $$w_t = \arg\max_{v \in \{1,2\}} \log p_{v,t}(y_t), \qquad \ell_t = 3 - w_t$$
   (nếu $\log p_{1,t}(y_t) = \log p_{2,t}(y_t)$, lấy trung bình đều hai chiều $0.5 / 0.5$).
2. **Hàm mục tiêu `GOC v1`:**
   $$\mathcal{L}_{\text{GOC}} = \frac{1}{2}\big(\text{CE}(z_1, y) + \text{CE}(z_2, y)\big) + \beta_{\text{goc}} \cdot \frac{1}{T}\sum_{t=1}^{T} \text{KL}\!\Big(\operatorname{sg}(p_{w_t, t}) \,\Big\|\, p_{\ell_t, t}\Big)$$
3. **Tính chất gradient chính xác (đã kiểm FP64, sai số tối đa `5.20e-18`) và giới hạn phạm vi:**
   $$\frac{\partial \mathcal{L}_{\text{GOC}}}{\partial z_{w_t, t}} = \frac{1}{2}\big(p_{w_t, t} - e_{y_t}\big), \qquad \frac{\partial \mathcal{L}_{\text{GOC}}}{\partial z_{\ell_t, t}} = \frac{1}{2}\big(p_{\ell_t, t} - e_{y_t}\big) + \beta_{\text{goc}}\big(p_{\ell_t, t} - p_{w_t, t}\big).$$
   - **Phạm vi bảo đảm đại số:** Tại vị trí đang xét, KL phụ không truyền gradient trực tiếp qua logits $z_{w_t,t}$ của view làm teacher, và đẩy logit của token đúng lên ở view yếu $\ell_t$ (vì $p_{w_t,t}(y_t) - p_{\ell_t,t}(y_t) > 0$).
   - **Giới hạn bắt buộc ghi rõ:** Hai view vẫn dùng chung tham số $\theta$, nên sau bước optimizer AdamW dự đoán của teacher vẫn có thể thay đổi; đồng thời việc logit gold $z_{\ell_t,t}(y_t)$ tăng chưa tự bảo đảm khoảng cách (margin) giữa gold và đối thủ mạnh nhất $\max_{v \ne y_t} z_{\ell_t,t}(v)$ tăng (cần đo trực tiếp trong chẩn đoán 0 bước).
   - **Hệ số thử $\beta_{\text{goc}} = 2.50$:** Đây là **hệ số thử đã định trước** dùng chung cho cả 3 nhánh `GOC`, `GOC-RAND`/`SYM-SG`, `GOC-INV` để so công bằng quy tắc chọn teacher (không tuyên bố đã khớp lực regularization với `W`: trên logits mô phỏng FP64, L2 norm gradient của `GOC` ở $\beta=2.50$ bằng khoảng `1.41×` của `W`, và ở $\beta=1.25$ bằng khoảng `0.705×`). Để biết lực regularization thực tế so với `W`, cần đo riêng full-backbone auxiliary gradient norm của `GOC` và `W` trên **cùng batch và cùng dropout views** trong chẩn đoán 0 bước.
4. **Chi phí & Inference:**
   - **0 tham số thêm**, **0 overhead lúc inference** (dùng backbone `mBART-50` gốc, beam 1 val / beam 4 test).
   - Chi phí train **tương đương `W_S42`**: 2 lượt native forward/backward, tính loss theo chunk 32 tokens với reentrant checkpointing (giống hệt `src/methods/antisymmetric_consistency/v1/model.py`), peak VRAM ~12–14 GiB trên RTX 6000 24 GB.

### 2.4. Audit Novelty & Đối chiếu Prior gần (đã mở đọc nguồn gốc)

| Prior gần | Nguồn / Section | Phần trùng / gần | Khác biệt cụ thể cần kiểm chứng |
|---|---|---|---|
| **R-Drop** (Wu/Liang et al., NeurIPS 2021) | `2106.14448v2.pdf` §2.1–2.4, Prop. 2.1 | Hai lượt dropout trên cùng mạng, CE + KL trên phân phối đầu ra | R-Drop đối xứng hoàn toàn, không `sg`, không phụ thuộc $y_t$ trong KL; Prop. 2.1 không phân tích phân rã view-axis của gradient SKL. |
| **EDWARD-NMT** (2026) | ACL Anthology / arXiv `2610.32115v1` §3.4 Eq. 13 | Dùng `stop-gradient` trong dual-consistency KL hai chiều $\frac{1}{2}[\text{KL}(\text{sg}(P_A)\|P_B)+\text{KL}(\text{sg}(P_B)\|P_A)]$ | Thiết kế cho elastic sub-network widths ($D_A \ne D_B$); khi $D_A = D_B$, `GOC-RAND` có cùng gradient **theo kỳ vọng** khi giữ cố định hai views, nhưng chọn ngẫu nhiên từng token làm phương sai gradient khác (đối chứng khớp sát hơn với Eq. 13 là `SYM-SG` tất định). Cả hai không định hướng teacher theo gold likelihood $y_t$. |
| **Unidirectional R-Drop ablations** | Literature (R-Drop / speech-MT ablations) | Thử $\text{KL}(p_1\|p_2)$ một chiều | Chọn hướng cố định/ngẫu nhiên không theo $y_t$ (chính là control `GOC-RAND`) và thường thua bidirectional KL. |
| **LR-Drop** (LREC-COLING 2024) & **Fraternal Dropout** (ICLR 2018) | `2024.lrec-main.891.pdf` §3; `1711.00066` | Consistency giữa hai dropout subnetworks | Đối xứng trên hidden/attention/logits, không chọn teacher theo gold token. |
| **Baziotis et al.** (EMNLP 2020) | `2020.emnlp-main.615.pdf` §3, §5 | KL một chiều từ teacher sang student | Dùng external target LM cố định làm teacher; không phải hai dropout views nội tại. |

- **Kết luận novelty:** **GẦN (chưa thấy exact match trong phạm vi đã đọc; không phải chứng nhận novelty toàn cầu).** Đóng góp nếu đạt chuẩn ACL phải gồm cả **3 trụ cột**: (1) phát hiện phân rã view-axis của SKL và cơ chế Case 2 giải thích sự tách rời Val CE vs BLEU (`AC` vs `W`), (2) quy tắc `GOC` gọn không thêm chi phí, và (3) bằng chứng thực nghiệm vượt cả `W`, `GOC-RAND`, `GOC-INV` kèm cải thiện recall từ hiếm và kiểm chứng chuyển ngôn ngữ (Ê Đê) + cohort A100.

### 2.5. Dự đoán có thể bác bỏ & Chẩn đoán 0-step trước train (Falsifiers)
- **Rủi ro chính được NotebookLM Vòng A/B chỉ ra (chấp nhận để kiểm):**
  1. Một dropout pass đơn lẻ $p_{w_t}$ có phương sai cao: có thể thắng trên $y_t$ nhờ nhiễu ngẫu nhiên hoặc quá tự tin (target-LM overconfidence) trong khi làm hỏng phần đuôi không-gold.
  2. Chọn teacher theo $y_t$ có thể tăng tốc memorization / overconfidence trên tập train 4.600 câu.
- **Chẩn đoán 0-optimizer-step bắt buộc trên `W_S42` cố định (0 bước train, RTX6000) trước khi duyệt pilot:**
  *(Tránh phép đo lặp định nghĩa như chỉ xác nhận $p_{w_t}(y_t) > p_{\ell_t}(y_t)$; tách rạch ròi đơn vị **subword của mô hình** với **ký tự Hán đầu ra**.)*
  1. **Phạm vi dữ liệu & lặp dropout:** Chạy trên cùng subset cố định (TRAIN để đo động học tối ưu + VALIDATION để so khớp lỗi dịch, tách bảng riêng), lặp qua **8 cặp dropout masks độc lập** trên từng batch để kiểm độ ổn định qua nhiều cặp dropout.
  2. **Đo tần suất Case 2 & tác động lên margin thật:** Báo cáo trên **cả toàn bộ vị trí subword hợp lệ** lẫn **nhóm bất đồng mạnh** ($|p_1(y_t)-p_2(y_t)| > 0.05$), đồng thời phân tầng theo tần suất **subword đích** trong `train.csv` ($\le 10$, $11\text{–}100$, $>100$):
     - Tần suất Case 2: tỷ lệ vị trí mà $-g_{\text{sym}}(y_t) < 0$ (hoặc tổng aux gradient của `W` kéo tụt margin gold-vs-top-competitor trên view $w_t$).
     - Tác động thực tế lên **khoảng cách (margin) giữa gold và đối thủ mạnh nhất** $\Delta m_t = z(y_t) - \max_{v \ne y_t} z(v)$ của `W`, `AC`, `GOC`, `GOC-RAND`, `GOC-INV` (trực tiếp kiểm tra liệu tăng logit gold ở view yếu có thực sự tăng margin hay cũng kéo đối thủ mạnh nhất lên theo).
     - Kiểm tra khối lượng xác suất của $p_{w_t}$ so với $p_{\ell_t}$ trên top-100 subword tần suất cao (bác bỏ rủi ro $w_t$ chỉ là nhiễu prior).
  3. **Đo riêng full-backbone auxiliary gradient norm trên cùng batch và cùng dropout views:** So $\|\nabla_\theta L_{\text{aux}}^{\text{GOC}}\|$ với $\|\nabla_\theta L_{\text{aux}}^{W}\|$ (và cosine giữa chúng + cosine với $\nabla_\theta L_{\text{CE}}$) tại cả `W_S42` lẫn pretrained initialization để xác định lực regularization thực tế của $\beta = 2.50$.
  4. **Cổng dừng sau 0-step (chưa duyệt 3 run train):** Nếu Case 2 rất hiếm, hoặc ảnh hưởng lên gold-vs-competitor margin quá nhỏ / không lặp qua 8 cặp dropout trên checkpoint thật ⇒ **DỪNG GOC ngay**, không mở pilot 3 run.

### 2.6. Bảng Pilot khóa trước (Amis seed 42, RTX6000, validation-only)

Giữ nguyên protocol lock [amis_drb_bleu.json](../../configs/protocols/amis_drb_bleu.json): `mBART-50` rev `e30b6cb8…`, FP32, SDPA, TF32 off, reentrant checkpointing, LR 5e-5, B4×GA32, max 20 epochs, ES10 theo **val BLEU beam 1**, không mở test.

| Nhánh | Trạng thái | Công thức aux ($\mathcal{L}_{\text{CE}} = \frac{1}{2}(\text{CE}_1 + \text{CE}_2)$) | Vai trò khoa học |
|---|---|---|---|
| `BASELINE_S42` (`R1`) | **REUSE** (`7fbb9e66…`) | 0 (1 view) | Baseline CE (Val BLEU `17.6823`) |
| `A_NATIVE_CE_S42` | **REUSE** (`40d704c7…`) | 0 (2 views) | Đối chứng 2 views không KL (Val BLEU `16.8619`) |
| `W_S42` | **REUSE** (`ebf1a2eb…`) | $1.25 \cdot \text{SKL}(p_1, p_2)$ | Anchor R-Drop mạnh nhất (Val BLEU `18.2724`) |
| `B_NATIVE_RDROP_S42` | **REUSE** | $2.50 \cdot \text{SKL}(p_1, p_2)$ | R-Drop tại $\beta=2.5$ (Val BLEU `17.9107`) |
| `AC_S42` | **REUSE** (`e0d2ca2e…`) | $1.25 \cdot \text{SKL}_{\text{anti}}(p_1, p_2)$ | Bỏ thành phần chung (Val BLEU `17.6130`) |
| `MIDDLE_S42` | **REUSE** | Middle contrastive RTX | Comparator alignment cùng cohort (`16.6775`) |
| **`GOC_S42`** | **NEW #1** | $2.50 \cdot \text{KL}(\operatorname{sg}(p_{w_t}) \| p_{\ell_t})$ | **Ứng viên chính** (gold-ordered) |
| **`GOC_RAND_S42`** | **NEW #2** | $2.50 \cdot \text{KL}(\operatorname{sg}(p_{r_t}) \| p_{3-r_t})$, $r_t \sim \text{Bern}(0.5)$ | **Đối chứng phân biệt #1**: tách `sg` 1 chiều khỏi thứ tự gold |
| **`GOC_INV_S42`** | **NEW #3** | $2.50 \cdot \text{KL}(\operatorname{sg}(p_{\ell_t}) \| p_{w_t})$ | **Đối chứng bác bỏ #2**: đảo ngược thứ tự (yếu dạy mạnh) |

- **Ngân sách tài nguyên (RTX 6000 24 GB):** Chỉ **3 runs mới** (~2.5 giờ/run, tổng ~7.5 giờ GPU; VRAM dự kiến ~12–14 GiB < ngưỡng 20.000 MiB trống, disk < 40 GiB trống; backup HF và dọn weights sau khi verify SHA).
- **Cổng GO / NO-GO khóa trước:**
  - **Technical Gate:** CPU unit tests (công thức, gradient chính xác, tie symmetry, mask `-100`, chunking, save/load, native inference) PASS + GPU 2-step optimizer/resume gate PASS.
  - **Quality GO:** `GOC_S42 - W_S42 >= +0.30` Val BLEU (`>= 18.5724`) VÀ `chrF0 >= 17.3888`.
  - **Mechanism GO:** `GOC_S42 - GOC_RAND_S42 >= +0.20` Val BLEU VÀ `GOC_S42 > GOC_INV_S42`, đồng thời recall ký tự hiếm (`<=10`) > `11.41%` và không có dấu hiệu trượt prior tần suất cao như `AC`.
  - **NO-GO:** Hụt bất kỳ điều kiện nào ⇒ DỪNG, chẩn đoán trajectory/lỗi, không mở test, không sweep $\beta$ ngẫu hứng.

---

## 3. Hồ sơ Hướng 2 tạm gác — IPAM / Self-PoE (PARKED / HOLD trước training)

- **Công thức đã kiểm:** $\mathcal{L}_{\text{IPAM}} = (1-\lambda)\text{CE}(z^{\text{full}}, y) + \lambda\text{CE}(z^{\text{full}} + \alpha\operatorname{sg}(z^{\text{ilm}}), y)$, với $z^{\text{ilm}}$ thu từ decoder tắt cross-attention. Kiểm FP64 (`MATH_CHECKS.json`) xác nhận đạo hàm đúng (`1.39e-17`) và đúng chiều phạt prior (sửa lỗi đảo dấu của bản nháp trừ logit `ISGS` trước đó).
- **Lý do tạm gác (ưu tiên nguồn lực, không tuyên bố đã bị bác trực tiếp trên dữ liệu của mình):**
  1. **Chưa có bằng chứng nhánh tắt cross-attention tạo ra prior đáng tin trên checkpoint `mBART-50` hiện tại:** Meng et al. (ICASSP 2021, `2011.01991`, §5.5 Fig. 1b) cho thấy trên AED speech, zero hóa context làm ILM-PPL tăng lên **796.7** (so với 30.1 của LM riêng).
  2. **Cảnh báo prior từ cộng logit:** Baziotis et al. (EMNLP 2020, `2020.emnlp-main.615.pdf`, Bảng 4) ghi nhận POSTNORM (dùng external LM cố định trên Đức→Anh) làm giảm BLEU (**26.9 → 24.5**). Dù IPAM dùng nhánh nội tại và trộn hai loss nên không đồng nhất hoàn toàn với POSTNORM, đây vẫn là cảnh báo rủi ro mạnh.
  3. **Rủi ro dependent-bias & calibration shift:** Clark et al. (EMNLP 2019, `D19-1418.pdf`, §5.1) cảnh báo over-compensation khi bias phụ thuộc đặc trưng chính, và việc bỏ $+\alpha z^{\text{ilm}}$ lúc suy luận có thể gây lệch calibration.
- **Quyết định:** **TẠM GÁC (HOLD).** Không viết trainer, không chạy GPU gate trong lượt này. Nếu sau này muốn mở lại IPAM, bước đầu tiên bắt buộc là đo chất lượng/perplexity của nhánh tắt cross-attention trên chính checkpoint `mBART-50` Amis.

---

## 4. Nhật ký truy vấn tìm kiếm & NotebookLM (Mục 4 Pha 3–4)

### 4.1. Các truy vấn Web đã thực hiện (≥6 mỗi hướng)
- **Hướng 1 (GOC — 9 truy vấn):**
  1. `"R-Drop" asymmetric OR unidirectional OR "stop-gradient" OR "one-way" neural machine translation`
  2. `"better view" OR "lower loss" OR "gold" teacher dropout consistency "KL" NMT OR language model`
  3. `"R-Drop" "symmetric" gradient decomposition OR "antisymmetric" OR "common" sharpening`
  4. `"asymmetric" dropout consistency "KL divergence" "stop gradient" neural machine translation ACL OR EMNLP OR arXiv`
  5. `"R-Drop" rare words OR "frequency" OR "calibration" OR "over-smoothing" machine translation`
  6. `"self-distillation" two dropout passes "higher probability" OR "better subnetwork" ACL OR ICLR OR NeurIPS`
  7. `"EDWARD-NMT" OR ("stop-gradient" "R-Drop" "KL") aclanthology OR arxiv`
  8. `"bidirectional KL" "unidirectional KL" "R-Drop" ablation OR comparison arxiv OR aclanthology`
  9. `"R-Drop" "select" OR "better" OR "confidence" OR "loss" "stop-gradient" self-distillation NMT OR NLP`
- **Hướng 2 (IPAM — 7 truy vấn):**
  1. `"Prevent the Language Model from Being Overconfident in Neural Machine Translation" Miao 2021`
  2. `"cross-attention" zeroed OR masked OR ablated "prior" training loss neural machine translation`
  3. `"target-side" OR "language model prior" debiased training "Product of Experts" neural machine translation ACL OR EMNLP`
  4. `"source contribution" OR "source-conditioned" logit subtraction training neural machine translation hallucination`
  5. `"rare words" low-resource neural machine translation "prior" OR "over-smoothing" OR "logit" regularization ACL OR EMNLP 2023 2024 2025`
  6. `"Internal Language Model" "cross-attention" zeroed OR masked training NMT OR ASR OR seq2seq ACL OR EMNLP OR ICASSP`
  7. `"Product of Experts" "neural machine translation" OR "seq2seq" "cross-attention" zeroed OR ablated training` (+ 1 truy vấn logit adjustment VLM/NMT)

### 4.2. NotebookLM (`bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`, conv `c5c26a27-ae04-4a33-98d3-51d9a99e6db9`)
- **4 paper gốc mới thêm (đã tra trùng trước, receipt `NLM_ADDED_SOURCES.json`):**
  - `79845b89-6411-4663-b8da-e84017cf5f80` — Miao et al. (ACL 2021)
  - `42a8bc65-5ec9-4935-9354-9d7d75cfb146` — Baziotis et al. (EMNLP 2020)
  - `abb9d87b-1499-490f-a241-6364bfd31d3b` — Meng et al. (ICASSP 2021, ILME)
  - `924ba3d2-bb50-4a2a-bae6-be998f8a79e6` — Clark et al. (EMNLP 2019, PoE)
- **3 paper gốc đã có sẵn dùng chung:** R-Drop (`3f48a525…`), Voita et al. ACL 2021 (`0f4ec420…`), Sennrich et al. EACL 2024 (`67b0c77f…`).
- **Điểm đã đính chính từ phản hồi NotebookLM (không chấp nhận mù quáng):**
  1. Bác nhận định ở Vòng A rằng Liang et al. Prop. 2.1 phản bác Case 2 (Prop. 2.1 chỉ là bound tuyến tính cho gap train–inference NLL, không phân tích thành phần `-g_sym` theo từng token).
  2. Hiệu chỉnh nhận định ở Vòng B về scale $\beta=2\alpha$: tại $\beta=2\alpha=2.50$, tổng độ dịch chuyển 1 chiều khớp `W` nhưng L2 logit-gradient norm là `1.410×` `W` (đã đo bằng FP64).
  3. Thay gợi ý dùng LRP + external LM của Vòng B bằng chẩn đoán 0-step gọn, tự khép kín trên `W_S42` cố định (không thêm external LM).
