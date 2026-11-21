# Chẩn đoán Mục 3 — Đối chiếu Baseline42, W42, AC42, DRM42 và VCO42 (Amis→Mandarin, RTX6000)

Cập nhật: **21/11/2026**. Phạm vi: **CPU-only, validation-only (576 câu gốc), không chạy optimizer step mới, không mở test mới**. Toàn bộ số liệu được tái tính (replay) khớp `0.00e+00` với receipts gốc.

---

## 1. Bảng bàn giao provenance & kết quả (`run | data / protocol | seed | split / metric | score | receipt`)

- **[ĐO ĐƯỢC]** File chuẩn `data_processed/amis_mandarin/validation.csv`: **576 dòng**, SHA256 `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4`. Toàn bộ 6 file `validation_predictions.csv` (5 model chính + đối chứng CE2 `A_NATIVE_CE_S42`) khớp 100% từng dòng `source` và `target` với `validation.csv`.
- **[ĐO ĐƯỢC]** Protocol chung: `mBART-50` revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`, `tl_XX → zh_CN`, FP32, SDPA (baseline42 có lưu ý provenance backend lịch sử tái dựng), B4×GA32, LR 5e-5, max 20 epochs (720 steps), ES10 chọn theo **original-reference validation BLEU, beam 1, FP32**. Replay `sacrebleu 2.6.0` khớp `max_diff = 0.00e+00` cho cả BLEU (`zh`), chrF (`word_order=0`, `chrf0`), chrF++ (`word_order=2`).

| Run | Data / Protocol | Seed | Split / Metric | Score (BLEU / chrF0 / chrF++) | Best Step / Stop | Receipt & Archive SHA256 |
|---|---|---:|---|---:|---|---|
| `BASELINE_S42` (`R1`) | `amis_mandarin` / RTX FP32 B4×GA32 ES10 val-BLEU beam1 (SDPA historical) | 42 | val576 / corpus | **17.6823** / 16.9547 / 14.7030 | 684 (ep19) / `epoch_cap` | `outputs/experiments/amis_token_rtx/runs/R1/VAL_COMPLETE.json` · SHA `7fbb9e66…` |
| `W_S42` (`weak-full-consistency-v1`, R-Drop $\beta=1.25$) | `amis_mandarin` / RTX FP32 SDPA B4×GA32 ES10 val-BLEU beam1 | 42 | val576 / corpus | **18.2724** / 17.3888 / 15.1728 | 720 (ep20) / `epoch_cap` | `outputs/experiments/amis_decoupled_consistency_rtx/runs/W_S42/VAL_COMPLETE.json` · SHA `ebf1a2eb…` |
| `AC_S42` (`antisymmetric-consistency-v1`, $\beta=1.25$) | `amis_mandarin` / RTX FP32 SDPA B4×GA32 ES10 val-BLEU beam1 | 42 | val576 / corpus | **17.6130** / 16.9289 / 14.7789 | 648 (ep18) / `epoch_cap` | `outputs/experiments/amis_two_method_round1_rtx/runs/AC_S42/VAL_COMPLETE.json` · SHA `e0d2ca2e…` |
| `DRM_S42` (`decision-residual-memory-v1`, $\beta=1.25, \eta=0.1$) | `amis_mandarin` / RTX FP32 SDPA B4×GA32 ES10 val-BLEU beam1 | 42 | val576 / corpus | **18.4724** / 17.4252 / 15.1557 | 648 (ep18) / `epoch_cap` | `outputs/experiments/sec3_diagnosis_vco_drm_ac_w/hf_artifacts/DRM_S42/VAL_COMPLETE.json` · SHA `470b695e…` |
| `VCO_S42` (`virtual-corrective-order-v1`, $r=32, \eta=1, \lambda=136.28$) | `amis_mandarin` / RTX FP32 SDPA B4×GA32 ES10 val-BLEU beam1 | 42 | val576 / corpus | **16.9681** / 16.2713 / 14.1664 | 648 (ep18) / `epoch_cap` | `outputs/experiments/amis_virtual_corrective_order_pilot_rtx/runs/VCO_S42/VAL_COMPLETE.json` · SHA `700c2320…` |
| `A_NATIVE_CE_S42` (`paired-native-ce-v1`, CE 2 views, $\beta=0$) *(đối chứng bổ sung)* | `amis_mandarin` / RTX FP32 SDPA B4×GA32 ES10 val-BLEU beam1 | 42 | val576 / corpus | **16.8619** / 16.3565 / 14.2623 | 612 (ep17) / `epoch_cap` | `outputs/experiments/amis_paired_route_rtx/runs/A_NATIVE_CE_S42/VAL_COMPLETE.json` · SHA `40d704c7…` |

---

## 2. Trajectory VCO_S42 theo epoch và so sánh động học tối ưu (Mục 3.1)

Nguồn: `hf_artifacts/VCO_S42/microbatch_diagnostics.jsonl` (23.000 microbatches, SHA khớp `bbbc71fb…`), `final_trainer_state.json` (SHA khớp `9f902ae0…`), và `final_trainer_state.json` của `BASELINE_S42`, `W_S42`, `AC_S42`, `DRM_S42`, `A_NATIVE_CE_S42`.

### 2.1. Tổng hợp `VCO_S42` theo từng epoch (1.150 microbatches/epoch)

| Ep | Mean CE | Mean Order (SKL) | Mean Weighted Order ($\lambda \cdot R$) | Ratio `W*Ord / CE` *(mô tả)* | Mean Endpoint Gap | Logged GradNorm | Val CE | Val BLEU | MB Wall Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.4459 | 4.466e-05 | 6.086e-03 | 1.369e-03 | 1.741e-02 | 4.2880 | 3.6249 | 1.3249 | 510.2 |
| 2 | 3.1575 | 2.691e-06 | 3.667e-04 | 1.161e-04 | 9.604e-03 | 3.2466 | 2.9903 | 5.9701 | 509.8 |
| 3 | 2.3766 | 1.333e-06 | 1.816e-04 | 7.643e-05 | 1.016e-02 | 3.0633 | 2.7242 | 9.5074 | 509.7 |
| 4 | 1.8100 | 1.063e-06 | 1.449e-04 | 8.003e-05 | 1.128e-02 | 2.9156 | 2.6292 | 11.8140 | 509.0 |
| 5 | 1.3515 | 9.571e-07 | 1.304e-04 | 9.651e-05 | 1.264e-02 | 2.8213 | **2.5916** | 11.9212 | 509.7 |
| 6 | 0.9782 | 1.185e-06 | 1.615e-04 | 1.651e-04 | 1.357e-02 | 2.9961 | 2.6219 | 12.7985 | 510.7 |
| 7 | 0.6875 | 8.047e-07 | 1.097e-04 | 1.595e-04 | 1.279e-02 | 2.4035 | 2.6506 | 13.8480 | 508.4 |
| 8 | 0.4765 | 5.986e-07 | 8.157e-05 | 1.712e-04 | 1.134e-02 | 1.9916 | 2.6819 | 14.8367 | 509.0 |
| 9 | 0.3309 | 3.167e-07 | 4.315e-05 | 1.304e-04 | 8.761e-03 | 1.7149 | 2.7393 | 14.9819 | 510.4 |
| 10 | 0.2337 | 1.583e-07 | 2.157e-05 | 9.232e-05 | 6.486e-03 | 1.3961 | 2.7857 | 15.1133 | 509.2 |
| 11 | 0.1687 | 7.007e-08 | 9.549e-06 | 5.662e-05 | 4.829e-03 | 1.1535 | 2.8381 | 16.6364 | 508.8 |
| 12 | 0.1257 | 4.799e-08 | 6.540e-06 | 5.203e-05 | 3.295e-03 | 0.9743 | 2.8894 | 16.5928 | 509.4 |
| 13 | 0.0965 | 1.120e-08 | 1.526e-06 | 1.582e-05 | 2.205e-03 | 0.8807 | 2.8911 | 16.2720 | 508.7 |
| 14 | 0.0761 | 7.112e-09 | 9.693e-07 | 1.274e-05 | 1.603e-03 | 0.8589 | 2.9285 | 16.9255 | 508.6 |
| 15 | 0.0630 | 1.688e-09 | 2.300e-07 | 3.653e-06 | 1.041e-03 | 0.7349 | 2.9437 | 16.7095 | 509.5 |
| 16 | 0.0535 | 1.122e-09 | 1.529e-07 | 2.857e-06 | 7.724e-04 | 0.5323 | 2.9619 | 16.7826 | 508.3 |
| 17 | 0.0480 | 5.630e-10 | 7.673e-08 | 1.600e-06 | 6.256e-04 | 0.4539 | 2.9693 | 16.6240 | 508.5 |
| 18 | 0.0439 | 3.574e-10 | 4.870e-08 | 1.110e-06 | 5.352e-04 | 0.4110 | 2.9838 | **16.9681** | 507.5 |
| 19 | 0.0411 | 2.801e-10 | 3.817e-08 | 9.293e-07 | 4.717e-04 | 0.4799 | 2.9847 | 16.9657 | 507.0 |
| 20 | 0.0399 | 2.615e-10 | 3.563e-08 | 8.940e-07 | 4.321e-04 | 0.4909 | 2.9860 | 16.6062 | 507.8 |

### 2.2. Đối chiếu động học 6 mô hình tại các mốc epoch

| Model | Val CE Ep 5 | Val CE Ep 10 | Val CE Ep 15 | Val CE Ep 20 | Best Val BLEU (Ep) | Last-10 logged Train CE | Last-10 logged Aux |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BASELINE_S42` | 2.5487 | 2.6734 | 2.8176 | 2.8517 | 17.6823 (Ep19) | — | — |
| `W_S42` (R-Drop 1.25) | 2.5123 | 2.5786 | 2.7174 | 2.7636 | 18.2724 (Ep20) | 0.0888 | KL = 0.0304 |
| `AC_S42` | **2.4860** | **2.4921** | **2.5583** | **2.5813** | 17.6130 (Ep18) | 0.0880 | KL = 0.0346 |
| `DRM_S42` | 2.5079 | 2.5746 | 2.7154 | 2.7625 | **18.4724** (Ep18) | 0.0870 | KL = 0.0314 |
| `VCO_S42` | 2.5916 | 2.7857 | 2.9437 | 2.9860 | 16.9681 (Ep18) | **0.0616** *(ep20 mean 0.0399)* | `w_ord` = **9.32e-09** |
| `A_NATIVE_CE_S42` (CE2) | 2.5952 | 2.7863 | 2.9560 | 2.9920 | 16.8619 (Ep17) | — | 0 |

- **[ĐO ĐƯỢC]** `VCO_S42` và `A_NATIVE_CE_S42` (CE hai views không có KL) trùng khớp gần như hoàn toàn về quỹ đạo Val CE (`2.5916/2.7857/2.9437/2.9860` vs `2.5952/2.7863/2.9560/2.9920`), điểm Val BLEU (`16.9681` vs `16.8619`), và có tới **147/576 (25.5%) câu dự đoán giống hệt nhau từng ký tự** (so với chỉ 50/576 với `BASELINE_S42` và 91/576 với `W_S42`).
- **[ĐO ĐƯỢC]** Thời gian microbatch của `VCO_S42` rất ổn định (~509 giây/epoch ≈ 8.5 phút/epoch; tổng ~2.83 giờ train chưa kể eval).
- **[ĐO ĐƯỢC — Dữ kiện còn thiếu trong log]** Logs `VCO_S42` **không ghi**: (a) gradient norm riêng của $L_{\text{CE}}$ và $\lambda R_{\text{order}}$ sau epoch 0; (b) độ lớn bước cập nhật Adam riêng từng thành phần; (c) năng lượng feature chiếu $\|P h\|^2 / \|h\|^2$ theo epoch. Do đó tỷ số `weighted_order / CE` giảm từ `1.37e-3` xuống `8.94e-7` chỉ là **thống kê mô tả**, chưa tự chứng minh lực cập nhật Adam bằng 0 hoặc loại trừ giả thuyết né subspace.

---

## 3. Đối chiếu lỗi validation: Baseline, AC, DRM, VCO so với W42 (Mục 3.2)

### 3.1. Thống kê toàn tập (576 câu validation, sentence `chrF0`)

| Model vs `W_S42` | Giống hệt W | Hơn W (`chrF0`) | Kém W (`chrF0`) | Hòa | Hơn $\ge 5$ / Kém $\ge 5$ | Hơn $\ge 10$ / Kém $\ge 10$ | Mean $\Delta$ chrF0 | Corpus $\Delta$ BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BASELINE_S42` | 59 | 232 | 237 | 107 | +82 / −86 | +38 / −48 | −0.512 | −0.5901 |
| `AC_S42` | **188** | 168 | 172 | 236 | +36 / −43 | +15 / −22 | −0.483 | −0.6594 |
| `DRM_S42` | 125 | **205** | 201 | 170 | **+64 / −56** | **+32 / −28** | −0.065 | **+0.2000** |
| `VCO_S42` | 91 | 192 | **237** | 147 | +43 / **−73** | +20 / **−37** | **−1.032** | **−1.3043** |
| `A_NATIVE_CE_S42` (CE2) | 94 | 190 | 236 | 150 | +46 / −83 | +21 / −37 | −0.992 | −1.4105 |

- **[ĐO ĐƯỢC — Cờ cấu trúc bề mặt trên 576 câu]** Tất cả 5 model đều có **0 câu rỗng**, **0 câu lặp 3-gram/4-gram $\ge 3$ lần**. Tỷ lệ độ dài output/ref ổn định (`0.9506–0.9707`). Số câu quá ngắn (`<0.5 ref`): `DRM` thấp nhất (4) vs `W`/`AC`/`VCO` (7). Số câu quá dài (`>1.5 ref`): `VCO` cao nhất (31) vs `W` (26), `DRM` (24), `AC` (22).
- **[ĐO ĐƯỢC — Phân tầng theo độ dài reference]**

| Độ dài ref (ký tự) | Số câu | W BLEU | BASE − W | AC − W | DRM − W | VCO − W |
|---|---:|---:|---:|---:|---:|---:|
| $\le 9$ | 241 | 20.262 | −0.692 | −0.679 | +0.397 | −1.389 |
| 10–11 | 141 | 18.710 | −2.228 | −0.983 | −1.432 | −2.178 |
| 12–14 | 117 | 16.753 | +0.367 | −1.204 | +1.174 | +0.056 |
| $> 14$ | 77 | 14.423 | −0.086 | +0.567 | +0.829 | −1.471 |

- **[ĐO ĐƯỢC — Chồng lấn nhóm lỗi (`|delta chrF0| >= 5.0` vs W)]** Trong 73 câu `VCO` làm hỏng so với `W`, có **41/73 (56.2%)** cũng bị `BASELINE_S42` làm hỏng và **21/73 (28.8%)** bị `AC_S42` làm hỏng. Chỉ có 5 câu giao nhau giữa `DRM` sửa được `W` (64 câu) và `VCO` làm hỏng `W` (73 câu).

### 3.2. Gắn nhãn thủ công mẫu lấy đều (40 câu, 10 mẫu/nhóm)

**Quy ước lấy mẫu & gắn nhãn:** Lấy 10 dòng cách đều theo thứ tự row ID trong từng nhóm `|delta chrF0| >= 5.0` so với `W_S42` (không chọn tay ví dụ đẹp). Nhãn đối chiếu theo single reference tiếng Hoa (chưa có chuyên gia song ngữ Amis):
- **L1**: Sửa hoặc làm hỏng từ vựng nội dung / thực thể / số cụ thể (danh từ, động từ, tính từ, số đếm).
- **L2**: Khôi phục hoặc làm mất (thiếu) một thành phần/ý nội dung nguồn.
- **L3**: Lặp từ cục bộ hoặc bịa thêm chi tiết không khớp nguồn.
- **L4**: Thay đổi bề mặt hợp lệ / đồng nghĩa / cấu trúc gần tương đương.
- **L5**: Cả `W` và model so sánh đều dịch sai nặng so với reference (không thể quy kết sửa/hỏng ngữ nghĩa chắc chắn).

| Nhóm mẫu (lấy đều) | Mẫu / Mẫu số | L1 (Từ vựng / Thực thể / Số) | L2 (Thiếu / Khôi phục nội dung) | L3 (Lặp / Bịa) | L4 (Bề mặt hợp lệ) | L5 (Cả hai sai / Không chắc) | Ví dụ tiêu biểu (Row ID) |
|---|---:|---:|---:|---:|---:|---:|---|
| **`DRM` sửa `W` ($\ge +5$)** | 10 / 64 | **6** | **3** *(1 kèm L1)* | 0 | 0 | 1 | Row 4 (`到→去`), Row 77 (khôi phục `像老人`), Row 136 (`紙張→地瓜`), Row 324 (khôi phục `快`), Row 378 (`房子→柱子`), Row 472 (`老師→師父`), Row 532 (khôi phục `來訪的`) |
| **`DRM` làm hỏng `W` ($\le -5$)** | 10 / 56 | **5** | 0 | 1 *(kèm L5)* | 0 | **4** | Row 138 (`褲子→顏色有毛`), Row 180 (`採蚌→採水草`), Row 383 (`哥哥→姐姐`), Row 438 (`當作老師→禁止去學校`) |
| **`VCO` làm hỏng `W` ($\le -5$)** | 10 / 73 | **8** *(2 kèm L2/L3)* | *(1)* | *(1)* | 0 | **2** | Row 17 (`小偷→逃走的逃走`), Row 58 (`謾罵→搖動`), Row 366 (`蒼蠅→無患子樹`), Row 465 (`風吹動→風扇吹掉`), Row 529 (`餵牛→剝皮牛奶`) |
| **`AC` làm hỏng `W` ($\le -5$)** | 10 / 43 | **7** *(1 kèm L2)* | **1** *(+1 kèm L3)* | *(1)* | 0 | **2** | Row 13 (`飼養鱷魚賺錢→製作風箏`, −85.95), Row 51 (`糯米酒→褲子`), Row 91 (`三個→兩個`), Row 169 (`眼鏡蛇→駱駝棚`), Row 233 (`跑步→河床抓蝦`) |

- **[ĐO ĐƯỢC — Đối chiếu với các báo cáo frozen cũ để tránh lặp]** Các báo cáo trước đã đo và không cần lặp lại: (i) donor sensitivity/shuffle trên `C0` (`amis_residual_seed_diagnostics/gpu/GPU_REPORT.md`); (ii) consensus câu khó giữa các seeds & source-gap (`amis_validation_seed_consensus/REPORT.md`); (iii) source-attribution / grounding `SAC` (`amis_source_attribution_frozen_rtx/ANALYSIS.md`: W giảm instability `C` nhưng tương quan với lỗi yếu); (iv) prefix cross-drift `PLC` (`docs/research/PLC_FAILURE_ANALYSIS.md`); (v) decision-branch rollouts (`amis_decision_branches_frozen_rtx/ANALYSIS.md`); (vi) frozen DRM direction probe (`drm_direction_probe_v3/REPORT.md`: lợi ích NLL ~`1e-5`, rất nhỏ); (vii) frozen VCO probes (`amis_virtual_corrective_order_probe_rtx/REPORT.md`).

---

## 4. Hiện tượng cốt lõi tái kiểm tra được & các giả thuyết cạnh tranh (Mục 3.3 & 3.4)

### Hiện tượng 1 (Đóng hồ sơ VCO): `VCO_S42` thoái hóa về hành vi của CE hai views không có KL (`A_NATIVE_CE_S42`)
- **[ĐO ĐƯỢC]** `VCO_S42` có quỹ đạo Val CE gần như trùng khít `A_NATIVE_CE_S42` (Ep 20: `2.9860` vs `2.9920`), điểm Val BLEU tương đương (`16.9681` vs `16.8619`), 25.5% câu dự đoán giống hệt nhau, và `weighted_order` giảm ~170.000× từ Ep 1 (`6.09e-3`) xuống Ep 20 (`3.56e-8`). Trong khi đó, bản thân `A_NATIVE_CE_S42` (CE 2 views, $\beta=0$) đã **kém `BASELINE_S42` (CE 1 view) −0.8204 BLEU** và overfit Val CE mạnh hơn (`2.9920` vs `2.8517`).
- **Ba giả thuyết cạnh tranh:**
  - **H1 — Suy kiệt quy mô penalty với $\lambda$ cố định `[SUY LUẬN]`**: Vì bước ảo $G_a(0) \propto \operatorname{softmax}(z) - e_y$ co nhanh khi Train CE giảm từ `4.45` xuống `0.04`, $R_{\text{order}}$ bậc cao giảm nhanh hơn nhiều bậc so với CE. Với $\lambda$ khóa tại pretrained, lực regularizer tắt dần sau vài epoch đầu.
  - **H2 — Né subspace cố định rank-32 `[PHỎNG ĐOÁN]`**: Backbone/head học cách xoay decoder features ra ngoài không gian chiếu cố định `basis_seed=11` để làm giảm `endpoint_gap` và `order` mà không tăng tính bất biến thực sự.
  - **H3 — Nhiễu hóa ngược do CE hai views không có output-space KL `[SUY LUẬN]`**: Trung bình CE của hai dropout views làm giảm phương sai gradient (làm yếu hiệu ứng regularization của dropout), khiến model ghi nhớ nhanh tập train 4.600 câu hơn cả single-view baseline.
- **Cách tách H1 vs H2 không cần optimizer step:** Đo $\|P h\|^2 / \|h\|^2$ và `order` trên nhiều basis trực chuẩn mới chưa thấy trên checkpoint cố định `VCO_S42` so với `A_NATIVE_CE_S42` và `W_S42`. Tuy nhiên vì **VCO đã đóng pilot (NO-GO)**, không cần mở train cho VCO; bài học bắt buộc cho vòng sau là **mọi regularizer 2 views phải so với matched control `CE2` và kiểm tra quy mô tín hiệu ở cả late-training**.

### Hiện tượng 2 (Trọng tâm cho vòng thiết kế method tiếp theo): Sự tách rời giữa Teacher-Forced Validation CE và Free-Running Generation BLEU (đặc biệt ở cặp `AC_S42` vs `W_S42` và lỗi sụp đổ từ vựng hiếm L1)
- **[ĐO ĐƯỢC]** Trên cùng protocol seed 42 RTX:
  - `AC_S42` đạt **Val CE tốt nhất trong toàn bộ 6 model (`2.5813` tại Ep 20, tốt hơn `W_S42` −0.1823 nats)**, và trước đó `Wcal` ($\beta=2.56$) cũng có Val CE tốt hơn `W` (`2.6709` vs `2.7636`, nguồn `PLC_FAILURE_ANALYSIS.md`), **nhưng cả `AC_S42` (`17.6130`) và `Wcal` (`17.2413`) đều thua `W_S42` (`18.2724`) rõ rệt về Val BLEU** (và `AC` thua `W` −0.3981 mean Test BLEU qua 3 seeds).
  - Trong 40 câu gắn nhãn thủ công, nguyên nhân áp đảo khiến `AC` (7/10) và `VCO` (8/10) làm hỏng câu `W` dịch đúng là **sụp đổ/thay thế từ vựng nội dung cụ thể/hiếm (L1)** thành từ phổ biến hoặc không liên quan (`鱷魚→風箏`, `糯米酒→褲子`, `眼鏡蛇→駱駝棚`, `蒼蠅→無患子樹`), chứ **không phải** lặp chữ (0/576) hay cắt cụt câu (0/576).
  - Ở chiều ngược lại, `DRM_S42` có Val CE gần như bằng `W_S42` (`2.7625` vs `2.7636`), sửa được nhiều từ vựng/nội dung thiếu của `W` (9/10 mẫu `DRM_fixes_W`), nhưng cũng tự phá mất một lượng tương đương từ đúng của `W` (5/10 mẫu `DRM_breaks_W`), dẫn đến test BLEU (−0.065) chưa vượt `W`.
- **Hai giả thuyết cạnh tranh cho Hiện tượng 2:**
  - **E1 — Thành phần common-view trong R-Drop (`W`) giúp ổn định margin top-1 của từ nội dung hiếm `[SUY LUẬN]`**: Khi `AC` loại bỏ thành phần đẩy chung $(g_1 + g_2)/2$ của symmetric KL, nó giữ tốt phân phối hiệu chuẩn xác suất (giảm Val CE mạnh xuống `2.58`), nhưng làm mất lực đẩy đồng thuận giữa hai views ở các token nội dung hiếm/bất định, khiến giải mã greedy/beam dễ trượt sang prior tần suất cao của tiếng Hoa.
  - **E2 — Đánh đổi giữa hiệu chuẩn xác suất (NLL/entropy) và độ sắc nét quyết định (top-1 margin) dưới tập train nhỏ `[PHỎNG ĐOÁN]`**: Trên 4.600 câu, các cơ chế giữ Val CE thấp (`AC`, `Wcal`, `CMC`) có phân phối đầu ra phẳng hơn (entropy cao hơn hoặc margin giữa từ đúng và đối thủ cạnh tranh thấp hơn ở các vị trí mang nội dung nguồn), làm tốt log-likelihood teacher-forced nhưng dễ sai từ khóa khi sinh chuỗi tự hồi quy.
- **Chẩn đoán 0-optimizer-step đề xuất trước khi khóa thiết kế vòng 2:** Trên 576 câu validation của 3 checkpoint cố định đã có (`BASELINE_S42`, `W_S42`, `AC_S42` — và tùy chọn `DRM_S42`), đo trực tiếp theo từng nhóm tần suất từ nguồn/đích (rare vs frequent content tokens): (1) entropy, (2) gold-vs-top-competitor logit margin, (3) xác suất top-1 và tỉ lệ sụp đổ sang high-frequency target prior, (4) khác biệt giữa teacher-forced top-1 và free-running prediction.

---

## 5. Bảng quyết định giữ / bỏ giả thuyết cho vòng tiếp theo

| Giả thuyết / Hướng | Nhãn bằng chứng | Quyết định | Căn cứ & Nguồn |
|---|---|---|---|
| VCO (`virtual-corrective-order-v1`) | **[ĐO ĐƯỢC]** | **BỎ / ĐÓNG PILOT (NO-GO)** | Val BLEU 16.9681 (−1.3043 vs W, −0.7142 vs Baseline), hành vi trùng `CE2` (147/576 câu giống hệt, Val CE 2.986 vs 2.992). |
| Ứng viên B (conditional innovation rewiring) | **[ĐO ĐƯỢC]** | **BỎ** | Đã loại trước train vì thuộc họ tuyến tính LAuReL (`TWO_METHOD_LOW_MEMORY_REVIEW.md`). |
| DRM v1 mở rộng ngay sang A100 / multi-seed | **[ĐO ĐƯỢC]** | **CHƯA THĂNG CẤP (GIỮ LÀM DỮ KIỆN)** | Val +0.20 nhưng Test −0.065 (mới 1 seed); sửa 64 câu nhưng cũng hỏng 56 câu so với W; frozen direction probe effect chỉ ~`1e-5` NLL. |
| Dùng Val CE (teacher-forced) làm proxy chọn method | **[ĐO ĐƯỢC]** | **BỎ** | `AC` (2.58) và `Wcal` (2.67) có Val CE tốt hơn `W` (2.76) nhưng BLEU thấp hơn rõ rệt. Giữ nguyên selector Val BLEU. |
| **Hiện tượng 2 (Val CE vs Generation BLEU decoupling & sụp đổ từ vựng hiếm L1 giữa W / AC / CE2)** | **[ĐO ĐƯỢC] + [SUY LUẬN]** | **GIỮ LÀM ĐIỂM XUẤT PHÁT VÒNG 2** | Lặp lại nhất quán qua `W` vs `AC`, `W` vs `Wcal`, `W` vs `CMC`, và được xác nhận trực tiếp trên 40 mẫu gắn nhãn thủ công (`L1` chiếm 7/10 ở AC và 8/10 ở VCO). |

---

*Artifacts đã lưu (không chứa weights):*
- `outputs/experiments/sec3_diagnosis_vco_drm_ac_w/SUMMARY.json`
- `outputs/experiments/sec3_diagnosis_vco_drm_ac_w/validation_5models_aligned.csv`
- `outputs/experiments/sec3_diagnosis_vco_drm_ac_w/hf_artifacts/FETCH_RECEIPT.json`
- `outputs/experiments/sec3_diagnosis_vco_drm_ac_w/REPORT.md`

---

## 6. Chẩn đoán bổ sung 0-optimizer-step (CPU) — Tách cơ chế lỗi `AC` vs `VCO`/`BASE` (E1 vs E2)

Đo trực tiếp trên 576 câu validation (so với tần suất ký tự Hán trong `data_processed/amis_mandarin/train.csv`, 4.600 câu; 0 GPU, 0 optimizer step).

### 6.1. Recall ký tự đích (CJK) theo tần suất xuất hiện trong `train.csv`

| Nhóm tần suất (train) | Tổng ký tự (REF) | BASE | W | AC | DRM | VCO | AC − W | DRM − W | VCO − W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Hiếm ($\le 10$)** | 789 | 10.65% | **11.41%** | 9.76% | **11.41%** | 9.25% | **−1.65%** *(−14.5% rel.)* | +0.00% | −2.15% |
| Trung bình (11–100) | 2.253 | 33.60% | **34.27%** | 33.56% | 34.22% | 32.36% | −0.71% | −0.04% | −1.91% |
| Phổ biến ($> 100$) | 2.930 | 60.61% | 60.61% | 60.00% | **61.23%** | 58.16% | −0.61% | +0.61% | −2.46% |

### 6.2. Kiểm tra trượt về prior tần suất cao (khi model khác `W`)

| Model vs `W` | Số câu khác W | Tần suất TB / trung vị ký tự **chỉ có ở Model** | Tần suất TB / trung vị ký tự **chỉ có ở W** | Chênh lệch TB (`Model − W`) |
|---|---:|---:|---:|---:|
| `BASELINE_S42` | 517 | 201.8 / 52.0 | 240.4 / 53.0 | **−38.6** |
| **`AC_S42`** | 388 | **220.4 / 54.0** | 204.3 / 48.0 | **+16.2** |
| `DRM_S42` | 451 | 220.2 / 53.0 | 231.4 / 51.0 | −11.1 |
| `VCO_S42` | 485 | 199.3 / 49.5 | 241.1 / 53.0 | **−41.7** |

- **[ĐO ĐƯỢC]** Tần suất trung bình toàn bộ ký tự dự đoán của `AC_S42` cao nhất trong 5 model (`433.9`, trung vị `125.0`, so với `REF = 391.1 / 93.0`, `W = 426.6 / 117.0`, `BASE = 409.6 / 117.0`).
- **[SUY LUẬN]** Kết quả này tách rõ hai cơ chế thất bại ngược chiều nhau:
  1. **`VCO_S42` và `BASELINE_S42`** (Val CE cao `2.986` / `2.852`) hỏng đồng đều trên mọi nhóm tần suất và thay thế bằng các ký tự có tần suất **thấp hơn** W (`−41.7` / `−38.6`) → đặc trưng của **overfitting / nhiễu không được regularize**.
  2. **`AC_S42`** (Val CE thấp nhất `2.581`) hỏng **tập trung vào ký tự hiếm ($\le 10$ trong train: −1.65%, tương đương −14.5% tương đối)** và khi bất đồng với W thì **trượt sang ký tự có tần suất cao hơn (`+16.2`)** → ủng hộ mạnh **E2 / E1**: việc bỏ thành phần common-view làm mượt phân phối (Val CE tốt hơn) nhưng làm mất độ sắc nét / trượt về prior tần suất cao tiếng Hoa trên từ nội dung hiếm.
- **[CHƯA ĐO — Cần GPU nếu muốn xác nhận ở mức logit]** Để đo trực tiếp **entropy** và **gold-vs-competitor logit margin** theo từng nhóm tần suất trên checkpoint cố định `W_S42` vs `AC_S42` vs `BASELINE_S42`, cần một job frozen-eval nhỏ (0 optimizer steps) trên RTX6000.

