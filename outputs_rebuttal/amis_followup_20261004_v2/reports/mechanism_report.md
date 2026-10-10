# Báo cáo Cơ chế và Động lực học Biểu diễn (F3, F4, F5)

> **Đính chính review độc lập 04/10/2026:** F4 hiện INVALID: hai wrappers chung base_model khiến hook α=0.10 còn hoạt động khi chạy route-off. Không dùng số 0 câu đổi để kết luận routing vô tác dụng hoặc chỉ có tác dụng khi training. F3 chỉ có aggregate endpoint summaries, chưa đủ gradients/rank từng layer; rank center tối đa 575, không phải 1024. Xem docs/operations/CONFIGURATION_SEARCH_HANDOFF.md mục16 và independent_review_20261004.json. Raw outputs giữ nguyên; các kết luận mạnh trong báo cáo gốc dưới đây cần đọc theo đính chính này.


**Dự án:** CLRR Rebuttal — Amis-Mandarin Low-Resource Translation  
**Thời điểm thực thi:** 04/10/2026  
**Dữ liệu:** 21 best checkpoints (7 họ $\times$ 3 seeds: 42, 43, 44) + 35 điểm quỹ đạo (Phase C)  
**Tập dữ liệu probe:** `data_processed/amis_mandarin/validation.csv` (576 dòng) + 32 fixed rows (gradient probe)  
**Phần cứng:** Colab NVIDIA A100-SXM4-40GB (Standard, non-high-mem)

---

## 1. Tóm tắt Phát hiện Cơ chế Cốt lõi

1. **Hiệu ứng Route-on vs Route-off (F4):**
   - Trên cả 3 seeds của mô hình `Full detached` ($\alpha=0.10, \lambda=0.10$), khi tắt routing tại inference ($\alpha=0.00$) trên 576 câu tập validation (greedy decode, beam 1):
     - **Số câu thay đổi:** **0 / 576 (0.00%)**.
     - $\Delta\text{BLEU} = +0.0000$, $\Delta\text{chrF++} = +0.0000$, $\Delta\text{chrF0} = +0.0000$.
   - **Kết luận:** Việc cộng skip connection ở inference không hề làm thay đổi chuỗi argmax của decoder. Đóng góp của CLRR (nếu có) nằm ở quá trình dẫn truyền gradient lúc training chứ không đóng vai trò một cấu trúc định tuyến biểu diễn động lúc suy luận.

2. **Căn chỉnh Biểu diễn Latent (`paired_source_target_cosine`):**
   - Độ tương đồng cosine giữa biểu diễn Amis (source) và Tiếng Trung (target) được chi phối gần như hoàn toàn bởi trọng số loss JEPA/LSR ($\lambda$):
     - $\lambda=0.00$ (`baseline`, `clrr_only`): mean cosine **0.325 – 0.354**.
     - $\lambda=0.03$ (`tuned_lsr`, `full_detached_l003`): mean cosine **0.592 – 0.607**.
     - $\lambda=0.10$ (`matched_lsr`, `original_full_detached`, `full_source_attached`): mean cosine **0.801 – 0.806**.
   - Việc có hay không có CLRR (và có detach source hay không) **không làm thay đổi** mức độ căn chỉnh latent space (0.806 vs 0.802 vs 0.801).

3. **Effective Rank & Nguy cơ Sụp đổ Biểu diễn (Representation Collapse):**
   - Effective rank đo trên ma trận hiệp phương sai singular values của encoder representations đạt từ **384.9 đến 390.0** (trên tối đa 1024 chiều) trên tất cả 21 checkpoints.
   - **Kết luận:** Hoàn toàn **không xảy ra hiện tượng sụp đổ chiều (dimensional collapse)** ở bất kỳ biến thể nào, kể cả khi $\lambda=0.10$ ép cosine alignment lên 0.80.

4. **Xung đột Gradient (CE vs LSR Gradients):**
   - Gradient cosine giữa nhiệm vụ dịch (Cross-Entropy) và căn chỉnh biểu diễn (LSR) trung bình đạt từ **0.047 đến 0.068** (gần như trực giao). Không có xung đột gradient đối kháng gay gắt ($\cos < -0.1$), nhưng hai hướng gradient gần như độc lập.
   - Tỷ số độ lớn gradient ($\|\nabla L_{LSR}\| / \|\nabla L_{CE}\|$) chỉ dao động trong khoảng **0.0007 đến 0.0037** (~0.1% đến 0.3% lực cập nhật chính của CE).

5. **Vai trò của Detach (Attached vs Detached):**
   - Khi cho gradient đi qua CLRR source cache (`full_source_attached`), tỷ số gradient LSR tăng nhẹ (0.00205 vs 0.00161), nhưng hiệu năng test thực tế không tăng (BLEU 19.57 vs 19.77).
   - Thao tác detach source ở CLRR **không phải là nút thắt kìm hãm hiệu năng**.

---

## 2. Bảng Tổng hợp Cơ chế trên 21 Checkpoints (Validation Endpoints)

| Họ mô hình (Family) | $\alpha$ | $\lambda$ | Detach Source | Source Effective Rank | Target Effective Rank | Paired Cosine (Mean $\pm$ Std) | Gradient Cosine (Mean $\pm$ Std) | LSR / CE Grad Ratio ($\times 10^{-3}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.00 | 0.00 | — | $389.99 \pm 2.86$ | $390.19 \pm 2.12$ | $0.354 \pm 0.011$ | $0.056 \pm 0.005$ | $3.68 \pm 0.09^*$ |
| **CLRR-only detached** | 0.10 | 0.00 | True | $385.02 \pm 1.49$ | $386.52 \pm 0.21$ | $0.325 \pm 0.004$ | $0.047 \pm 0.006$ | $2.65 \pm 0.07^*$ |
| **Matched LSR** | 0.00 | 0.10 | — | $386.33 \pm 0.87$ | $379.55 \pm 0.83$ | $0.806 \pm 0.009$ | $0.068 \pm 0.004$ | $2.33 \pm 0.08$ |
| **Tuned LSR** | 0.00 | 0.03 | — | $389.58 \pm 3.06$ | $384.49 \pm 2.89$ | $0.607 \pm 0.023$ | $0.063 \pm 0.013$ | $0.99 \pm 0.09$ |
| **Full detached** | 0.10 | 0.10 | True | $385.29 \pm 0.34$ | $379.91 \pm 0.69$ | $0.802 \pm 0.002$ | $0.054 \pm 0.007$ | $1.61 \pm 0.03$ |
| **Full attached** | 0.10 | 0.10 | False | $384.91 \pm 0.49$ | $379.49 \pm 0.69$ | $0.801 \pm 0.007$ | $0.062 \pm 0.006$ | $2.05 \pm 0.09$ |
| **Full detached $\lambda=0.03$** | 0.10 | 0.03 | True | $385.56 \pm 0.48$ | $383.16 \pm 0.41$ | $0.592 \pm 0.009$ | $0.048 \pm 0.002$ | $0.70 \pm 0.01$ |

*\* Ghi chú: Ở Baseline và CLRR-only, LSR không tham gia training ($\lambda=0$); tỷ số gradient ở đây được tính counterfactual với $\lambda=0.10$ trên cùng 32 batch cố định để làm mốc đối chứng.*

---

## 3. Động lực học Quỹ đạo Huấn luyện (35 Checkpoints từ Phase C)

Từ 35 checkpoint trung gian (Epoch 1, Epoch 5, và Endpoint qua 3 seeds):

1. **Tiến trình Hội tụ Loss:**
   - **Epoch 1:** $L_{CE} \approx 3.75 - 3.82$, $L_{LSR} \approx 0.05 - 0.06$. Tỷ số $\|\nabla L_{LSR}\| / \|\nabla L_{CE}\| \approx 0.006 - 0.007$.
   - **Epoch 5:** $L_{CE}$ giảm nhanh về $2.62 - 2.71$. $L_{LSR}$ giảm về $0.030 - 0.032$. Tỷ số gradient giảm còn $0.003 - 0.005$.
   - **Endpoint (Epoch 18–20):** $L_{CE}$ tăng nhẹ do overfitting ($2.90 - 3.04$), trong khi $L_{LSR}$ tiếp tục giảm xuống $0.019 - 0.020$. Tỷ số gradient giảm chạm đáy $0.0014 - 0.0022$.
2. **Tiến trình Căn chỉnh Biểu diễn:**
   - Cosine alignment tăng đơn điệu từ $\sim 0.55$ ở Epoch 1 lên $\sim 0.68$ ở Epoch 5 và đạt $\sim 0.78 - 0.80$ ở Endpoint.
   - Effective rank mở rộng từ $\sim 350$ ở Epoch 1 lên $\sim 385$ ở Endpoint, phản ánh không gian biểu diễn không bị co cụm mà dần bao quát dữ liệu.

---

## 4. Kiểm định Độc lập: Route-on vs Route-off trên Validation (F4)

Kết quả chi tiết khi bật routing ($\alpha=0.10$) so với tắt routing ($\alpha=0.00$) trên cùng checkpoint đã huấn luyện:

| Model ID | Seed | Validation Sentences | Changed Sentences | % Thay đổi | BLEU on | BLEU off | $\Delta$ BLEU | chrF++ on | chrF++ off | $\Delta$ chrF++ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full_a010_l010-seed42-replica` | 42 | 576 | 0 | **0.00%** | 17.6437 | 17.6437 | **0.0000** | 14.7317 | 14.7317 | **0.0000** |
| `full-seed43` | 43 | 576 | 0 | **0.00%** | 17.0645 | 17.0645 | **0.0000** | 14.4126 | 14.4126 | **0.0000** |
| `full-seed44` | 44 | 576 | 0 | **0.00%** | 16.9324 | 16.9324 | **0.0000** | 14.3483 | 14.3483 | **0.0000** |

**Kết luận thực nghiệm:** Routing layer không đóng vai trò trực tiếp trong việc phân tách từ vựng hay chọn token lúc inference.
