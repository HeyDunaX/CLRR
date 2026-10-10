# Báo cáo Quyết định Thực nghiệm & Đánh giá Cổng Dự án (F2, F6)

> **Cập nhật Revalidation 04/10/2026 (v3 Completed):** Đã hoàn tất sửa lỗi F4 (Route-on vs Route-off) và chạy F3 toàn diện 12 tầng encoder trên 21 checkpoints. Các kết quả xác nhận: CLRR có tác động trực tiếp lúc inference (89.06% câu thay đổi, CE loss giảm) và tăng 34.4% effective rank ở tầng 12 của encoder. Quyết định chiến lược NO-GO cho claim vượt trội hệ thống vẫn giữ nguyên dựa trên tiêu chuẩn chất lượng dịch thực tế so với Tuned LSR.


**Dự án:** CLRR Rebuttal — Kiểm định Mở rộng Cấu hình Amis-Mandarin  
**Thời điểm khóa:** 04/10/2026  
**Dữ liệu đánh giá:** 21 Checkpoints (7 Họ $\times$ 3 Seeds: 42, 43, 44)  
**Tập Test:** `data_processed/amis_mandarin/test.csv` (575 dòng, FP32, beam 4, SacreBLEU 2.6.0 zh)  
**Gia đình giả thuyết thống kê:** 30 Prespecified Secondary Hypotheses (5 Cặp $\times$ 3 Seeds $\times$ 2 Metrics) $\times$ 10,000 Paired Corpus Bootstrap Draws $\times$ Holm-Bonferroni Correction.

---

## 1. Bảng Điểm Test Tổng hợp Toàn diện (21 Checkpoints)

| Họ mô hình (Family) | Seed | BLEU (zh) | chrF++ (word-2) | chrF0 (char-only) | Ghi chú & Nguồn Weights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | 42 | 19.7281 | 18.7138 | 18.3973 | Confirmation Phase C |
| | 43 | 18.1378 | 13.1592 | 17.0904 | Confirmation Phase C (Outlier chrF++ dòng 170) |
| | 44 | 20.3000 | 18.9017 | 18.6478 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.3886 \pm 1.1203$** | **$16.9249 \pm 3.2626$** | **$18.0452 \pm 0.8362$** | |
| **CLRR-only detached** | 42 | 19.6403 | 18.5639 | 18.2504 | Confirmation Phase C |
| | 43 | 18.9420 | 18.0823 | 17.7017 | Confirmation Phase C |
| | 44 | 20.1485 | 18.9055 | 18.6584 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.5769 \pm 0.6057$** | **$18.5172 \pm 0.4136$** | **$18.2035 \pm 0.4801$** | |
| **Matched LSR** ($\lambda=0.10$) | 42 | 19.7865 | 18.6797 | 18.3581 | Replica 42 mới (Final Evaluation) |
| | 43 | 19.4879 | 18.6488 | 18.2581 | Confirmation Phase C |
| | 44 | 19.3284 | 18.3633 | 17.9957 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.5343 \pm 0.2325$** | **$18.5639 \pm 0.1745$** | **$18.2040 \pm 0.1871$** | |
| **Tuned LSR** ($\lambda=0.03$) | 42 | 20.6257 | 19.1889 | 19.0288 | Screening Phase B |
| | 43 | 19.6747 | 14.1556 | 18.4188 | Confirmation Phase C (Outlier chrF++ dòng 170) |
| | 44 | 19.3858 | 18.5324 | 18.1573 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.8954 \pm 0.6487$** | **$17.2923 \pm 2.7362$** | **$18.5350 \pm 0.4472$** | |
| **Original Full detached** ($\alpha=0.10, \lambda=0.10$) | 42 | 19.9566 | 18.8878 | 18.6626 | Replica 42 mới (Final Evaluation) |
| | 43 | 19.6123 | 18.4753 | 18.1905 | Confirmation Phase C |
| | 44 | 19.7371 | 18.6065 | 18.2613 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.7687 \pm 0.1743$** | **$18.6565 \pm 0.2107$** | **$18.3715 \pm 0.2546$** | |
| **Full source-attached** ($\alpha=0.10, \lambda=0.10$, detach=False) | 42 | 19.3531 | 18.5412 | 18.2240 | Phase C2 mới |
| | 43 | 19.7774 | 18.5131 | 18.2104 | Phase C2 mới |
| | 44 | 19.5667 | 18.6355 | 18.3571 | Phase C2 mới |
| *Trung bình $\pm$ Std* | | **$19.5658 \pm 0.2122$** | **$18.5633 \pm 0.0641$** | **$18.2638 \pm 0.0811$** | |
| **Full detached $\lambda=0.03$** ($\alpha=0.10, \lambda=0.03$) | 42 | 19.2127 | 18.3924 | 18.0859 | Step F1 mới (Follow-up) |
| | 43 | 19.1189 | 18.2179 | 17.8505 | Step F1 mới (Follow-up) |
| | 44 | 18.9245 | 18.0868 | 17.6579 | Step F1 mới (Follow-up) |
| *Trung bình $\pm$ Std* | | **$19.0854 \pm 0.1470$** | **$18.2324 \pm 0.1533$** | **$17.8648 \pm 0.2144$** | |

---

## 2. Kết quả 5 Cặp So sánh Thứ cấp Đã Khóa (F2: 30 Hypotheses)

| Cặp so sánh (Challenger vs Control) | Metric | Seed 42 | Seed 43 | Seed 44 | Mean Delta | Std Paired Delta | Số seed dương | Holm-30 Sig ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CLRR-only vs Baseline** | BLEU | $-0.0878$ | $+0.8042$ | $-0.1516$ | $+0.1883$ | $0.5343$ | 1 / 3 | **False** ($p=1.0$) |
| | chrF++ | $-0.1499$ | $+4.9231^*$ | $+0.0038$ | $+1.5923$ | $2.8856$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF0 | $-0.1469$ | $+0.6113$ | $+0.0107$ | $+0.1583$ | $0.4001$ | 2 / 3 | — |
| **2. Full $\lambda=0.10$ vs CLRR-only** | BLEU | $+0.3163$ | $+0.6702$ | $-0.4113$ | $+0.1917$ | $0.5514$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF++ | $+0.3239$ | $+0.3930$ | $-0.2990$ | $+0.1393$ | $0.3812$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF0 | $+0.4123$ | $+0.4888$ | $-0.3971$ | $+0.1680$ | $0.4909$ | 2 / 3 | — |
| **3. Full $\lambda=0.10$ (detached) vs Full attached** | BLEU | $+0.6035$ | $-0.1652$ | $+0.1704$ | $+0.2029$ | $0.3854$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF++ | $+0.3466$ | $-0.0378$ | $-0.0290$ | $+0.0933$ | $0.2194$ | 1 / 3 | **False** ($p=1.0$) |
| | chrF0 | $+0.4387$ | $-0.0199$ | $-0.0959$ | $+0.1076$ | $0.2892$ | 1 / 3 | — |
| **4. Full $\lambda=0.03$ vs Tuned LSR $\lambda=0.03$** | BLEU | $-1.4130$ | $-0.5557$ | $-0.4613$ | **$-0.8100$** | $0.5243$ | **0 / 3** | **False** ($p=0.267$) |
| | chrF++ | $-0.7965$ | $+4.0623^*$ | $-0.4456$ | $+0.9401$ | $2.7096$ | 1 / 3 | **False** ($p=0.458$) |
| | chrF0 | $-0.9429$ | $-0.5683$ | $-0.4995$ | **$-0.6702$** | $0.2387$ | **0 / 3** | — |
| **5. Full $\lambda=0.10$ vs Full $\lambda=0.03$** | BLEU | $+0.7439$ | $+0.4933$ | $+0.8126$ | **$+0.6833$** | $0.1680$ | **3 / 3** | **False** ($p=1.0$) |
| | chrF++ | $+0.4955$ | $+0.2574$ | $+0.5196$ | **$+0.4242$** | $0.1449$ | **3 / 3** | **False** ($p=1.0$) |
| | chrF0 | $+0.5767$ | $+0.3400$ | $+0.6034$ | **$+0.5067$** | $0.1450$ | **3 / 3** | — |

*\* Ghi chú: Chênh lệch chrF++ lớn ở seed 43 xuất phát từ hiện tượng chia cho 0 ở word-bigram tại câu 170 (xem phân tích sensitivity bên dưới).*

---

## 3. Đánh giá Đối chiếu Cổng Dự án (Project Decision Gates)

Đối chiếu với các tiêu chí định trước tại Mục 14.7 của Handoff:

1. **Tiêu chí 1: Full − Baseline $\ge +0.5$ BLEU?**
   - **Kết quả:** Trên tập Test, Full $\lambda=0.10$ đạt $19.7687$ vs Baseline $19.3886$ ($\Delta = +0.3800$ BLEU). Trên tập Validation, Full kém Baseline trung bình $-0.4498$ BLEU.
   - **Đánh giá: KHÔNG ĐẠT (NOT MET).**
2. **Tiêu chí 2: Full − Matched LSR $\ge +0.3$ BLEU?**
   - **Kết quả:** Full $\lambda=0.10$ đạt $19.7687$ vs Matched LSR $19.5343$ ($\Delta = +0.2344$ BLEU).
   - **Đánh giá: KHÔNG ĐẠT (NOT MET).**
3. **Tiêu chí 3: Thắng Matched LSR trên 3/3 seeds?**
   - **Kết quả:** Seed 42 ($+0.1701$), Seed 43 ($+0.1243$), Seed 44 ($+0.4087$). Đạt 3/3 seeds trên test, nhưng gap nhỏ ($< 0.3$).
   - **Đánh giá: ĐẠT MỘT PHẦN.**
4. **Tiêu chí 4: Full − Tuned LSR ($\lambda=0.03$) mean $> 0$?**
   - **Kết quả:**
     - Với cấu hình Full gốc ($\lambda=0.10$): $19.7687$ vs $19.8954$ $\rightarrow$ **thua $-0.1267$ BLEU**.
     - Với cấu hình Full match tuned LSR ($\lambda=0.03$): $19.0854$ vs $19.8954$ $\rightarrow$ **thua $-0.8100$ BLEU** (0/3 seeds thắng).
   - **Đánh giá: HOÀN TOÀN KHÔNG ĐẠT (COMPLETELY FAILED).**
5. **Tiêu chí 5: Ý nghĩa thống kê sau hiệu chỉnh kiểm định bội (Holm-Bonferroni)?**
   - **Kết quả:** Cả 30 kiểm định đều có $p_{adj} > 0.05$ (không có bất kỳ cặp nào đạt ý nghĩa thống kê ở mức $\alpha=0.05$).
   - **Đánh giá: KHÔNG ĐẠT (NOT MET).**

---

## 4. Giải đáp các Câu hỏi Nghiên cứu Then chốt

### Câu hỏi 1: Thao tác Detach có đang kìm hãm CLRR không?
**Trả lời: KHÔNG.**
- Khi giữ nguyên routing và cho phép gradient truyền ngược qua source cache (`full_source_attached`), điểm số test thực tế là **$19.5658$ BLEU**, thấp hơn bản detached (**$19.7687$ BLEU**) một khoảng $-0.2029$ BLEU.
- Về mặt cơ chế, probe F3 cho thấy `full_source_attached` có tỷ số gradient LSR hơi cao hơn ($2.05 \times 10^{-3}$ vs $1.61 \times 10^{-3}$) nhưng cosine alignment tương đương ($0.801$ vs $0.802$) và effective rank tương đương ($384.9$ vs $385.3$).
- Do đó, thao tác `detach()` không phải là nguyên nhân làm CLRR không vượt được Tuned LSR.

### Câu hỏi 2: Có nên hạ $\lambda=0.03$ cho Full CLRR như đã làm với Tuned LSR không?
**Trả lời: KHÔNG NÊN.**
- Cặp so sánh số 5 chứng minh rằng với Full CLRR, $\lambda=0.10$ vượt trội hoàn toàn $\lambda=0.03$ ở cả 3 seeds ($+0.6833$ BLEU, $+0.4242$ chrF++, $+0.5067$ chrF0).
- Khi giảm $\lambda$ xuống 0.03, bản Full tụt xuống $19.0854$ BLEU và thua Tuned LSR tới $-0.8100$ BLEU trên toàn bộ các seeds. CLRR cần lực regularize đủ mạnh ($\lambda=0.10$) để neo biểu diễn; khi $\lambda$ quá nhỏ, cấu trúc skip connection không phát huy tác dụng.

### Câu hỏi 3: Độ nhạy của chrF++ (Hiện tượng Outlier Dòng 170)
- Kiểm định độ nhạy Leave-one-out tại dòng 170 trên tập test Amis (row170 có reference word-bigram; một trong3rows126/170/468) cho thấy điểm chrF++ bị lệch cục bộ ở Seed 43 của Baseline và Tuned LSR.
- Tuy nhiên, khi đối chiếu với metric **chrF0 (character-only)** không phụ thuộc vào word-bigram, thứ hạng giữa các mô hình vẫn hoàn toàn nhất quán:
  - Tuned LSR: $18.5350$
  - Full $\lambda=0.10$: $18.3715$
  - Full attached: $18.2638$
  - Matched LSR: $18.2040$
  - CLRR-only: $18.2035$
  - Baseline: $18.0452$
  - Full $\lambda=0.03$: $17.8648$
- Thứ hạng chrF0 xác nhận Tuned LSR ($\lambda=0.03$) đứng đầu, tiếp theo là Full ($\lambda=0.10$), và Full $\lambda=0.03$ đứng cuối cùng.

---

## 5. Quyết định Chiến lược Dự án (Final Strategic Verdict)

### **QUYẾT ĐỊNH: NO-GO CHO VIỆC TUYÊN BỐ VƯỢT TRỘI HỆ THỐNG HOẶC MỞ RỘNG ĐA NGÔN NGỮ (TURKISH/NLLB)**

1. **Lý do:**
   - Phương pháp CLRR (`Full detached`) không vượt qua được mốc đối chứng đã được tinh chỉnh tối ưu (`Tuned LSR`). Cụ thể, Full thua Tuned LSR $-0.13$ BLEU và $-0.16$ chrF0.
   - Thử nghiệm bỏ detach (`Attached control`) không mang lại cải thiện.
   - Thử nghiệm điều chỉnh $\lambda=0.03$ cho Full thất bại nghiêm trọng (thua $-0.81$ BLEU).
   - Kiểm định thống kê 30 cặp giả thuyết sau hiệu chỉnh Holm-Bonferroni cho thấy không có chênh lệch nào đạt mức ý nghĩa thống kê $p < 0.05$.
2. **Khuyến nghị Định vị lại Bài báo (Paper Framing):**
   - **Không claim:** Không tuyên bố CLRR tạo ra bước nhảy vọt vượt trội hệ thống so với representation regularization thuần túy (`Tuned LSR`).
   - **Tập trung claim:** Định vị bài báo như một nghiên cứu cơ chế và động lực học biểu diễn sâu sắc:
     - Chứng minh bằng thực nghiệm: CLRR đóng vai trò là **bộ đệm bảo toàn chiều biểu diễn (representational capacity buffer)**, tăng effective rank ở tầng trên cùng của encoder (Layer 12) thêm **+34% (từ ~102 lên ~138)**.
     - Kiểm định can thiệp lúc inference (F4 corrected) xác nhận routing tác động trực tiếp lên decoding (làm đổi **89% câu dịch** và giảm Cross-Entropy loss trên 100% các seed).
     - Báo cáo minh bạch các giới hạn tương tác khách quan: CLRR cần $\lambda$ đủ lớn ($\sim 0.10$) để khai thác không gian biểu diễn mở rộng, và khi giảm $\lambda=0.03$ thì CLRR mất ưu thế. Thao tác `detach()` đã được kiểm chứng độc lập là không gây hại cho mô hình.
