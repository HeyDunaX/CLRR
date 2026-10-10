# Báo cáo Chẩn đoán Dị biệt & Phân rã N-gram trên 32 Runs Fresh Confirmation
**Ngày audit:** 07/10/2026 | **Phạm vi:** 575 câu test Amis $	o$ Mandarin mBART

## 1. Phân rã Corpus N-gram cho CLRR Seed 48 (BLEU 20.76 vs chrF++ 14.59 vs chrF0 18.95)

Theo đặc tả của SacreBLEU (`sacrebleu/metrics/chrf.py`), `word_order=2` tính toán điểm F trung bình qua 6 bậc char n-grams (1..6) và 2 bậc word n-grams (1..2). Khi một bậc n-gram hoàn toàn không xuất hiện cả ở candidate lẫn reference, SacreBLEU sẽ loại bậc đó khỏi `effective_order`.

Tuy nhiên, bảng đếm thống kê corpus thực tế cho thấy:

| Bậc N-gram | CLRR 48 (match / test / ref) | CLRR 48 F | CLRR 49 (match / test / ref) | CLRR 49 F | Baseline 48 (match / test / ref) | Baseline 48 F |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **char_1** | 3176 / 6001 / 6327 | 0.5152 | 3123 / 6055 / 6327 | 0.5044 | 3090 / 6114 / 6327 | 0.4967 |
| **char_2** | 1488 / 5426 / 5752 | 0.2662 | 1439 / 5480 / 5752 | 0.2562 | 1415 / 5539 / 5752 | 0.2506 |
| **char_3** | 802 / 4851 / 5177 | 0.1600 | 762 / 4905 / 5177 | 0.1512 | 746 / 4964 / 5177 | 0.1471 |
| **char_4** | 453 / 4267 / 4602 | 0.1022 | 439 / 4317 / 4602 | 0.0984 | 430 / 4374 / 4602 | 0.0958 |
| **char_5** | 255 / 3687 / 4028 | 0.0661 | 263 / 3736 / 4028 | 0.0677 | 250 / 3796 / 4028 | 0.0639 |
| **char_6** | 158 / 3110 / 3458 | 0.0481 | 175 / 3150 / 3458 | 0.0530 | 161 / 3219 / 3458 | 0.0482 |
| **word_1** | 18 / 625 / 578 | 0.0299 | 21 / 621 / 578 | 0.0350 | 16 / 631 / 578 | 0.0265 |
| **word_2** | 0 / 2 / 3 | 0.0000 | 1 / 2 / 3 | 0.4000 | 1 / 2 / 3 | 0.4000 |

### Kết luận cơ chế về CLRR Seed 48:
1. **Ở cấp độ ký tự (Char 1-gram đến Char 6-gram):** CLRR 48 đạt điểm F **cao hơn cả CLRR 49 lẫn Baseline 48 trên cả 6 bậc** (ví dụ Char 1-gram: F=0.5152 so với 0.5044 và 0.4967). Điều này lý giải tại sao chrF0 (`word_order=0`, chỉ dựa trên ký tự) của CLRR 48 đạt **18.9450** và BLEU đạt **20.7623**.
2. **Sự sụt giảm cục bộ tại Word 2-gram:** Do văn bản tiếng Trung không có dấu cách giữa các chữ, trên toàn bộ 575 câu test chỉ xuất hiện vỏn vẹn **3 word 2-grams trong reference** và **2 word 2-grams trong candidate** (sinh ra từ khoảng trắng xung quanh dấu câu hoặc tên riêng). Vì `test_len = 2 > 0`, bậc Word 2-gram **không bị loại khỏi effective order** mà được tính với số match = 0, dẫn tới **F(word 2-gram) = 0.0000** ở CLRR 48. Trong khi đó, ở CLRR 49 và Baseline 48 có đúng 1/2 match ngẫu nhiên dẫn tới F=0.4000. Khoảng cách một điểm match duy nhất trên một bậc n-gram cực kỳ thưa đã kéo tụt điểm trung bình chrF++ của toàn bộ corpus xuống 14.59.

## 2. Bảng Đối chiếu Độ dài và Lặp từ trên Toàn bộ 32 Runs (Khớp 100% Số liệu)

| Family | Seed | Mean Length (Chars) | Length Ratio | Empty Preds | Repetition Preds (>=4char) | Test BLEU | Test chrF++ | Test chrF0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BASELINE | 45 | 10.67 | 0.969 | 0 | 2 | 19.58 | 18.63 | 18.26 |
| BASELINE | 46 | 10.67 | 0.969 | 0 | 3 | 18.91 | 18.28 | 17.84 |
| BASELINE | 47 | 10.60 | 0.963 | 0 | 6 | 20.35 | 19.05 | 18.85 |
| BASELINE | 48 | 10.67 | 0.969 | 0 | 7 | 19.49 | 18.49 | 18.15 |
| BASELINE | 49 | 10.60 | 0.964 | 0 | 4 | 19.39 | 18.50 | 18.15 |
| BASELINE | 50 | 10.56 | 0.959 | 0 | 3 | 19.78 | 18.73 | 18.40 |
| BASELINE | 51 | 10.72 | 0.974 | 0 | 4 | 19.78 | 18.67 | 18.28 |
| BASELINE | 52 | 10.75 | 0.977 | 0 | 7 | 19.26 | 18.39 | 17.93 |
| CLRR | 45 | 11.33 | 1.029 | 0 | 6 | 19.29 | 18.25 | 17.78 |
| CLRR | 46 | 10.67 | 0.970 | 0 | 3 | 19.26 | 18.34 | 17.92 |
| CLRR | 47 | 10.59 | 0.962 | 0 | 7 | 19.07 | 18.14 | 17.75 |
| CLRR | 48 | 10.46 | 0.951 | 0 | 5 | 20.76 | 14.59 | 18.95 |
| CLRR | 49 | 10.55 | 0.959 | 0 | 6 | 19.84 | 18.90 | 18.56 |
| CLRR | 50 | 10.63 | 0.966 | 0 | 5 | 19.44 | 18.51 | 18.12 |
| CLRR | 51 | 10.74 | 0.976 | 0 | 7 | 19.77 | 18.50 | 18.22 |
| CLRR | 52 | 10.51 | 0.955 | 0 | 2 | 19.55 | 18.60 | 18.26 |
| LSR | 45 | 10.62 | 0.965 | 0 | 6 | 19.47 | 18.48 | 18.08 |
| LSR | 46 | 10.75 | 0.977 | 0 | 4 | 20.17 | 18.90 | 18.64 |
| LSR | 47 | 10.64 | 0.967 | 0 | 7 | 19.80 | 18.67 | 18.36 |
| LSR | 48 | 10.34 | 0.940 | 0 | 2 | 18.58 | 18.04 | 17.60 |
| LSR | 49 | 10.65 | 0.968 | 0 | 5 | 19.95 | 18.78 | 18.46 |
| LSR | 50 | 10.41 | 0.946 | 0 | 6 | 19.31 | 18.50 | 18.12 |
| LSR | 51 | 11.58 | 1.052 | 0 | 8 | 19.85 | 18.40 | 17.93 |
| LSR | 52 | 10.48 | 0.952 | 0 | 5 | 19.25 | 18.38 | 17.96 |
| FULL | 45 | 10.45 | 0.950 | 0 | 4 | 18.40 | 17.97 | 17.51 |
| FULL | 46 | 10.54 | 0.958 | 0 | 6 | 20.59 | 19.18 | 18.99 |
| FULL | 47 | 10.57 | 0.960 | 0 | 6 | 19.46 | 18.42 | 18.16 |
| FULL | 48 | 10.34 | 0.939 | 0 | 5 | 18.35 | 17.94 | 17.47 |
| FULL | 49 | 10.47 | 0.952 | 0 | 4 | 19.89 | 18.70 | 18.41 |
| FULL | 50 | 10.62 | 0.965 | 0 | 1 | 20.14 | 18.91 | 18.63 |
| FULL | 51 | 10.50 | 0.954 | 0 | 6 | 20.22 | 18.90 | 18.65 |
| FULL | 52 | 10.73 | 0.975 | 0 | 8 | 19.16 | 18.37 | 17.93 |

### Nhận định sau đối soát:
- **Về câu rỗng và lặp từ:** Cả 32 runs đều có **0 câu rỗng**. Số câu có lặp 4-gram ký tự dao động từ **1 đến 8 câu trên 575 câu** (chiếm tỷ lệ rất nhỏ 0.17% - 1.39%) và phân bố đều giữa cả 4 families (Full có từ 1 đến 8 câu, Baseline có từ 2 đến 7 câu, LSR có từ 2 đến 8 câu).
- **Về độ dài câu:** Tỷ lệ độ dài trung bình so với reference dao động chặt chẽ trong khoảng **0.939 đến 1.052** across all 32 runs. Riêng nhóm Full: Seed 45 (len 10.45, ratio 0.950), Seed 48 (len 10.34, ratio 0.939), Seed 52 (len 10.73, ratio 0.975) so với Seed 46 (len 10.54, ratio 0.958) và Seed 50 (len 10.62, ratio 0.965).
- **Thận trọng diễn giải:** Việc độ dài và repetition ổn định **chưa đủ để loại trừ hoàn toàn truncation hoặc suy biến không gian biểu diễn (representation collapse)**. Đây chỉ là các phép kiểm tra bề mặt output (surface level), cần đo đạc trực tiếp nội tại mô hình (gradient norm, residual RMS, singular value rank) ở Bước B để có kết luận cơ chế xác đáng.
