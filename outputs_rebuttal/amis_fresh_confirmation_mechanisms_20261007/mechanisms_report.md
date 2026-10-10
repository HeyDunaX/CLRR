# Báo cáo Cơ chế Thực nghiệm Bước B trên 32 Runs Fresh Confirmation
**Ngày đo đạc:** 07/10/2026 | **Môi trường:** NVIDIA A100-SXM4-40GB BF16 | **Dữ liệu:** Amis $	o$ Mandarin mBART

## 1. Tương tác Gradient: Cross-Entropy (CE) vs Weighted LSR

Đo trên 8 mini-batches (batch size 4, 32 ví dụ) từ tập validation cho từng mô hình:

| Family | CE Norm (All) | Weighted LSR Norm | LSR/CE Ratio | Cosine Similarity | Negative Cosine Batches (/8) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | 36.7435 ± 0.8533 | 0.1365 ± 0.0041 | 0.0038 ± 0.0001 | +0.0659 ± 0.0060 | 0.4/8 |
| **CLRR** | 32.7104 ± 1.1084 | 0.0885 ± 0.0008 | 0.0027 ± 0.0001 | +0.0527 ± 0.0070 | 0.1/8 |
| **LSR** | 34.3513 ± 0.6348 | 0.0831 ± 0.0047 | 0.0025 ± 0.0002 | +0.0684 ± 0.0096 | 0.1/8 |
| **FULL** | 31.4335 ± 1.7514 | 0.0565 ± 0.0061 | 0.0018 ± 0.0003 | +0.0579 ± 0.0084 | 0.2/8 |

## 2. Phổ Biểu diễn và Nguy cơ Thu hẹp Biểu diễn (Representation Spectrum & Collapse)

Đo bằng SVD trên biểu diễn encoder pooled của 575 câu test:

| Family | Source Eff Rank | Source Top-1 Var % | Source Top-10 Var % | Source Pairwise Cos | Target Eff Rank | Source-Target Alignment Cos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | 391.58 ± 2.01 | 4.67% ± 0.20% | 29.78% ± 0.55% | 0.4239 ± 0.0037 | 391.18 ± 1.89 | 0.3499 ± 0.0064 |
| **CLRR** | 387.90 ± 1.35 | 4.42% ± 0.18% | 28.59% ± 0.37% | 0.3967 ± 0.0075 | 389.50 ± 1.71 | 0.3267 ± 0.0081 |
| **LSR** | 388.71 ± 0.50 | 6.29% ± 0.22% | 31.98% ± 0.21% | 0.7095 ± 0.0147 | 381.65 ± 0.42 | 0.7997 ± 0.0122 |
| **FULL** | 386.54 ± 1.49 | 5.40% ± 0.21% | 29.90% ± 0.45% | 0.6969 ± 0.0330 | 383.64 ± 1.84 | 0.7828 ± 0.0272 |

## 3. Cường độ Residual và Đóng góp của Tuyến Rewire (Layer Residuals & Route On/Off)

### Đóng góp của Tuyến Rewiring (Tắt/Bật Routing tại Validation):

| Family | Route ON Val BLEU | Route OFF Val BLEU | $\Delta$ BLEU (ON - OFF) | Route ON chrF++ | Route OFF chrF++ | $\Delta$ chrF++ (ON - OFF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLRR** | 16.75 ± 0.48 | 14.08 ± 0.41 | **+2.6721 ± 0.4019** | 14.18 ± 0.33 | 12.00 ± 0.26 | **+2.1776 ± 0.2701** |
| **FULL** | 16.75 ± 0.49 | 14.11 ± 0.48 | **+2.6404 ± 0.5022** | 14.18 ± 0.30 | 11.88 ± 0.47 | **+2.3041 ± 0.5607** |

### Tỷ số Residual RMS theo Từng Layer (Source Encoder, Layers 3..12):

| Layer | CLRR Residual/Current RMS Ratio | Full Residual/Current RMS Ratio | CLRR Skip-Current Cosine | Full Skip-Current Cosine |
| :---: | :---: | :---: | :---: | :---: |
| Layer 3 | 0.0610 | 0.0614 | 0.8002 | 0.8100 |
| Layer 4 | 0.0636 | 0.0641 | 0.9095 | 0.9114 |
| Layer 5 | 0.0699 | 0.0702 | 0.9254 | 0.9228 |
| Layer 6 | 0.0749 | 0.0750 | 0.9153 | 0.9111 |
| Layer 7 | 0.0788 | 0.0788 | 0.9125 | 0.9070 |
| Layer 8 | 0.0820 | 0.0816 | 0.9221 | 0.9122 |
| Layer 9 | 0.0839 | 0.0825 | 0.9195 | 0.9069 |
| Layer 10 | 0.0233 | 0.0225 | 0.2912 | 0.2876 |
| Layer 11 | 0.0126 | 0.0126 | 0.2237 | 0.2278 |
| Layer 12 | 0.0338 | 0.0349 | 0.9627 | 0.9648 |

