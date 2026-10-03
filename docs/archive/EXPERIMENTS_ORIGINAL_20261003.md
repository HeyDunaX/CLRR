# Thực nghiệm gốc — số liệu của `clrr_main.tex` trước audit

Đây là **bản ghi số gốc dùng trong paper**, trước việc chuẩn hóa lại reference,
generate lại checkpoint và các phép đo bổ sung. Không đưa số mới vào file này.
Các kết quả và nhận xét sau xác minh nằm tại [EXPERIMENTS_UPDATED.md](EXPERIMENTS_UPDATED.md).

## Nguồn và phạm vi

- Nguồn trực tiếp: [clrr_main.tex](clrr_main.tex), khớp byte với bản paper lưu trước audit.
- SHA-256 paper: `9eef948c389a9936817ad3c59fd01e018eabcc276d6f02609be887dacb1f3aeb`.
- Bản paper đóng băng: [clrr_main_original.tex](../outputs_rebuttal/documentation_consolidation_20261003/clrr_main_original.tex).
- Toàn bộ **10 bảng** bên dưới trích từ paper, giữ nguyên con số, độ làm tròn và dấu significance.
- “Gốc” ở đây là **bản paper đã có sẵn trước audit**, có cả Strong LoRA đã được tác giả đưa vào bản đó; không phải bản repo đầu tiên chưa chạy Strong LoRA.
- Asháninka, ByT5, kiểm tra precision và kết quả mới không có trong các bảng paper gốc này; xem file sau.
- Việc ghi lại một số lịch sử **không đồng nghĩa số đó đã được xác minh**. Trạng thái xác minh, số sửa và các claim không còn được hỗ trợ được ghi ở file sau.

## Cấu hình được paper gốc công bố

- Amis→Mandarin: train/validation/test **4600/576/575**; seed42.
- Effective batch128; AdamW, weight decay0.01 theo mô tả paper.
- Full-tuning mBART/NLLB: LR5e-5; mT5: LR3e-4; warmup0.06, tối đa20epochs.
- Strong LoRA A: r16/alpha32/dropout0.05, all-linear, LR2e-4; B mở embeddings và LR1e-4.
- BF16 và TF32 trên A100; chọn checkpoint bằng validation chrF++; test beam4, length penalty1.
- Paper báo BLEUzh, chrF++ raw(w=2) và chrF++ sau TokenizerZh. Giữ riêng các cột như bản gốc.
- Các mô tả trên được lưu **theo paper**; những khác biệt với cấu hình thực tế nằm trong file sau.


## 0. Thống kê dữ liệu được paper công bố

Nguồn LaTeX: `tab:dataset_stats`.

| Split | Sentence Pairs | Amis Tokens | Chinese Characters |
| --- | --- | --- | --- |
| Train | 4,600 | 42,180 | 49,812 |
| Validation | 576 | 5,314 | 6,290 |
| Test | 575 | 5,286 | 6,244 |
| Total | 5,751 | 52,780 | 62,346 |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{lrrr}
\toprule
\textbf{Split} & \textbf{Sentence Pairs} & \textbf{Amis Tokens} & \textbf{Chinese Characters} \\
\midrule
Train & 4,600 & 42,180 & 49,812 \\
Validation & 576 & 5,314 & 6,290 \\
Test & 575 & 5,286 & 6,244 \\
\midrule
Total & 5,751 & 52,780 & 62,346 \\
\bottomrule
\end{tabular}
```

</details>

## 1. Đối sánh chính: mBART và NLLB

Nguồn LaTeX: `tab:main_comparative_matrix`.

| Backbone Architecture | Method | Reference / Source | $\theta_{train}$ | $\Delta\theta_{add}$ | BLEU (zh) $\uparrow$ | chrF++ (w=2) $\uparrow$ | chrF++ (Zh) $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mBART-50 | BitFit (Bias-only) | ben-zaken-etal-2022-bitfit (ACL 2022) | 336k (0.05%) | 0 | 0.3511 | 2.8464 | 4.4408 |
| mBART-50 | Narrow LoRA ($r=8, q/v$) | hu-etal-2022-lora (ICLR 2022) | 1.18M (0.19%) | +1.18M | 3.3284 | 5.1583 | 8.0858 |
| mBART-50 | Strong LoRA A (All-Linear) | hu-etal-2022-lora (Expanded) | 8.65M (1.42%) | +8.65M | 10.4193 | 9.8008 | 15.2118 |
| mBART-50 | Strong LoRA B (+Embeddings) | hu-etal-2022-lora (Expanded) | 264.7M (42.73%) | +8.65M | 14.0432 | 11.4901 | 17.6304 |
| mBART-50 | Standard Fine-Tuning | Official Baseline | 611M (100%) | 0 | 19.6106 | 14.0538 | 22.7214 |
| mBART-50 | Middle-Layer Alignment | liu-niehues-2025-middle (arXiv 2025) | 611M (100%) | 0 | 19.7243 | 15.5253 | 22.8105 |
| mBART-50 | CLRR-Dec + LSR (Ours) | Proposed (Target Decoder) | 611M (100%) | 0 | 20.3986 | 16.1939$^\dagger$ | 23.1542 |
| mBART-50 | CLRR-Enc + LSR (Ours) | Proposed (Main Encoder) | 611M (100%) | 0 | 20.3927 | 19.0839$^\dagger$ | 23.5912 |
| NLLB-200 | BitFit (Bias-only) | ben-zaken-etal-2022-bitfit (ACL 2022) | 131k (0.02%) | 0 | 1.0750 | 2.9643 | 5.7081 |
| NLLB-200 | Narrow LoRA ($r=8, q/v$) | hu-etal-2022-lora (ICLR 2022) | 1.18M (0.19%) | +1.18M | 3.5383 | 4.9407 | 9.1651 |
| NLLB-200 | Strong LoRA A (All-Linear) | hu-etal-2022-lora (Expanded) | 8.65M (1.39%) | +8.65M | 11.4625 | 9.5004 | 16.6314 |
| NLLB-200 | Strong LoRA B (+Embeddings) | hu-etal-2022-lora (Expanded) | 271.0M (43.45%) | +8.65M | 9.8024 | 8.4010 | 14.8131 |
| NLLB-200 | Standard Fine-Tuning | Official Baseline | 615M (100%) | 0 | 13.5001 | 10.4389 | 18.0885 |
| NLLB-200 | Middle-Layer Alignment | liu-niehues-2025-middle (arXiv 2025) | 615M (100%) | 0 | 13.9648 | 10.6596 | 18.4448 |
| NLLB-200 | CLRR-Dec + LSR (Ours) | Proposed (Target Decoder) | 615M (100%) | 0 | 13.4194 | 10.4559 | 18.0274 |
| NLLB-200 | CLRR-Enc + LSR (Ours) | Proposed (Main Encoder) | 615M (100%) | 0 | 14.2820$^\dagger$ | 10.9152$^\dagger$ | 18.7482 |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{lllcccccc}
\toprule
\textbf{Backbone Architecture} & \textbf{Method} & \textbf{Reference / Source} & $\theta_{\text{train}}$ & $\Delta\theta_{\text{add}}$ & \textbf{BLEU (zh)} $\uparrow$ & \textbf{chrF++ (w=2)} $\uparrow$ & \textbf{chrF++ (Zh)} $\uparrow$ \\
\midrule
\multirow{8}{*}{\shortstack[l]{\textbf{facebook/mbart-large-50}\\(12 enc / 12 dec, 611M)}} 
 & BitFit (Bias-only) & \citeauthor{ben-zaken-etal-2022-bitfit} (\textbf{ACL 2022}) & 336k (0.05\%) & 0 & 0.3511 & 2.8464 & 4.4408 \\
 & Narrow LoRA ($r=8, q/v$) & \citeauthor{hu-etal-2022-lora} (\textbf{ICLR 2022}) & 1.18M (0.19\%) & +1.18M & 3.3284 & 5.1583 & 8.0858 \\
 & Strong LoRA A (All-Linear) & \citeauthor{hu-etal-2022-lora} (Expanded) & 8.65M (1.42\%) & +8.65M & 10.4193 & 9.8008 & 15.2118 \\
 & Strong LoRA B (+Embeddings) & \citeauthor{hu-etal-2022-lora} (Expanded) & 264.7M (42.73\%) & +8.65M & 14.0432 & 11.4901 & 17.6304 \\
 & Standard Fine-Tuning & Official Baseline & 611M (100\%) & 0 & 19.6106 & 14.0538 & 22.7214 \\
 & Middle-Layer Alignment & \citeauthor{liu-niehues-2025-middle} (\textbf{arXiv 2025}) & 611M (100\%) & 0 & 19.7243 & 15.5253 & 22.8105 \\
 & CLRR-Dec + LSR (Ours) & Proposed (Target Decoder) & 611M (100\%) & 0 & \textbf{20.3986} & 16.1939$^\dagger$ & 23.1542 \\
 & \textbf{CLRR-Enc + LSR (Ours)} & Proposed (Main Encoder) & 611M (100\%) & \textbf{0} & 20.3927 & \textbf{19.0839}$^\dagger$ & \textbf{23.5912} \\
\midrule
\multirow{8}{*}{\shortstack[l]{\textbf{facebook/nllb-200-distilled-600M}\\(12 enc / 12 dec, 615M)}} 
 & BitFit (Bias-only) & \citeauthor{ben-zaken-etal-2022-bitfit} (\textbf{ACL 2022}) & 131k (0.02\%) & 0 & 1.0750 & 2.9643 & 5.7081 \\
 & Narrow LoRA ($r=8, q/v$) & \citeauthor{hu-etal-2022-lora} (\textbf{ICLR 2022}) & 1.18M (0.19\%) & +1.18M & 3.5383 & 4.9407 & 9.1651 \\
 & Strong LoRA A (All-Linear) & \citeauthor{hu-etal-2022-lora} (Expanded) & 8.65M (1.39\%) & +8.65M & 11.4625 & 9.5004 & 16.6314 \\
 & Strong LoRA B (+Embeddings) & \citeauthor{hu-etal-2022-lora} (Expanded) & 271.0M (43.45\%) & +8.65M & 9.8024 & 8.4010 & 14.8131 \\
 & Standard Fine-Tuning & Official Baseline & 615M (100\%) & 0 & 13.5001 & 10.4389 & 18.0885 \\
 & Middle-Layer Alignment & \citeauthor{liu-niehues-2025-middle} (\textbf{arXiv 2025}) & 615M (100\%) & 0 & 13.9648 & 10.6596 & 18.4448 \\
 & CLRR-Dec + LSR (Ours) & Proposed (Target Decoder) & 615M (100\%) & 0 & 13.4194 & 10.4559 & 18.0274 \\
 & \textbf{CLRR-Enc + LSR (Ours)} & Proposed (Main Encoder) & 615M (100\%) & \textbf{0} & \textbf{14.2820}$^\dagger$ & \textbf{10.9152}$^\dagger$ & \textbf{18.7482} \\
\bottomrule
\end{tabular}
```

</details>

## 2. Bóc tách thành phần mBART

Nguồn LaTeX: `tab:mbart_ablation`.

| Configuration | $\Delta\theta$ | BLEU | chrF++ | chrF++ (Zh) |
| --- | --- | --- | --- | --- |
| Vanilla Baseline ($\alpha=0, \lambda=0$) | 0 | 19.61 | 14.05 | 22.72 |
| CLRR-only ($\alpha=0.1, \lambda=0$) | 0 | 18.47 | 13.24 | 21.90 |
| LSR-only ($\alpha=0, \lambda=0.1$) | 0 | 20.08 | 16.07 | 22.80 |
| Full CLRR + LSR ($\alpha=0.1, \lambda=0.1$) | 0 | 20.39 | 19.08 | 23.59 |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & $\Delta\theta$ & \textbf{BLEU} & \textbf{chrF++} & \textbf{chrF++ (Zh)} \\
\midrule
Vanilla Baseline ($\alpha=0, \lambda=0$) & 0 & 19.61 & 14.05 & 22.72 \\
CLRR-only ($\alpha=0.1, \lambda=0$) & 0 & 18.47 & 13.24 & 21.90 \\
LSR-only ($\alpha=0, \lambda=0.1$) & 0 & 20.08 & 16.07 & 22.80 \\
\textbf{Full CLRR + LSR ($\alpha=0.1, \lambda=0.1$)} & \textbf{0} & \textbf{20.39} & \textbf{19.08} & \textbf{23.59} \\
\bottomrule
\end{tabular}
```

</details>

## 3. Tập con hình thái Amis

Nguồn LaTeX: `tab:morphology_breakdown`.

| Morphological Subset | Sample Count | Vanilla Baseline | CLRR-Enc (Ours) | $\Delta (chrF++)$ |
| --- | --- | --- | --- | --- |
| Full Test Set | 575 | 14.05 | 19.08 | +5.03 |
| mi- (Actor Voice) | 195 | 14.91 | 15.38 | +0.47 |
| ma- (Patient/Stative Voice) | 246 | 13.29 | 22.10 | +8.81 |
| pa- (Causative Voice) | 127 | 15.17 | 15.31 | +0.14 |
| Root / Simple (Control group, no voice affixes) | 161 | 15.19 | 14.75 | $-0.44$ |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{lcccc}
\toprule
\textbf{Morphological Subset} & \textbf{Sample Count} & \textbf{Vanilla Baseline} & \textbf{CLRR-Enc (Ours)} & $\mathbf{\Delta\text{ (chrF++)}}$ \\
\midrule
Full Test Set & 575 & 14.05 & 19.08 & +5.03 \\
\midrule
\textit{mi-} (Actor Voice) & 195 & 14.91 & 15.38 & +0.47 \\
\textit{ma-} (Patient/Stative Voice) & 246 & 13.29 & \textbf{22.10} & \textbf{+8.81} \\
\textit{pa-} (Causative Voice) & 127 & 15.17 & 15.31 & +0.14 \\
\midrule
Root / Simple (Control group, no voice affixes) & 161 & 15.19 & 14.75 & $-0.44$ \\
\bottomrule
\end{tabular}
```

</details>

## 4. Cosine theo tầng mT5

Nguồn LaTeX: `tab:layer_cosine`.

| Layer | Vanilla Baseline | CLRR-Enc (Ours) | $\Delta (Cosine)$ |
| --- | --- | --- | --- |
| Layer 1 | 0.3319 | 0.3345 | +0.0026 |
| Layer 2 | 0.4768 | 0.4800 | +0.0032 |
| Layer 3 | 0.7131 | 0.6387 | -0.0744 |
| Layer 4 | 0.7929 | 0.6539 | -0.1390 |
| Layer 5 | 0.8602 | 0.7422 | -0.1180 |
| Layer 6 | 0.9092 | 0.8143 | -0.0949 |
| Layer 7 | 0.9271 | 0.8346 | -0.0925 |
| Layer 8 | 0.9519 | 0.8745 | -0.0774 |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{lccc}
\toprule
\textbf{Layer} & \textbf{Vanilla Baseline} & \textbf{CLRR-Enc (Ours)} & $\mathbf{\Delta\text{ (Cosine)}}$ \\
\midrule
Layer 1 & 0.3319 & 0.3345 & +0.0026 \\
Layer 2 & 0.4768 & 0.4800 & +0.0032 \\
Layer 3 & 0.7131 & 0.6387 & \textbf{-0.0744} \\
Layer 4 & 0.7929 & 0.6539 & \textbf{-0.1390} \\
Layer 5 & 0.8602 & 0.7422 & \textbf{-0.1180} \\
Layer 6 & 0.9092 & 0.8143 & \textbf{-0.0949} \\
Layer 7 & 0.9271 & 0.8346 & \textbf{-0.0925} \\
Layer 8 & 0.9519 & 0.8745 & \textbf{-0.0774} \\
\bottomrule
\end{tabular}
```

</details>

## 5. Ví dụ dịch định tính trong paper

Nguồn LaTeX: `tab:qualitative_examples`.

| Amis Source Input | Reference (Mandarin) | Vanilla Baseline | CLRR + LSR (Ours) |
| --- | --- | --- | --- |
| Minengneng koya wawa to cudad. | 那個小孩在看書。 | 那裡的小孩書。 | 那個小孩在看書。 |
| (mi- actor voice; look child that book) | (That child is reading a book.) | (Omitted verb aspect: child book) | (Correct agent-voice and progressive verb: is reading) |
| Mafuti' ko wawa i sasingalan. | 小孩在窗邊睡著了。 | 小孩子窗戶。 | 小孩在窗邊睡覺。 |
| (ma- patient/stative; sleep child at window) | (The child fell asleep by the window.) | (Omitted stative predicate: child window) | (Accurately captured stative predicate: sleeping by window) |
| Pananom ko ina to wawa. | 媽媽給小孩喝水。 | 媽媽水小孩。 | 媽媽讓小孩喝水。 |
| (pa- causative; water mother child) | (Mother gives child water to drink.) | (Collapsed noun sequence: mother water child) | (Accurate causative syntax: makes child drink) |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabularx}{\textwidth}{XXXX}
\toprule
\textbf{Amis Source Input} & \textbf{Reference (Mandarin)} & \textbf{Vanilla Baseline} & \textbf{CLRR + LSR (Ours)} \\
\midrule
\textit{\textbf{Mi}nengneng koya wawa to cudad.} & \begin{CJK*}{UTF8}{bsmi}那個小孩在看書。\end{CJK*} & \begin{CJK*}{UTF8}{bsmi}那裡的小孩書。\end{CJK*} & \begin{CJK*}{UTF8}{bsmi}那個小孩在看書。\end{CJK*} \\
(\textit{mi-} actor voice; look child that book) & (That child is reading a book.) & (Omitted verb aspect: \textit{child book}) & (Correct agent-voice and progressive verb: \textit{is reading}) \\
\midrule
\textit{\textbf{Ma}futi' ko wawa i sasingalan.} & \begin{CJK*}{UTF8}{bsmi}小孩在窗邊睡著了。\end{CJK*} & \begin{CJK*}{UTF8}{bsmi}小孩子窗戶。\end{CJK*} & \begin{CJK*}{UTF8}{bsmi}小孩在窗邊睡覺。\end{CJK*} \\
(\textit{ma-} patient/stative; sleep child at window) & (The child fell asleep by the window.) & (Omitted stative predicate: \textit{child window}) & (Accurately captured stative predicate: \textit{sleeping by window}) \\
\midrule
\textit{\textbf{Pa}nanom ko ina to wawa.} & \begin{CJK*}{UTF8}{bsmi}媽媽給小孩喝水。\end{CJK*} & \begin{CJK*}{UTF8}{bsmi}媽媽水小孩。\end{CJK*} & \begin{CJK*}{UTF8}{bsmi}媽媽讓小孩喝水。\end{CJK*} \\
(\textit{pa-} causative; water mother child) & (Mother gives child water to drink.) & (Collapsed noun sequence: \textit{mother water child}) & (Accurate causative syntax: \textit{makes child drink}) \\
\bottomrule
\end{tabularx}
```

</details>

## 6. Bootstrap và significance trong paper

Nguồn LaTeX: `tab:bootstrap_details`.

| Comparison Type | Baseline | Challenger | Metric | $\Delta$ | 95% CI | Raw $p$ | \textbf{Adj. $p_{holm}$} |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Primary Hypothesis | mBART Baseline | CLRR-Enc + LSR | chrF++ | +5.03 | [+3.82, +6.29] | 0.0001 | 0.0006$^{***}$ |
| Primary Hypothesis | mBART Baseline | CLRR-Enc + LSR | BLEU | +0.78 | [+0.08, +1.52] | 0.0954 | 0.2862 |
| Primary Hypothesis | mT5 Baseline | CLRR-Dec + LSR | BLEU | +2.04 | [+1.35, +2.74] | 0.0001 | 0.0006$^{***}$ |
| Primary Hypothesis | mT5 Baseline | CLRR-Dec + LSR | chrF++ | +1.37 | [+0.85, +1.91] | 0.0001 | 0.0006$^{***}$ |
| Component Synergy | CLRR-only | Full CLRR+LSR | chrF++ | +5.84 | [+4.61, +7.12] | 0.0001 | 0.0006$^{***}$ |
| Component Synergy | LSR-only | Full CLRR+LSR | chrF++ | +3.01 | [+2.11, +3.94] | 0.0001 | 0.0006$^{***}$ |
| Dual Stack Penalty | mT5 Enc + LSR | CLRR-Both + LSR | BLEU | -1.15 | [-1.68, -0.62] | 0.0001 | 0.0006$^{***}$ |
| Dual Stack Penalty | mT5 Enc + LSR | CLRR-Both + LSR | chrF++ | -0.47 | [-0.78, -0.16] | 0.0013 | 0.0052$^{**}$ |
| PEFT vs.\ CLRR | LoRA ($r=8$) | CLRR-Enc + LSR | BLEU | +17.06 | [+15.82, +18.31] | 0.0001 | 0.0006$^{***}$ |
| PEFT vs.\ CLRR | LoRA ($r=8$) | CLRR-Enc + LSR | chrF++ | +13.93 | [+12.65, +15.20] | 0.0001 | 0.0006$^{***}$ |
| PEFT vs.\ CLRR | Strong LoRA A | CLRR-Enc + LSR | BLEU | +9.97 | [+8.84, +11.12] | 0.0001 | 0.0006$^{***}$ |
| PEFT vs.\ CLRR | Strong LoRA A | CLRR-Enc + LSR | chrF++ | +9.28 | [+8.15, +10.44] | 0.0001 | 0.0006$^{***}$ |
| PEFT vs.\ CLRR | Strong LoRA B | CLRR-Enc + LSR | BLEU | +6.35 | [+5.21, +7.49] | 0.0001 | 0.0006$^{***}$ |
| PEFT vs.\ CLRR | Strong LoRA B | CLRR-Enc + LSR | chrF++ | +7.59 | [+6.48, +8.72] | 0.0001 | 0.0006$^{***}$ |
| PEFT vs.\ CLRR | BitFit | CLRR-Enc + LSR | BLEU | +20.04 | [+18.91, +21.18] | 0.0001 | 0.0006$^{***}$ |
| PEFT vs.\ CLRR | BitFit | CLRR-Enc + LSR | chrF++ | +16.24 | [+15.01, +17.47] | 0.0001 | 0.0006$^{***}$ |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{lllccccc}
\toprule
\textbf{Comparison Type} & \textbf{Baseline} & \textbf{Challenger} & \textbf{Metric} & $\Delta$ & \textbf{95\% CI} & \textbf{Raw $p$} & \textbf{Adj. $p_{\text{holm}}$} \\
\midrule
Primary Hypothesis & mBART Baseline & CLRR-Enc + LSR & chrF++ & \textbf{+5.03} & [+3.82, +6.29] & 0.0001 & \textbf{0.0006}$^{***}$ \\
Primary Hypothesis & mBART Baseline & CLRR-Enc + LSR & BLEU & +0.78 & [+0.08, +1.52] & 0.0954 & 0.2862 \\
Primary Hypothesis & mT5 Baseline & CLRR-Dec + LSR & BLEU & \textbf{+2.04} & [+1.35, +2.74] & 0.0001 & \textbf{0.0006}$^{***}$ \\
Primary Hypothesis & mT5 Baseline & CLRR-Dec + LSR & chrF++ & +1.37 & [+0.85, +1.91] & 0.0001 & \textbf{0.0006}$^{***}$ \\
\midrule
Component Synergy & CLRR-only & Full CLRR+LSR & chrF++ & \textbf{+5.84} & [+4.61, +7.12] & 0.0001 & \textbf{0.0006}$^{***}$ \\
Component Synergy & LSR-only & Full CLRR+LSR & chrF++ & \textbf{+3.01} & [+2.11, +3.94] & 0.0001 & \textbf{0.0006}$^{***}$ \\
\midrule
Dual Stack Penalty & mT5 Enc + LSR & CLRR-Both + LSR & BLEU & \textbf{-1.15} & [-1.68, -0.62] & 0.0001 & \textbf{0.0006}$^{***}$ \\
Dual Stack Penalty & mT5 Enc + LSR & CLRR-Both + LSR & chrF++ & \textbf{-0.47} & [-0.78, -0.16] & 0.0013 & \textbf{0.0052}$^{**}$ \\
\midrule
PEFT vs.\ CLRR & LoRA ($r=8$) & CLRR-Enc + LSR & BLEU & \textbf{+17.06} & [+15.82, +18.31] & 0.0001 & \textbf{0.0006}$^{***}$ \\
PEFT vs.\ CLRR & LoRA ($r=8$) & CLRR-Enc + LSR & chrF++ & \textbf{+13.93} & [+12.65, +15.20] & 0.0001 & \textbf{0.0006}$^{***}$ \\
PEFT vs.\ CLRR & Strong LoRA A & CLRR-Enc + LSR & BLEU & \textbf{+9.97} & [+8.84, +11.12] & 0.0001 & \textbf{0.0006}$^{***}$ \\
PEFT vs.\ CLRR & Strong LoRA A & CLRR-Enc + LSR & chrF++ & \textbf{+9.28} & [+8.15, +10.44] & 0.0001 & \textbf{0.0006}$^{***}$ \\
PEFT vs.\ CLRR & Strong LoRA B & CLRR-Enc + LSR & BLEU & \textbf{+6.35} & [+5.21, +7.49] & 0.0001 & \textbf{0.0006}$^{***}$ \\
PEFT vs.\ CLRR & Strong LoRA B & CLRR-Enc + LSR & chrF++ & \textbf{+7.59} & [+6.48, +8.72] & 0.0001 & \textbf{0.0006}$^{***}$ \\
PEFT vs.\ CLRR & BitFit & CLRR-Enc + LSR & BLEU & \textbf{+20.04} & [+18.91, +21.18] & 0.0001 & \textbf{0.0006}$^{***}$ \\
PEFT vs.\ CLRR & BitFit & CLRR-Enc + LSR & chrF++ & \textbf{+16.24} & [+15.01, +17.47] & 0.0001 & \textbf{0.0006}$^{***}$ \\
\bottomrule
\end{tabular}
```

</details>

## 7. Sensitivity sweep trong paper

Nguồn LaTeX: `tab:sensitivity_sweep`.

| Skip Distance ($d$) | Strength ($\alpha$) | $\lambda_{align}$ | Val BLEU | Val chrF++ |
| --- | --- | --- | --- | --- |
| $d = 1$ | 0.10 | 0.10 | 4.12 | 4.88 |
| $d = 2$ (Default) | 0.10 | 0.10 | 4.60 | 5.04 |
| $d = 3$ | 0.10 | 0.10 | 4.38 | 4.95 |
| $d = 4$ | 0.10 | 0.10 | 3.95 | 4.76 |
| $d = 2$ | 0.05 | 0.10 | 4.41 | 4.96 |
| $d = 2$ | 0.10 | 0.10 | 4.60 | 5.04 |
| $d = 2$ | 0.20 | 0.10 | 4.29 | 4.91 |
| $d = 2$ | 0.10 | 0.05 | 4.48 | 4.99 |
| $d = 2$ | 0.10 | 0.10 | 4.60 | 5.04 |
| $d = 2$ | 0.10 | 0.20 | 4.35 | 4.92 |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{ccccc}
\toprule
\textbf{Skip Distance ($d$)} & \textbf{Strength ($\alpha$)} & $\lambda_{\text{align}}$ & \textbf{Val BLEU} & \textbf{Val chrF++} \\
\midrule
$d = 1$ & 0.10 & 0.10 & 4.12 & 4.88 \\
$d = 2$ (\textbf{Default}) & 0.10 & 0.10 & \textbf{4.60} & \textbf{5.04} \\
$d = 3$ & 0.10 & 0.10 & 4.38 & 4.95 \\
$d = 4$ & 0.10 & 0.10 & 3.95 & 4.76 \\
\midrule
$d = 2$ & 0.05 & 0.10 & 4.41 & 4.96 \\
$d = 2$ & \textbf{0.10} & 0.10 & \textbf{4.60} & \textbf{5.04} \\
$d = 2$ & 0.20 & 0.10 & 4.29 & 4.91 \\
\midrule
$d = 2$ & 0.10 & 0.05 & 4.48 & 4.99 \\
$d = 2$ & 0.10 & \textbf{0.10} & \textbf{4.60} & \textbf{5.04} \\
$d = 2$ & 0.10 & 0.20 & 4.35 & 4.92 \\
\bottomrule
\end{tabular}
```

</details>

## 8. Thời gian và latency trong paper

Nguồn LaTeX: `tab:computational_benchmarks`.

| Model | Params | Train Time | GPU Lat. | CPU Lat. |
| --- | --- | --- | --- | --- |
|  |  | (20 ep) | (ms, $B=1$) | (ms, $B=1$) |
| mT5 Baseline | 300M | 12.4 min | 18.2 ms | 142 ms |
| mT5 + CLRR | 300M | 12.6 min | 18.3 ms | 144 ms |
| mBART + BitFit | 611M (336k) | 27.3 min | 34.6 ms | 312 ms |
| mBART + Narrow LoRA | 612M (1.18M) | 57.5 min | 34.9 ms | 315 ms |
| mBART + Strong LoRA A | 620M (8.65M) | 93.5 min | 35.8 ms | 324 ms |
| mBART + Strong LoRA B | 620M (265M) | 95.1 min | 35.9 ms | 325 ms |
| mBART Baseline | 611M | 31.8 min | 34.6 ms | 312 ms |
| mBART + Mid-Align | 611M | 34.1 min | 34.8 ms | 314 ms |
| mBART + CLRR (Ours) | 611M | 32.5 min | 34.7 ms | 313 ms |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{Train Time} & \textbf{GPU Lat.} & \textbf{CPU Lat.} \\
 & & (20 ep) & (ms, $B=1$) & (ms, $B=1$) \\
\midrule
mT5 Baseline & 300M & 12.4 min & 18.2 ms & 142 ms \\
mT5 + CLRR & 300M & 12.6 min & 18.3 ms & 144 ms \\
\midrule
mBART + BitFit & 611M (336k) & 27.3 min & 34.6 ms & 312 ms \\
mBART + Narrow LoRA & 612M (1.18M) & 57.5 min & 34.9 ms & 315 ms \\
mBART + Strong LoRA A & 620M (8.65M) & 93.5 min & 35.8 ms & 324 ms \\
mBART + Strong LoRA B & 620M (265M) & 95.1 min & 35.9 ms & 325 ms \\
mBART Baseline & 611M & 31.8 min & 34.6 ms & 312 ms \\
mBART + Mid-Align & 611M & 34.1 min & 34.8 ms & 314 ms \\
mBART + CLRR (Ours) & 611M & 32.5 min & 34.7 ms & 313 ms \\
\bottomrule
\end{tabular}
```

</details>

## 9. Stack ablation mT5

Nguồn LaTeX: `tab:stack_ablation`.

| Configuration | Rewire Stack | BLEU | chrF++ |
| --- | --- | --- | --- |
| Standard Baseline | None | 2.7887 | 3.8129 |
| CLRR-Enc | Encoder only | 4.4393 | 5.1761 |
| LSR alone | None | 4.5422 | 5.0118 |
| CLRR-Enc + LSR | Encoder only | 4.5961 | 5.0425 |
| CLRR-Dec + LSR | Decoder only | 4.8315 | 5.1873 |
| CLRR-Both + LSR | Both Enc & Dec | 3.4478 | 4.5752 |

<details>
<summary>Bảng LaTeX nguyên bản để đối chiếu</summary>

```latex
\begin{tabular}{llcc}
\toprule
\textbf{Configuration} & \textbf{Rewire Stack} & \textbf{BLEU} & \textbf{chrF++} \\
\midrule
Standard Baseline & None & 2.7887 & 3.8129 \\
CLRR-Enc & Encoder only & 4.4393 & 5.1761 \\
LSR alone & None & 4.5422 & 5.0118 \\
CLRR-Enc + LSR & Encoder only & 4.5961 & 5.0425 \\
\textbf{CLRR-Dec + LSR} & \textbf{Decoder only} & \textbf{4.8315} & \textbf{5.1873} \\
CLRR-Both + LSR & Both Enc \& Dec & 3.4478 & 4.5752 \\
\bottomrule
\end{tabular}
```

</details>

## 10. Những số paper công bố ngoài bảng

- Amis train/validation/test: **42180/5314/5286 token Amis**, **49812/6290/6244 chữ Hán** theo paper.
- No-stop-gradient mT5: paper ghi gradient tối đa **>1.2×10³ trong 3 epoch**, so với **18.4±4.2** khi có stop-gradient.
- Appendix lý thuyết ghi **233** đường đi cho ví dụ12layer/distance2.
- Paper kết luận **≤0.4% inference overhead** và **+2.2% training overhead**.
- mBART: paper nêu tăng **+5.03 chrF++**; mT5 decoder: **+2.04 BLEU**.

Đây là các số được bản thảo công bố. Đối chiếu artifact và kết luận sau kiểm tra tại [file sau, mục 7](EXPERIMENTS_UPDATED.md#7-doi-chieu-claim-goc-voi-bang-chung-hien-tai).
