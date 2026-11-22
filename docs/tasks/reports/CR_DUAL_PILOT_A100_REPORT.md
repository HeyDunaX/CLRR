# Báo cáo Pilot A100 `CR-Dual` (4 runs + Step-0 Diagnostic, Validation-Only — `NO_GO`)

- **Ngày hoàn tất:** 22/11/2026
- **Commit mã nguồn:** `5fc80a7dc9394435f16fe7feb09fe20f56394fca` | **Bundle manifest SHA256:** `1852c18cb3fbff55f834b00766b1dc899fd065503fb189eb38f295320a8a0960`
- **Protocol khóa:** [amis_a100_bleu.json](../../../configs/protocols/amis_a100_bleu.json) (`mBART-50` rev `e30b6cb8…`, A100 40GB BF16, `B32×GA4`, LR `5e-5`, `ES10`, `eval_bleu` beam 1 FP32 `zh`, **không mở test**)
- **Thiết kế khóa trước:** [ROUND3_METHOD_DESIGN_AUDIT.md](../../research/ROUND3_METHOD_DESIGN_AUDIT.md)

---

## 1. Kết quả đối chiếu (`[ĐO ĐƯỢC]`, Amis Validation 576 câu, Seed 42)

| Run | Method | Nhóm | Val BLEU | Val chrF0 | Val chrF++ | $\Delta$ vs `BASELINE` | Archive SHA256 (prefix) |
|---|---|---|---:|---:|---:|---:|---|
| `BASELINE_S42` | `baseline` | REUSE | `17.5264` | `16.7632` | `14.5382` | `0.0000` | `985f9fc4` |
| `W_B125_S42` | `rdrop-consistency-v1` | REUSE | `17.7069` | `17.0817` | `14.8852` | `+0.1805` | `4df107c0` |
| `GOC_SYM_B250_S42` | `symmetric-stopgrad-consistency-v1` | REUSE | **`18.5143`** | **`17.5405`** | **`15.3266`** | **`+0.9879`** | `47f6f925` |
| `BIT_CE_S42` | `bidirectional-ce-v1` | NEW | `16.4574` | `15.8076` | `13.6946` | **`-1.0690`** | `55c51fddd989` |
| `CR_DUAL_S42` | `cross-ratio-dual-v1` | NEW | `16.1977` | `15.8281` | `13.7608` | **`-1.3287`** | `6045bdddb08d` |
| `CR_SHUFFLED_S42` | `cross-ratio-shuffled-v1` | NEW | `16.9163` | `16.4246` | `14.2490` | **`-0.6101`** | `6609bb24a96c` |
| `UNI_CONTRAST_S42` | `unidirectional-contrast-v1` | NEW | `16.7299` | `16.1342` | `13.9757` | **`-0.7965`** | `1334bdb1552d` |

---

## 2. Chẩn đoán Bước 0 trên `BIT_CE_S42` (0 optimizer steps, 112 cặp validation)

- **Bất đối xứng chéo trên `mBART-50` thật (`[ĐO ĐƯỢC]`):** `mean_m_fwd = 2.4677` vs `mean_m_rev = 3.4471` ($\Delta = -0.9794$, median `-0.9488`, `mean_abs_violation = 1.0388`, **`92.0%` cặp có $M^{\to} < M^{\leftarrow}$**). Bất đối xứng chiều thuận–nghịch tồn tại thật trên `mBART-50`.
- **Kiểm tra Paired-RNG Shared Dropout (8 lượt lặp):** phương sai giảm từ `0.0581 → 0.0481` (**chỉ giảm `17.1%`**, không đạt ngưỡng `70%` vì encoder dropout vẫn chi phối phương sai qua từng lượt).
- **Hiệu chuẩn gradient:** `grad_norm_ce = 15.075`, `grad_norm_cr = 20.704`, `rho_grad = 1.373`, `cos_cr_ce = +0.2185`. Với `cr_lambda = 0.25`, tỷ lệ hiệu dụng là `0.3433` (vượt khung `[0.02, 0.30]`; mức đề xuất cho `0.10` là `lambda = 0.0728`).

---

## 3. Phán quyết khóa trước: **`NO_GO`** (Đóng nhánh `CR-Dual` / `BiT` / `UNI_CONTRAST` trên Amis `mBART-50`)

1. **Tất cả 5 cổng khóa trước đều `FAIL`:** `CR_DUAL - BIT_CE = -0.2597`, `CR_DUAL - CR_SHUFFLED = -0.7185`, `CR_DUAL - UNI_CONTRAST = -0.5322`, `CR_DUAL - BASELINE = -1.3287`, `CR_DUAL - SYM_SG = -2.3166`.
2. **Nguyên nhân cấu trúc không thể vá bằng sweep $\lambda$:**
   - Ngay chính nền `BIT_CE_S42` ($\lambda = 0$) đã tụt **`-1.0690` BLEU** dưới `BASELINE_S42` do decoder `mBART-50` bị chia tải sinh ngôn ngữ Amis chưa từng xuất hiện trong pretraining (decoder capacity cannibalization).
   - Falsifier `CR_SHUFFLED_S42` (`16.9163`) thắng `CR_DUAL_S42` (`16.1977`) tới **`+0.7185` BLEU**, trực tiếp bác bỏ cơ chế ghép cặp tương thích Bayes Arnold-Press từng cặp khi chiều nghịch nhiễu.
   - Đẩy negative sequence-level một chiều (`UNI_CONTRAST_S42 = 16.7299`) cũng tụt **`-0.7965` BLEU** dưới `BASELINE_S42`.
3. **Quyết định:** Không sweep $\lambda$ / $\alpha_{\text{rev}}$, không mở thêm seed, không mở test. Lưu toàn bộ 4 runs vào sổ gốc để tránh lặp lại.
