# Chẩn đoán candidates seed42 — validation, không train

## Trạng thái

Được người dùng cho chạy sau pilot U1/U2. Bootstrap và GPU diagnostics
đã hoàn tất, 6/6 checkpoint replay PASS; Colab 1/2 đã tắt. Không sử dụng test để chọn phương pháp.
Không thay weights, không chạy optimizer, không thay số thí nghiệm cũ.

## Phạm vi và protocol

- Colab 1: Full cũ, Full-precache U1F, CLRR-precache U1C.
- Colab 2: Full-centered U2F, Centered-LSR U2L, baseline seed42.
- Revision HF khóa trong `outputs_rebuttal/method_candidate_diagnostics/PLAN.json`;
  xác minh SHA256 archive trước strict loading. Weights chỉ tải trên Colab.
- Validation Amis→Mandarin 576 câu, raw CSV references, source `tl_XX`, target `zh_CN`.
- FP32, dropout off, TF32 off; lengths256, beam1, generation batch8.
- Representation/rank: 128 câu cố định; gradient: 4 batch ×32 câu cố định,
  cùng indices cho mọi checkpoint. Centering dùng microbatch32 đúng công thức.
- Tắt rewiring theo layer 1-based: early3–5, middle6–9, late10–12.
  Đây là intervention inference trên frozen weights, không phải ablation train.
- Gradient CE và weighted LSR λ=.10 đo riêng theo layer; target detached.
  Target routed/raw chỉ là counterfactual cùng weights, không chứng minh raw tốt hơn.

## Bootstrap đã có

10.000 paired sentence draws; p từ native SacreBLEU2.6.0; CI percentile95%
của corpus delta; Holm4 cho 2 candidates ×2 metrics.

| Candidate − Full cũ | Metric | Delta | CI95% | p Holm4 |
|---|---|---:|---|---:|
| U1F | BLEU | −0.0261 | [−1.0097, +0.9928] | 0.9224 |
| U1F | chrF++ | +0.0739 | [−0.5616, +0.7437] | 0.9224 |
| U2F | BLEU | −0.3330 | [−1.4047, +0.7631] | 0.7743 |
| U2F | chrF++ | +0.0300 | [−0.7025, +0.7926] | 0.9224 |

Chưa chứng minh superiority hoặc inferiority. Đây là phân tích hậu nghiệm trên
validation từng được dùng chọn checkpoint, không phải kiểm định xác nhận độc lập
và không bao gồm training randomness. Không suy ra equivalence từ p không đạt.

## Mã thực thi

Trong conda `clrr`:

```powershell
python scripts/revalidation/diagnose_method_candidates.py --stage bootstrap
python scripts/revalidation/control_candidate_diagnostics.py --account 1 --stage monitor
python scripts/revalidation/control_candidate_diagnostics.py --account 2 --stage monitor
```

Không relaunch khi worker còn sống. Các receipts `*/COMPLETE.json` và predictions
nhỏ được collect về `outputs_rebuttal/method_candidate_diagnostics/`. Chỉ tắt
session sau khi collect và xác minh đủ kết quả của account đó.

## Cách ra quyết định

1. Kiểm tra generation-on replay khớp predictions/metrics đã lưu, strict loading,
   worker hash, finite gradients, batch identity và zero optimizer steps.
2. Báo cáo residual norm, rank và gradient conflict cùng số liệu batch/layer;
   không chọn phương pháp chỉ vì cosine đẹp hoặc norm LSR nhỏ.
3. Chỉ đề xuất đổi layer, target anchor hoặc lịch λ nếu có bằng chứng nhất quán.
   Không tự mở grid hoặc training mới sau chẩn đoán.
4. Nếu không có bằng chứng giải thích đủ rõ, ghi nhận U1/U2 chưa thành công và
   xây candidate card khác cùng rà soát novelty trước khi xin chạy pilot mới.

## Kết quả và quyết định sau đo

Chi tiết: [REPORT.md](../../outputs_rebuttal/method_candidate_diagnostics/REPORT.md).

- Full cũ/U1F: source effective covariance rank sau encoder final LayerNorm
  74.089/74.005, centered matched cosine0.2907/0.2910: thay donor chưa tạo khác biệt
  cơ chế rõ trên 128 câu này.
- U2F: centered matched cosine0.3554; CE–weighted-LSR cosine trung bình0.07827
  so với Full cũ0.01734, nhưng không cải thiện translation rõ. Alignment và
  gradient cùng hướng hơn không đủ bảo đảm chất lượng dịch.
- Không có bằng chứng collapse đặc thù CLRR: baseline final rank73.097;
  U2F67.499, U2L62.238 (rank limit127). Rank giảm ở block11/12 xảy ra cả baseline
  và hồi phục sau encoder final LayerNorm; không trộn hai vị trí đo.
- Full cũ: tắt nhóm3–5/6–9/10–12 giảm BLEU0.6159/0.6904/0.7244. Nhóm10–12
  giảm chrF++0.5001, p Holm18=.0378. Đây là dependence tại inference của seed42,
  không chứng minh phải giữ mọi layer khi huấn luyện lại.
- U2F tắt6–9: BLEU+0.6065 nhưng CI[−0.2124,+1.4449], p Holm18=.5599;
  chrF++−0.0240. Chưa đủ cơ sở ưu tiên topology này.
- Raw anchor giảm CE–LSR cosine trên Full cũ/U1F/U2F; không có bằng chứng
  directional-gradient ủng hộ thay raw anchor trong phép đo này.
- Không mở sweep lambda, không tự bỏ layer và không train rộng U1/U2. Các phép đo
  chưa tìm được một lỗi cơ chế đủ rõ để giải thích thất bại đa seed của Full cũ.
  Bước tiếp theo là xem lại hypothesis/objective và novelty trước candidate khác;
  nếu muốn xác nhận U1 qua seed, cần protocol matched và phê duyệt training riêng.

Lỗi worker tạo thư mục output đã sửa trước khi có completion receipt. Một số raw
decode có khoảng trắng đầu câu; sau `strip()` như worker gốc predictions khớp
100%, metrics khớp1e−9. Không sửa scores cũ, không có optimizer step.

GPU shutdown được kiểm tra: Colab1 balance45.46CU, Colab2 balance144.32CU;
cả hai active assignments0, usage rate0.00/hr. Không tải model weights về PC.

Người dùng đã xác nhận quyền công khai diagnostics sau khi bộ duyệt yêu cầu.
Public backup hoàn tất và checksum35files PASS tại
[HF diagnostics](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/analysis/amis_mandarin/mbart-large-50/method-candidate-diagnostics/seed-42),
revision `ba3f062c20be45ca9ac5aabbe7da1c43e6236c6d`.
Receipt local: `outputs_rebuttal/method_candidate_diagnostics/PUBLIC_BACKUP.json`.
Checkpoint public cũ và candidates giữ nguyên, không có weights bị mất.
