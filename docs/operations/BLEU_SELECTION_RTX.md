# RTX — kiểm chứng chọn checkpoint theo validation BLEU

## Trạng thái 30/10/2026

Người dùng duyệt chạy lại đúng4 runs. CPU scientific/selection gates21/21 PASS, cả4 tiny-model backward/save-load PASS. Bundle đã khóa; queue GPU gate→training được gửi bằng launcher chung. Xem receipt launch_rtx.json và rtx_status.json để biết tình trạng thực tế, không suy từ tài liệu này rằng training đã bắt đầu.

Config: `configs/experiments/amis_alignment_residual_bleu_selection_rtx.json`.
Workspace PC/RTX: `outputs/experiments/amis_alignment_residual_bleu_selection_rtx/`.
RTX project: `/workspace/ura/thanhdat/projects/CLRR/`; conda/cache nằm trong project.

| Thứ tự | Run | Seed | Mục đích |
|---|---|---:|---|
|1|C0_BLEU_S42|42|Khôi phục best validation BLEU của C0|
|2|LEARNED_DONOR_BLEU_S42|42|Control learned donor cùng selector|
|3|M0_BLEU_S42|42|Control RNG/dummy pass cùng selector|
|4|DONOR_BLEU_S44|44|Control donor-only cùng selector|

## Khóa thiết kế và công bằng

Giữ model/pretrained revision, method formula/module seed, train/data seed, dữ liệu, LR5e-5, optimizer/scheduler/warmup, RTX FP32/SDPA, B4×GA32, max length256 như đúng run nguồn. Train đủ20 epochs để chọn lại trên toàn trajectory; cả18 run cũ đủ20. Early stopping được tắt rõ ràng trong recipe này, patience10 giữ như metadata lịch sử nhưng không có callback; không tuyên bố đang dùng ES10 khi thực tế tắt. Đây là selector sensitivity, không tối ưu phương pháp hoặc sửa baseline. Lock cũ `amis_rtx_baseline_sdpa.json` không thay; lock mới `amis_rtx_bleu_selection.json` mô tả ngoại lệ selector/fixed20.

Best chính = original-reference validation BLEU zh, beam1 FP32; ties earliest. `best.zip` chứa weights best BLEU; `best_chrf.zip` chứa best validation chrF++ cùng training run, receipt `secondary_selection.json` chỉ rõ step/epoch/score. Cả hai chọn bằng validation, không chọn theo test. Các bản14/18 trùng selector tiếp tục reuse; không inference lại chỉ để tạo số giống hệt. Test đã có development exposure; giữ nhãn exploratory.

## Giới hạn dung lượng

Không giữ toàn bộ20 epochs. Trainer save_total_limit=2; thêm1 archive model-only chrF++ xoay vòng và file .pending khi đang ghi thay thế. Optimizer/RNG chỉ trong rolling trainer checkpoints, không nằm trong hai archive cuối. Sau mỗi run upload hai best và metadata vào public HF, verify SHA/size đầy đủ trước khi xóa weights riêng của run; nếu upload fail thì giữ weights và dừng queue. Không tải checkpoint về Windows. Không xóa cache/pretrained hoặc process của người khác.

Public prefix mỗi run:
`FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/<method>/seed-N/rtx6000-fp32-sdpa-b4-ga32-epochs20-select-bleu-rank64/`.

`best.zip` và `best_chrf.zip` nằm chung prefix mới; không overwrite các folder cũ.

## Lệnh tái lập / theo dõi

```powershell
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_alignment_residual_bleu_selection_rtx.json --account 1
```

Queue tách khỏi SSH/local, GPU gates của cả4 phải PASS trước training dài. Nếu thiếu VRAM thì tự chờ, không giảm batch/đổi precision/kill người khác. Gates kiểm tra BF/FP flags, accumulation32/checkpointing, optimizer/RNG resume, save/reload logits và secondary archive restore. Queue xong tự thoát, không tắt RTX host.

Sau4 runs: xác minh receipts, đóng băng selection BLEU, test4 checkpoint mới và ghép14 reused, recompute đủ contrasts/seeds/CI/Holm. Không có auto test hoặc tìm candidate mới trong queue hiện tại. Không sửa bảng kết quả chrF++ cũ; lưu bảng BLEU-selected riêng.


## Test chọn BLEU hoàn tất — 30/10/2026

4/4 checkpoint mới TEST COMPLETE_VERIFIED; ghép14 hệ thống có checkpoint không đổi. 575 câu test, 10000 paired bootstrap RNG42, Holm45. PC đã replay54 corpus metrics và Holm45 PASS. Kết quả riêng: `outputs/experiments/amis_alignment_residual_bleu_selection_rtx_test/statistics/SEED_SUMMARY.md` và `PAIRED_BOOTSTRAP.md`; không ghi đè số selector chrF++ cũ.

C0 BLEU mean19.6748 vs baseline19.7306 (delta−0.0558); chrF++18.6874 vs15.5468 (delta+3.1406), chrF0 delta+0.1776. C0−baseline BLEU seed42 +0.9326 (Holm1.0),43 +0.6175 (Holm1.0),44 −1.7175 (Holm0.026997). Không có seed nào C0 thắng baseline có ý nghĩa sau Holm; seed44 thua baseline có ý nghĩa BLEU. Exact seed p BLEU1.0, chrF++0.5. chrF++ nhạy sparse word-bigram: test chỉ3 reference word-bigrams. Đây là exploratory selector sensitivity, chưa có bằng chứng C0 vượt baseline ổn định; đổi selector chưa giải quyết bất ổn giữa seeds.

Predictions/metrics public trong prefix checkpoint; thống kê public `analysis/amis_mandarin/mbart-large-50/alignment-residual-bleu-selection/test`, verified revision `f80e6d7cddf7b17a39c6c266227af0c47badfd82`. Chỉ tải JSON/CSV/Markdown về PC.
