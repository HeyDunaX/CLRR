# Kế hoạch hiện hành sau revision paper — 03/10/2026

## Trạng thái

- Đã hoàn tất 12run matched mBART Amis (Baseline, CLRR-only, LSR-only, Full ×seeds42/43/44), 35probe và kiểm định Holm24.
- 10weights archive còn đầy đủ đã được backup HF; LSR/Full seed42 mất weights trước backup, predictions/metrics còn đủ.
- Turkish và Colab đã dừng; chưa có trained-test comparison Turkish.
- [Paper hiện hành](clrr_main.tex) đã viết lại theo [kết quả xác minh](EXPERIMENTS_UPDATED.md), với [metrics thống nhất](METRICS.md).

## Tiếp theo

Dùng [CLRR_GO_NOGO_PLAN.md](CLRR_GO_NOGO_PLAN.md): provenance → character/influence và validation audit → tối ưu chọn trên validation → matched confirmation → quyết định Go/No-Go. Chưa thực hiện tối ưu hoặc chạy lại training trong đợt sửa paper này.

Full hơn LSR-only trung bình0.2344BLEU, 3/3seed dương nhưng nhỏ; Full hơn baseline0.3800BLEU, 2/3seed dương. Chưa có per-seed comparison đạt Holm24. Chưa đủ bằng chứng claim robust independent CLRR improvement; phải chẩn đoán trước khi tăng trọng số hoặc mở rộng language.

## Revision paper và sửa source đã hoàn tất

- Paper hiện hành có11bảng sinh từ frozen CSV, với source SHA-256 trong `paper_tables/provenance.json`; PDF được compile và kiểm tra bố cục.
- Sửa ignored prediction IDs theo pad_token_id, kiểm tra saved targets khi chấm, precision defaultFP32 trên các trainer, NLLB tokenizer/config/beam policy, và checkpoint key prefixes ở hai comparator. Code cũ và metrics artifact không được tái diễn giải thành kết quả của code mới.
- Kiểm tra trong conda clrr:22unit tests, tiny mBART/NLLB train/eval/reload, correctness trên4architectures và parse76Pythonfiles. Không bảo đảm mọi lỗi có thể có đã được tìm thấy; còn phải đo quality theo các gate.
- Cleanup receipt: `../outputs_rebuttal/workspace_cleanup_20261003/cleanup_manifest.json`; tổng11.08GB đã dọn trong lượt revision, sau xác minh remote backup của các weights còn lại.
- Build: `conda run -n clrr python scripts/revalidation/build_paper_tables.py`, sau đó chạy LaTeX từ `docs/` (có acl.sty, acl_natbib.bst và clrr_references.bib). Bản local đã compile bằng portable Tectonic trong scratch.

## Lịch sử lưu trữ

[Kế hoạch/runbook cũ](archive/implementation_plan_original_20261003.md) và [experiments gốc](archive/EXPERIMENTS_ORIGINAL_20261003.md) chỉ lưu lịch sử; các claim và lệnh cũ không phải chỉ thị chạy.
