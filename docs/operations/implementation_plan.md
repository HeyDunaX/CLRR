# Kế hoạch hiện hành sau revision paper — 03/10/2026

## Trạng thái

- **Yêu cầu mới nhất:** hoàn tất Full α0.20/λ0.10/seed42 rồi tạm ngưng; không chạy trial tiếp.
  Hướng dẫn phục hồi và chạy tiếp: [CONFIGURATION_SEARCH_HANDOFF.md](CONFIGURATION_SEARCH_HANDOFF.md).
  File bàn giao là nơi tra điểm dừng mới nhất, thay các dòng trạng thái khởi chạy cũ bên dưới.

- Influence C đã hoàn tất; vòng D đã xong2trial Full α0.05 và α0.20, cùng λ0.10/seed42.
  Validation chrF++14.4092/14.5738, chưa vượt Full gốc14.7317. HF backups verified,
  metrics/predictions lưu local; suite tạm ngưng theo tác giả trước trial3.
- Đã hoàn tất 12run matched mBART Amis (Baseline, CLRR-only, LSR-only, Full ×seeds42/43/44), 35probe và kiểm định Holm24.
- 10weights archive còn đầy đủ đã được backup HF; LSR/Full seed42 mất weights trước backup, predictions/metrics còn đủ.
- Turkish và Colab đã dừng; chưa có trained-test comparison Turkish.
- [Paper hiện hành](clrr_main.tex) đã viết lại theo [kết quả xác minh](EXPERIMENTS_UPDATED.md), với [metrics thống nhất](METRICS.md).

## Tiếp theo

Dùng [RESEARCH_INSIGHTS.md](RESEARCH_INSIGHTS.md) mới cùng sổ experiments hiện hành.
Insight cũ giữ tại [archive](archive/experiment_reports_20261003/RESEARCH_INSIGHTS.md).
Bước gần nhất đã chuẩn bị là character-only chrF và influence audit trên 12 outputs C:
đã kiểm tra hashes, đủ575rows/file, source order và raw targets. Xem mục7 của insight
mới để biết scope, đầu ra và điều kiện chuyển bước; trạng thái thực thi mới xem mục9.

Dùng [CLRR_GO_NOGO_PLAN.md](CLRR_GO_NOGO_PLAN.md): provenance → character/influence và validation audit → tối ưu chọn trên validation → matched confirmation → quyết định Go/No-Go. Chưa thực hiện tối ưu hoặc chạy lại training trong đợt sửa paper này.

Kế hoạch tiếp theo đã cập nhật: chẩn đoán không train → screening Full/LSR cùng5trials
→ validation seeds43/44 → CLRR-only/no-detach controls → quyết định Amis. Tối đa20
lượt train mới nếu reuse hợp lệ; dừng sớm theo gate. Nếu không qua và có nguyên nhân
được hỗ trợ, tối đa2 nâng cấp với budget10lượt có điều kiện. Turkish/NLLB pilot4,
mở đủ12lượt chỉ khi có tín hiệu phù hợp. Đây là kế hoạch, chưa khởi chạy.

Full hơn LSR-only trung bình0.2344BLEU, 3/3seed dương nhưng nhỏ; Full hơn baseline0.3800BLEU, 2/3seed dương. Chưa có per-seed comparison đạt Holm24. Chưa đủ bằng chứng claim robust independent CLRR improvement; phải chẩn đoán trước khi tăng trọng số hoặc mở rộng language.

## Revision paper và sửa source đã hoàn tất

- Paper hiện hành có10bảng sinh từ frozen CSV, với source SHA-256 trong `paper_tables/provenance.json`; PDF được compile và kiểm tra bố cục.
- Theo yêu cầu tác giả, đã khôi phục format paper gốc: tiêu đề, preamble ACL review, tên các mục chính và cách trình bày bảng. Nội dung đã xác minh và toàn bộ số liệu giữ nguyên; PDF sau khôi phục có12trang gồm references/appendix. Receipt: `../outputs_rebuttal/format_restoration_20261003/receipt.json`.
- Sửa ignored prediction IDs theo pad_token_id, kiểm tra saved targets khi chấm, precision defaultFP32 trên các trainer, NLLB tokenizer/config/beam policy, và checkpoint key prefixes của comparator alignment. Code cũ và metrics artifact không được tái diễn giải thành kết quả của code mới.
- Kiểm tra trong conda clrr:22unit tests, tiny mBART/NLLB train/eval/reload và correctness trên4architectures trong lượt revision trước; các kiểm tra sau thu gọn được ghi trong scope_revision_20261003/verification.json. Không bảo đảm mọi lỗi có thể có đã được tìm thấy; còn phải đo quality theo các gate.
- Cleanup receipt: `../outputs_rebuttal/workspace_cleanup_20261003/cleanup_manifest.json`; tổng11.08GB đã dọn trong lượt revision, sau xác minh remote backup của các weights còn lại.
- Build bảng: `conda run -n clrr python scripts/revalidation/build_paper_tables.py`; nguồn master ở `docs/` có acl.sty, acl_natbib.bst và clrr_references.bib. Bản PDF hiện hành đã được compile thực tế bằng pdfLaTeX + BibTeX, 12trang/10bảng, không missing input hoặc unresolved citation. Receipt: `../outputs_rebuttal/pdftex_compile_20261003/verification.json`.
- File gốc `docs/clrr_main.tex` đã chứa trực tiếp cả 10 bảng, không còn `\input{paper_tables/...}`. Lệnh build bảng tự cập nhật các vùng bảng trong file gốc từ frozen CSV; không sửa số thủ công trong những vùng này.
- Upload với pdfLaTeX: chạy `conda run -n clrr python scripts/revalidation/package_paper_pdftex.py` để tạo [clrr_pdflatex_upload.zip](clrr_pdflatex_upload.zip), kèm bibliography hiện hành và style ACL gốc. Nếu dùng project Overleaf cũ có main `latex/clrr_main.tex`, thay nội dung đúng file đó bằng `docs/clrr_main.tex` mới, chọn compiler pdfLaTeX và Recompile from scratch. Nếu tạo project từ ZIP, chọn main `clrr_main.tex`; dùng bibliography đi kèm để tránh thiếu citation key.

## Lịch sử lưu trữ

[Kế hoạch/runbook cũ](archive/implementation_plan_original_20261003.md) và [experiments gốc](archive/EXPERIMENTS_ORIGINAL_20261003.md) chỉ lưu lịch sử; các claim và lệnh cũ không phải chỉ thị chạy.
