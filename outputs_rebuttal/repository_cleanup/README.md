# Kết quả gom repo và workflow

Ngày: 10/10/2026. Không train model, không mở GPU, không tính lại số trong paper.

## Đã dọn

- 2,141 bản output model: chỉ xóa khi trùng bytes/hash với nguồn public đã pin trong `model_output_index.json`.
- 22 scripts launcher/monitor/migration đã hoàn thành; 37 helper điều khiển sinh tự động; 106 file scratch dùng một lần hoặc đã chuyển code khoa học.
- Cache công cụ LaTeX, SDK migration và tiny model tạm.
- Giải phóng 1.08 GiB, chưa cộng các bytecode cache.

## Đã giữ

- 65 representation `.npz` (1813.1 MiB) và các output chẩn đoán khoa học.
- 705 file model chưa xác minh backup public; không tự xóa.
- 12 ZIP nguồn cũ cùng một manifest, trong `.cache/frozen_sources/`; mappings/hash ở `frozen_source_index.json`. Nguồn đầy đủ chưa được xác minh có trên HF nên cần giữ để replay.
- Code phân tích khoa học được đưa từ scratch vào scripts: `source_relocations.json`. Công thức phân tích giữ nguyên, chỉ sửa workspace root/đường dẫn.
- Helper và credentials đăng nhập cũ; scratch còn một shim nhỏ tới transport chung.
- Các worker khoa học lịch sử chưa có thay thế chung vẫn giữ để tái lập bằng bundle đã pin.

## Xác minh

- 14 đối sánh: loss, logits, gradient và state keys khớp code trước khi gom.
- 18 recipe: training args khớp; candidate giữ workers=0.
- 21 gate CPU: backward, checkpointing, accumulation/save/load theo từng gate.
- 8 kiểm tra CLI: args, chống ghi đè, hash nguồn/config, đích upload và fetch.
- 8 kiểm tra backup/thống kê lịch sử. Test launcher cũ được sửa để khai báo rõ partition 2 GPU và kiểm tra thêm partition 3 GPU; không đổi việc phân công hay số thực nghiệm.
- Syntax và imports tới scripts đã bỏ: PASS.

Receipts này là kiểm tra việc gom code. Mỗi run mới vẫn phải build lại bundle,
chạy CPU gate và GPU gate trên chính bundle đó trước khi train dài.

## Lệnh chạy mới

Xem [EXPERIMENT_WORKFLOW.md](../../docs/operations/EXPERIMENT_WORKFLOW.md).
Dùng `scripts/experiment.py create` với args, rồi `build`, `cpu`, `launch` hoặc
`run`. Model mới upload và kiểm tra checksum trước khi xóa weights local.

`deleted_files.json` lưu danh sách file đã xóa với checksum và lý do.
`model_output_index.json` lưu đường dẫn HF/revision để lấy lại các artifact nhỏ.
`retained_cache_receipt.json` giữ receipt từ cache cũ nếu có; không gán lại seed
hay cấu hình model từ tên cache.
