# Bối cảnh thí nghiệm bổ sung

Cập nhật: 27/09/2026. File này ghi trạng thái công việc hiện tại để tiếp tục sau khi đổi phiên làm việc. Bối cảnh chung và sáu kết quả ban đầu nằm trong [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) và [README](../README.md).

## Mục tiêu đã chốt

- Giữ nguyên bảng **sáu kết quả chính** đã chạy. Đây là bảng lịch sử, không dùng chênh lệch giữa điểm cũ và điểm tính lại để chặn thí nghiệm bổ sung.
- Phân tích năm checkpoint cũ: mT5 Baseline, CLRR-Enc, JEPA+CLRR-Enc; mBART Baseline, JEPA+CLRR-Enc. Sinh prediction trên 575 câu test và đo cosine theo tầng encoder của hai model mT5 Baseline/CLRR-Enc.
- Huấn luyện thêm bốn run: `byt5-small-ami-cmn-baseline`, `byt5-small-ami-cmn-jepa-clrr-enc`, `mt5-small-ami-cmn-jepa-clrr-dec`, `mt5-small-ami-cmn-jepa-clrr-both`.
- Chạy tối đa **20 epoch**, seed **42**, early stopping patience **4**; chọn best checkpoint theo validation chrF++, rồi mới đánh giá test. Cấu hình mới lấy từ script đã chạy và được đối chiếu với training arguments trong ZIP cũ ở bước `preflight`.

## Một giao thức chấm điểm cho phần bổ sung

Với **cả model cũ và mới**, lấy chuỗi prediction đã giải mã, ghép đúng thứ tự với cột `target` gốc của 575 dòng trong `data/processed/test.csv`, bỏ khoảng trắng đầu/cuối mỗi chuỗi, rồi tính bằng SacreBLEU `BLEU(tokenize="zh")` và `CHRF(word_order=2)` (chrF++). Hàm chung là `generation_metrics` trong [metrics.py](../src/amis_rewire/metrics.py). Điểm tính lại của model cũ ghi vào `analysis/old_model_scores.csv`; bảng so sánh chung ghi vào `analysis/all_scores.csv`. Bảng sáu điểm lịch sử vẫn giữ riêng.

Quyết định này giải quyết lỗi trước đây: prediction tính lại của mT5 Baseline cho điểm hơi khác `metrics/mt5-small-ami-cmn-baseline_metrics.json`. Không cần tái tạo đúng từng chữ số của file điểm cũ; cần dùng **cùng prediction/reference và cùng công thức** cho các so sánh mới. Khi viết bài, không trộn hai bảng như thể chúng có cùng cách tính.

## Trạng thái đã xác nhận

| Bước | Trạng thái |
| --- | --- |
| Colab cell 5: `preflight` và `smoke` | Người dùng đã chạy thành công: năm ZIP được kiểm tra, sinh thử hai câu thành công. |
| Colab cell 6: `analyze` | Các lần chạy trước lỗi do mã phân tích và cổng kiểm tra điểm cũ. Mã mới đã bỏ cổng đó; **chưa có xác nhận cell 6 chạy xong với phiên bản mới**. |
| Colab cell 7–8: bốn run mới | Chưa có xác nhận đã chạy xong hoặc đã tải đủ artifact lên Hugging Face. |
| Colab cell 9: báo cáo | Chưa có xác nhận hoàn tất. |

Trên máy cục bộ, các bài kiểm tra `test_followup.py` đã qua (12 test) và smoke training với model nhỏ đã tạo được checkpoint, best ZIP, metrics và predictions. Chúng không thay thế việc kiểm tra end-to-end với checkpoint riêng tư trên Colab.

## Điểm vào và nơi lưu

- [Notebook Colab](../notebooks/colab_followup_run.ipynb) — [mở trực tiếp trên Colab](https://colab.research.google.com/github/HeyDunaX/CLRR/blob/main/notebooks/colab_followup_run.ipynb). Notebook thiết lập môi trường và gọi mã Python; `HF_TOKEN` được cấp qua Colab Secret.
- [followup_analysis.py](../scripts/followup_analysis.py) chạy `preflight`, `smoke`, `analyze`, `report`. [run_followup_models.py](../scripts/run_followup_models.py) chạy ByT5 và ablation mT5 bằng pipeline huấn luyện hiện có.
- Repo backup riêng tư: `FiveC/amis-rewire-checkpoints`. ZIP cũ nằm ở `checkpoints/<run-name>/<run-name>-best.zip`; điểm lịch sử nằm ở `metrics/`. Run mới dùng `checkpoints/<run-name>/` cho checkpoint ZIP, best ZIP, `metrics.json`, `test_predictions.csv`. File phân tích được tải lên `analysis/`.
- `/content/CLRR/`, `outputs_extra/`, `backups_extra/` là bản làm việc trên Colab. Khi runtime mất, kiểm tra artifact trên Hugging Face; script huấn luyện bỏ qua run khi đủ best ZIP, metrics và predictions trên remote, hoặc khôi phục checkpoint đã backup.

## Việc tiếp theo

1. Mở notebook mới nhất, chạy các cell thiết lập/đăng nhập cần thiết, rồi chạy lại **cell 6**. Cell này `git pull` và reload module phân tích trước khi chạy. Xác nhận có đủ năm file prediction, `old_model_scores.csv` và đồ thị cosine trong `analysis/` của repo backup.
2. Khi cell 6 hoàn tất, chạy **cell 7** (hai run ByT5), rồi **cell 8** (hai ablation mT5). Mỗi run cần đủ best ZIP, metrics và test predictions trên Hugging Face.
3. Chạy **cell 9** để tạo `all_scores.csv`, `paired_bootstrap.csv`, `case_candidates.csv` và `analysis_notes.txt`. Kiểm định paired bootstrap lấy mẫu 10.000 lần, seed 42; hiệu chỉnh Holm cho sáu phép thử chính. Mỗi model hiện chỉ có một seed huấn luyện, nên phép thử này không đo biến thiên giữa các seed. Các câu ví dụ về phụ tố Amis cần kiểm tra thủ công trước khi đưa vào bài.

Nếu cell nào lỗi, lưu **traceback gốc phía trước `CalledProcessError`** cùng commit đang chạy trên Colab; chỉ dòng `CalledProcessError` không cho biết nguyên nhân.
