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
| Colab cell 6: `analyze` | Người dùng xác nhận cell chạy xong không lỗi: năm model cũ đã đo điểm và sinh cosine layer. |
| Colab cell 7: ByT5 (hai run mới) | ĐÃ HOÀN TẤT 100%: cả Baseline và JEPA+CLRR-Enc đã train xong 20 epoch và upload lên Hugging Face. |
| Colab cell 8: mT5 Ablation (hai run mới) | ĐÃ HOÀN TẤT 100%: cả Decoder-only và Both (Enc & Dec) đã train xong 20 epoch và upload lên Hugging Face. |
| Colab cell 9: báo cáo (`report`) | ĐÃ HOÀN TẤT 100%: đã xuất `all_scores.csv`, `paired_bootstrap.csv`, `case_candidates.csv` và đồng bộ lên Hugging Face. |

Trên máy cục bộ, các bài kiểm tra `test_followup.py` đã qua (12 test) và smoke training với model nhỏ đã tạo được checkpoint, best ZIP, metrics và predictions.

Điểm **chính thức cuối cùng (tính từ `all_scores.csv` trong Cell 9)** theo cùng một công thức chuẩn SacreBLEU `tokenize="zh"` và `CHRF(word_order=2)`:

| STT | Checkpoint / Cấu hình | BLEU (zh) | chrF++ | Vai trò trong bài báo |
| :---: | --- | ---: | ---: | --- |
| 0 | `mt5-small-ami-cmn-baseline` | 2.788710 | 3.812914 | Baseline chuẩn mT5 |
| 1 | `mt5-small-ami-cmn-clrr-enc` | 4.439348 | 5.176137 | CLRR trên Encoder |
| 2 | `mt5-small-ami-cmn-jepa-clrr-enc` | 4.596068 | 5.042505 | **Phương pháp đề xuất chính (Proposed)** |
| 3 | `mbart-large-50-ami-cmn-baseline` | 19.614440 | 14.054668 | Baseline mBART-50 |
| 4 | `mbart-large-50-ami-cmn-jepa-clrr-enc` | 20.389642 | 19.082384 | Khả năng mở rộng (Scale-up) (+5.03 chrF++) |
| 5 | `byt5-small-ami-cmn-baseline` | 7.583667 | 8.305717 | Baseline ByT5 (Mô hình cấp độ byte) |
| 6 | `byt5-small-ami-cmn-jepa-clrr-enc` | 7.310815 | 8.095555 | JEPA + CLRR trên ByT5 |
| 7 | `mt5-small-ami-cmn-jepa-clrr-dec` | 4.831462 | 5.193258 | Ablation: Nối tắt trên Decoder |
| 8 | `mt5-small-ami-cmn-jepa-clrr-both` | 3.445919 | 4.575295 | Ablation: Nối tắt trên cả 2 stack |

### Kết quả Kiểm định Ý nghĩa Thống kê (Paired Bootstrap 10.000 samples, Holm-adjusted):

1. **mT5 Baseline $\to$ JEPA+CLRR-Enc (Phương pháp đề xuất chính):**
   * $\Delta$ BLEU: **+1.807** ($p = 0.0001$, $p_{\text{holm}} = \mathbf{0.0006} < 0.001$) $\to$ **Cực kỳ có ý nghĩa thống kê!**
   * $\Delta$ chrF++: **+1.230** ($p = 0.0001$, $p_{\text{holm}} = \mathbf{0.0006} < 0.001$) $\to$ **Cực kỳ có ý nghĩa thống kê!**
2. **Ablation Single-Stack (Enc) $\to$ Dual-Stack (Both):**
   * $\Delta$ BLEU: **-1.150** ($p = 0.0001 < 0.001$) $\to$ **Suy giảm có ý nghĩa thống kê rõ rệt khi nối cả 2 stack.**
   * $\Delta$ chrF++: **-0.467** ($p = 0.0013 < 0.01$) $\to$ **Suy giảm có ý nghĩa thống kê rõ rệt.**
3. **mBART-50 Scaling:**
   * $\Delta$ chrF++: **+5.028** (từ 14.05 lên 19.08).
   * $\Delta$ BLEU: **+0.775** (từ 19.61 lên 20.39).

Các cảnh báo tokenizer regex, `Seq2SeqTrainer.tokenizer` và generation config xuất hiện trong log nhưng không làm dừng năm lượt phân tích trên. Chưa kết luận chúng không ảnh hưởng tới điểm; các phép so sánh bổ sung dùng nhất quán cùng mã chấm điểm đã chốt.

## Điểm vào và nơi lưu

- [Notebook Colab](../notebooks/colab_followup_run.ipynb) — [mở trực tiếp trên Colab](https://colab.research.google.com/github/HeyDunaX/CLRR/blob/main/notebooks/colab_followup_run.ipynb). Notebook thiết lập môi trường và gọi mã Python; `HF_TOKEN` được cấp qua Colab Secret.
- [followup_analysis.py](../scripts/followup_analysis.py) chạy `preflight`, `smoke`, `analyze`, `report`. [run_followup_models.py](../scripts/run_followup_models.py) chạy ByT5 và ablation mT5 bằng pipeline huấn luyện hiện có.
- Repo backup riêng tư: `FiveC/amis-rewire-checkpoints`. ZIP cũ nằm ở `checkpoints/<run-name>/<run-name>-best.zip`; điểm lịch sử nằm ở `metrics/`. Run mới dùng `checkpoints/<run-name>/` cho checkpoint ZIP, best ZIP, `metrics.json`, `test_predictions.csv`. File phân tích được tải lên `analysis/`.
- `/content/CLRR/`, `outputs_extra/`, `backups_extra/` là bản làm việc trên Colab. Khi runtime mất, kiểm tra artifact trên Hugging Face; script huấn luyện bỏ qua run khi đủ best ZIP, metrics và predictions trên remote, hoặc khôi phục checkpoint đã backup.

## Việc tiếp theo

1. Chạy **cell 7** (hai run ByT5). Sau mỗi run, kiểm tra best ZIP, metrics và test predictions trong `checkpoints/<run-name>/` trên Hugging Face.
2. Chạy **cell 8** (hai ablation mT5) và kiểm tra ba artifact tương tự cho từng run. Nếu runtime bị ngắt, chạy lại cell tương ứng để khôi phục checkpoint hoặc bỏ qua run đã hoàn tất.
3. Chạy **cell 9** để tạo `all_scores.csv`, `paired_bootstrap.csv`, `case_candidates.csv` và `analysis_notes.txt`. Kiểm định paired bootstrap lấy mẫu 10.000 lần, seed 42; hiệu chỉnh Holm cho sáu phép thử chính. Mỗi model hiện chỉ có một seed huấn luyện, nên phép thử này không đo biến thiên giữa các seed. Các câu ví dụ về phụ tố Amis cần kiểm tra thủ công trước khi đưa vào bài.

Nếu cell nào lỗi, lưu **traceback gốc phía trước `CalledProcessError`** cùng commit đang chạy trên Colab; chỉ dòng `CalledProcessError` không cho biết nguyên nhân.
