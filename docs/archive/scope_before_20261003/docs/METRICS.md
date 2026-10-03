# Giao thức metrics chuẩn của CLRR

File này định nghĩa cách đo; **không lưu bảng điểm thực nghiệm**.
Số paper gốc: [experiments gốc đã archive](archive/EXPERIMENTS_ORIGINAL_20261003.md).
Số sau xác minh: [EXPERIMENTS_UPDATED.md](EXPERIMENTS_UPDATED.md).

## 1. Giao thức chính duy nhất

- Reference là `target` nguyên bản trong `data_processed/<dataset>/test.csv`, đúng thứ tự và đủ số câu. Không tokenize/decode reference bằng tokenizer của mô hình.
- Hypothesis là predictions đã decode; chỉ bỏ whitespace ở hai đầu. Giữ nguyên punctuation, case và mọi prediction rỗng; không lọc câu khó hoặc đổi Unicode normalization.
- **BLEU:** SacreBLEU, corpus-level, case-sensitive, order 4, smoothing `exp`, `effective_order=False`; tokenizer `zh` cho Mandarin, `13a` cho Spanish/English.
- **chrF++ chuẩn:** `CHRF(char_order=6, word_order=2, beta=2, lowercase=False, whitespace=False, eps_smoothing=False)` trên văn bản gốc. Không dùng TokenizerZh, SentencePiece hay tách từ bên ngoài trước chrF++.
- Điểm ở thang 0–100; bảng hiển thị 4 chữ số thập phân, CSV/JSON giữ full precision. Không lấy trung bình sentence-level BLEU/chrF++ để thay corpus score.

Định nghĩa và reporting theo [SacreBLEU](https://github.com/mjpost/sacrebleu), [chrF++: words helping character n-grams](https://aclanthology.org/W17-4770/) và [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/). Cấu hình giữa các paper có thể khác; chỉ so điểm trực tiếp khi cùng test/reference và metric signature.

### Signatures đã dùng — SacreBLEU 2.6.0

```text
BLEU Mandarin: nrefs:1|case:mixed|eff:no|tok:zh|smooth:exp|version:2.6.0
BLEU Spanish/English: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF++: nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

Lệnh tương đương sau khi xuất reference gốc và predictions thành hai file UTF-8, mỗi dòng một câu:

```powershell
conda run -n clrr python -m sacrebleu reference.detok.txt -i prediction.detok.txt -m bleu chrf --tokenize zh --chrf-word-order 2 --format json --width 4
```

Đổi `--tokenize zh` thành `--tokenize 13a` cho Spanish/English. Option này dành cho BLEU; không tiền xử lý chrF++.

### Phân tích phụ

`chrF++ (TokenizerZh diagnostic)` là một phép đo khác: áp dụng TokenizerZh lên cả hypothesis và reference trước CHRF. TokenizerZh tách ký tự Hán và phần ngoài Hán; không phải morphological word segmentation. Không gọi số này là chrF++ chuẩn hoặc trộn với số chính. Không kết luận chrF++ chuẩn bị lỗi chỉ vì tiếng Trung thiếu khoảng trắng.


## 2. Precision và checkpoint selection

Các lần generate mới phải khai báo precision, TF32 policy, beams, batch/padding,
max_length, language codes, checkpoint revision và environment.
Lượt03/10 dùng FP32weights/generation,TF32off,beam4,batch8,padmultiple8.
CLI training hỗ trợ `--generation-precision fp32|bf16`; giữ training precision riêng.
Sau revision03/10, default FP32 được áp dụng ở trainer chính, NLLB, PEFT và hai comparator. Đây là cấu hình cho lần chạy mới, không sửa precision metadata hoặc scores của các run đã đóng băng.
Không thay checkpoint được chọn theo validation bằng checkpoint có test đẹp hơn.

## 3. Bootstrap và phiên bản kết quả

Paired corpus bootstrap10.000samples,seed42; resample cùng sentence indices
cho hai systems. CIpercentile95% của signed corpus delta; p-value theo
SacreBLEU centered absolute-difference bootstrap. Hai cách không phải test inversion.
Holm family phải khai báo và giữ rõ: audit42,character bổ sung50,FP32followup36.
Suite matched C dùng Holm24 (4comparisons ×2metrics ×3seeds), khác family B. Báo cáo sample std qua3training seeds và paired seed deltas; không gọi sentence bootstrap là seed stability.
Không ghép các family để suy significance mới; sentence bootstrap không thay training-seed uncertainty.
Chỉ so score trực tiếp khi cùng dataset,reference,signature và protocol;
không trộn predictions đã lưu với generate lại để tính kiểm định.
Nguồn CSV/JSON và toàn bộ số nằm ở file thực nghiệm sau.
