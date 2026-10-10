# Giao thức metrics chuẩn của CLRR

File này định nghĩa cách đo; **không lưu bảng điểm thực nghiệm**.
Số paper gốc: [experiments gốc đã archive](../archive/EXPERIMENTS_ORIGINAL_20261003.md).
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


## 4. Phạm vi manuscript hiện hành

Manuscript chỉ dùng Amis→Mandarin. A giữ28outputs, B giữ10checkpoint và4routing-off, C giữ12runs. Bảng lọc theo phạm vi nhưng p-value đã điều chỉnh giữ nguyên family gốc42/36/24; không tính lại family nhỏ hơn để tăng significance. Artifacts khoa học gốc bất biến, bản tài liệu trước thu gọn nằm trong archive.

## 5. Quy ước đo cơ chế và kiểm tra replay — bổ sung sau review V3

### Representation rank

Với matrix pooled representations `X` có n dòng và d chiều, center theo cột.
Giới hạn centered rank là `min(n−1,d)`; báo rõ source/target, pooling mask,
dtype và site (encoder block output trước final LayerNorm hay final output).
Không so ranks ở hai site khác nhau như cùng phép đo.

- **Singular-value entropy effective rank:** `exp(−Σ p_i log(p_i))`,
  `p_i = s_i/Σ s_i`, với `s_i` là singular values của centered X.
- **Covariance-energy entropy effective rank:** cùng entropy formula với
  `q_i = s_i²/Σ s_i²`.
- **Participation ratio:** `1/Σ q_i²`. Đây là chỉ số khác, phải mang tên riêng.
- Quy ước `0 log 0 = 0`; nếu toàn bộ spectrum bằng 0 thì báo zero-spectrum
  và rank=0 theo quy ước của project, không để phép chuẩn hóa tạo rank=1 giả.
- Lưu full spectrum; top50 không đủ tái tính entropy của toàn bộ spectrum.
  Dùng `s_i²` thay eigendecomposition covariance float32 khi cần tránh nhiễu
  các eigenvalues gần 0. Giữ raw representations để kiểm tra site/pooling.

### Gradient CE–LSR

Định nghĩa weighted LSR là đúng hệ số λ của run; counterfactual λ cho nhánh
không train LSR phải được gắn nhãn riêng. Với cùng parameter vector/group:

- Cộng `Σ||g_CE||²` trên mọi CE gradients tồn tại, độc lập với LSR gradients.
- Cộng `Σ||g_LSR||²` độc lập; missing gradient được coi bằng 0 trong vector.
- Dot product chỉ cần những vị trí cả hai tồn tại; cosine dùng **hai full norms**.
- Nếu norm CE=0 thì norm ratio undefined; nếu một norm=0 thì cosine undefined.
  Không báo “cosine=0” hoặc “không conflict” từ một gradient không tồn tại.
- Global group và các groups con phải có phạm vi tham số nhất quán. Shared
  embedding chỉ xuất hiện một lần trong partition; log tên/số phần tử mỗi group.
- Lưu từng batch indices, loss reconstruction, train/eval mode, precision,
  dropout seed, target-no-grad policy và code hash. Average cosine không chứng
  minh mọi batch/layer đều không conflict; norm ratio không phải Adam update ratio.

### Generation replay và route control

Chấm lại predictions đúng công thức chưa chứng nhận đúng generation protocol.
Trước queue, một checkpoint phải replay route-on cùng source/target codes,
weights/tokenizer revisions, FP32/TF32, beams, lengths, batch/padding và runtime.
Lưu prefix IDs, raw predictions và hash. Nếu replay lệch, dừng và giải thích
chênh lệch trước khi dùng on/off deltas làm bằng chứng cơ chế.

Một raw model chỉ có một routing wrapper; strength0 không tháo hook. Kiểm tra
strength0/raw logits và hook count, fresh cache mỗi forward/generate; nếu tháo
handles để route-off thì xác nhận không còn routing hooks. Receipt V3 và kế
hoạch sửa nằm ở bàn giao Mục 17; không sửa raw kết quả V3 thành số mới.

### Version đã xác minh: V4, 04/10/2026

Bàn giao Mục18 và `amis_followup_20261004_v4/independent_audit.json` ghi status
VERIFIED cho21endpointprobes/3route-controls. V4 dùng source `tl_XX`, target
`zh_CN`, pin tokenizer/checkpoint/codehash; route-on phải exact-match predictions
validation trước. Generation/representation FP32/TF32off; gradient train-mode
BF16/TF32off, checkpointingdisabled, cùng8×4indices/dropoutseeds đã ghi.
Teacher CE gộp bằng label-token-weighted mean, không mean8batchloss khi sốtokens
khác nhau. Các groups embedding dựa theo named_parameters phải đọc scope cụ
thể; không đồng nhất `embeddings` với riêng shared token embedding.
V4 không sửa protocol testbeam4 hoặc primary metric/selection của các runs cũ.

### V5 correction: beam validation khi phục hồi checkpoint

`training_args.bin` có thể lưu beam dùng ở lượt test cuối. Khi phục hồi model,
phải đặt rõ `generation_num_beams` cho validation trước `evaluate`, rồi đặt lại
cho test; metadata phải ghi số beam thực tế theo từng split. Không suy beam từ
CLI configuration nếu trainer args đã bị thay đổi sau khi parse.

V5 xác nhận Baseline seed42: validation17.873477BLEU/14.900876chrF++ là beam4;
beam1 đúng16.265888/13.901541. Cả3baseline archives tái lập chính xác575test
predictions cũ. Sửa nhãn validation không làm đổi metric implementation hoặc
test scores. Original selected step của Baseline42 chưa xác định; không dùng216.

Nguồn lượt sau: `amis_routing_diagnostics_20261004_v5/validation_inventory_21_corrected_v5.json`.
12entries có validation generation metrics đã remeasure;9entries còn lại ghi rõ
historical not remeasured. eval_loss lịch sử và CE fixed32 có scope riêng.
Inferenceα/group interventions chỉ dùng validation; test beam1 chỉ là control
khóa trước cho Full/tuned. Bootstrap5000draws V5 là intervals mô tả chưa chỉnh
multiple comparisons, conditional on3frozen seeds; không thay training-seed inference.
