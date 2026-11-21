# ComputEL-10: góp ý reviewer, đề xuất sửa bài và kiểm tra overlap bổ sung

**Cập nhật:** 16/11/2026 (Asia/Saigon).  
**Bài:** *Cross-Layer Residual Rewiring and Latent Semantic Regularization for Endangered-Language Translation: A Case Study on Amis–Mandarin*.  
**Submission:** #53, ComputEL-10 2027.  
**Mục đích:** tài liệu bàn giao cho người sửa bài; phân biệt nhận xét gốc, đề xuất xử lý và những việc thực sự đã chạy.

## 1. Phạm vi và nguồn

- Bản bài đối chiếu: `C:/Users/Lenovo/Downloads/53-2 (1).pdf`.
- Nguồn nhận xét: `C:/Users/Lenovo/Downloads/LoA-ComputEL-10.pdf`, reviewer #1 ở trang 2, #2 ở trang 2–3, #3 ở trang 3–4.
- Thư thông báo bài đã được chấp nhận; các reviewer đều có nhận xét nghiêng về acceptance. Tài liệu này không suy rằng mọi đề xuất thí nghiệm là điều kiện bắt buộc của acceptance.
- Kế hoạch hiện tại: sửa bản bài dùng kết quả lịch sử, tận dụng artifacts còn lưu, không huấn luyện lại. File này không tự sửa manuscript, bảng điểm hoặc dữ liệu.
- **Giữ số lịch sử không đồng nghĩa mọi số đã được xác minh.** Các sai lệch đã biết giữa bảng cũ và predictions phải được xử lý hoặc giải thích trước khi gọi số đó là kết quả có thể tái tính.

| Reviewer | Appropriateness | Clarity | Originality / Innovativeness | Nhận định tổng quát |
|---|---:|---:|---:|---|
| #1 | 5/5 | 5/5 | 4/5 | Đánh giá tích cực về kỹ thuật mới; cần giải thích metric và code tái lập. |
| #2 | 5/5 | 4/5 | 4/5 | Nghiêng nhận; cần làm rõ novelty, diễn đạt, code và giới hạn một cặp ngôn ngữ. |
| #3 | 4/5 | 4/5 | 4/5 | Borderline, leaning accept; lo ngại overlap, biến thiên seed và giới hạn đánh giá ngữ pháp/ứng dụng. |

## 2. Bảng góp ý gốc và đề xuất sửa tương ứng

Các trích dẫn giữ nguyên cách viết/typo của reviewer; chỉ nối các dòng bị ngắt trong PDF. “Đề xuất” là phương án của nhóm sửa bài, không phải lời reviewer.

| ID | Reviewer | Câu gốc | Đề xuất sửa / bổ sung | Tình trạng và giới hạn |
|---|---|---|---|---|
| R1-01 | #1 | “while their technique does not seem to significantly improve BLEU over standard fine-tuning it does significantly improve chrF++.” | Thêm thảo luận BLEU–chrF++, phân biệt mức tăng điểm với statistical significance; giải thích ảnh hưởng word n-gram trong Mandarin. | **Reviewer hiểu nhầm significance:** bản bài báo chrF++ CI [−0.19, 13.25], p = 0.12. Không viết rằng chrF++ significant theo câu này. Có phân tích metric đã lưu; chưa tích hợp vào bản bài. |
| R1-02 | #1 | “no clear replicable code” | Thêm Availability; công bố predictions, script chấm điểm, cấu hình/checkpoint có nguồn xác nhận. Phân biệt code hiện tại với code lần train lịch sử. | Chưa đóng gói bản phát hành trong công việc này. Không hứa tái lập chính xác training cũ nếu thiếu source/environment tương ứng. |
| R2-01 | #2 | “Overall, I would lean towards acceptance and encourage the author to reports results to at least one other language pair.” | Nếu không chạy thêm được, giữ phạm vi case study; nêu rõ chưa có bằng chứng khái quát sang cặp ngôn ngữ khác cho bộ kết quả này. | **Chưa đáp ứng bằng thí nghiệm.** Viết limitations không thay thế việc thử thêm ngôn ngữ. Không trộn kết quả từ recipe khác vào bảng cũ. |
| R2-02 | #2 | “it would be nice to know how it differs from the existing representation residual already included in the transformers architecture” | Thêm một đoạn hoặc sơ đồ đối chiếu residual thông thường với CLRR: tầng nguồn/đích, khoảng cách d, vị trí cộng, α, detach và phạm vi encoder/decoder. | Làm được bằng mô tả phương pháp; phải khớp implementation có bằng chứng, không suy đoán code train cũ. |
| R2-03 | #2 | “typo? largest improvement in sentences matching ma-. (line 021)” | Viết rõ “sentences whose Amis source matches the surface prefix pattern ma-”; dẫn quy tắc matching và số câu. | Giải thích đây là nhóm heuristic có overlap, không phải tập nhãn voice được chuyên gia xác nhận. |
| R2-04 | #2 | “The transition is a little bit awkward and seems disconnected to the internal architecture changes that the authors are introducing.” | Sửa đoạn chuyển từ “existing workflows and resources” sang vấn đề representation; giải thích cụ thể việc tận dụng parallel sentences có sẵn. | Không khẳng định phù hợp workflow tư liệu hóa thực tế nếu chưa có triển khai/đánh giá người dùng. |
| R2-05 | #2 | “It is not clear to me why the other focus on a single language pair (and single dataset).” | Giải thích lý do chọn Amis–Mandarin và giới hạn thiết kế nghiên cứu; thu hẹp kết luận theo dataset/direction đã đánh giá. | Sửa được lập luận, nhưng chưa xác nhận generalization. |
| R2-06 | #2 | “it is unclear whether it differs to other kind of morphological phenomenon that are usually carried by tokenization methods” | Làm rõ động cơ liên quan hình thái/voice và phân biệt tokenizer với can thiệp representation. | Không tuyên bố CLRR khắc phục tokenizer hoặc bảo toàn voice khi chưa có thí nghiệm/ngữ liệu chú giải hỗ trợ. |
| R2-07 | #2 | “its Mandarin ref- erence, tokenized by the backbone: backbone of what?” | Nêu rõ reference Mandarin được tokenize bằng tokenizer của backbone đang xét; ghi tên mô hình và phân biệt tokenizer phục vụ encoder với tokenizer phục vụ chấm metric. | Cần sửa câu và kiểm tra khớp phương pháp. Không tokenize/decode reference bằng tokenizer model để thay raw reference khi chấm điểm chuẩn. |
| R2-08 | #2 | “It is not clear if the code will be release” | Thêm link và nội dung cụ thể của gói artifacts; nêu phần nào có thể kiểm chứng. | Cùng việc với R1-02; chưa công bố package trong công việc này. |
| R3-01 | #3 | “whether dictionary entries or near-duplicate examples cross the train/test split” | Bổ sung audit exact/near-duplicate, số lượng, tiêu chí, ví dụ và giới hạn thiếu dictionary-entry IDs. Nêu rõ dùng split có sẵn của dataset. | **Đã chạy kiểm tra chuỗi trên CSV hiện tại**, xem mục 3. **Chưa kiểm tra được overlap theo entry**, chưa đo tác động lên điểm model. |
| R3-02 | #3 | “whether the gains persist across training seeds” | Nêu rõ bộ kết quả lịch sử là single-seed; sentence bootstrap không đo training-seed variability. | **Chưa đáp ứng bằng thí nghiệm.** Không dùng tính ổn định trên bootstrap câu thay cho replication. |
| R3-03 | #3 | “Prefix groups are heuristic and overlapping, and the three qualitative examples are not speaker- or linguist-validated” | Ghi rõ nhóm tiền tố chỉ là proxy, có overlap; kiểm tra provenance ví dụ và dùng predictions thật có ID bản ghi. | Chưa có đánh giá của người nói/nhà ngôn ngữ học. Không gọi khác biệt bản dịch quan sát được là grammatical preservation đã xác nhận. |
| R3-04 | #3 | “The evaluation does not establish use in a documentary workflow, community uptake, or speaker-validated grammatical quality.” | Thu hẹp practical-value claims: đây là đánh giá phương pháp trên dictionary-example translation; triển khai cộng đồng và đánh giá ngữ pháp là công việc tiếp theo. | Có thể sửa diễn giải ngay; không tự coi đây là bằng chứng ứng dụng đã hoàn tất. |

## 3. Kiểm tra bổ sung đã chạy: overlap giữa các split

### 3.1. Mục đích

Trả lời một phần R3-01 bằng dữ liệu thực tế đang dùng. Dataset được kế thừa từ nghiên cứu trước vẫn cần kiểm tra split vì kết luận hiệu quả của bài phụ thuộc vào mức độ độc lập của train và test. Đây không phải thao tác sửa dataset hoặc quy lỗi cho tác giả dữ liệu.

Manifest dataset hiện tại ghi nguồn `https://github.com/francisdzheng/amis-mandarin`, paper `https://aclanthology.org/2022.nlp4dh-1.11/`, và việc di chuyển các CSV byte-for-byte từ thư mục cũ. Audit này xác nhận các CSV hiện tại và hash của chúng, không thực hiện đối chiếu mới với dữ liệu upstream hoặc khôi phục ID mục từ.

### 3.2. Dữ liệu và phương pháp

| Hạng mục | Chi tiết |
|---|---|
| Dữ liệu | `data_processed/amis_mandarin/{train,validation,test}.csv`, cột `source,target` |
| Số bản ghi | Train 4.600; validation 576; test 575 |
| Phạm vi | Toàn bộ 5.625.800 cặp bản ghi giữa các split; không lấy top-k hoặc mẫu ngẫu nhiên |
| Raw exact | Chuỗi CSV bằng nhau, không strip/casefold |
| Chuẩn hóa | NFKC → casefold → bỏ ký tự Unicode loại P* (dấu câu) → gộp khoảng trắng |
| Độ tương đồng | Max hai chiều `difflib.SequenceMatcher.ratio`, `autojunk=False`, mức ký tự, thang 0–1 |
| Lưu ứng viên | Source similarity ≥ 0,85 **hoặc** target similarity ≥ 0,90 |
| Nhóm báo cáo gần trùng hai phía | Source similarity ≥ 0,85 **và** target similarity ≥ 0,90 |
| Kiểm tra ngưỡng chặt hơn | Cả nguồn và đích ≥ 0,95 |
| Câu ngắn | Vẫn giữ và gắn cờ; không tự loại khỏi số liệu |
| Trong từng split | Có thống kê exact duplicates; không chạy near-duplicate nội bộ |
| Can thiệp | Không thay đổi dataset/split, không lọc test, không train hoặc tính lại điểm dịch |

Ngưỡng là heuristic cho audit, chưa hiệu chỉnh theo nhãn chuyên gia. Similarity không phải Levenshtein, semantic similarity hoặc xác suất leakage. Screen theo giới hạn trên từ multiset ký tự chỉ giảm chi phí tính toán; mọi cặp vượt ngưỡng vẫn được giữ để tính similarity.

### 3.3. Kết quả

Các con số là **số cặp bản ghi**, không phải số mục từ hoặc số câu duy nhất trừ khi ghi rõ.

| Kiểm tra | Train–validation | Train–test | Validation–test |
|---|---:|---:|---:|
| Source trùng nguyên văn | 0 | 0 | 0 |
| Cả cặp dịch trùng nguyên văn | 0 | 0 | 0 |
| Cả cặp dịch trùng sau chuẩn hóa | 0 | 2 | 0 |
| Target trùng nguyên văn | 11 | 19 | 2 |
| Source ≥ 0,85 và target ≥ 0,90 | 9 | 13 | 3 |
| Cả source và target ≥ 0,95 | 6 | 5 | 0 |

- **13 cặp train–test** qua ngưỡng cả hai phía tương ứng **13/575 câu test = 2,26%**.
- **5 cặp** qua ngưỡng 0,95 cả hai phía tương ứng **5/575 câu test = 0,87%**.
- Hai cặp trùng sau chuẩn hóa đã nằm trong nhóm 13 và nhóm 5; **không cộng thêm**.
- 19 cặp target raw trùng giữa train–test liên quan 18 câu test; một câu test khớp hai dòng train. Target trùng riêng không đủ chứng minh cặp dịch bị lặp.
- Toàn bộ file ứng viên có 80 cặp vượt ngưỡng ở ít nhất một phía: 42 train–validation, 34 train–test và 4 validation–test. Không nhầm 34 train–test đơn ngữ với 13 train–test gần trùng hai phía.

### 3.4. Một số cặp cần xem xét

Chỉ số dưới đây là thứ tự bản ghi bắt đầu từ 1, không tính header. Đây là mô tả khác biệt chuỗi, không phải xác nhận các biến thể Amis có cùng nghĩa.

| Train | Test | Amis train | Amis test | Similarity nguồn / đích | Quan sát |
|---:|---:|---|---|---|---|
| 1913 | 171 | Acaw aka han ko niradoman no mako hana! | 'Acaw aka han ko niradoman no mako hana! | 1,0000 / 1,0000 | Khác dấu nháy đầu câu; target nguyên văn giống nhau. |
| 2884 | 267 | Ma'osi kako to acefel no tamako. | Ma'osi kako to 'acefel no tamako. | 1,0000 / 1,0000 | Khác dấu nháy trước acefel; target nguyên văn giống nhau. |
| 2810 | 125 | O pipanemnem no kawas i titaanan. | O pipanemnem no kawas ititaanan. | 0,9841 / 1,0000 | Khác khoảng trắng; target nguyên văn giống nhau. |
| 4533 | 536 | Konocen ko opih no kaciw! | Onocen ko opih no kaciw! | 0,9787 / 1,0000 | Khác chữ đầu; target nguyên văn giống nhau. |
| 3145 | 98 | Nga'ayay todohen ko katol no kolong a komaen. | Nga'ayay todohen ko katol no kolong a komaem. | 0,9767 / 1,0000 | Khác ký tự cuối của từ cuối; target nguyên văn giống nhau. |

### 3.5. Giới hạn cần giữ trong manuscript

1. Không có raw exact duplicate không chứng minh split hoàn toàn độc lập.
2. Dấu nháy có thể mang giá trị ngôn ngữ học trong Amis; bỏ dấu câu có thể gộp các chuỗi khác biệt. Hai normalized matches không tự động là semantic duplicates.
3. CSV không có dictionary-entry IDs; chưa xác định được các câu có cùng mục từ điển hay không.
4. Chưa đo tác động của các ứng viên lên điểm dịch hoặc mức tăng của phương pháp; không kết luận model ghi nhớ dữ liệu.
5. Không loại các câu nghi trùng khỏi điểm chính một cách hậu nghiệm. Nếu làm phân tích tập con về sau, phải ghi là phân tích bổ sung, nêu quy tắc và giữ điểm toàn test.
6. Kiểm tra này trả lời **một phần** lo ngại overlap của #3; không trả lời seed variability hoặc cross-language generalization.

### 3.6. Đoạn tiếng Anh đề xuất cho phần dữ liệu/limitations

Đây là đoạn dự thảo; cần thêm citation dataset theo bibliography của bài.

> We audited the provided 4,600/576/575 train/validation/test records for cross-split string overlap. No verbatim source sentence or bilingual pair was shared across splits. After NFKC normalization, case folding, punctuation removal, and whitespace normalization, two train–test pairs became identical on both sides. A heuristic character-similarity screen (source ≥ 0.85 and target ≥ 0.90) flagged 13 train–test pairs involving 13 test sentences (2.26%), including those two normalized matches; five pairs reached 0.95 on both sides. These are string-overlap candidates rather than confirmed semantic duplicates. In particular, apostrophe differences may be linguistically meaningful in Amis. Dictionary-entry overlap could not be established because entry identifiers were unavailable in the processed CSVs. We retained the original records and splits; this audit does not quantify the effect of overlap on model scores.

### 3.7. Artifacts, tái chạy và xác minh

| Tệp | Nội dung |
|---|---|
| [WRITER_BRIEF_VI.md](../../outputs_rebuttal/amis_split_overlap_20261116/WRITER_BRIEF_VI.md) | Báo cáo audit tiếng Việt độc lập, ví dụ và đoạn tiếng Anh |
| [summary.json](../../outputs_rebuttal/amis_split_overlap_20261116/summary.json) | Thống kê raw/normalized, các ngưỡng similarity và hash dữ liệu |
| [exact_matches.csv](../../outputs_rebuttal/amis_split_overlap_20261116/exact_matches.csv) | Match theo source/target/pair và raw/normalized; một cặp có thể xuất hiện ở nhiều loại |
| [near_candidates.csv](../../outputs_rebuttal/amis_split_overlap_20261116/near_candidates.csv) | Toàn bộ 80 cặp qua ngưỡng ở ít nhất một phía |
| [train_test_bilingual_candidates.csv](../../outputs_rebuttal/amis_split_overlap_20261116/train_test_bilingual_candidates.csv) | 13 cặp train–test qua ngưỡng cả hai phía, kèm nguyên văn |
| [verification.json](../../outputs_rebuttal/amis_split_overlap_20261116/verification.json) | Kết quả kiểm tra hash, chỉ số bản ghi, text và tính lại similarity |
| [audit_amis_split_overlap.py](../../scripts/analysis/audit_amis_split_overlap.py) | Script kiểm tra có thể chạy lại, không tải model |

```powershell
conda run --no-capture-output -n clrr python scripts/analysis/audit_amis_split_overlap.py
```

Đã xác minh: hash và số bản ghi của cả ba CSV; text và ID của 80 cặp ứng viên; tính lại similarity của 80 cặp; kiểm tra screen không bỏ sót trên tập synthetic ở ba ngưỡng 0,85/0,90/0,95, gồm chuỗi rỗng, Unicode và câu ngắn. Hash dữ liệu sau audit không đổi.

| Split | SHA-256 |
|---|---|
| Train | `4e13d83f035a18c67e1b2e3990baff3aa92154a34b754a4d0cdef9a71d85b8e6` |
| Validation | `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4` |
| Test | `2e3cea3f9011ef6ee237a437b57a1ac21bd9fae9682473bce00c99a80afde0aa` |

## 4. Những vấn đề của bản số cũ cần người viết xử lý

Các mục sau là phát hiện từ hồ sơ project, không phải mọi mục đều do reviewer nêu. Không cần train lại để thừa nhận hoặc sửa phạm vi claim.

| Vấn đề | Việc đề xuất | Nguồn hiện có |
|---|---|---|
| mBART LSR-only 20,08 / 16,07 trong bản bài không khớp lần chấm lại predictions | Xác định bộ predictions/reference dùng cho bảng. Hồ sơ ghi audit A: 19,6291 / 18,5753; FP32 regeneration B: 19,7067 / 18,6169. Không trình bày chúng như cùng một kết quả và không trộn A/B để kiểm định. | [EXPERIMENTS_UPDATED.md, mục 7](../experiments/EXPERIMENTS_UPDATED.md#7-doi-chieu-claim-goc-voi-bang-chung-hien-tai); [metrics_standardized_scores.csv](../../results/metrics_standardized_scores.csv) |
| Runtime/overhead lịch sử chưa đủ bằng chứng controlled benchmark | Rà từng giá trị có log; không tiếp tục claim +2,2% train / <0,4% inference như đã xác minh. Có thể bỏ claim nếu không tìm được chứng cứ. | [EXPERIMENTS_UPDATED.md](../experiments/EXPERIMENTS_UPDATED.md), mục 7 |
| Ví dụ định tính trong paper chưa khớp saved test sources theo audit cũ | Chọn ví dụ có thật từ predictions và kèm record IDs; không gọi ví dụ không truy được nguồn là đầu ra thực của test set. | [EXPERIMENTS_UPDATED.md](../experiments/EXPERIMENTS_UPDATED.md), mục 7 |
| mT5 LSR-only và sensitivity sweep còn thiếu chứng nhận | Không nâng thành kết quả đã được tái lập hoặc optimum đã chứng minh. Rà evidence trước khi giữ trong appendix. | [EXPERIMENTS_UPDATED.md](../experiments/EXPERIMENTS_UPDATED.md), mục 7 |

Sổ gốc dùng để truy nguồn bản paper: [EXPERIMENTS_ORIGINAL_20261003.md](../archive/EXPERIMENTS_ORIGINAL_20261003.md). Sổ hiện hành dùng để kiểm tra các đợt audit/corrected runs: [EXPERIMENTS_UPDATED.md](../experiments/EXPERIMENTS_UPDATED.md). Không ghép các cohort khác recipe/seed/inference thành một bảng matched hoặc p-value mới.

## 5. Checklist bàn giao cho người viết

- [x] Chạy audit overlap trên CSV hiện tại và lưu script/artifacts.
- [x] Xác minh hash, record IDs và similarity; giữ nguyên dataset.
- [ ] Thêm kết quả audit vào Data/Limitations, ghi rõ chưa có entry IDs.
- [ ] Làm rõ CLRR so với residual thông thường; thêm sơ đồ nếu phù hợp.
- [ ] Sửa “ma-”, “backbone”, và đoạn chuyển về workflow/resources.
- [ ] Thêm thảo luận BLEU–chrF++, giữ đúng p = 0.12 và giới hạn single-seed.
- [ ] Xử lý các sai lệch số lịch sử đã biết trước khi chốt bảng.
- [ ] Thay ví dụ không truy được nguồn bằng predictions thật; hạn chế claim ngữ pháp.
- [ ] Thêm Availability và đóng gói artifacts có thật; không gọi code hiện tại là code train cũ khi chưa chứng minh được.
- [ ] Rà runtime/overhead và bỏ hoặc hạ các claim thiếu evidence.
- [ ] Nêu rõ chưa đáp ứng replication nhiều seed và đánh giá thêm cặp ngôn ngữ cho bộ kết quả lịch sử.

**Ưu tiên thực tế:** đưa audit đã chạy vào bài → làm rõ phương pháp → giải thích metric → Availability và provenance → rà claims/ví dụ/số liệu. Không cần mở thêm training để hoàn thành các sửa đổi văn bản và bàn giao evidence đã có.
