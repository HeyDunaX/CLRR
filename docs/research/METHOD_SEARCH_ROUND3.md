# Method search vòng 3 — lỗi qua seed, độ hỗ trợ từ và novelty

2026-11-18. Phạm vi: đọc project, CPU trên CSV validation sẵn có, primary literature và hai vòng phản biện NotebookLM. Không mở GPU/training/test mới, không sửa recipe, số lịch sử hoặc queue. Chỉ thêm báo cáo và artifacts phân tích.

## 1. Quyết định hiện hành

**Chưa có method mới được nhận để code/train.** Lượt này tìm được một liên hệ mới về độ hỗ trợ từ nguồn, nhưng chưa xác định nguyên nhân có thể can thiệp. Không gọi subword consistency hoặc lexical routing có sẵn là sáng chế mới.

Hai hướng kiểm tra trong lượt này: A = khả năng tổ hợp từ nguồn qua các cách phân đoạn; B = liên kết tín hiệu từ nguồn với lựa chọn nội dung đích theo ngữ cảnh. A tối thiểu trùng cơ chế prior; B còn là giả thuyết, chưa có thuật toán đủ cụ thể để xác nhận khác biệt. Không coi B đã thất bại thực nghiệm hoặc đã được chứng nhận novel.

## 2. Nguồn và tái lập

Đọc run inventory/cohort/metrics đã được khóa trong [vòng 2](D:/Code/CLRR/docs/research/METHOD_SEARCH_ROUND2.md), [RUN_EVIDENCE.csv](D:/Code/CLRR/outputs/experiments/method_search_round2/RUN_EVIDENCE.csv). Không gom các cohort khác selector/backend/compute thành một ranking.

Nguồn CPU lần này:

| File | Vai trò |
|---|---|
|`data_processed/amis_mandarin/train.csv`|4.600 cặp gốc, không thay dữ liệu|
|`outputs/experiments/cmc_estimator_review/shared_errors/ROWS.csv`|576 source/reference và predictions W42/43/44 sẵn có|
|`outputs/experiments/method_search_round2/ANNOTATED_VALIDATION_SAMPLE.csv`|24 câu đã lấy ngẫu nhiên cố định trước lượt này; nhãn tương đối với reference của W42|
|`outputs/experiments/amis_residual_seed_diagnostics/inputs/R1/validation_predictions.csv`|Baseline42 gốc; không nhầm với B_NATIVE_RDROP beta2.5|
|`outputs/experiments/amis_consistency_strength_w_replication_rtx_suite/validation_error_audit/SYSTEM_DIAGNOSTICS.json`|Kiểm tra các phân tích độ dài/lặp có sẵn trước khi nghĩ thêm loss|

Script và SHA inputs: [reproduce.py](D:/Code/CLRR/outputs/experiments/method_search_round3/reproduce.py), [INPUT_MANIFEST.json](D:/Code/CLRR/outputs/experiments/method_search_round3/INPUT_MANIFEST.json).

Lệnh đã chạy từ `D:\Code\CLRR`:

```powershell
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/method_search_round3/reproduce.py
```

Script chỉ đọc bốn CSV trên, tính mô tả và ghi vào thư mục output vòng 3; không load model/optimizer/SSH/test. Kiểm tra thêm tokenizer cached bằng `AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', local_files_only=True)` không load thành công (`ConnectionError`); không có số đo subword/UNK nào được suy ra từ phép kiểm này. Không tải weights.

## 3. Bằng chứng mới và mâu thuẫn

### 3.1 Các seed có thật sự cùng chọn sai không?

**[ĐO ĐƯỢC]** Chuẩn hóa prediction bằng lowercase và chỉ giữ ký tự chữ/số: W42/43/44 trùng hoàn toàn ở **75/576 câu**. Trong số này **15 câu** có sentence chrF0 dưới10 ở cả ba seed. Ngưỡng10 là mô tả post-hoc, không phải nhãn hallucination đã xác minh.

Trong mẫu24 đã cố định, chỉ4 câu trùng ở cả3 seed:224,571,151,286. Hai câu224/151 tương thích reference;571 thay nội dung soup/delicious bằng dishes/sweet;286 mất mức cụ thể infertility thành generic illness so với reference. Không suy ra chỉ hai câu này sai, hoặc mọi câu không trùng đều khác loại lỗi. Các nhãn cũ chỉ áp dụng W42, không tự chuyển sang W43/44.

Nguồn: [SUMMARY.json](D:/Code/CLRR/outputs/experiments/method_search_round3/SUMMARY.json), [CROSS_SEED_SAMPLE.csv](D:/Code/CLRR/outputs/experiments/method_search_round3/CROSS_SEED_SAMPLE.csv).

**[SUY LUẬN]** “Các seed đồng thuận” không đủ để xác định câu đúng/sai hoặc mục tiêu negative learning. Các lỗi predicate/object có cả dạng lặp và dạng thay đổi theo seed; một loss phạt đồng thuận chung không suy ra trực tiếp từ dữ liệu này.

### 3.2 Độ hỗ trợ từ nguồn trong train

Tách từ bằng Unicode letters và apostrophe nội bộ, lowercase, bỏ dấu câu; đếm theo sentence document frequency. “Chưa gặp” chỉ nghĩa whole-word surface form không có trong train theo quy tắc này. **Không phải tokenizer OOV, morpheme OOV hoặc chứng minh tokenization sai.**

**[ĐO ĐƯỢC]** 486/576 câu có ít nhất một từ chưa gặp;90 câu toàn bộ từ đã gặp. Trong mẫu24,20 câu có từ chưa gặp.

| Nhóm validation | n | Baseline42 | W42 | W43 | W44 | W42−Baseline42 |
|---|---:|---:|---:|---:|---:|---:|
|Tất cả từ đã gặp|90|27.0050|25.2883|28.0424|26.6201|−1.7167|
|Có ít nhất một từ chưa gặp|486|15.6379|16.5621|16.2563|16.0053|+0.9242|

**Metric trong bảng là trung bình sentence chrF0, không phải corpus chrF, BLEU hoặc chrF++.** Không đưa bảng này thay main results. Signature `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`.

Spearman giữa tỷ lệ từ chưa gặp và mean sentence chrF của W3seed = **−0.308965**. Partial rank correlation sau khi residualize source word count và reference alphanumeric length = **−0.313313**. Không có kiểm định causal, CI hoặc confirmatory p trong phép mô tả này.

Nguồn: [WORD_SUPPORT_SUMMARY.json](D:/Code/CLRR/outputs/experiments/method_search_round3/WORD_SUPPORT_SUMMARY.json), [WORD_SUPPORT_ALL_ROWS.csv](D:/Code/CLRR/outputs/experiments/method_search_round3/WORD_SUPPORT_ALL_ROWS.csv).

**[SUY LUẬN]** Độ hỗ trợ từ nguồn là một trục phân tích có cơ sở hơn việc tiếp tục tăng consistency vô điều kiện. Tuy nhiên W42 đã cải thiện nhóm có từ chưa gặp so baseline42; không được nói W không giúp generalization. Chênh lệch nhóm toàn từ đã gặp đảo dấu giữa W seeds, nên chưa được suy ra một rule adaptive lambda từ bảng này.

Baseline42 reuse có replay best BLEU/chrF++ cùng epoch19/step684 trong vòng2; backend SDPA là reconstruction lịch sử, thiếu receipt đương thời. Chưa replay selector/baseline43/44 cho phân tích này. B_NATIVE_RDROP beta2.5 không được gọi baseline CE. Không suy rộng subgroup delta42 thành paired multi-seed effect.

### 3.3 Lỗi có đơn giản là chọn từ phổ biến hơn không?

**[ĐO ĐƯỢC]** Trong các biểu thức chọn tay sau khi xem prediction: 鈕扣 có1 train sentence, 褲子 có9; 不孕1, 生病8. Nhưng 鞋22, 口袋6; 竹蓆4, 行李3; 温4 và 溫11, 沸騰4. Các counts là substring sentence counts, không phải tokenizer frequency; các cụm có độ dài khác nhau, không dùng làm likelihood ratio.

Nguồn: [POSTHOC_EXPRESSION_SUPPORT.csv](D:/Code/CLRR/outputs/experiments/method_search_round3/POSTHOC_EXPRESSION_SUPPORT.csv).

**[SUY LUẬN]** Dữ liệu minh họa không hỗ trợ giải thích mọi lỗi bằng “target frequent-word bias”. **[PHỎNG ĐOÁN]** Một phần lỗi có thể liên quan composition/lexical selection; chưa định vị được encoder, routing hay prefix influence. Không dùng counts chọn tay để chốt calibration/frequency penalty.

## 4. Hai hướng tối thiểu và kiểm novelty trước training

### A — Tận dụng consistency, đổi sang segmentation views

Hiện tượng: unseen surface words gắn với quality thấp trong cả3 W seeds. Giả thuyết: random dropout không buộc mô hình hiểu cùng một từ khi phân đoạn khác nhau. Giả thuyết này chưa được đo trực tiếp.

Thiết kế tối thiểu để kiểm prior, **không phải method mới được công nhận**:

`L_A = 0.5 CE(y|s0(x)) + 0.5 CE(y|s1(x)) + lambda D(p(.|s0(x), y_<t), p(.|s1(x), y_<t))`.

`s0,s1` giữ nguyên chuỗi nguồn khi decode tokenizer, target prefix giữ nguyên; không tự sửa orthography/reference. D có thể là KL theo công thức prior; đổi thành symmetric KL không tự tạo contribution.

**Verdict: TRÙNG ở cơ chế tối thiểu segmentation-view consistency. Loại khỏi danh sách đóng góp thuật toán mới.** [Multi-view Subword Regularization, Wang et al.,2021](https://aclanthology.org/2021.naacl-main.40.pdf), §4, đã dùng deterministic/probabilistic segmentations, hai supervised losses và prediction divergence. Paper đánh giá cross-lingual transfer, không phải chính cohort mBART Amis; khác task không xóa overlap cơ chế. [Kudo2018](https://aclanthology.org/P18-1007.pdf) đã dùng stochastic segmentations cho NMT. Không kết luận phương án này vô ích về engineering.

[Ataman & Federico2018](https://aclanthology.org/P18-2049.pdf), §3, đã học word representation từ các đơn vị nhỏ qua bi-RNN với bilingual supervision. Vì vậy “giữ word boundary rồi compose subwords” cũng chưa đủ khác biệt.

Dự đoán cần kiểm nếu dùng như comparator sau này: error words phải thể hiện segmentation sensitivity vượt matched-correct controls, và can thiệp phải sửa gold-vs-wrong choice một cách chọn lọc. Representation drift đơn thuần không chứng minh lỗi composition hay translation gain. Không train A trong lượt này.

### B — Liên kết tín hiệu nguồn với lựa chọn nội dung đích theo ngữ cảnh

Hiện tượng: predicate/object substitutions trong mẫu, gồm cả câu source words đã biết và chưa biết; final-layer oracle trước đây chưa chứng minh selective gold recovery. Giả thuyết: tín hiệu từ vựng đã có nhưng bị gắn sai lựa chọn trong ngữ cảnh. **Chưa chứng minh cue đã được encoder biểu diễn đúng.**

Một đường sửa tối thiểu dễ nghĩ là bổ sung lexical correction `z'_t = z_t + c_theta(x, y_<t)` với c lấy từ source-span embeddings theo context. Đây mới là template; không gọi một hàm c chưa xác định là thuật toán novel.

**Verdict: GẦN prior; HOLD vì thiếu nguyên nhân và cơ chế khác biệt cụ thể.**

| Primary đã mở/đọc method | Điểm gần | Điều không được đánh đồng |
|---|---|---|
|[Nguyen & Chiang2018, Improving Lexical Choice](https://aclanthology.org/N18-1031.pdf), §4|Attention-weighted source embeddings qua FFNN, cộng lexical logits với decoder logits|Một template source lexical correction chưa đủ khác primitive này|
|[Liu et al.2021, Counterfactual Data Augmentation](https://aclanthology.org/2021.naacl-main.18.pdf), §3.1|Context/alignment-aware path-specific source/target phrase replacement|Paper có MLM/translation-LM/Gumbel-Max generative procedure; không chứng minh mọi binding objective tương đương|
|[DICTDIS2024](https://aclanthology.org/2024.findings-emnlp.643.pdf), §3.2|Dictionary candidates, copy gate và context-conditioned disambiguation|Có supplied constraints/dictionaries; corpus-internal method khác resource setting nhưng không tự đủ novelty|

Cùng corpus supervision không chứng minh hai method tương đương; cần so công thức, conditioning và information available. Tuy nhiên hiện chưa có lý do để gắn thêm một module correction trước khi xác minh cue/target binding đúng và selective failure.

Dự đoán bác bỏ B: nếu đổi đúng cue đã được người biết ngôn ngữ xác nhận không tạo thay đổi target đúng vượt matched unrelated-cue controls trên frozen model, hoặc lexical information không recoverable ở encoder dưới probe hợp lệ, không nhận giả thuyết “known cue nhưng binding sai”. Không dùng zero gradient/flat attention riêng lẻ làm kết luận.

## 5. NotebookLM debate và những điểm tôi bác bỏ từ phản hồi

Notebook `bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`, conversation `a5bcde00-88ad-48d7-9143-ce9f30e25b99`. Chỉ gửi mô tả lý thuyết và paper names; không gửi bản dịch/sample hoặc counts mới của project.

- Round A query `65a17693e934`: yêu cầu nearest prior, reviewer rejection và falsifier cho A/B.
- Round B query `5c443bcc4e5b`: phản biện equivalence overclaims, đối chiếu primary MVR/lexical routing/TCWR/DICTDIS, yêu cầu sửa reasoning.
- Full prompts/responses/citations: [ROUND_A](D:/Code/CLRR/outputs/experiments/method_search_round3/NOTEBOOKLM_ROUND_A.json), [ROUND_B](D:/Code/CLRR/outputs/experiments/method_search_round3/NOTEBOOKLM_ROUND_B.json).

Không nhận NotebookLM làm novelty certificate. Phản hồi có lỗi cần giữ trong audit:

1. A gắn trực tiếp với MVR theo primary; gọi toàn bộ mọi source binding bằng neural alignment là equivalence không có chứng minh.
2. Unseen words không đồng nghĩa tokenizer UNK; cũng không được mặc định vocabulary coverage hoàn chỉnh.
3. Flat attention + high LM probability không chứng minh sole LM causation; zero local gradient không loại saturation/redundancy.
4. Round B đổi sai TCWR thành “Target-Conditioned Word Representation / Tang”; primary đúng trong lượt này là **Translation-Counterfactual Word Replacement**, Liu/Kusner/Blunsom2021. Không đưa tên sai vào references.
5. Prompt tiếng Anh kiểu “The translation of [w] is:” không phải neutral prefix hợp lệ của mBART MT. Không áp dụng diagnostic đó. Empty source/isolated word cũng gây distribution shift, không tự tạo unconditional LM control.
6. Encoder cosine drift không tự chứng minh semantic failure; hai nhánh can thiệp vẫn cần matched controls và gold-vs-arbitrary correction specificity.

## 6. Bước kế tiếp có giới hạn — chưa triển khai GPU

Không tiếp tục vòng lặp “đặt tên method → train → mới tra prior”. Phần CSV đã làm được ở lượt này đã hoàn tất; những phép dưới đây chưa chạy:

1. Lấy đúng tokenizer artifact/revision và đo round-trip normalization, UNK, pieces/word, train subpiece coverage trên chính576 rows. Không gọi whole-word absence là tokenization bug. Phép này chạy CPU, không cần weights.
2. Xác minh một nhóm source cue/target expression bằng người biết Amis hoặc nguồn từ vựng có trích dẫn; hiện mẫu chỉ có Chinese-reference-relative judgments. Không tự tạo gold lexical alignments bằng đoán nghĩa Amis.
3. Nếu1–2 cho đủ premise, khóa **một** frozen validation diagnostic: alternate legal segmentations so với matched controls; conditional candidate phrase scores cùng prefix và length accounting, không English prompting, không optimizer/test. Cần quyền/job GPU riêng nếu sử dụng model trên RTX. Đây là phép đo đề xuất, không phải claim đã chạy.
4. Chỉ thiết kế module/loss sau khi intervention phân biệt được ít nhất một nguyên nhân với cạnh tranh. Nếu không đủ bằng chứng, dừng nhánh chứ không tăng lambda, mở thêm seeds hoặc chuyển data để tìm số đẹp. Giữ MVR/lexical routing làm nearest priors khi có candidate cụ thể.

Kết luận: hướng source lexical generalization đáng kiểm tra, nhưng hiện không có cơ sở chốt một method đủ novel cho ACL. Bằng chứng âm trước đây vẫn có giá trị giới hạn các giả thuyết; nó không tự cung cấp nguyên nhân thay thế.

## 7. Nhật ký web search

Mỗi hướng6 truy vấn. Search hits không đồng nghĩa đã đọc; chỉ primary links trong mục4 được dùng cho kết luận novelty. Semantic Scholar có trong queries; không tuyên bố truy cập đầy đủ index hoặc Google Scholar.

### A

1. `site:aclanthology.org subword regularization consistency machine translation segmentation Kudo`
2. `site:arxiv.org low resource neural translation morphological composition source unseen words`
3. `site:semanticscholar.org neural machine translation segmentation consistency word boundaries`
4. `"machine translation" "word boundaries" "consistency" subword`
5. `site:arxiv.org "segmentation" "consistency" "translation"`
6. `"neural machine translation" "compositional representations" morphology source`

### B

1. `site:aclanthology.org neural machine translation lexical binding source target counterfactual contrastive`
2. `site:arxiv.org neural machine translation word alignment causal interventions lexical substitution`
3. `site:semanticscholar.org neural machine translation context conditioned lexical translation alignment`
4. `"neural machine translation" "binding" "lexical" contrastive`
5. `"machine translation" "counterfactual" "unchanged" phrases context`
6. `"machine translation" "context-conditioned" "lexical" model`

Initial discovery: `neural machine translation source token lexical ambiguity context prototype translation low resource`; `machine translation source influence Jacobian token attribution regularization source target lexical errors`. Không trích nội dung search-only papers như đã đọc method.
