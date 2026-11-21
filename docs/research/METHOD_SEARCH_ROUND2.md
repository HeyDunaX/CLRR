# Phát triển method — vòng 2: bằng chứng, sàng lọc và quyết định

Ngày 2026-11-18. **CPU analysis + đọc primary papers + hai vòng NotebookLM. Không mở training, GPU job, test evaluation hoặc queue.** Các điểm lịch sử giữ nguyên. Chỉ thêm script phân tích và artifacts của lượt audit này.

## 1. Bảng bằng chứng và phạm vi so sánh

### 1.1 Danh mục thực nghiệm

Snapshot [RUN_EVIDENCE.csv](D:/Code/CLRR/outputs/experiments/method_search_round2/RUN_EVIDENCE.csv) chứa **97 checkpoint entries, 28 cohorts, 55 method IDs**. Đây không phải 97 phương pháp độc lập hoặc census toàn bộ HF. Mỗi dòng có dataset, seed, selector, backend, archive SHA nếu có, HF prefix, recipe label, configuration source, đường dẫn validation/test receipt và metrics. Recipe label là nhãn đã lưu, không thay thế audit resolved arguments. AC43/44 và DRM cập nhật sau snapshot được dẫn riêng dưới đây.

Không xếp hạng bằng cách gộp A100/RTX, Amis/Ê Đê, eager/SDPA, selector BLEU/chrF++, beam1/beam4 hoặc các split khác nhau. Matching metadata không chứng minh cùng chi phí hay quỹ đạo số học. `comparison_scope` là cảnh báo cần đọc receipt, không tự kết luận run sai hoặc không công bằng.

### 1.2 Validation Amis: tính lại trực tiếp từ CSV

576 câu, references CSV gốc, SacreBLEU BLEU tokenizer `zh`, chrF0 word_order0 và chrF++ word_order2. Các CSV source/target khớp nguyên thứ tự; code không đổi dữ liệu. Các nhánh consistency thuộc recipe RTX FP32/SDPA, B4/GA32, LR5e-5, max20/ES10, selection validation BLEU beam1; xem các ngoại lệ provenance ngay sau bảng.

| Run / seed | Thiết kế | BLEU | chrF | chrF++ | File đã đọc |
|---|---|---:|---:|---:|---|
| Baseline42 | CE gốc |17.682329|16.954748|14.703005|`outputs/experiments/amis_residual_seed_diagnostics/inputs/R1/validation_predictions.csv`|
| W42 | CE hai views +1.25 symmetric KL, R-Drop đã biết |18.272410|17.388831|15.172807|`outputs/experiments/cmc_estimator_review/collected/W/validation_predictions.csv`|
| CV42 | CMC estimator có control variate |17.148787|16.589912|14.439779|`outputs/experiments/cmc_estimator_review/collected/CV/validation_predictions.csv`|
| CMC42 | Continuation consistency |15.540216|15.393158|13.316294|`outputs/experiments/cmc_estimator_review/collected/CMC/validation_predictions.csv`|
| PLC42 | Prefix consistency |17.714812|17.034527|14.845499|`outputs/experiments/amis_prefix_consistency_failure_analysis/artifacts/P_PREFIX_S42/validation_predictions.csv`|
| Middle42 | Middle alignment |16.677522|16.033324|13.937579|`outputs/experiments/amis_prefix_consistency_failure_analysis/artifacts/MIDDLE_S42/validation_predictions.csv`|

**[ĐO ĐƯỢC]** Recheck baseline `final_trainer_state.json` có20 evaluations: best BLEU và best chrF++ cùng **epoch19/step684**. SHA `9c4ba82f9cc383b89cbc922f3244f025f0cf720820cd43041fa81903a5b6b04d` khớp reuse audit. Vì vậy reuse chính checkpoint42 này có bằng chứng tương đương selector trên trajectory đã hoàn tất; không suy rộng sang mọi checkpoint chrF++ cũ. SDPA baseline là reconstruction từ runtime lịch sử, thiếu receipt backend đương thời.

Nguồn recipe/audit: `configs/experiments/amis_cmc_estimator_rtx.json`, `configs/experiments/amis_prefix_consistency_pilot_rtx.json`, `docs/research/PLC_FAILURE_ANALYSIS.md`. Các so W/CMC/CV đã có kiểm matching fields riêng trong `outputs/experiments/cmc_estimator_review/REPORT.md`. Lượt này replay scores, không audit lại toàn bộ training implementations.

### 1.3 Kết quả mới hơn snapshot: không chỉ nhìn AC42

Test dưới đây là **kết quả đã tồn tại**, không chạy test lại và không dùng để đặt threshold pilot mới. 575 câu gốc, beam4 FP32; không so ngang con số validation beam1 phía trên.

| Seed | Baseline BLEU | W BLEU | AC BLEU | AC−W |
|---|---:|---:|---:|---:|
|42|19.3926|21.1631|21.2938|+0.1307|
|43|19.5540|21.0765|20.9167|−0.1598|
|44|20.2453|21.2454|20.0803|−1.1652|
|Mean|19.7306|21.1617|20.7636|−0.3981|

**[ĐO ĐƯỢC]** AC vượt baseline2/3 seeds nhưng vượt W1/3. Không có kiểm định aggregate mới trong lượt này; không chuyển significance seed42 thành significance giữa seeds.

Nguồn: `docs/experiments/EXPERIMENTS_RTX6000.md`, mục AC test đủ3 seeds; `outputs/experiments/amis_ac_replication_test_dispatch/latest_status.json`; AC42 validation receipt `outputs/experiments/amis_two_method_round1_rtx/runs/AC_S42/VAL_COMPLETE.json`. AC42 validation17.613024, thấp hơn W42.

**[ĐO ĐƯỢC]** DRM42 validation18.472439 BLEU/15.155722 chrF++, so W18.272410/15.172807: +0.200030 BLEU nhưng −0.017085 chrF++, không qua gate cũ. DRM test21.098343, W21.163097. Nguồn validation: `outputs/experiments/drm_direction_probe_dispatch/prior_validation_receipts.json`; test: ledger RTX và `docs/research/DRM_NOVELTY_AUDIT.md`. Không nhận các số này là bằng chứng DRM có hại có ý nghĩa thống kê.

### 1.4 Các run không được dùng để kết luận chung

| Nhóm | Vì sao không pool / giới hạn |
|---|---|
|A100 paper/comparators với RTX development|Khác hardware/runtime/precision; cần cohort A100 riêng trước claim paper.|
|Historical chrF++ vs BLEU selector|Giữ riêng, trừ checkpoint có trajectory chứng minh chọn cùng best như baseline42 vừa replay.|
|Early-depth eager vs SDPA|Backend khác; apparent gain cũ không tự chứng minh gain cùng SDPA.|
|Ê Đê vs Amis|Khác ngôn ngữ, phân phối và BLEU tokenizer; không average raw scores làm effect.|
|Sequence-risk vs ordinary CE|Khác candidate pools và compute; phải có controls đúng thiết kế.|
|Relational v2 interrupted|Không phải completed negative result: `docs/research/DIFFERENCE_PRESERVING_PILOT.md` ghi novelty hold, chỉ A bắt đầu.|
|DRM diagnostic V1/V2|Controls không hợp lệ/incomplete; chỉ dùng V3, không gọi lỗi diagnostic là thất bại train method.|
|Frozen oracle intervention|Gold-informed, token-level, fixed checkpoint; không phải deployable translation gain.|

## 2. Pattern, mâu thuẫn và phần còn thiếu

| Nhận định | Nhãn và bằng chứng | Điều chưa được phép kết luận |
|---|---|---|
|Dense W hữu ích; sampled direct và CV chưa bắt kịp.|**ĐO ĐƯỢC:** bảng1.2, `cmc_estimator_review/REPORT.md`. W/CV/CMC đều hoàn tất20epochs.|Không đổ lỗi early stopping; không chứng minh exact dense CMC chắc thắng/thua.|
|Giảm noise estimator không đủ cứu chất lượng.|**ĐO ĐƯỢC:** CV giảm conditional empirical MSE81.3695%, vẫn dưới W; `amis_cmc_update_geometry_rtx/INTERPRETATION.md`.|Không quy toàn bộ thất bại cho variance, clipping hoặc Adam khi chưa tách causal effect.|
|CE thấp hơn không đồng nghĩa generation tốt hơn.|**ĐO ĐƯỢC:** W final val CE2.7636, CMC2.6348, CV2.7058; thứ hạng BLEU ngược. `cmc_estimator_review/REPORT.md`.|Không dùng CE nhỏ hoặc gradient norm đẹp làm quality gate thay BLEU.|
|Chưa chứng minh auxiliary thường chống CE.|**ĐO ĐƯỢC:** auxiliary/CE cosine W+.395056, CMC+.361411 trong phép frozen đo. Cùng geometry report.|Aggregate dương không loại token conflict hoặc conflict ở thời điểm khác; cũng không tạo động cơ cho projection ngay.|
|W vẫn có sai khác nội dung so reference.|**ĐO ĐƯỢC trong mẫu**, xem2.1.|Không suy ra nguyên nhân nằm ở encoder hay decoder; không gán semantic adequacy từ Chinese reference một mình.|
|Native source-read repair chưa có selective gold signal.|**ĐO ĐƯỢC:** query/simplex0/20 recovery; direct context30%2/20; paired adjacent-rank gold1/18 và alternative1/18. `w_read_recoverability_rtx/REPORT.md`.|Local bounded search không chứng minh source values vô ích hoặc mọi attention method bất khả.|
|Nhiều lỗi không nằm sát quyết định hiện tại.|**ĐO ĐƯỢC trong20 native-wrong tokens:** median gold rank65.5;17/20 rank>5. Cùng report.|Không lấy32 middle tokens để suy tỉ lệ toàn corpus hoặc dùng NLL giảm để tuyên bố sửa dịch.|
|DRM update có hiệu ứng rất nhỏ trong probe.|**ĐO ĐƯỢC:** V3 DRM trajectory−zero NLL −.000003316, CI chứa0; `docs/research/DRM_MECHANISM_CHECK.md`.|Không chứng minh forgetting là nguyên nhân; không tăng eta chỉ để khuếch đại số.|
|Full cũ không thua W ở mọi chỗ.|**ĐO ĐƯỢC:** ordinary-full P3 val18.458409 ở cohort chrF++ cũ; `cmc_estimator_review/PRIOR_DIRECTIONS.md`, registry.|Khác selector/backend provenance; không gọi là cùng-protocol gain hoặc bỏ vì không khớp câu chuyện.|

**[SUY LUẬN]** Pattern phù hợp nhất hiện tại: ổn định dự đoán là một yếu tố hữu ích, nhưng các proxy “alignment tốt hơn”, “CE thấp hơn”, “gradient ít nhiễu hơn”, “oracle NLL giảm” chưa xác định phần thay đổi nào sẽ sửa lỗi dịch. **[PHỎNG ĐOÁN]** Một nguyên nhân chung là học correspondence nội dung chưa đủ; đây vẫn là giả thuyết, chưa được định vị causal.

### 2.1 Phân loại24 câu validation lấy ngẫu nhiên, không chọn theo gain

Seed lấy mẫu20261118. Chỉ phân loại **W42 so một reference Chinese**, một người đọc, chưa chuyên gia Amis, chưa có inter-annotator agreement. Các loại là primary discrepancy, có thể còn lỗi phụ.

| Loại | n/24 | Tỷ lệ trong mẫu |
|---|---:|---:|
|Predicate/relation khác reference|13|54.2%|
|Object/lexical/state khác reference|5|20.8%|
|Reference-compatible|3|12.5%|
|Mở rộng so reference ngắn|1|4.2%|
|Thiếu specificity|1|4.2%|
|Source-copy/omission|1|4.2%|

Không gọi21/24 là tỷ lệ lỗi dịch toàn tập hoặc21 nguồn chắc chắn bị dịch sai. Ví dụ row46 có reference ngắn hơn nhiều prediction; cần kiểm source/reference, không tự gán hallucination hay sửa dữ liệu. Row286 “bị bệnh” thay “vô sinh” là thiếu specificity; row424 đảo sự kiện mặt trời biến mất/trời tối thành mặt trời xuất hiện/trời quang. Rows224/151/352 compatible theo reference.

Toàn bộ source, reference, sáu predictions và ghi chú từng câu: `outputs/experiments/method_search_round2/ANNOTATED_VALIDATION_SAMPLE.csv`. Raw sample giữ riêng; không ghi đè outputs model.

### 2.2 Kiểm khả thi natural edits trước khi nghĩ thuật toán phức tạp

**[ĐO ĐƯỢC]** Dùng Unicode word tokens, lower-case, bỏ punctuation chỉ để phân tích; tìm câu có cùng số từ và cùng toàn bộ khung trừ một từ. Mỗi normalized word/frame dùng một representative xác định, loại cặp có normalized reference trùng.

| Phép đếm | Kết quả |
|---|---:|
|Train rows|4600|
|Natural contrasting train pairs|238|
|Train rows covered theo lần mining ban đầu|228|
|Phép thay từ khác nhau trong238 pairs|238|
|Cùng phép thay từ xuất hiện ở≥2 train frames|0|
|Khung có≥3 word choices trước reference filter|14|
|Natural contrasting validation pairs|14|
|Phép thay từ validation từng thấy trong train mining|0|
|Hai outputs giống hệt sau normalization /14: baseline,W,CV,CMC,PLC,Middle|1,0,1,0,0,1|

Đếm này **không chứng minh semantic minimal pairs**.14 khung có3 choices không phải14 semantic composition cycles. Chuỗi prediction khác nhau không chứng minh model dịch đúng sự thay đổi.

**[SUY LUẬN]** Chưa có bằng chứng generation-invariance collapse của W trên14 cặp này. Candidate cần học/chứng minh một phép edit chuyển sang context khác sẽ thiếu repeated witnesses theo mining rule hiện tại. Không kết luận mọi loại edit trong ngôn ngữ đều không thể học.

Không mâu thuẫn con số165 cặp Amis trong audit cũ: audit cũ ràng buộc3..50 từ, normalization/filter khác. `docs/research/DIFFERENCE_PRESERVING_PILOT.md` còn ghi0 cặp đã semantic-verified. Historical baseline own-source margins dương165/165 là TRAIN, selector cũ, không phải W validation quality.

Artifacts: `CPU_EVIDENCE.json`, `CONTRAST_AUDIT.json`, `NATURAL_EDIT_EXAMPLES.csv`, `VALIDATION_NATURAL_CONTRASTS.csv`, hai input manifests trong thư mục round2.

## 3. Bảy hướng sàng lọc; hai hướng được audit sâu

Đây là bảy giả thuyết cơ chế, **không phải bảy phát minh mới**. Năm hướng bị loại sớm không được coi đã qua literature census riêng; hai hướng A/B được tra≥6 queries/hướng và debate hai vòng.

| Hướng | Hiện tượng → giả thuyết → thay đổi tối thiểu → dự đoán | Quyết định |
|---|---|---|
|1. Sampled/CV consistency tốt hơn|CV cải thiện variance → estimator gây lỗi → exact/dense continuation → phải vượt W generation, không chỉ MSE.|Không mở lại: variance riêng không đủ; CV đã fail; dense mới còn tốn compute và thiếu động cơ ưu tiên.|
|2. Source-response preservation|Nội dung còn sai → source response bị xóa → giữ correct/wrong-source logit difference → sửa source-conditioned decisions.|Không nhận: source gap W≈baseline, proxy có confidence confounds; đã audit trong `SOURCE_ATTRIBUTION_METHOD.md`.|
|3. Shared wrong-view negative learning|Hai views có thể cùng sai → consensus củng cố lỗi → unlikelihood shared nongold argmax → giảm shared errors.|Loại novelty của loss: exact unlikelihood; token khác reference có thể hợp lệ. Audit primary đã có `post_drm_design_review/DESIGN_REVIEW.md`.|
|4. Read/erase source memory|W sai nội dung → đọc nguồn/consumption sai → sửa native read hoặc residual memory → selective gold repair vượt arbitrary-label control.|Không nhận lại DRM: quality gate/probe chưa hỗ trợ; local oracle cũng chưa có selective signal.|
|5. Shared-prefix crossed-source ranking|Reference contrast → source discrimination yếu → assignment logistic ranking → cả hai nguồn chọn đúng nội dung.|Đã audit family natural-pair; ranking có thể pass khi cả hai vẫn chọn cùng token; không đổi tên làm novelty.|
|A. Token halfspace-projected consistency|W tốt hơn sampled, còn lỗi → **phỏng đoán** token aux gây hại CE → chiếu riêng phần xung đột → giảm harmful token updates mà giữ W benefit.|Audit sâu: **TRÙNG projection primitive; GẦN complete design. REJECT làm đóng góp chính.** Không có measured cause đủ mạnh.|
|B. Context-indexed edit-action transport|Reference-relative content errors → **phỏng đoán** không chuyển được thay đổi nguồn sang quyết định đích → học partial action thay vì kéo representations bằng nhau → transfer được edit qua context.|Audit sâu: **GẦN** edit-vector/CKD, không chứng minh trùng complete method. **NO-GO với thiết kế hiện tại:** không repeated witnesses, chưa validated deficit.|

### 3.1 Card A — tận dụng W, nhưng loại trước training

RQ: Consistency có đang làm hại gold decision ở một nhóm token dù aggregate gradient thuận CE không?

Hai dropout logits z1,z2, p=softmax(z). CE vẫn mean hai views. Với mỗi view/token, h=∇z CE và u=∇z symmetricKL. Đặt:

\[
\Pi_h(u)=u-\frac{\min(h^\top u,0)}{\|h\|^2}h,
\qquad d=h+1.25\Pi_h(u).
\]

Nếu h=0 thì dùng u. Một surrogate `sum(stopgrad(d)*z)` cấp prescribed gradient; không giả vờ đây luôn là gradient của một scalar potential toàn cục. PAD/special handling cần khóa trước implementation.

**[SUY LUẬN toán học]** h·Π≥0: contribution của auxiliary cho thay đổi CE là không dương ở first order khi descent theo−d. Không bảo đảm finite Adam step, shared-parameter updates, generation quality hoặc tất cả gold-vs-rival margins.

Nearest prior đã đọc: PCGrad2020 §2.3/Algorithm1 có đúng projection primitive; TGP2021 đã áp gradient projection trong multilingual MT; TPKD2026 §5.1 có per-margin safe cone mạnh hơn CE halfspace. Đổi vị trí projection sang dropout logits không tự thành đóng góp ACL. Cũng chưa đo token-level harmful updates có dự báo lỗi generation. **Không implement/smoke/train card này.**

### 3.2 Card B — sáng tạo hơn nhưng không vượt evidence gate

RQ: Model có học một thay đổi nội dung có thể chuyển ngữ cảnh, hay chỉ học translations riêng rẽ?

Natural pairs (x,y),(x′,y′) có edit e và target prefix h chung. qθ(x,h)=Pzθ(x,h), P=I−11ᵀ/|V| loại additive logit gauge. Ý tưởng là một partial action T_e có domain theo context c, không giả arbitrary edits commute/invertible:

\[
q_\theta(x',h)\approx T_e(q_\theta(x,h);c),\quad
L_{transport}=\mathbb E\|q_\theta(x',h)-T_e(q_\theta(x,h);c)\|^2.
\]

Đây là **constraint ý tưởng, chưa là complete admitted trainer**: nếu cả hai phía cùng học thì identity/collapse/memorization có thể pass; fixed-W target có thể truyền lại chính lỗi W. Chưa có căn cứ chọn parameterization/target anchor/weight. Thêm chúng tùy ý sẽ thành engineering không suy ra từ nguyên nhân đã đo.

Dự đoán cần có trước train: cùng một source edit ở hai contexts phải có target contrast xác minh được; can thiệp edit từ context A phải sửa đúng decision B hơn norm-matched random/wrong-edit controls và giữ unchanged facts. Cosine shift cao một mình không đủ. Cosine thấp chỉ bác mô hình offset chung đơn giản, **không bác mọi nonlinear/contextual action**.

Nearest priors: Guu et al.2018 đã học reusable sentence edit vectors; CKD2023 khớp softmax differences của teacher/student; CrossConST2023 ép invariance cho semantic-equivalent encodings; REDER2021 là reversible bilingual architecture, không phải source-edit action. Vì vậy chỉ chuyển từ invariance sang difference **không đủ**; contribution sẽ phải là cơ chế transfer và bằng chứng mới cụ thể.

Mining cho thấy0 edit lặp ở≥2 frames. Không có dữ liệu để chạy falsifier transfer theo rule hiện tại;14 validation pairs không có verified semantic correspondence. Không nới rule hoặc sửa dataset để cứu card. **Không implement/smoke/train card này.**

## 4. Paper gốc đã mở/đọc và NotebookLM debate

| Paper / năm / phần đọc | Cơ chế liên quan và kết luận có giới hạn |
|---|---|
|[Gradient Surgery for Multi-Task Learning,2020](https://arxiv.org/pdf/2001.06782), §2.3, Algorithm1|Chiếu gradient khi dot product âm. Projection primitive A đã có.|
|[Improving Multilingual Translation by Representation and Gradient Regularization,2021](https://aclanthology.org/2021.emnlp-main.578.pdf), Algorithm1, §2.3|TGP chiếu training gradient theo oracle gradient. Khác token dropout, nhưng gradient projection trong MT đã có. Không copy cách dùng dev làm oracle vào protocol hiện tại.|
|[Learning Beyond Full Imitation: Task-Preserving Knowledge Distillation,2026 preprint](https://arxiv.org/html/2609.39338v1), §5.1 Eq6–8|Chiếu conditional teacher signal vào safe cone bảo vệ từng margin. Không đồng nhất toàn algorithm với A.|
|[Generating Sentences by Editing Prototypes,2018](https://aclanthology.org/Q18-1031.pdf), §3.1–3.4|Prototype/editor và latent edit vectors từ inserted/deleted word embeddings; analogical reuse là prior, không novelty tự thân của B.|
|[Comparative Knowledge Distillation,2023 preprint](https://arxiv.org/html/2311.02253), §3.1–3.3 Eq1–4|KL giữa softmax student/teacher representation differences. Không tự đồng nhất mọi learned edit action với CKD.|
|[CrossConST,2023](https://aclanthology.org/2023.findings-acl.766.pdf), §2.2–2.4 Eq4–5|KL của source-conditioned và target-autoencoded decoder predictions; backprop cả hai sides. Semantic invariance, không edit equivariance.|
|[Duplex Sequence-to-Sequence Learning for Reversible MT,2021](https://proceedings.neurips.cc/paper/2021/file/afecc60f82be41c1b52f6705ec69e0f1-Paper.pdf), §2.1–2.2|Reversible duplex coupling và non-autoregressive symmetric architecture; không phải cùng B.|
|[Learning From Wrong Predictions in Low-Resource NMT,2024](https://aclanthology.org/2024.lrec-main.896.pdf), §3.1 Eq2–6|USKI: Unaligned Sentences Keytokens pre-training; CE ở overlap target tokens trên crossed pairs, optional IDF. Không phải tên USKI do NotebookLM đoán.|
|[Composable Text Controls in Latent Space with ODEs,2022/2023](https://arxiv.org/html/2208.00638), §3|Attribute energies và ODE trong latent generation; composability prior rộng, chưa xác lập cùng source-edit operator B.|

Notebook `bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`, conversation `8ec5ca51-f2be-4cca-a640-c5b4035a99d5`. Payload chỉ công thức/literature, không gửi số liệu hoặc bản dịch mới của project. **Toàn bộ câu hỏi/trả lời/citations** lưu trong `outputs/experiments/method_search_round2/NOTEBOOKLM_ROUND_A.json` và `NOTEBOOKLM_ROUND_B.json`.

- A: yêu cầu reviewer phản bác hai hướng; nearest priors, assumptions, lý do reject; không certify novelty/effectiveness.
- B: sửa dấu điều kiện gradient, phân biệt group không giao hoán với bất khả, phân biệt REDER với source edit; yêu cầu phản bác partial/context-indexed action và falsifier trước training.
- Phản bác giữ lại: A incremental, logit guarantee không chuyển thẳng sang Adam; B có prefix mismatch, sparse witnesses và lookup-table risk. Primary đọc trực tiếp mới là căn cứ prior, không dựa số citations NotebookLM trả.
- Không nhận các lỗi: A round đầu dùng sai dấu; noncommutativity không bác group; REDER không phải edit-action equivalent; CrossConST có một hướng KL trong Eq5 dù gradient qua hai sides; USKI không phải “Universal Sequence Knowledge Imitation”; CKD analogy không chứng minh exact equivalence; cosine≈0 không chứng minh mọi context-dependent action bất khả; first-order condition không phải bảo đảm finite-step CE.

## 5. Chốt hướng và việc tiếp theo

**Chưa chọn một method mới làm đóng góp ACL. Hai card đều chưa đủ để đưa sang smoke/train.** Đây là kết quả sàng lọc, không phải tuyên bố không thể phát triển tiếp từ CLRR/LSR. Không dùng quota hai methods để đổi tên một hướng đã loại.

W là strong known-method control có lợi ích rõ hơn các candidate hiện tại. Mục tiêu một method mới cần chứng minh thêm giá trị so W và comparator gần nhất, không chỉ thắng baseline. Novelty không được suy từ gain và gain không được suy từ loss/probe.

| Thứ tự | Việc cụ thể | Điều kiện quyết định | Training mới |
|---|---|---|---:|
|1|Dùng sample24 đã khóa để bilingual review source/reference/prediction; ghi nhãn uncertainty, kiểm những câu như row46.|Nếu source/reference đã có vấn đề hoặc paraphrase hợp lệ, tách khỏi model-error mechanism. Không sửa dataset tự động.|0|
|2|Chỉ chọn một error class có≥2 người xác nhận và đủ instances; đối chiếu train support và outputs W42/43/44 nếu có.|Cần một failure prediction riêng, không chỉ BLEU thấp. Không bịa threshold chất lượng từ mẫu24.|0|
|3|Thiết kế một frozen intervention tác động đúng error class, với wrong-edit/arbitrary-label/norm controls.|Chỉ xin/chạy GPU diagnostic sau khi có protocol và quyền phù hợp; cần selective correction vượt numerical noise, không chỉ oracle NLL giảm.|0|
|4|Suy ra cơ chế tối thiểu từ kết quả3, rồi primary search + reviewer debate như lượt này.|TRÙNG/không có empirical premise thì loại trước code; chưa có candidate thay thế được duyệt.|0|

Không đề xuất lặp cùng source-read oracle với budget lớn hơn hoặc tăng lambda để tìm số. Nếu không có bilingual support hoặc selective falsifier, ghi thiếu bằng chứng thay vì train một loss mới.

### Thiết kế ACL có điều kiện, chưa phải queue được duyệt

Sau khi có candidate vượt evidence/novelty/correctness gates: pilot Amis seed42 validation-only với baseline/W/nearest-prior controls đã matching provenance; reuse controls phù hợp, không train lại toàn bộ đối sánh. Giữ ES10 theo validation BLEU, max epochs/data/backend/batch như cohort đang so; log cả CE, aux, gradients và error-class predictions cần cho mechanism. Compute của auxiliary phải ghi, có control nếu đóng góp phụ thuộc extra passes.

Nếu pilot đạt tiêu chí khóa trước: ≥3 paired training seeds, báo mean/SD và từng seed; checkpoint selector không đổi sau xem test. Amis và Ê Đê hiện đã phục vụ development, fresh seeds không làm test cũ sạch lại. Cần external untouched evaluation hoặc ghi rõ exploratory và scope để xác nhận generality. Ê Đê chỉ chuyển format, giữ split/reference; language proxy khóa trước. A100 là cohort confirm riêng sau RTX, cùng recipe giữa các methods; không trộn số hai GPU.

Primary metric BLEU; chrF và chrF++ secondary, giữ nguyên signatures. Paired sentence bootstrap10,000 cho fixed-checkpoint corpus deltas, CI95%; không coi resampled sentences là training seeds. Báo separate seed-level paired uncertainty; với3 seeds exact two-sided sign-flip không thể đạt p<.05, không nhập nhằng với bootstrap. Khóa family các comparisons/metrics trước Holm, ghi số tests thực tế; không tái dùng p_holm của family cũ. Ablations phải tách đúng thành phần mới, matched compute, parameter/capacity và unchanged-fact preservation; không sweep test. Backbone thứ hai và domain/ngôn ngữ thứ hai để kiểm generality sau method lock.

Rủi ro lớn nhất hiện tại là thiết kế một objective tối ưu proxy mà không sửa lỗi nội dung đã xác minh — điều nhiều family trước đã cảnh báo. Rủi ro đánh giá là test đã được xem nhiều lần và một reference không xác nhận đủ nghĩa.

## 6. Nhật ký tìm kiếm và lệnh tái lập

### Queries web đã chạy trong lượt này

Khám phá:
1. `neural machine translation supervised consistency regularization preserve helpful stochastic gold probability directional R-drop`
2. `site:aclanthology.org dropout consistency machine translation confidence direction teacher gold probability`
3. `site:arxiv.org neural machine translation error memory retrieved corrections residual vocabulary source`

Hướng A:
1. `site:arxiv.org PCGrad project conflicting gradients tasks gradient surgery 2020`
2. `site:aclanthology.org consistency regularization gradient projection cross entropy machine translation`
3. `"R-Drop" "gradient" "projection"`
4. `"consistency" "gold" "gradient" neural machine translation regularization`
5. `site:semanticscholar.org consistency auxiliary gradient projection supervised loss PCGrad`
6. `"token" "gradient surgery" "consistency" translation`

Hướng B:
1. `site:arxiv.org compositional text editing transformation equivariance Lie group operators`
2. `site:aclanthology.org neural machine translation edit representations analogical sentence transformations`
3. `site:semanticscholar.org translation edit equivariance compositional transformation operators`
4. `"text" "edit" "equivariance" "composition"`
5. `"neural machine translation" "edit" "equivariant"`
6. `"sentence" "edit vectors" "composition" translation`

Follow-up:
- `site:aclanthology.org REDER reversible machine translation 2021`
- `site:aclanthology.org Universal Source Knowledge Integration USKI 2024 translation` (tên thử tìm chưa đúng; sửa bằng primary paper)
- `site:aclanthology.org edit vector sentence analogy neural machine translation equivariance`
- `site:aclanthology.org CrossConST 2023 cross lingual consistency regularization`
- `"USKI" "translation" Hu 2024`
- `"REDER" "2021" Zheng reversible duplex`
- `"neural machine translation" "edit" "equivariance"`
- `"translation" "logits" "analogy" source edit`

Search hits không đồng nghĩa papers đã đọc. Chỉ primary links và method sections trong mục4 được dùng cho verdict hiện tại. Google Scholar không được truy cập trong lượt này; Semantic Scholar nằm trong queries, không tuyên bố đọc toàn bộ index. Không claim “chưa ai làm”.

### Commands đã chạy

```powershell
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 scripts/analysis/synthesize_method_search.py
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 scripts/analysis/audit_natural_contrasts.py
```

Hai script CPU không import/load model, không có optimizer, GPU/SSH hoặc test CSV. Các lần đọc metadata bổ sung dùng PowerShell Get-Content/rg và Python pandas/json/hashlib trong clrr. `git -c safe.directory=D:/Code/CLRR status --short` chỉ xem trạng thái; không sửa global git config, commit hay push. SHA inputs trong hai manifests; fixed sample và annotations độc lập với historical outputs. Không upload artifacts mới lên HF trong lượt này.
