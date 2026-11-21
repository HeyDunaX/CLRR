# Hai phương pháp mỗi vòng — vòng 1

Ngày 2026-11-17. DESIGN ONLY. Không có trainer mới, queue, GPU job hoặc training được mở trong lượt này. Đã đọc bằng chứng local, tra paper gốc, phản biện hai vòng với NotebookLM và kiểm tra toán trên tensor CPU tổng hợp.

## 1. Bằng chứng và giới hạn

| Nhận định | Nhãn | Nguồn |
|---|---|---|
| Cùng cohort RTX seed42: baseline 17.682329 BLEU /14.703005 chrF++; W 18.272410 /15.172807; CMC 15.540216 /13.316294; sampled-direct 16.133234 /13.699138; CV 17.148787 /14.439779. Đây là validation, không phải test. | ĐO ĐƯỢC | outputs/experiments/cmc_estimator_review/REPORT.md |
| Các nhánh consistency hoàn tất 20 epochs; thất bại này không do early stopping ngắt sớm. | ĐO ĐƯỢC | Cùng REPORT.md |
| Sau GA32 thật, RMS gradient error sampledCMC/dense reference =1.79784; CV=.776001. CV giảm conditional empirical MSE81.3695% nhưng không cứu được chất lượng đã train. | ĐO ĐƯỢC | outputs/experiments/amis_cmc_update_geometry_rtx/INTERPRETATION.md |
| Cosine auxiliary với CE: W +.395056, CMC +.361411 trên checkpoint/batch đo. Không có bằng chứng aggregate CE conflict ở phép đo này. | ĐO ĐƯỢC | Cùng INTERPRETATION.md; DENSE.json |
| 16 ví dụ phân tầng có lỗi nội dung/aspect tương đối reference ở cả W/CV; không phải đánh giá chuyên gia Amis và không ước lượng tỉ lệ toàn tập. Không thấy output rỗng hay lặp nặng trong audit corpus. | ĐO ĐƯỢC | outputs/experiments/cmc_estimator_review/REFERENCE_ERROR_AUDIT.md; REPORT.md |
| Đổi hướng sampling hoặc chỉnh donor chung chung không phải hướng chưa chạy. | ĐO ĐƯỢC về lịch sử | outputs/experiments/cmc_estimator_review/PRIOR_DIRECTIONS.md |
| Source-attribution consistency hiện thiếu động cơ: W đã giảm instability C; source information/gold gap không chứng minh deficit mới. | ĐO ĐƯỢC + SUY LUẬN có giới hạn | docs/research/SOURCE_ATTRIBUTION_METHOD.md |

Không pool A100 với RTX, BLEU-selected với chrF++-selected, eager với SDPA hoặc ngôn ngữ khác. Baseline SDPA historical có phần reconstruct backend từ runtime, không phải contemporaneous attention receipt: xem configs/experiments/amis_cmc_estimator_rtx.json/baseline_reuse_audit.

W là R-Drop beta1.25 đã biết, không phải novelty của chúng ta. W tốt nhất trong nhóm so trực tiếp trên, không phải mọi thí nghiệm lịch sử.

## 2. Method 1 — Antisymmetric Consistency (AC)

**Vai trò:** khai thác kết quả tốt nhất, thay đúng cách truyền gradient consistency. Đây là ứng viên tối ưu engineering; chưa đủ cơ sở làm đóng góp chính ACL.

**RQ1:** Tác dụng của consistency đến từ giảm khác biệt hai dropout views, hay cần cả thành phần gradient cùng đẩy hai views?

**Hiện tượng → giả thuyết:** Dense W thắng các bản sampled hiện tại. Giữ nguyên full-distribution, mọi vị trí. PHỎNG ĐOÁN bổ sung: có thể bỏ thành phần common-view của auxiliary mà vẫn giữ lợi ích robustness; chưa đo rằng thành phần này có hại. Không gọi đây là giải quyết conflict CE đã được chứng minh.

Với mỗi vị trí supervised, hai logits z1,z2:
- m=(z1+z2)/2; d=(z1-z2)/2.
- z1_tilde=sg(m)+d; z2_tilde=sg(m)-d.
- L_CE=mean(CE(z1,y),CE(z2,y)).
- L_aux=SKL(softmax(z1_tilde),softmax(z2_tilde)).
- SKL(p,q)=.5[KL(p||q)+KL(q||p)].
- L_update=L_CE+1.25 L_aux; valid-token mean per microbatch, không lấy mẫu vị trí.

sg là stop-gradient. Forward auxiliary giống hệt W; đây là thay đổi update rule thông qua autodiff, không phải tuyên bố một scalar objective mới độc lập.

Nếu g1,g2 là ordinary SKL logit gradients:
- g1_AC=(g1-g2)/2.
- g2_AC=(g2-g1)/2.
- Auxiliary gradients cộng bằng0 theo trục view tại logits.

**Không được suy ra:** deterministic inference prediction bất biến, CE không đổi sau Adam, hoặc parameter gradients trực giao CE. Với shared parameters, J1 khác J2 nên gradient theta=(J1-J2)^T(g1-g2)/2, nhìn chung khác0.

**Kiểm tra CPU:** forward difference0; max gradient formula error2.78e-17; view-gradient sum0. Một ví dụ cosine với isotropic logit MSE=.827921 chứng minh không đồng nhất trong ví dụ đó; không phải chứng minh novelty hay hiệu quả. Xem FORMULA_CHECKS.json.

**Novelty:** GẦN. R-Drop là prior trực tiếp; Gradient Centralization là prior projection ở parameter-coordinate axis. AC dùng projection theo hai dropout views. Không gọi zero-sum projection là phát minh mới. Khi hai views rất gần, common component có thể nhỏ, làm khác biệt thực tế yếu.

**Falsifier trước train:** trên frozen W, TRAIN batch cố định, replay cùng dropout, đo common/difference auxiliary parameter gradients sau accumulation thật. Kiểm tra AC có khác norm-scaled W đáng kể không; báo numerical error và variation, không đặt ngưỡng cosine .95 như một theorem. Nếu thay đổi chỉ là scale trong sai số đo, hạ thành engineering ablation, không bán như cơ chế mới.

**Control bắt buộc nếu có gain:** W với beta cố định calibrated từ frozen TRAIN gradient norms, so parameter-gradient norm (không chỉ logit norm). Không adaptive norm-match mỗi batch. Cân nhắc logit-MSE prior nếu claim mạnh hơn; không tự mở sweep.

## 3. Method 2 — Decision-Residual Memory (DRM)

**Vai trò:** sáng tạo cơ chế mới hơn; bằng chứng nguyên nhân yếu hơn AC. Đây là exploration card theo yêu cầu, chưa được nhận là ACL-ready.

**RQ2:** Theo dõi phần nguồn đã được dùng bằng hướng nhạy của quyết định dịch có tốt hơn theo attention mass hoặc context direction không?

**Hiện tượng → giả thuyết:** W còn sai nội dung tương đối reference dù consistency hữu ích. PHỎNG ĐOÁN: encoder value có thể được đọc nhiều lần nhưng decoder không phân biệt nội dung nào đã hỗ trợ output. Audit hiện tại KHÔNG chứng minh cơ chế này gây lỗi, cũng không thấy lặp nặng để lấy repetition làm động cơ chính.

### Toán tử tối thiểu

Chỉ sửa cross-attention ở decoder layer cuối. Không thêm donor, CLRR hay LSR để tránh nhiều thay đổi cùng lúc.

Cho mỗi head và mỗi source position j:
1. R_1,j=V_j từ native encoder cross-attention projection.
2. Giữ native Q/K, softmax source attention A_t,j (trước attention dropout; source pads bằng0).
3. Với attention dropout, dùng đúng native read weights cho context; R update dùng A chưa dropout để coefficient nằm[0,1]. Khi inference hai loại trọng số trùng nhau.
4. c_t=sum_j A_read,t,j R_t,j. Ghép các head, qua native output projection/residual/norm/FFN/vocab head để có p_t.
5. Sau khi quyết định a_t đã có, lấy riêng partial derivative theo context ở bước hiện tại:
   g_t,h = partial log p_t(a_t) / partial c_t,h.
6. u_t,h=sg(g_t,h/max(norm(g_t,h),epsilon)); nếu norm nhỏ hơn epsilon, bỏ update của head đó.
7. R_(t+1),j,h = R_t,j,h - eta A_t,j,h [R_t,j,h dot u_t,h]_+ u_t,h.

eta=.1 là giá trị pilot đề xuất trước khi xem kết quả, chưa được chứng minh tối ưu. [x]_+=max(x,0). Không cập nhật sau EOS/PAD. u detached, không second derivatives; vẫn backprop qua R/attention/native parameters theo truncated-sensitivity rule đã định nghĩa. Không detach toàn bộ R nếu chưa đổi candidate card.

Train: a_t=y_t, nhưng chỉ cập nhật memory cho t+1 sau khi đã tính p_t. Nhãn ở quá khứ giống teacher forcing; không dùng y_(t+1). Inference: a_t là token thực sự được sinh, memory riêng từng beam, reset giữa câu; logprob của beam được ghi trước khi update cho token của beam đó. Training objective giữ hai-pass CE+1.25SKL của W trên logits DRM để cô lập thay đổi kiến trúc. Không ghép AC vào DRM trong vòng này.

### Điều được bảo đảm và điều không

Với unit u, b=eta A thuộc[0,1], s=R dot u>0:
norm(R')^2=norm(R)^2-(2b-b^2)s^2 <= norm(R)^2.
Thành phần trực giao u được giữ. Đây là tính chất projection elementary, không phải novelty theorem hay bảo đảm giữ nghĩa.

CPU toy: một fact cùng hướng u có support1 →.9 sau update, còn fact trực giao giữ1. Như vậy toán tử có thể làm yếu CHÍNH fact còn cần cho token tiếp theo. Không diễn giải PASS toán học là PASS cơ chế dịch.

Không được gọi local derivative là causal attribution hay semantic evidence chính xác. Nó phụ thuộc tọa độ latent, local approximation và quyết định có thể sai.

### Novelty và prior gần

**Kết luận GẦN; trong các method sections đã đọc chưa thấy đúng tổ hợp update rule này.** Không khẳng định chưa ai làm hoặc novelty chắc đủ ACL.

- Coverage Embedding Models, Mi et al., EMNLP2016 §3.1.2 Eq5 đã trừ target embedding chiếu theo attention khỏi coverage embeddings phụ. Vì vậy “vector coverage + subtraction” bị loại khỏi novelty.
- DRM thay actual cross-attention values theo hướng derivative của quyết định, không update coverage side-state rồi thay attention energies.
- NTM2014 đã có learned elementwise erase/add.
- Fast Weight Programmers2021 đã có key-addressed delta memory.
- EDA2026 đã có independently addressed rank-one erase; không nhận projection primitive là mới.
- Titans2025 dùng gradient associative-memory loss để fit memory; DRM dùng gradient output sensitivity để attenuate source values.
- He2019 dùng integrated gradients phân tích word importance, không phải cùng local derivative hay runtime value update.
- Past/Future NMT2018 đã mô hình hóa nội dung dịch/chưa dịch bằng recurrent states. Bài này loại claim tổng quát rằng explicit remaining-content memory là mới.
- CARVE2026 có memory-content-conditioned erase/write ở key/value axes; khác target-decision-conditioned source slots. Chỉ khác thiết kế chưa chứng minh đóng góp đủ lớn.

### Falsifier trước train

1. CPU causal/padding/batch/beam tests; eta0 phải recover native cả logits, gradients, generation/cache, checkpoint replay. Check gold y_t không ảnh hưởng p_t trước update.
2. Toy nhiều output dùng cùng một fact và toy hai fact trực giao. So DRM, context-direction attenuation, scalar decay cùng mức memory-change norm. Có counterexample repeated-fact đã nêu; phải đo tần suất/độ nặng trên dữ liệu thật.
3. Frozen W trên TRAIN subset cố định: derivative finite differences ở bước hiện tại; đo next-gold-token logprob/decision rank sau update trên các trường hợp source-to-multiple-target có annotation được kiểm tra. Nếu không có annotation, chỉ báo proxy; không gọi là verified semantic alignment.
4. So decision direction với context/random norm-matched directions. Nếu không có ưu thế chọn lọc, hoặc systematically phá downstream needed evidence, NO-GO cho DRM v1. Frozen pretrained model không hưởng adaptation nên thất bại probe không phải chứng minh mọi learned variant bất khả; nó bác động cơ dùng toán tử v1 như đã định nghĩa.
5. Benchmark native vs sequential eta0 vs DRM compute/VRAM trước khi mở pilot. Không đổi batch/backend/checkpointing để “cứu” rồi so ngang số cũ; nếu cần đổi recipe phải matched controls.

## 4. NotebookLM debate và phần bác lại

Conversation8ec5ca51-f2be-4cca-a640-c5b4035a99d5, notebookbf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890. Query/answer đầy đủ:
- outputs/experiments/two_method_design_round1/NOTEBOOKLM_ROUND_A.json
- outputs/experiments/two_method_design_round1/NOTEBOOKLM_ROUND_B.json

Vòng A: reviewer reject common-mean preservation dưới shared theta; coverage/erase đã có; sequential memory tốn thời gian. Sửa B từ context-only deflation sang decision-conditioned rule sau khi đọc Mi2016; bỏ mọi claim conservation/novel projection.

Vòng B: công nhận AC khác isotropic logit MSE và B khác Mi2016 về state/update direction; vẫn bác readiness vì motive chưa đo và risk one-to-many. Đây là hỗ trợ phản biện, không chứng nhận.

Không nhận nguyên văn những điểm sau từ NotebookLM:
- Nó không mở được nhiều external PDFs trong notebook; citations/sources_used trả về rỗng. Đọc paper gốc qua web mới là căn cứ đối chiếu.
- Không nhận các ngưỡng cosine>.95 hoặc delta logprob<-.5 nó tự gợi ý thành scientific gates.
- Cosine>.95 không chứng minh equivalence với scalar update; cần cả residual, norm và uncertainty.
- Không gọi eta<1 là full Gram–Schmidt erasure.
- Taylor expansion symKL trong answer bỏ hệ số phụ thuộc định nghĩa; không dùng công thức hệ số đó.
- Nó gọi tác giả R-Drop không nhất quán; không sao chép tên tác giả thiếu xác minh.
- TPKD được nó nêu chưa được mở/kiểm chứng ở lượt này, không tính vào verified closest-prior verdict.
- Không có sẵn gold one-to-many Amis alignments; không nhận gợi ý “32 câu có alignment” là dữ kiện.

## 5. Kế hoạch có điều kiện, chưa chạy

| Bước | Data/backbone | Công việc | Số training mới |
|---|---|---|---:|
|0 đã làm|CPU tổng hợp|Kiểm tra công thức, primary literature, hai vòng debate|0|
|1 cần làm|Amis TRAIN, frozen W/mBART|AC decomposition và DRM downstream-preservation/latency probes; không test, không optimizer|0|
|2 nếu qua gate|Amis→Mandarin /mBART /RTX /seed42|AC một model, DRM một model; độc lập, không AC+DRM|2 tối đa|
|3 chỉ cho candidate thắng|Cùng recipe/seed42|AC norm-calibrated W control; DRM context-direction và norm-matched random-direction controls|Tối đa3 controls; chỉ nhánh còn sống|
|4 candidate còn sống|Amis và Ê Đê, >=3 paired seeds|Replication development; cả hai data đã được xem, không gọi holdout sạch|Lập card riêng sau screen|
|5 đóng góp paper|A100, matched comparators, dữ liệu xác nhận chưa dùng chọn method|Protocol/recipe lock, fresh seeds, mechanism, test/statistics và report compute|Chưa duyệt/chưa launch|

Stage2 giữ recipe hiện hành: train4600/validation576 Amis; model revision e30b6cb8eb0d43a0b73cab73c7676b9863223a30; src tl_XX/tgt zh_CN; FP32 SDPA; LR5e-5; microbatch4×GA32; max lengths256; max20epochs; ES10 theo original-reference validation BLEU/beam1, earliest tie; AdamW fused, clip1, warmup.06, linear scheduler, original-reference zh BLEU. Nguồn: configs/experiments/amis_cmc_estimator_rtx.json và configs/protocols/amis_drb_bleu.json. Không thay protocol file trong lượt này.

Reuse W/baseline hiện có sau audit bundle/metadata; Middle/LoRA cũ chỉ reuse nếu matched backbone/data/runtime/selector/compute được xác nhận. RTX là development, không lấy trực tiếp làm A100 confirmation.

Practical exploratory screen đề xuất khóa trước: candidate−W>=+.30 validation BLEU, chrF++ không thấp hơn W, không regression rõ về nội dung; đây là engineering decision margin, không alpha/p-value hay power guarantee. Kết quả sát ngưỡng không tự đổi weight/selector hoặc xem test để cứu.

ACL: chọn method bằng development trước; ít nhất3 seed (nên tính power trước khi chốt n), paired corpus bootstrap10000 cho BLEU/chrF++ từng seed, Holm theo family comparisons khóa trước, thêm phân tích giữa paired seeds; không thay seed-level uncertainty bằng bootstrap câu. Report tất cả seeds, failures và compute/inference overhead. Cần so baseline/Middle/strongLoRA/W và coverage/memory closest prior thích hợp cho DRM. Các đối sánh đã có chỉ tái dùng nếu protocol match.

## 6. Nguồn primary đã mở và đọc abstract/method

1. R-Drop, NeurIPS2021 §2 Eq2–4: https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf
2. Gradient Centralization, ECCV2020 §3.3: https://arxiv.org/html/2004.01461
3. Curriculum Consistency Learning, EMNLP2024 §3: https://aclanthology.org/2024.emnlp-main.768.pdf
4. Coverage Embedding Models for NMT, EMNLP2016 §3.1.2: https://aclanthology.org/D16-1096.pdf
5. Neural Turing Machines2014 §3.2: https://arxiv.org/html/1410.5401
6. Linear Transformers Are Secretly Fast Weight Programmers2021 §4.2: https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf
7. Erase-then-Delta Attention2026 §3.3: https://arxiv.org/html/2606.26560
8. Titans2025 §3.1: https://arxiv.org/html/2501.00663v1
9. Towards Understanding NMT with Word Importance2019 §3.3: https://aclanthology.org/D19-1088.pdf
10. Modeling Past and Future for NMT, TACL2018 §4.1: https://arxiv.org/html/1711.09502v2 (Anthology Q18-1011 PDF timeout; dùng bản arXiv method)
11. CARVE2026 §4.2: https://arxiv.org/html/2606.27229

Cùng cơ chế không đồng nghĩa trùng toàn method; khác vài chi tiết không tự đủ novelty. Phạm vi tìm trên arXiv/ACL Anthology và search Semantic Scholar, không phải systematic review toàn literature. Một số truy vấn không có kết quả hữu ích; không tính search hit chưa mở là paper đã review.

## 7. Nhật ký truy vấn web

Exploratory:
- site.aclanthology.org neural machine translation dropout consistency wrong predictions source contrastive regularization
- site.arxiv.org neural machine translation consistency regularization quotient distribution equivalence class token

AC:
- "R-Drop" "gradient" "mean"
- "consistency" "barycenter" "stop gradient" dropout logits
- "consistency regularization" "mean logits" gradient
- "R-Drop" "logit" "MSE"
- site.arxiv.org "dropout" "antisymmetric" gradient
- site.aclanthology.org "R-Drop" "logits" consistency
- site.semanticscholar.org "consistency" "gradient centralization"
- site.arxiv.org "mean-preserving" "consistency"
- "Gradient Centralization" Yong 2020 paper (arxiv filter)

DRM:
- "machine translation" "coverage" "residual" source memory
- "transformer" "evidence" "ledger" attention
- "translation" "coverage" "embedding" subtract
- "attention" "value" "depletion" translation
- site.aclanthology.org translation coverage gradient attribution memory
- site.arxiv.org "cross-attention" "erase" "gradient"
- site.semanticscholar.org translation "coverage" "orthogonal"
- site.arxiv.org "value memory" "projection" attention
- neural machine translation relevance coverage gradient source attribution decoding (ACL filter)
- "gradient" "erase" "memory" attention source token (arxiv filter)
- "translation" "coverage" "gradient-based" decoding
- "attention" "memory" "gradient" "erasure" token
- "cross-attention" "values" "decision" "projection" translation
- "neural machine translation" "relevance" "coverage" memory
- "Modeling Past and Future for Neural Machine Translation" pdf
- "translation" "memory" "gradient-based" coverage erasing (ACL/arxiv filter)
- "attention" "decision gradient" memory (arxiv/SemanticScholar filter)
- "coverage" "attribution" "decoding" neural machine translation (ACL/arxiv filter)
- "Titans" "Learning to Memorize at Test Time" gradient memory arxiv
- "Modeling Past and Future for Neural Machine Translation" arxiv

## 8. Chốt vòng này

Hai candidate cards đã rõ cơ chế và có paper/debate, không phải hai method đã được xác nhận tốt. AC là hướng engineering có chỗ tựa thực nghiệm; DRM có design difference lớn hơn nhưng causal motivation chưa đo và chi phí cao. Không có bằng chứng đủ để gọi một trong hai là đóng góp chính ACL ngay bây giờ. Bước hợp lý tiếp theo là đúng frozen discriminating probes đã nêu, không mở long training chỉ vì tên mới.



## 9. Implementation and pre-training admission (author approved 2026-11-17)

Two independent Amis/mBART seed42 pilots are authorized after frozen TRAIN probes and technical GPU gates. Original methods and protocol lock are unchanged. Implementation folders: `src/methods/antisymmetric_consistency/v1` and `src/methods/decision_residual_memory/v1`. Shared runner config: `configs/experiments/amis_two_method_round1_rtx.json`.

DRM v1 reconstructs each beam's memory from its actual emitted prefix; **native KV caching is explicitly disabled**. This is causal and stateless between sentences/beams, but more expensive. SDPA is retained for attention reads; erase weights use explicit softmax. These algorithmic costs are reported, not claimed identical compute to W. AC has native inference. Both start from the same original mBART revision as W, NOT from the trained W checkpoint. Frozen diagnostics alone use pinned W.

Admission rules locked before measuring:
- Probe uses four deterministic TRAIN rows (seed1729), no optimizer/no test; small-sample screen, not a population or causal conclusion. Full GA32 is exercised separately in the GPU gate.
- AC: finite nonzero full-parameter auxiliary gradients, relative residual after the best scalar multiple of W >=.01. This rejects an effectively redundant scalar-weight change on the probe; it does not prove a beneficial direction.
- DRM: finite nonzero memory update; mean gold-token NLL degradation <=.05 nat and <=25% valid tokens degraded by >.1 nat; forward latency <=10x native on the same batch; conservative projected validation prefix-replay scheduling cost <=1hour/epoch. These are exploratory preservation/resource guardrails, not calibrated significance thresholds. The projection uses TRAIN target lengths and a 16-token greedy benchmark; its uncertainty must be reported. No claim of semantic alignment from gold NLL.
- Any probe timeout (20minutes) or failed GPU correctness/resource gate (60minutes) skips that candidate, records the error and leaves the other eligible candidate available. Gate failures are not efficacy results. All attempted candidates finish probes/gates before any long training.
- Technical gate: same B4/GA32 FP32 SDPA, two optimizer steps, checkpoint reload, optimizer/RNG resume, beam4 replay checks. DRM's cache-request equality means both requests replay prefixes; it does NOT certify native caching.
- Train at most the two admitted methods:20epochs/ES10/validation BLEU beam1. No automatic test, control, replication or A100 runs. Candidate-minus-W >=.30 validation BLEU and nonnegative chrF++ remain exploratory follow-up criteria.
- Each completed checkpoint uploads publicly to the canonical method/seed42/config prefix, read-back SHA verification precedes removal of its local weights. Failed uploads preserve local weights.
