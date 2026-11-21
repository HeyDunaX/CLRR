# SAC-v1 — Source-Attribution Consistency

Ngày: 2026-11-15. Trạng thái: **đã chốt công thức nghiên cứu và prototype toán;
9 kiểm thử CPU PASS. Chưa có training wrapper, GPU gate hoặc kết quả pilot.**

### Evidence debate update — 2026-11-15

**Training admission: HOLD.** Hiện chỉ là giả thuyết kiểm thử được, chưa có
cơ sở để kỳ vọng chắc chắn SAC vượt W. Bảng pilot bên dưới là thiết kế có
điều kiện, không phải queue đã mở.

Đã đưa số W/PLC/confirmation và công thức coupled-mask cuối vào source riêng
của NotebookLM, rồi phản biện hai vòng trong conversation
`122e5d96-1146-495d-99b7-29abb9a24697`. W tăng mean validation BLEU khoảng1.4102
so baseline qua3seeds ủng hộ consistency nói chung, không chứng minh cần SAC.
Source margin W3.83621 gần baseline3.82411: chưa thấy deficit trung bình,
nhưng cũng không loại trừ bất ổn dưới dropout. PLC cancellation/tail-gradient
là quan sát hình học loss, chưa chứng minh nguyên nhân giảm generation.

Phản biện lại NotebookLM: bác các ngưỡng rho>.25, p<.01, C/B>.15 tự đề xuất;
không dùng token như quan sát độc lập; không gán output C cho riêng cross-attention.
Gradient conflict không tự động là lỗi vì W thành công cũng có conflict.
C=0 cho cả source-blind và source-discriminating ổn định là phản ví dụ thật.
Không có theorem đảm bảo thêm C sẽ giữ grounding hoặc tăng BLEU.

Nếu tiếp tục hướng này, đo coupled-mask đúng công thức trên checkpoint có sẵn
trước training, báo uncertainty theo câu và confounding. Không tự động GO chỉ
vì C khác0 hoặc tương quan với lỗi; frozen probe không chứng minh hiệu quả
can thiệp khi train. Probe mới đã hoàn tất; xem kết quả bên dưới. Receipt:
`outputs/experiments/source_attribution_design/EVIDENCE_DEBATE.json`.
NotebookLM hỗ trợ phản biện, không xác nhận độc lập novelty hay hiệu quả.

### Frozen probe completed — 2026-11-15

128 câu validation cách đều trong576 câu; baseline/W seed42 có sẵn, hai
derangements và hai dropout pairs giống nhau; FP32 SDPA TF32 off. Tất cả
CPU/native GPU replay gates PASS, archive SHA/provenance và CSV means verified.
Không optimizer, không test, không train lại. Transient restores đã dọn;
GPU trả về23952MiB free. Artifacts nhỏ tổng khoảng0.6MB, không tải weights local.

| Lexical diagnostic | Baseline | W |
|---|---:|---:|
| C |0.01173794|0.00877760|
| B |0.11311421|0.08041024|
| Mean C / mean B |0.10377072|0.10916025|
| Source information |0.40704737|0.40496691|
| Source gold gap |4.17587733|4.14305048|
| Absolute source-gap difference across masks |1.45389747|1.23111944|

W−baseline C=-0.00296034, approximate paired sentence-bootstrap95%CI
[-0.00364783,-0.00230133]. Absolute C thấp hơn khoảng25.2%; absolute source-gap
mask difference thấp hơn15.3%. Không dùng ratio C/B để kết luận phần tương tác
bị loại chọn lọc: ratio của W thực tế cao hơn nhẹ. Source-gap delta=-0.03282684,
CI[-0.12790443,+0.06313675]; source-information delta=-0.00208046,
CI[-0.00737780,+0.00315333]. Đây không phải equivalence test.

W C vs teacher-forced error Spearman0.3130; partial rank sau điều chỉnh entropy,
source/target character lengths0.03326, approximate sentence-bootstrap CI
[-0.17089,+0.24019]. Đây là association trên128 sentence averages, không phải
token-independent inference. Điều chỉnh có thể bỏ cả tín hiệu liên quan hoặc
còn confounding; không chứng minh C vô ích. C vs source-gap mask difference
partial rank0.30615 chỉ phù hợp với định nghĩa, không chứng minh gây lỗi dịch.

**Decision: giữ HOLD training SAC, giảm ưu tiên candidate.** W đã có C và
source-gap variability thấp hơn baseline trên probe; chưa có bằng chứng lợi ích
riêng từ giảm C thêm. Frozen endpoint không chứng minh SAC chắc chắn thất bại,
nhưng hiện không biện minh ưu tiên3 pilot four-forward. Không mở pilot tự động.
Validation đã dùng nhiều lần; một training seed; CI chỉ descriptive/conditional
on fixed pairings, chưa bao quát phụ thuộc do reuse distractors hoặc training seeds.

Results: `outputs/experiments/amis_source_attribution_frozen_rtx/ANALYSIS.md`,
`ANALYSIS.json`, `COLLECTION.json`, `PLAN.json`, per-model CSV/GPU gates.

Đã đưa số mới vào NotebookLM source `689ceb9b-a97b-4513-99bd-a66332ebed59`
và debate tiếp. Nó đồng ý HOLD nhưng vẫn có suy diễn quá mức; không nhận các
câu "proving artifacts", định vị cross-attention, preservation từ CI chứa0,
hoặc gọi chain rule là theorem mới. Đây là selected-best endpoints; không tự
gán tất cả là epoch20. Review đầy đủ và corrections lưu trong
`NOTEBOOKLM_MEASURED_REVIEW.json` cùng thư mục kết quả. Đây không phải replication
độc lập: NotebookLM đang phản biện chính dữ liệu nội bộ được cung cấp.

## 1. Quyết định phát triển

Phát triển một ứng viên: ổn định **khả năng phân biệt hai nguồn** dưới dropout.
Không mở thêm grid topology CLRR hoặc sửa hệ số PLC. SAC không phải đổi tên
CLRR/LSR; đây là hướng mới xuất phát từ kết quả consistency của dự án.

| Bằng chứng hiện có | Hệ quả thiết kế |
|---|---|
| Full CLRR+LSR không qua confirmation gates | Không bắt buộc giữ hai thành phần khi xây phương pháp mới |
| W/R-Drop beta1.25: mean validation BLEU18.2854 trên seeds42–44 | Giữ W làm đối chứng mạnh; không nhận R-Drop là đóng góp mới |
| PLC seed42 thấp hơn W0.5576 BLEU, chưa chứng minh tác hại qua seed | Không sweep PLC; giữ nguyên kết quả cũ |
| PLC cho phép bù trừ sai lệch giữa token, nhấn mạnh gold có xác suất thấp | Không dùng bình phương tổng log-likelihood hoặc chỉ gold-token discrepancy |
| Chưa chứng minh W mất khả năng dùng nguồn | Đây là giả thuyết cần kiểm chứng, không phải nguyên nhân đã xác lập |

Mục tiêu thực dụng: vượt W, không chỉ vượt baseline hoặc Middle yếu hơn trên
RTX. Mục tiêu khoa học: kiểm tra liệu ổn định source selectivity có giá trị
riêng ngoài việc tăng regularization đầu ra nói chung.

## 2. Ý tưởng dễ hiểu

Với cùng một tiền tố đích, mô hình dự đoán khác nhau khi đọc hai câu nguồn khác
nhau. Dropout có thể làm thay đổi cách phân biệt hai nguồn đó. SAC đo và phạt
thành phần thay đổi này, trong khi giữ CE và ordinary R-Drop của W.

Không ép hai câu nguồn có cùng bản dịch. Không gán nhãn đích của câu này cho
câu nguồn kia. Không giả sử câu nguồn còn lại là bản dịch sai về ngữ nghĩa.
Nó là một lựa chọn nguồn thay thế để định nghĩa phép đo.

"Attribution" ở đây chỉ posterior của **lựa chọn câu nguồn trong một cặp**.
Không phải alignment từ, attention explanation, hoặc causal attribution chuẩn
vàng. Độ ổn định và tính đúng của attribution là hai câu hỏi khác nhau.

## 3. Công thức chính xác

Cho mẫu (x0,y), h_t=y_<t. x1 là nguồn khác trong cùng TRAIN microbatch.
Lấy hai dropout realizations m=0,1 độc lập. Với mỗi m, **replay cùng randomness
trên cả x0 và x1**, cùng padded tensor shapes và cùng decoder prefix.

    p_ms,t(v) = P_theta,m(v | x_s, h_t),  m,s in {0,1}
    J_t(m,s,v) = p_ms,t(v)/4

Đây là một joint distribution nhân tạo, hữu hạn và tính chính xác được trên
toàn vocabulary; không phải ước lượng mutual information của dữ liệu tự nhiên.
J_t(M,S)=1/4, do đó M và S độc lập trước khi conditioning lên V.

    C_t = I_Jt(M;S | V)
        = sum_(m,s,v) J_t(m,s,v)
          log [ J_t(m,s,v) J_t(v) / (J_t(m,v) J_t(s,v)) ]

    A_t = I_Jt(M;V)
        = JS((p_00,t+p_01,t)/2, (p_10,t+p_11,t)/2)

    B_t = I_Jt(M;V | S)
        = [JS(p_00,t,p_10,t)+JS(p_01,t,p_11,t)]/2

    B_t = A_t + C_t,     0 <= C_t <= B_t <= log(2)

Proof: I(M;V,S)=I(M;S)+I(M;V|S)=I(M;V)+I(M;S|V), with I(M;S)=0.
Đây là chain rule có sẵn, **không phải theorem mới** của bài.

Tương đương:

    C_t = sum_v J_t(v) KL(J_t(M,S|v) || J_t(M|v) J_t(S|v))

Mỗi hạng theo token type v không âm trong số học chính xác. Không có bình
phương tổng sai lệch có dấu qua các vị trí như PLC. Tuy nhiên điều này không
bảo đảm từng gradient theo logit/parameter cùng chiều CE.

    L_SAC = L_W + lambda_C * mean_(labels!=-100)(C_t)
    L_W = mean_two_views(CE matched source)
          + 1.25 * half_symmetric_KL(p_00,t,p_10,t)

Giữ nguyên reduction của W, including supervised special tokens. Các mixture
weights đều có gradient; không detach posterior hoặc trọng số xác suất. Tính
log-softmax/logsumexp FP32; không clamp loss để che số âm do round-off. Bounded
loss không đồng nghĩa gradient Adam được kiểm soát hay chất lượng chắc chắn tăng.

## 4. Vì sao công thức có thể hữu ích, và vì sao vẫn có thể thất bại?

1. Hai views giống nhau nhưng hai nguồn được phân biệt rõ: C=0 được chấp nhận.
   Không cần xóa tín hiệu nguồn để làm C nhỏ.
2. Mô hình lờ nguồn hoàn toàn: **C=0 cũng được**. Vì vậy không được bỏ CE/W
   hoặc quảng cáo SAC bảo đảm grounding. Phải đo và loại source-blind collapse.
3. Dropout chỉ làm đổi output marginal còn source-choice posterior không đổi:
   C có thể bằng0 dù A>0. Đây là khác biệt có thể kiểm thử với ordinary JS.
4. C là conditional dependence, không phải độ lớn source contribution. Không
   được gọi A là "language-model noise" một cách mặc định: source marginal
   cũng chứa thông tin của nguồn. Không bảo đảm invariance với target-LM bias.
5. C lớn có thể là stochastic diversity có ích. Chỉ đo thấy C>0 không đủ kết
   luận phải giảm nó. Pilot phải thắng đối chứng chất lượng, không chỉ giảm C.
6. Tiền tố thật của x0 kết hợp x1 có thể ngoài phân phối joint huấn luyện.
   Dùng nguồn thật không loại bỏ rủi ro này. Control B dùng cùng các phép kết
   hợp để phân biệt lợi ích C với regularization trên context thay thế nói chung.

## 5. Literature audit trước pilot

Đã hỏi NotebookLM hai vòng và đọc nguồn primary bên dưới. Notebook có thêm
CCL nguồn `64103ceb-6788-4fe9-bc69-c3bdbda4a0dd`. Receipt lưu riêng; response
có liên kết trong prose nhưng `sources_used/citations` rỗng, nên không gọi đây
là source-use receipt đã xác minh hoặc giấy chứng nhận novelty.

| Prior đã đối chiếu | Điểm giao nhau | Khác biệt cụ thể của candidate |
|---|---|---|
| [R-Drop](https://proceedings.neurips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf), Sec2 | Two-view output consistency | Thêm bảng hai lựa chọn nguồn, phạt I(M;S|V) |
| [LR-Drop](https://aclanthology.org/2024.lrec-main.891.pdf), Sec3 | Hidden/attention/output consistency | Không thêm hidden matching; nguồn được crossed dưới mask replay |
| [DKD](https://arxiv.org/pdf/2203.08679), Sec3.1–3.2 | Phân rã divergence và chọn thành phần | DKD chia gold/rest trên label space; ở đây có biến source choice và dropout |
| [CBMI](https://aclanthology.org/2022.acl-long.169.pdf), Sec3 | Conditioning lên target history, thông tin nguồn | Không LM riêng, không log-ratio gold làm hệ số CE; exact finite-table conditional MI |
| [ContraDecode](https://arxiv.org/html/2309.07098v2), Sec2 | So correct/random source | Training-time dropout interaction; inference native, không contrastive decoding |
| [CCL](https://aclanthology.org/2024.emnlp-main.768.pdf), Sec3 | Cải thiện consistency | Không curriculum/difficulty weighting; thay đối tượng regularization |
| [RKD](https://arxiv.org/html/1904.05068v2), Sec3.2 | Quan hệ giữa inputs thay vì outputs riêng lẻ | Không feature distance/angle teacher-student; posterior source-choice hữu hạn |

**Verdict hiện tại:** chưa tìm thấy công thức SAC này trong nhóm prior đã đọc.
Đây là một candidate cụ thể có khác biệt thiết kế, chưa phải bằng chứng novelty
toàn lĩnh vực hoặc đủ sức ACL. CMI/nuisance regularization và relational
consistency là prior rộng; không nhận các primitive đó là mới. Đóng góp nếu
thành công phải gồm source-selectivity mechanism, đúng controls và gain lặp lại.

Hai lỗi trong phản biện NotebookLM đã được sửa độc lập:

- ||a-b||² không bằng ||a||²-||b||². Response consistency không tự động là
  "R-Drop trừ null-source R-Drop".
- Tổng quát B=A+C-I(M;S). Nonuniform nhưng independent priors vẫn có I(M;S)=0;
  chỉ đổi dropout rate không tự làm biến chỉ số M,S phụ thuộc.

Phiên bản independent mask ở từng cell bị loại khỏi thiết kế cuối. Algebra
vẫn đúng nhưng M không còn định danh cùng stochastic operator qua hai nguồn.

## 6. RQ và đóng góp có thể viết nếu thành công

- RQ1: Source-choice posterior thay đổi thế nào dưới dropout, tách khỏi
  output-marginal disagreement? Đây là đo bằng phép phân rã, không nhận diện
  lỗi dịch tự động.
- RQ2: Targeted source-attribution consistency có cải thiện generation hơn
  W và matched JS controls, trong khi giữ source sensitivity không?
- RQ3: Kết quả có lặp lại qua seed và Amis→Mandarin / Ê Đê→Vietnamese không?

Chỉ khi trả lời được RQ2/RQ3 mới đưa SAC thành main method. Nếu chỉ giảm C
mà không tăng chất lượng, dừng claim phương pháp này. Không xem proxy source
sensitivity là bằng chứng human faithfulness hay source-token alignment.

## 7. Pilot đề xuất: 3 runs mới, chưa khởi chạy

Amis→Mandarin, mBART-large-50, seed42, toàn bộ train gốc. Không dùng subset để
train lấy số. RTX là development; không trộn với A100 official results.

| Run | Objective ngoài W | Mục đích |
|---|---|---|
| S / SAC | lambda_C C | Candidate chính |
| J / matched-JS | lambda_J JS(p_00,p_10) | Control thêm bounded output consistency trên nguồn đúng |
| U / joint-JS | lambda_U B | Control cùng crossed contexts, regularize toàn bộ thay vì thành phần C |
| W, baseline, Middle | Reuse nếu protocol/provenance khớp | Không train lại để làm đối thủ yếu hơn |

S/J/U cùng bốn forward passes và mask coupling, chỉ khác auxiliary. J vẫn
thực hiện crossed forwards không tham gia loss để giữ luồng RNG và accounting.
Không tính optimizer steps hoặc epoch theo số forward passes.

Khóa lambda bằng một lần calibration không optimizer trên32 TRAIN rows lấy
theo seed42 permutation, không xem validation/test. Cho mỗi auxiliary, chọn
lambda để median norm gradient theo **cả bốn output-logit tensors** bằng10%
median CE norm trên cùng rows/masks; ghi cả parameter-block gradient diagnostics.
Nếu denominator ở mức round-off hoặc cần lambda ngoài[0.001,10], HOLD, không
tự clip hoặc thử grid. Đây là quy ước calibration, không phải optimum/theorem.
Không diễn giải matching output norms thành matching Adam updates.

Giữ pinned recipe, FP32/SDPA/TF32off, microbatch4/GA32, LR5e-5, ES10/max20,
selector validation BLEU beam1, tl_XX→zh_CN, reentrant checkpointing. Phải lấy
exact recipe/bundle hashes từ config W đã xác minh, không dựng recipe mới bằng
các giá trị rời ở đây. Native inference không thêm module/parameter.

Chỉ chọn nguồn khác qua cyclic derangement đã khóa trong microbatch; không
hard-negative mining, không thay dữ liệu. Nếu input IDs của hai nguồn trùng,
không giả vờ đó là semantic negative; log duplicate rate và giữ phép đo đúng.
Không dùng accumulation để lén tăng source pool. Microbatch1 không được hỗ trợ
bằng fallback im lặng. Attention mask của nguồn phải đi cùng source IDs.

## 8. Gates và quyết định tiếp theo

1. **Đã xong:** exact algebra, counterexamples, numeric gradients, finite extreme
   logits, invariances, CPU toy RNG replay và lambda0 gradient equivalence.
2. **Trước train:** training wrapper riêng, full-recipe tiny mBART GPU gate;
   verify actual mask coupling qua crossed sources under SDPA/checkpointing,
   accumulation equivalence, save/load/resume, lambda0 W replay. CPU toy không
   thay thế native GPU gate. Không tự đổi backend/batch để vượt gate.
3. **Frozen calibration:**32 TRAIN rows, bốn passes, C/A/B, source information,
   gold probability strata, output/parameter gradients, source margins; nếu
   metric chỉ là numerical residue hoặc source pairing sai thì không train.
4. **3 pilots S/J/U:** chỉ sau gates PASS, log diagnostics cố định tại init,
   25%/50%/100% planned updates và checkpoint được chọn. Log validation CE,
   BLEU/chrF++, loss components, grad norms/cosines, lengths/empty/repetition,
   source correct-vs-distractor margins trên cùng probe đã khóa.
5. **Promotion development:** S−W >=+0.30 validation BLEU; S hơn cả J và U;
   chrF++ không thấp hơn W quá0.10. Source-margin mean trên cùng fixed probe
   không giảm hơn5% so W; phải replay W bằng đúng probe/coupling trước khi so.
   Đây là practical pilot gates, không phải statistical significance claim.
6. **Nếu PASS:** replicate S và control mạnh nhất seeds43/44, reuse W tương ứng.
   Sau đó mới cân nhắc Ê Đê và A100. Không mở test ngay để lựa candidate.
   Nếu gain không lặp lại hoặc C giảm bằng source collapse, NO-GO cho SAC;
   không mở sweep lambda vô hạn.

Cost:4 forwards thay vì2 của W, dự kiến khoảng2x forward work; chưa đo native
wall time/VRAM. Không hứa thời gian hoặc bộ nhớ trước GPU gate. Shared source
encoder caching hoặc partial detach là thay đổi phương pháp, không tự tối ưu
ngầm để tiết kiệm. Checkpoints chỉ dọn sau HF backup verified như quy trình cũ.

## 9. Vị trí code và kiểm thử

    src/methods/source_attribution/v1/information.py  # mathematical core ONLY
    tests/test_source_attribution_information.py     # 9 CPU tests
    outputs/experiments/source_attribution_design/   # receipts, no weights

Không đăng ký vào method registry; chưa có launcher hoặc queue. Không thay
baseline, code phương pháp cũ, predictions hay số liệu paper.

    conda run -n clrr python tests/test_source_attribution_information.py

Research status phân biệt rõ: công thức/prototype toán PASS; full model/GPU
integration NOT RUN; chất lượng/novelty thực nghiệm NOT ESTABLISHED.
