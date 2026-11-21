# Thiết kế nâng cấp phương pháp — novelty trước training

## Trạng thái hiện hành 26/10/2026

Bước1–2 đã xác minh bằng cached predictions của đúng initialization R1, cùng evaluation protocol. Tất cả S0–S3 thấp hơn R1, bao gồm baseline+CE continuation; chưa quy được nguyên nhân riêng cho CLRR/risk. CI theo câu của một seed không xác nhận harm ổn định. Không train hoặc dùng GPU mới.

Bước3: learned-strength zero-init chỉ là adapted diagnostic control, chưa đủ novelty làm phương pháp chính; training HOLD. Xem [bản kiểm chứng và quyết định thiết kế](CONTINUATION_REVIEW_AND_NEXT_DESIGN.md) cho số liệu, công thức, prior, controls và gates. Các banner/recipe bên dưới là lịch sử, không phải lệnh chạy mới.

## K?t qu? x?c minh 20/10/2026 ? sequence-risk pilot: NO-GO

?? ho?n t?t 4/4 runs tr?n RTX6000, seed42, c?ng baseline R1, 3 continuation epochs; validation 576 c?u, beam1. Ch?a ??nh gi? test. ??y l? continuation controls, kh?ng thay s? baseline g?c.

| Run | C?u h?nh | BLEU | chrF++ | chrF0 |
|---|---|---:|---:|---:|
| S0 | Baseline + CE | 17.2803 | 14.4668 | 16.6790 |
| S1 | Baseline + CE + risk | 17.5593 | 14.4865 | 16.7025 |
| S2 | CLRR + CE | 17.4191 | 14.3061 | 16.4921 |
| S3 | CLRR + CE + risk | 17.2666 | 14.2728 | 16.4245 |

**Kh?ng ??t c?c development gates:** S3?S1 BLEU ?0.2927 (chrF0 ?0.2780), S3?S0 BLEU ?0.0137, interaction BLEU ?0.4315. Baseline+risk t?ng +0.2790 BLEU so v?i continuation CE control, nh?ng chrF0 ch? +0.0235; ch?a c? b?ng ch?ng ? ngh?a th?ng k? ho?c ?? ?n ??nh nhi?u seed. K?t qu? kh?ng ?ng h? gh?p CLRR hi?n t?i v?i risk theo recipe n?y; kh?ng ch?ng minh m?i bi?n th? CLRR ??u v? ?ch.

C? 4 checkpoint ?? upload v? x?c minh hash tr?n `FiveC/amis-rewire-checkpoints`; receipts n?m trong `runs/S*/VAL_COMPLETE.json`. Queue k?t th?c `RTX_PIPELINE_COMPLETE`; kh?ng m? th?m training/test. Ngu?n s?: `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/PILOT_SUMMARY.json`. C?c tr?ng th?i running/hold b?n d??i l? l?ch s?.


**Trạng thái: DESIGN ONLY / NOVELTY HOLD.** Chưa sửa phương pháp, chưa mở training. Hướng KL chọn lọc bên dưới đã bị loại khỏi vị trí đóng góp mới chính sau khi đọc prior gần. Bảng pilot là thiết kế đối chứng có điều kiện, không phải lịch chạy đã duyệt.

**Cập nhật hiện hành 18/10:** người dùng đã cho phép tiếp tục chuẩn bị để
vào train sau khi pool T1 bị reject. Probe T1.3 train-only đã PASS; recipe
mới và thứ tự gates ở §9. 25 numerical/pipeline/pool CPU tests PASS. MRT là
đối chứng đã biết; các trạng thái HOLD ở §1–7 ghi lại quyết định trước lần
ủy quyền này, không phải trạng thái triển khai hiện hành. Không claim novelty.

## 1. Bằng chứng dùng để thiết kế

Nguồn hiện hành: [Token pilot audit](TOKEN_PILOT_AUDIT.md), [insights](RESEARCH_INSIGHTS.md#verified-confirmation-insight). Các số dưới đây giữ nguyên protocol đã chạy.

| RTX6000, seed42 | Validation BLEU | Ý nghĩa giới hạn |
|---|---:|---|
| Baseline | 17.6823 | Mốc đối chiếu cùng recipe FP32 |
| Original Full | 17.1156 | CLRR + sentence LSR chưa vượt baseline |
| Token-LSR | 17.5811 | Đổi sang alignment token chưa tăng BLEU |
| Token-Full | 16.7533 | Ghép CLRR + Token-LSR chưa đạt validation gate |

Tất cả bốn run đạt epoch20; patience10. Audit không phát hiện lỗi decomposition CE + auxiliary loss hoặc gradient không hữu hạn trong các phép kiểm đã thực hiện. Pooled covariance rank không cho thấy catastrophic collapse. Can thiệp tắt các nhóm lớp không cho hiệu ứng nhất quán và tất cả CI chứa0.

Fresh confirmation Amis, seeds45–52: Original Full−LSR mean BLEU −0.019809 (3/8 dương); Full−Baseline −0.041468 (4/8 dương). Kết quả này không xác nhận lợi ích ổn định của cấu hình Full hiện tại, không chứng minh mọi LSR/CLRR đều vô ích.

## 2. Vì sao cân nhắc thay LSR? Vì sao chưa được kết luận phải bỏ?

LSR hiện hành căn chỉnh hidden representations: sentence LSR dùng pooled cosine; Token-LSR dùng detached balanced OT. Cả hai là surrogate objectives. Hidden vectors giống nhau hơn không trực tiếp bảo đảm decoder gán xác suất cao hơn cho bản dịch đúng.

Trong probe cùng16 validation examples, sentence loss của Original Full thấp hơn baseline (0.2473 so với0.6614); Token-LSR giảm token loss (0.6342 so với0.8325). Tuy vậy, BLEU trên toàn576 validation examples không tăng. Hai phạm vi đo khác nhau: đây là dấu hiệu cần kiểm tra sự liên hệ giữa objective và chất lượng, chưa chứng minh causal failure của LSR.

Các tỷ số instantaneous auxiliary/CE gradient nhỏ không đủ để quy lỗi cho lambda quá nhỏ; chúng không đo tích lũy cập nhật Adam. Audit cũng không hỗ trợ giải thích đơn giản bằng collapse hoặc early stopping.

**Quyết định:** có cơ sở để thử một objective khác với đối chứng phù hợp; chưa có cơ sở xóa LSR khỏi phương pháp hoặc lịch sử thực nghiệm. LSR-only và Original Full tiếp tục là controls. Nếu thay objective, dùng method ID mới; không gọi nó là LSR cũ được cải thiện.

## 3. Ứng viên đã thiết kế, rồi bị giữ lại bởi novelty gate

### Cơ chế

Giữ encoder CLRR gốc (distance2, strength0.1, detached post-rewiring donor). Cùng một bộ weights tạo hai dự đoán teacher-forced:

- `q`: rewiring OFF, dropout OFF, no_grad; teacher dùng weights hiện tại, không phải baseline độc lập đã train.
- `p`: rewiring ON, dropout như baseline; CE và auxiliary gradient cập nhật student.

Với token nội dung `y_t`, chọn `w_t = 1[argmax(q_t) = y_t AND q_t(y_t) > p_t(y_t)]`; detach teacher và gate. Loss đề xuất:

\[
L=L_{CE}(p,y)+\beta\frac{1}{B}\sum_b
\frac{\sum_{t\in C_b}w_{bt}\,KL(\operatorname{sg}(q_{bt})\|p_{bt})}
{\max(1,\sum_{t\in C_b}w_{bt})}.
\]

`C_b` loại PAD, ignore labels và special tokens; CE giữ mask baseline. Temperature1; beta0.1 chỉ là giá trị pilot dự kiến, chưa phải tối ưu hoặc giá trị được suy ra từ audit. Batch không có token được chọn cho auxiliary0. Inference chỉ dùng nhánh CLRR ON, không dùng labels/teacher/gate.

Nhánh teacher phải chạy trước student và khôi phục đúng training flags, routing state, cache và RNG. Kiểm tra checkpointing recomputation/backward là bắt buộc. Zero added parameters không đồng nghĩa miễn phí: thêm một encoder–decoder forward và logits toàn vocabulary khi train.

### Novelty audit và quyết định

| Prior đã đọc | Phần đã có | Ảnh hưởng tới đề xuất |
|---|---|---|
| [R-Drop, NeurIPS2021](https://arxiv.org/html/2106.14448) | Hai dropout views, output KL, shared weights | Output consistency/KL không phải ý tưởng mới |
| [SelFusion, ACL2026, §3.2–3.3](https://arxiv.org/html/2608.22898) | Hai masking modes; chọn hướng distillation theo token correctness và ground-truth probability | Quy tắc teacher đáng tin ở từng token có prior rất sát |
| [Multi-Teacher Distillation with Single Model for NMT, TASLP2022](https://lijuntaopku.github.io/TASLP_camera_ready.pdf) | Tạo teacher variants bằng reordering, LayerDrop và dropout | Architecture variants để tạo supervision đã có trong NMT |
| [Adaptive Depth Networks with Skippable Sub-Paths, NeurIPS2024, §3.2](https://proceedings.neurips.cc/paper_files/paper/2024/file/3a2d96d2eb2902043c2db705ca03e9a2-Paper-Conference.pdf) | Distillation giữa shared-weight super/base networks | Bật/tắt đường kiến trúc cùng distillation không đủ để tự nhận là mới |
| [A Self-Distillation Recipe for NMT, FindingsACL2025](https://aclanthology.org/2025.findings-acl.261.pdf) | Deep-to-shallow/iterative representation distillation, gradient adaptation | Cần phân biệt rõ với self-distillation và gradient gating đã có |
| [Confidence-Aware Multi-Teacher KD](https://arxiv.org/pdf/2201.00007) | Ground-truth-based confidence weighting | Confidence selection tự nó không phải novelty |

SelFusion dùng diffusion masking và bidirectional teaching; đề xuất dùng CLRR on/off và one-way teaching. Hai công thức không đồng nhất, nhưng **khác view/task/direction chưa đủ để khẳng định đóng góp thuật toán mạnh**. Đây là đánh giá nghiên cứu, không phải kết luận hai phương pháp giống hệt nhau.

**NO-GO chọn selective KL này làm phương pháp mới chính ở hiện trạng.** Có thể dùng như adapted control nếu một thiết kế mới cần nó. Không mở pilot chỉ để xem gain rồi dùng gain thay cho novelty.

## 4. Điều kiện cho thiết kế tiếp theo

1. Xác định vấn đề dịch cụ thể từ diagnostics, phân biệt evidence với giả thuyết. Không mặc định collapse, over-smoothing hoặc morphological loss là nguyên nhân.
2. Viết rõ phép toán mới tác động vào donor, routing hay supervision; mỗi candidate thay một yếu tố. Không mặc định CLRR/LSR phải được giữ nếu bằng chứng không hỗ trợ.
3. Đối chiếu công thức và backward với prior gần nhất, gồm references/citing papers và code chính thức khi có. Ghi exact section, source URL, điểm kế thừa và điểm khác.
4. Nếu chỉ thay layer/distance/lambda, thêm KL, hoặc dùng selection/gating đã có: xếp là configuration/adapted control, không tự nâng thành contribution mới.
5. Novelty GO cần một đóng góp có nội dung rõ, khác về cơ chế hoặc formulation, cùng đối chứng có thể bác bỏ đóng góp đó. Không thể chứng nhận tuyệt đối không có prior từ một vòng tìm kiếm.
6. Sau novelty GO mới khóa recipe, correctness gate và pilot; kết quả âm được lưu để giới hạn claim. Không sweep vô hạn.

## 5. Bảng pilot có điều kiện nếu một candidate tương tự vượt novelty gate

| Run | Nhánh | Câu hỏi |
|---|---|---|
| Reuse | Baseline RTX R1 | Mốc dịch cùng hardware/runtime |
| P1 | CLRR-only CE | Routing có đóng góp khi đứng riêng không? |
| P2 | CLRR + ungated KL | Generic consistency có đủ giải thích gain không? |
| P3 | CLRR + selective KL | Selection có đóng góp vượt P2 không? |
| P4 | No-CLRR + selective eval/train KL | Gain có cần CLRR không, hay chỉ self-distillation/dropout? |
| P5 | Adapted R-Drop | Có hơn prior output-consistency quan trọng không? |

**Đây là5 runs mới seed42 nếu sau này được chọn; hiện0 runs được mở.** P4 teacher dropoutOFF/student dropoutON nên objective không triệt tiêu như hai pass deterministic giống nhau. P5 giữ objective R-Drop với hai stochastic views và gradients cả hai, kiểm tra normalization theo code gốc; adaptation/tuning budget phải được khai báo. Pilot một coefficient không chứng minh thắng comparator đã tune.

Nếu P3 thắng, bổ sung matched selected-token CE weighting để loại giải thích chỉ là hard-example weighting trước khi viết claim cơ chế. R2–R4 cũ dùng để tham khảo; không tự thay checkpoint hoặc protocol của chúng.

### Fairness và metric selection đề xuất cho pilot mới

- RTX6000 FP32, Amis4600/576/575, pretrained revision/data hashes giống R1; LR5e-5, batch4×GA32, cap20, ES10, warmup0.06, optimizer/scheduler/clipping giống cohort đã audit.
- **Đề xuất chuyển selection và early stopping sang validation BLEU tokenizer zh, beam1 FP32**, khóa rõ trước run. chrF0 là secondary; native chrF++ tiếp tục báo nhưng không làm bằng chứng chính vì word bigrams tiếng Trung quá sparse. Đây là thay protocol công khai, chưa sửa config hiện hành.
- R1 đã đạt cap20 và best BLEU/best chrF++ cùng epoch19, step684 theo CPU_AUDIT.json. Muốn reuse dưới BLEU phải replay lịch patience theo BLEU; nếu trajectory sẽ dừng khác, cần matched baseline mới. Không dùng baseline A100/BF16 làm matched control RTX/FP32.
- Threshold screening dự kiến: candidate ≥ baseline+0.30 BLEU, ≥ ungated/no-CLRR controls+0.20, chrF0 không giảm; kiểm tra R-Drop trước GO. Các ngưỡng là practical screening choices, không phải power/statistical confirmation.
- Selection dùng validation; không dùng test575 để chỉnh loss hoặc chọn cấu hình. Seed42 chỉ development. Qua pilot mới thiết kế fresh-seed confirmation và chuyển cùng controls sang A100; chưa gán số run confirmation khi chưa có candidate.
- Trước GPU đo VRAM/processes, không đụng job người khác. Toàn bộ env/cache/output nằm trong `/workspace/ura/thanhdat/projects/CLRR/`. Khi có checkpoint mới, upload đúng public HF path, verify receipt/reload rồi mới xóa local weights.

## 6. Tổ chức code khi đủ điều kiện triển khai

Giữ `src/methods/original`, `precache/v1`, `centered_lsr/v1`, `token_lsr/v1`. Thiết kế được chọn sẽ có `src/methods/<method_id>/v1/` riêng; dùng registry + shared experiment runner/config/CLI args. Lưu formula, selection protocol, method revision, dataset/model hashes, runtime, arguments và receipts cùng checkpoint. Không tạo script trainer riêng cho từng run.

**Việc tiếp theo là thiết kế ứng viên có novelty thuyết phục hơn. Không có phương pháp mới nào đã được xác nhận novel hoặc hiệu quả trong lượt này.**

## 7. Quyết định tiếp theo — 18/10/2026

### 7.1 Bằng chứng và điều chưa được suy ra

Theo [token/beam diagnosis](TOKEN_DECODING_DIAGNOSIS.md), Full có NLL tốt
hơn nhưng sửa227 và làm mất272 reference-top1 decisions. Đây là liên hệ
ở checkpoint/validation, không chứng minh LSR gây lỗi hay CE bị cài sai.
Không điều chỉnh lambda theo riêng quan sát này.

Về đại số, với gold `y`:

```
CE_y = log(1 + sum_{j != y} exp(z_j - z_y)).
```

CE đã phụ thuộc vào margins với các competitors. Ví dụ phân phối
`[.40,.35,.25] -> [.42,.43,.15]`, gold index0: NLL giảm nhưng top1 từ đúng
thành sai. Test CPU đã xác nhận cả identity và counterexample này. Vì vậy
không gọi hiện tượng trên là lỗi metrics, cũng không mặc định thêm một
softplus margin sẽ tạo ra đóng góp mới hoặc cải thiện BLEU.

### 7.2 Đối chiếu prior mới đọc

| Primary source | Cơ chế kiểm tra | Quyết định cho dự án |
|---|---|---|
| [Selective KD, ACL2021 §5](https://aclanthology.org/2021.acl-long.504.pdf) | Chọn token khó theo CE để distill | Difficulty selection đã có; không gán novelty cho hard-token mask |
| [MTO/MSO, ACL2021 §3](https://aclanthology.org/2021.acl-long.268.pdf) | Chênh xác suất gold giữa NMT và LM; objective token/câu | Prior của source-dependence, **khác** margin gold-vs-competitor |
| [MRT, ACL2016 §3.2](https://aclanthology.org/P16-1159.pdf) | Expected sequence risk trên tập candidates | Dùng làm đối chứng đã biết; CLRR+MRT chưa mặc nhiên novel |

Đã đọc phần phương pháp/công thức của các nguồn trên. Đây là novelty audit
có phạm vi, không phải systematic review toàn bộ literature hoặc bảo đảm
không có prior khác. Các prior về selective self-KD/routing ở §3 tiếp tục
có hiệu lực. Chưa chốt tên phương pháp mới để đưa vào paper.

### 7.3 Audit không train đã thực hiện

Worker: `scripts/analysis/audit_sequence_candidate_pools.py`, có args cho
inputs, references và output. Dùng lại đúng R1/P1/R2 beam1/beam4 trên576
validation rows; xác minh hashes sáu prediction files, source/target/order
và replay sáu corpus BLEU/chrF0 với kết quả đã khóa trước khi tính oracle.
Không tải checkpoint, không GPU/generate/test/optimizer.

Oracle dùng reference để chọn output có **sentence chrF0 cao nhất** ở mỗi
câu. Nó chỉ là trần mean sentence chrF0 trong chính pool đó; corpus BLEU
của output được chọn là thống kê mô tả, không phải BLEU upper bound.

| Pool | Tối đa candidates/câu | Mean unique outputs/câu | Mean sentence chrF0 oracle | Corpus BLEU của oracle |
|---|---:|---:|---:|---:|
| Baseline beam1 + beam4 | 2 | 1.5556 | 18.887483 | 19.781382 |
| CLRR-only beam1 + beam4 | 2 | 1.5816 | 18.343225 | 18.807775 |
| Full beam1 + beam4 | 2 | 1.5747 | 18.559999 | 19.143823 |
| Baseline + CLRR-only, hai beams/model | 4 | 2.9792 | 21.130875 | 22.600413 |
| Baseline + Full, hai beams/model | 4 | 3.0069 | 21.364033 | 22.915261 |

Diễn giải: còn tính bổ sung giữa outputs của các model, nhưng khi mỗi model
chỉ có hai outputs, Full không vượt baseline ngay cả dưới oracle này.
Không dùng pool4 so pool2 để claim CLRR thắng; chưa có baseline bốn-output
control. Đây cũng chưa phải n-best list đầy đủ hoặc giới hạn khả năng của
architecture. Không có p-value confirmatory; không ghi các oracle numbers
vào bảng main như scores dịch thật.

Artifacts local `outputs/experiments/sequence_risk_preflight/POOL_AUDIT.json`
và `oracle_rows.csv`; CSV chỉ row indices/system names/scalars, không thêm
raw text. Chưa public các artifacts mới này lên HF.

Reproduce:

```powershell
conda run -n clrr python -X utf8 scripts/analysis/audit_sequence_candidate_pools.py --beam1-inputs outputs/experiments/amis_generation_validation_analysis/inputs --token-diagnostics outputs/experiments/token_decoding_validation --references data_processed/amis_mandarin/validation.csv --output outputs/experiments/sequence_risk_preflight
```

### 7.4 Prototype đã code và đã kiểm thử

Code riêng: `src/methods/sequence_risk/v1/loss.py`, không sửa original hoặc
registry/runner đang dùng. Hàm expected risk nhận differentiable sequence
sum log-probabilities, fixed costs và mask; posterior `softmax(alpha*score)`.
Cost detach, padding không có gradient, source-weighted mean. Đây là loss
core của **known control**, chưa phải pipeline có thể bấm chạy training.

CPU `clrr`: **8/8 tests PASS**, gồm công thức/analytic gradients; finite
differences; invalid-padding NaN; cost detach; offset/permutation invariance;
singleton/equal-cost zero gradients; update ưu tiên candidate tốt hơn;
invalid input guards; CE identity/counterexample. Một test có nhiều checks.
Không suy từ các gates toán học sang chất lượng model hoặc khả năng fit VRAM.

### 7.5 Thiết kế pilot có điều kiện: bốn runs, không phải chạy lại benchmark

Mục tiêu: kiểm tra tối ưu sequence có giúp riêng CLRR không, thay vì cứ
thêm alignment loss. Không đặt MRT làm main novelty. **Chưa mở queue.**

| Nhánh | Architecture | Objective | Câu hỏi |
|---|---|---|---|
| S0 | Baseline | CE continuation | Đối chứng cho việc train thêm |
| S1 | Baseline | CE + candidate-set risk | Known objective giúp baseline bao nhiêu? |
| S2 | Original encoder CLRR | CE continuation | Thêm route vào cùng initialization giúp gì? |
| S3 | Original encoder CLRR | CE + candidate-set risk | CLRR còn đóng góp khi objective đổi không? |

Đây là **đề xuất diagnostic**, chưa phải ứng viên novelty đã qua gate.
Nếu triển khai, cả bốn bắt đầu từ **cùng baseline R1 checkpoint đã khóa**,
bao gồm S2/S3 gắn route distance2/strength.1 vào weights đó. Không dùng
checkpoint CLRR đã train thêm làm initialization riêng. Không thêm LSR ở
factorial này; số LSR/Full cũ tiếp tục là lịch sử đối chiếu, chưa kết luận
phải bỏ LSR. Kết quả chỉ nói về adaptation từ baseline đã fine-tune, không
thay main benchmark training từ pretrained initialization.

Các điều kiện phải khóa trước khi chạy:

1. Dùng original Amis train/validation, seed42; không dùng test để lấy cost,
   chọn candidates, beta hoặc checkpoint. Validation đã dùng nhiều vòng
   development; không coi đây là confirmation độc lập.
2. Cùng RTX/runtime, FP32, LR, optimizer, batch/accumulation, shuffle,
   dropout, checkpoint selection và số **optimizer steps** cho bốn nhánh.
   Patience10 giữ policy hiện hành; không tự đổi tiêu chí selection sang BLEU.
   Cần ghi một budget continuation chung sau khi có benchmark step time.
3. Khóa candidate generation trên **train**: initial reference plus sampled
   outputs từ cùng R1, deduplicate token sequences; pool fixed và dùng chung
   cho S1/S3. Đây là restricted offline variant, không tuyên bố tái lập đầy
   đủ online MRT. Không lấy sáu validation outputs đã audit làm training pool.
4. Risk dự kiến `1 - sentence_chrF0/100` để tránh word-bigram sparsity tiếng
   Trung; là metric proxy, không thay human adequacy. Chưa có evidence metric
   này tối ưu. Alpha/beta/K và budget chưa khóa, **không sweep tự động**.
5. Candidate scores phải re-score bằng model hiện tại, sum token logP gồm
   EOS, bỏ pad và theo prefix convention đã khóa; không dùng beam scores có
   length penalty. Gradient đi qua scoring, không qua generate/metric.
6. Báo chi phí cả sampling/scoring; S0/S2 kiểm soát extra optimizer updates,
   không mặc định compute-matched. Thêm matched-compute control nếu paper
   cần claim efficiency hoặc attribution đòi hỏi nó.

Gates trước run: hoàn thiện generic args/runner, candidate dedup/masks/EOS,
Trainer accumulation và checkpoint reload; tiny random-model test với
dropout/checkpointing; GPU VRAM/forward/backward/optimizer smoke trên đúng
runtime. Chỉ kiểm tra GPU khi thật sự cần chạy, không đụng job người khác.

### 7.6 Cách ra quyết định sau pilot, nếu triển khai

Hai tương phản phải báo: `S3-S1` (CLRR contribution dưới risk objective) và
`(S3-S2)-(S1-S0)` (interaction giữa route và objective). Báo riêng
`S1-S0`, `S2-S0`, `S3-S0`; không chọn riêng seed/checkpoint thuận lợi.
S0/S2 cần chạy vì baseline cũ chưa được continuation cùng budget.

- Nếu chỉ S1/S3 cùng tăng tương tự: lợi ích của objective đã biết, **không
  cứu claim CLRR**. Không mở hàng loạt seeds cho nhánh này.
- Nếu S3 có lợi thế thêm nhất quán trên validation BLEU và chrF0, đồng thời
  margin/token gains không chỉ ở punctuation: mới cân nhắc fresh-seed pilot
  và ablation LSR; một seed chưa đủ xác nhận.
- Nếu không có gain hoặc pool quá suy biến: dừng objective control; không
  tăng loss weight vô hạn. Cần thay architectural hypothesis hoặc đổi framing
  bài, chưa được nói đã tìm ra phương pháp mới đủ sức ACL.

**Trạng thái kết thúc lượt này:** CPU preparation hoàn tất; chưa có train
mới hoặc số test mới. Novelty chưa đạt để gọi MRT/CLRR+MRT là contribution
mới; chưa chốt candidate architecture thay CLRR.

## 8. Recipe đã duyệt triển khai trên RTX6000 — 18/10/2026

**Kết quả gate:** full synthetic preflight PASS, nhưng shared train pool
không đạt readiness gate:2560/4600singleton (55.6522%),2572/4600equal-cost
(55.9130%). Ngưỡng khóa là<50% cho mỗi loại. Queue dừng trước training;
zero main models/steps. Còn2028rows có cost khác nhau, nên không kết luận
mọi MRT signal bằng0 hoặc dataset hỏng. Không thay threshold/K/temperature
trong recipe này. Cần thiết kế protocol mới nếu tiếp tục, giữ receipt thất
bại làm provenance. Chưa có phương án pool mới được chốt hoặc launch.

User: “rồi vậy thiết kế xong chạy trên RTX 6000 đi”. Chỉ validation, không
test. Recipe: `configs/experiments/amis_sequence_risk_continuation_rtx.json`.
Workspace: `outputs/experiments/amis_sequence_risk_continuation_rtx/`.

| Run | Method ID | Route | Risk weight |
|---|---|---|---:|
| S0 | sequence-ce-v1 | OFF, không gắn hook | 0 |
| S1 | sequence-risk-v1 | OFF, không gắn hook | 1 |
| S2 | sequence-clrr-ce-v1 | Original d2/strength.1 | 0 |
| S3 | sequence-clrr-risk-v1 | Original d2/strength.1 | 1 |

### Các lựa chọn đã khóa trước khi chạy

- Cùng R1 baseline archive pinned ở revision
  `e04195904361dcbdf36d4ae6c7f577ca37eb3256`, SHA256
  `7fbb9e66fc10ac9b1d75a0caaec9dcc314db0d1d5f450329bce5f810b4eec00c`.
  Tải trên RTX, kiểm hash, strict loading; cả bốn reset optimizer/scheduler.
- **3 continuation epochs**, danh định108 optimizer steps với4600train rows;
  cùng FP32/B4/GA32, **LR1e-5** (recipe continuation mới, khác LR5e-5 của
  benchmark gốc), linear scheduler/warmup.06, AdamW fused/weight_decay0,
  max_grad_norm1. Không gọi đây là chạy lại benchmark gốc. Patience10 giữ
  nguyên policy; cap3 là budget pilot nên không thể chạm patience10.
- Native HF flat-mean CE; whole-view non-reentrant checkpointing ở cả bốn.
  Selection giữ validation chrF++, beam1/FP32, earliest best của các epochs
  được train; step0 không nằm trong tập chọn best. BLEU zh/chrF0 log thêm.
- Alpha.05, beta1 cho risk arms, score_chunk2. Đây là fixed pilot choices,
  chưa tối ưu. Risk deterministic/dropout OFF, CE dropout như model gốc.
  CE arms không tính risk nên matched updates, **không compute-matched**.
- Một pool chung chỉ từ4600train rows: baseline R1 sample4outputs/câu,
  temperature1/top_k0/top_p1, sourcebatch2/seed1042 theo batch; cộng gold
  labels, dedup token sequences. Giữ EOS, bỏ decoder-start/padding sau EOS.
  Cost tính trên decoded candidate với **raw train reference**, không ép
  gold cost về0 nếu tokenizer/truncation làm khác raw string.
- Stop nếu ≥50%rows singleton hoặc equal-cost. Kiểm gate trước khi ghi
  COMPLETE và kiểm lại khi resume/load; không tự thay K/loss nếu pool fail.
- Primary GO development: S3−S1 BLEU≥.30, chrF0≥0; secondary S3−S0
  BLEU≥.30; interaction BLEU>0. Một seed không phải confirmation và MRT
  không trở thành novel chỉ vì có gain. Không sweep nếu fail.

### Thứ tự tự động

1. CPU gates trên frozen bundle, bốn arm wrappers/Trainer/save/reload.
2. Full-shape disposable synthetic GPU/AdamW step: B4,L256,K≤5,FP32,
   checkpointing; không train data hoặc giữ weights smoke.
3. Chuẩn bị/freeze shared train candidate pool; kiểm token/hash/gate signal.
4. **Cả bốn** real-data GPU gates, mỗi gate2optimizer steps với đúng GA32,
   optimizer/RNG save và best reload; xóa weights gate sau PASS.
5. Train S0→S1→S2→S3, độc lập từ R1. Mỗi model validation selection,
   save/reload, upload public, hash verification rồi mới xóa weights RTX.
6. Tạo `PILOT_SUMMARY.json` gồm đủ contrasts/interaction/GO; xóa bản tải
   initialization trong cache riêng khi đủ bốn verified backup receipts.
   Queue kết thúc tự nhiên; không shutdown RTX và không tác động job khác.

Public storage prefix cho mỗi run:
`checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/rtx6000-fp32-b4-ga32-es10-continuation-risk-pilot/`.
S0 dùng folder baseline, S2 dùng clrr; S1/S3 dùng sequence method IDs.
Prefix mới, không ghi đè checkpoint baseline hoặc CLRR cũ. Candidate pool
token IDs giữ trong project trên RTX, không tự public thành diagnostics.

### Điều phối và theo dõi

Toàn bộ env/cache/tmp/code/output trong
`/workspace/ura/thanhdat/projects/CLRR/`; lock `tmp/training.lock`.
Trước triển khai:23542MiB GPU free,666MiB used; project lock rảnh,
disk free49.57GiB, HF auth FiveC/public target đã xác minh. Tiến trình
462MiB của người khác giữ nguyên. Những số này là snapshot trước launch.

```powershell
conda run -n clrr python scripts/experiment.py inspect --config configs/experiments/amis_sequence_risk_continuation_rtx.json
conda run -n clrr python scripts/experiment.py build --config configs/experiments/amis_sequence_risk_continuation_rtx.json
conda run -n clrr python scripts/experiment.py cpu --config configs/experiments/amis_sequence_risk_continuation_rtx.json
conda run -n clrr python scripts/experiment.py launch --config configs/experiments/amis_sequence_risk_continuation_rtx.json --account 1
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_sequence_risk_continuation_rtx.json --account 1
```

Trạng thái thực tế sau launch phải đọc `launch_rtx.json`, `rtx_status.json`,
`PREFLIGHT.json`, `POOL_PROGRESS.json`, `gates/*/PASS.json` và progress từng
run. Không coi “queue started” là “đã train”; lỗi gate làm queue dừng.

Lưu ý provenance của template: recipe frozen còn các annotation lịch sử
`baseline_reuse`, `reuse_policy`, `reference_cohort`, `development_gates` từ
template token pilot. Shared engine **không đọc các annotation này** cho
continuation. Chúng không xác định initialization, hardware hay GO hiện hành.
Các trường có hiệu lực là `initialization`, `protocol`, `runs`,
`candidate_pool`, `decision_gates`; `run_metadata.json` xuất đúng các trường
run/protocol thực thi. Đặc biệt BF16/B32/GA4 trong historical
`baseline_reuse.matched_configuration` **không phải cấu hình pilot này**:
thực thi FP32/B4/GA32 từ R1 RTX đã pin. Giữ frozen plan nguyên trạng;
không sửa một lock đang chạy để xóa annotation. Recipe lần sau phải bỏ các
annotation template này trước khi freeze.

## 9. Tiếp tục sau pool gate: temperature1.3, namespace mới

**Trạng thái:** fullpool4600rows đã PASS:481singleton (10.4565%),
491equal-cost(10.6739%),meanunique3.2152,noEOS/lengthcap0/18400draws;
981.35s. Queue PID1620106 đang real-data GPU gates. Raw pool SHA256
`13e41666cce019cb1da270741f34d20b0c9c362a77f996d1222f1f14be68f493`.
Sourcebundle `3b401dba6705e91eef00fd04f4b9cfbb94f8d2118c07fda3b409c76643c66fae`.
Public reproducibility8files verified revision
`7d89f96eefb739e82dc011384e0b3d055744199c`, prefix
`analysis/amis_mandarin/mbart-large-50/sequence-risk-continuation/seed-42/t1p3`.
Không có quality result mới ở thời điểm cập nhật này.

Real-data GPU gates **4/4PASS** (2steps/save-reload/optimizer+RNG).
Queue đã chuyển **TRAINING/S0**, log1/108steps của maintraining được xác
nhận; chưa có model đã hoàn tất hoặc số chất lượng mới. Fullpool/preflight và
RUN_DESIGN.json cho cả4HFfolders đã public verified6files revision
`5f9af89e10dbcafcd36add92f7ae8dd248f159f2`. Folder có designmetadata không
đồng nghĩa đã có best.zip; checkpointweights được nativeworker upload sau
khi maintraining/validationselection/reloadverified hoàn tất.

Người dùng yêu cầu “chạy tiếp đi để chuẩn bị bước vô train”. Không sửa
old config/PLAN/lock/rejected pool trong `amis_sequence_risk_continuation_rtx`.

### 9.1 Probe không train đã xong

Worker args: `scripts/analysis/probe_sequence_pool.py`.
Chọn ngẫu nhiên128 original source batches (256train rows), seed2026;
giữ cặp câu và sampling seed `1042+original_batch_start`. Chỉ thử một
temperature mới đã chốt trước:1.3; samples4/gold/dedup/metric không đổi.
Không đọc validation/test cho selection, zero optimizer steps. T1 replay
khớp8/8rows về token sequences và costs; không thấy lỗi sampler gây lặp.

| Cùng256train rows | T1 cached | T1.3 |
|---|---:|---:|
| Singleton rows | 142 (55.4688%) | 28 (10.9375%) |
| Equal-cost rows | 142 (55.4688%) | 28 (10.9375%) |
| Mean unique candidates | 1.6953 | 3.1797 |
| Mean cost của unique non-gold candidates | .5472 | .6581 |

T1.3:1024draws,462gold duplicates,4non-gold duplicates,558new non-gold;
zero no-EOS/length-cap draws. Cost cao hơn là candidates kém reference
hơn, **không phải model tốt hơn**. Phép đo này chỉ chọn proposal support
đủ đa dạng cho known MRT control; không xác nhận quality hoặc novelty.

Artifacts: `outputs/experiments/sequence_pool_temperature_probe/`.
BaJSON local đã thu đúng bytes và SHA256 từ RTX; raw candidate IDs giữ RTX.
Report SHA256 `245c2e7af6541e8944dfa7677a1b9688829d288ef253b541825c5c1d3351692f`.

### 9.2 Recipe mới và điều không đổi

`configs/experiments/amis_sequence_risk_diverse_pool_rtx.json`, workspace
`outputs/experiments/amis_sequence_risk_diverse_pool_rtx/`.
S0–S3 vẫn cùng R1 pinned, seed42, FP32/B4/GA32, LR1e-5, cap3epochs,
patience10, alpha.05/beta1, whole-view checkpointing, validation chrF++
beam1 selection và GO thresholds ở §8. Không dùng baseline A100/BF16.

- Đổi đúng learning-data proposal: train sampling temperature1→1.3.
  Inference temperature không đổi. Cả S1/S3 dùng chung pool mới, native
  current-model sequence logP scoring không chia cho sampling temperature.
- Giữ singleton/equal-cost gate<50% trên **đủ4600rows**. Thêm guard<5%
  no-EOS và<5%actual length-cap draws; không đếm right-padding thành cap.
  Per-row sampling counters được lưu để resume và tổng hợp không mất số.
- Đã bỏ annotation template lịch sử trước freeze: baseline_reuse,
  reuse_policy, reference_cohort, development_gates và code_git_head cũ.
- Admission VRAM mới16000MiB, dựa full-shape measured peak9.2285GiB
  reserved và hơn6GiB headroom. Đây là resource guard, không đổi batch,
  precision/learning budget. Queue đợi nếu thiếu VRAM, không kill job khác.
- Public checkpoint prefixes mới: cùng §8 nhưng thêm hậu tố `-t1p3`.
  Không ghi đè baseline cũ hoặc outputs của rejected attempt.

### 9.3 Gates và execution

25/25 CPU unit gates PASS (8objective,10pipeline,7sampling accounting).
Bundle mới phải qua shared CPU4arms, full GPU/AdamW synthetic preflight,
full train-pool readiness và **cả bốn real-data GPU gates** trước training.
Sau đó tự train S0→S3, verified HF upload trước weight cleanup và summary.
Không test, không tự mở multiseed hoặc sweep weights.

```powershell
conda run -n clrr python scripts/experiment.py build --config configs/experiments/amis_sequence_risk_diverse_pool_rtx.json
conda run -n clrr python scripts/experiment.py cpu --config configs/experiments/amis_sequence_risk_diverse_pool_rtx.json
conda run -n clrr python scripts/experiment.py launch --config configs/experiments/amis_sequence_risk_diverse_pool_rtx.json --account 1
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_sequence_risk_diverse_pool_rtx.json --account 1
```

Receipt thực tế quyết định trạng thái: chưa có GPU PASS/POOL_COMPLETE thì
không ghi training đã bắt đầu; `runs/*/progress.json` xác nhận optimizer
step của model thật. Original failed recipe giữ nguyên làm audit history.
