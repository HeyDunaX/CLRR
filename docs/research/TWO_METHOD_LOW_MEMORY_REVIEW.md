# Hai hướng sau khi dừng exact corrective-order vì chi phí

Ngày 2026-11-19. Pha thiết kế: bằng chứng, primary papers, NotebookLM và CPU. Follow-up đã được tác giả cho làm: code riêng và ba chẩn đoán RTX trên checkpoint W cố định, **0 optimizer steps**, không validation/test. Không đổi implementation/recipe hoặc số lịch sử. Báo cáo đo: `outputs/experiments/amis_virtual_corrective_order_probe_rtx/REPORT.md`.

## Follow-up được duyệt: calibration và một pilot

2026-11-19: pretrained calibration PASS, lambda136.27606275570483, 3TRAIN batches,0steps;7CPU checks PASS. Queue PID139985 chạy GPU optimizer gate rồi chỉ khi PASS mới train VCO_S42, protocol native RTX FP32/SDPA/reentrant/B4GA32/ES10 BLEU. GPU gate đã PASS (save/load, cache, optimizer/scheduler/RNG resume; peak14.3486GiB). Training dài đã bắt đầu; chưa có chất lượng validation mới. Recipe `configs/experiments/amis_virtual_corrective_order_pilot_rtx.json`; ledger `docs/experiments/EXPERIMENTS_RTX6000.md` ghi trạng thái hiện hành. Các đoạn "chưa train" dưới đây là kết luận của pha chẩn đoán trước khi tác giả duyệt pilot, không phải lệnh cấm pilot đã duyệt. Không mở B hoặc test/replication.

## 1. Kết luận hiện hành

| Hướng | Vai trò | Kết luận sau audit |
|---|---|---|
| A — corrective-order trong tọa độ ảo nhỏ | Phát triển từ bằng chứng W và bài toán chi phí của exact unroll | Giữ làm ứng viên nghiên cứu.6CPU tests PASS; frozen probes và backward mBART hoàn tất trên24GB. Eta1 có signal/feature gradient ổn định; chưa train hoặc chứng minh chất lượng/cơ chế gây lỗi. |
| B — conditional innovation rewiring | Hướng kiến trúc riêng từ ý tưởng tránh truyền lại phần donor đã dự đoán được | Loại trước training: họ hàm tuyến tính thuộc LAuReL; chưa có bằng chứng redundancy gây lỗi để bảo vệ learning rule riêng. |

**Chưa có hai method cùng đủ điều kiện mở training.** Có hai hướng được kiểm tra, một hướng còn sống. Không gọi B là novel chỉ để đủ số lượng. A cũng chưa được chứng minh cải thiện dịch; phạm vi tìm kiếm chưa phát hiện exact match không phải chứng nhận novelty.

## 2. Bằng chứng và giới hạn

| Dữ kiện | Nhãn | Nguồn |
|---|---|---|
| Exact full-backbone corrective-order: 0/6 train dài; R2 gate xong 1/2 optimizer steps rồi OOM diagnostics microbatch33. Một optimizer step gate mất 7743.34 giây, không phải một epoch. | ĐO ĐƯỢC | `outputs/experiments/amis_corrective_order_finite_relu_rtx/rtx_status.json`; `gates/R2_ORDER_S42/microbatch_progress.json`; đầu `docs/experiments/EXPERIMENTS_RTX6000.md` |
| W là R-Drop beta1.25 đã có trong literature. Amis validation BLEU seeds42/43/44: 18.272410/18.395755/18.188117; baseline 17.682329/16.734501/16.208941. | ĐO ĐƯỢC, số lịch sử | `docs/research/METHOD_SEARCH_ROUND2.md`; `outputs/experiments/method_search_round2/RUN_EVIDENCE.csv`; ledger RTX |
| W test BLEU mean21.1617, baseline19.7306, AC20.7636 qua seeds42–44. Không tính significance mới trong lượt này. | ĐO ĐƯỢC, số lịch sử | `METHOD_SEARCH_ROUND2.md` §1.3 và `EXPERIMENTS_RTX6000.md` |
| CMC/CV validation42 15.540216/17.148787 dưới W18.272410. CV giảm empirical conditional gradient MSE81.3695% nhưng không cứu BLEU. Cả ba hoàn tất20epochs. | ĐO ĐƯỢC | `outputs/experiments/cmc_estimator_review/REPORT.md`; `outputs/experiments/amis_cmc_update_geometry_rtx/INTERPRETATION.md` |
| Auxiliary/CE cosine W+.395056, CMC+.361411 trên các batch đã đo; không có bằng chứng aggregate aux thường chống CE. | ĐO ĐƯỢC, phạm vi batch | Cùng geometry report; `METHOD_SEARCH_ROUND2.md` §2 |
| Learned-strength P1/P2 không vượt baseline trong pilot cũ; route probes nhỏ và CI chứa0. | ĐO ĐƯỢC | `docs/research/LEARNED_STRENGTH_DIAGNOSIS.md`; `RESIDUAL_UTILITY_ROUTING_AUDIT.md` |
| Ổn định dự đoán hữu ích, nhưng các proxy CE/alignment/gradient-noise không đủ dự báo generation. | SUY LUẬN | Các dòng trên; không chứng minh một nguyên nhân duy nhất. |
| Các views có đáp ứng sửa khác nhau và điều này gây lỗi dịch. | PHỎNG ĐOÁN cần đo | Chưa có phép đo trực tiếp trên checkpoint trained trong lượt này. |
| Truyền donor dư thừa là nguyên nhân CLRR không tăng. | PHỎNG ĐOÁN, chưa được hỗ trợ | Norm nhỏ, cosine cao/thấp hoặc negative quality riêng lẻ không xác nhận nguyên nhân này. |

Không pool A100/RTX, precision/backend khác nhau, BLEU/chrF++ selectors, beam1 validation/beam4 test hoặc ngôn ngữ khác nhau. Baseline42 có bằng chứng best BLEU/chrF++ cùng epoch19/step684; backend SDPA lịch sử vẫn là reconstruction. Số test đã xem chỉ là development evidence.

## 3. Sàng lọc trước khi giữ hai card

| Ý tưởng | Lý do không nhận làm method chính ngay |
|---|---|
| Corrective-order chỉ trên logits | Có nguy cơ rút về một consistency loss phụ thuộc label; khác biểu thức chưa đủ contribution. |
| Sửa token sai và bảo toàn token đúng | Model-editing locality (ENN/MEND) và gradient-interference priors rất gần; chưa có cơ chế riêng rõ. |
| Adapter thật nhỏ + order loss | Thêm inference parameters và confound capacity. Đã bỏ adapter persistent trong card A cuối cùng. |
| Thêm gradient projection vào W | Primitive PCGrad đã audit ở round2; aggregate conflict chưa được chứng minh. |
| Conditional innovation donor | Card B dưới đây: linear architecture nằm trong prior đã đọc. |
| Virtual reset order probes | Card A dưới đây: giữ để kiểm chứng, không phải bản exact full-parameter cũ chạy nhanh hơn mà là objective khác. |

## 4. Card A — finite corrective-order trong tọa độ ảo nhỏ

### RQ và động cơ

**RQ:** Học tính ổn định của *đáp ứng sau sửa* có giúp generalization hơn việc chỉ khớp dự đoán hiện tại giữa các dropout views không?

W tốt qua ba seeds là động cơ nghiên cứu consistency. Exact unroll quá đắt là động cơ thay không gian đo. Hai điều này chưa chứng minh noncommutativity là nguyên nhân lỗi dịch. Không gán cho R-Drop cơ chế chưa đo hoặc nhận nó làm phát minh của project.

### Công thức cuối cùng — không có adapter thật

Với backbone native cho hidden state cuối decoder `h_i ∈ R^d`, output head `E ∈ R^(V×d)` và logits gốc `z_i=E h_i` (giữ bias native nếu có):

1. Chọn basis trực chuẩn cố định `B ∈ R^(d×r)`, `P=B^T`.
2. `S=E B`, `u_i=P h_i`.
3. Tạo ma trận tọa độ **ảo** `A ∈ R^(r×r)`, reset về0 mỗi microbatch. A không nằm trong optimizer hoặc checkpoint model và không tồn tại lúc inference.
4. Logits dưới phép sửa ảo: `z_i(A)=E h_i + S A u_i`.

Hai dropout passes native a,b trên cùng microbatch cung cấp hidden states. Chia các TRAIN rows của chính microbatch thành support D và probe Q không giao nhau; không lấy validation/test làm inner support. CE vẫn dùng toàn bộ rows/tokens của cả hai views.

Với N_D là số target tokens hợp lệ trên D, gradient theo tọa độ A có biểu thức tường minh:

\[
G_a(A)=\frac1{N_D}\sum_{i\in D}
 \{S^T[\operatorname{softmax}(z_{a,i}(A))-e_{y_i}]\}u_{a,i}^T,
\quad U_a(A)=A-\eta G_a(A).
\]

Tương tự cho b. Tạo hai đường sửa:

\[
A_{ab}=U_b(U_a(0)),\qquad A_{ba}=U_a(U_b(0)).
\]

Trên cùng features probe q của mỗi view v, so hai endpoints (không resample dropout giữa AB/BA):

\[
R_{order}=\frac1{2|Q|}\sum_{v\in\{a,b\},q\in Q}
 \operatorname{SKL}(p_{v,q}(A_{ab}),p_{v,q}(A_{ba})),
\]

\[
L=\tfrac12(L_{CE,a}+L_{CE,b})+\lambda R_{order},
\qquad \operatorname{SKL}(p,q)=\tfrac12[KL(p\Vert q)+KL(q\Vert p)].
\]

Trong công thức trên Q ký hiệu các token hợp lệ thuộc probe rows; normalization phải thống nhất khi implementation. Chia theo rows, không chia các token của cùng câu sang cả support và probe.

`h,E` được giữ cố định khi định nghĩa phép đạo hàm theo tọa độ A, **nhưng không detach khỏi outer graph**. Outer optimizer cập nhật toàn bộ backbone như trước. Tính G bằng đại số trên hidden/logits, không gọi higher-order autograd qua một bản sao toàn backbone. Không tuyên bố objective không chứa đạo hàm của head gradient.

Không dùng `eta^-4` như một luật scaling đã xác minh. Chẩn đoán đã khóa r32, bases11/37, chia rows2/2 luân phiên, eta grid0.01/0.1/1 chỉ để đo số học. Eta1 được giữ sau numerical check; không phải chọn theo chất lượng. Lambda chưa khóa cho pilot từ pretrained: giá trị scale đo tại W chỉ áp dụng checkpoint W đó. Cần hiệu chuẩn tại đúng initialization trước pilot.

### Chi phí và khác biệt

- Với r32, mỗi A có1024 scalars thay cho một trạng thái virtual chứa toàn bộ backbone. Đây chỉ là đếm tọa độ, **không phải số lần giảm VRAM toàn hệ thống**.
- Tái sử dụng hai native backbone forwards; thêm các phép output-head/softmax cho virtual steps và probe. Không inference adaptation, không thêm tham số inference, không thêm dictionary hoặc dữ liệu mới.
- `E B` và tensors token×vocabulary vẫn tốn RAM/VRAM; outer graph vẫn có chi phí. Đã đo short-batch mBART forward/backward khoảng6.315GiB peak allocated; không phải optimizer/training VRAM hoặc bound cho256tokens. Chi tiết timing và eta trong REPORT.
- Không mô phỏng exact full-parameter update cũ. A là một phép đo đáp ứng trong subspace cố định, có thể bỏ sót phần quan trọng.
- Có thể áp cho autoregressive backbones có hidden state/output head tuyến tính; tokenizer/ngôn ngữ không nằm trong định nghĩa loss.

### Kiểm toán học đã thực hiện

CPU FP64, synthetic tensors, Python conda clrr. Kết quả giữ nguyên tại `outputs/experiments/two_method_low_memory_design/MATH_CHECKS.json`:

| Check | Kết quả |
|---|---:|
| Explicit inner gradient vs autograd | max error5.55e-17 |
| Hai endpoints explicit vs reference | max error5.55e-17 |
| Outer gradients qua features/head | max error1.84e-18 |
| Cùng view hai lần | endpoint distance0 |
| Đổi nhãn a/b | endpoint exchange error0 |
| Gradient outer về features | norm0.0005027034, khác0 trong fixture |
| Fixture native SKL=0 nhưng order-probe SKL>0 | 1.9591031e-7 |

Fixture cuối dùng output head rank-deficient. Nó bác **đồng nhất đại số phổ quát** với một loss chỉ phụ thuộc distributions ban đầu; không chứng minh checkpoint mBART thực có hidden nullspace này. Nếu E full column rank và softmax gauge không tạo thêm ambiguity, exact distribution equality có thể xác định hidden state; phải kiểm trực tiếp trên model thật thay vì dùng toy làm mechanism evidence.

Lệnh thực thi dạng: PowerShell inline Python qua `C:\Users\Lenovo\miniconda3\envs\clrr\python.exe -X utf8 -`. Không load weights, optimizer, CUDA hoặc test. Raw result được giữ; chưa tạo trainer hoặc một bộ gates production.

### Falsifiers trước training

1. Frozen checkpoint, TRAIN probe độc lập: so order response với null replay, numerical precision repeat và displacement-matched random updates. Nếu signal không tách được noise, dừng.
2. Kiểm order response có chỉ lặp lại static W disagreement, update norm hoặc entropy không. Không suy nhân quả từ correlation; dùng các controls này để đánh giá thông tin bổ sung.
3. Đo projected feature energy/rank và ảnh hưởng của basis. Nếu loss biến mất do model tránh subspace, không nhận cơ chế thành công.
4. Đo peak VRAM và wall time cùng batch/backend với CE2 và W. Nếu không đạt ngân sách thực tế tác giả chấp nhận, không mở pilot. Gate cũ PASS không chuyển sang objective mới.

Follow-up frozen GPU đã hoàn tất:48 settings trên8unique TRAIN microbatches, gradient precision/direction trên8batches và full-backbone norm trên3batches. Null/replay đúng; FP32 eta0.01/0.1 không đáng tin tại W, eta1 ổn định hơn. Không có training admission; không lấy gradient khác W làm bằng chứng sửa đúng lỗi dịch. Xem REPORT để phân biệt checks đã làm với causality/optimizer/quality gates còn thiếu.

### Nếu qua cơ chế và tài nguyên

Pilot Amis seed42 validation-only: A trung tâm + weight thấp/cao đã khóa; CE2 và W chỉ reuse khi metadata khớp, nếu không cần matched controls. Không thêm W vào A cùng lúc vì sẽ không biết benefit đến từ đâu. Cần order-vs-symmetric-response control và norm-matched random-update control trước claim cơ chế; số runs chốt sau audit reuse.

Giữ LR/batch/precision/backend, max20, ES10 validation BLEU beam1 như cohort RTX đã khóa. Chỉ sau khi khóa một cấu hình mới replication≥3seeds, đánh giá test beam4 một lần cho mỗi checkpoint; paired sentence bootstrap với family correction công bố rõ và paired seed-level inference tách biệt. Test Amis đã xem là exploratory. Ê Đê là kiểm tra chuyển ngôn ngữ theo recipe/data gốc sau khóa method; A100 là cohort paper riêng. Đo BLEU, chrF0, chrF++, variance và compute; đối sánh baseline/W/Middle/LoRA/BitFit cần cùng protocol và audit reuse, không trộn RTX vào A100.

### Rủi ro chính

Không có bằng chứng rằng bắt correction paths commute là mục tiêu tốt cho dịch; thứ tự sửa khác nhau có thể hữu ích. Fixed basis có thể bị backbone né; support nhỏ có thể nhiễu; signal có thể quá yếu hoặc saturation khi eta lớn. Vì vậy đại số PASS không đồng nghĩa nên train dài ngay.

## 5. Card B — conditional innovation rewiring, bị loại

RQ dự kiến: CLRR có truyền lại phần donor đã dự đoán được từ recipient, và chỉ giữ phần còn lại có tốt hơn không?

Với donor d_l và recipient h_l, dùng TRAIN running moments để fit predictor tuyến tính M_l; định nghĩa:

\[
r_l=d_l-M_l h_l,\qquad h'_l=h_l+W_l r_l.
\]

Giả thuyết: prediction error là phần donor chứa thông tin bổ sung. Không xây dictionary, không thêm pass backbone; nhưng giả thuyết chưa được chứng minh bởi số CLRR hiện có. Least-squares residual có điều kiện cross-covariance nhất định, không tự độc lập thống kê và không vuông góc từng token.

Với M cố định tại thời điểm forward:

\[
h'=(I-WM)h+Wd.
\]

Đây là tổ hợp tuyến tính activations hiện tại/cũ, thuộc framework LAuReL Eq2–6. Fit M bằng moments là learning rule khác, không tự tạo kiến trúc mới. ORU Eq3–4 đã có primitive loại thành phần residual song song với stream hiện tại, dù không đồng nhất toàn bộ với conditional regression này.

CPU kiểm affine collapse max error4.44e-16. NotebookLM ban đầu đánh giá lạc quan khi chưa đối chiếu LAuReL; vòng cuối đã rút lại. **NO-GO làm method chính ở thiết kế hiện tại.** Không thêm whitening/nonlinear gate chỉ để né prior. Có thể giữ ý tưởng như control kỹ thuật, không mở train tìm số rồi hợp thức hóa novelty.

## 6. Paper gốc đã mở và phần đối chiếu

| Paper | Phần đọc | Liên hệ và giới hạn |
|---|---|---|
| [R-Drop, NeurIPS2021](https://arxiv.org/html/2106.14448) | §2.1–2.2 | Static bidirectional KL giữa hai dropout outputs; đã có NMT. A khác ở finite correction-path response, không nhận two-pass consistency làm mới. |
| [ANIL, ICLR2020](https://arxiv.org/html/1909.09157) | §3.1, §4 | Restrict inner adaptation to head; A không có task/test-time adaptation, objective là path divergence. Restricted inner coordinates riêng lẻ không mới. |
| [CAVIA, ICML2019](https://proceedings.mlr.press/v97/zintgraf19a/zintgraf19a.pdf) | §2–3, Eq3–4 | Inner context adaptation, outer shared parameters. A reset virtual matrix để tạo regularizer; không được nói reset context là khác biệt, vì CAVIA cũng reset context. |
| [Editable Neural Networks, 2020](https://arxiv.org/html/2004.00345) | §3.1–3.2, Eq1–4 | Training for edits và locality; có translation experiments. Khác task MT không đủ novelty; không đồng nhất KL locality với gold-CE hinge. |
| [MEND, ICLR2022](https://arxiv.org/html/2110.11309) | §2–3 | Learned low-rank gradient transforms cho editing. A không có editor network hoặc persistent edit, nhưng correction-based learning có prior. |
| [Commute Your Domains, 2025](https://arxiv.org/html/2501.15556) | §3.2, Theorem3.1, Corollary3.2 | Update order/Lie bracket trong domain curricula. Không nhận commutator hoặc order effects làm phát minh của A. |
| [LAuReL, ICML2025](https://arxiv.org/html/2411.07501) | §2, Eq2–6 | Current/prior activation linear augmentation bao quát họ hàm B khi M cố định. Không khẳng định cùng mọi learning rule. |
| [Orthogonal Residual Update, 2025](https://arxiv.org/html/2505.11881) | §3.1–3.2, Eq2–4 | Residual orthogonalization primitive có trước; không đồng nhất per-token geometric projection với conditional regression. |

**A: GẦN.** Trong các truy vấn và paper đã đọc chưa xác minh được exact same reset-matrix, two-dropout AB/BA probe objective. Đây là kết luận giới hạn, không khẳng định chưa ai làm. Đổi inner coordinates một mình là incremental; contribution nếu có phải là objective, cơ chế thực nghiệm và quality/efficiency vượt matched controls.

**B: GẦN ở complete learning rule, họ kiến trúc đã có; LOẠI làm architectural novelty.** Không lấy một algebraic umbrella prior làm chứng minh mọi thuật toán đều trùng; ở đây còn thiếu distinct learning evidence để bảo vệ hướng.

## 7. Nhật ký NotebookLM và các lỗi đã bác

Notebook `bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`, conversation `8ec5ca51-f2be-4cca-a640-c5b4035a99d5`. Chỉ gửi lý thuyết/literature. Toàn bộ prompts/answers cho lượt redesign và ba vòng phản biện được lưu ở `outputs/experiments/two_method_low_memory_design/NOTEBOOKLM_DEBATE.json` (turns7–10).

- Vòng đầu: nearest priors, reviewer reject cases, A nhỏ/B locality. Không coi gradient-vector-field bracket là tích giao hoán của gradient matrices.
- Vòng hai: chỉnh công thức A, thử B innovation; yêu cầu tìm LAuReL/ORU. NotebookLM lúc này còn thiếu hai primary sources nên không thể nhận verdict tích cực.
- Vòng cuối: gửi đúng parameter-neutral reset design A và affine collapse B, yêu cầu phản bác; NotebookLM nhận rủi ro nullspace/noise/compute của A và rút lại clearance B.
- Không nhận các lỗi trong câu trả lời: CAVIA cũng reset context; không gọi A hoàn toàn first-order về mặt toán vì vẫn differentiate head-gradient algebra; không có real adapter nên adapter-only CE không còn là control thích hợp. Control cần CE2/W và các response alternatives đúng final design.
- Fixed finite eta không bảo đảm hết cancellation; bỏ eta^-4 không phải khẳng định chia eta^-4 vốn gây underflow. Fixed basis không ngăn backbone/E né penalty. Exact output equality không suy ra hidden khác nhau cho mọi full-rank output head.
- Cosine thấp không chứng minh không có redundancy; shuffled moments không tự tạo norm-matched noise; least-squares residual không tự vuông góc từng token. Không nhận numeric gates do NotebookLM tự bịa hoặc secondary citations chưa đọc.

Đã thêm sáu original-source URLs ANIL/CAVIA/ENN/MEND/LAuReL/ORU vào notebook trong lượt này. MCP trả nguồn được tạo; không dùng việc import như bằng chứng novelty.

## 8. Nhật ký truy vấn web

Các truy vấn sau đã thực hiện; paper chỉ xuất hiện trong search nhưng chưa mở không dùng làm căn cứ.

### A / corrective response và small-coordinate meta-learning

1. `"meta-learning" "dropout" "gradient alignment"`
2. `"neural machine translation" "correction" "low rank" meta learning`
3. `"dropout" "cross-view" "gradient" correction learning`
4. `site:arxiv.org "commutator" "regularization" "low rank"`
5. `site:aclanthology.org "meta-learning" "dropout" consistency`
6. `site:semanticscholar.org "order invariant" "meta learning" gradient`
7. `ANIL almost no inner loop paper arxiv`
8. `CAVIA fast context adaptation via meta learning arxiv`
9. `model editing gradient transform small network MEND locality objective paper`
10. `site:aclanthology.org "gradient" "interference" "regularization" translation`
11. `"order consistency" "meta learning" adapter`
12. `"commutator" "dropout" regularization`
13. `"corrective" "locality" "representation learning" gradient`
14. `"gradient interference" "meta-learning" "regularizer"`
15. `site:arxiv.org "virtual" "subspace" "consistency" gradient regularization`
16. `site:semanticscholar.org "correction" "order" "dropout"`
17. `"low rank" "Lie bracket" regularization neural networks`
18. `"dropout" "virtual update" consistency regularization`

### B / innovation residual

1. `"transformer" "innovation" "residual" connections redundancy`
2. `"conditional" "redundancy" "skip connections"`
3. `"orthogonal" "residual connections" transformer`
4. `"LAuReL" residual learned augmented`
5. `site:aclanthology.org "conditional" "innovation" residual`
6. `site:semanticscholar.org "residual" "innovation" neural network`
7. `"conditional covariance" "residual connections" transformer`
8. `"innovation residual" "neural network" skip`

### Discovery ban đầu

- `"neural machine translation" dropout "error correction" consistency`
- `"logit" "Jensen" dropout correction softmax`
- `"residual" "transport" "Jacobian" transformer layers`

## 9. Việc tiếp theo thực sự

A đã chạy xong chẩn đoán cố định, chưa train. Cần khóa initialization-specific scale và optimizer preflight trước validation pilot; không dùng lambda hiệu chuẩn trên trained W để khởi tạo pretrained model. B không đưa vào queue. Nếu yêu cầu hai hướng đều train-ready, phải audit một B khác. Thiếu causal/quality evidence vẫn được ghi rõ; toy PASS, frozen backward hoặc NotebookLM không thay kết quả trained method.
