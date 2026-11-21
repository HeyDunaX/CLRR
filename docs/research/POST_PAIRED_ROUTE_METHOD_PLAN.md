# Sau paired-route: chọn hướng tối ưu theo bằng chứng

## Trạng thái triển khai hiện hành — 13/11/2026

Tác giả đã duyệt triển khai ba pilot G/R/W. Code riêng tại
`src/methods/decoupled_consistency/v1/`, recipe
`configs/experiments/amis_decoupled_consistency_rtx.json`. Protocol A–D được
giữ nguyên: FP32/SDPA/TF32 off, B4/GA32, LR5e-5, ES10/max20, chọn validation
BLEU beam1. Không mở test, replication, Ê Đê hoặc A100 trong queue này.

**Chưa chạy GPU gates hay training:** RTX chỉ còn **3.89 GiB** trống dù GPU
rảnh (23952 MiB VRAM). Cần tối thiểu **40 GiB trống** trước restore/gates;
launcher kiểm tra điều kiện này trước khi triển khai. Không xóa dữ liệu người
khác hoặc môi trường conda để lấy chỗ. Chờ tác giả giải phóng dung lượng hoặc
chỉ định ổ được phép dùng; không tự đổi đường dẫn đã khóa.

Thứ tự queue đã chuẩn bị: CPU gate trên runtime RTX → frozen diagnostics
baseline/B → cả ba GPU gates → G → R → W. Diagnostics dùng **64 dòng đầu
validation**, dropout seeds **42001–42004**, hai native passes, không optimizer.
Gradient ở đây là **gradient theo logits**, không phải parameter gradient hoặc
cập nhật Adam tích lũy. Đây là phép chẩn đoán checkpoint cuối, không giải thích
toàn bộ trajectory. Checkpoint revisions và archive SHA được pin trong recipe.

Phần “DESIGN ONLY” dưới đây là lịch sử thiết kế; trạng thái hiện hành nằm ở đây.

## Cập nhật quyết định sau NotebookLM / primary-source audit — 13/11/2026

**DESIGN ONLY: chưa code trainer mới, chưa GPU gate/train.** Phần này thay thế
ngân sách pilot2runs ở §5 bằng **3runs đề xuất**, không sửa kết quả lịch sử.
Codex đề xuất công thức; NotebookLM được dùng để phản biện theo3 nguồn
R-Drop/LR-Drop/DKD; verdict dưới đây do Codex đối chiếu lại paper gốc.

### Chọn hướng và phạm vi novelty

Tiếp tục hai ablation G/R đã định nghĩa ở §4, thay vì thêm kiến trúc hoặc đưa
CLRR/LSR trở lại ngay. Tên mô tả: **gold/rest dropout-consistency ablations**.
Đây là ứng viên engineering để tìm cơ chế hữu ích, chưa là phương pháp novel
đủ làm đóng góp chính ACL. Gold/non-gold KL split đã có trong DKD, Eq4–7;
hai dropout views và symmetric KL đã có trong R-Drop. Hai views cùng weights,
hai phía nhận gradient, khác setting teacher/student nhưng không tự tạo novelty.

Giả thuyết cần đo: consistency trên gold-vs-rest có thể giữ ổn định xác suất
reference mà không buộc các lựa chọn khác khớp hoàn toàn. Giả thuyết cạnh tranh:
non-gold consistency mới là phần hữu ích, hoặc gain chỉ do giảm regularization.
Không có kết quả hiện tại chứng minh giả thuyết nào đúng. Gold consistency
không bảo đảm gold top1 đúng; non-gold có thể chứa lựa chọn dịch hợp lệ.

### Thêm một control cần thiết

| Run mới | Dataset/backbone/seed | Objective | Vai trò |
|---|---|---|---|
| G_S42 | Amis→Mandarin, mBART-50,42 | mean CE +2.5K_G | Gold-vs-rest consistency |
| R_S42 | như G | mean CE +2.5K_R | Weighted non-gold conditional consistency |
| W_S42 | như G | mean CE +1.25(K_G+K_R) | Full-KL nhẹ hơn, đúng nửa coefficient B |

Reuse baseline và A/B đã có sau kiểm provenance; không train lại C/D đã fail.
W là một control cố định, **không phải exact gradient-strength matching** và
không chứng minh loại hết strength confound. Không sweep lambda hoặc tự thêm
detach/constant-weight arms. Nếu muốn claim cơ chế vượt strength, phải thiết kế
đối chứng bổ sung trước khi kết luận; pilot này chỉ chọn recipe để kiểm chứng tiếp.

### Các phép cần chạy theo thứ tự

| Bước | Công việc | Runs training mới | Điều kiện |
|---|---|---:|---|
| 0 | Frozen baseline/B:64 validation rows cố định,4 cặp dropout masks khóa trước; KG/KR/full, logit-gradient norms/cosines với CE, gold probability/margin, agreement đúng/sai |0| Đọc best checkpoint đúng receipt; không optimizer, không chọn lại checkpoint/test; đo cuối training không giải thích được toàn bộ quá trình |
| 1 | Implement loss cô lập; kiểm padding/masks, large-vocab log-space/chunking, two-sided gradients, CE coefficient và save/load |0| CPU algebra4cases đã PASS chỉ là bước đầu |
| 2 | Tiny GPU gate cho G/R/W: accumulation/checkpointing, độc lập dropout, resume/generation equivalence, finite loss/gradient |0 train dài| Kiểm VRAM trước; không tác động jobs người khác |
| 3 | G/R/W seed42, pretrained initialization, cùng lock A–D |3| Chỉ sau mọi gate PASS và có quyền launch |
| 4 | Nếu G/R thắng gate: winner và full-KL control tốt hơn trong B/W, seeds43/44 |4 có điều kiện| Audit baseline43/44; thiếu matched baseline thì cần thêm2baseline, báo rõ trước chạy |
| 5 | Nếu qua nhiều seed: baseline/full-KL control/winner trên Ê Đê, seed42 |tối đa3 có điều kiện| Khóa recipe trước; chỉ reuse Ê Đê baseline nếu đúng protocol; chưa là confirmation độc lập vì Ê Đê đã được xem |

Ngân sách hiện đề xuất **3 runs**, không tự mở toàn bộ chuỗi. Nếu tất cả gates
nghiên cứu thuận lợi:3+4+tối đa3=**tối đa10 runs** trước replication Ê Đê/A100;
nếu baseline Amis43/44 không tái sử dụng được thì thêm2matched baselines.
Không triển khai test hoặc A100 trong bước thiết kế này.

### Gate trước pilot, thay thế ngưỡng pilot2runs ở §5

G/R chỉ được mở replication nếu validation BLEU ≥max(baseline,B,W)+0.30
và chrF++ ≥max(baseline,B,W). Ngưỡng BLEU tối thiểu hiện có là18.2107 nhưng
có thể tăng khi W hoàn tất; quy tắc max đã khóa trước nhìn số W. Nếu cả hai
pass: BLEU cao hơn, hòa thì chrF++, rồi G. Đây là practical development gate,
không là kiểm định thống kê. Nếu W thắng còn G/R fail, không nhận G/R là cải tiến;
chỉ giữ kết luận cần xem lại strength của R-Drop. Không tự mở grid.

Replication: báo paired deltas từng seed, mean/SD của deltas, CI và giới hạn n3;
không tuyên bố n3 đủ xác nhận significance. Chỉ mở transfer nếu mean advantage
vs baseline và best full-KL control đều≥.30BLEU, thắng ít nhất2/3seeds ở cả hai
so sánh, mean chrF++ không giảm; đây vẫn là practical gate đề xuất, không proof.
Nếu fail thì dừng nhánh hiện tại và trở lại thiết kế, không tune trên test.

### Đánh giá phản biện của NotebookLM

Receipt nhỏ:
`outputs/experiments/paired_route_error_analysis/NOTEBOOKLM_DECOUPLED_CONSISTENCY_AUDIT.json`.

- **Giữ:** prior split DKD; sự khác biệt teacher/student vs hai views cùng
  weights; strength confound; nguy cơ mất thông tin non-gold; kiểm masks/norms.
- **Sửa:** gradient qua1−p_gold là gradient hợp lệ của công thức đã chọn,
  không phải bug hoặc "spurious" mặc định. Nó có thể tăng gold mass để giảm
  penalty, nên R không phải phép đo thuần non-gold alignment. Đổi thành detach
  là đổi objective, không tự áp dụng. Nó không nói full R-Drop bỏ qua gradient đó:
  identity split bao gồm đường này trong gradient full KL.
- **Không nhận:** "LR-Drop proves hidden collapse" cho checkpoint CLRR;
  "DKD requires tuning" trong setting này; "gold-only chắc chắn kém" hoặc
  "non-gold chắc chắn là nhiễu". Đây là ngoại suy quá phạm vi bằng chứng.
- **Không làm:** lambda sweep5values, thêm detach/constant-weight variants
  hàng loạt như AI gợi ý. Chỉ thêm W như bounded control ở trên.

Primary papers đối chiếu:
[R-Drop](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html),
[DKD §3.1–3.3](https://arxiv.org/html/2203.08679v1),
[LR-Drop §3](https://arxiv.org/html/2402.16361v1).

---

Ngày 13/11/2026. **Đã phân tích CPU; chưa triển khai loss mới vào trainer,
chưa chạy GPU gate hoặc training mới.** Đây là kế hoạch development trên RTX,
không thay số hoặc protocol của cohort A100.

## 1. Kết luận hiện tại

Không bắt buộc giữ CLRR/LSR trong ứng viên mới. Trong factorial A–D vừa hoàn
tất, lợi ích quan sát được đến từ consistency; thêm CLRR làm giảm điểm ở cả
hai objective. Kết quả này không phủ định mọi biến thể CLRR, nhưng không ủng
hộ tiếp tục sửa topology/lambda vô hạn.

| Amis validation, seed42 | BLEU | chrF++ | ΔBLEU so baseline |
|---|---:|---:|---:|
| Baseline tái sử dụng | 17.6823 | 14.7030 | — |
| A: hai lượt native, mean CE | 16.8619 | 14.2623 | −0.8204 |
| B: A + symmetric KL | **17.9107** | **14.8427** | **+0.2284** |
| C: native/CLRR, mean CE | 16.6620 | 14.0011 | −1.0203 |
| D: C + symmetric KL | 17.3386 | 14.3730 | −0.3437 |

B là tốt nhất **trong bốn arms này**, chưa phải phương pháp mới được xác nhận
hoặc tốt nhất mọi thí nghiệm lịch sử. B−A=+1.0488 BLEU; D−C=+0.6766;
C−A=−0.1999; D−B=−0.5721. Một seed validation không xác lập hiệu quả qua seeds.
Baseline có caveat backend SDPA được tái dựng từ lịch sử, không có runtime
receipt đương thời; không xóa caveat này khi tái sử dụng.

## 2. Phân tích mới đã thực hiện

Worker có args: `scripts/analysis/analyze_paired_route_errors.py`.
Output nhỏ: `outputs/experiments/paired_route_error_analysis/`.

- Đối chiếu chính xác source/reference và thứ tự đủ576 rows với CSV validation.
- Replay BLEU zh/chrF++ khớp cả4 metrics đã lưu, sai số dưới1e-10.
- SHA256 của prediction/train/validation được ghi trong từng summary.
- Không đọc test, không chọn lại checkpoint, không tải weights.

| Nhóm B so baseline | n | ΔBLEU | ΔchrF++ |
|---|---:|---:|---:|
| Toàn bộ | 576 | +0.2284 | +0.1397 |
| Reference có0–10 Hán tự | 326 | +0.3570 | −0.1160 |
| Reference có11–20 Hán tự | 241 | +0.5879 | +0.2738 |
| Reference có21+ Hán tự | 9 | +1.5498 | +0.8724 |
| Có Hán tự xuất hiện≤5 lần trong train target | 306 | +0.4444 | +0.2424 |
| Không có Hán tự thuộc nhóm trên | 270 | −0.0377 | −0.0008 |

Bins là phân tích mô tả sau selection, không phải hypotheses đăng ký trước.
Nhóm9 câu quá nhỏ để làm claim. Tần suất **Hán tự** không phải tần suất từ,
subword tokenizer, hình thái hoặc bằng chứng đã giải quyết rare-word problem.
Subset corpus scores không cộng thành delta toàn bộ.

B thắng241/thua230/hòa105 câu theo sentence chrF++; chỉ57 câu có chuỗi
prediction giống baseline. Exact reference matches tăng7→9, gồm4 repairs và
2 breaks. Các thước này không đánh giá đầy đủ những bản dịch hợp lệ khác reference.
Tỷ lệ tổng Hán tự prediction/reference giảm0.9526→0.9352. Không được suy ra
hallucination/undertranslation hay tự chỉnh length penalty chỉ từ tỷ lệ này.

**Suy luận có giới hạn:** B có tín hiệu nhỏ, chưa sửa lỗi đồng đều. Phân tích
này không chỉ ra một bottleneck causal đủ chắc để thiết kế module kiến trúc.
Không ép dữ liệu thành kết luận rằng cần rare-token weighting hoặc source gate.

## 3. Related work đã đối chiếu trước khi chọn pilot

| Prior, phần đã đọc | Nội dung ảnh hưởng quyết định |
|---|---|
| [R-Drop, NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf), objective | Hai dropout passes và bidirectional KL đã có; B là engineering comparator kế thừa. |
| [LR-Drop, bản2024](https://arxiv.org/html/2402.16361v1), §3.1–3.3 | Đã regularize hidden states, attention và output giữa dropout views; không đề xuất thêm hidden alignment rồi nhận là mới. |
| [DKD, CVPR2022](https://arxiv.org/pdf/2203.08679), §3.1 và phần trình bày decoupling | Phân rã target-class/non-target KL đã có. Hai pilot bên dưới là ablation/ứng dụng kỹ thuật có dẫn nguồn, chưa phải novelty ACL. |
| [MTO/MSO, ACL2021](https://aclanthology.org/2021.acl-long.268.pdf), §3.1–3.2 | Margin giữa NMT và LM, cùng token objectives đã có. Không gọi source-confidence weighting là ý tưởng mới. |
| [ConsistTL, EMNLP2022](https://aclanthology.org/2022.emnlp-main.574.pdf), §3 | Parent/child consistency dùng pseudo source đã có; cần thêm teacher/data và không là thay đổi đơn giản trong paired objective hiện tại. |

Đây là audit các prior gần, không phải tuyên bố đã đọc hết literature.
Không có candidate nào dưới đây được chứng nhận là đóng góp mới đủ cho ACL.
Người dùng cho phép tối ưu engineering: có thể dùng chúng tìm recipe mạnh,
nhưng paper phải phân biệt recipe tốt với đóng góp phương pháp mới.

## 4. Hai pilot có kiểm soát: tách đúng objective đang có tín hiệu

Đây là **hai ablation engineering**, chưa là hai phương pháp mới để công bố.
Không thêm module, donor, teacher, dữ liệu hoặc loss alignment mới.

Với cùng gold prefix, hai dropout views cho phân phối p,q; gold token g:

\[
a=p_g,\quad b=q_g,\quad u=(a,1-a),\quad v=(b,1-b),
\qquad r_i=\frac{p_i}{1-a},\ s_i=\frac{q_i}{1-b}\ (i\ne g).
\]

\[
K_G=\tfrac12[KL(u\Vert v)+KL(v\Vert u)],
\qquad K_R=\tfrac12[(1-a)KL(r\Vert s)+(1-b)KL(s\Vert r)].
\]

Chain rule cho đúng \(K_{sym}(p,q)=K_G+K_R\). Giữ gradient qua cả hai
views **và** các khối lượng1−a,1−b; không detach hay bỏ weights để nhận nhầm
là cùng objective. Compute bằng logsumexp, không dùng log(1−p_gold) trực tiếp.
Gold labels chỉ dùng khi train/diagnostic, không dùng ở inference.

| Arm | Objective | Câu hỏi |
|---|---|---|
| A có sẵn | mean CE | Đối chứng hai lượt forward |
| B có sẵn | mean CE +2.5(K_G+K_R) | Đối chứng R-Drop-style mạnh nhất cohort |
| **G, pilot mới** | mean CE +2.5K_G | Chỉ ổn định gold-vs-rest probability có đủ, hoặc tốt hơn full-vocabulary consistency không? |
| **R, pilot mới** | mean CE +2.5K_R | Phần consistency giữa các lựa chọn còn lại đóng góp thế nào? |

Không khẳng định K_G làm gold top1 đúng hơn: hai views vẫn có thể đồng ý ở
một phân phối sai. Không mặc định K_R là nhiễu: DKD cho thấy non-target
knowledge có thể hữu ích trong setting của họ. Cần đo, không chọn trước kết luận.
Thay component cũng thay cường độ regularization; đây không tách hoàn toàn
hiệu ứng hình dạng objective khỏi gradient scale. Báo norms mô tả, không tự
chuẩn hóa gradient hoặc sweep hệ số trong cùng pilot.

## 5. Gate và thứ tự chạy

| Bước | Việc làm | Trạng thái / điều kiện |
|---|---|---|
| 1 | Replay và phân tích predictions A–D | **DONE**, đủ576rows/arm |
| 2 | Kiểm tra đại số loss và gradient hai phía trên CPU | **PASS**,4 cases float64; chưa là model gate |
| 3 | Code loss cô lập, correctness cho padding/ignore labels, chunking, vocabulary nhỏ/lớn, zero-KL, CE coefficient và save/load | **Chưa làm**; giữ code A–D cũ bất biến |
| 4 | GPU tiny gate trên RTX: forward/backward, checkpointing+accumulation, RNG/dropout, optimizer resume, generation bare-vs-wrapper | **Chưa chạy**; kiểm tra VRAM trước, không tác động process người khác |
| 5 | G và R, Amis→Mandarin mBART, seed42 | **2 runs mới đề xuất**, chỉ sau gates PASS; không chạy Ê Đê/test ngay |
| 6 | Nếu có winner: winner và B ở seeds43/44 | **4 runs bổ sung có điều kiện**; audit baseline43/44 có thật sự matched trước khi reuse |
| 7 | Chốt recipe rồi transfer Ê Đê / xác nhận A100 | Chưa mở; RTX exploratory không trộn số với A100 |

CPU algebra gate: `scripts/analysis/verify_consistency_decomposition.py`;
receipt `outputs/experiments/paired_route_error_analysis/KL_DECOMPOSITION_CPU_GATE.json`.
Vocabulary2/17 × logit scale1/10,128 examples/case, seed42. Max loss error
7.11e-15; max two-sided gradient error2.78e-17. Không kiểm chứng FP32/GPU,
large vocabulary, padding/chunked loss hoặc training quality bằng gate này.

Pilot giữ nguyên lock A–D: dữ liệu/split/tokenizer/revision, tl_XX→zh_CN,
FP32/SDPA/TF32off, b4×ga32, LR5e-5, warmup.06, weight decay0, fused AdamW,
maxgrad1, length256, ES10/max20, validation BLEU beam1 chọn checkpoint;
generation không thêm processor/length tuning. Công thức lấy mean trên cùng
supervised-label mask như B, không âm thầm loại EOS/language/special labels.

Prospective practical gate đề xuất cho2 pilots, khóa trước training:
BLEU≥max(B,baseline)+0.30, chrF++≥max(B,baseline). Với số hiện tại tương ứng
BLEU≥18.2107, chrF++≥14.8427. Đây là ngưỡng development thực dụng, không là
significance threshold hay bằng chứng power. Cả hai fail: dừng nhánh objective
này, không mở grid pha trộn K_G/K_R hoặc claim thành công từ subgroup.

Nếu cả hai pass, chọn BLEU cao hơn, hòa thì chrF++, rồi Arm G theo thứ tự
đã định. Confirmation nhiều seed phải báo toàn bộ seeds; không giữ riêng
seed đẹp. Phải đánh giá cả B để phân biệt đóng góp candidate với R-Drop.
Test Amis/Ê Đê đã từng được xem không trở thành holdout mới khi đổi seed.

**Ngân sách:** hiện0 training mới; bước pilot2 runs; nếu qua gate thêm4 runs
cho winner/B43–44. Baseline/A/B42 dùng lại receipts, không train lại để tô số.
Chưa có bằng chứng đủ để đưa CLRR/LSR trở lại hoặc gọi hai ablation này là
phương pháp chính của paper. Nếu cần novelty chính, phải có cơ chế và bằng
chứng bổ sung thực chất sau kết quả, không chỉ đổi tên loss kế thừa.
