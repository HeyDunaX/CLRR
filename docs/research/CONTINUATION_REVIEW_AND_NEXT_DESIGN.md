# Kiểm chứng continuation và quyết định thiết kế tiếp theo

## Material Passport

- Ngày: 26/10/2026. Trạng thái: bước 1–2 VERIFIED bằng artifacts; bước 3 DESIGN / NOVELTY HOLD.
- Dataset: Amis→Mandarin; validation 576 câu, seed42, beam1, FP32, TF32 off.
- Không train, không đánh giá test, không tải weights, không chạy GPU trong lượt này.
- Kiểm tra RTX trước khi dự kiến inference: Quadro RTX6000, 24.576 MiB tổng, 23.542 MiB trống, utilization 0%. Sau đó xác minh có thể dùng lại predictions nên không chiếm GPU.

## 1. Checkpoint khởi tạo: đã tìm và xác minh

R1 đúng là checkpoint dùng để khởi tạo bốn nhánh continuation. Archive SHA256:
`7fbb9e66fc10ac9b1d75a0caaec9dcc314db0d1d5f450329bce5f810b4eec00c`.
Repo `FiveC/amis-rewire-checkpoints`, commit khởi tạo được khóa trong PLAN:
`e04195904361dcbdf36d4ae6c7f577ca37eb3256`.

Đã lấy predictions/metadata nhỏ tại chính commit này, đối chiếu archive LFS hash qua API, rồi tái tính scores. Original references/hash, thứ tự câu, source/target token, max lengths 256/256, eval batch2, beam1, precision, metric settings và các phiên bản thư viện đều khớp. Không cần inference trùng.

Đây là tái sử dụng **inference artifacts**. R1 train 20 epochs, chọn epoch19; S0–S3 reset optimizer rồi continuation 3 epochs. Không gọi hai quá trình training là giống hệt nhau. Predictions R1 là step-zero baseline; chưa phải kết quả sau khi chèn CLRR vào R1 với strength0.1.

## 2. Đối chiếu với initialization

| Model | BLEU | chrF++ | chrF0 | Δ BLEU so với R1 |
|---|---:|---:|---:|---:|
| R1: checkpoint khởi tạo | 17.6823 | 14.7030 | 16.9547 | — |
| S0: baseline + CE continuation | 17.2803 | 14.4668 | 16.6790 | −0.4021 |
| S1: baseline + CE + risk | 17.5593 | 14.4865 | 16.7025 | −0.1230 |
| S2: CLRR + CE | 17.4191 | 14.3061 | 16.4921 | −0.2632 |
| S3: CLRR + CE + risk | 17.2666 | 14.2728 | 16.4245 | −0.4157 |

Cả 12 epoch evaluations (4 nhánh × 3 epochs) đều thấp hơn R1 trên cả ba metrics. Không thể giải thích toàn bộ phần giảm bằng riêng CLRR/risk vì nhánh CE thuần cũng giảm.

5.000 paired sentence bootstrap resamples (RNG20261026), CI95% percentile theo câu:

| So sánh | Δ BLEU | CI95% |
|---|---:|---|
| S0−R1 | −0.4021 | [−1.1679, +0.2884] |
| S1−R1 | −0.1230 | [−0.7914, +0.5519] |
| S2−R1 | −0.2632 | [−1.1126, +0.5847] |
| S3−R1 | −0.4157 | [−1.2574, +0.4368] |

Toàn bộ 12 CI của ba metrics đều chứa0. Không chứng minh harm ổn định, equivalence hoặc hiệu quả qua seed. Đây là validation đã được dùng để chọn R1; winner selection/regression to the mean là giới hạn thực tế, không phải confirmation độc lập. Không dùng các CI hậu nghiệm như familywise significance.

**Quyết định:** NO-GO tiếp tục mở rộng recipe continuation hiện tại. Kết quả phù hợp với suy giảm khi fine-tune tiếp checkpoint đã hội tụ, nhưng chưa tách được overfitting, reset optimizer và schedule. Không kết luận đã tìm ra nguyên nhân nhân quả. Kết luận cũ “CLRR+risk không đạt gate” vẫn giữ; gain S1−S0 không đồng nghĩa vượt checkpoint khởi tạo.

## 3. Chốt hướng nâng cấp và novelty gate

### 3.1 Phương án cụ thể được xem xét

Thay **strength cố định0.1** bằng scalar học được cho từng route, khởi tạo0:

\[
u_l=F_l(h_{l-1}),\qquad
h_l=u_l+a_l\,\operatorname{sg}(h_{l-2}),\qquad a_l(0)=0.
\]

Giữ distance2, encoder-only, detached post-rewiring donor, không thêm LSR/risk trong phép thử cơ chế. Với encoder12 layers có10 route scalars mới: **không còn parameter-neutral**. Không thêm clipping, warmup riêng hay loss mới cùng lúc.

- Bằng chứng động cơ: route và weights có co-adaptation trong audit cũ; continuation baseline cũng giảm trong audit mới.
- Giả thuyết cần kiểm tra: chèn route có thể làm lệch hàm pretrained trước khi weights thích nghi; bắt đầu với route0 giúp bảo toàn hàm ban đầu. Dữ liệu hiện tại **chưa chứng minh** sự lệch này là nguyên nhân giảm BLEU.
- Khi mọi `a_l=0`, forward khớp backbone gốc; gate vẫn nhận gradient CE. Detach chỉ chặn gradient bổ sung qua cạnh donor, không chặn đường tuần tự của encoder.
- Không bảo đảm gate sẽ học giá trị hữu ích hoặc BLEU sẽ tăng.

### 3.2 Đối chiếu prior đã đọc lại

| Prior | Phần đã có | Điểm khác và quyết định |
|---|---|---|
| [ReZero, UAI2021, §3 Eq.(4), §5.2 Eq.(10)](https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf) | Scalar residual học được khởi tạo0 | Áp dụng lên cạnh CLRR thay vì nhánh sublayer không đủ để nhận novelty thuật toán chính |
| [Lexical Shortcuts, WMT2019](https://aclanthology.org/W19-5211/) | Gated shortcuts từ embeddings tới encoder/decoder layers trong NMT | Donor distance2 và stop-gradient khác, nhưng generic gated shortcut đã có |
| [Value Residual Learning, ACL2025, §3.2, Tables3–4](https://aclanthology.org/2025.acl-long.1375.pdf) | Cross-layer residuals và các biến thể learnable mixing, gồm hidden/value residual controls | Không được xem “học strength thay constant” là đủ mới |

**Novelty verdict: NO-GO làm phương pháp ACL mới chính.** Phương án này chỉ là adapted diagnostic control có căn cứ để kiểm tra bảo toàn hàm khởi tạo. Không đăng ký thành method thắng, không code/train production để tìm gain rồi mới hợp thức hóa novelty. Novelty audit có phạm vi, không chứng nhận tuyệt đối vắng prior.

**Vì vậy bước3 kết thúc bằng quyết định HOLD, không phải một candidate mới đã được duyệt.** Chưa có đủ bằng chứng để chọn ngay một thay đổi kiến trúc vừa có lợi vừa đủ khác prior. Không lặp lại gate/centered-LSR/token-LSR đã được đánh giá chỉ với tên mới.

### 3.3 Đối chứng và correctness gates nếu quyết định dùng làm nghiên cứu cơ chế

Chưa mở những runs này. Pilot có điều kiện gồm CE baseline, fixed CLRR và zero-init learned-strength CLRR, cùng pretrained gốc và protocol đã khóa, seed42 trước. Không LSR/risk; cùng data/order/optimizer/update budget/checkpoint selection. Baseline cũ chỉ dùng lại khi gate recipe + gradient implementation equivalence cho phép; không mặc định phải train lại hoặc mặc định luôn tái sử dụng được. Các số baseline hiện hành không bị thay thế.

Các gate cần PASS trước training:

1. Scalar0: logits và CE bằng backbone gốc trong eval, cùng batch/language/masks; kiểm tra cả0-strength generation.
2. Gradient hữu hạn: gate nhận gradient khi donor khác0; donor chỉ mất gradient qua cạnh skip, đường encoder tuần tự vẫn có gradient. Không so gate snapshot với epoch-mean gradient.
3. Caches tách theo forward; train/eval/generate và checkpoint recomputation không dùng donor cũ; gate thuộc optimizer/state_dict, không bị detach nhầm.
4. Save/reload khớp logits, gates, metadata và số parameters; không missing/unexpected keys.
5. CPU numerical tests trước; tiny real-data GPU gate sau khi kiểm tra VRAM; không đổi batch/precision để tự qua gate.

Step-zero và mỗi epoch phải được ghi ở mọi arm. Quy tắc best checkpoint bao gồm hay loại step-zero phải khóa **trước run**, áp dụng đồng nhất; không sửa selection của S0–S3 sau khi xem số.

Gate phát triển đề nghị nếu dùng control này: validation chrF++ tăng ít nhất0.30 so với cả CE và fixed CLRR, BLEU không giảm; chỉ đủ mở kiểm chứng thêm seed, không phải statistical confirmation hoặc novelty GO. Chỉ một seed không cho kết luận ACL-ready. Nếu fail thì dừng control, không sweep gates/layers/lambda.

## 4. Bước được đề nghị tiếp theo

Tách hai mục tiêu: có thể kiểm tra learned-strength như một đối chứng cơ chế đã biết, hoặc tiếp tục thiết kế formulation khác có nội dung mới. **Chưa tự động mở hướng nào bằng training**. Nếu mục tiêu ưu tiên là một phương pháp mới cho ACL, nên giải quyết formulation/novelty trước; không có bằng chứng để khuyến nghị tốn thêm GPU cho MRT continuation hoặc một generic gate như đóng góp mới.

## Artifacts và tái lập

- `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/initialization_audit/REPORT.md`
- Cùng thư mục: `comparison.json`, `provenance.json`, năm file R1 nhỏ đã tải.
- `scripts/analysis/audit_sequence_initialization.py`
- Chạy CPU: `conda run -n clrr python scripts/analysis/audit_sequence_initialization.py --root outputs/experiments/amis_sequence_risk_diverse_pool_rtx`.
- Predictions/metrics gốc, checkpoint, baseline tables và test numbers được giữ nguyên.
