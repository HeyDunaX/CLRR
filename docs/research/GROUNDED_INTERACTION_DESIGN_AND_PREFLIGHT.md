# Nâng cấp sau residual-utility audit: thiết kế và kiểm thử

Ngày: 17/10/2026. Phạm vi được duyệt: bước 1–3 — bằng chứng, tối đa hai
thiết kế, prototype và correctness gates. **Chưa mở pilot training.**
Tài liệu này ghi đề xuất mới; không thay số trong `EXPERIMENTS_UPDATED.md`.

Cập nhật sau probe: [source-grounding report](SOURCE_GROUNDING_VALIDATION_REPORT.md)
đã COMPLETE, independent replay3/3 PASS. Route trong CLRR-only/Full đã tăng
source discrimination; maximize interaction chưa được xác nhận là bottleneck
còn thiếu. **HOLD pilot candidate này**; không tự mở ba runs ở mục7.

## 1. Vấn đề thực nghiệm đã xác nhận

| Bằng chứng hiện có | Kết luận dùng được | Điều chưa chứng minh |
|---|---|---|
| Amis fresh confirmation 8 seeds: Full−LSR −0.019809 BLEU, Full−baseline −0.041468 | Recipe Full cũ chưa có lợi ích ổn định | Không suy ra mọi thiết kế CLRR hoặc LSR đều vô ích |
| NCE cải thiện alignment/retrieval nhưng translation chưa tăng ổn định; Ê Đê Full−baseline +0.1115, P3−baseline −0.5437 BLEU ở seed42 | Alignment là surrogate, chưa đủ làm mục tiêu thiết kế | Chưa xác định decoder bỏ qua source hay mất morphology |
| Residual utility: route ON giúp khoảng 69–72% câu ở weights đã train với route | Có co-adaptation và heterogeneity ở fixed checkpoints | Oracle theo reference không phải learned gate có thể dùng khi inference |
| Cả 10 layer-removal means đều làm CE tăng ở ba route-trained checkpoints | Chưa có căn cứ bỏ hàng loạt tầng đầu/cuối | Không chứng minh topology hiện tại tối ưu |

Nguồn: [residual utility](RESIDUAL_UTILITY_ROUTING_AUDIT.md),
[insights](RESEARCH_INSIGHTS.md), [P3 review](P3_NOVELTY_AND_MECHANISM_REVIEW.md).
Không chạy lại test để chọn thiết kế này.

**Vấn đề chốt:** lợi ích trên representation chưa chuyển thành lợi ích dịch
ổn định. Hướng thử là đưa supervision về xác suất bản dịch và sự phụ thuộc
vào source. Đây là giả thuyết thiết kế, chưa phải chẩn đoán source-grounding
đã được xác nhận. Các phép đo CE cũ chỉ dùng đúng source.

## 2. Hai hướng đã đối chiếu

Ký hiệu `s_a(x,y)` là mean log-probability trên các target token có label,
teacher forcing; `a=1/0` là CLRR ON/OFF với cùng weights. `x−` là một source
khác trong microbatch; target vẫn là `y`. Đặt:

\[
G_a=s_a(x,y)-s_a(x^-,y),\qquad I=G_1-G_0.
\]

| Candidate | Công thức auxiliary | Gradient | Quyết định |
|---|---|---|---|
| A: reference OFF detached | `softplus(−[G1−sg(G0)]/T)` | Chỉ qua G1 | Loại khỏi vị trí đóng góp mới chính; với từng câu chỉ đổi hệ số nhân gradient source-contrastive |
| B: differentiable route/source interaction | `softplus(−[G1−G0]/T)` | Qua cả bốn cells | Chọn làm prototype có controls để kiểm tra; chưa chốt là phương pháp paper |

Với A: `∇L = −sigmoid(−I/T) ∇G1 / T`. Margin OFF làm thay đổi trọng số
theo mẫu, không tạo thêm hướng gradient OFF. Không được quảng bá việc thêm
reference detached là cơ chế hoàn toàn mới.

Với B: `∇L = −sigmoid(−I/T)(∇G1−∇G0)/T`. Đây là khác biệt backward
cụ thể so với A và ON-only source contrast. Nó không tự bảo đảm novelty
nghiên cứu hoặc chất lượng dịch.

## 3. Thiết kế B được code

Giữ CLRR gốc: encoder, distance2, strength0.1, detached post-rewiring donor.
Đổi objective; không thêm LSR, predictor, EMA, gate hay tham số mới.
Không sửa implementation hoặc ID của các phương pháp cũ.

\[
L_{pair}=\tfrac12 CE^{dropout}_{ON}(x,y)+\tfrac12 CE^{dropout}_{OFF}(x,y),
\]
\[
L=L_{pair}+\beta\,\operatorname{mean}_{valid\ pairs}
\operatorname{softplus}(-I/T).
\]

- `β=.1`, `T=1` là giá trị prototype được chọn trước correctness gate,
  không được suy từ một phép tuning hoặc coi là optimum.
- Hai CE views giữ dropout train; CE giảm theo tổng supervised tokens trong
  microbatch như baseline. Hai lượt stochastic độc lập.
- Bốn auxiliary views tắt dropout toàn encoder–decoder, dùng mean score theo
  câu rồi mean theo valid pairs. Cả ON/OFF, matched/mismatched đều có gradient.
  Đây là surrogate backward do donor của CLRR vẫn detach; không nhận là
  full derivative qua mọi đường donor.
- Negative: source đầu tiên hợp lệ theo vòng cyclic trong microbatch; loại
  source hoặc target token-identical sau masking. Có thể dùng lại một negative.
  Không phát hiện được mọi câu đồng nghĩa. Không sửa CSV hoặc split.
- Singleton/all-duplicate microbatch: auxiliary bằng 0, vẫn train paired CE.
  GA32 không biến microbatch4 thành negative pool128.
- Inference/eval chỉ route ON, không cần labels hoặc negative. Eval loss là
  CE ON thông thường; checkpoint selection bằng generation metrics của recipe,
  không bằng auxiliary train loss.

### Rủi ro phải kiểm tra trước khi train dài

Tăng `I` có thể đến từ tăng `G1` **hoặc giảm `G0`**. Dấu gradient auxiliary
đẩy `s_OFF(x,y)` xuống và `s_OFF(x−,y)` lên; paired CE chỉ tạo lực đối trọng,
**không có bảo đảm toán học chặn nhánh OFF bị làm yếu đi**. Vì vậy không
được dùng interaction tăng làm bằng chứng duy nhất cho source grounding.

Phải đo riêng bốn cells, ON/OFF matched CE, cả `G1`, `G0`, và generation.
Nếu tăng interaction chủ yếu do OFF suy giảm, hoặc không hơn source-only
control, loại claim cơ chế và không mở rộng multiseed. Không thay công thức
âm thầm sau khi nhìn test.

Chi phí: candidate6 full forward views/microbatch (2 CE + 4 auxiliary),
kèm recomputation khi bật checkpointing. Zero added parameters không có
nghĩa là rẻ ngang baseline. Chưa đo runtime train/Adam/validation thực tế.

## 4. Novelty: nguồn sơ cấp và giới hạn claim

| Prior đã đọc | Phần đã có | Khác biệt có thể kiểm chứng của B |
|---|---|---|
| [Sennrich et al., EACL2024, §2 Eq1](https://arxiv.org/pdf/2309.07098), [code chính thức](https://github.com/ZurichNLP/ContraDecode) | Đúng source so với random source tại decoding; Eq1 trừ probabilities trước log | B là training surrogate có route/source factorial contrast và gradient qua OFF; không nhận source contrast là mới |
| [CBMI, ACL2022, §3.1 Eq7 và §3.2](https://aclanthology.org/2022.acl-long.169.pdf) | Log NMT/LM ratio; token/sentence loss reweighting theo source dependence | B không có external target LM; so sánh source-match gaps giữa hai routing views |
| [CPO, §3.1](https://arxiv.org/html/2401.08417v2) | Logistic preference objective + NLL; preferred/rejected translations | B giữ target và đổi source/routing; OFF cùng weights nhận gradient, không là frozen reference. Logistic loss tự nó không mới |
| [SkipCLM, §3.1](https://aclanthology.org/2025.naacl-srw.50.pdf) | Cross-layer residual và bilingual contrastive representation learning | B dùng conditional target scores; không nhận ghép skip với contrastive loss là mới |
| [Lexical Shortcuts, §2.2](https://aclanthology.org/W19-5211.pdf) và [GateSkip, §3](https://arxiv.org/html/2510.13876v2) | Gated paths/conditional computation | Prototype B không học gate hoặc chọn depth |

Đã đọc phương pháp Source-Contrastive Decoding từ PDF arXiv khi endpoint
ACL PDF lỗi; CBMI từ PDF, CPO từ full HTML. Code ContraDecode được đối chiếu
README; chưa tái lập thực nghiệm prior. Các prior về gate/SkipCLM cũng đã
được đối chiếu trong residual-utility audit trước đó.

**Kết luận novelty:** B có khác biệt formulation/backward cụ thể trong nhóm
prior đã kiểm tra, đủ để làm prototype falsifiable. Chưa có cơ sở chứng nhận
không tồn tại prior trùng, hoặc novelty đã đủ cho ACL. Đặc biệt bốn-cell
contrast không phải một định lý causal identification mới. Giữ **HOLD** với
claim đóng góp chính và pilot dài cho tới khi kiểm tra được giả thuyết cùng
controls; nếu generic source contrast giải thích hết gain thì không giữ claim B.

## 5. Code riêng và tái sử dụng

`src/methods/grounded_interaction/v1/model.py`, qua registry/shared runner:

| Method ID | Loss | Vai trò |
|---|---|---|
| `route-paired-ce-v1` | Paired CE | Kiểm soát hai positive routing views |
| `route-source-contrast-v1` | Paired CE + `β softplus(−G1/T)` | Source-contrast adapted control |
| `route-source-interaction-v1` | Paired CE + `β softplus(−I/T)` | Candidate B |

Shared CLI có `--interaction-weight`, `--temperature`; config phải khai báo
`interaction_weight`, `jepa_weight=0` và
`protocol.view_checkpointing="whole-view-non-reentrant"`.
Lưu thêm `interaction_config.json`; shared loader so khớp chính xác với recipe.
**Không tạo recipe train hoặc queue trong lượt này.**

Checkpointing mới tái tính cả view thay vì block reentrant: mỗi view tự đặt
route/dropout/cache khi forward và recomputation. Đây là thay đổi memory
implementation được khai báo; loss/gradient được đối chiếu với không checkpoint.
Nếu pilot dùng nó, cả ba controls phải dùng cùng cách. Không sửa số/bundle cũ
để giả vờ chúng đã chạy cách checkpointing mới.

## 6. Correctness gates và artifacts

Artifacts nhỏ: `outputs/experiments/grounded_interaction_preflight/`.
GPU chạy trong snapshot riêng ở
`/workspace/ura/thanhdat/projects/CLRR/tmp/grounded_interaction_gate/`;
conda/cache/temp nằm trong project CLRR. Không tải checkpoint từ HF về Windows.

- `CPU_GATE.json`: CPU correctness và shared runner guards.
- `GPU_UNIT_GATE.json`: CUDA correctness suite.
- `TINY_GPU_GATE.json`: microbatch4, length16, GA32, FP32.
- `FULL_GPU_GATE.json`: mBART-large-50 geometry610,879,488 parameters,
  microbatch4, source/target length256, GA2, FP32.
- `GPU_SOURCE_MANIFEST.json`, `gate.log`: source hashes/log.

Kết quả cuối (đã đối chiếu SHA256 CPU/GPU với source hiện hành):

| Gate | Kết quả | Chi tiết |
|---|---|---|
| CPU correctness | PASS, 9/9 tests | Có shared ControlTrainer, metadata và config guards |
| CUDA correctness | PASS, 9/9 tests | Cùng test source SHA256 với CPU; `unit_final.log` |
| Tiny CUDA FP32 | PASS | Batch4, length16, GA32; 18.03 MiB peak allocated |
| Full geometry CUDA FP32 | PASS | Batch4, source/target256, GA2; 12,425.03 MiB allocated, 13,324 MiB reserved; 24.77s gồm khởi tạo |
| Existing contrastive controls regression | PASS, 3/3 tests | Các định nghĩa cũ không đổi |
| Final verification | PASS | Hash source trong CPU/GPU receipts khớp code hiện hành |

Trước GPU gate: 23,542 MiB trống trên Quadro RTX6000 24GiB. Sau gate,
GPU trở về 666 MiB used / 23,542 MiB free; không dừng tiến trình người khác.
Local receipts/logs khoảng27KB, không chứa weights hoặc representations.
`FINAL_VERIFICATION.json` xác nhận zero optimizer steps và chưa mở pilot.

Các gates kiểm tra: công thức/dấu gradient bốn cells; gradient khác ON-only;
mask negatives/singleton; zero-route auxiliary constant và gradient triệt tiêu;
checkpoint ON/OFF loss/gradient/RNG; restoration khi lỗi; accumulation4;
Trainer loss scaling một lần; save/load/Trainer best-load và generation;
zero extra parameters; config và metadata guards.

FULL gate chưa cấp phát Adam state, chưa dùng pretrained/trained weights hoặc
data thật. Nó chỉ xác minh geometry/FP32/backward/VRAM, không thay thế gate
hai optimizer steps + optimizer/RNG resume trên frozen pilot bundle sau này.
GA2 full size kiểm tra giữ gradient qua nhiều microbatch; GA32 được chạy ở
tiny size. Không được ghi là full-size GA32 đã chạy.

Hai lỗi trong vòng mở rộng CPU tests được sửa trước khi chốt receipt: kiểm
tra target rỗng phải đứng trước mBART shift-label; config test phải nằm trong
`configs/experiments/` đúng guard của shared runner. Không liên quan số cũ.

## 7. Quyết định sau bước 1–3

- Code và technical gates: xem receipt cuối; không suy ra quality GO.
- Main-method novelty/empirical GO: **chưa chốt**. Không mở training tự động.
- Việc không train tiếp theo: frozen validation source-mismatch probe với
  checkpoint baseline/CLRR-only/Full đã có, cùng pairs/cells, strict source và
  archive hashes. Đo xem routing thay đổi source-match gap ra sao; công bố
  cả bốn cells và negative-pair exclusions. Không dùng nó để hứa hẹn causal
  adequacy hoặc coi là fresh holdout.
- Chỉ nếu probe hỗ trợ giả thuyết và chấp nhận chi phí mới đề xuất pilot
  seed42 ba nhánh ở mục5. Cùng init/order/LR/batch/epochs20/ES10, negatives,
  runtime, validation selection và checkpointing. Baseline/CLRR cũ làm mốc
  tham khảo; paired-CE và source-only controls mới là phép tách contribution.
- Chưa ấn định gain/claim, chưa mở sweep, chưa mở seed khác hoặc A100.

Diễn giải và thiết kế có hỗ trợ AI; các số correctness được lấy từ receipts,
không phải kết quả dịch của phương pháp mới.
