# Source-grounding: frozen-checkpoint validation probe

Ngày: 17/10/2026. Trạng thái: **COMPLETE / independent replay PASS3/3**.
Không train; 0 optimizer steps, không đánh giá test.
Phạm vi: Amis validation576; baseline R1, CLRR-only P1, Full cũ R2,
seed42, cùng frozen selection từ `amis_residual_effect_rtx_test`.
Không thay số trong file experiments hiện hành, không đánh giá test.

## Protocol khóa trước kết quả

- Worker chung: `scripts/analysis/measure_source_grounding.py`.
- Remote output: `outputs/experiments/source_grounding_validation/`.
- Bốn cells: CLRR ON/OFF × source matched/mismatched; giữ nguyên target.
- ON là route gốc distance2/strength0.1, detached post-rewiring donor.
  ON ở baseline là can thiệp chèn route vào model không train cùng route;
  baseline native inference là OFF. CLRR-only/Full native inference là ON.
- Toàn bộ576 validation rows, đúng CSV/tokenization/language proxy đã khóa.
  Không sửa chuỗi, thứ tự, split hoặc lọc dataset.
- Negatives cố định theo các block4 câu liên tiếp: source cyclic đầu tiên
  khác source và target về token sau masking. Cùng mapping cho cả ba models;
  tokenization và mapping phải khớp. Invalid negatives vẫn giữ matched CE,
  nhưng không tham gia thống kê source-gap. Distinct tokens không bảo đảm
  distinct semantics; không khẳng định negatives đều không có quan hệ nghĩa.
- FP32, TF32/dropout/checkpointing OFF; teacher forcing. Nats/token trên
  non-pad supervised target labels, gồm special labels giống audit cũ.
- Strict archive SHA256 và strict weights loading; runtime và route/helper
  source hashes đối chiếu với frozen workspace. Không dùng training wrapper
  của candidate mới; chỉ dùng helper tạo negative pairs chung.
- Đối chiếu matched CE ON/OFF từng câu với residual-utility audit trước đó
  (tolerance1e-4 để cho phép khác batch/padding GEMM). Repeat/OFF-vs-bare
  logits phải khớp chính xác. Parameter versions giữ nguyên, gradients absent.
- CI mô tả: 10.000 bootstrap draws, seed42, resample nguyên block negative4
  để giữ phụ thuộc trong cặp; không phải 10.000 câu hoặc 10.000 lần train.
  Không tính confirmatory p-value hoặc gọi là bằng chứng qua nhiều seeds.
- Tải một checkpoint/lần từ HF đã pin; sau lưu kết quả hợp lệ, xóa weights
  tải tạm. Chỉ thu JSON/CSV/log nhỏ về Windows.

## Đại lượng và cách đọc

Đặt `s=−CE`, `G_on=s_on(x,y)−s_on(x−,y)` và tương tự `G_off`.

| Đại lượng | Công thức theo CE | Giá trị dương nghĩa là |
|---|---|---|
| G_on | CE(ON,mismatch)−CE(ON,match) | ON phân biệt source đúng tốt hơn |
| G_off | CE(OFF,mismatch)−CE(OFF,match) | OFF phân biệt source đúng tốt hơn |
| Route gain matched | CE(OFF,match)−CE(ON,match) | Route giúp dự đoán target khi dùng source đúng |
| Route gain mismatched | CE(OFF,mismatch)−CE(ON,mismatch) | Route cũng giúp dự đoán target dù source không khớp |
| Interaction I | G_on−G_off = gain_matched−gain_mismatched | Route gain lớn hơn ở source đúng |

Kết quả I thấp/âm có thể hỗ trợ việc nghiên cứu source-specific supervision;
không chứng minh decoder bỏ qua source hoặc objective mới tăng BLEU.
I dương rõ cũng không tự động ủng hộ maximize I: phải hỏi vấn đề còn thiếu
có thực sự là grounding và liệu gain chỉ có ở branch OFF bị làm yếu hay không.
Source-gap là diagnostic phụ thuộc reference/teacher forcing, không phải
thước đo hoàn chỉnh adequacy, hallucination hoặc chất lượng generation.

## Quyết định sau phép đo

Chỉ phân tích và đề xuất GO/HOLD/NO-GO cho pilot ba controls/candidate đã
thiết kế trong [design/preflight](GROUNDED_INTERACTION_DESIGN_AND_PREFLIGHT.md).
Không tự mở training. Nếu dữ liệu không ủng hộ cơ chế, giữ kết quả âm và
không dùng novelty/CE surrogate thay cho bằng chứng dịch.

## Kết quả đã xác minh

576/576 negatives hợp lệ trong mỗi model,144 blocks; cùng tokenization và
mapping cả ba checkpoints. Tổng3×576×4=6.912 sentence-condition CE values.
Đơn vị tất cả số dưới đây là nats/supervised target token; CE thấp hơn là tốt.

| Checkpoint | CE ON đúng source | CE OFF đúng source | CE ON source không khớp | CE OFF source không khớp |
|---|---:|---:|---:|---:|
| Baseline R1 | 2.894823 | 2.804171 | 5.966641 | 6.285184 |
| CLRR-only P1 | 2.869134 | 3.071728 | 6.291127 | 6.071424 |
| Full cũ R2 | 2.763204 | 3.003039 | 6.139774 | 5.971119 |

| Checkpoint | Route gain: đúng source | Route gain: source không khớp | Interaction I | Descriptive block CI95% của I |
|---|---:|---:|---:|---|
| Baseline R1 | −0.090652 | +0.318543 | −0.409195 | [−0.457494, −0.361542] |
| CLRR-only P1 | +0.202594 | −0.219703 | +0.422297 | [+0.369838, +0.474998] |
| Full cũ R2 | +0.239835 | −0.168655 | +0.408490 | [+0.361283, +0.455432] |

**Điểm có lợi cho cơ chế CLRR:** ở hai checkpoints đã train với route, bật
route vừa giúp dự đoán target với source đúng, vừa làm target kém probable
hơn khi source không khớp. Không phù hợp với giải thích đơn giản rằng route
chỉ tăng target likelihood bất kể source. Đây là can thiệp teacher-forcing
ở fixed weights, không phải xác nhận translation gain qua nhiều seeds.

**Baseline ON không phải baseline native.** Việc chèn route hậu huấn luyện
vào baseline làm kết quả xấu đi, phù hợp với co-adaptation đã đo trước đây;
không dùng hàng ON này làm baseline chính để tuyên bố thắng.

### So với baseline khi mỗi model dùng cấu hình inference gốc

| Model, native route | CE source đúng ↓ | Native source-gap ↑ | Δ CE so với baseline | Δ source-gap so với baseline |
|---|---:|---:|---:|---:|
| Baseline, OFF | 2.804171 | 3.481013 | — | — |
| CLRR-only, ON | 2.869134 | 3.421993 | +0.064963 | −0.059020 |
| Full cũ, ON | 2.763204 | 3.376570 | −0.040966 | −0.104443 |

Paired block CI95% mô tả:

- CLRR-only−baseline CE: [+0.034327,+0.095526]; gap:
  [−0.108568,−0.009050].
- Full−baseline CE: [−0.073235,−0.009155]; gap:
  [−0.153822,−0.054948].

Full có matched CE thấp hơn baseline trong probe này nhưng native source-gap
không cao hơn. Khoảng gap nhạy calibration của model và cách chọn negative;
không được đồng nhất gap thấp với hallucination nhiều, không dùng CE/gap
thay BLEU/chrF hoặc khẳng định chất lượng dịch có ý nghĩa qua seeds.
CIs chỉ mô tả resampling các blocks của validation với fixed checkpoints và
một mapping negative, không là fresh-seed confirmation hay holdout mới.

## Kết luận cho candidate mới

**HOLD pilot `route-source-interaction-v1` ở hiện trạng.**

1. Không xác nhận giả thuyết đơn giản “CLRR không dùng đúng source”: native
   gaps đều dương lớn; route đã tăng source discrimination trong P1/R2.
2. Tăng I so với OFF cùng weights chưa bảo đảm hơn baseline thật. Nhánh OFF
   trong route-trained model không phải một baseline đã train độc lập.
3. Loss mới có gradient có thể làm OFF yếu đi để tăng I; kết quả hiện có
   làm controls này quan trọng hơn, không giải quyết nguy cơ đó.
4. Chưa xác định nguồn bottleneck của translation bằng gap này. Không train
   ba runs chỉ vì prototype đã qua correctness gates.

**Không loại CLRR khỏi mọi hướng nâng cấp:** probe có lợi cho vai trò của
route trong checkpoint đã train. Nhưng chưa có bằng chứng route/source
interaction là mục tiêu còn thiếu. Bước hợp lý tiếp theo là phân tích lỗi
generation validation của baseline/CLRR-only/Full: câu nào gain/loss, độ dài,
thiếu/dư nội dung, rare lexical units và liên hệ với CE/gap đã đo. Tận dụng
predictions sẵn có trước; kiểm tra nguồn hypotheses trước khi chạy inference
mới. Probe này không mở training. Phân tích generation đã hoàn tất ở lượt
tiếp theo: [GENERATION_VALIDATION_DIAGNOSIS.md](GENERATION_VALIDATION_DIAGNOSIS.md).

## Correctness, sửa phép đo và lưu trữ

- `outputs/experiments/source_grounding_validation/LOCAL_VERIFICATION.json`:
  independent CSV arithmetic, NLL/token normalization, CI và hashes PASS3/3;
  `analyze_source_grounding.py` có args để replay.
- Từng checkpoint strict load, archive hash, model parameter versions và
  no-gradient checks PASS. Repeat OFF và OFF-vs-bare max logits error0.
- Matched CE đối chiếu audit cũ: max per-sentence error4.91143e-5, dưới
  tolerance1e-4 đã khóa trước khi nhìn kết quả. Không sửa số thực nghiệm cũ.
- Lượt đầu bị chặn ở CE reduction gate. Dạng3D và HF dạng phẳng dùng kernel
  floating-point khác nhau; batch đầu lệch tối đa2.52724e-5 nats/token.
  Worker chuyển về flat CE như Hugging Face/audit cũ, **không nới ngưỡng gate**;
  flat-vs-HF max error batch kiểm tra là1.43052e-6. Chưa có kết quả nào được
  ghi nhận ở attempt lỗi; plan/source/log attempt đó được giữ trên RTX.
- Tokenizer phát cảnh báo Mistral regex; checkpoint metadata là
  `MBart50Tokenizer`, backend Unigram/Metaspace. Không áp dụng flag sửa regex
  ngoài protocol: tokenization hashes khớp cả ba model và CE replay đã PASS.
  Không tự coi đây là bằng chứng cần sửa dataset/tokenizer cũ.
- Các engine files giữa local và remote có lịch sử cập nhật khác nhau; exact
  remote source được lưu. Riêng `configure_languages` dùng trong probe được
  xác minh cùng AST bằng một Python parser, sau kiểm tra file hash. Không so
  sánh trực tiếp AST dump hashes giữa hai Python versions.
- Chỉ thu JSON/CSV/log/source nhỏ, khoảng0.55MB ban đầu, không weights/NPZ.
  Ba archives/exports tải tạm trên RTX đã xóa sau khi lưu kết quả hợp lệ.
  GPU sau phép đo:666MiB used,23.542MiB free,0% utilization; không tắt máy
  hoặc tiến trình người khác.

Public backup đã upload và verify28files,599.705bytes tại
[HF source-grounding](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/4b31dbadb5be004d69e1ab282abadf8e08c2b70d/analysis/amis_mandarin/mbart-large-50/source-grounding/seed-42/rtx6000-fp32).
Revision dữ liệu: `4b31dbadb5be004d69e1ab282abadf8e08c2b70d`.
Receipt local: `outputs/experiments/source_grounding_validation/HF_UPLOAD.json`;
so khớp size và Git blob/LFS hashes của từng artifact. Chỉ khoảng0.60MB,
không checkpoint weights hoặc raw representation vectors.
