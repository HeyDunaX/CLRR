# Gold-token decisions và validation beam sensitivity

Ngày 17/10/2026. Trạng thái: **COMPLETE / independent replay PASS3/3**,
không train.
Theo sau [generation validation diagnosis](GENERATION_VALIDATION_DIAGNOSIS.md).

## Thiết kế khóa trước kết quả

- Ba checkpoints R1 baseline, P1 CLRR-only, R2 Full; dùng đúng archive/revision
  đã khóa trong `amis_residual_effect_rtx_test`. Không chọn lại checkpoint.
- Toàn bộ 576 validation rows, original CSV, target/source proxies và max
  lengths như các lượt trước; không chạy test. Runtime cũ, strict weights
  loading, route/helper source hashes đối chiếu bundle manifest cũ.
- Baseline native OFF; CLRR-only/Full native ON, distance 2, strength .1,
  detached post-rewiring donor. Không auxiliary wrapper lúc đánh giá.
- FP32, TF32/dropout/checkpointing OFF, không optimizer; RTX shared phải đủ
  16.000 MiB free và lấy được lock CLRR trước khi chạy. Trước queue có
  23.542 MiB free; không dừng tiến trình của người khác.

### Teacher-forced token decisions

Giữ gold prefix; CE dạng phẳng như Hugging Face và audit trước.
Mỗi target token có:

- Top1 đúng: argmax raw vocabulary logits bằng gold token.
- Strict rank = 1 + số logits lớn hơn gold; ties có thể rank1 nhưng top1 sai.
- Margin = gold logit − largest non-gold logit; là log odds với competitor.
- NLL và EOS probability; premature-EOS-top1 là EOS argmax ở một gold
  content position. Đây không phải actual free-generation early-stop count.

Loại toàn bộ tokenizer special IDs khỏi content metrics; EOS và other specials
báo riêng. Token tần suất thấp tính bằng **train target tokenizer IDs**, sau
cùng truncation, không phải source whitespace rarity đã đo ở lượt trước.
Các bins cố định: unseen0, 1–5, 6–50, >50 train occurrences.
Thống kê theo token; phân tích paired phải ghép cùng row/position và bootstrap
theo câu nếu cần CI, không coi mỗi token là một independent training seed.
Raw argmax không bao gồm generation logits processors; specials vẫn cạnh
tranh trong raw logits. Không suy ra quality chỉ từ margin hoặc mean NLL.

### Beam sensitivity

- Dùng lại beam1 predictions đã verified, không tái generate toàn validation.
- Gate tái tạo beam1 ở 8 rows đầu, batch2, exact decoded strings phải khớp;
  CE từng câu toàn validation khớp probe trước với tolerance1e-4.
- Generate beam4 cho cả ba models: batch2, do_sample=False, max_length256,
  cùng loaded generation configuration và language setup. Lưu đầy đủ config
  để phát hiện khác biệt length penalty, early stopping, cache hoặc processors.
- BLEU zh, chrF++, chrF0 giữ nguyên protocol. Đo thay đổi beam trong từng
  model và chênh lệch giữa models tại **cùng beam**; không dùng beam4 của
  candidate so beam1 của baseline làm claim.
- Bằng chứng fixed checkpoints/validation đã dùng cho selection; không phải
  fresh confirmation. Không tự thay beam trong main results hoặc mở training.

## Correctness và lưu trữ

Worker: `scripts/analysis/diagnose_token_decoding.py`, có args workspace,
grounding, output, IDs, token batch, generation batch, beams và greedy gate.
Hai CPU gates về rank/margin/ties/ignore-mask/softmax và frequency/special
categories đều PASS. GPU CE/greedy gates chạy trước khi chấp nhận kết quả.

Remote output: `outputs/experiments/token_decoding_validation/`.
Tải/load một checkpoint một lần để đo cả hai việc. Sau lưu kết quả hợp lệ,
xóa archive/export tải tạm. Không tải weights về Windows. Per-token CSV chỉ
chứa row/position, category, frequency và scalar metrics, không token IDs hoặc
hidden vectors. Beam4 predictions là text artifacts riêng; public upload sẽ
dùng allowlist diagnostics dẫn xuất, không gộp raw prediction snapshots.

## Diễn giải sẽ dùng

1. Nếu CE tốt hơn nhưng content top1/margin ở các quyết định khó không tốt:
   kiểm tra objective có đang cải thiện probability mà chưa sửa decision.
2. Nếu teacher-forced decisions không kém nhưng beam1 kém: beam4 và prefix
   sensitivity có thể giúp phân biệt search với các giải thích khác.
3. Nếu beam4 không sửa chênh lệch: không lấy decoding làm lời giải duy nhất.
4. Không mặc định gold-token accuracy, rare-token loss hoặc beam tuning là
   novelty. Có bằng chứng nguyên nhân rồi mới thiết kế và audit method mới.

## Kết quả teacher forcing đã xác minh

Mỗi model có 6.653 supervised tokens: 5.494 non-special content tokens,
576 EOS và 583 other specials. Content ở đây chỉ loại special IDs, vẫn có
punctuation/subwords; không coi tất cả là từ mang nghĩa. Cùng row/position,
frequency và tokenizer hashes ở cả ba models.

| Model | Content top1 (%) ↑ | Strict top5 (%) ↑ | Content NLL ↓ | Mean gold-vs-best-other margin ↑ | EOS top1 (%) |
|---|---:|---:|---:|---:|---:|
| Baseline | 50.4550 | 68.9115 | 3.474846 | −0.286092 | 98.0903 |
| CLRR-only | 49.3629 | 68.4383 | 3.555492 | −0.424661 | 98.0903 |
| Full | 49.6360 | 68.6203 | 3.427548 | −0.442526 | 98.0903 |

Content NLL/margin ở đây là pooled mean trên 5.494 non-special tokens.
Không so trực tiếp mức NLL3.4 này với sentence-mean CE2.8 của probe trước,
vốn tính cả special labels và có reduction khác.

Full có NLL tốt hơn nhưng chưa có top1/margin gain so với baseline. Paired
sentence bootstrap 10.000 draws, seed42, token accuracy weighted theo tổng
content count trong các câu được resample:

- CLRR-only−baseline: −1.092100 percentage points, descriptive CI95%
  [−1.851524,−0.331159].
- Full−baseline: −0.819075 percentage points, descriptive CI95%
  [−1.662052,+0.018216]. **CI Full có chứa0**; không gọi Full top1 kém hơn
  với significance confirmatory. Đây không phải uncertainty qua training seeds.

### Quyết định token và NLL benefit của Full−baseline

| Nhóm theo reference top1 | Tokens | Mean NLL benefit trong nhóm | Đóng góp vào mean content NLL benefit |
|---|---:|---:|---:|
| Cả hai đúng | 2.500 | −0.034053 | −0.015496 |
| Cả hai sai | 2.495 | +0.167641 | +0.076131 |
| Baseline sai → Full đúng | 227 | +1.967561 | +0.081295 |
| Baseline đúng → Full sai | 272 | −1.911443 | −0.094633 |
| Tổng | 5.494 | — | +0.047298 |

NLL benefit dương = baseline NLL−Full NLL. Bốn contributions cộng chính xác
ra mean content benefit. **Nhóm cả hai vẫn sai đóng góp +0.076131, lớn hơn
net benefit +0.047298**; các nhóm còn lại gộp làm giảm benefit. Vì vậy giảm
NLL ở checkpoint này chưa có nghĩa sửa thêm nhiều reference top1 decisions.
Đây là mô tả output của cùng gold prefixes, không xác nhận human adequacy
hoặc nguyên nhân training causal; reference alternatives cũng có thể hợp lệ.

### Target token frequency

| Train target occurrences | Validation tokens | Baseline top1 (%) | CLRR-only top1 (%) | Full top1 (%) |
|---|---:|---:|---:|---:|
| 0 | 171 | 4.0936 | 4.6784 | 3.5088 |
| 1–5 | 713 | 23.9832 | 22.7209 | 22.0196 |
| 6–50 | 1.853 | 39.8273 | 38.4781 | 39.1797 |
| >50 | 2.757 | 67.3196 | 66.3402 | 66.6667 |

Full ở nhóm1–5 sửa15 nhưng làm mất29 reference-correct decisions. Các bins
là mô tả một seed, không significance riêng cho mỗi bin. Không suy ra
morphology từ target BPE frequency hoặc source whitespace rarity.
EOS raw top1 ở gold content positions chỉ3/5.494 với baseline,3 với CLRR-only,
7 với Full; cùng EOS-boundary top1 98.09%. Chưa ủng hộ giải thích failure
hàng loạt do EOS/length limit. EOS errors ở free-running prefixes chưa được
đo riêng.

## Kết quả beam1/beam4: cùng beam giữa các models

| Model | BLEU beam1 | BLEU beam4 | Beam4−beam1 | chrF0 beam1 | chrF0 beam4 |
|---|---:|---:|---:|---:|---:|
| Baseline | 17.682329 | 18.174130 | +0.491801 | 16.954748 | 17.215860 |
| CLRR-only | 16.621620 | 17.467068 | +0.845448 | 16.223205 | 16.861942 |
| Full | 17.115572 | 17.779134 | +0.663562 | 16.480923 | 16.987409 |

- Full−baseline ở beam4: −0.394996 BLEU; descriptive paired sentence CI95%
  [−1.526070,+0.726329].
- CLRR-only−baseline ở beam4: −0.707062; CI [−1.922742,+0.530631].
- Các CI beam4−beam1 trong từng model cũng chứa0: baseline
  [−0.343208,+1.352256], CLRR-only [−0.060322,+1.878756], Full
  [−0.105258,+1.432620].

Không có confirmatory p-values hoặc tuyên bố beam4 cải thiện có ý nghĩa.
Point estimates tăng cho cả ba models nhưng thứ hạng giữ nguyên. Đổi beam
chưa đủ khôi phục lợi thế của Full trên validation này; cũng không chứng minh
mọi search/prefix effect đã bị loại. Không đổi checkpoint, main protocol hoặc
số test sau phép đo này.

## Quyết định sau kiểm chứng

**HOLD train source-interaction/global-alignment candidate hiện tại.**
Đã có bằng chứng cụ thể cho surrogate/decision mismatch ở checkpoint này:
Full tăng likelihood của gold tokens nhưng không vượt baseline về reference
top1, margin hoặc generation ở cả hai beams. Không tăng lambda hoặc dùng
rarity gate chỉ từ các bảng này.

Hướng thiết kế tiếp theo nên đặt câu hỏi: can thiệp residual có sửa gold-vs-
strongest-competitor decisions mà không làm mất những quyết định đang đúng
không? Cần đối chứng và novelty audit so với margin losses, distillation và
routing prior trước khi chọn formulation. Các kỹ thuật đó tự chúng đã có;
không coi việc ghép margin vào CLRR là đủ mới. Chưa code hoặc train một
candidate mới trong lượt kiểm chứng này; chưa kết luận LSR là causal culprit.

## Verification và artifacts

- CPU token formula/frequency gates:2/2 PASS.
- Ba checkpoints strict archive/model loading; runtime/routing hashes PASS.
- Greedy replay:8/8 decoded rows khớp ở mỗi model, tổng24/24.
- Tất cả native per-sentence CE khớp probe trước; max CSV replay error
  8.89e-16. No gradients/parameter updates,0 optimizer steps.
- `ANALYSIS.json`: độc lập tái tính tất cả token summaries,6sets generation
  scores, NLL decomposition và paired CIs theo câu. Loaded generation configs
  khớp ngoài các overrides/meta đã khai báo; không đổi length penalty hoặc
  decoding processor giữa models.
- Không beam4 output nào chạm max_length256. Weights tải tạm đã xóa; GPU
  sau queue666MiB used /23.542MiB free /0% utilization.
- Source/CSV/JSON/log được thu về local khoảng2.36MB; không checkpoint/NPZ.
  Public payload dùng allowlist số liệu dẫn xuất/source, không raw beam4
  predictions. Những raw predictions hiện vẫn có trong workspace local/RTX.

Public backup đã upload/verify 26 files, 2.199.980 bytes tại
[HF token-decoding](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/4e66c5fb50cdc3028bf23c7faad08419f9c16ef0/analysis/amis_mandarin/mbart-large-50/token-decoding/seed-42/beam1-beam4-fp32).
Receipt `outputs/experiments/token_decoding_validation/HF_UPLOAD.json`.
Không upload raw predictions hoặc token IDs; per-token public CSV chỉ là
row/position, frequency category và scalar decisions/NLL/margin.

CPU replay:

```bash
conda run -n clrr python scripts/analysis/analyze_token_decoding.py \
  --input outputs/experiments/token_decoding_validation \
  --beam1-inputs outputs/experiments/amis_generation_validation_analysis/inputs \
  --references data_processed/amis_mandarin/validation.csv \
  --draws 10000 --seed 42
```
