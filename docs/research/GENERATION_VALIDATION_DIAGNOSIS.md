# Amis: generation validation, CE và source grounding

Ngày 17/10/2026. **COMPLETE / direct replay PASS.** Phân tích CPU trong
conda `clrr`; không GPU, weights, generation mới, training, test hoặc chọn
lại checkpoint. Diễn giải có hỗ trợ AI; các mô tả lỗi là reference-relative,
không phải đánh giá adequacy bởi người thông thạo Amis.

## 1. Provenance và protocol

- 576 validation rows, SHA256 `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4`.
- R1 baseline/R2 Full: cohort `amis_token_rtx`; P1 CLRR-only:
  `amis_residual_effect_rtx`. Cùng beam 1, FP32, reference CSV gốc,
  BLEU tokenizer `zh`, sacreBLEU 2.6.0.
- Chín files predictions/metrics/metadata so hash với public HF revision
  `e04195904361dcbdf36d4ae6c7f577ca37eb3256`. Archive hashes khớp đúng
  selection lock của source-grounding probe.
- Source/target và thứ tự từng row khớp CSV gốc. Không sửa dataset; chỉ
  normalization tạm trong diagnostic frequency/overlap.
- Join native CE/gap: baseline OFF, CLRR-only/Full ON. Source gap là
  teacher-forcing diagnostic, không phải hallucination metric hoặc causal MI.
- `DESIGN.json` lưu trước phân tích outputs, nhưng đây vẫn là post-hoc
  development analysis sau khi đã xem kết quả tổng thể. Không gọi là
  preregistered confirmation hoặc fresh holdout. Không p-value confirmatory.

## 2. Số corpus tái tính, không thay số cũ

| Model | Validation BLEU | chrF++ | chrF0 |
|---|---:|---:|---:|
| Baseline R1 | 17.682329 | 14.703005 | 16.954748 |
| CLRR-only P1 | 16.621620 | 14.125185 | 16.223205 |
| Full R2 | 17.115572 | 14.345025 | 16.480923 |

Full−baseline: −0.566757 BLEU / −0.473825 chrF0.
Full−CLRR-only: +0.493952 BLEU / +0.257718 chrF0.
Một seed, checkpoints đã chọn bằng validation chrF++. Không suy ra significance
qua seeds hoặc đóng góp riêng của LSR từ ba model này.

## 3. Output không hỏng hàng loạt, không do một row đơn lẻ

| Chỉ báo tự động | Baseline | CLRR-only | Full |
|---|---:|---:|---:|
| Output rỗng | 0 | 0 | 0 |
| Character trigram lặp ít nhất 3 lần | 0 | 0 | 0 |
| Exact reference | 7 | 9 | 9 |
| Mean nonspace length ratio | 1.0115 | 0.9842 | 0.9869 |
| Median length ratio | 1 | 1 | 1 |
| Length ratio < 0.5 | 6 | 8 | 10 |
| Length ratio > 1.5 | 19 | 18 | 18 |
| Mean character-unigram precision | 0.5105 | 0.5043 | 0.5048 |
| Mean character-unigram recall | 0.4951 | 0.4795 | 0.4805 |

Full có character overlap recall giảm nhiều hơn precision. Đây là dấu hiệu
reference-relative coverage cần xem tiếp, không tự chứng minh undertranslation
hoặc mất morphology. Length/trigram flags không xác nhận semantic errors.
Không có original generated token IDs để xác minh tuyệt đối truncation.

Leave-one-row-out BLEU, cả hai systems bỏ cùng row:

| Cặp | Corpus delta | Khoảng delta khi bỏ từng câu | Số lần đổi dấu |
|---|---:|---|---:|
| CLRR-only−baseline | −1.060709 | [−1.244348,−0.830883] | 0 |
| Full−baseline | −0.566757 | [−0.739617,−0.436720] | 0 |
| Full−CLRR-only | +0.493952 | [+0.251924,+0.676197] | 0 |

Chỉ loại được giải thích rằng một row quyết định dấu; không chứng minh data
không có vấn đề. Không xóa row để làm đẹp số.

## 4. Rarity và độ dài

Source unit = `casefold().split()`, rare nếu train count ≤ 1 trong 4.600 câu.
Đây là proxy whitespace units, không phải morphemes được xác minh. Quartile
cuts: 1/6, 1/4, 1/3; ties giữ cùng nhóm nên counts không bằng nhau.

| Rare source fraction | Rows | Baseline BLEU | CLRR-only BLEU | Full BLEU | Full−baseline |
|---|---:|---:|---:|---:|---:|
| ≤ 1/6 | 172 | 24.064794 | 22.164597 | 23.783308 | −0.281486 |
| (1/6,1/4] | 185 | 18.369920 | 17.087524 | 19.205603 | +0.835683 |
| (1/4,1/3] | 106 | 11.611474 | 13.670725 | 10.490724 | −1.120750 |
| > 1/3 | 113 | 11.512476 | 8.923728 | 7.956659 | −3.555818 |

Cả hai nhánh CLRR yếu ở nhóm hiếm nhất, nhưng các nhóm khác không đơn điệu.
Chưa đủ cơ sở tạo rare-token gate/loss. Rarity liên quan length/difficulty;
không phải causal morphology probe. Không cộng subset BLEU deltas thành
đóng góp corpus hoặc chọn nhóm tốt thay main table.

| Reference nonspace characters | Rows | ΔBLEU CLRR-only−baseline | ΔBLEU Full−baseline |
|---|---:|---:|---:|
| ≤ 9 | 182 | +0.268529 | +1.015426 |
| (9,11] | 131 | −0.366607 | −0.954845 |
| (11,14] | 161 | −1.207587 | −0.680594 |
| > 14 | 102 | −1.972257 | −0.968140 |

Full tốt ở nhóm reference ngắn nhất, kém ba nhóm dài hơn. Theo source length,
Full chỉ tăng ở nhóm ≤ 5 units (+0.262678 BLEU). Không có normalized validation
source nào trùng train; điều này không chứng minh không có overlap ngữ nghĩa.

## 5. CE/gap chưa chuyển thành generation ổn định

Counts dưới đây là câu theo chrF0, không phải seed wins.

| Candidate−control | Câu tăng/giảm/tie | Same predictions | Spearman ΔchrF0 với CE benefit | Spearman với Δnative gap |
|---|---|---:|---:|---:|
| CLRR-only−baseline | 231/247/98 | 55 | 0.1069 | 0.0677 |
| Full−baseline | 211/274/91 | 47 | 0.1533 | 0.1239 |
| Full−CLRR-only | 209/256/111 | 68 | 0.1641 | 0.0438 |

CE benefit = nativeCE(control)−nativeCE(candidate), dương là CE tốt hơn.
Trong 315 câu Full cải thiện CE > 1e-4: chrF0 tăng 131, giảm 137, tie 47.
Trong 254 câu Full tăng native gap > 1e-4: tăng 109, giảm 106, tie 39.
Các tương quan là mô tả, không p-value hoặc causal estimate.

Full có mean teacher-forced CE tốt hơn nhưng corpus generation thấp hơn.
Điều này hỗ trợ kiểm tra gold-token ranking/argmax, free-running prefix và
decoding trước khi tối ưu thêm alignment/gap. Chưa xác định search error,
exposure bias hoặc calibration là nguyên nhân.

## 6. Ví dụ: có cả lexical gains/losses và nhiễu của metric

Rows là 0-based; toàn bộ top 10 gain/loss của mỗi cặp nằm trong `examples.csv`.

| Row | Reference | Baseline | Full | Quan sát reference-relative |
|---|---|---|---|---|
| 352 | 你們的房子很漂亮。 | 你們家的屋子很平滑。 | 你們的房子很漂亮。 | Full khớp reference; baseline dùng tính từ khác |
| 410 | 今天天氣很涼爽。 | 今天的天氣很好。 | 今天天氣很涼爽。 | Full khớp từ cụ thể trong reference |
| 366 | 西瓜園地有很多蒼蠅。 | 田裡有很多蒼蠅。 | 田裡的太陽很多。 | Full thay danh từ so với reference; baseline cũng thiếu chi tiết địa điểm |
| 127 | 今天的菜很鹹。 | 今天的菜很酸。 | 今天菜很酸。 | Cả hai dùng 酸 thay 鹹; mất 的 khiến chrF0 Full giảm 31.28 điểm, không tương ứng một lỗi lexical mới cùng mức độ |

Không xác nhận adequacy từ Amis hoặc coi mọi top chrF0 loss là lỗi nghĩa
nghiêm trọng hơn. Validation chỉ có 1 reference word bigram; cả ba systems
có 0 word-bigram hypotheses/matches theo chrF++ implementation. Không tự
suy ra chrF++ đã chọn sai checkpoint; giữ nguyên protocol và số cũ.

## 7. Quyết định và phép đo tiếp theo

**HOLD train interaction/global alignment mới.** Không tăng lambda LSR hoặc
thêm rarity heuristic từ một seed. Code prototype được giữ để truy vết;
correctness PASS không phải quality/novelty GO.

Hai phép đo phân biệt nguyên nhân tiếp theo, không train:

1. **Gold-token top1 accuracy và margin/rank**: cùng batch/reference, content
   mask rõ ràng, specials/EOS báo riêng; tách theo target token train frequency.
   Xem CE giảm do tăng xác suất token vốn đã đúng hay sửa quyết định top1 sai.
   Source rarity không tự xác định target rarity.
2. **Validation beam1/beam4**: cùng checkpoints, FP32 và decode limits; dùng
   beam1 sẵn có, chỉ generate beam4 nếu chưa có artifacts đã xác minh. So cả
   ba models cùng beam; không dùng test để chọn beam hoặc sửa method.

Nếu top1/rare target decisions đã kém, ưu tiên nghiên cứu can thiệp vào
information/decision ở token đó, rồi novelty audit trước code/train. Nếu
teacher-forced decisions không kém nhưng free generation kém, kiểm tra
prefix/search trước khi thay architecture. Không mặc định beam4 sẽ cứu.
Hai phép đo GPU này **chưa chạy** trong lượt hiện tại; kiểm tra VRAM trước
khi dùng RTX chung. Chưa có đủ bằng chứng chọn method mới.

Cập nhật lượt tiếp theo: cả hai phép đo đã COMPLETE,
[TOKEN_DECODING_DIAGNOSIS.md](TOKEN_DECODING_DIAGNOSIS.md). Phần trên giữ
quyết định tại thời điểm generation-strata analysis; không thay số cũ.

## 8. Artifacts và tái lập

Workspace: `outputs/experiments/amis_generation_validation_analysis/`.
Inputs 265.274 bytes, không checkpoint/NPZ. `HF_INPUT_VERIFICATION.json`
verify 9 input files; `LOCAL_VERIFICATION.json` replay 9 corpus scores,
117 strata score rows, 1.728 sentence pairs, PASS. Trong worker, leave-one-out
row ảnh hưởng nhất được direct corpus replay ở cả 9 cases, error 0.

Worker dùng lại theo args, mặc định test được giữ cho các lệnh cũ:

```bash
conda run -n clrr python scripts/analysis/analyze_prediction_strata.py \
  --predictions-dir outputs/experiments/amis_generation_validation_analysis/inputs \
  --flat-layout --split validation \
  --references data_processed/amis_mandarin/validation.csv \
  --train data_processed/amis_mandarin/train.csv \
  --design outputs/experiments/amis_generation_validation_analysis/DESIGN.json \
  --grounding outputs/experiments/source_grounding_validation \
  --output outputs/experiments/amis_generation_validation_analysis/results
```

Không thay `EXPERIMENTS_UPDATED.md`, checkpoint selection, predictions hoặc
metrics đã công bố; chỉ thêm diagnostics và cập nhật insight hiện hành.

Public derived diagnostics đã upload/verify 12 files, 295.532 bytes tại
[HF generation-validation](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/b50c4be12bb3e3eab48f2b71f33729c79accb3de/analysis/amis_mandarin/mbart-large-50/generation-validation/seed-42/beam1-fp32).
Receipt: `HF_UPLOAD.json` trong workspace local. Publication dùng allowlist
JSON/CSV số liệu dẫn xuất và source; không upload lại input snapshots,
training metadata hoặc raw qualitative examples. Full examples vẫn có local;
nguồn predictions gốc đã được xác minh qua HF revision ở mục 1.
