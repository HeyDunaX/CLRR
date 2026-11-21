# P3 — novelty và chẩn đoán checkpoint

Ngày 12/10/2026. Hoàn thành hai việc được yêu cầu: đọc prior gần nhất và đo
checkpoint đã train. **Không train mới; không thay checkpoint hay số cũ.**

## 1. Kết luận hiện hành

- **Giữ P3 làm ứng viên development:** P3 có representation phân biệt cặp dịch
  tốt hơn baseline và Full cũ trên toàn bộ validation trong phép đo này.
- **Chưa chốt P3 làm phương pháp chính của paper:** pooled InfoNCE đã có trong
  MT; skip kết hợp contrastive learning cũng đã có. Khác biệt implementation
  chưa tự chứng minh một đóng góp đủ mạnh cho ACL.
- **Không claim anti-collapse:** effective rank của P3 thấp hơn baseline, dù
  không có hiện tượng mọi câu dồn thành một vector giống nhau trong probe này.
- Gradient không có lỗi phân rã loss hoặc bất ổn số được phát hiện. Tín hiệu
  này không chứng minh nguyên nhân tăng BLEU, hoặc rằng mọi tầng không xung đột.
- Chưa tự mở confirmation nhiều seed. Bước tiếp theo cần khóa câu hỏi đóng góp
  của detached route và sự cần thiết của LSR, rồi mới duyệt một pilot có controls.

## 2. P3 đang thực sự làm gì?

P3 là `ordinary-full-v1`, không phải `effect-full-v1` (P5):

\[
\widetilde h_l=h_l+0.1\,\operatorname{sg}(\widetilde h_{l-2}),\qquad
\mathcal L=\mathcal L_{CE}+0.1\mathcal L_{LSR}+0.1\mathcal L_{NCE}.
\]

Donor được cache **sau rewiring**, trong encoder mBART. Hai tầng đầu không có
donor. `sg` ngắt gradient qua nhánh cộng thêm; đường encoder tuần tự vẫn nhận
gradient bình thường. Không có ma trận skip, gate học được, predictor hay EMA.

LSR dùng cosine giữa mean-pooled source và target encoder cuối, pooling theo
attention mask nguyên bản. Target pass không gradient; vẫn là cùng encoder
cập nhật từ source/CE, không phải một teacher có weights cố định riêng.

InfoNCE dùng mean pooling loại pad và toàn bộ special IDs của tokenizer:

\[
\mathcal L_{NCE}=-\frac1B\sum_i\log
\frac{\exp(\cos(z^s_i,\operatorname{sg}(z^t_i))/0.1)}
{\sum_j\exp(\cos(z^s_i,\operatorname{sg}(z^t_j))/0.1)}.
\]

Đây là loss một chiều source→target, cùng encoder, dropout tắt ở auxiliary
passes. Microbatch4 chỉ có3 negatives cho mỗi source; accumulation32 không
làm negative pool thành128. Không dùng hiệu ON−OFF của rewiring, queue hoặc
hard-negative mining. Không thêm parameters không có nghĩa không tốn thêm compute.

## 3. Đọc prior và đánh giá novelty

Đã đọc PDF trong `docs/paper/baseline-paper/`, trích text và kiểm tra trực quan
trang phương pháp. Đối chiếu thêm repository chính thức khi mã nguồn có sẵn.
Đây là review có giới hạn với ba prior gần nhất, không phải bằng chứng rằng
không còn công trình liên quan nào khác.

| Prior | Nội dung đã có | Khác biệt của P3 đã kiểm tra | Hệ quả cho claim |
|---|---|---|---|
| [mRASP2, ACL 2021](https://aclanthology.org/2021.acl-long.21.pdf), §2.2 | Mean-pooled encoder representations, translation positives, batch negatives, cosine/temperature, contrastive + translation loss | P3 dùng detached target, thêm CLRR và LSR, recipe low-resource khác | Không claim pooled InfoNCE trong MT là mới |
| [SkipCLM, NAACL SRW 2025](https://aclanthology.org/2025.naacl-srw.50.pdf), §3.1/Figure1 | Contrastive alignment kết hợp learned skip, linear/ReLU và warmup, trên decoder XGLM | P3 dùng encoder seq2seq, identity donor detached, distance2, cache post-rewire và strength cố định, không thêm learned weights | Không claim ý tưởng skip + contrastive là mới; phải chứng minh giá trị của route cụ thể |
| [Middle Alignment, ACL 2025](https://aclanthology.org/2025.acl-long.778.pdf), §2–3 và footnote5 | Retrieval để kiểm tra alignment; contrastive middle-layer và alternate training; đã thử stop-gradient trên English | P3 joint CE/LSR/NCE ở encoder cuối, có CLRR; không alternate task/alignment steps | Không dùng riêng stop-gradient hoặc retrieval làm novelty |

### Kiểm chứng code prior

- mRASP2 commit `d4d627b8442af062a5b6607a459fe53c6b516695`,
  [`label_smoothed_cross_entropy_with_contrastive.py`](https://github.com/PANXiao1994/mRASP2/blob/d4d627b8442af062a5b6607a459fe53c6b516695/mcolt/criterions/label_smoothed_cross_entropy_with_contrastive.py):
  masked mean pooling, cosine batch matrix, temperature/LogSoftmax và CE + contrastive.
- [SkipCLM](https://github.com/s-nlp/skipclm/tree/f286cc960ac92519e6906d0dfac57fc54515c1b3),
  commit `f286cc960ac92519e6906d0dfac57fc54515c1b3`: training code có triplet
  contrastive objective và lựa chọn skip/noskip. Tuy nhiên nhánh skip import
  `decoder.py` không có trong cây repository ở commit này; chỉ có `decoder_noskip.py`.
  Vì vậy route SkipCLM trong bảng dựa vào PDF; không khẳng định đã tái lập đầy đủ
  implementation skip, hoặc giải quyết được sự mơ hồ indexing trong công thức PDF.
- [Middle Alignment code](https://github.com/dannigt/mid-align/blob/c3fa8efec63f7bd8b0757280a6f89d96fc4060a8/src/transformers/models/llama/modeling_llama.py),
  commit `c3fa8efec63f7bd8b0757280a6f89d96fc4060a8`:
  mean pooling theo attention mask, cosine matrix và optional unidirectional
  detach trong contrastive branch. Dẫn chứng lịch sử stop-gradient dựa vào PDF;
  công thức và code được đối chiếu tại revision nêu trên.

**Đánh giá:** P3 khác implementation của ba prior này. Tuy nhiên, chưa đủ để
gọi P3 là nguyên lý contrastive/rewiring mới, hoặc đảm bảo novelty cho ACL.
Hướng đóng góp còn có thể kiểm tra là hiệu quả của **identity detached route
không thêm parameters dưới contrastive constraints**, so với route thông thường
và contrastive-only. Đây là giả thuyết nghiên cứu, chưa là claim được xác nhận.

## 4. Protocol đo checkpoint

- RTX6000 trong `/workspace/ura/thanhdat/projects/CLRR/`; kiểm tra VRAM trước
  mỗi model: lần lượt23380,23222,23222,23222 MiB free; ngưỡng tối thiểu16000.
  Peak allocated khoảng9772–9773 MiB. Không can thiệp jobs của người khác.
- Conda `clrr`, cùng runtime đã pin của pilot; FP32, TF32 off, dropout off.
- R1 baseline, R2 Full cũ, P2 LSR+InfoNCE, P3 CLRR+LSR+InfoNCE, seed42.
  Đúng best checkpoints đã khóa theo validation, không đổi theo test.
- HF revision `e04195904361dcbdf36d4ae6c7f577ca37eb3256`;
  archive path/SHA từng model có trong `PLAN.json` và `VERIFICATION.json`.
- Geometry: toàn bộ576 validation pairs, max_length256, content-token pooling.
  Raw cosine và centered cosine: trừ mean riêng từng ngôn ngữ trước normalize.
  Centering chỉ là phép đo, không sửa objective hoặc model.
- Retrieval: tìm target/source cosine cao nhất trong toàn bộ576 candidates;
  chance strict top1 là1/576≈0.174%. Duplicate-aware top1 cũng được kiểm tra,
  cho cùng scores trong cohort này. Shuffled cosine trung bình100 permutations;
  permutations có thể giữ một số cặp đúng, không phải derangements.
- Gradient: cùng4 microbatches×4 câu validation, indices khóa trước. CE, LSR,
  InfoNCE đo tách; gradients auxiliary đã nhân weight0.1. Giá trị report là
  mean các batch ratios/cosines, không phải ratio của mean norms.
- Baseline chưa train LSR/NCE: hai loss đó chỉ là counterfactual probes.
  R2 chưa train InfoNCE: NCE trên R2 cũng là counterfactual. P2/P3 dùng cả hai.
- CPU tiny-mBART gates:2 tests PASS. GPU phân rã loss đúng cả16 probes
  (absolute error0 ở FP32); finite gradients, NCE không gradient vào decoder
  parameters riêng, parameter versions giữ nguyên, **optimizer_steps=0**.
- Worker kết thúc `DIAGNOSTICS_COMPLETE`; checkpoint tải tạm đã được xóa.
  Warning generic tokenizer/Mistral xuất hiện khi load; không tự đổi tokenizer
  của frozen checkpoints. Pilot đã có tokenizer roundtrip gate trên train/validation.

## 5. Retrieval: có tín hiệu có lợi

Top1 là phần trăm câu tìm đúng đối tác; S→T là Amis→Mandarin representation
retrieval, không phải điểm chất lượng sinh bản dịch.

| Checkpoint | Raw S→T | Raw T→S | Centered S→T | Centered T→S | Raw matched cos | Raw shuffled cos | Centered matched−shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 Baseline | 54.69% | 44.44% | 65.97% | 62.85% | .3750 | .2179 | .2899 |
| R2 Full cũ | 55.56% | 40.28% | 66.49% | 62.50% | .7426 | .6498 | .3083 |
| P2 LSR + InfoNCE | 69.27% | 61.28% | 69.44% | 68.40% | .6915 | .4657 | .4718 |
| P3 CLRR + LSR + InfoNCE | **71.70%** | **67.19%** | **70.66%** | **73.44%** | .7138 | .4934 | .4716 |

**Có lợi:** P2/P3 cải thiện việc nhận diện cặp dịch; P3 vẫn cao hơn baseline
sau centering. Vì vậy lợi ích quan sát không chỉ là raw cosine cao do hướng
chung. P3 cao hơn P2 centered top1+1.22pp S→T và+5.03pp T→S.

**Giới hạn:** Full cũ có raw matched cosine cao nhất nhưng retrieval gần
baseline; cosine cao một mình không đủ giải thích BLEU. Centered cosine gap
của P3 gần như bằng P2, raw matched−shuffled của P3=.2204 còn P2=.2258.
Không phải mọi chỉ báo P3 đều tốt hơn P2. Không có phép kiểm định significance
cho retrieval trong report này; đây là thống kê mô tả trên validation đã dùng
chọn checkpoint, không chứng minh nguyên nhân hoặc tổng quát hóa độc lập.

## 6. Rank: không ủng hộ claim anti-collapse

Rank dưới đây là entropy effective rank của **centered covariance energy**;
khác effective rank tính trên singular values. Giới hạn rank mẫu là575.

| Checkpoint | Rank source | Rank target | Participation source | Participation target | Top1 variance source | Top1 variance target |
|---|---:|---:|---:|---:|---:|---:|
| R1 Baseline | 146.85 | 142.22 | 70.13 | 56.30 | 4.79% | 8.25% |
| R2 Full cũ | 141.03 | 139.75 | 66.77 | 60.49 | 5.74% | 7.18% |
| P2 | 101.24 | 100.46 | 39.57 | 34.44 | 8.82% | 11.96% |
| P3 | 103.93 | 113.62 | 41.13 | 43.53 | 8.36% | 9.30% |

P3 có diversity cao hơn P2 theo các đại lượng này, nhưng rank thấp hơn baseline.
Retrieval khá cao cùng rank khoảng104/114 không giống complete collapse về
một vector; tuy nhiên chưa có ngưỡng hoặc định lý bảo đảm không bị partial collapse.
Không chuyển kết quả này thành claim bảo toàn rank hoặc chống collapse.

## 7. Gradient và residual

Gradient ratios toàn bộ trainable parameters; weights auxiliary=.1.
`*` là counterfactual loss không có trong training objective của checkpoint.

| Checkpoint | ‖.1∇LSR‖/‖∇CE‖ | ‖.1∇NCE‖/‖∇CE‖ | cos(CE,LSR) | cos(CE,NCE) | cos(LSR,NCE) |
|---|---:|---:|---:|---:|---:|
| R1 | .3580%* | 1.0993%* | .0676* | .0831* | .5377* |
| R2 | .2147% | .7750%* | .0380 | .0674* | .3126* |
| P2 | .3436% | 1.1644% | .0962 | .1081 | .4474 |
| P3 | .2843% | .8219% | .0749 | .0558 | .4534 |

- Cả4 batch của từng model có global cosines dương. P3 NCE và LSR không hoàn
  toàn trùng hướng (cos≈.45); NCE norm khoảng3 lần LSR trong các probes.
- Không kết luận NCE là nguyên nhân duy nhất hoặc LSR vô dụng. Norm nhỏ là
  gradient tức thời ở checkpoint best, không phản ánh tác động Adam tích lũy.
- **Có xung đột cục bộ:** P3 CE–NCE âm ở ít nhất1/4 batches tại encoder layers
  3,4,5,6,7,8,10,11 (đánh số1-based); layer7 âm2/4. Không claim “loại bỏ
  gradient conflict”. Chưa có bằng chứng để bỏ các tầng này từ vài probes.
- Ratio RMS residual cộng thêm / RMS output block trước hook, trung bình theo
  câu ở source: R2 khoảng6.88–8.38%, P3 khoảng7.42–8.36% tại layers3–12.
  Baseline/P2 và hai tầng đầu đều0, đúng route. Không thấy spike chỉ ở tầng
  cuối trong phép đo này; không suy ra residual này tối ưu cho dịch thuật.

## 8. Quyết định và câu hỏi cho pilot tiếp theo

Kết hợp với [test report](RESIDUAL_EFFECT_TEST_REPORT.md): P3 test BLEU19.9158,
hơn baseline+.5232, Full cũ+1.1337 và P2+.8662. Chưa có corrected significance
trong family33 tests, và mới một training seed. Các probes hiện tại bổ sung
bằng chứng representation có ích; không nâng các chênh lệch đó thành kết luận robust.

| Quyết định | Trạng thái | Lý do |
|---|---|---|
| Giữ P3 làm ứng viên | GO development | Retrieval tốt hơn và pilot BLEU có tín hiệu |
| Chốt “P3 đủ mạnh và novel cho ACL” | HOLD | Prior overlap, một seed, significance chưa đạt |
| Claim anti-collapse / không xung đột gradient | NO-GO claim | Rank thấp hơn baseline, có layer-local conflicts |
| Train hàng loạt P3 ngay | HOLD | Cần làm rõ contribution, không chỉ thêm loss |

Nếu duyệt pilot tiếp, câu hỏi ưu tiên là **detached rewiring có đóng góp riêng
vượt skip thông thường không**, và **giữ LSR có cần thiết khi đã có NCE không**.
Các đối chứng cần khóa: P3, cùng route nhưng source-attached, LSR+NCE không
route (P2), CLRR+NCE không LSR, NCE-only; learned-skip adaptation là đối sánh
gần prior cần cân nhắc nếu claim kiến trúc. Không đổi nhiều yếu tố cùng lúc.

P2/P3 seed42 hiện có có thể dùng lại nếu hash/runtime/training recipe trùng;
không mặc định train lại controls có sẵn. Khóa validation endpoint và controls
trước khi mở runs. Test đã xem nên không dùng làm holdout sạch để chọn tiếp.
Report này chưa ấn định số runs hoặc cấp phép training mới.

## 9. Tái lập và artifacts

Worker: `scripts/analysis/diagnose_contrastive_checkpoint.py`.
SHA256: `6f9e15ad9ae0a30c0b70af39f17f2cea613c25f6e420ec39814eeded52f36219`.
CPU summary: `scripts/analysis/summarize_contrastive_diagnostics.py`.

```bash
python scripts/analysis/diagnose_contrastive_checkpoint.py \
  --workspace outputs/experiments/amis_residual_effect_rtx_test \
  --output outputs/experiments/amis_p3_diagnostics \
  --ids R1 R2 P2 P3 --rows 576 --gradient-batches 4 \
  --minimum-free-gpu-mib 16000
```

Lệnh worker chạy trong conda `clrr` trên RTX project đã chỉ định.
Local CPU tổng hợp:

```powershell
conda run -n clrr python scripts/analysis/summarize_contrastive_diagnostics.py `
  --input outputs/experiments/amis_p3_diagnostics
```

Local chỉ nhận JSON nhỏ: PLAN, COMPLETE,4 RESULT, SUMMARY,VERIFICATION.
Representations `.npz` giữ trong project trên RTX để phân tích CPU tiếp nếu cần;
không tải weights về máy. PDF/code đọc prior có receipts riêng tại
`outputs/experiments/amis_p3_diagnostics/prior_reading/`.

Public artifacts được lưu tại
`FiveC/amis-rewire-checkpoints/analysis/amis_mandarin/mbart-large-50/p3-diagnostics/seed-42/`;
revision và checksums sau upload có trong `HF_UPLOAD_VERIFIED.json`.
