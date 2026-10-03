# Kế hoạch tiếp theo cho CLRR sau audit — 2026-10-02

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: plan, thực hiện inline.
- Origin Date: 2026-10-02, Asia/Saigon.
- Verification Status: PROPOSED; số audit/CSV có provenance riêng. Chưa thực thi thí nghiệm đề xuất.
- Version Label: clrr_next_steps_v1.
- Request: đánh giá mức sẵn sàng cho A*, xác định vấn đề trước khi sửa phương pháp, tích hợp Turkish→English20k.
- Scope: file kế hoạch mới; giữ tài liệu, số liệu, dataset/checkpoint cũ. Không train, tải weights hay sửa implementation trong lượt này.

## 1. Đánh giá hiện tại

Nhận định của tôi: chưa đủ bằng chứng để tự tin nộp phiên bản hiện tại như method paper vào ACL/EMNLP Main. Đây không phải dự báo quyết định hội đồng.

Điểm yếu quyết định:
1. Full CLRR+LSR chưa xác lập ưu thế so với full-tuning mBART/NLLB trong bootstrap đã audit.
2. Phần tăng trên LSR-only mBART nhỏ và chưa có ý nghĩa; đóng góp riêng CLRR chưa rõ.
3. +5.0277 chrF++ mBART nhạy với word-bigram thưa và một câu.
4. Chưa multi-seed; cosine chỉ mT5, gradient trace cũ không hợp lệ, SVD/probe thiếu artifact.
5. Strong LoRA tốt hơn đối chứng cũ nhưng chưa equal-budget HPO, đủ seeds hay related routing control.

Bằng chứng có ích: Full vượt Strong LoRA A/B đã thử ở mBART/NLLB; mT5 gains có ý nghĩa; dual-stack mT5 thấp hơn encoder-only. Vẫn có lý do tiếp tục nghiên cứu, chưa chứng minh CLRR cần thiết.

ARR đánh giá claims/evidence, novelty/impact và reproducibility; không có ngưỡng BLEU hay p-value tự bảo đảm acceptance.
Nguồn: [ARR review form](https://aclrollingreview.org/reviewform).

## 2. Research questions

- RQ1: Cùng dữ liệu/backbone/protocol, Full có tăng chất lượng dịch trên **LSR-only**, và trên **Baseline**, qua training seeds không?
- RQ2: CLRR-only giảm do implementation, scale/direction residual, optimization/LSR balance, hay không có giá trị thêm trong điều kiện này?
- RQ3: Hiệu quả có chuyển từ Amis→Mandarin sang Turkish→English20k không?
- RQ4, nếu giữ morphological claim: Full có cải thiện dịch thông tin hình thái được người biết ngôn ngữ xác nhận, vượt LSR-only không?

Giữ Asháninka/ByT5 âm tính trong báo cáo. Không chọn dataset theo test score có lợi.

## 3. Thứ tự và điểm dừng

| Ưu tiên | Việc | Full train? | Evidence trước bước sau |
| --- | --- | --- | --- |
| P0 | Implementation/checkpoint/scoring + Turkish data audit | Không; có smoke/forward/backward nhỏ khi triển khai | Không còn lỗi làm vô hiệu đối sánh |
| P1 | Amis mBART: Baseline / CLRR-only / LSR-only / Full, seed42 | 4 pilot runs | Curves, residual statistics, gradient balance, metric chuẩn |
| P2 | Turkish sạch: zero-shot + Baseline / LSR-only / Full, mBART seed42 | Zero-shot không train + 3train runs | Signal generalization và CLRR incremental gain |
| P3 | Xác nhận3seeds; NLLB sau theo gate/ngân sách | Có chọn lọc | Mean±SD, matched-seed deltas và uncertainty |
| P4 | Sửa method có mục tiêu nếu chẩn đoán yêu cầu | 1–2variants/vòng | Variant hơn controls, selection trên validation |
| P5 | Equal-budget Strong LoRA + related residual control + mechanism | Có chọn lọc | Evidence cho novelty/fairness/interpretation |
| P6 | Cập nhật paper khi tác giả quyết định | Không | Claims phù hợp evidence |

Sau mỗi hạng mục: ghi report mới, báo tốt/xấu, chờ tác giả quyết định hạng mục tiếp. Khóa method/protocol trước confirmation; không chỉnh từ pilot test.

## 4. P0 — trước khi đổi method

### Implementation

1. Nạp đúng family/method/config; fail thiếu/unexpected keys, không âm thầm fallback baseline.
2. Cùng weights, dropout off:
   - Alpha0/lambda0 khớp baseline outputs/gradients trong sai số số học.
   - Routing lấy block i−d, detach đúng, thêm0parameters.
   - Cache reset giữa batches/source-target encoding; không lẫn residual nguồn/target anchor.
3. So forward/backward gradient checkpointing bật/tắt trên model nhỏ, cùng RNG/dtype/batch. Stateful hooks/cache cần kiểm tra; **chưa kết luận có bug**.
4. Save/reload giữ config routing và outputs.
5. Generation target language đúng; teacher-forcing/generation nhất quán. Decoder routing cần full-sequence vs incremental-cache/beam checks.
6. Validation/test bind raw references đúng split; không decode labels/lọc empty predictions; lưu signature.
7. Xác minh source hiện hành sau restore docs; không suy rằng code vẫn giống lúc audit.

### Data/tokenizer

- Hash/count/order, strings, empties/duplicates, source/pair overlaps cả ba cặp splits.
- Truncation, subword tokens/source word, language tokens/proxy, quality sample ngẫu nhiên cố định.
- Không đổi tokenizer chỉ cho Full; không chọn quality sample theo output Full tốt.
- Các runs mới chọn checkpoint bằng validation protocol mới; không trộn historical checkpoint selection như replicate tương đương.

### Gate

Có functionality/loader bug: sửa correctness trước, xác định affected runs, rồi quyết định rerun. Chỉ thiếu evidence: ghi UNVERIFIED, chưa kết luận algorithm sai.

## 5. P1 — chẩn đoán vì sao CLRR-only giảm

### Pilot Amis mBART

| Method | alpha | lambda | Updated backbone |
| --- | ---: | ---: | --- |
| Baseline | 0 | 0 | Full |
| CLRR-only | 0.1 | 0 | Full |
| LSR-only | 0 | 0.1 | Full |
| Full | 0.1 | 0.1 | Full |

Distance2/encoder; cùng LR/schedule/weight decay/batch/decoding/checkpoint selection. Seed42 pilot mới theo protocol chuẩn; không coi seed42 historical chọn theo metric khác là replicate tương đương.

Instrumentation:
- Train/validation CE, BLEU/chrF++, CE và LSR riêng.
- Masked RMS mỗi layer; norm(alpha×residual)/norm(current hidden).
- Cosine residual/current hidden.
- Trên train minibatches cố định: gradient CE vs gradient(lambda×LSR) norms/cosine. Pre-clip, không sau zero_grad.
- Anchor drift/dropout: no_grad không tự tắt dropout; đo trước khi gọi là lỗi.
- KV norms/attention entropy nếu cần; chúng là diagnostics, không causal proof.
- Lưu summary và sample IDs, không mọi full tensor mỗi step.

| Quan sát | Vấn đề cần kiểm tra | Hướng tiếp |
| --- | --- | --- |
| Bypass/reload/cache không khớp | Correctness | Sửa, rerun affected comparisons |
| Train CE tốt, validation xấu | Generalization/optimization mismatch | Scale/schedule/regularization trên validation |
| Residual ratio lớn ở vài layers | Fixed alpha đổi effective strength | Scale normalization/layer restriction |
| LSR gradient lấn CE hoặc đối hướng | Auxiliary trade-off | Lambda sweep/warmup; control LSR-only có cùng policy |
| Full≈LSR-only qua seeds/datasets | Chưa có incremental routing utility | Sửa routing có mục tiêu hoặc đổi framing |
| Gains nhưng không giữ affix gold | Có thể là semantic/optimization effect | Thu hẹp morphological claim |

Cosine thấp hơn không tự chứng minh affix retention; early stopping không tự chứng minh dataset hỏng.

## 6. Turkish→English20k

### Bộ hiện tại đã có

[data_processed/opus100_turkish_english_20k](../data_processed/opus100_turkish_english_20k/):

- Train20.000 / val2.000 / test2.000; source,target; hashes khớp manifest; không empties.
- Duplicate-pair excess rows: train399, val30, test18.
- Train–val/train–test: exact stripped source/pair overlap0.
- **Val–test:17exact sources,14exact pairs overlap**.
- Manifest chỉ kiểm train với val/test; câu strictly disjoint chưa đúng cho val–test.
- Thiếu pinned upstream revision/selected row IDs đủ tái lập sampling trong manifest.
- Turkish scripts cũ không còn trong working tree; không giả định launcher sẵn sàng.

Đã kiểm tra CSV read-only trong lượt này, chưa thay dữ liệu.

### Đề xuất v2 riêng

Tên dự kiến: data_processed/opus100_turkish_english_20k_v2/.

- Pin Helsinki-NLP/opus-100 revision, RNG/sampling algorithm, original selected row IDs.
- Source Turkish, target English.
- Train **20.000unique pairs** từ official train: loại empties/exact duplicates và nguồn trùng dev/test trước sample. Top-up theo sampling order cố định, không theo model outputs.
- Giữ official **test2.000** nguyên thứ tự, gồm18duplicate excess rows để giữ comparability. Báo duplicate structure; cluster-bootstrap sensitivity nếu cần.
- Validation pair-dedup, rồi loại sources xuất hiện trong test: bộ hiện tại còn **1.946rows**, đã tính read-only. Ghi cleaned validation; không gọi official val2.000 unchanged.
- Near-duplicate audit riêng; không tự tuyên bố đã loại mọi pretraining contamination.
- Không overwrite v1; comparisons cùng v2/hashes.
- Nếu giữ sample v1: báo20.000rows/19.601unique pairs, không20k unique.

OPUS-100 en-tr chính thức có train1.000.000/dev2.000/test2.000:
[Dataset card](https://huggingface.co/datasets/Helsinki-NLP/opus-100/blob/main/README.md).
Giữ OPUS-100, không đổi dataset khác.

### Ý nghĩa khoa học

- Cặp thứ hai, target English, fine-tuning budget20k; kiểm tra hiệu ứng ngoài Mandarin chrF++ word-bigram thưa.
- **Simulated low-resource fine-tuning**: mBART đã hỗ trợ Turkish/English. Không chứng minh unseen/endangered-language adaptation.
- Turkish→English đánh giá thông tin từ source; không trực tiếp kiểm tra sinh morphology target.
- Reverse direction là extension khác, chưa mở thêm.
- 20k giữ theo yêu cầu; so token/domain statistics trước suy nguyên nhân cỡ dữ liệu.
Nguồn: [mBART model card](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt).

### Matrix trước

mBART:
1. Zero-shot pretrained baseline.
2. Full-tuning Baseline.
3. LSR-only.
4. Full CLRR+LSR.

Ba train runs đầu seed42; sau khóa protocol, thêm13/2026 cho cả ba khi tiếp tục confirmation, tổng9train runs. Không mặc định Turkish sẽ thắng.

NLLB sau:
- Cùng v2, Baseline/LSR-only/Full một seed trước.
- Nếu claim chính: cả ba cùng3seeds.
- Strong LoRA Turkish sau khi incremental routing question có evidence.

Codes cần preflight: mBART tr_TR→en_XX; NLLB tur_Latn→eng_Latn. BLEU13a, raw chrF++6/2. Không dùng proxy cho Turkish khi source code native có sẵn.

## 7. P3 — confirmation và fairness

- Primary comparisons: Full−LSR-only, Full−Baseline.
- Seeds42/13/2026; giữ dataset sample, đổi training randomness. Tái sử dụng new-protocol runs hợp lệ.
- Amis minimum:3core methods×3seeds=9runs + CLRR-only pilot1 =>10runs. Nếu giữ synergy claim, CLRR-only3seeds =>12runs.
- Turkish minimum:3methods×3seeds=9runs/backbone đưa vào claim chính.
- Mean±SD, mỗi matched-seed delta. Single-checkpoint sentence bootstrap không thay training-seed variance.
- Không coi3seeds×Ntest là3N independent sentences; joint uncertainty cần hierarchical seed/sentence resampling chốt trước. Chỉ3seeds vẫn có precision hạn chế.
- BLEU primary, standard chrF++ secondary cho protocol mới; không đổi historical primary scores.
- Chốt comparison family/correction trước confirmation, không chuyển p/CI audit42 sang runs mới.
- Method-specific LR được phép; fairness là ngân sách tuning/selection công khai và phù hợp. Trong ablation, giữ LR/schedule/lambda để cô lập routing.
- Equal-budget validation tuning nếu giữ best-method/PEFT claim; save toàn bộ trials/compute/updates/tokens.
- Starting protocol core: max20epochs/patience4. Nếu pilot validation cần đổi, khóa thay đổi cho cả nhóm trước confirmation; không nâng patience riêng Full sau nhìn test.
- Cùng stopping/checkpoint-selection/decoding/precision/references.
- Không chọn seed tốt nhất hoặc epoch20 post-hoc làm kết quả confirmation.

## 8. P4 — cải thiện method nếu cần

Chỉ sau P0 sạch/P1 có giả thuyết cụ thể. Tối đa1–2variants/vòng:

- Residual scale quá mạnh: scale-normalized detached residual hoặc giới hạn layers. Định nghĩa masked RMS/detach của scale; chưa giả định thắng.
- Residual trùng hướng current hidden: một redundancy-removal control được định nghĩa rõ; kiểm literature trước novelty claim.
- LSR balance: lambda nhỏ/warmup auxiliary; LSR-only có cùng policy để không quy alignment gain vào routing.

Chọn trên validation với budget khóa. Controls: Baseline, LSR-only tương ứng, original Full, variant.
Learned gate đổi claim zero added parameters; ưu tiên parameter-neutral nếu giữ định vị.

Đã nhìn historical Amis test, phát triển method tiếp là post-hoc. Ghi rõ và ưu tiên confirm variant đã khóa trên Turkish held-out; thêm seeds/data độc lập khi cần. Không gọi mọi analysis là preregistered.
Normalization/gate đơn giản không tự đủ novelty; cần related-work comparison và mechanism.

## 9. P5 — evidence theo chi phí/lợi ích

1. mBART/NLLB diagnostics trên đúng Baseline/LSR-only/Full checkpoints. Save IDs/masks/full spectra nếu claim rank; không rerun mT5 cosine chỉ để thêm bảng.
2. Small blinded adequacy/morphology evaluation nếu có người biết ngôn ngữ: khoảng100items exploratory starting budget; không bảo đảm power. Cùng items, randomize method order, rubric/agreement; regex không gold.
3. Related residual control: LSR + cùng residual nhưng không detach hoặc variant đơn giản. Không gọi custom control là reproduction ResFormer/ResiDual.
4. Strong LoRA A/B equal-budget validation tuning, selected configs qua3seeds nếu giữ PEFT claim. Longer schedule nếu validation cho thấy chưa hội tụ và budget đã định.
5. Validation sensitivity nhỏ d/alpha/lambda, actual artifacts; không full-grid mọi backbone/default=optimum.
6. Matched timing repetitions/memory chỉ nếu giữ overhead claim; train totals khác epochs không benchmark overhead.

Chưa ưu tiên nhiều PEFT họ mới trước RQ1. IA3/adapter có thể bổ sung sau core evidence.

## 10. Decision gates cho bài

| Evidence | Quyết định |
| --- | --- |
| Full hơn LSR-only/Baseline, effects nhất quán, có generalization | Giữ method paper; bổ sung residual comparison/mechanism |
| Full hơn Baseline nhưng không hơn LSR-only | Alignment có đóng góp; routing novelty chưa rõ. Sửa có mục tiêu hoặc đổi câu chuyện |
| Chỉ mT5 gain | Thu hẹp scope, không universal |
| Turkish gain, Amis/Ash không | Simulated20k transfer, chưa giải quyết endangered-language claim |
| Correctness bug ảnh hưởng results | Sửa/tái lập affected runs trước đánh giá algorithm |
| Không improvement nhưng diagnosis chắc/mới | Cân nhắc empirical analysis với RQ riêng; đánh giá novelty/venue lại |

Đây là tiêu chí đề xuất, không acceptance rules. Nonsignificance chưa chứng minh equivalence; cần effect/CI/power/practical differences.

## 11. Vận hành khi được cho chạy

- Conda clrr cho Python/preparation/tests/training.
- Colab A100 **không high-mem**, theo yêu cầu đã có.
- Output riêng theo dataset/method/variant/seed; không auto-resume khác protocol.
- Freeze code/environment/data hashes; log process alive, progress, best epoch/validation, elapsed runtime và artifacts.
- Khi thực sự chạy: báo chat mỗi10phút và khi kết thúc/lỗi. Mỗi hạng mục xong ghi report mới rồi chờ quyết định bước tiếp.
- Đo throughput pilot để ước lượng GPU-hours; không hứa thời gian từ runs cũ.
- Scripts còn thiếu là phần triển khai sau; chưa tạo launcher/chạy lệnh training trong lượt này.

## 12. Handoff/deliverables riêng

- Bước đầu: **P0 correctness + Turkish data audit**, tiếp **P1 Amis component pilot**, rồi **P2 Turkish core pilot**.
- Kế hoạch: docs/NEXT_STEPS_AUDITED_20261002.md.
- Audit sources: outputs_rebuttal/metric_audit_20261002/full/.
- Dự kiến diagnosis: outputs_rebuttal/clrr_diagnosis_v1/.
- Dự kiến confirmation: outputs_rebuttal/clrr_confirmation_v1/.
- Dự kiến dataset v2: data_processed/opus100_turkish_english_20k_v2/ với source,target và manifest.
- Không sửa EXPERIMENTS/RESEARCH_INSIGHTS/implementation_plan/clrr_main.tex cũ trong lượt lập kế hoạch.
