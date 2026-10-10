# CLRR — tổng hợp bằng chứng và kế hoạch dài hạn GO / NO-GO hướng tới ACL

> **CẬP NHẬT CUỐI 05/10 — CORRECTED_CONTROLS_VERIFIED:** Sáu runs LSR.03/CLRR hoàn tất, metrics và backup verified, cả hai GPU đã tắt. Validation chọn LSR.10. Full meanBLEU20.1303; CLRR-only19.3539. [Bàn giao thực thi tiếp](../../../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md) cập nhật N1–N8. Fresh confirmation cấu hình hiện tại là4distinctfamilies×8seeds=32runs vì matching/tunedLSR trùng .10; bảng40–48runs phía dưới là budget khi configs khác/nâng cấp. Chưa launch fresh seeds/Turkish.


> **Hiện hành 05/10/2026 — G0 CORRECTED_CONTROLS:** accumulation fix đã qua CPU/GPU/full-size gates; baseline audited/reused; Full và LSRλ.10 corrected seeds42–44 đã VERIFIED. Full−LSR meantestBLEU+0.7373, nhưng chưa có significance robust qua seeds; xem [significance](CORRECTED_ACCUMULATION_SIGNIFICANCE_20261005.md). User đã cho chạy tiếp LSRλ.03 vàCLRR-only seeds42–44 (sáu runs), hai A100 Standard. Đây là hoàn thiện cohort đối chứng sau sửa lỗi, **chưa phải G2 nâng cấp, G3 fresh-seed confirmation hay G4Turkish**. G3 vẫn là đề xuất8freshseeds45–52/budget có điều kiện, chưa đổi thành6seeds. G0novelty/G1candidate cards chưa được chứng nhận hoàn tất; các ghi chú TECHNICAL_HOLD/chưatraining bên dưới là lịch sử.

> **Cập nhật 05/10/2026 — TECHNICAL_HOLD:** G0 tái hiện lỗi chuẩn hóa gradient accumulation ở wrappers CLRR/LSR dưới Transformers 4.57.6; bản sửa qua synthetic regression. Xem [G0/G1 verification](G0_G1_VERIFICATION_20261005.md). Giữ scores lịch sử, nhưng phải xử lý affected controls trước G2; các gate và claims bên dưới là kế hoạch trước phát hiện này.

**Ngày lập: 04/10/2026, sau khi warmup v2 hoàn tất.**

**Trạng thái: kế hoạch đề xuất, chưa chạy thực nghiệm mới.** Mục tiêu là xây dựng đóng góp đủ rõ, đúng và có ích để hướng tới ACL main conference. Những ngưỡng dưới đây là tiêu chí đầu tư của dự án, không phải quy định ACL hoặc bảo đảm acceptance.

## Material Passport

- Skill / Mode: academic-research-suite / experiment-agent, plan.
- Nguồn: paper, experiments, insight, bàn giao, metrics và artifacts C/confirmation/final/V4/V5/warmup v2.
- Verification Status: **kế hoạch chưa thực thi**; các số nền lấy từ artifacts đã audit.
- Version: `acl_long_term_plan_v1_20261004`.
- Phạm vi: tổng hợp bằng chứng, thiết kế nâng cấp và xác nhận, chuẩn bị manuscript.
- Quyền thực thi: lượt này chỉ lập tài liệu. Turkish vẫn chưa được bật; không tự mở Colab hoặc train theo bảng này.
- Giữ nguyên số lịch sử; không đưa Asháninka/LayerSkip trở lại phạm vi nghiên cứu.

## 0. Khuyến nghị chiến lược

**Tiếp tục CLRR có điều kiện, nhưng chưa mở rộng cấu hình hiện tại thành claim phương pháp mạnh.**

1. Routing thực sự tham gia computation. Tuy nhiên, lợi ích riêng trước LSR được tune còn nhỏ và chưa ổn định.
2. Tuning α/λ, bỏ detach và warmup20% đã được thử; không lập lại chúng như những việc chưa làm.
3. Ưu tiên xác định **routing bổ sung gì sau khi alignment đã mạnh**, đồng thời phân biệt điểm mới với skip/fusion có sẵn.
4. Chọn tối đa **hai nâng cấp có giả thuyết phân biệt được**. Sau pilot, xác nhận bằng seeds mới và direction khác.
5. Nếu vẫn không có lợi ích thuyết phục, chuyển sang đóng góp thực nghiệm/cơ chế có phạm vi rõ; không mở sweep vô hạn để tìm bảng điểm thuận lợi.

[ARR Review Form](https://aclrollingreview.org/reviewform) đánh giá mức hỗ trợ cho claims, tính ích lợi của đóng góp và khả năng tái lập. [ARR Authors Guidelines](https://aclrollingreview.org/authors) không đặt số ngôn ngữ/models cố định cho mọi claim. Số lượng thí nghiệm vì vậy phải phục vụ câu hỏi nghiên cứu.

### Phân biệt các quyết định

| Trạng thái | Nghĩa | Được làm tiếp |
|---|---|---|
| `TECHNICAL_HOLD` | Protocol, weights, labels hoặc metrics chưa hợp lệ | Sửa và kiểm chứng; chưa diễn giải quality |
| `GO_DIAGNOSTIC` | Phép đo trả lời một câu hỏi còn thiếu | Đo; không mặc định phương pháp tốt |
| `GO_PILOT` | Có giả thuyết, controls và budget cụ thể | Train trên development, chưa dùng test chọn |
| `GO_CONFIRMATION` | Ứng viên qua validation gate và đã khóa | Seeds mới và xác nhận ngoài bối cảnh chọn cấu hình |
| `GO_SCOPE_LIMITED` | Có ích trong một phạm vi, chưa tổng quát | Viết đúng phạm vi hoặc bổ sung phép thử |
| `NO_GO_CANDIDATE` | Một cấu hình không đủ ích lợi | Giữ kết quả, loại cấu hình khỏi ưu tiên |
| `NO_GO_SUPERIORITY` | Chưa đủ bằng chứng hơn đối chứng mạnh | Nâng cấp hoặc đổi narrative |
| `GO_SUBMISSION_REVIEW` | Đủ bộ bằng chứng để review bản thảo | Review trước submission; không bảo đảm accept |

**Hiện tại:** các phép đo đã audit đạt technical PASS; warmup20% là `NO_GO_CANDIDATE`; superiority/generalization rộng là `NO_GO_SUPERIORITY`; chẩn đoán có mục tiêu là `GO_DIAGNOSTIC`.

## 1. Tổng hợp thực nghiệm hiện có

### 1.1. Inventory

Không cộng mọi số dưới đây thành tổng models độc lập. A/B/C/final/V4/V5 dùng lại nhiều weights; generate nhiều beams vẫn chỉ là một training run.

| Nhóm | Đã có gì | Giá trị | Giới hạn / trạng thái |
|---|---|---|---|
| A: audit predictions lịch sử | 28 outputs Amis, nhiều methods/backbones | Truy vết và chấm đúng reference | Training/generation lịch sử chưa chuẩn hóa |
| B: regenerate weights lịch sử | 10 checkpoints FP32 và4 route-off outputs | Quality/backbone với inference chung | Single selected checkpoints; không thay matched multi-seed |
| C: matched components | FT / CLRR-only / LSR-only / Full ×3 seeds =12 runs/results | Phân tách routing và alignment | Một direction; một số weights seed42 mất, đã recover/replicate có nhãn |
| Character/influence C | Đủ12 systems, đủ575 test rows | Hiểu ảnh hưởng sparse Chinese word n-grams | Leave-one-out không chứng minh morphology hoặc tính độc lập |
| Strong LoRA | mBART A/B và NLLB A/B, all-linear, r16 | Trả lời một phần lo baseline LoRA quá yếu | Seed42; A/B đổi cả LR và embeddings; chưa HPO/seeds ngang nhau |
| Screening D | 8 trials seed42: Full4, LSR4 cấu hình mới, validation-only | Kiểm tra α/λ; tìm LSR λ=.03 | Trial thắng một seed chưa là cấu hình ổn định |
| Confirmation | 6 runs mới, gồm replica/recovery và controls/Fullλ=.03 | Full/matched/tuned LSR cùng3 seeds | Không phải6 ứng viên độc lập; đọc đúng run IDs |
| Source-attached | Full bỏ detach nguồn routing ×3 seeds | Vai trò detach | Chưa chứng minh detach tối ưu mọi bối cảnh |
| Final evaluation | 18 models =6 families×3 seeds, test beam4 | So FT, CLRR, matched/tuned LSR, attached và Full | Amis test đã được xem; phát triển tiếp là exploratory |
| Fullλ=.03 follow-up | Thêm3 test outputs, thành7 families×3 =21 checkpoints | Giảmλ đơn giản không giải quyết contribution gap | Giữ đúng statistical family |
| V4 đúng protocol | 21 selected-checkpoint probes,3 paired route-controls | Routing hoạt động; cơ chế có provenance | On/off cùng weights không là causal training benefit |
| V5 đúng protocol | 49 conditions:42 core +3 baseline +4 verification; error analysis | α/groups/beam/split; sửa baseline42 beam label | Metadata lịch sử chưa hoàn toàn phục hồi |
| Warmup v2 | 6 models,24 snapshots,36 probe conditions | Kiểm tra thời điểm đưa routing vào | NO-GO; validation-only, không có test mới |
| Turkish v2 | Train20k/dev1946/test2000, manifest và zero-shot | Sẵn cho direction confirmation | Chưa có trained component suite |
| Correctness / smoke | Cache, detach, α0, checkpointing, save/reload, precision | Giảm rủi ro implementation | Không thay kiểm chứng optimization hoặc quality |

**Không dùng làm evidence hiện hành:** F4 v2 bị hook contamination; V3 có vấn đề replay/runtime/gradient/rank; warmup v1 sai protocol. Giữ để audit, sử dụng V4/V5/warmup v2 thay thế tương ứng.

### 1.2. Chất lượng Amis hiện hành — test beam4, đủ575 câu

Mean qua seeds42/43/44; đây là models trước warmup, **không phải bảng validation mới**.

| Method | BLEU | chrF++ | chrF0 | Kết luận |
|---|---:|---:|---:|---|
| FT baseline | 19.3886 | 16.9249 | 18.0452 | Đối chứng full fine-tuning |
| CLRR-only α=.10 | 19.5769 | 18.5172 | 18.2035 | BLEU+0.1883 soFT; chỉ1/3 seeds dương |
| Matched LSR λ=.10 | 19.5343 | 18.5639 | 18.2040 | Cùngλ với Full |
| **Full α=.10/λ=.10** | **19.7687** | **18.6565** | **18.3715** | Hơn matchedLSR+0.2344BLEU,3/3 seeds; hơnFT+0.3800,2/3 |
| Tuned LSR λ=.03 | 19.8954 | 17.2923 | 18.5350 | Full thấp hơn0.1267BLEU và0.1635chrF0 |
| Full source-attached | 19.5658 | 18.5633 | 18.2638 | Bỏ detach chưa đủ ích lợi để thay method |
| Full α=.10/λ=.03 | 19.0854 | 18.2324 | 17.8648 | Giảmλ đơn giản chưa giúp |

Nguồn: [test_family_summary.csv](../../../../outputs_rebuttal/amis_final_evaluation_20261004/test_family_summary.csv), [experiments mục10–12](../../../experiments/EXPERIMENTS_UPDATED.md), follow-up21 checkpoints.

Full−tuned LSR chrF++ +1.3642 phụ thuộc mạnh seed43/sparse word-bigrams; không phủ nhận BLEU/chrF0 âm. Bootstrap câu không thay uncertainty qua training seeds. Full−matched LSR là tín hiệu có lợi nhất hiện tại, nhưng còn nhỏ.

### 1.3. Validation đã sửa protocol và warmup vừa hoàn tất

| Method | BLEU | chrF++ | chrF0 | Vai trò |
|---|---:|---:|---:|---|
| FT beam1 đã sửa | 17.1275 | 14.3003 | 16.4849 | V5: baseline42 đo lại đúngbeam1 |
| Fixed Full | 17.2135 | 14.4975 | 16.6586 | Frozen control cho warmup |
| Tuned LSR λ=.03 | 16.9942 | 14.3170 | 16.4950 | Đối chứng mạnh |
| Fixed CLRR-only | 16.8164 | 14.0765 | 16.2050 | Replay3 seeds trướcwarmup |
| Warmup Full20% | 17.0048 | 14.3859 | 16.4809 | NO-GO: thấp hơn fixedFull cả3 metrics |
| Warmup CLRR-only20% | 15.9842 | 13.5858 | 15.6606 | NO-GO: meanBLEU giảm0.8323 |

Full validation hơn tunedLSR nhưng test BLEU/chrF0 thấp hơn ở cảbeam1 và4. Beam mismatch **không giải thích hết** ranking instability. Baseline42 cũ17.8735BLEU làbeam4; test cũ không đổi.

Nguồn: [V5 summary](../../../../outputs_rebuttal/amis_routing_diagnostics_20261004_v5/diagnostic_summary.json), [warmup audit](../../../../outputs_rebuttal/amis_routing_warmup_20261004_v2/independent_audit_summary.json), [warmup insight](AMIS_WARMUP_V2_RESULTS.md).

### 1.4. Cơ chế và giới hạn

| Quan sát đã xác minh | Điều được hỗ trợ | Chưa được hỗ trợ |
|---|---|---|
| V4 routing-on−off mean+2.9676 validationBLEU | Model đã train với routing phụ thuộc computation đó | Gain+2.97 trướcLSR được train độc lập |
| V5 cả7 α/group interventions có meanquality giảm | Inference tuning đã thử không giúp fixedFull | Mọi train-time variant đều vô ích |
| Spectrum đổi trước finalLN, final pooled rank còn cao | Routing thay geometry | Mathematical non-collapse / causal anti-collapse |
| Global weightedLSR/CE norm nhỏ; có local negative cosines | Chưa thấy globalLSR domination | λ phải tăng/giảm hoặc không có conflict |
| Residual/current khoảng6%, không explosion ở sites đã đo | Biên độ routing có thể định lượng | Không thể có outlier hoặc không cần benchmark |
| Warmup alignment tăng, quality chưa tốt hơn | Alignment không tự chuyển thành gain dịch | Chọn method bằng cosine/rank cao |
| Historical mT5 Full−FT+1.8074BLEU, Holm36p=.0036 | Một checkpoint lịch sử có tín hiệu rõ | Gain matched multi-seed trên principal backbones |
| Historical NLLB+0.6650BLEU; ByT5−0.1696 | Backbone responses khác nhau | General superiority |

### 1.5. Reviewer → mức giải quyết

| Lo ngại | Đã làm | Còn thiếu |
|---|---|---|
| CLRR-only yếu, gain chủ yếu từLSR | 4-way3seeds +matched/tunedLSR | Incremental benefit trướcLSR mạnh hoặc thu hẹp claim |
| Mộtseed | 3seeds cho study chính | Fresh seeds và uncertainty |
| LoRA/BitFit yếu, fairnessLR | Strong all-linear LoRA mBART/NLLB | HPO và multi-seed nếu giữ PEFT claim |
| IA3/adapters/adaptiveLoRA thiếu | Chưa có matched suite | Một comparator có ích; không cần train mọi family |
| Raw/Zh chrF++ mơ hồ | Native formula, rawrefs, chrF0/influence | Đọc cùng BLEU/chrF0; không đổi metric hậu nghiệm |
| Non-collapse overstated | Rank/gradient/residual trên mBART | Bỏ proofclaim; intervention nếu muốn causal mechanism |
| Morphological fidelity | Regex/proxy chưa là gold | Expert annotation/challenge nếu giữ linguistic claim |
| Generalization | Chưa có direction thứhai đãtrain | Turkish sau khi khóa method |
| Residual/value-routing novelty | Attached control đã có | Formula audit và1–2 route comparators |
| Negligible overhead | Zero addedparameters đã kiểm tra | Matched latency/throughput/memory benchmark |

## 2. RQ và câu chuyện paper

**Hướng chính:** routing không thêm parameters có cải thiện adaptation ngoài CE và latent alignment đã tune, với hiệu quả lặp qua seeds và bối cảnh hay không?

| Claim mục tiêu | Evidence cần | Hiện tại |
|---|---|---|
| C1: Full hữu ích trướcFT mạnh | Fresh seeds, effect/uncertainty, direction confirmation | Có signal Amis |
| C2: Routing bổ sungLSR | Full−matchedLSR vàFull−best observed tunedLSR | Nút thắt chính |
| C3: Đổi computation không thêmweights | Formula/tests/parametercounts | Nền tảng tốt |
| C4: Có thể transfer | Khóa method trướcTurkish; NLLB replication | Chưa có |
| C5: Cơ chế giải thích help/harm | Intervention có dự đoán và taskmetric | Observations, chưa causal trainingmechanism |
| C6: Chi phí thấp | Matched train/inference benchmark | Chưa đủ |

CLRR-only không bắt buộc luôn hơnFT nếu claim là **tương tác routing–alignment**. Khi đó Full phải có lợi hơnLSR và interaction phải được kiểm tra; không viết standalone benefit nếu ablation không ủng hộ.

**Hướng dự phòng:** khi nào routing thay representation nhưng không cải thiện translation? Muốn đây là đóng góp chính, cần dự đoán có thể bác bỏ, intervention đúng, kiểm chứng ngoài context phát hiện và đối chiếu prior. Danh sách scores âm tự nó chưa là insight mạnh.

## 3. Bảng kế hoạch đường dài

**Mỗi hàng là một gate, không phải việc phải chạy hết.** Budget có điều kiện, không là lệnh launch.

| Gate | Chạy/đo cụ thể | Train mới | GO | NO-GO / bước sau |
|---|---|---:|---|---|
| **G0: nguồn số +novelty** | Inventory, claimledger, priorformulas; optimization smoke nếu thiếu | 0 fullmodels | Số hợp lệ; khác biệt test được | Sửa protocol hoặc đổi narrative |
| **G1: nguyên nhân** | Histories/selection, warmupbest on/off, target-anchor counterfactual | 0 | Chọn1–2 hypotheses phân biệt được | Dừng diagnostics lặp/sweep vô hạn |
| **G2: pilot nâng cấp** | Tốiđa2candidates×Full/CLRR-only×3developmentseeds | ≤12 | Effect/no-harm/stability gate đạt | Loại candidate; sau2 hướng mới thất bại review method/RQ |
| **G3: freshseed Amis** | 5corefamilies×8seeds; thêmoriginalFull×8 nếu đổi method | 40–48 | Gain lặp; uncertainty phù hợp claim | Inconclusive→scope hẹp; negative→dừng superiorityroute |
| **G4: language confirmation** | Turkish20k,5families×8seeds | 40 | Transfer benefit trướcFT/matched/tunedLSR | Giữ kết quả âm; không đổi data để xóa NO-GO |
| **G5: backbone confirmation** | NLLB Turkish20k,5families×5seeds | 25 | Lặp ngoài mBART | Thu hẹp backboneclaim |
| **G6: comparators** | Generic/prior routing; matchedPEFT nếuclaim cần | Theo§9 | Khác biệt hữu ích ngoài skip/LSR/tuningbudget | Đổi positioning; khôngclaimPEFTfails |
| **G7: dataregime +mechanism** | Turkish5k; intervention/goldquality nếuclaim cần | 15 cho5k; nhánh khác riêng | Pattern định trước có ích | Bỏ claim chưa được hỗ trợ |
| **G8: efficiency +paper** | Benchmark, claimcheck, ACLformat, review | 0 | Mainclaims đủsupport, rõnovelty vàreproducibility | Revision / empiricalroute / venue khác |

Thứ tự chính: **G0→G1→G2→G3→G4**. Literature G6 bắt đầu ngayG0; comparator training sau khóa candidate. G5/G7/G8 có thể song song trong budget đã chốt. G2/G3fail thì không tự khởi40 Turkishruns.

## 4. G0 — nguồn bằng chứng và novelty

### 4.1. Một inventory hiện hành

Tạo `outputs_rebuttal/acl_plan_<date>/evidence_inventory.csv`, gồm:

- Run ID, family, seed; training hay diagnostic; split và số beam thực tế.
- Code, environment, model/tokenizer revisions và data hashes.
- Selected step và selection metric; metadata chưa biết phải ghi `unknown`.
- Prediction SHA, metric signatures, HF path/revision/archive SHA.
- Verdict VERIFIED/PARTIAL/INVALID/HISTORICAL, lý do và phiên bản thay thế.

Giữ artifacts cũ bất biến. Handoff có labels cũ và correction V5: inventory phải chỉ rõ nguồn hiện hành. Chấm đúng công thức chưa chứng nhận đúng generation protocol. Baseline seed42 best step vẫn chưa biết nếu thiếu receipt gắn đúng weights; trainer state cùng tên không đủ.

**Paper đang chậm hơn kết quả:** manuscript còn ghi influence audit C chưa xong và chưa tích hợp đầy đủ tuned LSR/V5/warmup. Ghi các chỗ cần sửa vào `paper_claim_ledger.md`; lượt này chưa sửa manuscript hoặc bảng số.

### 4.2. Đối chiếu prior trước pilot

So công thức, vị trí cộng residual, đường gradient, projections, parameters, loss, pretrained/from-scratch, cost và task. Không so BLEU khác datasets như một ranking thực nghiệm.

| Prior | Liên hệ cần xử lý | Đối chứng phù hợp |
|---|---|---|
| [Multi-layer Representation Fusion](https://aclanthology.org/C18-1255/) | Fusion nhiều tầng đã có | Phân biệt routing/detach với output fusion |
| [Dense Information Flow](https://aclanthology.org/N18-1117/) | Dense connections cho NMT đã có | Một generic/dense route control cùng backbone |
| [SkipCLM](https://aclanthology.org/2025.naacl-srw.50/) | Đã đọc §3 PDF: learned skip + InfoNCE trên decoder XGLM, có warmup | CLRR fixed/detached encoder khác; skip + alignment không tự là novelty |
| ResiDual/ResFormer/value-residual | Reviewer nêu; cần đọc đúng primary paper/code/version | Không gọi generic control là reproduction |
| Middle-layer alignment / LSR / JEPA-related | So objective thực; LSR không predictor/EMA | Tên legacy `jepa` không thay novelty analysis |

SkipCLM là prior gần cần xử lý, **không phải kết luận CLRR trùng paper đó**. Search này có giới hạn; cập nhật trước submission và đọc công thức/code gốc cho comparator được chọn.

**GO:** viết được một câu khác biệt cụ thể và một ablation có thể bác bỏ nó. **NO-GO:** chỉ còn câu chuyện skip + alignment + warmup, không có ích lợi riêng; đổi narrative hoặc method trước pilot.

### 4.3. Correctness khi còn risk cụ thể

Nếu thiếu receipt, kiểm tra trên tiny/pinned model:

- Gradient/update của một effective batch so với bốn microbatches, cùng RNG, dropout tắt, đủ bốn branches. Kiểm tra aggregation CE/LSR và label-token normalization.
- Best reload đúng weights; restore args không đổi split/beam; α0 khớp raw logits; cache isolation và parameter partition.

Đây là gate đề xuất, **không phải lỗi gradient accumulation đã xác nhận**. Logged training loss khác scale evaluation CE chưa chứng minh optimizer sai. Nếu tìm được bug ảnh hưởng training: giữ archive, ghi impact matrix và rerun mọi affected controls; không sửa riêng Full rồi so baseline cũ.

## 5. G1 — đo để phân biệt nguyên nhân

| Mã | Việc cụ thể | Đầu ra | Giới hạn |
|---|---|---|---|
| D1 | So sáu warmup histories, selected steps, actual α, BLEU/chrF++/CE; fixed histories nếu gắn đúng weights | `history_selection_audit.md` | Không cần GPU nếu records đủ; thiếu ghi unknown |
| D2 | Sáu warmup best: reuse predictions α=.10, generate validation beam1 với α=0 | Sáu generations mới, paired ba seeds/family | Không test; không lặp sweep fixed Full |
| D3 | Ba fixed Full best: target anchor LSR routed/raw, giữ source pass | Loss, gradient groups, anchor change, cache checks | Counterfactual loss không chứng minh training quality |
| D4 | Đọc error analysis V5; annotation nếu cần | Ma trận giả thuyết lỗi | Không tune bằng Amis test subgroups |

D2 chỉ chạy nếu D1 cần biết warmup model còn phụ thuộc routing tại selected best. V4 đã trả lời cho fixed Full. Probe epoch20 không đại diện best epoch18; warmup NO-GO không cần test mới chỉ để hoàn bảng.

### Hai nâng cấp có điều kiện

Chọn tối đa hai hướng sau diagnostics/prior audit. Chưa có cơ sở tăng λ hoặc normalize chỉ vì gradient ratio nhỏ hay rank chưa đẹp.

| Hướng | Giả thuyết | Thay đổi đầu tiên | Control / dự đoán |
|---|---|---|---|
| **U1: target anchor không routing** | Anchor routed tối ưu geometry không giúp translation | Source encoder giữ CLRR; target no-grad pass tắt routing; α=.10/λ=.10 | Matched LSR raw target, old Full; đo anchor change và quality gain |
| **U2: topology/distance** | Recursive post-rewiring cache/distance2 tạo bias không phù hợp | Chọn **một**: cache trước rewiring hoặc d=4; giữ các settings khác | Old d2 Full, CLRR-only; generic d1 control ở G6 |
| Chỉ khi D1 hỗ trợ | Sparse word n-grams làm selection mất ổn định | Cohort mới chọn best bằng validation BLEU, chrF0 guard | Rerun mọi core families; không âm thầm reuse controls cũ |
| Chỉ khi local conflict đủ rõ | Conflict liên quan harm ở giai đoạn cụ thể | Một LSR schedule/gradient rule định trước | LSR-only cùng rule/λ |

U1/U2 chưa là nguyên nhân đã chứng minh. Nếu evidence không ủng hộ, không train chỉ vì có tên trong bảng. Thêm learned gate/projection làm mất parameter neutrality: phải đổi mô tả, counts, cost và controls.

**G1 GO:** khóa candidate cards, ghi dự đoán quality/mechanism, thay đổi, controls, budget và failure criteria. **NO-GO:** vẫn thiếu phép phân biệt; báo chưa biết nguyên nhân, dừng diagnostics lặp và chọn scope hẹp.

## 6. G2 — pilot nâng cấp trên Amis development

- Seeds42/43/44 đã được xem nhiều, không gọi independent confirmation.
- Một candidate: Full×3 và CLRR-only tương ứng×3 = **6 runs**. Tối đa hai candidates = **12 runs mới**.
- U1 không đổi CLRR-only: có thể reuse ba controls nếu source branch/optimization/RNG/policy thật sự khớp, giảm6→3 runs; phải có receipt.
- FT/old Full/matched/tuned LSR chỉ reuse sau G0. Đổi selection/LR/loss aggregation/proxy/runtime có ảnh hưởng thì mở cohort/budget mới.
- Objective candidate đổi λ/schedule cần thêm LSR-only matching ba runs/candidate; không giấu ngoài trần12.
- Protocol P0: validation beam1, native chrF++ chọn best; không dùng test chọn.

### Gate khóa trước train

Fnew/Fold/Lmatch/Ltuned là means ba seeds của Full mới/Full cũ/LSR matching/LSR tốt nhất đã quan sát trong grid công bố:

1. Fnew−Fold≥**+0.30 BLEU**, Fnew−Ltuned≥**+0.30 BLEU**.
2. Fnew−Lmatch≥**+0.30 BLEU**, ít nhất2/3 paired seeds dương.
3. Mean chrF0 không thấp hơn Fold/Ltuned; chrF++ không thấp hơn Fold.
4. Ít nhất2/3 seeds cùng tăng BLEU và chrF0 trước Fold.
5. Kiểm tra mechanistic prediction; nếu sai, giảm claim mechanism. Rank/cosine không thay quality gate.

Đạt → `GO_CONFIRMATION`. Quality tăng nhưng cơ chế dự đoán sai → có thể confirm quality với narrative đúng. Không đạt → giữ failure table và `NO_GO_CANDIDATE`.

Sau tối đa **hai nâng cấp mới** thất bại, dừng superiority route hiện tại và review method/RQ. Warmup fail vẫn giữ trong history. Không bật Turkish để tìm số thuận lợi.

## 7. G3 — Amis với fresh seeds

### Số seeds và power

Đề xuất **tám fresh seeds45–52**, khóa trước confirmation; không thêm lần lượt đến p<.05. Gain hiện tại nhỏ nên ba seeds chưa đủ claim stability/variance. Tám là thiết kế dự án, không phải ACL rule hoặc power guarantee.

Trước launch, ước lượng power/precision từ paired development deltas và effect tối thiểu có ích. Với effect0.3 BLEU và paired SD0.5, tám seeds có thể vẫn thiếu power. Nếu cần n khác, chốt trước khi xem confirmation test; không tăng n vì p xấu.

### Ma trận models

| Family | Câu hỏi | Runs |
|---|---|---:|
| FT baseline | Benefit trước CE-only adaptation | 8 |
| CLRR-only candidate | Routing đứng riêng | 8 |
| LSR-only matching | Routing bổ sung cùng latent objective | 8 |
| LSR-only best observed tuned | Control mạnh trong HPO budget công bố | 8 |
| Full candidate | Method khóa ở G2 | 8 |
| Original Full control | Benefit nâng cấp trên fresh seeds | 8 nếu đổi method |

**40–48 runs**, cùng init/data/optimizer/selection/generation trong mỗi seed. Matching/tuned LSR trùng config thì reuse cùng tám runs, không nhân tạo thành16 models. Old Full seed42 không làm control seed45.

Đổi LSR definition thì matching LSR phải đổi và tuned control phải có HPO phù hợp. λ=.03 là best observed trong grid cũ, không mặc định optimum của objective mới.

### Gate

- Technical PASS, đủ metadata/backup, selection locked trước mở test.
- Gate đầu tư: Full−FT mean≥**.50 BLEU**; Full−matching/tuned LSR mean≥**.30**; ≥**6/8** paired seeds dương trước LSR; mean chrF0 no-harm.
- Nếu kể nâng cấp, report Fullnew−Fullold cùng fresh seeds.
- Effects +seed CI +sentence CI theo §11. Threshold không thay uncertainty. CI qua0→inconclusive, không suy bằng nhau hoặc chắc chắn tốt hơn.
- Effect nhỏ hơn gate có thể report trong scope nhỏ; không đổi threshold hậu nghiệm để gọi GO.

Fresh seeds **không làm Amis test đã xem trở lại untouched holdout**. G4 vẫn cần.

## 8. G4/G5 — ngôn ngữ và backbone khác

### Turkish→English20k, mBART

- Data: `data_processed/opus100_turkish_english_20k_v2/`;20000/1946/2000; revision/hashes/source-disjoint splits giữ nguyên.
- Đây là **simulated low-resource fine-tuning** trên language có trong pretraining; khác unseen-language Amis.
- Pinned model/tokenizer; `tr_TR→en_XX`; BLEU13a/native chrF++, chrF0 diagnostic; train BF16, generate FP32, TF32 policy công bố.
- Bốn components +best tuned LSR nếu khác config =5 families×8 seeds45–52 = **40 runs**. Zero-shot là mốc phụ, không thay FT baseline.
- Transfer method khóa từ Amis. HPO thêm trên Turkish validation thì label **tuned transfer**, công bố budget; không viết zero-retuning.
- Có thể chạy ba seeds đầu/family =15 runs để technical/validation gate. Chốt trước rằng final test mở sau đủ tám hoặc stop declaration; không test pilot rồi sửa method và gọi independent confirmation.
- Test có18 duplicate-excess pairs: primary giữ đủ2000 rows; sensitivity/clustered resampling khi có duplicate clusters; không lọc để score tăng.
- Zero-shot test đã xem. Independent ở đây là **không dùng trained-method test chọn cấu hình**; không khẳng định corpus chưa từng xem. Nếu zero-shot/errors đã được dùng sửa method, công bố và chọn holdout mới cho strict confirmation.

**GO:** benefit trước FT/matched/tuned LSR, mean BLEU≥.30 trước LSR, ≥6/8 seed wins, mean chrF++/chrF0 no-harm; uncertainty được report. **NO-GO generality:** routing không bổ sung LSR hoặc quality giảm. Giữ kết quả âm. Đổi method dựa Turkish test thì cần holdout mới cho vòng confirm tiếp.

### NLLB distilled600M, Turkish20k

- Sau G4:5 families×5 seeds45–49 = **25 runs**; matching/tuned config trùng thì giảm5.
- Pin revision/tokenizer/language codes/forced BOS trước smoke; không reuse Amis NLLB proxy setting làm Turkish recipe.
- Batch fit A100 nhưng effective batch/token budget/steps/selection nhất quán trong NLLB families. LR/HPO có thể khác mBART, công bố budget.
- Default chuyển structural rule đã khóa; retuning theo backbone thì equal HPO budget và scope rõ.

GO khi routing benefit lặp ngoài mBART. Nếu fail, bỏ architecture-agnostic quality claim, có thể giữ mBART-specific. Portability của implementation và generality của quality là hai claims khác nhau.

### Direction thứ ba / reverse

Chỉ thêm khi RQ cần breadth hoặc decoder-side morphology. Chọn language/domain theo tiêu chí định trước, license/provenance/size/support model, không theo thử score. Mandarin→Amis là RQ khác, cần annotation. Budget/quyền chạy riêng; không khôi phục Asháninka.

## 9. G6 — novelty controls và baseline strength

### Routing comparators

Chọn **một comparator chính**, thứ hai chỉ nếu claim cần:

| Comparator | Câu hỏi | Budget |
|---|---|---:|
| Generic identity skip,d=1,attached +matched LSR | CLRR có ích ngoài một đường skip thêm? | 5 seeds/context;8 nếu main comparison đã khóa |
| Prior-inspired route/fusion +objective giống Full | Khác family routing gần ở đâu? | 5–8 seeds cho một comparator chọn từ G0 |
| Source-attached candidate | Detach còn role sau nâng cấp? | Old method đã có3 seeds; mới chỉ khi claim cần |

Adapt SkipCLM sang mBART encoder gọi **adapted control**, ghi giống/khác/counts, không gọi exact reproduction của XGLM decoder paper. Không có khác biệt hữu ích về quality/cost/mechanism trước prior/generic control thì đổi novelty narrative hoặc method.

### PEFT theo claim

CLRR vẫn full fine-tuning. Strong LoRA cũ chưa chứng minh PEFT failure/capacity ceiling.

- Claim chỉ trước FT/LSR: LoRA cũ supplementary có provenance; matched strong LoRA hữu ích nhưng không cần train mọi adapter.
- Claim trước PEFT: all-linear LoRA +**một** IA3/adapters/adaptive LoRA từ official implementation; report actual counts và budget.
- LoRA grid ví dụ: ranks8/16/32×LR1e-4/2e-4 =6 trials, embeddings frozen. Unfreezing là axis/cohort riêng; không đổi đồng thời LR để suy embedding effect.
- Equal opportunity cohort: mỗi family6 validation trials, step ceiling/early stop khóa; FT/Full/LSR có cùng quyền LR/tuning. Reuse old trial chỉ nếu dataset/protocol khớp; không tính Amis trials vào Turkish HPO.
- Ví dụ bốn families FT/Full/LoRA/IA3:24 development runs(seed53) +top config/family×5 fresh seeds=20 confirmation runs, **44 runs riêng**.
- Chốt cohort này là secondary; không âm thầm thay frozen-transfer matrix G4 bằng config thắng tuning.
- LR khác không tự unfair. Fixed-budget và individually-tuned studies là hai cách so; report tách.

Comparator tốt hơn thì report quality/cost/Pareto positioning thực tế; không hạ baseline để bảo vệ claim.

## 10. G7 — data regime và mechanism

### Learning curve

Sau G4 GO, lấy **nested5k subset** trong Turkish20k train, cùng dev/test.5 families×3 development seeds định trước = **15 runs**,20k point reuse.

Hai amounts chỉ hỗ trợ5k/20k. Nếu xu hướng khi data giảm là central claim, thêm10k×5 families×3=15 runs trong budget riêng, khóa trước test. Lưu subset IDs/hashes; không chọn sample thắng score. Pattern không lặp thì bỏ low-resource-specific claim nhưng report đầy đủ.

### Mechanism evidence

1. Đúng site/source/target, selected best và matched steps; lưu raw matrix/full spectrum.
2. Quality đi cùng residual distribution, CE/weighted LSR gradient groups, geometry; không chỉ figure cosine cao.
3. Intervention đổi đại lượng dự đoán và quality cùng direction qua seeds/context. Chỉ geometry đổi thì claim characterization.
4. Preserve-information claim cần task/challenge/perturbation có matched noise control; correlation rank→BLEU không là causal proof.
5. Anti-collapse cần operational definition, negative control có collapse, intervention đảo nó và task gain. Hiện chưa có; không viết proof.

### Human/morphology nếu giữ linguistic claim

- Regex mi/ma/pa chỉ là proxy; cần Amis/Chinese expert cho voice/morphology gold.
- Pilot30–50 items để thống nhất guidelines; sau đó fixed random sample khoảng200 items, stratify trước,2 raters, blind method labels; report agreement/uncertainty và adequacy/errors.
- Turkish challenge về negation/number/tense/case cần gold phù hợp. English không có cùng morphology hình thức; không đếm affix English để claim source preservation.
- Raters/data reuse/consent/community expectations chốt với user; không tự gửi dữ liệu đến external services. Không có expert thì bỏ claim chính, không dùng LLM labels làm gold.

## 11. Statistics và gate reporting

- Hai primary comparisons: **Full−FT** và **Full−best observed tuned LSR**. Full−matched LSR / Full−CLRR-only là component/secondary comparisons, vẫn report đủ.
- Primary metric: corpus BLEU (`zh` Mandarin,`13a` English). Native chrF++ là secondary reported metric; chrF0 là sensitivity guard cho Mandarin.
- Selection P0 vẫn native chrF++ beam1. Đổi reporting metric không cho phép chọn checkpoint mới từ test; đổi selection cần prospective cohort.

| Đơn vị | Phép đo | Cần tránh |
|---|---|---|
| Training seed | Paired corpus deltas,mean/SD,median/seed wins; paired seed bootstrap CI và exact randomization nếu assumptions hợp lệ |3 seeds×575 câu không là1725 training replicates |
| Corpus | Paired corpus bootstrap10000 shared indices,score lại full corpus | CI câu không thay seed uncertainty; không mean sentence BLEU |
| Seed +câu/cluster | Hierarchical resampling làm sensitivity, assumptions công bố | Không gộp với Holm p cũ hoặc gọi thành datasets độc lập |

Chọn test phù hợp metric/design; xem [Dror et al., ACL2018](https://aclanthology.org/P18-1128/). Các test/threshold cụ thể ở đây là đề xuất dự án, không là rule của nguồn tham khảo.

- Paired seed randomization cần exchangeability/symmetry dưới H0;8 pairs có256 sign patterns. Effect CI/n vẫn cần; exact p không là power guarantee.
- Primary family prospective G3+G4:2 comparisons×2 directions×BLEU = **4 tests** nếu global claim dựa cả hai. Giữ family kể cả condition âm.
- NLLB/G7/PEFT là replication/secondary families khóa riêng. Không chia family hậu nghiệm để p nhỏ; không trộn A42/B36/C24/final18/secondary30 lịch sử.
- Không mở test mỗi epoch, không thêm seeds đến significance. Fixed-budget stop/inconclusive hợp lệ.
- Superiority claim mạnh cần practical effect,uncertainty/corrected test hỗ trợ và replication; nếu CI chưa rõ thì report observed gain trong scope.
- No-harm guard không là equivalence test. Noninferiority/equivalence cần margins và test riêng trước chạy.

## 12. Fairness, vận hành và backup

- Data: `data_processed/<dataset_name>/`, CSV đúng`source,target`, hashes/order khóa.
- Local Python/tests/data dùng `conda run -n clrr ...`; Colab dùng isolated pinned runtime có receipt, không gọi remote venv là local conda.
- Init/tokenizer/model revision/lang prefix,optimizer/LR/dropout/WD/schedule,batch×accum/length/budget,selection/eval frequency/padding/precision/TF32 có receipt.
- P0 max20 epochs/patience4. Patience10/epochs40 phải đổi cả cohort, không kéo riêng Full.
- Actual α/λ,schedule,source cache pre/post,distance/layers,eval α explicit. Active PEFT weights khác registered frozen copies; counts report riêng.

### Hai A100 Standard, không high-mem

GPU1 seeds45/47/49/51; GPU2 seeds46/48/50/52. Families trong seed chạy theo order khóa/balanced order. Trước queue: smoke/reload/prediction replay, actual beam/dtype/hardware/permissions receipts.

Báo chat10 phút/lần khi monitoring active: model,epoch/step,last validation,done/total,upload,bất thường. Train xong chưa là COMPLETE. Monitor process/progress/timeout; protocol sai thì dừng dependent queue, crash giữ logs. Retry không đổi định nghĩa; failures/reruns giữ trong inventory.

### Artifacts và HF

Config/code/environment/model/data receipts,training args,step α/λ,histories,selection receipt,validation predictions,best weights,actual endpoint/snapshots,test predictions sau khóa,signatures,paired stats,raw mechanism NPZ/spectrum/probe JSON.

Namespace: `FiveC/amis-rewire-checkpoints/acl_<phase>_<date>/<dataset>/<model>/<method>/seed<seed>/`. Không đưa credentials vào artifact/log/docs.

Push best sau từng model; snapshots theo milestones. Verify SHA256/size và reload trước COMPLETE rồi mới xóa rolling weights đã backup. **Diagnostic ZIP không optimizer không là resumable checkpoint**; run dài cần optimizer/scheduler/RNG/global step theo policy riêng. Cuối queue sync scientific artifacts/receipts, tắt Colab theo vận hành đã chốt; không xóa kết quả âm hoặc weights chưa verify.

## 13. G8 — efficiency, bảng paper và review

### Benchmark

FT/CLRR-only/LSR-only/Full cùng A100,batch/length/dtype/beam; warmup hardware,CUDA synchronization,≥3 repetitions,median/spread. Đo riêng:

1. Training forward/backward/optimizer: tokens/s và peak allocated/reserved memory.
2. Inference encoder/generation: p50/p95 latency,throughput,memory,beam1/4 và length buckets.
3. Optimizer/disk footprint và CLRR activation cache.
4. Exclude HF transfer; LSR target pass chỉ có lúc training.

Practical target: CLRR-only inference overhead≤5% mean latency trong workload định trước. Cao hơn thì report actual cost, không claim negligible. Quality hữu ích với cost cao vẫn có thể đáng dùng, nhưng narrative efficiency phải đổi.

### Bộ bảng/figure đích

| Vị trí | Nội dung | Claim |
|---|---|---|
| Table1 main | FT/CLRR/LSR matched/LSR tuned/Full,Amis+Turkish,mean±SD/deltas | Quality/incremental benefit |
| Table2 main | NLLB replication và routing comparator | Generality/novelty |
| Table3 main | New rule/objective/gradient/distance ablation đã test | Vì sao method mới hữu ích |
| Table4 main/figure | Train/inference cost và parameter counts | Practicality |
| Figure1 | Computation/gradient paths/anchor policy | Khác biệt method |
| Figure2 | Quality và mechanism qua epoch/site/intervention | Insight có giới hạn |
| Figure3 nếu claim cần | Nested5k/20k curve,uncertainty | Data regime |
| Appendix | Every trial/seed/tests/signatures/hashes/invalidations/human rubric | Reproducibility |

Paper hiện có **10 table environments**, gồm appendix (đếm trực tiếp `table`/`table*` trong `clrr_main.tex`); không phải10 experiments độc lập. Gom theo claims, không tăng số bảng cho đẹp.

### Review trước submission

- Main quality có ích; baseline/HPO mạnh; CLRR phân biệt so LSR; failed trials được lưu.
- Novelty đối chiếu priors; adapted control không gọi reproduction.
- Test access/scope đúng;0 added parameters không biến thành0 trainable/memory.
- Limitations/negative findings ghi đúng. [ARR Reviewer Guidelines](https://github.com/acl-org/aclrollingreview/blob/main/reviewerguidelines.md) phân biệt soundness/excitement; không che uncertainty bằng câu chữ.
- Giữ ACL style/pdfLaTeX của user,citations đủ,tables từ canonical manifest,compile không phụ thuộc missing `paper_tables`.
- CFP/template/Responsible NLP/deadline verify ở cycle đích. [ACL2026 CFP](https://2026.aclweb.org/calls/main_conference_papers/) là tham khảo2026; không dùng deadline đã qua cho kế hoạch này.
- Review bởi người khác nếu có; cùng trợ lý đã sửa không gọi blind independent panel. Không upload unpublished paper đến external model nếu chưa ủy quyền.

GO_SUBMISSION_REVIEW khi main story rõ và claims đủ support; không đợi mọi bảng phụ positive. Historical mT5/chrF++ subset không thay incremental CLRR evidence còn thiếu.

## 14. Budget và lịch

| Khối | Maximum training runs | Khi nào chạy |
|---|---:|---|
| G0/G1 |0|Lượt đo được ủy quyền |
| G2 hai candidates,objective unchanged |12|Cards qua G1;reuse có thể giảm |
| G3 Amis8 seeds,6 families |48|Một candidate qua G2 |
| G4 Turkish8 seeds,5 families |40|G3 có ích và user cho Turkish |
| **Core path subtotal** |**100**|Trần có gate; không mặc định chạy hết |
| G5 NLLB5 seeds,5 families |25|Giữ backbone claim |
| G6 một/hai routing controls×5 seeds |5–10|Novelty cần;8 seeds là budget khác |
| G7 Turkish5k,5 families×3 seeds |15|Data-regime claim |
| **Breadth path maximum với hai controls** |**150**|100+25+10+15; chưa PEFT/third language |
| Optional matched PEFT |44|24 development+20 confirmation |

Trần giả định matching/tuned LSR khác config, không reuse CLRR mới, không có bug rerun hoặc objective controls ngoài budget. U1 có thể reuse CLRR-only; G2 fail dừng trước100. **Không cần chạy100/150 để nộp ACL.** Đây là đề xuất độ tin cậy cao lớn hơn budget pilot cũ, chưa đưa vào autorun queue. Mỗi phase chốt cards/protocol/run count trước; không hỏi lại từng run trong phase đã ủy quyền.

Warmup Amis vừa chạy cho mốc khoảng20–35 phút/run gồm các bước sau train, không là isolated GPU training time. Không nhân thẳng cho Turkish:20k/4600≈4.35 pairs nhưng tokens/eval/probes/transfer khác. Đo năm epochs đầu của run được ủy quyền,throughput và eval/upload/probe time để cập nhật ETA.

Ước lượng phase walltime≈tổng(train+eval+required probes+upload)/số GPU hoạt động,cộng setup/transfer. Walltime khác GPU utilization. Tuần1 gợi ý G0/G1/cards,tuần2 G2/decision; G3/G4/G5 xếp sau benchmark/gate, chưa hứa lịch submission khi chưa có cycle đích.

## 15. Điểm dừng và bước sau NO-GO

| Điều xảy ra | Quyết định | Hướng sau đó |
|---|---|---|
| Artifact/protocol sai |TECHNICAL_HOLD|Sửa/verify affected controls,giữ số cũ |
| Warmup NO-GO |Đã xong,loại20% schedule|Không bỏ CLRR chỉ từ một candidate |
| Hai upgrades không vượt LSR mạnh |NO-GO quality route|Empirical/mechanistic RQ mới hoặc method mới |
| Pilot positive,fresh seeds inconclusive |NO-GO strong superiority|Scoped observed gain;không thêm seed để p đẹp |
| Amis positive,Turkish negative |NO-GO generality|Context-specific;report Turkish đầy đủ |
| mBART positive,NLLB negative |NO-GO broad backbone claim|Backbone-specific narrative nếu có ích |
| Quality positive,mechanism sai |Giữ quality,bỏ causal claim|Tìm explanation khác |
| Priors bao phủ method,increment nhỏ |NO-GO novelty narrative|Insight/condition analysis hoặc new method |
| Hết budget,đóng góp vẫn không mạnh |NO-GO ACL main positioning|User chọn đổi RQ/method/venue;không hứa accept |

## 16. Việc đề xuất làm ngay

**Chỉ G0+G1 trước, chưa mở training.**

1. Canonical inventory và paper claim ledger; successor V5/warmup v2 rõ,old tables bất biến.
2. Đọc closest prior formulas/code,đặc biệt SkipCLM/generic fusion; một trang novelty/control choice.
3. Audit warmup/fixed histories gắn đúng weights; sáu validation generations on/off mới nếu cần.
4. D3 nếu có câu hỏi rõ: ba Full best anchor conditions trên8×4 fixed validation batches,label counterfactual.
5. Report `go_nogo_g1.md`: facts/inferences/unknowns,U1/U2 cards và exact G2 budget. Chưa có cards được chốt thì không launch12 trials.

Đầu ra: chọn được giả thuyết test được, hoặc báo **chưa đủ căn cứ nâng cấp method**. Đây là decision point, không là chạy đủ mọi diagnostics.

## 17. Nguồn và thứ tự đọc

1. [Warmup v2 results](AMIS_WARMUP_V2_RESULTS.md) và [audit](../../../../outputs_rebuttal/amis_routing_warmup_20261004_v2/independent_audit_summary.json).
2. [Experiments updated](../../../experiments/EXPERIMENTS_UPDATED.md),mục10–12 và historical scope.
3. [Handoff](../../../operations/CONFIGURATION_SEARCH_HANDOFF.md),mục18–19,receipts/corrections.
4. [Metrics](../../../experiments/METRICS.md),định nghĩa và scope thống kê.
5. [Research insights](RESEARCH_INSIGHTS.md): giữ insight lịch sử,không dùng pending status cũ thay new results.
6. [Paper](../../../paper/clrr_main.tex): format/claims hiện hành;lag ghi ở G0 ledger.
7. [Original experiments](../../../archive/EXPERIMENTS_ORIGINAL_20261003.md): số paper ban đầu,lưu bất biến.

**Kế hoạch này thay các đoạn “tiếp theo” đã cũ cho đường dài sau warmup v2; không thay scientific results,metric protocol hoặc quyền user.** Launch từng phase cần handoff cụ thể sau khi gate card/protocol được chốt.
