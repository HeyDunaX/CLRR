# Amis routing warmup v2 — kết quả và quyết định

**Hoàn tất:** 04/10/2026, 21:48 (Asia/Saigon). **Quyết định: NO-GO cho schedule warmup 20%.**

Đủ 6/6 model, ba seeds 42/43/44 cho Full và CLRR-only, chạy song song trên hai A100 40GB Standard. Cả hai hàng đợi COMPLETE. Không chạy Turkish, không mở thêm grid, không đánh giá test cho ứng viên không qua gate. Các kết quả cũ được giữ nguyên; báo cáo này chỉ chứa v2.

## 1. Protocol và kiểm chứng

- mBART-50, Amis→Mandarin; α tăng 0→0.10 trong 144/720 optimizer steps, sau đó cố định. Evaluation luôn α=0.10. Full λ=0.10; CLRR-only λ=0.
- Giữ LR=5e-5, batch32×accum4, maximum20epochs, patience4 và protocol controls. BF16 training; FP32 generation, TF32 off. Model/tokenizer revision và code/data hashes được khóa trước training.
- Chọn best bằng validation **native chrF++**, beam1; số dưới đây là **validation**, không phải test hay số để thay bảng test của paper.
- Runtime isolated: torch2.6.0+cu124, transformers4.57.6, sacrebleu2.6.0. Hai máy qua smoke, replay controls và gate trước training. Namespace v1 đã invalidated; không trộn vào bảng.
- Sáu COMPLETE receipts đều có best reload PASS, trajectory probes PASS và HF best archive SHA256/size verification. Tải về 180 scientific text records; tính lại BLEU/chrF++/chrF0 độc lập từ predictions, kiểm tra hash và đúng576 hàng validation: **PASS cả sáu**. Probe JSON hashes cũng khớp receipts.
- Có24 snapshots (epoch1/4/10/actual final) và36 điều kiện probe: tách α evaluation cố định với α training cuối snapshot ở giai đoạn warmup. CLRR-only seed43 dừng epoch12 theo patience; các run còn lại train20epochs.

## 2. Số từng seed

| Nhánh warmup | Seed | BLEU | chrF++ | chrF0 | ΔBLEU so fixed cùng seed | ΔchrF0 so fixed |
|---|---:|---:|---:|---:|---:|---:|
| Full | 42 | 17.7908 | 14.9599 | 17.0269 | +0.1472 | +0.1517 |
| Full | 43 | 16.2231 | 13.8586 | 15.9132 | −0.8414 | −0.6746 |
| Full | 44 | 17.0006 | 14.3394 | 16.5026 | +0.0681 | −0.0101 |
| CLRR-only | 42 | 16.6623 | 13.9601 | 16.0884 | +0.3968 | +0.2400 |
| CLRR-only | 43 | 14.4003 | 12.5549 | 14.5335 | −2.5095 | −1.7136 |
| CLRR-only | 44 | 16.8898 | 14.2423 | 16.3599 | −0.3840 | −0.1595 |

## 3. Tổng hợp ba seeds và gate đã khóa

| Cấu hình | BLEU mean | chrF++ mean | chrF0 mean |
|---|---:|---:|---:|
| Fixed Full control | 17.2135 | 14.4975 | 16.6586 |
| **Warmup Full** | **17.0048** | **14.3859** | **16.4809** |
| Tuned LSR-only, λ=.03 | 16.9942 | 14.3170 | 16.4950 |
| Fixed CLRR-only control | 16.8164 | 14.0765 | 16.2050 |
| **Warmup CLRR-only** | **15.9842** | **13.5858** | **15.6606** |

Sample SD warmup Full: BLEU0.7839, chrF++0.5521, chrF00.5572. Warmup CLRR-only: BLEU1.3763, chrF++0.9038, chrF00.9855. Ba seeds là pilot nhỏ; không tuyên bố statistical significance từ các chênh lệch này.

Warmup Full so fixed Full: **−0.2087 BLEU, −0.1116 chrF++, −0.1777 chrF0**. So tuned LSR: **+0.0106 BLEU, +0.0689 chrF++, −0.0141 chrF0** — gần hòa, chưa có bằng chứng tăng ích ổn định.

**Cả năm điều kiện gate đều không đạt:** BLEU+0.30; chrF+++0.30; no-harm chrF0; ≥2/3 seeds cùng tăng BLEU vàchrF0 (chỉseed42 đạt); và cả ba mean vượt tunedLSR. Đây là gate nội bộ, không phải quy định ACL.

## 4. Insight từ đo cơ chế

### Warmup này chưa giải quyết vấn đề

Seed42 tăng nhẹ nhưngseed43 giảm ở cảhai nhánh; CLRR-only mean giảm0.8323BLEU. Full giảm ít hơnCLRR-only, phù hợp với khả năng LSR bù một phần suy giảm, nhưng chênh lệch hai nhánh không tự chứng minh nguyên nhân. Không chọnseed42 để quảng bá warmup, và không chuyển sang Turkish để cứu một cấu hình chưa qua gate Amis.

### Alignment tăng không đồng nghĩa dịch tốt hơn

Ở Full, paired cosine tại **encoder final LayerNorm** tăng từ khoảng0.44–0.46 (epoch1) lên0.7966–0.7977 (epoch20). CLRR-only ở endpoint thấp hơn rõ rệt. LSR có tác động tới representation alignment, nhưng Full warmup vẫn gần hòa tunedLSR về quality. Vì vậy cosine cao chỉ là bằng chứng cơ chế alignment, chưa là bằng chứng CLRR tăng chất lượng dịch.

### Chưa có dấu hiệu rank collapse tiến triển trong các snapshots đã đo

Full final-LN source covariance effective rank cuối training khoảng138.6–140.5, cao hơn đầutraining khoảng108–115; target rank cuối khoảng140.6–143.1. Không có trajectory giảm rank xuyên suốt để quy kết collapse là nguyên nhân chất lượng thấp. Đây không phải chứng minh không collapse ở mọi vị trí/token/không gian; số đo dùng pooled representations576 câu, centered spectrum và rank tối đa575.

### Không có bằng chứng LSR gradient lấn át CE trên toàn bộ parameter union

Full all-parameter weightedLSR/CE gradient norm ratio khoảng0.39–0.42% ởepoch1 và0.16–0.18% ởepoch20; cosine gradient cuối khoảng0.055–0.056. Không nên kết luận λ=.10 quá lớn hoặc CE–LSR xung đột mạnh toàn mô hình. Tác động encoder từnglayer cần đọc riêng, vì union gồm decoder và LSR không có gradientdecoder. CLRR-only được đo **LSR counterfactual λ=.10**, không phải loss thực tế đãtrain.

### Residual không phình dần rõ rệt

Full source residual/current RMS trung bình ởαeval=.10 khoảng5.76% đầutraining và5.89–5.90% cuối. Đại lượng này là trung bình chẩn đoán trên các layer córewiring; không phải bound từnglayer/câu. Chưa có bằng chứng residualnorm explosion để tự động normalize/rescale phương pháp.

### Có vấn đề generalization/selection cần xác minh trước nâng cấp

Native CE token-weighted trên **cùng32 validationexamples** xuống khoảng2.77–2.79 ởepoch4 rồi lên2.92–2.97 ởepoch20 trongFull. chrF++ vẫn có thể chọncheckpointmuộn (Full44 chọnepoch18). Đây cho thấy CE vàgenerationmetric có trajectory khác nhau; chưa đủ để đổi selectionmetric riêng choCLRR. Loggedtrainingloss không dùng để so trực tiếp vì khácaggregation vàauxiliaryloss.

Các gradient là **offline BF16 diagnostic**, TF32off, dropout có kiểm soát vàcheckpointingdisabled; không phải bản sao từngbackward củatraining. Snapshots cuối không luôn làselectedbest (Full44bestepoch18, snapshotcuốiepoch20). Không lấyendpointmechanism làm số đo trực tiếp củabest nếu khôngloadbest đểđo.

## 5. Việc tiếp theo có cơ sở

1. **Giữ fixed routing là đối chứng; loại warmup20% khỏi danh sách ứng viên ưu tiên.** NO-GO này áp dụng cho schedule vừa thử, chưa chứng minh mọi biến thểCLRR vô ích. Hiện chưa đủ incremental evidence để hứa claim vượt tunedLSR choACL.
2. Trước training mới, làm **validation-only matched inference ablation** trên đúng selectedbest Full/CLRR-only/tunedLSR của ba seeds: α0/.10, cùngweights, cùngbeam1 và diagnosticbeam4; đoquality/nativeCE vàmechanism tạibest. α0 tắt routing lúcinference là counterfactual có distribution shift; không gọi đó là mô hìnhtrainkhôngCLRR. Mục tiêu: phân biệt routing tạiinference, quá trìnhtraining và beam sensitivity.
3. Lấy **fixed-control epoch histories** nếu có và đối chiếu theoepoch/step vớiwarmup, nhất làseed43. Thiếuhistories/snapshots thì ghi rõ, không tạo bằng chứng thay thế bằng endpointweights.
4. Nếu routing tạibest làmquality giảm nhất quán nhưng alignment/rank vẫn ổn, ưu tiên **layer-selective routing** dựa vào residual/perlayer evidence. Nếu vấn đề tập trung optimization/seed sensitivity, thiết kế một thay đổi optimization có kiểm soát. Không tăng/giảmλ chỉ vì BLEUchưađẹp; đo hiện tại chưa chứng minh gradient imbalance.
5. Chỉ sau khi khóa **một giả thuyết và một thay đổi** mới chạy lại bốnnhánh matched Baseline/CLRR-only/LSR-only/Full, ≥3seeds, cùng budget và gate validation. Gatefail thì báo thực chất lợi ích, không mở grid vô hạn. Turkish đểsauquyếtđịnh củauser.

## 6. Nguồn và checkpoint

- Local nguồn mới: `outputs_rebuttal/amis_routing_warmup_20261004_v2/`.
- `independently_verified_validation_scores.csv`: sáusốchạyđãtínhlại.
- `independent_audit_summary.json`: means/SD và tấtcảgatebooleans.
- `mechanism_trajectory_summary.csv`:36điềukiệnprobe, táchαmode.
- `collected_records/runs/`: predictions, histories, selection, snapshot/HF/probe receipts.
- HF: `FiveC/amis-rewire-checkpoints`, prefix **`amis_routing_warmup_20261004_v2/<run>/`**; sáubestarchives và24trajectoryarchives ởnamespace riêng. Bestarchive vàtrajectoryarchive đượccheckSHA/size trước COMPLETE; trajectoryweights không chứaoptimizer, khôngphải resumabletrainingcheckpoint.

Không thay sốpaper/EXPERIMENTS cũ bằng những sốvalidation này.
