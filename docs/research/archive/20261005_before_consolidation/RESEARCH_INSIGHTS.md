# CLRR — insight từ thực nghiệm đã xác minh

**HIỆN HÀNH — V5 đã đo và audit, 04/10/2026:** xem thí nghiệm Mục12 và bàn giao19.4–19.5.
42 điều kiện Full/LSR và các đối chứng baseline đã xác minh; A100 Standard đã tắt.

**Có lợi:** routing có tác dụng trên weights đã train; tắt từng nhóm layer không
tạo cải thiện trung bình. Cả ba baseline archives tái lập đủ575câu test cũ,
không có bằng chứng nhầm weights. Sửa nhãn beam của Baseline seed42 giúp
Full−Baseline validation mean đổi từ−.4498 thành+.0861BLEU, nhưng chỉ1/3seed dương.

**Không có lợi cho claim superiority:** bốn α khác và ba group ablations đều có
mean BLEU thấp hơnα.10/routing đầy đủ; chrF++/chrF0 giảm ở3/3seeds trong cả bảy
điều kiện. Không α nào qua guard không làm giảm các metric. Full−tuned LSR mean
BLEU: validation beam1+.2193/beam4+.2055; test beam1−.2168/beam4−.1267.
chrF0 test cũng âm ở cả hai beam. Chênh lệch không chỉ do beam; chưa xác định
overfitting/exposure bias là nguyên nhân. α.125/.15 giảm CE trên32câu cố định
nhưng làm điểm dịch giảm: CE subset và rank không thay cho đánh giá generation.

**Số sửa cần nhớ:** Baseline seed42 validation17.8735BLEU/14.9009chrF++ thực tế
là beam4; beam1 đúng16.2659/13.9015. Mean Baseline validation beam1=17.1275BLEU,
không phải17.6633. Test scores cũ giữ nguyên. Best step216 thiếu provenance:
chuyển sang unknown. Source recovery và F0 metadata đã sửa; lượt sau dùng
corrected V5 inventory riêng. Old CSV/inventory và các lần lỗi giữ để truy vết.

**Cơ chế chưa xác định:** V4 endpoints chưa cho thấy residual explosion hoặc
LSR domination.35probe lịch sử có gradient union/raw spectra tái tính đúng;
các thống kê mô tả rank tăng và relative auxiliary gradient giảm theo epoch.
Historical executable/runtime/language provenance còn thiếu, nên chưa coi đó
là chứng minh nhân quả hoặc xác minh đầy đủ trajectory training.

**Quyết định:** cần pilot thay đổi cách train để cải thiện quality/stability;
chưa đủ căn cứ bỏ CLRR hoặc sửaλ/normalization/gradient path tùy ý. Đề xuất
routing warmup cho Full và CLRR-only, mỗi nhánh ba seeds; controls và GO/NO-GO
ở19.5. **Chưa code/chưa chạy pilot; Turkish vẫn ngưng.** Chọn cấu hình bằng
validation đã khóa rồi confirm test và replication trên dữ liệu chưa xem.
Không hứa ACL hoặc chọn cấu hình bằng test.

**LẦN TRƯỚC — V4 đã thực thi và audit VERIFIED, 04/10/2026:** xem bàn giao
**Mục 18** và thực nghiệm **Mục 11**. Đã chạy A100 Standard, không high-mem;
21/21 endpoints,3/3 route-controls; tải artifacts/SHA và tắt Colab. CPU audit
trong `clrr` tái chấm/replay, tái SVD84endpoint matrices và kiểm tra gradient/
residual/provenance đều PASS. Không train model mới hoặc sửa test scores cũ.

**Có lợi cho phương pháp:** route-on tái lập đúng576predictions cũ ở cả3seeds;
on−off BLEU=+3.1060/+3.0783/+2.7187, mean **+2.9676**, validationbeam1.
Rút kết luận routing vô tác dụng. Đây là dependence của weights đã train với
routing, **không phải +2.97BLEU trước independently trained baseline/LSR**.

**Insight rank mới:** Fullλ0.10−matchedLSR có block12source singular rank tăng
15.83% (dương3/3), nhưng sau finalLN singular rank385.2893 so với386.3290
(âm3/3). Final energy entropy rank tăng5.21% (dương3/3). Không có một kết luận
“rank luôn tăng” theo mọi definition/site, không chứng minh chống collapse gây
ra quality gain. CLRR-only cũng có final singular rank thấp hơnbaseline3/3.

**Gradient/residual:** Fullλ0.10 weightedLSR/CEnormmean0.1570%, globalcos0.0516;
có13/288layer×batchcosinesâm, nên rút câu “không conflict” tuyệt đối. Residual
RMSratio max mọiroutingbranches/phase8.7783%; chưa thấy endpoint explosion,
không phải bảo đảm mọiphase vô hại. Không có căn cứ sửa loss/normalization ngay.

**Quality gate chưa đổi:** Full−matchedLSR+0.2344BLEU (3/3 dương), nhưng
Full−tunedLSR−0.1267; Fullλ0.03−tunedLSR−0.8100. Validation Full−tunedLSR
mean+0.2193BLEU trong inventory không chuyển thành test gain. Cần hiểu
generalization/selection; không dùng rank/alignment để thay translation quality.

**Tiếp theo đề xuất:** error analysis bằng predictions đã có; nếu cần,
validation-only inferenceαsweep12generations mới (3seeds,αchung), rồi routing
group intervention có điều kiện9generations. Đây là diagnostics/tuning mới,
chưa chạy; chỉ chọn pilot architecture/schedule khi có quan sát phân biệt,
không thêm λgrid hoặc mở Turkish tìm số thuận lợi. Xem18.5 để biết controls,
selection và phạm vi exploratory. Receipt:
`outputs_rebuttal/amis_followup_20261004_v4/independent_audit.json`.
Các block V3/V2 bên dưới là lịch sử và đã được thay thế trong phạm vi đo V4.

**Review V3 trước khi chạy V4, 04/10/2026 — xem bàn giao Mục 17.** CSV route-control
khớp phép chấm, nhưng route-on không tái lập validation trước đó (thấp hơn
1.6971–3.1833 BLEU). Source local và bundled worker thiếu `tl_XX/zh_CN`;
runtime tokenizer thực tế chưa được lưu để giải thích đầy đủ chênh lệch.
F4 đã sửa cách tháo hooks, nhưng chưa dùng “89% câu đổi” như evidence cùng
protocol. Chưa dùng rank layer12 cao hơn làm claim chống collapse: inputs cần
xác nhận, site trước final LayerNorm khác V2, covariance rank field thực ra là
participation ratio. Gradient accumulator bỏ decoder CE ở baseline/CLRR-only
(48 batches), nên global scope giữa controls chưa nhất quán.

**Quality đã xác minh giữ nguyên:** Full λ0.10 hơn matched LSR +0.2344 BLEU
(3/3 seeds), nhưng thấp hơn tuned LSR −0.1267; Full λ0.03 thấp hơn tuned LSR
−0.8100 (3/3 seeds). Có tín hiệu để thử cải tiến, chưa đạt gate mở rộng.
Kế hoạch: hai checkpoint sentinel kiểm tra protocol → 6 on/off generations
và 21 endpoint probes đã sửa, không train → khóa giả thuyết → pilot/confirmation
có controls phù hợp. Không mở grid hoặc Turkish trước gate. Receipt riêng:
`outputs_rebuttal/amis_followup_20261004_v2/v3_independent_review_20261004.json`.
Các block review V2/final-test bên dưới là lịch sử; ưu tiên Mục 17 khi bàn giao.

**Review follow-up mới nhất04/10/2026 — bàn giao mục16:** tái tính21test sets,
reference/order/SHA và scores khớp (maxerror≈1e-14); đúngsecondary30pairs/Holm,
nativeSacreBLEUp khớp choBLEU/chrF++ ởFullλ0.03−tunedLSRseed42.
Fullλ0.03 thua tunedLSR cả3seeds: mean−0.8100BLEU/−0.6702chrFcharacter.
Giảmλ đơn thuần chưa giúp; originalFull vẫn−0.1267BLEU trước tunedcontrol.
**Phát hiện F4 INVALID:** haiwrappers chungbase_model khiến hookα0.10 vẫn hoạt động
trongroute-off. Đã tái hiệnCPU; không dùng0câuđổi để kết luận routing vô tác dụng.
F3 có21endpoints/batchtables hợp cấu trúc nhưng thiếu per-layer gradients/ranks,
rawspectra/representations và runtime/loadingprovenance. Pooledfinalrank lớn không
chứng minh anti-collapse toànencoder. NO-GO mởrộng giữ nguyên theoquality;
chưa buộc chuyểnpaper sangmechanistic study hoặc đổi phươngpháp trước sửaF4.
Receipt riêng: `outputs_rebuttal/amis_followup_20261004_v2/independent_review_20261004.json`.
Các đoạn“secondarychưađủ” bên dưới là trước lầnreview này; familyđúng hiệnđãcó.

**Kết luận mới nhất04/10/2026 sau final test:** đã có17runs mới và18models test.
Review CPU độc lập trong conda `clrr` xác minh cả18prediction sets: raw575rows,
source/target/index, SHA256 và cả3metrics đều khớp (sai số<1e-8).
Xem [bàn giao mục13](../../../operations/CONFIGURATION_SEARCH_HANDOFF.md#13-review-độc-lập-sau-final-test--04102026)
và receipt `outputs_rebuttal/amis_final_evaluation_20261004/final_review_20261004.json`.

Full−baseline testmean+0.3800BLEU/+0.3263chrFcharacter;
Full−matchedLSR+0.2344BLEU(3/3seeds)/+0.1675chrFcharacter.
Nhưng Full−tunedLSRλ0.03 **−0.1267BLEU/−0.1635chrFcharacter** (thua2/3seeds).
Gain+1.3642chrF++ trước tunedLSR rất nhạy wordbigram: chỉ bỏ row170 để chẩn đoán
thì mean gap thành−0.1832; giữ nguyên mọi575rows cho score chính.
Không gọi tunedLSR seed43 collapse hoặc suy detach tối ưu từquality/sample std.
Không p-value primaryHolm18 nào đạt0.05; CI lưu là chưa điều chỉnh.
FilesecondaryHolm30 hiện dùng pairs khác kế hoạch, thiếu interactionλ0.03
và CLRR-only−baseline, nên cần hoàn tất đúng family ở artifacts mới.

**NO-GO mở rộng/claim superiority theo gate hiện hành**, chưa kết luận phương pháp vô ích.
Tiếp theo: hoàn tất controls không train → probe validation CE–LSR/residual/rank
→ chọn candidate có cơ sở và xác nhận trước mở rộng. Chưa chạy GPU/Turkish trong review.
Các cập nhật screening/confirmation dưới đây là lịch sử, không phải trạng thái cuối.

**Bàn giao chạy tiếp sau kiểm tra source:** xem mục14 của file bàn giao. Vòng kế
tiếp không train: chấm3Fullλ0.03, hoàn tất secondary family đúng30tests, probe21best
trên validation và route-on/offFull3seeds. Đã kiểm tra đủ35trajectory probesC cùng
8batches; global weightedLSR/CEnorm ở Full nhỏ và residual ratio max khoảng0.0868.
Chưa có căn cứ tăngλ hoặc normalization ngay. Các probe entries hiện cần adapter
cho λ/split/detach/output trước chạy; đây là kế hoạch, chưa mởGPU.

Ngày lập: 03/10/2026. Đi cùng [EXPERIMENTS_UPDATED.md](../../../experiments/EXPERIMENTS_UPDATED.md)
và [paper hiện hành](../../../paper/clrr_main.tex). File này diễn giải bằng chứng; sổ kết quả và
artifact gốc vẫn là nguồn số. Không có phép đo hoặc lượt training mới trong lần lập file này.

**Cập nhật04/10/2026 — screening D đã xong8/8:** xem [phân tích bàn giao, mục9](../../../operations/CONFIGURATION_SEARCH_HANDOFF.md#9-phân-tích-hiện-hành-và-quyết-định--04102026)
và receipt `outputs_rebuttal/amis_configuration_20261003_v1/screening_review_20261004.json`.
Tái tính8sets validation đạt; không có test mới hoặc training trong lượt phân tích này.
Best Full vẫn là gốc α0.10/λ0.10 (14.7317chrF++), không phải Full α0.20 mới (14.5738).
Best LSR λ0.03 đạt14.8587, baseline C seed42 đạt14.9009; Full chưa vượt hai mốc này.
Full−matchedLSR0.10 +0.5643chrF++, nhưng Full−bestLSR −0.1271;
gain matched-control chưa đủ bảo vệ contribution riêng qua tuning.
LSR0.03 đứng đầu8trial mới trong mọi576leave-one-out cả chrF/chrF++; ranking thấp hơn
có thể đổi nên không claim toàn ranking bền. Các điểm dừng/phân tích dưới đây là lịch sử.
Kế hoạch mới ưu tiên độ tin cậy theo tác giả: audit/reuse → phục hồi2replicas seed42
và xác nhận LSR0.03 seeds43/44 (4train) → interaction/detach controls (5train)
→ attached CLRR-only3train nếu architecture đó được chọn. Chưa khởi động GPU.
Hiện **chưa đủ vững cho claim CLRR cải thiện rõ**, nhưng chưa kết luận phải bỏ phương pháp.

**Sau confirmation6runs,04/10:** [bàn giao mục10](../../../operations/CONFIGURATION_SEARCH_HANDOFF.md)
đối chiếu Fullλ0.10 với tunedLSR0.03 đúng3seeds: mean+0.2193BLEU/+0.1805chrF++,
2/3seeds dương. Full hơn matchedLSR0.10+0.6705BLEU/+0.4368chrF++,3/3dương;
nhưng vẫn thua baseline validation mean−0.4498BLEU, BLEU giảm3/3seeds.
Đây là tín hiệu có lợi cho routing so với controls, chưa là superiority chắc chắn.
Fullλ0.03 gần hòa LSRλ0.03; sample std thấp ở3seeds không chứng minh representation stability.
Replicas42 khớp validation C, hỗ trợ metric reproducibility; không chứng minh tất định tuyệt đối.
Đã thấy4B2+2interaction, chưa thấy3no-detach runs hoặc final candidate test.
Giữλ0.10 làm selected Full để kiểm chứng tiếp; chưa có căn cứ đổi architecture hoặc tăng claim paper.

## 1. Bản cũ và phạm vi hiện hành

- [Insight cũ đã archive](../../../archive/experiment_reports_20261003/RESEARCH_INSIGHTS.md)
  được giữ nguyên để truy vết. Những nhận định lịch sử về bảo toàn morphology,
  chống collapse, gradient explosion hoặc PEFT overfitting không tự trở thành
  kết luận của bản mới.
- [Phiên bản insight sớm hơn](../../../archive/RESEARCH_INSIGHTS_v1_with_layerskip_byt5.md)
  và [phân tích paper cũ](../../../archive/PAPER_EXPERIMENT_INSIGHTS.md) chỉ là lịch sử.
- Phạm vi hiện hành: Amis→Mandarin. Không đưa Asháninka hoặc LayerSkip trở lại.
  ByT5 vẫn thuộc các đối sánh được giữ trong paper hiện hành.
- [METRICS.md](../../../experiments/METRICS.md) quy định cách đo;
  [CLRR_GO_NOGO_PLAN.md](CLRR_GO_NOGO_PLAN.md) quy định các gate phát triển.

## 2. Ba nhóm bằng chứng phải đọc riêng

| Nhóm | Đã có | Có thể trả lời | Giới hạn |
| --- | --- | --- | --- |
| A | 28 outputs lịch sử Amis, chấm lại saved predictions | Điểm của các cấu hình đã chạy với raw references | Training và inference lịch sử không matched |
| B | 10 selected checkpoints generate FP32 và 4 routing-off outputs | Điểm dưới inference chung; phụ thuộc routing của cùng weights | Không sửa khác biệt training/selection lịch sử |
| C | 12 lượt mBART, bốn nhánh × seeds 42/43/44; 35 trajectory probes | Đóng góp thành phần dưới protocol chung; hành vi trong training | Một direction/backbone, ba seed; endpoint không nhất thiết là best |

Không gộp A/B/C thành một mean/std hoặc kiểm định. Giữ các family Holm gốc
A42/B36/C24. Sentence bootstrap không đo biến thiên giữa các lần train.

## 3. Tín hiệu chính từ suite matched C

Nguồn: `outputs_rebuttal/amis_confirmation_20261003/seed_summary.csv`,
`paired_seed_deltas.csv`, `seed_scores.csv` và `paired_bootstrap24.csv`.

| Phương pháp | BLEU mean ± sample std | chrF++ mean ± sample std |
| --- | ---: | ---: |
| Baseline | 19.3886 ± 1.1203 | 16.9249 ± 3.2626 |
| CLRR-only | 19.5769 ± 0.6057 | 18.5172 ± 0.4136 |
| LSR-only | 19.5343 ± 0.2325 | 18.5639 ± 0.1745 |
| Full | 19.7687 ± 0.1743 | 18.6565 ± 0.2107 |

| So sánh | Mean ΔBLEU | Seed dương | Diễn giải hiện tại |
| --- | ---: | ---: | --- |
| Full−Baseline | +0.3800 | 2/3 | Gain trung bình dương, có seed giảm 0.5629 |
| Full−LSR-only | +0.2344 | 3/3 | Tín hiệu hữu ích nhất cho đóng góp routing trong tổ hợp |
| CLRR-only−Baseline | +0.1883 | 1/3 | Chưa chứng minh routing đứng riêng cải thiện ổn định |
| Full−CLRR-only | +0.1917 | 2/3 | Chưa chứng minh kết hợp luôn tốt hơn routing đơn |

**Có lợi:** Full có mean BLEU cao nhất và hơn LSR-only ở cả ba seed.
**Còn yếu:** không kiểm định nào trong Holm24 đạt p<0.05; ba seed chưa đủ
chứng minh tính ổn định tổng quát. Full có std nhỏ nhất là mô tả của ba lượt này,
không phải bằng chứng đã giảm variance của quá trình training.

Mean Full−Baseline chrF++ là +1.7316, phần lớn đến từ seed43 (+5.3162).
Mean Full−LSR-only chỉ +0.0926 chrF++, với hai seed dương.
Chưa diễn giải những số này thành cải thiện hình thái trước character/influence audit.

## 4. Historical results và cơ chế: điều gì được hỗ trợ?

| Quan sát | Có thể nói | Chưa thể nói |
| --- | --- | --- |
| mT5 Full−Baseline +1.8074 BLEU, Holm p=0.0036 ở B | Có gain trên các checkpoints lịch sử được đánh giá lại | Gain đã lặp qua seed/protocol matched; hữu ích trên mọi backbone |
| mBART routing on−off +3.9287 BLEU, Holm p=0.0036 ở B | Checkpoint Full đã sử dụng routing trong computation | CLRR hơn một LSR-only được train riêng 3.93 BLEU |
| ByT5 Full−Baseline −0.1696 BLEU, không significant | Có point estimate giảm ở backbone này | Phương pháp luôn có lợi hoặc ByT5 chắc chắn bị hại |
| Full weighted LSR/CE global gradient median khoảng 0.43–0.54% đầu và 0.15% cuối training C | Lực gradient tương đối của auxiliary objective giảm theo các mốc đo | Tăng lambda sẽ tự cải thiện dịch; gradient nhỏ đồng nghĩa không tác động |
| Pooled effective ranks còn cao; biến đổi khác nhau theo backbone | Chưa thấy global collapse trong các phép đo này | CLRR đã chứng minh chống collapse hoặc bảo toàn phụ tố |

CLRR thêm zero parameters nhưng vẫn full fine-tuning. Không gọi đó là giảm
optimizer memory hay tốc độ tốt hơn PEFT khi chưa benchmark cùng điều kiện.
Khác LR giữa các phương pháp không tự là bất công; khác tuning budgets/protocol
khiến các điểm LoRA lịch sử chưa đủ làm ranking tổng quát.

## 5. Rủi ro đo đạc cần giải quyết trước tối ưu

Historical mBART ở B có Full−Baseline chrF++ +5.0277, nhưng exploratory
leave-one-out bỏ row170 làm gap còn khoảng +0.3844. Character-only chrF trên
toàn test có gap +0.6907. Đây là sensitivity của metric trên Mandarin ít khoảng trắng;
không phải lý do xóa câu hoặc thay primary metric sau khi thấy test.

**Cập nhật:** influence audit của cả 12 outputs C đã hoàn tất ở mục9; phân tích
trước đó được giữ như bối cảnh, không suy từ B sang C.
Ba nhóm regex mi/ma/pa là proxy có overlap, không là gold morphology.

Chưa có matched training control cùng routing nhưng không detach, nên chưa
phân biệt lợi ích của activation reuse với lợi ích của stop-gradient.
Tiny-model correctness chỉ chứng minh hành vi implementation.

## 6. Đánh giá bài hiện tại

Quick review trong chat: soundness 3/5, excitement 2.5/5, overall 2.5/5
(Borderline Findings), confidence 3/5. Đây là nhận định của cùng trợ lý đã tham gia
sửa bài, **không phải panel độc lập hoặc dự báo acceptance đã hiệu chuẩn**.
Thang tham khảo: [ARR review form](https://aclrollingreview.org/reviewform).

Các vấn đề còn lại: đóng góp riêng nhỏ/chưa chắc chắn; thiếu control detach;
influence audit C chưa xong; novelty cần đối chiếu trực tiếp với nghiên cứu xuyên tầng,
như [Multi-layer Representation Fusion](https://aclanthology.org/C18-1255/)
và [Dense Information Flow](https://aclanthology.org/N18-1117/).
Những công trình này là đầu mối literature cần đọc sâu; không kết luận CLRR trùng chúng.
Một language pair không tự làm bài không hợp lệ; scope và sức mạnh claim phải phù hợp.

**Quyết định:** GO cho chẩn đoán rẻ; chưa đủ bằng chứng để nộp ACL main với kỳ vọng
một đóng góp phương pháp mạnh. Chưa có căn cứ phải bỏ CLRR hoặc làm lại toàn bộ phương pháp.

## 7. Bước tiếp theo đã chuẩn bị: character và influence audit C

### Đầu vào đã kiểm tra

- Đủ 12 `runs/mbart-amis-<method>-seed<seed>/test_predictions.csv` trong suite C.
- Mỗi file 575 rows; SHA-256 khớp `seed_scores.csv`; source đúng thứ tự và
  saved targets khớp raw targets sau strip ngoài chuỗi.
- Receipt: [input_manifest.json](../../../outputs_rebuttal/amis_confirmation_20261003/metric_influence_preparation/input_manifest.json).
- CPU trong conda `clrr`; không cần checkpoint, Colab hoặc training.

### Phép đo cần thực hiện

1. Tính character-only chrF cho cả 12 outputs; đối chiếu full-test chrF++ với số đã ghi.
2. Với từng seed, đo Full−Baseline, Full−LSR, CLRR−Baseline và Full−CLRR.
3. Leave-one-row-out cho chrF và chrF++: mỗi row bỏ đúng một lần; báo toàn bộ
   khoảng biến thiên, row ảnh hưởng nhất, độ thay đổi gap và số lần đổi dấu.
4. Kiểm tra shortcut sufficient statistics bằng direct corpus scoring ở những
   row cực trị; giữ mọi câu trong kết quả chính và không tính p-value mới từ leave-one-out.

Script lịch sử `scripts/revalidation/measure_prediction_influence.py` đang trỏ
nhóm A và output cũ. **Cần bổ sung chế độ riêng cho C trước khi chạy**, kiểm tra
hash/reference alignment và ghi outputs vào thư mục C riêng; không chạy nguyên
script cũ rồi gọi là audit suite mới.

### Đầu ra và quyết định sau khi đo

- Artifact mới dự kiến: `outputs_rebuttal/amis_confirmation_20261003/metric_influence/`.
- Ghi số vào EXPERIMENTS_UPDATED, diễn giải vào file insight này. Paper chưa đổi
  cho đến khi kết quả được đối chiếu; kết quả cũ và raw artifacts vẫn giữ nguyên.
- Nếu BLEU/char-only cùng ủng hộ gain và influence không tập trung mạnh:
  có cơ sở chuyển sang chẩn đoán/validation tuning theo Gate1–2.
- Nếu gain chrF++ dựa vào rất ít câu hoặc metrics bất đồng: audit validation
  và lỗi dịch trước khi đổi trọng số. Không loại câu để làm gain đẹp hơn.
- Báo từng kết quả có lợi/bất lợi lên chat trước khi chuyển sang thí nghiệm khác.

**Trạng thái khi lập kế hoạch:** đã chuẩn bị đầu vào; chưa chạy audit mới, sweep trọng số,
control không detach hay Turkish. Trạng thái thực thi mới xem mục9.

## 8. Kế hoạch chạy tiếp sau review

[CLRR_GO_NOGO_PLAN.md](CLRR_GO_NOGO_PLAN.md) đã được cập nhật thành lộ trình cụ thể:
audit không train → tuning Full/LSR cùng số trials → validation qua seed →
CLRR-only/no-detach controls → đánh giá Amis → mở rộng độc lập hoặc nâng cấp theo nguyên nhân.
Trần Amis20lượt mới nếu reuse hợp lệ, nâng cấp có điều kiện tối đa10lượt;
Turkish và NLLB mỗi bên pilot4, chỉ mở thành12 nếu qua gate tương ứng.
Không có kết quả mới trong lần cập nhật này. Các threshold là tiêu chí đầu tư dự án,
không là chuẩn acceptance hoặc bảo đảm phương pháp sẽ thắng.

## 9. Audit C đã xong và vòng D đã bắt đầu

**Điểm dừng mới:** tác giả yêu cầu xong Full α0.20/λ0.10/seed42 thì tạm ngưng.
Hướng dẫn và trạng thái cuối tại [CONFIGURATION_SEARCH_HANDOFF.md](../../../operations/CONFIGURATION_SEARCH_HANDOFF.md).
Trial α0.05/λ0.10/seed42 đã xong: validation chrF++14.4092, thấp hơn Full gốc14.7317.
Giảm routing chưa giúp ở trial này; chưa có kết luận từ cả screening hoặc test mới.
Validation chỉ có1word-bigram reference (row197); trial α0.05 match0, nên câu này không
tạo gain word-bigram cho trial đó. Bỏ row197 chỉ đổi chrF++ thành14.4440.
No-detach CPU gradient/checkpoint gate đã đạt; GPU/control training vẫn chưa chạy.

Trial α0.20/λ0.10/seed42 cũng đã xong20epochs, selected checkpoint720:
validation BLEU17.4586, chrF++14.5738, chrF16.7469. Tốt hơn α0.05 nhưng chưa vượt
α0.10 gốc (validation chrF++14.7317). Cả hai trial mới đều match0word-bigrams;
thứ hạng giữa hai trial giữ nguyên trên chrF và khi bỏ row197. Chưa có test mới.
**Insight tạm thời:** thay α theo hai hướng chưa giúp ở seed42; chưa đủ kết luận λ
hay contribution CLRR vì sáu screening trials và confirmation chưa chạy.
Đã tạm ngưng theo tác giả, backup cả hai checkpoints FiveC đã verify SHA/size,
metrics/predictions lưu local; chưa nâng cấp phương pháp.

Nguồn: `outputs_rebuttal/amis_confirmation_20261003/metric_influence/`.
Đủ12systems/575rows, chrF++ khớp số cũ <1e-8; shortcut leave-one-out được kiểm tra
với direct corpus scoring ở các cực trị, sai số <1e-10. Không sửa điểm cũ, không
xóa câu và không tạo kiểm định significance mới.

| So sánh character-only chrF | Seed42 | Seed43 | Seed44 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Full−Baseline | +0.2654 | +1.1001 | −0.3865 | +0.3263 |
| Full−LSR | +0.3046 | −0.0676 | +0.2655 | +0.1675 |

**Có lợi:** Full vẫn có mean character-only cao nhất và Full−LSR dương2/3seeds;
gap dương ở seed42/44 không đổi dấu khi bỏ bất kỳ một câu nào.
**Bất lợi:** không xác nhận lợi ích character-only ở3/3seeds. Seed43 Full−Baseline
chrF++ +5.3162 giảm còn +0.5803 khi bỏ row170; character-only gap +1.1001 còn
+0.8602. ChrF++ làm hiệu ứng seed43 trông lớn hơn mức character agreement.
Đây không là lỗi của primary metric hay bằng chứng phải loại câu; giữ full test.
Leave-one-out không chứng minh robustness khi bỏ nhiều câu hoặc qua training seeds.

**Hành động:** tiếp tục validation tuning có controls; chưa tăng claim trong paper.
Không dùng mean chrF++ +1.7316 làm bằng chứng gain rộng/ổn định hoặc morphology.

---

### Tổng kết Thực nghiệm Phase C2 (Attached Control) & Final Test Suite (04/10/2026)

1. **Hiệu ứng của Cơ chế Detach (Attached vs Detached):**
   - Ba seeds Full Source-Attached ($\alpha=0.10, \lambda=0.10$, `detach=False`) cho kết quả không đồng đều: seed42 giảm, seeds43/44 có cải thiện trên validation. Attached validation mean BLEU cao hơn detached khoảng0.1236, chrF++ thấp hơn0.0471.
   - Giữ detached theo validation chrF++ đã khóa. Quality và sample std chưa chứng minh `.detach()` tối ưu, bảo đảm ổn định biểu diễn hoặc tránh gradient interference; cần probes trực tiếp như bàn giao mục14.

2. **Đánh giá Test Checkpoints Cuối cùng (18 Models, 6 Nhánh, 3 Seeds):**
   - Test575rows, FP32, beam4, tokenizer `zh`, SacreBLEU2.6.0. Theo bàn giao, architecture khóa trước generation mới; testC đã được xem, không gọi là test chưa từng được biết.
   - **Độ phân tán quan sát:** Full có stdBLEU0.1743/chrF++0.2107 trên3seeds này. ChrF++ của baseline/tunedLSR nhạy wordbigram hiếm; không gọi giảmscore seed43 là collapse. TunedLSR seed43 vẫn hơn Full ở BLEU và chrF character-only.
   - **Tín hiệu gia tăng có tính lặp lại (Incremental Gain):** Full CLRR Detached vượt Matched LSR $\lambda=0.10$ ở **100% (3/3) các seeds** (+0.1701 ở seed 42, +0.1243 ở seed 43, +0.4087 ở seed 44; mean delta **+0.2344 $\pm$ 0.1527 BLEU**).
   - **Kiểm định và giới hạn:** không p-value primaryHolm18 nào đạt0.05. Holm điều chỉnh p-values, không mở rộng CI đang lưu; có2CI chưa điều chỉnh không chứa0. SecondaryHolm30 hiện khác pairs đã khóa và cần hoàn tất theo bàn giao mục14. Full thua tunedLSR mean0.1267BLEU/0.1635chrF character-only; chưa đủ claim superiority.
