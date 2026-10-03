# CLRR — insight từ thực nghiệm đã xác minh

Ngày lập: 03/10/2026. Đi cùng [EXPERIMENTS_UPDATED.md](EXPERIMENTS_UPDATED.md)
và [paper hiện hành](clrr_main.tex). File này diễn giải bằng chứng; sổ kết quả và
artifact gốc vẫn là nguồn số. Không có phép đo hoặc lượt training mới trong lần lập file này.

## 1. Bản cũ và phạm vi hiện hành

- [Insight cũ đã archive](archive/experiment_reports_20261003/RESEARCH_INSIGHTS.md)
  được giữ nguyên để truy vết. Những nhận định lịch sử về bảo toàn morphology,
  chống collapse, gradient explosion hoặc PEFT overfitting không tự trở thành
  kết luận của bản mới.
- [Phiên bản insight sớm hơn](archive/RESEARCH_INSIGHTS_v1_with_layerskip_byt5.md)
  và [phân tích paper cũ](archive/PAPER_EXPERIMENT_INSIGHTS.md) chỉ là lịch sử.
- Phạm vi hiện hành: Amis→Mandarin. Không đưa Asháninka hoặc LayerSkip trở lại.
  ByT5 vẫn thuộc các đối sánh được giữ trong paper hiện hành.
- [METRICS.md](METRICS.md) quy định cách đo;
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
- Receipt: [input_manifest.json](../outputs_rebuttal/amis_confirmation_20261003/metric_influence_preparation/input_manifest.json).
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

Vòng D đã được tác giả cho tự thực hiện; không yêu cầu xác nhận lại sau mỗi phép đo
trong phạm vi đã lập. Colab A10040GB Standard, conda clrr, preflight GPU đạt;
trial đầu Full α0.05/λ0.10/seed42 đã bắt đầu. Screening tối đa8lượt mới theo
kế hoạch; chỉ dùng validation. Checkpoint lưu dưới prefix FiveC riêng
`amis_configuration_20261003_v1`, xác minh remote trước dọn weights của trial đã xong.
Chưa có score mới từ training D tại thời điểm cập nhật này. Turkish vẫn dừng.
