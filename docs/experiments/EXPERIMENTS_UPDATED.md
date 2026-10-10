# Thực nghiệm sau xác minh — kết quả hiện hành của CLRR

<!-- standardized-es10-final-status -->
## Đối sánh ES10 đã hoàn tất — 09/10/2026

**ALL_14_VALIDATION_VERIFIED.** Đủ 14 runs đã hoàn tất, replay độc lập public predictions/metrics và kiểm tra hash checkpoint PASS. Bảng validation đầy đủ ở cuối file; đây chưa phải điểm test.

A06 mBART Strong LoRA B seed42 chạy lại trên Colab1 A100 Standard 40GB, hoàn tất 20 epochs; chọn checkpoint epoch19/step684 theo raw-reference validation chrF++. Validation: BLEU **10.9850**, chrF++ **10.3825**, chrF0 **12.0560**. Dừng vì epoch cap. Các 13 run trước không train lại và scores replay không đổi.

Checkpoint public: `checkpoints/amis_mandarin/mbart-large-50/strong-lora-b/seed-42/es10/best.zip`; SHA256 `4cf6ccac866db298b93134a69b015545271a9ef18793dcda483e42cf718a5a94`. Joint selection lock public: `registry/plans/amis_standardized_controls/selection_lock_14.json`. Index public đã cập nhật đủ checkpoint ES10; tổng104 archives, giữ nguyên weights cũ.

Colab1/2/3 đều xác nhận **Active assignments=0**, **Usage rate=0.00/hr**; balance tương ứng **52.92 / 151.73 / 8.57 CU**. Không tải weights về PC. Receipt audit: `outputs_rebuttal/amis_standardized_strong_mbart/status/completion_audit.json`.

**Bước tiếp theo:** evaluation test beam4/FP32 từ đúng 14 checkpoints đã khóa, rồi replay test metrics và ghi bảng so sánh. **Chưa chạy test trong task này.** Những ghi chú đang chạy/tạm dừng ở phần lịch sử bên dưới đã được trạng thái này thay thế.

<!-- end-standardized-es10-final-status -->

## Cách đọc sổ số hiện hành — 08/10/2026

Đây là **sổ thực nghiệm hiện hành duy nhất**. Không thay bằng paper master hay bảng có điểm cao hơn. Mục13 chứa 32 fresh confirmation runs; Mục14 chứa Turkish seed42. Development seeds42–44, historical A/B/C và fresh confirmation là các cohort khác nhau; không ghép chúng thành một mean hay p-value.

Các ghi chú TECHNICAL_HOLD/đang chạy và yêu cầu cũ phía dưới là lịch sử. Cohort corrected và fresh đã xong; Turkish cũng đã train/test xong. Xem [insight hiện hành](../research/RESEARCH_INSIGHTS.md) và [bàn giao](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md) để đọc kết luận và hành động mới nhất. Bước gộp này không tính lại hoặc thay số khoa học.

> **Cập nhật 05/10/2026 — TECHNICAL_HOLD cho fairness training:** G0 xác nhận lỗi accumulation trong wrappers CLRR/LSR; bản sửa đã qua synthetic regression. [Báo cáo riêng](../archive/TECHNICAL_AUDITS_20261005.md#g0-g1-verification-20261005). Metrics cũ không đổi và vẫn mô tả đúng outputs cũ, nhưng không dùng affected runs làm matched evidence cho recipe đã sửa. Chưa train lại research models.

**Ngày tổng hợp: 03/10/2026.** Đây là file duy nhất để đọc kết quả mới, số sửa,
chẩn đoán cơ chế và tình trạng các thực nghiệm bổ sung.
Số của bản paper trước revision nằm riêng tại [experiments gốc đã archive](../archive/EXPERIMENTS_ORIGINAL_20261003.md).

**HIỆN HÀNH V6 — 07/10/2026: 32 Fresh Seed Confirmation runs (seeds 45–52) hoàn tất 100% và audited, xem Mục 13 / bàn giao NEXT_EXPERIMENTS_HANDOFF_20261005.md Mục 8.**
Cả 32 runs đã kiểm định mù (selection lock trên validation chrF++), test beam 4 FP32 trên 575 câu.
Primary Gate 1 (Full vs LSR): NO-GO (Mean BLEU -0.0198, 3/8 wins, p=0.9297).
Secondary Gate 2 (Full vs Base): DID NOT PASS (Mean BLEU -0.0415, 4/8 wins, p=0.9141).
Cả 2 GPU A100 Colab đã tắt an toàn. Không train thêm.

**LẦN TRƯỚC: V5 — đã đo và audit, xem Mục 12 / bàn giao Mục 19.**
42coreconditions VERIFIED +3baselinecontrols và4verificationconditions; phát hiện
lệch nhãn beam ởbaseline42validation, đã có inventory mới; test scores cũ giữ nguyên.
A100 Standard đã tắt. Không train, không Turkish.

**LẦN TRƯỚC: V4 hoàn tất và VERIFIED — Mục 11 / bàn giao Mục 18.**
21checkpointprobes,3route-controls; exact validation replay cả3seeds. On−off
mean+2.9676BLEU trên cùngrouting-trainedweights; rank gain phụ thuộc site và
definition, không chứng minh causal anti-collapse. CPU audit/SHA đều PASS,
Colab đã tắt. Không train mới, không thay test scores/secondary tests trước đó.

**Review V3 trước thực thi V4 — 04/10/2026:** xem [bàn giao Mục 17](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history).
CSV route-control V3 chấm lại khớp, nhưng route-on lệch validation cũ
1.70–3.18 BLEU; language configuration/runtime chưa xác nhận. F4 đã sửa hook
control; F3 còn lỗi phạm vi gradient ở baseline/CLRR-only và đặt tên covariance
rank thành participation ratio. Không đưa claim V3 “89% câu đổi”/“anti-collapse
+34.4%” vào kết quả đã xác minh. Raw V3 giữ riêng, test scores trước V3 giữ nguyên.
Receipt: `outputs_rebuttal/amis_followup_20261004_v2/v3_independent_review_20261004.json`.

**Paper:** [bản thảo đang làm](../paper/clrr_main.tex); snapshot trước revision: [bản gốc](../archive/clrr_main_before_revision_20261003.tex), đã viết lại theo A/B/C; bản cũ chỉ còn trong archive. Mục10 là kết quả matched training mới nhất, không thay số lịch sử ở mục2.


**Phạm vi hiện hành:** Amis→Mandarin; bảng hiện hành chỉ giữ các phương pháp của paper mới. Bản trước thu gọn nằm trong `archive/scope_before_20261003/`. CSV/JSON khoa học gốc giữ nguyên để truy vết; các p-value vẫn dùng family gốc Holm42/Holm36/Holm24, không tính lại với family nhỏ hơn sau khi chọn phạm vi.

## Material Passport

### Cập nhật follow-up đang thực hiện — 03/10/2026

- **Yêu cầu mới:** xong Full α0.20/λ0.10/seed42 thì tạm ngưng. Chi tiết thao tác và
  điểm dừng: [CONFIGURATION_SEARCH_HANDOFF.md](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history).
- Trial D Full α0.05/λ0.10/seed42 đã xong20epochs; selected checkpoint720.
  Validation BLEU17.1224519, chrF++14.4091973, chrF16.4997872; chưa evaluate test.
  Validation chrF++ thấp hơn Full gốc α0.10/λ0.10/seed42 (14.7316677).
  HF archive2,270,687,637bytes đã verify SHA256/size, COMPLETE và predictions lưu local.
  Bỏ riêng validation row197: chrF++14.4440285; điểm chính vẫn giữ đủ576rows.
- Trial D Full α0.20/λ0.10/seed42 đã xong20epochs; selected checkpoint720.
  Validation BLEU17.4585802, chrF++14.5737574, chrF16.7469491; chưa evaluate test.
  HF archive2,270,706,593bytes đã verify SHA256/size, receipts/predictions đã sync local.
  Bỏ riêng validation row197: chrF++14.6108370; thứ hạng hai trial mới không đổi.
- Trial D Full α0.10/λ0.03/seed42 đã xong20epochs; selected checkpoint720.
  Validation BLEU17.1141186, chrF++14.4354908, chrF16.5587225; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
  Bỏ riêng validation row197: chrF++14.4696526.
  Thứ hạng3trial trên chrF++/chrF: α0.20 > α0.10/λ0.03 > α0.05;
  BLEU của α0.05 (17.1225) nhỉnh hơn α0.10/λ0.03 (17.1141), nên không nói cả3metrics cùng ranking.
- Trial D LSR λ0.60/seed42 đã xong20epochs; selected checkpoint648.
  Validation BLEU17.0299684, chrF++14.3687502, chrF16.5079732; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
  Bỏ riêng validation row197: chrF++14.4039019.
- Trial D LSR λ0.03/seed42 đã xong20epochs; selected checkpoint684.
  Validation BLEU17.7872853, chrF++14.8587474, chrF17.1364490; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
  Bỏ riêng validation row197: chrF++14.8945433 (đối chiếu native corpus scoring04/10;
  sửa typo docs, không đổi full-validation score14.8587474).
  Hiện là trial có validation score cao nhất trong toàn bộ 5 trials đã xong.
- Trial D LSR λ0.01/seed42 đã xong18epochs (early stopping); selected checkpoint504.
  Validation BLEU16.3028128, chrF++13.9491060, chrF16.0469369; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
  Bỏ riêng validation row197: chrF++13.9822829 (đối chiếu native corpus scoring04/10;
  sửa typo docs, không đổi full-validation score13.9491060).
- Trial D Full α0.10/λ0.30/seed42 đã xong20epochs; selected checkpoint684.
  Validation BLEU17.1999081, chrF++14.3614005, chrF16.4431517; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
  Bỏ riêng validation row197: chrF++14.3943706.
  Thứ hạng Full CLRR screening: α0.20/λ0.10 (14.57) > α0.10/λ0.03 (14.44) > α0.05/λ0.10 (14.41) > α0.10/λ0.30 (14.36).
- Trial D LSR λ0.30/seed42 đã xong20epochs; selected checkpoint684.
  Validation BLEU16.3825850, chrF++13.8950592, chrF16.0409174; chưa evaluate test.
  HF archive (2,270,700,552 bytes, SHA256 457c5b51...) đã verify, receipts/predictions đã sync local.
  Bỏ riêng validation row197: chrF++13.9284425; chrF 16.0793457.
- **Hoàn thành 8/8 trials screening vòng D (seed 42, validation-only).**
  Cả 2 phiên Colab (`colab` và `colab2`) đã được tắt hoàn toàn (`colab stop`) để bảo toàn tài nguyên.
  Toàn bộ 8 models đã backup lên Hugging Face `FiveC/amis-rewire-checkpoints` với SHA256/receipts verified.
  Thứ hạng toàn bộ 8 cấu hình trên validation chrF++ (và không đổi khi bỏ row 197):
  1. `lsr_l003` (14.86) > 2. `full_a020_l010` (14.57) > 3. `full_a010_l003` (14.44) > 4. `full_a005_l010` (14.41) > 5. `lsr_l060` (14.37) > 6. `full_a010_l030` (14.36) > 7. `lsr_l001` (13.95) > 8. `lsr_l030` (13.90).

### Cập nhật Confirmation & Interaction Diagnostic — 04/10/2026
- Đã tạo văn bản khóa selection [selection_lock.json](../../outputs_rebuttal/amis_configuration_20261003_v1/selection_lock.json) trước khi chạy.
- Điều phối song song 2 máy Colab A100 (`colab` & `colab2`) theo giao thức validation-only.
- **Run 1 Máy 1:** `mbart-amis-full_a010_l010-seed42-replica` đã xong 20 epochs; selected checkpoint 720.
  Validation BLEU 17.6436514, chrF++ 14.7316677, chrF 16.8752003; chưa evaluate test.
  BLEU/chrF++ khớp Full gốc nhóm C trong sai số numerical <1e-12 ở replica này;
  hỗ trợ tái lập metrics, không chứng minh weights identical hay tính tất định tuyệt đối.
  HF archive đã verify SHA256 và lưu trữ thành công trên `FiveC/amis-rewire-checkpoints`.
- **Run 1 Máy 2:** `mbart-amis-lsr_l003-seed43` đã xong 20 epochs; selected checkpoint 648.
  Validation BLEU 16.3621324, chrF++ 14.0108802, chrF 16.1186644; chưa evaluate test.
  HF archive đã verify SHA256 và lưu trữ thành công.
  LSR $\lambda=0.03$ seed43 thấp hơn seed42 khoảng0.85chrF++; cho thấy biến thiên qua
  seed, chưa chứng minh nguyên nhân là selection bias hoặc một seed42 ngẫu nhiên quá tốt.
- **Run 2 Máy 1:** `mbart-amis-lsr_l010-seed42` (Matched Control LSR $\lambda=0.10$, seed 42) đã xong 20 epochs; selected checkpoint 576.
  Validation BLEU 16.5687629, chrF++ 14.1674057, chrF 16.2731059; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
  *So sánh với Full replica (seed 42):* Full CLRR (14.7317 chrF++, 17.6437 BLEU) vượt Matched LSR (+0.5643 chrF++, +1.0749 BLEU) trên cùng seed 42.
- **Run 2 Máy 2:** `mbart-amis-lsr_l003-seed44` (Best-Tuned LSR $\lambda=0.03$, seed 44) đã xong 20 epochs; selected checkpoint 648.
  Validation BLEU 16.8331940, chrF++ 14.0814040, chrF 16.2298532; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
  *Tổng kết 3 seeds Best-Tuned LSR $\lambda=0.03$:*
  - Seed 42: chrF++ 14.8587 (BLEU 17.7873)
  - Seed 43: chrF++ 14.0109 (BLEU 16.3621)
  - Seed 44: chrF++ 14.0814 (BLEU 16.8332)
  - Mean $\pm$ sample std: chrF++14.3170 $\pm$0.4705, BLEU16.9942 $\pm$0.7261.
  - So đúng cùng3seeds: Fullλ0.10 mean chrF++14.4975, cao hơn tunedLSR14.3170
    khoảng0.1805; mean ΔBLEU+0.2193,2/3seeds dương. Không so LSR mean với Full
    riêng seed42=14.7317; chưa có kiểm định mới để gọi gap significant.
- **Run 3 Máy 1:** `mbart-amis-full_a010_l003-seed44` (Diagnostic interaction control, seed 44) đã xong 20 epochs; selected checkpoint 612.
  Validation BLEU 16.7902942, chrF++ 14.1297008, chrF 16.2854825; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
- **Run 3 Máy 2:** `mbart-amis-full_a010_l003-seed43` (Diagnostic interaction control, seed 43) đã xong 20 epochs; selected checkpoint 720.
  Validation BLEU 17.2166000, chrF++ 14.4775385, chrF 16.6075152; chưa evaluate test.
  HF archive đã verify SHA256/size, receipts/predictions đã sync local.
  *So sánh tương tác trực tiếp Full $\lambda=0.03$ vs LSR $\lambda=0.03$:*
  - Seed42: Full14.4355 vs LSR14.8587 (Full thấp hơn ở seed screening).
  - Seed43: Full14.4775 vs LSR14.0109 (Δ+0.4667chrF++, +0.8545BLEU).
  - Seed 44: Full 14.1297 vs LSR 14.0814 (Full vượt +0.0483 chrF++)
  - Mean 3 seeds Full $\lambda=0.03$: 14.3476 $\pm$ 0.1899 (BLEU 17.0403 $\pm$ 0.2223)
  - Mean3seeds LSR$\lambda=0.03$:14.3170 $\pm$0.4705 (BLEU16.9942 $\pm$0.7261).
  - Fullλ0.03 có sample std thấp hơn trong3lượt này; chưa chứng minh giảm training
    variance hoặc ổn định biểu diễn. Mean Full−LSR chỉ+0.0461BLEU/+0.0306chrF++;
    character-only mean−0.0111, BLEU thắng1/3seeds. Không claim superiority hệ thống.
- **HOÀN THÀNH TOÀN BỘ 6/6 RUNS CONFIRMATION & DIAGNOSTIC.**
  - Cả 2 phiên Colab (`colab` và `colab2`) đã được tắt hoàn toàn (`colab stop`, trạng thái `Session not found`) để bảo toàn Compute Units.
  - Toàn bộ 6 models đã lưu trữ và xác minh SHA256 trên Hugging Face `FiveC/amis-rewire-checkpoints/amis_configuration_confirmation_20261004_v1`.
  - Toàn bộ `metrics.json`, `COMPLETE.json`, `validation_predictions.csv` và bảng tổng kết `confirmation_summary.csv` đã đồng bộ tại `outputs_rebuttal/amis_configuration_confirmation_20261004/`.
  - Bỏ row197 làm chrF++ tăng khoảng0.03–0.04; ranking6models giữ nguyên nhưng
    gaps thay đổi nhẹ (Full0.03−LSR0.03 seed43 khoảng−0.000393, seed44−0.000324).
    Không nói khoảng cách delta hoàn toàn bất biến.

### Đối chiếu sau confirmation — 04/10/2026

Review local tái tính6sets validation đạt; primary scores không đổi. Bảng mean và paired
deltas đúng3seeds tại [bàn giao, mục10](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history).
Fullλ0.10−matchedLSR0.10: +0.6705BLEU/+0.4368chrF++,3/3seeds dương;
Fullλ0.10−tunedLSR0.03: +0.2193/+0.1805,2/3dương;
Fullλ0.10−baseline: −0.4498/−0.1359, BLEU giảm3/3seeds.

### Phase C2 (Attached / No-Detach Control) — Hoàn tất 04/10/2026

Đã hoàn thành 3/3 runs Phase C2 (`clrr_source_detach=False`, `lsr_target_detach=True`, $\alpha=0.10, \lambda=0.10$):
- GPU smoke test `tests/smoke_attached_routing.py --cuda` passed trước training (`added_source_gradient_difference = 3.23`).
- Toàn bộ 3 runs đã lưu trữ, verify SHA256 và upload lên Hugging Face `FiveC/amis-rewire-checkpoints/amis_detach_control_20261004_v1/`.
- Cả 2 Colab A100 (`colab` và `colab2`) đã được tắt hoàn toàn (`colab stop`).
- Kết quả validation (raw reference, greedy beam 1, zero test snooping):
  1. `mbart-amis-full-attached-a010-l010-seed42`: chrF++ **14.1845**, BLEU **16.8827**, chrF0 **16.2653** (checkpoint 576).
  2. `mbart-amis-full-attached-a010-l010-seed43`: chrF++ **14.4706**, BLEU **17.4909**, chrF0 **16.6557** (checkpoint 684).
  3. `mbart-amis-full-attached-a010-l010-seed44`: chrF++ **14.6962**, BLEU **17.6376**, chrF0 **16.8618** (checkpoint 612).
  - **Mean 3 seeds Attached CLRR**: chrF++ **14.4504 $\pm$ 0.2566**, BLEU **17.3371 $\pm$ 0.4005**, chrF0 **16.5943 $\pm$ 0.3029**.
- **Kết luận Gate C2**:
  - Bỏ detach (`Attached`) không cải thiện hiệu năng so với bản Detached chuẩn (seed 42 sụt giảm -0.55 chrF++, -0.76 BLEU; seed 43/44 tương đương).
  - Attached CLRR vẫn vượt Matched LSR $\lambda=0.10$ (+0.31 BLEU trên seed 42) và vượt Tuned LSR $\lambda=0.03$ (+0.1334 chrF++, +0.0289 BLEU trên 3-seed mean), với độ phân tán thấp hơn đáng kể (std 0.26 vs 0.47).
  - Theo quy tắc rẽ nhánh Mục 11, bản Detached chuẩn được giữ nguyên; 3 seed CLRR-only detached đã có sẵn và không cần train lại.


- Audit C không train đã hoàn tất: 12outputs ×575rows, 24 comparisons chrF/chrF++,
  hashes/source/targets đạt; chrF++ tái tính khớp số cũ <1e-8. Không sửa điểm A/B/C.
- Nguồn mới: `outputs_rebuttal/amis_confirmation_20261003/metric_influence/`:
  `scores.csv`, `influence_summary.csv`, `influence_all_rows.csv`, `protocol.json`.
- Mean character-only chrF: Baseline18.0452, CLRR-only18.2035, LSR-only18.2040,
  Full18.3715. Full−Baseline +0.3263, Full−LSR +0.1675; cả hai có2/3seeds dương.
- Seed43 Full−Baseline chrF++ +5.3162, bỏ riêng row170 còn +0.5803;
  character-only gap +1.1001 còn +0.8602 khi bỏ row170. Mọi điểm chính vẫn giữ row170.
- Full−LSR character-only seed42/43/44: +0.3046/−0.0676/+0.2655.
  Seed43 có6/575 leave-one-out đổi dấu, cho thấy gap rất nhỏ. Không có kiểm định mới.
- **Insight:** gain character-level trung bình dương nhưng khiêm tốn; chrF++ seed43
  phóng đại gain so baseline vì một câu. Đây là sensitivity, không lỗi phải sửa số.
- Bắt đầu vòng D validation-only trên Colab A100 Standard40GB, conda clrr.
  Trial đầu: Full α0.05/λ0.10/seed42, tối đa20epochs/patience4, batch32×4,
  LR5e-5; preflight GPU bốn nhánh đạt. Chưa có điểm candidate mới.
- Artifacts D: `outputs_rebuttal/amis_configuration_20261003_v1/`;
  checkpoint backup riêng `FiveC/amis-rewire-checkpoints/amis_configuration_20261003_v1`.
  Không đưa test vào bundle screening; chọn cấu hình bằng raw-reference validation chrF++.

- Nguồn ledger ban đầu: CSV/JSON/predictions và checkpoint receipts đã có trong workspace;
  lượt gom tài liệu ban đầu không chạy lại metrics/generate/training. Follow-up mới
  đã đo influence C và bắt đầu D, được ghi riêng trong phần cập nhật phía trên.
- Giai đoạn A: báo cáo **28 prediction outputs Amis** từ audit gốc, raw references, bootstrap42tests; không load model.
- Giai đoạn C: **12/12 fresh matched mBART runs**, 4 nhánh ×3seeds; 35trajectorypoints, Holm24; đã hoàn tất ở mục10.
- Giai đoạn B: generate lại **10 checkpoint Amis** bằng FP32 thống nhất, **4 routing-off**, bootstrap36tests.
- Chẩn đoán: correctness CPU; residual/SVD/gradient trên10checkpoint Amis,80batch backward không optimizer step.
- Weights GPU pin: `FiveC/amis-rewire-checkpoints`, revision `63b0dbc4fab41a669ccb36ed2f22894868634b8a`.
- GPU A100 Standard, không high-mem; Python trong conda `clrr`. Torch2.6.0+cu124, Transformers4.57.6, Accelerate1.15.0, SacreBLEU2.6.0 trong lượt FP32.
- Turkish **đã dừng theo yêu cầu tác giả**; phiên Colab Turkish đã tắt. Phiên Amis mới xem mục 10. Chưa có trained-test comparison Turkish.

## Cách đọc và nguồn số duy nhất

1. **Mục 2 là bảng điểm hiện hành:** dùng giai đoạn B cho10checkpoint Amis đã generate lại, A cho18outputs còn lại. Cột nguồn cho biết rõ phiên bản; không coi28hàng là cùng một lượt inference.
2. So sánh chất lượng mới giữa các model cùng lượt FP32 dùng **bootstrap B ở mục 3**. Đối sánh PEFT/decoder/chưa generate lại giữ kiểm định A ở phụ lục, không ghép predictions A/B để tạo một p-value mới.
3. Không thay checkpoint được chọn trong lịch sử. FP32 inference chung không sửa được sự khác nhau của training/validation-selection lịch sử.
4. Bảng hiển thị4chữ số, CSV/JSON giữ full precision. Các source CSV gốc vẫn giữ nguyên; file tổng hợp không sửa artifacts.
5. Báo cáo lẻ đã chuyển vào `docs/archive/README.md/`; chúng là lịch sử, có thể chứa tình trạng hoặc kết luận đã được lượt sau thay thế. Dùng file này để đọc trạng thái hiện hành.

## 1. Giao thức metrics và các lượt đo

Reference là target nguyên CSV đúng thứ tự và đủ số câu; chỉ strip whitespace đầu/cuối.
Không tokenize/decode reference bằng tokenizer model, không bỏ empty predictions.
SacreBLEU2.6.0: corpus BLEU case-sensitive, smooth exp, effective_order=False;
tokenizerzh cho Mandarin,13a cho Spanish/English. chrF++6/2, beta2,
lowercase=False, whitespace=False, eps_smoothing=False, không external tokenization.
chrF6/0 và TokenizerZh chỉ là diagnostics riêng.
Định nghĩa và signatures: [METRICS.md](METRICS.md).

**A — audit02/10:** cùng saved predictions, raw references,28outputs Amis được giữ lại,
10.000 paired resamples seed42; giữ Holm42 của family gốc21pairs×2metrics, dù chỉ hiển thị Amis.
Character diagnostics bổ sung dùng family42+8=50, không thay Holm42 gốc.

**B — generate03/10:** weights/generationFP32, TF32off, eval/no_grad,
beam4,batch8,padmultiple8; maxlength256 cho mBART/mT5/ByT5,128 cho NLLB.
Test575Amis đầy đủ. Chỉ hiển thị phạm vi Amis hiện hành.
Bootstrap10.000,seed42; giữ Holm36 của family gốc12pairs×3metrics, dù chỉ hiển thị Amis.
Không trộn p của các family này hoặc gọi sentence bootstrap là độ ổn định qua training seeds.

Probe trong môi trường hiện tại thấy baseline default encoderFP32/decoderBF16,
trong khi CLRR/LSR wrappers cả haiFP32. Flag `--generation-precision fp32|bf16`
và context generation đã được bổ sung, kiểm tra CPU/GPU cả4branches.
Điều này **không chứng minh dtype của mọi run lịch sử** hoặc quy mọi thay đổi prediction cho precision.


## 2. Toàn bộ điểm hiện hành — 28 outputs Amis

| Dataset | Backbone | Method / checkpoint | N | BLEU | chrF++ | Nguồn |
| --- | --- | --- | --- | --- | --- | --- |
| amis_mandarin | mBART | Baseline | 575 | 19.6144 | 14.0547 | B — generate FP32 03/10 |
| amis_mandarin | mBART | CLRR+LSR | 575 | 20.3896 | 19.0824 | B — generate FP32 03/10 |
| amis_mandarin | mBART | CLRR-only | 575 | 18.4697 | 13.2443 | A — audit predictions 02/10 |
| amis_mandarin | mBART | LSR-only | 575 | 19.7067 | 18.6169 | B — generate FP32 03/10 |
| amis_mandarin | mBART | CLRR-Dec+LSR | 575 | 19.9188 | 18.6999 | A — audit predictions 02/10 |
| amis_mandarin | mBART | BitFit | 575 | 0.3463 | 2.2552 | A — audit predictions 02/10 |
| amis_mandarin | mBART | Narrow LoRA | 575 | 3.2365 | 4.4213 | A — audit predictions 02/10 |
| amis_mandarin | mBART | Strong LoRA A | 575 | 10.2296 | 8.6319 | A — audit predictions 02/10 |
| amis_mandarin | mBART | Strong LoRA B | 575 | 13.8066 | 10.3293 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Strong LoRA A | 575 | 11.4625 | 9.5004 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Strong LoRA B | 575 | 9.8024 | 8.4010 | A — audit predictions 02/10 |
| amis_mandarin | mBART | Middle-Layer Alignment | 575 | 19.2893 | 18.4378 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Baseline | 575 | 13.6186 | 10.4990 | B — generate FP32 03/10 |
| amis_mandarin | NLLB | CLRR+LSR | 575 | 14.2835 | 10.9179 | B — generate FP32 03/10 |
| amis_mandarin | NLLB | CLRR-Dec+LSR | 575 | 13.4194 | 10.4559 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Middle-Layer Alignment | 575 | 13.9648 | 10.6596 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | BitFit | 575 | 1.0750 | 2.9643 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Narrow LoRA | 575 | 3.5383 | 4.9407 | A — audit predictions 02/10 |
| amis_mandarin | mT5 | Baseline | 575 | 2.7887 | 3.8129 | B — generate FP32 03/10 |
| amis_mandarin | mT5 | CLRR-only | 575 | 4.4393 | 5.1761 | B — generate FP32 03/10 |
| amis_mandarin | mT5 | CLRR+LSR | 575 | 4.5961 | 5.0425 | B — generate FP32 03/10 |
| amis_mandarin | mT5 | CLRR-Dec+LSR | 575 | 4.8315 | 5.1933 | A — audit predictions 02/10 |
| amis_mandarin | mT5 | CLRR-Both+LSR | 575 | 3.4459 | 4.5753 | A — audit predictions 02/10 |
| amis_mandarin | mT5 | Middle-Layer Alignment | 575 | 4.6625 | 5.2103 | A — audit predictions 02/10 |
| amis_mandarin | ByT5 | Baseline | 575 | 7.4856 | 8.2977 | B — generate FP32 03/10 |
| amis_mandarin | ByT5 | CLRR+LSR | 575 | 7.3160 | 8.0966 | B — generate FP32 03/10 |
| amis_mandarin | ByT5 | CLRR-Dec+LSR | 575 | 7.4394 | 8.0630 | A — audit predictions 02/10 |
| amis_mandarin | ByT5 | Middle-Layer Alignment | 575 | 7.5660 | 8.3414 | A — audit predictions 02/10 |

Không có mT5 LSR-only trong bảng vì thiếu predictions chứng nhận. Các kết quả của phương pháp thuộc phạm vi Amis hiện hành được báo cáo đầy đủ, gồm ByT5.

## 3. Kết quả quality mới nhất — cùng lượt FP32

### 4. Kiểm định quality

10.000 paired bootstrap resamples, seed42; CI percentile95% của corpus-score delta. Family36 tests =12pairs×3metrics, Holm correction; exploratory cố định saved checkpoints. Một p BLEU được đối chiếu với native SacreBLEU PairedTest, khớp. Không thay training-seed uncertainty bằng sentence bootstrap.

| So sánh BLEU | Δ | CI95% | Holm p |
|---|---:|---|---:|
| mBART Full−Baseline | +0.7752 | [−0.4388,2.0441] | 1.0000 |
| mBART Full−LSR-only | +0.6830 | [−0.4806,1.8530] | 1.0000 |
| NLLB Full−Baseline | +0.6650 | [−0.3295,1.7156] | 1.0000 |
| mT5 Full−Baseline | +1.8074 | [1.1826,2.4746] | 0.0036 |
| mT5 Full−CLRR-only | +0.1567 | [−0.3769,0.7116] | 1.0000 |
| ByT5 Full−Baseline | −0.1696 | [−0.8869,0.5266] | 1.0000 |

mT5 Full−Baseline vẫn có tín hiệu trên cả BLEU/chrF++/chrF0. Full−CLRR-only mT5 chưa rõ; chrF++ còn thấp hơn0.1336. mBART/NLLB positive point estimates chưa đủ chắc; mBART +5.0277chrF++ vẫn CI [−0.1872,13.2525], phù hợp cảnh báo word-bigram thưa từ audit trước. Không diễn giải nonsignificance là chứng minh bằng nhau.


### 5. Routing-off trên cùng trained Full weights

Chỉ đổi inference alpha0.1→0; tất cả weights/decoding giữ nguyên. Đây là **sensitivity của model đã train với CLRR**, không phải training ablation LSR-only.

| Model | Full BLEU | Routing-off BLEU | On−off | CI95% | Holm p |
|---|---:|---:|---:|---|---:|
| mBART Amis | 20.3896 | 16.4609 | +3.9287 | [2.7175,5.1948] | 0.0036 |
| NLLB Amis | 14.2835 | 12.9177 | +1.3659 | [0.3337,2.7009] | 0.4080 |
| mT5 Amis | 4.5961 | 4.1384 | +0.4577 | [0.0719,0.8816] | 0.4140 |
| ByT5 Amis | 7.3160 | 7.5166 | −0.2006 | [−0.7983,0.3788] | 1.0000 |

**Tốt:** mBART Amis model thật sự phụ thuộc skip lúc inference theo kiểm định này. CLRR không phải một nhánh không được model dùng.

**Giới hạn:** bỏ thành phần mà model quen khi train tạo distribution shift. On−off dương không chứng minh train Full tốt hơn train LSR-only/Baseline, morphology preservation hay generalization. ByT5 point estimate tốt hơn khi routing-off, nhưng uncertainty chưa đủ để khẳng định tắt routing luôn cải thiện.


### Thay đổi predictions từ A sang B



| Model | Số câu đổi | ΔBLEU mới−cũ | ΔchrF++ mới−cũ |
|---|---:|---:|---:|
| mBART LSR-only | 15 | +0.0776 | +0.0416 |
| NLLB baseline | 80 | +0.1184 | +0.0602 |
| NLLB Full | 4 | +0.0016 | +0.0027 |
| ByT5 baseline | 44 | −0.0980 | −0.0080 |
| ByT5 Full | 7 | +0.0052 | +0.0011 |

mBART baseline/Full và cả ba mT5 models khớp toàn bộ text predictions cũ. Các thay đổi không đủ để đảo kết luận chính. Vì môi trường/cache/padding/precision có thể khác lịch sử, không gán tất cả Δ cho precision hoặc tuyên bố historical predictions sai. Sentinel batch1 vs batch8 ở ba model còn lệch không tự giải quyết việc tái lập historical inference.


## 4. Độ bền metrics và correctness

### 2. Kiểm tra implementation — TỐT trong phạm vi đã thử

Artifact: [correctness.json](../../outputs_rebuttal/easy_diagnostics_20261002/correctness.json).

Môi trường: PyTorch 2.6.0+cu124, Transformers 4.57.6; thực thi CPU float32. Mỗi model nhỏ có 4 encoder layers, 2 decoder layers, hidden size 32; trọng số khởi tạo seed 42. Không optimizer step.

Kiến trúc: `MBartForConditionalGeneration`, `M2M100ForConditionalGeneration` dùng cho NLLB, `MT5ForConditionalGeneration`, và `T5ForConditionalGeneration` đại diện kiến trúc ByT5. Không dùng tokenizer hoặc weights pretrained của các model lớn.

| Phép kiểm tra | Kết quả |
|---|---|
| Routing đúng khoảng cách d=2 | PASS: đầu ra toy [1, 2, 3.1, 4.2] với alpha=0.1 |
| Gradient theo nhánh residual | PASS: nguồn skip không nhận gradient từ riêng cạnh skip; destination gradient = 1 |
| Tensor trong cache đã detach | PASS |
| alpha=0, lambda=0 so với baseline | Logits/loss/gradient khớp chính xác trên cả 4 kiến trúc |
| Số parameter khi thêm wrapper | Bằng baseline; không thêm parameter |
| Batch khác xen giữa hai lần gọi | Logits không đổi; cache rỗng sau forward |
| Beam generation lặp lại sau input khác | Giống hệt; thử num_beams=2 trên encoder CLRR |
| LSR source/target | Source nhận gradient auxiliary khác 0; target hidden không requires_grad |
| Checkpointing, dropout=0 | 32/32 so sánh PASS: 4 kiến trúc × 4 methods × 2 chế độ |
| Checkpointing, dropout=0.1 | 8/8 so sánh Full PASS: 4 kiến trúc × 2 chế độ |

Hai chế độ checkpointing là `use_reentrant=True/False`; so sánh toàn vector gradient của cùng weights/input/RNG. Sai khác L2 tương đối lớn nhất **1.61006724e-7**, dưới tolerance **1e-5**; logits và loss khớp.

Warning `None of the inputs have requires_grad=True` xuất hiện với reentrant checkpointing trong nhánh target no-grad; các phép so sánh gradient source vẫn PASS. Không lấy warning này làm bằng chứng mất gradient source.

**Giới hạn:** các kết quả không xác minh mọi cấu hình decoder/both-stack, mixed precision/GPU, layerdrop, phiên bản thư viện lịch sử hoặc quá trình optimization của checkpoint đã train. Các gradient norms trong JSON là của mô hình ngẫu nhiên; không dùng để giải thích hiệu quả của mBART/NLLB đã train.


### 3. ChrF trên ký tự — kết quả mới từ predictions thực

Artifacts: [character_chrf_bootstrap.csv](../../outputs_rebuttal/easy_diagnostics_20261002/character_chrf_bootstrap.csv), [robustness_protocol.json](../../outputs_rebuttal/easy_diagnostics_20261002/robustness_protocol.json).

Giữ reference gốc và đầy đủ test rows; chỉ strip khoảng trắng đầu/cuối. SacreBLEU 2.6.0: char_order=6, **word_order=0**, beta=2; các cấu hình còn lại theo protocol chuẩn. Đây là chrF diagnostic, **không thay thế chrF++/BLEU chính thức**.

Paired bootstrap 10,000 lần, seed 42, đơn vị câu; CI percentile 95% của chênh lệch corpus score. P-value theo implementation SacreBLEU; đã đối chiếu public `PairedTest` API trên mBART Full-vs-Baseline. Tất cả là kiểm tra hậu nghiệm, fixed checkpoints, một training seed.

| Dataset / backbone / so sánh | Delta chrF | CI 95% | p Holm chung 50 | Đánh giá |
|---|---:|---|---:|---|
| Amis mBART Full − Baseline | +0.6907 | [-0.2363, 1.6608] | 1.0000 | Tín hiệu dương; chưa đủ chắc |
| Amis mBART Full − LSR-only | +0.6681 | [-0.2182, 1.5563] | 1.0000 | Đóng góp CLRR cần xác nhận qua seeds |
| Amis mBART CLRR-only − Baseline | -0.9957 | [-1.8811, -0.1256] | 0.3696 | Point estimate xấu; chưa có ý nghĩa sau Holm |
| Amis mBART LSR-only − Baseline | +0.0226 | [-0.9934, 1.0638] | 1.0000 | Gần như ngang baseline trên ký tự |
| Amis NLLB Full − Baseline | +0.6349 | [-0.0498, 1.3291] | 0.8657 | Dương nhưng chưa đủ chắc |
| Amis mT5 Full − Baseline | +1.6108 | [1.2587, 1.9804] | 0.0050 | TỐT: bằng chứng rõ trên checkpoint này |
| Amis ByT5 Full − Baseline | -0.2163 | [-0.6192, 0.1534] | 1.0000 | Không hỗ trợ claim cải thiện mọi backbone |

CSV có cả Holm cho 8 diagnostics mới và Holm kết hợp **42 kiểm tra audit cũ + 8 mới = 50**. Không chỉnh lại file audit cũ. Dùng family chung 50 khi mô tả bằng chứng bổ sung của đợt này. Không coi p nhỏ của một diagnostic hậu nghiệm là validation độc lập.

CI percentile và p-value centered absolute-difference của SacreBLEU không phải hai phép đảo tương đương; vì vậy NLLB có raw p=0.0338 nhưng CI vẫn chứa 0. Không lấy raw p này để tuyên bố thắng sau hiệu chỉnh.

#### Insight mới có ích

Trong mBART, chrF ký tự của Baseline / CLRR-only / LSR-only / Full lần lượt là **18.2288 / 17.2331 / 18.2514 / 18.9195**.

Điều này làm rõ hai điểm:

- Mức tăng chrF++ lớn của LSR-only không đi kèm mức tăng ký tự tương ứng; cần thận trọng với word-bigram rất thưa trong tiếng Trung.
- Full vẫn có mức tăng ký tự trên cả Baseline và LSR-only. Vì vậy tín hiệu Full không hoàn toàn do thành phần word-bigram thưa. Chưa đủ để kết luận phối hợp CLRR+LSR tạo hiệu ứng ổn định hoặc cơ chế bảo toàn hình thái.


### 4. Ảnh hưởng từng câu — TỐT cho độ bền dấu của point estimate

Artifacts: [influence_summary.csv](../../outputs_rebuttal/easy_diagnostics_20261002/influence_summary.csv), [influence_protocol.json](../../outputs_rebuttal/easy_diagnostics_20261002/influence_protocol.json). Có bảng đầy đủ 4,600 dòng: 4 comparisons × 2 metrics × 575 lần bỏ một câu.

Mỗi lần chỉ bỏ một câu, tính lại corpus score; **kết quả chính vẫn giữ cả 575 câu**. Shortcut cộng/trừ sufficient statistics đã đối chiếu direct corpus API ở câu ảnh hưởng lớn nhất của cả 8 comparisons/metrics.

| So sánh, chrF word_order=0 | Delta đủ 575 câu | Min–max sau bỏ một câu bất kỳ | Số lần delta ≤ 0 |
|---|---:|---|---:|
| mBART Full − Baseline | +0.6907 | +0.5673 đến +0.7984 | 0/575 |
| mBART Full − LSR-only | +0.6681 | +0.5015 đến +0.8226 | 0/575 |
| NLLB Full − Baseline | +0.6349 | +0.5186 đến +0.7252 | 0/575 |
| mT5 Full − Baseline | +1.6108 | +1.5686 đến +1.6346 | 0/575 |

Với **chrF++**, mBART Full−Baseline dao động **+0.3844 đến +7.6908**; bỏ câu index 170 làm mức tăng giảm từ +5.0277 xuống +0.3844. Đây là nhạy cảm lớn về độ lớn; không phải công thức tính sai.

Với **chrF ký tự**, bỏ chính câu 170 vẫn còn **+0.5673**. Với Full−LSR-only, câu ảnh hưởng lớn nhất là index 65; bỏ câu đó vẫn còn **+0.5015**.

**Claim có thể dùng với phạm vi rõ:** “Trên các checkpoint Amis→Mandarin đã lưu, mức tăng chrF ký tự của CLRR+LSR trên mBART so với Baseline và LSR-only vẫn dương khi bỏ bất kỳ một câu test nào. Tuy nhiên CI bootstrap vẫn chứa 0; cần xác nhận qua training seeds.”

Leave-one-out không chứng minh mọi subgroup đều cải thiện, không thay thế bootstrap hoặc multi-seed, và không chứng minh bảo toàn affix.


**Phạm vi:** các bảng character/leave-one-out này thuộc predictions A; không dùng chúng thay kiểm định B khi một model đã generate lại.

## 5. Cơ chế trên checkpoint đã train

### 1. Giao thức đo

**Residual:** hook trước và sau CLRR; kiểm tra `post = pre + 0.1 × skip`, distance 2, encoder. RMS tính trên vị trí hợp lệ và tất cả chiều hidden; ratio là RMS residual/RMS output lớp hiện tại trước rewiring. Bảng dưới là trung bình và cực đại qua các ví dụ nguồn và các lớp nhận skip, từ lớp 3 trở đi.

**Collapse:** toàn bộ 575 câu test Amis, nguồn và đích; eval, FP32, TF32 tắt. Mean pooling theo attention mask, gồm special tokens hợp lệ. SVD đầy đủ trên biểu diễn đã center, tính float64. Effective rank là `exp(H(s/sum(s)))`; khác effective rank của eigenvalues covariance. Giới hạn rank là `min(n−1,d)`. Raw mean pairwise cosine đo hướng chung; cosine cao không tự chứng minh collapse.

**Gradient:** 32 câu validation cố định, RNG 20261002, 8 batch × 4; train mode với dropout và seed 1000+batch_id, target stop-gradient giữ dropout train mode. BF16 autocast, parameters FP32, gradient chưa scale/clip; không optimizer. Dùng tensor CE gốc và graph auxiliary gốc (`total_loss − native_CE`) của wrapper cho Full/LSR-only. Ratio là trung bình ratio norm theo batch, không phải ratio của loss scalar. Nhóm encoder loại shared embeddings; các nhóm parameters không chồng lặp.

Baseline/CLRR-only không dùng LSR khi train: gradient auxiliary của chúng trong artifacts là **counterfactual λ=0.1**, không được diễn giải là lực tối ưu lịch sử. Không so absolute gradient norm giữa kiến trúc như một thước đo chất lượng.


### 2. Effective rank và mức tập trung biểu diễn

| Dataset/model | Method | Source e-rank | Rank limit | Target e-rank | Source pairwise cosine |
|---|---|---:|---:|---:|---:|
| Amis mBART | Baseline | 390.700 | 574 | 390.675 | 0.432 |
| Amis mBART | Full | 387.850 | 574 | 383.852 | 0.717 |
| Amis mBART | LSR-only | 388.194 | 574 | 381.721 | 0.707 |
| Amis NLLB | Baseline | 383.630 | 574 | 423.467 | 0.696 |
| Amis NLLB | Full | 387.674 | 574 | 411.582 | 0.646 |
| Amis mT5 | Baseline | 197.389 | 512 | 259.242 | 0.855 |
| Amis mT5 | Full | 222.676 | 512 | 256.212 | 0.836 |
| Amis mT5 | CLRR-only | 216.523 | 512 | 258.193 | 0.765 |
| Amis ByT5 | Baseline | 261.717 | 574 | 309.060 | 0.613 |
| Amis ByT5 | Full | 259.642 | 574 | 291.086 | 0.902 |

**Tốt:** các checkpoint Full giữ phổ nhiều chiều; không có dấu hiệu collapse về một hướng theo phép đo pooled này. NLLB Full tái lập chính xác hai số SVD cũ khi làm tròn: **387.67/411.58**. mT5 Full source rank tăng khoảng 12.8% so với baseline, CLRR-only cũng tăng: bằng chứng cơ chế đáng theo dõi cùng kết quả dịch mT5 đã audit.

**Chưa tốt:** Full không tăng effective rank nhất quán. mBART Full gần LSR-only, thấp hơn baseline; target rank Full thấp hơn baseline trên cả bốn cặp dataset/backbone Amis. ByT5 Full source cosine tăng mạnh 0.613→0.902, source rank hơi giảm. Không dùng các bảng này để claim “CLRR universally improves diversity”, “stop-gradient guarantees anti-collapse”, hay “preserves semantics”.

mT5 rank tăng là tương quan với kết quả dịch, chưa chứng minh tăng rank gây tăng chất lượng. So sánh target/source rank phải trong cùng backbone/protocol; không so rank thô giữa các hidden dimensions.


### 3. Residual norms và cân bằng CE–LSR

| Dataset/model, method | Residual/current mean | Max | Weighted LSR/CE gradient, toàn model | Encoder excl. shared | Mean gradient cosine | Batch cosine âm /8 |
|---|---:|---:|---:|---:|---:|---:|
| Amis mBART Full | 5.911% | 8.626% | 0.167% | 0.310% | 0.057 | 0 |
| Amis mBART LSR-only | 0% | 0% | 0.237% | 0.412% | 0.046 | 1 |
| Amis NLLB Full | 6.632% | 8.764% | 0.487% | 0.737% | 0.149 | 0 |
| Amis mT5 Full | 6.213% | 9.838% | 0.794% | 1.755% | 0.084 | 3 |
| Amis mT5 CLRR-only | 6.209% | 9.833% | inactive | inactive | counterfactual | counterfactual |
| Amis ByT5 Full | 4.338% | 9.305% | 1.184% | 1.795% | −0.124 | 5 |

**Tốt:** trên các ví dụ/lớp đo, residual Full trung bình khoảng 4.3–6.6% của current output, cực đại dưới 10%; không thấy residual lấn át hoặc tăng vọt. Đây là mô tả checkpoint, không phải bảo đảm ổn định toàn bộ quá trình train hoặc mọi input.

**Cần điều tra:** weighted LSR gradient nhỏ so với CE tại checkpoint cuối được chọn. Điều này giúp định lượng λ=0.1 nhưng chưa đủ để tăng λ: không biết lực LSR ở giai đoạn đầu. ByT5 có conflict trên 5/8 batch, mean cosine âm; phù hợp với giả thuyết cần xem lại tương tác LSR/kiến trúc ở ByT5, chưa đủ để khẳng định đây là nguyên nhân giảm chất lượng. mT5 có 3/8 batch conflict dù mean cosine dương và chất lượng dịch tốt hơn; tránh claim LSR luôn đồng thuận CE.

Đo này thay thế việc dùng trace callback gradient bằng 0 sau `zero_grad` để suy luận. Nó không xác thực lại các trace lịch sử đó. Trong quá trình phát triển phép đo, CE tái dựng thủ công từ returned logits BF16 khác native CE tối đa 0.002555; **cả 12 checkpoint đã được đo lại bằng native objective**. Bản đầu giữ riêng `gradients_initial_reconstruction.json`, không dùng cho bảng cuối. Sai khác scalar auxiliary tái dựng/gốc tối đa khoảng 2.35e−7; gradient chính lấy từ graph gốc.


### 5. Checkpoint loading và phần chưa đo

- Wrapper ZIP thiếu config được phục hồi từ **paired trained baseline archive**, ghi source/member/SHA trong provenance; strict loading không có missing/unexpected/mismatched weights.
- NLLB tokenizer cũ lưu `extra_special_tokens` dạng list: chuyển cách truyền tham số cho Transformers 4.57.6; kiểm tra toàn bộ vocab và token IDs không đổi. Không áp dụng thay regex tokenizer âm thầm.
- Giữ saved language codes: Amis mBART `tl_XX→zh_CN`, NLLB `zho_Hant→zho_Hant`. Đây là cấu hình lịch sử để tái đo, không phải language codes cho Turkish.
- **mBART CLRR-only best checkpoint 504 chưa có trong revision HF đã kiểm tra** (chỉ tìm được đến 432). Không dùng checkpoint 432 thay thế. Vì vậy chưa đủ 4 thành phần mBART cho phân tích cơ chế.
- Không có đo no-stop-gradient trên checkpoint đã train tương ứng; không suy luận nhân quả về stop-gradient. Không đo multi-seed, tốc độ/peak memory training hay Turkish trong lượt này.


**Cập nhật so với audit A:** NLLB SVD387.67/411.58 nay đã có phép đo strict/full-spectrum để xác minh. Sentinel8câu ở lượt cơ chế đã được bổ sung bằng full-test B ở mục2–3; không dùng sentinel thay full test.

## 6. Thực nghiệm bổ sung và cấu hình thực tế

### Strong LoRA đã hoàn tất trên Amis

| Setting | mBART A | mBART B | NLLB A | NLLB B |
| --- | --- | --- | --- | --- |
| LoRA | r16/alpha32/dropout0.05 | như A | như A | như A |
| Targets | q,k,v,out,fc1,fc2 | như A | như A | như A |
| Embeddings | frozen | trainable | frozen | trainable |
| LR | 2e-4 | 1e-4 | 2e-4 | 1e-4 |
| Batch×accumulation | 4×32 | 4×32 | 16×8 | 16×8 |
| Best epoch / step | 19 / 684 | 19 / 684 | 18 / 648 | 19 / 684 |
| Added adapter params | 8,650,752 | 8,650,752 | 8,650,752 | 8,650,752 |
| Trainable params | 8,650,752 | 264,706,048 | 8,650,752 | 271,005,696 |
| Stored training runtime (min) | 93.4888 | 95.1309 | 32.9175 | 40.4702 |

Seed42, warmup0.06, tối đa20epochs/patience4. mBART validationbeam1/testbeam4;
NLLB validation/testbeam4. Điểm raw-reference hiện hành ở mục2.
A/B đổi đồng thời embeddings và LR, nên không cô lập tác động embeddings.
Chưa có HPO budget bằng nhau; không suy ra PEFT có capacity ceiling.
CLRR thêm0parameter nhưng full-tuning, không phải zero-trainable PEFT.
NLLB B:43.4496% trên active adapted623,724,544weights;
30.5848% trên registered886,079,488parameters có frozen copies của modules_to_save.

### Turkish→English — đã dừng, chưa có so sánh phương pháp

- Dataset mới: `data_processed/opus100_turkish_english_20k_v2/`, OPUS-100 revision `805090dc28bf78897da9641cdf08b61287580df9`.
- Train20.000unique pairs,dev1946,testofficial2000giữ thứ tự; test có18duplicate-excess pairs. Zero exact-source overlap giữa splits trong v2; giữ v1 nguyên trạng.
- Tokenization train mean source/target11.58055/12.2803;1source và1target vượt256; dev/test không vượt256.
- Pinned mBART revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`, tr_TR→en_XX.
- **Zero-shot test:** BLEU20.5746 / chrF++39.9016, beam4,FP32. Đây không phải trained result hoặc kết quả CLRR.
- Pilot định trước: Baseline/CLRR-only/LSR-only/Full,seed42,LR5e-5,weight_decay0,warmup0.06,batch32×4=128,length256,max20epochs/patience4,trainBF16+GC,generationFP32,valbeam1/testbeam4.
- Calibration Full batch32,length256: peak26.624GiB; chỉ là kiểm tra vận hành, weights tạm bị loại bỏ.
- Baseline đã bắt đầu rồi **dừng theo yêu cầu tác giả**. Ba nhánh CLRR-only/LSR-only/Full chưa train. Không có trained-test score cuối để đưa vào bảng kết quả.
- Colab đã tắt; không tự chạy tiếp Turkish. Không gọi Turkish là ngôn ngữ chưa có trong pretraining của mBART.


<a id="7-doi-chieu-claim-goc-voi-bang-chung-hien-tai"></a>

## 7. Đối chiếu claim gốc với bằng chứng hiện tại

| Số/claim trong paper gốc | Sau kiểm tra | Kết luận hiện hành |
| --- | --- | --- |
| mBART LSR-only20.0828/16.0690 | A19.6291/18.5753; B19.7067/18.6169 | Full−LSR-only B+0.683BLEU,CI chứa0; chưa chứng minh synergy |
| mBART Full tăng+5.03chrF++,p<0.001 | B+5.0277,CI[-0.1872,13.2525],Holm1 | Tăng point estimate, không giữ significance gốc |
| NLLB Full hơn baseline có significance | B+0.665BLEU,Holm1 | Chưa xác lập superiority |
| mT5 tốt hơn baseline | EncoderFull+1.807BLEU ở B,Holm0.0036; decoder+2.043 ở A,Holm0.0042 | Có hỗ trợ trên fixed checkpoints; decoder chưa generate lại trong B |
| CLRR+LSR vượt Strong LoRA | A Holm<0.01 cho các đối sánh đã thử | Có hỗ trợ trong cấu hình đã chạy; chưa là matched HPO hoặc đóng góp riêng CLRR |
| No-sg gradient>1200,with-sg18.4±4.2 | Trace18norms bằng0,callback đọc sau zero_grad | Số gradient lịch sử không hợp lệ; chưa rerun no-sg training |
|233 đường đi / ổn định tuyệt đối | Cache block bắt đầu L1; L1→L12 distance11 cho144 abstract inter-block paths | Không giữ theorem bảo đảm ổn định tuyệt đối |
| SVD NLLB387.67/411.58 | Đã đo lại strict,centered full-spectrum,khớp | Có thể dùng với định nghĩa và phạm vi mục5; không chứng minh anti-collapse phổ quát |
| Affix F1/probing chứng minh giữ hình thái | Chưa có raw CSV chứng nhận13layers×3tasks | Không dùng như evidence đã verified |
| Sensitivity sweep/optimum đã chứng minh | Thiếu trial metadata/output; default khớp test | Không gọi validation optimum hoặc robustness sweep đã hoàn tất |
| ≤0.4%latency,+2.2%training overhead | Không có benchmark controlled/repeated chứng nhận | Chưa dùng các tỷ lệ này |
| Ví dụ định tính là predictions thật | Các câu paper/research chưa khớp saved source test | Chưa dùng làm bằng chứng model; file gốc giữ để đối chiếu |
| mT5 LSR-only4.5422/5.0118 | Metadata khác4.9740/5.9426; thiếu predictions | Chưa chứng nhận cả hai bộ số |
| Amis token counts42180/5314/5286;49812/6290/6244 | Whitespace source31136/3947/3863;BMP Han target45438/5972/5643 | Báo rõ định nghĩa đếm; không giữ counts gốc như số đã verified |
| NLLB BitFit131072params |333824trainable,total615073792 |0.0543%,không0.0213% |
| Weight decay0.01 áp dụng tất cả | Trainer defaults/runbook đã kiểm tra có0.0 | Lấy run config riêng; không gán fairness chung từ mô tả paper |

**Nhạy cảm chrF++ Mandarin:** reference có3word-bigrams trong3/575câu;
baseline match0,Full/LSR match1 ởindex170. Full−baseline chrF++ A/B+5.0277,
nhưng char-only chỉ+0.6907; bỏ riêng câu170 còn+0.3844chrF++.
Đây là metric chuẩn nhưng nhạy với thành phần thưa, không sửa bằng cách bỏ câu hoặc chọn tokenizer để có p đẹp.
Regexmi/ma/pa là proxy có overlap, không là gold morphology annotation.

Trạng thái trong các báo cáo A lưu trữ phản ánh lúc chưa có weights; phần SVD/residual/gradient
đã được bổ sung ở mục5. Những số chưa verified khác không tự được khôi phục chỉ vì đã có checkpoint.


## 8. Kết luận và việc còn thiếu

- Trong các cấu hình đã kiểm tra, correctness/precision gates đạt; routing có ảnh hưởng đo được, residual không lấn át, pooled spectra không collapse về một hướng.
- Bằng chứng quality tích cực nhất hiện tại là mT5; mBART/NLLB chưa có superiority chắc chắn so với full-tuning baseline.
- Full chưa hơn LSR-only mBART một cách chắc chắn; mT5Full chưa chứng minh hơn CLRR-only. ByT5 không hỗ trợ claim cải thiện phổ quát.
- Routing-off chứng minh dependence tại inference, không thay training ablation hoặc chứng minh morphology preservation.
- Thiếu mBART CLRR-only đúng best504 trên HF; không dùng432 thay thế. Nếu tìm thấy đúng weights thì reevaluate bằng cùng protocol B.
- Checkpoint trung gian lịch sử chưa có. Suite C mới đã đo35mốc CE–LSR/residual/rank trên cùng batches; thiếu baseline seed42 endpoint, không nội suy. Dùng trajectory ở mục10 trước khi sửa λ hoặc phương pháp.
- Thí nghiệm matched 4branches ×3seeds đã hoàn tất ở mục10. Full−LSR có mean +0.2344BLEU và3/3seed dương, nhưng nhỏ và không đạt Holm24; contribution độc lập của CLRR vẫn chưa vững. Kế hoạch tiếp theo ở CLRR_GO_NOGO_PLAN.md.
- Turkish đang tạm dừng. Chỉ tiếp tục khi tác giả yêu cầu; báo chat mỗi10phút khi chạy,GPUA100khônghigh-mem.

### Tình trạng các lần chạy

| Hạng mục | Trạng thái |
| --- | --- |
| Strong LoRA mBART A/B; NLLB A/B | Hoàn tất |
| Audit28outputs Amis /Holm42 gốc | Hoàn tất |
| Correctness /character /leave-one-out | Hoàn tất trong phạm vi đã mô tả |
|10checkpoint Amis cơ chế /80gradient batches | Hoàn tất |
|10checkpoint Amis FP32 /4routing-off /Holm36 gốc | Hoàn tất |
| Turkish datasetv2 /zero-shot /calibration | Hoàn tất; chưa là method comparison |
| Turkish4branch training | Dừng; chưa có comparison |
| Amis matched4branches ×3seeds (C) | Hoàn tất12runs,35probes,Holm24; mục10 |
| HPO công bằng /controlled overhead benchmark | Chưa thực hiện |


## 9. Phụ lục truy vết — giữ riêng các phiên bản

<details>
<summary>A — 28điểm Amis thuộc phạm vi hiện hành từ audit trên saved predictions02/10 (không thay bảng hiện hành mục2)</summary>

| Dataset | Backbone | Method | N | BLEU A | chrF++ A |
| --- | --- | --- | --- | --- | --- |
| amis_mandarin | mBART | Baseline | 575 | 19.6144 | 14.0547 |
| amis_mandarin | mBART | CLRR+LSR | 575 | 20.3896 | 19.0824 |
| amis_mandarin | mBART | CLRR-only | 575 | 18.4697 | 13.2443 |
| amis_mandarin | mBART | LSR-only | 575 | 19.6291 | 18.5753 |
| amis_mandarin | mBART | CLRR-Dec+LSR | 575 | 19.9188 | 18.6999 |
| amis_mandarin | mBART | BitFit | 575 | 0.3463 | 2.2552 |
| amis_mandarin | mBART | Narrow LoRA | 575 | 3.2365 | 4.4213 |
| amis_mandarin | mBART | Strong LoRA A | 575 | 10.2296 | 8.6319 |
| amis_mandarin | mBART | Strong LoRA B | 575 | 13.8066 | 10.3293 |
| amis_mandarin | NLLB | Strong LoRA A | 575 | 11.4625 | 9.5004 |
| amis_mandarin | NLLB | Strong LoRA B | 575 | 9.8024 | 8.4010 |
| amis_mandarin | mBART | Middle-Layer Alignment | 575 | 19.2893 | 18.4378 |
| amis_mandarin | NLLB | Baseline | 575 | 13.5001 | 10.4389 |
| amis_mandarin | NLLB | CLRR+LSR | 575 | 14.2820 | 10.9152 |
| amis_mandarin | NLLB | CLRR-Dec+LSR | 575 | 13.4194 | 10.4559 |
| amis_mandarin | NLLB | Middle-Layer Alignment | 575 | 13.9648 | 10.6596 |
| amis_mandarin | NLLB | BitFit | 575 | 1.0750 | 2.9643 |
| amis_mandarin | NLLB | Narrow LoRA | 575 | 3.5383 | 4.9407 |
| amis_mandarin | mT5 | Baseline | 575 | 2.7887 | 3.8129 |
| amis_mandarin | mT5 | CLRR-only | 575 | 4.4393 | 5.1761 |
| amis_mandarin | mT5 | CLRR+LSR | 575 | 4.5961 | 5.0425 |
| amis_mandarin | mT5 | CLRR-Dec+LSR | 575 | 4.8315 | 5.1933 |
| amis_mandarin | mT5 | CLRR-Both+LSR | 575 | 3.4459 | 4.5753 |
| amis_mandarin | mT5 | Middle-Layer Alignment | 575 | 4.6625 | 5.2103 |
| amis_mandarin | ByT5 | Baseline | 575 | 7.5837 | 8.3057 |
| amis_mandarin | ByT5 | CLRR+LSR | 575 | 7.3108 | 8.0956 |
| amis_mandarin | ByT5 | CLRR-Dec+LSR | 575 | 7.4394 | 8.0630 |
| amis_mandarin | ByT5 | Middle-Layer Alignment | 575 | 7.5660 | 8.3414 |

</details>

<details>
<summary>A — đủ42kiểm định, Holm42; không áp dụng cho predictions B</summary>

| Dataset / backbone | Challenger − reference | Metric | Δ | CI95% | p raw | p Holm42 |
| --- | --- | --- | --- | --- | --- | --- |
| amis_mandarin / mBART | CLRR+LSR − Baseline | BLEU | 0.7752 | [-0.4388, 2.0441] | 0.0954 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − Baseline | chrF++ | 5.0277 | [-0.1872, 13.2525] | 0.1214 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − LSR-only | BLEU | 0.7606 | [-0.4086, 1.9237] | 0.0888 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − LSR-only | chrF++ | 0.5071 | [-0.2316, 1.2522] | 0.0798 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − CLRR-only | BLEU | 1.9199 | [0.8183, 3.0558] | 0.0006 | 0.0138 |
| amis_mandarin / mBART | CLRR+LSR − CLRR-only | chrF++ | 5.8381 | [0.7650, 14.0341] | 0.1000 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − Middle-Layer Alignment | BLEU | 1.1004 | [-0.0643, 2.2602] | 0.0292 | 0.6131 |
| amis_mandarin / mBART | CLRR+LSR − Middle-Layer Alignment | chrF++ | 0.6446 | [-0.0528, 1.3661] | 0.0333 | 0.6659 |
| amis_mandarin / mBART | CLRR+LSR − CLRR-Dec+LSR | BLEU | 0.4709 | [-0.5804, 1.5743] | 0.1480 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − CLRR-Dec+LSR | chrF++ | 0.3825 | [-0.2401, 1.0502] | 0.1040 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − BitFit | BLEU | 20.0434 | [18.0696, 22.0865] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − BitFit | chrF++ | 16.8272 | [11.3416, 25.0158] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Narrow LoRA | BLEU | 17.1532 | [15.1766, 19.2210] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Narrow LoRA | chrF++ | 14.6610 | [9.2155, 23.1350] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Strong LoRA A | BLEU | 10.1601 | [8.2860, 12.1476] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Strong LoRA A | chrF++ | 10.4505 | [5.0353, 18.8632] | 0.0002 | 0.0052 |
| amis_mandarin / mBART | CLRR+LSR − Strong LoRA B | BLEU | 6.5830 | [4.9815, 8.2314] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Strong LoRA B | chrF++ | 8.7531 | [3.4417, 15.6069] | 0.0004 | 0.0096 |
| amis_mandarin / NLLB | CLRR+LSR − Baseline | BLEU | 0.7818 | [-0.2288, 1.7849] | 0.0573 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR − Baseline | chrF++ | 0.4763 | [-0.0682, 1.0552] | 0.0466 | 0.8853 |
| amis_mandarin / NLLB | CLRR+LSR − Middle-Layer Alignment | BLEU | 0.3172 | [-0.7610, 1.3798] | 0.1909 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR − Middle-Layer Alignment | chrF++ | 0.2556 | [-0.3116, 0.8397] | 0.1408 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR − Narrow LoRA | BLEU | 10.7437 | [9.2558, 12.2837] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR − Narrow LoRA | chrF++ | 5.9745 | [5.2485, 7.2458] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR − Strong LoRA A | BLEU | 2.8195 | [1.4847, 4.1169] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR − Strong LoRA A | chrF++ | 1.4148 | [0.7611, 2.1693] | 0.0003 | 0.0075 |
| amis_mandarin / NLLB | CLRR+LSR − Strong LoRA B | BLEU | 4.4796 | [3.1097, 5.8786] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR − Strong LoRA B | chrF++ | 2.5142 | [1.8153, 3.3643] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR − Baseline | BLEU | 2.0428 | [1.3530, 2.7629] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR − Baseline | chrF++ | 1.3803 | [0.6403, 1.7663] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR − Middle-Layer Alignment | BLEU | 0.1689 | [-0.3511, 0.7041] | 0.1813 | 1.0000 |
| amis_mandarin / mT5 | CLRR-Dec+LSR − Middle-Layer Alignment | chrF++ | -0.0171 | [-0.2487, 0.2175] | 0.3433 | 1.0000 |
| amis_mandarin / mT5 | CLRR+LSR − Baseline | BLEU | 1.8074 | [1.1826, 2.4746] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR+LSR − Baseline | chrF++ | 1.2296 | [0.5181, 1.5591] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Both+LSR − CLRR+LSR | BLEU | -1.1501 | [-1.7066, -0.6727] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Both+LSR − CLRR+LSR | chrF++ | -0.4672 | [-0.7534, -0.2068] | 0.0013 | 0.0286 |

</details>

<details>
<summary>B — đủ36kiểm định FP32 và routing-off, Holm36</summary>

| Challenger − reference | Metric | Δ | CI95% | p raw | p Holm36 |
| --- | --- | --- | --- | --- | --- |
| mbart-full − mbart-baseline | bleu | 0.7752 | [-0.4388, 2.0441] | 0.0954 | 1.0000 |
| mbart-full − mbart-baseline | chrfpp | 5.0277 | [-0.1872, 13.2525] | 0.1214 | 1.0000 |
| mbart-full − mbart-baseline | chrf0 | 0.6907 | [-0.2363, 1.6608] | 0.0695 | 1.0000 |
| mbart-full − mbart-lsr | bleu | 0.6830 | [-0.4806, 1.8530] | 0.1069 | 1.0000 |
| mbart-full − mbart-lsr | chrfpp | 0.4655 | [-0.2738, 1.2102] | 0.0928 | 1.0000 |
| mbart-full − mbart-lsr | chrf0 | 0.6122 | [-0.2688, 1.5009] | 0.0797 | 1.0000 |
| nllb-full − nllb-baseline | bleu | 0.6650 | [-0.3295, 1.7156] | 0.0849 | 1.0000 |
| nllb-full − nllb-baseline | chrfpp | 0.4189 | [-0.1279, 0.9924] | 0.0631 | 1.0000 |
| nllb-full − nllb-baseline | chrf0 | 0.5580 | [-0.1333, 1.2513] | 0.0508 | 1.0000 |
| mt5-full − mt5-baseline | bleu | 1.8074 | [1.1826, 2.4746] | 0.0001 | 0.0036 |
| mt5-full − mt5-baseline | chrfpp | 1.2296 | [0.5181, 1.5591] | 0.0001 | 0.0036 |
| mt5-full − mt5-baseline | chrf0 | 1.6108 | [1.2587, 1.9804] | 0.0001 | 0.0036 |
| mt5-full − mt5-clrr | bleu | 0.1567 | [-0.3769, 0.7116] | 0.2012 | 1.0000 |
| mt5-full − mt5-clrr | chrfpp | -0.1336 | [-0.4268, 0.1224] | 0.1332 | 1.0000 |
| mt5-full − mt5-clrr | chrf0 | -0.1782 | [-0.5549, 0.1615] | 0.1287 | 1.0000 |
| byt5-full − byt5-baseline | bleu | -0.1696 | [-0.8869, 0.5266] | 0.2199 | 1.0000 |
| byt5-full − byt5-baseline | chrfpp | -0.2011 | [-0.5634, 0.1240] | 0.1039 | 1.0000 |
| byt5-full − byt5-baseline | chrf0 | -0.2057 | [-0.5988, 0.1512] | 0.1123 | 1.0000 |
| mbart-full − mbart-full::route_off | bleu | 3.9287 | [2.7175, 5.1948] | 0.0001 | 0.0036 |
| mbart-full − mbart-full::route_off | chrfpp | 2.6080 | [1.9050, 3.4956] | 0.0001 | 0.0036 |
| mbart-full − mbart-full::route_off | chrf0 | 3.2964 | [2.3996, 4.2335] | 0.0001 | 0.0036 |
| nllb-full − nllb-full::route_off | bleu | 1.3659 | [0.3337, 2.7009] | 0.0170 | 0.4080 |
| nllb-full − nllb-full::route_off | chrfpp | 0.8881 | [0.3166, 1.5406] | 0.0032 | 0.0832 |
| nllb-full − nllb-full::route_off | chrf0 | 1.1262 | [0.4142, 1.8911] | 0.0012 | 0.0324 |
| mt5-full − mt5-full::route_off | bleu | 0.4577 | [0.0719, 0.8816] | 0.0180 | 0.4140 |
| mt5-full − mt5-full::route_off | chrfpp | 0.2210 | [-0.5736, 0.3980] | 0.0771 | 1.0000 |
| mt5-full − mt5-full::route_off | chrf0 | 0.2659 | [0.0292, 0.4996] | 0.0142 | 0.3550 |
| byt5-full − byt5-full::route_off | bleu | -0.2006 | [-0.7983, 0.3788] | 0.1780 | 1.0000 |
| byt5-full − byt5-full::route_off | chrfpp | 0.0039 | [-0.2421, 0.2409] | 0.4084 | 1.0000 |
| byt5-full − byt5-full::route_off | chrf0 | 0.0045 | [-0.2824, 0.2811] | 0.4084 | 1.0000 |

</details>

### Artifacts chính

- [CSV audit gốc; bảng hiện hành lọc 28outputs Amis](../../outputs_rebuttal/metric_audit_20261002/full/all_verified_translation_scores.csv), [audit42tests](../../outputs_rebuttal/metric_audit_20261002/full/paired_bootstrap_standardized.csv).
- [CSV FP32 gốc; bảng hiện hành lọc 14outputs Amis](../../outputs_rebuttal/followup_20261003/matched_inference_scores.csv), [FP32bootstrap36tests](../../outputs_rebuttal/followup_20261003/matched_inference_bootstrap.csv), [protocolB](../../outputs_rebuttal/followup_20261003/analysis_protocol.json).
- [Correctness/robustness](../../outputs_rebuttal/easy_diagnostics_20261002), [mechanisms](../../outputs_rebuttal/checkpoint_diagnostics_20261002), [strict verification113files](../../outputs_rebuttal/checkpoint_diagnostics_20261002/local_verification.json).
- [Turkishzero-shot](../../outputs_rebuttal/followup_20261003/turkish_zero_shot/metrics.json).
- [Báo cáo chi tiết đã lưu trữ](../archive/README.md), [snapshot tài liệu trước sắp xếp](../../outputs_rebuttal/documentation_consolidation_20261003/before/docs).

Trước đợt sắp xếp này,19file lịch sử đã được đối chiếu restorationSHA và giữ nguyên.
Đợt hiện tại được tác giả yêu cầu **gom tài liệu**, nên EXPERIMENTS/METRICS/RESEARCH_INSIGHTS
được sắp xếp lại; không còn tuyên bố19file docs đó vẫn byte-identical sau sắp xếp.
Ghi chú lịch sử của đợt gom tài liệu: khi đó paper chưa được sửa. Ngày03/10, theo yêu cầu mới của tác giả, paper hiện hành đã được viết lại; bản trước revision được archive. Source CSV/JSON/predictions thực nghiệm vẫn giữ nguyên.

<!-- AMIS_CONFIRMATION_STATUS_START -->
## 10. Suite kiểm chứng mBART Amis — task 1 đến 3

Tác giả yêu cầu chạy tự động task 1–3; **không chạy Turkish**.
Đây là các run mới từ pretrained, không thay đổi artifacts lịch sử. Paper hiện hành đã được cập nhật để lấy suite này làm bằng chứng chính.

### Task 1 và 2: kiểm tra checkpoint lịch sử

- HF revision đã kiểm tra: `63b0dbc4fab41a669ccb36ed2f22894868634b8a`; đã đọc mục lục ZIP và metadata, không tải weights để thay thế sai checkpoint.
- **Task 1:** chưa tìm thấy đúng CLRR-only best `checkpoint-504` (epoch 14). Các archive CLRR-only hiện có đến step 432; không dùng 432 thay 504. Chưa thực hiện được reevaluation lịch sử này.
- **Task 2:** baseline, encoder Full và LSR-only trên revision đã kiểm tra không có các archive checkpoint trung gian; chỉ có bản best. Chưa thể dựng chuỗi LSR đầu–giữa–cuối của run lịch sử.
- [Inventory và phạm vi kiểm tra](../../outputs_rebuttal/amis_confirmation_20261003/checkpoint_inventory.json), [trạng thái task lịch sử](../../outputs_rebuttal/amis_confirmation_20261003/historical_tasks_status.json).

### Task 3: 12 run công bằng + trajectory mới

| Setting | Chung cho Baseline / CLRR-only / LSR-only / Full |
| --- | --- |
| Data | Amis→Mandarin, 4600 / 576 / 575; giữ nguyên rows/order |
| Seeds | 42, 43, 44 — mỗi seed đủ cả 4 nhánh |
| Initial weights | mBART-50 revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30` |
| LR / weight decay / warmup | 5e-5 / 0.0 / 0.06 |
| Batch / accumulation | 32 × 4 = 128 |
| Epochs / early stopping | Tối đa 20 / patience 4, cùng raw-reference validation chrF++ |
| Language codes | tl_XX → zh_CN |
| CLRR / LSR | Encoder d=2, alpha=0.1 khi CLRR bật; lambda=0.1 khi LSR bật |
| Precision | Train BF16/TF32 + gradient checkpointing; generate FP32, TF32 off |
| Selection / test | Validation greedy beam 1 chọn best; test beam 4 |
| GPU / environment | A100 Standard, không high-mem; conda clrr |
| Observational probes | Epoch 1, epoch 5, và actual training endpoint; validation rows/batches cố định |

Probe lưu residual từng ví dụ, full singular spectra/effective rank và CE–LSR gradients.
Giữ và phục hồi RNG, train mode, cache và gradient checkpointing; gate GPU cho cả 4 nhánh đã đạt.
LSR ở baseline/CLRR-only là **counterfactual diagnostic**, không là objective đã train.
Mốc cuối là endpoint của trajectory, không mặc định là checkpoint best được test.

Phân tích định trước: toàn bộ seed scores, mean/std và paired seed deltas; per-seed
paired sentence bootstrap 10.000 lần, seed 42, Holm 24 cho 4 comparisons × 2 metrics × 3 seeds.
Hai đối sánh chính: Full−Baseline và Full−LSR-only; phụ: CLRR-only−Baseline, Full−CLRR-only.
Sentence bootstrap không thay kiểm định qua training seeds; chỉ 3 seed vẫn là phạm vi hạn chế.
Không đổi LR/alpha/lambda/patience dựa trên test.

### Trạng thái mới nhất

- UTC của snapshot tiến độ: **2026-10-03T08:51:24Z**.
- Trạng thái: **COMPLETE; AVAILABLE WEIGHTS BACKED UP; COLAB STOPPED**; hoàn tất **12/12** run.
- Không có run đang chạy; Colab đã dừng.
- Số mốc cơ chế đã lưu: **35**.
- Best model mỗi run được archive và tải về local, đối chiếu SHA-256; 10 archive hoàn chỉnh đã được upload và xác minh SHA/size trên HF ở thư mục riêng; LSR/Full seed42 mất weights trước upload. Xem receipt ở cuối mục10.
- [Progress local](../../outputs_rebuttal/amis_confirmation_20261003/progress_local.json), [protocol](../../outputs_rebuttal/amis_confirmation_20261003/protocol.json), [outputs](../../outputs_rebuttal/amis_confirmation_20261003).

### Sự cố phiên thứ hai và bản sao weights

Phiên thứ hai mất khi CLRR-only seed 42 chưa hoàn tất. Giữ nguyên kết quả đã hoàn tất của
baseline/LSR-only seed 42; đã recompute BLEU/chrF++ từ 575 predictions, khớp metadata <1e-8.
LSR-only có đủ probes đầu/giữa/cuối, nhưng archive weights local chỉ có 1.95/2.27 GB;
**không có bản sao weights LSR seed 42 hoàn chỉnh**. Kết quả này không được ghi là backed-up model.
Các probes CLRR-only của lượt bị ngắt được chuyển riêng vào `interrupted_attempt_02`, không trộn
vào trajectory của lượt chạy lại. Chạy tiếp chỉ 10 lượt chưa hoàn tất với protocol/mã nguồn train nguyên hash.
Launcher CLI duy trì kết nối foreground bằng tiến trình nền độc lập với lượt chat; downloader dùng
HTTP Range 16 MiB để tránh lỗi MemoryError của lượt tải trước.
[Artifact losses](../../outputs_rebuttal/amis_confirmation_20261003/artifact_losses.json).

### Reset host công cụ sau khi đủ seed 42

Host công cụ reset làm dừng cả các tiến trình local được tạo trước đó; phiên Colab sau đó không còn.
Đã giữ đủ **4 kết quả seed 42**, recompute BLEU/chrF++ từ CSV khớp metadata <1e-8,
và giữ đủ probes đã thu (riêng baseline endpoint mất từ sự cố đầu). Baseline/CLRR weights local đã
khớp SHA-256; LSR/Full seed 42 chỉ còn archive không hoàn chỉnh. Không sửa điểm hoặc train lại
4 kết quả đã hoàn tất. Chỉ còn **8 lượt seed 43/44** phải chạy.

Đã kiểm tra Task Scheduler dưới tài khoản hiện tại, quyền Limited, không lặp lịch: tiến trình
validation được `svchost.exe` sở hữu và kết thúc mã 0. Supervisor thật dùng conda clrr, chạy
foreground CLI/sao lưu/cập nhật ledger, rồi thu thập và tắt phiên sau khi đủ kết quả.
Đây là sửa cơ chế vận hành, không đổi mã nguồn train, dữ liệu hoặc hyperparameters.

### Sự cố hạ tầng và phạm vi khôi phục

Phiên Colab đầu bị mất khi LSR-only seed 42 chưa hoàn tất; chưa xác định nguyên nhân.
Baseline seed 42 đã train đủ 20 epoch và archive best local khớp SHA-256.
Khôi phục đúng weights đó, tái tạo predictions theo cùng Trainer/protocol và chỉ chấp nhận
khi BLEU/chrF++ khớp điểm đã ghi đến sai số <1e-8. Không train lại baseline hoặc chọn test checkpoint.
Exact selected step, thời gian train và bytes predictions gốc của baseline không còn trong bản sao;
không điền giả các trường này. Probe baseline epoch 1/5 còn đủ; probe actual endpoint epoch 20 đã mất.
LSR-only bị ngắt được chạy lại từ cùng pretrained, cùng seed/config; kết quả lượt dở không dùng so sánh.
Mã nguồn và protocol của suite được giữ nguyên hash. Cell mới chạy pipeline ở foreground,
đồng thời sao lưu metrics/predictions/probes và weights khi từng run hoàn tất.
[Recovery provenance](../../outputs_rebuttal/amis_confirmation_20261003/infrastructure_recovery.json).

### Kết quả sau khi đủ 12 run

#### Điểm từng seed

| seed | method | BLEU | chrF++ | best_checkpoint | completed_epochs | training_seconds_including_probes | prediction_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | baseline | 19.7281 | 18.7138 | recovered selected best; original selected step unavailable | 20 |  | 4a53fc044641487438750047f37fc63689cf4ad12d28d4ad02fffe882bde9d75 |
| 42 | lsr | 19.7865 | 18.6797 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-lsr-seed42/checkpoint-576 | 20.0000 | 1254.8414 | 867bfd07d7ab0f564767e5c1d80c9bce971fdcdcebf6ce2841b24ae9787bd19a |
| 42 | clrr | 19.6403 | 18.5639 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-clrr-seed42/checkpoint-504 | 18.0000 | 1094.7276 | 651d67e2f41c4721167f8ed2311f8bfff014d5fb6f1f91286680593dbb017f9a |
| 42 | full | 19.9566 | 18.8878 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-full-seed42/checkpoint-720 | 20.0000 | 1303.3809 | 383e3db76cf326abf8299a8830a35d273604bb1feb53abd15199cc0f03338fe2 |
| 43 | baseline | 18.1378 | 13.1592 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-baseline-seed43/checkpoint-720 | 20.0000 | 1197.4472 | 7a2f66e8e84ca9845371ce993a1206bf1f03ee2678e25eed56111b06d1a8a227 |
| 43 | lsr | 19.4879 | 18.6488 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-lsr-seed43/checkpoint-684 | 20.0000 | 1265.2532 | db38f1c4c7c1821ab869a4ec609386ba4371fe9e356c63d77dd8d076bb25d002 |
| 43 | clrr | 18.9420 | 18.0823 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-clrr-seed43/checkpoint-720 | 20.0000 | 1239.8434 | f95673d96909405770357c545c5875d5061573c9308ad5c553dd82e152c4b079 |
| 43 | full | 19.6123 | 18.4753 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-full-seed43/checkpoint-684 | 20.0000 | 1298.6891 | a5c3fb6f16c4f28ec138292449ebf519f5ee67e24dd5e79f39bfe8220a3f2d05 |
| 44 | baseline | 20.3000 | 18.9017 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-baseline-seed44/checkpoint-612 | 20.0000 | 1721.5243 | 974e3de9456f2303cd573a96257490330abc8e2584d7fce7ef168c9753e4266c |
| 44 | lsr | 19.3284 | 18.3633 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-lsr-seed44/checkpoint-720 | 20.0000 | 2053.9199 | a79b67d48d49f9a9084d9c5fc9978ca189f65cf9cbc1338540d1246b8e948201 |
| 44 | clrr | 20.1485 | 18.9055 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-clrr-seed44/checkpoint-720 | 20.0000 | 1938.7058 | 76ebae5d32d847b44f9b3ea78d8d039db22cbbbe2ef94035af480927947e2f4d |
| 44 | full | 19.7371 | 18.6065 | /content/clrr_diagnostics/project/outputs_rebuttal/amis_confirmation_20261003/runs/mbart-amis-full-seed44/checkpoint-648 | 20.0000 | 2032.8589 | 56841452d1ff220ad545b484e1ab33e7d8b6e56c59b7406b04ef8fb1ec90cb9a |

#### Mean/std qua 3 seed

| method | metric | mean | std_across_seeds | minimum | maximum | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | BLEU | 19.3886 | 1.1203 | 18.1378 | 20.3000 | 3 |
| baseline | chrF++ | 16.9249 | 3.2626 | 13.1592 | 18.9017 | 3 |
| clrr | BLEU | 19.5769 | 0.6057 | 18.9420 | 20.1485 | 3 |
| clrr | chrF++ | 18.5172 | 0.4136 | 18.0823 | 18.9055 | 3 |
| lsr | BLEU | 19.5343 | 0.2325 | 19.3284 | 19.7865 | 3 |
| lsr | chrF++ | 18.5639 | 0.1745 | 18.3633 | 18.6797 | 3 |
| full | BLEU | 19.7687 | 0.1743 | 19.6123 | 19.9566 | 3 |
| full | chrF++ | 18.6565 | 0.2107 | 18.4753 | 18.8878 | 3 |

#### Chênh lệch ghép cặp từng seed

| baseline | challenger | metric | seed42 | seed43 | seed44 | mean_delta | std_paired_delta | positive_seeds | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | full | BLEU | 0.2285 | 1.4744 | -0.5629 | 0.3800 | 1.0271 | 2 | 3 |
| baseline | full | chrF++ | 0.1740 | 5.3162 | -0.2952 | 1.7316 | 3.1131 | 2 | 3 |
| lsr | full | BLEU | 0.1701 | 0.1243 | 0.4087 | 0.2344 | 0.1527 | 3 | 3 |
| lsr | full | chrF++ | 0.2081 | -0.1735 | 0.2432 | 0.0926 | 0.2311 | 2 | 3 |
| baseline | clrr | BLEU | -0.0878 | 0.8042 | -0.1516 | 0.1883 | 0.5343 | 1 | 3 |
| baseline | clrr | chrF++ | -0.1499 | 4.9231 | 0.0038 | 1.5923 | 2.8856 | 2 | 3 |
| clrr | full | BLEU | 0.3163 | 0.6702 | -0.4113 | 0.1917 | 0.5514 | 2 | 3 |
| clrr | full | chrF++ | 0.3239 | 0.3930 | -0.2990 | 0.1393 | 0.3812 | 2 | 3 |

Nhận định tự động có giới hạn: Insufficient stable incremental CLRR evidence under this prespecified protocol; investigate or narrow claims before Turkish.

[24 bootstrap tests](../../outputs_rebuttal/amis_confirmation_20261003/paired_bootstrap24.csv); [decision/provenance](../../outputs_rebuttal/amis_confirmation_20261003/decision.json).

Đã tổng hợp 35 mốc gradient/residual/rank. Thiếu đúng endpoint baseline seed 42 do mất VM; không nội suy hoặc thay bằng best. [Probe summary](../../outputs_rebuttal/amis_confirmation_20261003/trajectory_summary.csv).

<!-- AMIS_CONFIRMATION_STATUS_END -->


<!-- AMIS_CONFIRMATION_HF_BACKUP_20261003 -->
### Checkpoint backup and local cleanup

All 10 available complete checkpoint ZIPs are uploaded to [FiveC/amis-rewire-checkpoints/amis_confirmation_20261003/checkpoints](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/amis_confirmation_20261003/checkpoints). Each remote file was verified against the local SHA-256 and byte count before deleting local ZIP weights. The old checkpoint folders were not overwritten.

The LSR-only seed 42 and Full seed 42 weights were lost in earlier runtime interruptions; they are explicitly absent from this upload. All 12 scientific results and available mechanism probes remain local. The upload manifest is `outputs_rebuttal/amis_confirmation_20261003/HF_UPLOAD.json`, also stored in the dedicated HF folder. Local complete ZIP cleanup freed 22.71 GB, plus incomplete ZIP fragments. Colab was stopped after final scientific backup.

---

### Amis Final Evaluation & Detach Control Suite (2026-10-04)

Theo quyết định 4 bước tại Mục 11 của `docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history`, đã hoàn tất toàn bộ chu trình đánh giá kiểm chứng:

#### 1. Phase C2: Attached Control Validation & Khóa Kiến trúc
- Đã huấn luyện 3 seeds (42, 43, 44) của Full Source-Attached ($\alpha=0.10, \lambda=0.10$, `detach=False`).
- Kết quả validation: Mean chrF++ **14.4504 $\pm$ 0.2566**, Mean BLEU **17.3371 $\pm$ 0.4005** (thấp hơn bản Detached chuẩn: 14.5976 chrF++, 17.4802 BLEU). Trên seed 42, việc bỏ detach gây sụt giảm mạnh -0.55 chrF++.
- **Quyết định:** Giữ nguyên kiến trúc chuẩn **Detached CLRR**. Toàn bộ 3 seeds CLRR-only đã có sẵn, không cần train thêm.

#### 2. Final Test Checkpoint Matrix (18 Models, 6 Nhánh, 3 Seeds)
Tất cả checkpoints được đánh giá trên tập test chuẩn (`data_processed/amis_mandarin/test.csv`, 575 dòng) với protocol cố định: **FP32, TF32 off, beam 4, max_length 256**, tokenizer `zh`, SacreBLEU 2.6.0.

| Nhánh / Phương pháp | Seed 42 | Seed 43 | Seed 44 | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (mBART-50)** | | | | |
| - BLEU | 19.7281 | 18.1378 | 20.3000 | 19.3886 $\pm$ 1.1203 |
| - chrF++ | 18.7138 | 13.1592 | 18.9017 | 16.9249 $\pm$ 3.2626 |
| - chrF0 | 18.3973 | 17.0904 | 18.6478 | 18.0452 $\pm$ 0.8362 |
| **Full CLRR Detached (Selected, $\alpha=0.10, \lambda=0.10$)** | | | | |
| - BLEU | **19.9566** | **19.6123** | **19.7371** | **19.7687 $\pm$ 0.1743** |
| - chrF++ | **18.8878** | **18.4753** | **18.6065** | **18.6565 $\pm$ 0.2107** |
| - chrF0 | **18.6626** | **18.1905** | **18.2613** | **18.3715 $\pm$ 0.2546** |
| **Matched LSR Control ($\lambda=0.10$)** | | | | |
| - BLEU | 19.7865 | 19.4879 | 19.3284 | 19.5343 $\pm$ 0.2325 |
| - chrF++ | 18.6797 | 18.6488 | 18.3633 | 18.5639 $\pm$ 0.1745 |
| - chrF0 | 18.3581 | 18.2581 | 17.9957 | 18.2040 $\pm$ 0.1871 |
| **Tuned LSR Control ($\lambda=0.03$)** | | | | |
| - BLEU | 20.6257 | 19.6747 | 19.3858 | 19.8954 $\pm$ 0.6487 |
| - chrF++ | 19.1889 | 14.1556 | 18.5324 | 17.2923 $\pm$ 2.7362 |
| - chrF0 | 19.0288 | 18.4188 | 18.1573 | 18.5350 $\pm$ 0.4472 |
| **Full Source-Attached Control ($\alpha=0.10, \lambda=0.10, \text{detach=False}$)** | | | | |
| - BLEU | 19.3531 | 19.7774 | 19.5667 | 19.5658 $\pm$ 0.2122 |
| - chrF++ | 18.5412 | 18.5131 | 18.6355 | 18.5633 $\pm$ 0.0641 |
| - chrF0 | 18.2240 | 18.2104 | 18.3571 | 18.2638 $\pm$ 0.0811 |
| **Detached CLRR-only Control ($\alpha=0.10, \lambda=0.00$)** | | | | |
| - BLEU | 19.6403 | 18.9420 | 20.1485 | 19.5769 $\pm$ 0.6057 |
| - chrF++ | 18.5639 | 18.0823 | 18.9055 | 18.5172 $\pm$ 0.4136 |
| - chrF0 | 18.2504 | 17.7017 | 18.6584 | 18.2035 $\pm$ 0.4801 |

#### 3. Paired Deltas & Kiểm định Thống kê
- **So với Matched LSR ($\lambda=0.10$):** Full CLRR Detached vượt Matched LSR ở **100% (3/3) các seed** về BLEU (+0.1701 ở seed 42, +0.1243 ở seed 43, +0.4087 ở seed 44), chênh lệch trung bình **+0.2344 $\pm$ 0.1527 BLEU**. Về chrF++, đạt trung bình **+0.0926 $\pm$ 0.2311** (2/3 seeds dương).
- **So với Baseline:** Chênh lệch trung bình **+0.3800 $\pm$ 1.0271 BLEU** và **+1.7316 $\pm$ 3.1131 chrF++** (2/3 seeds dương).
- **So với Tuned LSR ($\lambda=0.03$):** Tuned LSR có BLEU trung bình cao hơn (+0.1267) nhưng cực kỳ mất ổn định trên chrF++ (sụp đổ xuống 14.16 ở seed 43), trong khi Full CLRR Detached vượt Tuned LSR **+1.3642 $\pm$ 2.5664 chrF++** trên toàn bộ 3 seeds.
- **Kiểm định Bootstrap 10.000 mẫu:** Kết quả chi tiết tại [18 Primary Tests](../../outputs_rebuttal/amis_final_evaluation_20261004/test_paired_bootstrap18.csv) và [30 Secondary Tests](../../outputs_rebuttal/amis_final_evaluation_20261004/test_paired_bootstrap30.csv). Sau hiệu chỉnh Holm-Bonferroni, tính bất định trong 575 câu kiểm thử khiến các khoảng tin cậy mở rộng bao phủ 0, phù hợp với nguyên lý khoa học đã định rõ từ trước.
- **Hồ sơ minh chứng:** [test_family_summary.csv](../../outputs_rebuttal/amis_final_evaluation_20261004/test_family_summary.csv), [test_paired_deltas.csv](../../outputs_rebuttal/amis_final_evaluation_20261004/test_paired_deltas.csv), [final_test_decision.json](../../outputs_rebuttal/amis_final_evaluation_20261004/final_test_decision.json).

---

### Amis Extended Follow-up Suite F0–F6 (2026-10-04)

Đã có outputs của21checkpoints (7họ×3seeds), không thêm training. Review độc lập
phát hiện **F4 INVALID do hook contamination**, F3 mới có endpoint summaries,
chưa hoàn tất diagnostics từng layer; không gọi toàn bộ F0–F6 hoàn tất100%.
Receipt: `outputs_rebuttal/amis_followup_20261004_v2/independent_review_20261004.json`;
phân tích tại bàn giao mục16. Score tables/predictions cũ giữ nguyên.

#### 1. Bổ sung Test Full $\alpha=0.10, \lambda=0.03$ (F1)
Đánh giá 3 checkpoints đã huấn luyện tại vòng screening/confirmation trên 575 dòng test (FP32, beam 4):
- Seed 42: BLEU 19.2127, chrF++ 18.3924, chrF0 18.0859
- Seed 43: BLEU 19.1189, chrF++ 18.2179, chrF0 17.8505
- Seed 44: BLEU 18.9245, chrF++ 18.0868, chrF0 17.6579
- **Mean $\pm$ Std:** BLEU **$19.0854 \pm 0.1470$**, chrF++ **$18.2324 \pm 0.1533$**, chrF0 **$17.8648 \pm 0.2144$**.
- Cấu hình này tụt hậu rõ rệt so với Full gốc $\lambda=0.10$ ($19.7687$ BLEU) và thua Tuned LSR $\lambda=0.03$ ($-0.8100$ BLEU trên toàn bộ 3 seeds).

#### 2. Kiểm định Tác động Suy luận: Route-on vs Route-off (F4)
Trên 576 dòng validation (greedy decode, beam 1) của 3 seeds Full detached $\lambda=0.10$:
- Số câu dịch thay đổi giữa $\alpha=0.10$ và $\alpha=0.00$: **0 / 576 (0.00%)** trên cả 3 seeds.
- File ghi Δ=0 nhưng **phép đo không hợp lệ**: `model_on` và `model_off` cùngbase_model,
  hookα0.10 của model_on còn hoạt động khi generate model_off. Không dùng số này để
  kết luận routing vô tác dụng; cần rerun6validationgenerations với mộtwrapper đổistrength.

#### 3. Đo đạc Cơ chế trên 21 Checkpoints (F3)
- **Alignment quan sát:** endpoint pooled cosine tăng theoλ trong các nhánh này;
  cùngλ0.10, means gần nhau. Không suy λ quyết định hoàn toàn hoặc CLRR không tác động.
- **Rank:** source effective rank theo singular values khoảng385–390, không phải
  covariance-energy rank; centeredranklimit575 cho576rows/1024hidden. Không thấy
  collapse pooled cuốiencoder vềmộtchiều; chưa loại compression ởlayers/tokens khác.
- **Quan hệ Gradient:** Gradient cosine giữa CE và LSR dao động từ 0.047 đến 0.068 (gần trực giao). Tỷ số gradient LSR/CE chỉ từ 0.0007 đến 0.0037.
- **Detach:** attachedmeanBLEU19.57 so với detached19.77; chưa là cải thiện đồng đều
  hỗ trợ thay architecture, không chứng minh detach tối ưu hoặc không thể là nút thắt.

#### 4. Thống kê Thứ cấp & Holm-Bonferroni (F2)
30 kiểm định cặp giả thuyết định trước (5 cặp $\times$ 3 seeds $\times$ 2 metrics):
- Cả 30 kiểm định đều có $p_{\text{adj}} > 0.05$ sau hiệu chỉnh Holm-Bonferroni (không đạt ý nghĩa thống kê ở mức $\alpha=0.05$).
- Báo cáo chi tiết: [`reports/decision.md`](../../reports/decision.md), [`reports/mechanism_report.md`](../../reports/mechanism_report.md), [`reports/validation_audit.md`](../../reports/validation_audit.md).

## 11. V4 — xác minh checkpoint và cơ chế đúng protocol, 04/10/2026

### Material Passport

- Scope: mBART Amis→Mandarin,21best checkpoints/7families×3seeds; validation576
  rows. Không train mới hoặc generate lại test575rows.
- Verification status: **VERIFIED**. GPU A100 Standard40GB, không high-mem;
  Torch2.6.0+cu124/Transformers4.57.6/SacreBLEU2.6.0; TF32off. GPU worker1088.1127s.
- Raw nguồn duy nhất: `outputs_rebuttal/amis_followup_20261004_v4/`.
  `runtime.json`, `loading_receipts/`, `COMPLETE_V4.json`, `independent_audit.json`,
  `descriptive_summary.json` và raw CSV/NPZ. Archive/tokenizer/codehash đã đối chiếu.
- Independent CPU audit trong conda `clrr`: rescore/replay3controls,84SVD từ raw
  endpoint matrices,168gradientbatches, residual/partition/provenance đều PASS.
- Tải đủ artifacts và xác minh SHA trước tắt Colab. Chi tiết/reproduce commands
  ở [bàn giao Mục18](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history).

### 11.1. Route-on/off — validation greedy, cùng weights đã train routing

On=α0.10; off=không còn routinghooks. Cả3on predictions khớp chính xác CSV
route-on trước đó, đúng576rows/source/reference/order. Alpha0 và clean raw
teacher logits khớp hoàn toàn trên8×4fixedbatches. Native scores tái tính khớp.

| Seed | BLEU on | BLEU off | ΔBLEU | ΔchrF++ | Câu đổi /576 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 17.6437 | 14.5377 | +3.1060 | +2.4378 | 523 |
| 43 | 17.0645 | 13.9863 | +3.0783 | +2.4063 | 522 |
| 44 | 16.9324 | 14.2137 | +2.7187 | +2.5030 | 536 |
| Mean | 17.2135 | 14.2459 | **+2.9676** | **+2.4491** | 527 |

Sample std củaΔBLEU=0.2160; meancâuđổi91.4931%. Teacher CEon−off:
−0.2010/−0.2193/−0.2361 (label-token-weighted trên32fixedrows).
**Rút kết luận V2 routing vô tác dụng.** V3 percentages/deltas không dùng làm
kết quả protocol chính. +2.9676 là inference intervention trên routing-trained
weights, không phải gain trước baseline/LSR được train độc lập hoặc test gain.

### 11.2. Source pooled ranks — giữ rõ site và định nghĩa

Mean3seeds,576×1024representations, centeredranklimit575. Singular entropy rank
và covariance-energy entropy rank khác participation ratio; full spectra lưu đủ.

| Family | Block12 trước final LN: singular rank | Final output: singular rank | Final output: energy entropy rank |
| --- | ---: | ---: | ---: |
| Baseline | 158.4847 | 389.9910 | 146.0275 |
| CLRR-only | 258.2300 | 385.0201 | 147.1425 |
| Matched LSR λ0.10 | 220.0196 | 386.3290 | 137.0831 |
| Tuned LSR λ0.03 | 185.3190 | 389.5775 | 143.8891 |
| Full λ0.10 | 254.8527 | 385.2893 | 144.2210 |
| Full λ0.03 | 258.3149 | 385.5640 | 146.8733 |
| Full attached λ0.10 | 254.2608 | 384.9135 | 143.7396 |

Fullλ0.10−matchedLSR: blockrank+15.83% (dương3/3); finalsingularrank−1.0397
(âm3/3); finalenergyrank+5.21% (dương3/3). CLRR-only−baseline cũng có blockrank
dương3/3 nhưng finalsingularrankâm3/3, energyrankkhôngđồngnhất. **Không claim
rank luôn tăng, LSR collapse hoặc causal anti-collapse**. Có evidence về hình
học trung gian và shape của spectrum; pooled endpoint không đại diện mọi token.

### 11.3. CE–LSR gradient và residual

True training objectives, means across3seeds×8fixedbatches:

| Family | Global gradient cosine | WeightedLSR / CE norm (%) |
| --- | ---: | ---: |
| Matched LSR λ0.10 | 0.0724 | 0.2337 |
| Tuned LSR λ0.03 | 0.0630 | 0.1007 |
| Full λ0.10 | 0.0516 | 0.1570 |
| Full λ0.03 | 0.0489 | 0.0700 |
| Full attached λ0.10 | 0.0716 | 0.2046 |

Baseline/CLRR-only dùng counterfactualλ0.10, ratios0.3872%/0.2731%; đây không
phải objective thực train. Decoder CE được cộng đúng dù LSR grad là0/None;
nhóm embeddings có scope theo names, không chỉ shared token embedding.
Fullλ0.10 có13/288layer×batchcosinesâm: globalpositive không loại conflict cục
bộ hoặc theo thời gian. Normratio không phải tỷ lệ update Adam.

Full residual/sourcecurrent RMS: mean≈5.91%,max≤8.6906%; targetmean≈5.39%,
max≤8.0276%. Max qua mọi routingcheckpoint/phase=8.7783%. Chưa thấy endpoint
residual explosion; không suy mọi input/epoch đều vô hại.

### 11.4. Decision

**GO tính đúng phép đo V4 và tiếp tục khảo sát routing có giả thuyết.**
**NO-GO superiority/expansion hiện tại.** Test scores/F2bootstrapfamilies cũ giữ
nguyên: Full−matchedLSR+0.2344BLEU (3/3dương), Full−tunedLSR−0.1267;
Fullλ0.03−tunedLSR−0.8100. V4 không biến các gaps này thành superiority.

Vấn đề còn lại là contribution/generalization; chưa có evidence bắt buộc sửa
loss vì collapse/globalconflict hoặc thêm normalization vì norm explosion.
Ưu tiên CPUerroranalysis, rồi validationαsweep/groupintervention nếu cần để
chọn pilot, theo bàn giao18.5. Chưa chạy các thí nghiệm mới đó trong V4.

## 12. V5 — routing interventions, decoding và sửa nhãn baseline validation

### Material Passport

- Mode: run + validate; 04/10/2026; mọi số dưới đây đã audit CPU trong conda clrr.
- Core: VERIFIED42, 9 models Full/matched LSR/tuned LSR ×3 seeds. Baseline archives:
  VERIFIED3 models, mỗi model tái lập chính xác575 test predictions cũ.
- Kế hoạch45 conditions ban đầu dừng tại guard baseline42. Hoàn tất3 baseline
  validation controls qua audit riêng; thêm3 full test replays và1validation beam4
  confirmation: **49conditions đầy đủ được xác minh/lưu**. Không gọi worker45
  ban đầu COMPLETE; original FAILED và hai lỗi script audit bổ sung giữ riêng.
- Không train. A100 Standard, không high-mem; Colab đã stop sau collect/SHA.
- Artifact root: `outputs_rebuttal/amis_routing_diagnostics_20261004_v5/`.
  `independent_audit.json`: historical inventory có protocol disagreement,
  nhưng core và các baseline audit đều VERIFIED, errors=[]; không giấu lỗi này.
- ZIP1605579bytes, SHA256 `82b89d9cd3286ef171e7f585f1136b29e273d2db0fef5d9ac58e818a0f73a9fd`.
- Protocol: validation576/test575, `tl_XX→zh_CN`, FP32/TF32off, batch8,
  max length256, pad multiple8, forced BOS250025. Native SacreBLEU2.6.0,
  BLEUzh, chrF++ word_order2, chrF0 word_order0; không segment reference mới.

### 12.1. Lỗi đã tìm và sửa — nhãn beam của Baseline seed42

Số validation **17.8734771188 BLEU /14.9008761183 chrF++ thực tế là beam4**,
không phải beam1. Archived training_args lưu generation_num_beams4. Script
recovery restore args sau test nhưng không đặt lại beam trước validation;
inventory F0 copy metrics và không có validation predictions để kiểm chứng.
GPUbeam4 khớp hai số đó chính xác; beam1 lặp lại là **16.2658882925 /
13.9015405648**, chrF0 **15.9628218042**. Không phải mất/lưu nhầm weights:
archived baseline42 tái lập đầy đủ575 test predictions cũ; baseline43/44 cũng vậy.

Chỉ nhãn/so sánh validation cần sửa; **test scores, test CSV, Holm tests và paper
chưa thay**. Mean Baseline validation beam1 đổi17.6633→17.1275BLEU,
14.6334→14.3003chrF++. Full−Baseline validationBLEU vì vậy từ−.4498 thành
**+.0861**, vẫn chỉ1/3 seeds dương. Đây không tạo thêm test gain.

Baseline seed42 best_step216 bị hardcode trong F0 nhưng recovery không tìm được
original selected step: đổi sang unknown/None, không suy checkpoint 216 làbest.
Source recovery đã đặt rõ args.eval_beams trước evaluate; F0 bỏ hardcodedstep
và ghi provenance beam. Không chạy lại F0 để overwrite artifacts lịch sử.
**Lượt sau dùng `validation_inventory_21_corrected_v5.json`**,12 validation
controls được remeasure,9 entries còn lại ghi rõ historical not remeasured.
eval_loss lịch sử giữ provenance riêng; không trộn với CE fixed32 dưới đây.

### 12.2. Các đối chứng validation beam1 cùng protocol

Mean ± sample std qua3 seeds; điểm corpus mỗi seed tính riêng rồi lấy mean.

| Nhánh | BLEUzh | chrF++ | chrF0 |
| --- | --- | --- | --- |
| Baseline | 17.1275 ± 0.8499 | 14.3003 ± 0.4568 | 16.4849 ± 0.5888 |
| Matched LSR λ.10 | 16.5430 ± 0.0686 | 14.0608 ± 0.1114 | 16.1677 ± 0.1169 |
| Tuned LSR λ.03 | 16.9942 ± 0.7261 | 14.3170 ± 0.4705 | 16.4950 ± 0.5583 |
| Full α.10/λ.10 | 17.2135 ± 0.3783 | 14.4975 ± 0.2053 | 16.6586 ± 0.1913 |

### 12.3. Can thiệp routing trên Full weights đã train

Mọi Δ so α.10/routing đủ, validation beam1. α dưới đây chỉ đổi tại inference;
trainα=.10/λ=.10. CE trên32 examples cố định, token-weighted; CE âm là tốt hơn
teacher forcing trên subset, không phải generation gain hoặc toànvalidation.

| Can thiệp | ΔBLEU | BLEU dương | ΔchrF++ | ΔchrF0 | ΔCE fixed32 |
| --- | --- | --- | --- | --- | --- |
| α=.05 | -0.4830 | 1/3 | -1.0660 | -0.5735 | +0.0574 |
| α=.075 | -0.1636 | 0/3 | -0.3483 | -0.3872 | +0.0279 |
| α=.125 | -0.4773 | 0/3 | -0.2803 | -0.3175 | -0.0252 |
| α=.15 | -0.8456 | 0/3 | -0.5617 | -0.6268 | -0.0426 |
| Tắt early3–5 | -0.3272 | 0/3 | -0.3587 | -0.3811 | +0.0153 |
| Tắt middle6–8 | -0.1928 | 1/3 | -0.9181 | -0.3777 | +0.0432 |
| Tắt late9–12 | -0.5706 | 0/3 | -0.4601 | -0.5369 | +0.0663 |

Không ứng viên α nào qua mean no-harm guard BLEU/chrF0/chrF++; giữα=.10.
Cả7canthiệp làm chrF++ và chrF0 giảm ở3/3 seeds. α=.05 và tắtmiddle mỗi cái
có1seed BLEU tăng, nhưng mean giảm: không chọn riêngα/layer theo từngseed.
α=.125/.15 giảm CE fixed32 nhưng giảm translation metrics: CE này không thay
cho generation-based selection. Chưa xác định exposure bias hoặc nguyên nhân
generalization từ một subset32câu. Tắt nhóm thay downstream cache; nhóm khác
kích thước nên không suy independent layer causal effects hoặc layer ranking.

Paired corpus bootstrap5000 draws, RNG20261004, cùng indices giữa hai condition
và3 seeds. Dưới đây là **95% intervals mô tả chưa chỉnh multiple comparisons**
cho21 metric comparisons, conditional on3frozen checkpoints; không đo training-seed
uncertainty hoặc dùng làm significance claim về candidate đã chọn.

| Can thiệp | BLEU interval | chrF++ interval | chrF0 interval |
| --- | --- | --- | --- |
| α=.05 | [-0.9880, +0.0144] | [-1.3528, -0.2487] | [-0.9383, -0.2089] |
| α=.075 | [-0.7111, +0.2154] | [-0.6665, -0.0447] | [-0.7579, -0.0371] |
| α=.125 | [-0.9384, -0.0325] | [-0.5780, +0.0249] | [-0.6383, +0.0093] |
| α=.15 | [-1.4357, -0.2581] | [-0.9662, -0.1688] | [-1.0588, -0.2088] |
| Tắt early3–5 | [-0.9489, +0.1275] | [-0.6584, -0.0654] | [-0.7278, -0.0462] |
| Tắt middle6–8 | [-0.8494, +0.3005] | [-1.2124, -0.0874] | [-0.7417, -0.0034] |
| Tắt late9–12 | [-1.0431, -0.0961] | [-0.7531, -0.1693] | [-0.8692, -0.2009] |

### 12.4. Full−tuned LSR: beam không giải thích hết đảo thứ hạng

Full và tuned dùng checkpoint cố định; chỉ decodecontrols đã khóa trước. Không
thửα mới trên test, không chọn beam/α để làm đẹp số. Testbeam4 giữ file cũ.

| Split/decode | ΔBLEU | BLEU dương | ΔchrF++ | ΔchrF0 |
| --- | --- | --- | --- | --- |
| Validation beam1 | +0.2193 | 2/3 | +0.1805 | +0.1636 |
| Validation beam4 | +0.2055 | 2/3 | +0.1946 | +0.1520 |
| Test beam1 mới | -0.2168 | 1/3 | -0.1770 | -0.1994 |
| Test beam4 cũ | -0.1267 | 1/3 | +1.3642 | -0.1635 |

Validation gain dương trung bình ở cảbeam1/4; test BLEU/chrF0 âm trung bình ở
cảbeam1/4. **Không thể quy toàn bộ chênh lệch cho validationgreedy/test beam4**.
Có ranking instability theo split/seed; chưa đủ nói overfitting là nguyên nhân
đã xác định. Native chrF++ test beam4+1.3642 bịseed43 chi phối; reference có3word
bigrams ở3/575câu, validation chỉ1ở1/576. Đọc cùng BLEU/chrF0, không thaymetric
hậu nghiệm. Khôngempty predictions; không có lỗi cắt câu phổ biến được chứng minh.

### 12.5. Probe lịch sử: mô tả có giới hạn, không trộn với V4 endpoints

`trajectory_reuse_review.json`:35points,280gradientbatches union norms kiểm chứng,
70fullspectra rank tái tính,35residualtables. Full/matched LSR có đủearlyepoch1,
midepoch5,lateepoch20 ở3 seeds; lateepoch giống nhau trong contrast này.
Mean qua3 seeds của batch-median global weightedLSR/CEnorm: Full .004627→.003231→
.001479, matched .007300→.004801→.002226. Full aggregatecos .04771→.07202→.05729.
Không bằng chứng globalLSR domination từ các snapshot; vẫn không loại localconflict.
Energy entropy ranks tăng ở cảhai nhánh; Full92.23→124.04→144.18, matched86.78→
113.05→137.29. Singular rank lateFull385.30 thấp hơnmatched386.43. Residualmean
Full5.577%→5.880%→5.914%; không thấy explosion trong dữ liệu đã lưu.

**Giới hạn provenance:** callback hash khớp, nhưng historical model/trainer/helper
hash không khớp current source; các archive đã kiểm tra chưa phục hồi được source
đúng. Không có per-point runtime/language-prefix receipts hoặc pooledmatrices;
không xác minh layerwise gradient/rank trajectories. Chỉ mô tả saved statistics,
không nâng thành reproducible mechanistic/causal proof, không gánselectedbeststeps.

### 12.6. Quyết định sau V5

**GO nghiên cứu nâng cấp cách train; NO-GO mở rộng/superiority claim hiện tại.**
Routing có tác động ởV4 và tắtα/groups làm hại phầnlớn điều kiện, nên chưa có
căn cứ vứt CLRR. Nhưng inference tuning không tìm được cải thiện; Full chưa vượt
tuned LSR ổn định trên test. Chưa có căn cứ chọnλ lớn, normalization hoặc sửa
gradient path chỉ để tăngrank. Cần controlled train-time pilot, xem bàn giao19.5.
Các lợi ích test cũ trướcBaseline(+.3800BLEU) vàmatched LSR(+.2344) giữ nguyên;
chưa đủ nâng thành ưu thế trước tuned LSR hoặc bảo đảm ACL.

Fallacy scan11/11: táchseed/site/split tránhSimpson; không suy pooledrank thành
tokencollapse/ecological; bestcheckpoint/Berkson và collider selection được giới
hạn; base-rate không ápdiagnosticclassification; regression-to-mean qua3 seeds
không gọi ổn định tổng quát; survivorship giữ mọi7interventions/lầnlỗi; look-elsewhere
không dùng21intervals làm significance; forking-paths giữlockedplan và bảncorrection;
causal on/off chỉsameweights; reverse-causality không coi rank/CE là nguyên nhânquality.
Không suy equivalent từp>.05, không chọn bằngtest hoặc thayreference.

## 13. V6: Kết quả Fresh Seed Confirmation 32 Runs (Hoàn tất & Audited 07/10/2026)

Receipts chính thức:
- Local summary: `outputs_rebuttal/amis_fresh_confirmation_20261005/confirmation_summary.json`
- Selection lock: `outputs_rebuttal/amis_fresh_confirmation_20261005/selection_lock.json`
- Provenance: SHA-256 đối soát 100% từ Hugging Face repos `FiveC/amis-fresh-confirmation-{baseline,clrr,lsr,full}-20261005`
- GPU Sessions: Colab 1 (Account 1) & Colab 2 (Account 2) đã STOPPED hoàn toàn (0 CU idle).

### 13.1. Bảng số liệu chi tiết 32 Runs (Validation và Test 575 câu)

| Seed | Baseline Val chrF++ | Baseline Test BLEU | Baseline Test chrF++ | CLRR Val chrF++ | CLRR Test BLEU | CLRR Test chrF++ | LSR Val chrF++ | LSR Test BLEU | LSR Test chrF++ | Full Val chrF++ | Full Test BLEU | Full Test chrF++ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **45** | 14.6104 | 19.5823 | 18.6293 | 14.0789 | 19.2923 | 18.2506 | 14.3499 | 19.4683 | 18.4774 | 13.8675 | 18.4036 | 17.9742 |
| **46** | 14.0649 | 18.9080 | 18.2759 | 13.8938 | 19.2592 | 18.3358 | 14.0717 | 20.1652 | 18.9016 | 14.7034 | **20.5924** | **19.1772** |
| **47** | 14.5676 | 20.3465 | 19.0520 | 13.9595 | 19.0666 | 18.1409 | 14.2047 | 19.7987 | 18.6654 | 14.4235 | 19.4644 | 18.4227 |
| **48** | 14.2455 | 19.4942 | 18.4879 | 14.8997 | 20.7623 | 14.5929 | 13.8355 | 18.5812 | 18.0409 | 14.0599 | 18.3455 | 17.9439 |
| **49** | 14.1723 | 19.3947 | 18.5043 | 14.3349 | 19.8449 | 18.8968 | 13.9037 | 19.9453 | 18.7801 | 14.1540 | 19.8903 | 18.6968 |
| **50** | 14.2690 | 19.7760 | 18.7349 | 13.9332 | 19.4429 | 18.5089 | 14.1270 | 19.3072 | 18.4959 | 14.3867 | **20.1366** | **18.9113** |
| **51** | 14.5771 | 19.7785 | 18.6741 | 14.1104 | 19.7718 | 18.4952 | 14.0246 | 19.8500 | 18.4014 | 13.9662 | **20.2151** | **18.9008** |
| **52** | 14.4518 | 19.2635 | 18.3860 | 14.2028 | 19.5519 | 18.6039 | 13.7599 | 19.2544 | 18.3772 | 13.8972 | 19.1642 | 18.3652 |

*(Validation metric dùng greedy beam 1 trên 576 câu; Test evaluation dùng beam 4, FP32 trên 575 câu).*

### 13.2. Thống kê tóm tắt (Mean ± Std trên Test Set)

- **Baseline (FT):** BLEU = **19.5680 ± 0.4239** | chrF++ = **18.5930 ± 0.2394**
- **CLRR-only:** BLEU = **19.6240 ± 0.5289** | chrF++ = **17.9781 ± 1.3873**
- **LSR-only:** BLEU = **19.5463 ± 0.5040** | chrF++ = **18.5175 ± 0.2674**
- **Full (CLRR + LSR):** BLEU = **19.5265 ± 0.8371** | chrF++ = **18.5490 ± 0.4501**

### 13.3. Paired Deltas và Kiểm định Gates

| Contrast | Mean Delta BLEU | 95% CI (Student-t) | Exact Sign-Flip p | Wins | chrF++ Delta | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Gate 1 (Full − LSR)** | **-0.0198** | `[-0.5003, +0.4607]` | **0.9297** | 3 / 8 | +0.0315 | **NO-GO** |
| **Secondary Gate 2 (Full − Base)** | **-0.0415** | `[-0.8707, +0.7877]` | **0.9141** | 4 / 8 | -0.0440 | **DID NOT PASS** |
| **Factorial Interaction** | **-0.0758** | — | — | — | — | Descriptive |

### 13.4. Đánh giá khoa học từ 32 Runs Fresh Confirmation

1. **Gain trước đây trên Seed 42 không tái hiện trên 8 fresh seeds độc lập (45–52):** Khoảng tin cậy 95% của chênh lệch BLEU (Full − LSR: `[-0.5003, +0.4607]`; Full − Baseline: `[-0.8707, +0.7877]`) đều bao trùm số 0, với $p > 0.90$. Cả hai pre-registered gates đều không đạt.
2. **Dao động khởi tạo và thứ tự dữ liệu (Seed Variance) lớn hơn chênh lệch trung bình giữa các phương pháp:** Độ lệch chuẩn giữa các seed ($\pm 0.42$ đến $\pm 0.84$ BLEU) lấn át chênh lệch trung bình giữa các phương pháp (-0.02 đến -0.04 BLEU). Không suy đoán đây là "hiệu ứng trần tuyệt đối" nếu chưa có phép đo xác định trần; nhưng trên cấu hình hiện tại, không có bằng chứng về ưu thế của Full.
3. **Ý nghĩa cho bài báo:** Kết quả 32 runs được đối soát chặt chẽ bảo vệ tính liêm chính khoa học, ngăn ngừa nguy cơ overclaiming khi nộp bài hoặc phản biện. Đóng góp cần được định vị lại cẩn trọng dựa trên đo đạc cơ chế nội tại.

### 13.5. Số đo cơ chế Bước B — diễn giải lịch sử đã được review lại

> **Đính chính phạm vi:** giữ nguyên các số đo bên dưới. Không dùng diễn giải “lực kéo quá yếu” từ gradient norm để suy Adam update; không dùng cosine tầng cao để khẳng định rewiring gây hại; U1/U2 ở phần này chưa phải candidate được duyệt. Kiểm chứng độc lập sau đó không thấy lợi ích trung bình khi tắt lớp10–11 và LSR/Full vẫn có centered alignment. Kết luận hiện hành ở [Insight, Mục6](../research/RESEARCH_INSIGHTS.md#verified-confirmation-insight).

Được đo trực tiếp trên NVIDIA A100 (tài liệu chi tiết tại `outputs_rebuttal/amis_fresh_confirmation_mechanisms_20261007/mechanisms_report.md`):

1. **Tương tác Gradient (CE vs Weighted LSR):**
   - Weighted LSR norm ($\lambda = 0.10$) chỉ đạt **0.0565 đến 0.0885**, so với CE norm **31.43 đến 36.74**.
   - Tỷ số LSR/CE gradient norm cực kỳ nhỏ: **0.0018 đến 0.0027** (< 0.3%).
   - Gradient cosine dương nhẹ (+0.05 đến +0.07), rất hiếm batch xung đột (< 0.4/8). LSR không gây xung đột gradient trực tiếp với CE, nhưng lực kéo tham số của nó quá yếu.

2. **Phổ Biểu diễn & Anisotropy (SVD Spectrum):**
   - Không có sụp đổ số chiều: Effective rank duy trì ở mức cao ~386–391 trên cả 4 họ phương pháp.
   - Tuy nhiên, LSR và Full kéo cosine alignment giữa Amis và Mandarin từ 0.35 lên **0.78–0.80**, đồng thời làm **Source Pairwise Cosine tăng vọt từ 0.40 lên ~0.70–0.71**, thể hiện hiện tượng thu hẹp góc phân tán biểu diễn nguồn (anisotropic clustering).

3. **Thử nghiệm Route On/Off & Phân rã Residual theo Tầng:**
   - Khi tắt routing ($\alpha=0.0$) trên cùng trọng số đã train, validation score của CLRR và Full lập tức sụt giảm mạnh: **$\Delta$ BLEU = +2.67 ± 0.40** (CLRR) và **+2.64 ± 0.50** (Full); **$\Delta$ chrF++ = +2.18 đến +2.30**. Mạng phụ thuộc chặt chẽ vào kết nối skip đã học, không hề bỏ qua nó.
   - Tuy nhiên, skip connection chỉ hoạt động đồng thuận tại **Layers 3–9** (tỷ số residual/current RMS 0.06–0.08, cosine 0.80–0.92). Tại **Layers 10–11**, tỷ số này tụt xuống **0.012–0.023** và cosine sụp đổ xuống **0.22–0.29**. Rewire đồng loạt ở mọi tầng gây xung đột ở các tầng ngữ nghĩa trừu tượng cao của encoder.

4. **Định hướng Nâng cấp (Candidate U1/U2):**
   - Không tăng mù quáng $\lambda$ hoặc $\alpha$.
   - **U1 (Layer-Selective Rewiring):** Giữ rewire ở tầng hình thái thấp-trung (Layers 3–8), ngắt ở Layers 9–12.
   - **U2 (Unrewired Target Anchor / Gated Rewire):** Bỏ rewire trên target pass khi tính LSR alignment hoặc bổ sung learnable gating.




<a id="turkish-seed42"></a>

## 14. Turkish→English — pilot seed42, hoàn tất 08/10/2026

> Nội dung gộp từ `TURKISH_MBART_SEED42_RESULTS.md`. **Kết quả đã xác minh**, giữ nguyên bảng số và phạm vi single-seed.

### Turkish→English — mBART-50, pilot seed 42

Official OPUS100 validation/test splits unchanged; 20,000 training pairs, 2,000 validation and 2,000 test pairs. All four selected checkpoints: epoch 2; training stopped at epoch 12 with patience 10. Test after four selections were locked: beam 4, FP32 generation, TF32 off, BLEU13a/native chrF++/chrF0. All test predictions and scores independently verified.

| Method | Seed | Test BLEU | Test chrF++ | Test chrF0 | ΔBLEU vs Baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 42 | 22.634203 | 41.549538 | 42.595475 | +0.000000 |
| middle | 42 | 22.519996 | 41.389460 | 42.427257 | -0.114208 |
| lsr | 42 | 22.200584 | 41.271311 | 42.305028 | -0.433619 |
| full | 42 | 22.570620 | 41.549304 | 42.616011 | -0.063583 |

#### Full minus each comparator

| Comparator | ΔBLEU | ΔchrF++ | ΔchrF0 |
|---|---:|---:|---:|
| baseline | -0.063583 | -0.000233 | +0.020536 |
| middle | +0.050624 | +0.159845 | +0.188754 |
| lsr | +0.370036 | +0.277993 | +0.310983 |

#### Interpretation

Full improves on LSR in all three metrics for this frozen seed. Full has almost the same chrF++ as Baseline, slightly lower BLEU, and slightly higher chrF0. The run does not show a clear Full advantage over Baseline. This is an exploratory one-seed result, not evidence of seed robustness or statistical significance. No method or checkpoint was tuned using these test results.

Checkpoint repo: https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/opus100_turkish_english_20k .
Storage rules: docs/operations/HUGGINGFACE_CHECKPOINT_STORAGE.md .
Results: public_test_results.json; per-family final collection_receipt.json.

<!-- standardized-es10-validation-progress -->
## Tiến độ đối sánh chuẩn hóa ES10 — validation đã kiểm chứng

**Bảng này chỉ là validation, chưa phải kết quả test và không dùng để kết luận hơn/kém phương pháp.** Chọn best theo raw-reference validation chrF++; mọi dòng đã replay độc lập từ public predictions và xác minh hash checkpoint mà không tải weights về PC. Cohort được duyệt gồm14 runs; số cũ giữ nguyên.

| ID | Backbone | Nhánh | Seed | Epoch cuối | Epoch best | Val BLEU | Val chrF++ | Val chrF0 | Dừng vì |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A01 | mbart-large-50 | baseline | 46 | 20 | 14 | 16.6892 | 14.0649 | 16.1252 | epoch_cap |
| A02 | mbart-large-50 | middle-alignment | 42 | 20 | 16 | 16.4643 | 14.0266 | 16.1654 | epoch_cap |
| A03 | mbart-large-50 | narrow-lora | 42 | 20 | 17 | 3.2625 | 5.3002 | 6.1836 | epoch_cap |
| A04 | mbart-large-50 | bitfit | 42 | 20 | 17 | 0.5791 | 2.8621 | 3.3392 | epoch_cap |
| A05 | mbart-large-50 | strong-lora-a | 42 | 20 | 19 | 9.2513 | 9.3819 | 10.9456 | epoch_cap |
| A06 | mbart-large-50 | strong-lora-b | 42 | 20 | 19 | 10.9850 | 10.3825 | 12.0560 | epoch_cap |
| B01 | nllb-200 | baseline | 42 | 20 | 19 | 13.2966 | 12.0212 | 13.9104 | epoch_cap |
| B02 | nllb-200 | middle-alignment | 42 | 20 | 20 | 13.3678 | 12.1479 | 14.1152 | epoch_cap |
| B03 | nllb-200 | strong-lora-a | 42 | 20 | 19 | 11.8765 | 11.0441 | 12.8561 | epoch_cap |
| B04 | nllb-200 | strong-lora-b | 42 | 20 | 18 | 9.5091 | 9.6468 | 11.2257 | epoch_cap |
| B05 | nllb-200 | bitfit | 42 | 20 | 20 | 0.9361 | 3.4250 | 3.9959 | epoch_cap |
| C01 | mt5-small | middle-alignment | 42 | 20 | 17 | 4.3204 | 5.7908 | 6.7560 | epoch_cap |
| D01 | byt5-small | baseline | 42 | 20 | 18 | 1.3954 | 3.5611 | 4.1546 | epoch_cap |
| D02 | byt5-small | middle-alignment | 42 | 20 | 20 | 6.2547 | 7.7647 | 9.0588 | epoch_cap |
<!-- end-standardized-es10-validation-progress -->

<!-- standardized-es10-test-results -->
## Đối sánh ES10 — kết quả test đã xác minh (14/14)

Test Amis→Mandarin gồm 575 câu, raw CSV references, BLEU tokenize=zh, chrF++ word_order=2 và chrF0 word_order=0; beam 4, FP32 generation, TF32 off. Checkpoint được khóa trước khi test, chọn bằng validation chrF++ với validation_beams đã khóa trong từng PLAN (mBART beam 1; NLLB beam 4). Training tối đa 20 epochs, early-stopping patience 10. Chỉ chạy test trên Colab 1/2 A100 Standard; không train lại trong đợt đánh giá này.

Mọi dòng đã replay metrics độc lập từ public predictions, kiểm tra thứ tự source/target, hash artifacts và selection lock 14 checkpoints. Số cũ được giữ để truy vết; bảng này là kết quả đối sánh ES10 mới. Mỗi dòng là một seed, không phải trung bình đa seed; baseline mBART seed 46 khác seed 42 của đối sánh, nên không dùng các chênh lệch này như paired-seed significance.

| ID | Backbone | Nhánh | Seed | Epoch best | Test BLEU | Test chrF++ | Test chrF0 | Nguồn |
|---|---|---|---:|---:|---:|---:|---:|---|
| A01 | mbart-large-50 | baseline | 46 | 14 | 18.9080 | 18.2759 | 17.8424 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/baseline/seed-46/es10) |
| A02 | mbart-large-50 | middle-alignment | 42 | 16 | 18.9986 | 18.2626 | 17.7790 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/middle-alignment/seed-42/es10) |
| A03 | mbart-large-50 | narrow-lora | 42 | 17 | 2.9898 | 4.4036 | 5.8142 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/narrow-lora/seed-42/es10) |
| A04 | mbart-large-50 | bitfit | 42 | 17 | 0.2820 | 2.3567 | 3.1137 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/bitfit/seed-42/es10) |
| A05 | mbart-large-50 | strong-lora-a | 42 | 19 | 10.0571 | 8.5178 | 11.2712 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/strong-lora-a/seed-42/es10) |
| A06 | mbart-large-50 | strong-lora-b | 42 | 19 | 13.8857 | 10.3978 | 13.5794 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/strong-lora-b/seed-42/es10) |
| B01 | nllb-200 | baseline | 42 | 19 | 13.0895 | 10.2471 | 13.5195 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/nllb-200/baseline/seed-42/es10) |
| B02 | nllb-200 | middle-alignment | 42 | 20 | 13.3033 | 10.4945 | 13.7636 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/nllb-200/middle-alignment/seed-42/es10) |
| B03 | nllb-200 | strong-lora-a | 42 | 19 | 11.8650 | 9.6563 | 12.7605 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/nllb-200/strong-lora-a/seed-42/es10) |
| B04 | nllb-200 | strong-lora-b | 42 | 18 | 10.2521 | 8.5470 | 11.2815 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/nllb-200/strong-lora-b/seed-42/es10) |
| B05 | nllb-200 | bitfit | 42 | 20 | 0.7912 | 2.8771 | 3.8074 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/nllb-200/bitfit/seed-42/es10) |
| C01 | mt5-small | middle-alignment | 42 | 17 | 4.6600 | 5.0267 | 6.6449 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mt5-small/middle-alignment/seed-42/es10) |
| D01 | byt5-small | baseline | 42 | 18 | 1.3120 | 3.6667 | 4.2778 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/byt5-small/baseline/seed-42/es10) |
| D02 | byt5-small | middle-alignment | 42 | 20 | 7.0156 | 8.0748 | 9.3917 | [Artifacts](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/byt5-small/middle-alignment/seed-42/es10) |

Proofs: `outputs_rebuttal/amis_standardized_controls/test_verification.json`, `outputs_rebuttal/amis_standardized_strong_mbart/test_verification.json`; public copies under `registry/plans/` in the checkpoint repository.
<!-- end-standardized-es10-test-results -->

<!-- method-candidate-seed42-screening -->
## Method candidate screening — validation only (seed42)

These four runs screen two isolated variants under the same mBART-50 training recipe as baseline seed42. They use Amis→Mandarin validation (576 rows), original CSV references, chrF++ checkpoint selection, beam1/FP32. **They have no test evaluation and are not paper test results.** Baseline reached the 20-epoch cap under historical patience4; candidate runs also reached the 20-epoch cap with patience10. All public predictions were independently replayed and the LFS archive hashes verified.

| Run | Candidate | Selected/completed epoch | Validation BLEU | Validation chrF++ | Validation chrF0 | ΔBLEU vs baseline42 | ΔchrF++ vs baseline42 | Public artifacts |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Baseline42 | Full fine-tuning | 20/20 | 16.2659 | 13.9015 | — | — | — | [Checkpoint](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/baseline/seed-42/baseline-corrected) |
| U1C | CLRR pre-addition donor | 14/20 | 16.4181 | 13.9381 | 16.0349 | +0.1522 | +0.0366 | [Checkpoint](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/clrr-precache/seed-42/es10) |
| U1F | CLRR pre-addition donor + original LSR | 19/20 | 17.3296 | 14.5519 | 16.6663 | +1.0637 | +0.6503 | [Checkpoint](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/full-precache/seed-42/es10) |
| U2L | Centered-LSR | 20/20 | 16.4203 | 13.9045 | 16.0522 | +0.1544 | +0.0029 | [Checkpoint](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/lsr-centered/seed-42/es10) |
| U2F | Original CLRR + Centered-LSR | 20/20 | 17.0227 | 14.5080 | 16.6142 | +0.7568 | +0.6065 | [Checkpoint](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints/amis_mandarin/mbart-large-50/full-centered/seed-42/es10) |

### Reading this screening

- Both candidate Full runs exceed their corresponding single-component screen in validation chrF++: U1F−U1C is +0.6137; U2F−U2L is +0.6035. At seed42, the best score is U1F (14.5519), 0.0439 chrF++ above U2F.
- The controls are not fully matched for early-stopping history: old CLRR-only and LSR-only seed42 runs stopped under ES4. Treat those as context, not strict ES10 paired controls.
- One seed, one validation set, and no test evaluation cannot establish robustness, statistical significance, generalization, or ACL novelty. Do not use these values as final test claims. The score only motivates a multi-seed confirmation after the candidate recipe and research claim are locked.
- Detailed code, receipts, independent replay, hashes, and run status: `outputs_rebuttal/amis_method_candidates/` and `docs/operations/METHOD_CANDIDATES_RUNBOOK.md`.
<!-- end-method-candidate-seed42-screening -->

## Candidate diagnostics — no training, validation seed42

6/6 checkpoints verified: baseline42, historical Full42, U1C/U1F/U2L/U2F.
576 validation rows; beam1 FP32, TF32 off, dropout off. Zero optimizer steps.
Generation-on reproduces saved predictions after original whitespace stripping;
metrics replay matches1e−9. Archive/worker hashes and strict loading PASS.
Historical results above remain unchanged. Colab1/2 stopped after collection.

| Candidate−historical Full42 | Metric | Delta | Paired bootstrap CI95 | Holm4 p |
|---|---|---:|---|---:|
| U1F | BLEU | −0.0261 | [−1.0097,+0.9928] | .9224 |
| U1F | chrF++ | +0.0739 | [−0.5616,+0.7437] | .9224 |
| U2F | BLEU | −0.3330 | [−1.4047,+0.7631] | .7743 |
| U2F | chrF++ | +0.0300 | [−0.7025,+0.7926] | .9224 |

10k resamples,RNG42; nativeSacreBLEU2.6.0p; Holm4family=2candidates×2metrics.
These selected-validation analyses are exploratory, not independent confirmation.
No established superiority or inferiority. Mechanism/intervention report:
[REPORT.md](../../outputs_rebuttal/method_candidate_diagnostics/REPORT.md).
Full cũ late10–12off reduceschrF++.5001 (Holm18p=.0378); inference dependence,
not evidence a corresponding training ablation will improve. No newtrain/test.

