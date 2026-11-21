# Residual-effect candidate — code và kiểm thử trước pilot

Trạng thái hiện hành ngày 12/10/2026: **P1–P5 hoàn tất, upload và checksum HF đã xác minh; queue đã kết thúc.** Kết quả và GO/NO-GO tại [RESIDUAL_EFFECT_VALIDATION_REPORT.md](RESIDUAL_EFFECT_VALIDATION_REPORT.md). Các mục bên dưới ghi lại quá trình triển khai và theo dõi theo thời gian.

Tài liệu này mô tả hướng residual-effect đã bàn trong chat. Selective KL trong
`METHOD_UPGRADE_DESIGN.md` vẫn là hướng đã loại khỏi đóng góp chính; bảng KL
trong file đó không phải 5 run hiện hành.

## 1. Thay đổi phương pháp

Giữ original CLRR distance2, strength0.10, detached post-rewiring donor.
Hai encoder views cùng weights, dropout OFF:

`delta_x = pool(E_on(x)) − pool(E_off(x))`

`delta_y = stopgrad(pool(E_on(y)) − pool(E_off(y)))`

Auxiliary là one-way InfoNCE giữa delta_x và delta_y của đúng cặp dịch,
temperature0.10; beta0.10. Gradient đi qua cả source ON và source OFF.
Pooling loại special tokens/PAD; không thêm parameters hoặc queue.
Nhánh Full vẫn giữ original sentence LSR lambda0.10. Nhánh không LSR giúp
kiểm tra LSR còn cần hay không; không kết luận LSR là nguyên nhân số thấp.

Source: `src/methods/residual_effect/v1/lsr.py`; định nghĩa và giới hạn tại
`src/methods/residual_effect/v1/README.md`. Registry/engine chung đã nhận IDs mới.
Không sửa công thức hoặc checkpoint của các phương pháp cũ.

## 2. Đúng 5 run dự kiến, sau khi tác giả đồng ý

| ID | Nhánh | Câu hỏi |
|---|---|---|
| P1 | Original CLRR-only | Đối chứng routing train RTX, hiện chưa có cùng recipe |
| P2 | Original LSR + ordinary InfoNCE | Generic contrastive không CLRR có giải thích gain không? |
| P3 | Original Full + ordinary InfoNCE | Subtraction ON−OFF có hơn alignment thông thường không? |
| P4 | CLRR + effect InfoNCE, bỏ LSR | Đóng góp của LSR khi đã có objective mới |
| P5 | Original Full + effect InfoNCE | Candidate kết hợp đầy đủ |

Reuse RTX R1 baseline và R2 original Full cùng recipe. Không ghép số A100 làm
matched RTX controls. Config: `configs/experiments/amis_residual_effect_rtx.json`.
FP32, seed42, LR5e-5, microbatch4×GA32, cap20, patience10; selection giữ nguyên
raw-reference validation chrF++, beam1 FP32 để khớp controls. Không đổi sang
BLEU selection trong đợt này. BLEU zh và chrF0 dùng đánh giá screening;
native chrF++ phải diễn giải cùng cảnh báo sparse word bigrams đã audit.

Temperature/beta0.10 là một lựa chọn pilot cần tác giả duyệt, không phải mức
tối ưu suy ra từ dữ liệu. Các nhánh contrastive đều dùng cùng microbatch4 và
3 negatives/câu; accumulation32 không tăng negative pool. Không sweep.
Ngưỡng screening dự kiến: P5 hơn baseline và old Full ít nhất0.30 BLEU,
hơn cả ordinary controls ít nhất0.20 BLEU, chrF0 không giảm. Đây là tiêu chí
development một seed, không phải statistical significance hay bảo đảm ACL.
Không dùng test để tuning; cần fresh-seed confirmation nếu pilot có triển vọng.

HF paths sau khi được phép train:
`checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/rtx6000-fp32-b4-ga32-es10-tau0.1-beta0.1/`.
P1 không có suffix tau/beta. Chưa upload hoặc ghi đè bất kỳ checkpoint nào.

## 3. Kiểm thử đã thực hiện

Python trong conda `clrr`, Torch2.6.0+cu124, CPU tiny mBART có dropout.

| Kiểm tra | Kết quả |
|---|---|
| InfoNCE khớp công thức; đúng cặp tốt hơn shuffled trên tensor chuẩn | PASS |
| Target detached; source gradient hữu hạn, auxiliary không đi vào decoder | PASS |
| Khôi phục encoder training flags/strength kể cả exception | PASS |
| Auxiliary không tiêu thụ RNG dropout của CE | PASS |
| Reentrant checkpointing: loss và parameter gradients khớp không checkpointing | PASS, cả 4 IDs mới |
| Zero-effect loss hữu hạn = log(batch size) | PASS; không bảo đảm tránh collapse |
| Special/PAD mask; inference không cần target auxiliary | PASS |
| Save/reload và reconstruction từ spec: logits khớp | PASS |
| Shared runner cả 5 nhánh, Trainer accumulation32 scale đúng một lần | PASS |

Receipts nhỏ local:
`outputs/experiments/amis_residual_effect_preflight/CPU_GATE.json`,
`outputs/experiments/amis_residual_effect_rtx/CPU_GATE.json`.
Gate có cảnh báo PyTorch cho target no_grad dưới reentrant checkpointing;
target được chủ đích detach, gradient source được kiểm tra riêng.

Lệnh tái lập CPU, không mở GPU/training dài:

```powershell
conda run -n clrr python tests/test_residual_effect.py
conda run -n clrr python scripts/experiment.py cpu --config configs/experiments/amis_residual_effect_rtx.json
```

## 4. Còn thiếu trước training

RTX khi kiểm tra: Quadro RTX6000, free20080MiB, used4128MiB, utilization94%.
Máy có người khác đang chạy. Không chạy phép GPU nào, không dừng process nào.
Full-size forward/backward, peak VRAM, tốc độ và tiny optimizer/save-reload
gate trên đúng GPU/bundle chưa có bằng chứng PASS. Auxiliary source eval views
không checkpoint nên giữ thêm activation graphs: có rủi ro OOM thực tế.
Nếu không đủ VRAM thì giữ HOLD, không tự đổi batch/precision hoặc tắt job khác.

Novelty vẫn HOLD: SkipCLM đã dùng skip + cross-language contrastive, còn
cross-corruption distillation đã align response changes. Điểm đề xuất là
alignment song ngữ của hiệu ứng can thiệp kiến trúc ON−OFF trong shared encoder.
Khác thiết kế chưa đủ bảo đảm novelty mạnh. Cần đối chiếu chi tiết trước khi
chọn làm đóng góp chính; CPU PASS không giải quyết novelty hoặc hiệu quả dịch.

## 5. Quyền chạy và kiểm tra lại GPU

Tác giả đã duyệt chạy GPU gate rồi 5 run nếu gate PASS. Kiểm tra lại ngày
11/10/2026: free18994MiB, used5214MiB, utilization94%. Recipe yêu cầu ít nhất
20000MiB trống; kiểm tra availability chưa đạt. Chưa launch gate/training,
không thay ngưỡng, batch hoặc precision, không tác động job khác.
Receipt: `outputs/experiments/amis_residual_effect_preflight/GPU_AVAILABILITY.json`.
Đây là WAITING_FOR_FREE_VRAM, không phải OOM hay thất bại correctness.
Khi GPU đủ bộ nhớ, quyền chạy gate rồi training đã có; không cần xin lại.

## 6. Queue RTX đã mở sau khi đủ bộ nhớ

Sau đó GPU trống 23542MiB, queue được mở trong project CLRR riêng trên RTX.
GPU gate P1 và P2 đã PASS; P2 peak allocated 12.0804GiB, checkpoint save/reload,
optimizer và RNG receipt đều PASS. Trước P3, GPU của người khác chiếm thêm VRAM
(đo được 19372 rồi 19294MiB trống), khiến lần launch đầu dừng đúng ngưỡng.
Queue tiếp tục bằng `outputs/experiments/amis_residual_effect_rtx/resume.sh`
trên RTX, giữ lock CLRR và đợi VRAM >=20000MiB trước mỗi gate/run. Nó bỏ qua
P1/P2 đã PASS, chạy P3–P5 gate, sau đó train P1–P5. Chưa khẳng định P3–P5
hoặc training đã hoàn tất. Theo dõi `resume.log` và các receipt PASS/VAL_COMPLETE.

## 7. Cập nhật mới nhất: 5/5 GPU gates PASS, training đã bắt đầu

Tác giả yêu cầu tiếp tục các gate khi GPU có 19294MiB trống. Peak P2/P3 là
12.0804GiB, P4/P5 là 12.1697GiB; vì vậy availability guard vận hành được
điều chỉnh từ 20000MiB xuống 16000MiB cho queue này. Đây chỉ là điều kiện
khởi chạy dựa trên VRAM trống; scientific recipe, source bundle, optimizer,
precision, batch và checkpoint selection giữ nguyên. Wrapper vận hành trên RTX
ghi tại `outputs/experiments/amis_residual_effect_rtx/guarded_runner.py`;
log `guarded.log`, process receipt `guarded.pid`.

P1–P5 đều có GPU `PASS.json`: 2 optimizer steps, finite loss/gradient,
save/reload logits, optimizer/RNG receipts PASS. P4/P5 là hai gate mới với
effect loss. Sau P5, queue đã bắt đầu training P1 (quan sát tiến độ 7/720
steps); P2–P5 sẽ train nối tiếp nếu các run trước hoàn tất. Chưa có validation
scores hoặc checkpoint cuối để kết luận phương pháp. Upload/verify public HF
được engine thực hiện sau từng model; chỉ công nhận hoàn tất khi có
`VAL_COMPLETE.json` và receipt backup xác minh.

### Theo dõi tiếp theo

Queue process `guarded.sh` còn sống (elapsed24:33). P1 đang TRAINING epoch4.72/20,
step170/720, patience counter0; validation tốt nhất hiện BLEU11.1676,
chrF++10.8558. So với RTX R1 baseline best BLEU17.6823, chrF++14.7030,
checkpoint P1 hiện thấp rõ rệt; đây là tín hiệu giữa training, chưa phải kết
quả cuối và P1 chưa dừng. GPU tổng trống5780MiB, utilization100% (có job khác
dùng GPU); queue tự tiếp tục. Chưa có P1 VAL_COMPLETE hoặc upload cuối được xác minh.

### Trạng thái theo dõi mới nhất

Queue `guarded.sh` còn sống. P1 đang TRAINING epoch7.50/20, step270/720,
patience counter0; validation tốt nhất BLEU14.2408, chrF++12.2311 tại epoch7.
So với RTX R1 baseline best BLEU17.6823, chrF++14.7030, P1 vẫn thấp ở thời
điểm này nhưng đã cải thiện so với epoch4. Đây chưa phải kết quả cuối. GPU tổng
trống5780MiB, utilization98%; queue tiếp tục. Chưa có P1 VAL_COMPLETE hoặc
upload cuối được xác minh.

### Kiểm tra queue tiếp theo

Đọc log guarded lúc process còn sống (elapsed35:47): P1 đã tới step279/720,
epoch xấp xỉ7.75. Progress receipt vẫn ghi step270 do receipt cập nhật theo
validation/step log; tail train log xác nhận step279. Chưa có validation sau
epoch7: best BLEU14.2408, chrF++12.2311. RTX còn5780MiB, utilization99%.
Queue chưa tạo VAL_COMPLETE; các checkpoint trung gian đang được trainer giữ
trong workspace, chưa tuyên bố HF upload cuối.

### Mốc theo dõi 10 phút tiếp theo

P1 vẫn TRAINING. Progress receipt epoch10.0, step360/720; log sau validation
đã đi tới khoảng step367. Validation tại epoch10: BLEU14.3297, chrF++12.5113.
Best selection metric là chrF++12.7957 (best checkpoint sau epoch7); patience
counter2. So với R1 baseline BLEU17.6823/chrF++14.7030, hiện P1 vẫn thấp lần
lượt khoảng3.35 BLEU và2.19 chrF++, nhưng đã tăng từ validation epoch7.
RTX còn5778MiB, utilization99%; queue process sống. Chưa có VAL_COMPLETE hoặc
HF upload cuối được xác minh; không kết luận sớm khi train mới nửa chặng.

### Mốc theo dõi kế tiếp

P1 còn TRAINING: epoch13/20, step468/720, patience counter1. Validation epoch13:
BLEU15.6019, chrF++13.4235. Best selection chrF++13.4603; best BLEU chưa được
đọc từ best checkpoint, nên không gán BLEU cho cùng epoch. So baseline RTX R1
BLEU17.6823/chrF++14.7030, validation hiện vẫn thấp nhưng tiếp tục cải thiện.
GPU còn5778MiB, utilization100%; guarded queue process sống, chỉ P1 hiện có thư
mục run. Chưa có VAL_COMPLETE hay HF final upload receipt.

### Mốc theo dõi tiếp theo

P1 còn TRAINING: epoch16/20, step576/720. Validation epoch16 BLEU16.1156,
chrF++13.6881, patience counter0. So với RTX R1 baseline BLEU17.6823 /
chrF++14.7030, chênh tạm thời −1.5667 BLEU / −1.0149 chrF++. Validation đang
tăng. Progress receipt vẫn ghi `best_metric=13.4603` trong khi epoch16 đạt13.6881;
log cho thấy patience0. Cần đối chiếu checkpoint/VAL_COMPLETE sau khi train kết thúc
để xác nhận best-step metadata. GPU trống5778MiB, utilization100%, queue process
còn sống; chưa có upload cuối được xác minh.

### P1 gần hoàn thành

P1 đang step680/720 (epoch18.89/20), training process còn sống. Validation
mới nhất epoch18: BLEU16.6216, chrF++14.1252, cũng là best chrF++ đã ghi.
So RTX R1 baseline BLEU17.6823/chrF++14.7030, tạm thời thấp hơn1.0607 /
0.5778 điểm. Gap đang thu hẹp. Còn40 optimizer steps và validation cuối;
chưa có VAL_COMPLETE hoặc HF final upload receipt.

### P1 hoàn tất; P2 đã bắt đầu

P1 hoàn tất 20 epoch/720 optimizer steps. Checkpoint được chọn theo validation
chrF++ (beam 1, 576 câu, references CSV gốc), đạt BLEU **16.6216**, chrF++
**14.1252**, chrF0 **16.2232**. Đây là số validation, chưa phải test. So với
RTX R1 baseline (BLEU 17.6823, chrF++ 14.7030), P1 thấp hơn lần lượt 1.061 và
0.578 điểm trên validation; vì vậy kết quả hiện tại chưa cho thấy CLRR-only vượt
baseline ở seed này.

Upload đã hoàn tất và được xác minh (`VAL_COMPLETE_VERIFIED`, receipt có
`verified: true`) trong repo `FiveC/amis-rewire-checkpoints`, tại
`checkpoints/amis_mandarin/mbart-large-50/clrr/seed-42/rtx6000-fp32-b4-ga32-es10/`.
Revision: `d2ad9ef81a450a0010c003be6ec655e013730a7a`. Queue vẫn hoạt động và đã
chuyển tự động sang P2; log xác nhận training bắt đầu ở step 1/720. GPU lúc poll:
5750 MiB trống, utilization 99%. Không có test evaluation trong bước này.

### Theo dõi P2 — mốc 10 phút tiếp theo

P2 (`ordinary-lsr-v1`) đang TRAINING. Receipt tại step80/720, epoch2.22/20,
patience0; log sau đó xác nhận tiến tới step88. Validation epoch2 (576 câu,
beam1) BLEU7.0428, chrF++8.0284; đây là mốc sớm, không dùng kết luận hiệu quả.
Queue `guarded.sh` còn sống. GPU còn5730MiB, utilization100%; không can thiệp các
job dùng chung. P1 đã có upload Hugging Face xác minh; P2 chưa có checkpoint cuối,
VAL_COMPLETE hoặc upload receipt.

### Theo dõi P2 — mốc tiếp theo

P2 tiếp tục TRAINING, receipt epoch4.45/20, step160/720, patience0; log tiến tới
step169. Validation chrF++ tốt nhất hiện 10.2446 (receipt không ghép chỉ số BLEU
với checkpoint này). Queue process còn sống; chưa có VAL_COMPLETE hay upload P2.
GPU còn5728MiB, utilization100%. Không có dấu hiệu lỗi trong log tại lần poll này.

### Theo dõi P2 — validation đang chạy

Receipt P2 hiện ở epoch6.95/20, step250/720, patience0; validation chrF++ tốt
nhất 11.8765. Log tail đang ở giữa lượt validation; chưa có artifact cuối,
VAL_COMPLETE hoặc upload P2. Queue còn sống; GPU còn5728MiB, utilization99%.

### Theo dõi P2 — epoch 9

P2 đang TRAINING ở epoch9/20, step324/720. Validation epoch9: BLEU15.0043,
chrF++13.0904; best selection chrF++ hiện13.1870, patience1. So với RTX R1
baseline (17.6823 / 14.7030), epoch9 còn thấp; đây vẫn là kết quả giữa train.
Log ước tính khoảng36.5 phút optimizer steps còn lại, chưa gồm final prediction,
checkpoint packaging và upload. Queue còn sống, GPU còn5728MiB và utilization86%.
Chưa có VAL_COMPLETE/upload receipt cho P2.

### Theo dõi P2 — epoch 11

P2 đang ở epoch11.11/20, step400/720, patience0. Validation epoch11 đạt
BLEU15.9796 và chrF++13.7193, hiện là best selection. So RTX R1 baseline
(17.6823 / 14.7030), gap hiện khoảng−1.70 BLEU / −0.98 chrF++; đây chưa phải
kết quả cuối. Còn khoảng29–30 phút optimizer steps theo tốc độ log, chưa gồm
final evaluation/upload. Queue còn sống; GPU trống5728MiB, utilization100%.

### Theo dõi P2 — tiếp tục

Poll sau câu hỏi giữa turn: P2 vẫn TRAINING, epoch11.95/20, step430/720,
patience0; best validation chrF++13.7193 tại epoch11. Log hiện đang đi qua một
lượt validation tiếp theo (100/288 batch). Chưa có final artifacts/upload.
Queue còn sống; GPU trống5728MiB, utilization100%.

### Theo dõi P2 — epoch 14

P2 ở epoch14/20, step504/720, patience0. Validation epoch14: BLEU16.6633,
chrF++14.0295; progress receipt ghi best_metric14.0078 (metric ghi nhận có sai
khác nhỏ so với eval log, cần đối chiếu checkpoint/VAL_COMPLETE trước khi chốt).
So baseline RTX R1 (17.6823 / 14.7030), mốc epoch14 thấp khoảng1.02 BLEU và
0.67 chrF++. Còn216 steps (~20 phút theo tốc độ gần đây), sau đó final prediction
và upload. Queue còn sống; GPU trống5728MiB, utilization85%.

### Theo dõi P2 — epoch 16

P2 đang epoch16.11/20, step580/720, patience0. Validation epoch16 đạt BLEU
16.7920, chrF++14.2138 và hiện là best. So với RTX R1 baseline (17.6823 /
14.7030), còn thấp khoảng0.89 BLEU / 0.49 chrF++; gap tiếp tục thu hẹp.
Còn140 steps (~13 phút theo tốc độ gần đây), chưa gồm final prediction/upload.
Queue còn sống; GPU trống5728MiB, utilization99%.

### Theo dõi P2 — epoch 18

P2 ở epoch18.33/20, step660/720, patience2. Validation epoch18: BLEU16.8317,
chrF++14.1759; thấp hơn best chrF++14.2138 tại epoch16. Còn60 steps (~6 phút),
sau đó full validation prediction, đóng gói và upload. Queue process còn sống,
GPU trống5728MiB, utilization100%; chưa có completion marker cho P2.

### P2 train xong; upload đang tiếp diễn

P2 hoàn tất 20 epoch/720 steps. Checkpoint được chọn theo validation chrF++ tại
step684 (epoch19), best chrF++14.4825; validation prediction cuối ở beam1 cho
BLEU17.2700, chrF++14.4825, chrF0 16.5855 (576 câu, original CSV refs,
SacreBLEU2.6.0). So RTX R1 baseline BLEU17.6823/chrF++14.7030, P2 thấp hơn
0.412 BLEU/0.221 chrF++; gần baseline nhưng chưa vượt. `best.zip` khoảng2.47GB
đang upload, log ở ~80%; chưa đánh dấu P2 hoàn tất cho tới khi có receipt xác minh
Hugging Face. GPU lúc poll còn23272MiB, utilization0%.

### P2 upload xác minh; P3 đã bắt đầu

P2 có `VAL_COMPLETE.json` trạng thái `VALIDATION_COMPLETE_VERIFIED`; receipt
`verified:true`, SHA-256 đối chiếu khớp với repo `FiveC/amis-rewire-checkpoints`
(revision `40cd0ffc25a4df820f3b9cd2ddd88365a530be6f`). Checkpoint path:
`checkpoints/amis_mandarin/mbart-large-50/ordinary-lsr-v1/seed-42/rtx6000-fp32-b4-ga32-es10-tau0.1-beta0.1/`.

Queue đã tự khởi chạy P3 (`ordinary-full-v1`). P3 hiện epoch2/20, step72/720;
validation epoch2 BLEU6.2741, chrF++7.7871, còn quá sớm để đánh giá. `progress.json`
`best_metric` đôi lúc trễ so với log epoch validation vì Trainer cập nhật best
checkpoint sau callback; kết luận cuối sẽ căn cứ selected trainer state và
`VAL_COMPLETE`, không lấy trường progress tức thời làm số cuối. GPU còn9970MiB,
utilization3% tại poll; queue còn sống.

### Theo dõi P3 — epoch 5

P3 (`ordinary-full-v1`) đang epoch5/20, receipt step180/720; log đã tới step185.
Validation epoch5 BLEU13.1778, chrF++11.9949, tăng so với epoch2 nhưng vẫn là
mốc sớm. Không kết luận phương pháp từ đây. Queue còn sống; GPU trống9974MiB,
utilization94%. Chưa có checkpoint cuối hoặc upload P3.

### Theo dõi P3 — epoch 8

P3 ở epoch8.06/20, step290/720, patience0. Validation epoch8 BLEU15.7848,
chrF++13.5003, tiếp tục tăng so epoch5. So RTX R1 baseline (17.6823/14.7030),
mốc này thấp hơn khoảng1.90 BLEU/1.20 chrF++, nhưng còn giữa training. Còn430
steps (~29 phút theo tốc độ hiện tại), chưa gồm final evaluation/upload. Queue
còn sống; GPU trống9974MiB, utilization92%.

### Theo dõi P3 — khoảng giữa run

P3 đang epoch10.83/20, step390/720, patience0; progress ghi best validation
chrF++14.0380. Còn330 steps; log vẫn ghi train, chưa có final metrics/checkpoint
upload. Queue còn sống; GPU trống7630MiB, utilization100%.

### Theo dõi P3 — epoch 12.8

P3 đang epoch12.78/20, step460/720, patience0; best validation chrF++14.2612.
Log tail đang giữa validation (190/288 batch). Còn260 steps (~24–25 phút theo tốc
độ gần đây), sau đó mới final evaluation/upload. Queue còn sống; GPU trống7798MiB,
utilization100%.

### Theo dõi P3 — epoch 15.8

P3 ở epoch15.83/20, step570/720, patience2; best validation chrF++14.4279.
Log cho thấy validation kế tiếp vừa bắt đầu tại step576. Còn150 optimizer steps
(~10–11 phút train theo tốc độ hiện tại), chưa gồm final prediction/upload.
Queue còn sống; GPU trống9970MiB, utilization63%.

### Theo dõi P3 — epoch 18.6

P3 ở epoch18.61/20, step670/720, patience2; best validation chrF++15.1246.
Mốc này cao hơn RTX R1 baseline chrF++14.7030 khoảng+0.42, tín hiệu tạm thời
đáng chú ý nhưng chưa phải số được khóa trong VAL_COMPLETE; BLEU cùng checkpoint
chưa được xác minh ở snapshot này. Còn50 steps, sau đó final validation và upload.
Queue còn sống; GPU trống9272MiB, utilization98%.

### P3 train/validation xong; đang upload

P3 train đủ20 epoch/720 steps. Checkpoint chọn tại step576 (epoch16), validation
beam1 trên 576 câu đạt BLEU18.4584, chrF++15.1246, chrF0 17.3609. So RTX R1
baseline (BLEU17.6823, chrF++14.7030), P3 validation cao hơn+0.7761 BLEU và
+0.4216 chrF++. So P2 (`ordinary-lsr-v1`: BLEU17.2700/chrF++14.4825), P3 cao
hơn+1.1884 BLEU và+0.6421 chrF++. Đây là tín hiệu validation một seed, chưa phải
test hay kết luận ổn định qua seeds. Checkpoint 2.47GB đã tạo; upload HF đang ở
khoảng7%, chưa có verified receipt, nên queue chưa sang P4 tại thời điểm ghi.

Cập nhật upload P3: đã lên khoảng83% (2.02/2.44GB phần dữ liệu mới) nhưng tốc độ
hiện dao động thấp khoảng0.3MB/s; chưa có receipt/VAL_COMPLETE, P4 chưa bắt đầu.
Queue process vẫn sống.

### Đã dừng theo yêu cầu sau P3; tiếp tục P4 vào ngày sau

P3 upload lên repo `FiveC/amis-rewire-checkpoints` đã được xác minh (`verified:true`),
revision `4860fed08b8fdea7c4d1662c412729297ec43286`. Đường dẫn:
`checkpoints/amis_mandarin/mbart-large-50/ordinary-full-v1/seed-42/rtx6000-fp32-b4-ga32-es10-tau0.1-beta0.1/`.
P3 được chọn tại step576/epoch16; validation BLEU18.4584, chrF++15.1246,
chrF0 17.3609. Queue tự động chuyển sang P4 ngay sau P3, trước khi poll nhận
được yêu cầu dừng; P4 đã chạy 10 optimizer steps rồi bị dừng theo yêu cầu. Không
có checkpoint epoch trong thư mục P4; `progress.json` được cập nhật thành
`STOPPED_BY_USER_AFTER_P3`. Process group của queue đã dừng và xác minh không còn
P4/queue process chạy. P4 cần chạy lại từ đầu theo cùng locked spec vào phiên sau;
P5 chưa bắt đầu.

### Tiếp tục P4–P5 ngày 12/10/2026

Theo yêu cầu tác giả, queue đã khởi chạy lại (PID2732284). Kiểm tra trước launch:
GPU trống19298MiB, utilization87%; không có guarded_runner đang chạy. P1–P3
VAL_COMPLETE/backup verified được kiểm tra và bỏ qua. P4 không có checkpoint epoch,
nên bắt đầu lại từ đầu bằng cùng locked spec; P5 chạy nối tiếp sau P4 upload và
xác minh. Process tree xác nhận P4 worker chạy trong conda clrr. Theo dõi mỗi10 phút.

### P4 — kiểm tra sau khoảng10 phút ngày12/10

Queue PID2732284 sống (elapsed10:31). Receipt P4 step70/720, epoch1.95;
log đang validation epoch2. Patience0; chưa có validation cuối hoặc upload.
GPU trống3034MiB, utilization99%; không có lỗi trong log kiểm tra. P5 chưa bắt đầu.

### P4 — mốc khoảng20 phút

Queue sống (elapsed21:05). P4 receipt step140/720, epoch3.89, patience0;
log đang validation epoch4. Validation hoàn tất gần nhất epoch3: BLEU9.3052,
chrF++9.1482. Mốc đầu training chưa dùng kết luận hiệu quả. GPU trống3034MiB,
utilization99%; P5 chưa bắt đầu. Không thấy lỗi trong log mới kiểm tra.

### P4 — mốc khoảng30 phút

Queue sống (elapsed31:30). P4 epoch6/20, step216/720; validation epoch6 đạt
BLEU12.8689, chrF++11.6760, tăng so epoch5 (11.7756/10.8247). Patience0;
chưa có final checkpoint/upload, P5 chưa bắt đầu. GPU trống9914MiB, utilization96%.
Không thấy lỗi trong phần log mới kiểm tra.

### P4 — mốc khoảng40 phút

Queue sống (elapsed41:56). P4 receipt step310/720, epoch8.61; log tới319.
Validation epoch8 BLEU15.2045, chrF++13.1933, tăng so epoch7 (13.9185/12.1129).
Patience0; chưa chốt kết quả cuối. GPU trống9914MiB, utilization96%; P5 chưa chạy.

### P4 — mốc khoảng50 phút

Queue sống (elapsed52:22). P4 receipt epoch11.11, step400/720, log tới406.
Validation epoch11 BLEU15.1805/chrF++12.9814, chưa vượt best selection ở epoch8
(BLEU15.2045/chrF++13.1933); patience3/10. Đây là dao động giữa training, chưa
kết luận kết quả cuối. GPU trống9914MiB, utilization96%; chưa có upload P4, P5 chưa chạy.

### P4 — mốc khoảng60 phút

Queue sống (elapsed1:03:09). P4 receipt epoch13.89, step500/720; log đang
validation epoch14. Best selection vẫn chrF++13.1933 ở epoch8; patience5/10
sau epoch13. Epoch13 BLEU14.8025/chrF++12.8968. P4 chưa cải thiện selection
trong các epoch gần đây; chờ kết quả cuối để chốt. GPU trống9914MiB,
utilization58%; P5 chưa bắt đầu, chưa có upload P4.

### P4 — mốc khoảng70 phút

Queue sống (elapsed1:13:33). P4 receipt epoch16.39/20, step590/720, log tới598.
Validation best mới tại epoch15 BLEU15.8586/chrF++13.6423; epoch16 giảm còn
15.3959/13.3796, patience1/10. Chờ validation cuối; chưa có upload P4. GPU trống
9914MiB, utilization94%; P5 chưa bắt đầu.

### P4 — mốc khoảng80 phút

Queue sống (elapsed1:23:58). P4 epoch19/20, step684/720; log tới687.
Validation epoch19 BLEU15.7011/chrF++13.6663, best chrF++ mới, patience0.
Chưa có final metrics/upload; P5 chưa bắt đầu. GPU trống9914MiB, utilization90%.

### P4 train/validation hoàn tất — đang upload

P4 đủ20 epoch/720 steps; checkpoint chọn step684/epoch19. Validation beam1:
BLEU15.7011, chrF++13.6663, chrF0 15.6602. So baseline RTX R1 BLEU17.6823,
P4 thấp1.9812 BLEU; so P1 CLRR-only BLEU16.6216 thấp0.9205 BLEU. Effect-clrr-v1
chưa cho tín hiệu cải thiện ở development seed42; chưa kết luận nhiều seed/test.
Upload checkpoint đang~98%; chưa có verified receipt/VAL_COMPLETE. Queue còn sống,
P5 chưa bắt đầu tại poll elapsed1:35:50.

### P4 upload xác minh; P5 bắt đầu

P4 có VAL_COMPLETE và receipt verified:true, HF revision
`3d6fdd544bd3cb3d8514b743ed3068feca2acfe4` trong FiveC/amis-rewire-checkpoints.
`best.zip` đã được engine xóa khỏi RTX sau xác minh. Queue tự khởi chạy P5
(effect-full-v1), log step4/720, worker PID2783729 còn sống. GPU trống9948MiB,
utilization92%. Tiếp tục theo dõi P5; không mở thêm runs ngoài kế hoạch.

### P5 — mốc10 phút đầu

P5 worker sống (elapsed10:38). Receipt epoch2.22, step80/720, log tới87;
validation epoch2 BLEU6.2385/chrF++7.4600, patience0. Còn đầu training nên chưa
đánh giá hiệu quả. GPU trống9916MiB, utilization94%; chưa có lỗi/final upload.

### P5 — mốc20 phút

P5 worker sống (elapsed21:21), step180/720/epoch5; validation epoch5 đang chạy.
Validation hoàn tất gần nhất epoch4 BLEU10.6726/chrF++10.2861, patience0.
GPU trống9916MiB, utilization65%; chưa có final artifact/upload hoặc lỗi mới.

### P5 — mốc khoảng30 phút

P5 worker sống (elapsed33:37). Receipt epoch7.78/20, step280/720; log đang
validation epoch8. Epoch7 BLEU13.4411/chrF++11.8028, patience0. Đây vẫn là
mốc giữa training. GPU trống9916MiB, utilization64%; chưa có final/upload hay lỗi mới.

### P5 — mốc khoảng40 phút

P5 worker sống (elapsed44:36), epoch10.56/20, step380/720. Validation epoch10
BLEU15.7597/chrF++13.5302, best mới, patience0. GPU trống9916MiB,
utilization92%; chưa có final checkpoint/upload hoặc lỗi mới trong log kiểm tra.

### P5 — mốc khoảng50 phút

P5 worker sống (elapsed54:59), epoch13/20, step468/720. Best validation tại
epoch12 BLEU16.1430/chrF++13.7952; epoch13 giảm15.6636/13.5080, patience1/10.
Worker trạng thái D tại poll ngay sau evaluation (có thể đang lưu checkpoint),
chưa có lỗi log; không suy ra run treo từ snapshot này. GPU trống9916MiB,
utilization0%; chưa final/upload.

### P5 — mốc khoảng60 phút

Worker sống (elapsed1:05:33), epoch15.56/20, step560/720. Best vẫn epoch12
BLEU16.1430/chrF++13.7952; epoch15 đạt15.8408/13.5668, patience3/10.
GPU trống9914MiB, utilization92%; chưa thấy lỗi mới, chưa final/upload.

### P5 — mốc khoảng70 phút

Worker sống (elapsed1:15:52), receipt epoch17.78/20, step640/720; log đang
validation epoch18. Best mới epoch17 BLEU16.4360/chrF++13.9875, patience0.
GPU trống9914MiB, utilization63%; chưa có final checkpoint/upload hoặc lỗi mới.

### P5 training/validation hoàn tất — đang upload

P5 đủ20 epoch/720 steps; checkpoint chọn step684/epoch19. Validation beam1:
BLEU17.1512, chrF++14.4008, chrF0 16.5448. So P3 ordinary-full-v1 BLEU18.4584,
P5 thấp1.3072 BLEU; so RTX R1 baseline BLEU17.6823 thấp0.5311 BLEU. Nhánh effect
Full chưa có tín hiệu vượt baseline hoặc ordinary Full ở seed42. Upload P5 đã
bắt đầu (~1%), chưa có verified receipt/VAL_COMPLETE. Cần chờ xác minh đủ5 runs
trước tổng hợp kết luận theo gates đã khóa.
