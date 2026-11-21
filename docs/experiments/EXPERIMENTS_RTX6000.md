# Thí nghiệm phát triển phương pháp — RTX6000

## Virtual corrective-order pilot — 2026-11-19, queue đã mở

Tác giả duyệt làm đủ initialization calibration → CPU → GPU optimizer gate → một pilot Amis seed42. Recipe mới: `configs/experiments/amis_virtual_corrective_order_pilot_rtx.json`; output cùng tên trong `outputs/experiments/`. Queue PID139985, detached trong đúng project RTX. **Trạng thái hiện hành: CPU/GPU gates PASS, training dài đã bắt đầu (720 optimizer steps tối đa).** Không mở test/replication.

| Bước | Kết quả/trạng thái | Bằng chứng |
|---|---|---|
| Hiệu chuẩn tại pretrained mBART revision e30b6cb8… | PASS; 3 TRAIN batches, 0 optimizer steps; lambda cố định136.27606275570483 | `outputs/experiments/amis_virtual_order_initialization_rtx/COMPLETE.json`, `CALIBRATION_BATCHES.json`, `COLLECTION.json` |
| Kiểm tra precision tại initialization, eta1 | Relative FP32/FP64 penalty errors0.000514%,0.000211%,0.001649% | Cùng `CALIBRATION_BATCHES.json` |
| CPU công thức/backward/padding/checkpoint/RNG cursor | 7/7 PASS local và RTX | `amis_virtual_corrective_order_pilot_rtx/CPU_GATE.json` |
| GPU optimizer gate | PASS: 2steps B4/GA32, save/load, beam-cache và optimizer/scheduler/RNG resume; peak14.3486GiB, gate390.18seconds | `gates/VCO_S42/PASS.json` khi hoàn tất |
| Training | VCO_S42 đã vào training dài sau gate PASS; max720steps; chưa có validation scores | `runs/VCO_S42/progress.json`, `VAL_COMPLETE.json` khi hoàn tất |

Giữ protocol lock `configs/protocols/amis_drb_bleu.json`: FP32, SDPA, native per-layer **reentrant** checkpointing, LR5e-5, B4/GA32, max20epochs, ES10 chọn validation BLEU beam1/FP32, original CSV references. Chunked head algebra dùng non-reentrant recomputation nội bộ; không đổi backbone backend hoặc thêm persistent parameters. Eta1/r32/basis11 đã khóa; lambda là median0.1×||gCE||/||gorder|| tại pretrained, không lấy4.13724e7 của trained W. Ratios trên ba batch khoảng0.1891/0.0583/0.1000; đây không phải Adam update ratio.

Support/probe luân phiên rows01/23. Cursor được lưu trong `virtual_order_config.json` và khôi phục khi resume; native inference không dùng virtual matrix. Log mỗi training microbatch: CE/order/weighted order, endpoint gap, partition, timing, memory; log Trainer mỗi10steps và validation mỗi epoch. Không dùng `torch.autograd.grad` unroll toàn backbone. Triển khai hiện được kiểm chứng cho mBART; không nhận đã hỗ trợ mọi backbone.

HF đích công khai: `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/virtual-corrective-order-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu/`. Runner giữ best.zip cùng metadata, calibration, validation predictions/metrics, trainer states và loss logs; chỉ dọn checkpoint local sau backup SHA verified. Chưa có checkpoint mới upload tại lúc mở queue. Bundle SHA256 `4b21cf1f3da50e6f072549ac3b28ba626844184f4ffe820560ffa113ec40dbb5`.

So sánh sau pilot: reused baseline42 validation BLEU17.6823291745 và W42 (R-Drop beta1.25)18.2724095712, cùng RTX cohort/selection; không trộn A100 hoặc test. Đây là **exploratory single-seed**, chưa kiểm định ổn định giữa seeds hoặc claim chất lượng. Gate kỹ thuật không phải quality gate. Nếu lỗi khoa học/OOM/nonfinite, queue dừng, không tự đổi batch/eta/precision.

Lệnh trong conda clrr (đã launch, không chạy launch lần hai):

```bash
python scripts/experiment.py monitor --config configs/experiments/amis_virtual_corrective_order_pilot_rtx.json --account 1
```


## Virtual corrective-order — pha chẩn đoán hoàn tất trước pilot

Code riêng `src/methods/virtual_corrective_order/v1/`, không thay exact method cũ.6CPU checks PASS local/RTX. Ba jobs fixed-W42 đã kết thúc, **0optimizer steps**, không validation/test: `amis_virtual_corrective_order_probe_rtx`, `amis_virtual_corrective_order_response_rtx`, `amis_virtual_corrective_order_backbone_rtx` trong `outputs/experiments/`.

| Phép đo | Kết quả | Giới hạn |
|---|---|---|
|48settings trên8TRAIN microbatches,2bases | Null/replay chính xác; eta1 median FP32/FP64 relative SKL error0.093%, max0.400% | Eta0.01/0.1 không đủ tin cậy; không phải48samples độc lập |
|Gradient decoder features, eta1 | FP32/FP64 cosine≥0.999979; median cosine order/W0.1774 | Không chứng minh sửa đúng lỗi dịch |
|Full-backbone gradients,3TRAIN batches | order/CE cos0.1765/0.6961/0.3807; order/W0.1352/0.4620/0.3476 | Chưa full-backbone FP64 gradient audit; không phải Adam-update ratio |
|Scale tại trained W | Median lambda4.13724e7 cho mục tiêu0.1×CE gradient norm; actual ratios0.1000/0.0563/0.1192 | **Không dùng lambda này cho pretrained initialization** |
|Short-batch total forward/backward | CE2/W/order0.365/0.371/0.558seconds; peak allocated≈6.315GiB | Source/target padded24, microbatch4, no optimizer; timing order tại eta0.1, chưa numerically admitted |

Báo cáo/source hashes/commands: `outputs/experiments/amis_virtual_corrective_order_probe_rtx/REPORT.md`. Runtime Torch2.6.0+cu124/Transformers4.57.6; tokenizer IDs của32rows khớp pinned original tokenizer dù runtime in cảnh báo regex. Chỉ lấy JSON về máy; đã dọn6.8926GiB weights tải/giải nén tạm trong đúng hai output mới, checkpoint HF gốc giữ nguyên. Tiếp theo cần scale tại đúng initialization và optimizer preflight rồi mới validation pilot; chưa có BLEU/chrF mới của candidate.

## Corrective-order — trạng thái hiện hành 2026-11-19: dừng vì chi phí

Queue **PID4014383 đã dừng**, recipe `configs/experiments/amis_corrective_order_finite_relu_rtx.json`, output cùng tên dưới `outputs/experiments/`. CPU local và RTX **14/14 PASS**; activation audit hoàn tất: **8790/10027008** vị trí đổi dấu ở eta5e-5 (48 layer/view records). Null replay và calibration đã qua; gate R2 hoàn tất **1/2 optimizer steps**, rồi OOM ở microbatch33 khi tính diagnostic endpoint distance bằng FP64. Receipt ghi process dùng21.80GiB, còn1.58GiB và yêu cầu thêm1.91GiB. Một optimizer step của gate mất **7743.34giây (~129phút)**; đây không phải thời gian một epoch train. **0/6 train dài**, chưa có số validation chất lượng dịch mới. Nguồn: `outputs/experiments/amis_corrective_order_finite_relu_rtx/rtx_status.json` và `gates/R2_ORDER_S42/microbatch_progress.json`.

**Quyết định thiết kế:** không relaunch exact full-parameter unroll trong vòng này; chuyển sang nghiên cứu cơ chế có chi phí thấp hơn theo yêu cầu tác giả. OOM trực tiếp nằm ở diagnostics, nhưng sửa diagnostics không giải quyết chi phí forward/backward bậc cao đã đo. Đây là NO-GO về chi phí triển khai hiện tại, **không phải** bằng chứng chất lượng dịch kém. Không sửa raw receipts, công thức hoặc protocol để biến gate thành PASS. Bundle archive SHA256 `e5f8656c6623cc5a34e81611e802ef2144867ec19e664cfd92dde10642969507`.

Queue cũ PID4003635 đã dừng: CPU11/11 và full-mBART NULL_REPLAY PASS (endpoint distance=0, probe SKL=0), nhưng4 eta nhỏ không đạt scaling gate;0/6 train dài,0 optimizer step, không test. Calibration895.38giây. Receipts vẫn nằm trong `outputs/experiments/amis_corrective_order_small_step_rtx`; không sửa thành PASS.

Đã kiểm tra integrity sau lần dừng: code model/prepare/storage và recipe trên RTX khớp SHA bundle4/4. AST của `compose_paths`, `replay`, `symmetric_kl`, `_evaluate`, `objective` giống bundle đầu tiên; cơ chế và công thức không đổi. Thay đổi thực sự: CPU→disk saved tensors; sửa lỗi ghi log eta; grid hiệu chuẩn .003/.01/.03→.00003/.0001/.0003/.001; thêm null replay. Đổi grid là đổi siêu tham số hiệu chuẩn, không phải thay đổi loss. Data và toàn bộ training recipe giữ nguyên. Nguồn `method_integrity_check.json`, `order_preparation/NULL_REPLAY.json`, `NUMERICAL_PROGRESS.json`, `FAILED.json` trong output hiện hành.

| Eta mới | Probe SKL thô | SKL / eta^4 | Endpoint distance |
|---|---:|---:|---:|
| .00003 | 1.29820072e-7 | 160271694133.02 | .0000801820 |
| .00010 | 4.48268752e-6 | 44826875205.79 | .0004418752 |
| .00030 | .00016282559 | 20101924527.21 | .0028146296 |
| .00100 | .00492295367 | 4922953672.50 | .0193356316 |

Đã xác định lỗi trong giả định của gate: config mBART dùng **ReLU**, trong khi gate bắt buộc smooth eta²/eta⁴ scaling. Phản ví dụ FP64 chạy bằng chính `compose_paths` cho endpoint exponent1.5 và KL exponent≈3; vì vậy không thể bắt buộc quy luật hàm trơn trên mọi ReLU finite map. Đây chưa phải chứng minh nguyên nhân duy nhất của số full-model: queue mới đo trực tiếp activation crossings và giữ null/gradient/resource gates. Nguồn `backbone_numerical_audit.json`, `relu_counterexample.json`, `notebook_relu_gate_debate.json` trong output cũ; amendment `docs/research/CORRECTIVE_ORDER_FINITE_RELU_FIX.md`.

Thay đổi admission được khai báo riêng: eta cố định5e-5 (không sweep, không chọn theo validation/test), yêu cầu audit ReLU có crossings, finite positive loss/outer gradients, original3TRAIN-batch0.10gradient-ratio calibration và đủ6GPU gates. Công thức loss giữ nguyên; không còn nhận eta^4 là quy luật đã xác minh. AST `compose_paths/replay/symmetric_kl/_evaluate` giống bundle cũ; AST `objective` giống sau khi bỏ trường **log** dropout_seeds mới. Không đổi ReLU/precision/batch/stop-gradient.

Audit trực tiếp `outputs/experiments/amis_corrective_order_finite_relu_rtx/order_preparation/ACTIVATION_AUDIT.json` ghi8790crossings/10027008positions, loại padding và so cùng dropout seed. Điều này xác nhận virtual steps thực sự đi qua ReLU boundaries; chưa cô lập đóng góp rounding nên không quy kết mọi scaling deviation chỉ do ReLU.

Log mới: `microbatch_diagnostics.jsonl` mỗi microbatch (seeds, fingerprint, loss components, thời gian, RAM/VRAM/disk); `microbatch_progress.json`; `training_events.jsonl` (LR, gradient sau clipping, validation, saves); `training_failure.json` traceback đầy đủ; lỗi ngoài microbatch còn lưu `failed_trainer_state.json`. GPU gates cũng có microbatch logs. Log thành công upload cùng checkpoint; log lỗi giữ trên RTX để tải. Chưa đo gradient riêng CE/auxiliary mỗi step.

| Nhánh | Seed | Vai trò |
|---|---|---|
| R0_CE2_S42 | 42 | CE hai dropout views |
| R1_RDROP_S42 | 42 | R-Drop beta1.25, cùng execution cohort |
| R2_ORDER_S42 | 42 | Corrective-order, trọng số trung tâm |
| R3_ORDER_LOW_S42 | 42 | Trọng số trung tâm /3 |
| R4_ORDER_HIGH_S42 | 42 | Trọng số trung tâm x3 |
| R5_CURVATURE_S42 | 42 | Đối chứng symmetric curvature |

- Amis4600train/576validation/575test; queue chỉ train/validation. B4, GA32, FP32, LR5e-5, tối đa20epochs, ES10 theo validation BLEU beam1.
- SDPA MATH, non-reentrant whole-functional checkpointing, exact full-backbone derivatives, tensor tạm xuống disk trong project; chừa6GiB host RAM và40GiB disk. W cũ không phải matched execution comparator của cohort này.
- Grid nhỏ cũ `[.00003,.0001,.0003,.001]` đã FAIL ngưỡng scaling [.5,2]; giữ nguyên receipt. Queue hiện hành dùng amendment finite-ReLU đã pin, không nới ngưỡng cũ hoặc đánh tráo PASS.
- Thứ tự hiện hành: CPU → ReLU activation audit → null replay → fixed-eta finite objective → gradient-scale calibration → đủ6 GPU gates → train lần lượt. Checkpoint/metrics/calibration/logs upload HF và xác minh SHA trước dọn weights.
- HF: `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/rtx6000-fp32-sdpa-math-disk-offload-b4-ga32-es10-select-bleu/finite-relu-eta-5em5/<variant>`.
- Code `src/methods/corrective_order/v1/`; thiết kế `docs/research/CORRECTIVE_ORDER_PILOT.md`; thực thi disk `CORRECTIVE_ORDER_DISK_PREFLIGHT.md`; sửa admission `CORRECTIVE_ORDER_SMALL_STEP_FIX.md` trong cùng thư mục research.

### Các lần khởi động trước, chưa train dài

| Output recipe | PID | Kết quả |
|---|---:|---|
| amis_corrective_order_rtx | 3931397 | CPU offload dừng ở ngưỡng chừa6GiB RAM |
| amis_corrective_order_disk_rtx | 3934133 | Disk chạy được; lỗi ghi diagnostics trùng khóa eta, đã sửa |
| amis_corrective_order_disk_retry_rtx | 3968175 | 3 eta probe hoàn tất582.48giây; không đạt scaling gate |

Biên nhận lỗi ở `outputs/experiments/<recipe>/order_preparation/FAILED.json`. Retry cũ có `NUMERICAL_PROGRESS.json`: eta .003/.01/.03, probe SKL .08067897/.48645709/1.92564488, SKL/eta^4 996036678.51/48645709.07/2377339.36, endpoint distance .07647776/.39795603/1.47338105. Tỷ số penalty chuẩn hóa khoảng .04884/.04887 nên dừng đúng gate. Chưa phải bằng chứng chất lượng dịch thấp hoặc toàn bộ cơ chế sai.

Phần bên dưới giữ nguyên snapshot và số cũ, không gộp với queue mới.

> Snapshot đã đối chiếu receipts: **2026-11-17 19:05:20 +07**. Đây là sổ tổng hợp hiện hành; trạng thái queue có thể thay đổi sau snapshot.
> RTX dùng để phát triển/sàng lọc. Không trộn các số này vào kết quả A100 và không coi test đã xem là xác nhận độc lập.

## Mục lục

1. [Trạng thái hiện tại](#1-trạng-thái-hiện-tại)
2. [Protocol và cách đọc số](#2-protocol-và-cách-đọc-số)
3. [Bảng đối chiếu trọng tâm](#3-bảng-đối-chiếu-trọng-tâm)
4. [Toàn bộ checkpoint hoàn tất](#4-toàn-bộ-checkpoint-hoàn-tất)
5. [Hướng đã dừng hoặc chưa train đủ](#5-hướng-đã-dừng-hoặc-chưa-train-đủ)
6. [Kết luận sàng lọc và thống kê](#6-kết-luận-sàng-lọc-và-thống-kê)
7. [Mục lục chẩn đoán cơ chế](#7-mục-lục-chẩn-đoán-cơ-chế)
8. [Nguồn, lưu trữ và cách cập nhật](#8-nguồn-lưu-trữ-và-cách-cập-nhật)

## 1. Trạng thái hiện tại

| Queue / nhóm | Trạng thái tại snapshot | Việc tiếp theo đã được duyệt |
|---|---|---|
| Test bổ sung model cũ, PID3454963 | 44/44 COMPLETE_VERIFIED; 0 lỗi mới | Hoàn tất; không chạy lặp |
| AC test, PID3456273 | COMPLETE_VERIFIED; BLEU21.2938 / chrF19.6703 / chrF++19.7285 | Hoàn tất seed42; không train lại |
| DRM sửa validation, PID3479703 | TRAINING; epoch9.17, step330; đã qua gates | Hoàn tất train/upload/test trước AC replication |
| AC replication, PID3532042 | WAITING_PRIOR_DRM; CPU9/9 PASS | AC43 → AC44; probes và cả hai GPU gates PASS trước train dài; chưa mở test |
| Baseline Amis42/43/44, W42/43/44, Middle42 | Có kết quả test | Tái sử dụng đúng archive SHA; không chạy lặp |

Kiểm kê **97 checkpoint chính đã hoàn tất validation**: 49 test cũ đã xác minh + 2 token test đã backup + 44 test bổ sung + AC42 = **96 checkpoint có test**;1 Turkish bị loại khỏi queue theo yêu cầu trước. Registry ở mục4 vẫn là snapshot kiểm kê ban đầu; kết quả test mới đủ44 nằm trong `outputs/experiments/rtx_documentation_snapshot/latest_test_status.json`. DRM chưa hoàn tất nên không thuộc97 checkpoint này. Bốn archive selector chrF phụ không nằm trong phạm vi kiểm kê checkpoint chính.

**DRM:** lần gốc lỗi tại validation đầu tiên, step36, chưa lưu epoch checkpoint. Sửa hai lỗi tích hợp: bỏ `labels` khỏi generation kwargs và ghi loss vào mapping `Seq2SeqLMOutput`. Không đổi công thức memory, objective, data, seed hoặc protocol. CPU regression tái hiện lỗi trước sửa rồi PASS sau sửa; GPU gate mới kiểm tra cả đường prediction của Trainer. Chưa coi CPU PASS là train thành công.

Bản sửa đã deploy sau khi hai queue test trước kết thúc và đã lấy project lock. Restart từ pretrained gốc vì không có checkpoint resume; không train lại AC. Nếu admission/gate lỗi, pipeline dừng trước train dài. Các process của người khác không bị tác động.

| Mục vận hành | Nguồn |
|---|---|
| Queue test44 | `outputs/experiments/rtx_test_backfill_dispatch/{manifest,status}.json` |
| Queue test AC | `outputs/experiments/amis_two_method_round1_rtx_ac_test_dispatch/status.json` |
| Queue DRM sửa | `outputs/experiments/amis_drm_validation_fix_dispatch/{status,launch}.json` |
| Recipe DRM sửa | [amis_drm_validation_fix_rtx.json](../../configs/experiments/amis_drm_validation_fix_rtx.json) |
| Output DRM sửa | `outputs/experiments/amis_drm_validation_fix_rtx/` và hậu tố `_test` |
| Thiết kế AC/DRM | [TWO_METHOD_DESIGN_ROUND1.md](../research/TWO_METHOD_DESIGN_ROUND1.md) |

Bundle DRM sửa SHA256: `20268e725987c66e3a04dec02e585e5a8868b2c49a6e00578971fc28b558b99c`; manifest/CPU receipt: `d20743c76a7dcf3288ac9b268a1d39b6e02e8a23d6d87c3992173396f69a1538`. Lần gốc và các lỗi vẫn giữ để truy vết.

### AC replication test — tác giả duyệt 2026-11-18

- DRM42 đã train/test `COMPLETE_VERIFIED`: validation BLEU18.472439; test BLEU21.098343 / chrF19.374074 / chrF++19.491063.
- AC43/44 đã train và backup `COMPLETE_VERIFIED`; validation BLEU18.289385 /18.002423.
- Đã mở test riêng AC43/44, PID3742542; trạng thái xác minh ban đầu `TEST_RUNNING`. Original test575 câu, FP32 beam4, checkpoint best validation BLEU giữ nguyên; không train/tune lại.
- Recipe test: `outputs/experiments/amis_ac_replication_rtx_test/PLAN.json`; frozen archive SHA43 `7fb10abdef4bddae02a4156fcb5d2461ebe57f07ab2b834ac1ea6d0b70d0a284`, SHA44 `553960c39ee7fdf9596b2672f7798d648793c26ece99d2ada4fa7dd2d5c04a02`.
- Local launch/verification: `outputs/experiments/amis_ac_replication_test_dispatch/`. Queue dùng evaluator chung, verified HF backup trước dọn weights; không chạy lại AC42/DRM42. Test này được duyệt sau recipe validation-only, không phải test tự động của recipe gốc.

### AC replication đã được tác giả duyệt

- Đúng2 runs mới: **AC seeds43/44**, giữ nguyên recipe seed42. Baseline/W42–44 và AC42 được tái sử dụng.
- Chờ DRM dispatcher `COMPLETE_VERIFIED`, sau đó project lock, CPU/probe/cả hai GPU gates trước train dài; không thay source đang chạy.
- Recipe: [amis_ac_replication_rtx.json](../../configs/experiments/amis_ac_replication_rtx.json); thiết kế: [AC_REPLICATION_RTX.md](../research/AC_REPLICATION_RTX.md).
- Queue receipt: `outputs/experiments/amis_ac_replication_dispatch/{launch,verification}.json`. Public HF: `checkpoints/amis_mandarin/mbart-large-50/antisymmetric-consistency-v1/seed-{43,44}/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`. Backup verified trước khi shared engine dọn weights.
- Đây là replication thăm dò sau khi xem test42; không đổi kết luận AC chưa qua validation promotion gate cũ, không xác nhận novelty, chưa mở test mới.

## 2. Protocol và cách đọc số

| Thành phần | Quy ước |
|---|---|
| Phạm vi RTX | `/workspace/ura/thanhdat/projects/CLRR/`; Python/conda `clrr`; Quadro RTX6000 |
| Backbone | mBART-50; revision gốc `e30b6cb8eb0d43a0b73cab73c7676b9863223a30` cho các cohort được ghi bên dưới |
| Amis → Mandarin | Train4600 / validation576 / test575; `tl_XX → zh_CN`; BLEU tokenizer `zh` |
| Ê Đê → Vietnamese | Train15092 / validation1000 / test1000; recipe chính `id_ID → vi_VN`; BLEU `13a` |
| Numerical recipe phổ biến | FP32, TF32off, B4×GA32, LR5e-5, cap20epochs, ES10, beam1 validation/beam4 test |
| Ngoại lệ bắt buộc tách | Eager depth screen; chrF++ và BLEU selectors; sequence-risk continuation LR1e-5/cap3; source proxy; cấu hình/module/compute khác |
| Định danh duy nhất | `(cohort, run ID, archive SHA256)`; P1/P2/C0/Full ở các cohort khác nhau không phải cùng method |
| Baseline tái sử dụng | Seed42 best BLEU và chrF++ cùng epoch19/step684; seeds43/44 đã audit full trajectory. SDPA lịch sử là evidence tái dựng, không phải runtime receipt đương thời |
| B và W | B = native paired R-Drop beta2.5; W = beta1.25. **B không phải baseline**; W là tuned R-Drop control, không phải novelty độc lập |
| Các tên LSR/Full | `original-lsr`, `original-full`, `ordinary-lsr-v1`, `ordinary-full-v1`, token/transition/learned-strength là các recipe khác; xem spec, không gộp theo tên viết tắt |
| Metrics | BLEU; chrF = `word_order=0,beta=2,char_order=6` (key raw `chrf0`); chrF++ = `word_order=2`. Giữ nguyên implementation/reference và số gốc |
| Mandarin chrF++ | Word n-gram rất thưa; audit cũ chỉ3/575 references có word-bigram nội bộ. Không dùng riêng mức tăng chrF++ lớn làm bằng chứng quality |
| Thống kê | Bootstrap theo câu có điều kiện trên checkpoint; không thay cho biến thiên seed. Test đã được dùng trong development |

Bảng đọc nhanh đặt BLEU và chrF cạnh nhau; registry vẫn giữ cả chrF++. Việc tổ chức tài liệu **không thay metric selector/early stopping** đã khóa của bất kỳ run nào.

## 3. Bảng đối chiếu trọng tâm

### 3.1. Amis test: baseline, B và W theo seed

Cùng575 câu gốc, beam4, FP32, BLEUzh. Provenance baseline và sự tương đương selector được ghi ở mục2; các số dưới đây là mô tả development, chưa phải kiểm định W–baseline mới.

| Seed | Baseline BLEU | B BLEU | W BLEU | W − baseline | Baseline chrF | W chrF |
|---|---:|---:|---:|---:|---:|---:|
|42|19.3926|19.8467|21.1631|+1.7705|18.0917|19.4598|
|43|19.5540|20.1469|21.0765|+1.5225|18.0048|19.3849|
|44|20.2453|21.3909|21.2454|+1.0001|18.5975|19.5383|

### 3.2. Amis seed42: validation và test tách riêng

| Method | Val BLEU | Val chrF | Test BLEU | Test chrF |
|---|---:|---:|---:|---:|
|Baseline|17.6823|16.9547|19.3926|18.0917|
|Middle|16.6775|16.0333|19.3753|17.8565|
|B: paired R-Drop beta2.5|17.9107|17.0603|19.8467|18.4733|
|W: paired R-Drop beta1.25|18.2724|17.3888|21.1631|19.4598|
|AC|17.6130|16.9289|—|—|
|DRM|—|—|—|—|

W hơn baseline ở cả3 seed test trong snapshot; chưa có paired-bootstrap/seed-test mới cho đối chiếu này. W không thắng B ở mọi seed. Middle RTX dùng microbatch4 negatives, không thay thế đối sánh A100 batch32. AC test đang chờ, không diễn giải ô trống là0.

## 4. Toàn bộ checkpoint hoàn tất

**Nguồn:** [97 receipt validation và test mới đã pin](../../outputs/experiments/rtx_documentation_snapshot/verified_receipts.json), [inventory/spec/protocol/HF prefix theo từng run](../../outputs/experiments/rtx_missing_test_inventory/classified_inventory.json), [test cũ theo archive SHA](../../outputs/experiments/rtx_missing_test_inventory/combined_inventory.json), [HF audit](../../outputs/experiments/rtx_missing_test_inventory/hf_test_audit.json). Registry CSV đầy đủ precision/SHA: [RUN_REGISTRY.csv](../../outputs/experiments/rtx_documentation_snapshot/RUN_REGISTRY.csv).

Mỗi bảng là một cohort provenance, **không tự động là một nhóm so sánh matched**. Điểm làm tròn4 chữ số; raw JSON/CSV giữ precision gốc. `DONE` = có test đúng archive đã đối chiếu; `QUEUED` = còn trong queue; `RETIRED` = chủ động không test. Dấu `—` là chưa có số trong receipt.

### R01. `amis_alignment_residual_bleu_selection_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_alignment_residual_bleu_selection_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|C0_BLEU_S42|alignment-residual-capacity-v1|42|18.0863|17.1478|14.8686|20.3252|19.0009|19.1402|DONE|
|LEARNED_DONOR_BLEU_S42|alignment-learned-donor-ce-v1|42|17.1248|16.4529|14.3214|20.1539|18.7504|18.9828|DONE|
|M0_BLEU_S42|alignment-matched-ce-control-v1|42|17.1720|16.2929|14.1599|19.7848|18.4608|18.7593|DONE|
|DONOR_BLEU_S44|alignment-donor-ce-control-v1|44|17.2686|16.3862|14.2402|19.8106|18.4544|18.7547|DONE|

### R02. `amis_alignment_residual_ce_controls_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_alignment_residual_ce_controls_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|CE_DONOR|alignment-donor-ce-control-v1|42|17.0378|16.3933|14.2944|19.3972|18.1445|18.4826|DONE|
|CE_LOCAL|alignment-local-ce-control-v1|42|17.0637|16.3762|14.2563|18.8421|17.5967|18.1123|DONE|

### R03. `amis_alignment_residual_replication_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_alignment_residual_replication_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|M0_S42|alignment-matched-ce-control-v1|42|17.1241|16.2932|14.1605|19.6381|18.4095|18.7196|DONE|
|C0_S43|alignment-residual-capacity-v1|43|17.5186|16.7544|14.5067|20.1715|18.6910|18.9469|DONE|
|CE_DONOR_S43|alignment-donor-ce-control-v1|43|16.8428|16.3122|14.1762|18.5958|17.5548|18.0549|DONE|
|CE_LEARNED_DONOR_S43|alignment-learned-donor-ce-v1|43|16.6133|16.2299|14.1550|19.1288|17.8419|18.2718|DONE|
|CE_LOCAL_S43|alignment-local-ce-control-v1|43|17.4941|16.8694|14.6540|20.5659|19.0283|19.2226|DONE|
|M0_S43|alignment-matched-ce-control-v1|43|17.0734|16.5222|14.3566|19.5627|18.2074|18.5214|DONE|
|C0_S44|alignment-residual-capacity-v1|44|17.0847|16.4161|14.2893|18.5278|17.5349|17.9750|DONE|
|CE_DONOR_S44|alignment-donor-ce-control-v1|44|17.2004|16.4945|14.3576|19.6747|18.4543|18.7552|DONE|
|CE_LEARNED_DONOR_S44|alignment-learned-donor-ce-v1|44|17.0501|16.4927|14.3076|18.7297|17.6332|18.1150|DONE|
|CE_LOCAL_S44|alignment-local-ce-control-v1|44|17.5116|16.7124|14.5683|19.7054|18.1137|18.4844|DONE|
|M0_S44|alignment-matched-ce-control-v1|44|17.5381|16.7126|14.5444|19.7834|18.2172|18.5794|DONE|

### R04. `amis_alignment_residual_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_alignment_residual_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|AR_C0|alignment-residual-capacity-v1|42|17.9412|17.2061|14.9673|20.1589|18.6812|18.9231|DONE|
|AR_C1|alignment-residual-fullgrad-v1|42|16.9277|16.2272|14.1528|19.0967|18.0700|18.4455|DONE|
|AR_C2|alignment-residual-confined-v1|42|16.8765|16.3203|14.2070|18.5636|17.7410|18.1971|DONE|
|AR_C3|alignment-local-confined-v1|42|17.4056|16.6518|14.5169|19.3392|18.0100|18.4616|DONE|

### R05. `amis_cmc_estimator_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_cmc_estimator_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|CV_S42|continuation-control-variate-v2|42|17.1488|16.5899|14.4398|19.7827|18.4854|18.7548|DONE|
|SAMPLED_S42|continuation-sampled-direct-v2|42|16.1332|15.8396|13.6991|19.0081|17.6498|13.5373|DONE|

### R06. `amis_consistency_strength_b_replication_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_consistency_strength_b_replication_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|B_S43|paired-native-rdrop-v1|43|17.5757|16.9585|14.7557|20.1469|18.6134|18.7659|DONE|
|B_S44|paired-native-rdrop-v1|44|18.3274|17.3530|15.0935|21.3909|19.5852|19.6675|DONE|

### R07. `amis_consistency_strength_w_replication_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_consistency_strength_w_replication_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|W_S43|weak-full-consistency-v1|43|18.3958|17.4644|15.2619|21.0765|19.3849|14.9451|DONE|
|W_S44|weak-full-consistency-v1|44|18.1881|17.1823|14.9470|21.2454|19.5383|19.5890|DONE|

### R08. `amis_continuation_consistency_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_continuation_consistency_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|CMC_S42|continuation-consistency-v1|42|15.5402|15.3932|13.3163|19.4186|18.1233|18.4194|DONE|
|NORM_S42|continuation-norm-control-v1|42|15.9929|15.7981|13.7123|19.3518|18.0539|18.4120|DONE|
|PERMUTED_S42|continuation-permuted-v1|42|15.8222|15.6701|13.5781|19.4382|18.0524|18.3639|DONE|

### R09. `amis_contrastive_controls_rtx_test`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `không ghi tường minh trong protocol; xem metadata/audit`; cap20/ES10. Nguồn `outputs/experiments/amis_contrastive_controls_rtx_test/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|A|nce-only-v1|42|17.0583|16.4069|14.3556|19.9879|18.6971|19.0258|DONE|
|B|clrr-nce-v1|42|16.8872|16.3714|14.2274|19.7927|18.4922|18.7785|DONE|
|C|attached-ordinary-full-v1|42|17.8040|16.9037|14.7318|19.8905|18.6160|18.8535|DONE|

### R10. `amis_contrastive_multiseed_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `không ghi tường minh trong protocol; xem metadata/audit`; cap20/ES10. Nguồn `outputs/experiments/amis_contrastive_multiseed_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|attached-s43|attached-ordinary-full-v1|43|17.6896|17.0744|14.8784|20.2035|18.6789|18.9207|DONE|
|baseline-s43|baseline|43|16.7345|16.3175|14.2061|19.5540|18.0048|13.8656|DONE|
|clrr_nce-s43|clrr-nce-v1|43|17.2068|16.6670|14.5303|19.8206|18.5033|18.7059|DONE|
|nce-s43|nce-only-v1|43|17.2229|16.4356|14.3065|19.6310|18.3821|18.7623|DONE|
|p2-s43|ordinary-lsr-v1|43|17.5989|16.9481|14.7462|19.2198|18.0172|18.3842|DONE|
|p3-s43|ordinary-full-v1|43|17.4966|16.7592|14.6083|20.5898|18.9980|19.1841|DONE|
|attached-s44|attached-ordinary-full-v1|44|17.6346|17.0090|14.7743|19.5308|18.1436|18.5242|DONE|
|baseline-s44|baseline|44|16.2089|15.8087|13.7450|20.2453|18.5975|18.8857|DONE|
|clrr_nce-s44|clrr-nce-v1|44|17.0276|16.2666|14.0883|19.7551|18.2982|18.6182|DONE|
|nce-s44|nce-only-v1|44|16.7508|16.2943|14.2105|19.4228|17.9660|18.3430|DONE|
|p2-s44|ordinary-lsr-v1|44|17.0216|16.3739|14.2298|19.3710|17.9302|18.3641|DONE|
|p3-s44|ordinary-full-v1|44|17.3707|16.7013|14.5100|19.3204|18.0066|18.3731|DONE|

### R11. `amis_decoder_residual_bridge_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_decoder_residual_bridge_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|D0_S42|drb-local-v1|42|16.7525|16.2512|14.1727|20.3978|18.9323|19.1552|DONE|
|D1_S42|drb-donor-v1|42|16.8277|16.3080|14.1486|20.1147|18.6459|18.8734|DONE|
|D2_S42|drb-source-v1|42|17.1272|16.4317|14.4005|19.6763|18.3045|18.7524|DONE|
|D3_S42|drb-query-v1|42|16.6736|16.3404|14.2976|20.3932|18.7862|19.0715|DONE|

### R12. `amis_decoupled_consistency_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_decoupled_consistency_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|G_S42|gold-consistency-v1|42|16.8007|16.2597|14.1567|19.7785|18.3783|18.7135|DONE|
|R_S42|rest-consistency-v1|42|17.2483|16.4760|14.4396|20.2553|18.7927|19.0280|DONE|
|W_S42|weak-full-consistency-v1|42|18.2724|17.3888|15.1728|21.1631|19.4598|19.5553|DONE|

### R13. `amis_depth_p1_sdpa_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_depth_p1_sdpa_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|P42|depth-ordinary-v2|42|16.5836|16.3074|14.2458|19.8407|18.6589|18.9698|DONE|
|P43|depth-ordinary-v2|43|16.6308|16.1315|14.0466|—|—|—|QUEUED|
|P44|depth-ordinary-v2|44|17.0428|16.3801|14.2107|—|—|—|QUEUED|

### R14. `amis_depth_screen_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `eager`; cap20/ES10. Nguồn `outputs/experiments/amis_depth_screen_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|B0|baseline|42|16.7318|16.3574|14.2890|20.6562|19.1003|19.2388|DONE|
|D0|depth-donor-addition-v2|42|16.4087|15.9793|13.9162|19.3687|18.2581|14.1640|DONE|
|P1|depth-ordinary-v2|42|17.3532|16.7157|14.4979|19.8699|18.4333|18.6491|DONE|
|P5|depth-difference-value-scaled-v2|42|16.7734|16.2875|14.2040|20.1427|18.5487|18.8735|DONE|

### R15. `amis_learned_donor_ce_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_learned_donor_ce_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|CE_LEARNED_DONOR|alignment-learned-donor-ce-v1|42|17.0231|16.4864|14.3745|20.0259|18.6591|18.9147|DONE|

### R16. `amis_learned_strength_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_learned_strength_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|P1_LEARNED_CLRR_S42|learned-strength-clrr-v1|42|16.6360|16.1669|14.1009|20.0167|18.7025|18.9378|DONE|
|P2_LEARNED_FULL_S42|learned-strength-full-v1|42|17.3459|16.5524|14.4554|19.7862|18.5342|18.8576|DONE|

### R17. `amis_middle_comparator_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_middle_comparator_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|MIDDLE_S42|middle-alignment|42|16.6775|16.0333|13.9376|19.3753|17.8565|18.3061|DONE|

### R18. `amis_original_lsr_control_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_original_lsr_control_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|LSR_COSINE_S42|original-lsr|42|17.2418|16.5527|14.4559|19.2662|18.1170|18.5186|DONE|

### R19. `amis_paired_route_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_paired_route_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|A_NATIVE_CE_S42|paired-native-ce-v1|42|16.8619|16.3565|14.2623|20.6156|18.9596|19.1768|DONE|
|B_NATIVE_RDROP_S42|paired-native-rdrop-v1|42|17.9107|17.0603|14.8427|19.8467|18.4733|18.7857|DONE|
|C_PAIRED_CLRR_CE_S42|paired-clrr-ce-v1|42|16.6620|16.0796|14.0011|20.0499|18.4625|18.7585|DONE|
|D_PAIRED_CLRR_RDROP_S42|paired-clrr-rdrop-v1|42|17.3386|16.5693|14.3730|—|—|—|QUEUED|

### R20. `amis_prefix_consistency_pilot_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_prefix_consistency_pilot_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|D_DIAGONAL_S42|diagonal-consistency-v1|42|18.2715|17.4256|15.1797|—|—|—|QUEUED|
|E_ENDPOINT_S42|endpoint-consistency-v1|42|17.9077|17.2103|14.9709|—|—|—|QUEUED|
|P_PREFIX_S42|prefix-consistency-v1|42|17.7148|17.0345|14.8455|—|—|—|QUEUED|
|WCAL_S42|calibrated-consistency-v1|42|17.2413|16.5969|14.4213|—|—|—|QUEUED|

### R21. `amis_residual_effect_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `không ghi tường minh trong protocol; xem metadata/audit`; cap20/ES10. Nguồn `outputs/experiments/amis_residual_effect_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|P1|clrr|42|16.6216|16.2232|14.1252|19.7806|18.4717|18.7598|DONE|
|P2|ordinary-lsr-v1|42|17.2700|16.5855|14.4825|19.0496|17.8877|18.3541|DONE|
|P3|ordinary-full-v1|42|18.4584|17.3609|15.1246|19.9158|18.5920|18.8726|DONE|
|P4|effect-clrr-v1|42|15.7011|15.6602|13.6663|19.2178|18.0395|18.4420|DONE|
|P5|effect-full-v1|42|17.1512|16.5448|14.4008|19.2836|18.1236|18.4591|DONE|

### R22. `amis_sequence_risk_diverse_pool_rtx`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `không ghi tường minh trong protocol; xem metadata/audit`; cap3/ES10. Nguồn `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|S0|sequence-ce-v1|42|17.2803|16.6790|14.4668|—|—|—|QUEUED|
|S1|sequence-risk-v1|42|17.5593|16.7025|14.4865|—|—|—|QUEUED|
|S2|sequence-clrr-ce-v1|42|17.4191|16.4921|14.3061|—|—|—|QUEUED|
|S3|sequence-clrr-risk-v1|42|17.2666|16.4245|14.2728|—|—|—|QUEUED|

### R23. `amis_token_rtx_test`

Dataset `amis_mandarin`; selector **original-reference validation chrF++; beam1 FP32**; backend `không ghi tường minh trong protocol; xem metadata/audit`; cap20/ES10. Nguồn `outputs/experiments/amis_token_rtx_test/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|R1|baseline|42|17.6823|16.9547|14.7030|19.3926|18.0917|13.8889|DONE|
|R2|full|42|17.1156|16.4809|14.3450|18.7821|17.6237|18.1317|DONE|
|R3|token-lsr-v1|42|17.5811|16.5823|14.5050|19.0641|17.9597|18.3830|DONE|
|R4|token-full-v1|42|16.7533|16.1561|14.0672|19.3822|18.2176|18.5570|DONE|

### R24. `amis_transition_lsr_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_transition_lsr_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|TRANSITION_FULL_S42|transition-full-v1|42|17.5220|16.8242|14.6387|—|—|—|QUEUED|
|TRANSITION_LSR_S42|transition-lsr-v1|42|17.0833|16.4927|14.2826|—|—|—|QUEUED|

### R25. `amis_two_method_round1_rtx`

Dataset `amis_mandarin`; selector **original-reference validation BLEU; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/amis_two_method_round1_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|AC_S42|antisymmetric-consistency-v1|42|17.6130|16.9289|14.7789|—|—|—|QUEUED|

### R26. `rhade_alignment_residual_rtx`

Dataset `rhade_vietnamese`; selector **original-reference validation chrF++; beam1 FP32**; backend `sdpa`; cap20/ES10. Nguồn `outputs/experiments/rhade_alignment_residual_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|BASELINE_S42|baseline|42|34.0232|51.1000|50.4714|34.0402|50.5887|50.0957|DONE|
|C0_S42|alignment-residual-capacity-v1|42|34.0381|51.1258|50.5047|33.9203|50.6000|50.0804|DONE|
|LEARNED_DONOR_S42|alignment-learned-donor-ce-v1|42|34.0729|51.0563|50.4590|34.0843|50.7116|50.1812|DONE|
|M0_S42|alignment-matched-ce-control-v1|42|34.3728|51.4447|50.8678|34.1761|50.7518|50.1916|DONE|

### R27. `rhade_contrastive_id_rtx`

Dataset `rhade_vietnamese`; selector **original-reference validation chrF++; beam1 FP32**; backend `không ghi tường minh trong protocol; xem metadata/audit`; cap20/ES10. Nguồn `outputs/experiments/rhade_contrastive_id_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|baseline-s42|baseline|42|33.9793|51.0720|50.4792|34.0215|50.3887|49.8624|DONE|
|full-old-s42|original-full|42|34.3453|51.2537|50.6336|34.1330|50.6668|50.1608|DONE|
|nce-s42|nce-only-v1|42|34.1126|51.1225|50.4884|34.1445|50.7030|50.1421|DONE|
|p3-s42|ordinary-full-v1|42|34.1195|51.1020|50.5071|33.4778|50.1493|49.6557|DONE|

### R28. `turkish_contrastive_rtx`

Dataset `opus100_turkish_english_20k`; selector **original-reference validation chrF++; beam1 FP32**; backend `không ghi tường minh trong protocol; xem metadata/audit`; cap20/ES10. Nguồn `outputs/experiments/turkish_contrastive_rtx/runs/<ID>/VAL_COMPLETE.json`.

| Run | Method ID | Seed | Val BLEU | Val chrF | Val chrF++ | Test BLEU | Test chrF | Test chrF++ | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|baseline-s42|baseline|42|20.8872|40.5219|39.5760|—|—|—|RETIRED|

## 5. Hướng đã dừng hoặc chưa train đủ

Các mục sau không có completed-validation receipt trong97 checkpoint trên; không đưa gate hoặc checkpoint giữa chừng vào bảng quality.

| Hướng / cohort | Trạng thái cuối đã ghi nhận | Điều không được suy ra |
|---|---|---|
| DRM gốc, `amis_two_method_round1_rtx` | FAILED ở validation đầu; bản sửa nằm trong queue mục1 | Không có số chất lượng DRM hoàn tất |
| Relational v2, `amis_relational_screen_rtx` | STOPPED_FOR_NOVELTY_REVIEW; A có checkpoint36, B/C/D chưa train dài | Bốn GPU gate PASS không phải bốn model hoàn tất; không tự resume |
| Difference-preserving v1, Amis/Ê Đê | TECHNICAL_ONLY;8/8 GPU fixtures PASS; semantic/novelty HOLD | Cyclic train peers không phải semantic labels; chưa có pilot quality |
| DRB Ê Đê | Resume gate FAIL; chỉ Amis được tách ra train | Không nới tolerance hoặc coi sai số nhỏ là PASS |
| Depth eager replication43/44 | Dừng ở gate; thay bằng P1 SDPA42/43/44 | Không ghép eager seed42 vào SDPA cohort |
| Sequence-risk temperature1.0 | Pool gate FAIL:55.6522% singleton; không train | Không gộp với temperature1.3 đã hoàn tất |
| Ê Đê proxy `tl_XX` | Chỉ preflight/gates; chuyển sang `id_ID` trước train dài | Không reuse gate giữa proxy |
| Turkish RTX | Tác giả dừng; baseline đã backup, NCE gián đoạn, P3 chưa train | Không mở lại/test; không trộn với Turkish A100 cũ |

## 6. Kết luận sàng lọc và thống kê

Các quyết định bên dưới là gate/nhận định tại thời điểm phát triển. Test bổ sung hiện tại không tự đảo quyết định hoặc tạo quyền mở sweep mới.

| Nhóm | Kết quả có ích để quyết định | Trạng thái phát triển |
|---|---|---|
| W/B | W validation mean18.2854, baseline16.8753, B17.9380; W−B dương2/3 seed. Test xem mục3 | Tín hiệu control tốt; W là tuned R-Drop, chưa thành novelty mới |
| G/R split | Validation G16.8007 / R17.2483 / W18.2724 | G/R không qua gate; không mở rộng split |
| Paired CLRR route | A16.8619 / B17.9107 / C16.6620 / D17.3386 validation BLEU | OFF/ON route không vượt native R-Drop trong seed42 |
| PLC | P−W−0.557598 BLEU; diagonal D−W−0.000954 | NO-GO replication/A100 cho recipe PLC hiện tại |
| CMC / estimator | CV−CMC+1.608571, nhưng CV−W−1.123622 validation BLEU | CV cải thiện estimator, không cứu quality; NO-GO mở rộng recipe |
| Learned-strength / original cosine | P2−P1+0.7099; original cosine LSR-only−baseline−0.4405 BLEU | Chưa có baseline superiority; không tăng weight/LR mù quáng |
| Transition LSR | Full−LSR+0.4387, Full−baseline−0.1604 validation BLEU | NO-GO advancement; prior gần FDD, chưa đủ novelty mạnh |
| DRB Amis | D3−D2−0.4536 BLEU; CI95%[−1.2668,+0.3607], Holm3 p0.3156 | Query gate không qua practical gate; chưa chứng minh gây hại |
| Alignment residual, chrF++ cohort | Test C0−baseline mean−0.1112 BLEU; C0 seed44 thấp hơn baseline, Holm45 p0.0269973 | HOLD promotion; không chỉ báo cáo gain validation |
| Alignment residual, BLEU cohort | Test C0 mean19.6748 vs baseline19.7306; delta−0.0558 | Selector rerun riêng; không ghi đè cohort chrF++ cũ |
| Ê Đê C0 | C0−baseline test−0.1200 BLEU; CI95%[−1.0910,+0.5726]; Holm9 p1.0 | Không qua gate; không suy đồng nhất mô hình hoặc data sai |
| Sequence-risk continuation | S3−S1−0.2927 BLEU; CI95%[−1.1405,+0.5426] (5000 draws exploratory) | NO_GO_CURRENT_RECIPE; cap3, không phải ES làm dừng |

### Các family kiểm định đã có — giữ nguyên phạm vi

| Phân tích | Family / số lần bootstrap | Kết quả và nguồn |
|---|---|---|
| Contrastive multiseed | Holm90 =10 contrasts×3 seeds×3 metrics;10000 paired draws,RNG42 | 0/90 significant; minimum Holm p0.57594241. `amis_contrastive_multiseed_rtx_test/statistics/` |
| Alignment residual test, chrF++ cohort | Holm45;10000 paired draws | 1/45 significant: C0 thua baseline seed44 BLEU. `amis_alignment_residual_replication_rtx_test/ANALYSIS.md` |
| Alignment residual test, BLEU cohort | Holm45 riêng | 2/45 significant: C0 thua baseline44 BLEU; C0 hơn Local42 chrF. Không có gain so baseline significant. `amis_alignment_residual_bleu_selection_rtx_test/` |
| Ê Đê residual test | Holm9;10000 paired draws | 0/9 significant; canonical statistics revision `2921e8b9dbc58e81f370265bea23d37a0d6a1941` |
| DRB validation D3−D2 | Holm3 cho3metrics;10000 draws,seed10042 | Post-hoc validation-only; không phải confirmation. `amis_decoder_residual_bridge_rtx_analysis/validation/` |
| W–baseline test mới | Chưa chạy kiểm định trong snapshot | Không gán significance từ3/3 delta dương; không tái dùng p từ family khác |

n=3 exact two-sided sign-flip có p nhỏ nhất0.25. p bootstrap theo câu không đo uncertainty giữa training seeds. Không gộp family hoặc chọn endpoint hậu nghiệm để tuyên bố xác nhận.

### AC42 test — paired bootstrap so baseline và W

575 câu test gốc; FP32 beam4; paired bootstrap10000, RNG20261117; p đối chiếu native SacreBLEU. Holm **6 phép** =2 cặp×3 metrics, không phải toàn bộ các candidate lịch sử. Đây là phân tích exploratory sau khi xem test, có điều kiện trên seed42; chưa kiểm tra độ ổn định qua seed.

| So sánh | Metric | Delta | CI95% | p raw | p Holm6 |
|---|---|---:|---|---:|---:|
| AC42 − BASELINE_S42 | BLEU | +1.9012 | [+0.6137,+3.1844] | 0.002000 | 0.009999 |
| AC42 − BASELINE_S42 | chrF++ | +5.8395 | [+0.6607,+14.1137] | 0.096490 | 0.385961 |
| AC42 − BASELINE_S42 | chrF0 | +1.5786 | [+0.6115,+2.5818] | 0.001000 | 0.005999 |
| AC42 − W_S42 | BLEU | +0.1307 | [-0.6422,+0.9097] | 0.271473 | 0.545345 |
| AC42 − W_S42 | chrF++ | +0.1732 | [-0.3428,+0.7132] | 0.182482 | 0.545345 |
| AC42 − W_S42 | chrF0 | +0.2105 | [-0.4162,+0.8458] | 0.181782 | 0.545345 |

AC42 hơn baseline42 có ý nghĩa ở BLEU/chrF trong family này; chưa hơn W42 có ý nghĩa. chrF++ không có ý nghĩa theo native paired bootstrap; không diễn giải mức tăng tuyệt đối lớn thành bằng chứng quality riêng. Không dùng kết quả này để sửa selector hoặc validation gate.

Nguồn: [PAIRED_BOOTSTRAP.md](../../outputs/experiments/amis_ac_seed42_test_statistics/statistics/PAIRED_BOOTSTRAP.md); design, exact p/CI, checkpoint revisions/archive SHA và prediction hashes nằm cùng thư mục cha.

### AC test đủ3 seeds — hoàn tất 2026-11-18

AC43/44 test `COMPLETE_VERIFIED`, HF backup verified; queue PID3742542 đã kết thúc. Same frozen575 original test rows, FP32 beam4, validation-selected checkpoints unchanged.

| Seed | Baseline BLEU | W BLEU | AC BLEU | AC−baseline | AC−W | AC chrF | AC chrF++ |
|---|---:|---:|---:|---:|---:|---:|---:|
|42|19.3926|21.1631|21.2938|+1.9012|+0.1307|19.6703|19.7285|
|43|19.5540|21.0765|20.9167|+1.3627|-0.1598|19.2258|19.3717|
|44|20.2453|21.2454|20.0803|-0.1650|-1.1652|18.6044|18.8409|
|Mean|19.7306|21.1617|20.7636|+1.0330|-0.3981|19.1668|19.3137|

AC cao hơn baseline BLEU2/3 seeds, cao hơn W1/3 seeds; chưa chứng minh AC tốt hơn W, mean AC−W âm. Chưa chạy kiểm định mới cho seeds43/44 hoặc aggregate seed comparison; không suy từ seed42 significance sang3 seeds. Đây là exploratory development. Nguồn: `outputs/experiments/amis_ac_replication_test_dispatch/latest_status.json`, AC42 receipt/statistics ở mục6.

## 7. Mục lục chẩn đoán cơ chế

Đây là phép đo frozen/CPU/GPU không train dài, không phải model mới trong registry. Các giới hạn quan trọng được giữ cùng số đo; raw report có đầy đủ công thức, batch và provenance.

| Chẩn đoán | Bằng chứng / giới hạn chính | Nguồn trong `outputs/experiments/` hoặc docs |
|---|---|---|
| Token R1–R4 | Covariance rank74.380/73.615/69.174/72.460; không có bằng chứng collapse nghiêm trọng ở pooled probe này | `amis_token_rtx_test/`; nhật ký gốc mục Token pilot audit |
| Contrastive multiseed | Alignment/retrieval tăng trong khi BLEU không ổn định; end-checkpoint CE–aux cosine không đủ mô tả toàn training/Adam | `amis_contrastive_multiseed_evidence/NONTRAINING_REPORT.md` |
| Amis C0 inference | Tắt donor giảm BLEU khoảng3.0–3.3; tắt learned module gần0. Không thay thế retrained ablation | `amis_residual_seed_diagnostics/{CPU_REPORT.md,gpu/GPU_REPORT.md}` |
| Ê Đê C0 inference | Tắt module−0.0065 BLEU; tắt donor−5.2337; shuffle−5.2668; correction/current0.0204–0.0419% theo layer | `rhade_residual_diagnostics/REPORT.md` |
| Route-token audit | CLRR-only và Full đều mất top1 khi tắt các nhóm rewiring; không xác nhận nên prune lúc train | `amis_route_token_interventions/REPORT.md` |
| Train-pair discrimination | Correct-source margin dương tất cả lexical near pairs đã đo; chưa có semantic pair labels; không phải generation accuracy | `difference_preserving_feasibility/baseline_train_contrasts/` |
| Transition gradient/rank | Matched/shuffled edge cosine gần nhau; chưa có bằng chứng gradient domination hoặc collapse nặng | [TRANSITION_FOLLOWUP_DIAGNOSIS.md](../research/TRANSITION_FOLLOWUP_DIAGNOSIS.md) |
| Learned strength | Injection/current norm P1/P2:0.24125%/0.17294%; OFF−ON CE CI chứa0; scalars có gradient hữu hạn | [LEARNED_STRENGTH_DIAGNOSIS.md](../research/LEARNED_STRENGTH_DIAGNOSIS.md) |
| Source response | Baseline/B/W source logp gap3.82411/3.56874/3.83621; sentence associations gần0; gap không chứng minh grounding tốt hơn | `amis_consistency_source_response_rtx/ANALYSIS.md` |
| PLC preparation/failure | Cross P−D0.02982932, CI[0.00053590,0.06009497], nhưng PLC không qua quality gate; gradient compensation không tự xác định nguyên nhân | [PLC_FAILURE_ANALYSIS.md](../research/PLC_FAILURE_ANALYSIS.md) |
| SAC | C baseline0.01173794 → W0.00877760; C/B không giảm; partial rank CI chứa0; training admission HOLD | `amis_source_attribution_frozen_rtx/ANALYSIS.md` |
| Decision branches | Local KL không được chứng minh dự báo downstream branch cost;32 validation rows/seed42; không chứng minh exposure-bias cause | `amis_decision_branches_frozen_rtx/ANALYSIS.md`; literature corrections cùng thư mục |
| CMC estimator variance | CV/old pooled variance ratio0.083786773 trên4 short TRAIN rows×2 dropout seeds; không phải generation gain | `amis_cmc_estimator_gradient_audit/REPORT.md` |
| CMC shared errors | CV42−W42 mean sentence chrF−0.859891; nhóm khó từW43/44 chỉ thăm dò, không phải fresh holdout | `cmc_estimator_review/shared_errors/INTERPRETATION_AND_NEXT_STEP.md` |
| CMC total-gradient geometry | W/CMC total cosine0.976959; cả hai auxiliary dương với CE; CV RMS error0.776001; không đo historical Adam updates | `amis_cmc_update_geometry_rtx/INTERPRETATION.md` |

**Novelty:** kiểm tra paper gốc và phản biện trước admission; NotebookLM là hỗ trợ, không phải chứng nhận novelty. CPU/GPU PASS chỉ xác nhận kỹ thuật. Các prior và câu trả lời phản biện lịch sử được giữ trong report thiết kế và nhật ký gốc; tài liệu này không tự nâng novelty verdict.

## 8. Nguồn, lưu trữ và cách cập nhật

### Nguồn chính

- Public HF: [FiveC/amis-rewire-checkpoints](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/checkpoints). Cấu trúc `checkpoints/<dataset>/<backbone>/<method>/seed-<n>/<recipe>/`; prefix chính xác theo run nằm trong registry CSV/spec.
- [RUN_REGISTRY.csv](../../outputs/experiments/rtx_documentation_snapshot/RUN_REGISTRY.csv):97 dòng, đủ precision, archive SHA, selector, provenance, đường dẫn HF và trạng thái test.
- [verified_receipts.json](../../outputs/experiments/rtx_documentation_snapshot/verified_receipts.json): snapshot remote đã đọc, includes97 validation receipts, test mới và hai biên nhận recovered token.
- [Nhật ký gốc trước chuẩn hóa](../archive/EXPERIMENTS_RTX6000_LOG_BEFORE_CONSOLIDATION_20261117.md): bảo toàn nguyên byte toàn bộ số cũ, log launch/gate, PIDs, hashes, lệnh và diễn giải. **Chỉ để truy vết; banner/lệnh cũ không phải trạng thái hoặc chỉ dẫn chạy hiện hành.**
- [Bàn giao hiện hành](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md) và [insight](../research/RESEARCH_INSIGHTS.md) là tài liệu liên quan; lần chuẩn hóa này không chỉnh nội dung hai file đó.

### Quy tắc cập nhật

1. Thay mục trạng thái hiện tại và timestamp khi có snapshot mới; không prepend thêm banner trạng thái.
2. Cập nhật đúng dòng `(cohort,ID,archive SHA)`. Test mới bổ sung cột test; không sửa số validation hoặc selector cũ.
3. Recipe/selector/backend mới có cohort riêng. Giữ full receipt; ghi rõ UNKNOWN nếu thiếu metadata, không tự gán.
4. GO/NO-GO quality, novelty, technical gate và trạng thái process là bốn điều khác nhau; ghi riêng.
5. Chỉ tải JSON/CSV/Markdown nhỏ về máy. Weights chỉ dọn sau HF backup và SHA verification; không xóa dữ liệu/process người khác.
6. Chạy Python bằng conda `clrr`. Không launch lại queue đã có PID/receipt; đọc `status.json` trên RTX trước.

### Lệnh theo dõi DRM sửa sau khi deploy

```powershell
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_drm_validation_fix_rtx.json --account 1
```

Khi còn WAITING_PRIOR_TESTS, recipe chưa được extract lên remote; đọc dispatcher `status.json`/`queue.log` qua helper `src/experiment_runner/rtx.py` thay vì kết luận monitor recipe thiếu file là queue chết.

### DRM mechanism diagnostics — 2026-11-18 COMPLETE

Không phải model training mới; không cộng vào số checkpoint. `outputs/experiments/drm_direction_probe_v3/` là kết quả hợp lệ, 32 fixed TRAIN rows ×2 pinned W42/DRM42 checkpoints, FP32 SDPA eval, optimizer0, không test. Source trained method không đổi. 3 CPU checks PASS; norm-match max error2.3164e-7, contraction coefficient max.554311; peak2484.89 MiB. Small JSON tải về, SHA khớp. V1 random control invalid, V2 guard-stopped; không dùng chúng để kết luận efficacy.

|Fixed checkpoint|DRM − no-update trajectory NLL|Pointwise sentence-bootstrap CI95%|
|---|---:|---|
|W42|−0.000004607|[−0.000013412,+0.000002902]|
|DRM42|−0.000003316|[−0.000013017,+0.000005761]|

Context/scalar controls có NLL cao hơn gradient khoảng1e-5 trên mẫu TRAIN. DRM-checkpoint gradient−random-axis CI chứa0. Các CI là exploratory TRAIN-sample uncertainty, không xác nhận BLEU gain hoặc seed stability. Không có gold semantic alignment để nhận “đã dịch hết thông tin”. Chi tiết controls, single-update, prior failures và commands: `docs/research/DRM_MECHANISM_CHECK.md` và artifact `REPORT.md`.

**Quality gate chưa đạt, không mở rộng:** validation DRM−W BLEU+.200029806<+.30, chrF++−.017085220<0. Không đổi số cũ, không dùng test để bỏ gate, không launch thêm training trong lượt này.

### Frozen W42 read recoverability — 2026-11-18 COMPLETE

Không phải training/checkpoint mới. `outputs/experiments/w_read_recoverability_rtx/REPORT.md`:32 fixed validation sentences, một middle non-special token mỗi câu, gold prefixes; W42 fixed public SHA. FP32 SDPA TF32off eval. 576 label-informed interventions, model optimizer0, không test; source K/V giữ nguyên. 12 native-correct,20 native-wrong.

|Intervention|Gold recovery3% context budget|10%|30%|Gold NLL delta30%,20 errors|
|---|---:|---:|---:|---:|
|Query|0/20|0/20|0/20|−.304510|
|Attention simplex|0/20|0/20|0/20|−.759516|
|Direct context|0/20|0/20|2/20|−2.378260|

Adjacent-rank paired control18 native-wrong positions: direct30% gold1/18 vs alternative1/18; query/simplex0/18. Direct30% gold−alternative margin-gain CI95%[−.328638,+.716139]. Không đủ tín hiệu gold-selective repair để mở train. 0 recoveries không chứng minh bất khả; local search2 restarts×24 steps không phải global optimum. Không so các số NLL này với BLEU hoặc DRM trajectory NLL khác mẫu/phép can thiệp.

2 CPU tests và real-checkpoint replay gates PASS, max logit error2.2888184e-5, parameters unchanged/no .grad; peak2394.90 MiB. PID3762972 đã xong, small JSON SHA-verified local, own restore removed, GPU free23952 MiB sau đo. Chưa upload artifacts mới HF. Commands, pinned provenance, tokenizer warning và giới hạn: `docs/research/DRM_MECHANISM_CHECK.md` mục W42 read recoverability. **Không có training queue tiếp theo.**

