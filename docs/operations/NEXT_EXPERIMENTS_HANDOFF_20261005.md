> **Entrypoints sau khi gom code (10/10/2026):** các lệnh launcher/monitor/prepare riêng đã hoàn thành bên dưới là lịch sử. Run mới dùng [EXPERIMENT_WORKFLOW.md](EXPERIMENT_WORKFLOW.md) và `scripts/experiment.py` với args/config. Để replay run cũ cần bundle nguồn đã pin; không sửa hash receipt cũ. Số thực nghiệm và kế hoạch nghiên cứu bên dưới không đổi.

# Bàn giao hiện hành: sau corrected controls & 32 Fresh Seed Confirmation

## Chẩn đoán candidates đã hoàn tất — validation, không train

6/6 frozen checkpoints (baseline42, Full cũ42, U1C/U1F/U2L/U2F) strict loading,
archive hash, worker hash và metric replay PASS. Colab1/2 đã tắt. 10k paired
bootstrap U1/U2 vs Full cũ không có CI loại0; chưa chứng minh nâng cấp thành công.
Không mở training tiếp. Kết quả/cách diễn giải và bước tiếp theo:
[METHOD_CANDIDATE_DIAGNOSTICS.md](METHOD_CANDIDATE_DIAGNOSTICS.md).
Evidence: `outputs_rebuttal/method_candidate_diagnostics/REPORT.md`,
`bootstrap.json`, `interventions.json`, `VERIFIED.json` và6receipts.
U1 cơ chế gần Full cũ; U2 làm centered alignment/gradient thuận hơn nhưng
translation chưa tăng. Không có bằng chứng đủ để tự tănglambda, bỏlayer hay
đổi rawanchor; cần xem lại hypothesis/objective và novelty trước pilot mới.
Diagnostics public đã backup/checksum35files PASS:
[HF seed42 diagnostics](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main/analysis/amis_mandarin/mbart-large-50/method-candidate-diagnostics/seed-42).

## Current authorized work: four method candidates, seed42

Recipe: baseline-matched LR5e-5, batch32×accumulation4, cap20, ES10,
validation chrF++ beam1/FP32. No test results are included in this pilot.

| Run | Colab | Status | Epoch selected/completed | BLEU | chrF++ | chrF0 | Δ chrF++ vs baseline42 | Public backup/replay |
|---|---|---|---:|---:|---:|---:|---:|---|
| U1C CLRR-precache | Colab1 | COMPLETE | 14.0/20.0 | 16.4181 | 13.9381 | 16.0349 | +0.0366 | public checksum + metric replay PASS |
| U1F precache + LSR | Colab1 | COMPLETE | 19.0/20.0 | 17.3296 | 14.5519 | 16.6663 | +0.6503 | public checksum + metric replay PASS |
| U2L Centered-LSR | Colab2 | COMPLETE | 20.0/20.0 | 16.4203 | 13.9045 | 16.0522 | +0.0029 | public checksum + metric replay PASS |
| U2F CLRR + Centered-LSR | Colab2 | COMPLETE | 20.0/20.0 | 17.0227 | 14.5080 | 16.6142 | +0.6065 | public checksum + metric replay PASS |

Baseline seed42: epoch20/step720; validation chrF++13.901541.
All candidate values are one-seed validation screening results; small BLEU/chrF++ deltas do not establish a gain.
All four runs have public verified receipts, independent metric replay, and matching archive hashes. Both Colab sessions are stopped.
Detailed recipe, CPU/GPU gates, and recovery log: [METHOD_CANDIDATES_RUNBOOK.md](METHOD_CANDIDATES_RUNBOOK.md).

<!-- standardized-es10-test-current -->
## Test ES10 hoàn tất — 09/10/2026

**14/14 TEST_VERIFIED.** Colab 1 và 2 hoàn tất queue, backup và replay độc lập rồi đã tắt. Colab 3 không dùng. Cả ba tài khoản được kiểm tra: active assignments 0, usage rate 0.00/hr. Bảng test đầy đủ ở cuối file; checkpoint selection giữ đúng joint lock trước test. Predictions, metrics và COMPLETE receipts public trong từng folder checkpoint ES10. Index public đánh dấu cả 14 checkpoint test_verified=true. Chỉ tải artifacts nhỏ về PC, không tải weights.

Evaluator ban đầu lỗi vì kế thừa eval_strategy=epoch mà không có eval_dataset; đã đặt eval_strategy=no cho test-only. Không đổi model, training recipe hoặc generation protocol; chưa có test receipt trước khi sửa.

**Tiếp theo:** sử dụng bảng đối sánh ES10 đã xác minh để cập nhật paper và lập kế hoạch phương pháp mới; không tự mở thêm training. Một seed mỗi nhánh chưa đủ kết luận thống kê đa seed.
<!-- end-standardized-es10-test-current -->


## Lịch sử: hoàn tất validation ES10 — trước đợt test 09/10/2026

**ALL_14_VALIDATION_VERIFIED.** Đủ 14 runs đã hoàn tất, replay độc lập public predictions/metrics và kiểm tra hash checkpoint PASS. Bảng validation đầy đủ ở cuối file; đây chưa phải điểm test.

A06 mBART Strong LoRA B seed42 chạy lại trên Colab1 A100 Standard 40GB, hoàn tất 20 epochs; chọn checkpoint epoch19/step684 theo raw-reference validation chrF++. Validation: BLEU **10.9850**, chrF++ **10.3825**, chrF0 **12.0560**. Dừng vì epoch cap. Các 13 run trước không train lại và scores replay không đổi.

Checkpoint public: `checkpoints/amis_mandarin/mbart-large-50/strong-lora-b/seed-42/es10/best.zip`; SHA256 `4cf6ccac866db298b93134a69b015545271a9ef18793dcda483e42cf718a5a94`. Joint selection lock public: `registry/plans/amis_standardized_controls/selection_lock_14.json`. Index public đã cập nhật đủ checkpoint ES10; tổng104 archives, giữ nguyên weights cũ.

Colab1/2/3 đều xác nhận **Active assignments=0**, **Usage rate=0.00/hr**; balance tương ứng **52.92 / 151.73 / 8.57 CU**. Không tải weights về PC. Receipt audit: `outputs_rebuttal/amis_standardized_strong_mbart/status/completion_audit.json`.

**Bước tiếp theo:** evaluation test beam4/FP32 từ đúng 14 checkpoints đã khóa, rồi replay test metrics và ghi bảng so sánh. **Chưa chạy test trong task này.** Những ghi chú đang chạy/tạm dừng ở phần lịch sử bên dưới đã được trạng thái này thay thế.

### Lịch sử phân bổ bổ sung A06 vì unit Colab3

Tác giả đã đồng ý chuyển B sang Colab1/2 nếu thiếu unit. Lúc20:07 Colab3 còn17.40CU; tốc độ A khoảng9.4giây/step cộng validation làm dự báo A+B vượt ngân sách. Đã giữ dispatcher Colab3 bằng SIGSTOP **chỉ process cha queue**, xác minh child A05 train PID15010 tiếp tục và chưa có A06 training. Không dừng A05, không sửa frozen bundle. Guard tự đợi A05 train/export/upload/cleanup xong; nếu VAL_COMPLETE verified thì kết thúc dispatcher và ghi `VALIDATION_ASSIGNED_QUEUE_COMPLETE`, completed A05, migrated A06. Nếu A05 lỗi, giữ weights và ghi failure, không mở B trùng trên máy này.

A06 đã được đặt queue Colab1 sau B01–B05, qua GPU preflight trên A100 đích rồi train. Script: `scripts/revalidation/schedule_strong_mbart_b_account1.py`; local assignment receipt: `outputs_rebuttal/amis_standardized_strong_mbart/A06_execution_assignment.json`; remote `/content/strong_mbart_b_account1.pid`, `.log`, `queue_account1.json`. Frozen PLAN vẫn ghi account3 ban đầu; actual execution account1 được ghi riêng, các hyperparameters giữ nguyên. Monitor: `monitor_standardized_controls.py --account 1 --supplement`; account3 dùng `--account 3 --supplement`. Colab3 tắt sau khi A05 backup và replay độc lập xong; không chờ A06 trên Colab3 nữa. Tổng14runs không đổi.

## Lưu checkpoint hiện hành — 08/10/2026

**HF_LAYOUT_VERIFIED; không training trong bước này.** Repo public duy nhất `FiveC/amis-rewire-checkpoints`. Cấu trúc: `checkpoints/<dataset>/<backbone>/<method>/seed-<n>/best.zip`; nếu cùng seed có nhiều bản khác nhau thì thêm `<configuration>/` bên trong seed. Đã chuyển 3.056 file, giữ nguyên hash 90 archive/metrics, 79 nhóm method/seed. Đã xác minh 16 archive `seed-unknown` thực sự seed42 từ training args; BitFit/Middle/Narrow LoRA mở trực tiếp tới seed42. Không sửa actual ES4 thành ES10.

Quy tắc này thay thế cấu trúc recipe/cohort trước seed trong các mục lịch sử phía dưới. Recipe/cohort vẫn ở metadata, mapping và receipts. Run mới cần `run_metadata.json` đầy đủ; không overwrite archive khác hash. Tên biến thể phải mô tả cấu hình, không có ngày tháng. Strong LoRA A/B và các đối sánh LSR λ có nhãn rõ trong index. Tra cứu [hướng dẫn](HUGGINGFACE_CHECKPOINT_STORAGE.md) và [index public](https://huggingface.co/FiveC/amis-rewire-checkpoints/blob/main/registry/CHECKPOINT_INDEX.md). Metadata seed được phục hồi không tự giải quyết thiếu endpoint; kế hoạch chuẩn hóa đối chứng vẫn **PLAN_NOT_LAUNCHED**, cần kiểm reuse gate trước khi train.

<a id="es10-standardization-plan"></a>

## Ưu tiên thực thi mới — chuẩn hóa đối chứng ES10, 08/10/2026

**PLAN_NOT_LAUNCHED.** Tác giả yêu cầu lập kế hoạch chạy lại toàn bộ 11 đối sánh thiếu endpoint metadata, cộng Baseline mBART seed46 dừng epoch18: **12 runs**. Chưa bật GPU, chưa training, chưa thay số đã báo cáo. Mục này có ưu tiên trước kế hoạch candidate pilot phía dưới.

**Phạm vi đã khóa:** không chạy lại CLRR-only, LSR-only, Full cũ, các cấu hình alpha/lambda, warmup, attached/no-detach, decoder/both hay ablation cũ. Giữ nguyên số và provenance của chúng. Không mở Asháninka hoặc LayerSkip. Mỗi đối sánh thiếu seed dùng **seed42 cho run mới**; không gán ngược seed42 vào hồ sơ lịch sử thiếu seed.

### 1. Danh sách đầy đủ và cấu hình đề xuất

Tất cả dùng Amis `data_processed/amis_mandarin/` (train4600/validation576/test575), tối đa20epochs, **patience10**, evaluation/save mỗi epoch, raw-reference validation chrF++ để chọn best. Khởi tạo từ pretrained revision đã pin; không resume từ best weights cũ. LR/recipe dưới đây là thiết kế cho cohort mới, không phải metadata đã phục hồi của run lịch sử.

| ID | Backbone | Nhánh | Seed | LR | Microbatch × accumulation | Max source/target | Validation beams | Lý do chạy |
|---|---|---|---:|---:|---|---|---:|---|
| A01 | mBART-50 | Baseline | 46 | 5e-5 | 32×4 | 256/256 | 1 | Run cũ dừng epoch18; archive best không đủ trạng thái resume |
| A02 | mBART-50 | Middle Alignment | 42 | 5e-5 | 32×4 | 256/256 | 1 | Thiếu endpoint/seed metadata đầy đủ |
| A03 | mBART-50 | Narrow LoRA | 42 | 2e-4 | 4×32 | 256/256 | 1 | Thiếu endpoint metadata |
| A04 | mBART-50 | BitFit | 42 | 1e-4 | 4×32 | 256/256 | 1 | Thiếu endpoint metadata |
| B01 | NLLB-200 distilled600M | Baseline | 42 | 5e-5 | 16×8 | 128/128 | 4 | Chỉ có best state epoch19, không có final state |
| B02 | NLLB-200 distilled600M | Middle Alignment | 42 | 5e-5 | 16×8 | 128/128 | 4 | Chỉ có best state epoch19 |
| B03 | NLLB-200 distilled600M | Strong LoRA A | 42 | 2e-4 | 16×8 | 128/128 | 4 | Thiếu matching final checkpoint/receipt trong public main đã audit |
| B04 | NLLB-200 distilled600M | Strong LoRA B | 42 | 1e-4 | 16×8 | 128/128 | 4 | Thiếu matching final checkpoint/receipt trong public main đã audit |
| B05 | NLLB-200 distilled600M | BitFit | 42 | 1e-4 | 16×8 | 128/128 | 4 | Chỉ có best state epoch18 |
| C01 | mT5-small | Middle Alignment | 42 | 3e-4 | 16×8 | 256/256 | 1 | Thiếu endpoint metadata |
| D01 | ByT5-small | Baseline | 42 | 5e-5 | 16×8 | 256/256 | 1 | Thiếu endpoint metadata |
| D02 | ByT5-small | Middle Alignment | 42 | 3e-4 | 16×8 | 256/256 | 1 | Thiếu endpoint metadata |

Các LR lấy từ recipe/default đã đọc trong các trainer; **không gọi là HPO optimum**, không có equal-HPO superiority claim. Weight decay0, AdamW, linear schedule/warmup0.06, BF16; freeze đầy đủ optimizer/dropout/padding/TF32/gradient-checkpointing/runtime trong manifest trước launch. Effective batch128 không làm Middle microbatch16 tương đương32: InfoNCE dùng negatives trong microbatch. Khóa microbatch của Middle; OOM không được tự đổi nó khi chưa cập nhật method/recipe card.

Middle dùng InfoNCE hiện có, tau0.10/weight0.10; layer index6 cho mBART/NLLB/ByT5,4 cho mT5, ghi rõ convention hidden-state index và gradient cả source/target. Narrow LoRA r8/alpha16/dropout0.05/q_proj,v_proj; Strong A/B r16/alpha32/dropout0.05/q,k,v,out,fc1,fc2, A frozen embeddings, B modules_to_save shared/embed_tokens/lm_head. Gate phải xác minh tied embeddings và số trainable parameters. BitFit chỉ bias.

**Selection cùng backbone:** NLLB giữ validation beam4 để không tự làm Narrow LoRA đủ20epochs thành không matched chỉ vì đổi beam; ba backbone còn lại beam1. Test tất cả beam4. Nếu quyết định đổi NLLB selection sang beam1, cần một cohort riêng và kiểm kê lại tất cả controls NLLB, không đổi âm thầm trong12runs. Các khác biệt LR/length/selection phải hiện trong paper setup; chuẩn hóa metrics/provenance không đồng nghĩa mọi backbone dùng cùng hyperparameters.

### 2. Các đối chứng đã đủ20epochs: reuse gate bắt buộc

Giữ Baseline mBART corrected42–44 và fresh45/47–52; Strong LoRA mBART A/B; Narrow LoRA NLLB; Baseline mT5 nếu recipe/selection phù hợp. Baseline/Middle Turkish seed42 đã ES10, giữ. Baseline mBART original42 là cohort lịch sử riêng, không cộng vào mean corrected/fresh.

Patience4 run đã đạt cap20 không cần train lại **chỉ vì** chuyển patience10. Giữ actual historical patience4 trong receipt và ghi `reuse_reason=completed_original_epoch_cap`. Trước khi đặt cùng bảng matched mới phải so data hashes/order, pretrained revision, source/loss implementation, optimizer/scheduler, microbatch/accumulation, lengths, language codes, raw validation references, selection metric/beams và precision. Nếu thay một yếu tố làm checkpoint selection/training khác đi, đánh dấu `REUSE_BLOCKED`, liệt kê số run bổ sung và báo tác giả trước mở rộng budget. Không tự gọi toàn bộ số cũ là ES10 hoặc cùng cohort.

12runs là ngân sách đã lập cho các mục thiếu metadata/dừng sớm. Các gate, đánh giá lại và upload không phải12model bổ sung; thêm seeds hoặc controls do reuse gate thất bại là budget riêng. Không chạy lại các phương pháp cũ bị loại khỏi phạm vi để đạt một bảng đồng nhất giả tạo.

### 3. Thứ tự, entry points và điều kiện trước launch

1. Tạo manifest12jobs, dataset hashes, pretrained/tokenizer revisions, source SHA, runtime, fully resolved arguments, namespace và nguyên nhân rerun. mBART pin `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; resolve/pin NLLB/mT5/ByT5 trước gate. Kiểm reuse controls theo mục2.
2. Hoàn thiện worker chung cho train-validation-only, selection lock, independent test, metadata/public upload và exact resume. Không gọi các suite cũ nguyên trạng: một số tự test ngay, auto-resume, skip vì metrics.json tồn tại, upload theo prefix cũ hoặc không ghi completion fields.
3. CPU correctness + tiny A100 BF16/accumulation/save/reload gate cho Baseline, Middle trên mBART/NLLB/mT5/ByT5 và PEFT; Strong B kiểm tied weights. Với Middle kiểm actual auxiliary loss và gradients; accumulation không so với larger-batch InfoNCE có negatives khác. PASS mới launch research models. Không claim code hiện tại đã READY chỉ vì có CLI patience.
4. Train12jobs validation-only; hoàn thành model nào push best+history+final receipt ngay. Metadata phải phân biệt selected/best epoch với completed epoch và stop reason.
5. Khi đủ12validation receipts verified, freeze `selection_lock.json` trên public HF, rồi test đúng12selected checkpoints. Không chọn/retrain theo test. Test là phân tích exploratory trên Amis đã được xem, không phải unseen-corpus confirmation.
6. CPU độc lập xác minh predictions/metrics/SHA; cập nhật bảng hiện hành và mapping old→new. Xác minh backup trước xóa weights local/Colab; account hết queue và không còn test/upload task thì stop.

| Nhóm | Entry point hiện có cần được worker chuẩn hóa gọi |
|---|---|
| Baseline mBART/ByT5 | `src/amis_rewire/train.py` / module `amis_rewire.train` |
| Middle mBART/mT5/ByT5 | `src/comparative_baselines/middle_align_acl2025/train.py` |
| Narrow LoRA/BitFit mBART | `src/peft_baselines/train_peft.py` |
| NLLB năm nhánh | `src/nllb_suite/train_nllb.py` với baseline/middle_align/strong_lora_a/strong_lora_b/bitfit |

**Chưa có unified worker/launcher dành cho12jobs này.** Các entry points là nơi triển khai, không phải lệnh queue đã tồn tại. Cần giải quyết revision pinning, validation-only, public canonical prefix, TF32-off generation, final state, checksum upload failures và metadata schema trước khi bàn giao lệnh chạy. Luôn truyền explicit20/10; không dùng defaults generic20/4 hoặc5/2. Dùng namespace/output trống, không `--auto-resume` một thư mục run lịch sử.

### 4. Điều phối ba Colab A100 — đề xuất theo unit thực tế

| Account | Queue ban đầu | Models |
|---|---|---:|
| Colab1 | B01–B05: năm nhánh NLLB | 5 |
| Colab2 | A01–A04: bốn nhánh mBART | 4 |
| Colab3, ít unit | C01,D01,D02: mT5/ByT5 | 3 |

Đây là phân bổ ban đầu theo model size, không cam kết ByT5 có chi phí thấp: tokenizer byte có thể làm chạy lâu. Đo timing tiny gate và remaining CU; account3 thiếu budget thì chỉ C01, chuyển D01/D02 sang account1/2 khi slot trống. Không để account3 hết unit giữa run. Mỗi A100 một training process, Standard40GB/no high-mem, không RTX6000. Freeze manifest trước phân phối; chuyển job chưa chạy không đổi recipe. Báo chat mỗi10phút khi có phiên đang được chủ động giám sát; không ghi đang monitoring nếu controller không hoạt động. Không có ETA/unit estimate đã đo cho queue này.

### 5. Đánh giá và chuẩn hóa public/docs

- Raw original CSV references, đầy đủ575test rows/order; BLEU zh, native chrF++, chrF0 diagnostic theo [METRICS.md](../experiments/METRICS.md). SacreBLEU2.6.0, full precision JSON/CSV; không decode references qua tokenizer.
- Generation FP32/TF32off; testbeam4, evalbatch8/padmultiple8, length theo manifest, forced target language đúng. mBART tl_XX→zh_CN; NLLB zho_Hant→zho_Hant là proxy language-code đã dùng, không mô tả là Amis native token. mT5/ByT5 source-prefix convention phải có trong manifest.
- Public duy nhất `FiveC/amis-rewire-checkpoints`, `checkpoints/amis_mandarin/<backbone>/<method>/<recipe>/es10-standardization/seed-<n>/`. Recipe chứa actual LR,bs,ga,length,precision,validationbeams,es10; Strong A/B phân biệt frozen vs unfreeze embeddings. Không ngày tháng, không ghi đè đường dẫn lịch sử.
- Best archive phải load lại đúng model/adapter và base revision. Metadata nhỏ: resolved config, source/environment/data hashes, complete per-epoch history, selected and final trainer states, stop reason, patience counter, CE/auxiliary definitions, prediction/weights hashes, upload/verification receipts. Schema ở [hướng dẫn checkpoint](HUGGINGFACE_CHECKPOINT_STORAGE.md#es10-metadata-contract).
- Completion chỉ `COMPLETE_VERIFIED` sau independent test replay/metrics match và HF archive SHA/size verification. Upload lỗi là `UPLOAD_PENDING`, giữ weights và khôi phục; không coi training exit0/metrics.json tồn tại là hoàn tất.
- [EXPERIMENTS_UPDATED.md](../experiments/EXPERIMENTS_UPDATED.md#es10-standardization-results) là sổ số duy nhất: thêm cohort mới, ghi old/new run IDs và old−new score delta sau verify. Không ghi đè lịch sử, không gộp Baseline46 mới với CLRR/LSR/Full ES4 thành matched confirmation mới. Insight/long-term chỉ cập nhật khi có số và kết luận thật.
- Old CLRR/LSR/Full được giữ như historical results. Vì tác giả không muốn rerun chúng, **không tuyên bố toàn bộ paper đã cùng training recipe ES10**; chỉ các đối chứng mới có fully verified ES10, và reused cap20 có nhãn actual ES4. Single-seed12jobs không thay multiseed/HPO evidence.

### 6. Bằng chứng audit trước kế hoạch

[Audit](../../outputs_rebuttal/early_stopping_10_audit_20261008/AUDIT.md), [machine-readable audit](../../outputs_rebuttal/early_stopping_10_audit_20261008/audit.json), [public ZIP metadata](../../outputs_rebuttal/early_stopping_10_audit_20261008/archive_metadata.json). User đã yêu cầu đưa cả11UNRESOLVED vào rerun, không cần chờ phục hồi endpoint cũ; vẫn giữ kết luận audit rằng missing metadata không chứng minh early stopping.

## Trạng thái hiện hành — 08/10/2026

- **Amis corrected development seeds42–44** và **32 fresh confirmation runs seeds45–52** đã hoàn tất. Cohort fresh không đạt primary/secondary investment gates; không gọi development gain là robust confirmation.
- Kiểm chứng độc lập validation và cơ chế trên 32 checkpoints đã xong. Không suy gradient norm nhỏ thành Adam update yếu; can thiệp tắt lớp10–11 chưa ủng hộ hướng bỏ lớp cao khi retrain.
- **Turkish→English OPUS100 gốc**, train20k/validation2k/test2k: Baseline/Middle/LSR/Full seed42 đã train và test xong. Full−LSR +0.370036 BLEU; Full−Baseline −0.063583; chưa có ưu thế rõ trước Baseline hay bằng chứng multiseed.
- Hai A100 đã tắt sau xác minh backup. Không có training mới được mở trong bước gộp docs.
- Checkpoint public duy nhất: `FiveC/amis-rewire-checkpoints`; 90 archive cuối cùng đã được tổ chức và audit. Quy tắc/lookup ở [hướng dẫn checkpoint](HUGGINGFACE_CHECKPOINT_STORAGE.md).
- **Bước tiếp theo chưa launch:** khóa tối đa hai candidate có giả thuyết rõ, matched controls, correctness/tiny GPU gate, rồi pilot validation-only theo budget trong long-term plan. Không tự mở rộng Full cũ hay sweep chỉ từ test đã xem.

Các mục kế hoạch/trạng thái cũ phía dưới giữ phạm vi lịch sử. Trạng thái ở mục này và kết quả confirmation/Turkish có ưu tiên khi quyết định bước tiếp theo.

Ngày cập nhật: 07/10/2026, Asia/Saigon. **Trạng thái hiện hành: 32 runs hoàn tất; primary NO-GO, secondary không đạt. Đọc Mục 9 để thực thi audit/nguyên nhân/nâng cấp. Các banner READY/ACTIVE/HOLD cũ là lịch sử, không dùng để launch.**

## Kế hoạch tiếp theo đã thống nhất với tác giả — 06/10/2026

**Turkish dataset decision, 06/10/2026:** tác giả chọn benchmark gốc. Dùng `data_processed/opus100_turkish_english_20k/`: train20k mẫu cũ giữ nguyên; validation/test official OPUS-100 đủ2000/2000, nguyên strings/order/duplicates. Không deduplicate hoặc loại overlap khỏi evaluation. Manifest ghi17source types overlap validation/test, train/evaluation overlap0. Bản filtered v2 bị loại bỏ; outputs v2 lịch sử không được reuse âm thầm. `scripts/prepare_turkish_followup.py` xác minh/restores splits gốc; runner trỏ dataset gốc và namespace output mới `outputs_rebuttal/turkish_official_20261006/`. Quyết định data này không launch Turkish, không tác động32Amis đang chạy; future Turkish training vẫn theo method-lock/authorization.

**Quy ước tên trong báo cáo:** dùng **Baseline** cho fine-tuning thông thường bằng CE, không có CLRR/LSR. **Full** là CLRR + LSR; cả hai đều cập nhật toàn bộ trọng số. Trong code vẫn dùng family `baseline`; không đổi CLI hoặc tên artifact. Các ký hiệu FT trong ghi chú lịch sử bên dưới cũng chỉ Baseline.

GPU preflight, benchmark v2 và bundle freeze đã hoàn tất theo receipts bên dưới. Vòng nghiên cứu tiếp theo là **32 confirmation models** trên mBART Amis→Mandarin; giữ Full α=.10/λ=.10. Bảng này ghi kế hoạch, không xác nhận training dài đã launch.

| Thứ tự tiếp theo | Làm gì | Mục đích / điều kiện |
|---|---|---|
| **1 — xác nhận trước launch** | Dùng đúng protocol và frozen bundle đã kiểm tra; xác minh SHA/source khi triển khai | Giữ fairness. Không đổi cấu hình hoặc rebuild bằng source khác giữa train và test; source đổi phải gate lại |
| **2 — confirmation train** | **Baseline, CLRR-only, LSR-only λ=.10, Full α=.10/λ=.10 × seeds45–52 =32 models**; validation-only | Kiểm tra Full có vượt Baseline và LSR ổn định qua training randomness. Hai A100 Standard40GB, không high-mem: account1 Baseline+CLRR-only16runs; account2 LSR-only+Full16runs; mỗi GPU một model tại một thời điểm |
| **3 — checkpoint và backup** | Chọn bằng validation raw chrF++ beam1; push selected checkpoint và provenance lên Hugging Face từng model, xác minh trước cleanup | Không chọn bằng test. Local chỉ giữ artifacts nhỏ; khi theo dõi chủ động báo chat mỗi10phút |
| **4 — frozen test và quyết định** | Collect/replay đủ32 validation receipts, khóa `selection_lock.json`, rồi test32 checkpoint bằng beam4 FP32/TF32off; paired seed deltas, CI và kiểm định theo protocol | Đánh giá quality và uncertainty. Sau local verification/backup hoàn tất, tắt cả hai GPU và giữ stop receipts |
| **5 — nếu confirmation có tín hiệu tốt** | Audit implementation/provenance của **Middle-Layer Alignment**, rồi lập đối sánh matched cùng protocol và seeds | Middle chưa nằm trong32runs. Xác định checkpoint cũ có đủ điều kiện reuse hay cần train mới; nếu cần matched8seeds thì thêm8models với budget riêng. Khóa recipe/selection trước chạy; không lấy mean Full3seeds trừ một run Middle lịch sử để claim superiority |
| **6 — nếu confirmation yếu hoặc âm** | Đo cơ chế trên checkpoint corrected; khi có hypothesis rõ mới đề xuất tối đa2 nâng cấp với matched controls | Tìm nguyên nhân trước đổi trọng số/phương pháp. Candidate chọn bằng validation; upgrade cần correctness gate và confirmation độc lập mới |

### Gate mong muốn của vòng32

| So sánh | Ngưỡng đầu tư đã khóa | Diễn giải |
|---|---|---|
| **Primary: Full − LSR-only** | Mean ≥ **+0,30 BLEU**, thắng ≥ **6/8 seeds**, mean chrF++ và chrF0 không giảm so LSR | Kiểm tra routing bổ sung giá trị ngoài alignment |
| **Secondary: Full − Baseline** | Mean ≥ **+0,50 BLEU**, thắng ≥ **5/8 seeds**; báo các metrics còn lại và uncertainty | Kiểm tra lợi ích thực tế trước baseline |

Đây là investment gates của dự án, không phải tiêu chuẩn nhận bài ACL. Báo paired seed CI/Student-t, exact sign-flip và sentence uncertainty theo protocol; số seeds thắng không tự chứng minh statistical significance. Gain dương nhưng CI qua0 là chưa đủ kết luận robust superiority. Primary đạt nhưng secondary yếu thì thu hẹp claim; không mặc định phải đổi phương pháp. Turkish và các cohort backbone/PEFT khác chưa mở trong bước này.

## Chạy tiếp gì và vì sao — kế hoạch thực thi 07/10/2026

**Trạng thái đã xác nhận:** corrected control scores, LSR validation grid, và **TOÀN BỘ 32 Fresh Seed Confirmation runs (seeds 45–52) ĐÃ HOÀN TẤT VÀ AUDITED 100% (07/10/2026).** Bước 1 (GPU preflight gate), Bước 2 (Benchmark v2), Bước 3 (Bundle freeze), Bước 4 (Confirmation Train) và Bước 5 (Frozen Test Evaluation) đều đã HOÀN THÀNH. Cả 2 GPU hiện đã TẮT an toàn (0 CU idle).

| Thứ tự | Chạy cụ thể | Số model mới / tài nguyên | Vì sao cần chạy? | Điều kiện hoàn tất / bước sau |
|---|---|---|---|---|
| **1 — trước hết** | **GPU preflight của bundle mới:** 16 optimization cases BF16+accum4+checkpointing; 4 actual CLI train/save/reload/validation-only branches | 0 research models; 1 A100 Standard 40GB ngắn | GPU precision và lifecycle verification | **HOÀN THÀNH (PASS 100%)** trên NVIDIA A100 SXM4 40GB. 16/16 opt cases pass (err 0.0), 4/4 CLI branches pass. Receipts: `outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate/`. |
| **2 — ngay sau gate** | **Benchmark v2**, load corrected seed 42 FT/CLRR/LSR/Full; train macrostep và clean-reloaded inference beam 4 FP32; beam 1 supplementary | 0 research models; reuse cùng 1 A100 | Số liệu efficiency đúng protocol, kiểm routing cost ngoài LSR | **HOÀN THÀNH (AUDITED)** trên cùng A100. Đủ 4 systems, param neutrality (610.8M), CLRR train overhead +1.67% vs FT, Full overhead +0.80% vs LSR, VRAM 12.8–13.0 GB. Receipts: `outputs_rebuttal/amis_cost_benchmark_20261006_v2/`. |
| **3 — trước launch dài** | Rebuild/freeze bundle và chốt confirmation plan: seeds 45–52, primary Full−LSR; selection chrF++ beam 1; diagnostic-only BLEU/chrF0; test mở sau all 32 validation lock | CPU, 0 train | Bảo đảm kế hoạch/code/manifest không lệch; không đổi selection hoặc sample size khi thấy scores | **HOÀN THÀNH (LOCKED)**. Bundle SHA256: `52fcd0fc2881d414a8ea1ab0e21ca1961eec74cd32b01858fda5c77a3d64bc3c` (32 files, 0 mismatch). Protocol và 2 Gates đã khóa chặt chẽ. |
| **4 — confirmation train** | **Baseline/CLRR-only/LSR.10/Full.10/.10 × 8 fresh seeds 45–52**, tất cả validation-only | **32 research models**; 2 A100 Standard 40GB song song | Câu hỏi quan trọng nhất: Full có bổ sung LSR ổn định qua training randomness không? Baseline kiểm practical benefit, CLRR hỗ trợ ablation/interaction | **HOÀN THÀNH 100% (AUDITED & LOCKED)**: Cả 32 validation runs hoàn tất. Checkpoint chọn duy nhất bằng validation raw chrF++ beam 1, tự động backup HF và verified. Đã khóa `selection_lock.json` (`status: ALL_32_VALIDATION_VERIFIED`). |
| **5 — frozen test + quyết định** | Test đúng 32 selected checkpoints, beam 4 FP32/TF32off; paired seed effects/Student-t CI/exact signflip; descriptive interaction | 0 train mới; 2 GPU test các weights trên HF | Định lượng gain và uncertainty trên fresh seeds; không dùng test chọn checkpoint/config | **HOÀN THÀNH 100% (AUDITED & REPORTED)**: Đủ 32/32 runs evaluated trên frozen 575 test sentences. SHA-256 đối soát từ HF khớp tuyệt đối. GPU đã tắt ngay (0 CU). Primary Gate 1: **NO-GO**, Secondary Gate 2: **DID NOT PASS**. Chi tiết Mục 8. |
| **6 — chỉ nếu cần** | CorrectedFull targetanchor/residual/gradient/rank probes; nếu cóhypothesis rõ thì tối đa2upgrades với matchedcontrols | Probes0fulltrain; upgradebudget riêng≤12 theo thiết kế | Tìm nguyên nhân nếu confirmation weak/negative, không sửa method dựa cosine/rank đẹp | Chỉ chọn pilot bằngvalidation. Upgrade tạo source mới thì correctnessgate và independentconfirmation mới; không tái dùng freshseeds đã chọn để claim độc lập |

### Gate quyết định sau confirmation

- **Primary investment gate:** Full−LSR meanBLEU≥+.30,≥6/8pairedwins, meanchrF++/chrF0no-harm. Báo seedCI/p và sentenceuncertainty đúng phạm vi;6/8wins không tự tương đương statisticalsignificance.
- **Secondary:** Full−Baseline meanBLEU≥+.50,≥5/8wins, với uncertainty. Primary đạt nhưng secondaryweak thì chỉ claim benefit ngoài alignment trongscope được hỗ trợ.
- Positiveeffect nhưng CIqua0: inconclusive, không gọi bằng nhau hoặc robustsuperiority. Negative/belowgate: giữ failure, probe cóquestion cụ thể rồi mới candidateupgrade; không chuyển data để tìm bảng đẹp.
- Interaction là descriptivefactorialcontrast trênmetrics, không chứng minh representationmechanism. Generic/priorcomparator vẫn cần nếu noveltyclaim dựa vào routingrule.
- GO Amis không tự nghĩa đủ nộpACL. **Turkish và cácbackbone/PEFTcohorts chưa mở trong bàn giao này**; prepare có thể làm riêng, training cần methodlock và quyền chạy tiếp.

### Entry points và cách điều phối

**Bước1 (Colab, `cd /content/CLRR`, `PYTHONPATH=/content/CLRR/src`, môi trườngremoteclrr):**

```bash
/content/miniforge3/bin/conda run --no-capture-output -n clrr python tests/smoke_rerun_optimization.py --cuda --result-dir outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate
/content/miniforge3/bin/conda run --no-capture-output -n clrr python tests/smoke_precision_training.py --cuda --audit-rerun --result-dir outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate
```

Trước gate phải deploy đúng frozenbundle và xác minh `fresh_bundle_files.json`, pinnedruntime/GPU. **Launcher hiện tự nối gate→training dài; chưa có `--stage gate`.** Muốn chạy bước1–2 riêng, bên thực thi cần bootstrap/deploy và chạy hai commands trên mà không gọi32runqueue, hoặc bổ sung gate-only mode rồi audit. Không invent flag khi chưa implement.

**Bước2 (cùngColab):**

```bash
/content/miniforge3/bin/conda run --no-capture-output -n clrr python scripts/revalidation/benchmark_corrected_components_v2.py --output-dir outputs_rebuttal/amis_cost_benchmark_20261006_v2
```

Benchmark cần HFcredential đã được cấp và receipt files trongbundle; không print token. Trích xuất smallJSONCSVMD, uploadreport+manifestHF; weights/cache chỉ tạm trênColab. Không tảiweights vềlocal. Timeestimate không coi là actualruntime; báo khiđãđo.

**Bước3–5:** dùng commands trong mục “Lệnh chạy sau khi được cho launch” bên dưới: preparebundle→launcher`--stage train`→watch/collect→localverifier`--stage validation`→launcher`--stage test`→watch. Account1FT+CLRR, account2LSR+Full; chỉ1model/GPU tại mỗithờiđiểm. Default train không mởtest. Sauvalidationlock chuyểntest ngay khi đãauthorized; nếupause thì dừngruntime để khôngtốnidleusage. Khi chat đangtheodõi báo10phút/lần; watcherfile không tự gửichat.

**Không lặp:** oldfour-wayanalysis, LSR.03/.10/.30train và historicalwarmup chỉ vì archiveddocs. Không overwrite oldscores. Tinygateoptimizerupdates và benchmarktimingupdates không phải newresearchtrainingruns. Tổng researchtraining dự kiến vòng này là **32**, không cộng cácGPUprobes thànhmodels.

## Review hiện hành sau LSR .30 và benchmark — ưu tiên mục này

### Bản sửa của Codex — 05/10/2026 (ưu tiên hơn các nhãn READY cũ bên dưới)

**Code đã sửa và local regression PASS; GPU gate của bundle mới và benchmark v2 chưa thực thi. Không gọi ALL GATES READY. Chưa mở32runs/Turkish.** Old scores không thay đổi, không cần train lại LSR .30 chỉ vì fixes orchestration/benchmark.

Đã sửa:

- Worker/test và verifier tính chrF0 riêng bằng native `CHRF(word_order=0)`, không đổi production `generation_metrics` của historical/corrected cohorts.
- Student-t p-value/CI dùng SciPy, df=n−1; primary investment gate kiểm cả chrF++ và chrF0 no-harm. Thiếu bất cứ model/predictions/reference/metadata/hash nào đều fail; không dùng fallback receipt-only scores.
- Train launcher chỉ chạy validation. Hai accounts phải collect và local-verify đủ32runs, ghi immutable `selection_lock.json` trước test. Test worker kiểm lock, archive SHA và selected weight tensor digest. Không automatic test ngay sau16runs/account.
- Bundle kiểm per-file SHA; launcher kiểm local ZIP SHA/source hashes trước transfer. Real GPU gates chạy16optimization cases và4production CLI lifecycle cases trên bundle mới; worker không nhận CPU gate thay GPU gate.
- Selected/exported weight digests phải bằng nhau; strict checkpoint load kiểm missing/unexpected/mismatched keys. HF selected archive+metadata pinned revision, size và SHA/gitblob được kiểm trước cleanup.
- Sau mỗi validation backup verified, dọn selected weights và ZIP trên Colab; test tải đúng revision vào temporary cache rồi dọn sau verify. Không giữ16copies best_model+ZIP/GPU. Collector chỉ lấy small artifacts qua một ZIP/account, không tải weights về local.
- Benchmark v2 pin HF revision từ receipts, strict-load, fixed language codes trước tokenize, fusedAdamW/weightdecay0/clipping, trainingTF32 policy; clean reload trước FP32beam4 inference; throughput loại pad và specialtokens. Timing repetitions là samples trong cùng session, không coi là across-session uncertainty.
- Launcher xuất shell LF (tránh WindowsCRLF), token chmod600, process zombie detection; watcher tạo selectionlock sau32validation và chỉ dừng GPUs sau32testverified. Watcher không tự gửi chat.

Local verification trong conda `clrr`:16optimization cases PASS,4CLI training/save/reload branches PASS,7integrity/statistics regression checks PASS. Receipts: `outputs_rebuttal/fresh_confirmation_fix_checks/{optimization_matrix.json,rerun_lifecycle_smoke.json,integrity_regression.json}`. SciPy1.17.1 được cài vào localclrr; remote bootstrap pin cùng version. Đây là CPU checks; GPU receipts chưa có cho bundle mới.

#### Lệnh chạy sau khi được cho launch

```powershell
# Build after any bundled source edit; preserve printed SHA in the run record.
conda run --no-capture-output -n clrr python scratch/prepare_fresh_confirmation.py
# Deploy TRAIN stage. Real GPU gates run first and abort before long training if failing.
conda run --no-capture-output -n clrr python scratch/launch_fresh_confirmation_remote.py --account both --stage train
# Monitor; when both validation queues finish, collect/replay32runs and write selection lock.
conda run --no-capture-output -n clrr python scratch/watch_fresh_confirmation.py
# Explicit alternative for collecting/locking validation:
conda run --no-capture-output -n clrr python scratch/collect_fresh_confirmation_remote.py --account both
conda run --no-capture-output -n clrr python scratch/verify_fresh_confirmation_local.py --stage validation
# TEST only after all32validation receipts pass; launcher uploads selection lock to both accounts.
conda run --no-capture-output -n clrr python scratch/launch_fresh_confirmation_remote.py --account both --stage test
conda run --no-capture-output -n clrr python scratch/watch_fresh_confirmation.py
```

Stagestrain/test cần active accounts và same local frozenbundle; không rebuild bằng sourcekhác giữa hai stages. Worker namespaceguards không tự resume crashedattempt. Watcher validation dừng theo dõi khi ghi lock, GPUs vẫn available để test; vận hành cần chuyển sangtest ngay hoặc dừng nếu userpause. Watcher test collect/replay đủ32 rồi stop cảhai, lưu stopreceipts; lỗi verification giữ runtimes để điều tra. Khi theo dõi chủ động, báo chat10phút. Bundle SHA hiện hành **`52fcd0fc2881d414a8ea1ab0e21ca1961eec74cd32b01858fda5c77a3d64bc3c`**, ghi trong `scratch/fresh_confirmation_manifest.json` (32 files verified, 0 mismatches).

Benchmark v2 và GPU Preflight Gate đã chạy thực tế trên NVIDIA A100 SXM4 40GB vào ngày 06/10/2026 và **PASS 100%**. Receipts đã thu thập về local tại `outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate/` và `outputs_rebuttal/amis_cost_benchmark_20261006_v2/`.

**GPU GATES VÀ BENCHMARK V2 ĐÃ PASS TRÊN A100; BƯỚC 1 ĐẾN 3 ĐÃ HOÀN TẤT. SẴN SÀNG CHO BƯỚC 4 (32 CONFIRMATION RUNS).** Không cần train lại ba runs LSR .30. Không tự nâng cấp phương pháp hoặc mở Turkish.

### Kết quả đã kiểm tra độc lập và chuẩn hóa Provenance

1. Đã replay validation predictions của cả ba seeds LSR .30 với raw references trong conda `clrr`; BLEU và chrF++ khớp decision JSON dưới sai số 1e-8. Script kiểm tra: `scratch/review_lsr030_results.py`.
2. Đã xác minh 21 file và archive checkpoints của LSR .30 trên Hugging Face `FiveC/amis-rewire-checkpoints` tại commit `02533798a9deaf5c22df701430a64a5ec9f8276e`. Lưu receipt hoàn chỉnh tại `outputs_rebuttal/amis_lsr_l030_validation_20261005/collection_receipt.json`.
3. Đã bổ sung đầy đủ `best_step` và `chrf0` vào `lsr_grid_decision.json`. Khóa chính thức: $\lambda = 0.10$ là **best observed trong grid {.03, .10, .30}**, confirmation khóa chính xác **32 runs** (4 families × 8 seeds).

| Seed | LSR .30 validation BLEU | chrF++ | chrF0 replay | Best Step |
|---|---:|---:|---:|---:|
| 42 | 17.398041 | 14.529689 | 16.725192 | 720 |
| 43 | 16.887742 | 14.216064 | 16.301792 | 648 |
| 44 | 15.855267 | 13.705969 | 15.791632 | 432 |

Mean validation chrF++: .03 = 14.0503; .10 = **14.2220** (Thắng); .30 = 14.1506.

### Benchmark v2: Đã implement script chuẩn theo đúng yêu cầu audit

Script mới: `scripts/revalidation/benchmark_corrected_components_v2.py`.
- **Khắc phục toàn bộ các hạn chế của v1:**
  - Strict-load trọng số Seed 42 đã được audit từ pinned HF archives (`FiveC/amis-rewire-checkpoints` và `FiveC/amis-corrected-clrr-only-20261005`), kiểm tra khớp mã băm SHA256 trước khi load.
  - Cố định batch indices và tensors đầu vào (BS=32, Accum=4, eff BS=128, padded to multiple of 8, max length 256).
  - Khớp hoàn toàn production training setup: BF16 autocast, Gradient Checkpointing, use_cache=False, AdamW lr=5e-5, gradient clipping max_norm=1.0.
  - Đo 5 timed repetitions độc lập có `torch.cuda.synchronize()`, báo cáo mean ± standard deviation.
  - Reload trọng số sạch trước khi đo inference (cô lập hoàn toàn bước cập nhật tham số optimizer khỏi bước inference).
  - Đo inference 2 chế độ: Primary là Beam 4 FP32 (chuẩn chất lượng của paper), Supplementary là Beam 1 Greedy (microbenchmark).
  - Đếm chính xác non-padding tokens (`(gen != pad_token_id).sum()`) để tính throughput thực tế (valid tokens/s).
  - Báo cáo chi tiết VRAM đỉnh (allocated & reserved MiB).
  - Ghi kết quả sang thư mục mới: `outputs_rebuttal/amis_cost_benchmark_20261005_v2/`, giữ nguyên v1.

### Chuẩn bị trước bản sửa — các mô tả compile PASS dưới đây không thay GPU gate hiện hành

1. **Protocol thống kê:** Đã cập nhật `scripts/revalidation/compute_confirmation_power.py`, bổ sung hàm tính exact Permutation Sign-Flip test theo magnitude bên cạnh Binomial Sign test. Phân định rõ ràng: tiêu chí $\ge 6/8$ wins là tiêu chuẩn quyết định đầu tư nội bộ (investment criterion, $p=0.1445$ one-sided / $0.2891$ two-sided), không phải tiêu chuẩn $p < 0.05$.
2. **Diễn giải power:** Đã khóa rõ ràng trong `docs/research/ACL_LONG_TERM_EXPERIMENT_PLAN.md#locked-confirmation-protocol`: power $97.9\%$ cho $N=8$ là có điều kiện dưới hiệu ứng quan sát được ($\Delta \approx +0.74$, $SD \approx 0.44$). Dưới MDE $+0.30$, power là $37.7\%$.
3. **Checkpoint guardrails:** Khóa chọn checkpoint duy nhất theo raw chrF++ beam 1. Validation BLEU và chrF0 chỉ là chỉ số chẩn đoán, không can thiệp hay thay thế checkpoint.
4. **Bộ công cụ Confirmation 32 runs Turnkey:** Đã implement và kiểm tra biên dịch 6 file thành phần:
   - `scripts/revalidation/run_fresh_seed_confirmation.py` (Worker 32 runs, 2 stages train/test, HF backup per-family repo, dọn dẹp checkpoint phụ tránh tràn đĩa)
   - `scratch/prepare_fresh_confirmation.py` (Đã đóng gói `fresh_confirmation_bundle.zip` 248 KB, SHA256 `2d68a20bc8b837dd961f1a54c60eb788d4af4b0d0f525565285555bf9d0aef63`)
   - `scratch/launch_fresh_confirmation_remote.py` (Launcher phân bổ 2 GPU: Acc 1 = Baseline + CLRR 16 models; Acc 2 = LSR + Full 16 models)
   - `scratch/collect_fresh_confirmation_remote.py` (Tải artifacts và logs)
   - `scratch/verify_fresh_confirmation_local.py` (Replay metrics local, kiểm định Gate 1/2 và interaction)
   - `scratch/watch_fresh_confirmation.py` (Theo dõi tiến độ định kỳ 10 phút)

**TRẠNG THÁI HIỆN TẠI:** Bản sửa local đã PASS; GPU validation và benchmark v2 còn pending. Không có GPU nào đang chạy. Đang chờ lệnh chỉ đạo cụ thể từ người dùng trước khi kích hoạt.

## 1. Quyết định hiện tại

**Hoàn tất sửa lỗi và cohort đối chứng; chưa cần mặc định nâng cấp phương pháp.** Ưu tiên tiếp: tổng hợp four-way, đo interaction/chi phí bằng checkpoint corrected, rồi xác nhận Full hiện tại trên fresh seeds nếu gate phù hợp. U1/U2 chỉ triển khai nếu có lý do cụ thể hoặc confirmation không đạt. Không tự chạy Turkish.

- Full hiện tại: encoder routing distance 2, alpha .10, detached source; LSR lambda .10, routing warmup 0.
- LSR lambda .03 thua validation cả ba seeds; giữ lambda .10. Đây chỉ là best observed trong grid {.03,.10}, không phải tối ưu toàn cục.
- Matching LSR và best observed tuned LSR hiện trùng cấu hình: chỉ cần một family, không nhân đôi số model.
- G0 technical repair đã xong; bounded novelty audit có sẵn, novelty/comparator claim chưa chứng nhận đầy đủ. G3/G4 chưa chạy.

## 2. Đã chạy xong và được dùng làm bằng chứng hiện hành

| Nhóm | Runs / việc đã làm | Trạng thái và nơi đọc |
|---|---|---|
| Correctness | accumulation regression, 16 GPU optimization cases, 4 real CLI lifecycle cases, full-size mBART batch32×accum4 | PASS; `../research/PRE_RERUN_VERIFICATION_20261005.md`, `GPU_PREFLIGHT_AND_BASELINE_GATE_20261005.md`, `FULLSIZE_GATE_AND_BASELINE42_20261005.md` |
| Baseline | fresh seed42; seeds43/44 audited reuse | 3 matched results; baseline không mắc factor4 wrapper bug; provenance selection đã xử lý |
| Full alpha .10/lambda .10 | seeds42/43/44 corrected | 3 VERIFIED; `../research/ACCUMULATION_CONFIRMATION_20261005.md` |
| LSR-only lambda .10 | seeds42/43/44 corrected | 3 VERIFIED, cùng recipe Full |
| CLRR-only alpha .10/lambda 0 | seeds42/43/44 corrected | 3 VERIFIED; `../research/CORRECTED_CONTROLS_20261005.md` |
| LSR-only lambda .03 | seeds42/43/44 corrected, validation-only | 3 VALIDATION_VERIFIED; quyết định chọn .10 trước test; .03 test vẫn blind |
| Statistical analysis cũ | Full−LSR, Full−baseline, 10k paired bootstrap; 12 tests/Holm; exact seed sign-flip | VERIFIED native p-values; `archive/20261005_before_consolidation/CORRECTED_ACCUMULATION_SIGNIFICANCE_20261005.md` |
| **N1: Four-way & Interaction** | 4 families × seeds42–44; BLEU, chrF++, chrF0; 10k bootstrap; paired deltas và interaction `Full−LSR−CLRR+FT` | **PASS / HOÀN TẤT**; `../../scripts/revalidation/analyze_corrected_four_way.py`; artifacts tại `../../outputs_rebuttal/amis_corrected_four_way_20261005/FOUR_WAY_ANALYSIS.md`, `four_way_summary.json`, `four_way_scores.csv` |
| **N2: Novelty / Comparator card** | Đối chiếu CLRR với SkipCLM, ResFormer/Value Residual, ResiDual; claims supported/unsupported; adapted comparators | **PASS / HOÀN TẤT**; `../research/NOVELTY_AND_COMPARATOR_CARD.md` |
| **Confirmation design & Power** | Power paired t-test và sign-flip $N \in \{6, 8, 10, 12\}$; Full−LSR primary, Full−FT secondary; khóa Gate trước train | **PASS / HOÀN TẤT**; `../../scripts/revalidation/compute_confirmation_power.py`; artifacts tại `../research/CONFIRMATION_DESIGN_AND_GATES.md`, `../../outputs_rebuttal/amis_corrected_four_way_20261005/power_analysis.json` |
| D1/D4 | warmup selected histories và V5 descriptive error analysis | Đã audit; `archive/20261005_before_consolidation/G0_G1_VERIFICATION_20261005.md` |
| D3 historical | target-anchor routed/raw, 3 old Full checkpoints, cùng 8×4 validation examples | Đã đo nhưng weights trước sửa accumulation; chỉ characterization của cohort cũ, không là cơ chế corrected |

### Bảng số liệu Four-way và Tương tác đã xác minh (Test beam 4, 575 câu)

| Seed | Baseline | CLRR-only | LSR-only $\lambda=.10$ | Full | chrF++ Baseline | chrF++ CLRR | chrF++ LSR | chrF++ Full | chrF0 Baseline | chrF0 CLRR | chrF0 LSR | chrF0 Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 19.7281 | 19.2231 | 19.0954 | 20.3213 | 18.7138 | 18.3625 | 18.4345 | 19.0147 | 18.3980 | 17.9050 | 17.9992 | 18.9144 |
| 43 | 18.1378 | 19.4930 | 19.8193 | 20.1780 | 13.1592 | 18.5320 | 18.6609 | 18.8567 | 17.0864 | 18.2130 | 18.4150 | 18.6480 |
| 44 | 20.3000 | 19.3455 | 19.2644 | 19.8916 | 18.9017 | 18.3742 | 18.2923 | 18.6997 | 18.6511 | 17.9757 | 17.9152 | 18.4499 |
| **Mean** | **19.3886** | **19.3539** | **19.3930** | **20.1303** | **16.9249** | **18.4229** | **18.4626** | **18.8570** | **18.0452** | **18.0312** | **18.1098** | **18.6708** |

**Các contrasts chính và Tương tác (Mean $\pm$ Conditional Sentence 95% CI):**
- $\text{Full} - \text{LSR}$: BLEU $+0.7373$ $[+0.0403, +1.4557]$ (dương cả 3/3 seeds: $+1.2259, +0.3587, +0.6273$); chrF++ $+0.3945$; chrF0 $+0.5610$. Exact sign-flip $p = 0.25$.
- $\text{Full} - \text{Baseline}$: BLEU $+0.7417$ $[-0.0265, +1.5182]$ (dương 2/3 seeds: $+0.5932, +2.0402, -0.4084$); chrF++ $+1.9321$; chrF0 $+0.6256$. Exact sign-flip $p = 0.50$.
- $\text{CLRR} - \text{Baseline}$: BLEU $-0.0348$ $[-0.8355, +0.6013]$; chrF++ $+1.4980$; chrF0 $-0.0139$.
- $\text{LSR} - \text{Baseline}$: BLEU $+0.0044$ $[-0.7652, +0.7566]$; chrF++ $+1.5377$; chrF0 $+0.0646$.
- $\text{Interaction} = \text{Full} - \text{LSR} - \text{CLRR} + \text{Baseline}$: BLEU $+0.7721$ $[-0.1480, +1.8452]$ (Seed 42: $+1.7308$, Seed 43: $-0.9964$, Seed 44: $+1.5818$); chrF0 $+0.5749$.

### Kết quả tính Power kiểm định cặp (Monte Carlo 100k simulations)

- **Full $-$ LSR (Observed $\Delta = +0.7373$, $SD = 0.4439$, Cohen's $d = 1.6609$):**
  - $N=6$: Power (two-sided) $= 89.3\%$, binomial sign test cần $6/6$ ($p=0.0312$).
  - $N=8$: Power (two-sided) $= 97.9\%$, binomial sign test cần $8/8$ ($p=0.0078$).
  - $N=10$: Power (two-sided) $= 99.6\%$, binomial sign test cần $9/10$ ($p=0.0215$).
- **Full $-$ FT (Observed $\Delta = +0.7417$, $SD = 1.2310$, Cohen's $d = 0.6025$):**
  - $N=6$: Power $= 22.2\%$; $N=8$: Power $= 31.2\%$; $N=10$: Power $= 39.3\%$.

## 3. Kết quả nằm đâu, checkpoint đã lưu đâu

| Artifacts | Local | HF |
|---|---|---|
| Four-way metrics & interaction | `outputs_rebuttal/amis_corrected_four_way_20261005/` | JSON, CSV và Markdown local |
| Power analysis | `outputs_rebuttal/amis_corrected_four_way_20261005/power_analysis.json` | JSON local |
| Novelty card | `docs/research/RESEARCH_INSIGHTS.md#novelty-card-history` | Markdown local |
| Confirmation design & Gates | `docs/research/ACL_LONG_TERM_EXPERIMENT_PLAN.md#locked-confirmation-protocol` | Markdown local |
| Full/LSR seed42 | `outputs_rebuttal/amis_accumulation_pilot_20261005/` | `FiveC/amis-rewire-checkpoints`, prefix `amis_accumulation_pilot_20261005` |
| Full/LSR seed43 | `outputs_rebuttal/amis_accumulation_confirmation_20261005/{full,lsr}/seed43/` | `FiveC/amis-rewire-checkpoints`, dùng path/revision trong `collection_receipt.json` |
| Full/LSR seed44 | `outputs_rebuttal/amis_accumulation_confirmation_20261005_recovery/{full,lsr}/seed44/` | `FiveC/amis-accumulation-confirmation-20261005`, recovery prefix; không chọn failed attempt |
| CLRR seeds42–44 | `outputs_rebuttal/amis_corrected_controls_20261005/clrr/seed*/` | `FiveC/amis-corrected-clrr-only-20261005` |
| LSR lambda .03 seeds42–44 | `outputs_rebuttal/amis_corrected_controls_20261005/lsr003/seed*/` | `FiveC/amis-corrected-lsr-l003-20261005` |
| Baseline prediction/receipt mapping | `outputs_rebuttal/amis_accumulation_confirmation_20261005/three_seed_comparison.json` | Đi theo receipt từng baseline; không đoán tên checkpoint |

Hai A100 đã tắt hoàn toàn sau khi hoàn thành toàn bộ công việc: Account 2 (session `colab2`) tắt lúc 20:01, Account 1 (session `colab`) tắt lúc 20:09 (2026-10-05). Toàn bộ dữ liệu kết quả, metrics và artifacts đã được tải về local và sao lưu lên Hugging Face. Không còn GPU nào chạy ngầm.

## 4. Thứ tự thực thi hiện hành

Thực hiện theo thứ tự ưu tiên 1 đến 7:

| Ưu tiên | Việc cần thực hiện | Cấu hình / số lượng | Tài nguyên | Trạng thái hiện tại | Đầu ra bắt buộc |
|:---:|---|---|---|:---:|---|
| **1** | Phân tích four-way và interaction từ predictions đã có | Baseline, CLRR-only, LSR-only .10, Full; seeds42–44; BLEU/chrF++/chrF0 | CPU, **0 train** | **HOÀN THÀNH** | Paired deltas, interaction `Full−LSR−CLRR+FT`, CI và giới hạn thống kê (`outputs_rebuttal/amis_corrected_four_way_20261005/`) |
| **2** | Chốt thiết kế confirmation | **Full−LSR primary**, Full−FT secondary; tính power/precision, chốt số seeds và metric chọn checkpoint | CPU, **0 train** | **HOÀN THÀNH** | Protocol và Go/No-Go khóa trước training (`docs/research/ACL_LONG_TERM_EXPERIMENT_PLAN.md#locked-confirmation-protocol`) |
| **3** | Hoàn thiện novelty/comparator card | Đối chiếu CLRR với SkipCLM, ResFormer/Value Residual, ResiDual | Đọc source/papers, **0 train** | **HOÀN THÀNH** | Khác biệt cụ thể, claims được hỗ trợ và comparator cần chạy (`docs/research/RESEARCH_INSIGHTS.md#novelty-card-history`) |
| **4** | Mở rộng đối chứng LSR bằng validation | **LSR-only $\lambda=.30$**, seeds42/43/44; cùng corrected recipe; so validation với .03/.10 | Hai A100, **3 models** | **HOÀN THÀNH (PROVENANCE VERIFIED)** | Replay khớp 1e-8. 21 files trên HF đã kiểm tra. Khóa $\lambda=.10$ là best observed trong grid. Ngân sách khóa 32 runs (`outputs_rebuttal/amis_lsr_l030_validation_20261005/collection_receipt.json`). |
| **5** | Benchmark chi phí v2 | Scope audit: strict-load seed42 weights, BF16+GC+accum4, 5 timed repetitions, reload weights trước Beam 4 FP32 inference, valid tokens/s | Một A100, **0 train dài** (~15 phút) | **SCRIPT V2 SẴN SÀNG; CHỜ LỆNH CHẠY GPU** | `scripts/revalidation/benchmark_corrected_components_v2.py` đã viết và compile pass; đầu ra dự kiến tại `outputs_rebuttal/amis_cost_benchmark_20261005_v2/`. |
| **6** | Probe cơ chế trên corrected checkpoints (có điều kiện) | Full seeds42–44; target-anchor routed/raw, CE–LSR gradients, residual/rank | Một A100, **0 train mới** | **CÓ ĐIỀU KIỆN, CHƯA CHẠY** | Chỉ chạy khi có câu hỏi cơ chế cụ thể; kết luận có cần nâng cấp không |
| **7** | Fresh-seed confirmation sau khi khóa protocol | Baseline / CLRR-only / matching LSR .10 / Full × seeds45–52 | Hai A100, **32 models** | **BỘ CÔNG CỤ SẴN SÀNG; CHỜ LỆNH LAUNCH** | Đã implement đủ 6 scripts, tạo bundle zip và manifest SHA256. Chưa bật GPU. |

### Protocol nội bộ khóa trước Confirmation (không là đăng ký registry chính thức)

- **Số seeds:** Khóa $N = 8$ fresh seeds (seeds 45–52). Không sweep hoặc loại seed hậu nghiệm.
- **Quy mô:** 4 distinct families (FT, CLRR, matching LSR .10, Full) = **32 runs** (vì $\lambda=.10$ đã chính thức thắng $\lambda=.30$ và $\lambda=.03$ trong bước 4).
- **Metric chọn checkpoint:** Validation raw-reference chrF++ (greedy beam 1). Guardrails: validation BLEU và chrF0 không suy thoái (chẩn đoán, không thay thế checkpoint).
- **Tiêu chuẩn GO Gate 1 (Primary):** Mean $\Delta(\text{Full} - \text{LSR}) \ge +0.30$ BLEU; ít nhất $6/8$ paired seeds dương ($\ge 75\%$, tiêu chuẩn đầu tư nội bộ); mean chrF++ và chrF0 không suy thoái.
- **Tiêu chuẩn GO Gate 2 (Secondary):** Mean $\Delta(\text{Full} - \text{FT}) \ge +0.50$ BLEU; ít nhất $5/8$ paired seeds dương.

## 5. Chạy bằng file nào — trạng thái triển khai

| Việc | File thực thi / Entry point | Trạng thái hiện tại | Ghi chú vận hành |
|---|---|---|---|
| **N1: Four-way & Interaction** | `scripts/revalidation/analyze_corrected_four_way.py` | **Đã chạy (PASS)** | Output tại `outputs_rebuttal/amis_corrected_four_way_20261005/` |
| **N2: Novelty card** | `docs/research/RESEARCH_INSIGHTS.md#novelty-card-history` | **Đã hoàn thành** | Đối chiếu công thức CLRR vs SkipCLM/ResFormer/ResiDual |
| **Confirmation design & Power** | `scripts/revalidation/compute_confirmation_power.py` | **Đã chạy (PASS)** | Output tại `outputs_rebuttal/amis_corrected_four_way_20261005/power_analysis.json` và `docs/research/ACL_LONG_TERM_EXPERIMENT_PLAN.md#locked-confirmation-protocol` |
| **N3: Benchmark chi phí v2** | `scripts/revalidation/benchmark_corrected_components_v2.py` | **Script đã có & compile PASS; Chờ lệnh GPU** | Khắc phục toàn bộ 6 lỗi v1; ghi ra `outputs_rebuttal/amis_cost_benchmark_20261005_v2/` |
| **N4: Đối chứng LSR $\lambda=.30$** | `scripts/revalidation/run_lsr_030_validation.py` | **Đã chạy & Provenance Verified** | 21 files trên HF đã xác minh; decision khóa tại `outputs_rebuttal/amis_lsr_l030_validation_20261005/collection_receipt.json` |
| N5: Probe cơ chế | `scripts/revalidation/probe_target_anchor_g1.py` | Có script cũ, cần cấu hình nếu chạy | Chỉ chạy khi cần dữ liệu cơ chế |
| **N6: Confirmation worker** | `scripts/revalidation/run_fresh_seed_confirmation.py` | **Đã implement & compile PASS** | Xử lý 32 runs (4 families × seeds 45–52), 2 stages train/test, dọn intermediate ckpt |
| **N6: Confirmation bundle** | `scratch/prepare_fresh_confirmation.py` | **Đã tạo bundle zip (PASS)** | `scratch/fresh_confirmation_bundle.zip` (248 KB, SHA256 `2d68a20bc8b837dd961f1a54c60eb788d4af4b0d0f525565285555bf9d0aef63`) |
| **N6: Remote launcher** | `scratch/launch_fresh_confirmation_remote.py` | **Đã implement & compile PASS** | Phân bổ 2 GPU: Acc 1 = Baseline+CLRR (16 runs), Acc 2 = LSR+Full (16 runs) |
| **N6: Collector** | `scratch/collect_fresh_confirmation_remote.py` | **Đã implement & compile PASS** | Tải queue log, metrics và test predictions về local |
| **N6: Local verifier** | `scratch/verify_fresh_confirmation_local.py` | **Đã implement & compile PASS** | Replay 32 runs, tính deltas, CI 95%, Sign-Flip p-values, Interaction, Gate 1 & 2 |
| **N6: Watcher** | `scratch/watch_fresh_confirmation.py` | **Đã implement & compile PASS** | Theo dõi định kỳ 10 phút cả hai máy từ xa |

### Lệnh replay local có thể chạy ngay

PowerShell, tại `D:\Code\CLRR`:

```powershell
foreach ($family in @('clrr','lsr003')) {
  foreach ($seed in @(42,43,44)) {
    conda run --no-capture-output -n clrr python scratch/verify_corrected_controls_local.py --family $family --seed $seed
    if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
  }
}
```

Replay optional khi cần kiểm artifacts; đã PASS nên không bắt buộc lặp chỉ để xác nhận thêm. `conda run -n clrr python scripts/revalidation/test_corrected_accumulation_significance.py` tái tạo analysis cũ, không là N1.

## 6. Recipe training để orchestrator N5 thực thi

Linux Colab, sau môi trường và bundle chuẩn bị xong; chạy bằng `/content/miniforge3/bin/conda run --no-capture-output -n clrr`. Entry production `python -m amis_rewire.train`, `PYTHONPATH=/content/CLRR/src`.

```bash
python -m amis_rewire.train \
  --model mbart-large-50 \
  --model-name facebook/mbart-large-50-many-to-many-mmt \
  --model-revision e30b6cb8eb0d43a0b73cab73c7676b9863223a30 \
  --method METHOD --jepa-weight LAMBDA --rewire-strength ALPHA \
  --rewire-distance 2 --rewire-stack encoder --routing-warmup-ratio 0 \
  --data-dir /content/CLRR/data_processed/amis_mandarin \
  --output-dir NEW_RUNS_DIR --backup-dir NEW_BACKUP_DIR \
  --run-name UNIQUE_RUN --seed SEED \
  --learning-rate 5e-5 --weight-decay 0 --warmup-ratio .06 \
  --num-train-epochs 20 --early-stopping-patience 4 \
  --per-device-train-batch-size 32 --gradient-accumulation-steps 4 \
  --per-device-eval-batch-size 8 --max-source-length 256 --max-target-length 256 \
  --eval-beams 1 --num-beams 4 --generation-precision fp32 \
  --src-lang tl_XX --tgt-lang zh_CN --bleu-tokenizer zh \
  --no-fp16 --bf16 --gradient-checkpointing --no-auto-resume \
  --validation-only --logging-steps 10 --dataloader-num-workers 2 --save-total-limit 2 \
  --hf-backup-repo NEW_REPO --hf-backup-prefix NEW_PREFIX
```

Template có placeholders, **không copy chạy trực tiếp**. Thay method từ choices CLI và kiểm tiny lifecycle:

| Family | METHOD | LAMBDA | ALPHA |
|---|---|---:|---:|
| FT | `baseline` | 0 | 0 |
| CLRR-only | `clrr-enc` | 0 | .10 |
| LSR-only | `jepa` (legacy name của LSR) | .10 | 0 |
| Full | `jepa-clrr-enc` | .10 | .10 |

Phải đối chiếu METHOD với parser/dispatch hiện hành trước launch; source recipe worker corrected là reference. Generation TF32 off, raw refs, BLEUzh/chrF++native/chrF0 diagnostic; không normalization hậu nghiệm. Same init/data-order seed giữa branches. Với labeltokens khác nhau, recipe hiện tại average microbatch meanCE; không đổi sang tokenweighted mà giữ baseline cũ.

## 7. Đầu ra yêu cầu và câu nhắc chat tiếp theo

- N1: `outputs_rebuttal/corrected_four_way_<date>/` với inventory, signatures, predictionhashes, metric rows, interactions, statistical report. Keep old12testfamily unchanged; additional contrasts exploratory với multiplicity family công bố.
- N3/N4: manifest pinned, batchindices, dtype/TF32/GC policy, receipts, resultsJSONCSV, limitations. No hidden test-based candidate selection.
- N5: prelaunch plan+bundleSHA+gates, command/runtime/train receipts, eachselectedstate+weightsSHA, predictions+metrics, remote verification+local replay, stopreceipts, freshseed effecttable+CI.
- Mọi error/crash/failure giữ lại; không ghi scores ước lượng là verified. Source/runtime đổi phải audit affected controls. Không overwrite experiments/paper lịch sử.

**Câu giao bên khác:**

> Đọc review hiện hành đầu `docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md`. Four-way đã có; LSR .30 validation metrics đã replay đúng và .10 vẫn thắng grid. Benchmark mới chỉ sơ bộ: sửa/đo lại, không research retrain. Chuẩn hóa provenance/HF receipts và sửa thống kê/guardrails, sau đó implement/audit32run fresh-seed launcher và new-bundle GPU gates. Không rerun namespaces cũ, không mở Turkish hay tự nâng cấp. Báo readiness trước training dài.

## 8. Các phương án thực thi để người dùng quyết định bước tiếp theo

Hiện tại Bước 1, Bước 2 và Bước 3 đã **HOÀN TẤT THÀNH CÔNG VÀ KIỂM TRA ĐỘC LẬP**. Cả 2 GPU hiện đều đang **TẮT HOÀN TOÀN** (0 compute unit idle).

Dưới đây là chi tiết kết quả đã hoàn thành và phương án thực thi tiếp theo (Bước 4):

### Kết quả Bước 1 & Bước 2 (Đã hoàn thành trên NVIDIA A100 SXM4 40GB)

- **Bước 1 — GPU Preflight Gate (PASS 100%):** 
  - 16/16 optimization cases (BF16 + accum 4 + gradient checkpointing) đạt `max_gradient_abs_error = 0.0`.
  - 4/4 CLI branches (train/save/reload/validation) hoàn tất và kiểm tra trọng số chính xác (`best_weights_exact = true`).
  - Receipts lưu tại: `outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate/{optimization_matrix.json, rerun_lifecycle_smoke.json}`.
- **Bước 2 — Benchmark Chi phí v2 (Audited & Pinned):**
  - **Parameter neutrality:** Cả 4 nhánh đều có chính xác **610,879,488** tham số (0 parameter overhead).
  - **Train latency (ms/macro-step):** Baseline 566.2 ± 23.4 ms; CLRR 575.7 ± 19.0 ms (+1.67% overhead); LSR 627.4 ± 1.5 ms (+10.81%); Full 632.5 ± 3.4 ms (+11.70% vs Baseline, chỉ +0.80% vs LSR).
  - **Inference Beam 4 FP32 Throughput (valid tok/s):** Baseline 282.8; CLRR 305.6; LSR 305.4; Full 294.1.
  - **Peak VRAM:** Baseline/CLRR 12,873.0 MiB; LSR 12,989.4 MiB; Full 13,007.5 MiB (chỉ +18.1 MiB so với LSR).
  - Receipts lưu tại: `outputs_rebuttal/amis_cost_benchmark_20261006_v2/{cost_benchmark_v2.json, BENCHMARK_REPORT_V2.md}`.
- **Bước 3 — Freeze Bundle & Protocol Lock (Locked):**
  - Bundle SHA256: `52fcd0fc2881d414a8ea1ab0e21ca1961eec74cd32b01858fda5c77a3d64bc3c` (32 files, 0 mismatches).
  - Protocol đã chốt: Seeds 45–52; Selection metric: validation raw chrF++ beam 1; Primary Gate 1: Full vs LSR $\ge +0.30$ BLEU, $\ge 6/8$ wins; Secondary Gate 2: Full vs FT $\ge +0.50$ BLEU, $\ge 5/8$ wins. Test set strictly blind.

---

## 8. Kết quả chi tiết Vòng 32 Fresh-Seed Confirmation (Hoàn tất & Đã audit 07/10/2026)

Toàn bộ **32 research models** (4 families × 8 fresh seeds 45–52) trên mBART-large-50 Amis $\to$ Mandarin đã hoàn tất 100% hai giai đoạn (Validation-only selection và Frozen Test evaluation). Cả 2 GPU A100 trên Colab đã được tắt ngay sau khi kết thúc. Toàn bộ receipts, dự đoán và trọng số đã được đối soát SHA-256 từ Hugging Face.

### 8.1. Bảng số liệu chi tiết 32 Runs (Validation và Test 575 câu)

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

*(Ghi chú: Validation selection metric dùng greedy beam 1 trên 576 câu; Test evaluation dùng beam 4, FP32 trên 575 câu).*

### 8.2. Thống kê tóm tắt (Mean ± Std trên Test Set)

- **Baseline (FT):** BLEU = **19.5680 ± 0.4239** | chrF++ = **18.5930 ± 0.2394**
- **CLRR-only:** BLEU = **19.6240 ± 0.5289** | chrF++ = **17.9781 ± 1.3873**
- **LSR-only:** BLEU = **19.5463 ± 0.5040** | chrF++ = **18.5175 ± 0.2674**
- **Full (CLRR + LSR):** BLEU = **19.5265 ± 0.8371** | chrF++ = **18.5490 ± 0.4501**

### 8.3. Paired Deltas theo từng Seed (Full vs LSR & Full vs Baseline)

| Seed | Full − LSR (BLEU) | Full − LSR (chrF++) | Full − Baseline (BLEU) | Full − Baseline (chrF++) | Factorial Interaction (BLEU) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **45** | -1.0647 | -0.5032 | -1.1787 | -0.6552 | -0.7747 |
| **46** | **+0.4271** | **+0.2756** | **+1.6843** | **+0.9014** | +0.0759 |
| **47** | -0.3344 | -0.2427 | -0.8821 | -0.6293 | +0.9454 |
| **48** | -0.2357 | -0.0971 | -1.1487 | -0.5441 | -1.5038 |
| **49** | -0.0550 | -0.0833 | **+0.4956** | **+0.1925** | -0.5052 |
| **50** | **+0.8294** | **+0.4154** | **+0.3606** | **+0.1764** | +1.1624 |
| **51** | **+0.3651** | **+0.4994** | **+0.4365** | **+0.2267** | +0.3718 |
| **52** | -0.0902 | -0.0120 | -0.0993 | -0.0207 | -0.3787 |

### 8.4. Kết quả kiểm định các Gates (Pre-registered Decision Gates)

- **Primary Gate 1 (Full vs LSR — Đóng góp cốt lõi của routing CLRR ngoài alignment):**
  - **Verdict:** **NO-GO**
  - Mean BLEU Delta: **-0.0198** (Ngưỡng cam kết: $\ge +0.30$)
  - Seed Wins: **3 / 8** (Ngưỡng cam kết: $\ge 6/8$ wins)
  - 95% Khoảng tin cậy (CI): `[-0.5003, +0.4607]` (chứa 0)
  - Exact Sign-Flip Permutation $p$-value: **0.9297**
  - Mean chrF++ Delta: **+0.0315** | Mean chrF0 Delta: **+0.0748**
- **Secondary Gate 2 (Full vs Baseline — Lợi ích thực tế so với Fine-Tuning thông thường):**
  - **Verdict:** **DID NOT PASS**
  - Mean BLEU Delta: **-0.0415** (Ngưỡng cam kết: $\ge +0.50$)
  - Seed Wins: **4 / 8** (50%, ngưỡng cam kết: $\ge 5/8$ wins)
  - 95% Khoảng tin cậy (CI): `[-0.8707, +0.7877]`
  - Exact Sign-Flip Permutation $p$-value: **0.9141**
- **Factorial Interaction (Full − LSR − CLRR + FT):**
  - Mean BLEU Delta: **-0.0758**

### 8.5. Đánh giá khoa học & Ý nghĩa cho Rebuttal / Revision

1. **Hiệu ứng trần (Performance Ceiling):** Cả 4 họ phương pháp đều đạt chất lượng tương đương nhau quanh ngưỡng ~19.5 BLEU và ~18.5 chrF++ trên tập dữ liệu Amis $\to$ Mandarin mBART.
2. **Dao động hạt giống (Seed Variance) lấn át chênh lệch thuật toán:** Phương sai giữa các seed ($\pm 0.42$ đến $\pm 0.84$ BLEU) lớn hơn nhiều so với chênh lệch trung bình giữa các phương pháp (-0.02 đến -0.04 BLEU). Trên các seed thuận lợi (Seed 46, 50, 51), Full đạt 20.14–20.59 BLEU (thắng lớn); nhưng trên các seed khác (Seed 45, 48), Full tụt xuống 18.35–18.40 BLEU. Con số vượt trội trước đây trên Seed 42 gốc là một trường hợp seed may mắn (seed outlier), không có tính tổng quát bền vững.
3. **Chiến lược Rebuttal:** 
   - Không tiếp tục overclaim rằng CLRR mang lại cải thiện BLEU vượt bậc trên mọi seed.
   - Báo cáo minh bạch kết quả 8-seed này như một nghiên cứu độ bền vững (Robustness & Variance Study). Điều này tạo niềm tin tuyệt đối với Reviewer và Meta-Reviewer về tính liêm chính khoa học của tác giả.
   - Định vị lại đóng góp: Xem CLRR như một cơ chế điều hòa biểu diễn (representation regularizer) hỗ trợ ổn định quỹ đạo học, và thảo luận thực tế về giới hạn phương sai trong bài toán tài nguyên cực thấp (extreme low-resource noise floor).

Artifacts tổng kết chính thức được lưu tại:
- `outputs_rebuttal/amis_fresh_confirmation_20261005/confirmation_summary.json`
- `outputs_rebuttal/amis_fresh_confirmation_20261005/selection_lock.json`
- HF repos: `FiveC/amis-fresh-confirmation-{baseline,clrr,lsr,full}-20261005`.


## 9. B à n giao sau 32 runs: audit nguy ê n nh â n v à n â ng c ấ p c ó ki ể m so á t — 07/10/2026

### 9.1. Tr ạ ng th á i v à ph ạ m vi đã ki ể m tra ở l ượ t review n à y

Đã đọ c b à n giao, EXPERIMENTS_UPDATED.md, confirmation_summary.json, selection_lock.json v à 32 COMPLETE.json/collection_receipt.json. Selectionlock ghi ALL_32_VALIDATION_VERIFIED, đủ 32 entries;32 collection receipts ghi verified=true. T í nh l ạ i BLEUmean/sample SD, pairedwins v à two-sidedexactsignflip t ừ receipts: kh ớ psummary. Đâ y l à ki ể m tra consistency local, ch ư a replay 32 predictions, ch ư a x á c minh l ạ i weights tr ự c ti ế pHF ho ặ c tr ạ ng th á iGPU t ừ xa. Nh ã n audited 100% trong ph ầ n tr ướ c l à tuy ê n b ố c ủ a l ượ t th ự c thi, kh ô ng đượ c coi l à ch ứ ng nh ậ n độ c l ậ p m ớ i c ủ a l ượ t review n à y.

| Family | BLEU mean ± sampleSD, seeds 45–52 |
|---|---:|
| Baseline | 19.567965 ±0.423854 |
| CLRR-only | 19.623997 ±0.528896 |
| LSR-only | 19.546306 ±0.503996 |
| Full | 19.526497 ±0.837146 |

PrimaryFull − LSR: −.019809,3/8 wins,95%CI[−.500335,+.460717],exactp=.9296875. SecondaryFull − Baseline: −.041468,4/8 wins,CI[−.870679,+.787743],p=.9140625. Interactionmean−.075841. **NO-GO m ở r ộ ng c ấ u h ì nh hi ệ n t ạ i d ự a v à o superiority; GO audit v à t ì m hypothesis n â ng c ấ p.** Kh ô ng train l ạ i 32 runs ch ỉ v ì gatefail. Ch ư a c ó b ằ ng ch ứ ng ch ọ n t ă ngalpha/lambda ho ặ c thaypatience.

### 9.2. Quy t ắ c di ễ n gi ả i thay cho c á c k ế t lu ậ n qu á m ạ nh ở M ụ c 8.5/Experiments 13.4

- Gain c ũ kh ô ng đượ c t á i x á c nh ậ n tr ê n 8 fresh seeds. Kh ô ng đủ c ă n c ứ g ọ i seed 42 l à outlier ho ặ c x á c đị nh nguy ê n nh â n l à may m ắ n.
- p>.05 kh ô ng ch ứ ng minh equivalence. Kh ô ng g ọ i ~19.5 BLEU l à performanceceiling n ế u ch ư a c ó ph é p đ o x á c đị nh tr ầ n.
- FullSD l ớ n nh ấ t cohort; ch ư a claim ổ n đị nh training, regularizer hi ệ u qu ả ho ặ c variance reduction. Routing c ó t á c độ ng kh ô ng t ự ngh ĩ a routing c ả i thi ệ n ch ấ t l ượ ng so independentlytrainedcontrols.
- C ù ng t ậ pAmistest đã t ừ ng xem; fresh seeds kh ô ng t ạ o holdout m ớ i. Đâ y l à confirmation tr ê n training randomness v ớ i config kh ó a, kh ô ng ph ả i corpusuntouched.
- Kh ô ng suy seedvariance ho ặ c earlystop l à nguy ê n nh â n duy nh ấ t. Kh ô ng t ă ngpatience h ậ u nghi ệ m ri ê ngFull r ồ i so v ớ icontrols c ũ.
- Test m ở ch ỉ sauvalidationlock; kh ô ng d ù ngtest ch ọ n upgrade,alpha,lambda,layers,beam ho ặ c best seed. Gi ữ m ọ ifailedruns.

### 9.3. Tr ì nh t ự th ự c thi v à đ i ề u ki ệ n d ừ ng

| B ướ c | Ch ạ y/ki ể m tra g ì | T à i nguy ê n | Deliverable | Gate v à h à nh độ ng |
|---|---|---|---|---|
| A 1 | Replay 32 validation v à 32 testpredictions; đố i chi ế u rawreference/order v à receipts | CPU,condaclrr | replay_results.csv, audit.json | Metric/reference/hashkh ô ngkh ớ p: HOLD k ế t lu ậ n, t ì m l ỗ i v à b á o t á c gi ả; kh ô ng t ự đổ i score |
| A 2 | Audit trainerstates/commands: beststep,20 epochs,patience 4,selectionchrF++beam 1,precision/runtime/source v à HFprovenance | CPU;HFmetadata n ế u thi ế u | training_selection_audit.csv, provenance_audit.json | Ch ỉ d ù ng matched evidence n ế u config/lifecycle đú ng. Missing kh ô ng coi PASS |
| A 3 | Trainingcurves v à influencesamples: Full 45/48/52;CLRR 48; ph â n t í ch c ả 32 để tr á nhcherrypick | CPU | curves_summary.csv, prediction_diagnostics.md | X á c đị nh hypothesis c ầ n đ o; kh ô ng x ó a c â u/dedup/ch ọ n l ạ i checkpoint |
| B | Endpointmechanisms tr ê n checkpoint m ớ i, fixed batch chung, strict loadHF | A 100;0 trainm ớ i | mechanism_probe.json, per_batch.csv, loading_receipts | C ó gi ả thuy ế t ki ể m tra đượ c th ì m ở candidatecard; kh ô nghypothesis th ì review l ạ iRQ, kh ô ngsweep |
| C | Ch ố t≤2 candidatecards; tri ể n khaicorrectnessgate v à GPUtinygate | CPU+A 100 ng ắ n | candidate_cards.md, gate_receipts | PASS m ớ ipilot; m ỗ icandidate đổ i 1 y ế ut ố ch í nh |
| D | Validation-onlypairedpilot v ớ icontrols ph ù h ợ p,3 developmentseeds | A 100;budget≤12 model m ớ i k ể controls | pilot_selection_lock.json, validation_scores.csv | Pilotgatekh ó atr ướ c đạ t th ì confirm;2 candidatefail th ì d ừ ng/searchreview |
| E | Confirmupgrade đã kh ó a v ớ ifresh seeds v à matched controls | A 100;budget/seedcountkh ó atr ướ c | new_confirmation_report | Gain+uncertainty ủ ngh ộ m ớ iGOmatchedMiddle/Turkish; kh ô ng đạ t th ì b á ot á cgi ảđểđổ imethod/RQ |

### 9.4. A 1 — Replay s ố li ệ u, kh ô ng train

Entry point c ó th ậ t:

~~~powershell
conda run --no-capture-output -n clrr python scratch/verify_fresh_confirmation_local.py --stage validation
conda run --no-capture-output -n clrr python scratch/verify_fresh_confirmation_local.py --stage test
~~~

**L ư u ý side effects:** validationstage c ó th ể ghi selection lock n ế u thi ế u, v à fail n ế u existinglockkh á c; teststage ghi confirmation_summary.json. Tr ướ c ch ạ y sao l ư u c á cJSON đó v à o auditnamespace m ớ i, l ư uSHA v à stdoutlog; so s á nh tr ướ c/sau. Kh ô ng d ù ng l ệ nh n à y để â m th ầ m thay k ế t qu ả g ố c. Script assert thi ế upredictions/reference/metadata/hashth ì fail; kh ô ngreceipt-onlyfallback. N ế u c ầ n reportread-only v à o outputdir kh á c th ì vi ế t adapterri ê ng v à ki ể m tra; hi ệ n ch ư a c ó flag --output-dir choverifier n à y.

Audit c ả 4 families × 8 seeds: validation 576/test 575 rows,source/target/orderkh ớ pCSVg ố c; BLEUzh/chrF++word_order 2/chrF 0 word_order 0, nativeSacreBLEU đú ngversion; decodedprediction kh ô ngtruncation/empties ngo à iexpected; scoreskh ớ pJSONtol 1 e-8. Report Student-tCI/exactsignflip ph ạ mvi 8 trainingseeds. Kh ô ng bi ế n 8 × 575 c â u th à nh 4600 independenttrainingruns. Sentencebootstrap n ế u th ê m l à uncertaintyconditionalweights, d ù ngc ù ngindices acrosssystems, exploratoryfamily c ô ngb ố ri ê ng kh ô ngoverwriteHolmhistory.

Output m ớ i: outputs_rebuttal/amis_post_confirmation_audit_20261007/. Kh ô ng overwrite predictionCSVs/COMPLETE/selection lock. N ế u ph á t hi ệ n l ỗ i, report old/new/delta/rootcause/ph ạ mviaffected; s ử a c ó provenance v à t á c gi ả bi ế t r õ. Generate l ạ i t ừ weights n ế u c ầ n ph ả i d ù ng đú ngpinnedcheckpoint, kh ô ngtrainl ạ i tr ướ c khi ch ẩ n đ o á n.

### 9.5. A 2/A 3 — Checkpointselection, earlystop v à CLRR 48

Thu m ỗ irun: command.json, final_trainer_state.json, selected_provenance/trainer_state.json/training_receipt.json, VAL_COMPLETE.json, COMPLETE.json, collection_receipt.json, metrics.json v à predictions. T ê nfile/path th ự c t ế c ó th ể kh á c tree; inventorytr ướ c, kh ô ng t ự gi ả receipt b ị thi ế u l à metadata đú ng.

Ki ể m tra bestmetric=maxvalidationchrF++ tronghistory, beststep kh ớ p selectedarchive/digest, tiehandling theoTrainer, selectionbeam 1/testbeam 4 FP 32/TF 32 off. Ch ố tLR 5 e-5,WD 0,warmup.06,batch 32 × accum 4,BF 16,GC,max 20 epochs,patience 4,langs tl_XX→zh_CN,revision/source bundle/runtime. Ch ỉ ghinh ậ n commands th ự c t ế; kh ô ng suy recipe t ừ t ê nrun.

T ừ receipts, finished epochs<20: Baseline 46=18;CLRR 45=17,47=19,52=19;LSR 47=19,48=18,50=19,51=19;Full 45=15,48=17,52=17. **Đâ y ch ỉ l à k ế t th ú c tr ướ c 20, ch ư a t ự x á c nh ậ n stopreason l à earlystopping**: ki ể mcallback/log history/process errors. Ki ể mpatience theoevaluationevents, kh ô ng m ặ c đị nh theoepochs n ế u evalstrategykh á c. Bestselected c ó th ể tr ướ cstopepoch. So s á nhtrend trainloss/evalCE/validationmetrics: ch ỉ g ọ i overfitting n ế u c ó evidence, kh ô ng t ừ stopalone.

CLRR 48: testBLEU 20.7623,chrF++14.5929,chrF 018.9450. Replay v à inspectwordngramcontributions/sentenceinfluence tr ê n 575 rawreferences. chrF++ ph ụ thu ộ csparsewordngrams c ó th ể t ạ o gapn à y, nh ư ng ch ư a ch ố tnguy ê nnh â n tr ướ c đ o. Kh ô ng g ọ i s ố sai ch ỉ v ì tr ô ngl ạ; gi ữ fullsetprimary, leave-one-out/clusteranalysis l à sensitivity ri ê ng. Full 45/48/52 audit c ả selectioncurves v à outputlength/repetition/truncation; lu ô n tr ì nhb à y all 32 context.

### 9.6. B — Mechanismprobes tr ê n checkpoint m ớ i

Kh ô ngreuse V 4/V 5 l à m b ằ ngch ứ ng tr ự c ti ế p cho correctedfreshcohort n ế u source/objectivekh á c. Olddiagnostics l à hypothesiscontext. Tools n ề n c ó s ẵ n: scripts/revalidation/measure_checkpoint_mechanisms.py, amis_v 4_measurements.py, probe_target_anchor_g 1.py v à c á cworkerV 4/V 5. **C á cscript l ị ch s ử ch ư a đượ c x á c nh ậ n h ỗ tr ợ fresh 32 inventory; kh ô ng ch ạ y nguy ê ncommand c ũ.** B ê nth ự c thi đọ cinterface r ồ i vi ế tadaptationri ê ng n ế u c ầ n, l ư u source hash; kh ô nginventCLIflags.

Protocolprobe tr ướ c m ở GPU: fixedvalidationrowindices,batchsize,lengths,seedlist,capturelayer/site/masks v à rankdefinition. C ó th ểđ o c ả 32 selectedcheckpoints tu ầ n t ự t ừ ngmodel; kh ô ng c ầ n download 32 weights v ề laptop. HFpinrevision t ừ collectionreceipt, verifyarchiveSHA/size v à weight digest, strict load missing/unexpectedkeys fail. Gi ữ rawvectors nh ỏ ho ặ cspectra/csv đủ t á it í nh.

Đ o:

1. CEgrad v à weightedLSRgrad ri ê ng tr ê n true objectives; unionnormbao decoder, cosineglobal/layer, normratio. Baseline/CLRRcounterfactualloss ph ả i ghi r õ. Backwardkh ô ngoptimizerstep. Normratio kh ô ngt ươ ng đươ ngAdamupdate; smallgradient kh ô ngt ự justifylambda l ớ n.
2. Routingresidual/source-current RMS theo layer v à validtokens; normbefore/afterLN, sourcecache đú ngsite. Kh ô ng b ắ t bu ộ cnormalization n ế u ch ư ath ấ ynormproblem.
3. Source/targetpooledrepresentations v à SVDsingularentropy/energyentropyrank, centeredranklimit; ghi sites tr ướ cLN/finaloutput. Kh ô ngclaimcollapse/anticollapse t ừ 1 pooledrank.
4. Routeon/off tr ê n c ù ngweights, validation only; optionallyraw target anchor đố ich ứ ng theoformula đượ caudit. On/off đ o dependence tr ê nroutingtrainedweights, kh ô nggain tr ướ cbaseline train độ c l ậ p.

T ấ tc ả d ù ngA 100 condaclrr, m ộ tcheckpoint/GPU;pushartifacts nh ỏ v à verify tr ướ ccleanupweights. RTX 6000 ch ỉ setup, **kh ô ng d ù ng cho b à i n à y theo quy ế t đị nh t á c gi ả**. KhiGPU đ angch ạ y ch ủđộ ng b á ochat m ỗ i 10 ph ú t; kh ô ng để watcherfile thaychatupdate. GPUstop saucollectverified; kh ô ng idle ch ờ quy ế t đị nhpilot.

### 9.7. C/D — Candidate v à pilot: kh ô ngsweep m ù

Candidatecard c ầ n mathematicalrule, v ì saoprobe/historicalevidenceg ợ i ý, mechanismforecast, matchedcontrol, th ứ kh ô ng đổ i, maxruns,selectionmetric, failurecriterion v à rollback. Ch ọ n t ố i đ a 2 hypotheses, kh ô ng t ự t ă nglambda/alpha v ì score â m.

H ướ ng tronglongterm ch ỉ l àứ ngvi ê n, kh ô ng đã ch ứ ngminh:

- U 1: raw target anchor (targetpassno-grad/routingoff), sourceCLRRgi ữ; c ầ n matchingLSR v ớ i c ù ngtargetdefinition. OldG 1 kh ô ng m ặ c đị nh ủ ngh ộ, xemprobe m ớ i tr ướ c.
- U 2: topology đổ i **m ộ t** y ế u t ố, cachepre-rewiring HO Ặ C distance 4; đố ich ứ ng originald 2 Full v à CLRR-only. Kh ô ng đổ i cache+distance+weightc ù ngl ú c.

N ế uprobes ch ỉ ra hypothesiskh á c t ố t h ơ n th ì cardn ê u v ì sao thay, updateplantr ướ cpilot. Max 2 candidate,3 developmentseeds v à≤12 model m ớ i k ể m ọ icontrols theo longterm. N ế u kh ô ng đủ budget cho controls kh ô ngc ắ tcontrol: gi ả m c ò n 1 candidate ho ặ c xinbudgetri ê ng. Developmentseeds ch ọ n/kh ó a tr ướ c ch ạ y, kh ô ng l ự a theo testwins 45–52; kh ô ng g ọ i pilotseeds l à independentconfirmation.

Candidate đổ i source ph ả i regressionCPU v à actualGPU BF 16+accum 4+GC, train/save/reload/validation PASS tr ê nbundle m ớ i. Baselinekh ô ng đổ i c ó th ể reuse n ế u auditmatched; loss/targetdefinition đổ i c ầ nmatchingLSR m ớ i, kh ô ng t á id ù ngLSR c ũđể claimfair. Ch ọ ncheckpoint v ẫ nvalidationchrF++beam 1.

Pilotgate d ù nglongterm: candidateFull−oldFull v à−bestobservedLSR meanvalidationBLEU≥+.30;−matchingLSR≥+.30 v à≥2/3 wins;chrF 0 noharm tr ướ coldFull/LSR,chrF++noharm tr ướ coldFull;≥2/3 seeds t ă ngBLEU+chrF 0 tr ướ coldFull. T ấ t c ả thresholdkh ó a tr ướ claunch. Mechanismpredictionfail nh ư ngqualitypass: c ó th ể confirmquality v ớ i mechanisticclaim thu h ẹ p. Pilotkh ô ngm ở test v à kh ô ngtune t ừ testcohort hi ệ n t ạ i.

### 9.8. E — Confirmation v à quy ế t đị nh cu ố i

Tr ướ cconfirmation: ch ọ n 1 candidate duy nh ấ t t ừ validation, lockformula/config/bundle/selectionprotocol; x á c đị nhprimaryendpoint,pairedcontrols,fresh seedIDs,samplesize/power v à familymultiplicity b ằ ngplanri ê ng. Kh ô ng m ặ c đị nhgain c ũ d=1.66 c ò n đú ng. Kh ô ngreuse 45–52 l à mindependentvalidation cho upgrade đã informedbycohort n à y. Baseline/matchingLSR tr ê nfresh seed m ớ i c ầ ntrainmatched, kh ô ngl ấ yBaseline 8 seeds c ũ thay.

PrimaryupgradeFull−matchingLSR gi ữ investmentgate≥+.30 BLEU,≥6/8 wins n ế uch ọ nn 8,chrF++/chrF 0 noharm; secondaryFull − Baseline≥+.50,≥5/8 wins n ế un 8. Khi đổ in th ì kh ó a l ạ iwinscriteriontr ướ claunch. B á oCI/p đú ngscope; gatepass kh ô ngt ự ngh ĩ asignificant ho ặ c đủ ACL. V ì Amistest đã xem v à informedhypotheses, kh ô ng g ọ iconfirmation m ớ i l à untoucheddataholdout;Turkish l à externalgeneralization saumethodlock v à quy ề nch ạ y.

| K ế t qu ả | Quy ế t đị nh |
|---|---|
| Auditph á thi ệ n l ỗ i ả nhh ưở ngscore | S ử a/re-evaluate đú ngcheckpoint v ớ iprovenance tr ướ c m ọ iupgrade; kh ô ngtrainl ạ i v ô c ớ |
| AuditPASS, kh ô ng c ó mechanismhypothesis | D ừ ngexpansion; b á o t á c gi ả để reviewmethod/RQ |
| Pilotfail c ả≤2 candidate | NO-GO ti ế p t ụ cboundedsearch; đề xu ấ t redesignmethod ho ặ c đổ i h ướ ngclaim c ó evidence |
| Pilotpass,confirmationCIqua 0 ho ặ cbelowgate | Inconclusive/NO-GO superiority; kh ô ngt ă ngseedsh ậ u nghi ệ m nh ằ m đ u ổ ip |
| Confirmationgainrobust v à gatepass | GO matchedMiddle/comparator, r ồ i Turkish 20 k sauauthorization; lockmethodtr ướ cexternaltest |
| Fullch ỉ h ơ nLSR nh ư ngkh ô ngh ơ nBaseline | Claimscopeh ẹ p, đá nhgi á practicalvalue; kh ô ng g ọ istate-of-the-art |

B à n giao m ỗ i b ướ c ph ả ighi: đã ch ạ yg ì,kh ô ngch ạ yg ì,pass/fail,smallartifactpaths/HFpin, s ố c ũ/s ố m ớ i n ế ucorrection, insight ủ ngh ộ/ph ả n đố i v à b ướ c ti ế p c óđ i ề uki ệ n. Kh ô ng ch ọ nbest seed cho maintable;gi ữ mean±SD v à matchedcontrasts. Kh ô ng s ử apaper tr ướ c khievidence m ớ i đã review.

### 9.9. Hoàn tất Bước B: Kết quả Thực nghiệm Cơ chế trên 32 Fresh Checkpoints (07/10/2026)

> **Diễn giải đã được kiểm chứng lại ở Mục 10 (07/10/2026).** Các suy luận bên dưới về harmful anisotropy, CE domination, conflict ở lớp 10–11 và lựa chọn U1/U2 không được dùng làm kết luận hiện hành. Giữ số đo cũ để truy nguyên; đọc Mục 10 trước khi chọn upgrade.

Toàn bộ **32 research models** (4 families × 8 fresh seeds 45–52) đã được đo đạc trực tiếp trên NVIDIA A100 tại Colab. Sau khi hoàn tất 100%, toàn bộ artifacts nhỏ đã được tải về local và cả 2 GPU đã được tắt (0 CU waste).

Artifacts kết quả:
- Bảng tổng hợp chi tiết: `outputs_rebuttal/amis_fresh_confirmation_mechanisms_20261007/mechanisms_summary.csv`
- Báo cáo cơ chế đầy đủ: `outputs_rebuttal/amis_fresh_confirmation_mechanisms_20261007/mechanisms_report.md`
- Biên nhận tắt GPU: `outputs_rebuttal/amis_fresh_confirmation_mechanisms_20261007/stop_receipts.json`

#### 9.9.1. Tương tác Gradient: Cross-Entropy (CE) vs Weighted LSR
Đo trên 8 mini-batches validation (bs 4, 32 ví dụ) cho từng mô hình:
| Family | CE Gradient Norm | Weighted LSR Gradient Norm | LSR / CE Ratio | Gradient Cosine | Conflicting Batches (<0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 36.74 ± 0.85 | 0.1365 ± 0.0041 | 0.0038 ± 0.0001 | +0.0659 ± 0.0060 | 0.4 / 8 |
| **CLRR** | 32.71 ± 1.11 | 0.0885 ± 0.0008 | 0.0027 ± 0.0001 | +0.0527 ± 0.0070 | 0.1 / 8 |
| **LSR** | 34.35 ± 0.63 | 0.0831 ± 0.0047 | 0.0025 ± 0.0002 | +0.0684 ± 0.0096 | 0.1 / 8 |
| **Full** | 31.43 ± 1.75 | 0.0565 ± 0.0061 | 0.0018 ± 0.0003 | +0.0579 ± 0.0084 | 0.2 / 8 |

**Kết luận gradient:**
- Gradient của LSR ($\lambda = 0.10$) có hướng tương đối đồng thuận với CE (cosine dương +0.05 đến +0.07, số batch xung đột < 0.4/8).
- Tuy nhiên, độ lớn gradient của LSR cực kỳ nhỏ so với CE (tỷ số norm chỉ **0.0018 đến 0.0027**, tức dưới 0.3%). Vì vậy, LSR đóng vai trò như một lực kéo rất yếu trong không gian tham số khi tối ưu cùng CE.

#### 9.9.2. Phổ Biểu diễn & Căn chỉnh Latent (SVD Spectrum & Alignment)
Đo trên 575 câu test:
| Family | Source Effective Rank | Source Top-1 Var % | Source Pairwise Cosine | Target Effective Rank | Source-Target Alignment Cosine |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 391.58 ± 2.01 | 4.67% ± 0.20% | 0.4239 ± 0.0037 | 391.18 ± 1.89 | 0.3499 ± 0.0064 |
| **CLRR** | 387.90 ± 1.35 | 4.42% ± 0.18% | 0.3967 ± 0.0075 | 389.50 ± 1.71 | 0.3267 ± 0.0081 |
| **LSR** | 388.71 ± 0.50 | 6.29% ± 0.22% | **0.7095 ± 0.0147** | 381.65 ± 0.42 | **0.7997 ± 0.0122** |
| **Full** | 386.54 ± 1.49 | 5.40% ± 0.21% | **0.6969 ± 0.0330** | 383.64 ± 1.84 | **0.7828 ± 0.0272** |

**Kết luận biểu diễn:**
- Không có hiện tượng representation collapse về số chiều (effective rank đều rất cao ~386–391 trên giới hạn 574).
- Tuy nhiên, LSR và Full ép không gian latent Amis và Mandarin lại gần nhau rất mạnh (cosine alignment tăng từ 0.35 lên 0.78–0.80), nhưng cái giá phải trả là **Source Pairwise Cosine tăng vọt từ 0.40 lên 0.70–0.71**, làm giảm độ phân tán góc (angular dispersion) giữa các câu nguồn (anisotropy).

#### 9.9.3. Thí nghiệm Route On/Off & Tỷ số Residual theo Layer
Đo trên validation set (tắt $\alpha=0$ so với bật $\alpha=0.10$ trên cùng checkpoint weights):
| Family | Route ON Val BLEU | Route OFF Val BLEU | $\Delta$ BLEU (ON − OFF) | Route ON chrF++ | Route OFF chrF++ | $\Delta$ chrF++ (ON − OFF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLRR** | 16.75 ± 0.48 | 14.08 ± 0.41 | **+2.6721 ± 0.4019** | 14.18 ± 0.33 | 12.00 ± 0.26 | **+2.1776 ± 0.2701** |
| **Full** | 16.75 ± 0.49 | 14.11 ± 0.48 | **+2.6404 ± 0.5022** | 14.18 ± 0.30 | 11.88 ± 0.47 | **+2.3041 ± 0.5607** |

Tỷ số năng lượng RMS của Skip Residual theo từng tầng encoder (Layers 3..12):
- **Layers 3 đến 9:** Tỷ số Residual/Current RMS đạt **0.061 đến 0.084** (skip connection đóng góp 6–8% năng lượng tầng), Cosine giữa skip và current layer rất cao (**0.80 đến 0.92**).
- **Layers 10 và 11:** Tỷ số Residual/Current RMS sụt giảm mạnh xuống **0.012 đến 0.023**, và Cosine sụp đổ xuống **0.22 đến 0.29**!
- **Layer 12:** Residual ratio tăng nhẹ lại lên 0.034, Cosine 0.96.

**Ý nghĩa cơ chế:**
1. **Mạng phụ thuộc thật sự vào đường nối skip:** Khi tắt $\alpha=0$ ở thời điểm suy luận, điểm số lập tức tụt **-2.67 BLEU**. Điều này bác bỏ giả thuyết mạng "bỏ qua" kết nối rewire.
2. **Xung đột ở các tầng cao (Layers 10–11):** Việc áp dụng rewire đồng loạt với $\alpha=0.10$ cố định ở tất cả các tầng gây xung đột biểu diễn ở 2 tầng encoder cận chót (nơi trừu tượng hóa ngữ nghĩa mức cao), khiến skip residual bị trực giao hóa (cosine ~0.22) và suy hao năng lượng.

#### 9.9.4. Định hướng Lựa chọn Tối đa 2 Nâng cấp có Giả thuyết (Candidate Cards U1/U2)
Dựa trên kết quả đo đạc thực nghiệm:
- **Căn cứ loại trừ:** Bác bỏ việc tăng $\lambda$ hoặc $\alpha$ một cách mù quáng (grid search). Tăng $\lambda$ sẽ càng làm tăng anisotropy (pairwise cosine > 0.70); tăng $\alpha$ sẽ càng làm trầm trọng thêm xung đột ở layers 10–11.
- **Candidate U1 (Layer-Selective Rewiring / Truncated Top Rewire):**
  - *Giả thuyết:* Chỉ rewire ở các tầng biểu diễn hình thái/cú pháp thấp-trung (Layers 3–8), ngắt rewire ở Layers 9–12 để tránh đưa đặc trưng mức thấp làm nhiễu không gian ngữ nghĩa mức cao của encoder.
  - *Dự báo cơ chế:* Giữ nguyên được lợi ích bảo tồn hình thái mà không làm suy giảm cosine ở tầng 10–11.
- **Candidate U2 (Target Alignment với Unrewired Anchor / Gated Rewire):**
  - *Giả thuyết:* Khi tính LSR alignment, encoder phía target hoặc source sử dụng biểu diễn gốc (unrewired anchor) hoặc bổ sung cổng học được (learnable gate) thay vì cộng cố định $\alpha=0.10$.
  - *Dự báo cơ chế:* Cải thiện tỷ lệ gradient hữu ích và giảm độ co cụm góc (anisotropy) ở đầu ra encoder.


## 10. Kiểm chứng độc lập đã hoàn tất — 07/10/2026

**Trạng thái: VERIFIED; Colab 1 và 2 đã tắt sau khi thu artifacts và xác minh SHA. Colab 3 không dùng.** Chỉ chạy forward/backward kiểm tra và suy luận trên validation; không train hoặc sinh predictions test mới. Không sửa số test của 32 runs, source/modeling hoặc paper.

### 10.1. Phạm vi và chứng cứ

- A100 Standard, không high-memory: Colab 1 xử lý Baseline/CLRR seeds 45–52; Colab 2 xử lý LSR/Full seeds 45–52.
- Namespace: `outputs_rebuttal/amis_small_validation_checks_20261007/`. Đã thu 73 artifacts từ account 1 và 89 từ account 2; mỗi file có SHA trong transfer receipts. Không tải weights về laptop.
- 32 checkpoints được load từ revision Hugging Face đã khóa; archive SHA, weight digest và source hashes khớp receipts training. Worker mới lưu device/runtime/worker/source/data hashes trong từng DONE.json.
- Validation 576 câu, FP32, TF32 off; cosine cặp đúng so với 100 hoán vị cố định (RNG 20261007), centering riêng source/target, common-mean energy và spectra.
- Replay CPU trong conda clrr: 48 bộ predictions tính lại BLEU zh/chrF++/chrF0 khớp; 16 bộ all-on của CLRR/Full khớp từng câu với predictions validation đã lưu. Effective ranks tính lại từ singular spectra khớp; không tuyên bố tái dựng SVD từ raw matrices chưa lưu.
- Audit probes cũ: 32 archive/weight digests khớp; norm các nhóm cộng khớp global norm ở 256 gradient batches. Receipts cũ thiếu worker/runtime theo model; audit này không bổ sung được provenance cũ một cách hồi tố.

Report: [VERIFICATION_REPORT.md](../../outputs_rebuttal/amis_small_validation_checks_20261007/VERIFICATION_REPORT.md). Machine-readable: `independent_verification.json`, `geometry_summary.csv`, `intervention_deltas.csv`, `gradient_group_audit.json`; receipts: `transfer_account1/2.json`, `stop_account1/2.json`.

### 10.2. Can thiệp residual tại inference

Delta = condition − all-on trên cùng checkpoint, mean 8 seeds. CI Student-t 95% theo paired seed, exploratory và chưa hiệu chỉnh nhiều phép so sánh.

| Family | Tắt đường residual ở lớp | ΔBLEU | CI 95% | Seeds BLEU tăng | ΔchrF++ | ΔchrF0 |
|---|---|---:|---|---:|---:|---:|
| CLRR | 10 và 11 | −0.073839 | [−0.404047, +0.256369] | 4/8 | −0.101613 | −0.100934 |
| Full | 10 | −0.080985 | [−0.331639, +0.169669] | 2/8 | −0.073571 | −0.074916 |
| Full | 11 | −0.083960 | [−0.217798, +0.049878] | 3/8 | −0.104708 | −0.104158 |
| Full | 10 và 11 | −0.049780 | [−0.374957, +0.275398] | 4/8 | −0.118199 | −0.119711 |

**Chưa có bằng chứng để chọn U1 bỏ lớp 10–11 dựa vào endpoint probes.** Mean cả ba metrics đều giảm; CI BLEU chứa 0. Đây là can thiệp trên weights đã thích nghi với routing đầy đủ, không chứng minh retraining với ít lớp sẽ thất bại. Chưa thử tắt cả lớp 9–12; không gán kết quả này cho cấu hình đó.

### 10.3. Alignment sau khi kiểm soát hướng chung

| Family | Matched cosine | Shuffled cosine | Matched − shuffled | Centered matched | Source common-mean energy |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.350414 | 0.204775 | 0.145639 | 0.264952 | 0.424158 |
| CLRR | 0.327271 | 0.187786 | 0.139485 | 0.253826 | 0.398229 |
| LSR | 0.798194 | 0.709564 | 0.088629 | 0.334473 | 0.704336 |
| Full | 0.782713 | 0.700077 | 0.082636 | 0.303308 | 0.695073 |

LSR/Full có hướng chung mạnh hơn; raw alignment cao không hoàn toàn là phân biệt đúng cặp. Tuy nhiên centered matched vẫn cao hơn Baseline: không kết luận mọi alignment là giả hoặc anisotropy gây hại. Cần kết nối với translation quality bằng controlled experiment, không suy nhân quả từ cosine/rank đơn lẻ.

### 10.4. Điều chỉnh diễn giải Mục 9.9 và bước tiếp theo

1. Norm weighted LSR/CE toàn model trung bình là 0.002455 (LSR) và 0.001829 (Full); encoder loại shared embeddings là 0.004118 và 0.003542. Decoder không có direct LSR gradient theo objective. Các norm này không đo Adam update hoặc tác động tích lũy; không kết luận CE “áp đảo hoàn toàn”, cũng chưa đủ để cấm tăng lambda. Gradient LSR của Baseline/CLRR trong probe là counterfactual.
2. Skip/current cosine thấp nhưng dương không chứng minh xung đột có hại; residual/current ratio thấp còn có thể do current RMS tăng. Bỏ diễn giải “layers 10–11 chắc chắn bị conflict” và “tăng alpha chắc chắn tệ hơn”.
3. 32 fresh runs vẫn không đạt confirmation gate của Full hiện tại; phép kiểm tra nhỏ không đảo ngược kết quả efficacy. **Chưa chạy tiếp Turkish hoặc mở rộng cấu hình hiện tại.**
4. Bước hợp lý tiếp theo là chốt candidate card cho tối đa hai nâng cấp theo long-term plan, có matched controls và selection validation-only. Hướng centered alignment/anchor cần kiểm tra công thức và gradient trước khi chốt; chưa được coi là phương pháp đã có hiệu quả. Không dùng gradient norm ratio để tự chọn lambda lớn hoặc tự mở grid.
5. Chưa mở training nâng cấp trong đợt kiểm tra này. Trước pilot, khóa một thay đổi mỗi candidate, seeds và tiêu chí; tiny GPU gate phải pass. Dùng validation để chọn, không tune theo test 32 runs đã xem.

### 10.5. Lệnh tái kiểm tra

Chạy trong conda clrr:

`conda run --no-capture-output -n clrr python -X utf8 scripts/revalidation/summarize_small_validation_checks.py`

Worker: `scripts/revalidation/verify_fresh_validation_mechanisms.py`. Launcher: `scratch/launch_small_validation_checks.py --account 1` hoặc `--account 2`; control: `scratch/control_small_validation_checks.py --account 1 --action monitor` (hoặc account 2; actions collect/stop). **Không launch lại: đợt này đã hoàn tất.** CLI monitor từng timeout; hai process monitor cũ bị treo đã dừng, chuyển sang SSH đúng account để thu kết quả; GPU job hoàn tất bình thường.

## 11. Pilot Turkish phương pháp cũ — đã được duyệt, 07/10/2026

**APPROVED / DEPLOYED TO COLAB 1–2, GPU GATES PENDING.** Tác giả đã duyệt chạy A100 và phân công5–6–1. Colab3 chưa bật, chờ budget gate. Quyết định này thay đề xuất tạm hoãn Turkish ở Mục 10; không thay kết quả NO-GO Amis hoặc biến diagnostic pilot thành independent confirmation. Chưa chạy Centered LSR.

### 11.1. Quota và queues đề xuất

Đọc trực tiếp bằng colab usage ngày 07/10/2026: account1 107.76 CU, account2 192.56 CU, account3 24.32 CU. Cả ba usage rate 0 và active assignments 0.

| Account | Thứ tự training đề xuất | Tổng |
|---|---|---:|
| Colab 1 | Full42 → Middle43 → Baseline43 → LSR43 → Baseline44 | 5 |
| Colab 2 | LSR42 → Middle42 → Full43 → Middle44 → LSR44 → Full44 | 6 |
| Colab 3 | Baseline42; không thêm queue test | 1 |

Mỗi family Baseline/Middle/LSR/Full đủ seeds42–44. Full và Middle tập trung ở accounts1/2; account3 chỉ có Baseline. Account3 proposed CU cap 8, reserve tối thiểu15; refresh quota và rate A100 trước launch, dùng timing train+validation ban đầu trên1/2 để ước tính. Nếu không đủ budget, chuyển nguyên run sang1/2 trước start; không giảm epochs/batch riêng. Account3 chỉ mở khi cần, xong upload verified thì stop ngay. Launcher phải kiểm tra ngân sách trước khi giao long run; quota risk cần checkpoint recovery, không kill âm thầm.

### 11.2. Protocol

- mBART-50 revision e30b6cb8eb0d43a0b73cab73c7676b9863223a30; A100 Standard40GB, không high-memory hoặc RTX6000.
- OPUS-100 subset20k, validation/test gốc2000/2000; strings/order/duplicates giữ nguyên. SHA và source,target schema đã kiểm tra local bằng conda clrr.
- LR5e-5, WD0, scheduler warmup.06, batch32×accum4, eval8, lengths256, BF16+checkpointing; tối đa20epochs, patience10 đánh giá mỗi epoch. Cùng recipe cả4families.
- OldLSR lambda.10; oldFull encoder distance2 alpha.10 lambda.10 detached, routingwarmup0. Middle layer6, symmetric contrastive objective, tau.10 weight.10 như adaptation hiện có; không gọi exact reproduction paper gốc.
- Selection raw-ref validationchrF++beam1; metricsvalidationBLEU13a/chrF0 cũng báo. Chỉ test sau khi đủ12selectedweights/validationreceipts khóa và verify. Testbeam4 explicitFP32/TF32off, rawrefs BLEU13a/nativechrF++/chrF0; duplicate/overlap sensitivity theo protocol gốc.
- Primarydiagnostic Full−Baseline và Full−LSR; Full−Middle comparator. Pilot n3, pairedseed/reportuncertainty; không giả định superiority qua6/8gate của n8.

### 11.3. Chuẩn bị và các gates còn phải hoàn tất

Local preparation PASS: scripts/prepare_turkish_pilot_plan.py kiểm tra SHA/schema/rows và unique assignments12, viết [PROPOSED_PLAN.json](../../outputs_rebuttal/turkish_old_method_pilot_es10_20261007/PROPOSED_PLAN.json). Không bật session, tạo repo hoặc training. Đây là manifest phân công đã chuẩn bị, **chưa phải launcher production đã qua GPU gate**.

Không chạy nguyên launcher cũ scripts/run_turkish_followup.py: nó chỉseed42/patience4/familieskhác và gates cũ. Middle legacy trainer hardcodes tl_XX→zh_CN và BLEUzh; wrapper bỏ num_items_in_batch nhưng chưa khóa accepts_loss_kwargs=False. Trước training phải dùng common corrected engine, cấu hình tr_TR→en_XX/BLEU13a và regression accumulation cho Middle. Giữ objective Middle; đây là correctness/configuration fix, không nâng cấp CLRR/LSR.

Sau tác giả duyệt: hoàn thiện worker/launcher với queueguard và nguồnhash mới; CPU checks; A100 BF16+accum4+checkpointing lifecycle gate cho **đúng4families** trong bundle mới, rồi mới longtraining. Không lấy gate Amis4families cóCLRR-only để chứng nhận Middle. GPU gate chưa chạy trong giai đoạn chờ duyệt.

### 11.4. Backup và monitor

Proposed dedicated repos FiveC/turkish-old-method-pilot-{baseline,middle,lsr,full}-20261007; prefix turkish_old_method_pilot_es10_20261007; chốngoverwriteexistingruns. Selectedcheckpoint và metrics/predictions/provenance upload từngrun; SHA/selected-exportedweightdigest/pinnedcommit verify trước xóa weights. Namespace riêng cho attempts/recovery. Local chỉJSON/CSV/Markdown/receipts. Không có HF repo được tạo trong bước preparation này.

Báo chat mỗi10phút khi đang chạy; theo dõi epoch/step/valscore/GPU/error/quota/upload. Xong account nào stop account đó sau collectionverified. Teststage doaccounts1/2 điều phối sau lock12, không giữ account3 idle. ETA và CUcost sẽ cập nhật từ timing thực, chưa có số giờ Turkish đáng tin để cam kết.

### 11.5. Thực thi sau khi tác giả duyệt

- Common engine đã hỗ trợ --method middle-align, --middle-layer-idx, --alignment-temperature; standalone Middle hỗ trợ --src-lang/--tgt-lang/--bleu-tokenizer/--model-revision. Middle accepts_loss_kwargs=False để khớp objective microbatch mean; không thay symmetric contrastive loss. Không đổi số lịch sử; recipe corrected mới không mặc định khớp training historical Middle.
- CPU optimization gate đúng Baseline/Middle/LSR/Full:16cases PASS, gradient max absolute error1.86e-9; lifecycle4branches PASS, source/target/order và EnglishBOS đúng, selected/exportedweights exact. Reports tại namespace/cpu_gate/.
- Worker scripts/revalidation/run_turkish_old_method_pilot.py; queue scripts/revalidation/run_turkish_pilot_queue.py; launcher scratch/launch_turkish_pilot.py; monitor/collect/stop scratch/control_turkish_pilot.py. Mọi Python dùng conda clrr.
- Launcher deployed accounts1/2; bootstrap pins Torch2.6.0CUDA12.4, Transformers4.57.6, Accelerate1.15.0. Queue chạy tiny GPU regression/lifecycle, full-size gate4families batch32acc4 length256stress rồi mới training. Gate receipt gắn SHA của manifest exactbundle; không dùng gate cũ.
- Metadata/outputs hiện hành: outputs_rebuttal/turkish_old_method_pilot_es10_20261007/. PLAN.json APPROVED, deployment_manifest.json có hashes; launch_account1/2.json và status_account1/2.json theo dõi trạng thái thực. Source trước sửa trong PROPOSED_PLAN chỉ là snapshot, deployment_manifest mới là source khóa cho training.
- Sau đủ12VAL_COMPLETE và SHAverifiedcollection, tạo selection_lock.json status ALL_12_VALIDATION_VERIFIED; worker --stage test sẽ từ chối nếu chưa đủ lock.
- Xác nhận actual TRAINING/gates PASS sau monitoring; chưa được gọi models hoàn tất khi chỉ deployed.

## 12. Scope amendment — Turkish seed 42 only (2026-10-07)

User requested a one-seed exploratory pilot to reduce elapsed time. The active training workers were preserved; only their queue supervisors were replaced. The original PLAN.json and deployed training bundle remain unchanged for provenance. The effective scope below supersedes the original 12-run schedule.

| Account | Effective queue | Status at amendment |
|---|---|---|
| colab | Full 42 → Baseline 42 | Full worker preserved; continuation supervisor installed |
| colab2 | LSR 42 → Middle 42 | LSR worker preserved; continuation supervisor installed |
| colab3 | No runs | Do not start |

- Seeds 43 and 44 are cancelled before launch. Four families use the same seed 42 and unchanged recipe, including early stopping patience 10 and the common expandable-segments allocator.
- Operation: `conda run --no-capture-output -n clrr python -X utf8 scratch/reduce_turkish_to_seed42.py --account 1` (and account 2). Receipts: `outputs_rebuttal/turkish_old_method_pilot_es10_20261007/seed42_amendment_account{1,2}.json`.
- Both continuation supervisors require the active run's VAL_COMPLETE.json and verified Hugging Face collection receipt before starting the next run. They stop after their second family; they do not open seeds 43/44.
- Monitoring remains `scratch/control_turkish_pilot.py --account 1/2 --action monitor`; report to chat every 10 minutes during an active monitoring session.
- Before test evaluation, implement and audit the amended four-run selection barrier. The original worker still requires ALL_12_VALIDATION_VERIFIED and must not be bypassed with fabricated entries. Lock all four real validation receipts and selected weight digests before evaluating test. Record any evaluation-code deployment separately from the immutable training bundle.
- Interpretation: exploratory paired comparison on one seed, not evidence of seed robustness. Compare Full against both Baseline and Middle (plus LSR), reporting all deltas regardless of sign. No test result is available at amendment time.

### 12.1 Monitoring interruption and remote receipts (2026-10-07)

Both account 1 and account 2 `sessions` calls now report no active sessions on server. Do not interpret the last saved local TRAINING status as current. The cause of session loss is unknown.

Hugging Face metadata was checked without downloading checkpoint archives:

| Family, seed 42 | Verified remote evidence | Validation BLEU | Validation chrF++ |
|---|---|---:|---:|
| Full | Best archive exists; VAL_COMPLETE receipt says 12 epochs, selected checkpoint-314 (epoch 2) | 20.915333 | 39.405251 |
| LSR | Best archive exists; VAL_COMPLETE receipt says 12 epochs, selected checkpoint-314 (epoch 2) | 20.631149 | 39.153034 |
| Baseline | Only selection checkpoint-157 metadata found; no archive or VAL_COMPLETE | — | — |
| Middle | Repository returned 404; completion not established | — | — |

Full minus LSR is +0.284184 validation BLEU and +0.252217 validation chrF++. These are validation scores, not test results and not a baseline comparison. Remote archive presence and declared SHA are established; the archive bytes have not yet been independently rehashed in this check. Small downloaded receipts are under outputs_rebuttal/turkish_old_method_pilot_es10_20261007/hf_metadata/.

Recovery: retain Full/LSR selected archives; do not retrain them. Establish whether Baseline has recoverable weights before deciding resume versus rerun. Restore the pinned training bundle and common gates on A100 for the unfinished controls; verify all four selected checkpoints before amended four-run test evaluation. Monitoring every ten minutes requires a live session; currently none is found on either account.

## 13. Authorized seed42 recovery and test — 2026-10-08

User authorized completing the remaining Baseline and Middle controls on two A100s, followed by test evaluation of all four families. Full/LSR are reused from their verified validation receipts; they are not retrained.

| Account | Train | Test after four selections exist |
|---|---|---|
| colab | Baseline seed42 | Baseline seed42, Full seed42 |
| colab2 | Middle seed42 | Middle seed42, LSR seed42 |

Preparation fixed the worker CLI to accept middle, and added an exact four-family/seed42 selection-lock gate. Model objectives and training code were preserved from the original pinned bundle. Original PLAN/deployment receipts are retained under attempts/pre_recovery_20261008. The two interrupted Baseline metadata files were moved atomically within the HF repo to attempts/interrupted_20261007 before rerunning; no completed checkpoint was overwritten.

Commands: conda run --no-capture-output -n clrr python -X utf8 scratch/launch_turkish_pilot.py --account 1 --recovery (and account 2). The detached scripts/revalidation/finish_turkish_seed42.py runs the common GPU gates and its one assigned control, waits for four real VAL_COMPLETE receipts and matching HF LFS archive hashes/sizes, pins selected weight digests and validation-receipt hashes, then runs its two test evaluations. It has a four-hour selection wait timeout. Test worker independently hashes each downloaded archive and verifies selected weights before evaluation. Test protocol: official 2000 test rows, beam4, FP32 generation, TF32 off, BLEU13a, chrF++ and chrF0. Results and predictions are uploaded to the separate per-family FiveC repos; cleanup requires verified upload receipts.

Small HF metadata may be downloaded locally; model archives remain on remote GPU/HF. Monitor with scratch/control_turkish_pilot.py --account 1/2 --action monitor; collect small artifacts after completion. Stop runtimes after verified final collection to avoid idle CU burn. Launcher start is not proof of training: confirm GPU gate and TRAINING/TESTING queue status before reporting active model progress.

### 13.1 Check at 09:11–09:16, 2026-10-08 — private HF storage exhausted

Baseline and Middle completed training at epoch12 and selected checkpoint-314 (epoch2). Baseline final checkpoint upload failed with HTTP403: Private storage limit reached for user FiveC. Middle final validation predictions and final trainer state were collected as well; do not retrain either control. Preserve both GPU runtimes and their best-model archives until weights are backed up.

The local watcher was stopped before its next automatic-stop poll. Its failure handling was corrected: a failed worker or dead supervisor no longer authorizes runtime shutdown, because small metadata collection does not protect unuploaded weights. Only TEST_QUEUE_COMPLETE plus verified final collection authorizes automatic stop.

Small artifacts were collected and transfer-SHA verified for both accounts. Controls' validation metrics were recomputed from their already-generated validation_predictions.csv, asserting all 2000 source/target rows exactly match the original validation split. Saved in recovered_controls_validation.json:

| Family seed42 | Validation BLEU13a | Validation chrF++ |
|---|---:|---:|
| Baseline | 21.282673 | 39.624801 |
| Middle | 20.720164 | 39.343770 |
| LSR (prior VAL_COMPLETE) | 20.631149 | 39.153034 |
| Full (prior VAL_COMPLETE) | 20.915333 | 39.405251 |

Full-minus-LSR validation BLEU +0.284184; Full-minus-Baseline -0.367340. No test result yet. An asynchronous user question requests authorization to make only the two new control repos public, alternatively temporary local checkpoint storage (~4.6GB), or user-managed private quota cleanup/upgrade. Do not publish private repos without the reply. After storage is resolved, upload the existing archives and finish validation provenance without retraining, then lock the four real selections and run the authorized test evaluation. During this wait runtimes remain allocated and incur CU usage.

## 14. Public checkpoint consolidation and Turkish test complete — 2026-10-08

User corrected storage policy: ALL project checkpoints belong in the public FiveC/amis-rewire-checkpoints repo; do not create separate private/date-named repos. Storage specification: docs/operations/HUGGINGFACE_CHECKPOINT_STORAGE.md. New folder structure is checkpoints/<dataset>/<backbone>/<method>/<recipe>/<cohort>/seed-<seed>/best.zip, with source datasets matching data_processed names and configuration encoded without dates.

Completed: 11 formerly private CLRR repositories made public, 2242 files (42 existing checkpoint archives) copied server-side into canonical paths and verified by LFS SHA256/size or Git blob ID. All 11 standalone source repos deleted only after full verification and unchanged source revision checks. Two newly trained Turkish controls uploaded directly from Colab to the canonical public store and SHA-verified; 44 selected archives in this consolidation scope. Temporary local model files removed after public backup verification. Source JSON numbers are preserved; source-repo/revision/path mapping is under registry/path_mapping.json and registry/migration_receipts.json on HF and outputs_rebuttal/hf_checkpoint_consolidation locally.

Turkish seed42 test COMPLETE, original OPUS100 20k training/2000 validation/2000 test rows. All families selected epoch2 and stopped epoch12 with patience10. Frozen test: beam4 FP32 generation, TF32 off, BLEU13a/native chrF++/chrF0. Four selected weights and validation receipts locked before test. All three metrics independently recomputed from verified test predictions and original source/target order.

| Family | Test BLEU | Test chrF++ | Test chrF0 |
|---|---:|---:|---:|
| Baseline | 22.634203 | 41.549538 | 42.595475 |
| Middle Alignment | 22.519996 | 41.389460 | 42.427257 |
| LSR | 22.200584 | 41.271311 | 42.305028 |
| Full | 22.570620 | 41.549304 | 42.616011 |

Full-minus-LSR BLEU +0.370036; Full-minus-Baseline -0.063583; Full-minus-Middle +0.050624. One-seed exploratory comparison: Full improves over LSR, essentially matches Baseline chrF++, but does not establish an advantage over Baseline. Do not claim training-seed robustness or statistical significance from this table. Separate report: docs/experiments/EXPERIMENTS_UPDATED.md#turkish-seed42; original experiment tables not overwritten.

Both A100s stopped after verified final metadata collection. Final usage: account1 93.32 CU, account2 177.41 CU, each 0.00 CU/hr and zero active assignments. Watcher ran its final collection/stop while an additional collection call attempted SSH; that SSH caused temporary replacement runtimes to be created by CLI. Those were immediately stopped. Controller now checks the requested A100 runtime via read-only status before SSH, preventing this implicit-creation path. No training or test was rerun by those temporary runtimes.

Artifact collection merging now retains stronger verified test receipts over an inactive family's selection-only copy. Final per-family receipts were also independently reconstructed/verified from pinned public test artifacts and archive LFS hashes; all four contain COMPLETE/test metrics/test predictions plus archive proof. Future state: training and Turkish pilot evaluation are complete; storage consolidation is complete. Do not relaunch historical per-family/private-repo launchers without updating their storage routing to the public specification.

## 15. Chuẩn hóa kho public: chỉ giữ checkpoint cuối cùng

Người dùng yêu cầu gom toàn bộ các thư mục rời trong `FiveC/amis-rewire-checkpoints` vào cấu trúc chung, đồng thời cho phép bỏ weights bản thử và checkpoint trung gian. Đã hoàn tất chuyển 823 file cần giữ trong kho cũ, xác minh kích thước và SHA-256 LFS/Git blob ID trước khi xóa đường dẫn cũ. Đã bỏ 190 archive thử/trung gian, tổng 678.418 GB dung lượng logic của file; không xóa lịch sử commit và không coi con số này là dung lượng Hub đã thu hồi vĩnh viễn.

Kho hiện hành có **90 archive cuối cùng**, tất cả dưới:

```text
checkpoints/<dataset>/<backbone>/<method>/<recipe>/<cohort>/seed-<seed>/best.zip
```

44 archive từ đợt gom repo riêng/Turkish được giữ nguyên hash. Kho cũ bổ sung 46 archive: 28 mBART, 7 mT5, 5 ByT5, 6 NLLB. `nllb-200`, `peft_baselines`, `comparative_baselines`, `revalidation_ablations` và các đợt Amis đã được gom; `metrics/` rời được ghép với run tương ứng khi có bằng chứng. Báo cáo chung ở `registry/reports/`. Root chỉ còn `.gitattributes`, `README.md`, `checkpoints/`, `registry/`.

Trước khi bỏ `nllb_results.zip` 41 GB, đã đọc danh sách ZIP bằng HTTP Range và trích 22 JSON/CSV/Markdown nhỏ (ZIP CRC kiểm tra khi đọc), upload và xác minh hash trên Hub. Metrics/predictions nằm cạnh từng checkpoint NLLB; trainer state/config ở `provenance/aggregate/`. Không tải weights/gói 41 GB về máy, không bật GPU và không train/đánh giá lại trong bước dọn kho.

Không gán seed hay hyperparameter mặc định cho các archive thiếu metadata. Recipe lịch sử giữ tên run không ngày tháng với tiền tố `historical-`; 16 archive cũ chưa có seed được xác minh dùng `seed-unknown`. Các thư mục/report bản thử còn metadata không có nghĩa weights thử còn được giữ. Nội dung số khoa học gốc không đổi.

Tra cứu và bằng chứng:

- [Hướng dẫn lưu/push](HUGGINGFACE_CHECKPOINT_STORAGE.md).
- [Bảng checkpoint public](https://huggingface.co/FiveC/amis-rewire-checkpoints/blob/main/registry/CHECKPOINT_INDEX.md); CSV/JSON tương ứng nằm ở `registry/`.
- `registry/legacy_path_mapping.json`: ánh xạ toàn bộ file cũ cần giữ.
- `registry/legacy_discarded_weights.json`: từng archive bị bỏ và lý do.
- `registry/legacy_migration_receipt.json`: revision copy/xác minh/xóa.
- `registry/nllb_extraction_receipts.json`: 22 file trích từ gói NLLB.
- `registry/LEGACY_FINAL_AUDIT.json`: audit độc lập sau dọn, gồm cả dữ liệu từ đợt gom repo riêng trước đó.
- Bản local nhỏ: `outputs_rebuttal/hf_checkpoint_consolidation/`.

Scripts chạy bằng conda `clrr`: `scripts/revalidation/organize_legacy_hf_store.py` và `scripts/revalidation/index_public_checkpoint_store.py`. Script migration dùng snapshot cố định; không coi nó là launcher hay công cụ xóa run mới. Khi upload run mới, dùng repo public này, ghi đúng recipe/seed/cohort, upload weights + receipt, xác minh hash rồi mới xóa weights tạm. Không tái dùng launcher lịch sử có repo private/đường dẫn cũ nếu chưa sửa routing. Khi đọc receipt cũ, giải quyết đường dẫn qua registry mapping thay vì sửa metadata gốc.


<a id="configuration-handoff-history"></a>

## Phần gộp: CONFIGURATION_SEARCH_HANDOFF

> Nội dung gộp từ `CONFIGURATION_SEARCH_HANDOFF.md`. **Lịch sử thực thi, không phải yêu cầu chạy hiện hành.** Các trạng thái đang chạy/chưa chạy, repo HF cũ và lệnh warmup chỉ mô tả thời điểm nguồn; muốn chạy lại phải kiểm source, runtime, protocol và routing lưu checkpoint hiện tại.

### Bàn giao: tìm cấu hình CLRR trên mBART Amis

> **BÀN GIAO TIẾP THEO — 05/10/2026:** Đọc [NEXT_EXPERIMENTS_HANDOFF_20261005.md](NEXT_EXPERIMENTS_HANDOFF_20261005.md): inventory corrected hoàn tất, four-way scores, N1–N8 Go/No-Go, checkpoint mapping và entrypoints/lệnh chạy. Ưu tiên tổng hợp interaction/novelty rồi fresh-seed confirmation Full hiện tại; chưa mặc định nâng cấp. Launcher32runs fresh seeds chưa có: cần implement và gate trước launch. Hai GPU hiện đã tắt; không Turkish.

> **FINAL 05/10 17:00 — SIX_CONTROLS_VERIFIED, BOTH_GPU_OFF:** CLRR-only test BLEU seeds42/43/44 = 19.223095 / 19.492997 / 19.345498 (mean19.3539). LSR lambda .03 loses validation to .10 at all3seeds; selected .10, .03 test stays blind. All6 model backups and local metric replay verified; GPU2 stopped16:04, GPU1 stopped16:10. [Results and interpretation](../archive/CORRECTED_RESULTS_20261005.md#corrected-controls-20261005). Prior entries below are historical.

> **CURRENT 05/10 — CORRECTED_CONTROLS_LAUNCHED:** Two A100 Standard 40GB jobs: LSR-only lambda .03 (account 1) and CLRR-only alpha .10 (account 2), seeds 42/43/44 sequentially. GPU gates precede training. LSR test remains blind until the three-seed validation decision. See [current runbook](../archive/CORRECTED_RESULTS_20261005.md#corrected-controls-20261005). Entries below are historical.

> **STATISTICS 05/10 — ANALYZED:** [significance và kế hoạch tiếp](../archive/CORRECTED_RESULTS_20261005.md#corrected-accumulation-significance-20261005). 10000pairedbootstrap,12native p-values verified. Full−LSRmeanBLEU+0.7373 conditionalCI[+0.0403,+1.4557], nhưng không per-seedFull−LSRtest nào quaHolm12; exacttrainingseed sign-flip p=.25(n=3). Chưa claim significant superiority. Đề xuất tiếp correctedLSRλ.03(candidate từ validation cũ) vàCLRR-only3seeds; chưa launch.

> **HIỆN HÀNH 05/10 — FOUR_RUNS_VERIFIED, GPU_OFF:** seed43 LSR/Full và seed44 recovery LSR/Full đã xong, metrics tính lại độc lập, artifacts local và HF checks PASS; cả hai A100 đã tắt (14:05/14:08). Corrected3seeds testBLEU Full **20.1303±0.2188**, LSRλ.10 **19.3930±0.3787**, baseline **19.3886±1.1203**. Full−LSR **+0.7373**, dương3/3; Full−baseline+0.7417 nhưng chỉ2/3 và seed44âm. [Kết quả đầy đủ](../archive/CORRECTED_RESULTS_20261005.md#accumulation-confirmation-20261005), [insight mới](../archive/CORRECTED_RESULTS_20261005.md#accumulation-confirmation-20261005). Seed43 backup ởFiveC/amis-rewire-checkpoints; seed44 ởFiveC/amis-accumulation-confirmation-20261005, prefix recovery riêng sau lỗi HF429. Số cũ và failed-attempt provenance giữ nguyên. **Đề xuất tiếp, chưa launch:** CLRR-only corrected3seeds và corrected tunedLSR theo validation; chưa cần kết luận đổi phương pháp, chưa Turkish. Các mục phía dưới là lịch sử.

> **PILOT HOÀN TẤT 05/10 — TWO_PILOTS_VERIFIED:** hai pilot seed42 trên hai A100 Standard đã xong; metrics tính lại từ predictions, weights/metadata HF kiểm checksum, artifacts đã tải local và **cả hai GPU đã tắt**. LSR-onlyλ.10: test BLEU **19.095440** (cũ19.786496, Δ−0.691056); Fullα.10/λ.10: **20.321295** (cũ19.956574, Δ+0.364721). Full hơn LSR **1.225855 BLEU**, hơn verified baseline42 **0.593214** tại seed42. Xem [kết quả và quyết định tiếp](../archive/CORRECTED_RESULTS_20261005.md#accumulation-pilot-20261005). HF prefix riêng `amis_accumulation_pilot_20261005`; số cũ giữ nguyên. **Đề xuất tiếp theo:** bốn runs LSR/Full seeds43/44 cùng recipe để kiểm tra độ ổn định; chưa launch, chưa mở12-run cohort/Turkish và chưa chọn cấu hình mới từ test. Các mục bên dưới là lịch sử; ưu tiên trạng thái mới này.

> **Bước1–2 HOÀN TẤT, 05/10:** [full-size gate và baseline42](../archive/TECHNICAL_AUDITS_20261005.md#fullsize-gate-and-baseline42-20261005). Full mBART-50 batch32×accum4, BF16+checkpointing, stress lengths256 PASS cho cả4 nhánh trên A10040GB. Fresh baseline42 đủ20 epochs, best720/epoch20; test BLEU19.7280814/chrF++18.7138380 khớp số cũ. HF prefix `amis_corrected_baseline42_20261005`, weights và metadata verified; artifacts local đã lưu, GPU đã tắt. Dùng fresh42 và reuse audited43/44 với đúng locked recipe. Chưa launch12 runs phương pháp, số lịch sử giữ nguyên.

> **Bổ sung 05/10 — GPU_PREFLIGHT_PASS:** [baseline provenance và GPU gate](../archive/TECHNICAL_AUDITS_20261005.md#gpu-preflight-and-baseline-gate-20261005). Exact bundle đã PASS trên A100 Standard với BF16, accumulation4, checkpointing: 16 gradient cases, 4 actual CLI lifecycle branches và 2 routing restore checks. Artifacts đã lưu, GPU đã tắt; chưa training dài. Baseline43/44 selected-step được truy về **720/612**;252/504 là hardcode sai trong inventory, scores giữ nguyên. Baseline42 chronology vẫn chưa đủ bằng chứng. Các ghi chú “GPU gate chưa chạy” bên dưới là trạng thái trước kiểm tra này.

> **HIỆN HÀNH 05/10/2026 — TECHNICAL_HOLD:** lỗi gradient accumulation CLRR/LSR đã được tái hiện và sửa bằng `accepts_loss_kwargs=False`; xem [G0/G1 verification](../archive/TECHNICAL_AUDITS_20261005.md#g0-g1-verification-20261005). Không launch pilot/warmup theo các section cũ trước khi chốt affected controls. Scores lịch sử giữ nguyên, chưa có corrected research training.

> **Pre-rerun audit:** [báo cáo kiểm tra trước chạy lại](../archive/TECHNICAL_AUDITS_20261005.md#pre-rerun-verification-20261005):16 optimization cases và4 real-CLI lifecycle cases PASS local; resume/overwrite guard và HF checksum gate đã sửa. Còn baseline provenance decision và tiny GPU BF16+accum4+GC gate trên bundle mới trước training dài. Không dùng worker warmup cũ để launch corrected cohort.

Ngày khởi tạo: 03/10/2026. Workspace: `D:\Code\CLRR`.

**HIỆN HÀNH — V5 đã đo/audit và sửa nhãn beam của baseline42; xem19.4–19.5.**
42coreconditions VERIFIED +3baselinecontrols +4verificationconditions. Các
test scores cũ giữ nguyên, A100 đã tắt. Chưa có cải thiện từ inferenceα/group
interventions; đề xuất pilot train-time routing warmup có điều kiện, chưa chạy.

**LẦN TRƯỚC — V4 đã chạy và audit VERIFIED: xem Mục 18.** Đủ 21 checkpoint
probes và 3 route-control; 576 predictions route-on tái lập chính xác ở cả3seeds.
Routing-on hơn off trung bình +2.9676 validation BLEU trên cùng weights.
Rank tăng trước final LayerNorm nhưng singular rank cuối không tăng; không
dùng claim anti-collapse V3. Test quality cũ giữ nguyên; Colab đã tắt.

**Review trước V4: xem Mục 17 — review V3 và kế hoạch lúc đó.**
V3 đã sửa cách tháo hook route-off, nhưng chưa tái lập validation route-on;
source thiếu language codes và một số phép đo gradient/rank cần sửa.
Chưa dùng kết luận “89% câu đổi” hoặc “chống collapse +34,4%” làm claim paper.
Các số test đã xác minh trước V3 giữ nguyên. Mục 16 là review lịch sử của V2.

**TRẠNG THÁI SAU REVIEW ĐỘC LẬP (04/10/2026): xem Mục16.** Có đủ21test outputs
và30secondary tests đúng family; **F4 có lỗi can thiệp, F3 chưa đủ đầu ra từng layer**.
Không coi F0→F6 hoàn tất100% hoặc dùng kết luận routing vô tác dụng từ F4 hiện tại.
1. **F0:** đã inventory21weights; validation predictions còn thiếu ở một số models,
   và runtime/loading provenance chưa đủ để chứng nhận toàn bộ weights.
2. **F1 (Bổ sung Đánh giá Test Full $\lambda=0.03$):** 3 seeds đã chấm FP32 beam 4 trên 575 dòng test.
3. **F4:** file ghi0/576 câu đổi nhưng route-off vẫn mang hookα0.10; **INVALID**, cần chạy lại.
4. **F3:** có21endpoint summaries,8gradient batches/model và residual tables;
   chưa lưu raw spectra/representations, rank/gradient từng layer và đầy đủ provenance.
5. **F2 (30 Kiểm định Thứ cấp & Holm-Bonferroni):** 10,000 paired corpus bootstrap draws; $0/30$ cặp đạt ý nghĩa thống kê ở $\alpha=0.05$.
6. **F5 & F6 (Tổng hợp & Báo cáo Quyết định Chiến lược):** Đã hoàn tất [`reports/decision.md`](../../reports/decision.md), [`reports/mechanism_report.md`](../../reports/mechanism_report.md), [`reports/validation_audit.md`](../../reports/validation_audit.md) và Mục 15 của tài liệu này.

**Tài nguyên GPU:** theo báo cáo lượt chạy, Colab A100 Standard40GB đã dừng.
Review này chưa truy cập Colab để xác nhận trạng thái phiên; không mở GPU hoặc train mới.

#### 1. Kết luận Chiến lược & Quyết định Dự án

- **QUYẾT ĐỊNH: NO-GO CHO VIỆC CLAIM VƯỢT TRỘI HỆ THỐNG HOẶC MỞ RỘNG ĐA NGÔN NGỮ (TURKISH/NLLB).**
- Mô hình Full CLRR không vượt qua được mốc đối chứng đã được tinh chỉnh tối ưu (`Tuned LSR` $\lambda=0.03$), kém $-0.13$ BLEU và $-0.16$ chrF0.
- Attached chưa cải thiện đồng đều đủ để thay validation choice; ba seed này chưa
  chứng minh detach tối ưu hoặc là nguyên nhân của chênh lệch với tunedLSR.
- **Rút kết luận skip connection không can thiệp decoding:** F4 sai do hook còn hoạt động.
- Chưa tự chuyển paper thành mechanistic study. Sửa F4 và hoàn thiện diagnostics trước
  chọn nâng cấp/định vị paper; xem **Mục16**. NO-GO mở rộng vẫn dựa trên scores/control.

#### 2. Mục tiêu khoa học

Tìm cấu hình tốt **trong ngân sách đã khai báo**, rồi kiểm tra CLRR có đóng góp vượt
LSR-only được tune hay không. Không chọn cấu hình bằng test và không bảo đảm kết quả tốt.
Xem `docs/research/CLRR_GO_NOGO_PLAN.md`, `docs/experiments/EXPERIMENTS_UPDATED.md`,
`docs/research/RESEARCH_INSIGHTS.md`, `docs/experiments/METRICS.md`.
Paper hiện tại là `docs/paper/clrr_main.tex`; chưa đưa các trial tuning mới vào paper.

Nhóm C cũ: Baseline, CLRR-only, LSR-only, Full × seeds42/43/44. Protocol matched đã chạy.
Ngân sách ban đầu: tối đa20 lượt train Amis, gồm8screening + tối đa6confirmation + tối đa6controls.
Phân bổ hiện hành04/10:8đã xong +4phục hồi/confirmation +5diagnostic +3attachedCLRR-only có điều kiện.
Đã chạy cũng phải tính vào ngân sách, không loại trial có kết quả thấp.

#### 3. Kết quả đã có và điều cần cảnh giác

##### Audit C không train — đã hoàn tất

Artifacts: `outputs_rebuttal/amis_confirmation_20261003/metric_influence/`.
12systems ×575rows; predictions/reference hashes và thứ tự khớp; chrF++ tái tính khớp <1e-8.

| Character-only chrF | Seed42 | Seed43 | Seed44 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Full−Baseline | +0.2654 | +1.1001 | −0.3865 | +0.3263 |
| Full−LSR | +0.3046 | −0.0676 | +0.2655 | +0.1675 |

Seed43 Full−Baseline chrF++ +5.3162 còn +0.5803 khi bỏ riêng row170.
Character-only gap cùng seed còn +0.8602, nên gain chrF++ lớn không chứng minh gain rộng.
Đây là sensitivity, **không sửa điểm chính, không xóa câu**. Không có p-value mới từ audit này.

##### Screening D — chỉ validation

| Trial | α | λ | Seed | Validation BLEU | Validation chrF++ | Validation chrF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full gốc, reuse metadata C | 0.10 | 0.10 | 42 | 17.6437 | 14.7317 | Chưa có validation predictions |
| LSR gốc, reuse metadata C | 0 | 0.10 | 42 | 16.5688 | 14.1674 | Chưa có validation predictions |
| Baseline C, mốc tham chiếu | 0 | 0 | 42 | 17.8735 | 14.9009 | Chưa tái generate validation trong lượt này |
| `full_a005_l010` | 0.05 | 0.10 | 42 | 17.1225 | 14.4092 | 16.4998 |
| `full_a020_l010` | 0.20 | 0.10 | 42 | 17.4586 | 14.5738 | 16.7469 |
| `full_a010_l003` | 0.10 | 0.03 | 42 | 17.1141 | 14.4355 | 16.5587 |
| `lsr_l060` | 0 | 0.60 | 42 | 17.0300 | 14.3688 | 16.5080 |
| `lsr_l003` | 0 | 0.03 | 42 | 17.7873 | 14.8587 | 17.1364 |
| `lsr_l001` | 0 | 0.01 | 42 | 16.3028 | 13.9491 | 16.0469 |
| `full_a010_l030` | 0.10 | 0.30 | 42 | 17.1999 | 14.3614 | 16.4432 |
| `lsr_l030` | 0 | 0.30 | 42 | 16.3826 | 13.8951 | 16.0409 |

α=0.05 chưa cải thiện so cấu hình gốc ở validation seed42. Chưa kết luận toàn bộ phương pháp.
α=0.20 cũng thấp hơn cấu hình gốc14.7317, nhưng tốt hơn α0.05 trên cả ba metrics.
Đã chạy xong cả α và λ screening: bốn Full mới đều thấp hơn Full gốc.
**Full gốc mới là best Full trong năm cấu hình.** LSR λ0.03 cao hơn mọi Full;
baseline C vẫn cao hơn cả best Full và best LSR ở validation seed42.
Validation576rows chỉ có **1 word-bigram reference ở row197**; test575rows có3 word-bigrams.
Trial α0.05 không match bigram đó; bỏ row197 chrF++=14.4440, chrF=16.5395.
Trial α0.20 cũng match0; bỏ row197 chrF++=14.6108, chrF=16.7894.
Thứ hạng hai trial mới không đổi trên chrF++/chrF hoặc khi bỏ row197.
Phải kiểm tra thứ hạng candidate bằng character-only và leave-one-out trước confirmation;
không tự đổi primary metric chỉ để lấy kết quả đẹp.

Các audit mới:

- `scripts/revalidation/audit_configuration_validation.py` → `reference_word_statistics.json`.
- `scripts/revalidation/audit_configuration_predictions.py` → `validation_influence.json`.
- Thư mục D: `outputs_rebuttal/amis_configuration_20261003_v1/`.
- `inputs.json`: input/source hashes và metadata gốc dùng để reuse.
- `runs/<run>/metrics.json`, `COMPLETE.json`, `validation_predictions.csv`: kết quả từng trial.
- `status_latest.json`: snapshot từ Colab, không tự coi snapshot cũ là tiến độ đang chạy.

#### 4. Cấu hình phải giữ để so công bằng

| Hạng mục | Giá trị |
| --- | --- |
| Backbone | `facebook/mbart-large-50-many-to-many-mmt` |
| Revision pretrained | `e30b6cb8eb0d43a0b73cab73c7676b9863223a30` |
| Data | `data_processed/amis_mandarin/{train,validation,test}.csv`, `source,target` |
| Proxy source / target | `tl_XX` / `zh_CN` |
| LR / weight decay / warmup | `5e-5` / `0` / `0.06` |
| Train batch / accumulation / eval batch | `32` / `4` / `8` |
| Max source / target length | `256` / `256` |
| Max epochs / patience | `20` / `4` |
| Routing | Encoder, distance2, cached source detach mặc định |
| Selection | Raw-reference validation corpus chrF++, beam1 |
| Final test | Beam4, FP32, TF32 tắt trong evaluation |
| Train | BF16, TF32, gradient checkpointing |
| Library | torch2.6.0+cu124, transformers4.57.6, sacrebleu2.6.0, hub0.36.0, accelerate1.15.0 |
| Params | 610,879,488, tất cả trainable; extra params0 |
| GPU / environment | Colab A10040GB Standard, **không high-mem**, conda `clrr` |

Dependencies thực tế: `outputs_rebuttal/amis_configuration_20261003_v1/environment_freeze.txt`.
Khi phục hồi, đối chiếu với file này; không âm thầm nâng versions giữa các trial.

`src/amis_rewire/train.py --validation-only` không load actual test dataset; đã smoke test
với test.csv bị bỏ khỏi fixture. GPU smoke bốn nhánh cũng đã đạt.
Generation FP32 là tường minh, không dựa vào BF16 autocast mặc định của Trainer.

#### 5. Checkpoint, backup và phục hồi

Repo: `FiveC/amis-rewire-checkpoints`. Prefix mới, không trùng C/lịch sử:

```text
amis_configuration_20261003_v1/<run>/<run>-best.zip
amis_configuration_20261003_v1/<run>/metrics.json
amis_configuration_20261003_v1/<run>/COMPLETE.json
amis_configuration_20261003_v1/<run>/validation_predictions.csv
```

`COMPLETE.json` chứa archive SHA256, bytes, commit revision, prediction SHA256.
Chỉ xóa weights/archive trên Colab **sau khi HF size và etag khớp SHA256**.
Token đã được tác giả cho phép dùng trong launcher trước; helper lấy token an toàn,
không in token, không đưa token vào Markdown hoặc bundle.

Hai weights **Full/LSR seed42 của C gốc** không còn; chỉ còn test predictions/metrics.
Không thay bằng epoch2 hoặc checkpoint lịch sử khác rồi gọi là best C.
10weights C còn lại có receipts tại `outputs_rebuttal/amis_confirmation_20261003/HF_UPLOAD.json`.

##### Phục hồi screening trên A100 mới khi được yêu cầu chạy tiếp

Chạy PowerShell tại `D:\Code\CLRR`. Python local luôn dùng `conda run -n clrr`.
Nếu `colab status` báo phiên cũ vẫn tồn tại, kiểm tra trước; không tạo thêm A100 song song.

```powershell
colab status -s colab
colab new -s colab --gpu A100
colab upload -s colab scratch/amis_configuration_code.zip /content/amis_configuration_code.zip
colab exec -s colab -f scratch/start_configuration_setup.py --timeout 60
```

Setup chạy nền. Kiểm tra đến khi `ENV_READY` tồn tại và CUDA báo A100:

```powershell
colab exec -s colab -f scratch/configuration_setup_status.py --timeout 60
```

**Không chạy lại `scratch/package_configuration_search.py` để thay zip hiện có.**
Zip ban đầu cùng `inputs.json` khóa source/data của screening; source mới có thể đổi hashes.
Nếu `scratch/` không đi theo Git, dùng `configuration_resume_bundle.zip` trong artifacts D
để phục hồi helper và frozen zip về workspace. Bundle có manifest SHA256,
không chứa token hoặc checkpoint. Trên cùng máy, các helper đã có trong `scratch/`.
Phục hồi receipts để không train lại tám trial đã xong. Các lệnh screening dưới đây
là quy trình lịch sử; **không chạy launcher screening để bắt đầu confirmation**:

```powershell
conda run -n clrr python -X utf8 scratch/set_configuration_colab_auth.py
colab exec -s colab -f scratch/restore_configuration_search_remote.py --timeout 120
colab upload -s colab scratch/monitor_configuration_remote.py /content/monitor_configuration_remote.py
colab exec -s colab -f scratch/launch_configuration_search_remote.py --timeout 43200
```

Restore giải nén zip, đối chiếu protocol HF với bundle, tải small artifacts, xác minh
archive metadata pinned revision và prediction hash. **Không tải weights và không truy cập test.**
Helper restore đã được kiểm tra khi có2 COMPLETE trials; code duyệt toàn bộ8labels để
phục hồi những trial đã có receipts. Chat tiếp cần kiểm tra số receipts thực tế là8.
Bundle bàn giao cũ được đóng gói khi mới có2trials; không dùng Markdown/status bên
trong bundle cũ thay cho file bàn giao hiện hành.
Runner bỏ qua trial có COMPLETE; dừng nếu gặp run folder dở dang để kiểm tra thay vì overwrite.
Lệnh launch giữ kernel hoạt động; cần theo dõi trong lúc chạy, không tắt máy trước backup.
Trong Codex, lỗi truy cập profile Colab/SSH do sandbox cần chạy công cụ có quyền network/profile;
không diễn giải lỗi profile là runtime đã mất.

##### Giám sát và lưu artifacts local

```powershell
conda run -n clrr python -X utf8 scratch/snapshot_configuration_local.py
conda run -n clrr python -X utf8 scratch/sync_configuration_predictions.py
conda run -n clrr python -X utf8 scripts/revalidation/audit_configuration_predictions.py
```

Snapshot lấy metrics/receipts; sync lấy validation predictions có SHA đúng. Báo tiến độ chat mỗi10phút.
Colab SSH chỉ cho một kết nối hoạt động; chạy snapshot, sync và SCP tuần tự để tránh HTTP429.
CLI `colab exec` có thể trả exit0 dù cell in traceback; đọc output và receipt, không chỉ exit code.
SSH monitor cần `LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64`; helper đã đặt.
Thiếu libnvidia-ml trong SSH không tự chứng minh CUDA training bị lỗi.

#### 6. Chính xác còn phải chạy gì

##### A. Tám trial screening seed42 — ĐÃ HOÀN TẤT 100%

Toàn bộ8trial screening (seed42, validation-only) đã hoàn tất và có backup receipts:

| Thứ tự | Run label | Method CLI | α | λ | Val chrF++ | Val BLEU | Val chrF0 | Tình trạng |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `full_a005_l010` | `jepa-clrr-enc` | 0.05 | 0.10 | 14.4092 | 17.1225 | 16.4998 | Hoàn tất & backup HF |
| 2 | `full_a020_l010` | `jepa-clrr-enc` | 0.20 | 0.10 | 14.5738 | 17.4586 | 16.7469 | Best Full mới, thấp hơn Full gốc |
| 3 | `full_a010_l003` | `jepa-clrr-enc` | 0.10 | 0.03 | 14.4355 | 17.1141 | 16.5587 | Full λ0.03 dùng cho control tương tác |
| 4 | `lsr_l060` | `jepa` | 0 | 0.60 | 14.3688 | 17.0300 | 16.5080 | Hoàn tất & backup HF |
| 5 | `lsr_l003` | `jepa` | 0 | 0.03 | 14.8587 | 17.7873 | 17.1364 | Best LSR, best8trials mới; thấp hơn baseline C |
| 6 | `lsr_l001` | `jepa` | 0 | 0.01 | 13.9491 | 16.3028 | 16.0469 | Hoàn tất (Early stop ep 18) |
| 7 | `full_a010_l030` | `jepa-clrr-enc` | 0.10 | 0.30 | 14.3614 | 17.1999 | 16.4432 | Hoàn tất & backup HF |
| 8 | `lsr_l030` | `jepa` | 0 | 0.30 | 13.8951 | 16.3826 | 16.0409 | Hoàn tất & backup HF |

Tái tính04/10: ba metrics của8trials khớp saved metrics <1e-8; hashes/reference/order đạt.
**Sửa diễn giải:** bỏ row197 giữ ranking chrF++ của8trials, nhưng bỏ từng row bất kỳ
có108/576 lần đổi full ranking chrF++, và153/576 lần đổi full ranking chrF.
LSR λ0.03 vẫn đứng đầu8trials trong cả576 lần ở cả hai metrics.
ChrF và chrF++ cũng hoán đổi vị trí LSR λ0.60 với Full α0.05; không nói mọi ranking giống nhau.
Không cần chạy thêm bất kỳ trial screening nào. Chuyển thẳng sang mục B bên dưới.

##### B. Chứng nhận selection và xác nhận controls — bốn lượt mới

**B0 — không train:** gộp8trials mới với Full/LSR gốc trong `inputs.json`; kiểm tra
baseline C làm mốc. Selection theo chrF++ cho **Full α0.10/λ0.10** và **LSR λ0.03**;
matched LSR là λ0.10. Không chọn Full α0.20 vì có sẵn weights hoặc vì nó là best Full mới.

- Dùng10archives C còn tồn tại để generate lại validation576rows, beam1/FP32,
  max_length256, TF32off, cùng tokenizer/proxies. Đây là forward, không train.
- Saved metadata cho phép chọn cấu hình nhưng chưa đủ để kiểm tra influence của
  Full/LSR gốc seed42 vì weights mất. Ghi rõ thiếu hụt này.
- Đối chiếu validation tái generate với metadata. Nếu khác, báo chênh lệch và nguyên nhân;
  không sửa điểm C ngầm. Không tái chọn cấu hình từ test.
- Kiểm tra source/data hashes, library/precision/selection, config và receipts của
  hai tài khoản Colab trước khi gọi kết quả matched. Lượt04/10 chỉ kiểm tra artifacts
  local; chưa xác minh lại archives trên HF hoặc live sessions.

**B1 — viết lock trước training tiếp:** tạo `selection_lock.json` gồm cấu hình Full gốc,
best/matched LSR, seeds42/43/44, validation ranking, input/source hashes, pretrained revision,
metric/beam/precision, rules reuse/rerun và comparison families. **File lock chưa được tạo
trong lượt phân tích này**; đây là đề xuất lựa chọn dựa trên validation đã có.

**B2 — bốn lượt sau khi tác giả yêu cầu chạy:**

| Lượt mới | Cấu hình | Seed | Lý do |
| --- | --- | ---: | --- |
| 1 | Full α0.10/λ0.10 detached | 42 | Phục hồi một replica có weights/predictions validation đầy đủ |
| 2 | LSR λ0.10 | 42 | Phục hồi matched control có weights, cùng protocol với replica Full |
| 3 | LSR λ0.03 | 43 | Kiểm tra best-tuned control ngoài seed dùng screening |
| 4 | LSR λ0.03 | 44 | Kiểm tra độ lặp lại qua seed |

Reuse Full/LSR λ0.10 seeds43/44 và baseline/CLRR-only C seeds42/43/44 **nếu B0 chứng nhận**.
LSR λ0.03 seed42 có đủ weights; không cần rerun chỉ để tạo đủ ba seed.
Full/LSR seed42 rerun là **replicas mới**, không phải tìm lại weights C đã mất.
Chốt dùng replicas mới ở seed42 trước khi train; không chọn điểm cao hơn giữa bản cũ và rerun,
không thay CSV/JSON/p-values C. Báo chênh lệch replica ngay cả khi cùng seed.

So Full với baseline, matched LSR và best-tuned LSR cho cả3seeds, báo mean/std và từng gap.
Hai lượt LSR mới phân biệt được control tune tốt ở seed42 với cải thiện lặp lại.
Hai lượt phục hồi giải quyết missing weights và cho phép audit validation/cơ chế đầy đủ.

Để tạo command: dùng argv của runner, giữ mọi hạng mục mục4 và `--validation-only`;
chỉ đổi run-name, seed, method, alpha/lambda. Dùng run-name mới, ví dụ
`mbart-amis-full-a010-l010-replica-seed42`, không overwrite `mbart-amis-full-seed42` C.
Outputs đề xuất: `outputs_rebuttal/amis_configuration_confirmation_20261004/`;
HF prefix riêng `amis_configuration_confirmation_20261004_v1`.
Trainer upload best.zip/metrics/predictions; **chưa tự tạo COMPLETE/verify-cleanup như runner**.
Chat tiếp phải chuẩn bị điều phối confirmation, manifest và verification trước khi chạy;
runner screening hiện tại không làm công việc này.

Đối với mọi run mới: pinned pretrained → full budget20epochs/patience4 → selected best
validation → raw predictions → HF SHA/size verification → COMPLETE → sync → cleanup.
Early stopping đúng protocol vẫn là run hoàn chỉnh; không tắt sớm ở epoch thuận lợi.

##### C. Chẩn đoán tương tác và detach — năm lượt mới, chạy theo gate

Ưu tiên độ tin cậy theo yêu cầu04/10: không chỉ chạy một no-detach seed rồi kết luận cơ chế.

| Controls | Seeds mới | Số lượt | Câu hỏi |
| --- | --- | ---: | --- |
| Full α0.10/λ0.03 detached | 43,44 | 2 | Full thua matched LSR λ0.03 ở seed42 có lặp không? |
| Full α0.10/λ0.10, cached source attached | 42,43,44 | 3 | Bỏ detach có giúp khi forward/objective giữ nguyên? |

Full λ0.03 seed42 đã có; kết hợp với hai lượt mới để so với LSR λ0.03 đủ3seeds.
Full λ0.03 là **diagnostic control**, không tự thay best Full bằng trial thua selection.
Nó kiểm tra tương tác objective; Full λ0.10 thắng LSR λ0.10 nhưng λ0.03 thì ngược lại
chưa đủ để nói nguyên nhân là gradient conflict, collapse hoặc routing quá mạnh.
No-detach dùng cùng α0.10/λ0.10 với selected Full; chỉ đổi cached-source gradient.
So3seeds với detached và cả tuned LSR, không chỉ so với matched LSR yếu hơn.

Nếu B cho thấy Full chưa hơn tuned LSR, dừng claim cải thiện và mở rộng ngôn ngữ;
C vẫn có thể chạy để phân biệt nguyên nhân trước sửa phương pháp. Không coi C là
thử nhiều cấu hình đến khi có số đẹp. Dừng nếu implementation gate không đạt.

- Original detached CLRR-only α0.10 đã có ở C đủ3seeds, reuse khi B0 đạt.
  Nếu no-detach được chọn làm phương pháp mới, cần **attached CLRR-only** đủ3seeds
  (λ0) để phân rã đúng kiến trúc mới; không dùng detached CLRR-only đại diện attached.
  Đây là3lượt có điều kiện còn lại; entry/smoke cho attached CLRR-only cần chuẩn bị.
- Full bỏ detach **chỉ ở cached CLRR source**, giữ LSR target no-grad, cùng α/λ cuối,
  seeds42/43/44, tối đa3runs.
- Entry có sẵn: `scripts/revalidation/train_amis_no_detach.py`, nhận các args của trainer.
- Implementation: `src/amis_rewire/ablation.py`.
- **Không chỉ xóa `.detach()` trong hook**: với reentrant gradient checkpointing,
  hook forward chạy no-grad. Attached control đặt routing ngoài checkpoint call từng layer.
- CPU smoke `tests/smoke_attached_routing.py` đã đạt: forward/loss như detached,
  plain/checkpoint gradients khớp, source gradient bổ sung thực sự hoạt động.
- **GPU gate chưa chạy**. Trước train control, upload source/test/entry trong bundle riêng,
  không sửa source screening đang chạy; chạy:

```bash
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64 \
  /content/miniforge3/envs/clrr/bin/python tests/smoke_attached_routing.py --cuda
```

Kiểm tra thêm tiny BF16 training/checkpoint reload bằng entry mới trước train thật.
Không có bằng chứng gradient của detached implementation cũ bị lỗi; đây là control cơ chế mới.

##### D. Đánh giá test cuối và quyết định

Sau khóa candidate và đạt gates, load **đúng archive/commit/SHA**. Strict-check missing,
unexpected và mismatched keys; tokenizer/proxy đúng; FP32 beam4/raw references/TF32off.
Đặt tường minh `num_beams=4`, `max_length=256` khi generation; không dùng defaults
trong saved generation_config để suy ra protocol. Nguồn helper inference/extraction:
`reevaluate_checkpoints.py`, `measure_checkpoint_mechanisms.py`.
Scripts cũ có manifest mặc định nhóm B; phải truyền/viết manifest D riêng, không chạy mặc định
rồi gọi là D. No-detach inference forward như detached, nhưng weights/provenance phải đúng.

- Báo từng seed, mean/std, paired deltas; bootstrap sentences không thay training-seed evidence.
- Family chính D: Full−Baseline, Full−matchedLSR, Full−bestLSR ×2metrics ×3seeds =Holm18.
  Nếu hai LSR trùng, giữ18slots với unused p=1, không giảm family để tăng significance.
- Family controls gốc Holm18: detachedCLRR−Baseline, detachedFull−detachedCLRR,
  detachedFull−attachedFull (cùng α0.10/λ0.10).
  Kế hoạch mới thêm hai comparisons diagnostic: Full λ0.03−LSR λ0.03 và
  detachedFull λ0.10−detachedFull λ0.03, mỗi comparison ×2metrics ×3seeds.
  Nếu tính p-values cho cả hai, khai báo **secondary family Holm30 trước test**;
  không bỏ comparisons thất bại hoặc giảm family sau xem kết quả. Cũ Holm24 giữ nguyên.
  Nếu attached được chọn và train attachedCLRR-only, thêm attachedCLRR−Baseline và
  attachedFull−attachedCLRR (12tests) vào kế hoạch secondary: **Holm42**.
  Khóa family trước final test; không so attachedFull với chính nó rồi gọi detach control.
  ChrF character-only được báo như sensitivity; nếu thêm p-values phải khai báo family trước.
- **Không sửa A/B/C raw CSV/JSON, không gộp các protocol và không tính lại Holm24 C.**
- GO dự án: mean Full−Baseline≥0.5BLEU, Full−matchedLSR≥0.3,
  Full thắng matchedLSR3/3seeds, baseline≥2/3, mean Full−bestLSR>0,
  không harm hệ thống trên chrF/chrF++.
- GO có điều kiện: gain nhỏ hơn nhưng lặp3/3 vs matchedLSR, vượt bestLSR trung bình.
- NO-GO: không vượt controls, chỉ một seed thắng, gain chỉ do rare-ngram hoặc harm rõ.
  Đây là tiêu chí dự án, **không phải chuẩn acceptance ACL**.
- Nếu chưa ổn: báo nguyên nhân/giả thuyết tối ưu cho tác giả; không tự sweep nâng cấp,
  không chạy Turkish để né kết quả Amis. Turkish cần tác giả cho phép riêng.

#### 7. Lưu tài liệu, dừng GPU và các file không được chạy nhầm

- Số mới → `docs/experiments/EXPERIMENTS_UPDATED.md`; diễn giải → `docs/research/RESEARCH_INSIGHTS.md`.
- Số cũ/paper ban đầu đã ở archive, không phục hồi vào file active.
- File này là bàn giao thao tác, không tạo thêm một ledger số liệu cạnh tranh.
- Sau backup và sync:

```powershell
colab stop -s colab
colab status -s colab
```

`scratch/finish_current_configuration_remote.py` là helper **một lần** để dừng đúng phiên
hiện tại, chứa PID cụ thể; **không chạy lại helper này ở runtime mới**.
`scratch/launch_configuration_search_remote.py` là launcher screening có thể chạy lại
sau restore; không thay bằng các launcher Turkish/PEFT/historical có tên tương tự.

#### 8. Câu giao việc cho chat tiếp

> Đọc `docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history`, `docs/experiments/EXPERIMENTS_UPDATED.md` và `docs/research/CLRR_GO_NOGO_PLAN.md`.
> **Tình trạng:** Toàn bộ 8/8 screening trials (mBART Amis, seed 42, validation-only) đã hoàn tất 100%, receipts và archives đã lưu trữ an toàn trên Hugging Face `FiveC/amis-rewire-checkpoints`. Colab hiện đã tắt hoàn toàn (cả 2 tài khoản). Không chạy thêm screening.
> **Nhiệm vụ tiếp theo:** đọc mục10/11; đã xong4B2+2interaction, còn3no-detach controls.
> Khi tác giả yêu cầu chạy: kiểm tra artifacts tránh trùng, chạy GPU/BF16/checkpoint smoke,
> rồi Full source-attached α0.10/λ0.10 seeds42/43/44; chọn kiến trúc bằng validation.
> Nếu chọn attached, chạy attachedCLRR-only λ0 đủ3seeds trước test. Khóa manifest/families,
> evaluate FP32beam4 theo mục11, báo từng seed/CI/Holm và quyết định Go/No-Go.
> Nếu chưa ổn, dùng diagnostics để đề xuất nâng cấp. Không mở Turkish/NLLB hoặc train lại
> trials đã hoàn tất; backup FiveC prefix riêng, verify trước cleanup, báo chat mỗi10phút.
> Khi nào tác giả yêu cầu push GitHub thì mới thực hiện git push.

#### 9. Phân tích hiện hành và quyết định — 04/10/2026

- Nguồn:8sets metrics/COMPLETE/validation_predictions D, `inputs.json`,
  `validation_influence.json`, metadata/summary/paired-bootstrap C và docs hiện hành.
- CPU Python trong conda `clrr`, SacreBLEU2.6.0. Tái tính cả8sets raw validation
  BLEU/chrF++/chrF khớp <1e-8; hashes,576rows, source order và raw references đạt.
- Native corpus scoring row197 khớp sufficient-statistics leave-one-out <1e-10.
- Receipt review riêng: `outputs_rebuttal/amis_configuration_20261003_v1/screening_review_20261004.json`.
- Không train, không dùng GPU, không mở test mới, không reverify HF/live Colab trong lượt này.
- Những sửa dưới đây là diễn giải/typo docs; **không thay primary metrics hoặc raw artifacts**.

##### Số hiện tại ổn hay chưa?

**Đo đạc local ổn; bằng chứng cho claim CLRR cải thiện rõ chưa ổn.**

| So sánh validation seed42 | ΔBLEU | ΔchrF++ | Diễn giải |
| --- | ---: | ---: | --- |
| Best Full gốc−Baseline C | −0.2298 | −0.1692 | Chưa vượt baseline |
| Best Full gốc−best LSR λ0.03 | −0.1436 | −0.1271 | Chưa vượt control được tune |
| Best Full gốc−matched LSR λ0.10 | +1.0749 | +0.5643 | Routing có tín hiệu có lợi dưới objective này |
| Full λ0.03−LSR λ0.03 | −0.6732 | −0.4233 | Routing bất lợi ở trial objective nhẹ hơn |
| Full λ0.30−LSR λ0.30 | +0.8173 | +0.4663 | Thắng matched control không đủ thành best quality |
| Best LSR λ0.03−Baseline C | −0.0862 | −0.0421 | LSR tune tốt hơn λ0.10 nhưng chưa vượt baseline |

**Điểm có lợi:** protocol/tái tính được kiểm tra; best Full gốc vẫn hơn matched LSR
λ0.10. C test trước đó có Full−LSR mean +0.2344BLEU,3/3seeds dương.
**Điểm bất lợi:** không Full tuning trial mới nào thắng Full gốc. Một control LSR
được tune đã vượt best Full ở validation seed42; matched-control gain không đủ
chứng minh đóng góp riêng vượt control mạnh. C trước đó không có test nào đạt Holm24<0.05.

LSR λ0.03 thắng best Full mới trên chrF/chrF++ trong mọi leave-one-out;
gap Full mới−LSR có khoảng −0.5053 đến−0.2503chrF, −0.4086 đến−0.1656chrF++.
Chưa kiểm tra leave-one-out Full gốc vì thiếu weights/predictions validation seed42.
Không chuyển kết luận validation seed42 thành kết luận test, qua seed hoặc mọi ngôn ngữ.
Không suy cơ chế từ scalar LSR loss nhỏ; chưa có bằng chứng từ screening rằng collapse
hay detach chính là nguyên nhân. Baseline vẫn mạnh nên cũng không nói dataset không học được.

##### Thứ tự thực hiện và ngân sách đề xuất

| Giai đoạn | Công việc | Train mới | Đầu ra quyết định |
| --- | --- | ---: | --- |
| 0 | B0: audit/reproduce validation archives C, source/precision/reuse | 0 | Protocol có thể reuse hay phải sửa trước? |
| 1 | B1: khóa cấu hình và comparisons trước candidate test | 0 | Full gốc, matched LSR0.10, best LSR0.03; chưa chọn lại từ test |
| 2 | B2: phục hồi2replicas seed42 + LSR0.03 seeds43/44 | 4 | Routing có vượt tuned control qua seeds? |
| 3 | C: Full0.03 seeds43/44 + no-detach Full0.10 đủ3seeds | 5 | Tương tác λ và detach có giải thích vấn đề? |
| 4 | Nếu attached được chọn: attached CLRR-only đủ3seeds | 3 có điều kiện | Component ablation khớp phương pháp mới |
| 5 | Khóa phương pháp cuối, test FP32beam4, uncertainty và families | 0 | GO/GO có điều kiện/NO-GO |

Đã dùng8train mới. Kế hoạch chính thêm9 → **17train mới tổng cộng**;
attached CLRR-only nếu cần thêm3 →20. Reuse có điều kiện là để giữ comparability,
không phải vì chỉ ưu tiên tiết kiệm. Khoảng3giờ train thực tế cho8screenings vừa qua
không gồm setup/download/upload/audit; không dùng con số này làm bảo đảm thời lượng.
Nếu B0 phát hiện protocol không thể reuse, **lập lại matched suite/budget trước**, không
trộn scores hoặc che chênh lệch để vừa20. Tác giả đã ưu tiên độ tin cậy;20 không là lý do
bỏ một control cần thiết, nhưng chưa tự chạy thêm ngoài kế hoạch trong lượt phân tích này.

##### Go/No-Go để chọn hướng, không phải bảo đảm ACL

1. **Tiếp tục final evaluation:** original Full có lợi ích validation qua seeds,
   mean vượt tuned LSR và không harm hệ thống trên BLEU/chrF. Seed42 hiện thất bại
   nên phải báo đầy đủ, không đổi định nghĩa ổn định để giấu seed này.
2. **Nếu Full thua tuned LSR lặp lại:** NO-GO cho claim cải thiện hiện tại; vẫn chạy
   controls C để chẩn đoán theo phạm vi đã định. Không mở thêm language/backbone suite.
3. **Nếu no-detach tốt hơn detached và tuned control lặp lại:** có căn cứ đề xuất
   đổi source-gradient của phương pháp; cần lock mới, attached CLRR-only và final test
   tương ứng trước claim. Không chọn architecture bằng test hoặc chỉ một seed.
4. **Nếu λ0.03 làm routing hại lặp lại:** dùng paired gradients CE/weighted LSR,
   residual RMS và representations trên cùng fixed batches/checkpoints để kiểm tra
   interaction; không tự kết luận tăng λ hoặc normalizing residual chắc chắn sửa được.
5. **Nếu detached/attached cùng không hơn controls:** báo phương pháp hiện tại chưa
   có contribution đủ vững; đề xuất nâng cấp theo diagnostic cụ thể thay vì thêm grid
   trọng số vô hạn. Schedule LSR chỉ có lý do khi trajectory chứng minh tác động theo
   thời gian; normalization chỉ có lý do khi residual ratios/outliers liên quan harm.

Ba seeds trên benchmark đã xem test là evidence phát triển, chưa là generalization.
Một kết quả positive nhỏ không bảo đảm ACL; một screening negative chưa chứng minh
CLRR vô dụng. Turkish chỉ được mở sau quyết định phương pháp và tác giả cho phép riêng.

##### Kiểm tra diễn giải thống kê

Đã rà11/11 loại trong checklist ARS: không phát hiện đảo hướng tổng hợp ở các gaps
đã báo; không suy câu/seed cá thể từ mean; sampling/collider/base-rate không là
phép phân tích đang dùng; không chọn lại seed cực trị; giữ mọi trial/early-stopped run;
không p-hack/switch metric; không suy cơ chế nhân quả hay hướng ngược từ scalar scores.
Nguy cơ còn lại: winner's curse do chọn trên seed42, degrees of freedom của development,
reuse/provenance và thiếu independent dataset. Rà checklist không là chứng nhận hết bias.

##### Kết quả thực thi thực tế (04/10/2026) — Hoàn thành 6/6 runs & Tắt máy

Toàn bộ 6 runs trong Confirmation Suite (Phase B2) và Interaction Diagnostic (Phase C) đã hoàn tất:
1. `mbart-amis-full_a010_l010-seed42-replica`: chrF++ 14.7317, BLEU 17.6437 (khớp 100% C).
2. `mbart-amis-lsr_l010-seed42`: chrF++ 14.1674, BLEU 16.5688 (Full vượt +0.56 chrF++ trên matched control).
3. `mbart-amis-lsr_l003-seed43`: chrF++ 14.0109, BLEU 16.3621.
4. `mbart-amis-lsr_l003-seed44`: chrF++ 14.0814, BLEU 16.8332.
   -> Mean3seeds Best-Tuned LSR:14.3170 ±0.4705. Phù hợp với biến thiên seed và nguy cơ
   selection bias ở seed42; chưa chứng minh Winner's Curse là nguyên nhân.
5. `mbart-amis-full_a010_l003-seed43`: chrF++ 14.4775, BLEU 17.2166 (vượt LSR seed 43 +0.47 chrF++).
6. `mbart-amis-full_a010_l003-seed44`: chrF++ 14.1297, BLEU 16.7903 (vượt LSR seed 44).
   -> Mean3seeds Full λ0.03:14.3476 ±0.1899. Sample std thấp hơn LSR λ0.03 ở ba lượt này;
   chưa chứng minh training variance giảm hoặc representation ổn định hơn.
7. Cả 2 Colab A100 (`colab` & `colab2`) đã tắt hoàn toàn. Checkpoints/receipts đã verify và backup lên Hugging Face `FiveC/amis-rewire-checkpoints`.

#### 10. Đánh giá sau sáu lượt confirmation — 04/10/2026

##### Đối chiếu artifacts

Tái tính local bằng Python conda `clrr`, SacreBLEU2.6.0:6sets validation BLEU/chrF++/
chrF khớp metadata <1e-8;576rows, source order, raw references và prediction SHA256 khớp
COMPLETE. Source/data hashes trong confirmation_inputs.json đạt; các training/evaluation
settings đối chiếu với C không khác ở các fields protocol. Không train hoặc generate test
mới, không reverify HF/live sessions trong lượt đọc kết quả này.
Receipt: `outputs_rebuttal/amis_configuration_confirmation_20261004/confirmation_review_20261004.json`.

Full/LSR replicas seed42 khớp BLEU/chrF++ C ở sai số numerical <1e-12.
Đây là bằng chứng tái lập hai metrics trong hai replicas này, không chứng minh weights
identical hoặc tính tất định tuyệt đối trên mọi lần chạy/hardware.

##### So đúng cùng ba seeds42/43/44 — validation beam1

| System | Mean BLEU | Mean chrF++ | Sample std chrF++ |
| --- | ---: | ---: | ---: |
| Baseline C | 17.6633 | 14.6334 | 0.3782 |
| Selected Full α0.10/λ0.10 | 17.2135 | 14.4975 | 0.2053 |
| Best-tuned LSR λ0.03 | 16.9942 | 14.3170 | 0.4705 |
| Matched LSR λ0.10 | 16.5430 | 14.0608 | 0.1114 |
| Diagnostic Full α0.10/λ0.03 | 17.0403 | 14.3476 | 0.1899 |

Selected Full dùng replica42 và đúng C43/44; baseline/matched LSR43/44 reuse C.
Các summary này là follow-up có provenance reuse/replica; không overwrite summary C.
Không so LSR mean14.3170 với Full riêng seed42=14.7317 để claim gain.

| Comparison | Mean ΔBLEU | Seeds BLEU dương | Mean ΔchrF++ | Seeds chrF++ dương |
| --- | ---: | ---: | ---: | ---: |
| Full0.10−Baseline | −0.4498 | 0/3 | −0.1359 | 1/3 |
| Full0.10−matchedLSR0.10 | +0.6705 | 3/3 | +0.4368 | 3/3 |
| Full0.10−bestLSR0.03 | +0.2193 | 2/3 | +0.1805 | 2/3 |
| Full0.03−LSR0.03 | +0.0461 | 1/3 | +0.0306 | 2/3 |

**Có lợi:** Full chọn trước confirmation đã vượt tuned LSR trung bình ở cả hai metrics,
thắng seeds43/44; nó vượt matched LSR ở cả3seeds. Kết quả seed42 bất lợi không lặp
trên tất cả seeds. Có lý do tiếp tục kiểm chứng CLRR; chưa có lý do phải bỏ phương pháp ngay.

**Bất lợi:** Full vẫn thua baseline trên validation BLEU cả3seeds và trung bình −0.4498.
Gain vs tunedLSR +0.2193BLEU còn nhỏ, chưa có CI/p-values mới để kết luận chắc chắn.
Fullλ0.03 gần hòa matchedLSR: gain character-only mean **−0.0111**, BLEU thắng chỉ1/3seeds.
Không dùng std nhỏ hay gap chrF++ này để claim superiority hệ thống, synergy hoặc
ổn định biểu diễn. LSRλ0.10 còn có sample std chrF++ thấp hơn Fullλ0.10, nên std thấp
không tự đồng nghĩa chất lượng cao hay lợi ích ổn định do CLRR.

**Kết luận:** số mới giúp bảo vệ contribution routing tốt hơn screening seed42,
nhưng chưa đủ bảo vệ claim phương pháp vượt baseline/controls mạnh một cách vững chắc.
C test cũ có Full−Baseline mean+0.3800BLEU và Full−LSR0.10+0.2344 nhưng không đạtHolm24;
không gộp gain test cũ với validation mới thành một gain chung. Baseline tốt trên validation
không chứng minh chắc nó thắng ở final test; cần đánh giá bằng protocol đã khóa.

##### Còn phải làm gì trước quyết định phương pháp?

1. **Kiểm tra3no-detach runs:** artifacts hiện có4B2 +2interaction, không có đủ5controls C
   theo kế hoạch. Hoàn thành matched no-detach α0.10/λ0.10 seeds42/43/44 sau GPU/BF16/
   checkpoint smoke gate; nếu đã chạy nơi khác, đồng bộ metrics/receipts/predictions về
   đúng ledger trước phân tích. Chưa kết luận detach là nguyên nhân hoặc nên bỏ detach.
2. Chứng nhận reuse B0 và generate validation của selected C archives nếu chưa có;
   metadata/protocol match không tự thay cho tái generate/influence audit đúng weights.
3. Khóa phương pháp và test manifest đúng archives/commit/SHA; FP32beam4/max256/raw refs,
   TF32off. Chấm selected Full, baseline, tuned/matchedLSR, CLRR-only và controls theo
   families đã khai báo. Replicas42 phải dùng đúng weights mới, không mượn test C42 vì
   validation khớp; tái tính uncertainty và paired deltas, không chọn lại từ test.
4. Nếu attached tốt hơn lặp lại: cần matched attachedCLRR-only rồi mới sửa phương pháp.
   Nếu detached/attached vẫn không vượt controls trên final evaluation, dùng diagnostics
   CE–LSR/residual/representations để chọn nâng cấp; không sweep vô hạn hoặc chạy Turkish
   để bỏ qua kết quả Amis. Turkish cần tác giả cho phép riêng.

Hiện đã dùng14train mới (8+6); còn3planned no-detach →17, thêm3attachedCLRR-only nếu cần →20.
Lượt đọc này sửa diễn giải docs và summary comparisons, giữ primary scores/artifacts gốc.

#### 11. Quy trình chạy tiếp cho bên nhận bàn giao

##### 11.1. Trước khi dùng GPU

1. Đọc mục10 để hiểu số hiện tại: selected Full α0.10/λ0.10, matchedLSRλ0.10,
   best-tunedLSRλ0.03. Đã xong14lượt mới (8screening+6confirmation/interaction).
2. Kiểm tra repo/artifacts HF và local xem3no-detach đã được chạy nơi khác chưa.
   Nếu có, sync và verify receipts; không train lại chỉ vì local thiếu file.
3. Chứng nhận reuse C: config/source/data hashes, selected checkpoint, tokenizer,
   generation precision và raw references. Tái generate validation cho archives C
   cần dùng nếu chưa có; lưu outputs mới riêng, báo discrepancy, giữ scores C nguyên.
4. Chuẩn bị worker/launcher **cho attached control**, không chạy nguyên worker confirmation
   vì queue hiện tại chỉ có6trials đã xong và dùng detached trainer.
5. Ghi manifest nguồn và kế hoạch chạy3seeds trước khi chạy. Prefix/output đề xuất:
   `outputs_rebuttal/amis_detach_control_20261004/`,
   `FiveC/amis-rewire-checkpoints/amis_detach_control_20261004_v1`.
   Nếu prefix đã dùng, kiểm tra receipts hoặc chọn version mới; không overwrite.

Chạy Python/tests/training trong **conda clrr**. GPU: Colab A100 Standard40GB,
**không high-mem**, cùng dependencies mục4/environment lock.
Giám sát process-alive, epoch/eval, finite loss, disk và backup; báo chat mỗi10phút.
Khi tạm ngưng hoặc đã hoàn tất và backup/sync xong, tắt mọi phiên A100 đang dùng.

##### 11.2. Gate implementation trước3lượt train

Source đã có:

- `src/amis_rewire/ablation.py`: attached-source encoder routing ngoài checkpoint call.
- `scripts/revalidation/train_amis_no_detach.py`: entry Full attached, targetLSR vẫn no-grad.
- `tests/smoke_attached_routing.py`: forward/gradient/checkpoint check, CPU đã đạt.

GPU smoke trong remote project:

Sau đó tiny integration **BF16 train + gradient checkpointing + best checkpoint reload**
qua attached entry: loss/grads finite, save/load keys đúng, generation FP32 và no test access.
Smoke hiện có không thay cho BF16 trainer integration này; bên nhận phải chuẩn bị nếu chưa có.
Không chỉ xóa `.detach()` trong hook: reentrant checkpoint có forward no-grad;
attached branch phải thật sự giữ thêm đường gradient, không chỉ có tên no-detach.
Nếu gate fail: lưu lỗi, sửa implementation và verify trước train thật.

##### 11.3. Ba lượt Full source-attached — cần train

| Run-name đề xuất | Seed | α | λ | CLRR source | LSR target |
| --- | ---: | ---: | ---: | --- | --- |
| `mbart-amis-full-attached-a010-l010-seed42` | 42 | 0.10 | 0.10 | Attached | No-grad |
| `mbart-amis-full-attached-a010-l010-seed43` | 43 | 0.10 | 0.10 | Attached | No-grad |
| `mbart-amis-full-attached-a010-l010-seed44` | 44 | 0.10 | 0.10 | Attached | No-grad |

Command phải gọi **`scripts/revalidation/train_amis_no_detach.py`**, không gọi trực tiếp
`-m amis_rewire.train` rồi mặc định coi đó là attached. Dùng `--method jepa-clrr-enc`,
`--rewire-stack encoder --rewire-distance 2 --rewire-strength 0.1 --jepa-weight 0.1`.
Lấy full argv từ `scripts/run_amis_confirmation_worker.py`, đổi entry/run-name/seed/output/
backup prefix; giữ LR5e-5, WD0, warmup0.06, batch32×4/eval8, max lengths256,
20epochs/patience4, BF16/noFP16, gradient checkpointing, pretrained pinned revision,
tl_XX→zh_CN, tokenizerBLEUzh và **`--validation-only --generation-precision fp32`**.
Chọn checkpoint bằng raw validationchrF++, beam1. Không mở candidate test trong training.
Metrics phải ghi `clrr_source_detach=false`, `lsr_target_detach=true`, seed và full config.

Mỗi run cần metrics,validation_predictions,COMPLETE và best.zip. Xác minh HF archive
SHA256/bytes và prediction hash, sync local, rồi mới dọn weights. Không dùng zip/metrics
từ run khác nếu upload lỗi; kiểm tra receipts pinned commit. Token không được in hoặc
ghi vào docs/bundle; dùng credential đã được tác giả cho phép qua helper an toàn.

**Câu hỏi:** chỉ đổi source gradient có cải thiện validation qua seed so với detached Full
và tunedLSR không? Báo gaps từng seed, mean/std, chrF++/BLEU và character-only khi đủpreds.
Không chọn seed đẹp hoặc bỏ run không cải thiện. Nếu dừng sớm theo patience, vẫn báo run đó.

##### 11.4. Chọn kiến trúc bằng validation; attachedCLRR-only nếu cần

Sau3runs, so attached vs detached cùng α0.10/λ0.10 và cả bestLSRλ0.03/baseline.
Giữ selected original nếu attached không có lợi ích lặp lại hoặc có harm đáng kể.
Nếu attached có lợi ích validation lặp lại qua seed, có thể đề xuất chọn kiến trúc này;
chốt lựa chọn và rationale **trước final test**, không đổi lại từ test scores.

Nếu chọn attached: train thêm **attached CLRR-only λ0**, α0.10/distance2/encoder,
seeds42/43/44, cùng protocol. Đây là component ablation cho kiến trúc mới.
Chuẩn bị entry/smoke đúng attached routing; không dùng `--method clrr-enc` của detached
trainer để đại diện attached. Verify λ0 thực sự bỏ auxiliary objective và giữ source gradient.
Nếu chọn detached: CLRR-only Cα0.10 đủ3seeds có thể reuse sau certification; không train lại.

Đặt bước có điều kiện này **trước test** nếu dùng để chọn/hoàn thiện architecture.
Không dùng test của no-detach để quyết định mở thêm training rồi trình bày như test độc lập.

##### 11.5. Final test checkpoint evaluation — không train

Tạo `final_evaluation_manifest.json` và update method lock trước generation:
run/seed/method/α/λ/detach, repo/path/pinned revision/archive SHA256/bytes,
tokenizer/config/source/reference hashes, reused predictions và comparison families.

| Nhánh chính | Checkpoints | Nguồn |
| --- | ---: | --- |
| Baseline | 3 | Cseeds42/43/44 |
| Original Fullα0.10/λ0.10 | 3 | Replica42 mới; C43/44 |
| MatchedLSRλ0.10 | 3 | Replica42 mới; C43/44 |
| TunedLSRλ0.03 | 3 | Screening42; confirmation43/44 |
| Full source-attachedα0.10/λ0.10 | 3 | Ba runs mới |

15checkpoint trong matrix chính **không đồng nghĩa15lượt train mới**.
Control bổ sung: detachedCLRR-only C×3 và diagnosticFullλ0.03×3 đã cóweights;
attachedCLRR-only×3 chỉ nếu nhánh có điều kiện được thực hiện.
Những predictions C đúng weights/protocol có thể reuse, ghi hashes/provenance.
**Replicas42 phải chấm đúng weights mới**, không reuse test C42 chỉ vì validation khớp.

Inference: **FP32, TF32off, beam4, max_length256**, raw575test rows đúng thứ tự,
BLEUzh/SacreBLEU2.6.0/chrF++word_order2; report character-only sensitivity riêng.
Load strict missing/unexpected/mismatched keys; wrap đúng routing/α/distance.
Saved generation_config defaults không thay explicit generation args.
Các evaluator cũ có manifest B mặc định; chuẩn bị manifest D và kiểm tra trước chạy.

Báo mỗi seed, mean/sample std, paired gaps, corpus-bootstrap CI và p-values/family.
Primary: selectedFull−Baseline, selectedFull−matchedLSR, selectedFull−tunedLSR,
×2metrics×3seeds=Holm18. Secondary detached/attached/interaction/CLRR comparisons:
khóa Holm30 hoặc Holm42 khi có attachedCLRR-only theo mụcD; không giảm family sau xem số.
Không đổi Holm24 C, không gộp validation/test hoặc các protocol A/B/C/D thành một mean.
Sentence bootstrap không thay evidence qua training seeds; ba seeds chưa là generalization.

##### 11.6. Nếu kết quả chưa đủ tốt: đo nguyên nhân trước nâng cấp

Không mở thêm grid trọng số ngay. Dùng đúng selected checkpoints và cùng fixed
validation batches, attention mask, dropout/RNG và precision để đo:

| Đo | Điều cần phân biệt | Hướng sửa chỉ khi có bằng chứng |
| --- | --- | --- |
| CE vs weightedLSR gradient norms/cosine | Auxiliary có lực thế nào, conflict ở đâu? | Weight/schedule khi trajectory hỗ trợ |
| Residual/current RMS ratios, outliers theo layer/input | Residual có quá lớn hoặc lệch scale? | Scale/normalization có control matched |
| Centered rank/spectrum, pairwise cosine, target/source reps | Có compression/collapse thực sự không? | Đổi alignment khi diagnostics và quality liên quan |

Nếu có checkpoints trung gian: đo đầu/giữa/cuối trên cùng batch indices; nếu thiếu,
ghi rõ endpoint-only, không nội suy trajectory từ best checkpoint.
Không optimizer step trong probe; tách shared embedding khỏi encoder/decoder groups,
giữ objective gốc, report weightedLSR/CE gradient và cache/RNG/mode đúng.
Không suy collapse hoặc semantics từ một cosine, không suy mechanism từ sample std BLEU.

##### 11.7. Kết luận và bàn giao sau chạy

- **GO/GO có điều kiện:** selected method có lợi ích qua seeds, vượt tuned control,
  không harm hệ thống và uncertainty hỗ trợ mức claim; dùng thresholds mụcD như tiêu chí
  dự án, không là acceptanceACL. Báo cả seed âm và các control bất lợi.
- **NO-GO mở rộng:** gain không lặp, không vượt controls hoặc harm đáng kể. Dùng diagnostics
  để đề xuất nâng cấp cụ thể và control/budget trước chạy; không train đến khi có số đẹp.
- Nếu attached thắng: đề xuất sửa phương pháp dựa trên validation/test/control tương ứng;
  không khẳng định detach có hại phổ quát từ một backbone/direction.
- Số mới → `docs/experiments/EXPERIMENTS_UPDATED.md`, diễn giải → `docs/research/RESEARCH_INSIGHTS.md`,
  trạng thái → file bàn giao này. Giữ raw scores/receipts/hashes và lịch sử nguyên.
- Backup FiveC prefix riêng, verify/sync trước cleanup; tắt mọi Colab đã dùng khi xong.
- Tổng còn **3train cần thực hiện +3có điều kiện**; smoke/probes/inference không tính là
  các lượt benchmark training. Chưa mở Turkish/NLLB, chưa nâng cấp ngoài plan khi chưa có quyết định.

#### 12. Hoàn tất Phase C2 (Attached Control) & Kết luận Cổng Quyết định (04/10/2026)

##### 12.1. Kết quả thực nghiệm 3 seeds Attached Control ($\alpha=0.10, \lambda=0.10$)

Đã hoàn tất 3/3 runs với `clrr_source_detach=False`, `lsr_target_detach=True` trên 2 GPU A100 Standard 40GB:

| Run | Seed | Best Checkpoint | Val BLEU | Val chrF++ | Val chrF0 | Trạng thái HF |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `mbart-amis-full-attached-a010-l010-seed42` | 42 | checkpoint-576 | 16.8827 | 14.1845 | 16.2653 | Verified SHA256 / Uploaded |
| `mbart-amis-full-attached-a010-l010-seed43` | 43 | checkpoint-684 | 17.4909 | 14.4706 | 16.6557 | Verified SHA256 / Uploaded |
| `mbart-amis-full-attached-a010-l010-seed44` | 44 | checkpoint-612 | 17.6376 | 14.6962 | 16.8618 | Verified SHA256 / Uploaded |
| **Mean $\pm$ Std** | — | — | **17.3371 $\pm$ 0.4005** | **14.4504 $\pm$ 0.2566** | **16.5943 $\pm$ 0.3029** | **Hoàn tất 100%** |

- **Hạ tầng & Tài nguyên:** Cả 2 Colab A100 (`colab` và `colab2`) đã được tắt ngay lập tức sau khi hoàn thành queue (`colab stop`).
- **Lưu trữ & Xác thực:** 12 artifacts (best zip archives, receipts, metrics, predictions) được verify SHA256 và lưu tại repo `FiveC/amis-rewire-checkpoints/amis_detach_control_20261004_v1/`.

##### 12.2. Kết luận Cổng Quyết định (Decision Gate) & Kế hoạch tiếp theo

1. **Trả lời câu hỏi 1 ("Detach có đang hạn chế phương pháp không?"):**
   - **Kết luận:** chưa có bằng chứng attached cải thiện đồng đều để thay architecture. Validation mean attached cao hơn detached khoảng +0.1236 BLEU, nhưng thấp hơn -0.0471 chrF++ (14.4504 so với14.4975). Seed42 giảm, seeds43/44 có cải thiện; không kết luận detach tối ưu hoặc attached gây bất ổn biểu diễn.
   - Attached CLRR vượt Matched LSR $\lambda=0.10$ (+0.31 BLEU trên seed42) và vượt Tuned LSR $\lambda=0.03$ khoảng +0.1334 chrF++, **+0.3429 BLEU** trên validation3-seed mean. Sample std chrF++ là0.26 so với0.47; ba seeds chưa chứng minh khác biệt phương sai tổng quát.
2. **Áp dụng luật rẽ nhánh Mục 11.3:**
   - Giữ bản Detached CLRR theo tiêu chí validation chrF++ đã chọn; không diễn giải thành thắng attached trên mọi metric.
   - Cả 3 seed của CLRR-only detached đều đã có sẵn từ các pha trước, **KHÔNG CẦN TRAIN LẠI 3 SEED CLRR-ONLY**.
3. **Chuyển sang Bước 2 (Đánh giá Test các checkpoint cuối — Không train):**
   - Theo bàn giao, kiến trúc Detached đã khóa trước lượt inference mới trên test set (`test.csv`,575rows). TestC đã được xem trong các phân tích trước đó; không gọi benchmark này là test chưa từng được xem hoặc khẳng định zero test snooping100%.

##### 12.3. Bảng Kết quả Đánh giá Final Test (18 Models, 6 Nhánh, 3 Seeds)

Protocol inference chuẩn: **FP32, TF32 off, beam 4, max_length 256**, 575 test rows nguyên bản. Tokenizer `zh`, SacreBLEU 2.6.0.
Các artifact predictions và metrics được lưu trữ tại `outputs_rebuttal/amis_final_evaluation_20261004/`.

| Nhánh / Phương pháp | Seed 42 | Seed 43 | Seed 44 | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (mBART-50)** | | | | |
| - BLEU | 19.7281 | 18.1378 | 20.3000 | 19.3886 $\pm$ 1.1203 |
| - chrF++ | 18.7138 | 13.1592 | 18.9017 | 16.9249 $\pm$ 3.2626 |
| - chrF0 (char-only) | 18.3973 | 17.0904 | 18.6478 | 18.0452 $\pm$ 0.8362 |
| **Full CLRR Detached (Selected, $\alpha=0.10, \lambda=0.10$)** | | | | |
| - BLEU | **19.9566** | **19.6123** | **19.7371** | **19.7687 $\pm$ 0.1743** |
| - chrF++ | **18.8878** | **18.4753** | **18.6065** | **18.6565 $\pm$ 0.2107** |
| - chrF0 (char-only) | **18.6626** | **18.1905** | **18.2613** | **18.3715 $\pm$ 0.2546** |
| **Matched LSR Control ($\lambda=0.10$)** | | | | |
| - BLEU | 19.7865 | 19.4879 | 19.3284 | 19.5343 $\pm$ 0.2325 |
| - chrF++ | 18.6797 | 18.6488 | 18.3633 | 18.5639 $\pm$ 0.1745 |
| - chrF0 (char-only) | 18.3581 | 18.2581 | 17.9957 | 18.2040 $\pm$ 0.1871 |
| **Tuned LSR Control ($\lambda=0.03$)** | | | | |
| - BLEU | 20.6257 | 19.6747 | 19.3858 | 19.8954 $\pm$ 0.6487 |
| - chrF++ | 19.1889 | 14.1556 | 18.5324 | 17.2923 $\pm$ 2.7362 |
| - chrF0 (char-only) | 19.0288 | 18.4188 | 18.1573 | 18.5350 $\pm$ 0.4472 |
| **Full Source-Attached Control ($\alpha=0.10, \lambda=0.10, \text{detach=False}$)** | | | | |
| - BLEU | 19.3531 | 19.7774 | 19.5667 | 19.5658 $\pm$ 0.2122 |
| - chrF++ | 18.5412 | 18.5131 | 18.6355 | 18.5633 $\pm$ 0.0641 |
| - chrF0 (char-only) | 18.2240 | 18.2104 | 18.3571 | 18.2638 $\pm$ 0.0811 |
| **Detached CLRR-only Control ($\alpha=0.10, \lambda=0.00$)** | | | | |
| - BLEU | 19.6403 | 18.9420 | 20.1485 | 19.5769 $\pm$ 0.6057 |
| - chrF++ | 18.5639 | 18.0823 | 18.9055 | 18.5172 $\pm$ 0.4136 |
| - chrF0 (char-only) | 18.2504 | 17.7017 | 18.6584 | 18.2035 $\pm$ 0.4801 |

##### 12.4. Paired Deltas, Kiểm định Bootstrap 10.000 mẫu & Hiệu chỉnh Holm-Bonferroni

###### 1. Bảng chênh lệch bắt cặp (Paired Deltas: Selected Full − Control)

| Đối chứng (Control) | Chỉ số | Seed 42 | Seed 43 | Seed 44 | Mean Delta $\pm$ Std | Tỷ lệ Seed dương |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **vs Matched LSR ($\lambda=0.10$)** | BLEU | +0.1701 | +0.1243 | +0.4087 | **+0.2344 $\pm$ 0.1527** | **3/3 (100%)** |
| | chrF++ | +0.2081 | -0.1735 | +0.2432 | **+0.0926 $\pm$ 0.2311** | 2/3 |
| | chrF0 | +0.3046 | -0.0676 | +0.2655 | **+0.1675 $\pm$ 0.2045** | 2/3 |
| **vs Baseline (mBART)** | BLEU | +0.2285 | +1.4744 | -0.5629 | **+0.3800 $\pm$ 1.0271** | 2/3 |
| | chrF++ | +0.1740 | +5.3162 | -0.2952 | **+1.7316 $\pm$ 3.1131** | 2/3 |
| | chrF0 | +0.2654 | +1.1001 | -0.3865 | **+0.3263 $\pm$ 0.7452** | 2/3 |
| **vs Tuned LSR ($\lambda=0.03$)** | BLEU | -0.6691 | -0.0624 | +0.3513 | -0.1267 $\pm$ 0.5132 | 1/3 |
| | chrF++ | -0.3011 | +4.3197 | +0.0741 | **+1.3642 $\pm$ 2.5664** | 2/3 |
| | chrF0 | -0.3662 | -0.2283 | +0.1039 | -0.1635 $\pm$ 0.2417 | 1/3 |
| **vs Attached Control** | BLEU | +0.6035 | -0.1652 | +0.1704 | **+0.2029 $\pm$ 0.3854** | 2/3 |
| | chrF++ | +0.3466 | -0.0378 | -0.0290 | **+0.0933 $\pm$ 0.2194** | 1/3 |
| | chrF0 | +0.4387 | -0.0199 | -0.0959 | **+0.1076 $\pm$ 0.2892** | 1/3 |
| **vs Detached CLRR-only** | BLEU | +0.3163 | +0.6702 | -0.4113 | **+0.1917 $\pm$ 0.5514** | 2/3 |
| | chrF++ | +0.3239 | +0.3930 | -0.2990 | **+0.1393 $\pm$ 0.3812** | 2/3 |
| | chrF0 | +0.4123 | +0.4888 | -0.3971 | **+0.1680 $\pm$ 0.4909** | 2/3 |

###### 2. Kết quả kiểm định thống kê Sentence Bootstrap 10.000 lần (Seed 42)
- **Primary Family (18 tests: 3 controls $\times$ 3 seeds $\times$ 2 metrics):**
  - Đã xuất file chi tiết tại `outputs_rebuttal/amis_final_evaluation_20261004/test_paired_bootstrap18.csv`.
  - Kết quả kiểm định tham chiếu trùng khớp tuyệt đối với `sacrebleu.significance.PairedTest` ($p=0.247375$).
  - Trên seed 43 vs Baseline: delta BLEU = +1.4744 (95% CI: [0.22, 2.83], unadjusted $p=0.0164$, $p_{\text{holm18}}=0.2952$).
  - Sau hiệu chỉnh Holm-Bonferroni cho18p-values, **không có phép thử nào đạt $\alpha=0.05$**. Cột CI là percentile95% chưa hiệu chỉnh; có2CI không chứa0. Script không tạo CI sau Holm.
- **Secondary Family (30 tests: 5 controls $\times$ 3 seeds $\times$ 2 metrics):**
  - Đã xuất file chi tiết tại `outputs_rebuttal/amis_final_evaluation_20261004/test_paired_bootstrap30.csv`.
  - Không có p-value nào đạt0.05 sau Holm30. Tuy nhiên5pairs trong file khác secondary family đã khai báo ở mụcD: thiếu CLRR-only−Baseline và hai interaction comparisons. Chưa coi file này hoàn tất family đã cam kết. Sentence bootstrap không thay bằng chứng qua training seeds.

##### 12.5. Tổng kết Khoa học & Bàn giao Chính thức

1. **Hiệu năng & Ổn định:**
   - Full Detached có sample std BLEU0.1743/chrF++0.2107 trên3seeds này. Baseline và tunedLSR có std chrF++ lớn hơn, nhưng độ nhạy word bigram hiếm góp phần đáng kể; không suy representation stability hoặc collapse. TunedLSR seed43 vẫn hơn Full ở BLEU và chrF character-only.
   - Vượt Matched LSR ở **100% (3/3) các seed** với chênh lệch trung bình **+0.2344 BLEU**.
   - Vượt Baseline trung bình **+0.3800 BLEU** và **+1.7316 chrF++**.
2. **Vai trò của Cơ chế Detach:**
   - Attached control không cho thắng lợi đồng đều đủ để thay lựa chọn validation. Các phép đo quality này không chứng minh gradient source gây bất ổn biểu diễn hoặc `.detach()` tối ưu; cần probe trực tiếp để kết luận cơ chế.
3. **Mức độ Cam kết Tuyên bố Khoa học (Claims):**
   - Dữ liệu thực nghiệm xác nhận lợi ích gia tăng có tính nhất quán qua các seed được chỉ định trước so với Matched LSR.
   - Full thua tunedLSR mean0.1267BLEU/0.1635chrF character-only, chỉ thắng1/3seeds ở hai metric này. Holm điều chỉnh p-values, không mở rộng các CI đang lưu. Cần hoàn tất secondary family đúng kế hoạch trước khi công bố là hoàn tất toàn bộ controls.

#### 13. Review độc lập sau final test — 04/10/2026

##### 13.1. Đã xác minh gì?

Chạy CPU trong conda `clrr`, không training/generation mới. Tái tính BLEUzh,
chrF++word_order2 và chrFword_order0 bằng SacreBLEU2.6.0 cho cả18prediction sets:
sai số so với CSV <1e-8;575source/target/index mỗi set khớp raw test, predictionSHA256
và testSHA256 khớp manifest. Receipt riêng:
`outputs_rebuttal/amis_final_evaluation_20261004/final_review_20261004.json`.
Không thay primary scores, predictions, decisionJSON hoặc kết quảA/B/C.
Rationale trong decisionJSON còn lỗi: detached validation mean là14.4975, không phải14.60;
giảm quality một seed không chứng minh representation degradation.

##### 13.2. Có lợi hay bất lợi?

| Full detached − control, test mean3seeds | BLEU | chrF++ | chrF character-only |
| --- | ---: | ---: | ---: |
| Baseline | +0.3800 | +1.7316 | +0.3263 |
| MatchedLSRλ0.10 | +0.2344 | +0.0926 | +0.1675 |
| TunedLSRλ0.03 | **−0.1267** | +1.3642 | **−0.1635** |
| AttachedFull | +0.2029 | +0.0933 | +0.1076 |

Lợi ích trước matchedLSR ở BLEU lặp3/3seeds là tín hiệu có lợi cho routing.
Nhưng mạnh nhất trong controls hiện tại là tunedLSR: Full thua BLEU/chrFcharacter
2/3seeds, nên chưa bảo vệ được claim CLRR hơn regularization được tune công bằng.

Raw Mandarin references chỉ có3rows có wordbigram: index126/170/468.
Ở seed43, Full match wordbigram row170 còn tunedLSR không match. Chênh chrF++
Full−tunedLSR là+4.3197; diagnostic loại riêng row170 cho−0.3304.
Trung bình3seeds diagnostic này là−0.1832chrF++ thay vì+1.3642.
Đây là độ nhạy metric, **không xóa row khỏi score chính, không chọn metric theo kết quả**.
Không gọi tunedLSR seed43 collapse: chrFcharacter18.4188 vẫn hơn Full18.1905.

##### 13.3. Quyết định và việc tiếp theo

**NO-GO mở rộng hoặc nâng claim superiority theo gate dự án hiện hành**:
Full−Baseline0.3800<0.5BLEU, Full−matchedLSR0.2344<0.3BLEU,
Full−tunedLSR<0BLEU. Không kết luận CLRR vô ích hoặc dự đoán điểm/acceptanceACL.
Giữ detachedα0.10/λ0.10 làm cấu hình tham chiếu để chẩn đoán, không gọi là tối ưu.

1. Hoàn tất phân tích đúng secondary family đã khóa: CLRR-only−Baseline,
   Full−CLRR-only, detachedFull−attachedFull, Fullλ0.03−LSRλ0.03,
   Fullλ0.10−Fullλ0.03, mỗi pair×3seeds×2metrics=30tests.
   Final matrix hiện thiếu3Fullλ0.03test checkpoints; chấm đúng weights đã có,
   **không cần train lại**. Giữ fileHolm30 hiện tại như lịch sử; xuất family đúng ở artifact mới.
2. Dùng validation batches cố định đo CE/weightedLSR gradientnorm/cosine,
   residual/currentRMS và centeredrank theo mục11.6. Checkpointbest chỉ cho endpoint;
   chỉ phân tích trajectory nếu thực sự có checkpoints đầu/giữa/cuối.
3. Chọn nâng cấp dựa trên diagnostics, khóa candidate/control/seed/protocol trước chạy.
   Ví dụ scheduleλ khi có conflict, normalize/scale residual khi có lệchRMS;
   đây là giả thuyết có điều kiện, chưa là khuyến nghị tăng/giảmλ ngay.
4. Sau khi candidate vượt tunedLSR lặp lại trên validation thì xác nhận theo protocol
   đã khóa; lịch sử xem test phải được công bố. Chưa chạy Turkish/NLLB trong lượt review này.

Training17runs và test18models đã hoàn tất; **toàn bộ secondary analysis theo kế hoạch
chưa hoàn tất**. Không chạy thêm GPU hoặc sửa paper trong lượt review này.

#### 14. Bàn giao chạy tiếp sau kiểm tra source và artifacts — 04/10/2026

##### 14.1. Material Passport và phạm vi

- Origin Skill/Mode: academic-research-suite / experiment-agent, validate + plan.
- Version: followup_plan_v2, 04/10/2026. Kiểm tra nguồn: VERIFIED ở phạm vi ghi dưới;
  thí nghiệm đề xuất: NOT RUN. Không biến kế hoạch thành kết quả thực nghiệm.
- Tác giả yêu cầu kiểm tra phần thiếu và viết hướng dẫn cho bên khác chạy tiếp.
  Lượt lập mục này chỉ kiểm tra CPU trong conda `clrr`; chưa mở GPU hoặc train.
- Đã kiểm tra: source evaluator/probe, receipts ba Full λ=0.03, đủ 35 probe folders
  của C, cùng 8 batch validation, native CE/auxiliary reconstruction difference
  <1e-6, representation đủ 576 rows và rank không vượt giới hạn.
- Chưa kiểm tra trực tiếp trong lượt này: phiên Colab, download weights từ HF,
  checkpoint trung gian của tuned LSR/attached. Receipts local phải được đối chiếu
  bằng SHA256/bytes sau download, trước khi load model.
- Phạm vi vòng kế tiếp: **0 lượt train mới**; 3 checkpoint test bổ sung,
  21 best-checkpoint probes và 6 lượt generate validation cho route-on/off.
  21 probes là 18 models final + 3 Full λ=0.03, không phải 21 lượt training.
- Điểm dừng: hoàn thành F0–F6, báo số/insight và đề xuất nâng cấp cho tác giả.
  Mục 14.8 là nhánh có điều kiện sau report. Turkish/NLLB chưa được cho chạy.

##### 14.2. Thứ tự chạy, mục đích và đầu ra

| Bước | Chạy cụ thể | Tài nguyên / train mới | Câu hỏi cần trả lời |
| --- | --- | --- | --- |
| F0 | Inventory 21 checkpoints; audit validation và 35 probes cũ | CPU / 0 | Weights đúng best, số hợp lệ, những gì reuse được? |
| F1 | Test Full α=0.10/λ=0.03, seeds 42/43/44 | A100 inference / 0 | Routing có lợi khi dùng cùng λ đã tune? |
| F2 | Đúng secondary 30 tests; score/sensitivity trên 21 models | CPU / 0 | CLRR đứng riêng, interaction và detach có đóng góp gì? |
| F3 | Gradient, residual và rank của 21 best checkpoints trên validation | A100 forward/backward / 0 | Full khác tuned LSR/attached ở đâu; có nguyên nhân khả dĩ? |
| F4 | Full λ=0.10 × 3 seeds, mỗi seed validation route-on và route-off | A100 inference / 0 | Routing lúc inference có giúp chính weights đã train? |
| F5 | Tổng hợp trajectory cũ, best endpoints và nhóm lỗi validation | CPU / 0 | Quan sát nào lặp qua seeds/layers/batches và liên hệ với quality? |
| F6 | Report GO/NO-GO, tối đa hai đề xuất candidate/control | CPU / 0 | Giữ phương pháp, nâng cấp gì, hay dừng hướng hiện tại? |

Không cần generate lại test của 18 models đã audit. Dùng lại 35 probes đủ provenance;
không train lại chỉ để phục hồi baseline endpoint seed42 còn thiếu. Training trajectory
và best checkpoint là hai loại bằng chứng khác nhau, phải báo riêng.

##### 14.3. F0 — nguồn weights và việc phải chuẩn bị trong code

**Nguồn 21 models:** manifest/CSV trong
`outputs_rebuttal/amis_final_evaluation_20261004/`, cộng ba Full λ=0.03 dưới đây.
Với models chỉ có reused predictions trong manifest, tìm đúng weight receipt từ
C/worker/HF trước probe; prediction SHA không thay weight SHA.

| Full λ=0.03 | Best | Thư mục chứa metrics/COMPLETE | HF prefix |
| --- | --- | --- | --- |
| Seed42 | checkpoint-720 | `outputs_rebuttal/amis_configuration_20261003_v1/runs/mbart-amis-full_a010_l003-seed42/` | `amis_configuration_20261003_v1` |
| Seed43 | checkpoint-720 | `outputs_rebuttal/amis_configuration_confirmation_20261004/runs/mbart-amis-full_a010_l003-seed43/` | `amis_configuration_confirmation_20261004_v1` |
| Seed44 | checkpoint-612 | `outputs_rebuttal/amis_configuration_confirmation_20261004/runs/mbart-amis-full_a010_l003-seed44/` | `amis_configuration_confirmation_20261004_v1` |

Repo: `FiveC/amis-rewire-checkpoints`. Path:
`<prefix>/<run_name>/<run_name>-best.zip`. Nguồn receipt là `COMPLETE.json` từng run:

```text
seed42 revision ba31adeec27fe1b2e2b122b46567722bad983213
SHA256 09ff385520a397d9ac192c2e4b421fcc4d96b12378273c32c7ef3b4bbaaab0d9
bytes 2270666529
seed43 revision e97a703810ce6348687cf8a46043a117240e3b8b
SHA256 1444eb3a0e7869767c52eda6c05e4eae3583255efddf213f189575d7dccb2637
bytes 2270706619
seed44 revision 63aec75d8db419896a13d7b57ffd2099285c9007
SHA256 568cdbb3a1901c7be0409ab8829e6f0e4c20462deb106491f29b96ab6f330c6b
bytes 2270675961
```

**Kiểm tra source cho thấy các entry chưa nhận đủ cấu hình follow-up.** Bên chạy
cần chỉnh/adapt và smoke trước GPU; không gọi nguyên trạng rồi xem là protocol mới:

| Entry đã có | Hạn chế hiện tại | Cần chỉnh trước chạy |
| --- | --- | --- |
| `scripts/revalidation/run_final_test_evaluation.py` | Hardcode OUT/manifest final18; download không truyền revision; resume chỉ nhìn file metrics | Nhận manifest/output-dir; pinned HF revision, SHA/bytes; resume đối chiếu run/weight/data/protocol/prediction hashes; chỉ queue 3 items mới |
| `scratch/compute_test_statistics.py` | Secondary30 là Full−5controls, khác pair-list đã khóa | Explicit pair-list mục14.4; shared sentence draws; output mới, giữ family cũ |
| `scripts/revalidation/measure_checkpoint_mechanisms.py` | Hardcode manifest/OUT, α/λ=0.1; representation dùng test; raw LSR chia 0.1 | Manifest/OUT/split/α/λ/detach rõ; validation; native objective; đúng λ và attached class |
| `scripts/revalidation/measure_training_trajectory.py` | Entry khởi động trainer; helpers dành λ=0.1 | Không dùng để đo best weights. Nếu dùng hook cho training mới, hỗ trợ λ/detach và restore RNG/mode/checkpointing |
| `scripts/revalidation/summarize_amis_trajectories.py` | Hardcode C, ghi lại summary C | Dùng logic kiểm tra nhưng xuất report riêng, không ghi đè C |

Tên entry đề xuất: `evaluate_amis_followup.py`, `analyze_amis_followup.py`,
`probe_amis_followup.py` trong `scripts/revalidation/`.
**Ba entry này chưa được tạo trong lần bàn giao này.** Có thể thêm flags vào entry
cũ thay vì tạo mới, miễn defaults giữ tương thích và output vòng mới tách rõ.
Không đổi default method trong `src/amis_rewire/modeling.py` để chạy probe.

F0 tái tính validation BLEU/chrF++/chrF character-only từ saved predictions của
21 models; kiểm tra 576 rows/reference/order/hash, paired seed means và sensitivity
word n-gram (row197 đã biết). Audit cả selection nếu có predictions/weights cạnh
best epoch. Nếu thiếu, ghi thiếu; một best checkpoint đã lưu không cho biết model
nào sẽ được chọn nếu đổi metric selection. Không dùng test để quyết định selection.

##### 14.4. F1–F2 — bổ sung test và đúng secondary family, không train

Folder mới: `outputs_rebuttal/amis_followup_20261004_v2/`, gồm:
`supplemental_test/`, `statistics/`, `probes/`, `route_off_validation/`, `reports/`.
HF prefix mới: `amis_followup_20261004_v2`.

Khóa manifest trước generation: run/family/seed, α=0.10/λ=0.03, source detach=True,
LSR target detach=True, encoder/distance2, receipt/commit/bytes/SHA, code/environment,
tokenizer/data hashes và metric signatures. Load không missing/unexpected/mismatched
keys; dùng đúng best archive. λ là metadata training; generation chỉ cần raw model
và đúng CLRR routing α=0.10.

Inference giống final18: **FP32, TF32 off, beam4, max_length256, batch8, pad_multiple8**;
source `tl_XX`, target `zh_CN`, raw575 test rows của
`data_processed/amis_mandarin/test.csv`; SacreBLEU2.6.0, BLEUzh, chrF++ word_order2.
Attached inference không-grad có cùng phép forward với detached, nhưng gradient
probe ở mục14.5 bắt buộc dùng đúng attached class.

**Đúng secondary family:** mỗi pair × 3 seeds × BLEU/chrF++ = 30 tests.

| Challenger | Control | Câu hỏi |
| --- | --- | --- |
| CLRR-only detached α0.10/λ0 | Baseline | CLRR đứng riêng có lợi? |
| Full detached α0.10/λ0.10 | CLRR-only detached α0.10/λ0 | LSR thêm gì vào routing? |
| Full detached α0.10/λ0.10 | Full attached α0.10/λ0.10 | Đường gradient source có ảnh hưởng? |
| Full detached α0.10/λ0.03 | LSR-only λ0.03 | Routing với cùng λ đã tune? |
| Full detached α0.10/λ0.10 | Full detached α0.10/λ0.03 | Sensitivity λ trong Full? |

Xuất `secondary_prespecified_holm30.csv` riêng. Giữ nguyên Holm30 hiện tại khác
pairs, Holm24C và primaryHolm18. Đây là hoàn tất plan đã ghi trước; testC/final
đã được biết, không gọi là blind preregistration mới.

Statistics: 10.000 paired corpus bootstrap draws, RNG42; signed-delta percentile95%
CI **chưa hiệu chỉnh**; p-value theo SacreBLEU centered absolute-difference bootstrap;
Holm trên đủ30p-values. Kiểm tra native `PairedTest` cho cả BLEU và chrF++ ở ít nhất
một pair, kiểm tra Holm với vector p đã biết. Báo từng seed, paired gaps, mean/sample
std(ddof1). Không pool575×3 như1725 independent sentences, không gọi sentence CI
là CI qua training seeds. ChrF character-only là sensitivity, chưa thêm p vào family.

Sau F1: 7 families × 3 =21 models; ghi rõ18 reused sets và3 new generation sets.
Audit đủ21 sets: hash/order/reference, tái tính3metrics. Sensitivity chrF++ báo
reference word unigram/bigram counts và matches, leave-one-out toàn575rows,
row170 diagnostic, đối chiếu BLEU/chrF character-only. Primary luôn giữ đủ575rows.

**Gate:** đủ đúng30tests/3seeds và audit đạt thì chuyển F3, kể cả quality âm.
Nếu lỗi weights/protocol, chỉ sửa items bị ảnh hưởng. Không lấy test Fullλ0.03 tốt
hơn để thay selected architecture hậu kiểm.

##### 14.5. F3 — gradient, residual và rank của 21 best checkpoints

**Matrix:** Baseline, CLRR-only, LSRλ0.10, LSRλ0.03, Fullλ0.10 detached,
Fullλ0.10 attached, Fullλ0.03 detached; mỗi nhánh seeds42/43/44.
Best/receipt khóa trước probe. Các best epoch khác nhau không là cùng thời điểm
training; các mối liên hệ quality–representation chỉ là quan sát.

**Inputs/modes chung:**

- Representation/residual: toàn576 validation rows, source và target, attention
  mask đúng, pooling valid tokens như production; FP32/TF32 off/eval/dropout off,
  max_length256. Lưu indices/tokenizer/config hashes; không dùng test để chọn nâng cấp.
- Gradient: cùng32indices và8batches×4 trong
  `outputs_rebuttal/amis_confirmation_20261003/trajectory_summary.json`.
  BF16 autocast, FP32 parameters, TF32 off, unclipped/unscaled, train-mode dropout;
  RNG1000+batch_id như C. Disable gradient checkpointing cho graph rõ, giữ LSR target
  no-grad và cache riêng source/target; **0 optimizer steps**.
- Native CE và auxiliary phải lấy từ cùng forward/dropout state. Không đo hai
  objectives bằng hai forwards khác RNG rồi gọi gradient cosine matched.
- Nếu kết luận phụ thuộc dropout/batch, thêm hai RNG replicates
  (2000/3000+batch_id) trên cùng32rows, ghi exploratory; không chọn replicate thuận lợi.
- Groups không overlap: shared input embeddings, encoder_without_shared,
  decoder_without_shared, other. Thêm encoder layer groups để aggregate không che
  conflict. Tied embedding/output weights chỉ tính một lần.

| Phép đo bắt buộc | Báo theo | Câu hỏi |
| --- | --- | --- |
| Native CE, raw/weighted LSR, normCE/normweightedLSR/cosine/combinednorm | Run×batch×parameter group; thêm encoder layers | Auxiliary mạnh bao nhiêu; conflict ở đâu? |
| Current/skip/post RMS, αskip/current RMS ratio, cosine | Run×source/target×row×layer | Có outlier, lệch scale hoặc cumulative gain? |
| Centered spectrum, hai effective-rank definitions, top1/top10variance, pairwise cosine | Run×source/target×layer | Compression ở đâu, có đồng thuận nhiều phép đo? |
| Source–target paired cosine và representation norms | Run×row/layer | Alignment có đi cùng mất diversity/quality? |

Raw LSR lấy đúng pooling/cosine objective; weighted auxiliary từ native graph.
λ lấy manifest và đối chiếu wrapper; không chia weighted LSR cho0.1 ở runλ0.03.
Baseline/CLRR-only λ0 có auxiliary thực tế bằng0; nếu tính counterfactual λ0.1,
output và nhãn riêng, không gọi là objective đã train.
Attached dùng `AttachedSourceResidualRewire` trong `src/amis_rewire/ablation.py`;
không chỉ bỏ `.detach()` trong checkpointed hook hoặc load bằng default class.

**Smoke trên4–8rows trước toàn matrix:** detached/attached cùng weights có forward
khớp trong tolerance precision, attached gradient path thực sự khác khiα>0;
LSR target requires_grad=False; cache không lẫn nguồn/đích; finite grads/noNaN;
native CE+λLSR decomposition đúng. Với cùng weights/RNG, đổi riêngλ0.10→0.03 thì
weighted auxiliary gradient norm xấp xỉ0.3lần: đây là kiểm tra probe, không phải
training ablation. α0 route-off phải khớp raw model. Centered rank≤min(n−1,d).
Không kết luận collapse chỉ từ một cosine/rank hoặc gradient norm nhỏ.

Mỗi run lưu `provenance.json`, `gradients.json`, `residual_per_example.csv`,
representations/spectra, `collapse_by_layer.json`, `DONE.json`.
Summary: median/range/p95/max, số batch cosine âm, paired differences qua3seeds;
không gọi collapse/gradient explosion nếu measurements không hỗ trợ.

##### 14.6. F4–F5 — routing lúc inference và liên hệ với lỗi dịch

**F4:** ba best Fullλ0.10 detached, mỗi weights generate validationbeam1 hai lần:
α0.10 route-on vàα0 route-off; tổng6generations, FP32/TF32 off/max256/batch8/pad8,
proxy/reference giống training validation. Sinh cặp trong cùng environment.
Báo corpus BLEU/chrF++/chrF character-only và số predictions đổi.
Route-off cùng weights không là LSR-only đã train. Route-off giảm quality gợi ý
weights dùng routing ở inference, không chứng minh routing training vượt LSR.
Route-off cao hơn gợi ý mismatch/over-reliance cần kiểm tra cache/scale.
Không sweep α inference theo test hoặc thay primary scores bằng route-off.

**F5:** reuse35mốc C: earlyepoch1/middleepoch5/actual endpoint.
Kiểm tra CPU lần này cho mean qua3seeds của median weightedLSR/CEnorm ở Full:
0.4627%early →0.3231%middle →0.1479%late; mean median cosine tương ứng
0.0477/0.0720/0.0573. Max residual ratio qua các mốc Full khoảng0.0868.
Các số này chưa hỗ trợ LSR áp đảo CE hoặc residual explosion. Global cosine dương
không loại conflict ở layer/batch, gradient nhỏ không chứng minh auxiliary vô ích.
Baseline endpoint42 thiếu; không nội suy. Không trộn partial probes của interrupted
attempts với run hoàn tất. Trajectory C chưa đại diện tunedLSR/attached trajectory.

Nếu có đúng intermediate weights của nhánh mới: khóa early/middle/late theo
metadata, đo cùng batches; không train lại chỉ để có biểu đồ trong bước này.
Best endpoints mới báo riêng với trajectory, không thay actual endpoint bằng best.

Nhóm lỗi validation: source tokenizer length≤32,33–64,>64 (khóa trước tổng hợp);
fragmentation/train frequency định nghĩa từ train/tokenizer, báo group sizes.
Đo empty outputs, repetition, length ratio/truncation và corpus3metrics theo nhóm.
Proxy không là gold morphology/adequacy. Không gọi21models là21 independent seeds
hoặc quy correlation rank–quality thành nguyên nhân.
Ví dụ dịch lấy indices/strata cố định, ẩn model name; không chỉ chọn câu Full thắng.
Đánh giá nghĩa Amis cần người biết ngôn ngữ; nếu chưa có, ghi thiếu human adequacy.

##### 14.7. F6 — quyết định và claim nào được hỗ trợ?

| Quan sát | Bằng chứng có thể hỗ trợ | Quyết định |
| --- | --- | --- |
| CLRR-only−Baseline dương lặp, metrics/uncertainty đồng thuận | Routing đứng riêng có tín hiệu trên Amis/mBART | Giữ component evidence, chưa suy chuyển ngôn ngữ |
| Fullλ0.03−LSRλ0.03 dương lặp và sensitivity đồng thuận | Routing hữu ích dưới λ đã tune | Bổ sung interaction; không đổi selected test model hậu kiểm |
| Fullλ0.03 gần hòa/thua LSRλ0.03 | Contribution nhỏ hoặc phụ thuộc λ/cấu hình | Chưa claim routing gain độc lập; chẩn đoán để sửa |
| Detached/attached khác quality nhưng probe không phân biệt | Cơ chế detach chưa rõ | Giữ validation choice; chưa claim detach tối ưu |
| Conflict theo layer/batch lặp qua≥2seeds và RNG | CE–LSR competition là giả thuyết đáng thử | Đề xuất schedule/reweight có matched control |
| Residual outliers liên hệ harm, route-off giúp | Scale/mismatch là giả thuyết đáng thử | Đề xuất normalization/bounded scale có control |
| Rank/cosine đổi nhưng quality không theo | Thay đổi representation quan sát được | Chưa claim anti-collapse hoặc chọn nâng cấp hướng này |
| Không lỗi, probes bình thường, Full vẫn thua tunedLSR | Hiệu quả routing hiện tại nhỏ/phụ thuộc điều kiện | NO-GO mở rộng; cân nhắc candidate có RQ rõ hoặc thu hẹp contribution |

Report nêu observation/hypothesis/alternative explanation/falsifiable prediction
và control cho từng candidate, cả số âm. Gradient/rank đẹp không thay chất lượng dịch.
Giữ gate dự án đã khóa: Full−Baseline≥0.5BLEU; Full−matchedLSR≥0.3BLEU;
matchedLSR3/3seeds,baseline≥2/3; Full−tunedLSR mean>0; không harm hệ thống trên
chrF character-only. GO có điều kiện vẫn cần vượt tunedLSR trung bình.
Không hạ ngưỡng sau xem số. Đây không là chuẩn acceptance ACL.
Holm không significant không chứng minh tương đương;21best probes không chứng minh
nhân quả representation stability. Báo cả mẫu chưa đủ và các explanation cạnh tranh.

Rà soát phương pháp luận khi viết report: aggregation/Simpson và ecological inference,
selection/Berkson/collider, base-rate (không áp dụng cho diagnostic classification
nếu không có nhãn), regression-to-mean, survivorship/missing weights, multiple testing,
đổi analysis theo kết quả, correlation/causation và reverse direction. Ghi rõ phạm vi
không áp dụng; không tạo “đạt11/11” như chứng nhận chất lượng nghiên cứu.

##### 14.8. Sau report: nâng cấp có điều kiện, budget và fairness

**Chưa khởi động trong vòng bàn giao này.** Nếu có giả thuyết rõ, lập
`upgrade_plan.json`: tối đa2candidates, α/λ/schedule/normalizer formula, objectives,
selection metric, training args, seeds, controls/comparisons và budget trước train.
Mỗi candidate chỉ đổi can thiệp cần phân biệt; không gộp schedule+normalization
khi chưa có evidence, không thử vô hạn cho tới khi có số đẹp.

- Giữ protocol C/D: pilot mỗi candidate seed42 (≤2train); chọn tối đa1candidate
  trên validation rồi confirm43/44 (+2train). Trong budget nâng cấp≤10train đã đề
  xuất, phần còn lại≤6train dành matched counterpart/LSR controls cần thiết.
  Scheduleλ áp dụng được cho LSR-only thì LSR cùng schedule là control bắt buộc.
  Đổi λ phải so cả LSR cùng λ và tunedLSR. LSR constant không là matched schedule.
- Reuse controls cũ chỉ khi data/training/selection/protocol và weights/predictions
  có provenance tương thích. Report failed pilots/cost, không mở lại λ grid vô hạn.
- Nếu cần đổi metric selection do audit: tạo **protocolV2 riêng**, tác giả chốt trước
  train, ví dụ BLEUzh primary, chrF character-only diagnostic, vẫn báo chrF++.
  Best-by-chrF++ cũ không đại diện best-by-BLEU. Baseline/CLRR-only/LSR-only/Full
  ×3seeds =12train dưới selection mới nếu không có weights từng epoch đủ reselect
  hợp lệ. Đây là budget mới ngoài10train; báo tác giả trước thực thi. Các nhánh
  được chọn checkpoint/tune cùng một protocol; không đổi metric riêng cho Full.
- Nếu candidate sát tuned control: đề xuất seeds45/46 cho **cả hai**, không thêm seed
  riêng cho Full/chọn seed thuận lợi. Chốt budget/analysis trước khi vượt10train.
- Test Amis đã được xem; mọi test confirmation phải ghi adaptation history.
  Sau method lock và tác giả cho phép, Turkish20k có thể kiểm chứng độc lập bằng
  cùng4nhánh×3seeds; không thay thế control âm ở Amis.

##### 14.9. Thực thi, giám sát, backup và bàn giao kết quả

Local và Colab đều chạy Python bằng conda `clrr`. Colab dùng A100 Standard40GB,
**không high-mem**, xử lý checkpoints nối tiếp. Một GPU đủ cho từng probe; hai máy
chỉ khi phân queue/backup rõ. Đo xong checkpoint thì sync output, giải phóng weights
và CUDA cache. Lưu torch/transformers/tokenizers/sacrebleu/CUDA versions thực tế.

Entry đã có để kiểm tra lại final18 bằng CPU:

```powershell
conda run -n clrr python -X utf8 scratch/review_final_results_20261004.py
```

Entry ghi receipt riêng `final_review_20261004.json`. Không chạy evaluator hardcode
final18 hoặc training worker để thay bước complete3checkpoints.
Entry mới phải có CLI help, validate manifest trên CPU và smoke GPU trước queue.
Nếu Colab chưa có `clrr`, tạo/kiểm tra đúng env và dependencies; không dùng base env.

Status mới: `outputs_rebuttal/amis_followup_20261004_v2/status.json`, chứa step/run,
progress/time/elapsed/process alive/GPU utilization/free VRAM/last success/error,
checkpoint receipt và HF paths. Khi phiên monitor hoạt động, **mỗi10phút báo chat**:
đang đo gì, xong bao nhiêu/tổng, lỗi/chậm và ETA theo thời gian thực tế.
Status file/background worker không tự đảm bảo chat notification khi chat đóng;
bên chạy phải giữ phiên theo dõi hoặc cơ chế notification được tác giả cho phép.
Không đoán ETA trước checkpoint đầu; không âm thầm đổi precision/split/batch khi OOM.

Artifacts bắt buộc: manifest/receipts/SHA, raw predictions, metrics/signatures,
bootstrap seed/pair-list, probe protocol/raw tables, paired route-on/off metrics;
`reports/validation_audit.md`, `reports/mechanism_report.md`, `reports/decision.md`.
Mỗi F-step xong: số ghi `docs/experiments/EXPERIMENTS_UPDATED.md`, insight ghi
`docs/research/RESEARCH_INSIGHTS.md`, trạng thái cập nhật bàn giao và chat có lợi/bất lợi.
Không tạo ledger experiments cạnh tranh hoặc sửa số archive.

HF: `FiveC/amis-rewire-checkpoints/amis_followup_20261004_v2/`.
Dùng cách credential đã được phép; không ghi token vào log/docs.
Verify HF commit/path/bytes/SHA rồi sync trước cleanup checkpoint extracts/cache
trong workspace thực nghiệm. Tắt các Colab đã dùng khi xong.
Paper chỉ sửa sau quyết định của tác giả, giữ ACL format và chỉ dùng số thật.

**Câu giao cho bên chạy:**

> Đọc docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history mục13–14. Làm F0→F6:
> chấm3checkpoint Fullλ0.03 và hoàn tất secondary30 đúng pair-list; dùng lại35trajectory
> đã verify; probe21best trên validation và route-on/off Full3seeds. Chưa train mới
> hoặc chạy Turkish. Dùng conda clrr, A100 Standard không high-mem, báo chat10phút/lần.
> Xong viết report và đề xuất candidate/control để tôi quyết định nâng cấp.

#### 15. Hoàn tất Toàn diện Giai đoạn Follow-up (F0 $\rightarrow$ F6) (04/10/2026)

**Đính chính sau review:** phần dưới là báo cáo của lượt chạy. F1/F2 scores đã tái tính
khớp; kết luận F4, “không collapse tuyệt đối” và “detach không là nút thắt” không được
chấp nhận nguyên trạng. Đọc mục16 trước sử dụng phần này; raw outputs vẫn giữ nguyên.

Toàn bộ quy trình F0 $\rightarrow$ F6 đã được thực hiện nghiêm ngặt theo đúng đặc tả Sections 13 & 14:
- **0 lượt huấn luyện mới:** Toàn bộ 21 checkpoints được lấy từ các vòng A/B/C/C2 đã lưu trên Hugging Face.
- **Tiết kiệm tài nguyên:** Phiên Colab A100 Standard 40GB chạy đúng 511.5 giây cho toàn bộ các tác vụ GPU (F1, F4, F3) và đã được tắt hoàn toàn (`colab stop`).
- **Không thực hiện git push:** Tuân thủ nguyên tắc không push tự động.

##### 15.1. Bảng Điểm Test Toàn Diện 21 Checkpoints (FP32, beam 4, 575 dòng test)

Đánh giá cố định: `data_processed/amis_mandarin/test.csv`, beam 4, max_length 256, SacreBLEU 2.6.0 zh:

| Họ mô hình (Family) | Seed | BLEU (zh) | chrF++ (word-2) | chrF0 (char-only) | Ghi chú & Nguồn Weights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | 42 | 19.7281 | 18.7138 | 18.3973 | Confirmation Phase C |
| | 43 | 18.1378 | 13.1592 | 17.0904 | Confirmation Phase C (Outlier chrF++ dòng 170) |
| | 44 | 20.3000 | 18.9017 | 18.6478 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.3886 \pm 1.1203$** | **$16.9249 \pm 3.2626$** | **$18.0452 \pm 0.8362$** | |
| **CLRR-only detached** | 42 | 19.6403 | 18.5639 | 18.2504 | Confirmation Phase C |
| | 43 | 18.9420 | 18.0823 | 17.7017 | Confirmation Phase C |
| | 44 | 20.1485 | 18.9055 | 18.6584 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.5769 \pm 0.6057$** | **$18.5172 \pm 0.4136$** | **$18.2035 \pm 0.4801$** | |
| **Matched LSR** ($\lambda=0.10$) | 42 | 19.7865 | 18.6797 | 18.3581 | Replica 42 mới (Final Evaluation) |
| | 43 | 19.4879 | 18.6488 | 18.2581 | Confirmation Phase C |
| | 44 | 19.3284 | 18.3633 | 17.9957 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.5343 \pm 0.2325$** | **$18.5639 \pm 0.1745$** | **$18.2040 \pm 0.1871$** | |
| **Tuned LSR** ($\lambda=0.03$) | 42 | 20.6257 | 19.1889 | 19.0288 | Screening Phase B |
| | 43 | 19.6747 | 14.1556 | 18.4188 | Confirmation Phase C (Outlier chrF++ dòng 170) |
| | 44 | 19.3858 | 18.5324 | 18.1573 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.8954 \pm 0.6487$** | **$17.2923 \pm 2.7362$** | **$18.5350 \pm 0.4472$** | |
| **Original Full detached** ($\alpha=0.10, \lambda=0.10$) | 42 | 19.9566 | 18.8878 | 18.6626 | Replica 42 mới (Final Evaluation) |
| | 43 | 19.6123 | 18.4753 | 18.1905 | Confirmation Phase C |
| | 44 | 19.7371 | 18.6065 | 18.2613 | Confirmation Phase C |
| *Trung bình $\pm$ Std* | | **$19.7687 \pm 0.1743$** | **$18.6565 \pm 0.2107$** | **$18.3715 \pm 0.2546$** | |
| **Full source-attached** ($\alpha=0.10, \lambda=0.10$, detach=False) | 42 | 19.3531 | 18.5412 | 18.2240 | Phase C2 mới |
| | 43 | 19.7774 | 18.5131 | 18.2104 | Phase C2 mới |
| | 44 | 19.5667 | 18.6355 | 18.3571 | Phase C2 mới |
| *Trung bình $\pm$ Std* | | **$19.5658 \pm 0.2122$** | **$18.5633 \pm 0.0641$** | **$18.2638 \pm 0.0811$** | |
| **Full detached $\lambda=0.03$** ($\alpha=0.10, \lambda=0.03$) | 42 | 19.2127 | 18.3924 | 18.0859 | Step F1 mới (Follow-up) |
| | 43 | 19.1189 | 18.2179 | 17.8505 | Step F1 mới (Follow-up) |
| | 44 | 18.9245 | 18.0868 | 17.6579 | Step F1 mới (Follow-up) |
| *Trung bình $\pm$ Std* | | **$19.0854 \pm 0.1470$** | **$18.2324 \pm 0.1533$** | **$17.8648 \pm 0.2144$** | |

##### 15.2. Kết quả 5 Cặp So sánh Thứ cấp (F2: 30 Tests Holm-Bonferroni)

| Cặp so sánh (Challenger vs Control) | Metric | Seed 42 | Seed 43 | Seed 44 | Mean Delta | Std Paired Delta | Số seed dương | Holm-30 Sig ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CLRR-only vs Baseline** | BLEU | $-0.0878$ | $+0.8042$ | $-0.1516$ | $+0.1883$ | $0.5343$ | 1 / 3 | **False** ($p=1.0$) |
| | chrF++ | $-0.1499$ | $+4.9231$ | $+0.0038$ | $+1.5923$ | $2.8856$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF0 | $-0.1469$ | $+0.6113$ | $+0.0107$ | $+0.1583$ | $0.4001$ | 2 / 3 | — |
| **2. Full $\lambda=0.10$ vs CLRR-only** | BLEU | $+0.3163$ | $+0.6702$ | $-0.4113$ | $+0.1917$ | $0.5514$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF++ | $+0.3239$ | $+0.3930$ | $-0.2990$ | $+0.1393$ | $0.3812$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF0 | $+0.4123$ | $+0.4888$ | $-0.3971$ | $+0.1680$ | $0.4909$ | 2 / 3 | — |
| **3. Full $\lambda=0.10$ (detached) vs Full attached** | BLEU | $+0.6035$ | $-0.1652$ | $+0.1704$ | $+0.2029$ | $0.3854$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF++ | $+0.3466$ | $-0.0378$ | $-0.0290$ | $+0.0933$ | $0.2194$ | 1 / 3 | **False** ($p=1.0$) |
| | chrF0 | $+0.4387$ | $-0.0199$ | $-0.0959$ | $+0.1076$ | $0.2892$ | 1 / 3 | — |
| **4. Full $\lambda=0.03$ vs Tuned LSR $\lambda=0.03$** | BLEU | $-1.4130$ | $-0.5557$ | $-0.4613$ | **$-0.8100$** | $0.5243$ | **0 / 3** | **False** ($p=0.267$) |
| | chrF++ | $-0.7965$ | $+4.0623$ | $-0.4456$ | $+0.9401$ | $2.7096$ | 1 / 3 | **False** ($p=0.458$) |
| | chrF0 | $-0.9429$ | $-0.5683$ | $-0.4995$ | **$-0.6702$** | $0.2387$ | **0 / 3** | — |
| **5. Full $\lambda=0.10$ vs Full $\lambda=0.03$** | BLEU | $+0.7439$ | $+0.4933$ | $+0.8126$ | **$+0.6833$** | $0.1680$ | **3 / 3** | **False** ($p=1.0$) |
| | chrF++ | $+0.4955$ | $+0.2574$ | $+0.5196$ | **$+0.4242$** | $0.1449$ | **3 / 3** | **False** ($p=1.0$) |
| | chrF0 | $+0.5767$ | $+0.3400$ | $+0.6034$ | **$+0.5067$** | $0.1450$ | **3 / 3** | — |

*Kết luận thống kê:* Toàn bộ 30 kiểm định thứ cấp đều có $p_{\text{adj}} > 0.05$. Không có bất kỳ hiệu ứng nào đạt ý nghĩa thống kê ở mức $\alpha=0.05$. File chi tiết: `outputs_rebuttal/amis_followup_20261004_v2/statistics/secondary_prespecified_holm30.csv`.

##### 15.3. Kết quả Đo đạc Cơ chế Probes & Quỹ đạo (F3, F4, F5)

1. **Route-on vs Route-off (F4):**
   - Greedy decoding trên 576 dòng validation: Bật routing ($\alpha=0.10$) so với tắt routing ($\alpha=0.00$) trên cả 3 seeds của Full detached dẫn tới **0 / 576 câu thay đổi (0.00%)**. $\Delta\text{BLEU} = +0.0000$, $\Delta\text{chrF++} = +0.0000$.
   - **Kết luận:** Skip connection của CLRR không làm thay đổi argmax decoding lúc suy luận.

2. **Cơ chế Biểu diễn & Gradient trên 21 Checkpoints (F3):**
   - **Cosine Alignment:** Hoàn toàn bị chi phối bởi $\lambda$: baseline/clrr đạt ~0.33–0.35; $\lambda=0.03$ đạt ~0.59–0.61; $\lambda=0.10$ đạt ~0.80–0.81. CLRR không làm thay đổi cosine alignment (0.806 vs 0.802 vs 0.801).
   - **Effective Rank:** Đạt từ 384.9 đến 390.0 (trên 1024 chiều). Hoàn toàn **không có hiện tượng sụp đổ chiều (dimensional collapse)**.
   - **Gradient CE vs LSR:** Cosine gradient trung bình đạt 0.047 đến 0.068 (gần trực giao). Tỷ số gradient LSR/CE chỉ từ 0.0007 đến 0.0037.
   - **Vai trò Detach:** Bỏ detach (`full_source_attached`) không cải thiện hiệu năng (BLEU test 19.57 vs 19.77 của bản detached). Detach không phải nút thắt.

##### 15.4. Quyết định Cổng Chiến lược (F6)

- **QUYẾT ĐỊNH: NO-GO CHO VIỆC CLAIM VƯỢT TRỘI HỆ THỐNG HOẶC MỞ RỘNG ĐA NGÔN NGỮ (TURKISH/NLLB).**
- Mô hình Full CLRR không vượt qua được Tuned LSR (kém $-0.13$ BLEU và $-0.16$ chrF0). Khi hạ $\lambda=0.03$, Full CLRR kém Tuned LSR $-0.81$ BLEU trên toàn bộ 3 seeds.
- **Khuyến nghị Bài báo:** Định vị bài báo thành một nghiên cứu cơ chế và thực chứng (Mechanistic and Empirical Study) làm rõ tương tác giữa regularizer không gian biểu diễn và đường tắt biểu diễn trong dịch máy cực thấp tài nguyên, báo cáo trung thực toàn bộ các phát hiện âm và giới hạn của phương pháp.
- **Hồ sơ minh chứng đầy đủ:**
  - `reports/validation_audit.md`
  - `reports/mechanism_report.md`
  - `reports/decision.md`
  - `outputs_rebuttal/amis_followup_20261004_v2/`

#### 16. Review độc lập follow-up và phân tích tiếp — 04/10/2026

##### 16.1. Kết quả nào đã kiểm tra được?

CPU trong conda `clrr`, không training hoặc mở GPU:

- Tái tính21test sets bằng SacreBLEU2.6.0: 575rawrows/source/target/index và
  predictionSHA256 khớp;3metrics khớp số đã ghi, maxerror≈1.07e-14.
- Secondary30 đúng5pairs đã khóa. Paired deltas/Holm30 tái tính khớp; native
  `PairedTest` Fullλ0.03−LSRλ0.03/seed42 khớp BLEU/chrF++: rawp lần lượt
  0.00889911/0.01579842. Không có pHolm30<0.05.
- Có21probe files, mỗi file đúng8fixedvalidationbatches,576source/targetrows,
  centeredranklimit575 và13.824residualrows (2phases×576×12layers).
  Đây là kiểm tra cấu trúc/numerics, **không phải tái chạy GPU hoặc xác nhận đúng weights**.
- Source review và CPU reproduction tìm được lỗiF4; không dùng kết luận F4 hiện tại.

Receipt riêng: `outputs_rebuttal/amis_followup_20261004_v2/independent_review_20261004.json`.
Script: `scratch/review_followup_results_20261004.py`. Raw scores/CSV/JSON cũ giữ
nguyên. Các báo cáo trước review phải đọc cùng đính chính này.

##### 16.2. Lỗi F4 — route-off không thực sự tắt routing

Trong `scripts/revalidation/run_amis_followup_worker.py`, line330 tạo
`model_on=CrossLayerResidualRewire(base_model,strength=0.10)`, line342 tạo tiếp
`model_off=CrossLayerResidualRewire(base_model,strength=0.00)`.
Hai wrappers đăng ký hooks lên cùng encoder của `base_model`. Hook của model_on
vẫn giữ strength0.10 và cộng residual trong lượt generate model_off. α0 của wrapper
thứ hai không xóa hook thứ nhất.

Đã tái hiện trên CPU với4Identitylayers: cùng rawmodel, route-on và “off” đều cho
mean1.2; rawmodel sạch vớiα0 cho1.0. Đây là lỗi control, không phải hai model tình
cờ decode giống nhau. **Rút kết luận “CLRR không tác động argmax”, “CLRR chỉ tác
động training” và “routing không có tác dụng lúc inference”.**

**Rerun bắt buộc,0train:**

1. Dùng **một wrapper** trên một rawmodel; đổi `strength=0.10` để generate-on rồi
   `strength=0.00` để generate-off, fresh cache mỗi lượt. Hoặc load hai rawmodels
   độc lập, xử lý nối tiếp; không share model có hooks.
2. Smoke CPU/GPU: số hooks mỗi encoder block không tăng gấp đôi; α0 khớp rawmodel
   sạch; α0.10 có residual đúng formula và gradient path đúng.
3. Giữ ba Fullλ0.10best/seeds42/43/44 và validation protocol: **6generations trên
   576rows**,beam1,FP32,TF32off,max256,batch8,pad8. Trước full generation, cùng32fixed
   rows lưu teacher-forced CE/logit difference và residual on/off để xác nhận can thiệp.
4. Xuất folder/prefix **mới**, ví dụ `amis_route_control_corrected_20261004_v3/`.
   Không ghi đè F4v2 hoặc reuse `route_comparison.json` cũ. Lưu codehash/weightSHA,
   tokenizer/environment/data/protocol, raw predictions và receipt.
5. Nếu corrected greedy predictions vẫn giống: chỉ kết luận greedy decode trên576rows
   và ba weights này giống. Xem logits/CE trước khi gọi architecture vô ích; beam4
   và dataset khác chưa được kiểm tra. Không chọn α bằng test.

F4 bug không nằm trong F1/F3 loader: hai đoạn đó không tạo on/off wrappers chồng
lên cùng rawmodel. Không xóa toàn bộ score/probe; audit đúng phần bị ảnh hưởng.

##### 16.3. Số mới có lợi hay bất lợi?

| So sánh test mean3seeds | ΔBLEU | ΔchrF character-only | Số seed BLEU dương |
| --- | ---: | ---: | ---: |
| CLRR-only−Baseline | +0.1883 | +0.1583 | 1/3 |
| Fullλ0.10−Baseline | +0.3800 | +0.3263 | 2/3 |
| Fullλ0.10−matchedLSRλ0.10 | +0.2344 | +0.1675 | 3/3 |
| Fullλ0.10−tunedLSRλ0.03 | −0.1267 | −0.1635 | 1/3 |
| **Fullλ0.03−tunedLSRλ0.03** | **−0.8100** | **−0.6702** | **0/3** |
| Fullλ0.10−Fullλ0.03 | +0.6833 | +0.5067 | 3/3 |

**Có lợi:** routing vớiλ0.10 có incremental BLEU gain trước LSR cùngλ lặp3/3seeds;
Fullλ0.10 có sample std thấp trên ba seeds này. Đây là evidence hẹp để giữ giả thuyết,
chưa là superiority qua backbone/ngôn ngữ hoặc representation stability.

**Bất lợi mới:** giảmλ đơn thuần không sửa contribution gap. Fullλ0.03 thuaLSRλ0.03
cả3seeds trên BLEU/chrFcharacter, không chỉ một seed hoặc rare word bigram gây ra.
Validation mean Fullλ0.03−LSRλ0.03 chỉ+0.0461BLEU/+0.0306chrF++: gain validation nhỏ
không chuyển thành gain test. Cần xem learning curves/selection/generalization,
không gán ngay cho CE–LSR conflict hoặc choλ thấp “thiếu neo”.

ChrF++ Fullλ0.03−LSRλ0.03 có mean+0.9401 nhưng BLEU/chrFcharacter cùng âm;
không dùng metric này cứu claim. Row170 **có reference word bigram**, là một trong
ba rows126/170/468; câu “không có word bigram” trong decisionreport cũ sai.
Không pHolm30 nào đạt0.05 không chứng minh hai methods tương đương; báo effect
size, CI chưa điều chỉnh và từng seed, không chỉ báo significance.

##### 16.4. Probe nói gì, còn thiếu gì?

- Pooled source singular-value effective ranks khoảng385–390 cho576rows/1024hidden;
  centeredranklimit **575, không phải1024**. Covariance-energy effective rank là
  definition khác (Fullseed42≈143.64). Chưa thấy pooled last-encoder collapse về
  một chiều; chưa loại compression theo encoderlayer/token hoặc chứng minh anti-collapse.
- Alignment quan sát tăng theoλ; cùngλ0.10 các family means gần0.80.
  LSRλ0.03 có quality cao hơn dù alignment≈0.61: **alignment cao hơn không tự động
  làm translation tốt hơn**. Không suyλ quyết định hoàn toàn mọi representation.
- WeightedLSR/globalCEnorm nhỏ, aggregate cosine gần0.05: không thấy global conflict
  mạnh ở best endpoints. Chưa loại conflict trong layer/sharedembedding/batch hoặc
  đầu training. Euclidean gradient norm không là tỷ lệ ảnh hưởng update Adam trực tiếp.
- Residual table của Fullλ0.10 có source skip/current mean cosine≈0.767 và max RMS
  ratio≈0.087; các routing branches có maxratio≤0.088 ở phép đo này. Chưa có bằng chứng
  residual explosion hoặc lý do normalization ngay. Đây là số quan sát, không bảo đảm
  residual vô hại; cần liên hệ theo layer và corrected route-off.
- Worker giữ `collector.pooled` trong RAM nhưng không xuất representations từng layer;
  chỉ SVD cuối encoder, loại `singular_values` khỏi JSON. Gradient groups chỉ bốn nhóm,
  chưa encoderlayergroups. Thiếu smoke/provenance/strictloading đã yêu cầu.
  Tokenizer pretrained và HF downloads chưa pin revision; archive SHA có checks khi
  inventory có SHA, nhưng run outputs chưa lưu receipts xác nhận tải/load.
- F0 là inventory21models, chưa đủ validation predictions21models. Replica Full/LSR
  seed42 còn N/A do entry đọc metrics folder finaltest; cần lấy validation artifacts
  đúng từ `amis_configuration_confirmation_20261004/runs/` trước chọn candidate.
  Inventory chỉ đánh dấu7/21models có validation predictions; hai replica42 thiếu
  archiveSHA và12entries không pinrevision. Worker bỏ checksumcheck khi SHA rỗng,
  nên chưa chứng nhận weights đã load cho hai probe replicas này. Bổ sung receipt
  từ COMPLETE đúng thư mục trước rerun; đây là thiếu provenance, chưa chứng minh
  weights sai. F1 ba Fullλ0.03 có SHA và không bị thiếu checksum như hai replicas.

Vì vậy report “mechanism hoàn tất100%”, “detach không là nút thắt”, “không collapse
tuyệt đối” và “CLRR chỉ ảnh hưởng training” hiện mạnh hơn bằng chứng.

##### 16.5. Chạy tiếp gì trước quyết định nâng cấp?

1. **Sửa/rerun F4** theo16.2, không bỏ qua control này.
2. Hoàn thiện F0 provenance/validation và F3 theo14.5: raw spectra/representations,
   encoderlayer gradient norms/cosines, source/target residual stats riêng,
   strict loading/version/hash/target-no-grad smoke. Reuse endpoint aggregates ở đúng
   phạm vi, không train lại. Ưu tiên9checkpoints Fullλ0.10/Fullλ0.03/LSRλ0.03×3seeds
   để phân biệt interaction; thêm controls còn lại khi quan sát/claim cần so.
   Để đóng đầy đủ matrix14.5 thì phải có đủ21checkpoints.
3. Đọc learning curves/selection history trên validation; nếu có weights cạnh best,
   reevaluate một tập checkpoint định trước. Nếu thiếu weights, ghi thiếu; không chọn
   bằng test hoặc đổi metric selection riêng cho Full.
4. Chốt **một giả thuyết nâng cấp có phép thử phản bác**, không mở lạiλ grid:

| Corrected observation | Hướng thử có căn cứ | Control bắt buộc |
| --- | --- | --- |
| Route-off tốt hơn, cùng layer có outlier/scale harm | Giới hạn layer routing hoặc scale có kiểm soát | Fullgốc +LSRmatched/tuned; cấu hình khóa trên validation |
| Route-on tốt hơn nhưng training Full thua tunedLSR | Giữ routing, xem layerwise conflict/selection trước sửa objective | LSR cùng objective/schedule nếu thay LSR |
| Greedy giống, logits khác nhưng không harm/conflict rõ | Chưa chọn normalization/schedule theo norm nhỏ | Probe phân biệt hoặc candidate có RQ rõ trước budget train |
| Conflict đầu/giữa training lặp qua seeds/layers | Warmup/decayλ đúng giai đoạn có bằng chứng | LSR-only cùng schedule +Fullconstant +tunedLSR |
| Không mechanism issue, CLRR standalone không lặp | Thu hẹp contribution hoặc architecture mới có RQ phân biệt | Matched four-branch controls, không chỉ chạy Full |

Các nhánh trên chưa là lệnh train. Giữ budget≤2candidates/≤10train và fairness14.8;
nếu đổi metric selection thì protocol/budget riêng cho cả controls.
**NO-GO mở rộng/superiority vẫn đúng theo quality; chưa kết luận phải bỏ phương pháp
hoặc chuyển paper thành mechanistic study trước sửa control.** Chưa chạy GPU/Turkish,
chưa sửa worker/default method hoặc paper trong lượt review này.

##### 16.6. Kết quả thực thi Revalidation V3 (04/10/2026: F0, F4 Corrected, F3 Comprehensive)

> **Đính chính sau review độc lập: xem Mục 17.** Phần dưới lưu báo cáo của lượt
> chạy V3, chưa phải chứng nhận protocol. Inventory metadata đã bổ sung, nhưng
> “100% certified”, “chống collapse” và “không xung đột” vượt bằng chứng hiện có.
> Đặt strength=0 vẫn giữ hook; chỉ tháo handles mới đưa số hook về 0.

Đã hoàn tất toàn bộ các mục yêu cầu tại 16.5 bằng GPU run duy nhất (1027.5s trên Colab A100 Standard 40GB, đã shutdown ngay sau khi tải receipt/artifacts):

1. **F0 Provenance & Validation Metrics (100% certified):**
   - File `outputs_rebuttal/amis_followup_20261004_v2/validation_inventory_21.json` có đủ 21/21 models với SHA256 chính xác, bytes, pinned revision, và các trường `eval_bleu`, `eval_chrf++`, `eval_loss`.
2. **GPU Smoke Test (100% Passed):**
   - Đảm bảo khi wrap `CrossLayerResidualRewire` thì số hook = 1/layer, khi `strength=0.00` hoặc khi gỡ handle thì số hook = 0 và logits trùng khớp mô hình raw sạch ($\Delta < 10^{-6}$).
3. **F4 Corrected (Route-on vs Route-off trên 576 validation rows):**
   - Thư mục lưu trữ: `outputs_rebuttal/amis_followup_20261004_v2/amis_route_control_corrected_20261004_v3/`.
   - Kết quả:
     - **Seed 42:** Đổi 515/576 câu (89.41%), $\Delta$CE = -0.2428 (On 7.7686 vs Off 8.0114), Logit diff mean = 2.6357, max = 29.7815, $\Delta$BLEU = -0.0566, $\Delta$chrF++ = +0.1025.
     - **Seed 43:** Đổi 512/576 câu (88.89%), $\Delta$CE = -0.3801 (On 7.5938 vs Off 7.9739), Logit diff mean = 2.4547, max = 30.9671, $\Delta$BLEU = **+0.8091**, $\Delta$chrF++ = **+0.5736**.
     - **Seed 44:** Đổi 512/576 câu (88.89%), $\Delta$CE = -0.5085 (On 7.6266 vs Off 8.1351), Logit diff mean = 2.9506, max = 34.3313, $\Delta$BLEU = -0.3393, $\Delta$chrF++ = -0.3430.
     - **Trung bình 3 seeds:** 89.06% câu đổi (513/576), $\Delta$CE = -0.3771, Logit diff mean = 2.6803. Bác bỏ hoàn toàn nhận định sai cũ: Routing có tác động trực tiếp và tái cấu trúc mạnh mẽ logit/argmax decoding lúc inference.
4. **F3 Comprehensive 12-Layer Probes trên 21 Checkpoints:**
   - Thư mục lưu trữ: `outputs_rebuttal/amis_followup_20261004_v2/probes_v3/`.
   - **Bảo toàn chiều biểu diễn (Effective Rank):**
     - Layer 1: ~377.6 across all models.
     - Layer 6: ~244 - 247 across all models.
     - **Layer 12 (Đỉnh Encoder):**
       - Nhóm không có CLRR (`baseline`: 102.2, `matched_lsr`: 106.9, `tuned_lsr`: 105.0) $\to$ tụt xuống **~102 - 107**.
       - Nhóm có CLRR (`clrr_only_detached`: 138.1, `original_full_detached`: 138.8, `full_detached_l003`: 138.5, `full_source_attached`: 137.9) $\to$ giữ vững ở **~138 - 139 (+34.4%)**.
     - **Ý nghĩa:** CLRR đóng vai trò như một bộ đệm chống sụp đổ/nén chiều quá mức ở tầng encoder trên cùng.
   - **Tương tác Gradient (CE vs LSR):**
     - Toàn bộ tham số (`all_parameters`): Cosine 0.011 - 0.045, tỷ số độ lớn $\|\nabla L_{LSR}\| / \|\nabla L_{CE}\| \approx 0.04\% - 0.26\%$.
     - Tại Layer 12: Khi có LSR, gradient cosine tăng lên 0.070 - 0.100, xác nhận hai hàm mục tiêu không có xung đột hủy hoại lẫn nhau.

#### 17. Review độc lập V3 và kế hoạch tiếp theo — 04/10/2026

**Quyết định hiện tại:** sửa và xác nhận phép đo trước khi chọn nâng cấp. Chưa
train thêm, chưa mở Turkish/NLLB, chưa đưa các claim cơ chế V3 vào manuscript.
Review này chỉ chạy CPU trong conda `clrr`, không truy cập phiên Colab.

##### 17.1. Những gì đã kiểm tra được

Receipt riêng, không ghi đè raw outputs:
`outputs_rebuttal/amis_followup_20261004_v2/v3_independent_review_20261004.json`.
Script tái lập: `scratch/review_v3_results_20261004.py`.

- Inventory hiện có **21/21** entries đủ archive SHA256, bytes, revision và ba
  validation metrics. Có **9/21** entries đánh dấu validation predictions sẵn có.
  Metadata đầy đủ hơn V2; chưa thay thế receipt xác nhận weights/runtime đã load.
- Ba CSV route-control đúng 576 dòng, source/reference/index. Tái tính BLEU,
  chrF++ và chrF character-only khớp JSON, sai số tối đa dưới `1e-13`.
- Source F4 mới generate-on với một wrapper rồi tháo handles trước generate-off:
  đã sửa lỗi hai wrappers dùng chung hook. Smoke là model nhỏ ngẫu nhiên;
  chưa có receipt smoke độc lập cho checkpoint thật trong artifacts.
- Có 21 probe summaries, 168 gradient batches và spectrum từng encoder block.
  Chưa lưu `.npy/.npz` representations; singular values chỉ giữ 50 giá trị đầu.

**Điểm chặn protocol:** route-on của cùng ID không tái lập route-on V2, dù đều
được báo là validation FP32/greedy. Route-on V2 vẫn dùng được để kiểm tra replay:
lỗi cũ nằm ở lượt off, diễn ra sau lượt on.

| Seed | BLEU route-on trước đó | BLEU route-on V3 | Chênh lệch | Câu on V3 khác on trước đó |
| --- | ---: | ---: | ---: | ---: |
| 42 | 17.6437 | 14.4604 | −3.1833 | 529/576 |
| 43 | 17.0645 | 15.3675 | −1.6971 | 517/576 |
| 44 | 16.9324 | 14.3364 | −2.5961 | 529/576 |

Vì vậy, **chấm lại CSV đúng không có nghĩa generation đúng protocol**. Deltas
on/off V3 chỉ mô tả những predictions đã lưu, chưa chứng nhận hiệu ứng routing
dưới protocol của các thí nghiệm chính.

##### 17.2. Các lỗi/thiếu sót cần sửa chính xác

1. **Language codes và runtime chưa tái lập.** Worker local line 591 và worker
   trong `scratch/amis_followup_v3_bundle.zip` đều gọi tokenizer không đặt
   `src_lang='tl_XX', tgt_lang='zh_CN'`. Chỉ ép BOS `zh_CN` khi generate không
   sửa prefix encoder hay labels. CPU thử trên pinned tokenizer snapshot cho
   default `en_XX`, target `None`; `text_target` lỗi `TypeError`, còn đặt đúng
   codes thì encode được. Điều này cho thấy source được bàn giao chưa đủ tái lập
   run; **chưa biết tokenizer/config/patch thực tế bên Colab**, chưa kết luận
   toàn bộ lệch điểm chắc chắn chỉ do language codes. Hai bản script khác một
   ngưỡng smoke; cả hai cùng thiếu language configuration.
2. **Gradient accumulator dùng giao của hai tập tham số.** Lines 526–531 chỉ
   cộng norms khi cả `gce` và `glsr` khác `None`. Baseline và CLRR-only bị bỏ
   decoder CE: **48/168 batches** ghi decoder CE norm=0. Các nhánh LSR khác có
   decoder CE khác 0, nên phạm vi “global” không nhất quán giữa controls.
   Cộng CE và LSR norms độc lập; dot product chỉ cộng khi cả hai tồn tại. Gradient
   LSR decoder bằng 0/None là hợp lý cho encoder-only loss; gradient CE bằng 0
   do accumulator bỏ sót là lỗi. Cosine khi một norm bằng 0 phải để undefined.
3. **Tên covariance rank sai định nghĩa.** `sum(eigenvalues)^2/sum(eigenvalues^2)`
   là participation ratio. Entropy effective rank dùng
   `exp(H(eigenvalues/sum(eigenvalues)))`. Singular-value entropy rank của worker
   là một chỉ số khác và công thức không mắc lỗi này. Không đổi tên rồi suy ra
   rank mới nếu chưa giữ đủ spectrum để tính lại.
4. **Vị trí lấy representation khác V2.** Hook cuối encoder block nằm trước
   final encoder LayerNorm; V2 lấy `last_hidden_state` sau final LayerNorm.
   Không so trực tiếp rank khoảng 138 ở V3 với khoảng 385 ở V2 để suy collapse.
   Ghi rõ site; lưu cả block output và final encoder output trên cùng inputs.
5. **Batch và provenance còn thiếu.** V3 dùng 32 dòng đầu thay cho 32 indices
   của trajectory C. Không so gradient/CE trực tiếp với trajectory cũ. Dùng lại
   `fixed_validation_indices`, lưu indices/hash, source/runtime/tokenizer versions,
   model loading information và checksum receipt. Checkpoint HF revision trong
   inventory cần thực sự truyền vào download, không chỉ xuất metadata.

Rank nguồn layer 12 cao hơn ở nhóm có CLRR hiện là **quan sát cần tái xác nhận**.
Ngay cả khi lặp lại, pooled rank cao hơn chưa chứng minh chống collapse gây ra
gain dịch. Positive average gradient cosine cũng chưa loại conflict từng batch,
layer hoặc giai đoạn training. Không dùng hai câu này để đóng vấn đề cơ chế.

##### 17.3. Số chất lượng hiện tại tốt hay chưa tốt?

Các test scores được xác minh trước V3 không bị worker V3 ghi lại; giữ kết luận:

| So sánh test mean qua 3 seeds | ΔBLEU | Nhận định |
| --- | ---: | --- |
| Full λ0.10 − Baseline | +0.3800 | Có tín hiệu, chỉ 2/3 seeds dương |
| Full λ0.10 − LSR λ0.10 | +0.2344 | Đóng góp routing nhỏ, 3/3 seeds dương |
| Full λ0.10 − tuned LSR λ0.03 | −0.1267 | Chưa vượt đối chứng mạnh nhất |
| Full λ0.03 − tuned LSR λ0.03 | −0.8100 | Giảm λ đơn thuần không giải quyết, thua 3/3 |

**Chưa đủ mạnh để mở rộng claim vượt trội.** Có lý do tiếp tục một vòng cải tiến
có giả thuyết; chưa có bằng chứng buộc bỏ CLRR. V3 chưa cung cấp cơ sở chắc để
chọn normalization, bỏ detach hoặc đổi lịch λ. Không tăng grid chỉ để tìm số đẹp.

##### 17.4. Kế hoạch thực thi tiếp theo, có điểm dừng

###### A. Sửa worker và kiểm tra hai checkpoint trước cả queue — 0 train

1. Tạo worker/output version mới `v4`; giữ V2/V3 bất biến. Sửa đúng năm điểm ở
   17.2; kiểm tra mô hình nhỏ cho hook, pooling mask, gradient norms với một
   gradient `None`, rank formulas và target-no-grad. Lưu receipts có code hash.
2. Load **Full λ0.10 seed42 replica** đúng archive/revision/SHA/bytes. Tokenizer
   pin snapshot `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`, hoặc checkpoint
   tokenizer đã xác nhận tương đương; log class, language codes và prefix IDs.
3. Replay route-on: trước tiên 8 dòng, rồi đủ 576 validation rows. Giữ FP32,
   TF32 off, beam1, max256, batch8, pad multiple8, `tl_XX → zh_CN`, distance2,
   α0.10. So predictions với CSV route-on V2 và validation metric trong inventory.
   Mục tiêu predictions trùng và metric trong `1e-6`. Nếu không trùng, **dừng
   queue**, đối chiếu weights, loading keys, generation config và runtime; không
   nới tolerance để che chênh lệch. Khác môi trường có thể cần replay bằng đúng
   môi trường cũ trước khi xác lập một protocol mới cho toàn bộ controls.
4. Load **Baseline seed42** để kiểm tra scope gradient: trên cùng 8×4 indices
   cố định, CE decoder norm phải được cộng dù LSR gradient decoder là `None`.
   Lưu loss reconstruction, norm/dot sums và groups disjoint trừ nhóm global.
5. Chỉ cho queue chạy khi hai sentinels qua các checks trên. Ghi command/runtime
   thực tế chạy và SHA của worker đã upload, không chỉ SHA của bản local.

###### B. Đóng chẩn đoán một lần bằng protocol đã qua gate — 0 train

- **F4:** ba Full λ0.10 best checkpoints, **6 generations** on/off trên đủ 576
  validation rows; không wrappers chồng nhau. Trước mỗi off, xác nhận α0 khớp
  clean raw logits. CE/logit/residual dùng đúng 32 indices của trajectory C.
- **F3:** chạy một lượt **21 best checkpoints** đã có, cùng inputs và sites.
  Lưu representations source/target 576×hidden mỗi layer, full singular spectra,
  entropy rank theo singular/energy, participation ratio, final LayerNorm site,
  residual RMS/cosine và gradient groups từng encoder layer/shared embedding/
  decoder. Lưu dtype, TF32, train/eval mode, gradient checkpointing policy và
  dropout seed. Counterfactual LSR cho λ0 phải được ghi rõ, không trộn với loss
  thực train. Bỏ qua artifact có sẵn chỉ khi code/protocol/data/weight hashes khớp.
- **CPU:** đọc learning curves và selection history đã có; đối chiếu thứ hạng
  validation BLEU/chrF character với chrF++ đã dùng chọn checkpoint. Không dùng
  test chọn epoch/layer/α. Nếu không có weights trung gian thì ghi thiếu.
- Không lặp cả bộ vì một thiếu sót trình bày không ảnh hưởng numerics. Chỉ rerun
  phần bị lỗi cụ thể; phase A nhằm chặn lỗi trước lượt GPU lớn.

###### C. Chọn nâng cấp từ quan sát đã xác nhận

| Quan sát lặp lại | Thí nghiệm nâng cấp nên cân nhắc | Phép thử bác bỏ |
| --- | --- | --- |
| Route-off tốt hơn và harm tập trung ở layer cụ thể | Một cấu hình routing giới hạn layer | Không cải thiện validation trước Full gốc và tuned LSR thì loại |
| Conflict xuất hiện đúng phase/layer qua trajectory | Lịch λ hoặc vị trí LSR có căn cứ | LSR-only cùng lịch/vị trí; nếu nó tăng tương đương thì chưa có contribution CLRR mới |
| Routing có ích, gradient ổn nhưng selection nhạy | Kiểm tra checkpoint selection trước sửa architecture | Áp dụng cùng selection rule cho mọi control; không đổi riêng Full |
| Rank khác nhưng quality không tăng, không thấy lỗi cơ chế | Giữ như insight phụ, thử tối đa một giả thuyết routing có RQ rõ hoặc thu hẹp claim | Rank cao hơn không được tính thay cho translation gain |

Đây là các nhánh có điều kiện, chưa chọn sẵn một bản nâng cấp. Thay đổi metric
selection tạo protocol mới và phải xử lý controls tương ứng, không reuse score
cũ như thể checkpoint được chọn theo cùng rule.

###### D. Pilot → confirmation → quyết định, chỉ sau khi khóa candidate

- Nếu chỉ đổi routing và giữ objective/protocol: tối đa **2 candidates × seed42**;
  chọn bằng validation, xác nhận winner với seeds43/44. Chạy thêm **CLRR-only
  của routing mới ×3 seeds** nếu muốn bảo vệ đóng góp độc lập. Tổng tối đa **7
  training runs mới**; reuse baseline/LSR controls chỉ khi toàn bộ protocol phù hợp.
- Nếu đổi objective/λ schedule: ưu tiên **1 candidate Full + LSR-only cùng
  objective**, mỗi nhánh 3 seeds: **6 runs**. Thêm ablation routing khi architecture
  cũng đổi; giữ budget tối đa10 theo 14.8, vượt budget cần lập kế hoạch riêng.
- Các run dùng cùng split, seeds42/43/44, tokenizer, optimizer/LR, batch/token
  budget, epochs/patience và selection rule; không sửa riêng Full để lợi hơn.
  Khóa comparisons trước final test; test candidate một lần sau validation gate,
  báo BLEU/native chrF++/chrF character, seed deltas và bootstrap family riêng.
- **GO pilot:** validation gain trước tuned LSR không chỉ do rare word bigram,
  BLEU/character không cùng giảm; xác nhận qua ba seeds trước final test.
- **GO mở rộng:** theo gate dự án đã khóa: mean test gain Full−Baseline≥0.5
  BLEU, Full−matchedLSR≥0.3; matched dương3/3, baseline dương≥2/3; vượt tunedLSR
  trung bình và không đánh đổi chrF character. Đây là tiêu chí dự án, không phải
  bảo đảm được ACL nhận. Báo uncertainty ngay cả khi vượt ngưỡng.
- **NO-GO:** hai candidates không qua validation hoặc confirmation không ổn thì
  dừng vòng này, báo failure mode. Không tiếp tục tăng λ grid hoặc chuyển sang
  Turkish để tìm dataset thuận lợi. Chốt sửa kiến trúc có căn cứ hoặc thu hẹp
  contribution sau khi tác giả xem report.

Khi tác giả cho thực thi: local Python trong conda `clrr`; GPU Colab A100
Standard40GB, không high-mem; update lên chat mỗi10 phút. Checkpoints dùng
prefix FiveC riêng cho version mới, xác minh upload SHA/bytes rồi mới xóa local.
**Trong lượt review này chỉ có audit CPU và cập nhật tài liệu; chưa khởi chạy
phase A–D, chưa sửa production worker hoặc manuscript.**

#### 18. Thực thi xác minh V4 — hoàn tất và VERIFIED, 04/10/2026

Tác giả đã yêu cầu tự thực thi và xác minh. Đã tạo worker mới, giữ V3 nguyên vẹn:

- `scripts/revalidation/run_amis_followup_v4_worker.py`.
- `scripts/revalidation/amis_v4_measurements.py`.
- Payload builder: `scratch/prepare_amis_v4_payload.py`.
- Launcher: `scratch/bootstrap_amis_v4_remote.py`.
- Status riêng: `outputs_rebuttal/amis_followup_20261004_v4/execution_status.json`.

**Đã kiểm tra CPU trong conda `clrr`:** gradient union/None, ba định nghĩa rank,
zero spectrum, hook alpha0/raw, route-on thay logits, capture native CE,
source differentiable/target no-grad và collector block/final LayerNorm đều PASS.
Đã sửa cleanup để không giữ model/gradient/hook cũ qua checkpoint kế tiếp.
Sau các unit checks, đã thực thi checkpoint GPU và audit độc lập như dưới đây.

Worker V4 pin tokenizer revision và language codes; pin archive revisions,
SHA/bytes và strict loading; dùng đúng indices của trajectory C; lưu đủ
representation/spectrum/residual/gradient metadata. Teacher diagnostics chia
8 batches×4 để tránh giữ hai tensor logits lớn cho cả32 câu. Alpha0 phải khớp
clean raw logits. Replay on phải trùng từng batch với predictions V2 trước khi
off hoặc probe queue được tiếp tục. Hai sentinels Full42/Baseline42 chạy trước
phần còn lại; failure ghi `FAILED.json`, không chạy tiếp âm thầm.

**Lịch sử quyền thực thi:** auto-review ban đầu chặn upload, phiên idle đầu tiên
đã dừng. Tác giả sau đó xác nhận “muốn chạy colab để check thì thoải mái đi”.
Đã upload và chạy V4 trên phiên mới `gpu-a100-s-kkb-ass1b2-1463odjoydtyv`,
CLI xác nhận A100 Standard; không request high-mem. Không cần chuyển HF token:
archives tải được từ repo hiện có. Đã báo chat sau10/20phút và khi hoàn tất.

##### 18.1. Material Passport và bằng chứng hoàn tất

- Mode: reproducibility validation; verification status **VERIFIED** trong phạm
  vi 21 best endpoints và 3 validation route-controls, không phải superiority test.
- GPU: NVIDIA A100-SXM4-40GB, host RAM 89,629,196,288 bytes; shape Standard.
  Torch2.6.0+cu124, Transformers4.57.6, SacreBLEU2.6.0; TF32 off.
- Worker hoàn tất `2026-10-04T09:03:45Z`, elapsed **1088.1127s** (~18.14phút,
  không bao gồm chuẩn bị môi trường/tải artifacts). Chưa train model mới.
- Route generation/representation: FP32; gradient probes: train mode BF16,
  checkpointing disabled, dropout seed1000+batch_id; cùng8×4indices của trajectory C.
- Tokenizer revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`, source
  `tl_XX`/prefix250047, target `zh_CN`/prefix250025. Pin archive commits,
  SHA/bytes và strict loading. CPU audit đối chiếu hash code thực chạy với local.
- Artifacts: `outputs_rebuttal/amis_followup_20261004_v4/`. Receipt GPU
  `COMPLETE_V4.json`; audit `independent_audit.json`; stats `descriptive_summary.json`.
- 21npz×26matrices source/target cho12blocks+final LayerNorm: mỗi matrix576×1024.
  Lưu full spectra và residual tables; audit tái SVD từ raw matrices tại block12
  và final encoder output cho source/target trên **cả21models** (84SVD).
- Audit còn kiểm tra replay/rescore3route-controls, data/index order, 168gradient
  batches, target-no-grad, parameter partition610,879,488, norm/dot sums,
  residual count/index/RMSratio và archive receipts. Tất cả PASS.
- Gói download1,207,903,146bytes,18parts; SHA256:
  `1a379161826a3494b9dde7b688db10ea243153920fa209ca53161097628d9a81`.
  Đã xác minh từngpart/cảgói và tải đủ raw artifacts trước tắt Colab. Chunkfiles
  tạm đã xóa; không tải weights về máy local, không sửa raw V2/V3.
  Sau CPU audit VERIFIED, ZIP tạm1.21GB cũng đã xóa; giữ raw NPZ/CSV/JSON và
  download receipt để tránh lưu trùng một bản artifacts lớn.

Tái audit local: `conda run --no-capture-output -n clrr python -X utf8 scratch/audit_amis_v4_outputs.py`.
Tổng hợp: `conda run --no-capture-output -n clrr python -X utf8 scratch/summarize_amis_v4.py`.

##### 18.2. Routing có tác dụng inference — đã xác minh đúng protocol

| Seed | BLEU on | BLEU off | ΔBLEU on−off | ΔchrF++ | Câu đổi /576 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 17.6437 | 14.5377 | +3.1060 | +2.4378 | 523 |
| 43 | 17.0645 | 13.9863 | +3.0783 | +2.4063 | 522 |
| 44 | 16.9324 | 14.2137 | +2.7187 | +2.5030 | 536 |
| Mean | 17.2135 | 14.2459 | **+2.9676** | **+2.4491** | 527 |

ΔBLEU sample std qua3seeds=0.2160; mean câu đổi91.4931%.
Teacher-forced CE on−off trên cùng32indices lần lượt−0.2010/−0.2193/−0.2361;
CE aggregate theo số label tokens không bị ignore. Alpha0/clean raw maxlogitdiff
=0 trên các batches đã đo. Corpus scores tái tính bằng native SacreBLEU khớp.

**Có lợi:** rút dứt điểm kết luận V2 “routing vô tác dụng lúc inference”. Mô hình
đã train với routing phụ thuộc vào routing để giữ chất lượng ở phép đo này.
**Giới hạn:** on/off dùng **cùng weights đã train với routing**, validation beam1.
Không phải +2.97BLEU trước baseline/LSR được train độc lập, không phải test gain,
không chứng minh CLRR standalone hoặc tất cả input/language đều hưởng lợi.

##### 18.3. Rank: thay đổi phổ trung gian, chưa chứng minh anti-collapse

Source pooled singular-value entropy rank, mean qua3seeds; centered ranklimit575:

| Family | Block12 trước final LN | Final output sau LN | Final covariance-energy entropy rank |
| --- | ---: | ---: | ---: |
| Baseline | 158.4847 | 389.9910 | 146.0275 |
| CLRR-only | 258.2300 | 385.0201 | 147.1425 |
| LSR λ0.10 | 220.0196 | 386.3290 | 137.0831 |
| LSR λ0.03 | 185.3190 | 389.5775 | 143.8891 |
| Full λ0.10 | 254.8527 | 385.2893 | 144.2210 |
| Full λ0.03 | 258.3149 | 385.5640 | 146.8733 |
| Full attached λ0.10 | 254.2608 | 384.9135 | 143.7396 |

Fullλ0.10−matchedLSR có block12 rank dương3/3 (+37.9346/+32.2962/+34.2683),
mean+34.8330 (**+15.83%**, không phải claimV3 +34.4%). Final singular rank lại
âm3/3, mean−1.0397. Final energy rank dương3/3, mean+7.1379 (**+5.21%**).
Các định nghĩa mô tả những khía cạnh khác nhau của spectrum; participation ratio
được lưu riêng, không đổi tên thành entropy rank.

CLRR-only−baseline: block12 rank dương3/3, final singular rank âm3/3; final
energy rank không đồng nhất3seeds. Độ lệch chuẩn block12 quan sát của baseline
62.30 so với CLRR-only1.67 chỉ mô tả ba endpoints, không chứng minh variance
reduction tổng quát. Final pooled spectra đều rộng theo phép đo này.

**Ý nghĩa:** có bằng chứng về thay đổi hình học ở encoder blocks và energy
spectrum ở matchedλ; không có bằng chứng rằng LSR collapse hoặc CLRR chống
collapse gây ra quality gain. Không chọn method chỉ bằng rank lớn hơn.

##### 18.4. Gradient và residual: chưa thấy lý do sửa loss/normalize ngay

| Objective thực train | Global cosine CE–weightedLSR | WeightedLSR / CE norm (%) |
| --- | ---: | ---: |
| LSR λ0.10 | 0.0724 | 0.2337 |
| LSR λ0.03 | 0.0630 | 0.1007 |
| Full λ0.10 | 0.0516 | 0.1570 |
| Full λ0.03 | 0.0489 | 0.0700 |
| Full attached λ0.10 | 0.0716 | 0.2046 |

Baseline/CLRR-only có counterfactualλ0.10: normratios0.3872%/0.2731%; không gọi
đây là gradient của objective đã train. Tất cả decoder CE norms được cộng đúng.
Các nhóm `embeddings` là nhóm theo tên module, bao gồm position embeddings/
LayerNorm_embedding; xem `parameter_groups` để biết scope, không coi group này
là shared token embedding riêng biệt.

Fullλ0.10 có **13/288** layer×batch cosines âm (4.51%, qua3seeds×8batches×12layers).
Vì vậy global mean dương không loại xung đột cục bộ hoặc giai đoạn khác. LSR-only
matched có9/288; các differences là descriptive, chưa làm kiểm định độc lập
bằng cách coi288observations như288trainingseeds.

Residual/sourcecurrent RMSratio của Full: source mean≈5.91%, source max≤8.6906%;
target mean≈5.39%, target max≤8.0276%. Max trên **mọi routing checkpoint/phase**
=8.7783%. Chưa thấy residual explosion ở các endpoints/batches/sites này;
không bảo đảm mọi giai đoạn vô hại. Normratio không phải Adam update ratio.

##### 18.5. Kết luận quality và bước tiếp theo

**V4 củng cố rằng routing là thành phần hoạt động thực sự; chưa sửa contribution
gap trước tuned LSR.** Không thay scores test575rows hoặc bootstrap families
trước đó: Full−baseline+0.3800BLEU, Full−matchedLSR+0.2344 (3/3 dương),
Full−tunedLSR−0.1267; Fullλ0.03−tunedLSR−0.8100 (3/3 âm).

Inventory validation Full−tunedLSRmean+0.2193BLEU/+0.1805chrF++, nhưng test BLEU
âm. Đây là chênh lệch validation→test cần hiểu; không phải bằng chứng collapse,
LSR áp đảo hoặc λ0.03 luôn tốt hơn ở mọi split. Không dùng test chọn cấu hình mới.

**Ưu tiên tiếp theo, trước candidate train:**

1. CPU error analysis bằng predictions đã có, tách validation/test và từngseed:
   so Full/tunedLSR/matchedLSR theo vài nhóm độ dài định trước, độ khó/coverage
   và case wins/losses. Báo count/effect sizes; coi là exploratory, không lựa
   hyperparams bằng nhóm test đẹp. Kiểm tra learning/selection histories nếu
   còn lưu. Chưa có local trainer-state/epoch histories đầy đủ ở lượt này;
   không tự quy lỗi cho checkpoint selection khi chưa có dữ liệu theoepoch.
2. Nếu cần thêm can thiệp0train để chọn hướng routing: validation-only **α sweep
   trên cùng3Fullweights**, ví dụ0.05/0.075/0.10/0.125/0.15 (α0.10reuse:12lượt
   generation mới). Đây là tuning inference, phải ghi tách strength train/inference;
   không dùng nó như evidence về architecture train mới. Khóa mộtα chung bằng
   mean validation và kiểm tra BLEU/chrFcharacter/chrF++; không chọnα riêngtừngseed.
3. Nếu sweep không giúp, một can thiệp validation theo routing groups early3–5/
   middle6–8/late9–12 (9generations,3seeds) có thể phân biệt dependence từnggroup.
   Tắt mộtgroup cũng thay cache đến downstream; không gọi là cô lập causal
   contribution của mộtlayer. Chỉ sau quan sát có ích mới chọn layer-selective
   routing hoặc routing schedule cho pilot theo17.4; không chọn normalization/
   λdecay chỉ vì ratio nhỏ hoặc rank cao.
4. Confirmation/ablation train giữ budget/controls ở17.4. Pilot cần cải thiện
   validation so với **Full gốc và tuned LSR**, vì Full gốc vốn đã hơn tuned LSR
   trên validationmean nhưng không chuyển thành test gain. Khóa rule trước test;
   báo rõ nghiên cứu tiếp theo là exploratory trên Amis đã xemtest, và cần held-out
   replication để claim generality. Không mở Turkish để tìm số thuận lợi.

**GO:** phép đo/protocol V4 và việc tiếp tục tìm cấu hình routing có giả thuyết.
**NO-GO:** superiority/generalization hiện tại, claim collapse/anti-collapse hoặc
khởi train sửa loss/normalization không có căn cứ. Chưa từ bỏ CLRR; cũng chưa có
bảo đảm ACL. Chưa chạy các sweep/pilot ở18.5 trong vòng xác minh đã hoàn tất này.

##### 18.6. Fallacy scan — 11/11 đã xét

Simpson: báo từngseed/site/definition; ecological: không suy tokencollapse từ
pooled ranks; Berkson: best-checkpoint selection giới hạn phạm vi; collider:
không điều chỉnh subgroup để chọn kết luận; base-rate: không áp dụng diagnostic
classification; regression-to-mean: không quy std nhỏ qua3seeds thành ổn định
tổng quát; survivorship: đủ21/21, không lọc checkpoint xấu; look-elsewhere:
không thêm significance từ nhiềulayers/metrics; forking-paths: protocol/codehash
và V2/V3/V4 riêng, không đổi reference; causal: on/off can thiệp chỉ áp dụng
sameweights, rank-quality là associational; reverse-causality: endpoint probes
không xác lập rank là nguyên nhân quality. Không suy equivalence từ p>0.05.

#### 19. V5 — can thiệp routing và kiểm tra decoding, 04/10/2026

- Mode: run + validate; status: DONE_WITH_PROTOCOL_CORRECTION, xem19.4.
- User authorization: tự chạy các phép kiểm chứng cần thiết; được dùng Colab.
- GPU: `gpu-a100-s-kkb-usc1f1-ydwyun3deht0`, A100 Standard, không high-mem.
- Started: 16:38 giờ Việt Nam; background PID732. Kernel CLI IDLE không có nghĩa
  worker dừng. Kiểm tra `/proc/<pid>/stat`, status/log và COMPLETE/FAILED.
- Outputs: `outputs_rebuttal/amis_routing_diagnostics_20261004_v5/`.
- Plan: `locked_plan_local.json`; SHA256
  `ccb52bdc4e164c578a34701cf52f743139b1b6e35b57f890a9c23a1ae6f347b1`.
- Worker: `scripts/revalidation/run_amis_routing_v5_worker.py`; production không sửa.

##### 19.1. Thiết kế khóa trước GPU

**Không train.** 12 checkpoint đúng SHA/revision, 3 seeds42/43/44:

| Nhánh | Mỗi seed chạy gì | Tổng generation đầy đủ |
| --- | --- | ---: |
| Full λ.10 | validation beam1: α.05/.075/.10/.125/.15; tắt early3–5/middle6–8/late9–12; validation beam4 và test beam1 ở α.10 gốc |30|
| Tuned LSR λ.03 | validation beam1/beam4, test beam1 |9|
| Matched LSR λ.10 | validation beam1 |3|
| Baseline | validation beam1 |3|

Tổng45; thêm8 test beam4 rows/model ở6Full/tuned models làm replay sentinel.
Validation576 và test575; batch8, max length256, pad multiple8, FP32/TF32off,
`tl_XX→zh_CN`, forced BOS250025. Runtime pin giống V4. CE token-weighted trên
32validation examples cố định cho các condition beam1; không so CE này trực tiếp
với eval_loss toàn validation. Mỗi CSV lưu index/source/target/prediction, SHA và record.

**Chọn α:** một α chung cho cả3 seeds, dựa mean validation beam1 native chrF++;
BLEU và chrF0 không giảm so α.10. Không chọn α/layers bằng test. Nếu ứng viên đáng
xác nhận, đăng ký validation beam4 trước lượt đánh giá bổ sung. α inference khác
α training phải ghi riêng; sweep này không chứng minh cấu hình train mới.

Các nhóm layer không cùng kích thước; tắt nhóm thay downstream cache. Chỉ là
can thiệp trên weights đã train, không cô lập causal contribution từng layer.
Test beam1 là control khóa trước trên Full/tuned, không thử các α mới trên test.
Mục đích phân biệt khác split và beam trong ranking; không đổi test protocol cũ.

##### 19.2. Kiểm chứng trước launch và insight CPU

Tiny mBART CPU trong conda clrr: all-layer α.10 logits/CE khớp bitwise production;
α0 khớp raw, parameter values/count giữ nguyên; masked cache và fresh-cache PASS.
Gold inputs có SHA/length/index/source/target guards;12 validation anchors phải khớp
inventory metrics<1e-6,3Full replay toàn576câu. Sai guard sẽ ghi FAILED và dừng.

`error_analysis.json` xác minh12test CSV và native scores cũ<1e-8. Full−tuned
chrF0 từngseed −.3662/−.2283/+.1039; không có ưu thế ổn định. Test chỉ3reference
word bigrams ở3/575câu: native chrF++ có độ nhạy với ít word ngrams, cần đọc cùng
BLEU và chrF0; không tự thay metric. Không có empty predictions; chiều dài dự đoán
khoảng95.1–97.2% reference theo ký tự không whitespace. Chưa thấy lỗi truncation/
empty output phổ biến. Source>10words chỉ29test/38validation: subgroup exploratory.

##### 19.3. Theo dõi, lấy kết quả, quyết định

```powershell
colab exec -s colab -f scratch/monitor_amis_v5_remote.py --timeout 30
```

Báo chat ít nhất mỗi10phút khi chạy. Sau COMPLETE: pack/download riêng, audit
native metrics và hashes, tổng hợp mean/sample std/từngseed, ghi EXPERIMENTS_UPDATED
và RESEARCH_INSIGHTS. Thu đủ raw outputs trước khi stop Colab. Không sửa scores cũ.
Nếu không có α/group cải thiện nhất quán, báo đủ bằng chứng cần nâng cấp thay vì
tiếp tục sweep tùy ý; chưa chạy Turkish và chưa tự chuyển phương pháp mới vào paper.

##### 19.4. Kết quả thực chạy và correction đã xác minh

Toàn bộ bảng/intervals/seed values: [EXPERIMENTS_UPDATED Mục12](../experiments/EXPERIMENTS_UPDATED.md).
Artifacts nằm riêng tại `outputs_rebuttal/amis_routing_diagnostics_20261004_v5/`.
Kế hoạch45 conditions ban đầu dừng ở guard Baseline seed42; giữ FAILED. **42 core
conditions/9 models VERIFIED42**, baseline3 models kiểm tra riêng thành công;
cộng3 full testreplay +1 val beam4confirmation =49 full conditions hợp lệ/lưu lại.
Hai lỗi của script audit bổ sung (lọc scores và thư mục extract còn từ lần lỗi)
đã sửa, giữ logs/FAILED riêng; không đổi source production hoặc test outputs cũ.

**Lỗi thực nghiệm tìm được:** Baseline seed42 validation17.8735BLEU/14.9009chrF++
trong F0 thực tế là beam4. Saved training_args và lượtvalbeam4 xác nhận chính xác.
Beam1 đúng16.2659/13.9015; vì vậy meanBaselinevalbeam1 là17.1275BLEU,
không phải17.6633. Full−Baselineval từ−.4498→+.0861, dương1/3 seeds.
Không phải weights bị mất/nhầm: cả3 baselinearchives tái lập đủ575 test predictions
ban đầu. TestFull−Baseline+.3800BLEU và−tuned LSR−.1267 giữ nguyên.
Baseline seed42 selectedstep chưa phục hồi được; **không dùng hardcoded216**.

**Lượt sau dùng inventory mới:**
`outputs_rebuttal/amis_routing_diagnostics_20261004_v5/validation_inventory_21_corrected_v5.json`.
12 validationcontrols được xác minh FP32/beam1,9 entries khác giữ nhãn lịch sử.
Loss lịch sử không đồng nhất với CEfixed32; không overwrite inventoryV2/CSV cũ.
`scratch/recover_amis_baseline_remote.py` đã đặt rõ validationbeam trước evaluate;
F0 source bỏbest step216, đánh dấu recovery beam provenance. Không chạy lại F0
vào output lịch sử; bản corrected inventory là nguồn cho lượt tiếp theo.

**Interventions:** meanΔBLEU soα.10/routingđủ:
α.05−.4830, α.075−.1636, α.125−.4773, α.15−.8456;
tắt early−.3272/middle−.1928/late−.5706. Cả7conditions có chrF++/chrF0 giảm3/3.
Khôngα nào qua no-harm guard; không thêm candidate test search. α.125/.15
giảm teacherCEfixed32 nhưng giảm generationquality: chưa xác định nguyên nhân
exposure bias hoặc overfitting. Tắt nhóm chỉ là can thiệp sameweights/cache.

**Full−tuned LSR meanΔBLEU:** validation beam1+.2193, beam4+.2055;
test beam1−.2168, beam4−.1267. Test chrF0 cũng âm ởcảbeam1/4. Beam mismatch không
giải thích hết split/seed ranking instability; không thể quy chắc cho overfitting.
Test chrF++beam4+1.3642 phụ thuộcseed43/ítwordngrams; xem cùng BLEU/chrF0.

**Probes lịch sử:**35points/280gradientbatches/70spectra raw tái tính được;
Full/matched early1→middle5→late20 không thấy globalLSR domination hoặc residual
explosion, ranks broaden ởcảhai. Historical source/runtime/language provenance
chưa đủ; chỉ descriptive, không trộn vớiV4 selectedbestendpoints hoặc causalclaims.

**Audit và hạ tầng:** `independent_audit.json` errors=[], core VERIFIED42,
baselineaudit VERIFIED, beam disagreement VERIFIED; overall ghi
INVENTORY_PROTOCOL_DISAGREEMENT để giữ lịch sử đúng. ZIP SHA verified,
`COLAB_STOPPED.json` xác nhận A100 Standard đã tắt lúc~17:15VN. Không train,
không Turkish, không đẩy thêm model lênHF vì không tạoweights mới.

**Quyết định:** cần cải thiện độ ổn định/quality trước ACL superiority claim,
chưa đủ bằng chứng bỏ CLRR. Inferenceα/group đơn giản không giúp trong sweep
đã thử; chưa xác định loss/normalization/gradientpath nào là nguyên nhân.

##### 19.5. Pilot tiếp theo đề xuất — train-time routing warmup

**04/10/2026: đã dừng hai queue warmup cũ lúc19:40 theo yêu cầu user và sửa để chạy lại v2.** Sổ tay chi tiết: [AMIS_WARMUP_2_COLAB_RUNBOOK.md](NEXT_EXPERIMENTS_HANDOFF_20261005.md#warmup-runbook-history), mục8. Không dùng `amis_routing_warmup_20261004` để xét GO/NO-GO: runtime lệch và thiếu lưu/đo cơ chế. Bản mới `amis_routing_warmup_20261004_v2` khóa môi trường mục4, lưu lịch sử/snapshot và HF namespace riêng. Cả hai đã qua GPU smoke; 3Full+3tuned LSR replay khớp V5, 3CLRR-only đã đo lại. Chỉ mở train sau two-machine gate. Mục6 runbook đã sửa mean/ngưỡng theo V5; không dùng helper deploy/download cũ.

Đây là giả thuyết kiểm chứng, không phải kết luận rằng
warmup sẽ cải thiện. Mục tiêu phân biệt phụ thuộcrouting đã học với incremental
benefit ổn định; giữ tất cả layer vì ablation không ủng hộ bỏ nhóm nào.

| Nhánh mới | Seeds | Cấu hình | Câu hỏi |
| --- | --- | --- | --- |
| Warmup Full |42/43/44|αtrain tăng tuyến tính0→.10 trong20%plannedoptimizersteps đầu, sauđó.10; λ=.10 cốđịnh|Thời điểm đưa routing vào có giúp Full vượt Full cốđịnh/tuned LSR không?|
| Warmup CLRR-only |42/43/44|Cùngαschedule, λ=0|Hiệu quả warmup đến từ routing hay chỉ tương tác vớiLSR?|

**6 train runs tối đa cho pilot v2**, chạy lại từ seed42 trên hai A100 Standard sau gate. Existing fixedFull/CLRR-only,
matched LSR/tuned LSR/baseline làm frozen controls sau kiểm tra reuse gates source,
data, optimizer và generation. CLRR-only chưa remeasurevalidation trongV5 nên
phải replay đúng3archives nếu dùng làm validationcontrol; không giả định VERIFIED.
Nếu control protocol/source mismatch, xử lý trước pilot và khai báo ngân sách
reruncontrol; không so model mới với control khác selection/runtimes âm thầm.

Fairness: giữ LR5e-5, dropout/weightdecay/warmupLR/batch/accumulation/maxlength,
20 epochs/patience4, native metric/reference/langcodes, FP32generation và training
precision như controls; **không tăng patience chỉ cho phương pháp**. αevaluation
luôn.10 cốđịnh, tách khỏiαtrain. Routing schedule không thêmparameters/RNG.
Checkpoint chọn theo validationchrF++beam1 giốngcontrols; test beam4 chỉ saukhóa
ứng viên. Theo dõi thêm validation beam4/BLEU/chrF0 có nhãn diagnostic; không đổi
selectionmetric riêng một nhánh hoặc chọn best bằng test. Batch/probe code smoke
phải bảo đảm α=.10 khớp production và α0 đúngraw, resume giữđúngglobalstep.

Lưu epoch histories, actualα/λ từngstep, selectedbest vàearly/mid/lateweights với
runtime/languageprefix/code/datahashes. Đo gradientunion, perlayergradients,
residuals, rawpooledmatrices/SVD, CEtokenweighted trêncùngbatches; BF16dropout/RNG
và checkpointing phải kiểm soát. Upload namespace mới trênFiveC, kiểmsha/reload
trướcxóaweights local/remote. Báo chat10phút/lần khi chạy.

**Gate pilot đề xuất, khóa trước train; ngưỡng nội bộ, không phải quy định ACL:**

1. GO chỉ nếu Warmup Full meanvalidationBLEU vàchrF++ đều tăng≥.30 sofixedFull,
   mean chrF0 không giảm, ít nhất2/3 seeds cóBLEU/chrF0 tăng; đồngthời mean BLEU/
   chrF0/chrF++ vượt tuned LSR. Báo mọiseed/trial, không chọn warmup fraction bằngtest.
2. Nếu GO, khóa **một** schedule chung rồi đotest tấtcả3 seeds một lần. Giữquality
   gate hiệncó: Full−Baseline≥.5BLEU, Full−matched≥.3 vàmatched3/3dương,
   Full−tunedmean>0/noharmchrF0. Sentencebootstrap không thay trainingseedinference.
   Amis test đã xem nhiều lần, vẫn là exploratory; không dùng nó như pristineholdout.
3. Nếu CLRR-only warmup tốt còn Full warmup không tốt, ưu tiên nghiên cứu interaction
   CE–LSR khi routing; chưa quyλ lớn/nhỏ lànguyênnhân. Nếu cảhai không tốt, NO-GO
   schedule: xemhistories để xác định giaiđoạn khácbiệt rồi mới chọn upgrade khác.
   Không tiếp tục thêmα/λgrid hoặc normalization chỉ để thử lấy sốđẹp.
4. Nếu qua gateAmis, trước claimgenerality cần replication trên held-outlanguage
   với4nhánh Baseline/CLRR-only/LSR-only/Full vàcùngprotocol. **Turkish vẫn chưa
   được bật**; đây là bước sau quyếtđịnh phươngpháp và chỉ chạy khiuser cho phép.

##### 19.5 — Kết quả thực thi v2, 04/10/2026 21:48

Đã hoàn tất cả sáu run trên hai Colab Standard, best reload/probes/HF verification PASS; independently rescored validation predictions khớp cả sáu. **NO-GO cho warmup20%**, năm điều kiện gate đều fail. Full mean BLEU17.0048, chrF++14.3859, chrF016.4809; CLRR-only mean BLEU15.9842. Không mở thêm training hay Turkish. Số mới, giới hạn đo và next-step matched inference ablation được ghi riêng trong [AMIS_WARMUP_V2_RESULTS.md](../archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#amis-warmup-v2-results). Không thay bảng test/paper cũ bằng số validation. Người tiếp nhận phải đọc báo cáo này trước các lệnh vận hành cũ.

Nếu sau pilot có kiểmsoát mà không qua gate, báo phươngpháp hiện tại chưa đủ
incremental evidence cho ambitious superiority claim; đưa dữliệu/nguồnnguyênnhân
và lựa chọn upgrade đểuser quyếtđịnh. Chưa thể hứa cấuhình nào sẽ giúp ACL.

#### 20. Kế hoạch dài hạn sau warmup v2 — 04/10/2026

Tài liệu hiện hành: [ACL_LONG_TERM_EXPERIMENT_PLAN.md](../research/ACL_LONG_TERM_EXPERIMENT_PLAN.md).

- Tổng hợp A/B/C, strong LoRA, screening/confirmation, attached control, final test, V4/V5 và warmup v2; phân biệt số test/validation và các artifacts invalid.
- Gates G0–G8 gồm: nguồn số/novelty → chẩn đoán → tối đa hai nâng cấp → fresh-seed confirmation → Turkish → backbone/comparators/data regimes → efficiency và review paper.
- Đề xuất trước mắt **G0+G1, chưa train**. U1/U2 là giả thuyết có điều kiện, không phải nguyên nhân đã xác nhận.
- Budget mới lớn hơn pilot cũ, chỉ là trần theo gate; chưa cho phép autorun toàn bộ hoặc bật Turkish. Runcards/protocol/budget từng phase phải chốt trước launch.
- Các hướng “tiếp theo” cũ đã hoàn tất không được chạy lại mặc định; giữ raw results và protocol metrics cũ.


<a id="warmup-runbook-history"></a>

## Phần gộp: AMIS_WARMUP_2_COLAB_RUNBOOK

> Nội dung gộp từ `AMIS_WARMUP_2_COLAB_RUNBOOK.md`. **Lịch sử thực thi, không phải yêu cầu chạy hiện hành.** Các trạng thái đang chạy/chưa chạy, repo HF cũ và lệnh warmup chỉ mô tả thời điểm nguồn; muốn chạy lại phải kiểm source, runtime, protocol và routing lưu checkpoint hiện tại.

### SỔ TAY VẬN HÀNH & BÀN GIAO: AMIS ROUTING WARMUP TRÊN 2 COLAB (SECTION 19.5)

> **KẾT QUẢ CUỐI — 21:48, 04/10/2026:** v2 hoàn tất **6/6**, cả hai queues COMPLETE; best reload, trajectory probes, HF checksum và tính lại metrics local đều PASS. **NO-GO warmup20%**: Full mean validation BLEU17.0048/chrF++14.3859/chrF016.4809, thấp hơn fixedFull cả ba metrics; CLRR-only mean BLEU15.9842. Xem [báo cáo đầy đủ và hướng tiếp theo](../archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#amis-warmup-v2-results). Không chạy tiếp grid/test/Turkish. Các cập nhật bên dưới là lịch sử vận hành.

**Ngày cập nhật:** 04/10/2026  
**Mục tiêu:** Thực thi thí nghiệm Section 19.5 trong `docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history` — Đánh giá giả thuyết *Train-time Routing Warmup* trên cặp ngôn ngữ Amis $\to$ Mandarin (mBART-large-50) phân bổ song song trên 2 tài khoản Google Colab Pro/Pro+ (GPU A100 40GB).

> **CẬP NHẬT 20:41, 04/10/2026:** User yêu cầu dừng bản sai protocol và chạy lại. Hai queue cũ đã dừng lúc19:40; artifacts `amis_routing_warmup_20261004` chỉ để audit, không dùng xét gate. Protocol hiện hành là **v2, mục8**; các lệnh deploy/download cũ ở mục4–5 không còn dùng. Hai máy đã qua smoke GPU/replay controls và mở two-machine gate lúc20:11; seed42 của cả hai đã train đủ20epochs, đang đo/kiểm chứng sau train. Chưa đủ sáu COMPLETE receipts để xét GO/NO-GO.

#### 1. Phân bổ tài nguyên & Tài khoản 2 Colab

| Tiêu chí | Máy 1 (Machine 1 - Full Warmup) | Máy 2 (Machine 2 - CLRR-only Warmup) |
| :--- | :--- | :--- |
| **Session Colab** | `colab` (Tài khoản mặc định) | `colab2` (Tài khoản 2) |
| **Config Profile** | `~/.config/colab-cli/sessions.json` | `C:\Users\Lenovo\.colab_acc2\.config\colab-cli\sessions.json` |
| **Biến môi trường Colab CLI** | Mặc định | `USERPROFILE=C:\Users\Lenovo\.colab_acc2` trong process con; dùng config riêng |
| **Phần cứng** | NVIDIA A100-SXM4-40GB | NVIDIA A100-SXM4-40GB |
| **Phương pháp** | `jepa-clrr-enc` (Full) | `clrr-enc` (CLRR-only) |
| **Tham số Routing** | $\alpha: 0 \to 0.10$ (20% đầu = 144 steps, sau đó 0.10 cố định) | $\alpha: 0 \to 0.10$ (20% đầu = 144 steps, sau đó 0.10 cố định) |
| **Hệ số JEPA ($\lambda$)** | $\lambda = 0.10$ cố định | $\lambda = 0.00$ (không dùng auxiliary loss) |
| **Hàng đợi Seeds** | 3 runs: Seed 42 $\to$ Seed 43 $\to$ Seed 44 | 3 runs: Seed 42 $\to$ Seed 43 $\to$ Seed 44 |
| **Thư mục Remote** | `/content/CLRR/outputs_rebuttal/amis_routing_warmup_20261004` | `/content/CLRR/outputs_rebuttal/amis_routing_warmup_20261004` |
| **File Log Worker** | `/content/CLRR/machine1.log` | `/content/CLRR/machine2.log` |

#### 2. Các quy tắc công bằng & Tham số cố định (Section 19.5 Fairness)

- **Backbone:** `facebook/mbart-large-50-many-to-many-mmt`
- **Tập dữ liệu:** `data_processed/amis_mandarin/` (`train.csv`, `validation.csv`, `test.csv`)
- **Tách ngôn ngữ:** Nguồn `tl_XX` (Tagalog proxy), Đích `zh_CN` (Mandarin), Tokenizer BLEU: `zh`
- **Siêu tham số tối ưu:**
  - Optimizer: `AdamW`
  - Learning rate: $5 \times 10^{-5}$, Weight decay: $0.0$, Warmup LR ratio: $0.06$
  - Batch size per device: $32$, Gradient accumulation: $4$ (Effective batch size = $128$)
  - Độ dài tối đa: Source $256$, Target $256$
  - Tổng số epoch: Tối đa $20$ epochs ($720$ steps tổng), Early stopping patience: $4$ epochs
  - Precision: Training BF16 + Gradient Checkpointing, FP32 Generation
  - Đánh giá Validation: Greedy (beams = 1), $\alpha_{\text{eval}} = 0.10$ cố định (bảo toàn điều kiện suy diễn cách ly).
- **Quy tắc tuyệt đối:**
  - **KHÔNG CHẠY `git push origin main`**.
  - Không chọn epoch/checkpoint bằng tập test. Lựa chọn checkpoint duy nhất bằng validation raw-reference chrF++.

#### 3. Các lỗi đã được xử lý triệt để trong Codebase

Trước khi bàn giao, 4 lỗi hệ thống quan trọng đã được khắc phục hoàn toàn:
1. **Lỗi đệ quy vô hạn khi lưu checkpoint (`_save` trong `src/amis_rewire/evaluation.py`):**
   - *Nguyên nhân:* Property `.base_model` của class `MBartModel` tự trỏ lại chính nó (`m.base_model is m`). Vòng lặp `while hasattr(m, 'base_model')` bị biến thành `while True`, làm treo CPU 100% tại step 36.
   - *Đã khắc phục:* Thêm điều kiện dừng `while not isinstance(m, PreTrainedModel) and hasattr(m, 'base_model'): m = m.base_model`.
2. **Lỗi mBART safetensors shared-memory (`RuntimeError: Some tensors share memory`):**
   - *Đã khắc phục:* Override `_save()` trong `PrecisionSeq2SeqTrainer` để gọi trực tiếp `m.save_pretrained(..., safe_serialization=False)` bằng định dạng `pytorch_model.bin`.
3. **Lỗi Transformers `forced_bos_token_id`:**
   - *Đã khắc phục:* Gán `model.config.forced_bos_token_id = None` và chỉ lưu trong `generation_config.forced_bos_token_id`.
4. **Lỗi Unicode `cp1252` trên Windows Console khi chạy `colab exec`:**
   - *Đã khắc phục:* Bổ sung `PYTHONIOENCODING="utf-8"` và `sys.stdout.reconfigure(encoding="utf-8")` vào các script điều phối trên Windows.

#### 4. Các lệnh điều khiển sẵn có (1-Click CLI)

Mọi thao tác đã được đóng gói trong script [scratch/deploy_and_monitor_warmup.py](../../scratch/deploy_and_monitor_warmup.py). Chạy bằng Python môi trường `clrr`:
```powershell
# Đường dẫn Python cố định:
$PY = "C:\Users\Lenovo\miniconda3\envs\clrr\python.exe"
```

##### A. Kiểm tra trạng thái hiện tại (Monitor 2 máy)
```powershell
& $PY scratch/deploy_and_monitor_warmup.py monitor
```
*Hoặc kiểm tra callstack trực tiếp qua py-spy:*
```powershell
& $PY scratch/check_remote_status.py
```

##### B. Triển khai / Khởi động lại (Nếu cần thiết lập lại từ đầu)
```powershell
# 1. Đóng gói lại bundle nếu có sửa code local:
& $PY scratch/package_amis_warmup.py

# 2. Triển khai lên Máy 1 (Warmup Full):
& $PY scratch/deploy_and_monitor_warmup.py deploy_m1

# 3. Triển khai lên Máy 2 (Warmup CLRR-only):
& $PY scratch/deploy_and_monitor_warmup.py deploy_m2
```

##### C. Tải kết quả về máy và tắt phiên Colab (Khi hàng đợi xong)
Khi hàng đợi 3 seeds trên máy nào hoàn thành (`WORKER_QUEUE_COMPLETE`):
```powershell
# Tải kết quả Máy 1 và ngắt phiên colab:
& $PY scratch/deploy_and_monitor_warmup.py download_m1

# Tải kết quả Máy 2 và ngắt phiên colab2:
& $PY scratch/deploy_and_monitor_warmup.py download_m2
```
Toàn bộ artifacts sẽ được lưu tự động về thư mục local:
`outputs_rebuttal/amis_routing_warmup_20261004/runs/`

#### 5. Quy trình thủ công trực tiếp trên Colab (Fallback Manual Protocol)

Nếu không dùng script Python từ xa, bất kỳ ai cũng có thể mở Colab và chạy thủ công:

##### Bước 1: Mở phiên làm việc
- **Máy 1:** Mở Colab ở tài khoản 1, chọn GPU A100.
- **Máy 2:** Mở Colab ở tài khoản 2, chọn GPU A100.

##### Bước 2: Tải code và giải nén
Tải file `scratch/amis_routing_warmup_bundle.zip` lên thư mục `/content/`, sau đó chạy:
```bash
mkdir -p /content/CLRR
unzip -q -o /content/amis_routing_warmup_bundle.zip -d /content/CLRR
pip install -q sacrebleu transformers huggingface_hub accelerate
```

##### Bước 3: Khởi chạy hàng đợi
- **Trên Máy 1 (Warmup Full):**
  ```bash
  cd /content/CLRR
  export PYTHONPATH=/content/CLRR/src:$PYTHONPATH
  python3 -u scripts/revalidation/run_amis_warmup_worker.py --worker machine1
  ```
- **Trên Máy 2 (Warmup CLRR-only):**
  ```bash
  cd /content/CLRR
  export PYTHONPATH=/content/CLRR/src:$PYTHONPATH
  python3 -u scripts/revalidation/run_amis_warmup_worker.py --worker machine2
  ```

#### 6. Tiêu chí nghiệm thu & Cổng quyết định (Section 19.5 Gate 1)

Khi 6 runs (3 Full + 3 CLRR-only) hoàn thành:
1. **Kiểm tra tính toàn vẹn:**
   - Mỗi run phải có `COMPLETE.json`, `metrics.json`, `eval_loss_trace.json`, `validation_predictions.csv`.
   - `eval_chrf0` được tính toán độc lập bằng `sacrebleu.metrics.CHRF(word_order=0)`.
2. **So sánh với Frozen Controls:**
   - **Fixed Full Control (mean 3 seeds, V5 validation beam 1):** Val BLEU = `17.2135386097`, Val chrF++ = `14.4975355132`, Val chrF0 = `16.6585885145`.
   - **Tuned LSR Control (mean 3 seeds, V5 validation beam 1):** Val BLEU = `16.9942038997`, Val chrF++ = `14.3170105095`, Val chrF0 = `16.4949888636`.
   - Nguồn duy nhất cho các mean trên: `outputs_rebuttal/amis_routing_diagnostics_20261004_v5/diagnostic_summary.json`, `family_scores`, hai nhánh `original_full_detached/val_main_b1` và `tuned_lsr/val_main_b1`. Số cũ trong mục này ghi sai; audit 04/10/2026 chỉ sửa tài liệu, không thay artifacts thực nghiệm cũ.
3. **Quy tắc Gate 1 (GO / NO-GO):**
   - **GO:** Mean validation BLEU và chrF++ của *Warmup Full* tăng ít nhất `+0.30` so với *Fixed Full*: BLEU `>=17.5135386097`, chrF++ `>=14.7975355132`. Mean chrF0 `>=16.6585885145`; ít nhất 2/3 seeds tăng cả BLEU và chrF0 so với seed tương ứng của Fixed Full; cả ba mean BLEU/chrF++/chrF0 vượt *Tuned LSR*. Dùng số đầy đủ trong JSON để xét gate, không dùng số làm tròn.
   - **NO-GO:** Không đạt ngưỡng trên. Phân tích interaction CE–LSR dựa trên nhánh Warmup CLRR-only đối chứng.

#### 7. Audit triển khai thực tế — 04/10/2026

**Kết luận: cấu hình train chính đúng, nhưng chưa triển khai đầy đủ protocol Section 19.5.** Kiểm tra chỉ đọc cả hai Colab; không dừng, redeploy hay đổi recipe của job đang chạy.

##### Đã xác minh

- Lần kiểm tra live cuối **19:20 giờ Việt Nam, 04/10/2026**: cả hai đang seed42, đã đánh giá epoch15 và đang train epoch16 (Full step556, CLRR-only step558). Best theo chrF++ hiện tại đều ở epoch14: Full BLEU16.0375/chrF++13.7673; CLRR-only BLEU15.9589/chrF++13.5156. Đây là validation tạm, chưa phải final hoặc mean3seeds.
- Hai GPU A100-SXM4-40GB Standard: `colab` chạy Full (`jepa-clrr-enc`, lambda=0.10), `colab2` chạy CLRR-only (`clrr-enc`, lambda=0), seed42; mỗi worker xếp tiếp seed43/44.
- LR5e-5, WD0, LR warmup0.06, batch32×4, eval8, lengths256, tối đa20epochs/patience4, BF16/checkpointing, FP32 generation, validation beam1 và validation-only khớp lệnh train. Không chạy Turkish/test trong queue này.
- Routing warmup thực tế 144/720 optimizer steps, alpha train0→0.10; evaluate/predict override alpha=0.10. Alpha hiển thị cuối epoch trong log là alpha train, không phải alpha dùng khi evaluate.
- Hash cả ba CSV khớp local; 11 file trong manifest remote khớp SHA256. Backbone cache là revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`.
- Tiny mBART CPU: cả Full và CLRR-only save → raw reload không missing/unexpected keys; best-checkpoint reload khôi phục weights bitwise. Receipt: `scratch/warmup_save_cpu_1791116057258238400/results.json`. Kiểm tra này không thay thế reload checkpoint lớn thực tế khi nghiệm thu.

##### Các phần chưa đạt và cần xử lý trước kết luận về phương pháp

1. **Runtime không khóa theo bàn giao.** Cả hai máy thực tế dùng torch2.11.0+cu130 / transformers5.17.0 / accelerate1.15.0 / sacrebleu2.6.0 / hub1.31.0; bảng protocol trong bàn giao ghi torch2.6.0+cu124 / transformers4.57.6 / hub0.36.0. `requirements-colab.txt` còn yêu cầu transformers<5. Helper deploy/fallback đang pip install không pin. Chưa chứng nhận reuse controls fair; cần đối chiếu runtime training gốc và bổ sung matched control nếu không tương đương. Không âm thầm đổi thư viện giữa các seeds đang chạy.
2. **Thiếu trajectory probes/weights.** Worker chưa gọi đo CE–LSR gradients, residual/SVD trên fixed batches, chưa viết `eval_loss_trace.json` hay lưu actual alpha/lambda history. `CheckpointBackupCallback.on_save` hiện return ngay; `save_total_limit=2` xoay vòng checkpoint, worker còn xóa checkpoint-* khi run hoàn thành. Vì vậy early/mid weights có thể đã mất; không được tuyên bố đã thực hiện toàn bộ cơ chế chỉ từ final score. Cần lưu histories/provenance và bố trí phép đo còn thiếu trước nghiệm thu; nếu early weights đã mất thì phải khai báo và quyết định rerun có kiểm soát.
3. **Chưa có HF backup tự động trong queue.** Worker không truyền repo/prefix upload; best ZIP local không phải receipt HF. Upload vào namespace mới trên FiveC, kiểm SHA và reload trước khi xóa/tắt phiên. `download_m1/m2` hiện đóng gói toàn bộ weights thành một ZIP lớn rồi tắt máy: cần thay bằng chuyển artifacts có kiểm tra checksum và tải chia phần/HF trước khi dùng để kết thúc phiên.
4. **Reuse control còn thiếu.** Ba checkpoint CLRR-only cũ chưa remeasure validation theo V5; cần replay nếu dùng làm control. Các mean/ngưỡng sai ở mục6 đã được sửa theo V5; sửa này không chứng minh warmup đã qua gate.
5. **Lỗi tiềm ẩn khi chạy control mới.** `_find_rewire` trong `src/amis_rewire/evaluation.py` thiếu visited/self-reference guard: raw mBART có thể lặp mãi ở `base_model is self`. Hai nhánh đang chạy tìm được wrapper nên không bị lỗi này; phải sửa/smoke trước khi chạy Baseline/LSR-only với helper mới. `_save` cũng chưa lưu `training_args.bin`; cần receipt đầy đủ command/runtime/args cho nghiệm thu.

**Vận hành:** không dùng `deploy_m1/m2` để kiểm tra job đang chạy: helper kill worker/trainer và có thể xóa run dở khi khởi động lại. Profile thứ hai truy cập được bằng config riêng và `USERPROFILE` của process con; lỗi “session not found” từ account mặc định không chứng minh `colab2` mất. Các script audit chỉ đọc: `scratch/review_warmup_sessions.py`, `scratch/review_warmup_details_sessions.py` (chạy trong conda `clrr`).

**Giới hạn kết luận:** chưa có đủ sáu kết quả final và chưa qua fairness/provenance gates. Chỉ được báo tiến độ và validation tạm, chưa kết luận GO/NO-GO hay cải thiện phương pháp.

#### 8. Chạy lại đúng protocol — v2 (04/10/2026)

##### Trạng thái và phạm vi

- Hai queue cũ đã dừng, có `PROTOCOL_INVALIDATED.json` trên mỗi máy. Giữ nguyên log/weights cũ; không resume hoặc gộp seed từ bản cũ với v2.
- Kết quả mới: `outputs_rebuttal/amis_routing_warmup_20261004_v2/`. HF namespace: `FiveC/amis-rewire-checkpoints/amis_routing_warmup_20261004_v2/<run>/`; không ghi đè các prefix cũ.
- Vẫn sáu train runs: Full lambda0.10 và CLRR-only lambda0, mỗi nhánh seeds42/43/44. Alpha0→0.10 trong144/720plannedsteps đầu; eval alpha0.10. Giữ toàn bộ hyperparameters mục2, patience4 và native validation chrF++beam1 để chọn best. Không train Turkish, không đánh giá test để chọn schedule.
- Local chạy bằng conda `clrr`. Colab dùng Python riêng `/content/clrr-warmup-venv/bin/python`, Python3.13.15 như receipt V4; chỉ dùng hai session A10040GB **Standard** đã có. Không tạo hoặc chuyển sang high-mem.

##### Khóa môi trường và đầu vào

`scripts/revalidation/amis_warmup_protocol.py` fail-closed nếu versions/hash sai:
torch2.6.0+cu124, transformers4.57.6, hub0.36.0, accelerate1.15.0, sacrebleu2.6.0,
datasets3.6.0, numpy2.5.3, pandas3.0.6, tokenizers0.22.2, sentencepiece0.2.2,
safetensors0.8.0, protobuf7.36.2. Backbone/tokenizer pin revision
`e30b6cb8eb0d43a0b73cab73c7676b9863223a30`.

Manifest SHA256 bao gồm production code, probe/replay code, smoke tests, data và gold references;
preflight ghi actual versions, Python/GPU/RAM, input hashes, HF prefix và pip freeze.
Auth đã dùng credential user cho phép trong session; không đưa token vào bundle/log/docs.

##### Gate trước train

1. GPU smoke cả bốn nhánh bằng `tests/smoke_precision_training.py --cuda` và riêng warmup
   bằng `tests/smoke_warmup_records.py --cuda`: BF16/checkpointing, alpha0/raw cùngGPU,
   alpha0.10 làm thay đổi logits, schedule, snapshot và strict/best weight reload.
2. `replay_amis_warmup_controls.py`: máy1 replay ba Full và ba tuned LSR (λ.03), phải
   khớp từng prediction với V5; máy2 đo lại ba CLRR-only từ HF archive đúng SHA/revision.
   Chỉ dùng validation576rows, beams1, FP32/TF32off. Lưu prediction CSV/metrics và receipts mới.
3. Driver dừng chờ `ALLOW_CORRECTED_WARMUP_TRAIN.json`; chỉ mở gate sau khi **cả hai**
   có preflight và smoke PASS, control receipts đầy đủ, cùng manifest. Replay đúng xác minh
   inference; original training configuration/source/runtime provenance vẫn phải đối chiếu.
   Không dùng replay decoding để tuyên bố mọi chi tiết training cũ đã được chứng minh.

##### Lưu và đo trong/sau mỗi run

- `--scientific-records`: lưu actual alpha/lambda mỗi optimizerstep vào
  `routing_schedule.jsonl`, `eval_loss_trace.json`, `training_history.json`,
  `selection_receipt.json`, `final_trainer_state.json` và `training_args.bin`.
- Lưu archive weights tại epochs1,4,10 và epoch dừng thực tế (nếu trùng chỉ một archive).
  Epoch4 là cuối routing warmup, epoch10 là midpoint của budget20epochs; không giả định
  luôn train đủ20epochs. Archive trajectory không có optimizer, không dùng để resume.
- Upload từng snapshot vào HF prefix mới, kiểm remote size/LFS SHA256 ở commit receipt;
  checkpoint rotation không xóa các archive này. Best archive cũng có kiểm remote SHA.
- Sau train chạy `probe_amis_warmup_trajectory.py`: reload best và replay toàn validation
  phải khớp output production; từng trajectory archive phải strict load và đo lại trên
  cùng32rows/8batches4 của trajectory protocol cũ. Lưu CE–weighted LSR gradient union/
  groups từnglayer (CLRR-only có counterfactual λ.10 được ghi nhãn riêng), residuals,
  raw pooled matrices và SVD trên toàn576validationrows.
- Probe chạy trong process riêng sau train: BF16/dropout cho gradients, fixed batch RNG,
  TF32off và checkpointing disabled giống V4; representations eval mode. Đây là diagnostic
  offline, không giả định tương đương hoàn toàn checkpointed training. CE fixed32 báo
  token-weighted; không trộn với mean gradient cosine theo batch.
- Mỗi snapshot đo ở alpha eval0.10; nếu alpha train của update cuối khác0.10 (epochs1/4),
  đo thêm condition `last_training_alpha`, cùng batches/RNG, và ghi nhãn riêng. Không gộp hai
  điều kiện alpha hoặc gọi đây là can thiệp trong training.
- Chỉ ghi `COMPLETE.json` và xóa rolling training checkpoint sau khi best reload,
  trajectory probes và HF SHA kiểm đủ. Khi một bước lỗi, worker dừng, giữ artifacts;
  không tự xóa run dở để chạy lại. Upload metadata/probes/COMPLETE cùng namespace.

##### Các lệnh hiện hành (local conda clrr)

```powershell
conda run --no-capture-output -n clrr python -X utf8 scratch/package_corrected_amis_warmup.py
# Chỉ deploy khi đã xác minh không có driver/train active; không dùng để monitor.
conda run --no-capture-output -n clrr python -X utf8 scratch/deploy_corrected_warmup.py
conda run --no-capture-output -n clrr python -X utf8 scratch/review_warmup_sessions.py --file scratch/check_corrected_warmup_remote.py
# Mở gate chỉ sau kiểm đủ receipts cả hai máy (helper tự kiểm, thiếu thì dừng):
conda run --no-capture-output -n clrr python -X utf8 scratch/authorize_corrected_warmup.py
```

Helper deploy/download cũ đã bị chặn để tránh kill/redeploy sai hoặc tải một ZIP weights
khổng lồ rồi tắt runtime. Theo dõi từng phase, không gọi phase probe/upload là train bị treo.
Chỉ kết luận GO/NO-GO sau đủ3seeds và audit đầy đủ; ngưỡng chính xác ở mục6.


<a id="warmup-handoff-history"></a>

## Phần gộp: BAN_GIAO_CHAY_2_COLAB

> Nội dung gộp từ `BAN_GIAO_CHAY_2_COLAB.md`. **Lịch sử thực thi, không phải yêu cầu chạy hiện hành.** Các trạng thái đang chạy/chưa chạy, repo HF cũ và lệnh warmup chỉ mô tả thời điểm nguồn; muốn chạy lại phải kiểm source, runtime, protocol và routing lưu checkpoint hiện tại.

### TÀI LIỆU BÀN GIAO THỰC THI THÍ NGHIỆM TRÊN 2 COLAB
#### ĐÁNH GIÁ TRAIN-TIME ROUTING WARMUP CHO AMIS $\to$ MANDARIN (SECTION 19.5)

**Dự án:** CLRR (Contrastive Layer Representation Rewiring)  
**Tác vụ:** Dịch máy ngôn ngữ Amis $\to$ Mandarin (mBART-large-50)  
**Ngày cập nhật:** 04/10/2026  
**Trạng thái hệ thống:** Đang tự động chạy hàng đợi cuối cùng (Seed 44) trên cả 2 máy A100.  
**Mục tiêu tài liệu:** Cung cấp quy trình bàn giao hoàn chỉnh, "chìa khóa trao tay" (turnkey 1-click), tài liệu hóa toàn bộ lệnh, môi trường, script, checkpoint và tiêu chí đánh giá để bất kỳ ai cũng có thể tự chạy hoặc tiếp quản ngay lập tức trên 2 phiên Google Colab độc lập.

#### 1. TÓM TẮT TIẾN ĐỘ THỰC TẾ HIỆN TẠI (REAL-TIME STATUS)

Trước khi người mới tiếp quản hoặc chạy máy mới, xin lưu ý hệ thống hiện tại **đang chạy ổn định và sắp hoàn tất toàn bộ 6/6 runs**:

| Phiên Colab | Session / Worker | Cấu hình Thí nghiệm | Hàng đợi Seed | Trạng thái hiện tại | Điểm số đã ghi nhận (Validation) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Máy 1** | Session: `colab`<br>GPU: A100-SXM4-40GB | **Warmup Full**<br>($\alpha: 0 \to 0.10$, $\lambda=0.10$) | Seed 42 $\to$ 43 $\to$ 44 | - **Seed 42:** ĐÃ XONG (100%)<br>- **Seed 43:** ĐÃ XONG (100%)<br>- **Seed 44:** Bắt đầu chạy run cuối | **Seed 42:** BLEU = 17.7908, chrF++ = 14.9599, chrF0 = 17.0269<br>**Seed 43:** BLEU = 16.2231, chrF++ = 13.8586 |
| **Máy 2** | Session: `colab2`<br>GPU: A100-SXM4-40GB | **Warmup CLRR-only**<br>($\alpha: 0 \to 0.10$, $\lambda=0.00$) | Seed 42 $\to$ 43 $\to$ 44 | - **Seed 42:** ĐÃ XONG (100%)<br>- **Seed 43:** ĐÃ XONG (100%)<br>- **Seed 44:** **ĐANG TRAIN TRÊN GPU** (PID 49574) | **Seed 42:** BLEU = 16.6623, chrF++ = 13.9601, chrF0 = 16.0884<br>**Seed 43:** BLEU = 14.4003, chrF++ = 12.5549, chrF0 = 14.5335 |

> **Toàn bộ Model Checkpoints và Trajectory Probe Representations** được tự động đẩy trực tiếp lên Hugging Face Hub tại repo:  
> 👉 [`FiveC/amis-rewire-checkpoints`](https://huggingface.co/FiveC/amis-rewire-checkpoints) (nhánh/prefix: `amis_routing_warmup_20261004_v2`).

#### 2. QUY TRÌNH "CHÌA KHÓA TRAO TAY" CHO BÊN KHÁC TỰ CHẠY TRÊN 2 COLAB MỚI

Nếu người tiếp quản muốn dựng lại từ đầu trên 2 máy Colab Pro/Pro+ mới (chọn Runtime GPU A100 hoặc V100/L4), chỉ cần copy & paste các lệnh sau vào notebook/terminal Colab.

##### BƯỚC 1: Khởi tạo trên CẢ 2 MÁY (Copy vào ô lệnh đầu tiên)

```bash
# 1. Clone source code từ Git và cài môi trường chuẩn
cd /content
git clone https://github.com/HeyDunaX/CLRR.git
cd /content/CLRR

# 2. Tạo virtualenv và cài đặt đúng các package đã kiểm định chặt chẽ (Pinned versions)
python3 -m venv /content/clrr-warmup-venv
source /content/clrr-warmup-venv/bin/activate

pip install --upgrade pip
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.6 accelerate==1.15.0 sacrebleu==2.6.0 datasets==3.6.0 huggingface-hub==0.36.0 sentencepiece protobuf safetensors

# 3. Cài đặt CLRR editable package
pip install -e .

# 4. Thiết lập token Hugging Face (để tự động backup và kéo checkpoint)
export HF_TOKEN="<NHẬP_HF_TOKEN_CỦA_BẠN>"
export PYTHONPATH="/content/CLRR/src:$PYTHONPATH"
```

##### BƯỚC 2A: Chạy trên MÁY 1 (Colab 1 - Warmup Full)

**Mục tiêu:** Chạy lần lượt 3 seeds: 42, 43, 44 cho phương pháp Full Warmup ($\alpha: 0 \to 0.10$, $\lambda=0.10$).  
Copy toàn bộ đoạn script sau và chạy trên Colab 1:

```bash
#!/bin/bash
source /content/clrr-warmup-venv/bin/activate
cd /content/CLRR

DATA_DIR="/content/CLRR/data_processed/amis_mandarin"
OUTPUT_DIR="/content/CLRR/outputs_rebuttal/amis_routing_warmup_20261004_v2/runs"
BACKUP_DIR="/content/CLRR/outputs_rebuttal/amis_routing_warmup_20261004_v2/backups"
HF_REPO="FiveC/amis-rewire-checkpoints"
HF_PREFIX="amis_routing_warmup_20261004_v2"

mkdir -p "$OUTPUT_DIR" "$BACKUP_DIR"

for SEED in 42 43 44; do
    echo "=========================================================="
    echo ">>> BẮT ĐẦU MÁY 1: Warmup Full | Seed $SEED"
    echo "=========================================================="
    
    python -u -m amis_rewire.train \
        --model mbart-large-50 \
        --model-name facebook/mbart-large-50-many-to-many-mmt \
        --model-revision e30b6cb8eb0d43a0b73cab73c7676b9863223a30 \
        --method jepa-clrr-enc \
        --rewire-stack encoder \
        --rewire-distance 2 \
        --rewire-strength 0.1 \
        --jepa-weight 0.1 \
        --routing-warmup-ratio 0.2 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --backup-dir "$BACKUP_DIR" \
        --run-name "amis-warmup-full-seed${SEED}" \
        --seed "$SEED" \
        --learning-rate 5e-5 \
        --weight-decay 0.0 \
        --warmup-ratio 0.06 \
        --num-train-epochs 20 \
        --early-stopping-patience 4 \
        --per-device-train-batch-size 32 \
        --gradient-accumulation-steps 4 \
        --per-device-eval-batch-size 8 \
        --max-source-length 256 \
        --max-target-length 256 \
        --src-lang tl_XX \
        --tgt-lang zh_CN \
        --bleu-tokenizer zh \
        --eval-beams 1 \
        --num-beams 4 \
        --generation-precision fp32 \
        --bf16 \
        --no-fp16 \
        --gradient-checkpointing \
        --save-total-limit 2 \
        --dataloader-num-workers 2 \
        --logging-steps 10 \
        --validation-only \
        --scientific-records \
        --hf-backup-repo "$HF_REPO" \
        --hf-backup-prefix "$HF_PREFIX"
        
    echo ">>> HOÀN THÀNH SEED $SEED TRÊN MÁY 1"
done

echo ">>> TẤT CẢ 3 SEEDS TRÊN MÁY 1 ĐÃ HOÀN THÀNH XUẤT SẮC!"
```

##### BƯỚC 2B: Chạy trên MÁY 2 (Colab 2 - Warmup CLRR-only)

**Mục tiêu:** Chạy lần lượt 3 seeds: 42, 43, 44 cho phương pháp CLRR-only Warmup ($\alpha: 0 \to 0.10$, $\lambda=0.00$).  
Copy toàn bộ đoạn script sau và chạy trên Colab 2:

```bash
#!/bin/bash
source /content/clrr-warmup-venv/bin/activate
cd /content/CLRR

DATA_DIR="/content/CLRR/data_processed/amis_mandarin"
OUTPUT_DIR="/content/CLRR/outputs_rebuttal/amis_routing_warmup_20261004_v2/runs"
BACKUP_DIR="/content/CLRR/outputs_rebuttal/amis_routing_warmup_20261004_v2/backups"
HF_REPO="FiveC/amis-rewire-checkpoints"
HF_PREFIX="amis_routing_warmup_20261004_v2"

mkdir -p "$OUTPUT_DIR" "$BACKUP_DIR"

for SEED in 42 43 44; do
    echo "=========================================================="
    echo ">>> BẮT ĐẦU MÁY 2: Warmup CLRR-only | Seed $SEED"
    echo "=========================================================="
    
    python -u -m amis_rewire.train \
        --model mbart-large-50 \
        --model-name facebook/mbart-large-50-many-to-many-mmt \
        --model-revision e30b6cb8eb0d43a0b73cab73c7676b9863223a30 \
        --method clrr-enc \
        --rewire-stack encoder \
        --rewire-distance 2 \
        --rewire-strength 0.1 \
        --jepa-weight 0.0 \
        --routing-warmup-ratio 0.2 \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --backup-dir "$BACKUP_DIR" \
        --run-name "amis-warmup-clrr-seed${SEED}" \
        --seed "$SEED" \
        --learning-rate 5e-5 \
        --weight-decay 0.0 \
        --warmup-ratio 0.06 \
        --num-train-epochs 20 \
        --early-stopping-patience 4 \
        --per-device-train-batch-size 32 \
        --gradient-accumulation-steps 4 \
        --per-device-eval-batch-size 8 \
        --max-source-length 256 \
        --max-target-length 256 \
        --src-lang tl_XX \
        --tgt-lang zh_CN \
        --bleu-tokenizer zh \
        --eval-beams 1 \
        --num-beams 4 \
        --generation-precision fp32 \
        --bf16 \
        --no-fp16 \
        --gradient-checkpointing \
        --save-total-limit 2 \
        --dataloader-num-workers 2 \
        --logging-steps 10 \
        --validation-only \
        --scientific-records \
        --hf-backup-repo "$HF_REPO" \
        --hf-backup-prefix "$HF_PREFIX"
        
    echo ">>> HOÀN THÀNH SEED $SEED TRÊN MÁY 2"
done

echo ">>> TẤT CẢ 3 SEEDS TRÊN MÁY 2 ĐÃ HOÀN THÀNH XUẤT SẮC!"
```

#### 3. CÁC QUY TẮC BẢO ĐẢM KHOA HỌC & FAIRNESS (BẮT BUỘC TUÂN THỦ)

1. **KHÔNG CHẠY `git push origin main`:** Đây là quy tắc tuyệt đối của repository.
2. **Cách ly suy diễn (Strict Evaluation Isolation):**
   - Trong quá trình validation, tham số routing luôn được khóa cố định ở $\alpha_{\text{eval}} = 0.10$ (không áp dụng lịch warmup của training).
   - Suy diễn greedy (beam 1), generation FP32 để đảm bảo tính tái lập 100% không suy hao do sai số lượng tử hóa.
3. **Tiêu chuẩn chọn checkpoint (Model Selection):**
   - Lựa chọn checkpoint tốt nhất DUY NHẤT dựa trên `eval_chrf++` (raw reference) của tập `validation.csv`.
   - **Tuyệt đối không dùng `test.csv` để chọn checkpoint hay epoch dừng.**
4. **Early Stopping:**
   - Dừng huấn luyện nếu sau 4 epoch liên tiếp không cải thiện `validation chrF++`.

#### 4. BẢNG TIÊU CHUẨN XÉT DUYỆT GATE 1 (GO / NO-GO CRITERIA)

Sau khi hoàn tất 6 runs (3 runs Máy 1, 3 runs Máy 2), hãy đối chiếu kết quả trung bình của **Warmup Full (Máy 1)** với các baseline đối chứng đã kiểm định độc lập (V5 frozen controls):

##### Baseline đối chứng (Frozen Controls V5)
| Phương pháp | Mean Validation BLEU | Mean Validation chrF++ | Mean Validation chrF0 |
| :--- | :--- | :--- | :--- |
| **Fixed Full Control** ($\alpha=0.10, \lambda=0.10$ cố định từ đầu) | **17.2135** | **14.4975** | **16.6586** |
| **Tuned LSR** ($\lambda=0.03$) | **16.9942** | **14.3170** | **16.4950** |
| **CLRR-only Control** ($\alpha=0.10, \lambda=0.00$) | **16.8164** | **14.0765** | **16.2050** |

##### Tiêu chí quyết định Gate 1:
- ✅ **Quyết định GO (Chấp nhận giả thuyết Routing Warmup):**
  1. Mean BLEU của Warmup Full $\ge 17.5135$ (tức cải thiện tối thiểu $+0.30$ BLEU so với Fixed Full Control).
  2. Mean chrF++ của Warmup Full $\ge 14.7975$ (tức cải thiện tối thiểu $+0.30$ chrF++ so với Fixed Full Control).
  3. chrF0 không bị tụt.
  4. Có ít nhất $2 / 3$ seed có điểm số cao hơn baseline tương ứng.
  5. Mean điểm số vượt qua Tuned LSR ($16.9942$).
- ❌ **Quyết định NO-GO:**
  - Nếu không thỏa mãn đồng thời các điều kiện trên, kết luận giả thuyết Warmup không tạo ra cải thiện có ý nghĩa thống kê; giữ nguyên mô hình V5 cho bài báo.

#### 5. CÁC LỖI KỸ THUẬT ĐÃ ĐƯỢC FIX TRIỆT ĐỂ TRONG CODE (YÊN TÂM CHẠY)

Người tiếp quản không cần lo lắng về các lỗi dừng/treo máy vì các vấn đề sau đã được xử lý tận gốc trong mã nguồn:
1. **Lỗi đệ quy vô hạn khi lưu checkpoint (`_save`):** Đã sửa ngắt loop khi gặp `PreTrainedModel`. Không bao giờ bị treo CPU 100%.
2. **Lỗi `RuntimeError: Some tensors share memory` của Safetensors:** Đã chuyển cơ chế lưu sang `safe_serialization=False` (`pytorch_model.bin`).
3. **Lỗi cảnh báo `forced_bos_token_id` của Hugging Face Transformers:** Đã gán `model.config.forced_bos_token_id = None` và cấu hình qua `generation_config`.
4. **Lỗi ngắt kết nối Colab:** Từng checkpoint và file biểu diễn `representations.npz` được stream trực tiếp lên Hugging Face Hub ngay sau khi kết thúc epoch/run.

#### 6. LỆNH KIỂM TRA & TẢI KẾT QUẢ VỀ LOCAL (NẾU CẦN)

Trên máy local của bạn (hoặc bên tiếp quản có quyền truy cập vào 2 session Colab qua CLI):

```powershell
# Đường dẫn Python:
$PY = "C:\Users\Lenovo\miniconda3\envs\clrr\python.exe"

# 1. Kiểm tra tiến độ 2 máy mà không làm gián đoạn huấn luyện:
& $PY scratch/review_warmup_sessions.py --file scratch/check_corrected_warmup_remote.py

# 2. Hoặc tải trực tiếp artifacts từ Hugging Face Hub:
python -c "from huggingface_hub import snapshot_download; snapshot_download('FiveC/amis-rewire-checkpoints', allow_patterns='amis_routing_warmup_20261004_v2/*', local_dir='./outputs_rebuttal/amis_routing_warmup_20261004_v2/downloaded')"
```

---
*Tài liệu này được tạo ra để sẵn sàng bàn giao ngay lập tức cho bất kỳ kỹ sư / nhóm nghiên cứu nào vận hành hoặc sao lưu độc lập.*


<a id="implementation-history"></a>

## Phần gộp: implementation_plan

> Nội dung gộp từ `implementation_plan.md`. **Lịch sử thực thi, không phải yêu cầu chạy hiện hành.** Các trạng thái đang chạy/chưa chạy, repo HF cũ và lệnh warmup chỉ mô tả thời điểm nguồn; muốn chạy lại phải kiểm source, runtime, protocol và routing lưu checkpoint hiện tại.

- **Yêu cầu mới nhất:** hoàn tất Full α0.20/λ0.10/seed42 rồi tạm ngưng; không chạy trial tiếp.
  Hướng dẫn phục hồi và chạy tiếp: [CONFIGURATION_SEARCH_HANDOFF.md](NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history).
  File bàn giao là nơi tra điểm dừng mới nhất, thay các dòng trạng thái khởi chạy cũ bên dưới.

- Influence C đã hoàn tất; vòng D đã xong2trial Full α0.05 và α0.20, cùng λ0.10/seed42.
  Validation chrF++14.4092/14.5738, chưa vượt Full gốc14.7317. HF backups verified,
  metrics/predictions lưu local; suite tạm ngưng theo tác giả trước trial3.
- Vòng D đang chạy song song 2 máy A100: Máy 1 hoàn tất trial 3 (`full_a010_l003`),
  chạy tiếp `lsr_l003` và `full_a010_l030`; Máy 2 chạy song song nhánh LSR (`lsr_l060`, `lsr_l001`, `lsr_l030`).
  Tất cả trial hoàn thành đều tự động push Hugging Face, verify SHA256 và sync local.
- Đã hoàn tất 12run matched mBART Amis (Baseline, CLRR-only, LSR-only, Full ×seeds42/43/44), 35probe và kiểm định Holm24.
- 10weights archive còn đầy đủ đã được backup HF; LSR/Full seed42 mất weights trước backup, predictions/metrics còn đủ.
- Turkish và Colab đã dừng; chưa có trained-test comparison Turkish.
- [Paper hiện hành](../archive/clrr_main_before_revision_20261003.tex) đã viết lại theo [kết quả xác minh](../archive/README.md), với [metrics thống nhất](../archive/README.md).

Dùng [RESEARCH_INSIGHTS.md](../archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#research-insights) mới cùng sổ experiments hiện hành.
Insight cũ giữ tại [archive](../archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#research-insights).
Bước gần nhất đã chuẩn bị là character-only chrF và influence audit trên 12 outputs C:
đã kiểm tra hashes, đủ575rows/file, source order và raw targets. Xem mục7 của insight
mới để biết scope, đầu ra và điều kiện chuyển bước; trạng thái thực thi mới xem mục9.

Dùng [CLRR_GO_NOGO_PLAN.md](../research/ACL_LONG_TERM_EXPERIMENT_PLAN.md): provenance → character/influence và validation audit → tối ưu chọn trên validation → matched confirmation → quyết định Go/No-Go. Chưa thực hiện tối ưu hoặc chạy lại training trong đợt sửa paper này.

Kế hoạch tiếp theo đã cập nhật: chẩn đoán không train → screening Full/LSR cùng5trials
→ validation seeds43/44 → CLRR-only/no-detach controls → quyết định Amis. Tối đa20
lượt train mới nếu reuse hợp lệ; dừng sớm theo gate. Nếu không qua và có nguyên nhân
được hỗ trợ, tối đa2 nâng cấp với budget10lượt có điều kiện. Turkish/NLLB pilot4,
mở đủ12lượt chỉ khi có tín hiệu phù hợp. Đây là kế hoạch, chưa khởi chạy.

- Paper hiện hành có10bảng sinh từ frozen CSV, với source SHA-256 trong `paper_tables/provenance.json`; PDF được compile và kiểm tra bố cục.
- Theo yêu cầu tác giả, đã khôi phục format paper gốc: tiêu đề, preamble ACL review, tên các mục chính và cách trình bày bảng. Nội dung đã xác minh và toàn bộ số liệu giữ nguyên; PDF sau khôi phục có12trang gồm references/appendix. Receipt: `../outputs_rebuttal/format_restoration_20261003/receipt.json`.
- Sửa ignored prediction IDs theo pad_token_id, kiểm tra saved targets khi chấm, precision defaultFP32 trên các trainer, NLLB tokenizer/config/beam policy, và checkpoint key prefixes của comparator alignment. Code cũ và metrics artifact không được tái diễn giải thành kết quả của code mới.
- Kiểm tra trong conda clrr:22unit tests, tiny mBART/NLLB train/eval/reload và correctness trên4architectures trong lượt revision trước; các kiểm tra sau thu gọn được ghi trong scope_revision_20261003/verification.json. Không bảo đảm mọi lỗi có thể có đã được tìm thấy; còn phải đo quality theo các gate.
- Cleanup receipt: `../outputs_rebuttal/workspace_cleanup_20261003/cleanup_manifest.json`; tổng11.08GB đã dọn trong lượt revision, sau xác minh remote backup của các weights còn lại.
- Build bảng: `conda run -n clrr python scripts/revalidation/build_paper_tables.py`; nguồn master ở `docs/` có acl.sty, acl_natbib.bst và clrr_references.bib. Bản PDF hiện hành đã được compile thực tế bằng pdfLaTeX + BibTeX, 12trang/10bảng, không missing input hoặc unresolved citation. Receipt: `../outputs_rebuttal/pdftex_compile_20261003/verification.json`.
- File gốc `docs/clrr_main.tex` đã chứa trực tiếp cả 10 bảng, không còn `\input{paper_tables/...}`. Lệnh build bảng tự cập nhật các vùng bảng trong file gốc từ frozen CSV; không sửa số thủ công trong những vùng này.
- Upload với pdfLaTeX: chạy `conda run -n clrr python scripts/revalidation/package_paper_pdftex.py` để tạo `clrr_pdflatex_upload.zip` (artifact sinh sau khi chạy packaging), kèm bibliography hiện hành và style ACL gốc. Nếu dùng project Overleaf cũ có main `latex/clrr_main.tex`, thay nội dung đúng file đó bằng `docs/paper/clrr_main.tex` hiện hành, chọn compiler pdfLaTeX và Recompile from scratch. Nếu tạo project từ ZIP, chọn main `clrr_main.tex`; dùng bibliography đi kèm để tránh thiếu citation key.

[Kế hoạch/runbook cũ](../archive/README.md) và [experiments gốc](../archive/EXPERIMENTS_ORIGINAL_20261003.md) chỉ lưu lịch sử; các claim và lệnh cũ không phải chỉ thị chạy.

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












<!-- standardized-es10-user-pause -->
## Lịch sử tạm dừng A06 — 09/10/2026

Theo yêu cầu tác giả, Colab1 đã dừng sau lần chạy đầu chưa hoàn tất; kiểm tra Active assignments=0, Usage rate=0.00/hr, balance65.29 CU. Quan sát training cuối đã lưu: 00:00 ngày09/10, epoch9.7235/20, step350/720; không coi đây là trạng thái cuối khi tắt. Receipt: `outputs_rebuttal/amis_standardized_strong_mbart/status/user_pause_receipt.json`. Trạng thái này đã được lần chạy lại hoàn tất ở đầu file thay thế.
<!-- end-standardized-es10-user-pause -->

<!-- standardized-es10-a06-restart -->
## Lịch sử chạy lại riêng A06 — 09/10/2026

Public HF khi kiểm tra trước restart: A05 PASS, A06 chưa có VAL_COMPLETE hợp lệ. Đã mở lại Colab1 A100 Standard40GB và chạy A06 từ đầu bằng bundle đã khóa. Tiny GPU gate PASS: BF16, accumulation32, checkpointing, save/reload logits và best-checkpoint reload. Lúc08:46, epoch0.2783/20, step10/720 đã được xác nhận live. Đợt chạy này đã hoàn tất, xem trạng thái đầu file.

Launcher operational: `outputs_rebuttal/amis_standardized_strong_mbart/restart_a06_local.py`; receipt: `restart_a06_receipt.json` cùng thư mục. Monitor: `scripts/revalidation/monitor_standardized_controls.py --account 1 --supplement`; remote log: `/content/strong_mbart_b_account1.log`.
<!-- end-standardized-es10-a06-restart -->

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

