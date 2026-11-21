# RTX depth screen — 4 runs seed42

## Trạng thái hiện hành: P1 theo baseline SDPA đã khóa

**Đã hoàn tất và kiểm chứng:** P42/P43/P44 đủ20epochs, GPU gates3/3PASS, backup verified3/3. Replay3metrics từ CSV của6systems khớp số lưu; shared Trainer args khớp trừ output/log paths. Validation mean P1−baseline: BLEU−0.1228, chrF++−0.0503, chrF0−0.0873; thắng1/3seeds. Paired bootstrap10000draws/RNG42, Holm9:0/9significant, mọi95%CI chứa0. **HOLD nâng P1 thành method chính hoặc tự mở rộng.** Không chạy test trong audit này. Toàn bộ bảng và giới hạn diễn giải: `outputs/experiments/amis_depth_p1_sdpa_rtx/analysis/VALIDATION_REPORT.md`; audit `AUDIT.json`, statistics `statistics/paired_bootstrap.json`. Các đoạn launch/gates bên dưới là lịch sử, không còn là queue đang chạy.

Tác giả chốt: dùng cấu hình baseline RTX cũ để đối chiếu, chạy lại **P1 seeds42/43/44** theo cấu hình đó. Không train baseline mới. Không gộp P1 eager seed42 vào cohort này. Queue eager4runs bên dưới đã hủy trước training dài; các banner HOLD cũ phía dưới là lịch sử audit.

Recipe thực thi: `configs/experiments/amis_depth_p1_sdpa_rtx.json`.
Protocol lock: `configs/protocols/amis_rtx_baseline_sdpa.json`, SHA256 `7e0edd270019b83813c33fba69f3f8b3724b70fdbd513770c6e4dfc91f31baf0`.
Runner kiểm tra hash lock, dataset/runtime, từng trường protocol và các trường run dùng chung trước build/run; đã kiểm chứng đổi LR sẽ bị từ chối. Không tự thay config theo kết quả.

| ID | Seed | Việc chạy mới | Đối chứng dùng lại |
|---|---|---|---|
| P42 | 42 | P1 ordinary depth difference, SDPA | Baseline R1 từ `amis_token_rtx` |
| P43 | 43 | P1 ordinary depth difference, SDPA | Baseline43 từ `amis_contrastive_multiseed_rtx` |
| P44 | 44 | P1 ordinary depth difference, SDPA | Baseline44 từ `amis_contrastive_multiseed_rtx` |

Backbone giữ SDPA ở encoder/decoder và native top-context của cross-attention được can thiệp. Correction của P1 vẫn tính tường minh `Ad@Vd - At@Vt`, shared Q/K/V/O, donor6 detached và cân norm như v2, decoder12,16 gates khởi tạo0. Không thay công thức theo kết quả. Explicit probabilities thuộc correction của phương pháp; không chuyển native backbone sang eager. Save/load giữ đúng backend, runner từ chối checkpoint backend lệch lock; checkpoint cũ không có trường backend vẫn load eager để tái lập lịch sử.

Recipe khóa: FP32/TF32off, LR5e-5, microbatch4/GA32, max lengths256,20epochs/ES10, AdamW fused, linear warmup.06, weight_decay0, clip1, reentrant checkpointing, CE microbatch mean; mBART revision và CSV hashes như baseline cũ. Validation576 original references, chọn earliest-best chrF++ beam1/FP32; chưa mở test. Kiểm thử CPU P1 SDPA10/10PASS và eager regression10/10PASS. Mọi GPU gates phải PASS trước training; launcher detached có project lock, resource admission và dừng khi lỗi.

HF public: `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/depth-ordinary-v2/seed-{42,43,44}/rtx6000-fp32-sdpa-b4-ga32-es10-depth-screen/`. Baseline giữ nguyên tại `baseline/seed-{42,43,44}/rtx6000-fp32-b4-ga32-es10/`. Upload/verify SHA từng model trước xóa weights local. Workspace mới `outputs/experiments/amis_depth_p1_sdpa_rtx/`; không ghi đè cohort eager.

Lệnh theo dõi (conda `clrr`): `python scripts/experiment.py monitor --config configs/experiments/amis_depth_p1_sdpa_rtx.json --account 1`.

### Receipt launch SDPA

Queue đã launch detached, PID `2058969`, bundle SHA256 `6132f653cbd13ae6e9ca6373663a11262b25ed6dba89e753d81e16e07259d0cb`, manifest SHA256 `30341db51f61edb044650899dd7d66661355a675bba9fd8d1ad01235fb3e87d7`. CPU bundle3/3PASS, project lock được giữ bởi queue. Snapshot sau launch: queue alive, P42 GPU gate đã optimizer step1 với loss6.1398 và gradient norm14.3391 hữu hạn; đang kiểm tra save/reload. **Chưa tuyên bố GPU gates PASS hoặc training dài đã bắt đầu** từ snapshot này. Khi cả3gates có PASS receipts, queue tự chuyển sang3training runs; lỗi bất kỳ sẽ dừng queue.

## Kết quả audit tái sử dụng baseline — 28/10/2026

Baseline RTX seeds43/44 đã có và backup verified; không thiếu model. Dataset hashes và runtime versions khớp recipe depth. Tuy nhiên, loader trong frozen bundle `amis_contrastive_multiseed_rtx.zip` không chỉ định attention backend. Kiểm tra CPU trên RTX bằng Transformers4.57.6/Torch2.6.0 và config đúng pretrained revision cho backend mặc định **SDPA**; recipe depth seed42/new replication chỉ định **eager**. Đây là tái dựng từ source và runtime, không phải receipt backend lấy lại từ lần training cũ.

Do đó chưa xác nhận tái sử dụng như đối chứng **cùng backend**. Khác biệt này không chứng minh số cũ sai, cũng không chứng minh SDPA/eager tạo gap có ý nghĩa; chưa có kiểm chứng tương đương quỹ đạo training. Không đổi implementation của P1 chỉ để ghép baseline mà không khai báo.

Artifact nhỏ: `outputs/experiments/amis_depth_replication_rtx/baseline_reuse_audit/REUSE_AUDIT.json` và `backend_probe.json`. Queue4runs tiếp tục HOLD, chưa training dài. Hai hướng cần tác giả chọn: (A) chạy2P1 và dùng baseline SDPA cũ cho so sánh exploratory, ghi rõ khác backend; (B) chạy4runs baseline/P1 eager để có paired comparison cùng recipe với seed42. Khuyến nghị B nếu mục tiêu là xác nhận gap nhỏ của P1; không tự mở lại training trước quyết định này.

**Cập nhật28/10: queue replication seeds43/44 HOLD trước training dài.** Tác giả nhắc baseline RTX cũ đã có; đã xác nhận hai `VAL_COMPLETE.json` trên HF cho seeds43/44 tại `rtx6000-fp32-b4-ga32-es10`, đủ20epochs và backup verified. Việc mở4runs mà chưa đọc lại đầy đủ `EXPERIMENTS_RTX6000.md` và kiểm tra tái sử dụng là thiếu sót. Backend cũ chưa ghi trong metadata, queue depth mới pin eager; đây là điểm cần xác minh equivalence, không phải bằng chứng baseline cũ sai hoặc bắt buộc chạy lại. Chưa model replication nào train dài. Không tự restart queue4runs trong lúc chưa chốt reuse.

## Phạm vi đã được duyệt

Ngày27/10/2026, tác giả cho triển khai kế hoạch tới bước queue RTX. Đây là screening validation-only, chưa phải phương pháp chính hoặc xác nhận novelty. Queue6 nhánh `amis_depth_difference_v2_rtx` trước đó vẫn dừng ở gates; không chạy lại queue đó.

Recipe: `configs/experiments/amis_depth_screen_rtx.json`.
Runner chung: `scripts/experiment.py`; code riêng: `src/methods/depth_difference/v2/`.
Remote project: `/workspace/ura/thanhdat/projects/CLRR/`.

| Thứ tự | ID | Nhánh | Correction trước head gate |
|---|---|---|---|
| 1 | B0 | Baseline matched | 0 |
| 2 | D0 | Simple donor addition | Ad @ Vd |
| 3 | P1 | Ordinary depth difference | Ad @ Vd - At @ Vt |
| 4 | P5 | U2 mixed | (Ad - At) @ (Vd - Vt) |

Ad/At dùng cùng Q và shared K projection, memories từ encoder6/final. Vd là projection của donor hidden đã cân norm source theo v2; Vt là native top value. Tất cả corrections dùng shared projections, donor detach, gate theo head khởi tạo0 và không thêm dropout riêng. Chỉ decoder12 nhận correction. CE thuần; không LSR/CLRR cũ. D0 là adapted depth-access control, không được gọi là tái lập đầy đủ một prior.

## Fairness và khóa dữ liệu

- Cùng pretrained mBART revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; không continuation từ checkpoint đã fine-tune.
- Amis→Mandarin: train4600, validation576; hashes trong config/PLAN. Source tl_XX, target zh_CN, max lengths256/256.
- RTX FP32, eager attention, BF16/FP16/TF32 OFF; microbatch4 × accumulation32 = effective batch128; checkpointing reentrant.
- LR5e-5, AdamW torch fused, weight decay0, warmup6%, linear schedule, clip1; seed/data_seed42.
- Max20epochs, patience10, chọn original-reference validation chrF++ beam1, earliest best tie; eval batch2. Không dùng test để chọn hoặc đánh giá trong queue này.
- Baseline matched được train mới; không thay hoặc gộp với baseline lịch sử khác backend/precision.
- CPU/GPU gates phải dùng cùng frozen bundle với training. GPU gate2steps, save/reload và beam-cache; mixed thêm optimizer/scheduler/RNG resume. Gate lỗi thì queue dừng trước training.

## Cổng quyết định sau 4 runs

GO exploratory nếu P5 hơn **từng** B0/D0/P1 ít nhất0.30 validation chrF++, BLEU không giảm so với từng nhánh. Báo kèm chrF0; không diễn giải một mình chrF++ như bằng chứng robust quality trên Mandarin.

Không đạt thì dừng recipe U2 hiện tại. Đạt thì mới trình tác giả quyết định3 ablations attention-only/value-only/additive-only. **Không tự động mở thêm3 runs**, không thêm seeds hoặc ngôn ngữ. Single-seed selected-validation không chứng minh significance, generalization hoặc novelty ACL. Amis validation đã được dùng nhiều lần trong development.

## Thực thi

Windows: dùng Python trong conda `clrr`. Các lệnh build/cpu/launch dưới đây cho recipe mới chưa được launch; không lặp build/launch lên workspace đã có receipts:

```powershell
python scripts/experiment.py build --config configs/experiments/amis_depth_screen_rtx.json
python scripts/experiment.py cpu --config configs/experiments/amis_depth_screen_rtx.json
python scripts/experiment.py launch --config configs/experiments/amis_depth_screen_rtx.json --account 1
python scripts/experiment.py monitor --config configs/experiments/amis_depth_screen_rtx.json --account 1
```

Queue detached: tác giả tắt máy cá nhân hoặc dừng chat không làm dừng queue. Script dùng project training.lock; không dừng server RTX hoặc process người khác. Ngưỡng nhận GPU là20000MiB trống; nếu dưới ngưỡng thì chờ. Tại kiểm tra chuẩn bị: free22490MiB, util88%, có process khác; chấp nhận chia sẻ compute theo recipe, không thay batch/precision để chạy nhanh hơn.

## Backup và artifacts

```text
FiveC/amis-rewire-checkpoints/
  checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/
    rtx6000-fp32-eager-b4-ga32-es10-depth-screen/
```

Mỗi model xong thì upload best.zip và artifacts nhỏ, xác minh remote hashes/revision rồi mới xóa weights local của run đó. Upload/verify lỗi: giữ weights và dừng queue. Không đè path pilot cũ. Metadata đầy đủ trong PLAN/spec, run_metadata và checkpoint prototype_config. Không cần tải weights về Windows.

Remote/local workspace receipts: `outputs/experiments/amis_depth_screen_rtx/`. Log remote `queue.log`, gates `gates/<ID>/PASS.json`, training `runs/<ID>/`, queue state `queue_validation_account1.json`. Build bundle SHA256: `608d7df8cf874fa6a5a6c085510008418a2c579c9af6679b03366b1d5ceeb462`.

## Kiểm thử chuẩn bị

- `test_depth_difference_controls.py`:3tests PASS, gồm6operators (cả donor mới), identity/gradients, all-mode cache/serialization và donor train zero-gate/RNG/detach.
- `test_depth_difference_v2.py`:10tests PASS; regression cho mixed, padding/causality, cache/beam, checkpointing, gradients và optimizer updates.
- Trạng thái GPU gates/training phải đọc receipts thực tế; không suy PASS training từ kết quả unit tests.

## Launch receipt

Queue mới đã launch detached, PID `1856223`; bundle SHA256 khớp giá trị trên. CPU gate runner **4/4 PASS** trên B0/D0/P1/P5. Snapshot kiểm tra sau launch: queue alive, B0 GPU gate đã thực hiện optimizer step1/2 với loss và gradient norm hữu hạn; đang lưu checkpoint kiểm thử. Chưa có model training dài tại snapshot này. Script chỉ mở training sau khi cả4 GPU gates có PASS receipts; không suy từ launch rằng mọi gate đã hoàn tất.

## Kết quả thu receipts ngày28/10/2026

Queue `QUEUE_COMPLETE`, hoàn tất B0/D0/P1/P5; process đã kết thúc. Cả4 train đủ20epochs/720steps, stop_reason epoch_cap, không early stopping. Mỗi run có `VALIDATION_COMPLETE_VERIFIED`, backup verified với archive SHA256 và revision riêng trong `VAL_COMPLETE.json`. GPU snapshot free23542MiB, utilization0%. Đây là số từ receipts/predictions đã thu bằng monitor; chưa phải một đợt replay metrics độc lập mới hoặc test evaluation.

| Run | Best epoch | Validation BLEU | chrF++ | chrF0 | ΔBLEU vs B0 | ΔchrF++ vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0 baseline | 19 | 16.7318 | 14.2890 | 16.3574 | 0 | 0 |
| D0 donor addition | 18 | 16.4087 | 13.9162 | 15.9793 | -0.3231 | -0.3728 |
| P1 ordinary difference | 20 | 17.3532 | 14.4979 | 16.7157 | +0.6214 | +0.2089 |
| P5 U2 mixed | 20 | 16.7734 | 14.2040 | 16.2875 | +0.0416 | -0.0850 |

**Gate: NO-GO cho mở rộng recipe U2 mixed.** P5 không vượt baseline chrF++, thua P1 khoảng0.5798BLEU/0.2939chrF++; không đạt điều kiện +0.30chrF++ so với từng comparator. Không mở3 ablations bổ sung. P1 có tín hiệu single-seed so baseline, chưa chứng minh superiority qua seeds hoặc novelty; không tự chuyển nó thành method thắng. Chưa đánh giá test hoặc kiểm định ý nghĩa thống kê trong screen này.

## Audit độc lập và bước xác nhận P1 — 28/10/2026

Tác giả yêu cầu thực hiện các bước1–4: audit, phân tích không train, novelty/cơ chế rồi triển khai baseline/P1 seeds43/44. Không mở lại mixed U2.

### Audit và thống kê

Artifacts: `outputs/experiments/amis_depth_screen_rtx/analysis/`.

- `AUDIT.json`:4/4PASS. Tái tính cả3metrics từ576predictions; source/target order và raw references khớp. So toàn bộ protocol/data với PLAN; resolved Trainer arguments bằng nhau trừ output/log/run paths. Chọn earliest maximum validation chrF++ đúng history, best/completed step và epoch khớp.
- Kiểm tra checkpoint archives bằng LFS SHA256 tại từng revision, không tải weights. Tất cả artifacts nhỏ từ HF được tải tại pinned revisions và so SHA256; JSON/CSV monitor local khớp nội dung. Bytes của CSV local có thể khác vì monitor serialize line endings; không dùng hash serialization local thay hash artifact remote.
- `statistics/paired_bootstrap.json`:10000 paired sentence resamples, RNG42, p-values đối chiếu native SacreBLEU; Holm family3metrics của riêng P1−B0. Không hiệu chỉnh toàn bộ historical search hoặc selection4arms; validation đã dùng chọn checkpoint nên inference chỉ exploratory.

| Metric P1−B0 | Delta | CI95% percentile | p raw | p Holm3 |
|---|---:|---|---:|---:|
| BLEU | +0.6214 | [-0.5612,+1.8503] | .120988 | .362964 |
| chrF++ | +0.2089 | [-0.6144,+1.0284] | .210079 | .362964 |
| chrF0 | +0.3583 | [-0.4860,+1.2054] | .145985 | .362964 |

`strata/`: bỏ từng câu vẫn giữ ΔBLEU dương, khoảng[+.4580,+.7611]; chrF++[+.0747,+.3290]; chrF0[+.2304,+.4702]. Không phải gain chỉ do một câu, nhưng chưa loại khả năng phụ thuộc nhóm câu nhỏ. Sentence chrF0 tăng245, giảm242, hòa89; không có cải thiện đồng đều. Nhóm source length Q1/Q3 BLEU giảm, Q2/Q4 tăng; các quartiles mô tả hậu nghiệm, không dùng tune layer hoặc chọn checkpoint. Không có source trùng training theo chuẩn hóa case/whitespace của script.

### Cơ chế và novelty P1

P1 correction `Cd-Ct`, output native main context cộng `g*(Cd-Ct)`. Ở eval (dropout OFF), trước output projection:

```text
C' = (1-g)*Ct + g*Cd
```

g theo head, không bị chặn vào[0,1], nên đây là affine/signed context fusion, không luôn là convex mixture. Trong train native main attention có dropout còn correction không dropout; không gọi toàn bộ train dynamics đồng nhất với mixture dropout chuẩn. Cùng output projection dùng một lần; detach donor chỉ chặn nhánh phụ, không chặn encoder tuần tự.

Prior đọc phương pháp trực tiếp: [Transparent Attention §2.4](https://aclanthology.org/D18-1338.pdf) trộn encoder representations trước attention; [Every Layer Counts](https://pure.ulster.ac.uk/ws/files/87417499/art_ampomah_et_al.pdf) cho decoder truy cập nhiều encoder layers; [Value Residual Learning §3](https://aclanthology.org/2025.acl-long.1375.pdf) trộn values qua depth trong self-attention. P1 khác chi tiết context-level fusion, shared projections, detach và scaling, nhưng không đủ để gọi là primitive mới. Không nhận là tái lập đầy đủ các papers.

**Verdict:** GO kiểm tra độ ổn định của control P1 theo yêu cầu tác giả; HOLD cho claim phương pháp chính/ACL novelty. Không cần p<.05 trên selected validation mới được đo replication; cũng không dùng việc không significant làm chứng minh tương đương hoặc harm. Four-run replication dưới đây có giá trị chẩn đoán dù P1 cuối cùng chỉ là control. Không chạy benchmark prior hoàn chỉnh hoặc test/Ê Đê trong bước này.

### Queue replication đã duyệt

Recipe mới `configs/experiments/amis_depth_replication_rtx.json`, cùng runner `scripts/experiment.py`. Giữ nguyên protocol/code method của seed42, chỉ đổi seed/data_seed và ID/output paths; giữ mBART revision, original references, eager FP32, reentrant checkpointing, microbatch4×accumulation32, LR5e-5, ES10/max20. Baseline/P1 mỗi seed bắt đầu cùng pretrained gốc. Không continuation từ seed42.

| Thứ tự | ID | Method | Seed |
|---|---|---|---:|
| 1 | B43 | baseline | 43 |
| 2 | P43 | depth-ordinary-v2 | 43 |
| 3 | B44 | baseline | 44 |
| 4 | P44 | depth-ordinary-v2 | 44 |

Reuse frozen seed42 B0/P1 cho tổng hợp3seeds; không train lại seed42. Remote/local workspace `outputs/experiments/amis_depth_replication_rtx/`. Toàn bộ4 GPU gates phải PASS trước training. Kiểm tra nhận GPU:23542MiB free, utilization0%, project lock trống; không can thiệp process khác. Public checkpoint path giữ `<method>/seed-43|44/rtx6000-fp32-eager-b4-ga32-es10-depth-screen/`, không trùng seed42. Upload và verify từng model trước cleanup weights như screen.

Khi xong: báo đủ3paired seed deltas, mean/SD theo seed, validation BLEU/chrF++/chrF0, checkpoint selections và chi phí. Không tự loại seed xấu, không sửa protocol, không tự mở test/Ê Đê hoặc thêm seeds. Consistency là evidence development; n=3 không đủ để hứa significance hay độc lập với selectionseed42. Tác giả quyết định bước tiếp sau tổng hợp.

```powershell
python scripts/experiment.py monitor --config configs/experiments/amis_depth_replication_rtx.json --account 1
```

Launch replication: CPU gates4/4PASS, queue detached PID `2052640`, trạng thái launcher `RTX_GATES_LAUNCHED`. Bundle zip SHA256 `0cd4cc855e4df41f0bee06612a15ef2895c60b3af538893bd5dae4c1a735939b`; manifest hash `6b77f1569fde2fd7b849c887bd0a9ed41eddeffc4068378637118cfed75b8f6c`. GPU free23542MiB lúc launch. Script chạy lại CPU gate trên remote, toàn bộ4GPUgates rồi mới train tuần tự4models. Launch không đồng nghĩa GPU gates/training đã hoàn tất. Queue độc lập máy local.
