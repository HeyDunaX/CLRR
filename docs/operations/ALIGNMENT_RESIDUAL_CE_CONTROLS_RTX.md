## Trạng thái hiện hành — 30/10/2026: mở4 selector reruns sau kiểm kê

Người dùng đã duyệt4 runs RTX chọn theo validation BLEU: C0 S42, learned-donor S42, M0 S42, donor S44. Giữ14 kết quả không đổi; train4 đủ20 epochs, lưu riêng best BLEU/chrF++ và dọn weights sau backup verified. Không giữ20 checkpoint. Recipe/receipt và lệnh theo dõi: [BLEU_SELECTION_RTX.md](BLEU_SELECTION_RTX.md). Các số test dưới đây vẫn là bảng chrF++-selected cũ, giữ nguyên; chưa có số mới.


Đã xong11 runs mới; test15 checkpoint và dùng lại3 baseline, đủ6 nhánh×3 seeds,575 câu/nhánh. Input/protocol/selection/hash audit PASS. Test beam4 FP32; recipe RTX SDPA B4×GA32 ES10 giữ nguyên. Baseline backend lịch sử được tái dựng từ stack/default, không có runtime receipt đầy đủ.

| Nhánh | Test BLEU mean ± SD |
|---|---:|
|Baseline|19.7306 ±0.4530|
|C0|19.6194 ±0.9454|
|M0|19.6614 ±0.1122|
|Local|19.7045 ±0.8619|
|Donor|19.2226 ±0.5602|
|LearnedDonor|19.2948 ±0.6639|

Validation practical gates đều PASS; test C0−baseline=−0.1112 BLEU, seed gaps +0.7663/+0.6175/−1.7175. Holm45:1/45 significant, là C0 kém baseline seed44 (p=.0269973). Chưa có improvement significant sau Holm. chrF++ +3.0682 dễ gây hiểu nhầm vì test chỉ có3 reference word-bigrams; chrF0 chỉ+.0711. Exact seed tests không có significance. Amis test đã có development exposure.

**Khuyến nghị HOLD nâng C0 thành phương pháp chính/A100**; không tự train thêm hoặc đổi selector. C0 chỉ vượt donor/learned-donor về mean, chưa vượt Local/M0. Báo cáo đầy đủ: `outputs/experiments/amis_alignment_residual_replication_rtx_test/ANALYSIS.md`; statistics, summary và INPUT_AUDIT cùng workspace. Public analysis prefix: `analysis/amis_mandarin/mbart-large-50/alignment-residual-replication/test` trong FiveC/amis-rewire-checkpoints; kiểm tra ANALYSIS_HF_UPLOAD.json để xác nhận backup.

Các phần dưới là lịch sử, giữ nguyên số và quyết định tại thời điểm ghi.


---

# RTX: donor/module controls và checkpoint probes

## 29/10/2026 — Learned-donor seed42 hoàn tất

CE_LEARNED_DONOR validation BLEU **17.0231**, chrF++ **14.3745**, chrF0 **16.4864**; best epoch20, train đủ20 epoch, stop=epoch cap. C0 tương ứng **17.9412 / 14.9673 / 17.2061**. C0−learned-donor: BLEU **+0.9181**, chrF++ **+0.5927**, chrF0 **+0.7197**. Bootstrap paired 10.000/seed42 trên 576 câu validation: BLEU CI95% [−0.1382,+2.0449], p raw=.046795, Holm3=.102290; cả 3 phép so sánh không đạt p Holm<.05. Kết quả thăm dò một seed, validation đã dùng để phát triển; chưa xác nhận novelty hoặc hiệu quả ổn định.

Checkpoint public `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/alignment-learned-donor-ce-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-rank64/`, upload receipt verified revision `440c15907ad71dcc63939c47889e4f018cd45c6e`, archive SHA256 `6a8fb4aa1d4a86d99494536fc2ded33d9a2961e18b9d17ddc6a856c79f2fff2d`. Đã xác nhận không còn best.zip/best_model/checkpoint-* của run trên RTX. SHA metadata/predictions khớp receipt; protocol và metric replay PASS. Thống kê: `outputs/experiments/amis_learned_donor_ce_rtx/analysis/statistics/PAIRED_BOOTSTRAP.md`. Không đánh giá test.

## Lịch sử setup — learned-donor CE seed42 (29/10/2026)

Tác giả yêu cầu “làm đi” sau review. Phạm vi: **một pilot mới**
`CE_LEARNED_DONOR`, không mở block9/12. Method riêng:
`src/methods/alignment_residual/learned_donor/`, ID
`alignment-learned-donor-ce-v1`. Công thức `h=u+.1[b+A(b)]`, đối chiếu C0
`h=u+.1[b+A(b-u)]`; cùng capacity/init và target dummy. Đây là adapted prior
control để kiểm tra hypothesis, chưa phải claim novelty chính.

Recipe: `configs/experiments/amis_learned_donor_ce_rtx.json`.
Workspace: `outputs/experiments/amis_learned_donor_ce_rtx/`.
Preflight RTX:23542MiB free, GPU0%, process khác462MiB giữ nguyên,
disk free131.88GiB. CPU16/16scientific tests PASS, per-arm wrap/reload PASS.
Bundle ZIP SHA256 `bca74d984df53a06927797f9670c256f66b959ef2e8efdd0b38db14fd7b50c3c`.
Queue detached PID2506609: **CPU16/16 PASS, GPU gate PASS, pilot đã TRAINING**.
GPU gate278.14s, peak14.299GiB;1.310.720extra parameters bằng C0. Cache beam4,
reload best checkpoint và optimizer/scheduler/RNG resume đều PASS. Queue state
`TRAIN`, run metadata `TRAINING`; chưa có final quality scores.
Receipt launch: `outputs/experiments/amis_learned_donor_ce_rtx/launch_rtx.json`.
GPU receipt: `gates/CE_LEARNED_DONOR/PASS.json` trong workspace mới.
Queue chạy lại CPU gate trên remote, GPU2steps đúng B4×GA32/FP32/SDPA,
cache/reload/optimizer-scheduler-RNG resume; chỉ PASS mới train20epochs/ES10.
Không tự thay recipe nếu thiếu VRAM; chờ theo resource admission cũ.

```bash
python scripts/experiment.py build --config configs/experiments/amis_learned_donor_ce_rtx.json
python scripts/experiment.py cpu --config configs/experiments/amis_learned_donor_ce_rtx.json
python scripts/experiment.py launch --config configs/experiments/amis_learned_donor_ce_rtx.json --account 1 --stage gate-train
python scripts/experiment.py monitor --config configs/experiments/amis_learned_donor_ce_rtx.json --account 1
```

Ba lệnh đầu là lịch sử setup; không build/launch lại workspace đã frozen.
Theo dõi trạng thái thực bằng `launch_rtx.json`, `gates/.../PASS.json`,
`runs/.../progress.json`, `VAL_COMPLETE.json`; không coi CPU PASS là GPU PASS.
Khi xong, worker chọn validation chrF++/beam1/FP32 earliest-best, evaluate576
câu, upload+verify rồi mới xóa weights RTX. Không tải weights Windows hoặc test.

HF public:
`FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/alignment-learned-donor-ce-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-rank64/`.

Sau hoàn tất: replay metrics/hash/protocol, so C0−CE_LEARNED_DONOR bằng paired
bootstrap10.000/RNG42 và Holm3 cho BLEU/chrF++/chrF0, family exploratory riêng
đã ghi trước run trong recipe. Giữ số và family cũ, không dùng test chọn method.
Không kết luận equivalence/harm/novelty chỉ từ một seed hoặc p>.05.

## Trạng thái hiện hành

29/10/2026: **queue hoàn tất hai CE controls và ba checkpoint probes**.
Audit SHA/selection/Trainer args/metric replay PASS. CE_DONOR validation
BLEU17.0378/chrF++14.2944; CE_LOCAL17.0637/14.2563; đều đủ20epochs.
C0−CE_DONOR+.9034BLEU, C0−CE_LOCAL+.8775; bootstrap10.000/Holm9 có0/9
significant. **Quality/novelty HOLD; chưa mở training mới/block9/test/A100.**

Kết quả, công thức, primary-source novelty audit, kế hoạch và lệnh tái lập:
[ALIGNMENT_RESIDUAL_CONTROLS_REVIEW.md](../research/ALIGNMENT_RESIDUAL_CONTROLS_REVIEW.md).
Artifacts mới: `outputs/experiments/amis_alignment_residual_ce_controls_rtx/analysis/`.
PID2326160 là queue cũ đã kết thúc, không phải training đang chạy.

Config: `configs/experiments/amis_alignment_residual_ce_controls_rtx.json`.
Bundle SHA256: `623466174d6c63989f39aef89808be32c453ea1fa745f2b6ad70025bbcffefc2`.
Các file method gốc `alignment_residual/v1/model.py` và `module.py` giữ nguyên;
controls riêng ở `src/methods/alignment_residual/controls/`.

## Điều cần phân biệt

C0 tăng+.2589validation BLEU nhưng CI chứa0; C2 không đạt quality gate.
Hai đối chứng mới có point estimates thấp hơn C0, nhưng chưa xác nhận gain
ổn định hoặc cơ chế training của module. C3 có LSR nên không phải
đối chứng local-module CE cho C0. Đây là nghiên cứu development, chưa có claim
novelty hoặc quality GO; không tự mở test/A100.

CLRR-only cũ P1 ở `amis_residual_effect_rtx` đã được lấy riêng JSON/CSV và replay:
BLEU16.6216198322, chrF++14.1251845771; metadata/prediction HF SHA256 khớp sau
khôi phục line endings LF. Không sửa số cũ. Recipe cùng các trường chính,
nhưng implementation gốc không chạy target dummy forward như C0. Vì training
dropout tiêu thụ RNG khác sau mỗi microbatch, không dùng chênh C0−P1 để cô lập
hiệu ứng module. Evidence này không khẳng định RNG là nguyên nhân chênh điểm.
Receipt: `outputs/experiments/amis_alignment_residual_ce_controls_rtx/historical_clrr/REUSE_AUDIT.json`.

## Queue đã chốt

| Thứ tự | Nội dung | Training dài | Mục đích |
|---|---|---:|---|
| 1 | C0: all-on, module-off, donor-off, content-token donor shuffle trên576validation rows; CE + greedy generation |0| Kiểm tra route/module tại checkpoint đã co-adapt; all-on phải replay predictions gốc |
| 2 | C1/C2: CE–LSR gradients theo policy thật,4batch×4câu giống nhau; module/backbone riêng; residual/current norms |0| Đo tác động tức thời của objective và confinement |
| 3 | GPU gates CE_DONOR và CE_LOCAL; đúng B4×GA32/FP32/checkpointing,2steps, reload/cache/resume checks |0 model hoàn chỉnh| Chỉ PASS mới cho train |
| 4 | CE_DONOR: donor d2/α.1, zero correction,0tham số thêm, CE-only + target dummy |1| CLRR-only matched forward/RNG schedule với C0 |
| 5 | CE_LOCAL: q0, rank64 module,1.310.720tham số thêm, CE-only + target dummy |1| Tách added capacity khỏi donor; cùng capacity/init như C0 |

**Tổng2runs seed42 mới.** Không train lại baseline/C0–C3. Các probe dùng checkpoint
validation-best đã khóa,0optimizer steps, kiểm tra parameter versions không đổi.
Gradient probe chạy eval/FP32, không phải phép đo training-time dropout hay tích
lũy Adam. Can thiệp inference không thay cho retrained ablation.

Donor shuffle giữ các vị trí padding/special, chỉ hoán vị content-token donor
trong từng source. Can thiệp đổi cả input của correction lẫn fixed donor addition.
All-on logits parity và donor-off bằng q0 đã được kiểm thử CPU. Nếu all-on
generation không khớp predictions gốc, dừng trước diễn giải và trước training.

Recipe giữ baseline lock SDPA/FP32,LR5e-5,B4/GA32,20epochs,ES10,original
validation chrF++ beam1/FP32 earliest-best. CE_LOCAL khởi tạo module bằng private
generator giống C0. CE_DONOR không có module trainable. Mỗi arm vẫn chạy target
dummy no-grad theo C0 để giữ training dropout draw schedule.

## Lưu và theo dõi

```text
outputs/experiments/amis_alignment_residual_checkpoint_probes_rtx/
  PLAN.json
  AR_C0/interventions.csv, RESULT.json, CLEANUP.json
  AR_C1/RESULT.json, CLEANUP.json
  AR_C2/RESULT.json, CLEANUP.json
  COMPLETE.json

outputs/experiments/amis_alignment_residual_ce_controls_rtx/
  PLAN.json, bundle_manifest.json, CPU_GATE.json, launch_rtx.json
  queue.log, queue.pid, gates/, runs/
  historical_clrr/                        # small local provenance only
```

HF public sau training/validation:

```text
FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/
  alignment-donor-ce-control-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-rank64/
  alignment-local-ce-control-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-rank64/
```

Donor control suffix rank64 biểu thị recipe đối chứng của family; metadata
ghi rõ extra_parameters=0. Không mô tả donor control có rank64 module trainable.
Engine upload/hash verify từng run trước khi xóa weights RTX. Probe chỉ tải
temporary weights trên RTX vào workspace probe; xóa sau khi lưu kết quả và
kiểm tra không update, vì originals đã có backup verified. Không tải weights Windows.

Monitor bằng Python conda `clrr`:

```bash
python scripts/experiment.py monitor --config configs/experiments/amis_alignment_residual_ce_controls_rtx.json --account 1
```

Lệnh monitor phía trên dành cho đọc receipts queue đã hoàn tất. Không còn phải
chờ probe hoặc training trong queue này. Không mở lại queue để theo dõi.

## Kế hoạch phân tích trước đây — đã thực hiện 29/10/2026

Tính lại validation metrics từ predictions; đối chiếu C0−CE_DONOR,
C0−CE_LOCAL, CE_LOCAL−baseline; bootstrap10.000 chung RNG42, Holm9 cho
3contrasts×3metrics, ghi rõ post-hoc development. Các quality gates locked của
C2 cũ giữ nguyên NO-GO. Chưa dùng phép so này để đổi checkpoint hoặc chọn theo test.

Toàn bộ phép so này đã xong; số chi tiết ở report hiện hành liên kết đầu file.

Nếu C0 không vượt matched donor/local controls, không nâng cấp chỉ bằng thêm
capacity. Nếu donor có đóng góp bổ sung, dùng probe để chốt một giả thuyết về
thông tin donor cần giữ trước khi thiết kế candidate. Gating/adapters đơn thuần
chưa có novelty; phải đối chiếu prior trước candidate training. Không tự mở
seeds43/44 hoặc A100 từ queue này.


## Test được duyệt — 30/10/2026

User đã yêu cầu chạy test, tải artifacts nhỏ về phân tích và xác nhận quyền công khai kết quả test/phân tích. Queue detached PID2994523, workspace `outputs/experiments/amis_alignment_residual_replication_rtx_test/`. Có15checkpoint mới cần test: C0/M0/Donor/Local/LearnedDonor, đủ seeds42/43/44; dùng lại3baseline test đã verified. Chọn đúng best validation chrF++/beam1 đã khóa, test575câu/beam4/FP32/TF32off.

Thống kê khóa trong `statistics_design.json`:15paired comparisons×3metrics=45tests, Holm45; báo thêm seed mean/SD và exact seed sign-flip trong family15riêng; interaction chỉ mô tả. Đây là test characterization của development đã xem Amis test trước, không gọi independent confirmation. Validation gates giữ đúng trước replication; không dùng test để chọn lại checkpoint hoặc thay gate.

Public output cùng thư mục checkpoint. Báo cáo phân tích được duyệt trong `analysis/amis_mandarin/mbart-large-50/alignment-residual-replication/test`. Chỉ CSV/JSON/Markdown tải về Windows; weights tải/verify/cleanup trên RTX. Trạng thái mới nhất trong workspace, chưa xem đoạn launch này là receipt hoàn thành.
