# Alignment residual: RTX development → A100 confirmation

## Trạng thái hiện hành sau pilot và audit

Đã mở follow-up được tác giả duyệt: checkpoint probes và2CE controls, tách
workspace khỏi4pilot cũ. Xem [ALIGNMENT_RESIDUAL_CE_CONTROLS_RTX.md](ALIGNMENT_RESIDUAL_CE_CONTROLS_RTX.md).
NO-GO của C2 vẫn giữ; câu không mở training dưới đây áp dụng cho mở rộng C2 cũ.

Queue hoàn thành4/4, cả4GPU gates PASS, checkpoint HF backups verified.
Tất cả train đủ20epochs. Validation audit/metric replay PASS;
**C2 quality NO-GO**, novelty vẫn HOLD. Không mở thêm training/test/A100.
Chi tiết kết quả, bootstrap10.000/Holm12 và bước tiếp theo:
[`ALIGNMENT_RESIDUAL_PILOT_AUDIT.md`](../research/ALIGNMENT_RESIDUAL_PILOT_AUDIT.md).
Các PID và snapshot deploy ở cuối file là lịch sử, không phải trạng thái hiện hành.

## Phạm vi

Triển khai theo `docs/research/CLRR_LSR_ADAPTER_FORMULATION.md`.
Novelty **HOLD**: confinement không phải nguyên lý mới; quality chưa có số.
Method cũ và baseline lock không thay đổi. Không tự mở A100 hoặc xem test.

**Quyền chạy hiện hành:** tác giả đã chốt “Tự chạy4pilot sau khi gates PASS”.
Đã nối queue training detached, chờ khóa project của gate queue hiện tại được
nhả rồi xác minh đủ CPU +4GPU receipts cùng manifest, cache parity và resume
PASS trước khi train C0→C1→C2→C3. Thiếu/sai receipt hoặc source thay đổi thì
dừng trước training. Không sửa/rebuild bundle đang chạy. Novelty HOLD vẫn giữ.
Receipt quyền chạy: `outputs/experiments/amis_alignment_residual_rtx/TRAINING_AUTHORIZATION.json`;
trạng thái chờ: `queue_dependency.json`; launcher: `launch_rtx.json`.
Training supervisor PID2160135; gate supervisor PID2156771. Queue chờ bằng
`flock` trên đúng khóa project, không chiếm GPU khi chờ. Cohort này đã được nối;
không chạy lại lệnh launch bên dưới cho cùng workspace. Dùng monitor để xem tiếp.

## Bốn run seed42, cùng pretrained initialization

| Run | Method ID | Điều kiểm tra | BLEU / chrF++ / chrF0 validation |
|---|---|---|---|
| AR_C0 | alignment-residual-capacity-v1 | CLRR + module, CE only; control capacity | 17.9412 / 14.9673 / 17.2061 |
| AR_C1 | alignment-residual-fullgrad-v1 | Cùng module, LSR cập nhật cả backbone | 16.9277 / 14.1528 / 16.2272 |
| AR_C2 | alignment-residual-confined-v1 | LSR chỉ cập nhật module, CE cập nhật toàn bộ | 16.8765 / 14.2070 / 16.3203 |
| AR_C3 | alignment-local-confined-v1 | Không có donor; control local adapter | 17.4056 / 14.5169 / 16.6518 |

C0/C1/C2 cùng fixed route; C3 q=0. Layer3–12, d2, α.1, r64,
module seed10042; λ0 cho C0, λ.1 cho ba nhánh còn lại.
Thêm1.310.720 tham số trên mBART-large-50, full fine-tuning, không gọi PEFT.

## Recipe đã khóa

- Config: `configs/experiments/amis_alignment_residual_rtx.json`.
- Baseline lock: `configs/protocols/amis_rtx_baseline_sdpa.json`.
- SHA256: `7e0edd270019b83813c33fba69f3f8b3724b70fdbd513770c6e4dfc91f31baf0`.
- RTX FP32, SDPA, LR5e-5, B4×GA32, 20epochs, patience10, reentrant checkpointing.
- Model revision/data hashes/runtime đúng baseline RTX; selection original
  validation chrF++576câu, beam1, FP32, earliest best; không test.
- Baseline42 tái dùng R1 cohort `amis_token_rtx`, không train baseline trùng.
- Không so số RTX với baseline A100 để tuyên bố gain.

## Thư mục và quy tắc lưu

Mọi thứ remote nằm trong `/workspace/ura/thanhdat/projects/CLRR/`:

```text
src/methods/alignment_residual/v1/                  # method riêng
configs/experiments/amis_alignment_residual_rtx.json
outputs/experiments/amis_alignment_residual_rtx/
  PLAN.json, bundle_manifest.json, CPU_GATE.json
  gates/AR_C0..AR_C3/                             # GPU PASS receipts
  runs/AR_C0..AR_C3/                              # training riêng từng ID
  analysis/                                      # summary/statistics sau khi chạy
  gate_queue.log, gate_queue.pid                  # chỉ gate
  queue.log, queue.pid                            # gate/train khi đã mở
```

HF canonical public:

```text
FiveC/amis-rewire-checkpoints/
  checkpoints/amis_mandarin/mbart-large-50/<method-id>/seed-42/
    rtx6000-fp32-sdpa-b4-ga32-es10-rank64/
```

Mỗi folder chứa best.zip, config/metadata/receipts, validation predictions và
metrics. Engine kiểm tra đích chưa tồn tại; không ghi đè. Upload/checksum verified
mới dọn weights remote; JSON/CSV/MD nhỏ giữ để tra cứu. Không tải weights Windows.
Nếu được duyệt chạy A100: tạo config/workspace mới có `a100` và hậu tố HF
`a100-<precision>-<batch>-<accumulation>-es10-rank64`; khóa recipe A100 theo
baseline A100, chạy controls tương ứng và ghi riêng `EXPERIMENTS_UPDATED.md`.
Không đổi nhãn RTX thành A100 hay gộp mean giữa hai GPU.

## Lệnh tái sử dụng

Trên Windows dùng Python conda clrr; trên RTX `source environment.sh` trước
và dùng `conda run --no-capture-output -n clrr python ...`.

```bash
# Một lần trước khi chạy, không rebuild workspace đã frozen:
python scripts/experiment.py build --config configs/experiments/amis_alignment_residual_rtx.json
python scripts/experiment.py cpu --config configs/experiments/amis_alignment_residual_rtx.json

# Từ Windows: deploy và chỉ chạy GPU gates, chưa training dài:
python scripts/experiment.py launch --config configs/experiments/amis_alignment_residual_rtx.json --account 1 --stage gate
python scripts/experiment.py monitor --config configs/experiments/amis_alignment_residual_rtx.json --account 1

# Sau khi được chốt training: gates đúng bundle được tái sử dụng;
# mọi gate thiếu phải PASS trước khi queue train C0→C1→C2→C3:
python scripts/experiment.py launch --config configs/experiments/amis_alignment_residual_rtx.json --account 1 --stage gate-train
```

Queue detached, giữ flock riêng project, chờ ≥20.000MiB trống, không đổi batch
khi thiếu VRAM, không tắt process của người khác. Tắt máy local không làm dừng queue.
Chỉ dùng lệnh training sau khi xác minh gate receipts/bundle và trạng thái queue.

## Gates và cách đọc kết quả

CPU: scalar/gradient correctness, zero-up CLRR/local parity, private-init RNG,
parameter-frozen auxiliary gradients, checkpoint replay với dropout, accumulation,
strict save/load và beam4 cache parity. Bundle CPU gate bao phủ đủ4IDs.

GPU cho mỗi ID: đúng B4×GA32/FP32/checkpointing, 2 optimizer steps,
save/reload, cached generation và resume từ step1 kèm optimizer/scheduler/RNG.
PASS kỹ thuật không phải quality GO. Nếu lỗi, dừng trước train và sửa, không lấy
số từ run lỗi. Sửa source sau khi frozen yêu cầu bundle/workspace mới để truy vết.

Quality thresholds đã có trong config/formulation, giữ nguyên trước training.
Primary C2−C1, thêm C2−C0/C3/baseline, 3metrics, bootstrap10.000/Holm12.
Một seed chỉ screening, không bảo đảm seed robustness hay novelty.
Chỉ cân nhắc A100 khi RTX có cải thiện qua controls và giải thích được cơ chế;
không tự chuyển A100 vì validation tăng vài phần trăm điểm.

## Trạng thái kiểm chứng

- Local CPU: 8 scientific tests PASS; bundle gate4/4 IDs PASS.
- GPU-gates-only queue đã launch detached PID2156771; bundle SHA256
  `9a4f152e0bf3be7b52cad832bf7dda22601c305d92090ddfa7b870935fe18ace`.
  Receipt: `outputs/experiments/amis_alignment_residual_rtx/launch_rtx_gate.json`.
  Đây là receipt gate-only ban đầu; queue training đã được nối sau quyền duyệt mới.
- Lock SHA256 xác minh nguyên vẹn.
- Regression của depth/eager/SDPA cũ:29test invocations PASS; không thay số cũ.
- Snapshot trước deploy: Quadro RTX6000, VRAM trống22.822/24.576MiB.
- GPU gate/training status: đọc `rtx_status.json`, `gates/*/PASS.json` và
  `runs/*/VAL_COMPLETE.json`; không coi bảng kế hoạch là kết quả đã chạy.
- Quality scores đã cập nhật ở bảng trên; snapshot gate dưới đây là lịch sử.
- Snapshot GPU: AR_C0 PASS, 253,68giây, peak14,305GiB; beam4 cache parity,
  strict reload và optimizer/scheduler/RNG resume đều PASS. C1–C3 chưa có
  receipt PASS ở snapshot này. Sau snapshot, tác giả đã duyệt tự chạy4pilot;
  queue training đợi tất cả gates PASS như phần quyền chạy hiện hành.
