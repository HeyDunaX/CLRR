# Quy trình thực nghiệm dùng chung

## Cấu trúc hiện hành

```text
configs/experiments/*.json    Recipe: dataset, backbone, method, seed, account, protocol
src/methods/                Công thức từng phương pháp, có thư mục phiên bản
src/experiment_runner/      Engine, bundle, gate, queue, Colab, upload và fetch
scripts/experiment.py       CLI chung
scripts/colab.py            Lệnh quản lý tài khoản Colab
requirements/colab-experiments.txt  Runtime A100 đã dùng
outputs/experiments/<name>/ Trạng thái và artifacts mới, không đưa vào Git
.cache/                    Bundle và file điều khiển sinh tự động
```

Không tạo một launcher, monitor hoặc trainer mới cho từng đợt chạy. Thêm phương
pháp trong `src/methods/` và registry, rồi khai báo run trong JSON. Logic chọn
checkpoint, accumulation, metrics, export và xác minh upload nằm ở một engine.

Ba JSON hiện có là **recipe tham khảo của các run đã hoàn thành**:
`amis_controls.json`, `amis_strong_lora.json`, `amis_candidates.json`.
Chúng không cấp quyền chạy lại và vẫn trỏ tới checkpoint đã lưu. Launcher chặn
đích public có `best.zip`. Chỉ tạo recipe mới và đường dẫn khác khi đã chốt kế
hoạch; không tự thay tham số để lấy số đẹp hơn.

## Các bước dùng chung

Mọi lệnh Python dùng conda `clrr`. Đây là **mẫu lệnh**, chưa chạy và không phải
kế hoạch train thêm baseline. Thông số không truyền được kế thừa từ `A01` của
recipe tham khảo. Dùng args để tạo recipe mới rồi kiểm tra trước khi chạy:

```bash
conda run -n clrr python scripts/experiment.py create \
  --template configs/experiments/amis_controls.json --from-run A01 \
  --name my_pilot --method baseline --seed 42 --account 1 \
  --lr 5e-5 --batch-size 32 --accumulation 4 --epochs 20 --patience 10 \
  --configuration my-pilot

conda run -n clrr python scripts/experiment.py inspect --config configs/experiments/my_pilot.json
conda run -n clrr python scripts/experiment.py build --config configs/experiments/my_pilot.json
conda run -n clrr python scripts/experiment.py cpu --config configs/experiments/my_pilot.json
```

`--configuration` tạo biến thể **bên trong seed** khi đích chuẩn đã có run cũ;
bỏ args này nếu đó là run chuẩn đầu tiên. CLI từ chối ghi đè recipe/checkpoint.
Dùng `--method full`, `clrr`, `lsr`, `middle-alignment`, `bitfit`, `narrow-lora`,
`strong-lora-a`, `strong-lora-b` hoặc ID candidate trong registry. Với Middle,
chọn template run Middle (ví dụ `A02`) hoặc truyền rõ `--middle-layer-idx`,
`--alignment-weight`, `--temperature`.

Args khác: `--dataset`, `--backbone`, `--model-id`, `--model-revision`,
`--src-lang`, `--tgt-lang`, `--source-prefix`, `--max-source-length`,
`--max-target-length`, `--validation-beams`, `--bleu-tokenizer`, `--workers`.
CLRR/LSR gốc có `--distance`, `--strength`, `--stack`, `--jepa-weight`.
Candidate đã khóa không đổi công thức qua args; thêm phiên bản riêng trong
`src/methods/` và registry nếu thay công thức.

Đổi dataset cần đủ ba CSV `source,target`, language codes và BLEU tokenizer;
CLI tính hash dữ liệu mới. Đổi backbone cần model/revision, language codes và
giới hạn chiều dài. Dùng `--src-lang none --tgt-lang none` cho backbone không
có language tokens. Xem toàn bộ args bằng `python scripts/experiment.py create --help`.

Recipe mới mặc định **validation-only**. Thêm `--allow-test` lúc `create` nếu
kế hoạch có phase test sau khi khóa checkpoint; args này không tự chạy test.
Nhiều run/seed: thêm entries trong `runs` của cùng JSON trước khi build, phân
`account` theo kế hoạch, không cần thêm trainer/script.

`inspect/build/cpu` không mở Colab. Build khóa config, hash nguồn và dữ liệu;
CPU gate kiểm tra backward, accumulation và save/load bằng model nhỏ. Không
thay config/source sau khi khóa. Run cũ dùng bundle nguồn đã pin, không sửa
hash trong receipt lịch sử để ép vượt gate.

Chuẩn bị A100 Standard trên tài khoản đã chọn, rồi triển khai cùng bundle:

```bash
conda run -n clrr python scripts/colab.py --account 1 status -s colab
conda run -n clrr python scripts/experiment.py launch --config configs/experiments/my_pilot.json --account 1
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/my_pilot.json --account 1
```

Lặp `launch/monitor` với account 2 hoặc 3 theo phân công trong JSON. Launcher
không tự mở hoặc thay thế session. GPU gate chạy trên mỗi account trước
training. Một run lỗi sẽ dừng queue đó. Monitor thu JSON nhỏ, không tải weights.
Giám sát 10 phút/lần là lịch vận hành của người/agent chạy, không phải một
daemon được tự mở khi đọc config.

Trên máy GPU đã có bundle và conda `clrr`, dùng chính CLI này:

```bash
conda run -n clrr python scripts/experiment.py run --config configs/experiments/my_pilot.json --account 1 --stage gate-train
```

Resume phải chọn `--id`, `--stage train` và `--resume` rõ ràng. Không resume cả
queue hoặc ghi đè run đã hoàn thành. Engine này giữ gate A100 Standard 40GB,
BF16/checkpointing theo protocol đã kiểm chứng; chưa là launcher RTX6000.

## Test sau khi khóa checkpoint

Sau khi tất cả selected checkpoints đã upload và monitor thu receipts:

```bash
conda run -n clrr python scripts/experiment.py lock --config configs/experiments/my_pilot.json
conda run -n clrr python scripts/experiment.py launch --config configs/experiments/my_pilot.json --account 1 --stage test
```

Lock đối chiếu SHA-256 weights trên HF, pin revision và hash PLAN. Test chạy
đúng selected checkpoints. Config validation-only bị chặn test; không dùng
test để chọn phương pháp, checkpoint hoặc hyperparameters.

## Output local và Hugging Face

Kho public chính: `FiveC/amis-rewire-checkpoints`. Giữ cấu trúc
`checkpoints/<dataset>/<backbone>/<method>/seed-<seed>/[<configuration>/]`.
Weights chỉ được xóa sau khi upload và đối chiếu checksum thành công. Upload
lỗi giữ weights. Config, receipts, số liệu cũ không đổi bởi đợt gom code này.

Đợt dọn repo chỉ xóa output model có bản public trùng hash. Mapping nằm trong
`outputs_rebuttal/repository_cleanup/model_output_index.json`, gồm đường dẫn
local cũ, repo/revision, đường dẫn public và checksum. Tokenizer dùng chung có
thể trỏ tới bản pretrained đã pin khi bytes trùng hoàn toàn. File chưa xác minh
giữ local; diagnostics riêng, probes và representations cũng giữ local.

Lấy lại một artifact nhỏ vào cache khi cần phân tích:

```bash
conda run -n clrr python scripts/experiment.py fetch --local-path <duong-dan-local-cu-trong-index>
```

Không khẳng định toàn bộ output đã lưu trên HF khi chưa đối chiếu. Không xóa
checkpoint chỉ vì tên folder có vẻ đã upload. Bundle nguồn cũ còn cần tái lập
được giữ trong cache; token và cấu hình đăng nhập hiện tại giữ nguyên.

## Mã lịch sử

Các worker/phân tích khoa học chưa có bản thay thế vẫn ở `scripts/revalidation`
để giữ khả năng tái lập. Các launcher/monitor và công cụ migration dùng một lần
đã hoàn thành được dọn. Import worker đối sánh/candidate cũ là lớp tương thích
với engine chung; đường chạy mới dùng `scripts/experiment.py`.
