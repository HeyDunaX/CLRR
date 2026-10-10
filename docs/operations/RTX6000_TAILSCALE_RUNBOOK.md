# RTX6000 qua Tailscale: SSH và chạy CLRR

Ngày: 07/10/2026 (Asia/Saigon).

## Trạng thái hiện hành: SSH ĐÃ KẾT NỐI — 07/10/2026

Dùng từ PowerShell:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' ssh ura@100.90.53.12
```

Đã chạy thành công whoami/hostname/nvidia-smi: userura, hostura-RTX6000, Quadro RTX6000 24576MiB, driver535.261.03. SSH server tự nhận Tailscale và dùng port22. Không cần dò các forwarded ports của IDE. OpenSSH thường chặn vì host key chưa có; Tailscale SSH đã kết nối thành công. Aliasrtx6000 chưa có trong SSHconfig local được kiểm tra.

GPU đang dùng666MiB, utilization0% ở thời điểm kiểm tra. Đĩa chứa home chỉ còn4.7GB (99%used). Có miniconda tại /home/ura/miniconda3 nhưng chưa có envclrr; không chạy Python bằng env khác, chưa cài môi trường hoặc train. Quadro RTX6000 là Turing, không có nativeBF16 như A100; cần đánh giá khả năng chạy recipe riêng trước khi dùng nghiên cứu. Không tự đổi precision trong32runs. Chưa chạy runtimeBF16 probe vì chưa có clrr.

Các mục chẩn đoán trước đó bên dưới là lịch sử, được thay bằng trạng thái này.

## Thông tin kết nối do tác giả cung cấp

- Tailscale IP máy GPU: 100.90.53.12.
- Username SSH: ura.
- Lệnh kết nối từ PowerShell: `ssh ura@100.90.53.12`.
- Đã thử từ công cụ của agent: SSH exit255, không có stdout/stderr. Chưa xác nhận đăng nhập hoặc GPU.

## 1. Trạng thái và thông tin còn thiếu

- Ảnh cho thấy peer tên ura-rtx6000. Chưa xác nhận IP, username SSH, hệ điều hành, GPU thực tế hoặc quyền truy cập.
- 100.122.134.41 trong ảnh là máy Windows desktop-biojune, không phải địa chỉ máy GPU.
- Công cụ terminal của phiên chat lỗi khởi tạo. Công cụ dự phòng gọi Tailscale status nhưng daemon trả Access is denied. Chưa SSH thành công và chưa chạy training trên RTX6000.
- Tác giả đã cung cấp username ura và IP100.90.53.12; cần xác nhận đăng nhập thực tế. Tài khoản GitHub/Tailscale không mặc định là username SSH.
- 32 Amis runs có thể đang chạy trên Colab: phải kiểm tra hàng đợi thực tế trước phân công máy mới. Không chạy lại family/seed đã chạy hoặc đang chạy. Turkish vẫn chưa mở training.

## 2. Kết nối từ PowerShell trên Windows

Chạy trong terminal của chính tài khoản Windows đang đăng nhập Tailscale:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' status
& 'C:\Program Files\Tailscale\tailscale.exe' ping ura-rtx6000
# Thay <user> bằng username Linux, <gpu-ip> bằng IP peer GPU.
Test-NetConnection <gpu-ip> -Port 22
ssh <user>@<gpu-ip>
# Nếu MagicDNS phân giải hostname:
ssh <user>@ura-rtx6000
```

Lần đầu xác minh fingerprint với chủ máy rồi chấp nhận. Không dùng StrictHostKeyChecking=no. Nếu dùng SSH key, chủ máy cần thêm public key vào authorized_keys của đúng user; không chia sẻ private key.

Tailscale tạo mạng riêng; máy đích vẫn cần OpenSSH hoặc Tailscale SSH. Tailscale SSH cần cấu hình SSH và policy cho phép user/thiết bị nguồn đến peer; máy tagged cần rule phù hợp. Không tự thay ACL của cả tailnet. Nếu OpenSSH thông thường chạy qua Tailscale thì xác thực bằng user/key của hệ điều hành như SSH bình thường.

Nguồn: [SSH qua Tailscale](https://tailscale.com/docs/reference/ssh-over-tailscale), [Tailscale SSH và policy](https://tailscale.com/docs/features/tailscale-ssh).

Các lỗi thường gặp:

| Lỗi | Kiểm tra |
|---|---|
| status: Access is denied | Chạy CLI trong terminal của tài khoản có quyền dùng daemon; nếu vẫn lỗi, chủ máy kiểm tra quyền Windows/Tailscale. Không suy ra peer offline |
| Không resolve ura-rtx6000 | Dùng IP 100.x của peer hoặc hostname đầy đủ trong Tailscale |
| Connection refused | SSH service/port trên máy đích |
| Timeout | Peer online, ACL/network access, firewall và port |
| Permission denied (publickey) | Username và SSH key được cấp |
| Permission denied (tailscale) | Tailscale SSH policy cho tagged device và user đích |

## 3. Inventory trên máy GPU (giả định Linux; phải xác nhận)

```bash
whoami
uname -a
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
conda env list
df -h
```

Không mặc định RTX6000 = A100. Tên có thể là Quadro RTX 6000, RTX A6000 hoặc RTX 6000 Ada, khác VRAM và BF16. PyTorch BF16 support và GPU preflight mới quyết định có chạy recipe được không. Nếu BF16 không hỗ trợ, dừng: không tự đổi FP16/FP32 trong cohort đã khóa.

## 4. Triển khai đúng frozen bundle

Dùng directory Linux riêng, không copy toàn bộ outputs/checkpoints. Bundle đã gồm source, Amis CSV, worker, integrity module và hai smoke tests. Không dùng launcher Colab trực tiếp: nó hard-code /content/miniforge3, /content/CLRR, accounts và kiểm tên A100.

Trên Windows, đọc SHA hiện hành và chuyển bundle đang khóa (không rebuild giữa cohort):

```powershell
Get-Content scratch/fresh_confirmation_manifest.json
Get-FileHash scratch/fresh_confirmation_bundle.zip -Algorithm SHA256
ssh <user>@<gpu-ip> 'mkdir -p ~/clrr_confirmation_20261007'
scp scratch/fresh_confirmation_bundle.zip <user>@<gpu-ip>:~/clrr_confirmation_20261007/
```

Trên Linux:

```bash
cd ~/clrr_confirmation_20261007
sha256sum fresh_confirmation_bundle.zip
unzip -q fresh_confirmation_bundle.zip
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false
conda run -n clrr python -c 'import hashlib,json,pathlib; r=pathlib.Path.cwd(); m=json.loads((r/"fresh_bundle_files.json").read_text()); bad=[n for n,h in m.items() if hashlib.sha256((r/n).read_bytes()).hexdigest()!=h]; print("files",len(m),"mismatches",bad); assert not bad'
```

SHA ghi trong bàn giao ngày06/10 là 52fcd0fc2881d414a8ea1ab0e21ca1961eec74cd32b01858fda5c77a3d64bc3c. Đối chiếu manifest thực tế; nếu khác, điều tra phiên bản đang chạy trước khi triển khai, không tự thay bằng bản cũ.

## 5. Môi trường và GPU gate

Mọi Python chạy bằng conda clrr. Nếu máy chưa có env này, tạo riêng với chủ máy; không ghi đè env đang dùng. Worker assert torch2.6.0, transformers4.57.6, accelerate1.15.0. Launcher hiện pin Python3.13 và torch CUDA12.4, cùng tokenizers0.22.2, huggingface_hub0.36.0, numpy2.5.3, sentencepiece0.2.2, protobuf7.36.2, safetensors0.8.0, pandas3.0.6, datasets3.6.0, sacrebleu2.6.0, scipy1.17.1. Đây là pins đọc từ source, chưa xác nhận có thể cài trên máy RTX. Kiểm tra runtime thực tế của cohort đang chạy và wheel availability trước bootstrap. Không dùng requirements-colab.txt rộng để tự suy ra môi trường confirmation tương đương.

```bash
conda run -n clrr python -c 'import torch,transformers,accelerate; print(torch.__version__,transformers.__version__,accelerate.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print("BF16",torch.cuda.is_bf16_supported())'
conda run --no-capture-output -n clrr python tests/smoke_rerun_optimization.py --cuda --result-dir outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate
conda run --no-capture-output -n clrr python tests/smoke_precision_training.py --cuda --audit-rerun --result-dir outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate
```

Gate cần đủ16 optimization cases và4 lifecycle branches PASS, có CUDA/BF16 flags. Lưu receipts mới của RTX6000; không copy gate A100 để giả PASS. Nếu OOM ở recipe batch32/accum4 thì dừng để quyết định protocol: không tự giảm batch hoặc tăng accumulation giữa cohort. GPU khác có thể ảnh hưởng numerical reproducibility; ghi GPU/runtime per run và kiểm tra phân bố family/seed không bị confound với hardware trước pooling.

## 6. Chỉ phân công run còn thiếu sau inventory

Worker đã đọc và có CLI thật:

```bash
conda run --no-capture-output -n clrr python scripts/revalidation/run_fresh_seed_confirmation.py --family full --seed 45 --stage train
```

Lệnh trên là ví dụ, KHÔNG chạy full45 nếu đã có hoặc đang chạy trên Colab. Choices family: baseline, clrr, lsr, full; seeds45–52. Worker dùng namespace amis_fresh_confirmation_20261005, repo HF riêng theo family, từ chối run đã tồn tại. Không đổi prefix để né guard và tạo duplicate.

Recipe đã khóa: mBART Amis→Mandarin, LR5e-5, weight_decay0, warmup.06, batch32×accum4, BF16+checkpointing, lengths256, max20epochs, patience4; chọn checkpoint validation raw chrF++ greedy beam1. Full encoder distance2 alpha.10/lambda.10, routing warmup0. Không đổi patience chỉ vì dừng sớm; các family phải cùng protocol.

HF_TOKEN cần quyền write đúng repos. Nhập kín trong terminal remote (không đặt token trong lệnh/history/docs):

```bash
read -rsp 'HF write token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
```

Để SSH ngắt mà training tiếp tục, dùng tmux đã cài trên máy:

```bash
tmux new -s clrr_confirmation
cd ~/clrr_confirmation_20261007
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false
# Nhập HF_TOKEN kín trong tmux rồi chạy đúng run còn thiếu được phân công.
conda run --no-capture-output -n clrr python scripts/revalidation/run_fresh_seed_confirmation.py --family <family> --seed <seed> --stage train > train_<family>_<seed>.log 2>&1
```

Detach bằng Ctrl-b rồi d; attach lại bằng tmux attach -t clrr_confirmation. Mỗi GPU một model tại một thời điểm. Không tạo queue32 mới khi queue Colab đang chạy. Cần bảng ownership family/seed/host/status trước khi chuyển bất kỳ run nào.

## 7. Theo dõi, HF backup và test

```bash
nvidia-smi
tail -n 40 train_<family>_<seed>.log
find outputs_rebuttal/amis_fresh_confirmation_20261005 -name VAL_COMPLETE.json -print
```

- Worker upload selected checkpoint/provenance lên HF, kiểm pinned revision/hash rồi mới cleanup weights. Chỉ coi hoàn tất khi có VAL_COMPLETE.json VÀ collection_receipt.json verified; file hoàn tất riêng lẻ chưa đủ xác nhận backup.
- Giữ JSON/CSV/logs nhỏ, không tải weights về laptop. Lấy artifacts từ RTX bằng scp theo thư mục cụ thể. Collector/watch hiện dành cho Colab; không giả định chúng tự biết host RTX.
- Chưa có collector RTX turnkey. Nếu chạy research run ở RTX, cần bổ sung thu artifacts vào đúng local family/seed tree và replay verifier, không invent flag --host trong launcher hiện tại.
- Test chỉ mở sau đủ32validation verified và selection_lock.json khóa. Test stage còn cần VAL_COMPLETE.json, collection_receipt.json và selection lock trong đúng paths trên host; không chỉ truyền --stage test lên máy trống.
- Frozen test beam4 FP32/TF32off; không dùng test chọn seed/config. Giữ code/bundle cùng version.
- Khi có người/agent theo dõi chủ động: kiểm10phút/lần và báo chat epoch, best validation, stop reason, HF backup status, lỗi. tail/tmux không tự gửi chat.

## 8. Điều kiện GO / HOLD

| GO | HOLD |
|---|---|
| SSH thành công, host/GPU/runtime xác nhận | Username/IP hoặc quyền SSH chưa có |
| BF16, batch32×accum4 và gate thực tế PASS | BF16 không hỗ trợ, OOM hoặc gate fail |
| Bundle/runtime khớp protocol đã khóa | Source/pins không thống nhất |
| Run còn thiếu có ownership, không duplicate | Colab vẫn đang chạy cùng run hoặc HF đã có run |
| HF write/verification và disk đủ cho training | Upload thất bại hoặc disk thiếu |

Tài liệu này hướng dẫn chuẩn bị và nêu các điểm cần hoàn thiện. Không xác nhận RTX6000 đã kết nối hay training đã chạy.

## Chẩn đoán thực tế mới — 07/10/2026

Đã chạy CLI ngoài sandbox sau khi sandbox bị Access is denied:

- tailscale status: chỉ có máy local desktop-biojune, IP100.76.168.27, account Thanhdat3010@.
- tailscale ping --c 1 --timeout 5s 100.90.53.12: no matching peer.
- tailscale switch --list: chỉ profile8daa, tailnet thanhdat3010.github, account Thanhdat3010@github đang active. Không có profile khác để switch.
- ssh ura@100.90.53.12: banner exchange Connection refused, chưa xác thực user.

Kết luận: peer100.90.53.12 chưa được công bố/hiển thị cho client trong tailnet hiện tại; chưa thể kết luận dịch vụ SSH trên GPU có bật hay không. Ảnh cũ có peerura-rtx6000 và địa chỉ local100.122.134.41, ảnh mới không có danh sách peer. Cần khôi phục quyền nhìn thấy peer qua đúng tailnet hoặc machine sharing từ quản trị viên. Khi status có peer, ping lại rồi SSH. Không mở SSH ra Internet, không tự thay ACL/tài khoản.

Nguồn: https://tailscale.com/docs/how-to/connect-to-devices và https://tailscale.com/docs/reference/troubleshooting/connectivity/connect-device-failure .

## Workspace riêng đã tạo — 07/10/2026

- Path: /workspace/ura/thanhdat, ownerura, mode700.
- Ổ /workspace: tổng458GB, còn111GB tại thời điểm kiểm tra; ổ /priority còn26GB, ổ / còn4.7GB. Không ghi project/cache vào ổroot.
- Đã tạo repo/, outputs/, logs/, cache/{huggingface,torch,pip}/, conda/{envs,pkgs}/, tmp/, credentials/. Chưa copy code, chưa cài env hoặc chạy training.
- environment.sh đặt THANHDAT_WORKSPACE, HF_HOME, TORCH_HOME, PIP_CACHE_DIR, CONDA_ENVS_PATH, CONDA_PKGS_DIRS, TMPDIR trong workspace riêng. Không sửa shell profile chung.

```bash
source /workspace/ura/thanhdat/environment.sh
cd "$THANHDAT_WORKSPACE/projects"
# Khi cài env sau này: dùng /home/ura/miniconda3/bin/conda và conda clrr.
```

Quyền700 tránh userLinux khác đọc nhưng mọi người dùng cùng userura vẫn có quyền. GPU và111GBfree là tài nguyên chung, không phải quota dành riêng. Outputs mặc định của scripts có thể nằm trong repo/outputs_rebuttal; vẫn ở dưới workspace này. Sau HFbackupverified mới cleanup weights thuộc project. Không xóa file người khác.

## Bố cục cuối cùng theo yêu cầu tác giả — 07/10/2026

Workspace tổng /workspace/ura/thanhdat, các project đặt trong projects/<project_name>/. Thư mục clrr_thanhdat đã được đổi tên; bỏ các thư mục repo/outputs/logs/credentials trống cũ. Giữ cache/, conda/, tmp/ và environment.sh dùng chung trong workspace. Chưa triển khai hoặc train CLRR trên RTX6000; tác giả chỉ chuẩn bị máy cho project tương lai.

## Quy định hiện hành — 09/10/2026

Người dùng yêu cầu ưu tiên RTX6000 cho diagnostics, forward/backward probes,
smoke tests và các phép đo không training trong tương lai để tiết kiệm Colab.
SSH qua `tailscale ssh ura@100.90.53.12` đã được kiểm tra lại thành công:
Quadro RTX6000,24576MiB; workspace `/workspace/ura/thanhdat`, ownerura, mode700.
Đã triển khai project và môi trường riêng, và chạy PASS phép đo rank đầu tiên;
chi tiết ở mục smoke verification bên dưới. Không chạy training.

Chỉ ghi và quản lý dữ liệu trong `/workspace/ura/thanhdat/`; project CLRR sẽ ở
`projects/CLRR/` dưới workspace này. Source `environment.sh` để cache, env và
tmp đi vào đúng vùng riêng. Không sửa/xóa các project, môi trường, process hoặc
checkpoint của người khác. Quyền700 không cách ly những người cùng userura.

Giữ cùng checkpoint SHA, validation rows/indices, tokenizer, công thức, versions,
FP32, dropout off và TF32 off khi so diagnostics giữa RTX6000/A100. Replay một
phần phép đo với artifacts A100 đã có; chênh lệch vượt tolerance cần audit trước
khi gộp bảng. Hardware và precision phải có trong runtime receipts.
Không bảo đảm predictions hoặc floating-point outputs giống bit-for-bit giữa GPU.

Centered-LSR gradient diagnostics phải giữ microbatch32 đã khóa. Không tự giảm
batch vì24GB VRAM: centering theo batch sẽ đổi objective. Nếu cần giải pháp bộ
nhớ, phải xác minh tương đương phép đo; chưa được thay protocol để lấy số.
Quy định này dành cho phép đo không training, không đổi hardware của các runs
training đối sánh đã khóa trên A100. Không tự mở Colab nếu có thể đo đúng trên RTX.

## Smoke verification thực tế — 09/10/2026

Trước setup: GPU dùng670MiB, còn23538MiB, utilization0%. Worker kiểm tra lại
trước tải checkpoint: free23380MiB, yêu cầu ít nhất8GiB; thiếu thì dừng,
không evict/kill process người khác. Sau khi worker kết thúc: used666MiB,
free23542MiB, utilization0%. Không tắt máy GPU dùng chung.

Mọi thứ được cài/ghi trong `/workspace/ura/thanhdat/`:

| Thành phần | Đường dẫn |
|---|---|
| Conda runtime riêng | `conda/runtime/` |
| Conda env `clrr` | `conda/envs/clrr/` |
| Conda packages/config | `conda/pkgs/`, `conda/condarc.yaml` |
| Project | `projects/CLRR/` |
| Data | `projects/CLRR/data_processed/amis_mandarin/validation.csv` |
| Code | `projects/CLRR/scripts/revalidation/rtx_rank_smoke.py` |
| Receipts/logs | `projects/CLRR/outputs/rtx_rank_smoke/` |
| HF/Torch/pip/CUDA/Triton/bytecode cache | `cache/` |
| File tạm | `tmp/` |

Không cài vào conda chung ở `/home/ura/miniconda3`, không `conda init` hoặc sửa
shell profile chung. Installer riêng giữ nguyên payload offsets, tắt env
registration vào home; config `register_envs:false`. `environment.sh` trong
workspace đặt private runtime PATH, CONDARC, cache và envs/pkgs/tmp paths.

Phép đo: baseline42 encoder final-LayerNorm source pooled effective covariance
rank trên cùng128validation rows, cùngbatch32 và checkpoint SHA của A100,
FP32/dropout off/TF32 off. Torch2.6.0+cu124, Transformers4.57.6.

| GPU | Effective rank |
|---|---:|
| A100 đã lưu | 73.0966697754 |
| RTX6000 đo lại | 73.0966729845 |
| Delta RTX−A100 | +0.0000032091 |

PASS tolerance absolute1e−3 đã đặt trước phép đo. Peak allocated2393MiB;
strict loading, archive hash, validation hash, indices và worker hash PASS;
zero optimizer steps. Chỉ xác minh phép đo này, không mặc định mọi generation,
gradient hoặc training đều giống trên haiGPU.

Receipt local: `outputs_rebuttal/rtx_rank_smoke/COMPLETE.json`, `VERIFIED.json`.
Chạy monitor bằng conda `clrr` local:

```powershell
conda run -n clrr python scripts/revalidation/control_rtx_rank_smoke.py --stage monitor
```

Để dùng env remote:

```bash
source /workspace/ura/thanhdat/environment.sh
cd /workspace/ura/thanhdat/projects/CLRR
conda run --no-capture-output -n clrr python <script>
```

Luôn kiểm tra `nvidia-smi` trước mỗi job. Không relaunch cùng smoke vì receipt
đã complete; nếu muốn đo lại phải dùng output/job riêng và giữ provenance.
