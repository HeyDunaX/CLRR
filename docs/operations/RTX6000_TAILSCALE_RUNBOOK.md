# Hướng dẫn Vận hành & Thực thi Thực nghiệm NCKH trên GPU Server (RTX 6000 qua Tailscale SSH)

Tài liệu này cung cấp quy chuẩn vận hành, thiết lập môi trường và quy trình chạy thực nghiệm Nghiên cứu Khoa học (NCKH) trên máy chủ GPU dùng chung (GPU Server), lấy cấu hình tham chiếu là cụm **NVIDIA Quadro RTX 6000** kết nối qua mạng an toàn **Tailscale SSH**.

> [!IMPORTANT]
> **Quy tắc Vàng trên Máy chủ Dùng chung (Shared Server Ethics):**
> 1. **Tuyệt đối không can thiệp vào tiến trình của người khác:** Không dừng (`kill`), đổi độ ưu tiên (`renice`), hoặc chiếm dụng GPU khi người khác đang chạy job.
> 2. **Cô lập triệt để tài nguyên:** Mọi dự án phải có workspace riêng, conda environment riêng, và chuyển toàn bộ cache/temp vào thư mục dự án của mình. Không ghi đè hay cài đè vào môi trường hệ thống.
> 3. **Kiểm tra tài nguyên trước khi chạy:** Luôn kiểm tra VRAM, tiến trình GPU và dung lượng đĩa trước khi xếp lịch chạy job.

---

## 1. Thông tin Máy chủ & Môi trường Tham khảo

Thông tin cấu hình mặc định dùng cho việc kết nối và quản lý:

| Thông số | Giá trị mặc định / Tham khảo | Ghi chú |
|---|---|---|
| **Tailscale IP** | `100.90.53.12` | Địa chỉ IP nội bộ qua Tailnet |
| **User / Hostname** | `ura` / `ura-RTX6000` | Tài khoản được cấp quyền |
| **GPU phần cứng** | NVIDIA Quadro RTX 6000 (24 GB VRAM) | Hỗ trợ CUDA, FP16/BF16 (tùy kiến trúc) |
| **Thư mục gốc người dùng** | `/workspace/ura/thanhdat/` | Toàn bộ dự án đặt tại đây |
| **Quy ước thư mục dự án** | `/workspace/ura/thanhdat/projects/<project_name>/` | Ví dụ: `.../projects/PhonoRepair/` |

---

## 2. Kết nối & Kiểm tra Trạng thái Máy chủ (Pre-flight Checks)

### 2.1. Kết nối qua Tailscale SSH

Từ máy cá nhân (Windows PowerShell hoặc Linux/macOS Terminal):

```powershell
# Kiểm tra trạng thái mạng Tailscale
& 'C:\Program Files\Tailscale\tailscale.exe' status

# Kết nối SSH vào GPU server qua Tailscale
& 'C:\Program Files\Tailscale\tailscale.exe' ssh ura@100.90.53.12
```

*(Trên Linux/macOS, có thể dùng trực tiếp lệnh `tailscale ssh ura@100.90.53.12` hoặc alias trong file `~/.ssh/config`).*

### 2.2. Kiểm tra Tài nguyên Phần cứng & Ổ đĩa

Ngay sau khi đăng nhập thành công vào remote host, thực hiện chuỗi lệnh kiểm tra:

```bash
# 1. Kiểm tra định danh người dùng và máy chủ
whoami
hostname

# 2. Kiểm tra trạng thái GPU (VRAM còn trống, tiến trình đang chạy)
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv

# 3. Kiểm tra dung lượng ổ đĩa khả dụng trong thư mục làm việc
df -h /workspace/ura/thanhdat

# 4. Kiểm tra Python và Conda hệ thống
command -v python3
python3 --version
command -v conda || which conda
```

> [!WARNING]
> Nếu VRAM khả dụng không đủ cho mô hình (ví dụ: cần tối thiểu 10–16 GiB nhưng máy chỉ còn dưới 4 GiB), hoặc GPU đang phục vụ dự án khác với mức sử dụng 100%, bạn **phải chờ hoặc trao đổi lịch chạy với người phụ trách**, tuyệt đối không tự ý kill tiến trình của người khác.

---

## 3. Cấu trúc Dự án & Cô lập Môi trường (Isolation)

### 3.1. Tạo Thư mục Dự án Mới

Mỗi dự án NCKH phải được tổ chức hoàn toàn độc lập theo cấu trúc chuẩn:

```bash
# Đặt tên dự án của bạn (ví dụ: PhonoRepair, ViNID, AudioGen, v.v.)
export PROJECT_NAME="<project_name>"

# Tạo cấu trúc thư mục
mkdir -p /workspace/ura/thanhdat/projects/$PROJECT_NAME
cd /workspace/ura/thanhdat/projects/$PROJECT_NAME

# Tạo các thư mục phục vụ cô lập cache, log, tạm và dữ liệu
mkdir -p cache/huggingface cache/torch cache/pip tmp logs configs data outputs checkpoints
```

Cấu trúc thư mục khuyến nghị:

```text
/workspace/ura/thanhdat/projects/<project_name>/
├── src/                          # Mã nguồn chính của dự án
├── scripts/                      # Scripts thực thi (run_train.py, run_eval.py, ...)
├── configs/                      # Cấu hình thực nghiệm (yaml, json)
├── data/                         # Dữ liệu huấn luyện, benchmark, cache dữ liệu
├── outputs/ (hoặc results/)      # Kết quả xuất ra (metrics, predictions, leaderboards)
├── checkpoints/                  # Lưu checkpoint tạm thời của mô hình
├── conda/envs/<project_name>/    # Môi trường Conda cô lập (hoặc named env)
├── cache/                        # Cache độc lập (huggingface, torch, pip)
│   ├── huggingface/
│   ├── torch/
│   └── pip/
├── logs/                         # File log chạy nền
└── tmp/                          # Thư mục tạm thời
```

### 3.2. Cấu hình Chuyển hướng Cache & Temp

Mặc định, Hugging Face, PyTorch và Pip sẽ lưu cache vào thư mục `/home/<user>/.cache`, dễ gây tràn dung lượng phân vùng hệ thống root `/`. Để tránh điều này, hãy xuất các biến môi trường trỏ trực tiếp về thư mục dự án:

```bash
# Đứng tại thư mục root của dự án (/workspace/ura/thanhdat/projects/<project_name>)
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export PIP_CACHE_DIR="$PWD/cache/pip"
export TMPDIR="$PWD/tmp"
export TOKENIZERS_PARALLELISM=false
```

### 3.3. Khởi tạo Môi trường Conda Riêng biệt

Khởi tạo môi trường ảo độc lập để không làm xung đột gói thư viện với các dự án khác:

```bash
# Cách 1: Tạo môi trường theo prefix đặt ngay trong project (khuyên dùng để quản lý gọn gàng)
conda create -p ./conda/envs/<project_name> python=3.10 -y
conda activate ./conda/envs/<project_name>

# Cách 2: Tạo named conda environment
# conda create -n <project_name> python=3.10 -y
# conda activate <project_name>
```

Cài đặt PyTorch và các thư viện tương thích với phiên bản CUDA của RTX 6000:

```bash
# Cài đặt PyTorch tương thích CUDA (ví dụ CUDA 11.8 hoặc 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt thư viện NCKH cơ bản
pip install transformers datasets accelerate evaluate sentencepiece scikit-learn pandas numpy tqdm

# Nếu dự án có requirements.txt hoặc pyproject.toml
pip install -r requirements.txt
# hoặc: pip install -e .
```

### 3.4. Xác thực Môi trường Đồ họa & GPU Stack

Xác minh xem PyTorch đã nhận diện đúng GPU Quadro RTX 6000 chưa:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('CUDA Version:', torch.version.cuda)"

# Lưu lại thông tin môi trường vào log để đảm bảo tính tái lập (reproducibility)
python -m pip freeze > logs/environment_snapshot.txt
```

---

## 4. Quy trình Thực thi Thực nghiệm (2 Bước Chuẩn)

### Bước 1: Smoke Test (Chạy thử nghiệm nhỏ)

Trước khi tiến hành chạy hàng ngàn mẫu hoặc huấn luyện qua đêm, **bắt buộc** phải chạy một smoke test nhỏ (ví dụ 16–32 mẫu hoặc 1–2 batch) để kiểm tra:
1. Pipeline load model lên CUDA thành công (không crash do thiếu thư viện hoặc thiếu VRAM).
2. Định dạng dữ liệu và tính toán metric không phát sinh ngoại lệ.
3. Thư mục xuất kết quả được tạo đúng quyền ghi.

```bash
# Ví dụ chạy smoke test đánh giá hoặc huấn luyện với giới hạn mẫu nhỏ:
python scripts/run_evaluation.py \
  --model_name <model_key_or_path> \
  --data_path data/benchmark_sample.jsonl \
  --output_dir outputs/smoke_test \
  --batch_size 8 \
  --limit 16 \
  --device cuda
```

Kiểm tra file log và `nvidia-smi` để đảm bảo mức tiêu thụ VRAM nằm trong ngưỡng an toàn.

---

### Bước 2: Chạy Thực nghiệm Chính thức với `tmux`

Để quá trình chạy không bị ngắt quãng khi mất kết nối mạng cá nhân, luôn thực thi qua công cụ quản lý terminal **`tmux`**.

```bash
# 1. Tạo session tmux mới với tên dự án và mã đợt chạy (run_id)
tmux new -s <project_name>_run1

# 2. Di chuyển vào thư mục dự án và kích hoạt môi trường
cd /workspace/ura/thanhdat/projects/<project_name>
conda activate ./conda/envs/<project_name>

# 3. Nạp lại biến môi trường cache
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export PIP_CACHE_DIR="$PWD/cache/pip"
export TMPDIR="$PWD/tmp"
export TOKENIZERS_PARALLELISM=false

# 4. Chạy lệnh thực nghiệm với logging ra file (sử dụng python -u để unbuffer log)
python -u scripts/run_experiment.py \
  --config configs/train_config.yaml \
  --output_dir outputs/run_full_001 \
  > logs/run_full_001.log 2>&1
```

**Các thao tác phím tắt cơ bản trong tmux:**
- **Rời khỏi session (Detach):** Bấm tổ hợp phím `Ctrl + b`, sau đó nhả ra và bấm phím `d`. Job vẫn sẽ tiếp tục chạy trên máy chủ.
- **Quay lại session (Attach):**
  ```bash
  tmux attach -t <project_name>_run1
  ```
- **Xem danh sách các session đang chạy:**
  ```bash
  tmux ls
  ```
- **Xóa session sau khi chạy xong:**
  ```bash
  tmux kill-session -t <project_name>_run1
  ```

---

## 5. Giám sát & Quản lý Tiến trình

Từ một terminal khác kết nối qua Tailscale, bạn có thể theo dõi tiến độ công việc:

```bash
# Theo dõi trực tiếp dòng log mới nhất
tail -f -n 50 /workspace/ura/thanhdat/projects/<project_name>/logs/run_full_001.log

# Giám sát GPU theo chu kỳ 2 giây/lần
watch -n 2 nvidia-smi
```

> [!TIP]
> **Xử lý sự cố Out-Of-Memory (OOM):**
> Nếu gặp lỗi `CUDA out of memory`:
> 1. Giảm `--batch_size` (ví dụ từ 32 xuống 16 hoặc 8).
> 2. Sử dụng Gradient Accumulation (`--gradient_accumulation_steps`) nếu đang train.
> 3. Kích hoạt Mixed Precision (`fp16` hoặc `bf16`) nếu mô hình và GPU hỗ trợ.
> 4. Dùng `torch.cuda.empty_cache()` giữa các chu kỳ đánh giá để dọn sạch bộ nhớ đệm.

---

## 6. Thu thập Kết quả (Artifacts) & Đồng bộ

Sau khi thực nghiệm hoàn tất, kiểm tra các tệp kết quả tạo ra trong thư mục output:
- `metrics.json`: Báo cáo chỉ số định lượng (Accuracy, F1, BLEU, ROUGE, Loss, v.v.).
- `predictions.jsonl` hoặc `.csv`: Dự đoán chi tiết của mô hình trên tập kiểm thử.
- `run_metadata.json`: Lưu vết cấu hình, git commit, seed, thời gian chạy.

### Tải Kết quả về Máy cá nhân

> [!WARNING]
> **Lưu ý về Xác thực Tailscale SSH:**
> `tailscale ssh` sử dụng chứng chỉ xác thực định danh của Tailnet, không mặc định dùng chung SSH Key thông thường của OpenSSH. Do đó, lệnh `scp` truyền thống có thể bị từ chối xác thực (Permission denied). 

Bạn nên sử dụng một trong các phương án truyền file sau:

**Cách 1: Sử dụng `tailscale cp` (Khuyên dùng khi dùng Tailscale):**
```powershell
# Chạy từ máy cá nhân Windows
& 'C:\Program Files\Tailscale\tailscale.exe' cp ura@100.90.53.12:/workspace/ura/thanhdat/projects/<project_name>/logs/run_full_001.log .\logs\
```

**Cách 2: Sử dụng tính năng Download qua VS Code Remote SSH:**
Khi kết nối vào server bằng VS Code Remote SSH, bạn chỉ cần click chuột phải vào thư mục `outputs/` hoặc file log trong cây thư mục Explorer và chọn **Download...**.

**Cách 3: Nếu máy chủ đã cấu hình OpenSSH Key riêng:**
```powershell
New-Item -ItemType Directory -Force "results\<project_name>\" | Out-Null
scp -r ura@100.90.53.12:/workspace/ura/thanhdat/projects/<project_name>/outputs/run_full_001 results\<project_name>\
```

### Lưu trữ Checkpoint lên Hugging Face

Để chia sẻ mô hình hoặc giải phóng dung lượng đĩa của server sau khi hoàn thành huấn luyện, hãy tải checkpoint lên tổ chức **FiveC** theo hướng dẫn tại:
👉 [Quy chuẩn Lưu trữ Checkpoints trên Hugging Face (FiveC)](HUGGINGFACE_CHECKPOINT_STORAGE.md).

---

## 7. Dọn dẹp sau Thực nghiệm

Sau khi đã xác minh kết quả đã được sao lưu an toàn về máy cá nhân hoặc đồng bộ lên Hugging Face:
1. Xóa các checkpoint phụ hoặc file tạm trong `tmp/` và `cache/`.
2. Đóng session tmux đã hoàn tất (`tmux kill-session -t ...`).
3. **Không bao giờ tắt máy chủ GPU dùng chung** (`shutdown` / `reboot`). Chỉ cần thoát SSH bằng lệnh `exit`.
