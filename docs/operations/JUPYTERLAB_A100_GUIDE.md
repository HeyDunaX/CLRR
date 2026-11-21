# CLRR trên JupyterLab A100

## Trạng thái bàn giao ngày 21/11/2026

- Mở JupyterLab của lab và dùng **Terminal** trong giao diện web. Repo trên server nằm tại `/workspace/long/CLRR`, nhánh `main`, commit đã xác nhận sau khi clone là `6bf13cb`.
- Server có NVIDIA A100 40 GB, driver `525.147.05` theo lần kiểm tra ban đầu. Conda 25.9.1 đã có sẵn. Môi trường `clrr` nằm tại `/workspace/envs/clrr` và gọi được bằng tên `clrr`. Python là 3.11.15.
- Theo báo cáo của Codex trên server, PyTorch chạy được trên GPU, phép thử tensor trả về `1.0`, và smoke test BF16 lưu rồi nạp lại checkpoint với trọng số khớp. Báo cáo nằm tại `/workspace/long/CLRR/setup-result.txt` và chứa đường dẫn JSON gốc. Đọc file đó để lấy phiên bản thực tế và kết quả chi tiết. Smoke test xác nhận môi trường chạy được cho phép thử nhỏ.

## Mở lại phiên làm việc

Trong terminal JupyterLab:

```bash
cd /workspace/long/CLRR
conda activate clrr
git status --short --branch
nvidia-smi
conda run -n clrr python --version
```

Codex CLI đã được cài và cập nhật trong container. Trong terminal cũ còn biến `CODEX_HOME` riêng, dùng `codex resume --last` để mở chat trước đó, hoặc `codex` để bắt đầu chat mới. Nếu mở terminal khác, biến đã export ở terminal cũ không tự chuyển sang. Để dùng lại lịch sử, lấy giá trị bằng `echo "$CODEX_HOME"` ở terminal cũ và export đúng đường dẫn đó trong terminal mới.

Để mở phiên có cấu hình và đăng nhập riêng, dùng một thư mục mới ngoài `/tmp` nhằm tránh cảnh báo tạo lệnh hỗ trợ đã gặp ở phiên trước:

```bash
export CODEX_HOME="$(mktemp -d "$HOME/.codex-clrr.XXXXXXXX")"
printf 'cli_auth_credentials_store = "file"\n' > "$CODEX_HOME/config.toml"
codex login --device-auth
codex
```

Mở link mà CLI in ra và xác nhận bằng tài khoản ChatGPT của bạn. Nếu CLI yêu cầu bật đăng nhập bằng mã thiết bị, bật trong phần Security của ChatGPT rồi chạy lại lệnh đăng nhập. Dùng `codex logout` trong terminal có đúng `CODEX_HOME` khi cần xóa thông tin đăng nhập. Nếu nhiều người dùng chung `root` trong container, họ có thể đọc file xác thực. Muốn cách ly thông tin đăng nhập, dùng tài khoản Linux hoặc container riêng do lab cấp.

## Kiểm tra môi trường khi cần

Cài phụ thuộc nếu dựng lại env từ đầu với driver như trên. Nếu môi trường hiện tại đã qua smoke test, kiểm tra trước khi đổi phiên bản gói.

```bash
conda run -n clrr python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
conda run -n clrr python -m pip install -e .
```

```bash
conda run -n clrr python -m pip check
conda run -n clrr python -c "import torch; x = torch.ones(1, device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), x.item())"
conda run --no-capture-output -n clrr python tests/smoke_precision_training.py --cuda --result-dir outputs_smoke/setup_a100
```

Nếu Codex báo `bwrap` không tạo được namespace, lệnh chưa chạy trong sandbox. Xem lệnh được đề nghị và chọn **Yes, proceed** để cho phép từng lần với lệnh bạn đã kiểm tra, chẳng hạn `nvidia-smi`. Chạy trực tiếp trong terminal JupyterLab cũng là một cách kiểm tra môi trường.

## Thao tác trong Codex CLI

- `/model` đổi model, `/status` xem phiên, `/permissions` xem quyền, `/exit` thoát ra shell.
- **Ctrl+J** xuống dòng, **Enter** gửi.
- Copy về Windows bằng cách **giữ Shift khi kéo chuột bôi đen**, rồi giữ **Shift** và nhấp chuột phải để chọn **Copy** trong menu trình duyệt. Có thể thử Ctrl+C khi terminal đã chọn chữ. Cách bôi đen bằng Shift đã dùng được trong phiên này.

## Việc tiếp theo

Đọc [quy trình chạy thí nghiệm](EXPERIMENT_WORKFLOW.md) và [sổ kết quả](../experiments/RESULTS_REGISTRY.md), rồi đối chiếu cấu hình, seed, controls và receipts để tìm bước kiểm chứng còn thiếu. Nếu kết quả không có trong bản clone, ghi chưa xác minh và tìm artifact gốc trước khi chạy lại. Xác nhận thí nghiệm được phép dùng A100 theo tài liệu dự án trước khi chuyển cấu hình từ RTX. Dữ liệu đã xử lý nằm ở `data_processed/<dataset_name>/` với `train.csv`, `validation.csv` và `test.csv` có cột `source,target`.

Prompt để mở phiên mới:

> Đọc `docs/operations/JUPYTERLAB_A100_GUIDE.md` và `setup-result.txt`. Rà trạng thái các run liên quan, đối chiếu config với receipt và tránh chạy trùng. Dùng Conda `clrr`. Chỉ cài phụ thuộc còn thiếu. Đề xuất tối đa ba việc tiếp theo và chạy một kiểm tra nhỏ nếu bước đó được phép dùng A100. Lưu kết quả và đường dẫn log vào `outputs_smoke/next_steps.md`. Dừng trước training dài, quét nhiều seed hoặc đánh giá tập test.
