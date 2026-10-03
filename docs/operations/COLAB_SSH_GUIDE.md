# HƯỚNG DẪN KỸ THUẬT GOOGLE COLAB SSH & QUY TRÌNH THỰC THI (COLAB & WORKFLOW GUIDE)

> **Tài liệu hướng dẫn kết nối Google Colab qua SSH, quy trình tự động hóa thực thi bằng Bash script trên máy chủ từ xa, và quy chuẩn lập kế hoạch thực nghiệm.**  
> *Dự án: Parameter-Neutral Cross-Layer Residual Routing and Latent Regularization for Low-Resource Polysynthetic Translation (CLRR)*  
> *Cập nhật mới nhất: Tháng 9/2026*

---

## 1. Giới Thiệu & Kiến Trúc Kết Nối

Hệ thống kết nối SSH Colab trên máy của bạn sử dụng công cụ chính chủ **Google Colab CLI (`colab`)**:
* **Bảo mật & Trực tiếp:** Kết nối đi thẳng qua hạ tầng WebSocket bảo mật của Google, không phụ thuộc vào các dịch vụ bên thứ ba (như `ngrok` hay `Cloudflare Tunnel`).
* **Cấu hình sẵn trên máy cá nhân:**
  * SSH Key chuẩn Ed25519 tại: `~/.ssh/id_ed25519`.
  * Cấu hình SSH host tại: `~/.ssh/config` (`Host colab`).
  * Token xác thực Google lưu tại: `~/.config/colab-cli/token.json`.

---

## 2. Các Lệnh Điều Khiển Thường Dùng Hằng Ngày

### 2.1. Quản lý Máy Ảo Colab (Session Lifecycle)
* **Khởi tạo máy ảo mới (GPU A100):**
  ```powershell
  colab new -s colab_a100 --gpu A100
  ```
* **Bật lại phiên làm việc đã có:**
  ```powershell
  colab start -s colab_a100
  ```
* **Tắt máy ảo ngay lập tức (Bảo toàn Compute Units):**
  ```powershell
  colab stop -s colab_a100
  ```
  *(Lưu ý: Luôn gọi lệnh này ngay khi hoàn tất huấn luyện để không tiêu hao phí CU khi máy nhàn rỗi).*

### 2.2. Tải File Nhanh Qua REST API (Đáng Tin Cậy 100%)
Lệnh `colab download` chạy qua REST API trực tiếp của Google, cực nhanh (2–3 giây) và không bao giờ bị đứt gãy kết nối:
```powershell
# Tải log tiến trình về thư mục scratch
colab download -s colab_a100 /content/lsr_run.log scratch/lsr_run.log

# Tải metrics và predictions về máy
colab download -s colab_a100 /content/CLRR/outputs_revalidation/metrics.json outputs_revalidation/metrics.json
```

### 2.3. Kết Nối Terminal SSH
Mở PowerShell và gõ:
```powershell
ssh colab
```
Bạn sẽ được đưa thẳng vào thư mục `/content` của máy ảo Colab với quyền `root`:
* Kiểm tra GPU: `nvidia-smi`
* Kiểm tra RAM/CPU: `htop`

### 2.4. Mở Toàn Bộ Dự Án Bằng Antigravity IDE / VS Code (Remote - SSH)
1. Trong IDE, nhấn **`Ctrl + Shift + P`** (hoặc `F1`).
2. Gõ và chọn: **`Remote-SSH: Connect to Host...`**
3. Chọn host: **`colab`**.
4. Chọn **Open Folder** và trỏ đến `/content/CLRR`.

---

## 3. Quy Trình Tự Động Hóa Huấn Luyện Bằng Bash Script (Autonomous Workflow)

> [!TIP]
> **TẠI SAO BASH QUA SSH TỐT HƠN NOTEBOOK:**
> 1. **Không bị đứt phiên do trình duyệt:** Trình duyệt tắt hay máy cá nhân sleep thì tiến trình chạy ngầm (`nohup` / `run_wrapper.py`) trên máy chủ vẫn chạy ổn định.
> 2. **Tốc độ I/O tối đa:** Lưu checkpoint trực tiếp trên ổ SSD NVMe `/content/`, loại bỏ cơ chế nén zip trung gian từng epoch.
> 3. **Tự động hóa hoàn toàn từ A đến Z:** Train 20 epochs $\to$ Test 575 câu $\to$ Nén Best Model $\to$ Stream upload Hugging Face $\to$ Tắt máy giải phóng GPU.

### Sơ đồ quy trình thực thi chuẩn:
```
[BƯỚC 1: Khởi tạo Máy ảo A100]
  Lệnh: colab new -s colab_a100 --gpu A100
         │
[BƯỚC 2: Đồng bộ mã nguồn & Preflight Smoke Test]
  Lệnh: python scratch/sync_and_smoke.py (Đảm bảo 1 forward/backward step hoàn tất trong 20s)
         │
[BƯỚC 3: Kích hoạt Huấn Luyện Nền & Giám sát theo chu kỳ]
  Lệnh: nohup python /content/run_wrapper.py > /content/lsr_run.log 2>&1 &
  - Giám sát định kỳ 5-10 phút/lần qua REST API download, không spam lệnh liên tục.
         │
[BƯỚC 4: Xuất Kết quả & Upload Hugging Face]
  - Tự động nén checkpoint tốt nhất thành best.zip (2.27 GB).
  - Stream upload lên FiveC/amis-rewire-checkpoints.
         │
[BƯỚC 5: Tắt máy ảo ngay lập tức]
  Lệnh: colab stop -s colab_a100
```

---

## 4. Bộ Khung Quy Chuẩn 9 Phần Khi Lập Kế Hoạch Thực Nghiệm (Canonical Template Blueprint)

Mọi kế hoạch thử nghiệm mới cần bám sát 9 phần bất biến:

1. **Phần 1: Callout Nguyên Tắc Bất Biến & Tối Ưu Hóa Tài Nguyên:** Cam kết bảo tồn 100% code cũ, giao thức công bằng, smoke test bắt buộc, và bảo vệ Compute Units.
2. **Phần 2: Cây Thư Mục Dự Án (Directory Tree):** Phân định ranh giới module cũ được đóng băng và module mới độc lập.
3. **Phần 3: Phương Pháp & Cơ Sở Toán Học:** Trích dẫn paper chuẩn, số trang, số mục, công thức chính xác.
4. **Phần 4: Bảng Siêu Tham Số Công Bằng Tuyệt Đối:** Cố định seed 42, split dữ liệu, effective batch size, optimizer, learning rate, precision.
5. **Phần 5: Quy Trình Preflight Smoke Test:** Chạy thử batch nhỏ kiểm tra gradient backward trước khi train chính thức.
6. **Phần 6: Cơ Chế Thực Thi Bằng Bash Qua SSH Colab:** Đường dẫn SSD `/content/`, script điều phối, cơ chế lưu best model.
7. **Phần 7: Dự Toán Tài Nguyên Trên A100:** Thời gian dự kiến, ước tính Compute Units tiêu hao, kiểm tra dung lượng đĩa.
8. **Phần 8: Bảng Kết Quả Thực Nghiệm Dự Kiến:** Định dạng các bảng so sánh BLEU và chrF++ theo chuẩn ACL.
9. **Phần 9: Phân Tích Chuyên Sâu & Kế Hoạch Viết Bài Báo:** Luận điểm khoa học thu được và lộ trình đưa vào bản thảo bài báo.
