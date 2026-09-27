# Hướng Dẫn Sử Dụng Google Colab Qua SSH & VS Code / Antigravity IDE

Tài liệu này tổng hợp toàn bộ quy trình, lệnh thường dùng và cách xử lý sự cố khi làm việc với **Google Colab từ xa qua SSH** trực tiếp trên máy Windows cá nhân của bạn.

---

## 1. Giới thiệu & Cơ chế hoạt động

Hệ thống kết nối SSH Colab trên máy của bạn sử dụng công cụ chính chủ **Google Colab CLI (`colab`)**:
* **Không cần bên thứ ba:** Kết nối đi thẳng qua hạ tầng WebSocket bảo mật của Google, không phụ thuộc vào `ngrok` hay `Cloudflare Tunnel`.
* **Cấu hình sẵn trên máy:** 
  * SSH Key chuẩn Ed25519 tại `~/.ssh/id_ed25519`.
  * Token xác thực Google lưu tại `~/.config/colab-cli/token.json`.
  * Cấu hình SSH host `colab` sẵn trong `~/.ssh/config`.

---

## 2. Các cách sử dụng hằng ngày

### Cách 1: Kết nối Terminal nhanh (Khuyên dùng khi chạy train/kiểm tra)
Mở một cửa sổ Terminal (PowerShell) trên máy và gõ:

```powershell
ssh colab
```

👉 Bạn sẽ lập tức được đưa vào môi trường máy ảo Google Colab với quyền `root` tại thư mục `/content`.
* Kiểm tra GPU: `nvidia-smi`
* Kiểm tra CPU / RAM: `htop`
* Chạy code huấn luyện: `python /content/CLRR/src/amis_rewire/train.py`

---

### Cách 2: Mở toàn bộ dự án trong IDE (Remote - SSH)
Dùng khi bạn muốn duyệt cây thư mục, sửa code, debug trực tiếp trên Colab giống như code trên máy cá nhân:

1. Trong **Antigravity IDE** hoặc **VS Code**, nhấn **`Ctrl + Shift + P`** (hoặc `F1`).
2. Gõ và chọn: **`Remote-SSH: Connect to Host...`**
3. Chọn host **`colab`** từ danh sách.
4. Một cửa sổ IDE mới sẽ mở ra kết nối với Colab. Bạn chỉ cần:
   * Bấm nút **Open Folder** (Mở thư mục).
   * Điền đường dẫn: **`/content`**.
   * Toàn bộ file, terminal tích hợp, extension Python sẽ sẵn sàng làm việc!

---

## 3. Quản lý phiên máy ảo (Session & GPU)

> [!NOTE]
> Mặc định cấu hình SSH được gán sẵn cho phiên tên là **`colab`**. Khi tạo phiên, bạn nên đặt cờ `-s colab` để kết nối vào đúng máy ảo đó.

### Khởi tạo máy ảo mới có GPU:
```powershell
# Khởi tạo GPU T4
colab new -s colab --gpu T4

# Khởi tạo GPU A100 (tài khoản Pro/Compute units)
colab new -s colab --gpu A100

# Khởi tạo GPU L4
colab new -s colab --gpu L4
```

### Xem các phiên đang chạy:
```powershell
colab sessions
```
*Kết quả sẽ hiển thị mã phiên, loại GPU (Hardware: T4 / A100 / CPU) và trạng thái.*

### Kiểm tra số dư Compute Units còn lại:
```powershell
colab usage
```

### Dừng máy ảo khi dùng xong (TIẾT KIỆM COMPUTE UNITS):
Sau khi chạy thử nghiệm hoặc training xong, hãy giải phóng máy ảo để tránh bị trừ đơn vị tính toán chạy nền:

```powershell
colab stop -s colab
```

---

## 4. Đồng bộ mã nguồn & Dữ liệu

### Clone project lên Colab:
Khi đã SSH vào Colab (`ssh colab`):
```bash
cd /content
git clone https://github.com/HeyDunaX/CLRR.git
cd CLRR
pip install -r requirements.txt
```

### Copy file giữa máy cá nhân và Colab (SCP):
Từ PowerShell trên máy Windows:
```powershell
# Copy file từ máy cá nhân lên Colab (/content/):
scp -F $HOME/.ssh/config ./data.zip colab:/content/

# Tải checkpoint/kết quả từ Colab về máy cá nhân:
scp -F $HOME/.ssh/config colab:/content/CLRR/checkpoints/best_model.pt ./checkpoints/
```

### Mount Google Drive trực tiếp (để backup dữ liệu không lo mất khi tắt máy ảo):
Trong Terminal Colab hoặc qua lệnh CLI:
```powershell
colab drivemount
```

---

## 5. Xử lý các sự cố thường gặp (Troubleshooting)

### 1. Lỗi `Already-active SSH session (HTTP 429)`
* **Nguyên nhân:** Colab chỉ cho phép duy nhất **1 kết nối SSH đồng thời** vào một máy ảo.
* **Cách xử lý:** Đảm bảo bạn đã đóng các cửa sổ Terminal hoặc IDE cũ đang mở kết nối. Đợi khoảng 10-15 giây để máy chủ giải phóng socket rồi kết nối lại.

### 2. Lỗi `No active sessions found on server`
* **Nguyên nhân:** Bạn chưa bật máy ảo nào hoặc phiên cũ đã hết thời gian (timeout).
* **Cách xử lý:** Tạo lại phiên mới bằng lệnh:
  ```powershell
  colab new -s colab --gpu T4
  ```
  Sau đó gõ `ssh colab` để vào lại.

### 3. Lỗi `Multiple active sessions found`
* **Nguyên nhân:** Bạn có nhiều hơn 1 phiên Colab đang bật cùng lúc.
* **Cách xử lý:** Chạy `colab sessions` xem danh sách các phiên, sau đó dừng bớt các phiên không dùng bằng `colab stop -s <tên_phiên>`.

### 4. Đăng nhập lại nếu token hết hạn (Hiếm khi xảy ra)
Nếu sau này token bị thu hồi hoặc hết hạn, chỉ cần chạy lệnh sau trên PowerShell:
```powershell
colab sessions
```
Làm theo đường dẫn hiển thị để đăng nhập lại tài khoản Google tương tự như lần đầu thiết lập.
