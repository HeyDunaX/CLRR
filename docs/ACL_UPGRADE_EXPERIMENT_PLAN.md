# QUY CHUẨN ĐỊNH DẠNG KẾ HOẠCH THỰC NGHIỆM (STANDARD EXPERIMENTAL PLAN TEMPLATE)

> [!NOTE]
> **QUY ƯỚC QUẢN LÝ KẾ HOẠCH CHO TOÀN BỘ CÁC PHIÊN LÀM VIỆC (SESSION WORKFLOW CONVENTION):**
> 1. **Chỉ Chỉnh Sửa Trực Tiếp Trên Artifact `implementation_plan`:**
>    * Trong các phiên làm việc hiện tại và tương lai, mọi kế hoạch cụ thể, các bước triển khai chi tiết và danh sách task cần thực hiện sẽ **chỉ chỉnh sửa và cập nhật trực tiếp trên artifact `implementation_plan.md`** của hệ thống Antigravity.
>    * Tránh tạo thêm nhiều file kế hoạch `.md` rời rạc trong thư mục `docs/`.
> 2. **Tài Liệu Này Là Bộ Khung Chuẩn (Canonical Blueprint):**
>    * File này lưu trữ mẫu cấu trúc định dạng (Format Template) chuẩn mực mà USER yêu cầu.
>    * Mọi phiên làm việc sau khi cần khởi tạo hoặc tái cấu trúc `implementation_plan` bắt buộc phải đọc và bám sát chính xác **8 phần bất biến** được định nghĩa dưới đây.

---

## CẤU TRÚC 8 PHẦN BẤT BIẾN CỦA MỘT PLAN THỰC NGHIỆM CHUẨN

Mỗi khi soạn thảo hoặc cập nhật kế hoạch trong `implementation_plan`, nội dung phải tuân thủ nghiêm ngặt cấu trúc 8 phần sau:

---

### PHẦN 1: CALLOUT NGUYÊN TẮC BẤT BIẾN & TỐI ƯU HÓA TÀI NGUYÊN
* **Thẻ định dạng:** Sử dụng GitHub alert `> [!IMPORTANT]`.
* **Nội dung bắt buộc:**
  1. **Tính công bằng học thuật (Fairness):** Nêu rõ backbone chuẩn mực và quy chuẩn đối sánh đồng nhất.
  2. **Code chuẩn xác theo bài báo gốc (Faithful Implementation):** Trích dẫn công thức toán học, số trang, số mục trong paper đối chuẩn.
  3. **Bảo tồn 100% mã nguồn và số liệu cũ:** Tuyên bố đóng băng toàn bộ code trong `src/amis_rewire/`, các kết quả lịch sử và checkpoint đã chốt.
  4. **Tổ chức thư mục riêng:** Mỗi baseline/phương pháp mới phải nằm trong một folder độc lập.
  5. **Bắt buộc chạy Smoke Test:** 1 batch, 1 step trước khi train full để không lãng phí tài nguyên.
  6. **Chạy thuần túy bằng Bash qua SSH Colab (Không dùng Notebook):** Tự động hóa qua terminal, không phụ thuộc trình duyệt.
  7. **Tận dụng ổ SSD NVMe Colab & Bỏ nén ZIP lắt nhắt:** Lưu 2–3 checkpoint tốt nhất trực tiếp trên SSD, chỉ nén Best Model khi kết thúc.
  8. **Tự động push Hugging Face & Tắt máy ngay khi xong:** Đọc token từ môi trường, push checkpoint, gọi `colab stop` để bảo toàn compute units.

---

### PHẦN 2: CÂY THƯ MỤC DỰ ÁN (PROJECT DIRECTORY TREE)
* **Thẻ định dạng:** Khối mã ASCII tree dạng `bash` hoặc `text`.
* **Nội dung bắt buộc:**
  * Thể hiện rõ ranh giới giữa phần cũ được đóng băng (`[KHÔNG ĐỤNG VÀO - BẢO TỒN NGUYÊN VẸN 100%]`) và phần module mới độc lập (`[MODULE MỚI]`).
  * Chỉ rõ vị trí các file mô hình (`model.py`), kịch bản huấn luyện (`train.py`), file chạy điều phối (`scripts/`), và thư mục lưu trữ artifacts (`outputs_.../`).

---

### PHẦN 3: ĐẶC TẢ KỸ THUẬT CHI TIẾT (TECHNICAL SPECIFICATIONS)
* **Nội dung bắt buộc cho từng phương pháp / backbone:**
  * **Trích dẫn BibTeX chính thức** của bài báo tham chiếu.
  * **Công thức toán học chuẩn LaTeX:** Diễn giải rõ các biến, chỉ số tầng, hàm loss, hệ số phạt, nhiệt độ $\tau$.
  * **Cơ chế hoạt động:** Sự khác biệt về mặt cấu trúc so với baseline và mô hình đề xuất của chúng ta.
  * **Chế độ Training vs. Inference:** Cách xử lý đặc thù trong từng chế độ (ví dụ: tắt dropout khi suy luận).

---

### PHẦN 4: BẢNG QUY CHUẨN SIÊU THAM SỐ CÔNG BẰNG TUYỆT ĐỐI (FAIR COMPARISON PROTOCOL)
* **Thẻ định dạng:** Bảng Markdown (`| Cột 1 | Cột 2 | ... |`).
* **Các tham số bắt buộc phải liệt kê so sánh:**
  * Model Backbone
  * Loại Tokenizer & Kích thước từ vựng
  * Số tầng Encoder / Decoder
  * Số tham số huấn luyện bổ sung ($\Delta\theta$)
  * Tập dữ liệu & Tỷ lệ phân chia Train / Val / Test
  * Random Seed
  * Số Epoch tối đa & Điều kiện Early Stopping (tiêu chí metric và patience)
  * Effective Batch Size (Per-device batch size $\times$ Gradient accumulation steps)
  * Learning Rate & Warmup Ratio
  * Precision (BF16 / FP16)
  * Generation Beam Size cho Test set
  * Công cụ và cấu hình đo lường Metric (SacreBLEU tokenizer, chrF++ word order)

---

### PHẦN 5: QUY TRÌNH PREFLIGHT SMOKE TEST BẮT BUỘC
* **Nội dung bắt buộc:**
  * Mục tiêu kiểm thử: Nạp mô hình trên 1 batch nhỏ ($B=2$), chạy 1 forward pass, 1 backward pass (`loss.backward()`), 1 bước sinh mẫu (`generate()`), và kiểm tra quyền ghi Hugging Face Hub.
  * Giới hạn thời gian: $\le 30$ giây.
  * Câu lệnh thực thi cụ thể: `python scripts/smoke_test_....py`.

---

### PHẦN 6: CƠ CHẾ THỰC THI THUẦN TÚY BẰNG BASH QUA SSH COLAB
* **Nội dung bắt buộc:** Sơ đồ luồng xử lý dạng khối (ASCII Flowchart):
  * **Bước 1:** Khởi tạo phiên máy ảo (`colab new -s colab --gpu A100`).
  * **Bước 2:** Chạy Smoke Test xác thực môi trường (`ssh colab "python scripts/smoke_test_..."`).
  * **Bước 3:** Kích hoạt kịch bản chạy nền tự động (`ssh colab "bash scripts/run_....sh"`).
  * **Bước 4:** Tự động tổng hợp bảng điểm, đẩy artifact lên Hugging Face và **ngắt máy ảo ngay lập tức (`colab stop -s colab`)**.

---

### PHẦN 7: DỰ TOÁN THỜI GIAN & TÀI NGUYÊN TRÊN GPU A100
* **Nội dung bắt buộc:**
  * Tổng thời gian GPU thực tế dự kiến (phút / giờ).
  * Lượng Compute Units tiêu hao dự kiến (tính theo đơn giá ~4.2 CU/giờ cho A100).
  * Số dư khả dụng hiện tại và tỷ lệ tiêu hao so với tài khoản.
  * Dung lượng lưu trữ đĩa SSD NVMe tạm thời tại `/content`.

---

### PHẦN 8: BẢNG KẾT QUẢ KỲ VỌNG CHO BÀI BÁO (EXPECTED RESULT TABLE/MATRIX)
* **Thẻ định dạng:** Bảng Markdown hoàn chỉnh với đầy đủ các cột:
  * Backbone Kiến trúc
  * Tên Phương pháp
  * Bài báo tham chiếu
  * Số tham số thêm ($\Delta\theta$)
  * BLEU (zh)
  * chrF++ (word order = 2)
  * Vai trò / Đánh giá học thuật
* **Phần nhận xét sâu sắc:** Luận điểm học thuật then chốt rút ra từ bảng số liệu để đưa thẳng vào phần Results & Discussion của bài báo.

---

## BỘ KHUNG MẪU DÙNG ĐỂ KHỞI TẠO ARTIFACT (STARTER SNIPPET)

```markdown
# [Tên Kế Hoạch Thực Nghiệm Cụ Thể]

> [!IMPORTANT]
> **NGUYÊN TẮC BẤT BIẾN & TỐI ƯU HÓA TÀI NGUYÊN (FAIRNESS, SPEED, BASH SSH & CU PROTECTION):**
> 1. **Mục tiêu học thuật:** ...
> 2. **Bảo tồn 100% mã nguồn và số liệu cũ:** Đóng băng toàn bộ code src/amis_rewire/ và các kết quả đã chốt.
> 3. **Tổ chức module độc lập:** ...
> 4. **Cấu hình công bằng tuyệt đối:** Cố định Seed, Split, Batch size 128, BF16, SacreBLEU zh + chrF++ w=2.
> 5. **Tự động hóa SSH Bash:** Chạy không cần notebook, tự động upload HF và tắt máy colab stop.

## 1. Cấu Trúc Tổng Quan Thực Nghiệm
...

## 2. Cây Thư Mục Dự Án (Project Directory Tree)
...

## 3. Đặc Tả Kỹ Thuật Chi Tiết (Technical Specifications)
...

## 4. Bảng Quy Chuẩn Siêu Tham Số Công Bằng Tuyệt Đối (Fair Comparison Protocol)
...

## 5. Quy Trình Preflight Smoke Test Bắt Buộc
...

## 6. Cơ Chế Thực Thi Thuần Túy Bằng BASH Qua SSH Colab
...

## 7. Dự Toán Thời Gian & Tài Nguyên Trên GPU A100
...

## 8. Bảng Kết Quả Kỳ Vọng Cho Bài Báo (Expected Result Table/Matrix)
...
```
