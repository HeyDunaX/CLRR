# Thí nghiệm RTX6000

File này ghi các phép đo và pilot của CLRR chạy trên RTX6000. RTX pilot dùng để
sàng lọc phương pháp; kết quả chính của paper vẫn cần được xác nhận bằng các
run A100 cùng protocol.

## Đã chạy: kiểm tra effective rank

| Model | Dữ liệu | GPU | Câu validation | Precision | Effective rank |
|---|---|---|---:|---|---:|
| Baseline mBART seed42 | Amis→Mandarin | RTX6000 | 128 | FP32 | 73.096673 |
| Baseline mBART seed42 | Amis→Mandarin | A100 tham chiếu | 128 | FP32 | 73.096670 |

Chênh lệch RTX−A100 là **+0.0000032**. Checkpoint, các câu và batch32 giống
nhau; strict loading và SHA xác minh thành công. Phép đo dùng tối đa 2393 MiB
VRAM. Không có optimizer step.

Artifacts: `outputs_rebuttal/rtx_rank_smoke/COMPLETE.json` và `VERIFIED.json`.

## Pilot train phương pháp mới: kế hoạch

**Mục tiêu:** kiểm tra xem candidate có tín hiệu đủ hứa hẹn để đáng chạy xác nhận
trên A100 hay không. Đây là pilot development, không phải số kết quả cuối trong
paper.

| Run | Backbone / dữ liệu | Seed | Mục đích |
|---|---|---:|---|
| RTX baseline control | mBART-50 / Amis→Mandarin | 42 | Đối chứng cùng GPU và cùng precision với candidate |
| RTX candidate | mBART-50 / Amis→Mandarin | 42 | Sàng lọc phương pháp mới đã khóa công thức |

### Cấu hình cần khóa trước khi train

- Dùng cùng model revision, dữ liệu và seed42 cho hai run.
- Giữ effective batch, optimizer, learning rate, max20 epochs, early-stopping
  patience10, cách chọn checkpoint bằng validation chrF++ beam1 như protocol
  đã thống nhất.
- RTX6000 không hỗ trợ BF16 như A100; chọn precision chạy được trên RTX và dùng
  giống nhau cho cả baseline control lẫn candidate. Ghi rõ precision thực tế.
- Không tự giảm effective batch hay đổi cách lấy batch của centered LSR để né
  OOM. Nếu không giữ được cấu hình, dừng pilot và ghi cấu hình nào chưa chạy được.
- Ghi revision code, cấu hình đầy đủ, memory, thời gian, best epoch và hash
  checkpoint/predictions trong workspace `/workspace/ura/thanhdat/`.

### Cổng quyết định RTX → A100

So candidate với RTX baseline control trên cùng validation và cùng metric
replay. Candidate chỉ được đề xuất sang A100 nếu đạt ít nhất **+0.5 BLEU hoặc
+0.5 chrF++**, metric còn lại không giảm quá0.2, không có lỗi huấn luyện và
checkpoint/predictions được kiểm tra lại. Nếu không đạt, chưa mở run xác nhận
A100 cho candidate đó.

Đạt cổng chỉ có nghĩa là **đáng xác nhận**. Một seed và một validation set không
chứng minh ý nghĩa thống kê hoặc hiệu quả tổng quát. Không ghép điểm RTX với
điểm A100 thành đối sánh cuối. Trên A100 cần chạy lại candidate đã khóa theo
protocol paper và so với các đối chứng A100.

## Trạng thái

Pilot train RTX **chưa chạy**. Candidate, precision và resolved arguments phải
được ghi vào file này trước khi bắt đầu; không thay đổi phương pháp cũ.
