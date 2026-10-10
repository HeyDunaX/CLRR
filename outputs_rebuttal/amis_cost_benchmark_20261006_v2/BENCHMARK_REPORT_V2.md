# BÁO CÁO BENCHMARK CHI PHÍ & TÍNH TOÁN HIỆU NĂNG V2 (AUDITED)

- **Thiết bị đo:** cuda (NVIDIA A100-SXM4-40GB)
- **Phương pháp nạp trọng số:** Strict-load từ pinned archives của Seed 42, kiểm tra khớp SHA256
- **Quy trình Train:** BF16 autocast, Gradient Checkpointing, BS=32, Accum=4 (eff BS=128), AdamW lr=5e-5, clip norm 1.0 (5 timed repetitions)
- **Quy trình Inference:** Trọng số sạch được reload trước inference; Beam 4 FP32 (Primary) & Beam 1 Greedy (Supplementary), đếm non-padding tokens

## 1. Bảng Tổng Hợp Chi Phí Giữa 4 Nhánh (Seed 42)

| Family | Tổng tham số | Trainable | Train Latency (ms/macro-step) | Overhead vs FT (%) | Overhead vs LSR (%) | Inf Beam 4 Latency (ms/sent) | Inf Beam 4 Throughput (tok/s) | Inf Beam 1 Latency (ms/sent) | Peak Train VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **BASELINE** | 610,879,488 | 610,879,488 | 566.2 ± 23.4 | +0.00% | -9.75% | 32.3 ± 0.5 | 282.8 | 26.4 | 12873.0 |
| **CLRR** | 610,879,488 | 610,879,488 | 575.7 ± 19.0 | +1.67% | -8.25% | 29.3 ± 0.7 | 305.6 | 25.9 | 12873.0 |
| **LSR** | 610,879,488 | 610,879,488 | 627.4 ± 1.5 | +10.81% | +0.00% | 29.2 ± 0.2 | 305.4 | 26.1 | 12989.4 |
| **FULL** | 610,879,488 | 610,879,488 | 632.5 ± 3.4 | +11.70% | +0.80% | 31.7 ± 0.3 | 294.1 | 27.6 | 13007.5 |

## 2. Kết Luận Khoa Học Từ Benchmark V2

1. **Parameter Neutrality:** Cả 4 nhánh đều có chính xác **610,879,488** tham số (0 parameter overhead).
2. **Overhead của CLRR-only:** Train overhead là +1.67% so với FT, xác nhận tính chất routing nhẹ không làm thay đổi đáng kể chi phí huấn luyện.
3. **So sánh Full vs LSR:** Full tăng +0.80% chi phí train so với matching LSR control.
4. **Inference Latency:** Trên chất lượng chuẩn (Beam 4 FP32), Full đạt 31.7 ms/câu so với 29.2 ms/câu của LSR.