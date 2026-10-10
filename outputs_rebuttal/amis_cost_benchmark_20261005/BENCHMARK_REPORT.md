# Benchmark Chi Phí & Overhead (Cost & Overhead Benchmark)

**Hardware:** NVIDIA A100-SXM4-40GB | **Precision:** BF16 Training + GC / FP32 Inference
**Effective Batch Size:** 32 per-device x 4 accumulation = 128 (Length: 256/256)

| Family | Total Params | Trainable Params | Train Step (ms) | Train Overhead vs FT | Inf Latency (ms/sent) | Inf Overhead vs FT | Throughput (tok/s) | Peak VRAM Train (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 610,879,488 | 610,879,488 | 586.19 | +0.00% | 23.88 | +0.00% | 612.4 | 12862.7 |
| **clrr** | 610,879,488 | 610,879,488 | 588.13 | +0.33% | 25.33 | +6.04% | 612.0 | 12862.7 |
| **lsr** | 610,879,488 | 610,879,488 | 667.09 | +13.80% | 23.37 | -2.17% | 615.2 | 12923.8 |
| **full** | 610,879,488 | 610,879,488 | 675.53 | +15.24% | 26.08 | +9.19% | 599.1 | 12953.8 |
