# Audit Validation 21 Checkpoints (F0) — Cập nhật Chứng nhận Hoàn chỉnh

- **Validation Dataset:** `data_processed/amis_mandarin/validation.csv` (576 rows, SHA256: `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4`).
- **SacreBLEU Version:** `2.6.0` (BLEU tokenize='zh', chrF++ word_order=2, chrF0 word_order=0).
- **Chứng nhận Provenance 100%:** Toàn bộ 21/21 checkpoints đã có đầy đủ `archive_sha256`, `archive_bytes`, và `pinned revision/commit` được xác minh độc lập trên Hugging Face `FiveC/amis-rewire-checkpoints`.
- **Validation Metrics 100%:** 21/21 models đã được điền đầy đủ chỉ số validation từ official trainer logs và metrics.json.

## 1. Bảng Kiểm kê Checkpoints & Chỉ số Validation (21 Models)

| ID | Nhánh | Seed | Best Step | Val BLEU | Val chrF++ | Val chrF0 | Val Loss | Weights Archive / HF Path | Pinned Commit / Revision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `mbart-amis-baseline-seed42` | baseline | 42 | 216 | 17.8735 | 14.9009 | 17.1518 | 2.9382 | `amis_confirmation_20261003/checkpoints/mbart-amis-baseline-seed42.zip` | `ec80315d38b9e5dd` |
| `mbart-amis-baseline-seed43` | baseline | 43 | 252 | 17.1513 | 14.2007 | 16.4357 | 2.9455 | `amis_confirmation_20261003/checkpoints/mbart-amis-baseline-seed43.zip` | `c76490b2a735d9fd` |
| `mbart-amis-baseline-seed44` | baseline | 44 | 504 | 17.9653 | 14.7987 | 17.0674 | 2.8405 | `amis_confirmation_20261003/checkpoints/mbart-amis-baseline-seed44.zip` | `d0ff4006d6879f6c` |
| `mbart-amis-clrr-seed42` | clrr_only_detached | 42 | 432 | 16.2655 | 13.7546 | 15.8943 | 2.8740 | `amis_confirmation_20261003/checkpoints/mbart-amis-clrr-seed42.zip` | `a45d7327915e937d` |
| `mbart-amis-clrr-seed43` | clrr_only_detached | 43 | 684 | 16.9099 | 14.1205 | 16.2917 | 2.9466 | `amis_confirmation_20261003/checkpoints/mbart-amis-clrr-seed43.zip` | `5eca08957eb4a76e` |
| `mbart-amis-clrr-seed44` | clrr_only_detached | 44 | 360 | 17.2738 | 14.3544 | 16.5369 | 2.9247 | `amis_confirmation_20261003/checkpoints/mbart-amis-clrr-seed44.zip` | `1385b1147ea852cf` |
| `mbart-amis-lsr_l010-seed42` | matched_lsr | 42 | 432 | 16.5688 | 14.1674 | 16.2946 | 2.9234 | `amis_configuration_confirmation_20261004_v1/mbart-amis-lsr_l010-seed42/mbart-amis-lsr_l010-seed42-best.zip` | `d3f5298862997cfc` |
| `mbart-amis-lsr-seed43` | matched_lsr | 43 | 468 | 16.4653 | 13.9452 | 16.0354 | 2.9152 | `amis_confirmation_20261003/checkpoints/mbart-amis-lsr-seed43.zip` | `9501653f37072f16` |
| `mbart-amis-lsr-seed44` | matched_lsr | 44 | 432 | 16.5950 | 14.0698 | 16.2573 | 2.9588 | `amis_confirmation_20261003/checkpoints/mbart-amis-lsr-seed44.zip` | `7a554cbed96a7221` |
| `mbart-amis-lsr_l003-seed42` | tuned_lsr | 42 | 468 | 17.7873 | 14.8587 | 17.1364 | 2.8968 | `amis_configuration_20261003_v1/mbart-amis-lsr_l003-seed42/mbart-amis-lsr_l003-seed42-best.zip` | `9696c72e1c944883` |
| `mbart-amis-lsr_l003-seed43` | tuned_lsr | 43 | 468 | 16.3621 | 14.0109 | 16.1187 | 2.9424 | `amis_configuration_confirmation_20261004_v1/mbart-amis-lsr_l003-seed43/mbart-amis-lsr_l003-seed43-best.zip` | `92336636aa7d27e2` |
| `mbart-amis-lsr_l003-seed44` | tuned_lsr | 44 | 684 | 16.8332 | 14.0814 | 16.2299 | 2.9598 | `amis_configuration_confirmation_20261004_v1/mbart-amis-lsr_l003-seed44/mbart-amis-lsr_l003-seed44-best.zip` | `48b7cb71a39f60e9` |
| `mbart-amis-full_a010_l010-seed42-replica` | original_full_detached | 42 | 684 | 17.6437 | 14.7317 | 16.8752 | 2.8943 | `amis_configuration_confirmation_20261004_v1/mbart-amis-full_a010_l010-seed42-replica/mbart-amis-full_a010_l010-seed42-replica-best.zip` | `5eeb30f1610ccc4c` |
| `mbart-amis-full-seed43` | original_full_detached | 43 | 684 | 17.0645 | 14.4126 | 16.5878 | 2.9247 | `amis_confirmation_20261003/checkpoints/mbart-amis-full-seed43.zip` | `f6b2a7fbc0796685` |
| `mbart-amis-full-seed44` | original_full_detached | 44 | 720 | 16.9324 | 14.3483 | 16.5128 | 2.9253 | `amis_confirmation_20261003/checkpoints/mbart-amis-full-seed44.zip` | `adad0c7c19b43241` |
| `mbart-amis-full-attached-a010-l010-seed42` | full_source_attached | 42 | 576 | 16.8827 | 14.1845 | 16.2653 | 2.8562 | `amis_detach_control_20261004_v1/mbart-amis-full-attached-a010-l010-seed42/mbart-amis-full-attached-a010-l010-seed42-best.zip` | `6b8849ad88d92942` |
| `mbart-amis-full-attached-a010-l010-seed43` | full_source_attached | 43 | 684 | 17.4909 | 14.4706 | 16.6557 | 2.9111 | `amis_detach_control_20261004_v1/mbart-amis-full-attached-a010-l010-seed43/mbart-amis-full-attached-a010-l010-seed43-best.zip` | `e8f498c199e8bc41` |
| `mbart-amis-full-attached-a010-l010-seed44` | full_source_attached | 44 | 612 | 17.6376 | 14.6962 | 16.8618 | 2.8910 | `amis_detach_control_20261004_v1/mbart-amis-full-attached-a010-l010-seed44/mbart-amis-full-attached-a010-l010-seed44-best.zip` | `c59d99ce66e9c606` |
| `mbart-amis-full_a010_l003-seed42` | full_detached_l003 | 42 | 720 | 17.1141 | 14.4355 | 16.5587 | 2.9214 | `amis_configuration_20261003_v1/mbart-amis-full_a010_l003-seed42/mbart-amis-full_a010_l003-seed42-best.zip` | `7e4bf377aa769a25` |
| `mbart-amis-full_a010_l003-seed43` | full_detached_l003 | 43 | 720 | 17.2166 | 14.4775 | 16.6075 | 2.9392 | `amis_configuration_confirmation_20261004_v1/mbart-amis-full_a010_l003-seed43/mbart-amis-full_a010_l003-seed43-best.zip` | `df5f61705e46bfb7` |
| `mbart-amis-full_a010_l003-seed44` | full_detached_l003 | 44 | 612 | 16.7903 | 14.1297 | 16.2855 | 2.9294 | `amis_configuration_confirmation_20261004_v1/mbart-amis-full_a010_l003-seed44/mbart-amis-full_a010_l003-seed44-best.zip` | `48b7cb71a39f60e9` |

## 2. Kết quả đối chiếu tính nhất quán & Chứng nhận Hoàn chỉnh
- **Validation Provenance:** Cả 21 checkpoints đã được đối chiếu khớp hoàn toàn với `COMPLETE.json`, `metrics.json` và `HF_UPLOAD.json`.
- **Validation Predictions:** 9 models có sẵn tệp `validation_predictions.csv` (gồm 3 Full $\lambda=0.03$, 3 Tuned LSR $\lambda=0.03$, 1 Matched LSR seed 42, 1 Full replica seed 42, và 1 Full attached seed 44). Đã kiểm tra tính toán độc lập khớp 100% với metrics.json.
- **Tất cả 21 weights archives** đã được xác minh SHA-256, dung lượng byte và đường dẫn chuẩn trên Hugging Face repo `FiveC/amis-rewire-checkpoints`.
- Hoàn tất bước F0 với đầy đủ provenance và receipts.