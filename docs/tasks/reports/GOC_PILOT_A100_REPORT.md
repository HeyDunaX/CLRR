# Báo cáo Pilot GOC A100 — 9 runs, validation-only

**Đã hoàn tất và xác minh backup cả 9 runs.** Chỉ seed42; kết quả thăm dò trên validation đã dùng phát triển, chưa chứng minh ổn định đa seed, test độc lập hoặc novelty ACL.

Source commit: `bb4c0fcd8d5f9014544c090dd55a50b27216998a`. Bundle manifest SHA256: `a45119f868d9574cae99e655aed39e646988efb176ca53b3fccb4baeff634308`.
Tất cả 9 GPU gates PASS trước khi model đầu tiên train dài. Lệnh mới nhất của tác giả cho phép tự mở Pha2 sau rào này, thay cho câu chờ duyệt trong file kế hoạch. Không chạy stage test, IPAM hoặc thêm run ngoài thiết kế.
CPU unit tests 15/15 PASS; CPU gate đủ9cases PASS. Đã sửa duy nhất lifetime object trong hai fixture CPU để tránh NFS giữ mmap khi dọn temp; không đổi objective, tolerance, recipe. Bundle chứa đúng source đã sửa trước khi đóng băng.

## Protocol

Amis→Mandarin, 4600train/576validation; mBART-50 revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; tl_XX→zh_CN; seed/data_seed42; BF16/TF32 training, SDPA, B32×GA4, LR5e-5, cap20epochs/ES10; selection eval_bleu, earliest-best; validation beam1 FP32, TF32 off, original CSV refs, BLEU zh, chrF0/chrF++. Test CSV chỉ hash/đóng bundle.

## GPU correctness gates

| ID | PASS | Peak allocated GiB | Seconds | Save/reload logits | Explicit optimizer/scheduler/RNG resume replay |
|---|---|---:|---:|---|---|
| BASELINE_S42 | True | 11.873824 | 198.499201 | True | NA |
| W_B125_S42 | True | 15.753971 | 363.584793 | True | True |
| W_B250_S42 | True | 15.753971 | 316.888268 | True | True |
| GOC_B125_S42 | True | 15.753971 | 317.788419 | True | True |
| GOC_B250_S42 | True | 15.753971 | 336.853563 | True | True |
| GOC_SYM_B125_S42 | True | 15.753971 | 359.295167 | True | True |
| GOC_SYM_B250_S42 | True | 15.753971 | 363.547678 | True | True |
| GOC_INV_B125_S42 | True | 15.753971 | 335.818412 | True | True |
| GOC_INV_B250_S42 | True | 15.753971 | 342.909705 | True | True |

Baseline gate kiểm checkpoint/optimizer/RNG hiện diện và save/reload; runner không ghi explicit resume-replay flag cho baseline. NA được giữ nguyên, không tự gán PASS cho phép đo chưa thực hiện. Tám consistency gates có replay flag theo receipt.

## Kết quả validation576

| ID | Method | beta | Best epoch/step | Stop | BLEU | chrF0 | chrF++ | Val loss at best | Sampled train GPU GiB | Wall train/eval/export s |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| BASELINE_S42 | baseline | 0 | 19.0/684 | epoch_cap | 17.526392 | 16.763185 | 14.538191 | 2.891342 | 16.324219 | 3048.728595 |
| W_B125_S42 | rdrop-consistency-v1 | 1.250000 | 16.0/576 | epoch_cap | 17.706895 | 17.081689 | 14.885190 | 2.749928 | 20.039062 | 3963.160393 |
| W_B250_S42 | rdrop-consistency-v1 | 2.500000 | 14.0/504 | epoch_cap | 17.380221 | 16.802230 | 14.596897 | 2.585387 | 20.039062 | 3912.711515 |
| GOC_B125_S42 | gold-ordered-consistency-v1 | 1.250000 | 20.0/720 | epoch_cap | 18.094444 | 17.330529 | 15.171115 | 2.870528 | 20.039062 | 4129.207590 |
| GOC_B250_S42 | gold-ordered-consistency-v1 | 2.500000 | 16.0/576 | epoch_cap | 17.809720 | 17.088690 | 14.866493 | 2.748828 | 20.039062 | 6589.488249 |
| GOC_SYM_B125_S42 | symmetric-stopgrad-consistency-v1 | 1.250000 | 19.0/684 | epoch_cap | 18.402209 | 17.440653 | 15.217122 | 2.732147 | 20.039062 | 4429.530766 |
| GOC_SYM_B250_S42 | symmetric-stopgrad-consistency-v1 | 2.500000 | 18.0/648 | epoch_cap | 18.514274 | 17.540467 | 15.326565 | 2.604904 | 15.548828 | 4678.398864 |
| GOC_INV_B125_S42 | inverted-gold-consistency-v1 | 1.250000 | 18.0/648 | epoch_cap | 16.221322 | 15.632818 | 13.569997 | 2.507108 | 17.230469 | 4601.412699 |
| GOC_INV_B250_S42 | inverted-gold-consistency-v1 | 2.500000 | 17.0/612 | epoch_cap | 15.146553 | 15.089962 | 13.081228 | 2.467135 | 17.230469 | 4772.396309 |

Training VRAM là high-water quan sát bằng nvidia-smi, sampling2s; PID host khác PID container nên chỉ nhận mẫu khi có duy nhất1processGPU và duy nhất1pilotprocess đang giữ /dev/nvidia0, bỏ qua mẫu mơ hồ. Số này gồm driver/context/cache, không phải exact torch allocator peak và có thể bỏ lỡ spike giữa samples. Gate peak là torch.cuda.max_memory_allocated, báo riêng. Wall time dùng metadata completed_unix−started_unix: gồm train/eval/export/reload, không bao gồm tải model ban đầu và upload cuối.

| ID | Archive SHA256 | Verified HF revision |
|---|---|---|
| BASELINE_S42 | `985f9fc49574c111fd885b3ef076bba477aeb9f6bd6fd01d6434f2f9b86953ad` | `01a0b7a709ea5fcd24a95982f22c8150adaca4d5` |
| W_B125_S42 | `4df107c087aa3b4a6389121fc4965111c91f3e8ea3d1ef1870400e426405f4a0` | `d964649a6651c90580b342862694fb6d19fdc11d` |
| W_B250_S42 | `98fa0c08cf98e7e8065b538408719bdd4f616a1e8d2ca849d9f93ff838e747c3` | `81c721448cd129f8e659405a2f856d0169e036da` |
| GOC_B125_S42 | `fe326c55034f41c6bba61c63ee8570e9a9cbe20d7b0723175857329dbb40d903` | `07f062d2dc07b6ef682e60b1b46742658fd3f874` |
| GOC_B250_S42 | `13ed5e0b139d85b23941d3e46c4343fb065a39b883476016326ece9bab27c2bf` | `7688f2fd26463e9ef1e27b669eb6d3c86bab8b0c` |
| GOC_SYM_B125_S42 | `ef577b252de407876fae2c3899f37375ed823f9ca09ed7515a8cdb670f3d6e9c` | `c0fa794374eb20bec5f7c428130c8b1f83372783` |
| GOC_SYM_B250_S42 | `47f6f925be8558c265e95aa0b64065200dc4065542d23b6a6456feba215a53a9` | `dfc11d02c7fe17fccd8d699b46c406d9bb3af1fe` |
| GOC_INV_B125_S42 | `01d854ee523e6f0c5a3e6a0cbbb38f212d89ba69da463227a841966fdf796c7a` | `b07289d0987fe571b18d9909d1eba1fb6a07087a` |
| GOC_INV_B250_S42 | `2b453561eeff43e215befd18f06f2bb527a191fa6c9bef439dbd1cd379cd3a33` | `d63078cf2c799d4c1d3ac1a8dd135ec352c964d0` |

## Đối chiếu ngưỡng đã khóa

W_best trên validation là `W_B125_S42`, BLEU=17.706895. Chọn W_best từ đúng hai mức beta, không dùng test.

| GOC beta | Δ vs W_best | Δ vs W cùng beta | Δ vs SYM cùng beta | Δ vs INV cùng beta | Đạt cả3ngưỡng BLEU |
|---:|---:|---:|---:|---:|---|
| 1.250000 | 0.387549 | 0.387549 | -0.307765 | 1.873122 | False |
| 2.500000 | 0.102825 | 0.429499 | -0.704555 | 2.663167 | False |

Ngưỡng: GOC−W_best≥0.30 BLEU; GOC(beta)−SYM(beta)≥0.20; GOC(beta)>INV(beta). Không đổi ngưỡng hoặc mở sweep sau nhìn số. Đây là practical development gates, không phải kiểm định significance. Không trung bình các objectives/betas thành “9 seeds”.

## Recall ký tự Hán trên bản dịch validation

Đếm occurrences U+4E00..U+9FFF từ raw TRAIN targets. Mỗi câu, số khớp là sum_c min(count_prediction(c),count_reference(c)); chia tổng reference occurrences của bin. Rare<=10 gồm unseen0; mẫu số rare789, medium2253, frequent2930. Đây là ký tự Hán đầu ra, khác subword ID dùng đo gradient trong 0-step.

| ID | Rare matched/789 (recall) | Medium matched/2253 (recall) | Frequent matched/2930 (recall) |
|---|---|---|---|
| BASELINE_S42 | 78/789 (9.8859%) | 740/2253 (32.8451%) | 1781/2930 (60.7850%) |
| W_B125_S42 | 79/789 (10.0127%) | 778/2253 (34.5317%) | 1780/2930 (60.7509%) |
| W_B250_S42 | 83/789 (10.5196%) | 749/2253 (33.2446%) | 1787/2930 (60.9898%) |
| GOC_B125_S42 | 76/789 (9.6324%) | 759/2253 (33.6884%) | 1780/2930 (60.7509%) |
| GOC_B250_S42 | 74/789 (9.3790%) | 742/2253 (32.9339%) | 1784/2930 (60.8874%) |
| GOC_SYM_B125_S42 | 79/789 (10.0127%) | 763/2253 (33.8660%) | 1772/2930 (60.4778%) |
| GOC_SYM_B250_S42 | 83/789 (10.5196%) | 762/2253 (33.8216%) | 1788/2930 (61.0239%) |
| GOC_INV_B125_S42 | 71/789 (8.9987%) | 697/2253 (30.9365%) | 1742/2930 (59.4539%) |
| GOC_INV_B250_S42 | 49/789 (6.2104%) | 627/2253 (27.8296%) | 1749/2930 (59.6928%) |

## Tần suất ký tự khác W_best

Chỉ trên câu có prediction khác W_best: lấy multiset prediction_model−prediction_W và chiều ngược lại, rồi báo tần suất TRAIN trung bình/trung vị theo occurrences. Đây không phải alignment thay thế từ hoặc xác nhận lỗi ngữ nghĩa; paraphrase hợp lệ có thể khác single reference.

| ID | Câu khác W_best | Model-only occurrences | W-only occurrences | Mean freq model-only | Mean freq W-only | Δmean freq |
|---|---:|---:|---:|---:|---:|---:|
| BASELINE_S42 | 495 | 2263 | 2203 | 214.547945 | 225.116205 | -10.568260 |
| W_B125_S42 | 0 | 0 | 0 | NA | NA | NA |
| W_B250_S42 | 452 | 1886 | 1838 | 228.965005 | 219.609358 | 9.355647 |
| GOC_B125_S42 | 433 | 1770 | 1815 | 210.740113 | 238.293664 | -27.553551 |
| GOC_B250_S42 | 470 | 1956 | 1930 | 216.754601 | 200.340933 | 16.413669 |
| GOC_SYM_B125_S42 | 411 | 1631 | 1671 | 233.178418 | 236.517056 | -3.338638 |
| GOC_SYM_B250_S42 | 404 | 1502 | 1572 | 231.127164 | 222.031807 | 9.095357 |
| GOC_INV_B125_S42 | 479 | 1893 | 2071 | 258.477549 | 204.459198 | 54.018350 |
| GOC_INV_B250_S42 | 525 | 2385 | 2386 | 238.328302 | 206.538558 | 31.789744 |

## Epoch trajectories và giới hạn

CSV kèm đủ eval records từng epoch của từng run, epoch/step, loss/BLEU/chrF0/chrF++, selected flag. Số epoch thực tế theo stop-state; không bịa thêm đến20 nếu ES dừng trước. Không gọi last-microbatch loss_components là epoch mean.
Mọi prediction CSV đã kiểm576 dòng và source/target/order khớp original validation. Backup receipts verified ở pinned HF revision trước runner dọn weights. Các số RTX/A100 cũ không được trộn vào cohort này. 0-step logit-margin lợi hơn không tự chứng minh chất lượng hay grounding; chỉ so sánh pilot hiện tại với matched controls.

Files: `goc_pilot_a100_summary.json`, `goc_pilot_a100_epoch_trajectories.csv`; raw artifacts ở `outputs/experiments/amis_goc_pilot_a100/`. Không sửa docs/README.md, outputs_rebuttal/, tmp/ hoặc receipts lịch sử.
