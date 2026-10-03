# Chẩn đoán checkpoint CLRR — đo 02/10/2026, tổng hợp 03/10/2026

## Material Passport

- Trạng thái: hoàn tất đo và xác minh artifacts cho 12 checkpoint đã train; 0 optimizer steps.
- Nguồn weights: `FiveC/amis-rewire-checkpoints`, revision `63b0dbc4fab41a669ccb36ed2f22894868634b8a`. Archive, kích thước, SHA-256 và prediction đối chiếu của từng model nằm trong `protocol.json`.
- Máy đo: Colab A100 40 GB, shape Standard, không yêu cầu high-mem; Python conda `clrr`, Torch 2.6.0+cu124, Transformers 4.57.6. Phiên không còn hoạt động khi kiểm tra ngày 03/10.
- Dữ liệu: public GitHub `HeyDunaX/CLRR`, commit `db4c2322e11725db3aa6bc8cf4c4c812e11bf8e9`; các hàng CSV được kiểm tra trùng dữ liệu local. Không upload CSV local.
- Artifacts: [thư mục kết quả](../outputs_rebuttal/checkpoint_diagnostics_20261002/), [xác minh local](../outputs_rebuttal/checkpoint_diagnostics_20261002/local_verification.json), [bảng tổng hợp](../outputs_rebuttal/checkpoint_diagnostics_20261002/mechanism_summary.csv).
- Đã kiểm tra SHA-256 của archive 709,830,219 bytes và 113 file cloud; strict load weights, 96 batch gradient, chỉ số mẫu validation giống nhau giữa các method cùng dataset. 19 file lịch sử trong restoration manifest giữ nguyên SHA-256.
- Phạm vi: chẩn đoán quan sát tại checkpoint đã chọn; không phải đo tiến trình train, không phải can thiệp nhân quả, không phải chứng minh đa seed. Không thay BLEU/chrF hay sửa EXPERIMENTS/RESEARCH_INSIGHTS/TeX cũ.

## 1. Giao thức đo

**Residual:** hook trước và sau CLRR; kiểm tra `post = pre + 0.1 × skip`, distance 2, encoder. RMS tính trên vị trí hợp lệ và tất cả chiều hidden; ratio là RMS residual/RMS output lớp hiện tại trước rewiring. Bảng dưới là trung bình và cực đại qua các ví dụ nguồn và các lớp nhận skip, từ lớp 3 trở đi.

**Collapse:** toàn bộ 575 câu test Amis và 1,003 câu test Asháninka, nguồn và đích; eval, FP32, TF32 tắt. Mean pooling theo attention mask, gồm special tokens hợp lệ. SVD đầy đủ trên biểu diễn đã center, tính float64. Effective rank là `exp(H(s/sum(s)))`; khác effective rank của eigenvalues covariance. Giới hạn rank là `min(n−1,d)`. Raw mean pairwise cosine đo hướng chung; cosine cao không tự chứng minh collapse.

**Gradient:** 32 câu validation cố định, RNG 20261002, 8 batch × 4; train mode với dropout và seed 1000+batch_id, target stop-gradient giữ dropout train mode. BF16 autocast, parameters FP32, gradient chưa scale/clip; không optimizer. Dùng tensor CE gốc và graph auxiliary gốc (`total_loss − native_CE`) của wrapper cho Full/LSR-only. Ratio là trung bình ratio norm theo batch, không phải ratio của loss scalar. Nhóm encoder loại shared embeddings; các nhóm parameters không chồng lặp.

Baseline/CLRR-only không dùng LSR khi train: gradient auxiliary của chúng trong artifacts là **counterfactual λ=0.1**, không được diễn giải là lực tối ưu lịch sử. Không so absolute gradient norm giữa kiến trúc như một thước đo chất lượng.

## 2. Effective rank và mức tập trung biểu diễn

| Dataset/model | Method | Source e-rank | Rank limit | Target e-rank | Source pairwise cosine |
|---|---|---:|---:|---:|---:|
| Amis mBART | Baseline | 390.700 | 574 | 390.675 | 0.432 |
| Amis mBART | Full | 387.850 | 574 | 383.852 | 0.717 |
| Amis mBART | LSR-only | 388.194 | 574 | 381.721 | 0.707 |
| Amis NLLB | Baseline | 383.630 | 574 | 423.467 | 0.696 |
| Amis NLLB | Full | 387.674 | 574 | 411.582 | 0.646 |
| Amis mT5 | Baseline | 197.389 | 512 | 259.242 | 0.855 |
| Amis mT5 | Full | 222.676 | 512 | 256.212 | 0.836 |
| Amis mT5 | CLRR-only | 216.523 | 512 | 258.193 | 0.765 |
| Amis ByT5 | Baseline | 261.717 | 574 | 309.060 | 0.613 |
| Amis ByT5 | Full | 259.642 | 574 | 291.086 | 0.902 |
| Asháninka mBART | Baseline | 535.223 | 1002 | 600.183 | 0.621 |
| Asháninka mBART | Full, patience 4 | 535.568 | 1002 | 576.261 | 0.690 |

**Tốt:** các checkpoint Full giữ phổ nhiều chiều; không có dấu hiệu collapse về một hướng theo phép đo pooled này. NLLB Full tái lập chính xác hai số SVD cũ khi làm tròn: **387.67/411.58**. mT5 Full source rank tăng khoảng 12.8% so với baseline, CLRR-only cũng tăng: bằng chứng cơ chế đáng theo dõi cùng kết quả dịch mT5 đã audit.

**Chưa tốt:** Full không tăng effective rank nhất quán. mBART Full gần LSR-only, thấp hơn baseline; target rank Full thấp hơn baseline trên cả năm cặp dataset/backbone. ByT5 Full source cosine tăng mạnh 0.613→0.902, source rank hơi giảm. Asháninka source rank gần như không đổi. Không dùng các bảng này để claim “CLRR universally improves diversity”, “stop-gradient guarantees anti-collapse”, hay “preserves semantics”.

mT5 rank tăng là tương quan với kết quả dịch, chưa chứng minh tăng rank gây tăng chất lượng. So sánh target/source rank phải trong cùng backbone/protocol; không so rank thô giữa các hidden dimensions.

## 3. Residual norms và cân bằng CE–LSR

| Dataset/model, method | Residual/current mean | Max | Weighted LSR/CE gradient, toàn model | Encoder excl. shared | Mean gradient cosine | Batch cosine âm /8 |
|---|---:|---:|---:|---:|---:|---:|
| Amis mBART Full | 5.911% | 8.626% | 0.167% | 0.310% | 0.057 | 0 |
| Amis mBART LSR-only | 0% | 0% | 0.237% | 0.412% | 0.046 | 1 |
| Amis NLLB Full | 6.632% | 8.764% | 0.487% | 0.737% | 0.149 | 0 |
| Amis mT5 Full | 6.213% | 9.838% | 0.794% | 1.755% | 0.084 | 3 |
| Amis mT5 CLRR-only | 6.209% | 9.833% | inactive | inactive | counterfactual | counterfactual |
| Amis ByT5 Full | 4.338% | 9.305% | 1.184% | 1.795% | −0.124 | 5 |
| Asháninka mBART Full, p4 | 5.391% | 8.033% | 0.223% | 0.459% | 0.068 | 0 |

**Tốt:** trên các ví dụ/lớp đo, residual Full trung bình khoảng 4.3–6.6% của current output, cực đại dưới 10%; không thấy residual lấn át hoặc tăng vọt. Đây là mô tả checkpoint, không phải bảo đảm ổn định toàn bộ quá trình train hoặc mọi input.

**Cần điều tra:** weighted LSR gradient nhỏ so với CE tại checkpoint cuối được chọn. Điều này giúp định lượng λ=0.1 nhưng chưa đủ để tăng λ: không biết lực LSR ở giai đoạn đầu. ByT5 có conflict trên 5/8 batch, mean cosine âm; phù hợp với giả thuyết cần xem lại tương tác LSR/kiến trúc ở ByT5, chưa đủ để khẳng định đây là nguyên nhân giảm chất lượng. mT5 có 3/8 batch conflict dù mean cosine dương và chất lượng dịch tốt hơn; tránh claim LSR luôn đồng thuận CE.

Đo này thay thế việc dùng trace callback gradient bằng 0 sau `zero_grad` để suy luận. Nó không xác thực lại các trace lịch sử đó. Trong quá trình phát triển phép đo, CE tái dựng thủ công từ returned logits BF16 khác native CE tối đa 0.002555; **cả 12 checkpoint đã được đo lại bằng native objective**. Bản đầu giữ riêng `gradients_initial_reconstruction.json`, không dùng cho bảng cuối. Sai khác scalar auxiliary tái dựng/gốc tối đa khoảng 2.35e−7; gradient chính lấy từ graph gốc.

## 4. Đối chiếu generation: còn giới hạn tái lập

8 câu sentinel cố định/model; beams 4, max length theo manifest; so text với prediction lịch sử bất biến. Đây là kiểm tra mẫu nhỏ, không phải full-test re-evaluation.

| Checkpoint | BF16 khớp /8 | FP32 khớp /8 |
|---|---:|---:|
| mBART baseline | 8 | 8 |
| mBART Full | 8 | 8 |
| mBART LSR-only | 6 | 7 |
| NLLB baseline | 6 | 6 |
| NLLB Full | 8 | 8 |
| mT5 baseline | 6 | 8 |
| mT5 Full | 7 | 8 |
| mT5 CLRR-only | 7 | 8 |
| ByT5 baseline | 8 | 8 |
| ByT5 Full | 7 | 8 |
| Asháninka mBART baseline | 8 | 4 |
| Asháninka mBART Full p4 | 4 | 7 |

Precision thay đổi có thể đổi beam search: trực tiếp thấy ở mT5/ByT5/Asháninka. Tuy nhiên, **FP32 không giải quyết hết mismatch** của mBART LSR-only, NLLB baseline và Asháninka Full. Không kết luận “đúng hoàn toàn predictions lịch sử” hoặc “sai checkpoint” chỉ từ sentinel. Chi tiết câu thay đổi có trong `sentinel_comparison.csv`; archive SHA/strict loading/tokenizer vocab đã kiểm tra.

Code review cho thấy wrapper `.generate` gọi base model, có khả năng bypass autocast bọc root forward của Accelerate, trong khi baseline đi qua root. Đây là **giả thuyết về pipeline**, chưa xác minh precision thực thi của từng historical run. Trước thí nghiệm mới phải đặt cùng generation precision rõ ràng cho tất cả method, cùng tokenizer/language codes/beams/max length và lưu environment. Không thay điểm lịch sử từ 8 câu sentinel này. Muốn sửa điểm checkpoint cần chạy lại toàn test cùng protocol rồi lưu phiên bản kết quả mới.

## 5. Checkpoint loading và phần chưa đo

- Wrapper ZIP thiếu config được phục hồi từ **paired trained baseline archive**, ghi source/member/SHA trong provenance; strict loading không có missing/unexpected/mismatched weights.
- NLLB tokenizer cũ lưu `extra_special_tokens` dạng list: chuyển cách truyền tham số cho Transformers 4.57.6; kiểm tra toàn bộ vocab và token IDs không đổi. Không áp dụng thay regex tokenizer âm thầm.
- Giữ saved language codes: Amis mBART `tl_XX→zh_CN`, NLLB `zho_Hant→zho_Hant`, Asháninka mBART `es_XX→es_XX`. Đây là cấu hình lịch sử để tái đo, không phải language codes cho Turkish.
- **mBART CLRR-only best checkpoint 504 chưa có trong revision HF đã kiểm tra** (chỉ tìm được đến 432). Không dùng checkpoint 432 thay thế. Vì vậy chưa đủ 4 thành phần mBART cho phân tích cơ chế.
- Asháninka Full trong báo cáo là checkpoint patience 4 theo manifest; **không phải** run patience 10/epoch 20. Không gán kết quả cơ chế này cho epoch 20.
- Không có đo no-stop-gradient trên checkpoint đã train tương ứng; không suy luận nhân quả về stop-gradient. Không đo multi-seed, tốc độ/peak memory training hay Turkish trong lượt này.

## 6. Các kết luận dùng được và bước kế tiếp

1. **Bảo vệ được ở phạm vi đo:** CLRR residual có độ lớn vừa phải tại saved checkpoints; pooled representations chưa collapse về một hướng; gradient CE và weighted LSR đo được bằng objective thực tế. NLLB SVD cũ nay có artifacts để tái lập.
2. **Insight có ích:** mT5 có source effective rank tăng ở cả CLRR-only và Full; ByT5 có anisotropy tăng và CE–LSR conflict. Có thể báo cáo cơ chế phụ thuộc backbone, với giới hạn quan sát rõ ràng.
3. **Không bảo vệ được claim mạnh:** ưu thế diversity phổ quát, stop-gradient bảo đảm chống collapse, semantic preservation, CE–LSR luôn đồng thuận, hay ổn định xuyên suốt training. Residual nhỏ/không collapse không chứng minh BLEU tốt hơn.
4. **Trước Turkish:** thống nhất generation precision và ghi provenance; xử lý overlap/duplicate đã phát hiện ở dataset trong kế hoạch riêng; giữ budget 20k và dùng language codes đúng. Pilot baseline/CLRR-only/LSR-only/Full với training/selection/evaluation protocol bằng nhau, sau đó mới quyết định mở rộng seed và λ.
5. **Chưa cần sửa phương pháp ngay:** thiếu bằng chứng theo thời gian và can thiệp. Nếu cần cải thiện LSR, nên đo gradient tại các checkpoint sớm/giữa/cuối và kiểm tra ablation λ/LSR trên validation theo kế hoạch định trước; không chọn cấu hình bằng test và không tăng λ chỉ vì norm ở checkpoint cuối nhỏ.

## Tái lập và kết quả riêng

- Script đo: `scripts/revalidation/measure_checkpoint_mechanisms.py`; bản thực thi và SHA trong outputs.
- Tổng hợp local trong conda `clrr`: `python scripts/revalidation/summarize_checkpoint_mechanisms.py`.
- Raw: pooled representations từng lớp `.npz`, residual từng ví dụ `.csv`, full spectra `collapse.json`, native gradients `gradients.json`, loading provenance, BF16/FP32 sentinels, frozen requirements và logs.
- SHA archive: `66a9eb8b9ed530970476a0399a20ba66d5cf81f3457c02b2383ffaf61b9efac4`.
- Tất cả số mới ở thư mục/report riêng; không train Turkish và không sửa số cũ trong lượt này.
