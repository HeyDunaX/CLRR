# CLRR — Experiments tổng hợp theo bố cục paper

**Snapshot: 06/10/2026, Asia/Saigon.** Main table dành cho đối sánh phương pháp trên **mBART-50 và NLLB-200**. Các bảng seed, ablation, tuning, efficiency, cơ chế và kiểm toán đặt trong appendix. Chỉ sửa file tổng hợp mới này; các tài liệu và artifacts nguồn giữ nguyên.

## Main Table 1 — Translation quality trên hai backbone chính

**Cặp ngôn ngữ:** Amis→Mandarin, test575câu. BLEU dùng tokenizer zh;chrF++ native raw references. **CLRR+LSR** là nhánh Full. Mỗi cấu hình dùng checkpoint được chọn bằng validation; “tốt nhất” nghĩa best theo validation trong phạm vi cấu hình đã chạy, không lấy max test hoặc chọn seed cao nhất. Corrected LSR và Full giữ λ=.10; encoder routing α=.10,d=2.

| Method | mBART-50 BLEU ↑ | mBART-50 chrF++ ↑ | NLLB-200 BLEU ↑ | NLLB-200 chrF++ ↑ |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 19.3886 ± 1.1203 | 16.9249 ± 3.2626 | 13.6186 | 10.4990 |
| Middle-Layer Alignment | 19.2893 | 18.4378 | 13.9648 | 10.6596 |
| BitFit | 0.3463 | 2.2552 | 1.0750 | 2.9643 |
| Narrow LoRA | 3.2365 | 4.4213 | 3.5383 | 4.9407 |
| Strong LoRA A — all-linear | 10.2296 | 8.6319 | 11.4625 | 9.5004 |
| Strong LoRA B — all-linear + embeddings | 13.8066 | 10.3293 | 9.8024 | 8.4010 |
| CLRR-only (ours) | 19.3539 ± 0.1351 | 18.4229 ± 0.0947 | — | — |
| LSR-only (ours) | 19.3930 ± 0.3787 | 18.4626 ± 0.1860 | — | — |
| CLRR + LSR (ours) | **20.1303 ± 0.2188** | **18.8570 ± 0.1575** | **14.2835** | **10.9179** |

**Chú thích quan trọng khi dùng bảng:**

- Bốn nhánh mBART Baseline/CLRR/LSR/CLRR+LSR là **corrected seeds42–44, mean ± sample SD**. Nguồn: [four-way CSV](../../outputs_rebuttal/amis_corrected_four_way_20261005/four_way_scores.csv).
- Middle/PEFT của mBART và các điểm NLLB hiện là **single-seed historical records**, không có SD multiseed tương ứng. Nguồn: [ledger28outputs](EXPERIMENTS_UPDATED.md). Baseline/Full NLLB dùng lượt FP32 B; Middle/PEFT dùng audit A.
- Đây là **bố cục main table và inventory số hiện có**, chưa phải toàn bộ bảng đã matched về training recipe/inference. Khi hoàn thiện paper cần audit/re-evaluate hoặc bổ sung các đối sánh bị ảnh hưởng; không diễn giải chênh mean3seeds−single-run như paired significance. Chi tiết nguồn/cohort ở Appendix A/H/K.
- Dấu **—**: chưa có điểm NLLB CLRR-only/LSR-only được chứng nhận trong nguồn tổng hợp; không thay bằng0 và không lấy nhánh decoder làm encoder-only.
- In đậm thể hiện nhánh tổ hợp đề xuất, **không phải dấu statistical significance**. Không lấy score cao nhất riêng cho BLEU vàchrF++ từ hai checkpoint khác nhau.
- Strong LoRA A vàB giữ riêng để thấy cả all-linear và embedding-unfreeze. Nếu muốn chỉ một dòng Strong LoRA, phải chọn bằng validation với tuning budget công bố; không chọn A/B theo test. CLRR-only/LSR-only vẫn ở main theo yêu cầu tác giả; bảng ablation chi tiết ở Appendix C.

## Thiết kế các bảng còn lại khi viết paper

| Vị trí | Bảng | Nội dung hiện có |
| --- | --- | --- |
| **Main Table1** | Two-backbone method comparison | Baseline,Middle,BitFit,Narrow/Strong LoRA,CLRR,LSR,CLRR+LSR |
| **Appendix B** | Dataset/recipe | Split,định nghĩa token,training/selection/inference |
| **Appendix C–D** | Component ablation/seed robustness | Corrected3seeds,mean±SD,paired effects,CI,p vàinteraction |
| **Appendix E/I** | Hyperparameters và controls | Validation LSRgrid,screening,attached,warmup |
| **Appendix F** | Computational cost | Parameters,train/inference latency,throughput,VRAM |
| **Appendix H** | Mechanism/extended results | mT5/ByT5,rank,residual,gradient,route-on/off,metric sensitivity |
| **Appendix G/J/K** | Reproducibility/evidence | Gates,provenance,nguồn số vàpending; chọn phần phù hợp để đưa vào paper |

Các appendix dưới đây là kho bảng đầy đủ để tác giả chọn, không yêu cầu đưa toàn bộ50bảng vào bản nộp. Số gốc/negative findings được giữ đúngscope; không tự chuyển các probes thành efficacy claims.

## Appendix A — Phạm vi bằng chứng


| Nhóm | Nội dung | Phạm vi / sử dụng |
| --- | --- | --- |
| Corrected main | Baseline/CLRR-only/LSR-only/Full, seeds42–44 | Main ablation cùng corrected recipe; baseline42 fresh,43/44 audited reuse |
| Corrected tuning | LSR λ=.03/.10/.30, seeds42–44, validation | Chọn λ=.10; .03/.30 test blind, không đếm thành hai tuned/matching families |
| Benchmark v2 | 4 corrected seed42 checkpoints trên A100 | Efficiency table; 5 timings cùng session, không multiseed significance |
| Historical A | Raw-reference audit prediction outputs | Backbone/PEFT records; Holm42, không matched corrected training |
| Historical B | FP32 reevaluation saved checkpoints | Same inference pass; Holm36, không sửa training fairness lịch sử |
| Historical C/D | 3-seed controls, tuning, attached, warmup trước fix | Historical ablations; không thay corrected controls |
| Historical V4/V5 | Routing interventions, rank, CE–LSR gradients | Characterization; cùng weights on/off không phải independent training ablation |
| Excluded scope | Asháninka/LayerSkip | Không đưa vào bảng paper mới |
| Planned | Fresh seeds45–52,32models; Middle matched; Turkish | Không có completed quality result trong snapshot; không ghi thành số đã chạy |


**Thứ tự tin cậy:** raw score/decision JSON và CSV có receipt → báo cáo cùng cohort → prose trong docs. Không ghép A/B/corrected để tạo p-value mới. Metric chrF++ raw khác chrF0; không dùng chrF++ sau external Chinese tokenization thay native chrF++. Điểm cao nhất không đồng nghĩa statistically significant. Những bảng lịch sử bên dưới là kết quả của checkpoint lịch sử, không tự bị vô hiệu chỉ vì source sau này được sửa.


## Appendix B — Dataset và protocol


| Hạng mục | Giá trị |
| --- | --- |
| Dataset | Amis→Mandarin; data_processed/amis_mandarin/{train,validation,test}.csv; source,target |
| Split | 4600 /576 /575 sentence pairs |
| Whitespace source tokens | 31136 /3947 /3863; không model subwords |
| BMP Han target characters | 45438 /5972 /5643; định nghĩa BMP Han, không mọi codepoints |
| Backbone main | facebook/mbart-large-50-many-to-many-mmt;610879488 parameters |
| Model revision | e30b6cb8eb0d43a0b73cab73c7676b9863223a30 |
| Language codes | tl_XX source proxy; zh_CN target |
| Corrected training | LR5e-5; AdamW fused;weight decay0;warmup ratio.06;batch32×accum4;eval8;max lengths256;BF16+gradient checkpointing |
| Budget / selection | Max20epochs;early stopping patience4;validation raw chrF++ greedy beam1 selects best |
| Full | Encoder routing d2 α.10, detached source, routing warmup0;LSR λ.10 with detached target |
| Corrected aggregation | Average microbatch means; not necessarily global token-weighted CE |
| Frozen test | 575 raw-reference rows;beam4;FP32;TF32off;length penalty1 |
| BLEU | SacreBLEU2.6.0;Chinese tokenizer zh |
| chrF++ / chrF0 | Native CHRF char_order6;word_order2 /0;beta2;raw references |
| Mean ± SD | Sample standard deviation over training seeds, not CI |
| Local execution | Conda clrr; this consolidation only reads artifacts |


train CSV SHA256: `4e13d83f035a18c67e1b2e3990baff3aa92154a34b754a4d0cdef9a71d85b8e6`.


validation CSV SHA256: `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4`.


test CSV SHA256: `2e3cea3f9011ef6ee237a437b57a1ac21bd9fae9682473bce00c99a80afde0aa`.


## Appendix C — Component ablation và từng seed corrected


| Family | n seeds | BLEU mean ± SD | chrF++ mean ± SD | chrF0 mean ± SD |
| --- | --- | --- | --- | --- |
| Baseline | 3 | 19.3886 ± 1.1203 | 16.9249 ± 3.2626 | 18.0452 ± 0.8362 |
| CLRR-only α=.10 | 3 | 19.3539 ± 0.1351 | 18.4229 ± 0.0947 | 18.0312 ± 0.1645 |
| LSR-only λ=.10 | 3 | 19.3930 ± 0.3787 | 18.4626 ± 0.1860 | 18.1098 ± 0.2712 |
| Full α=.10/λ=.10 | 3 | 20.1303 ± 0.2188 | 18.8570 ± 0.1575 | 18.6708 ± 0.2341 |


| Seed | Family | BLEU | chrF++ | chrF0 |
| --- | --- | --- | --- | --- |
| 42 | Baseline | 19.728081 | 18.713838 | 18.397282 |
| 42 | CLRR-only α=.10 | 19.223095 | 18.362492 | 17.904266 |
| 42 | LSR-only λ=.10 | 19.095440 | 18.434505 | 17.998477 |
| 42 | Full α=.10/λ=.10 | 20.321295 | 19.014707 | 18.913663 |
| 43 | Baseline | 18.137832 | 13.159189 | 17.090430 |
| 43 | CLRR-only α=.10 | 19.492997 | 18.531956 | 18.217028 |
| 43 | LSR-only λ=.10 | 19.819251 | 18.660999 | 18.418971 |
| 43 | Full α=.10/λ=.10 | 20.177995 | 18.856774 | 18.652019 |
| 44 | Baseline | 20.300016 | 18.901673 | 18.647752 |
| 44 | CLRR-only α=.10 | 19.345498 | 18.374130 | 17.972397 |
| 44 | LSR-only λ=.10 | 19.264374 | 18.292201 | 17.911947 |
| 44 | Full α=.10/λ=.10 | 19.891624 | 18.699629 | 18.446629 |


Nguồn: [four_way_scores.csv](../../outputs_rebuttal/amis_corrected_four_way_20261005/four_way_scores.csv). SHA256: `5df1c0af18e583ad2100b854b5a0a66d8d02fc82c80dd64799122aedcfa6d2d7`.


**Sai khác tài liệu được phát hiện:** bảng bàn giao hiển thị Baseline seed42 chrF0=18.3980, seed43=17.0864; CSV four-way ghi18.397282 và17.090430. File này dùng CSV, không sửa bàn giao. Đây là lệch giữa nguồn tài liệu, chưa phải kết luận prediction hoặc metric implementation sai. Cần trích số paper từ đúng một CSV và definition, không ghép các chữ số từ prose.


## Appendix D — Paired effects, interaction và significance


| Contrast | Metric | Seed42 | Seed43 | Seed44 | Mean Δ | Wins /3 |
| --- | --- | --- | --- | --- | --- | --- |
| Full α=.10/λ=.10 − LSR-only λ=.10 | bleu | +1.225855 | +0.358744 | +0.627250 | +0.737283 | 3 |
| Full α=.10/λ=.10 − LSR-only λ=.10 | chrfpp | +0.580202 | +0.195775 | +0.407428 | +0.394468 | 3 |
| Full α=.10/λ=.10 − LSR-only λ=.10 | chrf0 | +0.915187 | +0.233048 | +0.534683 | +0.560972 | 3 |
| Full α=.10/λ=.10 − Baseline | bleu | +0.593214 | +2.040163 | -0.408392 | +0.741662 | 2 |
| Full α=.10/λ=.10 − Baseline | chrfpp | +0.300869 | +5.697585 | -0.202043 | +1.932137 | 2 |
| Full α=.10/λ=.10 − Baseline | chrf0 | +0.516381 | +1.561589 | -0.201122 | +0.625616 | 2 |
| Full α=.10/λ=.10 − CLRR-only α=.10 | bleu | +1.098200 | +0.684998 | +0.546127 | +0.776441 | 3 |
| Full α=.10/λ=.10 − CLRR-only α=.10 | chrfpp | +0.652215 | +0.324818 | +0.325499 | +0.434177 | 3 |
| Full α=.10/λ=.10 − CLRR-only α=.10 | chrf0 | +1.009397 | +0.434991 | +0.474232 | +0.639540 | 3 |
| CLRR-only α=.10 − Baseline | bleu | -0.504986 | +1.355166 | -0.954518 | -0.034780 | 1 |
| CLRR-only α=.10 − Baseline | chrfpp | -0.351346 | +5.372767 | -0.527543 | +1.497959 | 1 |
| CLRR-only α=.10 − Baseline | chrf0 | -0.493016 | +1.126598 | -0.675354 | -0.013924 | 1 |
| LSR-only λ=.10 − Baseline | bleu | -0.632641 | +1.681419 | -1.035642 | +0.004378 | 1 |
| LSR-only λ=.10 − Baseline | chrfpp | -0.279333 | +5.501810 | -0.609472 | +1.537669 | 1 |
| LSR-only λ=.10 − Baseline | chrf0 | -0.398805 | +1.328541 | -0.735805 | +0.064644 | 1 |


| Interaction Full−LSR−CLRR+Baseline | Seed42 | Seed43 | Seed44 | Mean |
| --- | --- | --- | --- | --- |
| bleu | +1.730841 | -0.996421 | +1.581769 | +0.772063 |
| chrfpp | +0.931547 | -5.176992 | +0.934971 | -1.103491 |
| chrf0 | +1.408202 | -0.893550 | +1.210037 | +0.574896 |


Interaction là descriptive metric contrast, không chứng minh nhân quả cơ chế hoặc synergy có ý nghĩa thống kê. Các effect/means ở trên tổng hợp từ CSV; CI/p bên dưới trích bảng analysis cùng cohort.


### T07 — Four-Way & 2x2 Factorial Interaction Analysis (Corrected Controls) → 2. Key Paired Contrasts & 2x2 Factorial Interaction

**Phạm vi:** Corrected test; CI conditional sentence bootstrap giữ cố định3seeds; sign-flip theo3paired seeds. Nguồn: [FOUR_WAY_ANALYSIS.md, dòng20](../../outputs_rebuttal/amis_corrected_four_way_20261005/FOUR_WAY_ANALYSIS.md).


| Contrast | Metric | Seed 42 | Seed 43 | Seed 44 | Mean Delta | Conditional Sentence 95% CI | Sign-flip p (n=3) |
|---|---|---:|---:|---:|---:|---|---:|
| Full_minus_LSR | BLEU | +1.2259 | +0.3587 | +0.6273 | **+0.7373** | [+0.0403, +1.4557] | 0.25 |
| Full_minus_LSR | chrF++ | +0.5802 | +0.1958 | +0.4074 | **+0.3945** | [-0.0392, +0.8657] | 0.25 |
| Full_minus_LSR | chrF0 | +0.9152 | +0.2330 | +0.5347 | **+0.5610** | [+0.0209, +1.1149] | 0.25 |
| Full_minus_Baseline | BLEU | +0.5932 | +2.0402 | -0.4084 | **+0.7417** | [-0.0265, +1.5182] | 0.50 |
| Full_minus_Baseline | chrF++ | +0.3009 | +5.6976 | -0.2020 | **+1.9321** | [+0.0084, +4.7936] | 0.50 |
| Full_minus_Baseline | chrF0 | +0.5164 | +1.5616 | -0.2011 | **+0.6256** | [+0.0364, +1.2083] | 0.50 |
| Full_minus_CLRR | BLEU | +1.0982 | +0.6850 | +0.5461 | **+0.7764** | [+0.0741, +1.6307] | 0.25 |
| Full_minus_CLRR | chrF++ | +0.6522 | +0.3248 | +0.3255 | **+0.4342** | [-0.0314, +0.9322] | 0.25 |
| Full_minus_CLRR | chrF0 | +1.0094 | +0.4350 | +0.4742 | **+0.6395** | [+0.0754, +1.2125] | 0.25 |
| CLRR_minus_Baseline | BLEU | -0.5050 | +1.3552 | -0.9545 | **-0.0348** | [-0.8355, +0.6013] | 1.00 |
| CLRR_minus_Baseline | chrF++ | -0.3513 | +5.3728 | -0.5275 | **+1.4980** | [-0.3987, +4.3143] | 1.00 |
| CLRR_minus_Baseline | chrF0 | -0.4930 | +1.1266 | -0.6754 | **-0.0139** | [-0.5118, +0.4980] | 1.00 |
| LSR_minus_Baseline | BLEU | -0.6326 | +1.6814 | -1.0356 | **+0.0044** | [-0.7652, +0.7566] | 1.00 |
| LSR_minus_Baseline | chrF++ | -0.2793 | +5.5018 | -0.6095 | **+1.5377** | [-0.4348, +4.3850] | 1.00 |
| LSR_minus_Baseline | chrF0 | -0.3988 | +1.3285 | -0.7358 | **+0.0646** | [-0.5262, +0.6607] | 1.00 |
| Interaction_Synergy | BLEU | +1.7308 | -0.9964 | +1.5818 | **+0.7721** | [-0.1480, +1.8452] | 0.50 |
| Interaction_Synergy | chrF++ | +0.9315 | -5.1770 | +0.9350 | **-1.1035** | [-3.9680, +0.9927] | 1.00 |
| Interaction_Synergy | chrF0 | +1.4082 | -0.8936 | +1.2100 | **+0.5749** | [-0.1628, +1.3149] | 0.50 |


CI conditional không bao gồm training randomness. Với n3, minimum two-sided exact sign-flip p=.25. Mean BLEU Full−LSR positive3/3 là tín hiệu để confirm, chưa robust superiority. Các diễn giải mạnh trong report nguồn không được tự chuyển thành claim paper.


## Appendix E — Validation tuning LSR


| λ | Seed42 validation chrF++ | Seed43 | Seed44 | Mean | Decision |
| --- | --- | --- | --- | --- | --- |
| 0.03 | 13.906081 | 14.066061 | 14.178781 | 14.050307 | Not selected; test blind |
| 0.10 | 14.190174 | 14.274377 | 14.201443 | 14.221998 | Selected |
| 0.30 | 14.529689 | 14.216064 | 13.705969 | 14.150574 | Not selected; test blind |


| LSR .30 seed | Validation BLEU | Validation chrF++ | Validation chrF0 | Best step |
| --- | --- | --- | --- | --- |
| 42 | 17.398041 | 14.529689 | 16.725192 | 720 |
| 43 | 16.887742 | 14.216064 | 16.301792 | 648 |
| 44 | 15.855267 | 13.705969 | 15.791632 | 432 |


Nguồn: [lsr_grid_decision.json](../../outputs_rebuttal/amis_lsr_l030_validation_20261005/lsr_grid_decision.json). SHA256: `a9f613bdaa6112fd0d444a22757529b755d367d5445e045578eb7ce91066ffd9`.


λ=.10 là best observed trong3giá trị, không optimum toàn cục. Không mở test .03/.30 chỉ để thêm bảng. Collection receipt .30 liệt kê paths/sizes; không tự đồng nghĩa full archive SHA verification.


## Appendix F — Efficiency benchmark v2


| Family | Total/trainable | Train ms/macrostep ± SD | Train overhead vs Baseline | Train overhead vs LSR | Beam4 ms/sentence ± SD | Beam4 valid tokens/s | Train peak allocated MiB | Train peak reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 610,879,488 / 610,879,488 | 566.23 ± 23.44 | +0.00% | -9.75% | 32.274 ± 0.497 | 282.79 | 12872.96 | 15036.00 |
| CLRR-only α=.10 | 610,879,488 / 610,879,488 | 575.66 ± 18.96 | +1.67% | -8.25% | 29.312 ± 0.729 | 305.59 | 12872.96 | 15056.00 |
| LSR-only λ=.10 | 610,879,488 / 610,879,488 | 627.43 ± 1.47 | +10.81% | +0.00% | 29.165 ± 0.218 | 305.39 | 12989.39 | 15056.00 |
| Full α=.10/λ=.10 | 610,879,488 / 610,879,488 | 632.47 ± 3.36 | +11.70% | +0.80% | 31.715 ± 0.250 | 294.14 | 13007.50 | 15056.00 |


| Family | Beam4 inference overhead vs Baseline | Beam4 overhead vs LSR | Beam1 supplementary ms/sentence | Inference peak allocated MiB beam4 |
| --- | --- | --- | --- | --- |
| Baseline | +0.00% | +10.66% | 26.422 | 2659.67 |
| CLRR-only α=.10 | -9.18% | +0.50% | 25.862 | 2645.92 |
| LSR-only λ=.10 | -9.63% | +0.00% | 26.078 | 2629.17 |
| Full α=.10/λ=.10 | -1.73% | +8.74% | 27.637 | 2663.17 |


Nguồn: [cost_benchmark_v2.json](../../outputs_rebuttal/amis_cost_benchmark_20261006_v2/cost_benchmark_v2.json). SHA256: `b3bb4d582c740e3fee8a217c306b0afb1958a4a6e9b76d8964bac976e074912c`.


A100-SXM4-40GB;train128 fixed rows,4microbatches×32,3warmups+5timings;validation64rows,batch8,2warmups+5timings. BF16+GC/fusedAdamW/WD0/maxnorm1 training; sạch weights reload trước inference FP32/TF32off/use_cacheFalse. Beam4 max_length256 primary;beam1 max_new_tokens64 supplementary. Valid token throughput loại pad/special tokens. Five timings là within-session samples; generated lengths có thể khác model, không suy inference speedup thuần kiến trúc. VRAM khoảng12.57–12.70GiB allocated, không cam kết peak cho mọi batch. Zero added parameters không đồng nghĩa PEFT hoặc zero overhead.


## Appendix G — Correctness và provenance


| Kiểm tra | Observed receipt | Phạm vi |
| --- | --- | --- |
| GPU optimization matrix | 16/16 PASS;max error 0.0 | Tiny BF16 fixtures;SGD/AdamW,GC on/off,unequal labels,partialgroup |
| Actual CLI lifecycle | 4/4 PASS;best weights exact | Train/save/reload;validation-only;FP32 generation |
| Frozen bundle | 52fcd0fc2881d414a8ea1ab0e21ca1961eec74cd32b01858fda5c77a3d64bc3c | 32 files;source/bundle matched in preceding audit |
| Baseline42 | Fresh selected step720/epoch20 | Fresh selection provenance, not retroactive proof for old archive |
| Baseline43 /44 | Selected steps720 /612 | Audited reuse;old inventory252/504 wrong metadata, not automatically wrong BLEU |
| Accumulation fix | accepts_loss_kwargs=False | Fixed wrappers;normupdate factor4 reproduction is not AdamW LR×4 |
| Full-size smoke historical | 4branches;stress256allocated≈33.45–33.48GiB | Different input shapes from benchmark;not contradiction with12.7GiB microbenchmark |


Nguồn: [optimization matrix](../../outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate/optimization_matrix.json), [CLI lifecycle](../../outputs_rebuttal/amis_fresh_confirmation_20261005/gpu_gate/rerun_lifecycle_smoke.json), [bundle manifest](../../scratch/fresh_confirmation_manifest.json), [technical audit](../archive/TECHNICAL_AUDITS_20261005.md).


## Appendix H — Backbones bổ sung và diagnostics lịch sử

**Tất cả bảng trong Appendix H là historical.** Dữ liệu/metrics có thể verified cho checkpoint đó nhưng training trước accumulation fix không làm matched corrected controls. Không trộn validation với test hoặc single-seed với mean3seeds. Số trích giữ theo nguồn; nguồn có nhiều version phải đọc label A/B/C/V4/V5. Xóa scope Asháninka/LayerSkip khỏi bảng tổng hợp, giữ source cũ nguyên trạng.


### T13 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 2. Toàn bộ điểm hiện hành — 28 outputs Amis

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng226](EXPERIMENTS_UPDATED.md).


| Dataset | Backbone | Method / checkpoint | N | BLEU | chrF++ | Nguồn |
| --- | --- | --- | --- | --- | --- | --- |
| amis_mandarin | mBART | Baseline | 575 | 19.6144 | 14.0547 | B — generate FP32 03/10 |
| amis_mandarin | mBART | CLRR+LSR | 575 | 20.3896 | 19.0824 | B — generate FP32 03/10 |
| amis_mandarin | mBART | CLRR-only | 575 | 18.4697 | 13.2443 | A — audit predictions 02/10 |
| amis_mandarin | mBART | LSR-only | 575 | 19.7067 | 18.6169 | B — generate FP32 03/10 |
| amis_mandarin | mBART | CLRR-Dec+LSR | 575 | 19.9188 | 18.6999 | A — audit predictions 02/10 |
| amis_mandarin | mBART | BitFit | 575 | 0.3463 | 2.2552 | A — audit predictions 02/10 |
| amis_mandarin | mBART | Narrow LoRA | 575 | 3.2365 | 4.4213 | A — audit predictions 02/10 |
| amis_mandarin | mBART | Strong LoRA A | 575 | 10.2296 | 8.6319 | A — audit predictions 02/10 |
| amis_mandarin | mBART | Strong LoRA B | 575 | 13.8066 | 10.3293 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Strong LoRA A | 575 | 11.4625 | 9.5004 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Strong LoRA B | 575 | 9.8024 | 8.4010 | A — audit predictions 02/10 |
| amis_mandarin | mBART | Middle-Layer Alignment | 575 | 19.2893 | 18.4378 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Baseline | 575 | 13.6186 | 10.4990 | B — generate FP32 03/10 |
| amis_mandarin | NLLB | CLRR+LSR | 575 | 14.2835 | 10.9179 | B — generate FP32 03/10 |
| amis_mandarin | NLLB | CLRR-Dec+LSR | 575 | 13.4194 | 10.4559 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Middle-Layer Alignment | 575 | 13.9648 | 10.6596 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | BitFit | 575 | 1.0750 | 2.9643 | A — audit predictions 02/10 |
| amis_mandarin | NLLB | Narrow LoRA | 575 | 3.5383 | 4.9407 | A — audit predictions 02/10 |
| amis_mandarin | mT5 | Baseline | 575 | 2.7887 | 3.8129 | B — generate FP32 03/10 |
| amis_mandarin | mT5 | CLRR-only | 575 | 4.4393 | 5.1761 | B — generate FP32 03/10 |
| amis_mandarin | mT5 | CLRR+LSR | 575 | 4.5961 | 5.0425 | B — generate FP32 03/10 |
| amis_mandarin | mT5 | CLRR-Dec+LSR | 575 | 4.8315 | 5.1933 | A — audit predictions 02/10 |
| amis_mandarin | mT5 | CLRR-Both+LSR | 575 | 3.4459 | 4.5753 | A — audit predictions 02/10 |
| amis_mandarin | mT5 | Middle-Layer Alignment | 575 | 4.6625 | 5.2103 | A — audit predictions 02/10 |
| amis_mandarin | ByT5 | Baseline | 575 | 7.4856 | 8.2977 | B — generate FP32 03/10 |
| amis_mandarin | ByT5 | CLRR+LSR | 575 | 7.3160 | 8.0966 | B — generate FP32 03/10 |
| amis_mandarin | ByT5 | CLRR-Dec+LSR | 575 | 7.4394 | 8.0630 | A — audit predictions 02/10 |
| amis_mandarin | ByT5 | Middle-Layer Alignment | 575 | 7.5660 | 8.3414 | A — audit predictions 02/10 |


### T14 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 3. Kết quả quality mới nhất — cùng lượt FP32 → 4. Kiểm định quality

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng265](EXPERIMENTS_UPDATED.md).


| So sánh BLEU | Δ | CI95% | Holm p |
|---|---:|---|---:|
| mBART Full−Baseline | +0.7752 | [−0.4388,2.0441] | 1.0000 |
| mBART Full−LSR-only | +0.6830 | [−0.4806,1.8530] | 1.0000 |
| NLLB Full−Baseline | +0.6650 | [−0.3295,1.7156] | 1.0000 |
| mT5 Full−Baseline | +1.8074 | [1.1826,2.4746] | 0.0036 |
| mT5 Full−CLRR-only | +0.1567 | [−0.3769,0.7116] | 1.0000 |
| ByT5 Full−Baseline | −0.1696 | [−0.8869,0.5266] | 1.0000 |


### T15 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 3. Kết quả quality mới nhất — cùng lượt FP32 → 5. Routing-off trên cùng trained Full weights

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng281](EXPERIMENTS_UPDATED.md).


| Model | Full BLEU | Routing-off BLEU | On−off | CI95% | Holm p |
|---|---:|---:|---:|---|---:|
| mBART Amis | 20.3896 | 16.4609 | +3.9287 | [2.7175,5.1948] | 0.0036 |
| NLLB Amis | 14.2835 | 12.9177 | +1.3659 | [0.3337,2.7009] | 0.4080 |
| mT5 Amis | 4.5961 | 4.1384 | +0.4577 | [0.0719,0.8816] | 0.4140 |
| ByT5 Amis | 7.3160 | 7.5166 | −0.2006 | [−0.7983,0.3788] | 1.0000 |


### T16 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 3. Kết quả quality mới nhất — cùng lượt FP32 → Thay đổi predictions từ A sang B

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng297](EXPERIMENTS_UPDATED.md).


| Model | Số câu đổi | ΔBLEU mới−cũ | ΔchrF++ mới−cũ |
|---|---:|---:|---:|
| mBART LSR-only | 15 | +0.0776 | +0.0416 |
| NLLB baseline | 80 | +0.1184 | +0.0602 |
| NLLB Full | 4 | +0.0016 | +0.0027 |
| ByT5 baseline | 44 | −0.0980 | −0.0080 |
| ByT5 Full | 7 | +0.0052 | +0.0011 |


### T17 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 4. Độ bền metrics và correctness → 2. Kiểm tra implementation — TỐT trong phạm vi đã thử

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng318](EXPERIMENTS_UPDATED.md).


| Phép kiểm tra | Kết quả |
|---|---|
| Routing đúng khoảng cách d=2 | PASS: đầu ra toy [1, 2, 3.1, 4.2] với alpha=0.1 |
| Gradient theo nhánh residual | PASS: nguồn skip không nhận gradient từ riêng cạnh skip; destination gradient = 1 |
| Tensor trong cache đã detach | PASS |
| alpha=0, lambda=0 so với baseline | Logits/loss/gradient khớp chính xác trên cả 4 kiến trúc |
| Số parameter khi thêm wrapper | Bằng baseline; không thêm parameter |
| Batch khác xen giữa hai lần gọi | Logits không đổi; cache rỗng sau forward |
| Beam generation lặp lại sau input khác | Giống hệt; thử num_beams=2 trên encoder CLRR |
| LSR source/target | Source nhận gradient auxiliary khác 0; target hidden không requires_grad |
| Checkpointing, dropout=0 | 32/32 so sánh PASS: 4 kiến trúc × 4 methods × 2 chế độ |
| Checkpointing, dropout=0.1 | 8/8 so sánh Full PASS: 4 kiến trúc × 2 chế độ |


### T18 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 4. Độ bền metrics và correctness → 3. ChrF trên ký tự — kết quả mới từ predictions thực

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng346](EXPERIMENTS_UPDATED.md).


| Dataset / backbone / so sánh | Delta chrF | CI 95% | p Holm chung 50 | Đánh giá |
|---|---:|---|---:|---|
| Amis mBART Full − Baseline | +0.6907 | [-0.2363, 1.6608] | 1.0000 | Tín hiệu dương; chưa đủ chắc |
| Amis mBART Full − LSR-only | +0.6681 | [-0.2182, 1.5563] | 1.0000 | Đóng góp CLRR cần xác nhận qua seeds |
| Amis mBART CLRR-only − Baseline | -0.9957 | [-1.8811, -0.1256] | 0.3696 | Point estimate xấu; chưa có ý nghĩa sau Holm |
| Amis mBART LSR-only − Baseline | +0.0226 | [-0.9934, 1.0638] | 1.0000 | Gần như ngang baseline trên ký tự |
| Amis NLLB Full − Baseline | +0.6349 | [-0.0498, 1.3291] | 0.8657 | Dương nhưng chưa đủ chắc |
| Amis mT5 Full − Baseline | +1.6108 | [1.2587, 1.9804] | 0.0050 | TỐT: bằng chứng rõ trên checkpoint này |
| Amis ByT5 Full − Baseline | -0.2163 | [-0.6192, 0.1534] | 1.0000 | Không hỗ trợ claim cải thiện mọi backbone |


### T19 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 4. Độ bền metrics và correctness → 4. Ảnh hưởng từng câu — TỐT cho độ bền dấu của point estimate

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng376](EXPERIMENTS_UPDATED.md).


| So sánh, chrF word_order=0 | Delta đủ 575 câu | Min–max sau bỏ một câu bất kỳ | Số lần delta ≤ 0 |
|---|---:|---|---:|
| mBART Full − Baseline | +0.6907 | +0.5673 đến +0.7984 | 0/575 |
| mBART Full − LSR-only | +0.6681 | +0.5015 đến +0.8226 | 0/575 |
| NLLB Full − Baseline | +0.6349 | +0.5186 đến +0.7252 | 0/575 |
| mT5 Full − Baseline | +1.6108 | +1.5686 đến +1.6346 | 0/575 |


### T20 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 5. Cơ chế trên checkpoint đã train → 2. Effective rank và mức tập trung biểu diễn

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng409](EXPERIMENTS_UPDATED.md).


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


### T21 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 5. Cơ chế trên checkpoint đã train → 3. Residual norms và cân bằng CE–LSR

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng431](EXPERIMENTS_UPDATED.md).


| Dataset/model, method | Residual/current mean | Max | Weighted LSR/CE gradient, toàn model | Encoder excl. shared | Mean gradient cosine | Batch cosine âm /8 |
|---|---:|---:|---:|---:|---:|---:|
| Amis mBART Full | 5.911% | 8.626% | 0.167% | 0.310% | 0.057 | 0 |
| Amis mBART LSR-only | 0% | 0% | 0.237% | 0.412% | 0.046 | 1 |
| Amis NLLB Full | 6.632% | 8.764% | 0.487% | 0.737% | 0.149 | 0 |
| Amis mT5 Full | 6.213% | 9.838% | 0.794% | 1.755% | 0.084 | 3 |
| Amis mT5 CLRR-only | 6.209% | 9.833% | inactive | inactive | counterfactual | counterfactual |
| Amis ByT5 Full | 4.338% | 9.305% | 1.184% | 1.795% | −0.124 | 5 |


### T22 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 6. Thực nghiệm bổ sung và cấu hình thực tế → Strong LoRA đã hoàn tất trên Amis

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng462](EXPERIMENTS_UPDATED.md).


| Setting | mBART A | mBART B | NLLB A | NLLB B |
| --- | --- | --- | --- | --- |
| LoRA | r16/alpha32/dropout0.05 | như A | như A | như A |
| Targets | q,k,v,out,fc1,fc2 | như A | như A | như A |
| Embeddings | frozen | trainable | frozen | trainable |
| LR | 2e-4 | 1e-4 | 2e-4 | 1e-4 |
| Batch×accumulation | 4×32 | 4×32 | 16×8 | 16×8 |
| Best epoch / step | 19 / 684 | 19 / 684 | 18 / 648 | 19 / 684 |
| Added adapter params | 8,650,752 | 8,650,752 | 8,650,752 | 8,650,752 |
| Trainable params | 8,650,752 | 264,706,048 | 8,650,752 | 271,005,696 |
| Stored training runtime (min) | 93.4888 | 95.1309 | 32.9175 | 40.4702 |


### T23 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 9. Phụ lục truy vết — giữ riêng các phiên bản

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng559](EXPERIMENTS_UPDATED.md).


| Dataset | Backbone | Method | N | BLEU A | chrF++ A |
| --- | --- | --- | --- | --- | --- |
| amis_mandarin | mBART | Baseline | 575 | 19.6144 | 14.0547 |
| amis_mandarin | mBART | CLRR+LSR | 575 | 20.3896 | 19.0824 |
| amis_mandarin | mBART | CLRR-only | 575 | 18.4697 | 13.2443 |
| amis_mandarin | mBART | LSR-only | 575 | 19.6291 | 18.5753 |
| amis_mandarin | mBART | CLRR-Dec+LSR | 575 | 19.9188 | 18.6999 |
| amis_mandarin | mBART | BitFit | 575 | 0.3463 | 2.2552 |
| amis_mandarin | mBART | Narrow LoRA | 575 | 3.2365 | 4.4213 |
| amis_mandarin | mBART | Strong LoRA A | 575 | 10.2296 | 8.6319 |
| amis_mandarin | mBART | Strong LoRA B | 575 | 13.8066 | 10.3293 |
| amis_mandarin | NLLB | Strong LoRA A | 575 | 11.4625 | 9.5004 |
| amis_mandarin | NLLB | Strong LoRA B | 575 | 9.8024 | 8.4010 |
| amis_mandarin | mBART | Middle-Layer Alignment | 575 | 19.2893 | 18.4378 |
| amis_mandarin | NLLB | Baseline | 575 | 13.5001 | 10.4389 |
| amis_mandarin | NLLB | CLRR+LSR | 575 | 14.2820 | 10.9152 |
| amis_mandarin | NLLB | CLRR-Dec+LSR | 575 | 13.4194 | 10.4559 |
| amis_mandarin | NLLB | Middle-Layer Alignment | 575 | 13.9648 | 10.6596 |
| amis_mandarin | NLLB | BitFit | 575 | 1.0750 | 2.9643 |
| amis_mandarin | NLLB | Narrow LoRA | 575 | 3.5383 | 4.9407 |
| amis_mandarin | mT5 | Baseline | 575 | 2.7887 | 3.8129 |
| amis_mandarin | mT5 | CLRR-only | 575 | 4.4393 | 5.1761 |
| amis_mandarin | mT5 | CLRR+LSR | 575 | 4.5961 | 5.0425 |
| amis_mandarin | mT5 | CLRR-Dec+LSR | 575 | 4.8315 | 5.1933 |
| amis_mandarin | mT5 | CLRR-Both+LSR | 575 | 3.4459 | 4.5753 |
| amis_mandarin | mT5 | Middle-Layer Alignment | 575 | 4.6625 | 5.2103 |
| amis_mandarin | ByT5 | Baseline | 575 | 7.5837 | 8.3057 |
| amis_mandarin | ByT5 | CLRR+LSR | 575 | 7.3108 | 8.0956 |
| amis_mandarin | ByT5 | CLRR-Dec+LSR | 575 | 7.4394 | 8.0630 |
| amis_mandarin | ByT5 | Middle-Layer Alignment | 575 | 7.5660 | 8.3414 |


### T24 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 9. Phụ lục truy vết — giữ riêng các phiên bản

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng595](EXPERIMENTS_UPDATED.md).


| Dataset / backbone | Challenger − reference | Metric | Δ | CI95% | p raw | p Holm42 |
| --- | --- | --- | --- | --- | --- | --- |
| amis_mandarin / mBART | CLRR+LSR − Baseline | BLEU | 0.7752 | [-0.4388, 2.0441] | 0.0954 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − Baseline | chrF++ | 5.0277 | [-0.1872, 13.2525] | 0.1214 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − LSR-only | BLEU | 0.7606 | [-0.4086, 1.9237] | 0.0888 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − LSR-only | chrF++ | 0.5071 | [-0.2316, 1.2522] | 0.0798 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − CLRR-only | BLEU | 1.9199 | [0.8183, 3.0558] | 0.0006 | 0.0138 |
| amis_mandarin / mBART | CLRR+LSR − CLRR-only | chrF++ | 5.8381 | [0.7650, 14.0341] | 0.1000 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − Middle-Layer Alignment | BLEU | 1.1004 | [-0.0643, 2.2602] | 0.0292 | 0.6131 |
| amis_mandarin / mBART | CLRR+LSR − Middle-Layer Alignment | chrF++ | 0.6446 | [-0.0528, 1.3661] | 0.0333 | 0.6659 |
| amis_mandarin / mBART | CLRR+LSR − CLRR-Dec+LSR | BLEU | 0.4709 | [-0.5804, 1.5743] | 0.1480 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − CLRR-Dec+LSR | chrF++ | 0.3825 | [-0.2401, 1.0502] | 0.1040 | 1.0000 |
| amis_mandarin / mBART | CLRR+LSR − BitFit | BLEU | 20.0434 | [18.0696, 22.0865] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − BitFit | chrF++ | 16.8272 | [11.3416, 25.0158] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Narrow LoRA | BLEU | 17.1532 | [15.1766, 19.2210] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Narrow LoRA | chrF++ | 14.6610 | [9.2155, 23.1350] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Strong LoRA A | BLEU | 10.1601 | [8.2860, 12.1476] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Strong LoRA A | chrF++ | 10.4505 | [5.0353, 18.8632] | 0.0002 | 0.0052 |
| amis_mandarin / mBART | CLRR+LSR − Strong LoRA B | BLEU | 6.5830 | [4.9815, 8.2314] | 0.0001 | 0.0042 |
| amis_mandarin / mBART | CLRR+LSR − Strong LoRA B | chrF++ | 8.7531 | [3.4417, 15.6069] | 0.0004 | 0.0096 |
| amis_mandarin / NLLB | CLRR+LSR − Baseline | BLEU | 0.7818 | [-0.2288, 1.7849] | 0.0573 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR − Baseline | chrF++ | 0.4763 | [-0.0682, 1.0552] | 0.0466 | 0.8853 |
| amis_mandarin / NLLB | CLRR+LSR − Middle-Layer Alignment | BLEU | 0.3172 | [-0.7610, 1.3798] | 0.1909 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR − Middle-Layer Alignment | chrF++ | 0.2556 | [-0.3116, 0.8397] | 0.1408 | 1.0000 |
| amis_mandarin / NLLB | CLRR+LSR − Narrow LoRA | BLEU | 10.7437 | [9.2558, 12.2837] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR − Narrow LoRA | chrF++ | 5.9745 | [5.2485, 7.2458] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR − Strong LoRA A | BLEU | 2.8195 | [1.4847, 4.1169] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR − Strong LoRA A | chrF++ | 1.4148 | [0.7611, 2.1693] | 0.0003 | 0.0075 |
| amis_mandarin / NLLB | CLRR+LSR − Strong LoRA B | BLEU | 4.4796 | [3.1097, 5.8786] | 0.0001 | 0.0042 |
| amis_mandarin / NLLB | CLRR+LSR − Strong LoRA B | chrF++ | 2.5142 | [1.8153, 3.3643] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR − Baseline | BLEU | 2.0428 | [1.3530, 2.7629] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR − Baseline | chrF++ | 1.3803 | [0.6403, 1.7663] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Dec+LSR − Middle-Layer Alignment | BLEU | 0.1689 | [-0.3511, 0.7041] | 0.1813 | 1.0000 |
| amis_mandarin / mT5 | CLRR-Dec+LSR − Middle-Layer Alignment | chrF++ | -0.0171 | [-0.2487, 0.2175] | 0.3433 | 1.0000 |
| amis_mandarin / mT5 | CLRR+LSR − Baseline | BLEU | 1.8074 | [1.1826, 2.4746] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR+LSR − Baseline | chrF++ | 1.2296 | [0.5181, 1.5591] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Both+LSR − CLRR+LSR | BLEU | -1.1501 | [-1.7066, -0.6727] | 0.0001 | 0.0042 |
| amis_mandarin / mT5 | CLRR-Both+LSR − CLRR+LSR | chrF++ | -0.4672 | [-0.7534, -0.2068] | 0.0013 | 0.0286 |


### T25 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 9. Phụ lục truy vết — giữ riêng các phiên bản

**Phạm vi:** Historical audit/inference;metric/source version theo tiêu đề và cột nguồn. Nguồn: [EXPERIMENTS_UPDATED.md, dòng639](EXPERIMENTS_UPDATED.md).


| Challenger − reference | Metric | Δ | CI95% | p raw | p Holm36 |
| --- | --- | --- | --- | --- | --- |
| mbart-full − mbart-baseline | bleu | 0.7752 | [-0.4388, 2.0441] | 0.0954 | 1.0000 |
| mbart-full − mbart-baseline | chrfpp | 5.0277 | [-0.1872, 13.2525] | 0.1214 | 1.0000 |
| mbart-full − mbart-baseline | chrf0 | 0.6907 | [-0.2363, 1.6608] | 0.0695 | 1.0000 |
| mbart-full − mbart-lsr | bleu | 0.6830 | [-0.4806, 1.8530] | 0.1069 | 1.0000 |
| mbart-full − mbart-lsr | chrfpp | 0.4655 | [-0.2738, 1.2102] | 0.0928 | 1.0000 |
| mbart-full − mbart-lsr | chrf0 | 0.6122 | [-0.2688, 1.5009] | 0.0797 | 1.0000 |
| nllb-full − nllb-baseline | bleu | 0.6650 | [-0.3295, 1.7156] | 0.0849 | 1.0000 |
| nllb-full − nllb-baseline | chrfpp | 0.4189 | [-0.1279, 0.9924] | 0.0631 | 1.0000 |
| nllb-full − nllb-baseline | chrf0 | 0.5580 | [-0.1333, 1.2513] | 0.0508 | 1.0000 |
| mt5-full − mt5-baseline | bleu | 1.8074 | [1.1826, 2.4746] | 0.0001 | 0.0036 |
| mt5-full − mt5-baseline | chrfpp | 1.2296 | [0.5181, 1.5591] | 0.0001 | 0.0036 |
| mt5-full − mt5-baseline | chrf0 | 1.6108 | [1.2587, 1.9804] | 0.0001 | 0.0036 |
| mt5-full − mt5-clrr | bleu | 0.1567 | [-0.3769, 0.7116] | 0.2012 | 1.0000 |
| mt5-full − mt5-clrr | chrfpp | -0.1336 | [-0.4268, 0.1224] | 0.1332 | 1.0000 |
| mt5-full − mt5-clrr | chrf0 | -0.1782 | [-0.5549, 0.1615] | 0.1287 | 1.0000 |
| byt5-full − byt5-baseline | bleu | -0.1696 | [-0.8869, 0.5266] | 0.2199 | 1.0000 |
| byt5-full − byt5-baseline | chrfpp | -0.2011 | [-0.5634, 0.1240] | 0.1039 | 1.0000 |
| byt5-full − byt5-baseline | chrf0 | -0.2057 | [-0.5988, 0.1512] | 0.1123 | 1.0000 |
| mbart-full − mbart-full::route_off | bleu | 3.9287 | [2.7175, 5.1948] | 0.0001 | 0.0036 |
| mbart-full − mbart-full::route_off | chrfpp | 2.6080 | [1.9050, 3.4956] | 0.0001 | 0.0036 |
| mbart-full − mbart-full::route_off | chrf0 | 3.2964 | [2.3996, 4.2335] | 0.0001 | 0.0036 |
| nllb-full − nllb-full::route_off | bleu | 1.3659 | [0.3337, 2.7009] | 0.0170 | 0.4080 |
| nllb-full − nllb-full::route_off | chrfpp | 0.8881 | [0.3166, 1.5406] | 0.0032 | 0.0832 |
| nllb-full − nllb-full::route_off | chrf0 | 1.1262 | [0.4142, 1.8911] | 0.0012 | 0.0324 |
| mt5-full − mt5-full::route_off | bleu | 0.4577 | [0.0719, 0.8816] | 0.0180 | 0.4140 |
| mt5-full − mt5-full::route_off | chrfpp | 0.2210 | [-0.5736, 0.3980] | 0.0771 | 1.0000 |
| mt5-full − mt5-full::route_off | chrf0 | 0.2659 | [0.0292, 0.4996] | 0.0142 | 0.3550 |
| byt5-full − byt5-full::route_off | bleu | -0.2006 | [-0.7983, 0.3788] | 0.1780 | 1.0000 |
| byt5-full − byt5-full::route_off | chrfpp | 0.0039 | [-0.2421, 0.2409] | 0.4084 | 1.0000 |
| byt5-full − byt5-full::route_off | chrf0 | 0.0045 | [-0.2824, 0.2811] | 0.4084 | 1.0000 |


### T26 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 10. Suite kiểm chứng mBART Amis — task 1 đến 3 → Task 3: 12 run công bằng + trajectory mới

**Phạm vi:** Historical training/probes trước accumulation fix;không matched corrected cohort. Nguồn: [EXPERIMENTS_UPDATED.md, dòng702](EXPERIMENTS_UPDATED.md).


| Setting | Chung cho Baseline / CLRR-only / LSR-only / Full |
| --- | --- |
| Data | Amis→Mandarin, 4600 / 576 / 575; giữ nguyên rows/order |
| Seeds | 42, 43, 44 — mỗi seed đủ cả 4 nhánh |
| Initial weights | mBART-50 revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30` |
| LR / weight decay / warmup | 5e-5 / 0.0 / 0.06 |
| Batch / accumulation | 32 × 4 = 128 |
| Epochs / early stopping | Tối đa 20 / patience 4, cùng raw-reference validation chrF++ |
| Language codes | tl_XX → zh_CN |
| CLRR / LSR | Encoder d=2, alpha=0.1 khi CLRR bật; lambda=0.1 khi LSR bật |
| Precision | Train BF16/TF32 + gradient checkpointing; generate FP32, TF32 off |
| Selection / test | Validation greedy beam 1 chọn best; test beam 4 |
| GPU / environment | A100 Standard, không high-mem; conda clrr |
| Observational probes | Epoch 1, epoch 5, và actual training endpoint; validation rows/batches cố định |


### T27 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 11. V4 — xác minh checkpoint và cơ chế đúng protocol, 04/10/2026 → 11.1. Route-on/off — validation greedy, cùng weights đã train routing

**Phạm vi:** Historical V4/V5 verified interventions/probes;không train-time causal ablation. Nguồn: [EXPERIMENTS_UPDATED.md, dòng946](EXPERIMENTS_UPDATED.md).


| Seed | BLEU on | BLEU off | ΔBLEU | ΔchrF++ | Câu đổi /576 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 17.6437 | 14.5377 | +3.1060 | +2.4378 | 523 |
| 43 | 17.0645 | 13.9863 | +3.0783 | +2.4063 | 522 |
| 44 | 16.9324 | 14.2137 | +2.7187 | +2.5030 | 536 |
| Mean | 17.2135 | 14.2459 | **+2.9676** | **+2.4491** | 527 |


### T28 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 11. V4 — xác minh checkpoint và cơ chế đúng protocol, 04/10/2026 → 11.2. Source pooled ranks — giữ rõ site và định nghĩa

**Phạm vi:** Historical V4/V5 verified interventions/probes;không train-time causal ablation. Nguồn: [EXPERIMENTS_UPDATED.md, dòng964](EXPERIMENTS_UPDATED.md).


| Family | Block12 trước final LN: singular rank | Final output: singular rank | Final output: energy entropy rank |
| --- | ---: | ---: | ---: |
| Baseline | 158.4847 | 389.9910 | 146.0275 |
| CLRR-only | 258.2300 | 385.0201 | 147.1425 |
| Matched LSR λ0.10 | 220.0196 | 386.3290 | 137.0831 |
| Tuned LSR λ0.03 | 185.3190 | 389.5775 | 143.8891 |
| Full λ0.10 | 254.8527 | 385.2893 | 144.2210 |
| Full λ0.03 | 258.3149 | 385.5640 | 146.8733 |
| Full attached λ0.10 | 254.2608 | 384.9135 | 143.7396 |


### T29 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 11. V4 — xác minh checkpoint và cơ chế đúng protocol, 04/10/2026 → 11.3. CE–LSR gradient và residual

**Phạm vi:** Historical V4/V5 verified interventions/probes;không train-time causal ablation. Nguồn: [EXPERIMENTS_UPDATED.md, dòng984](EXPERIMENTS_UPDATED.md).


| Family | Global gradient cosine | WeightedLSR / CE norm (%) |
| --- | ---: | ---: |
| Matched LSR λ0.10 | 0.0724 | 0.2337 |
| Tuned LSR λ0.03 | 0.0630 | 0.1007 |
| Full λ0.10 | 0.0516 | 0.1570 |
| Full λ0.03 | 0.0489 | 0.0700 |
| Full attached λ0.10 | 0.0716 | 0.2046 |


### T30 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 12. V5 — routing interventions, decoding và sửa nhãn baseline validation → 12.2. Các đối chứng validation beam1 cùng protocol

**Phạm vi:** Historical V4/V5 verified interventions/probes;không train-time causal ablation. Nguồn: [EXPERIMENTS_UPDATED.md, dòng1061](EXPERIMENTS_UPDATED.md).


| Nhánh | BLEUzh | chrF++ | chrF0 |
| --- | --- | --- | --- |
| Baseline | 17.1275 ± 0.8499 | 14.3003 ± 0.4568 | 16.4849 ± 0.5888 |
| Matched LSR λ.10 | 16.5430 ± 0.0686 | 14.0608 ± 0.1114 | 16.1677 ± 0.1169 |
| Tuned LSR λ.03 | 16.9942 ± 0.7261 | 14.3170 ± 0.4705 | 16.4950 ± 0.5583 |
| Full α.10/λ.10 | 17.2135 ± 0.3783 | 14.4975 ± 0.2053 | 16.6586 ± 0.1913 |


### T31 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 12. V5 — routing interventions, decoding và sửa nhãn baseline validation → 12.3. Can thiệp routing trên Full weights đã train

**Phạm vi:** Historical V4/V5 verified interventions/probes;không train-time causal ablation. Nguồn: [EXPERIMENTS_UPDATED.md, dòng1074](EXPERIMENTS_UPDATED.md).


| Can thiệp | ΔBLEU | BLEU dương | ΔchrF++ | ΔchrF0 | ΔCE fixed32 |
| --- | --- | --- | --- | --- | --- |
| α=.05 | -0.4830 | 1/3 | -1.0660 | -0.5735 | +0.0574 |
| α=.075 | -0.1636 | 0/3 | -0.3483 | -0.3872 | +0.0279 |
| α=.125 | -0.4773 | 0/3 | -0.2803 | -0.3175 | -0.0252 |
| α=.15 | -0.8456 | 0/3 | -0.5617 | -0.6268 | -0.0426 |
| Tắt early3–5 | -0.3272 | 0/3 | -0.3587 | -0.3811 | +0.0153 |
| Tắt middle6–8 | -0.1928 | 1/3 | -0.9181 | -0.3777 | +0.0432 |
| Tắt late9–12 | -0.5706 | 0/3 | -0.4601 | -0.5369 | +0.0663 |


### T32 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 12. V5 — routing interventions, decoding và sửa nhãn baseline validation → 12.3. Can thiệp routing trên Full weights đã train

**Phạm vi:** Historical V4/V5 verified interventions/probes;không train-time causal ablation. Nguồn: [EXPERIMENTS_UPDATED.md, dòng1097](EXPERIMENTS_UPDATED.md).


| Can thiệp | BLEU interval | chrF++ interval | chrF0 interval |
| --- | --- | --- | --- |
| α=.05 | [-0.9880, +0.0144] | [-1.3528, -0.2487] | [-0.9383, -0.2089] |
| α=.075 | [-0.7111, +0.2154] | [-0.6665, -0.0447] | [-0.7579, -0.0371] |
| α=.125 | [-0.9384, -0.0325] | [-0.5780, +0.0249] | [-0.6383, +0.0093] |
| α=.15 | [-1.4357, -0.2581] | [-0.9662, -0.1688] | [-1.0588, -0.2088] |
| Tắt early3–5 | [-0.9489, +0.1275] | [-0.6584, -0.0654] | [-0.7278, -0.0462] |
| Tắt middle6–8 | [-0.8494, +0.3005] | [-1.2124, -0.0874] | [-0.7417, -0.0034] |
| Tắt late9–12 | [-1.0431, -0.0961] | [-0.7531, -0.1693] | [-0.8692, -0.2009] |


### T33 — Thực nghiệm sau xác minh — kết quả hiện hành của CLRR → 12. V5 — routing interventions, decoding và sửa nhãn baseline validation → 12.4. Full−tuned LSR: beam không giải thích hết đảo thứ hạng

**Phạm vi:** Historical V4/V5 verified interventions/probes;không train-time causal ablation. Nguồn: [EXPERIMENTS_UPDATED.md, dòng1112](EXPERIMENTS_UPDATED.md).


| Split/decode | ΔBLEU | BLEU dương | ΔchrF++ | ΔchrF0 |
| --- | --- | --- | --- | --- |
| Validation beam1 | +0.2193 | 2/3 | +0.1805 | +0.1636 |
| Validation beam4 | +0.2055 | 2/3 | +0.1946 | +0.1520 |
| Test beam1 mới | -0.2168 | 1/3 | -0.1770 | -0.1994 |
| Test beam4 cũ | -0.1267 | 1/3 | +1.3642 | -0.1635 |


**Historical morphology:** mi/ma/pa regex là surface proxies, không gold voice annotation;Root nghĩa không khớp3regex. Overlap tồn tại. **Rank:** pooled centered singular entropy rank khác energy entropy rank/participation ratio;limit min(n−1,d). **Gradients:** offline fixedbatch probes, không trainingtrace;counterfactualLSR cho Baseline/CLRR-only không là objective train. **Intervention:** αoff/layermask trên model trained routing có distribution shift.


### Historical final test — 21 saved checkpoints, cùng raw-score CSV


| Family | Seed | BLEU | chrF++ | chrF0 |
| --- | --- | --- | --- | --- |
| Baseline (mBART) | 42 | 19.728081 | 18.713838 | 18.397282 |
| Baseline (mBART) | 43 | 18.137832 | 13.159189 | 17.090430 |
| Baseline (mBART) | 44 | 20.300016 | 18.901673 | 18.647752 |
| Full CLRR Detached (alpha=0.10, lambda=0.10) | 42 | 19.956574 | 18.887811 | 18.662645 |
| Full CLRR Detached (alpha=0.10, lambda=0.10) | 43 | 19.612260 | 18.475346 | 18.190512 |
| Full CLRR Detached (alpha=0.10, lambda=0.10) | 44 | 19.737135 | 18.606483 | 18.261258 |
| Matched LSR (lambda=0.10) | 42 | 19.786496 | 18.679745 | 18.358093 |
| Matched LSR (lambda=0.10) | 43 | 19.487941 | 18.648797 | 18.258137 |
| Matched LSR (lambda=0.10) | 44 | 19.328419 | 18.363270 | 17.995749 |
| Tuned LSR (lambda=0.03) | 42 | 20.625679 | 19.188905 | 19.028835 |
| Tuned LSR (lambda=0.03) | 43 | 19.674662 | 14.155648 | 18.418840 |
| Tuned LSR (lambda=0.03) | 44 | 19.385821 | 18.532387 | 18.157330 |
| Full Attached Control (alpha=0.10, lambda=0.10, detach=False) | 42 | 19.353091 | 18.541196 | 18.223969 |
| Full Attached Control (alpha=0.10, lambda=0.10, detach=False) | 43 | 19.777445 | 18.513107 | 18.210414 |
| Full Attached Control (alpha=0.10, lambda=0.10, detach=False) | 44 | 19.566730 | 18.635474 | 18.357129 |
| CLRR-only Detached (alpha=0.10, lambda=0.00) | 42 | 19.640293 | 18.563897 | 18.250377 |
| CLRR-only Detached (alpha=0.10, lambda=0.00) | 43 | 18.942021 | 18.082320 | 17.701680 |
| CLRR-only Detached (alpha=0.10, lambda=0.00) | 44 | 20.148450 | 18.905467 | 18.658406 |


| Historical family | n | BLEU mean ± SD | chrF++ mean ± SD | chrF0 mean ± SD |
| --- | --- | --- | --- | --- |
| baseline | 3 | 19.388643 ± 1.120346 | 16.924900 ± 3.262553 | 18.045155 ± 0.836246 |
| original_full_detached | 3 | 19.768656 ± 0.174308 | 18.656547 ± 0.210741 | 18.371471 ± 0.254632 |
| matched_lsr | 3 | 19.534285 ± 0.232528 | 18.563937 ± 0.174470 | 18.203993 ± 0.187142 |
| tuned_lsr | 3 | 19.895387 ± 0.648731 | 17.292313 ± 2.736194 | 18.535002 ± 0.447214 |
| full_source_attached | 3 | 19.565755 ± 0.212179 | 18.563259 ± 0.064098 | 18.263837 ± 0.081077 |
| clrr_only_detached | 3 | 19.576921 ± 0.605706 | 18.517228 ± 0.413553 | 18.203488 ± 0.480083 |


Nguồn: [test_scores_all_runs.csv](../../outputs_rebuttal/amis_final_evaluation_20261004/test_scores_all_runs.csv). SHA256: `18899ebd5ea89fa5fe12eb9c52da0c8b10dd624e08026e8e7562584e8657754b`.


Raw CSV ưu tiên hơn các bảng mean đã làm tròn trong handoff. Đây là checkpoints lịch sử trước corrected recipe; không dùng làm main corrected comparison. Baseline audited reuse có thể trùng điểm nhưng không biến phương pháp lịch sử thành corrected.


## Appendix I — Configuration search, attached và warmup


### T36 — Amis routing warmup v2 — kết quả và quyết định → 2. Số từng seed

**Phạm vi:** Warmup20%historical,validation-only3seeds;NO-GO cho schedule này,không cho toàn bộ corrected method. Nguồn: [AMIS_WARMUP_V2_RESULTS.md, dòng18](../research/archive/20261005_before_consolidation/AMIS_WARMUP_V2_RESULTS.md).


| Nhánh warmup | Seed | BLEU | chrF++ | chrF0 | ΔBLEU so fixed cùng seed | ΔchrF0 so fixed |
|---|---:|---:|---:|---:|---:|---:|
| Full | 42 | 17.7908 | 14.9599 | 17.0269 | +0.1472 | +0.1517 |
| Full | 43 | 16.2231 | 13.8586 | 15.9132 | −0.8414 | −0.6746 |
| Full | 44 | 17.0006 | 14.3394 | 16.5026 | +0.0681 | −0.0101 |
| CLRR-only | 42 | 16.6623 | 13.9601 | 16.0884 | +0.3968 | +0.2400 |
| CLRR-only | 43 | 14.4003 | 12.5549 | 14.5335 | −2.5095 | −1.7136 |
| CLRR-only | 44 | 16.8898 | 14.2423 | 16.3599 | −0.3840 | −0.1595 |


### T37 — Amis routing warmup v2 — kết quả và quyết định → 3. Tổng hợp ba seeds và gate đã khóa

**Phạm vi:** Warmup20%historical,validation-only3seeds;NO-GO cho schedule này,không cho toàn bộ corrected method. Nguồn: [AMIS_WARMUP_V2_RESULTS.md, dòng29](../research/archive/20261005_before_consolidation/AMIS_WARMUP_V2_RESULTS.md).


| Cấu hình | BLEU mean | chrF++ mean | chrF0 mean |
|---|---:|---:|---:|
| Fixed Full control | 17.2135 | 14.4975 | 16.6586 |
| **Warmup Full** | **17.0048** | **14.3859** | **16.4809** |
| Tuned LSR-only, λ=.03 | 16.9942 | 14.3170 | 16.4950 |
| Fixed CLRR-only control | 16.8164 | 14.0765 | 16.2050 |
| **Warmup CLRR-only** | **15.9842** | **13.5858** | **15.6606** |


### T38 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 3. Kết quả đã có và điều cần cảnh giác → Audit C không train — đã hoàn tất

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng90](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Character-only chrF | Seed42 | Seed43 | Seed44 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Full−Baseline | +0.2654 | +1.1001 | −0.3865 | +0.3263 |
| Full−LSR | +0.3046 | −0.0676 | +0.2655 | +0.1675 |


### T39 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 3. Kết quả đã có và điều cần cảnh giác → Screening D — chỉ validation

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng101](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Trial | α | λ | Seed | Validation BLEU | Validation chrF++ | Validation chrF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full gốc, reuse metadata C | 0.10 | 0.10 | 42 | 17.6437 | 14.7317 | Chưa có validation predictions |
| LSR gốc, reuse metadata C | 0 | 0.10 | 42 | 16.5688 | 14.1674 | Chưa có validation predictions |
| Baseline C, mốc tham chiếu | 0 | 0 | 42 | 17.8735 | 14.9009 | Chưa tái generate validation trong lượt này |
| `full_a005_l010` | 0.05 | 0.10 | 42 | 17.1225 | 14.4092 | 16.4998 |
| `full_a020_l010` | 0.20 | 0.10 | 42 | 17.4586 | 14.5738 | 16.7469 |
| `full_a010_l003` | 0.10 | 0.03 | 42 | 17.1141 | 14.4355 | 16.5587 |
| `lsr_l060` | 0 | 0.60 | 42 | 17.0300 | 14.3688 | 16.5080 |
| `lsr_l003` | 0 | 0.03 | 42 | 17.7873 | 14.8587 | 17.1364 |
| `lsr_l001` | 0 | 0.01 | 42 | 16.3028 | 13.9491 | 16.0469 |
| `full_a010_l030` | 0.10 | 0.30 | 42 | 17.1999 | 14.3614 | 16.4432 |
| `lsr_l030` | 0 | 0.30 | 42 | 16.3826 | 13.8951 | 16.0409 |


### T40 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 10. Đánh giá sau sáu lượt confirmation — 04/10/2026 → So đúng cùng ba seeds42/43/44 — validation beam1

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng545](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| System | Mean BLEU | Mean chrF++ | Sample std chrF++ |
| --- | ---: | ---: | ---: |
| Baseline C | 17.6633 | 14.6334 | 0.3782 |
| Selected Full α0.10/λ0.10 | 17.2135 | 14.4975 | 0.2053 |
| Best-tuned LSR λ0.03 | 16.9942 | 14.3170 | 0.4705 |
| Matched LSR λ0.10 | 16.5430 | 14.0608 | 0.1114 |
| Diagnostic Full α0.10/λ0.03 | 17.0403 | 14.3476 | 0.1899 |


### T41 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 10. Đánh giá sau sáu lượt confirmation — 04/10/2026 → So đúng cùng ba seeds42/43/44 — validation beam1

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng557](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Comparison | Mean ΔBLEU | Seeds BLEU dương | Mean ΔchrF++ | Seeds chrF++ dương |
| --- | ---: | ---: | ---: | ---: |
| Full0.10−Baseline | −0.4498 | 0/3 | −0.1359 | 1/3 |
| Full0.10−matchedLSR0.10 | +0.6705 | 3/3 | +0.4368 | 3/3 |
| Full0.10−bestLSR0.03 | +0.2193 | 2/3 | +0.1805 | 2/3 |
| Full0.03−LSR0.03 | +0.0461 | 1/3 | +0.0306 | 2/3 |


### T42 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 12. Hoàn tất Phase C2 (Attached Control) & Kết luận Cổng Quyết định (04/10/2026) → 12.1. Kết quả thực nghiệm 3 seeds Attached Control ($\alpha=0.10, \lambda=0.10$)

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng762](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Run | Seed | Best Checkpoint | Val BLEU | Val chrF++ | Val chrF0 | Trạng thái HF |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `mbart-amis-full-attached-a010-l010-seed42` | 42 | checkpoint-576 | 16.8827 | 14.1845 | 16.2653 | Verified SHA256 / Uploaded |
| `mbart-amis-full-attached-a010-l010-seed43` | 43 | checkpoint-684 | 17.4909 | 14.4706 | 16.6557 | Verified SHA256 / Uploaded |
| `mbart-amis-full-attached-a010-l010-seed44` | 44 | checkpoint-612 | 17.6376 | 14.6962 | 16.8618 | Verified SHA256 / Uploaded |
| **Mean $\pm$ Std** | — | — | **17.3371 $\pm$ 0.4005** | **14.4504 $\pm$ 0.2566** | **16.5943 $\pm$ 0.3029** | **Hoàn tất 100%** |


### T43 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 12. Hoàn tất Phase C2 (Attached Control) & Kết luận Cổng Quyết định (04/10/2026) → 12.4. Paired Deltas, Kiểm định Bootstrap 10.000 mẫu & Hiệu chỉnh Holm-Bonferroni → 1. Bảng chênh lệch bắt cặp (Paired Deltas: Selected Full − Control)

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng823](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Đối chứng (Control) | Chỉ số | Seed 42 | Seed 43 | Seed 44 | Mean Delta $\pm$ Std | Tỷ lệ Seed dương |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **vs Matched LSR ($\lambda=0.10$)** | BLEU | +0.1701 | +0.1243 | +0.4087 | **+0.2344 $\pm$ 0.1527** | **3/3 (100%)** |
| | chrF++ | +0.2081 | -0.1735 | +0.2432 | **+0.0926 $\pm$ 0.2311** | 2/3 |
| | chrF0 | +0.3046 | -0.0676 | +0.2655 | **+0.1675 $\pm$ 0.2045** | 2/3 |
| **vs Baseline (mBART)** | BLEU | +0.2285 | +1.4744 | -0.5629 | **+0.3800 $\pm$ 1.0271** | 2/3 |
| | chrF++ | +0.1740 | +5.3162 | -0.2952 | **+1.7316 $\pm$ 3.1131** | 2/3 |
| | chrF0 | +0.2654 | +1.1001 | -0.3865 | **+0.3263 $\pm$ 0.7452** | 2/3 |
| **vs Tuned LSR ($\lambda=0.03$)** | BLEU | -0.6691 | -0.0624 | +0.3513 | -0.1267 $\pm$ 0.5132 | 1/3 |
| | chrF++ | -0.3011 | +4.3197 | +0.0741 | **+1.3642 $\pm$ 2.5664** | 2/3 |
| | chrF0 | -0.3662 | -0.2283 | +0.1039 | -0.1635 $\pm$ 0.2417 | 1/3 |
| **vs Attached Control** | BLEU | +0.6035 | -0.1652 | +0.1704 | **+0.2029 $\pm$ 0.3854** | 2/3 |
| | chrF++ | +0.3466 | -0.0378 | -0.0290 | **+0.0933 $\pm$ 0.2194** | 1/3 |
| | chrF0 | +0.4387 | -0.0199 | -0.0959 | **+0.1076 $\pm$ 0.2892** | 1/3 |
| **vs Detached CLRR-only** | BLEU | +0.3163 | +0.6702 | -0.4113 | **+0.1917 $\pm$ 0.5514** | 2/3 |
| | chrF++ | +0.3239 | +0.3930 | -0.2990 | **+0.1393 $\pm$ 0.3812** | 2/3 |
| | chrF0 | +0.4123 | +0.4888 | -0.3971 | **+0.1680 $\pm$ 0.4909** | 2/3 |


### T44 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 15. Hoàn tất Toàn diện Giai đoạn Follow-up (F0 $\rightarrow$ F6) (04/10/2026) → 15.2. Kết quả 5 Cặp So sánh Thứ cấp (F2: 30 Tests Holm-Bonferroni)

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng1294](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Cặp so sánh (Challenger vs Control) | Metric | Seed 42 | Seed 43 | Seed 44 | Mean Delta | Std Paired Delta | Số seed dương | Holm-30 Sig ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CLRR-only vs Baseline** | BLEU | $-0.0878$ | $+0.8042$ | $-0.1516$ | $+0.1883$ | $0.5343$ | 1 / 3 | **False** ($p=1.0$) |
| | chrF++ | $-0.1499$ | $+4.9231$ | $+0.0038$ | $+1.5923$ | $2.8856$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF0 | $-0.1469$ | $+0.6113$ | $+0.0107$ | $+0.1583$ | $0.4001$ | 2 / 3 | — |
| **2. Full $\lambda=0.10$ vs CLRR-only** | BLEU | $+0.3163$ | $+0.6702$ | $-0.4113$ | $+0.1917$ | $0.5514$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF++ | $+0.3239$ | $+0.3930$ | $-0.2990$ | $+0.1393$ | $0.3812$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF0 | $+0.4123$ | $+0.4888$ | $-0.3971$ | $+0.1680$ | $0.4909$ | 2 / 3 | — |
| **3. Full $\lambda=0.10$ (detached) vs Full attached** | BLEU | $+0.6035$ | $-0.1652$ | $+0.1704$ | $+0.2029$ | $0.3854$ | 2 / 3 | **False** ($p=1.0$) |
| | chrF++ | $+0.3466$ | $-0.0378$ | $-0.0290$ | $+0.0933$ | $0.2194$ | 1 / 3 | **False** ($p=1.0$) |
| | chrF0 | $+0.4387$ | $-0.0199$ | $-0.0959$ | $+0.1076$ | $0.2892$ | 1 / 3 | — |
| **4. Full $\lambda=0.03$ vs Tuned LSR $\lambda=0.03$** | BLEU | $-1.4130$ | $-0.5557$ | $-0.4613$ | **$-0.8100$** | $0.5243$ | **0 / 3** | **False** ($p=0.267$) |
| | chrF++ | $-0.7965$ | $+4.0623$ | $-0.4456$ | $+0.9401$ | $2.7096$ | 1 / 3 | **False** ($p=0.458$) |
| | chrF0 | $-0.9429$ | $-0.5683$ | $-0.4995$ | **$-0.6702$** | $0.2387$ | **0 / 3** | — |
| **5. Full $\lambda=0.10$ vs Full $\lambda=0.03$** | BLEU | $+0.7439$ | $+0.4933$ | $+0.8126$ | **$+0.6833$** | $0.1680$ | **3 / 3** | **False** ($p=1.0$) |
| | chrF++ | $+0.4955$ | $+0.2574$ | $+0.5196$ | **$+0.4242$** | $0.1449$ | **3 / 3** | **False** ($p=1.0$) |
| | chrF0 | $+0.5767$ | $+0.3400$ | $+0.6034$ | **$+0.5067$** | $0.1450$ | **3 / 3** | — |


### T45 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 18. Thực thi xác minh V4 — hoàn tất và VERIFIED, 04/10/2026 → 18.2. Routing có tác dụng inference — đã xác minh đúng protocol

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng1749](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Seed | BLEU on | BLEU off | ΔBLEU on−off | ΔchrF++ | Câu đổi /576 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 17.6437 | 14.5377 | +3.1060 | +2.4378 | 523 |
| 43 | 17.0645 | 13.9863 | +3.0783 | +2.4063 | 522 |
| 44 | 16.9324 | 14.2137 | +2.7187 | +2.5030 | 536 |
| Mean | 17.2135 | 14.2459 | **+2.9676** | **+2.4491** | 527 |


### T46 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 18. Thực thi xác minh V4 — hoàn tất và VERIFIED, 04/10/2026 → 18.3. Rank: thay đổi phổ trung gian, chưa chứng minh anti-collapse

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng1771](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Family | Block12 trước final LN | Final output sau LN | Final covariance-energy entropy rank |
| --- | ---: | ---: | ---: |
| Baseline | 158.4847 | 389.9910 | 146.0275 |
| CLRR-only | 258.2300 | 385.0201 | 147.1425 |
| LSR λ0.10 | 220.0196 | 386.3290 | 137.0831 |
| LSR λ0.03 | 185.3190 | 389.5775 | 143.8891 |
| Full λ0.10 | 254.8527 | 385.2893 | 144.2210 |
| Full λ0.03 | 258.3149 | 385.5640 | 146.8733 |
| Full attached λ0.10 | 254.2608 | 384.9135 | 143.7396 |


### T47 — Bàn giao: tìm cấu hình CLRR trên mBART Amis → 18. Thực thi xác minh V4 — hoàn tất và VERIFIED, 04/10/2026 → 18.4. Gradient và residual: chưa thấy lý do sửa loss/normalize ngay

**Phạm vi:** Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning. Nguồn: [CONFIGURATION_SEARCH_HANDOFF.md, dòng1798](../operations/CONFIGURATION_SEARCH_HANDOFF.md).


| Objective thực train | Global cosine CE–weightedLSR | WeightedLSR / CE norm (%) |
| --- | ---: | ---: |
| LSR λ0.10 | 0.0724 | 0.2337 |
| LSR λ0.03 | 0.0630 | 0.1007 |
| Full λ0.10 | 0.0516 | 0.1570 |
| Full λ0.03 | 0.0489 | 0.0700 |
| Full attached λ0.10 | 0.0716 | 0.2046 |


## Appendix J — Evidence chưa đủ và kết quả pending


| Hạng mục | Trạng thái | Cách xử lý khi viết paper |
| --- | --- | --- |
| Archived original tables | Historical/unverified where unsupported | Tra EXPERIMENTS_ORIGINAL;không copy dấu significant gốc |
| Original p_holm=.0006 toàn16rows | Không có family/provenance hỗ trợ toànbảng | Dùng CSV audit cùngfamily42/36/24/12;không thu nhỏfamily sau lọcscope |
| Old no-SG >1200 và18.4±4.2 | INVALID gradienttrace:18zeros,callback afterzero | Không đưa vào bằng chứng |
| V3/F3 probes/F4 contaminated route-off | Superseded/contaminated | Dùng V4/V5 scopedmeasurements;không giữ contradictory conclusions |
| Old sensitivity optimum vàmT5timing | Thiếu trial/timing receipts | Không gọi validationoptimum hoặc confirmedoverhead |
| Original qualitative examples | Không khớp source/test savedoutputs | Không dùng như output model thật |
| AffixF1/morphologygold | Thiếu verifiedrawreport/goldannotation | Không claim morphological preservation dựa regex |
| Fresh32seeds45–52 | PLANNED;chưa tìm đủ completedmodelreceipts | Không có mean/SD hoặc testresult để điền |
| Middle matchedcorrected | Chưa có | Middle lịch sử chỉ standalonebaseline record |
| Turkish→English | Zero-shot/pilot không finalcomparison;đãdừng | Không claim generality từ completedsecondlanguage |
| Asháninka/LayerSkip | Ngoài scope theo tácgiả | Không đưa vào main/appendix tables mới |


Không bỏ negative results: historical warmup/attached/ByT5/LSR-controls còn trong các bảng đúngscope. Không biến pending thành0score hoặc estimatedresult. Tuning/probe/benchmark không cộng thành fullresearchtrainingmodels.


## Appendix K — Inventory nguồn và bảo toàn


| Table | File nguồn | Dòng | Phạm vi |
| --- | --- | --- | --- |
| T07 | outputs_rebuttal/amis_corrected_four_way_20261005/FOUR_WAY_ANALYSIS.md | 20 | Corrected test; CI conditional sentence bootstrap giữ cố định3seeds; sign-flip theo3paired seeds |
| T13 | docs/experiments/EXPERIMENTS_UPDATED.md | 226 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T14 | docs/experiments/EXPERIMENTS_UPDATED.md | 265 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T15 | docs/experiments/EXPERIMENTS_UPDATED.md | 281 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T16 | docs/experiments/EXPERIMENTS_UPDATED.md | 297 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T17 | docs/experiments/EXPERIMENTS_UPDATED.md | 318 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T18 | docs/experiments/EXPERIMENTS_UPDATED.md | 346 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T19 | docs/experiments/EXPERIMENTS_UPDATED.md | 376 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T20 | docs/experiments/EXPERIMENTS_UPDATED.md | 409 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T21 | docs/experiments/EXPERIMENTS_UPDATED.md | 431 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T22 | docs/experiments/EXPERIMENTS_UPDATED.md | 462 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T23 | docs/experiments/EXPERIMENTS_UPDATED.md | 559 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T24 | docs/experiments/EXPERIMENTS_UPDATED.md | 595 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T25 | docs/experiments/EXPERIMENTS_UPDATED.md | 639 | Historical audit/inference;metric/source version theo tiêu đề và cột nguồn |
| T26 | docs/experiments/EXPERIMENTS_UPDATED.md | 702 | Historical training/probes trước accumulation fix;không matched corrected cohort |
| T27 | docs/experiments/EXPERIMENTS_UPDATED.md | 946 | Historical V4/V5 verified interventions/probes;không train-time causal ablation |
| T28 | docs/experiments/EXPERIMENTS_UPDATED.md | 964 | Historical V4/V5 verified interventions/probes;không train-time causal ablation |
| T29 | docs/experiments/EXPERIMENTS_UPDATED.md | 984 | Historical V4/V5 verified interventions/probes;không train-time causal ablation |
| T30 | docs/experiments/EXPERIMENTS_UPDATED.md | 1061 | Historical V4/V5 verified interventions/probes;không train-time causal ablation |
| T31 | docs/experiments/EXPERIMENTS_UPDATED.md | 1074 | Historical V4/V5 verified interventions/probes;không train-time causal ablation |
| T32 | docs/experiments/EXPERIMENTS_UPDATED.md | 1097 | Historical V4/V5 verified interventions/probes;không train-time causal ablation |
| T33 | docs/experiments/EXPERIMENTS_UPDATED.md | 1112 | Historical V4/V5 verified interventions/probes;không train-time causal ablation |
| T36 | docs/research/archive/20261005_before_consolidation/AMIS_WARMUP_V2_RESULTS.md | 18 | Warmup20%historical,validation-only3seeds;NO-GO cho schedule này,không cho toàn bộ corrected method |
| T37 | docs/research/archive/20261005_before_consolidation/AMIS_WARMUP_V2_RESULTS.md | 29 | Warmup20%historical,validation-only3seeds;NO-GO cho schedule này,không cho toàn bộ corrected method |
| T38 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 90 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T39 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 101 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T40 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 545 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T41 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 557 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T42 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 762 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T43 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 823 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T44 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 1294 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T45 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 1749 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T46 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 1771 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |
| T47 | docs/operations/CONFIGURATION_SEARCH_HANDOFF.md | 1798 | Historical configuration search;đọc header split/beam;không current GO state hoặc corrected tuning |


| Nguồn chính | SHA256 |
| --- | --- |
| [four_way_scores.csv](../../outputs_rebuttal/amis_corrected_four_way_20261005/four_way_scores.csv) | 5df1c0af18e583ad2100b854b5a0a66d8d02fc82c80dd64799122aedcfa6d2d7 |
| [four_way_summary.json](../../outputs_rebuttal/amis_corrected_four_way_20261005/four_way_summary.json) | 444ddc22eb3918d31ecaea6f7eb9ff0ec0ac0df633d7877e50ba2761d2f04a73 |
| [lsr_grid_decision.json](../../outputs_rebuttal/amis_lsr_l030_validation_20261005/lsr_grid_decision.json) | a9f613bdaa6112fd0d444a22757529b755d367d5445e045578eb7ce91066ffd9 |
| [cost_benchmark_v2.json](../../outputs_rebuttal/amis_cost_benchmark_20261006_v2/cost_benchmark_v2.json) | b3bb4d582c740e3fee8a217c306b0afb1958a4a6e9b76d8964bac976e074912c |
| [EXPERIMENTS_UPDATED.md](EXPERIMENTS_UPDATED.md) | 7b4323d4bd658f25f8dd9da6de28771904061121845f9ebb349f7fbb329bb8c7 |
| [CONFIGURATION_SEARCH_HANDOFF.md](../operations/CONFIGURATION_SEARCH_HANDOFF.md) | 0444164e3836dd088b56a22b6efee970e8482359833706048752f114e444ec09 |
| [NEXT_EXPERIMENTS_HANDOFF_20261005.md](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md) | 864309eb77846a5ae7dafacd6c6cab92463e19adf46651a715146ae0246c9bd7 |
| [EXPERIMENTS_ORIGINAL_20261003.md](../archive/EXPERIMENTS_ORIGINAL_20261003.md) | b07eac7e6f2382dbfa843fd027d0e8a4503987e6a292e43693af25ea3ca68d6e |


**Kiểm tra bảo toàn ở lần tổng hợp ban đầu:** 143 file có sẵn (README và toàn bộdocs) giữ nguyên SHA256 trước/sau quá trình đọc/tổng hợp. Chỉ tạo/sửa file tổng hợp này. Snapshot chứa 51 bảng, bao gồm inventory/source tables; chưa chạy thêm thực nghiệm. Raw outputs chỉ được đọc.



**Redesign 06/10/2026:** chỉ cập nhật file master; main comparison chuyển lên đầu, các bảng chi tiết xuống Appendix A–K. 142 files docs khác không đổi SHA256.
