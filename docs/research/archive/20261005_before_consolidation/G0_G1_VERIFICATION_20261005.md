# G0/G1 — kiểm chứng trước nâng cấp phương pháp

## Material Passport

- Ngày: 05/10/2026, Asia/Saigon.
- Workflow: academic-research-suite / experiment-agent, validate và diagnostic execution.
- Phạm vi: G0, D1, D3, D4; không train research models, không chạy Turkish.
- Verification Status: **TECHNICAL_HOLD cho so sánh training lịch sử; bản sửa accumulation qua synthetic regression.**
- Nguồn: [local audit](G0_G1_LOCAL_AUDIT_20261005.md), [artifacts mới](../../../../outputs_rebuttal/amis_g0_g1_20261005/), source hiện tại và archived AST, primary literature.

## 1. Lỗi thực đã phát hiện

Với Transformers 4.57.6, Trainer coi `forward(**kwargs)` là nhận loss kwargs trừ khi model khai báo khác. Hai wrappers `CrossLayerResidualRewire` và `JEPAGuidedSeq2SeqLM` bỏ `num_items_in_batch`, trả mean loss của microbatch. Trainer vì thế bỏ phép chia gradient accumulation khi token count được cung cấp. Baseline mBART không nhận kwargs này và được chia đúng.

Đã kiểm chứng bằng **một synthetic SGD update**, tiny mBART, FP32 CPU, dropout=0, bốn examples có số label tokens bằng nhau; so batch4/accum1 với batch1/accum4. Đây là correctness test, không train mô hình nghiên cứu.

| Family | Tỷ lệ norm update accum4/batch4 trước sửa | Sau sửa | Kết luận |
|---|---:|---:|---|
| Baseline | 1.0000 | 1.0000 | PASS |
| CLRR-only | 4.0000 | 1.0000 | Bug được tái hiện và sửa |
| LSR-only | 4.0000 | 1.0000 | Bug được tái hiện và sửa |
| Full | 4.0000 | 1.0000 | Bug được tái hiện và sửa |

Sau sửa, relative update differences dưới 3e-7. Receipts: [trước sửa](../../../../outputs_rebuttal/amis_g0_g1_20261005/accumulation/receipt.json), [sau sửa](../../../../outputs_rebuttal/amis_g0_g1_20261005/accumulation_fixed/receipt.json).

Sửa tối thiểu: khai báo `accepts_loss_kwargs = False` trên cả hai wrapper để Trainer chuẩn hóa mean loss theo số microbatches. Attached-source wrapper kế thừa CLRR cũng nhận khai báo này. Forward/inference, weights và metrics đã lưu không thay đổi.

### Giới hạn và impact

- **Không kết luận LR AdamW tăng 4 lần.** Scaling gradient có tác động khác với SGD vì Adam moments, epsilon, clipping và precision. Cần corrected-training cohort để biết quality thay đổi thế nào.
- Mean CE của từng microbatch được average theo accumulation. Với số label tokens khác nhau, không mặc định tương đương global token-weighted loss của một batch lớn; regression bằng equal-token examples cô lập lỗi factor4.
- Archived AST xác nhận historical wrappers có cùng pattern bỏ token count. Runtime/Trainer của từng cohort vẫn phải được truy vết; không tự gán bug cho mọi backbone và mọi PEFT run.
- Amis C/confirmation/detach/warmup dùng wrappers và accumulation4 dưới pinned runtime là các cohort cần đánh dấu affected/pending repair. FT baseline không mắc cơ chế lỗi này trong regression; khả năng reuse còn phụ thuộc initialization, shuffle, selection và recipe phù hợp.
- CLRR/LSR cùng bị scale trong test. So giữa hai wrapped methods vẫn bị ảnh hưởng bởi optimizer/clipping; không kết luận mọi relative ranking sai hoặc mọi ranking giữ nguyên.
- BLEU/chrF của outputs cũ vẫn là phép đo hợp lệ của đúng checkpoint cũ. Không đổi số hoặc gọi chúng là kết quả train đã sửa.
- Không tiếp tục G2 bằng code đã sửa rồi so với wrapped controls cũ như một matched study. Phải tạo cohort được chuẩn hóa cho affected branches trước.

## 2. D1/D4 — những gì đã xác nhận

- Sáu warmup chọn đúng max validation chrF++; đồng thời cũng là max BLEU trong history.
- Full best epochs 20/20/18; CLRR-only 18/8/20. Chỉ CLRR43 early-stop epoch12, best epoch8. Không có bằng chứng mọi runs bị stopping quá sớm.
- Routing alpha schedule đúng; evaluation alpha=.10. Selected-best records/hashes và native metrics khớp.
- Warmup không tốt hơn không phải do nhầm endpoint với best; interpretation quality vẫn bị giới hạn bởi lỗi optimization mới phát hiện.
- V5 length bins đảo dấu giữa validation/test, không đủ chọn một nâng cấp chuyên trị câu dài.
- D2 warmup routing-off không cần để xác nhận selection; chưa chạy vì không giải quyết technical hold hiện tại.

## 3. D3 — target-anchor counterfactual

Đã hoàn thành trên một A100 Standard, không high-mem. Ba fixed Full selected checkpoints tải bằng pinned HF revision, xác minh size/SHA256 và strict weight loading. Dùng cùng tám batches × bốn validation examples như V4.

Source giữ routing α=.10; target pass lần lượt routing .10 và raw .0, cùng source graph/CE, target detached và caches độc lập. Đo anchor cosine/L2, LSR loss và CE–LSR gradients. FP32/eval/dropout-off/TF32-off, không optimizer updates. Đây là counterfactual characterization của weights cũ, không chứng minh training gain của U1.

| Seed | Cosine anchor routed/raw | LSR routed | LSR raw | Cosine hai gradients LSR | CE–LSR cosine routed / raw |
|---|---:|---:|---:|---:|---|
| 42 | .97307 | .19972 | .20425 | .95928 | .04718 / .03044 |
| 43 | .97249 | .20201 | .20744 | .95617 | .05897 / .04180 |
| 44 | .97283 | .20542 | .21106 | .95970 | .05845 / .04239 |

Số là mean qua tám fixed batches cho từng checkpoint. Tắt target routing đổi anchor và gradient, nhưng routed anchor có LSR loss thấp hơn và CE–LSR cosine cao hơn trong cả ba seeds. Không có evidence từ probe này rằng raw anchor giảm global gradient conflict. Điều đó không chứng minh raw anchor kém về quality, nhưng cũng không đủ khóa U1 như một giải pháp đã có cơ sở mạnh. LSR loss thấp hơn không tự chứng minh translation tốt hơn.

Collection SHA256, cả ba archive size/SHA256 và cache/target-detach checks PASS. [Summary](../../../../outputs_rebuttal/amis_g0_g1_20261005/verification_summary.json). Regression bản sửa trên GPU pinned cũng PASS cả bốn families, relative update differences dưới 3e-7: [receipt](../../../../outputs_rebuttal/amis_g0_g1_20261005/accumulation_gpu_fixed/receipt.json). Không có research optimizer steps trong D3; synthetic regression thực hiện một tiny update mỗi điều kiện.

A100 đã tắt sau khi thu/audit kết quả: Colab CLI xác nhận `Session terminated`. Lần stop đầu gặp DNS lỗi, lần thử lại thành công. Chỉ dùng một GPU vì D3 ba checkpoints nhỏ chạy tuần tự đủ; không cấp runtime thứ hai.

## 4. G0 novelty — đối chiếu công thức gần

| Primary source | Công thức/cấu trúc đã đọc | Khác với CLRR hiện tại | Việc cần chứng minh |
|---|---|---|---|
| [SkipCLM, §3](https://aclanthology.org/2025.naacl-srw.50.pdf) | Skip learned linear+ReLU, scaling warmup; InfoNCE trên decoder XGLM | CLRR identity fixed/detached encoder, nhiều block, LSR shared-encoder cosine | Skip+alignment không đủ làm novelty; cần ích lợi của rule cụ thể |
| [Value Residual Learning, §3.2](https://aclanthology.org/2025.acl-long.1375.pdf) | ResFormer trộn first-layer value với current value trước attention; có constants và learned scalars | CLRR cộng hidden block output sau whole block, source cách d và detached | Không claim constant/zero-added routing tự nó chưa từng có; kiểm tra location/source/gradient choices |
| [ResiDual, §2.2](https://arxiv.org/html/2304.14802v1) | Hai residual streams, cuối stack cộng Post-LN output và normalized dual sum | CLRR không đổi backbone LN thành dual streams, cộng activation xa tại mỗi block | Không mượn collapse guarantees của ResiDual cho CLRR |

Đây là bounded formula audit, không phải chứng nhận novelty toàn bộ literature. Exact reproduction comparator cần official code/version và matching recipe trước khi triển khai. U1/U2 còn là giả thuyết; chưa khóa candidate khi technical hold chưa xử lý.

## 5. Statistical fallacy scan — 11/11 đã xét

| Loại | Kết luận trong phạm vi audit |
|---|---|
| Simpson's paradox | Các length bins và splits khác dấu; không gộp thành một cơ chế phổ quát |
| Ecological fallacy | Mean checkpoints không chứng minh lợi ích cho từng câu hoặc morphology |
| Berkson's paradox | Selected-best và tuned candidates là mẫu được chọn; không đại diện mọi cấu hình |
| Collider bias | Không regression điều kiện hóa quality vào rank/loss đã chọn; không làm causal inference từ selected checkpoints |
| Base-rate neglect | Không có classification/sensitivity/PPV claim; không áp dụng |
| Regression to mean | Không gọi một seed thắng hoặc tuned winner là cải thiện ổn định; cần cohort xác nhận |
| Survivorship bias | Sáu warmup đều được report, gồm early-stop seed43; metadata/weights thiếu ghi unknown |
| Look-elsewhere effect | Không tìm subgroup/metric thắng mới; giữ các statistical families lịch sử |
| Forking paths | Lỗi được ghi riêng, không đổi metric/selection hậu nghiệm; recipe đã sửa là cohort mới |
| Correlation vs causation | Counterfactual anchor là thay đổi tính toán, không chứng minh training translation gain |
| Reverse causality | Không suy geometry tạo quality hoặc quality tạo geometry từ các endpoint quan sát |

Không thực hiện kiểm định significance mới cho diagnostics; tám batches trong một checkpoint không phải tám independent training seeds.

## 6. Quyết định tiếp theo

1. D3 đã hoàn thành và lưu số riêng; không cần lặp warmup on/off để sửa selection.
2. Correction đã qua runtime GPU pinned; archived wrapper pattern được chứng nhận. Tiếp tục impact matrix theo từng historical cohort, không tự invalidate mọi PEFT run.
3. Chốt corrected control cohort trước khi train nâng cấp. Không dùng số tuned LSR cũ như mốc tối ưu của recipe đã sửa.
4. Sau đó mới đánh giá có cần U1/U2. Chưa có bằng chứng để kết luận phải bỏ CLRR hoặc LSR vì quality hiện tại.

Không train lại research models trong lượt kiểm tra này. Không đổi manuscript hoặc ghi đè historical experiment numbers.

### Cohort sửa lỗi đề xuất, chưa launch

Để có mốc sạch trước khi sửa phương pháp, có thể chạy lại năm cấu hình gốc FT / CLRR-only .10 / LSR-only .10 / LSR-only .03 / Full .10/.10, seeds42/43/44: **15 runs**, cùng code/runtime/selection đã khóa. Việc chạy lại FT giúp có full receipt của cohort mới; chỉ giảm số runs khi G0 chứng nhận reuse đủ. Đây là đề xuất scientific repair, không phải training đã được thực hiện trong lượt diagnostic này.

λ=.03 tiếp tục là một configuration đối sánh đã định trước, không mặc định vẫn là tuned optimum sau sửa lỗi. Re-tuning cần validation-only và budget công bố. Những thresholds cũ dựa trên wrapped controls chưa sửa phải được gắn lại với corrected controls trước pilot nâng cấp. Nếu corrected four-way study đã đủ tốt, có thể xác nhận nó trước khi cần đổi kiến trúc.
