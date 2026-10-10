# Pilot ảnh hưởng accumulation — LSR và Full, seed42

## Material Passport

- Ngày05/10/2026, Asia/Saigon; workflow academic-research-suite / experiment-agent.
- Phạm vi được user cho chạy: **hai pilot** trên hai Colab A100 Standard, không mở12-run cohort hoặc Turkish.
- Trạng thái cuối: **TWO_PILOTS_VERIFIED**. Source/data hash khóa; GPU optimization/lifecycle gates PASS trên từng máy trước train. Metrics được tính lại độc lập từ predictions, checkpoint HF và metadata đã kiểm checksum; cả hai GPU đã tắt sau khi tải artifacts về.
- Source: `scripts/revalidation/run_accumulation_pilot.py`; manifest `outputs_rebuttal/amis_accumulation_pilot_20261005/bundle_manifest.json`.

## Câu hỏi và cấu hình đã khóa

## Kết quả cuối và quyết định tiếp

| Seed42, mBART Amis | Test BLEU cũ | Test BLEU sửa accumulation | Δ BLEU | chrF++ cũ | chrF++ mới | Best epoch mới |
|---|---:|---:|---:|---:|---:|---:|
| LSR-only λ=.10 | 19.786496 | 19.095440 | −0.691056 | 18.679745 | 18.434505 | 14 |
| Full α=.10, λ=.10 | 19.956574 | 20.321295 | +0.364721 | 18.887811 | 19.014707 | 20 |

LSR dừng ở epoch18 theo patience4, chọn checkpoint504/epoch14; Full hoàn thành20 epochs, chọn checkpoint720/epoch20. Cả hai chọn best bằng validation chrF++, test được đánh giá sau khi chọn. Validation chrF++ LSR tăng14.167406→14.190174; Full giảm14.731668→14.477980. Vì vậy không kết luận sửa accumulation cải thiện mọi split. Test hypotheses thay đổi442/575 (LSR) và400/575 (Full); đây không phải tái lập chính xác số cũ.

Full−LSR test BLEU tăng từ+0.170078 lên+1.225855; chrF++ chênh mới+0.580202. So với verified baseline42 (BLEU19.728081, chrF++18.713838), Full mới tăng+0.593214 BLEU/+0.300869 chrF++; LSR mới giảm−0.632641 BLEU. **Tín hiệu seed42 có lợi cho Full, bất lợi cho LSR-only.** Chưa đủ bằng chứng để kết luận ổn định nhiều seed, ý nghĩa thống kê, hay yêu cầu đổi phương pháp. Số lịch sử vẫn là metrics của checkpoint cũ; không sửa hoặc ghi đè chúng bằng số mới.

**GO đề xuất:** chạy LSR-only λ=.10 và Full α=.10/λ=.10 ở seeds43/44 (bốn runs), giữ recipe, selection, references và generation protocol; reuse audited baseline43/44. Đây là bước xác minh độ ổn định, chưa được launch trong phạm vi hai pilot. Không tune theo test pilot. Nếu Full vượt LSR/baseline ổn định nhiều seed thì tiếp tục kiểm tra đóng góp CLRR-only và generalization theo kế hoạch. Nếu gain không ổn định thì kiểm tra training curves/validation và cơ chế trước khi mở tối ưu phương pháp. Chưa có kết luận No-Go phương pháp từ một seed.

Evidence đầy đủ: `outputs_rebuttal/amis_accumulation_pilot_20261005/{lsr,full}/local_verification.json`, `pilot_comparison.json`, `PILOT_COMPARISON.md`. Recomputed native metrics khớp worker; 14 source/data bundle hashes khớp; selected/exported tensors khớp; HF archive size/SHA và sáu primary metadata files của mỗi run đã verified. Best ZIPs trên `FiveC/amis-rewire-checkpoints`, prefix riêng `amis_accumulation_pilot_20261005`. Artifacts nhẹ đã tải về; hai Colab báo `Session terminated`. Không tải weights lớn về máy local.

## Thiết kế pilot lúc launch

Lỗi loss-kwargs làm gradient wrappers scale khác baseline trong accumulation4. AdamW và clipping có thể giảm ảnh hưởng; chưa biết corrected training sẽ đổi chất lượng bao nhiêu. Pilot kiểm tra mức thay đổi thực, không mặc định scores cũ sai và không mặc định phải train lại mọi nhánh.

| Colab | Pilot | Seed | α routing | λ LSR |
|---|---|---:|---:|---:|
| Account1 / colab | LSR-only (`jepa`) | 42 | 0 | .10 |
| Account2 / colab2 | Full (`jepa-clrr-enc`, detached source) | 42 | .10 | .10 |

Common recipe: pretrained mBART-50 revision`e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; Amis4600/576/575; tl_XX→zh_CN; LR5e-5, WD0, LRwarmup.06; batch32×accum4, evalbatch8; max source/target256;20epochs, patience4; BF16+gradient checkpointing, fused AdamW, default max_grad_norm1; data loader workers2. Routing strength cố định, **routing warmup0**. Fresh pretrained, no auto-resume.

Giữ validation-only training như old comparators: chọn best bằng native original-reference validationchrF++, greedybeam1, FP32 generation/TF32 off. Chỉ sau khi best đã chọn và archive thì load frozen best, đánh giá test một lần beam4, FP32, batch8. File test_metrics riêng không sửa trainingmetrics.test_evaluated=false.

Môi trường GPU: conda **clrr**, Python3.13.15; torch2.6.0+cu124, Transformers4.57.6, Accelerate1.15.0, SacreBLEU2.6.0. Production source khớp bundle đã qua full-size gate; rollout chạy lại16 tiny optimization cases và4 actual CLI lifecycle cases trên mỗi GPU.

## Mốc lịch sử dùng so sánh

| Comparator seed42 | Selected step/epoch theo primary metrics | Val BLEU / chrF++ | Test BLEU / chrF++ |
|---|---|---|---|
| Matched LSR λ=.10 | 576 /16 | 16.5687629 /14.1674057 | 19.7864956 /18.6797449 |
| Full α=.10 λ=.10 replica | 720 /20 | 17.6436514 /14.7316677 | 19.9565738 /18.8878112 |

Số Full validation ở bảng phải lấy full precision từ historical_comparators.json nếu dùng tiếp trong calculations; bảng chỉ tóm tắt. Hash-verified predictions576validation/575test được tính lại native metrics. Original test hypotheses đã lưu khớp replicas. V5 inventory hardcode LSR432 và Full468 là nhãn không có provenance; không dùng làm actual best chronology. Old scores giữ nguyên.

Comparator evidence: `outputs_rebuttal/amis_accumulation_pilot_20261005/historical_comparators.json`, `HISTORICAL_COMPARATORS.md`.

## Thu kết quả và backup

- HF repo`FiveC/amis-rewire-checkpoints`, prefix riêng **amis_accumulation_pilot_20261005**, hai run_names kết thúc`seed42-accumfixed`.
- Sau mỗi epoch upload trainer_state + training_receipt; final selected state lưu riêng. Final best ZIP phải khớp remote size/LFS SHA256.
- Check selected checkpoint tensors và best export bằng canonical weight digest. Upload metrics, test predictions, worker source và manifest; COMPLETE sau final provenance uploads.
- Metadata upload không phải full resumable backup. Optimizer/scheduler/RNG vẫn local; không hứa resume từ selected ZIP sau mất runtime.
- Chỉ tắt từng GPU sau khi xác minh upload và artifacts đã tải về. Nếu download chậm, dùng HF pinned revision để lấy primary records; không coi command đang chạy là download đã xong.
- Status local: `outputs_rebuttal/amis_accumulation_pilot_20261005/status/account1.json`, `account2.json`; user được báo mốc10phút và các thay đổi trạng thái.

## Đọc pilot sau khi xong

1. Verify config/source/data, max validation selection, best tensor export, aligned references, native metric recomputation, HF archive + metadata integrity.
2. So old/new val/test scores, selected epoch, hypotheses changed count và mọi historical trajectory records thực có. Không dựng lại history bị mất.
3. So Full−LSR trước/sau, cùng baseline42 đã xác minh. Không chọn λ/α/seed mới từ pilot test.
4. Nếu hypotheses và scores khớp: evidence empirical ổn cho hai selected seed42 runs, không tự chứng nhận mọi seed/method.
5. Nếu có khác biệt: báo độ lớn và hướng; xem trajectory và gradient norms/clipping. Chỉ mở rộng corrected cohort sau phân tích này.

Giới hạn: n=1seed/method không chứng nhận seed robustness hoặc paper superiority. Old exact training runtime receipt chưa đầy đủ; old/new difference không tự là causal estimate thuần của một source-line change. Không có equivalence margin hoặc threshold“đủ đẹp” được suy ra sau khi nhìn test.
