# Corrected results — archived evidence

Final evidence: corrected FT/CLRR/LSR/Full seeds42–44 completed. Mean test BLEU: 19.3886 / 19.3539 / 19.3930 / 20.1303. LSR .10 remains selected after the later {.03,.10,.30} validation grid; .03/.30 test stays blind. Historical recommendations below are not pending tasks. No scores were recomputed during this cleanup.

Current interpretation: [RESEARCH_INSIGHTS](../research/RESEARCH_INSIGHTS.md). Execution status: [current handoff](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md).

## Contents
- [ACCUMULATION_PILOT_20261005](#accumulation-pilot-20261005)
- [ACCUMULATION_CONFIRMATION_20261005](#accumulation-confirmation-20261005)
- [CORRECTED_CONTROLS_20261005](#corrected-controls-20261005)
- [CORRECTED_ACCUMULATION_SIGNIFICANCE_20261005](#corrected-accumulation-significance-20261005)

---

<a id="accumulation-pilot-20261005"></a>

## ACCUMULATION_PILOT_20261005

Original source: `docs/archive/CORRECTED_RESULTS_20261005.md#accumulation-pilot-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

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


---

<a id="accumulation-confirmation-20261005"></a>

## ACCUMULATION_CONFIRMATION_20261005

Original source: `docs/archive/CORRECTED_RESULTS_20261005.md#accumulation-confirmation-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

# Kiểm chứng accumulation: LSR/Full seeds43–44

## Material Passport

- Ngày05/10/2026, Asia/Saigon; academic-research-suite / experiment-agent.
- User cho chạy bốn runs sau hai pilot seed42. Trạng thái cuối: **FOUR_RUNS_VERIFIED**, gồm seed43 hai nhánh và seed44 recovery hai nhánh. Cả hai A100 đã tắt sau download/verification; stop receipts ghi `Session terminated`.
- Hai A100 Standard40GB, không high-mem; remote conda clrr và runtime giống pilot.
- Source production giữ nguyên; bundle16files có hashes khóa. Audit76CLI tokens khớp pilot sau khi đổi seed và namespace đầu ra.

## Kết quả cuối — corrected accumulation, seeds42–44

| Test BLEU | Seed42 | Seed43 | Seed44 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Baseline audited | 19.7281 | 18.1378 | 20.3000 | 19.3886 ± 1.1203 |
| LSR-only λ=.10 | 19.0954 | 19.8193 | 19.2644 | 19.3930 ± 0.3787 |
| Full α=.10 λ=.10 | 20.3213 | 20.1780 | 19.8916 | **20.1303 ± 0.2188** |
| Full−LSR | +1.2259 | +0.3587 | +0.6273 | **+0.7373** |
| Full−Baseline | +0.5932 | +2.0402 | −0.4084 | +0.7417 |

Mean test chrF++ LSR18.4626±0.1860, Full18.8570±0.1575; Full−LSR +0.3945, dương3/3seeds. Full−LSR validation BLEU +0.6126 và chrF++ +0.1185, cũng dương3/3. **GO tiếp tục kiểm chứng đóng góp Full so với matched LSRλ.10.** Chưa chứng minh vượt baseline mọi seed: seed44 Full thấp hơn baseline cả BLEU/chrF++; baseline43 chrF++13.1592 thấp rõ và làm mean baseline chrF++ gain lớn, nên báo paired differences, không chỉ mean. Baseline43 predictions đã được tái tính; chưa phát hiện lỗi đo từ việc score thấp.

So lịch sử đúng cấu hình, Full testBLEU tăng ở3/3seeds, mean+0.3616; LSR mean giảm−0.1413, dấu khác nhau theo seed. Full validation chrF++ giảm ở3/3seeds, mean−0.1571, dù test tốt hơn. Vì vậy số mới không tái lập số cũ và không có claim cải thiện mọi split. Số cũ giữ nguyên trong artifacts cũ; cohort corrected phải được ghi riêng và dùng nhất quán.

Đây là mô tả3seeds, chưa có significance/equivalence claim, chưa đủ kết luận ACL-ready. Không chọn α/λ theo test. **Bước tiếp đề xuất, chưa chạy:** CLRR-only corrected3seeds để hoàn thiện factorial ablation; xác minh LSR mạnh hơn/tuned theo validation với corrected accumulation trước claim vượt tunedLSR. Sau các gate đó mới mở generalization khi user cho phép. Chưa có căn cứ bắt buộc nâng cấp phương pháp từ cohort này.

Evidence: `outputs_rebuttal/amis_accumulation_confirmation_20261005/three_seed_comparison.json`, `THREE_SEED_COMPARISON.md`; local_verification seeds43 tại output này, seeds44 tại `outputs_rebuttal/amis_accumulation_confirmation_20261005_recovery/`. Every result replays native metrics using aligned original references; source/data hashes, selected/export tensors, best archive LFS SHA/size and6primaryHFmetadata checks PASS.

Backup: seed43 ở `FiveC/amis-rewire-checkpoints/amis_accumulation_confirmation_20261005`; seed44 recovery ở `FiveC/amis-accumulation-confirmation-20261005/amis_accumulation_confirmation_20261005_recovery`. Best weights đã upload/checksum và xóa khỏi output run trên GPU; failed-attempt metadata ZIPs đã tải local riêng. Không cần tải modelweights lớn về local. GPU1 tắt14:05, GPU2 tắt14:08 (Asia/Saigon); receipts ở recovery/watchers/account1.json và account2.json.

## Thiết kế và nhật ký lúc chạy

| Máy | Run đầu | Run tiếp | Mục tiêu |
|---|---|---|---|
| Account1 / colab | LSR-only λ=.10 seed43 | LSR-only λ=.10 seed44 | Kiểm tra kết quả LSR qua3seeds |
| Account2 / colab2 | Full α=.10 λ=.10 seed43 | Full α=.10 λ=.10 seed44 | Kiểm tra đóng góp Full qua3seeds |

Giữ recipe seed42: mBART revision e30b6cb8eb0d43a0b73cab73c7676b9863223a30; Amis4600/576/575; tl_XX→zh_CN; LR5e-5, weight_decay0, LRwarmup.06; batch32×accum4, evalbatch8; max lengths256;20epochs/patience4; BF16+checkpointing; workers2; routing fixed, warmup0; validation chrF++ beam1 chọn best; test frozen best beam4, FP32 generation/TF32off, references gốc. Fresh pretrained, không auto-resume.

Pilot seed42: LSR testBLEU19.095440/chrF++18.434505; Full20.321295/19.014707. Baseline42 verified19.728081/18.713838, baseline43/44 reuse theo audit. Không chọn hyperparameter theo test, không mở Turkish hoặc các nhánh khác.

## Backup và điều phối

HF repo `FiveC/amis-rewire-checkpoints`; prefix riêng **amis_accumulation_confirmation_20261005**. Run_names kết thúc seed43/44-accumfixed. Mỗi máy: GPUgates→run43→frozen test→upload→HF archive size/SHA+6metadata verification→đóng gói artifacts nhỏ→xóa weights trong outputroot đã kiểm tra→run44 theo cùng chuỗi. Chỉ xóa checkpoint sau receipt verified; nếu run/checksum fail, shell dừng trước run tiếp. Metadata từng epoch được upload; best ZIP là model export, không phải full optimizer/RNG resumable backup.

Không sửa số cũ. Artifacts mới ở `outputs_rebuttal/amis_accumulation_confirmation_20261005/`. Khi xong, tải `/content/confirmation_results_<family>_seed<seed>.zip`, tính lại metrics bằng `scratch/verify_accumulation_confirmation_local.py`, rồi tắt máy tương ứng. Chỉ báo VERIFIED sau khi metrics/predictions/references khớp và HF integrity PASS.

## Lệnh kiểm tra

Chạy local qua conda clrr; account1 dùng session colab, account2 session colab2:

```powershell
conda run --no-capture-output -n clrr python scratch/colab_pilot_control.py --account 1 exec -s colab -f scratch/monitor_accumulation_confirmation_remote.py --timeout 45
conda run --no-capture-output -n clrr python scratch/colab_pilot_control.py --account 2 exec -s colab2 -f scratch/monitor_accumulation_confirmation_remote.py --timeout 45
```

Sau download xong:

```powershell
conda run --no-capture-output -n clrr python scratch/verify_accumulation_confirmation_local.py --family lsr --seed 43
```

Đổi family/seed cho ba outputs còn lại. Không launch lại khi namespace đã tồn tại.

## Sự cố backup seed44 và recovery

Khoảng13:31, cả hai seed44 bị gián đoạn do HF429: repo cũ đạt giới hạn128commits/giờ; callback on_save assert upload metadata. LSR lưu checkpoint216/epoch6, Full180/epoch5 trước lỗi. Không có final test seed44; không dùng chúng làm kết quả nghiên cứu. Seed43 hai nhánh đã VERIFIED và giữ nguyên. Watcher cũ đã dừng; recovery dùng worker/namespace riêng, giảm commits và repo backup riêng dướiFiveC. Recipe nghiên cứu giữ nguyên; chạy lại seed44 từ pretrained để giữ protocol fresh start, không tự gộp trajectory hai attempts. Không xóa attempt bị lỗi trước khi lưu provenance.

## Quyết định sau kết quả

### Mốc13:23

Seed43 hai nhánh đã VERIFIED local: LSR BLEU19.8192509251/chrF++18.6609992118, Full20.1779951721/18.8567741443; Full−LSR +0.3587442470 BLEU. HF archive+metadata verified, weights remote được dọn sau checksum; hai máy đang chạy seed44. Chưa kết luận cohort từ2seeds.

Watcher local đang chạy hidden bằng Python của conda clrr: `scratch/watch_accumulation_confirmation.py --account 1|2`, poll60giây, state/log trong `outputs_rebuttal/amis_accumulation_confirmation_20261005/watchers/`. Chỉ tắt GPU tương ứng sau khi cả hai seed VERIFIED local và remote queue DONE. Nếu verifier fail hoặc tới6giờ, ghi manual attention và giữ runtime; không restart training. Watcher phụ thuộc máy local còn bật/kết nối mạng; training/backup queue chạy remote độc lập.

Tổng hợp paired differences Full−LSR và Full−baseline ở từng seed, mean/SD3seeds; báo cả validation/test và selection epochs. Không suy luận ổn định chỉ từ mean dương, không coi3seeds đủ chứng minh mọi claim. Nếu lợi ích nhất quán, tiếp tục ablation CLRR-only/generalization theo kế hoạch; nếu không nhất quán, phân tích validation/training và cơ chế trước tối ưu phương pháp.


---

<a id="corrected-controls-20261005"></a>

## CORRECTED_CONTROLS_20261005

Original source: `docs/archive/CORRECTED_RESULTS_20261005.md#corrected-controls-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

# Corrected controls — 05/10/2026

## Final status — checked 17:00, Asia/Saigon

All six trainings completed. Independent local prediction replay and remote archive/metadata checks passed. Colab 2 stopped at 16:04:35; Colab 1 stopped at 16:10:08, with explicit session termination receipts. The execution notes below describe launch-time status.

| Seed | CLRR-only test BLEU | CLRR-only test chrF++ | LSR lambda .03 validation chrF++ |
|---|---:|---:|---:|
| 42 | 19.223095 | 18.362492 | 13.906081 |
| 43 | 19.492997 | 18.531956 | 14.066061 |
| 44 | 19.345498 | 18.374130 | 14.178781 |

LSR lambda .03 mean validation chrF++ is **14.050307473361284**, below lambda .10 **14.221997785916189**, and lower at every matched seed. The validation decision selects **lambda .10**. Lambda .03 test predictions remain blind, as planned; this is a completed tuning experiment.

CLRR-only mean test BLEU is approximately **19.3539**, versus corrected baseline **19.3886**, LSR lambda .10 **19.3930**, and Full **20.1303**. CLRR alone does not improve mean BLEU here; Full exceeds CLRR at all three seeds and exceeds LSR lambda .10 at all three seeds. This supports investigating the combination, but does not establish statistically significant superiority or ACL readiness. Prior corrected Full-versus-LSR statistical caveats remain applicable.

All six model backups are in the two repositories listed below. Small artifacts and verification receipts are local under `outputs_rebuttal/amis_corrected_controls_20261005/`. Old experiment scores remain unchanged.

## Scope and purpose

Six authorized mBART Amis runs, seeds 42/43/44. This completes the corrected G0 controls before deciding whether to upgrade the method.

| GPU | Family | Question | Backup repository |
|---|---|---|---|
| Colab 1, A100 Standard 40GB | LSR-only, lambda .03 | Does Full beat an LSR control selected using validation? | FiveC/amis-corrected-lsr-l003-20261005 |
| Colab 2, A100 Standard 40GB | CLRR-only, alpha .10, lambda 0 | What does rewiring contribute without LSR? | FiveC/amis-corrected-clrr-only-20261005 |

Launch-time PIDs were3940/7642; both jobs subsequently completed and were verified, as recorded in Final status above.

## Locked protocol

Same corrected source and data as the preceding cohort. Model revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; LR 5e-5, batch 32, accumulation 4, BF16, gradient checkpointing, lengths 256, maximum 20 epochs, early stopping patience 4, warmup .06, weight decay 0. Validation chrF++ with greedy decoding selects checkpoint; frozen test uses beam 4, FP32 generation, Chinese BLEU tokenization. No automatic resume.

Bundle SHA256: `e6889d3356e18d59b24700d843a59c3558d35a92206dd531e364a10fc4ff872b`. Per-file hashes are in `outputs_rebuttal/amis_corrected_controls_20261005/bundle_manifest.json`. Do not regenerate using the stale initial scaffolding script `scratch/build_corrected_controls.py`.

## Validation selection and test access

Train all three lambda .03 seeds with validation only. Compare their mean validation chrF++ with corrected lambda .10 scores: 14.190173628560752, 14.274377140832009, 14.201442588355809. Lock and upload `lsr_selection_decision.json` before test access. A strict validation win selects .03; a tie selects .10. Only test .03 if selected; otherwise keep all .03 test predictions blind.

CLRR-only has a fixed configuration and receives test evaluation after its validation checkpoint selection. Test results do not select a configuration.

## Preservation and completion

Each run must upload its selected model archive and selection provenance, verify HF checksums, collect small artifacts locally and replay metrics from predictions. Only then delete checkpoint weights on Colab. Keep lambda .03 model archives until the validation decision and any required test evaluations finish.

Stop each GPU independently after its job finishes and all required local and HF verification passes. Connection failure requires retrying monitoring, not restarting training. Old experiment numbers remain unchanged. The setup above is historical; final verified results are recorded at the start of this section.


---

<a id="corrected-accumulation-significance-20261005"></a>

## CORRECTED_ACCUMULATION_SIGNIFICANCE_20261005

Original source: `docs/archive/CORRECTED_RESULTS_20261005.md#corrected-accumulation-significance-20261005`. Status: historical record; the summary at the top and the current handoff take precedence.

# Ý nghĩa thống kê và bước tiếp — corrected accumulation

## Material Passport

- Ngày05/10/2026, Asia/Saigon; academic-research-suite, validate mode.
- 9 predictions sets đã verified: baseline/LSR/Full seeds42–44,575test rows.
- Phân tích hậu nghiệm, không chọn model/seed/hyperparameter; chỉ CPU trong conda clrr, không mở GPU.
- 10000paired bootstrap draws, RNG42, BLEU tokenizezh và chrF++word_order2; p-values tái lập chính xác public SacreBLEU PairedTest API ở cả12tests.
- Evidence: `outputs_rebuttal/amis_accumulation_confirmation_20261005/significance_corrected.json`, `SIGNIFICANCE_CORRECTED.md`; script `scripts/revalidation/test_corrected_accumulation_significance.py`.

## Full so với matched LSR λ=.10

| Seed | Δ testBLEU | p native bootstrap | p Holm trên12tests |
|---|---:|---:|---:|
| 42 | +1.2259 | 0.021598 | 0.237576 |
| 43 | +0.3587 | 0.161084 | 0.863014 |
| 44 | +0.6273 | 0.095890 | 0.863014 |

Seed42 đạt p<.05 trước hiệu chỉnh, không còn sauHolm;43/44 chưa đạt. Cả3chrF++ comparisons cũng chưa đạt. Family12 được khóa trong script trước khi xem p-values:3seeds×2comparators(LSR/baseline)×2metrics. Không dùng significance của một seed để kết luận cả phương pháp.

Giữ cố định3pairs checkpoints, dùng cùng sentence indices cho mọi seed, mean Full−LSR BLEU **+0.7373**, percentile95%CI **[+0.0403,+1.4557]**. Đây là tín hiệu conditional trên tập câu cho trung bình các checkpoint này, CI chưa hiệu chỉnh multiplicity và không chứa bất định training. Mean chrF++ +0.3945, CI[−0.0392,+0.8657] bao gồm0. Không coi3×575 là1725quan sát độc lập.

Đổi đơn vị suy luận sang training seeds: exact two-sided paired sign-flip cho Full−LSR p=.25 trên cảBLEU/chrF++; chỉ3seeds, p nhỏ nhất có thể đạt là.25. Kiểm định giả định symmetric exchangeability của paired effects dưới null, khó kiểm tra khi n=3. **Chưa có kết luận significance bao gồm training variability; không có nghĩa đã chứng minh hiệu ứng bằng0.**

## Full so với baseline

Chỉseed43BLEU còn p<.05 sauHolm12: p=.001700→.020398. Seed42/44 không đạt. MeanBLEU+0.7417 có conditional sentenceCI[−0.0265,+1.5182], chứa0; exactseed sign-flip p=.50. Baseline43 chrF++ thấp, gainmeanchrF++ bị ảnh hưởng lớn từseed này.

Một số percentileCI và native bootstrap p không cho cùng quyết định ngưỡng, đặc biệt baseline43chrF++ (CI dương nhưng p=.106689). Chúng là các thủ tục bootstrap khác nhau: percentileCI của signed differences và SacreBLEU/Moses centered absolute-difference statistic; không đảo CI thành p-value hoặc chọn thủ tục cho số thuận lợi. Báo nguyên cảhai, dùng native p+Holm cho table significance. Điều này không phải lỗi tính metrics.

## Kết luận và kế hoạch đề xuất

**GO kiểm chứng tiếp, chưa claim superiority có ý nghĩa thống kê vững qua seeds.** Tín hiệu Full−matchedLSR nhất quán và meanBLEU CIconditional dương, nhưng uncertainty/strongestLSR/generalization chưa chốt. Không cần đổi phương pháp ngay.

1. **LSR-only λ=.03 seeds42–44**, giữ corrected accumulation và recipe; λ=.03 là ứng viên từ validation lịch sử, không chọn từ test mới. Chạy validation-only, so vớiλ.10 đã có để chốt LSRcontrol theo validation trước khi mở final test. Không gọiλ.03 tự động là optimum sau khi sửa accumulation.
2. **CLRR-only seeds42–44** cùng recipe để hoàn thiện factorial ablation và kiểm tra đóng góp độc lập. Nếu cho chạy, hai A100 có thể chạy nhómLSR và nhómCLRRsong song; chưa launch.
3. Chốt Full và comparatorLSR mạnh nhất theo validation rồi thêm paired seeds được chọn trước, hướng tới tối thiểu6seeds tổng nếu cần exactsign-flip có khả năng đạt p<.05. Sáuseeds chỉ cho phép kiểm định đạt ngưỡng, không đảm bảo significance. Báo paired effects, CIs và kế hoạch power; không chạy thêm seed tới khi p vừa đạt rồi dừng.
4. Nếu đóng góp mất đi trước corrected tunedLSR, phân tích curves/gradient/routing trước tối ưu α/λ/cơ chế; nếu còn ổn định thì mở generalization khi user cho phép. Chưa chạy Turkish.

## Kiểm tra ngụy biện — 11/11

| Loại | Kiểm tra / giới hạn |
|---|---|
| Simpson | Báo từngseed; Full−LSR cùngdấu3/3, Full−baseline2/3 |
| Ecological | Sentence uncertainty tách trainingseed uncertainty |
| Berkson | Không loại seed/câu dựa trên điểm; corpus Amis chỉ là một population |
| Collider | Không điều kiện hóa trên scores/test outcomes |
| Base rate | Không có bài toán prevalence/classification; không áp dụng |
| Regression to mean | Không chọn nhóm theo score thấp; không dùng cải thiện baseline43 làm bằng chứng chung |
| Survivorship | Hai failed44 do quota được lưu và chạy lại fresh; mọi finalizedseed đều báo |
| Look elsewhere | Báo đủ12tests vàHolm; đây chỉ family hiện tại, không phủ toàn bộ historical search |
| Forking paths | Post-hoc analysis được ghi rõ; future protocol đóng băng trước chạy |
| Causal overclaim | Controlled method contrast không chứng minh cơ chế/rank/anti-collapse |
| Reverse causality | Không phải dữ liệu cross-sectional; không có suy luận đảo chiều |

Phương pháp MT: [SacreBLEU](https://github.com/mjpost/sacrebleu), [Koehn2004](https://aclanthology.org/W04-3250/). Sentence-level exchangeability có thể quá lạc quan nếu corpus có tương quan theo document; chưa có document-cluster bootstrap. Chưa có claim novelty hoặc ACL acceptance từ kiểm định này.

