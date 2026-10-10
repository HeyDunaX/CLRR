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
