# Insight mới — corrected accumulation, mBART Amis

## Material Passport

- Ngày05/10/2026, Asia/Saigon; học thuật / phân tích mô tả.
- Evidence: hai pilot42 và bốn runs43–44 đã VERIFIED, baseline audited42–44; không sửa số lịch sử.
- Chi tiết: [runbook và kết quả](ACCUMULATION_CONFIRMATION_20261005.md); raw comparison `outputs_rebuttal/amis_accumulation_confirmation_20261005/three_seed_comparison.json`.

## Có lợi

Fullα.10/λ.10 vượt matchedLSRλ.10 ở3/3seeds trên cả validation/test BLEU và chrF++. Test meanBLEU20.1303±0.2188 so LSR19.3930±0.3787, paired mean+0.7373; testchrF++ mean+0.3945. Đây là bằng chứng mới ủng hộ lợi ích thêm CLRR vào LSR ở cấu hình cố định này. Full test cũng cao hơn historicalFull đúng cấu hình ở3/3seeds, mean+0.3616BLEU/+0.2005chrF++.

## Những điều chưa chốt

- Full chỉ vượt baseline ở2/3seeds; seed44−0.4084BLEU. Mean+0.7417 có ảnh hưởng lớn từ baseline43 thấp. Không gọi superiority ổn định trước mọi baseline seed.
- LSR-only corrected meanBLEU gần baseline (+0.0044), dấu khác nhau theo seed. LSR-only contribution độc lập chưa thuyết phục ởλ.10.
- Full validation chrF++ thấp hơn historicalFull ở3/3seeds, dù test tốt hơn. Không suy luận sửa lỗi cải thiện mọi split hoặc đã giải quyết generalization.
- ComparatorLSR là matchedλ.10; chưa có kết luận vượt tunedLSR mạnh nhất dưới corrected training. Các so sánh tunedLSR lịch sử chạy trước sửa accumulation giữ giá trị mô tả checkpoint cũ, không thay corrected tuned control.
- Chưa có significance claim, chưa có CLRR-only corrected cohort và chưa thêm ngôn ngữ. Chưa chứng minh cơ chế/novelty hoặc đạt tiêu chuẩn nhận bài chỉ từ bảng này.

## Quyết định

**Bổ sung thống kê:** [phân tích significance](CORRECTED_ACCUMULATION_SIGNIFICANCE_20261005.md). MeanFull−LSRBLEU CI95%conditional sentence[+0.0403,+1.4557], nhưng per-seed native p+Holm12 không có Full−LSRtest nào đạt. Exactpairedseed sign-flip p=.25 với3seeds; chưa chứng minh significance qua training variability. Giữ quyết định GO kiểm chứng tiếp, không viết claim significant/robust superiority từ3seeds. Đề xuất chốt correctedLSR theo validation vàCLRR-only trước khi mở cohort lớn.

**GO cho kiểm chứng tiếp; chưa No-Go phương pháp.** Tín hiệu Full−LSR nhất quán3seeds đủ lý do hoàn thiện ablation. Ưu tiên CLRR-only corrected seeds42–44; sau đó correctedLSR ởλ có cơ sở validation mạnh nhất trong kế hoạch cũ, không chọnλ từ test mới. Dùng validation để quyết định cấu hình, đóng băng trước test. Chỉ chuyển tối ưu phương pháp nếu controls mới làm mất lợi ích hoặc chỉ ra lỗi cơ chế; không sửa phương pháp chỉ vì một seed thua baseline. Turkish vẫn chưa khởi chạy.

## Sự cố vận hành đã xử lý

Seed44 attempts đầu chết do HF429 callback upload, không phải loss/OOM. Recovery fresh pretrained cùng recipe, gộp epochmetadata mộtcommit và repoFiveC riêng; final model/metadata verification bắt buộc. Metrics epoch đầu khớp attempts trước. Cả bốn final runs đã VERIFIED, hai GPU đã tắt; failed logs/metadata lưu riêng, không dùng checkpoint giữa chừng làm final result.
