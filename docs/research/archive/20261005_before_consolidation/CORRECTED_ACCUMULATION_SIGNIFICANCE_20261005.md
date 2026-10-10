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
