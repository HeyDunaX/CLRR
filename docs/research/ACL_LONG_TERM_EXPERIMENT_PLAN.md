# CLRR — kế hoạch dài hạn và gates Go/No-Go

## Trạng thái hiện hành — 08/10/2026

- **Amis corrected development seeds42–44** và **32 fresh confirmation runs seeds45–52** đã hoàn tất. Cohort fresh không đạt primary/secondary investment gates; không gọi development gain là robust confirmation.
- Kiểm chứng độc lập validation và cơ chế trên 32 checkpoints đã xong. Không suy gradient norm nhỏ thành Adam update yếu; can thiệp tắt lớp10–11 chưa ủng hộ hướng bỏ lớp cao khi retrain.
- **Turkish→English OPUS100 gốc**, train20k/validation2k/test2k: Baseline/Middle/LSR/Full seed42 đã train và test xong. Full−LSR +0.370036 BLEU; Full−Baseline −0.063583; chưa có ưu thế rõ trước Baseline hay bằng chứng multiseed.
- Hai A100 đã tắt sau xác minh backup. Không có training mới được mở trong bước gộp docs.
- Checkpoint public duy nhất: `FiveC/amis-rewire-checkpoints`; 90 archive cuối cùng đã được tổ chức và audit. Quy tắc/lookup ở [hướng dẫn checkpoint](../operations/HUGGINGFACE_CHECKPOINT_STORAGE.md).
- **Bước tiếp theo chưa launch:** khóa tối đa hai candidate có giả thuyết rõ, matched controls, correctness/tiny GPU gate, rồi pilot validation-only theo budget trong long-term plan. Không tự mở rộng Full cũ hay sweep chỉ từ test đã xem.

Các mục kế hoạch/trạng thái cũ phía dưới giữ phạm vi lịch sử. Trạng thái ở mục này và kết quả confirmation/Turkish có ưu tiên khi quyết định bước tiếp theo.

**Cập nhật: 08/10/2026, Asia/Saigon.** Tài liệu chính duy nhất cho chiến lược và quyết định. [Insight hiện hành](RESEARCH_INSIGHTS.md) chứa số và diễn giải; [bàn giao thực thi](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md) chứa scripts/lệnh/checkpoints. Báo cáo bằng chứng lịch sử được gom tại [archive chung](../archive/README.md); trạng thái thực thi ưu tiên bàn giao hiện hành.

## 1. Mục tiêu, trạng thái và quyết định

Xác định routing không thêm parameters có đóng góp ổn định ngoài CE và latent alignment được tune, với chi phí và phạm vi generalization rõ. Mục tiêu ACL main; gates là tiêu chí đầu tư dự án, không phải quy định ACL hoặc bảo đảm acceptance.

Correctness accumulation qua CPU/GPU/full-size gates; corrected FT/CLRR/LSR/Full đủ3seeds. Lambda .03 thua .10 trên validation, không mở .03 test. Hai GPU đã tắt, backup và metrics verified. Full−LSR meanBLEU+.7373 positive3/3 nhưng significance qua trainingseeds chưa đủ. **Quyết định tại 05/10 trước fresh confirmation:** GO kiểm chứng lúc đó; nay cohort đã xong và không đạt gates, xem trạng thái ở đầu file.

Giữ Full alpha .10/lambda .10/distance2/encoder/detached/warmup0. Matching/tuned LSR hiện trùng .10. Novelty audit có bounded prior comparison, chưa có đủ comparator confirmation. Amis test đã được xem; freshseeds không biến nó thành untouched corpus.

## 2. Điều đã hoàn tất và không nên lặp vô ích

| Nhóm | Trạng thái |
|---|---|
| Accumulation fix, baseline provenance, pinned tiny/fullsize GPU gates | VERIFIED |
| Baseline42fresh,43/44audited; corrected Full/LSR.10/CLRR seeds42–44 | VERIFIED |
| Corrected LSR.03 validation3seeds và decision chọn .10 | VERIFIED; .03testblind |
| Full−LSR/FT sentence/seed statistics | VERIFIED, giới hạn n3 |
| D1/D4, historical D3, V4/V5, warmup, source-attached, grids | Đã có; historical training không làm matched corrected controls |
| Strong LoRA/backbone records | Đã có; không mặc định equal-HPO multiseed evidence |

Chi tiết hiện hành xem insight; inventory lịch sử/receipts/report đầy đủ trong archive và bàn giao. Không mở sweep lại chỉ vì file đã chuyển archive.

## 3. Lộ trình và budget gốc — đọc cùng trạng thái hiện hành

| Gate / thứ tự | Việc cần làm | Train mới | GO | NO-GO / chuyển hướng |
|---|---|---:|---|---|
| G0/N1 | Consolidate four-way corrected, native3metrics, paired effects và interaction Full−LSR−CLRR+FT | 0 | Provenance đúng, analysis scope rõ | Sửa nguồn số trước qualityclaim |
| G0/N2 | Formula/prior/comparator card, claim ledger | 0 | Khác biệt cụ thể và ablation bác bỏ được | Thu hẹp novelty/narrative; không chỉ skip+alignment |
| G1/N3 | Matched overhead FT/CLRR/LSR/Full | 0 fulltrain | Chi phí định lượng, claim phù hợp | Bỏ negligible-overhead claim nếu không đạt |
| G1/N4 conditional | Corrected Full targetanchor/residual/CE–LSR/rank, fixedvalidationbatches | 0 fulltrain | Trả lời question cần thiết / hypothesis cụ thể | Dừng probes lặp không ra lựa chọn |
| G3/N5 ưu tiên hiện tại | Full fixed/FT/CLRR/LSR.10×8freshseeds45–52 | **32** | Gate quality/stability dưới đây | Inconclusive→scope/diagnostic; negative→G2 |
| G2/N6 conditional | Tối đa2hypothesis upgrades,3developmentseeds, matching controls | ≤12 theo controls | Validation gate khóa trước đạt | Sau2hướng thất bại review method/RQ, không infinite sweep |
| G3 nếu đổi method | Confirmation candidate và oldFull với8freshseeds chưa dùng chọn candidate | 32–48 tùy distinctfamilies | Gain lặp trước strongLSR/oldFull | Không reuse seed đã chọn làm independent confirmation |
| G4/N7 sau user bật Turkish | Turkish→English20k,8seeds,4–5distinctfamilies | 32–40 | Transfer benefit trước FT/LSR | Giữ negative, scope hẹp; không đổi data để xóa NO-GO |
| G5 | NLLB direction confirmation,5seeds×4–5families | 20–25 | Benefit ngoài mBART | Bỏ architecture-agnostic qualityclaim |
| G6 | Generic/prior routing control; matched PEFT nếu claim cần | Budget riêng | Ích lợi ngoài existing skip và tuningbudget | Đổi positioning/method |
| G7 | Dataregime/challenge/linguistic intervention nếu claim cần | Budget riêng | Predicted pattern/taskquality được xác nhận | Bỏ claim unsupported |
| G8 | Final overhead/reproducibility/claim audit, ACL manuscript review | 0 | Mainclaims đủsupport, formatACL đúng | Revision; không bảo đảm acceptance |

Thứ tự mới: **N1/N2 → N3 (N4 conditional) → xác nhận cấu hình hiện tại N5**. G2 không bắt buộc nếu Full đang có đủ signal để confirm. Nếu mở upgrade dựa n5results, cần confirmation seeds/context độc lập mới. Không tự bật Turkish hoặc mọi dòng có budget.

## 4. Protocol khóa và fairness

- Dataset `data_processed/amis_mandarin/`, source,target; train/validation/test hashes theo corrected bundle. mBARTrevision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; proxy tl_XX→zh_CN.
- LR5e-5, weightdecay0, warmup.06, batch32×accum4, eval8, lengths256, BF16+checkpointing,20epochs, ES4, no-auto-resume; pinned corrected runtime.
- Validation chrF++beam1 selectsbest, testbeam4 FP32/TF32off; BLEUzh/nativechrF++rawrefs, chrF0diagnostic. Không chọn config qua test/rank/cosine.
- Same init/data-order seed và optimization policy giữa fourfamilies. Đổi selection/LR/loss aggregation/objective cần audit/rerun affectedcontrols; không sửa riêngFull rồi dùng mismatched controls.
- CPU/Python local chạy conda clrr; Colab pinned conda clrr, A100 Standard40GB, nohighmem. Tiny lifecycle gate đúng **newbundle** trước longtraining.
- Backup selectedmodel từngrun, pinned HF SHA/metadata verify+local metricsreplay trước cleanup. Lưu failedattempts, không overwrite namespace. Remote reports không thay independent localverification.

## 5. Gate đã dùng cho fresh confirmation — không sửa hậu nghiệm

Seeds45–52,8seedpairs khóa trước training. Chốt power/precision/smallest useful effect từ developmentdeltas trước launch;8seeds không bảo đảm power. Không tăng n đến p<.05 hoặc bỏ seedxấu.

| Điều kiện đầu tư | Ngưỡng |
|---|---|
| Full−FT meanBLEU | ≥+.50 |
| Full−matching/bestobserved LSR meanBLEU | ≥+.30 |
| Paired wins trướcLSR | ≥6/8 |
| Mean chrF0 | No-harm trước relevantcontrols |
| Uncertainty | Seed effects/CI và sentencebootstrap, scope và multiplicity rõ |

Effects đạt nhưng CI qua0: inconclusive, không đổi threshold để gọi GO. Có positive scope hẹp có thể report với uncertainty. Negative hoặc belowgate: dừng expansion, phân tích cause/upgrade nếu testablehypothesis. Nonsignificant không nghĩa equivalence. Recompute interaction dùng shared sentenceresampling across frozen systems; không coi3×575 độc lập. Bổ sung contrasts không ghi đè oldHolm12family; công bố exploratoryfamily riêng.

## 6. Nâng cấp có điều kiện

| Candidate | Giả thuyết và thay đổi | Controls bắt buộc |
|---|---|---|
| U1 raw targetanchor | SourceCLRR giữ, targetpass no-grad/routingoff; alpha/lambda cũ | OldFull, matchingLSR; oldD3 chưa ủng hộ conflict reduction |
| U2 topology | Chọn **một** pre-rewiringcache hoặc distance4, không đổi cảhai | Oldd2Full và CLRR-only, genericroute khiclaim cần |

Không tăng lambda/normalize chỉ vì gradientnorm nhỏ hoặc rank chưa đẹp. Candidate card phải chỉ rõ mathematicalrule, mechanisticprediction, comparison, budget và failurecriterion. Max2candidates×Full/CLRR-only×3developmentseeds≤12; objective đổi cần matchingLSR mới và tính vào budget. U1 không đổiCLRR-only có thể auditedreuse, không tự coi reuse hợp lệ.

Validation pilot gate: Fnew−Fold và Fnew−bestobservedLSR≥.30BLEU; Fnew−matchingLSR≥.30 và≥2/3wins; chrF0no-harm trướcFold/tunedLSR, chrF++no-harm trướcFold;≥2/3seeds tăngBLEU+chrF0 trướcFold. Qualitypass nhưng mechanisticpredictionfail: có thể confirmquality với narrativerútgọn. Code/source/runtime changes qua correctnessgate trước pilot.

## 7. Direction, comparator và efficiency

Turkish data `data_processed/opus100_turkish_english_20k/`:20000/2000/2000; train20k mẫu cũ giữ nguyên, validation/test official giữ raw strings/order/duplicates; audit17source types overlap validation/test, train/evaluation0; không lọc evaluation. Bản filtered v2 đã xóa theo tác giả06/10/2026; simulated low-resource adaptation trên languagepretrained, khác Amis. Codes tr_TR→en_XX, BLEU13a, cùng nativechrF/FP32policy. Methodlock trước trainedtest; retuning chỉvalidation, công bốbudget. Primary giữ2000rows,18duplicateexcesspairs cần sensitivity/clusteredresampling. Turkishzero-shot đã xem; không gọi corpusuntouched. GateFull−LSR≥.30,≥6/8wins, chrF++/chrF0no-harm và uncertainty. Trainedtest usedforswitchmethod→holdoutmới cho strictconfirmation.

Novelty: chọn một genericidentityskip hoặc prior-adaptedcontrol,5–8seeds/context sau card. Đừng gọi adaptationmBART là exactreproductionXGLM. ClaimPEFTsuperiority cần actualcounts và equalHPOopportunity; LRkhác tự nó không unfair. StrongLoRA cũ có thể supplement, không proofPEFTfailure. NLLBcode/revision/lang/BOS cần smoke riêng, không reuse Amisproxy choTurkish.

Benchmark cùng hardware/batches/lengths/precision, warmup, synchronization, lặp; train-step bao gồm extra targetforward củaLSR; inference không tính trainingonlyloss. Báo latency/throughput/peakmemory/parametercounts, batchpadding và lengths. Zeroaddedparams không nghĩa runtimezero.

## 8. Manuscript và quản lý thông tin

Paperclaimledger: usefulFull, routingincrement, noaddedweights, contexttransfer, mechanism và efficiency mỗiclaim có evidence/limit. Morphology phải expert/challengegold nếu giữclaim. Không collapseproof khi chỉ endpointSVD. Giữ ACLformat/pdflatex, bảng số cóprovenance; không che bằng cách bỏ records âm hoặc chọn metric hậu nghiệm.

`docs/research/` chỉ giữ **hai tài liệu chính**. Chi tiết dài/report từngđợt vào archive, kết quảraw ởoutputs, scripts/lệnh và executionstate ởoperationshandoff, metricdefinition ở [METRICS](../experiments/METRICS.md). Khi có newverifiedresults: cập nhật insight rồi decisionplan/handoff; không prepend nhiều đoạncurrentmâu thuẫn. Kế hoạch này là source cho gates; thresholds đổi phải ghi lýdo và trước chạycohort bịảnhhưởng.

## 9. Tra cứu archive

| Cần tra cứu | Report |
|---|---|
| Corrections/code/provenance | [G0/G1](../archive/TECHNICAL_AUDITS_20261005.md#g0-g1-verification-20261005), [pre-rerun](../archive/TECHNICAL_AUDITS_20261005.md#pre-rerun-verification-20261005) |
| Correctedquality | [confirmation](../archive/CORRECTED_RESULTS_20261005.md#accumulation-confirmation-20261005), [sixcontrols](../archive/CORRECTED_RESULTS_20261005.md#corrected-controls-20261005) |
| Statisticalmethods/results | [significance](../archive/CORRECTED_RESULTS_20261005.md#corrected-accumulation-significance-20261005) |
| Historical/fullbudgets | Kế hoạch hiện hành ở file này; [oldinsight](../archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#research-insights) |
| Fixedcomparators nếu đổi method | [fixedexperiments](../archive/HISTORICAL_DIAGNOSTICS_20261002_05.md#fixed-experiments-if-method-changes) |

Archive giữ nguyên số và historicalstates; relocation chỉnh relativeMarkdownlinks cho đúng nơi mới. Manifest dọn tài liệu: [CLEANUP_MANIFEST](../archive/CLEANUP_MANIFEST_20261006.json). Việc xóa nguồn cũ đang chờ xác nhận.



<a id="locked-confirmation-protocol"></a>

## Phần gộp: CONFIRMATION_DESIGN_AND_GATES

> Nội dung gộp từ `CONFIRMATION_DESIGN_AND_GATES.md`. **Protocol nội bộ đã khóa trước cohort seeds45–52.** Giữ nguyên để truy vết; cohort đã hoàn tất, không dùng phần này như lệnh mở lại 32 runs.

### THIẾT KẾ THỰC NGHIỆM XÁC NHẬN VÀ TIÊU CHÍ GO/NO-GO KHÓA TRƯỚC (CONFIRMATION PROTOCOL & GATES)

**Ngày cập nhật:** 05/10/2026  
**Dự án:** CLRR hướng tới ACL Main Conference  
**Trạng thái:** Khóa trước huấn luyện (Protocol nội bộ khóa trước thực nghiệm).

---

#### 1. Phân định Primary và Secondary Endpoints

Dựa trên phân tích tương tác 4-way và kiểm định độ biến thiên qua các seeds:

1. **PRIMARY ENDPOINT: Full vs. LSR-only (Matching Alignment Control)**
   - **Lý do khoa học:** Cả hai nhánh đều cùng nhận chung thành phần căn chỉnh tiềm ẩn $\lambda = 0.10$. Phép so sánh này cô lập **chính xác giá trị gia tăng của cơ chế Rewiring ($\alpha = 0.10$)**.
   - **Tính ổn định thống kê:** Trên cohort corrected, Full thắng LSR trên **3/3 seeds** (Seed 42: $+1.23$, Seed 43: $+0.36$, Seed 44: $+0.63$). Độ lệch chuẩn giữa các seeds thấp ($SD = 0.4439$), tạo ra Effect Size $d = 1.6609$.
2. **SECONDARY ENDPOINT: Full vs. FT Baseline**
   - **Lý do khoa học:** Đánh giá tổng lợi ích của hệ thống hoàn chỉnh so với tinh chỉnh chuẩn (CE-only).
   - **Tính biến thiên:** FT Baseline có phương sai giữa các seeds rất cao ($SD = 1.2310$, Seed 43 tụt xuống 18.14 trong khi Seed 44 vọt lên 20.30), dẫn đến Effect Size nhỏ hơn ($d = 0.6025$). Do đó, nếu lấy Full - FT làm điều kiện tiên quyết duy nhất sẽ có nguy cơ Type II error (bỏ sót phương pháp tốt do phương sai lớn của baseline).

#### 2. Kết quả phân tích Power & Khuyến nghị số lượng Seeds

Kết quả mô phỏng Monte Carlo ($100,000$ lần lặp) cho kiểm định cặp (Paired t-test, $\alpha=0.05$):

| Số lượng Seeds ($N$) | Power Full - LSR (Kịch bản observed $d=1.66$) | Power Full - LSR (Kịch bản MDE $=+0.30$) | Power Full - FT (Kịch bản observed $d=0.60$) | Điều kiện p < 0.05 của Binomial Sign Test (thắng/thua) |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 6$** | $89.3\%$ | $26.8\%$ | $22.2\%$ | Cần thắng **6/6** ($p = 0.0312$) |
| **$N = 8$ (Khuyến nghị)** | **$97.9\%$** | **$37.7\%$** | **$31.2\%$** | Cần thắng **8/8** ($p = 0.0078$) |
| **$N = 10$** | $99.6\%$ | $47.3\%$ | $39.3\%$ | Cần thắng **9/10** ($p = 0.0215$) |
| **$N = 12$** | $99.9\%$ | $56.6\%$ | $47.4\%$ | Cần thắng **10/12** ($p = 0.0386$) |

##### Lưu ý quan trọng về Power và Cỡ mẫu:
- **Phạm vi áp dụng của số liệu Power:** Mức power $97.9\%$ cho $N=8$ là kịch bản giả định chênh lệch thật vẫn duy trì mức cao $\approx +0.74$ BLEU, độ lệch chuẩn ổn định và hiệu ứng phân phối chuẩn ước tính từ 3 seeds ban đầu. Nếu chênh lệch thật chỉ ở mức tối thiểu có ích (MDE $= +0.30$ BLEU), power của $N=8$ giảm xuống khoảng **$37.7\%$**. Đối với Full - FT, power dưới effect size quan sát được chỉ khoảng **$31.2\%$**. Do đó, $N=8$ là thiết kế khả thi theo ngân sách dự án, **không phải bảo đảm chắc chắn đạt ý nghĩa thống kê**.
- **Phân biệt kiểm định & Tiêu chí đầu tư (Investment Criterion):** 
  - Phép thử theo số trận thắng là **Binomial Sign Test**. Với $N=8$, để đạt mức ý nghĩa $p < 0.05$ đòi hỏi phải thắng **8/8 seeds** ($p = 0.0078$).
  - Tiêu chuẩn **$\ge 6/8$ seeds thắng** trong Primary Gate là **tiêu chuẩn quyết định đầu tư (investment criterion)** của nhóm nghiên cứu để đánh giá phương pháp có hiệu quả ổn định, **không phải tiêu chuẩn p-value** (vì xác suất nhị thức $6/8$ thắng là $p = 0.1445$).
  - Đối với kiểm định ý nghĩa thống kê magnitude-based, sử dụng **Permutation Sign-Flip Test** và **Paired t-test** hai phía trên toàn bộ vector paired differences.
- **Quy mô ngân sách khóa cứng:**
  - Kết quả Bước 4 đã hoàn thành: $\lambda=0.10$ đạt mean validation chrF++ 14.2220, thắng $\lambda=0.30$ (14.1506) và $\lambda=0.03$ (14.0503).
  - Khóa chính thức: **4 families $\times$ 8 seeds = 32 runs** (Baseline, CLRR-only, matching LSR .10, Full).

#### 3. Tiêu chí chọn Checkpoint (Model Selection Policy)

- **Metric chọn duy nhất (Single Locked Selection Metric):** **Validation raw-reference chrF++** (Greedy beam 1).
- **Lý do:** Bảo đảm tính nhất quán 100% với các protocol lịch sử, hoàn toàn tự động theo receipt.
- **Vai trò của Guardrail Metrics:** Validation BLEU và chrF0 chỉ là **chỉ số chẩn đoán (diagnostic guardrails)**. Tuyệt đối **không can thiệp, không reject và không replace checkpoint** hậu nghiệm dựa trên BLEU hoặc chrF0. Checkpoint đã chọn theo raw chrF++ beam 1 là bất biến.
- **Quy tắc bảo mật tập Test:** Tuyệt đối không dùng Test set để chọn epoch hay checkpoint. Test set chỉ được mở sau khi toàn bộ 32 checkpoints của cohort đã bị khóa (selection lock).

#### 4. Tiêu chuẩn quyết định Gate (Khóa trước khi chạy Fresh Seeds)

> **Lưu ý về Naming & Kỳ vọng:** Đây là **Protocol nội bộ khóa trước thực nghiệm (Internal pre-locked protocol)**, không phải đăng ký thử nghiệm lâm sàng công khai (formal clinical registry). Đạt các tiêu chuẩn Gate (PASS) là điều kiện cần của nhóm tác giả để quyết định viết và nộp bài, không tự động đồng nghĩa với việc hội nghị ACL chắc chắn chấp nhận bài báo.

##### ✅ TIÊU CHUẨN GO (Xác nhận thành công - Tiến hành hoàn thiện bài báo):
1. **Primary Gate (Bắt buộc):**
   - Mean $\Delta(\text{Full} - \text{LSR}) \ge \mathbf{+0.30}$ BLEU trên 8 fresh seeds.
   - Ít nhất **$6 / 8$** paired seeds dương ($\ge 75\%$ tỷ lệ thắng trước matching LSR, tiêu chuẩn đầu tư).
   - Mean chrF++ và chrF0 không bị suy thoái so với LSR.
2. **Secondary Gate (Hỗ trợ):**
   - Mean $\Delta(\text{Full} - \text{Baseline}) \ge \mathbf{+0.50}$ BLEU.
   - Ít nhất $5 / 8$ paired seeds dương trước FT.
3. **Thống kê:**
   - Báo cáo đầy đủ 95% Confidence Interval theo seed và bootstrap câu, kèm p-value chính xác từ Paired t-test và Permutation Sign-Flip Test.

##### ❌ TIÊU CHUẨN NO-GO:
- Mean $\Delta(\text{Full} - \text{LSR}) < +0.30$ BLEU, hoặc số seed thắng $< 6/8$.
- Trong trường hợp này: Dừng tuyên bố "General Superiority", chuyển hướng bài báo sang phân tích cơ chế tương tác và giới hạn thực nghiệm (empirical / diagnostic contribution), không cố ý sweep thêm seeds hay đổi metrics hậu nghiệm.
