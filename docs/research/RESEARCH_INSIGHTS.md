# CLRR — insight nghiên cứu hiện hành

## Cập nhật Turkish và giới hạn claim — 08/10/2026

Turkish OPUS100 gốc seed42 đã train/test xong: Full hơn LSR +0.370036 BLEU, hơn Middle +0.050624 nhưng thấp hơn Baseline −0.063583. Đây là exploratory single-seed, không đảo kết luận NO-GO của confirmation Amis, không xác nhận robust/general superiority. Bảng đầy đủ nằm ở [sổ thực nghiệm, Mục14](../experiments/EXPERIMENTS_UPDATED.md#turkish-seed42).

### Đính chính các tài liệu novelty/PEFT được gộp

- CLRR là **Cross-Layer Residual Rewiring**. Code hiện tại tính LSR giữa masked mean-pooled **encoder cuối của source và target**, với target `no_grad`/detach; không phải giữa tầng giữa và tầng sâu. Công thức cosine không có predictor/EMA teacher. Xem `src/amis_rewire/modeling.py`, lớp `JEPAGuidedSeq2SeqLM`.
- Novelty card cũ là đối chiếu giới hạn với prior; chưa bảo đảm novelty và không chứng minh empirical interaction ổn định. Gain/interaction seeds42–44 không được áp thành kết luận fresh seeds45–52.
- Ước lượng tham số trong PEFT reference không thay actual measured counts của từng run. Không suy nguyên nhân overfitting/morphology hay ưu thế anti-collapse từ score thấp của LoRA/BitFit.
- Đối sánh học thêm gate hoặc skip thông thường là đề xuất cần protocol/matched controls; chưa chứng minh learned gate sẽ overfit. Các đoạn tham khảo cũ bên dưới không là candidate card đã được duyệt.

**Cập nhật: 07/10/2026, Asia/Saigon.** Đây là tài liệu diễn giải duy nhất được cập nhật tiếp. [Kế hoạch dài hạn](ACL_LONG_TERM_EXPERIMENT_PLAN.md) chứa quyết định và gates; [bàn giao thực thi](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md) chứa entrypoints, checkpoint mapping và lệnh. Reports chi tiết và insight cũ nằm trong [archive](../archive/README.md).

## 1. Trạng thái và phạm vi bằng chứng

Corrected control study mBART Amis→Mandarin đã hoàn tất. Baseline seed42 train lại, seeds43/44 audited reuse; Full, CLRR-only, LSR lambda .10 và LSR lambda .03 đều đủ seeds42–44. LSR .03 validation-only đúng kế hoạch. Mọi kết quả hiện hành có prediction replay/metadata verification; hai Colab đã tắt sau backup verified.

Wrapper CLRR/LSR trước sửa bỏ loss kwargs, khiến Transformers4.57.6 không chia mean microbatch loss đúng accumulation4. Regression tái hiện norm update factor4; fix `accepts_loss_kwargs=False` qua CPU/GPU/full-size gates. Không diễn giải thành AdamW LR tăng4lần. Recipe hiện tại average microbatch mean; chưa đồng nghĩa global tokenweighted CE khi tokencounts khác nhau.

Scores lịch sử vẫn đúng cho checkpoint lịch sử, nhưng không làm matched training controls cho code corrected. Không ghi đè số cũ, không gán bug tự động cho mọi backbone/PEFT. Nguồn: [technical audit](../archive/TECHNICAL_AUDITS_20261005.md#g0-g1-verification-20261005), [gates](../archive/TECHNICAL_AUDITS_20261005.md#fullsize-gate-and-baseline42-20261005).

## 2. Corrected development study — seeds 42–44

Test 575 raw-reference câu, beam4, FP32 generation, BLEU Chinese tokenizer; checkpoint chọn bằng validation chrF++ beam1. Seeds cùng recipe. Mean ± sample SD, không phải confidence interval.

| Seed | FT baseline BLEU | CLRR-only BLEU | LSR-only .10 BLEU | Full .10/.10 BLEU |
|---|---:|---:|---:|---:|
| 42 | 19.728081 | 19.223095 | 19.095440 | 20.321295 |
| 43 | 18.137832 | 19.492997 | 19.819251 | 20.177995 |
| 44 | 20.300016 | 19.345498 | 19.264374 | 19.891624 |
| Mean ± SD | 19.3886 ± 1.1203 | 19.3539 ± .1351 | 19.3930 ± .3787 | 20.1303 ± .2188 |

Nguồn: [Full/LSR confirmation](../archive/CORRECTED_RESULTS_20261005.md#accumulation-confirmation-20261005), [CLRR/LSR tuning](../archive/CORRECTED_RESULTS_20261005.md#corrected-controls-20261005). Receipts/raw predictions: `outputs_rebuttal/amis_accumulation_confirmation_20261005/three_seed_comparison.json` và `outputs_rebuttal/amis_corrected_controls_20261005/{clrr,lsr003}/seed*/local_verification.json`.

### Chọn LSR bằng validation

| Seed | chrF++ lambda .03 | chrF++ lambda .10 |
|---|---:|---:|
| 42 | 13.906081 | 14.190174 |
| 43 | 14.066061 | 14.274377 |
| 44 | 14.178781 | 14.201443 |
| Mean | 14.050307 | 14.221998 |

Chọn **lambda .10** trước test; .03 thấp hơn cả3seeds, giữ .03 test blind. Đây là best observed trong grid {.03,.10}, không phải optimum toàn cục. Matching/tuned LSR hiện trùng config; không đếm hai families độc lập. Decision: `outputs_rebuttal/amis_corrected_controls_20261005/lsr_selection_decision.json`.

## 3. Diễn giải development study — seeds 42–44

| Quan sát | Diễn giải được hỗ trợ | Giới hạn |
|---|---|---|
| Full−LSR mean +.7373BLEU, dương3/3 | Có tín hiệu routing bổ sung LSR trong recipe hiện tại | Chưa robust seed-level superiority |
| Full−CLRR dương3/3 | Tổ hợp tốt hơn routing đơn lẻ trong3seeds | Chưa chứng minh cơ chế tương tác nhân quả; factorial contrast cần analysis riêng |
| Full−FT mean +.7417, dương2/3 | Có tín hiệu adaptation benefit Amis | Seed44 −.4084; baseline43 thấp ảnh hưởng mean |
| CLRR−FT mean −.0348 | Chưa có standalone meanBLEU benefit | Không suy CLRR vô ích khi kết hợp LSR |
| LSR−FT mean +.0044 | LSR đơn lẻ chưa cho meanBLEU gain đáng kể | Dấu khác giữa seeds; không kết luận equivalence |
| Lambda .03 thua validation | Giảm lambda đơn giản chưa giúp corrected LSR | Không phải exhaustive HPO; không suy mọi lambda thấp đều kém |

Full−LSR test chrF++ mean +.3945 theo confirmation report. Baseline43 chrF++ thấp đã replay đúng; không gọi metric bug khi raw predictions/reference khớp. Đọc per-seed và chrF0, không dựa chrF++ aggregate duy nhất.

### Statistical uncertainty

10k paired sentencebootstrap, RNG42, native SacreBLEU p-values independently matched. Family12 =3seeds×2contrasts(Full−LSR/FT)×2metrics, Holm12. Không có Full−LSR per-seed test nào qua Holm12. Full−baseline BLEU seed43 qua Holm12; không thay conclusion seed robustness.

Conditional sentence CI của mean Full−LSR BLEU [+ .0403,+1.4557] giữ cố định3trainingseeds, unadjusted; không bao gồm training randomness. Full−FT meanBLEU CI [−.0265,+1.5182]. Exact pairedseed signflip: Full−LSR p=.25, Full−FT p=.50, n3; minimum attainable two-sided p=.25. CI chứa0/pkhông đạt không chứng minh không có effect. Nguồn: [statistical report](../archive/CORRECTED_RESULTS_20261005.md#corrected-accumulation-significance-20261005).

## 4. Historical diagnostics — dùng đúng phạm vi

| Evidence | Đã thấy gì | Không được suy |
|---|---|---|
| V4 route on/off trên old Full | Routing computation thực sự được weights sử dụng | +2.9676 validationBLEU không là gain trước independently trained LSR |
| V4/V5 gradient/residual/rank | Không thấy endpoint explosion/global LSR domination; local negative cosines; rank phụ thuộc site/definition | Anti-collapse proof, không conflict tuyệt đối, causal geometry→quality |
| D3 routed/raw target anchor, old3Full | Anchor/gradient đổi; raw target không cho global conflict thấp hơn | U1 raw anchor đã chứng minh cải thiện dịch hoặc mô tả corrected Full |
| D1 warmup selection | Chọn đúng best; không chỉ do dùng last epoch | Warmup thất bại do early stopping ở mọi seed |
| Warmup/source-attached/grid cũ | Đã thử và lưu đúng historical checkpoints | NO-GO áp dụng y nguyên cho corrected training |
| Historical strong LoRA/NLLB/mT5/ByT5 | Có baseline/backbone records và common inference replay | Matched multi-seed superiority/generality hoặc PEFT capacity ceiling |

V3/F4 contaminated probes/warmup v1 không dùng làm evidence hiện hành. Asháninka/LayerSkip ngoài phạm vi. Không lặp diagnostics cũ chỉ để tạo thêm bảng.

## 5. Novelty và claim cho paper

CLRR có zero added trainable parameters nhưng vẫn full fine-tuning. Không đồng nghĩa PEFT hoặc negligible runtime overhead. Skip/fusion/value residual đã có; bounded formula audit phân biệt encoder hidden block routing/detach/shared-encoder cosine với prior, chưa chứng nhận novelty rộng.

Câu chuyện có cơ sở để kiểm tiếp là **routing bổ sung latent alignment**, chưa phải routing luôn tốt khi đứng riêng. Cần freshseed confirmation, routing comparator có ý nghĩa, efficiency và direction khác sau methodlock. Không claim morphological fidelity nếu chỉ có regex/proxy; không mượn collapse guarantee của prior cho CLRR. Giữ format ACL, viết claims tương ứng evidence.

<a id="verified-confirmation-insight"></a>

## 6. Quyết định hiện hành — sau 32 fresh runs và kiểm chứng 07/10/2026

**NO-GO mở rộng Full với cấu hình hiện tại; GO xây dựng giả thuyết nâng cấp có controls.** Fresh confirmation seeds 45–52 không xác nhận tín hiệu ở development seeds 42–44: Full−LSR mean BLEU −0.019809 (3/8 seeds dương), Full−Baseline −0.041468 (4/8 dương). Không đạt gate đã khóa; không suy rằng mọi biến thể CLRR/LSR đều vô ích. Bảng 2–3 phía trên là development study, không đại diện kết quả confirmation.

Kiểm chứng validation độc lập trên cả 32 checkpoints đã hoàn tất bằng A100 Colab 1/2; cả hai phiên đã tắt sau SHA-verified collection. Không training hoặc sinh predictions test mới. 48 bộ predictions replay metrics khớp; 16 bộ all-on CLRR/Full khớp validation predictions cũ từng câu. Checkpoint archive/weight/source/data hashes khớp; rank replay từ saved spectra, không từ raw matrices.

| Phép kiểm tra | Kết quả | Diễn giải hiện hành |
|---|---|---|
| Full tắt residual lớp 10 và 11 | ΔBLEU −0.049780; 4/8 tăng; CI 95% [−0.374957, +0.275398]; ΔchrF++ −0.118199 | Chưa hỗ trợ hướng bỏ hai lớp này; không chứng minh retraining ít lớp sẽ thất bại |
| Full tắt riêng lớp 10 / 11 | ΔBLEU −0.080985 / −0.083960; 2/8 / 3/8 tăng | Không có lợi ích nhất quán từ endpoint intervention |
| CLRR tắt lớp 10 và 11 | ΔBLEU −0.073839; 4/8 tăng; ΔchrF++ −0.101613 | Không thấy cải thiện trung bình |
| LSR / Full raw matched cosine | 0.798194 / 0.782713; shuffled 0.709564 / 0.700077 | Phần lớn mức cosine cao có hướng chung; raw cosine chưa đo riêng semantic alignment |
| Centered matched cosine | Baseline 0.264952; CLRR 0.253826; LSR 0.334473; Full 0.303308 | LSR/Full vẫn có tín hiệu alignment sau centering; không kết luận toàn bộ gain cosine là giả hoặc collapse |
| Weighted LSR/CE global gradient norm ratio | LSR 0.002455; Full 0.001829 | Không tương đương tỷ lệ Adam update; chưa đủ cơ sở để cấm hoặc chọn tăng lambda |

CI can thiệp là paired seed Student-t, exploratory, chưa hiệu chỉnh nhiều phép so sánh. Can thiệp chỉ tại inference trên weights đóng băng. Baseline/CLRR gradient LSR cũ là counterfactual; provenance worker/runtime theo model thiếu trong probes cũ, không retroactively xác minh được.

**Điều chỉnh diễn giải:** cosine skip/current thấp nhưng dương không chứng minh conflict có hại; tỷ lệ residual/current giảm còn có thể do current RMS tăng. Chưa có bằng chứng để ưu tiên U1 bỏ lớp cao hoặc khẳng định tăng alpha/lambda chắc chắn có hại. Có thể nghiên cứu centered alignment/anchor như giả thuyết, nhưng phải chốt công thức, gradient và matched controls trước pilot. Chưa mở training nâng cấp hoặc Turkish trong đợt này.

Nguồn số mới: [verification report](../../outputs_rebuttal/amis_small_validation_checks_20261007/VERIFICATION_REPORT.md), [independent replay](../../outputs_rebuttal/amis_small_validation_checks_20261007/independent_verification.json). Quy trình/receipts/entrypoints: [bàn giao Mục 10](../operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#10-kiểm-chứng-độc-lập-đã-hoàn-tất--07102026). Gates và budget nâng cấp tiếp tục theo [kế hoạch dài hạn](ACL_LONG_TERM_EXPERIMENT_PLAN.md).

## 7. Quy tắc cập nhật

1. Chỉ thêm kết quả VERIFIED kèm source/receipt; thiếu ghi unknown/pending.
2. Tách hiện hành, historical, invalid và planned. Không biến probe thành training efficacy.
3. Raw artifacts/checkpoints là nguồn số, insight là diễn giải. Không đổi số archived.
4. Thay trạng thái hiện hành trong file này; không chồng nhiều đoạn “hiện hành” mâu thuẫn. Chuyển chi tiết dài vào archive/report và link từ đây.
5. Gates/next actions chỉ định nghĩa trong long-term plan; commands chỉ trong operations handoff. Báo thay đổi interpretation hoặc protocol rõ ràng.


<a id="novelty-card-history"></a>

## Phần gộp: NOVELTY_AND_COMPARATOR_CARD

> Nội dung gộp từ `NOVELTY_AND_COMPARATOR_CARD.md`. **Tài liệu tham khảo lịch sử, không là kết luận quality hiện hành.** Các claim development, công thức mô tả cũ và số tham số ước lượng phải đọc cùng đính chính phía trên.

### THẺ NOVELTY & ĐỐI CHIẾU COMPARATORS (NOVELTY & COMPARATOR CARD)
#### ĐỊNH VỊ ĐÓNG GÓP CỦA CLRR HƯỚNG TỚI ACL MAIN CONFERENCE

**Ngày lập:** 05/10/2026  
**Dự án:** CLRR (Contrastive Layer Representation Rewiring)  
**Phạm vi:** Đối chiếu công thức, kiến trúc, tham số và cơ chế giữa CLRR và các nghiên cứu gần nhất (SkipCLM, Value Residual / ResFormer, ResiDual).

---

#### 1. Bản chất phương pháp của CLRR hiện tại (Current Recipe)

CLRR áp dụng trên mBART-large-50 Seq2Seq backbone:
1. **Kiến trúc Routing:**
   - Vị trí: Bên trong Encoder của mBART, sau mỗi Transformer block (sau LayerNorm và Residual add của block).
   - Công thức: $h_{l}^{(\text{routed})} = h_l + \alpha \cdot \text{detach}(h_{l-d})$, với khoảng cách cố định $d = 2$, hệ số cố định $\alpha = 0.10$.
   - Tính chất: **Zero added parameters** (không thêm bất kỳ ma trận chiếu, tham số gating hay scalar nào). Gradient không truyền ngược về tầng nguồn $l-d$ nhờ toán tử `detach`.
2. **Mục tiêu căn chỉnh tiềm ẩn (Latent Similarity Regularization - LSR):**
   - Công thức: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{LSR}}$, với $\lambda = 0.10$.
   - $\mathcal{L}_{\text{LSR}}$ tối ưu khoảng cách cosine giữa tầng encoder giữa và tầng encoder sâu (shared-encoder alignment).
   - Lịch huấn luyện: Không dùng warmup cho routing ($\alpha$ cố định từ step 0).

#### 2. Bảng đối chiếu chi tiết với các công trình liên quan trực tiếp

| Trục so sánh | CLRR (Bài báo này) | SkipCLM (NAACL SRW 2025, §3) | Value Residual / ResFormer (ACL 2025, §3.2) | ResiDual (NeurIPS 2023 / ArXiv) |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone & Vị trí** | Encoder của Seq2Seq (mBART-50) | Decoder của Causal LM (XGLM) | Self-Attention Values ($V$) toàn mạng | Toàn bộ Transformer stack |
| **Công thức Routing** | Identity skip toàn khối: $h_l + \alpha \cdot \text{detach}(h_{l-d})$ | Skip có học: $W_{\text{skip}} h_{l-k} + \text{ReLU}$ | Trộn value tầng 1 vào value tầng $l$: $\tilde{v}_l = v_l + \beta v_1$ | Hai luồng residual song song (dual streams) |
| **Tham số bổ sung** | **0 tham số** (Zero added parameters) | Thêm ma trận $W_{\text{skip}}$ và gating ($>0$ params) | 0 hoặc thêm scalar có học | 0 tham số mới nhưng đổi cấu trúc stream |
| **Đường Gradient** | **Detached** (ngắt gradient tầng nguồn) | Backprop xuyên suốt qua $W_{\text{skip}}$ | Backprop bình thường qua attention | Backprop trên cả 2 luồng Post-LN |
| **Mục tiêu căn chỉnh** | Cosine Latent Regularization (LSR, $\lambda=0.10$) | InfoNCE Contrastive Loss giữa các tokens | Không có (chỉ CE) | Không có (chỉ CE) |
| **Lịch Routing (Schedule)** | Cố định $\alpha=0.10$ từ step 0 (Warmup là NO-GO) | Warmup tăng dần độ mạnh skip | Cố định hoặc learned | Cố định theo kiến trúc |
| **Tác vụ & Ngôn ngữ** | Dịch máy cực thấp tài nguyên (Amis $\to$ Zh) | Sinh ngôn ngữ đa ngữ (Multilingual Causal LM) | Đa tác vụ NLP / NMT chuẩn | Huấn luyện mô hình sâu (Deep Transformer) |

#### 3. Ranh giới đóng góp: Điều được khẳng định vs. Điều không được phóng đại

##### ✅ Những khẳng định được số liệu thực nghiệm ủng hộ (Supported Claims):
1. **Tính chất trung hòa tham số (Parameter Neutrality):** CLRR can thiệp sâu vào cấu trúc biểu diễn của mạng mà không tăng dù chỉ 1 tham số huấn luyện hay tham số suy diễn.
2. **Hiệu ứng hiệp đồng thực nghiệm (Empirical Interaction):**
   - Routing đứng một mình (CLRR-only) **không tạo ra ưu thế dịch thuật** ($-0.0348$ BLEU so với Baseline).
   - Alignment đứng một mình (LSR-only) tạo cải thiện nhỏ ($+0.0044$ BLEU so với Baseline).
   - Nhưng khi kết hợp cả hai (Full), hiệu ứng đạt **$+0.7417$ BLEU** so với Baseline và **$+0.7373$ BLEU** so với LSR-only (dương cả 3/3 seeds).
   - Tương tác nhân tố đại số đạt $+0.7721$ BLEU trên tập test hiện tại; đây là quan sát tương tác thực nghiệm, chưa chứng minh cơ chế giải phóng không gian biểu diễn.
3. **Không thêm tham số học (Zero Added Parameters):** Không tốn thêm trọng số ma trận học như SkipCLM; chi phí phụ trội thực tế về thời gian bước train và độ trễ suy diễn được đo lường cụ thể qua benchmark chi phí.

##### ❌ Những điều TUYỆT ĐỐI KHÔNG được tuyên bố trong bài báo ACL (Overstated / Banned Claims):
1. **Không tuyên bố "Skip connection / layer fusion trong Transformer là phát minh mới của chúng tôi":** Ý tưởng nối tầng (DenseNet, Dense Information Flow, SkipCLM) đã tồn tại từ lâu. Điểm mới của CLRR là **công thức identity detached ở cự ly $d=2$ kết hợp hiệp đồng với latent alignment không cần tham số**.
2. **Không mượn lý thuyết chống suy sụp (Mathematical Anti-collapse):** Không dùng các định lý bảo toàn rank của ResiDual để áp đặt cho CLRR vì CLRR không chạy dual-stream Post-LN.
3. **Không tuyên bố ưu thế tuyệt đối trước PEFT:** Trừ khi chạy đủ HPO và multi-seed cho LoRA/BitFit, chỉ định vị CLRR là phương pháp bổ trợ thích nghi biểu diễn tầng sâu (deep representation adaptation), không tuyên bố thay thế hoàn toàn LoRA.

#### 4. Thiết kế Adapted Comparators cho Paper (Phục vụ Reviewer)

Để bài báo ACL thuyết phục trước reviewer hỏi *"Tại sao không dùng skip thông thường hoặc SkipCLM?"*, cần chuẩn bị 2 adapted controls:
1. **Generic Identity Skip (No Detach):**
   - Đã có dữ liệu từ cohort đối chứng (`Full source-attached`): Test BLEU $= 19.5658$ (thấp hơn Full detached $= 19.7687$). Tuy nhiên đây là kết quả lựa chọn cấu hình trên dữ liệu lịch sử, chưa kiểm chứng cơ chế gradient trực tiếp dưới training corrected.
2. **Adapted SkipCLM-style Gating (Learned Gate Baseline):**
   - Nếu reviewer yêu cầu so sánh trực tiếp, ta triển khai adapted linear gate trên mBART encoder để chứng minh: Việc thêm tham số $W_{\text{skip}}$ trong bối cảnh cực thấp tài nguyên (Amis 575 câu) có nguy cơ overfitting so với fixed identity routing của CLRR.


<a id="peft-reference-history"></a>

## Phần gộp: PEFT_PAPERS_REFERENCE

Các đường dẫn PDF local trong bảng nguồn cũ không còn có file tương ứng; liên kết đã trỏ sang nguồn công bố gốc. Tên file/kích thước trong bảng chỉ là ghi nhận lịch sử.

> Nội dung gộp từ `PEFT_PAPERS_REFERENCE.md`. **Tài liệu tham khảo lịch sử, không là kết luận quality hiện hành.** Các claim development, công thức mô tả cũ và số tham số ước lượng phải đọc cùng đính chính phía trên.

### TÀI LIỆU KỸ THUẬT THAM KHẢO & ĐẶC TẢ HAI PHƯƠNG PHÁP PEFT: LORA & BITFIT

Tài liệu này tổng hợp toàn bộ thông tin học thuật, đường dẫn file PDF đã tải về, trích dẫn BibTeX chính thức và công thức toán học/code chuẩn bị cho việc triển khai bộ đối chuẩn PEFT (**Ưu tiên 1**) trên `mBART-large-50`.

#### 1. Danh Mục File PDF Đã Tải Về Trong Thư Mục `docs/`

| Phương pháp | Hội nghị / Năm | File PDF Trong Kho | Kích thước | Nguồn gốc chính thức |
| :--- | :---: | :--- | :---: | :--- |
| **LoRA** | **ICLR 2022** | [`docs/baseline-paper/hu2022_lora_2106.09685.pdf`](https://openreview.net/forum?id=nZeVKeeFYf9) | 1.6 MB | arXiv:2106.09685 / OpenReview |
| **BitFit** | **ACL 2022 (Short)** | [`docs/baseline-paper/benzaken2022_bitfit_2022.acl-short.1.pdf`](https://aclanthology.org/2022.acl-short.1/) | 328 KB | ACL Anthology: 2022.acl-short.1 |

#### 2. Thông Tin Trích Dẫn BibTeX Chính Thức (Đã cập nhật vào `clrr_references.bib`)

##### 2.1. LoRA (Hu et al., ICLR 2022)
```bibtex
@inproceedings{hu-etal-2022-lora,
  title = "Lo{RA}: Low-Rank Adaptation of Large Language Models",
  author = "Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu",
  booktitle = "International Conference on Learning Representations (ICLR 2022)",
  year = "2022",
  url = "https://openreview.net/forum?id=nZeVKeeFYf9"
}
```

##### 2.2. BitFit (Ben-Zaken et al., ACL 2022)
```bibtex
@inproceedings{ben-zaken-etal-2022-bitfit,
  title = "{B}it{F}it: Simple Parameter-efficient Fine-tuning for {T}ransformer-based Masked Language-models",
  author = "Ben-Zaken, Elad and Ravfogel, Shauli and Goldberg, Yoav",
  booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
  month = may,
  year = "2022",
  address = "Dublin, Ireland",
  publisher = "Association for Computational Linguistics",
  pages = "1--9",
  doi = "10.18653/v1/2022.acl-short.1",
  url = "https://aclanthology.org/2022.acl-short.1"
}
```

#### 3. Đặc Tả Toán Học & Thiết Kế Kiến Trúc

##### 3.1. LoRA: Low-Rank Adaptation (ICLR 2022)
* **Bản chất toán học:**
  Giả sử ma trận trọng số tiền huấn luyện là $W_0 \in \mathbb{R}^{d \times k}$. Thay vì cập nhật trực tiếp toàn bộ $W_0$, LoRA phân rã ma trận cập nhật $\Delta W$ thành tích của hai ma trận hạng thấp (low-rank matrices):
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A$$
  trong đó $B \in \mathbb{R}^{d \times r}$ và $A \in \mathbb{R}^{r \times k}$ với rank $r \ll \min(d, k)$.
* **Quy tắc khởi tạo:**
  * Ma trận $A$ được khởi tạo ngẫu nhiên theo phân phối chuẩn Gaussian: $A \sim \mathcal{N}(0, \sigma^2)$.
  * Ma trận $B$ được khởi tạo bằng **0**: $B = 0$.
  * Hệ quả: Tại bước huấn luyện đầu tiên ($t=0$), $\Delta W = B \cdot A = 0$, mô hình hoàn toàn tương đương với mô hình tiền huấn luyện gốc ban đầu.
* **Hệ số co giãn (Scaling factor $\frac{\alpha}{r}$):**
  * Thường chọn $r = 8$ hoặc $r = 4$, $\alpha = 16$ hoặc $2r$. Hệ số này giúp ổn định gradient khi thử nghiệm các giá trị rank $r$ khác nhau.
* **Module can thiệp trên mBART-50:**
  * Áp dụng trên các ma trận Query và Value chú ý (`q_proj`, `v_proj`) của cả Self-Attention và Cross-Attention (Encoder và Decoder).
  * **Số tham số huấn luyện thêm ($\Delta\theta$):** $\approx 1.54\text{ M}$ tham số (chiếm khoảng **0.25%** tổng số 611M tham số của `mBART-large-50`).

##### 3.2. BitFit: Bias-Term Fine-Tuning (ACL 2022)
* **Bản chất toán học:**
  Đóng băng 100% tất cả các ma trận trọng số $W \in \mathbb{R}^{d \times k}$, **chỉ cho phép cập nhật các vector bias $b \in \mathbb{R}^d$**:
  $$y = W_0 x + b, \quad \theta_{\text{trainable}} = \{b\}$$
* **Các vector bias được cập nhật trong Transformer:**
  * Biases của Multi-Head Attention: Query, Key, Value, Output projection.
  * Biases của Feed-Forward Networks: Intermediate dense layer (FC1) và Output dense layer (FC2).
  * Biases của Layer Normalization.
* **Đặc điểm nổi bật:**
  * Không thêm bất kỳ module ngoại lai nào vào đồ thị tính toán (không tăng inference latency, không cần tensor product phụ).
  * **Số tham số huấn luyện thêm ($\Delta\theta$):** $\approx 0.12\text{ M}$ tham số (chiếm khoảng **0.02%** tổng số tham số của `mBART-large-50`).

#### 4. Code Mẫu Triển Khai Chuẩn Xác (Hugging Face / PyTorch Reference)

##### 4.1. Cấu hình LoRA qua thư viện `peft` chính thức:
```python
from peft import LoraConfig, get_peft_model, TaskType

def apply_lora_to_mbart(model, r=8, lora_alpha=16, lora_dropout=0.05):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"], # hoặc thêm ["k_proj", "out_proj"]
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
```

##### 4.2. Cấu hình BitFit thuần túy bằng PyTorch:
```python
def apply_bitfit_to_model(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "bias" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            
    print(f"BitFit Active: Trainable {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.3f}%)")
    return model
```

#### 5. Ý Nghĩa Học Thuật Khi Đối Chuẩn Với CLRR
1. **BitFit ($\Delta\theta \approx 0.12\text{M}$):** Chỉ tinh chỉnh bias nên dung lượng biểu diễn (capacity) bị giới hạn, khó có thể nắn dòng đặc trưng phức tạp của hiện tượng chắp dính phụ tố Amis (*mi-*, *ma-*).
2. **LoRA ($\Delta\theta \approx 1.54\text{M}$):** Mặc dù giảm đáng kể tham số so với Full Fine-Tuning (611M), nhưng việc học 1.54M tham số mới trên tập huấn luyện cực nhỏ (**4.600 câu**) vẫn rất dễ dẫn đến hiện tượng quá khớp (overfitting) hoặc học mẹo bề mặt.
3. **CLRR-Enc + LSR ($\Delta\theta = 0$):** Không thêm bất kỳ tham số nào cần học mới. Cơ chế chống over-smoothing là giả thuyết nghiên cứu, cần được xác nhận trực tiếp bằng phép đo cơ chế và thực nghiệm đối chứng; không được suy ra chỉ từ một điểm BLEU/chrF.

## Candidate screening Amis seed42 — 2026-10-09

Four seed42 runs completed with the baseline-matched mBART-50 recipe, selecting
on validation chrF++/beam1. Public predictions were replayed independently and
archive LFS hashes verified. They are validation results only; no test set was
evaluated.

| Run | Val BLEU | Val chrF++ | Val chrF0 | Δ chrF++ vs baseline42 |
|---|---:|---:|---:|---:|
| Baseline42 | 16.2659 | 13.9015 | — | — |
| CLRR pre-addition donor | 16.4181 | 13.9381 | 16.0349 | +0.0366 |
| Precache + original LSR | 17.3296 | 14.5519 | 16.6663 | +0.6503 |
| Centered-LSR | 16.4203 | 13.9045 | 16.0522 | +0.0029 |
| CLRR + Centered-LSR | 17.0227 | 14.5080 | 16.6142 | +0.6065 |

The two Full variants exceed their corresponding single-component runs on this
validation seed (chrF++ differences +0.6137 and +0.6035). Precache+LSR leads
Centered-LSR Full by only +0.0439 chrF++. This is a promising screening signal
for the combinations, not evidence of statistical reliability, generalization,
mechanism, or novelty. The two LSR-only variants are almost tied with baseline.
Keep all results labeled validation-only; confirm with fresh seeds and a locked
test evaluation before presenting the candidate as the final method. The
literature novelty concerns remain: pre-addition caching and centered cosine
each have close precedents, so improved scores alone do not settle novelty.

## Frozen candidate diagnostics — validation seed42, 2026-10-09

Six checkpoints strict-loaded from pinned public archives. Validation576,
FP32/eval/dropout off/TF32 off;128 shared representation rows and4 shared
gradient microbatches32. Zero optimizer steps. All scores replayed, worker hash
matched, and saved predictions match after original whitespace stripping.
Colab1/2 were stopped after collection. Historical scores remain unchanged.

| Comparison / measurement | Result | Interpretation |
|---|---|---|
| U1F−oldFull BLEU | −0.0261; CI95[−1.0097,+0.9928]; Holm4p=.9224 | No established improvement or inferiority |
| U2F−oldFull BLEU | −0.3330; CI95[−1.4047,+0.7631]; Holm4p=.7743 | No established improvement or inferiority |
| U1F−oldFull chrF++ | +0.0739; CI95[−0.5616,+0.7437] | No established improvement |
| U2F−oldFull chrF++ | +0.0300; CI95[−0.7025,+0.7926] | No established improvement |
| Final source covariance effective rank: baseline/oldFull/U1F/U2F | 73.097/74.089/74.005/67.499; rank limit127 | No evidence of CLRR-specific final representation collapse |
| Centered matched cosine: baseline/oldFull/U1F/U2F | .2533/.2907/.2910/.3554 | U1 mechanism nearly unchanged; U2 alignment improves without a clear translation gain |
| CE–weighted-LSR gradient cosine: oldFull/U1F/U2F | +.01734/+.01473/+.07827 | No strong mean directional conflict in these frozen batches |
| Raw-anchor CE–LSR cosine: oldFull/U1F/U2F | +.00863/+.00493/+.06904 | Raw anchor not supported as a directional-gradient fix here |
| OldFull late10–12 off: chrF++ | −.5001, CI95[−.8432,−.1608], Holm18p=.0378 | Fixed seed42 checkpoint depends on these routes at inference |
| U2F middle6–9 off: BLEU/chrF++ | +.6065/−.0240; BLEUCI[−.2124,+1.4449], Holm18p=.5599 | No robust evidence supporting this topology change |

Rank at block11/12 is low even for baseline (2.758/1.907) but baseline final
LayerNorm representation rank is73.097. Do not treat pre-final-LN rank as final
encoder collapse. Weighted gradient norm ratios .0031–.0062 are instantaneous,
not accumulated Adam updates and not a reason to increase lambda automatically.

All bootstrap statements are exploratory and conditional on selected validation
checkpoints; no selection-adjusted confirmation or multi-seed causal inference.
Inference ablations do not predict retraining gains. These diagnostics do not
identify a sufficient cause of the old Full's8-seed failure. Do not broaden U1/U2
training or sweep blindly; reconsider the hypothesis/objective and literature
novelty before locking another candidate and requesting its pilot.

Evidence: [full report](../../outputs_rebuttal/method_candidate_diagnostics/REPORT.md),
[protocol/decision](../operations/METHOD_CANDIDATE_DIAGNOSTICS.md),
[bootstrap](../../outputs_rebuttal/method_candidate_diagnostics/bootstrap.json),
[layer intervention bootstrap](../../outputs_rebuttal/method_candidate_diagnostics/interventions.json).
