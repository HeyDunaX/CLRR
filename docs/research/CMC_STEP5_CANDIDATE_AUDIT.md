# Bước 5 — ứng viên tối thiểu và quyết định sau kiểm novelty

Ngày: 2026-11-17. Phạm vi: đọc code/receipts, phân tích CPU, paper gốc, phản biện NotebookLM. Không train, không GPU, không xem test; không sửa recipe hoặc method hiện có.

## 1. Đính chính điều kiện đầu vào

Các bước2–4 trong `outputs/experiments/cmc_estimator_review/shared_errors/INTERPRETATION_AND_NEXT_STEP.md` là **kế hoạch đo**, chưa phải kết quả: chưa có phép so gradient đầy đủ W/denseCMC/sampledCMC/CV dưới cùng accumulation32 và clipping. Không được viết rằng chúng đã xác định nguyên nhân thất bại.

Bằng chứng hiện có:

| Nhãn | Dữ kiện | Nguồn |
|---|---|---|
| ĐO ĐƯỢC | CMC15.540216; sampled-direct16.133234; CV17.148787; W18.272410 BLEU validation seed42 | `outputs/experiments/cmc_estimator_review/ANALYSIS.json` |
| ĐO ĐƯỢC | CV giảm91.621% conditional position-sampling gradient variance ở8 điều kiện, frozen W,4 câu train ngắn | `outputs/experiments/amis_cmc_estimator_gradient_audit/COMPLETE.json` |
| ĐO ĐƯỢC | CV vẫn không vượt W; tất cả consistency arms đã đủ20epochs | `outputs/experiments/cmc_estimator_review/REPORT.md` |
| ĐO ĐƯỢC | Không tập trung duy nhất ở nhóm other-seed có overlap thấp; thiếu bằng chứng cho rarity weighting | `outputs/experiments/cmc_estimator_review/shared_errors/ANALYSIS.json` |
| CHƯA BIẾT | Exact dense CMC có tốt hơn W không; clipping/Adam góp phần bao nhiêu | Chưa có phép đo tương ứng |

## 2. Tận dụng số cũ trước khi xin GPU

Đọc lại `scripts/analysis/analyze_cmc_gradient_estimator.py`: audit đã tính gradient từng vị trí nhưng **không lưu vector gradient**. Chỉ giữ second moments và trace variance; không có CE gradient, cross-position Gram matrix, optimizer state hoặc accumulated gradients.

Vẫn khôi phục được hai norm bằng đồng nhất thức:

\[
\|\bar c\|^2=E_t\|c_t\|^2-\operatorname{tr}\operatorname{Cov}_t(c_t),
\]
\[
\|\bar c-\bar d\|^2=E_t\|c_t-d_t\|^2-\operatorname{tr}\operatorname{Cov}_t(c_t-d_t).
\]

Ở đây c,d là gradient phụ CMC và direct SKL, trung bình đều theo vị trí của **một câu cố định**, không gồm CE/beta. Tỷ lệ `||mean(c-d)||/||mean(c)||`:

| Train row | Dropout19421 | Dropout19422 |
|---|---:|---:|
|1001|0.575534|0.569681|
|2335|0.052659|0.344172|
|3643|0.432141|0.541002|
|2193|0.328906|0.425760|

**[ĐO ĐƯỢC]** Dense CMC và direct có gradient phụ khác nhau trong cả8 điều kiện. **Không suy ra góc, xung đột CE, hướng có hại hay khác biệt update Adam** từ norm này; thay đổi có thể bao gồm magnitude. Không ngoại suy4 câu ngắn sang cả training.

Lệnh CPU, conda clrr:

```powershell
python -X utf8 outputs/experiments/cmc_estimator_review/shared_errors/recover_dense_moments.py
```

Output: `outputs/experiments/cmc_estimator_review/shared_errors/DENSE_MOMENT_RECOVERY.json`, gồm SHA256 nguồn.

## 3. Một ứng viên tối thiểu được đưa ra kiểm tra

**Exact-position CMC**, không đặt tên như một method mới.

- Hiện tượng: sampled-direct thua dense W; CV giảm variance nhưng chưa thắng W.
- Giả thuyết: position sampling có thể góp phần làm khác update sau accumulation/clipping. Đây không phải kết luận nguyên nhân duy nhất.
- Thay đổi duy nhất: lấy tổng đầy đủ mọi supervised position thay vì lấy mẫu một vị trí/câu.
- Giữ nguyên: channel hybrid top4 hiện có, CE, beta1.25, backbone/native inference, optimizer, global clipping1 sau accumulation32, dữ liệu và selector BLEU/ES10.

Với N tổng supervised tokens trong microbatch, hai dropout distributions p1_it,p2_it và channel hybrid K_it cũ:

\[
L_{CE}=-\frac1{2N}\sum_{i,t}\left[\log p_{1,it}(y_{it})+\log p_{2,it}(y_{it})\right],
\]
\[
L_{exact}=L_{CE}+\frac{1.25}{N}\sum_{i,t}\operatorname{SKL}(p_{1,it}K_{it},p_{2,it}K_{it}).
\]

SKL=(KL(P||Q)+KL(Q||P))/2. K/shortlist detached; gradient qua cả hai p. Selected actions là union top4 mỗi view và gold, bỏ EOS; các action ngoài shortlist giữ identity symbols riêng. Không bỏ mass/renormalize shortlist. **Exact chỉ theo vị trí**, không phải full-vocabulary continuation oracle; hybrid K và dropout randomness vẫn tồn tại. Đây là cùng expected stop-gradient surrogate như CMC cũ, với shortlist/channel cố định khi backward, không phải derivative toàn phần xuyên qua K(theta).

Microbatch reduction và accumulation phải giữ đúng code hiện có; không âm thầm chuyển thành một token mean khác trên toàn effective batch. Nếu cần chia channel probes thành chunks, chunking chỉ là memory optimization và phải giữ cùng dropout graphs/channel.

## 4. Paper gốc đã mở và đọc phần phương pháp

| Paper | Phần đã đọc | Giống/khác và kết luận |
|---|---|---|
| [R-Drop — Liang et al., NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf) | §2.1, Eq.2–4 | Hai dropout views, supervised NLL và symmetric KL đã có. Khác ở shared continuation channel; phần này chính CMC cũ đã có, không phải nâng cấp mới. Chuẩn hóa beta phải theo code project, không đồng nhất hệ số paper vì NLL sum/mean khác nhau. |
| [Rao-Blackwellized Stochastic Gradients — Liu et al., ICML2019](https://proceedings.mlr.press/v97/liu19c/liu19c.pdf) | §2 Eq.5–8; §3 Eq.14–16 | Exact/partial summation để giảm variance là established principle. Paper xét discrete expectations tổng quát; không phải bằng chứng hiệu quả cho CMC/MT. |
| [Revisiting Gradient Clipping — Koloskova et al., ICML2023](https://proceedings.mlr.press/v202/koloskova23a/koloskova23a.pdf) | Eq.2–3; §3.1, Theorem3.1; Eq.13 | Clipped stochastic gradient có thể bị bias. Không chứng minh bias của mBART/AdamW hiện tại hoặc rằng giảm variance chắc chắn tăng BLEU. |
| [TeaForN — Goodman et al., EMNLP2020](https://aclanthology.org/2020.emnlp-main.702.pdf) | §3 và Figure1 | Dùng stacked decoders/multi-step prediction; khác detached shared-channel pushforward. Khác thiết kế không tự tạo novelty cho việc bỏ position sampling. |

**Novelty của thay đổi mới: TRÙNG nguyên lý numerical integration/variance reduction; không chấp nhận làm đóng góp độc lập.** Không kết luận toàn bộ CMC đã trùng một paper cụ thể; lượt này không cấp lại novelty clearance cho CMC cũ.

## 5. Phản biện NotebookLM

Notebook CLRR `bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890`. Chỉ gửi công thức và câu hỏi literature, không gửi dữ liệu/số riêng mới. Raw prompts/answers nằm trong `outputs/experiments/cmc_estimator_review/shared_errors/STEP5_NOTEBOOKLM_ROUND_{A,B}.json`.

- VòngA: yêu cầu reviewer ACL phản bác novelty, động cơ, compute và đề xuất phép bác bỏ.
- Phản hồiA: reject như standalone contribution; giữ làm control. R-Drop có trong notebook; Liu2019/TeaForN được báo không có trong sources, nên tôi tra paper gốc bên ngoài.
- VòngB: bảo vệ vai trò causal control, đồng thời phản bác các overclaim của chính NotebookLM: full sum không phải quality upper bound; N samples with replacement không tương đương enumeration; Gaussian covariance matching không đủ để chứng minh nguyên nhân; Adam damping không mặc nhiên vĩnh viễn. Global clipping giữ hướng từng realized vector. Protocol phải clip sau GA32, dùng TRAIN, không đổi sang validation.
- Phản hồiB hoàn tất: chấp nhận các đính chính và vẫn reject-as-method/retain-as-control. Tôi không nhận nguyên xi protocol mới của nó: enumerate **vị trí** không phải enumerate vocabulary/chuỗi output; full position sum tốn nhưng không mặc nhiên bất khả thi; Monte Carlo1024 lần không phải ngân sách đã duyệt; cosine gần1 và clipping frequency giống nhau không đủ loại mọi ảnh hưởng norm/Adam. LượtB không có source citations, chỉ dùng làm phản biện logic.
- Kết luận của tôi: không thể bảo vệ novelty độc lập cho full summation. Vai trò control có cơ sở, nhưng không được gọi là method ACL đã chốt. Verdict của NotebookLM là ý kiến phản biện, không phải bảo chứng novelty hay quyết định thật của ACL.

## 6. Phép đo bác bỏ rẻ nhất và quyết định

Không có đủ artifacts để tính E[clip(G)] từ norm/variance cũ. Phép đo còn thiếu cần checkpoint fixed và gradient vectors, nhưng chưa chạy trong bước5:

1. Tiny TRAIN fixture: enumerate toàn bộ position-sampling outcomes cho vài câu ngắn, cố định dropout/channel. Kiểm tra expected preclip gradient bằng exact-position CMC.
2. So **mean của clipped total gradients** với **clipped exact total gradient**, gồm CE và beta. Không so mỗi auxiliary norm rồi kết luận bias của total update.
3. Với micro4×GA32 thật, không thể enumerate product space: dùng Monte Carlo, giữ rõ sampling error; clip một lần sau accumulation. Có thể dùng bootstrap trên các sampling repeats có điều kiện; không coi đó là training-seed significance.
4. Nếu local clipping discrepancy không phân biệt được với sampling/numerical error, bác bỏ động cơ clipping ở fixture đó; không ngoại suy thành không có Adam/trajectory effect ở mọi checkpoint. Nếu discrepancy có thật, mới có bằng chứng cho một artifact cục bộ, vẫn chưa chứng minh nó gây giảm BLEU.
5. Không chạy pilot dài chỉ vì variance giảm. Full-sum channel cost lớn; matched-step không có nghĩa matched-compute. Một paper claim hiệu quả cần cost accounting và compute-matched W, không chỉ tái sử dụng W cũ để kết luận hiệu suất.

**Chốt bước5: không mở training.** Exact-position CMC chỉ giữ làm diagnostic control có điều kiện; không đề cử làm phương pháp ACL mới. Nguyên nhân chưa đủ xác định để hợp thức hóa thêm module/loss. Nếu chẩn đoán tiếp không tách được hai cách giải thích objective-change và estimator/update-change, dừng nhánh CMC thay vì sweep thêm.

## 7. Nhật ký truy vấn web

1. `R-Drop regularized dropout original paper symmetric KL 2021 pdf`
2. `consistency regularization supervised gradient conflict projection gradient vaccine PCGrad`
3. `machine translation consistency regularization gradient conflict cross entropy regularization`
4. `site.aclanthology.org "consistency" "token" "sampling" "R-Drop"`
5. `site.arxiv.org continuation distribution consistency regularization marginalization language model`
6. `site.semanticscholar.org Rao Blackwellized stochastic gradients discrete distributions Liu 2019`
7. `site.aclanthology.org "continuation" "consistency" "translation"`
8. `site.arxiv.org "R-Drop" "variance" "sampling"`
9. `site.semanticscholar.org "continuation" "consistency regularization"`
10. `"stochastic gradient" "clipping" "bias" convergence sampling`

Chỉ bốn primary papers ở §4 được dùng làm nguồn cho kết luận. Search hit khác không được coi là đã đọc/xác minh. Không tuyên bố exhaustive search hoặc chưa ai làm.
