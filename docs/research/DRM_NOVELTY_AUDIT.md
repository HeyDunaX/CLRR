# DRM — novelty audit trước khi chốt đóng góp
Ngày: 2026-11-18. Chỉ đọc source/kết quả và audit literature; không mở GPU/training hoặc đổi recipe.

## 1. Quyết định hiện hành

**GẦN prior; giữ làm ứng viên nghiên cứu, chưa chốt là đóng góp đủ mạnh cho ACL.**
Trong các method sections đã đọc chưa tìm thấy đúng quy tắc DRM. Đây là kết luận trong phạm vi tìm kiếm, không chứng minh toàn literature không có.
Một công thức khác và điểm cao hơn baseline chưa chứng minh lợi ích của phần mới; đối chứng trực tiếp là W/R-Drop cùng objective.

## 2. Cơ chế đã đọc trong code

Nguồn: src/methods/decision_residual_memory/v1/model.py, update_memory và _last_layer.

- Chỉ đổi cross-attention layer decoder cuối; R khởi tạo native V, Q/K giữ phép chiếu native.
- c_t=sum_j A_read,tj R_tj; output p_t tính trước cập nhật.
- u_t=stopgrad(normalize(partial log p_t(a_t)/partial c_t)), từng head.
- R_(t+1,j)=R_tj-eta*A_tj*max(R_tj dot u_t,0)*u_t.
- Train dùng y_t sau khi đã tính p_t, inference dùng token trong prefix đã sinh; không được dùng gold lúc generation.
- Eta=.1; objective giữ paired CE+1.25 symmetric KL của W.
- Không thêm trainable parameters; có chi phí sequential/autograd, native KV cache bị tắt trong implementation hiện tại.

**SUY LUẬN toán học:** giữ u và A cố định, toán tử là một bước gradient descent trên E(r)=.5[max(r dot u,0)]^2, step size eta*A.
Không nhận gradient descent, rank-one attenuation hay norm contraction làm novelty. Diễn giải này không có nghĩa toàn training algorithm là gradient descent của E: u và A phụ thuộc forward, và gradient qua chúng đã được quy định riêng trong code.

## 3. Primary methods đã mở và đọc

| Prior | Phần đã đọc | Điểm đã có | Khác biệt DRM cần kiểm chứng |
|---|---|---|---|
| [Coverage Embedding Models for NMT, EMNLP2016](https://aclanthology.org/D16-1096.pdf) | §3.1.2 Eq5, §3.2 | Per-source auxiliary coverage vector, trừ target embedding chiếu theo attention, đưa coverage vào attention energies | DRM cập nhật actual native value slots; hướng là task-output derivative |
| [Modeling Past and Future for NMT, TACL2018](https://arxiv.org/html/1711.09502v2) | §4.1, subtraction variants | Recurrent translated/untranslated state, contextual subtraction | Không thêm Past/Future RNN; decision-conditioned native values |
| [PPLM, ICLR2020](https://arxiv.org/html/1912.02164) | §3.1–3.3 Eq3 | Normalized output-attribute gradients cập nhật cached latent K/V khi generation | DRM không attribute classifier; dùng current-token task logprob để attenuate source values sau quyết định, ảnh hưởng bước sau |
| [Titans, 2025](https://arxiv.org/html/2501.00663v1) | §3.1 Eq8, associative-memory loss, §3.2 | Memory update bằng gradient, forgetting | Associative reconstruction gradient khác derivative của task output với context; không nhận gradient-memory là mới |
| [Erase-then-Delta, 2026](https://arxiv.org/html/2606.26560) | §3.3 Eq8–13 | Independent rank-one erase, squared-readout objective, orthogonal readout preservation | EDA erase theo learned key-space address trong associative state; DRM per-source value-space, decision gradient, positive part |
| [CARVE, 2026](https://arxiv.org/html/2606.27229) | §4.2 Eq7–8 | Low-rank chunk-boundary state readout quyết định erase/write gates | Không phải cùng output-Jacobian-conditioned source-value recurrence |
| [The Query Knows What to Forget / QED, 2026](https://arxiv.org/html/2608.13668) | §3 Eq6–12 | Query-derived key-orthogonal erase direction; kiểm soát direction bằng random arm | Read-dependent erasure đã có; DRM direction còn đi qua output/decoder Jacobian và cập nhật source value slots |

Những khác biệt trên loại kết luận “đúng cùng phương pháp” với các công thức cụ thể đã đọc; chúng chưa chứng minh đóng góp đủ lớn. Các preprint2026 là prior cần đối chiếu, không tự nhận đã peer-review.

## 4. NotebookLM debate

Notebook bf1aec9f-aab7-4a1d-8ca5-d5fbea3a1890; conversation8ec5ca51-f2be-4cca-a640-c5b4035a99d5.
Payload chỉ gồm công thức và literature; không gửi kết quả/translation mới của project.
Raw câu hỏi/câu trả lời: outputs/experiments/drm_novelty_audit/NOTEBOOKLM_ROUND_A.json và ROUND_B.json.

- Vòng A: phản bác vector coverage variant, mất parallelism, one-to-many và attention sai.
- Vòng B: sau khi cung cấp phương trình đã đọc, phân biệt distinct implementation với đủ scientific contribution; vẫn yêu cầu directional controls và cơ chế.
- **Không nhận lỗi của NotebookLM:** coverage initialization zero/source states trong A; diễn giải CARVE sai trong A; ngưỡng cost>50%/gain<.3 tự đặt; coi overlap family là exact equivalence; “against attention over-concentration” trong suggested claim chưa có dữ kiện.
- Vòng A sources_used rỗng và tự báo các closest papers không có trong notebook. Vòng B dùng phương trình do assistant đọc từ primary rồi cung cấp; các citations nó trả không xác nhận toàn bộ closest papers. Vì vậy primary paper reading là căn cứ novelty; NotebookLM là công cụ phản biện, không chứng nhận.
- Không bắt random control phải thất bại hoặc dùng fixed inference interventions thay cho matched training control để kết luận chất lượng sau training.

## 5. Số hiện có và giới hạn

Cùng RTX seed42 original Amis test575, beam4; checkpoint chọn validation BLEU:
- Baseline BLEU19.392604.
- W/R-Drop BLEU21.163097.
- DRM BLEU21.098343.
- DRM−baseline=+1.705739; DRM−W=−.064755.

Nguồn: docs/experiments/EXPERIMENTS_RTX6000.md mục AC replication test; outputs/experiments/amis_drm_validation_fix_dispatch/verification.json; outputs/experiments/amis_ac_seed42_test_statistics/statistics/paired_bootstrap.json (baseline/W).
DRM mới một seed; chưa có kiểm định DRM−W trong audit này. Không gọi gần bằng là statistical equivalence. Không dùng AC bootstrap làm p-value của DRM.

## 6. Đóng góp có thể bảo vệ và kiểm chứng cần thiết

**RQ:** Điều kiện hóa cập nhật source-value memory bằng độ nhạy của quyết định dịch có giúp hơn attention/context/directionless attenuation không?

Claim có điều kiện: một quy tắc cập nhật native cross-attention values theo local sensitivity của quyết định vừa hoàn tất, không thêm tham số, với lợi ích gia tăng được đo so với W và các hướng cập nhật đơn giản hơn.
Không nói đã xác định đúng semantic consumption, giữ nguyên nghĩa, causal coverage, hay thuật toán memory hoàn toàn mới.

| Thứ tự | Kiểm chứng | Ý nghĩa quyết định |
|---|---|---|
|1|Đọc lại receipts/probes sẵn có; eta0 so native W về logits, gradients, replay và generation theo correctness contract|Không quy sai tác dụng cho sequential wrapper hoặc lỗi code; kiểm tra lại chỉ khi receipt thiếu|
|2|Frozen TRAIN/validation diagnostic: direction/context/random/scalar attenuation, kiểm soát mức thay đổi memory; kiểm tra token sau vẫn cần cùng thông tin|Đánh giá giả thuyết, không phải thay thế training ablation; annotation thiếu thì ghi proxy|
|3|Matched training controls ở seed42: W có sẵn sau audit, DRM có sẵn; context-direction và random-direction là hai controls đầu tiên. Khóa step calibration trên TRAIN trước; không tune test, không tự norm-match adaptive mỗi batch|Nếu hướng gradient không tốt hơn controls trong uncertainty, novelty cơ chế yếu; không tiếp tục sweep để cứu claim|
|4|Nếu hướng còn sống: >=3 paired seeds, ít nhất2 language pairs, metrics/protocol lock, paired corpus bootstrap và seed-level uncertainty, cost/inference latency|Đánh giá ổn định và generality; datasets đã dùng chọn method phải ghi development|
|5|A100 confirmation sau khóa method; audit reuse comparators, thêm coverage comparator/adaptation đủ rõ nếu claim vượt coverage|Chỉ tái dùng đối sánh khi protocol phù hợp; không pool RTX với A100|

Target-embedding directional ablation cần định nghĩa explicit mapping sang per-head value space và ghi tham số của map; không gọi nó faithful Mi2016. Mi2016 Transformer adaptation phải ghi rõ phần adaptation, không nhận là original architecture replication.
Nếu claim chỉ là chất lượng dịch, norm-preservation toy và công thức khác chưa đủ; nếu mean gain so W không rõ mà chi phí cao, chưa có case chất lượng/hiệu quả cho paper. Có thể làm bài cơ chế với kết quả phù hợp, nhưng chưa có bằng chứng đó ở đây.

## 7. Nhật ký tìm kiếm

Phạm vi: web search, primary arXiv/ACL Anthology methods; không phải systematic review toàn bộ literature. Search hit chưa mở không dùng làm nguồn kết luận.

1. "neural machine translation" "coverage" "gradient" memory
2. "cross attention" "values" "erase" gradient decision
3. "Coverage Embedding Models for Neural Machine Translation"
4. site.aclanthology.org "coverage" "gradient" translation
5. site.arxiv.org "attention" "decision" "erasure" memory
6. site.semanticscholar.org "translation" "memory" "subtraction"
7. site.arxiv.org "Erase-then-Delta" "CARVE"
8. "translation" "coverage" "output gradient" memory
9. "attention" "gradient" "value erasure"
10. "decision" "coverage" "memory" "translation" gradient
11. "neural machine translation" "erasure" gradient memory
12. "translation" "gradient-guided" "coverage"
13. "attention" "gradient-guided" "forgetting"
14. "memory" "log probability" "erasure" transformer
15. "Plug and Play Language Models" hidden activations gradients

Không launch run mới trong audit này.

