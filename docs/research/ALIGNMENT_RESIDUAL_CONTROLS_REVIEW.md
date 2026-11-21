## Trạng thái hiện hành — 29/10/2026: mở replication RTX sau gates

Người dùng đã yêu cầu tiếp tục chạy theo plan trên RTX. Khối được mở gồm **11 runs mới**: C0 và learned-donor seeds43/44 (4); M0 seeds42/43/44 (3); CE_DONOR và CE_LOCAL seeds43/44 (4). Dùng lại bốn nhánh seed42 đã hoàn thành và baseline thường có sẵn; không train lại số đã có. M0 là control không donor/module nhưng cùng target dummy/RNG schedule, không phải đổi baseline. Các trạng thái HOLD block9 ở phần lịch sử bên dưới đã được thay thế bởi quyết định mở replication này; HOLD test/A100 và chưa xác nhận novelty vẫn giữ.

Recipe: `configs/experiments/amis_alignment_residual_replication_rtx.json`; workspace: `outputs/experiments/amis_alignment_residual_replication_rtx/`. Chạy đúng RTX FP32/SDPA, LR5e-5, B4×GA32, 20epoch/ES10, cùng dữ liệu, pretrained revision, selection validation chrF++ beam1 earliest-best. Mỗi run phải PASS CPU/GPU gate trước train. Queue tuần tự, tự upload public và xác minh trước khi xóa weights riêng của run; không tắt GPU/máy dùng chung. Không tự chạy test hoặc chuyển sang A100.

| Thứ tự | Runs mới | Mục đích |
|---|---|---|
|1–4|C0 S43 → learned-donor S43 → C0 S44 → learned-donor S44|Kiểm tra correction A(b−u) so với A(b) cùng dung lượng|
|5–7|M0 S42, S43, S44|Ô zero của factorial, cùng RNG schedule|
|8–11|donor S43 → local S43 → donor S44 → local S44|Đủ 3 seeds cho donor-only/local-only và interaction|

Khóa thiết kế trước các runs bổ sung: primary contrast C0−learned-donor, primary metric BLEU zh; không đổi selection chrF++. Secondary: C0−baseline thường/M0/donor/local. Báo đủ từng seed, mean và SD; interaction = C0−donor−local+M0. Gate phát triển: mean C0−baseline ≥+.30 BLEU, mean chrF0 gap ≥0; mean BLEU so với từng matched control >0 và ít nhất2/3 seeds dương. Đây là development đã xem seed42, không phải confirmation sạch. Holm family mới gồm5contrasts×3metrics×3seeds=45; interaction chỉ mô tả, giữ nguyên family thống kê cũ. Không tăng seeds để tìm p<.05. Novelty vẫn cần lập luận thiết kế và đối chiếu prior, không suy ra từ điểm tăng.

Preflight tài nguyên: Quadro RTX6000, free23542MiB, GPU0%, ổ còn132GB, training lock trống. Local M0 tests2/2 PASS (native loss/gradient/RNG và checkpointing/save-load). Trạng thái triển khai thực tế xem receipts trong workspace; không coi phần kế hoạch này là bằng chứng train đã hoàn tất.

---

# Alignment residual: kiểm chứng đối chứng và quyết định tiếp theo

## 29/10/2026 — Learned-donor seed42 hoàn tất

CE_LEARNED_DONOR validation BLEU **17.0231**, chrF++ **14.3745**, chrF0 **16.4864**; best epoch20, train đủ20 epoch, stop=epoch cap. C0 tương ứng **17.9412 / 14.9673 / 17.2061**. C0−learned-donor: BLEU **+0.9181**, chrF++ **+0.5927**, chrF0 **+0.7197**. Bootstrap paired 10.000/seed42 trên 576 câu validation: BLEU CI95% [−0.1382,+2.0449], p raw=.046795, Holm3=.102290; cả 3 phép so sánh không đạt p Holm<.05. Kết quả thăm dò một seed, validation đã dùng để phát triển; chưa xác nhận novelty hoặc hiệu quả ổn định.

Checkpoint public `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/alignment-learned-donor-ce-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-rank64/`, upload receipt verified revision `440c15907ad71dcc63939c47889e4f018cd45c6e`, archive SHA256 `6a8fb4aa1d4a86d99494536fc2ded33d9a2961e18b9d17ddc6a856c79f2fff2d`. Đã xác nhận không còn best.zip/best_model/checkpoint-* của run trên RTX. SHA metadata/predictions khớp receipt; protocol và metric replay PASS. Thống kê: `outputs/experiments/amis_learned_donor_ce_rtx/analysis/statistics/PAIRED_BOOTSTRAP.md`. Không đánh giá test.

## Trạng thái — 29/10/2026

**Follow-up sau review:** tác giả đã duyệt triển khai một learned-donor CE
pilot seed42. Code riêng, CPU16/16tests và GPU cache/reload/resume gate PASS;
queue PID2506609 đã TRAINING theo recipe
`configs/experiments/amis_learned_donor_ce_rtx.json`.
Đây là thực thi bước kiểm tra control trước block lớn; HOLD block9/test/A100
giữ nguyên. Trạng thái/receipts mới xem
[runbook](../operations/ALIGNMENT_RESIDUAL_CE_CONTROLS_RTX.md).

**Audit PASS; quality/novelty HOLD. Chưa mở block 9 runs, test hoặc A100.**
Hai CE controls và ba checkpoint probes đã hoàn tất. Kết quả có tín hiệu đáng
nghiên cứu về phối hợp donor–module, nhưng chưa xác nhận tăng chất lượng so với
baseline hoặc một đóng góp phương pháp đủ khác prior. C2 vẫn NO-GO theo gate cũ.

## Material Passport

- Phạm vi: Amis→Mandarin, RTX6000, FP32/SDPA, seed42; validation 576 câu đã dùng
  trong development. Đây không phải kết quả confirmation hoặc bảng A100.
- Đầu vào: các checkpoint đã chọn trước theo validation chrF++/beam1, original
  references. Audit không đổi selection, dữ liệu, method hoặc protocol.
- VERIFIED: SHA/size của 24 file metadata/predictions/state nhỏ tại pinned HF
  revisions; replay ba corpus metrics cho bốn systems; selection; probe replay.
  ANALYZED: diễn giải cơ chế và novelty. Không tải weights về Windows.
- Backup weights dựa trên receipts upload đã verified; audit này không tải lại
  toàn bộ weights để tự kiểm hash một lần nữa.
- Artifacts: `outputs/experiments/amis_alignment_residual_ce_controls_rtx/analysis/`
  gồm `AUDIT.json`, `statistics_design.json`, `inputs/`, `statistics/`.
- Probe gốc: `outputs/experiments/amis_alignment_residual_checkpoint_probes_rtx/`.
  `COMPLETE.json`: COMPLETE; mọi probe 0 optimizer steps, parameter versions
  unchanged; checkpoint SHA/revision và worker SHA khớp.

## 1. Số đã xác minh

| System | BLEU | chrF++ | chrF0 | Best epoch | Train đến |
|---|---:|---:|---:|---:|---:|
| Baseline R1, dùng lại |17.6823|14.7030|16.9547|19|20|
| C0: donor + difference module, CE |17.9412|14.9673|17.2061|14|20|
| CE_DONOR: donor, zero correction, CE |17.0378|14.2944|16.3933|19|20|
| CE_LOCAL: local module, CE |17.0637|14.2563|16.3762|17|20|

Resolved Trainer arguments của cả bốn systems giống nhau trừ output/log paths.
Metadata pretrained revision, data hashes, seeds, LR5e-5, B4×GA32, max length256,
20 epochs, ES10, original-reference selection đều khớp. Baseline SDPA được xác minh
bằng backend audit cũ; không thêm field hồi tố vào metadata baseline.
Đợt này không có model nào dừng sớm: best epoch và completed epoch là hai trường khác nhau.

### Thống kê

Paired corpus bootstrap 10.000 draws, RNG42. Family đã ghi trong runbook:
**C0−CE_DONOR, C0−CE_LOCAL, CE_LOCAL−baseline × ba metrics = Holm9**.
Đây là post-hoc development; không thay family12/18 của audit pilot cũ.
Native SacreBLEU p-values được đối chiếu khớp độc lập cho mọi pair/metric.

| Contrast | Metric | Δ | Marginal 95% CI | p raw | p Holm9 |
|---|---|---:|---|---:|---:|
| C0−CE_DONOR | BLEU |+.9034|[−.1104,+1.9326]|.038396|.194581|
| | chrF++ |+.6729|[−.0217,+1.3874]|.030497|.194581|
| | chrF0 |+.8128|[+.0529,+1.6113]|.020998|.180882|
| C0−CE_LOCAL | BLEU |+.8775|[−.2760,+2.0221]|.059794|.239176|
| | chrF++ |+.7109|[−.0191,+1.4493]|.027797|.194581|
| | chrF0 |+.8299|[+.0398,+1.6319]|.020098|.180882|
| CE_LOCAL−baseline | BLEU |−.6186|[−1.7995,+.5569]|.116888|.242976|
| | chrF++ |−.4467|[−1.2548,+.3499]|.109989|.242976|
| | chrF0 |−.5785|[−1.4672,+.2810]|.080992|.242976|

**0/9 significant sau Holm.** Không gọi nonsignificance là equivalence/harm.
CI là marginal percentile, không phải simultaneous CI hoặc phép đảo native
SacreBLEU p; không ép hai thủ tục phải cho cùng một quyết định trước hiệu chỉnh.
576 câu bootstrap không phải 576 training seeds. Resampling không bao phủ
training randomness, checkpoint selection hoặc nhiều vòng chọn phương pháp.

So với baseline, C0 chỉ +.2589 BLEU; audit cũ đã tính CI[−.9107,+1.4384],
p raw .230177, p Holm .644336 trong exploratory12. Giữ nguyên family và số đó.
Không lấy gain gần +.9 so với controls để viết thành gain +.9 so với baseline.

## 2. Frozen probes nói gì?

### Can thiệp suy luận trên C0

| Can thiệp | BLEU | Δ vs all-on | chrF++ | Token CE |
|---|---:|---:|---:|---:|
| All-on |17.9412|0|14.9673|2.809377|
| Module-off |18.0245|+.0833|15.0082|2.809419|
| Donor-off |14.0572|−3.8840|12.0589|2.986866|
| Content-token donor shuffle |14.9135|−3.0277|11.1652|2.951503|

Audit replay đủ 4×576 prediction rows, hai generation metrics và token-weighted
CE; all-on khớp 576 predictions C0 gốc. Các delta này là mô tả, chưa chạy một
family kiểm định can thiệp mới. Shuffle giữ padding/special; thay cả donor
addition và input correction. Đây là tác động can thiệp lên checkpoint đã
co-adapt, không phải retrained ablation hay chứng minh donor mang đúng ngữ nghĩa.

Điều có cơ sở: checkpoint đang dùng đường donor. Điều chưa có cơ sở: module là
training scaffold gây ra gain, module hoàn toàn vô ích, hoặc bỏ module khi train
sẽ giữ được điểm. Module-off chỉ +.0833; không dùng điều đó để chọn deployment
variant tốt nhất rồi coi là xác nhận độc lập.

C0 correction/current không nhân α nằm trong [.0000864,.003496] trên các norm
samples đã đo, còn tổng α(donor+correction)/current trong [.01160,.08982].
Correction thực sự cộng vào residual còn nhân α=.1. C2 correction/current lớn
hơn [.1081,.2370] nhưng quality kém C0; norm lớn không tự đồng nghĩa hữu ích.
Đây là range trên các samples/layers của probe, không phải toàn bộ validation.

### Gradient đúng policy C1/C2

Cùng bốn batch×bốn câu, eval/dropout-off FP32. Mean của bốn ratios/cosines:

| Model / nhóm | ‖λ∇LSR‖ / ‖∇CE‖ | Cosine(λ∇LSR,∇CE) |
|---|---:|---:|
| C1 module |.5365%|+.13835|
| C1 backbone |.1842%|+.06359|
| C2 module |1.1047%|+.07248|
| C2 backbone |0%|không xác định vì zero gradient|

Confinement làm đúng: C2 không truyền gradient LSR vào backbone trong probe.
Không thấy bằng chứng xung đột hướng gradient mạnh phổ quát tại các batch này.
Không suy rộng snapshot thành cumulative Adam update hoặc nguyên nhân training
kém. Không có cơ sở từ đây để tăng λ, bỏ stop-gradient hoặc gọi collapse.

## 3. Công thức hiện tại và đối chứng còn thiếu

Với tầng encoder l≥3, u_l=F_l(h_(l−1)), b_l=sg(h_(l−2)) là output đã rewiring
được cache. A_l(z)=U_l GELU(D_l RMS(z)), rank64, U_l zero-init:

```text
C0:       h_l = u_l + .1 [b_l + A_l(b_l − u_l)]
CE_DONOR: h_l = u_l + .1 b_l
CE_LOCAL: h_l = u_l + .1 A_l(−u_l)
M0 cần có:h_l = u_l                         (cùng target dummy như C0)
```

Tầng1–2 không thêm route. C0/CE_LOCAL thêm1.310.720 tham số; toàn backbone vẫn
được CE fine-tune. Đây không phải PEFT hoặc parameter-free method. Donor control
không có tham số thêm; chữ rank64 trong suffix là recipe family, không phải module.

C0 và hai controls cùng target no-grad dummy pass trong training để khớp
dropout draw schedule; baseline R1 thông thường không có dummy đó. Baseline vẫn
là đối sánh thuật toán cần báo, nhưng chưa dùng nó làm cell00 của factorial
để kết luận interaction nhân quả. Không biết dummy có gây gap hay không.
**Interaction C0−CE_DONOR−CE_LOCAL+M0 hiện chưa đo được vì thiếu M0.**

## 4. Đối chiếu novelty bằng primary sources

Đọc phần method/equations của các nguồn dưới đây; đây là targeted prior audit,
không phải tuyên bố đã tìm hết mọi related work. Khác task/backbone hoặc thay
một hằng số không tự đủ thành đóng góp ACL.

| Prior | Phần đã có | Khác biệt cần chứng minh của thiết kế này |
|---|---|---|
| [Houlsby et al., ICML2019](https://proceedings.mlr.press/v97/houlsby19a.html) | Bottleneck residual adapters; backbone frozen trong phương pháp đó | Module nhỏ tự nó không mới; ở đây full fine-tuning, input là donor−recipient |
| [SkipCLM, NAACL SRW2025, §3.1](https://aclanthology.org/2025.naacl-srw.50.pdf) | Learned skip filter, scaled addition, warmup kết hợp cross-lingual alignment; Eq. Rα=Hα+λ/3 Skip(Hβ) | Skip của prior nhận donor; ở đây rank bottleneck nhận hiệu với recipient, cached stop-gradient donor, encoder route l−2; phải so với learned-donor control cùng recipe |
| [Lexical Shortcuts, WMT2019, §2.2](https://aclanthology.org/W19-5211.pdf) | Gated lexical shortcuts; so sánh lexical/latent để trộn attention K/V | Gating, lexical preservation và ghép hai biểu diễn không phải claim mới; ở đây can thiệp block output thay vì embedding→K/V |
| [NeuTRENO, NeurIPS2023, Eq.26](https://proceedings.neurips.cc/paper_files/paper/2023/file/fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf) | Attention output cộng λ(v_first−v_current), fidelity/anti-smoothing motivation | **Hiệu hai tầng đã có prior.** Ở đây learned nonlinear correction của hiệu hidden outputs với donor l−2; chưa chứng minh phần khác này tạo lợi ích |
| [Value Residual Learning, ACL2025, §3](https://aclanthology.org/2025.acl-long.1375.pdf) | First-layer value residual; learned/fixed mixing, dense variants | Không được claim cross-layer reuse chung là mới; khác ở hidden block output và difference-conditioned correction |
| [CAFE, arXiv v1 13/10/2026, Eq.6–8](https://arxiv.org/html/2610.19134v1) | Zero-up bottleneck và feature-loss gradient confinement vào adapters; supervised loss vẫn train backbone | Confined LSR của C2 không mới chỉ vì đổi ASR→MT. Khác objective/input không cứu được quality gate đã fail. Nguồn này là preprint |
| [AdapterDrop, EMNLP2021, §3](https://aclanthology.org/2021.emnlp-main.626.pdf) | Bỏ adapters ở lower layers khi train/inference; variants specialized/robust | Module-off một checkpoint không đủ claim training-only module mới hoặc zero-cost method |

**Novelty hiện HOLD.** Không khẳng định C0 trùng hệt prior; cũng không bảo đảm
novel chỉ vì tổ hợp khác. Hypothesis có thể bảo vệ là correction phụ thuộc
donor−recipient giúp route hữu ích hơn learned-donor skip cùng capacity. Hiện
chưa có đối chứng A_l(b_l), nên hai CE controls mới chưa trả lời hypothesis đó.
Không dùng lý thuyết anti-smoothing của NeuTRENO làm theorem cho code hiện tại.

### Candidate/comparator card cho bước kế tiếp

- Candidate giữ nguyên: C0 hiện có, không thêm λ/gate/topology mới.
- Câu hỏi: input donor−recipient có lợi hơn donor-only input cho cùng module
  hay không? So sánh trực tiếp `A(b−u)` với `A(b)`, giữ fixed donor addition.
- Learned-donor control đề xuất: `h=u+.1[b+A(b)]`, layers3–12, d2,
  rank64, RMS/GELU/bias-free, zero-up init, module_seed10042; seed42.
  Backbone full fine-tuning bằng CE; λLSR=0; target dummy giống C0.
- Protocol đề xuất dùng nguyên lock RTX: FP32/SDPA, LR5e-5, B4×GA32,
  20epochs/ES10, cùng data hashes/model revision, validation chrF++ beam1,
  earliest-best. Không chuyển sang eager/BF16 hoặc thay tokenizer/proxy.
- Trước run phải có method ID/folder riêng, recipe/manifest và tests: tham số
  thêm bằng C0; zero-init forward khớp C0; target/RNG draw schedule khớp;
  backward hữu hạn đúng đường; checkpointing/reload/resume parity. Gradient
  module không bắt buộc giống C0 vì input module chính là biến can thiệp.
- Tiny GPU gate chỉ khi VRAM đủ, không dừng process của người khác; tất cả
  file/env/cache trong `/workspace/ura/thanhdat/projects/CLRR/`.
- Đánh giá validation-only; không thay checkpoint theo BLEU hoặc xem test.
  Một seed chỉ screening: point estimate C0 không hơn learned-donor thì chưa
  có cơ sở gán gain cho difference input; không tự kết luận equivalence.
- Failure/rollback: gate correctness fail thì dừng trước train; nếu input
  difference không thể phân biệt được khỏi prior về giả thuyết và đối chứng,
  không quảng bá thành novelty chính hoặc tự mở replication lớn.
- Thiết kế này đã được tác giả duyệt triển khai sau review; code tách riêng
  và CPU gates PASS, GPU gates phải PASS trước training. Không sửa
  implementation/receipts C0 đã có. Kết quả quality của arm mới chưa có.

## 5. Kế hoạch tiếp theo — thiết kế, chưa launch

### Quyết định hiện tại

| Gate | Kết luận | Hành động |
|---|---|---|
| Artifact / metric / selection integrity | PASS | Có thể dùng các số trên trong sổ development |
| C2 confined LSR quality | NO-GO cũ giữ nguyên | Không mở rộng C2 hoặc tự tăng λ |
| C0 vượt baseline ổn định | Chưa xác nhận | Không chuyển thẳng A100/test hoặc tuyên bố method chính |
| Novelty đủ căn cứ mở chiến dịch | HOLD | Cần candidate card phân biệt prior và control A(b) trước |
| Block 9 runs | HOLD, bản thiết kế bên dưới | Không xem 9 runs là đã duyệt/đang chạy |

Trước training, viết candidate card giữ nguyên C0 làm ứng viên hiện có; định
nghĩa learned-donor comparison `u+.1[b+A(b)]` như adapted control, không gọi là
reproduction nguyên bản SkipCLM. Chốt cùng params/init, target dummy, recipe,
checkpoint selection; chỉ biến đổi input A(b−u)→A(b). Nếu không có câu hỏi
khoa học rõ ngoài thêm adapter vào skip thì không mở block lớn chỉ để tìm số đẹp.

### Block replication 9 runs được tính như sau, nếu được mở sau gate

| Cell | Seed42 | Seed43 | Seed44 | Runs mới |
|---|---|---|---|---:|
| M0: CE, không donor/module, cùng dummy schedule | mới | mới | mới |3|
| CE_DONOR | reuse | mới | mới |2|
| CE_LOCAL | reuse | mới | mới |2|
| C0 | reuse | mới | mới |2|
| **Tổng mới** | | | |**9**|

Ba baselines thông thường có sẵn giữ để đối chiếu chính; M0 là đối chứng RNG
schedule riêng, không thay tên hoặc ghi đè baseline. C3 có LSR không thay CE_LOCAL.
Block9 trả lời độ lặp lại/factorial, **chưa đủ trả lời novelty A(b−u) vs A(b)**.
Nếu thêm learned-donor control cùng3seeds thì thêm3runs, tổng12 mới; đây chỉ là
phương án ngân sách rõ ràng, chưa tự cộng vào queue hoặc giả vờ nằm trong9.

Trình tự đề xuất: candidate card/CPU correctness → tiny RTX GPU gate → seed42
learned-donor control trước → review cùng toàn bộ seed42 đã có → quyết định có
cần block9 và hai learned-donor seeds43/44 không. Screening seed42 có nhiễu;
không yêu cầu p<.05 của một seed làm điều kiện duy nhất và không khẳng định
equivalence khi gap nhỏ. Không train lại C0/controls42 đã verified.

Trước chạy block đa seed phải khóa primary metric/contrasts, ngưỡng thực dụng,
chiều của gate và family thống kê. Đề xuất thảo luận: mean C0−baseline≥+.30 BLEU,
chrF0 không âm, mean C0−từng matched control dương và ít nhất2/3seeds dương.
Đây là **ngưỡng đề xuất mới, chưa được duyệt/lock**, không phải gate cũ và không
bảo đảm significance với3seeds. Không tăng seed đến khi p nhỏ; nếu cần confirmation
thì chọn cỡ mẫu trước, tách khỏi development đã xem.

Nếu replication/novelty fail: dừng C0 làm main method; giữ số âm và insight,
thiết kế thay đổi có cơ sở tiếp theo sau review. Không tự đổi data/λ để cứu claim.
Nếu pass: khóa method rồi lập evaluation Ê Đê và A100 với controls phù hợp,
giữ cohort RTX/A100 riêng; không coi Ê Đê đã được xem trong development là holdout mới.

## 6. Tái lập audit trên CPU

Trong conda `clrr`, tại root repo:

```bash
python scripts/analysis/audit_alignment_residual_controls.py --pilot outputs/experiments/amis_alignment_residual_rtx/analysis/inputs --controls outputs/experiments/amis_alignment_residual_ce_controls_rtx --probes outputs/experiments/amis_alignment_residual_checkpoint_probes_rtx --output outputs/experiments/amis_alignment_residual_ce_controls_rtx/analysis
python scripts/analysis/paired_checkpoint_statistics.py --predictions-dir outputs/experiments/amis_alignment_residual_ce_controls_rtx/analysis/inputs --references data_processed/amis_mandarin/validation.csv --design outputs/experiments/amis_alignment_residual_ce_controls_rtx/analysis/statistics_design.json --output outputs/experiments/amis_alignment_residual_ce_controls_rtx/analysis/statistics --split validation
```

Nếu thiếu file nhỏ, thêm `--fetch-missing` vào lệnh audit; chỉ tải file size<1MB
tại pinned revisions rồi kiểm SHA/size. Không tải best.zip. CRLF→LF chỉ được
chấp nhận khi bytes sau chuyển khớp SHA gốc; bản gốc không bị sửa.

## 7. Kiểm tra diễn giải thống kê — 11/11 mục đã rà

| Nguy cơ | Xử lý/phạm vi |
|---|---|
| Simpson | Không kết luận mọi nhóm câu đều tăng từ corpus score; chưa audit strata hai controls |
| Ecological | Corpus delta không suy thành mọi câu/người nói/ngôn ngữ đều hưởng lợi |
| Berkson | Một dataset/seed và validation được chọn, không suy thành population đại diện |
| Collider | Không conditioning trên nhóm câu được candidate sửa đúng để suy hiệu ứng chung |
| Base-rate neglect | Không dùng phân loại chẩn đoán/PPV; không áp dụng trực tiếp |
| Regression to mean | C0 được chọn sau pilot; cần seeds khác, không gọi replication đã xong |
| Survivorship | Báo cả hai controls và giữ C1/C2/C3 fail trong pilot cũ |
| Look-elsewhere | Báo toàn Holm9, không chọn p raw hoặc đổi/xóa family cũ |
| Forking paths | Post-hoc development ghi rõ; ngưỡng tương lai là đề xuất, không prereg hồi tố |
| Causation | Inference intervention không chứng minh lợi ích nhân quả của training module |
| Reverse causality | Không suy norm/gradient snapshot là nguyên nhân của score |

Độ tin cậy diễn giải quality: CAUTION. Không có claim ACL acceptance hoặc novel
được bảo đảm từ audit này. Tài liệu pilot cũ và các con số lịch sử giữ nguyên.
