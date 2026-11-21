## Trạng thái hiện hành — 06/11/2026

Bốn runs, test và frozen validation diagnostics đã hoàn tất. C0 NO-GO validation; test C0−baseline −0.1200 BLEU, CI95% [−1.0910,+0.5726], p Holm9=1. Không mở replication/A100. Xem bảng và receipts trong `docs/experiments/EXPERIMENTS_RTX6000.md` và `docs/research/RESEARCH_INSIGHTS.md`. Các phần dưới là kế hoạch/lịch sử tại thời điểm ghi, không phải queue đang chạy.

---

# Ê Đê → Việt: dataset thứ hai và kế hoạch kiểm chứng

## Kế hoạch hiện hành — 31/10/2026: queue kiểm chứng chuyển ngôn ngữ của C0

### Material Passport

Tác giả yêu cầu: nếu chẩn đoán Amis chưa chắc chắn, lên kế hoạch trên Ê Đê. Tác giả đã duyệt triển khai queue RTX sau kế hoạch. Không tự mở replication hoặc test. Chẩn đoán frozen C0 ba seeds Amis đã hoàn tất riêng; không chờ test Ê Đê để sửa công thức. Không tuyên bố Ê Đê là holdout sạch: seed42 baseline/Full/NCE/P3 và test đã có lịch sử development.

Metadata baseline và Full seed42 được đọc lại từ HF ngày31/10, lưu nhỏ trong `outputs/experiments/rhade_alignment_residual_preflight/REUSE_AUDIT.json`; không tải weights. Baseline validation BLEU33.979337/chrF++50.479153; test BLEU34.021487/chrF++49.862390. Full cũ validation34.345322/50.633556; test34.133002/50.160821. Các số này thuộc **CLRR+LSR gốc**, không phải C0. Metadata có runtime versions; thiếu backend runtime receipt riêng, vì vậy reuse phải qua gate dưới đây. Tám file metadata nhỏ đã được xác minh SHA tại revision `f80e6d7cddf7b17a39c6c266227af0c47badfd82`; receipt `PINNED_METADATA_VERIFICATION.json` ghi PASS cho phạm vi này. Đây chưa phải gate tái sử dụng checkpoint hoàn chỉnh: vẫn cần đối chiếu backend, archive hash và replay trước khi chốt baseline dùng lại.

### Test được tác giả yêu cầu — 01/11/2026

Bốn checkpoint seed42 đã khóa theo validation, queue test RTX PID `3920536` tại `outputs/experiments/rhade_alignment_residual_rtx_test_queue`; kết quả tại `outputs/experiments/rhade_alignment_residual_rtx_test`. GPU đang được người khác sử dụng, queue chờ tối thiểu 16.000 MiB VRAM trống trước khi chạy. Beam4, BLEU13a/chrF++/chrF0; paired bootstrap 10.000, Holm9 cho C0 so với LearnedDonor, Baseline và M0. Test là exploratory, không đảo quyết định NO-GO validation hay tự mở replication. Upload predictions và metrics vào cùng prefix checkpoint, báo cáo vào `analysis/rhade_vietnamese/mbart-large-50/alignment-residual-pilot/seed-42/rtx6000-fp32-sdpa/test`.

### Triển khai được duyệt — queue PID3505181 đã launch

- Config: `configs/experiments/rhade_alignment_residual_rtx.json`; lock: `configs/protocols/rhade_rtx_sdpa_seed42.json`.
- Queue: **BASELINE_S42 → C0_S42 → LEARNED_DONOR_S42 → M0_S42**. Bốn runs mới, validation-only.
- Quyết định audit: baseline cũ có scores/checkpoint/runtime versions và GPU gate, nhưng không có receipt runtime attention backend. Chưa kết luận số cũ sai; không dùng nó làm baseline duy nhất của cohort explicit-SDPA mới. Chạy baseline mới ở prefix SDPA riêng, giữ nguyên số/prefix lịch sử.
- CPU gate local và remote, rồi toàn bộ bốn GPU gates phải PASS trước training dài. VRAM admission tối thiểu20.000MiB; nếu thiếu thì chờ, không giảm batch/đổi precision/tắt process khác.
- Mỗi model xong upload public vào prefix trong config; chỉ xóa weights/checkpoint tạm sau SHA/size verification. Queue dừng khi đủ4runs hoặc gặp lỗi; không tắt RTX chung.
- Lệnh tái lập (conda `clrr`): `python scripts/experiment.py build --config configs/experiments/rhade_alignment_residual_rtx.json`, tiếp theo `cpu`, rồi `launch --account 1`. Không build/launch trùng workspace đã có receipt; theo dõi bằng `monitor --account 1`.

### Câu hỏi cần trả lời

1. C0 với correction trên donor−recipient có cải thiện ngoài Amis không?
2. Hiệu ứng có vượt learned-donor cùng capacity và M0 cùng dummy/RNG schedule không?
3. Kết quả có lặp lại qua seed, hay tiếp tục validation tốt nhưng test không ổn định?

Không lấy "ngôn ngữ chưa có language token" làm bằng chứng tuyệt đối không xuất hiện trong pretraining. Không đổi data để tìm gain; giữ nguyên15.092/1.000/1.000 dòng, mọi split/text/order/hash theo manifest. Giữ `id_ID→vi_VN`. 94source test từng xuất hiện trong train vẫn giữ và báo, không tự deduplicate.

### Các run, theo từng cổng

| Giai đoạn | Nhánh | Seed | Run mới | Vai trò |
|---|---|---|---:|---|
| Audit CPU | Baseline và Full cũ |42|0|Pin revision, SHA/data/recipe/backend/selection; xác nhận số dùng lại |
| Pilot validation | C0 `alignment-residual-capacity-v1` |42|1|`h=u+.1[b+A(b-u)]`, b=stop-gradient donor l−2, rank64 |
| Pilot validation | LearnedDonor `alignment-learned-donor-ce-v1` |42|1|`h=u+.1[b+A(b)]`, cùng params/init/dummy schedule |
| Pilot validation | M0 `alignment-matched-ce-control-v1` |42|1|CE không donor/module, cùng dummy/RNG schedule |
| Chỉ nếu baseline reuse fail | Baseline chuẩn |42|+1|Cùng recipe đã khóa; giữ số cũ nguyên, không relabel checkpoint |
| Replication nếu pilot GO | C0/LearnedDonor/M0/Baseline |43,44|8|Hai seeds bổ sung, giữ cùng selector/recipe; chỉ reuse baseline nếu thật sự có và pass audit |

**Pilot:3 runs mới nếu baseline reuse PASS;4 nếu cần baseline mới.** Sau đó nếu mở replication là tổng11 hoặc12 runs mới cho cả chiến dịch. Full/NCE/P3 cũ chỉ là bối cảnh; không train lại toàn bộ phương pháp cũ. Không thêm Local/Donor hoặc phương pháp nâng cấp vào queue này nếu chưa có câu hỏi mới cụ thể.

### Khóa recipe và metric trước pilot

- Dùng recipe Ê Đê cũ `rhade_contrastive_id_rtx.json`: RTX FP32, TF32off, SDPA chỉ sau khi đối chiếu backend cũ, mBART/pretrained revision như cũ, LR5e-5, B4×GA32, max length256, fusedAdamW, warmup.06, weight_decay0, clipping1, linear scheduler, max20epochs và early-stopping patience10 **đang hoạt động**.
- Selector giữ earliest-best original-reference validation chrF++/beam1, phù hợp baseline Ê Đê đang có. BLEU13a và chrF0 ghi đồng thời như secondary; test beam4. Không đổi selector sau khi xem test. Vietnamese có khoảng trắng, không tự suy vấn đề sparse word-bigram tiếng Trung xảy ra như nhau ở đây. Audit support word ngram thực tế trước chạy.
- CPU audit native SacreBLEU CHRF(word_order=2) ngày31/10: reference word-bigrams validation **14.549**, test **14.252**, so với chỉ3 trong Amis test. File `outputs/experiments/rhade_alignment_residual_preflight/WORD_NGRAM_SUPPORT.json`. Điều này giảm mối lo riêng về sparse word-bigram; không bảo đảm chrF++ phản ánh đầy đủ ngữ nghĩa hoặc pilot sẽ tăng điểm.
- C0 có CE thuần, không thêm LSR. Bỏ LSR ở cohort này để kiểm tra riêng correction, không kết luận LSR nói chung vô ích. Module init độc lập, module_seed=10000+training_seed, rank64, distance2, strength.1. Giữ target dummy/RNG policy như C0/M0 Amis.
- Không continuation từ weights Amis, Turkish hoặc Ê Đê đã fine-tune. Tất cả run mới bắt đầu từ cùng pretrained revision; đối chứng baseline cũ cũng phải có provenance tương ứng.
- Giữ riêng namespace Ê Đê/RTX, cùng cấu trúc `checkpoints/rhade_vietnamese/mbart-large-50/<method>/seed-N/<recipe>/`. Không đè prefix baseline cũ; không gộp scores RTX/A100.

### GO / NO-GO đề xuất, khóa trước khi launch

Đây là ngưỡng thực tiễn đề xuất để review, không phải preregistration của những test đã xem trước.

| Gate | GO | Nếu không đạt |
|---|---|---|
| Correctness | CPU formula/gradient/initialization/save-load/accumulation tests PASS; GPU đúng B4/GA32, FP32 SDPA, cached generation/reload/resume PASS | Sửa và đóng băng bundle mới trước training |
| Pilot seed42 validation | C0−baseline≥+.30 BLEU, C0−M0>0 BLEU, C0−LearnedDonor>0 BLEU; chrF++ và chrF0 của C0 không thấp hơn baseline | Không mở seeds43/44 chỉ để tìm kết quả thuận lợi; phân tích validation, báo kết quả âm |
| Replication validation | Mean C0−baseline≥+.30 BLEU, mean C0−LearnedDonor≥+.20; cả hai thắng≥2/3 seeds; chrF++/chrF0 mean không âm | Không nâng C0 thành main method; phân biệt kết quả không đủ bằng chứng với tương đương/thất bại mọi biến thể |
| Test sau khóa model | Test tất cả nhánh/seeds trong block đã duyệt; báo toàn bộ delta/CI/p và seed variability | Không chọn lại checkpoint, bỏ seed hoặc thêm run theo p-value |

Nếu pilot không GO, seed42 vẫn có thể được test **sau quyết định kết thúc pilot và khóa checkpoint**, nhằm mô tả kết quả âm; không dùng test đó để đảo quyết định mở replication. Test không phải cổng chọn phương pháp tiếp theo.

### Thống kê và diễn giải

Primary scientific contrast C0−LearnedDonor, BLEU13a; bắt buộc báo C0−baseline và C0−M0. 10.000 paired corpus-bootstrap draws RNG42. Family mới:3contrasts×3metrics=Holm9 cho pilot; nếu block3seeds được mở, báo lại cả3seeds trong **family Holm27 riêng của block đó**, giữ nguyên báo cáo pilot và ghi rõ hai giai đoạn, không dùng alpha lại như một confirmation độc lập. Các CI marginal không phải simultaneous. Mean/SD và exact paired-seed sign-flip riêng; n=3 không thể đạt two-sided p<.05. Không coi1000câu test là1000training seeds. Nếu có document grouping chính thức thì bổ sung cluster-aware sensitivity; chưa tự bịa document IDs.

### Novelty và quyết định nâng cấp

Thêm gating hoặc bottleneck không tự bảo đảm novelty. Các prior cần đối chiếu vẫn là [Houlsby adapters](https://proceedings.mlr.press/v97/houlsby19a.html), [SkipCLM](https://aclanthology.org/2025.naacl-srw.50/), [NeuTRENO](https://arxiv.org/abs/2312.00751) và [ReZero](https://arxiv.org/abs/2003.04887). Đây là targeted audit, không bảo đảm không có prior khác.

Ê Đê ở đây kiểm chứng **cùng C0 hiện có**, chưa cần bịa phương pháp mới khi chưa rõ nguyên nhân. Chỉ sau GPU probe Amis và kết quả validation Ê Đê mới chốt một nâng cấp có giả thuyết, công thức, matched control và kiểm tra novelty cụ thể. Nếu donor-off có hại nhưng module-off gần0 trên cả3seeds, không suy thẳng rằng tăng module/λ sẽ tốt; nếu khác biệt nhỏ hoặc không nhất quán thì kết luận chưa xác định cơ chế.

---

## Lịch sử chuẩn bị và chạy Ê Đê tháng10/2026

## Hiện hành: source proxy Indonesian `id_ID`, 15/10/2026

Sau khi tra cứu và được tác giả duyệt chạy, **cả bốn nhánh chuyển sang `id_ID→vi_VN`**.
Recipe hiện hành: `configs/experiments/rhade_contrastive_id_rtx.json`; workspace
`outputs/experiments/rhade_contrastive_id_rtx/`. Không thay text, split, hash dữ
liệu, loss, seed, LR, batch, precision, max epochs, ES10 hoặc selector/test recipe.
Chạy lại đầy đủ CPU/GPU gates trên bundle mới trước training.

Queue hiện hành đã launch RTX lúc khoảng22:09, PID373554; **CPU4/4 và GPU4/4 PASS**.
**22:17:51: Baseline đã train thật tới step5/2360**, GPU97%; thứ tự tiếp theo
Full cũ→NCE→P3. Cả4nhánh có610.879.488trainable parameters; đủ gate2steps,
finite training loss, optimizer/RNG checkpoint và save/reload logits match.
GPU gate receipts đã public SHA/size verified revision
`2e0d0f4fac077c30595190053610d653fe26fd4c`.
Bundle `bcaa8ef6c5b72bc3e4188ac1a630509fc77907f70b527c27795acf96d19b1446`.
Source token ID250032, target BOS250024 đã kiểm tra đúng toàn bộ dữ liệu.
Preflight8artifacts đã public SHA/size verified, revision
`87c0ff82962df078182b8e2254e9d56132ef41e6`, prefix
`analysis/rhade_vietnamese/mbart-large-50/contrastive-controls/seed-42/rtx6000-fp32-src-id/preflight/`.
Checkpoints mới có hậu tố cấu hình `...-src-id` bên trong seed; analysis/test
dùng prefix `rtx6000-fp32-src-id`. Không nhầm với preflight proxy cũ.

Cơ sở chọn: Ê Đê/Rhade thuộc Chamic; quan hệ lịch sử Chamic–Malayic được thảo
luận trong [Adelaar (2005), Malayo-Sumbawan](https://www.researchgate.net/publication/236789707_Malayo-Sumbawan).
[mBART model card](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages-covered)
có Indonesian `id_ID` và Vietnamese `vi_VN`, không có Rhade/Cham.
Đây là prior ngôn ngữ học, không khẳng định `id_ID` là proxy tốt nhất về scores.
Không tune proxy bằng test hoặc chạy thêm grid proxy trong pilot này.
Vocabulary/tokenizer dùng chung; đổi language token không loại được UNK trong
văn bản Ê Đê. Mỗi nhánh phải dùng cùng proxy; không yêu cầu proxy của Amis giống Ê Đê.

Queue `tl_XX` PID370375 đã kết thúc theo yêu cầu trước bất kỳ research training
nào: chỉ Baseline/Full cũ GPU gates PASS, NCE gate bị ngắt. Weights gate tạm đã
dọn; receipts và bundle cũ giữ để truy vết. Preflight cũ public revision
`8b1531142dd08debcc678f692e1be1609a040282` **không phải kết quả model**.
Không dùng lại gate, checkpoint hoặc metric của proxy cũ cho recipe mới.
Các thông tin PID/token `tl_XX` bên dưới mô tả giai đoạn preflight đã thay thế.

Ngày chuẩn bị: 15/10/2026. **Tác giả đã duyệt code và chạy cả bốn nhánh.**
Queue RTX đã launch lúc khoảng21:59, PID370375. CPU4/4 PASS cả local/RTX;
GPU gate Baseline PASS lúc22:02, các gate còn lại đang chạy. Chỉ mở training
dài sau khi tất cả4gates PASS; chưa có model scores Ê Đê tại mốc cập nhật này.
Bundle SHA256: `82210272ddfb4bc0c5ebe71c823e10e23830ddb4b819c41bcc5360fd5f3b121a`.

Audit fairness: cùng spec/optimizer/data/selection/generation và seed cho4nhánh;
code model/loss/optimizer không đổi. Chỉ CLI RTX dùng chung được nối thêm bước
khóa/test/bootstrap/upload sau training, tránh worker riêng từng thí nghiệm.
Receipts: `outputs/experiments/rhade_contrastive_rtx/{CPU_GATE.json,FAIRNESS_AUDIT.json,launch_rtx.json}`.
Audit tokenizer RTX: source `tl_XX` ID250047, target `vi_VN` BOS250024; đúng
language/EOS ở mọi dòng. Train4/15.092source vượt256tokens, target0; validation
và test0source/target vượt256. Không đổi max length hay lọc dòng.
Source có5464/523502UNKtoken train (~1.04%), target0; đây là coverage tokenizer,
không sửa văn bản/Unicode. Ghi nhận giới hạn coverage, không suy riêng nguyên nhân lỗi dịch.

Tác giả chọn Ê Đê thay Turkish để kiểm tra source ngoài danh mục ngôn ngữ
mBART, cùng loại bài toán thích nghi với Amis. Không khẳng định backbone chưa
từng tiếp xúc bất kỳ văn bản Ê Đê nào. Turkish đã dừng theo yêu cầu; early
stopping của Turkish không chứng minh dữ liệu đó bị lỗi. Số và checkpoints
Turkish đã có vẫn là lịch sử, không dùng thay control Ê Đê.

## Dữ liệu đã khóa

- Dataset: `data_processed/rhade_vietnamese/`, từ `NIRVLab/rhade-vietnamese-mt`.
- Revision: `dfbbffa09652e34054ef676589067f2ef965633b`.
- Train/validation/test: **15.092 / 1.000 / 1.000** dòng.
- Chỉ đổi cấu trúc `translation.{ede,vi}` thành CSV `source,target`.
- Không strip, normalize Unicode, sửa dấu, lọc câu, loại trùng, xáo trộn,
  lấy mẫu hoặc chia lại split. Đối chiếu từng chuỗi/thứ tự khi đọc CSV lại.
- Manifest lưu hash Parquet/CSV, số dòng và overlap. Các nguồn trùng được giữ:
  94/1.000 source test xuất hiện ở train; không trùng nguyên cặp dịch giữa split.
- Mẫu test đã được xem khi audit chất lượng dữ liệu; chưa có prediction/model
  scores Ê Đê. Đây là audit đầu vào, không phải blind human assessment.
- Corpus có nhiều từ/cụm ngắn và nhiều miền; không tự gán nhãn miền hoặc xem
  mọi dòng là câu dài. Không chọn dữ liệu theo khả năng tạo gain/significance.
- Nguồn: [dataset và điều khoản/citation](https://huggingface.co/datasets/NIRVLab/rhade-vietnamese-mt).

## Câu hỏi và bốn run đề xuất

Mỗi run bắt đầu độc lập từ cùng pretrained checkpoint, không từ weights Amis
hoặc Turkish. Seed/data seed đều 42. Đây là **pilot 4 models**, chưa là xác nhận
nhiều seed. Hai gates CPU/GPU không được tính thành research training runs.

| Thứ tự | Run ID | Method ID | Huấn luyện | Câu hỏi |
|---|---|---|---|---|
| 1 | `baseline-s42` | `baseline` | CE | Mốc thích nghi thông thường trên Ê Đê |
| 2 | `full-old-s42` | `original-full` | CLRR encoder d2, α=.10; CE+.10 LSR | CLRR+LSR cũ có lợi trên ngôn ngữ thứ hai không? |
| 3 | `nce-s42` | `nce-only-v1` | CE+.10 InfoNCE, τ=.10 | Alignment contrastive đứng riêng tạo được lợi ích nào? |
| 4 | `p3-s42` | `ordinary-full-v1` | CLRR d2, α=.10; CE+.10 LSR+.10 InfoNCE, τ=.10 | P3 có vượt baseline, NCE-only và Full cũ không? |

Không đưa CLRR-only/LSR-only/Middle/LoRA vào pilot này. Nếu Full cũ có tín hiệu
thì cần ablation CLRR-only/LSR-only riêng trước khi quy công cho từng thành phần.
P3−NCE đo phần đóng góp chung của CLRR+LSR, chưa tách riêng CLRR với LSR.

## Recipe chung và điểm phải duyệt

Recipe hiện hành: `configs/experiments/rhade_contrastive_id_rtx.json`.
Mã model/loss cũ và mới giữ đúng method IDs; không viết trainer riêng cho Ê Đê.

| Thành phần | Đề xuất |
|---|---|
| Backbone | mBART-large-50-many-to-many-mmt, revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30` |
| Language policy | Đã chốt: source proxy Indonesian `id_ID`, target `vi_VN`; không có token Rhade native |
| GPU | RTX6000 trong `/workspace/ura/thanhdat/projects/CLRR/`; cùng GPU/precision cho cả bốn nhánh |
| Precision | FP32; FP16/BF16/TF32 off |
| Batch | microbatch 4 × accumulation 32 = effective batch 128 |
| InfoNCE negatives | Trong microbatch 4, không phải effective batch 128 |
| Optimizer | fused AdamW; LR5e-5; WD0; betas .9/.999; epsilon1e-8; clip1 |
| Schedule | linear, warmup .06, cap20 epochs, patience10 |
| Token length | source/target256; đo mức truncation trước run, không tự sửa recipe hoặc lọc dữ liệu |
| Checkpointing | reentrant gradient checkpointing; pad multiple8; eval batch2 |
| Checkpoint selection | Earliest best raw-reference validation chrF++, greedy beam1 FP32 |
| Các metric phụ | BLEU13a và chrF0 mỗi epoch; cùng reference gốc |
| Test | Đủ1.000 dòng gốc, beam4 FP32, sau khi khóa toàn bộ checkpoint bằng validation |

Language proxy là quy tắc điều kiện mô hình, không đổi chuỗi CSV. Không chọn
proxy dựa trên test. Không ghép baseline A100/Amis/Turkish làm đối chứng cho
pilot này. Early stopping không đạt20epochs vẫn là kết quả hợp lệ; không tự
tăng patience hoặc chọn epoch sau khi nhìn test.

## Trình tự và GO/NO-GO

| Bước | Làm gì | Điều kiện đi tiếp / hành động nếu chưa đạt |
|---|---|---|
| 0 — đã xong | Export và đối chiếu100% chuỗi/thứ tự/split, hash | Đúng15.092/1.000/1.000; dữ liệu khác upstream thì dừng sửa exporter |
| 1 — đã duyệt | Chốt4run, language proxy và recipe trên | Đã freeze bundle và launch RTX |
| 2 — đang chạy | CPU loss/gradient/accumulation/save-load; GPU tiny cho đủ4nhánh; kiểm tra language/BOS/label và token lengths | CPU4/4 PASS; phải đủGPU4/4 PASS trước train; VRAM trước launch23.0GiB, giữ tiến trình người khác |
| 3 — pilot | Train4run tuần tự; log mỗi epoch; chọn đúng validation checkpoint | Mỗi model upload SHA/size verified rồi mới xóa weights RTX; lỗi kỹ thuật thì giữ bản chưa backup |
| 4 — đọc validation | So P3−Baseline, P3−NCE; Full cũ−Baseline; P3−Full cũ | P3 tăng BLEU và chrF++ trước cảBaseline/NCE là tín hiệu để đề xuất fresh seeds; metric trái dấu thì chưa GO. Không sweep theo test |
| 5 — tự động sau train | Đã duyệt trong plan: khóa cả4selection rồi test nguyên1.000dòng và bootstrap | Báo đủ4model và4contrast, không chỉ chọn contrast có lợi |
| 6 — quyết định sau pilot | Nếu validation có tín hiệu: đề xuất thêm Baseline/NCE/P3 seeds43/44 =9run | Tổng13run nếu được duyệt mở rộng; chưa tự chạy. Nếu không có tín hiệu, phân tích lỗi/recipe trước, không tuyên bố phương pháp đã tốt |

Ngưỡng chọn mô hình cho pilot là validation đã định nghĩa; test được dùng để
báo kết quả recipe đã khóa. Nếu dùng test để chọn/sửa phương pháp về sau, phải
ghi Ê Đê là development. Một seed và CI bootstrap theo câu không chứng minh
độ ổn định qua lần train. Đạt pilot cũng chưa xác nhận novelty hoặc khả năng
được nhận ACL.

## Thống kê dự kiến

- Bốn contrast đã định trước: P3−Baseline (primary), P3−NCE, Full cũ−Baseline,
  P3−Full cũ.
- Ba metrics: BLEU13a, native chrF++ và chrF0; 10.000 paired sentence bootstrap
  với chung resample indices, RNG42; delta,95%CI và rawp theo implementation
  SacreBLEU đã dùng. Một family **Holm12 =4contrast×3metrics**, không thu nhỏ
  family theo kết quả. Không tự thêm6cặp sau khi nhìn điểm.
- Primary score giữ nguyên1.000test rows. Có thể báo phụ theo source đã/chưa
  có ở train và độ dài; đó là strata phân tích, không tạo benchmark mới hoặc
  xóa94dòng overlap. Không gán ý nghĩa thống kê cho strata khi chưa có design.
- Test1.000dòng không bảo đảm đạtp<.05; độ lớn effect, phân bố câu, tương quan
  predictions và training seeds đều ảnh hưởng độ chắc chắn.

## Lệnh chuẩn bị đã dùng và lệnh kiểm tra cấu hình

```bash
conda run -n clrr python scripts/prepare_hf_parallel.py --repo NIRVLab/rhade-vietnamese-mt --revision dfbbffa09652e34054ef676589067f2ef965633b --source ede --target vi --dataset rhade_vietnamese
conda run -n clrr python scripts/prepare_hf_parallel.py --repo NIRVLab/rhade-vietnamese-mt --revision dfbbffa09652e34054ef676589067f2ef965633b --source ede --target vi --dataset rhade_vietnamese --verify-only
conda run -n clrr python scripts/experiment.py inspect --config configs/experiments/rhade_contrastive_id_rtx.json
```

Các lệnh trên không launch GPU. Các lệnh sau đã dùng sau khi tác giả duyệt:

```bash
conda run -n clrr python scripts/experiment.py build --config configs/experiments/rhade_contrastive_id_rtx.json
conda run -n clrr python scripts/experiment.py cpu --config configs/experiments/rhade_contrastive_id_rtx.json
conda run -n clrr python scripts/experiment.py launch --config configs/experiments/rhade_contrastive_id_rtx.json --account 1
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/rhade_contrastive_id_rtx.json --account 1
```

Không launch lại workspace đã có PID/receipts. Queue chạy nền độc lập với chat,
dùng CLI/evaluator/statistics chung và tự kết thúc sau báo cáo được backup.
Training failure dừng shell bằng `set -e`; không tự retry hoặc đổi recipe.

## Lưu trữ sau khi chạy

Repo public: `FiveC/amis-rewire-checkpoints`.
Namespace mới: `checkpoints/rhade_vietnamese/mbart-large-50/`
`{baseline,full,nce-only-v1,ordinary-full-v1}/seed-42/<configuration>/`.
Thư mục cấu hình chỉ để phân biệt recipe khi cần; metadata lưu đủ language
proxy, loss weights, data/model/code revision và selected/completed epoch.
Analysis: `analysis/rhade_vietnamese/mbart-large-50/contrastive-controls/`.
Không ghi đè hoặc xóa checkpoint/results cũ trên HF trong bước đổi dataset.
