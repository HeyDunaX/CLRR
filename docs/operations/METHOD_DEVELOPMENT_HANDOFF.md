# Bàn giao phát triển phương pháp CLRR

Cập nhật **21/11/2026**. Mục tiêu là phát triển một phương pháp dịch máy ít tài nguyên có câu hỏi nghiên cứu rõ, cơ chế kiểm chứng được và kết quả đủ ổn định để viết bài ACL. CLRR và LSR là nền tảng thực nghiệm. Có thể đổi hẳn cơ chế nếu giả thuyết mới có bằng chứng và đóng góp rõ. Ưu tiên thiết kế gọn, chạy được trên RTX 6000 24 GB trong giai đoạn phát triển, không phụ thuộc từ điển hoặc tập dữ liệu xây riêng cho method.

## 1. Mở project và đọc theo thứ tự

| Cần tìm | Vị trí |
|---|---|
| Repo trên máy tác giả | `D:\Code\CLRR` |
| Project riêng trên RTX | `/workspace/ura/thanhdat/projects/CLRR/` |
| Dữ liệu | `data_processed/<dataset>/`, ba CSV `train.csv`, `validation.csv`, `test.csv`, mỗi file có cột `source,target` |
| Kết quả RTX và chẩn đoán | [Sổ RTX6000](../experiments/EXPERIMENTS_RTX6000.md), [Danh mục tránh chạy trùng](../experiments/RESULTS_REGISTRY.md), [Index artifacts](../../outputs/results_registry/INDEX.md) |
| Sổ số tổng hợp, protocol đo, kế hoạch dài hạn | [Experiments Updated](../experiments/EXPERIMENTS_UPDATED.md), [Metrics](../experiments/METRICS.md), [Kế hoạch ACL](../research/ACL_LONG_TERM_EXPERIMENT_PLAN.md) |
| Insight, thiết kế và audit cũ | [Research Insights](../research/RESEARCH_INSIGHTS.md), [AC/DRM](../research/TWO_METHOD_DESIGN_ROUND1.md), [VCO/B](../research/TWO_METHOD_LOW_MEMORY_REVIEW.md), [Sổ thiết kế](../research/METHOD_REDESIGN_CANDIDATES.md) |
| Paper nền tảng | [clrr_main.tex](../paper/clrr_main.tex). Dùng để hiểu công thức và động cơ. Đối chiếu số hiện hành bằng sổ số và receipts. |
| Code, recipe và CLI | [Registry](../../src/methods/registry.py), [configs/experiments](../../configs/experiments), [CLI](../../scripts/experiment.py), [Workflow](EXPERIMENT_WORKFLOW.md) |
| Checkpoints, RTX, literature | [Kho HF](https://huggingface.co/FiveC/amis-rewire-checkpoints), [Quy ước HF](EXPERIMENT_WORKFLOW.md#output-local-và-hugging-face), [RTX runbook](RTX6000_TAILSCALE_RUNBOOK.md), [NotebookLM](NOTEBOOKLM_MCP_SETUP.md) |

Các đường dẫn trong bảng dùng từ checkout hiện tại. Nếu người nhận làm trên máy khác, cấp checkout có các file phát triển mới và chuyển kèm **các receipts/JSON/CSV nhỏ cần dùng** trong `outputs/`. Một số artifacts nằm ngoài Git. Bản ghi nguồn và hash quan trọng hơn ngày sửa file. Dùng handoff này làm điểm vào cho vòng phát triển method. Bàn giao tháng 10 và các banner trạng thái cũ giữ giá trị truy vết.

## 2. Trạng thái đã đối chiếu

**VCO_S42 hoàn tất.** Queue PID 139985 báo `QUEUE_COMPLETE`. Virtual Corrective-Order trên Amis→Mandarin, mBART-50, seed 42, kết thúc tại epoch 20, step 720 theo giới hạn epoch. Checkpoint tốt nhất theo validation BLEU là epoch 18, step 648. Biên nhận `VALIDATION_COMPLETE_VERIFIED` xác nhận điểm, checkpoint và SHA trên HF. Phạm vi run là validation-only. [Trạng thái queue](../../outputs/experiments/amis_virtual_corrective_order_pilot_rtx/queue_validation_account1.json).

| Amis, RTX, seed 42 | Validation BLEU | chrF (word order 0) | chrF++ | Test BLEU đã có |
|---|---:|---:|---:|---:|
| Baseline | 17.6823 | 16.9547 | 14.7030 | 19.3926 |
| Middle Alignment RTX | 16.6775 | 16.0333 | 13.9376 | 19.3753 |
| W, R-Drop β=1.25 | 18.2724 | 17.3888 | 15.1728 | 21.1631 |
| AC | 17.6130 | 16.9289 | 14.7789 | 21.2938 |
| DRM | 18.4724 | — | — | 21.0983 |
| VCO | 16.9681 | 16.2713 | 14.1664 | Chưa đánh giá |

Nguồn: [Sổ RTX, các mục 1, 3 và 6](../experiments/EXPERIMENTS_RTX6000.md), [AC42 validation](../../outputs/experiments/amis_two_method_round1_rtx/runs/AC_S42/VAL_COMPLETE.json), [VCO validation](../../outputs/experiments/amis_virtual_corrective_order_pilot_rtx/runs/VCO_S42/VAL_COMPLETE.json) và [stop state](../../outputs/experiments/amis_virtual_corrective_order_pilot_rtx/runs/VCO_S42/stop_state.json). Baseline RTX có lưu ý về provenance backend lịch sử và audit chọn cùng best step. Middle RTX có pool negatives trong microbatch 4. So sánh với Middle A100 phải tra cohort A100 riêng. Ô trống là số chưa trích vào handoff.

| Seed Amis test, RTX | Baseline | W | AC | AC − W |
|---|---:|---:|---:|---:|
| 42 | 19.3926 | 21.1631 | 21.2938 | +0.1307 |
| 43 | 19.5540 | 21.0765 | 20.9167 | −0.1598 |
| 44 | 20.2453 | 21.2454 | 20.0803 | −1.1652 |
| Mean BLEU | 19.7306 | 21.1617 | 20.7636 | −0.3981 |

Cùng 575 câu test, beam 4 FP32. AC42 hơn baseline42 có ý nghĩa ở BLEU và chrF trong family Holm 6 đã báo cáo. AC42 chưa có bằng chứng hơn W42 có ý nghĩa. Bảng ba seed là thống kê mô tả. Replay test AC43/44 đã hoàn tất nên không đọc banner cũ “chờ DRM” như lệnh cần chạy. DRM42 có tín hiệu validation cao hơn W42 khoảng 0.20 BLEU nhưng test thấp hơn khoảng 0.065 BLEU và mới có một seed. Giữ DRM như bằng chứng để đặt câu hỏi, chưa thăng cấp sang A100. [AC43/44 test receipt](../../outputs/experiments/amis_ac_replication_test_dispatch/latest_status.json), [phân tích AC42](../../outputs/experiments/amis_ac_seed42_test_statistics/statistics/PAIRED_BOOTSTRAP.md).

### VCO và ứng viên B

- VCO dùng hai lượt dropout. Từ support rows, cập nhật một ma trận ảo rank 32 trong tọa độ head theo hai thứ tự AB và BA. Loss là trung bình CE của hai views cộng symmetric KL giữa hai endpoints trên probe rows tách biệt. Ma trận được reset mỗi microbatch. Inference dùng backbone gốc. Code tại [VCO v1](../../src/methods/virtual_corrective_order/v1), recipe tại [VCO pilot](../../configs/experiments/amis_virtual_corrective_order_pilot_rtx.json).
- Code qua 7 kiểm tra CPU và GPU gate 2 optimizer steps, peak gate 14.35 GiB. Hệ số 136.2760627557 được hiệu chuẩn từ 3 batch train tại pretrained initialization. Nó dùng tỷ số gradient norm làm quy tắc hiệu chuẩn. [Hiệu chuẩn](../../outputs/experiments/amis_virtual_order_initialization_rtx/COMPLETE.json), [chẩn đoán frozen](../../outputs/experiments/amis_virtual_corrective_order_probe_rtx/REPORT.md).
- VCO thấp hơn baseline **0.7142 BLEU** và W **1.3043 BLEU** trên validation. **NO-GO mở thêm seed hoặc test cho recipe này.** Một seed đo được quality của recipe đã chạy. Muốn giải thích nguyên nhân, cần đọc trajectory và có can thiệp phân biệt ảnh hưởng của penalty với hai lượt CE, thời gian train, cơ chế đầu ra và hệ số loss.
- Exact corrective-order toàn backbone từng dừng tại GPU gate vì chi phí và OOM, chưa có train dài hoặc kết quả chất lượng. B conditional innovation rewiring bị loại trước pilot vì dạng affine gần họ LAuReL và thiếu bằng chứng phân biệt cơ chế học. Phân biệt B này với **B = R-Drop β=2.5** trong sổ RTX. [Audit thiết kế và prior](../research/TWO_METHOD_LOW_MEMORY_REVIEW.md).
- [Checkpoint VCO, revision 624275d…](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/624275d24846104ce1bc300f29519a99a3445728/checkpoints/amis_mandarin/mbart-large-50/virtual-corrective-order-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu). Archive SHA256 `700c232040a1990fba1e2c48fbd24c7fccdec0ad67d3dffaa7973ee3f9ac7af6`. Thư mục có validation predictions, states, calibration và logs từng microbatch. [Biên nhận backup](../../outputs/experiments/amis_virtual_corrective_order_pilot_rtx/runs/VCO_S42/collection_receipt.json).

### Dùng lại những gì đã chạy

| Nhóm | Điều người nhận cần nhớ |
|---|---|
| CLRR, LSR và Full cũ | Amis fresh confirmation 32 runs đã xong. Full cũ không qua các gate xác nhận. Giữ insight về tương tác và donor làm điểm xuất phát cho chẩn đoán. Tra kết quả theo cohort trước khi trích. |
| Rewiring, donor, C0, M0, LearnedDonor, Local, depth, learned strength | Đã có train và nhiều kiểm tra can thiệp. Một số phiên bản có đủ validation/test theo seed. Tắt donor lúc suy luận và train lại khi bỏ donor trả lời hai câu hỏi khác nhau. |
| W/B, gold-only/non-gold, PLC, CMC | W là comparator R-Drop đã có. Các cách tách gold/rest, prefix và continuation đã thử. PLC thấp hơn W dù frozen probe qua ngưỡng. CMC và các controls seed42 đã hoàn tất, dưới W. |
| Sequence risk, AC, DRM, VCO | Phân biệt continuation với train từ pretrained. AC test đã đủ seeds 42–44. DRM42 và VCO42 đã có kết quả như trên. |
| Ê Đê và Turkish | Ê Đê→Việt có baseline và các controls alignment residual. Recipe đó dùng source proxy `id_ID`, target `vi_VN`. Kiểm tra recipe/selector riêng trước khi so. Giữ nguyên các split gốc. Turkish là lịch sử đã dừng sử dụng cho hướng hiện hành. |
| A100 | Dùng sổ số chính để tra baseline, Middle Alignment, Strong LoRA, BitFit và các đối sánh khác đã chạy. Reuse từng checkpoint nếu metadata đáp ứng recipe bài báo. Pilot RTX và số A100 là hai cohort riêng. |

Danh mục có nhiều bản sao receipts cùng một model. Deduplicate bằng archive SHA và đối chiếu dataset, split, pretrained revision, method version, seed, precision, backend, batch/accumulation, selector và beam trước khi quyết định reuse. Trường thiếu ghi `UNKNOWN`. Đọc receipt mới nhất trước khi tin số đếm tổng của snapshot. [Danh mục và quy trình reuse](../experiments/RESULTS_REGISTRY.md).

## 3. Việc đầu tiên giao cho bên tiếp theo

**Chẩn đoán bằng artifacts sẵn có trước.** Đối chiếu validation predictions của baseline42, W42, AC42, DRM42 và VCO42 theo cùng row ID hoặc fingerprint đã xác minh. Xác nhận reference hash, số dòng, selector và beam. Ghi số câu mỗi model sửa được và làm hỏng so với W, kèm mẫu lỗi và số mẫu từng nhóm. Dùng validation cho vòng phát triển.

1. Lấy toàn bộ validation trajectory VCO và các logs loss/endpoint gap từ revision đã pin. Tổng hợp theo epoch gồm CE, order penalty, weighted penalty, endpoint gap, BLEU và thời gian. Tỷ số giữa các loss chỉ là mô tả. Để kết luận về lực cập nhật cần gradient hoặc optimizer update đo trực tiếp. Nếu log thiếu dữ kiện, ghi rõ cần phép đo nào.
2. Đối chiếu nơi DRM sửa đúng lỗi W mắc với nơi VCO hoặc AC làm hỏng bản dịch. Ghi lỗi như thiếu nội dung nguồn, nhầm thực thể/số, lặp và thay đổi bề mặt hợp lệ. Quy định cách gắn nhãn, lấy mẫu đều, ghi mẫu số và trường hợp không chắc. Đọc lại frozen reports để tránh lặp phép đo donor, grounding, prefix, cùng chọn sai hoặc chọn sai khác nhau.
3. Chốt một hiện tượng có thể tái kiểm tra và ≥2 cách giải thích cạnh tranh. Ví dụ, penalty giảm cùng BLEU thấp chỉ là phép quan sát. Để kiểm giả thuyết học cách né subspace cần đo projected feature energy/response dưới nhiều basis cố định và so với đối chứng cùng checkpoint. Nếu bằng chứng chưa tách được nguyên nhân, ưu tiên chẩn đoán không có optimizer step.
4. Bàn giao bảng `run | data / protocol | seed | split / metric | score | receipt`, bảng lỗi và quyết định giữ hay bỏ từng giả thuyết. Gắn **[ĐO ĐƯỢC]**, **[SUY LUẬN]** hoặc **[PHỎNG ĐOÁN]** kèm nguồn. Vòng tiếp theo bắt đầu từ bảng này. VCO và B ở trạng thái đóng pilot theo kết quả trên.

## 4. Quy trình chọn method tiếp theo

| Pha | Việc cần làm | Điều kiện để đi tiếp |
|---|---|---|
| 1. Bằng chứng | Kiểm kê trước, tìm quan sát lặp lại và phản ví dụ. Chọn một nguyên nhân có thể đo. | Có artifact và phép đo phân biệt được với ít nhất một giải thích khác. |
| 2. Hai hướng | **Hướng 1** tận dụng cơ chế hoặc kết quả tốt nhất đã đo và giải quyết một điểm yếu cụ thể. **Hướng 2** thử cơ chế học độc lập hơn, có RQ và đóng góp riêng. Viết công thức, nguồn tín hiệu, gradient, chi phí, inference và dự đoán có thể bác bỏ cho từng hướng. | Cả hai hướng đem ra trình cần đủ căn cứ để viết bài nếu kết quả tốt. Có thể giữ một hướng nếu chỉ một hướng qua audit. |
| 3. Novelty trước smoke | Tìm ≥6 truy vấn mỗi hướng với thuật ngữ và cơ chế tương đương trên ACL Anthology, arXiv, Semantic Scholar. Mở đọc abstract/method paper gần nhất, đối chiếu công thức và gradient. Thêm paper gốc đã đọc vào notebook CLRR, tra trùng trước. | Ghi TRÙNG / GẦN / CHƯA THẤY trong phạm vi đã đọc. GẦN cần khác biệt cơ chế và dự đoán kiểm chứng được. Với TRÙNG hoặc HOLD, sửa thiết kế hoặc loại trước khi mở pilot. |
| 4. Phản biện | NotebookLM vòng A đóng vai reviewer tìm lý do reject. Vòng B phản biện bằng nguồn gốc và bằng chứng project được phép chia sẻ. Lưu prompt, đáp án và source IDs. Đọc paper để kiểm lại kết luận. | Có claim rõ về đóng góp, prior gần nhất và thí nghiệm có thể bác claim. Tác giả đã cho phép gửi công thức lý thuyết CLRR vào notebook đó. Dữ liệu, bản dịch và số nội bộ cần quyền phù hợp với phần gửi. |
| 5. Khóa pilot và ablation | Chốt Amis seed42, validation-only. Lập bảng candidate, đối chứng mạnh W, baseline, Middle nếu cùng cohort và ablation tối thiểu. Với regularizer hai views, cân nhắc CE hai views khi cần tách hiệu ứng từ thêm lượt forward. Dùng ablation đổi đúng yếu tố của RQ. Khóa số run, phương án reuse, ngân sách và ngưỡng GO/NO-GO trước khi thấy kết quả mới. | Kế hoạch có đủ đối chứng và mỗi run trả lời một câu hỏi. Không tự dùng lại ngưỡng từ method khác. |
| 6. Code và gates | Mỗi method ở `src/methods/<mechanism>/v1/`, thêm ID registry và JSON config. Đổi công thức thì tạo version mới. Test công thức, gradient, masks, accumulation, checkpointing, save/reload, generation và resume. Chạy tiny GPU gate bằng recipe thật, ghi peak VRAM và wall time. | Tất cả gates đã khóa PASS và phương án chạy được duyệt. Khi OOM hoặc số không hữu hạn, ghi lỗi và rà chi phí trước khi đổi recipe. |
| 7. Quyết định | Đối chiếu gain với baseline và đối chứng mạnh nhất theo RQ. Kiểm tra ablation, prediction lỗi và compute. | GO khi đạt ngưỡng đã khóa và có bằng chứng hỗ trợ cơ chế. NO-GO khi hụt ngưỡng hoặc lợi ích biến mất trước matched control. Chẩn đoán rồi mới quyết định thiết kế khác. |

Mỗi candidate card ghi đủ **RQ → hiện tượng có nguồn → nguyên nhân thử kiểm → công thức → prior gần → dự đoán và falsifier → bảng runs → gate chất lượng và tài nguyên**. Giữ các kết quả âm. [NotebookLM setup và prompt](NOTEBOOKLM_MCP_SETUP.md).

## 5. Recipe RTX và cách vận hành khi kế hoạch được duyệt

Tham chiếu khóa Amis BLEU tại [amis_drb_bleu.json](../../configs/protocols/amis_drb_bleu.json). Amis có 4.600/576/575 câu train/validation/test, `tl_XX → zh_CN`, mBART-50 revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`. Recipe RTX dùng FP32, SDPA, TF32 off, checkpointing reentrant, LR 5e-5, batch 4 × accumulation 32, max length 256, AdamW fused, warmup .06, tối đa 20 epochs, ES10 theo **validation BLEU**, beam 1 và references CSV gốc. `chrF` là `chrf0` trong JSON. Report lưu cả `chrf++`. Khi đem sang A100, khóa settings của cohort A100 riêng và audit comparator theo bài.

Mọi Python chạy trong conda `clrr`. Các lệnh dưới dùng cho **JSON mới đã review** trong checkout Windows. Chỉ phần monitor ở block đầu áp dụng cho VCO đã hoàn thành.

```powershell
Set-Location D:\Code\CLRR
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_virtual_corrective_order_pilot_rtx.json --account 1
```

```powershell
$recipe = 'configs/experiments/<new_candidate_rtx>.json'
conda run -n clrr python scripts/experiment.py inspect --config $recipe
conda run -n clrr python scripts/experiment.py build --config $recipe
conda run -n clrr python scripts/experiment.py cpu --config $recipe
# Với recipe đã được duyệt chạy, queue chạy GPU gates trước, rồi mới train.
conda run -n clrr python scripts/experiment.py launch --config $recipe --account 1 --stage gate-train
conda run -n clrr python scripts/experiment.py monitor --config $recipe --account 1
```

- RTX `--account 1` là queue RTX theo `hardware_profile=rtx6000-fp32`. Kiểm tra GPU, VRAM, disk, queue và project lock trước khi launch. Với VCO, các ngưỡng đã khóa là 20.000 MiB VRAM trống và 40 GiB disk. Giữ conda, HF cache, tmp và outputs trong project RTX. Chờ khi GPU bận, giữ nguyên tiến trình của người khác. Queue detached tiếp tục khi máy local tắt.
- Build khóa recipe, source và data hashes. Nếu candidate thay công thức, build bundle mới và chạy lại gates tương ứng. Khi trục trặc, tra `queue.log`, `progress.json`, `training_failure.json` và receipts trước khi resume. Resume chọn đúng run ID, dùng đúng checkpoint, bundle và recipe đã pin.
- Lưu `checkpoints/<dataset>/<backbone>/<method>/seed-<n>/`. Chỉ thêm thư mục cấu hình bên trong seed khi có nhiều biến thể. Dùng tên method/recipe để phân biệt. Ghi repo/revision, SHA, protocol, data hash, selected step, metrics và lý do dừng. Kiểm tra backup SHA trước khi dọn weights RTX. Chỉ thu JSON/CSV/Markdown nhỏ về local khi phân tích.
- Bàn giao quyền repo, notebook và máy tính qua tài khoản do người nhận được cấp. Credentials ở máy hiện tại dùng riêng theo runbook. Khi làm ở máy khác, kiểm tra checkout đang có đủ source, recipe và receipts vì một số artifacts được lưu ngoài Git.

## 6. Lộ trình từ pilot đến bảng ACL

1. **RTX validation**. Khóa method và ablation có tác dụng trước khi nhân seed. Báo số từng model, giá trị trung bình khi đủ seed, chi phí và lỗi. Tách quyết định GO khỏi việc ngưỡng thống kê trên test.
2. **Xác nhận**. Chọn trước checkpoint bằng validation BLEU, giữ các seed ghép cặp (≥3, chốt số theo biến thiên và hiệu ứng cần phát hiện), và chọn test độc lập theo khả năng dữ liệu. Test Amis đã xem trong phát triển, nên báo là exploratory. Ê Đê có thể dùng kiểm tra chuyển ngôn ngữ theo recipe/split gốc, cần ghi mức độ đã sử dụng trong development. Muốn claim rộng hơn, thêm một setting chưa dùng để chọn method và giữ nguyên quy tắc chọn trước khi mở test.
3. **A100**. Train method đã khóa và các đối chứng thiếu metadata hoặc protocol khớp. Reuse baseline, Middle, Strong LoRA, BitFit và đối sánh khác đã xác minh phù hợp. Nếu bài claim qua hai backbone hoặc nhiều ngôn ngữ, phải có chạy thật ở từng setting. Ghi model revision, runtime, effective batch, bước train, peak VRAM và thời gian. Đối chứng phù hợp với cơ chế mới có thể gồm W/R-Drop, CE cùng số views hoặc prior gần nhất.
4. **Test và thống kê**. Dùng checkpoint đã khóa, beam 4 theo protocol, cùng reference và corpus BLEU, chrF, chrF++. Báo paired sentence bootstrap 10.000 mẫu, CI95% và p theo convention đã khóa. Chốt rõ family Holm và số phép so sánh. Báo riêng paired inference giữa seeds, effect và biến thiên. 10.000 là số lần resample từ tập câu, còn số câu và seed độc lập giữ nguyên.
5. **Viết bài**. Bảng chính gồm method đã khóa và đối chứng phù hợp trên cùng cohort, ưu tiên baseline, Middle Alignment, Strong LoRA, BitFit và prior gần. Báo ablations, nhóm lỗi, analysis cơ chế, compute trong bảng/phụ lục phù hợp. Mọi số có receipt và protocol. Quyết định claim sau khi có kết quả lặp lại và phân tích cơ chế.

### Câu giao việc có thể dán vào chat mới

> Mở `D:\Code\CLRR\docs\operations\METHOD_DEVELOPMENT_HANDOFF.md` và làm mục 3 trước. Kiểm kê để tái sử dụng kết quả, đọc logs và phân loại lỗi validation để tìm hiện tượng đo được. Từ đó thiết kế hai hướng hoặc giữ một hướng nếu chỉ một hướng qua audit. Đọc paper gốc và debate với NotebookLM trước khi trình card, kèm RQ, công thức, novelty, falsifier và danh sách pilot/ablations. Dùng RTX cho pilot sau khi chốt kế hoạch và gates PASS. Dùng cohort A100 cho số xác nhận của bài.
