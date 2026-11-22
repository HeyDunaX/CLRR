# Các hướng phát triển method trên RTX6000 — tránh làm trùng

Ngày kiểm kê: **16/11/2026**. Đây là cửa vào để tìm bằng chứng, tránh chạy trùng;
không thay thế sổ số `EXPERIMENTS_UPDATED.md`, không sửa điểm hay chọn lại checkpoint.

## 1. Mở file nào?

| Nhu cầu | File |
|---|---|
| Từng receipt: run, method, seed, điểm, selector, hardware và nguồn | [INDEX.md](../../outputs/results_registry/INDEX.md) |
| Lọc toàn bộ completion receipts bằng spreadsheet/Python | [completion_receipts.csv](../../outputs/results_registry/completion_receipts.csv) |
| Từng checkpoint public: SHA, metadata, điểm gốc và receipt local khớp hash | [public_checkpoints.csv](../../outputs/results_registry/public_checkpoints.csv) |
| Các run trong PLAN và bằng chứng hoàn tất tìm được | [planned_runs.csv](../../outputs/results_registry/planned_runs.csv) |
| Mọi bảng Markdown nguồn, giữ nguyên số và vị trí | [source_tables.md](../../outputs/results_registry/source_tables.md) |
| Các block metrics JSON, có đường dẫn và JSON pointer | [metric_blocks.csv](../../outputs/results_registry/metric_blocks.csv) |
| Danh mục JSON/CSV/Markdown, SHA và số dòng CSV | [artifacts.csv](../../outputs/results_registry/artifacts.csv) |
| Phạm vi, số lượng, lỗi đọc và hash nguồn | [manifest.json](../../outputs/results_registry/manifest.json) |
| Danh mục file HF tại revision đã khóa | [HF_INDEX.json](../../outputs/results_registry/HF_INDEX.json) |

**Một receipt không phải một model.** Cùng checkpoint có thể có receipt ở folder
train, test, provenance hoặc recovery. Gom bản sao bằng archive SHA khi có;
hai archive khác SHA cũng có thể chứa cùng weights nhưng khác packaging/selector.
Không báo số models độc lập bằng số file hoặc số receipts.

## 2. Các nhóm phát triển RTX đã thử

Phạm vi bảng này chỉ gồm phát triển RTX. Các CSV hỗ trợ phía trên còn chứa lịch sử ngoài RTX. Các nhóm không mặc định dùng cùng protocol.
Điểm cụ thể từng run nằm trong các CSV/index phía trên và báo cáo nguồn.

| Nhóm | Phạm vi / nhánh cần nhớ | Bằng chứng chính | Quy tắc tránh chạy trùng |
|---|---|---|---|
| RTX original/token | R1 baseline, P1 CLRR, Full, token-LSR; validation và test | [Token diagnosis](../research/TOKEN_DECODING_DIAGNOSIS.md), [RTX](EXPERIMENTS_RTX6000.md) | Gold-prefix accuracy khác generation; không trộn số A100. |
| RTX residual-effect | Baseline/CLRR/Full và frozen interventions | [Validation](../research/RESIDUAL_EFFECT_VALIDATION_REPORT.md), [test](../research/RESIDUAL_EFFECT_TEST_REPORT.md) | Đã đo; donor-off không phải retraining ablation. |
| Depth / topology | Depth screens, replication, P1 SDPA, depth-difference v2 | [RTX](EXPERIMENTS_RTX6000.md), [run inventory](../../outputs/results_registry/INDEX.md) | Eager và SDPA phải phân biệt; không chạy lại P1 chỉ vì quên cohort. |
| Contrastive controls | NCE-only, CLRR-NCE, attached Full, multi-seed, test | [RTX](EXPERIMENTS_RTX6000.md), [inventory](../../outputs/results_registry/INDEX.md) | Có cả Amis và Ê Đê/proxy khác nhau; tra dataset và source language. |
| Learned strength | Learned-strength CLRR/Full | [Diagnosis](../research/LEARNED_STRENGTH_DIAGNOSIS.md) | Đã có kết quả; scalar gate không phải candidate mới. |
| Sequence risk | CE, risk, CLRR-CE, CLRR-risk continuation; pool probes | [Review](../research/CONTINUATION_REVIEW_AND_NEXT_DESIGN.md) | Đã thử; phân biệt continuation với train từ pretrained. |
| Alignment residual C0–C3 | Capacity, full-gradient, confined, local-confined | [Audit](../research/ALIGNMENT_RESIDUAL_PILOT_AUDIT.md) | Đã train; không đề xuất lại cùng module dưới tên khác. |
| Donor matched controls | C0, Local, Donor, LearnedDonor, M0; seeds42–44 và test | [Test analysis](../../outputs/experiments/amis_alignment_residual_replication_rtx_test/ANALYSIS.md) | Đã kiểm chứng; đây chính là donor controls đã chạy. |
| BLEU-selector replay/reruns | C0 và các controls trong plan BLEU | [Artifacts](../../outputs/experiments/amis_alignment_residual_bleu_selection_rtx/), [test](../../outputs/experiments/amis_alignment_residual_bleu_selection_rtx_test/) | Không nhập chung với cohort chọn theo chrF++. |
| Frozen donor/rank/residual | C0 module-off/donor-off/shuffle ba seed, common-hard errors | [GPU report](../../outputs/experiments/amis_residual_seed_diagnostics/gpu/GPU_REPORT.md), [consensus](../../outputs/experiments/amis_validation_seed_consensus/) | Đã đo; không lặp probe donor chung chung. |
| Ê Đê alignment residual | Baseline, C0, LearnedDonor, M0; validation và test | [RTX](EXPERIMENTS_RTX6000.md), [artifacts](../../outputs/experiments/rhade_alignment_residual_rtx_test/) | Có số rồi; giữ proxy/runtime/selector của cohort. |
| Decoder Residual Bridge | Amis D0/D1/D2/D3; Ê Đê có gate/resume trạng thái riêng | [Analysis](../../outputs/experiments/amis_decoder_residual_bridge_rtx_analysis/validation/VALIDATION_ANALYSIS.md) | Amis đã chạy; không suy ra Ê Đê đã train chỉ từ gate. |
| Transition/difference-preserving | Transition-LSR/Full; relational prototypes/technical gates | [Transition](../research/TRANSITION_FOLLOWUP_DIAGNOSIS.md), [relational](../research/DIFFERENCE_PRESERVING_PILOT.md) | Tách run hoàn tất khỏi queue bị dừng vì novelty; technical PASS không phải kết quả training. |
| Paired route | Native CE, native R-Drop, CLRR CE, CLRR R-Drop | [Design/results](../research/CLRR_RECOVERY_DESIGN.md), [artifacts](../../outputs/experiments/amis_paired_route_rtx/) | Đã có đối chứng hai views; không gọi train lại để kiểm tra donor lần đầu. |
| Decoupled consistency | G gold-only, R non-gold, W full weak; B ordinary | [RTX](EXPERIMENTS_RTX6000.md), [artifacts](../../outputs/experiments/amis_decoupled_consistency_rtx/) | Gold/rest split đã thử và không được quảng bá là mới. |
| W/B replication | Beta1.25 và2.5; seeds42–44 | [SUMMARY](../../outputs/experiments/amis_consistency_strength_w_replication_rtx_suite/SUMMARY.json) | W là R-Drop prior; đã có ba seed validation, không mặc định đã test. |
| PLC và controls | Prefix, diagonal, endpoint, calibrated W, Middle | [Failure analysis](../research/PLC_FAILURE_ANALYSIS.md) | Đã train20epochs; đừng ghi thành dừng vì ES10. |
| CMC và controls | CMC, NORM, PERMUTED seed42 | [Completion folder](../../outputs/experiments/amis_continuation_consistency_rtx/runs/) | Ba completion receipts được đọc bổ sung từ RTX ngày16/11; banner local cũ “đang chạy” không còn đủ để kết luận. |
| Source attribution | Source-response, SAC frozen, source grounding | [SAC](../research/SOURCE_ATTRIBUTION_METHOD.md), [frozen](../../outputs/experiments/amis_source_attribution_frozen_rtx/) | Chẩn đoán không optimizer; không tính như model train mới. |
| Decision branches | Frozen baseline/W, same-prefix interventions và rollouts | [Artifacts](../../outputs/experiments/amis_decision_branches_frozen_rtx/) | Đã đo; không đủ chứng minh KL quyết định branch cost. |
| Design-only / CPU-only | Margin envelope, conditional dropout, increment reuse và các novelty HOLD | [Design ledger](../research/METHOD_REDESIGN_CANDIDATES.md), [insight](../research/RESEARCH_INSIGHTS.md) | Không coi có code/CPU PASS là đã train. Đọc quyết định mới nhất trước khi mở pilot. |

## Trạng thái quan trọng để tránh phát triển nhầm

| Hướng | Đã làm / trạng thái | Cần nhớ |
|---|---|---|
| C0 / donor / LearnedDonor / Local / M0 | Có replication Amis và test; cả cohort chọn chrF++ và chọn BLEU. | Chưa có lợi ích test ổn định; không đề xuất donor hoặc đổi selector như hướng chưa thử. |
| Ê Đê alignment residual | Baseline/C0/LearnedDonor/M0 đã validation/test. | Chưa xác nhận C0 hơn baseline. |
| DRB | Amis D0–D3 hoàn tất; Ê Đê chỉ gate/resume hold. | Không gán kết quả Amis cho Ê Đê. |
| Sequence risk | Bốn continuation arms đã chạy sau pool gate. | Không vượt checkpoint khởi tạo; phân biệt continuation với train từ pretrained. |
| Paired-route / decoupled consistency | CE/KL với và không CLRR; gold-only, non-gold, full-distribution. | Các cách chia gold/rest đã thử. |
| W/B | Có validation seeds42–44. | W là anchor tích cực thuộc prior R-Drop; không nhận là method mới, không mặc định đã test. |
| PLC | Prefix/diagonal/endpoint/calibrated W/Middle đã train. | Prefix chưa vượt gate; tránh đề xuất lại bằng tên khác. |
| CMC/NORM/PERMUTED | Đã hoàn tất, validation BLEU 15.5402/15.9929/15.8222, dưới W 18.2724. | Banner cũ ghi đang chạy đã lỗi thời; đây là Amis seed42 validation. |
| A100 GOC pilot 9 runs (22/11/2026) | `BASELINE_S42` (`17.5264`), `W_B125`/`B250` (`17.7069`/`17.3802`), `GOC_B125`/`B250` (`18.0944`/`17.8097`), `GOC_SYM_B125`/`B250` (`18.4022`/`18.5143`), `GOC_INV_B125`/`B250` (`16.2213`/`15.1466`) — A100 BF16 B32×GA4 `eval_bleu` seed42, validation-only. | **NO-GO cho GOC** (thua `GOC_SYM` cùng $\beta$ ở cả 2 mức và tụt rare CJK recall). `GOC_SYM` (`symmetric-stopgrad-consistency-v1`, khớp dạng EDWARD-NMT Eq. 13 ở chiều rộng bằng nhau) là **anchor/comparator A100 mạnh nhất hiện có (`18.5143` Val BLEU)**; không chạy lại 9 runs seed42 này và không nhận riêng `GOC_SYM` là novelty độc lập. Receipt: [GOC_PILOT_A100_REPORT.md](../tasks/reports/GOC_PILOT_A100_REPORT.md). |
| A100 CR-Dual pilot 4 runs (22/11/2026) | `BIT_CE_S42` (`16.4574`), `CR_DUAL_S42` (`16.1977`), `CR_SHUFFLED_S42` (`16.9163`), `UNI_CONTRAST_S42` (`16.7299`) + Step-0 diagnostic trên `BIT_CE_S42` — cùng cohort A100 `eval_bleu` seed42, validation-only. | **NO-GO toàn diện (0/5 cổng).** Huấn luyện hai chiều `BiT` ăn mòn decoder `mBART-50` (`-1.07` BLEU vs `BASELINE`), `CR_DUAL` thua cả `BIT_CE` (`-0.26`) lẫn falsifier `CR_SHUFFLED` (`-0.72`), và `UNI_CONTRAST` cũng tụt `-0.80` dưới `BASELINE`. Đóng nhánh bidirectional/off-diagonal sequence contrast; không lặp lại. Receipt: [CR_DUAL_PILOT_A100_REPORT.md](../tasks/reports/CR_DUAL_PILOT_A100_REPORT.md). |
| Turkish RTX | Baseline hoàn tất; các nhánh còn lại dừng/chưa bắt đầu. | Dataset đã bỏ; không nhầm với Turkish A100. |

Nguồn CMC: `outputs/experiments/amis_continuation_consistency_rtx/runs/{CMC_S42,NORM_S42,PERMUTED_S42}/VAL_COMPLETE.json` và `validation_metrics.json`. Nguồn W: `outputs/experiments/amis_decoupled_consistency_rtx/`. Các nhóm khác có báo cáo nguồn trong bảng mục2 và [sổ RTX](EXPERIMENTS_RTX6000.md).

### Chưa có training hoàn tất — không nhầm với kết quả âm

| Hướng | Bằng chứng | Trạng thái |
|---|---|---|
| Difference-preserving / relational | [Pilot](../research/DIFFERENCE_PRESERVING_PILOT.md) | Dừng vì novelty; A mới bắt đầu, checkpoint36; B/C/D chưa train dài. |
| SAC / source attribution | `outputs/experiments/amis_source_attribution_frozen_rtx/` | Chỉ frozen diagnostics, chưa train SAC. |
| Decision branches | `outputs/experiments/amis_decision_branches_frozen_rtx/` | Chỉ chẩn đoán checkpoint cố định. |
| Margin envelope | [Candidate](../research/MARGIN_ENVELOPE_CANDIDATE.md) | HOLD, chưa train. |
| Conditional dropout | [Candidate](../research/CONDITIONAL_DROPOUT_CANDIDATE.md) | HOLD, chưa train. |
| Increment reuse RO/DW/RW | `outputs/experiments/increment_reuse_preflight/` | CPU tests; overlap Hyper-Connections, chưa train GPU. |
| Các hướng khác đã audit | `outputs/experiments/method_redesign_audit/REVIEW.md` | Tra verdict trước khi đề xuất lại. |

Cơ chế đã thử vẫn có thể nghiên cứu tiếp nếu chỉ rõ khác biệt, nguyên nhân đo được và dự đoán mới. Đổi tên hoặc thêm hệ số không tự tạo ra hướng mới.

## 3. Tra cứu trước mọi lệnh train

1. Tìm **dataset + backbone + method + seed** trong completion CSV, public checkpoint
   CSV và planned-runs CSV. Run ID khác tên chưa chứng minh là một thí nghiệm khác.
2. Đối chiếu data hashes/split, pretrained revision, code/method version, LR,
   batch/accumulation, precision, attention backend, ES, epoch budget, selector,
   metric/tokenizer và beam. Cấu hình thiếu trường nào thì ghi UNKNOWN, không đoán.
3. Tách ba việc: **train hoàn tất**, **đã đánh giá validation/test**, **backup đã xác minh**.
   Một việc hoàn tất không tự chứng minh hai việc còn lại.
4. Nếu archive SHA đã tồn tại: mở receipt/metadata hiện có. Không train để tạo lại
   checkpoint chỉ vì folder local thiếu weights hoặc tên HF đã đổi sau migration.
5. Nếu chưa thấy receipt: kiểm tra HF và báo cáo nguồn. `UNRESOLVED_NOT_PROOF_UNRUN`
   nghĩa là chưa giải quyết được bằng chứng local, **không phải lệnh chạy mới**.
6. Các bản ghi historical/bug-affected, validation đã dùng development, test đã xem,
   A100 và RTX giữ riêng. Không tự cộng mean, sửa family p-value hoặc chọn số đẹp hơn.

## 4. Giới hạn và quy tắc đọc

- Snapshot là kiểm kê bằng chứng, không phải replay lại metrics hay xác nhận mọi
  run scientifically comparable. Score được chép nguyên từ nguồn, không tính lại.
- `metric_blocks.csv` có thể chứa mean, controls, diagnostic/synthetic values và
  bản sao từ báo cáo. JSON pointer/context là bắt buộc; không cộng chúng thành run count.
- `source_tables.md` cố ý giữ cả bảng kế hoạch, historical và số bị thay thế,
  kèm file/section/dòng. Không dùng mọi bảng trích làm bảng kết quả hiện hành.
- HF được liệt kê tại revision trong manifest; metadata tải chỉ JSON nhỏ,
  không tải weights. File JSON thiếu metadata không tự chứng minh run thiếu.
- File HF trong `analysis/*/bundle.zip` là source bundle, không phải checkpoint.
- Không thay đổi nội dung các sổ số cũ, paper, dữ liệu hay checkpoint trong đợt gom này.

## 5. Cập nhật sau này

Chạy trong repo bằng conda `clrr`:

```powershell
conda run -n clrr python -X utf8 scripts/analysis/build_results_registry.py
```

Lệnh đọc lại sources local và metadata HF đã lưu; không tự SSH, train, đánh giá test,
upload hoặc tải checkpoint. `--fetch-hf-metadata` chỉ bổ sung JSON nhỏ từ **revision
HF_INDEX đã khóa**, không tự đổi sang main mới. Snapshot HF mới cần khóa revision
mới rồi mới cập nhật; không sửa nguồn lịch sử để làm khớp danh mục.
