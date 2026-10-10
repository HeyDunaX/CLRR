# CLRR — tài liệu hiện hành

**Tài liệu hiện hành.** Dùng các tài liệu chính dưới đây. Archive và source paper giữ nguyên trong đợt gom code.

| Cần xem | File chính |
|---|---|
| Mục lục | `docs/README.md` |
| Toàn bộ bảng số, cohort và provenance | [EXPERIMENTS_UPDATED.md](experiments/EXPERIMENTS_UPDATED.md) |
| Protocol metrics, selection và thống kê | [METRICS.md](experiments/METRICS.md) |
| Diễn giải kết quả, novelty và hypothesis | [RESEARCH_INSIGHTS.md](research/RESEARCH_INSIGHTS.md) |
| Chiến lược, budget và GO/NO-GO | [ACL_LONG_TERM_EXPERIMENT_PLAN.md](research/ACL_LONG_TERM_EXPERIMENT_PLAN.md) |
| Trạng thái thực thi, scripts và lệnh | [NEXT_EXPERIMENTS_HANDOFF_20261005.md](operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md) |
| CLI args, lệnh bash và workflow train dùng chung | [EXPERIMENT_WORKFLOW.md](operations/EXPERIMENT_WORKFLOW.md) |
| Điều khiển ba Colab | [COLAB_SSH_GUIDE.md](operations/COLAB_SSH_GUIDE.md) |
| Tìm/lưu/push checkpoint public | [HUGGINGFACE_CHECKPOINT_STORAGE.md](operations/HUGGINGFACE_CHECKPOINT_STORAGE.md) |
| Máy RTX6000 qua Tailscale | [RTX6000_TAILSCALE_RUNBOOK.md](operations/RTX6000_TAILSCALE_RUNBOOK.md) |

## Trạng thái nghiên cứu

Amis fresh confirmation32runs đã xong và không đạt gates của Full cũ. Kiểm chứng cơ chế đã xong. Turkish OPUS100 gốc seed42, bốn nhánh Baseline/Middle/LSR/Full đã train/test xong. GPU đã tắt sau backup. Bước tiếp theo là candidate card có controls; chưa launch nâng cấp. Đọc trạng thái ở đầu Insight/Long-term/Handoff thay vì lệnh cũ trong phụ lục lịch sử.

## Nội dung đã gộp và cách đọc tham chiếu cũ

| File riêng đã bỏ | Nội dung nằm tại |
|---|---|
| `TURKISH_MBART_SEED42_RESULTS.md` | [EXPERIMENTS_UPDATED.md](experiments/EXPERIMENTS_UPDATED.md#turkish-seed42) |
| `CONFIRMATION_DESIGN_AND_GATES.md` | [ACL_LONG_TERM_EXPERIMENT_PLAN.md](research/ACL_LONG_TERM_EXPERIMENT_PLAN.md#locked-confirmation-protocol) |
| `NOVELTY_AND_COMPARATOR_CARD.md` | [RESEARCH_INSIGHTS.md](research/RESEARCH_INSIGHTS.md#novelty-card-history) |
| `PEFT_PAPERS_REFERENCE.md` | [RESEARCH_INSIGHTS.md](research/RESEARCH_INSIGHTS.md#peft-reference-history) |
| `CONFIGURATION_SEARCH_HANDOFF.md` | [NEXT_EXPERIMENTS_HANDOFF_20261005.md](operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#configuration-handoff-history) |
| `AMIS_WARMUP_2_COLAB_RUNBOOK.md` | [NEXT_EXPERIMENTS_HANDOFF_20261005.md](operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#warmup-runbook-history) |
| `BAN_GIAO_CHAY_2_COLAB.md` | [NEXT_EXPERIMENTS_HANDOFF_20261005.md](operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#warmup-handoff-history) |
| `implementation_plan.md` | [NEXT_EXPERIMENTS_HANDOFF_20261005.md](operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md#implementation-history) |

## Paper và lịch sử

- [Source paper](paper/clrr_main.tex), [bibliography](paper/clrr_references.bib), style ACL và assets paper được giữ nguyên. Source paper không tự được viết lại hay compile trong bước gộp docs.
- [Archive](archive/README.md) chỉ dùng để truy vết. Không sửa/xóa bất kỳ file nào trong `docs/archive/` hoặc `docs/research/archive/`.
- Trên filesystem vẫn có `experiments/EXPERIMENTS_PAPER_MASTER_20261006.md` ngoài archive. Giữ nguyên snapshot này để tránh tác động nhầm bản của tác giả; **không dùng làm sổ số hiện hành** và không nhập bảng snapshot cũ để thay số trong `EXPERIMENTS_UPDATED.md`.
- Tham chiếu cũ trong archive/report immutable có thể còn dùng tên file đã gộp: tra bảng phía trên và manifest, không chỉnh lại lịch sử.
- Manifest gộp và kiểm tra bảo toàn nội dung: `outputs_rebuttal/docs_consolidation_20261008/manifest.json`.

Số liệu gốc trong CSV/JSON, checkpoint và archive không đổi. Các bảng đã gộp giữ nguyên số; chỉ bổ sung phạm vi và kết luận review hiện hành. Python local chạy conda `clrr`; dữ liệu ở `data_processed/<dataset>/`.
