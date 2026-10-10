# Bước1–2: baseline42 và smoke kích thước thật — 05/10/2026

## Trạng thái

- **Bước2: FULLSIZE_GPU_PASS**, artifacts đã tải về và SHA khớp source local.
- **Bước1: VERIFIED**. Original selected-step42 không phục hồi được; fresh FT seed42 đã chạy đủ20 epochs, best step720/epoch20, test scores khớp hoàn toàn số cũ. Namespace riêng, scores cũ giữ nguyên. Chưa chạy12 runs phương pháp.
- **A100 đã tắt**, CLI xác nhận `Session terminated`.

## Kết quả baseline42 mới

| Đại lượng | Run mới | So với số cũ |
|---|---:|---|
| Completed epochs | 20 | Không early stopping |
| Selected step/epoch | 720 /20 | Đây là provenance của fresh run; không gán ngược cho archive cũ |
| Best validation chrF++ (beam1) | 13.901540564847089 | Khớp preserved weights đã replay |
| Test BLEU (beam4, FP32) | 19.728081433402032 | Khớp hoàn toàn |
| Test chrF++ (beam4, FP32) | 18.713838023321145 | Khớp hoàn toàn |

Bốn primary surviving checkpoints216/252/468/504: BLEU, chrF++ và loss đều khớp old/new, abs difference0. Test predictions đủ575 rows, source/target khớp raw CSV; tính lại metrics từ predictions bằng original references PASS. Source và cả ba data split hashes khớp receipt.

Best ZIP2,270,665,528 bytes, SHA256 `e2d5c884c8c2da270aa1bbeaa4e36990021f1559b5fca2926d05cca3f5b0e53c`. HF archive LFS SHA và size khớp; final Trainer state, selected Trainer state, metrics và predictions cũng khớp remote Git blob/LFS checksums. Không tải model weights xuống máy local.

Colab collection download trả kết quả chậm; đã khởi động thêm recovery small metadata từ HF. Gói Colab sau đó về đầy đủ, giữ được full console log. Verification cuối dùng `colab_zip` và kiểm tra archive/source/data/metrics, không coi việc gọi lệnh download là bằng chứng file đã về.

Artifacts baseline: `outputs_rebuttal/amis_corrected_baseline42_20261005/local_verification.json`, `collection_receipt.json`, `final_trainer_state.json`, full training log, metrics và predictions. Evidence immutable ZIP checksum ở local verification.

## Quyết định reuse cho corrected cohort

| Seed | Baseline dùng tiếp | Selection provenance |
|---|---|---|
| 42 | Fresh run `mbart-amis-baseline-seed42-corrected` | 720/epoch20, full fresh chronology, HF archive và metadata verified |
| 43 | Reuse preserved selected checkpoint theo protocol đã audit | 720/epoch20, hash-verified original log/metrics/history; inventory252 được sửa riêng |
| 44 | Reuse preserved selected checkpoint theo protocol đã audit | 612/epoch17, hash-verified original log/metrics/history; inventory504 được sửa riêng |

**GO cho baseline controls và triển khai cohort theo recipe đã khóa**, không cần train lại43/44 chỉ vì lỗi inventory. Fresh42 tái lập đúng các mốc old baseline; source accumulation fix áp dụng wrappers, baseline training đã qua actual Trainer checks. Điều kiện reuse: giữ data/model revision, optimizer, selection và inference protocol đã audit. Đổi recipe thì quyết định này phải được xem lại.

Manifest cho bước sau: **`outputs_rebuttal/amis_fullsize_gate_20261005/baseline_controls.json`** khóa selected-step và HF archive identity của cả ba seeds. Historical V5 inventory42 vẫn giữ best_step unknown; không dùng historical42 thay fresh42 trong corrected cohort.

**Chưa GO chất lượng phương pháp**: kết quả này hoàn thiện control/provenance, không chứng minh CLRR hoặc Full mạnh hơn LSR. Bước sau là12 corrected runs (CLRR-only, matched LSR.10, tuned LSR.03, Full ×seeds42/43/44), chưa launch trong lượt này.

## Bước1 — truy provenance

Đã kiểm tra local log, surviving partial Trainer states, recovery receipts, HF file metadata và ZIP central directory bằng HTTP Range64KB. Original log chỉ giữ eval đến504; partial state504 không chứng minh final selected-step. Selected ZIP không có trainer_state/optimizer/scheduler/RNG. Không gán216 hoặc504 cho final selected archive.

Evidence: `outputs_rebuttal/amis_fullsize_gate_20261005/BASELINE42_SELECTION_AUDIT.md` và `baseline42_selection_trace.json`.

Run mới đã hoàn tất: `mbart-amis-baseline-seed42-corrected`, HF repo `FiveC/amis-rewire-checkpoints`, prefix **`amis_corrected_baseline42_20261005`**. Khởi đầu từ pretrained pinned revision, không resume hoặc sửa archive cũ.

Recipe: mBART-50 `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`; Amis4600/576/575; tl_XX→zh_CN; seed42; LR5e-5, WD0, warmup.06; batch32×accum4; evalbatch8; lengths256; BF16 + gradient checkpointing + fused AdamW; max20 epochs, patience4; validationbeam1, testbeam4, generationFP32; selection validationchrF++.

Sau mỗi checkpoint, callback upload Trainer state và training receipt riêng lên HF; final selected provenance được giữ riêng. Final best ZIP phải qua remote size và LFS SHA256 gate. Full optimizer/RNG chưa được backup remote; selection metadata không thay resumable state. Không claim có thể resume sau mất runtime từ selected best ZIP.

## Bước2 — kiểm tra thực thi

Sử dụng mBART-50 đầy đủ610,879,488 parameters, actual Amis train128 rows và validation8 rows. Mỗi family chạy2 optimizer steps (8 microbatches): một step với dynamic padding dữ liệu thật; một step stress với source/target padded256. Sequential sampling bảo đảm thứ tự hai điều kiện; đây là smoke, không phải training benchmark.

| Family | Actual batches peak allocated GiB | Stress256 peak allocated GiB | Stress256 peak reserved GiB | Kết quả |
|---|---:|---:|---:|---|
| Baseline | 7.30 | 33.45 | 36.38 | PASS |
| CLRR-only | 7.31 | 33.45 | 36.40 | PASS |
| LSR-only | 7.31 | 33.48 | 36.40 | PASS |
| Full | 7.33 | 33.48 | 36.40 | PASS |

Tất cả dùng batch32, accumulation4, BF16, checkpointing, fused AdamW, FP32 parameters và pinned pretrained revision. Loss/parameters finite; Trainer xác nhận model_accepts_loss_kwargs=False. Validation generation FP32, beam1, batch8 chạy được. Các tiny save/reload/gradient equivalence checks trước đó vẫn là bằng chứng riêng.

GPU: A100-SXM4-40GB Standard; không yêu cầu high-mem. Runtime: torch2.6.0+cu124, Transformers4.57.6, Accelerate1.15.0. Bundle11 source/script/data files: pre-upload, remote và local hashes khớp.

Artifacts: `outputs_rebuttal/amis_fullsize_gate_20261005/`: `FULLSIZE_PASS.json`, từng family.json, `local_verification.json`, source/data manifests, environment và log.

PASS xác nhận footprint đã thử. Stress là padding256, không phải128 examples có256 tokens thật; mask/value có thể làm thay đổi hành vi numeric. Không đảm bảo mọi batch hoặc inferencebeam4worstcase đều có footprint giống smoke. Quality/superiority cần cohort nhiều seeds; smoke không tạo score paper.
