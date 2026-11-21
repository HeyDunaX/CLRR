# P3 matched controls — queue RTX6000

## Queue12 mới được duyệt — seeds43/44, 13/10/2026

Tác giả yêu cầu chạy tự động đủ12 models, gồm cả đối chứng đóng góp; không
dừng theo quality gate giữa hai giai đoạn. Đây vẫn là development trên Amis
đã xem test, chưa là independent confirmation.

Recipe: `configs/experiments/amis_contrastive_multiseed_rtx.json`.
Workspace local/RTX: `outputs/experiments/amis_contrastive_multiseed_rtx/`.
Source bundle SHA256: `86871431852504c5ff45f6c6c62ac10ef98b975be9a42aa7912a4dd3985efe6b`.
Local CPU gate12/12 PASS; regression original-reference/optional chrF0 logging
5/5 PASS. Source reuse audit: protocol/runtime/data khớp seed42; model/loss,
optimizer và scheduler không đổi. Thêm chrF0 vào log validation từng epoch,
selector vẫn chrF++ beam1. HTTP download và helper dùng project conda clrr.

| Thứ tự | ID | Nhánh | Seed |
|---|---|---|---:|
| 1 | baseline-s43 | Baseline CE | 43 |
| 2 | nce-s43 | NCE-only A | 43 |
| 3 | p3-s43 | CLRR+LSR+NCE, detached P3 | 43 |
| 4 | baseline-s44 | Baseline CE | 44 |
| 5 | nce-s44 | NCE-only A | 44 |
| 6 | p3-s44 | CLRR+LSR+NCE, detached P3 | 44 |
| 7 | p2-s43 | LSR+NCE P2 | 43 |
| 8 | clrr_nce-s43 | CLRR+NCE, không LSR B | 43 |
| 9 | attached-s43 | Full donor attached C | 43 |
| 10 | p2-s44 | LSR+NCE P2 | 44 |
| 11 | clrr_nce-s44 | CLRR+NCE, không LSR B | 44 |
| 12 | attached-s44 | Full donor attached C | 44 |

RTX FP32, TF32 off, LR5e-5, b4×GA32, ES10/cap20, same source/target256,
AdamW/warmup/scheduler và CSV hashes như cohort seed42. Dùng lại seed42;
không train lại seed42 hoặc ghép A100 thành controls của cohort RTX.

Queue background: remote CPU gate, đủ12 full-backbone GPU gates, rồi train12
tuần tự. Mỗi run validation/upload/SHA verified xong mới dọn weights local và
chuyển sang run sau. Nếu gate/train/upload lỗi, queue dừng và giữ checkpoint
chưa được backup; không tự đổi batch/precision hoặc restart training từ đầu.
Sau run12 queue tự exit, giải phóng training lock; **không tắt máy RTX shared**.
Đánh giá test là bước riêng sau khi khóa toàn bộ12 selected checkpoints.

Chỉ launch một lần:

```powershell
conda run -n clrr python scripts/experiment.py launch --config configs/experiments/amis_contrastive_multiseed_rtx.json --account 1
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_contrastive_multiseed_rtx.json --account 1
```

Xem `launch_rtx.json`, `rtx_status.json`, `gates/<ID>/PASS.json`,
`runs/<ID>/progress.json` và `queue_validation_account1.json`. Khi agent đóng,
queue vẫn chạy độc lập; chat chỉ có cập nhật khi agent đang theo dõi.

Public checkpoint paths: giữ rule
`checkpoints/amis_mandarin/mbart-large-50/<method>/seed-43|seed-44/<recipe>/`.
Methods: `baseline`, `nce-only-v1`, `ordinary-full-v1`, `ordinary-lsr-v1`,
`clrr-nce-v1`, `attached-ordinary-full-v1`. Baseline recipe
`rtx6000-fp32-b4-ga32-es10`; các contrastive methods thêm `-tau0.1-beta0.1`.
Không overwrite checkpoint seed42 hoặc recipe A100.

Preflight bundle/receipts được lưu trong
`analysis/amis_mandarin/mbart-large-50/contrastive-controls/multiseed/preflight/`.

Đã deploy queue background PID3396045; remote CPU12/12 PASS. Preflight7files
public SHA verified ở revision `da7985a5bca0b21a09bc484c7eec644873b29ede`.
GPU gates đang chạy tự động trước training; xem `launch_observation.json`
hoặc CLI monitor để đọc trạng thái mới. Queue độc lập với phiên chat/SSH.

## Seed42: kết quả đã hoàn tất

**Cập nhật13/10/2026:** train/test và ba bước kiểm chứng không train đã xong.
Không training mới. Bootstrap18tests:0 đạt Holm<.05; interaction CI chứa0.
Audit6runs PASS, valBLEU chọn cùng epochs như chrF++. Frozen A diagnostic
PASS: centered retrieval68.58% S→T,69.27% T→S; rank119.96/120.06.
Zero optimizer steps, weights tải tạm đã dọn. Xem
[evidence report](../../outputs/experiments/amis_contrastive_evidence/EVIDENCE_REPORT.md).

Public evidence18files verified ở revision
`050e140d13e18b563b0df108a82c7ad9bfa8db6e`, prefix
`analysis/amis_mandarin/mbart-large-50/contrastive-controls/seed-42/evidence/`.
NPZ chưa công khai vì auto-review yêu cầu quyền riêng; giữ trong project RTX.

Ngày12/10/2026. Tác giả đã duyệt code, setup và chạy3 models seed42 trên RTX.
Chỉ validation; không tự mở test hoặc confirmation nhiều seed.

**Trạng thái đã xác minh:** CPU local/remote và GPU gates A/B/C đều PASS.
GPU gates lần lượt111.7/121.7/111.2giây; peak allocated12.08GiB mỗi nhánh.
Queue PID3035827 đã bắt đầu training A sau đủ3 gates, rồi tự chạy B và C.
Đây là trạng thái lúc setup; progress/VAL_COMPLETE là nguồn trạng thái mới.
GPU receipts đã verify public tại revision
`f6e0f1c4a1ad39cc34e687054fc657aa996833ed`, trong cùng preflight prefix bên dưới.

**Trạng thái cuối 12/10/2026:** A/B/C hoàn tất đủ20 epochs, không dừng sớm.
Checkpoint được chọn trên validation chrF++ beam1 tại epoch A19, B19, C18.
Artifacts/checkpoint public đã xác minh và weights local đã dọn. Queue kết thúc
sau C; không có run nào tiếp theo và không dùng test.

| ID | Nhánh | Best epoch | Val BLEU | Val chrF++ | Val chrF0 | HF revision |
|---|---|---:|---:|---:|---:|---|
| A | NCE-only | 19 | 17.0583 | 14.3556 | 16.4069 | `f70bae74090f513e7490a113415302a76a8f3709` |
| B | CLRR + NCE, không LSR | 19 | 16.8872 | 14.2274 | 16.3714 | `a38327ca33509abfd68c63b617eb4c8823e77568` |
| C | P3 với donor CLRR attached | 18 | 17.8040 | 14.7318 | 16.9037 | `db91428b3f63c8111df8d9f7ec583a3d7ec4bfbc` |
| P3 reference đã có | CLRR + LSR + ordinary InfoNCE, donor detached | 16 | 18.4584 | 15.1246 | 17.3609 | `e04195904361dcbdf36d4ae6c7f577ca37eb3256` |

**Đọc kết quả:** C cao hơn A (+0.7457 BLEU, +0.3761 chrF++, +0.4968 chrF0)
và B (+0.9168, +0.5044, +0.5323), nhưng thấp hơn P3 reference đã có
(−0.6544, −0.3928, −0.4572). B thấp hơn A nhẹ (−0.1711 BLEU, −0.1283 chrF++,
−0.0355 chrF0), nên nhánh CLRR+NCE không LSR chưa cho tín hiệu lợi trong
seed này. C so với P3 là phép đối chiếu trực tiếp về donor attached/detached
trong cùng recipe; kết quả validation nghiêng về donor detached. Đây chỉ là
development screening một seed, không có kiểm định thống kê, không chứng minh
hiệu ứng ổn định theo seed hoặc novelty. Không đánh giá test.

## Ba runs được duyệt

### Test được yêu cầu sau validation — 13/10/2026

Tác giả cho phép test cả3 checkpoint best-validation đã khóa. Engine test
chung chạy đủ575 câu gốc, beam4, FP32, TF32 off, batch2, max lengths256;
không training/checkpoint reselection. Source/target và replay cả3 metrics
PASS; public artifacts SHA/size verified tại HF revision
`c781e6bcaea2bb04af30ac9f6272bcadb2160e1d`. Weights tải tạm đã dọn sau upload
verified; queue hoàn tất. Tải C bằng Xet bị đứng; đã tiếp tục riêng C qua HTTP,
archive SHA vẫn khớp selection lock. A/B không đánh giá lại.

| Run | BLEU test | chrF++ test | chrF0 test |
|---|---:|---:|---:|
| A | 19.987915 | 19.025753 | 18.697140 |
| B | 19.792665 | 18.778548 | 18.492190 |
| C | 19.890514 | 18.853494 | 18.616045 |

A cao nhất trong ba run mới. P3 detached reference đã có: BLEU19.9158,
chrF++18.8726, chrF0 18.5920; C−P3 chỉ−0.0253 BLEU và A−P3+0.0721 BLEU.
Tín hiệu validation nghiêng về donor detached chưa chuyển thành khác biệt
BLEU rõ trên test. NCE-only vẫn cạnh tranh với P3, nên chưa bảo vệ được ưu thế
Full so với contrastive thông thường. Kiểm định mới không đạt Holm18;
một seed và test đã xem phục vụ development, không phải confirmation độc lập.
Evidence local/RTX: `outputs/experiments/amis_contrastive_controls_rtx_test/TEST_VERIFICATION.json`.
Artifacts test nằm ngay trong ba prefix checkpoint đã liệt kê bên dưới.

| Thứ tự/ID | Method ID | Objective | Câu hỏi |
|---|---|---|---|
| 1 / A | `nce-only-v1` | CE + .1 InfoNCE | Contrastive thông thường đã đủ tốt chưa? |
| 2 / B | `clrr-nce-v1` | CE + detached CLRR + .1 InfoNCE; không LSR | Có cần giữ LSR khi đã có negatives? |
| 3 / C | `attached-ordinary-full-v1` | P3 với donor CLRR attached | Detach có đóng góp riêng không? |

CLRR d2, strength.1, encoder/post-rewire cache. CE không có hệ số CLRR trong
loss; CLRR là thay đổi encoder. Target encoder vẫn stop-gradient ở cả3 runs,
kể cả C; C chỉ bỏ detach ở donor route source. Microbatch4 có3 negatives,
temperature.1; không có queue/cross-microbatch negatives hoặc parameters mới.

## Recipe và tính công bằng

- Amis→Mandarin:4600 train/576 validation; CSV hashes giống cohort trước.
- mBART-large-50, cùng model revision, seed/data_seed42.
- RTX6000 FP32, BF16/FP16/TF32 off; LR5e-5, microbatch4×accumulation32;
  ES10, cap20epochs, gradient checkpointing reentrant.
- Optimizer, warmup, scheduler, clipping, lengths256, pad_multiple8 và recipe
  khác giữ nguyên từ `amis_residual_effect_rtx.json`.
- Best checkpoint: validation chrF++ greedy beam1, original references,
  earliest best nếu tie. Supporting scores BLEU zh và chrF0. Không lựa chọn theo test.
- Baseline R1, CLRR-only P1, Full cũ R2, P2 và P3 dùng lại cohort RTX hiện có.
  Không so trực tiếp A100 thành matched training controls của cohort này.
- `REUSE_AUDIT.json`: protocol khớp hoàn toàn, run args khớp ngoại trừ
  ablation được duyệt; hash original CLRR/LSR/P3, metric và precision trainer
  không đổi so với bundle test cũ. Không sửa receipts/bundle lịch sử.

## Mã nguồn và gates

Code riêng: `src/methods/contrastive_controls/v1/`; dùng lại ordinary InfoNCE
trong `src/methods/residual_effect/v1/lsr.py`. Original CLRR và P3 không đổi.

C dùng donor addition ngoài checkpoint operation: backward recompute chỉ
chạy block nguyên bản, không đọc/ghi mutable route cache. CPU test xác minh
forward/loss khớp detached P3 và gradient attached khớp checkpointing on/off.
Không đổi lịch optimizer, RNG của CE hoặc bật dropout cho auxiliary views.

Gates bắt buộc:

1. CPU formula/donor gradient/checkpoint parity/export PASS:3 tests mới.
2. Regression ordinary/effect cũ PASS:4 tests.
3. CPU shared Trainer gate cho A/B/C PASS, kiểm tra accumulation32/save/load.
4. Remote lặp CPU gate trên đúng bundle.
5. **Cả3 full-backbone GPU gates PASS trước training A**:2 optimizer steps
   trên longest examples, đúng microbatch/accumulation/checkpointing, rồi
   save/reload logits, optimizer và RNG receipts. Model gate được bỏ sau PASS;
   mỗi training model khởi tạo lại từ pretrained cùng seed42.

Nếu bất kỳ gate nào lỗi, shell `set -e` dừng queue; không tự đổi batch hoặc
recipe để lách OOM. Không dừng jobs người khác. Trước launch free23542MiB,
utilization0%; gate và mỗi model yêu cầu ít nhất20000MiB free.

## Vận hành bằng CLI chung

Đã freeze: `configs/experiments/amis_contrastive_controls_rtx.json`.
Workspace: `outputs/experiments/amis_contrastive_controls_rtx/`.
Bundle SHA256: `e9cf3f39fbb8700f0758ad6e9eed9b9465e99286fde7283854ee97bd5efd3590`.

Sau khi tác giả xác nhận cho phép công khai bundle gồm code/recipe/tests/CSV,
preflight đã upload và verify6 files tại
`analysis/amis_mandarin/mbart-large-50/contrastive-controls/seed-42/preflight/`
trong canonical HF repo, revision `aa145806fc9bdac5ca8a36167bc2ffabbb339481`.
`bundle.zip` chỉ327544 bytes, không có token hay weights. Bản runbook trên HF
là snapshot lúc launch; trạng thái mới xem progress/receipts trong workspace.

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
conda run -n clrr python scripts/experiment.py inspect --config configs/experiments/amis_contrastive_controls_rtx.json
conda run -n clrr python scripts/experiment.py monitor --config configs/experiments/amis_contrastive_controls_rtx.json --account 1
```

Build/CPU/launch đã được thực hiện trong đợt này; không launch lần2 khi đã có
receipt/PID. Queue background tự chạy A→B→C, không phụ thuộc agent còn mở.
Toàn bộ repo/conda/cache/weights/logs nằm dưới
`/workspace/ura/thanhdat/projects/CLRR/`. Launcher giữ project training lock;
HF token tái sử dụng từ `.secrets/hf_token`, không vận chuyển trong bundle.

Remote log: `outputs/experiments/amis_contrastive_controls_rtx/queue.log`;
trạng thái từng model: `runs/<ID>/progress.json`; GPU gate: `gates/<ID>/PASS.json`.
Monitor thu JSON/CSV nhỏ, không kéo weights về local. Kiểm tra mỗi10phút khi
agent/người vận hành đang theo dõi; CLI không tự gửi chat khi agent đã ngừng.

## Hugging Face và cleanup

Public repo `FiveC/amis-rewire-checkpoints`, từng checkpoint:

```text
checkpoints/amis_mandarin/mbart-large-50/
  nce-only-v1/seed-42/rtx6000-fp32-b4-ga32-es10-tau0.1-beta0.1/
  clrr-nce-v1/seed-42/rtx6000-fp32-b4-ga32-es10-tau0.1-beta0.1/
  attached-ordinary-full-v1/seed-42/rtx6000-fp32-b4-ga32-es10-tau0.1-beta0.1/
```

Ba model đã upload `best.zip`, metadata, arguments, selected/final Trainer
states, predictions, metrics và receipts. Checksum/size remote đã xác minh ở
các revision trong bảng kết quả; weights local trên RTX đã dọn sau xác minh.
Các đường dẫn là ba prefix bên trên. Queue đã kết thúc sau C.

## Cách đọc kết quả

Các delta validation đã ghi ở phần kết quả. Trong ba run mới, C cao hơn A/B,
nhưng C thấp hơn P3 reference detached. B thấp hơn A nhẹ, nên chưa thấy lợi
ích của CLRR khi bỏ LSR trong đối chứng này. C so với P3 cũ là tín hiệu rằng
donor detached hữu ích hơn attached trong seed42; đây chưa phải kiểm định
thống kê hay bằng chứng ổn định theo seed. Các kết luận trên thuộc validation;
test được yêu cầu ngày13/10 đã báo riêng ở trên và cho thứ hạng khác.
Queue training/test đều kết thúc sau C.
Không mở thêm run. Novelty vẫn HOLD như
[review P3](P3_NOVELTY_AND_MECHANISM_REVIEW.md).

### X?c nh?n GPU gates ? 13/10/2026, 14:46 (Asia/Saigon)

- ?? 12/12 GPU gates PASS tr?n Quadro RTX 6000, c?ng bundle manifest `ffec529bd310aaeb3639c1699f10c7e7570a522a1d84f05c6fb26d80b8e640cb`.
- T?t c? receipts x?c nh?n save/reload logits kh?p, reload best checkpoint v? c? optimizer/RNG; FP32, microbatch 4, accumulation 32, checkpointing b?t. Peak allocated VRAM kho?ng 11.99?12.08 GiB trong tiny gates.
- Queue PID 3396045 ch?y ??c l?p (PPID 1, SID 3396045). Training d?i ??u ti?n ?? b?t ??u, log ??t step 2/720.
- Ng??i d?ng x?c nh?n ch? d?ng chat/t?t m?y c? nh?n; queue ti?p t?c to?n b? 12 runs, kh?ng t?t m?y RTX.

### Test ?a seed ???c duy?t ? 14/10/2026

Training 12/12 runs ho?n t?t l?c 05:14, t?t c? backups verified, weights tr?n RTX ?? d?n. T?c gi? cho ph?p test ng?y 14/10. Test workspace: `outputs/experiments/amis_contrastive_multiseed_rtx_test/`; worker d?ng engine test chung, c?ng bundle model g?c. 12 selections ?? kh?a t?i HF revision `b6e40b2d7166f16d5412c211f6465eb25b2027ba`, SHA256 t?ng archive kh?p VAL_COMPLETE. Queue PID 3867190, autonomous; ch?a c? k?t qu? test m?i t?i th?i ?i?m ghi.

Protocol: 575 c?u CSV g?c, beam4 FP32, BLEU zh, chrF++ native word_order2, chrF0 word_order0; kh?ng ch?n l?i checkpoint. Worker t? gh?p 6 checkpoints seed42 ?? test tr??c ?? ?? l?p mean/SD v? paired seed deltas cho seeds42/43/44. Thi?t k? bootstrap kh?a tr??c test m?i: 10000 shared paired sentence draws, RNG42; Holm90 = 10 c?p x 3 seeds x 3 metrics. N?m nh?nh so baseline v? n?m so s?nh: P3?NCE, P3?P2, P3?CLRR+NCE, Attached?P3, Attached?NCE. ??y l? exploratory development; n=3 exact two-sided seed signflip kh?ng th? ??t p<.05 (p t?i thi?u .25). Kh?ng coi sentence bootstrap l? b?ng ch?ng ?n ??nh gi?a seeds.

Artifacts test upload trong prefix checkpoint t?ng run; th?ng k? t?i `analysis/amis_mandarin/mbart-large-50/contrastive-controls/multiseed/test/`. Test queue t? d?ng sau upload x?c minh; kh?ng t?t RTX.

### K?t qu? test v? th?ng k? ?? x?c minh ? 14/10/2026

12/12 test runs m?i ho?n t?t; gh?p 6 checkpoint seed42 ?? test = 18 checkpoints, seeds42/43/44. To?n b? 575 c?u test g?c, beam4 FP32; test kh?ng d?ng ch?n checkpoint. Worker test/metrics d?ng shared engine; predictions order, 18 score replays v? Holm90 ?? ki?m ch?ng ??c l?p local PASS. RTX queue terminal QUEUE_COMPLETE, weights t?i test ?? d?n; GPU nh?n.

| Method | BLEU mean ? SD | chrF++ mean ? SD | chrF0 mean ? SD | BLEU ? baseline |
|---|---:|---:|---:|---:|
| Baseline | 19.7306 ? 0.4530 | 15.5468 ? 2.8916 | 18.2313 ? 0.3200 | +0.0000 |
| NCE-only | 19.6806 ? 0.2858 | 18.7104 ? 0.3443 | 18.3484 ? 0.3667 | -0.0501 |
| LSR + NCE | 19.2135 ? 0.1608 | 18.3675 ? 0.0153 | 17.9450 ? 0.0660 | -0.5172 |
| CLRR + NCE | 19.7895 ? 0.0329 | 18.7009 ? 0.0803 | 18.4312 ? 0.1154 | +0.0588 |
| Full detached (CLRR + LSR + NCE) | 19.9420 ? 0.6351 | 18.8099 ? 0.4091 | 18.5322 ? 0.4984 | +0.2114 |
| Full attached (CLRR + LSR + NCE) | 19.8749 ? 0.3366 | 18.7661 ? 0.2122 | 18.4795 ? 0.2926 | +0.1443 |

| BLEU contrast | Seed42 | Seed43 | Seed44 | Mean ? | Wins |
|---|---:|---:|---:|---:|---:|
| Full detached (CLRR + LSR + NCE) ? Baseline | +0.5232 | +1.0359 | -0.9249 | +0.2114 | 2/3 |
| Full attached (CLRR + LSR + NCE) ? Baseline | +0.4979 | +0.6495 | -0.7145 | +0.1443 | 2/3 |
| Full detached (CLRR + LSR + NCE) ? NCE-only | -0.0721 | +0.9588 | -0.1024 | +0.2614 | 1/3 |
| Full detached (CLRR + LSR + NCE) ? LSR + NCE | +0.8662 | +1.3700 | -0.0506 | +0.7285 | 2/3 |
| Full detached (CLRR + LSR + NCE) ? CLRR + NCE | +0.1232 | +0.7692 | -0.4347 | +0.1526 | 2/3 |
| Full attached (CLRR + LSR + NCE) ? Full detached (CLRR + LSR + NCE) | -0.0253 | -0.3864 | +0.2104 | -0.0671 | 1/3 |
| Full attached (CLRR + LSR + NCE) ? NCE-only | -0.0974 | +0.5725 | +0.1081 | +0.1944 | 2/3 |

Statistics: 10000 paired sentence bootstrap draws, RNG42. Single Holm family90 = 10 frozen pairs x 3 seeds x 3 metrics; all pairwise native SacreBLEU p-values matched independently. 0/90 pass corrected p<.05. Minimum raw p=.00639936; minimum Holm p=.57594241. P3?baseline BLEU: seed42 CI[-.7101,1.7748], seed43[-.1469,2.2581], seed44[-2.0390,.2074]; all three Holm p=1. Exact seed signflip P3?baseline raw p=.75, P3?NCE raw p=1; n=3 minimum attainable two-sided p=.25. Seed signflip values are descriptive raw p, not part of Holm90.

Interpretation: P3 mean BLEU leads this cohort by +.2114 over baseline, but only2/3 seeds win; P3?NCE wins1/3. Attached mean falls .0671 below detached. LSR+NCE falls below baseline3/3. Every candidate is below baseline in seed44. No demonstrated robust superiority, no basis to prefer attached based on validation alone. CLRR+NCE has small absolute BLEU SD=.0329, but baseline-relative advantage is not stable.

chrF++ has large baseline spread (13.8889,13.8656,18.8857), while chrF0 is steadier. Raw-reference replay PASS: this is not an identified scoring error. Sparse Mandarin word n-gram sensitivity was documented earlier; do not present the +3 chrF++ mean gap alone as robust quality improvement.

This remains exploratory development: seed42 test was previously seen, models/Amis test have been inspected. Sentence bootstrap conditions on frozen training runs and does not prove training-seed stability or novelty. Next scientific task is diagnose seed44 and the contribution of LSR; no additional training started here.

Artifacts: `outputs/experiments/amis_contrastive_multiseed_rtx_test/statistics/` and `TEST_VERIFICATION_LOCAL.json`. Public stats revision `2ef77e02ecd237de4b64ef064bd29929ab30154a`, path `analysis/amis_mandarin/mbart-large-50/contrastive-controls/multiseed/test/`; per-run test artifacts are inside each canonical checkpoint prefix. Local collection contains only JSON/CSV/Markdown; no weights.

### Nontraining followup complete ? 14/10/2026

CPU histories/predictions audit complete; eight new GPU probes and four verified seed42 reuses complete, zero optimizer steps. Final report: `outputs/experiments/amis_contrastive_multiseed_evidence/NONTRAINING_REPORT.md`; current interpretation recorded in `docs/research/RESEARCH_INSIGHTS.md`. All temporary weights cleaned. No training queue active.
