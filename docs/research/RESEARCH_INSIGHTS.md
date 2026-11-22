## Cập nhật 2026-11-22 — Pilot A100 GOC 9 runs (validation-only, NO-GO cho GOC)

**[ĐO ĐƯỢC]** Hoàn tất và xác minh 9/9 HF backups trên A100 40GB (`BF16`, `TF32 train=on`, `SDPA`, `B32×GA4`, `LR 5e-5`, `max 20 epochs`, `ES10`, selector `eval_bleu` beam1 FP32, seed 42; báo cáo [GOC_PILOT_A100_REPORT.md](../tasks/reports/GOC_PILOT_A100_REPORT.md), commit `63364038c0392a26874b1841bbdf16ba7d44c07f`):
- `BASELINE_S42`: `17.5264` BLEU / `16.7632` chrF0 / Val Loss `2.8913` / Rare CJK `78/789 (9.89%)`.
- `W_B125_S42` / `W_B250_S42` (R-Drop): `17.7069` / `17.3802` BLEU (`17.0817` / `16.8022` chrF0); Rare CJK `79` (`10.01%`) / `83` (`10.52%`).
- `GOC_B125_S42` / `GOC_B250_S42` (`w→l`, sg): `18.0944` / `17.8097` BLEU (`17.3305` / `17.0887` chrF0); Rare CJK `76` (`9.63%`) / `74` (`9.38%`).
- `GOC_SYM_B125_S42` / `GOC_SYM_B250_S42` (`0.5/0.5`, sg, EDWARD-NMT Eq. 13): **`18.4022` / `18.5143` BLEU** (`17.4407` / **`17.5405`** chrF0); Rare CJK `79` (`10.01%`) / **`83` (`10.52%`)**.
- `GOC_INV_B125_S42` / `GOC_INV_B250_S42` (`l→w`, sg falsifier): `16.2213` / `15.1466` BLEU (`15.6328` / `15.0900` chrF0); Val Loss thấp nhất bảng `2.5071` / `2.4671`; Rare CJK `71` (`9.00%`) / `49` (`6.21%`).

**[QUYẾT ĐỊNH — NO-GO CHO GOC]** GOC đạt `GOC > W` và `GOC >> INV`, nhưng **thất bại trên cổng cơ chế trước đối chứng `GOC_SYM` ở cả 2 mức $\beta$** (`-0.3078` BLEU tại `1.25`, `-0.7046` BLEU tại `2.50`) và giảm recall ký tự hiếm. Dừng GOC, không mở thêm seed hay test.

**[ĐO ĐƯỢC — CHẨN ĐOÁN LỖI CÒN LẠI TRÊN A100 VALIDATION]**
1. **`W` vs `SYM-SG`:** Loại bỏ thành phần Reverse-KL Jacobian `0.5*beta*p_v*(log(p_v/p_bar)-KL)` bằng `stop-gradient` hai chiều tăng `+0.70` ($\beta=1.25$) đến `+1.13` ($\beta=2.50$) Val BLEU, chủ yếu ở 3-gram/4-gram precision (`p3: 12.6→13.8`, `p4: 7.3→8.6`). Tuy nhiên `SYM-SG` trùng dạng với EDWARD-NMT Eq. 13 nên giữ làm **comparator A100 mạnh nhất mới (`18.5143`)**, không dùng làm claim novelty độc lập.
2. **Mất cân bằng teacher-forced CE vs autoregressive BLEU:** Trên cả `BASELINE`, `W`, `SYM-SG`, `eval_loss` chạm đáy ở epoch 6–7 (`2.46–2.60`) rồi tăng đều tới epoch 18–20, trong khi beam-1 Val BLEU vẫn tăng thêm `+4.15` đến `+4.18` điểm. Ngược lại `GOC_INV_B250` giữ `eval_loss` thấp nhất (`2.467`) nhưng BLEU tệ nhất (`15.15`).
3. **Khủng hoảng source OOV & hallucination trên Amis vs Ê Đê:** Trên Amis validation, **84.2% (485/576)** câu có $\ge 1$ từ nguồn OOV (16.6% tổng token; 98.2% OOV chia sẻ chuỗi con $\ge 4$ ký tự với train); BLEU của `GOC_SYM_B250` tụt từ `29.73` (`0% OOV`, `n=90`) xuống `18.50` (`0–25% OOV`, `n=386`) và sụp xuống `7.93` (`>25% OOV`, `n=100`), kèm hiện tượng hallucination toàn câu dù các từ còn lại đều quen. Trên Ê Đê (`rhade_vietnamese`, 15.092 train / 1.000 val), OOV nguồn chỉ là **1.0%** (`10.0%` số câu), cho thấy method mới hướng tới bài ACL đa ngôn ngữ cần giải quyết vấn đề tổng quát hơn chỉ riêng hình thái OOV của Amis.

---

## Cập nhật 2026-11-19 — initialization calibration và pilot mới

**[ĐO ĐƯỢC]** Hiệu chuẩn đúng pretrained mBART, 3 fixed TRAIN batches, 0 optimizer steps: median lambda136.27606275570483 cho target0.1×full-backbone CE gradient norm. FP32/FP64 penalty error lớn nhất0.001649%; eta1/r32/basis11. Không dùng lambda4.13724e7 tại trained W. Receipt/hash: `outputs/experiments/amis_virtual_order_initialization_rtx/{COMPLETE,CALIBRATION_BATCHES,COLLECTION}.json`.

**[ĐO ĐƯỢC]** 7CPU tests PASS, gồm wrapper đối chiếu công thức/gradient dưới reentrant checkpointing và checkpoint cursor. Queue PID139985 mở theo tác giả: GPU optimizer+resume gate trước, PASS mới train đúng một Amis seed42. **GPU gate đã PASS**: 2optimizer steps B4/GA32, save/reload và optimizer/scheduler/RNG resume khớp, beam4 cache khớp; peak14.3486GiB,390.18seconds cả gate. Training dài đã bắt đầu, max720steps; chưa có validation chất lượng mới. Recipe `configs/experiments/amis_virtual_corrective_order_pilot_rtx.json`; ledger RTX có cấu hình, HF đích và commands.

**[SUY LUẬN]** Scale của surrogate thay đổi mạnh theo trạng thái model; hệ số từ trained model không thể chuyển thẳng về initialization. Hệ số fixed có thể làm auxiliary suy yếu khi training tiến triển; đây là rủi ro cần đọc trajectory, không tự thêm adaptive weight. **[CHƯA XÁC MINH]** method vượt W/baseline hoặc cải thiện generation. Không mở test hoặc nhân seeds.

---

## Cập nhật 2026-11-19 — virtual corrective-order, fixed checkpoint

**[ĐO ĐƯỢC]** Code riêng đã qua6CPU checks và ba chẩn đoán RTX,0optimizer steps, không validation/test.8TRAIN microbatches tại W42: eta1 có feature-gradient FP32/FP64 cosine≥0.999979; median cosine auxiliary order/W0.1774. Full-backbone3batches cũng cho hướng khác W (cos0.1352/0.4620/0.3476). Đây là tín hiệu khác local static consistency, không phải bằng chứng generation tăng.

**[ĐO ĐƯỢC]** Eta0.01/0.1 bị sai số FP32 lớn; eta1 median relative SKL error0.093%. Objective thô rất nhỏ; median scale tại **trained W**4.13724e7 cho gradient target0.1×CE. Không dùng hệ số này để train từ pretrained; cần scale tại đúng initialization. Gradient norm không đo Adam update ratio.

**[ĐO ĐƯỢC]** Forward/backward trên microbatch4 ngắn padded24 không OOM, peak allocated≈6.315GiB. Timing CE2/W/order≈0.365/0.371/0.558seconds, order timing tại eta0.1; không gọi là complete training VRAM hoặc epoch time. Weights không đổi; chỉ dọn weights tạm đã xác minh, giải phóng6.8926GiB.

**[SUY LUẬN]** Hướng nhỏ tránh được bottleneck triển khai full-backbone unroll trong phép đo này; còn đáng kiểm chứng bằng pilot khóa trước. **[CHƯA XÁC MINH]** correction-order sensitivity gây lỗi dịch, basis không bị né khi train, hoặc method tăng BLEU. Chưa mở training. Báo cáo: `outputs/experiments/amis_virtual_corrective_order_probe_rtx/REPORT.md`; thiết kế/novelty: [TWO_METHOD_LOW_MEMORY_REVIEW.md](TWO_METHOD_LOW_MEMORY_REVIEW.md).

---

> Historical follow-up (2026-11-15): [SAC-v1 method design](SOURCE_ATTRIBUTION_METHOD.md)
> Evidence debate: SAC training admission HOLD; W gains motivate consistency,
> not a demonstrated source-attribution deficit. Two-round NotebookLM debate
> and corrections: `outputs/experiments/source_attribution_design/EVIDENCE_DEBATE.json`.
> Frozen SAC probe completed: W absolute C25.2% lower than baseline; adjusted
> C/error association weak (partial rank0.0333, approximate95%CI[-0.1709,0.2402]).
> Training remains HOLD; no SAC training launched. Results below and in SAC design.
> has a mathematical prototype and9 CPU checks PASS; no GPU integration or
> training has started. [PLC failure analysis](PLC_FAILURE_ANALYSIS.md) remains
> the diagnostic evidence. Historical pilot scores are unchanged.

## PLC RTX pilot completed — 2026-11-15

All five runs have VALIDATION_COMPLETE_VERIFIED and public HF backup receipts.
Both queues report QUEUE_COMPLETE; suite log ends SUITE_TRAINING_COMPLETE.
No test was evaluated. These are exploratory seed42 validation beam1 results.

| Model | BLEU | chrF++ | Delta BLEU vs baseline | Delta BLEU vs W |
|---|---:|---:|---:|---:|
| Baseline (reused) | 17.682329 | 14.703005 | +0.000000 | -0.590080 |
| W (reused) | 18.272410 | 15.172807 | +0.590080 | +0.000000 |
| P_PREFIX_S42 | 17.714812 | 14.845499 | +0.032482 | -0.557598 |
| D_DIAGONAL_S42 | 18.271456 | 15.179737 | +0.589127 | -0.000954 |
| E_ENDPOINT_S42 | 17.907733 | 14.970888 | +0.225404 | -0.364677 |
| WCAL_S42 | 17.241326 | 14.421296 | -0.441004 | -1.031084 |
| MIDDLE_S42 | 16.677522 | 13.937579 | -1.004807 | -1.594887 |

PLC fails its predeclared gate: P−W=-0.557598 (<+0.30), and P is below D/E.
P−baseline=+0.032482 and P−Middle=+1.037289 do not rescue the primary failure.
D is effectively tied with W (delta=-0.000954 BLEU); no significance claim follows.
The narrowly positive frozen-W cross-drift diagnostic did not translate into a
benefit from penalizing ordered prefixes at lambda0.10. This does not prove that
cross-position covariance is universally harmful or that other weights would work.
Wcal is lower than W by1.031084 BLEU; this initial gradient-norm match does not
preserve optimizer directions or trajectories. Do not infer that every stronger KL
coefficient fails from a single seed. Middle uses batch4 negatives and is not the
A100 batch32 comparator. Baseline backend retains its historical provenance caveat.

Decision: NO-GO for automatic PLC replication/A100 promotion or further sweep.
No new training/test launched. Preserve all results, including unsuccessful controls.

All five HF best.zip SHA256 metadata values independently match completion receipts;
no .pt/.bin/.safetensors/.zip files remain in the five remote run directories.
Only small JSON receipts were downloaded locally. Verification artifacts:
`outputs/experiments/amis_prefix_consistency_pilot_rtx_suite/HF_COMPLETION_VERIFICATION.json`
and `LATEST_COMPLETE_STATUS.json`.

## PLC implementation and scientific admission — 2026-11-14

- Isolated v1 implementation and five CPU tests passed locally/RTX: exact kernel
  and gradients, W equivalence at lambda0, dropout/checkpointing/accumulation,
  native inference/save-load, masking/extreme-score checks.
- Frozen W42, validation576×8 dropout pairs: mean cross=P−D=+0.02982932,
  sentence-bootstrap95% CI[+0.00053590,+0.06009497]. This narrowly passes the
  predeclared mechanism admission, with zero optimizer updates. It does not
  establish a translation gain or a causal explanation of existing W gains.
- Train-only initial gradient-norm calibration on64 rows yields beta=2.5639090251.
  Wcal is thus close to the old beta2.5 B control, but is a separately specified
  fixed control; do not silently substitute B's old scores for Wcal.
- Author requested baseline and Middle comparisons. Reuse RTX baseline/W; no
  matching RTX Middle metadata found, so add Middle42 with the same locked RTX
  protocol. Its microbatch4 contrastive negative pool differs from A100's32.
- Next: all GPU gates then five seed42 validation-only runs P/D/E/Wcal/Middle.
  No success claim before scores; no automatic test or seed expansion.

## Proposed next design: Prefix Likelihood Consistency — 2026-11-14

Complete review card is now at the TOP of METHOD_REDESIGN_CANDIDATES.md.
Design only: no new method implementation, GPU, training, test or score.
The proposal keeps ordinary W(beta1.25) and adds a squared cumulative
gold-log-probability difference between its two dropout views, normalized by
prefix length. Its exact kernel has trace1 and exposes signed cross-position
terms. This is not sequence KL, an exposure-bias fix or a grounding guarantee.

Primary R-Drop, Fraternal Dropout, LR-Drop, SeqKD and CCL were inspected;
NotebookLM queries79f9282a190b/4a7c296e29b1 completed. No exact kernel was
identified in that bounded audit; no global novelty certificate follows.
Variance regularization itself is prior, and a loss modification alone is not
enough for the proposed paper contribution. Required evidence is measured
ordered score drift plus gains beyond diagonal/endpoint/strength controls.

CPU float64 algebra checks PASS, including2Ka, trace/positive definiteness and
both-view/all-logit gradients; receipt outputs/experiments/prefix_path_design/
MATH_CHECKS.json. These are not a method smoke or performance evidence.
Next proposed action is a zero-update W42 dropout diagnostic; token-aggregated
older CSVs cannot supply its missing ordered log probabilities. Only after
that evidence and technical admission: at most4 new Amis seed42 models P/D/E/
Wcal, followed conditionally by4 replication models. No queue is opened.
All old scores, recipes, data, checkpoints and remote processes are untouched.

## Stronger-design screen: source-conditioned projected consistency — 2026-11-14

Design/literature only; no new training, GPU gate or translation score.
Candidate maps two dropout output distributions through a frozen TRAIN-only
lexical stochastic matrix to source-type/NULL bins, then applies symmetricKL.
Primary SHR(NAACL2019)section3.2/Figure3/Eq7 already uses vocabulary projection
followed by KL in NMT; its global hierarchy and cross-layer classifiers differ
from this draft. KNOT already incorporates output-label semantic relations
through OT; MLS addresses source-only shared-vocabulary smoothing mass.
Those differences do not alone establish a sufficient new contribution.

Constructed probability example checked in conda clrr: p=(.6,.1,.3),
r=(.1,.6,.3), grouping first2 gives fullSKL=.8958797346 and projectedSKL=0.
This demonstrates ignored distinctions, not an observed translation failure.
NULL degeneration and merging incompatible words are risks, NOT measurements.
W>B is evidence of strength sensitivity, not proof of source neglect or harm
from matching valid alternatives. Source-response diagnostics already exist.

NotebookLM queries6de8746e370e and ca9fae5cb023 completed with actual sources;
the latter retrieved SHR/KNOT/MLS and agreed projectedKL itself is prior work.
Current decision: HOLD/NO-GO for long training in this form. Do not adopt its
unsupported gradient-norm bound, claim of no encoder gradients, predicted
NULL mass or guaranteed paper rejection. Review record/formula/corrections
are at the TOP of METHOD_REDESIGN_CANDIDATES.md. Original numbers unchanged.

## Increment reuse: CPU prototype and contribution decision — 2026-11-14

Actual progress: isolated RO/DW/RW prototype implemented at
src/methods/increment_reuse/v1/model.py; CPU gates cover exact W initialization,
nonzero-route donor/scalar/backbone gradients with reentrant checkpointing and
GA32, masks, save/load and generation. See CPU_GATE.json under
outputs/experiments/increment_reuse_preflight. No new translation scores,
GPU gate, remote upload or long training. Not registered in the runner.

Primary HC Eq2 locally subsumes RO through read weights(1,alpha,-alpha),
carryu, and first-stream FFN write. MUDDFormer Eq7 separates attention inputs
but still couples FFN read/residual. NotebookLM query87efecb66e05 actually used
both newly added primary sources and agrees on this comparison. Do not claim
new read/write mathematics, zero added parameters(alpha addsONE), or novelty
from missing search hits. Local operator subsumption is not full-model
architectural equivalence. This direction currently supports an engineering
RQ, not a standalone architecture claim. Non-significance would not establish
no effect. W remains an ordinary R-Drop comparator, not a ceiling.

Read-only RTX check succeeded:23952MiB free of24576MiB; no GPU operation or
process mutation. Author explicitly chose stronger-design-first, BEFORE training, on2026-11-14.
RO/DW/RW remains reference-only; its3+4run budget is INACTIVE. No GPU gate,
engineering screen or replication is admitted for it.
Historical conditional3+4run plan and unchanged BLEU/ES10 recipe are in the
current section of METHOD_REDESIGN_CANDIDATES.md. Historical entries below
record earlier states; the prior DESIGN ONLY status is superseded by this CPU
prototype result. No experiment numbers were overwritten.

## RTX-first next-method design — 2026-11-14

Author confirms RTX development before A100 final runs. Design/literature only;
no implementation, GPU use, training or test. CE-oracle routing was screened
out because it repeats the16-Oct routing audit and adaptive-depth prior.
Recipient-response draft simplified algebraically to input-only FFN increment
reuse; inspected local MBartEncoderLayer is pre-LN. Conditional RO/DW/RW screen
and exact formulas are in METHOD_REDESIGN_CANDIDATES.md (current top section).

NotebookLM reviews c05c4efab4ca/03162c6a17eb/e7f3be931df8 completed; primary
papers read include Depth-AdaptiveTransformer, DynamicLayerAggregation,
AACL2026 inter-layer coordination AppendixA, Hyper-Connections and ATO.
Response finite differences and separated read/write routing already have
prior. No standalone algorithmic novelty clearance or new score is claimed.
HC embedding/reduction remains an explicit admission gate, not a reason to
open training now. Existing W/baseline/14 A100 comparator results retained.
Reject NotebookLM's arbitrary numeric SNR gates, sentence-BLEU oracle bound,
generic LayerNorm projection equalities and claims a nonzero scalar or damaged
inference donor proves a causal training mechanism. No extra diagnostic rerun
is justified merely because an already-completed probe was rediscovered.

## Method redesign and comparator reuse audit — 2026-11-14

No new method admitted; no implementation, GPU, training or test. Two designs
were rejected before pilot: decoder-Fisher alignment overlaps local predictive
KL; token-transport layer differences have close SMD/FDD prior and no supported
central contribution yet. NotebookLM dc27ac1819af completed; its exact/global
KL, complexity, finite-difference-order and causal overclaims were corrected
against algebra and primary sources. This is not a negative translation result.

Retain14 existing A100 ES10 comparator results; local frozen-plan data hashes
match RTX, but precision/selector/microbatch/seed differences remain. No claim
of fully matched RTX comparisons or fresh HF verification. Author BLEU selector
stays unchanged. No blanket rerun; only concrete matched-control or closest-prior
gaps can justify future added runs. Full audit and reuse inventory:
`outputs/experiments/method_redesign_audit/REVIEW.md`.
Current design ledger: `METHOD_REDESIGN_CANDIDATES.md`.

## Conditional dropout pilot HOLD — author clarification, 2026-11-14

E/D/Wcal will not be launched. The author requires a credible paper-level
contribution before pilot training; a weak-novelty mask-coupling mechanism
probe does not satisfy that objective. No new trainer implementation, GPU gate
or training was performed. A read-only RTX availability check failed because
local Tailscale pipe access was denied; this did not launch remote work.
The prior three-run budget below is historical, not an active plan.

## Conditional dropout candidate audit — 2026-11-14

Design and prior audit only; no new training or test. Independent encoder /
shared active decoder masks with ordinary R-Drop loss is a mechanism probe,
not established standalone ACL novelty. DDR, ECT and DuetMatch already cover
important ingredients. NotebookLM follow-up c9ca71500f59 completed; primary
source corrections are recorded. Five exact finite-mask CPU checks PASS;
these are toy mathematical checks, not translation improvements or GPU gates.
Proposed Amis mBART seed42 budget: E candidate, D swapped perturbation control,
Wcal initial-strength control (up to3 new models), existing W/B/A/baseline reused
only after provenance checks. No queue opened. Full formula, prior comparison,
locked recipe, implementation gates and promotion/stop rules:
`docs/research/CONDITIONAL_DROPOUT_CANDIDATE.md`.
Small check/review artifacts: `outputs/experiments/conditional_dropout_design/`.

## Source-response frozen probe — COMPLETE, 2026-11-14

No new training. Baseline/B(KL2.5)/W(KL1.25) seed42 measured on first64 Amis
validation rows,4 fixed dropout pairs and2 wrong-source pairings. FP32 SDPA,
TF32 off, teacher forcing, no optimizer or test. CPU checks and all3 GPU gates
PASS (strict checkpoint loading, exact RNG replay, different masks, no grads).
Queue exited0; all transient weights removed. Final free VRAM23952MiB.
Collected16 small artifacts with SHA verification; all summary means replayed
from1024 CSV records/checkpoint within1e-10. No weights downloaded locally.

| Lexical diagnostic | Baseline | B KL2.5 | W KL1.25 |
|---|---:|---:|---:|
| Ordinary dropout symmetric KL |0.53734|0.28508|0.36922|
| Gold logp correct-minus-wrong source |3.82411|3.56874|3.83621|
| Matched CE |3.55714|3.31033|3.45383|
| Gold-minus-competitor margin |-0.51002|-0.52944|-0.45422|
| Teacher-forced top1 |0.49509|0.50208|0.49717|

Compatible with a regularization-strength tradeoff, not a causal mechanism:
W has source gap greater than B on52/64 rows, dropout KL greater on59/64;
but its CE is worse than B and top1 slightly lower. A larger source gap does
not imply better correct-source likelihood: W correct/wrong CE both worsen
relative to B. Shuffled negatives and confidence scaling confound grounding.
Existing generation/diagnostic sentence-delta correlations are near zero
(|Spearman rho|<=0.090 on64 rows); no simple generation explanation established.
W remains tuned R-Drop, not a new method or significance claim.

Raw q=softmax(z_correct-z_wrong) gives89-93% mass to tokens whose ordinary
matched p<1e-5. Tail mass92.65-93.82% must be contextualized: train-count<=1
IDs occupy98.8978% of vocabulary, so raw tail mass alone is NOT collapse or
pathological enrichment evidence. The rejected response-KL candidate remains
NO-GO for the previously documented prior overlap and uninformative minimum;
this diagnostic does not train that loss.

Full verified report:
`outputs/experiments/amis_consistency_source_response_rtx/ANALYSIS.md`.
Plan SHA256:0920cc8eba608d0ced6bcdea37e1782efec1add8f37419af3ea3960a2f93d110.
NotebookLM review87d9c27d8535 completed; its causal/global-collapse/noise
overclaims and CKD author misattribution are explicitly rejected in the output
NOTEBOOKLM_REVIEW.md. Primary-source audit remains authoritative.
Next: design a falsifiable source-conditioned consistency RQ and audit against
R-Drop/DDR/ALD, CBMI and CKD BEFORE any method implementation or pilot. Current
probe alone does not justify another sweep. No test, Ede or A100 opened.

## Immediate novelty review + W/B CPU error audit — 2026-11-14

No new training opened. Two candidate directions were rejected before method
implementation: per-token gradient-budget consistency overlaps norm-balancing
prior; source-response consistency uses the softmax-of-logit-differences
transform already explicit in Comparative Knowledge Distillation, Equations
1–3. Source-response self-consistency additionally admits zero auxiliary loss
under source insensitivity. This does not prove the complete algorithms were
previously published or that CE training must collapse. See
[audit and next frozen-probe plan](TOKEN_CONSISTENCY_NOVELTY_AUDIT.md).

New CPU-only analysis reuses six B/W prediction sets, 576 validation rows and
the existing analysis script; provenance and metric replay PASS. W−B BLEU
leave-one-row-out ranges: seed42 [+0.1847,+0.4545], seed43 [+0.6394,+0.9171],
seed44 [−0.3202,−0.0117]. No single-row omission changes each seed's sign.
Longest-source quartile (>8 whitespace words) W−B BLEU is +0.8579/+1.9164/
+0.6764, a post-hoc descriptive lead only. Sentence chrF0 wins/losses are
197/200,187/211,173/215; corpus gains do not imply improvement on most rows.
No statistical significance, causality or mechanism claim follows from this
audit. Artifacts:
`outputs/experiments/amis_consistency_strength_w_replication_rtx_suite/validation_error_audit/`.

NotebookLM reviewed the candidate with explicit author permission. Its initial
contrastive-decoding and gradient-explosion statements were corrected against
primary papers and algebra; do not reuse the uncorrected response as evidence.
The then-proposed frozen baseline/B/W source-response probe is now complete;
see the dated entry above. No test, Ê Đê or A100 opened. W remains a tuned
R-Drop comparator.

## W/B replication COMPLETE — validation only (2026-11-14)

All four new runs completed20epochs/720steps. HF backup verified for all four;
24 small JSON/CSV artifacts collected, original576 validation references aligned
and 8 corpus metrics replayed within1e-10. No checkpoint weights downloaded.

| Seed | Baseline BLEU | B beta2.5 | W beta1.25 | W minus B |
|---|---:|---:|---:|---:|
| 42 | 17.6823 | 17.9107 | 18.2724 | +0.3617 |
| 43 | 16.7345 | 17.5757 | 18.3958 | +0.8200 |
| 44 | 16.2089 | 18.3274 | 18.1881 | -0.1393 |
| Mean | 16.8753 | 17.9380 | 18.2854 | +0.3475 |

W-minus-baseline mean BLEU +1.4102, positive3/3; W-minus-B positive2/3. W mean chrF++ 15.1272, baseline 14.2180, B 14.8973.
Practical development gate PASS; not statistical significance or novelty proof.
W44 is slightly below B44, so do not claim uniform W superiority. Baseline
backend provenance caveat remains. Seed42 guided this development follow-up.
No test, Ede, new seeds or A100 launched; queue exited normally. Full small
summary: outputs/experiments/amis_consistency_strength_w_replication_rtx_suite/SUMMARY.json.

## W/B strength replication — QUEUE RUNNING (2026-11-13)

After seed42 W improved, author requested continuing. Four new runs only:
W_S43/W_S44 (beta1.25) and B_S43/B_S44 (beta2.5), Amis/mBART, same fixed
FP32 SDPA B4 GA32 ES10 max20 validation-BLEU recipe. No test or transfer.
Detached suite PID **1830468**, both remote CPU gates PASS (10 tests total),
first GPU gate observed at step0/2. All four GPU gates precede long training.
Public HF canonical seed43/44 paths; verified backup before weight cleanup.
Baseline43/44 reused after full-trajectory audit: BLEU and chrF++ select the same
saved checkpoints, full20epochs, matching runtime/data/optimizer. Historical SDPA
backend evidence is reconstructed, not a contemporaneous receipt. No baseline
rewrites or retraining. Details/gates: `docs/research/CONSISTENCY_STRENGTH_REPLICATION.md`.
W is a reduced-strength R-Drop control, not a new ACL-method novelty claim.

## G/R/W COMPLETE — validation seed42, NO-GO for split components (2026-11-13)

All three runs completed 20 epochs / 720 steps, not early stopping. All best
checkpoints uploaded to FiveC/amis-rewire-checkpoints with verified hashes.
Queue completed and GPU idle. No test was run.

| Recipe | Validation BLEU | chrF++ |
|---|---:|---:|
| Baseline reused | 17.6823 | 14.7030 |
| B full consistency, beta2.5 | 17.9107 | 14.8427 |
| G gold-only, beta2.5 | 16.8007 | 14.1567 |
| R non-gold conditional, beta2.5 | 17.2483 | 14.4396 |
| W full consistency, beta1.25 | **18.2724** | **15.1728** |

G/R fail the predeclared gate; do not automatically replicate them. W gains
+0.5901 BLEU / +0.4698 chrF++ over baseline and about +0.3617 BLEU over B.
This supports investigating consistency strength, not a novel gold/rest split.
Single-seed validation development evidence only; no statistical or test claim.
Further training is not automatically authorized by this result.

## G/R/W implementation status — CPU PASS; no new model evidence (2026-11-13)

The three engineering ablations in [the plan](POST_PAIRED_ROUTE_METHOD_PLAN.md)
are implemented in `src/methods/decoupled_consistency/v1/` and integrated with the
shared args/config runner. Five tests pass: full-KL decomposition and both logit
gradients, extreme/large vocabulary, dropout/checkpoint/accumulation replay,
native inference/persistence, and frozen-probe gradient statistics. Five original
paired-route regression tests also pass. Algebra and implementation correctness
are not evidence of improved translation quality or ACL novelty.

No frozen checkpoint diagnostics, GPU gates or training have run. RTX disk has
only 3.89 GiB free; admission requires 40 GiB before restore/gates. The model
comparison and winner gate remain prospective. Existing A-D and baseline scores
are unchanged. See `outputs/experiments/amis_decoupled_consistency_rtx/IMPLEMENTATION_STATUS.json`.

## Post paired-route error audit — COMPLETE CPU / DESIGN ONLY (2026-11-13)

See [method selection and bounded pilot plan](POST_PAIRED_ROUTE_METHOD_PLAN.md).
Replayed all4 arms against aligned576 validation rows; BLEU/chrF++ match saved
scores within1e-10. B−baseline=+0.2284/+0.1397, sentence chrF wins/losses241/230;
Han output/reference ratio0.9352 vs baseline0.9526. These are descriptive,
not a demonstrated causal bottleneck or fresh-seed confirmation.
R-Drop/LR-Drop/DKD/MTO/ConsistTL prior reviewed; no certified new ACL method.
Two engineering ablations of gold-vs-rest / remaining-vocabulary consistency
are formulated with matched A/B controls. KL chain-rule and two-sided gradient
CPU gate PASS4cases; no new trainer implementation, GPU gate or training.

## Paired-route implementation — COMPLETE / NO-GO (2026-11-10)

Four approved Amis/mBART seed42 arms A–D deployed via shared runner:
`configs/experiments/amis_paired_route_rtx.json`.
Five CPU tests and all4 remote GPU gates PASS, including checkpointed
dropout/accumulation gradients, both-argument KL derivatives, save/reload,
generation-cache and optimizer/scheduler/RNG resume equivalence. All4 completed
20epochs/720steps; queue PID478144 exited normally. Final validation BLEU/chrF++:
A16.8619/14.2623, B17.9107/14.8427, C16.6620/14.0011,
D17.3386/14.3730. D−B=-0.5721 BLEU/-0.4697 chrF++; D−baseline=-0.3437/-0.3300.
The predeclared primary gate fails: NO-GO for replication/transfer/test of this
recipe. B's+0.2284 BLEU vs reused baseline is an exploratory consistency signal,
not evidence of CLRR benefit. No significance claim from validation seed42.
All4 public backups verified; CPU replay576 predictions/arm matches metrics,
12 small raw public artifacts SHA256-verified. No checkpoint weights downloaded.
Evidence in
output `outputs/experiments/amis_paired_route_rtx`.
Original numerical/data protocol lock preserved. No LSR/test/replication/transfer.
New implementation is isolated under `src/methods/paired_route/v1/`.
See `../experiments/EXPERIMENTS_RTX6000.md` for launch and HF storage details.
The following review-only entry records the earlier design stage, now superseded
by this authorized implementation; its existing evidence remains unchanged.

## Current next-design decision — REVIEW ONLY (2026-11-10)

Evidence synthesis and related-work audit:
[CLRR recovery design](CLRR_RECOVERY_DESIGN.md).
Fresh CPU inventory contains66 completed local validation receipts/3548 history
entries (not66 independent models or an exhaustive HF inventory).
Original cosine LSR-only17.2418 BLEU remains below reused baseline17.6823;
learned-strength Full improves only+0.1041 BLEU over LSR-only in seed42.
Together with failed fresh-seed confirmation, this does not justify another
unrestricted strength/lambda/layer sweep.

Proposed engineering direction: shared-weight native/CLRR paired supervised
training, with a four-arm factorial controlling for two-pass CE and R-Drop-style
output consistency. Original distance2/strength0.1 detached-donor rewiring stays;
LSR is omitted initially to isolate the training-objective hypothesis.
Two-view KL and branch regularization are established prior art, not a newly
certified contribution. Existing paired-CE prototype has a checkpointing
implementation difference and is not approved for unchanged launch.
Status: formula/controls/prospective thresholds documented; **no new method code,
GPU gates or training launched**. Initial4 seed42 runs are proposed only; no
automatic test, replication, Ê Đê transfer or A100 expansion.

## Original cosine LSR-only control — COMPLETE (2026-11-10)

LSR_COSINE_S42 finished20epochs/720steps (epoch cap, not early stopping),
selected epoch19/step684 by validation BLEU. CPU/GPU gates PASS. Public HF
backup hash-verified; queue exited, no remaining gate/training weights found.
Only small evidence files collected locally; no test opened.

| Model | Validation BLEU | Validation chrF++ |
|---|---:|---:|
| Reused baseline | 17.6823 | 14.7030 |
| Original cosine LSR-only | 17.2418 | 14.4559 |
| P1 learned CLRR + CE | 16.6360 | 14.1009 |
| P2 learned CLRR + original LSR | 17.3459 | 14.4554 |

LSR-only minus baseline: -0.4405 BLEU/-0.2471 chrF++.
P2 minus LSR-only: +0.1041 BLEU/-0.0005 chrF++.
The matched control does not show a substantial CLRR addition in this seed;
P2 remains below baseline. No significance or interaction claim follows from
one exploratory validation seed. Keep learned-strength expansion NO-GO;
do not automatically open test, more seeds, or parameter sweeps.
Recipe: `configs/experiments/amis_original_lsr_control_rtx.json`.
Evidence: `outputs/experiments/amis_original_lsr_control_rtx/runs/LSR_COSINE_S42/`.

## Original cosine LSR-only control — QUEUE LAUNCHED (2026-11-10)

Author approved the single missing matched control after checkpoint diagnosis.
Recipe: `configs/experiments/amis_original_lsr_control_rtx.json`.
Run `LSR_COSINE_S42`: Amis/mBART, original stop-gradient cosine LSR lambda=.1,
no CLRR, seed/data_seed42. Data/runtime/protocol lock and critical numerical
source hashes match the completed learned-strength cohort. FP32/SDPA,
b4/ga32, LR5e-5, ES10/max20, original-reference validation BLEU beam1.
CPU gate PASS locally and on RTX; exact GPU gate PASS (2 optimizer steps,
save/reload and Trainer best-checkpoint reload verified, peak11.99GiB).
Detached queue PID426253 alive and automatically proceeds to long training. No test or replication queued.
Prelaunch GPU free23952MiB; disk free70.89GiB; project lock free.
Bundle SHA256 `e8496913aa7119bc730e78600c411f50cbb35b4908df20a772feb78217bd9138`.
Public HF destination:
`checkpoints/amis_mandarin/mbart-large-50/lsr/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`
in `FiveC/amis-rewire-checkpoints`. Worker uploads/validates backup hashes
before removing trained weights. Results pending; compare validation with
baseline and learned-strength P2, retaining exploratory one-seed scope.

## Follow-up diagnosis COMPLETE (2026-11-10)

Selected checkpoint probes (64 shared validation rows, zero optimizer steps)
found weak routes: mean injection/recipient norm P1 **0.24125%**, P2
**0.17294%**. Route OFF-minus-ON CE = +0.00115966 / +0.00006713 nats/token;
both exploratory sentence-bootstrap 95% CIs include zero. Scalars have finite
nonzero gradients and appear once in the reconstructed optimizer; no defect
was identified by these checks. This does not prove a learning-rate cause.
Inventory did not identify a metadata-certified original cosine LSR-only
control matching RTX FP32/b4/ga32/ES10/BLEU selection. Existing ordinary-lsr
RTX runs use InfoNCE and cannot substitute. Next missing comparison is one
matched original LSR-only seed42 control; **not launched**. No test opened,
old scores unchanged, temporary diagnostic weights cleaned, queue exited.
Full scope, caveats, coefficients and provenance:
[Learned-strength diagnosis](LEARNED_STRENGTH_DIAGNOSIS.md).

## Current finding — learned-strength pilot COMPLETE / NO-GO (2026-11-10)

Both Amis seed42 arms finished20epochs/720steps, both GPU gates PASS,
both best checkpoints SHA-verified on public HF and remote weights cleaned.
Original-reference CPU validation replay576/576 matched recorded metrics.
Baseline BLEU/chrF++17.6823/14.7030; learned CLRR CE16.6360/14.1009;
learned CLRR + original LSR17.3459/14.4554. Best epochs18/19 respectively.
LSR arm gains+.7099 BLEU over learned CE, but remains-.3364 BLEU/-.2476
chrF++ below baseline. No original LSR-only matched arm here, so this does
not establish CLRR-LSR interaction. Scalar adaptation alone did not rescue
this recipe under the prospective gate. Stop expansion; no test/replication
or Ede/A100 opened. These are exploratory one-seed validation results, not
a significance claim. No further scalar/layer/lambda sweep authorized.
See `outputs/experiments/amis_learned_strength_rtx/VALIDATION_SCREEN.json` and
[RTX experiments](../experiments/EXPERIMENTS_RTX6000.md).

### Design provenance

The author approved two Amis seed42 engineering pilots after the Transition
NO-GO: CE with zero-init learned CLRR edge scalars, and the same route with
original LSR(.1). This tests fixed-strength assumptions; it does not establish
that fixed strength caused prior failures. ReZero/gated shortcut priors prevent
claiming generic learned scalars as standalone ACL novelty.
CPU4/4 PASS; RTX queue PID379839 launched to run BOTH GPU gates before any
long training. No new validation result yet. Baseline data/runtime/protocol
lock retained; ES10/max20, validation BLEU beam1, FP32/SDPA/B4/GA32/LR5e-5.
The prior Transition expansion HOLD and its scores remain unchanged.
Design, prospective development gates and reproducible commands:
[LEARNED_STRENGTH_PILOT.md](LEARNED_STRENGTH_PILOT.md).

## Previous result — Transition LSR engineering pilot COMPLETE (2026-11-10)

Follow-up audit **COMPLETE (2026-11-10)**: 63 local validation receipts, five
20-epoch trajectories, five pinned checkpoint geometry/gradient probes (zero
optimizer steps), and 576-row validation strata/leave-one-out replay PASS.
Transition Full edge cosine matched/shuffled = 0.983797/0.981994; baseline =
0.968193/0.964215. Endpoint centered cosine: 0.245356 versus baseline 0.275199.
No severe endpoint rank collapse shown in this 64-example probe; no evidence
here for auxiliary gradient domination. Route dependence remains, without
baseline superiority. No additional training/test/replication admitted.
Details and next-stage gates: [TRANSITION_FOLLOWUP_DIAGNOSIS.md](TRANSITION_FOLLOWUP_DIAGNOSIS.md).


Both Amis seed42 runs completed 20 epochs / 720 steps (epoch cap, not early
stopping). CPU5/5 and both GPU gates PASS. Protocol remained FP32/SDPA,
B4/GA32/LR5e-5/ES10/max20, original-reference validation BLEU selection beam1.

| Model | Validation BLEU | Validation chrF++ | Selected epoch |
|---|---:|---:|---:|
| Reused baseline | 17.6823 | 14.7030 | 19 |
| Transition LSR-only | 17.0833 | 14.2826 | 20 |
| Original CLRR + Transition LSR | 17.5220 | 14.6387 | 17 |

Full minus Transition LSR: +0.4387 BLEU; Full minus baseline: -0.1604 BLEU
and -0.0643 chrF++. Advancement gate FAIL: do not open replication, test,
Ede or A100 from this pilot. These are exploratory single-seed validation
results, not evidence of statistical significance or a test-set gain.

Both checkpoints have SHA-verified HF backups in `FiveC/amis-rewire-checkpoints`,
under `checkpoints/amis_mandarin/mbart-large-50/{transition-lsr-v1,transition-full-v1}/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`.
Backup revisions: LSR `4282e56df20c4fbe28da93a6322e208597fa9eb9`,
Full `33ea4df8bcb503132feb0249cf0bf87cf7a60686`.
Queue completed; GPU observed idle. Only small artifacts retrieved locally.
Raw metadata/prediction bytes were independently SHA-verified, parsed metadata
matched monitor copies, and both metrics recomputed exactly against original CSV.
Monitor text reserialization changes byte hashes; this was a transfer-format
issue, not changed predictions or scores.

Receipt: `outputs/experiments/amis_transition_lsr_rtx/PILOT_SUMMARY.json`.
Author-approved bounded engineering pilot superseded the earlier execution HOLD;
FDD overlap and historical novelty assessment below remain historical evidence.

---

## Current novelty verdict — Transition LSR v1: NO-GO

Focused review identified FDD (ACL2025), Section3.3 Eq7: cosine alignment
of changes across selected depths. The proposed bilingual pooled-state variant
is not identical, but the existing evidence does not establish a sufficiently
substantive new contribution. Provisional design admission is superseded.
See `docs/research/TRANSITION_LSR_NOVELTY_VERDICT.md` for exact differences.
No GPU gate or training started. CPU5/5 PASS is technical evidence only.
Preserve frozen code/recipe and historical results; no automatic retry.

---

## Current candidate — Transition LSR v1: PREPARED, LAUNCH BLOCKED (2026-11-09)

Author requested implementing the two-arm proposal and an RTX queue. Design and
primary-prior comparison are in `docs/research/TRANSITION_LSR_DESIGN.md`.
Code lives separately in `src/methods/transition_lsr/v1/`; frozen recipe:
`configs/experiments/amis_transition_lsr_rtx.json`.

- Amis/mBART seed42: TRANSITION_LSR_S42 then TRANSITION_FULL_S42.
- CPU scientific gate: 5/5 PASS, both cases PASS; original method/data untouched.
- FP32/SDPA/B4/GA32/LR5e-5/ES10/max20, validation BLEU beam1; no test stage.
- RTX read-only check: 23952/24576MiB free, GPU0%, project lock free, disk71GiB.
- **Automatic approval review REJECTED launch. No remote GPU gate or training
  was started.** Publication-level novelty remains unresolved; do not turn
  `novelty_certified` true or retry via another transport to bypass rejection.
- Full automatic queue command is prepared, but awaits an explicit author
  decision on admitting this bounded exploratory pilot, or stronger novelty
  evidence/redesign satisfying the existing gate. No automatic retry.
- Planned public paths (NOT uploaded):
  `checkpoints/amis_mandarin/mbart-large-50/{transition-lsr-v1,transition-full-v1}/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu`.
  Existing runner verifies SHA backup before deleting transient weights.
- Receipts: `outputs/experiments/amis_transition_lsr_rtx/CPU_GATE.json`,
  `LAUNCH_BLOCKED.json`. No new translation scores exist.

Earlier STOPPED banner below concerns relational-v2; that queue stays stopped.

---

## CURRENT STATUS — STOPPED FOR NOVELTY REVIEW (2026-11-09)

Author corrected the required sequence: design → related-work/novelty review →
only if sufficiently justified, smoke/correctness gates → only after all gates
PASS, long training. Translation quality cannot substitute for novelty review.
Do not run smoke, resume or launch a candidate whose novelty is UNESTABLISHED/HOLD.
A focused prior list or different implementation alone does not clear this gate.

Relational v2 queue PID170990 was stopped; all five queue processes exited.
Only A had begun long training. Existing A checkpoint-36 is preserved; B/C/D
have no long-training results. This interrupted run is not a completed result.
No other user's process was signalled. Previous launch/PASS entries below are
historical; they do not authorize resumption. Current candidate is NOT admitted.

Before proposing a candidate again, document primary-paper method sections,
formula/mechanism overlap, the concrete remaining contribution, evidence and
uncertainty. If the distinction is insufficient, reject/redesign before GPU use.

---

## Relational v2 — authorized four-arm screen, 09/11/2026

Đã triển khai riêng v2: adapted RKD-D và difference-direction cosine × CLRR
off/on, Amis seed42. Không sửa CSV, không dùng semantic minimal-pair labels,
không thêm tham số; CE + .1 auxiliary, SDPA/FP32/B4/GA32/ES10/max20,
chọn checkpoint theo validation BLEU beam1. Novelty UNESTABLISHED; đây là
pilot cơ chế có đối chứng, chưa phải kết luận phương pháp đủ tốt cho ACL.

CPU wrapper checks PASS (4arms); geometry8tests có sẵn. Bốn real GPU gates đã PASS hết trước training dài; queue PID170990
đã vào A_S42 (quan sát step8/720), tiếp theo B→C→D. Chưa có số chất lượng mới. Chi tiết công thức, tests, config, baseline
reuse và lưu checkpoint: `DIFFERENCE_PRESERVING_PILOT.md`, mục18.
Baseline lịch sử đủ20epochs có cùng best BLEU/chrF++ tại epoch19/step684:
17.682329 BLEU,14.703005 chrF++. Giữ nguyên nhãn cũ, lưu audit riêng;
backend SDPA dựa trên tái dựng stack/default, không giả thành runtime receipt.
Không mở test/Ede/A100. Trạng thái launch/gate/train xem receipts thực tế.

---

## Relational design screen — CPU only, 09/11/2026

Đã đọc lại method sections RKD/CCPL và mở rộng đối chiếu HRKD. Không nhận
generic relational/difference alignment là novelty mới. Prototype loss riêng
`src/methods/difference_preserving/v2/geometry.py`: direct difference-direction
và adapted RKD-D, all unordered pairs trong microbatch B4;8CPU tests PASS.
Đây chưa phải wrapper train, chưa có GPU gate hoặc quality results.

Phát hiện thiết kế cần tránh: RKD-D chuẩn hóa trên đúng1pair luôn cho1vs1,
loss0; không dùng đối chứng suy biến đó. Cả2objectives vẫn có nghiệm rank1
chung và co nhỏ cho loss0: không claim chống collapse. Không suy từ phản ví
dụ tổng hợp rằng real checkpoint đã collapse. Train ranking tốt cũng không
bác bỏ generalization benefit của regularizer chưa train.

Đề xuất bounded4arms Amis seed42: distance/direction × CLRRoff/on, toàn bộ
train anchors và pair eligibility theo content IDs, không gắn nhãn minimal
pairs hoặc sửa data. Chưa tích hợp runner/launch. Chỉ reuse controls cùng
BLEU-selector/protocol đã audit; không dùng historical chrF++ checkpoint thay.
Formulas, prior links, degeneracy policies, fairness và advancement targets ở
`DIFFERENCE_PRESERVING_PILOT.md`, mục17. V1 near-pair recipes giữ nguyên.

## Route-token interventions — RTX COMPLETE 09/11/2026

Đã forward-only trên đúng P1 CLRR-only và R2 Full cũ seed42;576validation
câu,5494content positions/model,4conditions: native/all-off/lower-off3–7/
upper-off8–12. Baseline R1 tái sử dụng token tables đã pin/hash. Không train,
không test, không reselect checkpoint. Tắt cạnh cộng CLRR, không bỏ block.

| Model | Tắt toàn bộ: sửa / hỏng | Tắt3–7: sửa / hỏng | Tắt8–12: sửa / hỏng |
|---|---:|---:|---:|
| CLRR-only |183 /327|120 /165|114 /137|
| Full |175 /341|123 /166|94 /132|

So với native cùng weights, CẢ6can thiệp có net top1 và mean margin âm.
Native so baseline vẫn sửa205/hỏng265 ở CLRR và sửa227/hỏng272 ở Full.
Route giúp checkpoint đã co-adapt; điều đó chưa làm CLRR thắng baseline.
Không đủ cơ sở chọn pruning nhóm route cố định hoặc tăng lambda làm repair.
Token oracle chọn gold-dependent giữa3models có3092/5494đúng, baseline2772;
chỉ320baseline-wrong được ít nhất1route model sửa;2402positions sai ở cả3.
Trong2402positions này, baseline gold-rank <=5:741;6–50:685;51–500:510;
>500:466. Không suy từ rank rằng mọi alternative đều sai nghĩa hoặc donor
không chứa thông tin. Oracle không phải router chạy được khi inference.

Verification:2CPUgates PASS; all-off=bare và layer1negative-control errors0.
Native B4 replay NLL/margin khớp <=3.6e-15; all correct flags khớp bảng cũ.
Lượt đầu dùngB2 đã dừng vì replay lệch3.8e-4; sửa về đúngB4cũ, giữ nguyên
tolerance1e-4, lưu failed logs. Runner đã đổi nên dùng đúng historical
configure_languages function từ source SHA-verified; không rollback code train.
Warning Mistral regex không dẫn tới đổi tokenizer: toàn bộ token hash khớp.
Independent CPU replay và10000sentence-bootstrap draws đã xong; CI mô tả,
không hiệu chỉnh và không phải significance qua training seeds. Xem REPORT.

**Quyết định:** kết thúc vòng chẩn đoán pruning/utility này; không mở lại
cùng gate với tên khác. Chưa có method mới được chứng minh hoặc novelty GO.
Hướng tiếp theo cần tác động vào khả năng đưa gold content lên top1, có đối
chứng mạnh và bằng chứng generation; không dùng một oracle routing ceiling
làm lời hứa về gain. Generic selective-KD/gates/contrastive logits/route
consistency đã có prior gần; từng thiết kế đầy đủ phải được đối chiếu riêng,
không kết luận mọi tổ hợp các thành phần cũ đều không thể có đóng góp mới.

Artifacts `outputs/experiments/amis_route_token_interventions/`: PLAN,
per-token CSV, ANALYSIS và REPORT.15files nhỏ,~4.1MB đã thu về và SHA verify;
temporary weights trên RTX đã xóa,queue kết thúc,GPU23952MiB free/0%.
Không thay numbers ở A100/EXPERIMENTS_UPDATED hoặc paper.

## Validation qua ba seed — CPU verified 09/11/2026

Tái tính6outputs baseline/C0,576câu validation, nguyên references/beam1/FP32;
hashes và18corpus scores khớp receipts. Không GPU/train/test mới.
86câu nằm trong144câu chrF0 thấp nhất ở CẢ3baseline seeds42–44;139câu ở ít
nhất2seeds. Đây là xếp hạng theo reference, không phải86lỗi nghĩa đã xác minh.
C0 tăng sentence chrF0 cả3seeds ở12/86câu, giảm cả3 ở9/86; không sửa đồng
loạt nhóm khó. Chỉ1/86câu baseline sinh cùng một chuỗi ở cả3seeds.

Join đúng checkpoint với source-grounding audit cũ:84/86câu vẫn có native
source gap dương, mean2.703416nats/supervised token. Negatives cyclic không
phải hard semantic contrasts; không suy ra grounding đã hoàn hảo hoặc thiếu
grounding là nguyên nhân. Mean sentence chrF0 delta không phải corpus delta;
nhóm chọn theo baseline thấp bị regression-to-mean, không claim selective repair.

**Quyết định:** chưa mở loss/train mới. Bước kế là MỘT token-level route
intervention validation trên frozen checkpoints để phân biệt donor sửa/làm
hỏng reference decisions, theo contract trong báo cáo. Nếu không có tín hiệu
repair nhất quán, dừng nhánh sửa CLRR này; không lặp sweep alignment/gating.
Selective donor chỉ là hướng có điều kiện, chưa novelty GO. Semantics cần
review song ngữ, không được tự tạo nhãn chuyên gia.

Artifacts/report: `outputs/experiments/amis_validation_seed_consensus/`.
Replay: `scripts/analysis/analyze_validation_seed_consensus.py`.
Không sửa bảng baseline, selector hoặc số thí nghiệm cũ.

## Baseline train contrasts — đã đo RTX 09/11/2026

Đã chạy forward-only trên baseline seed42 của hai dataset, checkpoint public
pin revision40fac65856b883e999bee74a45075d9c9d5dc7a5, SHA archive khớp receipts.
Cả hai là snapshot chọn chrF++ lịch sử; không thay baseline BLEU cho pilot.
RTX FP32/SDPA/TF32off, chỉ train CSV; không train/generate/đọc validation/test.

| Dataset | Near pairs | Ưu tiên source đúng ở cả hai chiều | Mean symmetric margin (nats/content token) | Generic mean | Cặp có reference-choice margin không dương ở ít nhất một chiều |
|---|---:|---:|---:|---:|---:|
| Amis | 165 | 165/165 | 1.9214 | 6.4709 (193cặp) | 1 |
| Ê Đê | 2679 | 2679/2679 | 2.8923 | 8.2560 (1461cặp) | 3 |

Source margin giữ nguyên target rồi đổi source; reference-choice giữ source
rồi đổi target, hai phép đo khác nhau. Không được gọi bảng này là100% dịch đúng.
Một cặp Amis chứa reference nghi bị cắt; ba cặp ÊĐê khác wording/pronoun;
chưa được review song ngữ. Train memorization có thể giải thích kết quả.
Không có p-value/iid CI cho các graph edges phụ thuộc và chỉ một seed.

**Insight:** train near-pairs khó hơn generic nhưng chưa bộc lộ lỗi phân biệt
source cần sửa bằng loss difference-preserving. Điều này làm yếu lý do mở pilot
hiện tại, không chứng minh mọi relational loss vô ích hay phương pháp mới chắc
chắn thất bại. Tiếp theo nên kiểm tra lỗi generation VALIDATION có sẵn của
baseline trước khi thiết kế repair. Không tự mở thêm training hoặc đo test.

CSV/formulas/provenance/verification và4reference-choice cases ở
`outputs/experiments/difference_preserving_feasibility/baseline_train_contrasts/`.
Chi tiết mục16 trong `DIFFERENCE_PRESERVING_PILOT.md`. Chỉ tải JSON/CSV nhỏ về PC,
SHA đã khớp; weights restore tạm đã xóa. Queue hoàn tất, GPU23952MiB free/0%.

## Difference pilot — code đã sửa 09/11/2026

Unicode miner đã sửa giữ combining marks; audit mới Amis165cặp/193dòng, ÊĐê
2679cặp/1461dòng, conditional selected edges109 và1044. Đã sửa generic control
loại TẤT CẢ reviewed neighbors, tránh lấy lại contrast khác làm generic peer.
19CPUtests PASS, gồm Unicode và multi-neighbor regression. Audit mới ở
`outputs/experiments/difference_preserving_feasibility/unicode_fix_v2/`.
Không đổi CSV, loss, baseline/protocol; không chạy RTX. Bundle/gate cũ gắn với
source cũ, không được dùng receipt đó để chứng nhận source vừa sửa. Semantic
review/novelty vẫn chưa đủ; xem mục15 của kế hoạch.

## Difference pilot — quyết định khoa học 09/11/2026

**NO-GO cho queue train dài R0–R3 hiện tại dưới vai trò ứng viên phương pháp
chính;8/8 technical PASS vẫn giữ nguyên.** Đây là quyết định về độ sẵn sàng
thiết kế, không phải kết quả train thất bại. Audit lại primary methods
RKD/CCPL/DiffCSE/CL2CM: có khác biệt triển khai, chưa đủ cơ sở về đóng góp chính
ACL; không có chứng nhận novelty tuyệt đối hoặc xác nhận trùng toàn bộ công thức.

Đọc cả hai mẫu100cặp và kiểm tra lại code/train bằng conda clrr. Số cặp song ngữ
được xác minh vẫn0. Phát hiện lỗi miner ÊĐê: regex tách dấu Unicode kết hợp;
4277/15092 dòng khác chuỗi token khi giữ dấu trong từ.84cặp cũ không đạt quy
tắc độ dài/thay từ theo audit thay thế; tổng2679 thay vì2460 chỉ là số chẩn đoán,
chưa áp dụng. CSV, tokenizer và số baseline cũ không bị sửa/ảnh hưởng; chưa có
training dài dùng miner này. Amis không đổi165cặp trong kiểm tra Unicode này.

Nếu toàn bộ cặp cũ được chấp nhận, quy tắc chọn peer nhỏ nhất theo row index
chỉ sử dụng109/165 cạnh Amis và1031/2460 cạnh ÊĐê. Đây là giả lập đường code,
không phải độ phủ ngữ nghĩa thực tế. Mẫu reference có gần đồng nghĩa, thay đổi
nhiều chi tiết và nghi vấn thiếu đoạn; không tự gán nhãn đúng/sai song ngữ.

Tiếp theo: sửa miner và audit mới; xác minh correspondence; chẩn đoán lỗi dịch
cụ thể của baseline trước khi chọn loss/candidate. Pilot heuristic nếu có phải
có contract riêng, không giả PASS cổng semantic. Chi tiết/bảng prior/kế hoạch:
[mục14](DIFFERENCE_PRESERVING_PILOT.md). JSON nhỏ:
`outputs/experiments/difference_preserving_feasibility/novelty_decision_20261109/`.
Đợt review này không dùng GPU/test, không đổi code/recipe hoặc mở queue.

## Difference pilot — RTX technical-only hoàn tất 09/11/2026

RTX đã online; tác giả cho phép kiểm thử kỹ thuật. Suite riêng PID11153 gồm
R0–R3 Amis và ÊĐê, không có lệnh train dài/test. **8/8case PASS** và queue đã
kết thúc. Receipts đã verify theo bundle manifest:2steps, B4/GA32, FP32/SDPA,
checkpointing, save/reload, beam4cache và resume optimizer/scheduler/RNG đạt
tolerance cũ. Peak allocated14.41GiB Amis/15.16GiB ÊĐê. Không còn checkpoint
tạm; GPU về0%,23952MiB trống. Fixture dùng cyclic train peers,
không giả nhận cặp được review. Scientific gates giữHOLD; technical receipt
không thay thế training admission. Xem mục13 trong
[kế hoạch](DIFFERENCE_PRESERVING_PILOT.md). Báo cáo:
`outputs/experiments/amis_difference_preserving_gpu_technical_suite/REPORT.md`.
Tokenizer audit: tokenIDs toàn bộ train hai dataset khớp sau reload; warning
Mistral không dẫn tới đổi regex. Không có số chất lượng dịch mới.

## Difference pilot — chuẩn bị local, chưa chạy RTX 07/11/2026

Theo yêu cầu mới nhất: chưa kết nối/launch/chạy RTX. Đã audit cấu trúc toàn bộ
165cặp Amis và2460cặp ÊĐê; tương ứng69 và1457cặp có một đoạn ký tự đích thay đổi.
Đây không phải nhãn khác biệt nghĩa; số cặp được xác minh song ngữ vẫn0.
Đã đọc thêm công thức CL2CM từ fulltext arXivv2; relational cross-lingual
transfer là prior, chưa đủ căn cứ chốt prototype thành đóng góp chính ACL.

23tests CPU PASS:15method,5metrics,3selector. Đã freeze hai config
`{amis,rhade}_difference_preserving_local_review.json` và verify manifest.
Đã khóa practical pilot gates trước training: R3−R2 và R3−baseline >=0.30BLEU,
R3−R1 >0, chrF++ không giảm ở các so sánh đó; thiếu baseline cùng protocol thì
HOLD. Đây là tiêu chí development, không phải kiểm định ý nghĩa thống kê.
Không đổi recipe baseline, không sửa dataset, không có điểm chất lượng mới.

Chi tiết công thức/gradient, prior, artifacts và điều kiện còn thiếu:
[mục12 của kế hoạch](DIFFERENCE_PRESERVING_PILOT.md#12-local-preparation-completed-rtx-explicitly-deferred--2026-11-07).
Receipt: `outputs/experiments/difference_preserving_feasibility/LOCAL_PREPARATION.json`.

## CLRR/LSR foundation — verified gradient synthesis 07/11/2026

Đã đọc lại phần phương pháp trong `53-2 (1).pdf`; giữ câu hỏi CLRR đưa tín hiệu
tầng sớm và LSR hướng representation theo bản dịch. Đã verify SHA256 của12
diagnostics cũ (baseline/NCE/P2/P3 ×3seeds), replay norm/cosine và cùng4batches.
Không train hay đo GPU mới. P2/P3 có LSR thật: CE–LSR cosine mean theo checkpoint
0.0749–0.1232, không âm ở24/24batch probes. Chưa có bằng chứng xung đột gradient
phổ biến ở các checkpoint này; không lấy norm nhỏ làm lý do tăng lambda.
Không suy ra Adam update hoặc cơ chế toàn quá trình từ4batches cuối.

MRL2026 đã có prior về alignment/task gradient compatibility; không nhận câu
chuyện chung đó là novelty mới. Prototype difference vẫn cần semantic-pair và
novelty gate, chưa mở training. Báo cáo:
`outputs/experiments/amis_contrastive_multiseed_evidence/lsr_compatibility/REPORT.md`.
Lộ trình và đối chiếu manuscript: [mục11](DIFFERENCE_PRESERVING_PILOT.md#11-grounding-in-the-authors-manuscript-and-verified-evidence--2026-11-07).

## Difference-preserving LSR — feasibility 07/11/2026

Đã triển khai prototype riêng, chưa train. Train-only lexical coverage:
Amis193/4600câu (165cặp), Ê Đê1461/15092câu (2460cặp); không tự xem là cặp
khác biệt nghĩa đã được xác minh. CPU correctness15/15 PASS (sau sửa metadata/resume); chưa có bằng chứng
chất lượng. RKD, CCPL và DiffCSE làm rõ relational/difference preservation đã có
prior; công thức này chưa đủ căn cứ nhận novelty. **Giữ scientific HOLD**, chưa
mở4pilot hoặc test; RTX hiện SSH502, VRAM chưa biết. Xem
[thiết kế và quyết định](DIFFERENCE_PRESERVING_PILOT.md). Số cũ giữ nguyên.

## DRB Amis — phân tích validation hoàn tất 06/11/2026

Đã verify hash predictions/JSON từ HF revision40fac65856b883e999bee74a45075d9c9d5dc7a5,
576câu và metrics khớp<1e-8; cả4run cùng recipe/runtime/bundle. SelectorBLEU
replay đúng, cả4train20epochs nên không phải dừng sớm làm mất checkpoint.

D3−D2: BLEU−0.4536, CI95%[−1.2668,+0.3607], p native0.1052, Holm3=0.3156.
chrF++−0.1030, CI[−0.5976,+0.3975], Holm3=0.4716. Pairedbootstrap10k,
seed10042; family3metrics, exploratory validation-only, không kiểm định xác nhận.
Không chứng minh querygate gây hại: D3 thắngD2 ở10/20epochs vàCIchứa0.
Không có predictionrỗng hoặc repetitionflag; meanlengthratioD2/D3~0.997;
strata không cho pattern giảm đơn điệu theo độ dài. Chưa xác định cơ chế nguyên nhân.

**NO-GO thăng hạng D3 theo practicalgate** vì BLEU thấp hơn cả3controls.
Không tự đưaD2 thành methodchính, không mởtest/seed/A100 hoặc sweep. GiữHFweights
và artifacts nhỏ. ÊĐê vẫnHOLD. Bước thiết kế sau phải dựa cơ chế/prior, không
quy lỗi chắc chắn cho detachment/gate/data từ các chỉ báo hiện có.

Báo cáo: `outputs/experiments/amis_decoder_residual_bridge_rtx_analysis/validation/VALIDATION_ANALYSIS.md`.

---

## DRB Amis — cả4runs đã train xong (xác minh HF ngày06/11/2026)

Kiểm tra repo public tại revision `40fac65856b883e999bee74a45075d9c9d5dc7a5`: cả4prefix có
best.zip, validation_metrics.json và stop_state.json. Cả4 hoàn tất20epochs/720steps;
không early-stop. Đây là validationbeam1, BLEUzh, chọn checkpoint theoBLEU.

| Run | BLEU | chrF++ | Best epoch | Completed epoch |
|---|---:|---:|---:|---:|
| D0_S42 | 16.7525 | 14.1727 | 19 | 20 |
| D1_S42 | 16.8277 | 14.1486 | 18 | 20 |
| D2_S42 | 17.1272 | 14.4005 | 20 | 20 |
| D3_S42 | 16.6736 | 14.2976 | 19 | 20 |

D2 cao nhất trong4arms; D3 không vượt các matched controls theo BLEU nên
không đạt gate so controls đã khóa. Baseline audit trước có17.682329BLEU;
chưa dùng đối chiếu đó làm kết luận cuối trước receipt/runtime replay. Không
mởtest hoặc trainbổsung. SSH RTX hiện lỗi502/timeout: chưa xác minh trực tiếp
queueexit, receipts SHAverification hoặc cleanup trênmáy. Sự tồn tại4archive
trênHF được xác nhận độc lập, không đồng nghĩa đã kiểm traSHAtoànbộ tại lượt này.
ÊĐê vẫnHOLD doresume gatefail; chưa cótraining ÊĐê trongqueue này.

Artifacts nhỏ: `outputs/experiments/amis_decoder_residual_bridge_rtx_training/hf_status.json`.

---

## DRB — mở train dài Amis ngày 06/11/2026

Theo yêu cầu tác giả cho train dài, tách admission theo dataset: **Amis có cả4
CPU/GPU gates PASS nên mở4runs D0→D1→D2→D3 seed42**; Ê Đê tiếp tục HOLD vì
resume gate chưa đạt. Đây là thay đổi thứ tự điều phối so với suite cũ yêu cầu
8gates trước mọi training; không đổi recipe, tolerance, dữ liệu hoặc score cũ.

Queue detached PID2001404, thư mục remote
`outputs/experiments/amis_decoder_residual_bridge_rtx_training/`.
VRAM lúc mở23540MiB free. Giữ nguyên FP32/SDPA, B4/GA32, LR5e-5,
validation BLEU zh/beam1, ES10, max20epochs. Không mở test/replication/A100.
Upload từng final archive vào prefixDRB đã khóa, SHA-verify rồi xóa weights tạm.

Ê Đê: weights liên tục so với resume sau2steps lệch tối đa4.0233e-7; loss
log trùng ở4chữ số, gradnorm lệch~9.54e-7. Đây là bằng chứng định lượng về
mức độ lệch, **chưa xác nhận nguyên nhân, không biến gate FAIL thành PASS**.
Không đổi backend/determinism hoặc nới tolerance để ép qua gate. Chưa train Ê Đê.

Launch receipt: `outputs/experiments/amis_decoder_residual_bridge_rtx_training/launch_rtx.json`.

---

## Cập nhật DRB 06/11/2026 — queue dừng tại GPU resume gate Ê Đê

Queue PID1886492 đã dừng; chưa mở training dài (0/8 models). Cả4 GPU gates
Amis PASS. Gate D0 Ê Đê thất bại khi so logits sau optimizer/scheduler/RNG resume
với chạy liên tục: 3/512 phần tử ngoài tolerance, max absolute difference
4.6133995056152344e-05; atol2e-05, rtol1e-05. Các gate D1–D3 Ê Đê chưa chạy.
GPU đã giải phóng:23540MiB free,util0% ở lần kiểm tra này.
Không thay tolerance/protocol, không tự đánh PASS, không suy đây là điểm chất
lượng phương pháp. Cần chẩn đoán tính tái lập resume trước khi mở lại queue.

---

## DRB v1 — triển khai 06/11/2026 (validation-only)

Tác giả duyệt tám pilots seed42: D0 local, D1 donor, D2 source-gated difference,
D3 decoder-query-gated difference, trên Amis rồi Ê Đê. Đây là CE-only full
fine-tuning với module196608parameters; không phải LSR hoặc PEFT.

- Code riêng: `src/methods/decoder_residual_bridge/v1/`.
- Config: `configs/experiments/{amis,rhade}_decoder_residual_bridge_rtx.json`.
- Hai bundle đã PASS CPU: 9 scientific tests và exporter/roundtrip cho cả bốn arms.
- RTX queue detached PID1886492 tại `outputs/experiments/amis_decoder_residual_bridge_rtx_suite/`.
- VRAM lúc admission23540MiB free; GPU gate D0 Amis đã vào bước optimizer.
- Tất cả8 GPU gates phải PASS trước bất kỳ training dài nào. Gate gồm2optimizer
  steps, exactB4/GA32, checkpoint/reload, beam4cache và optimizer/scheduler/RNGresume.
- Lock FP32/SDPA/TF32off, LR5e-5, max20epochs, ES10; **chọn best và early stopping
  bằng validation BLEU beam1**, zh cho Amis,13a cho Ê Đê. Không mở test/A100/seed khác.
- HF public `FiveC/amis-rewire-checkpoints/checkpoints/<dataset>/mbart-large-50/drb-<arm>-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu-rank64-donor6-dec12.../`.
  Upload và SHA-verify rồi mới xóa weights tạm; local chỉ giữ artifacts nhỏ.
- Baseline reuse: Amis đã có audit cùng BLEU-best epoch19, cần receipt runtime
  khi kết luận cuối. Ê Đê baseline cũ chọn chrF++ epoch17, BLEU-best epoch18
  (34.029051); **chưa được coi là baseline BLEU-selected**. Kiểm kê weights trước
  quyết định chạy bổ sung. Không tự mở baseline ngoài8runs.
- Chưa có điểm validation mới. Novelty và hiệu quả vẫn là giả thuyết cần kiểm chứng.

Chi tiết: `docs/research/DECODER_RESIDUAL_BRIDGE_PROPOSAL.md`.
Baseline audit: `outputs/audits/drb_baseline_reuse/REUSE_AUDIT.json`.

---

## Cập nhật 06/11/2026 — Ê Đê pilot và test đã hoàn tất

Cùng seed42, RTX FP32/SDPA, checkpoint chọn theo validation chrF++, beam1. Test beam4 trên 1.000 câu, BLEU 13a; original data và splits. Chỉ một seed, test đã tham gia development.

| Model | Validation BLEU | Validation chrF++ | Test BLEU | Test chrF++ |
|---|---:|---:|---:|---:|
| Baseline | 34.0232 | 50.4714 | 34.0402 | 50.0957 |
| C0 | 34.0381 | 50.5047 | 33.9203 | 50.0804 |
| LearnedDonor | 34.0729 | 50.4590 | 34.0843 | 50.1812 |
| M0 | 34.3728 | 50.8678 | 34.1761 | 50.1916 |

C0 không đạt gate validation đã định trước (+0.0149 BLEU so baseline; −0.0348 so learned donor; −0.3347 so M0), nên không tự mở replication. Test C0−baseline −0.1200 BLEU, CI95% [−1.0910,+0.5726]; paired bootstrap 10.000, p Holm9=1.0. Cả 9 phép so sánh chưa significant sau Holm. Không suy không có hiệu ứng hay đồng nhất mô hình; bootstrap câu không đo biến thiên giữa seed. Checkpoint và predictions public, báo cáo thống kê revision `2921e8b9dbc58e81f370265bea23d37a0d6a1941`.

### Đối chiếu trước khi thiết kế lại

- Full gốc trên Amis, 8 fresh seeds: Full−LSR −0.0198 BLEU, 3/8 wins, exact sign-flip p=.9297; Full−baseline −0.0415 BLEU, 4/8 wins, p=.9141. Nguồn `docs/experiments/EXPERIMENTS_UPDATED.md` §13; tập test đã có development exposure. Không suy hai phương pháp tương đương chỉ từ p lớn.
- C0 trên Amis, cohort BLEU-selected ba seeds: mean C0−baseline −0.0558 BLEU; +.9326/+.6175/−1.7175 theo seed. C0 validation có lợi ba seeds, test đổi dấu ở seed44. C0 Ede seed42 không đạt validation gate, test kém baseline .1200 BLEU; không gộp cohort, backbone, hardware hay seed.
- Probe Amis: tắt donor trong mô hình đã train làm validation BLEU giảm 2.98–3.31; tắt module khoảng −.060 đến +.010. Đây là độ nhạy checkpoint đã co-adapt, không chứng minh bỏ module khi train hay donor gây cải thiện so baseline.
- Validation Ê Đê: chrF0 tính theo từng câu, C0 so M0 có 311 câu điểm cao hơn, 307 thấp hơn, 382 bằng; so baseline là 315/301/384. Đây là điểm theo câu dùng mô tả, không đồng nhất chênh lệch chrF corpus và không xác nhận lỗi ngữ nghĩa. Toàn bộ kết quả cặp câu tại `outputs/experiments/rhade_residual_diagnostics/validation_paired_examples.csv`.
- Related work cần kiểm tra khi chốt phương pháp: [Lexical Shortcuts (WMT 2019)](https://aclanthology.org/W19-5211/) đã thêm gated shortcut từ embedding tới từng layer cho NMT; [SkipCLM](https://aclanthology.org/2025.naacl-srw.50/) kết hợp trainable skip và contrastive alignment; [NeuTRENO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fde1a69a5b6e554b2f1f727197d2651d-Abstract.html) dùng regularization dựa trên sai khác representation. Skip, gate, difference hay kết hợp alignment không tự đủ novelty. Đây là audit mục tiêu, chưa bảo đảm đã tìm đủ mọi prior.
- Probe Ê Đê đã hoàn tất; hướng pilot tiếp theo cần được xác định từ kết quả; cần công thức, hypothesized mechanism, matched controls, kiểm tra gradient/init/save-reload, protocol khóa, tiêu chí GO/NO-GO và chỉ dùng validation để phát triển. Giữ LSR là một giả thuyết cần đối chứng, không tự coi vô dụng hoặc có lợi.

### Audit optimization và cổng novelty — 06/11/2026

**Đã thực thi:** 12 checks CPU và cùng 12 checks CUDA RTX đều PASS; 12 tests kiến trúc/can thiệp PASS. Trước GPU audit có 23.540MiB trống; chỉ tiny synthetic model, không load checkpoint lớn, không train dataset. Worker `scripts/analysis/audit_residual_optimization.py` hỗ trợ `--device cpu|cuda --output ...`; JSON `OPTIMIZATION_CPU.json`, `OPTIMIZATION_GPU.json`, `CORRECTNESS_TESTS.json` trong workspace diagnostics. Không sửa định nghĩa C0 gốc.

Với C0, `z=RMSNorm(b-u)`, `A=W_up GELU(W_down z)`, `h=u+alpha*(b+A)`; `b` lấy từ updated donor cache và detach đúng định nghĩa. Ở init W_up=0, dL/dW_down=0 là hệ quả chain rule, không phải bug; dL/dW_up khác0 và sau một bước cập nhật W_up thì W_down nhận gradient. Sai phân hữu hạn khớp đạo hàm; alpha nhân local gradient đúng tỷ lệ; mọi up-layer và backbone nhận CE gradient; reentrant checkpointing khớp forward/gradients khi có dropout. Gradient scaling không dự đoán tỷ lệ Adam updates hay gain chất lượng. Các checks này không chứng minh optimizer trên full training tối ưu, không loại trừ mọi bug.

Receipts optimization đã public và xác minh 3 files tại HF revision `66b81ea5deda53dc2c76f86ac73bf843792f0854`; xem `OPTIMIZATION_HF_UPLOAD.json`.

**Hai hướng vừa sàng lọc, chưa triển khai training:**

| Hướng | Công thức định hướng | Đối chiếu prior | Quyết định |
|---|---|---|---|
| U1: donor orthogonal với current | `r=b-<b,u>u/(||u||²+eps); h=u+alpha*r` | Orthogonal Residual Update NeurIPS2025 đã dùng phép loại thành phần song song trước cộng residual. Đổi nguồn residual thành donor là khác thiết kế, chưa đủ chứng minh đóng góp đáng kể | HOLD làm method chính; chỉ có thể là prior-inspired comparator, không gọi là ý tưởng orthogonal mới |
| U2: gated early donor | `h=u+sigmoid(g(u,b))*b` | Lexical Shortcuts WMT2019 dùng gated embedding shortcuts cho NMT; SkipCLM2025 kết hợp trainable skip với contrastive alignment | HOLD novelty; thêm gate/early-layer hoặc ghép LSR tự nó chưa đủ |

Nguồn đã mở paper/method: [Orthogonal Residual Update, §3, Eq.3–4](https://papers.neurips.cc/paper_files/paper/2025/file/67c15da4a9340140c60783d9a175fd3f-Paper-Conference.pdf), [Lexical Shortcuts §2](https://aclanthology.org/W19-5211.pdf), [SkipCLM](https://aclanthology.org/2025.naacl-srw.50.pdf). Đây là targeted prior audit, không phải bảo đảm phủ hết literature và không chứng minh C0 gốc giống hệt các prior.

**Quyết định thực thi:** không mở training dài cho hai hướng trên khi novelty chưa qua gate đã yêu cầu. Không tăng alpha/rank hoặc bỏ LSR chỉ dựa norm. Bằng chứng hiện có đủ để dừng mở rộng C0 hiện tại, chưa đủ chọn một công thức mới có xác suất thành công cao. Hướng nghiên cứu tiếp theo phải nêu thông tin donor nào giúp dự đoán bản dịch, vì sao khác current và prior; ưu tiên counterfactual/representation evidence trước khi ghép thêm module. Chưa có candidate mới được khóa; số runs training mới trong đợt audit này = 0. Không gán PASS kỹ thuật thành PASS khoa học.

### Quyết định và bước phát triển tiếp theo — 06/11/2026

| Bước | Việc cụ thể | Điều kiện đi tiếp |
|---|---|---|
| 1 — hoàn tất | Đối chiếu Full gốc, C0 Amis, C0 Ê Đê và can thiệp donor/module | C0 hiện tại NO-GO mở rộng; giữ toàn bộ số lịch sử |
| 2 — audit trước thiết kế | Kiểm tra công thức/gradient: correction zero-init có nhận gradient đúng không, alpha có làm nhánh học quá yếu không, donor cache/detach có đúng không; dùng unit/tiny batch và receipts cũ | Có bằng chứng lỗi thì sửa lỗi trước; chưa gọi module nhỏ là lỗi tối ưu |
| 3 — chốt một giả thuyết | Xem xét thay reinjection toàn donor bằng phần thông tin donor chưa được current biểu diễn; đây chỉ là hướng nghiên cứu, chưa chốt công thức hay novelty | Đối chiếu trực tiếp prior projection/orthogonal residual, highway/gated shortcuts, adapters; nếu chỉ là prior đổi tên thì dừng |
| 4 — pilot có điều kiện | Chỉ sau card/formula và correctness gates: candidate seed42, đối chứng donor cố định cùng recipe và đối chứng capacity phù hợp; baseline cũ chỉ reuse khi metadata/protocol khớp | Khóa số runs trước launch; giữ data, SDPA/FP32, batch, LR, ES10, selector; validation-only, không dùng test để chọn |
| 5 — replication | Chỉ candidate qua practical validation gate và vượt đối chứng mới chạy thêm seeds | Không chuyển A100 để lấy số chính chỉ vì seed42 tăng; cần độ ổn định và audit novelty |

Chưa launch training mới. Chưa đủ bằng chứng để bỏ LSR toàn bộ, tăng alpha/rank, hoặc kết luận chỉ do test nhỏ. Giữ LSR làm đối chứng nếu candidate mới có alignment; không ghép LSR vào mọi thiết kế mặc định. Gate practical +0.30 BLEU trên validation có thể dùng làm ngưỡng sàng lọc, không phải ý nghĩa thống kê và không phải bảo đảm tăng test. Candidate card phải khóa ngưỡng theo metric/selector trước khi train; không đổi hậu nghiệm cho qua gate.

### Chẩn đoán frozen C0 Ê Đê — hoàn tất 06/11/2026

1.000 câu validation, seed42, checkpoint và protocol giữ nguyên; 0 optimizer steps. All-on khớp chính xác predictions validation gốc. Không sinh thêm test để chọn thiết kế.

| Can thiệp inference | BLEU | ΔBLEU so all-on | ΔchrF++ |
|---|---:|---:|---:|
| All-on | 34.0381 | 0 | 0 |
| Tắt module học | 34.0316 | −0.0065 | −0.0174 |
| Tắt donor | 28.8044 | −5.2337 | −4.5247 |
| Shuffle donor | 28.7713 | −5.2668 | −5.0632 |

Mean theo câu của tỷ lệ norm correction đã nhân alpha / current là 0.0204–0.0419% theo layer3–12; toàn residual / current là 7.82–8.30%. Đây là độ lớn activation, không phải tỷ lệ đóng góp chất lượng hoặc cập nhật optimizer.

**Kết luận:** module học có sensitivity inference rất nhỏ trên checkpoint này; donor tạo phụ thuộc sau training. Cả Amis và Ê Đê chưa chứng minh C0 vượt baseline ổn định. Không suy bỏ donor khi train sẽ tốt, module vô dụng trong training, hoặc nguyên nhân chắc chắn do data. Không mở replication/A100 cho C0 hiện tại.

Artifacts nhỏ: `outputs/experiments/rhade_residual_diagnostics/{SUMMARY.json,REPORT.md,gpu/}`. Weights khôi phục tạm đã dọn theo CLEANUP receipt. Public diagnostics: `FiveC/amis-rewire-checkpoints/analysis/rhade_vietnamese/mbart-large-50/alignment-residual-diagnostics/seed-42`, revision `de794fd0b53fbc51b3f263204b4c193777357271`, 9 files verified. Không tải weights về Windows.



---

## Trạng thái hiện hành — 31/10/2026: selector audit xong, chẩn đoán ba seed hoàn tất, mở pilot Ê Đê

BLEU-selected C0 test mean19.6748 vs baseline19.7306 (−.0558), chrF++ +3.1406 và chrF0 +.1776. Seed BLEU gaps +.9326/+.6175/−1.7175; chỉ seed44 thua baseline significant sau Holm45 (p=.026997). Không có bằng chứng C0 vượt baseline ổn định; đổi selector chưa giải quyết vấn đề. Không gộp số này vào cohort chrF++ cũ.

CPU trajectory/prediction audit6systems hoàn tất: validation C0−baseline BLEU +.4040/+.7841/+.8758. Best C0 epoch20/19/16 đều đúng selector BLEU. Seed44 baseline val16.2089→test20.2453, C0 val17.0847→test18.5278; direction reversal chưa xác định overfitting hay nguyên nhân module. Trên test seed44, C0 giảm ở cả nhóm target≤10chars (282rows) và11–25chars (290rows), không chỉ3câu dài. Không có empty output; short/long flags không tăng rõ. BLEU precisions2–4gram đều giảm; brevity penalty chỉ .94004→.93377, không phải toàn bộ gap do độ dài. Một số ví dụ mất điểm là đổi trật tự/dấu/chữ; cũng có ví dụ đổi từ hoặc thiếu thành phần theo reference. Không suy các tails được chọn là tỷ lệ lỗi ngữ nghĩa toàn corpus.

Trajectory rerun C0seed42 khớp20validation BLEU cũ chính xác (max difference0), hỗ trợ đây là selector replay cho seed đó. Chưa gán kết luận replay này cho mọi run khác.



GPU diagnostics hoàn tất:3seeds×4conditions×576validation rows,0optimizer steps, all-on replay PASS. Tắt module ΔBLEU +.0101/−.0596/+.0050; tắt donor −3.0082/−2.9801/−3.3059; shuffle donor −2.7787/−2.9260/−2.9021. Mạng đã phụ thuộc donor; sensitivity correction lúc inference nhỏ. Không suy ra bỏ correction khi train sẽ tốt, cũng không đủ cơ sở tăng rank/strength. Báo cáo `outputs/experiments/amis_residual_seed_diagnostics/gpu/GPU_REPORT.md`; public artifacts SHA verified HF revision `b3ab39c228a1e49d2fa0f6c25c3d33d5a059bb8a`.

Tác giả đã duyệt và launch queue Ê Đê RTX PID3505181: BASELINE_S42→C0_S42→LEARNED_DONOR_S42→M0_S42. CPU local18tests PASS,4arm roundtrip PASS. Remote CPU rồi cả4GPU gates phải PASS mới train. Recipe `configs/experiments/rhade_alignment_residual_rtx.json`, lock `configs/protocols/rhade_rtx_sdpa_seed42.json`; source id_ID,target vi_VN, original data unchanged, FP32 SDPA B4×GA32 ES10/20epoch, selector chrF++ beam1. Baseline cũ thiếu receipt runtime backend nên giữ lịch sử, chạy baseline SDPA riêng. Validation-only, không tự mở test/replication/A100. Tình trạng gate/train thực tế xem queue.log và receipts, không coi launch là train đã bắt đầu.

---

## Lịch sử — 30/10/2026: replication và test selector chrF++ hoàn tất

Đã xong11 runs mới; test15 checkpoint và dùng lại3 baseline, đủ6 nhánh×3 seeds,575 câu/nhánh. Input/protocol/selection/hash audit PASS. Test beam4 FP32; recipe RTX SDPA B4×GA32 ES10 giữ nguyên. Baseline backend lịch sử được tái dựng từ stack/default, không có runtime receipt đầy đủ.

| Nhánh | Test BLEU mean ± SD |
|---|---:|
|Baseline|19.7306 ±0.4530|
|C0|19.6194 ±0.9454|
|M0|19.6614 ±0.1122|
|Local|19.7045 ±0.8619|
|Donor|19.2226 ±0.5602|
|LearnedDonor|19.2948 ±0.6639|

Validation practical gates đều PASS; test C0−baseline=−0.1112 BLEU, seed gaps +0.7663/+0.6175/−1.7175. Holm45:1/45 significant, là C0 kém baseline seed44 (p=.0269973). Chưa có improvement significant sau Holm. chrF++ +3.0682 dễ gây hiểu nhầm vì test chỉ có3 reference word-bigrams; chrF0 chỉ+.0711. Exact seed tests không có significance. Amis test đã có development exposure.

**Khuyến nghị HOLD nâng C0 thành phương pháp chính/A100**; không tự train thêm hoặc đổi selector. C0 chỉ vượt donor/learned-donor về mean, chưa vượt Local/M0. Báo cáo đầy đủ: `outputs/experiments/amis_alignment_residual_replication_rtx_test/ANALYSIS.md`; statistics, summary và INPUT_AUDIT cùng workspace. Public analysis prefix: `analysis/amis_mandarin/mbart-large-50/alignment-residual-replication/test` trong FiveC/amis-rewire-checkpoints; kiểm tra ANALYSIS_HF_UPLOAD.json để xác nhận backup.

Các phần dưới là lịch sử, giữ nguyên số và quyết định tại thời điểm ghi.


---

# CLRR — insight nghiên cứu hiện hành

## 29/10/2026 — Learned-donor seed42 hoàn tất

CE_LEARNED_DONOR validation BLEU **17.0231**, chrF++ **14.3745**, chrF0 **16.4864**; best epoch20, train đủ20 epoch, stop=epoch cap. C0 tương ứng **17.9412 / 14.9673 / 17.2061**. C0−learned-donor: BLEU **+0.9181**, chrF++ **+0.5927**, chrF0 **+0.7197**. Bootstrap paired 10.000/seed42 trên 576 câu validation: BLEU CI95% [−0.1382,+2.0449], p raw=.046795, Holm3=.102290; cả 3 phép so sánh không đạt p Holm<.05. Kết quả thăm dò một seed, validation đã dùng để phát triển; chưa xác nhận novelty hoặc hiệu quả ổn định.

Checkpoint public `FiveC/amis-rewire-checkpoints/checkpoints/amis_mandarin/mbart-large-50/alignment-learned-donor-ce-v1/seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-rank64/`, upload receipt verified revision `440c15907ad71dcc63939c47889e4f018cd45c6e`, archive SHA256 `6a8fb4aa1d4a86d99494536fc2ded33d9a2961e18b9d17ddc6a856c79f2fff2d`. Đã xác nhận không còn best.zip/best_model/checkpoint-* của run trên RTX. SHA metadata/predictions khớp receipt; protocol và metric replay PASS. Thống kê: `outputs/experiments/amis_learned_donor_ce_rtx/analysis/statistics/PAIRED_BOOTSTRAP.md`. Không đánh giá test.

## 29/10/2026 — Lịch sử mở matched learned-donor control

Code riêng `alignment_residual/learned_donor/` kiểm tra `A(b)` so với C0
`A(b-u)`; giữ params/init/target-dummy và lock RTX. CPU16/16tests PASS, queue
PID2506609 đã qua GPU correctness gate (cache/reload/optimizer-RNG resume)
và bắt đầu train pilot seed42. Chưa có quality
score mới. Đây là đối chứng hypothesis/novelty, không phải đổi tên prior thành
method mới. HOLD block9/test/A100 giữ nguyên; C2 NO-GO không đổi. Trạng thái:
[runbook](../operations/ALIGNMENT_RESIDUAL_CE_CONTROLS_RTX.md).

## 29/10/2026 — C0 vượt CE controls về điểm, chưa xác nhận vượt baseline

Hai controls đã hoàn tất/audit PASS: CE_DONOR17.0378BLEU, CE_LOCAL17.0637;
C0 reused17.9412, baseline17.6823. C0−donor+.9034, C0−local+.8775,
nhưng0/9tests significant sau Holm9; C0−baseline chỉ+.2589, CI cũ chứa0.
Không biến gain so với controls thành gain so với baseline. Mọi model đủ20epochs.

Frozen C0 module-off chỉ+.0833BLEU; donor-off−3.8840, shuffle−3.0277.
Ủng hộ checkpoint phụ thuộc donor khi inference; chưa chứng minh module làm
training scaffold hay bỏ module lúc train vẫn tốt. C1 gradient LSR/CE trung bình
module .5365%, backbone .1842%; C2 module1.1047%, backbone0 đúng confinement.
Eval-mode snapshot không xác định cumulative Adam update hoặc nguyên nhân quality.

Novelty audit: learned skips/adapters/confinement có prior; NeuTRENO đã dùng
hiệu value giữa hai tầng. C0 khác ở learned correction trên donor−recipient
hidden outputs, nhưng chưa có learned-donor A(b) control để chứng minh phần khác.
**HOLD block9/test/A100, C2 vẫn NO-GO.** Bản thiết kế tiếp theo phân biệt9runs
factorial với thêm3runs learned-donor nếu cần; chưa launch hoặc đổi baseline lock.
Kết quả và nguồn primary đầy đủ:
[ALIGNMENT_RESIDUAL_CONTROLS_REVIEW.md](ALIGNMENT_RESIDUAL_CONTROLS_REVIEW.md).

## 28/10/2026 — Alignment residual: confined LSR không vượt quality gate

Bốn pilot seed42 trên RTX6000/FP32/SDPA hoàn thành, metric replay và protocol
audit PASS, cùng baseline R1. C0 BLEU17.9412 (+.2589 vs baseline17.6823);
C1=16.9277, C2=16.8765, C3=17.4056. C2 thua cả baseline/C0/C1/C3 về BLEU,
không đạt thresholds khóa trước: **NO-GO mở rộng C2**. Mọi model đủ20epochs;
không phải lỗi early stopping. Không sửa lock hay số cũ.

Bootstrap10.000/RNG42: C0−baseline CI95%[−.9107,+1.4384], pHolm=.644336
trong post-hoc family12. Locked12 và sensitivity18unique tests cũng không
có kết quả significant. C0 tăng chủ yếu ở source≤7words, không tăng trên
mọi độ dài. C1/C2 thua C0 ở cả4nhóm source-length; dấu gap giữ nguyên dưới
mọi single-row deletion. Đây là validation reused/một seed, chưa xác nhận
ổn định hay equivalence/harm tổng quát.

C1 giảm LSR snapshot median .4563→.1825 nhưng generation kém C0;
C2 confined .5915→.6139 và không phục hồi quality. Điều này không ủng hộ
giả thuyết confinement hiện tại đã giải quyết vấn đề. Scalar loss không đo
gradient conflict/Adam update; không tự tăng λ hoặc kết luận collapse.
C0 cần đối chiếu CLRR-only matched trước khi gán gain cho module. Novelty
HOLD. Chưa mở test/A100/training mới. Chi tiết, giới hạn và bước tiếp theo:
[ALIGNMENT_RESIDUAL_PILOT_AUDIT.md](ALIGNMENT_RESIDUAL_PILOT_AUDIT.md).

## 28/10/2026 — P1 SDPA không lặp lại lợi thế của screening

P1 ordinary depth fusion đã train seeds42/43/44 theo baseline RTX cũ đã khóa SDPA/FP32/B4×GA32/LR5e-5/ES10; dùng lại baseline tương ứng. Validation576/beam1: meanΔBLEU−0.1228, ΔchrF++−0.0503, ΔchrF0−0.0873; chỉ seed44 dương, tức1/3seeds trên cả3metrics. Mọi model đủ20epochs/720steps; không giải thích kết quả bằng early stopping. Trainer args matched trừ output/log paths, replay CSV6systems khớp số và archive SHA verified.

Paired bootstrap10000draws/RNG42, Holm9=3seed pairs×3metrics:0/9significant, mọi percentile95%CI chứa0. Không phải chỉ có gain dương mà thiếu power: mean point estimates hiện âm và seed chưa ổn định. Cũng không kết luận equivalence hoặc harm tổng quát. Eager seed42 screening không được gộp vào cohort SDPA, và chênh giữa cohorts không cô lập hiệu ứng nhân quả của backend.

**HOLD mở rộng P1 hoặc dùng làm method chính dựa trên cohort này.** Không chọn riêng seed44, không tự đổi lock/λ/topology hoặc mở test/Ê Đê. Kết luận này chỉ dành cho P1, không phải đánh giá mới mọi biến thể CLRR/LSR. Bảng đầy đủ tại `EXPERIMENTS_RTX6000.md`; audit/statistics/report: `outputs/experiments/amis_depth_p1_sdpa_rtx/analysis/`. Chỉ xử lý JSON/CSV trên CPU, không tải weights hoặc chạy inference mới.

## 26/10/2026 — Initialization audit và quyết định nâng cấp

Đã xác minh đúng R1 khởi tạo qua archive SHA256 + LFS metadata tại pinned HF commit và tái tính predictions cùng protocol; không cần inference mới. R1 validation BLEU17.6823 / chrF++14.7030 / chrF0 16.9547. S0/S1/S2/S3 thấp hơn R1 lần lượt −0.4021/−0.1230/−0.2632/−0.4157 BLEU; cả12 epoch evaluations thấp hơn R1 trên cả3 metrics. Mọi12 CI95% bootstrap so với R1 chứa0. Đây là validation-selected initialization và một seed; không suy ra harm ổn định hay nguyên nhân nhân quả.

Giữ NO-GO MRT continuation. Learned-strength zero-init được đối chiếu ReZero/gated lexical shortcuts/Value Residual Learning: chỉ đủ làm adapted control, **NOVELTY HOLD cho đóng góp phương pháp chính**; không mở training. Chi tiết bước1–3, công thức, đối chứng và correctness gates: [CONTINUATION_REVIEW_AND_NEXT_DESIGN.md](CONTINUATION_REVIEW_AND_NEXT_DESIGN.md). Artifacts: `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/initialization_audit/`. Chỉ CPU và đọc metadata RTX/HF; không tải weights hoặc xem test.

## 23/10/2026 — CPU diagnostics: sequence-risk continuation

Đã tái tính khớp 12 corpus scores của S0–S3 trên validation 576 câu; paired bootstrap 5.000 lần, toàn bộ 21 CI thăm dò chứa 0. S3−S1 BLEU −0.2927, CI95% [−1.1405, +0.5426]; S1−S0 +0.2790, CI95% [−0.4484, +1.0617]. Không có bằng chứng ổn định về gain, cũng chưa chứng minh harm/equivalence.

S1/S2/S3 best epoch 1; S0 best epoch 2 theo validation chrF++. Cả bốn hoàn tất 3 epochs (epoch cap, không phải early stopping); eval loss đều tăng epoch 1→3. Không thấy empty output hoặc lặp trigram ký tự >=3; các proxy độ dài không chứng minh lỗi bỏ sót ngữ nghĩa. Giữ NO_GO_CURRENT_RECIPE; chưa mở rộng training hay đổi trọng số dựa trên scalar loss snapshots.

Báo cáo: `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/cpu_diagnostics/REPORT.md`; code tái lập: `scripts/analysis/analyze_sequence_risk_cpu.py`. 12 file log nhỏ từ HF đã xác minh SHA256 theo commit receipt. Không tải weights, dùng GPU, train hoặc xem test. CI theo câu của một seed không thay cho multi-seed evidence.


## K?t qu? x?c minh 20/10/2026 ? sequence-risk pilot: NO-GO

?? ho?n t?t 4/4 runs tr?n RTX6000, seed42, c?ng baseline R1, 3 continuation epochs; validation 576 c?u, beam1. Ch?a ??nh gi? test. ??y l? continuation controls, kh?ng thay s? baseline g?c.

| Run | C?u h?nh | BLEU | chrF++ | chrF0 |
|---|---|---:|---:|---:|
| S0 | Baseline + CE | 17.2803 | 14.4668 | 16.6790 |
| S1 | Baseline + CE + risk | 17.5593 | 14.4865 | 16.7025 |
| S2 | CLRR + CE | 17.4191 | 14.3061 | 16.4921 |
| S3 | CLRR + CE + risk | 17.2666 | 14.2728 | 16.4245 |

**Kh?ng ??t c?c development gates:** S3?S1 BLEU ?0.2927 (chrF0 ?0.2780), S3?S0 BLEU ?0.0137, interaction BLEU ?0.4315. Baseline+risk t?ng +0.2790 BLEU so v?i continuation CE control, nh?ng chrF0 ch? +0.0235; ch?a c? b?ng ch?ng ? ngh?a th?ng k? ho?c ?? ?n ??nh nhi?u seed. K?t qu? kh?ng ?ng h? gh?p CLRR hi?n t?i v?i risk theo recipe n?y; kh?ng ch?ng minh m?i bi?n th? CLRR ??u v? ?ch.

C? 4 checkpoint ?? upload v? x?c minh hash tr?n `FiveC/amis-rewire-checkpoints`; receipts n?m trong `runs/S*/VAL_COMPLETE.json`. Queue k?t th?c `RTX_PIPELINE_COMPLETE`; kh?ng m? th?m training/test. Ngu?n s?: `outputs/experiments/amis_sequence_risk_diverse_pool_rtx/PILOT_SUMMARY.json`. C?c tr?ng th?i running/hold b?n d??i l? l?ch s?.


## Pool T1.3 — 18/10/2026, FULL POOL PASS / GPU GATES

Queue PID1620106; source bundle
`3b401dba6705e91eef00fd04f4b9cfbb94f8d2118c07fda3b409c76643c66fae`.
**Full4600rows PASS** trong981.35s trên GPU dùng chung:481singleton
(10.4565%),491equal-cost(10.6739%),meanunique3.2152. Đủ18400draws,
8177golddups,33otherdups,10190newnongold,0noEOS/lengthcap. Không loại
train rows. Pool SHA `13e41666cce019cb1da270741f34d20b0c9c362a77f996d1222f1f14be68f493`.
Đây là readiness/diversity, chưa phải quality gain. **Cả4real-data GPU gates
PASS** (2steps/save-reload/optimizer+RNG). **S0 maintraining đã bắt đầu**,
queue TRAINING/S0 và log1/108steps đã xác nhận. Chưa có model quality
result. Không nhầm epochs/steps của gate với main108steps.

8files source/PLAN/manifest/CPUgate/probe public verified ở
`analysis/amis_mandarin/mbart-large-50/sequence-risk-continuation/seed-42/t1p3`,
revision `7d89f96eefb739e82dc011384e0b3d055744199c`. Fullpool/preflight và
RUN_DESIGN.json ở từng HF checkpoint folder đã public verified6files,
revision `5f9af89e10dbcafcd36add92f7ae8dd248f159f2`: pinnedR1, poolSHA,
source revision và protocol. Đây là metadata, chưa phải weights training
hoàn tất. Raw candidate tokens giữ RTX.

User yêu cầu tiếp tục chuẩn bị vào train. Sampler T1 replay exact8/8rows;
không thấy code bug gây trùng. Probe256train rows (128original batches,
seed2026, samplingseed1042+originalstart), một lựa chọn T1.3 chốt trước:
singleton/equal-cost142/256(55.47%)→28/256(10.94%); meanunique1.695→3.180.
1024draws:462golddups,4otherdups,558newnon-gold,0noEOS/lengthcap.
Mean cost unique nongold tăng.547→.658, tức mẫu kém reference hơn, không
phải cải thiện BLEU/model. Không dùng validation/test hoặc optimizer.

Reports local/RTX `sequence_pool_temperature_probe`, exact byte hashes
verified. Recipe mới `amis_sequence_risk_diverse_pool_rtx.json` giữ cùng
R1/seed42/FP32/B4/GA32/LR1e-5/cap3/ES10/alpha.05/beta1/selection/GO.
Chỉ proposal temperature đổi1→1.3, thêm noEOS/cap<5%guard và per-row
sampling accounting. Singleton/equalcost gate<50% giữ nguyên trên full4600.
HF prefix mới hậu tố `-t1p3`; old rejected attempt giữ nguyên.

25/25 CPU tests và shared four-arm CPU gate PASS. Queue mới đi qua
full GPU smoke → fullpool → cả bốn real GPU gates trước training. Admission
VRAM16000MiB (>6GiB headroom từ peak reserved9.2285GiB), đợi tài nguyên
thay vì đổi recipe hoặc kill job khác. Đây là operational guard, không
đổi batch/precision. Chưa có model quality result mới ở thời điểm ghi mục.

[Recipe/commands/provenance hiện hành §9](METHOD_UPGRADE_DESIGN.md#9-tiếp-tục-sau-pool-gate-temperature13-namespace-mới).

## Sequence-risk continuation — 18/10/2026, STOPPED BY POOL GATE

**Cập nhật kiểm tra queue:** đã sinh đủ4600train rows trong465.51s nhưng
pool không đạt gate đã khóa. Candidate counts1/2/3/4/5 lần lượt
2560/1229/421/191/199rows. Có2560singleton rows (55.6522%) và2572equal-cost
rows (55.9130%), vượt mức yêu cầu mỗi tỷ lệ<50%. Vẫn có2028rows cost khác
nhau; không diễn giải là toàn bộ pool vô ích hoặc data lỗi. Đây là practical
readiness gate, không phải statistical test hoặc kết luận MRT không hiệu quả.

Queue dừng **trước real-data GPU gates/training**, zero main models/steps;
chỉ full-shape synthetic smoke đã có một disposable optimizer step. GPU
trở về23542MiB free/0%. Baseline/data/recipe giữ nguyên, không hạ threshold
hoặc tự tăng temperature/K để qua gate. Không có checkpoint training mới
để upload. Receipt: `amis_sequence_risk_continuation_rtx/POOL_REJECTED.json`.
Phải thiết kế/freeze lại pool nếu tiếp tục hướng này; chưa chốt sửa gì.

User đã duyệt thiết kế và chạy bốn arms trên RTX6000. Recipe
`configs/experiments/amis_sequence_risk_continuation_rtx.json`; workspace cùng
tên trong `outputs/experiments/`. Detached PID1505797, source bundle SHA256
`cf5a6ecac313a86aeb1e5fb50c1b679abd66660e085d744857fe9f7f77c3554b`.
18/18 unit CPU checks và shared CPU gate S0–S3 local/RTX đều PASS.
Full-shape GPU/AdamW preflight **PASS**: B4,L256,K5,FP32; peak9.1199GiB
allocated/9.2285GiB reserved, gradient hữu hạn, một disposable optimizer
step, không dùng data. Queue chuyển sang chuẩn bị train pool. **Chưa có
model training hoàn tất hoặc kết quả chất lượng mới**. Trạng thái trực tiếp
ở `rtx_status.json` và receipts.

S0=baseline CE continuation; S1=baseline CE+risk; S2=original CLRR CE;
S3=CLRR CE+risk. Tất cả từ cùng pinned baseline R1, LR1e-5/B4/GA32/FP32,
3epochs (nominal108steps), ES10, original chrF++ beam1 selection. Pool chỉ
từ train, R1 samples4+gold, token dedup; risk=1-sentence_chrF0/100,
alpha.05/beta1/scorechunk2. Cùng update budget, không compute-matched.

Queue: CPU → full synthetic GPU/AdamW smoke → shared train pool/signal gate
→ cả bốn real-data GPU gates → train từng model → verified public HF backup
→ xóa weights model → summary/initialization cleanup. Không test, không sweep,
không sửa số baseline cũ. Nếu gate lỗi thì dừng, không tự đổi precision/batch.

Đây là known-objective diagnostic, chưa phải phương pháp mới có novelty.
Chi tiết khóa và GO/NO-GO: [METHOD_UPGRADE_DESIGN.md §8](METHOD_UPGRADE_DESIGN.md#8-recipe-đã-duyệt-triển-khai-trên-rtx6000--18102026).

## Sequence-decision preparation — 18/10/2026, CPU COMPLETE

Đã đối chiếu hướng loss mới với MRT (ACL2016), selective KD (ACL2021),
MTO/MSO (ACL2021). MTO là margin **NMT–LM**, khác gold-vs-competitor;
không coi chúng là cùng công thức. CE vốn đã là log-sum-exp của các
competitor margins; thêm margin loss hoặc selective KD chưa đủ novelty.

Prototype **đối chứng đã biết** `src/methods/sequence_risk/v1`: finite-candidate
expected risk, 8/8 CPU gates PASS, gồm analytic gradient/finite differences,
cost detach, masks và degenerate pools. Chưa tích hợp trainer/generation,
chưa GPU gate hoặc train; không đăng ký thành method mới.

Audit CPU dùng lại 6 prediction files đã khóa: 576 validation rows, không
generate/test/train. Oracle chọn bằng **reference sentence chrF0** giữa hai
outputs beam1/beam4 của mỗi model:

| Pool, tối đa hai outputs/câu | Mean sentence chrF0 oracle | Corpus BLEU của lựa chọn oracle, chỉ mô tả |
|---|---:|---:|
| Baseline | 18.8875 | 19.7814 |
| CLRR-only | 18.3432 | 18.8078 |
| Full | 18.5600 | 19.1438 |

Full vẫn dưới baseline −0.3275 mean sentence chrF0 trong hai-output pool.
Gộp baseline+Full, tối đa bốn outputs/câu, oracle đạt21.3640; baseline+CLRR
đạt21.1309. Pool lớn hơn, nhiều model hơn và chọn bằng đáp án: **không phải
gain triển khai được, evidence novelty, hay upper bound corpus BLEU**.
Không suy ra MRT chắc chắn sửa được CLRR hoặc architecture đã hết tiềm năng.

**Quyết định:** numerical prototype GO; novelty/new training HOLD. Đã viết
thiết kế bốn nhánh đối chứng có điều kiện, chưa tạo queue. Muốn chạy phải
hoàn thiện scoring/generation/Trainer và GPU gates trước. Nếu MRT giúp cả
baseline lẫn CLRR như nhau thì cải thiện thuộc objective đã biết, không
phải bằng chứng CLRR đóng góp.

Chi tiết và lý do: [METHOD_UPGRADE_DESIGN.md §7](METHOD_UPGRADE_DESIGN.md#7-quyết-định-tiếp-theo--18102026).
Artifacts local: `outputs/experiments/sequence_risk_preflight/`; chưa upload
các artifacts mới của mục này lên HF. Các số thí nghiệm cũ giữ nguyên.

## Token decisions + beam validation — 17/10/2026, COMPLETE

R1/P1/R2,576validation rows,5.494content tokens/model; strict archive/source
checks,24/24 greedy gate rows, native CE replay và independent summaries PASS.
Không train/test; tải một checkpoint/lần rồi xóa, GPU trở về idle.

Baseline/CLRR-only/Full content top1:50.4550/49.3629/49.6360%; Full có content
NLL3.427548 tốt hơn baseline3.474846 nhưng mean margin−.442526 kém hơn−.286092.
Full sửa227 nhưng mất272 reference top1 decisions. Nhóm cả hai vẫn sai đóng
góp+.076131 vào net NLL benefit+.047298: surrogate gain chưa thành decision gain.
Full−baseline top1 CI theo câu[−1.662052,+.018216]pp có chứa0; không claim
Full giảm significant qua seeds. Rare target count1–5:23.9832→22.0196% top1.

Beam4 BLEU baseline18.174130,CLRR-only17.467068,Full17.779134. Full−baseline
−.394996,descriptive CI[−1.526070,+.726329],không đảo thứ hạng beam1.
**HOLD interaction/global alignment pilot.** Chuyển sang thiết kế/counterexample
về gold-vs-strongest-competitor decisions và bảo vệ quyết định đang đúng,
nhưng phải novelty-audit margin/distillation/routing prior trước khi code/train;
không kết luận causal culprit là LSR hoặc mọi search effect đã bị loại.

Tables, masks, decomposition, CIs và replay CLI:
[TOKEN_DECODING_DIAGNOSIS.md](TOKEN_DECODING_DIAGNOSIS.md).
26 files diagnostics/source đã public và verify ở
`analysis/amis_mandarin/mbart-large-50/token-decoding/seed-42/beam1-beam4-fp32`;
weights tải tạm đã dọn, raw beam4 predictions giữ local/RTX.

## Generation validation diagnosis — 17/10/2026, COMPLETE

Dùng lại đúng 576 predictions beam1 FP32 của R1/P1/R2; 9 input files HF pinned
và archive selections verified. CPU clrr, không GPU/weights/generation/train.
Direct replay 9 corpus scores, 117 strata rows, 1.728 sentence pairs PASS;
số cũ giữ nguyên.

Full−baseline validation −0.566757 BLEU/−0.473825 chrF0. Ở rare-source fraction
>1/3 (train-unit count≤1), 113 rows: Full−baseline −3.555818 BLEU, CLRR-only
−2.588749. Proxy lexical rarity, một seed; chưa chứng minh causal morphology.
Full tốt ở reference≤9 chars (+1.015426 BLEU), kém ba nhóm dài hơn.
Không có empty/repeated-trigram outputs; bỏ một row không đảo dấu deltas.

Full có CE tốt hơn ở 315 câu nhưng chrF0 giảm 137, tăng 131/tie 47;
Spearman ΔchrF0 với CE benefit chỉ .1533, với Δnative gap .1239.
N-gram losses có trường hợp không tương ứng lỗi nghĩa mới (row127).
**HOLD train interaction/global alignment mới.** Bước phân biệt nguyên nhân
tiếp theo: gold-token top1/margin theo target rarity và matched validation
beam1/beam4; chưa chạy các phép đo GPU đó trong lượt này.

Tables, examples, giới hạn và CLI tái lập:
[GENERATION_VALIDATION_DIAGNOSIS.md](GENERATION_VALIDATION_DIAGNOSIS.md).
12 artifacts dẫn xuất/source đã public và verify ở
`analysis/amis_mandarin/mbart-large-50/generation-validation/seed-42/beam1-fp32`;
raw examples/snapshots giữ local, không upload lại.

## Source-grounding probe — 17/10/2026, COMPLETE

Ba checkpoints Amis R1/P1/R2, toàn bộ576validation rows, cùng576negative
pairs theo144blocks, bốn cells ON/OFF×matched/mismatched;6.912CE values.
Zero optimizer steps/test; strict load và independent replay3/3 PASS,
gồm10k descriptive block bootstrap. Weights tải tạm đã xóa khỏi RTX.

Trong CLRR-only/Full, route tăng source-match gap lần lượt+.422297/+.408490
nats/token: vừa hạ CE đúng source, vừa tăng CE source không khớp. Đây là
bằng chứng có lợi cho route trong weights đã co-adapt, không ủng hộ giải
thích route chỉ tăng target likelihood bất kể source.

Nhưng native gap baseline(OFF)3.481013 vẫn cao hơn CLRR-only(ON)3.421993
và Full(ON)3.376570; Full matched CE2.763204 thấp hơn baseline2.804171.
Không đồng nhất gap/calibration với BLEU hoặc hallucination. **HOLD pilot
interaction:** maximize I chưa được xác nhận là mục tiêu còn thiếu, và loss
có thể tăng I bằng cách làm OFF yếu đi. Không mở ba runs mới.

Worker đã dùng flat CE đúng HF/audit sau khi gate chặn khác biệt rounding
của3D kernel; không sửa số cũ. Chi tiết bảng, CIs, provenance và hướng
phân tích lỗi generation tiếp theo:
[SOURCE_GROUNDING_VALIDATION_REPORT.md](SOURCE_GROUNDING_VALIDATION_REPORT.md).
Artifacts nhỏ đã public trên HF ở `analysis/amis_mandarin/mbart-large-50/source-grounding/seed-42/rtx6000-fp32`;
28files/0.60MB đã verify, receipt `source_grounding_validation/HF_UPLOAD.json`.

## Prototype route/source interaction — 17/10/2026

Đã thiết kế hai hướng objective ở prediction level: reference OFF detached
không được chọn làm contribution chính vì gradient chỉ là source-contrast
được reweight; bốn-cell differentiable interaction được code riêng với
paired-CE/source-only controls. Không sửa phương pháp gốc hoặc số đã chạy.
Source-grounding vẫn là giả thuyết: residual utility cũ chưa đo wrong source.

Code: `src/methods/grounded_interaction/v1/`. Prototype có nguy cơ tăng
interaction bằng cách làm OFF yếu đi; paired CE không bảo đảm chặn việc đó.
Technical gate PASS không phải quality/novelty GO. Chưa mở pilot training;
chưa có BLEU mới. Công thức, prior và receipts:
[design/preflight](GROUNDED_INTERACTION_DESIGN_AND_PREFLIGHT.md).

CPU9/9 và CUDA9/9 checks PASS; tiny FP32 batch4/GA32 PASS; full mBART
geometry610.9M, batch4/length256/GA2 PASS, peak allocated12,425MiB.
Full gate dùng random weights, không optimizer/Adam state; chưa chứng minh
full training hoặc generation gain. CPU/GPU source hashes đã đối chiếu khớp.

## Residual utility và novelty gate — 16/10/2026

Hoàn tất bước1–2 được tác giả duyệt:5checkpoint (Amis baseline/CLRR-only/Full,
Ê Đê baseline/Full), toàn bộ576/1000validation rows;12conditions mỗi model,
44.736CE rows. FP32/dropoutoff, zero optimizer steps, strict archives/source,
15correctness checks và5CSV replays PASS. Weights tải tạm đã dọn; chỉ giữ
JSON/CSV nhỏ. Không train hoặc chọn lại checkpoint.

CE(OFF)−CE(ON) trung bình: Amis baseline−.09065, CLRR-only+.20259,
Full+.23984; Ê Đê baseline−.02622, Full+.17053nats/token. Route giúp khoảng
69–72% câu ở weights đã train cùng CLRR, nhưng chỉ36–37% khi chèn route vào
baseline. Đây là heterogeneity/co-adaptation ở fixed checkpoints, không phải
bằng chứng train gate sẽ tăngBLEU. Cả10layer-removal means đều tăngCE ở các
route-trained checkpoints; chưa có cơ sở bỏ các tầng đầu/cuối theo quy tắc chung.

Novelty screen: lexical shortcuts/WMT2019, GateSkip và conditional-depth
counterfactual CE supervision có prior rất gần. **NO-GO chọn generic gate hoặc
CE-supervised gate làm đóng góp mới chính ở formulation hiện tại; HOLD training.**
Nhãn CE cả câu cũng không xác định token source nào hữu ích: thiết kế ban đầu
phải chuyển sang gate theo câu/tầng hoặc có can thiệp từng token riêng.
Chi tiết công thức, controls, số đo và giới hạn:
[RESIDUAL_UTILITY_ROUTING_AUDIT.md](RESIDUAL_UTILITY_ROUTING_AUDIT.md).

Ê Đê seed42 đã hoàn tất train/test4model trước lượt đo này: Full cũ−baseline
+.1115BLEU, P3−baseline−.5437; không contrast nào đạt Holm12 trong test pilot.
Checkpoint/test artifacts và thống kê đã verified trên HF; không có test inference
mới trong residual-utility audit. Đây chưa là bằng chứng cải thiện ổn định.

## Dataset thứ hai: Ê Đê→Việt — lịch sử chuẩn bị 15/10/2026

Tác giả đã dừng Turkish và chọn Ê Đê để kiểm tra source ngoài danh mục mBART,
cùng tình huống thích nghi với Amis. Early stopping Turkish không chứng minh
dataset đó lỗi. Chưa có kết quả model Ê Đê, chưa thay kết luận của Amis.
Đã export nguyên văn15.092/1.000/1.000 từ revision HF đã khóa, giữ overlaps,
Unicode và thứ tự; chỉ đổi sang CSV `source,target`. Kế hoạch4run seed42
Baseline/Full cũ/NCE/P3 đã được duyệt và launch gates với sourceproxy `id_ID`,
target `vi_VN`; proxy chọn theo prior Chamic–Malayic, chưa theo model scores.
Queue PID373554, recipe `rhade_contrastive_id_rtx.json`; CPU4/4 và GPU4/4 PASS,
Baseline đã train step5/2360 lúc22:17. Bundle/audit/gates public đã xác minh;
chưa có validation/test model scores Ê Đê. Chi tiết:
[EDE_SECOND_LANGUAGE_PLAN.md](EDE_SECOND_LANGUAGE_PLAN.md).

## P3 vs NCE-only: kiểm tra không train hoàn tất — 13/10/2026

Paired bootstrap10k trên575 câu test: P3−A NCE-only=−0.0721 BLEU,
CI95[−1.2435,+1.1079]; P3−B=+0.1232, P3−C attached=+0.0253.
Tương tác P3−P2−B+A=+1.0614 nhưng CI[−0.5060,+2.7332]. Không phép nào
trong18 kiểm định mới đạt Holm<.05; tất cả CI chứa0. Đây là exploratory sau
khi đã xem test, không thay historical Holm33 hoặc chứng minh equivalence.

Audit A/B/C/P2/P3/baseline: selected đúng; chọn theo validation BLEU cũng
cho cùng epochs19/19/18/19/16/19. Cả6 đủ20epochs, nên early stopping hoặc
đổi selector sang BLEU không giải thích kết quả hiện tại. Không có chrF0
theo từng epoch để kiểm tra selector đó; không chọn lại checkpoint bằng test.

Đo thêm riêng A trên cùng576 validation rows và4batch gradient: centered
retrieval S→T68.58%, T→S69.27%, covariance rank119.96/120.06. P3 cao hơn
retrieval ở70.66%/73.44% nhưng rank thấp hơn103.93/113.62. NCE-only đã tạo
phần lớn raw retrieval gain; representation bổ sung của P3 chưa chuyển
thành ưu thế dịch được xác nhận. Không claim anti-collapse.
Loss decomposition, finite gradients, parameter versions và zero optimizer
steps PASS; weights tải tạm đã dọn. LSR probe của A là counterfactual.

**HOLD training hàng loạt; giữ P3 như development candidate.** Chưa đủ
căn cứ tănglambda, tăngpatience hoặc chọn method theo test. Nếu kiểm tra
thêm seed, đề xuất baseline/NCE-only/P3 cùng recipe trước nâng cấp khác.
Chi tiết, CI, giới hạn và lệnh ở
[evidence report](../../outputs/experiments/amis_contrastive_evidence/EVIDENCE_REPORT.md).

18 artifacts JSON/CSV/Markdown và mã tái lập public đã verify SHA/size ở
`analysis/amis_mandarin/mbart-large-50/contrastive-controls/seed-42/evidence/`,
HF revision `050e140d13e18b563b0df108a82c7ad9bfa8db6e`. Representation NPZ
chỉ giữ trên RTX: automatic review từ chối công khai vì chưa có quyền riêng.

## P3: novelty và checkpoint probes — 12/10/2026

Đã đo baseline/Full cũ/P2/P3 seed42 trên576 validation pairs và cùng4
microbatches gradient, RTX FP32, dropout off; zero optimizer steps, đúng
archive SHA, loss decomposition và parameter-version checks PASS.
P3 raw retrieval Amis→Mandarin71.70% vs baseline54.69%/Full55.56%; centered
70.66% vs65.97%/66.49%. Đây là tín hiệu phân biệt cặp dịch có ích, không chỉ
raw cosine cao. P2 centered69.44%; gap nhỏ này chưa được kiểm định significance.

P3 covariance effective rank source/target103.93/113.62, thấp hơn baseline
146.85/142.22: không claim anti-collapse. Global CE–NCE cosine dương cả4
batches nhưng vẫn có layer-local conflicts; không tăng lambda hoặc bỏ tầng
chỉ theo instantaneous gradients. Full cũ raw matched cosine cao nhất nhưng
retrieval gần baseline, nên cosine cao không đủ giải thích translation gain.

Đọc PDF và code prior: mRASP2 đã có mean-pooled contrastive MT; SkipCLM đã
có skip+contrastive; Middle Alignment đã thử stop-gradient. P3 khác thiết kế
cụ thể nhưng novelty cho ACL chưa xác nhận. **GO giữ ứng viên development;
HOLD chốt phương pháp chính hoặc mở confirmation hàng loạt.** Số test cũ
giữ nguyên, không train mới. Chi tiết, sources, bảng đo và câu hỏi đối chứng:
[P3 review](P3_NOVELTY_AND_MECHANISM_REVIEW.md).

## Token pilot: audit CPU/GPU hoàn tất — 11/10/2026

Pilot RTX seed42: Token-Full validation BLEU16.7533, thấp hơn baseline17.6823
và Token-LSR17.5811; không vượt quality gate đã khóa. Cả4model đủ20epochs,
cùng runtime/bundle/arguments, nên thiếu epoch do early stopping không giải thích
được kết quả này. Loss wrapper và gradient PASS, chưa thấy pooled rank collapse
nghiêm trọng. Token-LSR giảm token objective trong4batch cố định nhưng không tăng
full-validation BLEU; thêm CLRR tăng token cost ở probe này và giảm validation scores.
Đây là bằng chứng về hạn chế objective/proxy, chưa xác định quan hệ nhân quả.

Mức tăng test chrF++ +4.6681 của Token-Full nhạy với word-bigram cực thưa:
chỉ3reference bigrams, baseline match0 và candidate match1. chrF0 chỉ tăng+.1259.
Can thiệp tầng có dấu trái chiều giữa Full gốc và Token-Full; tất cảCI chứa0 sau
paired bootstrap10k, không đạt Holm12. Chưa có căn cứ cho quy tắc bỏ tầng đầu chung
hoặc tănglambda lớn chỉ dựa vào instantaneous gradient norm.

Giữ trạng thái NO-GO cấu hình Token-Full v1 trong pilot này; bước tiếp theo cần
giả thuyết nâng cấp có đối chứng, chưa tự mở training nhiều seed.
[Báo cáo đầy đủ](TOKEN_PILOT_AUDIT.md) và
[sổ RTX](../experiments/EXPERIMENTS_RTX6000.md) chứa số liệu và giới hạn phép đo.

## Token alignment probes — RTX6000, 10/10/2026

Đã đo baseline42 và corrected Full42 trên128validation rows,20target
derangements chung, FP32/dropout off/TF32 off; SHA archive/data và strict load
PASS, zero optimizer steps. Đây là cost alignment, không phải BLEU/chrF mới.

| Checkpoint | True token-transport cost | Shuffled cost | Shuffled−true | CE–token gradient cosine mean |
|---|---:|---:|---:|---:|
| Baseline42 | .813909 | .912818 | +.098909 | +.104515 |
| Full42 | .615566 | .691098 | +.075533 | +.020910 |

Cả hai phân biệt true/shuffled ở127/128rows; sensitivity giới hạn độ dài
giữ gap+.094374/+.071492. Full giảm cost cảtrue vàshuffled; gap tương đối
gần nhau (~10.84%/10.93%), không kết luận giảmdiscrimination từ absolutegap.
Đã có tín hiệu tokencorrespondence; chưa chứng minh thiếu tokenalignment là
nguyên nhân confirmation thất bại. Gradientfinite8/8 trênmicrobatch4; không
gộp với diagnosticsbatch32 hoặc suy gradientnorm thành Adamupdate.

CPU/GPU correctness PASS. Nhưng tokenOT bất biến với hoán vị thứ tựtoken và
chấp nhận identicalcollapsedtokens với loss≈0: không claim syntaxpreservation
hoặc anti-collapse. **Technical GO cho pilot cócontrols nếu được duyệt; quality
vànovelty chưa xác nhận.** Không training mới hoặc thay số cũ trong lượt này.
Chi tiết vàprovenance: [report](../../outputs_rebuttal/token_alignment_probe/REPORT.md).

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

### Nontraining audit completed ? 14/10/2026

18/18 validation selections PASS; all18ran20epochs. No early-stopping or selector explanation of the seed instability. Full44?baseline short sources?5words:?3.3693BLEU/178rows; whole-test loss remains negative under every single-row deletion. Only3/575refs contain native word bigrams; Full?baseline chrF++ is fragile to row170 (over4.6point influence in seeds42/43). Primary575rows and scores remain unchanged.

GPU probes:8new+4reused after exact batch/data/archive checks; same576validation rows/fourbatch4, FP32 eval, zero optimizer steps, parameter versions unchanged, loss decomposition PASS. Full CE?LSR cosine .0749/.1056/.0819 and CE?NCE .0558/.1179/.0737 across seeds42/43/44; no mean-negative layer groups. End-checkpoint probes do not prove training-time conflict is absent. Weighted LSR/CE norms?.27?.28%, NCE/CE?.82?.91%; instantaneous norms are not Adam update magnitudes.

Centered S?T retrieval mean: Baseline63.89%, NCE69.27%, LSR+NCE69.04%, Full70.08%. Source covariance effective rank mean:144.69/119.11/102.23/105.51. Alignment helps retrieval while BLEU gains are inconsistent. Reduced rank is compression, not proven harmful collapse. Full residual/current RMS?.070?.084 in layers3?12, no seed44 spike.

Conclusion: revisit alignment objective/detail preservation rather than increasing weights blindly or changing patience. No new method/training started. Full evidence: `outputs/experiments/amis_contrastive_multiseed_evidence/NONTRAINING_REPORT.md`. CPU public revision c7eed619c590c0932056ec0b18872223c21efdde; GPU public revision224d68f0fe7ed8bc4c21298858bc0169818a9427. Temporary weights cleaned; NPZ only retained on RTX.
# PLC post-hoc failure audit — 2026-11-15

Completed CPU prediction/history analysis and no-optimizer RTX probes on pinned
W/P/D checkpoints; no training or test. Full report:
[PLC_FAILURE_ANALYSIS.md](PLC_FAILURE_ANALYSIS.md).

- PLC−W BLEU−0.557598, descriptive paired sentence-bootstrap95%CI
  [−1.567756,+0.365787] on already-selected validation checkpoints. This is a
  failed practical promotion gate, not demonstrated statistically significant
  harm across seeds. All five runs completed20epochs/720steps.
- PLC loses reference-length groups<=9/10–11/12–14 but gains>14; no empty or
  repeated-trigram output for PLC/W. Sentence chrF++ improves175, loses172,
  ties229. Reject a blanket 'long sequences collapse' explanation.
- Real loss geometry: on W's16fixed rows x2masks,39.665% token observations
  have PLC derivative opposite to their own logp discrepancy, allowing
  compensation.21.229% low-gold-probability tokens(<.01) carry61.674% of PLC
  squared output-gradient norm. These are measured design properties, not a
  complete causal training explanation.
- 0.1PLC weighted logit-gradient/CE median0.428,max3.152 at W. Thus a small
  scalar auxiliary value does not guarantee a negligible gradient. Parameter
  and logit spaces differ. On four rows x three named blocks, negative cosine
  counts PLC5/12 vs D1/12, but W's KL also6/12: conflict alone is not sufficient
  to explain translation harm. Measurements are correlated, not independent seeds.
- Mean cross component P−D is0.455540 at W vs0.496137 at PLC on the fixed probe;
  selective suppression of the motivating cross-drift was not demonstrated.
- CPU analytic/autograd and frozen-block gradient checks PASS; exact RNG replay,
  strict reload, artifact SHA checks, original target tokenizer IDs on16rows
  PASS. Diagnostic embedding-hook error was fixed before completed probes;
  no evidence it affected historical full-backbone training.
- NotebookLM restricted review used actual cited sources; generic gradient
  surgery/normalization remains prior (PCGrad/MetaBalance), not an admitted new
  method. No weight sweep, new training, replication, test or A100 launch.

All new small artifacts remain separate under
`outputs/experiments/amis_prefix_consistency_failure_analysis/`.
Temporary restored weights were removed; GPU returned to23952MiB free.

# SAC-v1 design follow-up — 2026-11-15

New research candidate after the PLC failure audit: Source-Attribution
Consistency. Exact finite-table loss and protocol are in
`docs/research/SOURCE_ATTRIBUTION_METHOD.md`. Mathematical prototype and nine
CPU checks pass; no training wrapper, GPU gate, pilot scores or queue exists.
The source-choice/dropout conditional-MI term is distinct from the objectives
in the specific R-Drop/LR-Drop/DKD/CBMI/ContraDecode/CCL/RKD papers checked, but
this is not worldwide novelty certification. The chain rule and CMI primitives
are prior. Source-blind collapse remains an explicit counterexample; proposed
controls and source-retention checks are mandatory. Existing results remain
unchanged. Proposed next training budget is three Amis seed42 runs, conditional
on native GPU/RNG/accumulation gates and train-only calibration; not launched.


## SAC exact coupled-mask frozen probe — COMPLETE, 2026-11-15

No training or test. Two pinned existing seed42 checkpoints: baseline and W.
128 evenly spaced Amis validation sentences, two source derangements, two dropout
pairs. FP32 SDPA TF32 off; shared padded shapes and RNG replay across sources.
CPU checks and both native GPU gates PASS. Checkpoint/archive and collected CSV
hashes verified; all lexical summary means replay within1e-10. Transient restored
HF weights removed only inside this probe; GPU free23952MiB after completion.
Only small JSON/CSV/Markdown collected locally, approximately0.6MB.

| Lexical diagnostic | Baseline | W |
|---|---:|---:|
| C = I(M;S given V) |0.01173794|0.00877760|
| B = I(M;V given S) |0.11311421|0.08041024|
| Ratio mean C / mean B |0.10377072|0.10916025|
| Source information |0.40704737|0.40496691|
| Correct-minus-distractor gold logp |4.17587733|4.14305048|
| Absolute source-gap change across masks |1.45389747|1.23111944|

W absolute C lower25.2%; mask source-gap variability lower15.3%. W-minus-baseline
C=-0.00296034, approximate paired sentence-bootstrap95%CI[-0.00364783,-0.00230133].
Source-gap delta=-0.03282684 CI[-0.12790443,+0.06313675]; source-information delta
=-0.00208046 CI[-0.00737780,+0.00315333]. CI crossing0 is not proof of equivalence.
Do not claim selective removal of C: C/B ratio is slightly higher in W.

W C/error Spearman0.3130 falls to partial rank0.03326 after entropy and source/
target character-length adjustment; approximate sentence-bootstrap95%CI
[-0.17089,+0.24019]. This is descriptive, not causal identification. Conditioning
can overadjust or leave confounding; source-pair dependence and training-seed
uncertainty are not covered by these fixed-pairing intervals. Onlyseed42 and an
extensively reused validation set; teacher-forced errors are not generation BLEU.

Decision: SAC training admission remains HOLD; deprioritize three four-forward
pilots because ordinary W already exhibits the targeted stability proxy and no
separate benefit from further reducing C is demonstrated. Not proof SAC cannot
work. No new BLEU/test score, no method-quality claim, no automatic pilot launch.

Full artifacts: outputs/experiments/amis_source_attribution_frozen_rtx/ANALYSIS.md,
ANALYSIS.json, PLAN.json, COLLECTION.json and per-checkpoint CSV/GPU_GATE.json.
Reproduce: python scripts/analysis/operate_source_attribution_probe.py monitor
(or collect); python scripts/analysis/analyze_source_attribution.py --output
outputs/experiments/amis_source_attribution_frozen_rtx. Use conda clrr.

## 2026-11-15 — Frozen decision-branch probe and literature screen

Status: COMPLETE diagnostic; **no new method or training admitted**.

- Exact pinned baseline/W seed42 checkpoints reused from the SAC plan. 32 evenly
  spaced rows of 576 Amis validation sentences, two fixed dropout pairs per row,
  FP32 / SDPA / TF32 off. No optimizer or test access.
- Each checkpoint generates its own native greedy trajectory. At the first
  lexical argmax disagreement between two full-prefix dropout forwards, force
  each proposed action after the SAME prefix and complete with dropout-off
  greedy decoding. This is not a pair of cached stochastic decoding trajectories.
- CPU gate and both native GPU replay gates PASS. Exact source/reference rows,
  checkpoint provenance, CSV SHA and all branch sentence scores replay verified.
  11 remote JSON/CSV files, 62,632 bytes collected; no weights downloaded locally.
  Remote verified transient checkpoint copies removed; project lock free and
  GPU returned to 23,952 MiB free after completion.

| Descriptive quantity | Baseline S42 | W S42 |
|---|---:|---:|
| Pairs with at least one eligible lexical disagreement |54/64|52/64|
| Sentences with any measured event |29/32|29/32|
| Conditional mean absolute branch chrF0 gap |6.7825|8.2054|
| Conditional mean absolute branch chrF++ gap |6.2996|7.5253|
| Partial rank: local symmetric KL vs absolute chrF0 gap |−0.0825|−0.1398|
| Approximate sentence-bootstrap 95% interval |[−0.4197,0.2684]|[−0.5221,0.2773]|

Average events within sentences before associations; adjust for entropy, prefix
length and target character length. 2,000 descriptive fixed-rank sentence
bootstrap replicates. No-event pairs are missing for conditional gap analyses,
not zero-regret observations. Neither masks nor sentences are training seeds.

Interpretation: this small probe does not establish that larger local dropout
KL predicts larger downstream reference-relative branch costs. It does not
justify selecting training positions by KL or claiming a causal exposure-bias
failure. The larger conditional gap for W does NOT show W is worse: its generated
prefixes, chosen positions and event coverage differ. No rollout hit length cap.
chrF0 and sentence BLEU preferred opposite branches in 7 baseline and 4 W events;
one reference cannot establish which paraphrases are semantically valid.

Literature decisions:
- The proposal to sample model prefixes, branch actions, roll out and optimize
  fixed reference-relative local costs is substantially covered by SEARNN:
  https://www.jbalayrac.com/papers/searnn_paper.pdf . Dropout action sampling
  alone is not accepted as an adequate methodological distinction.
- R-Drop remains prior art and the successful W control, not our new method:
  https://papers.nips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf .
- Decoder dropout consistency plus an anti-language-model contrastive objective
  has prior in Section3.4 of What Works and Doesn't Work, A Deep Decoder for NMT:
  https://aclanthology.org/2022.findings-acl.39.pdf .
  Related goals are NOT proof every source-sensitive consistency objective is
  algebraically identical.
- Vocabulary hierarchies are a close prior requiring formula-level comparison:
  https://aclanthology.org/N19-1046.pdf . This is inter-layer hierarchical
  regularization, not proof every final-layer dropout grouping is identical.
- NotebookLM screen overclaimed complete collisions, confused Amis SOURCE
  resource scarcity with Mandarin TARGET grouping, and treated near-zero SAC
  association as contradiction of all source-neglect hypotheses. These claims
  were challenged and corrected; raw replies are retained as unverified research
  assistance, not authoritative novelty evidence. The correction also incorrectly
  describes this probe as independent stochastic trajectory divergence; actual
  implementation is native-prefix forwards, as stated above.

Decision: no blind KL-weighted/selective/on-policy pilot. No claim that all
candidate mechanisms are impossible, or that a novel useful objective is now
available. Further design must identify an intervention and substantive
difference from the closest prior; these diagnostics alone do not supply either.

Artifacts: outputs/experiments/amis_decision_branches_frozen_rtx/
PLAN.json, CPU_GATE.json, per-checkpoint GPU_GATE.json/RESULT.json/decisions.csv,
COLLECTION.json, ANALYSIS.json/md, LITERATURE_SCREEN.json and
LITERATURE_CORRECTIONS.json.

Reproduce (conda clrr):
```powershell
python scripts/analysis/operate_source_attribution_probe.py monitor --output outputs/experiments/amis_decision_branches_frozen_rtx
python scripts/analysis/operate_source_attribution_probe.py collect --output outputs/experiments/amis_decision_branches_frozen_rtx
python scripts/analysis/analyze_decision_branches.py --output outputs/experiments/amis_decision_branches_frozen_rtx
```



## CMC continuation consistency — RTX pilot queue (2026-11-15)

- Author authorized implementation and RTX pilots; no test stage.
- Design/formula/prior-work screen: `docs/research/CONTINUATION_CONSISTENCY_PILOT.md`.
- Recipe: `configs/experiments/amis_continuation_consistency_rtx.json`.
- Separate implementation: `src/methods/continuation_consistency/v1/`.
- Three new Amis seed42 runs: CMC_S42, NORM_S42, PERMUTED_S42.
- Reuse pinned W42 (validation BLEU18.2724095712), baseline42 (17.6823291745);
  no retraining of these controls. Middle remains a secondary existing RTX comparator.
- Protocol locked: FP32 SDPA, LR5e-5, micro4/GA32, cap20, ES10,
  validation BLEU beam1; original CSV references, no dataset modifications.
- Four CMC CPU test groups PASS locally and on RTX; five paired-route regression
  tests PASS locally. GPU gates and long-training results are NOT yet complete.
- Bundle ZIP SHA256: a3b92743013b3abf2fbe4c003a578068e01381b17328624592c87c33e0c9b5e8.
- Manifest SHA256: 867725ec181dda138900477d35b5c738512eba11cd5cd85f8aafccaa6824a49c.
- Detached queue PID2642063, remote output:
  `/workspace/ura/thanhdat/projects/CLRR/outputs/experiments/amis_continuation_consistency_rtx`.
- Initial monitoring: alive; CPU PASS; restoring pinned W for train-only channel
  admission (zero optimizer steps). All three GPU gates must pass before any long run.
- Scientific admission/success thresholds fixed in design before results; no
  semantic-equivalence or global novelty claim. NotebookLM raw screen retained at
  `outputs/experiments/amis_continuation_consistency_rtx/NOTEBOOKLM_SCREEN.json`.
- Checkpoints upload to canonical public FiveC/amis-rewire-checkpoints under
  checkpoints/amis_mandarin/mbart-large-50/<method>/seed-42/
  rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu; shared engine removes transient weights
  only after remote SHA verification.

Frozen-W train-only channel admission PASS: retained KL ratio0.6731495449, mean channel separation0.9494252521;32 train rows x2 dropout seeds,0 optimizer steps. This establishes a nontrivial channel, not translation gain. Receipt: outputs/experiments/amis_continuation_consistency_rtx/continuation_probe/ADMISSION.json. GPU gates are now running.


## CMC estimator frozen-gradient audit — 2026-11-16

TRAIN-only, frozen W42, zero optimizer steps. All-parameter conditional position-sampling variance: CV/old pooled ratio **0.083786773**, reduction **91.621%**, improved **8/8** conditions (four short TRAIN rows × two dropout seeds). CPU expectation/identity tests PASS; temporary restored checkpoint removed. Source and data hashes are pinned in `outputs/experiments/amis_cmc_estimator_gradient_audit/COMPLETE.json` and `SOURCE_MANIFEST.json`; full table in that folder's `REPORT.md`. This is not evidence of generation gains or the historical failure cause, and control variates are not a new statistical principle. Two validation-only seed42 pilot arms are authorized conditional on both GPU correctness gates; no new test or replication is authorized.

## CMC estimator follow-up: useful diagnostic, failed method gate (2026-11-17)

Source: `outputs/experiments/cmc_estimator_review/REPORT.md`, `ANALYSIS.json`, SHA manifest, manual `REFERENCE_ERROR_AUDIT.md`; completed receipts under `outputs/experiments/amis_cmc_estimator_rtx/runs/`. No new training, GPU or test in this follow-up. Both new checkpoints backed up with verified receipts; scores independently replayed on576 aligned validation rows.

| Seed42 validation | BLEU | chrF++ |
|---|---:|---:|
| Baseline |17.682329|14.703005|
| W: dense weak full consistency |18.272410|15.172807|
| CMC |15.540216|13.316294|
| Plain sampled direct SKL |16.133234|13.699138|
| CV-CMC |17.148787|14.439779|

1. **[MEASURED] Better estimator does not imply a better translation method.** CV gains1.608571 BLEU over CMC, still trails W by1.123622 and baseline by0.533542. The preregistered CV−W>=+0.30 screen fails. No seed replication/test is warranted by that gate. This one-seed pilot does not establish population effects.
2. **[MEASURED] Early stopping is not the explanation.** All four W/CMC/Sampled/CV arms reached20epochs. Best epochs20/18/17/19 respectively. Checked data, seed, backbone revision, scientific recipe and runtime fields agree.
3. **[INFERENCE] Sampling/optimizer dynamics deserve part of the explanation, not all of it.** Plain sampled SKL also trails dense W; CV reduces logged late gradient norms relative to CMC. However gradients are clipped at1.0, and Adam is nonlinear. The frozen-W91.62% conditional variance reduction cannot be extrapolated to every training step or converted into measured translation gains. Exact dense CMC usefulness remains unknown.
4. **[MEASURED] Teacher-forced CE is not an adequacy proxy.** Final validation CE is lower in CMC/Sampled/CV than W, but generated BLEU is worse. Last-microbatch training CE and pre-clipping logged norms are not epoch means or covariance estimates.
5. **[MEASURED] Counting metric wins can mislead.** CV beats W sentence chrF0 at196 rows and loses192, but mean delta is−0.859891. A stratified16-row inspection shows content substitutions, omitted predicates and relation/aspect errors in both outputs, including metric-gain rows. Row188 gains19.491 sentence chrF0 while omitting cross-making; row484 gains0.687 while replacing worry with dislike. These are reference-relative examples, not expert Amis semantic labels or corpus prevalence estimates.
6. **[MEASURED] No obvious generation collapse.** All five576-row prediction files have zero empty outputs and zero occurrences of the defined severe repetition flag (4-character substring>=3). This does not rule out semantic errors or omission.
7. **[DECISION] Avoid repeating families.** `PRIOR_DIRECTIONS.md` indexes87 local shared-runner receipts including donor, alignment residual, token/transition alignment, depth/strength, contrastive, sequence-risk, native/paired/prefix consistency and CMC. It is not a complete HF census; A100, other languages, selectors and backends must not be pooled. Ordinary Full seed42 in a separate cohort reaches18.458409, so W42 is not universally best; W is the successful multiseed comparator for the present question.

Next research requirement: identify a source/translation error with an intervention that distinguishes competing causes before proposing another loss. Saved strings and scalar norms do not identify source-neglect, lexical encoding failure or harmful semantic invariance. No new method or novelty claim is established here.

### Other-seed stratification follow-up — 2026-11-17

CPU only. Source: `outputs/experiments/cmc_estimator_review/shared_errors/INTERPRETATION_AND_NEXT_STEP.md`, with seven verified input hashes and all576 rows saved. W43/W44 define post-hoc groups; W42/CV42 are evaluated. These are sentence chrF0 differences, not corpus chrF++: CV−W is+0.1233 on185 rows where both other seeds score below10, versus−1.3251 on391 remaining rows. Thresholds5/10/15 preserve this descriptive pattern. Agreement alone does not identify semantic error. Removing the largest absolute delta leaves−0.745311; trimming5% per tail leaves−0.309643. Rare/unseen-word and nearest-training-source partial rank associations are small (absolute value<0.041), supplying no positive mechanism for rarity/source-distance weighting. No new method admitted. Next proposed diagnostic compares exact-objective versus sampling/clipping gradient directions, reusing existing frozen-audit artifacts first; no GPU/training/test launched. NotebookLM follow-up failed connection; no novelty clearance claimed.

### Step5 candidate screen — 2026-11-17

`docs/research/CMC_STEP5_CANDIDATE_AUDIT.md` records the formula, ten web queries, four primary papers read at method level, and completed NotebookLM roundsA/B. The minimum exact-position CMC candidate changes only numerical evaluation of the old stop-gradient surrogate: **reject as independent method novelty; retain only as conditional diagnostic control**. No training/GPU/test launched. Previously proposed gradient steps2–4 remain unperformed, not causal evidence. Existing moments recover nonzero dense CMC-minus-direct auxiliary-gradient norm in8/8 short-TRAIN conditions (relative difference0.052659–0.575534); this does not identify gradient angle, CE conflict, clipping bias or Adam behavior. Global norm clipping preserves each realized gradient direction; the potential bias concerns E[clip(G)] versus clip(E[G]) after the true accumulation32 pipeline. Raw debate and reproducible CPU recovery live in `outputs/experiments/cmc_estimator_review/shared_errors/`. No new method admitted and no claim of exhaustive novelty verification.


## Frozen CMC CE-inclusive update geometry — 2026-11-17 (COMPLETE)

Artifacts: `outputs/experiments/amis_cmc_update_geometry_rtx/INTERPRETATION.md`, `ANALYSIS.json`, `DENSE.json`, `TRIALS.json`, `COMPLETE.json`, `COLLECTION.json`. Reproduce analysis with `python -X utf8 scripts/analysis/summarize_cmc_update_geometry.py` in conda clrr. Fixed W42 checkpoint,128 TRAIN rows,32 microbatches of4,8 conditional position-sampling repeats with identical replayed dropout. FP32/SDPA, no optimizer constructed/updates, parameters unchanged, no test/validation decoding. CPU/GPU gates PASS; native GPU VJP relative error0. Runtime942.20s, peak6819.07MiB allocated; temporary restored weights removed, only small JSON collected and SHA verified.

| Quantity | Result |
|---|---:|
| Dense W auxiliary cosine with CE |+0.395056|
| Dense CMC auxiliary cosine with CE |+0.361411|
| CE-inclusive W/CMC cosine |0.976959|
| Relative total-gradient change CMC versus W |22.4718%|
| Sampled CMC: RMS error / dense total norm; clipped |1.79784;7/8|
| CV: RMS error / dense total norm; clipped |0.776001;1/8|
| Sampled direct: RMS error / dense total norm; clipped |1.94545;8/8|

**[MEASURED]** Both dense auxiliaries align positively with CE on this batch; weighted auxiliary/CE norm ratios1.40717(W),1.21785(CMC). Sampling remains noisy after accumulation32. CV empirical conditional MSE decreases81.3695% versus sampled CMC here; this differs from the earlier91.621% variance result on four short rows. Dense W/CMC total norms0.756193/0.685585 lie below clip1, whereas sampled estimates frequently cross it.

**[LIMIT]** Eight repeats do not establish systematic clipping bias: distances of clipped MC means from clipped exact totals0.457718/0.171450/0.511076(CMC/CV/direct) include conditional Monte Carlo error (trace SE0.305463/0.184057/0.321143). No optimizer-state files exist in this pinned archive; historical Adam effects remain unmeasured. These results do not identify the causal training failure, population effects or dense-CMC translation quality.

**[DECISION]** No new method/training admitted. Avoid another sampler/CV-only proposal, and do not motivate gradient-conflict surgery or larger weights using an unsupported claim of CE conflict/negligible auxiliary gradients. Prior CV training still trails W; estimator improvement alone did not rescue it. New method design needs a discriminating explanation of actual translation failures. Full interpretation and commands are in the artifact report.

## DRM novelty and frozen direction diagnostics — 2026-11-18

Primary audit/debate: `docs/research/DRM_NOVELTY_AUDIT.md`. Distinct decision-gradient-conditioned source-value dynamics from the specific Mi2016/Titans/EDA/QED/CARVE methods read; projection, coverage and gradient-memory primitives are not new. Novelty classification remains GẦN, not ACL certification.

**[MEASURED]** Corrected artifact `outputs/experiments/drm_direction_probe_v3/`:32 fixed TRAIN sentences on each pinned W42/DRM42, FP32 SDPA eval, no optimizer/test, unchanged parameters. V3 matches update norms per head; max error2.3164e-7; max contraction coefficient.554311; peak2484.89 MiB. Raw JSON collected with hash verification; temporary restores removed by worker after public archive SHA verification. V1 random control was invalid; V2 guard failed. Neither is a training run or usable efficacy result.

DRM trajectory minus sequential-zero mean NLL: W−4.6072e-6 (sentence-bootstrap CI95%[−1.3412e-5,+2.9017e-6]); DRM−3.3162e-6 ([−1.3017e-5,+5.7609e-6]). Directional comparisons favor gradient over context/scalar by about1e-5 NLL, but this does not establish practical translation gain. On DRM weights, gradient−random-axis NLL−6.6137e-6 with CI containing0. Single-update future-token NLL on DRM−1.6183e-6, CI containing0. Random-axis is best-of8/sign selected by label-free state support, not an isotropic signed random baseline. Attention-argmax reuse is not semantic alignment; no semantic-consumption claim established. Pointwise TRAIN bootstrap is exploratory, not corrected multi-comparison or seed-level confirmation.

**[DECISION]** No long training/replication/A100 launch. Existing recipe screen also fails: DRM−W validation BLEU+.200029806 versus required+.30; chrF++−.017085220 versus required>=0 (original metrics read back from RTX, saved in `outputs/experiments/drm_direction_probe_dispatch/prior_validation_receipts.json`). Do not use already-viewed test to override that gate. These findings weaken the case for current DRM v1, not a proof that all output-conditioned memory designs fail. See `docs/research/DRM_MECHANISM_CHECK.md` for limitations, code and reproducible commands.

## Frozen W42 native-source recoverability — 2026-11-18

**[MEASURED]** `outputs/experiments/w_read_recoverability_rtx/REPORT.md`:32 fixed validation middle tokens,20 initially wrong; pinned W42, gold prefixes, FP32 SDPA eval, no model optimizer/test. Query and simplex re-reading native source values recovered0/20 at context-norm budgets3/10/30%; unrestricted direct context recovered0/0/2. At30%, wrong-token gold NLL changes:−.304510/−.759516/−2.378260. Context/direct intervention is label-informed, not a translation method.

**[MEASURED]** On18 paired adjacent-rank positions with both targets initially wrong, direct30% recovered gold1/18 and arbitrary alternative1/18; query/simplex0/18. Direct gold−alternative margin-gain mean+.192066, exploratory sentence-bootstrap CI95%[−.328638,+.716139]. Reduced NLL/margin alone does not establish selective correction; two direct-context cases beat the fixed native rival without becoming actual gold top1.

**[INFERENCE / DECISION]** No gold-selective repair signal sufficient to admit a new source-read training method from this probe. This does not establish absent semantic information or impossibility:32 tokens, local2-restart search and bounded displacements. Zero recovery is not equivalence/zero population rate. Retain the negative result to avoid proposing an attention reweighting or context correction solely from oracle NLL improvements. No new training launched. Full gates, provenance and limits in `docs/research/DRM_MECHANISM_CHECK.md`; all old generation scores remain unchanged.
