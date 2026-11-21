# Decoder-conditioned residual bridge — proposal v1

Date: 2026-11-06. Status: IMPLEMENTATION APPROVED; Amis + Ede, eight seed42 pilots; gates precede training.
No code, GPU queue, training, or test run is authorized by this proposal itself.
Working name DRB; not an established novel contribution or final paper title.

## Evidence and RQ

C0 has no stable baseline improvement in the audited Amis/Ede cohorts. Frozen
module-off effects are small while donor removal is harmful after co-adaptation.
CPU/CUDA synthetic gradient correctness passed. These findings motivate changing
where/how donor information is consumed; they do not prove that lower layers
contain missing translation information or identify a causal training defect.

RQ: Does a decoder read of cross-depth differences improve translation beyond
an equally sized local adapter, donor-only read, and source-conditioned control?

## Architecture, fully specified pilot

- Native pretrained mBART encoder remains unchanged; do not insert original CLRR.
- Capture H6 (after encoder block6) and HN (normal final encoder output).
  Layer6 is a fixed midpoint design choice, not an empirically optimal lexical layer.
- Inject exactly one module at decoder block12 cross-attention output, before
  its existing output dropout/residual-add/layernorm. No changes elsewhere.
- Full backbone fine-tuning plus module, not PEFT and not parameter-neutral.
- D=1024, r=64. Norm nu(v)=v/sqrt(mean(v²)+1e-6), parameter-free.
- No LSR, InfoNCE, distillation, external labels, extra data, target encoder pass,
  route-oracle supervision or new loss coefficient in the initial pilot.

Let q_t be the decoder input to native cross-attention. For each existing head h:

A_t^h = softmax((Q_t^h K_N^{h,T})/sqrt(d_head) + source_pad_mask).

Q and K are the native projections using their exact pretrained/trained weights
and biases. A is pre-attention-dropout, deterministic. Native SDPA computation
continues unchanged; recompute A only for this side branch, under no_grad.
Do not switch attention backend to obtain attention weights.

For k in {6,N}, using the SAME native value projection and SAME A:

c_t^k = concat_h[A_t^h * V^h(nu(H_k))].

The norm here applies to both depths. c_t^N is the side branch normalized-top
reference, not exactly the native unnormalized cross-attention output.
Shared attention fixes the read positions across the two depths, but does not
prove attention is a gold alignment or that the feature bases are aligned.

Main candidate:

e_t = sg(c_t^6 - c_t^N)
z_t = GELU(W_d nu(e_t))
g_t = sigmoid(W_q nu(sg(q_t)))
delta_t = W_u (z_t * g_t)
o'_t = o_native,t + delta_t.

Use native existing dropout ONCE on o'_t at the original location, then native
residual-add/layernorm/FFN. No extra stochastic module/dropout or RNG draws.

W_d,W_q in R^(r x D); W_u in R^(D x r); no biases.
196608 extra parameters. Private CPU generator seed10042 initializes W_d/W_q
uniform(-1/sqrt(D),1/sqrt(D)); W_u=0. No global RNG consumption at insertion.
No outer alpha=.1: delta is the only added branch, initialized to zero.
This avoids mandatory step-zero donor injection; it does NOT guarantee larger
useful gradients, stable training, a bounded correction, or improved quality.

Side feature extraction, including q/K/V/A/native projection weights, is no_grad
with ordinary training/eval flags intact; deterministic feature math has no
random draws. Native backbone paths still receive ordinary CE gradients.
The stop-gradient policy is a design choice, not a new learning principle.
At W_u=0 only W_u initially receives nonzero module gradients; W_d/W_q should
receive gradients once W_u moves. Do not incorrectly fail this expected pattern.

Training loss is native non-pad target CE, same reduction as baseline. Decoder
inputs are shifted target prefixes during teacher forcing, generated prefixes
at inference. The side branch never accesses current/future target tokens.
There is no training-only target memory or frozen external teacher.

## Four matched pilot arms, seed42 only

All have identical W_d/W_q/W_u shapes, initialization, backbone policy, location,
norms, detach policy, attention reads and dropout RNG schedule.

| ID | e_t | gate input | Question |
|---|---|---|---|
| D0 local | sg(cN) | sg(q_t) | Is gain just extra nonlinear capacity? |
| D1 donor | sg(c6) | sg(q_t) | Is cross-depth subtraction necessary? |
| D2 source-modulated difference | sg(c6-cN) | sg(masked mean nu(HN)) | Is direct decoder-query modulation necessary? |
| D3 DRB | sg(c6-cN) | sg(q_t) | Main candidate |

D2 uses source non-special/non-pad positions; if there are none, use non-pad
positions. Broadcast the same source vector over decoder positions. It STILL
uses target-conditioned A_t: do not claim it removes all target conditioning.
Equal parameter count does not prove equal effective capacity or compute; log
parameter/trainable counts, wall time and peak VRAM. Execute the same side reads
in all four arms and discard unused contexts to match side-path compute.

Existing native baseline may be reused only after exact recipe/version/data/
RNG/backend/selection audit. Zero-up logits/loss, representative CE gradients
and RNG must match the native baseline in train and eval. If not, this is a
TECHNICAL HOLD: fix the implementation, do not redefine baseline silently.
Do not repeat C0, old Full/CLRR/LSR or comparator models just for this pilot.

## Relationship to CLRR/LSR and prior work

This replaces the original CLRR forward rule: preserves the cross-depth
information-reuse question, not the exact encoder rewire. LSR is absent in the
first pilot to isolate mechanism and CE translation utility; it is not declared
useless. Add LSR only after an architecture GO, in a separately locked factorial.

| Prior read | What is already known | What this design tests |
|---|---|---|
| Transparent Attention, EMNLP2018 | Decoder access to mixtures of encoder depths | Same-position low-minus-top read with nonlinear query-conditioned correction |
| Exploiting Deep Representations, EMNLP2018 | Aggregation and multi-layer attention for NMT | Difference rather than simply exposing/fusing more layers |
| Value Residual Learning, ACL2025 | Cross-layer values, shared attention, learnable mixing | Encoder-depth contrast read at a decoder step; not fixed first-value addition |
| Lexical Shortcuts, WMT2019 | Gated access to lower lexical features in NMT | No claim that shortcuts or gates are new |
| Orthogonal Residual Update, NeurIPS2025 | Orthogonal projection before residual addition | No orthogonal projection; subtraction is NOT proof of removing redundancy |
| Standard adapters/ReZero | Bottleneck modules and zero-init residual paths | Neither claimed as novel |

Sources:
- https://aclanthology.org/D18-1338.pdf
- https://arxiv.org/abs/1810.10181
- https://aclanthology.org/2025.acl-long.1375.pdf
- https://aclanthology.org/W19-5211.pdf
- https://papers.neurips.cc/paper_files/paper/2025/file/67c15da4a9340140c60783d9a175fd3f-Paper-Conference.pdf

Novelty verdict: plausible specific combination for a mechanistic pilot, NOT
novelty-cleared or ACL-ready. Related ingredients are established. Need direct
comparisons above and broader citation audit before a main contribution claim.
A name, algebraic difference, new language, or positive BLEU alone does not
establish a substantial contribution. Full prior reimplementations are not
part of the first four-run screen; they become required if DRB is promoted.

## Locked experiment and gates (author approved)

Run Amis first (4600/576/575, tl_XX->zh_CN), then Ede original data/splits: train15092/validation1000/test1000, id_ID->vi_VN.
RTX FP32, TF32off, SDPA, B4/GA32, LR5e-5, AdamW fused decay0, warmup.06,
linear schedule, clipping1, length256, checkpointing reentrant, max20epochs,
ES10, validation selector BLEU beam1 earliest best. BLEU tokenizer zh for Amis,
13a for Ede. New explicit BLEU protocol locks retain the original data hashes
and training settings. No test access during pilot. chrF++ is secondary.
Record chrF0 as secondary. Amis chrF++ sparse-word issue remains documented.

G0 correctness, CPU then tiny real-data RTX (check >=20GiB free; otherwise wait):
1. Zero-up output/RNG matches native model, native gradient check.
2. Gradients/detach, finite difference, all four arms actually distinct.
3. H6=HN implies zero difference correction even after nonzero module weights.
4. Padding independence, no future target leakage, batch/beam independence.
5. Generation cache, beam reorder, incremental vs full decoding equivalence.
6. Checkpointing/dropout forward+gradient replay, accumulation4 gate.
7. Save/reload exact metadata/weights/logits, optimizer resume and RNG replay.
8. Actual SDPA/FP32/backend and data/protocol hash receipts.

G1 seed42 practical screen (proposed thresholds fixed before any pilot):
- D3 exceeds native baseline by >=.30 validation chrF++ AND >=.30 BLEU.
- D3 exceeds each D0/D1/D2 on validation BLEU, and exceeds the strongest of
  them by >=.10 BLEU; chrF++ >= each matched control.
- All technical gates passed; no silent precision/batch/selector changes.
These are engineering screening thresholds, not significance or established
minimum meaningful effect sizes. Compare each run's own BLEU-selected checkpoint.
No test fishing or post-hoc threshold relaxation. If fail, report and stop this
candidate without sweeping donor layers/rank/coefficient.

G2 conditional replication: D3 and validation-best matched control on seeds43/44
=4 additional runs; baseline reuse requires per-seed audit, otherwise report the
needed baseline runs before any launch. Gate: positive D3-baseline on BOTH BLEU
and chrF++ in >=2/3 seeds, means >=+.30 for both; mean D3-control positive on
both. n3 is development evidence, not statistical confirmation.

Then language transfer: D3 plus strongest fixed control seeds42/43/44 on Amis
=6 runs, with matched baseline reuse audit. Both datasets already exposed to
development, so this is transfer/robustness, not a clean external confirmation.
Only after recipe lock and appropriate seed/power design consider fresh seeds,
A100 matched controls/primary endpoints and a new untouched evaluation resource.
Never pool RTX/A100 numerical results as if same recipe. Do not tune by tests.

At final test: pinned validation-selected models, beam4, original references,
BLEU13a for Ede, 10k paired bootstrap, explicit multiplicity family BEFORE tests,
plus independent seed-level uncertainty. No significance claim from seed42.

## Implementation boundary and storage when approved

src/methods/decoder_residual_bridge/v1/ (model, module, metadata, README), separate
from all old methods. Integrate through existing registry/args engine, not new
per-model launchers. Configs amis_decoder_residual_bridge_rtx.json and rhade_decoder_residual_bridge_rtx.json; output same
stem under outputs/experiments/. Reproduction bundle hashes and gates required.

Public HF destination pattern:
checkpoints/<amis_mandarin|rhade_vietnamese>/mbart-large-50/drb-<arm>-v1/
seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu-rank64-donor6-dec12[-src-id]/

Only verified final selected archive plus metadata/predictions; delete remote
temporary weights after SHA backup confirmation. Store diagnostics separately
under analysis/rhade_vietnamese/mbart-large-50/decoder-residual-bridge/.
No date suffix, no overwriting any old prefix. Windows receives small artifacts.

Approved execution: eight new runs (D0–D3 on each dataset), seed42 only.
No replication, test or A100 run is authorized by this queue. Further stages
above remain conditional proposals and require review of the pilot outcomes.
