# Decoder Residual Bridge v1

CE-only full fine-tuning of native mBART with one zero-initialized low-rank
side module at decoder block 12 cross-attention output. Native attention remains
SDPA. Source block 6 and final source states are read using the same detached,
pre-dropout decoder attention distribution and native value projection.

Arms: `drb-local-v1` uses final source read; `drb-donor-v1` uses block6 read;
`drb-source-v1` uses their difference and pooled-source gate; `drb-query-v1`
uses their difference and current decoder-query gate. Rank64 adds196608
parameters. No LSR, additional dropout, or target-encoder pass.

Side inputs are detached; zero-up initialization preserves native logits,
backbone CE gradients and RNG. Packed graph-local encoder memory supports
checkpoint recomputation and beam generation without mutable donor caches.

Recipe args: method, donor_layer6, decoder_layer12, adapter_rank64,
module_seed=seed+10000. The pilot validator locks these design values.

Run through the shared CLI (Python must be in conda clrr):

```bash
python scripts/experiment.py build --config configs/experiments/amis_decoder_residual_bridge_rtx.json
python scripts/experiment.py cpu --config configs/experiments/amis_decoder_residual_bridge_rtx.json
# Repeat build/cpu for rhade_decoder_residual_bridge_rtx.json, then:
python scripts/experiment.py launch-suite --configs configs/experiments/amis_decoder_residual_bridge_rtx.json configs/experiments/rhade_decoder_residual_bridge_rtx.json
```

Suite holds the project lock; all eight GPU gates must pass before training.
Validation BLEU selects checkpoints and ES10, maximum20epochs. No test.
Design rationale and limits: docs/research/DECODER_RESIDUAL_BRIDGE_PROPOSAL.md.
Novelty remains a research hypothesis; a gate or low-rank bridge alone is not
claimed as a new contribution.
