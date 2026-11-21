# Decoder depth-difference prototype v1 (U2)

**Correctness prototype, not an approved training method or a novelty claim.**
Implemented against Torch 2.6.0 / Transformers 4.57.6 mBART, eager attention,
FP32, LayerDrop zero. It is deliberately absent from `methods.registry`.
Existing methods, checkpoints and training recipes are unchanged.

## Definition

For one decoder cross-attention layer, use its existing Q/K/V/O projections
on the final encoder output and a detached intermediate encoder state:

```text
delta = (softmax(Q K_donor^T / sqrt(head_dim) + mask)
       - softmax(Q K_top^T   / sqrt(head_dim) + mask)) @ (V_donor - V_top)
context = native_attention_context + gate[head] * delta
```

Apply the original output projection/bias once. Gates start at zero. The
main branch retains native attention dropout; delta has no additional
dropout. Detach affects donor hidden states only, not shared projections or
the sequential encoder gradient. For mBART-large-50 the proposed locations
are encoder layer 6 and decoder layer 12 (1-based): 16 additional scalars.

The prototype internally packs recipient/donor along the memory feature
dimension. Ordinary decoder layers read recipient only. The selected layer
packs both projected K/V pairs in its dynamic cross-attention cache, so
native beam expansion/reorder acts on both. No donor is held on a module
between calls. Public `encode()` and forward output expose normal-width
encoder states; precomputed memory must use `DepthEncoderOutput`.

## CPU gates

Run in the existing conda environment:

```bash
conda run -n clrr python tests/test_depth_difference.py
conda run -n clrr python scripts/analysis/preflight_depth_difference.py \
  --model tiny --donor-layer 2 --device cpu \
  --output outputs_rebuttal/method_design/u2_tiny_cpu_preflight.json
```

The unit tests use a random tiny mBART and synthetic token IDs; they are not
BLEU experiments. They cover native equality/RNG/base gradients at gate0,
the formula and donor detach, checkpointing in both modes, causal/padding
masks, cache reorder, greedy/beam cache equivalence, stale-memory isolation,
extra parameter count and strict save/load.

## Cached pretrained preflight on RTX (no training)

Use the project-local environment, caches and temp directories documented in
the RTX runbook. The script checks the CLRR lock and free VRAM before model
loading, requires local files, performs zero optimizer steps, and does not
export a checkpoint. It evaluates only the first two training rows to
inspect initialization behavior; these are not quality estimates.

```bash
./conda/envs/clrr/bin/python scripts/analysis/preflight_depth_difference.py \
  --model facebook/mbart-large-50-many-to-many-mmt \
  --cache-dir cache/huggingface/hub \
  --data data_processed/amis_mandarin/train.csv \
  --donor-layer 6 --device cuda --rows 2 --minimum-free-gpu-mib 16000 \
  --output outputs/experiments/u2_preflight/pretrained_gpu.json
```

The gate measures context/correction norms and gate gradients, checks native
logits and greedy generation at gate0, then uses temporary gate=.1 to test
beam4 with/without caching. This temporary value is a numerical test input,
not a selected hyperparameter or an optimizer update.

## Explicit custom serialization

```python
model.save_pretrained(folder)
restored = DepthDifferencePrototype.from_pretrained(folder)
```

Files: config, generation config, `prototype_config.json` and
`prototype.safetensors`. Use this loader. Generic HF mBART loading must fail
rather than silently dropping the gates. Resume/Trainer integration,
arbitrary attention backends, static caches, other backbones, mixed
precision and distributed training are outside this prototype's verified
scope. Native `get_encoder()` returns the backbone encoder; use
`prototype.encode()` for precomputed U2 decoder memory.

CPU/GPU correctness does not establish a BLEU improvement or solve the
prior overlap documented in `docs/research/METHOD_REDESIGN_CANDIDATES.md`.
