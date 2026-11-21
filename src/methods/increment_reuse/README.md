# Increment reuse: RO / DW / RW

Research prototype, isolated from the original CLRR/LSR and existing W runs.
It is not registered with the training runner. Long-training admission is
pending the contribution decision in `docs/research/METHOD_REDESIGN_CANDIDATES.md`.

Let `d = output_k - input_k`, and `u` be the receiver's state after attention.
`G` includes the native FFN pre-LayerNorm and dropout:

| Variant | Receiver output |
| --- | --- |
| `ro` | `u + G(u + alpha*d)` |
| `dw` | `u + G(u) + alpha*d` |
| `rw` | `u + alpha*d + G(u + alpha*d)` |

One scalar `alpha` starts at zero. The donor stays differentiable. All
tokenizer special IDs and padding are excluded from the reused increment.
Both independent-dropout views use the same variant and W's exact loss:
mean CE plus 1.25 times symmetric full-vocabulary KL.

Pass zero-based layer indices explicitly; no layer pair is selected yet.
Construct each variant from a fresh unwrapped mBART. Do not deepcopy an
already patched model. Enable gradient checkpointing on the wrapper after
construction. Receiver checkpoints receive the donor as an explicit tensor
input so its gradient is not silently detached. Per-invocation masks survive
recomputation without retaining the encoder's mutable cache.

`save_pretrained` stores the backbone and `increment_reuse_config.json`,
including the learned scalar. Restore with `IncrementReuseMT.from_pretrained`,
not bare `MBartForConditionalGeneration.from_pretrained`. This does not yet
constitute a verified optimizer-resume path through the shared runner.

Verification in conda `clrr`:

```powershell
python -m unittest discover -s tests -p test_increment_reuse.py -v
```

Tests use a random tiny mBART, not dataset training or translation evaluation.
There are no new BLEU scores. GPU correctness, full-backbone resource use,
frozen recipe integration, upload/restore, and resume gates remain pending.

## Prior and claim boundary

The local RO mapping is expressible using Hyper-Connections Eq.2 with cached
states `(u,H_k,H_(k-1))`, read weights `(1,alpha,-alpha)`, carry `u`, and FFN
write to the current stream only. This is local operator subsumption, not a
claim that our whole backbone equals the full multi-stream HC architecture.
MUDDFormer Eq.7 decouples attention Q/K/V/residual while its FFN still reads
the same state used by its residual. These comparisons establish overlap,
not novelty certification. A possible contribution is an empirical finding
about read versus write in pretrained low-resource MT; that finding has not
been demonstrated.

- https://arxiv.org/pdf/2409.19606
- https://arxiv.org/pdf/2502.12170
