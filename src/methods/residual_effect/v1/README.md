# Residual-effect alignment v1 — experimental

Implementation: `lsr.py`. Method IDs and exact controls:

| ID | Encoder CLRR | Original sentence LSR | New alignment |
|---|---|---|---|
| `ordinary-lsr-v1` | No | 0.10 | Pooled source/target InfoNCE |
| `ordinary-full-v1` | d2, strength0.10 | 0.10 | Pooled source/target InfoNCE |
| `effect-clrr-v1` | d2, strength0.10 | No | CLRR ON−OFF effect InfoNCE |
| `effect-full-v1` | d2, strength0.10 | 0.10 | CLRR ON−OFF effect InfoNCE |

For paired sentences, `s = pool(E_on(x)) − pool(E_off(x))`,
`t = stopgrad(pool(E_on(y)) − pool(E_off(y)))` in effect mode.
Ordinary controls use ON pooled representations without subtraction.
Loss: mean one-way cross entropy over paired normalized dot products / temperature.
Source ON and OFF both receive gradients; target receives none from the new loss.
Original LSR retains its original target stop-gradient and training/dropout behavior.

Auxiliary pooling excludes tokenizer special IDs and padding; encoder attention
still includes special tokens. Auxiliary views disable dropout and checkpointing
through encoder eval mode, then restore every module training flag and strength.
They run before the original CE/LSR forward, leaving CE dropout RNG unchanged.
No optimizer steps, queue, additional parameters, EMA or teacher checkpoint.
Singleton evaluation batches have zero contrastive loss (no negatives); the
training guard still requires at least two paired sentences. This does not alter logits.
Generation delegates to the original backbone with CLRR active where applicable.
Checkpoint reconstruction requires the run spec, as with existing wrappers.

Temperature and weight are explicit spec arguments, not inferred from gradient
ratios. Proposed pilot uses both0.10, one seed42, microbatch4: three negatives
per example. GA32 does not make a 128-example contrastive batch. This objective
can degenerate for zero/common effects; it has no anti-collapse guarantee.
Effect mode adds four encoder forwards, ordinary mode adds two; differentiable
source auxiliary views retain activation graphs without checkpointing. Full-size
VRAM and speed must be measured before training; tiny CPU PASS cannot establish
RTX memory feasibility.

Novelty HOLD: cross-language contrastive alignment and response-change alignment
both have prior work. The proposed architecture-response formulation is a
hypothesis requiring further comparison, not a certified novel contribution.
