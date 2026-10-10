# Ki?m k? chuy?n early stopping sang patience 10

Ng?y: 08/10/2026. Gi?i h?n gi? nguy?n 20 epoch. Ch?a train, ch?a s?a m? ngu?n/defaults/launcher.

**Theo y?u c?u m?i: kh?ng ch?y l?i CLRR, LSR, Full c? v? c?c ablation/tuning/warmup c?a ch?ng.**

## K?t qu?

| D? li?u | Backbone | Nh?nh | Seed | S? h? s? | Epoch k?t th?c / b?ng ch?ng | Quy?t ??nh |
|---|---|---|---|---:|---|---|
| Amis | mBART | Baseline corrected development | 42,43,44 | 3 | 20 / 20 / 20 | Gi? |
| Amis | mBART | Baseline fresh confirmation | 45,47,48,49,50,51,52 | 7 | 20 each | Gi? |
| Amis | mBART | Baseline fresh confirmation | 46 | 1 | 18 | C?n ch?y l?i |
| Amis | mBART | Baseline original historical | 42 | 1 | 20 | Gi? ri?ng l?ch s? |
| Amis | mBART | Middle Alignment | unrecorded | 1 | unknown | Ch?a ch?t; thi?u endpoint |
| Amis | mBART | Strong LoRA A | 42 | 1 | 20 | Gi? |
| Amis | mBART | Strong LoRA B | 42 | 1 | 20 | Gi? |
| Amis | mBART | Narrow LoRA | unrecorded | 1 | unknown | Ch?a ch?t; thi?u endpoint |
| Amis | mBART | BitFit | unrecorded | 1 | unknown | Ch?a ch?t; thi?u endpoint |
| Amis | NLLB | Baseline | unrecorded | 1 | best19; endpoint unknown | Ch?a ch?t; thi?u endpoint |
| Amis | NLLB | Middle Alignment | unrecorded | 1 | best19; endpoint unknown | Ch?a ch?t; thi?u endpoint |
| Amis | NLLB | Strong LoRA A | 42 (docs) | 1 | best18; endpoint unknown | Ch?a ch?t; thi?u endpoint |
| Amis | NLLB | Strong LoRA B | 42 (docs) | 1 | best19; endpoint unknown | Ch?a ch?t; thi?u endpoint |
| Amis | NLLB | Narrow LoRA | unrecorded | 1 | 20 | Gi? |
| Amis | NLLB | BitFit | unrecorded | 1 | best18; endpoint unknown | Ch?a ch?t; thi?u endpoint |
| Amis | mT5 | Baseline | 42 | 1 | 20 | Gi? |
| Amis | mT5 | Middle Alignment | unrecorded | 1 | unknown | Ch?a ch?t; thi?u endpoint |
| Amis | ByT5 | Baseline | 42 | 1 | unknown | Ch?a ch?t; thi?u endpoint |
| Amis | ByT5 | Middle Alignment | unrecorded | 1 | unknown | Ch?a ch?t; thi?u endpoint |
| Turkish | mBART | Baseline | 42 | 1 | 12, already ES10 | Gi?, ?? ES10 |
| Turkish | mBART | Middle Alignment | 42 | 1 | 12, already ES10 | Gi?, ?? ES10 |

## S? l??ng

- Gi?: **17 h? s?** (bao g?m historical Baseline mBART42 ri?ng v? hai Turkish ES10).
- C?n ch?y l?i ch?c ch?n: **1 run ? Baseline mBART Amis seed46**, d?ng epoch18.
- Ch?a ch?t: **11 h? s?**; thi?u b?ng ch?ng epoch k?t th?c. Kh?ng g?i ch?ng l? early-stopped.
- N?u kh?ng ph?c h?i ???c b?ng ch?ng v? mu?n t?i t?o t?t c? m?c ch?a ch?t: **t?i ?a12 run** ??i ch?ng. ??y kh?ng ph?i12 run ?? ch?ng minh c?n train l?i.
- Kh?ng bao g?m ?ng vi?n m?i, th?m seeds, hay benchmark hi?u su?t.

## Gi?i h?n v? quy t?c

- Best epoch18/19 kh?c completed epoch. NLLB Baseline/Middle/BitFit c? counter0 t?i best; kh?ng c? b?ng ch?ng ch?ng d?ng v? patience. V?i completion b?nh th??ng, best mu?n c?ng g?i ? ?? ch?m cap20, nh?ng ch?a c? final state ?? ch?ng nh?n endpoint.
- Hai Strong LoRA NLLB c? s? trong EXPERIMENTS_UPDATED.md; ch?a t?m th?y matching archives/final receipts trong main public inventory. Kh?ng k?t lu?n weights ?? m?t ? m?i revision/repo.
- Archive Baseline46 ch? c? model/tokenizer/training_args/training_receipt. Kh?ng c? optimizer/scheduler/RNG v? stopping-state weights ?? resume ch?nh x?c.
- Gi? metadata l?ch s? patience4; th?m ghi ch? ho?n th?nh cap20 n?n d?ng l?i. Kh?ng s?a th?nh ?? train patience10.
- Kh?ng g?p original Baseline42, corrected development42-44 v? fresh45-52 v?o m?t mean.
- Patience t?nh b?ng evaluation events. Run m?i d?ng eval m?i epoch, patience10, max20, gi? c?c y?u t? recipe kh?c c?a cohort.
- ??i patience kh?ng t? s?a c?c kh?c bi?t selection beam/LR/batch gi?a c?c cohort. N?u th?ng nh?t c?c y?u t? ?? n?a c?n ??nh gi? reuse ri?ng.
- Kh?ng m? l?i Ashaninka ho?c LayerSkip; zero-shot/probes/metrics/bootstrap kh?ng c?n train l?i v? patience.

## B?ng ch?ng

- [Audit JSON](audit.json)
- [Public inventory](public_inventory.json)
- [Small ZIP metadata](archive_metadata.json)
- [Baseline46 resume inspection](baseline46_resume_check.json)
- [Fresh selection audit](../amis_post_confirmation_audit_20261007/training_selection_audit.csv)
- [S? th? nghi?m hi?n h?nh](../../docs/experiments/EXPERIMENTS_UPDATED.md)
- Preserved Strong LoRA run A/B verification JSON + final logs under workspace_cleanup_20261003/preserved_bundle_reports/.

Public revision inspected: `a719a52fd9d178621d3f00cee23e15984546c63e`. ZIP evidence transferred about2.82MB plus a small Baseline46 directory range; no weight members were downloaded.
