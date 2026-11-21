# Corrective-order numerical admission correction

Author requested repairing admission and running the full queue. Original formulas and scientific training recipe remain in CORRECTIVE_ORDER_PILOT.md; disk execution amendment remains CORRECTIVE_ORDER_DISK_PREFLIGHT.md.

Observed on fixed TRAIN batch: eta .003,.01,.03 produced normalized penalties 996036678.51,48645709.07,2377339.36; no accepted scaling region. This does not establish the small-step limit; the original range may be too large. No training took place.

Amended train-only calibration: first require exactly zero endpoint distance and probe SKL for identical dropout masks (101,101), using full model and unchanged replay/checkpoint path. Then use fixed eta grid [.00003,.0001,.0003,.001] and masks (101,211). Retain original adjacent-ratio acceptance [.5,2] for both SKL/eta^4 and parameter distance/eta^2. Choose smaller eta of first accepted pair. No lowering quality gates, no validation/test access, no optimizer steps in calibration. If no resolvable limit or null failure, stop with receipt. Report the failure without converting it into a training success.

After acceptance, calibrate both coefficient scales using the same three TRAIN batches and original gradient ratio .10, freeze receipt with bundle hash, run all six original GPU gates, then six original training arms with ES10 validation BLEU. Exact full-backbone derivatives, batch4 accumulation32 FP32, data and AdamW recipe remain unchanged. New output recipe avoids overwriting failed receipts. HF paths remain original disk-cohort paths because no earlier run produced any checkpoint there.
