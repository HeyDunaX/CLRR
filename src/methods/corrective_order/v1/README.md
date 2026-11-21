# Corrective-order consistency v1

Exact virtual-SGD endpoint consistency. The six matched RTX arms, formulas,
calibration and execution differences are defined in
`docs/research/CORRECTIVE_ORDER_PILOT.md`.

`model.py`: objective and functional checkpoint replay.
`data.py`: shared TRAIN probes and admission.
`prepare.py --config ...`: frozen numerical/gradient calibration, no optimizer.

Train via the shared `scripts/experiment.py` entry point. No standalone trainer,
no modifications to original CLRR/LSR, no test selection in this pilot.
