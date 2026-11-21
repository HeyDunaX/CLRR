# Corrective-order: exact disk storage preflight

Theory and six-arm design: CORRECTIVE_ORDER_PILOT.md. No scientific recipe, data, batch, virtual update, parameter subset or objective change.

CPU saved-tensor offload failed the 6 GiB host RAM reserve before calibration completed. Zero long training runs started.

Execution amendment: spill saved autograd tensors >=1 MiB to temporary files under /workspace/ura/thanhdat/projects/CLRR/tmp/corrective-order-autograd/. Preserve 6 GiB host RAM and 40 GiB free disk. Files survive until their graph is freed. Full precision and exact higher-order gradients remain enabled; all six arms use this execution cohort.

Admission: CPU exact-gradient roundtrip tests, train-only numerical calibration, then all six GPU gates before any long training. Disk/RAM/OOM failure blocks training; do not silently change batch or method. No test evaluation. Calibration receipt is copied into each successful run and uploaded with its checkpoint.

Recipe: configs/experiments/amis_corrective_order_disk_rtx.json.
