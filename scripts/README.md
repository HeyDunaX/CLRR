# Script entry points

- `experiment.py`: shared configuration-driven A100 workflow; see
  [the runbook](../docs/operations/EXPERIMENT_WORKFLOW.md).
- `colab.py`: existing account-specific Colab CLI transport.
- `analysis/`: analysis utilities.
- `revalidation/`: retained scientific audits, probes and historical replay;
  current comparator/candidate imports delegate to `experiment_runner`.
- `nllb/` and `peft/`: low-level historical suite utilities.

Create new recipes under `configs/experiments/` and method versions under
`src/methods/`. Do not clone launch/monitor/train scripts for each new run.
