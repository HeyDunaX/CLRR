# Workspace execution

- Run Python commands, dataset preparation, tests, and training in the existing
  conda environment `clrr` (for example, `conda run -n clrr python ...`).
- Processed datasets live under `data_processed/<dataset_name>/` with
  `train.csv`, `validation.csv`, and `test.csv`, each containing `source,target`.
