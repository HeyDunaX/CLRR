# When the Same Layers Learn to Translate: Parameter-Neutral Residual Rewiring for Low-Resource Amis-to-Chinese Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HeyDunaX/CLRR/blob/main/notebooks/colab_a100_run.ipynb)


This repository contains reproducible experiments for a ComputEL-10 study on Amis-to-Chinese (Mandarin) neural machine translation. The proposed intervention, **Cross-Layer Residual Rewiring (CLRR)**, changes how already-computed hidden states are connected between Transformer layers. It adds no trainable parameter tensor and keeps the tokenizer, data, optimizer, and training protocol fixed.

The title makes the required formulation explicit:

- **Problem:** Amis-to-Chinese translation.
- **Specific context:** low-resource endangered-language translation.
- **Approach:** parameter-neutral residual rewiring.

The method is an experimental hypothesis, not a guaranteed improvement. This repository measures whether CLRR outperforms the corresponding unmodified backbone under a controlled protocol.

## What CLRR changes & JEPA for NMT

### 1. Context-Encoder Cross-Layer Residual Rewiring (CLRR)
Amis is a Formosan language with agglutinative and polysynthetic morphology (voice affixes, aspect markers, reduplication). In deep Transformer encoders, lower-level morphosyntactic distinctions are easily over-smoothed into broad semantic abstractions.

To preserve these granular structural cues, CLRR is applied specifically to the **Context Encoder** (source stack) at each layer index `i` from distance `d = 2`:

```text
h_i = TransformerEncoderLayer_i(h_{i-1})
h_i' = h_i + alpha * stop_gradient(h_{i-d}')
```

The residual is inserted through a deterministic forward hook with `alpha = 0.1` and `stop_gradient` detach, introducing **zero trainable parameters** while keeping the tokenizer, optimizer, and model capacity identical.

### 2. JEPA-Guided Latent Alignment for NMT
In Joint-Embedding Predictive Architectures (JEPA), learning occurs in representation space rather than solely reconstructing surface tokens. We formulate **JEPA-guided Seq2Seq**:
- **Context Representation** $H_X$: Output of the Context Encoder (Amis), optionally enhanced by CLRR.
- **Target Anchor** $H_Y$: Encoded target sequence (Chinese) computed under `stop-gradient`.
- **Training Loss**:
  $$\mathcal{L} = \mathcal{L}_{\text{NMT}} + \lambda_{\text{JEPA}} \cdot (1 - \text{CosineSimilarity}(\bar{H}_X, \bar{H}_Y))$$
  with $\lambda_{\text{JEPA}} = 0.1$. This enforces latent semantic alignment between source and target while the decoder generates surface tokens to compute standard BLEU and chrF++.

## Models & Evaluation Protocol (Focused 6-Run Setup)

To maximize scientific rigor while respecting workshop compute budgets, we adopt a **focused 6-run experiment protocol**:
1. **Central Research Backbone (`google/mt5-small`)**: Undergoes a full 4-condition ablation study to isolate the contribution of Context-Encoder residual rewiring vs. JEPA representation alignment.
2. **Cross-Architecture Validation Backbone (`facebook/mbart-large-50-many-to-many-mmt`)**: Evaluates the generalization of the proposed method (Baseline vs. JEPA + CLRR-Enc).

| Backbone | Role | Conditions |
| --- | --- | --- |
| `google/mt5-small` | Central research backbone (full ablation) | Baseline, CLRR-Enc, JEPA, JEPA+CLRR-Enc |
| `facebook/mbart-large-50-many-to-many-mmt` | Cross-architecture validation | Baseline, JEPA+CLRR-Enc |

For mBART, Amis has no dedicated mBART-50 language ID. The tokenizer uses its shared vocabulary and Chinese (`zh_CN`) as the target language tag. This limitation must be reported in the paper.

## Data

Processed datasets are organized under `data_processed/<dataset_name>/`:
`amis_mandarin` contains the original Amis splits; `ashaninka_spanish` contains
the prepared AmericasNLP splits (3,883 train / 881 validation / 1,003 test).
See [dataset sources, manifests, and language configuration notes](data_processed/README.md).
Run local Python commands in the existing conda environment `clrr`.

```powershell
conda run -n clrr python scripts/prepare_ashaninka_spanish.py
```

Asháninka data preparation is complete. Its training configuration still needs
Spanish language tags and evaluation settings; changing `--data-dir` alone is
insufficient with the current Mandarin-specific trainers.

The parallel corpus used in this study is the 5,751-sentence Amis–Mandarin dataset introduced by [Zheng et al. (2022)](https://aclanthology.org/2022.nlp4dh-1.11/). The data archive can be downloaded from Google Drive:

- **Google Drive Folder**: [Amis–Chinese Dataset](https://drive.google.com/drive/folders/1W-glGBpCz9R16Oy-P96jdK7YGSVuvdg2)

Download `parallel.zip` from the Drive folder into the repository root (or pass `--zip-path` to `scripts/unpack_parallel_data.py`). The archive contains pre-split parallel text files:

```text
parallel-data/ami.train  (4600 sentences)  ←→  parallel-data/cmn.train  (4600 sentences)
parallel-data/ami.dev    (576 sentences)   ←→  parallel-data/cmn.dev    (576 sentences)
parallel-data/ami.test   (575 sentences)   ←→  parallel-data/cmn.test   (575 sentences)
```

Run `scripts/unpack_parallel_data.py` to convert the aligned text files into the CSV format expected by the training pipeline:

```csv
source,target
AMIS_SENTENCE,CHINESE_TRANSLATION
```

Use `data/amis_chinese.example.csv` only as a schema example. Do not train on its placeholder rows. The preparation script validates all three splits and writes `data_processed/amis_mandarin/{train,validation,test}.csv`. It runs on CPU and does not require CUDA.

## Installation

Requires Python 3.10–3.12 and a CUDA-capable GPU (Ampere or newer recommended for BF16/TF32 acceleration).

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
pip install -e .
```

If you already have a CUDA-matched PyTorch and want to avoid reinstalling it,
install only the remaining dependencies:

```bash
pip install -r requirements-colab.txt
pip install -e . --no-deps
```

Verify the environment:

```bash
python -c "
import torch, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('bf16', torch.cuda.is_bf16_supported())
"
python -m amis_rewire.train --help
```

## Prepare the CSV once before pushing the repository

Run this once after extracting the parallel data from `parallel.zip`:

```bash
python scripts/unpack_parallel_data.py \
  --zip-path parallel.zip \
  --output-dir data_processed/amis_mandarin
```

Alternatively, if you have a single CSV with columns `amis,chinese,split`:

```bash
python -m amis_rewire.prepare_data \
  --input-csv data/amis_chinese.csv \
  --output-dir data_processed/amis_mandarin \
  --source-col amis \
  --target-col chinese \
  --split-col split
```

The command validates UTF-8, removes empty pairs, normalizes split names, requires all three splits, and writes `data_processed/amis_mandarin/{train,validation,test}.csv`. No tokenizer or language-specific preprocessing is performed. This step is CPU-only and does not require CUDA; only model training and generation use the GPU.

## One experiment

Run the proposed method (JEPA-guided Seq2Seq with Context-Encoder CLRR):

```bash
python -m amis_rewire.train \
  --model mt5-small \
  --method jepa-clrr-enc \
  --rewire-stack encoder \
  --jepa-weight 0.1 \
  --data-dir data_processed/amis_mandarin \
  --output-dir outputs \
  --seed 42 \
  --backup-dir backups \
  --num-train-epochs 20 \
  --early-stopping-patience 4 \
  --learning-rate 3e-4 \
  --warmup-ratio 0.06 \
  --per-device-train-batch-size 128 \
  --per-device-eval-batch-size 128 \
  --gradient-accumulation-steps 1 \
  --max-source-length 256 \
  --max-target-length 256 \
  --num-beams 4 \
  --eval-beams 1 \
  --rewire-distance 2 \
  --rewire-strength 0.1 \
  --bf16 \
  --no-gradient-checkpointing \
  --dataloader-num-workers 4 \
  --dataloader-pin-memory
```

Supported `--method` options:
- `baseline`: Standard Seq2Seq (Cross-Entropy loss only).
- `clrr-enc`: Context-Encoder CLRR without JEPA latent loss.
- `jepa`: JEPA-guided Seq2Seq without residual rewiring.
- `jepa-clrr-enc`: Proposed method (JEPA representation alignment + Context-Encoder CLRR).

## Full model matrix

```bash
bash scripts/run_all_models.sh
```

The script runs six conditions across two backbones (mT5-small and mBART-50)
with seed `42`. Its defaults are a maximum of 20 epochs, validation after every
epoch, and early stopping after four non-improving validation checks. The
learning rate is `3e-4` for mT5-small and `5e-5` for mBART-50. Completed runs
are skipped automatically; an interrupted run resumes from its latest checkpoint.

### Follow-up Colab experiments

After pushing the repository changes, open
[the follow-up A100 notebook](notebooks/colab_followup_run.ipynb) and add a Colab
Secret named `HF_TOKEN` with access to the private
`FiveC/amis-rewire-checkpoints` model repository. The notebook first checks the
saved main-run hyperparameters and regenerates test predictions and encoder
cosine measurements. It then runs two ByT5-small conditions and two mT5-small
decoder/both rewiring ablations. The main six-run matrix is not rerun.
For the follow-up tables, all saved checkpoints and new models are scored the
same way: decode predictions, pair them with the original `target` text in all
575 rows of `data_processed/amis_mandarin/test.csv`, and compute SacreBLEU
`BLEU(tokenize="zh")` and `CHRF(word_order=2)` (chrF++). Leading and trailing
whitespace is stripped from predictions and targets. Scores are recomputed
from predictions; matching the historical score files is not a run condition.

New checkpoints, best-model ZIPs, metrics, and test predictions are uploaded to
`checkpoints/<run-name>/` in that private repository. Analysis tables, case
candidates, and the encoder-cosine figure are uploaded to `analysis/`.

Checkpoints are saved locally under `outputs/<run-name>/checkpoint-*` and are
also zipped after every save under `backups/<run-name>/`. To use a custom backup
location:

```bash
BACKUP_DIR=/path/to/backups bash scripts/run_all_models.sh
```

At startup, `run_all_models.sh` restores the newest ZIP checkpoint for each
incomplete run before invoking the trainer. An interrupted run therefore resumes
from the latest checkpoint automatically.

Run and artifact names follow the six-condition matrix:

```text
mt5-small-ami-cmn-baseline/
mt5-small-ami-cmn-clrr-enc/
mt5-small-ami-cmn-jepa/
mt5-small-ami-cmn-jepa-clrr-enc/
mbart-large-50-ami-cmn-baseline/
mbart-large-50-ami-cmn-jepa-clrr-enc/
```

After each run, the selected best checkpoint is saved under
`outputs/<run-name>/best_model/` and archived as
`backups/<run-name>/<run-name>-best.zip`. The selected checkpoint is the model
chosen by validation chrF++.

For remote backup, set `HF_BACKUP_REPO` to a private Hugging Face model
repository and export `HF_TOKEN`:

```bash
export HF_TOKEN=<your-token>
export HF_BACKUP_REPO=<account>/amis-rewire-checkpoints
bash scripts/run_all_models.sh
```

The training callback uploads each checkpoint ZIP to that repository; the runner
restores remote checkpoints before resuming.

To continue one interrupted run explicitly, rerun its command with
`--auto-resume`; this is the default. To disable it, pass `--no-auto-resume`.

## Outputs and logging

Each run writes to `outputs/<run-name>/` and produces `metrics.json`. During training, stdout reports:

```text
[step=...] loss=...
[step=...] eval_loss=... | eval_bleu=... | eval_chrf++=...
...
{
  "test_loss": ...,
  "test_bleu": ...,
  "test_chrf++": ...
}
```

The primary automatic metrics are **BLEU** and **chrF++**. BLEU gives comparability with MT work; chrF++ is important for low-resource and morphologically rich settings. COMET may be added as a secondary analysis only after verifying an appropriate model and validity for Amis; it is not part of the fixed main protocol.

## Main results table

The six main runs are complete. Scores below are the historical test-set results.
The follow-up analysis keeps this table unchanged and reports its uniformly
recomputed scores separately in `analysis/all_scores.csv`.

| Model | Method | Role | BLEU | chrF++ |
| --- | --- | --- | ---: | ---: |
| mT5-small | Baseline | Standard Seq2Seq (CE) | 2.81 | 4.49 |
| mT5-small | CLRR-Enc | Context-Encoder Rewiring only | 4.50 | 5.90 |
| mT5-small | JEPA | Latent Alignment only | 4.66 | 5.224 |
| mT5-small | JEPA + CLRR-Enc | Proposed Method | 5.17 | 5.76 |
| mBART-50 | Baseline | Translation Baseline | 20.09 | 15.72 |
| mBART-50 | JEPA + CLRR-Enc | Cross-Architecture Validation | 20.81 | 16.56 |

## Main six-run protocol

- seed and data seed: `42`;
- full-model fine-tuning for baseline and CLRR;
- AdamW from `Seq2SeqTrainer`; learning rate `3e-4` for mT5-small and `5e-5` for mBART-50;
- maximum 20 epochs, warmup ratio `0.06`, early stopping patience `4`;
- effective batch size `128` on one A100 GPU (train batch size 128, gradient accumulation 1); evaluation batch size `128`;
- A100 acceleration: BF16, TF32 matmul, fused AdamW, pinned-memory dataloaders,
  and four dataloader workers; gradient checkpointing disabled;
- source/target truncation: `256/256` tokens;
- beam size: `1` during validation and `4` for final test generation;
- best checkpoint selected by validation chrF++;
- no data augmentation, tokenizer changes, prompt tokens, LoRA, or adapters;
- test set is evaluated once after model selection.

## Repository layout

```text
docs/
├── EXPERIMENTS.md           # Full 16-model matrix, mBART ablation suite, Task 1-5 revalidation
├── RESEARCH_INSIGHTS.md     # Theory, stop-gradient math derivation, ByT5 UTF-8 analysis, case studies
├── COLAB_SSH_GUIDE.md       # Google Colab SSH workflows, autonomous bash scripts, templates
└── archive/                 # Historical phase documentation (backed up)
src/amis_rewire/
├── modeling.py              # CLRR hooks, stop-gradient routing, model loading
├── train.py                 # CLI, fixed fairness protocol, training, validation, test
├── metrics.py               # Standardized SacreBLEU (tokenize='zh') and chrF++ (word_order=2)
└── prepare_data.py          # CSV validation and split export
src/comparative_baselines/   # Isolated implementations of LayerSkip (ACL 2024) and Mid-Align (ACL 2025)
data_processed/
├── amis_mandarin/            # train.csv (4600), validation.csv (576), test.csv (575)
└── ashaninka_spanish/        # train.csv (3883), validation.csv (881), test.csv (1003)
scripts/
├── run_all_models.sh        # Batch execution matrix
└── revalidation/            # Targeted defense & ablation scripts
```

## Data citation

This work uses the Amis–Mandarin parallel corpus introduced by:

> Francis Zheng, Edison Marrese-Taylor, and Yutaka Matsuo. 2022.
> [A Parallel Corpus and Dictionary for Amis-Mandarin Translation](https://aclanthology.org/2022.nlp4dh-1.11/).
> In *Proceedings of the 2nd International Workshop on Natural Language Processing for Digital Humanities*, pages 79–84, Taipei, Taiwan. Association for Computational Linguistics.

## Reproducibility and limitations

The method adds no parameters but changes hidden-state values, so it still needs to be evaluated for optimization stability and numerical overhead. Forward hooks are intentionally simple, but wall-clock latency and GPU memory must be reported. The Amis corpus is small, domain-limited, speaker-limited, and sensitive to documentation ethics. Translation quality must not be presented as a substitute for community or linguist validation.

## Citation

```bibtex
@inproceedings{amis-rewire-computel10,
  title = "When the Same Layers Learn to Translate: Parameter-Neutral Residual Rewiring for Low-Resource Amis-to-Chinese Translation",
  author = "Anonymous",
  booktitle = "Proceedings of the Tenth Workshop on the Use of Computational Methods in the Study of Endangered Languages",
  year = "2027",
  note = "Under review"
}
```
