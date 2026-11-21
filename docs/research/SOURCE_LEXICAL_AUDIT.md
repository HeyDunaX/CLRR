# Source lexical audit — CPU, 2026-11-18

## Scope and reproduction

Completed the two CPU follow-ups in METHOD_SEARCH_ROUND3.md: pinned tokenizer audit and bounded lexical verification. No training, GPU, test evaluation, data edits, metric recalculation, or queue changes.

Run from `D:/Code/CLRR` in conda `clrr`:

```powershell
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_lexical_audit/fetch_tokenizer.py
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_lexical_audit/analyze.py
```

Public tokenizer assets only: 5,071,658 bytes, upstream digests checked; no weights. See `outputs/experiments/source_lexical_audit/TOKENIZER_DOWNLOAD_RECEIPT.json`. Base model `facebook/mbart-large-50-many-to-many-mmt`, revision `e30b6cb8eb0d43a0b73cab73c7676b9863223a30`, as specified by `configs/experiments/amis_cmc_estimator_rtx.json` reused W42 spec. This verifies the pinned base tokenizer, not byte identity with tokenizer assets inside every trained export. W42 provenance: `outputs/experiments/amis_decoupled_consistency_rtx/runs/W_S42/VAL_COMPLETE.json`.

Runtime: MBart50TokenizerFast; transformers 4.57.6, tokenizers 0.22.2, sentencepiece 0.2.2. Source tl_XX, target zh_CN. Full input hashes in AUDIT.json. Only train (4,600) and validation (576) CSVs read; test not read.

## 1. Measured tokenizer results

| Split / side | UNK tokens | Maximum length including specials | Over 256 | Raw roundtrip differences | Content difference beyond normalization/whitespace |
|---|---:|---:|---:|---:|---:|
| Train Amis | 0 | 39 | 0 | 19 | 0 |
| Validation Amis | 0 | 33 | 0 | 3 | 0 |
| Train Chinese | 54 | 30 | 0 | 572 | 54 |
| Validation Chinese | 7 | 31 | 0 | 68 | 7 |

**[ĐO ĐƯỢC]** Fast and slow tokenizers produce identical source IDs on all 5,176 rows. Source roundtrip differences are whitespace normalization, not lost letters. Chinese differences include normalizer transformations and whitespace; 54 train rows and 7 validation rows additionally contain unsupported characters. `TARGET_UNKNOWNS.csv` lists every affected row. Validation unsupported characters: 塱 (twice), 盬, 徧, 駡, 摏, 鼈. These are vocabulary limitations; they are not evidence of an encoding failure. Do not silently replace them or change reference protocol.

**[ĐO ĐƯỢC]** 105/576 validation sentences contain at least one source subpiece ID absent from the training sources, despite zero tokenizer UNK. Vocabulary availability and fine-tuning exposure are different quantities.

| Feature versus mean W42/W43/W44 sentence chrF0 | Spearman | Partial rank correlation controlling source word count |
|---|---:|---:|
| Mean pieces per word | −0.1948 | −0.1990 |
| Fraction of source piece IDs unseen in train | −0.1471 | −0.1481 |
| Fraction of whole words unseen in train | −0.3090 | −0.3186 |

Mean of sentence-level pieces/word averages: 1.8686; median 1.8333; p90 2.25. Word spans use the previous regex (apostrophes allowed, caret splits words), and piece overlaps use full-sentence offsets. A piece spanning two regex words can count in both spans. These are descriptive features, not gold morphological segmentation. Partial correlation here controls word count only; it differs from the previous report's reference-length control.

**[SUY LUẬN]** No evidence for source UNK/truncation corruption as the broad failure explanation. Lexical exposure has a stronger association with quality than this segmentation-complexity measure. Neither correlation establishes a causal mechanism or proves segmentation harmless.

## 2. Lexical verification and train support

Rows are zero-based validation indices. Predictions: `outputs/experiments/cmc_estimator_review/shared_errors/ROWS.csv`; selected copies: `LEXICAL_CASES.csv`. Exact whole-word training support and all matching bilingual rows: `CUE_SUPPORT.csv`, `CUE_TRAIN_EXAMPLES.csv`. No automatic word alignment is claimed.

| Row | Verified source meaning in context | W42/43/44 result | Exact cue train sentences | Assessment |
|---|---|---|---:|---|
| 562 | sakofit: button; fodoy: clothing | All substitute trousers and tying | sakofit 1; fodoy 64 | Confirmed wrong object/action in this context |
| 424 | molecep with cidal: sunset/disappearance of sun | All say sun comes out | molecep 0 | Confirmed reversal of source event |
| 28 | milolod to sikal: roll up the mat | luggage / schoolbag / luggage | milolod 0; sikal 2 | Confirmed object substitution |
| 75 | koco: shoes in the cited example | All substitute pocket | koco 7 | Confirmed location/object substitution |
| 286 | Claimed infertility versus generic illness | All generic illness | ma'acicay 0 | Still reference-relative; external lexical verification incomplete |
| 303 | Claimed warm versus boiling water | All boiling | Not established for inflected cue | Still reference-relative; external lexical verification incomplete |
| 571 | Claimed soup/delicious versus dish/sweet | All dish/sweet | nisakafian 0; aredet 0 | Downgrade earlier certainty: taste term may allow sweet senses; derived soup term not independently verified here |

Dictionary sources opened/read:

- [sakofit](https://new-amis.moedict.tw/terms/sakofit): several named dictionaries agree on button; official online-dictionary section contains the same sentence as row562.
- [fodoy](https://new-amis.moedict.tw/terms/fodoy): clothing.
- [molecep](https://new-amis.moedict.tw/terms/molecep): solar-context sunset examples plus the same row424 sentence in the official section; other senses also exist.
- [milolod](https://new-amis.moedict.tw/terms/milolod): rolling a mat and the same row28 example. A different dialect list includes another sense, so context matters.
- [sikal](https://new-amis.moedict.tw/terms/sikal): mat, also other senses; interpretation uses the full cited phrase.
- [himed](https://new-amis.moedict.tw/terms/himed): official section contains the same row75 sentence with shoes.
- [cokap](https://new-amis.moedict.tw/terms/cokap): named coastal/Hengchun lists explicitly identify koco as a shoe-word variant/borrowing.
- ['aredet](https://new-amis.moedict.tw/terms/%27aredet): retrieved search fulltext lists delicious and Siwkulan sweet senses; direct page open failed. [cedem](https://new-amis.moedict.tw/terms/cedem), successfully opened, also links 'aredet to sweet/good taste. Spelling/dialect identity for row571 has not been established; no unqualified semantic-error label for sweet.

**Limits:** This is dictionary-backed verification, not annotation by an Amis speaker. Some exact examples match dataset references and may share their original source; they are not independent human judgments. Other dictionary definitions support the lexical reading, but the four hand-selected cases cannot estimate dataset-wide error prevalence. Whole-word absence does not exclude related inflections in train. Only sakofit support includes an immediately visible exact training pair with button; a cue's presence alone does not establish adequate learning signal.

## 3. What follows — no new method claimed

**[ĐO ĐƯỢC]** Several shared W errors survive external semantic checks and occur both when the exact cue is absent and when it is present in training. Thus neither “all reference noise” nor “all unseen-word failure” explains these selected cases. The seven Chinese-UNK validation rows do not overlap the four confirmed cases.

**[SUY LUẬN]** A more defensible next hypothesis is failure to condition lexical choice on a source cue, rather than missing source characters. Current measurements do not distinguish weak source evidence from an overly strong target-context preference, or insufficient learning of the cue.

**[PHỎNG ĐOÁN]** A lexical conditioning intervention might help, but a new loss/module is not justified yet. Next diagnostic, if separately authorized: fixed checkpoint and validation-only, matched target alternatives under the same prefix, compare score margins with source cue intact versus removed/replaced and matched non-cue controls. Specify length accounting and dictionary-backed alternatives before GPU use. Reject the proposed source-binding explanation if wrong alternatives remain indistinguishable from correct ones for reasons attributable to tokenization/phrase construction, or if cue changes have the expected discriminative effect and the error instead arises only during search. No such intervention was run in this audit.

Do not propose segmentation consistency as a novel algorithm: nearest prior MVR was already found in METHOD_SEARCH_ROUND3.md. This audit adds evidence and corrects one overconfident semantic label; it does not establish novelty or an ACL-ready method.

## 4. Search record

Bounded lexical searches included `sakofit`, `fodoy`, `kafer`, `molecep`, `milolod`, `sikal`, `himed`, `koco`, `cokap`, `kafi`, `'aredet`, plus exact variants/Chinese glosses. Unresolved queries included `"Ma'acicay" 不孕`, `"Ma’acicay" 阿美`, `"Saalaf" 阿美`, `"saalaf"`, and `"nisakafian" "Aredet"`; irrelevant hits were not used. Verification pages and confidence limits are listed above. No novelty search or NotebookLM debate was newly claimed in this CPU audit.
