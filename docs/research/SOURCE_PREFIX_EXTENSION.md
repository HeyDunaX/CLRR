# Source-prefix diagnostic: two seeds and lexical controls

## What was measured

Completed 2026-11-18 after SOURCE_PREFIX_AUDIT.md. Fixed W42 and W43 checkpoints evaluated on the same 10 validation rows, same source interventions and the same two target prefixes per row. Prefixes come from the reference-compatible template and the previously observed **W42** output; W43 was also tested under W42's prefix, not its own output prefix. No training, optimizer step, test-set access, beam generation, or new public upload.

Selection was locked before either GPU job: five manually selected rows with the source-backed lexical item present in W42 (231,94,124,226,232), and five previously observed lexical-error rows (562,424,28,75,317). “Correct” describes the selected lexical item only, not the entire translation. Contrasts such as water/tea and clothing/skirt are constructed alternatives, not independent human annotations that the alternative is semantically false in every context. Reference and W42 output were already inspected to select examples; this is exploratory and selection-dependent, not a blind or representative sample.

Sources: [nanom = water](https://new-amis.moedict.tw/terms/nanom), [fodoy = clothing](https://new-amis.moedict.tw/terms/fodoy), [koco as a shoe term](https://new-amis.moedict.tw/terms/cokap); prior four sources in SOURCE_LEXICAL_AUDIT.md. Other portions of each sentence were not independently verified by a native speaker.

Each target contrast is scored by teacher forcing from the longest common target-token prefix up to the end of the phrase. Tokenization can merge characters across the textual phrase boundary: the scored continuation may therefore include some shared prefix characters. The shared token prefix and resulting scored tokens are retained in EXTENSION_RESULTS.csv. This differs slightly from the prior phrase-boundary rule where no token crossed the boundary. Whole-sentence comparisons score all target tokens after the language code, including EOS; sum and token-mean are both recorded.

## Observations — source intact

Margin = log-score(correct lexical candidate) minus log-score(alternative); positive favors the source-backed item.

| Validation row | W42 group | Comparison | W42: reference prefix → W42 prefix | W43: same prefixes |
|---|---|---|---|---|
| 562 | error | button / trousers | −1.42 → −1.36 | −2.10 → −1.87 |
| 424 | error | disappear / emerge | −6.84 → −6.84 | −6.20 → −6.20 |
| 28 | error | mat / luggage | +1.87 → −6.67 | +4.69 → −0.35 |
| 75 | error | shoes / pocket | −9.13 → −5.84 | −11.67 → −8.37 |
| 317 | error | shoes / clothes | +3.15 → −1.09 | +3.80 → −0.29 |
| 231 | lexical control | clothes / skirt | +11.22 → +16.67 | +11.78 → +18.72 |
| 94 | lexical control | water / tea | +16.48 → +17.41 | +17.74 → +18.57 |
| 124 | lexical control | water / oil | +12.30 → +19.13 | +10.32 → +15.29 |
| 226 | lexical control | shoes / hat | +2.69 → +9.84 | +3.57 → +10.14 |
| 232 | lexical control | shoes / hat | +9.39 → +9.39 | +8.48 → +8.48 |

Both seeds show two reference-to-W42-prefix reversals toward the alternative among the five selected lexical-error rows, and none among the five controls. Rows 424 and 232 have identical two prefixes and necessarily cannot reverse. Observations for the same row across two seeds are not independent examples.

For rows 28 and 317, complete-sentence lexical substitution preserves the reversal in summed log scores for both seeds. W43 row28's token-mean comparison under the W42 template is **+0.064**, whereas its sum is **−0.357**: this case is not robust to length normalization. Row317's corresponding W43 sum/mean are −1.430/−0.111. Token counts, source removal, and all sentence scores are in the full receipts.

## Interpretation and limits

**[ĐO ĐƯỢC]** Two carefully selected error cases show direction reversal under the W42 prefix in both checkpoints; three others remain wrong-preferred regardless of prefix. All five selected controls remain correct-preferred. This extends the prior four-row observation, but does not prove a dataset-wide rate or statistical significance.

**[SUY LUẬN]** History sensitivity is a candidate phenomenon worth investigating alongside direct source-conditioned lexical errors. It is not a complete explanation. Changing the target prefix also changes context, and not every alternative is equally fluent or equally probable under both histories. Automatic token splitting and phrase length add confounds; complete-sentence sum and mean can disagree. In row317 the reference and generated prefixes express different events (going to school vs wearing wet clothes), so the switch may be normal contextual dependence. The same concern applies to the other manually constructed cases.

**[PHỎNG ĐOÁN]** A method that preserves source-conditioned meaning across meaning-compatible histories might help, but these prefixes are not established meaning-preserving paraphrases. This is not enough evidence to choose a new loss or to claim novelty. Before any causal or mechanism claim, obtain independently checked, meaning-compatible prefix pairs and matched plausible alternatives; define sample and controls without looking at their output margins. Compare with related work and, per author preference, debate the concrete candidate with NotebookLM.

## Provenance / reproduction

RTX Quadro 6000, 23,952 MiB free at each launch; seed42 then seed43 sequentially, no other processes touched. Both COMPLETE receipts confirm validation split, zero optimizer steps, unchanged parameter version counters, strict loading and checkpoint SHA verification. Plan and worker SHA verified on collection. W42 HF revision `de91462e792e66a92d51770b00cb674f4566b15d`, archive SHA `ebf1a2ebf080340e133be07cb902908d55f48f3165a4c973acfe358e0f457c74`; W43 revision `b4d32b1ceb0b6a1964e0c9b217a76303ddd6df56`, archive SHA `d486d0e2ae58d5cdd6310bcd98bd7d0ea4f1323be2c6b1df3be99c1ee05d0b73`. Validation hash checked remotely. Transient restored weights were deleted by each job on successful completion.

Local artifacts: `outputs/experiments/source_prefix_extension_s42/` and `source_prefix_extension_s43/` contain locked plan, worker, launcher and COMPLETE; `outputs/experiments/source_lexical_audit/EXTENSION_RESULTS.csv` contains the summarized margins and tokenizations. The immutable historical reports and metric tables were not changed.

Commands used in the existing conda clrr environment:

```powershell
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_lexical_audit/prepare_extension.py
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_prefix_extension_s42/operate.py launch
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_prefix_extension_s42/operate.py status
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_prefix_extension_s43/operate.py launch
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_prefix_extension_s43/operate.py status
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_lexical_audit/summarize_extension.py
```

The preparation and launch commands refuse existing output directories; they are included to document the actual run, not to imply they can be repeated in place.
