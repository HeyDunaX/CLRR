# Frozen prefix and complete-sentence diagnostic — 2026-11-18

## Scope / reproducibility

Completed the author-approved follow-up to SOURCE_CUE_INTERVENTION.md. Same W42 checkpoint and four dictionary-checked validation error cases. No optimizer, training, test access, beam generation, protocol edits, or public upload. RTX6000 before launch: 23,952/24,576 MiB free; project lock free. One detached, lock-owning job completed and removed only its disposable checkpoint restore.

Artifacts in `outputs/experiments/source_prefix_audit/`: PLAN.json, measure.py, operate.py, LAUNCH.json, COMPLETE.json, summarize.py, PHRASE_SUMMARY.csv, SENTENCE_SUMMARY.csv. The previous audit remains unchanged.

Checkpoint revision `de91462e792e66a92d51770b00cb674f4566b15d`, SHA256 `ebf1a2ebf080340e133be07cb902908d55f48f3165a4c973acfe358e0f457c74`. Native W inference; FP32/SDPA, TF32 off, dropout off. Empty checkpoint-loading error lists; parameter version counters unchanged. Plan/worker SHA256 verified on collection. Validation file hash checked remotely. PLAN also records the hash of the local predictions CSV from which W42 prefixes were extracted; prefix text was checked against those existing predictions before deployment.

Commands, existing conda clrr Python:

```powershell
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_prefix_audit/operate.py launch
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_prefix_audit/operate.py status
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_prefix_audit/summarize.py
```

Do not repeat launch: operator refuses an existing remote diagnostic directory. CPU preflight checked syntax, deletion matches, identical target-prefix IDs within each candidate pair, no boundary-crossing token and no candidate UNK. Worker repeats tokenizer assertions using the actual checkpoint export tokenizer.

## Locked comparisons

For each case, compare identical correct/wrong lexical alternatives under (a) the previous reference-compatible prefix and (b) the prefix from existing W42 free generation. The sunset prefixes are identical, providing an internal identity check. No new prefix selection after GPU results.

Also score four complete candidates: original reference, reference with the wrong lexeme substituted, original W42 output, and W42 output with the correct lexeme substituted. Score all target tokens after zh_CN, including EOS. Report sum and mean log-probability per scored token. Same three source conditions as previous audit: intact, cue removed, control removed. Total 96 forwards; these are teacher-forced scoring calls, not newly generated translations.

Margin = log-score(correct candidate) minus log-score(wrong candidate). Positive means correct favored. Sentence means and sums are descriptive; they are not BLEU/chrF scores or calibrated semantic probabilities.

## 1. Phrase results, source intact [ĐO ĐƯỢC]

| Row / correct versus wrong | Reference-compatible prefix | W42-generated prefix | Margin under reference prefix | Margin under generated prefix |
|---|---|---|---:|---:|
|562 button / trousers|請媽媽縫補你衣服的|請媽媽將你衣服的|−1.4216|−1.3643|
|424 disappearance / emergence|太陽|太陽|−6.8393|−6.8393|
|28 mat / luggage|我已將|我已整理好了|+1.8669|−6.6711|
|75 shoes / pocket|他把錢藏在|錢放在他的|−9.1301|−5.8362|

Mean-per-token margin signs agree in all four cases. Reference-prefix scores reproduce the previous GPU audit exactly at the saved float precision. Identical sunset prefixes give identical scores, as expected.

## 2. Complete-sentence results, source intact [ĐO ĐƯỢC]

| Row | Reference minus original W42 output | Correct minus wrong lexeme within reference template | Corrected minus original W42 output within W42 template |
|---|---:|---:|---:|
|562|−4.0749|−1.9005|−5.8120|
|424|−22.8211|−3.3507|−9.5363|
|28|−13.5162|+3.1840|−6.6770|
|75|−29.5349|−8.1695|−6.1211|

All signs also agree under token-mean normalization. Complete candidate strings, token counts, token-level log probabilities, source perturbations and means are saved. Mat/luggage uses unequal token counts (correct phrase2, wrong phrase1); normalization agreement limits but does not eliminate length effects. Reference-versus-W42 differs in multiple words and sometimes length/punctuation, so it is not the controlled lexical comparison.

## Interpretation and falsification

1. **[ĐO ĐƯỢC]** Mat preference reverses with the target prefix, and the reversal survives scoring the remaining sentence and EOS within each template. The original provisional interpretation was not merely an artifact of stopping at the phrase boundary.
2. **[ĐO ĐƯỢC]** In the other three examples, the correct lexeme loses under both prefixes and both complete sentence templates. Prefix choice alone does not explain all observed errors.
3. **[ĐO ĐƯỢC]** All four original W42 outputs have higher model score than their references under both sum and token mean. These particular alternatives do not support the claim that search simply failed to retrieve a higher-scoring reference. This does not rule out other correct paraphrases or all search errors.
4. **[SUY LUẬN]** Evidence points to learned scoring preferences, with a prefix-dependent component in one case. Combined with the previous cue-deletion result, “entirely ignores source” and “just increase beam width” are inadequate universal explanations.
5. **[PHỎNG ĐOÁN]** A useful research target could be preserving lexical discrimination across meaning-compatible target histories. This is a mechanism hypothesis, not yet a novel algorithm. Different prefixes also change syntax/semantics and can be unequally compatible with the two phrases. Do not call the effect pathological without controls on normal examples.

## Limits / next gate

Four examples were selected for failure, one trained seed, several hand-constructed lexical substitutions, no normal-example sample, no matched fluency judgments. No population-level significance or error prevalence claim. Source deletion is not a guaranteed grammatical intervention. The inherited Mistral-regex warning and its inspected Transformers4.57.6 detection behavior are documented in SOURCE_CUE_INTERVENTION.md; tokenizer was not modified.

Next gate before designing/training a new objective: predeclare broader dictionary-backed lexical contrasts with meaning-compatible prefixes, include correctly translated controls and at least another fixed seed, and test whether harmful prefix-dependent reversal is enriched among errors. Reject a general prefix-instability explanation if normal examples show the same behavior or if the effect vanishes after fluency/length controls. Related-work audit must compare any eventual objective with exposure-bias, target-history robustness and lexical contrastive training; no novelty assessment was newly claimed here. No new training queue was launched.
