# Frozen source-cue intervention — 2026-11-18

## Execution and provenance

Author requested this diagnostic after SOURCE_LEXICAL_AUDIT.md. Completed on RTX6000: free VRAM before launch 23,952/24,576 MiB. Fixed W_S42 checkpoint, FP32 SDPA, eval/inference mode, TF32 off. Zero optimizer steps; no test read or generation, no training. Native base inference is appropriate for W (training-only weak-full consistency, strength0).

Artifacts: `outputs/experiments/source_lexical_audit/{PLAN.json,measure.py,operate.py,LAUNCH.json,COMPLETE.json,INTERVENTION_SUMMARY.csv}`. COMPLETE verifies checkpoint archive SHA, empty loading-error lists, unchanged parameter version counters, and plan/worker hashes. Local collection verified those plan/worker hashes. Disposable restore inside this diagnostic was removed after successful scoring; no original checkpoints/processes were touched. Nothing newly uploaded publicly.

Checkpoint HF revision `de91462e792e66a92d51770b00cb674f4566b15d`, archive SHA256 `ebf1a2ebf080340e133be07cb902908d55f48f3165a4c973acfe358e0f457c74`. Validation SHA256 `eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4`.

Commands (existing conda clrr Python):

```powershell
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_lexical_audit/operate.py launch
& C:/Users/Lenovo/miniconda3/envs/clrr/python.exe -X utf8 outputs/experiments/source_lexical_audit/operate.py status
```

Launch refuses a duplicate remote directory or occupied project lock. Do not rerun launch for an already-completed diagnostic.

## Design fixed before GPU

Four previously dictionary-verified validation error cases, selected because W42/43/44 had semantic substitutions. For each: original source, one source cue removed, one control word removed (punctuation preserved). Two target phrases share an identical teacher-forced target token prefix, verified by assertion. Score phrase tokens only, excluding EOS and prefix. Report both summed and per-token mean log probabilities. Total 12 source conditions / 24 candidate forward passes.

Margin M = log P(correct phrase | fixed prefix, source) − log P(wrong phrase | same prefix, source). Positive favors correct phrase. Cue effect = M(intact) − M(cue removed). Control effect defined analogously. Difference of effects = cue effect − control effect. These are log-score differences, not BLEU improvements or accuracy percentages.

| Validation row | Cue / deleted control | Target prefix | Correct / wrong |
|---|---|---|---|
|562|sakofit / ina|請媽媽縫補你衣服的|鈕扣 / 褲子|
|424|Molecep / romi'ad|太陽|消失 / 出來|
|28|sikal / kako|我已將|竹蓆 / 行李|
|75|koco / pida|他把錢藏在|鞋裡 / 口袋裡|

## Results [ĐO ĐƯỢC]

| Row / contrast | M intact | M cue removed | M control removed | Cue effect | Difference of effects |
|---|---:|---:|---:|---:|---:|
|562 button / trousers|−1.4216|−2.7427|−1.7684|+1.3211|+0.9743|
|424 disappear / emerge|−6.8393|−6.6250|−6.3736|−0.2143|+0.2515|
|28 mat / luggage|+1.8669|−4.2491|+4.7407|+6.1160|+8.9898|
|75 shoes / pocket|−9.1301|−15.0930|−9.9898|+5.9629|+5.1032|

Intact mean-per-token margins: −0.7108, −6.8393, +4.0546, −4.5650 respectively. Signs agree with summed margins. Mat has two scored tokens versus one for luggage; other pairs have equal token counts within each pair. Difference-of-effects directions also agree under token-mean normalization. This sensitivity check does not remove all length or prefix confounding.

## Interpretation

- **[ĐO ĐƯỢC]** Button and shoes cues improve the correct-versus-wrong margin more than their chosen control deletions, but neither overcomes the wrong-phrase preference. Therefore these examples do not support “source cue is entirely ignored.”
- **[ĐO ĐƯỢC]** The mat cue flips the phrase preference under this prefix. Yet previous free generation used a different prefix and produced luggage. This is a concrete conflict between fixed-prefix and free-generation behavior.
- **[SUY LUẬN]** Mat may involve a prefix/trajectory-dependent error; this diagnostic does not prove a beam-search defect or show that the complete correct translation has greater model probability.
- **[ĐO ĐƯỢC]** The sunset cue deletion has a small negative raw effect; controls also change the score. The resulting +0.2515 difference is too fragile to establish successful source grounding.
- **[SUY LUẬN]** At least two distinct behaviors are plausible: insufficient discrimination despite using source evidence, and target-prefix-dependent lexical choice. A universal “increase source attention/consistency” fix is not justified.
- **[PHỎNG ĐOÁN]** Prefix-dependent amplification could motivate a training objective only after showing this pattern beyond selected examples and separating it from ordinary teacher forcing. This is not yet a method or novelty claim.

## Limits and warning audit

Four hand-selected cases, one checkpoint, no prevalence estimate, confidence interval or significance claim. Deletion can create unnatural grammar. Controls match one deleted word but not part of speech, token count or semantic role; e.g. removing a mother/money cue can affect the fixed target prefix. They are perturbation controls, not clean causal identification. The wrong candidate may not match the exact previously generated wording. Do not infer target-prior domination merely from a negative margin.

Transformers emitted a Mistral regex warning when loading the mBART export. Inspected local `transformers/tokenization_utils_base.py`: its 4.57.6 warning path sets `mistral_config_detected=True` for a local config unless an earlier version-based return applies, including versions4.57.3–4.x with a non-Mistral model. Thus the warning alone does not identify a mBART regex defect. No `fix_mistral_regex` flag was applied and the frozen export tokenizer was used unchanged. The prior pinned-base audit established fast/slow agreement; export/base tokenizer byte identity was not newly established here.

## Next discriminating measurement, not executed

Before a method claim: compare the same object contrasts under both the original model-generated prefix and the reference-compatible prefix, then score complete candidate translations with consistent length accounting. Expand with predeclared, dictionary-backed examples and preserve controls. If the claimed prefix-specific reversal does not replicate, reject that mechanism. Any follow-up should remain validation-only and frozen-weight before new training. This audit does not authorize launching a new training queue.
