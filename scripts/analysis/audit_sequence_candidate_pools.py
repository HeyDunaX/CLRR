"""Reference-dependent validation pool audit; no training or deployable selector.

Uses existing beam1/beam4 outputs. A per-sentence chrF0 oracle is an upper
bound on mean sentence chrF0 within that exact pool, not on corpus BLEU,
human quality, or a trained model's achievable improvement.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beam1-inputs', type=Path, required=True)
    parser.add_argument('--token-diagnostics', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    root = args.token_diagnostics
    plan = read(root / 'PLAN.json')
    assert plan['split'] == 'validation' and plan['optimizer_steps'] == 0
    assert digest(args.references) == plan['data']['validation']['sha256']
    refs = pd.read_csv(args.references, dtype=str, keep_default_na=False)
    assert len(refs) == plan['rows']
    targets = refs.target.str.strip().tolist()
    chrf = CHRF(word_order=0)
    bleu = BLEU(tokenize='zh')
    systems, sentence, inputs = {}, {}, {}
    for ident in ['R1', 'P1', 'R2']:
        result = read(root / ident / 'RESULT.json')
        assert result['selection'] == plan['selections'][ident]
        for beam, path, expected in [
            (1, args.beam1_inputs / ident / 'validation_predictions.csv', plan['beam1_inputs'][ident]['predictions_sha256']),
            (plan['beams'], root / ident / 'validation_beam_predictions.csv', result['file_sha256']['validation_beam_predictions.csv']),
        ]:
            assert digest(path) == expected
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            assert frame[['source', 'target']].equals(refs[['source', 'target']])
            name = f'{ident}_beam{beam}'
            systems[name] = frame.prediction.str.strip().tolist()
            sentence[name] = np.array([chrf.sentence_score(hyp, [ref]).score for hyp, ref in zip(systems[name], targets)])
            inputs[name] = {'path': str(path), 'sha256': digest(path)}
            frozen = result['beam1_scores'] if beam == 1 else result['beam_scores']
            assert abs(bleu.corpus_score(systems[name], [targets]).score - frozen['bleu']) < 1e-8
            assert abs(chrf.corpus_score(systems[name], [targets]).score - frozen['chrf0']) < 1e-8
    pools = {f'{ident}_own_beams': [f'{ident}_beam1', f'{ident}_beam{plan["beams"]}'] for ident in ['R1', 'P1', 'R2']}
    pools.update({f'R1_plus_{ident}': pools['R1_own_beams'] + pools[f'{ident}_own_beams'] for ident in ['P1', 'R2']})
    summary = {}
    rows = []
    base_scores = sentence[f'R1_beam{plan["beams"]}']
    for pool, members in pools.items():
        values = np.stack([sentence[name] for name in members], axis=1)
        chosen = values.argmax(axis=1)  # deterministic tie: declared member order
        hypotheses = [systems[members[index]][row] for row, index in enumerate(chosen)]
        oracle = values.max(axis=1)
        unique = [len({systems[name][row] for name in members}) for row in range(len(refs))]
        summary[pool] = dict(members=members, nominal_candidates=len(members),
            mean_unique_candidates=float(np.mean(unique)), rows_with_multiple_candidates=int(np.sum(np.array(unique) > 1)),
            mean_sentence_chrf0=float(oracle.mean()),
            descriptive_corpus_chrf0=chrf.corpus_score(hypotheses, [targets]).score,
            descriptive_corpus_bleu=bleu.corpus_score(hypotheses, [targets]).score,
            mean_sentence_chrf0_gain_vs_baseline_beam4=float((oracle - base_scores).mean()),
            baseline_beam4_better_rows=int((oracle < base_scores - 1e-12).sum()))
        if 'R1_beam4' in members:
            assert (oracle >= base_scores - 1e-12).all()
        for row in range(len(refs)):
            rows.append(dict(pool=pool, row=row, unique_candidates=unique[row],
                selected_system=members[chosen[row]], oracle_sentence_chrf0=oracle[row],
                baseline_beam4_sentence_chrf0=base_scores[row]))
    result = dict(status='PASS', split='validation', rows=len(refs),
        reference_sha256=digest(args.references), input_files=inputs, pools=summary,
        native_mean_sentence_chrf0={name: float(values.mean()) for name, values in sentence.items()},
        uses_reference_to_select=True, deployable=False, optimizer_steps=0, test_evaluated=False,
        limitation='Oracle bound on mean sentence chrF0 for existing outputs only; no attainable/BLEU/human-quality bound.',
        source_sha256=digest(__file__))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / 'oracle_rows.csv', index=False)
    (args.output / 'POOL_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'pools': summary}, indent=2))


if __name__ == '__main__':
    main()
