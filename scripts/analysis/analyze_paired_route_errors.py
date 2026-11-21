"""Descriptive validation error audit; never reads test or changes model selection."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from sacrebleu.metrics import BLEU, CHRF


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def chars(text):
    # Han characters only: character frequency, NOT word/subword frequency.
    return re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate, train, validation = map(read, [args.baseline, args.candidate, args.train, args.validation])
    if not len(baseline) == len(candidate) == len(validation) or not validation:
        raise ValueError('Missing or unequal validation rows')
    for rows in (baseline, candidate):
        if [(x['source'], x['target']) for x in rows] != [(x['source'], x['target']) for x in validation]:
            raise ValueError('Source/reference row alignment failed')
    bleu, chrf = BLEU(tokenize='zh'), CHRF(word_order=2)
    frequency = Counter(c for row in train for c in chars(row['target']))
    rows = []
    for i, (ref, b, c) in enumerate(zip(validation, baseline, candidate)):
        target = ref['target']
        n = len(chars(target))
        bscore = chrf.sentence_score(b['prediction'], [target]).score
        cscore = chrf.sentence_score(c['prediction'], [target]).score
        rows.append(dict(row=i, reference_han=n,
                         length_bin='0-10' if n <= 10 else '11-20' if n <= 20 else '21+',
                         rare_reference=any(frequency[x] <= 5 for x in chars(target)),
                         baseline_chrf=bscore, candidate_chrf=cscore, delta_chrf=cscore-bscore,
                         identical=b['prediction'] == c['prediction'],
                         baseline_han=len(chars(b['prediction'])), candidate_han=len(chars(c['prediction'])),
                         baseline_exact=b['prediction'] == target, candidate_exact=c['prediction'] == target))

    def summarize(indices):
        refs = [validation[i]['target'] for i in indices]
        result = dict(n=len(indices))
        for name, predictions in [('baseline', baseline), ('candidate', candidate)]:
            hyp = [predictions[i]['prediction'] for i in indices]
            result[name] = dict(bleu=bleu.corpus_score(hyp, [refs]).score,
                                chrf=chrf.corpus_score(hyp, [refs]).score,
                                exact=sum(h == r for h, r in zip(hyp, refs)),
                                han_length_ratio=sum(len(chars(h)) for h in hyp)/max(1, sum(len(chars(r)) for r in refs)))
        result['delta_bleu'] = result['candidate']['bleu']-result['baseline']['bleu']
        result['delta_chrf'] = result['candidate']['chrf']-result['baseline']['chrf']
        result['sentence_chrf_wins'] = sum(rows[i]['delta_chrf'] > 1e-9 for i in indices)
        result['sentence_chrf_losses'] = sum(rows[i]['delta_chrf'] < -1e-9 for i in indices)
        return result

    groups = {'all': list(range(len(rows)))}
    for key in ['0-10', '11-20', '21+']:
        groups['reference_han_'+key] = [i for i, r in enumerate(rows) if r['length_bin'] == key]
    for value in [False, True]:
        groups['rare_reference_'+str(value)] = [i for i, r in enumerate(rows) if r['rare_reference'] == value]
    summary = {key: summarize(ids) for key, ids in groups.items() if ids}
    changes = dict(identical=sum(r['identical'] for r in rows),
                   exact_repairs=sum(not r['baseline_exact'] and r['candidate_exact'] for r in rows),
                   exact_breaks=sum(r['baseline_exact'] and not r['candidate_exact'] for r in rows))
    provenance = {key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                  for key, path in [('baseline', args.baseline), ('candidate', args.candidate), ('train', args.train), ('validation', args.validation)]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(dict(provenance=provenance, groups=summary, changes=changes,
        bleu_signature=str(bleu.get_signature()), chrf_signature=str(chrf.get_signature())), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (args.output/'sentence_changes.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ['# Baseline vs candidate: descriptive validation audit', '',
              'Fixed predictions, original references; no test, no checkpoint reselection.',
              'Bins are descriptive after model selection, not hypothesis tests. Han frequency is not token/word rarity.', '',
              '| Group | n | Baseline BLEU | Candidate BLEU | Delta BLEU | Delta chrF++ | sentence wins/losses |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, result in summary.items():
        report.append(f"| {name} | {result['n']} | {result['baseline']['bleu']:.4f} | {result['candidate']['bleu']:.4f} | {result['delta_bleu']:+.4f} | {result['delta_chrf']:+.4f} | {result['sentence_chrf_wins']}/{result['sentence_chrf_losses']} |")
    report += ['', f'Changes: `{json.dumps(changes)}`', '',
               'Subset corpus scores are not additive. One validation seed cannot establish stability or causal error mechanisms.']
    (args.output/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps(dict(groups=summary, changes=changes), indent=2))


if __name__ == '__main__':
    main()
