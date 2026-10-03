"""Read-only validation influence audit; never inspect candidate test outputs."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs_rebuttal/amis_configuration_20261003_v1'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    assert sacrebleu.__version__ == '2.6.0'
    raw = read(ROOT/'data_processed/amis_mandarin/validation.csv')
    refs = [row['target'].strip() for row in raw]
    results = []
    for folder in sorted((OUT/'runs').glob('*')):
        path = folder/'validation_predictions.csv'
        if not path.exists() or not (folder/'COMPLETE.json').exists():
            continue
        rows = read(path)
        assert len(rows) == len(raw) == 576
        assert [r['source'] for r in rows] == [r['source'] for r in raw]
        assert [r['target'].strip() for r in rows] == refs
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = json.loads((folder/'COMPLETE.json').read_text())
        assert sha == receipt['validation_prediction_sha256']
        metadata = json.loads((folder/'metrics.json').read_text())
        predictions = [row['prediction'].strip() for row in rows]
        result = dict(run=folder.name, prediction_sha256=sha, rows=len(rows))
        for order in (0, 2):
            metric = CHRF(word_order=order)
            stats = np.asarray(metric._extract_corpus_statistics(predictions, [refs]), dtype='float64')
            total = stats.sum(axis=0)
            score = metric.corpus_score(predictions, [refs]).score
            key = 'chrf0' if order == 0 else 'chrfpp'
            expected = metadata['eval_chrf0' if order == 0 else 'eval_chrf++']
            assert abs(score-expected) < 1e-8
            leave_out = [metric._compute_score_from_stats(total-row).score for row in stats]
            low, high = int(np.argmin(leave_out)), int(np.argmax(leave_out))
            for index in {197, low, high}:
                direct = metric.corpus_score(predictions[:index]+predictions[index+1:],
                    [refs[:index]+refs[index+1:]]).score
                assert abs(direct-leave_out[index]) < 1e-10
            result[key] = dict(score=score, without_row197=leave_out[197],
                leave_one_out_min=leave_out[low], min_removed_index=low,
                leave_one_out_max=leave_out[high], max_removed_index=high,
                signature=str(metric.get_signature()))
            if order == 2:
                result['word_bigram_reference_count'] = int(total[22])
                result['word_bigram_match_count'] = int(total[23])
        results.append(result)
    report = dict(scope='Validation-only descriptive sensitivity; primary scores retained',
        sacrebleu_version=sacrebleu.__version__, systems=results,
        ranking_chrfpp=[r['run'] for r in sorted(results,key=lambda r:-r['chrfpp']['score'])],
        ranking_without_row197=[r['run'] for r in sorted(results,key=lambda r:-r['chrfpp']['without_row197'])],
        ranking_chrf0=[r['run'] for r in sorted(results,key=lambda r:-r['chrf0']['score'])])
    (OUT/'validation_influence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
