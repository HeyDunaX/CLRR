"""Audit Mandarin reference word statistics before interpreting validation tuning."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from sacrebleu.metrics import CHRF

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs_rebuttal/amis_configuration_20261003_v1'


def main():
    report={}
    metric=CHRF(word_order=2)
    for split in ('validation','test'):
        path=ROOT/'data_processed/amis_mandarin'/(split+'.csv')
        with path.open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        refs=[r['target'].strip() for r in rows]
        stats=np.asarray(metric._extract_corpus_statistics(refs,[refs]),dtype='float64')
        assert stats.shape[1]==24
        unigram_counts=stats[:,19]
        bigram_counts=stats[:,22]
        report[split]={'rows':len(rows),'reference_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'word_unigrams':int(unigram_counts.sum()),'word_bigrams':int(bigram_counts.sum()),
            'rows_with_word_bigrams':np.flatnonzero(bigram_counts).tolist(),
            'interpretation':'Word-ngram sparsity is a selection/metric sensitivity flag, not proof of invalid data or model errors.'}
    OUT.mkdir(exist_ok=True)
    (OUT/'reference_word_statistics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
