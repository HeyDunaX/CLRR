"""Reviewed train-pair contracts and collation without changing CE sampling."""
import hashlib
import json
from pathlib import Path
import random

import torch
from transformers import DataCollatorForSeq2Seq


def require_scientific_admission(plan):
    admission = plan.get('scientific_admission', {})
    if admission.get('novelty') != 'PASS' or admission.get('semantic_pairs') != 'PASS':
        raise ValueError('Difference candidate scientific gates have not passed; do not launch GPU/train')


def read_pairs(path, train_path, rows, mode, seed):
    """Fail closed: lexical discovery output is NOT a reviewed pair manifest."""
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = hashlib.sha256(Path(train_path).read_bytes()).hexdigest()
    if record.get('split') != 'train' or record.get('train_sha256') != digest:
        raise ValueError('Pair manifest must pin the original train.csv')
    if record.get('status') != 'SEMANTIC_REVIEW_PASS' or not record.get('pairs'):
        raise ValueError('Semantic train-pair review has not passed')
    if mode not in ('generic', 'candidate'):
        raise ValueError('Unknown pairing control')
    peers, reviewed_neighbors = {}, {}
    for pair in record['pairs']:
        i, j = pair.get('left'), pair.get('right')
        if (type(i) is not int or type(j) is not int or i == j
                or not 0 <= i < len(rows) or not 0 <= j < len(rows)):
            raise ValueError('Invalid train row pair')
        if (pair.get('meaning_difference_valid') is not True
                or pair.get('translation_correspondence_valid') is not True
                or not isinstance(pair.get('reviewer'), str) or not pair['reviewer'].strip()):
            raise ValueError('Missing bilingual semantic review')
        if any(not rows[k][column].strip() for k in (i, j) for column in ('source', 'target')):
            raise ValueError('Empty reviewed source/target')
        if any(rows[i][column].strip() == rows[j][column].strip() for column in ('source', 'target')):
            raise ValueError('Reviewed pair must differ on both sides')
        peers[i] = min(peers.get(i, j), j)
        peers[j] = min(peers.get(j, i), i)
        reviewed_neighbors.setdefault(i, set()).add(j)
        reviewed_neighbors.setdefault(j, set()).add(i)
    if mode == 'generic':
        controls = {}
        for anchor in sorted(peers):
            choices = [j for j, row in enumerate(rows)
                if j != anchor and j not in reviewed_neighbors[anchor]
                and all(row[k].strip() and row[k].strip() != rows[anchor][k].strip()
                        for k in ('source', 'target'))]
            if not choices:
                raise ValueError('Cannot construct distinct generic peer')
            controls[anchor] = random.Random(seed + 10000 + anchor).choice(choices)
        peers = controls
    return peers


class DifferenceCollator:
    def __init__(self, tokenizer, model, pad_to_multiple_of=8):
        self.base = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                         pad_to_multiple_of=pad_to_multiple_of)

    def __call__(self, features):
        has_peers = ['peer_input_ids' in row for row in features]
        if not any(has_peers):
            return self.base(features)
        if not all(has_peers):
            raise ValueError('Mixed training/evaluation features')
        anchors, peers, eligible = [], [], []
        for row in features:
            anchors.append({k: v for k, v in row.items()
                            if not k.startswith('peer_') and k != 'pair_eligible'})
            peers.append(dict(input_ids=row['peer_input_ids'],
                attention_mask=row['peer_attention_mask'], labels=row['peer_labels']))
            eligible.append(row['pair_eligible'])
        result, other = self.base(anchors), self.base(peers)
        for key in ('input_ids', 'attention_mask', 'labels'):
            result['peer_' + key] = other[key]
        result['pair_eligible'] = torch.tensor(eligible, dtype=torch.bool)
        return result
