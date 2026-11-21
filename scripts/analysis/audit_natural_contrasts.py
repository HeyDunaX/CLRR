"""CPU-only audit of existing natural pairs and a fixed, reference-relative sample."""
from collections import Counter, defaultdict
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='outputs/experiments/method_search_round2')
args = parser.parse_args()
OUT = ROOT / args.output
OUT.mkdir(parents=True, exist_ok=True)
manifest = {}


def read(relative):
    path = ROOT / relative
    manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, keep_default_na=False)


def source_words(text):
    return re.findall(r"[^\W\d_]+(?:[’'][^\W\d_]+)*", text.lower())


def normalized(text):
    return ''.join(char.lower() for char in text if char.isalnum())


def pairs(frame):
    groups = defaultdict(list)
    for row, text in enumerate(frame.source):
        tokens = source_words(text)
        for position, token in enumerate(tokens):
            key = (len(tokens), position, tuple(tokens[:position]), tuple(tokens[position + 1:]))
            groups[key].append((row, token))
    result = {}
    for key, members in groups.items():
        # Same deterministic representative per normalized word as the initial audit.
        representatives = {word: row for row, word in members}
        for (word_a, row_a), (word_b, row_b) in combinations(sorted(representatives.items()), 2):
            if normalized(frame.iloc[row_a].target) == normalized(frame.iloc[row_b].target):
                continue
            result[tuple(sorted((row_a, row_b)))] = (word_a, word_b, key)
    return result


train = read('data_processed/amis_mandarin/train.csv')
validation = read('data_processed/amis_mandarin/validation.csv')
train_pairs = pairs(train)
validation_pairs = pairs(validation)
edit_frames = defaultdict(set)
for word_a, word_b, key in train_pairs.values():
    edit_frames[(word_a, word_b)].add(key)

predictions = {}
for name, relative in {
    'baseline42': 'outputs/experiments/amis_residual_seed_diagnostics/inputs/R1/validation_predictions.csv',
    'W42': 'outputs/experiments/cmc_estimator_review/collected/W/validation_predictions.csv',
    'CV42': 'outputs/experiments/cmc_estimator_review/collected/CV/validation_predictions.csv',
    'CMC42': 'outputs/experiments/cmc_estimator_review/collected/CMC/validation_predictions.csv',
    'PLC42': 'outputs/experiments/amis_prefix_consistency_failure_analysis/artifacts/P_PREFIX_S42/validation_predictions.csv',
    'Middle42': 'outputs/experiments/amis_prefix_consistency_failure_analysis/artifacts/MIDDLE_S42/validation_predictions.csv',
}.items():
    frame = read(relative)
    assert frame[['source', 'target']].equals(validation[['source', 'target']]), name
    predictions[name] = frame.prediction.tolist()

records = []
for (row_a, row_b), (word_a, word_b, key) in sorted(validation_pairs.items()):
    record = dict(row_a=row_a, row_b=row_b, word_a=word_a, word_b=word_b,
                  train_context_count=len(edit_frames.get((word_a, word_b), set())))
    for side, row in [('a', row_a), ('b', row_b)]:
        record['source_' + side] = validation.iloc[row].source
        record['target_' + side] = validation.iloc[row].target
        for name, values in predictions.items():
            record[name + '_' + side] = values[row]
    for name, values in predictions.items():
        record[name + '_identical'] = normalized(values[row_a]) == normalized(values[row_b])
    records.append(record)
pd.DataFrame(records).to_csv(OUT / 'VALIDATION_NATURAL_CONTRASTS.csv', index=False)

# Manual Chinese-reference-relative inspection of W on the preregistered 24-row
# sample. No bilingual adequacy judgment or population inference is implied.
annotations = {
    224: ('reference-compatible', 'Exact reference match.'),
    562: ('lexical-object-state', 'Trousers/tying replaces clothes-button/mending.'),
    571: ('lexical-object-state', 'Dishes/sweet replaces soup/tasty.'),
    151: ('reference-compatible', 'Called Kacaw: minor acceptable wording difference.'),
    438: ('predicate-relation', 'Regards as a teacher differs from recruiting as teacher.'),
    424: ('predicate-relation', 'Sun comes out/clear weather reverses disappearance/darkness.'),
    26: ('predicate-relation', 'Want him to follow me differs from resenting insufficient support.'),
    212: ('predicate-relation', 'Want to buy money replaces often losing wallet.'),
    441: ('predicate-relation', 'Go weigh books replaces asking father to stamp contact book.'),
    215: ('predicate-relation', 'Reclamation and malformed storm clause differ from righting a house.'),
    300: ('predicate-relation', 'Following child differs from taking all children; quantifier missing.'),
    46: ('expansion-vs-reference', 'Long sister/earring/car clause absent from short father-hits-brother reference; source adequacy unverified.'),
    303: ('lexical-object-state', 'Boiling differs from warming before infant drinks.'),
    28: ('lexical-object-state', 'Luggage replaces rolled bamboo mat.'),
    199: ('predicate-relation', 'Drowsiness replaces eating vegetables but not rice.'),
    301: ('predicate-relation', 'Malformed breathlessness differs from extreme hunger; lack of food retained.'),
    360: ('predicate-relation', 'Added iron bar and incomplete destination; bicycle pushing only partly recovered.'),
    56: ('predicate-relation', 'People locked in a room replaces bus passengers needing to urinate.'),
    377: ('predicate-relation', 'Bringing rice pot to mill replaces returning for salt.'),
    286: ('missing-specificity', 'Generic illness omits infertility.'),
    75: ('lexical-object-state', 'Pocket replaces shoe; feared theft becomes actual robbery.'),
    524: ('source-copy-omission', 'Untranslated Amis fragment and omitted sister accompaniment.'),
    402: ('predicate-relation', 'Finishing work differs from carefully obtaining job.'),
    352: ('reference-compatible', 'Beautiful house compatible; adds generalizing all.'),
}
sample = read(str((OUT / 'FIXED_VALIDATION_SAMPLE.csv').relative_to(ROOT)).replace('\\', '/'))
assert set(sample.row) == set(annotations)
sample['W_primary_category'] = [annotations[row][0] for row in sample.row]
sample['W_reference_relative_note'] = [annotations[row][1] for row in sample.row]
sample.to_csv(OUT / 'ANNOTATED_VALIDATION_SAMPLE.csv', index=False)
counts = Counter(sample.W_primary_category)
summary = {
    'train_contrasting_pairs': len(train_pairs),
    'distinct_word_edits': len(edit_frames),
    'edits_seen_in_at_least_two_train_frames': sum(len(frames) >= 2 for frames in edit_frames.values()),
    'largest_train_frame_count_for_an_edit': max(map(len, edit_frames.values()), default=0),
    'validation_contrasting_pairs': len(records),
    'validation_pairs_with_edit_seen_in_train': sum(r['train_context_count'] > 0 for r in records),
    'identical_predictions_despite_different_references': {
        name: sum(record[name + '_identical'] for record in records) for name in predictions
    },
    'fixed_sample_W_categories': {name: {'n': count, 'fraction': count / len(sample)} for name, count in counts.items()},
    'limitations': [
        'Different normalized references do not certify a semantic minimal pair.',
        'Pairs overlap in sentences and are not independent observations.',
        'Identical strings miss semantic invariance expressed by different words.',
        'No source-language expert validated the manual sample; only reference-relative discrepancies.',
        'Exact normalized word frames exclude morphology, reordering and longer edits.',
        'No model logits, causal localization, bilingual adequacy score or new GPU measurement.',
    ],
}
(OUT / 'CONTRAST_AUDIT.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
(OUT / 'CONTRAST_INPUT_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
