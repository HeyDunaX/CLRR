"""Audit train-only lexical candidates, without weights/GPU or semantic claims."""
import argparse
import csv
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from methods.difference_preserving.v1.pairs import discover_pairs, TOKENIZATION_POLICY


def audit_target_structure(rows, pairs, output, dataset, digest):
    """Describe string edits only. Never infer bilingual semantic correctness."""
    records = []
    for pair in pairs:
        i, j = pair['left'], pair['right']
        a, b = [unicodedata.normalize('NFKC', rows[k]['target']).casefold().strip()
                for k in (i, j)]
        matcher = SequenceMatcher(None, a, b, autojunk=False)
        edits = [dict(operation=tag, left=a[x:y], right=b[u:v])
                 for tag, x, y, u, v in matcher.get_opcodes() if tag != 'equal']
        records.append(dict(left=i, right=j, source_left=rows[i]['source'],
            source_right=rows[j]['source'], target_left=rows[i]['target'],
            target_right=rows[j]['target'], normalized_target_identical=a == b,
            target_character_similarity=matcher.ratio(), target_edit_spans=edits,
            semantic_status='UNVERIFIED', translation_status='UNVERIFIED'))
    with (output / 'TARGET_STRUCTURE.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    counts = Counter(len(r['target_edit_spans']) for r in records)
    result = dict(dataset=dataset, split='train', train_sha256=digest,
        tokenization_policy=TOKENIZATION_POLICY,
        candidates=len(records), target_identical_after_normalization=sum(
            r['normalized_target_identical'] for r in records),
        target_edit_span_histogram=dict(sorted(counts.items())),
        verified_semantic_pairs=0, reviewer='automated_string_structure_only',
        status='HOLD_BILINGUAL_REVIEW_AND_NOVELTY', gpu_used=False,
        dataset_modified=False, validation_or_test_read=False,
        limitations=['Character similarity is not semantic similarity.',
            'Multiple edit spans do not prove multiple semantic changes.',
            'No automatic edit-span threshold selects or rejects training pairs.',
            'No bilingual expertise or translation correctness inferred.',
            'Amis and Ede raw character statistics are not quality rankings.'])
    (output / 'TARGET_STRUCTURE_AUDIT.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Train-pair structural audit', '',
        f"Dataset: {dataset}; pairs: {len(records)}; verified semantic pairs: 0.", '',
        '| Target character edit spans | Pairs |', '|---|---:|',
        *[f'| {k} | {v} |' for k, v in sorted(counts.items())], '',
        'All counts describe Unicode-normalized character strings, not meaning. '
        'Original CSV and previous reviews are unchanged. No pair is admitted by this audit.', '',
        '## Review examples (not semantic annotations)', '']
    # Deterministic examples from every structural category; do not cherry-pick only neat cases.
    for spans in sorted(counts):
        for r in [r for r in records if len(r['target_edit_spans']) == spans][:3]:
            lines += [f"### Rows {r['left']} / {r['right']} ({spans} edit spans)", '',
                      f"- Source A: {r['source_left']}", f"- Source B: {r['source_right']}",
                      f"- Reference A: {r['target_left']}", f"- Reference B: {r['target_right']}",
                      '- Bilingual correspondence: UNVERIFIED.', '']
    (output / 'TARGET_STRUCTURE_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['amis_mandarin', 'rhade_vietnamese'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-structure', action='store_true',
                        help='Describe all candidate target edits; does not authorize semantic pairs')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must stay under workspace outputs/')
    if output.exists():
        raise FileExistsError('Preserve prior audit; choose a new output directory')
    train = ROOT / 'data_processed' / args.dataset / 'train.csv'
    digest = hashlib.sha256(train.read_bytes()).hexdigest()
    with train.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['source', 'target']:
            raise ValueError('Expected source,target')
        rows = list(reader)
    pairs = discover_pairs(rows)
    covered = {p[key] for p in pairs for key in ('left', 'right')}
    # Describes code behavior IF every candidate were reviewed; admits no pair.
    peers = {}
    for pair in pairs:
        i, j = pair['left'], pair['right']
        peers[i] = min(peers.get(i, j), j)
        peers[j] = min(peers.get(j, i), i)
    selected_edges = {tuple(sorted((i, j))) for i, j in peers.items()}
    output.mkdir(parents=True)
    if args.target_structure:
        result = audit_target_structure(rows, pairs, output, args.dataset, digest)
        if hashlib.sha256(train.read_bytes()).hexdigest() != digest:
            raise ValueError('Original training CSV changed during audit')
        print(json.dumps(result, indent=2))
        return
    with (output / 'lexical_candidates.jsonl').open('w', encoding='utf-8') as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + '\n')
    sample = random.Random(42).sample(pairs, min(100, len(pairs)))
    with (output / 'PAIR_REVIEW.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['left', 'right', 'source_left', 'source_right', 'target_left',
                  'target_right', 'meaning_difference_valid', 'translation_correspondence_valid',
                  'reviewer', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in sample:
            i, j = pair['left'], pair['right']
            writer.writerow(dict(left=i, right=j, source_left=rows[i]['source'],
                source_right=rows[j]['source'], target_left=rows[i]['target'],
                target_right=rows[j]['target']))
    assert hashlib.sha256(train.read_bytes()).hexdigest() == digest
    result = dict(dataset=args.dataset, split='train', train_sha256=digest,
        tokenization_policy=TOKENIZATION_POLICY,
        rows=len(rows), lexical_candidate_pairs=len(pairs), covered_rows=len(covered),
        coverage=len(covered)/len(rows), verified_semantic_pairs=0,
        status='HOLD_SEMANTIC_AND_NOVELTY_GATES', gpu_used=False,
        dataset_modified=False, validation_or_test_read=False,
        conditional_peer_policy=dict(
            condition='hypothetical: all candidates accepted; NOT semantic admission',
            candidate='minimum reviewed neighbor row index',
            generic='seeded choice excluding self, ALL reviewed neighbors and identical strings',
            directed_anchors=len(peers), selected_undirected_edges=len(selected_edges),
            unselected_candidate_edges=len(pairs)-len(selected_edges)),
        rule='same normalized source words except exactly one substitution; length3to50; distinct nonempty raw target',
        limitations=['lexical closeness is not semantic verification',
                     'no claim of morphology or translation correctness',
                     'generic pairing must use the same eligible anchors as candidate',
                     'sampler/CE schedule must remain unchanged'])
    (output / 'PAIR_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
