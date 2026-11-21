"""Export published Hugging Face Parquet splits without changing text or rows."""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def export(args):
    if not re.fullmatch(r'[a-z0-9_-]+', args.dataset):
        raise ValueError('Dataset must be a simple directory name')
    if not re.fullmatch(r'[a-f0-9]{40}', args.revision):
        raise ValueError('Pin a full upstream commit SHA')
    folder = ROOT / 'data_processed' / args.dataset
    records, original, csv_bytes = {}, {}, {}
    for split in SPLITS:
        filename = f'data/{split}-00000-of-00001.parquet'
        url = f'https://huggingface.co/datasets/{args.repo}/resolve/{args.revision}/{filename}'
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        rows = pq.read_table(pa.BufferReader(response.content)).to_pylist()
        pairs = [(row['translation'][args.source], row['translation'][args.target])
                 for row in rows]
        if any(not isinstance(s, str) or not isinstance(t, str) for s, t in pairs):
            raise ValueError(f'{split}: non-string field; no implicit repair allowed')
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(['source', 'target'])
        writer.writerows(pairs)
        encoded = buffer.getvalue().encode('utf-8')
        reader = csv.DictReader(io.StringIO(encoded.decode('utf-8'), newline=''))
        assert reader.fieldnames == ['source', 'target']
        assert [(row['source'], row['target']) for row in reader] == pairs
        destination = folder / f'{split}.csv'
        if destination.exists() and destination.read_bytes() != encoded:
            raise ValueError(f'{destination}: refusing to overwrite different data')
        if args.verify_only and not destination.exists():
            raise FileNotFoundError(destination)
        original[split], csv_bytes[split] = pairs, encoded
        records[split] = {
            'rows': len(pairs), 'sha256': sha256(encoded),
            'upstream_file': filename, 'upstream_parquet_sha256': sha256(response.content),
            'upstream_row_order': '0..rows-1, unchanged',
            'roundtrip_text_and_order_equal': True,
        }
    sources = {s: {x for x, _ in rows} for s, rows in original.items()}
    pairsets = {s: set(rows) for s, rows in original.items()}
    overlap = {}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        overlap[f'{a}/{b}'] = {
            'exact_shared_source_types': len(sources[a] & sources[b]),
            'exact_source_overlap_rows_in_second': sum(x in sources[a] for x, _ in original[b]),
            'exact_shared_pair_types': len(pairsets[a] & pairsets[b]),
        }
    manifest = {
        'dataset': args.dataset, 'upstream_repo': args.repo,
        'upstream_revision': args.revision,
        'source_language': args.source, 'target_language': args.target,
        'mapping': {f'translation.{args.source}': 'source', f'translation.{args.target}': 'target'},
        'conversion': 'UTF-8 CSV with source,target header; text, row order and split membership unchanged',
        'text_operations': [], 'filtering': False, 'deduplication': False,
        'resampling': False, 'unicode_normalization': False,
        'audit_only_cross_split_overlap': overlap, 'splits': records,
    }
    if args.verify_only:
        assert json.loads((folder / 'manifest.json').read_text(encoding='utf-8')) == manifest
    else:
        folder.mkdir(parents=True, exist_ok=True)
        for split, encoded in csv_bytes.items():
            (folder / f'{split}.csv').write_bytes(encoded)
        (folder / 'manifest.json').write_bytes(
            (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'status': 'VERIFIED' if args.verify_only else 'EXPORTED_AND_VERIFIED',
                      'dataset': args.dataset, 'splits': records, 'overlap': overlap}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--verify-only', action='store_true')
    export(parser.parse_args())
