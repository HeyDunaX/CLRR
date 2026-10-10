"""Add independently verified ES10 archives to the existing public index."""
import csv
import io
import json
import sys
from pathlib import Path
import requests
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner.colab import get_hf_token

REPO = 'FiveC/amis-rewire-checkpoints'


def main():
    api = HfApi(token=get_hf_token())
    info = api.model_info(REPO)
    assert not info.private
    response = requests.get(f'https://huggingface.co/{REPO}/resolve/{info.sha}/registry/checkpoint_index.json', timeout=60)
    response.raise_for_status()
    index = response.json()
    rows = {r['path']: r for r in index['checkpoints']}
    original_paths = set(rows)
    for cohort in ['amis_standardized_controls', 'amis_standardized_strong_mbart']:
        folder = ROOT / 'outputs_rebuttal' / cohort
        report_path = folder / 'validation_verification.json'
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        test_report = folder / 'test_verification.json'
        test_verified = json.loads(test_report.read_text())['runs'] if test_report.exists() else {}
        specs = {r['id']: r for r in json.loads((folder / 'PLAN.json').read_text())['runs']}
        for ident, receipt in report['runs'].items():
            spec = specs[ident]
            path = receipt['archive_path']
            assert path == spec['remote_prefix'] + '/best.zip'
            entry = api.get_paths_info(REPO, [path], revision=info.sha)[0]
            assert entry.lfs.sha256 == receipt['archive_sha256']
            if path in rows:
                assert rows[path]['sha256'] == receipt['archive_sha256']
            rows[path] = {'dataset': 'amis_mandarin', 'backbone': spec['backbone'],
                'method': spec['method'], 'seed': f"seed-{spec['seed']}", 'recipe': 'es10',
                'cohort': 'standardized-controls', 'primary': False,
                'path': path, 'source_path': path, 'bytes': entry.size,
                'sha256': entry.lfs.sha256, 'seed_evidence': {'source': 'verified run_metadata and training arguments'},
                'metadata_complete': True, 'validation_verified': True,
                'selected_step': receipt['selected_step'], 'selected_epoch': receipt['selected_epoch'],
                'completed_epoch': receipt['completed_epoch'],
                'test_verified': ident in test_verified}
    # Preserve every indexed legacy archive and its digest; no weights are moved/deleted.
    entries = {e.path: e for e in api.get_paths_info(REPO, list(rows), revision=info.sha)}
    for path, row in rows.items():
        assert entries[path].size == row['bytes'] and entries[path].lfs.sha256 == row['sha256']
    assert original_paths.issubset(rows)
    index.update(source_revision=info.sha, checkpoints=[rows[p] for p in sorted(rows)])
    out = ROOT / 'outputs_rebuttal/amis_standardized_controls/public_index'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoint_index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    fields = ['dataset', 'backbone', 'method', 'seed', 'primary', 'recipe', 'cohort', 'path', 'bytes', 'sha256', 'metadata_complete']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader(); writer.writerows(index['checkpoints'])
    (out / 'checkpoint_index.csv').write_text(stream.getvalue(), encoding='utf-8')
    lines = ['# Checkpoint index', '', 'Public final checkpoints; original archives remain unchanged.', '',
             '`metadata_complete = null` means legacy metadata was not reassessed. ES10 entries have independently verified validation receipts. Validation completion does not imply test completion.', '',
             '| Dataset | Backbone | Method | Seed | Configuration | Metadata verified | Checkpoint |',
             '|---|---|---|---|---|---|---|']
    for row in index['checkpoints']:
        lines.append('| ' + ' | '.join(str(row.get(k, '')) for k in ['dataset', 'backbone', 'method', 'seed', 'recipe', 'metadata_complete'])
                     + f" | [best.zip](https://huggingface.co/{REPO}/blob/main/{row['path']}) |")
    (out / 'CHECKPOINT_INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    result = api.create_commit(repo_id=REPO, parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo='registry/' + name, path_or_fileobj=str(out / name))
                    for name in ['checkpoint_index.json', 'checkpoint_index.csv', 'CHECKPOINT_INDEX.md']],
        commit_message='Index verified ES10 comparator checkpoints without modifying weights')
    print('PUBLIC_INDEX_UPDATED', len(rows), result.oid)


if __name__ == '__main__':
    main()
