"""Index existing result evidence without scoring, training, or altering sources."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

METRICS = {'bleu', 'chrf', 'chrf++', 'chrf0', 'chrfpp', 'eval_bleu',
           'eval_chrf++', 'validation_bleu', 'validation_chrf0',
           'best_metric_chrfpp', 'test_bleu', 'test_chrf++'}
SKIP_KEYS = {'log_history', 'trajectories', 'bundle_files', 'source_sha256',
             'files', 'environment', 'env', 'citations'}


def dump_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def cell(value):
    return str(value if value is not None else '—').replace('|', '\\|').replace('\n', ' ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', default='outputs/results_registry')
    parser.add_argument('--fetch-hf-metadata', action='store_true',
                        help='Read public small JSONs at the pinned HF_INDEX revision; no weights.')
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    hf_index = json.loads((out/'HF_INDEX.json').read_text(encoding='utf-8')) if (out/'HF_INDEX.json').exists() else {}
    hf_files = {f['path']: f for f in hf_index.get('files', [])}
    fetch_errors = []
    if args.fetch_hf_metadata:
        import requests
        allowed = {'run_metadata.json', 'validation_metrics.json', 'test_metrics.json', 'metrics.json'}
        targets = [p for p, f in hf_files.items() if p.startswith('checkpoints/') and
                   Path(p).name in allowed and f['size'] <= 300000]
        def fetch(name):
            target = out/'hf_metadata'/name
            if target.exists():
                return
            try:
                response = requests.get('https://huggingface.co/FiveC/amis-rewire-checkpoints/resolve/' +
                    hf_index['revision']+'/'+name, timeout=40)
                response.raise_for_status()
                payload = response.content
                assert len(payload) == hf_files[name]['size'], 'size mismatch'
                digest = hf_files[name].get('sha256')
                if digest:
                    assert hashlib.sha256(payload).hexdigest() == digest, 'SHA256 mismatch'
                json.loads(payload)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            except (requests.RequestException, ValueError, AssertionError) as exc:
                return dict(path=name, error=str(exc))
        with ThreadPoolExecutor(max_workers=6) as executor:
            fetch_errors = [r for r in executor.map(fetch, targets) if r]
    artifacts, metrics, receipts, errors, tables = [], [], [], [], []
    sources, cohort_files, parsed = {}, defaultdict(list), {}
    roots = [p for p in root.iterdir() if p.is_dir() and
             (p.name == 'docs' or p.name.startswith('outputs'))]
    for base in sorted(roots):
        for path in sorted(base.rglob('*')):
            if not path.is_file() or out in path.parents:
                continue
            if path.suffix.lower() not in {'.json', '.csv', '.md'}:
                continue
            if path.name == 'RESULTS_REGISTRY.md':
                continue
            rel = path.relative_to(root).as_posix()
            historical = 'archive' in [part.lower() for part in path.parts]
            parts = Path(rel).parts
            cohort = '/'.join(parts[:3] if parts[:2] == ('outputs', 'experiments') else parts[:2])
            cohort_files[cohort].append(rel)
            size = path.stat().st_size
            item = dict(path=rel, cohort=cohort, historical=historical,
                        bytes=size, sha256='', status='', csv_rows='', csv_header='')
            if size > 16 * 1024 * 1024:
                item['status'] = 'NOT_READ_OVER_16_MIB'
                artifacts.append(item)
                continue
            raw = path.read_bytes()
            item['sha256'] = hashlib.sha256(raw).hexdigest()
            sources[rel] = item['sha256']
            try:
                text = raw.decode('utf-8-sig')
                if path.suffix.lower() == '.json':
                    obj = json.loads(text)
                    parsed[rel] = obj
                    if isinstance(obj, dict):
                        item['status'] = str(obj.get('status', obj.get('stage', '')))
                    def walk(value, pointer='', context=None):
                        if isinstance(value, dict):
                            context = dict(context or {})
                            for key in ('id', 'run_name', 'method', 'family', 'seed', 'split', 'status'):
                                if key in value and isinstance(value[key], (str, int, float, bool)):
                                    context[key] = value[key]
                            found = {k: v for k, v in value.items() if k.lower() in METRICS and
                                     isinstance(v, (int, float)) and not isinstance(v, bool)}
                            if found:
                                metrics.append(dict(source=rel, pointer=pointer or '/', cohort=cohort,
                                    historical=historical, context=json.dumps(context, ensure_ascii=False),
                                    values=json.dumps(found, ensure_ascii=False)))
                            for key, child in value.items():
                                if key not in SKIP_KEYS and isinstance(child, (dict, list)):
                                    walk(child, pointer+'/'+key.replace('~', '~0').replace('/', '~1'), context)
                        elif isinstance(value, list):
                            for i, child in enumerate(value):
                                walk(child, pointer+'/'+str(i), context)
                    walk(obj)
                elif path.suffix.lower() == '.csv':
                    reader = csv.reader(text.splitlines())
                    item['csv_header'] = json.dumps(next(reader, []), ensure_ascii=False)
                    item['csv_rows'] = sum(1 for _ in reader)
                else:
                    headings, block, start = [], [], 0
                    for i, line in enumerate(text.splitlines()+[''], 1):
                        if line.startswith('#'):
                            level = len(line)-len(line.lstrip('#'))
                            headings = headings[:level-1]+[line.lstrip('# ').strip()]
                        if line.lstrip().startswith('|'):
                            if not block:
                                start = i
                            block.append(line)
                        elif block:
                            if len(block) >= 2 and re.search(r'\|\s*:?-{3}', block[1]):
                                tables.append(dict(source=rel, line=start, historical=historical,
                                    heading=' / '.join(headings), table='\n'.join(block)))
                            block = []
            except (UnicodeError, ValueError, csv.Error) as exc:
                item['status'] = 'READ_ERROR'
                errors.append(dict(source=rel, error=str(exc)))
            artifacts.append(item)

    # Every completion marker is evidence, not automatically a distinct model.
    # Copied markers in test/provenance directories retain the same archive ID.
    for rel, obj in sorted(parsed.items()):
        if Path(rel).name not in {'VAL_COMPLETE.json', 'COMPLETE.json'} or not isinstance(obj, dict):
            continue
        folder = (root / rel).parent
        metadata = parsed.get((folder/'run_metadata.json').relative_to(root).as_posix(), {})
        if not isinstance(metadata, dict):
            metadata = {}
        spec = obj.get('spec') or metadata.get('run') or {}
        if not isinstance(spec, dict):
            spec = {}
        plan, plan_path = {}, None
        for parent in [folder, *folder.parents]:
            if parent == root:
                break
            key = (parent/'PLAN.json').relative_to(root).as_posix()
            if isinstance(parsed.get(key), dict):
                plan, plan_path = parsed[key], key
                break
        run_id = spec.get('id') or obj.get('id') or obj.get('run_name') or folder.name
        if not spec:
            spec = next((s for s in plan.get('runs', []) if isinstance(s, dict) and s.get('id') == run_id), {})
        protocol_raw = metadata.get('protocol') or plan.get('protocol') or {}
        protocol = protocol_raw if isinstance(protocol_raw, dict) else {}
        score = obj.get('scores') or {}
        if not isinstance(score, dict):
            score = {}
        backup = obj.get('backup') or {}
        if not isinstance(backup, dict):
            backup = {}
        identity = obj.get('archive_sha256') or obj.get('selected_and_exported_weight_digest')
        row = dict(run=run_id, method=spec.get('method', obj.get('family', '')),
            seed=spec.get('seed', obj.get('seed', '')), dataset=plan.get('dataset', ''),
            backbone=spec.get('backbone', ''), status=obj.get('status', ''),
            bleu=score.get('bleu', obj.get('validation_bleu', '')),
            chrfpp=score.get('chrf++', obj.get('best_metric_chrfpp', '')),
            chrf0=score.get('chrf0', obj.get('validation_chrf0', '')),
            score_scope='validation' if Path(rel).name == 'VAL_COMPLETE.json' else 'CHECK_SOURCE',
            selector=protocol.get('selection', ''), hardware=protocol.get('hardware_profile', ''),
            backend=protocol.get('attention_implementation', ''), patience=protocol.get('patience', ''),
            selected_epoch=metadata.get('selected_epoch', ''),
            completed_epoch=metadata.get('completed_epoch', obj.get('completed_epochs', '')),
            archive_sha256=identity or '', backup_verified=backup.get('verified', ''),
            hf_repo=backup.get('repo', obj.get('hf_repo', plan.get('repo', ''))),
            hf_prefix=spec.get('remote_prefix', obj.get('hf_prefix', '')),
            hf_revision=backup.get('revision', ''), receipt=rel, plan=plan_path or '',
            metadata=(folder/'run_metadata.json').relative_to(root).as_posix() if metadata else '',
            historical='archive' in Path(rel).parts,
            declaration_sha256=hashlib.sha256(json.dumps(dict(spec=spec, protocol=protocol,
                data=plan.get('data'), runtime=plan.get('runtime')), sort_keys=True).encode()).hexdigest(),
            declaration=json.dumps(dict(spec=spec, protocol=protocol, data=plan.get('data'),
                runtime=plan.get('runtime')), ensure_ascii=False, sort_keys=True))
        receipts.append(row)

    dump_csv(out/'artifacts.csv', artifacts, list(artifacts[0]))
    dump_csv(out/'metric_blocks.csv', metrics, ['source','pointer','cohort','historical','context','values'])
    dump_csv(out/'completion_receipts.csv', receipts, list(receipts[0]) if receipts else ['receipt'])
    duplicates = defaultdict(list)
    for row in receipts:
        if row['archive_sha256']:
            duplicates[row['archive_sha256']].append(row['receipt'])
    public = []
    for name, entry in sorted(hf_files.items()):
        if not name.startswith('checkpoints/') or not name.endswith(('.zip', '.safetensors', '.bin')):
            continue
        folder = Path(name).parent
        metadata_path = out/'hf_metadata'/folder/'run_metadata.json'
        meta = json.loads(metadata_path.read_text(encoding='utf-8-sig')) if metadata_path.exists() else {}
        if not isinstance(meta, dict):
            meta = {}
        run = meta.get('run', {})
        run = run if isinstance(run, dict) else {}
        protocol = meta.get('protocol', {})
        protocol = protocol if isinstance(protocol, dict) else {}
        found_metrics = {}
        for metric_name in ('validation_metrics.json', 'test_metrics.json', 'metrics.json'):
            mp = out/'hf_metadata'/folder/metric_name
            if mp.exists():
                found_metrics[metric_name] = json.loads(mp.read_text(encoding='utf-8-sig'))
        matches = duplicates.get(entry.get('sha256'), [])
        public.append(dict(path=name, sha256=entry.get('sha256', ''), bytes=entry['size'],
            method=run.get('method', ''), seed=run.get('seed', ''),
            backbone=run.get('backbone', ''), selector=protocol.get('selection', ''),
            hardware=protocol.get('hardware_profile', ''), patience=protocol.get('patience', ''),
            metadata_present=metadata_path.exists(),
            metric_files=';'.join(found_metrics), metrics=json.dumps(found_metrics, ensure_ascii=False),
            matching_local_receipts=';'.join(matches),
            url='https://huggingface.co/FiveC/amis-rewire-checkpoints/blob/'+hf_index['revision']+'/'+name))
    dump_csv(out/'public_checkpoints.csv', public, list(public[0]) if public else ['path'])
    planned = []
    for rel, plan in sorted(parsed.items()):
        if Path(rel).name != 'PLAN.json' or not isinstance(plan, dict):
            continue
        specs = plan.get('runs', [])
        if not isinstance(specs, list):
            continue
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            prefix = spec.get('remote_prefix', '')
            matching = [r for r in receipts if r['plan'] == rel and r['run'] == spec.get('id')]
            hf_matches = [r['path'] for r in public if prefix and str(Path(r['path']).parent).replace('\\', '/') == prefix]
            planned.append(dict(plan=rel, run=spec.get('id', ''), method=spec.get('method', ''),
                seed=spec.get('seed', ''), dataset=plan.get('dataset', ''),
                local_receipts=';'.join(r['receipt'] for r in matching),
                public_archives=';'.join(hf_matches),
                evidence_state='HAS_LOCAL_COMPLETION_RECEIPT' if matching else
                    ('HAS_PUBLIC_ARCHIVE_CHECK_METADATA' if hf_matches else 'UNRESOLVED_NOT_PROOF_UNRUN'),
                declared_plan_status=plan.get('status', ''),
                spec=json.dumps(spec, ensure_ascii=False, sort_keys=True)))
    dump_csv(out/'planned_runs.csv', planned, list(planned[0]) if planned else ['plan'])
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        scope='Local docs/outputs* plus pinned public HF filename/metadata snapshot. No weights or dataset changes.',
        hf_revision=hf_index.get('revision'), public_checkpoint_files=len(public),
        public_noncheckpoint_zips=[p for p in hf_files if p.endswith('.zip') and not p.startswith('checkpoints/')],
        public_with_metadata=sum(r['metadata_present'] for r in public),
        public_with_local_receipt_hash=sum(bool(r['matching_local_receipts']) for r in public),
        hf_fetch_errors=fetch_errors, planned_run_references=len(planned),
        hf_metadata_sha256={p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((out/'hf_metadata').rglob('*.json'))},
        artifacts=len(artifacts), metric_blocks=len(metrics), completion_receipts=len(receipts),
        distinct_archive_hashes=len(duplicates), doc_tables=len(tables), cohorts=len(cohort_files),
        read_errors=errors, unread_large=[a['path'] for a in artifacts if a['status']=='NOT_READ_OVER_16_MIB'],
        repeated_archive_receipts={k:v for k,v in duplicates.items() if len(v)>1}, source_sha256=sources)
    (out/'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Bảng nguồn được giữ nguyên', '',
             'Bản trích nguyên văn để tra cứu, không xác nhận mọi bảng là kết quả hiện hành.',
             'Các banner cũ, số historical, bảng kế hoạch và bảng đã bị thay thế vẫn có thể xuất hiện.', '']
    for table in tables:
        lines += [f"## {table['source']}:{table['line']}", '',
                  f"Historical/archive: {table['historical']}. Section: {table['heading']}", '', table['table'], '']
    (out/'source_tables.md').write_text('\n'.join(lines), encoding='utf-8')
    lines = ['# Chi tiết receipt hoàn tất', '',
             'Một dòng là một receipt; bản sao không phải model mới. Gom bằng archive SHA khi có.',
             'VAL_COMPLETE là validation; COMPLETE có scope CHECK_SOURCE phải đọc nguồn trước khi dùng.',
             'SHA declaration chỉ dùng tra cứu cấu hình khai báo; không phải chứng nhận fairness.', '',
             '| Run | Method | Seed | Dataset | BLEU | chrF++ | Scope | HW / selector | Receipt |',
             '|---|---|---:|---|---:|---:|---|---|---|']
    for r in receipts:
        link = (root/r['receipt']).as_posix()
        lines.append('| '+' | '.join(cell(r[k]) for k in ['run','method','seed','dataset','bleu','chrfpp','score_scope'])+
            ' | '+cell(str(r['hardware'])+' / '+str(r['selector']))+f' | [source]({link}) |')
    lines += ['', '# Danh mục nhóm artifacts', '',
              'Có artifacts không đồng nghĩa đã train. Mục không có completion marker phải đọc báo cáo và HF trước khi chạy lại.', '',
              '| Nhóm | Files | Completion markers | Reports / plans |', '|---|---:|---:|---|']
    for cohort, files in sorted(cohort_files.items()):
        reports = [p for p in files if p.endswith('.md') or Path(p).name == 'PLAN.json']
        links = [f'[{Path(p).name}]({(root/p).as_posix()})' for p in reports[:5]]
        count = sum(1 for r in receipts if r['receipt'] in files)
        lines.append(f'| {cohort} | {len(files)} | {count} | '+ '; '.join(links)+' |')
    lines += ['', '# Checkpoint public tại revision đã khóa', '',
              'Một dòng là một archive, không phải một training run. best/best_chrf và historical variants giữ riêng.',
              'Chi tiết điểm nguyên bản trong public_checkpoints.csv; thiếu metadata không đồng nghĩa chưa chạy.', '',
              '| HF path | Method / seed từ metadata | Selector | Local receipt khớp SHA |',
              '|---|---|---|---:|']
    for r in public:
        lines.append(f"| [{cell(r['path'])}]({r['url']}) | {cell(r['method'])} / {cell(r['seed'])} | "
                     f"{cell(r['selector'])} | {len(r['matching_local_receipts'].split(';')) if r['matching_local_receipts'] else 0} |")
    (out/'INDEX.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'source_sha256','hf_metadata_sha256','repeated_archive_receipts'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
