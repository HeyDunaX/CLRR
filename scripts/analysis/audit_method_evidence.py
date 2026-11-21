"""Inventory completed validation receipts without mixing protocols or using test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/experiments'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, trajectories, incomplete = [], [], []
    for cohort in sorted(args.root.iterdir()):
        plan_path = cohort / 'PLAN.json'
        if not plan_path.exists() or not (cohort/'runs').exists():
            continue
        plan = read(plan_path)
        # Test-evaluation directories contain validation receipts copied for
        # provenance. Do not count those as new completed training runs.
        if plan.get('test_authorization') or cohort.name.endswith('_test'):
            continue
        protocol = plan.get('protocol', {})
        for folder in sorted((cohort/'runs').iterdir()):
            marker = folder/'VAL_COMPLETE.json'
            if not marker.exists():
                incomplete.append(dict(cohort=cohort.name, run=folder.name))
                continue
            value = read(marker)
            spec = value.get('spec', {})
            metadata_path = folder/'run_metadata.json'
            metadata = read(metadata_path) if metadata_path.exists() else {}
            key = dict(dataset=plan.get('dataset'), data=plan.get('data'),
                       runtime=plan.get('runtime'), protocol=protocol,
                       model_id=spec.get('model_id'), revision=spec.get('model_revision'),
                       **{k:spec.get(k) for k in ('learning_rate','microbatch','accumulation',
                           'max_source_length','max_target_length','validation_beams','src_lang','tgt_lang')})
            fingerprint = hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:16]
            scores = value.get('scores', {})
            row = dict(cohort=cohort.name, run=folder.name, dataset=plan.get('dataset'),
                       method=spec.get('method'), seed=spec.get('seed'),
                       bleu=scores.get('bleu'), chrfpp=scores.get('chrf++'),
                       selected_epoch=metadata.get('selected_epoch'),
                       completed_epoch=metadata.get('completed_epoch'),
                       selector=protocol.get('selection'), hardware=protocol.get('hardware_profile'),
                       backend=protocol.get('attention_implementation','UNRECORDED'),
                       comparison_group=fingerprint, metadata_present=bool(metadata),
                       backup_verified=value.get('backup',{}).get('verified',False),
                       archive_sha256=value.get('archive_sha256'), receipt=str(marker))
            records.append(row)
            state_path = folder/'final_trainer_state.json'
            if state_path.exists():
                for entry in read(state_path).get('log_history', []):
                    if any(k in entry for k in ('eval_bleu','eval_chrf++','loss')):
                        trajectories.append(dict(cohort=cohort.name,run=folder.name,**entry))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'VALIDATION_INVENTORY.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(completed_receipts=len(records),incomplete_local_folders=incomplete,
                trajectories=trajectories,records=records,
                scope='Local shared-runner validation receipts only; not exhaustive historical paper inventory. Missing local receipt does not prove run missing on HF.',
                grouping='Exact declared recipe groups only. Different fingerprints are not automatically unfair; inspect differences. Backend defaults are not runtime proof.',
                test_consulted=False)
    (args.output/'EVIDENCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Completed validation evidence inventory','',result['scope'],'',
           '| Cohort / run | Method | Seed | BLEU | chrF++ | Recipe group |',
           '|---|---|---:|---:|---:|---|']
    for row in records:
        lines.append(f"| {row['cohort']} / {row['run']} | {row['method']} | {row['seed']} | {row['bleu']} | {row['chrfpp']} | {row['comparison_group']} |")
    (args.output/'INVENTORY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed_receipts=len(records),trajectory_entries=len(trajectories),output=str(args.output))))


if __name__ == '__main__':
    main()
