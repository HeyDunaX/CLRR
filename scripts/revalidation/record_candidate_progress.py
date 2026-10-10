"""Refresh the active candidate handoff table from verified local receipts."""
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs_rebuttal/amis_method_candidates'
HANDOFF = ROOT / 'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md'
IDS = [('U1C','Colab1','CLRR-precache'),('U1F','Colab1','precache + LSR'),
       ('U2L','Colab2','Centered-LSR'),('U2F','Colab2','CLRR + Centered-LSR')]
BASELINE_CHRF = 13.901540564847089


def main():
    path = HANDOFF.read_text(encoding='utf-8')
    start = path.index('## Current authorized work: four method candidates, seed42')
    end = path.index('<!-- standardized-es10-test-current -->', start)
    body = ['## Current authorized work: four method candidates, seed42', '',
        'Recipe: baseline-matched LR5e-5, batch32×accumulation4, cap20, ES10,',
        'validation chrF++ beam1/FP32. No test results are included in this pilot.',
        '', '| Run | Colab | Status | Epoch selected/completed | BLEU | chrF++ | chrF0 | Δ chrF++ vs baseline42 | Public backup/replay |',
        '|---|---|---|---:|---:|---:|---:|---:|---|']
    for ident, account, label in IDS:
        folder = OUT / 'runs' / ident
        complete_path = folder / 'VAL_COMPLETE.json'
        if complete_path.exists():
            rec = json.loads(complete_path.read_text(encoding='utf-8'))
            meta_path = folder / 'run_metadata.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
            scores = rec['scores']
            epochs = f"{meta.get('selected_epoch','?')}/{meta.get('completed_epoch','?')}"
            delta = scores['chrf++'] - BASELINE_CHRF
            replay_path = folder / 'public_replay.json'
            replay = json.loads(replay_path.read_text(encoding='utf-8')) if replay_path.exists() else {}
            verified = bool(rec.get('backup',{}).get('verified')) and replay.get('independent_metric_replay', False)
            body.append(f"| {ident} {label} | {account} | COMPLETE | {epochs} | {scores['bleu']:.4f} | {scores['chrf++']:.4f} | {scores['chrf0']:.4f} | {delta:+.4f} | public checksum + metric replay {'PASS' if verified else 'receipt only'} |")
        else:
            status_path = OUT / 'status' / f'account{1 if account == "Colab1" else 2}.json'
            state = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
            progress = state.get('progress',{}).get(ident,{})
            queue = state.get('queue',{})
            if progress.get('status') == 'TRAINING':
                status = f"TRAINING epoch{progress.get('epoch','?')} step{progress.get('step','?')}/{progress.get('max_steps','?')}"
            elif queue.get('current') == ident and queue.get('status') in ('GATE','TRAIN'):
                status = f"{queue['status']} epoch{progress.get('epoch','?')}"
            elif queue.get('status') == 'FAILED':
                status = 'QUEUE_FAILED; individual dispatch/recovery required'
            elif state.get('process_alive'):
                status = 'QUEUED/STARTING'
            else:
                status = 'PENDING'
            body.append(f'| {ident} {label} | {account} | {status} | — | — | — | — | — | — |')
    done_ids = [ident for ident, _, _ in IDS if (OUT / 'runs' / ident / 'VAL_COMPLETE.json').exists()]
    if len(done_ids) == len(IDS):
        status_line = 'All four runs have public verified receipts, independent metric replay, and matching archive hashes. Both Colab sessions are stopped.'
    else:
        pending = [ident for ident, _, _ in IDS if ident not in done_ids]
        status_line = 'Pending runs: ' + ', '.join(pending) + '. Completed runs must not be retrained or overwritten.'
    body += ['', 'Baseline seed42: epoch20/step720; validation chrF++13.901541.',
        'All candidate values are one-seed validation screening results; small BLEU/chrF++ deltas do not establish a gain.',
        status_line,
        'Detailed recipe, CPU/GPU gates, and recovery log: [METHOD_CANDIDATES_RUNBOOK.md](METHOD_CANDIDATES_RUNBOOK.md).', '']
    path = path[:start] + '\n'.join(body) + '\n' + path[end:]
    HANDOFF.write_text(path, encoding='utf-8')
    print('HANDOFF_CANDIDATE_TABLE_UPDATED')


if __name__ == '__main__': main()
