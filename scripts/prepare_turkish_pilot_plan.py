"""Validate local inputs and write the proposed 12-run plan. Never starts a GPU."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'turkish_old_method_pilot_es10_20261007'
DATA = ROOT / 'data_processed/opus100_turkish_english_20k'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))
    for split, count in [('train', 20000), ('validation', 2000), ('test', 2000)]:
        path = DATA / f'{split}.csv'
        with path.open(encoding='utf8', newline='') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ['source', 'target'], (split, reader.fieldnames)
            rows = sum(1 for _ in reader)
        assert rows == count == manifest['splits'][split]['rows']
        assert sha(path) == manifest['splits'][split]['sha256'], split
    queues = {
        '1': [('full', 42), ('middle', 43), ('baseline', 43), ('lsr', 43), ('baseline', 44)],
        '2': [('lsr', 42), ('middle', 42), ('full', 43), ('middle', 44), ('lsr', 44), ('full', 44)],
        '3': [('baseline', 42)],
    }
    seen = [(family, seed) for runs in queues.values() for family, seed in runs]
    assert len(seen) == len(set(seen)) == 12
    assert set(seen) == {(f, s) for f in ['baseline', 'middle', 'lsr', 'full'] for s in [42, 43, 44]}
    source_names = [
        'src/amis_rewire/train.py', 'src/amis_rewire/modeling.py',
        'src/amis_rewire/evaluation.py', 'src/amis_rewire/metrics.py',
        'src/comparative_baselines/middle_align_acl2025/model.py',
        'src/comparative_baselines/middle_align_acl2025/train.py',
        'scripts/run_turkish_followup.py', 'scripts/prepare_turkish_pilot_plan.py',
    ]
    source_names += sorted(p.relative_to(ROOT).as_posix()
        for p in (ROOT / "src/methods").rglob("*.py"))
    plan = {
        'status': 'PROPOSED_AWAITING_USER_APPROVAL_NOT_LAUNCHED',
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'namespace': PREFIX,
        'scope': 'old methods; diagnostic pilot; no centered objective',
        'quota_observed_cu': {'1': 107.76, '2': 192.56, '3': 24.32},
        'quota_observation': 'colab usage queried 2026-10-07; rate zero and no active assignments for all three accounts',
        'gpu': 'A100 Standard 40GB; no high-memory; no RTX6000',
        'dataset': manifest,
        'model': 'facebook/mbart-large-50-many-to-many-mmt',
        'model_revision': 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30',
        'protocol': {
            'seeds': [42, 43, 44], 'src_lang': 'tr_TR', 'tgt_lang': 'en_XX',
            'learning_rate': 5e-5, 'weight_decay': 0.0, 'lr_warmup_ratio': 0.06,
            'max_epochs': 20, 'early_stopping_patience_evaluations': 10,
            'microbatch': 32, 'gradient_accumulation': 4, 'effective_batch': 128,
            'eval_batch': 8, 'max_source_length': 256, 'max_target_length': 256,
            'bf16_training': True, 'gradient_checkpointing': True,
            'selection': 'raw-reference validation chrF++; beam 1; evaluate each epoch',
            'test': 'beam 4; explicit FP32; TF32 off; BLEU13a/chrF++/chrF0',
            'test_barrier': 'all 12 selected weights and validation receipts verified before test',
        },
        'families': {
            'baseline': {'objective': 'CE'},
            'middle': {'objective': 'CE + symmetric in-batch contrastive alignment',
                       'layer_index': 6, 'temperature': 0.10, 'alignment_weight': 0.10,
                       'scope': 'existing shared-encoder adaptation, not an exact original-paper reproduction'},
            'lsr': {'objective': 'CE + old source-target cosine LSR', 'lambda': 0.10},
            'full': {'objective': 'CE + old LSR with encoder CLRR', 'lambda': 0.10,
                     'alpha': 0.10, 'distance': 2, 'detached_skip': True, 'routing_warmup': 0.0},
        },
        'queues': {
            account: [{'family': family, 'seed': seed,
                       'run_name': f'mbart-tr-en-{family}-seed{seed}-es10'}
                      for family, seed in runs]
            for account, runs in queues.items()
        },
        'account3_policy': {
            'max_training_runs': 1, 'family': 'baseline', 'additional_test_queue': False,
            'proposed_cu_cap': 8.0, 'proposed_reserve_cu': 15.0,
            'insufficient_budget': 'do not start the assigned model; transfer the full run to account 1/2 before launch',
            'check': 'refresh balance and active A100 rate; estimate run cost from account1/2 timing before enabling account3 long training',
        },
        'backup': {
            'proposed_repositories': {f: f'FiveC/turkish-old-method-pilot-{f}-20261007' for f in ['baseline','middle','lsr','full']},
            'namespace': PREFIX, 'verify_before_cleanup': True,
            'local_artifacts': 'JSON, predictions CSV, receipts, Markdown; no weights',
        },
        'before_launch_required': [
            'user approval of this allocation',
            'new common corrected training worker; do not launch legacy one-seed/patience4 Turkish runner',
            'Middle Turkish language/BOS/BLEU13a configuration; do not launch legacy hardcoded Chinese trainer',
            'Middle accumulation loss kwargs contract (accepts_loss_kwargs=False) and regression verification',
            'CPU checks, then A100 BF16+accumulation4+checkpointing lifecycle gate for all four actual families on the new bundle',
            'final bundle/source hashes and runtime versions recorded; no reuse of an old GPU gate as proof for Middle',
        ],
        'monitor_interval_minutes': 10,
        'time_estimate': 'pending first-epoch train+validation and upload timing; no unmeasured Turkish runtime guarantee',
        'source_snapshot_before_setup_changes': {n: sha(ROOT / n) for n in source_names},
    }
    out = ROOT / 'outputs_rebuttal' / PREFIX
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'PROPOSED_PLAN.json'
    if target.exists():
        existing = json.loads(target.read_text(encoding='utf8'))
        assert existing['status'] == plan['status'], 'Do not overwrite an approved/running plan'
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': plan['status'], 'data_hashes_verified': True,
                      'runs': 12, 'allocation': {k: len(v) for k, v in queues.items()},
                      'path': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
