import hashlib
import json
import sys
from pathlib import Path
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay(path, split='test', dataset='amis_mandarin', expected=None):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw = pd.read_csv(ROOT / 'data_processed' / dataset / (split + '.csv'), dtype=str, keep_default_na=False)
    assert frame[['source', 'target']].equals(raw[['source', 'target']]), str(path)
    pred, refs = frame.prediction.str.strip().tolist(), [raw.target.str.strip().tolist()]
    scores = {'bleu': BLEU(tokenize='zh' if dataset == 'amis_mandarin' else '13a').corpus_score(pred, refs).score,
              'chrfpp': CHRF(word_order=2).corpus_score(pred, refs).score,
              'chrf0': CHRF(word_order=0).corpus_score(pred, refs).score}
    if expected:
        for name, value in expected.items():
            normalized = name.replace('test_', '').replace('eval_', '').replace('chrf++', 'chrfpp')
            if normalized in scores:
                assert abs(scores[normalized] - value) < 1e-8, (str(path), name, scores[normalized], value)
    return {'path': path.relative_to(ROOT).as_posix(), 'rows': len(frame), 'scores': scores,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    result = {'corrected': [], 'fresh': [], 'es10': [], 'turkish': [], 'lsr003_validation': [], 'seed42_controls': []}
    three = read(ROOT / 'outputs_rebuttal/amis_accumulation_confirmation_20261005/three_seed_comparison.json')
    observed = read(ROOT / 'outputs_rebuttal/amis_corrected_four_way_20261005/four_way_summary.json')['observed_scores']
    for seed in [42, 43, 44]:
        for family in ['baseline', 'clrr', 'lsr', 'full']:
            if family == 'clrr':
                verification = ROOT / f'outputs_rebuttal/amis_corrected_controls_20261005/clrr/seed{seed}/local_verification.json'
                v = read(verification)
                pred = verification.parent / 'runs' / v['hf_archive']['run'] / 'test_predictions.csv'
            else:
                row = next(r for r in three['rows'] if r['seed'] == seed and r['family'] == family and r['regime'] == ('baseline_audited' if family == 'baseline' else 'corrected'))
                if family == 'baseline':
                    pred = ROOT / row['scores']['test']['prediction_file']
                    verification = ROOT / 'outputs_rebuttal/amis_corrected_baseline42_20261005/local_verification.json' if seed == 42 else None
                    v = read(verification) if verification else {}
                else:
                    verification = ROOT / row['verification_file']
                    v = read(verification)
                    pred = verification.parent / 'runs' / v['hf_archive']['run'] / 'test_predictions.csv'
            expected = next(o for o in observed if o['seed'] == seed and o['family'] == family)
            result['corrected'].append(dict(seed=seed, family=family, **replay(pred, expected={k: expected[k] for k in ['bleu', 'chrfpp', 'chrf0']})))
            if seed == 42:
                parent = verification.parent
                state_path = parent / 'final_trainer_state.json'
                state = read(state_path) if state_path.exists() else {}
                result['seed42_controls'].append({'family': family, 'verification': verification.relative_to(ROOT).as_posix(),
                    'status': v.get('status'), 'completed_epochs': v.get('completed_epochs', state.get('epoch')),
                    'selected_epoch': v.get('selected_epoch'), 'selected_step': v.get('selected_step'),
                    'validation': v.get('scores', {}).get('validation'), 'metadata_folder': parent.relative_to(ROOT).as_posix()})
    for seed in [42, 43, 44]:
        vpath = ROOT / f'outputs_rebuttal/amis_corrected_controls_20261005/lsr003/seed{seed}/local_verification.json'
        v = read(vpath)
        pred = vpath.parent / 'runs' / v['hf_archive']['run'] / 'validation_predictions.csv'
        result['lsr003_validation'].append(dict(seed=seed, **replay(pred, split='validation', expected=v['scores']['validation'])))
        if seed == 42:
            result['seed42_controls'].append({'family': 'lsr003', 'status': v['status'], 'completed_epochs': v['completed_epochs'], 'selected_epoch': v['selected_epoch'], 'selected_step': v['selected_step'], 'validation': v['scores']['validation'], 'metadata_folder': vpath.parent.relative_to(ROOT).as_posix()})
    fresh = ROOT / 'outputs_rebuttal/amis_fresh_confirmation_20261005'
    for family in ['baseline', 'clrr', 'lsr', 'full']:
        for seed in range(45, 53):
            matches = list((fresh / family / f'seed{seed}' / 'runs').glob('*/test_predictions.csv'))
            assert len(matches) == 1
            path = matches[0]
            result['fresh'].append(dict(seed=seed, family=family, **replay(path, expected=read(path.with_name('test_metrics.json')))))
    for cohort in ['amis_standardized_controls', 'amis_standardized_strong_mbart']:
        folder = ROOT / 'outputs_rebuttal' / cohort
        report = read(folder / 'test_verification.json')
        for ident, row in report['runs'].items():
            result['es10'].append(dict(id=ident, **replay(folder / 'runs' / ident / 'test_predictions.csv', expected=row['scores'])))
    turkish = ROOT / 'outputs_rebuttal/turkish_old_method_pilot_es10_20261007'
    for family in ['baseline', 'middle', 'lsr', 'full']:
        path = turkish / family / 'seed42' / 'test_predictions.csv'
        metrics = list((path.parent / 'runs').glob('*/test_metrics.json'))
        assert len(metrics) == 1
        result['turkish'].append(dict(family=family, **replay(path, dataset='opus100_turkish_english_20k', expected=read(metrics[0]))))
    assert len(result['corrected']) == 12 and len(result['fresh']) == 32 and len(result['es10']) == 14 and len(result['turkish']) == 4
    result['status'] = 'ALL_65_PREDICTION_SETS_REPLAY_PASS'
    (OUT / 'number_replay.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['status'])
    print(json.dumps(result['seed42_controls'], indent=2))


if __name__ == '__main__':
    main()
