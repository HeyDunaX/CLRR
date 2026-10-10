"""Write a new inventory from verified measurements; preserve historical JSON."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs_rebuttal/amis_routing_diagnostics_20261004_v5'
audit = json.loads((OUT / 'independent_audit.json').read_text())
assert not audit['errors'] and audit['core_status'] == 'VERIFIED42'
assert audit['baseline_archive_audit']['optional_baseline42_valbeam4']['status'] == 'VERIFIED_BEAM_PROTOCOL_MISMATCH'
inventory = json.loads((ROOT / 'outputs_rebuttal/amis_followup_20261004_v2/validation_inventory_21.json').read_text())
measurements = {r['id']: (r['scores'], r['prediction_sha256'], f"runs/{r['id']}/val_main_b1/predictions.csv")
                for r in json.loads((OUT / 'summary.json').read_text()) if r['condition'] == 'val_main_b1'}
for r in json.loads((OUT / 'baseline_archive_audit_v3/summary.json').read_text()):
    item = r['conditions']['validation']
    measurements[r['id']] = (item['scores'], item['prediction_sha256'], f"baseline_archive_audit_v3/{r['id']}/validation_predictions.csv")
assert len(measurements) == 12
for entry in inventory:
    if entry['id'] not in measurements:
        entry['v5_validation_status'] = 'HISTORICAL_NOT_REMEASURED_IN_V5'
        continue
    values, digest, relative = measurements[entry['id']]
    entry['historical_validation_metadata'] = {k: entry.get(k) for k in ['eval_bleu', 'eval_chrf++', 'eval_chrf0', 'eval_loss', 'best_step']}
    entry['eval_bleu'], entry['eval_chrf++'], entry['eval_chrf0'] = values['bleu'], values['chrf++'], values['chrf0']
    entry['validation_predictions_exists'] = True
    entry['validation_prediction_sha256'] = digest
    entry['validation_prediction_file'] = (OUT.relative_to(ROOT) / relative).as_posix()
    entry['v5_validation_status'] = 'VERIFIED_NATIVE_GENERATION_METRICS'
    entry['validation_protocol'] = {'beams': 1, 'max_length': 256, 'batch_size': 8, 'precision': 'fp32',
                                    'tf32': False, 'src_lang': 'tl_XX', 'tgt_lang': 'zh_CN'}
    entry['eval_loss_provenance'] = 'Historical full-validation eval loss retained; not recomputed by this generation audit. Fixed32 CE is a separate diagnostic.'
    if entry['id'] == 'mbart-amis-baseline-seed42':
        entry['best_step'] = None
        entry['best_step_provenance'] = 'Original selected step unavailable after recovery; hardcoded216 is unsupported.'
        entry['historical_validation_metadata']['actual_beams_confirmed'] = 4
path = OUT / 'validation_inventory_21_corrected_v5.json'
path.write_text(json.dumps(inventory, indent=2) + '\n')
print('NEW INVENTORY', path, '12 remeasured +9 historical; old inventory untouched')
