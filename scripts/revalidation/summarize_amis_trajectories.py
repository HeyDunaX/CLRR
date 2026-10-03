"""Summarize observational probes; distinguish actual and counterfactual LSR."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

root=Path(__file__).resolve().parents[2]
out=root/'outputs_rebuttal/amis_confirmation_20261003'
assert (out/'COMPLETE.json').exists()
protocol=json.loads((out/'protocol.json').read_text())
records=[];common_batches=None;missing=[]
for run in protocol['runs']:
    method=run['method'];roles=set()
    for done in sorted((out/'trajectory'/run['run_name']).glob('*/DONE.json')):
        metadata=json.loads(done.read_text());roles.update(metadata['roles'])
        assert metadata['method']==method and metadata['training_seed']==run['seed']
        gradients=json.loads((done.parent/'gradients.json').read_text())
        batch_indices=[g['validation_indices'] for g in gradients]
        if common_batches is None:common_batches=batch_indices
        assert batch_indices==common_batches and len(gradients)==8
        assert all(g['lsr_active_in_training_method']==(method in ('lsr','full')) for g in gradients)
        assert all(abs(g['ce_reconstructed_difference'])<1e-6 and abs(g['weighted_lsr_reconstructed_difference'])<1e-6 for g in gradients)
        collapse=json.loads((done.parent/'collapse.json').read_text())
        residual=json.loads((done.parent/'residual_summary.json').read_text())
        all_ratios=[g['groups']['all_parameters']['lsr_to_ce_norm_ratio'] for g in gradients]
        encoder_ratios=[g['groups']['encoder_without_shared_embeddings']['lsr_to_ce_norm_ratio'] for g in gradients]
        row={'method':method,'seed':run['seed'],'epoch':metadata['epoch'],'step':metadata['step'],
            'roles':';'.join(metadata['roles']),'lsr_active':method in ('lsr','full'),
            'ce_mean':statistics.mean(g['ce_loss'] for g in gradients),
            'weighted_lsr_mean':statistics.mean(g['weighted_lsr_loss'] for g in gradients),
            'weighted_lsr_ce_gradient_ratio_median':statistics.median(all_ratios),
            'weighted_lsr_ce_gradient_ratio_min':min(all_ratios),'weighted_lsr_ce_gradient_ratio_max':max(all_ratios),
            'encoder_gradient_ratio_median':statistics.median(encoder_ratios),
            'all_gradient_cosine_median':statistics.median(g['groups']['all_parameters']['gradient_cosine'] for g in gradients),
            'residual_rms_ratio_mean':residual['mean'],'residual_rms_ratio_max':residual['max']}
        for phase in ['source','target']:
            assert collapse[phase]['rows']==576
            for field in ['effective_rank_covariance_energy','effective_rank_singular_values','top1_variance_fraction','mean_pairwise_cosine']:
                row[phase+'_'+field]=collapse[phase][field]
        records.append(row)
    required={'early','middle_fixed_epoch5','late_actual_training_endpoint'}
    for role in required-roles:missing.append({'run_name':run['run_name'],'role':role})
assert missing==[{'run_name':'mbart-amis-baseline-seed42','role':'late_actual_training_endpoint'}],missing
with (out/'trajectory_summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
summary={'points':len(records),'missing':missing,'fixed_validation_indices':common_batches,
    'scope':'Observational probes; no causal attribution. Baseline/CLRR auxiliary gradients are counterfactual.',
    'endpoint_caution':'Actual endpoints may have different epochs and are not selected best models; no missing probe imputation.',
    'summary_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'trajectory_summary.json').write_text(json.dumps(summary,indent=2))
print('TRAJECTORY_SUMMARY_COMPLETE',len(records),'MISSING_ENDPOINT_BASELINE42',flush=True)
