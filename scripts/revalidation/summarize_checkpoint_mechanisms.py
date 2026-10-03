"""Summarize downloaded diagnostics in a separate report folder."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs_rebuttal/checkpoint_diagnostics_20261002'


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def main():
    manifest=json.loads((OUT/'protocol.json').read_text())['manifest']
    summary,layer_summary,sentinels=[] ,[],[]
    for model in manifest['models']:
        folder=OUT/model['id']
        if not (folder/'DONE.json').exists(): continue
        provenance=json.loads((folder/'provenance.json').read_text())
        collapse=json.loads((folder/'collapse.json').read_text())
        gradients=json.loads((folder/'gradients.json').read_text())
        residual=read(folder/'residual_per_example.csv')
        for phase in ['source','target']:
            diag=collapse[phase]
            assert len(diag['singular_values'])==min(diag['rows'],diag['hidden_dim'])
            assert diag['numerical_rank']<=diag['maximum_centered_rank']
        row={'id':model['id'],'dataset':model['dataset'],'method':model['method'],
             'source_effective_rank':collapse['source']['effective_rank_singular_values'],
             'source_rank_limit':collapse['source']['maximum_centered_rank'],
             'source_pairwise_cosine':collapse['source']['mean_pairwise_cosine'],
             'source_top1_variance_fraction':collapse['source']['top1_variance_fraction'],
             'source_covariance_trace':collapse['source']['covariance_trace'],
             'target_effective_rank':collapse['target']['effective_rank_singular_values'],
             'source_target_alignment_cosine':collapse['source_target_alignment_cosine_mean'],
             'native_gradient_objective':all('gradient_objective' in g for g in gradients)}
        values=[float(r['residual_to_current_rms']) for r in residual if r['phase']=='source' and int(r['layer'])>2]
        row.update(residual_ratio_mean=float(np.mean(values)),residual_ratio_p95=float(np.quantile(values,.95)),
                   residual_ratio_max=float(np.max(values)),gradient_batches=len(gradients))
        for group in ['all_parameters','encoder_without_shared_embeddings','shared_input_embeddings']:
            norms=[g['groups'][group] for g in gradients]
            row[group+'_mean_ce_norm']=float(np.mean([r['ce_gradient_norm'] for r in norms]))
            row[group+'_mean_weighted_lsr_norm']=float(np.mean([r['weighted_lsr_gradient_norm'] for r in norms]))
            row[group+'_mean_ratio']=float(np.mean([r['lsr_to_ce_norm_ratio'] for r in norms if r['lsr_to_ce_norm_ratio'] is not None]))
            cos=[r['gradient_cosine'] for r in norms if r['gradient_cosine'] is not None]
            row[group+'_mean_cosine']=float(np.mean(cos)) if cos else None
            row[group+'_negative_cosine_batches']=sum(v<0 for v in cos)
        row['max_abs_ce_reconstruction_difference']=max(abs(g.get('ce_reconstructed_difference',0)) for g in gradients)
        row['max_abs_aux_reconstruction_difference']=max(abs(g.get('weighted_lsr_reconstructed_difference',0)) for g in gradients)
        summary.append(row)
        for phase in ['source','target']:
            layers=sorted({int(r['layer']) for r in residual if r['phase']==phase})
            for layer in layers:
                part=[r for r in residual if r['phase']==phase and int(r['layer'])==layer]
                assert len(part)==collapse[phase]['rows']
                assert sorted(int(r['row_index']) for r in part)==list(range(len(part)))
                layer_row={'id':model['id'],'phase':phase,'layer':layer,'rows':len(part)}
                for key in ['pre_rms','post_rms','residual_to_current_rms']:
                    data=[float(r[key]) for r in part]
                    layer_row[key+'_mean']=float(np.mean(data)); layer_row[key+'_p95']=float(np.quantile(data,.95))
                cos=[float(r['skip_current_cosine']) for r in part if r['skip_current_cosine']]
                layer_row['skip_current_cosine_mean']=float(np.mean(cos)) if cos else None
                layer_summary.append(layer_row)
        expected=read(ROOT/model['reference_prediction_path'])
        assert hashlib.sha256((ROOT/model['reference_prediction_path']).read_bytes()).hexdigest()==model['reference_prediction_sha256']
        for filename,precision in [('sentinel_predictions.json','BF16'),('sentinel_predictions_fp32.json','FP32')]:
            if not (folder/filename).exists(): continue
            for p in json.loads((folder/filename).read_text()):
                old=expected[p['index']]['prediction'].strip()
                sentinels.append({'id':model['id'],'precision':precision,'index':p['index'],
                                  'matches':p['prediction']==old,'old_prediction':old,'new_prediction':p['prediction']})
    for name,rows in [('mechanism_summary.csv',summary),('residual_by_layer.csv',layer_summary),('sentinel_comparison.csv',sentinels)]:
        if not rows: continue
        with (OUT/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
