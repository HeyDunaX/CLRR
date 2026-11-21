"""Summarize frozen gradient diagnostics; never load model/data or use GPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', default='outputs/experiments/amis_cmc_update_geometry_rtx')
    args = parser.parse_args()
    out = Path(args.output)
    names = ['PLAN.json', 'CPU_GATE.json', 'GPU_GATE.json', 'DENSE.json', 'TRIALS.json', 'COMPLETE.json']
    data = {name: json.loads((out/name).read_text()) for name in names}
    receipt = json.loads((out/'COLLECTION.json').read_text())
    for name in names:
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == receipt['files'][name], name
    complete = data['COMPLETE.json']
    assert complete['optimizer_steps'] == 0 and complete['parameters_unchanged']
    assert not complete['test_used'] and not complete['validation_used']
    assert data['CPU_GATE.json']['passed'] and data['GPU_GATE.json']['passed']
    gram = np.asarray(data['DENSE.json']['gram'], dtype=float)
    assert data['DENSE.json']['gram_order'] == ['CE', 'W', 'CMC']
    def norm(v, g=gram):
        return math.sqrt(max(0., float(v@g@v)))
    def cosine(v, w, g=gram):
        return float(v@g@w)/max(norm(v,g)*norm(w,g), 1e-30)
    ce, w, c = np.eye(3)
    beta = 1.25
    dense = {
        'norm_CE': norm(ce), 'norm_W_aux': norm(w), 'norm_CMC_aux': norm(c),
        'cosine_CE_W_aux': cosine(ce,w), 'cosine_CE_CMC_aux': cosine(ce,c),
        'cosine_W_CMC_aux': cosine(w,c),
        'cosine_total_W_total_CMC': cosine(ce+beta*w,ce+beta*c),
        'relative_total_change_CMC_vs_W': norm(beta*(c-w))/max(norm(ce+beta*w),1e-30),
        'norm_total_W': norm(ce+beta*w), 'norm_total_CMC': norm(ce+beta*c),
        'scaled_W_aux_over_CE': beta*norm(w)/max(norm(ce),1e-30),
        'scaled_CMC_aux_over_CE': beta*norm(c)/max(norm(ce),1e-30),
    }
    trials = data['TRIALS.json']
    assert len(trials) == complete['sampling_repeats'] == 8
    vectors = {'sampled_CMC': np.array([1,0,0,beta,0]),
               'CV': np.array([1,beta,0,beta,-beta]),
               'sampled_direct': np.array([1,0,0,0,beta])}
    sampling = {}
    for method, v in vectors.items():
        reference = np.array([1,beta,0,0,0]) if method == 'sampled_direct' else np.array([1,0,beta,0,0])
        errors, angles, norms = [], [], []
        for trial in trials:
            g = np.asarray(trial['gram'])
            errors.append(norm(v-reference,g)**2)
            angles.append(cosine(v,reference,g))
            norms.append(norm(v,g))
        sampling[method] = dict(
            rms_preclip_error=math.sqrt(float(np.mean(errors))),
            relative_rms_preclip_error=math.sqrt(float(np.mean(errors)))/max(norm(reference,np.asarray(trials[0]['gram'])),1e-30),
            mean_cosine_to_own_dense_reference=float(np.mean(angles)),
            preclip_norm_min=min(norms), preclip_norm_max=max(norms),
            **complete['summaries'][method])
    result = dict(dense=dense, sampling=sampling,
                  mean_microbatch_losses={k:float(np.mean([r[k] for r in data['DENSE.json']['losses']])) for k in ['ce','direct','cmc']},
                  scope='One fixed W42 checkpoint, 128 TRAIN rows, fixed dropout; eight conditional sampling repeats, not trained seeds.',
                  historical_adam_updates_measured=complete['historical_adam_updates_measured'],
                  optimizer_state_files=complete['optimizer_state_files'],
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  verified_input_sha256={n:receipt['files'][n] for n in names})
    (out/'ANALYSIS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    lines = ['# Frozen CMC gradient geometry', '', result['scope'], '',
             'No optimizer step, test decoding or validation decoding. Gates PASS; collected hashes verified.', '',
             '## Dense objective geometry', '', '| Quantity | Value |', '|---|---:|']
    lines += [f'| {k} | {v:.8g} |' for k,v in dense.items()]
    lines += ['', '## Conditional position sampling', '',
              '| Method | RMS preclip error | Mean cosine to own dense reference | Clipped repeats | Distance of clipped MC mean | Trace SE of MC mean |',
              '|---|---:|---:|---:|---:|---:|']
    for k,v in sampling.items():
        lines.append(f"| {k} | {v['rms_preclip_error']:.7g} | {v['mean_cosine_to_own_dense_reference']:.7g} | {v['clip_frequency']:.0%} | {v['mean_postclip_distance']:.7g} | {v['trace_standard_error_mean']:.7g} |")
    lines += ['', 'CMC/CV reference dense CMC; sampled direct references dense W. RMS error includes Monte Carlo noise and is not an exact population variance.', '',
              'Clipping preserves each realized direction. The distance of the clipped Monte Carlo mean is a finite-sample estimate; eight repeats do not establish nonzero clipping bias. Trace SE is an uncertainty scale, not a confidence interval or a significance test.', '',
              'Historical Adam updates measured: '+str(result['historical_adam_updates_measured'])+'. State files: '+str(result['optimizer_state_files'])+'. No zero-state optimizer is substituted for historical state.', '',
              'These measurements distinguish frozen-objective geometry and sampling effects. They cannot identify why trained CMC loses translation quality or establish a new method.']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
