"""Audit frozen Token-LSR pilot artifacts and checkpoints, without training."""
import argparse
import base64
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import sacrebleu

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/workspace/ura/thanhdat/projects/CLRR'
SOURCE = 'outputs/experiments/amis_token_rtx'
JOB = 'outputs/experiments/amis_token_pilot_audit'
sys.path.insert(0, str(ROOT / 'src'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def remote(code):
    result = subprocess.run(['C:/Program Files/Tailscale/tailscale.exe', 'ssh',
        'ura@100.90.53.12', REMOTE + '/conda/envs/clrr/bin/python', '-'],
        input=code, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return json.loads(result.stdout)


def collect():
    state = remote(f'''import base64,hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
files={{}}
for rid in ('R1','R2','R3','R4'):
 for p in (root/{SOURCE!r}/'runs'/rid).iterdir():
  if p.suffix in ('.json','.csv'):
   b=p.read_bytes();files[p.relative_to(root).as_posix()]={{'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}}
print(json.dumps(files))
''')
    receipts = {}
    for name, record in state.items():
        content = base64.b64decode(record['data'])
        assert hashlib.sha256(content).hexdigest() == record['sha256']
        path = ROOT / JOB / 'artifacts' / Path(name).relative_to(SOURCE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        receipts[path.relative_to(ROOT / JOB).as_posix()] = record['sha256']
    write(ROOT / JOB / 'TRANSFER_SHA256.json', receipts)
    print('COLLECTED', len(receipts), 'small artifacts; no checkpoint download')


def lengths():
    state=remote(f'''import csv,json,os
from pathlib import Path
root=Path({REMOTE!r})
os.environ['HF_HOME']=str(root/'cache/huggingface')
os.environ['TMPDIR']=str(root/'tmp')
from transformers import AutoTokenizer
plan=json.loads((root/{SOURCE!r}/'PLAN.json').read_text())
spec=plan['runs'][0]
tok=AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['model_revision'],src_lang=spec['src_lang'],tgt_lang=spec['tgt_lang'],cache_dir=str(root/'cache/tokenizer-audit'),token=False,use_fast=True)
rows=list(csv.DictReader((root/'data_processed/amis_mandarin/validation.csv').open(encoding='utf-8-sig')))
values={{}}
for column in ('source','target'):
 text=[r[column] for r in rows]
 ids=tok(text_target=text,truncation=False)['input_ids'] if column=='target' else tok(text,truncation=False)['input_ids']
 lens=[len(x) for x in ids]
 values[column]={{'max_tokens':max(lens),'at_or_above256':sum(n>=256 for n in lens)}}
for rid in ('R1','R2','R3','R4'):
 rows=list(csv.DictReader((root/{SOURCE!r}/'runs'/rid/'validation_predictions.csv').open(encoding='utf-8-sig')))
 lens=[len(x) for x in tok(text_target=[r['prediction'] for r in rows],truncation=False)['input_ids']]
 values[rid]={{'retokenized_max_tokens':max(lens),'retokenized_at_or_above255':sum(n>=255 for n in lens)}}
result={{'model_revision':spec['model_revision'],'validation_token_lengths':values,'limits':'Retokenization of saved predictions is a cap-screening proxy; original generated token IDs were not retained.'}}
(root/{JOB!r}/'TOKEN_LENGTHS.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
''')
    write(ROOT/JOB/'TOKEN_LENGTHS.json',state);print(json.dumps(state))


def metrics(hyps, refs):
    return {'bleu': sacrebleu.corpus_bleu(hyps, [refs], tokenize='zh').score,
            'chrf++': sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
            'chrf0': sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score}


def cpu():
    plan = read(ROOT / SOURCE / 'PLAN.json')
    rows = list(csv.DictReader((ROOT / 'data_processed/amis_mandarin/validation.csv').open(encoding='utf-8-sig')))
    assert sha(ROOT / 'data_processed/amis_mandarin/validation.csv') == plan['data']['validation']['sha256']
    results, resolved = {}, {}
    for spec in plan['runs']:
        rid = spec['id']; folder = ROOT / JOB / 'artifacts/runs' / rid
        meta = read(folder / 'run_metadata.json')
        assert meta['run'] == spec and meta['protocol'] == plan['protocol'] and meta['data'] == plan['data']
        assert meta['save_reload_logits_match'] and not meta['test_used_for_selection']
        assert meta['archive_sha256'] == read(ROOT / 'outputs/experiments/amis_token_rtx_test/selection_lock.json')['runs'][rid]['archive_sha256']
        args = read(folder / 'resolved_training_arguments.json')
        resolved[rid] = args
        predictions = list(csv.DictReader((folder / 'validation_predictions.csv').open(encoding='utf-8-sig')))
        assert [(x['source'],x['target']) for x in predictions] == [(x['source'],x['target']) for x in rows]
        hyps = [x['prediction'] for x in predictions]; refs = [x['target'] for x in rows]
        observed = metrics(hyps, refs)
        saved = read(folder / 'validation_metrics.json')['scores']
        assert all(abs(observed[k]-saved[k]) < 1e-8 for k in observed)
        chinese = lambda x: sum('\u4e00' <= c <= '\u9fff' for c in x)
        subsets = {}
        for name, lo, hi in [('short',0,5),('medium',6,10),('long',11,100000)]:
            ids = [i for i,r in enumerate(rows) if lo <= len(r['source'].split()) <= hi]
            if ids:
                subsets[name] = {'rows':len(ids), 'source_word_range':[lo,hi],
                                 'scores':metrics([hyps[i] for i in ids],[refs[i] for i in ids])}
        counts = {'empty':sum(not x.strip() for x in hyps),
                  'exact_reference_matches':sum(h==r for h,r in zip(hyps,refs)),
                  'repeated_character_trigram':sum(any(x.count(x[j:j+3])>=3 for j in range(len(x)-2)) for x in hyps),
                  'prediction_reference_character_ratio':sum(chinese(x) for x in hyps)/sum(chinese(x) for x in refs),
                  'reference_whitespace_tokens_histogram':{str(n):sum(len(x.split())==n for x in refs) for n in range(1,6)},
                  'prediction_whitespace_tokens_histogram':{str(n):sum(len(x.split())==n for x in hyps) for n in range(1,6)}}
        chrf=sacrebleu.metrics.CHRF(word_order=2)
        stats=np.asarray(chrf._extract_corpus_statistics(hyps,[refs])).sum(0)
        counts['native_chrf_word_statistics']={str(order):{'hyp':int(stats[3*(5+order)]),
            'ref':int(stats[3*(5+order)+1]),'match':int(stats[3*(5+order)+2])} for order in (1,2)}
        results[rid] = {'method':spec['method'],'selected_epoch':meta['selected_epoch'],
            'selected_step':meta['selected_step'],'completed_epoch':meta['completed_epoch'],
            'stop_state':meta['stop_state'],'scores_recomputed':observed,'translation_counts':counts,
            'length_subsets':subsets,'bundle_sha256':meta['bundle_sha256'],'runtime':meta['runtime']}
        logs=read(folder/'final_trainer_state.json')['log_history']
        results[rid]['best_validation_in_training_logs']={metric:{k:v for k,v in max(
            [record for record in logs if metric in record],key=lambda record:record[metric]).items()
            if k in ('epoch','step','eval_bleu','eval_chrf++')} for metric in ('eval_bleu','eval_chrf++')}
    differing = {k:{rid:v.get(k) for rid,v in resolved.items()} for k in set().union(*(v.keys() for v in resolved.values()))
                 if any(v.get(k)!=resolved['R1'].get(k) for v in resolved.values())}
    audit = {'status':'METADATA_AND_VALIDATION_METRICS_VERIFIED','runs':results,
        'resolved_argument_differences':differing,'same_bundle':len({v['bundle_sha256'] for v in results.values()})==1,
        'same_runtime':all(v['runtime']==results['R1']['runtime'] for v in results.values()),
        'limits':'Length subsets and repetition are exploratory diagnostics, not new efficacy claims. Character repetition is a heuristic.'}
    write(ROOT / JOB / 'CPU_AUDIT.json', audit)
    sample=np.random.default_rng(7).choice(len(rows),12,replace=False).tolist()
    predictions={rid:list(csv.DictReader((ROOT/JOB/'artifacts/runs'/rid/'validation_predictions.csv').open(encoding='utf-8-sig'))) for rid in results}
    with (ROOT/JOB/'QUALITATIVE_SAMPLE.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['index','source','target','R1','R2','R3','R4']);writer.writeheader()
        for i in sample:writer.writerow({'index':i,**rows[i],**{rid:predictions[rid][i]['prediction'] for rid in results}})
    # Audit the already reported test metric's sparse word orders, without
    # selecting or changing any checkpoint from test results.
    test_counts={}
    for rid in results:
        frame=list(csv.DictReader((ROOT/'outputs/experiments/amis_token_rtx_test/runs'/rid/'test_predictions.csv').open(encoding='utf-8-sig')))
        metric=sacrebleu.metrics.CHRF(word_order=2)
        stats=np.asarray(metric._extract_corpus_statistics([r['prediction'] for r in frame],[[r['target'] for r in frame]])).sum(0)
        test_counts[rid]={'word1_hyp_ref_match':stats[18:21].tolist(),'word2_hyp_ref_match':stats[21:24].tolist(),
            'chrf0':sacrebleu.metrics.CHRF(word_order=0)._compute_f_score(stats[:18]),
            'chrf_plus_word1':sacrebleu.metrics.CHRF(word_order=1)._compute_f_score(stats[:21]),
            'chrf_plus_plus_word2':metric._compute_f_score(stats)}
    write(ROOT/JOB/'TEST_METRIC_WORD_COUNTS.json',test_counts)
    print(json.dumps(audit, ensure_ascii=False))


def launch():
    files = {'scripts/analysis/audit_token_pilot.py':Path(__file__).read_bytes()}
    files['scripts/revalidation/amis_v4_measurements.py']=(ROOT/'scripts/revalidation/amis_v4_measurements.py').read_bytes()
    shell = f'''#!/usr/bin/env bash
set -euo pipefail
source {REMOTE}/environment.sh
cd {REMOTE}
conda run --no-capture-output -n clrr python -u scripts/analysis/audit_token_pilot.py --stage gpu
'''
    files[JOB + '/run.sh'] = shell.encode()
    payload = {k:base64.b64encode(v).decode() for k,v in files.items()}
    state = remote(f'''import base64,json,subprocess
from pathlib import Path
root=Path({REMOTE!r});out=root/{JOB!r};out.mkdir(parents=True,exist_ok=True)
if (out/'worker.pid').exists():
 stat=Path('/proc')/(out/'worker.pid').read_text().strip()/'stat'
 assert not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'), 'Worker still alive'
 assert not any(out.glob('R*/RESULT.json')) and not (out/'COMPLETE.json').exists(), 'Use completed measurements; do not overwrite'
 if (out/'worker.log').exists():(out/'worker.log').rename(out/'initial_failure.log')
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=20000, 'Insufficient VRAM'
for rel,data in {payload!r}.items():
 p=(root/rel).resolve();assert p.is_relative_to(root);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
with (out/'worker.log').open('w') as f:
 p=subprocess.Popen(['bash',str(out/'run.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(out/'worker.pid').write_text(str(p.pid))
print(json.dumps({{'pid':p.pid,'free_MiB':free,'status':'LAUNCHED'}}))
''')
    write(ROOT / JOB / 'launch.json', state); print(json.dumps(state))


def monitor():
    state = remote(f'''import base64,json,subprocess
from pathlib import Path
out=Path({REMOTE!r})/{JOB!r}
pid=(out/'worker.pid').read_text().strip();stat=Path('/proc')/pid/'stat'
alive=stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
files={{p.relative_to(out).as_posix():json.loads(p.read_text()) for p in out.rglob('*.json') if p.name in ('RESULT.json','COMPLETE.json','PLAN.json')}}
worker=(Path({REMOTE!r})/'scripts/analysis/audit_token_pilot.py').read_bytes()
print(json.dumps({{'alive':alive,'files':files,'executed_worker':base64.b64encode(worker).decode(),'tail':(out/'worker.log').read_text(errors='replace').splitlines()[-15:],'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader'],text=True)}}))
''')
    snapshot=base64.b64decode(state.pop('executed_worker'))
    (ROOT/JOB/'executed_worker.py').write_bytes(snapshot)
    for name,value in state.pop('files').items(): write(ROOT / JOB / name,value)
    write(ROOT / JOB / 'status.json',state); print(json.dumps(state))


def summarize():
    from sacrebleu.significance import PairedTest
    out=ROOT/JOB;complete=read(out/'COMPLETE.json');assert complete['status']=='COMPLETE'
    assert sha(out/'executed_worker.py')==complete['worker_sha256']
    cpu_audit=read(out/'CPU_AUDIT.json');measurement=read(out/'PLAN.json')
    rows=list(csv.DictReader((ROOT/'data_processed/amis_mandarin/validation.csv').open(encoding='utf-8-sig')))
    refs=[rows[i]['target'] for i in measurement['indices']]
    os.environ['SACREBLEU_SEED']='42'
    metrics_by_name={'BLEU':sacrebleu.metrics.BLEU(tokenize='zh'),'chrF2':sacrebleu.metrics.CHRF(word_order=0)}
    indices=np.random.default_rng(42).integers(0,len(refs),size=(10000,len(refs)))
    summary={};comparisons=[]
    for rid in ('R1','R2','R3','R4'):
        result=read(out/rid/'RESULT.json');final=result['geometry']['final']
        assert result['archive_sha256']==measurement['checkpoint_lock'][rid]['archive_sha256']
        assert result['optimizer_steps']==0 and result['parameter_versions_unchanged']
        summary[rid]={'method':result['method'],'validation_scores':cpu_audit['runs'][rid]['scores_recomputed'],
            'source_covariance_effective_rank':final['source']['effective_rank_covariance_energy'],
            'target_covariance_effective_rank':final['target']['effective_rank_covariance_energy'],
            'centered_matched_cosine':final['centered_matched_cosine'],'centered_shuffled_cosine':final['centered_shuffled_cosine'],
            'weighted_lsr_ce_gradient_ratio_mean':float(np.mean([r['groups']['all_parameters']['lsr_to_ce_ratio'] for r in result['gradients']])),
            'ce_lsr_gradient_cosine_mean':float(np.mean([r['groups']['all_parameters']['gradient_cosine'] for r in result['gradients']])),
            'probe_ce_mean':float(np.mean([r['ce'] for r in result['gradients']])),
            'probe_sentence_lsr_mean':float(np.mean([r['sentence_lsr'] for r in result['gradients']])),
            'probe_token_lsr_mean':float(np.mean([r['token_lsr'] for r in result['gradients']])),
            'wrapper_loss_max_abs_error':max(r['wrapper_loss_max_abs_error'] for r in result['gradients']),
            'gradient_objective':result['gradients'][0]['gradient_objective'],
            'counterfactual_for_baseline':rid=='R1','peak_allocated_MiB':result['peak_allocated_MiB'],
            'source_residual_ratio_by_layer':{str(layer):float(np.mean([v for r in result['residuals'] if r['phase']=='source' and r['layer']==layer for v in r['ratios']])) for layer in range(1,13)}}
        if result['interventions']:
            conditions=result['interventions'];assert not conditions['on']['saved_prediction_mismatch_indices']
            names=['early_off','middle_off','late_off']
            _,native=PairedTest([('on',conditions['on']['predictions'])]+[(n,conditions[n]['predictions']) for n in names],metrics_by_name,[refs],test_type='bs',n_samples=10000,n_jobs=1)()
            for metric_name,metric in metrics_by_name.items():
                draws={};observed={}
                for condition in ['on']+names:
                    stats=np.asarray(metric._extract_corpus_statistics(conditions[condition]['predictions'],[refs]),dtype=np.float64)
                    observed[condition]=metric._compute_score_from_stats(stats.sum(0)).score
                    draws[condition]=np.array([metric._compute_score_from_stats(stats[ix].sum(0)).score for ix in indices])
                for pos,name in enumerate(names,1):
                    delta=draws[name]-draws['on']
                    comparisons.append({'model':rid,'condition':name,'metric':metric_name,
                        'delta':observed[name]-observed['on'],'paired_bootstrap_ci95':np.quantile(delta,[.025,.975]).tolist(),
                        'sacrebleu_p':native[metric_name][pos].p_value})
    running=0.
    for rank,i in enumerate(sorted(range(len(comparisons)),key=lambda i:comparisons[i]['sacrebleu_p'])):
        running=max(running,min(1.,(len(comparisons)-rank)*comparisons[i]['sacrebleu_p']));comparisons[i]['holm12_p']=running
    analysis={'status':'COMPLETE','summary':summary,'intervention_tests':comparisons,
        'statistical_family':'2 models x 3 intervention groups x 2 metrics = 12 exploratory comparisons; Holm correction',
        'limits':'One seed, selected endpoints, validation already used for checkpoint selection. Gradients are instantaneous, dropout disabled; not accumulated Adam updates. Inference intervention is not evidence that retraining the modified topology will improve.'}
    write(out/'ANALYSIS.json',analysis)
    lines=['# Token-LSR pilot audit — 11/10/2026','',
        'No training or optimizer steps. Same frozen 128 validation rows; four microbatch4 gradient probes. FP32, dropout/TF32 off. Public checkpoint revision pinned and archives SHA256 verified. Temporary checkpoint copies removed from RTX.','',
        '## Metadata and validation','',
        'All four runs reached epoch20 with patience10. Source bundle/runtime and resolved training arguments match except output/log paths. Recomputed validation metrics match saved values.','',
        '| Model | Selected epoch | Validation BLEU | chrF++ | chrF0 | Source effective covariance rank | Mean weighted LSR/CE grad ratio |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for rid,r in summary.items():
        val=r['validation_scores'];lines.append(f"| {rid} {r['method']} | {cpu_audit['runs'][rid]['selected_epoch']:.0f} | {val['bleu']:.4f} | {val['chrf++']:.4f} | {val['chrf0']:.4f} | {r['source_covariance_effective_rank']:.3f} | {100*r['weighted_lsr_ce_gradient_ratio_mean']:.4f}% |")
    lines+=['','Baseline gradient ratio is a counterfactual token-loss probe, not a loss used by baseline training. Rank is calculated from squared singular values of centered pooled vectors (128 rows, rank limit127), not token rank.','',
        '## Loss and alignment probes','',
        '| Model | CE | Sentence LSR | Token LSR | Centered matched cosine | Weighted aux/CE gradient cosine |',
        '|---|---:|---:|---:|---:|---:|']
    for rid,r in summary.items():lines.append(f"| {rid} | {r['probe_ce_mean']:.4f} | {r['probe_sentence_lsr_mean']:.4f} | {r['probe_token_lsr_mean']:.4f} | {r['centered_matched_cosine']:.4f} | {r['ce_lsr_gradient_cosine_mean']:.4f} |")
    lines+=['','CE/loss means above use16 validation examples in4 fixed microbatches, not the whole validation corpus. Baseline auxiliary objectives are counterfactual probes. Actual trained wrappers agree exactly with independently decomposed CE+weighted auxiliary losses in all16 model/batch cases. All gradients are finite, auxiliary decoder gradients are absent, and parameter version counters are unchanged.','',
        '## Inference interventions on the locked validation subset','',
        '| Model | Disabled layers (1-based) | Delta BLEU | 95% paired bootstrap delta CI | Raw p | Holm12 p |',
        '|---|---|---:|---|---:|---:|']
    layer_names={'early_off':'3–5','middle_off':'6–9','late_off':'10–12'}
    for r in comparisons:
        if r['metric']=='BLEU':lines.append(f"| {r['model']} | {layer_names[r['condition']]} | {r['delta']:+.4f} | [{r['paired_bootstrap_ci95'][0]:+.4f}, {r['paired_bootstrap_ci95'][1]:+.4f}] | {r['sacrebleu_p']:.4f} | {r['holm12_p']:.4f} |")
    counts=read(out/'TEST_METRIC_WORD_COUNTS.json')
    lines+=['','## Metric interpretation','',
        'Existing test chrF++ uses only3 reference word bigrams across575 rows. Baseline matches0; R2/R3/R4 each match1. This sparse order strongly affects the +4.6681 chrF++ gap; it is not broad evidence of improved character fidelity. Existing official scores are unchanged. Word-order0 chrF is included as a diagnostic.','',
        '| Model | Existing test chrF0 | Diagnostic chrF+ (word-order1) | Existing test chrF++ |',
        '|---|---:|---:|---:|']
    for rid,r in counts.items():lines.append(f"| {rid} | {r['chrf0']:.4f} | {r['chrf_plus_word1']:.4f} | {r['chrf_plus_plus_word2']:.4f} |")
    lengths=read(out/'TOKEN_LENGTHS.json')
    lines+=['','## Length and output checks','',
        f"Validation maximum untruncated token length: source{lengths['validation_token_lengths']['source']['max_tokens']}, target{lengths['validation_token_lengths']['target']['max_tokens']}; neither reaches256. Retokenized predictions reach at most47 tokens, far below the cap. Original generated token IDs were not retained, so this is a screening proxy for generation truncation.",
        'No empty validation translations. The predefined repeated-character-trigram heuristic flags2 of576 Token-Full outputs and0 for the other models; this is not a manual error rate. Twelve randomly selected validation examples are saved in QUALITATIVE_SAMPLE.csv.','',
        '## Interpretation','',
        'Token-LSR reduces the measured token objective compared with baseline, yet does not improve full-validation BLEU. Token-Full has a higher token loss and lower centered pooled matched cosine than Token-LSR in these probes. This supports investigating the interaction between alignment and rewiring; it does not establish causation or gold token-alignment accuracy.',
        'No catastrophic rank collapse is apparent in the pooled representation probe. Source residual RMS ratios are approximately6.7–8.4% at layers3–9 and0.8–3.2% at layers10–12. Rewiring is active; low efficacy is not explained by an inactive hook.',
        'All intervention CIs include0; none is significant after Holm12 correction. Opposite early-layer intervention signs across original Full and Token-Full do not support a universal early-layer removal rule.','',
        '## Decision boundaries','',
        'Token-Full fails the locked validation screening gate: its BLEU is below baseline, original Full and Token-LSR. No automatic extra seeds or A100 training. Changing selection from chrF++ to BLEU would select epoch19 for R4 and improve logged validation BLEU by only0.1114; it remains below baseline. This alternative is descriptive and does not replace the frozen selection.','',analysis['limits'],'',
        'Artifacts: CPU_AUDIT.json, ANALYSIS.json, R1–R4/RESULT.json, PLAN.json, QUALITATIVE_SAMPLE.csv, TOKEN_LENGTHS.json, TEST_METRIC_WORD_COUNTS.json, TRANSFER_SHA256.json and executed_worker.py.']
    (out/'AUDIT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(analysis))


def gpu():
    import torch
    import torch.nn.functional as F
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from experiment_runner.integrity import restore_archive
    from methods.original.clrr import CrossLayerResidualRewire
    from methods.common.helpers import _masked_mean_pool
    from methods.token_lsr.v1.lsr import token_alignment_loss
    from methods.token_lsr.v1.lsr import TokenLSRSeq2SeqLM
    from methods.original.lsr import JEPAGuidedSeq2SeqLM
    sys.path.insert(0,str(ROOT/'scripts/revalidation'))
    from amis_v4_measurements import spectrum, gradient_group
    from contextlib import nullcontext
    assert ROOT.resolve() == Path(REMOTE) and Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    plan=read(ROOT/SOURCE/'PLAN.json'); lock=read(ROOT/'outputs/experiments/amis_token_rtx_test/selection_lock.json')
    out=ROOT/JOB; assert not (out/'COMPLETE.json').exists()
    manifest=read(ROOT/SOURCE/'bundle_manifest.json')
    assert all(sha(ROOT/name)==digest for name,digest in manifest.items()), 'Frozen source changed'
    data=ROOT/'data_processed/amis_mandarin/validation.csv'
    assert sha(data)==plan['data']['validation']['sha256']
    rows=list(csv.DictReader(data.open(encoding='utf-8-sig')))
    indices=np.random.default_rng(42).permutation(len(rows))[:128].tolist()
    batches=np.random.default_rng(43).permutation(len(rows))[:16].reshape(4,4).tolist()
    write(out/'PLAN.json',{'indices':indices,'gradient_batches':batches,'checkpoint_lock':lock['runs'],
        'precision':'FP32','tf32':False,'dropout':False,'optimizer_steps':0,
        'intervention_layers_1based':{'early_off':[3,4,5],'middle_off':[6,7,8,9],'late_off':[10,11,12]},
        'limits':'128 frozen validation rows for interventions/geometry; four microbatch4 gradients; selected checkpoints only; not training-time causal proof.'})
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.__version__.split('+')[0]==plan['runtime']['torch'] and transformers.__version__==plan['runtime']['transformers']
    all_results={}
    for spec in plan['runs']:
        rid=spec['id'];dest=out/rid;dest.mkdir(parents=True,exist_ok=True)
        free=torch.cuda.mem_get_info()[0]//2**20;print('GPU_BEFORE',rid,free,flush=True);assert free>=20000
        selected=lock['runs'][rid]; restore=dest/'restore';extracted=dest/'extracted'
        archive=Path(hf_hub_download(plan['repo'],selected['archive_path'],revision=selected['revision'],token=False,local_dir=restore))
        assert sha(archive)==selected['archive_sha256']
        restore_archive(archive,extracted)
        base,loading=AutoModelForSeq2SeqLM.from_pretrained(extracted,torch_dtype=torch.float32,output_loading_info=True)
        assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        tok=AutoTokenizer.from_pretrained(extracted,use_fast=True,src_lang=spec['src_lang'],tgt_lang=spec['tgt_lang'])
        base.gradient_checkpointing_disable();base.config.use_cache=False
        routed=rid in ('R2','R4')
        model=CrossLayerResidualRewire(base,distance=spec['distance'],strength=spec['strength'],stack='encoder') if routed else base
        model.cuda().eval(); versions={n:p._version for n,p in model.named_parameters()}
        fresh=lambda:model._fresh_cache() if routed else nullcontext()
        def tokens(ids,target=False):
            texts=[rows[i]['target' if target else 'source'] for i in ids]
            b=tok(text_target=texts,padding=True,pad_to_multiple_of=8,truncation=True,max_length=256,return_tensors='pt') if target else tok(texts,padding=True,pad_to_multiple_of=8,truncation=True,max_length=256,return_tensors='pt')
            return {k:v.cuda() for k,v in b.items() if k in ('input_ids','attention_mask')}
        def content(b):
            mask=b['attention_mask'].bool().clone()
            for tid in tok.all_special_ids: mask &= b['input_ids']!=tid
            assert mask.any(1).all();return mask
        residuals=[]; pooled={}; pre={}; holder={}; handles=[]
        for i,layer in enumerate(base.get_encoder().layers):
            def before(_m,_inp,value,i=i): pre[i]=(value[0] if isinstance(value,(tuple,list)) else value).detach()
            def after(_m,_inp,value,i=i):
                post=(value[0] if isinstance(value,(tuple,list)) else value).detach();raw=pre.pop(i)
                mask=holder['mask'].float();rms=lambda h:((h.float().square()*mask[...,None]).sum((1,2))/(mask.sum(1)*h.shape[-1])).sqrt()
                ratio=rms(post-raw)/rms(raw).clamp_min(1e-12)
                residuals.append({'phase':holder['phase'],'layer':i+1,'ratios':ratio.cpu().tolist()})
                pooled.setdefault((holder['phase'],i+1),[]).append(_masked_mean_pool(post,mask).cpu().numpy())
            handles.extend([layer.register_forward_hook(before,prepend=True),layer.register_forward_hook(after)])
        reps={}
        with torch.no_grad():
            for target in (False,True):
                phase='target' if target else 'source';values=[]
                for start in range(0,len(indices),4):
                    b=tokens(indices[start:start+4],target);holder.update(mask=b['attention_mask'],phase=phase)
                    with fresh(): h=model.get_encoder()(**b).last_hidden_state
                    values.append(_masked_mean_pool(h,b['attention_mask']).cpu().numpy())
                reps[phase]=np.concatenate(values)
        for handle in handles:handle.remove()
        geometry={f'{phase}_layer{layer}':spectrum(np.concatenate(v)) for (phase,layer),v in pooled.items()}
        s,t=map(torch.tensor,(reps['source'],reps['target']));sc=s-s.mean(0);tc=t-t.mean(0)
        cosine=lambda a,b:float(F.cosine_similarity(a,b,dim=-1).mean())
        geometry['final']={'source':spectrum(reps['source']),'target':spectrum(reps['target']),
            'matched_cosine':cosine(s,t),'centered_matched_cosine':cosine(sc,tc),
            'shuffled_cosine':float(np.mean([cosine(s,t[np.random.default_rng(42+j).permutation(len(t))]) for j in range(100)])),
            'centered_shuffled_cosine':float(np.mean([cosine(sc,tc[np.random.default_rng(42+j).permutation(len(t))]) for j in range(100)]))}
        named=list(model.named_parameters());params=[p for _,p in named]
        groups=['layer'+n.split('encoder.layers.')[1].split('.')[0] if 'encoder.layers.' in n else ('encoder_other' if 'encoder' in n or 'shared' in n else 'decoder') for n,_ in named]
        gradients=[]
        for ids in batches:
            sb,tb=tokens(ids),tokens(ids,True);labels=tb['input_ids'].masked_fill(tb['input_ids']==tok.pad_token_id,-100)
            outputs=model(**sb,labels=labels,use_cache=False)
            with torch.no_grad(),fresh():th=model.get_encoder()(**tb).last_hidden_state.detach()
            sentence=(1-F.cosine_similarity(_masked_mean_pool(outputs.encoder_last_hidden_state,sb['attention_mask']),_masked_mean_pool(th,tb['attention_mask']),dim=-1)).mean()
            token=token_alignment_loss(outputs.encoder_last_hidden_state,th,content(sb),content(tb))
            active=sentence if rid=='R2' else token
            with torch.no_grad():
                if rid=='R1': verified_loss=model(**sb,labels=labels,use_cache=False).loss
                elif rid=='R2': verified_loss=JEPAGuidedSeq2SeqLM(model,jepa_weight=.1)(**sb,labels=labels,use_cache=False).loss
                else: verified_loss=TokenLSRSeq2SeqLM(model,tok.all_special_ids)(**sb,labels=labels,use_cache=False).loss
            expected=outputs.loss if rid=='R1' else outputs.loss+.1*active
            loss_error=float((verified_loss-expected).detach().abs())
            assert loss_error<1e-6, ('Actual wrapper loss disagrees',rid,loss_error)
            ce=torch.autograd.grad(outputs.loss,params,retain_graph=True,allow_unused=True)
            aux=torch.autograd.grad(.1*active,params,allow_unused=True)
            assert all(g is None or torch.isfinite(g).all() for g in ce+aux)
            summary={g:gradient_group(groups,ce,aux,g) for g in ['all_parameters']+sorted(set(groups))}
            assert all(g is None or torch.count_nonzero(g)==0 for group,g in zip(groups,aux) if group=='decoder')
            gradients.append({'indices':ids,'ce':float(outputs.loss.detach()),'sentence_lsr':float(sentence.detach()),'token_lsr':float(token.detach()),
                'gradient_objective':'sentence' if rid=='R2' else 'token','auxiliary_is_counterfactual_for_baseline':rid=='R1','weighted_lambda':.1,'wrapper_loss_max_abs_error':loss_error,'groups':summary})
            del ce,aux,outputs,sentence,token,active,th,expected,verified_loss;gc.collect();torch.cuda.empty_cache()
            print('GRADIENT',rid,len(gradients),flush=True)
        interventions={}
        if routed:
            disabled=set()
            for h in model._handles:h.remove()
            model._handles=[]
            for i,layer in enumerate(base.get_encoder().layers):
                hook=model._make_hook('encoder',i)
                def intervene(module,inputs,output,i=i,hook=hook):
                    strength=model.strength
                    if i+1 in disabled:model.strength=0.
                    try:return hook(module,inputs,output)
                    finally:model.strength=strength
                model._handles.append(layer.register_forward_hook(intervene))
            conditions={'on':[],'early_off':[3,4,5],'middle_off':[6,7,8,9],'late_off':[10,11,12]}
            for condition,layers in conditions.items():
                disabled.clear();disabled.update(layers);by_index={}
                selected_indices=set(indices)
                with torch.no_grad():
                    for start in range(0,len(rows),2):
                        pair=list(range(start,min(start+2,len(rows))))
                        if not selected_indices.intersection(pair):continue
                        generated=model.generate(**tokens(pair),num_beams=1,max_length=256,forced_bos_token_id=tok.convert_tokens_to_ids(spec['tgt_lang']),decoder_start_token_id=tok.eos_token_id)
                        by_index.update(zip(pair,(x.strip() for x in tok.batch_decode(generated,skip_special_tokens=True))))
                hyps=[by_index[i] for i in indices]
                interventions[condition]={'scores':metrics(hyps,[rows[i]['target'] for i in indices]),'predictions':hyps}
                if condition=='on':
                    saved=list(csv.DictReader((ROOT/SOURCE/'runs'/rid/'validation_predictions.csv').open(encoding='utf-8-sig')))
                    mismatches=[i for i,h in zip(indices,hyps) if h!=saved[i]['prediction']]
                    interventions[condition]['saved_prediction_mismatch_indices']=mismatches
                    assert not mismatches, ('Frozen checkpoint generation replay mismatch',rid,mismatches)
                print('INTERVENTION',rid,condition,interventions[condition]['scores'],flush=True)
            disabled.clear()
        assert versions=={n:p._version for n,p in model.named_parameters()}
        result={'method':spec['method'],'archive_sha256':sha(archive),'strict_loading':loading,'residuals':residuals,'geometry':geometry,
            'gradients':gradients,'interventions':interventions,'optimizer_steps':0,'parameter_versions_unchanged':True,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()//2**20}
        write(dest/'RESULT.json',result);all_results[rid]=result
        if routed:
            for handle in model._handles:handle.remove()
            model._handles=[]
            del hook,intervene
        del params,named,model,base;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        for owned in (restore,extracted):
            assert owned.resolve().is_relative_to(out.resolve());shutil.rmtree(owned)
        print('MODEL_COMPLETE_CLEANED',rid,flush=True)
    write(out/'COMPLETE.json',{'status':'COMPLETE','runs':list(all_results),'gpu':torch.cuda.get_device_name(0),'optimizer_steps':0,
        'worker_sha256':sha(Path(__file__)),'completed_unix':time.time(),'weights_cleanup':'Only worker-owned downloaded archives/extractions removed after verified measurement.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['collect','cpu','lengths','launch','monitor','gpu','summarize'],required=True)
    args=parser.parse_args();globals()[args.stage]()
