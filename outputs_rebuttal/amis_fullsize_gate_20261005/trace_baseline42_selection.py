"""Read-only baseline42 selection audit; optional small HF metadata only."""
import argparse
import ast
import hashlib
import json
import re
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--hf',action='store_true');args=parser.parse_args()
conf=ROOT/'outputs_rebuttal/amis_confirmation_20261003'
name='mbart-amis-baseline-seed42'
states=[]
for p in sorted((conf/'runs'/name).rglob('trainer_state.json')):
    r=read(p)
    states.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p),'global_step':r['global_step'],
      'epoch':r['epoch'],'best_step':r.get('best_global_step'),'best_checkpoint':r.get('best_model_checkpoint'),
      'best_metric':r.get('best_metric'),'last_evaluation':next((h for h in reversed(r['log_history']) if 'eval_chrf++' in h),None)})
log=conf/(name+'.log');text=log.read_text(encoding='utf-8-sig')
evaluations=[dict(step=int(s),loss=float(l),bleu=float(b),chrfpp=float(c)) for s,l,b,c in re.findall(r'\[eval @ step (\d+)\] eval_loss=([\d.]+) \| eval_bleu=([\d.]+) \| eval_chrf\+\+=([\d.]+)',text)]
record={'date':'2026-10-05','local_trainer_states':states,'original_log_file':str(log.relative_to(ROOT)),
 'original_log_sha256':sha(log),'original_log_last_step_mentions':re.findall(r'\[step=(\d+)',text)[-5:],
 'original_log_evaluations_rounded':evaluations,'original_log_has_selected_checkpoint_json':'"best_model_checkpoint"' in text,
 'infrastructure_recovery':read(conf/'infrastructure_recovery.json'),
 'selected_archive_receipt':read(conf/'best_archives'/f'{name}.verified.json'),
 'recovered_metrics_selection':read(conf/'runs'/name/'metrics.json')['best_model_checkpoint'],
 'v5_checkpoint_metadata_files':list(read(ROOT/'outputs_rebuttal/amis_routing_diagnostics_20261004_v5/baseline_archive_audit_v3'/name/'checkpoint_metadata.json').get('metadata',{})),
 'local_conclusion':'UNKNOWN: partial states/log do not establish selected archive original step; snapshot step504 is not proof selected best at completed training.',
 'HF_audit':{'status':'NOT_REQUESTED'}}
if args.hf:
    from huggingface_hub import HfApi, get_token, hf_hub_download
    token=get_token()
    if not token:
        # User authorized token in launcher; parse constants, never execute or print it.
        tree=ast.parse((ROOT/'scratch/launch_peft_colab.py').read_text(encoding='utf-8-sig'))
        candidates=list({n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('hf_') and len(n.value)>25})
        if len(candidates)!=1:raise RuntimeError('Authorized HF credential not uniquely identifiable; no credential output.')
        token=candidates[0]
    api=HfApi(token=token);repo='FiveC/amis-rewire-checkpoints'
    entries=list(api.list_repo_tree(repo,recursive=True,repo_type='model'))
    candidate=[];small=[]
    for e in entries:
        path=e.path
        if 'amis_confirmation_20261003' not in path:continue
        if name not in path and not any(path.endswith('/'+x) for x in ['protocol.json','infrastructure_recovery.json','source_manifest.json']):continue
        if not hasattr(e,'size'):continue
        candidate.append({'path':path,'bytes':e.size})
        if e.size>1024*1024 or Path(path).suffix not in ['.json','.log','.txt']:continue
        downloaded=Path(hf_hub_download(repo,path,token=token,local_dir=OUT/'hf_small_metadata'))
        item={'path':path,'bytes':e.size,'sha256':sha(downloaded)}
        if Path(path).suffix=='.json':
            d=read(downloaded)
            if isinstance(d,dict):item['selection_fields']={k:d.get(k) for k in ['best_global_step','global_step','epoch','best_model_checkpoint','best_metric','baseline_original_selected_step_unavailable'] if k in d}
        small.append(item)
    record['HF_audit']={'status':'READ_SMALL_METADATA_ONLY','candidate_files':candidate,'downloaded_small_files':small,
      'note':'No weights/archive download. HF presence and selected best-step fields inspected only; absent fields do not prove global absence outside inspected matching namespace.'}
    # Inspect ZIP directory only with a bounded HTTP Range; never accept full-body fallback.
    import requests
    from huggingface_hub import hf_hub_url
    archive=next((e for e in candidate if e['path'].endswith(name+'.zip')),None)
    if archive:
        size=archive['bytes'];start=max(0,size-65536)
        with requests.get(hf_hub_url(repo,archive['path']),headers={'Authorization':'Bearer '+token,'Range':f'bytes={start}-{size-1}'},stream=True,timeout=45) as response:
            if response.status_code!=206:
                record['HF_audit']['zip_directory']={'status':'RANGE_UNSUPPORTED_BODY_NOT_READ','http_status':response.status_code}
            else:
                content_range=response.headers.get('Content-Range','')
                if content_range!=f'bytes {start}-{size-1}/{size}':raise RuntimeError('Unexpected range response; refusing body.')
                tail=response.raw.read(65537)
                if len(tail)>65536:raise RuntimeError('Range response exceeded requested metadata budget.')
                end=tail.rfind(b'PK\x05\x06')
                if end<0:raise RuntimeError('ZIP directory footer unavailable in bounded range.')
                footer=struct.unpack('<4s4H2LH',tail[end:end+22]);cdsize,cdoffset=footer[5:7]
                begin=cdoffset-start
                if begin<0 or begin+cdsize>len(tail):raise RuntimeError('ZIP directory exceeds bounded metadata budget.')
                cursor=begin;members=[]
                while cursor<begin+cdsize:
                    fields=struct.unpack('<4s6H3L5H2L',tail[cursor:cursor+46])
                    if fields[0]!=b'PK\x01\x02':raise RuntimeError('Invalid central directory entry.')
                    fl,el,cl=fields[10:13]
                    member=tail[cursor+46:cursor+46+fl].decode('utf-8')
                    members.append({'name':member,'compressed_bytes':None if fields[8]==0xffffffff else fields[8],
                       'uncompressed_bytes':None if fields[9]==0xffffffff else fields[9],
                       'zip64_sizes_not_expanded':fields[8]==0xffffffff or fields[9]==0xffffffff})
                    cursor+=46+fl+el+cl
                record['HF_audit']['zip_directory']={'status':'BOUNDED_RANGE_PASS','bytes_read':len(tail),'members':members,
                    'has_trainer_state':any(m['name'].endswith('trainer_state.json') for m in members)}
(OUT/'baseline42_selection_trace.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'local_states':[(r['global_step'],r['best_metric']) for r in states],
 'log_evals':[(r['step'],r['chrfpp']) for r in evaluations], 'selected_checkpoint_string':record['recovered_metrics_selection'],
 'HF_status':record['HF_audit']['status'],'HF_files':len(record['HF_audit'].get('candidate_files',[]))},indent=2))
