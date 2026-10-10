"""One frozen baseline rank measurement, wholly inside the user's RTX workspace."""
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
for name, relative in [('CUDA_CACHE_PATH','cache/cuda'),('TRITON_CACHE_DIR','cache/triton')]:
    path=Path('/workspace/ura/thanhdat')/relative
    path.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(path)
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=Path('/workspace/ura/thanhdat').resolve()


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert ROOT.resolve().is_relative_to(WORKSPACE)
    assert Path(sys.prefix).resolve().is_relative_to(WORKSPACE/'conda')
    for key in ('HF_HOME','TORCH_HOME','PIP_CACHE_DIR','CONDA_ENVS_PATH','CONDA_PKGS_DIRS','TMPDIR'):
        assert Path(os.environ[key]).resolve().is_relative_to(WORKSPACE),(key,os.environ.get(key))
    plan=json.loads((ROOT/'PLAN.json').read_text())
    entry=next(e for e in plan['entries'] if e['id']=='BASE')
    csv_path=ROOT/'data_processed/amis_mandarin/validation.csv'
    assert digest(csv_path)==plan['validation_sha256']
    out=ROOT/'outputs/rtx_rank_smoke';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists()
    free,total=torch.cuda.mem_get_info()
    print('GPU_BEFORE',json.dumps({'name':torch.cuda.get_device_name(0),'free_MiB':free//2**20,'total_MiB':total//2**20}),flush=True)
    assert free>=8*2**30,'Insufficient free GPU memory; do not evict other users'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    archive=Path(hf_hub_download('FiveC/amis-rewire-checkpoints',entry['archive_path'],revision=plan['revision']))
    assert archive.resolve().is_relative_to(WORKSPACE)
    assert digest(archive)==entry['archive_sha256'] and archive.stat().st_size==entry['archive_bytes']
    extracted=out/'extracted';extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():assert (extracted/n).resolve().is_relative_to(extracted.resolve())
        z.extractall(extracted)
    folders=[p.parent for p in extracted.rglob('config.json') if (p.parent/'pytorch_model.bin').exists() or (p.parent/'model.safetensors').exists()]
    assert len(folders)==1
    model,info=AutoModelForSeq2SeqLM.from_pretrained(folders[0],output_loading_info=True,torch_dtype=torch.float32)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.gradient_checkpointing_disable();model=model.cuda().eval()
    tok=AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',revision=plan['model_revision'],src_lang='tl_XX',tgt_lang='zh_CN')
    with csv_path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    vectors=[]
    for start in range(0,128,32):
        idx=plan['representation_indices'][start:start+32]
        batch=tok([rows[i]['source'] for i in idx],padding=True,truncation=True,max_length=256,return_tensors='pt')
        batch={k:v.cuda() for k,v in batch.items() if k in ('input_ids','attention_mask')}
        with torch.no_grad():h=model.get_encoder()(**batch).last_hidden_state.float()
        mask=batch['attention_mask'].float().unsqueeze(-1)
        pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1e-6)
        vectors.append(pooled.cpu().numpy())
    matrix=np.concatenate(vectors).astype(np.float64);matrix-=matrix.mean(0,keepdims=True)
    singular=np.linalg.svd(matrix,compute_uv=False);energy=singular**2
    q=energy/energy.sum();positive=q[q>0]
    rank=float(np.exp(-(positive*np.log(positive)).sum()))
    reference=plan['a100_source_covariance_rank']
    result={'status':'PASS' if abs(rank-reference)<1e-3 else 'NEEDS_NUMERICAL_AUDIT',
        'measurement':'baseline42 encoder final-LayerNorm pooled source covariance effective rank',
        'rows':128,'batch_size':32,'precision':'FP32','dropout':False,'tf32':False,
        'rank':rank,'a100_rank':reference,'delta':rank-reference,'tolerance_absolute':1e-3,
        'archive_sha256':entry['archive_sha256'],'validation_sha256':plan['validation_sha256'],
        'indices':plan['representation_indices'],'strict_loading':info,'optimizer_steps':0,
        'gpu':torch.cuda.get_device_name(0),'peak_allocated_MiB':torch.cuda.max_memory_allocated()//2**20,
        'torch':torch.__version__,'conda_prefix':sys.prefix,'worker_sha256':digest(__file__),'completed_unix':time.time()}
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    print('RANK_RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
