"""Check a worst-length Full-model batch before choosing one shared microbatch."""
from pathlib import Path
import gc
import json
import sys
import time
import torch
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from amis_rewire.modeling import load_model


def main():
    path=snapshot_download(repo_id='facebook/mbart-large-50-many-to-many-mmt',
       revision='e30b6cb8eb0d43a0b73cab73c7676b9863223a30',
       allow_patterns=['config.json','generation_config.json','model.safetensors'],token=False)
    results=[]
    for size in [32,16]:
        model=optimizer=loss=inputs=labels=None
        try:
            torch.manual_seed(42)
            model=load_model(path,'jepa-clrr-enc',distance=2,strength=.1,stack='encoder',jepa_weight=.1).cuda().train()
            model.gradient_checkpointing_enable();model.config.use_cache=False
            optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,fused=True)
            inputs=torch.randint(10,1000,(size,256),device='cuda');labels=inputs.clone()
            torch.cuda.reset_peak_memory_stats();started=time.time()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=labels).loss
            loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            results.append({'microbatch':size,'passed':True,'worst_length':256,
                            'peak_gib':torch.cuda.max_memory_allocated()/1024**3,'elapsed_seconds':time.time()-started})
            selected=size
            break
        except torch.cuda.OutOfMemoryError:
            results.append({'microbatch':size,'passed':False,'reason':'CUDA OOM at length256'})
        finally:
            del model,optimizer,loss,inputs,labels
            gc.collect();torch.cuda.empty_cache()
    else:
        raise RuntimeError('Neither proposed microbatch fits; do not start pilot')
    out=ROOT/'outputs_rebuttal/followup_20261003'
    record={'selected_microbatch':selected,'gradient_accumulation':128//selected,'global_batch':128,
            'scope':'throwaway pretrained Full-model optimizer smoke at worst length; no quality result or saved calibrated weights',
            'trials':results}
    (out/'batch_calibration.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
