"""GPU gate: probes preserve RNG, train/cache/checkpointing state for every branch."""
from pathlib import Path
from types import SimpleNamespace
import json
import random
import sys
import tempfile
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import MBartConfig, MBartForConditionalGeneration, PreTrainedTokenizerFast

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts/revalidation'))
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from measure_training_trajectory import TrajectoryCallback


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    tok=Tokenizer(WordLevel({'<s>':0,'<pad>':1,'</s>':2,'<unk>':3,'a':4,'b':5},unk_token='<unk>'))
    tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='<pad>',eos_token='</s>',bos_token='<s>',unk_token='<unk>',model_input_names=['input_ids','attention_mask'])
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        data=Path(tmp)/'data';data.mkdir()
        (data/'validation.csv').write_text('source,target\n'+'a b,b a\n'*32)
        for method in ['baseline','clrr-enc','jepa','jepa-clrr-enc']:
            torch.manual_seed(42)
            raw=MBartForConditionalGeneration(MBartConfig(vocab_size=8,d_model=16,encoder_layers=3,decoder_layers=1,
                encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,
                max_position_embeddings=32,pad_token_id=1,eos_token_id=2,bos_token_id=0,decoder_start_token_id=2)).cuda()
            model=CrossLayerResidualRewire(raw,distance=2,strength=.1,stack='encoder') if 'clrr' in method else raw
            if method in ['jepa','jepa-clrr-enc']:model=JEPAGuidedSeq2SeqLM(model,jepa_weight=.1)
            model.gradient_checkpointing_enable();model.config.use_cache=False;model.train()
            cpu=torch.get_rng_state().clone();gpu=[v.clone() for v in torch.cuda.get_rng_state_all()]
            pyr=random.getstate();npr=np.random.get_state()
            callback=TrajectoryCallback(Path(tmp)/method,SimpleNamespace(data_dir=str(data),method=method,seed=42))
            callback.on_save(SimpleNamespace(gradient_checkpointing_kwargs=None),
                SimpleNamespace(epoch=1,global_step=1,max_steps=20),SimpleNamespace(should_training_stop=False),model=model,processing_class=tokenizer)
            assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
            assert pyr==random.getstate() and np.array_equal(npr[1],np.random.get_state()[1])
            assert model.training and not model.config.use_cache and raw.is_gradient_checkpointing
            assert (Path(tmp)/method/'epoch-01-step-1/DONE.json').exists()
            # A training backward remains possible after the probe restores checkpointing.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(input_ids=torch.tensor([[4,5]],device='cuda'),attention_mask=torch.ones((1,2),device='cuda'),labels=torch.tensor([[5,4]],device='cuda')).loss
            loss.backward()
            assert any(p.grad is not None and p.grad.norm()>0 for p in raw.parameters())
            results.append({'method':method,'passed':True,'rng_and_model_state_restored':True,'backward_after_probe':True})
            del model,raw,loss;torch.cuda.empty_cache()
    out=ROOT/'outputs_rebuttal/amis_confirmation_20261003'
    out.mkdir(parents=True,exist_ok=True)
    (out/'trajectory_smoke_gpu.json').write_text(json.dumps(results,indent=2))
    print('ALL_FOUR_TRAJECTORY_BRANCHES_PASS',flush=True)


if __name__=='__main__':main()
