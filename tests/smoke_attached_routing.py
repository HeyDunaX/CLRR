"""Verify identical forwards and checkpoint-consistent added routing gradients."""
from pathlib import Path
import copy
import json
import sys
import torch
from transformers import MBartConfig, MBartForConditionalGeneration

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire.ablation import AttachedSourceResidualRewire
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM


def main():
    torch.set_num_threads(2)
    device='cuda' if '--cuda' in sys.argv else 'cpu'
    config=MBartConfig(vocab_size=32,d_model=16,encoder_layers=4,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,
        decoder_ffn_dim=32,max_position_embeddings=32,pad_token_id=1,bos_token_id=0,
        eos_token_id=2,decoder_start_token_id=2,dropout=.1,attention_dropout=.1,activation_dropout=.1,
        encoder_layerdrop=0.,decoder_layerdrop=0.,use_cache=False)
    torch.manual_seed(42)
    initial=MBartForConditionalGeneration(config)
    batch={'input_ids':torch.tensor([[4,6,7,2],[4,8,9,2]],device=device),
           'attention_mask':torch.ones(2,4,dtype=torch.long,device=device),
           'labels':torch.tensor([[5,10,11,2],[5,12,13,2]],device=device)}
    results={}
    for attached in (False,True):
        for checkpointed in (False,True):
            raw=copy.deepcopy(initial).to(device)
            cls=AttachedSourceResidualRewire if attached else CrossLayerResidualRewire
            route=cls(raw,distance=2,strength=.2,stack='encoder')
            model=JEPAGuidedSeq2SeqLM(route,jepa_weight=.1).train()
            if checkpointed:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
            torch.manual_seed(2026)
            output=model(**batch)
            assert all(not values for values in route._cache.values())
            output.loss.backward()
            results[attached,checkpointed]=(output.logits.detach().cpu(),output.loss.detach().cpu(),
                {name:p.grad.detach().cpu().clone() for name,p in raw.named_parameters() if p.grad is not None})
            assert all(torch.isfinite(g).all() for g in results[attached,checkpointed][2].values())
    for attached in (False,True):
        plain,checked=results[attached,False],results[attached,True]
        torch.testing.assert_close(plain[0],checked[0],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(plain[1],checked[1],rtol=1e-5,atol=1e-6)
        assert plain[2].keys()==checked[2].keys()
        errors={name:float((plain[2][name]-checked[2][name]).abs().max()) for name in plain[2]}
        print('CHECKPOINT_GRADIENT_ERRORS',attached,sorted(errors.items(),key=lambda p:p[1],reverse=True)[:4],flush=True)
        for name in plain[2]:
            torch.testing.assert_close(plain[2][name],checked[2][name],rtol=1e-4,atol=1e-6,msg=name)
    torch.testing.assert_close(results[False,False][0],results[True,False][0],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(results[False,False][1],results[True,False][1],rtol=1e-5,atol=1e-6)
    difference=sum((results[False,False][2][name]-results[True,False][2][name]).abs().sum().item()
        for name in results[False,False][2])
    assert difference>1e-6,'Added source gradient path was not activated'
    print(json.dumps(dict(passed=True,device=device,forward_equivalent=True,
        checkpoint_gradients_equivalent=True,added_source_gradient_difference=difference,
        lsr_target_remains_no_grad=True)))


if __name__=='__main__':
    main()
