"""Check actual encoder dtypes through Accelerate and the production wrappers."""
from pathlib import Path
import json
import sys
import torch
from accelerate import Accelerator
from transformers import MBartConfig, MBartForConditionalGeneration
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from amis_rewire.modeling import CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM
from amis_rewire.evaluation import generation_precision


def main():
    assert torch.cuda.is_available()
    accelerator = Accelerator(mixed_precision='bf16')
    rows = []
    for method in ['baseline', 'clrr', 'lsr', 'full']:
        config = MBartConfig(vocab_size=32, d_model=16, encoder_layers=4, decoder_layers=2,
                            encoder_attention_heads=2, decoder_attention_heads=2,
                            encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32,
                            pad_token_id=0, bos_token_id=1, eos_token_id=2,
                            decoder_start_token_id=2, forced_eos_token_id=2)
        raw = MBartForConditionalGeneration(config)
        model = CrossLayerResidualRewire(raw, distance=2, strength=.1) if method in ['clrr','full'] else raw
        if method in ['lsr','full']:
            model = JEPAGuidedSeq2SeqLM(model, jepa_weight=.1)
        model = accelerator.prepare_model(model).eval()
        observed = {'encoder':[], 'decoder':[]}
        # Dense projection output makes the active autocast visible (LayerNorm output may be FP32).
        handles = [raw.model.encoder.layers[0].self_attn.q_proj.register_forward_hook(
            lambda _m,_i,o: observed['encoder'].append(str(o.dtype))),
            raw.model.decoder.layers[0].self_attn.q_proj.register_forward_hook(
            lambda _m,_i,o: observed['decoder'].append(str(o.dtype)))]
        inputs = {'input_ids':torch.tensor([[4,5,2]],device='cuda'),
                  'attention_mask':torch.ones((1,3),dtype=torch.long,device='cuda')}
        with torch.no_grad():
            model.generate(**inputs,max_length=6,num_beams=1)
        historical = {key:values[0] for key,values in observed.items()}
        for values in observed.values(): values.clear()
        with torch.no_grad(), generation_precision(model,'fp32'):
            model.generate(**inputs,max_length=6,num_beams=1)
        fp32 = {key:values[0] for key,values in observed.items()}
        for values in observed.values(): values.clear()
        with torch.no_grad(), generation_precision(model,'bf16'):
            model.generate(**inputs,max_length=6,num_beams=1)
        bf16 = {key:values[0] for key,values in observed.items()}
        assert all(dtype=='torch.float32' for dtype in fp32.values())
        assert all(dtype=='torch.bfloat16' for dtype in bf16.values())
        for handle in handles: handle.remove()
        rows.append({'method':method,'default_accelerate_generation_projection':historical,
                     'explicit_fp32_projection':fp32,'explicit_bf16_projection':bf16})
        print(rows[-1],flush=True)
        del model,raw
        torch.cuda.empty_cache()
    out = Path('outputs_rebuttal/followup_20261003')
    out.mkdir(parents=True,exist_ok=True)
    (out/'generation_precision_probe.json').write_text(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
