"""Exercise all four mBART training branches with the explicit precision Trainer."""
from pathlib import Path
import copy
import json
import sys
import tempfile
from unittest.mock import patch
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import MBartConfig, MBartForConditionalGeneration, PreTrainedTokenizerFast, AutoModelForSeq2SeqLM
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire import train


class MBartTinyTokenizer(PreTrainedTokenizerFast):
    lang_code_to_id={'tr_TR':4,'en_XX':5}


def main():
    torch.set_num_threads(2)
    cuda='--cuda' in sys.argv
    if cuda: assert torch.cuda.is_available()
    config=MBartConfig(vocab_size=32,d_model=16,encoder_layers=3,decoder_layers=1,
                       encoder_attention_heads=2,decoder_attention_heads=2,
                       encoder_ffn_dim=32,decoder_ffn_dim=32,max_position_embeddings=32,
                       pad_token_id=1,bos_token_id=0,eos_token_id=2,decoder_start_token_id=2)
    torch.manual_seed(42)
    initial=MBartForConditionalGeneration(config)
    vocab={'<s>':0,'<pad>':1,'</s>':2,'<unk>':3,'tr_TR':4,'en_XX':5,'merhaba':6,'dunya':7,'hello':8,'world':9}
    vocab.update({f'word{i}':i for i in range(10,32)})
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
    tokenizer=MBartTinyTokenizer(tokenizer_object=backend,bos_token='<s>',pad_token='<pad>',
                                  eos_token='</s>',unk_token='<unk>',model_input_names=['input_ids','attention_mask'])
    results=[]
    with tempfile.TemporaryDirectory() as temporary:
        temporary=Path(temporary);data=temporary/'turkish';data.mkdir()
        frame=pd.DataFrame({'source':['merhaba dunya','dunya merhaba'],'target':['hello world','world hello']})
        for split in ['train','validation','test']:frame.to_csv(data/(split+'.csv'),index=False)
        for method in ['baseline','clrr-enc','jepa','jepa-clrr-enc']:
            argv=['train','--model','mbart-large-50','--model-name','tiny-offline','--method',method,
                  '--jepa-weight','0' if method in ['baseline','clrr-enc'] else '.1',
                  '--data-dir',str(data),'--output-dir',str(temporary/'results'),'--backup-dir',str(temporary/'backups'),
                  '--run-name',method,'--num-train-epochs','1','--per-device-train-batch-size','2',
                  '--per-device-eval-batch-size','2','--gradient-accumulation-steps','1',
                  '--max-source-length','8','--max-target-length','8','--num-beams','1',
                  '--generation-precision','fp32','--no-auto-resume','--src-lang','tr_TR','--tgt-lang','en_XX',
                  '--dataloader-num-workers','0','--logging-steps','1','--no-fp16',
                  '--bf16' if cuda else '--no-bf16','--gradient-checkpointing' if cuda else '--no-gradient-checkpointing']
            with patch.object(sys,'argv',argv),patch.object(train.AutoTokenizer,'from_pretrained',return_value=tokenizer),\
                 patch('amis_rewire.modeling.AutoModelForSeq2SeqLM.from_pretrained',side_effect=lambda *a,**kw:copy.deepcopy(initial)):
                train.main()
            folder=temporary/'results'/method
            metadata=json.loads((folder/'metrics.json').read_text())
            assert metadata['configuration']['generation_precision']=='fp32'
            assert metadata['trainable_parameters']==metadata['parameters']
            restored,info=AutoModelForSeq2SeqLM.from_pretrained(folder/'best_model',output_loading_info=True)
            assert not any(info.get(key) for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
            assert restored.generation_config.forced_bos_token_id==5
            rows=pd.read_csv(folder/'test_predictions.csv',keep_default_na=False)
            assert rows[['source','target']].equals(frame)
            results.append({'method':method,'passed':True,'cuda_bf16_training':cuda,'generation_precision':'fp32'})
    out=ROOT/'outputs_rebuttal/followup_20261003';out.mkdir(parents=True,exist_ok=True)
    (out/('precision_training_smoke_gpu.json' if cuda else 'precision_training_smoke_cpu.json')).write_text(json.dumps(results,indent=2))
    print('ALL_FOUR_TRAINING_BRANCHES_PASS',flush=True)


if __name__=='__main__': main()
