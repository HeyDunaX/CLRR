"""Fresh seed42 FT control with explicit selection provenance and HF backup."""
import hashlib,json,os,sys
from pathlib import Path
from transformers import TrainerCallback
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire import train
OUT=ROOT/'outputs_rebuttal/amis_corrected_baseline42_20261005'
RUN='mbart-amis-baseline-seed42-corrected'
PREFIX='amis_corrected_baseline42_20261005'
REPO='FiveC/amis-rewire-checkpoints'

class SelectionBackup(TrainerCallback):
    def on_save(self,args,state,control,**kwargs):
        checkpoint=Path(args.output_dir)/f'checkpoint-{state.global_step}'
        for name in ('trainer_state.json','training_receipt.json'):
            path=checkpoint/name
            assert path.exists()
            assert train.safe_hf_upload(api,path,REPO,f'{PREFIX}/{RUN}/selection/checkpoint-{state.global_step}/{name}')
        state.save_to_json(str(OUT/'latest_trainer_state.json'))
    def on_train_end(self,args,state,control,**kwargs):
        state.save_to_json(str(OUT/'final_trainer_state.json'))
        selected=Path(state.best_model_checkpoint)
        for name in ('trainer_state.json','training_receipt.json'):
            path=selected/name
            assert train.safe_hf_upload(api,path,REPO,f'{PREFIX}/{RUN}/selected_provenance/{name}')

if __name__=='__main__':
    assert (ROOT/'outputs_rebuttal/amis_fullsize_gate_20261005/FULLSIZE_PASS.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    api=HfApi(token=os.environ['HF_TOKEN'])
    assert not any(p.startswith(PREFIX+'/') for p in api.list_repo_files(REPO)), 'HF namespace already exists'
    original=train.PrecisionSeq2SeqTrainer
    def factory(*a,**kw):
        kw['callbacks'].append(SelectionBackup())
        return original(*a,**kw)
    train.PrecisionSeq2SeqTrainer=factory
    sys.argv=['baseline42','--model','mbart-large-50','--model-name','facebook/mbart-large-50-many-to-many-mmt',
        '--model-revision','e30b6cb8eb0d43a0b73cab73c7676b9863223a30','--method','baseline',
        '--jepa-weight','0','--rewire-strength','.1','--routing-warmup-ratio','0',
        '--data-dir',str(ROOT/'data_processed/amis_mandarin'),'--output-dir',str(OUT/'runs'),
        '--backup-dir',str(OUT/'backups'),'--run-name',RUN,'--seed','42',
        '--learning-rate','5e-5','--weight-decay','0','--warmup-ratio','.06',
        '--num-train-epochs','20','--early-stopping-patience','4',
        '--per-device-train-batch-size','32','--gradient-accumulation-steps','4',
        '--per-device-eval-batch-size','8','--max-source-length','256','--max-target-length','256',
        '--eval-beams','1','--num-beams','4','--generation-precision','fp32',
        '--src-lang','tl_XX','--tgt-lang','zh_CN','--bleu-tokenizer','zh',
        '--no-fp16','--bf16','--gradient-checkpointing','--no-auto-resume',
        '--logging-steps','10','--dataloader-num-workers','0','--save-total-limit','2',
        '--hf-backup-repo',REPO,'--hf-backup-prefix',PREFIX]
    (OUT/'command.json').write_text(json.dumps(sys.argv,indent=2))
    train.main()
    folder=OUT/'runs'/RUN
    metrics=json.loads((folder/'metrics.json').read_text())
    state=json.loads((OUT/'final_trainer_state.json').read_text())
    assert state['best_model_checkpoint']==metrics['best_model_checkpoint']
    receipt={'status':'COMPLETE','run':RUN,'selected_checkpoint':state['best_model_checkpoint'],
        'best_metric':state['best_metric'],'completed_epochs':metrics['completed_epochs'],
        'hf_repo':REPO,'hf_prefix':PREFIX,'old_scores_untouched':True,
        'backup_policy':'Final best archive checksum verified; epoch selection metadata uploaded. Full optimizer/RNG remains local.'}
    (OUT/'COMPLETE.json').write_text(json.dumps(receipt,indent=2))
    for path in list(OUT.glob('*.json'))+[folder/'training_receipt.json']:
        assert train.safe_hf_upload(api,path,REPO,f'{PREFIX}/{RUN}/provenance/{path.name}')
    print('CORRECTED_BASELINE42_COMPLETE',json.dumps(receipt),flush=True)
