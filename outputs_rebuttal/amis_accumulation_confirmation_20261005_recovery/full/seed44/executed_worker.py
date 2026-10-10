"""Seed43/44 LSR/Full confirmation with corrected accumulation and frozen old recipe."""
import argparse,gc,hashlib,json,os,sys
from pathlib import Path
import torch
import transformers,accelerate
from transformers import TrainerCallback,AutoTokenizer,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments
from huggingface_hub import HfApi
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire import train
from amis_rewire.evaluation import PrecisionSeq2SeqTrainer
from amis_rewire.metrics import safe_decode_inputs,generation_metrics,metric_protocol
PREFIX='amis_accumulation_confirmation_20261005_recovery'
REPO='FiveC/amis-accumulation-confirmation-20261005'

def weight_digest(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str((t.dtype,tuple(t.shape))).encode())
        h.update(memoryview(t.contiguous().numpy()).cast('B'))
    del state;gc.collect()
    return h.hexdigest()

def evaluate_frozen_test(folder,family):
    """One post-selection test evaluation, separate from validation-only training."""
    import pandas as pd
    gc.collect();torch.cuda.empty_cache()
    tokenizer=AutoTokenizer.from_pretrained(folder/'best_model',use_fast=True)
    model=train.load_model(str(folder/'best_model'),'jepa' if family=='lsr' else 'jepa-clrr-enc',
        distance=2,strength=0. if family=='lsr' else .1,stack='encoder',jepa_weight=.1)
    train.configure_mbart(tokenizer,model,'tl_XX','zh_CN');model.config.use_cache=False
    prep=argparse.Namespace(source_prefix='',max_source_length=256,max_target_length=256)
    data=ROOT/'data_processed/amis_mandarin'
    dataset=train.tokenize_dataset(train.load_split(data,'test'),tokenizer,prep)
    args=Seq2SeqTrainingArguments(output_dir=str(OUT/'test_only'),per_device_eval_batch_size=8,
        predict_with_generate=True,generation_num_beams=4,generation_max_length=256,bf16=True,fp16=False,
        tf32=True,report_to=[],dataloader_num_workers=2,remove_unused_columns=False)
    trainer=PrecisionSeq2SeqTrainer(model=model,args=args,processing_class=tokenizer,generation_dtype='fp32',
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,pad_to_multiple_of=8))
    result=trainer.predict(dataset)
    prediction=result.predictions[0] if isinstance(result.predictions,tuple) else result.predictions
    decoded=[s.strip() for s in tokenizer.batch_decode(safe_decode_inputs(prediction),skip_special_tokens=True)]
    raw=pd.read_csv(data/'test.csv',dtype=str,keep_default_na=False);assert len(decoded)==len(raw)==575
    frame=raw.copy();frame['prediction']=decoded;frame.to_csv(folder/'test_predictions.csv',index=False)
    scores=generation_metrics(decoded,raw['target'].tolist(),tokenize='zh')
    metadata={'selected_checkpoint_only':True,'selection_split':'validation','test_used_for_selection':False,
        'beams':4,'batch_size':8,'generation_precision':'fp32','tf32_generation':False,
        'raw_test_sha256':hashlib.sha256((data/'test.csv').read_bytes()).hexdigest(),
        'protocol':metric_protocol('zh'),**{f'test_{k}':v for k,v in scores.items()}}
    (folder/'test_metrics.json').write_text(json.dumps(metadata,indent=2))
    for path in (folder/'test_metrics.json',folder/'test_predictions.csv'):
        assert train.safe_hf_upload(api,path,REPO,f'{PREFIX}/{RUN}/{path.name}')
    del trainer,model;gc.collect();torch.cuda.empty_cache()
    return metadata

def upload_metadata_folder(folder,remote,required=False):
    import time
    for attempt in range(3 if required else 1):
        try:
            api.upload_folder(repo_id=REPO,folder_path=str(folder),path_in_repo=remote,repo_type='model')
            return True
        except Exception as error:
            # Never include request headers or token in error logs.
            print('METADATA_UPLOAD_DEFERRED',type(error).__name__,remote,flush=True)
            if required and attempt<2:time.sleep(30)
    if required:raise RuntimeError('Required final metadata upload failed')
    return False

class SelectionBackup(TrainerCallback):
    def on_save(self,args,state,control,**kwargs):
        import shutil
        checkpoint=Path(args.output_dir)/f'checkpoint-{state.global_step}'
        staged=OUT/'selection_staging'/f'checkpoint-{state.global_step}';staged.mkdir(parents=True,exist_ok=True)
        for name in ('trainer_state.json','training_receipt.json'):
            assert (checkpoint/name).exists();shutil.copy2(checkpoint/name,staged/name)
        uploaded=upload_metadata_folder(staged,f'{PREFIX}/{RUN}/selection/checkpoint-{state.global_step}')
        (staged/'upload_status.json').write_text(json.dumps({'uploaded':uploaded,'step':state.global_step}))
        state.save_to_json(str(OUT/'latest_trainer_state.json'))
    def on_train_end(self,args,state,control,**kwargs):
        import shutil
        state.save_to_json(str(OUT/'final_trainer_state.json'))
        staged=OUT/'selected_provenance';staged.mkdir(exist_ok=True)
        for name in ('trainer_state.json','training_receipt.json'):
            shutil.copy2(Path(state.best_model_checkpoint)/name,staged/name)
        upload_metadata_folder(staged,f'{PREFIX}/{RUN}/selected_provenance',required=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('lsr','full'),required=True)
    p.add_argument('--seed',type=int,choices=(44,),required=True)
    parsed=p.parse_args();family=parsed.family;seed=parsed.seed
    assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.57.6' and accelerate.__version__=='1.15.0'
    assert torch.cuda.is_bf16_supported()
    gate=ROOT/f'outputs_rebuttal/{PREFIX}/gpu_gate'
    matrix=json.loads((gate/'optimization_matrix.json').read_text())
    assert len(matrix)==16 and all(r['passed'] and r['cuda_bf16'] for r in matrix)
    lifecycle=json.loads((gate/'rerun_lifecycle_smoke.json').read_text())
    assert len(lifecycle)==4 and all(r['passed'] and r['best_weights_exact'] for r in lifecycle)
    manifest=json.loads((ROOT/'recovery_bundle_manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
    OUT=ROOT/f'outputs_rebuttal/{PREFIX}/{family}/seed{seed}';OUT.mkdir(parents=True,exist_ok=True)
    RUN=f'mbart-amis-{ "lsr_l010" if family=="lsr" else "full_a010_l010" }-seed{seed}-accumfixed-recovery'
    api=HfApi(token=os.environ['HF_TOKEN'])
    api.create_repo(REPO,repo_type='model',private=True,exist_ok=True)
    assert not any(n.startswith(f'{PREFIX}/{RUN}/') for n in api.list_repo_files(REPO)),'Run namespace already exists'
    original=train.PrecisionSeq2SeqTrainer
    def factory(*a,**kw):
        kw['callbacks'].append(SelectionBackup())
        trainer=original(*a,**kw)
        assert trainer.model_accepts_loss_kwargs is False
        assert trainer.args.gradient_accumulation_steps==4 and trainer.args.bf16 and trainer.args.gradient_checkpointing
        assert trainer.args.metric_for_best_model=='eval_chrf++' and trainer.args.generation_num_beams==1
        (OUT/'trainer_gate.json').write_text(json.dumps({'accepts_loss_kwargs':False,'accumulation4':True,'bf16':True,'checkpointing':True,'source_bundle_sha256':manifest},indent=2))
        return trainer
    train.PrecisionSeq2SeqTrainer=factory
    sys.argv=['pilot','--model','mbart-large-50','--model-name','facebook/mbart-large-50-many-to-many-mmt',
        '--model-revision','e30b6cb8eb0d43a0b73cab73c7676b9863223a30',
        '--method','jepa' if family=='lsr' else 'jepa-clrr-enc','--jepa-weight','.1',
        '--rewire-strength','0' if family=='lsr' else '.1','--rewire-distance','2','--rewire-stack','encoder',
        '--routing-warmup-ratio','0','--data-dir',str(ROOT/'data_processed/amis_mandarin'),
        '--output-dir',str(OUT/'runs'),'--backup-dir',str(OUT/'backups'),'--run-name',RUN,'--seed',str(seed),
        '--learning-rate','5e-5','--weight-decay','0','--warmup-ratio','.06',
        '--num-train-epochs','20','--early-stopping-patience','4',
        '--per-device-train-batch-size','32','--gradient-accumulation-steps','4',
        '--per-device-eval-batch-size','8','--max-source-length','256','--max-target-length','256',
        '--eval-beams','1','--num-beams','4','--generation-precision','fp32',
        '--src-lang','tl_XX','--tgt-lang','zh_CN','--bleu-tokenizer','zh',
        '--no-fp16','--bf16','--gradient-checkpointing','--no-auto-resume',
        '--validation-only','--logging-steps','10','--dataloader-num-workers','2','--save-total-limit','2',
        '--hf-backup-repo',REPO,'--hf-backup-prefix',PREFIX]
    (OUT/'command.json').write_text(json.dumps(sys.argv,indent=2))
    train.main()
    folder=OUT/'runs'/RUN;metrics=json.loads((folder/'metrics.json').read_text())
    state=json.loads((OUT/'final_trainer_state.json').read_text())
    assert state['best_model_checkpoint']==metrics['best_model_checkpoint']
    selected_digest=weight_digest(Path(state['best_model_checkpoint'])/'pytorch_model.bin')
    best_digest=weight_digest(folder/'best_model/pytorch_model.bin')
    assert selected_digest==best_digest,'Best export does not match selected checkpoint tensors'
    test_metrics=evaluate_frozen_test(folder,family)
    receipt={'status':'COMPLETE','family':family,'seed':seed,'run':RUN,'selected_checkpoint':state['best_model_checkpoint'],
        'best_metric':state['best_metric'],'completed_epochs':metrics['completed_epochs'],
        'hf_repo':REPO,'hf_prefix':PREFIX,'selected_and_exported_weight_digest':best_digest,
        'post_selection_test_bleu':test_metrics['test_bleu'],'post_selection_test_chrfpp':test_metrics['test_chrf++'],
        'old_scores_untouched':True,'backup_policy':'Final best archive checksum verified; full epoch selection metadata on HF; optimizer/RNG local only'}
    import shutil
    staged=OUT/'final_provenance';staged.mkdir(exist_ok=True)
    for path in list(OUT.glob('*.json'))+[folder/'training_receipt.json',ROOT/'recovery_bundle_manifest.json',ROOT/'scripts/revalidation/run_accumulation_recovery.py']:
        shutil.copy2(path,staged/path.name)
    (OUT/'COMPLETE.json').write_text(json.dumps(receipt,indent=2));shutil.copy2(OUT/'COMPLETE.json',staged/'COMPLETE.json')
    upload_metadata_folder(staged,f'{PREFIX}/{RUN}/provenance',required=True)
    print('ACCUMULATION_PILOT_COMPLETE',json.dumps(receipt),flush=True)
