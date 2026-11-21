"""Calibrate virtual-order scale on three TRAIN batches at pinned initialization."""
import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from probe_virtual_corrective_order import write, digest


def run(path):
    import torch
    import pandas as pd
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed
    from methods.virtual_corrective_order.v1.objective import paired_ce_kl, order_penalty
    plan = json.loads(Path(path).read_text())
    out = ROOT / plan['output']
    assert plan['split'] == 'train' and plan['optimizer_steps'] == 0
    assert not plan['test_used'] and plan['batches'] == 3
    for name, sha in plan['source_sha256'].items():
        assert digest(ROOT / name) == sha
    assert digest(ROOT / plan['train_path']) == plan['train_sha256']
    assert not (out / 'COMPLETE.json').exists()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.mem_get_info()[0] >= plan['minimum_free_gpu_mib'] * 2**20
    set_seed(42)
    model, info = AutoModelForSeq2SeqLM.from_pretrained(plan['model_id'],
        revision=plan['model_revision'], attn_implementation='sdpa', output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.cuda().train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    tokenizer = AutoTokenizer.from_pretrained(plan['model_id'], revision=plan['model_revision'], use_fast=True)
    tokenizer.src_lang, tokenizer.tgt_lang = 'tl_XX', 'zh_CN'
    frame = pd.read_csv(ROOT / plan['train_path'], keep_default_na=False)
    ids = torch.randperm(len(frame), generator=torch.Generator().manual_seed(plan['data_seed'])).tolist()[:12]
    examples = [tokenizer(frame.iloc[i].source, text_target=frame.iloc[i].target,
        max_length=256, truncation=True) for i in ids]
    collate = DataCollatorForSeq2Seq(tokenizer, model=None, pad_to_multiple_of=8, label_pad_token_id=-100)
    basis = torch.linalg.qr(torch.randn(model.config.d_model, plan['rank'],
        generator=torch.Generator().manual_seed(plan['basis_seed']))).Q.cuda()
    versions = [p._version for p in model.parameters()]
    records = []
    for mb in range(3):
        batch = {k:v.cuda() for k,v in collate(examples[4*mb:4*mb+4]).items()}
        norms, values = {}, {}
        for mode in ('CE', 'ORDER'):
            model.zero_grad(set_to_none=True)
            hs = []
            for seed in (plan['dropout_seed']+2*mb, plan['dropout_seed']+2*mb+1):
                set_seed(seed)
                hs.append(model.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    decoder_input_ids=model.prepare_decoder_input_ids_from_labels(batch['labels']),
                    use_cache=False).last_hidden_state)
            w, b = model.lm_head.weight, model.final_logits_bias.squeeze(0)
            support, probe = ([0,1],[2,3]) if mb%2 == 0 else ([2,3],[0,1])
            if mode == 'CE':
                loss = paired_ce_kl(*hs,batch['labels'],w,b,plan['chunk'],True,False)[0]
            else:
                loss,_,_ = order_penalty(*hs,batch['labels'],support,probe,w,b,basis,
                    plan['eta'],plan['chunk'],True)
                # FP64 head-only replay checks cancellation at this initialization.
                with torch.no_grad():
                    ref,*_ = order_penalty(*(h.double() for h in hs),batch['labels'],support,probe,
                        w.double(),b.double(),basis.double(),plan['eta'],plan['chunk'],False)
                rel = abs(float(loss.detach())-float(ref))/max(abs(float(ref)),1e-30)
                assert ref > 0 and rel <= .05, ('FP32 order signal unreliable', rel)
            loss.backward()
            squares = []
            for p in model.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
                    squares.append(p.grad.detach().double().square().sum())
            norms[mode] = float(torch.stack(squares).sum().sqrt())
            values[mode] = float(loss.detach())
            assert norms[mode] > 0
            del loss, hs
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        record = dict(batch=mb,rows=ids[4*mb:4*mb+4],norms=norms,values=values,
            fp32_relative_error=rel,proposed_lambda=plan['calibration_gradient_ratio']*norms['CE']/norms['ORDER'])
        records.append(record)
        write(out/'CALIBRATION_BATCHES.json',records)
        write(out/'PROGRESS.json',dict(stage='INITIALIZATION_CALIBRATION',batches_done=mb+1,optimizer_steps=0))
    assert versions == [p._version for p in model.parameters()]
    coefficient = sorted(r['proposed_lambda'] for r in records)[1]
    receipt = dict(passed=True,completed=True,optimizer_steps=0,parameters_unchanged=True,
        model_id=plan['model_id'],model_revision=plan['model_revision'],coefficient=coefficient,
        rank=plan['rank'],basis_seed=plan['basis_seed'],eta=plan['eta'],chunk=plan['chunk'],
        calibration_gradient_ratio=plan['calibration_gradient_ratio'],
        train_sha256=plan['train_sha256'],validation_used=False,test_used=False,
        rule='median over three fixed TRAIN batches of 0.1*full-backbone CE norm/order norm; fixed thereafter',
        limitation='Gradient scale heuristic, not Adam update ratio or quality tuning')
    write(out/'COMPLETE.json',receipt)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    try:
        run(args.plan)
    except Exception:
        write(ROOT/json.loads(Path(args.plan).read_text())['output']/'ERROR.json',
            dict(traceback=traceback.format_exc(),optimizer_steps=0))
        raise
