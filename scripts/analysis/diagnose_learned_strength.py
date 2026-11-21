"""Read frozen learned CLRR scalars and measure route dependence; no updates."""
import argparse
from contextlib import contextmanager
import gc
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from experiment_runner import engine
from experiment_runner.integrity import restore_archive, sha256
from methods.common.helpers import _find_stack_layers
from methods.learned_strength import METHODS


@contextmanager
def routing_off(route):
    """Disable only the registered CLRR hooks; preserve weights and other hooks."""
    layers = _find_stack_layers(route, 'encoder')
    replaced = []
    try:
        for layer, handle in zip(layers, route._handles[:len(layers)]):
            original = layer._forward_hooks[handle.id]
            replaced.append((layer, handle.id, original))
            layer._forward_hooks[handle.id] = lambda m, args, output: output
        yield
    finally:
        for layer, ident, original in replaced:
            layer._forward_hooks[ident] = original
        route._cache = {'encoder': {}, 'decoder': {}}


def measure_route(route, ids, mask):
    before, records, handles = {}, [], []
    layers = _find_stack_layers(route, 'encoder')
    def pre(index):
        def hook(module, args, output):
            before[index] = output[0] if isinstance(output, (tuple, list)) else output
        return hook
    def post(index):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if index < route.distance:
                return
            valid = mask.unsqueeze(-1)
            u = before[index]
            injection = hidden-u
            recipient = (u.square()*valid).sum((1,2)).sqrt().clamp_min(1e-12)
            ratio = (injection.square()*valid).sum((1,2)).sqrt()/recipient
            scalar = route.route_strengths[index-route.distance]
            donor = route._cache['encoder'][index-route.distance]
            formula = u + scalar*donor
            torch.testing.assert_close(hidden, formula, rtol=0, atol=0)
            records.append(dict(layer=index+1,scalar=float(scalar),
                row_injection_to_recipient_frobenius=ratio.cpu().tolist()))
        return hook
    try:
        for i, layer in enumerate(layers):
            handles += [layer.register_forward_hook(pre(i),prepend=True),
                        layer.register_forward_hook(post(i))]
        route(input_ids=ids, attention_mask=mask,
              decoder_input_ids=torch.full((len(ids),1),route.config.decoder_start_token_id,
                                           device=ids.device,dtype=torch.long))
        return records
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minimum-free-gpu-mib',type=int,default=20000)
    args=parser.parse_args()
    assert ROOT==Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    out=args.output.resolve();cohort=args.cohort.resolve()
    assert out.is_relative_to(ROOT/'outputs/experiments')
    assert cohort.is_relative_to(ROOT/'outputs/experiments')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine.configure(cohort)
    plan=engine.plan();assert plan['dataset']=='amis_mandarin'
    assert plan['protocol']['hardware_profile']=='rtx6000-fp32'
    assert torch.cuda.get_device_name(0)=='Quadro RTX 6000'
    manifest=json.loads((cohort/'bundle_manifest.json').read_text())
    dependencies=['src/methods/learned_strength/v1/model.py',
        'src/methods/original/clrr.py','src/methods/original/lsr.py',
        'src/methods/common/helpers.py','src/methods/registry.py',
        'src/experiment_runner/engine.py']
    for name in dependencies:
        assert sha256(ROOT/name)==manifest[name],('Frozen code changed',name)
    data=ROOT/'data_processed/amis_mandarin/validation.csv'
    assert sha256(data)==plan['data']['validation']['sha256']
    rows=pd.read_csv(data,dtype=str,keep_default_na=False)
    indices=np.random.default_rng(42).permutation(len(rows))[:64].tolist()
    design=dict(rows=indices,gradient_batches=[indices[i:i+4] for i in range(0,16,4)],
        worker_sha256=sha256(Path(__file__)),cohort=cohort.name,
        optimizer_steps=0,test_consulted=False,precision='FP32/TF32off',
        scope='Post-hoc frozen-checkpoint validation probes; not retrained ablations or training-time Adam history')
    if (out/'DESIGN.json').exists():
        assert json.loads((out/'DESIGN.json').read_text())==design
    else:
        engine.write(out/'DESIGN.json',design)
    for spec in plan['runs']:
        folder=out/spec['id']
        if (folder/'RESULT.json').exists():
            continue
        folder.mkdir(parents=True,exist_ok=True)
        assert spec['method'] in METHODS
        free=torch.cuda.mem_get_info()[0]//2**20
        print('GPU_BEFORE',spec['id'],free,flush=True)
        assert free>=args.minimum_free_gpu_mib,'Do not modify other users processes'
        marker=json.loads((cohort/'runs'/spec['id']/'VAL_COMPLETE.json').read_text())
        assert marker['backup']['verified'] and marker['spec']==spec
        revision=marker['backup']['revision']
        archive=Path(hf_hub_download(plan['repo'],spec['remote_prefix']+'/best.zip',
            revision=revision,token=False,local_dir=folder/'download'))
        assert sha256(archive)==marker['archive_sha256']
        restore_archive(archive,folder/'model')
        model,tok,metadata=engine.load_model(spec,folder/'model')
        model.gradient_checkpointing_disable();model.cuda().eval()
        route=model.base_model if METHODS[spec['method']] else model
        scalars=route.route_strengths.detach().cpu().tolist()
        assert len(scalars)==10 and all(np.isfinite(scalars))
        versions={n:p._version for n,p in model.named_parameters()}
        scalar_snapshot=route.route_strengths.detach().clone()
        # Only construct optimizer groups; no datasets/evaluation/training here.
        trainer=engine.ControlTrainer(model=model,args=engine.arguments(folder/'optimizer_inspection',spec,gate=True),
            processing_class=tok,generation_dtype='fp32')
        trainer.create_optimizer()
        groups=[]
        for i,g in enumerate(trainer.optimizer.param_groups):
            if any(p is route.route_strengths for p in g['params']):
                groups.append(dict(group=i,learning_rate=g['lr'],weight_decay=g['weight_decay'],
                    occurrences=sum(p is route.route_strengths for p in g['params'])))
        assert len(groups)==1 and groups[0]['occurrences']==1
        assert groups[0]['learning_rate']==spec['learning_rate']
        del trainer,g
        model.eval()
        def tensors(selected):
            part=rows.iloc[selected]
            opts=dict(padding=True,pad_to_multiple_of=8,truncation=True,max_length=256,return_tensors='pt')
            x=tok(part.source.tolist(),**opts).to('cuda')
            y=tok(text_target=part.target.tolist(),**opts).to('cuda')
            labels=y.input_ids.masked_fill(y.attention_mask.eq(0),-100)
            return x,labels
        edges={i:[] for i in range(3,13)};ce=[]
        for start in range(0,len(indices),4):
            selected=indices[start:start+4];x,labels=tensors(selected)
            with torch.no_grad():
                for record in measure_route(route,x.input_ids,x.attention_mask):
                    edges[record['layer']].extend(record['row_injection_to_recipient_frobenius'])
                on=route(**x,labels=labels)
                expected=on.logits.detach().clone()
                nll_on=torch.nn.functional.cross_entropy(on.logits.transpose(1,2),labels,
                    ignore_index=-100,reduction='none')
                with routing_off(route):
                    off=route(**x,labels=labels)
                    nll_off=torch.nn.functional.cross_entropy(off.logits.transpose(1,2),labels,
                        ignore_index=-100,reduction='none')
                restored=route(**x,labels=labels)
                torch.testing.assert_close(restored.logits,expected,rtol=0,atol=0)
                counts=labels.ne(-100).sum(-1)
                for i,row in enumerate(selected):
                    ce.append(dict(row=row,tokens=int(counts[i]),
                        ce_on=float(nll_on[i].sum()/counts[i]),ce_off=float(nll_off[i].sum()/counts[i])))
                del on,off,restored,expected,nll_on,nll_off
            del x,labels
        gradients=[]
        for selected in design['gradient_batches']:
            model.zero_grad(set_to_none=True)
            x,labels=tensors(selected)
            output=model(**x,labels=labels)
            loss=output.loss;loss.backward()
            norms=[torch.linalg.vector_norm(p.grad.detach().float()) for p in model.parameters() if p.grad is not None]
            norm=float(torch.linalg.vector_norm(torch.stack(norms)))
            assert np.isfinite(norm) and norm>0
            scalar_grad=route.route_strengths.grad.detach().cpu()
            assert torch.isfinite(scalar_grad).all()
            clip=min(1.,plan['protocol']['max_grad_norm']/(norm+1e-6))
            gradients.append(dict(rows=selected,eval_objective=float(loss.detach()),
                total_gradient_norm=norm,scalar_gradient=scalar_grad.tolist(),
                scalar_gradient_norm=float(scalar_grad.norm()),
                hypothetical_global_clip_factor=clip,
                note='Eval-mode single microbatch, no accumulation/optimizer; not historical train clipping'))
            del x,labels,output,loss,norms,scalar_grad
            model.zero_grad(set_to_none=True)
        assert all(p._version==versions[n] for n,p in model.named_parameters())
        assert torch.equal(route.route_strengths,scalar_snapshot)
        ratios=[dict(layer=i,scalar=scalars[i-3],mean=float(np.mean(v)),
            median=float(np.median(v)),p95=float(np.quantile(v,.95)),max=float(max(v))) for i,v in edges.items()]
        delta=np.array([r['ce_off']-r['ce_on'] for r in ce])
        state=json.loads((cohort/'runs'/spec['id']/'final_trainer_state.json').read_text())
        logged=[v['grad_norm'] for v in state['log_history'] if 'grad_norm' in v]
        result=dict(status='PASS',id=spec['id'],scalars=scalars,optimizer_scalar_groups=groups,
            parameter_count=route.extra_parameter_count(),injection_to_recipient=ratios,
            row_ce=ce,ce_off_minus_on_mean=float(delta.mean()),
            ce_on_mean=float(np.mean([r['ce_on'] for r in ce])),
            ce_off_mean=float(np.mean([r['ce_off'] for r in ce])),
            gradients=gradients,training_logged_preclip_norms=logged,
            historical_optimizer_states_available=False,
            unchanged_weights=True,route_hooks_restored=True,optimizer_steps=0,
            archive_sha256=marker['archive_sha256'],revision=revision,
            restored_backend=model.config._attn_implementation,design=design,
            verified_numerical_sources={n:manifest[n] for n in dependencies},
            caveat='Dependency after training is not superiority over separately trained baseline; scalar gradients/clipping probes are eval-mode snapshots, not optimizer trajectory')
        del model,route;gc.collect();torch.cuda.empty_cache()
        for temporary in [folder/'model',folder/'download']:
            assert temporary.resolve().is_relative_to(out)
            shutil.rmtree(temporary)
        result['temporary_weights_removed']=True
        engine.write(folder/'RESULT.json',result)
        print('MEASUREMENT_COMPLETE',spec['id'],json.dumps(dict(scalars=scalars,
            ce_off_minus_on_mean=result['ce_off_minus_on_mean'])),flush=True)
    engine.write(out/'COMPLETE.json',dict(status='COMPLETE',runs=[s['id'] for s in plan['runs']],
        optimizer_steps=0,test_consulted=False,temporary_weights_removed=True,
        worker_sha256=sha256(Path(__file__))))


if __name__=='__main__':
    main()
