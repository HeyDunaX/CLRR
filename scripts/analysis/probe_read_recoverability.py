"""Frozen W42 validation oracle diagnostic. No training or test access.

Optimize ONLY intervention tensors for a fixed gold-versus-native-rival margin.
These label-informed upper-bound probes are not deployable translation methods.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def simplex(x):
    """Euclidean projection, independently for each attention head."""
    import torch
    ordered = x.sort(dim=-1, descending=True).values
    sums = ordered.cumsum(-1) - 1
    index = torch.arange(1, x.shape[-1] + 1, device=x.device)
    support = ordered - sums / index > 0
    rho = support.sum(-1, keepdim=True).clamp_min(1)
    theta = sums.gather(-1, rho - 1) / rho
    return (x - theta).clamp_min(0)


def search(q, k, v, native_context, native_attention, scorer, arm, budget,
           seed, iterations=24, restarts=2):
    """Local projected margin ascent; retain native if no improvement exists.

    Query feasibility is enforced by backtracking, not claimed convex/global.
    Simplex and direct-context sets have exact radial context-budget projection.
    """
    import torch
    scale = native_context.norm().clamp_min(1e-8)
    radius = budget * scale
    initial = {'query': q, 'simplex': native_attention,
               'context': native_context}[arm].detach()
    def context(x):
        if arm == 'query':
            return ((x @ k.transpose(-1, -2) / q.shape[-1] ** .5).softmax(-1) @ v).flatten()
        if arm == 'simplex':
            return (x @ v).flatten()
        return x
    def project(x):
        if arm == 'simplex':
            x = simplex(x)
        if arm == 'query':
            # Bounded local query search. Halving guarantees convergence to native.
            for _ in range(32):
                if (context(x) - native_context).norm() <= radius * (1 + 1e-5):
                    return x
                x = (x + initial) / 2
            return initial.clone()
        delta = context(x) - native_context
        fraction = (radius / delta.norm().clamp_min(1e-8)).clamp(max=1)
        return initial + fraction * (x - initial)
    with torch.no_grad():
        best_score = float(scorer(native_context))
    best, history = native_context.clone(), []
    generator = torch.Generator(device=q.device).manual_seed(seed)
    for restart in range(restarts):
        x = initial.clone()
        if restart:
            noise = torch.randn(x.shape, device=x.device, generator=generator)
            x = project(x + noise / noise.norm().clamp_min(1e-8) * initial.norm().clamp_min(1e-3) * .03)
        for _ in range(iterations):
            x = x.detach().requires_grad_(True)
            value = scorer(context(x))
            gradient, = torch.autograd.grad(value, x)
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite intervention gradient')
            step = (radius if arm == 'context' else initial.norm().clamp_min(1e-3) * .2)
            direction = gradient / gradient.norm().clamp_min(1e-12)
            with torch.no_grad():
                candidate, accepted = x.detach(), False
                for _ in range(7):
                    proposal = project(x.detach() + step * direction)
                    trial = float(scorer(context(proposal)))
                    if trial > float(value) + 1e-7:
                        candidate, accepted = proposal, True
                        break
                    step = step / 2
                x = candidate
                score = float(scorer(context(x)))
                if score > best_score:
                    best_score, best = score, context(x).detach().clone()
                if not accepted:
                    break
        history.append(best_score)
    displacement = float((best-native_context).norm() / scale)
    assert displacement <= budget * (1 + 2e-4) + 1e-7
    return best.detach(), dict(relative_displacement=displacement,
                              best_margin=best_score, restart_best=history)


def main(args):
    import pandas as pd
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from experiment_runner.integrity import sha256, restore_archive
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda')
    out = (ROOT / args.output).resolve()
    assert out.is_relative_to(ROOT / 'outputs/experiments')
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'COMPLETE.json').exists()
    def write(name, value):
        target = out / name
        temp = out / (name + '.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(target)
    config_path = ROOT / 'configs/experiments/amis_drm_validation_fix_rtx.json'
    config = json.loads(config_path.read_text())
    data = ROOT / 'data_processed/amis_mandarin/validation.csv'
    assert sha256(data) == config['data']['validation']['sha256']
    frame = pd.read_csv(data, keep_default_na=False)
    rows = torch.randperm(len(frame), generator=torch.Generator().manual_seed(1729)).tolist()[:args.rows]
    pinned = config['reused_w42']
    plan = dict(checkpoint=pinned, script_sha256=sha256(__file__),
        config_sha256=sha256(config_path), validation_sha256=sha256(data), rows=rows,
        position='middle non-special gold target token per fixed sentence, independent of correctness',
        budgets=[.03, .10, .30], restarts=2, iterations=24,
        objective='target logit minus fixed highest native non-target rival; gold and rank-adjacent alternative target',
        intervention='last decoder cross attention only; native values/keys and all model weights fixed',
        precision='FP32', backend='SDPA', tf32=False, dropout=False,
        model_optimizer_steps=0, test_used=False, exploratory=True,
        limitations='Local oracle search, not global optimum, semantic sufficiency, or deployable method; alternative target is approximately rank matched, rank difference recorded')
    write('PLAN.json', plan)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    transient = out / 'transient_W'
    write('status.json', dict(stage='RESTORE'))
    archive = Path(hf_hub_download(config['repo'], pinned['prefix']+'/best.zip',
        revision=pinned['revision'], token=False, local_dir=transient/'download'))
    assert sha256(archive) == pinned['archive_sha256']
    restore_archive(archive, transient/'export')
    tokenizer = AutoTokenizer.from_pretrained(transient/'export', use_fast=True)
    tokenizer.src_lang, tokenizer.tgt_lang = 'tl_XX', 'zh_CN'
    model, info = AutoModelForSeq2SeqLM.from_pretrained(transient/'export',
        attn_implementation='sdpa', output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), info
    model = model.cuda().eval().requires_grad_(False)
    versions = [p._version for p in model.parameters()]
    layer = model.model.decoder.layers[-1]
    attn = layer.encoder_attn
    capture = {}
    def before_norm(module, inputs):
        capture['residual'] = inputs[0].detach()
    def before_attention(module, inputs, kwargs):
        capture['query_input'] = kwargs['hidden_states'].detach()
        capture['source'] = kwargs['key_value_states'].detach()
    def before_output(module, inputs):
        capture['context'] = inputs[0].detach()
    handles = [layer.encoder_attn_layer_norm.register_forward_pre_hook(before_norm),
               attn.register_forward_pre_hook(before_attention, with_kwargs=True),
               attn.out_proj.register_forward_pre_hook(before_output)]
    results, gates = [], []
    special = set(tokenizer.all_special_ids)
    for number, row in enumerate(rows):
        encoded = tokenizer(frame.iloc[row].source, text_target=frame.iloc[row].target,
                            truncation=True, max_length=256, return_tensors='pt')
        batch = {k:v.cuda() for k,v in encoded.items()}
        with torch.no_grad():
            native = model(**batch, use_cache=False).logits[0]
        labels = batch['labels'][0]
        positions = [i for i,t in enumerate(labels.tolist()) if t not in special]
        if not positions:
            gates.append(dict(row=row, skipped='no nonspecial target'))
            continue
        pos = positions[len(positions)//2]
        H, D = attn.num_heads, attn.head_dim
        with torch.no_grad():
            q = attn.q_proj(capture['query_input'])[0,pos].reshape(H,1,D)
            k = attn.k_proj(capture['source'])[0].reshape(-1,H,D).transpose(0,1)
            v = attn.v_proj(capture['source'])[0].reshape(-1,H,D).transpose(0,1)
            weights = (q @ k.transpose(-1,-2) / D**.5).softmax(-1)
            context = (weights @ v).flatten()
            captured = capture['context'][0,pos]
            torch.testing.assert_close(context, captured, atol=3e-5, rtol=3e-5)
            residual = capture['residual'][0,pos]
        def hidden(c):
            x = residual + attn.out_proj(c)
            x = x + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(x))))
            return model.model.decoder.layer_norm(x)
        def logits(c):
            return model.lm_head(hidden(c)) + model.final_logits_bias[0]
        with torch.no_grad():
            replay = logits(context)
            torch.testing.assert_close(replay, native[pos], atol=4e-5, rtol=4e-5)
            order = replay.argsort(descending=True)
            gold = int(labels[pos])
            gold_rank = int((order==gold).nonzero()[0])+1
            # Adjacent eligible token in native ranking, no semantic selection.
            alternative = next(int(t) for t in order[max(0,gold_rank-2):].tolist()
                               if t not in special and t != gold)
        gates.append(dict(row=row, context_error=float((context-captured).abs().max()),
                          logit_error=float((replay-native[pos]).abs().max())))
        for label_kind, target in [('gold',gold),('alternative',alternative)]:
            rival = next(int(t) for t in order.tolist() if t != target)
            rank = int((order==target).nonzero()[0])+1
            def scorer(c):
                h = hidden(c)
                return (model.lm_head.weight[target]-model.lm_head.weight[rival]) @ h + model.final_logits_bias[0,target]-model.final_logits_bias[0,rival]
            def metrics(c):
                with torch.no_grad():
                    z = logits(c)
                    return dict(margin=float(z[target]-z[rival]), nll=float(-z.log_softmax(-1)[target]),
                                top1=int(z.argmax()), target_top1=bool(int(z.argmax())==target),
                                gold_top1=bool(int(z.argmax())==gold))
            baseline = metrics(context)
            for arm in ['query','simplex','context']:
                for budget in [.03,.10,.30]:
                    best, stats = search(q,k,v,context,weights,scorer,arm,budget,
                                         seed=1729+row*101)
                    result = dict(row=row,position=pos,label_kind=label_kind,target=target,
                        gold=gold,rival=rival,native_rank=rank,gold_rank=gold_rank,
                        arm=arm,budget=budget,native=baseline,intervened=metrics(best),**stats)
                    results.append(result)
        assert versions == [p._version for p in model.parameters()]
        assert all(p.grad is None for p in model.parameters())
        write('measurements.json', results)
        write('GATES.json', gates)
        write('status.json',dict(stage='MEASURE',completed_rows=number+1,total_rows=len(rows)))
    for handle in handles:
        handle.remove()
    write('COMPLETE.json',dict(records=len(results),sentences=len(gates),model_optimizer_steps=0,
                              parameter_versions_unchanged=True,parameter_grads_absent=True,
                              peak_vram_mib=torch.cuda.max_memory_allocated()/2**20))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    assert transient.resolve().is_relative_to(out)
    shutil.rmtree(transient)
    write('CLEANUP.json',dict(own_restore_removed=True,verified_public_archive_sha256=pinned['archive_sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/experiments/w_read_recoverability_rtx')
    parser.add_argument('--rows',type=int,default=32)
    main(parser.parse_args())
