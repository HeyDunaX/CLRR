"""Frozen training-only U2 scale audit; no optimizer or checkpoint writes."""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import transformers
from transformers import AutoTokenizer, MBartForConditionalGeneration
from methods.depth_difference.v1 import DepthDifferencePrototype


def squared_heads(x, valid):
    """Sum squares over batch/valid target queries/features; retain heads."""
    return (x.float().square() * valid[:, None, :, None]).sum((0, 2, 3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.rows < 1 or a.batch_size < 1:
        p.error('rows and batch-size must be positive')
    import fcntl
    if not Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda'):
        raise RuntimeError('Use project-local conda')
    lock = (ROOT / 'tmp/training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    free, _ = torch.cuda.mem_get_info()
    if free < 16000 * 2**20:
        raise RuntimeError('Less than 16000 MiB free; leave other users untouched')
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with a.data.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    indices = random.Random(a.seed).sample(range(len(rows)), a.rows)
    model_id = 'facebook/mbart-large-50-many-to-many-mmt'
    revision = 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'
    base = MBartForConditionalGeneration.from_pretrained(model_id, revision=revision,
        cache_dir=str(a.cache_dir), local_files_only=True,
        attn_implementation='eager', torch_dtype=torch.float32).cuda().eval()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision,
        cache_dir=str(a.cache_dir), local_files_only=True, src_lang='tl_XX', tgt_lang='zh_CN')
    model = DepthDifferencePrototype(base, donor_layer=6).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = model.correction_attention
    selected.gates.requires_grad_(True)
    batches = []
    for offset in range(0, a.rows, a.batch_size):
        ids = indices[offset:offset + a.batch_size]
        enc = tok([rows[i]['source'] for i in ids], text_target=[rows[i]['target'] for i in ids],
            padding=True, truncation=True, max_length=256, return_tensors='pt').to('cuda')
        enc['labels'] = enc['labels'].masked_fill(enc['labels'] == tok.pad_token_id, -100)
        valid_target = enc['labels'].ne(-100)
        valid_source = enc['attention_mask'].bool()
        record = {'row_indices': ids, 'target_positions': int(valid_target.sum()), 'variants': {}}

        @torch.no_grad()
        def observe(module, args, kwargs):
            attn = module.attention
            hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            top, donor = kwargs['key_value_states'].split(attn.embed_dim, -1)
            b, t, _ = hidden.shape
            s = top.shape[1]
            def heads(x):
                return x.reshape(b, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
            q = heads(attn.q_proj(hidden))
            def probs(k):
                logits = q @ k.transpose(-1, -2) * attn.scaling
                return logits.masked_fill(~valid_source[:, None, None, :], float('-inf')).softmax(-1)
            source_mask = valid_source[:, :, None]
            top_sq = (top.square() * source_mask).sum((1, 2))
            donor_sq = (donor.square() * source_mask).sum((1, 2))
            scale = (top_sq / donor_sq.clamp_min(1e-30)).sqrt()
            scaled = donor * scale[:, None, None]
            kt, vt = heads(attn.k_proj(top)), heads(attn.v_proj(top))
            pt = probs(kt)
            context = pt @ vt
            context_sq = squared_heads(context, valid_target)
            record.update(source_scales=scale.tolist(), top_hidden_sq=top_sq.tolist(),
                donor_hidden_sq=donor_sq.tolist(), base_context_sq_heads=context_sq.tolist())
            for key_scaled in (False, True):
                pd = probs(heads(attn.k_proj(scaled if key_scaled else donor)))
                for value_scaled in (False, True):
                    vd = heads(attn.v_proj(scaled if value_scaled else donor))
                    delta = (pd - pt) @ (vd - vt)
                    name = f'K{int(key_scaled)}_V{int(value_scaled)}'
                    value_mask = valid_source[:, None, :, None]
                    record['variants'][name] = {
                        'correction_sq_heads': squared_heads(delta, valid_target).tolist(),
                        'attention_l1_sum_heads': ((pd-pt).abs().sum(-1) * valid_target[:, None, :]).sum((0, 2)).tolist(),
                        'donor_value_sq': float((vd.square()*value_mask).sum()),
                        'top_value_sq': float((vt.square()*value_mask).sum()),
                    }
                    assert torch.isfinite(delta).all()
        handle = selected.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            loss = model(**enc, use_cache=False).loss
            loss.backward()
            gradient = selected.gates.grad.detach()
            assert torch.isfinite(gradient).all()
            record.update(ce=float(loss.detach()), gate_gradient=gradient.tolist())
            model.zero_grad(set_to_none=True)
        finally:
            handle.remove()
        batches.append(record)
        print(f'Batch {len(batches)}: {offset + len(ids)}/{a.rows}', flush=True)
    base_sq = torch.tensor([b['base_context_sq_heads'] for b in batches], dtype=torch.float64).sum(0)
    summary = {}
    for name in batches[0]['variants']:
        values = [b['variants'][name] for b in batches]
        delta_sq = torch.tensor([v['correction_sq_heads'] for v in values], dtype=torch.float64).sum(0)
        summary[name] = {
            'correction_norm': float(delta_sq.sum().sqrt()),
            'base_context_norm': float(base_sq.sum().sqrt()),
            'correction_to_base_norm': float((delta_sq.sum()/base_sq.sum()).sqrt()),
            'correction_to_base_per_head': (delta_sq/base_sq.clamp_min(1e-30)).sqrt().tolist(),
            'attention_l1_mean': sum(sum(v['attention_l1_sum_heads']) for v in values) / (sum(b['target_positions'] for b in batches)*len(base_sq)),
            'donor_to_top_value_norm': (sum(v['donor_value_sq'] for v in values)/sum(v['top_value_sq'] for v in values))**.5,
        }
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (a.data, Path(__file__), ROOT/'src/methods/depth_difference/v1/model.py')}
    report = dict(scope='Frozen pretrained, training-only; analytical scale counterfactuals; zero optimizer steps',
        seed=a.seed, row_indices=indices, rows=a.rows, batch_size=a.batch_size,
        model=model_id, revision=revision, source_lang='tl_XX', target_lang='zh_CN', max_length=256,
        donor_layer=6, decoder_layer=12, dtype='FP32', eval_mode=True, tf32=False,
        torch=torch.__version__, transformers=transformers.__version__, hashes=hashes,
        reduction='Frobenius norms pooled over valid positions; target labels != -100; source attention_mask=1; pre-output-projection, before gate',
        scale='per example ||top_valid_source||_F / ||donor_valid_source||_F; 1 indicates diagnostic rescaling, 0 raw',
        free_gpu_before_mib=free//2**20, summary=summary, batches=batches)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
