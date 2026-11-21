"""Frozen token-transport diagnostics. No optimizer or training checkpoint writes."""
import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from methods.original.clrr import CrossLayerResidualRewire


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def transport(cost, sm, tm, epsilon, iterations):
    """Detached entropic balanced plan; report marginal error explicitly."""
    with torch.no_grad():
        a = sm.float() / sm.sum(1, keepdim=True)
        b = tm.float() / tm.sum(1, keepdim=True)
        kernel = torch.exp(-cost.detach().float() / epsilon)
        kernel = kernel * sm.unsqueeze(2) * tm.unsqueeze(1)
        v = tm.float()
        for _ in range(iterations):
            u = a / torch.bmm(kernel, v.unsqueeze(2)).squeeze(2).clamp_min(1e-30)
            v = b / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(2)).squeeze(2).clamp_min(1e-30)
        p = u.unsqueeze(2) * kernel * v.unsqueeze(1)
        error = torch.maximum((p.sum(2) - a).abs().amax(1), (p.sum(1) - b).abs().amax(1))
    return p.detach(), error


def token_loss(source, target, sm, tm, args):
    cost = 1 - torch.bmm(F.normalize(source.float(), dim=-1), F.normalize(target.detach().float(), dim=-1).transpose(1, 2))
    p, error = transport(cost, sm, tm, args.epsilon, args.iterations)
    uniform = sm.float().unsqueeze(2) * tm.float().unsqueeze(1)
    uniform = uniform / (sm.sum(1) * tm.sum(1)).view(-1, 1, 1)
    return (p * cost).sum((1, 2)), (uniform * cost).sum((1, 2)), error


def correctness(args):
    torch.manual_seed(42)
    s = torch.randn(2, 4, 8, requires_grad=True)
    t = torch.randn(2, 5, 8, requires_grad=True)
    sm = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    tm = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    loss, _, error = token_loss(s, t, sm, tm, args)
    loss.mean().backward()
    assert torch.isfinite(s.grad).all() and s.grad.norm() > 0
    assert t.grad is None and torch.count_nonzero(s.grad[~sm]) == 0
    assert error.max() < 1e-3
    s2 = s.detach().clone(); s2[~sm] = 999
    t2 = t.detach().clone(); t2[~tm] = -999
    assert torch.allclose(loss.detach(), token_loss(s2, t2, sm, tm, args)[0], atol=1e-6)
    return {'finite_source_gradient': True, 'target_gradient_absent': True, 'padding_invariant': True, 'marginal_error': float(error.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', nargs='+', default=['BASE', 'OLD'])
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--shuffles', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    assert args.epsilon > 0 and args.iterations > 0 and args.shuffles > 0
    checks = correctness(args)
    if args.cpu_check:
        print(json.dumps(checks)); return
    workspace = Path('/workspace/ura/thanhdat').resolve()
    assert ROOT.resolve().is_relative_to(workspace)
    assert Path(sys.prefix).resolve().is_relative_to(workspace / 'conda')
    out = Path(args.output).resolve(); assert out.is_relative_to(workspace)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'COMPLETE.json').exists(), 'Already completed'
    plan = json.loads(Path(args.plan).read_text())
    data = ROOT / 'data_processed/amis_mandarin/validation.csv'
    assert sha(data) == plan['validation_sha256']
    with data.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    indices = plan['representation_indices'][:128]
    rng = np.random.default_rng(42)
    permutations = []
    for _ in range(args.shuffles):
        perm = rng.permutation(len(indices))
        while np.any(perm == np.arange(len(indices))):
            perm = rng.permutation(len(indices))
        permutations.append(perm.tolist())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {}
    for model_id in args.models:
        free, total = torch.cuda.mem_get_info()
        print('GPU_BEFORE', model_id, free // 2**20, flush=True)
        assert free >= 12 * 2**30, 'Insufficient free memory; leave other users alone'
        entry = next(e for e in plan['entries'] if e['id'] == model_id)
        archive = Path(hf_hub_download(plan['repo'], entry['archive_path'], revision=plan['revision'], token=False))
        assert archive.resolve().is_relative_to(workspace)
        assert sha(archive) == entry['archive_sha256'] and archive.stat().st_size == entry['archive_bytes']
        extracted = out / model_id / 'extracted'; extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                assert (extracted / name).resolve().is_relative_to(extracted.resolve())
            z.extractall(extracted)
        folders = [p.parent for p in extracted.rglob('config.json') if any((p.parent / n).exists() for n in ['pytorch_model.bin', 'model.safetensors'])]
        assert len(folders) == 1
        base, info = AutoModelForSeq2SeqLM.from_pretrained(folders[0], output_loading_info=True, torch_dtype=torch.float32)
        assert not any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
        base.gradient_checkpointing_disable()
        model = CrossLayerResidualRewire(base, distance=2, strength=0.1, stack='encoder') if model_id == 'OLD' else base
        model.cuda().eval()
        versions = {n: p._version for n, p in model.named_parameters()}
        tok = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', revision=plan['model_revision'], src_lang='tl_XX', tgt_lang='zh_CN')

        def inputs(ids, target=False):
            text = [rows[i]['target' if target else 'source'] for i in ids]
            batch = tok(text_target=text, padding=True, truncation=True, max_length=256, return_tensors='pt') if target else tok(text, padding=True, truncation=True, max_length=256, return_tensors='pt')
            return {k: v.cuda() for k, v in batch.items() if k in ['input_ids', 'attention_mask']}

        def content(batch):
            mask = batch['attention_mask'].bool().clone()
            for token_id in tok.all_special_ids:
                mask &= batch['input_ids'] != token_id
            assert mask.sum(1).min() > 0
            return mask

        def encode(batch):
            context = model._fresh_cache() if model_id == 'OLD' else nullcontext()
            with context:
                return model.get_encoder()(**batch).last_hidden_state

        source, target = [], []
        with torch.no_grad():
            for start in range(0, len(indices), args.batch_size):
                ids = indices[start:start + args.batch_size]
                sb, tb = inputs(ids), inputs(ids, True)
                sh, th = encode(sb), encode(tb)
                source.extend([h[m].detach() for h, m in zip(sh, content(sb))])
                target.extend([h[m].detach() for h, m in zip(th, content(tb))])

            def score(perm):
                s = torch.nn.utils.rnn.pad_sequence(source, batch_first=True)
                ts = [target[j] for j in perm]
                t = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True)
                sm = torch.arange(s.shape[1], device='cuda')[None] < torch.tensor([len(x) for x in source], device='cuda')[:, None]
                tm = torch.arange(t.shape[1], device='cuda')[None] < torch.tensor([len(x) for x in ts], device='cuda')[:, None]
                a, u, e = token_loss(s, t, sm, tm, args)
                return a.cpu().tolist(), u.cpu().tolist(), float(e.max())

            matched, uniform, marginal = score(list(range(len(indices))))
            shuffled = []
            for number, perm in enumerate(permutations):
                values, _, error = score(perm)
                shuffled.append(values); marginal = max(marginal, error)
                if number % 5 == 0:
                    print('SHUFFLE', model_id, number, flush=True)
        del source, target
        gradients = []
        parameters = [p for p in model.parameters() if p.requires_grad]
        for ids in [x[:4] for x in plan['gradient_batches'][:4]]:
            sb, tb = inputs(ids), inputs(ids, True)
            labels = tb['input_ids'].masked_fill(~tb['attention_mask'].bool(), -100)
            output = model(**sb, labels=labels, use_cache=False, output_hidden_states=True)
            with torch.no_grad():
                th = encode(tb).detach()
            tl, _, err = token_loss(output.encoder_last_hidden_state, th, content(sb), content(tb), args)
            ce_grad = torch.autograd.grad(output.loss, parameters, retain_graph=True, allow_unused=True)
            tg_grad = torch.autograd.grad(0.1 * tl.mean(), parameters, allow_unused=True)
            ce_norm = sum(float(g.double().square().sum()) for g in ce_grad if g is not None) ** 0.5
            tg_norm = sum(float(g.double().square().sum()) for g in tg_grad if g is not None) ** 0.5
            dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(ce_grad, tg_grad) if a is not None and b is not None)
            finite = all(bool(torch.isfinite(g).all()) for g in tg_grad if g is not None)
            gradients.append({'indices': ids, 'microbatch': 4, 'ce': float(output.loss.detach()), 'token_loss': float(tl.mean().detach()), 'weighted_norm_ratio': tg_norm / ce_norm, 'cosine': dot / max(ce_norm * tg_norm, 1e-30), 'finite': finite, 'marginal_error': float(err.max())})
            assert finite
            del ce_grad, tg_grad, output, th, tl
        assert versions == {n: p._version for n, p in model.named_parameters()}
        result = {'matched_cost': matched, 'uniform_cost': uniform, 'shuffled_costs': shuffled, 'mean_matched': float(np.mean(matched)), 'mean_shuffled': float(np.mean(shuffled)), 'mean_uniform': float(np.mean(uniform)), 'shuffled_minus_matched': float(np.mean(shuffled) - np.mean(matched)), 'marginal_error_max': marginal, 'gradients': gradients, 'parameter_versions_unchanged': True, 'archive_sha256': entry['archive_sha256'], 'strict_loading': info, 'optimizer_steps': 0}
        results[model_id] = result
        (out / model_id / 'RESULT.json').write_text(json.dumps(result, indent=2))
        print('RESULT', model_id, result['mean_matched'], result['shuffled_minus_matched'], flush=True)
        del parameters, model, base
        torch.cuda.empty_cache()
    summary = {'models': results, 'correctness': checks, 'indices': indices, 'permutations': permutations, 'arguments': vars(args), 'precision': 'FP32', 'dropout': False, 'tf32': False, 'gpu': torch.cuda.get_device_name(0), 'worker_sha256': sha(__file__), 'validation_sha256': sha(data), 'revision': plan['revision'], 'optimizer_steps': 0, 'completed_unix': time.time(), 'limits': 'Frozen selected validation checkpoints; not translation efficacy, gold alignment accuracy, or training-time Adam updates. Token loss uses content masks, unlike historical sentence pooling. Gradients use microbatch4, not centered-LSR diagnostics microbatch32.'}
    (out / 'COMPLETE.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
