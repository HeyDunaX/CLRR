"""Frozen-checkpoint TRAIN diagnostics. No optimizer, generation, or test data.

Direction controls are matched to the local DRM update norm PER HEAD. This is
an intervention, not a training recipe or a faithful coverage-paper replication.
"""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def controlled_update(memory, attention, direction, eta, active, arm, seed):
    import torch
    # Compute deltas before subtraction to avoid FP32 cancellation near zero.
    def delta_for(vector):
        support = (memory * vector.unsqueeze(-2)).sum(-1).clamp_min(0)
        return -(eta * attention * support * active[:, None, None]).unsqueeze(-1) * vector.unsqueeze(-2)
    reference = delta_for(direction)
    if arm == 'drm':
        return memory + reference, 0., 0.
    if arm == 'zero':
        return memory, 0., 0.
    if arm == 'context':
        vector = (attention.unsqueeze(-1) * memory).sum(-2)
    elif arm == 'random':
        generator = torch.Generator(device=memory.device).manual_seed(seed)
        vector, best = None, None
        # Eight random axes; sign and axis chosen by label-free support only.
        # This avoids magnifying an almost orthogonal/empty random half-space.
        for _ in range(8):
            candidate = torch.randn(direction.shape, generator=generator,
                                    device=memory.device, dtype=memory.dtype)
            candidate = candidate / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            positive = delta_for(candidate).norm(dim=(-2, -1))
            negative = delta_for(-candidate).norm(dim=(-2, -1))
            candidate = torch.where((positive >= negative).unsqueeze(-1), candidate, -candidate)
            score = torch.maximum(positive, negative)
            if vector is None:
                vector, best = candidate, score
            else:
                vector = torch.where((score > best).unsqueeze(-1), candidate, vector)
                best = torch.maximum(best, score)
    elif arm == 'scalar':
        vector = None
    else:
        raise ValueError(arm)
    if vector is None:
        raw = -eta * attention.unsqueeze(-1) * memory * active[:, None, None, None]
    else:
        vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        raw = delta_for(vector)
    wanted = reference.norm(dim=(-2, -1), keepdim=True)
    available = raw.norm(dim=(-2, -1), keepdim=True)
    scale = torch.where(available > 1e-12, wanted / available.clamp_min(1e-12),
                        torch.zeros_like(wanted))
    delta = raw * scale
    error = ((delta.norm(dim=(-2, -1), keepdim=True) - wanted).abs()
             / wanted.clamp_min(1e-8)).max().item()
    # A large coefficient can overshoot the scalar/projection contraction.
    coefficient = (eta * attention * scale.squeeze(-1)).max().item()
    if error > 1e-5 or coefficient > 1.:
        raise ValueError(f'Invalid matched contraction: error={error}, coefficient={coefficient}')
    return memory + delta, error, coefficient


@contextmanager
def intervention(arm, row, at=None):
    import torch
    import methods.decision_residual_memory.v1.model as module
    original = module.update_memory
    stats = dict(step=0, norm_match_error=0., max_coefficient=0., reads=[])
    def update(memory, attention, direction, eta, active):
        step = stats['step']
        stats['step'] += 1
        stats['reads'].append(int(attention.mean(1).argmax(-1)[0]))
        if at is not None and step != at:
            return memory
        result, error, coefficient = controlled_update(
            memory, attention, direction, eta, active, arm,
            seed=1729 + row * 1009 + step)
        stats['norm_match_error'] = max(stats['norm_match_error'], error)
        stats['max_coefficient'] = max(stats['max_coefficient'], coefficient)
        return result
    module.update_memory = update
    try:
        yield stats
    finally:
        module.update_memory = original


def main(args):
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
    from experiment_runner.integrity import restore_archive, sha256
    from methods.decision_residual_memory import DecisionResidualMemoryMT
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda')
    out = (ROOT / args.output).resolve()
    assert out.is_relative_to(ROOT / 'outputs/experiments')
    assert not (out / 'COMPLETE.json').exists()
    out.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        temp = out / (name + '.tmp')
        temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(out / name)
    plan = json.loads((ROOT / args.config).read_text())
    train = ROOT / 'data_processed/amis_mandarin/train.csv'
    assert sha256(train) == plan['data']['train']['sha256']
    frame = pd.read_csv(train, keep_default_na=False)
    rows = torch.randperm(len(frame), generator=torch.Generator().manual_seed(1729)).tolist()[:args.rows]
    record = plan['reused_w42']
    drm_receipt = json.loads((ROOT / 'outputs/experiments/amis_drm_validation_fix_rtx/runs/DRM_S42/collection_receipt.json').read_text())
    assert drm_receipt['verified']
    archive_path, archive_record = next((k, v) for k, v in drm_receipt['files'].items() if k.endswith('/best.zip'))
    checkpoints = {
        'W': dict(path=record['prefix'] + '/best.zip', revision=record['revision'], sha256=record['archive_sha256']),
        'DRM': dict(path=archive_path, revision=drm_receipt['revision'], sha256=archive_record['sha256'])}
    lock = dict(checkpoints=checkpoints, train_sha256=sha256(train), rows=rows,
                script_sha256=sha256(__file__), config_sha256=sha256(ROOT / args.config),
                implementation_sha256=sha256(ROOT / 'src/methods/decision_residual_memory/v1/model.py'),
                optimizer_steps=0, test_used=False, eta=.1, seed=1729,
                precision='FP32', backend='SDPA', dropout=False,
                scope='32 fixed TRAIN rows; B1 diagnostics, not B4/GA32 training or translation quality',
                controls='per-head local norm-matched; controls still compute gradient for isolation, not optimized latency',
                random_control='best-of-8 deterministic random axes and sign by attention-weighted positive support, no labels or gradients for selection; coefficients must be <=1',
                nll_reference='sequential-zero, after native/zero logits check; removes parallel-versus-sequential kernel offset',
                single_intervention='one update at middle supervised position, other updates zero; no verified semantic alignments')
    write('PLAN.json', lock)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for checkpoint, pinned in checkpoints.items():
        transient = out / ('transient_' + checkpoint)
        write('status.json', dict(stage='RESTORE', checkpoint=checkpoint))
        archive = Path(hf_hub_download(plan['repo'], pinned['path'], revision=pinned['revision'],
                                     token=False, local_dir=transient / 'download'))
        assert sha256(archive) == pinned['sha256']
        restore_archive(archive, transient / 'export')
        tokenizer = AutoTokenizer.from_pretrained(transient / 'export', use_fast=True)
        tokenizer.src_lang, tokenizer.tgt_lang = 'tl_XX', 'zh_CN'
        base, info = AutoModelForSeq2SeqLM.from_pretrained(transient / 'export',
                         attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
        model = DecisionResidualMemoryMT(base).cuda().eval()
        if checkpoint == 'DRM':
            model.verify_checkpoint_metadata(transient / 'export')
        versions = [p._version for p in model.parameters()]
        collator = DataCollatorForSeq2Seq(tokenizer, model=None, pad_to_multiple_of=8)
        for n, row in enumerate(rows):
            batch = {k:v.cuda() for k,v in collator([tokenizer(frame.iloc[row].source,
                text_target=frame.iloc[row].target, truncation=True, max_length=256)]).items()}
            labels = batch['labels'][0]
            valid = labels.ne(-100) & labels.ne(tokenizer.eos_token_id) & labels.ne(tokenizer.pad_token_id)
            safe = labels.clamp_min(0)
            with torch.no_grad():
                model.eta = 0.
                native = model(**batch).logits[0].detach()
                reference = -native.log_softmax(-1).gather(-1, safe[:,None]).squeeze(-1)
                model.eta = .1
                for arm in ('zero', 'drm', 'context', 'random', 'scalar'):
                    torch.cuda.synchronize(); started = time.perf_counter()
                    with intervention(arm, row) as stats:
                        logits = model(**batch).logits[0].detach()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    if arm == 'zero':
                        error = float((logits - native).abs().max())
                        torch.testing.assert_close(logits, native, atol=3e-5, rtol=3e-5)
                    else:
                        error = None
                    nll = -logits.log_softmax(-1).gather(-1, safe[:,None]).squeeze(-1)
                    results.append(dict(checkpoint=checkpoint, row=row, kind='trajectory', arm=arm,
                        mean_nll_delta=float((nll-reference)[valid].mean()), tokens=int(valid.sum()),
                        seconds=elapsed, zero_max_logit_error=error,
                        norm_match_error=stats['norm_match_error'], max_coefficient=stats['max_coefficient']))
                    if arm == 'zero':
                        reference = nll
                positions = valid.nonzero().flatten().tolist()
                at = positions[len(positions)//2] if len(positions)>2 else None
                if at is not None:
                    for arm in ('drm', 'context', 'random', 'scalar'):
                        with intervention(arm, row, at=at) as stats:
                            logits = model(**batch).logits[0].detach()
                        torch.testing.assert_close(logits[:at+1], native[:at+1], atol=3e-5, rtol=3e-5)
                        nll = -logits.log_softmax(-1).gather(-1, safe[:,None]).squeeze(-1)
                        future = valid.clone(); future[:at+1] = False
                        if future.any():
                            indices = future.nonzero().flatten().tolist()
                            proxy = [i for i in indices if stats['reads'][i] == stats['reads'][at]]
                            results.append(dict(checkpoint=checkpoint, row=row, kind='single', arm=arm,
                                at=at, mean_nll_delta=float((nll-reference)[future].mean()),
                                next_nll_delta=float(nll[indices[0]]-reference[indices[0]]),
                                same_attention_argmax_tokens=len(proxy),
                                same_attention_argmax_nll_delta=float((nll-reference)[proxy].mean()) if proxy else None,
                                norm_match_error=stats['norm_match_error'], max_coefficient=stats['max_coefficient']))
            write('measurements.json', results)
            write('status.json', dict(stage='MEASURE', checkpoint=checkpoint, completed_rows=n+1, total_rows=len(rows)))
        assert versions == [p._version for p in model.parameters()]
        assert all(p.grad is None for p in model.parameters())
        del model, base, batch, native, logits
        gc.collect(); torch.cuda.empty_cache()
        # Public archive is pinned and verified above; remove only our restore.
        assert transient.resolve().is_relative_to(out)
        shutil.rmtree(transient)
    summary = []
    rng = np.random.default_rng(1729)
    for checkpoint in checkpoints:
        for kind in ('trajectory', 'single'):
            for arm in ('drm', 'context', 'random', 'scalar'):
                subset = [r for r in results if r['checkpoint']==checkpoint and r['kind']==kind and r['arm']==arm]
                values = np.array([r['mean_nll_delta'] for r in subset])
                means = values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
                summary.append(dict(checkpoint=checkpoint, kind=kind, arm=arm, sentences=len(values),
                    mean_nll_delta=float(values.mean()), ci95=np.quantile(means,[.025,.975]).tolist(),
                    harmed_sentences=int((values>0).sum()),
                    max_norm_match_error=max(r['norm_match_error'] for r in subset),
                    max_coefficient=max(r['max_coefficient'] for r in subset)))
    write('SUMMARY.json', summary)
    write('COMPLETE.json', dict(status='COMPLETE', optimizer_steps=0, test_used=False,
          parameters_unchanged=True, checkpoints=checkpoints, measurements_sha256=sha256(out/'measurements.json'),
          summary_sha256=sha256(out/'SUMMARY.json'), peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))
    write('status.json', dict(status='COMPLETE'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=32)
    main(parser.parse_args())
