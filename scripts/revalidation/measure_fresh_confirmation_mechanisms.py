"""Measure Fresh Confirmation Mechanisms on Checkpoints (Step B).
Measures:
1. CE grad vs Weighted LSR grad (norm ratio, cosine, dot product).
2. Layer-wise residual RMS and skip-current cosine across encoder layers.
3. Representation spectra: SVD effective rank, participation ratio, pairwise cosine.
4. Route On/Off comparison on the exact same weights (Validation set).

Strictly adheres to Section 9.6:
- Zero optimizer steps.
- Temporary checkpoint download and verification from Hugging Face.
- Immediate cleanup of weights after measurement.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from amis_rewire import train
from amis_rewire.modeling import (
    CrossLayerResidualRewire,
    JEPAGuidedSeq2SeqLM,
    _find_stack_layers,
    _masked_mean_pool,
)
from amis_rewire.metrics import generation_metrics
from sacrebleu.metrics import CHRF

FAMILY_CONFIGS = {
    'baseline': {'method': 'baseline', 'alpha': 0.0, 'lambda': 0.0, 'repo': 'FiveC/amis-fresh-confirmation-baseline-20261005'},
    'clrr':     {'method': 'clrr-enc', 'alpha': 0.10, 'lambda': 0.0, 'repo': 'FiveC/amis-fresh-confirmation-clrr-20261005'},
    'lsr':      {'method': 'jepa',     'alpha': 0.0,  'lambda': 0.10, 'repo': 'FiveC/amis-fresh-confirmation-lsr-20261005'},
    'full':     {'method': 'jepa-clrr-enc', 'alpha': 0.10, 'lambda': 0.10, 'repo': 'FiveC/amis-fresh-confirmation-full-20261005'}
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda: s.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def weight_digest(folder):
    from safetensors.torch import load_file
    for name in ('model.safetensors', 'pytorch_model.bin'):
        p = Path(folder) / name
        if p.exists():
            if name.endswith('.safetensors'):
                state = load_file(str(p), device='cpu')
            else:
                state = torch.load(p, map_location='cpu', weights_only=True)
            h = hashlib.sha256()
            for k, t in sorted(state.items()):
                h.update(k.encode())
                h.update(str((t.dtype, tuple(t.shape))).encode())
                h.update(memoryview(t.contiguous().numpy()).cast('B'))
            del state
            gc.collect()
            return h.hexdigest()
    raise FileNotFoundError(f"No weights found in {folder}")

def spectrum_diagnostics(representations):
    raw = np.asarray(representations, dtype='float64')
    centered = raw - raw.mean(axis=0, keepdims=True)
    singular = torch.linalg.svdvals(torch.from_numpy(centered).cuda()).cpu().numpy()
    numerical_tolerance = singular[0] * max(centered.shape) * np.finfo('float64').eps
    def erank(values):
        if values.sum() == 0: return 0.0
        p = values[values > 0] / values.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    energy = singular ** 2
    norm = np.linalg.norm(raw, axis=1)
    unit = raw / np.maximum(norm[:, None], 1e-12)
    pairs = (unit @ unit.T)[np.triu_indices(len(raw), 1)]
    return dict(
        rows=len(raw),
        hidden_dim=raw.shape[1],
        maximum_centered_rank=min(len(raw) - 1, raw.shape[1]),
        numerical_rank=int((singular > numerical_tolerance).sum()),
        effective_rank_singular_values=erank(singular),
        effective_rank_covariance_energy=erank(energy),
        participation_ratio=float(energy.sum() ** 2 / (energy ** 2).sum()) if energy.sum() > 0 else 0.0,
        covariance_trace=float(energy.sum() / (len(raw) - 1)),
        top1_variance_fraction=float(energy[0] / energy.sum()) if energy.sum() > 0 else 0.0,
        top10_variance_fraction=float(energy[:10].sum() / energy.sum()) if energy.sum() > 0 else 0.0,
        mean_pairwise_cosine=float(pairs.mean()),
        std_pairwise_cosine=float(pairs.std()),
        pooled_norm_mean=float(norm.mean()),
        pooled_norm_std=float(norm.std())
    )

class LayerMeasurements:
    def __init__(self, model, rewired):
        self.rewired = rewired
        self.enabled = False
        self.phase = ''
        self.row_start = 0
        self.mask = None
        self.pre = {}
        self.rows = []
        self.pooled = {}
        self.handles = []
        for index, layer in enumerate(_find_stack_layers(model, 'encoder')):
            self.handles.append(layer.register_forward_hook(self.before(index), prepend=True))
            self.handles.append(layer.register_forward_hook(self.after(index)))

    def before(self, index):
        def hook(_module, _inputs, output):
            if self.enabled:
                self.pre[index] = (output[0] if isinstance(output, (tuple, list)) else output).detach()
        return hook

    def after(self, index):
        def hook(_module, _inputs, output):
            if not self.enabled: return
            post = (output[0] if isinstance(output, (tuple, list)) else output).detach().float()
            pre = self.pre.pop(index).float()
            mask = self.mask.to(pre.device).float().unsqueeze(-1)
            denom = (mask.sum(dim=1).squeeze(-1) * pre.shape[-1]).clamp_min(1)
            rms = lambda t: ((t.square() * mask).sum((1, 2)) / denom).sqrt()
            pre_rms, post_rms = rms(pre), rms(post)
            ratio = torch.zeros_like(pre_rms)
            skip_rms = torch.zeros_like(pre_rms)
            cosine = None
            if self.rewired is not None and index >= self.rewired.distance:
                skip = self.rewired._cache['encoder'][index - self.rewired.distance].float()
                residual = self.rewired.strength * skip
                skip_rms = rms(skip)
                ratio = rms(residual) / pre_rms.clamp_min(1e-12)
                dot = (pre * skip * mask).sum((1, 2))
                cosine = dot / ((pre.square() * mask).sum((1, 2)).sqrt() * (skip.square() * mask).sum((1, 2)).sqrt()).clamp_min(1e-12)
            for i in range(len(pre_rms)):
                self.rows.append({
                    'phase': self.phase,
                    'row_index': self.row_start + i,
                    'layer': index + 1,
                    'pre_rms': float(pre_rms[i]),
                    'post_rms': float(post_rms[i]),
                    'skip_rms': float(skip_rms[i]),
                    'residual_to_current_rms': float(ratio[i]),
                    'skip_current_cosine': float(cosine[i]) if cosine is not None else None
                })
            pooled = _masked_mean_pool(post, self.mask).cpu().numpy()
            self.pooled.setdefault((self.phase, index + 1), []).append(pooled)
        return hook

    def close(self):
        for h in self.handles: h.remove()

def gradient_group(name, parameter, embedding):
    if parameter is embedding: return 'shared_input_embeddings'
    if 'encoder' in name: return 'encoder_without_shared_embeddings'
    if 'decoder' in name: return 'decoder_without_shared_embeddings'
    return 'other_parameters'

def measure_gradients(model, tokenizer, rows, max_length, method):
    indices = sorted(np.random.default_rng(20261002).choice(len(rows), size=32, replace=False).tolist())
    model.train()
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    names, parameters = zip(*model.named_parameters())
    embedding = model.get_input_embeddings().weight
    groups = [gradient_group(n, p, embedding) for n, p in zip(names, parameters)]
    results = []
    
    for batch_id, start in enumerate(range(0, len(indices), 4)):
        selected = indices[start:start+4]
        batch_rows = [rows[i] for i in selected]
        x = tokenizer([r['source'] for r in batch_rows], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        y = tokenizer(text_target=[r['target'] for r in batch_rows], padding=True, truncation=True, max_length=max_length, return_tensors='pt')['input_ids']
        y[y == tokenizer.pad_token_id] = -100
        inputs = {k: v.cuda() for k, v in x.items() if k in ['input_ids', 'attention_mask']}
        labels = y.cuda()
        
        target_captures = []
        native_ce = []
        def capture(_module, _inputs, output): target_captures.append(output.last_hidden_state)
        handle = model.get_encoder().register_forward_hook(capture)
        underlying = model
        while isinstance(underlying, (CrossLayerResidualRewire, JEPAGuidedSeq2SeqLM)):
            underlying = underlying.base_model
        def capture_ce(_module, _inputs, output): native_ce.append(output.loss)
        loss_handle = underlying.register_forward_hook(capture_ce)
        
        torch.manual_seed(1000 + batch_id)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**inputs, labels=labels, output_hidden_states=True, return_dict=True)
            assert len(native_ce) == 1
            ce = native_ce[0]
            if method in ['baseline', 'clrr']:
                target_ids = labels.clone()
                target_ids[target_ids == -100] = model.config.pad_token_id
                target_mask = target_ids.ne(model.config.pad_token_id).long()
                fresh_ctx = model._fresh_cache() if hasattr(model, '_fresh_cache') else contextlib.nullcontext()
                with torch.no_grad(), fresh_ctx:
                    target = model.get_encoder()(input_ids=target_ids, attention_mask=target_mask, return_dict=True).last_hidden_state
            else:
                assert len(target_captures) == 2
                target = target_captures[-1]
                target_mask = labels.ne(-100).long()
            src = F.normalize(_masked_mean_pool(outputs.encoder_last_hidden_state, inputs['attention_mask']), dim=-1)
            tgt = F.normalize(_masked_mean_pool(target, target_mask), dim=-1)
            lsr = (1 - (src * tgt).sum(-1)).mean()
            weighted = 0.1 * lsr.to(ce.dtype)
            if method in ['full', 'lsr']:
                weighted = outputs.loss - ce
                lsr = weighted / 0.1
                
        handle.remove()
        loss_handle.remove()
        
        ce_grads = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
        lsr_grads = torch.autograd.grad(weighted, parameters, allow_unused=True)
        
        sums = {key: [0., 0., 0.] for key in ['all_parameters', *sorted(set(groups))]}
        for group, gce, glsr in zip(groups, ce_grads, lsr_grads):
            nce = float(gce.float().norm()) ** 2 if gce is not None else 0.
            nlsr = float(glsr.float().norm()) ** 2 if glsr is not None else 0.
            dot = float((gce.float() * glsr.float()).sum()) if gce is not None and glsr is not None else 0.
            for key in ['all_parameters', group]:
                sums[key][0] += nce
                sums[key][1] += nlsr
                sums[key][2] += dot
                
        item = {
            'batch_id': batch_id,
            'ce_loss': float(ce.detach()),
            'lsr_loss': float(lsr.detach()),
            'weighted_lsr_loss': float(weighted.detach()),
            'groups': {}
        }
        for key, (a, b, c) in sums.items():
            item['groups'][key] = {
                'ce_gradient_norm': a ** 0.5,
                'weighted_lsr_gradient_norm': b ** 0.5,
                'lsr_to_ce_norm_ratio': (b / a) ** 0.5 if a > 0 else None,
                'gradient_cosine': c / (a * b) ** 0.5 if (a > 0 and b > 0) else None,
                'combined_gradient_norm': max(0, a + b + 2 * c) ** 0.5
            }
        results.append(item)
        del outputs, target, src, tgt, ce, lsr, weighted, ce_grads, lsr_grads
        model.zero_grad(set_to_none=True)
        
    model.eval()
    return results

def measure_route_on_off(model, tokenizer, rows, max_length):
    """Measures validation loss and BLEU with routing ON vs routing forced OFF on same weights."""
    if not hasattr(model, 'strength'):
        # Not a rewired model (Baseline or LSR)
        return None
        
    orig_strength = model.strength
    results = {}
    
    # 1. Route ON
    model.strength = orig_strength
    val_preds_on = []
    val_refs = [r['target'] for r in rows]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch = rows[start:start+16]
            inputs = tokenizer([r['source'] for r in batch], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            generated = model.generate(**inputs, num_beams=1, max_length=max_length)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            val_preds_on.extend([s.strip() for s in decoded])
    m_on = generation_metrics(val_preds_on, val_refs, tokenize='zh')
    
    # 2. Route OFF
    model.strength = 0.0
    val_preds_off = []
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch = rows[start:start+16]
            inputs = tokenizer([r['source'] for r in batch], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            generated = model.generate(**inputs, num_beams=1, max_length=max_length)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            val_preds_off.extend([s.strip() for s in decoded])
    m_off = generation_metrics(val_preds_off, val_refs, tokenize='zh')
    
    # Restore strength
    model.strength = orig_strength
    
    return {
        'route_on': {'bleu': m_on['bleu'], 'chrfpp': m_on['chrf++']},
        'route_off': {'bleu': m_off['bleu'], 'chrfpp': m_off['chrf++']},
        'delta_bleu_on_minus_off': m_on['bleu'] - m_off['bleu'],
        'delta_chrfpp_on_minus_off': m_on['chrf++'] - m_off['chrf++']
    }

def probe_single_model(family, seed, out_dir, cache_dir):
    from huggingface_hub import hf_hub_download
    
    cfg = FAMILY_CONFIGS[family]
    repo = cfg['repo']
    run_name = f"mbart-amis-{family}-seed{seed}-fresh-confirmation"
    remote_zip = f"amis_fresh_confirmation_20261005/{run_name}/{run_name}-best.zip"
    
    model_out = out_dir / family / f"seed{seed}"
    model_out.mkdir(parents=True, exist_ok=True)
    done_file = model_out / 'DONE.json'
    if done_file.exists():
        print(f"[{family.upper()} seed {seed}] Already measured.")
        return
        
    print(f"\n[{family.upper()} seed {seed}] Downloading checkpoint from Hugging Face ({repo})...")
    dl_zip = Path(hf_hub_download(repo, remote_zip, cache_dir=str(cache_dir)))
    
    # Read VAL_COMPLETE.json to verify archive SHA
    val_complete_path = ROOT / f"outputs_rebuttal/amis_fresh_confirmation_20261005/{family}/seed{seed}/VAL_COMPLETE.json"
    val_data = json.loads(val_complete_path.read_text())
    expected_archive_sha = val_data['archive_sha256']
    actual_archive_sha = digest(dl_zip)
    assert actual_archive_sha == expected_archive_sha, f"Archive SHA mismatch for {family} seed {seed}"
    
    # Extract
    extract_dest = cache_dir / f"extracted_{family}_{seed}"
    if extract_dest.exists():
        import shutil
        shutil.rmtree(extract_dest)
    extract_dest.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(dl_zip) as z:
        z.extractall(extract_dest)
        
    # Checkpoint dir inside zip
    if (extract_dest / 'config.json').exists():
        ckpt_dir = extract_dest
    else:
        ckpts = [p for p in extract_dest.rglob('*') if p.is_dir() and (p / 'config.json').exists()]
        assert len(ckpts) >= 1, f"No config.json found in {extract_dest}"
        ckpt_dir = ckpts[0]
    
    # Verify weight digest
    actual_weight_digest = weight_digest(ckpt_dir)
    assert actual_weight_digest == val_data['selected_and_exported_weight_digest'], f"Weight digest mismatch {family} seed {seed}"
    print(f"[{family.upper()} seed {seed}] Verified archive and weights SHA successfully!")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    raw_model, loading = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, output_loading_info=True)
    for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']:
        assert not loading.get(k), f"Loading error: {k}={loading[k]}"
        
    train.configure_mbart(tokenizer, raw_model, 'tl_XX', 'zh_CN')
    
    rewired = None
    if family in ['full', 'clrr']:
        rewired = CrossLayerResidualRewire(raw_model, distance=2, strength=cfg['alpha'], stack='encoder')
        model = rewired
    else:
        model = raw_model
        
    if family in ['full', 'lsr']:
        model = JEPAGuidedSeq2SeqLM(model, jepa_weight=cfg['lambda'])
        
    model.cuda().eval()
    model.config.use_cache = False
    
    val_rows = list(csv.DictReader(open(ROOT / 'data_processed/amis_mandarin/validation.csv', encoding='utf-8-sig')))
    test_rows = list(csv.DictReader(open(ROOT / 'data_processed/amis_mandarin/test.csv', encoding='utf-8-sig')))
    
    # 1. Gradients
    print(f"[{family.upper()} seed {seed}] Measuring gradients...")
    gradients = measure_gradients(model, tokenizer, val_rows, max_length=256, method=family)
    (model_out / 'gradients.json').write_text(json.dumps(gradients, indent=2))
    
    # 2. Representations & Residuals
    print(f"[{family.upper()} seed {seed}] Extracting representations & layer residuals...")
    collector = LayerMeasurements(model, rewired)
    collector.enabled = True
    representations = {}
    for phase, rows in [('source', test_rows), ('target', test_rows)]:
        collector.phase = phase
        vals = []
        for start in range(0, len(rows), 16):
            batch = rows[start:start+16]
            texts = [r[phase] for r in batch]
            if phase == 'target':
                ids = tokenizer(text_target=texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
            else:
                ids = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
            ids = {k: v.cuda() for k, v in ids.items() if k in ['input_ids', 'attention_mask']}
            collector.row_start = start
            collector.mask = ids['attention_mask']
            with torch.no_grad():
                out = model.get_encoder()(**ids, return_dict=True)
                vals.append(_masked_mean_pool(out.last_hidden_state.float(), ids['attention_mask']).cpu().numpy())
        representations[phase] = np.concatenate(vals)
    collector.close()
    
    # Residual CSV
    with (model_out / 'residual_per_layer.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(collector.rows[0].keys()))
        writer.writeheader()
        writer.writerows(collector.rows)
        
    # Spectrum / Collapse
    print(f"[{family.upper()} seed {seed}] Computing SVD representation spectrum...")
    collapse = {phase: spectrum_diagnostics(reps) for phase, reps in representations.items()}
    s, t = representations['source'], representations['target']
    alignment = (s * t).sum(1) / (np.linalg.norm(s, axis=1) * np.linalg.norm(t, axis=1)).clip(1e-12)
    collapse['source_target_alignment_cosine_mean'] = float(alignment.mean())
    (model_out / 'collapse.json').write_text(json.dumps(collapse, indent=2))
    
    # 3. Route On/Off
    if family in ['full', 'clrr']:
        print(f"[{family.upper()} seed {seed}] Measuring Route On/Off effect on validation set...")
        route_eval = measure_route_on_off(rewired, tokenizer, val_rows, max_length=256)
        (model_out / 'route_on_off.json').write_text(json.dumps(route_eval, indent=2))
        
    # Done & Cleanup
    done_file.write_text(json.dumps({
        'family': family,
        'seed': seed,
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'archive_sha256': actual_archive_sha,
        'weight_digest': actual_weight_digest
    }, indent=2))
    print(f"[{family.upper()} seed {seed}] Completed and saved to {model_out}!")
    
    del model, raw_model, tokenizer, collector, representations, gradients
    import shutil
    shutil.rmtree(extract_dest, ignore_errors=True)
    dl_zip.unlink(missing_ok=True)
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['baseline', 'clrr', 'lsr', 'full', 'all'], default='all')
    parser.add_argument('--seed', type=str, default='all')
    parser.add_argument('--output-dir', type=str, default='outputs_rebuttal/amis_fresh_confirmation_mechanisms_20261007')
    parser.add_argument('--cache-dir', type=str, default='/content/probe_cache')
    args = parser.parse_args()
    
    fams = FAMILIES if args.family == 'all' else [args.family]
    seeds = SEEDS if args.seed == 'all' else [int(args.seed)]
    
    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== STARTING STEP B MECHANISM PROBE RUNNER ===")
    print(f"Families: {fams}")
    print(f"Seeds: {seeds}")
    print(f"Output Dir: {out_dir}")
    print(f"Cache Dir: {cache_dir}")
    
    for fam in fams:
        for seed in seeds:
            probe_single_model(fam, seed, out_dir, cache_dir)
            
    print("\nALL MECHANISM PROBES COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
