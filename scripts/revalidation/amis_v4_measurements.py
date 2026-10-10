"""Auditable measurements for the V4 checkpoint verification worker."""
import hashlib
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def teacher_diagnostics(raw, routed, tokenizer, rows, indices):
    batches = []
    ce_on = ce_off = labels_total = diff_sum = diff_count = 0
    max_diff = zero_diff = 0.0
    for batch_indices in indices:
        chunk = [rows[i] for i in batch_indices]
        x = tokenizer([r['source'] for r in chunk], padding=True, truncation=True,
                      max_length=256, pad_to_multiple_of=8, return_tensors='pt')
        labels = tokenizer(text_target=[r['target'] for r in chunk], padding=True,
                           truncation=True, max_length=256, return_tensors='pt')['input_ids'].cuda()
        labels[labels == tokenizer.pad_token_id] = -100
        inputs = {k: v.cuda() for k, v in x.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad(), routed._fresh_cache():
            routed.strength = 0
            zero = routed(**inputs, labels=labels)
        # Same hooked model at alpha=0; then remove hooks briefly to get truly raw output.
        for h in routed._handles:
            h.remove()
        routed._handles.clear()
        with torch.no_grad():
            off = raw(**inputs, labels=labels)
        zero_diff = max(zero_diff, float((zero.logits - off.logits).abs().max()))
        del zero
        routed._install_hooks()
        routed.strength = .1
        with torch.no_grad(), routed._fresh_cache():
            on = routed(**inputs, labels=labels)
        difference = (on.logits - off.logits).abs()
        count = int(labels.ne(-100).sum())
        ce_on += float(on.loss) * count
        ce_off += float(off.loss) * count
        labels_total += count
        diff_sum += float(difference.double().sum())
        diff_count += difference.numel()
        max_diff = max(max_diff, float(difference.max()))
        batches.append({'indices': batch_indices, 'ce_on': float(on.loss), 'ce_off': float(off.loss),
                        'label_tokens': count, 'logit_diff_mean': float(difference.mean()),
                        'logit_diff_max': float(difference.max())})
        del on, off, difference
    assert zero_diff < 1e-6, f'Checkpoint alpha0/raw mismatch: {zero_diff}'
    return {'teacher_forced_ce_on': ce_on / labels_total, 'teacher_forced_ce_off': ce_off / labels_total,
            'ce_delta_on_minus_off': (ce_on - ce_off) / labels_total,
            'logit_diff_mean': diff_sum / diff_count, 'logit_diff_max': max_diff,
            'alpha0_vs_clean_raw_max': zero_diff, 'batches': batches,
            'ce_aggregation': 'nonignored_label_token_weighted', 'logit_scope': 'all logits including label padding'}


def spectrum(matrix):
    centered = matrix.astype(np.float64) - matrix.astype(np.float64).mean(axis=0, keepdims=True)
    values = np.linalg.svd(centered, compute_uv=False)
    energy = values ** 2
    def rank(weights):
        if weights.sum() == 0:
            return 0.0
        p = weights[weights > 0] / weights.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    q = energy / energy.sum() if energy.sum() else np.zeros_like(energy)
    return {'effective_rank_singular_values': rank(values),
            'effective_rank_covariance_energy': rank(energy),
            'participation_ratio': float(1 / (q ** 2).sum()) if energy.sum() else 0.0,
            'singular_values': values.tolist(),
            'top1_variance_fraction': float(q[0]),
            'centered_rank_limit': min(len(matrix) - 1, matrix.shape[1])}


def gradient_group(groups, ce_grads, lsr_grads, name):
    ce_sq = lsr_sq = dot = 0.0
    ce_present = lsr_present = 0
    for group, ce, lsr in zip(groups, ce_grads, lsr_grads):
        if name != 'all_parameters' and group != name:
            continue
        if ce is not None:
            ce_sq += float(ce.float().square().sum())
            ce_present += 1
        if lsr is not None:
            lsr_sq += float(lsr.float().square().sum())
            lsr_present += 1
        if ce is not None and lsr is not None:
            dot += float((ce.float() * lsr.float()).sum())
    a, b = ce_sq ** .5, lsr_sq ** .5
    return {'ce_gradient_norm': a, 'lsr_gradient_norm': b,
            'gradient_cosine': dot / (a * b) if a > 0 and b > 0 else None,
            'lsr_to_ce_ratio': b / a if a > 0 else None,
            'ce_norm_squared': ce_sq, 'lsr_norm_squared': lsr_sq, 'dot_product': dot,
            'ce_present_parameters': ce_present, 'lsr_present_parameters': lsr_present}


def load_checkpoint(entry, destination, receipts):
    archive = Path(hf_hub_download(repo_id=entry['archive_repo'], filename=entry['archive_path'],
                                 revision=entry['revision']))
    assert archive.stat().st_size == entry['archive_bytes'], 'archive size mismatch'
    actual = digest(archive)
    assert actual == entry['archive_sha256'], 'archive SHA mismatch'
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        root = destination.resolve()
        assert all((root / n).resolve().is_relative_to(root) for n in z.namelist())
        z.extractall(destination)
    configs = [p.parent for p in destination.rglob('config.json')
               if (p.parent / 'model.safetensors').exists() or (p.parent / 'pytorch_model.bin').exists()]
    assert len(configs) == 1, f'Expected one unambiguous checkpoint, found {configs}'
    receipt = {**entry, 'download_revision': entry['revision'], 'verified_sha256': actual,
               'verified_bytes': archive.stat().st_size, 'checkpoint_relative_path': str(configs[0].relative_to(destination))}
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / (entry['id'] + '.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return configs[0]


def strict_model(path):
    model, info = AutoModelForSeq2SeqLM.from_pretrained(path, output_loading_info=True, torch_dtype=torch.float32)
    assert not any(info[k] for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), info
    model.gradient_checkpointing_disable()
    model.config.forced_bos_token_id = 250025
    model.generation_config.forced_bos_token_id = 250025
    return model.cuda()


class Collector:
    """Post-routing block outputs, final LN output, and per-sentence residuals."""
    def __init__(self, model, rewired):
        self.rewired = rewired
        self.enabled = False
        self.phase = ''
        self.mask = None
        self.row_start = 0
        self.pooled_by_layer = {}
        self.residual_stats = []
        encoder = model.get_encoder()
        self.num_layers = len(encoder.layers)
        self.handles = [layer.register_forward_hook(self.hook(i)) for i, layer in enumerate(encoder.layers)]
        self.handles.append(encoder.layer_norm.register_forward_hook(self.hook('final')))

    def hook(self, index):
        def collect(module, inputs, output):
            if not self.enabled:
                return
            post = (output[0] if isinstance(output, (tuple, list)) else output).detach().float()
            mask = self.mask.to(post.device).float()
            pooled = (post * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
            site = index + 1 if isinstance(index, int) else 'final'
            self.pooled_by_layer.setdefault((self.phase, site), []).append(pooled.cpu().numpy())
            if self.rewired is None or not isinstance(index, int):
                return
            source = self.rewired._cache['encoder'].get(index - self.rewired.distance)
            if source is None:
                return
            skip = source.detach().float()
            residual = self.rewired.strength * skip
            current = post - residual
            count = (mask.sum(1) * post.shape[-1]).clamp(min=1)
            res_rms = ((residual.square() * mask.unsqueeze(-1)).sum((1, 2)) / count).sqrt()
            cur_rms = ((current.square() * mask.unsqueeze(-1)).sum((1, 2)) / count).sqrt()
            cosine = torch.nn.functional.cosine_similarity(skip, current, dim=-1)
            cos = (cosine * mask).sum(1) / mask.sum(1).clamp(min=1)
            for row in range(len(post)):
                self.residual_stats.append({'phase': self.phase, 'index': self.row_start + row,
                    'layer': index + 1, 'residual_rms': float(res_rms[row]), 'current_rms': float(cur_rms[row]),
                    'residual_to_current_rms': float(res_rms[row] / cur_rms[row].clamp(min=1e-12)),
                    'source_current_token_cosine_mean': float(cos[row])})
        return collect

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def runtime(tokenizer, root):
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
            'host_ram_bytes': __import__('os').sysconf('SC_PAGE_SIZE') * __import__('os').sysconf('SC_PHYS_PAGES'),
            'tokenizer_class': type(tokenizer).__name__, 'src_lang': tokenizer.src_lang, 'tgt_lang': tokenizer.tgt_lang,
            'source_prefix': tokenizer('source')['input_ids'][0],
            'target_prefix': tokenizer(text_target='target')['input_ids'][0],
            'tokenizer_revision': 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30',
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'source_hashes': {str(p.relative_to(root)): digest(p) for folder in ['scripts/revalidation', 'src']
                              for p in (root / folder).rglob('*.py')},
            'validation_sha256': digest(root / 'data_processed/amis_mandarin/validation.csv')}


def local_checks():
    ce = [torch.tensor([3.]), torch.tensor([4.])]
    lsr = [torch.tensor([2.]), None]
    g = gradient_group(['encoder', 'decoder'], ce, lsr, 'all_parameters')
    assert g['ce_gradient_norm'] == 5 and g['lsr_gradient_norm'] == 2 and g['gradient_cosine'] == .6
    assert gradient_group(['encoder', 'decoder'], ce, lsr, 'decoder')['ce_gradient_norm'] == 4
    s = spectrum(np.array([[3., 0.], [-3., 0.], [0., 1.], [0., -1.]]))
    assert abs(s['participation_ratio'] - 1.2195121951219512) < 1e-10
    assert abs(s['effective_rank_covariance_energy'] - 1.384145488461686) < 1e-10
    assert spectrum(np.zeros((4, 2)))['effective_rank_singular_values'] == 0
    return {'gradient_union_check': 'PASS', 'rank_definition_check': 'PASS', 'zero_spectrum_check': 'PASS'}
