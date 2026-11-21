"""Frozen GOC diagnostics. No optimizer, generation, test split, or training entry point.

Part A uses analytic per-token logit gradients. Part B applies their exact VJP
through the tied head and replays each native backbone with identical RNG.
Chunking bounds vocabulary memory; CPU tests compare with ordinary autograd.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/experiments/goc_zero_step_a100'
CACHE = ROOT / '.cache/goc_zero_step'
PIN = '4df19ac6dde9c0c629214d31400064be7247d604'
W_REPO = 'FiveC/amis-rewire-checkpoints'
W_REV = 'de91462e792e66a92d51770b00cb674f4566b15d'
W_FILE = ('checkpoints/amis_mandarin/mbart-large-50/weak-full-consistency-v1/'
          'seed-42/rtx6000-fp32-sdpa-b4-ga32-es10-select-bleu/best.zip')
W_SHA = 'ebf1a2ebf080340e133be07cb902908d55f48f3165a4c973acfe358e0f457c74'
MODEL = 'facebook/mbart-large-50-many-to-many-mmt'
REV = 'e30b6cb8eb0d43a0b73cab73c7676b9863223a30'
DATA_SHA = {'train': '4e13d83f035a18c67e1b2e3990baff3aa92154a34b754a4d0cdef9a71d85b8e6',
            'validation': 'eac078ab237b55c10614f4db5ff026dc3a96a09a69c648ced29b5cdd8d4d11b4'}
BASE_SEED = 20261121
METHODS = ('CE', 'W', 'AC', 'GOC1.25', 'GOC2.50', 'SYM2.50', 'RAND2.50', 'INV2.50')
B_METHODS = ('CE', 'W', 'GOC1.25', 'SYM2.50', 'INV2.50')
TOL = 1e-7  # un-reduced per-token logit derivatives, fixed before GPU observations


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seed_views(batch_idx, pair_idx, split=0):
    return tuple(BASE_SEED + split * 100000 + batch_idx * 100 + pair_idx * 10 + v
                 for v in (0, 1))


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def distributions(z1, z2, labels):
    """Retain input dtype for FP64 auditing. Padding gets a safe gather index."""
    a, b = z1.log_softmax(-1), z2.log_softmax(-1)
    p, q = a.exp(), b.exp()
    valid = labels.ne(-100)
    safe = labels.masked_fill(~valid, 0).unsqueeze(-1)
    pg, qg = p.gather(-1, safe).squeeze(-1), q.gather(-1, safe).squeeze(-1)
    # Compare log-probabilities: no probability underflow changes ordering.
    ag, bg = a.gather(-1, safe).squeeze(-1), b.gather(-1, safe).squeeze(-1)
    winner = torch.where(ag > bg, 1., torch.where(ag < bg, 0., .5))
    return a, b, p, q, valid, safe, pg, qg, winner


def logit_gradients(z1, z2, labels, random_teacher=None):
    """Unreduced per-token auxiliary gradients, including locked beta values."""
    a, b, p, q, valid, safe, pg, qg, winner = distributions(z1, z2, labels)
    delta = a - b
    kl_pq, kl_qp = (p * delta).sum(-1, keepdim=True), -(q * delta).sum(-1, keepdim=True)
    w1 = .625 * (p * (delta - kl_pq) + p - q)
    w2 = .625 * (q * (-delta - kl_qp) + q - p)
    common = (w1 + w2) / 2
    anti = (w1 - w2) / 2
    ce1, ce2 = p.clone(), q.clone()
    ce1.scatter_add_(-1, safe, -torch.ones_like(safe, dtype=p.dtype))
    ce2.scatter_add_(-1, safe, -torch.ones_like(safe, dtype=p.dtype))
    selector = winner.unsqueeze(-1)
    d = p - q
    if random_teacher is None:
        random_teacher = torch.zeros_like(winner)
    rand = random_teacher.unsqueeze(-1).to(p.dtype)
    gradients = {
        'CE': (.5 * ce1, .5 * ce2), 'W': (w1, w2), 'AC': (anti, -anti),
        'GOC1.25': (1.25 * (1-selector) * d, -1.25 * selector * d),
        'GOC2.50': (2.5 * (1-selector) * d, -2.5 * selector * d),
        'SYM2.50': (1.25 * d, -1.25 * d),
        'RAND2.50': (2.5 * (1-rand) * d, -2.5 * rand * d),
        'INV2.50': (2.5 * selector * d, -2.5 * (1-selector) * d),
        'COMMON': (common, common),
    }
    mask = valid.unsqueeze(-1)
    return {key: (g1 * mask, g2 * mask) for key, (g1, g2) in gradients.items()}


def scalar_loss(z1, z2, labels, method, random_teacher=None):
    """Independent differentiable reference used by CPU tests (token mean)."""
    a, b, p, q, valid, safe, _, _, winner = distributions(z1, z2, labels)
    if method == 'CE':
        token = -.5 * (a.gather(-1, safe) + b.gather(-1, safe)).squeeze(-1)
    elif method == 'W':
        token = .625 * ((p-q) * (a-b)).sum(-1)
    elif method == 'AC':
        midpoint, difference = (z1+z2)/2, (z1-z2)/2
        x = (midpoint.detach()+difference).log_softmax(-1)
        y = (midpoint.detach()-difference).log_softmax(-1)
        token = .625 * ((x.exp()-y.exp())*(x-y)).sum(-1)
    else:
        beta = 1.25 if method == 'GOC1.25' else 2.5
        if method == 'SYM2.50':
            direction = torch.full_like(winner, .5)
        elif method == 'RAND2.50':
            direction = random_teacher.to(p.dtype)
        else:
            direction = winner if method.startswith('GOC') else 1-winner
        ab = (p.detach() * (a.detach()-b)).sum(-1)
        ba = (q.detach() * (b.detach()-a)).sum(-1)
        token = beta * (direction.detach()*ab + (1-direction.detach())*ba)
    return token.masked_select(valid).mean()


def margin_direction(z, g, labels):
    """Exact one-sided directional derivative of gold - max(non-gold).

    At tied maxima, max chooses the largest descent among tied competitors.
    """
    safe = labels.masked_fill(labels.eq(-100), 0).unsqueeze(-1)
    candidates = z.clone().scatter_(-1, safe, -torch.inf)
    highest = candidates.max(-1, keepdim=True).values
    descent_competitor = (-g).masked_fill(candidates.ne(highest), -torch.inf).max(-1).values
    direction = -g.gather(-1, safe).squeeze(-1) - descent_competitor
    return direction.masked_fill(labels.eq(-100), 0)


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_inputs():
    manifest_path = ROOT / 'docs/tasks/artifacts/goc_zero_step/MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    artifacts = []
    for item in manifest['files']:
        path = ROOT / item['tracked_path']
        data_bytes = path.read_bytes()
        checkout_sha = hashlib.sha256(data_bytes).hexdigest()
        restored = data_bytes.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        exact = len(data_bytes) == item['size_bytes'] and checkout_sha == item['sha256']
        crlf_only = (len(restored) == item['size_bytes'] and
                     hashlib.sha256(restored).hexdigest() == item['sha256'])
        assert exact or crlf_only, ('Artifact content mismatch beyond Git EOL', str(path))
        # Actually parse all small JSON artifacts, including the reviewer transcripts.
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
        else:
            path.read_text(encoding='utf-8')
        artifacts.append(dict(path=item['tracked_path'], manifest_sha256=item['sha256'],
                              checkout_sha256=checkout_sha,
                              verification='exact bytes' if exact else 'exact original CRLF restored in memory'))
    frames, data = {}, {}
    for split, expected in DATA_SHA.items():
        path = ROOT / f'data_processed/amis_mandarin/{split}.csv'
        assert sha(path) == expected, split
        with path.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ['source', 'target']
            frames[split] = list(reader)
        assert len(frames[split]) == (4600 if split == 'train' else 576)
        data[split] = dict(rows=len(frames[split]), sha256=expected)
    return frames, dict(artifacts=artifacts, data=data, manifest_sha256=sha(manifest_path))


def prepare():
    """CPU/network only; verify and stage immutable inputs before sole GPU job."""
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    frames, receipt = verify_inputs()
    archive = hf_hub_download(W_REPO, W_FILE, revision=W_REV, cache_dir=str(CACHE/'hf'))
    assert sha(archive) == W_SHA, 'W archive SHA mismatch'
    extracted = CACHE/'trained'
    if not extracted.exists():
        extracted.mkdir()
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                destination = (extracted/info.filename).resolve()
                if not destination.is_relative_to(extracted.resolve()):
                    raise ValueError('Unsafe archive member')
            z.extractall(extracted)
    configs = list(extracted.rglob('config.json'))
    assert len(configs) == 1, configs
    trained_dir = configs[0].parent
    allowed = ['config.json', 'generation_config.json', 'tokenizer*', 'special_tokens_map.json',
               'sentencepiece*', 'added_tokens.json', 'model.safetensors', 'pytorch_model.bin']
    # Avoid downloading both equivalent weight formats. Pin server LFS hashes as well.
    info = HfApi().model_info(MODEL, revision=REV, files_metadata=True)
    names = {s.rfilename for s in info.siblings}
    if 'model.safetensors' in names:
        allowed.remove('pytorch_model.bin')
    else:
        allowed.remove('model.safetensors')
    initial = snapshot_download(MODEL, revision=REV, allow_patterns=allowed, cache_dir=str(CACHE/'hf'))
    files = []
    for path in sorted(Path(initial).iterdir()):
        if path.is_file():
            digest = sha(path)
            meta = next((s for s in info.siblings if s.rfilename == path.name), None)
            lfs = getattr(meta, 'lfs', None)
            expected = getattr(lfs, 'sha256', None) if lfs else None
            if expected:
                assert digest == expected, path.name
            files.append(dict(path=path.name, sha256=digest, lfs_sha256=expected,
                              size_bytes=path.stat().st_size))
    receipt.update(dict(base_commit=PIN, source_sha256=sha(Path(__file__)),
                        trained_dir=str(trained_dir), initial_dir=str(initial),
                        w=dict(repo=W_REPO, revision=W_REV, file=W_FILE, archive_sha256=W_SHA),
                        pretrained=dict(repo=MODEL, revision=REV, files=files)))
    dump(OUT/'preparation.json', receipt)
    print('PREPARATION_PASS', flush=True)


def verify_tokenizer_cpu():
    """Avoid fast-tokenizer conversion: compare archived fast to pinned native SP.

    SentencePiece 0.2.3's generated protobuf is newer than the installed runtime;
    using the saved fast tokenizer avoids changing any installed package.
    """
    from transformers import AutoTokenizer
    prep = json.loads((OUT/'preparation.json').read_text())
    initial, trained = Path(prep['initial_dir']), Path(prep['trained_dir'])
    assert sha(initial/'sentencepiece.bpe.model') == sha(trained/'sentencepiece.bpe.model')
    fast = AutoTokenizer.from_pretrained(trained, local_files_only=True, use_fast=True)
    slow = AutoTokenizer.from_pretrained(initial, local_files_only=True, use_fast=False)
    fast.src_lang = slow.src_lang = 'tl_XX'
    fast.tgt_lang = slow.tgt_lang = 'zh_CN'
    assert fast.get_vocab() == slow.get_vocab()
    assert fast.all_special_ids == slow.all_special_ids
    frames, _ = verify_inputs()
    digests = {}
    for split, rows in frames.items():
        h = hashlib.sha256()
        for row in rows:
            for column in ('source', 'target'):
                kwargs = dict(max_length=256, truncation=True)
                if column == 'target':
                    kwargs['text_target'] = row[column]
                    a, b = fast(**kwargs)['input_ids'], slow(**kwargs)['input_ids']
                else:
                    a = fast(row[column], **kwargs)['input_ids']
                    b = slow(row[column], **kwargs)['input_ids']
                assert a == b, ('Fast/slow token IDs differ', split, column)
                h.update(json.dumps(a).encode())
        digests[split] = h.hexdigest()
    receipt = dict(vocab_match=True, special_ids_match=True, vocab_size=len(fast.get_vocab()),
                   source='Archive W fast tokenizer; exact vocab and source/target IDs vs pinned original slow tokenizer',
                   sentencepiece_sha256=sha(initial/'sentencepiece.bpe.model'),
                   tokenizer_json_sha256=sha(trained/'tokenizer.json'),
                   compared_rows={'train':4600, 'validation':576}, token_stream_sha256=digests,
                   environment_note='Avoided incompatible sentencepiece 0.2.3 / protobuf 6.33.6 fast conversion; no packages changed')
    dump(OUT/'tokenizer_verification.json', receipt)
    print('TOKENIZER_ALL_ROWS_EXACT_MATCH', flush=True)
    return receipt


def check_budget(start):
    if time.monotonic()-start > 1200:
        raise RuntimeError('20-minute GPU-job wall-time limit exceeded; no retry')
    peak = torch.cuda.max_memory_allocated()/2**30
    if peak >= 32 or torch.cuda.max_memory_reserved()/2**30 >= 32:
        raise RuntimeError('32 GiB GPU memory limit exceeded; no retry')


def load_model(path):
    from transformers import AutoModelForSeq2SeqLM
    model, info = AutoModelForSeq2SeqLM.from_pretrained(
        path, torch_dtype=torch.float32, attn_implementation='sdpa',
        local_files_only=True, output_loading_info=True)
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        assert not info.get(key), (key, info.get(key))
    assert model.config._attn_implementation == 'sdpa'
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # No layerdrop in pinned native recipe: seeded views replay exactly.
    assert model.config.encoder_layerdrop == model.config.decoder_layerdrop == 0
    model.config.use_cache = False
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    return model.cuda(), info


def batch_data(tokenizer, model, frame, ids):
    from transformers import DataCollatorForSeq2Seq
    examples = [tokenizer(frame[i]['source'], text_target=frame[i]['target'],
                          max_length=256, truncation=True) for i in ids]
    batch = DataCollatorForSeq2Seq(tokenizer, model=None, pad_to_multiple_of=8)(examples)
    batch = {k: v.cuda() for k, v in batch.items()}
    labels = batch.pop('labels')
    batch['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
    return batch, labels


def features(model, batch, seed, grad=False):
    set_seed(seed)
    with torch.set_grad_enabled(grad):
        return model.model(**batch, use_cache=False, return_dict=True).last_hidden_state


def head(model, h):
    return F.linear(h, model.lm_head.weight, model.final_logits_bias)


def cpu_gradient_geometry(gradients):
    keys = list(gradients)
    dots = {(a, b): 0. for i, a in enumerate(keys) for b in keys[i:]}
    for name in gradients[keys[0]]:
        # Float64 accumulation in parameter chunks, bounded RAM; no flatten concatenation.
        for offset in range(0, gradients[keys[0]][name].numel(), 1000000):
            pieces = {k: gradients[k][name].reshape(-1)[offset:offset+1000000].double() for k in keys}
            for a, b in dots:
                dots[a, b] += float(torch.dot(pieces[a], pieces[b]))
    norms = {k: math.sqrt(dots[k, k]) for k in keys}
    cosines = {f'{a}:{b}': value/(norms[a]*norms[b]) if norms[a]*norms[b] else None
               for (a, b), value in dots.items() if a != b}
    return norms, cosines


def cleanup_owned_cache():
    """Delete only the unique job's disposable weights after receipt verification."""
    removed = []
    for name in ('trained', 'hf', 'gradients'):
        path = CACHE/name
        if path.exists():
            assert path.resolve().is_relative_to(CACHE.resolve())
            shutil.rmtree(path)
            removed.append(str(path.relative_to(ROOT)))
    return removed


def token_measurements(z1, z2, labels, top_ids, random_teacher):
    """Return scalar measurements only; vocabulary-sized tensors stay on device."""
    a, b, p, q, valid, safe, pg, qg, winner = distributions(z1, z2, labels)
    grad = logit_gradients(z1, z2, labels, random_teacher)
    choose = lambda x, y: winner*x+(1-winner)*y
    gold_grad = lambda g: g.gather(-1, safe).squeeze(-1)
    common = grad['COMMON'][0]
    gw = choose(gold_grad(grad['W'][0]), gold_grad(grad['W'][1]))
    anti = (grad['W'][0]-grad['W'][1])/2
    anti_w = (2*winner-1)*gold_grad(anti)
    # dH/dz=-p*(log p + H); directional derivative along -common.
    h1, h2 = -(p*a).sum(-1), -(q*b).sum(-1)
    entropy_common = choose((p*(a+h1.unsqueeze(-1))*common).sum(-1),
                            (q*(b+h2.unsqueeze(-1))*common).sum(-1))
    freq_mass_p, freq_mass_q = p[:, top_ids].sum(-1), q[:, top_ids].sum(-1)
    is_top = (safe == top_ids.unsqueeze(0)).any(-1)
    freq_mass_p -= pg*is_top
    freq_mass_q -= qg*is_top
    # Entropy of the normalized NON-gold distribution, stable even near p_gold=1.
    non_a = a.clone().scatter_(-1, safe, -torch.inf)
    non_b = b.clone().scatter_(-1, safe, -torch.inf)
    ca = non_a - non_a.logsumexp(-1, keepdim=True)
    cb = non_b - non_b.logsumexp(-1, keepdim=True)
    ha = -(ca.exp()*ca.masked_fill(torch.isneginf(ca), 0)).sum(-1)
    hb = -(cb.exp()*cb.masked_fill(torch.isneginf(cb), 0)).sum(-1)
    values = {
        'common_lowers_gold': (gold_grad(common) > TOL).float(),
        'w_lowers_winner_gold': (gw > TOL).float(),
        'strict_case2': ((gold_grad(common) > TOL) & (anti_w > TOL)).float(),
        'common_lowers_winner_entropy': (entropy_common < -TOL).float(),
        'common_entropy_derivative': entropy_common,
        'prior_top100_winner_minus_loser': (2*winner-1)*(freq_mass_p-freq_mass_q),
        'nongold_entropy_winner_minus_loser': (2*winner-1)*(ha-hb),
    }
    for method in METHODS:
        g1, g2 = grad[method]
        d1, d2 = margin_direction(z1, g1, labels), margin_direction(z2, g2, labels)
        for view, value in [('winner', choose(d1, d2)), ('loser', choose(d2, d1)), ('mean', (d1+d2)/2)]:
            values[f'{method}:{view}'] = value
    values['case2_margin'] = (values['W:winner'] < -TOL).float()
    # Scale-free comparison divides by joint per-token two-view L2 norm.
    for method in ('W', 'GOC2.50', 'SYM2.50', 'RAND2.50', 'INV2.50'):
        g1, g2 = grad[method]
        norm = (g1.square().sum(-1)+g2.square().sum(-1)).sqrt().clamp_min(1e-20)
        values[f'{method}:normalized_mean'] = values[f'{method}:mean']/norm
    for control in ('W', 'SYM2.50', 'RAND2.50', 'INV2.50'):
        values[f'GOC_minus_{control}:mean'] = values['GOC2.50:mean']-values[f'{control}:mean']
        values[f'GOC_minus_{control}:normalized_mean'] = (
            values['GOC2.50:normalized_mean']-values[f'{control}:normalized_mean'])
    values['strong'] = (pg-qg).abs() > .05
    return {k: v.detach().cpu().numpy() for k, v in values.items()}


def summarize_part_a(records, metadata):
    """Sentence-cluster bootstrap retains all masks and target positions together."""
    groups = defaultdict(list)
    for r in records:
        for group in dict.fromkeys(('valid', 'special' if r['special'] else 'lexical', r['frequency_group'])):
            groups[r['split'], group, 'all'].append(r)
            if r['strong']:
                groups[r['split'], group, 'strong'].append(r)
    metrics = [k for k in records[0] if k not in
               ('split', 'row_id', 'position', 'pair', 'special', 'frequency_group', 'strong', 'token_id')]
    result = {}
    for (split, group, denominator), rows in sorted(groups.items()):
        key = f'{split}/{group}/{denominator}'
        ids = sorted({r['row_id'] for r in rows})
        id_index = {r: i for i, r in enumerate(ids)}
        values = np.array([[r[k] for k in metrics] for r in rows], dtype=np.float64)
        sums = np.zeros((len(ids), len(metrics)))
        counts = np.zeros(len(ids))
        for r, v in zip(rows, values):
            sums[id_index[r['row_id']]] += v
            counts[id_index[r['row_id']]] += 1
        rng = np.random.default_rng(BASE_SEED)
        weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=2000)
        boot = (weights @ sums)/(weights @ counts)[:, None]
        ci = np.quantile(boot, [.025, .975], axis=0)
        summary = {}
        for j, metric in enumerate(metrics):
            pair_means = [float(np.mean([r[metric] for r in rows if r['pair'] == pair]))
                          if any(r['pair'] == pair for r in rows) else None for pair in range(8)]
            finite_pairs = [v for v in pair_means if v is not None]
            summary[metric] = dict(mean=float(values[:, j].mean()), median=float(np.median(values[:, j])),
                                   positive_fraction=float((values[:, j] > TOL).mean()),
                                   ci95=[float(ci[0, j]), float(ci[1, j])],
                                   pair_means=pair_means,
                                   pair_mean_sd=float(np.std(finite_pairs, ddof=1)) if len(finite_pairs)>1 else 0)
        case_by_position = defaultdict(list)
        for r in rows:
            case_by_position[r['row_id'], r['position']].append(r['case2_margin'])
        histogram = Counter(sum(v) for v in case_by_position.values())
        result[key] = dict(observations=len(rows), sentences=len(ids), unique_positions=len(case_by_position),
                           case2_occurrence_histogram={str(int(k)): v for k, v in sorted(histogram.items())},
                           histogram_note='Strong group: number of harmful occurrences among qualifying masks; '
                                          'denominator varies by position. All group: 8 masks per position.',
                           metrics=summary)
    return dict(metadata=metadata, groups=result)


def part_a(model, tokenizer, frames, start):
    freq = Counter()
    for row in frames['train']:
        # Count the same truncated supervised target stream used in the recipe.
        freq.update(tokenizer(text_target=row['target'], max_length=256, truncation=True)['input_ids'])
    special = set(tokenizer.all_special_ids)
    top = [t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0])) if t not in special][:100]
    top_ids = torch.tensor(top, device='cuda')
    ids_by_split = {'train': list(range(64)), 'validation': [int(i*576/128) for i in range(128)]}
    records, replay = [], []
    for split_index, (split, ids) in enumerate(ids_by_split.items()):
        for offset in range(0, len(ids), 4):
            check_budget(start)
            batch_ids = ids[offset:offset+4]
            batch, labels = batch_data(tokenizer, model, frames[split], batch_ids)
            flat_y = labels.reshape(-1)
            ntime = labels.shape[1]
            for pair in range(8):
                seeds = seed_views(offset//4, pair, split_index)
                with torch.no_grad():
                    h1, h2 = (features(model, batch, s) for s in seeds)
                    if pair == 0:
                        hr = features(model, batch, seeds[0])
                        err = float((hr-h1).abs().max())
                        assert err < 1e-6, ('RNG replay failure', err)
                        del hr
                    else:
                        err = None
                    # CPU generator keeps teacher randomness independent of native dropout.
                    gen = torch.Generator().manual_seed(seeds[0]+5000000)
                    random_teacher = torch.randint(0, 2, flat_y.shape, generator=gen).cuda()
                    difference = 0.
                    for pos in range(0, flat_y.numel(), 16):
                        sl = slice(pos, pos+16)
                        z1 = head(model, h1.reshape(-1, h1.shape[-1])[sl])
                        z2 = head(model, h2.reshape(-1, h2.shape[-1])[sl])
                        difference = max(difference, float((z1-z2).abs().max()))
                        vals = token_measurements(z1, z2, flat_y[sl], top_ids, random_teacher[sl])
                        ys = flat_y[sl].cpu().tolist()
                        for j, y in enumerate(ys):
                            if y == -100:
                                continue
                            row = (pos+j)//ntime
                            group = ('special' if y in special else
                                     'rare' if freq[y] <= 10 else 'medium' if freq[y] <= 100 else 'frequent')
                            record = dict(split=split, row_id=batch_ids[row], position=(pos+j)%ntime,
                                          pair=pair, token_id=y, special=y in special, frequency_group=group)
                            record.update({k: bool(v[j]) if k == 'strong' else float(v[j]) for k, v in vals.items()})
                            assert all(math.isfinite(v) for v in record.values() if isinstance(v, float))
                            records.append(record)
                    assert difference > 0, 'Dropout views unexpectedly identical'
                    replay.append(dict(split=split, batch=offset//4, pair=pair, seeds=seeds,
                                       replay_hidden_max_error=err, view_logits_max_difference=difference))
                    del h1, h2, z1, z2
            print(f'A {split} {offset+len(batch_ids)}/{len(ids)} seconds={time.monotonic()-start:.1f}', flush=True)
    # Raw token scalar records are outside Git; no translation text is exported.
    raw = OUT/'token_measurements.jsonl'
    with raw.open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    metadata = dict(row_ids=ids_by_split, dropout_pairs=8, batch_size=4, pad_multiple=8,
                    top100_ids=top, target_frequency='truncated max256 supervised target token stream, train4600',
                    subword_frequency_sha256=hashlib.sha256(json.dumps(sorted(freq.items())).encode()).hexdigest(),
                    tolerance=TOL, bootstrap=dict(draws=2000, seed=BASE_SEED, unit='sentence cluster; all masks retained'),
                    replay=replay, raw_file=str(raw.relative_to(ROOT)), raw_sha256=sha(raw),
                    zero_frequency_lexical_observations=sum(1 for r in records if not r['special'] and freq[r['token_id']]==0))
    # Bootstrap is CPU work after CUDA phase A; allow its own timing to appear in wall time.
    result = summarize_part_a(records, metadata)
    dump(OUT/'part_a.json', result)
    return result


def head_backbone_gradient(model, batch, labels, seeds, method, start):
    """Exact head/backbone VJP with seeded recomputation, including tied embedding.

    No detached gradient is mistaken for a new scalar loss. At fixed weights,
    dL/dtheta = sum_v [head VJP + backbone VJP] for both native views.
    """
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        h = [features(model, batch, seed) for seed in seeds]
    dh = [torch.zeros_like(value) for value in h]
    y = labels.reshape(-1)
    count = labels.ne(-100).sum()
    for offset in range(0, y.numel(), 16):
        check_budget(start)
        sl = slice(offset, offset+16)
        with torch.no_grad():
            z = [head(model, value.reshape(-1, value.shape[-1])[sl]) for value in h]
            g = logit_gradients(*z, y[sl])[method]
        for view in (0, 1):
            leaf = h[view].reshape(-1, h[view].shape[-1])[sl].detach().requires_grad_(True)
            logits = head(model, leaf)
            logits.backward(g[view]/count)
            dh[view].reshape(-1, dh[view].shape[-1])[sl] = leaf.grad
            del logits, leaf
        del z, g
    # Accumulate encoder/decoder derivatives on top of head gradients; shared
    # lm_head/model.shared.weight is a single Parameter and receives both paths.
    errors = []
    for view, seed in enumerate(seeds):
        replay = features(model, batch, seed, grad=True)
        err = float((replay.detach()-h[view]).abs().max())
        assert err < 1e-6, ('Backbone replay mismatch', err)
        errors.append(err)
        replay.backward(dh[view])
        del replay
    result = {}
    for name, param in model.named_parameters():
        # Native layers may be unused only if no-grad path was accidentally cut.
        assert param.grad is not None, ('Missing parameter gradient', name)
        assert torch.isfinite(param.grad).all(), ('Nonfinite gradient', name)
        result[name] = param.grad.detach().cpu()
    model.zero_grad(set_to_none=True)
    return result, errors


def part_b(model, tokenizer, frames, checkpoint_name, start):
    rows = []
    for size in (4, 32):
        # Resolve 3x32 vs 64-row ambiguity without touching validation: use first
        # 96 TRAIN rows for B32; B4 uses 0:4,4:8,8:12. Explicitly record all IDs.
        for batch_idx in range(3):
            ids = list(range(batch_idx*size, (batch_idx+1)*size))
            batch, labels = batch_data(tokenizer, model, frames['train'], ids)
            seeds = seed_views(batch_idx, 0, 10+size)
            gradients, errors = {}, {}
            for method in B_METHODS:
                gradients[method], errors[method] = head_backbone_gradient(model, batch, labels, seeds, method, start)
                print(f'B {checkpoint_name} B{size} batch{batch_idx} {method} '
                      f'seconds={time.monotonic()-start:.1f}', flush=True)
            norms, cosines = cpu_gradient_geometry(gradients)
            norms['GOC2.50'] = 2*norms['GOC1.25']
            for key, value in list(cosines.items()):
                if 'GOC1.25' in key:
                    cosines[key.replace('GOC1.25', 'GOC2.50')] = value
            rows.append(dict(checkpoint=checkpoint_name, batch_size=size, batch_index=batch_idx,
                             row_ids=ids, seeds=seeds, valid_tokens=int(labels.ne(-100).sum()),
                             source_padded_length=batch['input_ids'].shape[1], target_padded_length=labels.shape[1],
                             norms=norms, cosines=cosines, replay_hidden_errors=errors,
                             goc_over_w={beta: norms[f'GOC{beta}']/norms['W'] for beta in ('1.25', '2.50')},
                             goc_over_ce={beta: norms[f'GOC{beta}']/norms['CE'] for beta in ('1.25', '2.50')},
                             goc2_derivation='Exact 2x scalar multiple of GOC1.25 gradient on identical views'))
            dump(OUT/f'part_b_{checkpoint_name}.json', rows)
            del gradients, batch, labels
            gc.collect()
            check_budget(start)
    return rows


def decision(part_a_results):
    group = part_a_results['groups'].get('validation/rare/strong')
    if not group:
        return dict(verdict='HOLD', reason='No rare strong-disagreement observations')
    m = group['metrics']
    conditions = {
        'case2_margin_at_least_5_percent': m['case2_margin']['mean'] >= .05,
        'goc_mean_margin_beats_w': m['GOC_minus_W:mean']['mean'] > TOL,
        'goc_mean_margin_beats_sym': m['GOC_minus_SYM2.50:mean']['mean'] > TOL,
        'goc_mean_margin_beats_inv': m['GOC_minus_INV2.50:mean']['mean'] > TOL,
        'winner_does_not_increase_top100_prior': m['prior_top100_winner_minus_loser']['mean'] <= TOL,
        'goc_margin_gain_vs_sym_repeats_all_8': all(v is not None and v > TOL
                                                   for v in m['GOC_minus_SYM2.50:mean']['pair_means']),
    }
    # Plan has no numeric tail-entropy harm threshold, gradient-norm reasonable
    # interval, or minimum effect size: never silently invent a GO admission.
    failed = [key for key, passed in conditions.items() if not passed]
    return dict(verdict='STOP' if failed else 'HOLD', conditions=conditions, failed_conditions=failed,
                note='GO not automatic: plan leaves effect-size, tail-risk and norm interval qualitative; '
                     'report CIs and per-mask values for local review. No pilot authorized.')


def run():
    import platform
    import transformers
    import accelerate
    from transformers import AutoTokenizer
    if (OUT/'GPU_JOB_STARTED.json').exists():
        raise RuntimeError('The sole authorized GPU job has already started; do not retry automatically')
    frames, verification = verify_inputs()
    prep = json.loads((OUT/'preparation.json').read_text())
    assert prep['w']['archive_sha256'] == W_SHA
    # Reverify actual archive immediately before GPU work, not only preparation receipt.
    archives = list((CACHE/'hf').rglob('best.zip'))
    assert len(archives) == 1 and sha(archives[0]) == W_SHA
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Fix a reproducible SDPA kernel family for no-grad/grad seeded replay.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    assert 'A100' in props.name and 39 <= props.total_memory/2**30 <= 41
    free, total = torch.cuda.mem_get_info()
    assert free/2**30 >= 35, 'GPU busy; do not start diagnostic'
    torch.cuda.set_per_process_memory_fraction(31/40)
    torch.cuda.reset_peak_memory_stats()
    receipt = dict(status='RUNNING', optimizer_steps=0, generation_calls=0, test_split_opened=False,
                   started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   base_commit=PIN, source_sha256=sha(Path(__file__)), inputs=verification,
                   preparation_sha256=sha(OUT/'preparation.json'),
                   hardware=dict(gpu=props.name, memory_bytes=props.total_memory,
                                 driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()),
                   runtime=dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                                accelerate=accelerate.__version__, dtype='float32', tf32=False, attention='sdpa',
                                sdpa_backends='PyTorch SDPA MATH only; flash/mem-efficient off; deterministic algorithms on'),
                   scope='Frozen RTX-trained W42 and pinned pretrained measured on A100; not A100 training cohort')
    dump(OUT/'GPU_JOB_STARTED.json', receipt)
    model = None
    try:
        token_receipt = json.loads((OUT/'tokenizer_verification.json').read_text())
        assert token_receipt['vocab_match'] and token_receipt['special_ids_match']
        assert sha(Path(prep['trained_dir'])/'tokenizer.json') == token_receipt['tokenizer_json_sha256']
        receipt['tokenizer_verification'] = token_receipt
        tokenizer = AutoTokenizer.from_pretrained(prep['trained_dir'], local_files_only=True, use_fast=True)
        tokenizer.src_lang, tokenizer.tgt_lang = 'tl_XX', 'zh_CN'
        receipt['tokenizer'] = dict(source=MODEL, revision=REV, src_lang='tl_XX', tgt_lang='zh_CN',
                                   special_ids=tokenizer.all_special_ids)
        model, loading = load_model(prep['trained_dir'])
        before = parameter_hash(model)
        receipt['trained'] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                  state_sha256_before=before, strict_loading=loading)
        a = part_a(model, tokenizer, frames, start)
        trained_b = part_b(model, tokenizer, frames, 'W42', start)
        after = parameter_hash(model)
        assert before == after
        receipt['trained']['state_sha256_after'] = after
        receipt['trained']['unchanged'] = True
        model.zero_grad(set_to_none=True)
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        model, loading = load_model(prep['initial_dir'])
        assert sum(p.numel() for p in model.parameters()) == receipt['trained']['parameters']
        before = parameter_hash(model)
        receipt['initial'] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                  state_sha256_before=before, strict_loading=loading)
        initial_b = part_b(model, tokenizer, frames, 'initial', start)
        after = parameter_hash(model)
        assert before == after
        receipt['initial'].update(state_sha256_after=after, unchanged=True)
        check_budget(start)
        receipt.update(status='COMPLETE', part_a=a, part_b=trained_b+initial_b, decision=decision(a))
    except Exception as exc:
        import traceback
        receipt.update(status='FAILED_NO_RETRY', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
            del model
        gc.collect()
        torch.cuda.synchronize()
        receipt.update(wall_seconds=time.monotonic()-start,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        dump(OUT/'results.json', receipt)
        assert json.loads((OUT/'results.json').read_text())['status'] == receipt['status']
        if receipt['status'] == 'COMPLETE':
            receipt['temporary_weights_cleanup'] = cleanup_owned_cache()
            dump(OUT/'results.json', receipt)
        torch.cuda.empty_cache()
        print('GPU_JOB_END', receipt['status'], receipt.get('decision'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='CPU-only verified download')
    parser.add_argument('--run', action='store_true', help='One GPU diagnostic, guarded against retries')
    parser.add_argument('--verify-tokenizer', action='store_true', help='CPU-only exact fast/slow ID audit')
    args = parser.parse_args()
    if sum((args.prepare, args.run, args.verify_tokenizer)) != 1:
        parser.error('Choose exactly one of --prepare, --verify-tokenizer, --run')
    (prepare if args.prepare else verify_tokenizer_cpu if args.verify_tokenizer else run)()


if __name__ == '__main__':
    main()
