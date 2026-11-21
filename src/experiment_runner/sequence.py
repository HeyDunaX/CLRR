"""Pinned initialization and one shared train-only candidate pool for MRT controls."""
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sacrebleu.metrics import CHRF

from .config import ROOT
from .integrity import sha256, restore_archive
from methods.sequence_risk.v1.data import normalize_pool_candidates


def initialization_folder():
    from . import engine as e
    selected = e.plan()['initialization']
    root = ROOT / 'cache/initializations' / selected['archive_sha256']
    folder = root / 'export'
    marker = root / 'RESTORED.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['selection'] == selected
        for name, digest in saved['files'].items():
            assert sha256(folder / name) == digest, ('Initialization mutated', name)
        return folder
    archive = Path(hf_hub_download(selected['repo'], selected['archive_path'],
        revision=selected['revision'], token=os.environ.get('HF_TOKEN'), local_dir=root / 'download'))
    assert sha256(archive) == selected['archive_sha256']
    restore_archive(archive, folder)
    files = {p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob('*') if p.is_file()}
    assert any(name in files for name in ('pytorch_model.bin', 'model.safetensors'))
    e.write(marker, dict(selection=selected, files=files))
    return folder


def _pool_lock():
    from . import engine as e
    p = e.plan()
    return dict(initialization=p['initialization'], train=p['data']['train'],
        tokenizer_source='same pinned initialization export', candidate_pool=p['candidate_pool'],
        source_sha256=sha256(Path(__file__)), bundle_sha256=sha256(e.OUT / 'bundle_manifest.json'),
        src_lang=p['runs'][0]['src_lang'], tgt_lang=p['runs'][0]['tgt_lang'],
        max_source_length=p['runs'][0]['max_source_length'], max_target_length=p['runs'][0]['max_target_length'])


def read_pool(tokenizer, spec):
    from . import engine as e
    receipt = json.loads((e.OUT / 'POOL_COMPLETE.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['lock'] == _pool_lock()
    path = e.OUT / 'candidate_pool.jsonl'
    assert sha256(path) == receipt['pool_sha256']
    records = [json.loads(line) for line in path.read_text().splitlines()]
    raw = pd.read_csv(e.DATA / 'train.csv', dtype=str, keep_default_na=False)
    assert len(records) == len(raw) == receipt['rows']
    gold = tokenizer(text_target=raw.target.tolist(), truncation=True,
        max_length=spec['max_target_length'])['input_ids']
    for row, record in enumerate(records):
        assert record['row'] == row and record['labels'][0] == gold[row]
        assert len(record['labels']) == len({tuple(x) for x in record['labels']})
        assert len(record['labels']) == len(record['costs'])
        assert all(np.isfinite(c) and 0 <= c <= 1 for c in record['costs'])
    assert_pool_signal(records, **_signal_limits())
    return records


def _signal_limits():
    from . import engine as e
    cfg = e.plan()['candidate_pool']
    return dict(max_no_eos_fraction=cfg.get('max_no_eos_fraction'),
        max_length_cap_fraction=cfg.get('max_length_cap_fraction'))


def assert_pool_signal(records, *, max_no_eos_fraction=None, max_length_cap_fraction=None):
    assert records, 'Empty pool'
    assert sum(len(r['labels']) == 1 for r in records) < .5 * len(records), 'Degenerate singleton pool; do not train'
    assert sum(max(r['costs']) - min(r['costs']) < 1e-8 for r in records) < .5 * len(records), 'Degenerate equal-cost pool; do not train'
    if max_no_eos_fraction is not None or max_length_cap_fraction is not None:
        assert all('sampling_stats' in r for r in records), 'Sampling diagnostics required'
        stats = aggregate_sampling_stats(records)
        assert stats['sample_draws'] > 0
        if max_no_eos_fraction is not None:
            assert stats['no_eos']/stats['sample_draws'] < max_no_eos_fraction, 'Too many incomplete samples'
        if max_length_cap_fraction is not None:
            assert stats['length_cap']/stats['sample_draws'] < max_length_cap_fraction, 'Too many length-capped samples'


def aggregate_sampling_stats(records):
    import collections
    total = collections.Counter()
    for record in records:
        s = record['sampling_stats']
        assert all(isinstance(v, int) and v >= 0 for v in s.values())
        assert s['no_eos'] + s['gold_duplicate_draws'] + s['non_gold_duplicate_draws'] + s['new_non_gold_draws'] == s['sample_draws']
        assert s['new_non_gold_draws'] == len(record['labels']) - 1
        assert s['length_cap'] <= s['sample_draws']
        total.update(s)
    return dict(total)


def prepare_pool(spec):
    from . import engine as e
    e.validate_environment()
    p = e.plan(); cfg = p['candidate_pool']; lock = _pool_lock()
    assert cfg['split'] == 'train' and cfg['include_gold'] and not spec['risk_weight']
    assert spec['method'] == 'sequence-ce-v1', 'Pool generator is the native baseline'
    complete = e.OUT / 'POOL_COMPLETE.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        assert saved['status'] == 'COMPLETE' and saved['lock'] == lock and sha256(e.OUT / 'candidate_pool.jsonl') == saved['pool_sha256']
        records = [json.loads(line) for line in (e.OUT / 'candidate_pool.jsonl').read_text().splitlines()]
        assert_pool_signal(records, **_signal_limits())
        print('POOL_ALREADY_VERIFIED', saved['rows'], flush=True)
        return
    marker = e.OUT / 'POOL_LOCK.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == lock
    else:
        e.write(marker, lock)
    partial = e.OUT / 'candidate_pool.jsonl.partial'
    records = [json.loads(line) for line in partial.read_text().splitlines()] if partial.exists() else []
    assert all(r['row'] == i for i, r in enumerate(records))
    raw = pd.read_csv(e.DATA / 'train.csv', dtype=str, keep_default_na=False)
    assert len(records) <= len(raw)
    model, tokenizer, _ = e.load_model(spec)
    assert all(p.dtype == torch.float32 for p in model.parameters()), 'Pool scorer must be FP32'
    assert not torch.backends.cuda.matmul.allow_tf32, 'Pool TF32 must be disabled'
    model.to('cuda'); model.eval()
    gold = tokenizer(text_target=raw.target.tolist(), truncation=True,
        max_length=spec['max_target_length'])['input_ids']
    for i, record in enumerate(records):
        assert record['labels'][0] == gold[i]
    metric = CHRF(word_order=0)
    generated_dropped = 0
    started = time.time()
    batch_size = cfg['source_batch']
    # Batch-specific seed makes a completed batch reproducible after interruption.
    first = len(records) // batch_size * batch_size
    with partial.open('a', encoding='utf-8') as stream:
        for start in range(first, len(raw), batch_size):
            texts = raw.source.iloc[start:start + batch_size].tolist()
            encoded = tokenizer([spec['source_prefix'] + text for text in texts], padding=True,
                truncation=True, max_length=spec['max_source_length'], return_tensors='pt').to('cuda')
            torch.manual_seed(cfg['seed'] + start)
            with torch.no_grad():
                generated = model.generate(**encoded, do_sample=True, num_beams=1,
                    num_return_sequences=cfg['samples'], max_length=spec['max_target_length'],
                    temperature=cfg['temperature'], top_k=0, top_p=1., use_cache=True)
            generated = generated.cpu().tolist()
            for offset in range(len(texts)):
                row = start + offset
                if row < len(records):
                    continue
                choices, stats = normalize_pool_candidates(gold[row],
                    generated[offset * cfg['samples']:(offset + 1) * cfg['samples']],
                    decoder_start=model.config.decoder_start_token_id,
                    target_bos=model.config.forced_bos_token_id, eos=tokenizer.eos_token_id,
                    max_length=spec['max_target_length'])
                generated_dropped += stats['no_eos']
                costs = [max(0., min(1., 1. - metric.sentence_score(
                    tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                    [raw.target.iloc[row].strip()]).score / 100.)) for tokens in choices]
                record = dict(row=row, labels=choices, costs=costs, sampling_stats=stats)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush(); os.fsync(stream.fileno())
                records.append(record)
            if start % (batch_size * 10) == 0 or len(records) == len(raw):
                state = dict(status='GENERATING_TRAIN_POOL', rows=len(records), total=len(raw),
                    elapsed_seconds=time.time() - started, updated_unix=time.time())
                e.write(e.OUT / 'POOL_PROGRESS.json', state)
                print('POOL_PROGRESS', json.dumps(state), flush=True)
    destination = e.OUT / 'candidate_pool.jsonl'
    assert not destination.exists()
    try:
        assert_pool_signal(records, **_signal_limits())
    except AssertionError as error:
        e.write(e.OUT / 'POOL_REJECTED.json', dict(status='STOPPED_BY_POOL_GATE',
            reason=str(error), rows=len(records), singleton_rows=sum(len(r['labels']) == 1 for r in records),
            equal_cost_rows=sum(max(r['costs']) - min(r['costs']) < 1e-8 for r in records),
            main_training_models_started=0, recipe_changed=False, test_evaluated=False))
        raise
    partial.replace(destination)
    counts = [len(r['labels']) for r in records]
    summary = dict(status='COMPLETE', lock=lock, rows=len(raw), pool_sha256=sha256(destination),
        mean_candidates=float(np.mean(counts)), min_candidates=min(counts), max_candidates=max(counts),
        singleton_rows=sum(c == 1 for c in counts),
        equal_cost_rows=sum(max(r['costs']) - min(r['costs']) < 1e-8 for r in records),
        dropped_no_eos_this_attempt=generated_dropped, seconds_this_attempt=time.time() - started,
        gold_cost_mean=float(np.mean([r['costs'][0] for r in records])),
        sampled_candidates_at_length_limit=sum(len(c) >= spec['max_target_length'] - 1
            for record in records for c in record['labels'][1:]),
        sampling_stats=aggregate_sampling_stats(records),
        dtype='float32', tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        train_only=True, test_evaluated=False, optimizer_steps=0)
    e.write(complete, summary)
    del model
    gc.collect(); torch.cuda.empty_cache()
    print('POOL_COMPLETE', json.dumps(summary), flush=True)


def preflight(spec):
    """One disposable full-geometry optimizer step; no dataset or retained weights."""
    from . import engine as e
    e.validate_environment()
    assert spec['method'] == 'sequence-clrr-risk-v1'
    destination = e.OUT / 'PREFLIGHT.json'
    assert not destination.exists(), 'Preserve the existing full GPU gate'
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    before_free = torch.cuda.mem_get_info()[0] // 2**20
    started = time.time()
    model, tokenizer, _ = e.load_model(spec)
    model.to('cuda'); model.train(); model.gradient_checkpointing_enable()
    batch, length = spec['microbatch'], spec['max_target_length']
    ids = torch.randint(10, model.config.vocab_size - 100, (batch, spec['max_source_length']), device='cuda')
    labels = torch.randint(10, model.config.vocab_size - 100, (batch, length), device='cuda')
    labels[:, 0] = model.config.forced_bos_token_id
    labels[:, -1] = tokenizer.eos_token_id
    n = e.plan()['candidate_pool']['samples'] + 1
    candidates = labels[:, None, :].repeat(1, n, 1)
    for k in range(n):
        candidates[:, k, 1] = (candidates[:, k, 1] + k) % model.config.vocab_size
    opt = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'],
        betas=(e.protocol()['adam_beta1'], e.protocol()['adam_beta2']),
        eps=e.protocol()['adam_epsilon'], weight_decay=e.protocol()['weight_decay'], fused=True)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
        candidate_labels=candidates, candidate_costs=torch.linspace(0, 1, n, device='cuda').repeat(batch, 1),
        candidate_valid=torch.ones(batch, n, dtype=torch.bool, device='cuda'))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), e.protocol()['max_grad_norm'])
    assert torch.isfinite(norm) and norm > 0
    opt.step()
    torch.cuda.synchronize()
    receipt = dict(passed=True, scope='disposable synthetic full-shape AdamW optimizer smoke',
        gpu=torch.cuda.get_device_name(0), before_free_mib=before_free,
        max_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
        max_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
        microbatch=batch, length=length, candidates=n, score_chunk=spec['score_chunk'],
        loss=output.loss.item(), diagnostics=model.last_diagnostics, grad_norm=norm.item(),
        optimizer_steps=1, dataset_rows_used=0, seconds=time.time() - started,
        bundle_sha256=sha256(e.OUT / 'bundle_manifest.json'))
    del model, tokenizer, opt, output, ids, labels, candidates
    gc.collect(); torch.cuda.empty_cache()
    e.write(destination, receipt)
    print('SEQUENCE_FULL_PREFLIGHT_PASS', json.dumps(receipt), flush=True)


def summarize(spec=None):
    from . import engine as e
    values = {}
    for run in e.plan()['runs']:
        r = json.loads((e.OUT / 'runs' / run['id'] / 'VAL_COMPLETE.json').read_text())
        assert r['status'] == 'VALIDATION_COMPLETE_VERIFIED' and r['backup']['verified']
        values[run['id']] = r['scores']
    assert set(values) == {'S0', 'S1', 'S2', 'S3'}
    contrasts = {}
    for metric in ('bleu', 'chrf++', 'chrf0'):
        v = {k: values[k][metric] for k in values}
        contrasts[metric] = {'S3-S1': v['S3'] - v['S1'], 'S3-S0': v['S3'] - v['S0'],
            'S1-S0': v['S1'] - v['S0'], 'S2-S0': v['S2'] - v['S0'],
            'S3-S2': v['S3'] - v['S2'], 'interaction': v['S3'] - v['S2'] - v['S1'] + v['S0']}
    go = (contrasts['bleu']['S3-S1'] >= .3 and contrasts['chrf0']['S3-S1'] >= 0
        and contrasts['bleu']['S3-S0'] >= .3 and contrasts['bleu']['interaction'] > 0)
    summary = dict(status='COMPLETE', scores=values, contrasts=contrasts,
        development_gate='GO_TO_MULTISEED_DESIGN' if go else 'NO_GO_CURRENT_RECIPE',
        no_confirmatory_p_value=True, seed=42, test_evaluated=False,
        known_objective_not_new_novelty=True, gates=e.plan()['decision_gates'])
    e.write(e.OUT / 'PILOT_SUMMARY.json', summary)
    print('PILOT_SUMMARY', json.dumps(summary), flush=True)


def cleanup_initialization(spec=None):
    """Only after every declared run has a verified backup; never touch HF history."""
    from . import engine as e
    for spec in e.plan()['runs']:
        receipt = json.loads((e.OUT / 'runs' / spec['id'] / 'collection_receipt.json').read_text())
        assert receipt['verified'] and receipt['files']
    target = (ROOT / 'cache/initializations' / e.plan()['initialization']['archive_sha256']).resolve()
    assert target.is_relative_to((ROOT / 'cache/initializations').resolve())
    shutil.rmtree(target)
