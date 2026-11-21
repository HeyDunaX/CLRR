"""Frozen same-prefix dropout decisions and reference-relative greedy rollouts.

Diagnostic only: no optimizer, test access, or training admission. Costs are
single-reference proxies, never labels of semantic equivalence or adequacy.
"""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
from experiment_runner.integrity import sha256, restore_archive


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def first_decision(a, b, decoder, special):
    """Logit index t predicts AFTER decoder[:t+1], not decoder[:t]."""
    left, right = a.argmax(-1), b.argmax(-1)
    for t in range(1, decoder.shape[1]):
        x, y = int(left[0, t]), int(right[0, t])
        if x != y and x not in special and y not in special:
            return t, x, y
    return None


def branch_decode(model, source, prefix, action, total_limit):
    forced = torch.cat([prefix, prefix.new_tensor([[action]])], dim=1)
    if forced.shape[1] >= total_limit:
        return forced
    result = model.generate(**source, decoder_input_ids=forced, num_beams=1,
        do_sample=False, max_length=total_limit, use_cache=True)
    assert torch.equal(result[:, :forced.shape[1]], forced), 'Forced prefix changed'
    return result


def cpu_gate():
    from transformers import MBartConfig, MBartForConditionalGeneration
    config = MBartConfig(vocab_size=19, d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16,
        decoder_ffn_dim=16, max_position_embeddings=32, pad_token_id=1,
        bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
        forced_bos_token_id=5, dropout=.3, attention_dropout=.1)
    torch.manual_seed(4)
    model = MBartForConditionalGeneration(config).eval()
    source = dict(input_ids=torch.tensor([[3, 4, 2]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    with torch.inference_mode():
        a = model.generate(**source, max_length=9, num_beams=1, do_sample=False)
        b = model.generate(**source, max_length=9, num_beams=1, do_sample=False)
        assert torch.equal(a, b) and a[0, 1] == 5
        branch = branch_decode(model, source, torch.tensor([[2, 5]]), 7, 9)
        assert branch[0, :3].tolist() == [2, 5, 7]
        model.train()
        torch.manual_seed(11)
        p = model(**source, decoder_input_ids=torch.tensor([[2, 5, 7]]), use_cache=False).logits
        torch.manual_seed(11)
        q = model(**source, decoder_input_ids=torch.tensor([[2, 5, 7]]), use_cache=False).logits
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    x = torch.zeros(1, 3, 19); y = x.clone()
    x[0, 1, 7] = 3; y[0, 1, 8] = 3
    assert first_decision(x, y, torch.tensor([[2, 5, 9]]), {0, 1, 2, 5}) == (1, 7, 8)
    assert first_decision(x, x, torch.tensor([[2, 5, 9]]), {0, 1, 2, 5}) is None
    assert all(p.grad is None for p in model.parameters())
    return dict(passed=True, optimizer_steps=0, checks=['deterministic_generation',
        'target_language_token', 'prefix_action_preserved', 'logit_index_alignment',
        'no_disagreement_not_zero_regret', 'dropout_rng_replay', 'no_gradients'])


def run(args):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from sacrebleu.metrics import BLEU, CHRF
    out = Path(args.output).resolve(); plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda')
    assert out.is_relative_to(ROOT / 'outputs/experiments')
    assert plan['optimizer_steps'] == 0 and plan['split'] == 'validation'
    assert plan['metric_family'] == 'decision_branches_v1'
    for key, version in plan['runtime'].items():
        if key != 'python': assert importlib.metadata.version(key).split('+')[0] == version
    for name, digest in plan['source_sha256'].items(): assert sha256(ROOT / name) == digest
    data_path = ROOT / 'data_processed' / plan['dataset'] / 'validation.csv'
    assert sha256(data_path) == plan['data']['validation']['sha256']
    raw = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    assert len(raw) == plan['data']['validation']['rows']
    rows = plan['validation_rows']; assert len(set(rows)) == len(rows)
    write(out / 'CPU_GATE.json', cpu_gate())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    metrics = {'chrf++': CHRF(word_order=2), 'chrf0': CHRF(word_order=0),
        'bleu': BLEU(tokenize='zh', effective_order=True)}
    for checkpoint in plan['checkpoints']:
        ident = checkpoint['id']; folder = out / ident; folder.mkdir(exist_ok=True)
        if (folder / 'RESULT.json').exists():
            saved = json.loads((folder / 'RESULT.json').read_text())
            assert saved['plan_sha256'] == sha256(plan_path)
            assert saved['csv_sha256'] == sha256(folder / 'decisions.csv')
            continue
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free',
            '--format=csv,noheader,nounits'], text=True).strip()
        assert gpu.startswith('Quadro RTX 6000,') and int(gpu.rsplit(',', 1)[1]) >= plan['minimum_free_gpu_mib'], gpu
        assert shutil.disk_usage(out).free >= 40 * 2**30
        write(out / 'RESOURCE_ADMISSION.json', dict(id=ident, gpu=gpu, optimizer_steps=0))
        temporary = folder / 'restore'; temporary.mkdir(exist_ok=False)
        archive = Path(hf_hub_download(plan['repo'], checkpoint['prefix'].rstrip('/') + '/best.zip',
            revision=checkpoint['revision'], token=False, local_dir=temporary / 'download'))
        assert sha256(archive) == checkpoint['archive_sha256']
        export = temporary / 'export'; restore_archive(archive, export)
        spec = checkpoint['spec']
        assert spec['method'] in ('baseline', 'weak-full-consistency-v1')
        assert spec['strength'] == 0 and spec['jepa_weight'] == 0
        if spec['method'] != 'baseline':
            saved = json.loads((export / 'decoupled_consistency_config.json').read_text())
            assert saved['method'] == spec['method'] and saved['beta'] == spec['consistency_weight']
            assert saved['inference_route'] == 'native'
        model, info = AutoModelForSeq2SeqLM.from_pretrained(export, attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
        model.gradient_checkpointing_disable(); model.cuda().eval()
        assert model.config._attn_implementation == 'sdpa' and next(model.parameters()).dtype == torch.float32
        tokenizer = AutoTokenizer.from_pretrained(export)
        tokenizer.src_lang = spec['src_lang']; tokenizer.tgt_lang = spec['tgt_lang']
        target_id = tokenizer.convert_tokens_to_ids(spec['tgt_lang'])
        assert target_id != tokenizer.unk_token_id
        model.generation_config.forced_bos_token_id = target_id
        model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
        special = set(tokenizer.all_special_ids); records = []; start_time = time.time()
        total_limit = spec['max_target_length'] + 1
        with torch.inference_mode():
            for row in rows:
                item = raw.iloc[row]
                source = tokenizer(item.source, truncation=True, max_length=spec['max_source_length'], return_tensors='pt')
                source = {k: v.cuda() for k, v in source.items() if k in ('input_ids', 'attention_mask')}
                model.eval()
                native = model.generate(**source, num_beams=1, do_sample=False, max_length=total_limit, use_cache=True)
                assert native[0, 0] == tokenizer.eos_token_id and native[0, 1] == target_id
                assert native.shape[1] <= total_limit
                decoder = native[:, :-1]
                native_text = tokenizer.decode(native[0], skip_special_tokens=True)
                for pair, seed in enumerate(plan['dropout_seeds']):
                    model.train()
                    views = []
                    for view in range(2):
                        rng = seed + row * 1009 + view * 100000
                        torch.manual_seed(rng); torch.cuda.manual_seed_all(rng)
                        views.append(model(**source, decoder_input_ids=decoder, use_cache=False).logits)
                    a, b = views
                    event = first_decision(a, b, decoder, special)
                    record = dict(row=row, pair=pair, dropout_seed=seed, source=item.source,
                        target=item.target, native=native_text, generated_tokens=int(native.shape[1]-1),
                        source_characters=len(item.source), target_characters=len(item.target),
                        has_lexical_disagreement=event is not None)
                    if event is not None:
                        t, left, right = event
                        lp, lq = a[0, t].float().log_softmax(-1), b[0, t].float().log_softmax(-1)
                        probability = .5 * (lp.exp() + lq.exp())
                        logmean = probability.clamp_min(torch.finfo(torch.float32).tiny).log()
                        record.update(position=t, prefix_length=t+1, action_a=left, action_b=right,
                            entropy=float(-(probability * logmean).sum()),
                            symmetric_kl=float(.5 * ((lp.exp()-lq.exp())*(lp-lq)).sum()),
                            js=float(.5*((lp.exp()*(lp-logmean)).sum()+(lq.exp()*(lq-logmean)).sum())),
                            mean_top1_probability=float(probability.max()))
                        model.eval(); prefix = decoder[:, :t+1]
                        for label, action in [('a', left), ('b', right)]:
                            result = branch_decode(model, source, prefix, action, total_limit)
                            text = tokenizer.decode(result[0], skip_special_tokens=True)
                            record['branch_'+label] = text
                            record['branch_'+label+'_at_length_limit'] = bool(result.shape[1] >= total_limit)
                            for metric, scorer in metrics.items():
                                record[label+'_'+metric] = scorer.sentence_score(text, [item.target]).score
                        for metric in metrics:
                            record['abs_gap_'+metric] = abs(record['a_'+metric]-record['b_'+metric])
                        record['same_decoded_branches'] = record['branch_a'] == record['branch_b']
                    if row == rows[0] and pair == 0:
                        model.train(); torch.manual_seed(seed+row*1009); torch.cuda.manual_seed_all(seed+row*1009)
                        replay = model(**source, decoder_input_ids=decoder, use_cache=False).logits
                        torch.testing.assert_close(replay, a, rtol=0, atol=0)
                        assert not torch.equal(a, b) and all(p.grad is None for p in model.parameters())
                        model.eval()
                        native_replay = model.generate(**source, num_beams=1, do_sample=False, max_length=total_limit, use_cache=True)
                        assert torch.equal(native_replay, native)
                        if event is not None:
                            replay_branch = branch_decode(model, source, decoder[:, :t+1], left, total_limit)
                            assert tokenizer.decode(replay_branch[0], skip_special_tokens=True) == record['branch_a']
                        write(folder / 'GPU_GATE.json', dict(passed=True, strict_loading=True,
                            rng_replay=True, native_replay=True, branch_replay=event is not None,
                            no_gradients=True, optimizer_steps=0))
                        del replay
                    records.append(record)
                    del views, a, b
                pd.DataFrame(records).to_csv(folder / 'decisions.csv', index=False)
                write(out / 'PROGRESS.json', dict(id=ident, rows_done=rows.index(row)+1,
                    rows=len(rows), pairs=len(plan['dropout_seeds']), seconds=time.time()-start_time))
                print(json.dumps(dict(id=ident, row=row, rows_done=rows.index(row)+1, seconds=round(time.time()-start_time, 1))), flush=True)
        assert all(p.grad is None for p in model.parameters())
        write(folder / 'RESULT.json', dict(id=ident, checkpoint=checkpoint,
            plan_sha256=sha256(plan_path), csv_sha256=sha256(folder / 'decisions.csv'),
            optimizer_steps=0, test_used=False, rows=len(rows), observations=len(records),
            scope='Exploratory selected final checkpoints; first lexical disagreement on native greedy prefix; single-reference costs; not causal or semantic-equivalence labels'))
        del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
        assert temporary.resolve().is_relative_to(folder.resolve())
        shutil.rmtree(temporary)
    write(out / 'COMPLETE.json', dict(passed=True, ids=[c['id'] for c in plan['checkpoints']],
        optimizer_steps=0, test_used=False, plan_sha256=sha256(plan_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan'); parser.add_argument('--output')
    parser.add_argument('--cpu-gate', action='store_true')
    args = parser.parse_args()
    if args.cpu_gate: print(json.dumps(cpu_gate()))
    else:
        assert args.plan and args.output
        run(args)
