"""Train-only, fixed-sample proposal probe; no optimization or validation/test.

Replay cached T1 source batches before testing one declared temperature.
This never changes the rejected recipe, pool, baseline, or gate thresholds.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from experiment_runner import engine as e


def wilson_upper(count, total):
    z = 1.959963984540054
    p = count / total
    return (p + z*z/(2*total) + z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def summarize(records):
    single = sum(len(r['labels']) == 1 for r in records)
    equal = sum(max(r['costs']) - min(r['costs']) < 1e-8 for r in records)
    sampled = [c for r in records for c in r['costs'][1:]]
    return dict(rows=len(records), candidate_counts=dict(collections.Counter(len(r['labels']) for r in records)),
        singleton_rows=single, equal_cost_rows=equal, singleton_fraction=single/len(records),
        equal_cost_fraction=equal/len(records),
        descriptive_wilson_upper95_singleton=wilson_upper(single, len(records)),
        descriptive_wilson_upper95_equal_cost=wilson_upper(equal, len(records)),
        mean_unique_candidates=float(np.mean([len(r['labels']) for r in records])),
        mean_unique_sample_cost=float(np.mean(sampled)) if sampled else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--temperature', type=float, required=True)
    p.add_argument('--source-batches', type=int, default=128)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--replay-batches', type=int, default=4)
    args = p.parse_args()
    assert math.isfinite(args.temperature) and args.temperature > 1
    assert args.source_batches >= args.replay_batches >= 1
    assert args.output.resolve().is_relative_to(ROOT.resolve())
    assert not args.output.exists(), 'Preserve old probe; use a new output directory'
    e.configure(args.workspace)
    e.validate_environment()
    torch.set_num_threads(4)
    plan = e.plan(); cfg = plan['candidate_pool']; spec = plan['runs'][0]
    assert spec['method'] == 'sequence-ce-v1' and cfg['source_batch'] == 2 and cfg['temperature'] == 1.
    assert cfg['split'] == 'train'
    old_path = args.workspace / 'candidate_pool.jsonl.partial'
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    raw = pd.read_csv(e.DATA / 'train.csv', dtype=str, keep_default_na=False)
    assert len(old) == len(raw) == plan['data']['train']['rows']
    assert all(r['row'] == i for i, r in enumerate(old))
    pairs = np.random.default_rng(args.seed).choice(len(raw)//2, args.source_batches, replace=False)
    starts = sorted(int(x)*2 for x in pairs)
    args.output.mkdir(parents=True)
    design = dict(workspace=str(args.workspace), old_pool_sha256=e.sha256(old_path),
        baseline=plan['initialization'], train=plan['data']['train'], temperature=args.temperature,
        samples=cfg['samples'], seed=args.seed, original_source_batch_starts=starts,
        sampling_seed_rule='1042 + original batch start', replay_batches=args.replay_batches,
        no_validation_or_test=True, optimizer_steps=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        declared_gates=dict(each_degenerate_fraction_below=.5, length_cap_sample_fraction_below=.05,
            no_eos_sample_fraction_below=.05, exact_temperature1_replay=True))
    e.write(args.output / 'DESIGN.json', design)
    model, tokenizer, _ = e.load_model(spec)
    model.to('cuda'); model.eval()
    metric = CHRF(word_order=0)
    draw_stats = collections.Counter()
    new = []
    started = time.time()

    def generate(start, temperature, stats):
        batch = raw.iloc[start:start+2]
        encoded = tokenizer([spec['source_prefix'] + x for x in batch.source], padding=True,
            truncation=True, max_length=spec['max_source_length'], return_tensors='pt').to('cuda')
        torch.manual_seed(cfg['seed'] + start)
        with torch.no_grad():
            outputs = model.generate(**encoded, do_sample=True, num_beams=1,
                num_return_sequences=cfg['samples'], max_length=spec['max_target_length'],
                temperature=temperature, top_k=0, top_p=1., use_cache=True).cpu().tolist()
        records = []
        for offset in range(2):
            row = start + offset
            gold = tokenizer(text_target=raw.target.iloc[row], truncation=True,
                max_length=spec['max_target_length'])['input_ids']
            assert gold == old[row]['labels'][0]
            labels = [gold]
            for sequence in outputs[offset*cfg['samples']:(offset+1)*cfg['samples']]:
                stats['sample_draws'] += 1
                assert sequence[0] == model.config.decoder_start_token_id
                sequence = sequence[1:]
                assert sequence[0] == model.config.forced_bos_token_id
                if tokenizer.eos_token_id not in sequence:
                    stats['no_eos'] += 1
                    stats['length_cap'] += int(len(sequence) + 1 == spec['max_target_length'])
                    continue
                sequence = sequence[:sequence.index(tokenizer.eos_token_id)+1]
                stats['length_cap'] += int(len(sequence) + 1 == spec['max_target_length'])
                if sequence == gold:
                    stats['gold_duplicate_draws'] += 1
                elif sequence in labels:
                    stats['non_gold_duplicate_draws'] += 1
                else:
                    stats['new_non_gold_draws'] += 1
                if sequence not in labels:
                    labels.append(sequence)
            costs = [max(0., min(1., 1. - metric.sentence_score(
                tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                [raw.target.iloc[row].strip()]).score/100.)) for tokens in labels]
            records.append(dict(row=row, labels=labels, costs=costs))
        return records

    replay_stats = collections.Counter()
    for start in starts[:args.replay_batches]:
        for result in generate(start, 1., replay_stats):
            expected = old[result['row']]
            assert result['labels'] == expected['labels'], ('T1 replay failed', result['row'])
            np.testing.assert_allclose(result['costs'], expected['costs'], atol=1e-12, rtol=0)
    e.write(args.output / 'REPLAY_PASS.json', dict(passed=True, rows=args.replay_batches*2,
        draw_stats=dict(replay_stats), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32, generation_config=model.generation_config.to_dict()))
    for index, start in enumerate(starts):
        new.extend(generate(start, args.temperature, draw_stats))
        if index % 8 == 0:
            e.write(args.output / 'progress.json', dict(rows=len(new), total=args.source_batches*2,
                seconds=time.time()-started))
            print('PROBE_PROGRESS', len(new), args.source_batches*2, flush=True)
    baseline = summarize([old[r['row']] for r in new])
    candidate = summarize(new)
    cap_fraction = draw_stats['length_cap']/draw_stats['sample_draws']
    no_eos_fraction = draw_stats['no_eos']/draw_stats['sample_draws']
    passed = (candidate['singleton_fraction'] < .5 and candidate['equal_cost_fraction'] < .5
        and cap_fraction < .05 and no_eos_fraction < .05)
    report = dict(status='PASS' if passed else 'NO_GO', baseline_temperature1=baseline,
        candidate_temperature=args.temperature, candidate=candidate, draw_stats=dict(draw_stats),
        length_cap_fraction=cap_fraction, no_eos_fraction=no_eos_fraction,
        replay_rows=args.replay_batches*2, seconds=time.time()-started,
        descriptive_subset_only=True, no_validation_or_test=True, optimizer_steps=0,
        instruction='Only full 4600-row pool may satisfy final readiness gate; no claim of quality improvement')
    e.write(args.output / 'PROBE_REPORT.json', report)
    # Token IDs remain on RTX; only the derived report is collected/public eligible.
    (args.output / 'candidate_probe.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
    print('PROBE_REPORT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
