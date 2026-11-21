"""Frozen validation: route ON/OFF crossed with matched/mismatched source, no training."""
import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from experiment_runner import engine
from experiment_runner.integrity import sha256, restore_archive
from methods.original.clrr import CrossLayerResidualRewire
from methods.grounded_interaction.v1.model import source_negatives

CELLS = ('on_match', 'on_mismatch', 'off_match', 'off_mismatch')
METRICS = ('g_on', 'g_off', 'interaction', 'route_matched_gain', 'route_mismatched_gain')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def add_metrics(frame):
    frame = frame.copy()
    # Scores s=-CE. Positive g: gold target more probable with its own source.
    frame['g_on'] = frame.on_mismatch - frame.on_match
    frame['g_off'] = frame.off_mismatch - frame.off_match
    frame['route_matched_gain'] = frame.off_match - frame.on_match
    frame['route_mismatched_gain'] = frame.off_mismatch - frame.on_mismatch
    frame['interaction'] = frame.g_on - frame.g_off
    assert np.allclose(frame.interaction, frame.route_matched_gain-frame.route_mismatched_gain)
    return frame


def block_intervals(values, blocks, repetitions, seed=42):
    """Resample whole negative-pair microbatch blocks, retaining row weights."""
    values = np.asarray(values, dtype=np.float64)
    _, inverse = np.unique(blocks, return_inverse=True)
    n = inverse.max() + 1
    sums = np.zeros((n, values.shape[1])); np.add.at(sums, inverse, values)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(0, n, (repetitions, n))
    means = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
    return np.quantile(means, [.025, .975], axis=0).T


def summarize(frame, repetitions):
    valid = frame[frame.valid_negative]
    assert len(valid) > 0
    cis = block_intervals(valid[list(METRICS)].values, valid.block.values, repetitions)
    return dict(rows=len(frame), valid_pairs=len(valid), excluded_pairs=len(frame)-len(valid),
        all_rows_mean_CE={k: float(frame[k].mean()) for k in CELLS},
        metrics_valid_pairs={key: dict(mean=float(valid[key].mean()),
            descriptive_block_ci95=ci.tolist(), fraction_gt_0_001=float((valid[key] > .001).mean()))
            for key, ci in zip(METRICS, cis)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', required=True)
    p.add_argument('--reference-audit', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--bootstrap-repetitions', type=int, default=10000)
    p.add_argument('--minimum-free-gpu-mib', type=int, default=16000)
    args = p.parse_args()
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda')
    assert args.batch_size >= 2 and args.bootstrap_repetitions >= 100
    workspace = ROOT / args.workspace
    engine.configure(workspace)
    plan, lock = read(workspace/'PLAN.json'), read(workspace/'selection_lock.json')
    runtime = {name: getattr(module, '__version__').split('+')[0] for name, module in
        [('torch', torch), ('transformers', engine.transformers), ('accelerate', engine.accelerate),
         ('peft', engine.peft), ('sacrebleu', engine.sacrebleu)]}
    assert all(runtime[k] == plan['runtime'][k] for k in runtime)
    old_manifest = read(workspace/'bundle_manifest.json')
    core_sources = ['src/methods/original/clrr.py', 'src/methods/common/helpers.py']
    assert all(sha256(ROOT/f) == old_manifest[f] for f in core_sources)
    specs = [next(s for s in plan['runs'] if s['id'] == ident) for ident in args.ids]
    language_fields = ['src_lang', 'tgt_lang', 'source_prefix', 'max_source_length', 'max_target_length']
    assert all([s[k] for k in language_fields] == [specs[0][k] for k in language_fields] for s in specs)
    data = ROOT / 'data_processed' / plan['dataset'] / 'validation.csv'
    assert sha256(data) == plan['data']['validation']['sha256']
    raw = pd.read_csv(data, dtype=str, keep_default_na=False)
    assert len(raw) == plan['data']['validation']['rows']
    out = (ROOT / args.output).resolve()
    assert out.is_relative_to(ROOT/'outputs/experiments')
    out.mkdir(parents=True, exist_ok=True)
    sources = core_sources + ['scripts/analysis/measure_source_grounding.py',
        'src/experiment_runner/engine.py', 'src/experiment_runner/integrity.py',
        'src/methods/grounded_interaction/v1/model.py']
    design = dict(dataset=plan['dataset'], split='validation', rows=len(raw),
        validation_sha256=sha256(data), selections={i: lock['runs'][i] for i in args.ids},
        runtime=runtime, cells=list(CELLS), batch_size=args.batch_size,
        negatives='First cyclic token-distinct source AND target within fixed contiguous blocks; all models share pairs',
        units='nats per non-pad supervised target token, including special labels',
        formulas=dict(g_on='CE(on,mismatch)-CE(on,match)', g_off='CE(off,mismatch)-CE(off,match)',
            interaction='g_on-g_off', route_matched_gain='CE(off,match)-CE(on,match)'),
        bootstrap=dict(repetitions=args.bootstrap_repetitions, seed=42, unit='whole negative-pair block',
                       descriptive_only=True, no_confirmatory_p_values=True),
        precision='FP32, TF32/dropout/checkpointing OFF', optimizer_steps=0, test_evaluated=False,
        source_sha256={f: sha256(ROOT/f) for f in sources},
        baseline_on_is_artificial_route=True, causal_identification_claim=False)
    if (out/'PLAN.json').exists(): assert read(out/'PLAN.json') == design, 'Frozen plan/source changed'
    else: write(out/'PLAN.json', design)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for ident, spec in zip(args.ids, specs):
        folder = out/ident; folder.mkdir(exist_ok=True)
        if (folder/'RESULT.json').exists():
            assert sha256(folder/'per_sentence.csv') == read(folder/'RESULT.json')['csv_sha256']
            print('VERIFIED_SKIP', ident, flush=True); continue
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= args.minimum_free_gpu_mib, ('Insufficient free VRAM', free)
        print('RESTORING', ident, 'FREE_MIB', free, flush=True)
        selected = lock['runs'][ident]
        archive = Path(hf_hub_download(plan['repo'], selected['archive_path'], revision=selected['revision'],
            token=False, local_dir=folder/'restore'))
        assert sha256(archive) == selected['archive_sha256']
        export = folder/'model'; restore_archive(archive, export)
        core, loading = AutoModelForSeq2SeqLM.from_pretrained(export, output_loading_info=True)
        assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        tok = AutoTokenizer.from_pretrained(export, use_fast=True)
        engine.configure_languages(core, tok, spec)
        sb = tok([spec['source_prefix']+s for s in raw.source], padding=True, pad_to_multiple_of=8,
            truncation=True, max_length=spec['max_source_length'], return_tensors='pt')
        tb = tok(text_target=raw.target.tolist(), padding=True, pad_to_multiple_of=8,
            truncation=True, max_length=spec['max_target_length'], return_tensors='pt')
        labels = tb['input_ids'].masked_fill(tb['input_ids']==tok.pad_token_id, -100)
        assert labels.ne(-100).sum(1).gt(0).all()
        token_hash = __import__('hashlib').sha256(b''.join(t.numpy().tobytes() for t in
            (sb['input_ids'], sb['attention_mask'], labels))).hexdigest()
        token_record = dict(sha256=token_hash, source_shape=list(sb['input_ids'].shape), target_shape=list(labels.shape))
        if (out/'TOKENIZATION.json').exists(): assert read(out/'TOKENIZATION.json') == token_record
        else: write(out/'TOKENIZATION.json', token_record)
        pairs = []
        for start in range(0, len(raw), args.batch_size):
            stop = min(start+args.batch_size, len(raw))
            ix, valid = source_negatives(sb['input_ids'][start:stop], sb['attention_mask'][start:stop], labels[start:stop])
            pairs.extend(dict(row=i, negative_row=start+int(ix[i-start]), valid_negative=bool(valid[i-start]),
                              block=start//args.batch_size) for i in range(start, stop))
        pair_frame = pd.DataFrame(pairs)
        if (out/'NEGATIVE_PAIRS.csv').exists():
            pd.testing.assert_frame_equal(pd.read_csv(out/'NEGATIVE_PAIRS.csv'), pair_frame)
        else: pair_frame.to_csv(out/'NEGATIVE_PAIRS.csv', index=False)
        core.gradient_checkpointing_disable(); core.float().cuda().eval()
        versions = {n: value._version for n, value in core.named_parameters()}
        route = CrossLayerResidualRewire(core, distance=2, strength=.1, stack='encoder'); route.eval()
        records = []; checks = {}
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0, len(raw), args.batch_size):
                part = pair_frame.iloc[start:start+args.batch_size]
                target = labels[part.row.to_numpy(copy=True)].cuda()
                counts = target.ne(-100).sum(1)
                values = {}; nlls = {}
                for cell in CELLS:
                    route.strength = .1 if cell.startswith('on_') else 0.
                    rows = part.negative_row.to_numpy(copy=True) if cell.endswith('mismatch') else part.row.to_numpy(copy=True)
                    batch = dict(input_ids=sb['input_ids'][rows].cuda(), attention_mask=sb['attention_mask'][rows].cuda())
                    outputs = route(**batch, labels=target, use_cache=False)
                    token_nll = F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                        target.reshape(-1), ignore_index=-100, reduction='none').reshape_as(target)
                    nll = token_nll.sum(1)
                    assert torch.isfinite(nll).all()
                    loss_error = abs(float(nll.sum()/counts.sum())-float(outputs.loss))
                    assert loss_error < 3e-6, (cell, float(nll.sum()/counts.sum()), float(outputs.loss), loss_error)
                    if start == 0:
                        old_nll = F.cross_entropy(outputs.logits.float().transpose(1,2), target,
                            ignore_index=-100, reduction='none').sum(1)
                        checks[cell+'_CE_reduction'] = dict(flat_vs_HF_abs_error=loss_error,
                            transpose_vs_flat_max_sentence_error=float(((old_nll-nll)/counts).abs().max()))
                        print('CE_CHECK',ident,cell,checks[cell+'_CE_reduction'],flush=True)
                        del old_nll
                    values[cell]=(nll/counts).cpu().numpy(); nlls[cell]=nll.cpu().numpy()
                    if start == 0 and cell == 'off_match':
                        repeated = route(**batch, labels=target, use_cache=False)
                        repeat_error = float((repeated.logits-outputs.logits).abs().max())
                        assert repeat_error == 0.
                        # Remove route hooks to compare OFF against the bare checkpoint.
                        for handle in route._handles: handle.remove()
                        route._handles=[]
                        bare = core(**batch, labels=target, use_cache=False)
                        bare_error = float((bare.logits-outputs.logits).abs().max())
                        assert bare_error == 0.
                        route._install_hooks()
                        checks.update(repeat_off_max_error=repeat_error, zero_strength_vs_bare_max_error=bare_error)
                        del repeated, bare
                    del outputs, nll, token_nll
                for i, row in enumerate(part.to_dict('records')):
                    records.append(dict(**row, tokens=int(counts[i]),
                        **{key:float(values[key][i]) for key in CELLS},
                        **{'nll_'+key:float(nlls[key][i]) for key in CELLS}))
                if start % 100 == 0: print('ROWS', ident, start+len(part), '/', len(raw), flush=True)
        frame = add_metrics(pd.DataFrame(records))
        previous = pd.read_csv(ROOT/args.reference_audit/ident/'per_sentence.csv')
        replay = {}
        for old, new in [('all-on','on_match'),('all-off','off_match')]:
            expected = previous[previous.condition==old].sort_values('row')
            assert expected.row.tolist()==frame.row.tolist()
            error=float(np.max(np.abs(expected.ce.values-frame[new].values)))
            assert error < 1e-4, ('Prior CE disagreement',ident,new,error)
            replay[new]=dict(max_abs_CE_error=error, tolerance=1e-4, pass_check=True)
        assert versions == {n:value._version for n,value in core.named_parameters()}
        assert all(value.grad is None for value in core.parameters())
        frame.to_csv(folder/'per_sentence.csv', index=False)
        write(folder/'RESULT.json', dict(id=ident, method=spec['method'], selection=selected,
            summary=summarize(frame,args.bootstrap_repetitions), strict_loading=loading, checks=checks,
            prior_CE_replay=replay, parameter_versions_unchanged=True, gradients_none=True, optimizer_steps=0,
            csv_sha256=sha256(folder/'per_sentence.csv'), pairs_sha256=sha256(out/'NEGATIVE_PAIRS.csv'),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        for handle in route._handles: handle.remove()
        del route, core, labels, sb, tb, target, batch, frame
        gc.collect(); torch.cuda.empty_cache()
        for path in (export, folder/'restore'):
            assert path.resolve().is_relative_to(out)
            shutil.rmtree(path)
        print('MEASURED_CLEANED', ident, flush=True)
    write(out/'COMPLETE.json',dict(status='COMPLETE', ids=args.ids, rows=len(raw),
        optimizer_steps=0, test_evaluated=False, temporary_weights_removed=True))


if __name__ == '__main__':
    main()
