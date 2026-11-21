"""Validation-only CE interventions on pinned checkpoints; no training or generation."""
import argparse
from collections import Counter
import gc
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
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from experiment_runner import engine
from experiment_runner.integrity import restore_archive, sha256
from methods.original.clrr import CrossLayerResidualRewire
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, data):
    Path(p).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def summarize(frame):
    rng = np.random.default_rng(42)
    out = {}
    for condition in frame.condition.unique():
        part = frame[frame.condition == condition].sort_values('row')
        delta = part.utility.to_numpy()
        means = np.concatenate([delta[rng.integers(0, len(delta), (200, len(delta)))].mean(1)
                                for _ in range(50)])
        out[condition] = {
            'rows': len(part), 'mean_sentence_CE': float(part.ce.mean()),
            'corpus_token_CE': float(part.nll_sum.sum() / part.tokens.sum()),
            'mean_utility_vs_all_on': float(delta.mean()),
            'descriptive_ci95': np.quantile(means, [.025, .975]).tolist(),
            'benefit_fraction_gt_0.001_nats': float((delta > .001).mean()),
            'harm_fraction_lt_minus_0.001_nats': float((delta < -.001).mean()),
        }
    # These reference-dependent choices are optimistic CE diagnostics, not usable predictions.
    off = frame[frame.condition == 'all-off'].sort_values('row')
    on = frame[frame.condition == 'all-on'].sort_values('row')
    oracle = np.minimum(on.ce.to_numpy(), off.ce.to_numpy())
    out['reference_dependent_sentence_oracle'] = {
        'mean_sentence_CE': float(oracle.mean()),
        'gain_vs_best_constant_mean_CE': float(min(on.ce.mean(), off.ce.mean()) - oracle.mean()),
        'not_generation_or_BLEU': True, 'no_gate_fitted': True,
    }
    strata = {}
    for column in ['source_length_bin', 'rare_fraction_bin', 'has_unk']:
        strata[column] = []
        for group, part in off.groupby(column):
            strata[column].append({'group': str(group), 'rows': len(part),
                                  'mean_off_minus_on_CE': float(part.utility.mean())})
    out['all_off_strata'] = strata
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', required=True, help='Existing frozen evaluation workspace with selection_lock.json')
    ap.add_argument('--output', required=True)
    ap.add_argument('--ids', nargs='+', required=True)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--minimum-free-gpu-mib', type=int, default=16000)
    args = ap.parse_args()
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda')
    workspace = ROOT / args.workspace
    engine.configure(workspace)
    plan = read(workspace / 'PLAN.json'); lock = read(workspace / 'selection_lock.json')
    versions = {'torch': torch.__version__.split('+')[0], 'transformers': engine.transformers.__version__,
                'accelerate': engine.accelerate.__version__, 'peft': engine.peft.__version__,
                'sacrebleu': engine.sacrebleu.__version__}
    assert all(v == plan['runtime'][k] for k,v in versions.items())
    old_manifest = read(workspace / 'bundle_manifest.json')
    route_sources = ['src/methods/original/clrr.py', 'src/methods/common/helpers.py']
    assert all(sha256(ROOT / f) == old_manifest[f] for f in route_sources)
    out = (ROOT / args.output).resolve()
    assert out.is_relative_to(ROOT / 'outputs/experiments') and not out.exists()
    out.mkdir(parents=True)
    data = ROOT / 'data_processed' / plan['dataset']
    raw = pd.read_csv(data / 'validation.csv', dtype=str, keep_default_na=False)
    train = pd.read_csv(data / 'train.csv', dtype=str, keep_default_na=False)
    assert sha256(data / 'validation.csv') == plan['data']['validation']['sha256']
    assert sha256(data / 'train.csv') == plan['data']['train']['sha256']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    design = {'dataset': plan['dataset'], 'split': 'validation', 'rows': len(raw),
              'data_hashes': {s: plan['data'][s] for s in ['train', 'validation']},
              'checkpoint_selections': {i: lock['runs'][i] for i in args.ids},
              'source_worker_sha256': sha256(Path(__file__)), 'optimizer_steps': 0,
              'runtime': versions, 'routing_sources_sha256': {f: sha256(ROOT/f) for f in route_sources},
              'loader': 'Strict bare HF backbone; no Trainer, auxiliary wrappers, registry, or metrics used',
              'conditions': ['all-on', 'all-off'] + ['layer-%02d-off' % i for i in range(3, 13)],
              'utility': 'CE(condition)-CE(all-on); positive means the removed route helped',
              'reduction': 'Per-example mean over all non-pad target labels, including specials; corpus token CE also reported',
              'precision': 'FP32, TF32 off, dropout off, checkpointing off',
              'rare_token_definition': 'Source tokenizer content token train frequency <=5; no filtering',
              'bins': {'source_whitespace_units': ['<=2', '3..10', '>10'],
                       'rare_fraction': ['0', '(0,.25]', '>.25']},
              'inference_interventions_not_training_ablation': True,
              'oracle_uses_validation_references': True, 'test_examples_evaluated': False}
    write(out / 'PLAN.json', design)
    for ident in args.ids:
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
        print('GPU_BEFORE', ident, free, flush=True)
        assert free >= args.minimum_free_gpu_mib, 'Insufficient VRAM; leave other jobs alone'
        folder = out / ident; folder.mkdir()
        spec = next(s for s in plan['runs'] if s['id'] == ident)
        selected = lock['runs'][ident]
        archive = Path(hf_hub_download(plan['repo'], selected['archive_path'],
                      revision=selected['revision'], token=False, local_dir=folder/'restore'))
        assert sha256(archive) == selected['archive_sha256']
        export = folder / 'model'; restore_archive(archive, export)
        assert spec['method'] in ['baseline', 'clrr', 'full', 'original-full']
        model, loading = AutoModelForSeq2SeqLM.from_pretrained(export, output_loading_info=True)
        assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        tok = AutoTokenizer.from_pretrained(export, use_fast=True)
        engine.configure_languages(model, tok, spec)
        model.gradient_checkpointing_disable(); model.cuda().eval()
        versions = {n: p._version for n, p in model.named_parameters()}
        core = model
        route = CrossLayerResidualRewire(core, distance=2, strength=.1)
        assert route.distance == 2 and route.stack == 'encoder' and route.strength == .1
        assert len(core.get_encoder().layers) == 12
        # Remove original hooks, then reinstall the same hooks with explicit per-layer switches.
        for handle in route._handles: handle.remove()
        route._handles = []; disabled = set()
        for i, layer in enumerate(core.get_encoder().layers):
            original = route._make_hook('encoder', i)
            def intervene(module, inputs, value, i=i, original=original):
                strength = route.strength
                if i + 1 in disabled: route.strength = 0.
                try: return original(module, inputs, value)
                finally: route.strength = strength
            route._handles.append(layer.register_forward_hook(intervene))
        specials = set(tok.all_special_ids)
        frequency = Counter()
        for start in range(0, len(train), 256):
            ids = tok([spec['source_prefix'] + s for s in train.source.iloc[start:start+256]],
                      truncation=True, max_length=spec['max_source_length'])['input_ids']
            frequency.update(t for seq in ids for t in seq if t not in specials)
        records = []; gates = {}
        conditions = {'all-on': set(), 'all-off': set(range(1, 13)),
                      **{'layer-%02d-off' % i: {i} for i in range(3, 13)}}
        with torch.inference_mode():
            for start in range(0, len(raw), args.batch_size):
                chunk = raw.iloc[start:start+args.batch_size]
                sb = tok([spec['source_prefix']+s for s in chunk.source], padding=True,
                         pad_to_multiple_of=8, truncation=True, max_length=spec['max_source_length'], return_tensors='pt')
                tb = tok(text_target=chunk.target.tolist(), padding=True, pad_to_multiple_of=8,
                         truncation=True, max_length=spec['max_target_length'], return_tensors='pt')
                labels = tb['input_ids'].cuda().masked_fill(tb['input_ids'].cuda() == tok.pad_token_id, -100)
                batch = {k: v.cuda() for k,v in sb.items() if k in ['input_ids','attention_mask']}
                def forward():
                    with route._fresh_cache():
                        return core(**batch, labels=labels, use_cache=False)
                losses = {}; lengths = (labels != -100).sum(1).cpu().numpy()
                for condition, mask in conditions.items():
                    disabled.clear(); disabled.update(mask)
                    outputs = forward()
                    token_loss = F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                                                 labels.reshape(-1), reduction='none', ignore_index=-100).reshape_as(labels)
                    assert torch.isfinite(token_loss).all()
                    assert abs(float(token_loss.sum() / (labels != -100).sum()) - float(outputs.loss)) < 2e-6
                    losses[condition] = token_loss.sum(1).cpu().numpy()
                    if start == 0 and condition == 'all-on':
                        reference_logits = outputs.logits.clone()
                    del outputs, token_loss
                if start == 0:
                    for name, mask, strength in [('repeat-all-on', set(), .1),
                              ('first-layer-negative-control', {1}, .1), ('zero-strength', set(), 0.)]:
                        disabled.clear(); disabled.update(mask); route.strength = strength
                        value = forward()
                        if strength:
                            error = float((value.logits-reference_logits).abs().max()); assert error <= 1e-6
                        else:
                            nll = F.cross_entropy(value.logits.reshape(-1,value.logits.shape[-1]),labels.reshape(-1),
                                                 ignore_index=-100,reduction='none').reshape_as(labels).sum(1).cpu().numpy()
                            error = float(np.max(np.abs(nll-losses['all-off']))); assert error <= 1e-5
                        gates[name] = {'max_abs_error': error, 'pass': True}; del value
                    del reference_logits; route.strength = .1
                for offset, (_, row) in enumerate(chunk.iterrows()):
                    content = [t for t in sb['input_ids'][offset].tolist() if t not in specials]
                    rare = sum(frequency[t] <= 5 for t in content) / max(1, len(content))
                    words = len(row.source.split())
                    for condition in conditions:
                        nll = float(losses[condition][offset]); count = int(lengths[offset])
                        records.append({'row': start+offset, 'condition': condition, 'ce': nll/count,
                            'nll_sum': nll, 'tokens': count,
                            'utility': float((losses[condition][offset]-losses['all-on'][offset])/count),
                            'source_length_bin': '<=2' if words <= 2 else ('3..10' if words <= 10 else '>10'),
                            'rare_fraction_bin': '0' if rare == 0 else ('(0,.25]' if rare <= .25 else '>.25'),
                            'has_unk': tok.unk_token_id in sb['input_ids'][offset].tolist()})
                if start % 100 == 0: print('ROWS', ident, start+len(chunk), '/', len(raw), flush=True)
        assert versions == {n:p._version for n,p in model.named_parameters()}
        assert all(p.grad is None for p in model.parameters())
        frame = pd.DataFrame(records); frame.to_csv(folder/'per_sentence.csv', index=False)
        write(folder/'RESULT.json', {'id': ident, 'method': spec['method'], 'selection': selected,
              'summary': summarize(frame), 'checks': gates, 'strict_loading': loading,
              'parameter_versions_unchanged': True, 'optimizer_steps': 0, 'gradients_none': True,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated()//2**20,
              'results_csv_sha256': sha256(folder/'per_sentence.csv')})
        for handle in route._handles: handle.remove()
        route._handles = []
        del model, core, route, original, intervene, layer, batch, labels, frame
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for target in [export, folder/'restore']:
            assert target.resolve().is_relative_to(out); shutil.rmtree(target)
        print('MEASURED_CLEANED', ident, flush=True)
    write(out/'COMPLETE.json', {'status': 'COMPLETE', 'ids': args.ids, 'optimizer_steps': 0,
          'test_examples_evaluated': False, 'temporary_weights_removed': True, 'time': time.time()})


if __name__ == '__main__':
    main()
