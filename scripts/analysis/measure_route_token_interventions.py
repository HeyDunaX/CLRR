"""Frozen validation token interventions; no optimizer, generation, or test data."""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from diagnose_token_decoding import token_decisions, read, write
from experiment_runner import engine
from experiment_runner.integrity import sha256, restore_archive
from methods.original.clrr import CrossLayerResidualRewire

CONDITIONS = {'native': set(), 'all-off': set(range(3, 13)),
              'lower-off': set(range(3, 8)), 'upper-off': set(range(8, 13))}


def install_switches(route, disabled):
    """Keep native cache semantics, switching only the extra receiving edge."""
    for handle in route._handles:
        handle.remove()
    route._handles = []
    for index, layer in enumerate(route.base_model.get_encoder().layers):
        native = route._make_hook('encoder', index)
        def hook(module, inputs, output, index=index, native=native):
            strength = route.strength
            if index + 1 in disabled:
                route.strength = 0.
            try:
                return native(module, inputs, output)
            finally:
                route.strength = strength
        route._handles.append(layer.register_forward_hook(hook))


def comparison(native, other):
    assert native.index.equals(other.index)
    return dict(tokens=len(native), repairs=int((~native.correct & other.correct).sum()),
                damages=int((native.correct & ~other.correct).sum()),
                accuracy_delta_pp=100 * float(other.correct.mean() - native.correct.mean()),
                mean_margin_delta=float((other.margin - native.margin).mean()),
                mean_nll_delta=float((other.nll - native.nll).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume-incomplete', action='store_true')
    parser.add_argument('--minimum-free-gpu-mib', type=int, default=20000)
    args = parser.parse_args()
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda')
    previous = ROOT / args.previous
    old_plan = read(previous / 'PLAN.json')
    assert old_plan['split'] == 'validation' and old_plan['optimizer_steps'] == 0
    assert args.batch_size == old_plan['token_batch']
    runtime = {k: m.__version__.split('+')[0] for k, m in
               [('torch', torch), ('transformers', engine.transformers), ('accelerate', engine.accelerate),
                ('peft', engine.peft), ('sacrebleu', engine.sacrebleu)]}
    assert runtime == old_plan['runtime']
    for name, digest in old_plan['source_sha256'].items():
        if name == 'src/experiment_runner/engine.py':
            continue
        assert sha256(ROOT / name) == digest, ('Frozen source changed', name)
    # Use the exact historical language helper, independently of runner changes.
    historical_engine = (ROOT / args.output).parent / 'frozen_engine.py'
    assert sha256(historical_engine) == old_plan['source_sha256']['src/experiment_runner/engine.py']
    tree = ast.parse(historical_engine.read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_languages')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(historical_engine), 'exec'), namespace)
    configure_languages = namespace['configure_languages']
    data = ROOT / 'data_processed' / old_plan['dataset']
    assert sha256(data / 'validation.csv') == old_plan['data']['validation']['sha256']
    raw = pd.read_csv(data / 'validation.csv', dtype=str, keep_default_na=False)
    references = {}
    for ident in ('R1', 'P1', 'R2'):
        result = read(previous / ident / 'RESULT.json')
        assert result['selection'] == old_plan['selections'][ident]
        assert sha256(previous / ident / 'per_token.csv') == result['file_sha256']['per_token.csv']
        references[ident] = pd.read_csv(previous / ident / 'per_token.csv').set_index(['row', 'position'])
    assert references['R1'].index.equals(references['P1'].index)
    assert references['R1'].index.equals(references['R2'].index)
    out = (ROOT / args.output).resolve()
    assert out.is_relative_to(ROOT / 'outputs/experiments')
    if out.exists():
        assert args.resume_incomplete and not list(out.rglob('RESULT.json'))
        assert not list(out.rglob('TOKENS.csv'))
        (out / 'PLAN.json').rename(out / 'FAILED_NUMERICAL_GATE_PLAN.json')
    else:
        out.mkdir(parents=True)
    frozen_root = out.parent
    recipe = read(frozen_root / 'frozen_recipe.json')
    recipe_lock = read(frozen_root / 'frozen_selection_lock.json')
    assert sha256(frozen_root / 'frozen_recipe.json') == recipe_lock['plan_sha256']
    assert all(recipe_lock['runs'][i] == old_plan['selections'][i] for i in ('R1', 'P1', 'R2'))
    design = dict(split='validation', rows=len(raw), conditions={k: sorted(v) for k, v in CONDITIONS.items()},
                  ids=['P1', 'R2'], baseline_reused='R1', previous=old_plan,
                  current_engine_sha256=sha256(ROOT / 'src/experiment_runner/engine.py'),
                  configure_languages_from_hash_verified_frozen_source=True,
                  script_sha256=sha256(Path(__file__)), precision='fp32', backend='sdpa',
                  tf32=False, dropout=False, batch_size=args.batch_size, optimizer_steps=0, test_access=False,
                  caution='Inference interventions on co-adapted weights, not retrained ablations or novelty evidence')
    write(out / 'PLAN.json', design)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for ident in ('P1', 'R2'):
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= args.minimum_free_gpu_mib, ('Leave other GPU users alone', free)
        print('RESTORE', ident, 'FREE_MIB', free, flush=True)
        folder = out / ident
        folder.mkdir(exist_ok=args.resume_incomplete)
        selected = old_plan['selections'][ident]
        spec = next(s for s in recipe['runs'] if s['id'] == ident)
        archive = Path(hf_hub_download(recipe['repo'], selected['archive_path'], revision=selected['revision'],
                                      token=False, local_dir=folder / 'restore'))
        assert sha256(archive) == selected['archive_sha256']
        export = folder / 'model'
        if not export.exists():
            restore_archive(archive, export)
        core, loading = AutoModelForSeq2SeqLM.from_pretrained(export, output_loading_info=True,
                                                            attn_implementation='sdpa')
        assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
        tok = AutoTokenizer.from_pretrained(export, use_fast=True)
        configure_languages(core, tok, spec)
        sb = tok([spec['source_prefix'] + s for s in raw.source], padding=True, pad_to_multiple_of=8,
                 truncation=True, max_length=spec['max_source_length'], return_tensors='pt')
        tb = tok(text_target=raw.target.tolist(), padding=True, pad_to_multiple_of=8,
                 truncation=True, max_length=spec['max_target_length'], return_tensors='pt')
        labels = tb.input_ids.masked_fill(tb.input_ids.eq(tok.pad_token_id), -100)
        token_hash = hashlib.sha256(b''.join(t.numpy().tobytes() for t in
                                           (sb.input_ids, sb.attention_mask, labels))).hexdigest()
        assert token_hash == read(previous / 'TOKENIZATION.json')['sha256']
        core.gradient_checkpointing_disable()
        core.requires_grad_(False).float().cuda().eval()
        core.config.use_cache = False
        versions = {n: p._version for n, p in core.named_parameters()}
        route = CrossLayerResidualRewire(core, distance=2, strength=.1, stack='encoder').eval()
        disabled = set()
        install_switches(route, disabled)
        records, gates = [], {}
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0, len(raw), args.batch_size):
                stop = min(start + args.batch_size, len(raw))
                batch = {k: sb[k][start:stop].cuda() for k in ('input_ids', 'attention_mask')}
                target = labels[start:stop].cuda()
                for condition, mask in CONDITIONS.items():
                    disabled.clear()
                    disabled.update(mask)
                    output = route(**batch, labels=target, use_cache=False)
                    values = token_decisions(output.logits, target, tok.eos_token_id)
                    assert torch.isfinite(output.logits).all()
                    if start == 0 and condition == 'native':
                        native_logits = output.logits.clone()
                    if start == 0 and condition == 'all-off':
                        # Remove hooks for the true backbone control.
                        for handle in route._handles:
                            handle.remove()
                        bare = core(**batch, labels=target, use_cache=False).logits
                        error = float((bare - output.logits).abs().max())
                        assert error < 1e-5, ('all-off != bare', error)
                        gates['all_off_equals_bare_max_error'] = error
                        install_switches(route, disabled)
                        del bare
                    cpu = {k: values[k].cpu() for k in ('correct', 'margin', 'nll')}
                    for offset in range(stop - start):
                        for pos, token in enumerate(labels[start + offset].tolist()):
                            if token == -100:
                                continue
                            key = (start + offset, pos)
                            category = references[ident].loc[key, 'category']
                            records.append(dict(row=key[0], position=pos, category=category, condition=condition,
                                                correct=bool(cpu['correct'][offset, pos]),
                                                margin=float(cpu['margin'][offset, pos]), nll=float(cpu['nll'][offset, pos])))
                    del output, values, cpu
                if start == 0:
                    disabled.clear()
                    disabled.add(1)
                    control = route(**batch, labels=target, use_cache=False).logits
                    error = float((control - native_logits).abs().max())
                    assert error < 1e-6
                    gates['first_layer_negative_control_error'] = error
                    del control, native_logits
                if start % 100 == 0:
                    print('ROWS', ident, stop, '/', len(raw), flush=True)
            assert versions == {n: p._version for n, p in core.named_parameters()}
            assert all(p.grad is None for p in core.parameters())
        frame = pd.DataFrame(records)
        native = frame[frame.condition == 'native'].set_index(['row', 'position'])
        old = references[ident]
        assert native.index.equals(old.index)
        assert native.correct.equals(old.correct)
        replay = {k: float((native[k] - old[k]).abs().max()) for k in ('nll', 'margin')}
        assert max(replay.values()) < 1e-4, ('Native token replay changed', replay)
        summary = {}
        for condition in CONDITIONS:
            part = frame[frame.condition == condition].set_index(['row', 'position'])
            for category, mask in [('content', native.category.str.startswith('content')), ('eos', native.category == 'eos')]:
                summary[condition + '_' + category] = comparison(native[mask], part[mask])
            if condition == 'native':
                mask = native.category.str.startswith('content')
                summary['native_vs_baseline_content'] = comparison(references['R1'][mask], native[mask])
        frame.to_csv(folder / 'TOKENS.csv', index=False)
        write(folder / 'RESULT.json', dict(id=ident, selection=selected, summary=summary, gates=gates,
              native_replay_max_error=replay, token_csv_sha256=sha256(folder / 'TOKENS.csv'),
              peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, optimizer_steps=0,
              parameters_unchanged=True, gradients_none=True))
        for handle in route._handles:
            handle.remove()
        del core, route, batch, target, sb, tb, labels
        gc.collect()
        torch.cuda.empty_cache()
        for path in (export, folder / 'restore'):
            assert path.resolve().is_relative_to(out)
            shutil.rmtree(path)
        print('COMPLETE_CLEANED', ident, flush=True)
    write(out / 'COMPLETE.json', dict(status='COMPLETE', ids=['P1', 'R2'], optimizer_steps=0,
                                     test_access=False, temporary_weights_removed=True))


if __name__ == '__main__':
    main()
