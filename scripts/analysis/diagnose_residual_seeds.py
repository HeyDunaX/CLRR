"""Frozen validation interventions on C0 seeds; no training or test generation."""
import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from experiment_runner import engine
from experiment_runner.integrity import sha256, restore_archive
from methods.alignment_residual.controls.probes import residual_intervention


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-free-gpu-mib', type=int, default=20000)
    args = parser.parse_args()
    approved = Path('/workspace/ura/thanhdat/projects/CLRR')
    assert ROOT == approved and Path(sys.prefix).resolve().is_relative_to(ROOT / 'conda')
    out = args.output.resolve()
    assert out.is_relative_to(ROOT / 'outputs/experiments')
    out.mkdir(parents=True, exist_ok=True)
    engine.configure(args.cohort)
    for name in ('torch', 'transformers', 'sacrebleu'):
        value = {'torch': torch.__version__.split('+')[0], 'transformers': engine.transformers.__version__,
                 'sacrebleu': engine.sacrebleu.__version__}[name]
        assert value == engine.plan()['runtime'][name]
    manifest = read(args.cohort / 'bundle_manifest.json')
    for name in ('src/methods/alignment_residual/v1/model.py', 'src/methods/alignment_residual/v1/module.py'):
        assert sha256(ROOT / name) == manifest[name]
    assert torch.cuda.get_device_name(0) == 'Quadro RTX 6000'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = ROOT / 'data_processed' / engine.plan()['dataset'] / 'validation.csv'
    assert sha256(data) == engine.plan()['data']['validation']['sha256']
    rows = pd.read_csv(data, dtype=str, keep_default_na=False)
    design = read(args.design)
    plan = dict(design_sha256=sha256(args.design), validation_sha256=sha256(data), validation_rows=len(rows),
                conditions=['all-on', 'module-off', 'donor-off', 'donor-shuffled'],
                worker_sha256=sha256(Path(__file__)), probe_sha256=sha256(ROOT / 'src/methods/alignment_residual/controls/probes.py'),
                optimizer_steps=0, split='validation', beams=1,
                scope='Post-hoc frozen-model sensitivity; no retrained ablation or causal attribution to training.')
    if (out / 'PLAN.json').exists():
        assert read(out / 'PLAN.json') == plan
    else:
        engine.write(out / 'PLAN.json', plan)
    for run in [r for r in design['runs'] if r['arm'] == 'C0']:
        ident, spec, lock = run['id'], run['spec'], run['selection']
        folder = out / ident
        if (folder / 'RESULT.json').exists():
            assert read(folder / 'RESULT.json')['archive_sha256'] == lock['archive_sha256']
            continue
        folder.mkdir(exist_ok=True)
        while torch.cuda.mem_get_info()[0] // 2**20 < args.minimum_free_gpu_mib:
            engine.write(out / 'progress.json', dict(status='WAIT_FREE_VRAM', id=ident, updated_unix=time.time()))
            time.sleep(30)
        engine.write(out / 'progress.json', dict(status='RESTORING', id=ident, updated_unix=time.time(),
                                                free_mib=torch.cuda.mem_get_info()[0] // 2**20))
        archive = Path(hf_hub_download(engine.REPO, lock['archive_path'], revision=lock['revision'],
                                      token=False, local_dir=folder / 'restore'))
        assert sha256(archive) == lock['archive_sha256']
        restore_archive(archive, folder / 'model')
        model, tokenizer, _ = engine.load_model(spec, folder / 'model')
        model.gradient_checkpointing_disable()
        model.cuda().eval()
        versions = {n: p._version for n, p in model.named_parameters()}
        original = pd.read_csv(run['validation_predictions'].replace('\\', '/'), dtype=str, keep_default_na=False)
        assert original[['source', 'target']].equals(rows[['source', 'target']])
        records, scores, norms = [], {}, []
        with torch.no_grad():
            for condition in plan['conditions']:
                preds, total_loss, total_tokens = [], 0., 0
                for start in range(0, len(rows), 2):
                    frame = rows.iloc[start:start + 2]
                    opts = dict(padding=True, pad_to_multiple_of=8, truncation=True,
                                max_length=spec['max_source_length'], return_tensors='pt')
                    source = tokenizer(frame.source.tolist(), **opts)
                    target = tokenizer(text_target=frame.target.tolist(), **opts)
                    inputs = {k: v.cuda() for k, v in source.items() if k in ('input_ids', 'attention_mask')}
                    labels = target['input_ids'].cuda().masked_fill(target['attention_mask'].cuda().eq(0), -100)
                    mask = inputs['attention_mask'].bool().clone()
                    for special in tokenizer.all_special_ids:
                        mask &= inputs['input_ids'] != special
                    with residual_intervention(model, condition, mask, seed=42+start,
                                               statistics=norms if condition == 'all-on' else None):
                        logits = model.base_model(**inputs, labels=labels).logits
                    losses = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), labels,
                                                                 reduction='none', ignore_index=-100)
                    with residual_intervention(model, condition, mask, seed=42+start):
                        ids = model.generate(**inputs, num_beams=1, max_length=spec['max_target_length'], use_cache=True)
                    batch_preds = [s.strip() for s in tokenizer.batch_decode(ids, skip_special_tokens=True)]
                    preds.extend(batch_preds)
                    nll, tokens = losses.sum(1).cpu().tolist(), labels.ne(-100).sum(1).cpu().tolist()
                    total_loss += sum(nll)
                    total_tokens += sum(tokens)
                    for i in range(len(frame)):
                        records.append(dict(row=start+i, condition=condition, prediction=batch_preds[i],
                                            nll_sum=nll[i], tokens=tokens[i]))
                    if start % 48 == 0:
                        engine.write(out / 'progress.json', dict(status='MEASURING', id=ident, condition=condition,
                                     row=start, total=len(rows), updated_unix=time.time()))
                if condition == 'all-on':
                    assert preds == original.prediction.str.strip().tolist(), 'All-on replay mismatch; no interventions accepted'
                scores[condition] = engine.generation_metrics(preds, rows.target.str.strip().tolist(), tokenize='zh')
                scores[condition]['token_ce'] = total_loss / total_tokens
                print('CONDITION_COMPLETE', ident, condition, scores[condition], flush=True)
        assert versions == {n: p._version for n, p in model.named_parameters()}
        pd.DataFrame(records).to_csv(folder / 'interventions.csv', index=False)
        result = dict(status='PASS', id=ident, seed=run['seed'], archive_sha256=lock['archive_sha256'],
                      checkpoint_revision=lock['revision'], scores=scores, residual_norms=norms,
                      parameter_versions_unchanged=True, optimizer_steps=0, all_on_replay=True,
                      worker_sha256=plan['worker_sha256'], scope=plan['scope'])
        engine.write(folder / 'RESULT.json', result)
        del logits, losses, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        for target in (folder / 'model', folder / 'restore'):
            assert target.resolve().is_relative_to(out)
            shutil.rmtree(target)
        engine.write(folder / 'CLEANUP.json', dict(temporary_weights_removed=True, archive_verified=True))
    engine.write(out / 'COMPLETE.json', dict(status='COMPLETE', optimizer_steps=0, completed_unix=time.time()))
    print('SEED_DIAGNOSTICS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
