"""Frozen RTX validation interventions and actual-policy CE/LSR gradients; no updates."""
import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from experiment_runner import engine
from experiment_runner.integrity import sha256, restore_archive
from methods.alignment_residual.controls.probes import residual_intervention


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    engine.write(path, value)


def tensors(tokenizer, frame, spec):
    options = dict(padding=True, pad_to_multiple_of=8, truncation=True,
                   max_length=spec['max_source_length'], return_tensors='pt')
    source = tokenizer(frame.source.tolist(), **options)
    target = tokenizer(text_target=frame.target.tolist(), **options)
    source = {k: v.cuda() for k, v in source.items() if k in ('input_ids', 'attention_mask')}
    labels = target['input_ids'].cuda().masked_fill(target['attention_mask'].cuda().eq(0), -100)
    mask = source['attention_mask'].bool().clone()
    for special in tokenizer.all_special_ids:
        mask &= source['input_ids'] != special
    return source, labels, mask


def grouped_gradients(names, ce, auxiliary):
    groups = {'module': [i for i,n in enumerate(names) if '.residual_modules.' in n],
              'backbone': [i for i,n in enumerate(names) if '.residual_modules.' not in n]}
    result = {}
    for group, indices in groups.items():
        a2 = sum(float(ce[i].double().square().sum()) for i in indices if ce[i] is not None)
        b2 = sum(float(auxiliary[i].double().square().sum()) for i in indices if auxiliary[i] is not None)
        dot = sum(float((ce[i].double()*auxiliary[i].double()).sum()) for i in indices
                  if ce[i] is not None and auxiliary[i] is not None)
        result[group] = dict(ce_norm=a2**.5, weighted_lsr_norm=b2**.5,
            weighted_lsr_over_ce=(b2/a2)**.5 if a2 else None,
            cosine=dot/(a2*b2)**.5 if a2 and b2 else None)
    return result


def measure(args, ident, cohort, out, rows, batches):
    folder = out / ident
    if (folder / 'RESULT.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    marker = read(cohort / 'runs' / ident / 'VAL_COMPLETE.json')
    assert marker['status'] == 'VALIDATION_COMPLETE_VERIFIED' and marker['backup']['verified']
    spec = marker['spec']
    free = torch.cuda.mem_get_info()[0] // 2**20
    print('GPU_BEFORE', ident, free, flush=True)
    assert free >= args.minimum_free_gpu_mib
    archive = Path(hf_hub_download(engine.REPO, spec['remote_prefix']+'/best.zip',
        revision=marker['backup']['revision'], token=False, local_dir=folder/'restore'))
    assert sha256(archive) == marker['archive_sha256']
    export = folder / 'model'
    restore_archive(archive, export)
    model, tokenizer, _ = engine.load_model(spec, export)
    model.gradient_checkpointing_disable()
    model.cuda().eval()
    model.config.use_cache = False
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    records, scores, gradients, residual_norms = [], {}, [], []
    if ident == 'AR_C0':
        original = pd.read_csv(cohort/'runs'/ident/'validation_predictions.csv', dtype=str, keep_default_na=False)
        assert original[['source','target']].equals(rows[['source','target']])
        for condition in ('all-on', 'module-off', 'donor-off', 'donor-shuffled'):
            predictions = []
            nlls, counts = [], []
            for start in range(0, len(rows), 2):
                part = rows.iloc[start:start+2]
                source, labels, mask = tensors(tokenizer, part, spec)
                with torch.no_grad():
                    if start == 0:
                        expected = model.base_model(**source, labels=labels).logits
                    with residual_intervention(model, condition, mask, seed=42+start,
                            statistics=residual_norms if condition == 'all-on' else None):
                        logits = model.base_model(**source, labels=labels).logits
                    if start == 0 and condition == 'all-on':
                        torch.testing.assert_close(logits, expected, rtol=1e-5, atol=2e-5)
                    losses = torch.nn.functional.cross_entropy(logits.float().transpose(1,2), labels,
                        reduction='none', ignore_index=-100)
                    nlls.extend(losses.sum(1).cpu().tolist())
                    counts.extend(labels.ne(-100).sum(1).cpu().tolist())
                    with residual_intervention(model, condition, mask, seed=42+start):
                        ids = model.generate(**source, num_beams=1, max_length=spec['max_target_length'], use_cache=True)
                    predictions.extend(s.strip() for s in tokenizer.batch_decode(ids, skip_special_tokens=True))
                if start % 96 == 0:
                    print('INTERVENTION', condition, start, len(rows), flush=True)
            if condition == 'all-on':
                assert predictions == original.prediction.str.strip().tolist(), 'All-on generation replay mismatch'
            scores[condition] = engine.generation_metrics(predictions, rows.target.str.strip().tolist(), tokenize='zh')
            for index in range(len(rows)):
                records.append(dict(row=index,condition=condition,prediction=predictions[index],
                    nll_sum=nlls[index],tokens=counts[index],ce=nlls[index]/counts[index]))
            scores[condition]['token_ce'] = sum(nlls)/sum(counts)
            print('CONDITION_COMPLETE', condition, scores[condition], flush=True)
        pd.DataFrame(records).to_csv(folder/'interventions.csv', index=False)
    else:
        parameters = list(model.named_parameters())
        for indices in batches:
            source, labels, mask = tensors(tokenizer, rows.iloc[indices], spec)
            with residual_intervention(model, 'all-on', mask, statistics=residual_norms):
                ce = model.base_model(**source, labels=labels).loss
            ce_grad = torch.autograd.grad(ce, [p for _,p in parameters], allow_unused=True)
            target_ids = labels.masked_fill(labels.eq(-100), model.config.pad_token_id)
            target_mask = target_ids.ne(model.config.pad_token_id).long()
            with torch.no_grad():
                target = model.get_encoder()(input_ids=target_ids, attention_mask=target_mask).last_hidden_state
                anchor = model.pooled(target, target_mask)
            lsr = model.auxiliary_loss(source['input_ids'], source['attention_mask'], anchor)
            auxiliary = torch.autograd.grad(model.jepa_weight*lsr, [p for _,p in parameters], allow_unused=True)
            assert all(g is None or torch.isfinite(g).all() for g in ce_grad+auxiliary)
            values = grouped_gradients([n for n,_ in parameters], ce_grad, auxiliary)
            if ident == 'AR_C2':
                assert values['backbone']['weighted_lsr_norm'] == 0., 'Confinement violated'
            with torch.no_grad():
                actual = model(**source, labels=labels).loss
                assert float((actual-ce.detach()-model.jepa_weight*lsr.detach()).abs()) < 3e-6
            gradients.append(dict(rows=indices,ce=float(ce.detach()),lsr=float(lsr.detach()),groups=values))
            del ce_grad, auxiliary, ce, lsr, actual
            print('GRADIENT_BATCH', ident, indices, values, flush=True)
    assert all(p._version == versions[n] for n,p in model.named_parameters()), 'Parameters mutated'
    result = dict(status='PASS', id=ident, optimizer_steps=0, parameter_versions_unchanged=True,
        archive_sha256=marker['archive_sha256'], checkpoint_revision=marker['backup']['revision'],
        validation_rows=len(rows), scores=scores, gradients=gradients, residual_norms=residual_norms,
        scope='Post-hoc frozen checkpoint eval-mode probes. Not training-time conflict, optimizer update or retrained ablation evidence.',
        worker_sha256=sha256(Path(__file__)))
    write(folder/'RESULT.json', result)
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    for target in (export, folder/'restore'):
        assert target.resolve().is_relative_to(out.resolve())
        shutil.rmtree(target)
    write(folder/'CLEANUP.json', dict(temporary_weights_removed=True, original_backup_verified=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-free-gpu-mib', type=int, default=20000)
    args = parser.parse_args()
    assert ROOT == Path('/workspace/ura/thanhdat/projects/CLRR')
    assert Path(sys.prefix).resolve().is_relative_to(ROOT/'conda')
    cohort, out = args.cohort.resolve(), args.output.resolve()
    assert cohort.is_relative_to(ROOT/'outputs/experiments') and out.is_relative_to(ROOT/'outputs/experiments')
    engine.configure(cohort)
    # The original numerical method files must match their original bundle;
    # current registry/diagnostic glue has its own frozen control bundle.
    manifest = read(cohort/'bundle_manifest.json')
    for name in ('src/methods/alignment_residual/v1/model.py', 'src/methods/alignment_residual/v1/module.py'):
        assert sha256(ROOT/name) == manifest[name], name
    for key, value in engine.plan()['runtime'].items():
        if key in ('torch','transformers','sacrebleu'):
            version = {'torch':torch.__version__.split('+')[0], 'transformers':engine.transformers.__version__,
                       'sacrebleu':engine.sacrebleu.__version__}[key]
            assert version == value
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=ROOT/'data_processed'/engine.plan()['dataset']/'validation.csv'
    assert sha256(data) == engine.plan()['data']['validation']['sha256']
    rows=pd.read_csv(data,dtype=str,keep_default_na=False)
    batches=np.random.default_rng(42).permutation(len(rows))[:16].reshape(4,4).tolist()
    out.mkdir(parents=True,exist_ok=True)
    design=dict(ids=['AR_C0','AR_C1','AR_C2'],validation_sha256=sha256(data),
        gradient_batches=batches,optimizer_steps=0,worker_sha256=sha256(Path(__file__)),
        probe_sha256=sha256(ROOT/'src/methods/alignment_residual/controls/probes.py'))
    if (out/'PLAN.json').exists():
        assert read(out/'PLAN.json') == design
    else:
        write(out/'PLAN.json',design)
    for ident in design['ids']:
        measure(args,ident,cohort,out,rows,batches)
    write(out/'COMPLETE.json',dict(status='COMPLETE',ids=design['ids'],optimizer_steps=0,completed_unix=time.time()))
    print('DIAGNOSTICS_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
