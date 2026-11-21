"""Pinned frozen-checkpoint logit diagnostics, never optimizer or test selection."""
import argparse
import gc
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from experiment_runner import engine
from experiment_runner.config import verify_frozen
from experiment_runner.integrity import restore_archive, sha256
from methods.decoupled_consistency.v1.model import token_components


def measure_logits(left, right, labels):
    """Sufficient statistics; gradients in logit space, not parameter/Adam updates."""
    mask = labels.ne(-100)
    a = left[mask].detach().requires_grad_()
    b = right[mask].detach().requires_grad_()
    y = labels[mask]
    ce, gold, rest = token_components(a, b, y)
    objectives = dict(ce=ce.sum(), gold=2.5*gold.sum(), rest=2.5*rest.sum(),
                      full=2.5*(gold+rest).sum(), weak_full=1.25*(gold+rest).sum())
    gradients = {name: torch.autograd.grad(value, (a, b), retain_graph=True)
                 for name, value in objectives.items()}
    stats = dict(tokens=len(y), ce=float(ce.detach().sum()),
                 gold=float(gold.detach().sum()), rest=float(rest.detach().sum()))
    for name, grads in gradients.items():
        stats[name+'_grad_sq'] = sum(float(g.double().square().sum()) for g in grads)
        stats[name+'_ce_dot'] = sum(float((g.double()*c.double()).sum())
                                   for g, c in zip(grads, gradients['ce']))
    with torch.no_grad():
        pa, pb = a.softmax(-1), b.softmax(-1)
        ca, cb = a.argmax(-1), b.argmax(-1)
        stats.update(gold_probability_sum=float(.5*(pa.gather(-1,y[:,None])+
                                                   pb.gather(-1,y[:,None])).sum()),
                     agreement=int(ca.eq(cb).sum()),
                     both_correct=int((ca.eq(y)&cb.eq(y)).sum()),
                     agreed_wrong=int((ca.eq(cb)&ca.ne(y)).sum()))
        margins = []
        for logits in (a, b):
            other = logits.detach().clone().scatter_(-1, y[:,None], -torch.inf)
            margins.append(logits.gather(-1,y[:,None]).squeeze(-1)-other.amax(-1))
        stats['gold_logit_margin_sum'] = float(.5*(margins[0]+margins[1]).sum())
    return stats


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    out, plan = verify_frozen(args.config)
    engine.configure(out)
    engine.validate_environment(gpu=True)
    probe = plan['pre_training_consistency_probe']
    destination = out/'frozen_diagnostics'
    destination.mkdir(exist_ok=True)
    frame = pd.read_csv(ROOT/'data_processed'/plan['dataset']/'validation.csv', keep_default_na=False)
    frame = frame.iloc[probe['validation_rows']]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for checkpoint in probe['checkpoints']:
        result_path = destination/(checkpoint['id']+'.json')
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            assert saved['checkpoint'] == checkpoint and saved['bundle_sha256'] == sha256(out/'bundle_manifest.json')
            continue
        assert shutil.disk_usage(out).free >= 40*2**30, 'Need 40 GiB before checkpoint restore'
        temporary = destination/(checkpoint['id']+'_restore')
        temporary.mkdir(exist_ok=False)
        archive = Path(hf_hub_download(plan['repo'], checkpoint['prefix'].rstrip('/')+'/best.zip',
            revision=checkpoint['revision'], token=False, local_dir=temporary/'download'))
        assert sha256(archive) == checkpoint['archive_sha256']
        export = temporary/'export'
        restore_archive(archive, export)
        model, tokenizer, _ = engine.load_model(checkpoint['spec'], export)
        model.gradient_checkpointing_disable()
        model.cuda().train()
        spec = checkpoint['spec']
        records = []
        for seed in probe['dropout_seeds']:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            sums = {}
            for start in range(0, len(frame), 2):
                rows = frame.iloc[start:start+2]
                inputs = tokenizer(rows.source.tolist(), padding=True, pad_to_multiple_of=8,
                    truncation=True, max_length=spec['max_source_length'], return_tensors='pt')
                target = tokenizer(text_target=rows.target.tolist(), padding=True, pad_to_multiple_of=8,
                    truncation=True, max_length=spec['max_target_length'], return_tensors='pt')
                inputs = {k:v.cuda() for k,v in inputs.items() if k in ('input_ids','attention_mask')}
                labels = target['input_ids'].cuda().masked_fill(target['attention_mask'].cuda().eq(0), -100)
                decoder = model.prepare_decoder_input_ids_from_labels(labels)
                # Base model: two native routes, with no model graph or optimizer.
                core = getattr(model, 'base_model', model) if hasattr(model, '_route') else model
                with torch.no_grad():
                    a = core(**inputs, decoder_input_ids=decoder, use_cache=False).logits
                    b = core(**inputs, decoder_input_ids=decoder, use_cache=False).logits
                for offset in range(0, labels.shape[1], 8):
                    stats = measure_logits(a[:,offset:offset+8], b[:,offset:offset+8], labels[:,offset:offset+8])
                    for key, value in stats.items():
                        sums[key] = sums.get(key, 0)+value
                del a, b
            n = sums['tokens']
            normalized = {k:sums[k]/n for k in ('ce','gold','rest','gold_probability_sum',
                          'gold_logit_margin_sum','agreement','both_correct','agreed_wrong')}
            for name in ('gold','rest','full','weak_full'):
                auxiliary, ce_sq = sums[name+'_grad_sq'], sums['ce_grad_sq']
                normalized[name+'_logit_grad_norm'] = math.sqrt(auxiliary)/n
                normalized[name+'_over_ce_norm'] = math.sqrt(auxiliary/ce_sq) if ce_sq else None
                normalized[name+'_ce_cosine'] = sums[name+'_ce_dot']/math.sqrt(auxiliary*ce_sq) if auxiliary*ce_sq else None
            assert all(v is None or math.isfinite(v) for v in normalized.values())
            records.append(dict(dropout_seed=seed, supervised_tokens=n, **normalized))
        engine.write(result_path, dict(checkpoint=checkpoint, validation_rows=probe['validation_rows'],
            bundle_sha256=sha256(out/'bundle_manifest.json'), records=records, optimizer_steps=0,
            test_used=False, gradient_space='logits; not parameter gradients or accumulated Adam updates'))
        del model, core, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        # Temporary downloads are pinned HF copies, never the only checkpoint.
        assert temporary.resolve().is_relative_to(destination.resolve())
        shutil.rmtree(temporary)
    engine.write(destination/'COMPLETE.json', dict(passed=True, optimizer_steps=0,
        ids=[c['id'] for c in probe['checkpoints']], bundle_sha256=sha256(out/'bundle_manifest.json')))


if __name__ == '__main__':
    main()
