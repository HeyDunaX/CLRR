"""Train-only frozen W channel admission; zero optimizer steps, no test access."""
import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from experiment_runner import engine
from experiment_runner.config import verify_frozen
from experiment_runner.integrity import sha256, restore_archive
from .model import ContinuationConsistencyMT


def main(config):
    out, plan = verify_frozen(config)
    engine.configure(out)
    engine.wait_for_gpu()
    engine.validate_environment()
    dest = out/'continuation_probe'
    dest.mkdir(exist_ok=True)
    receipt = dest/'ADMISSION.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved['bundle_sha256'] == sha256(out/'bundle_manifest.json') and saved['passed']
        return
    checkpoint = plan['reused_w42']
    temporary = dest/'restore'
    temporary.mkdir(exist_ok=False)
    archive = Path(hf_hub_download(plan['repo'],checkpoint['prefix']+'/best.zip',
        revision=checkpoint['revision'],token=False,local_dir=temporary/'download'))
    assert sha256(archive) == checkpoint['archive_sha256']
    export = temporary/'export'
    restore_archive(archive,export)
    saved = json.loads((export/'decoupled_consistency_config.json').read_text())
    assert saved['method'] == 'weak-full-consistency-v1' and saved['beta'] == 1.25
    base = AutoModelForSeq2SeqLM.from_pretrained(export,attn_implementation='sdpa')
    model = ContinuationConsistencyMT(base,'continuation-consistency-v1').cuda().train()
    tokenizer = AutoTokenizer.from_pretrained(export,use_fast=True)
    spec = plan['runs'][0]
    engine.configure_languages(model,tokenizer,spec)
    _, data = engine.dataset('train',tokenizer,spec)
    rows = sorted(np.random.default_rng(314159).choice(len(data),32,replace=False).tolist())
    collate = engine.collator(model,tokenizer,spec)
    records = []
    for seed in (19421,19422):
        set_seed(seed)
        for start in range(0,len(rows),2):
            inputs = {k:v.cuda() for k,v in collate([data[i] for i in rows[start:start+2]]).items()}
            with torch.no_grad():
                model(**inputs)
            records.append(dict(seed=seed,rows=rows[start:start+2],**model.last_diagnostics))
            engine.write(dest/'PROGRESS.json',dict(stage='FROZEN_W_TRAIN_ONLY_PROBE',
                batches_done=len(records),batches=32,optimizer_steps=0))
    direct = sum(r['direct_kl'] for r in records)
    retention = sum(r['continuation_kl'] for r in records)/max(direct,1e-30)
    separation = float(np.mean([r['channel_separation'] for r in records]))
    # Technical scientific admission, not evidence of semantic equivalence or gain.
    passed = bool(np.isfinite(retention) and direct > 0 and 0 <= retention <= .99
                  and separation > 1e-6)
    result = dict(passed=passed,status='PASS' if passed else 'HOLD',
        checkpoint={k:checkpoint[k] for k in ('prefix','revision','archive_sha256')},
        split='train',rows=rows,optimizer_steps=0,test_used=False,
        retained_kl_ratio=retention,mean_channel_separation=separation,records=records,
        criteria='0 <= sum(CMC)/sum(direct SKL) <= .99; positive direct KL; channel separation > 1e-6',
        interpretation='Nontrivial, nonconstant kernel only; no semantic or translation-quality certification',
        bundle_sha256=sha256(out/'bundle_manifest.json'))
    engine.write(receipt,result)
    del model,base,tokenizer;gc.collect();torch.cuda.empty_cache()
    assert temporary.resolve().is_relative_to(dest.resolve())
    shutil.rmtree(temporary)
    print(json.dumps(result),flush=True)
    if not passed:
        raise RuntimeError('Frozen W channel admission HOLD; no gates/training launched')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    main(parser.parse_args().config)
