"""Run final test checkpoint evaluation across 8 newly trained models with FP32 beam 4."""
from pathlib import Path
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

import pandas as pd
import sacrebleu
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from amis_rewire.modeling import CrossLayerResidualRewire
from amis_rewire.evaluation import generation_precision

OUT = ROOT / 'outputs_rebuttal/amis_final_evaluation_20261004'
OUT.mkdir(parents=True, exist_ok=True)

REPO = 'FiveC/amis-rewire-checkpoints'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def predict(model, tokenizer, rows, max_length=256, batch_size=8):
    predictions = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = tokenizer(
            [r['source'] for r in chunk],
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors='pt'
        )
        batch = {k: v.cuda() for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad(), generation_precision(model, 'fp32'):
            ids = model.generate(
                **batch,
                num_beams=4,
                max_length=max_length,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids('zh_CN')
            )
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
        predictions.extend(p.strip() for p in decoded)
        if start % 80 == 0:
            print(f'GENERATED {min(start + len(ids), len(rows))} / {len(rows)}', flush=True)
    return predictions


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    manifest_path = OUT / 'final_evaluation_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    test_path = ROOT / manifest['protocol']['test_dataset']
    test_df = pd.read_csv(test_path, encoding='utf-8', keep_default_na=False).fillna('')
    assert len(test_df) == 575
    test_rows = test_df.to_dict('records')
    targets = [r['target'] for r in test_rows]

    api = HfApi(token=os.environ.get('HF_TOKEN'))
    assert api.whoami()['name'] == 'FiveC'

    results = {}

    for item in manifest['models_to_evaluate']:
        model_id = item['id']
        run_folder = OUT / 'runs' / model_id
        run_folder.mkdir(parents=True, exist_ok=True)

        if (run_folder / 'test_metrics.json').exists():
            print(f'ALREADY_DONE: {model_id}', flush=True)
            results[model_id] = json.loads((run_folder / 'test_metrics.json').read_text())
            continue

        print(f'\n=== EVALUATING {model_id} (seed={item["seed"]}, alpha={item["alpha"]}, lambda={item["lambda"]}) ===', flush=True)

        # 1. Download zip archive from HF
        print(f'Downloading {item["path"]}...', flush=True)
        local_zip = hf_hub_download(
            repo_id=REPO,
            filename=item['path'],
            token=os.environ.get('HF_TOKEN')
        )
        assert digest(Path(local_zip)) == item['sha256'], f'Archive SHA mismatch for {model_id}'

        # 2. Extract checkpoint
        extract_dir = ROOT / 'scratch' / 'eval_extracted' / model_id
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(local_zip) as z:
            z.extractall(extract_dir)

        # Find model directory
        ckpt_dir = extract_dir
        if (extract_dir / 'best_model').exists():
            ckpt_dir = extract_dir / 'best_model'
        elif any(extract_dir.glob('checkpoint-*')):
            ckpt_dir = next(extract_dir.glob('checkpoint-*'))

        # 3. Load model and tokenizer
        print(f'Loading model from {ckpt_dir}...', flush=True)
        raw_model, loading = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float32,
            output_loading_info=True,
            local_files_only=True
        )
        for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']:
            assert not loading.get(key), (model_id, key, loading.get(key))

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True, use_fast=True)
        tokenizer.src_lang = 'tl_XX'
        tokenizer.tgt_lang = 'zh_CN'

        if item['alpha'] > 0:
            model = CrossLayerResidualRewire(raw_model, distance=2, strength=item['alpha'], stack='encoder')
        else:
            model = raw_model

        model = model.cuda().eval()
        model.config.use_cache = True

        # 4. Generate predictions
        t0 = time.time()
        preds = predict(model, tokenizer, test_rows, max_length=256, batch_size=8)
        elapsed = time.time() - t0

        # 5. Compute metrics
        bleu = sacrebleu.corpus_bleu(preds, [targets], tokenize='zh').score
        chrf_pp = sacrebleu.corpus_chrf(preds, [targets], word_order=2).score
        chrf_0 = sacrebleu.corpus_chrf(preds, [targets], word_order=0).score

        print(f'FINISHED {model_id} in {elapsed:.1f}s | BLEU: {bleu:.4f} | chrF++: {chrf_pp:.4f} | chrF: {chrf_0:.4f}', flush=True)

        # 6. Save predictions and metrics
        pred_records = [
            {'index': i, 'source': r['source'], 'target': r['target'], 'prediction': p}
            for i, (r, p) in enumerate(zip(test_rows, preds))
        ]
        pred_csv = run_folder / 'test_predictions.csv'
        with pred_csv.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'source', 'target', 'prediction'])
            writer.writeheader()
            writer.writerows(pred_records)

        metadata = {
            'id': model_id,
            'family': item['family'],
            'seed': item['seed'],
            'alpha': item['alpha'],
            'lambda': item['lambda'],
            'detach': item['detach'],
            'test_rows': len(preds),
            'test_bleu': bleu,
            'test_chrf++': chrf_pp,
            'test_chrf0': chrf_0,
            'elapsed_seconds': elapsed,
            'prediction_sha256': digest(pred_csv),
            'archive_sha256': item['sha256']
        }
        (run_folder / 'test_metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        results[model_id] = metadata

        # 7. Upload to HF
        for artifact in [run_folder / 'test_predictions.csv', run_folder / 'test_metrics.json']:
            # Path on HF: e.g. amis_final_evaluation_20261004/runs/{model_id}/{name}
            # Also upload into the original run folder on HF if desired
            api.upload_file(
                path_or_fileobj=str(artifact),
                path_in_repo=f'amis_final_evaluation_20261004/runs/{model_id}/{artifact.name}',
                repo_id=REPO
            )

        # 8. Clean up
        del model, raw_model
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(extract_dir)

    summary_path = OUT / 'test_evaluation_summary.json'
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo='amis_final_evaluation_20261004/test_evaluation_summary.json',
        repo_id=REPO
    )
    print('\n=== ALL 8 TEST EVALUATIONS COMPLETE AND UPLOADED TO HF ===', flush=True)


if __name__ == '__main__':
    main()
