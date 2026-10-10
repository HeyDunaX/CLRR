"""Test only a preselected checkpoint after the joint 14-run selection lock."""
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import run_standardized_controls as worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    args = parser.parse_args()
    lock_path = worker.ROOT / 'outputs_rebuttal/amis_standardized_controls/selection_lock_14.json'
    lock = json.loads(lock_path.read_text())
    assert lock['status'] == 'ALL_14_VALIDATION_VERIFIED' and len(lock['runs']) == 14
    assert lock['test_used_for_selection'] is False
    found = []
    for name in ['amis_standardized_controls', 'amis_standardized_strong_mbart']:
        folder = worker.ROOT / 'outputs_rebuttal' / name
        plan_path = folder / 'PLAN.json'
        assert worker.sha256(plan_path) == lock['cohort_proofs'][name]['plan_sha256']
        for spec in json.loads(plan_path.read_text())['runs']:
            if spec['id'] == args.id:
                found.append((folder, spec))
    assert len(found) == 1
    worker.OUT, spec = found[0]
    worker.validate_environment()
    selected = lock['runs'][spec['id']]
    assert selected['archive_path'] == spec['remote_prefix'] + '/best.zip'
    folder = worker.OUT / 'runs' / spec['id']
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'COMPLETE.json').exists(), 'Already tested; collect existing result instead'
    archive = Path(worker.hf_hub_download(worker.REPO, selected['archive_path'],
        revision=selected['revision'], token=worker.os.environ['HF_TOKEN'], local_dir=folder / 'restore'))
    assert worker.sha256(archive) == selected['archive_sha256']
    export = folder / 'best_model'
    worker.restore_archive(archive, export)
    worker.set_seed(spec['seed'])
    model, tokenizer, _ = worker.load_model(spec, export)
    raw, dataset = worker.dataset('test', tokenizer, spec)
    training_args = worker.arguments(folder / 'test_stage', spec)
    training_args.gradient_checkpointing = False
    training_args.load_best_model_at_end = False
    training_args.eval_strategy = 'no'
    trainer = worker.ControlTrainer(model=model, args=training_args, processing_class=tokenizer,
        data_collator=worker.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8),
        generation_dtype='fp32')
    scores = worker.predict(trainer, dataset, tokenizer, raw, 'test', folder, 4)
    frame = worker.pd.read_csv(folder / 'test_predictions.csv', dtype=str, keep_default_na=False)
    replay = worker.generation_metrics(frame['prediction'].tolist(), frame['target'].tolist(), tokenize='zh')
    assert all(abs(replay[k] - scores[k]) < 1e-10 for k in replay)
    receipt = worker.upload_and_verify(folder, spec, ['test_metrics.json', 'test_predictions.csv'])
    worker.write(folder / 'COMPLETE.json', {'status': 'COMPLETE_VERIFIED', 'id': spec['id'],
        'scores': scores, 'test_backup': receipt, 'selection': selected,
        'joint_selection_lock_sha256': worker.sha256(lock_path), 'independent_metric_replay': True})
    worker.upload_and_verify(folder, spec, ['COMPLETE.json'])
    del trainer, model
    gc.collect()
    worker.torch.cuda.empty_cache()
    for target in [export, folder / 'restore']:
        assert target.resolve().is_relative_to(folder.resolve())
        shutil.rmtree(target)
    print('COMPLETE_VERIFIED', spec['id'], json.dumps(scores), flush=True)


if __name__ == '__main__':
    main()
