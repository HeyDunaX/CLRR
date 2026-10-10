"""Recover and replay completed candidate runs from their public receipts."""
import json
import sys
from pathlib import Path
import pandas as pd
import sacrebleu
from huggingface_hub import HfApi, hf_hub_download
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from amis_rewire.metrics import generation_metrics
from fresh_confirmation_integrity import verify_remote_files, sha256
OUT = ROOT / 'outputs_rebuttal/amis_method_candidates'
REPO = 'FiveC/amis-rewire-checkpoints'


def main():
    api = HfApi()
    val = pd.read_csv(ROOT / 'data_processed/amis_mandarin/validation.csv', dtype=str, keep_default_na=False)
    for ident in ('U1C', 'U1F', 'U2L', 'U2F'):
        if not api.file_exists(REPO, next(x for x in json.loads((OUT / 'PLAN.json').read_text())['runs'] if x['id'] == ident)['remote_prefix'] + '/VAL_COMPLETE.json'):
            print(ident, 'PENDING_PUBLIC_VALIDATION_RECEIPT')
            continue
        plan = next(x for x in json.loads((OUT / 'PLAN.json').read_text())['runs'] if x['id'] == ident)
        prefix = plan['remote_prefix']
        files = ['VAL_COMPLETE.json', 'run_metadata.json', 'validation_predictions.csv']
        paths = {f'{prefix}/{name}': hf_hub_download(REPO, f'{prefix}/{name}') for name in files}
        receipt = json.loads(Path(paths[f'{prefix}/VAL_COMPLETE.json']).read_text())
        metadata = json.loads(Path(paths[f'{prefix}/run_metadata.json']).read_text())
        frame = pd.read_csv(paths[f'{prefix}/validation_predictions.csv'], dtype=str, keep_default_na=False)
        assert receipt['status'] == 'VALIDATION_COMPLETE_VERIFIED' and receipt['backup']['verified']
        assert receipt['id'] == ident and metadata['run']['seed'] == 42
        assert receipt['archive_sha256'] == metadata['archive_sha256']
        assert metadata['run']['method'] == plan['method']
        assert len(frame) == len(val) == 576
        assert frame['source'].tolist() == val['source'].tolist()
        assert frame['target'].tolist() == val['target'].tolist()
        scores = generation_metrics(frame['prediction'].tolist(), frame['target'].tolist(), tokenize='zh')
        scores['chrf0'] = sacrebleu.metrics.CHRF(word_order=0).corpus_score(
            frame['prediction'].tolist(), [frame['target'].tolist()]).score
        for key in ('bleu', 'chrf++', 'chrf0'):
            assert abs(scores[key] - receipt['scores'][key]) < 1e-9, (ident, key, scores[key], receipt['scores'][key])
        # Verify small metadata/predictions locally. Check the large model archive
        # using its Hub LFS digest and size; never download weights to this PC.
        verified = verify_remote_files(api, REPO, {
            f'{prefix}/{name}': paths[f'{prefix}/{name}'] for name in files
        })
        revision = verified['revision']
        archive_path = f'{prefix}/best.zip'
        archive_info = {x.path: x for x in api.get_paths_info(REPO, [archive_path], revision=revision)}[archive_path]
        assert archive_info.lfs.sha256 == receipt['archive_sha256'] == metadata['archive_sha256']
        assert archive_info.size == receipt['backup']['files'][archive_path]['size']
        verified['files'][archive_path] = {'size': archive_info.size, 'sha256': archive_info.lfs.sha256}
        verified['verified'] = True
        assert verified['verified']
        local = OUT / 'runs' / ident
        local.mkdir(parents=True, exist_ok=True)
        (local / 'VAL_COMPLETE.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        (local / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        (local / 'public_replay.json').write_text(json.dumps({'scores': scores, 'rows': len(frame),
            'source_target_order_match': True, 'independent_metric_replay': True,
            'public_files_verified': verified}, indent=2), encoding='utf-8')
        print(ident, json.dumps({'scores': scores, 'selected_epoch': metadata['selected_epoch'],
            'completed_epoch': metadata['completed_epoch'], 'patience_counter': metadata['stop_state']['patience_counter'],
            'remote_prefix': prefix, 'archive_sha256': metadata['archive_sha256'],
            'public_verified': True, 'metric_replay': True}))


if __name__ == '__main__': main()
