"""Independently verify public HTTP inventory, then publish current storage docs."""
import json
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
from huggingface_hub import HfApi, CommitOperationAdd
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner.colab import get_hf_token

REPO = 'FiveC/amis-rewire-checkpoints'
OUT = ROOT / 'outputs_rebuttal/hf_seed_layout_20261008'


def main():
    plan = json.loads((OUT / 'mapping.json').read_text(encoding='utf-8'))
    receipt = json.loads((OUT / 'FINAL_RECEIPT.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'COMPLETE_VERIFIED'
    response = requests.get(f'https://huggingface.co/api/models/{REPO}?blobs=true', timeout=60)
    response.raise_for_status()
    remote = response.json()
    assert not remote['private']
    files = {e['rfilename']: e for e in remote['siblings']}
    for item in plan['files']:
        assert item['source_path'] not in files
        entry = files[item['destination']]
        assert entry['size'] == item['bytes']
        assert entry.get('lfs', {}).get('sha256', entry['blobId']) == item['digest']
    for path in files:
        if path.startswith('checkpoints/'):
            assert path.split('/')[4].startswith('seed-'), path
            assert '/seed-unknown/' not in path, path
    assert sum(p.startswith('checkpoints/') and p.endswith('/best.zip') for p in files) == 90
    assert 'checkpoints/amis_mandarin/mbart-large-50/bitfit/seed-42/best.zip' in files
    report = {'status': 'PUBLIC_HTTP_INDEPENDENT_VERIFICATION_PASS', 'revision': remote['sha'],
              'public': True, 'files_identical': len(plan['files']), 'archives_identical': 90,
              'verified_unknown_seeds': plan['resolved_unknown_seeds'],
              'method_seed_folders': plan['primary_count'], 'training_started': False}
    guide = ROOT / 'docs/operations/HUGGINGFACE_CHECKPOINT_STORAGE.md'
    text = guide.read_text(encoding='utf-8')
    heading = '## Cập nhật cấu trúc seed — 08/10/2026'
    if heading not in text:
        text += '\n\n' + heading + '\n\n**HOÀN TẤT, xác minh độc lập qua API public.** Đã chuyển 3.056 file và giữ nguyên hash của 90 archive; mở dataset/backbone/method là thấy seed. Có 79 nhóm method/seed; các bản cùng nhóm được giữ trong thư mục cấu hình bên trong seed. Cả 16 archive từng ghi `seed-unknown` đã xác minh seed42 bằng `training_args.bin`, không suy đoán từ tên run. Không train, không đổi metrics, không đổi actual patience.\n\nCác mô tả `historical-*`/`seed-unknown` ở phần migration trước là trạng thái lịch sử. Registry hiện hành: `seed_layout_mapping.json`, `seed_layout_seed_evidence.json`, `SEED_LAYOUT_VERIFIED.json`, `SEED_LAYOUT_PUBLIC_VERIFICATION.json` và `checkpoint_index.json/csv`. `metadata_complete=null` trong index mới nghĩa là bước di chuyển không đánh giá lại tính đầy đủ metadata; xác minh seed không đồng nghĩa đủ endpoint/training provenance.\n\nCông cụ chuyển: `scripts/revalidation/flatten_hf_seed_layout.py`; công cụ kiểm tra: `scripts/revalidation/verify_hf_seed_layout.py`, chạy trong conda `clrr`. Script chuyển dành cho snapshot đã khóa, không chạy lại lên cây đã chuyển. Công cụ index lịch sử yêu cầu cấu trúc cũ; không chạy lại để ghi đè index hiện hành.\n'
        guide.write_text(text, encoding='utf-8')
    handoff = ROOT / 'docs/operations/NEXT_EXPERIMENTS_HANDOFF_20261005.md'
    text = handoff.read_text(encoding='utf-8')
    heading = '## Lưu checkpoint hiện hành — 08/10/2026'
    if heading not in text:
        first, rest = text.split('\n', 1)
        text = first + '\n\n' + heading + '\n\n**HF_LAYOUT_VERIFIED; không training trong bước này.** Repo public duy nhất `FiveC/amis-rewire-checkpoints`. Cấu trúc: `checkpoints/<dataset>/<backbone>/<method>/seed-<n>/best.zip`; nếu cùng seed có nhiều bản khác nhau thì thêm `<configuration>/` bên trong seed. Đã chuyển 3.056 file, giữ nguyên hash 90 archive/metrics, 79 nhóm method/seed. Đã xác minh 16 archive `seed-unknown` thực sự seed42 từ training args; BitFit/Middle/Narrow LoRA mở trực tiếp tới seed42. Không sửa actual ES4 thành ES10.\n\nQuy tắc này thay thế cấu trúc recipe/cohort trước seed trong các mục lịch sử phía dưới. Recipe/cohort vẫn ở metadata, mapping và receipts. Run mới cần `run_metadata.json` đầy đủ; không overwrite archive khác hash. Tên biến thể phải mô tả cấu hình, không có ngày tháng. Strong LoRA A/B và các đối sánh LSR λ có nhãn rõ trong index. Tra cứu [hướng dẫn](HUGGINGFACE_CHECKPOINT_STORAGE.md) và [index public](https://huggingface.co/FiveC/amis-rewire-checkpoints/blob/main/registry/CHECKPOINT_INDEX.md). Metadata seed được phục hồi không tự giải quyết thiếu endpoint; kế hoạch chuẩn hóa đối chứng vẫn **PLAN_NOT_LAUNCHED**, cần kiểm reuse gate trước khi train.\n' + rest
        handoff.write_text(text, encoding='utf-8')
    readme = '''# CLRR checkpoints

## Canonical checkpoint storage

Public project checkpoint repository. Open a method to see its seeds directly:

`checkpoints/<dataset>/<backbone>/<method>/seed-<n>/best.zip`

When the same method/seed has multiple distinct configurations, each archive is stored inside a configuration folder below that seed. Configuration names describe the actual method or protocol; dates belong in metadata. Never overwrite different weights. Strong LoRA A/B and LSR lambda comparators have explicit method labels in the index.

The verified reorganization preserved all 90 final archives and all existing score-bearing file contents. All 16 previously unknown archive seeds were verified as 42 from archived training arguments. Actual ES4/ES10 settings remain unchanged. Recovering a seed does not establish complete training provenance.

- [Current checkpoint table](registry/CHECKPOINT_INDEX.md)
- [CSV index](registry/checkpoint_index.csv)
- [Storage guide](registry/STORAGE_GUIDE.md)
- [Current old-to-new mapping](registry/seed_layout_mapping.json)
- [Seed evidence](registry/seed_layout_seed_evidence.json)
- [Public independent verification](registry/SEED_LAYOUT_PUBLIC_VERIFICATION.json)
- [Checkpoints](checkpoints)

## Provenance

Original recipes, cohorts and receipts remain available beside each checkpoint and in the registry. New runs require complete run_metadata.json. The index primary flag recommends the corrected implementation rather than selecting on test scores. Layout receipts record migration provenance, not new training results.

Earlier path mappings and migration receipts remain historical snapshots: resolve their destinations through the current seed layout mapping. Metadata-only development reports are under registry/reports/development. Only selected final archives remain in the current tree; this does not claim deletion of historical Git blobs or permanent storage reclamation.
'''
    (OUT / 'PUBLIC_VERIFICATION.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    api = HfApi(token=get_hf_token())
    assert api.model_info(REPO).sha == remote['sha'], 'Remote changed during final verification'
    result = api.create_commit(REPO, parent_commit=remote['sha'], commit_message='Publish current seed layout and independent public verification', operations=[
        CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=readme.encode('utf-8')),
        CommitOperationAdd(path_in_repo='registry/STORAGE_GUIDE.md', path_or_fileobj=str(guide)),
        CommitOperationAdd(path_in_repo='registry/SEED_LAYOUT_PUBLIC_VERIFICATION.json', path_or_fileobj=str(OUT / 'PUBLIC_VERIFICATION.json'))])
    print('VERIFIED_AND_DOCUMENTED', result.oid, json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
