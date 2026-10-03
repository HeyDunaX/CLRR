"""Read public HF metadata/ZIP directories and local names without downloading weights."""
import io
import json
import zipfile
from pathlib import Path
from urllib.parse import quote
import requests
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs_rebuttal/amis_confirmation_20261003'


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.transferred = 0
        self.blocks = {}
        self.session = requests.Session()
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position
    def read(self, size=-1):
        if size < 0: size = self.size - self.position
        assert size <= 16 * 1024**2, 'Refusing a large member read'
        end = min(self.size, self.position + size)
        parts = []
        while self.position < end:
            block = self.position // 262144
            start = block * 262144
            if block not in self.blocks:
                last = min(self.size - 1, start + 262143)
                response = self.session.get(self.url, headers={'Range': f'bytes={start}-{last}'}, timeout=60)
                response.raise_for_status()
                assert response.status_code == 206 and len(response.content) == last - start + 1
                self.blocks[block] = response.content
                self.transferred += len(response.content)
            count = min(end - self.position, len(self.blocks[block]) - (self.position - start))
            parts.append(self.blocks[block][self.position-start:self.position-start+count])
            self.position += count
        return b''.join(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    repo = 'FiveC/amis-rewire-checkpoints'
    api = HfApi(token=False)
    revision = api.model_info(repo).sha
    files = [{'path': x.path, 'size': x.size} for x in api.list_repo_tree(repo, revision=revision, recursive=True)
             if hasattr(x, 'size')]
    candidates = [x for x in files if x['path'].endswith('.zip') and
                  ('mbart' in x['path'].lower() or 'revalidation' in x['path'].lower()) and
                  not any(k in x['path'].lower() for k in ['ashaninka', 'lora', 'middle', 'no-sg'])]
    archives = []
    for item in candidates:
        entry = {'path': item['path'], 'bytes': item['size']}
        try:
            remote = RangeFile(f'https://huggingface.co/{repo}/resolve/{revision}/{quote(item["path"])}', item['size'])
            with zipfile.ZipFile(remote) as archive:
                entry['weight_members'] = [x.filename for x in archive.infolist()
                    if x.filename.endswith(('.bin', '.safetensors')) and
                       Path(x.filename).name.startswith(('pytorch_model', 'model.'))]
                entry['metadata'] = {}
                for member in archive.infolist():
                    if Path(member.filename).name in ['trainer_state.json', 'metrics.json', 'run_manifest.json'] and member.file_size < 1024**2:
                        entry['metadata'][member.filename] = json.loads(archive.read(member))
            entry['metadata_bytes_transferred'] = remote.transferred
            entry['status'] = 'INSPECTED'
        except Exception as error:
            entry['status'] = 'UNINSPECTED_' + type(error).__name__
        archives.append(entry)
        print('ARCHIVE', entry['path'], entry['status'], 'weight_members', len(entry.get('weight_members', [])), flush=True)
    local = []
    for parent in ['checkpoints', 'outputs', 'backups', 'results', 'outputs_rebuttal']:
        for path in (ROOT / parent).rglob('*'):
            if path.is_file() and (path.name in ['pytorch_model.bin', 'model.safetensors', 'trainer_state.json']):
                local.append(str(path.relative_to(ROOT)))
    best = [x for x in files if '504' in x['path'] and 'mbart' in x['path'].lower() and 'clrr-only' in x['path'].lower()]
    embedded_best = [(x['path'], member) for x in archives for member in x.get('weight_members', [])
                     if '504' in member and 'clrr-only' in member.lower()]
    report = {'repo': repo, 'revision': revision, 'files': files, 'archives': archives,
              'local_weight_or_state_files': local, 'best504_direct_candidates': best,
              'best504_embedded_candidates': embedded_best,
              'task1': 'CANDIDATES_REQUIRE_STRICT_VERIFICATION' if best or embedded_best else 'BEST504_NOT_FOUND_IN_INSPECTED_LOCATIONS',
              'scope': 'No substitute checkpoint; archive failures are reported and not treated as inspected.'}
    (OUT / 'checkpoint_inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('TASK1', report['task1'], 'REVISION', revision, flush=True)


if __name__ == '__main__': main()
