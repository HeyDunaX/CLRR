"""Offline checks for frozen recipes, source coverage, and public fetch safety."""
import hashlib
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner import config, storage


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recipe = json.loads((ROOT / 'configs/experiments/amis_candidates.json').read_text())
        for path in ('src/methods/new_method.py', 'tests/gate.py', 'scripts/experiment.py',
                     'scripts/colab.py', 'requirements/colab-experiments.txt'):
            file = self.root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('# fixture\n')
        for split in self.recipe['data']:
            file = self.root / 'data_processed/amis_mandarin' / f'{split}.csv'
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('source,target\na,b\n')
            self.recipe['data'][split]['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
        self.path = self.root / 'configs/experiments/pilot.json'
        self.path.parent.mkdir(parents=True)
        self.save()
        self.root_patch = patch.object(config, 'ROOT', self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.recipe))

    def build(self):
        with patch.object(config.subprocess, 'check_output', return_value='fixture-commit'):
            return config.build(self.path)

    def test_bundle_covers_method_source_and_detects_changes(self):
        self.assertFalse(self.build()['gpu_started'])
        out, _ = config.verify_frozen(self.path)
        hashes = json.loads((out / 'bundle_manifest.json').read_text())
        self.assertIn('src/methods/new_method.py', hashes)
        (self.root / 'src/methods/new_method.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            config.verify_frozen(self.path)

    def test_frozen_recipe_and_workspace_cannot_be_overwritten(self):
        self.build()
        with self.assertRaises(FileExistsError): self.build()
        self.recipe['protocol']['patience'] = 11
        self.save()
        with self.assertRaisesRegex(ValueError, 'Configuration changed'):
            config.verify_frozen(self.path)

    def test_rejects_duplicate_upload_destinations_and_seed_mismatch(self):
        self.recipe['runs'][1]['remote_prefix'] = self.recipe['runs'][0]['remote_prefix']
        self.save()
        with self.assertRaisesRegex(ValueError, 'same destination'):
            config.read_config(self.path)
        self.recipe['runs'][0]['remote_prefix'] = 'checkpoints/amis_mandarin/mbart-large-50/x/seed-99'
        self.save()
        with self.assertRaisesRegex(ValueError, 'seed'):
            config.read_config(self.path)

    def test_rejects_remote_path_traversal(self):
        self.recipe['runs'][0]['remote_prefix'] += '/../other'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unsafe public'):
            config.read_config(self.path)

    def create_args(self, **overrides):
        values = dict.fromkeys(('lr', 'batch_size', 'accumulation', 'max_source_length',
            'max_target_length', 'validation_beams', 'src_lang', 'tgt_lang', 'model_id',
            'model_revision', 'middle_layer_idx', 'alignment_weight', 'temperature',
            'distance', 'strength', 'jepa_weight', 'source_prefix', 'stack', 'backbone',
            'dataset', 'epochs', 'patience', 'workers', 'bleu_tokenizer', 'storage_method',
            'configuration'))
        values.update(template=self.path, from_run=self.recipe['runs'][0]['id'],
            name='new_pilot', method='full', seed=43, account=2, allow_test=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_create_records_overrides_without_changing_template(self):
        original = self.path.read_bytes()
        result = config.create_recipe(self.create_args(lr=2e-5, batch_size=4,
            accumulation=8, patience=10, configuration='variant'))
        target = self.root / result['config']
        _, new = config.read_config(target)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(new['runs'][0]['learning_rate'], 2e-5)
        self.assertEqual(new['runs'][0]['microbatch'], 4)
        self.assertEqual(new['runs'][0]['accumulation'], 8)
        self.assertTrue(new['runs'][0]['remote_prefix'].endswith('/full/seed-43/variant'))
        self.assertTrue(new['protocol']['test_barrier'].startswith('No test'))
        self.assertFalse(result['gpu_started'])
        with self.assertRaises(FileExistsError): config.create_recipe(self.create_args())

    def test_create_rejects_unsupported_candidate_override(self):
        args = self.create_args(method='full-precache', distance=4)
        with self.assertRaisesRegex(ValueError, 'Frozen candidate'):
            config.create_recipe(args)
        self.assertFalse((self.root / 'configs/experiments/new_pilot.json').exists())

    def test_create_rejects_arguments_that_method_would_ignore(self):
        with self.assertRaisesRegex(ValueError, 'CLRR method'):
            config.create_recipe(self.create_args(method='baseline', distance=4))
        self.assertFalse((self.root / 'configs/experiments/new_pilot.json').exists())

    def test_fetch_uses_pinned_revision_and_checks_content(self):
        payload = self.root / 'metrics.json'
        payload.write_bytes(b'{"bleu":20}')
        index = self.root / 'index.json'
        record = {'local': 'outputs_rebuttal/old/metrics.json', 'size': payload.stat().st_size,
                  'remote': 'checkpoints/data/model/method/seed-42/metrics.json',
                  'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}
        index.write_text(json.dumps({'repo': config.REPO, 'revision': 'pinned', 'verified': [record]}))
        with patch.object(storage, 'INDEX', index), patch.object(storage, 'ROOT', self.root), \
             patch('huggingface_hub.hf_hub_download', return_value=str(payload)) as download:
            self.assertEqual(storage.fetch(record['local']), payload)
            self.assertEqual(download.call_args.kwargs['revision'], 'pinned')
            payload.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'): storage.fetch(record['local'])


if __name__ == '__main__': unittest.main()
