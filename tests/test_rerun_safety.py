"""Offline checks preventing expensive wrong resumes and unverified backups."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from amis_rewire import train


class RerunSafety(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.data=self.root/'data'
        self.data.mkdir()
        for split in ('train','validation','test'):
            (self.data/f'{split}.csv').write_text('source,target\na,b\n')
        with patch.object(sys,'argv',['train','--data-dir',str(self.data),'--no-bf16']):
            self.args=train.parse_args()
        self.output=self.root/'out'

    def tearDown(self):self.tmp.cleanup()

    def checkpoint(self):
        train.check_training_run(self.args,self.output,self.data)
        path=self.output/'checkpoint-1'
        path.mkdir()
        shutil.copyfile(self.output/'training_receipt.json',path/'training_receipt.json')
        for name in ('optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth','config.json','pytorch_model.bin'):
            (path/name).write_text('{}')
        return path

    def test_no_implicit_resume_or_overwrite(self):
        self.assertFalse(self.args.auto_resume)
        self.checkpoint()
        with self.assertRaises(FileExistsError):train.check_training_run(self.args,self.output,self.data)

    def test_compatible_resume(self):
        self.args.resume_from_checkpoint=str(self.checkpoint())
        self.assertEqual(train.check_training_run(self.args,self.output,self.data),self.args.resume_from_checkpoint)

    def test_rejects_old_code_receipt(self):
        path=self.checkpoint()
        receipt=json.loads((path/'training_receipt.json').read_text())
        receipt['source_sha256']['modeling.py']='historical-without-correction'
        (path/'training_receipt.json').write_text(json.dumps(receipt))
        self.args.resume_from_checkpoint=str(path)
        with self.assertRaisesRegex(ValueError,'receipt'):train.check_training_run(self.args,self.output,self.data)

    def test_rejects_recipe_change(self):
        self.args.resume_from_checkpoint=str(self.checkpoint())
        self.args.learning_rate*=2
        with self.assertRaisesRegex(ValueError,'receipt'):train.check_training_run(self.args,self.output,self.data)

    def test_rejects_data_change(self):
        self.args.resume_from_checkpoint=str(self.checkpoint())
        (self.data/'train.csv').write_text('source,target\nc,d\n')
        with self.assertRaisesRegex(ValueError,'receipt'):train.check_training_run(self.args,self.output,self.data)

    def test_requires_resumable_state(self):
        path=self.checkpoint()
        (path/'optimizer.pt').unlink()
        self.args.resume_from_checkpoint=str(path)
        with self.assertRaisesRegex(ValueError,'optimizer'):train.check_training_run(self.args,self.output,self.data)

    def test_validation_only_never_reads_test(self):
        self.args.validation_only=True
        (self.data/'test.csv').unlink()
        train.check_training_run(self.args,self.output,self.data)
        receipt=json.loads((self.output/'training_receipt.json').read_text())
        self.assertEqual(set(receipt['data_sha256']),{'train','validation'})

    def backup(self,upload_success=True,bad_checksum=False):
        best=self.root/'best'
        best.mkdir()
        (best/'pytorch_model.bin').write_bytes(b'weights')
        api=Mock()
        def paths(*args,**kwargs):
            archive=self.root/'backups/run/run-best.zip'
            sha=hashlib.sha256(archive.read_bytes()).hexdigest()
            return [SimpleNamespace(size=archive.stat().st_size,lfs=SimpleNamespace(sha256='wrong' if bad_checksum else sha))]
        api.get_paths_info.side_effect=paths
        with patch.dict('os.environ',{'HF_TOKEN':'synthetic-not-a-credential'}),patch.object(train,'HfApi',return_value=api),\
             patch.object(train,'safe_hf_upload',return_value=upload_success):
            train.archive_best_model(best,self.root/'backups','run','test/repo','new-isolated-prefix')
        return api

    def test_verified_upload(self):self.backup().get_paths_info.assert_called_once()

    def test_failed_upload_stops_and_retains_archive(self):
        with self.assertRaisesRegex(RuntimeError,'upload failed'):self.backup(upload_success=False)
        self.assertTrue((self.root/'backups/run/run-best.zip').exists())

    def test_corrupt_upload_stops(self):
        with self.assertRaisesRegex(RuntimeError,'SHA256'):self.backup(bad_checksum=True)


if __name__=='__main__':unittest.main()
