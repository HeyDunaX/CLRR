"""Frozen bundle CPU gates and all four DRB exporter contracts."""
import copy
import tempfile
import unittest
import torch
from transformers import MBartForConditionalGeneration
from test_alignment_residual import native, batch
from test_decoder_residual_bridge import BridgeGate
from experiment_runner import engine as worker


def main():
    worker.validate_environment(gpu=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BridgeGate))
    assert result.wasSuccessful()
    cases=[]
    for spec in worker.plan()['runs']:
        config=copy.deepcopy(native().config)
        config.encoder_layers=8;config.decoder_layers=12
        config._attn_implementation='sdpa'
        model,meta=worker.wrap_model(MBartForConditionalGeneration(config),spec)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        model(**batch()).loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval();expected=model(**batch()).logits
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder,state_dict=model.state_dict())
            restored=type(model).from_pretrained(folder).eval()
            torch.testing.assert_close(expected,restored(**batch()).logits)
        cases.append(dict(id=spec['id'],passed=True,parameters=meta,save_reload_logits_match=True))
    worker.write(worker.OUT/'CPU_GATE.json',dict(passed=True,cases=cases,
        scientific_tests=result.testsRun,bundle_sha256=worker.sha256(worker.OUT/'bundle_manifest.json')))


if __name__=='__main__': main()
