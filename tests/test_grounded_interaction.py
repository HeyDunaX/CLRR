"""Correctness gates for the prototype, without dataset optimization."""
import json
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from transformers import MBartForConditionalGeneration
from test_residual_effect import backbone, batch
from methods.grounded_interaction.v1.model import (
    GroundedInteractionSeq2SeqLM, source_negatives, contrastive_objective)
from methods.registry import wrap_method

DEVICE = os.environ.get('CLRR_GATE_DEVICE', 'cpu')


class InteractionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, mode='interaction', strength=.1, dropout=True):
        torch.manual_seed(42)
        core = backbone()
        if not dropout:
            for m in core.modules():
                if isinstance(m, torch.nn.Dropout): m.p = 0.
            # mBART also uses F.dropout with numeric attributes.
            for m in core.modules():
                for name in ('dropout', 'attention_dropout', 'activation_dropout'):
                    if isinstance(getattr(m, name, None), float): setattr(m, name, 0.)
        return GroundedInteractionSeq2SeqLM(core, mode=mode, strength=strength).to(DEVICE)

    def inputs(self):
        return {k: v.to(DEVICE) for k, v in batch().items()}

    def test_formula_and_four_gradient_signs(self):
        scores = {key: torch.tensor([value], device=DEVICE, requires_grad=True)
            for key, value in dict(on_match=-1., on_mismatch=-3., off_match=-2., off_mismatch=-3.).items()}
        valid = torch.tensor([True], device=DEVICE)
        loss = contrastive_objective(scores, valid, 1., 'interaction')
        torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(-1., device=DEVICE)))
        loss.backward()
        for key, sign in dict(on_match=-1, on_mismatch=1, off_match=1, off_mismatch=-1).items():
            self.assertEqual(scores[key].grad.sign().item(), sign)
        # The OFF gradient terms make this different from ON-only source contrast.
        source = contrastive_objective(scores, valid, 1., 'source')
        self.assertIsNone(torch.autograd.grad(source, scores['off_match'], allow_unused=True)[0])

    def test_negative_mask_and_singleton(self):
        b = self.inputs()
        indices, valid = source_negatives(**b)
        self.assertEqual(indices.tolist(), [1, 0]); self.assertTrue(valid.all())
        duplicate = {k: v[:1].repeat(2, 1) for k, v in b.items()}
        self.assertFalse(source_negatives(**duplicate)[1].any())
        for key in ('input_ids', 'labels'):
            duplicate = {k: v.clone() for k, v in b.items()}
            duplicate[key][1] = duplicate[key][0]
            if key == 'input_ids': duplicate['attention_mask'][1] = duplicate['attention_mask'][0]
            self.assertFalse(source_negatives(**duplicate)[1].any())
        for mode in ('paired-ce', 'source', 'interaction'):
            model = self.make(mode)
            single = {k: v[:1] for k, v in b.items()}
            model(**single).loss.backward()
            self.assertEqual(model.last_diagnostics['valid_negatives'], 0)

    def test_checkpointing_preserves_loss_gradients_rng_flags(self):
        for mode in ('paired-ce', 'source', 'interaction'):
            model = self.make(mode)
            model.train()
            flags = [m.training for m in model.modules()]
            torch.manual_seed(123)
            loss = model(**self.inputs()).loss
            loss.backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            rng = torch.random.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone() if DEVICE == 'cuda' else None
            model.zero_grad(set_to_none=True)
            model.gradient_checkpointing_enable({'use_reentrant': True})
            torch.manual_seed(123)
            cp_loss = model(**self.inputs()).loss
            cp_loss.backward()
            torch.testing.assert_close(cp_loss, loss)
            for name, p in model.named_parameters():
                if name in grads:
                    torch.testing.assert_close(p.grad, grads[name], atol=3e-6, rtol=4e-4)
                    self.assertTrue(torch.isfinite(p.grad).all())
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            if cuda_rng is not None: self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
            self.assertEqual(flags, [m.training for m in model.modules()])
            self.assertEqual(model.strength, .1)
            self.assertTrue(all(not v for v in model._cache.values()))
            self.assertGreater(model.base_model.model.decoder.layers[0].fc1.weight.grad.norm(), 0)

    def test_accumulation_four_matches_mean_of_microbatch_gradients(self):
        model = self.make(dropout=False)
        model.gradient_checkpointing_enable()
        expected = {}
        inputs = self.inputs()
        for k in range(4):
            b = {name: value.clone() for name, value in inputs.items()}
            b['input_ids'][:, 1] += k
            model.zero_grad(set_to_none=True)
            (model(**b).loss / 4).backward()
            for name, p in model.named_parameters():
                if p.grad is not None: expected[name] = expected.get(name, 0) + p.grad.clone()
        model.zero_grad(set_to_none=True)
        for k in range(4):
            b = {name: value.clone() for name, value in inputs.items()}
            b['input_ids'][:, 1] += k
            (model(**b).loss / 4).backward()
        for name, p in model.named_parameters():
            if name in expected: torch.testing.assert_close(p.grad, expected[name])
        self.assertFalse(model.accepts_loss_kwargs)

    def test_zero_route_null_auxiliary_gradient(self):
        model = self.make(strength=0., dropout=False)
        b = self.inputs()
        s = {key: model._scores(b['input_ids'].roll(1, 0) if mismatch else b['input_ids'],
            b['attention_mask'].roll(1, 0) if mismatch else b['attention_mask'], b['labels'], route, True)
            for key, route, mismatch in [('on_match', True, False), ('on_mismatch', True, True),
                                         ('off_match', False, False), ('off_mismatch', False, True)]}
        aux = contrastive_objective(s, torch.ones(2, dtype=torch.bool, device=DEVICE), 1., 'interaction')
        self.assertAlmostEqual(aux.item(), .69314718056, places=6)
        aux.backward()
        self.assertLess(max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None), 1e-7)

    def test_save_reload_eval_generation_and_registry(self):
        model = self.make(); model.eval()
        b = self.inputs()
        with torch.no_grad():
            expected = model(**b)
            gen = model.generate(input_ids=b['input_ids'], attention_mask=b['attention_mask'], max_new_tokens=3)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            meta = json.loads((Path(folder) / 'interaction_config.json').read_text())
            core, info = MBartForConditionalGeneration.from_pretrained(folder, output_loading_info=True)
            self.assertFalse(info['missing_keys'] or info['unexpected_keys'] or info['mismatched_keys'])
            restored = wrap_method(core, dict(method='route-source-interaction-v1', interaction_weight=meta['weight'],
                temperature=meta['temperature'], distance=meta['distance'], strength=meta['strength'])).to(DEVICE)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(**b).logits, expected.logits)
                torch.testing.assert_close(restored(**b).loss, expected.loss)
                torch.testing.assert_close(restored.generate(input_ids=b['input_ids'], attention_mask=b['attention_mask'],
                    max_new_tokens=3), gen)
            self.assertEqual(model.extra_parameter_count(), 0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in core.parameters()))


    def test_trainer_accumulation_and_best_checkpoint_reload(self):
        from transformers import Seq2SeqTrainingArguments
        from experiment_runner.engine import ControlTrainer
        model = self.make(dropout=False)
        with tempfile.TemporaryDirectory() as folder:
            trainer = ControlTrainer(model=model, args=Seq2SeqTrainingArguments(
                output_dir=folder, use_cpu=DEVICE == 'cpu', report_to=[],
                gradient_accumulation_steps=4, save_safetensors=False), generation_dtype='fp32')
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            b = self.inputs()
            model.zero_grad(set_to_none=True)
            direct = model(**b).loss
            direct.backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            trainer.current_gradient_accumulation_steps = 4
            scaled = trainer.training_step(model, dict(b), num_items_in_batch=torch.tensor(999))
            torch.testing.assert_close(scaled, direct.detach() / 4)
            for n, p in model.named_parameters():
                if n in grads: torch.testing.assert_close(p.grad, grads[n] / 4)
            model.eval()
            with torch.no_grad(): expected = model(**b).logits.clone()
            trainer._save(folder)
            self.assertEqual(json.loads((Path(folder) / 'interaction_config.json').read_text()), model.method_metadata())
            trainer.state.best_model_checkpoint = folder
            with torch.no_grad(): next(model.parameters()).add_(.01)
            trainer._load_best_model()
            with torch.no_grad(): torch.testing.assert_close(model(**b).logits, expected)

    def test_paired_ce_reduction_and_error_restoration(self):
        model = self.make(mode='paired-ce', dropout=False)
        b = self.inputs()
        on = model._scores(b['input_ids'], b['attention_mask'], b['labels'], True, False)
        off = model._scores(b['input_ids'], b['attention_mask'], b['labels'], False, False)
        count = b['labels'].ne(-100).sum(1)
        expected = -((on + off) * count).sum() / (2 * count.sum())
        torch.testing.assert_close(model(**b).loss, expected)
        flags = [m.training for m in model.modules()]
        with self.assertRaises(ValueError):
            model._scores(b['input_ids'], b['attention_mask'], b['labels'].fill_(-100), False, True)
        self.assertEqual(model.strength, .1)
        self.assertEqual(flags, [m.training for m in model.modules()])
        self.assertTrue(all(not v for v in model._cache.values()))

    def test_shared_runner_metadata_and_config_guards(self):
        from experiment_runner.config import read_config
        from experiment_runner.engine import wrap_model
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'configs/experiments/amis_residual_effect_rtx.json').read_text())
        original = copy.deepcopy(cfg['runs'][0])
        cfg['runs'] = []
        for method in ('route-paired-ce-v1', 'route-source-contrast-v1', 'route-source-interaction-v1'):
            spec = dict(original, id=method, method=method, jepa_weight=0., interaction_weight=.1,
                        temperature=1., distance=2, strength=.1)
            spec['remote_prefix'] = f"checkpoints/{cfg['dataset']}/{spec['backbone']}/{method}/seed-{spec['seed']}"
            cfg['runs'].append(spec)
            model, metadata = wrap_model(backbone(), spec)
            self.assertEqual(metadata['interaction_objective'], model.method_metadata())
        cfg['protocol']['view_checkpointing'] = 'whole-view-non-reentrant'
        with tempfile.TemporaryDirectory(dir=root / 'configs/experiments') as folder:
            path = Path(folder) / 'interaction_recipe.json'
            path.write_text(json.dumps(cfg))
            read_config(path)
            cfg['runs'][0]['jepa_weight'] = .1
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError): read_config(path)
            cfg['runs'][0]['jepa_weight'] = 0.
            del cfg['protocol']['view_checkpointing']
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError): read_config(path)


if __name__ == '__main__':
    unittest.main()
