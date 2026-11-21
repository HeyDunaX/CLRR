"""Independent tiny-mBART gates for continuation CE/risk; no data or GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, Seq2SeqTrainingArguments
from test_residual_effect import backbone, batch
from methods.sequence_risk.v1.model import SequenceRiskSeq2SeqLM


class SequenceRiskPipelineGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, rewired=False, weight=.7, dropout=True, score_chunk=2):
        torch.manual_seed(42)
        core = backbone()
        if not dropout:
            for module in core.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
                for name in ('dropout', 'attention_dropout', 'activation_dropout'):
                    if isinstance(getattr(module, name, None), float):
                        setattr(module, name, 0.)
        return SequenceRiskSeq2SeqLM(core, rewired=rewired, weight=weight,
                                     alpha=.2, score_chunk=score_chunk)

    def inputs(self):
        result = batch()
        candidates = result['labels'][:, None].repeat(1, 3, 1)
        candidates[:, 1, 1] += 3
        candidates[:, 2, 2] += 4
        # A shorter valid candidate tests SUM likelihood and ignored padding.
        candidates[1, 1] = torch.tensor([0, 16, 2, -100, -100])
        candidates[0, 2] = -100
        result.update(candidate_labels=candidates,
            candidate_costs=torch.tensor([[0., .8, float('nan')], [.1, .4, .7]]),
            candidate_valid=torch.tensor([[True, True, False], [True, True, True]]))
        return result

    def independent_objective(self, model, values):
        # Forward each valid candidate separately; do not reuse model score helper.
        flags = [(m, m.training) for m in model.modules()]
        try:
            model.eval()
            with torch.no_grad():
                ce = model(**{k: values[k] for k in ('input_ids', 'attention_mask', 'labels')}).loss
                expected_rows = []
                for row in range(len(values['labels'])):
                    scores, costs = [], []
                    for col in range(values['candidate_valid'].shape[1]):
                        if not values['candidate_valid'][row, col]:
                            continue
                        labels = values['candidate_labels'][row:row + 1, col]
                        output = model(input_ids=values['input_ids'][row:row + 1],
                            attention_mask=values['attention_mask'][row:row + 1], labels=labels)
                        token_nll = F.cross_entropy(output.logits.reshape(-1, output.logits.size(-1)),
                            labels.reshape(-1), ignore_index=-100, reduction='none')
                        scores.append(-token_nll.sum())
                        costs.append(values['candidate_costs'][row, col])
                    posterior = torch.softmax(.2 * torch.stack(scores), 0)
                    expected_rows.append((posterior * torch.stack(costs)).sum())
                return ce + .7 * torch.stack(expected_rows).mean()
        finally:
            for module, flag in flags:
                module.training = flag

    def test_flat_ce_and_independent_candidate_sum_log_probability(self):
        for rewired in (False, True):
            model = self.make(rewired=rewired, dropout=False)
            values = self.inputs()
            expected = self.independent_objective(model, values)
            model.train()
            output = model(**values)
            torch.testing.assert_close(output.loss, expected, atol=1e-6, rtol=1e-6)
            output.loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.grad is not None))
            self.assertGreater(model.base_model.model.decoder.layers[0].fc1.weight.grad.norm(), 0)

    def test_zero_weight_is_exact_stochastic_ce_and_no_candidate_dependency(self):
        model = self.make(weight=0.)
        model.train()
        values = self.inputs()
        native = {key: values[key] for key in ('input_ids', 'attention_mask', 'labels')}
        torch.manual_seed(928)
        expected = model.base_model(**native).loss
        expected_rng = torch.random.get_rng_state().clone()
        expected.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        torch.manual_seed(928)
        actual = model(**values).loss
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
        actual.backward()
        for name, parameter in model.named_parameters():
            if name in grads:
                torch.testing.assert_close(parameter.grad, grads[name], atol=0, rtol=0)
        torch.manual_seed(928)
        torch.testing.assert_close(model(**native).loss, expected, atol=0, rtol=0)

    def test_costs_detached_and_padding_has_no_effect(self):
        model = self.make(dropout=False)
        values = self.inputs()
        costs = values['candidate_costs'].clone().requires_grad_(True)
        values['candidate_costs'] = costs
        reference = model(**values).loss
        reference.backward()
        self.assertIsNone(costs.grad)
        values['candidate_labels'][0, 2] = 99999
        values['candidate_costs'] = costs.detach().clone()
        values['candidate_costs'][0, 2] = float('inf')
        torch.testing.assert_close(model(**values).loss, reference)

    def test_checkpoint_recompute_preserves_gradients_rng_and_route(self):
        for rewired in (False, True):
            model = self.make(rewired=rewired)
            model.train()
            flags = [m.training for m in model.modules()]
            torch.manual_seed(74)
            plain = model(**self.inputs()).loss
            plain.backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            rng = torch.random.get_rng_state().clone()
            model.zero_grad(set_to_none=True)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            torch.manual_seed(74)
            checked = model(**self.inputs()).loss
            checked.backward()
            torch.testing.assert_close(checked, plain)
            for name, parameter in model.named_parameters():
                if name in grads:
                    torch.testing.assert_close(parameter.grad, grads[name], atol=3e-6, rtol=4e-4)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(flags, [m.training for m in model.modules()])
            self.assertTrue(all(not cache for cache in model._cache.values()))
            self.assertEqual(model.strength, .1)

    def test_actual_trainer_ga4_and_save_best_reload(self):
        from experiment_runner.engine import ControlTrainer
        model = self.make(rewired=True, dropout=False)
        model.gradient_checkpointing_enable()
        values = self.inputs()
        with tempfile.TemporaryDirectory() as folder:
            trainer = ControlTrainer(model=model, args=Seq2SeqTrainingArguments(
                output_dir=folder, use_cpu=True, report_to=[], gradient_accumulation_steps=4,
                save_safetensors=False), generation_dtype='fp32')
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            self.assertFalse(model.accepts_loss_kwargs)
            direct = model(**values).loss
            direct.backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            trainer.current_gradient_accumulation_steps = 4
            scaled = trainer.training_step(model, dict(values), num_items_in_batch=torch.tensor(999))
            torch.testing.assert_close(scaled, direct.detach() / 4)
            for name, parameter in model.named_parameters():
                if name in grads:
                    torch.testing.assert_close(parameter.grad, grads[name] / 4)
            model.eval()
            native = {key: values[key] for key in ('input_ids', 'attention_mask', 'labels')}
            with torch.no_grad():
                expected = model(**native).logits.clone()
                generated = model.generate(input_ids=values['input_ids'],
                    attention_mask=values['attention_mask'], max_new_tokens=3)
            trainer._save(folder)
            saved_metadata = json.loads((Path(folder) / 'sequence_risk_config.json').read_text())
            self.assertEqual(saved_metadata, model.method_metadata())
            core, info = MBartForConditionalGeneration.from_pretrained(folder, output_loading_info=True)
            self.assertFalse(info['missing_keys'] or info['unexpected_keys'] or info['mismatched_keys'])
            restored = SequenceRiskSeq2SeqLM(core, rewired=True, weight=.7, alpha=.2, score_chunk=2)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(**native).logits, expected)
                torch.testing.assert_close(restored.generate(input_ids=values['input_ids'],
                    attention_mask=values['attention_mask'], max_new_tokens=3), generated)
            trainer.state.best_model_checkpoint = folder
            with torch.no_grad():
                next(model.parameters()).add_(.1)
            trainer._load_best_model()
            with torch.no_grad():
                torch.testing.assert_close(model(**native).logits, expected)
            self.assertEqual(model.extra_parameter_count(), 0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in core.parameters()))

    def test_candidate_chunking_invariant(self):
        reference = self.make(rewired=True, dropout=False, score_chunk=1)
        other = self.make(rewired=True, dropout=False, score_chunk=7)
        first = reference(**self.inputs()).loss
        second = other(**self.inputs()).loss
        torch.testing.assert_close(first, second)
        first.backward()
        second.backward()
        for (name, parameter), (other_name, other_parameter) in zip(reference.named_parameters(), other.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other_parameter.grad, atol=3e-6, rtol=4e-4)

    def test_eval_ignores_candidate_fields_and_baseline_installs_no_route_hooks(self):
        for rewired in (False, True):
            model = self.make(rewired=rewired)
            self.assertEqual(len(model._handles), 4 if rewired else 0)
            model.eval()
            values = self.inputs()
            native = {key: values[key] for key in ('input_ids', 'attention_mask', 'labels')}
            values['candidate_labels'][:] = 99999
            values['candidate_costs'][:] = float('nan')
            with torch.no_grad():
                expected = model(**native)
                actual = model(**values)
            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
            torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)

    def test_collator_variable_candidates_and_validation(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from methods.sequence_risk.v1.data import SequenceRiskCollator
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'a': 4}, unk_token='<unk>')),
            pad_token='<pad>', bos_token='<s>', eos_token='</s>', unk_token='<unk>')
        model = self.make()
        collator = SequenceRiskCollator(tokenizer, model, pad_to_multiple_of=8)
        rows = [dict(input_ids=[0, 4, 2], attention_mask=[1, 1, 1], labels=[0, 4, 2],
            candidate_labels=[[0, 4, 2], [0, 4, 4, 2]], candidate_costs=[.0, .5]),
            dict(input_ids=[0, 4, 4, 2], attention_mask=[1] * 4, labels=[0, 4, 4, 2],
            candidate_labels=[[0, 4, 2]], candidate_costs=[.3])]
        result = collator(rows)
        self.assertEqual(result['candidate_labels'].shape, (2, 2, 8))
        self.assertEqual(result['candidate_valid'].tolist(), [[True, True], [True, False]])
        self.assertTrue(result['candidate_labels'][1, 1].eq(-100).all())
        self.assertEqual(result['candidate_labels'][0, 1, :4].tolist(), [0, 4, 4, 2])
        self.assertEqual(result['labels'][0, :3].tolist(), [0, 4, 2])
        self.assertTrue(result['labels'][0, 3:].eq(-100).all())
        self.assertIn('decoder_input_ids', result)
        clean = [{k: value for k, value in row.items() if not k.startswith('candidate_')} for row in rows]
        self.assertNotIn('candidate_labels', collator(clean))
        with self.assertRaises(ValueError):
            collator([rows[0], clean[1]])

    def test_pool_signal_helper_rejects_degenerate_sets(self):
        from experiment_runner.sequence import assert_pool_signal
        varied = dict(labels=[[0, 4, 2], [0, 5, 2]], costs=[0., .4])
        singleton = dict(labels=[[0, 4, 2]], costs=[0.])
        equal = dict(labels=[[0, 4, 2], [0, 5, 2]], costs=[.2, .2])
        assert_pool_signal([varied, varied])
        for records in ([], [singleton], [equal], [varied, singleton], [varied, equal]):
            with self.subTest(records=records), self.assertRaises(AssertionError):
                assert_pool_signal(records)

    def test_prepare_pool_existing_complete_cannot_bypass_signal_gate(self):
        from experiment_runner import engine, sequence
        from experiment_runner.integrity import sha256
        spec = dict(method='sequence-ce-v1', risk_weight=0.)
        cfg = dict(candidate_pool=dict(split='train', include_gold=True))
        lock = {'frozen': 'regression-test'}
        variants = [
            (dict(row=0, labels=[[0, 4, 2]], costs=[0.]), False),
            (dict(row=0, labels=[[0, 4, 2], [0, 5, 2]], costs=[.2, .2]), False),
            (dict(row=0, labels=[[0, 4, 2], [0, 5, 2]], costs=[0., .4]), True),
        ]
        for record, valid in variants:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                pool = root / 'candidate_pool.jsonl'
                pool.write_text(json.dumps(record) + '\n', encoding='utf-8')
                (root / 'POOL_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',
                    lock=lock, pool_sha256=sha256(pool), rows=1)), encoding='utf-8')
                with patch.object(engine, 'OUT', root), \
                     patch.object(engine, 'validate_environment') as environment, \
                     patch.object(engine, 'plan', return_value=cfg), \
                     patch.object(sequence, '_pool_lock', return_value=lock), \
                     patch.object(engine, 'load_model') as load:
                    if valid:
                        sequence.prepare_pool(spec)
                    else:
                        with self.assertRaisesRegex(AssertionError, 'Degenerate'):
                            sequence.prepare_pool(spec)
                    environment.assert_called_once()
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
