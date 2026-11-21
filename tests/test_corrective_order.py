"""Gradient, state and functional-replay tests for exact corrective-order paths."""
import copy
import tempfile
import unittest

import torch
from transformers import MBartConfig, MBartForConditionalGeneration
from methods.corrective_order import CorrectiveOrderMT
from methods.corrective_order.v1.model import compose_paths, symmetric_kl, offload_saved_tensors


def tiny(activation='gelu'):
    cfg = MBartConfig(vocab_size=19, d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=12, decoder_ffn_dim=12, max_position_embeddings=16,
        dropout=.2, attention_dropout=.1, activation_dropout=.1,
        pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
        use_cache=False, activation_function=activation)
    cfg._attn_implementation = 'sdpa'
    return MBartForConditionalGeneration(cfg)


def batch(offset=0):
    return dict(input_ids=torch.tensor([[4+offset, 5, 2], [7, 8, 2]]),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        labels=torch.tensor([[6+offset, 2, -100], [9, 10, 2]]))


class CorrectiveOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_relu_scaling_counterexample(self):
        from methods.corrective_order.v1.audit import relu_counterexample
        result = relu_counterexample()
        for pair in result['exponents']:
            self.assertAlmostEqual(pair['endpoint'], 1.5, places=9)
            self.assertAlmostEqual(pair['skl'], 3., places=4)

    def test_activation_audit_does_not_mutate_or_leave_hooks(self):
        from methods.corrective_order.v1.audit import ActivationAudit
        torch.manual_seed(19)
        model = CorrectiveOrderMT(tiny('relu'), 'corrective-order-consistency-v1').train()
        model.gradient_checkpointing_enable({'use_reentrant': False})
        initial = {k: p.clone() for k, p in model.state_dict().items()}
        audit = ActivationAudit(model.base_model, batch())
        result = audit.run(model, batch(), .1)
        self.assertEqual(len(result['layers']), 4)
        self.assertEqual(result['positions'], 2*(6+5)*12)
        self.assertGreater(result['crossings'], 0)
        self.assertTrue(all(not m._forward_hooks for _, m in audit.modules))
        for k, p in model.state_dict().items():
            torch.testing.assert_close(p, initial[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_training_failure_keeps_traceback(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from methods.corrective_order.v1.telemetry import record_training_step
        def fail(*args):
            raise RuntimeError('synthetic microbatch failure')
        with tempfile.TemporaryDirectory() as folder:
            trainer = SimpleNamespace(args=SimpleNamespace(output_dir=folder),
                state=SimpleNamespace(global_step=3, epoch=.5))
            with self.assertRaisesRegex(RuntimeError, 'synthetic microbatch failure'):
                record_training_step(trainer, fail, None, batch(), None)
            failure = json.loads((Path(folder)/'training_failure.json').read_text())
            self.assertIn('synthetic microbatch failure', failure['traceback'])
            self.assertEqual(failure['step_before_update'], 3)

    def test_exact_outer_gradient_finite_difference(self):
        def objective(v):
            def loss(p, seed):
                u, w = p['v'].unbind()
                logit = u if seed == 0 else u+u*w
                return torch.nn.functional.softplus(-logit)
            ce, ab, ba, _, _ = compose_paths({'v': v}, loss, .1, (0, 1))
            def logits(p):
                return torch.stack((p['v'].sum(), p['v'].sum()*0)).reshape(1, 1, 2)
            return ce + .2 * symmetric_kl(logits(ab), logits(ba), torch.zeros(1, 1).long())/.1**4
        v = torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)
        grad, = torch.autograd.grad(objective(v), v)
        expected = []
        for i in range(2):
            delta = torch.zeros_like(v); delta[i] = 1e-5
            expected.append((objective(v+delta)-objective(v-delta))/(2e-5))
        torch.testing.assert_close(grad, torch.stack(expected), atol=2e-8, rtol=2e-6)

    def test_scaling_gate_rejects_observed_failure_and_accepts_quadratic_limit(self):
        from methods.corrective_order.v1.prepare import eligible_steps
        observed = [dict(eta=e, auxiliary=r, parameter_order_distance=d)
            for e, r, d in [(0.003, 996036678.51, .07647776),
                            (.01, 48645709.07, .39795603), (.03, 2377339.36, 1.47338105)]]
        self.assertEqual(eligible_steps(observed), [])
        p = {'w': torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)}
        def loss(p, seed):
            h = p['w'].new_tensor([[2., .4], [.4, 3.]]) if seed == 0 else p['w'].new_tensor([[1., .2], [.2, 4.]])
            return .5*p['w'] @ h @ p['w']
        records = []
        labels = torch.zeros(1, 1, dtype=torch.long)
        for eta in (.0003, .001, .003):
            _, ab, ba, _, _ = compose_paths(p, loss, eta, (0, 1))
            def logits(state):
                return torch.stack((state['w'].sum(), state['w'].sum()*0)).reshape(1, 1, 2)
            records.append(dict(eta=eta,
                auxiliary=float(symmetric_kl(logits(ab), logits(ba), labels).detach())/eta**4,
                parameter_order_distance=float((ab['w']-ba['w']).norm().detach())))
        self.assertEqual(eligible_steps(records), [.0003, .001])

    def test_mbart_identical_masks_are_null_with_checkpoint_replay(self):
        torch.manual_seed(423)
        model = CorrectiveOrderMT(tiny(), 'corrective-order-consistency-v1', eta=.0003).train()
        model.gradient_checkpointing_enable({'use_reentrant': False})
        loss, _, penalty = model.objective(batch(), batch(1), seeds=(101, 101))
        self.assertEqual(model.last_diagnostics['parameter_order_distance'], 0)
        self.assertEqual(float(penalty.detach()), 0)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_null_swap_and_symmetric_control_is_different(self):
        p = {'w': torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)}
        def loss(p, seed):
            a = torch.tensor([[2., .4], [.4, 3.]], dtype=torch.float64)
            b = torch.tensor([[1., .2], [.2, 4.]], dtype=torch.float64)
            return .5 * p['w'] @ (a if seed == 0 else b) @ p['w']
        _, ab, ba, a, b = compose_paths(p, loss, .01, (0, 1))
        _, revab, revba, _, _ = compose_paths(p, loss, .01, (1, 0))
        torch.testing.assert_close(ab['w'], revba['w'], rtol=0, atol=0)
        torch.testing.assert_close(ba['w'], revab['w'], rtol=0, atol=0)
        _, aa, aa2, a1, _ = compose_paths(p, loss, .01, (0, 0))
        torch.testing.assert_close(aa['w'], aa2['w'], rtol=0, atol=0)
        self.assertGreater((aa['w'] - (2*a1['w']-p['w'])).abs().max(), 0)

    def test_mbart_checkpoint_gradients_and_no_parameter_mutation(self):
        torch.manual_seed(410)
        base = tiny()
        for mode in ('corrective-order-consistency-v1', 'corrective-order-curvature-v1'):
            plain = CorrectiveOrderMT(copy.deepcopy(base), mode, eta=.03, coefficient=.001).train()
            checked = CorrectiveOrderMT(copy.deepcopy(base), mode, eta=.03, coefficient=.001).train()
            checked.gradient_checkpointing_enable({'use_reentrant': False})
            before = {k: p.detach().clone() for k, p in plain.named_parameters()}
            for model in (plain, checked):
                loss, _, penalty = model.objective(batch(), batch(1), seeds=(23, 29))
                self.assertGreater(penalty.detach(), 0)
                loss.backward()
                self.assertTrue(model.base_model.training)
            for (name, p), q in zip(plain.named_parameters(), checked.parameters()):
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-3)

    def test_ce_and_rdrop_definition_and_probe_padding(self):
        torch.manual_seed(412)
        base = tiny()
        ce_model = CorrectiveOrderMT(copy.deepcopy(base), 'corrective-order-ce-v1').train()
        rdrop = CorrectiveOrderMT(copy.deepcopy(base), 'corrective-order-rdrop-v1').train()
        ce, _, _ = ce_model.objective(batch(), seeds=(4, 8))
        loss, same_ce, kl = rdrop.objective(batch(), seeds=(4, 8))
        torch.testing.assert_close(ce, same_ce)
        torch.testing.assert_close(loss, same_ce + 1.25*kl)
        a = torch.randn(2, 3, 19, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        symmetric_kl(a, b, batch()['labels']).backward()
        self.assertEqual(a.grad[0, 2].abs().sum(), 0)

    def test_native_inference_serialization_and_replay_rng(self):
        torch.manual_seed(413)
        model = CorrectiveOrderMT(tiny(), 'corrective-order-consistency-v1').train()
        state = dict(model.base_model.named_parameters())
        rng = torch.get_rng_state().clone()
        a = model._evaluate(state, batch(), 44, True)
        b = model._evaluate(state, batch(), 44, True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        model.eval()
        with torch.no_grad():
            expected = model(**batch()).logits
            torch.testing.assert_close(expected, model.base_model(**batch()).logits)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder, state_dict=model.state_dict(), safe_serialization=False)
            restored = CorrectiveOrderMT(MBartForConditionalGeneration.from_pretrained(folder), model.method).eval()
            restored.verify_checkpoint_metadata(folder)
            with torch.no_grad():
                torch.testing.assert_close(restored(**batch()).logits, expected, rtol=0, atol=0)


    def test_offload_hooks_preserve_higher_order_gradient(self):
        torch.manual_seed(421)
        plain = CorrectiveOrderMT(tiny().double(), 'corrective-order-consistency-v1', eta=.03)
        hooked = copy.deepcopy(plain)
        plain.objective(batch(), batch(1), seeds=(42, 43))[0].backward()
        with offload_saved_tensors():
            loss = hooked.objective(batch(), batch(1), seeds=(42, 43))[0]
        loss.backward()
        for a, b in zip(plain.parameters(), hooked.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-9)

    def test_disk_spill_preserves_higher_order_gradient(self):
        from methods.corrective_order.v1.storage import disk_saved_tensors
        torch.manual_seed(421)
        plain = CorrectiveOrderMT(tiny().double(), 'corrective-order-consistency-v1', eta=.03)
        spilled = copy.deepcopy(plain)
        plain.objective(batch(), batch(1), seeds=(42, 43))[0].backward()
        with tempfile.TemporaryDirectory() as root:
            with disk_saved_tensors(root, minimum_bytes=0, reserve_bytes=0, disk_reserve=0):
                loss = spilled.objective(batch(), batch(1), seeds=(42, 43))[0]
            loss.backward()
        for a, b in zip(plain.parameters(), spilled.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-9)

    def test_train_probe_derangement(self):
        from datasets import Dataset
        from methods.corrective_order.v1.data import add_probes
        ds = Dataset.from_dict(dict(input_ids=[[i] for i in range(12)],
            attention_mask=[[1]]*12, labels=[[i] for i in range(12)]))
        a, b = add_probes(ds, 42), add_probes(ds, 42)
        self.assertEqual(list(a['probe_input_ids']), list(b['probe_input_ids']))
        self.assertEqual(sorted(x[0] for x in a['probe_input_ids']), list(range(12)))
        self.assertTrue(all(a[i]['input_ids'] != a[i]['probe_input_ids'] for i in range(12)))

    def test_trainer_accumulation_save_and_resume(self):
        import json
        from pathlib import Path
        from experiment_runner.engine import ControlTrainer
        from methods.corrective_order.v1.telemetry import TrainingTelemetry
        from transformers import Seq2SeqTrainingArguments
        torch.manual_seed(500)
        initial = tiny()
        model = CorrectiveOrderMT(copy.deepcopy(initial), 'corrective-order-consistency-v1',
                                  eta=.03, coefficient=.001)
        examples = list(range(12))
        def collate(rows):
            data = batch()
            data.update({'probe_'+k: v for k, v in batch(1).items()})
            return data
        def args(path):
            return Seq2SeqTrainingArguments(output_dir=str(path), use_cpu=True,
                seed=71, data_seed=71, max_steps=2, per_device_train_batch_size=2,
                gradient_accumulation_steps=2, learning_rate=1e-4,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                save_strategy='steps', save_steps=1, save_safetensors=False,
                optim='adamw_torch', report_to=[], disable_tqdm=True,
                dataloader_pin_memory=False, remove_unused_columns=False)
        with tempfile.TemporaryDirectory() as folder:
            first = ControlTrainer(model=model, args=args(folder),
                train_dataset=examples, data_collator=collate, callbacks=[TrainingTelemetry(folder)])
            first.train()
            diagnostics = [json.loads(s) for s in (Path(folder)/'microbatch_diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(len(diagnostics), 4)
            self.assertTrue(all('dropout_seeds' in r['diagnostics'] for r in diagnostics))
            events = [json.loads(s) for s in (Path(folder)/'training_events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r.get('event') == 'PRE_OPTIMIZER' for r in events), 2)
            self.assertEqual(sum(r.get('event') == 'CHECKPOINT' for r in events), 2)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored = CorrectiveOrderMT(MBartForConditionalGeneration.from_pretrained(Path(folder)/'checkpoint-1'),
                model.method, eta=.03, coefficient=.001)
            restored.verify_checkpoint_metadata(Path(folder)/'checkpoint-1')
            second = ControlTrainer(model=restored, args=args(Path(folder)/'resume'),
                train_dataset=examples, data_collator=collate)
            second.train(resume_from_checkpoint=str(Path(folder)/'checkpoint-1'))
            for k, v in restored.state_dict().items():
                torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
            first.state.best_model_checkpoint = str(Path(folder)/'checkpoint-2')
            first._load_best_model()
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, expected[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
