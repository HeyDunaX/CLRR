"""Mechanism gates for the isolated, not-yet-approved difference prototype."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from transformers import MBartForConditionalGeneration
from methods.difference_preserving.v1 import difference_alignment_loss, DifferencePreservingSeq2SeqLM
from methods.difference_preserving.v1.pairs import discover_pairs, words
from methods.original.clrr import CrossLayerResidualRewire
from methods.difference_preserving.v1.data import read_pairs, DifferenceCollator, require_scientific_admission
from test_alignment_residual import native, batch


def training_batch():
    data = batch()
    data.update(peer_input_ids=data['input_ids'].flip(0),
                peer_attention_mask=data['attention_mask'].flip(0),
                peer_labels=data['labels'].flip(0),
                pair_eligible=torch.ones(2, dtype=torch.bool))
    return data


class DifferenceGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_target_is_detached_and_both_source_views_learn(self):
        values = [torch.randn(3, 7, requires_grad=True) for _ in range(4)]
        difference_alignment_loss(*values).backward()
        for value in values[:2]:
            self.assertGreater(value.grad.abs().sum().item(), 0.)
        for value in values[2:]:
            self.assertIsNone(value.grad)

    def test_checkpoint_metadata_rejects_wrong_pooling_or_rewiring(self):
        model = DifferencePreservingSeq2SeqLM(CrossLayerResidualRewire(
            native(), distance=2, strength=.1, stack='encoder'), special_token_ids=(2, 1, 2))
        self.assertEqual(model.method_metadata()['special_token_ids'], [1, 2])
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder, safe_serialization=False)
            model.verify_checkpoint_metadata(folder)
            model.special_token_ids = (1, 2, 3)
            with self.assertRaisesRegex(ValueError, 'recipe mismatch'):
                model.verify_checkpoint_metadata(folder)
            model.special_token_ids = (1, 2)
            model.base_model.strength = .2
            with self.assertRaisesRegex(ValueError, 'recipe mismatch'):
                model.verify_checkpoint_metadata(folder)

    def test_offset_invariance_and_swap_symmetry(self):
        x, xp, y, yp = [torch.randn(4, 8) for _ in range(4)]
        expected = difference_alignment_loss(x, xp, y, yp)
        a, b = torch.randn(1, 8), torch.randn(1, 8)
        torch.testing.assert_close(expected, difference_alignment_loss(x+a, xp+a, y+b, yp+b))
        torch.testing.assert_close(expected, difference_alignment_loss(xp, x, yp, y))

    def test_zero_target_or_empty_mask_is_differentiable_zero(self):
        x, xp = [torch.randn(2, 5, requires_grad=True) for _ in range(2)]
        y = torch.randn(2, 5)
        loss = difference_alignment_loss(x, xp, y, y)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(x.grad.abs().sum().item(), 0.)
        self.assertEqual(difference_alignment_loss(x, xp, y, -y,
            torch.zeros(2, dtype=torch.bool)).item(), 0.)

    def test_collapsed_source_is_not_hidden_and_has_finite_gradient(self):
        x = torch.zeros(2, 5, requires_grad=True)
        xp = torch.zeros(2, 5, requires_grad=True)
        y = torch.randn(2, 5)
        loss = difference_alignment_loss(x, xp, y, -y)
        self.assertAlmostEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.)

    def test_invalid_shapes_nan_and_mask_rejected(self):
        x = torch.ones(2, 5)
        for args in [(x, x[:1], x, x), (x, x, x * float('nan'), x)]:
            with self.assertRaises(ValueError):
                difference_alignment_loss(*args)
        with self.assertRaises(ValueError):
            difference_alignment_loss(x, x, x, x, torch.ones(2))

    def test_miner_is_deterministic_and_does_not_invent_semantic_labels(self):
        rows = [dict(source='a b c.', target='甲乙'),
                dict(source='a d c.', target='甲丙'),
                dict(source='a b c!', target='其他'),
                dict(source='a e c.', target='甲乙')]
        snapshot = copy.deepcopy(rows)
        pairs = discover_pairs(rows)
        self.assertEqual(pairs, discover_pairs(rows))
        self.assertEqual(rows, snapshot)
        self.assertIn((0, 1), [(p['left'], p['right']) for p in pairs])
        self.assertNotIn((0, 2), [(p['left'], p['right']) for p in pairs])
        self.assertNotIn((0, 3), [(p['left'], p['right']) for p in pairs])
        self.assertTrue(all(not p['semantic_verified'] for p in pairs))

    def test_unicode_marks_do_not_split_words_or_inflate_minimum_length(self):
        import unicodedata
        self.assertEqual(words('Pô krơ̆ng'), ('pô', 'krơ̆ng'))
        self.assertEqual(words('Ƀơ̆ng bur'), ('ƀơ̆ng', 'bur'))
        text = "Drei rao amrêč. Ca'edes!"
        self.assertEqual(words(text), words(unicodedata.normalize('NFD', text)))
        self.assertEqual(words("'a' a’b a_b"), ('a', 'a’b', 'a', 'b'))
        rows = [dict(source='Pô krơ̆ng', target='A'),
                dict(source='Djă krơ̆ng', target='B')]
        self.assertEqual(discover_pairs(rows), [])
        self.assertEqual(len(discover_pairs(rows, minimum_words=2)), 1)

    def test_miner_rejects_invalid_word_bounds(self):
        for bounds in [(0, 50), (5, 3), (True, 50), (3, 3.5)]:
            with self.assertRaises(ValueError):
                discover_pairs([], *bounds)

    def test_generic_peer_excludes_every_reviewed_neighbor(self):
        import hashlib
        rows = [dict(source=f'source {i}', target=f'target {i}') for i in range(5)]
        with tempfile.TemporaryDirectory() as folder:
            train = Path(folder, 'train.csv')
            train.write_text('synthetic test fixture', encoding='utf-8')
            record = dict(split='train', train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
                status='SEMANTIC_REVIEW_PASS', pairs=[dict(left=0, right=j,
                    meaning_difference_valid=True, translation_correspondence_valid=True,
                    reviewer='synthetic unit-test fixture') for j in (1, 2, 3)])
            manifest = Path(folder, 'pairs.json')
            manifest.write_text(json.dumps(record), encoding='utf-8')
            for seed in range(20):
                generic = read_pairs(manifest, train, rows, 'generic', seed)
                self.assertEqual(generic[0], 4)
                self.assertEqual(set(generic), {0, 1, 2, 3})
            candidate = read_pairs(manifest, train, rows, 'candidate', 42)
            self.assertEqual(candidate[0], 1)
            record['pairs'].reverse()
            manifest.write_text(json.dumps(record), encoding='utf-8')
            self.assertEqual(candidate, read_pairs(manifest, train, rows, 'candidate', 42))

    def test_eval_and_generation_equal_underlying_backbone(self):
        for rewired in (False, True):
            core = native()
            if rewired:
                core = CrossLayerResidualRewire(core, distance=2, strength=.1, stack='encoder')
            model = DifferencePreservingSeq2SeqLM(core, special_token_ids=(1, 2)).eval()
            data = batch()
            torch.testing.assert_close(model(**data).loss, core(**data).loss)
            inputs = {k: data[k] for k in ('input_ids', 'attention_mask')}
            for cache in (True, False):
                torch.testing.assert_close(model.generate(**inputs, num_beams=2, max_new_tokens=3,
                    use_cache=cache), core.generate(**inputs, num_beams=2, max_new_tokens=3, use_cache=cache))

    def test_auxiliary_does_not_change_ce_dropout_rng_or_add_parameters(self):
        core = native(.15)
        model = DifferencePreservingSeq2SeqLM(core, special_token_ids=(1, 2)).train()
        torch.manual_seed(103)
        expected = core(**batch())
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(103)
        actual = model(**training_batch())
        torch.testing.assert_close(expected.logits, actual.logits)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertGreater(actual.loss.item(), expected.loss.item())
        self.assertEqual(model.extra_parameter_count(), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in core.parameters()))

    def test_missing_peers_fail_and_ineligible_batch_matches_ce(self):
        core = native(.15)
        model = DifferencePreservingSeq2SeqLM(core, special_token_ids=(1, 2)).train()
        with self.assertRaises(ValueError):
            model(**batch())
        data = training_batch()
        data['pair_eligible'].zero_()
        torch.manual_seed(15)
        expected = core(**batch()).loss
        torch.manual_seed(15)
        torch.testing.assert_close(model(**data).loss, expected)

    def test_checkpoint_replay_and_save_reload_two_topologies_two_peer_patterns(self):
        for rewired in (False, True):
            for pairing in ('generic', 'candidate'):
                plain = native(.15)
                replay = copy.deepcopy(plain)
                models = []
                for core in (plain, replay):
                    if rewired:
                        core = CrossLayerResidualRewire(core, distance=2, strength=.1, stack='encoder')
                    models.append(DifferencePreservingSeq2SeqLM(core, special_token_ids=(1, 2)))
                a, b = models
                data = training_batch()
                if pairing == 'candidate':
                    data['peer_input_ids'] = data['input_ids'].clone()
                    data['peer_input_ids'][:, 0] += 1
                b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
                for model in models:
                    torch.manual_seed(89)
                    model.train()(**data).loss.backward()
                for (name, left), (_, right) in zip(a.named_parameters(), b.named_parameters()):
                    self.assertIsNotNone(left.grad, name)
                    torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=5e-5)
                a.eval()
                expected = a(**batch()).logits
                with tempfile.TemporaryDirectory() as folder:
                    a.save_pretrained(folder, state_dict=a.state_dict(), safe_serialization=False)
                    core = MBartForConditionalGeneration.from_pretrained(folder, attn_implementation='sdpa')
                    if rewired:
                        core = CrossLayerResidualRewire(core, distance=2, strength=.1, stack='encoder')
                    restored = DifferencePreservingSeq2SeqLM(core, special_token_ids=(1, 2)).eval()
                    saved = json.loads(Path(folder, 'difference_preserving_config.json').read_text())
                    self.assertEqual(saved, restored.method_metadata())
                    torch.testing.assert_close(expected, restored(**batch()).logits)

    def test_trainer_accumulation_checkpoint_resume_matches_uninterrupted(self):
        from transformers import Seq2SeqTrainingArguments, TrainerCallback
        from experiment_runner.engine import ControlTrainer

        class StopAfterTwo(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 2:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        data = training_batch()
        rows = [{key: value[i] for key, value in data.items()} for i in range(2)] * 12
        def collate(features):
            return {key: torch.stack([row[key] for row in features]) for key in features[0]}
        for rewired in (False, True):
            with self.subTest(rewired=rewired), tempfile.TemporaryDirectory() as folder:
                initial = native(.1)
                def make_model():
                    core = copy.deepcopy(initial)
                    if rewired:
                        core = CrossLayerResidualRewire(core, distance=2, strength=.1, stack='encoder')
                    return DifferencePreservingSeq2SeqLM(core, special_token_ids=(1, 2))
                def trainer(model, name, callbacks=None):
                    args = Seq2SeqTrainingArguments(output_dir=str(Path(folder, name)),
                        use_cpu=True, max_steps=3, per_device_train_batch_size=2,
                        gradient_accumulation_steps=4, gradient_checkpointing=True,
                        gradient_checkpointing_kwargs={'use_reentrant': True},
                        learning_rate=5e-5, save_strategy='steps', save_steps=2,
                        report_to=[], disable_tqdm=True, logging_strategy='no',
                        remove_unused_columns=False, dataloader_pin_memory=False, seed=42)
                    return ControlTrainer(model=model, args=args, train_dataset=rows,
                        data_collator=collate, callbacks=callbacks)
                continuous = trainer(make_model(), 'continuous')
                continuous.train()
                interrupted = trainer(make_model(), 'interrupted', [StopAfterTwo()])
                interrupted.train()
                checkpoint = Path(folder, 'interrupted', 'checkpoint-2')
                interrupted.model.verify_checkpoint_metadata(checkpoint)
                resumed = trainer(make_model(), 'resumed')
                resumed.train(resume_from_checkpoint=str(checkpoint))
                self.assertEqual(resumed.state.global_step, 3)
                for name, value in continuous.model.state_dict().items():
                    torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)

    def test_review_gate_rejects_discovery_and_matches_control_anchor_coverage(self):
        import hashlib
        rows = [dict(source=f'source {i}', target=f'target {i}') for i in range(6)]
        with tempfile.TemporaryDirectory() as folder:
            train = Path(folder, 'train.csv')
            train.write_text('source,target\n', encoding='utf-8')
            manifest = Path(folder, 'pairs.json')
            record = dict(split='train', train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
                          status='PENDING', pairs=[])
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'review has not passed'):
                read_pairs(manifest, train, rows, 'candidate', 42)
            record.update(status='SEMANTIC_REVIEW_PASS', pairs=[dict(left=0, right=1,
                meaning_difference_valid=True, translation_correspondence_valid=True, reviewer='fixture')])
            manifest.write_text(json.dumps(record))
            candidate = read_pairs(manifest, train, rows, 'candidate', 42)
            generic = read_pairs(manifest, train, rows, 'generic', 42)
            self.assertEqual(set(candidate), set(generic))
            self.assertEqual(generic, read_pairs(manifest, train, rows, 'generic', 42))
            self.assertTrue(all(generic[i] not in (i, candidate[i]) for i in candidate))
            record['pairs'][0]['reviewer'] = ''
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'semantic review'):
                read_pairs(manifest, train, rows, 'candidate', 42)

    def test_collator_preserves_anchor_batch_and_pads_peers_separately(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
            WordLevel({'[UNK]':0, '[PAD]':1, '[EOS]':2, 'a':3}, unk_token='[UNK]')),
            unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
        model = DifferencePreservingSeq2SeqLM(native())
        collator = DifferenceCollator(tokenizer, model)
        features = [dict(input_ids=[4,5,2], attention_mask=[1,1,1], labels=[6,2],
            peer_input_ids=[7,2], peer_attention_mask=[1,1], peer_labels=[8,9,2], pair_eligible=True),
            dict(input_ids=[5,2], attention_mask=[1,1], labels=[7,8,2],
            peer_input_ids=[6,7,8,2], peer_attention_mask=[1,1,1,1], peer_labels=[9,2], pair_eligible=False)]
        before = copy.deepcopy(features)
        actual = collator(features)
        original = collator.base([{k:v for k,v in row.items()
            if not k.startswith('peer_') and k != 'pair_eligible'} for row in features])
        self.assertEqual(before, features)
        for key in original:
            torch.testing.assert_close(actual[key], original[key])
        self.assertEqual(actual['pair_eligible'].tolist(), [True, False])
        self.assertEqual(actual['peer_labels'][1, 2].item(), -100)


    def test_correctness_pass_cannot_override_scientific_hold(self):
        for admission in ({}, {'novelty':'PASS', 'semantic_pairs':'HOLD'},
                          {'novelty':'HOLD', 'semantic_pairs':'PASS'}):
            with self.assertRaisesRegex(ValueError, 'scientific gates'):
                require_scientific_admission(dict(scientific_admission=admission))
        require_scientific_admission(dict(scientific_admission=dict(novelty='PASS', semantic_pairs='PASS')))


class TechnicalAdmissionGate(unittest.TestCase):
    def test_technical_probe_does_not_admit_real_gate_or_training(self):
        from unittest.mock import patch
        from experiment_runner import engine
        spec = dict(method='difference-candidate-v1')
        held = dict(scientific_admission=dict(novelty='HOLD', semantic_pairs='HOLD'))
        with patch.object(engine, 'plan', return_value=held):
            with self.assertRaisesRegex(ValueError, 'scientific gates'):
                engine.gpu_gate(spec)
            with self.assertRaisesRegex(ValueError, 'scientific gates'):
                engine.train_run(spec)
            with patch.object(engine, 'validate_environment', side_effect=RuntimeError('resource-check')):
                with self.assertRaisesRegex(RuntimeError, 'resource-check'):
                    engine.gpu_gate(spec, technical_only=True)
        self.assertEqual(held['scientific_admission']['semantic_pairs'], 'HOLD')


if __name__ == '__main__':
    unittest.main()
