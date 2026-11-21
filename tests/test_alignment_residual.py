"""Scientific correctness of the residual architecture and gradient policy."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from transformers import MBartConfig, MBartForConditionalGeneration
from methods.alignment_residual.v1 import AlignmentResidualMT, METHODS
from methods.original.clrr import CrossLayerResidualRewire


def native(dropout=0.):
    config = MBartConfig(vocab_size=40, d_model=16, encoder_layers=4, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_position_embeddings=32, pad_token_id=1, eos_token_id=2,
        decoder_start_token_id=2, dropout=dropout, attention_dropout=dropout,
        activation_dropout=dropout, encoder_layerdrop=0., decoder_layerdrop=0.)
    config._attn_implementation = 'sdpa'
    config.use_cache = False
    return MBartForConditionalGeneration(config)


def wrapped(base, arm):
    method = list(METHODS)[arm]
    return AlignmentResidualMT(copy.deepcopy(base), method, rank=4,
        module_seed=10042, jepa_weight=0. if arm == 0 else .1)


def batch():
    return dict(input_ids=torch.tensor([[4, 5, 2, 1], [6, 7, 8, 2]]),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        labels=torch.tensor([[9, 10, 2, -100], [11, 12, 13, 2]]))


def nonzero_modules(model):
    with torch.no_grad():
        for module in model.get_encoder().residual_modules.values():
            module.up.fill_(.03)


class AlignmentResidualGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.base = native()

    def test_init_rng_and_zero_up_forward(self):
        rng = torch.get_rng_state().clone()
        full = wrapped(self.base, 2).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        old = CrossLayerResidualRewire(copy.deepcopy(self.base), distance=2,
                                      strength=.1, stack='encoder').eval()
        local = wrapped(self.base, 3).eval()
        with torch.no_grad():
            torch.testing.assert_close(full(**batch()).logits, old(**batch()).logits)
            torch.testing.assert_close(local(**batch()).logits, self.base.eval()(**batch()).logits)
        self.assertEqual(full.extra_parameter_count(), 2 * 2 * 16 * 4)

    def test_confined_gradient_equals_unrestricted_module_gradient(self):
        full, confined = wrapped(self.base, 1), wrapped(self.base, 2)
        nonzero_modules(full); nonzero_modules(confined)
        target = torch.randn(2, 16)
        for model in (full, confined):
            loss = model.auxiliary_loss(batch()['input_ids'], batch()['attention_mask'], target)
            loss.backward()
        a, b = dict(full.named_parameters()), dict(confined.named_parameters())
        for name in a:
            if '.residual_modules.' in name:
                self.assertIsNotNone(a[name].grad)
                torch.testing.assert_close(a[name].grad, b[name].grad, rtol=2e-4, atol=5e-5)
                self.assertGreater(float(b[name].grad.abs().sum()), 0.)
            else:
                self.assertIsNone(b[name].grad, name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
            for n, p in a.items() if '.residual_modules.' not in n))

    def test_auxiliary_cannot_change_backbone_ce_gradient(self):
        capacity, confined = wrapped(self.base, 0), wrapped(self.base, 2)
        nonzero_modules(capacity); nonzero_modules(confined)
        capacity(**batch()).loss.backward(); confined(**batch()).loss.backward()
        a, b = dict(capacity.named_parameters()), dict(confined.named_parameters())
        for name in a:
            if '.residual_modules.' not in name:
                torch.testing.assert_close(a[name].grad, b[name].grad, rtol=2e-4, atol=5e-5)

    def test_reentrant_checkpoint_replay_matches_dropout_and_gradients(self):
        base = native(.15)
        ordinary, replay = wrapped(base, 2), wrapped(base, 2)
        nonzero_modules(ordinary); nonzero_modules(replay)
        replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
        for model in (ordinary, replay):
            torch.manual_seed(123)
            model(**batch()).loss.backward()
        for (name, a), (_, b) in zip(ordinary.named_parameters(), replay.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=5e-5, msg=name)

    def test_all_arms_consume_same_dropout_rng(self):
        base = native(.15)
        endings = []
        for arm in range(4):
            model = wrapped(base, arm)
            torch.manual_seed(456)
            model(**batch()).loss.backward()
            endings.append(torch.get_rng_state())
        self.assertTrue(all(torch.equal(endings[0], value) for value in endings[1:]))

    def test_accumulation_equal_full_batch_without_dropout(self):
        whole, micro = wrapped(self.base, 2), wrapped(self.base, 2)
        nonzero_modules(whole); nonzero_modules(micro)
        # Equal valid-label counts make the microbatch-mean contract comparable.
        data = batch(); data['labels'][0, -1] = 2
        whole(**data).loss.backward()
        for index in range(2):
            (micro(**{k: v[index:index+1] for k, v in data.items()}).loss / 2).backward()
        for (name, a), (_, b) in zip(whole.named_parameters(), micro.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=5e-5, msg=name)

    def test_strict_save_load_and_beam_cache(self):
        model = wrapped(self.base, 2).eval(); nonzero_modules(model)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = AlignmentResidualMT.from_pretrained(directory).eval()
            torch.testing.assert_close(model(**batch()).logits, restored(**batch()).logits)
        inputs = {k: v for k, v in batch().items() if k != 'labels'}
        with torch.no_grad():
            cached = model.generate(**inputs, num_beams=4, max_new_tokens=5, use_cache=True)
            uncached = model.generate(**inputs, num_beams=4, max_new_tokens=5, use_cache=False)
        self.assertTrue(torch.equal(cached, uncached))

    def test_zero_up_down_gradient_and_live_up(self):
        model = wrapped(self.base, 2)
        model.auxiliary_loss(batch()['input_ids'], batch()['attention_mask'],
                             torch.randn(2, 16)).backward()
        for module in model.get_encoder().residual_modules.values():
            self.assertTrue(module.down.grad.eq(0).all())
            self.assertGreater(float(module.up.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
