"""U2 CPU gates on random tiny mBART, without model/data downloads."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from transformers import MBartConfig, MBartForConditionalGeneration
from methods.depth_difference.v1 import DepthDifferencePrototype, DepthEncoderOutput


def backbone():
    config = MBartConfig(vocab_size=43, d_model=16, encoder_layers=4, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_position_embeddings=32, dropout=.1,
        attention_dropout=.1, activation_dropout=.1, encoder_layerdrop=0.,
        decoder_layerdrop=0., pad_token_id=1, bos_token_id=0, eos_token_id=2,
        decoder_start_token_id=2, forced_eos_token_id=None, use_cache=True)
    config._attn_implementation = 'eager'
    return MBartForConditionalGeneration(config)


def batch():
    return dict(input_ids=torch.tensor([[0, 5, 6, 2, 1], [0, 7, 8, 9, 2]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
        labels=torch.tensor([[0, 10, 11, 2, -100], [0, 12, 13, 14, 2]]))


class DepthDifferenceGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.native = backbone()
        self.model = DepthDifferencePrototype(copy.deepcopy(self.native), donor_layer=2)

    def test_zero_gate_eval_and_train_rng_and_base_gradients(self):
        self.assertEqual(self.model.extra_parameter_count(), 2)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()) -
                         sum(p.numel() for p in self.native.parameters()), 2)
        for training in (False, True):
            self.model.train(training); self.native.train(training)
            torch.manual_seed(123)
            reference = self.native(**batch())
            rng = torch.random.get_rng_state()
            torch.manual_seed(123)
            output = self.model(**batch())
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            torch.testing.assert_close(output.logits, reference.logits, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(output.loss, reference.loss)
            self.assertEqual(output.encoder_last_hidden_state.shape[-1], 16)
            if training:
                reference.loss.backward(); output.loss.backward()
                native_grads = dict(self.native.named_parameters())
                for name, parameter in self.model.base_model.named_parameters():
                    if name.endswith('.gates'):
                        continue
                    original = name.replace('.encoder_attn.attention.', '.encoder_attn.')
                    torch.testing.assert_close(parameter.grad, native_grads[original].grad,
                                               atol=2e-7, rtol=2e-5)
                gate = self.model.correction_attention.gates
                self.assertTrue(torch.isfinite(gate.grad).all())
                self.assertGreater(gate.grad.norm().item(), 0.)

    def test_attention_formula_detach_shared_projections_and_head_mask(self):
        module = self.model.correction_attention.eval().double()
        module.gates.data.fill_(.3)
        attn = module.attention
        query = torch.randn(2, 3, 16, dtype=torch.float64, requires_grad=True)
        recipient = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        donor = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        mask = torch.zeros(2, 1, 3, 5, dtype=torch.float64); mask[:, :, :, -1] = -1e30
        head_mask = torch.tensor([1., 0.], dtype=torch.float64)
        packed = torch.cat((recipient, donor), -1)
        actual, _ = module(query, packed, attention_mask=mask, layer_head_mask=head_mask)
        shape = lambda tensor: tensor.view(2, -1, 2, 8).transpose(1, 2)
        q = shape(attn.q_proj(query))
        k = shape(attn.k_proj(recipient)); kd = shape(attn.k_proj(donor.detach()))
        v = shape(attn.v_proj(recipient)); vd = shape(attn.v_proj(donor.detach()))
        a = torch.softmax(q @ k.transpose(-1, -2) * attn.scaling + mask, -1)
        ad = torch.softmax(q @ kd.transpose(-1, -2) * attn.scaling + mask, -1)
        context = (a @ v + .3 * ((ad-a) @ (vd-v))) * head_mask.view(1, 2, 1, 1)
        expected = attn.out_proj(context.transpose(1, 2).reshape(2, 3, 16))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        actual.square().sum().backward()
        # Splitting a differentiable packed tensor can allocate zero gradient
        # for the detached donor's half through CatBackward.
        self.assertTrue(donor.grad is None or torch.count_nonzero(donor.grad).item() == 0)
        self.assertGreater(recipient.grad.norm().item(), 0)
        for parameter in (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)

    def test_checkpoint_recomputation_matches_all_parameter_gradients(self):
        self.model.train()
        self.model.correction_attention.gates.data.fill_(.2)
        torch.manual_seed(123)
        reference = self.model(**batch()).loss
        reference.backward()
        gradients = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.grad is not None}
        for reentrant in (True, False):
            self.model.zero_grad(set_to_none=True)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': reentrant})
            torch.manual_seed(123)
            result = self.model(**batch()).loss
            result.backward()
            torch.testing.assert_close(reference, result)
            for name, parameter in self.model.named_parameters():
                if name in gradients:
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-7, rtol=2e-4)
            self.model.gradient_checkpointing_disable()

    def test_full_sequence_matches_incremental_cache_and_reorder(self):
        self.model.eval(); self.model.correction_attention.gates.data.fill_(.4)
        data = batch(); ids = torch.tensor([[2, 4, 6, 8], [2, 5, 7, 9]])
        with torch.no_grad():
            memory = self.model.encode(data['input_ids'], data['attention_mask'])
            full = self.model(encoder_outputs=memory, attention_mask=data['attention_mask'],
                              decoder_input_ids=ids, use_cache=False).logits
            cache = None; parts = []
            for index in range(ids.shape[1]):
                result = self.model(encoder_outputs=memory, attention_mask=data['attention_mask'],
                    decoder_input_ids=ids[:, index:index+1], past_key_values=cache, use_cache=True)
                cache = result.past_key_values; parts.append(result.logits)
            torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-7, rtol=2e-5)
            key = cache.cross_attention_cache.layers[1].keys.clone()
            self.assertEqual(key.shape[-1], 16)  # two head_dim=8 K tensors packed together
            order = torch.tensor([1, 0]); cache.reorder_cache(order)
            torch.testing.assert_close(cache.cross_attention_cache.layers[1].keys, key[order])
            reordered_memory = DepthEncoderOutput(last_hidden_state=memory.last_hidden_state[order],
                                                  donor_hidden_state=memory.donor_hidden_state[order])
            next_ids = torch.tensor([[10], [11]])
            next_result = self.model(encoder_outputs=reordered_memory,
                attention_mask=data['attention_mask'][order], decoder_input_ids=next_ids,
                past_key_values=cache, use_cache=True).logits
            reference = self.model(encoder_outputs=reordered_memory,
                attention_mask=data['attention_mask'][order],
                decoder_input_ids=torch.cat((ids[order], next_ids), 1), use_cache=False).logits[:, -1:]
            torch.testing.assert_close(next_result, reference, atol=2e-7, rtol=2e-5)

    def test_greedy_and_beam_generation_cache_and_native_zero_gate(self):
        self.model.eval(); self.native.eval(); data = batch(); data.pop('labels')
        for beams in (1, 4):
            args = dict(max_new_tokens=5, num_beams=beams, do_sample=False)
            baseline = self.native.generate(**data, **args)
            self.assertTrue(torch.equal(baseline, self.model.generate(**data, **args)))
            self.model.correction_attention.gates.data.fill_(.4)
            cached = self.model.generate(**data, **args, use_cache=True)
            uncached = self.model.generate(**data, **args, use_cache=False)
            self.assertTrue(torch.equal(cached, uncached))
            self.model.correction_attention.gates.data.zero_()

    def test_padding_and_causality(self):
        self.model.eval(); self.model.correction_attention.gates.data.fill_(.4)
        data = batch(); data.pop('labels'); decoder = torch.tensor([[2, 6, 7, 8], [2, 9, 10, 11]])
        with torch.no_grad():
            original = self.model(**data, decoder_input_ids=decoder).logits
            source = data['input_ids'].clone(); source[0, -1] = 42
            padded = self.model(input_ids=source, attention_mask=data['attention_mask'],
                                decoder_input_ids=decoder).logits
            torch.testing.assert_close(padded, original, atol=2e-7, rtol=2e-5)
            changed = decoder.clone(); changed[:, -1] = 30
            future = self.model(**data, decoder_input_ids=changed).logits
            torch.testing.assert_close(future[:, :-1], original[:, :-1], atol=2e-7, rtol=2e-5)

    def test_precomputed_memory_and_multiple_forwards_no_stale_donor(self):
        self.model.eval(); self.native.eval(); self.model.correction_attention.gates.data.fill_(.4)
        with torch.no_grad():
            first = self.model(**batch())
            memory = self.model.encode(batch()['input_ids'], batch()['attention_mask'])
            torch.testing.assert_close(memory.last_hidden_state,
                                       self.native.get_encoder()(batch()['input_ids'],
                                           attention_mask=batch()['attention_mask']).last_hidden_state)
            self.assertFalse(memory.donor_hidden_state.requires_grad)
            smaller = {key: value[:1, :3] for key, value in batch().items()}
            self.model(**smaller)
            replay = self.model(encoder_outputs=memory, attention_mask=batch()['attention_mask'],
                                labels=batch()['labels'])
            torch.testing.assert_close(first.logits, replay.logits)
            self.assertEqual(self.model(**batch(), output_hidden_states=True).encoder_hidden_states[0].shape[-1], 16)
            self.assertIsInstance(self.model(**batch(), return_dict=False), tuple)

    def test_strict_save_load_retains_gates_logits_and_generation(self):
        self.model.eval(); self.model.correction_attention.gates.data.copy_(torch.tensor([.3, -.2]))
        with torch.no_grad(): reference = self.model(**batch()).logits
        data = batch(); data.pop('labels')
        generated = self.model.generate(**data, max_new_tokens=4, num_beams=4)
        with tempfile.TemporaryDirectory() as folder:
            self.model.save_pretrained(folder)
            self.assertFalse((Path(folder)/'model.safetensors').exists())
            rebuilt = DepthDifferencePrototype.from_pretrained(folder).eval()
            with torch.no_grad(): torch.testing.assert_close(reference, rebuilt(**batch()).logits)
            self.assertTrue(torch.equal(generated, rebuilt.generate(**data, max_new_tokens=4, num_beams=4)))
            with self.assertRaises(OSError):
                MBartForConditionalGeneration.from_pretrained(folder, local_files_only=True)

    def test_unsupported_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError): self.model.encode(torch.tensor([[1, 1]]), torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            self.model(encoder_outputs=self.native.get_encoder()(batch()['input_ids']),
                       labels=batch()['labels'])
        with self.assertRaises(ValueError): DepthDifferencePrototype(self.model.base_model, donor_layer=2)
        with self.assertRaises(ValueError): DepthDifferencePrototype(backbone(), donor_layer=4)
        native = backbone(); native.config.encoder_layerdrop = .1
        with self.assertRaises(ValueError): DepthDifferencePrototype(native, donor_layer=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
