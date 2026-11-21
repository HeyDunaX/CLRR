"""View-axis gradient projection; forward KL is ordinary R-Drop, not a new loss."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...paired_route.v1.model import PairedRouteMT


def antisymmetric_loss(left, right, labels, beta=1.25, chunk_size=32):
    if left.shape != right.shape or left.shape[:-1] != labels.shape:
        raise ValueError('Expected two [batch,time,vocab] logits and labels')
    valid_count = labels.ne(-100).sum()
    if not valid_count or chunk_size < 1:
        raise ValueError('Require supervised tokens and positive chunk size')
    a, b, y = left.flatten(0, 1), right.flatten(0, 1), labels.flatten()
    ce, kl = left.new_zeros(()), left.new_zeros(())
    for offset in range(0, y.numel(), chunk_size):
        def reduced(z1, z2, target):
            z1, z2 = z1.float(), z2.float()
            mask = target.ne(-100)
            safe = target.masked_fill(~mask, 0).unsqueeze(-1)
            native1, native2 = z1.log_softmax(-1), z2.log_softmax(-1)
            ce = -.5 * (native1.gather(-1, safe) + native2.gather(-1, safe)).squeeze(-1)
            mean, difference = (z1 + z2) / 2, (z1 - z2) / 2
            p = F.log_softmax(mean.detach() + difference, -1)
            q = F.log_softmax(mean.detach() - difference, -1)
            kl = .5 * ((p.exp() - q.exp()) * (p - q)).sum(-1)
            return (ce * mask).sum(), (kl * mask).sum()
        args = (a[offset:offset+chunk_size], b[offset:offset+chunk_size], y[offset:offset+chunk_size])
        c, k = (checkpoint(reduced, *args, use_reentrant=True, preserve_rng_state=True)
                if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad) else reduced(*args))
        ce, kl = ce + c, kl + k
    ce, kl = ce / valid_count, kl / valid_count
    return ce + beta * kl, ce, kl


class AntisymmetricConsistencyMT(PairedRouteMT):
    def __init__(self, base_model, beta=1.25):
        super().__init__(base_model, 'paired-native-rdrop-v1')
        self.method, self.beta = 'antisymmetric-consistency-v1', beta

    def forward(self, input_ids=None, attention_mask=None, labels=None, decoder_input_ids=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, decoder_input_ids=decoder_input_ids, **kwargs)
        decoder = self.prepare_decoder_input_ids_from_labels(labels) if decoder_input_ids is None else decoder_input_ids
        logits = []
        for _ in range(2):
            with self._route(False):
                logits.append(self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=decoder, use_cache=False, return_dict=True).logits)
        loss, ce, kl = antisymmetric_loss(*logits, labels, self.beta)
        self.last_diagnostics = dict(ce=float(ce.detach()), kl=float(kl.detach()), beta=self.beta,
            views=2, checkpoint_context_empty=not self._checkpoint_context)
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        return dict(version=1, method=self.method, beta=self.beta, extra_parameters=0,
            ce='native two-view valid-token mean', auxiliary='full-vocabulary symmetric KL',
            backward='stop-gradient logit midpoint; antisymmetric view-axis gradients',
            inference='native; no additional parameters', novelty='engineering candidate, not certification')

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'antisymmetric_consistency_config.json').write_text(json.dumps(self.method_metadata(), indent=2))

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'antisymmetric_consistency_config.json').read_text())
        if saved != self.method_metadata():
            raise ValueError('Antisymmetric consistency checkpoint recipe mismatch')
