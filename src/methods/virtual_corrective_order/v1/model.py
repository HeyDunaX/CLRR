"""Native mBART with a reset virtual-coordinate training regularizer."""
import json
import math
from pathlib import Path

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from ...original.lsr import JEPAGuidedSeq2SeqLM
from .objective import paired_ce_kl, order_penalty

METHODS = {'virtual-corrective-order-v1'}


class VirtualCorrectiveOrderMT(JEPAGuidedSeq2SeqLM):
    def __init__(self, base_model, *, rank, basis_seed, eta, coefficient, chunk=16):
        super().__init__(base_model, jepa_weight=0.)
        if not (0 < rank <= base_model.config.d_model and eta > 0 and coefficient > 0
                and math.isfinite(coefficient) and chunk > 0):
            raise ValueError('Invalid virtual-order recipe')
        self.method = 'virtual-corrective-order-v1'
        self.rank, self.basis_seed = rank, basis_seed
        self.eta, self.coefficient, self.chunk = eta, coefficient, chunk
        basis = torch.linalg.qr(torch.randn(base_model.config.d_model, rank,
            generator=torch.Generator().manual_seed(basis_seed))).Q
        self.register_buffer('virtual_basis', basis, persistent=False)
        self.microbatch_calls = 0
        self.last_diagnostics = {}

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, decoder_attention_mask=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return self.base_model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,
                decoder_input_ids=decoder_input_ids,decoder_attention_mask=decoder_attention_mask,**kwargs)
        if len(labels) != 4:
            raise ValueError('Locked pilot requires exactly four rows per training microbatch')
        decoder_input_ids = (self.prepare_decoder_input_ids_from_labels(labels)
            if decoder_input_ids is None else decoder_input_ids)
        hs = [self.base_model.model(input_ids=input_ids,attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,decoder_attention_mask=decoder_attention_mask,
            use_cache=False,return_dict=True).last_hidden_state for _ in range(2)]
        w, b = self.base_model.lm_head.weight, self.base_model.final_logits_bias.squeeze(0)
        support, probe = ([0,1],[2,3]) if self.microbatch_calls%2 == 0 else ([2,3],[0,1])
        ce = paired_ce_kl(*hs,labels,w,b,self.chunk,True,False)[0]
        order, ab, ba = order_penalty(*hs,labels,support,probe,w,b,self.virtual_basis,
            self.eta,self.chunk,True)
        weighted = self.coefficient * order
        loss = ce + weighted
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite virtual-order objective; stop without changing recipe')
        self.last_diagnostics = dict(ce=float(ce.detach()),order=float(order.detach()),
            weighted_order=float(weighted.detach()),coefficient=self.coefficient,
            endpoint_gap=float((ab-ba).detach().norm()),microbatch=self.microbatch_calls,
            support_rows=support,probe_rows=probe,virtual_steps_committed=0)
        self.microbatch_calls += 1
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        return dict(method=self.method,rank=self.rank,basis_seed=self.basis_seed,
            eta=self.eta,coefficient=self.coefficient,chunk=self.chunk,
            partitions='alternating rows01/23; cursor restored with checkpoint',
            ce='mean two native dropout views, all non-PAD tokens',
            order='mean SKL on disjoint probe rows after support AB/BA virtual steps',
            inference='native; no virtual matrix retained',extra_parameters=0)

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder,'virtual_order_config.json').write_text(json.dumps(dict(
            method=self.method_metadata(),microbatch_calls=self.microbatch_calls),indent=2))

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder,'virtual_order_config.json').read_text())
        if saved['method'] != self.method_metadata():
            raise ValueError('Virtual-order checkpoint recipe mismatch')
        self.microbatch_calls = saved['microbatch_calls']
