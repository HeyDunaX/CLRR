"""CMC control-variate estimator and a plain sampled-SKL matched control.

An estimator refinement, not an independent novelty claim. V1 stays unchanged.
Both arms retain detached channel probes to match the original CMC compute.
"""
import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from ..v1.model import ContinuationConsistencyMT, pushforward_kl, symmetric_kl
from ...paired_route.v1.model import paired_loss

METHODS = {'continuation-control-variate-v2': 'control-variate',
           'continuation-sampled-direct-v2': 'sampled-direct'}


class EstimatorConsistencyMT(ContinuationConsistencyMT):
    def __init__(self, base_model, method, top_k=4, beta=1.25, probe_chunk=2):
        super().__init__(base_model, 'continuation-consistency-v1', top_k, beta, probe_chunk)
        self.method,self.mode = method,METHODS[method]

    def forward(self,input_ids=None,attention_mask=None,labels=None,
                decoder_input_ids=None,**kwargs):
        kwargs.pop('num_items_in_batch',None)
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids,attention_mask=attention_mask,
                labels=labels,decoder_input_ids=decoder_input_ids,**kwargs)
        decoder = (self.prepare_decoder_input_ids_from_labels(labels)
                   if decoder_input_ids is None else decoder_input_ids)
        logits=[]
        for _ in range(2):
            with self._route(False):
                logits.append(self.base_model(input_ids=input_ids,attention_mask=attention_mask,
                    decoder_input_ids=decoder,use_cache=False,return_dict=True).logits)
        _,ce,dense = paired_loss(*logits,labels,1. if self.mode=='control-variate' else 0.)
        valid=labels.ne(-100);lengths=valid.sum(-1)
        rows=[]
        devices=[input_ids.device.index] if input_ids.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            for row in range(labels.size(0)):
                positions=valid[row].nonzero().flatten()
                if not len(positions):
                    continue
                pos=int(positions[torch.randint(len(positions),(),device=labels.device)])
                a,b=[v[row,pos].detach() for v in logits]
                actions=torch.cat((a.topk(min(self.top_k,a.numel())).indices,
                    b.topk(min(self.top_k,b.numel())).indices,labels[row,pos:pos+1])).unique(sorted=True)
                actions=actions[actions.ne(self.config.eos_token_id)]
                rows.append((row,pos,actions))
        # Same probes in the plain sampled control: explicit compute match.
        channels=self.continuation_channel(input_ids,attention_mask,decoder,rows)
        continuation=[];direct=[];separation=[];mass=[]
        for (row,pos,actions),channel in zip(rows,channels):
            a,b=[v[row,pos] for v in logits]
            continuation.append(lengths[row]*pushforward_kl(a,b,actions,channel))
            direct.append(lengths[row]*symmetric_kl(a.float().log_softmax(-1),b.float().log_softmax(-1)))
            with torch.no_grad():
                q=channel.exp()
                separation.append((q-q.mean(0)).abs().sum(-1).mean())
                mass.append(.5*(a.float().softmax(-1)[actions].sum()+b.float().softmax(-1)[actions].sum()))
        cmc=torch.stack(continuation).sum()/lengths.sum()
        sampled=torch.stack(direct).sum()/lengths.sum()
        # Negative estimates are possible. Clamping would bias the surrogate.
        auxiliary=dense+cmc-sampled if self.mode=='control-variate' else sampled
        self.last_diagnostics=dict(ce=float(ce.detach()),kl=float(auxiliary.detach()),
            continuation_kl=float(cmc.detach()),direct_kl=float(sampled.detach()),
            dense_kl=float(dense.detach()) if self.mode=='control-variate' else None,
            beta=self.beta,channel_separation=float(torch.stack(separation).mean()),
            selected_mass=float(torch.stack(mass).mean()),sampled_positions=len(rows),
            checkpoint_context_empty=not self._checkpoint_context)
        return Seq2SeqLMOutput(loss=ce+self.beta*auxiliary)

    def method_metadata(self):
        result=super().method_metadata()
        result.update(version=2,mode=self.mode,control=None,
            estimator='dense direct SKL + sampled(CMC-direct SKL)' if self.mode=='control-variate'
                else 'plain unscaled sampled direct SKL',
            negative_estimates_allowed=self.mode=='control-variate',
            channel_probe_compute_matched=True,novelty_claim=False)
        return result
