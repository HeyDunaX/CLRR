"""Decision-sensitive value attenuation, last mBART decoder cross-attention only.

Correctness-first generation reconstructs beam-local memory from each emitted
prefix. Native KV caching is disabled explicitly, not represented as supported.
This has extra sequential/autograd cost; admission must measure it before train.
"""
import json
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...paired_route.v1.model import PairedRouteMT, paired_loss


def update_memory(memory, attention, direction, eta, active):
    support = (memory * direction.unsqueeze(-2)).sum(-1).clamp_min(0)
    amount = eta * attention * support * active[:, None, None]
    return memory - amount.unsqueeze(-1) * direction.unsqueeze(-2)


class DecisionResidualMemoryMT(PairedRouteMT):
    def __init__(self, base_model, eta=.1, beta=1.25):
        if base_model.config.model_type != 'mbart' or not 0 <= eta <= 1:
            raise ValueError('DRM v1 requires mBART and eta in [0,1]')
        super().__init__(base_model, 'paired-native-rdrop-v1')
        self.method, self.eta, self.beta = 'decision-residual-memory-v1', eta, beta
        self._memory_labels, self._memory_decoder = None, None
        layer = base_model.get_decoder().layers[-1]
        self._native_last_forward = layer.forward
        owner = self

        def modified(layer_self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                     encoder_attention_mask=None, layer_head_mask=None, cross_attn_layer_head_mask=None,
                     past_key_values=None, output_attentions=False, use_cache=True, cache_position=None, **kwargs):
            if owner.eta == 0:
                return owner._native_last_forward(hidden_states, attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=layer_head_mask, cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_values=past_key_values, output_attentions=output_attentions,
                    use_cache=use_cache, cache_position=cache_position, **kwargs)
            return owner._last_layer(layer_self, hidden_states, attention_mask, encoder_hidden_states,
                encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_values,
                output_attentions, cache_position)
        layer.forward = MethodType(modified, layer)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        super().gradient_checkpointing_enable(*args, **kwargs)
        layer = self.base_model.get_decoder().layers[-1]
        native = layer._gradient_checkpointing_func
        def captured(function, *values, **options):
            labels, decoder = self._memory_labels, self._memory_decoder
            def replay(*inputs):
                old = self._memory_labels, self._memory_decoder
                self._memory_labels, self._memory_decoder = labels, decoder
                try:
                    return function(*inputs)
                finally:
                    self._memory_labels, self._memory_decoder = old
            return native(replay, *values, **options)
        layer._gradient_checkpointing_func = captured

    def _last_layer(self, layer, hidden, mask, encoder, encoder_mask, head_mask, cross_mask,
                    past, output_attentions, cache_position):
        if past is not None or output_attentions or head_mask is not None or cross_mask is not None:
            raise ValueError('DRM v1 uses prefix replay, no past KV cache/head masks/attention outputs')
        if encoder is None or self._memory_decoder is None:
            raise ValueError('DRM needs source and full decoder prefix')
        record_graph = torch.is_grad_enabled()
        # Reentrant checkpoint first pass runs under no_grad, but a local VJP
        # is needed to define the forward algorithm. Replay captures labels.
        with torch.enable_grad():
            residual = hidden
            normalized = layer.self_attn_layer_norm(hidden)
            self_output, _ = layer.self_attn(hidden_states=normalized, attention_mask=mask,
                output_attentions=False, cache_position=cache_position)
            hidden = residual + F.dropout(self_output, p=layer.dropout, training=layer.training)
            residual = hidden
            attention = layer.encoder_attn
            batch, length, _ = hidden.shape
            heads, dim = attention.num_heads, attention.head_dim
            normalized = layer.encoder_attn_layer_norm(hidden)
            query = attention.q_proj(normalized).view(batch,length,heads,dim).transpose(1,2)
            key = attention.k_proj(encoder).view(batch,-1,heads,dim).transpose(1,2)
            memory = attention.v_proj(encoder).view(batch,-1,heads,dim).transpose(1,2)
            outputs, norms, changes = [], [], []
            if not record_graph:
                query, key, memory, residual = (x.detach() for x in (query,key,memory,residual))
            for t in range(length):
                q = query[:,:,t:t+1]
                source_mask = encoder_mask
                if source_mask is not None and source_mask.shape[-2] != 1:
                    source_mask = source_mask[:,:,t:t+1]
                context = F.scaled_dot_product_attention(q,key,memory,attn_mask=source_mask,
                    dropout_p=attention.dropout if layer.training else 0., is_causal=False,
                    scale=attention.scaling)
                if not context.requires_grad:
                    context = context.detach().requires_grad_(True)
                merged = context.transpose(1,2).reshape(batch,1,heads*dim)
                cross = attention.out_proj(merged)
                state = residual[:,t:t+1] + F.dropout(cross,p=layer.dropout,training=layer.training)
                fc = layer.final_layer_norm(state)
                fc = layer.activation_fn(layer.fc1(fc))
                fc = F.dropout(fc,p=layer.activation_dropout,training=layer.training)
                fc = layer.fc2(fc)
                state = state + F.dropout(fc,p=layer.dropout,training=layer.training)
                logits = self.base_model.lm_head(self.base_model.get_decoder().layer_norm(state))
                logits = logits + self.base_model.final_logits_bias
                labels = self._memory_labels
                if labels is not None:
                    action = labels[:,t]
                elif t+1 < self._memory_decoder.shape[1]:
                    action = self._memory_decoder[:,t+1]
                else:
                    action = logits[:,0].detach().argmax(-1)
                active = action.ne(-100) & action.ne(self.config.pad_token_id) & action.ne(self.config.eos_token_id)
                score = logits[:,0].log_softmax(-1).gather(-1,action.clamp_min(0)[:,None]).sum()
                grad, = torch.autograd.grad(score,context,retain_graph=record_graph,create_graph=False)
                grad = grad[:,:,0].detach()
                norm = grad.norm(dim=-1,keepdim=True)
                direction = torch.where(norm>1e-8,grad/norm.clamp_min(1e-8),torch.zeros_like(grad))
                score_attention = (q @ key.transpose(-1,-2))*attention.scaling
                if source_mask is not None:
                    score_attention = score_attention + source_mask
                weights = score_attention.softmax(-1)[:,:,0]
                updated = update_memory(memory,weights,direction,self.eta,active)
                norms.append(norm.detach().mean())
                changes.append((updated-memory).detach().norm()/memory.detach().norm().clamp_min(1e-8))
                memory = updated if record_graph else updated.detach()
                outputs.append(state if record_graph else state.detach())
            self._last_memory_stats = dict(decision_gradient_norm=float(torch.stack(norms).mean()),
                relative_memory_change=float(torch.stack(changes).mean()),positions=length)
            result = torch.cat(outputs,dim=1)
        return (result if record_graph else result.detach(),)

    def _one_view(self,input_ids,attention_mask,decoder,labels,**kwargs):
        previous = self._memory_labels,self._memory_decoder
        self._memory_labels,self._memory_decoder = labels,decoder
        try:
            with self._route(False):
                return self.base_model(input_ids=input_ids,attention_mask=attention_mask,
                    decoder_input_ids=decoder,labels=None,use_cache=False,return_dict=True,**kwargs)
        finally:
            self._memory_labels,self._memory_decoder = previous

    def forward(self,input_ids=None,attention_mask=None,labels=None,decoder_input_ids=None,**kwargs):
        kwargs.pop('num_items_in_batch',None)
        kwargs.pop('use_cache',None)
        kwargs.pop('return_dict',None)
        if kwargs.get('past_key_values') is not None:
            raise ValueError('DRM v1 must receive full prefixes, not past KV state')
        decoder = self.prepare_decoder_input_ids_from_labels(labels) if decoder_input_ids is None and labels is not None else decoder_input_ids
        if decoder is None:
            raise ValueError('Supply labels or full decoder prefix')
        if self.training and labels is not None:
            logits = [self._one_view(input_ids,attention_mask,decoder,labels,**kwargs).logits for _ in range(2)]
            loss,ce,kl = paired_loss(*logits,labels,self.beta)
            self.last_diagnostics = dict(ce=float(ce.detach()),kl=float(kl.detach()),beta=self.beta,
                eta=self.eta,checkpoint_context_empty=not self._checkpoint_context,
                **getattr(self,'_last_memory_stats',{}))
            return Seq2SeqLMOutput(loss=loss)
        output = self._one_view(input_ids,attention_mask,decoder,labels,**kwargs)
        if labels is not None:
            output['loss'] = F.cross_entropy(output.logits.reshape(-1,output.logits.shape[-1]),labels.reshape(-1))
        return output

    def generate(self,*args,**kwargs):
        # Replay the actual selected prefix for each beam. No persistent state
        # can leak between beams, input sentences, cached/uncached requests.
        # Seq2SeqTrainer passes labels for scoring; generation must never use
        # reference tokens to update the decision memory.
        kwargs.pop('labels',None)
        previous = self.base_model.forward
        previous_prepare = self.base_model.prepare_inputs_for_generation
        def prepare(input_ids,*values,**options):
            options['use_cache'] = False
            options.pop('past_key_values',None)
            return previous_prepare(input_ids,*values,**options)
        owner = self
        def forward(base_self,input_ids=None,attention_mask=None,decoder_input_ids=None,
                    labels=None,use_cache=False,return_dict=True,**options):
            old = owner._memory_labels,owner._memory_decoder
            owner._memory_labels,owner._memory_decoder = None,decoder_input_ids
            try:
                return previous(input_ids=input_ids,attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,use_cache=False,return_dict=return_dict,**options)
            finally:
                owner._memory_labels,owner._memory_decoder = old
        self.base_model.forward = MethodType(forward,self.base_model)
        self.base_model.prepare_inputs_for_generation = prepare
        kwargs['use_cache'] = False
        try:
            return self.base_model.generate(*args,**kwargs)
        finally:
            self.base_model.forward = previous
            self.base_model.prepare_inputs_for_generation = previous_prepare

    def method_metadata(self):
        return dict(version=1,method=self.method,eta=self.eta,beta=self.beta,extra_parameters=0,
            epsilon=1e-8,layer='last decoder cross-attention values',backend='SDPA reads; softmax for erase weights',
            state='per-head per-source values; reset each forward/view/prefix',
            decision='gold after current output in teacher forcing; actual emitted past tokens in generation',
            gradient='detached local context sensitivity; differentiable memory recurrence',
            generation='full-prefix beam-local reconstruction; KV cache disabled explicitly',
            objective='native paired CE + full SKL beta1.25',novelty='exploratory, not certification')

    def save_pretrained(self,folder,**kwargs):
        self.base_model.save_pretrained(folder,**kwargs)
        Path(folder,'decision_residual_memory_config.json').write_text(json.dumps(self.method_metadata(),indent=2))

    def verify_checkpoint_metadata(self,folder):
        saved=json.loads(Path(folder,'decision_residual_memory_config.json').read_text())
        if saved != self.method_metadata():
            raise ValueError('DRM checkpoint recipe mismatch')
