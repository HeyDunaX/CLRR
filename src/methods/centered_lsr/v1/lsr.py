"""FP32 microbatch centered alignment; historical U2 definition."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from ...original.lsr import JEPAGuidedSeq2SeqLM
from ...common.helpers import _masked_mean_pool

def centered_cosine_loss(source, target):
    source = source.float()
    target = target.detach().float()
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    source = F.normalize(source, dim=-1, eps=1e-6)
    target = F.normalize(target, dim=-1, eps=1e-6)
    return (1. - (source * target).sum(dim=-1)).mean()


class CenteredLSRSeq2SeqLM(JEPAGuidedSeq2SeqLM):
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, output_hidden_states=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
            labels=labels, output_hidden_states=True, **kwargs)
        if labels is None or self.jepa_weight <= 0:
            return outputs
        source_hidden = outputs.encoder_last_hidden_state
        pad = getattr(self.config, 'pad_token_id', 0) or 0
        target_ids = labels.masked_fill(labels == -100, pad)
        target_mask = (target_ids != pad).long()
        cache = self.base_model._fresh_cache() if hasattr(self.base_model, '_fresh_cache') else nullcontext()
        with cache, torch.no_grad():
            target = self.get_encoder()(input_ids=target_ids, attention_mask=target_mask)
            target_hidden = target.last_hidden_state.detach()
        loss = centered_cosine_loss(_masked_mean_pool(source_hidden.float(), attention_mask),
                                    _masked_mean_pool(target_hidden.float(), target_mask))
        if outputs.loss is not None:
            outputs.loss = outputs.loss + self.jepa_weight * loss.to(outputs.loss.dtype)
        return outputs
