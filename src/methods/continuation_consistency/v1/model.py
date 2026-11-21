"""Sparse mass-preserving continuation consistency, native mBART inference."""
import json
from pathlib import Path

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...paired_route.v1.model import PairedRouteMT, paired_loss

METHODS = {'continuation-consistency-v1': 'continuation',
           'continuation-permuted-v1': 'permuted',
           'continuation-norm-control-v1': 'norm-control'}


def symmetric_kl(loga, logb):
    return .5 * ((loga.exp() - logb.exp()) * (loga - logb)).sum(-1)


def pushforward_kl(left, right, actions, log_channel):
    """Exact SKL for hybrid kernel: shortlisted rows -> future vocabulary;
    other rows -> distinct identity symbols. EOS belongs to the identity tail.
    The detached channel is stochastic; both current distributions get gradients.
    """
    a, b = left.float().log_softmax(-1), right.float().log_softmax(-1)
    q = log_channel.detach().float()
    if actions.numel() == 0 or q.shape != (actions.numel(), a.numel()):
        raise ValueError('Nonempty action/channel shape mismatch')
    ra = (a[actions, None] + q).logsumexp(0)
    rb = (b[actions, None] + q).logsumexp(0)
    tail = torch.ones_like(a, dtype=torch.bool)
    tail[actions] = False
    return symmetric_kl(ra, rb) + symmetric_kl(a[tail], b[tail])


class ContinuationConsistencyMT(PairedRouteMT):
    def __init__(self, base_model, method, top_k=4, beta=1.25, probe_chunk=2):
        super().__init__(base_model, 'paired-native-rdrop-v1')
        self.method, self.mode = method, METHODS[method]
        self.top_k, self.beta, self.probe_chunk = top_k, beta, probe_chunk
        if top_k < 1 or probe_chunk < 1 or beta < 0:
            raise ValueError('Invalid continuation recipe')

    @torch.no_grad()
    def continuation_channel(self, input_ids, attention_mask, decoder_ids, rows):
        # No dropout or gradients in K. Restore individual module flags exactly;
        # do not alter checkpoint replay context or the two-view dropout streams.
        flags = [(m, m.training) for m in self.base_model.modules()]
        try:
            self.base_model.eval()
            with self._route(False):
                encoder = self.base_model.model.encoder(input_ids=input_ids,
                    attention_mask=attention_mask, return_dict=True).last_hidden_state
                channels = []
                for row, pos, actions in rows:
                    values = []
                    for offset in range(0, actions.numel(), self.probe_chunk):
                        part = actions[offset:offset+self.probe_chunk]
                        prefix = decoder_ids[row:row+1, :pos+1].expand(len(part), -1)
                        prefix = torch.cat((prefix, part[:, None]), dim=1)
                        hidden = self.base_model.model.decoder(input_ids=prefix,
                            encoder_hidden_states=encoder[row:row+1].expand(len(part), -1, -1),
                            encoder_attention_mask=attention_mask[row:row+1].expand(len(part), -1),
                            use_cache=False, return_dict=True).last_hidden_state[:, -1]
                        score = self.base_model.lm_head(hidden) + self.base_model.final_logits_bias
                        values.append(score.float().log_softmax(-1))
                    channels.append(torch.cat(values))
                return channels
        finally:
            for module, training in flags:
                module.training = training

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                decoder_input_ids=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, labels=labels, **kwargs)
        decoder_ids = (self.prepare_decoder_input_ids_from_labels(labels)
                       if decoder_input_ids is None else decoder_input_ids)
        logits = []
        for _ in range(2):
            with self._route(False):
                logits.append(self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=decoder_ids, use_cache=False, return_dict=True).logits)
        _, ce, _ = paired_loss(*logits, labels, 0.)
        valid = labels.ne(-100)
        lengths = valid.sum(-1)
        rows = []
        # Fork the active device RNG so sampling does not perturb dropout order.
        devices = [input_ids.device.index] if input_ids.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            for row in range(labels.size(0)):
                positions = valid[row].nonzero().flatten()
                if not len(positions):
                    continue
                pos = int(positions[torch.randint(len(positions), (), device=labels.device)])
                a, b = [value[row, pos].detach() for value in logits]
                actions = torch.cat((a.topk(min(self.top_k, a.numel())).indices,
                                     b.topk(min(self.top_k, b.numel())).indices,
                                     labels[row, pos:pos+1])).unique(sorted=True)
                # Termination remains a separate identity output, never a decoder probe.
                actions = actions[actions.ne(self.config.eos_token_id)]
                rows.append((row, pos, actions))
        channels = self.continuation_channel(input_ids, attention_mask, decoder_ids, rows)
        continuation, direct, raw_direct, scales, separation, mass = [], [], [], [], [], []
        for (row, pos, actions), channel in zip(rows, channels):
            a, b = [value[row, pos] for value in logits]
            if self.mode == 'permuted':
                channel = channel.roll(1, 0)
            cmc = pushforward_kl(a, b, actions, channel)
            current = symmetric_kl(a.float().log_softmax(-1), b.float().log_softmax(-1))
            scale = a.new_tensor(1.)
            if self.mode == 'norm-control':
                # Local logit-space calibration only; never traverse backbone or
                # accumulate parameter gradients. Ratio is deliberately stop-grad.
                with torch.enable_grad():
                    x, y = a.detach().requires_grad_(), b.detach().requires_grad_()
                    c = pushforward_kl(x, y, actions, channel)
                    d = symmetric_kl(x.float().log_softmax(-1), y.float().log_softmax(-1))
                    cg = torch.autograd.grad(c, (x, y))
                    dg = torch.autograd.grad(d, (x, y))
                    cn = sum(g.square().sum() for g in cg).sqrt()
                    dn = sum(g.square().sum() for g in dg).sqrt()
                    scale = torch.where(dn > 0, cn / dn.clamp_min(1e-30), dn * 0).detach()
            continuation.append(cmc * lengths[row])
            raw_direct.append(current * lengths[row])
            direct.append(current * scale * lengths[row])
            scales.append(scale.detach())
            with torch.no_grad():
                q = channel.exp()
                separation.append((q - q.mean(0)).abs().sum(-1).mean())
                mass.append(.5 * (a.float().softmax(-1)[actions].sum() +
                                  b.float().softmax(-1)[actions].sum()))
        cmc = torch.stack(continuation).sum() / lengths.sum()
        regularizer = (torch.stack(direct).sum() / lengths.sum()
                       if self.mode == 'norm-control' else cmc)
        self.last_diagnostics = dict(ce=float(ce.detach()), kl=float(regularizer.detach()),
            continuation_kl=float(cmc.detach()), beta=self.beta,
            direct_kl=float((torch.stack(raw_direct).sum()/lengths.sum()).detach()),
            channel_separation=float(torch.stack(separation).mean()),
            selected_mass=float(torch.stack(mass).mean()),
            norm_scale=float(torch.stack(scales).mean()), sampled_positions=len(rows),
            checkpoint_context_empty=not self._checkpoint_context)
        return Seq2SeqLMOutput(loss=ce + self.beta * regularizer)

    def method_metadata(self):
        return dict(version=1, method=self.method, mode=self.mode, top_k=self.top_k,
            beta=self.beta, probe_chunk=self.probe_chunk, extra_parameters=0,
            kernel='shared detached dropout-off next-token; identity tail including EOS',
            positions='one uniform supervised position per sentence; length weighted token mean',
            shortlist='union top-k each view plus gold excluding EOS; no renormalization',
            control='per-position detached logit-gradient norm ratio' if self.mode == 'norm-control' else None,
            inference_route='native', original_lsr_weight=0.)

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'continuation_consistency_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8')

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(Path(folder, 'continuation_consistency_config.json').read_text(encoding='utf-8'))
        if saved != self.method_metadata():
            raise ValueError('Continuation checkpoint recipe mismatch')
