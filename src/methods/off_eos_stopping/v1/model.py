"""Off-EOS Trajectory Huber Margin Regularization (`off_eos_stopping` v1) and Length-Only Control.

Candidate B (`off-eos-chrf-v1`) and matched causal falsifier Control B1 (`off-eos-length-v1`):
  - Keeps clean gold teacher-forced CE unchanged (`L_CE`).
  - Runs deterministic eval-mode greedy lexical rollout (with KV-cache and anti-repetition
    horizon guard) continuing past natural EOS to capture alternative stopping points.
  - Scores prefix stopping candidates by sentence chrF0 (`off-eos-chrf-v1`) or gold lexical
    body length (`off-eos-length-v1`), then executes Pass 2 (with gradients) over the
    self-generated trajectory and optimizes soft-min Huber-smoothed stopping margin loss
    on m_t = z_{t,EOS} - max_{v in V_lex} z_{t,v}.

Inference (eval / generate) delegates directly to the unwrapped native backbone with zero
extra parameters and zero inference overhead.
"""
import json
import math
from pathlib import Path

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...compositional_consistency.v1.model import chunked_ce_loss


METHODS = {
    'off-eos-chrf-v1': 'chrf',
    'off-eos-length-v1': 'length',
}


def huber_one_sided(x):
    """Huber-1 smoothed one-sided hinge: 0.5*u^2 for u in [0,1), u - 0.5 for u >= 1."""
    u = F.relu(x)
    return torch.where(u < 1.0, 0.5 * u.square(), u - 0.5)


def extract_eos_margins(logits, eos_token_id, non_lexical_ids):
    """Compute m_t = z_{EOS} - max_{v in V_lex} z_v masking all special/non-lexical IDs.

    Preserves FP64 precision when given FP64 logits; uses FP32 otherwise.
    Gradient is strictly zero on every token outside {eos_token_id, argmax_{V_lex} z}.
    """
    vocab = int(logits.shape[-1])
    compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    logits_c = logits.to(compute_dtype)

    eos_idx = int(eos_token_id)
    if eos_idx < 0 or eos_idx >= vocab:
        raise ValueError(f'eos_token_id={eos_idx} outside vocabulary size {vocab}')

    non_lex_mask = torch.zeros(vocab, dtype=torch.bool, device=logits.device)
    for tid in non_lexical_ids:
        t_int = int(tid)
        if 0 <= t_int < vocab:
            non_lex_mask[t_int] = True
    non_lex_mask[eos_idx] = True
    if bool(non_lex_mask.all()):
        raise ValueError('No lexical vocabulary tokens remain after excluding special tokens')

    z_eos = logits_c[..., eos_idx]
    max_lex = logits_c.masked_fill(non_lex_mask, float('-inf')).max(dim=-1).values
    return z_eos - max_lex


def off_eos_stop_loss(margins_1d, feasible_ks, delta=0.1, tau=0.25):
    """Exact soft-min over feasible stop indices k in {1..T} of one-sided Huber stopping costs.

    Gradient on suffix positions t > max(feasible_ks) is identically 0.0.
    """
    if margins_1d.ndim != 1:
        raise ValueError('margins_1d must be a 1-D tensor of shape [T + 1]')
    if not feasible_ks:
        raise ValueError('feasible_ks must be non-empty')
    n_steps = int(margins_1d.numel())
    compute_dtype = torch.float64 if margins_1d.dtype == torch.float64 else torch.float32
    m = margins_1d.to(compute_dtype)
    delta_f = float(delta)
    tau_f = float(tau)
    if tau_f <= 0.0:
        raise ValueError('tau must be strictly positive')

    costs = []
    for k in feasible_ks:
        k_int = int(k)
        if k_int < 1 or k_int >= n_steps:
            raise ValueError(f'Stop index k={k_int} invalid for margins length {n_steps}')
        prefix_cost = huber_one_sided(m[:k_int] + delta_f).sum()
        stop_cost = huber_one_sided(delta_f - m[k_int])
        costs.append(prefix_cost + stop_cost)

    stacked = torch.stack(costs)
    return -tau_f * (torch.logsumexp(-stacked / tau_f, dim=0) - math.log(len(costs)))


def detect_repetition_horizon(tokens, native_eos_step=None, min_keep=1):
    """Detect degenerate 1-gram or 2-gram tail repetition and return trimmed length + flag."""
    n = len(tokens)
    if n <= min_keep:
        return n, False
    post_eos_gate = max(4, int(native_eos_step)) if native_eos_step is not None else 4
    for i in range(2, n):
        # Global safeguard against 1-gram quadruple loops anywhere in the rollout.
        if i >= 3 and tokens[i] == tokens[i - 1] == tokens[i - 2] == tokens[i - 3]:
            return max(min_keep, i), True
        if i >= post_eos_gate:
            if tokens[i] == tokens[i - 1] == tokens[i - 2]:
                return max(min_keep, i), True
            if i >= 3 and tokens[i] == tokens[i - 2] and tokens[i - 1] == tokens[i - 3]:
                return max(min_keep, i), True
    return n, False


def select_chrf_stop_set(traj_tokens, ref_text, decode_fn, chrf_band=0.5, chrf_metric=None):
    """Select stop indices k in {1..T} within `chrf_band` of peak sentence chrF0.

    Returns `(feasible_ks, max_score, best_k)`. If `ref_text` is empty or `max_score <= 0`,
    returns `([], 0.0, None)` so ungrounded rows are cleanly skipped.
    """
    ref_clean = ref_text.strip()
    if not traj_tokens or not ref_clean:
        return [], 0.0, None
    metric = chrf_metric if chrf_metric is not None else sacrebleu.metrics.CHRF(word_order=0)
    scores = []
    for k in range(1, len(traj_tokens) + 1):
        hyp = decode_fn(traj_tokens[:k]).strip()
        sc = float(metric.sentence_score(hyp, [ref_clean]).score) if hyp else 0.0
        scores.append(sc)
    max_score = max(scores)
    if max_score <= 0.0:
        return [], 0.0, None
    best_k = scores.index(max_score) + 1
    cutoff = max_score - float(chrf_band)
    feasible = [k for k, sc in enumerate(scores, 1) if sc >= cutoff]
    return feasible, float(max_score), int(best_k)


class OffEosStoppingMT(nn.Module):
    """Unidirectional Off-EOS Trajectory Huber Margin wrapper (zero extra parameters)."""

    accepts_loss_kwargs = False

    def __init__(
        self,
        base_model,
        method='off-eos-chrf-v1',
        stop_weight=0.10,
        stop_margin=0.1,
        stop_tau=0.25,
        chrf_band=0.5,
        extra_past_eos=12,
        max_rollout_cap=96,
        max_len_ratio=1.5,
        stop_warmup_fraction=0.10,
        max_target_length=256,
        logit_chunk_tokens=32,
        tokenizer=None,
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown off-eos stopping method: {method}')
        mode = METHODS[method]
        if not math.isfinite(stop_weight) or stop_weight <= 0.0:
            raise ValueError('stop_weight must be finite and strictly positive')
        if not math.isfinite(stop_margin) or stop_margin <= 0.0:
            raise ValueError('stop_margin must be finite and strictly positive')
        if not math.isfinite(stop_tau) or stop_tau <= 0.0:
            raise ValueError('stop_tau must be finite and strictly positive')
        if not math.isfinite(chrf_band) or chrf_band < 0.0:
            raise ValueError('chrf_band must be finite and nonnegative')
        if int(extra_past_eos) < 1 or int(max_rollout_cap) < 8:
            raise ValueError('extra_past_eos must be >= 1 and max_rollout_cap >= 8')
        if not math.isfinite(max_len_ratio) or max_len_ratio < 1.0:
            raise ValueError('max_len_ratio must be finite and >= 1.0')
        if not math.isfinite(stop_warmup_fraction) or not (0.0 <= stop_warmup_fraction <= 0.5):
            raise ValueError('stop_warmup_fraction must be in [0.0, 0.5]')
        if int(max_target_length) < 4 or int(logit_chunk_tokens) < 1:
            raise ValueError('max_target_length >= 4 and logit_chunk_tokens >= 1 required')

        self.base_model = base_model
        self.method = method
        self.mode = mode
        self.stop_weight = float(stop_weight)
        self.stop_margin = float(stop_margin)
        self.stop_tau = float(stop_tau)
        self.chrf_band = float(chrf_band)
        self.extra_past_eos = int(extra_past_eos)
        self.max_rollout_cap = int(max_rollout_cap)
        self.max_len_ratio = float(max_len_ratio)
        self.stop_warmup_fraction = float(stop_warmup_fraction)
        self.max_target_length = int(max_target_length)
        self.logit_chunk_tokens = int(logit_chunk_tokens)

        self._tokenizer = None
        self._special_token_ids = ()
        if tokenizer is not None:
            self.set_tokenizer(tokenizer)
        self._chrf_metric = sacrebleu.metrics.CHRF(word_order=0)
        self._progress_step = None
        self._progress_max_steps = None
        self.last_diagnostics = {}

    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.base_model.generation_config = value

    def get_encoder(self):
        return self.base_model.get_encoder()

    def get_decoder(self):
        return self.base_model.get_decoder()

    def prepare_decoder_input_ids_from_labels(self, labels):
        if hasattr(self.base_model, 'prepare_decoder_input_ids_from_labels'):
            return self.base_model.prepare_decoder_input_ids_from_labels(labels)
        pad_id, dec_start_id, _, _ = self._resolve_special_ids()
        shifted = labels.new_full(labels.shape, pad_id)
        if labels.shape[1] > 1:
            shifted[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, pad_id)
        shifted[:, 0] = dec_start_id
        return shifted

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        return self.base_model.gradient_checkpointing_disable(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def extra_parameter_count(self):
        return 0

    def num_parameters(self, only_trainable=False, exclude_embeddings=False, **kwargs):
        if hasattr(self.base_model, 'num_parameters'):
            return self.base_model.num_parameters(
                only_trainable=only_trainable, exclude_embeddings=exclude_embeddings, **kwargs
            )
        params = [p for p in self.parameters() if (not only_trainable or p.requires_grad)]
        return sum(p.numel() for p in params)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def enable_input_require_grads(self):
        return self.base_model.enable_input_require_grads()

    def disable_input_require_grads(self):
        return self.base_model.disable_input_require_grads()

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.base_model.load_state_dict(state_dict, strict=strict)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer
        special = getattr(tokenizer, 'all_special_ids', None) or ()
        self._special_token_ids = tuple(sorted(set(int(x) for x in special)))

    def set_training_progress(self, current_step, max_steps):
        self._progress_step = int(current_step)
        self._progress_max_steps = int(max_steps) if max_steps is not None else None

    def effective_weight(self):
        if (
            self.stop_warmup_fraction <= 0.0
            or self._progress_step is None
            or self._progress_max_steps is None
            or self._progress_max_steps <= 0
        ):
            return self.stop_weight
        warmup_steps = max(1, int(round(self.stop_warmup_fraction * self._progress_max_steps)))
        scale = min(1.0, float(max(0, self._progress_step) + 1) / float(warmup_steps))
        return self.stop_weight * scale

    def _resolve_special_ids(self):
        cfg = self.base_model.config
        pad_id = cfg.pad_token_id
        eos_id = cfg.eos_token_id
        dec_start = getattr(cfg, 'decoder_start_token_id', None)
        if dec_start is None:
            dec_start = eos_id
        forced_bos = getattr(cfg, 'forced_bos_token_id', None)
        if forced_bos is None and getattr(self.base_model, 'generation_config', None) is not None:
            forced_bos = getattr(self.base_model.generation_config, 'forced_bos_token_id', None)
        if pad_id is None or eos_id is None or dec_start is None:
            raise ValueError('Backbone config must define pad_token_id, eos_token_id, and decoder_start_token_id')
        non_lex = {int(pad_id), int(eos_id), int(dec_start)}
        bos_id = getattr(cfg, 'bos_token_id', None)
        if bos_id is not None:
            non_lex.add(int(bos_id))
        if forced_bos is not None:
            non_lex.add(int(forced_bos))
        non_lex.update(self._special_token_ids)
        return int(pad_id), int(dec_start), (int(forced_bos) if forced_bos is not None else None), tuple(sorted(non_lex))

    def _extract_gold_body(self, labels, non_lex_set):
        body_tokens = []
        body_lens = []
        for b in range(int(labels.shape[0])):
            row = labels[b]
            valid = [int(x) for x in row[row.ne(-100)].tolist()]
            lex = [x for x in valid if x not in non_lex_set]
            body_tokens.append(lex)
            body_lens.append(max(1, len(lex)))
        return body_tokens, body_lens

    def _decode_tokens(self, token_ids):
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return ' '.join(str(x) for x in token_ids)

    def _forward_logits(self, input_ids, attention_mask, decoder_input_ids):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        ).logits

    def _greedy_rollout_past_eos(self, input_ids, attention_mask, gold_body_lens):
        """Deterministic eval-mode greedy lexical rollout past native EOS using KV-cache."""
        bsz = int(input_ids.shape[0])
        device = input_ids.device
        eos_id = int(self.base_model.config.eos_token_id)
        _, dec_start_id, forced_bos_id, non_lex_ids = self._resolve_special_ids()
        vocab = int(self.base_model.config.vocab_size)
        non_lex_mask = torch.zeros(vocab, dtype=torch.bool, device=device)
        for tid in non_lex_ids:
            if 0 <= tid < vocab:
                non_lex_mask[tid] = True
        if bool(non_lex_mask.all()):
            raise ValueError('No lexical vocabulary tokens remain for greedy rollout')

        prefix = [dec_start_id, forced_bos_id] if forced_bos_id is not None else [dec_start_id]
        max_len = min(self.max_target_length - len(prefix), self.max_rollout_cap)
        if max_len < 1:
            raise ValueError('Target rollout budget too small for decoder prefix')
        caps = [
            min(max_len, max(2, int(round(self.max_len_ratio * gold_body_lens[b])) + self.extra_past_eos))
            for b in range(bsz)
        ]
        batch_max_steps = max(caps)

        raw_tokens = [[] for _ in range(bsz)]
        native_eos_step = [None] * bsz
        done = [False] * bsz

        was_training = self.base_model.training
        self.base_model.eval()
        try:
            with torch.no_grad():
                encoder_outputs = self.base_model.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                step_input = torch.tensor([prefix], dtype=torch.long, device=device).expand(bsz, len(prefix))
                past_key_values = None

                for t in range(batch_max_steps):
                    out = self.base_model(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        decoder_input_ids=step_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = out.past_key_values
                    step_logits = out.logits[:, -1, :]
                    raw_top = step_logits.argmax(dim=-1)
                    lex_tok = step_logits.masked_fill(non_lex_mask, float('-inf')).argmax(dim=-1)

                    raw_list = raw_top.tolist()
                    lex_list = lex_tok.tolist()
                    all_done = True
                    for b in range(bsz):
                        if done[b]:
                            continue
                        if native_eos_step[b] is None and int(raw_list[b]) == eos_id:
                            native_eos_step[b] = t
                        raw_tokens[b].append(int(lex_list[b]))
                        cur_len = len(raw_tokens[b])
                        if cur_len >= caps[b] or (
                            native_eos_step[b] is not None and (cur_len - native_eos_step[b]) >= self.extra_past_eos
                        ):
                            done[b] = True
                        else:
                            all_done = False

                    if all_done:
                        break
                    step_input = lex_tok.unsqueeze(1)
        finally:
            self.base_model.train(was_training)

        traj_tokens = []
        rep_flags = []
        for b in range(bsz):
            keep_len, rep_hit = detect_repetition_horizon(raw_tokens[b], native_eos_step[b], min_keep=1)
            traj_tokens.append(raw_tokens[b][:keep_len])
            rep_flags.append(rep_hit)
        return traj_tokens, native_eos_step, rep_flags

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        **kwargs,
    ):
        kwargs.pop('num_items_in_batch', None)
        if not self.training or labels is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                **kwargs,
            )

        pad_id, dec_start_id, forced_bos_id, non_lex_ids = self._resolve_special_ids()
        eos_id = int(self.base_model.config.eos_token_id)
        if attention_mask is None:
            attention_mask = input_ids.ne(pad_id).long()
        dec_gold = (
            self.prepare_decoder_input_ids_from_labels(labels)
            if decoder_input_ids is None else decoder_input_ids
        )

        # Pass 1: standard teacher-forced CE on gold parallel targets.
        z_gold = self._forward_logits(input_ids, attention_mask, dec_gold)
        ce = chunked_ce_loss(z_gold, labels, chunk_size=self.logit_chunk_tokens)
        del z_gold

        # Deterministic eval-mode greedy rollout past native EOS (no gradients).
        non_lex_set = set(non_lex_ids)
        gold_body_tokens, gold_body_lens = self._extract_gold_body(labels, non_lex_set)
        traj_tokens, native_eos_steps, rep_flags = self._greedy_rollout_past_eos(
            input_ids, attention_mask, gold_body_lens
        )

        bsz = int(input_ids.shape[0])
        feasible_sets = []
        best_ks = []
        best_chrfs = []
        active_rows = []
        for b in range(bsz):
            if self.mode == 'length':
                k_gold = max(1, min(gold_body_lens[b], len(traj_tokens[b])))
                fset = [k_gold]
                feasible_sets.append(fset)
                best_ks.append(k_gold)
                active_rows.append(b)
            else:
                ref_text = self._decode_tokens(gold_body_tokens[b])
                fset, best_sc, best_k = select_chrf_stop_set(
                    traj_tokens[b],
                    ref_text,
                    self._decode_tokens,
                    chrf_band=self.chrf_band,
                    chrf_metric=self._chrf_metric,
                )
                feasible_sets.append(fset)
                if fset:
                    best_ks.append(best_k)
                    best_chrfs.append(best_sc)
                    active_rows.append(b)

        eff_w = self.effective_weight()
        if not active_rows:
            stop_loss = ce.new_zeros(())
            loss = ce + eff_w * stop_loss
        else:
            prefix = [dec_start_id, forced_bos_id] if forced_bos_id is not None else [dec_start_id]
            prefix_len = len(prefix)
            active_traj = [traj_tokens[b] for b in active_rows]
            max_t = max(len(t) for t in active_traj)
            dec_traj = torch.full((len(active_rows), prefix_len + max_t), pad_id, dtype=torch.long, device=input_ids.device)
            for i, toks in enumerate(active_traj):
                seq = prefix + toks
                dec_traj[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=input_ids.device)

            act_idx = torch.tensor(active_rows, dtype=torch.long, device=input_ids.device)
            # Pass 2 (with gradients): score stopping margins along self-generated trajectories.
            z_traj = self._forward_logits(input_ids.index_select(0, act_idx), attention_mask.index_select(0, act_idx), dec_traj)
            z_steps = z_traj[:, prefix_len - 1:, :]
            margins = extract_eos_margins(z_steps, eos_token_id=eos_id, non_lexical_ids=non_lex_ids)
            del z_traj, z_steps

            per_row_losses = [
                off_eos_stop_loss(
                    margins[i, :len(traj_tokens[b]) + 1],
                    feasible_sets[b],
                    delta=self.stop_margin,
                    tau=self.stop_tau,
                )
                for i, b in enumerate(active_rows)
            ]
            stop_loss = torch.stack(per_row_losses).mean().to(ce.dtype)
            loss = ce + eff_w * stop_loss

        self.last_diagnostics = dict(
            mode=self.mode,
            stop_weight=self.stop_weight,
            effective_weight=float(eff_w),
            ce=float(ce.detach()),
            stop_loss=float(stop_loss.detach()),
            weighted_stop=float((eff_w * stop_loss).detach()),
            active_fraction=float(len(active_rows)) / float(max(1, bsz)),
            mean_rollout_len=float(sum(len(t) for t in traj_tokens) / max(1, bsz)),
            mean_native_eos_step=float(
                sum(s if s is not None else len(t) for s, t in zip(native_eos_steps, traj_tokens)) / max(1, bsz)
            ),
            native_eos_hit_fraction=float(sum(1 for s in native_eos_steps if s is not None) / max(1, bsz)),
            mean_best_stop_k=float(sum(best_ks) / len(best_ks)) if best_ks else 0.0,
            mean_gold_body_len=float(sum(gold_body_lens) / max(1, bsz)),
            mean_feasible_size=float(sum(len(feasible_sets[b]) for b in active_rows) / len(active_rows)) if active_rows else 0.0,
            rep_triggered_fraction=float(sum(1 for r in rep_flags if r) / max(1, bsz)),
            mean_best_chrf=float(sum(best_chrfs) / len(best_chrfs)) if best_chrfs else 0.0,
        )
        return Seq2SeqLMOutput(loss=loss)

    def method_metadata(self):
        return dict(
            version=1,
            method=self.method,
            mode=self.mode,
            stop_weight=self.stop_weight,
            stop_margin=self.stop_margin,
            stop_tau=self.stop_tau,
            chrf_band=self.chrf_band,
            extra_past_eos=self.extra_past_eos,
            max_rollout_cap=self.max_rollout_cap,
            max_len_ratio=self.max_len_ratio,
            stop_warmup_fraction=self.stop_warmup_fraction,
            max_target_length=self.max_target_length,
            logit_chunk_tokens=self.logit_chunk_tokens,
            stop_target_policy='chrf0-utility-band' if self.mode == 'chrf' else 'gold-lexical-length',
            extra_parameters=0,
            external_lm=False,
            inference_route='native',
            attribution=(
                'Off-EOS Trajectory Huber Margin Regularization (Round 5 Candidate B)'
                if self.mode == 'chrf'
                else 'Matched Length-Only Control B1 on identical off-EOS greedy trajectories'
            ),
        )

    def save_pretrained(self, folder, **kwargs):
        self.base_model.save_pretrained(folder, **kwargs)
        Path(folder, 'off_eos_stopping_config.json').write_text(
            json.dumps(self.method_metadata(), indent=2), encoding='utf-8'
        )

    def verify_checkpoint_metadata(self, folder):
        saved = json.loads(
            Path(folder, 'off_eos_stopping_config.json').read_text(encoding='utf-8')
        )
        if saved != self.method_metadata():
            raise ValueError('Off-EOS stopping checkpoint recipe mismatch')
