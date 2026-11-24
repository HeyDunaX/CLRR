"""Collator, accumulation-window annotation, and per-epoch fork builder callback."""
import json
import math
import time
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, TrainerCallback

from .model import build_epoch_fork_table


class LexicalForkCollator:
    """Wrap DataCollatorForSeq2Seq while preserving integer train row_ids cleanly."""

    def __init__(self, tokenizer, model, pad_to_multiple_of=8):
        self._inner = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features):
        cleaned = []
        row_ids = []
        has_row_id = 'row_id' in features[0]
        for feat in features:
            copy = dict(feat)
            if has_row_id:
                row_ids.append(int(copy.pop('row_id')))
            cleaned.append(copy)
        batch = self._inner(cleaned)
        if has_row_id:
            batch['row_ids'] = torch.tensor(row_ids, dtype=torch.long)
        return batch


def annotate_window_fork_counts(batch_samples, selected_map):
    """Count total selected forks K_w across all microbatches of one accumulation window."""
    if not batch_samples:
        return 0
    if selected_map and 'row_ids' in batch_samples[0]:
        k_w = sum(
            1
            for b in batch_samples
            for rid in b['row_ids'].tolist()
            if int(rid) in selected_map
        )
    else:
        k_w = 0
    g_w = len(batch_samples)
    for b in batch_samples:
        b['fork_window_count'] = k_w
        b['fork_window_gas'] = g_w
    return k_w


class WindowedForkDataLoader:
    """Optional standalone lazy DataLoader wrapper precomputing K_w per GA window."""

    def __init__(self, base_loader, model, gradient_accumulation_steps):
        self.base_loader = base_loader
        self.model = model
        self.gas = max(1, int(gradient_accumulation_steps))

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        it = iter(self.base_loader)
        while True:
            window = []
            for _ in range(self.gas):
                try:
                    window.append(next(it))
                except StopIteration:
                    break
            if not window:
                break
            annotate_window_fork_counts(window, self.model.selected_map)
            for b in window:
                yield b

    def __getattr__(self, name):
        return getattr(self.base_loader, name)


class LexicalForkEpochCallback(TrainerCallback):
    """Build per-epoch frozen-snapshot balanced fork tables and separate phase telemetry."""

    def __init__(
        self,
        folder,
        train_sources,
        train_refs,
        tokenizer,
        seed: int,
        row_ids=None,
        gate_mode: bool = False,
        bleu_tokenizer: str = 'zh',
    ):
        self.folder = Path(folder)
        self.train_sources = list(train_sources)
        self.train_refs = list(train_refs)
        self.row_ids = [int(x) for x in row_ids] if row_ids is not None else list(range(len(self.train_sources)))
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.gate_mode = bool(gate_mode)
        self.bleu_tokenizer = str(bleu_tokenizer)
        self._epoch_train_started = None
        self._eval_started = None
        self._current_epoch_idx = 0
        self._current_labeling_sec = 0.0
        self._current_train_sec = 0.0

    def _append_jsonl(self, name: str, payload: dict):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, default=str) + '\n')

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        unwrapped = model.module if hasattr(model, 'module') else model
        epoch_idx = int(math.floor(float(state.epoch or 0.0) + 1e-6)) + 1
        self._current_epoch_idx = epoch_idx
        self._current_labeling_sec = 0.0

        if self.gate_mode:
            if unwrapped.active_epoch_idx == 2 and unwrapped.selected_map:
                self._epoch_train_started = time.perf_counter()
                return
            selected_map, stats = build_epoch_fork_table(
                unwrapped,
                self.tokenizer,
                self.train_sources,
                self.train_refs,
                mode=unwrapped.mode,
                seed=self.seed,
                epoch_idx=2,
                row_ids=self.row_ids,
                bleu_tok=self.bleu_tokenizer,
                num_strata=unwrapped.num_margin_strata,
                min_pos_2x2=1,
                batch_size=unwrapped.rollout_batch_size,
                max_source_length=unwrapped.max_source_length,
                max_target_length=unwrapped.max_target_length,
            )
            if len(selected_map) < 8:
                # Gate-only fallback so Pass-2 backward & window normalization execute cleanly even on tiny/untrained slices.
                from .model import _greedy_native_with_rivals, _resolve_prefix_ids, build_lexical_boundary_masks, hash_prefix_index
                base_m = unwrapped.base_model
                dev = next(base_m.parameters()).device
                _, eos_id, pref = _resolve_prefix_ids(base_m)
                bow_m, cont_m = (
                    (unwrapped._bow_mask, unwrapped._cont_mask)
                    if unwrapped._bow_mask is not None
                    else build_lexical_boundary_masks(self.tokenizer, int(base_m.config.vocab_size), [eos_id, *pref])
                )
                was_tr = base_m.training
                base_m.eval()
                nat_g = _greedy_native_with_rivals(
                    base_m, self.tokenizer, self.train_sources, bow_m, cont_m, dev,
                    row_ids=self.row_ids, batch_size=64, max_source_length=unwrapped.max_source_length, max_new_tokens=32,
                )
                base_m.train(was_tr)
                selected_map = {}
                for idx_g, r_g in enumerate(nat_g):
                    if r_g['eligible_positions']:
                        jg = hash_prefix_index(self.seed, 2, r_g['row_id'], r_g['eligible_positions'])
                        selected_map[int(r_g['row_id'])] = dict(
                            j=jg, h=r_g['native_tokens'][:jg],
                            a=int(r_g['native_tokens'][jg]), b=int(r_g['rivals'][jg]),
                            sign=1 if (idx_g % 2 == 0) else -1, margin=float(r_g['margins'][jg]), stratum=0,
                        )
            eff_lam = unwrapped.lambda_epoch2 if selected_map else 0.0
            unwrapped.set_epoch_fork_state(2, eff_lam, selected_map)
            self._current_labeling_sec = float(stats.get('seconds_total', 0.0))
            self._append_jsonl('lexical_fork_epochs.jsonl', {**stats, 'effective_lambda': eff_lam, 'gate_mode': True, 'gate_selected': len(selected_map)})
            self._epoch_train_started = time.perf_counter()
            return

        # Real training path
        if epoch_idx == 1:
            unwrapped.set_epoch_fork_state(1, 0.0, {})
            self._append_jsonl(
                'lexical_fork_epochs.jsonl',
                dict(
                    epoch=1,
                    mode=unwrapped.mode,
                    active=False,
                    deactivated_reason='epoch1_ce_parity_phase',
                    effective_lambda=0.0,
                    selected_pos=0,
                    selected_neg=0,
                    selected_total=0,
                    seconds_total=0.0,
                ),
            )
            self._epoch_train_started = time.perf_counter()
            return

        # Preserve restored state cleanly when resuming mid-epoch from checkpoint.
        if unwrapped.active_epoch_idx == epoch_idx:
            self._epoch_train_started = time.perf_counter()
            return

        selected_map, stats = build_epoch_fork_table(
            unwrapped,
            self.tokenizer,
            self.train_sources,
            self.train_refs,
            mode=unwrapped.mode,
            seed=self.seed,
            epoch_idx=epoch_idx,
            row_ids=self.row_ids,
            bleu_tok=self.bleu_tokenizer,
            num_strata=unwrapped.num_margin_strata,
            min_pos_2x2=unwrapped.min_pos_2x2,
            batch_size=unwrapped.rollout_batch_size,
            max_source_length=unwrapped.max_source_length,
            max_target_length=unwrapped.max_target_length,
        )
        eff_lam = unwrapped.scheduled_lambda(epoch_idx) if stats['active'] else 0.0
        unwrapped.set_epoch_fork_state(epoch_idx, eff_lam, selected_map)
        self._current_labeling_sec = float(stats.get('seconds_total', 0.0))
        record = {
            **stats,
            'effective_lambda': float(eff_lam),
            'global_step_start': int(state.global_step),
        }
        self._append_jsonl('lexical_fork_epochs.jsonl', record)
        print('LEXICAL_FORK_EPOCH_TABLE', json.dumps(record), flush=True)
        self._epoch_train_started = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_train_started is not None:
            self._current_train_sec = round(time.perf_counter() - self._epoch_train_started, 2)
        self._eval_started = time.perf_counter()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_sec = round(time.perf_counter() - self._eval_started, 2) if self._eval_started is not None else 0.0
        self._append_jsonl(
            'lexical_fork_timing.jsonl',
            dict(
                epoch=int(self._current_epoch_idx),
                global_step=int(state.global_step),
                seconds_labeling=round(self._current_labeling_sec, 2),
                seconds_train_epoch=self._current_train_sec,
                seconds_eval=eval_sec,
                eval_bleu=(metrics or {}).get('eval_bleu'),
                eval_chrf0=(metrics or {}).get('eval_chrf0'),
                eval_loss=(metrics or {}).get('eval_loss'),
            ),
        )
