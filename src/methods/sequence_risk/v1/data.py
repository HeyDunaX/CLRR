"""Pad candidate sets without passing dataset metadata into the model."""
import torch
from transformers import DataCollatorForSeq2Seq


def normalize_pool_candidates(gold, sequences, *, decoder_start, target_bos, eos, max_length):
    """Dedup complete labels and count actual EOS lengths, not padded lengths."""
    gold = list(gold)
    choices = [gold]
    stats = dict(sample_draws=0, no_eos=0, length_cap=0, gold_duplicate_draws=0,
        non_gold_duplicate_draws=0, new_non_gold_draws=0)
    for sequence in sequences:
        if len(sequence) < 2 or sequence[0] != decoder_start or sequence[1] != target_bos:
            raise ValueError('Generated sequence must start with decoder-start and target language')
        stats['sample_draws'] += 1
        tokens = list(sequence[1:])
        if eos not in tokens:
            stats['no_eos'] += 1
            stats['length_cap'] += int(len(sequence) >= max_length)
            continue
        tokens = tokens[:tokens.index(eos) + 1]
        stats['length_cap'] += int(len(tokens) + 1 >= max_length)
        if tokens == gold:
            stats['gold_duplicate_draws'] += 1
        elif tokens in choices:
            stats['non_gold_duplicate_draws'] += 1
        else:
            stats['new_non_gold_draws'] += 1
            choices.append(tokens)
    return choices, stats


class SequenceRiskCollator:
    def __init__(self, tokenizer, model, pad_to_multiple_of=8):
        self.base = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
            pad_to_multiple_of=pad_to_multiple_of)
        self.multiple = pad_to_multiple_of

    def __call__(self, features):
        names = {'candidate_labels', 'candidate_costs'}
        batch = self.base([{k: v for k, v in item.items() if k not in names} for item in features])
        present = ['candidate_labels' in item for item in features]
        if not any(present):
            return batch
        if not all(present):
            raise ValueError('Cannot mix training pool rows and validation rows')
        candidates = max(len(item['candidate_labels']) for item in features)
        length = max(len(labels) for item in features for labels in item['candidate_labels'])
        length = ((length + self.multiple - 1) // self.multiple) * self.multiple
        labels = torch.full((len(features), candidates, length), -100, dtype=torch.long)
        costs = torch.zeros((len(features), candidates), dtype=torch.float32)
        valid = torch.zeros((len(features), candidates), dtype=torch.bool)
        for row, item in enumerate(features):
            if not item['candidate_labels'] or len(item['candidate_labels']) != len(item['candidate_costs']):
                raise ValueError('Invalid candidate set')
            for index, (tokens, cost) in enumerate(zip(item['candidate_labels'], item['candidate_costs'])):
                if not tokens or not 0 <= cost <= 1:
                    raise ValueError('Need nonempty candidate tokens and normalized chrF0 cost')
                labels[row, index, :len(tokens)] = torch.tensor(tokens)
                costs[row, index] = cost
                valid[row, index] = True
        batch.update(candidate_labels=labels, candidate_costs=costs, candidate_valid=valid)
        return batch
