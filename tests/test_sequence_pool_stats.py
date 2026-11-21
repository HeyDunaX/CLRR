"""Train-only pool accounting and feasibility gates; no model or GPU."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from methods.sequence_risk.v1.data import normalize_pool_candidates
from experiment_runner.sequence import aggregate_sampling_stats, assert_pool_signal


class PoolStatsGate(unittest.TestCase):
    def normalize(self, sequences, max_length=8):
        return normalize_pool_candidates([9, 4, 2], sequences,
            decoder_start=2, target_bos=9, eos=2, max_length=max_length)

    def record(self, sequences):
        labels, stats = self.normalize(sequences)
        return dict(labels=labels, costs=[index * .2 for index in range(len(labels))],
                    sampling_stats=stats)

    def test_right_padding_after_early_eos_is_not_length_cap(self):
        choices, stats = self.normalize([[2, 9, 5, 2, 1, 1, 1, 1]])
        self.assertEqual(choices, [[9, 4, 2], [9, 5, 2]])
        self.assertEqual(stats['length_cap'], 0)
        self.assertEqual(stats['no_eos'], 0)
        _, boundary = self.normalize([[2, 9, 5, 6, 7, 8, 10, 2]])
        self.assertEqual(boundary['length_cap'], 1)

    def test_no_eos_dropped_and_cap_uses_actual_draw_length(self):
        choices, stats = self.normalize([[2, 9, 5], [2, 9, 5, 6, 7, 8, 10, 11]])
        self.assertEqual(choices, [[9, 4, 2]])
        self.assertEqual(stats['sample_draws'], 2)
        self.assertEqual(stats['no_eos'], 2)
        self.assertEqual(stats['length_cap'], 1)

    def test_dedup_gold_non_gold_and_conservation(self):
        gold = [9, 4, 2]
        samples = [[2, 9, 4, 2], [2, 9, 5, 2], [2, 9, 5, 2, 1, 1],
                   [2, 9, 4, 2, 1], [2, 9, 6]]
        labels, stats = normalize_pool_candidates(gold, samples, decoder_start=2,
            target_bos=9, eos=2, max_length=8)
        self.assertEqual(labels, [[9, 4, 2], [9, 5, 2]])
        self.assertEqual(gold, [9, 4, 2])
        self.assertEqual(stats, dict(sample_draws=5, no_eos=1, length_cap=0,
            gold_duplicate_draws=2, non_gold_duplicate_draws=1, new_non_gold_draws=1))
        self.assertEqual(aggregate_sampling_stats([dict(labels=labels, sampling_stats=stats)]), stats)

    def test_invalid_generation_prefix_rejected(self):
        for tokens in ([], [2], [0, 9, 5, 2], [2, 8, 5, 2]):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, 'decoder-start'):
                self.normalize([tokens])

    def test_resumed_json_records_aggregate_identically(self):
        first = self.record([[2, 9, 4, 2], [2, 9, 5, 2]])
        second = self.record([[2, 9, 6, 2], [2, 9, 6, 2], [2, 9, 7]])
        resumed = json.loads(json.dumps([first])) + [second]
        actual = aggregate_sampling_stats(resumed)
        expected = {key: first['sampling_stats'][key] + second['sampling_stats'][key]
                    for key in first['sampling_stats']}
        self.assertEqual(actual, expected)
        self.assertEqual(actual['sample_draws'], 5)
        for field, bad in [('sample_draws', 99), ('new_non_gold_draws', 99), ('length_cap', 99)]:
            corrupt = copy.deepcopy(first)
            corrupt['sampling_stats'][field] = bad
            with self.subTest(field=field), self.assertRaises(AssertionError):
                aggregate_sampling_stats([corrupt])

    def test_missing_stats_allowed_only_without_declared_sampling_limits(self):
        legacy = dict(labels=[[9, 4, 2], [9, 5, 2]], costs=[0., .2])
        assert_pool_signal([legacy])
        for limit in ('max_no_eos_fraction', 'max_length_cap_fraction'):
            with self.subTest(limit=limit), self.assertRaisesRegex(AssertionError, 'diagnostics required'):
                assert_pool_signal([legacy], **{limit: .05})

    def test_declared_five_percent_limits_are_strict(self):
        good = self.record([[2, 9, 5, 2]] * 20)
        assert_pool_signal([good], max_no_eos_fraction=.05, max_length_cap_fraction=.05)
        for field, limit in [('no_eos', 'max_no_eos_fraction'), ('length_cap', 'max_length_cap_fraction')]:
            boundary = copy.deepcopy(good)
            boundary['sampling_stats'][field] = 1
            if field == 'no_eos':
                boundary['sampling_stats']['non_gold_duplicate_draws'] -= 1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                assert_pool_signal([boundary], **{limit: .05})
            boundary['sampling_stats']['sample_draws'] += 1
            boundary['sampling_stats']['gold_duplicate_draws'] += 1
            assert_pool_signal([boundary], **{limit: .05})


if __name__ == '__main__':
    unittest.main(verbosity=2)
