"""Algebra and failure-case checks for an unregistered research prototype."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from methods.source_attribution.v1.information import information_components


def test_chain_rule_and_nonnegativity():
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(23, 2, 2, 17, generator=generator, dtype=torch.float64)
    result = information_components(values)
    torch.testing.assert_close(result['joint_dropout'],
        result['marginal_dropout'] + result['source_attribution'], atol=2e-15, rtol=2e-14)
    for name in ('joint_dropout', 'marginal_dropout', 'source_attribution', 'source_information'):
        assert result[name].min() >= -2e-15
        assert result[name].max() <= math.log(2.) + 2e-15
    assert result['attribution_by_token_type'].min() >= -2e-15


def test_source_blind_collapse_is_explicit_counterexample():
    # Different dropout predictions, identical two-source responses: C=0.
    values = torch.tensor([[[3., -2.], [3., -2.]],
                           [[-2., 3.], [-2., 3.]]], dtype=torch.float64)
    result = information_components(values)
    assert abs(float(result['source_attribution'])) < 1e-15
    assert abs(float(result['source_information'])) < 1e-15
    assert float(result['marginal_dropout']) > .6


def test_consistent_source_information_is_not_forced_to_zero():
    # Identical views distinguish sources sharply: C=0, I(S;V)>0.
    values = torch.tensor([[[3., -3.], [-3., 3.]],
                           [[3., -3.], [-3., 3.]]], dtype=torch.float64)
    result = information_components(values)
    assert abs(float(result['source_attribution'])) < 1e-15
    assert float(result['source_information']) > .65


def test_source_view_interaction_can_be_maximal_with_zero_marginal_dropout():
    values = torch.tensor([[[40., -40.], [-40., 40.]],
                           [[-40., 40.], [40., -40.]]], dtype=torch.float64)
    result = information_components(values)
    assert abs(float(result['marginal_dropout'])) < 1e-15
    assert abs(float(result['source_attribution']) - math.log(2.)) < 1e-14


def test_gradients_match_numerical_derivatives_and_chain_rule():
    generator = torch.Generator().manual_seed(43)
    values = torch.randn(2, 2, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: information_components(x)['source_attribution'], (values,))
    result = information_components(values)
    left = torch.autograd.grad(result['source_attribution'], values, retain_graph=True)[0]
    right = torch.autograd.grad(result['joint_dropout'] - result['marginal_dropout'], values)[0]
    torch.testing.assert_close(left, right, atol=1e-15, rtol=1e-12)


def test_permutations_and_logit_offsets():
    generator = torch.Generator().manual_seed(44)
    values = torch.randn(3, 2, 2, 7, generator=generator, dtype=torch.float64)
    expected = information_components(values)
    for transformed in (values.flip(-3), values.flip(-2), values.flip(-1),
                        values + torch.randn(3, 2, 2, 1, generator=generator, dtype=torch.float64)):
        actual = information_components(transformed)
        for name in ('source_attribution', 'marginal_dropout', 'joint_dropout', 'source_information'):
            torch.testing.assert_close(actual[name], expected[name], atol=2e-15, rtol=2e-13)


def test_finite_extreme_logits_and_float32():
    values = torch.tensor([[[1000., -1000., 0.], [0., 1000., -1000.]],
                           [[-1000., 0., 1000.], [1000., 0., -1000.]]], requires_grad=True)
    result = information_components(values)
    assert all(torch.isfinite(value).all() for value in result.values())
    result['source_attribution'].backward()
    assert torch.isfinite(values.grad).all()
    assert abs(float(result['joint_dropout'] - result['marginal_dropout']
                     - result['source_attribution'])) < 5e-7


def test_coupled_masks_do_not_advance_the_original_view_stream():
    # CPU toy checks the proposed coupling contract, NOT native GPU SDPA replay.
    torch.manual_seed(20261115)
    model = torch.nn.Sequential(torch.nn.Linear(3, 9), torch.nn.Dropout(.5),
                                torch.nn.Linear(9, 5)).train()
    inputs = torch.randn(4, 3)
    initial = torch.get_rng_state()
    reference = torch.stack([model(inputs), model(inputs)], dim=-2)
    reference_rng = torch.get_rng_state()
    reference_gradient = torch.autograd.grad(reference.square().mean(), tuple(model.parameters()))
    torch.set_rng_state(initial)
    rows = []
    for _ in range(2):
        before = torch.get_rng_state()
        original = model(inputs)
        with torch.random.fork_rng():
            torch.set_rng_state(before)
            duplicate = model(inputs)
        torch.testing.assert_close(original, duplicate, atol=0, rtol=0)
        rows.append(torch.stack((original, duplicate), dim=-2))
    table = torch.stack(rows, dim=-3)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    torch.testing.assert_close(table[..., 0, :], reference, atol=0, rtol=0)
    c = information_components(table)['source_attribution'].mean()
    assert abs(float(c)) < 2e-7
    candidate_loss = table[..., 0, :].square().mean() + 0. * c
    candidate_gradient = torch.autograd.grad(candidate_loss, tuple(model.parameters()))
    for actual, expected in zip(candidate_gradient, reference_gradient):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def test_invalid_tables_fail():
    for values in (torch.zeros(2, 3, 4), torch.ones(2, 2, 4, dtype=torch.long),
                   torch.full((2, 2, 4), float('inf'))):
        try:
            information_components(values)
        except ValueError:
            continue
        raise AssertionError('Invalid table accepted')


if __name__ == '__main__':
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in sorted(globals().copy().items()) if name.startswith('test_'))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if outcome.wasSuccessful() else 1)
