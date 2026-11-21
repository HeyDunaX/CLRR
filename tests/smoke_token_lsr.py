"""Token formula, detached targets, checkpoint recomputation and loss scaling."""
import torch
from methods.token_lsr.v1.lsr import token_alignment_loss, transport_plan


def check_formula():
    torch.manual_seed(42)
    source = torch.randn(2, 4, 8, requires_grad=True)
    target = torch.randn(2, 5, 8, requires_grad=True)
    sm = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    tm = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    cost = 1 - torch.nn.functional.normalize(source, dim=-1) @ torch.nn.functional.normalize(target, dim=-1).transpose(1, 2)
    plan, error = transport_plan(cost, sm, tm)
    assert not plan.requires_grad and error.max() < 1e-3
    torch.testing.assert_close(token_alignment_loss(source, target, sm, tm), (plan * cost).sum((1, 2)).mean())
    actual = token_alignment_loss(source, target, sm, tm)
    actual.backward()
    assert target.grad is None and source.grad.norm() > 0 and torch.isfinite(source.grad).all()
    assert torch.count_nonzero(source.grad[~sm]) == 0
    modified_s, modified_t = source.detach().clone(), target.detach().clone()
    modified_s[~sm] = 999; modified_t[~tm] = -999
    torch.testing.assert_close(actual.detach(), token_alignment_loss(modified_s, modified_t, sm, tm))
    try:
        token_alignment_loss(source, target, torch.zeros_like(sm), tm)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty content must fail explicitly')
    return {'formula': True, 'source_gradient_finite': True, 'target_and_plan_detached': True,
            'padding_invariant': True, 'empty_content_rejected': True, 'marginal_error': float(error.max())}


def check_wrapper(worker, base_model, spec, batch):
    model, _ = worker.wrap_model(base_model(spec['backbone']), spec)
    model.train()
    outputs = model(**batch)
    outputs.loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    checkpointed = model(**batch)
    torch.testing.assert_close(checkpointed.loss, outputs.loss, rtol=2e-4, atol=2e-6)
    checkpointed.loss.backward()
    for n, p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(p.grad, gradients[n], rtol=2e-4, atol=2e-6)
    assert model.content_mask(torch.tensor([[0, 2, 8]]), torch.ones(1, 3)).tolist() == [[False, False, True]]
    return {'checkpoint_gradient_match': True, 'special_tokens_excluded': True}
