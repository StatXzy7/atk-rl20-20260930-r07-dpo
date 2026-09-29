from __future__ import annotations

import torch

from dpo_lab.scoring import dpo_loss


def tensor(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float64)


def analytic_chosen_grad(x: float, beta: float) -> float:
    return -beta * torch.sigmoid(torch.tensor(-beta * x, dtype=torch.float64)).item()


def central_gradient(x: float, beta: float, epsilon: float = 1e-6) -> float:
    values = []
    for delta in (epsilon, -epsilon):
        output = dpo_loss(
            tensor(x + delta),
            tensor(0.0),
            tensor(0.0),
            tensor(0.0),
            beta=beta,
        )
        values.append(output.loss.item())
    return (values[0] - values[1]) / (2 * epsilon)


def scalar_loss(chosen: float, rejected: float, beta: float) -> float:
    return dpo_loss(
        tensor(chosen),
        tensor(rejected),
        tensor(0.0),
        tensor(0.0),
        beta=beta,
    ).loss.item()


def central_pair_gradient(
    chosen: float, rejected: float, beta: float, epsilon: float = 1e-6
) -> tuple[float, float]:
    chosen_gradient = (
        scalar_loss(chosen + epsilon, rejected, beta)
        - scalar_loss(chosen - epsilon, rejected, beta)
    ) / (2 * epsilon)
    rejected_gradient = (
        scalar_loss(chosen, rejected + epsilon, beta)
        - scalar_loss(chosen, rejected - epsilon, beta)
    ) / (2 * epsilon)
    return chosen_gradient, rejected_gradient


def test_log_sigmoid_is_finite_at_large_margins():
    output = dpo_loss(tensor(1000.0), tensor(0.0), tensor(0.0), tensor(0.0), beta=1.0)
    assert torch.isfinite(output.loss)
    assert output.loss.item() < 1e-300
    output = dpo_loss(tensor(-1000.0), tensor(0.0), tensor(0.0), tensor(0.0), beta=1.0)
    assert torch.isfinite(output.loss)
    assert output.loss.item() > 900.0


def test_reward_margin_sign_follows_chosen_minus_rejected():
    output = dpo_loss(tensor(1.5), tensor(0.25), tensor(0.5), tensor(0.75), beta=2.0)
    expected_logit_margin = (1.5 - 0.5) - (0.25 - 0.75)
    assert torch.allclose(output.logit_margins, tensor(expected_logit_margin))
    assert torch.allclose(output.reward_margins, tensor(2.0 * expected_logit_margin))


def test_double_finite_difference_beta_changes_gradient_scale_near_zero():
    x = 0.0
    low_beta = central_gradient(x, beta=0.5)
    high_beta = central_gradient(x, beta=2.0)
    assert abs(low_beta - analytic_chosen_grad(x, 0.5)) < 1e-7
    assert abs(high_beta - analytic_chosen_grad(x, 2.0)) < 1e-7
    assert abs(high_beta / low_beta - 4.0) < 1e-4


def test_double_finite_difference_swap_reverses_direction():
    original_chosen_value = 0.4
    original_rejected_value = -0.1
    original_chosen = torch.tensor([original_chosen_value], dtype=torch.float64)
    original_rejected = torch.tensor([original_rejected_value], dtype=torch.float64)
    reference_chosen = torch.tensor([0.0], dtype=torch.float64)
    reference_rejected = torch.tensor([0.0], dtype=torch.float64)

    original = dpo_loss(
        original_chosen,
        original_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.7,
    )
    original_grad_chosen, original_grad_rejected = central_pair_gradient(
        original_chosen_value, original_rejected_value, beta=0.7
    )
    assert original.reward_margins.item() > 0.0
    assert original_grad_chosen < 0.0
    assert original_grad_rejected > 0.0

    swapped_chosen = original_rejected.detach().clone()
    swapped_rejected = original_chosen.detach().clone()
    swapped = dpo_loss(
        swapped_chosen,
        swapped_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.7,
    )
    swapped_grad_chosen, swapped_grad_rejected = central_pair_gradient(
        original_rejected_value, original_chosen_value, beta=0.7
    )
    assert swapped.reward_margins.item() == -original.reward_margins.item()
    assert swapped_grad_chosen < 0.0
    assert swapped_grad_rejected > 0.0
