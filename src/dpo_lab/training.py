from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache import ReferenceScoreCache
from .scoring import DPOLossOutput, dpo_loss, sequence_logprobs
from .sequences import PreferenceBatch


@dataclass(frozen=True)
class DPOStepOutput:
    loss: torch.Tensor
    metrics: DPOLossOutput


def _assert_distinct_models(
    policy: torch.nn.Module, reference: torch.nn.Module
) -> None:
    policy_params = {param.data_ptr() for param in policy.parameters()}
    for param in reference.parameters():
        if param.data_ptr() in policy_params:
            raise ValueError("reference model must not share trainable policy parameters")


def dpo_step(
    policy: torch.nn.Module,
    reference: torch.nn.Module | ReferenceScoreCache,
    batch: PreferenceBatch,
    beta: float,
    optimizer: torch.optim.Optimizer | None = None,
    reduction: str = "mean",
) -> DPOStepOutput:
    if not isinstance(reference, ReferenceScoreCache):
        reference_cache = ReferenceScoreCache(reference)
    else:
        reference_cache = reference

    _assert_distinct_models(policy, reference_cache.reference_model)
    policy.train()
    reference_cache.reference_model.eval()

    policy_chosen = sequence_logprobs(
        policy,
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_completion_mask,
    )
    policy_rejected = sequence_logprobs(
        policy,
        batch.rejected_input_ids,
        batch.rejected_attention_mask,
        batch.rejected_completion_mask,
    )
    reference_chosen, reference_rejected = reference_cache.score_preference_batch(batch)

    metrics = dpo_loss(
        policy_chosen.sequence_logprobs,
        policy_rejected.sequence_logprobs,
        reference_chosen.sequence_logprobs,
        reference_rejected.sequence_logprobs,
        beta=beta,
        reduction=reduction,
    )

    policy.zero_grad(set_to_none=True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    metrics.loss.backward()
    if optimizer is not None:
        optimizer.step()

    for param in reference_cache.reference_model.parameters():
        if param.grad is not None:
            raise AssertionError("reference parameters received a gradient")

    return DPOStepOutput(loss=metrics.loss, metrics=metrics)
