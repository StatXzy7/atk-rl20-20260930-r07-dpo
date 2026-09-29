from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SequenceScoreResult:
    sequence_logprobs: torch.Tensor
    token_logprobs: torch.Tensor
    completion_lengths: torch.Tensor


@dataclass(frozen=True)
class DPOLossOutput:
    loss: torch.Tensor
    per_example_loss: torch.Tensor
    logit_margins: torch.Tensor
    reward_margins: torch.Tensor
    chosen_sequence_logprobs: torch.Tensor
    rejected_sequence_logprobs: torch.Tensor


def sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> SequenceScoreResult:
    """Score completion targets with the autoregressive one-position label shift.

    Logits at position t predict token t+1. The returned objective is the sum of
    completion token log-probabilities; prompt and right-padding positions have
    zero weight. EOS is included by the batch builder and scored here.
    """

    if input_ids.shape != attention_mask.shape or input_ids.shape != completion_mask.shape:
        raise ValueError("input_ids, attention_mask, and completion_mask must match")

    logits = model(input_ids, attention_mask)
    prediction_logits = logits[:, :-1]
    target_ids = input_ids[:, 1:]
    target_mask = completion_mask[:, 1:].to(torch.bool)
    valid_targets = attention_mask[:, 1:].to(torch.bool)
    if bool((target_mask & ~valid_targets).any()):
        raise ValueError("completion targets must be valid, non-padded positions")

    log_probs = F.log_softmax(prediction_logits, dim=-1)
    token_logprobs = torch.gather(
        log_probs, dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    token_logprobs = token_logprobs * target_mask.to(token_logprobs.dtype)
    completion_lengths = target_mask.sum(dim=1)
    if bool((completion_lengths == 0).any()):
        raise ValueError("each example must score at least the completion EOS token")

    return SequenceScoreResult(
        sequence_logprobs=token_logprobs.sum(dim=1),
        token_logprobs=token_logprobs,
        completion_lengths=completion_lengths,
    )


def dpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    reduction: str = "mean",
) -> DPOLossOutput:
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError("chosen and rejected log-probabilities must have the same shape")
    if reference_chosen_logps.shape != chosen_logps.shape:
        raise ValueError("reference chosen scores must match policy chosen scores")
    if reference_rejected_logps.shape != rejected_logps.shape:
        raise ValueError("reference rejected scores must match policy rejected scores")

    chosen_margin = chosen_logps - reference_chosen_logps
    rejected_margin = rejected_logps - reference_rejected_logps
    logit_margins = chosen_margin - rejected_margin
    reward_margins = beta * logit_margins
    per_example_loss = -F.logsigmoid(reward_margins)

    if reduction == "none":
        loss = per_example_loss
    elif reduction == "sum":
        loss = per_example_loss.sum()
    elif reduction == "mean":
        loss = per_example_loss.mean()
    else:
        raise ValueError("reduction must be one of: none, sum, mean")

    return DPOLossOutput(
        loss=loss,
        per_example_loss=per_example_loss,
        logit_margins=logit_margins,
        reward_margins=reward_margins,
        chosen_sequence_logprobs=chosen_logps,
        rejected_sequence_logprobs=rejected_logps,
    )
