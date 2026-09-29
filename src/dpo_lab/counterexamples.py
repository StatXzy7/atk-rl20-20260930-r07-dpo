from __future__ import annotations

import torch

from .scoring import DPOLossOutput, SequenceScoreResult, dpo_loss
from .sequences import PreferenceBatch


def prompt_leak_chosen_completion_mask(batch: PreferenceBatch) -> torch.Tensor:
    """Deliberately wrong: score prompt targets on the chosen side only.

    The correct shifted completion mask starts at the first completion label. This
    helper instead marks every non-first chosen label, including prompt labels. It
    exists only to make the failure mode executable.
    """

    wrong_mask = batch.chosen_completion_mask.clone()
    wrong_mask[:, 1:] = batch.chosen_attention_mask[:, 1:]
    return wrong_mask


def token_averaged_dpo_loss(
    chosen: SequenceScoreResult,
    rejected: SequenceScoreResult,
    reference_chosen: SequenceScoreResult,
    reference_rejected: SequenceScoreResult,
    beta: float,
    reduction: str = "mean",
) -> DPOLossOutput:
    """A different objective: per-token mean log-probability instead of sum."""

    return dpo_loss(
        chosen.sequence_logprobs / chosen.completion_lengths,
        rejected.sequence_logprobs / rejected.completion_lengths,
        reference_chosen.sequence_logprobs / reference_chosen.completion_lengths,
        reference_rejected.sequence_logprobs / reference_rejected.completion_lengths,
        beta=beta,
        reduction=reduction,
    )
