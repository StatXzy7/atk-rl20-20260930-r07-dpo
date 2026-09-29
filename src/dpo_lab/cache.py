from __future__ import annotations

import hashlib

import torch

from .scoring import SequenceScoreResult, sequence_logprobs
from .sequences import PreferenceBatch


def _tensor_fingerprint(tensor: torch.Tensor) -> bytes:
    detached = tensor.detach().cpu().contiguous()
    payload = (
        str(tuple(detached.shape)).encode()
        + b"|"
        + str(detached.dtype).encode()
        + b"|"
        + detached.numpy().tobytes()
    )
    return hashlib.sha256(payload).digest()


def parameter_fingerprint(model: torch.nn.Module) -> bytes:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict(keep_vars=True).items():
        digest.update(name.encode())
        digest.update(_tensor_fingerprint(tensor))
    return digest.digest()


def input_fingerprint(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> bytes:
    digest = hashlib.sha256()
    for tensor in (input_ids, attention_mask, completion_mask):
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.device).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(_tensor_fingerprint(tensor))
    return digest.digest()


class ReferenceScoreCache:
    """Reference sequence scores keyed by inputs, masks, and reference weights."""

    def __init__(self, reference_model: torch.nn.Module) -> None:
        self.reference_model = reference_model
        self.reference_model.requires_grad_(False)
        self._entries: dict[bytes, SequenceScoreResult] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def get_or_compute(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> SequenceScoreResult:
        was_training = self.reference_model.training
        self.reference_model.eval()
        key = (
            parameter_fingerprint(self.reference_model)
            + input_fingerprint(input_ids, attention_mask, completion_mask)
            + str(was_training).encode()
        )
        if key not in self._entries:
            with torch.no_grad():
                result = sequence_logprobs(
                    self.reference_model,
                    input_ids,
                    attention_mask,
                    completion_mask,
                )
            self._entries[key] = SequenceScoreResult(
                sequence_logprobs=result.sequence_logprobs.detach(),
                token_logprobs=result.token_logprobs.detach(),
                completion_lengths=result.completion_lengths.detach(),
            )
        if was_training:
            self.reference_model.train()
        return self._entries[key]

    def score_preference_batch(
        self, batch: PreferenceBatch
    ) -> tuple[SequenceScoreResult, SequenceScoreResult]:
        chosen = self.get_or_compute(
            batch.chosen_input_ids,
            batch.chosen_attention_mask,
            batch.chosen_completion_mask,
        )
        rejected = self.get_or_compute(
            batch.rejected_input_ids,
            batch.rejected_attention_mask,
            batch.rejected_completion_mask,
        )
        return chosen, rejected
