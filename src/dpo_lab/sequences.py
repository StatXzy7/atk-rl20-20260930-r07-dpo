from __future__ import annotations

from dataclasses import dataclass

import torch


PAD_ID = 0


@dataclass(frozen=True)
class PreferenceBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_completion_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_completion_mask: torch.Tensor
    prompt_lengths: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "chosen_input_ids",
            "chosen_attention_mask",
            "chosen_completion_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
            "rejected_completion_mask",
        ):
            if getattr(self, name).ndim != 2:
                raise ValueError(f"{name} must have shape [batch, seq]")
        pairs = (
            (self.chosen_input_ids, self.chosen_attention_mask, self.chosen_completion_mask),
            (
                self.rejected_input_ids,
                self.rejected_attention_mask,
                self.rejected_completion_mask,
            ),
        )
        for input_ids, attention_mask, completion_mask in pairs:
            if input_ids.shape != attention_mask.shape:
                raise ValueError("input and attention masks must have the same shape")
            if input_ids.shape != completion_mask.shape:
                raise ValueError("input and completion masks must have the same shape")

    @property
    def batch_size(self) -> int:
        return int(self.chosen_input_ids.shape[0])

    @property
    def device(self) -> torch.device:
        return self.chosen_input_ids.device

    def select(self, index: int) -> "PreferenceBatch":
        tensors = {
            field: getattr(self, field)[index : index + 1]
            for field in (
                "chosen_input_ids",
                "chosen_attention_mask",
                "chosen_completion_mask",
                "rejected_input_ids",
                "rejected_attention_mask",
                "rejected_completion_mask",
                "prompt_lengths",
            )
        }
        return PreferenceBatch(**tensors)

    def to(self, device: torch.device | str) -> "PreferenceBatch":
        return PreferenceBatch(
            chosen_input_ids=self.chosen_input_ids.to(device),
            chosen_attention_mask=self.chosen_attention_mask.to(device),
            chosen_completion_mask=self.chosen_completion_mask.to(device),
            rejected_input_ids=self.rejected_input_ids.to(device),
            rejected_attention_mask=self.rejected_attention_mask.to(device),
            rejected_completion_mask=self.rejected_completion_mask.to(device),
            prompt_lengths=self.prompt_lengths.to(device),
        )


def _encode_side(
    prompts: list[list[int]],
    completions: list[list[int]],
    eos_token_id: int,
    pad_token_id: int = PAD_ID,
    include_eos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not include_eos:
        raise ValueError("include_eos must be True so both sides follow the same EOS rule")
    if eos_token_id == pad_token_id:
        raise ValueError("EOS and pad tokens must differ")

    encoded: list[list[int]] = []
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    for prompt, completion in zip(prompts, completions):
        if not prompt:
            raise ValueError("each prompt must contain at least one token")
        if pad_token_id in prompt or eos_token_id in prompt:
            raise ValueError("prompt tokens must not contain pad or EOS")
        if pad_token_id in completion or eos_token_id in completion:
            raise ValueError("completions must not contain pad or an explicit EOS token")
        prompt_lengths.append(len(prompt))
        completion_with_eos = [*completion, eos_token_id]
        completion_lengths.append(len(completion_with_eos))
        encoded.append([*prompt, *completion_with_eos])

    max_length = max(len(tokens) for tokens in encoded)
    input_ids = torch.full(
        (len(encoded), max_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(encoded), max_length), dtype=torch.long)
    completion_mask = torch.zeros((len(encoded), max_length), dtype=torch.long)

    for row, (tokens, prompt_length, completion_length) in enumerate(
        zip(encoded, prompt_lengths, completion_lengths)
    ):
        end = len(tokens)
        input_ids[row, :end] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, :end] = 1
        completion_mask[row, prompt_length:end] = 1

    return (
        input_ids,
        attention_mask,
        completion_mask,
        torch.tensor(prompt_lengths, dtype=torch.long),
    )


def _validate_pair_prompts(
    chosen_prompts: list[list[int]], rejected_prompts: list[list[int]]
) -> None:
    if len(chosen_prompts) != len(rejected_prompts):
        raise ValueError("chosen and rejected prompt batches must have equal size")
    for chosen_prompt, rejected_prompt in zip(chosen_prompts, rejected_prompts):
        if chosen_prompt != rejected_prompt:
            raise ValueError("each chosen/rejected pair must use exactly the same prompt")


def _pad_side_to_match(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    target_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_length = int(input_ids.shape[1])
    if current_length == target_length:
        return input_ids, attention_mask, completion_mask
    padding = target_length - current_length
    input_ids = _pad_last(input_ids, padding, pad_token_id)
    attention_mask = _pad_last(attention_mask, padding, 0)
    completion_mask = _pad_last(completion_mask, padding, 0)
    return input_ids, attention_mask, completion_mask


def _pad_last(tensor: torch.Tensor, padding: int, value: int) -> torch.Tensor:
    return torch.nn.functional.pad(tensor, (0, padding), value=value)


def build_preference_batch(
    chosen_prompts: list[list[int]],
    chosen_completions: list[list[int]],
    rejected_prompts: list[list[int]],
    rejected_completions: list[list[int]],
    eos_token_id: int,
    pad_token_id: int = PAD_ID,
    include_eos: bool = True,
) -> PreferenceBatch:
    """Build right-padded chosen/rejected tensors.

    An empty completion is valid and therefore has exactly one scored token: EOS.
    Prompt tokens are represented in the context but are zero in completion masks.
    """

    _validate_pair_prompts(chosen_prompts, rejected_prompts)
    chosen = _encode_side(
        chosen_prompts, chosen_completions, eos_token_id, pad_token_id, include_eos
    )
    rejected = _encode_side(
        rejected_prompts, rejected_completions, eos_token_id, pad_token_id, include_eos
    )

    max_length = max(int(chosen[0].shape[1]), int(rejected[0].shape[1]))
    chosen_ids, chosen_attention, chosen_completion_mask, chosen_prompt_lengths = chosen
    (
        rejected_ids,
        rejected_attention,
        rejected_completion_mask,
        rejected_prompt_lengths,
    ) = rejected
    chosen_ids, chosen_attention, chosen_completion_mask = _pad_side_to_match(
        chosen_ids, chosen_attention, chosen_completion_mask, max_length, pad_token_id
    )
    rejected_ids, rejected_attention, rejected_completion_mask = _pad_side_to_match(
        rejected_ids,
        rejected_attention,
        rejected_completion_mask,
        max_length,
        pad_token_id,
    )
    if not torch.equal(chosen_prompt_lengths, rejected_prompt_lengths):
        raise AssertionError("validated prompt lengths unexpectedly differ")

    return PreferenceBatch(
        chosen_input_ids=chosen_ids,
        chosen_attention_mask=chosen_attention,
        chosen_completion_mask=chosen_completion_mask,
        rejected_input_ids=rejected_ids,
        rejected_attention_mask=rejected_attention,
        rejected_completion_mask=rejected_completion_mask,
        prompt_lengths=chosen_prompt_lengths,
    )
