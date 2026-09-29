from __future__ import annotations

import math

import pytest
import torch

from dpo_lab.sequences import build_preference_batch
from dpo_lab.scoring import sequence_logprobs
from conftest import FixedLogitsModel, make_tiny_model


def test_label_shift_scores_next_token_not_current_token():
    input_ids = torch.tensor([[1, 2, 3, 4, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    completion_mask = torch.tensor([[0, 0, 1, 1, 0]])
    logits = torch.full((1, 5, 6), -20.0, dtype=torch.float64)
    for position in range(4):
        logits[0, position, input_ids[0, position + 1]] = 20.0
    model = FixedLogitsModel(logits)

    result = sequence_logprobs(model, input_ids, attention_mask, completion_mask)

    assert torch.allclose(result.completion_lengths, torch.tensor([2]))
    assert result.token_logprobs.shape == (1, 4)
    assert torch.allclose(result.token_logprobs[0, :1], torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(
        result.token_logprobs[0, 2:4], torch.zeros(2, dtype=torch.float64), atol=1e-10
    )
    assert result.token_logprobs[0, 3].item() == 0.0
    assert torch.allclose(result.sequence_logprobs, torch.zeros(1, dtype=torch.float64), atol=1e-10)


def test_objective_is_completion_logprob_sum_not_average():
    input_ids = torch.tensor([[1, 1, 1, 1, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    completion_mask = torch.tensor([[0, 1, 1, 1, 0]])
    logits = torch.zeros((1, 5, 2), dtype=torch.float64)
    logits[..., 1] = math.log(3.0)
    model = FixedLogitsModel(logits)

    result = sequence_logprobs(model, input_ids, attention_mask, completion_mask)

    expected = 3.0 * math.log(0.75)
    assert result.completion_lengths.item() == 3
    assert torch.allclose(
        result.sequence_logprobs, torch.tensor([expected], dtype=torch.float64)
    )


def test_prompt_labels_are_masked_and_padding_has_no_scored_target():
    input_ids = torch.tensor([[1, 2, 3, 7, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    completion_mask = torch.tensor([[0, 0, 1, 1, 0]])
    logits = torch.randn(1, 5, 8, dtype=torch.float64)
    model = FixedLogitsModel(logits)

    result = sequence_logprobs(model, input_ids, attention_mask, completion_mask)

    assert torch.count_nonzero(result.token_logprobs).item() == 2
    assert result.token_logprobs.shape == (1, 4)
    assert result.token_logprobs[0, 0].item() == 0.0
    assert result.token_logprobs[0, 1].item() != 0.0
    assert result.token_logprobs[0, 3].item() == 0.0


def test_empty_completion_scores_only_eos():
    batch = build_preference_batch(
        chosen_prompts=[[1, 2]],
        chosen_completions=[[]],
        rejected_prompts=[[1, 2]],
        rejected_completions=[[3]],
        eos_token_id=7,
    )

    assert batch.chosen_input_ids.tolist() == [[1, 2, 7, 0]]
    assert batch.chosen_completion_mask.tolist() == [[0, 0, 1, 0]]
    model = make_tiny_model(vocab_size=8)
    result = sequence_logprobs(
        model,
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_completion_mask,
    )

    assert result.completion_lengths.item() == 1
    assert torch.isfinite(result.sequence_logprobs).all()


def test_eos_rule_is_explicit_shared_and_required():
    batch = build_preference_batch(
        [[1]],
        [[2]],
        [[1]],
        [[3]],
        eos_token_id=7,
    )
    assert batch.chosen_input_ids[0, -1].item() == 7
    assert batch.rejected_input_ids[0, -1].item() == 7
    with pytest.raises(ValueError, match="EOS"):
        build_preference_batch([[1]], [[2]], [[1]], [[3]], eos_token_id=7, include_eos=False)


def test_different_prompt_lengths_in_pair_are_rejected():
    with pytest.raises(ValueError, match="same prompt"):
        build_preference_batch(
            chosen_prompts=[[1, 2]],
            chosen_completions=[[3]],
            rejected_prompts=[[1]],
            rejected_completions=[[4]],
            eos_token_id=7,
        )


def test_batch_allows_different_prompt_lengths_between_examples():
    batch = build_preference_batch(
        chosen_prompts=[[1, 2], [1]],
        chosen_completions=[[3], [4, 5]],
        rejected_prompts=[[1, 2], [1]],
        rejected_completions=[[6], [7]],
        eos_token_id=8,
    )
    assert batch.prompt_lengths.tolist() == [2, 1]
    assert batch.chosen_attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]


def test_causal_model_does_not_look_at_future_tokens():
    model = make_tiny_model(vocab_size=8).eval()
    left = torch.tensor([[1, 2, 3, 4]])
    right = torch.tensor([[1, 2, 3, 5]])
    mask = torch.ones_like(left)
    with torch.no_grad():
        left_logits = model(left, mask)
        right_logits = model(right, mask)
    assert torch.allclose(left_logits[:, :3], right_logits[:, :3], atol=1e-12)
    assert not torch.allclose(left_logits[:, 3], right_logits[:, 3])


def test_padding_does_not_change_other_rows():
    model = make_tiny_model(vocab_size=10).eval()
    single = torch.tensor([[1, 2, 3]])
    single_mask = torch.ones_like(single)
    padded = torch.tensor([[1, 2, 3], [4, 5, 0]])
    padded_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with torch.no_grad():
        single_logits = model(single, single_mask)
        padded_logits = model(padded, padded_mask)
    assert torch.allclose(single_logits[0], padded_logits[0], atol=1e-12)
