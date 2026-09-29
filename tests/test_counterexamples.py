from __future__ import annotations

import torch

from dpo_lab.counterexamples import (
    prompt_leak_chosen_completion_mask,
    token_averaged_dpo_loss,
)
from dpo_lab.scoring import SequenceScoreResult, dpo_loss, sequence_logprobs
from dpo_lab.sequences import build_preference_batch
from conftest import FixedLogitsModel, make_tiny_model


def test_prompt_mask_error_makes_prompt_target_receive_preference_gradient():
    batch = build_preference_batch(
        chosen_prompts=[[1, 2]],
        chosen_completions=[[3]],
        rejected_prompts=[[1, 2]],
        rejected_completions=[[4]],
        eos_token_id=11,
    )
    correct_chosen_model = FixedLogitsModel(
        torch.randn(1, 4, 12, dtype=torch.float64)
    )
    wrong_chosen_model = FixedLogitsModel(
        correct_chosen_model.logits.detach().clone()
    )
    rejected_model = FixedLogitsModel(torch.randn(1, 4, 12, dtype=torch.float64))

    correct_chosen = sequence_logprobs(
        correct_chosen_model,
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_completion_mask,
    )
    wrong_chosen = sequence_logprobs(
        wrong_chosen_model,
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        prompt_leak_chosen_completion_mask(batch),
    )
    rejected = sequence_logprobs(
        rejected_model,
        batch.rejected_input_ids,
        batch.rejected_attention_mask,
        batch.rejected_completion_mask,
    )

    dpo_loss(
        correct_chosen.sequence_logprobs,
        rejected.sequence_logprobs,
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        beta=0.5,
    ).loss.backward(retain_graph=True)
    dpo_loss(
        wrong_chosen.sequence_logprobs,
        rejected.sequence_logprobs,
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        beta=0.5,
    ).loss.backward()

    correct_prompt_target_grad = correct_chosen_model.logits.grad[0, 0, 2]
    wrong_prompt_target_grad = wrong_chosen_model.logits.grad[0, 0, 2]
    assert correct_prompt_target_grad.item() == 0.0
    assert abs(wrong_prompt_target_grad.item()) > 1e-6


def test_token_average_changes_gradient_weighting_not_just_global_scale():
    chosen_sum = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
    rejected_sum = torch.tensor([-8.0], dtype=torch.float64, requires_grad=True)
    reference_chosen_sum = torch.zeros(1, dtype=torch.float64)
    reference_rejected_sum = torch.zeros(1, dtype=torch.float64)

    sum_output = dpo_loss(
        chosen_sum,
        rejected_sum,
        reference_chosen_sum,
        reference_rejected_sum,
        beta=0.5,
    )
    sum_output.loss.backward()
    assert sum_output.logit_margins.item() == 6.0

    chosen_average = chosen_sum.detach().clone().requires_grad_()
    rejected_average = rejected_sum.detach().clone().requires_grad_()
    chosen_result = SequenceScoreResult(
        sequence_logprobs=chosen_average,
        token_logprobs=torch.empty(1, 2, dtype=torch.float64),
        completion_lengths=torch.tensor([2]),
    )
    rejected_result = SequenceScoreResult(
        sequence_logprobs=rejected_average,
        token_logprobs=torch.empty(1, 4, dtype=torch.float64),
        completion_lengths=torch.tensor([4]),
    )
    reference_chosen_result = SequenceScoreResult(
        sequence_logprobs=reference_chosen_sum,
        token_logprobs=torch.empty(1, 2, dtype=torch.float64),
        completion_lengths=torch.tensor([2]),
    )
    reference_rejected_result = SequenceScoreResult(
        sequence_logprobs=reference_rejected_sum,
        token_logprobs=torch.empty(1, 4, dtype=torch.float64),
        completion_lengths=torch.tensor([4]),
    )
    average_output = token_averaged_dpo_loss(
        chosen_result,
        rejected_result,
        reference_chosen_result,
        reference_rejected_result,
        beta=0.5,
    )
    average_output.loss.backward()

    assert average_output.logit_margins.item() == 1.0
    chosen_scale = abs(chosen_average.grad.item() / chosen_sum.grad.item())
    rejected_scale = abs(rejected_average.grad.item() / rejected_sum.grad.item())
    assert chosen_scale > 3.5
    assert 1.5 < rejected_scale < 2.5
    assert abs(chosen_scale - rejected_scale) > 1.0


def test_real_model_scores_eos_once_for_each_nonempty_or_empty_completion():
    model = make_tiny_model(vocab_size=12, seed=51)
    batch = build_preference_batch(
        chosen_prompts=[[1, 2], [3]],
        chosen_completions=[[], [4, 5]],
        rejected_prompts=[[1, 2], [3]],
        rejected_completions=[[6], []],
        eos_token_id=11,
    )
    chosen = sequence_logprobs(
        model,
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_completion_mask,
    )
    rejected = sequence_logprobs(
        model,
        batch.rejected_input_ids,
        batch.rejected_attention_mask,
        batch.rejected_completion_mask,
    )
    assert chosen.completion_lengths.tolist() == [1, 3]
    assert rejected.completion_lengths.tolist() == [2, 1]
