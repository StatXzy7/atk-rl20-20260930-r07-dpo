from __future__ import annotations

import pytest
import torch

from dpo_lab.cache import ReferenceScoreCache
from dpo_lab.sequences import build_preference_batch
from dpo_lab.training import dpo_step
from conftest import FixedLogitsModel, clone_grads, clone_model, make_tiny_model


def make_batch() -> object:
    return build_preference_batch(
        chosen_prompts=[[1, 2], [1]],
        chosen_completions=[[3, 4], [5]],
        rejected_prompts=[[1, 2], [1]],
        rejected_completions=[[6], [7, 8]],
        eos_token_id=11,
    )


def test_dpo_step_performs_a_real_parameter_update_and_freezes_reference():
    policy = make_tiny_model(seed=1)
    reference = make_tiny_model(seed=2)
    batch = make_batch()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    before = {name: param.detach().clone() for name, param in policy.named_parameters()}

    output = dpo_step(policy, reference, batch, beta=0.5, optimizer=optimizer)

    assert torch.isfinite(output.loss)
    assert output.loss.requires_grad is True
    changed = []
    for name, param in policy.named_parameters():
        assert not torch.equal(param.detach(), before[name])
        changed.append(torch.linalg.vector_norm(param.detach() - before[name]).item())
    assert sum(changed) > 1e-8
    assert all(param.grad is None for param in reference.parameters())
    assert all(not param.requires_grad for param in reference.parameters())


def test_one_update_increases_policy_preference_margin_on_synthetic_pair():
    torch.manual_seed(7)
    policy = make_tiny_model(seed=7)
    reference = make_tiny_model(seed=17)
    batch = build_preference_batch(
        chosen_prompts=[[1, 2]],
        chosen_completions=[[3, 4]],
        rejected_prompts=[[1, 2]],
        rejected_completions=[[5, 6]],
        eos_token_id=11,
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    first = dpo_step(policy, reference, batch, beta=0.7, optimizer=optimizer)
    second = dpo_step(policy, reference, batch, beta=0.7)

    assert second.metrics.logit_margins.item() > first.metrics.logit_margins.item()
    assert second.loss.item() < first.loss.item()


def test_identical_chosen_and_rejected_has_finite_loss_but_zero_training_gradient():
    policy = make_tiny_model(seed=3)
    reference = make_tiny_model(seed=4)
    batch = build_preference_batch(
        chosen_prompts=[[1, 2]],
        chosen_completions=[[3, 4]],
        rejected_prompts=[[1, 2]],
        rejected_completions=[[3, 4]],
        eos_token_id=11,
    )
    output = dpo_step(policy, reference, batch, beta=0.9)

    assert output.metrics.logit_margins.item() == pytest.approx(0.0, abs=1e-14)
    assert output.loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    for param in policy.parameters():
        assert param.grad is None or torch.allclose(
            param.grad, torch.zeros_like(param.grad), atol=1e-14
        )


def test_per_example_loop_matches_batch_loss_and_all_trainable_gradients():
    batch_policy = make_tiny_model(seed=21)
    reference = make_tiny_model(seed=22).requires_grad_(False)
    batch_reference_cache = ReferenceScoreCache(reference)

    batch_output = dpo_step(
        batch_policy,
        batch_reference_cache,
        make_batch(),
        beta=0.6,
        reduction="mean",
    )

    loop_losses = []
    loop_gradients = []
    loop_reference_cache = ReferenceScoreCache(reference)
    batch_data = make_batch()
    for index in range(batch_data.batch_size):
        single_policy = clone_model(batch_policy)
        single = batch_data.select(index)
        loop_output = dpo_step(
            single_policy,
            loop_reference_cache,
            single,
            beta=0.6,
            reduction="sum",
        )
        loop_losses.append(loop_output.loss / batch_data.batch_size)
        loop_gradients.append(clone_grads(single_policy))
    loop_loss = torch.stack(loop_losses).sum()

    assert torch.allclose(batch_output.loss, loop_loss, atol=1e-12, rtol=1e-12)
    batch_grads = clone_grads(batch_policy)
    loop_grads = {
        name: torch.stack(
            [single_grads[name] / batch_data.batch_size for single_grads in loop_gradients]
        ).sum(dim=0)
        for name in batch_grads
    }
    assert set(batch_grads) == set(loop_grads)
    for name in batch_grads:
        assert torch.allclose(batch_grads[name], loop_grads[name], atol=1e-11, rtol=1e-11)


def test_reference_cache_binds_scores_to_tokens_masks_and_reference_fingerprint():
    model = make_tiny_model(vocab_size=12, seed=31)
    cache = ReferenceScoreCache(model)
    batch = build_preference_batch(
        chosen_prompts=[[1, 2]],
        chosen_completions=[[3]],
        rejected_prompts=[[1, 2]],
        rejected_completions=[[4]],
        eos_token_id=11,
    )

    chosen = cache.get_or_compute(
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_completion_mask,
    )
    rejected = cache.get_or_compute(
        batch.rejected_input_ids,
        batch.rejected_attention_mask,
        batch.rejected_completion_mask,
    )
    assert len(cache) == 2

    chosen_again = cache.get_or_compute(
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_completion_mask,
    )
    assert len(cache) == 2
    assert chosen_again.sequence_logprobs.data_ptr() == chosen.sequence_logprobs.data_ptr()

    changed_ids = batch.chosen_input_ids.clone()
    changed_ids[0, 2] = 9
    changed_token_score = cache.get_or_compute(
        changed_ids,
        batch.chosen_attention_mask,
        batch.chosen_completion_mask,
    )
    assert len(cache) == 3
    assert not torch.allclose(
        changed_token_score.token_logprobs, chosen.token_logprobs
    )

    changed_mask = batch.chosen_completion_mask.clone()
    changed_mask[0, 1] = 1
    wrong_mask_score = cache.get_or_compute(
        batch.chosen_input_ids, batch.chosen_attention_mask, changed_mask
    )
    assert len(cache) == 4
    assert not torch.allclose(
        wrong_mask_score.sequence_logprobs, chosen.sequence_logprobs
    )

    with torch.no_grad():
        model.token_embedding.weight.add_(0.01)
    after_weight_change = cache.get_or_compute(
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_completion_mask,
    )
    assert len(cache) == 5
    assert not torch.allclose(
        after_weight_change.sequence_logprobs, chosen.sequence_logprobs
    )
    assert rejected.sequence_logprobs.requires_grad is False


def test_reference_cache_key_includes_attention_mask():
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    completion_mask = torch.tensor([[0, 1, 1, 0]])
    changed_attention_mask = torch.tensor([[1, 1, 1, 1]])
    model = FixedLogitsModel(torch.randn(1, 4, 12, dtype=torch.float64))
    cache = ReferenceScoreCache(model)

    first = cache.get_or_compute(input_ids, attention_mask, completion_mask)
    second = cache.get_or_compute(input_ids, changed_attention_mask, completion_mask)

    assert len(cache) == 2
    assert torch.allclose(first.sequence_logprobs, second.sequence_logprobs)


def test_shared_policy_and_reference_parameters_are_rejected():
    policy = make_tiny_model(seed=41)
    with pytest.raises(ValueError, match="share"):
        dpo_step(
            policy,
            policy,
            make_batch(),
            beta=0.5,
        )
