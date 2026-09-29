from __future__ import annotations

import copy

import torch
from torch import nn

from dpo_lab.model import TinyCausalTransformer, TinyCausalTransformerConfig


class FixedLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.detach().clone())

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        del input_ids, attention_mask
        return self.logits


def make_tiny_model(
    vocab_size: int = 12,
    hidden_size: int = 16,
    n_heads: int = 2,
    mlp_intermediate_size: int = 32,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> TinyCausalTransformer:
    torch.manual_seed(seed)
    config = TinyCausalTransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        mlp_intermediate_size=mlp_intermediate_size,
    )
    return TinyCausalTransformer(config).to(dtype=dtype)


def clone_model(model: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(model)
    cloned.load_state_dict(model.state_dict())
    cloned.to(next(model.parameters()).dtype)
    return cloned


def clone_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def zero_grads(model: nn.Module) -> None:
    for param in model.parameters():
        param.grad = None
