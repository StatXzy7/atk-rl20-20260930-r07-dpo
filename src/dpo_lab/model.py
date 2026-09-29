from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TinyCausalTransformerConfig:
    vocab_size: int
    hidden_size: int = 32
    n_heads: int = 2
    mlp_intermediate_size: int = 64

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyCausalTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.hidden_size // config.n_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, seq_len, self.n_heads, self.head_size
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        key_padding = attention_mask[:, None, None, :].to(torch.bool)
        allowed = causal[None, None, :, :] & key_padding

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size**0.5)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        return self.proj(context)


class TinyCausalTransformer(nn.Module):
    """Small embedding + one causal transformer block used for real updates."""

    def __init__(self, config: TinyCausalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_intermediate_size),
            nn.Tanh(),
            nn.Linear(config.mlp_intermediate_size, config.hidden_size),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must have shape [batch, seq]")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")
        invalid_padding = (attention_mask.to(torch.bool) == 0) & (
            input_ids != self.pad_token_id
        )
        if bool(invalid_padding.any()):
            raise ValueError("masked positions must contain the pad token")

        hidden = self.token_embedding(input_ids)
        attention_input = self.attention_norm(hidden)
        hidden = hidden + self.attention(attention_input, attention_mask)
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return self.lm_head(hidden)

    @property
    def pad_token_id(self) -> int:
        return 0
