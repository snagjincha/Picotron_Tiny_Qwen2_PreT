"""Grouped-query causal self-attention with RoPE."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import Qwen2Config
from .rope import apply_rope, rotary_frequencies


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.dropout_p = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        kv_size = config.num_key_value_heads * config.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).view(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        cos, sin = rotary_frequencies(sequence_length, self.head_dim, self.rope_theta, hidden_states.device)
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        repeat_factor = self.num_attention_heads // self.num_key_value_heads
        keys = keys.repeat_interleave(repeat_factor, dim=2)
        values = values.repeat_interleave(repeat_factor, dim=2)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.o_proj(attended)
