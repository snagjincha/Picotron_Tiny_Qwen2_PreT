"""Minimal Qwen2-style dense causal language model for pretraining."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .attention import GroupedQueryAttention
from .config import Qwen2Config
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(self.self_attn(self.input_layernorm(hidden_states)))
        return hidden_states + self.dropout(self.mlp(self.post_attention_layernorm(hidden_states)))


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return F.linear(hidden_states, self.embed_tokens.weight)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
