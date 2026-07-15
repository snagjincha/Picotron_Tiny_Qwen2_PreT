"""Configuration for a compact, bias-free Qwen2-style dense language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 1_000_000.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.vocab_size <= 0 or self.max_position_embeddings <= 0:
            raise ValueError("vocab_size and max_position_embeddings must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Qwen2Config":
        with Path(path).open() as file:
            values = yaml.safe_load(file)
        allowed = {
            "vocab_size",
            "max_position_embeddings",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
            "attention_dropout",
            "residual_dropout",
            "initializer_range",
            "tie_word_embeddings",
        }
        return cls(**{key: value for key, value in values.items() if key in allowed})
