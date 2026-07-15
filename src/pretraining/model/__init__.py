"""Qwen2-style dense causal language-model components."""

from .config import Qwen2Config
from .qwen2 import Qwen2ForCausalLM

__all__ = ["Qwen2Config", "Qwen2ForCausalLM"]
