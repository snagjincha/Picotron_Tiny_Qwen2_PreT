#!/usr/bin/env python3
"""Validate the static parameter budget for the Qwen2-style 100M configuration."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


def parameter_count(config: dict[str, object]) -> int:
    """Return parameter count for bias-free, tied-embedding Qwen2-style dense LM."""
    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config["num_key_value_heads"])
    num_heads = int(config["num_attention_heads"])
    intermediate_size = int(config["intermediate_size"])

    if hidden_size % num_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if num_heads % num_kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if not bool(config["tie_word_embeddings"]):
        raise ValueError("This budget assumes tied input/output embeddings")

    head_dim = hidden_size // num_heads
    embedding_parameters = vocab_size * hidden_size
    attention_parameters = (
        hidden_size * hidden_size
        + 2 * hidden_size * (num_kv_heads * head_dim)
        + hidden_size * hidden_size
    )
    mlp_parameters = 3 * hidden_size * intermediate_size
    rmsnorm_parameters = 2 * hidden_size
    transformer_block_parameters = attention_parameters + mlp_parameters + rmsnorm_parameters
    final_norm_parameters = hidden_size

    return embedding_parameters + num_layers * transformer_block_parameters + final_norm_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/qwen2_100m.yaml"),
        help="Path to the model YAML configuration.",
    )
    parser.add_argument(
        "--global-batch-tokens",
        type=int,
        default=524_288,
        help="Tokens consumed by each optimizer step.",
    )
    args = parser.parse_args()

    with args.config.open() as file:
        config = yaml.safe_load(file)

    parameters = parameter_count(config)
    expected = int(config["expected_parameter_count"])
    token_budget = parameters * 20
    required_steps = math.ceil(token_budget / args.global_batch_tokens)

    print(f"model: {config['name']}")
    print(f"parameters: {parameters:,}")
    print(f"expected_parameters: {expected:,}")
    print(f"chinchilla_train_token_budget: {token_budget:,}")
    print(f"steps_at_{args.global_batch_tokens:,}_tokens_per_step: {required_steps:,}")

    if parameters != expected:
        raise SystemExit("Configured expected_parameter_count does not match the architecture.")


if __name__ == "__main__":
    main()
