"""Single-batch overfit test required before any full-corpus run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from pretraining.model import Qwen2Config, Qwen2ForCausalLM

from .optimizer import build_optimizer
from .train import build_scheduler, train_step


@dataclass
class OverfitResult:
    initial_loss: float
    final_loss: float
    steps: int


def tiny_overfit_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
    )


def run_single_batch_overfit(
    optimizer_config: Mapping[str, Any],
    steps: int = 80,
    device: torch.device | None = None,
) -> OverfitResult:
    """Fit a tiny repeated sequence; final loss must be materially lower than initial loss."""
    torch.manual_seed(42)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2ForCausalLM(tiny_overfit_config()).to(device)
    optimizer = build_optimizer(model, optimizer_config)
    training_config = {
        "max_steps": steps,
        "scheduler": {"name": "cosine", "warmup_ratio": 0.0, "min_lr_ratio": 1.0},
    }
    scheduler = build_scheduler(optimizer, training_config)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long),
        "labels": torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long),
    }
    first_metrics = train_step(
        model, [batch], optimizer, scheduler, device, precision="fp32", gradient_clip_norm=1.0
    )
    for _ in range(steps - 1):
        last_metrics = train_step(
            model, [batch], optimizer, scheduler, device, precision="fp32", gradient_clip_norm=1.0
        )
    return OverfitResult(initial_loss=first_metrics.loss, final_loss=last_metrics.loss, steps=steps)
