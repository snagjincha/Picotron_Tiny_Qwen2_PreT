"""Warmup followed by cosine decay, shared by AdamW and Muon hybrid optimizers."""

from __future__ import annotations

import math
from typing import Protocol

import torch

from .optimizer import MuonHybridOptimizer


class _HasParameterGroups(Protocol):
    param_groups: list[dict[str, float]]


def optimizer_parts(optimizer: torch.optim.Optimizer | MuonHybridOptimizer) -> list[_HasParameterGroups]:
    if isinstance(optimizer, MuonHybridOptimizer):
        return [optimizer.muon, optimizer.adamw]
    return [optimizer]


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer | MuonHybridOptimizer,
        total_steps: int,
        warmup_ratio: float,
        min_lr_ratio: float,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.parts = optimizer_parts(optimizer)
        self.total_steps = total_steps
        self.warmup_steps = max(1, round(total_steps * warmup_ratio))
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [[group["lr"] for group in part.param_groups] for part in self.parts]
        self.step_count = 0

    def scale_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self) -> None:
        scale = self.scale_at(self.step_count)
        for part, lrs in zip(self.parts, self.base_lrs):
            for group, base_lr in zip(part.param_groups, lrs):
                group["lr"] = base_lr * scale
        self.step_count += 1

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for part in self.parts for group in part.param_groups]
