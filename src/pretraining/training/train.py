"""Single-process training primitives, used before Picotron distributed launch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any

import torch
from torch import nn

from .loss import causal_lm_loss
from .optimizer import MuonHybridOptimizer
from .scheduler import WarmupCosineScheduler, optimizer_parts


@dataclass
class TrainMetrics:
    loss: float
    gradient_norm: float
    learning_rates: list[float]


def _autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _optimizer_zero_grad(optimizer: torch.optim.Optimizer | MuonHybridOptimizer) -> None:
    optimizer.zero_grad(set_to_none=True)


def _optimizer_step(optimizer: torch.optim.Optimizer | MuonHybridOptimizer) -> None:
    optimizer.step()


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    gradients = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return torch.stack(gradients).norm(2).item()


def train_step(
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | MuonHybridOptimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    precision: str,
    gradient_clip_norm: float,
) -> TrainMetrics:
    """Run one optimizer update, optionally accumulating several micro-batches."""
    model.train()
    _optimizer_zero_grad(optimizer)
    batches = list(batches)
    if not batches:
        raise ValueError("At least one micro-batch is required")
    total_loss = 0.0
    for batch in batches:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with _autocast_context(device, precision):
            loss = causal_lm_loss(model(input_ids), labels)
            (loss / len(batches)).backward()
        total_loss += loss.detach().float().item()
    unclipped_norm = gradient_norm(model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    _optimizer_step(optimizer)
    scheduler.step()
    return TrainMetrics(
        loss=total_loss / len(batches),
        gradient_norm=unclipped_norm,
        learning_rates=scheduler.get_last_lr(),
    )


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    precision: str,
) -> float:
    model.eval()
    losses: list[torch.Tensor] = []
    for batch in batches:
        with _autocast_context(device, precision):
            logits = model(batch["input_ids"].to(device, non_blocking=True))
            losses.append(causal_lm_loss(logits, batch["labels"].to(device, non_blocking=True)).float())
    if not losses:
        raise ValueError("At least one evaluation batch is required")
    return torch.stack(losses).mean().item()


def build_scheduler(
    optimizer: torch.optim.Optimizer | MuonHybridOptimizer,
    training_config: Mapping[str, Any],
) -> WarmupCosineScheduler:
    settings = training_config["scheduler"]
    if settings["name"] != "cosine":
        raise ValueError(f"Unsupported scheduler: {settings['name']}")
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_steps=int(training_config["max_steps"]),
        warmup_ratio=float(settings["warmup_ratio"]),
        min_lr_ratio=float(settings["min_lr_ratio"]),
    )
