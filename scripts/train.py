#!/usr/bin/env python3
"""Run a single-process smoke or pilot pretraining job on packed token files."""

from __future__ import annotations

import argparse
from itertools import cycle, islice
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from pretraining.data import PackedTokenDataset
from pretraining.model import Qwen2Config, Qwen2ForCausalLM
from pretraining.training import build_optimizer, build_scheduler, evaluate_loss, train_step


def load_yaml(path: str) -> dict:
    with Path(path).open() as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/qwen2_100m.yaml")
    parser.add_argument("--train-config", default="configs/train/smoke.yaml")
    parser.add_argument("--train-tokens", required=True, help="Packed train .bin file")
    parser.add_argument("--validation-tokens", required=True, help="Packed validation .bin file")
    parser.add_argument("--max-steps", type=int, help="Override configured max_steps")
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    training_config = load_yaml(args.train_config)
    if args.max_steps is not None:
        training_config["max_steps"] = args.max_steps
    torch.manual_seed(training_config["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available in this process")

    model = Qwen2ForCausalLM(Qwen2Config.from_yaml(args.model_config)).to(device)
    optimizer = build_optimizer(model, training_config["optimizer"])
    scheduler = build_scheduler(optimizer, training_config)
    batch_size = int(training_config["micro_batch_size"])
    accumulation = int(training_config["gradient_accumulation_steps"])
    train_loader = DataLoader(PackedTokenDataset(args.train_tokens), batch_size=batch_size, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(
        PackedTokenDataset(args.validation_tokens), batch_size=batch_size, pin_memory=device.type == "cuda"
    )
    train_iterator = cycle(train_loader)

    for step in range(1, int(training_config["max_steps"]) + 1):
        batches = list(islice(train_iterator, accumulation))
        metrics = train_step(
            model=model,
            batches=batches,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            precision=training_config["precision"],
            gradient_clip_norm=float(training_config["gradient_clip_norm"]),
        )
        if step % int(training_config["log_interval"]) == 0:
            lr_string = ",".join(f"{lr:.2e}" for lr in metrics.learning_rates)
            print(f"step={step} train_loss={metrics.loss:.4f} grad_norm={metrics.gradient_norm:.4f} lr={lr_string}")
        if step % int(training_config["validation_interval"]) == 0:
            validation_loss = evaluate_loss(
                model=model,
                batches=islice(validation_loader, args.validation_batches),
                device=device,
                precision=training_config["precision"],
            )
            print(f"step={step} validation_loss={validation_loss:.4f} validation_ppl={torch.exp(torch.tensor(validation_loss)).item():.4f}")


if __name__ == "__main__":
    main()
