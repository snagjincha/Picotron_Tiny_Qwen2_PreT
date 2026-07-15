#!/usr/bin/env python3
"""Run the mandatory fixed-batch overfit check before full pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pretraining.training.overfit import run_single_batch_overfit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/overfit.yaml")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    with Path(args.config).open() as file:
        optimizer_config = yaml.safe_load(file)["optimizer"]
    result = run_single_batch_overfit(optimizer_config, steps=args.steps)
    print(f"optimizer: {optimizer_config['name']}")
    print(f"initial_loss: {result.initial_loss:.4f}")
    print(f"final_loss: {result.final_loss:.4f}")
    if result.final_loss >= result.initial_loss * 0.5:
        raise SystemExit("Overfit test failed: loss did not decrease by at least 50%.")


if __name__ == "__main__":
    main()
