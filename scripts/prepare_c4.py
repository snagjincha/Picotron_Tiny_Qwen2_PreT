#!/usr/bin/env python3
"""Prepare the configured C4 subset for 100M-model pretraining."""

from __future__ import annotations

import argparse
import json

from pretraining.data.prepare_c4 import prepare_c4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/c4_en_2b.yaml")
    args = parser.parse_args()
    manifest = prepare_c4(args.config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
