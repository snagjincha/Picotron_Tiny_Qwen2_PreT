"""PyTorch dataset over a packed token stream."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, token_path: str | Path) -> None:
        self.token_path = Path(token_path)
        metadata = json.loads(self.token_path.with_suffix(".json").read_text())
        self.sequence_length = int(metadata["sequence_length"])
        self.train_token_count = int(metadata["train_token_count"])
        if self.train_token_count % self.sequence_length:
            raise ValueError("Packed metadata has a non-integral number of sequences")
        self.tokens = np.memmap(
            self.token_path,
            mode="r",
            dtype=np.dtype(metadata["dtype"]),
            shape=(int(metadata["source_token_count"]),),
        )

    def __len__(self) -> int:
        return self.train_token_count // self.sequence_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        token_window = np.array(self.tokens[start : start + self.sequence_length + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(token_window[:-1]),
            "labels": torch.from_numpy(token_window[1:]),
        }
