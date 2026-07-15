"""Fixed-length causal-LM token packing and memory-mapped storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PackedTokenMetadata:
    sequence_length: int
    train_token_count: int
    source_token_count: int
    dtype: str
    sha256: str


class TokenStreamWriter:
    """Write ``token_budget + 1`` source tokens needed for shifted LM labels."""

    def __init__(self, path: str | Path, token_budget: int) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.token_budget = token_budget
        self.source_token_count = token_budget + 1
        self.tokens = np.memmap(self.path, mode="w+", dtype=np.uint32, shape=(self.source_token_count,))
        self.written = 0
        self.documents = 0

    @property
    def complete(self) -> bool:
        return self.written == self.source_token_count

    def append(self, token_ids: Iterable[int]) -> None:
        if self.complete:
            return
        values = np.asarray(list(token_ids), dtype=np.uint32)
        if not len(values):
            return
        count = min(len(values), self.source_token_count - self.written)
        self.tokens[self.written : self.written + count] = values[:count]
        self.written += count
        self.documents += 1

    def finalize(self, sequence_length: int) -> PackedTokenMetadata:
        if not self.complete:
            raise RuntimeError(f"Token stream is incomplete: {self.written:,}/{self.source_token_count:,}")
        if self.token_budget % sequence_length:
            raise ValueError("token_budget must be divisible by sequence_length")
        self.tokens.flush()
        digest = hashlib.sha256()
        with self.path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        metadata = PackedTokenMetadata(
            sequence_length=sequence_length,
            train_token_count=self.token_budget,
            source_token_count=self.source_token_count,
            dtype="uint32",
            sha256=digest.hexdigest(),
        )
        self.path.with_suffix(".json").write_text(json.dumps(asdict(metadata), indent=2) + "\n")
        return metadata
