"""Stable document-level splitting and exact deduplication."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_validation_document(digest: str, validation_fraction: float, seed: int) -> bool:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    value = hashlib.sha256(f"{seed}:{digest}".encode("ascii")).digest()
    bucket = int.from_bytes(value[:8], "big") / 2**64
    return bucket < validation_fraction


class ExactDeduplicator:
    """Disk-backed hash set suitable for a streamed corpus."""

    def __init__(self, database_path: str | Path) -> None:
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS document_hashes (digest TEXT PRIMARY KEY)")

    def add_if_new(self, digest: str) -> bool:
        cursor = self.connection.execute("INSERT OR IGNORE INTO document_hashes VALUES (?)", (digest,))
        return cursor.rowcount == 1

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "ExactDeduplicator":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
