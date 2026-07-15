"""C4 streaming preparation entry point; no dataset is downloaded until called."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .cleaning import clean_document
from .packing import TokenStreamWriter
from .split import ExactDeduplicator, document_hash, is_validation_document


@dataclass
class PreparationStats:
    seen_documents: int = 0
    kept_documents: int = 0
    duplicate_documents: int = 0
    rejected_documents: int = 0


def prepare_c4(config_path: str | Path) -> dict[str, Any]:
    """Stream C4, clean/split/tokenize documents, and write packed token streams."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    with Path(config_path).open() as file:
        config = yaml.safe_load(file)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenization = config["tokenization"]
    sequence_length = int(tokenization["sequence_length"])
    train_writer = TokenStreamWriter(output_dir / "train.bin", int(config["train_token_budget"]))
    validation_writer = TokenStreamWriter(output_dir / "validation.bin", int(config["validation_token_budget"]))
    tokenizer = AutoTokenizer.from_pretrained(tokenization["tokenizer_name"])
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer must define eos_token_id")
    dataset_config = config["dataset"]
    stream = load_dataset(
        dataset_config["name"],
        dataset_config["config"],
        split=dataset_config["split"],
        streaming=dataset_config["streaming"],
    )
    preprocessing = config["preprocessing"]
    stats = PreparationStats()
    with ExactDeduplicator(output_dir / "dedup.sqlite") as deduplicator:
        for row in stream:
            if train_writer.complete and validation_writer.complete:
                break
            stats.seen_documents += 1
            text = clean_document(row["text"], int(preprocessing["min_characters"]))
            if text is None:
                stats.rejected_documents += 1
                continue
            digest = document_hash(text)
            if not deduplicator.add_if_new(digest):
                stats.duplicate_documents += 1
                continue
            stats.kept_documents += 1
            token_ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            if is_validation_document(digest, preprocessing["validation_fraction"], preprocessing["split_seed"]):
                validation_writer.append(token_ids)
            else:
                train_writer.append(token_ids)

    train_metadata = train_writer.finalize(sequence_length)
    validation_metadata = validation_writer.finalize(sequence_length)
    manifest = {
        "dataset": dataset_config,
        "tokenizer": tokenization["tokenizer_name"],
        "sequence_length": sequence_length,
        "preprocessing": preprocessing,
        "stats": stats.__dict__,
        "train": {**train_metadata.__dict__, "documents": train_writer.documents},
        "validation": {**validation_metadata.__dict__, "documents": validation_writer.documents},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
