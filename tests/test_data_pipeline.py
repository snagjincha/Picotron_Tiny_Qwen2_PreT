import tempfile
import unittest
from pathlib import Path

import torch

from pretraining.data.cleaning import clean_document
from pretraining.data.dataset import PackedTokenDataset
from pretraining.data.packing import TokenStreamWriter
from pretraining.data.split import ExactDeduplicator, document_hash, is_validation_document


class DataPipelineTest(unittest.TestCase):
    def test_cleaning_normalizes_and_rejects(self) -> None:
        cleaned = clean_document("<p>  caf\u00e9\ttext  </p>", min_characters=5)
        self.assertEqual(cleaned, "café text")
        self.assertIsNone(clean_document("too short", min_characters=100))
        self.assertIsNone(clean_document("a" * 100, min_characters=5))

    def test_split_and_deduplication_are_deterministic(self) -> None:
        digest = document_hash("a stable document")
        self.assertEqual(is_validation_document(digest, 0.01, 42), is_validation_document(digest, 0.01, 42))
        with tempfile.TemporaryDirectory() as directory:
            with ExactDeduplicator(Path(directory) / "dedup.sqlite") as dedup:
                self.assertTrue(dedup.add_if_new(digest))
                self.assertFalse(dedup.add_if_new(digest))

    def test_packing_generates_shifted_fixed_length_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "train.bin"
            writer = TokenStreamWriter(token_path, token_budget=8)
            writer.append([1, 2, 3, 4])
            writer.append([5, 6, 7, 8, 9, 10])
            metadata = writer.finalize(sequence_length=4)
            self.assertEqual(metadata.train_token_count, 8)
            dataset = PackedTokenDataset(token_path)
            self.assertEqual(len(dataset), 2)
            torch.testing.assert_close(dataset[0]["input_ids"], torch.tensor([1, 2, 3, 4]))
            torch.testing.assert_close(dataset[0]["labels"], torch.tensor([2, 3, 4, 5]))
            torch.testing.assert_close(dataset[1]["input_ids"], torch.tensor([5, 6, 7, 8]))
            torch.testing.assert_close(dataset[1]["labels"], torch.tensor([6, 7, 8, 9]))
