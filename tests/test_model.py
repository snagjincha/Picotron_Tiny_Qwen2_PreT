import unittest

import torch

from pretraining.model import Qwen2Config, Qwen2ForCausalLM


def tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
    )


class Qwen2ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = Qwen2ForCausalLM(tiny_config()).eval()

    def test_logits_shape(self) -> None:
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        logits = self.model(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 3, 32))

    def test_future_tokens_do_not_change_past_logits(self) -> None:
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 9, 10]])
        first_logits = self.model(first)
        second_logits = self.model(second)
        torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])

    def test_embedding_and_lm_head_are_tied(self) -> None:
        self.assertFalse(hasattr(self.model, "lm_head"))
        self.assertEqual(self.model.num_parameters, 5_200)

    def test_rejects_long_sequences(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 17), dtype=torch.long))
