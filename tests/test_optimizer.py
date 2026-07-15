import unittest

import torch

from pretraining.model import Qwen2Config, Qwen2ForCausalLM
from pretraining.training.optimizer import MuonHybridOptimizer, build_optimizer


def tiny_model() -> Qwen2ForCausalLM:
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            max_position_embeddings=16,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
        )
    )


class OptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adamw = {"learning_rate": 3e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}
        self.muon = {
            "muon_learning_rate": 0.02,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "weight_decay": 0.1,
            "auxiliary_adamw_learning_rate": 3e-4,
            "auxiliary_adamw_betas": [0.9, 0.95],
            "auxiliary_adamw_eps": 1e-8,
            "auxiliary_adamw_weight_decay": 0.1,
        }

    def test_adamw_recipe(self) -> None:
        optimizer = build_optimizer(tiny_model(), {"name": "adamw", "adamw": self.adamw})
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_muon_hybrid_keeps_embedding_in_adamw(self) -> None:
        model = tiny_model()
        optimizer = build_optimizer(
            model, {"name": "muon_hybrid", "adamw": self.adamw, "muon_hybrid": self.muon}
        )
        self.assertIsInstance(optimizer, MuonHybridOptimizer)
        muon_ids = {id(parameter) for group in optimizer.muon.param_groups for parameter in group["params"]}
        adamw_ids = {id(parameter) for group in optimizer.adamw.param_groups for parameter in group["params"]}
        self.assertIn(id(model.layers[0].self_attn.q_proj.weight), muon_ids)
        self.assertIn(id(model.embed_tokens.weight), adamw_ids)
