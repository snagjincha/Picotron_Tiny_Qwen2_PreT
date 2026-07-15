import unittest

import torch

from pretraining.training.overfit import run_single_batch_overfit
from pretraining.training.scheduler import WarmupCosineScheduler


ADAMW = {
    "name": "adamw",
    "adamw": {"learning_rate": 3e-3, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.0},
}


class TrainingTest(unittest.TestCase):
    def test_overfit_loss_decreases(self) -> None:
        result = run_single_batch_overfit(ADAMW, steps=60, device=torch.device("cpu"))
        self.assertLess(result.final_loss, result.initial_loss * 0.5)

    def test_warmup_cosine_scale(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        scheduler = WarmupCosineScheduler(optimizer, total_steps=10, warmup_ratio=0.2, min_lr_ratio=0.1)
        scheduler.step()
        self.assertEqual(scheduler.get_last_lr(), [0.5])
        scheduler.step()
        self.assertEqual(scheduler.get_last_lr(), [1.0])
