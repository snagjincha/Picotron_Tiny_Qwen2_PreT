"""Loss functions for causal language-model pretraining."""

import torch
from torch.nn import functional as F


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return mean next-token cross entropy for already-shifted labels."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch and sequence dimensions")
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())
