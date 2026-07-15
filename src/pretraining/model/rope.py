"""Rotary positional embeddings used by Qwen2 attention."""

import torch


def rotary_frequencies(
    sequence_length: int,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    inverse_frequencies = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    frequencies = torch.outer(positions, inverse_frequencies)
    return frequencies.cos(), frequencies.sin()


def apply_rope(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to tensors shaped ``[batch, sequence, heads, head_dim]``."""
    cos = cos[None, :, None, :].to(dtype=states.dtype)
    sin = sin[None, :, None, :].to(dtype=states.dtype)
    even = states[..., 0::2]
    odd = states[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(start_dim=-2)
