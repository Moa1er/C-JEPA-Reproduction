from typing import Dict, Any

import torch


def validate_batch_output(batch: Dict[str, Any]) -> None:
    slots = batch["slots"]
    if not isinstance(slots, torch.Tensor):
        raise TypeError("slots must be torch.Tensor")
    if slots.ndim != 4:
        raise ValueError(f"slots must be [B, T, N, D], got {tuple(slots.shape)}")
    if not torch.isfinite(slots).all():
        raise ValueError("slots contains NaN or Inf")
