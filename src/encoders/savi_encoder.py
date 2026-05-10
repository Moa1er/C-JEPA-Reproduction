from typing import Any, Dict

import torch
import torch.nn as nn


class SAViEncoder(nn.Module):
    """
    Thin frozen wrapper around a pretrained SAVi-like checkpoint.

    Expected forward contract from loaded model:
      - input: dict or tensor containing frames [B, T, C, H, W]
      - output: either
          a) dict with key `output_slots_key`, or
          b) tensor slots directly with shape [B, T, N, D]

    TODO(teammate): Replace `_load_raw_checkpoint` and `_run_model_forward`
    with the exact SlotFormer/SAVi loading + forward logic used in your
    validated local diagnostics.
    """

    def __init__(
        self,
        checkpoint_path: str,
        input_key: str = "frames",
        output_slots_key: str = "slots",
        slot_masks_key: str = "slot_masks",
        map_location: str = "cpu",
    ):
        super().__init__()
        if not checkpoint_path:
            raise ValueError("encoder.checkpoint_path must be set")

        self.input_key = input_key
        self.output_slots_key = output_slots_key
        self.slot_masks_key = slot_masks_key
        self.model = self._load_raw_checkpoint(checkpoint_path, map_location)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _load_raw_checkpoint(self, checkpoint_path: str, map_location: str) -> nn.Module:
        """Placeholder checkpoint loader for SAVi-style models."""
        loaded = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(loaded, dict) and "model" in loaded:
            loaded = loaded["model"]

        if not isinstance(loaded, nn.Module):
            raise TypeError(
                "Loaded checkpoint is not nn.Module. "
                "TODO(teammate): adapt _load_raw_checkpoint for your SAVi format."
            )
        return loaded

    def _run_model_forward(self, frames: torch.Tensor) -> Any:
        """
        Placeholder forward adapter.

        TODO(teammate): if your SAVi checkpoint expects a different input structure
        (e.g. specific key names, batch dict layout, or preprocessing metadata),
        adapt this function.
        """
        return self.model({self.input_key: frames}) if self.input_key else self.model(frames)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> Dict[str, Any]:
        out = self._run_model_forward(frames)

        if isinstance(out, dict):
            if self.output_slots_key not in out:
                raise KeyError(
                    f"Output dict does not contain '{self.output_slots_key}'. "
                    f"Available keys: {list(out.keys())}"
                )
            slots = out[self.output_slots_key]
            slot_masks = out.get(self.slot_masks_key)
            metadata = {
                k: v for k, v in out.items() if k not in (self.output_slots_key, self.slot_masks_key)
            }
        else:
            slots = out
            slot_masks = None
            metadata = {}

        if not isinstance(slots, torch.Tensor):
            raise TypeError("slots output must be a torch.Tensor")
        if slots.ndim != 4:
            raise ValueError(f"Expected slots shape [B, T, N, D], got {tuple(slots.shape)}")

        return {
            "slots": slots,
            "slot_masks": slot_masks,
            "metadata": metadata,
        }
