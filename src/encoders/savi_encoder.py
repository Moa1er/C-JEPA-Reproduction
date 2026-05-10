from typing import Any, Dict

import torch
import torch.nn as nn

from src.encoders.savi import StoSAVi, STOSAVI_CLEVRER_CFG


class SAViEncoder(nn.Module):
    """
    Thin frozen wrapper around the released SlotFormer StoSAVi-on-CLEVRER
    checkpoint.

    The wrapped model is constructed via `StoSAVi(**STOSAVI_CLEVRER_CFG)`
    and loaded from a state_dict (the released `.pth` is a state_dict, not a
    pickled `nn.Module`). At inference we call `model({"img": frames})` and
    surface `post_slots` / `post_masks` under configurable keys.

    Output contract:
      slots:      Tensor[B, T, num_slots=7, slot_size=128]
      slot_masks: Tensor[B, T, num_slots, 1, H, W] or None
    """

    def __init__(
        self,
        checkpoint_path: str,
        input_key: str = "img",
        output_slots_key: str = "post_slots",
        slot_masks_key: str = "post_masks",
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
        """Build StoSAVi from the CLEVRER config and load the released weights."""
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            sd = ckpt

        model = StoSAVi(**STOSAVI_CLEVRER_CFG)
        result = model.load_state_dict(sd, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"SAVi state_dict did not match: "
                f"missing={result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''} "
                f"unexpected={result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}"
            )
        # decoder path active in eval -> we get post_recons / post_masks
        model.testing = False
        return model

    def _run_model_forward(self, frames: torch.Tensor) -> Any:
        return self.model({self.input_key: frames})

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
