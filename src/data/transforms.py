from typing import Callable, Dict, Any

import cv2
import numpy as np
import torch


def _resolve_interpolation(name: str) -> int:
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported interpolation: {name}")
    return mapping[name]


def build_frame_transform(cfg: Dict[str, Any]) -> Callable[[np.ndarray], torch.Tensor]:
    resize = int(cfg["resize"])
    interpolation = _resolve_interpolation(cfg.get("interpolation", "bilinear"))
    mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(cfg["std"], dtype=torch.float32).view(3, 1, 1)

    def transform(frame_bgr: np.ndarray) -> torch.Tensor:
        # OpenCV reads BGR; convert to RGB to match pretrained vision backbones.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (resize, resize), interpolation=interpolation)
        frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame = (frame - mean) / std
        return frame

    return transform
