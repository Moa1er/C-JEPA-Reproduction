from pathlib import Path
from typing import Any, Dict

import torch


def save_video_slots(
    output_dir: str,
    video_id: str,
    slots_t_n_d: torch.Tensor,
    frame_indices_t: torch.Tensor,
    metadata: Dict[str, Any],
    slot_masks: torch.Tensor = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.pt"

    payload = {
        "video_id": video_id,
        "slots": slots_t_n_d.cpu(),
        "frame_indices": frame_indices_t.cpu(),
        "metadata": metadata,
    }
    if slot_masks is not None:
        payload["slot_masks"] = slot_masks.cpu()

    torch.save(payload, out_path)
    return out_path
