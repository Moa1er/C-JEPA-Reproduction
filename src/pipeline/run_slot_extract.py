from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.clevrer_dataset import CLEVRERVideoDataset, collate_video_batch
from src.data.transforms import build_frame_transform
from src.encoders.savi_encoder import SAViEncoder
from src.pipeline.save_slots import save_video_slots
from src.pipeline.validate_slots import validate_batch_output


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def run_slot_extract(cfg: Dict[str, Any]) -> None:
    device = _resolve_device(cfg["device"])
    torch.manual_seed(int(cfg["seed"]))

    transform = build_frame_transform(cfg["data"])
    dataset = CLEVRERVideoDataset(
        index_file=cfg["data"]["index_file"],
        clip_len=int(cfg["data"]["clip_len"]),
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["run"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        collate_fn=collate_video_batch,
    )

    encoder = SAViEncoder(
        checkpoint_path=cfg["encoder"]["checkpoint_path"],
        input_key=cfg["encoder"].get("input_key", "frames"),
        output_slots_key=cfg["encoder"].get("output_slots_key", "slots"),
        slot_masks_key=cfg["encoder"].get("slot_masks_key", "slot_masks"),
        map_location=str(device),
    ).to(device)

    max_videos = int(cfg["run"]["max_videos"])
    processed = 0

    for batch in tqdm(loader, desc="Extracting slots"):
        frames = batch["frames"].to(device, non_blocking=True)  # [B, T, C, H, W]
        out = encoder(frames)
        validate_batch_output(out)

        slots = out["slots"]  # [B, T, N, D]
        slot_masks = out.get("slot_masks")
        out_metadata = out.get("metadata", {})
        expected_num_slots = cfg["encoder"].get("expected_num_slots")
        expected_slot_dim = cfg["encoder"].get("expected_slot_dim")

        if expected_num_slots is not None and slots.shape[2] != int(expected_num_slots):
            raise ValueError(
                f"Expected N={expected_num_slots} slots, got N={slots.shape[2]}. "
                "Update encoder.expected_num_slots or SAVi checkpoint mapping."
            )
        if expected_slot_dim is not None and slots.shape[3] != int(expected_slot_dim):
            raise ValueError(
                f"Expected D={expected_slot_dim} slot dim, got D={slots.shape[3]}. "
                "Update encoder.expected_slot_dim or SAVi checkpoint mapping."
            )

        bsz = slots.shape[0]
        for i in range(bsz):
            metadata = {
                "checkpoint_path": cfg["encoder"]["checkpoint_path"],
                "clip_len": int(cfg["data"]["clip_len"]),
                "resize": int(cfg["data"]["resize"]),
                "model_metadata": out_metadata,
            }
            save_video_slots(
                output_dir=cfg["run"]["output_dir"],
                video_id=batch["video_id"][i],
                slots_t_n_d=slots[i],
                frame_indices_t=batch["frame_indices"][i],
                metadata=metadata,
                slot_masks=slot_masks[i] if slot_masks is not None else None,
            )
            processed += 1
            if processed >= max_videos:
                return
