"""Generate slot diagnostic figures from a single CLEVRER mp4.

End-to-end smoke for the merged pipeline: drives src.encoders.SAViEncoder
(not StoSAVi directly) and uses cfg["data"] preprocessing to match what
the rest of the repo feeds the encoder.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/make_diagnostic_figures.py \\
        /path/to/video_NNNNN.mp4 \\
        --tag v_NNNNN --start-frame 0 --stride 2

Outputs to figures/<tag>_slot_masks.png,
            figures/<tag>_slot_reconstructions.png,
            figures/<tag>_temporal_consistency_slot<K>.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transforms import build_frame_transform  # noqa: E402
from src.diagnostics import (  # noqa: E402
    visualize_slot_masks,
    visualize_slot_reconstructions,
    visualize_temporal_consistency,
)
from src.encoders.savi_encoder import SAViEncoder  # noqa: E402


SEED = 42


def load_single_clip(
    video_path: Path,
    transform: Callable,
    n_frames: int,
    stride: int,
    start_frame: int,
) -> torch.Tensor:
    """Load n_frames from a video at fixed stride/start, apply per-frame transform.

    Mirrors src.data.clevrer_dataset's reader (cv2 seek + per-frame transform)
    so the diagnostics see exactly what training/extraction sees.

    Returns: Tensor[T, 3, H, W] float32.
    """
    indices = [start_frame + n * stride for n in range(n_frames)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if indices[-1] >= total:
        cap.release()
        raise ValueError(
            f"Video {video_path.name} has {total} frames; need index "
            f"{indices[-1]} (start={start_frame}, stride={stride}, n={n_frames})."
        )

    raw_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        raw_frames.append(frame)
    cap.release()

    return torch.stack([transform(f) for f in raw_frames], dim=0)


def autopick_non_background_slot(masks: torch.Tensor) -> int:
    """Pick the slot with the second-largest total mask mass as a rough
    default.

    The largest is usually background, and the second is often a foreground
    object — but not always. CLEVRER scenes vary: a static foreground slot
    can have high mass while the moving slot has lower mass. For figures
    intended for the report, pass --slot-track explicitly after eyeballing
    the slot_masks figure.
    """
    # masks: (T, N, 1, H, W)
    mass = masks.sum(dim=(0, 2, 3, 4))                     # (N,)
    order = torch.argsort(mass, descending=True)
    return int(order[1].item()) if mass.numel() >= 2 else int(order[0].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("video", type=Path, help="path to a CLEVRER .mp4")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--slot-track", type=int, default=None,
                        help="slot to track in the temporal-consistency figure "
                             "(default: auto-pick non-background = 2nd-largest mask mass)")
    parser.add_argument("--tag", type=str, default="default",
                        help="filename prefix for the three output PNGs")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building SAViEncoder from {cfg['encoder']['checkpoint_path']}")
    encoder = SAViEncoder(
        checkpoint_path=cfg["encoder"]["checkpoint_path"],
        input_key=cfg["encoder"]["input_key"],
        output_slots_key=cfg["encoder"]["output_slots_key"],
        slot_masks_key=cfg["encoder"]["slot_masks_key"],
        map_location="cpu",
    )

    transform = build_frame_transform(cfg["data"])
    n_frames = int(cfg["data"]["clip_len"])
    print(f"Loading clip: {args.video} "
          f"(start={args.start_frame}, stride={args.stride}, "
          f"n={n_frames})")
    clip = load_single_clip(
        args.video, transform,
        n_frames=n_frames,
        stride=args.stride,
        start_frame=args.start_frame,
    )                                                         # (T, 3, H, W)
    print(f"  clip: shape={tuple(clip.shape)} "
          f"range=[{clip.min():+.3f}, {clip.max():+.3f}]")

    print(f"Forward pass (torch.manual_seed({SEED}))...")
    torch.manual_seed(SEED)
    out = encoder(clip.unsqueeze(0))                         # adds batch dim
    masks = out["slot_masks"][0]                             # (T, N, 1, H, W)
    recons = out["metadata"]["post_recons"][0]               # (T, N, 3, H, W)
    frames = out["metadata"]["img"][0]                       # (T, 3, H, W)

    if args.slot_track is None:
        slot_track = autopick_non_background_slot(masks)
        print(f"  auto-picked slot {slot_track} "
              f"(2nd-largest mask mass; largest is presumed background)")
    else:
        slot_track = args.slot_track

    out_paths = {
        "masks":  args.output_dir / f"{args.tag}_slot_masks.png",
        "recons": args.output_dir / f"{args.tag}_slot_reconstructions.png",
        "track":  args.output_dir / f"{args.tag}_temporal_consistency_slot{slot_track}.png",
    }

    print("Rendering figures...")
    visualize_slot_masks(
        frames, masks,
        save_path=out_paths["masks"],
        title=f"{args.tag}  start={args.start_frame}  stride={args.stride}",
    )
    visualize_slot_reconstructions(
        frames, recons, masks,
        save_path=out_paths["recons"],
    )
    visualize_temporal_consistency(
        masks, slot_idx=slot_track,
        save_path=out_paths["track"],
    )

    print("\nWrote:")
    for k, p in out_paths.items():
        print(f"  {k:6s} {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
