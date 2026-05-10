"""
Smoke-test CLEVRER indexing + dataset without loading SAVi.

Run from repo root (so `import src` works):

  $env:PYTHONPATH="."
  python scripts/test_dataloader.py --config configs/base.yaml

Requires a non-empty jsonl index (run prepare_clevrer_index.py first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clevrer_dataset import CLEVRERVideoDataset, collate_video_batch
from src.data.transforms import build_frame_transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CLEVRER data loading (no encoder).")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--index-file", type=str, default=None, help="Override data.index_file")
    parser.add_argument("--num-batches", type=int, default=1, help="How many batches to load")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    index_file = args.index_file or data_cfg["index_file"]
    index_path = Path(index_file)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Index not found: {index_path.resolve()}\n"
            "Build it first: python scripts/prepare_clevrer_index.py --config configs/base.yaml --with-frame-count\n"
            "(after setting data.video_root in the config.)"
        )

    transform = build_frame_transform(data_cfg)
    dataset = CLEVRERVideoDataset(
        index_file=str(index_path),
        clip_len=int(data_cfg["clip_len"]),
        transform=transform,
    )
    print(f"Dataset size: {len(dataset)} videos (from {index_path})")

    loader = DataLoader(
        dataset,
        batch_size=min(2, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_video_batch,
    )

    n = 0
    for batch in loader:
        frames = batch["frames"]
        idx = batch["frame_indices"]
        print(f"batch {n}: video_id={batch['video_id']}")
        print(f"  frames shape: {tuple(frames.shape)} dtype={frames.dtype}")
        print(f"  frame_indices shape: {tuple(idx.shape)}")
        print(f"  frames finite: {torch.isfinite(frames).all().item()}")
        n += 1
        if n >= args.num_batches:
            break

    if n == 0:
        raise RuntimeError("DataLoader produced no batches (empty dataset?)")

    print("OK: data loading works.")


if __name__ == "__main__":
    main()
