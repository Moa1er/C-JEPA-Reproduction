import argparse
import json
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def get_num_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL index for CLEVRER videos.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--video-root", type=str, default=None, help="Override data.video_root")
    parser.add_argument("--video-glob", type=str, default=None, help="Override data.video_glob")
    parser.add_argument("--output", type=str, default=None, help="Override output jsonl path")
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Override split name stored in each record (e.g. train/val/test)",
    )
    parser.add_argument("--with-frame-count", action="store_true", help="Probe frame counts with OpenCV")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})

    raw_root = args.video_root if args.video_root is not None else data_cfg.get("video_root", "")
    if raw_root is None or str(raw_root).strip() == "":
        raise ValueError(
            "video_root is empty. Set data.video_root in configs/base.yaml to your CLEVRER "
            "video folder, or pass --video-root C:\\path\\to\\videos"
        )
    video_root = Path(raw_root).expanduser().resolve()
    video_glob = args.video_glob or data_cfg.get("video_glob", "**/*.mp4")
    split_name = args.split_name or data_cfg.get("split_name", "train")
    output_path = args.output or data_cfg.get("index_file", "data/train.jsonl")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_root.glob(video_glob))
    if not videos:
        raise ValueError(f"No videos found under {video_root} with glob '{video_glob}'")

    with output.open("w", encoding="utf-8") as f:
        for vp in tqdm(videos, desc="Indexing videos"):
            rec = {
                "video_id": vp.stem,
                "video_path": str(vp.resolve()),
                "split": split_name,
            }
            if args.with_frame_count:
                rec["num_frames"] = get_num_frames(vp)
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(videos)} records to {output}")


if __name__ == "__main__":
    main()
