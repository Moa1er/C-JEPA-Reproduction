import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from src.pipeline.run_slot_extract import run_slot_extract


def load_yaml(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.index_file:
        cfg["data"]["index_file"] = args.index_file
    if args.checkpoint_path:
        cfg["encoder"]["checkpoint_path"] = args.checkpoint_path
    if args.output_dir:
        cfg["run"]["output_dir"] = args.output_dir
    if args.max_videos is not None:
        cfg["run"]["max_videos"] = args.max_videos
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen slot encoder extraction on CLEVRER.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--index-file", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args)
    run_slot_extract(cfg)


if __name__ == "__main__":
    main()
