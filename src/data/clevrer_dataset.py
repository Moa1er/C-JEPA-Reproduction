import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import torch
from torch.utils.data import Dataset

from .temporal_sampling import sample_uniform_indices


class CLEVRERVideoDataset(Dataset):
    """
    Returns one video clip item:
      {
        "video_id": str,
        "frames": Tensor[T, C, H, W],
        "frame_indices": Tensor[T],
      }
    """

    def __init__(self, index_file: str, clip_len: int, transform):
        self.index_file = Path(index_file)
        self.clip_len = int(clip_len)
        self.transform = transform
        self.records = self._load_index(self.index_file)

    @staticmethod
    def _load_index(index_file: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with index_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        if not records:
            raise ValueError(f"No records found in index file: {index_file}")
        return records

    @staticmethod
    def _read_selected_frames(video_path: str, frame_indices: List[int]) -> List[Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frames.append(frame)

        cap.release()
        return frames

    @staticmethod
    def _get_num_frames(video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if num_frames <= 0:
            raise RuntimeError(f"Invalid frame count for video: {video_path}")
        return num_frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        video_path = rec["video_path"]
        video_id = rec.get("video_id", Path(video_path).stem)
        num_frames = int(rec.get("num_frames", 0)) or self._get_num_frames(video_path)

        frame_indices = sample_uniform_indices(num_frames=num_frames, clip_len=self.clip_len)
        raw_frames = self._read_selected_frames(video_path=video_path, frame_indices=frame_indices)
        frames = [self.transform(f) for f in raw_frames]

        return {
            "video_id": video_id,
            "frames": torch.stack(frames, dim=0),  # [T, C, H, W]
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
        }


def collate_video_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "video_id": [x["video_id"] for x in batch],
        "frames": torch.stack([x["frames"] for x in batch], dim=0),  # [B, T, C, H, W]
        "frame_indices": torch.stack([x["frame_indices"] for x in batch], dim=0),  # [B, T]
    }
