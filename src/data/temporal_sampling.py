from typing import List


def sample_uniform_indices(num_frames: int, clip_len: int) -> List[int]:
    """Uniformly sample clip_len indices from [0, num_frames-1]."""
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if clip_len <= 0:
        raise ValueError("clip_len must be > 0")

    if num_frames == 1:
        return [0] * clip_len

    if num_frames >= clip_len:
        step = (num_frames - 1) / float(clip_len - 1) if clip_len > 1 else 0.0
        return [int(round(i * step)) for i in range(clip_len)]

    # Repeat boundary frames deterministically when video is short.
    step = (num_frames - 1) / float(clip_len - 1) if clip_len > 1 else 0.0
    return [min(int(round(i * step)), num_frames - 1) for i in range(clip_len)]
