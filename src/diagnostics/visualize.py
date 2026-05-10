"""Visualization utilities for SAVi slot attention outputs.

Functions accept torch tensors on CPU and return matplotlib Figures.
Tensors are auto-detached and converted to numpy internally. Inputs may
optionally include a leading batch dim of size 1, which is squeezed.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colormaps


# --- helpers --------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_squeeze_batch(x, expected_ndim, name):
    if x.ndim == expected_ndim + 1:
        if x.shape[0] != 1:
            raise ValueError(
                f'{name}: expected leading batch dim of size 1, '
                f'got shape={x.shape}')
        x = x[0]
    if x.ndim != expected_ndim:
        raise ValueError(
            f'{name}: expected ndim={expected_ndim} '
            f'(optionally with a B=1 prefix), got shape={x.shape}')
    return x


def _to_unit_range(x):
    """Map images to [0, 1]. Detects [-1, 1] inputs by checking the min."""
    x = np.asarray(x, dtype=np.float32)
    if x.size and x.min() < -0.1:
        x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def _chw_to_hwc(frame_chw):
    return np.transpose(frame_chw, (1, 2, 0))


def _slot_colors(n):
    cmap = colormaps.get_cmap('tab10')
    return np.array([cmap(i % 10)[:3] for i in range(n)], dtype=np.float32)


def _row_label(ax, text):
    """Print a small row label to the left of an axes that has axis('off')."""
    ax.text(-0.08, 0.5, text, transform=ax.transAxes,
            ha='right', va='center', fontsize=8)


def _save(fig, save_path):
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches='tight')


# --- public API -----------------------------------------------------------

def visualize_slot_masks(frames, masks, save_path=None, title=None):
    """Per-timestep, show the original frame plus each slot's mask overlay.

    Layout: T rows x (N+1) cols. Column 0 is the bare frame; columns 1..N
    overlay slot s's mask on the frame in slot-s's color (~50% alpha).

    Args:
        frames: (T, 3, H, W) or (1, T, 3, H, W). Values in [0, 1] or [-1, 1].
        masks:  (T, N, 1, H, W) or (1, T, N, 1, H, W). Per-slot alpha masks.
        save_path: optional PNG path.
        title: optional figure suptitle.

    Returns:
        matplotlib.figure.Figure
    """
    frames = _maybe_squeeze_batch(_to_numpy(frames), 4, 'frames')
    masks = _maybe_squeeze_batch(_to_numpy(masks), 5, 'masks')

    T, C, H, W = frames.shape
    Tm, N, Mc, Hm, Wm = masks.shape
    if (T, H, W) != (Tm, Hm, Wm) or C != 3 or Mc != 1:
        raise ValueError(
            f'shape mismatch: frames {frames.shape}, masks {masks.shape}')

    frames_01 = _to_unit_range(frames)
    masks_01 = np.clip(masks.astype(np.float32), 0.0, 1.0)
    colors = _slot_colors(N)

    fig, axes = plt.subplots(
        T, N + 1,
        figsize=((N + 1) * 1.6, T * 1.6),
        dpi=100, squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=12)

    alpha = 0.5
    for t in range(T):
        frame_hwc = _chw_to_hwc(frames_01[t])

        ax = axes[t, 0]
        ax.imshow(frame_hwc)
        ax.axis('off')
        if t == 0:
            ax.set_title('frame', fontsize=9)
        _row_label(ax, f't={t}')

        for s in range(N):
            ax = axes[t, s + 1]
            m = masks_01[t, s, 0][..., None]                  # (H, W, 1)
            color = colors[s].reshape(1, 1, 3)
            blended = frame_hwc * (1.0 - alpha * m) + color * (alpha * m)
            ax.imshow(np.clip(blended, 0.0, 1.0))
            ax.axis('off')
            if t == 0:
                ax.set_title(f'slot {s}', fontsize=9,
                             color=tuple(colors[s]), fontweight='bold')

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def visualize_slot_reconstructions(frames, recons, masks, save_path=None):
    """Per-timestep, show GT / combined recon / each slot's masked recon.

    Layout: T rows x (N+2) cols.
        col 0     : ground-truth frame
        col 1     : combined recon = sum_s recons[s] * masks[s]
        col 2..N+1: each slot's masked reconstruction (recons[s] * masks[s])

    Args:
        frames: (T, 3, H, W) or (1, T, 3, H, W).
        recons: (T, N, 3, H, W) or (1, T, N, 3, H, W).
        masks:  (T, N, 1, H, W) or (1, T, N, 1, H, W).
        save_path: optional PNG path.

    Returns:
        matplotlib.figure.Figure
    """
    frames = _maybe_squeeze_batch(_to_numpy(frames), 4, 'frames')
    recons = _maybe_squeeze_batch(_to_numpy(recons), 5, 'recons')
    masks = _maybe_squeeze_batch(_to_numpy(masks), 5, 'masks')

    T, _, H, W = frames.shape
    Tr, N, Cr, Hr, Wr = recons.shape
    Tm, Nm, Mc, Hm, Wm = masks.shape
    if (T, H, W) != (Tr, Hr, Wr) or (T, H, W) != (Tm, Hm, Wm) \
            or N != Nm or Cr != 3 or Mc != 1:
        raise ValueError(
            f'shape mismatch: frames {frames.shape}, recons {recons.shape}, '
            f'masks {masks.shape}')

    frames_01 = _to_unit_range(frames)
    recons_01 = _to_unit_range(recons)
    masks_01 = np.clip(masks.astype(np.float32), 0.0, 1.0)

    # combined recon = sum_s recon_s * mask_s, computed in unit range
    combined = (recons_01 * masks_01).sum(axis=1)             # (T, 3, H, W)
    combined = np.clip(combined, 0.0, 1.0)

    # per-slot masked recon (slot's "explanation" of its claimed pixels)
    masked_recons = recons_01 * masks_01                      # (T, N, 3, H, W)

    cols = N + 2
    fig, axes = plt.subplots(
        T, cols,
        figsize=(cols * 1.6, T * 1.6),
        dpi=100, squeeze=False,
    )

    for t in range(T):
        ax = axes[t, 0]
        ax.imshow(_chw_to_hwc(frames_01[t]))
        ax.axis('off')
        if t == 0:
            ax.set_title('GT', fontsize=9)
        _row_label(ax, f't={t}')

        ax = axes[t, 1]
        ax.imshow(_chw_to_hwc(combined[t]))
        ax.axis('off')
        if t == 0:
            ax.set_title('recon', fontsize=9)

        for s in range(N):
            ax = axes[t, s + 2]
            ax.imshow(_chw_to_hwc(np.clip(masked_recons[t, s], 0.0, 1.0)))
            ax.axis('off')
            if t == 0:
                ax.set_title(f'slot {s}', fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def visualize_temporal_consistency(masks, slot_idx, save_path=None):
    """Show a single slot's mask across all T timesteps in one horizontal strip.

    Useful for verifying that a slot tracks the same object over time.

    Args:
        masks: (T, N, 1, H, W) or (1, T, N, 1, H, W).
        slot_idx: which slot to track.
        save_path: optional PNG path.

    Returns:
        matplotlib.figure.Figure
    """
    masks = _maybe_squeeze_batch(_to_numpy(masks), 5, 'masks')
    T, N, _, H, W = masks.shape
    if not (0 <= slot_idx < N):
        raise IndexError(f'slot_idx={slot_idx} out of range for N={N}')

    slot_masks = np.clip(masks[:, slot_idx, 0].astype(np.float32),
                         0.0, 1.0)                            # (T, H, W)

    fig, axes = plt.subplots(
        1, T,
        figsize=(T * 1.6, 2.2),
        dpi=100, squeeze=False,
    )
    fig.suptitle(f'slot {slot_idx} over time', fontsize=10)

    for t in range(T):
        ax = axes[0, t]
        ax.imshow(slot_masks[t], cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_title(f't={t}', fontsize=9)
        ax.axis('off')

    fig.tight_layout()
    _save(fig, save_path)
    return fig
