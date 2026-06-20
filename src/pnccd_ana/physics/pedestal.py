"""
pnccd_ana.physics.pedestal
======================
Per-pixel pedestal (offset) estimation from dark frames.

Two methods are provided:
  - Median  : robust to rare signal hits (< 50 % occupancy per pixel)
  - Sigma-clip : iterative upper-tail clipping, returns the keep-mask for
                 downstream noise estimation
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Pedestal methods
# ──────────────────────────────────────────────────────────────────────────────

def compute_offset_median(data: np.ndarray, label: str = "") -> np.ndarray:
    """
    Median across frames for each pixel.

    Robust to Fe-55 signal hits as long as hit rate < 50 % per pixel.

    Parameters
    ----------
    data : float32 (n_frames, Y, X)

    Returns
    -------
    offset : float32 (Y, X)
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Computing median offsets …")
    return np.median(data, axis=0).astype(np.float32)


def compute_offset_sigma_clip(
        data:     np.ndarray,
        n_sigma:  float = 3.0,
        max_iter: int   = 5,
        label:    str   = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative sigma-clipping pedestal.

    Upper-tail only (signal hits are always positive excursions).
    After convergence, the surviving frame mask is returned so that
    noise estimation can exclude hit-contaminated frames.

    Parameters
    ----------
    data     : float32 (n_frames, Y, X)
    n_sigma  : clipping threshold (default 3.0)
    max_iter : maximum iterations (default 5)

    Returns
    -------
    offset          : float32 (Y, X)   – clipped mean pedestal
    keep_mask       : bool    (n, Y, X) – True where frame survived clipping
    n_clipped_map   : float32 (Y, X)   – frames clipped per pixel
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Sigma-clip offsets  (n_sigma={n_sigma}) …")

    n_frames = data.shape[0]
    d        = data.astype(np.float64)
    mask     = np.ones_like(d, dtype=bool)

    for it in range(max_iter):
        count = np.maximum(mask.sum(axis=0), 1)
        mu    = np.sum(d * mask, axis=0) / count
        var   = np.sum((d - mu[np.newaxis]) ** 2 * mask, axis=0) / np.maximum(count - 1, 1)
        sigma = np.sqrt(var)

        new_mask  = (d <= (mu + n_sigma * sigma)[np.newaxis]) & mask
        n_clipped = int(mask.sum()) - int(new_mask.sum())
        mask = new_mask
        print(f"    iter {it+1}: {n_clipped:,} pixel-frame samples clipped")
        if n_clipped == 0:
            break

    n_clipped_map = (n_frames - mask.sum(axis=0)).astype(np.float32)
    avg_c = float(n_clipped_map.mean())
    max_c = int(n_clipped_map.max())
    print(f"  {tag}Clip summary: avg {avg_c:.2f} frames/pixel  "
          f"({avg_c/n_frames*100:.2f}%),  max {max_c}")

    count  = np.maximum(mask.sum(axis=0), 1)
    offset = (np.sum(d * mask, axis=0) / count).astype(np.float32)
    return offset, mask, n_clipped_map
