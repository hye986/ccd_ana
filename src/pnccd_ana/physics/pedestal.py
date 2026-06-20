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
    Memory-efficient sigma-clip using iterative statistics only.

    Peak memory: ~3 × (Y, X) float32 arrays = 3 × 2 MB = 6 MB
    The keep_mask is computed on the fly during the final pass.
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Sigma-clip offsets  (n_sigma={n_sigma}, memory-efficient) …")

    n_frames, Y, X = data.shape
    d = data.astype(np.float32)

    # ── Iterative statistics using Welford-style incremental update ───────────
    # First pass: compute initial mean and variance
    mu  = d.mean(axis=0)                    # (Y, X) float32 — 2 MB
    std = d.std(axis=0)                     # (Y, X) float32 — 2 MB

    for it in range(max_iter):
        upper    = mu + n_sigma * std       # (Y, X)
        # Per-pixel count of surviving frames
        survive  = (d <= upper[np.newaxis]) # (N, Y, X) bool — 1 GB peak, freed immediately
        count    = survive.sum(axis=0).astype(np.float32)          # (Y, X)
        count    = np.maximum(count, 1)

        mu_new   = (d * survive).sum(axis=0) / count               # (Y, X)

        # variance of surviving frames
        diff     = (d - mu_new[np.newaxis]) * survive
        var_new  = (diff * diff).sum(axis=0) / np.maximum(count - 1, 1)
        std_new  = np.sqrt(var_new)

        n_changed = int(((mu_new - mu) ** 2 > 1e-6).sum())
        print(f"    iter {it+1}: {int((~survive).sum()):,} clipped  "
              f"({n_changed} pixels changed μ)")
        del survive, diff   # free 1 GB immediately

        mu  = mu_new
        std = std_new

        if n_changed == 0:
            break

    # ── Final pass: build keep_mask and count clipped frames ─────────────────
    upper         = mu + n_sigma * std
    keep_mask     = d <= upper[np.newaxis]                          # (N, Y, X) bool
    n_clipped_map = (n_frames - keep_mask.sum(axis=0)).astype(np.float32)

    avg_c = float(n_clipped_map.mean())
    max_c = int(n_clipped_map.max())
    print(f"  {tag}Clip summary: avg {avg_c:.2f} frames/pixel  "
          f"({avg_c/n_frames*100:.2f}%),  max {max_c}")

    # Final offset = clipped mean
    count  = np.maximum(keep_mask.sum(axis=0), 1).astype(np.float32)
    offset = (d * keep_mask).sum(axis=0) / count

    return offset.astype(np.float32), keep_mask, n_clipped_map


