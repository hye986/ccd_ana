"""
pnccd_ana.lib.noise
===================
Per-pixel electronic noise estimation from CM-corrected dark frames.

When a sigma-clip keep_mask is supplied, only the un-clipped (clean)
frames are included in the std calculation, preventing signal-hit frames
from inflating the noise estimate.
"""

from __future__ import annotations

import numpy as np


def compute_noise(
        corrected:  np.ndarray,
        keep_mask:  np.ndarray | None = None,
        label:      str               = "",
) -> np.ndarray:
    """
    Pixel-wise RMS across CM-corrected dark frames.

    Parameters
    ----------
    corrected : float32 (n_frames, Y, X)
    keep_mask : bool    (n_frames, Y, X) or None
                True where a frame should be included.
                Pass the mask from compute_offset_sigma_clip to exclude
                signal-contaminated frames from the noise estimate.
                None → use all frames (median-only mode).

    Returns
    -------
    noise : float32 (Y, X)  [ADU RMS]
    """
    tag = f"[{label}] " if label else ""
    n_frames = corrected.shape[0]

    if keep_mask is None:
        print(f"  {tag}Pixel noise (all {n_frames} frames) …")
        return np.std(corrected, axis=0, ddof=1).astype(np.float32)

    print(f"  {tag}Pixel noise (sigma-clip mask applied) …")
    d     = corrected.astype(np.float64)
    m     = keep_mask.astype(np.float64)
    count = np.maximum(m.sum(axis=0), 2)
    mu    = np.sum(d * m, axis=0) / count
    var   = np.sum((d - mu[np.newaxis]) ** 2 * m, axis=0) / (count - 1)

    avg_kept = float(m.sum(axis=0).mean())
    print(f"  {tag}  avg {avg_kept:.1f}/{n_frames} frames kept per pixel "
          f"({avg_kept/n_frames*100:.1f}%)")
    return np.sqrt(var).astype(np.float32)
