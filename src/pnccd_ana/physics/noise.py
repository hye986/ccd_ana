"""
pnccd_ana.physics.noise
===================
Per-pixel electronic noise estimation from CM-corrected dark frames.

When a sigma-clip keep_mask is supplied, only the un-clipped (clean)
frames are included in the std calculation, preventing signal-hit frames
from inflating the noise estimate.

This module also provides ``build_bad_pixel_mask`` for flagging pixels
that should be excluded from photon-event recognition (hot, cold/stuck,
or heavily clipped during sigma-clip pedestal estimation).
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


# ──────────────────────────────────────────────────────────────────────────────
# Bad-pixel mask
# ──────────────────────────────────────────────────────────────────────────────

def build_bad_pixel_mask(
        noise_map:        np.ndarray,
        n_clipped_map:    np.ndarray | None = None,
        n_dark_frames:    int               = 0,
        hot_rms_multiple: float             = 5.0,
        cold_rms_fraction: float            = 0.1,
        max_clip_fraction: float            = 0.5,
        active_mask:      np.ndarray | None = None,
        label:            str               = "",
) -> np.ndarray:
    """
    Flag detector pixels that should be excluded from event recognition.

    A pixel is marked *bad* when any of the following holds:
      • ``noise > hot_rms_multiple × median(noise)``  (HOT — fires false seeds)
      • ``noise < cold_rms_fraction × median(noise)`` (COLD / stuck-at-value)
      • ``noise`` is NaN, Inf, or non-finite                (UNUSABLE)
      • ``n_clipped_map / n_dark_frames > max_clip_fraction`` (UNSTABLE — too
        many dark frames had to be clipped at this pixel)

    The median used to set the hot/cold thresholds is taken **inside the
    active region** (``active_mask``) so that geometric padding outside the
    ASICs doesn't poison the reference scale.

    Parameters
    ----------
    noise_map         : float (Y, X) — per-pixel RMS from dark calibration.
    n_clipped_map     : float (Y, X) or None — frames clipped per pixel
                        during sigma-clip pedestal (from
                        ``compute_offset_sigma_clip``).
    n_dark_frames     : int — total number of dark frames the noise was
                        estimated from.  Required if n_clipped_map is given.
    hot_rms_multiple  : pixels with noise above this multiple of the median
                        active-pixel noise are flagged HOT.
    cold_rms_fraction : pixels with noise below this fraction of the median
                        active-pixel noise are flagged COLD (includes
                        stuck-at-value pixels with rms ≈ 0).
    max_clip_fraction : pixels whose dark-frame clip rate exceeds this are
                        flagged UNSTABLE.  Ignored if n_clipped_map is None.
    active_mask       : bool (Y, X) or None — True where pixels physically
                        belong to a read-out ASIC.  Pixels outside are
                        always marked bad and excluded from the median used
                        to set hot/cold thresholds.  None → use whole frame.
    label             : tag prepended to diagnostic prints.

    Returns
    -------
    bad_pixel_mask : bool (Y, X) — True where the pixel is bad.
    """
    tag = f"[{label}] " if label else ""
    if noise_map.ndim != 2:
        raise ValueError(f"noise_map must be 2-D, got shape {noise_map.shape}")

    Y, X = noise_map.shape

    # Pixels outside the active region are bad by definition.
    if active_mask is None:
        active = np.ones((Y, X), dtype=bool)
    else:
        if active_mask.shape != (Y, X):
            raise ValueError(
                f"active_mask shape {active_mask.shape} != noise_map shape {(Y, X)}")
        active = active_mask.astype(bool)

    bad = ~active                                  # start: everything outside ASICs

    # Always flag non-finite or non-positive noise as bad.
    nonfinite = ~np.isfinite(noise_map)
    nonpositive = noise_map <= 0
    bad |= nonfinite
    bad |= nonpositive & active                    # active+0-noise = stuck

    # Reference noise scale from active, finite, positive pixels only.
    ref = noise_map[active & np.isfinite(noise_map) & (noise_map > 0)]
    if ref.size == 0:
        raise RuntimeError("No usable pixels to compute reference noise median.")
    med = float(np.median(ref))

    hot_thr  = hot_rms_multiple  * med
    cold_thr = cold_rms_fraction * med
    hot      = active & (noise_map > hot_thr)
    cold     = active & np.isfinite(noise_map) & (noise_map < cold_thr)
    bad |= hot
    bad |= cold

    n_hot, n_cold, n_clip = int(hot.sum()), int(cold.sum()), 0
    n_nonfinite = int((nonfinite & active).sum())

    # Clipping-rate criterion (optional).
    if n_clipped_map is not None and max_clip_fraction is not None and n_dark_frames > 0:
        if n_clipped_map.shape != (Y, X):
            raise ValueError(
                f"n_clipped_map shape {n_clipped_map.shape} != noise_map shape {(Y, X)}")
        clip_frac = n_clipped_map.astype(np.float32) / float(n_dark_frames)
        unstable  = active & (clip_frac > max_clip_fraction)
        bad |= unstable
        n_clip = int(unstable.sum())

    n_active = int(active.sum())
    n_bad    = int((bad & active).sum())
    print(f"  {tag}Bad-pixel mask: median noise={med:.2f}  "
          f"hot(>{hot_rms_multiple}×med)={n_hot}  "
          f"cold(<{cold_rms_fraction}×med)={n_cold}  "
          f"non-finite={n_nonfinite}  "
          + (f"unstable(clip>{max_clip_fraction:.0%})={n_clip}  "
             if n_clip or n_clipped_map is not None else "")
          + f"total-bad-in-active={n_bad}/{n_active} "
          f"({100*n_bad/max(n_active,1):.2f}%)")
    return bad
