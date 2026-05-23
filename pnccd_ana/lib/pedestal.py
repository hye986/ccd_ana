"""
pnccd_ana.lib.pedestal
======================
Per-pixel pedestal (offset) estimation from dark frames.

Two methods are provided:
  - Median  : robust to rare signal hits (< 50 % occupancy per pixel)
  - Sigma-clip : iterative upper-tail clipping, returns the keep-mask for
                 downstream noise estimation

ADC rollover detection and correction is also handled here.
"""

from __future__ import annotations

import numpy as np

from .geometry import ADC_MAX, ADC_RANGE


# ──────────────────────────────────────────────────────────────────────────────
# Rollover
# ──────────────────────────────────────────────────────────────────────────────

def detect_and_unwrap_rollover(
        data:       np.ndarray,
        low_frac:   float = 0.10,
        high_frac:  float = 0.80,
        unwrap:     bool  = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect (and optionally correct) 16-bit ADC rollover.

    A pixel-frame is flagged when:
      value < low_frac × ADC_MAX   AND
      pixel median > high_frac × ADC_MAX

    Parameters
    ----------
    data      : float32 (n_frames, Y, X) raw ADU values
    low_frac  : suspicious-low threshold  (default 0.10)
    high_frac : normally-high threshold   (default 0.80)
    unwrap    : add ADC_RANGE to flagged values if True

    Returns
    -------
    out      : corrected copy (or original if nothing to do)
    rollover : bool mask (n_frames, Y, X)
    """
    low_thresh  = low_frac  * ADC_MAX
    high_thresh = high_frac * ADC_MAX

    pixel_median = np.median(data, axis=0)
    high_pixels  = pixel_median > high_thresh
    rollover     = (data < low_thresh) & high_pixels[np.newaxis]

    n_events = int(rollover.sum())
    n_pixels = int(rollover.any(axis=0).sum())
    frac     = n_events / data.size * 100.0

    print()
    print("┌─ ROLLOVER CHECK " + "─" * 50)
    print(f"│  Thresholds : low < {low_frac*100:.0f}%  ({low_thresh:.0f} ADU),  "
          f"typical > {high_frac*100:.0f}%  ({high_thresh:.0f} ADU)")
    print(f"│  Flagged pixel-frame events : {n_events:,}  ({frac:.4f}%)")
    print(f"│  Affected unique pixels     : {n_pixels:,}")
    if n_events > 0:
        print("│  ⚠  ROLLOVER DETECTED — use unwrap=True to correct.")
    else:
        print("│  ✓  No rollover events detected.")
    print("└" + "─" * 67)

    out = data.copy()
    if unwrap and n_events > 0:
        out[rollover] += ADC_RANGE
        print(f"  → Unwrapped {n_events:,} events (+{ADC_RANGE} ADU each).")

    return out, rollover


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
