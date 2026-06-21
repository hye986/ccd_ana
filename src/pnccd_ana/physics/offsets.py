"""
pnccd_ana.physics.offsets
=========================
Per-pixel offset estimation from dark frames.

Two methods are provided:
  - Median     : robust to rare signal hits (< 50 % occupancy per pixel)
  - Sigma-clip : iterative upper-tail clipping, returns the keep-mask for
                 downstream noise estimation

Overflow/underflow handling
---------------------------
Raw ADC sentinel values (0 = underflow, 65535 = overflow for 16-bit) are
excluded from all statistics, matching ROOT HStepOffNoiMapHLL which flags
OverflowPixel and UnderflowPixel per frame per pixel before any median or
mean computation.
"""

from __future__ import annotations

import numpy as np

# 16-bit ADC sentinels (ROOT defaults)
_UNDERFLOW_VALUE: int = 0
_OVERFLOW_VALUE:  int = 65535


def _make_valid_mask(data: np.ndarray,
                     n_bits: int = 16) -> np.ndarray:
    """
    Return bool mask (n_frames, Y, X) True where pixel is NOT a sentinel.

    Works on integer or float arrays.  For float arrays the sentinel check
    uses the same integer boundary values cast to float.
    """
    uv = float(_UNDERFLOW_VALUE)
    ov = float((1 << n_bits) - 1)
    return (data != uv) & (data != ov)


def compute_offset_median(data: np.ndarray,
                          label: str = "",
                          n_bits: int = 16) -> np.ndarray:
    """
    Median across frames for each pixel, excluding overflow/underflow.

    Robust to Fe-55 signal hits as long as hit rate < 50 % per pixel.
    Overflow/underflow sentinels are replaced with NaN so np.nanmedian
    skips them, matching ROOT which excludes them before TMath::Median.

    Parameters
    ----------
    data   : uint16 or float32 (n_frames, Y, X)
    n_bits : ADC bit depth (default 16)

    Returns
    -------
    offset : float32 (Y, X)
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Computing median offsets …")

    d = data.astype(np.float32)

    # Mask sentinels → NaN so nanmedian ignores them
    valid = _make_valid_mask(data, n_bits=n_bits)
    d[~valid] = np.nan

    return np.nanmedian(d, axis=0).astype(np.float32)


def compute_offset_sigma_clip(
        data:     np.ndarray,
        n_sigma:  float = 3.0,
        max_iter: int   = 5,
        label:    str   = "",
        n_bits:   int   = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient sigma-clip using iterative statistics only.

    Overflow/underflow sentinels are excluded from all iterations and from
    the final mean, matching ROOT HStepOffNoiMapHLL which skips
    OverflowPixel and UnderflowPixel in every accumulation loop.

    Peak memory: ~3 × (Y, X) float32 + (n_frames, Y, X) bool keep_mask.

    Parameters
    ----------
    data    : uint16 or float32 (n_frames, Y, X)
    n_sigma : upper-tail clip threshold
    max_iter: maximum iterations
    n_bits  : ADC bit depth for sentinel detection

    Returns
    -------
    offset        : float32 (Y, X)
    keep_mask     : bool    (n_frames, Y, X) — True = frame used for this pixel
    n_clipped_map : float32 (Y, X) — frames clipped per pixel
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Sigma-clip offsets  (n_sigma={n_sigma}, memory-efficient) …")

    n_frames, Y, X = data.shape
    d = data.astype(np.float32)

    # Build sentinel mask (True = valid, not a sentinel)
    sentinel_valid = _make_valid_mask(data, n_bits=n_bits)   # (N, Y, X) bool

    # Replace sentinels with NaN so they are ignored in all statistics
    d_masked = d.copy()
    d_masked[~sentinel_valid] = np.nan

    # ── Iterative statistics ──────────────────────────────────────────────────
    # Use nanmean / nanstd so sentinels (NaN) are automatically excluded
    mu  = np.nanmean(d_masked, axis=0)    # (Y, X)
    std = np.nanstd( d_masked, axis=0)    # (Y, X)  ddof=0 for speed here

    active = sentinel_valid.copy()        # (N, Y, X) — frames in use

    for it in range(max_iter):
        upper   = mu + n_sigma * std      # (Y, X)
        # Clip: keep frame if value <= upper AND not a sentinel
        survive = active & (d <= upper[np.newaxis])   # (N, Y, X)

        count   = np.maximum(survive.sum(axis=0).astype(np.float32), 1)
        mu_new  = np.where(
            count > 0,
            (d * survive).sum(axis=0) / count,
            mu,
        )

        diff    = (d - mu_new[np.newaxis]) * survive
        var_new = (diff * diff).sum(axis=0) / np.maximum(count - 1, 1)
        std_new = np.sqrt(var_new)

        n_changed = int(((mu_new - mu) ** 2 > 1e-6).sum())
        n_clipped_now = int((~survive & sentinel_valid).sum())
        print(f"    iter {it+1}: {n_clipped_now:,} clipped  "
              f"({n_changed} pixels changed μ)")

        del diff
        active = survive
        mu     = mu_new
        std    = std_new

        if n_changed == 0:
            break

    # ── Final keep_mask and n_clipped_map ────────────────────────────────────
    # keep_mask is True where frame is used (not clipped, not sentinel)
    keep_mask     = active                              # (N, Y, X) bool
    # n_clipped = sentinel-valid frames that were clipped (not sentinel frames)
    n_clipped_map = (sentinel_valid & ~keep_mask).sum(axis=0).astype(np.float32)

    avg_c = float(n_clipped_map.mean())
    max_c = int(n_clipped_map.max())
    print(f"  {tag}Clip summary: avg {avg_c:.2f} frames/pixel  "
          f"({avg_c/n_frames*100:.2f}%),  max {max_c}")

    # Final offset = mean over kept frames only
    count  = np.maximum(keep_mask.sum(axis=0), 1).astype(np.float32)
    offset = (d * keep_mask).sum(axis=0) / count

    return offset.astype(np.float32), keep_mask, n_clipped_map
