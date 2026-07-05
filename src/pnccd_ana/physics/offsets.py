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


# ══════════════════════════════════════════════════════════════════════════════
# Enhanced bad pixel detection (matches ROOT HStepBrightPixelMap2, 
# HStepEmptyPixelMap, HStepHotPixelMap)
# ══════════════════════════════════════════════════════════════════════════════

def find_bright_pixels(
        corrected_frames: np.ndarray,
        noise_map: np.ndarray,
        threshold_sigma: float = 5.0,
        min_fraction: float = 0.5,
        label: str = "",
) -> np.ndarray:
    """
    Find pixels with consistently high signal above noise (bright pixels).
    
    Matches ROOT HStepBrightPixelMap2.
    
    A pixel is flagged as bright if its mean signal is consistently
    above threshold_sigma * noise in at least min_fraction of frames.
    
    Parameters
    ----------
    corrected_frames : (n_frames, n_rows, n_cols) corrected dark frames
    noise_map        : (n_rows, n_cols) noise map
    threshold_sigma  : number of sigma above noise to flag as bright
    min_fraction     : minimum fraction of frames that must exceed threshold
    label            : label for logging
    
    Returns
    -------
    bright_mask : (n_rows, n_cols) bool, True = bright pixel
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Finding bright pixels (threshold={threshold_sigma}σ, "
          f"min_fraction={min_fraction}) …")
    
    n_frames, n_rows, n_cols = corrected_frames.shape
    
    # Threshold for each pixel
    threshold = noise_map * threshold_sigma
    
    # Count frames where each pixel exceeds threshold
    exceed_count = (corrected_frames > threshold[np.newaxis]).sum(axis=0)
    
    # Flag pixels that exceed threshold in enough frames
    bright_mask = exceed_count >= (min_fraction * n_frames)
    
    n_bright = int(bright_mask.sum())
    print(f"  {tag}Bright pixels: {n_bright} / {n_rows * n_cols} "
          f"({100*n_bright/(n_rows*n_cols):.2f}%)")
    
    return bright_mask


def find_empty_pixels(
        corrected_frames: np.ndarray,
        noise_map: np.ndarray,
        threshold_sigma: float = 3.0,
        min_fraction: float = 0.8,
        label: str = "",
) -> np.ndarray:
    """
    Find pixels with consistently low signal (dead/empty pixels).
    
    Matches ROOT HStepEmptyPixelMap.
    
    A pixel is flagged as empty if its signal is consistently
    below threshold_sigma * noise in at least min_fraction of frames.
    
    Parameters
    ----------
    corrected_frames : (n_frames, n_rows, n_cols) corrected dark frames
    noise_map        : (n_rows, n_cols) noise map
    threshold_sigma  : number of sigma below noise to flag as empty
    min_fraction     : minimum fraction of frames that must be below threshold
    label            : label for logging
    
    Returns
    -------
    empty_mask : (n_rows, n_cols) bool, True = empty pixel
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Finding empty pixels (threshold={threshold_sigma}σ, "
          f"min_fraction={min_fraction}) …")
    
    n_frames, n_rows, n_cols = corrected_frames.shape
    
    # Threshold for each pixel (lower bound)
    threshold = -threshold_sigma * noise_map
    
    # Count frames where each pixel is below threshold
    below_count = (corrected_frames < threshold[np.newaxis]).sum(axis=0)
    
    # Flag pixels that are below threshold in enough frames
    empty_mask = below_count >= (min_fraction * n_frames)
    
    n_empty = int(empty_mask.sum())
    print(f"  {tag}Empty pixels: {n_empty} / {n_rows * n_cols} "
          f"({100*n_empty/(n_rows*n_cols):.2f}%)")
    
    return empty_mask


def find_hot_pixels(
        cluster_data: dict,
        hit_count: np.ndarray,
        mean_adu: np.ndarray,
        noise_map: np.ndarray,
        hot_hit_threshold: float = 3.0,
        hot_adu_threshold: float = 5.0,
        label: str = "",
) -> np.ndarray:
    """
    Find pixels with excessive hit rate or ADU in photon data.
    
    Matches ROOT HStepHotPixelMap.
    
    A pixel is flagged as hot if:
    - Hit rate > hot_hit_threshold * median hit rate, OR
    - Mean ADU > hot_adu_threshold * median mean ADU
    
    Parameters
    ----------
    cluster_data     : CSR cluster data dict
    hit_count        : (n_rows, n_cols) hit count per pixel
    mean_adu         : (n_rows, n_cols) mean ADU per pixel
    noise_map        : (n_rows, n_cols) noise map
    hot_hit_threshold : multiple of median hit rate to flag as hot
    hot_adu_threshold : multiple of median mean ADU to flag as hot
    label            : label for logging
    
    Returns
    -------
    hot_mask : (n_rows, n_cols) bool, True = hot pixel
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Finding hot pixels (hit_threshold={hot_hit_threshold}x, "
          f"adu_threshold={hot_adu_threshold}x) …")
    
    n_rows, n_cols = hit_count.shape
    
    # Compute median statistics (excluding zeros)
    valid_hits = hit_count[hit_count > 0]
    valid_adu = mean_adu[mean_adu > 0]
    
    if len(valid_hits) == 0 or len(valid_adu) == 0:
        print(f"  {tag}Warning: No valid pixels for hot pixel detection")
        return np.zeros((n_rows, n_cols), dtype=bool)
    
    median_hit_rate = np.median(valid_hits)
    median_mean_adu = np.median(valid_adu)
    
    # Flag pixels with excessive hit rate
    hot_by_hits = hit_count > (hot_hit_threshold * median_hit_rate)
    
    # Flag pixels with excessive mean ADU
    hot_by_adu = mean_adu > (hot_adu_threshold * median_mean_adu)
    
    # Combine flags
    hot_mask = hot_by_hits | hot_by_adu
    
    n_hot = int(hot_mask.sum())
    print(f"  {tag}Hot pixels: {n_hot} / {n_rows * n_cols} "
          f"({100*n_hot/(n_rows*n_cols):.2f}%)")
    print(f"  {tag}  Median hit rate: {median_hit_rate:.2f} events/pixel")
    print(f"  {tag}  Median mean ADU: {median_mean_adu:.2f} ADU")
    
    return hot_mask


def build_comprehensive_bad_pixel_mask(
        noise_map: np.ndarray,
        n_clipped_map: np.ndarray | None = None,
        n_dark_frames: int = 0,
        corrected_frames: np.ndarray | None = None,
        cluster_data: dict | None = None,
        hit_count: np.ndarray | None = None,
        mean_adu: np.ndarray | None = None,
        active_mask: np.ndarray | None = None,
        hot_rms_multiple: float = 5.0,
        cold_rms_fraction: float = 0.1,
        max_clip_fraction: float = 0.5,
        bright_threshold_sigma: float = 5.0,
        bright_min_fraction: float = 0.5,
        empty_threshold_sigma: float = 3.0,
        empty_min_fraction: float = 0.8,
        hot_hit_threshold: float = 3.0,
        hot_adu_threshold: float = 5.0,
        label: str = "",
) -> np.ndarray:
    """
    Build comprehensive bad pixel mask combining all detection methods.
    
    Combines:
    - Noisy pixels (high noise from dark frames)
    - Cold pixels (low noise from dark frames)
    - Over-clipped pixels (excessive signal hits in dark frames)
    - Bright pixels (consistently high signal in dark frames)
    - Empty pixels (consistently low signal in dark frames)
    - Hot pixels (excessive hit rate or ADU in photon data)
    
    Parameters
    ----------
    noise_map        : (n_rows, n_cols) noise map
    n_clipped_map    : (n_rows, n_cols) number of clipped frames per pixel
    n_dark_frames    : number of dark frames
    corrected_frames : (n_frames, n_rows, n_cols) corrected dark frames (optional)
    cluster_data     : CSR cluster data dict (optional)
    hit_count        : (n_rows, n_cols) hit count per pixel (optional)
    mean_adu         : (n_rows, n_cols) mean ADU per pixel (optional)
    active_mask      : (n_rows, n_cols) bool mask for active pixels
    hot_rms_multiple : multiple of median noise for hot pixels
    cold_rms_fraction: fraction of median noise for cold pixels
    max_clip_fraction: maximum fraction of frames that can be clipped
    bright_threshold_sigma : sigma threshold for bright pixels
    bright_min_fraction    : minimum fraction of frames for bright pixels
    empty_threshold_sigma  : sigma threshold for empty pixels
    empty_min_fraction     : minimum fraction of frames for empty pixels
    hot_hit_threshold      : multiple of median hit rate for hot pixels
    hot_adu_threshold      : multiple of median mean ADU for hot pixels
    label            : label for logging
    
    Returns
    -------
    bad_mask : (n_rows, n_cols) bool, True = bad pixel
    """
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Building comprehensive bad pixel mask …")
    
    n_rows, n_cols = noise_map.shape
    
    # Initialize mask
    bad_mask = np.zeros((n_rows, n_cols), dtype=bool)
    
    # Apply active mask if provided
    if active_mask is not None:
        bad_mask[~active_mask] = True
    
    # 1. Noisy pixels (high noise)
    med_noise = np.median(noise_map[active_mask]) if active_mask is not None else np.median(noise_map)
    noisy_mask = noise_map > (hot_rms_multiple * med_noise)
    bad_mask |= noisy_mask
    print(f"  {tag}Noisy pixels: {int(noisy_mask.sum())}")
    
    # 2. Cold pixels (low noise)
    cold_mask = noise_map < (cold_rms_fraction * med_noise)
    bad_mask |= cold_mask
    print(f"  {tag}Cold pixels: {int(cold_mask.sum())}")
    
    # 3. Over-clipped pixels
    if n_clipped_map is not None and n_dark_frames > 0:
        clip_fraction = n_clipped_map / n_dark_frames
        overclipped_mask = clip_fraction > max_clip_fraction
        bad_mask |= overclipped_mask
        print(f"  {tag}Over-clipped pixels: {int(overclipped_mask.sum())}")
    
    # 4. Bright pixels (requires corrected frames)
    if corrected_frames is not None:
        bright_mask = find_bright_pixels(
            corrected_frames, noise_map,
            threshold_sigma=bright_threshold_sigma,
            min_fraction=bright_min_fraction,
            label=label,
        )
        bad_mask |= bright_mask
    
    # 5. Empty pixels (requires corrected frames)
    if corrected_frames is not None:
        empty_mask = find_empty_pixels(
            corrected_frames, noise_map,
            threshold_sigma=empty_threshold_sigma,
            min_fraction=empty_min_fraction,
            label=label,
        )
        bad_mask |= empty_mask
    
    # 6. Hot pixels (requires photon data)
    if hit_count is not None and mean_adu is not None:
        hot_mask = find_hot_pixels(
            cluster_data, hit_count, mean_adu, noise_map,
            hot_hit_threshold=hot_hit_threshold,
            hot_adu_threshold=hot_adu_threshold,
            label=label,
        )
        bad_mask |= hot_mask
    
    total_bad = int(bad_mask.sum())
    print(f"\n{tag}Total bad pixels: {total_bad} / {n_rows * n_cols} "
          f"({100*total_bad/(n_rows*n_cols):.2f}%)")
    
    return bad_mask
