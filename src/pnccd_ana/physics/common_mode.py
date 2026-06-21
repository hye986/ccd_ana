"""
pnccd_ana.physics.common_mode
=========================
Common-mode (CM) correction.

Rolling shutter: sensor reads rows from bottom to top.
A detector "row" is a horizontal line of pixels (constant Y, all X) that are
read out simultaneously.

For multi-ASIC detectors, CM is computed per ASIC to avoid cross-ASIC
interference in the correction.

Array layout: data[frame, Y, X]
CM is computed per ASIC: median over ASIC_WIDTH X pixels for each Y row.
Resulting cm_map shape: (n_frames, n_Y, n_asics)

Overflow/underflow handling
---------------------------
Raw ADC sentinel values (underflow=0, overflow=2^n_bits - 1) are excluded
from the CM median, matching ROOT HCommonModeMedian behaviour which sets
bad/excluded pixels to +inf before nth_element so they never enter the
median count.  The same exclusion applies during dark calibration
(apply_common_mode_correction).
"""

from __future__ import annotations

import numpy as np

from ..io.geometry import ASIC_WIDTH

# Default 16-bit ADC sentinel values (matching ROOT HFrameSource defaults)
_ADC_BITS        = 16
_UNDERFLOW_VALUE = np.uint16(0)
_OVERFLOW_VALUE  = np.uint16((1 << _ADC_BITS) - 1)   # 65535


def _masked_median_axis1(data: np.ndarray,
                         bad: np.ndarray | None) -> np.ndarray:
    """
    Compute median along axis=1 (columns) for each row, excluding bad pixels.

    Matches ROOT HCommonModeMedian: bad pixels are set to +inf before the
    partial sort so they never contribute to the median count.

    Parameters
    ----------
    data : float32 (n_Y, n_cols) — one ASIC or full row per call
    bad  : bool   (n_Y, n_cols) or None — True = exclude from median

    Returns
    -------
    medians : float32 (n_Y,)
    """
    if bad is None:
        # Fast path: no exclusions, pure numpy
        return np.median(data, axis=1).astype(np.float32)

    # Slow path: per-row masked median
    # Copy so we do not modify caller's array
    buf = data.astype(np.float32, copy=True)
    buf[bad] = np.inf                           # excluded → +inf (ROOT style)

    n_Y, n_cols = buf.shape
    medians = np.empty(n_Y, dtype=np.float32)

    for row in range(n_Y):
        row_vals = buf[row]                     # view
        finite   = row_vals[np.isfinite(row_vals)]
        if finite.size == 0:
            medians[row] = np.nan
        else:
            medians[row] = np.median(finite)    # numpy median on finite subset

    return medians


def _make_bad_mask(residual: np.ndarray,
                   bad_pixel_mask: np.ndarray | None,
                   overflow_mask:  np.ndarray | None,
                   underflow_mask: np.ndarray | None) -> np.ndarray | None:
    """
    Combine static bad-pixel mask with per-frame overflow/underflow masks.

    Returns None when no exclusions are needed (fast path).
    """
    bad: np.ndarray | None = None

    if bad_pixel_mask is not None:
        bad = bad_pixel_mask.astype(bool, copy=True)

    if overflow_mask is not None:
        bad = overflow_mask if bad is None else (bad | overflow_mask)

    if underflow_mask is not None:
        bad = underflow_mask if bad is None else (bad | underflow_mask)

    return bad


def cm_correct_frame_per_asic(
        residual: np.ndarray,
        asic_slices: dict[str, tuple[int, int, int, int]],
        bad_pixel_mask: np.ndarray | None = None,
        overflow_mask:  np.ndarray | None = None,
        underflow_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply CM correction to a single 2-D residual frame (Y, X) per ASIC.

    Bad pixels (bad_pixel_mask), overflow pixels, and underflow pixels are
    excluded from the median computation (set to +inf before median, matching
    ROOT HCommonModeMedian behaviour).  The CM value is still subtracted from
    ALL pixels in the ASIC row including bad ones — matching ROOT which
    subtracts the median from the whole segment regardless.

    Parameters
    ----------
    residual        : float32 (n_Y, n_X)  — offset-subtracted frame
    asic_slices     : dict mapping ASIC name to (Y0, Y1, X0, X1) bounds
    bad_pixel_mask  : bool   (n_Y, n_X) or None — True = bad (static)
    overflow_mask   : bool   (n_Y, n_X) or None — True = overflow this frame
    underflow_mask  : bool   (n_Y, n_X) or None — True = underflow this frame

    Returns
    -------
    corrected  : float32 (n_Y, n_X)
    cm_values  : float32 (n_Y, n_asics) — CM subtracted from each ASIC
    asic_names : ndarray of ASIC name strings in sorted order
    """
    n_Y    = residual.shape[0]
    n_asics = len(asic_slices)
    cm_values = np.zeros((n_Y, n_asics), dtype=np.float32)
    corrected = residual.astype(np.float32, copy=True)

    # Build combined bad mask once for the whole frame
    bad_full = _make_bad_mask(residual, bad_pixel_mask,
                              overflow_mask, underflow_mask)

    asic_names = sorted(asic_slices.keys())
    for i, name in enumerate(asic_names):
        _, _, x0, x1 = asic_slices[name]

        asic_data = residual[:, x0:x1].astype(np.float32)

        bad_asic = bad_full[:, x0:x1] if bad_full is not None else None

        cm_values[:, i] = _masked_median_axis1(asic_data, bad_asic)

        # Subtract CM from ALL pixels in this ASIC (including bad ones),
        # replacing NaN CM with 0 so bad rows are not further corrupted
        cm_col = np.where(np.isfinite(cm_values[:, i]),
                          cm_values[:, i], 0.0).astype(np.float32)
        corrected[:, x0:x1] = asic_data - cm_col[:, np.newaxis]

    return corrected, cm_values, np.array(asic_names)


def cm_correct_frame(
        residual: np.ndarray,
        asic_slices: dict[str, tuple[int, int, int, int]] | None = None,
        bad_pixel_mask: np.ndarray | None = None,
        overflow_mask:  np.ndarray | None = None,
        underflow_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Apply CM correction to a single 2-D residual frame (Y, X).

    If asic_slices is provided, computes CM per ASIC (recommended for
    multi-ASIC detectors with independent baselines).
    Otherwise falls back to legacy per-row CM (all X columns), which
    matches ROOT HCommonModeMedian with NADCs=1.

    Bad pixels, overflow, and underflow are excluded from the median
    (set to +inf before nth_element, matching ROOT exactly).

    Parameters
    ----------
    residual        : float32 (n_Y, n_X)  — offset-subtracted frame
    asic_slices     : optional dict mapping ASIC name to (Y0, Y1, X0, X1)
    bad_pixel_mask  : bool   (n_Y, n_X) or None — True = bad (static)
    overflow_mask   : bool   (n_Y, n_X) or None — True = overflow this frame
    underflow_mask  : bool   (n_Y, n_X) or None — True = underflow this frame

    Returns
    -------
    corrected  : float32 (n_Y, n_X)
    cm_values  : float32 (n_Y,) or (n_Y, n_asics)
    asic_names : None or ndarray of ASIC name strings
    """
    if asic_slices:
        return cm_correct_frame_per_asic(
            residual, asic_slices,
            bad_pixel_mask=bad_pixel_mask,
            overflow_mask=overflow_mask,
            underflow_mask=underflow_mask,
        )

    # Legacy: CM over all X columns per row
    bad_full = _make_bad_mask(residual, bad_pixel_mask,
                              overflow_mask, underflow_mask)
    cm = _masked_median_axis1(
        residual.astype(np.float32), bad_full
    )
    # Replace NaN CM (all-bad row) with 0
    cm_safe = np.where(np.isfinite(cm), cm, 0.0).astype(np.float32)
    corrected = (residual.astype(np.float32) - cm_safe[:, np.newaxis])
    return corrected, cm.astype(np.float32), None


def apply_common_mode_correction(
        data:           np.ndarray,
        offset:         np.ndarray,
        label:          str = "",
        asic_slices:    dict[str, tuple[int, int, int, int]] | None = None,
        bad_pixel_mask: np.ndarray | None = None,
        n_bits:         int = 16,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """
    Subtract per-pixel offset then apply CM correction to a stack of frames.

    Overflow and underflow pixels (raw ADC sentinels) are detected from the
    raw integer data BEFORE offset subtraction and excluded from the CM
    median, matching ROOT HStepOffNoiMapHLL behaviour.

    Parameters
    ----------
    data            : uint16 or float32 (n_frames, n_Y, n_X) — raw or
                      already-cast frames.  Overflow/underflow detection
                      works on integer values; if float32 is passed the
                      sentinel check is skipped.
    offset          : float32 (n_Y, n_X) — offset map
    label           : tag for diagnostic prints
    asic_slices     : optional ASIC geometry
    bad_pixel_mask  : bool (n_Y, n_X) or None — static bad-pixel map
    n_bits          : ADC bit depth for sentinel detection (default 16)

    Returns
    -------
    corrected  : float32 (n_frames, n_Y, n_X)
    cm_map     : float32 (n_frames, n_Y) or (n_frames, n_Y, n_asics)
    asic_names : None or list of ASIC names
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Applying CM correction …")

    underflow_val = 0
    overflow_val  = (1 << n_bits) - 1   # 65535 for 16-bit

    n_frames = data.shape[0]

    # Detect overflow/underflow from integer raw values if available
    if np.issubdtype(data.dtype, np.integer):
        overflow_stack  = (data == overflow_val)   # (N, Y, X) bool
        underflow_stack = (data == underflow_val)
    else:
        # float input — sentinel detection not possible, skip
        overflow_stack  = None
        underflow_stack = None

    residual = (data.astype(np.float32) - offset[np.newaxis])

    if asic_slices:
        asic_names = sorted(asic_slices.keys())
        n_asics    = len(asic_names)
        cm_map     = np.zeros((n_frames, data.shape[1], n_asics),
                              dtype=np.float32)
        corrected  = np.empty_like(residual)

        for f in range(n_frames):
            of  = overflow_stack[f]  if overflow_stack  is not None else None
            uf  = underflow_stack[f] if underflow_stack is not None else None
            corr_f, cm_f, _ = cm_correct_frame_per_asic(
                residual[f], asic_slices,
                bad_pixel_mask=bad_pixel_mask,
                overflow_mask=of,
                underflow_mask=uf,
            )
            corrected[f]  = corr_f
            cm_map[f]     = cm_f

        return corrected, cm_map, asic_names

    else:
        # Legacy: per-row CM over all columns
        cm_map    = np.zeros((n_frames, data.shape[1]), dtype=np.float32)
        corrected = np.empty_like(residual)

        for f in range(n_frames):
            of  = overflow_stack[f]  if overflow_stack  is not None else None
            uf  = underflow_stack[f] if underflow_stack is not None else None
            corr_f, cm_f, _ = cm_correct_frame(
                residual[f], asic_slices=None,
                bad_pixel_mask=bad_pixel_mask,
                overflow_mask=of,
                underflow_mask=uf,
            )
            corrected[f] = corr_f
            cm_map[f]    = cm_f

        return corrected, cm_map.astype(np.float32), None


def compute_cm_noise(
        cm_map:      np.ndarray,
        label:       str = "",
        asic_names:  list[str] | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """
    CM noise = per-Y RMS of the CM correction values across frames.

    Parameters
    ----------
    cm_map     : float32 (n_frames, n_Y) or (n_frames, n_Y, n_asics)
    asic_names : list of ASIC names if multi-ASIC

    Returns
    -------
    cm_noise : float32 (n_Y,) or dict mapping ASIC name → float32 (n_Y,)
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}CM noise …")

    if asic_names and cm_map.ndim == 3:
        result = {}
        for i, name in enumerate(asic_names):
            result[name] = np.std(cm_map[:, :, i], axis=0,
                                  ddof=1).astype(np.float32)
        return result

    return np.std(cm_map, axis=0, ddof=1).astype(np.float32)
