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

SplitEvenOdd mode (matching ROOT HCommonModeMedianEvenOdd / Analysis.Filter.SplitEvenOdd=1):
  Within each ASIC segment, compute separate medians for even-indexed and
  odd-indexed columns (software column index, 0-based).  Each median is
  subtracted only from its own parity columns.  This corrects for the
  correlated noise pattern seen on alternating readout channels in pnCCDs.

Overflow/underflow handling
---------------------------
Raw ADC sentinel values (underflow=0, overflow=2^n_bits-1) are excluded
from the CM median, matching ROOT HCommonModeMedian which sets bad/excluded
pixels to +inf before nth_element so they never enter the median count.
"""

from __future__ import annotations

import numpy as np

from ..io.geometry import ASIC_WIDTH

# Default 16-bit ADC sentinel values
_ADC_BITS        = 16
_UNDERFLOW_VALUE = 0
_OVERFLOW_VALUE  = (1 << _ADC_BITS) - 1   # 65535


def _masked_median_axis1(data: np.ndarray,
                         bad:  np.ndarray | None) -> np.ndarray:
    """
    Compute median along axis=1 (columns) for each row, excluding bad pixels.

    Matches ROOT HCommonModeMedian: bad pixels set to +inf before partial
    sort so they never contribute to the median count.

    Parameters
    ----------
    data : float32 (n_Y, n_cols)
    bad  : bool   (n_Y, n_cols) or None — True = exclude

    Returns
    -------
    medians : float32 (n_Y,)
    """
    if bad is None:
        return np.median(data, axis=1).astype(np.float32)

    buf = data.astype(np.float32, copy=True)
    buf[bad] = np.inf

    n_Y, n_cols = buf.shape
    n_good = np.isfinite(buf).sum(axis=1)   # (n_Y,)

    # Full sort — n_cols is small (64 per ASIC), so this is fast
    sorted_buf = np.sort(buf, axis=1)       # inf floats to right end

    lo = n_good // 2
    hi = np.maximum(lo - 1, 0)

    rows  = np.arange(n_Y)
    lower = sorted_buf[rows, lo]
    upper = sorted_buf[rows, hi]

    even_mask = (n_good > 0) & ((n_good & 1) == 0)
    medians   = np.where(even_mask, (lower + upper) * 0.5, lower)
    medians   = np.where(n_good == 0, np.nan, medians)

    return medians.astype(np.float32)


def _masked_median_parity(data: np.ndarray,
                           bad:  np.ndarray | None,
                           parity: int) -> np.ndarray:
    """
    Compute median along axis=1 for even (parity=0) or odd (parity=1)
    column indices only, excluding bad pixels.

    Matches ROOT HCommonModeMedianEvenOdd which separates even/odd columns
    within each segment and computes independent medians.

    Parameters
    ----------
    data   : float32 (n_Y, n_cols)  — full ASIC slice
    bad    : bool   (n_Y, n_cols) or None
    parity : 0 for even columns, 1 for odd columns

    Returns
    -------
    medians : float32 (n_Y,)
    """
    # Select columns by parity (0-based column index within the ASIC slice)
    col_indices = np.arange(data.shape[1])
    sel = col_indices[col_indices % 2 == parity]   # e.g. [0,2,4,...] or [1,3,5,...]

    sub_data = data[:, sel].astype(np.float32, copy=True)  # (n_Y, n_sel)
    sub_bad  = bad[:, sel] if bad is not None else None

    return _masked_median_axis1(sub_data, sub_bad)


def _make_bad_mask(bad_pixel_mask: np.ndarray | None,
                   overflow_mask:  np.ndarray | None,
                   underflow_mask: np.ndarray | None) -> np.ndarray | None:
    """
    Combine static bad-pixel mask with per-frame overflow/underflow masks.

    Returns None when no exclusions needed (triggers fast path).
    """
    bad: np.ndarray | None = None

    if bad_pixel_mask is not None:
        bad = bad_pixel_mask.astype(bool, copy=True)
    if overflow_mask is not None:
        bad = overflow_mask.astype(bool) if bad is None else (bad | overflow_mask)
    if underflow_mask is not None:
        bad = underflow_mask.astype(bool) if bad is None else (bad | underflow_mask)

    return bad


def cm_correct_frame_per_asic(
        residual:       np.ndarray,
        asic_slices:    dict[str, tuple[int, int, int, int]],
        bad_pixel_mask: np.ndarray | None = None,
        overflow_mask:  np.ndarray | None = None,
        underflow_mask: np.ndarray | None = None,
        split_even_odd: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply CM correction to a single 2-D residual frame (Y, X) per ASIC.

    When split_even_odd=True, computes independent medians for even- and
    odd-indexed columns within each ASIC segment, matching ROOT
    HCommonModeMedianEvenOdd / Analysis.Filter.SplitEvenOdd=1.

    Bad pixels, overflow, and underflow are excluded from the median
    (set to +inf before sort, matching ROOT HCommonModeMedian).
    The CM value is subtracted from ALL pixels including bad ones —
    matching ROOT which subtracts median from the whole segment.

    Parameters
    ----------
    residual        : float32 (n_Y, n_X)
    asic_slices     : dict mapping ASIC name to (Y0, Y1, X0, X1)
    bad_pixel_mask  : bool (n_Y, n_X) or None
    overflow_mask   : bool (n_Y, n_X) or None
    underflow_mask  : bool (n_Y, n_X) or None
    split_even_odd  : if True, use separate even/odd column medians
                      (ROOT HCommonModeMedianEvenOdd)

    Returns
    -------
    corrected  : float32 (n_Y, n_X)
    cm_values  : float32 (n_Y, n_asics) — mean of even+odd medians per ASIC row
    asic_names : ndarray of ASIC name strings
    """
    n_Y     = residual.shape[0]
    n_asics = len(asic_slices)
    cm_values = np.zeros((n_Y, n_asics), dtype=np.float32)
    corrected = residual.astype(np.float32, copy=True)

    bad_full = _make_bad_mask(bad_pixel_mask, overflow_mask, underflow_mask)

    asic_names = sorted(asic_slices.keys())
    for i, name in enumerate(asic_names):
        _, _, x0, x1 = asic_slices[name]

        asic_data = residual[:, x0:x1].astype(np.float32)
        bad_asic  = bad_full[:, x0:x1] if bad_full is not None else None

        if split_even_odd:
            # ROOT HCommonModeMedianEvenOdd: independent median per parity
            cm_even = _masked_median_parity(asic_data, bad_asic, parity=0)  # (n_Y,)
            cm_odd  = _masked_median_parity(asic_data, bad_asic, parity=1)  # (n_Y,)

            cm_even_safe = np.where(np.isfinite(cm_even), cm_even, 0.0).astype(np.float32)
            cm_odd_safe  = np.where(np.isfinite(cm_odd),  cm_odd,  0.0).astype(np.float32)

            # Subtract parity-specific medians from ALL columns (including bad)
            # matching ROOT: Frame[seg + i] -= EvenMedian  (for even i)
            col_indices = np.arange(asic_data.shape[1])
            even_cols = col_indices[col_indices % 2 == 0]
            odd_cols  = col_indices[col_indices % 2 == 1]

            corrected[:, x0 + even_cols] = (asic_data[:, even_cols]
                                             - cm_even_safe[:, np.newaxis])
            corrected[:, x0 + odd_cols]  = (asic_data[:, odd_cols]
                                             - cm_odd_safe[:, np.newaxis])

            # Store mean of even+odd as the representative CM value for display
            with np.errstate(invalid="ignore"):
                cm_mean = np.where(
                    np.isfinite(cm_even) & np.isfinite(cm_odd),
                    (cm_even + cm_odd) * 0.5,
                    np.where(np.isfinite(cm_even), cm_even, cm_odd),
                )
            cm_values[:, i] = cm_mean.astype(np.float32)

        else:
            # Standard: single median over all columns in ASIC
            cm_col  = _masked_median_axis1(asic_data, bad_asic)   # (n_Y,)
            cm_safe = np.where(np.isfinite(cm_col), cm_col, 0.0).astype(np.float32)
            cm_values[:, i]     = cm_col
            corrected[:, x0:x1] = asic_data - cm_safe[:, np.newaxis]

    return corrected, cm_values, np.array(asic_names)


def cm_correct_frame(
        residual:       np.ndarray,
        asic_slices:    dict[str, tuple[int, int, int, int]] | None = None,
        bad_pixel_mask: np.ndarray | None = None,
        overflow_mask:  np.ndarray | None = None,
        underflow_mask: np.ndarray | None = None,
        split_even_odd: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Apply CM correction to a single 2-D residual frame (Y, X).

    If asic_slices is provided, computes CM per ASIC (with optional
    even/odd split matching ROOT HCommonModeMedianEvenOdd).
    Otherwise falls back to per-row CM (all X columns).

    Parameters
    ----------
    residual        : float32 (n_Y, n_X)
    asic_slices     : optional ASIC geometry
    bad_pixel_mask  : bool (n_Y, n_X) or None
    overflow_mask   : bool (n_Y, n_X) or None
    underflow_mask  : bool (n_Y, n_X) or None
    split_even_odd  : separate even/odd column medians (ROOT SplitEvenOdd=1)

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
            split_even_odd=split_even_odd,
        )

    # Legacy: CM over all X columns per row (no even/odd split)
    bad_full = _make_bad_mask(bad_pixel_mask, overflow_mask, underflow_mask)

    if split_even_odd:
        # Even/odd split on full row (no ASIC boundaries)
        n_Y, n_X = residual.shape
        corrected = residual.astype(np.float32, copy=True)
        col_indices = np.arange(n_X)
        even_cols = col_indices[col_indices % 2 == 0]
        odd_cols  = col_indices[col_indices % 2 == 1]

        data_f32 = residual.astype(np.float32)

        cm_even = _masked_median_parity(data_f32, bad_full, parity=0)
        cm_odd  = _masked_median_parity(data_f32, bad_full, parity=1)

        cm_even_safe = np.where(np.isfinite(cm_even), cm_even, 0.0).astype(np.float32)
        cm_odd_safe  = np.where(np.isfinite(cm_odd),  cm_odd,  0.0).astype(np.float32)

        corrected[:, even_cols] = data_f32[:, even_cols] - cm_even_safe[:, np.newaxis]
        corrected[:, odd_cols]  = data_f32[:, odd_cols]  - cm_odd_safe[:, np.newaxis]

        with np.errstate(invalid="ignore"):
            cm_mean = np.where(
                np.isfinite(cm_even) & np.isfinite(cm_odd),
                (cm_even + cm_odd) * 0.5,
                np.where(np.isfinite(cm_even), cm_even, cm_odd),
            )
        return corrected, cm_mean.astype(np.float32), None

    cm      = _masked_median_axis1(residual.astype(np.float32), bad_full)
    cm_safe = np.where(np.isfinite(cm), cm, 0.0).astype(np.float32)
    corrected = residual.astype(np.float32) - cm_safe[:, np.newaxis]
    return corrected, cm.astype(np.float32), None


def apply_common_mode_correction(
        data:           np.ndarray,
        offset:         np.ndarray,
        label:          str = "",
        asic_slices:    dict[str, tuple[int, int, int, int]] | None = None,
        bad_pixel_mask: np.ndarray | None = None,
        n_bits:         int = 16,
        split_even_odd: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """
    Subtract per-pixel offset then apply CM correction to a stack of frames.

    Overflow/underflow pixels are detected from raw integer data BEFORE
    offset subtraction and excluded from the CM median, matching ROOT
    HStepOffNoiMapHLL.

    Parameters
    ----------
    data            : uint16 or float32 (n_frames, n_Y, n_X)
    offset          : float32 (n_Y, n_X)
    label           : tag for diagnostic prints
    asic_slices     : optional ASIC geometry
    bad_pixel_mask  : bool (n_Y, n_X) or None
    n_bits          : ADC bit depth (default 16)
    split_even_odd  : separate even/odd column medians per ASIC

    Returns
    -------
    corrected  : float32 (n_frames, n_Y, n_X)
    cm_map     : float32 (n_frames, n_Y) or (n_frames, n_Y, n_asics)
    asic_names : None or list of ASIC names
    """
    tag = f"[{label}] " if label else ""
    eo_tag = " [even/odd split]" if split_even_odd else ""
    print(f"  {tag}Applying CM correction{eo_tag} …")

    overflow_val = (1 << n_bits) - 1

    if np.issubdtype(data.dtype, np.integer):
        overflow_stack  = (data == overflow_val)
        underflow_stack = (data == 0)
    else:
        overflow_stack  = None
        underflow_stack = None

    residual = (data.astype(np.float32) - offset[np.newaxis])

    n_frames = data.shape[0]

    if asic_slices:
        asic_names = sorted(asic_slices.keys())
        n_asics    = len(asic_names)
        cm_map     = np.zeros((n_frames, data.shape[1], n_asics), dtype=np.float32)
        corrected  = np.empty_like(residual)

        for f in range(n_frames):
            of = overflow_stack[f]  if overflow_stack  is not None else None
            uf = underflow_stack[f] if underflow_stack is not None else None
            corr_f, cm_f, _ = cm_correct_frame_per_asic(
                residual[f], asic_slices,
                bad_pixel_mask=bad_pixel_mask,
                overflow_mask=of,
                underflow_mask=uf,
                split_even_odd=split_even_odd,
            )
            corrected[f] = corr_f
            cm_map[f]    = cm_f

        return corrected, cm_map, asic_names

    else:
        cm_map    = np.zeros((n_frames, data.shape[1]), dtype=np.float32)
        corrected = np.empty_like(residual)

        for f in range(n_frames):
            of = overflow_stack[f]  if overflow_stack  is not None else None
            uf = underflow_stack[f] if underflow_stack is not None else None
            corr_f, cm_f, _ = cm_correct_frame(
                residual[f], asic_slices=None,
                bad_pixel_mask=bad_pixel_mask,
                overflow_mask=of,
                underflow_mask=uf,
                split_even_odd=split_even_odd,
            )
            corrected[f] = corr_f
            cm_map[f]    = cm_f

        return corrected, cm_map.astype(np.float32), None


def compute_cm_noise(
        cm_map:     np.ndarray,
        label:      str = "",
        asic_names: list[str] | None = None,
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
        return {
            name: np.std(cm_map[:, :, i], axis=0, ddof=1).astype(np.float32)
            for i, name in enumerate(asic_names)
        }

    return np.std(cm_map, axis=0, ddof=1).astype(np.float32)
