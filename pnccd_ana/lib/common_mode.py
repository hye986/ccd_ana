"""
pnccd_ana.lib.common_mode
=========================
Common-mode (CM) correction.

Rolling shutter: sensor reads rows from bottom to top.
A detector "row" is a horizontal line of pixels (constant Y, all X) that are
read out simultaneously. 

For multi-ASIC detectors (8 ASICs × 64 columns = 512 total), CM is computed
per ASIC to avoid cross-ASIC interference in the correction.

Array layout: data[frame, Y, X]
CM is computed per ASIC: median over 64 X pixels for each Y row.
Resulting cm_map shape: (n_frames, n_Y, n_asics)
"""

from __future__ import annotations

import numpy as np

from .geometry import ASIC_WIDTH, N_ASICS


def cm_correct_frame_per_asic(
        residual: np.ndarray,
        asic_slices: dict[str, tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply CM correction to a single 2-D residual frame (Y, X) per ASIC.

    Parameters
    ----------
    residual    : float32 (n_Y, n_X)  — offset-subtracted frame
    asic_slices : dict mapping ASIC name to (Y0, Y1, X0, X1) bounds

    Returns
    -------
    corrected   : float32 (n_Y, n_X)
    cm_values   : float32 (n_Y, n_asics) — CM subtracted from each ASIC
    asic_names  : list of ASIC names in order
    """
    n_Y = residual.shape[0]
    n_asics = len(asic_slices)
    cm_values = np.zeros((n_Y, n_asics), dtype=np.float32)
    corrected = residual.copy()
    
    asic_names = sorted(asic_slices.keys())
    for i, name in enumerate(asic_names):
        _, _, x0, x1 = asic_slices[name]
        # CM = median over X for this ASIC's columns
        asic_data = corrected[:, x0:x1+1]
        cm_values[:, i] = np.median(asic_data, axis=1)
        # Subtract CM from this ASIC's columns
        corrected[:, x0:x1+1] = asic_data - cm_values[:, i:i+1]
    
    return corrected.astype(np.float32), cm_values, np.array(asic_names)


def cm_correct_frame(
        residual: np.ndarray,
        asic_slices: dict[str, tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Apply CM correction to a single 2-D residual frame (Y, X).
    
    If asic_slices is provided, computes CM per ASIC.
    Otherwise falls back to legacy per-row CM (all X columns).

    Parameters
    ----------
    residual    : float32 (n_Y, n_X)  — offset-subtracted frame
    asic_slices : optional dict mapping ASIC name to (Y0, Y1, X0, X1)

    Returns
    -------
    corrected   : float32 (n_Y, n_X)
    cm_values   : float32 (n_Y,) or (n_Y, n_asics)
    asic_names  : None or list of ASIC names
    """
    if asic_slices:
        return cm_correct_frame_per_asic(residual, asic_slices)
    
    # Legacy: CM over all X columns
    cm = np.median(residual, axis=1)
    return (residual - cm[:, np.newaxis]).astype(np.float32), cm.astype(np.float32), None


def apply_common_mode_correction(
        data:         np.ndarray,
        offset:       np.ndarray,
        label:        str = "",
        asic_slices:  dict[str, tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """
    Subtract per-pixel offset then apply CM correction to a stack of frames.

    Parameters
    ----------
    data        : float32 (n_frames, n_Y, n_X) — raw frames
    offset      : float32 (n_Y, n_X)           — pedestal map
    asic_slices : optional ASIC geometry

    Returns
    -------
    corrected   : float32 (n_frames, n_Y, n_X)
    cm_map      : float32 (n_frames, n_Y) or (n_frames, n_Y, n_asics)
    asic_names  : None or list of ASIC names
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Applying CM correction …")
    residual = (data - offset[np.newaxis]).astype(np.float32)
    
    if asic_slices:
        # Per-ASIC CM correction
        asic_names = sorted(asic_slices.keys())
        n_asics = len(asic_names)
        cm_map = np.zeros((data.shape[0], data.shape[1], n_asics), dtype=np.float32)
        corrected = residual.copy()
        
        for i, name in enumerate(asic_names):
            _, _, x0, x1 = asic_slices[name]
            # CM = median over X for this ASIC
            cm_map[:, :, i] = np.median(residual[:, :, x0:x1+1], axis=2)
            corrected[:, :, x0:x1+1] = residual[:, :, x0:x1+1] - cm_map[:, :, i:i+1]
        
        return corrected, cm_map, asic_names
    else:
        # Legacy: CM over all X columns
        cm_map = np.median(residual, axis=2).astype(np.float32)   # axis=2 = X
        corrected = residual - cm_map[:, :, np.newaxis]
        return corrected, cm_map, None


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
    cm_noise : float32 (n_Y,) or dict of per-ASIC noise arrays
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}CM noise …")
    
    if asic_names and cm_map.ndim == 3:
        result = {}
        for i, name in enumerate(asic_names):
            result[name] = np.std(cm_map[:, :, i], axis=0, ddof=1).astype(np.float32)
        return result
    
    return np.std(cm_map, axis=0, ddof=1).astype(np.float32)
