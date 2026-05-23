"""
pnccd_ana.lib.common_mode
=========================
Common-mode (CM) correction.

A detector "row" is a vertical line of pixels (constant X, all Y) that are
read out simultaneously.  The CM value for each X position is the median
over all Y pixels in that column of the residual image, and is subtracted
from every pixel in that detector row.

Array layout: data[frame, Y, X]
CM is computed along axis=1 (Y), giving cm_map shape (n_frames, n_X).
"""

from __future__ import annotations

import numpy as np


def cm_correct_frame(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply CM correction to a single 2-D residual frame (Y, X).

    Parameters
    ----------
    residual : float32 (n_Y, n_X)  — offset-subtracted frame

    Returns
    -------
    corrected : float32 (n_Y, n_X)
    cm_values : float32 (n_X,)     — CM subtracted from each X position
    """
    cm = np.median(residual, axis=0)          # (n_X,) — median over Y
    return (residual - cm[np.newaxis, :]).astype(np.float32), cm.astype(np.float32)


def apply_common_mode_correction(
        data:   np.ndarray,
        offset: np.ndarray,
        label:  str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subtract per-pixel offset then apply CM correction to a stack of frames.

    Parameters
    ----------
    data   : float32 (n_frames, n_Y, n_X) — raw or rollover-corrected frames
    offset : float32 (n_Y, n_X)           — pedestal map

    Returns
    -------
    corrected : float32 (n_frames, n_Y, n_X)
    cm_map    : float32 (n_frames, n_X)    — CM value per frame per X position
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Applying CM correction …")
    residual  = (data - offset[np.newaxis]).astype(np.float32)
    cm_map    = np.median(residual, axis=1).astype(np.float32)   # axis=1 = Y
    corrected = residual - cm_map[:, np.newaxis, :]
    return corrected, cm_map


def compute_cm_noise(cm_map: np.ndarray, label: str = "") -> np.ndarray:
    """
    CM noise = per-X RMS of the CM correction values across frames.

    Parameters
    ----------
    cm_map : float32 (n_frames, n_X)

    Returns
    -------
    cm_noise : float32 (n_X,)
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}CM noise …")
    return np.std(cm_map, axis=0, ddof=1).astype(np.float32)
