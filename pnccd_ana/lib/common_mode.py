"""
pnccd_ana.lib.common_mode
=========================
Common-mode (CM) correction.

Rolling shutter: sensor reads rows from bottom to top.
A detector "row" is a horizontal line of pixels (constant Y, all X) that are
read out simultaneously. The CM value for each Y position is the median
over all X pixels in that row of the residual image, and is subtracted
from every pixel in that detector row.

Array layout: data[frame, Y, X]
CM is computed along axis=1 (X), giving cm_map shape (n_frames, n_Y).
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
    cm_values : float32 (n_Y,)     — CM subtracted from each Y position
    """
    cm = np.median(residual, axis=1)          # (n_Y,) — median over X
    return (residual - cm[:, np.newaxis]).astype(np.float32), cm.astype(np.float32)


def apply_common_mode_correction(
        data:   np.ndarray,
        offset: np.ndarray,
        label:  str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subtract per-pixel offset then apply CM correction to a stack of frames.

    Parameters
    ----------
    data   : float32 (n_frames, n_Y, n_X) — raw frames
    offset : float32 (n_Y, n_X)           — pedestal map

    Returns
    -------
    corrected : float32 (n_frames, n_Y, n_X)
    cm_map    : float32 (n_frames, n_Y)   — CM value per frame per Y position
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Applying CM correction …")
    residual  = (data - offset[np.newaxis]).astype(np.float32)
    cm_map    = np.median(residual, axis=2).astype(np.float32)   # axis=2 = X
    corrected = residual - cm_map[:, :, np.newaxis]
    return corrected, cm_map


def compute_cm_noise(cm_map: np.ndarray, label: str = "") -> np.ndarray:
    """
    CM noise = per-Y RMS of the CM correction values across frames.

    Parameters
    ----------
    cm_map : float32 (n_frames, n_Y)

    Returns
    -------
    cm_noise : float32 (n_Y,)
    """
    tag = f"[{label}] " if label else ""
    print(f"  {tag}CM noise …")
    return np.std(cm_map, axis=0, ddof=1).astype(np.float32)
