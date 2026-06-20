"""
pnccd_ana.plotting.common
==========================
Shared plotting helpers used across all plotting modules.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import lru_cache


def _cb(ax, im, label="ADU"):
    """Add colorbar to axis."""
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)


def _stats_box(ax, arr, fmt=".2f"):
    """Add statistics box to axis."""
    if isinstance(arr, dict):
        vals = np.concatenate(list(arr.values()))
    else:
        vals = arr.ravel()
    txt  = (f"μ={float(vals.mean()):{fmt}}\n"
            f"σ={float(vals.std()):{fmt}}\n"
            f"med={float(np.median(vals)):{fmt}}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))


def _asic_guides(ax):
    """Stub for backward compatibility — does nothing in single-hybrid mode."""
    pass


# Module-level variables (initialized on first use via lazy initialization)
_GRADE_PALETTE: dict = {}
_GROUP_LABEL: dict = {}


@lru_cache(maxsize=1)
def _build_grade_palette() -> dict:
    """Build a fixed color palette for grade IDs (cached)."""
    from ..physics.pattern_recognition import N_GRADES
    n = N_GRADES
    cmap = plt.cm.get_cmap("tab20")
    palette = {}
    for gid in range(n):
        palette[gid] = cmap(gid / n)
    return palette


def _get_grade_palette() -> dict:
    """Get or initialize the grade palette (lazy initialization)."""
    global _GRADE_PALETTE
    if not _GRADE_PALETTE:
        _GRADE_PALETTE = _build_grade_palette()
    return _GRADE_PALETTE


@lru_cache(maxsize=1)
def _build_group_label() -> dict:
    """Build grade-name lookup for grouped spectra (cached)."""
    from ..physics.pattern_recognition import _GRADE_DEFS, GRADE_OTHER
    label = {}
    for gid, name, _ in _GRADE_DEFS:
        label[gid] = name.title()
    label[GRADE_OTHER] = "Other"
    return label


def _get_group_label() -> dict:
    """Get or initialize the group label (lazy initialization)."""
    global _GROUP_LABEL
    if not _GROUP_LABEL:
        _GROUP_LABEL = _build_group_label()
    return _GROUP_LABEL


def _auto_bin_edges(data, n_bins=80, method="sqrt"):
    """Compute histogram bin edges automatically."""
    arr = np.asarray(data)
    # Handle structured arrays (e.g., event record with named fields)
    if arr.dtype.names is not None:
        # Prefer adu_seed or adu_sum if present; fall back to first numeric field
        for preferred in ("adu_seed", "adu_sum"):
            if preferred in arr.dtype.names:
                arr = arr[preferred]
                break
        else:
            # Use first field that can be cast to float
            for name in arr.dtype.names:
                try:
                    arr = arr[name].astype(float)
                    break
                except (ValueError, TypeError):
                    continue
            else:
                raise TypeError(f"No numeric field found in structured array: {arr.dtype}")
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        raise ValueError("No finite data for binning")
    if method == "sqrt":
        n = max(int(np.sqrt(len(flat))), 10)
    elif method == "rice":
        n = max(int(2 * np.power(len(flat), 1/3)), 10)
    else:
        n = n_bins
    lo, hi = np.percentile(flat, [0.5, 99.5])
    return np.linspace(lo, hi, n + 1)
