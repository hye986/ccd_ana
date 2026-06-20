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


_GRADE_PALETTE = None


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


_GRADE_LABEL = None


@lru_cache(maxsize=1)
def _build_group_label() -> dict:
    """Build grade-name lookup for grouped spectra (cached)."""
    from ..physics.pattern_recognition import _GRADE_DEFS, GRADE_OTHER
    label = {}
    for gid, name, _ in _GRADE_DEFS:
        label[gid] = name.title()
    label[GRADE_OTHER] = "Other"
    return label


def _auto_bin_edges(data, n_bins=80, method="sqrt"):
    """Compute histogram bin edges automatically."""
    flat = np.asarray(data).ravel()
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
