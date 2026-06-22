"""
pnccd_ana.plotting.common
==========================
Shared plotting helpers used across all plotting modules.

Deliberately kept free of physics imports so that offset and event_rec
stages do not trigger loading of calibrate.py / gain.py / cti.py.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _cb(ax, im, label="ADU"):
    """Add a compact colorbar to an axis."""
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)


def _stats_box(ax, arr, fmt=".2f"):
    """Overlay a μ / σ / median text box on an axis."""
    if isinstance(arr, dict):
        vals = np.concatenate([np.asarray(v).ravel() for v in arr.values()])
    else:
        vals = np.asarray(arr).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    txt = (f"μ={float(vals.mean()):{fmt}}\n"
           f"σ={float(vals.std()):{fmt}}\n"
           f"med={float(np.median(vals)):{fmt}}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))


def _adjust_bin_range(data, bin_edges: np.ndarray) -> np.ndarray:
    """
    Adjust bin range to data percentiles, keeping the same number of bins.

    Works with plain arrays and structured arrays (picks adu_sum or adu_seed
    if available).
    """
    arr = np.asarray(data)
    if arr.dtype.names is not None:
        for preferred in ("adu_seed", "adu_sum"):
            if preferred in arr.dtype.names:
                arr = arr[preferred]
                break
        else:
            for name in arr.dtype.names:
                try:
                    arr = arr[name].astype(float)
                    break
                except (ValueError, TypeError):
                    continue
            else:
                return bin_edges

    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return bin_edges
    lo, hi = np.percentile(flat, [0.5, 99.5])
    return np.linspace(lo, hi, len(bin_edges))


def save_figure(fig, path, dpi: int = 150) -> None:
    """Save and close a figure."""
    from pathlib import Path
    fig.savefig(Path(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
