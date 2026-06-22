"""
pnccd_ana.plotting.cti_plots
=============================
Diagnostic plots for CTI/CTE calibration.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_cte_map(
        cte_map: np.ndarray,
        out_dir: Path,
) -> None:
    """2-D colour map of cumulative CTE correction."""
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 6))
    vals    = cte_map[cte_map > 0]
    vmin    = float(np.percentile(vals, 1))  if len(vals) else 0.9
    vmax    = 1.0

    im = ax.imshow(cte_map, origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap="plasma")
    plt.colorbar(im, ax=ax, label="Cumulative CTE")
    ax.set_title("CTE Map")
    ax.set_xlabel("X [detector column]")
    ax.set_ylabel("Y [detector row]")
    fig.tight_layout()
    fig.savefig(out_dir / "cte_map.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir/'cte_map.png'}")


def plot_cti_summary(
        cte_map:      np.ndarray,
        bad_gain_map: np.ndarray,
        out_dir:      Path,
) -> None:
    """
    CTI per column (row=1 CTE value) for good columns.
    Shows distribution of 1-CTE as histogram and scatter vs column.
    """
    out_dir   = Path(out_dir)
    good_cols = bad_gain_map[0, :] == 0
    cti_vals  = 1.0 - cte_map[1, good_cols]
    cols      = np.where(good_cols)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(cols, cti_vals, s=3, color="steelblue", alpha=0.6)
    axes[0].axhline(float(np.median(cti_vals)), color="tomato",
                    linestyle="--", label=f"median={np.median(cti_vals):.2e}")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("CTI = 1 – CTE (row 1)")
    axes[0].set_title("CTI per column")
    axes[0].legend()

    lo, hi = np.percentile(cti_vals, [0.5, 99.5]) if len(cti_vals) > 1 \
             else (0, 1e-3)
    axes[1].hist(cti_vals, bins=100, range=(lo, hi),
                 color="steelblue", histtype="stepfilled", alpha=0.7)
    axes[1].set_xlabel("CTI = 1 – CTE (row 1)")
    axes[1].set_ylabel("Columns")
    axes[1].set_title("CTI distribution")

    fig.tight_layout()
    fig.savefig(out_dir / "cti_summary.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir/'cti_summary.png'}")


def plot_signal_vs_row(
        filtered,
        out_dir:   Path,
        n_col_sample: int = 6,
        n_events_max: int = 2000,
) -> None:
    """
    Signal vs row scatter for a sample of columns.
    Illustrates CTI-induced signal degradation with row.
    """
    out_dir  = Path(out_dir)
    cols_arr = filtered.col.astype(np.int32)
    unique_cols = np.unique(cols_arr)

    # Pick evenly spaced sample columns
    step     = max(1, len(unique_cols) // n_col_sample)
    sample_c = unique_cols[::step][:n_col_sample]

    fig, axes = plt.subplots(1, len(sample_c),
                              figsize=(4 * len(sample_c), 4),
                              sharey=True)
    if len(sample_c) == 1:
        axes = [axes]

    for ax, col in zip(axes, sample_c):
        mask = cols_arr == col
        rows = filtered.row[mask]
        adus = filtered.adu_sum[mask]
        if len(rows) > n_events_max:
            idx  = np.random.choice(len(rows), n_events_max, replace=False)
            rows = rows[idx]
            adus = adus[idx]
        ax.scatter(adus, rows, s=2, alpha=0.4, color="steelblue")
        ax.set_title(f"Col {col}")
        ax.set_xlabel("ADU sum")
        if ax is axes[0]:
            ax.set_ylabel("Row")

    fig.suptitle("Signal vs Row (CTI effect) — sample columns")
    fig.tight_layout()
    fig.savefig(out_dir / "signal_vs_row.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir/'signal_vs_row.png'}")
