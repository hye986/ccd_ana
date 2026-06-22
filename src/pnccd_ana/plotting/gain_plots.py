"""
pnccd_ana.plotting.gain_plots
==============================
Diagnostic plots for gain calibration.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..physics.calibrate import GRADE_NAMES, N_GRADES


def plot_gain_map(
        gain_map:     np.ndarray,
        bad_gain_map: np.ndarray,
        out_dir:      Path,
        vmin:         float | None = None,
        vmax:         float | None = None,
) -> None:
    """2-D colour map of gain [eV/ADU] with bad-pixel overlay."""
    out_dir = Path(out_dir)
    good    = gain_map[bad_gain_map == 0]
    if len(good) == 0:
        return
    vmin = vmin or float(np.percentile(good, 1))
    vmax = vmax or float(np.percentile(good, 99))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gain map
    im = axes[0].imshow(gain_map, origin="lower",
                        aspect="auto", vmin=vmin, vmax=vmax,
                        cmap="viridis")
    plt.colorbar(im, ax=axes[0], label="Gain [eV/ADU]")
    axes[0].set_title("Gain Map")
    axes[0].set_xlabel("X [detector column]")
    axes[0].set_ylabel("Y [detector row]")

    # Bad-gain quality map
    im2 = axes[1].imshow(bad_gain_map, origin="lower",
                         aspect="auto", cmap="RdYlGn_r",
                         vmin=0, vmax=2)
    cbar = plt.colorbar(im2, ax=axes[1], ticks=[0, 1, 2])
    cbar.set_ticklabels(["good", "fallback", "kept"])
    axes[1].set_title("Bad Gain Map (quality)")
    axes[1].set_xlabel("X [detector column]")
    axes[1].set_ylabel("Y [detector row]")

    fig.tight_layout()
    fig.savefig(out_dir / "gain_map.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir/'gain_map.png'}")


def plot_gain_histogram(
        gain_map:       np.ndarray,
        bad_gain_map:   np.ndarray,
        split_even_odd: bool,
        out_dir:        Path,
        n_bins:         int = 200,
) -> None:
    """
    Three-panel gain histogram:
      Panel 1: Row-0 gain per column (intrinsic column gain, split by parity)
      Panel 2: Gain vs row for sample columns (shows CTE effect)
      Panel 3: All-pixel gain distribution (full 2D)
    """
    out_dir        = Path(out_dir)
    n_rows, n_cols = gain_map.shape
    good_cols      = bad_gain_map[0, :] == 0
    rows           = np.arange(n_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: row-0 gain split by parity
    ax = axes[0]
    if split_even_odd:
        for parity, label, color in [(0, "odd col (p=0)",  "steelblue"),
                                      (1, "even col (p=1)", "tomato")]:
            cols = np.array([c for c in range(n_cols)
                             if (c + 1) % 2 == parity and good_cols[c]])
            if not len(cols):
                continue
            vals = gain_map[0, cols]
            vals = vals[vals > 0]
            if not len(vals):
                continue
            lo, hi = np.percentile(vals, [0.5, 99.5])
            ax.hist(vals, bins=n_bins, range=(lo, hi),
                    histtype="step", label=label, color=color, linewidth=1.5)
    else:
        vals = gain_map[0, good_cols]
        vals = vals[vals > 0]
        if len(vals):
            lo, hi = np.percentile(vals, [0.5, 99.5])
            ax.hist(vals, bins=n_bins, range=(lo, hi),
                    histtype="step", color="steelblue", linewidth=1.5)
    ax.set_xlabel("Gain [eV/ADU]")
    ax.set_ylabel("Columns")
    ax.set_title("Base gain — row 0\n(intrinsic column gain, CTE=1)")
    ax.legend()

    # Panel 2: gain vs row for sample columns
    ax = axes[1]
    good_col_idx = np.where(good_cols)[0]
    step         = max(1, len(good_col_idx) // 10)
    sample_cols  = good_col_idx[::step][:10]
    for col in sample_cols:
        g = np.where(gain_map[:, col] > 0, gain_map[:, col], np.nan)
        ax.plot(rows, g, lw=0.8, alpha=0.7)
    ax.set_xlabel("Row (Y)")
    ax.set_ylabel("Gain [eV/ADU]")
    ax.set_title("Gain vs row — sample columns\n"
                 "(increases with row due to CTE correction)")
    ax.grid(alpha=0.3)

    # Panel 3: all-pixel gain
    ax = axes[2]
    good_pixels = bad_gain_map == 0
    all_gains   = gain_map[good_pixels]
    all_gains   = all_gains[all_gains > 0]
    if len(all_gains):
        lo, hi = np.percentile(all_gains, [0.5, 99.5])
        ax.hist(all_gains, bins=n_bins, range=(lo, hi),
                histtype="stepfilled", color="steelblue", alpha=0.7)
    ax.set_xlabel("Gain [eV/ADU]")
    ax.set_ylabel("Pixels")
    ax.set_title("All-pixel gain distribution\n"
                 "(includes CTE variation across rows)")

    fig.tight_layout()
    fig.savefig(out_dir / "gain_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'gain_histogram.png'}")


def plot_gain_vs_row(
        gain_map:     np.ndarray,
        cte_map:      np.ndarray,
        bad_gain_map: np.ndarray,
        out_dir:      Path,
        n_sample:     int = 20,
) -> None:
    """
    Gain and CTE as a function of row — key diagnostic for CTI calibration.

    Panel 1: Normalised gain vs row (gain / gain[row=0])
             Should increase monotonically — slope = CTI per transfer
    Panel 2: Mean CTE vs row
             Should decrease monotonically from 1.0 at row=0
    """
    out_dir        = Path(out_dir)
    n_rows, n_cols = gain_map.shape
    rows           = np.arange(n_rows)

    good_cols    = bad_gain_map[0, :] == 0
    good_col_idx = np.where(good_cols)[0]
    step         = max(1, len(good_col_idx) // n_sample)
    sample_cols  = good_col_idx[::step][:n_sample]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: normalised gain vs row
    ax = axes[0]
    for col in sample_cols:
        g0 = gain_map[0, col]
        if g0 <= 0:
            continue
        g_norm = np.where(gain_map[:, col] > 0,
                          gain_map[:, col] / g0, np.nan)
        ax.plot(rows, g_norm, lw=0.6, alpha=0.5, color="steelblue")

    # Mean normalised gain across good columns
    if len(good_col_idx) > 0:
        g0_arr = gain_map[0, good_col_idx]
        safe   = g0_arr > 0
        if safe.any():
            g_matrix = gain_map[:, good_col_idx[safe]]
            g0_vec   = g0_arr[safe][np.newaxis, :]
            g_norm_m = np.where(g_matrix > 0, g_matrix / g0_vec, np.nan)
            mean_norm = np.nanmean(g_norm_m, axis=1)
            ax.plot(rows, mean_norm, lw=2.0, color="tomato",
                    label="mean over good columns")

    ax.set_xlabel("Row (Y)")
    ax.set_ylabel("Gain / Gain[row=0]")
    ax.set_title("Normalised gain vs row\n"
                 "(slope = CTI, should increase linearly)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: mean CTE vs row
    ax = axes[1]
    if len(good_col_idx) > 0:
        cte_matrix = np.where(cte_map[:, good_col_idx] > 0,
                               cte_map[:, good_col_idx], np.nan)
        cte_mean   = np.nanmean(cte_matrix, axis=1)
        ax.plot(rows, cte_mean, color="steelblue", lw=1.5,
                label="mean CTE (good columns)")
    ax.set_xlabel("Row (Y)")
    ax.set_ylabel("Cumulative CTE")
    ax.set_title("Mean CTE vs row\n"
                 "(decreases from 1.0 — more loss at higher rows)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "gain_vs_row.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'gain_vs_row.png'}")


def plot_column_peaks(
        col_peaks,
        n_cols:   int,
        out_dir:  Path,
) -> None:
    """Per-column peak positions and fallback flags."""
    out_dir = Path(out_dir)
    n_halves = col_peaks.ppos.shape[1]
    cols     = np.arange(n_cols)

    fig, axes = plt.subplots(n_halves, 1,
                              figsize=(12, 4 * n_halves), squeeze=False)
    labels = ["bottom half", "top half"] if n_halves == 2 else ["all rows"]

    for h in range(n_halves):
        ax      = axes[h, 0]
        good    = ~col_peaks.used_fallback[:, h]
        fallb   = col_peaks.used_fallback[:, h]
        ax.scatter(cols[good],  col_peaks.ppos[good,  h],
                   s=4, color="steelblue", label="fit", zorder=3)
        ax.scatter(cols[fallb], col_peaks.ppos[fallb, h],
                   s=4, color="tomato",    label="fallback", zorder=3)
        ax.set_xlabel("Column")
        ax.set_ylabel("Peak position [ADU]")
        ax.set_title(f"Per-column orientation peaks — {labels[h]}")
        ax.legend(markerscale=3)

    fig.tight_layout()
    fig.savefig(out_dir / "column_peaks.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir/'column_peaks.png'}")


def plot_grade_spectrum(
        energy_sum:   np.ndarray,
        grades:       np.ndarray,
        out_dir:      Path,
        e_min:        float = 0.0,
        e_max:        float = 8000.0,
        n_bins:       int   = 400,
) -> None:
    """Calibrated energy spectrum by grade group."""
    out_dir   = Path(out_dir)
    bin_edges = np.linspace(e_min, e_max, n_bins + 1)
    centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    groups = {
        "singles (g0)":   [0],
        "doubles (g1-4)": [1, 2, 3, 4],
        "triples (g5-8)": [5, 6, 7, 8],
        "quads (g9-12)":  [9, 10, 11, 12],
        "other (g13)":    [13],
    }
    colors = ["steelblue", "tomato", "seagreen", "orange", "grey"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for (label, gids), color in zip(groups.items(), colors):
        mask   = np.isin(grades, gids)
        if not mask.any():
            continue
        counts, _ = np.histogram(energy_sum[mask], bins=bin_edges)
        ax.step(centers, counts, where="mid",
                label=f"{label} (n={mask.sum():,})",
                color=color, linewidth=1.2)

    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("Counts")
    ax.set_title("Calibrated spectrum by grade")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "grade_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir/'grade_spectrum.png'}")
