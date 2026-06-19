"""
pnccd_ana.cli.calibration
==========================
Command-line entry point for Fe-55 gain and CTI calibration.

Reads the events.h5 produced by source_ana, runs the four-phase calibration
pipeline, and writes a gain_calibration.h5 containing all calibration
constants.  Diagnostic plots are saved alongside.

Usage
-----
  python -m pnccd_ana.cli.calibration analysis.yaml

Pipeline
--------
  Phase 1 : Rough global gain (even / odd columns) from single-pixel events
  Phase 2 : Preliminary energy reconstruction for all event grades
  Phase 3 : CTI coefficient estimation + correction (row-dependent)
  Phase 4 : Per-column fine-gain factors

Output HDF5 (gain_calibration.h5)
----------------------------------
  /gain/g_even              scalar float64 — eV/ADU for even columns
  /gain/g_odd               scalar float64 — eV/ADU for odd columns
  /gain/peak_even_adu       scalar float64 — fitted Kα peak [ADU], even pool
  /gain/peak_odd_adu        scalar float64 — fitted Kα peak [ADU], odd pool
  /cti/cti_coefficient      scalar float64 — CTI [1/pixel]
  /cti/e0                   scalar float64 — extrapolated peak at row=0 [eV]
  /cti/row_bins             (n_bins,) float64
  /cti/peak_per_bin         (n_bins,) float64 — measured Kα peak per row bin
  /cti/peak_success         (n_bins,) bool
  /column_gain/f_col        (n_cols,) float32 — per-column scale factors
  /column_gain/peak_col     (n_cols,) float32 — fitted Kα peak per column [eV]
  /column_gain/n_events_col (n_cols,) int32
  /column_gain/success_col  (n_cols,) bool
  /meta/...                 — config snapshot and run statistics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from ..config import Config
from ..lib.calibration import (
    MN_KALPHA_EV,
    MN_KBETA_EV,
    SINGLE_GRADES,
    SPLIT_GRADES,
    RoughGainCalibrator,
    CtiCalibrator,
    ColumnGainCalibrator,
    apply_rough_gain,
    apply_full_calibration,
    fit_peak,
    _gaussian,
    RoughGainResult,
    CtiResult,
    ColumnGainResult,
)
from ..lib.pattern_recognition import _GRADE_DEFS, GRADE_OTHER
from ..utils.plotting import _build_grade_palette, _build_group_label
from ..utils.io_h5 import load_events_h5


# ──────────────────────────────────────────────────────────────────────────────
# Config defaults (merged into Config._DEFAULTS via config.py stub)
# ──────────────────────────────────────────────────────────────────────────────

_CAL_DEFAULTS: dict = {
    "events_file":        None,   # input: events.h5 from source_ana
    "output_file":        None,   # output: gain_calibration.h5
    "target_ev":          MN_KALPHA_EV,
    "fit_window_frac":    0.20,   # Gaussian fit window ± fraction of target
    "rough_min_events":   100,    # Phase 1: min singles per parity pool
    "cti_row_bin_size":   64,     # Phase 3: rows per CTI bin
    "cti_min_events":     50,     # Phase 3: min events per row bin
    "cti_grade_filter":   None,   # None = all grades; list of ints to restrict
    "col_min_events":     30,     # Phase 4: min events per column
    "col_with_bg":        False,  # Phase 4: add linear background to Gaussian
    "col_grade_filter":   [0],    # Phase 4: grades used (default: singles only)
    "save_plots":         True,
}


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 save / load for calibration constants
# ──────────────────────────────────────────────────────────────────────────────

def save_gain_cal_h5(
        path:        str | Path,
        rough:       RoughGainResult,
        cti:         CtiResult,
        col:         ColumnGainResult,
        g_even:      float,
        g_odd:       float,
        energy_ev:   np.ndarray | None = None,
        metadata:    dict | None = None,
) -> None:
    """
    Write calibration constants to HDF5.

    Structure
    ---------
    /gain/g_even              scalar
    /gain/g_odd               scalar
    /gain/peak_even_adu       scalar
    /gain/peak_odd_adu        scalar
    /gain/n_singles           scalar
    /cti/cti_coefficient      scalar
    /cti/e0                   scalar
    /cti/row_bins             array
    /cti/peak_per_bin         array
    /cti/peak_success         array
    /cti/fit_residuals        array
    /column_gain/f_col        array
    /column_gain/peak_col     array
    /column_gain/n_events_col array
    /column_gain/success_col  array
    /calibrated_events/energy_ev  array (optional)
    /meta/...                 scalars/strings
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving gain calibration: {path}")

    with h5py.File(path, "w") as f:
        # ── Phase 1 ──────────────────────────────────────────────────────────
        gg = f.require_group("gain")
        gg.attrs["description"] = "Rough per-parity gain (Phase 1)"
        gg.create_dataset("g_even",        data=float(g_even))
        gg.create_dataset("g_odd",         data=float(g_odd))
        gg.create_dataset("peak_even_adu", data=float(rough.peak_even.peak_ev))
        gg.create_dataset("peak_odd_adu",  data=float(rough.peak_odd.peak_ev))
        gg.create_dataset("n_singles",     data=int(rough.n_singles))

        # ── Phase 3 ──────────────────────────────────────────────────────────
        cg = f.require_group("cti")
        cg.attrs["description"] = "CTI coefficient (Phase 3)"
        cg.create_dataset("cti_coefficient", data=float(cti.cti))
        cg.create_dataset("e0",              data=float(cti.e0))
        cg.create_dataset("row_bins",        data=cti.row_bins)
        cg.create_dataset("peak_per_bin",    data=cti.peak_per_bin)
        cg.create_dataset("peak_success",    data=cti.peak_success)
        cg.create_dataset("fit_residuals",   data=cti.fit_residuals)
        cg.attrs["n_bins_used"] = int(cti.n_bins_used)

        # ── Phase 4 ──────────────────────────────────────────────────────────
        fg = f.require_group("column_gain")
        fg.attrs["description"] = "Per-column fine-gain factors (Phase 4)"
        fg.create_dataset("f_col",         data=col.f_col,         compression="gzip")
        fg.create_dataset("peak_col",      data=col.peak_col,      compression="gzip")
        fg.create_dataset("n_events_col",  data=col.n_events_col,  compression="gzip")
        fg.create_dataset("success_col",   data=col.success_col,   compression="gzip")
        fg.attrs["n_cols_fit"] = int(col.n_cols_fit)

        # ── Calibrated energies ───────────────────────────────────────────────
        if energy_ev is not None:
            eg = f.require_group("calibrated_events")
            eg.attrs["description"] = "Fully calibrated event energies (Phase 1+3+4)"
            eg.create_dataset("energy_ev", data=energy_ev, compression="gzip")

        # ── Metadata ─────────────────────────────────────────────────────────
        mg = f.require_group("meta")
        mg.attrs["mn_kalpha_ev"] = MN_KALPHA_EV
        if metadata:
            for k, v in metadata.items():
                try:
                    mg.attrs[k] = v
                except TypeError:
                    mg.attrs[k] = str(v)

    print("  ✓ saved.")


def load_gain_cal_h5(path: str | Path) -> dict:
    """
    Load calibration constants from gain_calibration.h5.

    Returns
    -------
    dict with keys:
        g_even, g_odd, cti, e0, f_col, peak_col, success_col, n_events_col,
        peak_even_adu, peak_odd_adu, row_bins, peak_per_bin, peak_success,
        energy_ev (if available)
    """
    path = Path(path)
    out: dict = {}
    with h5py.File(path, "r") as f:
        out["g_even"]        = float(f["gain/g_even"][()])
        out["g_odd"]         = float(f["gain/g_odd"][()])
        out["peak_even_adu"] = float(f["gain/peak_even_adu"][()])
        out["peak_odd_adu"]  = float(f["gain/peak_odd_adu"][()])
        out["n_singles"]     = int(f["gain/n_singles"][()])
        out["cti"]           = float(f["cti/cti_coefficient"][()])
        out["e0"]            = float(f["cti/e0"][()])
        out["row_bins"]      = f["cti/row_bins"][:]
        out["peak_per_bin"]  = f["cti/peak_per_bin"][:]
        out["peak_success"]  = f["cti/peak_success"][:]
        out["f_col"]         = f["column_gain/f_col"][:]
        out["peak_col"]      = f["column_gain/peak_col"][:]
        out["n_events_col"]  = f["column_gain/n_events_col"][:]
        out["success_col"]   = f["column_gain/success_col"][:]

        # Calibrated energies (optional)
        if "calibrated_events/energy_ev" in f:
            out["energy_ev"] = f["calibrated_events/energy_ev"][:]

    print(f"  Loaded gain calibration from {path}")
    print(f"    g_even={out['g_even']:.4f}  g_odd={out['g_odd']:.4f}  "
          f"CTI={out['cti']:.3e}  cols_fit={out['success_col'].sum()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ──────────────────────────────────────────────────────────────────────────────

def _plot_rough_gain(rough: RoughGainResult,
                     events: np.ndarray,
                     out_dir: Path,
                     target_ev: float,
                     kalpha_adu: float,
                     kalpha_window: float) -> None:
    """Phase 1: ADU histograms for even/odd single-pixel events."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    singles   = events[np.isin(events["grade"], list(SINGLE_GRADES))]
    even_mask = (singles["X"] % 2) == 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 1 — Rough Gain: Single-pixel ADU spectra",
                 fontsize=13, fontweight="bold")

    lo_adu = kalpha_adu * (1.0 - kalpha_window)
    hi_adu = kalpha_adu * (1.0 + kalpha_window)

    for ax, mask, label, res, g in [
        (axes[0], even_mask,  "Even columns", rough.peak_even, rough.g_even),
        (axes[1], ~even_mask, "Odd columns",  rough.peak_odd,  rough.g_odd),
    ]:
        adu = singles["adu_sum"][mask]
        if len(adu) == 0:
            ax.set_title(f"{label} — no data")
            continue

        # Show a wider view (±40%) so the user can see the full peak context
        view_lo = kalpha_adu * 0.60
        view_hi = kalpha_adu * 1.40
        ax.hist(adu, bins=150, range=(view_lo, view_hi),
                color="steelblue", alpha=0.75, label=f"N={len(adu):,}")

        # Mark the fit window
        ax.axvspan(lo_adu, hi_adu, alpha=0.12, color="red",
                   label=f"Fit window [{lo_adu:.0f}, {hi_adu:.0f}]")

        if res.success:
            xs = np.linspace(lo_adu, hi_adu, 300)
            ys = _gaussian(xs, res.amplitude, res.peak_ev, res.sigma_ev)
            ax.plot(xs, ys, "r-", lw=2,
                    label=f"Kα fit: {res.peak_ev:.1f} ADU\n"
                          f"G = {g:.5f} eV/ADU\n"
                          f"σ = {res.sigma_ev:.1f} ADU")
            ax.axvline(res.peak_ev, color="red", lw=1, ls="--")
            ax.axvline(kalpha_adu,  color="gray", lw=1, ls=":",
                       label=f"kalpha_adu = {kalpha_adu:.0f}")

        ax.set_xlabel("ADU  (adu_sum, grade-0 single events)")
        ax.set_ylabel("Counts / bin")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Second x-axis in eV using the fitted gain
        if res.success and g > 0 and np.isfinite(g):
            ax2 = ax.twiny()
            ax2.set_xlim(np.array(ax.get_xlim()) * g)
            ax2.set_xlabel("Energy [eV]  (using fitted gain)", fontsize=8)

    plt.tight_layout()
    p = out_dir / "cal_phase1_rough_gain.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _plot_cti(cti_result: CtiResult, out_dir: Path) -> None:
    """Phase 3: peak position vs row and linear CTI fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Phase 3 — CTI Estimation  "
                 f"(CTI = {cti_result.cti:.3e} /pixel, "
                 f"E₀ = {cti_result.e0:.1f} eV)",
                 fontsize=13, fontweight="bold")

    rows  = cti_result.row_bins
    peaks = cti_result.peak_per_bin
    good  = cti_result.peak_success

    # Panel 1: peak vs row with linear fit
    ax = axes[0]
    ax.scatter(rows[good], peaks[good], s=30, color="steelblue",
               label="Measured Kα peak", zorder=3)
    if np.any(~good):
        ax.scatter(rows[~good], np.full((~good).sum(), np.nan),
                   s=15, color="lightgray", marker="x", label="Fit failed")

    r_model = np.linspace(rows.min(), rows.max(), 200)
    e_model = cti_result.e0 * (1.0 - r_model * cti_result.cti)
    ax.plot(r_model, e_model, "r-", lw=2,
            label=f"Linear fit\nE₀={cti_result.e0:.1f} eV\n"
                  f"CTI={cti_result.cti:.3e}")
    ax.set_xlabel("Row (Y)"); ax.set_ylabel("Kα peak position [eV]")
    ax.set_title("Peak position vs detector row")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: residuals
    ax2 = axes[1]
    resid = cti_result.fit_residuals
    ax2.scatter(rows[good], resid[good], s=30, color="darkorange")
    ax2.axhline(0, color="k", lw=1, ls="--")
    ax2.set_xlabel("Row (Y)"); ax2.set_ylabel("Residual [eV]")
    ax2.set_title("Fit residuals (measured − model)")
    ax2.grid(alpha=0.3)
    if np.any(np.isfinite(resid[good])):
        rms = float(np.nanstd(resid[good]))
        ax2.text(0.97, 0.97, f"RMS = {rms:.1f} eV",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    p = out_dir / "cal_phase3_cti.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _plot_column_gain(col_result: ColumnGainResult, out_dir: Path,
                      target_ev: float) -> None:
    """Phase 4: per-column gain factor map and distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f_col    = col_result.f_col
    good     = col_result.success_col
    n_cols   = len(f_col)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 4 — Per-Column Fine Gain Factors",
                 fontsize=13, fontweight="bold")

    # Panel 1: f_col vs column index
    ax = axes[0]
    cols = np.arange(n_cols)
    ax.scatter(cols[good],  f_col[good],  s=8, color="steelblue",
               alpha=0.7, label=f"Fitted ({good.sum()})")
    ax.scatter(cols[~good], f_col[~good], s=8, color="lightgray",
               alpha=0.5, label=f"Default=1 ({(~good).sum()})")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Column (X)"); ax.set_ylabel("f_col")
    ax.set_title("Column gain factor vs X")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: histogram of f_col (fitted only)
    ax2 = axes[1]
    if good.sum() > 0:
        vals = f_col[good]
        lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        ax2.hist(vals, bins=50, range=(max(lo, 0.5), min(hi, 2.0)),
                 color="steelblue", alpha=0.8)
    ax2.axvline(1.0, color="k", lw=1, ls="--", label="f=1 (no correction)")
    ax2.set_xlabel("f_col"); ax2.set_ylabel("Columns")
    ax2.set_title("Distribution of column gain factors")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Panel 3: fitted Kα peak per column
    ax3 = axes[2]
    pk = col_result.peak_col.copy()
    pk[~good] = np.nan
    ax3.scatter(cols[good], pk[good], s=8, color="darkorange", alpha=0.7)
    ax3.axhline(target_ev, color="k", lw=1, ls="--",
                label=f"Target {target_ev:.0f} eV")
    ax3.set_xlabel("Column (X)"); ax3.set_ylabel("Kα peak [eV]")
    ax3.set_title("Fitted Kα peak per column (before f_col correction)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / "cal_phase4_column_gain.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _plot_cti_per_col(events: np.ndarray,
                      e_cti: np.ndarray,
                      cti_result: CtiResult,
                      out_dir: Path,
                      target_ev: float,
                      n_rows: int,
                      row_bin_size: int = 64,
                      min_events_per_bin: int = 20) -> None:
    """
    Per-column CTI: for each column fit peak-vs-row slope and extract a
    local CTI coefficient.

    Because single columns have limited statistics, events are grouped
    into column bins (default 8 columns per bin) before fitting.
    The global CTI from Phase 3 is shown as a horizontal reference line.

    Panel 1 : local CTI vs column index
    Panel 2 : histogram of per-column-bin CTI values
    Panel 3 : 2-D heat map of peak position vs (col_bin, row_bin)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use singles only for this diagnostic (best energy resolution)
    s_mask = np.isin(events["grade"], list(SINGLE_GRADES))
    ev_s   = events[s_mask]
    ec_s   = e_cti[s_mask]

    n_cols_total = int(events["X"].max()) + 1

    # Choose column bin size so each bin has ~enough statistics
    # Default: 8 columns per bin (adjustable)
    col_bin_size = max(1, n_cols_total // 64)   # ≤64 bins across the detector
    col_edges    = np.arange(0, n_cols_total + col_bin_size, col_bin_size)
    col_centres  = 0.5 * (col_edges[:-1] + col_edges[1:])
    n_cbins      = len(col_centres)

    row_edges    = np.arange(0, n_rows + row_bin_size, row_bin_size)
    row_centres  = 0.5 * (row_edges[:-1] + row_edges[1:])
    n_rbins      = len(row_centres)

    # Arrays to fill
    cti_per_cbin = np.full(n_cbins, np.nan)
    cti_success  = np.zeros(n_cbins, dtype=bool)

    # 2-D peak map: (col_bin, row_bin)
    peak_map     = np.full((n_cbins, n_rbins), np.nan)

    x_coords = ev_s["X"].astype(int)
    y_coords = ev_s["Y"].astype(int)

    for ci, (x0, x1) in enumerate(zip(col_edges[:-1], col_edges[1:])):
        cx_mask = (x_coords >= x0) & (x_coords < x1)
        if cx_mask.sum() < min_events_per_bin * 2:
            continue

        row_peaks  = []
        row_rows   = []

        for ri, (y0, y1) in enumerate(zip(row_edges[:-1], row_edges[1:])):
            bm = cx_mask & (y_coords >= y0) & (y_coords < y1)
            if bm.sum() < min_events_per_bin:
                continue
            res = fit_peak(ec_s[bm], nominal=target_ev,
                           window_frac=0.12, n_bins=40,
                           min_events=min_events_per_bin)
            if res.success:
                row_peaks.append(res.peak_ev)
                row_rows.append(row_centres[ri])
                peak_map[ci, ri] = res.peak_ev

        if len(row_peaks) < 2:
            continue

        # Linear fit: peak = E0 * (1 - row * CTI_local)
        rr = np.array(row_rows)
        pp = np.array(row_peaks)
        coeffs = np.polyfit(rr, pp, 1)
        slope, e0_local = coeffs
        cti_local = -slope / e0_local if e0_local != 0 else 0.0
        cti_per_cbin[ci] = cti_local
        cti_success[ci]  = True

    # Global CTI for reference
    global_cti = cti_result.cti
    valid_cti  = cti_per_cbin[cti_success]
    mean_cti   = float(np.nanmean(valid_cti)) if len(valid_cti) else global_cti

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f"CTI per Column Bin  "
        f"(global CTI = {global_cti:.3e} /pixel,  "
        f"mean local = {mean_cti:.3e} /pixel)",
        fontsize=13, fontweight="bold")

    # ── Panel 1: local CTI vs column bin centre ───────────────────────────────
    ax = axes[0]
    good_c = col_centres[cti_success]
    good_v = cti_per_cbin[cti_success]

    ax.scatter(good_c, good_v, s=20, color="steelblue", zorder=3,
               label=f"Local CTI  ({cti_success.sum()} col bins)")
    ax.axhline(global_cti, color="red", lw=1.5, ls="--",
               label=f"Global CTI = {global_cti:.3e}")
    ax.axhline(mean_cti,   color="darkorange", lw=1.2, ls="-.",
               label=f"Mean local = {mean_cti:.3e}")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Column (X)")
    ax.set_ylabel("CTI  [1/pixel]")
    ax.set_title("Local CTI per column bin\n"
                 f"(each bin = {col_bin_size} cols × all rows)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if len(valid_cti) > 1:
        ax.text(0.97, 0.97,
                f"mean = {mean_cti:.3e}\n"
                f"std  = {float(np.nanstd(valid_cti)):.3e}\n"
                f"min  = {float(np.nanmin(valid_cti)):.3e}\n"
                f"max  = {float(np.nanmax(valid_cti)):.3e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # ── Panel 2: histogram of local CTI values ────────────────────────────────
    ax2 = axes[1]
    if len(valid_cti) > 1:
        lo_c = float(np.percentile(valid_cti, 2))
        hi_c = float(np.percentile(valid_cti, 98))
        margin = (hi_c - lo_c) * 0.4
        ax2.hist(valid_cti, bins=min(30, len(valid_cti)),
                 range=(lo_c - margin, hi_c + margin),
                 color="steelblue", alpha=0.75,
                 label=f"N = {len(valid_cti)} col bins")
        ax2.axvline(mean_cti,   color="darkorange", lw=1.5, ls="-.",
                    label=f"Mean = {mean_cti:.3e}")
        ax2.axvline(global_cti, color="red", lw=1.5, ls="--",
                    label=f"Global = {global_cti:.3e}")
        ax2.axvline(0, color="k", lw=0.5, ls=":")
    ax2.set_xlabel("CTI  [1/pixel]")
    ax2.set_ylabel("Column bins")
    ax2.set_title("Distribution of local CTI values")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: 2-D heat map: peak position vs (col_bin, row_bin) ────────────
    ax3 = axes[2]
    # peak_map shape: (n_cbins, n_rbins) — transpose for imshow (row_bin on Y)
    img_data = peak_map.T   # shape (n_rbins, n_cbins)
    finite = img_data[np.isfinite(img_data)]
    if len(finite):
        vlo = float(np.percentile(finite, 2))
        vhi = float(np.percentile(finite, 98))
        im = ax3.imshow(img_data, origin="lower", aspect="auto",
                        cmap="RdYlGn",
                        vmin=vlo, vmax=vhi,
                        extent=[col_edges[0], col_edges[-1],
                                row_edges[0], row_edges[-1]])
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04,
                     label="Kα peak [eV]")
    ax3.set_xlabel("Column (X)")
    ax3.set_ylabel("Row (Y)")
    ax3.set_title("Kα peak position [eV]\nvs (column bin, row bin)\n"
                  "(after CTI correction — should be uniform)")

    # Annotate global CTI on panel 3
    ax3.text(0.02, 0.98,
             f"Global CTI = {global_cti:.3e} /pixel\n"
             f"Mean local = {mean_cti:.3e} /pixel",
             transform=ax3.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    plt.tight_layout()
    p = out_dir / "cal_cti_per_col.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _plot_pixel_gain_map(rough: RoughGainResult,
                         col_result: ColumnGainResult,
                         out_dir: Path,
                         target_ev: float) -> None:
    """
    Histogram of the effective per-pixel gain across all columns.

    The effective gain for pixel at column X is:
        G_eff(X) = G_rough(parity) × f_col(X)   [eV/ADU]

    This combines the even/odd rough gain from Phase 1 with the
    per-column fine-tuning factor from Phase 4.

    Panel 1 : G_eff vs column index  (scatter, coloured by parity)
    Panel 2 : Histogram of G_eff for all columns
    Panel 3 : f_col vs column index  (Phase 4 fine factor only)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols   = len(col_result.f_col)
    cols     = np.arange(n_cols)
    parity   = cols % 2   # 0=even, 1=odd

    g_rough  = np.where(parity == 0,
                        float(rough.g_even),
                        float(rough.g_odd)).astype(np.float64)
    g_eff    = g_rough * col_result.f_col.astype(np.float64)

    good     = col_result.success_col
    g_fitted = g_eff[good]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Per-Pixel Gain Map  (G_rough × f_col)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: G_eff vs column ──────────────────────────────────────────────
    ax = axes[0]
    even_cols = cols[(parity == 0) & good]
    odd_cols  = cols[(parity == 1) & good]
    ax.scatter(even_cols, g_eff[(parity == 0) & good],
               s=6, color="#2176ae", alpha=0.7, label="Even cols")
    ax.scatter(odd_cols,  g_eff[(parity == 1) & good],
               s=6, color="#f7931e", alpha=0.7, label="Odd cols")
    # Show unfitted columns as grey
    if (~good).any():
        ax.scatter(cols[~good], g_eff[~good],
                   s=6, color="lightgray", alpha=0.5, label="Default (no fit)")
    ax.axhline(float(rough.g_even), color="#2176ae", lw=1, ls="--",
               label=f"G_even = {rough.g_even:.5f}")
    ax.axhline(float(rough.g_odd),  color="#f7931e", lw=1, ls="--",
               label=f"G_odd  = {rough.g_odd:.5f}")
    ax.set_xlabel("Column (X)")
    ax.set_ylabel("G_eff  [eV / ADU]")
    ax.set_title("Effective gain per column")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    if len(g_fitted):
        mean_g = float(g_fitted.mean())
        std_g  = float(g_fitted.std())
        ax.text(0.97, 0.03,
                f"mean = {mean_g:.5f} eV/ADU\n"
                f"σ    = {std_g:.5f} eV/ADU\n"
                f"σ/μ  = {std_g/mean_g*100:.2f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # ── Panel 2: Histogram of G_eff ───────────────────────────────────────────
    ax2 = axes[1]
    if len(g_fitted) > 0:
        lo_g = float(np.percentile(g_fitted, 1))
        hi_g = float(np.percentile(g_fitted, 99))
        margin = (hi_g - lo_g) * 0.3
        lo_g = max(lo_g - margin, 0)
        hi_g = hi_g + margin

        # Split by parity for stacked histogram
        g_even_fit = g_eff[(parity == 0) & good]
        g_odd_fit  = g_eff[(parity == 1) & good]

        bins_g = np.linspace(lo_g, hi_g, 60)
        ax2.hist(g_even_fit, bins=bins_g, color="#2176ae", alpha=0.65,
                 label=f"Even (N={len(g_even_fit)})")
        ax2.hist(g_odd_fit,  bins=bins_g, color="#f7931e", alpha=0.65,
                 label=f"Odd  (N={len(g_odd_fit)})")

        # Combined stats
        mean_g = float(g_fitted.mean())
        std_g  = float(g_fitted.std())
        ax2.axvline(mean_g, color="black", lw=1.5, ls="-",
                    label=f"Mean = {mean_g:.5f}")
        ax2.axvline(float(rough.g_even), color="#2176ae", lw=1.2, ls="--",
                    label=f"G_even = {rough.g_even:.5f}")
        ax2.axvline(float(rough.g_odd),  color="#f7931e", lw=1.2, ls="--",
                    label=f"G_odd  = {rough.g_odd:.5f}")
        ax2.text(0.97, 0.97,
                 f"All fitted columns:\n"
                 f"mean = {mean_g:.5f} eV/ADU\n"
                 f"std  = {std_g:.5f} eV/ADU\n"
                 f"σ/μ  = {std_g/mean_g*100:.3f}%",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    ax2.set_xlabel("G_eff  [eV / ADU]")
    ax2.set_ylabel("Columns")
    ax2.set_title("Gain distribution across all columns")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: f_col fine factor ────────────────────────────────────────────
    ax3 = axes[2]
    ax3.scatter(cols[good],  col_result.f_col[good],
                s=6, color="steelblue", alpha=0.7,
                label=f"Fitted ({good.sum()} cols)")
    if (~good).any():
        ax3.scatter(cols[~good], col_result.f_col[~good],
                    s=6, color="lightgray", alpha=0.5,
                    label=f"Default=1 ({(~good).sum()} cols)")
    ax3.axhline(1.0, color="k", lw=0.8, ls="--", label="f=1 (no correction)")

    if good.sum() > 0:
        f_vals = col_result.f_col[good]
        ax3.text(0.97, 0.03,
                 f"mean = {f_vals.mean():.4f}\n"
                 f"std  = {f_vals.std():.4f}\n"
                 f"min  = {f_vals.min():.4f}\n"
                 f"max  = {f_vals.max():.4f}",
                 transform=ax3.transAxes, ha="right", va="bottom", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    ax3.set_xlabel("Column (X)")
    ax3.set_ylabel("f_col  (fine gain factor)")
    ax3.set_title("Phase-4 per-column fine factor")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / "cal_pixel_gain_map.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _plot_final_spectrum(events: np.ndarray,
                         energy_ev: np.ndarray,
                         out_dir: Path,
                         target_ev: float) -> None:
    """
    Calibrated energy spectrum for ALL grades with Kα resolution fit.

    Grade groups, colours and labels are derived entirely from
    _GRADE_DEFS + GRADE_OTHER at call time — adding or removing grades
    in pattern_recognition.py automatically updates this plot.

    Panel 1 (log scale)  : per-grade-group spectra
    Panel 2 (linear)     : all-grades sum with Gaussian fit to Kα peak
                           → energy resolution FWHM and R = FWHM/E
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..lib.pattern_recognition import _GRADE_DEFS, GRADE_OTHER
    from ..utils.plotting import _build_grade_palette, _build_group_label

    # Both dicts are keyed by the *first* grade ID in each group.
    # _build_grade_palette : grade_id → colour string
    # _build_group_label   : first_grade_id_in_group → label string
    palette     = _build_grade_palette()   # {grade_id: colour}
    group_label = _build_group_label()     # {first_gid: label_str}

    # Build ordered list of (group_key, [grade_ids], colour, label)
    # by grouping _GRADE_DEFS entries by their label prefix, then appending
    # GRADE_OTHER — identical logic to _build_group_label() so the two
    # are always in sync.
    from collections import defaultdict
    prefix_to_gids: dict[str, list[int]] = defaultdict(list)
    for gid, label, _ in _GRADE_DEFS:
        prefix = label.split()[0]
        prefix_to_gids[prefix].append(gid)

    # Ordered list of groups: [(first_gid, all_gids_in_group)]
    groups: list[tuple[int, list[int]]] = []
    for prefix, gids in sorted(prefix_to_gids.items(),
                                key=lambda kv: min(kv[1])):
        gids_sorted = sorted(gids)
        groups.append((gids_sorted[0], gids_sorted))
    # Append the catch-all "other" group
    groups.append((GRADE_OTHER, [GRADE_OTHER]))

    # Energy axis
    lo      = target_ev * 0.60
    hi      = MN_KBETA_EV * 1.30
    bins    = np.linspace(lo, hi, 350)
    centres = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Final Calibrated Fe-55 Spectrum — All Grades",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: per-group log scale ──────────────────────────────────────────
    ax1 = axes[0]
    all_counts = np.zeros(len(centres), dtype=np.float64)

    for first_gid, gids in groups:
        mask = np.isin(events["grade"], gids)
        if not mask.any():
            continue
        e    = energy_ev[mask]
        c, _ = np.histogram(e, bins=bins)
        all_counts += c.astype(np.float64)

        colour    = palette.get(first_gid, "#aaaaaa")
        lbl       = group_label.get(first_gid, f"G{first_gid}")
        n_ev      = int(mask.sum())
        ax1.step(centres, c, where="mid", color=colour,
                 lw=1.1, alpha=0.85,
                 label=f"{lbl}  N={n_ev:,}")

    ax1.step(centres, all_counts, where="mid", color="black",
             lw=1.3, ls="--", alpha=0.7,
             label=f"All grades  N={len(events):,}")
    ax1.axvline(target_ev,   color="red",  lw=1.2, ls="--",
                label=f"Mn Kα {target_ev:.0f} eV")
    ax1.axvline(MN_KBETA_EV, color="blue", lw=1.2, ls="--",
                label=f"Mn Kβ {MN_KBETA_EV:.0f} eV")
    ax1.set_xlabel("Energy [eV]")
    ax1.set_ylabel("Counts / bin")
    ax1.set_title("Per-grade group  (log scale)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)

    # ── Panel 2: all-grades sum + Kα Gaussian fit ─────────────────────────────
    ax2 = axes[1]
    ax2.step(centres, all_counts, where="mid", color="steelblue",
             lw=1.2, alpha=0.9,
             label=f"All grades  N={len(events):,}")

    fit_window = 0.12
    res = fit_peak(energy_ev, nominal=target_ev,
                   window_frac=fit_window, n_bins=150,
                   min_events=50, with_bg=False)

    if res.success:
        fwhm   = 2.3548 * res.sigma_ev
        resoln = fwhm / res.peak_ev * 100.0

        xs = np.linspace(target_ev * (1 - fit_window),
                         target_ev * (1 + fit_window), 500)
        ys = _gaussian(xs, res.amplitude, res.peak_ev, res.sigma_ev)
        ax2.plot(xs, ys, "r-", lw=2.5,
                 label=(f"Gaussian fit\n"
                        f"Peak = {res.peak_ev:.1f} eV\n"
                        f"σ    = {res.sigma_ev:.1f} eV\n"
                        f"FWHM = {fwhm:.1f} eV\n"
                        f"R    = {resoln:.2f}%"))
        ax2.axvline(res.peak_ev, color="red", lw=1, ls="--", alpha=0.6)
        half_max = res.amplitude / 2.0
        ax2.hlines(half_max,
                   res.peak_ev - res.sigma_ev * 1.1774,
                   res.peak_ev + res.sigma_ev * 1.1774,
                   colors="red", lw=1.5, ls=":",
                   label=f"FWHM = {fwhm:.1f} eV")
        ax2.axvspan(target_ev * (1 - fit_window),
                    target_ev * (1 + fit_window),
                    alpha=0.07, color="red", label="Fit window")

        print(f"\n  ── Kα energy resolution ──")
        print(f"     Peak  = {res.peak_ev:.2f} eV")
        print(f"     σ     = {res.sigma_ev:.2f} eV")
        print(f"     FWHM  = {fwhm:.2f} eV")
        print(f"     R     = {resoln:.3f}%")
        print(f"     N_fit = {res.n_events:,} events in fit window")
    else:
        print(f"  ⚠ Kα resolution fit failed: {res.message}")

    ax2.axvline(target_ev,   color="red",  lw=1.2, ls="--", alpha=0.5)
    ax2.axvline(MN_KBETA_EV, color="blue", lw=1.2, ls="--", alpha=0.5,
                label=f"Mn Kβ {MN_KBETA_EV:.0f} eV")
    ax2.set_xlabel("Energy [eV]")
    ax2.set_ylabel("Counts / bin")
    ax2.set_title("All-grades sum  (linear scale) + Kα resolution fit")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(lo, hi)

    plt.tight_layout()
    p = out_dir / "cal_final_spectrum.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _plot_cti_correction_check(events: np.ndarray,
                                e_prelim: np.ndarray,
                                e_cti: np.ndarray,
                                cti_result: CtiResult,
                                out_dir: Path) -> None:
    """Before/after CTI correction: peak-vs-row comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CTI Correction Check — Peak position vs Row",
                 fontsize=13, fontweight="bold")

    s_mask = np.isin(events["grade"], list(SINGLE_GRADES))

    for ax, ep, label, colour in [
        (axes[0], e_prelim[s_mask], "Before CTI correction", "steelblue"),
        (axes[1], e_cti[s_mask],   "After CTI correction",  "darkorange"),
    ]:
        rows_s = events["Y"][s_mask].astype(int)
        bin_size = cti_result.row_bins[1] - cti_result.row_bins[0] if len(cti_result.row_bins) > 1 else 64
        bin_edges = np.arange(0, rows_s.max() + bin_size + 1, bin_size)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        pk_arr  = []
        row_arr = []
        for y0, y1, rc in zip(bin_edges[:-1], bin_edges[1:], bin_centres):
            bm = (rows_s >= y0) & (rows_s < y1)
            if bm.sum() < 30:
                continue
            res = fit_peak(ep[bm], nominal=MN_KALPHA_EV,
                           window_frac=0.15, n_bins=50, min_events=30)
            if res.success:
                pk_arr.append(res.peak_ev)
                row_arr.append(rc)

        if row_arr:
            ax.scatter(row_arr, pk_arr, s=25, color=colour, zorder=3)
        ax.axhline(MN_KALPHA_EV, color="k", lw=1, ls="--",
                   label=f"Mn Kα = {MN_KALPHA_EV:.0f} eV")
        ax.set_xlabel("Row (Y)")
        ax.set_ylabel("Kα peak [eV]")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        spread = float(np.std(pk_arr)) if len(pk_arr) > 1 else 0.0
        ax.text(0.97, 0.03, f"σ = {spread:.1f} eV",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    p = out_dir / "cal_cti_correction_check.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: Config) -> dict:
    """
    Execute the gain + CTI calibration pipeline from a Config object.

    Reads  : {output_dir}/events.h5  (or gain_calibration.events_file in YAML)
    Writes : {output_dir}/gain_calibration.h5
             {output_dir}/cal_phase1_rough_gain.png
             {output_dir}/cal_phase3_cti.png
             {output_dir}/cal_phase4_column_gain.png
             {output_dir}/cal_cti_correction_check.png
             {output_dir}/cal_final_spectrum.png

    Returns dict with all intermediate and final results.
    """
    gc      = {**_CAL_DEFAULTS, **(cfg.gain_calibration or {})}
    gen     = cfg.general
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ev    = float(gc["target_ev"])
    window_frac  = float(gc["fit_window_frac"])

    # ── Validate kalpha_adu (must be set by user) ─────────────────────────────
    kalpha_adu = gc.get("kalpha_adu")
    if kalpha_adu is None:
        raise ValueError(
            "gain_calibration.kalpha_adu is not set in your analysis.yaml.\n"
            "  Open your source_ana spectrum plot (spectrum_full_detector.png),\n"
            "  read the Kα peak position in ADU for single-pixel events,\n"
            "  and add it to your config:\n\n"
            "    gain_calibration:\n"
            "      kalpha_adu: 15000    # your value here\n")
    kalpha_adu     = float(kalpha_adu)
    kalpha_window  = float(gc.get("kalpha_adu_window", 0.20))

    print(f"  Kα peak (user-supplied): {kalpha_adu:.0f} ADU  "
          f"(window ±{kalpha_window:.0%})")
    print(f"  Target energy: {target_ev:.1f} eV  →  "
          f"expected gain ≈ {target_ev/kalpha_adu:.4f} eV/ADU")

    # ── Resolve input events file ─────────────────────────────────────────────
    events_file = gc.get("events_file") or None
    if events_file:
        events_path = cfg.resolve_input_path(events_file)
        if not (events_path and events_path.exists()):
            events_path = cfg.resolve_output_path(events_file)
    else:
        events_path = out_dir / "events.h5"

    if not events_path or not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found: {events_path}\n"
            "Run source_ana first, or set gain_calibration.events_file in config.")

    print(f"\nLoading events from: {events_path}")
    data    = load_events_h5(events_path)
    events  = data["events"]
    n_total = len(events)

    if n_total == 0:
        raise RuntimeError("No events found in events file.")

    # Determine detector dimensions from the event coordinates
    n_rows = int(events["Y"].max()) + 1
    n_cols = int(events["X"].max()) + 1
    # Prefer geometry from general config if available
    if gen.get("frame_rows"):
        n_rows = int(gen["frame_rows"])
    if gen.get("frame_cols"):
        n_cols = int(gen["frame_cols"])

    print(f"  {n_total:,} events  detector {n_rows}×{n_cols}")
    print(f"  Grade distribution: "
          + "  ".join(f"G{g}={int((events['grade']==g).sum())}"
                      for g in sorted(np.unique(events["grade"]))))

    target_ev    = float(gc["target_ev"])
    window_frac  = float(gc["fit_window_frac"])

    # ── Phase 1: Rough gain ───────────────────────────────────────────────────
    print("\n── Phase 1: Rough global gain (even / odd columns) ──")
    rough_cal = RoughGainCalibrator(
        kalpha_adu  = kalpha_adu,
        target_ev   = target_ev,
        window_frac = kalpha_window,
        n_bins      = 100,
        min_events  = int(gc["rough_min_events"]),
    )
    rough = rough_cal.run(events)

    print(f"  Singles used: {rough.n_singles:,}")
    print(f"  Even: peak={rough.peak_even.peak_ev:.2f} ADU  "
          f"G_even={rough.g_even:.5f} eV/ADU  "
          f"({rough.peak_even.message if not rough.peak_even.success else 'ok'})")
    print(f"  Odd:  peak={rough.peak_odd.peak_ev:.2f} ADU  "
          f"G_odd={rough.g_odd:.5f} eV/ADU  "
          f"({rough.peak_odd.message if not rough.peak_odd.success else 'ok'})")

    if np.isnan(rough.g_even) or np.isnan(rough.g_odd):
        raise RuntimeError(
            "Phase 1 gain fit failed.  Check that events.h5 contains "
            "single-pixel events near the expected Kα peak.  "
            "Adjust fit_window_frac or rough_min_events in config.")

    # ── Phase 2: Preliminary energy reconstruction ────────────────────────────
    print("\n── Phase 2: Preliminary energy reconstruction ──")
    e_prelim = apply_rough_gain(events, rough.g_even, rough.g_odd)
    print(f"  e_prelim: min={e_prelim.min():.0f}  "
          f"max={e_prelim.max():.0f}  "
          f"median={float(np.median(e_prelim)):.0f}  eV")

    # ── Phase 3: CTI estimation and correction ────────────────────────────────
    print("\n── Phase 3: CTI estimation ──")
    cti_grade_filter = gc.get("cti_grade_filter")
    if cti_grade_filter is not None:
        cti_grade_filter = [int(g) for g in cti_grade_filter]

    cti_cal = CtiCalibrator(
        row_bin_size = int(gc["cti_row_bin_size"]),
        target_ev    = target_ev,
        window_frac  = window_frac * 0.75,   # tighter window in eV space
        n_bins_hist  = 60,
        min_events   = int(gc["cti_min_events"]),
        grade_filter = cti_grade_filter,
    )
    cti_result = cti_cal.estimate(events, e_prelim, n_rows)

    print(f"  CTI = {cti_result.cti:.4e} /pixel")
    print(f"  E₀  = {cti_result.e0:.2f} eV  (extrapolated peak at row=0)")
    print(f"  Bins used: {cti_result.n_bins_used} / {len(cti_result.row_bins)}")
    if abs(cti_result.cti) < 1e-8:
        print("  ⚠  CTI is essentially zero — either the detector has "
              "negligible CTI or there are insufficient statistics per row bin.")

    e_cti = cti_cal.correct(events, e_prelim, cti_result.cti, cti_result.e0)
    print(f"  e_cti: min={e_cti.min():.0f}  "
          f"max={e_cti.max():.0f}  "
          f"median={float(np.median(e_cti)):.0f}  eV")

    # ── Phase 4: Per-column fine gain ─────────────────────────────────────────
    print("\n── Phase 4: Per-column fine-gain calibration ──")
    col_grade_filter = gc.get("col_grade_filter")
    if col_grade_filter is not None:
        col_grade_filter = [int(g) for g in col_grade_filter]

    col_cal = ColumnGainCalibrator(
        target_ev    = target_ev,
        window_frac  = window_frac * 0.75,
        n_bins       = 60,
        min_events   = int(gc["col_min_events"]),
        with_bg      = bool(gc["col_with_bg"]),
    )
    col_result = col_cal.run(events, e_cti, n_cols,
                             grade_filter=col_grade_filter)

    print(f"  Columns fitted: {col_result.n_cols_fit} / {n_cols}")
    fitted = col_result.f_col[col_result.success_col]
    if len(fitted):
        print(f"  f_col: mean={fitted.mean():.4f}  "
              f"std={fitted.std():.4f}  "
              f"min={fitted.min():.4f}  max={fitted.max():.4f}")

    # ── Final calibrated energies ─────────────────────────────────────────────
    energy_ev = apply_full_calibration(
        events, rough.g_even, rough.g_odd, cti_result.cti, col_result.f_col)

    # Quick sanity check: singles Kα peak after full calibration
    s_mask = events["grade"] == 0
    if s_mask.sum() > 50:
        chk = fit_peak(energy_ev[s_mask], nominal=target_ev,
                  window_frac=0.10, n_bins=80, min_events=30)
        if chk.success:
            print(f"\n  ✓ Final singles Kα peak: {chk.peak_ev:.1f} eV  "
                  f"(target {target_ev:.0f} eV, "
                  f"Δ={chk.peak_ev - target_ev:+.1f} eV)")
        else:
            print(f"\n  ⚠ Final Kα peak fit failed: {chk.message}")

    # ── Save calibration constants ─────────────────────────────────────────────
    output_file = gc.get("output_file")
    if output_file:
        out_path = cfg.resolve_output_path(output_file)
    else:
        out_path = out_dir / "gain_calibration.h5"

    save_gain_cal_h5(
        out_path, rough, cti_result, col_result,
        g_even=rough.g_even, g_odd=rough.g_odd,
        energy_ev=energy_ev,
        metadata={
            **gen.get("metadata", {}),
            "events_file":       str(events_path),
            "n_events":          int(n_total),
            "n_rows":            int(n_rows),
            "n_cols":            int(n_cols),
            "target_ev":         float(target_ev),
            "cti_row_bin_size":  int(gc["cti_row_bin_size"]),
            "col_grade_filter":  str(col_grade_filter),
        },
    )

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    if gen.get("save_frame_plots", True) and gc.get("save_plots", True):
        print("\nGenerating calibration diagnostic plots …")
        _plot_rough_gain(rough, events, out_dir, target_ev, kalpha_adu, kalpha_window)
        _plot_cti(cti_result, out_dir)
        _plot_cti_correction_check(events, e_prelim, e_cti, cti_result, out_dir)
        _plot_column_gain(col_result, out_dir, target_ev)
        _plot_final_spectrum(events, energy_ev, out_dir, target_ev)
        _plot_pixel_gain_map(rough, col_result, out_dir, target_ev)
        _plot_cti_per_col(                                      
            events, e_cti, cti_result, out_dir, target_ev,
            n_rows=n_rows,
            row_bin_size=int(gc["cti_row_bin_size"]),
            min_events_per_bin=max(10, int(gc["cti_min_events"]) // 3),
        )

    print(f"\n✓ Gain + CTI calibration complete.  Output: {out_dir}/")

    return dict(
        rough=rough, cti=cti_result, col=col_result,
        e_prelim=e_prelim, e_cti=e_cti, energy_ev=energy_ev,
        g_even=rough.g_even, g_odd=rough.g_odd,
        cti_coeff=cti_result.cti, f_col=col_result.f_col,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="pnCCD Fe-55 gain and CTI calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("config", help="Path to analysis.yaml")
    args = parser.parse_args(argv)
    cfg  = Config.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
