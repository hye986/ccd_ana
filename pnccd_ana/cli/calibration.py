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
    RoughGainCalibrator,
    CtiCalibrator,
    ColumnGainCalibrator,
    apply_rough_gain,
    apply_full_calibration,
    RoughGainResult,
    CtiResult,
    ColumnGainResult,
)
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
        peak_even_adu, peak_odd_adu, row_bins, peak_per_bin, peak_success
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
                     target_ev: float) -> None:
    """Phase 1: ADU histograms for even/odd single-pixel events."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..lib.calibration import SINGLE_GRADES
    singles = events[np.isin(events["grade"], list(SINGLE_GRADES))]
    even_mask = (singles["X"] % 2) == 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 1 — Rough Gain: Single-pixel ADU spectra",
                 fontsize=13, fontweight="bold")

    for ax, mask, label, res, g in [
        (axes[0], even_mask,  "Even columns",
         rough.peak_even, rough.g_even),
        (axes[1], ~even_mask, "Odd columns",
         rough.peak_odd,  rough.g_odd),
    ]:
        adu = singles["adu_sum"][mask]
        if len(adu) == 0:
            ax.set_title(f"{label} — no data")
            continue
        lo = target_ev * 0.5 / g if (g and np.isfinite(g)) else float(np.percentile(adu, 1))
        hi = target_ev * 1.5 / g if (g and np.isfinite(g)) else float(np.percentile(adu, 99))
        ax.hist(adu, bins=120, range=(max(lo, 0), hi),
                color="steelblue", alpha=0.75, label=f"N={len(adu):,}")
        if res.success:
            xs = np.linspace(max(lo, 0), hi, 300)
            from ..lib.calibration import _gaussian
            ys = _gaussian(xs, res.amplitude, res.peak_ev, res.sigma_ev)
            ax.plot(xs, ys, "r-", lw=2,
                    label=f"Kα fit: {res.peak_ev:.1f} ADU\n"
                          f"G={g:.4f} eV/ADU")
            ax.axvline(res.peak_ev, color="red", lw=1, ls="--")
        ax.set_xlabel("ADU (adu_sum, single events)")
        ax.set_ylabel("Counts / bin")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

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


def _plot_final_spectrum(events: np.ndarray,
                         energy_ev: np.ndarray,
                         out_dir: Path,
                         target_ev: float) -> None:
    """Calibrated energy spectrum overlaid with Kα / Kβ markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..lib.calibration import MN_KBETA_EV, SINGLE_GRADES, SPLIT_GRADES

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Final Calibrated Fe-55 Spectrum",
                 fontsize=13, fontweight="bold")

    lo = target_ev * 0.6
    hi = MN_KBETA_EV * 1.3
    bins = np.linspace(lo, hi, 300)

    grade_groups = {
        "singles (G0)":    np.isin(events["grade"], list(SINGLE_GRADES)),
        "doubles (G1-4)":  np.isin(events["grade"], [1,2,3,4]),
        "triples (G5-8)":  np.isin(events["grade"], [5,6,7,8]),
        "quads (G9-12)":   np.isin(events["grade"], [9,10,11,12]),
    }
    colours = ["#2176ae", "#f7931e", "#57cc99", "#c77dff"]

    for ax, yscale in zip(axes, ["log", "linear"]):
        for (label, mask), col in zip(grade_groups.items(), colours):
            e = energy_ev[mask]
            if len(e) == 0:
                continue
            c, _ = np.histogram(e, bins=bins)
            centres = 0.5 * (bins[:-1] + bins[1:])
            ax.step(centres, c, where="mid", color=col, lw=1.0,
                    alpha=0.85, label=label)

        ax.axvline(target_ev,  color="red",  lw=1.2, ls="--",
                   label=f"Mn Kα {target_ev:.0f} eV")
        ax.axvline(MN_KBETA_EV, color="blue", lw=1.2, ls="--",
                   label=f"Mn Kβ {MN_KBETA_EV:.0f} eV")
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Counts / bin")
        ax.set_title(f"{'Log' if yscale == 'log' else 'Linear'} scale")
        ax.set_yscale(yscale)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

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

    from ..lib.calibration import SINGLE_GRADES, fit_peak, MN_KALPHA_EV
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
        target_ev   = target_ev,
        window_frac = window_frac,
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
        from ..lib.calibration import fit_peak as _fp
        chk = _fp(energy_ev[s_mask], nominal=target_ev,
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

    # Also save final energies as numpy array for quick downstream use
    np.save(out_dir / "energy_ev.npy", energy_ev)
    print(f"  → energy_ev.npy  ({len(energy_ev):,} events)")

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    if gen.get("save_frame_plots", True) and gc.get("save_plots", True):
        print("\nGenerating calibration diagnostic plots …")
        _plot_rough_gain(rough, events, out_dir, target_ev)
        _plot_cti(cti_result, out_dir)
        _plot_cti_correction_check(events, e_prelim, e_cti, cti_result, out_dir)
        _plot_column_gain(col_result, out_dir, target_ev)
        _plot_final_spectrum(events, energy_ev, out_dir, target_ev)

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
