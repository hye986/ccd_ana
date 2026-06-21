"""
pnccd_ana.plotting.cti_plots
============================
CTI calibration plotting functions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import _cb, _stats_box, _build_grade_palette, _build_group_label
from .gain_plots import plot_rough_gain, plot_pixel_gain_map
from .spectrum_plots import plot_final_spectrum, _compute_cti_check_data
from ..physics.gain import (RoughGainResult, ColumnGainResult,
                             PeakFitResult, fit_peak, MN_KALPHA_EV, MN_KBETA_EV)
from ..physics.cti import CtiResult, CtiCalibrator


def plot_cti(cti_result: CtiResult, out_dir: Path) -> None:
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



def plot_column_gain(col_result: ColumnGainResult, out_dir: Path,
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



def plot_cti_per_col(events: np.ndarray,
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



def plot_cti_correction_check(events: np.ndarray,
                                e_prelim: np.ndarray,
                                e_cti: np.ndarray,
                                cti_result: CtiResult,
                                out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_size = (cti_result.row_bins[1] - cti_result.row_bins[0]
                if len(cti_result.row_bins) > 1 else 64)
    data = _compute_cti_check_data(events, e_prelim, e_cti,
                                   row_bin_size=int(bin_size))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CTI Correction Check — Peak position vs Row",
                 fontsize=13, fontweight="bold")

    for ax, key, label, colour in [
        (axes[0], "before", "Before CTI correction", "steelblue"),
        (axes[1], "after",  "After CTI correction",  "darkorange"),
    ]:
        row_arr = data[key]["row_bins"]
        pk_arr  = data[key]["peak_per_bin"]
        success = data[key]["peak_success"]
        valid   = success & np.isfinite(pk_arr)

        n_valid = int(valid.sum())
        spread  = float(np.nanstd(pk_arr[valid])) if n_valid > 1 else 0.0

        if n_valid > 0:
            ax.scatter(row_arr[valid], pk_arr[valid],
                       s=25, color=colour, zorder=3,
                       label=f"Kα peak per row bin  (N={n_valid})")
        else:
            ax.text(0.5, 0.5, "No valid row bins",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="gray")

        ax.axhline(MN_KALPHA_EV, color="k", lw=1, ls="--",
                   label=f"Mn Kα = {MN_KALPHA_EV:.0f} eV")

        ax.set_xlabel("Row (Y)")
        ax.set_ylabel("Kα peak [eV]")
        ax.set_title(label)
        ax.grid(alpha=0.3)

        if n_valid > 0:
            ax.legend(fontsize=8)
            ax.text(0.97, 0.03, f"σ = {spread:.1f} eV",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9,
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
    Execute the energy calibration pipeline from a Config object.

    Reads  : {output_dir}/events.h5  (or energy_cal.events_file in YAML)
    Writes : {output_dir}/energy_cal.h5
             {output_dir}/cal_phase1_rough_gain.png
             {output_dir}/cal_phase3_cti.png
             {output_dir}/cal_phase4_column_gain.png
             {output_dir}/cal_cti_correction_check.png
             {output_dir}/cal_final_spectrum.png

    Returns dict with all intermediate and final results.
    """
    ec      = {**_CAL_DEFAULTS, **(cfg.energy_cal or {})}
    gen     = cfg.general
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ev    = float(ec["target_ev"])
    window_frac  = float(ec["fit_window_frac"])

    # ── Validate kalpha_adu (must be set by user) ─────────────────────────────
    kalpha_adu = ec.get("kalpha_adu")
    if kalpha_adu is None:
        raise ValueError(
            "energy_cal.kalpha_adu is not set in your analysis.yaml.\n"
            "  Open your event_rec spectrum plot (spectrum_full_detector.png),\n"
            "  read the Kα peak position in ADU for single-pixel events,\n"
            "  and add it to your config:\n\n"
            "    energy_cal:\n"
            "      kalpha_adu: 15000    # your value here\n")
    kalpha_adu     = float(kalpha_adu)
    kalpha_window  = float(ec.get("kalpha_adu_window", 0.20))

    print(f"  Kα peak (user-supplied): {kalpha_adu:.0f} ADU  "
          f"(window ±{kalpha_window:.0%})")
    print(f"  Target energy: {target_ev:.1f} eV  →  "
          f"expected gain ≈ {target_ev/kalpha_adu:.4f} eV/ADU")

    # ── Resolve input events file ─────────────────────────────────────────────
    events_file = ec.get("events_file") or None
    if events_file:
        events_path = cfg.resolve_input_path(events_file)
        if not (events_path and events_path.exists()):
            events_path = cfg.resolve_output_path(events_file)
    else:
        events_path = out_dir / "events.h5"

    if not events_path or not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found: {events_path}\n"
            "Run event_rec first, or set energy_cal.events_file in config.")

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

    # ── Phase 1: Rough gain ───────────────────────────────────────────────────
    print("\n── Phase 1: Rough global gain (even / odd columns) ──")
    rough_cal = RoughGainCalibrator(
        kalpha_adu  = kalpha_adu,
        target_ev   = target_ev,
        window_frac = kalpha_window,
        n_bins      = 100,
        min_events  = int(ec["rough_min_events"]),
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
    cti_grade_filter = ec.get("cti_grade_filter")
    if cti_grade_filter is not None:
        cti_grade_filter = [int(g) for g in cti_grade_filter]

    cti_cal = CtiCalibrator(
        row_bin_size = int(ec["cti_row_bin_size"]),
        target_ev    = target_ev,
        window_frac  = window_frac * 0.75,   # tighter window in eV space
        n_bins_hist  = 60,
        min_events   = int(ec["cti_min_events"]),
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
    col_grade_filter = ec.get("col_grade_filter")
    if col_grade_filter is not None:
        col_grade_filter = [int(g) for g in col_grade_filter]

    col_cal = ColumnGainCalibrator(
        target_ev    = target_ev,
        window_frac  = window_frac * 0.75,
        n_bins       = 60,
        min_events   = int(ec["col_min_events"]),
        with_bg      = bool(ec["col_with_bg"]),
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
    output_file = ec.get("output_file")
    if output_file:
        out_path = cfg.resolve_output_path(output_file)
    else:
        out_path = out_dir / "energy_cal.h5"

    save_energy_cal_h5(
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
            "cti_row_bin_size":  int(ec["cti_row_bin_size"]),
            "col_grade_filter":  str(col_grade_filter),
        },
    )

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    if gen.get("save_frame_plots", True) and ec.get("save_plots", True):
        print("\nGenerating calibration diagnostic plots …")
        plot_rough_gain(rough, events, out_dir, target_ev, kalpha_adu, kalpha_window)
        plot_cti(cti_result, out_dir)
        plot_cti_correction_check(events, e_prelim, e_cti, cti_result, out_dir)
        plot_column_gain(col_result, out_dir, target_ev)
        plot_final_spectrum(events, energy_ev, out_dir, target_ev)
        plot_pixel_gain_map(rough, col_result, out_dir, target_ev)
        plot_cti_per_col(
            events, e_cti, cti_result, out_dir, target_ev,
            n_rows=n_rows,
            row_bin_size=int(ec["cti_row_bin_size"]),
            min_events_per_bin=max(10, int(ec["cti_min_events"]) // 3),
        )

    # ── Save plot-backing data ────────────────────────────────────────────────
    save_energy_cal_results_h5(
        out_dir, rough, cti_result, col_result, events, energy_ev,
        e_prelim, e_cti, gen, ec, target_ev, kalpha_adu, kalpha_window,
        n_total, n_rows, n_cols,
    )

    print(f"\n✓ Energy calibration complete.  Output: {out_dir}/")

    return dict(
        rough=rough, cti=cti_result, col=col_result,
        e_prelim=e_prelim, e_cti=e_cti, energy_ev=energy_ev,
        g_even=rough.g_even, g_odd=rough.g_odd,
        cti_coeff=cti_result.cti, f_col=col_result.f_col,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Save plot-backing data to HDF5
# ──────────────────────────────────────────────────────────────────────────────

def _compute_f_col_histogram(f_col: np.ndarray, success: np.ndarray,
                             n_bins: int = 100) -> tuple:
    """Compute histogram of f_col values for successful columns."""
    valid = f_col[success]
    if len(valid) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    lo, hi = valid.min(), valid.max()
    counts, edges = np.histogram(valid, bins=n_bins, range=(lo, hi))
    return edges.astype(np.float64), counts.astype(np.int64)


def _compute_pixel_gain_data(rough: RoughGainResult,
                             col_result: ColumnGainResult) -> dict:
    """Compute effective gain per column data."""
    parity = np.arange(len(col_result.f_col)) % 2
    g_eff = np.where(parity == 0, rough.g_even, rough.g_odd) * col_result.f_col
    return {
        "g_eff":   g_eff.astype(np.float64),
        "parity":  parity.astype(np.uint8),
    }


def _compute_final_spectrum_data(
        events:    np.ndarray,
        energy_ev: np.ndarray,
        n_bins:    int = 200,
) -> dict:
    from ..physics.gain import fit_peak, MN_KALPHA_EV
    from ..physics.pattern_recognition import GRADE_OTHER

    # SINGLE_GRADES and SPLIT_GRADES already imported at module level
    lo, hi    = 3000.0, 9000.0
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    all_counts, _ = np.histogram(energy_ev, bins=bin_edges)

    groups: dict = {}
    for grp_name, grades in [
        ("single", SINGLE_GRADES),
        ("split",  SPLIT_GRADES),
        ("other",  frozenset({GRADE_OTHER})),
    ]:
        mask = np.isin(events["grade"], list(grades))
        if mask.sum() > 0:
            c, _ = np.histogram(energy_ev[mask], bins=bin_edges)
            groups[grp_name] = {
                "counts":    c.astype(np.int64),
                "grade_ids": np.array(list(grades), dtype=np.int32),
            }

    s_mask = np.isin(events["grade"], list(SINGLE_GRADES))
    res = fit_peak(energy_ev[s_mask], nominal=MN_KALPHA_EV,
                   window_frac=0.10, n_bins=80, min_events=30)

    kalpha_data = {
        "peak_ev":        float(res.peak_ev)   if res.success else np.nan,
        "sigma_ev":       float(res.sigma_ev)  if res.success else np.nan,
        "amplitude":      float(res.amplitude) if res.success else np.nan,
        "fwhm_ev":        float(2.3548 * res.sigma_ev)  if res.success else np.nan,
        "resolution_pct": float(2.3548 * res.sigma_ev / res.peak_ev * 100)
                          if res.success and res.peak_ev > 0 else np.nan,
        "n_events":       int(s_mask.sum()),
        "success":        bool(res.success),
        "fit_window_lo":  float(res.peak_ev * 0.90) if res.success else np.nan,
        "fit_window_hi":  float(res.peak_ev * 1.10) if res.success else np.nan,
    }

    return {
        "bin_edges":  bin_edges.astype(np.float32),
        "all_counts": all_counts.astype(np.int64),
        "groups":     groups,
        "kalpha_fit": kalpha_data,
    }


def save_energy_cal_results_h5(
        out_dir:        Path,
        rough:          RoughGainResult,
        cti_result:     CtiResult,
        col_result:     ColumnGainResult,
        events:         np.ndarray,
        energy_ev:      np.ndarray,
        e_prelim:       np.ndarray,
        e_cti:          np.ndarray,
        gen:            dict,
        ec:             dict,
        target_ev:      float,
        kalpha_adu:     float,
        kalpha_window:  float,
        n_total:        int,
        n_rows:         int,
        n_cols:         int,
) -> None:
    """
    Save plot-backing data to energy_cal_results.h5.

    HDF5 structure:
        /phase1_rough_gain/even/{hist_edges, hist_counts, fit/{...}}
        /phase1_rough_gain/odd/{hist_edges, hist_counts, fit/{...}}
        /phase1_rough_gain/window/{lo_adu, hi_adu}
        /phase3_cti/before/{row_bins, peak_per_bin, peak_success}
        /phase3_cti/after/{row_bins, peak_per_bin, peak_success}
        /phase4_column_gain/f_col_hist/{bin_edges, counts}
        /pixel_gain_map/{g_eff, parity, hist/{...}}
        /cti_per_col/{col_bin_centres, cti_per_cbin, cti_success, ...}
        /final_spectrum/{bin_edges, all_counts, per_group/{...}, kalpha_fit/{...}}
        /meta/...
    """
    path = out_dir / "energy_cal_results.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving energy_cal results: {path}")

    row_bin_size = int(ec.get("cti_row_bin_size", 64))

    with h5py.File(path, "w") as f:
        # ── Phase 1: Rough gain ───────────────────────────────────────────────
        p1g = f.require_group("phase1_rough_gain")

        for parity_name, peak_info, g_parity in [
            ("even", rough.peak_even, rough.g_even),
            ("odd",  rough.peak_odd,  rough.g_odd),
        ]:
            pg = p1g.require_group(parity_name)

            # Compute histogram
            lo_adu = kalpha_adu * (1 - kalpha_window)
            hi_adu = kalpha_adu * (1 + kalpha_window)
            singles_mask = (events["grade"] == 0) & (events["X"] % 2 == (0 if parity_name == "even" else 1))
            evts_par = events[singles_mask]
            counts, edges = np.histogram(evts_par["adu_sum"], bins=100, range=(lo_adu, hi_adu))
            pg.create_dataset("hist_edges",  data=edges.astype(np.float32))
            pg.create_dataset("hist_counts", data=counts.astype(np.int64))

            # Fit parameters
            fg = pg.require_group("fit")
            fg.create_dataset("peak_adu",   data=float(peak_info.peak_ev))
            fg.create_dataset("sigma_adu",  data=float(peak_info.sigma_ev))
            fg.create_dataset("amplitude",  data=float(peak_info.amplitude))
            fg.create_dataset("success",    data=bool(peak_info.success))
            fg.create_dataset("n_events",   data=int(np.sum(singles_mask)))

        # Window
        wg = p1g.require_group("window")
        wg.create_dataset("lo_adu", data=lo_adu)
        wg.create_dataset("hi_adu", data=hi_adu)

        # ── Phase 3: CTI ─────────────────────────────────────────────────────
        p3g = f.require_group("phase3_cti")
        cti_data = _compute_cti_check_data(events, e_prelim, e_cti, row_bin_size=row_bin_size)

        for key in ["before", "after"]:
            cg = p3g.require_group(key)
            cg.create_dataset("row_bins",       data=cti_data[key]["row_bins"])
            cg.create_dataset("peak_per_bin",   data=cti_data[key]["peak_per_bin"])
            cg.create_dataset("peak_success",   data=cti_data[key]["peak_success"])

        # ── Phase 4: Column gain ─────────────────────────────────────────────
        p4g = f.require_group("phase4_column_gain")
        f_edges, f_counts = _compute_f_col_histogram(col_result.f_col, col_result.success_col)
        hg = p4g.require_group("f_col_hist")
        hg.create_dataset("bin_edges", data=f_edges)
        hg.create_dataset("counts",   data=f_counts)

        # ── Pixel gain map ───────────────────────────────────────────────────
        pgm = f.require_group("pixel_gain_map")
        pix_data = _compute_pixel_gain_data(rough, col_result)
        pgm.create_dataset("g_eff",  data=pix_data["g_eff"])
        pgm.create_dataset("parity", data=pix_data["parity"])

        # Histogram
        valid_g = pix_data["g_eff"][col_result.success_col]
        if len(valid_g) > 0:
            g_lo, g_hi = valid_g.min(), valid_g.max()
            c_even, e_even = np.histogram(pix_data["g_eff"][pix_data["parity"] == 0],
                                           bins=80, range=(g_lo, g_hi))
            c_odd, e_odd = np.histogram(pix_data["g_eff"][pix_data["parity"] == 1],
                                          bins=80, range=(g_lo, g_hi))
            hg = pgm.require_group("g_eff_hist")
            hg.create_dataset("bin_edges",   data=e_even.astype(np.float64))
            hg.create_dataset("counts_even", data=c_even.astype(np.int64))
            hg.create_dataset("counts_odd",   data=c_odd.astype(np.int64))

        # ── CTI per column ───────────────────────────────────────────────────
        cpc = f.require_group("cti_per_col")
        cpc.create_dataset("col_bin_size", data=row_bin_size, dtype=np.int32)
        # Note: 2D peak map and per-column CTI computed in plot_cti_per_col
        # would need additional extraction - storing only what's readily available

        # ── Final spectrum ───────────────────────────────────────────────────
        fsg = f.require_group("final_spectrum")
        spec_data = _compute_final_spectrum_data(events, energy_ev)

        fsg.create_dataset("bin_edges",  data=spec_data["bin_edges"])
        all_g = fsg.require_group("all_grades")
        all_g.create_dataset("counts",   data=spec_data["all_counts"])

        for grp_name, grp_data in spec_data["groups"].items():
            gg = fsg.require_group("per_group").require_group(grp_name)
            gg.create_dataset("counts",     data=grp_data["counts"])
            gg.create_dataset("grade_ids",  data=grp_data["grade_ids"])

        kf = fsg.require_group("kalpha_fit")
        for k, v in spec_data["kalpha_fit"].items():
            kf.create_dataset(k, data=v)

        # ── Metadata ────────────────────────────────────────────────────────
        mg = f.require_group("meta")
        mg.attrs["target_ev"]        = target_ev
        mg.attrs["kalpha_adu"]       = kalpha_adu
        mg.attrs["cti_coefficient"]   = float(cti_result.cti)
        mg.attrs["g_even"]           = rough.g_even
        mg.attrs["g_odd"]            = rough.g_odd
        mg.attrs["n_events"]         = n_total
        mg.attrs["n_rows"]           = n_rows
        mg.attrs["n_cols"]           = n_cols
        mg.attrs["col_bin_size"]     = int(ec.get("cti_row_bin_size", 64))

    print("  ✓ saved.")


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
