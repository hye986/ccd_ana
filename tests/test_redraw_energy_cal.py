"""
test_redraw_energy_cal.py
=========================
Redraw diagnostic plots from energy_cal_results.h5.

This script reloads the plot-backing data saved by the energy_cal stage
and regenerates the same diagnostic plots without rerunning the pipeline.

Usage
-----
    python -m pytest tests/test_redraw_energy_cal.py -v
    python tests/test_redraw_energy_cal.py [--results RESULTS_FILE]

Output
------
    energy_cal_plots/cal_phase1_rough_gain.png   - even/odd ADU spectra + Gaussian fits
    energy_cal_plots/cal_phase3_cti.png          - peak vs row + CTI linear fit + residuals
    energy_cal_plots/cal_cti_correction_check.png - before/after CTI peak-vs-row
    energy_cal_plots/cal_phase4_column_gain.png  - f_col map + histogram + peak per column
    energy_cal_plots/cal_final_spectrum.png      - all-grades spectra + Kα resolution fit
    energy_cal_plots/cal_pixel_gain_map.png      - G_eff vs column + gain histogram
    energy_cal_plots/cal_cti_per_col.png         - local CTI per column + 2-D peak map
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MN_KALPHA_EV = 5895.0  # Mn K-alpha line energy in eV


def redraw_energy_cal_plots(results_path: str | Path, output_dir: str | Path | None = None):
    """
    Load energy_cal_results.h5 and redraw all energy_cal stage plots.

    Parameters
    ----------
    results_path : path to energy_cal_results.h5
    output_dir   : output directory for plots (default: ./energy_cal_plots)
    """
    from pnccd_ana.analysis import load_energy_cal_results

    results_path = Path(results_path)
    if output_dir is None:
        output_dir = results_path.parent / "energy_cal_plots"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading energy_cal results from: {results_path}")
    data = load_energy_cal_results(results_path)

    # ── Extract data ───────────────────────────────────────────────────────────
    phase1 = data.get("phase1_rough_gain", {})
    phase3 = data.get("phase3_cti", {})
    phase4 = data.get("phase4_column_gain", {})
    pixel_gain = data.get("pixel_gain_map", {})
    cti_per_col = data.get("cti_per_col", {})
    final_spectrum = data.get("final_spectrum", {})
    meta = data.get("meta", {})

    # ── 1. Phase 1: Rough gain ─────────────────────────────────────────────────
    _plot_rough_gain_redraw(phase1, output_dir, meta)

    # ── 2. Phase 3: CTI ──────────────────────────────────────────────────────
    _plot_cti_redraw(phase3, output_dir, meta)

    # ── 3. CTI correction check ───────────────────────────────────────────────
    _plot_cti_correction_check_redraw(phase3, output_dir, meta)

    # ── 4. Phase 4: Column gain ──────────────────────────────────────────────
    _plot_column_gain_redraw(phase4, output_dir, meta)

    # ── 5. Pixel gain map ───────────────────────────────────────────────────
    _plot_pixel_gain_map_redraw(pixel_gain, output_dir, meta)

    # ── 6. Final spectrum ────────────────────────────────────────────────────
    _plot_final_spectrum_redraw(final_spectrum, output_dir, meta)

    print(f"\n✓ Energy_cal plots redrawn to: {output_dir}/")


def _plot_rough_gain_redraw(phase1: dict, out_dir: Path, meta: dict):
    """Redraw Phase 1: Rough gain spectra with Gaussian fits."""
    if not phase1:
        _empty_plot("cal_phase1_rough_gain.png",
                    "Phase 1: No rough gain data", out_dir)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 1: Rough Gain Calibration", fontsize=14, fontweight="bold")

    for ax, parity in zip(axes, ["even", "odd"]):
        if parity not in phase1:
            ax.text(0.5, 0.5, f"No {parity} data",
                   ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        p = phase1[parity]
        edges = p.get("hist_edges", np.array([]))
        counts = p.get("hist_counts", np.array([]))
        fit = p.get("fit", {})

        if len(edges) > 0:
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centres, counts, "b-", lw=1.5, label="Data")

            # Plot Gaussian fit if available
            if fit.get("success"):
                peak = fit.get("peak_adu", 0)
                sigma = fit.get("sigma_adu", 1)
                amp = fit.get("amplitude", 1)
                x_fit = np.linspace(centres.min(), centres.max(), 200)
                y_fit = amp * np.exp(-0.5 * ((x_fit - peak) / sigma) ** 2)
                ax.plot(x_fit, y_fit, "r-", lw=2,
                       label=f"Gaussian\npeak={peak:.0f} ADU\nσ={sigma:.0f} ADU")

            ax.axvline(fit.get("peak_adu", 0), color="red", linestyle="--",
                      alpha=0.5, label=f"Fit peak")

        ax.set_xlabel("ADU (adu_sum)")
        ax.set_ylabel("Count")
        ax.set_title(f"{parity.capitalize()} Columns")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "cal_phase1_rough_gain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'cal_phase1_rough_gain.png'}")


def _plot_cti_redraw(phase3: dict, out_dir: Path, meta: dict):
    """Redraw Phase 3: CTI coefficient fit."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 3: CTI Correction", fontsize=14, fontweight="bold")

    # Check if we have after-correction data (contains CTI coefficient)
    after = phase3.get("after", {})
    before = phase3.get("before", {})

    if not after:
        for ax in axes:
            ax.text(0.5, 0.5, "No CTI data",
                   ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "cal_phase3_cti.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_dir / 'cal_phase3_cti.png'}")
        return

    row_bins = after.get("row_bins", np.array([]))
    peak_after = after.get("peak_per_bin", np.array([]))
    success = after.get("peak_success", np.array([]))

    if len(row_bins) > 0:
        valid = success & ~np.isnan(peak_after)

        # Peak vs row
        ax0 = axes[0]
        ax0.scatter(row_bins[valid], peak_after[valid], c="blue", s=30, zorder=3)
        ax0.axhline(MN_KALPHA_EV, color="k", lw=1, ls="--",
                   label=f"Mn Kα = {MN_KALPHA_EV:.0f} eV")
        ax0.set_xlabel("Row (Y)")
        ax0.set_ylabel("Kα peak [eV]")
        ax0.set_title("Peak Position vs Row (After CTI)")
        ax0.legend(fontsize=9)
        ax0.grid(alpha=0.3)

        # Linear fit for CTI
        ax1 = axes[1]
        if valid.sum() > 1:
            # Simple linear fit: peak = e0 - CTI * row
            valid_idx = np.where(valid)[0]
            x = row_bins[valid_idx]
            y = peak_after[valid_idx]
            coeffs = np.polyfit(x, y, 1)
            cti_coeff = -coeffs[0]  # negative slope = CTI
            e0 = coeffs[1]

            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = np.polyval(coeffs, x_fit)

            ax1.scatter(x, y, c="blue", s=30, zorder=3)
            ax1.plot(x_fit, y_fit, "r-", lw=2,
                    label=f"CTI = {cti_coeff:.4f}/pixel\ne0 = {e0:.1f} eV")
            ax1.axhline(MN_KALPHA_EV, color="k", lw=1, ls="--")
            ax1.set_xlabel("Row (Y)")
            ax1.set_ylabel("Kα peak [eV]")
            ax1.set_title("Linear Fit: CTI Coefficient")
            ax1.legend(fontsize=9)
            ax1.grid(alpha=0.3)

        # Residuals
        ax2 = axes[2]
        if valid.sum() > 1:
            residuals = peak_after[valid] - MN_KALPHA_EV
            ax2.scatter(row_bins[valid], residuals, c="green", s=30, zorder=3)
            ax2.axhline(0, color="k", lw=1)
            ax2.set_xlabel("Row (Y)")
            ax2.set_ylabel("Residual [eV]")
            ax2.set_title("Residuals")
            ax2.grid(alpha=0.3)

            # Add std annotation
            std_res = np.std(residuals)
            ax2.text(0.97, 0.03, f"σ = {std_res:.1f} eV",
                    transform=ax2.transAxes, ha="right", va="bottom",
                    fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_dir / "cal_phase3_cti.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'cal_phase3_cti.png'}")


def _plot_cti_correction_check_redraw(phase3: dict, out_dir: Path, meta: dict):
    """Redraw before/after CTI correction comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CTI Correction Check — Peak position vs Row",
                 fontsize=13, fontweight="bold")

    for ax, key, label, colour in [
        (axes[0], "before", "Before CTI correction", "steelblue"),
        (axes[1], "after",  "After CTI correction",  "darkorange"),
    ]:
        if key not in phase3:
            ax.text(0.5, 0.5, f"No {key} data",
                   ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        d = phase3[key]
        row_bins = d.get("row_bins", np.array([]))
        peak_arr = d.get("peak_per_bin", np.array([]))
        success = d.get("peak_success", np.array([]))

        valid = success & ~np.isnan(peak_arr)
        if np.any(valid):
            ax.scatter(row_bins[valid], peak_arr[valid], s=25,
                      color=colour, zorder=3)
        ax.axhline(MN_KALPHA_EV, color="k", lw=1, ls="--",
                   label=f"Mn Kα = {MN_KALPHA_EV:.0f} eV")
        ax.set_xlabel("Row (Y)")
        ax.set_ylabel("Kα peak [eV]")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        spread = float(np.nanstd(peak_arr[valid])) if np.any(valid) else 0.0
        ax.text(0.97, 0.03, f"σ = {spread:.1f} eV",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_dir / "cal_cti_correction_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'cal_cti_correction_check.png'}")


def _plot_column_gain_redraw(phase4: dict, out_dir: Path, meta: dict):
    """Redraw Phase 4: Column gain f_col."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 4: Column Gain Correction", fontsize=14, fontweight="bold")

    # f_col histogram
    ax0 = axes[0]
    f_col_hist = phase4.get("f_col_hist", {})
    if f_col_hist:
        edges = f_col_hist.get("bin_edges", np.array([]))
        counts = f_col_hist.get("counts", np.array([]))
        if len(edges) > 0:
            ax0.hist(edges[:-1], bins=edges, weights=counts,
                    alpha=0.7, color="blue", edgecolor="black")
            ax0.axvline(1.0, color="red", linestyle="--", lw=2,
                       label="Nominal (f_col = 1.0)")
            ax0.set_xlabel("f_col (column gain factor)")
            ax0.set_ylabel("Count")
            ax0.set_title("f_col Distribution")
            ax0.legend()
            ax0.grid(alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "No column gain data",
                ha="center", va="center", transform=ax0.transAxes)
        ax0.axis("off")

    # Placeholder for per-column plot (needs original data)
    ax1 = axes[1]
    ax1.text(0.5, 0.5,
            "Per-column f_col vs column\n(requires full column gain data)",
            ha="center", va="center", transform=ax1.transAxes)
    ax1.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "cal_phase4_column_gain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'cal_phase4_column_gain.png'}")


def _plot_pixel_gain_map_redraw(pixel_gain: dict, out_dir: Path, meta: dict):
    """Redraw pixel gain map and histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Pixel Gain Map", fontsize=14, fontweight="bold")

    g_eff = pixel_gain.get("g_eff", np.array([]))
    parity = pixel_gain.get("parity", np.array([]))
    hist = pixel_gain.get("g_eff_hist", {})

    # G_eff vs column
    ax0 = axes[0]
    if len(g_eff) > 0:
        n_cols = len(g_eff)
        x = np.arange(n_cols)
        ax0.scatter(x, g_eff, c=parity, cmap="bwr", s=1, alpha=0.5)
        ax0.set_xlabel("Column")
        ax0.set_ylabel("G_eff [eV/ADU]")
        ax0.set_title("Effective Gain per Column")
        ax0.grid(alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "No gain map data",
                ha="center", va="center", transform=ax0.transAxes)
        ax0.axis("off")

    # Histogram even columns
    ax1 = axes[1]
    if hist:
        edges = hist.get("bin_edges", np.array([]))
        counts_even = hist.get("counts_even", np.array([]))
        if len(edges) > 0:
            ax1.hist(edges[:-1], bins=edges, weights=counts_even,
                    alpha=0.7, color="blue", label="Even columns")
            ax1.set_xlabel("G_eff [eV/ADU]")
            ax1.set_ylabel("Count")
            ax1.set_title("Even Columns Gain Distribution")
            ax1.legend()
            ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No histogram data",
                ha="center", va="center", transform=ax1.transAxes)
        ax1.axis("off")

    # Histogram odd columns
    ax2 = axes[2]
    if hist:
        edges = hist.get("bin_edges", np.array([]))
        counts_odd = hist.get("counts_odd", np.array([]))
        if len(edges) > 0:
            ax2.hist(edges[:-1], bins=edges, weights=counts_odd,
                    alpha=0.7, color="red", label="Odd columns")
            ax2.set_xlabel("G_eff [eV/ADU]")
            ax2.set_ylabel("Count")
            ax2.set_title("Odd Columns Gain Distribution")
            ax2.legend()
            ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No histogram data",
                ha="center", va="center", transform=ax2.transAxes)
        ax2.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "cal_pixel_gain_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'cal_pixel_gain_map.png'}")


def _plot_final_spectrum_redraw(final_spectrum: dict, out_dir: Path, meta: dict):
    """Redraw final calibrated spectrum with K-alpha fit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Final Calibrated Energy Spectrum", fontsize=14, fontweight="bold")

    bin_edges = final_spectrum.get("bin_edges", np.array([]))
    all_grades = final_spectrum.get("all_grades", {})
    all_counts = all_grades.get("counts", np.array([]))
    kalpha = final_spectrum.get("kalpha_fit", {})

    if len(bin_edges) > 0 and len(all_counts) > 0:
        centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Main spectrum
        ax0 = axes[0]
        ax0.plot(centres, all_counts, "b-", lw=1.5, label="All grades")
        ax0.axvline(MN_KALPHA_EV, color="k", lw=1, ls="--",
                   label=f"Mn Kα = {MN_KALPHA_EV:.0f} eV")

        # Add K-alpha fit if available
        if kalpha.get("success"):
            peak = kalpha.get("peak_ev", MN_KALPHA_EV)
            fwhm = kalpha.get("fwhm_ev", 0)
            res = kalpha.get("resolution_pct", 0)
            ax0.axvline(peak, color="red", lw=2,
                       label=f"Fit: {peak:.1f} eV\nFWHM: {fwhm:.1f} eV ({res:.2f}%)")

        ax0.set_xlabel("Energy [eV]")
        ax0.set_ylabel("Count")
        ax0.set_title("Calibrated Energy Spectrum")
        ax0.legend(fontsize=8)
        ax0.grid(alpha=0.3)
        ax0.set_xlim(0, None)

        # Zoom around K-alpha
        ax1 = axes[1]
        ax1.plot(centres, all_counts, "b-", lw=1.5)

        if kalpha.get("success"):
            peak = kalpha.get("peak_ev", MN_KALPHA_EV)
            fwhm = kalpha.get("fwhm_ev", 0)
            sigma = kalpha.get("sigma_ev", 0)

            # Plot Gaussian fit
            x_fit = np.linspace(max(0, peak - 3*fwhm), peak + 3*fwhm, 200)
            amp = kalpha.get("amplitude", all_counts.max())
            y_fit = amp * np.exp(-0.5 * ((x_fit - peak) / sigma) ** 2)
            ax1.plot(x_fit, y_fit, "r-", lw=2, label="Gaussian fit")

        ax1.axvline(MN_KALPHA_EV, color="k", lw=1, ls="--",
                   label=f"Mn Kα = {MN_KALPHA_EV:.0f} eV")

        if kalpha.get("success"):
            peak = kalpha.get("peak_ev", MN_KALPHA_EV)
            ax1.axvline(peak, color="red", lw=2, label=f"Fit: {peak:.1f} eV")

        # Zoom window
        if kalpha.get("success"):
            lo = kalpha.get("fit_window_lo", MN_KALPHA_EV * 0.9)
            hi = kalpha.get("fit_window_hi", MN_KALPHA_EV * 1.1)
            ax1.set_xlim(lo, hi)

        ax1.set_xlabel("Energy [eV]")
        ax1.set_ylabel("Count")
        ax1.set_title("Zoom: K-alpha Region")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No spectrum data",
                   ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "cal_final_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'cal_final_spectrum.png'}")


def _empty_plot(filename: str, message: str, out_dir: Path):
    """Create an empty plot with a message."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha="center", va="center",
           transform=ax.transAxes, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / filename}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Redraw energy_cal stage plots from energy_cal_results.h5")
    parser.add_argument("results", nargs="?", default="output/energy_cal_results.h5",
                       help="Path to energy_cal_results.h5")
    parser.add_argument("-o", "--output", default=None,
                       help="Output directory for plots")
    args = parser.parse_args()

    redraw_energy_cal_plots(args.results, args.output)


if __name__ == "__main__":
    main()
