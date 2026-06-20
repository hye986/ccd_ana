"""
pnccd_ana.plotting.gain_plots
==============================
Gain calibration plotting functions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import _cb, _stats_box, _build_grade_palette, _build_group_label
from ..physics.gain import (RoughGainResult, ColumnGainResult,
                             PeakFitResult, fit_peak, MN_KALPHA_EV, MN_KBETA_EV,
                             _gaussian)


def plot_rough_gain(rough: RoughGainResult,
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



def plot_pixel_gain_map(rough: RoughGainResult,
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


