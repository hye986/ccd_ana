"""
pnccd_ana.plotting.spectrum_plots
==================================
Final calibrated energy spectrum plot.

Only imported by cli/energy_cal.py — never at offset or event_rec time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..physics.gain      import MN_KALPHA_EV, MN_KBETA_EV, fit_peak
from ..physics.calibrate import GRADE_NAMES, GRADE_OTHER, _GRADE_DEFS


# ══════════════════════════════════════════════════════════════════════════════
# Grade colour palette  (built locally — no import from common.py)
# ══════════════════════════════════════════════════════════════════════════════

def _grade_palette() -> dict[int, str]:
    """Fixed colour per grade 0–13."""
    cmap = plt.cm.get_cmap("tab20")
    from ..physics.calibrate import N_GRADES
    return {gid: cmap(gid / N_GRADES) for gid in range(N_GRADES)}


# ══════════════════════════════════════════════════════════════════════════════
# Final spectrum
# ══════════════════════════════════════════════════════════════════════════════

def plot_final_spectrum(
        grades:      np.ndarray,
        energy_sum:  np.ndarray,
        out_dir:     Path,
        target_ev:   float = MN_KALPHA_EV,
) -> None:
    """
    Calibrated energy spectrum for all grade groups with Kα resolution fit.

    Panel 1 (log)    : per-grade-group spectra
    Panel 2 (linear) : all-grades sum + Gaussian fit to Kα peak

    Parameters
    ----------
    grades     : int8  (n_events,) — from assign_grades
    energy_sum : float32 (n_events,) — from compute_final_energies [eV]
    out_dir    : output directory
    target_ev  : calibration line energy [eV]
    """
    out_dir = Path(out_dir)
    palette = _grade_palette()

    # Group grades
    from collections import defaultdict
    prefix_to_gids: dict[str, list[int]] = defaultdict(list)
    for gid, label, _ in _GRADE_DEFS:
        prefix = label.split()[0]
        prefix_to_gids[prefix].append(gid)

    groups: list[tuple[str, list[int]]] = []
    for prefix, gids in sorted(prefix_to_gids.items(),
                                key=lambda kv: min(kv[1])):
        groups.append((prefix, sorted(gids)))
    groups.append(("other", [GRADE_OTHER]))

    # Energy axis
    lo      = target_ev * 0.60
    hi      = MN_KBETA_EV * 1.30
    bins    = np.linspace(lo, hi, 350)
    centres = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Final Calibrated Fe-55 Spectrum — All Grades",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: per-group log ────────────────────────────────────────────────
    ax1         = axes[0]
    all_counts  = np.zeros(len(centres), dtype=np.float64)

    for prefix, gids in groups:
        mask = np.isin(grades, gids)
        if not mask.any():
            continue
        c, _  = np.histogram(energy_sum[mask], bins=bins)
        all_counts += c.astype(np.float64)
        colour = palette.get(gids[0], "#aaaaaa")
        label  = (f"{GRADE_NAMES.get(gids[0], prefix).title()}"
                  f" (g{gids[0]}–g{gids[-1]})"
                  if len(gids) > 1
                  else f"{GRADE_NAMES.get(gids[0], prefix).title()} (g{gids[0]})")
        ax1.step(centres, c, where="mid", color=colour, lw=1.1, alpha=0.85,
                 label=f"{label}  N={mask.sum():,}")

    ax1.step(centres, all_counts, where="mid", color="black",
             lw=1.3, ls="--", alpha=0.7,
             label=f"All grades  N={len(grades):,}")
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

    # ── Panel 2: all-grades sum + Kα fit ─────────────────────────────────────
    ax2 = axes[1]
    ax2.step(centres, all_counts, where="mid", color="steelblue",
             lw=1.2, alpha=0.9,
             label=f"All grades  N={len(grades):,}")

    # Gaussian fit to Kα window
    fit_window = 0.12
    lo_fit = target_ev * (1 - fit_window)
    hi_fit = target_ev * (1 + fit_window)
    mask_fit = (energy_sum > lo_fit) & (energy_sum < hi_fit)
    res = fit_peak(energy_sum[mask_fit], lo_fit, hi_fit, n_params=3)

    if res.success:
        from scipy.stats import norm as _norm
        fwhm   = 2.3548 * res.sigma_adu
        resoln = fwhm / res.peak_adu * 100.0

        xs = np.linspace(lo_fit, hi_fit, 500)
        # Gaussian with amplitude scaled to histogram bin width
        bin_w  = bins[1] - bins[0]
        ys     = (res.amplitude * bin_w *
                  _norm.pdf(xs, res.peak_adu, res.sigma_adu))
        # Re-normalise: amplitude from fit_peak is histogram counts
        # so just use the Gaussian directly
        from ..physics.gain import _gauss
        ys = _gauss(xs, res.amplitude, res.peak_adu, res.sigma_adu)

        ax2.plot(xs, ys, "r-", lw=2.5,
                 label=(f"Gaussian fit\n"
                        f"Peak = {res.peak_adu:.1f} eV\n"
                        f"σ    = {res.sigma_adu:.1f} eV\n"
                        f"FWHM = {fwhm:.1f} eV\n"
                        f"R    = {resoln:.2f}%"))
        ax2.axvline(res.peak_adu, color="red", lw=1, ls="--", alpha=0.6)

        print(f"\n  ── Kα energy resolution ──")
        print(f"     Peak  = {res.peak_adu:.2f} eV")
        print(f"     σ     = {res.sigma_adu:.2f} eV")
        print(f"     FWHM  = {fwhm:.2f} eV")
        print(f"     R     = {resoln:.3f}%")
        print(f"     N_fit = {res.n_events:,} events in fit window")
    else:
        print("  ⚠  Kα resolution fit failed — check ROI and statistics.")

    ax2.axvline(target_ev,   color="red",  lw=1.2, ls="--", alpha=0.5)
    ax2.axvline(MN_KBETA_EV, color="blue", lw=1.2, ls="--", alpha=0.5,
                label=f"Mn Kβ {MN_KBETA_EV:.0f} eV")
    ax2.set_xlabel("Energy [eV]")
    ax2.set_ylabel("Counts / bin")
    ax2.set_title("All-grades sum  (linear) + Kα resolution fit")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(lo, hi)

    plt.tight_layout()
    p = out_dir / "cal_final_spectrum.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")
