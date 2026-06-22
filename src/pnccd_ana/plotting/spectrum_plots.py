"""
pnccd_ana.plotting.spectrum_plots
==================================
Calibrated energy spectrum and resolution fit.

Only imported by cli/energy_cal.py — never at offset or event_rec time.

Inputs (from energy_cal.h5):
    grades      : int8  (n_events,)   — 0=single .. 13=other
    energy_sum  : float32 (n_events,) — calibrated energy [eV]

Output plots:
    cal_spectrum_by_grade.png  — per-grade-group spectra (log scale)
    cal_spectrum_resolution.png — all-grades sum + Gaussian fit to Kα peak
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ..physics.calibrate import (GRADE_NAMES, GRADE_OTHER, N_GRADES,
                                  _GRADE_DEFS)
from ..physics.gain import MN_KALPHA_EV, MN_KBETA_EV


# ══════════════════════════════════════════════════════════════════════════════
# Grade colour palette  (built locally — no dependency on common.py)
# ══════════════════════════════════════════════════════════════════════════════

def _grade_palette() -> dict[int, str]:
    """Fixed colour per grade 0–13 using tab20 colormap."""
    cmap = plt.cm.get_cmap("tab20")
    return {gid: cmap(gid / N_GRADES) for gid in range(N_GRADES)}


def _grade_groups() -> list[tuple[str, list[int], str]]:
    """
    Build ordered list of (label, [grade_ids], color) for spectrum plots.

    Groups _GRADE_DEFS entries by name prefix (single/double/triple/quadruple)
    then appends GRADE_OTHER. Colors are fixed for readability.
    """
    prefix_to_gids: dict[str, list[int]] = defaultdict(list)
    for gid, label, _ in _GRADE_DEFS:
        prefix = label.split()[0]   # "single", "double", "triple", "quadruple"
        prefix_to_gids[prefix].append(gid)

    _COLORS = {
        "single":     "steelblue",
        "double":     "tomato",
        "triple":     "seagreen",
        "quadruple":  "orange",
        "other":      "grey",
    }

    groups = []
    for prefix, gids in sorted(prefix_to_gids.items(),
                                key=lambda kv: min(kv[1])):
        gids_sorted = sorted(gids)
        if len(gids_sorted) == 1:
            label = f"{prefix} (g{gids_sorted[0]})"
        else:
            label = (f"{prefix} "
                     f"(g{gids_sorted[0]}–g{gids_sorted[-1]})")
        groups.append((label, gids_sorted, _COLORS.get(prefix, "grey")))

    groups.append((f"other (g{GRADE_OTHER})", [GRADE_OTHER],
                   _COLORS["other"]))
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian model for resolution fit
# ══════════════════════════════════════════════════════════════════════════════

def _gauss(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _gauss_plus_bg(x, amplitude, mean, sigma, bg):
    return _gauss(x, amplitude, mean, sigma) + bg


def _fit_kalpha(
        energy_ev:   np.ndarray,
        target_ev:   float,
        window_frac: float = 0.12,
        n_bins:      int   = 150,
        with_bg:     bool  = False,
) -> dict:
    """
    Fit a Gaussian (+ optional constant background) to the Mn Kα peak.

    Parameters
    ----------
    energy_ev   : calibrated energies of ALL events [eV]
    target_ev   : expected peak position [eV]  (Mn Kα = 5898.8 eV)
    window_frac : fit window = target_ev × (1 ± window_frac)
    n_bins      : histogram bins inside the fit window
    with_bg     : include constant background term

    Returns
    -------
    dict with keys:
        success      : bool
        peak_ev      : float  — fitted peak position [eV]
        sigma_ev     : float  — Gaussian sigma [eV]
        fwhm_ev      : float  — FWHM = 2.3548 × sigma [eV]
        resolution   : float  — FWHM / peak_ev × 100 [%]
        amplitude    : float
        bg           : float  — background (0 if with_bg=False)
        n_events     : int    — events in fit window
        lo_ev        : float  — fit window lower bound [eV]
        hi_ev        : float  — fit window upper bound [eV]
        message      : str    — failure reason if success=False
    """
    lo_ev = target_ev * (1.0 - window_frac)
    hi_ev = target_ev * (1.0 + window_frac)

    mask = (energy_ev > lo_ev) & (energy_ev < hi_ev)
    n    = int(mask.sum())

    fail = dict(success=False, peak_ev=target_ev, sigma_ev=0.0,
                fwhm_ev=0.0, resolution=0.0, amplitude=0.0, bg=0.0,
                n_events=n, lo_ev=lo_ev, hi_ev=hi_ev, message="")

    if n < 50:
        fail["message"] = f"Too few events in fit window: {n} < 50"
        return fail

    counts, edges = np.histogram(energy_ev[mask], bins=n_bins,
                                  range=(lo_ev, hi_ev))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Initial guesses
    pk    = int(np.argmax(counts))
    amp0  = float(counts[pk])
    mu0   = float(centers[pk])
    sig0  = (hi_ev - lo_ev) / 6.0
    bg0   = float(np.percentile(counts, 10))

    try:
        if with_bg:
            popt, _ = curve_fit(
                _gauss_plus_bg, centers, counts.astype(float),
                p0=[amp0, mu0, sig0, bg0],
                bounds=([0, lo_ev, 0, 0],
                        [np.inf, hi_ev, hi_ev - lo_ev, np.inf]),
                maxfev=5000)
            amp, mu, sig, bg = popt
        else:
            popt, _ = curve_fit(
                _gauss, centers, counts.astype(float),
                p0=[amp0, mu0, sig0],
                bounds=([0, lo_ev, 0],
                        [np.inf, hi_ev, hi_ev - lo_ev]),
                maxfev=5000)
            amp, mu, sig = popt
            bg = 0.0

        sig = abs(sig)
        if not (lo_ev < mu < hi_ev) or sig <= 0:
            fail["message"] = (f"Fit result out of range: "
                               f"peak={mu:.1f} sigma={sig:.1f}")
            return fail

        fwhm = 2.3548 * sig
        return dict(
            success    = True,
            peak_ev    = float(mu),
            sigma_ev   = float(sig),
            fwhm_ev    = float(fwhm),
            resolution = float(fwhm / mu * 100.0),
            amplitude  = float(amp),
            bg         = float(bg),
            n_events   = n,
            lo_ev      = lo_ev,
            hi_ev      = hi_ev,
            message    = "",
        )

    except (RuntimeError, ValueError) as e:
        fail["message"] = str(e)
        return fail


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: spectrum by grade group
# ══════════════════════════════════════════════════════════════════════════════

def plot_spectrum_by_grade(
        grades:     np.ndarray,
        energy_sum: np.ndarray,
        out_dir:    Path,
        target_ev:  float = MN_KALPHA_EV,
        e_min:      float | None = None,
        e_max:      float | None = None,
        n_bins:     int   = 400,
) -> None:
    """
    Log-scale spectrum split by grade group.

    Parameters
    ----------
    grades     : int8  (n_events,)   from assign_grades()
    energy_sum : float32 (n_events,) calibrated energy [eV]
    out_dir    : output directory
    target_ev  : Mn Kα reference energy for vertical line
    e_min/max  : energy axis range (default: 0.6×Kα to 1.3×Kβ)
    n_bins     : histogram bins
    """
    out_dir = Path(out_dir)

    lo = e_min if e_min is not None else target_ev * 0.60
    hi = e_max if e_max is not None else MN_KBETA_EV * 1.30
    bins    = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    groups     = _grade_groups()
    all_counts = np.zeros(n_bins, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, gids, color in groups:
        mask = np.isin(grades, gids)
        if not mask.any():
            continue
        c, _ = np.histogram(energy_sum[mask], bins=bins)
        all_counts += c.astype(np.float64)
        ax.step(centers, c, where="mid",
                color=color, lw=1.2, alpha=0.85,
                label=f"{label}  N={mask.sum():,}")

    ax.step(centers, all_counts, where="mid",
            color="black", lw=1.3, ls="--", alpha=0.7,
            label=f"all grades  N={len(grades):,}")

    ax.axvline(target_ev,   color="red",  lw=1.2, ls="--",
               alpha=0.7, label=f"Mn Kα {target_ev:.0f} eV")
    ax.axvline(MN_KBETA_EV, color="blue", lw=1.2, ls="--",
               alpha=0.7, label=f"Mn Kβ {MN_KBETA_EV:.0f} eV")

    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("Counts / bin")
    ax.set_title("Calibrated Fe-55 Spectrum — by grade group  (log scale)",
                 fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = out_dir / "cal_spectrum_by_grade.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: all-grades sum + Kα resolution fit
# ══════════════════════════════════════════════════════════════════════════════

def plot_resolution_fit(
        grades:      np.ndarray,
        energy_sum:  np.ndarray,
        out_dir:     Path,
        target_ev:   float = MN_KALPHA_EV,
        window_frac: float = 0.12,
        with_bg:     bool  = False,
        grade_filter: list[int] | None = None,
        e_min:       float | None = None,
        e_max:       float | None = None,
        n_bins:      int   = 400,
) -> dict:
    """
    All-grades (or filtered) sum spectrum with Gaussian fit to Mn Kα.

    Parameters
    ----------
    grades       : int8  (n_events,)
    energy_sum   : float32 (n_events,) [eV]
    out_dir      : output directory
    target_ev    : Mn Kα reference [eV]
    window_frac  : fit window = target_ev × (1 ± window_frac)
    with_bg      : include constant background in Gaussian fit
    grade_filter : list of grades to include (None = all grades)
                   e.g. [0] for singles only, [0,1,2,3,4] for singles+doubles
    e_min/max    : energy axis range
    n_bins       : histogram bins

    Returns
    -------
    fit_result dict (see _fit_kalpha return)
    """
    out_dir = Path(out_dir)

    # Apply grade filter
    if grade_filter is not None:
        mask       = np.isin(grades, grade_filter)
        energy_plt = energy_sum[mask]
        grade_label = (f"grades {grade_filter}"
                       if len(grade_filter) <= 4
                       else f"{len(grade_filter)} grades")
    else:
        energy_plt  = energy_sum
        grade_label = "all grades"

    lo = e_min if e_min is not None else target_ev * 0.60
    hi = e_max if e_max is not None else MN_KBETA_EV * 1.30
    bins    = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(energy_plt, bins=bins)

    # Gaussian fit
    fit = _fit_kalpha(energy_plt, target_ev,
                      window_frac=window_frac,
                      with_bg=with_bg)

    # Print resolution to terminal
    print(f"\n  ── Mn Kα energy resolution ({grade_label}) ──")
    if fit["success"]:
        print(f"     Peak       = {fit['peak_ev']:.2f} eV")
        print(f"     σ          = {fit['sigma_ev']:.2f} eV")
        print(f"     FWHM       = {fit['fwhm_ev']:.2f} eV")
        print(f"     R = FWHM/E = {fit['resolution']:.3f} %")
        print(f"     N in window= {fit['n_events']:,}")
    else:
        print(f"     ⚠  Fit failed: {fit['message']}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Calibrated Fe-55 Spectrum — Mn Kα Resolution Fit\n"
        f"({grade_label},  N={len(energy_plt):,})",
        fontweight="bold", fontsize=12)

    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.step(centers, counts, where="mid",
                color="steelblue", lw=1.2, alpha=0.9,
                label=f"{grade_label}  N={len(energy_plt):,}")

        if fit["success"]:
            xs  = np.linspace(fit["lo_ev"], fit["hi_ev"], 500)
            if with_bg:
                ys = _gauss_plus_bg(xs, fit["amplitude"], fit["peak_ev"],
                                     fit["sigma_ev"], fit["bg"])
            else:
                ys = _gauss(xs, fit["amplitude"],
                             fit["peak_ev"], fit["sigma_ev"])

            ax.plot(xs, ys, "r-", lw=2.5, zorder=5,
                    label=(f"Gaussian fit\n"
                           f"Peak = {fit['peak_ev']:.1f} eV\n"
                           f"FWHM = {fit['fwhm_ev']:.1f} eV\n"
                           f"R    = {fit['resolution']:.2f}%"))

            ax.axvline(fit["peak_ev"], color="red",
                       lw=1.0, ls="--", alpha=0.6)
            ax.axvspan(fit["lo_ev"], fit["hi_ev"],
                       alpha=0.06, color="red", label="Fit window")

            if yscale == "linear":
                # FWHM indicator
                half_max = fit["amplitude"] / 2.0 + fit["bg"]
                hw       = fit["sigma_ev"] * 1.1774   # half-width at half-max
                ax.hlines(half_max,
                          fit["peak_ev"] - hw,
                          fit["peak_ev"] + hw,
                          colors="red", lw=1.5, ls=":",
                          label=f"FWHM = {fit['fwhm_ev']:.1f} eV")

        ax.axvline(target_ev,   color="red",  lw=1.0, ls="--",
                   alpha=0.4, label=f"Mn Kα {target_ev:.0f} eV")
        ax.axvline(MN_KBETA_EV, color="blue", lw=1.0, ls="--",
                   alpha=0.4, label=f"Mn Kβ {MN_KBETA_EV:.0f} eV")

        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Counts / bin")
        ax.set_yscale(yscale)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(f"{'Linear' if yscale == 'linear' else 'Log'} scale")

    fig.tight_layout()
    slug = grade_label.replace(" ", "_").replace(",", "").replace("[", "").replace("]", "")
    p = out_dir / f"cal_spectrum_resolution_{slug}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    return fit


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper — called from cli/energy_cal.py
# ══════════════════════════════════════════════════════════════════════════════

def plot_final_spectrum(
        grades:      np.ndarray,
        energy_sum:  np.ndarray,
        out_dir:     Path,
        target_ev:   float = MN_KALPHA_EV,
        window_frac: float = 0.12,
        with_bg:     bool  = False,
        e_min:       float | None = None,
        e_max:       float | None = None,
        n_bins:      int   = 400,
) -> dict:
    """
    Produce all calibrated spectrum plots and return the resolution fit result.

    Calls:
      1. plot_spectrum_by_grade   → cal_spectrum_by_grade.png
      2. plot_resolution_fit (all grades) → cal_spectrum_resolution_all_grades.png
      3. plot_resolution_fit (singles only) → cal_spectrum_resolution_singles.png

    Returns the fit result dict for all-grades (for saving to HDF5).
    """
    plot_spectrum_by_grade(
        grades, energy_sum, out_dir,
        target_ev=target_ev, e_min=e_min, e_max=e_max, n_bins=n_bins)

    fit_all = plot_resolution_fit(
        grades, energy_sum, out_dir,
        target_ev=target_ev, window_frac=window_frac,
        with_bg=with_bg, grade_filter=None,
        e_min=e_min, e_max=e_max, n_bins=n_bins)

    # Singles only — best energy resolution
    plot_resolution_fit(
        grades, energy_sum, out_dir,
        target_ev=target_ev, window_frac=window_frac,
        with_bg=with_bg, grade_filter=[0],
        e_min=e_min, e_max=e_max, n_bins=n_bins)

    return fit_all
