"""
pnccd_ana.plotting.spectrum_plots
==================================
Energy spectrum plotting functions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import _cb, _stats_box, _build_grade_palette, _build_group_label
from ..physics.gain import (RoughGainResult, PeakFitResult, fit_peak,
                             MN_KALPHA_EV, MN_KBETA_EV, _gaussian,
                             SINGLE_GRADES, SPLIT_GRADES)


def plot_final_spectrum(events: np.ndarray,
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

    from ..physics.pattern_recognition import _GRADE_DEFS, GRADE_OTHER
    from ..plotting import _build_grade_palette, _build_group_label

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


def _compute_cti_check_data(
        events:       np.ndarray,
        e_prelim:     np.ndarray,
        e_cti:        np.ndarray,
        row_bin_size: int = 64,
) -> dict:
    """
    Compute before/after CTI correction peak-vs-row data for single events.

    Both e_prelim and e_cti must have the same length as events (all grades).
    The function filters to singles internally using a mask over the full array.

    Returns dict with:
        before: {row_bins, peak_per_bin, peak_success}
        after:  {row_bins, peak_per_bin, peak_success}
    """
    # Build singles mask over the FULL events array
    s_mask   = np.isin(events["grade"], list(SINGLE_GRADES))
    rows_s   = events["Y"][s_mask].astype(int)      # rows of single events
    ep_s     = e_prelim[s_mask]                      # prelim energy, singles only
    ec_s     = e_cti[s_mask]                         # CTI-corrected, singles only

    max_row   = int(rows_s.max()) + 1 if len(rows_s) > 0 else 1
    bin_edges = np.arange(0, max_row + row_bin_size + 1, row_bin_size)
    bin_ctrs  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    result = {
        "before": {"row_bins": bin_ctrs,
                   "peak_per_bin": np.full(len(bin_ctrs), np.nan),
                   "peak_success": np.zeros(len(bin_ctrs), dtype=bool)},
        "after":  {"row_bins": bin_ctrs,
                   "peak_per_bin": np.full(len(bin_ctrs), np.nan),
                   "peak_success": np.zeros(len(bin_ctrs), dtype=bool)},
    }

    for ri, (y0, y1) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        bm = (rows_s >= y0) & (rows_s < y1)   # mask over singles arrays
        if bm.sum() < 30:
            continue
        for key, ep in [("before", ep_s), ("after", ec_s)]:
            res = fit_peak(ep[bm], nominal=MN_KALPHA_EV,
                           window_frac=0.15, n_bins=50, min_events=30)
            if res.success:
                result[key]["peak_per_bin"][ri] = res.peak_ev
                result[key]["peak_success"][ri] = True

    return result


