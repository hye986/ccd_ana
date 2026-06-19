"""
test_redraw_event_rec.py
========================
Redraw diagnostic plots from event_rec_results.h5.

This script reloads the plot-backing data saved by the event_rec stage
and regenerates the same diagnostic plots without rerunning the pipeline.

Usage
-----
    python -m pytest tests/test_redraw_event_rec.py -v
    python tests/test_redraw_event_rec.py [--results RESULTS_FILE]

Output
------
    event_rec_plots/event_maps_global.png    - hit count + mean ADU 2-D maps
    event_rec_plots/spectrum_full_detector.png - per-grade + grouped spectra
    event_rec_plots/grade_distribution.png    - event count per grade bar chart
    event_rec_plots/raw_spectrum_global.png   - pixel-level ADU before recognition
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Grade names (copied from lib/pattern_recognition.py)
GRADE_NAMES = {
    0: "single",
    1: "double→",
    2: "double↓",
    3: "double←",
    4: "double↑",
    5: "triple→+↓",
    6: "triple←+↓",
    7: "triple←+↑",
    8: "triple↑+→",
    9: "quad→+↓+↘",
    10: "quad↓+←+↙",
    11: "quad←+↑+↖",
    12: "quad↑+→+↗",
    13: "other",
}


def redraw_event_rec_plots(results_path: str | Path, output_dir: str | Path | None = None):
    """
    Load event_rec_results.h5 and redraw all event_rec stage plots.

    Parameters
    ----------
    results_path : path to event_rec_results.h5
    output_dir   : output directory for plots (default: ./event_rec_plots)
    """
    from pnccd_ana.analysis import load_event_rec_results

    results_path = Path(results_path)
    if output_dir is None:
        output_dir = results_path.parent / "event_rec_plots"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading event_rec results from: {results_path}")
    data = load_event_rec_results(results_path)

    # ── Extract data ───────────────────────────────────────────────────────────
    raw_spectrum = data.get("raw_spectrum", {})
    grade_dist = data.get("grade_distribution", {})
    meta = data.get("meta", {})

    # ── 1. Hit count + mean ADU maps ───────────────────────────────────────────
    _plot_hitmap_redraw(data, output_dir)

    # ── 2. Spectrum (from grade distribution) ──────────────────────────────────
    _plot_spectrum_redraw(grade_dist, output_dir)

    # ── 3. Grade distribution ──────────────────────────────────────────────────
    _plot_grade_distribution_redraw(grade_dist, output_dir)

    # ── 4. Raw spectrum ───────────────────────────────────────────────────────
    _plot_raw_spectrum_redraw(raw_spectrum, output_dir)

    print(f"\n✓ Event_rec plots redrawn to: {output_dir}/")


def _plot_hitmap_redraw(data: dict, out_dir: Path):
    """Redraw hit count and mean ADU maps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Event Hit Maps", fontsize=14, fontweight="bold")

    grade_dist = data.get("grade_distribution", {})
    spectra = grade_dist.get("spectra", {})

    # Check if we have hit_count/mean_adu data
    # The results file stores spectra per grade, not hit_count directly
    # We'll reconstruct from spectra if available

    if spectra:
        # Get bin edges from first spectrum
        first_grade = min(spectra.keys())
        bin_edges = spectra[first_grade][0]  # tuple (edges, counts)
        n_bins = len(bin_edges) - 1

        # Sum all grades to get total counts per bin
        hit_count = np.zeros(n_bins, dtype=np.int64)
        sum_adu = np.zeros(n_bins, dtype=np.float64)

        for grade, (edges, counts) in spectra.items():
            hit_count += counts
            bin_centres = 0.5 * (edges[:-1] + edges[1:])
            sum_adu += bin_centres * counts

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_adu = np.where(hit_count > 0, sum_adu / hit_count, np.nan)

        # For 2D maps, we'd need the original hit_count/mean_adu arrays
        # These are stored in events.h5 /maps/ section
        ax0, ax1 = axes
    else:
        ax0, ax1 = axes

    ax0.text(0.5, 0.5,
            "Hit count map requires events.h5\n(maps/hit_count)",
            ha="center", va="center", transform=ax0.transAxes,
            fontsize=11)
    ax0.axis("off")

    ax1.text(0.5, 0.5,
            "Mean ADU map requires events.h5\n(maps/mean_adu)",
            ha="center", va="center", transform=ax1.transAxes,
            fontsize=11)
    ax1.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "event_maps_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'event_maps_global.png'}")


def _plot_spectrum_redraw(grade_dist: dict, out_dir: Path):
    """Redraw per-grade and grouped spectra."""
    spectra = grade_dist.get("spectra", {})

    if not spectra:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No spectrum data available",
               ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "spectrum_full_detector.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_dir / 'spectrum_full_detector.png'}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Event Spectrum Analysis", fontsize=14, fontweight="bold")

    # ── Top: All grades individually ─────────────────────────────────────────
    ax_top = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, 14))

    for grade in sorted(spectra.keys()):
        edges, counts = spectra[grade]
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax_top.plot(centres, counts, color=colors[grade % 20],
                   label=f"G{grade} ({GRADE_NAMES.get(grade, '?')})",
                   alpha=0.8, lw=1.0)

    ax_top.set_xlabel("adu_sum [ADU]")
    ax_top.set_ylabel("Count")
    ax_top.set_title("Per-Grade Spectra")
    ax_top.legend(fontsize=7, ncol=3, loc="upper right")
    ax_top.grid(alpha=0.3)
    ax_top.set_xlim(0, None)

    # ── Bottom: Grouped (single, split, other) ─────────────────────────────────
    ax_bot = axes[1]

    single_grades = {0}
    split_grades = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    other_grades = {13}

    for group_name, group_grades in [("Single-pixel", single_grades),
                                      ("Split-event", split_grades),
                                      ("Other", other_grades)]:
        # Sum spectra for this group
        group_edges = None
        group_counts = None
        for grade in group_grades:
            if grade in spectra:
                edges, counts = spectra[grade]
                if group_edges is None:
                    group_edges = edges
                    group_counts = counts.copy()
                else:
                    group_counts += counts

        if group_edges is not None:
            centres = 0.5 * (group_edges[:-1] + group_edges[1:])
            total = group_counts.sum()
            ax_bot.plot(centres, group_counts,
                       label=f"{group_name} (n={total:,})",
                       alpha=0.9, lw=1.5)

    ax_bot.set_xlabel("adu_sum [ADU]")
    ax_bot.set_ylabel("Count")
    ax_bot.set_title("Grouped Spectra")
    ax_bot.legend(fontsize=9)
    ax_bot.grid(alpha=0.3)
    ax_bot.set_xlim(0, None)

    plt.tight_layout()
    fig.savefig(out_dir / "spectrum_full_detector.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'spectrum_full_detector.png'}")


def _plot_grade_distribution_redraw(grade_dist: dict, out_dir: Path):
    """Redraw grade distribution bar chart."""
    grades = grade_dist.get("grades", np.array([]))
    counts = grade_dist.get("counts", np.array([]))

    if len(grades) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No grade distribution data available",
               ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "grade_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_dir / 'grade_distribution.png'}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Grade Distribution", fontsize=14, fontweight="bold")

    # Sort by grade
    sort_idx = np.argsort(grades)
    grades = grades[sort_idx]
    counts = counts[sort_idx]

    colors = plt.cm.Set3(np.linspace(0, 1, len(grades)))
    bars = ax.bar(grades, counts, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    ax.set_xticks(grades)
    ax.set_xticklabels([f"G{g}\n{GRADE_NAMES.get(g, '?')}" for g in grades],
                       fontsize=8)

    # Add count labels on bars
    for bar, count, grade in zip(bars, counts, grades):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"{count:,}", ha="center", va="bottom", fontsize=7)

    # Add percentage labels
    total = counts.sum()
    if total > 0:
        ax2 = ax.twinx()
        ax2.plot(grades, 100 * counts / total, "ro-", markersize=4)
        ax2.set_ylabel("% of Total", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "grade_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'grade_distribution.png'}")


def _plot_raw_spectrum_redraw(raw_spectrum: dict, out_dir: Path):
    """Redraw raw pixel-level spectrum before event recognition."""
    if not raw_spectrum:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No raw spectrum data available\n"
               "(requires sample frames during event_rec)",
               ha="center", va="center", transform=ax.transAxes,
               fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "raw_spectrum_global.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_dir / 'raw_spectrum_global.png'}")
        return

    bin_edges = raw_spectrum.get("bin_edges", np.array([]))
    all_pixels = raw_spectrum.get("all_pixels", np.array([]))
    positive_pixels = raw_spectrum.get("positive_pixels", np.array([]))
    above_seed = raw_spectrum.get("above_seed", np.array([]))
    seed_sigma = raw_spectrum.get("seed_sigma", 5.0)
    seed_threshold = raw_spectrum.get("seed_threshold_adu", 0)
    median_noise = raw_spectrum.get("median_noise_adu", 0)

    if len(bin_edges) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No raw spectrum data",
               ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "raw_spectrum_global.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_dir / 'raw_spectrum_global.png'}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Raw Pixel Spectrum (Before Event Recognition)",
                 fontsize=14, fontweight="bold")

    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot all distributions
    if len(all_pixels) > 0:
        ax.plot(centres, all_pixels, "b-", alpha=0.7,
               label=f"All pixels (n={all_pixels.sum():,})", lw=1.5)

    if len(positive_pixels) > 0:
        ax.plot(centres, positive_pixels, "g-", alpha=0.7,
               label=f"Positive pixels (n={positive_pixels.sum():,})", lw=1.5)

    if len(above_seed) > 0:
        ax.plot(centres, above_seed, "r-", alpha=0.7,
               label=f"Above seed threshold (n={above_seed.sum():,})", lw=1.5)

    # Add threshold line
    if seed_threshold > 0:
        ax.axvline(seed_threshold, color="red", linestyle="--", lw=2,
                  label=f"Seed threshold ({seed_sigma}σ = {seed_threshold:.1f} ADU)")

    ax.set_xlabel("Pixel Value [ADU]")
    ax.set_ylabel("Count per Bin")
    ax.set_title(f"Median noise = {median_noise:.2f} ADU")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(out_dir / "raw_spectrum_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'raw_spectrum_global.png'}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Redraw event_rec stage plots from event_rec_results.h5")
    parser.add_argument("results", nargs="?", default="output/event_rec_results.h5",
                       help="Path to event_rec_results.h5")
    parser.add_argument("-o", "--output", default=None,
                       help="Output directory for plots")
    args = parser.parse_args()

    redraw_event_rec_plots(args.results, args.output)


if __name__ == "__main__":
    main()
