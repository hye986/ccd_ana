"""
test_redraw_offset.py
=====================
Redraw diagnostic plots from offset_results.h5.

This script reloads the plot-backing data saved by the offset stage
and regenerates the same diagnostic plots without rerunning the pipeline.

Usage
-----
    python -m pytest tests/test_redraw_offset.py -v
    python tests/test_redraw_offset.py [--output OUTPUT_DIR]

Output
------
    offset_plots/offsets_global.png     - offset maps + histograms
    offset_plots/noise_global.png       - noise map, CM noise, clip map
    offset_plots/cm_map_C{i}.png        - CM correction per ASIC
    offset_plots/bad_pixels_global.png  - bad pixel categories + per-ASIC stats
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def redraw_offset_plots(results_path: str | Path, output_dir: str | Path | None = None):
    """
    Load offset_results.h5 and redraw all offset stage plots.

    Parameters
    ----------
    results_path : path to offset_results.h5
    output_dir   : output directory for plots (default: ./offset_plots)
    """
    from pnccd_ana.analysis import load_offset_results

    results_path = Path(results_path)
    if output_dir is None:
        output_dir = results_path.parent / "offset_plots"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading offset results from: {results_path}")
    data = load_offset_results(results_path)

    # ── Extract data ───────────────────────────────────────────────────────────
    offsets = data.get("offsets", {})
    noise = data.get("noise", {})
    bad_pixels = data.get("bad_pixels", {})
    meta = data.get("meta", {})

    # ── 1. Offset maps + histograms ─────────────────────────────────────────────
    _plot_offsets_redraw(offsets, output_dir, meta)

    # ── 2. Noise map + CM noise + clip map ────────────────────────────────────
    _plot_noise_redraw(noise, output_dir, meta)

    # ── 3. Bad pixel categories ────────────────────────────────────────────────
    _plot_bad_pixels_redraw(bad_pixels, noise, output_dir, meta)

    print(f"\n✓ Offset plots redrawn to: {output_dir}/")


def _plot_offsets_redraw(offsets: dict, out_dir: Path, meta: dict):
    """Redraw offset maps and histograms."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Offset Maps and Histograms", fontsize=14, fontweight="bold")

    methods = ["median", "sigmaclip"]
    for col, method in enumerate(methods):
        if method not in offsets:
            continue
        m = offsets[method]

        # Map
        ax_map = axes[0, col]
        im = ax_map.imshow(m["map"], origin="lower", cmap="coolwarm",
                          vmin=np.nanpercentile(m["map"], 1),
                          vmax=np.nanpercentile(m["map"], 99))
        ax_map.set_title(f"{method.capitalize()} Offset Map")
        ax_map.set_xlabel("X [detector column]")
        ax_map.set_ylabel("Y [detector row]")
        plt.colorbar(im, ax=ax_map, label="Offset [ADU]")

        # Histogram
        ax_hist = axes[1, col]
        ax_hist.hist(m["map"].ravel(), bins=100, alpha=0.7, color=f"C{col}")
        ax_hist.set_xlabel("Offset [ADU]")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"{method.capitalize()} Histogram")
        ax_hist.grid(alpha=0.3)

        # Stats
        mean_val = np.nanmean(m["map"])
        std_val = np.nanstd(m["map"])
        ax_hist.text(0.97, 0.97,
                    f"μ={mean_val:.1f}\nσ={std_val:.1f}",
                    transform=ax_hist.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Difference map (if both methods exist)
    if "median" in offsets and "sigmaclip" in offsets:
        ax_diff = axes[0, 2]
        diff = offsets["sigmaclip"]["map"] - offsets["median"]["map"]
        im = ax_diff.imshow(diff, origin="lower", cmap="PiYG",
                           vmin=-np.nanpercentile(np.abs(diff), 99),
                           vmax=np.nanpercentile(np.abs(diff), 99))
        ax_diff.set_title("Difference (sigclip - median)")
        ax_diff.set_xlabel("X [detector column]")
        ax_diff.set_ylabel("Y [detector row]")
        plt.colorbar(im, ax=ax_diff, label="Diff [ADU]")

        ax_hist2 = axes[1, 2]
        ax_hist2.hist(diff.ravel(), bins=100, alpha=0.7, color="purple")
        ax_hist2.set_xlabel("Difference [ADU]")
        ax_hist2.set_ylabel("Count")
        ax_hist2.set_title("Difference Histogram")
        ax_hist2.grid(alpha=0.3)
    else:
        axes[0, 2].axis("off")
        axes[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "offsets_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'offsets_global.png'}")


def _plot_noise_redraw(noise: dict, out_dir: Path, meta: dict):
    """Redraw noise map, CM noise, and clip map."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Noise Analysis", fontsize=14, fontweight="bold")

    # Noise map
    ax0 = axes[0]
    noise_map = noise.get("map")
    if noise_map is not None:
        im = ax0.imshow(noise_map, origin="lower", cmap="viridis",
                       vmin=np.nanpercentile(noise_map, 1),
                       vmax=np.nanpercentile(noise_map, 99))
        ax0.set_title("Pixel Noise (RMS)")
        ax0.set_xlabel("X [detector column]")
        ax0.set_ylabel("Y [detector row]")
        plt.colorbar(im, ax=ax0, label="Noise [ADU RMS]")
        ax0.text(0.03, 0.97, f"median={np.nanmedian(noise_map):.2f}",
                transform=ax0.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax0.text(0.5, 0.5, "No noise data", ha="center", va="center",
                transform=ax0.transAxes)
        ax0.axis("off")

    # Histogram
    ax1 = axes[1]
    if noise_map is not None:
        ax1.hist(noise_map.ravel(), bins=100, alpha=0.7, color="green")
        ax1.axvline(np.nanmedian(noise_map), color="red", lw=2,
                   label=f"median={np.nanmedian(noise_map):.2f}")
        ax1.set_xlabel("Noise [ADU RMS]")
        ax1.set_ylabel("Count")
        ax1.set_title("Noise Distribution")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No noise data", ha="center", va="center",
                transform=ax1.transAxes)
        ax1.axis("off")

    # Clipped pixels map
    ax2 = axes[2]
    n_clipped = noise.get("n_clipped_map")
    if n_clipped is not None:
        im = ax2.imshow(n_clipped, origin="lower", cmap="hot",
                       vmin=0, vmax=np.percentile(n_clipped, 95) if n_clipped.max() > 0 else 1)
        ax2.set_title("Clipped Frames per Pixel")
        ax2.set_xlabel("X [detector column]")
        ax2.set_ylabel("Y [detector row]")
        plt.colorbar(im, ax=ax2, label="N clipped frames")
    else:
        ax2.text(0.5, 0.5, "No clip data", ha="center", va="center",
                transform=ax2.transAxes)
        ax2.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "noise_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'noise_global.png'}")


def _plot_bad_pixels_redraw(bad_pixels: dict, noise: dict,
                            out_dir: Path, meta: dict):
    """Redraw bad pixel categories and per-ASIC stats."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Bad Pixel Analysis", fontsize=14, fontweight="bold")

    # Bad pixel mask
    ax0 = axes[0]
    if "mask" in bad_pixels:
        mask = bad_pixels["mask"]
        im = ax0.imshow(mask.astype(float), origin="lower", cmap="Reds",
                       vmin=0, vmax=1)
        ax0.set_title(f"Bad Pixel Mask (n={mask.sum():,})")
        ax0.set_xlabel("X [detector column]")
        ax0.set_ylabel("Y [detector row]")
        plt.colorbar(im, ax=ax0, label="Bad (1) / Good (0)")
    else:
        ax0.text(0.5, 0.5, "No bad pixel data", ha="center", va="center",
                transform=ax0.transAxes)
        ax0.axis("off")

    # Noise histogram with thresholds
    ax1 = axes[1]
    if "noise_hist_edges" in bad_pixels and "noise_hist_counts" in bad_pixels:
        edges = bad_pixels["noise_hist_edges"]
        counts = bad_pixels["noise_hist_counts"]
        ax1.hist(edges[:-1], bins=edges, weights=counts, alpha=0.7, color="blue")

        thresholds = bad_pixels.get("thresholds", {})
        if thresholds:
            hot = thresholds.get("hot_threshold")
            cold = thresholds.get("cold_threshold")
            if hot:
                ax1.axvline(hot, color="red", lw=2, ls="--",
                           label=f"HOT threshold={hot:.2f}")
            if cold:
                ax1.axvline(cold, color="orange", lw=2, ls="--",
                           label=f"COLD threshold={cold:.2f}")
            ax1.legend(fontsize=9)

        ax1.set_xlabel("Noise [ADU RMS]")
        ax1.set_ylabel("Count")
        ax1.set_title("Noise Distribution with Thresholds")
        ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No noise histogram", ha="center", va="center",
                transform=ax1.transAxes)
        ax1.axis("off")

    # Per-ASIC bad pixel counts
    ax2 = axes[2]
    if "per_asic" in bad_pixels:
        per_asic = bad_pixels["per_asic"]
        n_bad = per_asic.get("n_bad", [])
        labels = per_asic.get("asic_labels", [])

        if len(n_bad) > 0:
            colors = plt.cm.Set3(np.linspace(0, 1, len(n_bad)))
            bars = ax2.bar(labels if labels else range(len(n_bad)),
                          n_bad, color=colors)
            ax2.set_xlabel("ASIC")
            ax2.set_ylabel("N Bad Pixels")
            ax2.set_title("Bad Pixels per ASIC")
            ax2.tick_params(axis="x", rotation=45)

            # Add count labels on bars
            for bar, count in zip(bars, n_bad):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        str(count), ha="center", va="bottom", fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No per-ASIC data", ha="center", va="center",
                    transform=ax2.transAxes)
            ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "No per-ASIC data", ha="center", va="center",
                transform=ax2.transAxes)
        ax2.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "bad_pixels_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir / 'bad_pixels_global.png'}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Redraw offset stage plots from offset_results.h5")
    parser.add_argument("results", nargs="?", default="output/offset_results.h5",
                       help="Path to offset_results.h5")
    parser.add_argument("-o", "--output", default=None,
                       help="Output directory for plots")
    args = parser.parse_args()

    redraw_offset_plots(args.results, args.output)


if __name__ == "__main__":
    main()
