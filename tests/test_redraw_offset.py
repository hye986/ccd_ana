"""
test_redraw_offset.py
=====================
Redraw diagnostic plots from the offset-calibration .npy files.

This script reloads the calibration arrays saved by the offset stage
(offset, noise, CM map, bad pixel mask) and regenerates diagnostic plots
without rerunning the pipeline.

Usage
-----
    python -m pytest tests/test_redraw_offset.py -v
    python tests/test_redraw_offset.py [--output OUTPUT_DIR]

Output
------
    offset_plots/offsets_global.png     - offset maps + histograms
    offset_plots/noise_global.png       - noise map + histogram
    offset_plots/bad_pixels_global.png  - noise with bad pixels hidden + mask
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def redraw_offset_plots(npy_dir: str | Path,
                        output_dir: str | Path | None = None) -> None:
    """
    Load offset .npy calibration files and redraw the offset stage plots.

    Parameters
    ----------
    npy_dir    : directory containing offset/noise/cm/bad-pixel .npy files
    output_dir : output directory for plots (default: <npy_dir>/offset_plots)
    """
    npy_dir = Path(npy_dir)
    output_dir = Path(output_dir) if output_dir is not None \
        else npy_dir / "offset_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading offset calibration from: {npy_dir}")
    data = _load_npy(npy_dir)

    _plot_offsets_redraw(data, output_dir)
    _plot_noise_redraw(data, output_dir)
    _plot_bad_pixels_redraw(data, output_dir)

    print(f"\n✓ Offset plots redrawn to: {output_dir}/")


def _load_npy(npy_dir: Path) -> dict:
    """Load whichever .npy products are present in *npy_dir*."""
    d: dict = {}
    for key, fname in [("median",         "offset_median.npy"),
                       ("sigmaclip",      "offset_sigclip.npy"),
                       ("noise",          "noise.npy"),
                       ("n_clipped_map",  "n_clipped_map.npy"),
                       ("cm_map",         "cm_map.npy"),
                       ("bad_pixel_mask", "bad_pixel_mask.npy")]:
        fp = npy_dir / fname
        if fp.exists():
            d[key] = np.load(fp)
            print(f"  loaded {fname}  {d[key].shape}")
    for p in sorted(npy_dir.glob("cm_noise_*.npy")):
        d.setdefault("cm_noise", {})[p.stem.split("cm_noise_")[1]] = np.load(p)
    return d


def _plot_offsets_redraw(d: dict, out_dir: Path) -> None:
    """Redraw offset maps and histograms."""
    present = [(n, d[n]) for n in ("median", "sigmaclip") if n in d]
    if not present:
        return
    ncols = len(present) + (1 if len(present) == 2 else 0)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols + 5, 10), squeeze=False)
    fig.suptitle("Offset Maps and Histograms", fontsize=14, fontweight="bold")

    for col, (name, arr) in enumerate(present):
        im = axes[0, col].imshow(arr, origin="lower", cmap="coolwarm",
                                 vmin=np.nanpercentile(arr, 1),
                                 vmax=np.nanpercentile(arr, 99), aspect="auto")
        axes[0, col].set_title(f"{name.capitalize()} Offset Map")
        axes[0, col].set_xlabel("X [detector column]")
        axes[0, col].set_ylabel("Y [detector row]")
        plt.colorbar(im, ax=axes[0, col], label="Offset [ADU]")

        axes[1, col].hist(arr.ravel(), bins=100, alpha=0.7, color=f"C{col}")
        axes[1, col].set_xlabel("Offset [ADU]")
        axes[1, col].set_ylabel("Count")
        axes[1, col].set_title(f"{name.capitalize()} Histogram")
        axes[1, col].grid(alpha=0.3)

    if len(present) == 2:
        diff = present[1][1] - present[0][1]
        lim = float(np.nanpercentile(np.abs(diff), 99)) or 1.0
        im = axes[0, ncols - 1].imshow(diff, origin="lower", cmap="PiYG",
                                       vmin=-lim, vmax=lim, aspect="auto")
        axes[0, ncols - 1].set_title("Difference (sigclip − median)")
        axes[0, ncols - 1].set_xlabel("X [detector column]")
        axes[0, ncols - 1].set_ylabel("Y [detector row]")
        plt.colorbar(im, ax=axes[0, ncols - 1], label="Diff [ADU]")
        axes[1, ncols - 1].hist(diff.ravel(), bins=100, alpha=0.7, color="purple")
        axes[1, ncols - 1].set_xlabel("Difference [ADU]")
        axes[1, ncols - 1].set_ylabel("Count")
        axes[1, ncols - 1].grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / "offsets_global.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _plot_noise_redraw(d: dict, out_dir: Path) -> None:
    """Redraw noise map, histogram, and clip map."""
    noise_map = d.get("noise")
    if noise_map is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Noise Analysis", fontsize=14, fontweight="bold")

    im = axes[0].imshow(noise_map, origin="lower", cmap="viridis",
                        vmin=np.nanpercentile(noise_map, 1),
                        vmax=np.nanpercentile(noise_map, 99), aspect="auto")
    axes[0].set_title("Pixel Noise (RMS)")
    axes[0].set_xlabel("X [detector column]")
    axes[0].set_ylabel("Y [detector row]")
    plt.colorbar(im, ax=axes[0], label="Noise [ADU RMS]")

    axes[1].hist(noise_map.ravel(), bins=100, alpha=0.7, color="green")
    axes[1].axvline(np.nanmedian(noise_map), color="red", lw=2,
                    label=f"median={np.nanmedian(noise_map):.2f}")
    axes[1].set_xlabel("Noise [ADU RMS]")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Noise Distribution")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    n_clipped = d.get("n_clipped_map")
    if n_clipped is not None:
        im = axes[2].imshow(n_clipped, origin="lower", cmap="hot",
                            vmin=0,
                            vmax=float(np.percentile(n_clipped, 95)) or 1.0,
                            aspect="auto")
        axes[2].set_title("Clipped Frames per Pixel")
        axes[2].set_xlabel("X [detector column]")
        axes[2].set_ylabel("Y [detector row]")
        plt.colorbar(im, ax=axes[2], label="N clipped frames")
    else:
        axes[2].axis("off")

    plt.tight_layout()
    p = out_dir / "noise_global.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _plot_bad_pixels_redraw(d: dict, out_dir: Path) -> None:
    """Redraw noise with bad pixels hidden, and the bad pixel mask."""
    noise_map = d.get("noise")
    bad_mask = d.get("bad_pixel_mask")
    if noise_map is None or bad_mask is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Bad Pixel Analysis", fontsize=14, fontweight="bold")

    im = axes[0].imshow(np.where(bad_mask, np.nan, noise_map), origin="lower",
                        cmap="viridis", aspect="auto")
    axes[0].set_title("Noise Map (bad pixels hidden)")
    axes[0].set_xlabel("X [detector column]")
    axes[0].set_ylabel("Y [detector row]")
    plt.colorbar(im, ax=axes[0], label="ADU RMS")

    axes[1].imshow(bad_mask, origin="lower", cmap="gray_r", aspect="auto")
    axes[1].set_title(f"Bad Pixel Mask (N={int(bad_mask.sum())})")
    axes[1].set_xlabel("X [detector column]")
    axes[1].set_ylabel("Y [detector row]")

    plt.tight_layout()
    p = out_dir / "bad_pixels_global.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Redraw offset stage plots from .npy calibration files")
    parser.add_argument("npy_dir", nargs="?", default="output",
                        help="Directory containing offset .npy files")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    redraw_offset_plots(args.npy_dir, args.output)


if __name__ == "__main__":
    main()
