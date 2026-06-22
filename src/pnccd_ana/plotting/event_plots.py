"""
pnccd_ana.plotting.event_plots
===============================
Event recognition plotting functions.

No imports from physics.gain / physics.cti / physics.calibrate —
those belong to the energy_cal stage only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..io.geometry import ASIC_COLORS, ADC_MAX
from .common import _cb, _stats_box, _adjust_bin_range


# ══════════════════════════════════════════════════════════════════════════════
# Common-mode map
# ══════════════════════════════════════════════════════════════════════════════

def plot_cm_map(
        scope_name:  str,
        cm_map:      np.ndarray,
        out_dir:     Path,
        asic_names:  list[str] | None = None,
) -> None:
    """
    (Frame × Y) heatmap of CM correction values.

    For per-ASIC CM (3-D input), produces one plot per ASIC.
    """
    out_dir = Path(out_dir)

    if cm_map.ndim == 3 and asic_names:
        for i, name in enumerate(asic_names):
            asic_cm = cm_map[:, :, i]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Common-Mode Correction — ASIC {name}",
                         fontsize=13, fontweight="bold")

            vext = float(np.percentile(np.abs(asic_cm), 99))
            im   = axes[0].imshow(asic_cm, origin="lower", cmap="RdBu_r",
                                   vmin=-vext, vmax=vext, aspect="auto")
            axes[0].set_xlabel("Y [detector row]")
            axes[0].set_ylabel("Frame index")
            axes[0].set_title(f"CM Value per (Frame, Detector Row) — {name}")
            _cb(axes[0], im)

            axes[1].plot(asic_cm.mean(axis=1), lw=0.8, color="slateblue")
            axes[1].axhline(0, color="k", lw=0.5, ls="--")
            axes[1].set_xlabel("Frame index")
            axes[1].set_ylabel("Mean CM (ADU)")
            axes[1].set_title(f"Frame-Average CM — {name}")
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            p = out_dir / f"cm_map_{name}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {p}")
        return

    # 2-D legacy path (n_frames, n_Y)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Common-Mode Correction — {scope_name}",
                 fontsize=13, fontweight="bold")

    vext = float(np.percentile(np.abs(cm_map), 99))
    im   = axes[0].imshow(cm_map, origin="lower", cmap="RdBu_r",
                           vmin=-vext, vmax=vext, aspect="auto")
    axes[0].set_xlabel("Y [detector row]")
    axes[0].set_ylabel("Frame index")
    axes[0].set_title("CM Value per (Frame, Detector Row)")
    _cb(axes[0], im)

    axes[1].plot(cm_map.mean(axis=1), lw=0.8, color="slateblue")
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Mean CM (ADU)")
    axes[1].set_title("Frame-Average CM  (outliers = bad frames)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"cm_map_{scope_name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Hit map
# ══════════════════════════════════════════════════════════════════════════════

def plot_hitmap(
        hit_count:  np.ndarray,
        mean_adu:   np.ndarray,
        out_dir:    Path,
        asics:      list[str] | None = None,
) -> None:
    """Hit-count and mean-ADU 2-D maps for the full detector."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fe-55 Event Maps  (full detector)",
                 fontsize=13, fontweight="bold")

    pos = hit_count[hit_count > 0]
    if len(pos):
        im0 = axes[0].imshow(hit_count, origin="lower", cmap="hot",
                              norm=LogNorm(vmin=1, vmax=int(pos.max())),
                              aspect="auto")
    else:
        im0 = axes[0].imshow(hit_count, origin="lower", cmap="hot",
                              aspect="auto")
    axes[0].set_title("Hit Count (log)")
    _cb(axes[0], im0, "hits")
    axes[0].set_xlabel("X [detector column]")
    axes[0].set_ylabel("Y [detector row]")

    finite = mean_adu[np.isfinite(mean_adu)]
    if len(finite):
        v0, v1 = np.percentile(finite, [1, 99])
        im1 = axes[1].imshow(mean_adu, origin="lower", cmap="plasma",
                              vmin=v0, vmax=v1, aspect="auto")
    else:
        im1 = axes[1].imshow(mean_adu, origin="lower", cmap="plasma",
                              aspect="auto")
    axes[1].set_title("Mean Event ADU")
    _cb(axes[1], im1, "ADU")
    axes[1].set_xlabel("X [detector column]")
    axes[1].set_ylabel("Y [detector row]")

    plt.tight_layout()
    p = out_dir / "event_maps_global.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Spectrum by cluster size
# ══════════════════════════════════════════════════════════════════════════════

def plot_spectrum(
        spectra:      dict[int, np.ndarray],
        bin_edges:    np.ndarray,
        out_dir:      Path,
        title_suffix: str  = "full detector",
        cluster_data: dict | None = None,   # replaces old `events` arg
) -> None:
    """
    Cluster-size spectra (adu_sum = cluster-summed charge).

    spectra keys are n_pixels (1=single, 2=double, 3=triple, 4=quad, 5=large).
    Grade assignment is deferred to energy_cal.

    Panel 1: per cluster-size spectra (log scale)
    Panel 2: seed vs cluster-sum comparison (if cluster_data supplied)
    """
    out_dir = Path(out_dir)

    # Optionally adjust bin range from cluster_data adu sums
    if cluster_data is not None and len(cluster_data.get("flag", [])) > 0:
        from ..physics.event_filter import adu_sums as _adu_sums, seed_pixels
        sums = _adu_sums(cluster_data)
        bin_edges = _adjust_bin_range(sums, bin_edges)
        _, _, seed_adus = seed_pixels(cluster_data)
    else:
        seed_adus = None

    centres  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_panels = 2 if seed_adus is not None else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5),
                              squeeze=False)
    fig.suptitle(f"Fe-55 Spectrum — {title_suffix}\n"
                 f"(cluster-summed ADU, no grade assignment yet)",
                 fontsize=12, fontweight="bold")

    _SIZE_COLOUR = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800",
                    4: "#E91E63", 5: "#9C27B0"}
    _SIZE_LABEL  = {1: "single (n=1)", 2: "double (n=2)",
                    3: "triple (n=3)", 4: "quadruple (n=4)",
                    5: "large (n≥5)"}

    ax = axes[0, 0]
    total = np.zeros(len(centres), dtype=np.int64)
    for n in sorted(_SIZE_COLOUR):
        counts = spectra.get(n, np.zeros(len(centres), dtype=np.int64))
        if counts.sum() == 0:
            continue
        ax.step(centres, counts, where="mid",
                color=_SIZE_COLOUR[n], lw=1.2, alpha=0.85,
                label=_SIZE_LABEL[n])
        total += counts
    ax.step(centres, total, where="mid", color="black",
            lw=1.0, ls="--", alpha=0.7,
            label=f"all (cluster sum)")
    ax.set_xlabel("Summed ADU (cluster)")
    ax.set_ylabel("Counts / bin")
    ax.set_title("Per Cluster Size\n"
                 "(grade assigned after energy calibration)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    if n_panels == 2 and seed_adus is not None:
        from ..physics.event_filter import adu_sums as _adu_sums
        sums = _adu_sums(cluster_data)
        ax2  = axes[0, 1]
        c_seed, _ = np.histogram(seed_adus, bins=bin_edges)
        c_sum,  _ = np.histogram(sums,      bins=bin_edges)
        ax2.step(centres, c_seed, where="mid", color="tomato",    lw=1.2,
                 alpha=0.9, label="seed pixel ADU")
        ax2.step(centres, c_sum,  where="mid", color="steelblue", lw=1.2,
                 alpha=0.9, label="cluster sum ADU")
        ax2.set_xlabel("ADU")
        ax2.set_ylabel("Counts / bin")
        ax2.set_title("Seed vs Cluster-sum\n"
                      "(splits push cluster sum right of seed)")
        ax2.set_yscale("log")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)

    plt.tight_layout()
    slug = title_suffix.replace(" ", "_").replace("/", "_")
    p = out_dir / f"spectrum_{slug}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Cluster size distribution bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_cluster_size_distribution(
        cluster_data: dict,
        out_dir:      Path,
) -> None:
    """
    Bar chart of event count per cluster size (n_pixels).

    Uses cluster_data CSR dict — no EVENT_DTYPE needed.
    Grade distribution will be plotted by energy_cal after calibration.
    """
    out_dir = Path(out_dir)
    from ..physics.event_filter import n_pixels_per_event
    npix = n_pixels_per_event(cluster_data)

    sizes  = [1, 2, 3, 4, 5, 6]
    counts = [int((npix == n).sum()) for n in sizes[:-1]]
    counts.append(int((npix >= 6).sum()))   # merge n≥6 into last bin
    labels = ["single\n(n=1)", "double\n(n=2)", "triple\n(n=3)",
              "quad\n(n=4)",   "penta\n(n=5)",  "large\n(n≥6)"]
    colours = ["#2196F3", "#4CAF50", "#FF9800",
               "#E91E63", "#9C27B0", "#607D8B"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(sizes)), counts,
                  color=colours, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Cluster size (pixels)")
    ax.set_ylabel("Event count")
    ax.set_title("Event Count per Cluster Size\n"
                 "(grade assigned after energy calibration)",
                 fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, cnt * 1.15,
                    f"{cnt:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    p = out_dir / "cluster_size_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Raw pixel spectrum
# ══════════════════════════════════════════════════════════════════════════════

def plot_raw_spectrum(
        corrected_frames: np.ndarray,
        noise_map:        np.ndarray,
        bin_edges:        np.ndarray,
        out_dir:          Path,
        seed_sigma:       float = 5.0,
        title_suffix:     str   = "global",
) -> None:
    """
    Pixel-level ADU histograms from CM-corrected frames.

    Three distributions:
      grey : all CM-corrected pixel values
      blue : pixels > 0 ADU
      red  : pixels above seed threshold (seed_sigma × noise)
    """
    out_dir  = Path(out_dir)
    centres  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    corr_flat  = corrected_frames.ravel()
    pos_flat   = corr_flat[corr_flat > 0]
    seed_thr   = seed_sigma * noise_map
    above_seed = corrected_frames > seed_thr[np.newaxis]
    hits_flat  = corrected_frames[above_seed].ravel()

    med_noise  = float(np.median(noise_map))
    seed_adu   = seed_sigma * med_noise

    c_all,  _ = np.histogram(corr_flat, bins=bin_edges)
    c_pos,  _ = np.histogram(pos_flat,  bins=bin_edges)
    c_hits, _ = np.histogram(hits_flat, bins=bin_edges)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Pixel-Level ADU Distribution — {title_suffix}\n"
        f"(Before event recognition / cluster summation.  "
        f"seed={seed_sigma}σ ≈ {seed_adu:.0f} ADU)",
        fontsize=12, fontweight="bold")

    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.step(centres, c_all,  where="mid", color="lightgray",
                lw=0.8, alpha=0.9,
                label="all pixels (offset peak near 0)")
        ax.step(centres, c_pos,  where="mid", color="steelblue",
                lw=1.0, alpha=0.85, label="pixels > 0 ADU")
        ax.step(centres, c_hits, where="mid", color="tomato",
                lw=1.2, alpha=0.9,
                label=f"pixels > seed ({seed_sigma}σ ≈ {seed_adu:.0f} ADU)")
        ax.axvline(seed_adu, color="tomato", lw=1.0, ls="--", alpha=0.7)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("ADU  (CM-corrected, per pixel — not cluster sum)")
        ax.set_ylabel("Pixel count / bin")
        ax.set_title(f"{'Linear' if yscale == 'linear' else 'Log'} scale"
                     f"  —  {len(hits_flat):,} pixels above seed")
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    slug = title_suffix.replace(" ", "_")
    p = out_dir / f"raw_spectrum_{slug}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")
