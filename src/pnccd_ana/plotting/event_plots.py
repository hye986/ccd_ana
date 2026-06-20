"""
pnccd_ana.plotting.event_plots
===============================
Event recognition plotting functions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..io.geometry import ASIC_COLORS, ADC_MAX
from ..physics.pattern_recognition import N_GRADES, GRADE_NAMES, _GRADE_DEFS, GRADE_OTHER
from .common import _cb, _stats_box, _get_grade_palette, _get_group_label, _adjust_bin_range


def plot_cm_map(
        scope_name:  str,
        cm_map:      np.ndarray,
        out_dir:     Path,
        asic_names:  list[str] | None = None,
) -> None:
    """
    (Frame × Y) heatmap of CM correction values.
    
    For per-ASIC CM, shows separate plots for each ASIC.
    Files are named cm_map_{name}.png (scope is implicit from directory).
    """
    # Handle 3D cm_map (per-ASIC: n_frames, n_Y, n_asics)
    if cm_map.ndim == 3 and asic_names:
        # Per-ASIC CM correction - create one plot per ASIC
        for i, name in enumerate(asic_names):
            asic_cm = cm_map[:, :, i]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Common-Mode Correction — ASIC {name}", fontsize=13, fontweight="bold")
            
            vext = float(np.percentile(np.abs(asic_cm), 99))
            im = axes[0].imshow(asic_cm, origin="lower", cmap="RdBu_r",
                                 vmin=-vext, vmax=vext, aspect="auto")
            axes[0].set_xlabel("Y [detector row]"); axes[0].set_ylabel("Frame index")
            axes[0].set_title(f"CM Value per (Frame, Detector Row) — {name}")
            _cb(axes[0], im)

            axes[1].plot(asic_cm.mean(axis=1), lw=0.8, color="slateblue")
            axes[1].axhline(0, color="k", lw=0.5, ls="--")
            axes[1].set_xlabel("Frame index"); axes[1].set_ylabel("Mean CM (ADU)")
            axes[1].set_title(f"Frame-Average CM — {name}"); axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            p = out_dir / f"cm_map_{name}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  → {p}")
        return
    
    # Legacy 2D cm_map (n_frames, n_Y)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Common-Mode Correction — {scope_name}", fontsize=13, fontweight="bold")

    vext = float(np.percentile(np.abs(cm_map), 99))
    im = axes[0].imshow(cm_map, origin="lower", cmap="RdBu_r",
                         vmin=-vext, vmax=vext, aspect="auto")
    axes[0].set_xlabel("Y [detector row]"); axes[0].set_ylabel("Frame index")
    axes[0].set_title("CM Value per (Frame, Detector Row)")
    _cb(axes[0], im)

    axes[1].plot(cm_map.mean(axis=1), lw=0.8, color="slateblue")
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_xlabel("Frame index"); axes[1].set_ylabel("Mean CM (ADU)")
    axes[1].set_title("Frame-Average CM  (outliers = bad frames)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"cm_map_{scope_name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Source analysis plots
# ──────────────────────────────────────────────────────────────────────────────


def plot_hitmap(hit_count: np.ndarray, mean_adu: np.ndarray,
                out_dir: Path, asics: list[str] | None = None) -> None:
    """Hit-count and mean-ADU 2-D maps for the full detector."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fe-55 Event Maps  (full detector)", fontsize=13, fontweight="bold")

    pos = hit_count[hit_count > 0]
    if len(pos):
        im0 = axes[0].imshow(hit_count, origin="lower", cmap="hot",
                              norm=LogNorm(vmin=1, vmax=int(pos.max())), aspect="auto")
    else:
        im0 = axes[0].imshow(hit_count, origin="lower", cmap="hot", aspect="auto")
    axes[0].set_title("Hit Count (log)"); _cb(axes[0], im0, "hits")
    axes[0].set_xlabel("X [detector column]"); axes[0].set_ylabel("Y [detector row]")

    finite = mean_adu[np.isfinite(mean_adu)]
    if len(finite):
        v0, v1 = np.percentile(finite, [1, 99])
        im1 = axes[1].imshow(mean_adu, origin="lower", cmap="plasma",
                              vmin=v0, vmax=v1, aspect="auto")
    else:
        im1 = axes[1].imshow(mean_adu, origin="lower", cmap="plasma", aspect="auto")
    axes[1].set_title("Mean Event ADU"); _cb(axes[1], im1, "ADU")
    axes[1].set_xlabel("X [detector column]"); axes[1].set_ylabel("Y [detector row]")

    plt.tight_layout()
    p = out_dir / "event_maps_global.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")



def plot_spectrum(spectra: dict[int, np.ndarray], bin_edges: np.ndarray,
                  out_dir: Path, title_suffix: str = "full detector",
                  events: np.ndarray | None = None) -> None:
    """
    Per-grade and grouped event spectra (adu_sum = cluster-summed charge).

    If *events* is supplied, also plots the seed spectrum (adu_seed = centre
    pixel only) overlaid on the all-grades sum, so you can see the shift
    caused by charge splitting.

    Automatically detects if the bin range misses the peaks and adjusts.
    """
    if events is not None and len(events):
        bin_edges = _adjust_bin_range(events, bin_edges)

    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_panels = 3 if (events is not None and len(events)) else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 5))
    fig.suptitle(f"Fe-55 Spectrum  ({title_suffix})", fontsize=13,
                 fontweight="bold")

    # Panel 1: per-grade — iterate only over grades that exist in palette
    ax = axes[0]
    for g in sorted(_get_grade_palette().keys()):
        counts = spectra.get(g, None)
        if counts is None or counts.sum() == 0:
            continue
        ax.step(centres, counts, where="mid",
                color=_get_grade_palette()[g], alpha=0.75, lw=0.9,
                label=f"G{g} {GRADE_NAMES.get(g,'')}")
    ax.set_xlabel("Summed ADU (cluster)"); ax.set_ylabel("Counts / bin")
    ax.set_title("Per-Grade  (cluster-summed charge)")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)

    # Panel 2: grouped — built dynamically from _GRADE_DEFS
    ax2 = axes[1]
    from ..physics.pattern_recognition import _GRADE_DEFS, GRADE_OTHER
    from collections import defaultdict

    # Re-group by label prefix (same logic as _build_group_label)
    prefix_groups: dict[str, list[int]] = defaultdict(list)
    for gid, label, _ in _GRADE_DEFS:
        prefix_groups[label.split()[0]].append(gid)

    for prefix, gids in sorted(prefix_groups.items(),
                                key=lambda kv: min(kv[1])):
        gids_sorted = sorted(gids)
        total = sum(spectra.get(g, np.zeros(len(centres), dtype=int))
                    for g in gids_sorted)
        if total.sum() == 0:
            continue
        key = gids_sorted[0]
        ax2.step(centres, total, where="mid",
                 color=_get_grade_palette().get(key, "#aaaaaa"),
                 lw=1.2, alpha=0.9,
                 label=_get_group_label().get(key, prefix))

    # "other" group
    other_counts = spectra.get(GRADE_OTHER,
                               np.zeros(len(centres), dtype=int))
    if other_counts.sum() > 0:
        ax2.step(centres, other_counts, where="mid",
                 color=_get_grade_palette()[GRADE_OTHER],
                 lw=1.2, alpha=0.9,
                 label=_get_group_label()[GRADE_OTHER])

    # All-grades sum
    all_c = sum(spectra.get(g, np.zeros(len(centres), dtype=int))
                for g in _get_grade_palette().keys())
    ax2.step(centres, all_c, where="mid", color="black",
             lw=1.0, ls="--", alpha=0.8, label="all grades (cluster sum)")
    ax2.set_xlabel("Summed ADU (cluster)"); ax2.set_ylabel("Counts / bin")
    ax2.set_title("Grouped  (cluster-summed charge)")
    ax2.set_yscale("log"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    # Panel 3: seed vs sum comparison (only when events available)
    if n_panels == 3 and events is not None:
        ax3 = axes[2]
        c_seed, _ = np.histogram(events["adu_seed"], bins=bin_edges)
        c_sum,  _ = np.histogram(events["adu_sum"],  bins=bin_edges)
        ax3.step(centres, c_seed, where="mid", color="tomato", lw=1.2,
                 alpha=0.9, label="seed pixel only  (centre, not summed)")
        ax3.step(centres, c_sum,  where="mid", color="steelblue", lw=1.2,
                 alpha=0.9, label="cluster sum  (adu_sum)")
        ax3.set_xlabel("ADU"); ax3.set_ylabel("Counts / bin")
        ax3.set_title("Seed vs Cluster-sum\n"
                      "(splits push adu_sum right of adu_seed)")
        ax3.set_yscale("log"); ax3.grid(alpha=0.3); ax3.legend(fontsize=8)
        ax3.text(0.02, 0.97,
                 "If peaks coincide → mostly singles\n"
                 "If adu_sum peak shifts right → charge shared\n"
                 "adu_seed always shows single-pixel energy",
                 transform=ax3.transAxes, va="top", fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    slug = title_suffix.replace(" ","_").replace("/","_")
    p = out_dir / f"spectrum_{slug}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")



def plot_grade_distribution(events: np.ndarray, out_dir: Path) -> None:
    """Bar chart of event count per grade."""
    # Build grade list dynamically so new grades are picked up automatically
    all_grades = sorted(_get_grade_palette().keys())
    counts     = [int((events["grade"] == g).sum()) for g in all_grades]
    labels     = [f"G{g}" for g in all_grades]
    colours    = [_get_grade_palette()[g] for g in all_grades]

    fig, ax = plt.subplots(figsize=(max(12, len(all_grades)), 4))
    bars = ax.bar(range(len(all_grades)), counts,
                  color=colours, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(all_grades)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Grade"); ax.set_ylabel("Event count")
    ax.set_title("Event Count per Grade", fontsize=12, fontweight="bold")
    ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width()/2, cnt*1.1,
                    f"{cnt:,}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    p = out_dir / "grade_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")



def plot_raw_spectrum(
        corrected_frames:  np.ndarray,
        noise_map:         np.ndarray,
        bin_edges:         np.ndarray,
        out_dir:           Path,
        seed_sigma:        float = 5.0,
        title_suffix:      str = "global",
) -> None:
    """
    Pixel-level ADU histograms from CM-corrected frames — before event recognition.

    This shows the pixel-value distribution from CM-corrected frames,
    NOT the summed-cluster spectrum from find_events().  The two plots serve
    different purposes:

      plot_raw_spectrum : pixel-level view — offset peak near 0, single-pixel
                          photon hits appearing as a shoulder/peak at higher ADU.
                          Useful for checking the threshold and offset subtraction.

      plot_spectrum_*   : event-level view — events whose cluster pixels are summed.
                          A double-pixel event with two 800 ADU pixels appears at
                          ~1600 ADU, not 800 ADU.

    Three distributions shown:
      grey   : all CM-corrected pixel values  (dominated by the offset peak at ~0)
      blue   : pixels above zero only         (offset-subtracted view)
      red    : pixels above the seed threshold (candidates entering event recognition)
    """
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    corr_flat  = corrected_frames.ravel()
    pos_flat   = corr_flat[corr_flat > 0]

    seed_thr   = seed_sigma * noise_map
    above_seed = corrected_frames > seed_thr[np.newaxis]
    hits_flat  = corrected_frames[above_seed].ravel()

    med_noise  = float(np.median(noise_map))
    seed_adu   = seed_sigma * med_noise

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Pixel-Level ADU Distribution — {title_suffix}\n"
        f"(Before event recognition / cluster summation.  seed={seed_sigma}σ ≈ {seed_adu:.0f} ADU)",
        fontsize=12, fontweight="bold")

    for ax, yscale in zip(axes, ["linear", "log"]):
        c_all, _  = np.histogram(corr_flat, bins=bin_edges)
        c_pos, _  = np.histogram(pos_flat,  bins=bin_edges)
        c_hits, _ = np.histogram(hits_flat, bins=bin_edges)

        ax.step(centres, c_all,  where="mid", color="lightgray",
                lw=0.8, alpha=0.9, label="all pixels (offset peak near 0)")
        ax.step(centres, c_pos,  where="mid", color="steelblue",
                lw=1.0, alpha=0.85, label="pixels > 0 ADU")
        ax.step(centres, c_hits, where="mid", color="tomato",
                lw=1.2, alpha=0.9,
                label=f"pixels > seed ({seed_sigma}σ ≈ {seed_adu:.0f} ADU)\n"
                      f"[individual pixel hits, NOT cluster sums]")

        ax.axvline(seed_adu, color="tomato", lw=1.0, ls="--", alpha=0.7)
        ax.axvline(0, color="k", lw=0.5, ls=":")

        ax.set_xlabel("ADU  (CM-corrected, per pixel — not summed over cluster)")
        ax.set_ylabel("Pixel count / bin")
        ax.set_title(f"{'Linear' if yscale == 'linear' else 'Log'} scale  "
                     f"— {len(hits_flat):,} pixels above seed threshold")
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    slug = title_suffix.replace(" ", "_")
    p = out_dir / f"raw_spectrum_{slug}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")
