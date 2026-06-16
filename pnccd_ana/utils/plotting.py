"""
pnccd_ana.utils.plotting
========================
Plotting utilities for dark-frame calibration and Fe-55 source analysis.

Single-hybrid mode (512×512 or 1024×512 frames).

All 2-D maps follow the detector convention:
  x-axis: X (detector column),  y-axis: Y (detector row)
  imshow(arr, origin="lower") — NO transpose needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..lib.geometry import ASIC_COLORS, ADC_MAX
from ..lib.pattern_recognition import N_GRADES, GRADE_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cb(ax, im, label="ADU"):
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)


def _stats_box(ax, arr, fmt=".2f"):
    flat = arr.ravel()
    txt  = (f"μ={float(flat.mean()):{fmt}}\n"
            f"σ={float(flat.std()):{fmt}}\n"
            f"med={float(np.median(flat)):{fmt}}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))


def _asic_guides(ax):
    """Stub for backward compatibility — does nothing in single-hybrid mode."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Dark-frame calibration plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_offsets(scope_name: str, r: dict, out_dir: Path,
                 active_mask: np.ndarray | None = None) -> None:
    """2-D offset maps + histograms for both/either method."""
    present = []
    if "offset_median"  in r: present.append(("Median",     r["offset_median"]))
    if "offset_sigclip" in r: present.append(("Sigma-Clip", r["offset_sigclip"]))
    if not present:
        return

    show_diff = len(present) == 2
    ncols = len(present) + int(show_diff)
    fig, axes = plt.subplots(2, ncols, figsize=(5.5*ncols, 9), squeeze=False)
    fig.suptitle(f"Offsets (Pedestal) — {scope_name}", fontsize=13, fontweight="bold")

    all_vals = np.concatenate([a.ravel() for _, a in present])
    vmin, vmax = np.percentile(all_vals, [1, 99])

    for col, (mname, arr) in enumerate(present):
        # Apply masked region visualization
        disp = arr.copy()
        if active_mask is not None:
            disp = np.where(active_mask, disp, np.nan)
        
        im = axes[0, col].imshow(disp, origin="lower", cmap="viridis",
                                  vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, col].set_title(mname, fontsize=10)
        axes[0, col].set_xlabel("X [detector column]")
        axes[0, col].set_ylabel("Y [detector row]")
        _cb(axes[0, col], im); _stats_box(axes[0, col], arr, fmt=".1f")
        flat = arr.ravel()
        axes[1, col].hist(flat, bins=200,
                          range=(np.percentile(flat, 0.5), np.percentile(flat, 99.5)),
                          color="steelblue", edgecolor="none", alpha=0.85)
        axes[1, col].set_xlabel("Offset (ADU)"); axes[1, col].set_ylabel("Pixel count")
        axes[1, col].grid(axis="y", alpha=0.3); _stats_box(axes[1, col], flat, fmt=".1f")

    if show_diff:
        diff   = present[0][1] - present[1][1]
        if active_mask is not None:
            diff = np.where(active_mask, diff, np.nan)
        absmax = np.percentile(np.abs(diff), 99)
        im = axes[0, ncols-1].imshow(diff, origin="lower", cmap="RdBu_r",
                                      vmin=-absmax, vmax=absmax, aspect="auto")
        axes[0, ncols-1].set_title("Difference (Median − Sigma-Clip)", fontsize=10)
        axes[0, ncols-1].set_xlabel("X [detector column]")
        axes[0, ncols-1].set_ylabel("Y [detector row]")
        _cb(axes[0, ncols-1], im)
        flat = diff.ravel()
        axes[1, ncols-1].hist(flat, bins=200,
                               range=(np.percentile(flat, 0.5), np.percentile(flat, 99.5)),
                               color="coral", edgecolor="none", alpha=0.85)
        axes[1, ncols-1].set_xlabel("Δ Offset (ADU)")
        axes[1, ncols-1].set_ylabel("Pixel count")
        axes[1, ncols-1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"offsets_{scope_name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


def plot_noise(scope_name: str, r: dict, out_dir: Path,
               active_mask: np.ndarray | None = None) -> None:
    """Noise map, histogram, CM-noise profile, and clipped-frames map."""
    noise     = r.get("noise")
    cm_noise  = r.get("cm_noise")
    n_clipped = r.get("n_clipped_map")
    if noise is None:
        return

    ncols = 3 + int(n_clipped is not None)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5*ncols, 5))
    fig.suptitle(f"Electronic Noise — {scope_name}", fontsize=13, fontweight="bold")

    # Apply active_mask to noise for visualization
    noise_disp = noise.copy()
    if active_mask is not None:
        noise_disp = np.where(active_mask, noise_disp, np.nan)
    
    im = axes[0].imshow(noise_disp, origin="lower", cmap="inferno",
                         vmin=0, vmax=float(np.percentile(noise, 99)), aspect="auto")
    axes[0].set_title("Per-Pixel Noise (RMS)")
    axes[0].set_xlabel("X [detector column]"); axes[0].set_ylabel("Y [detector row]")
    _cb(axes[0], im, "ADU RMS"); _stats_box(axes[0], noise)

    flat = noise.ravel()
    axes[1].hist(flat, bins=200, range=(0, float(np.percentile(flat, 99.5))),
                 color="darkorange", edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("Noise (ADU RMS)"); axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Noise Distribution"); axes[1].grid(axis="y", alpha=0.3)
    _stats_box(axes[1], flat)

    if cm_noise is not None:
        axes[2].plot(np.arange(len(cm_noise)), cm_noise, color="teal", lw=0.9)
        axes[2].set_xlabel("Y [detector row]"); axes[2].set_ylabel("CM Noise (ADU RMS)")
        axes[2].set_title("CM Noise per Detector Row"); axes[2].grid(alpha=0.3)
        _stats_box(axes[2], cm_noise)

    if n_clipped is not None:
        n_max = int(n_clipped.max()) or 1
        n_clipped_disp = n_clipped.copy()
        if active_mask is not None:
            n_clipped_disp = np.where(active_mask, n_clipped_disp, np.nan)
        im2 = axes[3].imshow(n_clipped_disp, origin="lower", cmap="hot_r",
                              vmin=0, vmax=n_max, aspect="auto")
        axes[3].set_title("Frames Clipped / Pixel")
        axes[3].set_xlabel("X [detector column]"); axes[3].set_ylabel("Y [detector row]")
        _cb(axes[3], im2, "# frames")
        axes[3].text(0.97, 0.97,
                     f"avg={float(n_clipped.mean()):.2f}\nmax={n_max}",
                     transform=axes[3].transAxes, ha="right", va="top", fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

    plt.tight_layout()
    p = out_dir / f"noise_{scope_name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


def plot_bad_pixels(
        scope_name:   str,
        noise_map:    np.ndarray,
        bad_mask:     np.ndarray,
        out_dir:      Path,
        active_mask:  np.ndarray | None = None,
) -> None:
    """
    Visualize bad pixel detection results.
    
    Shows:
    - Noise map with bad pixels overlaid
    - Bad pixel categories (hot, cold, clipped)
    - Per-ASIC statistics
    
    Parameters
    ----------
    scope_name  : label for the scope (e.g., "global", "H0")
    noise_map   : float (Y, X) — per-pixel RMS
    bad_mask    : bool  (Y, X) — True where pixel is bad
    out_dir     : output directory
    active_mask : bool  (Y, X) — True where pixel is in active region
    """
    Y, X = noise_map.shape
    
    # Categorize bad pixels
    nonfinite = ~np.isfinite(noise_map)
    nonpositive = (noise_map <= 0) & (active_mask if active_mask is not None else np.ones_like(noise_map, dtype=bool))
    
    ref = noise_map[active_mask & np.isfinite(noise_map) & (noise_map > 0)]
    if ref.size > 0:
        med = float(np.median(ref))
    else:
        med = 1.0
    
    # Hot: noise > 5x median (using the same thresholds as build_bad_pixel_mask)
    hot_thr = 5.0 * med
    cold_thr = 0.1 * med
    
    hot_mask = (noise_map > hot_thr) & active_mask if active_mask is not None else (noise_map > hot_thr)
    cold_mask = (noise_map < cold_thr) & np.isfinite(noise_map) & active_mask if active_mask is not None else (noise_map < cold_thr) & np.isfinite(noise_map)
    
    n_hot = int(hot_mask.sum())
    n_cold = int(cold_mask.sum())
    n_nonfinite = int((nonfinite & (active_mask if active_mask is not None else np.ones_like(noise_map, dtype=bool))).sum())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Bad Pixel Analysis — {scope_name}", fontsize=13, fontweight="bold")
    
    # Panel 1: Noise map with bad pixels overlaid
    ax = axes[0, 0]
    im = ax.imshow(noise_map, origin="lower", cmap="viridis",
                   vmin=0, vmax=float(np.percentile(noise_map[np.isfinite(noise_map)], 99)),
                   aspect="auto")
    ax.set_title("Noise Map (with bad pixels marked)")
    ax.set_xlabel("X [detector column]"); ax.set_ylabel("Y [detector row]")
    _cb(ax, im, "ADU RMS")
    
    # Overlay bad pixels in red
    bad_overlay = np.zeros((*bad_mask.shape, 4), dtype=np.float32)
    bad_overlay[bad_mask, 0] = 1.0  # Red channel
    bad_overlay[bad_mask, 3] = 0.7  # Alpha
    ax.imshow(bad_overlay, origin="lower", aspect="auto")
    
    # Panel 2: Bad pixel categorization map
    ax = axes[0, 1]
    cat_map = np.zeros((Y, X), dtype=int)
    cat_map[hot_mask] = 1        # Hot
    cat_map[cold_mask] = 2       # Cold
    cat_map[nonfinite] = 3       # Non-finite
    cat_map[bad_mask & ~hot_mask & ~cold_mask & ~nonfinite] = 4  # Other bad
    
    # Custom colormap: 0=good (transparent), 1=hot (red), 2=cold (blue), 3=nonfinite (purple), 4=other (gray)
    from matplotlib.colors import ListedColormap
    colors = ['green', 'red', 'blue', 'purple', 'gray']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    
    im = ax.imshow(cat_map, origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto")
    ax.set_title("Bad Pixel Categories")
    ax.set_xlabel("X [detector column]"); ax.set_ylabel("Y [detector row]")
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'Good ({int((~bad_mask).sum())})'),
        Patch(facecolor='red', label=f'Hot (>{hot_thr:.1f} RMS) ({n_hot})'),
        Patch(facecolor='blue', label=f'Cold (<{cold_thr:.2f} RMS) ({n_cold})'),
        Patch(facecolor='purple', label=f'Non-finite ({n_nonfinite})'),
        Patch(facecolor='gray', label=f'Other ({int(bad_mask.sum()) - n_hot - n_cold - n_nonfinite})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Panel 3: Noise distribution with thresholds marked
    ax = axes[1, 0]
    flat = noise_map[np.isfinite(noise_map) & (noise_map > 0)].ravel()
    ax.hist(flat, bins=200, range=(0, float(np.percentile(flat, 99.5))),
            color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(med, color='green', lw=2, ls='-', label=f'Median ({med:.2f})')
    ax.axvline(hot_thr, color='red', lw=2, ls='--', label=f'Hot threshold ({hot_thr:.2f})')
    ax.axvline(cold_thr, color='blue', lw=2, ls='--', label=f'Cold threshold ({cold_thr:.3f})')
    ax.set_xlabel("Noise (ADU RMS)"); ax.set_ylabel("Pixel count")
    ax.set_title("Noise Distribution with Bad Pixel Thresholds")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    
    # Panel 4: Per-ASIC bad pixel statistics
    ax = axes[1, 1]
    asic_width = 64
    n_asics = X // asic_width
    
    asic_stats = []
    asic_labels = []
    for i in range(n_asics):
        x0, x1 = i * asic_width, (i + 1) * asic_width
        asic_bad = bad_mask[:, x0:x1].sum()
        asic_total = x1 - x0
        asic_active = (active_mask[:, x0:x1] if active_mask is not None else np.ones((Y, asic_width), dtype=bool))[:, x0:x1].sum()
        asic_stats.append(asic_bad)
        asic_labels.append(f'ASIC{i}')
    
    bars = ax.bar(asic_labels, asic_stats, color='tomato', edgecolor='white')
    ax.set_xlabel("ASIC"); ax.set_ylabel("Bad Pixel Count")
    ax.set_title("Bad Pixels per ASIC")
    ax.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, asic_stats):
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 5,
                f"{cnt}", ha='center', va='bottom', fontsize=9)
    
    # Summary text
    total_bad = int(bad_mask.sum())
    total_pixels = int(active_mask.sum()) if active_mask is not None else X * Y
    summary = f"Total bad: {total_bad}/{total_pixels} ({100*total_bad/total_pixels:.2f}%)"
    fig.text(0.5, 0.02, summary, ha='center', fontsize=11, 
             bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    p = out_dir / f"bad_pixels_{scope_name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


def plot_cm_map(
        scope_name:  str,
        cm_map:      np.ndarray,
        out_dir:     Path,
        asic_names:  list[str] | None = None,
) -> None:
    """
    (Frame × X) heatmap of CM correction values.
    
    For per-ASIC CM, shows separate plots for each ASIC.
    """
    # Handle 3D cm_map (per-ASIC: n_frames, n_Y, n_asics)
    if cm_map.ndim == 3 and asic_names:
        # Per-ASIC CM correction - create one plot per ASIC
        n_asics = len(asic_names)
        for i, name in enumerate(asic_names):
            asic_cm = cm_map[:, :, i]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Common-Mode Correction — {scope_name} {name}", fontsize=13, fontweight="bold")
            
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
            p = out_dir / f"cm_map_{scope_name}_{name}.png"
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

_GRADE_PALETTE = {
    0: "#2176ae",
    1: "#f7931e", 2: "#f7931e", 3: "#f7931e", 4: "#f7931e",
    5: "#57cc99", 6: "#57cc99", 7: "#57cc99", 8: "#57cc99",
    9: "#c77dff", 10: "#c77dff", 11: "#c77dff", 12: "#c77dff",
    13: "#aaaaaa",
}
_GROUP_LABEL = {
    0:  "single (G0)",
    1:  "double (G1-4)",
    5:  "triple (G5-8)",
    9:  "quadruple (G9-12)",
    13: "other (G13)",
}


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


def _auto_bin_edges(events: np.ndarray,
                    bin_edges: np.ndarray) -> np.ndarray:
    """
    Return bin_edges adjusted to cover the actual event adu_sum range.

    If fewer than 1% of events fall within the supplied bin range, issue
    a clear warning and return auto-ranged edges so the plot is not empty.
    """
    if len(events) == 0:
        return bin_edges
    lo, hi = bin_edges[0], bin_edges[-1]
    n_bins  = len(bin_edges) - 1
    in_range = ((events["adu_sum"] >= lo) & (events["adu_sum"] <= hi)).sum()
    if in_range < 0.01 * len(events):
        p1, p99 = np.percentile(events["adu_sum"], [0.5, 99.5])
        margin  = (p99 - p1) * 0.1
        # Do NOT clamp to the original lo/hi — when auto-ranging is needed the
        # original bounds are wrong by definition.  Floor at 0 (ADU ≥ 0).
        new_lo  = max(p1 - margin, 0.0)
        new_hi  = p99 + margin
        if new_hi <= new_lo:          # degenerate data guard
            new_hi = new_lo + 1.0
        print(f"  ⚠  WARNING: only {in_range}/{len(events)} events fall within "
              f"adu_min={lo:.0f}..adu_max={hi:.0f}.")
        print(f"     Auto-ranging to {new_lo:.0f}..{new_hi:.0f}. "
              f"Update adu_min/adu_max in your config to suppress this.")
        return np.linspace(new_lo, new_hi, n_bins + 1)
    return bin_edges


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
        bin_edges = _auto_bin_edges(events, bin_edges)

    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_panels = 3 if (events is not None and len(events)) else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 5))
    fig.suptitle(f"Fe-55 Spectrum  ({title_suffix})", fontsize=13,
                 fontweight="bold")

    # Panel 1: per-grade
    ax = axes[0]
    for g in range(N_GRADES):
        if spectra.get(g, np.array([])).sum() == 0:
            continue
        ax.step(centres, spectra[g], where="mid",
                color=_GRADE_PALETTE[g], alpha=0.75, lw=0.9,
                label=f"G{g} {GRADE_NAMES.get(g,'')}")
    ax.set_xlabel("Summed ADU (cluster)"); ax.set_ylabel("Counts / bin")
    ax.set_title("Per-Grade  (cluster-summed charge)")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)

    # Panel 2: grouped
    ax2 = axes[1]
    groups = [(0,[0]),(1,[1,2,3,4]),(5,[5,6,7,8]),(9,[9,10,11,12]),(13,[13])]
    for gkey, gids in groups:
        total = sum(spectra.get(g, np.zeros(len(centres), dtype=int)) for g in gids)
        if total.sum() == 0:
            continue
        ax2.step(centres, total, where="mid",
                 color=_GRADE_PALETTE[gkey], lw=1.2, alpha=0.9,
                 label=_GROUP_LABEL[gkey])
    all_c = sum(spectra.get(g, np.zeros(len(centres), dtype=int))
                for g in range(N_GRADES))
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
    counts = np.zeros(N_GRADES, dtype=int)
    for g in range(N_GRADES):
        counts[g] = int((events["grade"] == g).sum())

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(N_GRADES), counts,
                  color=[_GRADE_PALETTE[g] for g in range(N_GRADES)],
                  edgecolor="white", lw=0.5)
    ax.set_xticks(range(N_GRADES))
    ax.set_xticklabels([f"G{g}" for g in range(N_GRADES)], fontsize=8)
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

      plot_raw_spectrum : pixel-level view — pedestal peak near 0, single-pixel
                          photon hits appearing as a shoulder/peak at higher ADU.
                          Useful for checking the threshold and pedestal subtraction.

      plot_spectrum_*   : event-level view — events whose cluster pixels are summed.
                          A double-pixel event with two 800 ADU pixels appears at
                          ~1600 ADU, not 800 ADU.

    Three distributions shown:
      grey   : all CM-corrected pixel values  (dominated by the pedestal peak at ~0)
      blue   : pixels above zero only         (pedestal-subtracted view)
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
                lw=0.8, alpha=0.9, label="all pixels (pedestal peak near 0)")
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
