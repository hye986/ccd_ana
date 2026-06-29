"""
pnccd_ana.plotting.offset_plots
===============================
Offset calibration plotting functions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..io.geometry import ASIC_COLORS, ASIC_NAMES, ASIC_MASK, ADC_MAX
from .common import _cb, _stats_box


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
        _cb(axes[0, col], im); _stats_box(axes[0, col], disp, fmt=".1f")

        # 1D histogram: only active pixels
        if active_mask is not None:
            flat = arr[active_mask]
        else:
            flat = arr.ravel()
        if flat.size > 0:
            axes[1, col].hist(flat, bins=200,
                              range=(np.percentile(flat, 0.5), np.percentile(flat, 99.5)),
                              color="steelblue", edgecolor="none", alpha=0.85)
            _stats_box(axes[1, col], flat, fmt=".1f")
        axes[1, col].set_xlabel("Offset (ADU)"); axes[1, col].set_ylabel("Pixel count")
        axes[1, col].grid(axis="y", alpha=0.3)

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
        flat_valid = flat[~np.isnan(flat)]
        if flat_valid.size > 0:
            axes[1, ncols-1].hist(flat_valid, bins=200,
                                   range=(np.percentile(flat_valid, 0.5), np.percentile(flat_valid, 99.5)),
                                   color="coral", edgecolor="none", alpha=0.85)
            _stats_box(axes[1, ncols-1], flat_valid, fmt=".2f")
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
    _cb(axes[0], im, "ADU RMS"); _stats_box(axes[0], noise_disp)

    # Apply active_mask to noise histogram (1D plot)
    if active_mask is not None:
        flat = noise[active_mask]
    else:
        flat = noise.ravel()
    if flat.size > 0:
        axes[1].hist(flat, bins=200, range=(0, float(np.percentile(flat, 99.5))),
                     color="darkorange", edgecolor="none", alpha=0.85)
        _stats_box(axes[1], flat)
    axes[1].set_xlabel("Noise (ADU RMS)"); axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Noise Distribution"); axes[1].grid(axis="y", alpha=0.3)

    if cm_noise is not None:
        # cm_noise may be a dict {asic_name: array} for multi-ASIC, or a single array
        if isinstance(cm_noise, dict):
            for name, vals in cm_noise.items():
                axes[2].plot(np.arange(len(vals)), vals, lw=0.9, label=name)
            axes[2].legend(fontsize=7, ncol=2)
        else:
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
    # Count good pixels only from active region
    if active_mask is not None:
        n_good = int((~bad_mask & active_mask).sum())
    else:
        n_good = int((~bad_mask).sum())
    legend_elements = [
        Patch(facecolor='green', label=f'Good ({n_good})'),
        Patch(facecolor='red', label=f'Hot (>{hot_thr:.1f} RMS) ({n_hot})'),
        Patch(facecolor='blue', label=f'Cold (<{cold_thr:.2f} RMS) ({n_cold})'),
        Patch(facecolor='purple', label=f'Non-finite ({n_nonfinite})'),
        Patch(facecolor='gray', label=f'Other ({int(bad_mask.sum()) - n_hot - n_cold - n_nonfinite})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Panel 3: Noise distribution with thresholds marked (apply active_mask)
    ax = axes[1, 0]
    mask_3d = np.isfinite(noise_map) & (noise_map > 0)
    if active_mask is not None:
        mask_3d = mask_3d & active_mask
    flat = noise_map[mask_3d].ravel()
    if flat.size > 0:
        ax.hist(flat, bins=200, range=(0, float(np.percentile(flat, 99.5))),
                color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(med, color='green', lw=2, ls='-', label=f'Median ({med:.2f})')
    ax.axvline(hot_thr, color='red', lw=2, ls='--', label=f'Hot threshold ({hot_thr:.2f})')
    ax.axvline(cold_thr, color='blue', lw=2, ls='--', label=f'Cold threshold ({cold_thr:.3f})')
    ax.set_xlabel("Noise (ADU RMS)"); ax.set_ylabel("Pixel count")
    ax.set_title("Noise Distribution with Bad Pixel Thresholds")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Panel 4: Per-ASIC bad pixel statistics (respect ASIC mask)
    ax = axes[1, 1]
    asic_width = 64
    n_asics = X // asic_width

    asic_stats = []
    asic_labels = []
    for i in range(n_asics):
        # Use reversed index to match ASIC naming (C7=cols 0-63, C0=cols 448-511)
        rev = n_asics - 1 - i
        asic_idx = rev  # numeric ASIC index
        x0, x1 = i * asic_width, (i + 1) * asic_width
        asic_bad = bad_mask[:, x0:x1].sum()
        asic_stats.append(asic_bad)
        # Use C{n} naming with color coding; grey out masked ASICs
        label = f"C{asic_idx}"
        asic_labels.append(label)

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


