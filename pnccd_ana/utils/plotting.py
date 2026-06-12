"""
pnccd_ana.utils.plotting
========================
Plotting utilities for dark-frame calibration and Fe-55 source analysis.

All 2-D maps follow the detector convention:
  x-axis: X (detector row),  y-axis: Y (detector column),  origin="lower"
  imshow(arr) — NO transpose needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..lib.geometry import (ASIC_SLICES, ALL_ASICS, ASIC_LABEL,
                             ASIC_COLORS, ASIC_GRID_POS, ADC_MAX)
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
    """Overlay ASIC boundary lines and name labels on a full-frame 2-D plot."""
    ax.axhline(511.5, color="white", lw=0.9, ls="--", alpha=0.7)
    ax.axvline(511.5, color="white", lw=0.9, ls="--", alpha=0.7)
    for aname, (Y0, Y1, X0, X1) in ASIC_SLICES.items():
        ax.text((X0+X1)/2, (Y0+Y1)/2, aname,
                ha="center", va="center", color="white",
                fontsize=14, fontweight="bold", alpha=0.4)


# ──────────────────────────────────────────────────────────────────────────────
# Dark-frame calibration plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_offsets(scope_name: str, r: dict, out_dir: Path) -> None:
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
        im = axes[0, col].imshow(arr, origin="lower", cmap="viridis",
                                  vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, col].set_title(mname, fontsize=10)
        axes[0, col].set_xlabel("X (detector row)")
        axes[0, col].set_ylabel("Y (detector column)")
        _cb(axes[0, col], im); _stats_box(axes[0, col], arr, fmt=".1f")
        flat = arr.ravel()
        axes[1, col].hist(flat, bins=200,
                          range=(np.percentile(flat, 0.5), np.percentile(flat, 99.5)),
                          color="steelblue", edgecolor="none", alpha=0.85)
        axes[1, col].set_xlabel("Offset (ADU)"); axes[1, col].set_ylabel("Pixel count")
        axes[1, col].grid(axis="y", alpha=0.3); _stats_box(axes[1, col], flat, fmt=".1f")

    if show_diff:
        diff   = present[0][1] - present[1][1]
        absmax = np.percentile(np.abs(diff), 99)
        im = axes[0, ncols-1].imshow(diff, origin="lower", cmap="RdBu_r",
                                      vmin=-absmax, vmax=absmax, aspect="auto")
        axes[0, ncols-1].set_title("Difference (Median − Sigma-Clip)", fontsize=10)
        axes[0, ncols-1].set_xlabel("X (detector row)")
        axes[0, ncols-1].set_ylabel("Y (detector column)")
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


def plot_noise(scope_name: str, r: dict, out_dir: Path) -> None:
    """Noise map, histogram, CM-noise profile, and clipped-frames map."""
    noise     = r.get("noise")
    cm_noise  = r.get("cm_noise")
    n_clipped = r.get("n_clipped_map")
    if noise is None:
        return

    ncols = 3 + int(n_clipped is not None)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5*ncols, 5))
    fig.suptitle(f"Electronic Noise — {scope_name}", fontsize=13, fontweight="bold")

    im = axes[0].imshow(noise, origin="lower", cmap="inferno",
                         vmin=0, vmax=float(np.percentile(noise, 99)), aspect="auto")
    axes[0].set_title("Per-Pixel Noise (RMS)")
    axes[0].set_xlabel("X (detector row)"); axes[0].set_ylabel("Y (detector column)")
    _cb(axes[0], im, "ADU RMS"); _stats_box(axes[0], noise)

    flat = noise.ravel()
    axes[1].hist(flat, bins=200, range=(0, float(np.percentile(flat, 99.5))),
                 color="darkorange", edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("Noise (ADU RMS)"); axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Noise Distribution"); axes[1].grid(axis="y", alpha=0.3)
    _stats_box(axes[1], flat)

    if cm_noise is not None:
        axes[2].plot(np.arange(len(cm_noise)), cm_noise, color="teal", lw=0.9)
        axes[2].set_xlabel("X (detector row)"); axes[2].set_ylabel("CM Noise (ADU RMS)")
        axes[2].set_title("CM Noise per Detector Row"); axes[2].grid(alpha=0.3)
        _stats_box(axes[2], cm_noise)

    if n_clipped is not None:
        n_max = int(n_clipped.max()) or 1
        im2 = axes[3].imshow(n_clipped, origin="lower", cmap="hot_r",
                              vmin=0, vmax=n_max, aspect="auto")
        axes[3].set_title("Frames Clipped / Pixel")
        axes[3].set_xlabel("X (detector row)"); axes[3].set_ylabel("Y (detector column)")
        _cb(axes[3], im2, "# frames")
        axes[3].text(0.97, 0.97,
                     f"avg={float(n_clipped.mean()):.2f}\nmax={n_max}",
                     transform=axes[3].transAxes, ha="right", va="top", fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

    plt.tight_layout()
    p = out_dir / f"noise_{scope_name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


def plot_cm_map(scope_name: str, cm_map: np.ndarray, out_dir: Path) -> None:
    """(Frame × X) heatmap of CM correction values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Common-Mode Correction — {scope_name}", fontsize=13, fontweight="bold")

    vext = float(np.percentile(np.abs(cm_map), 99))
    im = axes[0].imshow(cm_map.T, origin="lower", cmap="RdBu_r",
                         vmin=-vext, vmax=vext, aspect="auto")
    axes[0].set_xlabel("Frame index"); axes[0].set_ylabel("X (detector row)")
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


def plot_asic_overview(results_asics: dict[str, dict], quantity: str,
                       title: str, label: str, cmap: str,
                       out_dir: Path, log_scale: bool = False) -> None:
    """2×2 physical ASIC grid for a single calibration quantity."""
    available = {k: v for k, v in results_asics.items() if quantity in v}
    if not available:
        return

    sample = next(iter(available.values()))[quantity]
    is_1d  = sample.ndim == 1

    vkw: dict = {}
    if not is_1d:
        all_v = np.concatenate([v[quantity].ravel() for v in available.values()])
        if log_scale:
            pos  = all_v[all_v > 0]
            vkw  = {"norm": LogNorm(vmin=float(pos.min()), vmax=float(pos.max()))}
        else:
            vkw  = {"vmin": float(np.percentile(all_v, 1)),
                    "vmax": float(np.percentile(all_v, 99))}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Per-ASIC  {title}", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "← low X  |  high X →  (detector row)",
             ha="center", fontsize=9, color="gray")
    fig.text(0.01, 0.5, "↑ high Y  |  low Y ↓  (detector column)",
             va="center", rotation="vertical", fontsize=9, color="gray")

    for aname in ALL_ASICS:
        gr, gc = ASIC_GRID_POS[aname]
        ax = axes[gr, gc]
        if aname in available:
            arr = available[aname][quantity]
            ax.set_title(f"{aname}  ({ASIC_LABEL[aname]})",
                         color=ASIC_COLORS[aname], fontsize=10, fontweight="bold")
            if is_1d:
                ax.plot(np.arange(len(arr)), arr,
                        color=ASIC_COLORS.get(aname, "gray"), lw=0.9)
                ax.set_xlabel("X (ASIC-local)"); ax.set_ylabel(label)
                ax.grid(alpha=0.3); _stats_box(ax, arr)
            else:
                im = ax.imshow(arr, origin="lower", cmap=cmap,
                               aspect="auto", **vkw)
                ax.set_xlabel("X (ASIC-local)"); ax.set_ylabel("Y (ASIC-local)")
                _cb(ax, im, label); _stats_box(ax, arr)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    slug = (quantity + ("_log" if log_scale else "")).replace("_","")
    p = out_dir / f"asic_overview_{slug}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


def plot_summary_dashboard(results_global: dict, out_dir: Path) -> None:
    """One-page summary: offset map + noise map + CM noise profile."""
    method_key = ("offset_median"  if "offset_median"  in results_global
                  else "offset_sigclip" if "offset_sigclip" in results_global
                  else None)
    offset   = results_global.get(method_key) if method_key else None
    noise    = results_global.get("noise")
    cm_noise = results_global.get("cm_noise")

    panels = [(t, a) for t, a in [("Offset", offset), ("Noise", noise)] if a is not None]
    ncols  = len(panels) + int(cm_noise is not None)
    if ncols == 0:
        return

    fig, axes = plt.subplots(1, ncols, figsize=(6.5*ncols, 5.5))
    if ncols == 1: axes = [axes]
    fig.suptitle("pnCCD Dark-Frame Calibration Summary", fontsize=13, fontweight="bold")

    cmaps  = {"Offset": "viridis", "Noise": "inferno"}
    labels = {"Offset": "ADU",     "Noise": "ADU RMS"}

    for idx, (pt, arr) in enumerate(panels):
        im = axes[idx].imshow(arr, origin="lower", cmap=cmaps[pt],
                               vmin=float(np.percentile(arr, 1)),
                               vmax=float(np.percentile(arr, 99)), aspect="auto")
        axes[idx].set_title(pt, fontsize=11)
        axes[idx].set_xlabel("X (detector row)"); axes[idx].set_ylabel("Y (detector column)")
        _asic_guides(axes[idx]); _cb(axes[idx], im, labels[pt])
        _stats_box(axes[idx], arr, fmt=".1f")

    if cm_noise is not None:
        ax = axes[len(panels)]
        ax.plot(np.arange(len(cm_noise)), cm_noise, color="teal", lw=0.9)
        ax.set_xlabel("X (detector row)"); ax.set_ylabel("CM Noise (ADU RMS)")
        ax.set_title("CM Noise per Detector Row"); ax.grid(alpha=0.3)
        ax.axvline(511.5, color="gray", lw=0.8, ls="--", label="ASIC boundary")
        ax.legend(fontsize=8)

    plt.tight_layout()
    p = out_dir / "summary_dashboard.png"
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
    axes[0].set_xlabel("X (detector row)"); axes[0].set_ylabel("Y (detector column)")
    _asic_guides(axes[0])

    finite = mean_adu[np.isfinite(mean_adu)]
    if len(finite):
        v0, v1 = np.percentile(finite, [1, 99])
        im1 = axes[1].imshow(mean_adu, origin="lower", cmap="plasma",
                              vmin=v0, vmax=v1, aspect="auto")
    else:
        im1 = axes[1].imshow(mean_adu, origin="lower", cmap="plasma", aspect="auto")
    axes[1].set_title("Mean Event ADU"); _cb(axes[1], im1, "ADU")
    axes[1].set_xlabel("X (detector row)"); axes[1].set_ylabel("Y (detector column)")
    _asic_guides(axes[1])

    plt.tight_layout()
    p = out_dir / "event_maps_global.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


def plot_asic_hitmaps(hit_count: np.ndarray, asics: list[str],
                      out_dir: Path) -> None:
    """2×2 per-ASIC hit-count maps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
    fig.suptitle("Hit Count per ASIC", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "← low X  |  high X →", ha="center", fontsize=9, color="gray")
    fig.text(0.01, 0.5, "↑ high Y  |  low Y ↓", va="center",
             rotation="vertical", fontsize=9, color="gray")

    for aname in ALL_ASICS:
        gr, gc = ASIC_GRID_POS[aname]
        ax = axes[gr, gc]
        if aname not in asics:
            ax.set_visible(False); continue
        Y0, Y1, X0, X1 = ASIC_SLICES[aname]
        sub = hit_count[Y0:Y1+1, X0:X1+1]
        pos = sub[sub > 0]
        im  = (ax.imshow(sub, origin="lower", cmap="hot",
                          norm=LogNorm(vmin=1, vmax=int(pos.max())), aspect="auto")
               if len(pos) else
               ax.imshow(sub, origin="lower", cmap="hot", aspect="auto"))
        ax.set_title(f"{aname}  ({ASIC_LABEL[aname]})",
                     color=ASIC_COLORS[aname], fontsize=10, fontweight="bold")
        ax.set_xlabel("X (ASIC-local)"); ax.set_ylabel("Y (ASIC-local)")
        _cb(ax, im, "hits")

    plt.tight_layout()
    p = out_dir / "hit_maps_per_asic.png"
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


def plot_asic_spectra(events: np.ndarray, bin_edges: np.ndarray,
                      asics: list[str], out_dir: Path) -> None:
    """2×2 per-ASIC spectrum grid (singles + all grades)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    fig.suptitle("Fe-55 Spectra per ASIC", fontsize=13, fontweight="bold")
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for aname in ALL_ASICS:
        gr, gc = ASIC_GRID_POS[aname]
        ax = axes[gr, gc]
        if aname not in asics:
            ax.set_visible(False); continue

        Y0, Y1, X0, X1 = ASIC_SLICES[aname]
        m_asic = ((events["Y"] >= Y0) & (events["Y"] <= Y1) &
                  (events["X"] >= X0) & (events["X"] <= X1))

        for grade, lbl, color in [(0, "single", "#2176ae"), (None, "all", "black")]:
            m = m_asic & (events["grade"] == grade) if grade is not None else m_asic
            if m.any():
                counts, _ = np.histogram(events["adu_sum"][m], bins=bin_edges)
                ax.step(centres, counts, where="mid", color=color,
                        lw=1.0, alpha=0.85, label=lbl)

        ax.set_title(f"{aname}  ({ASIC_LABEL[aname]})",
                     color=ASIC_COLORS[aname], fontsize=10, fontweight="bold")
        ax.set_xlabel("ADU"); ax.set_ylabel("Counts / bin")
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    p = out_dir / "spectrum_per_asic.png"
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
        offset_map:        np.ndarray,
        raw_frames:        np.ndarray | None,
        bin_edges:         np.ndarray,
        out_dir:           Path,
        seed_sigma:        float = 5.0,
        asic_mask:         np.ndarray | None = None,
        title_suffix:      str = "global",
) -> None:
    """
    Pixel-level ADU histograms — before event recognition.

    This shows the raw pixel-value distribution from the corrected frames,
    NOT the summed-cluster spectrum from find_events().  The two plots serve
    different purposes:

      raw_spectrum  : pixel-level view — pedestal peak near 0, single-pixel
                      photon hits appearing as a shoulder/peak at higher ADU.
                      Useful for checking the threshold and pedestal subtraction.

      spectrum_*    : event-level view — events whose cluster pixels are summed.
                      A double-pixel event with two 800 ADU pixels appears at
                      ~1600 ADU, not 800 ADU.

    Three distributions shown:
      grey   : all CM-corrected pixel values  (dominated by the pedestal peak at ~0)
      blue   : pixels above zero only         (pedestal-subtracted view)
      red    : pixels above the seed threshold (candidates entering event recognition)
    """
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _flat(frames: np.ndarray) -> np.ndarray:
        if asic_mask is not None:
            return frames[:, asic_mask].ravel()
        return frames.ravel()

    corr_flat  = _flat(corrected_frames)
    pos_flat   = corr_flat[corr_flat > 0]

    seed_thr   = seed_sigma * noise_map
    above_seed = corrected_frames > seed_thr[np.newaxis]
    hits_flat  = _flat(np.where(above_seed, corrected_frames, np.nan))
    hits_flat  = hits_flat[np.isfinite(hits_flat)]

    med_noise  = float(np.median(noise_map[asic_mask] if asic_mask is not None
                                 else noise_map))
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


def plot_spectrum_comparison(
        events:            np.ndarray,
        corrected_frames:  np.ndarray,
        noise_map:         np.ndarray,
        bin_edges:         np.ndarray,
        out_dir:           Path,
        seed_sigma:        float = 5.0,
        asic_mask:         np.ndarray | None = None,
        title_suffix:      str = "global",
) -> None:
    """
    Three-panel comparison: seed pixel / cluster sum / grade breakdown.

    Panel 1  adu_seed  – centre pixel value only.  For a double event
             with 0.8/0.2 charge split, the centre holds 0.8 × E_photon,
             so this peak sits *below* the full photon energy.

    Panel 2  adu_sum   – full cluster sum.  Singles and split events both
             reconstruct to the same peak at E_photon (if the split is
             correctly recovered).  Any residual shift indicates missing
             charge (partial collection, CTI, etc.).

    Panel 3  adu_sum   by grade group overlaid, so you can see whether
             singles, doubles, triples reconstruct to the same energy.
    """
    if len(events) == 0:
        print(f"  ⚠  No events — skipping spectrum comparison")
        return

    bin_edges = _auto_bin_edges(events, bin_edges)
    centres   = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle(
        f"Seed vs Cluster-sum Spectrum — {title_suffix}\n"
        f"(Split events: adu_seed < E_photon,  adu_sum ≈ E_photon)",
        fontsize=12, fontweight="bold")

    # ── Panel 1: seed pixel spectrum per grade group ──────────────────────────
    ax = axes[0]
    groups = [(0,[0]),(1,[1,2,3,4]),(5,[5,6,7,8]),(9,[9,10,11,12])]
    for gkey, gids in groups:
        m = np.isin(events["grade"], gids)
        if not m.any():
            continue
        c, _ = np.histogram(events["adu_seed"][m], bins=bin_edges)
        ax.step(centres, c, where="mid", color=_GRADE_PALETTE[gkey],
                lw=1.1, alpha=0.85, label=_GROUP_LABEL[gkey])
    c_all, _ = np.histogram(events["adu_seed"], bins=bin_edges)
    ax.step(centres, c_all, where="mid", color="black", lw=1.0,
            ls="--", alpha=0.7, label="all grades")
    ax.set_xlabel("ADU (centre pixel only — adu_seed)")
    ax.set_ylabel("Events / bin")
    ax.set_title("Seed Pixel Spectrum\n(centre pixel, not cluster-summed)")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.text(0.02, 0.97,
            "Doubles/triples peak below E_photon\n"
            "(centre holds only a fraction of charge)",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", alpha=0.85))

    # ── Panel 2: cluster sum spectrum per grade group ─────────────────────────
    ax2 = axes[1]
    for gkey, gids in groups:
        m = np.isin(events["grade"], gids)
        if not m.any():
            continue
        c, _ = np.histogram(events["adu_sum"][m], bins=bin_edges)
        ax2.step(centres, c, where="mid", color=_GRADE_PALETTE[gkey],
                 lw=1.1, alpha=0.85, label=_GROUP_LABEL[gkey])
    c_all2, _ = np.histogram(events["adu_sum"], bins=bin_edges)
    ax2.step(centres, c_all2, where="mid", color="black", lw=1.0,
             ls="--", alpha=0.7, label="all grades")
    ax2.set_xlabel("ADU (cluster sum — adu_sum)")
    ax2.set_ylabel("Events / bin")
    ax2.set_title("Cluster-Sum Spectrum\n(all pattern pixels summed)")
    ax2.set_yscale("log"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax2.text(0.02, 0.97,
             "All grades should reconstruct\nto same peak if charge is recovered",
             transform=ax2.transAxes, va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", fc="#d4edda", alpha=0.85))

    # ── Panel 3: seed vs sum overlay (all grades, single histogram each) ──────
    ax3 = axes[2]
    ax3.step(centres, c_all,  where="mid", color="tomato",    lw=1.4,
             alpha=0.9, label=f"adu_seed  (n={len(events):,} events, centre only)")
    ax3.step(centres, c_all2, where="mid", color="steelblue", lw=1.4,
             alpha=0.9, label=f"adu_sum   (n={len(events):,} events, cluster total)")
    ax3.set_xlabel("ADU")
    ax3.set_ylabel("Events / bin")
    ax3.set_title("Overlay: seed vs cluster sum\n"
                  "(shift = recovered split charge)")
    ax3.set_yscale("log"); ax3.grid(alpha=0.3); ax3.legend(fontsize=9)

    # Annotate the shift
    if c_all.sum() > 0 and c_all2.sum() > 0:
        peak_seed = centres[c_all.argmax()]
        peak_sum  = centres[c_all2.argmax()]
        ax3.axvline(peak_seed, color="tomato",    lw=1.0, ls=":",
                    label=f"seed peak ≈ {peak_seed:.0f}")
        ax3.axvline(peak_sum,  color="steelblue", lw=1.0, ls=":",
                    label=f"sum peak  ≈ {peak_sum:.0f}")
        ax3.legend(fontsize=8)

    plt.tight_layout()
    slug = title_suffix.replace(" ", "_")
    p = out_dir / f"spectrum_comparison_{slug}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")


def plot_raw_spectrum_per_asic(
        corrected_frames: np.ndarray,
        noise_map:        np.ndarray,
        raw_frames:       np.ndarray | None,
        bin_edges:        np.ndarray,
        asics:            list[str],
        out_dir:          Path,
        seed_sigma:       float = 5.0,
) -> None:
    """
    2×2 per-ASIC raw spectrum grid.

    Same as plot_raw_spectrum but one panel per ASIC in physical layout.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)
    fig.suptitle(f"Raw ADU Spectrum per ASIC  (seed={seed_sigma}σ, log scale)",
                 fontsize=13, fontweight="bold")
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for aname in ALL_ASICS:
        gr, gc = ASIC_GRID_POS[aname]
        ax = axes[gr, gc]
        if aname not in asics:
            ax.set_visible(False); continue

        Y0, Y1, X0, X1 = ASIC_SLICES[aname]
        corr_sub  = corrected_frames[:, Y0:Y1+1, X0:X1+1].ravel()
        noise_sub = noise_map[Y0:Y1+1, X0:X1+1]
        thr_sub   = seed_sigma * noise_sub
        above     = corrected_frames[:, Y0:Y1+1, X0:X1+1] > thr_sub[np.newaxis]
        hits_vals = corrected_frames[:, Y0:Y1+1, X0:X1+1][above].ravel()

        if raw_frames is not None:
            raw_sub = raw_frames[:, Y0:Y1+1, X0:X1+1].ravel()
            c_raw, _ = np.histogram(raw_sub, bins=bin_edges)
            ax.step(centres, c_raw, where="mid", color="gray",
                    lw=0.8, alpha=0.6, label="raw")

        c_corr, _ = np.histogram(corr_sub, bins=bin_edges)
        ax.step(centres, c_corr, where="mid", color="steelblue",
                lw=1.0, alpha=0.85, label="CM-corrected")

        c_hits, _ = np.histogram(hits_vals, bins=bin_edges)
        ax.step(centres, c_hits, where="mid",
                color=ASIC_COLORS[aname], lw=1.2, alpha=0.9,
                label=f"above {seed_sigma}σ")

        med_noise = float(np.median(noise_sub))
        ax.axvline(seed_sigma * med_noise, color=ASIC_COLORS[aname],
                   lw=1.0, ls="--",
                   label=f"seed ≈ {seed_sigma*med_noise:.0f} ADU")
        ax.axvline(0, color="k", lw=0.5, ls=":")

        ax.set_title(f"{aname}  ({ASIC_LABEL[aname]})",
                     color=ASIC_COLORS[aname], fontsize=10, fontweight="bold")
        ax.set_xlabel("ADU (CM-corrected)")
        ax.set_ylabel("Pixel count / bin")
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    plt.tight_layout()
    p = out_dir / "raw_spectrum_per_asic.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")
