"""
pnccd_ana.plotting.time_dependency_plots
=========================================
Diagnostic plots for time dependency analysis.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..physics.time_dependency import TimeDependencyResult


def plot_gain_vs_time(
        result: TimeDependencyResult,
        out_dir: Path,
        show_drift_line: bool = True,
) -> None:
    """
    Plot gain drift over time.
    
    Parameters
    ----------
    result : TimeDependencyResult
        Analysis results containing gain vs time data
    out_dir : Path
        Output directory for plots
    show_drift_line : bool
        Whether to show linear drift fit line
    """
    out_dir = Path(out_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot gain vs time with error bars
    valid_mask = result.gain_n_events_vs_time > 0
    time_centers = result.time_bin_centers[valid_mask]
    gain_mean = result.gain_vs_time[valid_mask]
    gain_std = result.gain_std_vs_time[valid_mask]
    
    ax.errorbar(time_centers, gain_mean, yerr=gain_std,
                fmt='o-', markersize=4, linewidth=1.5,
                capsize=3, capthick=1, color='steelblue',
                label='Gain')
    
    # Show drift line if requested
    if show_drift_line and len(time_centers) >= 2:
        drift_line = result.gain_vs_time[0] + result.gain_drift_rate * time_centers
        ax.plot(time_centers, drift_line, '--', color='tomato',
                linewidth=2, alpha=0.7,
                label=f'Drift: {result.gain_drift_rate:.6f} eV/ADU/frame')
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Gain [eV/ADU]')
    ax.set_title('Gain Drift Over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add stability score annotation
    ax.text(0.02, 0.98, 
            f'Stability Score: {result.gain_stability_score:.3f}\n'
            f'Drift Rate: {result.gain_drift_rate:.6f} eV/ADU/frame',
            transform=ax.transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(out_dir / "gain_vs_time.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'gain_vs_time.png'}")


def plot_event_rate_vs_time(
        result: TimeDependencyResult,
        out_dir: Path,
) -> None:
    """
    Plot event rate variation over time.
    
    Parameters
    ----------
    result : TimeDependencyResult
        Analysis results containing event rate vs time data
    out_dir : Path
        Output directory for plots
    """
    out_dir = Path(out_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_mask = result.gain_n_events_vs_time > 0
    time_centers = result.time_bin_centers[valid_mask]
    event_rate = result.event_rate_vs_time[valid_mask]
    
    ax.plot(time_centers, event_rate, 'o-', markersize=4,
            linewidth=1.5, color='seagreen', label='Event Rate')
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Events per Frame')
    ax.set_title('Event Rate Variation Over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add variation annotation
    ax.text(0.02, 0.98,
            f'Variation (CV): {result.event_rate_variation:.3f}\n'
            f'Mean Rate: {event_rate.mean():.2f} events/frame',
            transform=ax.transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(out_dir / "event_rate_vs_time.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'event_rate_vs_time.png'}")


def plot_noise_vs_time(
        result: TimeDependencyResult,
        out_dir: Path,
) -> None:
    """
    Plot noise drift over time (if available).
    
    Parameters
    ----------
    result : TimeDependencyResult
        Analysis results containing noise vs time data
    out_dir : Path
        Output directory for plots
    """
    out_dir = Path(out_dir)
    
    if result.noise_vs_time is None:
        print("  Skipping noise_vs_time plot (no noise data available)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_mask = result.noise_vs_time > 0
    time_centers = result.time_bin_centers[valid_mask]
    noise_mean = result.noise_vs_time[valid_mask]
    noise_std = result.noise_std_vs_time[valid_mask] if result.noise_std_vs_time is not None else None
    
    if noise_std is not None:
        ax.errorbar(time_centers, noise_mean, yerr=noise_std,
                    fmt='o-', markersize=4, linewidth=1.5,
                    capsize=3, capthick=1, color='orange',
                    label='Noise')
    else:
        ax.plot(time_centers, noise_mean, 'o-', markersize=4,
                linewidth=1.5, color='orange', label='Noise')
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Noise [ADU]')
    ax.set_title('Noise Drift Over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add drift annotation
    ax.text(0.02, 0.98,
            f'Drift Rate: {result.noise_drift_rate:.6f} ADU/frame',
            transform=ax.transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(out_dir / "noise_vs_time.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'noise_vs_time.png'}")


def plot_offset_vs_time(
        result: TimeDependencyResult,
        out_dir: Path,
) -> None:
    """
    Plot offset drift over time (if available).
    
    Parameters
    ----------
    result : TimeDependencyResult
        Analysis results containing offset vs time data
    out_dir : Path
        Output directory for plots
    """
    out_dir = Path(out_dir)
    
    if result.offset_vs_time is None:
        print("  Skipping offset_vs_time plot (no offset data available)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_mask = result.offset_vs_time > 0
    time_centers = result.time_bin_centers[valid_mask]
    offset_mean = result.offset_vs_time[valid_mask]
    offset_std = result.offset_std_vs_time[valid_mask] if result.offset_std_vs_time is not None else None
    
    if offset_std is not None:
        ax.errorbar(time_centers, offset_mean, yerr=offset_std,
                    fmt='o-', markersize=4, linewidth=1.5,
                    capsize=3, capthick=1, color='purple',
                    label='Offset')
    else:
        ax.plot(time_centers, offset_mean, 'o-', markersize=4,
                linewidth=1.5, color='purple', label='Offset')
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Offset [ADU]')
    ax.set_title('Offset Drift Over Time')
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / "offset_vs_time.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'offset_vs_time.png'}")


def plot_time_dependency_summary(
        result: TimeDependencyResult,
        out_dir: Path,
) -> None:
    """
    Create a multi-panel summary plot of time dependency analysis.
    
    Parameters
    ----------
    result : TimeDependencyResult
        Analysis results
    out_dir : Path
        Output directory for plots
    """
    out_dir = Path(out_dir)
    
    # Determine which panels to show
    has_noise = result.noise_vs_time is not None
    has_offset = result.offset_vs_time is not None
    
    if has_noise and has_offset:
        n_panels = 4
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    elif has_noise or has_offset:
        n_panels = 3
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        # Hide the unused panel
        axes[3].set_visible(False)
    else:
        n_panels = 2
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.flatten()
    
    # Panel 1: Gain vs time
    ax = axes[0]
    valid_mask = result.gain_n_events_vs_time > 0
    time_centers = result.time_bin_centers[valid_mask]
    gain_mean = result.gain_vs_time[valid_mask]
    gain_std = result.gain_std_vs_time[valid_mask]
    
    ax.errorbar(time_centers, gain_mean, yerr=gain_std,
                fmt='o-', markersize=3, linewidth=1.2,
                capsize=2, color='steelblue')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Gain [eV/ADU]')
    ax.set_title('Gain Drift')
    ax.grid(alpha=0.3)
    
    # Panel 2: Event rate vs time
    ax = axes[1]
    event_rate = result.event_rate_vs_time[valid_mask]
    ax.plot(time_centers, event_rate, 'o-', markersize=3,
            linewidth=1.2, color='seagreen')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Events per Frame')
    ax.set_title('Event Rate')
    ax.grid(alpha=0.3)
    
    # Panel 3: Noise vs time (if available)
    if has_noise:
        ax = axes[2]
        noise_valid = result.noise_vs_time > 0
        noise_time = result.time_bin_centers[noise_valid]
        noise_mean = result.noise_vs_time[noise_valid]
        noise_std = result.noise_std_vs_time[noise_valid] if result.noise_std_vs_time is not None else None
        
        if noise_std is not None:
            ax.errorbar(noise_time, noise_mean, yerr=noise_std,
                        fmt='o-', markersize=3, linewidth=1.2,
                        capsize=2, color='orange')
        else:
            ax.plot(noise_time, noise_mean, 'o-', markersize=3,
                    linewidth=1.2, color='orange')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Noise [ADU]')
        ax.set_title('Noise Drift')
        ax.grid(alpha=0.3)
    
    # Panel 4: Offset vs time (if available)
    if has_offset:
        ax = axes[3] if has_noise else axes[2]
        offset_valid = result.offset_vs_time > 0
        offset_time = result.time_bin_centers[offset_valid]
        offset_mean = result.offset_vs_time[offset_valid]
        offset_std = result.offset_std_vs_time[offset_valid] if result.offset_std_vs_time is not None else None
        
        if offset_std is not None:
            ax.errorbar(offset_time, offset_mean, yerr=offset_std,
                        fmt='o-', markersize=3, linewidth=1.2,
                        capsize=2, color='purple')
        else:
            ax.plot(offset_time, offset_mean, 'o-', markersize=3,
                    linewidth=1.2, color='purple')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Offset [ADU]')
        ax.set_title('Offset Drift')
        ax.grid(alpha=0.3)
    
    # Add overall title with quality flags
    title = 'Time Dependency Analysis Summary'
    if result.has_significant_drift:
        title += ' [⚠ SIGNIFICANT DRIFT]'
    if result.has_instability:
        title += ' [⚠ INSTABILITY]'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    fig.savefig(out_dir / "time_dependency_summary.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'time_dependency_summary.png'}")


def plot_all_time_dependency(
        result: TimeDependencyResult,
        out_dir: Path,
) -> None:
    """
    Generate all time dependency plots.
    
    Parameters
    ----------
    result : TimeDependencyResult
        Analysis results
    out_dir : Path
        Output directory for plots
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating time dependency plots...")
    plot_gain_vs_time(result, out_dir)
    plot_event_rate_vs_time(result, out_dir)
    plot_noise_vs_time(result, out_dir)
    plot_offset_vs_time(result, out_dir)
    plot_time_dependency_summary(result, out_dir)
    print("  ✓ All time dependency plots complete")