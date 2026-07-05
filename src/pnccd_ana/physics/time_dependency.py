"""
pnccd_ana.physics.time_dependency
==================================

Time dependency analysis for pnCCD calibration.

Matches ROOT HStepTimeDependency:
  - Track gain, noise, offset drift over time
  - Analyze event rate variations
  - Identify temporal trends and instabilities

Functions
---------
  analyze_time_dependency : Main analysis function
  compute_gain_vs_time    : Gain drift analysis
  compute_noise_vs_time   : Noise drift analysis
  compute_event_rate_vs_time : Event rate analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .calibrate import FilteredEvents


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimeDependencyResult:
    """Results of time dependency analysis."""
    # Time bins
    time_bin_edges: np.ndarray      # (n_bins + 1,) frame indices
    time_bin_centers: np.ndarray    # (n_bins,) frame indices
    
    # Gain vs time
    gain_vs_time: np.ndarray        # (n_bins,) mean gain per bin
    gain_std_vs_time: np.ndarray    # (n_bins,) std gain per bin
    gain_n_events_vs_time: np.ndarray  # (n_bins,) events per bin
    
    # Noise vs time (if frames available)
    noise_vs_time: np.ndarray | None = None
    noise_std_vs_time: np.ndarray | None = None
    
    # Offset vs time (if frames available)
    offset_vs_time: np.ndarray | None = None
    offset_std_vs_time: np.ndarray | None = None
    
    # Event rate
    event_rate_vs_time: np.ndarray   # (n_bins,) events per frame
    
    # Quality metrics
    gain_drift_rate: float = 0.0     # ADU per frame
    gain_stability_score: float = 1.0  # 1.0 = stable, <1.0 = drifting
    noise_drift_rate: float = 0.0    # ADU per frame
    event_rate_variation: float = 0.0  # Coefficient of variation
    
    # Flags
    has_significant_drift: bool = False
    has_instability: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis function
# ══════════════════════════════════════════════════════════════════════════════

def analyze_time_dependency(
        filtered: "FilteredEvents",
        gain_map: np.ndarray,
        n_frames: int,
        frame_indices: np.ndarray | None = None,
        n_bins: int = 50,
        min_events_per_bin: int = 100,
        drift_threshold: float = 0.01,  # 1% drift considered significant
) -> TimeDependencyResult:
    """
    Analyze temporal drift of calibration parameters.
    
    Parameters
    ----------
    filtered      : FilteredEvents object with event data
    gain_map      : (n_rows, n_cols) gain map [eV/ADU]
    n_frames      : total number of frames in dataset
    frame_indices : (n_events,) frame index for each event, or None
    n_bins        : number of time bins
    min_events_per_bin : minimum events required per bin
    drift_threshold : relative gain change considered significant
    
    Returns
    -------
    TimeDependencyResult
    """
    n_events = len(filtered.adu_sum)
    
    # Generate frame indices if not provided
    if frame_indices is None:
        # Assume uniform distribution across frames
        frame_indices = np.linspace(0, n_frames - 1, n_events, dtype=np.int32)
    
    # Create time bins
    time_bin_edges = np.linspace(0, n_frames, n_bins + 1, dtype=np.float64)
    time_bin_centers = 0.5 * (time_bin_edges[:-1] + time_bin_edges[1:])
    
    # Assign events to time bins
    event_bin_indices = np.digitize(frame_indices, time_bin_edges) - 1
    event_bin_indices = np.clip(event_bin_indices, 0, n_bins - 1)
    
    # Compute gain vs time
    gain_vs_time, gain_std_vs_time, gain_n_events_vs_time = \
        compute_gain_vs_time(
            filtered.adu_sum,
            filtered.col,
            filtered.row,
            gain_map,
            event_bin_indices,
            n_bins,
        )
    
    # Compute event rate vs time
    event_rate_vs_time = compute_event_rate_vs_time(
        event_bin_indices,
        n_bins,
        n_frames,
    )
    
    # Compute drift metrics
    valid_bins = gain_n_events_vs_time >= min_events_per_bin
    if valid_bins.sum() >= 2:
        # Linear fit to gain vs time
        valid_centers = time_bin_centers[valid_bins]
        valid_gains = gain_vs_time[valid_bins]
        
        # Simple linear regression
        gain_drift_rate = np.polyfit(valid_centers, valid_gains, 1)[0]
        
        # Stability score: 1.0 if no drift, decreases with drift
        gain_range = valid_gains.max() - valid_gains.min()
        gain_mean = valid_gains.mean()
        relative_drift = gain_range / gain_mean if gain_mean > 0 else 0
        gain_stability_score = max(0.0, 1.0 - relative_drift / drift_threshold)
        
        # Event rate variation
        valid_rates = event_rate_vs_time[valid_bins]
        event_rate_variation = valid_rates.std() / valid_rates.mean() if valid_rates.mean() > 0 else 0
    else:
        gain_drift_rate = 0.0
        gain_stability_score = 1.0
        event_rate_variation = 0.0
    
    # Determine flags
    has_significant_drift = abs(relative_drift) > drift_threshold if 'relative_drift' in locals() else False
    has_instability = event_rate_variation > 0.5  # 50% variation considered unstable
    
    return TimeDependencyResult(
        time_bin_edges=time_bin_edges,
        time_bin_centers=time_bin_centers,
        gain_vs_time=gain_vs_time,
        gain_std_vs_time=gain_std_vs_time,
        gain_n_events_vs_time=gain_n_events_vs_time,
        noise_vs_time=None,
        noise_std_vs_time=None,
        offset_vs_time=None,
        offset_std_vs_time=None,
        event_rate_vs_time=event_rate_vs_time,
        gain_drift_rate=gain_drift_rate,
        gain_stability_score=gain_stability_score,
        noise_drift_rate=0.0,
        event_rate_variation=event_rate_variation,
        has_significant_drift=has_significant_drift,
        has_instability=has_instability,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Gain vs time analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_gain_vs_time(
        adu_sum: np.ndarray,
        col: np.ndarray,
        row: np.ndarray,
        gain_map: np.ndarray,
        bin_indices: np.ndarray,
        n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean gain and calibrated energy vs time.
    
    Parameters
    ----------
    adu_sum    : (n_events,) ADU sum per event
    col        : (n_events,) column index per event
    row        : (n_events,) row index per event
    gain_map   : (n_rows, n_cols) gain map [eV/ADU]
    bin_indices: (n_events,) time bin index for each event
    n_bins     : number of time bins
    
    Returns
    -------
    gain_vs_time : (n_bins,) mean gain per bin [eV/ADU]
    gain_std_vs_time : (n_bins,) std gain per bin
    n_events_vs_time : (n_bins,) number of events per bin
    """
    n_rows, n_cols = gain_map.shape
    
    # Get gain for each event
    event_gain = gain_map[row.astype(np.int32), col.astype(np.int32)]
    
    # Compute calibrated energy
    energy_ev = adu_sum * event_gain
    
    # Compute statistics per time bin
    gain_vs_time = np.zeros(n_bins, dtype=np.float64)
    gain_std_vs_time = np.zeros(n_bins, dtype=np.float64)
    n_events_vs_time = np.zeros(n_bins, dtype=np.int64)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            gains = event_gain[mask]
            gain_vs_time[i] = gains.mean()
            gain_std_vs_time[i] = gains.std()
            n_events_vs_time[i] = mask.sum()
    
    return gain_vs_time, gain_std_vs_time, n_events_vs_time


# ══════════════════════════════════════════════════════════════════════════════
# Event rate vs time analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_event_rate_vs_time(
        bin_indices: np.ndarray,
        n_bins: int,
        n_frames: int,
) -> np.ndarray:
    """
    Compute event rate (events per frame) vs time.
    
    Parameters
    ----------
    bin_indices : (n_events,) time bin index for each event
    n_bins      : number of time bins
    n_frames    : total number of frames
    
    Returns
    -------
    event_rate_vs_time : (n_bins,) events per frame per bin
    """
    n_events_per_bin = np.bincount(bin_indices, minlength=n_bins)
    frames_per_bin = n_frames / n_bins
    event_rate_vs_time = n_events_per_bin / frames_per_bin
    
    return event_rate_vs_time


# ══════════════════════════════════════════════════════════════════════════════
# Noise vs time analysis (requires frame data)
# ══════════════════════════════════════════════════════════════════════════════

def compute_noise_vs_time(
        corrected_frames: np.ndarray,
        frame_indices: np.ndarray,
        n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute noise vs time from corrected frames.
    
    Parameters
    ----------
    corrected_frames : (n_frames, n_rows, n_cols) corrected frames
    frame_indices    : (n_frames,) frame index for each frame
    n_bins           : number of time bins
    
    Returns
    -------
    noise_vs_time : (n_bins,) mean noise per bin
    noise_std_vs_time : (n_bins,) std noise per bin
    """
    n_frames = corrected_frames.shape[0]
    
    # Create time bins
    time_bin_edges = np.linspace(0, n_frames, n_bins + 1, dtype=np.float64)
    
    # Assign frames to time bins
    frame_bin_indices = np.digitize(frame_indices, time_bin_edges) - 1
    frame_bin_indices = np.clip(frame_bin_indices, 0, n_bins - 1)
    
    # Compute noise per frame (std dev)
    noise_per_frame = np.std(corrected_frames, axis=(1, 2))
    
    # Compute statistics per time bin
    noise_vs_time = np.zeros(n_bins, dtype=np.float64)
    noise_std_vs_time = np.zeros(n_bins, dtype=np.float64)
    
    for i in range(n_bins):
        mask = frame_bin_indices == i
        if mask.sum() > 0:
            noises = noise_per_frame[mask]
            noise_vs_time[i] = noises.mean()
            noise_std_vs_time[i] = noises.std()
    
    return noise_vs_time, noise_std_vs_time


# ══════════════════════════════════════════════════════════════════════════════
# Offset vs time analysis (requires frame data)
# ══════════════════════════════════════════════════════════════════════════════

def compute_offset_vs_time(
        corrected_frames: np.ndarray,
        frame_indices: np.ndarray,
        n_bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean offset vs time from corrected frames.
    
    Parameters
    ----------
    corrected_frames : (n_frames, n_rows, n_cols) corrected frames
    frame_indices    : (n_frames,) frame index for each frame
    n_bins           : number of time bins
    
    Returns
    -------
    offset_vs_time : (n_bins,) mean offset per bin
    offset_std_vs_time : (n_bins,) std offset per bin
    """
    n_frames = corrected_frames.shape[0]
    
    # Create time bins
    time_bin_edges = np.linspace(0, n_frames, n_bins + 1, dtype=np.float64)
    
    # Assign frames to time bins
    frame_bin_indices = np.digitize(frame_indices, time_bin_edges) - 1
    frame_bin_indices = np.clip(frame_bin_indices, 0, n_bins - 1)
    
    # Compute mean offset per frame
    offset_per_frame = np.mean(corrected_frames, axis=(1, 2))
    
    # Compute statistics per time bin
    offset_vs_time = np.zeros(n_bins, dtype=np.float64)
    offset_std_vs_time = np.zeros(n_bins, dtype=np.float64)
    
    for i in range(n_bins):
        mask = frame_bin_indices == i
        if mask.sum() > 0:
            offsets = offset_per_frame[mask]
            offset_vs_time[i] = offsets.mean()
            offset_std_vs_time[i] = offsets.std()
    
    return offset_vs_time, offset_std_vs_time