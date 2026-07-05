"""
pnccd_ana.cli.time_dependency
==============================
Command-line entry point for time dependency analysis.

Usage
-----
  python -m pnccd_ana.cli.time_dependency analysis.yaml
  pnccd-time-dependency analysis.yaml

  To generate a template config:
  pnccd-template analysis.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import Config
from ..physics.time_dependency import (
    analyze_time_dependency,
    TimeDependencyResult,
)
from ..plotting.time_dependency_plots import plot_all_time_dependency


def run(cfg: Config) -> dict:
    """
    Execute the time dependency analysis pipeline from a Config object.

    Returns a results dict (useful when called programmatically).
    """
    td = cfg.time_dependency if hasattr(cfg, 'time_dependency') else {}
    gen = cfg.general
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get events file path
    events_file = td.get("events_file")
    if not events_file:
        # Try to use default events path
        events_file = cfg.events_path()
        if not events_file.exists():
            raise ValueError(
                "[time_dependency] events_file must be set in config, or "
                f"{events_file} must exist from event_rec stage.\n"
                "  Run  pnccd-event-rec  first to generate events.h5")
    
    events_path = cfg.resolve_input_path(events_file)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    # Get gain map file path
    gain_file = td.get("gain_file")
    if not gain_file:
        # Try to use default energy calibration path
        gain_file = out_dir / "energy_cal.h5"
        if not gain_file.exists():
            raise ValueError(
                "[time_dependency] gain_file must be set in config, or "
                f"{gain_file} must exist from energy_cal stage.\n"
                "  Run  pnccd-energy-cal  first to generate energy_cal.h5")
    
    gain_path = cfg.resolve_input_path(gain_file)
    if not gain_path.exists():
        raise FileNotFoundError(f"Gain file not found: {gain_path}")

    # Analysis parameters
    n_bins = int(td.get("n_bins", 50))
    min_events_per_bin = int(td.get("min_events_per_bin", 100))
    drift_threshold = float(td.get("drift_threshold", 0.01))
    skip_frames = int(td.get("skip_frames", 0))
    max_frames = td.get("max_frames", None)
    if max_frames is not None:
        max_frames = int(max_frames)

    print(f"\nTime dependency analysis")
    print(f"  Events file: {events_path}")
    print(f"  Gain file: {gain_path}")
    print(f"  Time bins: {n_bins}")
    print(f"  Min events per bin: {min_events_per_bin}")
    print(f"  Drift threshold: {drift_threshold:.1%}")

    # Load events data
    import h5py
    print(f"\n  Loading events from: {events_path}")
    with h5py.File(events_path, "r") as f:
        # Load event data
        adu_sum = f["events"]["adu_sum"][:]
        col = f["events"]["col"][:]
        row = f["events"]["row"][:]
        
        # Try to load frame indices if available
        if "frame_idx" in f["events"]:
            frame_indices = f["events"]["frame_idx"][:]
        else:
            print("  Warning: frame_idx not found in events file, "
                  "assuming uniform distribution")
            frame_indices = None
        
        # Get total number of frames
        if "meta" in f and "n_frames" in f["meta"]:
            n_frames = int(f["meta"]["n_frames"][()])
        else:
            # Estimate from frame indices
            if frame_indices is not None:
                n_frames = int(frame_indices.max()) + 1
            else:
                raise ValueError("Cannot determine n_frames: "
                               "frame_idx not in events file")
        
        # Apply frame selection
        if skip_frames > 0 or max_frames is not None:
            if frame_indices is not None:
                mask = frame_indices >= skip_frames
                if max_frames is not None:
                    mask &= frame_indices < (skip_frames + max_frames)
                adu_sum = adu_sum[mask]
                col = col[mask]
                row = row[mask]
                frame_indices = frame_indices[mask]
                n_frames = min(n_frames, skip_frames + max_frames) if max_frames else n_frames
                print(f"  Frame selection: skip={skip_frames} max={max_frames}")
                print(f"  Events after selection: {len(adu_sum)}")
            else:
                print("  Warning: Cannot apply frame selection without frame_idx")

    print(f"  Total events: {len(adu_sum)}")
    print(f"  Total frames: {n_frames}")

    # Load gain map
    print(f"\n  Loading gain map from: {gain_path}")
    with h5py.File(gain_path, "r") as f:
        if "gain_map" in f:
            gain_map = f["gain_map"][:]
        elif "calibration" in f and "gain_map" in f["calibration"]:
            gain_map = f["calibration"]["gain_map"][:]
        else:
            raise ValueError(f"Cannot find gain_map in {gain_path}")
    
    print(f"  Gain map shape: {gain_map.shape}")

    # Create FilteredEvents-like object
    from dataclasses import dataclass
    
    @dataclass
    class FilteredEvents:
        adu_sum: np.ndarray
        col: np.ndarray
        row: np.ndarray
    
    filtered = FilteredEvents(adu_sum=adu_sum, col=col, row=row)

    # Run time dependency analysis
    print(f"\n  Running time dependency analysis...")
    result = analyze_time_dependency(
        filtered=filtered,
        gain_map=gain_map,
        n_frames=n_frames,
        frame_indices=frame_indices,
        n_bins=n_bins,
        min_events_per_bin=min_events_per_bin,
        drift_threshold=drift_threshold,
    )

    # Print summary
    print(f"\n  Results:")
    print(f"    Gain drift rate: {result.gain_drift_rate:.6e} eV/ADU/frame")
    print(f"    Gain stability score: {result.gain_stability_score:.3f}")
    print(f"    Event rate variation: {result.event_rate_variation:.3f}")
    print(f"    Significant drift: {result.has_significant_drift}")
    print(f"    Instability detected: {result.has_instability}")

    # Save results
    save_results = td.get("save_results", True)
    if save_results:
        results_file = out_dir / "time_dependency.h5"
        print(f"\n  Saving results to: {results_file}")
        with h5py.File(results_file, "w") as f:
            # Save time bin information
            f.create_dataset("time_bin_edges", data=result.time_bin_edges)
            f.create_dataset("time_bin_centers", data=result.time_bin_centers)
            
            # Save gain vs time
            f.create_dataset("gain_vs_time", data=result.gain_vs_time)
            f.create_dataset("gain_std_vs_time", data=result.gain_std_vs_time)
            f.create_dataset("gain_n_events_vs_time", data=result.gain_n_events_vs_time)
            
            # Save event rate
            f.create_dataset("event_rate_vs_time", data=result.event_rate_vs_time)
            
            # Save noise vs time if available
            if result.noise_vs_time is not None:
                f.create_dataset("noise_vs_time", data=result.noise_vs_time)
            if result.noise_std_vs_time is not None:
                f.create_dataset("noise_std_vs_time", data=result.noise_std_vs_time)
            
            # Save offset vs time if available
            if result.offset_vs_time is not None:
                f.create_dataset("offset_vs_time", data=result.offset_vs_time)
            if result.offset_std_vs_time is not None:
                f.create_dataset("offset_std_vs_time", data=result.offset_std_vs_time)
            
            # Save quality metrics as attributes
            grp = f.create_group("metrics")
            grp.attrs["gain_drift_rate"] = result.gain_drift_rate
            grp.attrs["gain_stability_score"] = result.gain_stability_score
            grp.attrs["noise_drift_rate"] = result.noise_drift_rate
            grp.attrs["event_rate_variation"] = result.event_rate_variation
            grp.attrs["has_significant_drift"] = result.has_significant_drift
            grp.attrs["has_instability"] = result.has_instability
            
            # Save analysis parameters
            grp.attrs["n_bins"] = n_bins
            grp.attrs["min_events_per_bin"] = min_events_per_bin
            grp.attrs["drift_threshold"] = drift_threshold
            grp.attrs["n_frames"] = n_frames
            grp.attrs["n_events"] = len(adu_sum)
        
        print(f"  ✓ Results saved")

    # Generate plots
    save_plots = td.get("save_plots", True)
    if save_plots:
        plot_dir = out_dir / "time_dependency"
        plot_all_time_dependency(result, plot_dir)

    print(f"\n✓ Time dependency analysis complete.  Output: {out_dir}/")
    
    return {
        "result": result,
        "config": {
            "n_bins": n_bins,
            "min_events_per_bin": min_events_per_bin,
            "drift_threshold": drift_threshold,
            "n_frames": n_frames,
            "n_events": len(adu_sum),
        }
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="pnCCD time dependency analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("config", help="Path to analysis.yaml  "
                        "(run  pnccd-template  to generate one)")
    args = parser.parse_args(argv)
    cfg = Config.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()