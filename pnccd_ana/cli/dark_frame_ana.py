"""
pnccd_ana.cli.dark_frame_ana
=============================
Command-line entry point for dark-frame calibration.

Usage
-----
  python -m pnccd_ana.cli.dark_frame_ana analysis.yaml
  python -m pnccd_ana.cli.dark_frame_ana analysis.yaml --section dark_frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..config import Config
from ..lib    import (compute_offset_median,
                       compute_offset_sigma_clip,
                       apply_common_mode_correction,
                       compute_noise,
                       compute_cm_noise,
                       resolve_asics, split_asics,
                       ASIC_SLICES, ASIC_MASK, build_bad_pixel_mask,
                       configure_asics, get_active_mask)
from ..utils  import (save_calibration_h5, save_calibration_npy,
                       plot_offsets, plot_noise, plot_cm_map, plot_bad_pixels)


# ──────────────────────────────────────────────────────────────────────────────
# Per-scope analysis (single-hybrid)
# ──────────────────────────────────────────────────────────────────────────────

def _analyse_scope(data:        np.ndarray,
                   label:       str,
                   methods:     list[str],
                   n_sigma:     float,
                   out_dir:     Path,
                   asic_slices: dict[str, tuple[int, int, int, int]] | None = None,
                   build_bp_mask: bool = True,
                   active_mask: np.ndarray | None = None) -> dict:
    """
    Run pedestal → CM → noise pipeline for one data block.

    Parameters
    ----------
    data       : (N, H, W) float32 array — frames from RAW file
    label      : scope name used in prints and filenames
    methods    : ['median'], ['sigclip'], or both
    n_sigma    : sigma-clip threshold
    out_dir    : output directory
    asic_slices: dict mapping ASIC name to (Y0, Y1, X0, X1) for per-ASIC CM
    build_bp_mask: whether to build bad pixel mask and plot it
    active_mask: bool (H, W) — True where pixel is active (non-masked)

    Returns dict with keys: offset_median?, offset_sigclip?,
                             noise, cm_noise, n_clipped_map?,
                             cm_map, data_corrected, bad_pixel_mask?
    """
    print(f"\n── {label} ──")
    r: dict       = {}
    keep_mask     = None
    n_clipped_map = None

    if "median" in methods:
        r["offset_median"] = compute_offset_median(data, label)
    if "sigclip" in methods:
        r["offset_sigclip"], keep_mask, n_clipped_map = \
            compute_offset_sigma_clip(data, n_sigma=n_sigma, label=label)
        r["n_clipped_map"] = n_clipped_map

    ref_offset = r.get("offset_sigclip", r.get("offset_median"))
    corrected, cm_map, asic_names = apply_common_mode_correction(
        data, ref_offset, label, asic_slices=asic_slices)
    r["noise"]          = compute_noise(corrected, keep_mask, label)
    
    # Handle both dict and array cm_noise
    cm_noise = compute_cm_noise(cm_map, label, asic_names=asic_names)
    if isinstance(cm_noise, dict):
        r["cm_noise"] = cm_noise.get(list(cm_noise.keys())[0] if cm_noise else "global")
    else:
        r["cm_noise"] = cm_noise
    
    r["cm_map"]         = cm_map
    r["asic_names"]     = asic_names
    r["data_corrected"] = corrected

    # Build bad pixel mask if requested
    bad_pixel_mask = None
    if build_bp_mask:
        bad_pixel_mask = build_bad_pixel_mask(
            r["noise"],
            n_clipped_map=n_clipped_map,
            n_dark_frames=data.shape[0] if n_clipped_map is not None else 0,
            active_mask=active_mask,
            label=label,
        )
        r["bad_pixel_mask"] = bad_pixel_mask

    scope_dir = out_dir / label
    scope_dir.mkdir(parents=True, exist_ok=True)
    plot_offsets(label, r, scope_dir)
    plot_noise(label, r, scope_dir)
    plot_cm_map(label, cm_map, scope_dir, asic_names=asic_names)
    
    if build_bp_mask and bad_pixel_mask is not None:
        plot_bad_pixels(label, r["noise"], bad_pixel_mask, scope_dir, active_mask=active_mask)

    return r


# ──────────────────────────────────────────────────────────────────────────────
# Multi-file path resolution
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_paths(file_spec) -> list[Path]:
    """
    Resolve a file specification to a sorted list of concrete paths.

    Accepts:
      - a single path string, optionally containing shell glob wildcards
        (e.g. "data/dark_run00*_H1.raw")
      - a list of path strings (each may itself contain globs)

    Raises FileNotFoundError if no files match.
    """
    from pathlib import Path
    from glob import glob

    if isinstance(file_spec, (str, Path)):
        specs = [str(file_spec)]
    else:
        specs = [str(s) for s in file_spec]
    paths: list[Path] = []
    for spec in specs:
        matched = sorted(glob(str(spec)))
        if not matched:
            # Try as a literal path (no wildcard) — gives a clear error
            p = Path(spec)
            if not p.exists():
                raise FileNotFoundError(
                    f"No files found matching: {spec!r}")
            matched = [str(p)]
        paths.extend(Path(m) for m in matched)

    if not paths:
        raise FileNotFoundError(f"No files found for spec: {file_spec!r}")
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: Config) -> dict:
    """
    Execute the dark-frame calibration pipeline from a Config object.

    Returns a results dict (useful when called programmatically).
    """
    dc       = cfg.dark_frames
    gen      = cfg.general
    out_dir  = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)


    dark_run_file = dc["dark_run_file"]
    if not dark_run_file:
        raise ValueError("[dark_frames] dark_run_file must be set in config.")

    # Resolve input path relative to data_dir
    dark_run_path = cfg.resolve_input_path(dark_run_file)

    methods  = cfg.methods_for_dark()
    n_sigma  = float(dc["sigma_clip_nsigma"])
    asics    = cfg.asics_for("dark_frames")

    # ── Resolve file list (single path, glob, or list) ────────────────────────
    run_files = _resolve_paths(dark_run_path)
    print(f"\nDark run file(s): {len(run_files)} file(s) matched")
    for p in run_files:
        print(f"  {p}")

    # ── Load all dark frames (dark frames fit in memory for typical runs) ──────
    # Always uses RAW format (512x512 or 1024x512 based on frame_rows config)
    from ..utils.io_raw import get_io_module as raw_get_io_module
    io = raw_get_io_module(gen.get("data_format", "raw"))

    raw_kwargs = {}
    if gen.get("frame_rows"):
        raw_kwargs["height"] = gen["frame_rows"]
    if gen.get("frame_cols"):
        raw_kwargs["width"] = gen["frame_cols"]

    def _load_chunk(raw: np.ndarray, _idx: np.ndarray) -> np.ndarray:
        return raw   # pass raw frames through unchanged

    all_chunks: list[np.ndarray] = []
    remaining = gen["max_frames"]   # None = unlimited; decremented per file

    for fpath in run_files:
        if remaining is not None and remaining <= 0:
            break
        print(f"\nLoading dark frames from: {fpath}")
        indices = io.get_frame_indices(fpath,
                                       complete_only=gen["complete_only"],
                                       max_frames=remaining,
                                       **raw_kwargs)
        chunks = io.process_frames_mt(fpath, indices, _load_chunk,
                                      chunk_size=gen["chunk_size"],
                                      n_workers=gen["n_workers"],
                                      desc="dark frames",
                                      **raw_kwargs)
        all_chunks.extend(chunks)
        if remaining is not None:
            remaining -= len(indices)

    data_raw = np.concatenate(all_chunks, axis=0)
    print(f"\n  Total loaded: {data_raw.shape[0]} frames  "
          f"({data_raw.shape[1]} × {data_raw.shape[2]} pixels)"
          f"  from {len(run_files)} file(s)")

    # Configure ASIC geometry from config
    n_asics = int(gen.get("ASIC_num", 8))
    asic_mask = gen.get("ASIC_mask", [])
    configure_asics(n_asics, data_raw.shape[2], mask=asic_mask)
    
    if asic_mask:
        print(f"  ASIC mask applied: excluding ASICs {asic_mask}")
    print(f"  ASIC configuration: {n_asics} ASICs × {ASIC_WIDTH} columns = {data_raw.shape[2]} total columns")

    # Use per-ASIC CM correction
    asic_slices = ASIC_SLICES if len(ASIC_SLICES) > 1 else None

    # ── Full-frame analysis ───────────────────────────────────────────────────
    all_results: dict = {"asics": {}}
    
    # Create active mask for bad pixel detection
    active_mask = get_active_mask(data_raw.shape[1], data_raw.shape[2], n_asics, asic_mask)
    
    all_results["global"] = _analyse_scope(
        data_raw, "global", methods, n_sigma, out_dir,
        asic_slices=asic_slices, build_bp_mask=True, active_mask=active_mask)

    # ── Per-ASIC analysis ─────────────────────────────────────────────────────
    if asics:
        asic_data = split_asics(data_raw, asics)
        for aname in asics:
            # Per-ASIC data uses full width (single ASIC slice), so no active_mask needed
            all_results["asics"][aname] = _analyse_scope(
                asic_data[aname], aname, methods, n_sigma, out_dir,
                asic_slices=None, build_bp_mask=False)

        # ── Save results ──────────────────────────────────────────────────────────
    def _strip(r: dict) -> dict:
        return {k: v for k, v in r.items()
                if k not in ("cm_map", "data_corrected", "keep_mask")}

    save_payload = {
        "global": _strip(all_results["global"]),
        "asics":  {k: _strip(v) for k, v in all_results["asics"].items()},
    }

    if dc["save_h5"]:
        save_calibration_h5(
            out_dir / "dark_calibration.h5",
            save_payload,
            n_sigma=n_sigma,
            n_frames=len(indices),
            methods=methods,
            metadata=gen.get("metadata", {}),
        )
    if dc["save_npy"]:
        save_calibration_npy(out_dir, save_payload)

    print(f"\n✓ Dark-frame calibration complete.  Output: {out_dir}/")
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="pnCCD dark-frame calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("config", help="Path to analysis.yaml")
    parser.add_argument("--template", action="store_true",
                        help="Write a template config and exit")
    args = parser.parse_args(argv)

    if args.template:
        from ..config import write_template
        write_template(args.config)
        return

    cfg = Config.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
