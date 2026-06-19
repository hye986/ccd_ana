"""
pnccd_ana.cli.offset
=====================
Command-line entry point for offset calibration.

Usage
-----
  python -m pnccd_ana.cli.offset analysis.yaml
  python -m pnccd_ana.cli.offset analysis.yaml --section offset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from ..config import Config
from ..lib    import (compute_offset_median,
                       compute_offset_sigma_clip,
                       apply_common_mode_correction,
                       compute_noise,
                       compute_cm_noise,
                       ASIC_SLICES, ASIC_MASK, ASIC_WIDTH,
                       build_bad_pixel_mask,
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
    
    # Store full per-ASIC dict or 1-D array — both handled by save_calibration_h5
    r["cm_noise"] = compute_cm_noise(cm_map, label, asic_names=asic_names)

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
    plot_offsets(label, r, scope_dir, active_mask=active_mask)
    plot_noise(label, r, scope_dir, active_mask=active_mask)
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
    Execute the offset calibration pipeline from a Config object.

    Returns a results dict (useful when called programmatically).
    """
    oc       = cfg.offset
    gen      = cfg.general
    out_dir  = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)


    dark_run_file = oc["dark_run_file"]
    if not dark_run_file:
        raise ValueError("[offset] dark_run_file must be set in config.")

    # Resolve input path relative to data_dir
    dark_run_path = cfg.resolve_input_path(dark_run_file)

    methods  = cfg.methods_for_offset()
    n_sigma  = float(oc["sigma_clip_nsigma"])

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
    remaining = gen["max_frames"]
    total_frames_loaded = 0

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
        total_frames_loaded += len(indices)

    data_raw = np.concatenate(all_chunks, axis=0)
    print(f"\n  Total loaded: {data_raw.shape[0]} frames  "
          f"({data_raw.shape[1]} × {data_raw.shape[2]} pixels)"
          f"  from {len(run_files)} file(s)")

    # Configure ASIC geometry from config
    n_asics = int(gen.get("ASIC_num", 8))
    asic_mask = gen.get("ASIC_mask", [])
    configure_asics(n_asics, data_raw.shape[2], data_raw.shape[1], mask=asic_mask)
    
    if asic_mask:
        print(f"  ASIC mask applied: excluding ASICs {asic_mask}")
    print(f"  ASIC configuration: {n_asics} ASICs × {ASIC_WIDTH} columns = {data_raw.shape[2]} total columns")

    # Per-ASIC CM correction - include ALL ASICs (even masked ones)
    # CM is computed per ASIC using 64 columns each
    asic_slices = ASIC_SLICES if len(ASIC_SLICES) > 1 else None

    # ── Full-frame analysis (with per-ASIC CM correction) ─────────────────────
    all_results: dict = {}
    
    # Create active mask for bad pixel detection and plotting
    active_mask = get_active_mask(data_raw.shape[1], data_raw.shape[2], n_asics, asic_mask)
    
    all_results["global"] = _analyse_scope(
        data_raw, "global", methods, n_sigma, out_dir,
        asic_slices=asic_slices, build_bp_mask=True, active_mask=active_mask)

    # ── Save results ──────────────────────────────────────────────────────────
    def _strip(r: dict) -> dict:
        return {k: v for k, v in r.items()
                if k not in ("cm_map", "data_corrected", "keep_mask")}

    save_payload = {"global": _strip(all_results["global"])}

    if oc["save_h5"]:
        save_calibration_h5(
            out_dir / "offset.h5",
            save_payload,
            n_sigma=n_sigma,
            n_frames=total_frames_loaded,
            methods=methods,
            metadata=gen.get("metadata", {}),
        )
    if oc["save_npy"]:
        save_calibration_npy(out_dir, save_payload)

    # Save plot-backing data
    save_offset_results_h5(out_dir, all_results, active_mask, gen, n_sigma,
                           total_frames_loaded, methods, n_asics)

    print(f"\n✓ Offset calibration complete.  Output: {out_dir}/")
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Save plot-backing data to HDF5
# ──────────────────────────────────────────────────────────────────────────────

def save_offset_results_h5(
        out_dir:    Path,
        results:    dict,
        active_mask: np.ndarray,
        gen:       dict,
        n_sigma:   float,
        n_frames:  int,
        methods:   list[str],
        n_asics:   int,
) -> None:
    """
    Save plot-backing data to offset_results.h5.

    HDF5 structure:
        /offsets/median/{map, hist_edges, hist_counts}
        /offsets/sigmaclip/{map, hist_edges, hist_counts}
        /offsets/diff/{map, hist_edges, hist_counts}  (if both methods run)
        /noise/{map, hist_edges, hist_counts}
        /noise/cm_noise/{asic_name}  (one per ASIC)
        /noise/n_clipped/{map}
        /bad_pixels/{mask, category_map, noise_hist_edges, noise_hist_counts,
                     thresholds/{median_noise, hot_threshold, cold_threshold},
                     per_asic/{n_bad, asic_labels}}
        /meta/...
    """
    path = out_dir / "offset_results.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving offset results: {path}")

    r = results.get("global", {})

    def _compute_hist(arr: np.ndarray, n_bins: int = 200) -> tuple:
        """Compute histogram from array, excluding masked regions."""
        flat = arr.ravel()
        if np.any(np.isnan(flat)):
            flat = flat[~np.isnan(flat)]
        lo, hi = np.percentile(flat, [0.5, 99.5]) if len(flat) > 0 else (0, 1)
        counts, edges = np.histogram(flat, bins=n_bins, range=(lo, hi))
        return edges.astype(np.float32), counts.astype(np.int64)

    with h5py.File(path, "w") as f:
        # ── Offsets ────────────────────────────────────────────────────────────
        og = f.require_group("offsets")
        for method in methods:
            key = f"offset_{method}"
            if key not in r:
                continue
            arr = r[key].astype(np.float32)
            mg = og.require_group(method)
            mg.create_dataset("map", data=arr, compression="gzip")

            # Apply mask for histogram
            masked = np.where(active_mask, arr, np.nan)
            edges, counts = _compute_hist(masked)
            mg.create_dataset("hist_edges",  data=edges)
            mg.create_dataset("hist_counts", data=counts)

        # Difference map if both methods exist
        if "offset_median" in r and "offset_sigclip" in r:
            diff = (r["offset_median"] - r["offset_sigclip"]).astype(np.float32)
            masked_diff = np.where(active_mask, diff, np.nan)
            dg = og.require_group("diff")
            dg.create_dataset("map", data=diff, compression="gzip")
            edges, counts = _compute_hist(masked_diff)
            dg.create_dataset("hist_edges",  data=edges)
            dg.create_dataset("hist_counts", data=counts)

        # ── Noise ──────────────────────────────────────────────────────────────
        ng = f.require_group("noise")
        noise = r["noise"].astype(np.float32)
        masked_noise = np.where(active_mask, noise, np.nan)
        ng.create_dataset("map", data=noise, compression="gzip")
        edges, counts = _compute_hist(masked_noise)
        ng.create_dataset("hist_edges",  data=edges)
        ng.create_dataset("hist_counts", data=counts)

        # CM noise per ASIC
        cm_noise = r.get("cm_noise")
        if cm_noise is not None:
            cmg = ng.require_group("cm_noise")
            if isinstance(cm_noise, dict):
                for name, vals in cm_noise.items():
                    cmg.create_dataset(name, data=vals.astype(np.float32))
            else:
                cmg.create_dataset("global", data=np.asarray(cm_noise).astype(np.float32))

        # n_clipped map
        n_clipped = r.get("n_clipped_map")
        if n_clipped is not None:
            cg = ng.require_group("n_clipped")
            cg.create_dataset("map", data=n_clipped.astype(np.float32), compression="gzip")

        # ── Bad pixels ────────────────────────────────────────────────────────
        bp = f.require_group("bad_pixels")
        bp_mask = r.get("bad_pixel_mask")
        if bp_mask is not None:
            bp.create_dataset("mask", data=bp_mask)

            # Category map (if available from bad pixel mask)
            if hasattr(bp_mask, "category_map"):
                bp.create_dataset("category_map", data=bp_mask.category_map)

            # Noise histogram for threshold plot
            masked_noise = np.where(active_mask, noise, np.nan)
            edges, counts = _compute_hist(masked_noise)
            bp.create_dataset("noise_hist_edges",  data=edges)
            bp.create_dataset("noise_hist_counts", data=counts)

            # Thresholds
            tg = bp.require_group("thresholds")
            med_noise = float(np.nanmedian(masked_noise))
            hot_thr = med_noise * gen.get("bad_pixel_mask", {}).get("hot_rms_multiple", 5.0)
            cold_thr = med_noise * gen.get("bad_pixel_mask", {}).get("cold_rms_fraction", 0.1)
            tg.create_dataset("median_noise",  data=med_noise)
            tg.create_dataset("hot_threshold",  data=hot_thr)
            tg.create_dataset("cold_threshold", data=cold_thr)

            # Per-ASIC bad pixel counts
            if ASIC_WIDTH > 0:
                n_cols = int(noise.shape[1])
                asic_width = n_cols // n_asics if n_asics > 0 else ASIC_WIDTH
                asic_labels = [f"C{i}" for i in range(n_asics)]
                n_bad = []
                for i in range(n_asics):
                    x0 = i * asic_width
                    x1 = x0 + asic_width
                    n_bad.append(int(bp_mask[x0:x1].any(axis=(0, 1)).sum()
                                   if bp_mask.ndim == 2 else bp_mask[x0:x1].sum()))
                ag = bp.require_group("per_asic")
                ag.create_dataset("asic_labels", data=[l.encode() for l in asic_labels])
                ag.create_dataset("n_bad", data=np.array(n_bad, dtype=np.int32))

        # ── Metadata ──────────────────────────────────────────────────────────
        mg = f.require_group("meta")
        mg.attrs["n_dark_frames"] = n_frames
        mg.attrs["sigma_clip_nsigma"] = n_sigma
        mg.attrs["methods"] = ",".join(methods)
        mg.attrs["n_asics"] = n_asics
        mg.attrs["asic_width"] = ASIC_WIDTH

    print("  ✓ saved.")


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
