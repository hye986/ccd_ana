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
                       ASIC_SLICES, ALL_ASICS)
from ..utils  import (save_calibration_h5, save_calibration_npy,
                       plot_offsets, plot_noise,
                       plot_cm_map, plot_asic_overview,
                       plot_summary_dashboard)


# ──────────────────────────────────────────────────────────────────────────────
# Per-scope analysis (single-hybrid)
# ──────────────────────────────────────────────────────────────────────────────

def _analyse_scope(data:        np.ndarray,
                   label:       str,
                   methods:     list[str],
                   n_sigma:     float,
                   out_dir:     Path) -> dict:
    """
    Run pedestal → CM → noise pipeline for one data block (single-hybrid).

    Parameters
    ----------
    data       : (N, H, W) float32 array — frames from RAW file
    label      : scope name used in prints and filenames
    methods    : ['median'], ['sigclip'], or both
    n_sigma    : sigma-clip threshold
    out_dir    : output directory

    Returns dict with keys: offset_median?, offset_sigclip?,
                             noise, cm_noise, n_clipped_map?,
                             cm_map, data_corrected
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
    corrected, cm_map   = apply_common_mode_correction(data, ref_offset, label)
    r["noise"]          = compute_noise(corrected, keep_mask, label)
    r["cm_noise"]       = compute_cm_noise(cm_map, label)
    r["cm_map"]         = cm_map
    r["data_corrected"] = corrected

    scope_dir = out_dir / label
    scope_dir.mkdir(parents=True, exist_ok=True)
    plot_offsets(label, r, scope_dir)
    plot_noise(label, r, scope_dir)
    plot_cm_map(label, cm_map, scope_dir)

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
    from glob import glob

    specs = [file_spec] if isinstance(file_spec, str) else list(file_spec)
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
    # Always uses RAW format (512x512 or 1024x512 based on raw_height config)
    from ..utils.io_raw import get_io_module as raw_get_io_module
    io = raw_get_io_module()

    raw_kwargs = {}
    if dc.get("raw_height"):
        raw_kwargs["height"] = dc["raw_height"]
    if dc.get("raw_width"):
        raw_kwargs["width"] = dc["raw_width"]

    def _load_chunk(raw: np.ndarray, _idx: np.ndarray) -> np.ndarray:
        return raw   # pass raw frames through unchanged

    all_chunks: list[np.ndarray] = []
    remaining = dc["max_frames"]   # None = unlimited; decremented per file

    for fpath in run_files:
        if remaining is not None and remaining <= 0:
            break
        print(f"\nLoading dark frames from: {fpath}")
        indices = io.get_frame_indices(fpath,
                                       complete_only=dc["complete_only"],
                                       max_frames=remaining,
                                       **raw_kwargs)
        chunks = io.process_frames_mt(fpath, indices, _load_chunk,
                                      chunk_size=dc["chunk_size"],
                                      n_workers=dc["n_workers"],
                                      desc="dark frames",
                                      **raw_kwargs)
        all_chunks.extend(chunks)
        if remaining is not None:
            remaining -= len(indices)

    data_raw = np.concatenate(all_chunks, axis=0)
    print(f"\n  Total loaded: {data_raw.shape[0]} frames  "
          f"({data_raw.shape[1]} × {data_raw.shape[2]} pixels)"
          f"  from {len(run_files)} file(s)")

    # Update ASIC slices to match detected frame dimensions
    from ..lib.geometry import _update_asic_slices
    _update_asic_slices(data_raw.shape[1], data_raw.shape[2])

    # ── Full-frame analysis ───────────────────────────────────────────────────
    all_results: dict = {"asics": {}}
    all_results["global"] = _analyse_scope(
        data_raw, "global", methods, n_sigma, out_dir)

    # ── Per-ASIC analysis ─────────────────────────────────────────────────────
    if asics:
        asic_data = split_asics(data_raw, asics)
        for aname in asics:
            all_results["asics"][aname] = _analyse_scope(
                asic_data[aname], aname, methods, n_sigma, out_dir)

        if gen["save_asic_plots"]:
            for qty, title, lbl, cmap, log in [
                ("offset_median",  "Median Offset",     "ADU",     "viridis", False),
                ("offset_sigclip", "SigmaClip Offset",  "ADU",     "plasma",  False),
                ("noise",          "Pixel Noise (RMS)", "ADU RMS", "inferno", False),
                ("noise",          "Pixel Noise (log)", "ADU RMS", "inferno", True),
            ]:
                plot_asic_overview(all_results["asics"], qty, title, lbl,
                                   cmap, out_dir, log_scale=log)

    # ── Summary ───────────────────────────────────────────────────────────────
    plot_summary_dashboard(all_results["global"], out_dir)

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
