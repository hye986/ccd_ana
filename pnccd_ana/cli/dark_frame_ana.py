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
# RAW sub-frame embedding
# ──────────────────────────────────────────────────────────────────────────────

def _embed_into_full_frame(
        data:   np.ndarray,
        asics:  list[str],
        label:  str = "",
) -> np.ndarray:
    """
    Embed a sub-frame RAW array into a zero-padded 1024×1024 canvas.

    When a RAW file contains data for only one ASIC (or a subset), the
    loaded frames have shape (N, H, W) where H×W < 1024×1024.  All
    downstream analysis (geometry slicing, noise maps, hit maps, etc.)
    assumes 1024×1024 frames, so we place the sub-frame data at the
    correct pixel position within a full-size canvas.

    The target region is determined by the bounding box of the requested
    ASICs according to ASIC_SLICES.  If the data shape already matches
    1024×1024 this function is a no-op.

    Parameters
    ----------
    data  : (N, H, W) float32 array as loaded from the RAW file
    asics : ASIC names that were recorded (e.g. ['H1'])
    label : string used in the printed message only

    Returns
    -------
    (N, 1024, 1024) float32 array, or the original array if already full-size
    """
    N, H, W = data.shape
    if H == 1024 and W == 1024:
        return data                          # already full-size, nothing to do

    # Compute the bounding box of all requested ASICs in full-frame coordinates
    y0 = min(ASIC_SLICES[a][0] for a in asics)
    y1 = max(ASIC_SLICES[a][1] for a in asics)
    x0 = min(ASIC_SLICES[a][2] for a in asics)
    x1 = max(ASIC_SLICES[a][3] for a in asics)
    bbox_h = y1 - y0 + 1
    bbox_w = x1 - x0 + 1

    if H != bbox_h or W != bbox_w:
        raise ValueError(
            f"RAW frame shape ({H}×{W}) does not match the bounding box of "
            f"ASIC(s) {asics} in full-frame coordinates ({bbox_h}×{bbox_w}). "
            "Check that 'asics' in the config matches the recorded sub-frame."
        )

    canvas = np.zeros((N, 1024, 1024), dtype=data.dtype)
    canvas[:, y0:y1+1, x0:x1+1] = data
    tag = f" [{label}]" if label else ""
    print(f"  Embedded{tag} {H}×{W} RAW sub-frame → 1024×1024 "
          f"at rows {y0}:{y1+1}, cols {x0}:{x1+1}")
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Per-scope analysis (full frame or one ASIC)
# ──────────────────────────────────────────────────────────────────────────────

def _analyse_scope(data:        np.ndarray,
                   label:       str,
                   methods:     list[str],
                   n_sigma:     float,
                   out_dir:     Path,
                   pixel_mask:  np.ndarray | None = None) -> dict:
    """
    Run pedestal → CM → noise pipeline for one data block.

    Parameters
    ----------
    data       : (N, H, W) float32 array — always 3-D, always the full
                 1024×1024 canvas (or a pre-sliced ASIC sub-array)
    label      : scope name used in prints and filenames
    methods    : ['median'], ['sigclip'], or both
    n_sigma    : sigma-clip threshold
    out_dir    : output directory
    pixel_mask : optional boolean (H, W) array; True = active pixel.
                 When set (sub-frame RAW input embedded into 1024×1024),
                 the analysis is run on the tight bounding-box sub-volume
                 so that zero-padded inactive regions are excluded from
                 all pedestal, CM, and noise calculations.  The resulting
                 2-D maps are re-expanded to (H, W) = (1024, 1024) with
                 zeros outside the active region before plotting.

    Returns dict with keys: offset_median?, offset_sigclip?,
                             noise, cm_noise, n_clipped_map?,
                             cm_map, data_corrected
    """
    print(f"\n── {label} ──")
    r: dict       = {}
    keep_mask     = None
    n_clipped_map = None

    # When a pixel_mask is provided (sub-frame RAW embedded in 1024×1024),
    # slice out the tight bounding box of active pixels as a proper 3-D
    # sub-volume.  All lib functions (pedestal, CM, noise) receive a clean
    # (N, bbox_H, bbox_W) array with no zero-padded rows or columns that
    # could bias column medians or noise estimates.
    bbox_slices = None   # (slice_Y, slice_X) used for re-expansion later
    full_shape  = data.shape[1:]   # (H, W) of the canvas before any slicing
    if pixel_mask is not None:
        rows = np.any(pixel_mask, axis=1)
        cols = np.any(pixel_mask, axis=0)
        y0, y1 = int(np.argmax(rows)),  int(len(rows) - 1 - np.argmax(rows[::-1]))
        x0, x1 = int(np.argmax(cols)),  int(len(cols) - 1 - np.argmax(cols[::-1]))
        bbox_slices = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        data = data[:, bbox_slices[0], bbox_slices[1]]   # (N, bbox_H, bbox_W)

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

    # Re-expand 2-D result maps from bbox back to full canvas size so that
    # all plotting functions receive the expected (H, W) shape.
    # Inactive pixels stay zero (plotted as background / masked).
    if bbox_slices is not None:
        H, W = full_shape
        def _expand(arr: np.ndarray) -> np.ndarray:
            if arr is None or arr.ndim != 2:
                return arr
            canvas = np.zeros((H, W), dtype=arr.dtype)
            canvas[bbox_slices[0], bbox_slices[1]] = arr
            return canvas
        for key in ("offset_median", "offset_sigclip",
                    "noise", "cm_noise", "n_clipped_map"):
            if key in r:
                r[key] = _expand(r[key])

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

    # ── Embed sub-frame RAW data into 1024×1024 canvas if needed ─────────────
    # All downstream code (split_asics, ASIC_SLICES, plotting) assumes 1024×1024.
    # If the RAW file is a single-ASIC or partial readout, place it at the
    # correct pixel position and zero-pad the rest.
    # Also build a boolean mask of the active (recorded) pixels so that the
    # global analysis ignores the zero-padded regions in statistics and plots.
    global_mask = None   # None → all pixels active (full 1024×1024 input)
    if dc["data_format"] == "raw" and asics:
        orig_h, orig_w = data_raw.shape[1], data_raw.shape[2]
        data_raw = _embed_into_full_frame(data_raw, asics)
        if orig_h < 1024 or orig_w < 1024:
            # Mark only the pixels that actually contain recorded data.
            # The bounding box is the union of all configured ASICs.
            y0 = min(ASIC_SLICES[a][0] for a in asics)
            y1 = max(ASIC_SLICES[a][1] for a in asics)
            x0 = min(ASIC_SLICES[a][2] for a in asics)
            x1 = max(ASIC_SLICES[a][3] for a in asics)
            global_mask = np.zeros((1024, 1024), dtype=bool)
            global_mask[y0:y1+1, x0:x1+1] = True

    # ── Full-frame analysis ───────────────────────────────────────────────────
    # global_mask restricts the analysis to recorded pixels when the input is
    # a sub-frame RAW file; it is None (= no restriction) for full 1024×1024 inputs.
    all_results: dict = {"asics": {}}
    all_results["global"] = _analyse_scope(
        data_raw, "global", methods, n_sigma, out_dir,
        pixel_mask=global_mask)

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
