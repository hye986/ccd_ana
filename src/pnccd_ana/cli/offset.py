"""
pnccd_ana.cli.offset
=====================
Command-line entry point for offset calibration.

Usage
-----
  python -m pnccd_ana.cli.offset analysis.yaml
  pnccd-offset analysis.yaml

  To generate a template config:
  pnccd-template analysis.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import Config
from ..physics import (compute_offset_median,
                       compute_offset_sigma_clip,
                       apply_common_mode_correction,
                       compute_noise,
                       compute_cm_noise,
                       ASIC_SLICES, ASIC_WIDTH,
                       build_bad_pixel_mask,
                       configure_asics, get_active_mask)
from ..io import save_calibration_npy
from ..plotting import (plot_offsets, plot_noise, plot_cm_map, plot_bad_pixels)


# ──────────────────────────────────────────────────────────────────────────────
# Per-scope analysis (single-hybrid)
# ──────────────────────────────────────────────────────────────────────────────

def _analyse_scope(data:          np.ndarray,
                   label:         str,
                   methods:       list[str],
                   n_sigma:       float,
                   out_dir:       Path,
                   asic_slices:   dict[str, tuple[int, int, int, int]] | None = None,
                   build_bp_mask: bool = True,
                   active_mask:   np.ndarray | None = None,
                   split_even_odd: bool = False,
                   save_plots:    bool = True,
                   bpm_cfg:       dict | None = None) -> dict:
    """
    Run offset → CM → noise pipeline for one data block.

    Parameters
    ----------
    data          : (N, H, W) float32 — dark frames
    label         : scope name used in prints and filenames
    methods       : ['median'], ['sigclip'], or both
    n_sigma       : sigma-clip threshold
    out_dir       : output directory
    asic_slices   : ASIC geometry for per-ASIC CM correction
    build_bp_mask : whether to build and plot the bad pixel mask
    active_mask   : bool (H, W) — True where pixel is in an active ASIC

    Returns
    -------
    dict with keys: offset_median?, offset_sigclip?, noise, cm_noise,
                    n_clipped_map?, cm_map, asic_names, bad_pixel_mask?
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
        data, ref_offset, label, asic_slices=asic_slices, split_even_odd=split_even_odd)
    r["noise"]    = compute_noise(corrected, keep_mask, label)
    r["cm_noise"] = compute_cm_noise(cm_map, label, asic_names=asic_names)
    r["cm_map"]    = cm_map
    r["asic_names"] = asic_names

    if build_bp_mask:
        bpm = bpm_cfg or {}
        bad_pixel_mask = build_bad_pixel_mask(
            r["noise"],
            n_clipped_map=n_clipped_map,
            n_dark_frames=data.shape[0] if n_clipped_map is not None else 0,
            hot_rms_multiple=float(bpm.get("hot_rms_multiple", 5.0)),
            cold_rms_fraction=float(bpm.get("cold_rms_fraction", 0.1)),
            max_clip_fraction=float(bpm.get("max_clip_fraction", 0.5)),
            active_mask=active_mask,
            label=label,
        )
        r["bad_pixel_mask"] = bad_pixel_mask

    if save_plots:
        scope_dir = out_dir / label
        scope_dir.mkdir(parents=True, exist_ok=True)
        plot_offsets(label, r, scope_dir, active_mask=active_mask)
        plot_noise(label, r, scope_dir, active_mask=active_mask)
        plot_cm_map(label, cm_map, scope_dir, asic_names=asic_names)
        if build_bp_mask and r.get("bad_pixel_mask") is not None:
            plot_bad_pixels(label, r["noise"], r["bad_pixel_mask"],
                            scope_dir, active_mask=active_mask)

    return r


# ──────────────────────────────────────────────────────────────────────────────
# Multi-file path resolution
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_paths(file_spec) -> list[Path]:
    """
    Resolve a file specification to a sorted list of concrete paths.

    Accepts a single path string (with optional shell globs) or a list
    of path strings.  Raises FileNotFoundError if no files match.
    """
    from glob import glob

    specs = [str(file_spec)] if isinstance(file_spec, (str, Path)) \
            else [str(s) for s in file_spec]
    paths: list[Path] = []
    for spec in specs:
        matched = sorted(glob(spec))
        if not matched:
            p = Path(spec)
            if not p.exists():
                raise FileNotFoundError(f"No files found matching: {spec!r}")
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
    oc      = cfg.offset
    gen     = cfg.general
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dark_run_file = oc["dark_run_file"]
    if not dark_run_file:
        raise ValueError(
            "[offset] dark_run_file must be set in config.\n"
            "  Run  pnccd-template  to generate a starter config.")

    dark_run_path = cfg.resolve_input_path(dark_run_file)
    methods       = cfg.methods_for_offset()
    n_sigma       = float(oc["sigma_clip_nsigma"])

    # ── Resolve file list ─────────────────────────────────────────────────────
    run_files = _resolve_paths(dark_run_path)
    print(f"\nOffset calibration")
    print(f"  Dark run file(s): {len(run_files)} matched")
    for p in run_files:
        print(f"    {p}")

    # ── Load dark frames ──────────────────────────────────────────────────────
    from ..io import raw as _raw

    def _passthrough(raw: np.ndarray, _idx: np.ndarray) -> np.ndarray:
        return raw

    stage_skip = int(oc.get("skip_frames", 0))
    stage_max = oc.get("max_frames", None)
    if stage_max is not None:
        stage_max = int(stage_max)
    effective_max = stage_max if stage_max is not None else gen["max_frames"]

    all_chunks: list[np.ndarray] = []
    remaining = effective_max
    total_frames = 0
    is_first_file = True

    for fpath in run_files:
        if remaining is not None and remaining <= 0:
            break
        print(f"\n  Loading: {fpath}")
        request_max = remaining if remaining is None else remaining + (stage_skip if is_first_file else 0)
        indices = _raw.get_frame_indices(fpath, max_frames=request_max)
        if is_first_file and stage_skip > 0:
            indices = _raw._apply_frame_selection(indices,
                                                  skip_frames=stage_skip,
                                                  max_frames=remaining)
        is_first_file = False
        chunks = _raw.process_frames_mt(fpath, indices, _passthrough,
                                        chunk_size=gen["chunk_size"],
                                        n_workers=gen["n_workers"],
                                        desc="dark frames")
        all_chunks.extend(chunks)
        if remaining is not None:
            remaining -= len(indices)
        total_frames += len(indices)

    print(f"\n  Frame selection: skip={stage_skip}  "
          f"max={effective_max if effective_max is not None else 'all'}  "
          f"total loaded={total_frames}")

    data_raw = np.concatenate(all_chunks, axis=0)
    H, W = data_raw.shape[1], data_raw.shape[2]
    print(f"\n  Total: {data_raw.shape[0]} frames  ({H} × {W} pixels)"
          f"  from {len(run_files)} file(s)")

    # ── Configure ASIC geometry ───────────────────────────────────────────────
    n_asics   = int(gen.get("ASIC_num", 8))
    asic_mask = gen.get("ASIC_mask", [])
    configure_asics(n_asics, W, H, mask=asic_mask)
    if asic_mask:
        print(f"  ASIC mask: excluding {asic_mask}")
    print(f"  ASICs: {n_asics} × {ASIC_WIDTH} columns = {W} total")

    asic_slices = ASIC_SLICES if len(ASIC_SLICES) > 1 else None
    active_mask = get_active_mask(H, W, n_asics, asic_mask)

    # ── Run analysis ──────────────────────────────────────────────────────────
    all_results: dict = {}
    split_even_odd = bool(oc.get("split_even_odd", True))
    # PLOT_LABEL is used for the plot subfolder name and plot titles.
    PLOT_LABEL = "offset"
    all_results["global"] = _analyse_scope(
        data_raw, PLOT_LABEL, methods, n_sigma, out_dir,
        asic_slices=asic_slices, build_bp_mask=True, active_mask=active_mask,
        split_even_odd=split_even_odd, save_plots=bool(oc.get("save_plots", True)),
        bpm_cfg=oc.get("bad_pixel_mask"))

    # ── Save .npy outputs ─────────────────────────────────────────────────────
    r = all_results["global"]
    if oc["save_npy"]:
        save_calibration_npy(
            out_dir,
            {"global": r},
            cm_map=r.get("cm_map"),
            bad_pixel_mask=r.get("bad_pixel_mask"),
        )

    print(f"\n✓ Offset calibration complete.  Output: {out_dir}/")
    return all_results
# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="pnCCD offset calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("config", help="Path to analysis.yaml  "
                        "(run  pnccd-template  to generate one)")
    args = parser.parse_args(argv)
    cfg  = Config.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
