"""
pnccd_ana.cli.event_rec
========================
Command-line entry point for event recognition.

Usage
-----
  python -m pnccd_ana.cli.event_rec analysis.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from ..config import Config
from ..physics import (find_events,
                       EVENT_DTYPE,
                       build_bad_pixel_mask,
                       ASIC_SLICES, ASIC_WIDTH,
                       configure_asics, get_active_mask,
                       cm_correct_frame)
from ..io import (load_calibration_h5, load_calibration_npy,
                  save_events_h5,
                  save_event_rec_results_h5)
from ..io.hdf5 import _compute_raw_spectrum_data
from ..plotting import (plot_hitmap, plot_spectrum, plot_raw_spectrum)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_cal(cal_source: str,
              asics: list[str] | None,
              prefer: str = "sigclip") -> dict[str, dict]:
    """Load calibration from HDF5 file or directory of .npy files."""
    p = Path(cal_source)
    if p.is_file():
        return load_calibration_h5(p, asics=asics, prefer=prefer)
    elif p.is_dir():
        return load_calibration_npy(p, asics=asics, prefer=prefer)
    else:
        raise FileNotFoundError(f"Calibration source not found: {p}")


def _build_noise_map(cal: dict,
                     asics: list[str] | None = None,
                     noise_scope: str = "auto") -> np.ndarray:
    """
    Assemble a full noise map from calibration for single-hybrid.
    
    For single-hybrid mode, returns the global noise map directly.
    """
    noise_scope = (noise_scope or "auto").lower()
    
    # Get dimensions from calibration
    if "global" in cal and "noise" in cal["global"]:
        noise = cal["global"]["noise"].copy()
        return noise
    else:
        raise RuntimeError("No global noise map found in calibration.")


# ──────────────────────────────────────────────────────────────────────────────
# Bad-pixel mask construction
# ──────────────────────────────────────────────────────────────────────────────

_BAD_PIXEL_DEFAULTS = {
    "enabled":           True,
    "hot_rms_multiple":  5.0,
    "cold_rms_fraction": 0.1,
    "max_clip_fraction": 0.5,
}


def _build_bad_pixel_mask(
        cal:         dict,
        noise_map:   np.ndarray,
        search_mask: np.ndarray | None,
        user_cfg:    dict | None,
) -> np.ndarray | None:
    """
    Assemble a bad-pixel mask from the dark calibration for single-hybrid.

    For single-hybrid mode, uses the global noise map directly.
    """
    cfg = {**_BAD_PIXEL_DEFAULTS, **(user_cfg or {})}
    if not cfg["enabled"]:
        print("\nBad-pixel mask: DISABLED via config (bad_pixel_mask.enabled=false)")
        return None

    print("\nBuilding bad-pixel mask:")
    
    active = (search_mask if search_mask is not None
              else np.ones(noise_map.shape, dtype=bool))
    
    clip = None
    if "global" in cal:
        clip = cal["global"].get("n_clipped_map")
    
    mask = build_bad_pixel_mask(
        noise_map,
        n_clipped_map     = clip,
        n_dark_frames     = int(cfg.get("n_dark_frames") or 0),
        hot_rms_multiple  = float(cfg["hot_rms_multiple"]),
        cold_rms_fraction = float(cfg["cold_rms_fraction"]),
        max_clip_fraction = float(cfg["max_clip_fraction"]),
        active_mask       = active,
        label             = "global",
    )

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Per-chunk worker (single-hybrid)
# ──────────────────────────────────────────────────────────────────────────────

def _correct_frame(raw: np.ndarray,
                   cal: dict,
                   asic_slices: dict | None = None,
                   bad_pixel_mask: np.ndarray | None = None,
                   n_bits: int = 16,
                   split_even_odd: bool = False) -> np.ndarray:
    """
    Offset subtract + CM correct one raw frame.

    Overflow and underflow sentinel values are detected from the raw
    integer frame BEFORE offset subtraction and excluded from the CM
    median, matching ROOT HCommonModeMedian.

    When split_even_odd=True, uses HCommonModeMedianEvenOdd logic:
    independent medians for even- and odd-indexed columns within each
    ASIC segment (matching Analysis.Filter.SplitEvenOdd=1).

    Parameters
    ----------
    raw            : uint16 (n_Y, n_X) — raw ADC frame
    cal            : calibration dict with cal["global"]["offset"]
    asic_slices    : ASIC geometry for per-ASIC CM (None = full-row CM)
    bad_pixel_mask : bool (n_Y, n_X) or None — static bad pixels
    n_bits         : ADC bit depth (default 16)
    split_even_odd : separate even/odd column medians per ASIC

    Returns
    -------
    corrected : float32 (n_Y, n_X)
    """
    overflow_val = (1 << n_bits) - 1

    if np.issubdtype(raw.dtype, np.integer):
        overflow_mask  = (raw == overflow_val)
        underflow_mask = (raw == 0)
    else:
        overflow_mask  = None
        underflow_mask = None

    full = raw.astype(np.float32) - cal["global"]["offset"]

    corrected, _, _ = cm_correct_frame(
        full,
        asic_slices=asic_slices,
        bad_pixel_mask=bad_pixel_mask,
        overflow_mask=overflow_mask,
        underflow_mask=underflow_mask,
        split_even_odd=split_even_odd,
    )
    return corrected


# ── replace the worker ────────────────────────────────────────────────────────

def _make_worker(cal, asics, noise_map,
                 seed_sigma, split_sigma,
                 search_mask,
                 bad_pixel_mask=None,
                 sample_buf=None, sample_max=50,
                 asic_slices=None,
                 split_even_odd=False):
    """
    Return a closure for process_frames_mt.
    Returns concatenated cluster_data dict (CSR) per chunk.
    """
    import threading
    _lock = threading.Lock()

    def _worker(raw_chunk: np.ndarray,
                frame_indices: np.ndarray) -> dict:
        chunk_Y:   list[np.ndarray] = []
        chunk_X:   list[np.ndarray] = []
        chunk_adu: list[np.ndarray] = []
        chunk_off: list[np.ndarray] = []   # per-frame offset arrays
        chunk_flg: list[np.ndarray] = []
        running_offset = np.int64(0)

        for frame in raw_chunk:
            corrected = _correct_frame(frame, cal,
                                       asic_slices=asic_slices,
                                       bad_pixel_mask=bad_pixel_mask,
                                       split_even_odd=split_even_odd)
            if sample_buf is not None:
                with _lock:
                    if len(sample_buf) < sample_max:
                        sample_buf.append(corrected.copy())

            cd = find_events(corrected, noise_map,
                             search_mask=search_mask,
                             seed_sigma=seed_sigma,
                             split_sigma=split_sigma,
                             bad_pixel_mask=bad_pixel_mask)

            n_evt = len(cd["flag"])
            if n_evt == 0:
                continue

            chunk_Y.append(cd["pixel_Y"])
            chunk_X.append(cd["pixel_X"])
            chunk_adu.append(cd["pixel_adu"])
            # Shift offsets by running total pixel count
            chunk_off.append(cd["offsets"][:-1] + running_offset)
            chunk_flg.append(cd["flag"])
            running_offset += np.int64(len(cd["pixel_Y"]))

        if not chunk_flg:
            return _empty_cluster_chunk()

        # Merge all frames in this chunk into one CSR block
        return {
            "pixel_Y":        np.concatenate(chunk_Y).astype(np.int16),
            "pixel_X":        np.concatenate(chunk_X).astype(np.int16),
            "pixel_adu":      np.concatenate(chunk_adu).astype(np.float32),
            "offsets_no_end": np.concatenate(chunk_off).astype(np.int64),
            "flag":           np.concatenate(chunk_flg).astype(np.uint8),
            "n_pixels_total": int(running_offset),
        }
    return _worker


def _empty_cluster_chunk() -> dict:
    return {
        "pixel_Y":        np.empty(0, dtype=np.int16),
        "pixel_X":        np.empty(0, dtype=np.int16),
        "pixel_adu":      np.empty(0, dtype=np.float32),
        "offsets_no_end": np.empty(0, dtype=np.int64),
        "flag":           np.empty(0, dtype=np.uint8),
        "n_pixels_total": 0,
    }


def _merge_chunks(results: list[dict]) -> dict:
    """
    Merge per-chunk CSR dicts into one global CSR cluster_data dict.

    Each chunk's offsets_no_end is shifted by the cumulative pixel count
    of all previous chunks, then the final sentinel is appended.
    """
    non_empty = [r for r in results if r["n_pixels_total"] > 0]
    if not non_empty:
        from ..physics.event_filter import _empty_result
        return _empty_result()

    all_Y   = np.concatenate([r["pixel_Y"]   for r in non_empty])
    all_X   = np.concatenate([r["pixel_X"]   for r in non_empty])
    all_adu = np.concatenate([r["pixel_adu"] for r in non_empty])
    all_flg = np.concatenate([r["flag"]      for r in non_empty])

    # Rebuild global offsets: shift each chunk's offsets by cumulative pixels
    cum_pixels = np.int64(0)
    off_parts: list[np.ndarray] = []
    for r in non_empty:
        off_parts.append(r["offsets_no_end"] + cum_pixels)
        cum_pixels += np.int64(r["n_pixels_total"])
    all_off = np.concatenate(off_parts)
    # Append final sentinel
    all_off = np.append(all_off, cum_pixels).astype(np.int64)

    return {
        "pixel_Y":   all_Y,
        "pixel_X":   all_X,
        "pixel_adu": all_adu,
        "offsets":   all_off,
        "flag":      all_flg,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Multi-file path resolution
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_paths(file_spec) -> list[Path]:
    """
    Resolve a file specification to a sorted list of concrete paths.

    Accepts:
      - a single path string, optionally containing shell glob wildcards
        (e.g. "data/source_run00*_H1.raw")
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
    """Execute the event recognition pipeline from a Config object."""
    ec      = cfg.event_rec
    gen     = cfg.general
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source_run_file = ec["source_run_file"]
    calibration_file = ec["calibration_file"]
    
    # Resolve paths: input files from data_dir, output files to output_dir
    if not source_run_file:
        raise ValueError("[event_rec] source_run_file must be set.")
    
    source_run_path = cfg.resolve_input_path(source_run_file)
    
    # Calibration file: default to output_dir/offset.h5
    # If user provides a path, try data_dir first, then output_dir
    if calibration_file:
        cal_input = cfg.resolve_input_path(calibration_file)
        cal_output = cfg.resolve_output_path(calibration_file)
        if cal_input and cal_input.exists():
            calibration_path = cal_input
        elif cal_output and cal_output.exists():
            calibration_path = cal_output
        else:
            raise FileNotFoundError(
                f"Calibration file not found: tried {cal_input} and {cal_output}")
    else:
        calibration_path = cfg.calibration_path()

    asics           = cfg.asics_for("event_rec")
    seed_sigma      = float(ec.get("seed_sigma",  ec.get("threshold_sigma", 5.0)))
    split_sigma     = float(ec.get("split_sigma", 3.0))
    reject_extra    = bool(ec.get("reject_extra", False))
    prefer          = ec.get("prefer_offset", "sigclip")
    noise_scope     = ec.get("noise_scope", "auto")
    stage_skip      = int(ec.get("skip_frames", 0))
    stage_max       = ec.get("max_frames", None)
    if stage_max is not None:
        stage_max = int(stage_max)
    effective_max   = stage_max if stage_max is not None else gen["max_frames"]

    # ── Load calibration ──────────────────────────────────────────────────────
    print(f"\nLoading calibration from: {calibration_path}")
    cal = _load_cal(calibration_path, asics=None, prefer=prefer)
    
    # Ensure global calibration exists
    if "global" not in cal:
        raise RuntimeError("Calibration must contain 'global' section for single-hybrid mode.")

    noise_map = _build_noise_map(cal, noise_scope=noise_scope)
    H, W = noise_map.shape

    # ── Configure ASIC geometry ───────────────────────────────────────────────
    n_asics = int(gen.get("ASIC_num", 8))
    asic_mask = gen.get("ASIC_mask", [])
    configure_asics(n_asics, W, H, mask=asic_mask)
    
    # Per-ASIC CM slices - include ALL ASICs for CM correction (even masked ones)
    # CM is computed per ASIC using 64 columns each
    asic_slices = ASIC_SLICES if len(ASIC_SLICES) > 1 else None

    # ── Build search mask (respects ASIC mask) ──────────────────────────────
    # Create active mask that excludes masked ASICs from event search
    active_mask = get_active_mask(H, W, n_asics, asic_mask)
    search_mask: np.ndarray | None = active_mask

    # ── Build bad-pixel mask from calibration ─────────────────────────────────
    bad_pixel_mask = _build_bad_pixel_mask(cal, noise_map=noise_map, search_mask=active_mask,
                                            user_cfg=ec.get("bad_pixel_mask"))
    if bad_pixel_mask is not None and bool(bad_pixel_mask.any()):
        np.save(out_dir / "bad_pixel_mask.npy", bad_pixel_mask)

    # ── Resolve file list (single path, glob, or list) ────────────────────────
    run_files = _resolve_paths(source_run_path)
    print(f"\nSource run file(s): {len(run_files)} file(s) matched")
    for p in run_files:
        print(f"  {p}")

    # ── Discover and process frames across all files ──────────────────────────
    # Always uses RAW format (512x512 or 1024x512 based on frame_rows config)
    from ..io.raw import get_io_module as raw_get_io_module
    io = raw_get_io_module(gen.get("data_format", "raw"))

    raw_kwargs = {}
    if gen.get("frame_rows"):
        raw_kwargs["height"] = gen["frame_rows"]
    if gen.get("frame_cols"):
        raw_kwargs["width"] = gen["frame_cols"]

    # Print ASIC configuration
    if asic_mask:
        print(f"  ASIC mask applied: excluding ASICs {asic_mask}")
    print(f"  ASIC configuration: {n_asics} ASICs × {ASIC_WIDTH} columns = {W} total columns")
    print(f"  Per-ASIC CM correction: {'enabled' if asic_slices else 'disabled'}")

    # Sample buffer for raw spectrum plots (collect up to 200 corrected frames)
    # Shared across all input files — worker appends to it as frames are processed.
    sample_buf: list = []

    split_even_odd = bool(ec.get("split_even_odd", True))
    print(f"  Even/odd CM split: {'enabled' if split_even_odd else 'disabled'}")

    worker = _make_worker(cal, asics, noise_map,
                          seed_sigma, split_sigma,
                          search_mask,
                          bad_pixel_mask=bad_pixel_mask,
                          sample_buf=sample_buf, sample_max=200,
                          asic_slices=asic_slices,
                          split_even_odd=split_even_odd)

    all_results: list[np.ndarray] = []
    remaining = effective_max
    total_frames_processed = 0
    is_first_file = True

    for fpath in run_files:
        if remaining is not None and remaining <= 0:
            break
        print(f"\nOpening source file: {fpath}")
        request_max = remaining if remaining is None else remaining + (stage_skip if is_first_file else 0)
        indices = io.get_frame_indices(fpath,
                                       complete_only=gen["complete_only"],
                                       max_frames=request_max,
                                       **raw_kwargs)
        if is_first_file and stage_skip > 0:
            from ..io.raw import _apply_frame_selection
            indices = _apply_frame_selection(indices,
                                             skip_frames=stage_skip,
                                             max_frames=remaining)
        is_first_file = False
        print(f"Processing {len(indices)} frames  "
              f"(seed={seed_sigma}σ, split={split_sigma}σ,  "
              f"workers={gen['n_workers']}, chunk={gen['chunk_size']})")
        file_results = io.process_frames_mt(fpath, indices, worker,
                                            chunk_size=gen["chunk_size"],
                                            n_workers=gen["n_workers"],
                                            desc="source frames",
                                            **raw_kwargs)
        all_results.extend(file_results)
        if remaining is not None:
            remaining -= len(indices)
        total_frames_processed += len(indices)

    print(f"\n  Frame selection: skip={stage_skip}  "
          f"max={effective_max if effective_max is not None else 'all'}  "
          f"total loaded={total_frames_processed}")


    results = all_results

    # ── Merge chunks ──────────────────────────────────────────────────────────
    cluster_data = _merge_chunks(all_results)
    n_events     = len(cluster_data["flag"])
    print(f"\nTotal events: {n_events:,}")
    print(f"Total pixels: {len(cluster_data['pixel_Y']):,}")

    # ── Diagnostic summary ────────────────────────────────────────────────────
    if n_events > 0:
        from ..physics.event_filter import n_pixels_per_event, adu_sums
        npix    = n_pixels_per_event(cluster_data)
        adusums = adu_sums(cluster_data)

        npix_vals, npix_counts = np.unique(npix, return_counts=True)
        size_dist = {int(n): int(c)
                     for n, c in zip(npix_vals, npix_counts)}
        print(f"  Cluster-size distribution: {size_dist}")
        print(f"  adu_sum: min={adusums.min():.0f}  "
              f"max={adusums.max():.0f}  "
              f"median={np.median(adusums):.0f}  ADU")

    # ── Spectra by cluster size ───────────────────────────────────────────────
    bin_edges = np.linspace(ec["adu_min"], ec["adu_max"], ec["n_bins"] + 1)
    spectra: dict[int, np.ndarray] = {}
    if n_events > 0:
        npix    = n_pixels_per_event(cluster_data)
        adusums = adu_sums(cluster_data)
        for n in range(1, 6):
            m = npix == n
            spectra[n], _ = (np.histogram(adusums[m], bins=bin_edges)
                             if m.any() else
                             (np.zeros(ec["n_bins"], dtype=np.int64), bin_edges))
        m_large = npix >= 5
        spectra[5], _ = (np.histogram(adusums[m_large], bins=bin_edges)
                         if m_large.any() else
                         (np.zeros(ec["n_bins"], dtype=np.int64), bin_edges))
    else:
        for n in range(1, 6):
            spectra[n] = np.zeros(ec["n_bins"], dtype=np.int64)

    # ── 2-D maps ──────────────────────────────────────────────────────────────
    hit_count = np.zeros((H, W), dtype=np.int32)
    hit_adu   = np.zeros((H, W), dtype=np.float64)
    if n_events > 0:
        from ..physics.event_filter import seed_pixels, adu_sums as _sums
        sY, sX, _ = seed_pixels(cluster_data)
        adusums   = _sums(cluster_data)
        np.add.at(hit_count, (sY.astype(int), sX.astype(int)), 1)
        np.add.at(hit_adu,   (sY.astype(int), sX.astype(int)),
                  adusums.astype(np.float64))
    with np.errstate(invalid="ignore"):
        mean_adu = np.where(hit_count > 0,
                            hit_adu / hit_count, np.nan).astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_events_cfg = ec.get("save_events", "events.h5")
    if save_events_cfg:
        events_path = cfg.resolve_output_path(save_events_cfg)
        if events_path is None:
            events_path = cfg.events_path()
        save_events_h5(
            events_path,
            cluster_data, spectra, bin_edges, hit_count, mean_adu,
            metadata={**gen.get("metadata", {}),
                      "source_file":    str(source_run_path),
                      "calibration":    str(calibration_path),
                      "seed_sigma":     seed_sigma,
                      "split_sigma":    split_sigma,
                      "n_frames":       int(total_frames_processed)},
        )

    # Maps are saved in events.h5 under /maps/

    # ── Save plot-backing data ────────────────────────────────────────────────
    if sample_buf:
        sample_arr = np.stack(sample_buf, axis=0)
    else:
        sample_arr = None
    save_event_rec_results_h5(
        out_dir, events, spectra, bin_edges, hit_count, mean_adu,
        sample_arr, noise_map, gen, seed_sigma, total_frames_processed,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    if gen["save_frame_plots"]:
        plot_hitmap(hit_count, mean_adu, out_dir, asics)
        plot_spectrum(spectra, bin_edges, out_dir, events=events)
        # plot_grade_distribution removed — no grades assigned here

        if sample_arr is not None:
            plot_raw_spectrum(
                corrected_frames=sample_arr,
                noise_map=noise_map,
                bin_edges=bin_edges,
                out_dir=out_dir,
                seed_sigma=seed_sigma,
                title_suffix="global",
            )

    print(f"\n✓ Source analysis complete.  Output: {out_dir}/")
    return dict(events=events, spectra=spectra, bin_edges=bin_edges,
                hit_count=hit_count, mean_adu=mean_adu)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="pnCCD Fe-55 source analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("config", help="Path to analysis.yaml")
    args = parser.parse_args(argv)
    cfg  = Config.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
