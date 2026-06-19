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
from ..lib    import (find_events,
                       N_GRADES, EVENT_DTYPE,
                       build_bad_pixel_mask,
                       ASIC_SLICES, ASIC_WIDTH,
                       configure_asics, get_active_mask)
from ..lib.common_mode import cm_correct_frame
from ..utils  import (load_calibration_h5, load_calibration_npy,
                       save_events_h5,
                       plot_hitmap, plot_spectrum,
                       plot_grade_distribution,
                       plot_raw_spectrum)


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

def _make_worker(cal: dict,
                 asics: list[str] | None,
                 noise_map: np.ndarray,
                 seed_sigma: float,
                 split_sigma: float,
                 reject_extra: bool,
                 search_mask: np.ndarray | None,
                 bad_pixel_mask: np.ndarray | None = None,
                 sample_buf: list | None = None,
                 sample_max: int = 50,
                 asic_slices: dict | None = None):
    """
    Return a closure suitable for process_frames_mt for single-hybrid mode.
    """
    import threading
    _lock = threading.Lock()

    def _worker(raw_chunk: np.ndarray,
                frame_indices: np.ndarray) -> np.ndarray:
        chunk_events: list[np.ndarray] = []
        for frame in raw_chunk:
            corrected = _correct_frame(frame, cal, asic_slices=asic_slices)

            # Collect sample frames for raw spectrum (cheap copy of one frame)
            if sample_buf is not None:
                with _lock:
                    if len(sample_buf) < sample_max:
                        sample_buf.append(corrected.copy())

            evts = find_events(corrected, noise_map,
                               search_mask=search_mask,
                               seed_sigma=seed_sigma,
                               split_sigma=split_sigma,
                               reject_extra=reject_extra,
                               bad_pixel_mask=bad_pixel_mask)
            if len(evts):
                chunk_events.append(evts)
        if chunk_events:
            return np.concatenate(chunk_events)
        return np.empty(0, dtype=EVENT_DTYPE)
    return _worker


def _correct_frame(raw: np.ndarray, cal: dict,
                   asic_slices: dict | None = None) -> np.ndarray:
    """
    Offset subtract + CM correct one raw frame for single-hybrid.
    
    If asic_slices is provided, CM correction is applied per-ASIC.
    """
    full = raw.astype(np.float32) - cal["global"]["offset"]
    corrected, _, _ = cm_correct_frame(full, asic_slices=asic_slices)
    return corrected


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
    from ..utils.io_raw import get_io_module as raw_get_io_module
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

    worker = _make_worker(cal, asics, noise_map,
                          seed_sigma, split_sigma, reject_extra,
                          search_mask,
                          bad_pixel_mask=bad_pixel_mask,
                          sample_buf=sample_buf, sample_max=200,
                          asic_slices=asic_slices)

    all_results: list[np.ndarray] = []
    remaining = gen["max_frames"]
    total_frames_processed = 0

    for fpath in run_files:
        if remaining is not None and remaining <= 0:
            break
        print(f"\nOpening source file: {fpath}")
        indices = io.get_frame_indices(fpath,
                                       complete_only=gen["complete_only"],
                                       max_frames=remaining,
                                       **raw_kwargs)
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


    results = all_results

    # Concatenate event arrays from all chunks
    non_empty = [r for r in results if len(r) > 0]
    events    = np.concatenate(non_empty) if non_empty else np.empty(0, dtype=EVENT_DTYPE)
    print(f"\nTotal events: {len(events):,}")

    # ── Diagnostic summary ────────────────────────────────────────────────────
    if len(events):
        from ..lib.pattern_recognition import GRADE_OTHER as _GRADE_OTHER
        grade_counts = {int(g): int(n)
                        for g, n in zip(*np.unique(events["grade"], return_counts=True))}
        print(f"  Grade distribution: {grade_counts}")
        print(f"  adu_sum  : min={events['adu_sum'].min():.0f}  "
              f"max={events['adu_sum'].max():.0f}  "
              f"median={np.median(events['adu_sum']):.0f}  ADU")
        print(f"  adu_seed : min={events['adu_seed'].min():.0f}  "
              f"max={events['adu_seed'].max():.0f}  "
              f"median={np.median(events['adu_seed']):.0f}  ADU")
        # Check whether adu_sum and adu_seed are suspiciously identical.
        # GRADE_OTHER events are deliberately centre-only, so diagnose that
        # separately from a real failure to sum recognised split patterns.
        n_identical = int((np.abs(events["adu_sum"] - events["adu_seed"]) < 0.1).sum())
        frac_identical = n_identical / len(events) * 100
        n_other = grade_counts.get(_GRADE_OTHER, 0)
        frac_other = n_other / len(events) * 100
        split_mask = (events["grade"] > 0) & (events["grade"] < _GRADE_OTHER)
        n_split = int(split_mask.sum())
        n_split_summed = int((np.abs(events["adu_sum"][split_mask] -
                                     events["adu_seed"][split_mask]) >= 0.1).sum())
        if frac_other > 50:
            print(f"  ⚠  WARNING: {frac_other:.1f}% of events are grade {_GRADE_OTHER} ('other')")
            print("     'other' events are unrecognized 5x5 patterns and store centre ADU only.")
            print("     Use reject_extra: true for spectra, or raise split_sigma if random")
            print("     neighbour noise is creating extra above-split pixels.")
        elif n_split and n_split_summed < 0.8 * n_split:
            print(f"  ⚠  WARNING: only {n_split_summed}/{n_split} recognized split events "
                  "have adu_sum > adu_seed")
            print("     This would indicate a summing bug for grades 1 to (GRADE_OTHER-1).")
        elif frac_identical > 80 and grade_counts.get(0, 0) < len(events) * 0.8:
            print(f"  ⚠  WARNING: {frac_identical:.1f}% of events have adu_sum ≈ adu_seed")
            print("     Most accepted events are centre-only; inspect grade distribution.")
        elif n_split == 0:
            print(f"  ✓  No recognized split events in grades 1–{_GRADE_OTHER - 1}")
        else:
            print(f"  ✓  {n_split_summed}/{n_split} recognized split events have "
                  "adu_sum > adu_seed")

        # Check if bin range covers the peaks
        in_range = int(((events["adu_sum"] >= ec["adu_min"]) &
                        (events["adu_sum"] <= ec["adu_max"])).sum())
        if in_range < len(events) * 0.5:
            print(f"  ⚠  Only {in_range}/{len(events)} events within "
                  f"adu_min={ec['adu_min']:.0f}..adu_max={ec['adu_max']:.0f}")
            print(f"     Plots will auto-range but UPDATE adu_min/adu_max in config.")

    # ── Spectrum histograms ───────────────────────────────────────────────────
    bin_edges = np.linspace(ec["adu_min"], ec["adu_max"], ec["n_bins"] + 1)
    from ..lib.pattern_recognition import _GRADE_DEFS, GRADE_OTHER
    spectra: dict[int, np.ndarray] = {}
    all_grade_ids = [gid for gid, _, _ in _GRADE_DEFS] + [GRADE_OTHER]
    for g in all_grade_ids:
        m = events["grade"] == g
        spectra[g], _ = (np.histogram(events["adu_sum"][m], bins=bin_edges)
                         if m.any() else
                         (np.zeros(ec["n_bins"], dtype=np.int64), bin_edges))

    # ── 2-D maps ──────────────────────────────────────────────────────────────
    hit_count = np.zeros((H, W), dtype=np.int32)
    hit_adu   = np.zeros((H, W), dtype=np.float64)
    if len(events):
        np.add.at(hit_count,
                  (events["Y"].astype(int), events["X"].astype(int)), 1)
        np.add.at(hit_adu,
                  (events["Y"].astype(int), events["X"].astype(int)),
                  events["adu_sum"].astype(np.float64))
    with np.errstate(invalid="ignore"):
        mean_adu = np.where(hit_count > 0,
                            hit_adu / hit_count, np.nan).astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────────
    if events_h5_path:
        # Resolve output path relative to output_dir
        events_path = cfg.resolve_output_path(events_h5_path)
        if events_path is None:
            events_path = cfg.events_path()
        save_events_h5(
            events_path,
            events, spectra, bin_edges, hit_count, mean_adu,
            metadata={**gen.get("metadata", {}),
                      "source_file":    str(source_run_path),
                      "calibration":    str(calibration_path),
                      "threshold_sigma": seed_sigma,
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
        if len(events):
            plot_grade_distribution(events, out_dir)

        # Raw pixel-level spectrum (needs sample frames)
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
# Save plot-backing data to HDF5
# ──────────────────────────────────────────────────────────────────────────────

def _compute_raw_spectrum_data(
        corrected_frames: np.ndarray | None,
        noise_map: np.ndarray,
        bin_edges: np.ndarray,
        seed_sigma: float,
) -> dict | None:
    """
    Compute raw spectrum histogram data from CM-corrected frames.

    Returns dict with:
        - bin_edges: shared bin edges
        - all_pixels: histogram of all CM-corrected pixels
        - positive_pixels: histogram of pixels > 0 ADU
        - above_seed: histogram of pixels above seed threshold
        - seed_threshold_adu: median seed threshold in ADU
        - seed_sigma: seed sigma value
        - median_noise_adu: median noise in ADU
    """
    if corrected_frames is None:
        return None

    corr_flat = corrected_frames.ravel()
    pos_flat  = corr_flat[corr_flat > 0]

    med_noise  = float(np.median(noise_map))
    seed_adu   = seed_sigma * med_noise
    seed_thr   = seed_sigma * noise_map
    above_seed_mask = corrected_frames > seed_thr[np.newaxis]
    hits_flat  = corrected_frames[above_seed_mask].ravel()

    c_all,  _ = np.histogram(corr_flat, bins=bin_edges)
    c_pos,  _ = np.histogram(pos_flat,  bins=bin_edges)
    c_hits, _ = np.histogram(hits_flat, bins=bin_edges)

    return {
        "bin_edges":           bin_edges.astype(np.float32),
        "all_pixels":          c_all.astype(np.int64),
        "positive_pixels":     c_pos.astype(np.int64),
        "above_seed":          c_hits.astype(np.int64),
        "seed_threshold_adu": float(seed_adu),
        "seed_sigma":          float(seed_sigma),
        "median_noise_adu":    med_noise,
    }


def _compute_grade_distribution(events: np.ndarray) -> dict:
    """Compute grade distribution data from events array."""
    from ..lib.pattern_recognition import _GRADE_DEFS, GRADE_OTHER

    all_grades = sorted(set(g for g, _, _ in _GRADE_DEFS) | {GRADE_OTHER})
    grades = []
    counts = []
    names  = []

    for g in all_grades:
        cnt = int((events["grade"] == g).sum())
        if cnt > 0 or g in [x[0] for x in _GRADE_DEFS]:
            grades.append(g)
            counts.append(cnt)
            # Get grade name
            for gd_g, gd_label, _ in _GRADE_DEFS:
                if gd_g == g:
                    names.append(gd_label)
                    break
            else:
                if g == GRADE_OTHER:
                    names.append("other")

    return {
        "grades":      np.array(grades, dtype=np.int32),
        "counts":      np.array(counts, dtype=np.int64),
        "grade_names": names,
    }


def save_event_rec_results_h5(
        out_dir:        Path,
        events:         np.ndarray,
        spectra:        dict[int, np.ndarray],
        bin_edges:      np.ndarray,
        hit_count:      np.ndarray,
        mean_adu:       np.ndarray,
        sample_arr:     np.ndarray | None,
        noise_map:      np.ndarray,
        gen:            dict,
        seed_sigma:     float,
        n_frames:       int,
) -> None:
    """
    Save plot-backing data to event_rec_results.h5.

    HDF5 structure:
        /raw_spectrum/{bin_edges, all_pixels, positive_pixels, above_seed,
                       seed_threshold_adu, seed_sigma, median_noise_adu}
        /grade_distribution/{grades, counts, grade_names}
        /meta/...
    """
    path = out_dir / "event_rec_results.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving event_rec results: {path}")

    with h5py.File(path, "w") as f:
        # ── Raw spectrum ───────────────────────────────────────────────────────
        raw_data = _compute_raw_spectrum_data(sample_arr, noise_map, bin_edges, seed_sigma)
        if raw_data is not None:
            rg = f.require_group("raw_spectrum")
            rg.create_dataset("bin_edges",          data=raw_data["bin_edges"])
            rg.create_dataset("all_pixels",         data=raw_data["all_pixels"])
            rg.create_dataset("positive_pixels",    data=raw_data["positive_pixels"])
            rg.create_dataset("above_seed",         data=raw_data["above_seed"])
            rg.create_dataset("seed_threshold_adu", data=raw_data["seed_threshold_adu"])
            rg.create_dataset("seed_sigma",          data=raw_data["seed_sigma"])
            rg.create_dataset("median_noise_adu",    data=raw_data["median_noise_adu"])

        # ── Grade distribution ─────────────────────────────────────────────────
        if len(events) > 0:
            grade_data = _compute_grade_distribution(events)
            gg = f.require_group("grade_distribution")
            gg.create_dataset("grades",      data=grade_data["grades"])
            gg.create_dataset("counts",      data=grade_data["counts"])
            gg.create_dataset("grade_names", data=[n.encode() for n in grade_data["grade_names"]])

        # ── Metadata ───────────────────────────────────────────────────────────
        mg = f.require_group("meta")
        mg.attrs["seed_sigma"]   = seed_sigma
        mg.attrs["split_sigma"]  = float(gen.get("split_sigma", 3.0))
        mg.attrs["n_frames"]     = n_frames
        mg.attrs["n_events"]     = len(events)
        mg.attrs["frame_shape"]  = f"{noise_map.shape[0]}x{noise_map.shape[1]}"

    print("  ✓ saved.")


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
