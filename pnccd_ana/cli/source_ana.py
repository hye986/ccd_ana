"""
pnccd_ana.cli.source_ana
=========================
Command-line entry point for Fe-55 source analysis.

Usage
-----
  python -m pnccd_ana.cli.source_ana analysis.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import Config
from ..lib    import (find_events, resolve_asics,
                       ASIC_SLICES, ALL_ASICS, N_GRADES, EVENT_DTYPE)
from ..lib.common_mode import cm_correct_frame
from ..utils  import (get_io_module,
                       load_calibration_h5, load_calibration_npy,
                       save_events_h5,
                       plot_hitmap, plot_asic_hitmaps,
                       plot_spectrum, plot_asic_spectra,
                       plot_grade_distribution,
                       plot_raw_spectrum, plot_raw_spectrum_per_asic,
                       plot_spectrum_comparison)


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
                     asics: list[str] | None,
                     noise_scope: str = "auto") -> np.ndarray:
    """
    Assemble a full 1024×1024 noise map from calibration.

    If per-ASIC noise maps are available they are stitched in;
    otherwise the global map is used.  Regions outside the active
    ASICs get a huge noise value (1e9) so they never trigger.
    """
    noise_scope = (noise_scope or "auto").lower()
    noise = np.full((1024, 1024), 1e9, dtype=np.float32)

    def _fill_from_global() -> np.ndarray:
        if "global" not in cal or "noise" not in cal["global"]:
            raise RuntimeError("Global noise map requested but not found in calibration.")
        if not asics:
            return cal["global"]["noise"].copy()
        out = np.full((1024, 1024), 1e9, dtype=np.float32)
        for aname in asics:
            Y0, Y1, X0, X1 = ASIC_SLICES[aname]
            out[Y0:Y1+1, X0:X1+1] = cal["global"]["noise"][Y0:Y1+1, X0:X1+1]
        return out

    use_global = noise_scope == "global"
    if noise_scope == "auto" and asics and "global" in cal and all(a in cal for a in asics):
        global_vals = []
        asic_vals = []
        for aname in asics:
            Y0, Y1, X0, X1 = ASIC_SLICES[aname]
            global_vals.append(cal["global"]["noise"][Y0:Y1+1, X0:X1+1])
            asic_vals.append(cal[aname]["noise"])
        global_med = float(np.nanmedian(np.concatenate([a.ravel() for a in global_vals])))
        asic_med = float(np.nanmedian(np.concatenate([a.ravel() for a in asic_vals])))
        if global_med > 0 and asic_med < 0.5 * global_med:
            print(f"  ⚠  ASIC noise median ({asic_med:.1f} ADU) is much lower than "
                  f"global median ({global_med:.1f} ADU); using global noise slices "
                  "for event thresholds.")
            use_global = True

    if use_global:
        noise = _fill_from_global()
    elif asics and all(a in cal for a in asics):
        for aname in asics:
            Y0, Y1, X0, X1 = ASIC_SLICES[aname]
            noise[Y0:Y1+1, X0:X1+1] = cal[aname]["noise"]
    elif "global" in cal and "noise" in cal["global"]:
        noise = _fill_from_global()
    else:
        raise RuntimeError("No noise map found in calibration.")

    return noise


# ──────────────────────────────────────────────────────────────────────────────
# RAW sub-frame embedding
# ──────────────────────────────────────────────────────────────────────────────

def _embed_into_full_frame(
        data:  np.ndarray,
        asics: list[str],
) -> np.ndarray:
    """
    Embed a sub-frame RAW chunk into a zero-padded 1024×1024 canvas.

    When the RAW file records only a single ASIC (or a subset), each
    loaded chunk has shape (N, H, W) where H×W < 1024×1024.  All
    downstream code (_correct_frame, find_events, noise maps, hit maps)
    assumes 1024×1024 frames, so we place the sub-frame at the correct
    pixel position before passing it on.

    If the data is already 1024×1024 this function is a no-op.
    Raises ValueError if the sub-frame shape does not match the ASIC
    bounding box (catches mismatches between config and file early).
    """
    N, H, W = data.shape
    if H == 1024 and W == 1024:
        return data

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
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Per-chunk worker
# ──────────────────────────────────────────────────────────────────────────────

def _make_worker(cal: dict,
                 asics: list[str] | None,
                 noise_map: np.ndarray,
                 seed_sigma: float,
                 split_sigma: float,
                 reject_extra: bool,
                 search_mask: np.ndarray | None,
                 sample_buf: list | None = None,
                 sample_max: int = 50):
    """
    Return a closure suitable for process_frames_mt.

    If sample_buf is a list, up to sample_max corrected frames are appended
    to it (thread-safe via the list.append GIL) for use in raw spectrum plots.
    """
    import threading
    _lock = threading.Lock()

    def _worker(raw_chunk: np.ndarray,
                frame_indices: np.ndarray) -> np.ndarray:
        # Embed sub-frame RAW data into 1024×1024 if needed.
        # This is cheap (one zero allocation per chunk) and keeps all
        # downstream code (ASIC slicing, noise map, find_events) unchanged.
        if asics and (raw_chunk.shape[1] != 1024 or raw_chunk.shape[2] != 1024):
            raw_chunk = _embed_into_full_frame(raw_chunk, asics)

        chunk_events: list[np.ndarray] = []
        for frame in raw_chunk:
            corrected = _correct_frame(frame, cal, asics)

            # Collect sample frames for raw spectrum (cheap copy of one frame)
            if sample_buf is not None:
                with _lock:
                    if len(sample_buf) < sample_max:
                        sample_buf.append(corrected.copy())

            evts = find_events(corrected, noise_map,
                               search_mask=search_mask,
                               seed_sigma=seed_sigma,
                               split_sigma=split_sigma,
                               reject_extra=reject_extra)
            if len(evts):
                chunk_events.append(evts)
        if chunk_events:
            return np.concatenate(chunk_events)
        return np.empty(0, dtype=EVENT_DTYPE)
    return _worker


def _correct_frame(raw: np.ndarray,
                   cal: dict,
                   asics: list[str] | None) -> np.ndarray:
    """
    Offset subtract + CM correct one raw frame.

    If per-ASIC mode: each ASIC is corrected independently.
    Otherwise: global offset used on the whole frame.
    """
    corrected = np.zeros_like(raw, dtype=np.float32)

    if asics:
        for aname in asics:
            Y0, Y1, X0, X1 = ASIC_SLICES[aname]
            sub    = raw[Y0:Y1+1, X0:X1+1].astype(np.float32)
            offset = cal[aname]["offset"]
            corr, _ = cm_correct_frame(sub - offset)
            corrected[Y0:Y1+1, X0:X1+1] = corr
    else:
        full = raw.astype(np.float32) - cal["global"]["offset"]
        corrected, _ = cm_correct_frame(full)

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
    from glob import glob

    specs = [file_spec] if isinstance(file_spec, str) else list(file_spec)
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
    """Execute the source analysis pipeline from a Config object."""
    sc      = cfg.source_spectrum
    gen     = cfg.general
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = sc["source_run_file"]
    cal_src = sc["calibration_file"]
    if not h5_path:
        raise ValueError("[source_spectrum] source_run_file must be set.")
    if not cal_src:
        raise ValueError("[source_spectrum] calibration_file must be set.")

    asics           = cfg.asics_for("source_spectrum")
    seed_sigma      = float(sc.get("seed_sigma",  sc.get("threshold_sigma", 5.0)))
    split_sigma     = float(sc.get("split_sigma", 3.0))
    reject_extra    = bool(sc.get("reject_extra", False))
    prefer          = sc.get("prefer_offset", "sigclip")
    noise_scope     = sc.get("noise_scope", "auto")

    # ── Load calibration ──────────────────────────────────────────────────────
    print(f"\nLoading calibration from: {cal_src}")
    cal = _load_cal(cal_src, asics, prefer=prefer)

    # Fill per-ASIC from global slice if missing
    if asics:
        for aname in asics:
            if aname not in cal and "global" in cal:
                Y0, Y1, X0, X1 = ASIC_SLICES[aname]
                cal[aname] = {
                    "offset": cal["global"]["offset"][Y0:Y1+1, X0:X1+1],
                    "noise":  cal["global"]["noise"][Y0:Y1+1, X0:X1+1],
                }
                print(f"  ℹ  {aname}: using global calibration slice")

    noise_map = _build_noise_map(cal, asics, noise_scope=noise_scope)

    # ── Build search mask ─────────────────────────────────────────────────────
    search_mask: np.ndarray | None = None
    if asics:
        search_mask = np.zeros((1024, 1024), dtype=bool)
        for aname in asics:
            Y0, Y1, X0, X1 = ASIC_SLICES[aname]
            search_mask[Y0:Y1+1, X0:X1+1] = True

    # ── Resolve file list (single path, glob, or list) ────────────────────────
    run_files = _resolve_paths(h5_path)
    print(f"\nSource run file(s): {len(run_files)} file(s) matched")
    for p in run_files:
        print(f"  {p}")

    # ── Discover and process frames across all files ──────────────────────────
    io = get_io_module(sc["data_format"])
    raw_kwargs = {}
    if sc["data_format"] == "raw":
        if sc.get("raw_height"):
            raw_kwargs["height"] = sc["raw_height"]
        if sc.get("raw_width"):
            raw_kwargs["width"] = sc["raw_width"]

    # Sample buffer for raw spectrum plots (collect up to 200 corrected frames)
    # Shared across all input files — worker appends to it as frames are processed.
    sample_buf: list = []

    worker = _make_worker(cal, asics, noise_map,
                          seed_sigma, split_sigma, reject_extra,
                          search_mask,
                          sample_buf=sample_buf, sample_max=200)

    all_results: list[np.ndarray] = []
    remaining = sc["max_frames"]   # None = unlimited; decremented per file

    for fpath in run_files:
        if remaining is not None and remaining <= 0:
            break
        print(f"\nOpening source file: {fpath}")
        indices = io.get_frame_indices(fpath,
                                       complete_only=sc["complete_only"],
                                       max_frames=remaining,
                                       **raw_kwargs)
        print(f"Processing {len(indices)} frames  "
              f"(seed={seed_sigma}σ, split={split_sigma}σ,  "
              f"workers={sc['n_workers']}, chunk={sc['chunk_size']})")
        file_results = io.process_frames_mt(fpath, indices, worker,
                                            chunk_size=sc["chunk_size"],
                                            n_workers=sc["n_workers"],
                                            desc="source frames",
                                            **raw_kwargs)
        all_results.extend(file_results)
        if remaining is not None:
            remaining -= len(indices)

    results = all_results

    # Concatenate event arrays from all chunks
    non_empty = [r for r in results if len(r) > 0]
    events    = np.concatenate(non_empty) if non_empty else np.empty(0, dtype=EVENT_DTYPE)
    print(f"\nTotal events: {len(events):,}")

    # ── Diagnostic summary ────────────────────────────────────────────────────
    if len(events):
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
        # Grade 13 is deliberately centre-only, so diagnose that separately
        # from a real failure to sum recognized split patterns.
        n_identical = int((np.abs(events["adu_sum"] - events["adu_seed"]) < 0.1).sum())
        frac_identical = n_identical / len(events) * 100
        n_other = grade_counts.get(13, 0)
        frac_other = n_other / len(events) * 100
        split_mask = (events["grade"] > 0) & (events["grade"] < 13)
        n_split = int(split_mask.sum())
        n_split_summed = int((np.abs(events["adu_sum"][split_mask] -
                                     events["adu_seed"][split_mask]) >= 0.1).sum())
        if frac_other > 50:
            print(f"  ⚠  WARNING: {frac_other:.1f}% of events are grade 13 ('other')")
            print("     Grade-13 events are unrecognized 5×5 patterns and store centre ADU only.")
            print("     Use reject_extra: true for spectra, or raise split_sigma if random")
            print("     neighbour noise is creating extra above-split pixels.")
        elif n_split and n_split_summed < 0.8 * n_split:
            print(f"  ⚠  WARNING: only {n_split_summed}/{n_split} recognized split events "
                  "have adu_sum > adu_seed")
            print("     This would indicate a summing bug for grades 1-12.")
        elif frac_identical > 80 and grade_counts.get(0, 0) < len(events) * 0.8:
            print(f"  ⚠  WARNING: {frac_identical:.1f}% of events have adu_sum ≈ adu_seed")
            print("     Most accepted events are centre-only; inspect grade distribution.")
        elif n_split == 0:
            print("  ✓  No recognized split events in grades 1-12")
        else:
            print(f"  ✓  {n_split_summed}/{n_split} recognized split events have "
                  "adu_sum > adu_seed")

        # Check if bin range covers the peaks
        in_range = int(((events["adu_sum"] >= sc["adu_min"]) &
                        (events["adu_sum"] <= sc["adu_max"])).sum())
        if in_range < len(events) * 0.5:
            print(f"  ⚠  Only {in_range}/{len(events)} events within "
                  f"adu_min={sc['adu_min']:.0f}..adu_max={sc['adu_max']:.0f}")
            print(f"     Plots will auto-range but UPDATE adu_min/adu_max in config.")

    # ── Spectrum histograms ───────────────────────────────────────────────────
    bin_edges = np.linspace(sc["adu_min"], sc["adu_max"], sc["n_bins"] + 1)
    spectra: dict[int, np.ndarray] = {}
    for g in range(N_GRADES):
        m = events["grade"] == g
        spectra[g], _ = (np.histogram(events["adu_sum"][m], bins=bin_edges)
                         if m.any() else
                         (np.zeros(sc["n_bins"], dtype=np.int64), bin_edges))

    # ── 2-D maps ──────────────────────────────────────────────────────────────
    hit_count = np.zeros((1024, 1024), dtype=np.int32)
    hit_adu   = np.zeros((1024, 1024), dtype=np.float64)
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
    if sc["save_events"]:
        np.save(out_dir / "events.npy", events)

    events_h5_path = sc.get("save_events_to_file")
    if events_h5_path:
        save_events_h5(
            events_h5_path,
            events, spectra, bin_edges, hit_count, mean_adu,
            metadata={**gen.get("metadata", {}),
                      "source_file":    str(h5_path),
                      "calibration":    str(cal_src),
                      "threshold_sigma": seed_sigma,
                      "n_frames":       int(len(indices))},
        )

    # Also save maps as .npy for quick access
    np.save(out_dir / "hit_count.npy",   hit_count)
    np.save(out_dir / "mean_adu_map.npy", mean_adu)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    if gen["save_frame_plots"]:
        plot_hitmap(hit_count, mean_adu, out_dir, asics)
        plot_spectrum(spectra, bin_edges, out_dir, events=events)
        if len(events):
            plot_grade_distribution(events, out_dir)
            plot_spectrum_comparison(
                events=events,
                corrected_frames=np.zeros((1,1,1), dtype=np.float32),  # unused now
                noise_map=noise_map,
                bin_edges=bin_edges,
                out_dir=out_dir,
                seed_sigma=seed_sigma,
                title_suffix="global",
            )

        # Raw pixel-level spectrum (needs sample frames)
        if sample_buf:
            sample_arr = np.stack(sample_buf, axis=0)
            plot_raw_spectrum(
                corrected_frames=sample_arr,
                noise_map=noise_map,
                offset_map=(cal.get("global") or next(iter(cal.values())))["offset"],
                raw_frames=None,
                bin_edges=bin_edges,
                out_dir=out_dir,
                seed_sigma=seed_sigma,
                asic_mask=search_mask,
                title_suffix="global",
            )

    if asics and gen["save_asic_plots"]:
        plot_asic_hitmaps(hit_count, asics, out_dir)
        if len(events):
            plot_asic_spectra(events, bin_edges, asics, out_dir)
        if sample_buf:
            sample_arr = np.stack(sample_buf, axis=0)
            plot_raw_spectrum_per_asic(
                corrected_frames=sample_arr,
                noise_map=noise_map,
                raw_frames=None,
                bin_edges=bin_edges,
                asics=asics,
                out_dir=out_dir,
                seed_sigma=seed_sigma,
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
