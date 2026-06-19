"""
pnccd_ana.utils.io_h5
=====================
HDF5 I/O utilities.

Reading strategy for large files (100 GB+)
─────────────────────────────────────────────
  Frame indices are split into chunks.  Each worker thread opens its own
  HDF5 file handle (safe with libhdf5 ≥ 1.8 in SWMR or read-only mode).
  Per-thread handles avoid the GIL bottleneck and serialised reads.

  Usage pattern::

      for chunk_idx, frame_slice in iter_chunks(h5_path, chunk_size):
          raw_frames = read_chunk(h5_path, frame_slice)
          ...

  Or use ThreadPoolExecutor with read_chunk as the worker function.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterator

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Frame index discovery
# ──────────────────────────────────────────────────────────────────────────────

def get_frame_indices(
        h5_path:       str,
        complete_only: bool      = True,
        max_frames:    int | None = None,
) -> np.ndarray:
    """
    Return the indices of usable frames in the HDF5 file.

    Parameters
    ----------
    h5_path       : path to HDF5 file
    complete_only : skip frames where frames/complete != 1
    max_frames    : cap on number of indices returned

    Returns
    -------
    indices : int array of frame indices
    """
    with h5py.File(h5_path, "r") as f:
        n_total  = int(f["frames"].attrs.get("n_frames",
                                             f["frames/data"].shape[0]))
        complete = f["frames/complete"][:n_total]
        shape    = f["frames/data"].shape
    height, width = shape[1], shape[2]

    idx = np.where(complete == 1)[0] if complete_only else np.arange(n_total)
    if max_frames is not None:
        idx = idx[:max_frames]
    if len(idx) == 0:
        raise ValueError("No usable frames found in file.")

    print(f"  {len(idx)} usable frames  ({height} × {width} pixels)")
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Chunk iteration helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_chunks(indices: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """Split *indices* into a list of sub-arrays of length ≤ chunk_size."""
    return [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]


def read_chunk(h5_path: str, frame_indices: np.ndarray) -> np.ndarray:
    """
    Read a chunk of frames from HDF5 into float32.

    Strategy for speed on large files:
      - Sort the requested indices and find contiguous runs.
      - Each contiguous run is read as a single slice (one HDF5 call),
        which is far faster than N individual reads because it avoids
        per-call HDF5 overhead and allows the library to use its read-ahead.
      - Each thread opens its own file handle (thread-safe with libhdf5 ≥ 1.8).

    Returns
    -------
    data : float32 (len(frame_indices), Y, X)  in the original index order
    """
    with h5py.File(h5_path, "r", swmr=True) as f:
        dset = f["frames/data"]
        H, W = dset.shape[1], dset.shape[2]
        n    = len(frame_indices)
        out  = np.empty((n, H, W), dtype=np.float32)

        # Sort to find contiguous runs; keep a mapping back to original order
        sort_order = np.argsort(frame_indices)
        sorted_idx = frame_indices[sort_order]

        # Read contiguous runs in one slice each
        pos = 0
        while pos < n:
            run_start = pos
            run_end   = pos + 1
            while (run_end < n and
                   sorted_idx[run_end] == sorted_idx[run_end - 1] + 1):
                run_end += 1
            # Single slice read for this run
            slc = dset[int(sorted_idx[run_start]):int(sorted_idx[run_end-1])+1]
            for k in range(run_end - run_start):
                out[sort_order[run_start + k]] = slc[k].astype(np.float32)
            pos = run_end

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Multithreaded frame processor
# ──────────────────────────────────────────────────────────────────────────────

def process_frames_mt(
        h5_path:    str,
        indices:    np.ndarray,
        worker_fn:  Callable[[np.ndarray, np.ndarray], object],
        chunk_size: int = 64,
        n_workers:  int = 8,
        desc:       str = "frames",
) -> list:
    """
    Process all frames in parallel using a thread pool.

    Each worker:
      1. Opens its own HDF5 handle.
      2. Reads the chunk of raw frames.
      3. Calls worker_fn(raw_frames, chunk_frame_indices) → result.

    The list of results (one per chunk, in completion order) is returned.
    Order within the list is non-deterministic; if ordering matters, include
    the frame indices in the result and sort afterwards.

    Parameters
    ----------
    h5_path    : source HDF5 file
    indices    : frame indices to process (from get_frame_indices)
    worker_fn  : callable(raw_chunk: float32 array, frame_indices: int array) → any
    chunk_size : frames per chunk
    n_workers  : threads
    desc       : label for progress output

    Returns
    -------
    results : list of worker_fn return values (one per chunk)
    """
    import time

    chunks   = make_chunks(indices, chunk_size)
    n_chunks = len(chunks)
    print(f"  {n_chunks} chunks × ≤{chunk_size} {desc}  ({n_workers} workers)")

    results  = []
    lock     = threading.Lock()
    done     = [0]
    t0       = time.time()

    def _task(chunk_idx: np.ndarray) -> object:
        raw = read_chunk(h5_path, chunk_idx)
        return worker_fn(raw, chunk_idx)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_task, ch): i for i, ch in enumerate(chunks)}
        for fut in as_completed(futures):
            res = fut.result()   # re-raises exceptions from worker
            with lock:
                results.append(res)
                done[0] += 1
                n_done = done[0]
            if n_done % max(1, n_chunks // 20) == 0 or n_done == n_chunks:
                elapsed = time.time() - t0
                fps     = n_done * chunk_size / max(elapsed, 1e-3)
                eta     = (n_chunks - n_done) * chunk_size / max(fps, 1e-3)
                print(f"  [{n_done:4d}/{n_chunks}]  {fps:.1f} frames/s  "
                      f"ETA {eta:.0f}s", flush=True)

    print(f"  Done in {time.time()-t0:.1f}s")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Calibration HDF5 save / load
# ──────────────────────────────────────────────────────────────────────────────


def _write_cal_group(grp: h5py.Group, r: dict) -> None:
    """Write calibration arrays into an open HDF5 group."""
    og = grp.require_group("offset")
    ng = grp.require_group("noise")
    for key, dset_name, parent in [
        ("offset_median",  "median",              og),
        ("offset_sigclip", "sigmaclip",           og),
        ("noise",          "pixel_rms",           ng),
        ("n_clipped_map",  "n_clipped_per_pixel", ng),
    ]:
        if key in r:
            parent.create_dataset(dset_name, data=r[key], compression="gzip")

    # CM noise: either a 1-D array (legacy) or a dict of per-ASIC arrays
    if "cm_noise" in r:
        cm = r["cm_noise"]
        if isinstance(cm, dict):
            # Per-ASIC: save each ASIC as noise/cm_rms_<NAME>
            for asic_name, arr in cm.items():
                ng.create_dataset(f"cm_rms_{asic_name}", data=arr,
                                  compression="gzip")
        else:
            # Legacy 1-D array
            ng.create_dataset("cm_rms", data=cm, compression="gzip")


def save_calibration_h5(
        out_path: str | Path,
        results:  dict,
        n_sigma:  float,
        n_frames: int,
        methods:  list[str],
        metadata: dict | None = None,
) -> None:
    """
    Save dark-frame calibration products to HDF5.

    File structure::

        /global/offset/median
        /global/offset/sigmaclip
        /global/noise/pixel_rms
        /global/noise/cm_rms
        /global/noise/n_clipped_per_pixel
        /asics/<NAME>/offset/...
        /asics/<NAME>/noise/...
        /meta/<key>          ← arbitrary metadata scalars/strings

    Parameters
    ----------
    out_path : output .h5 file path
    results  : dict with keys "global" and optionally "asics"
    n_sigma  : sigma-clip threshold used
    n_frames : number of dark frames processed
    methods  : list of methods used, e.g. ["median", "sigclip"]
    metadata : optional dict of extra key→value pairs stored under /meta
    """
    print(f"\nSaving calibration HDF5: {out_path}")
    with h5py.File(out_path, "w") as f:
        f.attrs["description"]       = "pnCCD dark-frame calibration"
        f.attrs["n_dark_frames"]     = n_frames
        f.attrs["sigma_clip_nsigma"] = n_sigma
        f.attrs["methods"]           = ",".join(methods)

        if "global" in results:
            _write_cal_group(f.require_group("global"), results["global"])
        if "asics" in results:
            ag = f.require_group("asics")
            for aname, adata in results["asics"].items():
                _write_cal_group(ag.require_group(aname), adata)

        if metadata:
            mg = f.require_group("meta")
            for k, v in metadata.items():
                mg.attrs[k] = v

    print("  ✓ saved.")


def load_calibration_h5(
        h5_path:   str | Path,
        asics:     list[str] | None = None,
        prefer:    str = "sigclip",   # "sigclip" or "median"
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load calibration maps from a dark_calibration.h5 file.

    Parameters
    ----------
    h5_path : calibration HDF5 file produced by save_calibration_h5
    asics   : ASIC names to load individually (None → global only)
    prefer  : which offset to load as the primary "offset" key

    Returns
    -------
    cal["global"]["offset"]  float32 (1024,1024)
    cal["global"]["noise"]   float32 (1024,1024)
    cal["H0"]["offset"]      float32 (512,512)   — if requested
    ...
    """
    # Map user-facing names to the HDF5 dataset names used by save_calibration_h5
    _H5_NAME = {"sigclip": "sigmaclip", "median": "median"}
    prefer_h5 = _H5_NAME.get(prefer, prefer)
    alt        = "median" if prefer == "sigclip" else "sigclip"
    alt_h5     = _H5_NAME.get(alt, alt)
    cal: dict[str, dict] = {}

    with h5py.File(h5_path, "r") as f:

        def _list_offsets(grp_path: str) -> list[str]:
            grp = f.get(grp_path)
            if grp is None:
                return []
            og = grp.get("offset")
            return list(og.keys()) if og is not None else []

        available_global = _list_offsets("global")
        if available_global:
            print(f"  Available offsets in file (global): {available_global}")

        def _load(grp_path: str, name: str) -> dict | None:
            grp = f.get(grp_path)
            if grp is None:
                return None
            d: dict = {}
            offset_found = False

            # Try preferred h5 name, then alternative, then any available
            for h5key, label in [(prefer_h5, prefer), (alt_h5, alt)]:
                dset = grp.get(f"offset/{h5key}")
                if dset is not None:
                    d["offset"]      = dset[:].astype(np.float32)
                    d["offset_type"] = label   # store as user-facing name
                    offset_found = True
                    break

            if not offset_found:
                og = grp.get("offset")
                if og is not None:
                    for k in og.keys():
                        d["offset"]      = og[k][:].astype(np.float32)
                        d["offset_type"] = k
                        offset_found = True
                        print(f"  Warning: [{name}] neither '{prefer}' nor '{alt}' found; "
                              f"falling back to 'offset/{k}'")
                        break

            if not offset_found:
                print(f"  ✗  [{name}] no offset dataset found under '{grp_path}/offset'")
                return None

            # Noise: required
            ng = grp.get("noise")
            if ng is None or ng.get("pixel_rms") is None:
                print(f"  ✗  [{name}] noise/pixel_rms not found in '{grp_path}'")
                return None
            d["noise"] = ng["pixel_rms"][:].astype(np.float32)

            # Optional extras
            # n_clipped_map: single dataset
            dset = grp.get("noise/n_clipped_per_pixel")
            if dset is not None:
                d["n_clipped_map"] = dset[:].astype(np.float32)

            # CM noise: may be legacy 1-D array or per-ASIC datasets
            ng = grp.get("noise")
            if ng is not None:
                # Check for per-ASIC cm_rms_<NAME> datasets first
                asic_cm = {k[7:]: ng[k][:].astype(np.float32)
                           for k in ng.keys() if k.startswith("cm_rms_")}
                if asic_cm:
                    d["cm_noise"] = asic_cm          # dict of per-ASIC arrays
                elif "cm_rms" in ng:
                    d["cm_noise"] = ng["cm_rms"][:].astype(np.float32)  # legacy

            return d

        g = _load("global", "global")
        if g:
            cal["global"] = g
            print(f"  global  offset({g['offset_type']}) {g['offset'].shape}  "
                  f"noise {g['noise'].shape}")
        else:
            print("  Warning: global calibration not loaded — check HDF5 structure")

        for aname in (asics or []):
            g = _load(f"asics/{aname}", aname)
            if g:
                cal[aname] = g
                print(f"  {aname}  offset({g['offset_type']}) {g['offset'].shape}  "
                      f"noise {g['noise'].shape}")
            else:
                print(f"  Warning: {aname} calibration not loaded")

    if not cal:
        raise RuntimeError(
            f"No calibration data could be loaded from {h5_path}.\n"
            f"Run dark_frame_ana first, or check that the HDF5 file contains "
            f"'global/offset/median' or 'global/offset/sigmaclip'.")

    return cal


# ──────────────────────────────────────────────────────────────────────────────
# Event HDF5 save / load
# ──────────────────────────────────────────────────────────────────────────────

def save_events_h5(
        out_path:  str | Path,
        events:    np.ndarray,
        spectra:   dict[int, np.ndarray],
        bin_edges: np.ndarray,
        hit_count: np.ndarray,
        mean_adu:  np.ndarray,
        metadata:  dict | None = None,
) -> None:
    """
    Save source-analysis results to HDF5.

    Structure::

        /events/Y, X, grade, adu_sum    (structured array columns)
        /spectra/grade<NN>              (histogram counts per grade)
        /spectra/bin_edges
        /maps/hit_count
        /maps/mean_adu
        /meta/<key>                     (optional metadata)
    """
    print(f"\nSaving events HDF5: {out_path}")
    with h5py.File(out_path, "w") as f:
        if len(events):
            eg = f.require_group("events")
            for field in events.dtype.names:
                eg.create_dataset(field, data=events[field], compression="gzip")

        sg = f.require_group("spectra")
        sg.create_dataset("bin_edges", data=bin_edges)
        for g, counts in spectra.items():
            sg.create_dataset(f"grade{g:02d}", data=counts, compression="gzip")

        mg = f.require_group("maps")
        mg.create_dataset("hit_count", data=hit_count, compression="gzip")
        mg.create_dataset("mean_adu",  data=mean_adu,  compression="gzip")

        if metadata:
            mm = f.require_group("meta")
            for k, v in metadata.items():
                mm.attrs[k] = v

    print("  ✓ saved.")


def load_events_h5(
        h5_path: str | Path,
) -> dict:
    """
    Load source-analysis results from HDF5 (produced by save_events_h5).

    Returns dict with keys: events, spectra, bin_edges, hit_count, mean_adu, meta.
    Suitable as input for downstream gain/CTI calibration steps.
    """
    from ..lib.pattern_recognition import EVENT_DTYPE

    out: dict = {}
    with h5py.File(h5_path, "r") as f:
        # Events
        eg = f.get("events")
        if eg is not None:
            n = len(eg["Y"])
            arr = np.empty(n, dtype=EVENT_DTYPE)
            for field in EVENT_DTYPE.names:
                arr[field] = eg[field][:]
            out["events"] = arr
        else:
            out["events"] = np.empty(0, dtype=EVENT_DTYPE)

        # Spectra
        sg = f["spectra"]
        out["bin_edges"] = sg["bin_edges"][:]
        out["spectra"] = {}
        for key in sg:
            if key.startswith("grade"):
                g = int(key[5:])
                out["spectra"][g] = sg[key][:]

        # Maps
        mg = f["maps"]
        out["hit_count"] = mg["hit_count"][:]
        out["mean_adu"]  = mg["mean_adu"][:]

        # Metadata
        meta = f.get("meta")
        out["meta"] = dict(meta.attrs) if meta is not None else {}

    print(f"  Loaded {len(out['events']):,} events from {h5_path}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# .npy save / load
# ──────────────────────────────────────────────────────────────────────────────

def save_calibration_npy(out_dir: str | Path, results: dict) -> None:
    """Save calibration arrays as <scope>_<quantity>.npy files."""
    out_dir = Path(out_dir)
    mapping = {
        "offset_median":  "offset_median",
        "offset_sigclip": "offset_sigclip",
        "noise":          "noise",
        "cm_noise":       "cm_noise",
        "n_clipped_map":  "n_clipped_map",
    }
    print(f"\nSaving .npy calibration to {out_dir}/")

    def _save(scope: str, r: dict) -> None:
        for key, suffix in mapping.items():
            if key in r:
                p = out_dir / f"{scope}_{suffix}.npy"
                np.save(p, r[key])
                print(f"  → {p.name}")

    if "global" in results:
        _save("global", results["global"])
    if "asics" in results:
        for aname, adata in results["asics"].items():
            _save(aname, adata)


def load_calibration_npy(
        npy_dir:   str | Path,
        asics:     list[str] | None = None,
        prefer:    str = "sigclip",
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load calibration from .npy files in *npy_dir*.

    Files are expected as  <scope>_offset_sigclip.npy,  <scope>_noise.npy, etc.
    Falls back gracefully when the preferred offset variant is absent.
    """
    npy_dir = Path(npy_dir)
    alt     = "median" if prefer == "sigclip" else "sigclip"
    cal: dict[str, dict] = {}

    def _load(scope: str) -> dict | None:
        d: dict = {}

        # Offset: try preferred → alternative → any matching file
        offset_found = False
        for fname in [f"{scope}_offset_{prefer}.npy",
                      f"{scope}_offset_{alt}.npy"]:
            fp = npy_dir / fname
            if fp.exists():
                d["offset"]      = np.load(fp).astype(np.float32)
                d["offset_type"] = fname.split("_offset_")[1].replace(".npy", "")
                print(f"  loaded {fp.name}  {d['offset'].shape}")
                offset_found = True
                break
        if not offset_found:
            found = sorted(npy_dir.glob(f"{scope}_offset_*.npy"))
            if found:
                d["offset"]      = np.load(found[0]).astype(np.float32)
                d["offset_type"] = found[0].stem.split("_offset_")[1]
                print(f"  Warning: [{scope}] preferred='{prefer}' not found, "
                      f"using '{found[0].name}'")
                offset_found = True
        if not offset_found:
            print(f"  ✗  [{scope}] no offset .npy file found in {npy_dir}")
            return None

        # Noise: required
        fp = npy_dir / f"{scope}_noise.npy"
        if not fp.exists():
            print(f"  ✗  [{scope}] {fp.name} not found in {npy_dir}")
            return None
        d["noise"] = np.load(fp).astype(np.float32)
        print(f"  loaded {fp.name}  {d['noise'].shape}")

        # Optional extras
        for key, fname in [("cm_noise",      f"{scope}_cm_noise.npy"),
                            ("n_clipped_map", f"{scope}_n_clipped_map.npy")]:
            fp = npy_dir / fname
            if fp.exists():
                d[key] = np.load(fp).astype(np.float32)
        return d

    g = _load("global")
    if g:
        cal["global"] = g

    for aname in (asics or []):
        g = _load(aname)
        if g:
            cal[aname] = g
        else:
            print(f"  Warning: {aname} calibration .npy files not fully available")

    if not cal:
        raise RuntimeError(
            f"No calibration data loaded from {npy_dir}.\n"
            f"Run dark_frame_ana first.")
    return cal


# ──────────────────────────────────────────────────────────────────────────────
# Results HDF5 load functions
# ──────────────────────────────────────────────────────────────────────────────

def load_offset_results_h5(path: str | Path) -> dict:
    """
    Load plot-backing data from offset_results.h5.

    Returns dict with keys: offsets, noise, bad_pixels, meta.
    Each contains nested arrays/dicts matching the HDF5 structure.
    """
    path = Path(path)
    out: dict = {}
    with h5py.File(path, "r") as f:
        # Offsets
        offsets: dict = {}
        for method in ["median", "sigmaclip", "diff"]:
            if method in f["offsets"]:
                g = f[f"offsets/{method}"]
                offsets[method] = {
                    "map":          g["map"][:],
                    "hist_edges":   g["hist_edges"][:],
                    "hist_counts":  g["hist_counts"][:],
                }
        out["offsets"] = offsets

        # Noise
        ng = f["noise"]
        out["noise"] = {
            "map":          ng["map"][:],
            "hist_edges":   ng["hist_edges"][:],
            "hist_counts":  ng["hist_counts"][:],
        }
        if "cm_noise" in ng:
            cm: dict = {}
            for name in ng["cm_noise"]:
                cm[name] = ng[f"cm_noise/{name}"][:]
            out["noise"]["cm_noise"] = cm
        if "n_clipped/map" in ng:
            out["noise"]["n_clipped_map"] = ng["n_clipped/map"][:]

        # Bad pixels
        if "bad_pixels" in f:
            bg = f["bad_pixels"]
            out["bad_pixels"] = {
                "mask":              bg["mask"][:],
                "noise_hist_edges":  bg["noise_hist_edges"][:],
                "noise_hist_counts": bg["noise_hist_counts"][:],
                "thresholds": {
                    "median_noise":   float(bg["thresholds/median_noise"][()]),
                    "hot_threshold":  float(bg["thresholds/hot_threshold"][()]),
                    "cold_threshold": float(bg["thresholds/cold_threshold"][()]),
                },
            }
            if "per_asic/n_bad" in bg:
                out["bad_pixels"]["per_asic"] = {
                    "n_bad":       bg["per_asic/n_bad"][:],
                    "asic_labels": [l.decode() for l in bg["per_asic/asic_labels"][:]],
                }
            if "category_map" in bg:
                out["bad_pixels"]["category_map"] = bg["category_map"][:]

        # Metadata
        if "meta" in f:
            out["meta"] = dict(f["meta"].attrs)

    print(f"  Loaded dark results from {path}")
    return out


def load_event_rec_results_h5(path: str | Path) -> dict:
    """
    Load plot-backing data from event_rec_results.h5.

    Returns dict with keys: raw_spectrum, grade_distribution, meta.
    """
    path = Path(path)
    out: dict = {}
    with h5py.File(path, "r") as f:
        # Raw spectrum
        if "raw_spectrum" in f:
            rg = f["raw_spectrum"]
            out["raw_spectrum"] = {
                "bin_edges":          rg["bin_edges"][:],
                "all_pixels":        rg["all_pixels"][:],
                "positive_pixels":    rg["positive_pixels"][:],
                "above_seed":         rg["above_seed"][:],
                "seed_threshold_adu": float(rg["seed_threshold_adu"][()]),
                "seed_sigma":         float(rg["seed_sigma"][()]),
                "median_noise_adu":   float(rg["median_noise_adu"][()]),
            }

        # Grade distribution
        if "grade_distribution" in f:
            gg = f["grade_distribution"]
            out["grade_distribution"] = {
                "grades":      gg["grades"][:],
                "counts":      gg["counts"][:],
                "grade_names": [n.decode() for n in gg["grade_names"][:]],
            }

        # Metadata
        if "meta" in f:
            out["meta"] = dict(f["meta"].attrs)

    print(f"  Loaded source results from {path}")
    return out


def load_energy_cal_results_h5(path: str | Path) -> dict:
    """
    Load plot-backing data from energy_cal_results.h5.

    Returns dict with keys: phase1_rough_gain, phase3_cti, phase4_column_gain,
    pixel_gain_map, cti_per_col, final_spectrum, meta.
    """
    path = Path(path)
    out: dict = {}
    with h5py.File(path, "r") as f:
        # Phase 1: Rough gain
        if "phase1_rough_gain" in f:
            p1: dict = {}
            for parity in ["even", "odd"]:
                if parity in f["phase1_rough_gain"]:
                    pg = f[f"phase1_rough_gain/{parity}"]
                    p1[parity] = {
                        "hist_edges":  pg["hist_edges"][:],
                        "hist_counts": pg["hist_counts"][:],
                        "fit": {
                            "peak_adu":   float(pg["fit/peak_adu"][()]),
                            "sigma_adu":  float(pg["fit/sigma_adu"][()]),
                            "amplitude":  float(pg["fit/amplitude"][()]),
                            "success":    bool(pg["fit/success"][()]),
                            "n_events":   int(pg["fit/n_events"][()]),
                        },
                    }
            if "window" in f["phase1_rough_gain"]:
                p1["window"] = {
                    "lo_adu": float(f["phase1_rough_gain/window/lo_adu"][()]),
                    "hi_adu": float(f["phase1_rough_gain/window/hi_adu"][()]),
                }
            out["phase1_rough_gain"] = p1

        # Phase 3: CTI
        if "phase3_cti" in f:
            p3: dict = {}
            for key in ["before", "after"]:
                if key in f["phase3_cti"]:
                    cg = f[f"phase3_cti/{key}"]
                    p3[key] = {
                        "row_bins":      cg["row_bins"][:],
                        "peak_per_bin":  cg["peak_per_bin"][:],
                        "peak_success":  cg["peak_success"][:],
                    }
            out["phase3_cti"] = p3

        # Phase 4: Column gain
        if "phase4_column_gain" in f:
            p4: dict = {}
            if "f_col_hist/bin_edges" in f["phase4_column_gain"]:
                hg = f["phase4_column_gain/f_col_hist"]
                p4["f_col_hist"] = {
                    "bin_edges": hg["bin_edges"][:],
                    "counts":    hg["counts"][:],
                }
            out["phase4_column_gain"] = p4

        # Pixel gain map
        if "pixel_gain_map" in f:
            pgm = f["pixel_gain_map"]
            out["pixel_gain_map"] = {
                "g_eff":  pgm["g_eff"][:],
                "parity": pgm["parity"][:],
            }
            if "g_eff_hist/bin_edges" in pgm:
                hg = pgm["g_eff_hist"]
                out["pixel_gain_map"]["g_eff_hist"] = {
                    "bin_edges":   hg["bin_edges"][:],
                    "counts_even": hg["counts_even"][:],
                    "counts_odd":  hg["counts_odd"][:],
                }

        # CTI per column
        if "cti_per_col" in f:
            cpc = f["cti_per_col"]
            out["cti_per_col"] = {
                "col_bin_size": int(cpc["col_bin_size"][()]),
            }

        # Final spectrum
        if "final_spectrum" in f:
            fsg = f["final_spectrum"]
            fs: dict = {
                "bin_edges": fsg["bin_edges"][:],
            }
            if "all_grades/counts" in fsg:
                fs["all_grades"] = {"counts": fsg["all_grades/counts"][:]}
            if "per_group" in fsg:
                groups: dict = {}
                for grp_name in fsg["per_group"]:
                    gg = fsg[f"per_group/{grp_name}"]
                    groups[grp_name] = {
                        "counts":    gg["counts"][:],
                        "grade_ids": gg["grade_ids"][:],
                    }
                fs["per_group"] = groups
            if "kalpha_fit/peak_ev" in fsg:
                kf = fsg["kalpha_fit"]
                fs["kalpha_fit"] = {
                    "peak_ev":         float(kf["peak_ev"][()]),
                    "sigma_ev":        float(kf["sigma_ev"][()]),
                    "amplitude":       float(kf["amplitude"][()]),
                    "fwhm_ev":         float(kf["fwhm_ev"][()]),
                    "resolution_pct":  float(kf["resolution_pct"][()]),
                    "n_events":        int(kf["n_events"][()]),
                    "success":         bool(kf["success"][()]),
                    "fit_window_lo":   float(kf["fit_window_lo"][()]),
                    "fit_window_hi":   float(kf["fit_window_hi"][()]),
                }
            out["final_spectrum"] = fs

        # Metadata
        if "meta" in f:
            out["meta"] = dict(f["meta"].attrs)

    print(f"  Loaded gain results from {path}")
    return out
