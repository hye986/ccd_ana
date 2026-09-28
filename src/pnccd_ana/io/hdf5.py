"""
pnccd_ana.io.hdf5
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
# Calibration .npy save / load (Stage 1: offset calibration)
# ──────────────────────────────────────────────────────────────────────────────

def save_calibration_npy(
        out_dir:         str | Path,
        results:         dict,
        cm_map:          np.ndarray | None = None,
        bad_pixel_mask:  np.ndarray | None = None,
) -> None:
    """
    Save offset-calibration arrays as .npy files.

    Files written to *out_dir*::

        offset_median.npy        (H, W) float32   — median offset
        offset_sigclip.npy       (H, W) float32   — sigma-clip offset
        noise.npy                (H, W) float32   — per-pixel RMS
        n_clipped_map.npy        (H, W) float32   — frames clipped per pixel
        cm_map.npy               (n_frames, H, n_asics) float32 — CM correction
        cm_noise_<ASIC>.npy      (H,) float32     — per-ASIC CM noise
        bad_pixel_mask.npy       (H, W) bool      — True = bad pixel

    Parameters
    ----------
    out_dir        : output directory
    results        : results dict ("global" → per-pixel maps)
    cm_map         : (n_frames, H, n_asics) CM correction values, or None
    bad_pixel_mask : (H, W) bool bad pixel mask, or None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving .npy calibration to {out_dir}/")

    r = results.get("global", results)
    mapping = {
        "offset_median":  "offset_median.npy",
        "offset_sigclip": "offset_sigclip.npy",
        "noise":          "noise.npy",
        "n_clipped_map":  "n_clipped_map.npy",
    }
    for key, fname in mapping.items():
        if key in r and r[key] is not None:
            np.save(out_dir / fname, np.asarray(r[key], dtype=np.float32))
            print(f"  → {fname}")

    if cm_map is not None:
        np.save(out_dir / "cm_map.npy", np.asarray(cm_map, dtype=np.float32))
        print("  → cm_map.npy")

    cm_noise = r.get("cm_noise")
    if isinstance(cm_noise, dict):
        for name, arr in cm_noise.items():
            np.save(out_dir / f"cm_noise_{name}.npy",
                    np.asarray(arr, dtype=np.float32))
            print(f"  → cm_noise_{name}.npy")
    elif cm_noise is not None:
        np.save(out_dir / "cm_noise.npy", np.asarray(cm_noise, dtype=np.float32))
        print("  → cm_noise.npy")

    if bad_pixel_mask is not None:
        np.save(out_dir / "bad_pixel_mask.npy",
                np.asarray(bad_pixel_mask, dtype=bool))
        print("  → bad_pixel_mask.npy")


def load_calibration_npy(
        npy_dir: str | Path,
        prefer:  str = "sigclip",
) -> dict:
    """
    Load offset calibration from .npy files written by save_calibration_npy.

    Parameters
    ----------
    npy_dir : directory containing the .npy files
    prefer  : which offset method to load as the primary "offset" key
              ("sigclip" | "median")

    Returns
    -------
    dict with keys: offset, offset_type, noise, n_clipped_map (optional),
    cm_map (optional), cm_noise (optional; dict for per-ASIC),
    bad_pixel_mask (optional).
    """
    npy_dir = Path(npy_dir)
    alt = "median" if prefer == "sigclip" else "sigclip"
    d: dict = {}

    # Offset: preferred → alternative → any
    for name in (prefer, alt):
        fp = npy_dir / f"offset_{name}.npy"
        if fp.exists():
            d["offset"] = np.load(fp).astype(np.float32)
            d["offset_type"] = name
            print(f"  loaded {fp.name}  {d['offset'].shape}")
            break
    else:
        found = sorted(npy_dir.glob("offset_*.npy"))
        if found:
            d["offset"] = np.load(found[0]).astype(np.float32)
            d["offset_type"] = found[0].stem.split("offset_")[1]
            print(f"  Warning: preferred '{prefer}' not found, "
                  f"using '{found[0].name}'")
        else:
            raise RuntimeError(
                f"No offset_*.npy found in {npy_dir}.\n"
                f"Run  pnccd-offset  first.")

    # Required noise map
    fp = npy_dir / "noise.npy"
    if not fp.exists():
        raise RuntimeError(f"{fp.name} not found in {npy_dir}.")
    d["noise"] = np.load(fp).astype(np.float32)
    print(f"  loaded {fp.name}  {d['noise'].shape}")

    # Optional maps
    for key, fname in [("n_clipped_map",  "n_clipped_map.npy"),
                       ("cm_map",         "cm_map.npy"),
                       ("bad_pixel_mask", "bad_pixel_mask.npy")]:
        fp = npy_dir / fname
        if fp.exists():
            d[key] = np.load(fp)
            print(f"  loaded {fp.name}  {d[key].shape}")

    # CM noise: per-ASIC files, else a single file
    per_asic = {p.stem.split("cm_noise_")[1]: np.load(p).astype(np.float32)
                for p in sorted(npy_dir.glob("cm_noise_*.npy"))}
    if per_asic:
        d["cm_noise"] = per_asic
    elif (npy_dir / "cm_noise.npy").exists():
        d["cm_noise"] = np.load(npy_dir / "cm_noise.npy").astype(np.float32)

    return d
