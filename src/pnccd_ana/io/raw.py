"""
pnccd_ana.io.raw
======================
RAW-format I/O utilities.

RAW format description
──────────────────────
  File layout (produced by udp_recorder_hdf5.py):

    [8-byte header: b'16BU0000']
    For each frame:
      [0xFFFF marker (2B)] [ADC counter (4B)] [W × uint16 pixels]   ← first row
      [0xFFFE marker (2B)] [ADC counter (4B)] [W × uint16 pixels]   ← remaining rows
      ...

  Each record encodes one readout ROW (W pixels per row).
  The number of records per frame equals the frame height H.
  Geometry (H, W) is auto-detected from the marker pattern; pass height
  explicitly for non-standard or ROI sizes.

  Supported frame sizes for single-hybrid:
    - 512×512  (H=512 rows, W=512 pixels per row)
    - 1024×512 (H=1024 rows, W=512 pixels per row)
    - Other heights auto-detected (128, 256, 512, 1024, 2048, 4096)

  Array layout after reading: data[frame, Y, X]
    - Y = row index = 0..H-1 (first row in file = Y=0 = bottom with origin="lower")
    - X = column index = 0..W-1

RAW files carry no per-frame metadata (no frame32 counter, no timestamp,
no completeness flag).  All frames are therefore treated as complete.
complete_only and max_frames are still honoured: complete_only has no
effect (every frame is considered complete), and max_frames caps the count.

Reading strategy
────────────────
  The entire pixel array is memory-mapped via np.memmap on first access
  inside RawFrameStore, so random-access reads for arbitrary chunks are
  O(1) seeks with no repeated file-open overhead.  The memmap is shared
  across threads (read-only), so process_frames_mt works without any
  per-thread file-open logic.

Metadata caching
───────────────
  Geometry and frame metadata are cached in .metadata/ subfolder for faster
  subsequent access. Cache is invalidated if source file is modified.

  Usage pattern::

      indices = get_frame_indices(raw_path, max_frames=1000)
      for chunk in make_chunks(indices, chunk_size=64):
          frames = read_chunk(raw_path, chunk)
          ...

  Or multithreaded::

      results = process_frames_mt(raw_path, indices, worker_fn)
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# RAW format constants  (must match udp_recorder_hdf5.py)
# ──────────────────────────────────────────────────────────────────────────────

_HEADER_MAGIC  = b"16BU0000"
_FRAME_MARKER  = 0xFFFF
_LINE_MARKER   = 0xFFFE
_COMMON_HEIGHTS = (128, 256, 512, 1024, 2048, 4096)
_HEADER_SIZE   = 8            # bytes in file header


# ──────────────────────────────────────────────────────────────────────────────
# Metadata caching
# ──────────────────────────────────────────────────────────────────────────────

_METADATA_DIR = ".metadata"


def _get_metadata_paths(raw_path: str | Path) -> tuple[Path, Path]:
    """Get paths to metadata cache files (JSON + numpy offsets)."""
    raw_path = Path(raw_path).resolve()
    meta_dir = raw_path.parent / _METADATA_DIR
    json_path = meta_dir / f"{raw_path.name}.json"
    offsets_path = meta_dir / f"{raw_path.name}.frame_offsets.npy"
    return json_path, offsets_path


def _load_metadata(raw_path: str | Path,
                   height: int | None = None,
                   width:  int | None = None) -> dict | None:
    """
    Load cached metadata if valid and consistent with current file.
    
    The cache is validated against the actual file to ensure geometry matches.
    Returns None if cache is missing, stale, inconsistent, or fails validation.
    """
    raw_path = Path(raw_path).resolve()
    json_path, offsets_path = _get_metadata_paths(raw_path)
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    
    # Verify file hasn't changed (fast stat check)
    try:
        stat = raw_path.stat()
        if stat.st_mtime != meta.get("file_mtime"):
            return None
        if stat.st_size != meta.get("file_size"):
            return None
    except OSError:
        return None
    
    # Verify geometry constraints match if explicitly specified
    if height is not None and meta["geometry"]["height"] != height:
        return None
    if width is not None and meta["geometry"]["width"] != width:
        return None
    
    # Validate cached record_size against actual file size
    file_size = stat.st_size
    header_size = meta.get("frame0_offset", _HEADER_SIZE)
    record_size = meta.get("record_size", 0)
    cached_height = meta["geometry"]["height"]
    cached_n_frames = meta["geometry"]["n_frames"]
    
    total_records = cached_n_frames * cached_height
    expected_file_size = header_size + (total_records * record_size)
    
    if abs(expected_file_size - file_size) > record_size:
        return None
    
    # Load pre-computed frame record offsets if available (fast numpy load)
    if offsets_path.exists():
        try:
            meta["_frame_record_offsets"] = np.load(offsets_path, mmap_mode="r")
        except Exception:
            meta["_frame_record_offsets"] = None
    else:
        meta["_frame_record_offsets"] = None
    
    return meta


def _save_metadata(raw_path: str | Path, geom: dict) -> None:
    """Save geometry metadata and pre-computed frame offsets to cache files."""
    raw_path = Path(raw_path).resolve()
    
    meta_dir = raw_path.parent / _METADATA_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    json_path, offsets_path = _get_metadata_paths(raw_path)
    stat = raw_path.stat()
    
    H = geom["height"]
    n_frames = geom["n_frames"]
    frame_size = geom["record_size"] * H
    header_size = _HEADER_SIZE
    
    # Pre-compute frame record offsets: [0, H, 2H, 3H, ...]
    frame_record_offsets = np.arange(n_frames, dtype=np.int64) * H
    np.save(offsets_path, frame_record_offsets)
    
    meta = {
        "file_path": str(raw_path),
        "file_mtime": stat.st_mtime,
        "file_size": stat.st_size,
        "geometry": {
            "height": H,
            "width": geom["width"],
            "n_frames": n_frames,
        },
        "record_size": geom["record_size"],
        "frame_size": frame_size,
        "frame0_offset": header_size,
        "format": "raw",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry detection  (shared with view_recording.py / raw_io.py)
# ──────────────────────────────────────────────────────────────────────────────

def _record_dtype(width: int) -> np.dtype:
    """Return dtype for a record with the given number of pixels (per row)."""
    return np.dtype([
        ("marker", "<u2"),
        ("adc",    "<u4"),
        ("pix",    "<u2", (width,)),
    ])


def _try_geometry(path: str | Path, data_bytes: int,
                  candidate_h: int, expected_width: int | None) -> dict | None:
    """
    Return geometry dict if candidate_h is consistent with the file, else None.
    
    Records are rows: each record has W pixels, H records per frame.
    File: [header] [record0][record1]...[recordH-1][recordH]... (no gaps)
    """
    # Detect width W from first record
    w = _detect_width_from_first_record(path)
    if w is None:
        return None
    if expected_width is not None and w != expected_width:
        return None
    
    record_size = 6 + w * 2
    if data_bytes % record_size != 0:
        return None
    n_records = data_bytes // record_size
    if n_records == 0:
        return None

    dtype = _record_dtype(w)
    recs  = np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n_records,))
    markers = np.asarray(recs["marker"])

    if not np.all((markers == _FRAME_MARKER) | (markers == _LINE_MARKER)):
        return None

    frame_starts = np.flatnonzero(markers == _FRAME_MARKER)
    if len(frame_starts) == 0 or frame_starts[0] != 0:
        return None

    # Height H = records per frame (rows per frame)
    if len(frame_starts) == 1:
        # Only one frame marker at start — use provided candidate_h or infer from n_records
        if candidate_h is not None:
            # Trust the explicitly provided height (single-frame file or user knows the geometry)
            records_per_frame = candidate_h
        elif n_records in _COMMON_HEIGHTS:
            # Auto-detect: single-frame file where n_records IS the height
            records_per_frame = n_records
        else:
            # Cannot infer height from marker pattern
            return None
    else:
        # Multiple frames — H is the gap between frame markers
        gaps = np.diff(frame_starts)
        if not np.all(gaps == gaps[0]):
            return None
        records_per_frame = int(gaps[0])

    if records_per_frame <= 0:
        return None
    # For multi-frame files, verify records_per_frame is consistent
    # For single-frame files with explicit candidate_h, trust it (n_records is just file size)
    if candidate_h is None and n_records % records_per_frame != 0:
        return None
    if not np.array_equal(frame_starts,
                          np.arange(0, n_records, records_per_frame)):
        return None
    if candidate_h is not None and records_per_frame != candidate_h:
        return None

    return {
        "height":      int(records_per_frame),
        "width":       int(w),
        "n_frames":    int(len(frame_starts)),
        "n_records":   int(n_records),
        "record_size": int(record_size),
        "dtype":       dtype,
    }


def _detect_width_from_first_record(path: str | Path) -> int | None:
    """
    Detect frame width W by reading the first record after the 8-byte header.
    Returns None if the file is too short or invalid.
    
    Tries all candidate widths and returns the one that produces the correct
    record alignment for the file size.
    """
    with open(path, "rb") as fh:
        fh.read(8)  # skip header
        fh.seek(0, 2)          # seek to end of file
        file_size = fh.tell()  # read position = file size in bytes
        fh.seek(8)             # back to start of data
        
        candidates = []
        for w in (256, 512, 768, 1024):
            record_size = 6 + w * 2
            fh.seek(8)
            data = fh.read(record_size)
            if len(data) < record_size:
                continue
            marker = int.from_bytes(data[0:2], "little")
            if marker in (_FRAME_MARKER, _LINE_MARKER):
                candidates.append((w, record_size))
        
        if not candidates:
            return None
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0][0]
        
        # Multiple candidates - find the one that aligns with file size
        # The file size (after header) should be a multiple of record_size
        data_bytes = file_size - 8
        for w, record_size in candidates:
            if data_bytes % record_size == 0:
                return w
        
        # Fallback: return the first candidate (512 is most common)
        return 512


def detect_raw_geometry(
        path:   str | Path,
        height: int | None = None,
        width:  int | None = None,
) -> dict:
    """
    Infer RAW file geometry without reading pixel data.

    Uses cached metadata if available and valid, otherwise detects from
    file and caches result.

    Parameters
    ----------
    path   : RAW file produced by udp_recorder_hdf5.py
    height : frame height in pixels; auto-detected if None
    width  : frame width  in pixels; used as a consistency check if given

    Returns
    -------
    dict with keys: height, width, n_frames, n_records, record_size, dtype
    """
    path = Path(path)
    
    # Check cache first
    cached = _load_metadata(path, height=height, width=width)
    if cached is not None:
        file_size  = path.stat().st_size
        header_size = cached.get("frame0_offset", _HEADER_SIZE)
        record_size = cached["record_size"]
        H           = cached["geometry"]["height"]
        W           = cached["geometry"]["width"]

        # Recalculate from file size to catch files that grew (ongoing
        # acquisition) or shrank (truncation) since the cache was written.
        data_bytes        = file_size - header_size
        n_records_actual  = data_bytes // record_size
        n_frames_actual   = n_records_actual // H
        n_orphan          = n_records_actual % H

        # Use cached n_frames only when file size is unchanged (fast path).
        # If the file grew or shrank, trust the recalculated value and warn.
        cached_n_frames = cached["geometry"]["n_frames"]
        if n_frames_actual != cached_n_frames:
            print(f"  NOTE: cached n_frames={cached_n_frames} differs from "
                  f"recalculated n_frames={n_frames_actual} "
                  f"(file size changed). Using recalculated value.")

        if n_orphan != 0:
            print(f"  WARNING: {n_orphan} orphaned rows after {n_frames_actual} "
                  f"complete frames. Last partial frame ignored.")

        return {
            "height":      H,
            "width":       W,
            "n_frames":    n_frames_actual,       # always from file size
            "record_size": record_size,
            "n_records":   n_records_actual,
            "dtype":       _record_dtype(W),
            "_frame_record_offsets": cached.get("_frame_record_offsets"),
        }

    
    # Detect from file
    with open(path, "rb") as fh:
        magic = fh.read(8)
    if magic != _HEADER_MAGIC:
        raise ValueError(
            f"Unexpected RAW header {magic!r}; expected {_HEADER_MAGIC!r}.")

    data_bytes = path.stat().st_size - 8
    candidates = (int(height),) if height is not None else _COMMON_HEIGHTS
    matches    = [m for h in candidates
                  if (m := _try_geometry(path, data_bytes, h, width)) is not None]

    if len(matches) == 1:
        geom = matches[0]
        # Cache the result
        _save_metadata(path, geom)
        return geom
    if len(matches) > 1:
        choices = ", ".join(f"{m['height']}×{m['width']}" for m in matches)
        raise ValueError(
            f"Ambiguous RAW geometry ({choices}). "
            "Pass height= (and optionally width=) to resolve.")
    extra = f" with height={height}" if height is not None else ""
    raise ValueError(
        f"Cannot infer RAW geometry from marker pattern{extra}. "
        "Pass height= for ROI or non-standard frame sizes.")


# ──────────────────────────────────────────────────────────────────────────────
# Shared memmap store  (one per file path, re-used across threads)
# ──────────────────────────────────────────────────────────────────────────────

class _RawStore:
    """
    Caches the memmap, geometry, and frame offsets for one RAW file.

    The pixel memmap is read-only and safe to share across threads.
    Uses pre-computed frame offsets when available (from cache).
    """

    _cache: dict[str, "_RawStore"] = {}
    _lock  = threading.Lock()

    @classmethod
    def get(cls, path: str | Path,
            height: int | None = None,
            width:  int | None = None) -> "_RawStore":
        key = f"{Path(path).resolve()}::{height}x{width}"
        with cls._lock:
            if key not in cls._cache:
                cls._cache[key] = cls(path, height, width)
            return cls._cache[key]

    def __init__(self, path: str | Path,
                 height: int | None = None,
                 width:  int | None = None) -> None:
        path        = Path(path)
        geom        = detect_raw_geometry(path, height=height, width=width)
        self.height = geom["height"]
        self.width  = geom["width"]
        self.n      = geom["n_frames"]
        dtype       = geom["dtype"]
        n_records   = geom["n_records"]

        # Full memmap of the record array (header already skipped via offset=8)
        self._recs = np.memmap(path, dtype=dtype, mode="r",
                               offset=8, shape=(n_records,))
        
        # Pre-computed frame record offsets (from cache) or computed on-demand
        cached_offsets = geom.get("_frame_record_offsets")
        if cached_offsets is not None:
            self._frame_offsets = cached_offsets
        else:
            # Compute record offsets: each frame starts at frame_index * height
            self._frame_offsets = np.arange(self.n, dtype=np.int64) * self.height

    def get_frames(self, indices: np.ndarray) -> np.ndarray:
        """
        Return frames for the given indices as float32 (N, H, W).

        Indices must be in [0, n_frames).  Order is preserved.
        
        Optimized for large files: reads contiguous memory regions instead of
        iterating row-by-row. Uses pre-computed frame offsets.
        """
        W  = self.width
        H  = self.height
        n  = len(indices)
        out = np.empty((n, H, W), dtype=np.float32)

        # Validate indices before any IO
        bad = indices >= self.n
        if bad.any():
            raise IndexError(
                f"Frame indices out of range: requested max={int(indices.max())}, "
                f"file contains {self.n} complete frames. "
                f"File may be truncated.")

        for i, fi in enumerate(indices):
            rec_start = int(self._frame_offsets[fi])
            # Check we won't read past end of memmap (truncated file guard)
            rec_end = rec_start + H
            if rec_end > len(self._recs):
                raise IOError(
                    f"Frame {fi}: records [{rec_start}:{rec_end}] exceed "
                    f"memmap size {len(self._recs)}. "
                    f"File is truncated — expected {H} rows but only "
                    f"{len(self._recs) - rec_start} available.")
            frame_rows = self._recs[rec_start:rec_end]["pix"]
            out[i] = np.ascontiguousarray(frame_rows, dtype=np.float32)

        return out

# ──────────────────────────────────────────────────────────────────────────────
# Frame index discovery  (mirrors io_h5.get_frame_indices)
# ──────────────────────────────────────────────────────────────────────────────

def get_frame_indices(
        raw_path:      str | Path,
        complete_only: bool       = True,
        max_frames:    int | None = None,
        height:        int | None = None,
        width:         int | None = None,
) -> np.ndarray:
    """
    Return the indices of usable frames in the RAW file.

    RAW files carry no completeness metadata, so complete_only has no
    effect — every frame is treated as complete.  The parameter is kept
    for API symmetry with io_h5.get_frame_indices.

    Parameters
    ----------
    raw_path      : path to RAW file
    complete_only : ignored (all RAW frames are considered complete)
    max_frames    : cap on number of indices returned
    height        : frame height; auto-detected if None
    width         : frame width;  used as a consistency check if given

    Returns
    -------
    indices : int array of frame indices
    """
    store = _RawStore.get(raw_path, height=height, width=width)
    n_total = store.n

    if complete_only:
        # No completeness info in RAW — note it once and continue
        pass   # all frames used; nothing to filter

    idx = np.arange(n_total)
    if max_frames is not None:
        idx = idx[:max_frames]
    if len(idx) == 0:
        raise ValueError("No usable frames found in RAW file.")

    print(f"  {len(idx)} frames  ({store.height} × {store.width} pixels)  "
          f"[RAW — all frames treated as complete]")
    return idx


def _apply_frame_selection(
        indices:     np.ndarray,
        skip_frames: int = 0,
        max_frames:  int | None = None,
) -> np.ndarray:
    """
    Apply a leading skip and optional cap to a frame index array.

    Parameters
    ----------
    indices     : frame indices returned by get_frame_indices()
    skip_frames : drop the first N indices
    max_frames  : cap total count after skipping
    """
    if skip_frames > 0:
        if skip_frames >= len(indices):
            raise ValueError(
                f"skip_frames={skip_frames} >= total frames "
                f"in first file ({len(indices)}). "
                "Nothing left to process.")
        indices = indices[skip_frames:]
        print(f"  Skipping first {skip_frames} frames "
              f"({len(indices)} remaining)")

    if max_frames is not None:
        indices = indices[:max_frames]

    return indices


# ──────────────────────────────────────────────────────────────────────────────
# Chunk iteration helpers  (identical API to io_h5)
# ──────────────────────────────────────────────────────────────────────────────

def make_chunks(indices: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """Split *indices* into a list of sub-arrays of length ≤ chunk_size."""
    return [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]


def read_chunk(
        raw_path:      str | Path,
        frame_indices: np.ndarray,
        height:        int | None = None,
        width:         int | None = None,
) -> np.ndarray:
    """
    Read a chunk of frames from a RAW file into float32.

    The underlying pixel data is memory-mapped, so only the requested
    records are paged in by the OS.  The memmap is shared across threads
    (read-only), so this function is safe to call concurrently.

    Returns
    -------
    data : float32 (len(frame_indices), height, width)
    """
    store = _RawStore.get(raw_path, height=height, width=width)
    return store.get_frames(frame_indices)


# ──────────────────────────────────────────────────────────────────────────────
# Multithreaded frame processor  (mirrors io_h5.process_frames_mt)
# ──────────────────────────────────────────────────────────────────────────────

def process_frames_mt(
        raw_path:   str | Path,
        indices:    np.ndarray,
        worker_fn:  Callable[[np.ndarray, np.ndarray], object],
        chunk_size: int = 64,
        n_workers:  int = 8,
        desc:       str = "frames",
        height:     int | None = None,
        width:      int | None = None,
) -> list:
    """
    Process all frames in parallel using a thread pool.

    The RAW memmap is shared read-only across all threads — no per-thread
    file handles are needed (unlike the HDF5 version).

    Each worker:
      1. Reads the chunk of raw frames from the shared memmap.
      2. Calls worker_fn(raw_frames, chunk_frame_indices) → result.

    The list of results (one per chunk, in completion order) is returned.
    Order within the list is non-deterministic; if ordering matters, include
    the frame indices in the result and sort afterwards.

    Parameters
    ----------
    raw_path   : source RAW file
    indices    : frame indices to process (from get_frame_indices)
    worker_fn  : callable(raw_chunk: float32 array, frame_indices: int array) → any
    chunk_size : frames per chunk
    n_workers  : threads
    desc       : label for progress output
    height     : frame height; auto-detected if None
    width      : frame width; consistency check if given

    Returns
    -------
    results : list of worker_fn return values (one per chunk)
    """
    import time

    # Warm up the store (geometry detection + memmap) once before threading
    store = _RawStore.get(raw_path, height=height, width=width)

    chunks   = make_chunks(indices, chunk_size)
    n_chunks = len(chunks)
    print(f"  {n_chunks} chunks × ≤{chunk_size} {desc}  ({n_workers} workers)  [RAW]")

    results = []
    lock    = threading.Lock()
    done    = [0]
    t0      = time.time()

    def _task(chunk_idx: np.ndarray) -> object:
        raw = store.get_frames(chunk_idx)
        return worker_fn(raw, chunk_idx)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_task, ch): i for i, ch in enumerate(chunks)}
        for fut in as_completed(futures):
            res = fut.result()
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
# Format-dispatching convenience layer
# ──────────────────────────────────────────────────────────────────────────────

def get_io_module(data_format: str):
    """
    Return the appropriate I/O module for *data_format*.

    Parameters
    ----------
    data_format : 'h5' or 'raw'

    Returns
    -------
    module with get_frame_indices, read_chunk, process_frames_mt, make_chunks

    Example
    -------
    ::

        io = get_io_module(cfg.offset["data_format"])
        indices = io.get_frame_indices(cfg.offset["dark_run_file"])
        results = io.process_frames_mt(path, indices, worker_fn)
    """
    fmt = data_format.lower().strip()
    if fmt == "h5":
        from . import hdf5 as io_h5
        return io_h5
    if fmt == "raw":
        return _RawIOProxy()
    raise ValueError(
        f"Unknown data_format {data_format!r}. Use 'h5' or 'raw'.")


class _RawIOProxy:
    """
    Thin proxy that exposes this module's functions as attributes, so
    get_io_module('raw') returns an object with the same dot-access API
    as get_io_module('h5').
    """
    get_frame_indices = staticmethod(get_frame_indices)
    read_chunk        = staticmethod(read_chunk)
    make_chunks       = staticmethod(make_chunks)
    process_frames_mt = staticmethod(process_frames_mt)
