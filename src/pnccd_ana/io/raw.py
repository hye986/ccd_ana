"""
pnccd_ana.io.raw
================
RAW-format reader for 512×512 single-hybrid pnCCD dark runs.

RAW format
──────────
  File layout (produced by udp_recorder_hdf5.py):

    [8-byte header: b'16BU0000']
    For each frame:
      [0xFFFF marker (2B)] [ADC counter (4B)] [512 × uint16 pixels]   ← first row
      [0xFFFE marker (2B)] [ADC counter (4B)] [512 × uint16 pixels]   ← remaining rows
      ...

  Each record encodes one readout ROW (512 pixels).  A frame is 512
  records (H=512, W=512).  The geometry is fixed; there is no
  auto-detection.

  Array layout after reading: data[frame, Y, X]
    - Y = row index = 0..511 (first row in file = Y=0 = bottom with origin="lower")
    - X = column index = 0..511

  RAW files carry no per-frame metadata (no frame counter, no timestamp,
  no completeness flag), so every complete frame is used.

Reading strategy
────────────────
  The pixel array is memory-mapped via np.memmap so random-access reads
  for arbitrary chunks are O(1) seeks.  The memmap is shared read-only
  across threads, so process_frames_mt needs no per-thread file handles.

  Usage pattern::

      indices = get_frame_indices(raw_path, max_frames=1000)
      for chunk in make_chunks(indices, chunk_size=64):
          frames = read_chunk(raw_path, chunk)

  Or multithreaded::

      results = process_frames_mt(raw_path, indices, worker_fn)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# RAW format constants  (must match udp_recorder_hdf5.py)
# ──────────────────────────────────────────────────────────────────────────────

_HEADER_MAGIC   = b"16BU0000"
_HEADER_SIZE    = 8            # bytes in file header
_RECORD_HEADER  = 6            # 2-byte marker + 4-byte ADC counter
_FRAME_MARKER   = 0xFFFF
_LINE_MARKER    = 0xFFFE

# Fixed 512×512 single-hybrid geometry
FRAME_HEIGHT = 512             # rows per frame (Y)
FRAME_WIDTH  = 512             # columns per row (X)
_RECORD_SIZE = _RECORD_HEADER + FRAME_WIDTH * 2   # bytes per row record

_HEADER_NOTE = "16BU0000"


def _record_dtype() -> np.dtype:
    """Structured dtype for one row record (marker, ADC counter, pixels)."""
    return np.dtype([
        ("marker", "<u2"),
        ("adc",    "<u4"),
        ("pix",    "<u2", (FRAME_WIDTH,)),
    ])


def _validate_header(path: str | Path) -> None:
    """Raise ValueError if *path* does not start with the RAW magic header."""
    with open(path, "rb") as fh:
        magic = fh.read(_HEADER_SIZE)
    if magic != _HEADER_MAGIC:
        raise ValueError(
            f"Unexpected RAW header {magic!r}; expected {_HEADER_MAGIC!r}.")


# ──────────────────────────────────────────────────────────────────────────────
# Shared memmap store  (one per file path, re-used across threads)
# ──────────────────────────────────────────────────────────────────────────────

class _RawStore:
    """
    Caches the memmap and frame count for one RAW file.

    The pixel memmap is read-only and safe to share across threads.
    """

    _cache: dict[str, "_RawStore"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, path: str | Path) -> "_RawStore":
        key = str(Path(path).resolve())
        size = Path(path).stat().st_size
        with cls._lock:
            store = cls._cache.get(key)
            # Rebuild if the file grew/shrank since it was cached
            # (e.g. ongoing acquisition or truncation).
            if store is None or store.file_size != size:
                store = cls._cache[key] = cls(path)
            return store

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        _validate_header(path)

        self.height = FRAME_HEIGHT
        self.width  = FRAME_WIDTH

        data_bytes = path.stat().st_size - _HEADER_SIZE
        if data_bytes < 0:
            raise ValueError(f"{path}: file too small for RAW header.")
        self.file_size = path.stat().st_size
        self.n_records = data_bytes // _RECORD_SIZE
        self.n = self.n_records // self.height

        if self.n == 0:
            raise ValueError(
                f"{path}: no complete 512×512 frames found "
                f"({self.n_records} rows, {self.height} rows/frame).")

        self._recs = np.memmap(path, dtype=_record_dtype(), mode="r",
                               offset=_HEADER_SIZE, shape=(self.n_records,))

    def get_frames(self, indices: np.ndarray) -> np.ndarray:
        """
        Return frames for the given indices as float32 (N, H, W).

        Indices must be in [0, n).  Order is preserved.
        """
        H, W = self.height, self.width
        out  = np.empty((len(indices), H, W), dtype=np.float32)

        for i, fi in enumerate(indices):
            fi = int(fi)
            if fi < 0 or fi >= self.n:
                raise IndexError(
                    f"Frame index {fi} out of range (file contains "
                    f"{self.n} complete frames).")
            rec_start = fi * H
            rows = self._recs[rec_start:rec_start + H]["pix"]
            out[i] = np.ascontiguousarray(rows, dtype=np.float32)

        return out


# ──────────────────────────────────────────────────────────────────────────────
# Frame index discovery
# ──────────────────────────────────────────────────────────────────────────────

def get_frame_indices(
        raw_path:   str | Path,
        max_frames: int | None = None,
) -> np.ndarray:
    """
    Return the indices of usable frames in the RAW file.

    Every complete frame in the file is used.  A trailing partial frame
    (fewer than 512 rows) is ignored.

    Parameters
    ----------
    raw_path   : path to RAW file
    max_frames : cap on number of indices returned

    Returns
    -------
    indices : int array of frame indices
    """
    store = _RawStore.get(raw_path)

    if store.n_records % store.height != 0:
        print(f"  WARNING: {store.n_records % store.height} orphaned rows "
              f"after {store.n} complete frames. Last partial frame ignored.")

    idx = np.arange(store.n)
    if max_frames is not None:
        idx = idx[:max_frames]
    if len(idx) == 0:
        raise ValueError("No usable frames found in RAW file.")

    print(f"  {len(idx)} frames  ({store.height} × {store.width} pixels)")
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
                f"in first file ({len(indices)}). Nothing left to process.")
        indices = indices[skip_frames:]
        print(f"  Skipping first {skip_frames} frames "
              f"({len(indices)} remaining)")

    if max_frames is not None:
        indices = indices[:max_frames]

    return indices


# ──────────────────────────────────────────────────────────────────────────────
# Chunk iteration helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_chunks(indices: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """Split *indices* into a list of sub-arrays of length ≤ chunk_size."""
    return [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]


def read_chunk(raw_path: str | Path,
               frame_indices: np.ndarray) -> np.ndarray:
    """
    Read a chunk of frames from a RAW file into float32.

    The underlying pixel data is memory-mapped, so only the requested
    records are paged in by the OS.  Safe to call concurrently.

    Returns
    -------
    data : float32 (len(frame_indices), 512, 512)
    """
    return _RawStore.get(raw_path).get_frames(frame_indices)


# ──────────────────────────────────────────────────────────────────────────────
# Multithreaded frame processor
# ──────────────────────────────────────────────────────────────────────────────

def process_frames_mt(
        raw_path:   str | Path,
        indices:    np.ndarray,
        worker_fn:  Callable[[np.ndarray, np.ndarray], object],
        chunk_size: int = 64,
        n_workers:  int = 8,
        desc:       str = "frames",
) -> list:
    """
    Process all frames in parallel using a thread pool.

    The RAW memmap is shared read-only across all threads.

    Each worker:
      1. Reads the chunk of raw frames from the shared memmap.
      2. Calls worker_fn(raw_frames, chunk_frame_indices) → result.

    Parameters
    ----------
    raw_path   : source RAW file
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

    store = _RawStore.get(raw_path)   # warm up memmap once before threading

    chunks   = make_chunks(indices, chunk_size)
    n_chunks = len(chunks)
    print(f"  {n_chunks} chunks × ≤{chunk_size} {desc}  ({n_workers} workers)")

    results = []
    lock    = threading.Lock()
    done    = [0]
    t0      = time.time()

    def _task(chunk_idx: np.ndarray) -> object:
        return worker_fn(store.get_frames(chunk_idx), chunk_idx)

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
