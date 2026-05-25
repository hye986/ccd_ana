"""
pnccd_ana.lib.process_frames
============================
Frame-batch processing: offset subtraction, CM correction, and event
recognition for raw frames or sub-frames (partial-ASIC readout data).

Only the per-frame correction and the worker factory live here.
The I/O lives in utils/io_h5.py and utils/io_raw.py.
The CLI wires these through cli/source_ana.py.

Public API  (also re-exported by ..lib.__init__)
───────────────────────────────────────────────
  correct_frame      — offset-sub → CM-correct a single (Y, X) frame
  make_worker        — builds the per-chunk callable for process_frames_mt
  process_frames_mt  — thread-parallel chunk dispatcher
"""

from __future__ import annotations

import threading
import concurrent.futures
from typing import Any

import numpy as np

from .common_mode         import cm_correct_frame
from .pattern_recognition import find_events, EVENT_DTYPE
from .geometry            import ASIC_SLICES


def correct_frame(
        raw:   np.ndarray,
        cal:   dict,
        asics: list[str] | None,
) -> np.ndarray:
    """
    Apply pedestal offset then common-mode correction to a single frame.

    Parameters
    ----------
    raw   : float32 (Y, X)  — raw ADC values after optional rollover correction
    cal   : calibration dict from load_calibration_h5()
    asics : list of ASIC names, or None for global

    Returns
    -------
    corrected : float32 (Y, X) — cleaned frame ready for find_events
    """
    frame = raw.astype(np.float32)

    # Subtract per-pixel offset (prefer sigclip, fall back to median)
    off = _load_offset(cal, asics)
    if off is not None:
        if off.shape == frame.shape:
            frame = frame - off
        elif off.ndim == 2:
            # Sub-frame: clip offset to what fits
            frame = frame - off[:frame.shape[0], :frame.shape[1]]

    # Common-mode correction: median over Y for each X column.
    # This is exactly "subtract the median of all pixels in that row" in
    # your detector terminology (each "row" = fixed Y = runs vertically;
    # the pnCCD column electronics give one CM value per horizontal X position).
    corrected, _ = cm_correct_frame(frame)
    return corrected


def _load_offset(cal: dict, asics: list[str] | None) -> np.ndarray | None:
    """Load the offset map from the calibration dict (prefer sigclip > median)."""
    for preference in ("sigmaclip", "median"):
        key = f"offset/{preference}"
        if asics:
            for aname in asics:
                if cal.get(aname, {}).get(key) is not None:
                    return cal[aname][key].astype(np.float32)
        if cal.get("global", {}).get(key) is not None:
            return cal["global"][key].astype(np.float32)
    return None


def make_worker(
        cal:             dict,
        asics:           list[str] | None,
        noise_map:       np.ndarray,
        seed_sigma:      float,
        split_sigma:     float,
        reject_extra:    bool,
        search_mask:     np.ndarray | None,
        bad_pixel_mask:  np.ndarray | None = None,
        sample_buf:      list | None = None,
        sample_max:      int         = 0,
) -> Any:
    """
    Build the per-chunk worker callable passed to process_frames_mt.

    Thread safety: only sample_buf.append() is GIL-locked.
    """
    _lock = threading.Lock() if sample_buf is not None else None

    def _worker(raw_chunk: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
        # Embed sub-frame into 1024×1024 canvas when raw is not full-size
        if asics and (raw_chunk.shape[1] != 1024 or raw_chunk.shape[2] != 1024):
            raw_chunk = _embed_into_full_frame(raw_chunk, asics)

        chunk_events: list[np.ndarray] = []
        for frame in raw_chunk:
            corrected = correct_frame(frame, cal, asics)

            if sample_buf is not None:
                with _lock:                             # type: ignore
                    if len(sample_buf) < sample_max:  # type: ignore
                        sample_buf.append(corrected.copy())  # type: ignore

            evts = find_events(
                corrected, noise_map,
                search_mask=search_mask,
                seed_sigma=seed_sigma,
                split_sigma=split_sigma,
                reject_extra=reject_extra,
                bad_pixel_mask=bad_pixel_mask,
            )
            if len(evts):
                chunk_events.append(evts)

        if chunk_events:
            return np.concatenate(chunk_events)
        return np.empty(0, dtype=EVENT_DTYPE)

    return _worker


def _embed_into_full_frame(
        data:  np.ndarray,
        asics: list[str],
) -> np.ndarray:
    """
    Place each ASIC sub-frame from a (n_frames, H, W) chunk into a
    (n_frames, 1024, 1024) canvas.

    Handles two common raw formats:
      · 512×1024  : 2 ASICs stacked vertically
      · 1024×512  : 2 ASICs side-by-side
    """
    n_frames, sub_h, sub_w = data.shape
    canvas = np.zeros((n_frames, 1024, 1024), dtype=data.dtype)

    if sub_h == 512 and sub_w == 1024:
        canvas[:, :512, :], canvas[:, 512:, :] = data, data
    elif sub_h == 1024 and sub_w == 512:
        canvas[:, :, :512], canvas[:, :, 512:] = data, data
    else:
        canvas[:, :sub_h, :sub_w] = data

    return canvas


def process_frames_mt(
        worker:      Any,
        raw_frames: np.ndarray,
        n_workers: int = 8,
        chunk_size: int = 64,
) -> np.ndarray:
    """
    Dispatch frames across *n_workers* threads in chunks of *chunk_size*.

    Parameters
    ----------
    worker     : callable returned by make_worker()
    raw_frames: uint16 (N, H, W)
    n_workers : thread pool size
    chunk_size: frames per chunk

    Returns
    -------
    events : structured array 1-D  (all chunks concatenated)
    """
    indices = np.arange(raw_frames.shape[0], dtype=np.uint32)
    chunks = [
        (raw_frames[i:i + chunk_size], indices[i:i + chunk_size])
        for i in range(0, len(indices), chunk_size)
    ]
    all_results: list[np.ndarray] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker, chunk, idx) for chunk, idx in chunks]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            if result is not None and len(result):
                all_results.append(result)

    if not all_results:
        return np.empty(0, dtype=EVENT_DTYPE)
    return np.concatenate(all_results)
