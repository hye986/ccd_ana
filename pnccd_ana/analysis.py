"""
pnccd_ana.analysis
====================
High-level public API for pnCCD Fe-55 analysis.

This module is the entry point for users who want to understand or script the
analysis without reading the internals.  All analysis happens in four distinct
stages that are executed in order:

Stage 1 — Offset calibration
──────────────────────────────────
Purpose  : Measure per-pixel electronic offset and noise from dark frames
          (frames with no source illuminating the sensor).
Inputs  : dark_run.raw   shape (N, H, W) uint16 (single hybrid: H×W can be 512×512, 1024×512, etc.)
Outputs : run0001/offset.h5
          ├── asics/H{0,1,2,3}/
          │   ├── offset/median       (512, 512) float32
          │   ├── offset/sigmaclip    (512, 512) float32
          │   ├── noise/pixel_rms     (512, 512) float32
          │   ├── noise/cm_rms        (512, 512) float32
          │   └── noise/n_clipped     (512, 512) float32
          └── global/                 same maps, full H×W (single hybrid)

Key results:
  offset (float32)  : μ per pixel — subtract from each raw frame pixel
  noise  (float32)   : σ per pixel — used for event-detection thresholds

Algorithms used:
  · Pedestal — "both" (median + sigma-clip) or pick one
  · Sigma-clip — iterative μ ± 3σ reject, up to 20 iterations
  · Common-mode (CM) — per-row median over all X columns was applied to the
    residual (raw − offset) before noise estimation.  This CM correction is
    exactly what "subtract the median value of all pixels in each row from
    every pixel in that row" describes in rolling-shutter terms: each horizontal
    row (fixed Y) is independent, so median over X for each Y = per-row correction.

CLI shortcut:
  python -m pnccd_ana.cli.offset --config your.yaml

Stage 2 — Noise map and bad-pixel mask from calibration
────────────────────────────────────────────────────────
Purpose  : Derive per-pixel noise RMS and an optional bad-pixel mask.
Inputs   : offset.h5
Outputs  : noise_map                                        (H, W) float32
           bad_pixel_mask | None                           (H, W) bool

  Bad-pixel criteria (a pixel is bad when ANY of these holds):
    HOT      : noise > hot_rms_multiple  × median(active_noise) → fires false seeds
    COLD     : noise < cold_rms_fraction × median(active_noise) → stuck / dead
    NONFINITE: noise is NaN / Inf / ≤ 0                          → unusable
    UNSTABLE : n_clipped/n_dark > max_clip_fraction             → unreliable σ

  A bad pixel is zeroed out in a copy of the analysis frame.  It can be neither
  a seed centre nor a split neighbour.

Stage 3 — Source: offset → CM → event recognition
─────────────────────────────────────────────────
Purpose  : Turn raw voltage frames into photon-event lists.
Inputs   : source_run.raw   shape (N, H, W) uint16 (single hybrid: H×W can be 512×512, 1024×512, etc.)
           offset.h5
           noise_map                                   (H, W) float32
           search_mask    | None                       (H, W) bool
           bad_pixel_mask | None                      (H, W) bool
           seed_sigma = 5.0,  split_sigma = 3.0       (threshold multipliers)
Outputs  : events structured array 1-D with fields:
             Y, X    — coordinates (Y=row 0..H-1, X=col 0..W-1)
             grade   — 0=single, 1-4=double, 5-8=triple, 9-12=quad, 13=other
             adu_sum — total charge in all pattern pixels
             adu_seed — charge at the centre pixel

Event-detection algorithm (seed → pattern → grade):
--------------------------------------------------------------------------------
  1. Mask:  frame_copy[~include] = 0.0   where include = search_mask AND ~(bad)
           → bad pixels can be neither seeds nor neighbours.
  2. Local-max 5×5:
           lmax[y,x] = max of the 25-pixel window centred on (y,x)
  3. Seed candidates — every (y,x) in ±2 boundary where ALL of:
           a. centre[y,x]  > seed_sigma × noise[y,x]          (above threshold)
           b. centre[y,x]  == lmax[y,x]                        (local peak)
           c. centre is lex-smallest on tie (min y; on tie, min x)
              among all 5×5-window pixels sharing the same peak value.
           Rule (c) collapses 2-pixel charge clusters and 2×2 hot patches
           into exactly one event each.
  4. For each surviving seed — look at the central 3×3 around it.
           Every neighbour value > split_sigma × noise[neighbour]
           sets its bit in the neighbour mask.
  5. Map (centre_is_local_max, neighbour_mask) → grade 0-12 (or 13=other).
  6. Sum all pattern pixels for that grade → adu_sum.
     Also record the centre pixel value as adu_seed.
  7. If reject_extra=True: discard grade-13 events.

Grade table (centre is always the local-max / seed; neighbour positions are
relative to centre at (y,x)):
--------------------------------------------------------------------------------
  Grade | Name    | Central 3×3 neighbours above split_sigma threshold
  ──────┼─────────┼─────────────────────────────────────────────────────────
    0   | single  | none
    1   | double↔ | right  (y,   x+1)
    2   | double↓ | below  (y+1, x  )
    3   | double← | left   (y,   x-1)
    4   | double↑ | above  (y-1, x  )
    5   | triple↔+↓| right  (y,   x+1) + below (y+1, x  )
    6   | triple←+↓| left   (y,   x-1) + below (y+1, x  )
    7   | triple←+↑| left   (y,   x-1) + above (y-1, x  )
    8   | triple↑+→| above  (y-1, x  ) + right (y,   x+1)
    9   | quad ↔+↓+↘ | right (y,x+1)+below (y+1,x)+diag (y+1,x+1)
   10   | quad ↓+←+↙ | below (y+1,x)+left (y,x-1)+diag (y+1,x-1)
   11   | quad ←+↑+↖ | left  (y,x-1)+above(y-1,x)+diag (y-1,x-1)
   12   | quad ↑+→+↗ | above (y-1,x)+right(y,x+1)+diag (y-1,x+1)
   13   | other   | any 3×3 pattern not listed above
--------------------------------------------------------------------------------
  Fe-55 doublet: Mn Kα 5.89 keV → ≈1586 ADU,  Kβ 6.39 keV → ≈1723 ADU.

CLI shortcut:
  python -m pnccd_ana.cli.event_rec --config your.yaml

Stage 4 — Energy calibration (future)
───────────────────────────────────
Events from Stage 3 are calibrated to photon energy using Fe-55 peaks.


═════════════════════════════════════════════════════════════════════════════
Array / axis convention — read before you plot
═════════════════════════════════════════════════════════════════════════════
  Array layout:  data[frame, Y, X]
    axis 0 = Y = row index = VERTICAL direction on screen = "detector row" axis
    axis 1 = X = col index = HORIZONTAL direction on screen = "detector column" axis

    The pnCCD sensor has 4 ASICs (hybrids), each 512×512 pixels, but this
    version is configured for SINGLE HYBRID operation. Supported frame sizes:
      - 512×512  (full single ASIC)
      - 1024×512 (2 ASICs vertically stacked)
      - Other heights supported via frame_rows config option

      H0 (top-right) — single ASIC readout
                          Y
                          ↑   0 … H-1  (Y increases going down-screen if
                          │                 origin=upper = imshow default)
                          │   This analysis uses origin="lower" everywhere,
                          └──X→  0 … W-1  so Y=0 is at the BOTTOM of
                                                   the figure (readout is at
                                                   the bottom of the sensor).

  Plot axis labels used throughout:
    · x-axis label : "X [detector column]"  (horizontal)
    · y-axis label : "Y [detector row]"       (vertical)
  Your team uses "row = x-axis = vertical" exactly matching what the Y
  variable represents.  There is a naming conflict with older docstrings
  that called X "detector row" — those labels are now superseded.
  The coordinate (Y=H//2, X=W//2) lands in the center of the single ASIC.


═════════════════════════════════════════════════════════════════════════════
Minimal usage examples for scripts / notebooks
═════════════════════════════════════════════════════════════════════════════

Stage 1 — dark calibration
───────────────────────────────────
>>> from pnccd_ana.analysis import run_offset
>>> run_offset(
...     "dark_run.raw",          # RAW 512x512 format
...     output_dir="run0001",
...     asics=["H0"],            # single ASIC (H0, H1, H2, or H3)
...     pedestal_method="both",
... )
#  →  run0001/offset.h5

Stage 2 — noise map + bad-pixel mask
───────────────────────────────────
>>> from pnccd_ana.analysis import load_calibration, build_noise_map, build_bad_pixel_mask
>>> cal = load_calibration("run0001/offset.h5", asics=["H0"])
>>> noise = build_noise_map(cal, asics=["H0"])
>>> bad = build_bad_pixel_mask(
...     noise,
...     n_clipped_map=cal["noise"]["n_clipped"],
...     n_dark_frames=N_DARK_FRAMES,
...     hot_rms_multiple=5.0,
...     cold_rms_fraction=0.1,
...     max_clip_fraction=0.5,
... )
>>> print(f"{bad.sum():,} bad pixels out of {noise.size:,} total")

Stage 3 — process source frames (batch, no CLI)
─────────────────────────────────────────────
>>> from pnccd_ana.analysis import load_raw_h5, process_frames
>>> raw = load_raw_h5("fe55_run.raw", max_frames=50000)  # RAW format
>>> evts = process_frames(
...     raw,
...     cal,
...     noise,
...     seed_sigma=5.0,
...     split_sigma=3.0,
...     asics=["H0"],            # single ASIC
...     bad_pixel_mask=bad,
... )
>>> print(f"{len(evts):,} events")
>>> import numpy as np
>>> # Fe-55 doublet peaks in ADU
>>> print("adu_sum percentiles:", np.percentile(evts["adu_sum"], [50, 90, 95]))
>>> # Grade distribution
>>> grades, cnts = np.unique(evts["grade"], return_counts=True)
>>> for g, n in zip(grades, cnts):
...     print(f"  grade {g:2d}: {n:7d}  ({100*n/len(evts):.1f}%)")
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Stage 1 ─────────────────────────────────────────────────────────────────────────
from .cli.offset import run as run_offset

# ── Stages 2-3: loaders and processing ─────────────────────────────────────────
from .lib.noise               import build_bad_pixel_mask
from .lib.geometry            import ASIC_SLICES, ASIC_NAMES, ASIC_GRID_POS
from .lib.pattern_recognition import find_events, EVENT_DTYPE
from .utils                 import get_io_module

# ── Stage 4: Gain + CTI calibration ─────────────────────────────────────────
from .cli.energy_cal import run as run_energy_cal
from .cli.energy_cal import load_energy_cal_h5, save_energy_cal_h5
from .lib.calibration import apply_full_calibration, MN_KALPHA_EV

# ── Results loaders ──────────────────────────────────────────────────────────────
from .utils.io_h5 import (
    load_offset_results_h5,
    load_event_rec_results_h5,
    load_energy_cal_results_h5,
)

def load_calibration(path:  str | Path,
                     asics: list[str] | None = None,
                    ) -> dict:
    """
    Load a dark calibration HDF5 into a dict keyed by ASIC name or "global".

    Parameters
    ----------
    path  : path to offset.h5
    asics : list of ASIC names, e.g. ["H0", "H1"], or None for global-only

    Returns
    -------
    dict with per-ASIC (and "global") keys, each containing:
      offset["median"],  offset["sigmaclip"],
      noise["pixel_rms"], noise["cm_rms"],
      noise["n_clipped"] (if sigma-clip was used).
    """
    from .utils.io_h5 import load_calibration_h5
    return load_calibration_h5(path, asics=asics)


def build_noise_map(cal:         dict,
                   asics:       list[str] | None = None,
                   noise_scope: str          = "auto",
                  ) -> np.ndarray:
    """
    Assemble a 1024×1024 noise map from the calibration dict.

    Parameters
    ----------
    cal         : from load_calibration()
    asics       : list of ASIC names, or None (global only)
    noise_scope : "auto" | "global" | "asic"
                  "auto": use per-ASIC if its median ≥ ½ of global median; else global

    Returns
    -------
    noise_map : float32 (1024, 1024) — per-pixel electronic noise [ADU RMS]
    """
    from .cli.event_rec import _build_noise_map
    return _build_noise_map(cal, asics=asics, noise_scope=noise_scope)


def load_raw_h5(path:       str | Path,
                max_frames: int | None = None,
               ) -> np.ndarray:
    """
    Load raw frames from an HDF5 file.

    Parameters
    ----------
    path       : path to frames HDF5
    max_frames : cap or None for all

    Returns
    -------
    raw : uint16 (N, 1024, 1024)
    """
    io   = get_io_module("h5")
    idx  = io.get_frame_indices(path, complete_only=True, max_frames=max_frames)
    return io.read_frames(path, idx).astype(np.uint16)


def process_frames(raw_frames:     np.ndarray,
                   cal:            dict,
                   noise_map:      np.ndarray,
                   seed_sigma:     float = 5.0,
                   split_sigma:    float = 3.0,
                   asics:          list[str] | None        = None,
                   search_mask:    np.ndarray | None        = None,
                   bad_pixel_mask: np.ndarray | None        = None,
                   reject_extra:   bool   = False,
                  ) -> np.ndarray:
    """
    Offset → CM-correct → find-events for a stack of raw frames.

    Parameters
    ----------
    raw_frames     : uint16 (N, 1024, 1024) — raw ADC readings
    cal            : calibration dict from load_calibration()
    noise_map       : float32 (1024, 1024)  — per-pixel noise [ADU RMS]
    seed_sigma      : seed threshold multiplier   (default 5.0 ≈ 5σ)
    split_sigma     : neighbour threshold multiplier (default 3.0 ≈ 3σ)
    asics           : list of ASIC names or None
    search_mask     : bool (1024, 1024) or None — True = inside active ASIC region
    bad_pixel_mask  : bool (1024, 1024) or None — True = bad; always exclude
    reject_extra    : drop grade-13 events if True

    Returns
    -------
    events : structured array 1-D with fields Y, X, grade, adu_sum, adu_seed
    """
    from .cli.event_rec import _correct_frame, _make_worker

    n_frames  = raw_frames.shape[0]
    CHUNK     = 256
    sample_buf: list = []
    all_evts: list[np.ndarray] = []

    worker = _make_worker(
        cal, asics, noise_map,
        seed_sigma, split_sigma, reject_extra,
        search_mask, bad_pixel_mask,
        sample_buf=sample_buf, sample_max=0,
    )

    for start in range(0, n_frames, CHUNK):
        chunk   = raw_frames[start:start + CHUNK]
        indices = np.arange(len(chunk), dtype=np.uint32)
        result  = worker(chunk, indices)
        if result is not None and len(result):
            all_evts.append(result)

    if not all_evts:
        return np.empty(0, dtype=EVENT_DTYPE)
    return np.concatenate(all_evts)


# ──────────────────────────────────────────────────────────────────────────────
# Results HDF5 loaders — reload plot-backing data for redrawing / refitting
# ──────────────────────────────────────────────────────────────────────────────

def load_dark_results(path: str | Path) -> dict:
    """
    Load plot-backing data from offset_results.h5.

    Use this to reload data for replotting or refitting without rerunning
    the dark frame calibration pipeline.

    Parameters
    ----------
    path : path to offset_results.h5

    Returns
    -------
    dict with keys: offsets, noise, bad_pixels, meta.
    Each contains nested arrays/dicts matching the HDF5 structure.

    Example
    -------
    >>> from pnccd_ana.analysis import load_dark_results
    >>> dark = load_dark_results("run0001/offset_results.h5")
    >>> offsets = dark["offsets"]
    >>> noise_map = dark["noise"]["map"]
    >>> hist_edges = dark["noise"]["hist_edges"]
    >>> hist_counts = dark["noise"]["hist_counts"]
    """
    from .utils.io_h5 import load_offset_results_h5 as _load
    return _load(path)


def load_source_results(path: str | Path) -> dict:
    """
    Load plot-backing data from event_rec_results.h5.

    Use this to reload data for replotting or refitting without rerunning
    the source analysis pipeline.

    Parameters
    ----------
    path : path to event_rec_results.h5

    Returns
    -------
    dict with keys: raw_spectrum, grade_distribution, meta.

    Example
    -------
    >>> from pnccd_ana.analysis import load_source_results
    >>> src = load_source_results("run0001/event_rec_results.h5")
    >>> raw = src["raw_spectrum"]
    >>> bin_edges = raw["bin_edges"]
    >>> counts = raw["all_pixels"]
    >>> grades = src["grade_distribution"]["grades"]
    >>> grade_counts = src["grade_distribution"]["counts"]
    """
    from .utils.io_h5 import load_event_rec_results_h5 as _load
    return _load(path)


def load_gain_results(path: str | Path) -> dict:
    """
    Load plot-backing data from energy_cal_results.h5.

    Use this to reload data for replotting or refitting without rerunning
    the gain calibration pipeline.

    Parameters
    ----------
    path : path to energy_cal_results.h5

    Returns
    -------
    dict with keys: phase1_rough_gain, phase3_cti, phase4_column_gain,
    pixel_gain_map, cti_per_col, final_spectrum, meta.

    Example
    -------
    >>> from pnccd_ana.analysis import load_gain_results
    >>> gain = load_gain_results("run0001/energy_cal_results.h5")
    >>> rough = gain["phase1_rough_gain"]
    >>> cti = gain["phase3_cti"]
    >>> col = gain["phase4_column_gain"]
    >>> final = gain["final_spectrum"]
    >>> kalpha = final["kalpha_fit"]
    >>> print(f"K-alpha peak: {kalpha['peak_ev']:.1f} eV")
    >>> print(f"FWHM: {kalpha['fwhm_ev']:.1f} eV")
    >>> print(f"Resolution: {kalpha['resolution_pct']:.2f}%")
    """
    from .utils.io_h5 import load_energy_cal_results_h5 as _load
    return _load(path)
