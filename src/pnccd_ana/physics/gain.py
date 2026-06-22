"""
pnccd_ana.physics.gain
======================
Per-column Gaussian peak fitting for gain orientation.

Matches ROOT HStepGainMapCCDHLL Phase 2 (orientation fit):
  - Global peak fallback (per parity, per half if SplitFrame)
  - Per-column histogram + Gaussian fit
  - Fallback to global peak if <MinHitsPerColumn or fit failed

No gain map filling here — that happens in cti.py after CTE fit.

Coordinate convention
─────────────────────
  Y = row (axis 0),  X = col (axis 1)
  parity = (col + 1) % 2   matches ROOT (col+1)%2
  even columns: col=0,2,4,... → parity=1
  odd  columns: col=1,3,5,... → parity=0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

MN_KALPHA_EV = 5898.8   # Mn Kα energy [eV]
MN_KBETA_EV  = 6490.4   # Mn Kβ energy [eV]

MIN_HITS_PER_COLUMN = 20    # ROOT MinHitsPerColumn
MIN_HITS_GLOBAL     = 200   # ROOT MinHitsGlobal


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PeakFitResult:
    peak_adu:  float         # Gaussian mean
    sigma_adu: float         # Gaussian sigma
    amplitude: float         # Gaussian amplitude
    success:   bool          # fit converged and peak in ROI
    n_events:  int           # histogram entries used


@dataclass
class GlobalPeakResult:
    """
    Fallback peaks per parity (0/1) and per half (0=bottom, 1=top).

    ppos[parity, half]  — peak position [ADU]
    sigma[parity, half] — Gaussian sigma [ADU]

    For non-SplitFrame detectors: only half=0 is used.
    For non-SplitEvenOdd:        only parity=0 is used.
    """
    ppos:  np.ndarray   # shape (2, 2)  float64
    sigma: np.ndarray   # shape (2, 2)  float64


@dataclass
class ColumnPeakResult:
    """
    Per-column, per-half peak positions and sigmas.

    ppos[col, half]  — [ADU]
    sigma[col, half] — [ADU]
    used_fallback[col, half] — True if global peak was used
    """
    ppos:          np.ndarray   # (n_cols, n_halves)  float64
    sigma:         np.ndarray   # (n_cols, n_halves)  float64
    used_fallback: np.ndarray   # (n_cols, n_halves)  bool


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian model
# ══════════════════════════════════════════════════════════════════════════════

def _gauss(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _gauss_plus_bg(x, amplitude, mean, sigma, bg):
    return _gauss(x, amplitude, mean, sigma) + bg


def _gauss_plus_linear(x, amplitude, mean, sigma, bg, slope):
    return _gauss(x, amplitude, mean, sigma) + bg + slope * x


# ══════════════════════════════════════════════════════════════════════════════
# Single peak fit
# ══════════════════════════════════════════════════════════════════════════════

def fit_peak(
        values:     np.ndarray,
        roi_low:    float,
        roi_high:   float,
        n_params:   int = 3,
        n_bins:     int | None = None,
) -> PeakFitResult:
    """
    Fit a Gaussian to a 1-D array of ADU values within [roi_low, roi_high].

    Parameters
    ----------
    values   : 1-D float array — raw ADU values (already in ROI or not)
    roi_low  : lower bound of fit region [ADU]
    roi_high : upper bound of fit region [ADU]
    n_params : 3 = pure Gaussian
               4 = Gaussian + constant background
               5 = Gaussian + linear background
    n_bins   : histogram bins (default: roi_high - roi_low, capped at 4000)

    Returns
    -------
    PeakFitResult
    """
    mask = (values > roi_low) & (values < roi_high)
    v    = values[mask]
    n    = len(v)
    _fail = PeakFitResult(
        peak_adu  = 0.5 * (roi_low + roi_high),
        sigma_adu = (roi_high - roi_low) / 6.0,
        amplitude = 0.0,
        success   = False,
        n_events  = n,
    )
    if n < 10:
        return _fail

    nbins  = n_bins if n_bins else min(int(roi_high - roi_low), 4000)
    nbins  = max(nbins, 20)
    counts, edges = np.histogram(v, bins=nbins,
                                  range=(roi_low, roi_high))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Initial guesses
    peak_bin  = int(np.argmax(counts))
    amp0      = float(counts[peak_bin])
    mean0     = float(centers[peak_bin])
    sigma0    = (roi_high - roi_low) / 6.0

    if n_params == 3:
        fn     = _gauss
        p0     = [amp0, mean0, sigma0]
        bounds = ([0, roi_low, 0],
                  [np.inf, roi_high, roi_high - roi_low])
    elif n_params == 4:
        fn     = _gauss_plus_bg
        bg0    = float(np.percentile(counts, 10))
        p0     = [amp0, mean0, sigma0, bg0]
        bounds = ([0, roi_low, 0, 0],
                  [np.inf, roi_high, roi_high - roi_low, np.inf])
    else:  # 5
        fn     = _gauss_plus_linear
        bg0    = float(np.percentile(counts, 10))
        p0     = [amp0, mean0, sigma0, bg0, 0.0]
        bounds = ([0, roi_low, 0, 0, -np.inf],
                  [np.inf, roi_high, roi_high - roi_low, np.inf, np.inf])

    try:
        popt, _ = curve_fit(fn, centers, counts.astype(float),
                            p0=p0, bounds=bounds,
                            maxfev=5000)
        amp, mean, sigma = popt[0], popt[1], abs(popt[2])
        if not (roi_low < mean < roi_high) or sigma <= 0:
            return _fail
        return PeakFitResult(
            peak_adu  = float(mean),
            sigma_adu = float(sigma),
            amplitude = float(amp),
            success   = True,
            n_events  = n,
        )
    except (RuntimeError, ValueError):
        return _fail


# ══════════════════════════════════════════════════════════════════════════════
# Global peak fallback  (Iteration 0 only — ROOT GlobalPeakHist)
# ══════════════════════════════════════════════════════════════════════════════

def fit_global_peak(
        adu_values:    np.ndarray,   # (n_events,) filtered adu_sum
        col_indices:   np.ndarray,   # (n_events,) column of each event
        row_indices:   np.ndarray,   # (n_events,) row of each event
        roi_low:       float,
        roi_high:      float,
        mid_row:       int,
        n_params:      int  = 3,
        split_even_odd: bool = True,
        split_frame:    bool = False,
) -> GlobalPeakResult:
    """
    Fit global Gaussian peaks for use as column-fit fallback.

    Matches ROOT GlobalPeakHist logic:
      - Separate histograms per parity (if split_even_odd)
      - Separate histograms per half   (if split_frame)
      - Minimum MIN_HITS_GLOBAL entries required to trust fit

    Returns GlobalPeakResult with ppos[parity, half], sigma[parity, half].
    Falls back to ROI center / broad sigma if insufficient data.
    """
    roi_center  = 0.5 * (roi_low + roi_high)
    roi_sigma   = (roi_high - roi_low) / 6.0
    ppos  = np.full((2, 2), roi_center,  dtype=np.float64)
    sigma = np.full((2, 2), roi_sigma,   dtype=np.float64)

    n_parity = 2 if split_even_odd else 1
    n_halves = 2 if split_frame    else 1

    parity_arr = ((col_indices + 1) % 2).astype(np.int32)  # ROOT (col+1)%2
    half_arr   = (row_indices >= mid_row).astype(np.int32)

    for p in range(n_parity):
        for h in range(n_halves):
            pmask = (parity_arr == p) if n_parity == 2 else np.ones(len(adu_values), bool)
            hmask = (half_arr   == h) if n_halves == 2 else np.ones(len(adu_values), bool)
            vals  = adu_values[pmask & hmask]
            if len(vals) < MIN_HITS_GLOBAL:
                continue
            r = fit_peak(vals, roi_low, roi_high, n_params=n_params)
            if r.success:
                ppos[p, h]  = r.peak_adu
                sigma[p, h] = r.sigma_adu

    return GlobalPeakResult(ppos=ppos, sigma=sigma)


# ══════════════════════════════════════════════════════════════════════════════
# Per-column peak fitting  (ROOT Phase 2 orientation fit)
# ══════════════════════════════════════════════════════════════════════════════

def fit_all_columns(
        adu_values:    np.ndarray,   # (n_events,) filtered adu_sum
        col_indices:   np.ndarray,   # (n_events,) int
        row_indices:   np.ndarray,   # (n_events,) int/float
        roi_low:       float,
        roi_high:      float,
        n_cols:        int,
        mid_row:       int,
        global_peak:   GlobalPeakResult,
        n_params:      int  = 3,
        split_even_odd: bool = True,
        split_frame:    bool = False,
        min_hits:       int  = MIN_HITS_PER_COLUMN,
) -> ColumnPeakResult:
    """
    Fit per-column (and per-half) Gaussian peaks.

    Matches ROOT per-column FitHist loop with fallback logic.

    Parameters
    ----------
    adu_values   : filtered signal sums  (already ROI-cut by caller)
    col_indices  : column index per event (int, 0-based)
    row_indices  : row index per event (may be float COG)
    roi_low/high : ADU region of interest
    n_cols       : total number of columns
    mid_row      : row splitting bottom/top halves (SplitFrame)
    global_peak  : fallback from fit_global_peak
    n_params     : Gaussian parameter count (3, 4, or 5)
    split_even_odd: separate even/odd column parity
    split_frame  : separate top/bottom halves
    min_hits     : minimum events per column before using fallback

    Returns
    -------
    ColumnPeakResult  ppos/sigma shape (n_cols, n_halves)
    """
    n_halves = 2 if split_frame else 1
    ppos          = np.empty((n_cols, n_halves), dtype=np.float64)
    sigma         = np.empty((n_cols, n_halves), dtype=np.float64)
    used_fallback = np.zeros((n_cols, n_halves), dtype=bool)

    # Initialise with global fallback (ROOT initialises PPositions to ROI center)
    roi_center = 0.5 * (roi_low + roi_high)
    roi_sigma  = (roi_high - roi_low) / 6.0
    ppos[:]    = roi_center
    sigma[:]   = roi_sigma
    used_fallback[:] = True

    # Sort events by column for efficient per-column slicing
    sort_idx    = np.argsort(col_indices, kind="stable")
    s_cols      = col_indices[sort_idx]
    s_adu       = adu_values[sort_idx]
    s_rows      = row_indices[sort_idx]

    # Build start indices per column (ROOT StartIndices)
    start = np.searchsorted(s_cols, np.arange(n_cols),     side="left")
    end   = np.searchsorted(s_cols, np.arange(n_cols) + 1, side="left")

    n_parity = 2 if split_even_odd else 1

    for col in range(n_cols):
        parity = int((col + 1) % 2) if n_parity == 2 else 0
        col_adu  = s_adu[start[col]:end[col]]
        col_rows = s_rows[start[col]:end[col]]

        for h in range(n_halves):
            if n_halves == 2:
                hmask   = (col_rows >= mid_row) if h == 1 else (col_rows < mid_row)
                fit_adu = col_adu[hmask]
            else:
                fit_adu = col_adu

            # Fallback: too few events
            if len(fit_adu) < min_hits:
                ppos[col, h]  = global_peak.ppos[parity, h]
                sigma[col, h] = global_peak.sigma[parity, h]
                used_fallback[col, h] = True
                continue

            r = fit_peak(fit_adu, roi_low, roi_high, n_params=n_params)

            if r.success:
                ppos[col, h]  = r.peak_adu
                sigma[col, h] = r.sigma_adu
                used_fallback[col, h] = False
            else:
                # Fit failed — use global fallback (ROOT behaviour)
                ppos[col, h]  = global_peak.ppos[parity, h]
                sigma[col, h] = global_peak.sigma[parity, h]
                used_fallback[col, h] = True

    return ColumnPeakResult(ppos=ppos, sigma=sigma,
                            used_fallback=used_fallback)
