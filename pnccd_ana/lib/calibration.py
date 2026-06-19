"""
pnccd_ana.lib.calibration
==========================
Core mathematics for Fe-55 gain and CTI calibration.

Pipeline overview
─────────────────
  Phase 1 — Rough global gain (even / odd columns separately)
    · Single-pixel events only (grade 0)
    · Split by X parity → fit Mn Kα peak → G_even, G_odd  [eV / ADU]

  Phase 2 — Preliminary energy reconstruction
    · Apply G_even / G_odd to every pixel's raw ADU contribution
    · For split events: sum the per-pixel eV values  →  E_prelim  [eV]

  Phase 3 — CTI estimation and correction
    · Bin events by Y (row); fit Kα peak per bin
    · Linear model: E_meas(row) = E0 × (1 − row × CTI)
    · Correct every event:  E_cti = E_meas / (1 − row × CTI)

  Phase 4 — Per-column fine-gain
    · For each column X fit Kα peak in CTI-corrected data
    · f_col = MN_KALPHA_EV / E_fit_col
    · Final energy = raw_adu × G_rough(parity) × cti_factor(row) × f_col(X)

Physical constants
──────────────────
  Mn Kα  5895.0 eV   (used as calibration reference)
  Mn Kβ  6490.0 eV   (identified in plots but not used in fitting)

Array / coordinate convention (matches source_ana)
───────────────────────────────────────────────────
  Y = row index, 0 = bottom (readout side)
  X = column index, 0..W-1
  Charge drifts upward (toward higher Y) before collection.
  CTI loss therefore increases with Y: higher row → more loss → lower measured E.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ── Physical constants ────────────────────────────────────────────────────────

MN_KALPHA_EV: float = 5895.0   # Mn Kα reference energy [eV]
MN_KBETA_EV:  float = 6490.0   # Mn Kβ  (informational)

# Grades treated as "single-pixel" for rough gain (grade 0 only)
SINGLE_GRADES: frozenset = frozenset({0})

# All grades that carry a valid summed cluster energy
SPLIT_GRADES: frozenset = frozenset({1, 2, 3, 4,       # doubles
                                      5, 6, 7, 8,       # triples
                                      9, 10, 11, 12})   # quadruples


# ── Gaussian model ────────────────────────────────────────────────────────────

def _gaussian(x: np.ndarray,
              amp: float, mu: float, sigma: float) -> np.ndarray:
    """Simple Gaussian: amp × exp(−(x−μ)² / (2σ²))."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _gaussian_plus_linear(x: np.ndarray,
                           amp: float, mu: float, sigma: float,
                           a: float, b: float) -> np.ndarray:
    """Gaussian on a linear background: used for noisier per-column fits."""
    return _gaussian(x, amp, mu, sigma) + a * x + b


# ── Peak fitting ──────────────────────────────────────────────────────────────

@dataclass
class PeakFitResult:
    """Result of a single Gaussian peak fit."""
    peak_ev:    float          # fitted peak position [eV or ADU]
    sigma_ev:   float          # fitted σ (FWHM = 2.355 × sigma)
    amplitude:  float          # fitted amplitude
    n_events:   int            # events in the fit window
    success:    bool           # False when fit failed / too few events
    message:    str = ""       # human-readable status


def fit_peak(
        values:        np.ndarray,
        nominal:       float,
        window_frac:   float = 0.20,
        n_bins:        int   = 80,
        min_events:    int   = 30,
        with_bg:       bool  = False,
) -> PeakFitResult:
    """
    Fit a Gaussian to a histogram of *values* near *nominal*.

    Parameters
    ----------
    values      : 1-D array of event energies (ADU or eV)
    nominal     : expected peak position (used to set window)
    window_frac : half-width of fit window as fraction of nominal
                  e.g. 0.20 → window is [nominal×0.80, nominal×1.20]
    n_bins      : histogram bins inside the window
    min_events  : minimum number of events required; returns failure if fewer
    with_bg     : if True, fit Gaussian + linear background

    Returns
    -------
    PeakFitResult
    """
    lo = nominal * (1.0 - window_frac)
    hi = nominal * (1.0 + window_frac)
    mask = (values >= lo) & (values <= hi)
    n = int(mask.sum())

    if n < min_events:
        return PeakFitResult(peak_ev=np.nan, sigma_ev=np.nan,
                             amplitude=np.nan, n_events=n, success=False,
                             message=f"too few events ({n} < {min_events})")

    counts, edges = np.histogram(values[mask], bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Initial guesses
    p0_amp   = float(counts.max())
    p0_mu    = float(centres[counts.argmax()])
    p0_sigma = (hi - lo) / 6.0

    try:
        if with_bg:
            p0 = [p0_amp, p0_mu, p0_sigma, 0.0, 0.0]
            bounds = ([0, lo, 1e-3, -np.inf, -np.inf],
                      [np.inf, hi, hi - lo, np.inf, np.inf])
            popt, _ = curve_fit(_gaussian_plus_linear, centres, counts,
                                p0=p0, bounds=bounds, maxfev=5000)
            amp, mu, sigma = popt[0], popt[1], abs(popt[2])
        else:
            bounds = ([0, lo, 1e-3], [np.inf, hi, hi - lo])
            popt, _ = curve_fit(_gaussian, centres, counts,
                                p0=[p0_amp, p0_mu, p0_sigma],
                                bounds=bounds, maxfev=5000)
            amp, mu, sigma = popt[0], popt[1], abs(popt[2])

        return PeakFitResult(peak_ev=mu, sigma_ev=sigma,
                             amplitude=amp, n_events=n, success=True)

    except (RuntimeError, ValueError) as exc:
        return PeakFitResult(peak_ev=np.nan, sigma_ev=np.nan,
                             amplitude=np.nan, n_events=n, success=False,
                             message=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Rough global gain (even / odd columns)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoughGainResult:
    """Output of Phase 1."""
    g_even:       float          # gain for even columns [eV / ADU]
    g_odd:        float          # gain for odd columns  [eV / ADU]
    peak_even:    PeakFitResult  # fit details for even pool
    peak_odd:     PeakFitResult  # fit details for odd pool
    n_singles:    int            # total single-pixel events used


class RoughGainCalibrator:
    """
    Phase 1 — estimate per-parity (even/odd column) conversion gain.

    Only single-pixel events (grade 0) are used because their full charge
    is contained in one pixel, making ADU → eV conversion direct.

    Parameters
    ----------
    target_ev   : reference peak energy in eV (default Mn Kα = 5895 eV)
    window_frac : Gaussian fit window half-width as fraction of target_ev
    n_bins      : histogram bins for the fit
    min_events  : minimum events per parity pool required
    """

    def __init__(self,
                 target_ev:   float = MN_KALPHA_EV,
                 window_frac: float = 0.20,
                 n_bins:      int   = 100,
                 min_events:  int   = 100):
        self.target_ev   = target_ev
        self.window_frac = window_frac
        self.n_bins      = n_bins
        self.min_events  = min_events

    def run(self, events: np.ndarray) -> RoughGainResult:
        """
        Parameters
        ----------
        events : structured array with fields Y, X, grade, adu_sum, adu_seed

        Returns
        -------
        RoughGainResult
        """
        singles = events[np.isin(events["grade"], list(SINGLE_GRADES))]
        n_singles = len(singles)

        # Split by column parity
        even_mask = (singles["X"] % 2) == 0
        adu_even = singles["adu_sum"][even_mask]
        adu_odd  = singles["adu_sum"][~even_mask]

        # Fit peaks in ADU space; gain = target_ev / peak_adu
        fit_even = fit_peak(adu_even, nominal=self.target_ev / 1.0,
                            window_frac=self.window_frac,
                            n_bins=self.n_bins,
                            min_events=self.min_events)
        # nominal in ADU unknown until we fit, but window_frac is wide enough
        # For first pass: use adu_sum median as rough nominal if fit fails
        if not fit_even.success:
            med = float(np.median(adu_even)) if len(adu_even) else 1000.0
            fit_even = fit_peak(adu_even, nominal=med,
                                window_frac=self.window_frac,
                                n_bins=self.n_bins,
                                min_events=self.min_events)

        fit_odd = fit_peak(adu_odd, nominal=self.target_ev / 1.0,
                           window_frac=self.window_frac,
                           n_bins=self.n_bins,
                           min_events=self.min_events)
        if not fit_odd.success:
            med = float(np.median(adu_odd)) if len(adu_odd) else 1000.0
            fit_odd = fit_peak(adu_odd, nominal=med,
                               window_frac=self.window_frac,
                               n_bins=self.n_bins,
                               min_events=self.min_events)

        g_even = (self.target_ev / fit_even.peak_ev
                  if fit_even.success and fit_even.peak_ev > 0 else np.nan)
        g_odd  = (self.target_ev / fit_odd.peak_ev
                  if fit_odd.success and fit_odd.peak_ev > 0 else np.nan)

        if np.isnan(g_even) or np.isnan(g_odd):
            warnings.warn(
                "Rough gain fit failed for one or both parities. "
                f"even: {fit_even.message}  odd: {fit_odd.message}",
                RuntimeWarning, stacklevel=2)

        return RoughGainResult(g_even=g_even, g_odd=g_odd,
                               peak_even=fit_even, peak_odd=fit_odd,
                               n_singles=n_singles)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Preliminary energy reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def apply_rough_gain(events: np.ndarray,
                     g_even: float,
                     g_odd:  float) -> np.ndarray:
    """
    Convert adu_sum → preliminary energy [eV] using per-parity rough gains.

    For split events the adu_sum already contains the correct pixel sum
    (computed by find_events / pattern_recognition), so multiplying by the
    gain of the *seed* pixel's column parity is an approximation that is
    corrected in Phase 4.  The error is small because adjacent pixels share
    the same parity pattern within a cluster.

    Parameters
    ----------
    events : structured array (Y, X, grade, adu_sum, adu_seed)
    g_even : eV/ADU for even-column seeds
    g_odd  : eV/ADU for odd-column seeds

    Returns
    -------
    e_prelim : float32 (N,) — preliminary energy in eV
    """
    g = np.where((events["X"] % 2) == 0, g_even, g_odd).astype(np.float32)
    return (events["adu_sum"].astype(np.float32) * g)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — CTI estimation and correction
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CtiResult:
    """Output of Phase 3."""
    cti:          float                      # CTI coefficient [1/pixel]
    e0:           float                      # extrapolated peak at row=0 [eV]
    row_bins:     np.ndarray                 # bin centre rows used in fit
    peak_per_bin: np.ndarray                 # measured peak eV per bin
    peak_success: np.ndarray                 # bool — which bins converged
    fit_residuals: np.ndarray                # measured − model [eV]
    n_bins_used:  int                        # bins with successful fits


class CtiCalibrator:
    """
    Phase 3 — estimate Charge Transfer Inefficiency from row-dependent
    peak position degradation.

    Physical model
    ──────────────
    During readout the charge packet traverses (H − 1 − Y) transfers from
    its row to the readout register at Y=0.  Each transfer loses a fraction
    CTI of the charge.  For small CTI:

        E_meas(row) ≈ E0 × (1 − (H − 1 − Y) × CTI)

    where H is the sensor height.  We use Y directly (larger Y = fewer
    transfers = less loss) so the measured energy *increases* with Y for
    positive CTI — which matches what we fit below.

    If instead the data shows energy *decreasing* with Y, CTI is negative
    in this convention (possible if there is a gain gradient rather than
    true CTI).  The code handles both cases.

    Parameters
    ----------
    row_bin_size : number of rows per bin for peak-vs-row fitting
    target_ev    : reference peak energy in eV
    window_frac  : fit window half-width fraction
    n_bins_hist  : histogram bins per row-bin fit
    min_events   : minimum events per row bin
    grade_filter : which grades to include (None = all)
    """

    def __init__(self,
                 row_bin_size: int   = 64,
                 target_ev:   float = MN_KALPHA_EV,
                 window_frac: float = 0.15,
                 n_bins_hist: int   = 60,
                 min_events:  int   = 50,
                 grade_filter: list[int] | None = None):
        self.row_bin_size = row_bin_size
        self.target_ev    = target_ev
        self.window_frac  = window_frac
        self.n_bins_hist  = n_bins_hist
        self.min_events   = min_events
        self.grade_filter = grade_filter

    def estimate(self,
                 events:    np.ndarray,
                 e_prelim:  np.ndarray,
                 n_rows:    int) -> CtiResult:
        """
        Fit peak position vs row and extract CTI.

        Parameters
        ----------
        events   : structured array (Y, X, grade, adu_sum, adu_seed)
        e_prelim : float32 (N,) — preliminary energy from Phase 2
        n_rows   : total sensor height (Y dimension)

        Returns
        -------
        CtiResult
        """
        # Optionally restrict grades
        if self.grade_filter is not None:
            mask = np.isin(events["grade"], self.grade_filter)
            ev   = events[mask]
            ep   = e_prelim[mask]
        else:
            ev = events
            ep = e_prelim

        y_coords = ev["Y"].astype(int)

        # Build row bins
        bin_edges  = np.arange(0, n_rows + self.row_bin_size, self.row_bin_size)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        n_rbins     = len(bin_centres)

        peak_arr    = np.full(n_rbins, np.nan)
        success_arr = np.zeros(n_rbins, dtype=bool)

        for i, (y0, y1) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            bm = (y_coords >= y0) & (y_coords < y1)
            if bm.sum() < self.min_events:
                continue
            res = fit_peak(ep[bm], nominal=self.target_ev,
                           window_frac=self.window_frac,
                           n_bins=self.n_bins_hist,
                           min_events=self.min_events)
            if res.success:
                peak_arr[i]    = res.peak_ev
                success_arr[i] = True

        # Linear fit: E_meas = E0 × (1 − row × CTI)
        # Rearranged: E_meas = E0 − E0×CTI × row
        # → linear in row with intercept=E0 and slope=−E0×CTI
        good = success_arr & np.isfinite(peak_arr)
        n_good = int(good.sum())

        if n_good < 2:
            warnings.warn(
                f"CTI fit: only {n_good} valid row bins — cannot fit slope. "
                "CTI set to 0.", RuntimeWarning, stacklevel=2)
            e0  = float(np.nanmedian(peak_arr)) if np.any(np.isfinite(peak_arr)) else self.target_ev
            cti = 0.0
        else:
            rows_good   = bin_centres[good]
            peaks_good  = peak_arr[good]
            coeffs      = np.polyfit(rows_good, peaks_good, 1)  # [slope, intercept]
            slope, e0   = float(coeffs[0]), float(coeffs[1])
            # slope = −E0 × CTI  →  CTI = −slope / E0
            cti = -slope / e0 if e0 != 0 else 0.0

        # Residuals
        model        = e0 * (1.0 - bin_centres * cti)
        residuals    = peak_arr - model   # NaN where fit failed

        return CtiResult(cti=cti, e0=e0,
                         row_bins=bin_centres,
                         peak_per_bin=peak_arr,
                         peak_success=success_arr,
                         fit_residuals=residuals,
                         n_bins_used=n_good)

    def correct(self,
                events:   np.ndarray,
                e_prelim: np.ndarray,
                cti:      float,
                e0:       float) -> np.ndarray:
        """
        Apply CTI correction.

            E_cti = E_prelim / (1 − row × CTI)

        Parameters
        ----------
        events   : structured array
        e_prelim : float32 (N,) preliminary energy [eV]
        cti      : CTI coefficient [1/pixel]
        e0       : fitted peak at row=0 [eV] (unused in correction formula,
                   kept for signature symmetry)

        Returns
        -------
        e_cti : float32 (N,) CTI-corrected energy [eV]
        """
        rows    = events["Y"].astype(np.float32)
        denom   = 1.0 - rows * float(cti)
        # Guard against division by zero or negative denominators
        safe    = np.where(np.abs(denom) > 1e-6, denom, 1.0).astype(np.float32)
        return (e_prelim / safe).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Per-column fine-gain calibration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnGainResult:
    """Output of Phase 4."""
    f_col:       np.ndarray   # float32 (n_cols,) — per-column scale factors
    peak_col:    np.ndarray   # float32 (n_cols,) — fitted peak per column [eV]
    n_events_col: np.ndarray  # int32   (n_cols,) — events used per column
    success_col: np.ndarray   # bool    (n_cols,) — which columns converged
    n_cols_fit:  int          # columns with successful fits


class ColumnGainCalibrator:
    """
    Phase 4 — per-column fine-gain factors using CTI-corrected energies.

    For columns with too few events the factor defaults to 1.0 (no correction).

    Parameters
    ----------
    target_ev    : reference peak energy in eV
    window_frac  : Gaussian fit window half-width fraction
    n_bins       : histogram bins per column fit
    min_events   : minimum events required for a column fit
    with_bg      : fit Gaussian + linear background (more robust for noisy cols)
    """

    def __init__(self,
                 target_ev:  float = MN_KALPHA_EV,
                 window_frac: float = 0.15,
                 n_bins:     int   = 60,
                 min_events: int   = 30,
                 with_bg:    bool  = False):
        self.target_ev    = target_ev
        self.window_frac  = window_frac
        self.n_bins       = n_bins
        self.min_events   = min_events
        self.with_bg      = with_bg

    def run(self,
            events: np.ndarray,
            e_cti:  np.ndarray,
            n_cols: int,
            grade_filter: list[int] | None = None) -> ColumnGainResult:
        """
        Fit Kα peak per column in CTI-corrected data.

        Parameters
        ----------
        events       : structured array (Y, X, grade, adu_sum, adu_seed)
        e_cti        : float32 (N,) CTI-corrected energy [eV]
        n_cols       : total number of detector columns
        grade_filter : which grades to include (None → singles only)

        Returns
        -------
        ColumnGainResult
        """
        if grade_filter is None:
            grade_filter = list(SINGLE_GRADES)

        mask = np.isin(events["grade"], grade_filter)
        ev   = events[mask]
        ec   = e_cti[mask]

        x_coords = ev["X"].astype(int)

        f_col        = np.ones(n_cols,  dtype=np.float32)
        peak_col     = np.full(n_cols,  np.nan, dtype=np.float32)
        n_ev_col     = np.zeros(n_cols, dtype=np.int32)
        success_col  = np.zeros(n_cols, dtype=bool)

        for col in range(n_cols):
            cm = x_coords == col
            n_ev_col[col] = int(cm.sum())
            if n_ev_col[col] < self.min_events:
                continue
            res = fit_peak(ec[cm], nominal=self.target_ev,
                           window_frac=self.window_frac,
                           n_bins=self.n_bins,
                           min_events=self.min_events,
                           with_bg=self.with_bg)
            if res.success and res.peak_ev > 0:
                f_col[col]       = self.target_ev / res.peak_ev
                peak_col[col]    = res.peak_ev
                success_col[col] = True

        n_fit = int(success_col.sum())
        if n_fit == 0:
            warnings.warn("Column gain: no columns had sufficient events to fit.",
                          RuntimeWarning, stacklevel=2)

        return ColumnGainResult(f_col=f_col, peak_col=peak_col,
                                n_events_col=n_ev_col,
                                success_col=success_col,
                                n_cols_fit=n_fit)


# ══════════════════════════════════════════════════════════════════════════════
# Combined application: compute final calibrated energy for an event array
# ══════════════════════════════════════════════════════════════════════════════

def apply_full_calibration(
        events: np.ndarray,
        g_even: float,
        g_odd:  float,
        cti:    float,
        f_col:  np.ndarray,
) -> np.ndarray:
    """
    Apply the complete gain + CTI calibration chain to an event array.

    Final energy [eV] = adu_sum × G_rough(parity) × cti_factor(row) × f_col(X)

    where:
        G_rough = g_even if X even else g_odd
        cti_factor(row) = 1 / (1 − row × CTI)
        f_col(X) = per-column residual scale factor from Phase 4

    Parameters
    ----------
    events  : structured array (Y, X, grade, adu_sum, adu_seed)
    g_even  : Phase-1 gain for even columns [eV/ADU]
    g_odd   : Phase-1 gain for odd  columns [eV/ADU]
    cti     : Phase-3 CTI coefficient [1/pixel]
    f_col   : Phase-4 per-column factors, float32 (n_cols,)

    Returns
    -------
    energy_ev : float32 (N,) — fully calibrated energy per event
    """
    x = events["X"].astype(int)
    y = events["Y"].astype(np.float32)

    g_rough     = np.where((x % 2) == 0, g_even, g_odd).astype(np.float32)
    cti_factor  = 1.0 / np.maximum(1.0 - y * float(cti), 1e-6)
    col_factor  = f_col[x]

    return (events["adu_sum"].astype(np.float32)
            * g_rough * cti_factor * col_factor).astype(np.float32)
