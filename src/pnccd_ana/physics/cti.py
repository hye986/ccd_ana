"""
pnccd_ana.physics.cti
======================
Charge Transfer Inefficiency (CTI) calibration.

Phase 3 — CTI estimation and correction
  · Bin events by Y (row); fit Kα peak per bin
  · Linear model: E_meas(row) = E0 × (1 − row × CTI)
  · Correct every event:  E_cti = E_meas / (1 − row × CTI)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .gain import fit_peak, MN_KALPHA_EV


# ── CTI calibration ────────────────────────────────────────────────────────────

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
