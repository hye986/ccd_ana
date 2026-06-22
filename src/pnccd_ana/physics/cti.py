"""
pnccd_ana.physics.cti
=====================
CTE map construction and iterative per-column CTE fitting.

Matches ROOT HStepGainMapCCDHLL Phase 3:
  - build_cte_map    : propagate CTE correction factors row-by-row
  - cte_model        : expected signal vs row (ROOT CTEFunction)
  - fit_cte_column   : iterative fit (up to 10 sub-iterations, relax=0.5)
  - fit_all_columns_cte : outer loop over all columns

Sensor geometry
───────────────
  Bottom half  Y = 0 .. RowIndexROBorder-1
    Frame store: Y = 0 .. NRowFS-1          (CTE = CTEfs_b)
    Image area:  Y = NRowFS .. ROBorder-1   (CTE = CTEim_b)

  Top half (SplitFrame only)  Y = ROBorder .. RowCount-1
    Image area:  Y = ROBorder .. RowCount-NRowFS-1   (CTE = CTEim_t)
    Frame store: Y = RowCount-NRowFS .. RowCount-1   (CTE = CTEfs_t)
    Readout runs from Y=RowCount-1 downward → CTEMap filled top-to-bottom.

  CTEMap[row, col] = cumulative transfer efficiency for a charge
  originating at that row reaching the readout register.
  Signal corrected = raw_signal × CTEMap[row, col].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CTEParams:
    """CTE parameters for one column."""
    cte_fs_b: float = 1.0
    cte_im_b: float = 1.0
    cte_fs_t: float = 1.0
    cte_im_t: float = 1.0
    ppos_b:   float = 0.0   # peak position bottom half [ADU]
    ppos_t:   float = 0.0   # peak position top half [ADU]


@dataclass
class CTEFitResult:
    """Result of iterative CTE fit for one column."""
    params:     CTEParams
    converged:  bool
    n_iter:     int
    fit_good_b: bool   # bottom half fit acceptable
    fit_good_t: bool   # top half fit acceptable (always True if not SplitFrame)


@dataclass
class CTEMapResult:
    """Full-detector outputs after fitting all columns."""
    gain_map:     np.ndarray   # (n_rows, n_cols) float64  eV/ADU
    cte_map:      np.ndarray   # (n_rows, n_cols) float64  cumulative CTE
    bad_gain_map: np.ndarray   # (n_rows, n_cols) int8
                               # 0=good, 1=fallback, 2=kept previous


# ══════════════════════════════════════════════════════════════════════════════
# CTE map builder  (ROOT CTEMap[row*ColCount+col] array)
# ══════════════════════════════════════════════════════════════════════════════

def build_cte_map_column(
        n_rows:        int,
        params:        CTEParams,
        n_row_fs:      int,
        row_border:    int,
        split_frame:   bool,
) -> np.ndarray:
    """
    Build a 1-D CTE correction array for one column.

    CTEMap[row] = cumulative transfer efficiency from row to register.
    Charge at row 0 (bottom register) needs 0 transfers → CTE=1.

    Parameters
    ----------
    n_rows     : total number of rows
    params     : current CTE parameters for this column
    n_row_fs   : number of frame-store rows (0 if FullFrame=False)
    row_border : RowIndexROBorder (= n_rows//2 for SplitFrame, else n_rows)
    split_frame: sensor has two readout directions

    Returns
    -------
    cte_col : float64 (n_rows,)
    """
    cte_col = np.zeros(n_rows, dtype=np.float64)

    # ── Bottom half: readout from row 0 upward ────────────────────────────────
    cte_col[0] = 1.0
    row = 1
    # Frame store area (bottom)
    while row < n_row_fs:
        cte_col[row] = cte_col[row - 1] / params.cte_fs_b
        row += 1
    # Image area (bottom)
    while row < row_border:
        cte_col[row] = cte_col[row - 1] / params.cte_im_b
        row += 1

    # ── Top half: readout from row (n_rows-1) downward ────────────────────────
    if split_frame:
        cte_col[n_rows - 1] = 1.0
        row = n_rows - 1
        # Frame store area (top)
        while row > n_rows - n_row_fs:
            cte_col[row - 1] = cte_col[row] / params.cte_fs_t
            row -= 1
        # Image area (top)
        while row > row_border:
            cte_col[row - 1] = cte_col[row] / params.cte_im_t
            row -= 1

    return cte_col


def build_cte_map(
        n_rows:       int,
        n_cols:       int,
        params_cols:  list[CTEParams],
        n_row_fs:     int,
        row_border:   int,
        split_frame:  bool,
) -> np.ndarray:
    """
    Build the full (n_rows, n_cols) CTE map.

    Parameters
    ----------
    params_cols : list of CTEParams, one per column

    Returns
    -------
    cte_map : float64 (n_rows, n_cols)
    """
    cte_map = np.zeros((n_rows, n_cols), dtype=np.float64)
    for col, params in enumerate(params_cols):
        cte_map[:, col] = build_cte_map_column(
            n_rows, params, n_row_fs, row_border, split_frame)
    return cte_map


# ══════════════════════════════════════════════════════════════════════════════
# CTE model function  (ROOT CTEFunction / HUtils::CTEFunction)
# ══════════════════════════════════════════════════════════════════════════════

def cte_model(
        rows:       np.ndarray,
        ppos_b:     float,
        ppos_t:     float,
        cte_fs_b:   float,
        cte_im_b:   float,
        cte_fs_t:   float,
        cte_im_t:   float,
        n_row_fs:   float,    # fixed — passed as float for scipy compat
        row_border: float,    # fixed
        n_rows:     float,    # fixed
) -> np.ndarray:
    """
    Expected CTE-attenuated signal at each row position.

    For each row r:
      bottom half (r < row_border):
        signal = ppos_b × cte_fs_b^min(r, n_row_fs)
                        × cte_im_b^max(0, r - n_row_fs)
      top half (r >= row_border, SplitFrame only):
        transfers_from_top = (n_rows - 1) - r
        signal = ppos_t × cte_fs_t^min(transfers, n_row_fs)
                        × cte_im_t^max(0, transfers - n_row_fs)

    Parameters match ROOT CTEFitFunction parameter order:
      0: ppos_b   1: ppos_t
      2: CTEfs_b  3: CTEim_b  4: CTEfs_t  5: CTEim_t
      6: NRowFS   7: RowIndexROBorder   8: RowCount
    """
    n_row_fs   = int(round(n_row_fs))
    row_border = int(round(row_border))
    n_rows     = int(round(n_rows))
    rows       = np.asarray(rows, dtype=np.float64)
    out        = np.empty_like(rows)

    bot = rows < row_border
    top = ~bot

    if bot.any():
        r_b     = rows[bot]
        fs_xfrs = np.minimum(r_b, n_row_fs)
        im_xfrs = np.maximum(0.0, r_b - n_row_fs)
        out[bot] = (ppos_b
                    * np.power(cte_fs_b, fs_xfrs)
                    * np.power(cte_im_b, im_xfrs))

    if top.any():
        r_t     = rows[top]
        xfrs    = (n_rows - 1) - r_t
        fs_xfrs = np.minimum(xfrs, n_row_fs)
        im_xfrs = np.maximum(0.0, xfrs - n_row_fs)
        out[top] = (ppos_t
                    * np.power(cte_fs_t, fs_xfrs)
                    * np.power(cte_im_t, im_xfrs))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Single-column iterative CTE fit  (ROOT inner 10-iteration loop)
# ══════════════════════════════════════════════════════════════════════════════

_CONVERGE_PPOS_REL  = 0.01       # ROOT: 1% relative change in peak position
_CONVERGE_CTE_ABS   = 5e-6       # ROOT: |CTE - 1| < 5e-6
_ROI_SIGMA_MULT     = 2.0        # ROOT RoiSigmaMult
_MIN_EVENTS_FIT     = 10         # ROOT: skip if EventCounter < 10


def fit_cte_column(
        rows:           np.ndarray,    # event rows in this column (float, COG)
        adu_corrected:  np.ndarray,    # CTE-corrected signal sums
        ppos_b_init:    float,         # initial peak position bottom
        ppos_t_init:    float,         # initial peak position top (SplitFrame)
        roi_low:        float,
        roi_high:       float,
        n_row_fs:       int,
        row_border:     int,
        n_rows:         int,
        split_frame:    bool,
        full_frame:     bool,
        relax:          float = 0.5,
        max_iter:       int   = 10,
        n_roi_bins:     int   = 200,
) -> CTEFitResult:
    """
    Iterative CTE fit for one column.

    Matches ROOT inner loop:
      1. Build CTEMap for current CTE params
      2. Apply CTE correction to signal sums
      3. Determine narrow ROI from Gaussian fit to corrected histogram
      4. Fit cte_model to (row, corrected_signal) scatter
      5. Update CTE with relaxation:  CTE *= fitted_CTE^relax
      6. Converge when peak positions stable and all CTE ≈ 1

    Parameters
    ----------
    rows          : row positions (float for COG events)
    adu_corrected : CTE-corrected ADU sums (updated each sub-iteration)
    ppos_b_init   : initial peak position estimate bottom [ADU]
    ppos_t_init   : initial peak position estimate top [ADU]
    roi_low/high  : hard ADU bounds
    n_row_fs      : frame-store row count (0 = no frame store)
    row_border    : row index separating bottom/top halves
    n_rows        : total row count
    split_frame   : sensor has bidirectional readout
    full_frame    : frame-store CTE also fitted
    relax         : relaxation factor on CTE update (ROOT default 0.5)
    max_iter      : maximum sub-iterations (ROOT default 10)

    Returns
    -------
    CTEFitResult
    """
    params = CTEParams(
        cte_fs_b = 1.0,
        cte_im_b = 1.0,
        cte_fs_t = 1.0,
        cte_im_t = 1.0,
        ppos_b   = ppos_b_init,
        ppos_t   = ppos_t_init,
    )

    # Fixed geometry parameters passed to cte_model
    geom = (float(n_row_fs), float(row_border), float(n_rows))

    # Build initial CTE map for this column
    cte_col = build_cte_map_column(n_rows, params, n_row_fs,
                                   row_border, split_frame)
    adu_corr = adu_corrected.copy()

    converged  = False
    n_iter_run = 0
    roi_lo, roi_hi = roi_low, roi_high

    for iteration in range(max_iter):
        n_iter_run = iteration + 1
        old_ppos_b = params.ppos_b
        old_ppos_t = params.ppos_t

        # Apply CTE correction to signal sums
        adu_corr = adu_corrected * cte_col[np.clip(
            rows.astype(int), 0, n_rows - 1)]

        # ── Determine narrow ROI from Gaussian fit to corrected histogram ──────
        in_roi = (adu_corr > roi_low) & (adu_corr < roi_high)
        if in_roi.sum() >= _MIN_EVENTS_FIT and not split_frame:
            # Gaussian fit to histogram for ROI narrowing (ROOT: FitHist "gaus")
            vals_roi = adu_corr[in_roi]
            counts, edges = np.histogram(vals_roi, bins=n_roi_bins,
                                          range=(roi_low, roi_high))
            # Smooth (ROOT: Smooth(10) ≈ simple moving average)
            kernel = np.ones(5) / 5.0
            counts_s = np.convolve(counts.astype(float), kernel, mode="same")
            centers = 0.5 * (edges[:-1] + edges[1:])
            try:
                pk = int(np.argmax(counts_s))
                p0 = [float(counts_s[pk]), float(centers[pk]),
                      (roi_high - roi_low) / 6.0]
                from scipy.optimize import curve_fit as _cf
                from .gain import _gauss
                popt, _ = _cf(_gauss, centers, counts_s, p0=p0, maxfev=3000)
                mu, sig = popt[1], abs(popt[2])
                if roi_low < mu < roi_high and sig > 0:
                    roi_lo = mu - _ROI_SIGMA_MULT * sig
                    roi_hi = mu + _ROI_SIGMA_MULT * sig
                    roi_lo = max(roi_lo, roi_low)
                    roi_hi = min(roi_hi, roi_high)
            except (RuntimeError, ValueError):
                roi_lo, roi_hi = roi_low, roi_high
        else:
            # SplitFrame: use orientation peak sigma for ROI (ROOT fallback)
            mu  = params.ppos_b
            sig_est = (roi_high - roi_low) / 6.0
            roi_lo = max(mu - _ROI_SIGMA_MULT * sig_est, roi_low)
            roi_hi = min(mu + _ROI_SIGMA_MULT * sig_est, roi_high)

        # ── Select events in narrow ROI ───────────────────────────────────────
        fit_mask = (adu_corr > roi_lo) & (adu_corr < roi_hi)
        n_fit    = int(fit_mask.sum())
        if n_fit < _MIN_EVENTS_FIT:
            break

        fit_rows = rows[fit_mask]
        fit_adu  = adu_corr[fit_mask]

        # ── Fit cte_model to (row, signal) scatter ────────────────────────────
        # Build parameter bounds matching ROOT FixParameter / SetParLimits
        p0     = [params.ppos_b, params.ppos_t,
                  params.cte_fs_b, params.cte_im_b,
                  params.cte_fs_t, params.cte_im_t]
        lo_b   = [roi_low, roi_low if split_frame else 0.9,
                  0.0 if full_frame else 1.0 - 1e-9,
                  0.0,
                  0.0 if (split_frame and full_frame) else 1.0 - 1e-9,
                  0.0 if split_frame else 1.0 - 1e-9]
        hi_b   = [roi_high, roi_high if split_frame else 1.1,
                  1.0,
                  1.0,
                  1.0,
                  1.0 if split_frame else 1.0 + 1e-9]

        try:
            popt, _ = curve_fit(
                lambda r, pb, pt, cfsb, cimb, cfst, cimt:
                    cte_model(r, pb, pt, cfsb, cimb, cfst, cimt, *geom),
                fit_rows, fit_adu,
                p0=p0,
                bounds=(lo_b, hi_b),
                maxfev=10000,
            )
            fitted_ppos_b, fitted_ppos_t = popt[0], popt[1]
            fitted_fs_b, fitted_im_b     = popt[2], popt[3]
            fitted_fs_t, fitted_im_t     = popt[4], popt[5]
        except (RuntimeError, ValueError):
            break

        # ── Update CTE with relaxation (ROOT: CTE *= fitted^relax) ───────────
        params.cte_fs_b *= fitted_fs_b ** relax
        params.cte_im_b *= fitted_im_b ** relax
        params.cte_fs_t *= fitted_fs_t ** relax
        params.cte_im_t *= fitted_im_t ** relax
        params.ppos_b    = fitted_ppos_b
        params.ppos_t    = fitted_ppos_t

        # Rebuild CTE map with updated params
        cte_col = build_cte_map_column(n_rows, params, n_row_fs,
                                       row_border, split_frame)

        # ── Convergence check (ROOT convergence criteria) ─────────────────────
        ppos_b_ok = abs(fitted_ppos_b - old_ppos_b) / max(abs(old_ppos_b), 1e-9) < _CONVERGE_PPOS_REL
        ppos_t_ok = abs(fitted_ppos_t - old_ppos_t) / max(abs(old_ppos_t), 1e-9) < _CONVERGE_PPOS_REL
        cte_ok    = (abs(fitted_fs_b - 1.0) < _CONVERGE_CTE_ABS and
                     abs(fitted_im_b - 1.0) < _CONVERGE_CTE_ABS and
                     abs(fitted_fs_t - 1.0) < _CONVERGE_CTE_ABS and
                     abs(fitted_im_t - 1.0) < _CONVERGE_CTE_ABS)
        if ppos_b_ok and ppos_t_ok and cte_ok:
            converged = True
            break

    # ── Assess fit quality per half ───────────────────────────────────────────
    fit_good_b = (roi_low < params.ppos_b < roi_high and
                  params.cte_im_b > 0.995 and
                  params.cte_fs_b > 0.995)
    fit_good_t = True
    if split_frame:
        fit_good_t = (roi_low < params.ppos_t < roi_high and
                      params.cte_im_t > 0.995 and
                      params.cte_fs_t > 0.995)

    return CTEFitResult(params=params, converged=converged,
                        n_iter=n_iter_run,
                        fit_good_b=fit_good_b, fit_good_t=fit_good_t)


# ══════════════════════════════════════════════════════════════════════════════
# Outer loop: fit all columns, fill gain_map and cte_map
# ══════════════════════════════════════════════════════════════════════════════

def fit_all_columns_cte(
        adu_values:    np.ndarray,   # (n_events,) filtered signal sums
        col_indices:   np.ndarray,   # (n_events,) int column
        row_indices:   np.ndarray,   # (n_events,) float (COG for splits)
        col_peaks:     "ColumnPeakResult",  # from gain.fit_all_columns
        global_peak:   "GlobalPeakResult",  # fallback from gain.fit_global_peak
        calib_energy:  float,        # [eV] e.g. MN_KALPHA_EV
        prev_gain_map: np.ndarray,   # (n_rows, n_cols) — zeros on first call
        n_rows:        int,
        n_cols:        int,
        n_row_fs:      int,
        row_border:    int,
        split_frame:   bool,
        full_frame:    bool,
        split_even_odd: bool,
        roi_low:       float,
        roi_high:      float,
        relax:         float = 0.5,
        max_iter:      int   = 10,
) -> CTEMapResult:
    """
    Fit CTE and gain for all columns, filling gain_map and cte_map.

    Matches ROOT outer column loop including:
      - Per-column CTE fit (fit_cte_column)
      - gain[row] = calib_energy / ppos × CTE_propagation
      - Fallback filling for failed columns (last iteration only)
      - BadGainMap quality flags

    Parameters
    ----------
    adu_values    : event signal sums (already filtered by calibrate.py)
    col_indices   : event column (seed column or COG column)
    row_indices   : event row   (seed row or COG row, float)
    col_peaks     : ColumnPeakResult from gain.fit_all_columns
    global_peak   : GlobalPeakResult from gain.fit_global_peak (fallback)
    calib_energy  : calibration line energy [eV]
    prev_gain_map : gain map from previous outer iteration (zeros=first time)
    n_rows/n_cols : detector dimensions
    n_row_fs      : frame-store row count (0 if full_frame=False)
    row_border    : n_rows//2 for SplitFrame, else n_rows
    split_frame   : bidirectional readout
    full_frame    : frame-store CTE fitted
    split_even_odd: separate parity tracking for mean_gain
    roi_low/high  : ADU bounds
    relax         : CTE update relaxation (ROOT default 0.5)
    max_iter      : CTE sub-iterations per column (ROOT default 10)

    Returns
    -------
    CTEMapResult  (gain_map, cte_map, bad_gain_map)
    """
    gain_map     = prev_gain_map.copy()
    cte_map      = np.zeros((n_rows, n_cols), dtype=np.float64)
    bad_gain_map = np.zeros((n_rows, n_cols), dtype=np.int8)

    n_parity = 2 if split_even_odd else 1
    # Running mean gain per parity (for fallback)
    mean_gain_sum = np.zeros(n_parity, dtype=np.float64)
    mean_gain_cnt = np.zeros(n_parity, dtype=np.int64)

    # Sort events by column for fast per-column lookup
    sort_idx = np.argsort(col_indices, kind="stable")
    s_cols   = col_indices[sort_idx]
    s_adu    = adu_values[sort_idx]
    s_rows   = row_indices[sort_idx]

    col_start = np.searchsorted(s_cols, np.arange(n_cols),     side="left")
    col_end   = np.searchsorted(s_cols, np.arange(n_cols) + 1, side="left")

    n_halves = 2 if split_frame else 1

    for col in range(n_cols):
        parity   = int((col + 1) % 2) if n_parity == 2 else 0
        col_adu  = s_adu[col_start[col]:col_end[col]]
        col_rows = s_rows[col_start[col]:col_end[col]]

        if len(col_adu) == 0:
            # No events for this column — will be handled as fallback below
            _fill_fallback_column(
                col, n_rows, row_border, n_halves,
                gain_map, cte_map, bad_gain_map,
                prev_gain_map, global_peak, calib_energy, parity)
            continue

        # Initial CTE-corrected signal = raw signal (CTE=1.0 at start)
        adu_corr = col_adu.copy()

        result = fit_cte_column(
            rows          = col_rows,
            adu_corrected = adu_corr,
            ppos_b_init   = col_peaks.ppos[col, 0],
            ppos_t_init   = col_peaks.ppos[col, min(1, n_halves-1)],
            roi_low       = roi_low,
            roi_high      = roi_high,
            n_row_fs      = n_row_fs,
            row_border    = row_border,
            n_rows        = n_rows,
            split_frame   = split_frame,
            full_frame    = full_frame,
            relax         = relax,
            max_iter      = max_iter,
        )

        params = result.params

        # ── Fill bottom half ──────────────────────────────────────────────────
        if result.fit_good_b:
            last_gain = calib_energy / params.ppos_b
            last_cte  = 1.0
            mean_gain_sum[parity] += last_gain
            mean_gain_cnt[parity] += 1

            gain_map[0, col]     = last_gain
            cte_map[0, col]      = last_cte
            bad_gain_map[0, col] = 0

            row = 1
            while row < n_row_fs:
                last_cte  *= params.cte_fs_b
                last_gain /= params.cte_fs_b
                gain_map[row, col]     = last_gain
                cte_map[row, col]      = last_cte
                bad_gain_map[row, col] = 0
                row += 1
            while row < row_border:
                last_cte  *= params.cte_im_b
                last_gain /= params.cte_im_b
                gain_map[row, col]     = last_gain
                cte_map[row, col]      = last_cte
                bad_gain_map[row, col] = 0
                row += 1
        else:
            _fill_fallback_column_half(
                col, 0, row_border, "bottom",
                gain_map, cte_map, bad_gain_map,
                prev_gain_map, global_peak, calib_energy, parity)

        # ── Fill top half (SplitFrame) ────────────────────────────────────────
        if split_frame:
            if result.fit_good_t:
                last_gain = calib_energy / params.ppos_t
                last_cte  = 1.0
                mean_gain_sum[parity] += last_gain
                mean_gain_cnt[parity] += 1

                gain_map[n_rows - 1, col]     = last_gain
                cte_map[n_rows - 1, col]      = last_cte
                bad_gain_map[n_rows - 1, col] = 0

                row = n_rows - 1
                while row > n_rows - n_row_fs:
                    last_cte  *= params.cte_fs_t
                    last_gain /= params.cte_fs_t
                    gain_map[row - 1, col]     = last_gain
                    cte_map[row - 1, col]      = last_cte
                    bad_gain_map[row - 1, col] = 0
                    row -= 1
                while row > row_border:
                    last_cte  *= params.cte_im_t
                    last_gain /= params.cte_im_t
                    gain_map[row - 1, col]     = last_gain
                    cte_map[row - 1, col]      = last_cte
                    bad_gain_map[row - 1, col] = 0
                    row -= 1
            else:
                _fill_fallback_column_half(
                    col, row_border, n_rows, "top",
                    gain_map, cte_map, bad_gain_map,
                    prev_gain_map, global_peak, calib_energy, parity)

    return CTEMapResult(gain_map=gain_map,
                        cte_map=cte_map,
                        bad_gain_map=bad_gain_map)


# ── Fallback helpers ──────────────────────────────────────────────────────────

def _fill_fallback_column(
        col, n_rows, row_border, n_halves,
        gain_map, cte_map, bad_gain_map,
        prev_gain_map, global_peak, calib_energy, parity):
    _fill_fallback_column_half(col, 0, row_border, "bottom",
                               gain_map, cte_map, bad_gain_map,
                               prev_gain_map, global_peak, calib_energy, parity)
    if n_halves == 2:
        _fill_fallback_column_half(col, row_border, n_rows, "top",
                                   gain_map, cte_map, bad_gain_map,
                                   prev_gain_map, global_peak, calib_energy, parity)


def _fill_fallback_column_half(
        col, row_lo, row_hi, half_label,
        gain_map, cte_map, bad_gain_map,
        prev_gain_map, global_peak, calib_energy, parity):
    """
    Fill rows [row_lo, row_hi) with fallback or kept-previous values.

    Matches ROOT:
      - If prev_gain_map has no fill yet (≤ 0) → use global peak → flag=1
      - If prev_gain_map already has values      → keep them       → flag=2
    """
    h = 0 if half_label == "bottom" else 1
    anchor_row = row_lo  # first row of this half

    if prev_gain_map[anchor_row, col] <= 0.0:
        # True fallback — use global peak
        fallback_gain = calib_energy / global_peak.ppos[parity, h]
        for row in range(row_lo, row_hi):
            gain_map[row, col]     = fallback_gain
            cte_map[row, col]      = 1.0
            bad_gain_map[row, col] = 1
    else:
        # Keep values from previous outer iteration
        for row in range(row_lo, row_hi):
            bad_gain_map[row, col] = 2
