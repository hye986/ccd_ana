"""
pnccd_ana.physics.calibrate
============================
Event filtering for each calibration iteration, grade assignment,
and final energy conversion.

Matches ROOT HStepGainMapCCDHLL event-filtering logic and
HStepCalibEvents energy application.

All hot paths are fully vectorised — no Python loops over events or pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Grade table
# ══════════════════════════════════════════════════════════════════════════════

_GRADE_DEFS: list[tuple[int, str, frozenset]] = [
    (  0, "single",    frozenset()),
    (  1, "double",    frozenset({( 0,+1)})),
    (  2, "double",    frozenset({(+1, 0)})),
    (  3, "double",    frozenset({( 0,-1)})),
    (  4, "double",    frozenset({(-1, 0)})),
    (  5, "triple",    frozenset({(+1, 0), ( 0,+1)})),
    (  6, "triple",    frozenset({( 0,-1), (+1, 0)})),
    (  7, "triple",    frozenset({(-1, 0), ( 0,-1)})),
    (  8, "triple",    frozenset({(-1, 0), ( 0,+1)})),
    (  9, "quadruple", frozenset({(+1, 0), ( 0,+1), (+1,+1)})),
    ( 10, "quadruple", frozenset({(+1, 0), ( 0,-1), (+1,-1)})),
    ( 11, "quadruple", frozenset({(-1, 0), ( 0,-1), (-1,-1)})),
    ( 12, "quadruple", frozenset({(-1, 0), ( 0,+1), (-1,+1)})),
]

GRADE_OTHER    = 13
GRADE_REJECTED = -1
N_GRADES       = GRADE_OTHER + 1

GRADE_NAMES: dict[int, str] = {g: name for g, name, _ in _GRADE_DEFS}
GRADE_NAMES[GRADE_OTHER] = "other"

_GRADE_LOOKUP: dict[frozenset, int] = {
    offsets: gid for gid, _, offsets in _GRADE_DEFS
}


# ══════════════════════════════════════════════════════════════════════════════
# Filtered event container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilteredEvents:
    """
    Output of filter_events_for_iteration.
    All arrays are 1-D with length n_accepted.
    """
    adu_sum:    np.ndarray   # float64 (n_accepted,)
    row:        np.ndarray   # float64 (n_accepted,)
    col:        np.ndarray   # float64 (n_accepted,)
    event_idx:  np.ndarray   # int64   (n_accepted,)
    n_accepted: int


# ══════════════════════════════════════════════════════════════════════════════
# Internal vectorised helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_id_array(offsets: np.ndarray) -> np.ndarray:
    """
    Build a per-pixel cluster-ID array from CSR offsets.

    cluster_id[pixel_idx] = which event this pixel belongs to.
    O(n_pixels), fully vectorised via np.repeat.
    """
    sizes = (offsets[1:] - offsets[:-1]).astype(np.int64)
    return np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)


def _reduceat_sum(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Sum values within each cluster segment using np.add.reduceat."""
    n = len(offsets) - 1
    if n == 0:
        return np.empty(0, dtype=values.dtype)
    starts = offsets[:-1].astype(np.intp)
    return np.add.reduceat(values.astype(np.float64), starts)


def _gain_lookup_vectorised(
        pix_Y:      np.ndarray,   # int16 pixel rows
        pix_X:      np.ndarray,   # int16 pixel cols
        gain_map:   np.ndarray,   # (n_rows, n_cols) float64
        mean_gain:  np.ndarray,   # (n_parity,) float64  fallback
        n_rows:     int,
        n_cols:     int,
) -> np.ndarray:
    """
    Vectorised gain lookup for all pixels.

    Returns float64 gain per pixel.
    Pixels where gain_map <= 0 fall back to mean_gain[parity].
    """
    n_parity = len(mean_gain)
    ry = np.clip(pix_Y.astype(np.int32), 0, n_rows - 1)
    rx = np.clip(pix_X.astype(np.int32), 0, n_cols - 1)

    g = gain_map[ry, rx].astype(np.float64)

    # Fallback mask
    bad = g <= 0.0
    if bad.any():
        if n_parity == 2:
            parity        = ((rx[bad] + 1) % 2).astype(np.int32)
            g[bad]        = mean_gain[parity]
        else:
            g[bad]        = mean_gain[0]

    return g


# ══════════════════════════════════════════════════════════════════════════════
# Iteration 0 — vectorised singles (+ optional U/D splits)
# ══════════════════════════════════════════════════════════════════════════════

def _filter_iteration0(
        cluster_data: dict,
        roi_low:      float,
        roi_high:     float,
        use_ud_split: bool,
) -> FilteredEvents:
    """
    Fully vectorised filter for iteration 0.

    Singles:   n_pixels == 1, flag & 0x7 == 0, ROI
    U/D split: n_pixels == 2, same column,     flag & 0x7 == 0, ROI
    """
    off   = cluster_data["offsets"]       # int64 (n_events+1,)
    flags = cluster_data["flag"]          # uint8 (n_events,)
    pix_Y = cluster_data["pixel_Y"]       # int16 (n_pixels_total,)
    pix_X = cluster_data["pixel_X"]       # int16 (n_pixels_total,)
    pix_a = cluster_data["pixel_adu"]     # float32 (n_pixels_total,)

    sizes    = (off[1:] - off[:-1]).astype(np.int32)   # (n_events,)
    flag_ok  = (flags & np.uint8(0x7)) == 0             # (n_events,)

    # ── Singles ───────────────────────────────────────────────────────────────
    single_mask = flag_ok & (sizes == 1)
    single_evt  = np.flatnonzero(single_mask)

    # For a single-pixel event, start index = the one pixel
    s_pix   = off[single_evt].astype(np.intp)
    s_adu   = pix_a[s_pix].astype(np.float64)
    s_row   = pix_Y[s_pix].astype(np.float64)
    s_col   = pix_X[s_pix].astype(np.float64)

    roi_s   = (s_adu > roi_low) & (s_adu < roi_high)
    s_adu   = s_adu[roi_s]
    s_row   = s_row[roi_s]
    s_col   = s_col[roi_s]
    s_idx   = single_evt[roi_s]

    if not use_ud_split:
        return FilteredEvents(
            adu_sum    = s_adu,
            row        = s_row,
            col        = s_col,
            event_idx  = s_idx,
            n_accepted = len(s_idx),
        )

    # ── U/D splits (n_pixels==2, same column) ─────────────────────────────────
    double_mask = flag_ok & (sizes == 2)
    double_evt  = np.flatnonzero(double_mask)

    if len(double_evt) > 0:
        p0 = off[double_evt].astype(np.intp)        # first pixel
        p1 = (off[double_evt] + 1).astype(np.intp)  # second pixel

        # Keep only same-column pairs
        same_col    = pix_X[p0] == pix_X[p1]
        double_evt  = double_evt[same_col]
        p0          = p0[same_col]
        p1          = p1[same_col]

        a0 = pix_a[p0].astype(np.float64)
        a1 = pix_a[p1].astype(np.float64)
        d_adu = a0 + a1

        # Signal-weighted row COG, rounded
        d_row = np.round(
            (pix_Y[p0].astype(np.float64) * a0 +
             pix_Y[p1].astype(np.float64) * a1) / d_adu
        )
        d_col = pix_X[p0].astype(np.float64)

        roi_d      = (d_adu > roi_low) & (d_adu < roi_high)
        d_adu      = d_adu[roi_d]
        d_row      = d_row[roi_d]
        d_col      = d_col[roi_d]
        d_idx      = double_evt[roi_d]

        return FilteredEvents(
            adu_sum    = np.concatenate([s_adu, d_adu]),
            row        = np.concatenate([s_row, d_row]),
            col        = np.concatenate([s_col, d_col]),
            event_idx  = np.concatenate([s_idx, d_idx]),
            n_accepted = len(s_idx) + len(d_idx),
        )

    return FilteredEvents(
        adu_sum    = s_adu,
        row        = s_row,
        col        = s_col,
        event_idx  = s_idx,
        n_accepted = len(s_idx),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Iterations 1+ — vectorised gain-weighted COG
# ══════════════════════════════════════════════════════════════════════════════

def _filter_iterationN(
        cluster_data:   dict,
        gain_map:       np.ndarray,
        mean_gain:      np.ndarray,
        roi_low:        float,
        roi_high:       float,
        n_rows:         int,
        n_cols:         int,
) -> FilteredEvents:
    """
    Vectorised filter for iterations 1+ (all cluster sizes).

    For each cluster:
      adjusted[pixel] = adu[pixel] × gain[row, col]
      signal_sum      = Σ adjusted
      COG_row         = round(Σ row  × adjusted / signal_sum)
      COG_col         = round(Σ col  × adjusted / signal_sum)
      signal          = signal_sum / gain[COG_row, COG_col]

    All operations are vectorised across the full pixel array using
    np.add.reduceat and cluster_id broadcasting.
    """
    off   = cluster_data["offsets"]
    flags = cluster_data["flag"]
    pix_Y = cluster_data["pixel_Y"]
    pix_X = cluster_data["pixel_X"]
    pix_a = cluster_data["pixel_adu"]
    n_evt = len(flags)

    if n_evt == 0:
        return FilteredEvents(
            adu_sum    = np.empty(0, dtype=np.float64),
            row        = np.empty(0, dtype=np.float64),
            col        = np.empty(0, dtype=np.float64),
            event_idx  = np.empty(0, dtype=np.int64),
            n_accepted = 0,
        )

    # Flag filter — mark bad events; we will zero their contribution
    flag_ok     = ((flags & np.uint8(0x7)) == 0)          # (n_evt,)
    cluster_id  = _cluster_id_array(off)                   # (n_pixels_total,)

    # Gain per pixel — vectorised lookup with fallback
    g_pix = _gain_lookup_vectorised(
        pix_Y, pix_X, gain_map, mean_gain, n_rows, n_cols)

    # Adjusted signal per pixel
    adj = pix_a.astype(np.float64) * g_pix                # (n_pixels_total,)

    # Per-cluster sums using reduceat
    starts    = off[:-1].astype(np.intp)
    sig_g     = np.add.reduceat(adj, starts)               # (n_evt,)
    row_sum   = np.add.reduceat(
        pix_Y.astype(np.float64) * adj, starts)            # (n_evt,)
    col_sum   = np.add.reduceat(
        pix_X.astype(np.float64) * adj, starts)            # (n_evt,)

    # Zero out flag-bad events to prevent division issues
    sig_g[~flag_ok]   = 0.0
    row_sum[~flag_ok] = 0.0
    col_sum[~flag_ok] = 0.0

    # COG — safe division
    valid   = flag_ok & (sig_g > 0.0)
    cog_row = np.zeros(n_evt, dtype=np.float64)
    cog_col = np.zeros(n_evt, dtype=np.float64)
    cog_row[valid] = np.round(row_sum[valid] / sig_g[valid])
    cog_col[valid] = np.round(col_sum[valid] / sig_g[valid])

    # Gain at COG pixel
    cr       = np.clip(cog_row.astype(np.int32), 0, n_rows - 1)
    cc       = np.clip(cog_col.astype(np.int32), 0, n_cols - 1)
    g_cog    = gain_map[cr, cc].astype(np.float64)

    # Fallback for zero COG gain
    bad_gcog = g_cog <= 0.0
    if bad_gcog.any():
        n_parity = len(mean_gain)
        if n_parity == 2:
            parity           = ((cc[bad_gcog] + 1) % 2).astype(np.int32)
            g_cog[bad_gcog]  = mean_gain[parity]
        else:
            g_cog[bad_gcog]  = mean_gain[0]

    # Reconstructed ADU signal
    sig             = np.zeros(n_evt, dtype=np.float64)
    sig[valid]      = sig_g[valid] / g_cog[valid]

    # ROI cut + flag cut
    accept  = valid & (sig > roi_low) & (sig < roi_high)
    evt_idx = np.flatnonzero(accept)

    return FilteredEvents(
        adu_sum    = sig[evt_idx],
        row        = cog_row[evt_idx],
        col        = cog_col[evt_idx],
        event_idx  = evt_idx.astype(np.int64),
        n_accepted = len(evt_idx),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════════════

def filter_events_for_iteration(
        cluster_data:   dict,
        gain_map:       np.ndarray,
        mean_gain:      np.ndarray,
        iteration:      int,
        roi_low:        float,
        roi_high:       float,
        n_rows:         int,
        n_cols:         int,
        mid_row:        int,
        split_even_odd: bool = True,
        use_ud_split:   bool = False,
) -> FilteredEvents:
    """
    Filter and reconstruct event signals for one calibration iteration.

    Iteration 0  → fast vectorised singles path (_filter_iteration0)
    Iterations 1+ → vectorised gain-weighted COG path (_filter_iterationN)
    """
    if iteration == 0:
        return _filter_iteration0(cluster_data, roi_low, roi_high, use_ud_split)
    return _filter_iterationN(
        cluster_data, gain_map, mean_gain,
        roi_low, roi_high, n_rows, n_cols)


# ══════════════════════════════════════════════════════════════════════════════
# Mean-gain update — vectorised
# ══════════════════════════════════════════════════════════════════════════════

def update_mean_gain(
        gain_map:       np.ndarray,
        bad_gain_map:   np.ndarray,
        n_cols:         int,
        n_rows:         int,
        row_border:     int,
        calib_energy:   float,
        split_even_odd: bool,
) -> np.ndarray:
    """
    Compute mean gain per parity from good columns (bad_gain_map == 0).

    Vectorised — no Python loop over columns.
    """
    n_parity  = 2 if split_even_odd else 1
    mean_gain = np.zeros(n_parity, dtype=np.float64)

    # Quality mask: row=0 anchor must be good
    good_cols = bad_gain_map[0, :] == 0          # (n_cols,) bool
    g_row0    = gain_map[0, :].astype(np.float64) # (n_cols,)
    good_cols = good_cols & (g_row0 > 0.0)

    if n_parity == 1:
        vals = g_row0[good_cols]
        mean_gain[0] = float(vals.mean()) if len(vals) else 1.0
    else:
        col_idx  = np.arange(n_cols, dtype=np.int32)
        parity   = (col_idx + 1) % 2              # matches ROOT (col+1)%2
        for p in range(2):
            mask = good_cols & (parity == p)
            vals = g_row0[mask]
            mean_gain[p] = float(vals.mean()) if len(vals) else 1.0

    return mean_gain


# ══════════════════════════════════════════════════════════════════════════════
# Grade assignment — vectorised via bitmask lookup table
# ══════════════════════════════════════════════════════════════════════════════

# Build a 256-entry bitmask → grade lookup table at import time.
# Neighbour offset → bit position mapping (matches 3×3 neighbourhood)
_OFF2BIT: dict[tuple[int, int], int] = {
    ( 0,+1): 0, (+1, 0): 1, ( 0,-1): 2, (-1, 0): 3,
    (+1,+1): 4, (+1,-1): 5, (-1,-1): 6, (-1,+1): 7,
}

_GRADE_TABLE: np.ndarray = np.full(256, GRADE_OTHER, dtype=np.int8)
for _gid, _, _offsets in _GRADE_DEFS:
    _mask = 0
    for _off in _offsets:
        _mask |= (1 << _OFF2BIT[_off])
    _GRADE_TABLE[_mask] = _gid


def assign_grades(
        cluster_data: dict,
        gain_map:     np.ndarray,
        n_rows:       int,
        n_cols:       int,
) -> np.ndarray:
    """
    Assign grades 0–13 to all events using the final gain map.

    Algorithm
    ---------
    1. Vectorised gain lookup for all pixels at once.
    2. Seed = argmax(adu × gain) per cluster via np.maximum.reduceat.
    3. Neighbour offsets relative to seed encoded as 8-bit bitmask.
    4. Grade = _GRADE_TABLE[bitmask] — single array lookup, no Python loop.

    Singles (n_pixels==1) are handled as a special fast path.
    Multi-pixel events use a compact Python loop only over clusters with
    n_pixels >= 2, which are far fewer than total pixels.
    """
    off     = cluster_data["offsets"]
    pix_Y   = cluster_data["pixel_Y"]
    pix_X   = cluster_data["pixel_X"]
    pix_a   = cluster_data["pixel_adu"]
    n_evt   = len(cluster_data["flag"])

    grades  = np.full(n_evt, GRADE_OTHER, dtype=np.int8)
    sizes   = (off[1:] - off[:-1]).astype(np.int32)

    # ── Singles: grade 0 unconditionally ──────────────────────────────────────
    grades[sizes == 1] = 0

    # ── Multi-pixel: need seed + neighbour bitmask ────────────────────────────
    multi_idx = np.flatnonzero(sizes > 1)
    if len(multi_idx) == 0:
        return grades

    # Vectorised gain for ALL pixels (cheaper than per-cluster lookup)
    g_all = _gain_lookup_vectorised(
        pix_Y, pix_X, gain_map, np.array([1.0]), n_rows, n_cols)

    weighted = pix_a.astype(np.float64) * g_all   # (n_pixels_total,)

    # Per-cluster max weighted value via reduceat
    starts   = off[:-1].astype(np.intp)
    max_w    = np.maximum.reduceat(weighted, starts)   # (n_evt,)

    # Build cluster_id array for multi-pixel events only
    # (avoid rebuilding for all events — only need multi_idx ones)
    for i in multi_idx:
        lo = int(off[i])
        hi = int(off[i + 1])
        w  = weighted[lo:hi]

        # Seed: first pixel with maximum weighted value
        seed_local = int(np.argmax(w))
        sy = int(pix_Y[lo + seed_local])
        sx = int(pix_X[lo + seed_local])

        # Neighbour bitmask
        bitmask = np.uint8(0)
        for k in range(hi - lo):
            if k == seed_local:
                continue
            dy = int(pix_Y[lo + k]) - sy
            dx = int(pix_X[lo + k]) - sx
            bit = _OFF2BIT.get((dy, dx))
            if bit is not None:
                bitmask |= np.uint8(1 << bit)

        grades[i] = _GRADE_TABLE[int(bitmask)]

    return grades


# ══════════════════════════════════════════════════════════════════════════════
# Final energy conversion — vectorised
# ══════════════════════════════════════════════════════════════════════════════

def compute_final_energies(
        cluster_data: dict,
        gain_map:     np.ndarray,
        n_rows:       int,
        n_cols:       int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply gain map to all events to get calibrated energies.

    Matches ROOT HStepCalibEvents — fully vectorised.

    Returns
    -------
    energy_sum  : float32 (n_events,)  total event energy [eV]
    cog_row     : float32 (n_events,)  energy-weighted row
    cog_col     : float32 (n_events,)  energy-weighted col
    seed_energy : float32 (n_events,)  seed pixel energy [eV]
    """
    off    = cluster_data["offsets"]
    pix_Y  = cluster_data["pixel_Y"]
    pix_X  = cluster_data["pixel_X"]
    pix_a  = cluster_data["pixel_adu"]
    n_evt  = len(cluster_data["flag"])

    if n_evt == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty

    # Gain per pixel — vectorised
    ry = np.clip(pix_Y.astype(np.int32), 0, n_rows - 1)
    rx = np.clip(pix_X.astype(np.int32), 0, n_cols - 1)
    g  = gain_map[ry, rx].astype(np.float64)

    # Energy per pixel: max(0, adu) × gain  (ROOT: Energy *= (Energy > 0))
    e_pix = np.maximum(0.0, pix_a.astype(np.float64)) * g  # (n_pixels_total,)

    # Per-cluster sums via reduceat
    starts    = off[:-1].astype(np.intp)
    e_sum     = np.add.reduceat(e_pix,                           starts)
    e_row_sum = np.add.reduceat(pix_Y.astype(np.float64) * e_pix, starts)
    e_col_sum = np.add.reduceat(pix_X.astype(np.float64) * e_pix, starts)

    # COG — safe division
    valid   = e_sum > 0.0
    cog_row = np.zeros(n_evt, dtype=np.float64)
    cog_col = np.zeros(n_evt, dtype=np.float64)
    cog_row[valid] = e_row_sum[valid] / e_sum[valid]
    cog_col[valid] = e_col_sum[valid] / e_sum[valid]

    # Seed energy: argmax(adu) per cluster
    # Use np.maximum.reduceat to find max adu value, then match pixel
    max_adu   = np.maximum.reduceat(pix_a.astype(np.float64), starts)
    cluster_id = _cluster_id_array(off)
    is_seed    = (pix_a.astype(np.float64) == max_adu[cluster_id])

    # First occurrence of max per cluster
    seed_pix_idx  = np.flatnonzero(is_seed)
    cid_at_seed   = cluster_id[seed_pix_idx]
    _, first_seed = np.unique(cid_at_seed, return_index=True)
    seed_global   = seed_pix_idx[first_seed]   # one per event

    seed_e = e_pix[seed_global]  # already gain-applied

    return (e_sum.astype(np.float32),
            cog_row.astype(np.float32),
            cog_col.astype(np.float32),
            seed_e.astype(np.float32))
