"""
pnccd_ana.physics.event_filter
==============================
Photon-event detection using connected-component clustering.

Stores full per-pixel cluster data (CSR format) — no grade assignment,
no summary EVENT_DTYPE.  Grade classification happens in calibrate.py
after gain and CTI correction stabilise the seed pixel.

Algorithm
─────────
  1. THRESHOLD SCAN
       sec_mask  = frame > split_sigma × noise   (ROOT ThresSec)
       prim_mask = frame > seed_sigma  × noise   (ROOT ThresPrm)
     ThresSec is clamped to ThresPrm per pixel (ROOT behaviour).
     Excluded pixels (bad_pixel_mask, search_mask) are zeroed.

  2. 4-CONNECTED COMPONENT LABELLING on sec_mask.
     scipy.ndimage.label with cross-shaped structure (no diagonals).

  3. ACCEPT cluster if ≥1 pixel > prim_mask (ROOT HasExceededPrimThresh).

  4. STORE per cluster in CSR format:
       pixel_Y   — Y coordinate of each cluster pixel
       pixel_X   — X coordinate of each cluster pixel
       pixel_adu — ADU value of each cluster pixel
       offsets   — int64[n_events+1], cluster i spans [offsets[i]:offsets[i+1]]
       flag      — per-event bitmask: border/overflow/underflow

Coordinate convention
─────────────────────
  data[Y, X]   Y = row (axis 0),  X = col (axis 1)
  Y = 0 is the first readout row (rolling shutter bottom).
"""

from __future__ import annotations

import numpy as np
try:
    from scipy.ndimage import label as _scipy_label
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
# Flag bits  (matching ROOT HEventFlags kPix* values)
# ══════════════════════════════════════════════════════════════════════════════

FLAG_BORDER    = np.uint8(1 << 3)   # kPixBorder
FLAG_OVERFLOW  = np.uint8(1 << 0)   # kPixOverflow
FLAG_UNDERFLOW = np.uint8(1 << 1)   # kPixUnderflow
FLAG_MISFIT    = np.uint8(1 << 2)   # kPixMisfit


# ══════════════════════════════════════════════════════════════════════════════
# Connected-component labelling — 4-connected, no diagonals
# ══════════════════════════════════════════════════════════════════════════════

_STRUCT_NO_DIAG = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.int32)


def _find_clusters(sec_mask: np.ndarray) -> np.ndarray:
    """
    4-connected component labelling on sec_mask.

    Parameters
    ----------
    sec_mask : bool (n_Y, n_X)

    Returns
    -------
    label_map : int32 (n_Y, n_X), 0=background, >0=cluster ID
    """
    if _HAVE_SCIPY:
        label_map, _ = _scipy_label(sec_mask, structure=_STRUCT_NO_DIAG)
        return label_map.astype(np.int32)

    import warnings
    warnings.warn(
        "scipy not found — using slow Python union-find.\n"
        "Install scipy: pip install scipy",
        RuntimeWarning, stacklevel=3)

    n_Y, n_X  = sec_mask.shape
    label_map = np.zeros((n_Y, n_X), dtype=np.int32)
    parent: list[int] = [0]

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> int:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return ra
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep
        return keep

    for idx in np.flatnonzero(sec_mask):
        y = int(idx // n_X)
        x = int(idx  % n_X)
        left_label  = int(label_map[y, x - 1]) if x > 0 else 0
        below_label = int(label_map[y - 1, x]) if y > 0 else 0
        left_root   = _find(left_label)  if left_label  else 0
        below_root  = _find(below_label) if below_label else 0

        if left_root == 0 and below_root == 0:
            new_lbl = len(parent)
            parent.append(new_lbl)
            label_map[y, x] = new_lbl
        elif left_root != 0 and below_root == 0:
            label_map[y, x] = left_root
        elif left_root == 0 and below_root != 0:
            label_map[y, x] = below_root
        else:
            label_map[y, x] = _union(left_root, below_root)

    for idx in np.flatnonzero(sec_mask):
        y = int(idx // n_X)
        x = int(idx  % n_X)
        label_map[y, x] = _find(int(label_map[y, x]))

    return label_map


# ══════════════════════════════════════════════════════════════════════════════
# Main public function
# ══════════════════════════════════════════════════════════════════════════════

def find_events(
        corrected:         np.ndarray,
        noise_map:         np.ndarray,
        search_mask:       np.ndarray | None = None,
        seed_sigma:        float = 5.0,
        split_sigma:       float = 3.0,
        bad_pixel_mask:    np.ndarray | None = None,
        clamp_sec_to_prim: bool  = True,
        flag_border:       bool  = True,
) -> dict:
    """
    Find photon events in a single CM-corrected frame.

    Returns full per-pixel cluster data in CSR format.
    No grade assignment — deferred to calibrate.py.

    Parameters
    ----------
    corrected          : float32 (n_Y, n_X) — CM-corrected frame
    noise_map          : float32 (n_Y, n_X) — per-pixel noise [ADU RMS]
    search_mask        : bool (n_Y, n_X) or None — True = active pixel
    seed_sigma         : primary threshold   (ROOT ThresPrm, typ. 5)
    split_sigma        : secondary threshold (ROOT ThresSec, typ. 3)
    bad_pixel_mask     : bool (n_Y, n_X) or None
    clamp_sec_to_prim  : clamp sec threshold to prim per pixel (ROOT)
    flag_border        : attach FLAG_BORDER to edge-touching clusters

    Returns
    -------
    dict with keys:
        pixel_Y   : int16  (n_pixels_total,) — Y of each cluster pixel
        pixel_X   : int16  (n_pixels_total,) — X of each cluster pixel
        pixel_adu : float32(n_pixels_total,) — ADU of each cluster pixel
        offsets   : int64  (n_events+1,)     — CSR offsets
        flag      : uint8  (n_events,)        — per-event flag bitmask

    Empty result: all arrays have length 0 (offsets has length 1 = [0]).
    """
    frame = np.ascontiguousarray(corrected, dtype=np.float32)
    noise = np.ascontiguousarray(noise_map, dtype=np.float32)
    H, W  = frame.shape

    # ── Border mask ───────────────────────────────────────────────────────────
    border_mask = np.zeros((H, W), dtype=bool)
    if flag_border:
        border_mask[0,  :] = True
        border_mask[-1, :] = True
        border_mask[:,  0] = True
        border_mask[:, -1] = True

    # ── Exclusion mask ────────────────────────────────────────────────────────
    include: np.ndarray | None = None
    if search_mask is not None:
        include = search_mask.astype(bool, copy=False)
    if bad_pixel_mask is not None:
        bad_bool = bad_pixel_mask.astype(bool, copy=False)
        include  = (~bad_bool) if include is None else (include & ~bad_bool)
    if include is not None:
        frame = frame.copy()
        frame[~include] = 0.0

    # ── Threshold maps ────────────────────────────────────────────────────────
    prim_thr = (seed_sigma  * noise).astype(np.float32)
    sec_thr  = (split_sigma * noise).astype(np.float32)
    if clamp_sec_to_prim:
        np.minimum(sec_thr, prim_thr, out=sec_thr)

    prim_mask = frame > prim_thr
    sec_mask  = frame > sec_thr

    # ── Empty frame fast-path ─────────────────────────────────────────────────
    if not sec_mask.any():
        return _empty_result()

    # ── Clustering ────────────────────────────────────────────────────────────
    label_map = _find_clusters(sec_mask)

    primary_labels = set(
        int(v) for v in np.unique(label_map[prim_mask]) if v > 0
    )
    if not primary_labels:
        return _empty_result()

    # ── Build CSR output ──────────────────────────────────────────────────────
    all_Y:   list[np.ndarray] = []
    all_X:   list[np.ndarray] = []
    all_adu: list[np.ndarray] = []
    flags_l: list[int]        = []
    offsets: list[int]        = [0]

    for cid in primary_labels:
        pix_mask = label_map == cid
        ys, xs   = np.nonzero(pix_mask)
        vals     = frame[ys, xs]

        evt_flag = np.uint8(0)
        if flag_border and border_mask[ys, xs].any():
            evt_flag |= FLAG_BORDER

        all_Y.append(ys.astype(np.int16))
        all_X.append(xs.astype(np.int16))
        all_adu.append(vals)
        flags_l.append(int(evt_flag))
        offsets.append(offsets[-1] + len(ys))

    if not all_Y:
        return _empty_result()

    return {
        "pixel_Y":   np.concatenate(all_Y).astype(np.int16),
        "pixel_X":   np.concatenate(all_X).astype(np.int16),
        "pixel_adu": np.concatenate(all_adu).astype(np.float32),
        "offsets":   np.array(offsets, dtype=np.int64),
        "flag":      np.array(flags_l, dtype=np.uint8),
    }


def _empty_result() -> dict:
    return {
        "pixel_Y":   np.empty(0, dtype=np.int16),
        "pixel_X":   np.empty(0, dtype=np.int16),
        "pixel_adu": np.empty(0, dtype=np.float32),
        "offsets":   np.zeros(1, dtype=np.int64),
        "flag":      np.empty(0, dtype=np.uint8),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CSR cluster accessor helpers  (used by gain.py, cti.py, calibrate.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_cluster(cluster_data: dict, event_idx: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (pixel_Y, pixel_X, pixel_adu) for a single event.

    Parameters
    ----------
    cluster_data : dict from find_events or load_clusters_h5
    event_idx    : event index

    Returns
    -------
    ys   : int16  (n_pixels,)
    xs   : int16  (n_pixels,)
    adus : float32(n_pixels,)
    """
    lo = int(cluster_data["offsets"][event_idx])
    hi = int(cluster_data["offsets"][event_idx + 1])
    return (cluster_data["pixel_Y"][lo:hi],
            cluster_data["pixel_X"][lo:hi],
            cluster_data["pixel_adu"][lo:hi])


def n_pixels_per_event(cluster_data: dict) -> np.ndarray:
    """Return int32 array of cluster sizes (one per event)."""
    off = cluster_data["offsets"]
    return (off[1:] - off[:-1]).astype(np.int32)


def seed_pixels(cluster_data: dict) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (seed_Y, seed_X, seed_adu) for every event.

    Seed = argmax ADU pixel within the cluster.
    O(n_pixels_total) — single pass.
    """
    n_events  = len(cluster_data["offsets"]) - 1
    seed_Y    = np.empty(n_events, dtype=np.int16)
    seed_X    = np.empty(n_events, dtype=np.int16)
    seed_adu  = np.empty(n_events, dtype=np.float32)

    for i in range(n_events):
        ys, xs, adus = get_cluster(cluster_data, i)
        best         = int(np.argmax(adus))
        seed_Y[i]    = ys[best]
        seed_X[i]    = xs[best]
        seed_adu[i]  = adus[best]

    return seed_Y, seed_X, seed_adu


def adu_sums(cluster_data: dict) -> np.ndarray:
    """Return float32 array of total ADU per event."""
    off  = cluster_data["offsets"]
    adus = cluster_data["pixel_adu"]
    n    = len(off) - 1
    out  = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = adus[off[i]:off[i+1]].sum()
    return out
