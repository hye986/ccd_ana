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
        max_cluster_size:  int   = 0,
) -> dict:
    """
    Find photon events in a single CM-corrected frame.

    Parameters
    ----------
    max_cluster_size : int, default 0
        Reject clusters larger than this many pixels.
        0 (default) disables the filter.
        Fe55 Mn Kα produces at most 4-pixel clusters; set 9 as generous limit
        to reject cosmic rays and particle tracks.

    Returns full per-pixel cluster data in CSR format.
    No grade assignment — deferred to calibrate.py.
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

    if not sec_mask.any():
        return _empty_result()

    # ── Clustering ────────────────────────────────────────────────────────────
    label_map = _find_clusters(sec_mask)

    # Only keep clusters that contain at least one primary-threshold pixel
    primary_labels = np.unique(label_map[prim_mask])
    primary_labels = primary_labels[primary_labels > 0]

    if len(primary_labels) == 0:
        return _empty_result()

    # ── Vectorised CSR construction ───────────────────────────────────────────
    # Build a boolean mask for all pixels belonging to accepted clusters.
    # Uses np.isin on the full label_map — O(H*W), single C call.
    accepted_mask = np.isin(label_map, primary_labels)

    # Flat indices of all accepted pixels
    flat_idx  = np.flatnonzero(accepted_mask)
    pix_Y     = (flat_idx // W).astype(np.int16)
    pix_X     = (flat_idx  % W).astype(np.int16)
    pix_adu   = frame.ravel()[flat_idx].astype(np.float32)
    pix_label = label_map.ravel()[flat_idx]   # cluster ID per pixel

    # Sort by cluster label so CSR offsets are contiguous
    sort_idx  = np.argsort(pix_label, kind="stable")
    pix_Y     = pix_Y[sort_idx]
    pix_X     = pix_X[sort_idx]
    pix_adu   = pix_adu[sort_idx]
    pix_label = pix_label[sort_idx]

    # Build CSR offsets from cluster label runs
    # unique_labels are in sorted order; counts give cluster sizes
    unique_labels, counts = np.unique(pix_label, return_counts=True)

    # ── Cluster size filter ───────────────────────────────────────────────────
    # Operates on sorted_flat so pixel coordinate arrays are rebuilt cleanly.
    if max_cluster_size > 0:
        keep       = counts <= max_cluster_size
        if not keep.all():
            keep_labels = unique_labels[keep]
            pix_keep    = np.isin(pix_label, keep_labels)
            # Re-extract and re-sort from accepted flat indices
            flat_idx    = flat_idx[pix_keep]
            pix_label   = pix_label[pix_keep]
            sort_idx    = np.argsort(pix_label, kind="stable")
            pix_label   = pix_label[sort_idx]
            unique_labels = keep_labels
            counts      = counts[keep]

    n_events = len(unique_labels)
    if n_events == 0:
        return _empty_result()

    # Rebuild pixel coordinate arrays from sorted flat indices
    pix_Y   = (flat_idx // W).astype(np.int16)
    pix_X   = (flat_idx  % W).astype(np.int16)
    pix_adu = frame.ravel()[flat_idx].astype(np.float32)

    offsets = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    # ── Per-event flag ────────────────────────────────────────────────────────
    flags = np.zeros(n_events, dtype=np.uint8)
    if flag_border:
        # For each cluster, check if any pixel is on the border
        border_flat = border_mask.ravel()[flat_idx].view(np.uint8)
        has_border  = np.maximum.reduceat(
            border_flat, offsets[:-1].astype(np.intp)
        ).astype(bool)
        flags[has_border] |= FLAG_BORDER

    return {
        "pixel_Y":   pix_Y,
        "pixel_X":   pix_X,
        "pixel_adu": pix_adu,
        "offsets":   offsets,
        "flag":      flags,
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
    """Return (pixel_Y, pixel_X, pixel_adu) for a single event."""
    lo = int(cluster_data["offsets"][event_idx])
    hi = int(cluster_data["offsets"][event_idx + 1])
    return (cluster_data["pixel_Y"][lo:hi],
            cluster_data["pixel_X"][lo:hi],
            cluster_data["pixel_adu"][lo:hi])


def n_pixels_per_event(cluster_data: dict) -> np.ndarray:
    """Return int32 array of cluster sizes — O(n_events), vectorised."""
    off = cluster_data["offsets"]
    return (off[1:] - off[:-1]).astype(np.int32)


def adu_sums(cluster_data: dict) -> np.ndarray:
    """
    Return float32 total ADU per event — fully vectorised.

    Uses np.add.reduceat which operates in C on the flat pixel array.
    O(n_pixels_total), no Python loop.
    """
    adus = cluster_data["pixel_adu"]
    off  = cluster_data["offsets"]
    n    = len(off) - 1
    if n == 0:
        return np.empty(0, dtype=np.float32)
    # reduceat needs start indices only (not the sentinel)
    starts = off[:-1].astype(np.intp)
    return np.add.reduceat(adus.astype(np.float64),
                           starts).astype(np.float32)


def seed_pixels(cluster_data: dict) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (seed_Y, seed_X, seed_adu) for every event — vectorised.

    Seed = argmax ADU pixel within each cluster.
    Uses a vectorised segment-argmax via np.maximum.reduceat.
    O(n_pixels_total), no Python loop.
    """
    adus = cluster_data["pixel_adu"]
    ys   = cluster_data["pixel_Y"]
    xs   = cluster_data["pixel_X"]
    off  = cluster_data["offsets"]
    n    = len(off) - 1

    if n == 0:
        return (np.empty(0, dtype=np.int16),
                np.empty(0, dtype=np.int16),
                np.empty(0, dtype=np.float32))

    starts = off[:-1].astype(np.intp)

    # Max ADU per cluster (vectorised)
    max_adu = np.maximum.reduceat(adus, starts).astype(np.float32)

    # For each pixel, does it equal the max of its cluster?
    # Build a per-pixel cluster index to look up max_adu
    # cluster_id[pixel] = which cluster this pixel belongs to
    sizes      = (off[1:] - off[:-1]).astype(np.int32)   # (n,)
    cluster_id = np.repeat(np.arange(n, dtype=np.int32), sizes)

    is_max = (adus == max_adu[cluster_id])

    # For each cluster, find the FIRST pixel that equals the max
    # (ties broken by first occurrence = lowest index in cluster)
    # We want one seed per cluster.
    # Strategy: among pixels where is_max, keep only the first per cluster.
    max_pix_idx = np.flatnonzero(is_max)   # global pixel indices of max pixels

    # cluster_id at these positions
    cid_at_max  = cluster_id[max_pix_idx]

    # First occurrence per cluster: use np.unique with return_index
    _, first_in_cid = np.unique(cid_at_max, return_index=True)
    seed_pix_idx    = max_pix_idx[first_in_cid]   # global pixel index of seed

    return (ys[seed_pix_idx].copy(),
            xs[seed_pix_idx].copy(),
            adus[seed_pix_idx].copy())
