"""
pnccd_ana.physics.pattern_recognition
======================================
Photon-event recognition using connected-component clustering.

Algorithm (matches ROOT HStepFilterEvents4)
────────────────────────────────────────────
  1. THRESHOLD SCAN
     Collect every pixel above split_sigma × noise into the candidate set
     (secondary threshold = ROOT ThresSec).
     ThresSec is clamped to ThresPrm per-pixel (ROOT behaviour):
       if ThresSec[px] > ThresPrm[px]: ThresSec[px] = ThresPrm[px]

  2. CONNECTED-COMPONENT LABELLING  (4-connected, no diagonal)
     Neighbours: left (same row, col-1) and below (row-1, same col).
     Matches ROOT HStepFilterEvents4 exactly.
     scipy.ndimage.label with _STRUCT_NO_DIAG is used when available.

  3. ACCEPT / REJECT
     A cluster is accepted only if at least one pixel exceeds
     seed_sigma × noise (primary threshold = ROOT ThresPrm).
     Border pixels are FLAGGED (kPixBorder) but NOT excluded from seeding,
     matching ROOT which stores border flag for downstream filtering.

  4. SHAPE CLASSIFICATION
     Within the accepted cluster:
       · seed     = pixel with maximum ADU value
       · offsets  = (dY, dX) of every other pixel relative to seed
       · grade    = _GRADE_LOOKUP.get(frozenset(offsets), GRADE_OTHER)

  5. ADU SUM
     Sum ALL pixels in the cluster regardless of grade.

Why the seed = argmax is noise-sensitive for split events
──────────────────────────────────────────────────────────
  For a 2×2 quadruple with nearly equal charge sharing, thermal noise
  can shift which pixel is argmax frame-by-frame.  Each shift changes
  the frozenset of neighbour offsets and may mis-classify the event as
  GRADE_OTHER.  ROOT avoids this by storing raw clusters and classifying
  grade in a separate downstream step with gain-corrected values.
  Here we mitigate by trying ALL pixels as seed candidate and accepting
  the grade that matches a known pattern; only if no candidate yields a
  known grade do we assign GRADE_OTHER.

Grade definitions — SINGLE SOURCE OF TRUTH
────────────────────────────────────────────
  Neighbour offsets are (dY, dX) relative to the seed pixel (0, 0).
  The seed is the maximum-ADU pixel in the cluster.

   0  single      no neighbours above split threshold
   1  double      (0,+1)
   2  double      (+1, 0)
   3  double      (0,-1)
   4  double      (-1, 0)
   5  triple      (+1, 0)+(0,+1)
   6  triple      (0,-1)+(+1, 0)
   7  triple      (-1, 0)+(0,-1)
   8  triple      (-1, 0)+(0,+1)
   9  quadruple   (+1, 0)+(0,+1)+(+1,+1)
  10  quadruple   (+1, 0)+(0,-1)+(+1,-1)
  11  quadruple   (-1, 0)+(0,-1)+(-1,-1)
  12  quadruple   (-1, 0)+(0,+1)+(-1,+1)
  13  T-left      (+1, 0)+(0,+1)+(-1, 0)
  14  T-right     (+1, 0)+(0,-1)+(-1, 0)
  15  b-left      (+1, 0)+(0,+1)+(-1, 0)+(-1,+1)
  16  b-right     (+1, 0)+(0,-1)+(-1, 0)+(-1,-1)
  17  I-shape     (+1, 0)+(-1, 0)
  18  other       any cluster shape not listed above

Coordinate convention
──────────────────────
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
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Grade definitions — edit ONLY this table
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
    ( 13, "T-left",    frozenset({(+1, 0), ( 0,+1), (-1, 0)})),
    ( 14, "T-right",   frozenset({(+1, 0), ( 0,-1), (-1, 0)})),
    ( 15, "b-left",    frozenset({(+1, 0), ( 0,+1), (-1, 0), (-1,+1)})),
    ( 16, "b-right",   frozenset({(+1, 0), ( 0,-1), (-1, 0), (-1,-1)})),
    ( 17, "I-shape",   frozenset({(+1, 0), (-1, 0)})),
]

GRADE_OTHER    = 18
GRADE_REJECTED = -1   # internal sentinel, never stored in output
N_GRADES       = GRADE_OTHER + 1   # 0..18 inclusive

GRADE_NAMES: dict[int, str] = {g: name for g, name, _ in _GRADE_DEFS}
GRADE_NAMES[GRADE_OTHER] = "other"

# frozenset(offsets) → grade_id  (built once at import)
_GRADE_LOOKUP: dict[frozenset, int] = {
    offsets: gid for gid, _, offsets in _GRADE_DEFS
}

_GRADE_DEFS_BY_ID: dict[int, tuple[str, frozenset]] = {
    gid: (label, offsets) for gid, label, offsets in _GRADE_DEFS
}

# Pre-built set of all known offset frozensets for fast membership test
_KNOWN_OFFSET_SETS: set[frozenset] = {offsets for _, _, offsets in _GRADE_DEFS}


# ══════════════════════════════════════════════════════════════════════════════
# Output dtype
# ══════════════════════════════════════════════════════════════════════════════

EVENT_DTYPE = np.dtype([
    ("Y",        np.int16),
    ("X",        np.int16),
    ("grade",    np.int8),
    ("adu_sum",  np.float32),   # sum of ALL pixels in the cluster
    ("adu_seed", np.float32),   # value of the maximum-ADU pixel
    ("flag",     np.uint8),     # bitmask: border / overflow / misfit
])

# Flag bits (matching ROOT HEventFlags kPix* values)
FLAG_BORDER    = np.uint8(1 << 3)   # kPixBorder
FLAG_OVERFLOW  = np.uint8(1 << 0)   # kPixOverflow
FLAG_UNDERFLOW = np.uint8(1 << 1)   # kPixUnderflow
FLAG_MISFIT    = np.uint8(1 << 2)   # kPixMisfit


# ══════════════════════════════════════════════════════════════════════════════
# Connected-component labelling (4-connected, no diagonal)
# ══════════════════════════════════════════════════════════════════════════════

# Structure matches ROOT: left, right, above, below — no diagonal
_STRUCT_NO_DIAG = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.int32)


def _find_clusters(sec_mask: np.ndarray) -> np.ndarray:
    """
    Assign a cluster ID to every above-threshold pixel.

    Uses scipy.ndimage.label with 4-connected structure when available.
    Falls back to a correct union-find implementation otherwise.

    Parameters
    ----------
    sec_mask : bool (n_Y, n_X) — True where pixel > split_sigma × noise

    Returns
    -------
    label_map : int32 (n_Y, n_X), 0 = background, >0 = cluster ID
    """
    if _HAVE_SCIPY:
        label_map, _ = _scipy_label(sec_mask, structure=_STRUCT_NO_DIAG)
        return label_map.astype(np.int32)

    # ── Pure-Python union-find fallback ───────────────────────────────────────
    import warnings
    warnings.warn(
        "scipy not found — using slow Python union-find.\n"
        "Install scipy: pip install scipy",
        RuntimeWarning, stacklevel=3)

    n_Y, n_X  = sec_mask.shape
    label_map = np.zeros((n_Y, n_X), dtype=np.int32)

    # parent[i] = root of label i; label 0 = background (unused)
    parent: list[int] = [0]

    def _find(x: int) -> int:
        # Iterative path-halving (safe for large trees)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> int:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return ra
        # Smaller root wins (consistent with ROOT which keeps lower EventID)
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep
        return keep

    # Raster-scan: Y=0 first, left to right — same order as ROOT MaskIndices
    flat_idx = np.flatnonzero(sec_mask)   # already in raster order

    for idx in flat_idx:
        y = int(idx // n_X)
        x = int(idx  % n_X)

        # Left neighbour: same row, col-1  (ROOT: ThisIndex-1)
        left_label = int(label_map[y, x - 1]) if x > 0 else 0
        # Below neighbour: row-1, same col  (ROOT: ThisIndex-ColCount)
        below_label = int(label_map[y - 1, x]) if y > 0 else 0

        left_root  = _find(left_label)  if left_label  else 0
        below_root = _find(below_label) if below_label else 0

        if left_root == 0 and below_root == 0:
            new_lbl = len(parent)
            parent.append(new_lbl)
            label_map[y, x] = new_lbl
        elif left_root != 0 and below_root == 0:
            label_map[y, x] = left_root
        elif left_root == 0 and below_root != 0:
            label_map[y, x] = below_root
        else:
            merged = _union(left_root, below_root)
            label_map[y, x] = merged

    # Flatten: resolve all labels to their canonical root
    for idx in flat_idx:
        y = int(idx // n_X)
        x = int(idx  % n_X)
        label_map[y, x] = _find(int(label_map[y, x]))

    return label_map


# ══════════════════════════════════════════════════════════════════════════════
# Shape classification
# ══════════════════════════════════════════════════════════════════════════════

def _classify_cluster(ys: np.ndarray,
                      xs: np.ndarray,
                      seed_idx: int) -> int:
    """
    Classify cluster shape using the argmax pixel as seed.

    The grade is determined by the frozenset of (dY, dX) offsets of all
    other pixels relative to the seed (argmax) pixel.  This matches ROOT
    which uses the pixel with maximum ADU as the seed.

    If the argmax seed gives GRADE_OTHER, we do NOT try other seeds —
    the shape is genuinely unrecognised.  The multi-seed fallback was
    incorrect: for a 2×2 quadruple it always returned grade 9 regardless
    of which corner was the true argmax, making grades 10/11/12 impossible.

    Parameters
    ----------
    ys       : int array — Y coordinates of all cluster pixels
    xs       : int array — X coordinates of all cluster pixels
    seed_idx : int — index of the argmax (seed) pixel in ys/xs

    Returns
    -------
    grade : int — from _GRADE_LOOKUP or GRADE_OTHER
    """
    sy = ys[seed_idx]
    sx = xs[seed_idx]

    neighbours = frozenset(
        (int(ys[j] - sy), int(xs[j] - sx))
        for j in range(len(ys))
        if j != seed_idx
    )
    return _GRADE_LOOKUP.get(neighbours, GRADE_OTHER)


# ══════════════════════════════════════════════════════════════════════════════
# Main public function
# ══════════════════════════════════════════════════════════════════════════════

def find_events(
        corrected:         np.ndarray,
        noise_map:         np.ndarray,
        search_mask:       np.ndarray | None = None,
        seed_sigma:        float = 5.0,
        split_sigma:       float = 3.0,
        reject_extra:      bool  = False,
        bad_pixel_mask:    np.ndarray | None = None,
        clamp_sec_to_prim: bool  = True,
        flag_border:       bool  = True,
) -> np.ndarray:
    """
    Find and classify photon events in a single CM-corrected frame.

    Algorithm  (ROOT HStepFilterEvents4 equivalent)
    ────────────────────────────────────────────────
    1. Build pixel masks:
         prim_mask = frame > seed_sigma  × noise   (ThresPrm)
         sec_mask  = frame > split_sigma × noise   (ThresSec)
       If clamp_sec_to_prim: sec_thr = min(sec_thr, prim_thr) per pixel.
       Excluded pixels (search_mask=False or bad_pixel_mask=True) are
       zeroed so they cannot be seeds or neighbours.

    2. 4-connected component labelling on sec_mask.
       No diagonal neighbours — matches ROOT exactly.

    3. Accept cluster if ≥1 pixel in prim_mask.

    4. Within each accepted cluster:
         seed     = pixel with maximum ADU
         grade    = _classify_cluster (tries all seed candidates to
                    recover near-equal-charge-sharing events)
         flag     = OR of per-pixel flags (border / overflow / underflow)

    5. adu_sum  = sum of ALL cluster pixels
       adu_seed = ADU of the seed (argmax) pixel

    Parameters
    ----------
    corrected          : float32 (n_Y, n_X) — CM-corrected frame
    noise_map          : float32 (n_Y, n_X) — per-pixel noise [ADU RMS]
    search_mask        : bool (n_Y, n_X) or None — True = active pixel
    seed_sigma         : primary threshold multiplier   (ROOT ThresPrm, typ. 5)
    split_sigma        : secondary threshold multiplier (ROOT ThresSec, typ. 3)
    reject_extra       : if True, discard GRADE_OTHER events
    bad_pixel_mask     : bool (n_Y, n_X) or None — True = bad, always excluded
    clamp_sec_to_prim  : clamp sec threshold to prim per pixel (ROOT behaviour)
    flag_border        : attach FLAG_BORDER to events touching detector edge

    Returns
    -------
    events : structured array with fields Y, X, grade, adu_sum, adu_seed, flag
             Y, X are the coordinates of the seed (argmax) pixel.
    """
    frame = np.ascontiguousarray(corrected, dtype=np.float32)
    noise = np.ascontiguousarray(noise_map, dtype=np.float32)
    H, W  = frame.shape

    # ── Border mask ───────────────────────────────────────────────────────────
    # Flags pixels on the detector edge.  ROOT sets kPixBorder but does NOT
    # exclude border pixels from seeding — flag is for downstream filtering.
    border_mask = np.zeros((H, W), dtype=bool)
    if flag_border:
        border_mask[0,  :]  = True
        border_mask[-1, :]  = True
        border_mask[:,  0]  = True
        border_mask[:, -1]  = True

    # ── Build combined exclusion mask and zero excluded pixels ────────────────
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

    # ROOT: if ThresSec[px] > ThresPrm[px] or not finite → ThresSec = ThresPrm
    if clamp_sec_to_prim:
        np.minimum(sec_thr, prim_thr, out=sec_thr)

    prim_mask = frame > prim_thr   # must have ≥1 per cluster
    sec_mask  = frame > sec_thr    # defines cluster extent (⊇ prim_mask)

    if not sec_mask.any():
        return np.empty(0, dtype=EVENT_DTYPE)

    # ── Connected-component labelling ─────────────────────────────────────────
    label_map = _find_clusters(sec_mask)

    # ── Cluster IDs that contain ≥1 primary pixel ─────────────────────────────
    primary_labels = set(
        int(v) for v in np.unique(label_map[prim_mask]) if v > 0
    )
    if not primary_labels:
        return np.empty(0, dtype=EVENT_DTYPE)

        # ── Cluster IDs that contain ≥1 primary pixel ─────────────────────────────
    primary_labels = set(
        int(v) for v in np.unique(label_map[prim_mask]) if v > 0
    )
    if not primary_labels:
        return np.empty(0, dtype=EVENT_DTYPE)

    # ── Process each accepted cluster ─────────────────────────────────────────
    rows_l:   list[int]   = []
    cols_l:   list[int]   = []
    grades_l: list[int]   = []
    sigs_l:   list[float] = []
    seeds_l:  list[float] = []
    flags_l:  list[int]   = []

    for cid in primary_labels:
        pix_mask = label_map == cid
        ys, xs   = np.nonzero(pix_mask)        # pixel coordinates
        vals     = frame[ys, xs]

        # Seed = maximum ADU pixel (ROOT: first hit over ThresPrm in sorted
        # order, but effectively argmax for the dominant pixel)
        seed_idx = int(np.argmax(vals))
        sy       = int(ys[seed_idx])
        sx       = int(xs[seed_idx])
        seed_val = float(vals[seed_idx])

        # ── Shape classification using argmax seed ────────────────────────────
        # Grade is determined by offsets relative to the argmax pixel.
        # Matches ROOT HStepFilterEvents4 which uses the pixel with maximum
        # ADU as the reference for pattern classification.
        grade = _classify_cluster(ys, xs, seed_idx)

        if reject_extra and grade == GRADE_OTHER:
            continue

        # ── Per-event flag (OR of per-pixel flags) ────────────────────────────
        evt_flag = np.uint8(0)
        if flag_border and border_mask[ys, xs].any():
            evt_flag |= FLAG_BORDER

        adu_sum = float(vals.sum())

        rows_l.append(sy)
        cols_l.append(sx)
        grades_l.append(grade)
        sigs_l.append(adu_sum)
        seeds_l.append(seed_val)
        flags_l.append(int(evt_flag))

    if not rows_l:
        return np.empty(0, dtype=EVENT_DTYPE)

    out = np.empty(len(rows_l), dtype=EVENT_DTYPE)
    out["Y"]        = np.array(rows_l,   dtype=np.int16)
    out["X"]        = np.array(cols_l,   dtype=np.int16)
    out["grade"]    = np.array(grades_l, dtype=np.int8)
    out["adu_sum"]  = np.array(sigs_l,   dtype=np.float32)
    out["adu_seed"] = np.array(seeds_l,  dtype=np.float32)
    out["flag"]     = np.array(flags_l,  dtype=np.uint8)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Helpers retained for energy_cal / plotting compatibility
# ══════════════════════════════════════════════════════════════════════════════

def _offsets_to_bitmask(offsets: frozenset) -> int:
    _OFFSET_TO_BIT = {
        ( 0,+1): 0, (+1, 0): 1, ( 0,-1): 2, (-1, 0): 3,
        (+1,+1): 4, (+1,-1): 5, (-1,-1): 6, (-1,+1): 7,
    }
    mask = 0
    for off in offsets:
        if off not in _OFFSET_TO_BIT:
            raise ValueError(f"Offset {off} outside central 3×3 neighbours.")
        mask |= (1 << _OFFSET_TO_BIT[off])
    return mask


def build_c_grade_table() -> list[int]:
    table = [GRADE_OTHER] * 256
    for gid, _, offsets in _GRADE_DEFS:
        mask = _offsets_to_bitmask(offsets)
        if table[mask] != GRADE_OTHER:
            raise RuntimeError(
                f"Grade {gid} bitmask 0x{mask:02X} collides with "
                f"grade {table[mask]} — check _GRADE_DEFS.")
        table[mask] = gid
    return table


def write_grade_table_header(path: str | Path | None = None) -> Path:
    table = build_c_grade_table()
    if path is None:
        path = Path(__file__).with_name("_grade_table_generated.h")
    path = Path(path)
    lines = [
        "/* AUTO-GENERATED — DO NOT EDIT BY HAND. */",
        f"#define N_GRADE_DEFS {len(_GRADE_DEFS)}",
        f"#define GRADE_OTHER  {GRADE_OTHER}",
        "static const uint8_t GRADE_TABLE[256] = {",
    ]
    for i in range(0, 256, 16):
        row = ", ".join(f"{table[i+j]:3d}" for j in range(16))
        lines.append(f"    {row},  /* 0x{i:02X}–0x{i+15:02X} */")
    lines += ["};", ""]
    path.write_text("\n".join(lines))
    return path
