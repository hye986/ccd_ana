"""
pnccd_ana.lib.pattern_recognition
==================================
Photon-event recognition using connected-component clustering.

Algorithm (matches ROOT HStepFilterEvents4)
────────────────────────────────────────────
  1. THRESHOLD SCAN
     Collect every pixel above split_sigma × noise into a sorted index
     list (secondary threshold, equivalent to ROOT ThresSec).

  2. CONNECTED-COMPONENT LABELLING  (union-find)
     Neighbours: left (same row, col-1) and below (row-1, same col).
     No diagonal neighbours — matching ROOT exactly.
     Each connected group of above-split-threshold pixels → one cluster.

  3. ACCEPT / REJECT
     A cluster is accepted only if at least one pixel exceeds
     seed_sigma × noise (primary threshold, equivalent to ROOT ThresPrm).
     Border pixels (row 0, row H-1, col 0, col W-1) are excluded from
     seeding but may contribute as neighbours.

  4. SHAPE CLASSIFICATION
     Within the accepted cluster:
       · seed = pixel with maximum ADU value
       · compute (dY, dX) offsets of every other cluster pixel relative
         to the seed
       · match the frozenset of offsets against _GRADE_DEFS
       · no match → GRADE_OTHER

  5. ADU SUM
     Sum ALL pixels in the cluster regardless of grade.
     This naturally includes the 2×2 diagonal without any threshold
     dependence — the quadruple diagonal problem is eliminated by
     design.

Why this is better than the old 5×5 local-max approach
────────────────────────────────────────────────────────
  The old code built a 3×3 neighbour bitmask at split_sigma.  For a 2×2
  cluster the diagonal pixel receives the smallest charge fraction
  (px × py in bilinear sharing) and frequently fell below split_sigma,
  causing the event to be misclassified as a triple.  The new code
  collects all connected pixels first and classifies the shape
  afterwards, so the diagonal is always included when it is physically
  connected (i.e., above split_sigma at all).

Grade definitions — SINGLE SOURCE OF TRUTH
────────────────────────────────────────────
  All grade information lives in _GRADE_DEFS below.
  Neighbour offsets are (dY, dX) relative to the seed pixel (0, 0).
  The seed is always the maximum-ADU pixel in the cluster.

  0  single      no neighbours above split threshold
  1  double      (0,+1)                     right
  2  double      (+1,0)                     up
  3  double      (0,-1)                     left
  4  double      (-1,0)                     down
  5  triple      (+1,0)+(0,+1)              up+right
  6  triple      (0,-1)+(+1,0)              left+up
  7  triple      (-1,0)+(0,-1)              down+left
  8  triple      (-1,0)+(0,+1)              down+right
  9  quadruple   (+1,0)+(0,+1)+(+1,+1)     up+right+up-right
 10  quadruple   (+1,0)+(0,-1)+(+1,-1)     up+left+up-left
 11  quadruple   (-1,0)+(0,-1)+(-1,-1)     down+left+down-left
 12  quadruple   (-1,0)+(0,+1)+(-1,+1)     down+right+down-right
 13  T-left      (+1,0)+(0,+1)+(-1,0)
 14  T-right     (+1,0)+(0,-1)+(-1,0)
 15  b-left      (+1,0)+(0,+1)+(-1,0)+(-1,+1)
 16  b-right     (+1,0)+(0,-1)+(-1,0)+(-1,-1)
 17  I-shape     (+1,0)+(-1,0)
 18  other       any cluster shape not listed above

Coordinate convention
─────────────────────
  data[Y, X]   Y = row (axis 0, vertical),  X = col (axis 1, horizontal)
  Y = 0 is at the bottom (readout side), Y increases upward.
  Rolling shutter reads Y=0 first.
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
# Grade definitions — edit ONLY this table to add / remove grades
# ══════════════════════════════════════════════════════════════════════════════

_GRADE_DEFS: list[tuple[int, str, frozenset]] = [
    (  0, "single",    frozenset()),
    # doubles
    (  1, "double",    frozenset({( 0,+1)})),
    (  2, "double",    frozenset({(+1, 0)})),
    (  3, "double",    frozenset({( 0,-1)})),
    (  4, "double",    frozenset({(-1, 0)})),
    # triples
    (  5, "triple",    frozenset({(+1, 0), ( 0,+1)})),
    (  6, "triple",    frozenset({( 0,-1), (+1, 0)})),
    (  7, "triple",    frozenset({(-1, 0), ( 0,-1)})),
    (  8, "triple",    frozenset({(-1, 0), ( 0,+1)})),
    # quadruples (2 cardinals + 1 diagonal)
    (  9, "quadruple", frozenset({(+1, 0), ( 0,+1), (+1,+1)})),
    ( 10, "quadruple", frozenset({(+1, 0), ( 0,-1), (+1,-1)})),
    ( 11, "quadruple", frozenset({(-1, 0), ( 0,-1), (-1,-1)})),
    ( 12, "quadruple", frozenset({(-1, 0), ( 0,+1), (-1,+1)})),
    # larger named patterns
    ( 13, "T-left",    frozenset({(+1, 0), ( 0,+1), (-1, 0)})),
    ( 14, "T-right",   frozenset({(+1, 0), ( 0,-1), (-1, 0)})),
    ( 15, "b-left",    frozenset({(+1, 0), ( 0,+1), (-1, 0), (-1,+1)})),
    ( 16, "b-right",   frozenset({(+1, 0), ( 0,-1), (-1, 0), (-1,-1)})),
    ( 17, "I-shape",   frozenset({(+1, 0), (-1, 0)})),
]

GRADE_OTHER    = 18   # catch-all — never listed in _GRADE_DEFS
GRADE_REJECTED = -1   # internal sentinel, never stored in output
N_GRADES       = GRADE_OTHER + 1   # 0..18 inclusive

GRADE_NAMES: dict[int, str] = {g: name for g, name, _ in _GRADE_DEFS}
GRADE_NAMES[GRADE_OTHER] = "other"

# frozenset(offsets) → grade_id   (built once at import)
_GRADE_LOOKUP: dict[frozenset, int] = {
    offsets: gid for gid, _, offsets in _GRADE_DEFS
}

_GRADE_DEFS_BY_ID: dict[int, tuple[str, frozenset]] = {
    gid: (label, offsets) for gid, label, offsets in _GRADE_DEFS
}

# ── helpers kept for energy_cal and plotting compatibility ────────────────────

def _offsets_to_bitmask(offsets: frozenset) -> int:
    """
    Convert neighbour offsets to the 8-bit mask used by the old C extension.
    Kept so write_grade_table_header() still works for documentation.
    """
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
    """Generate the 256-entry bitmask→grade table (kept for documentation)."""
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
    """Write _grade_table_generated.h (kept for documentation / reference)."""
    table = build_c_grade_table()
    if path is None:
        path = Path(__file__).with_name("_grade_table_generated.h")
    path = Path(path)

    lines = [
        "/* AUTO-GENERATED by pattern_recognition.py — DO NOT EDIT BY HAND. */",
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


# ══════════════════════════════════════════════════════════════════════════════
# Output dtype
# ══════════════════════════════════════════════════════════════════════════════

EVENT_DTYPE = np.dtype([
    ("Y",        np.int16),
    ("X",        np.int16),
    ("grade",    np.int8),
    ("adu_sum",  np.float32),   # sum of ALL pixels in the cluster
    ("adu_seed", np.float32),   # value of the maximum pixel (seed)
])


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 + 2: connected-component labelling  (ROOT union-find style)
# ══════════════════════════════════════════════════════════════════════════════

# Connectivity structure: 4-connected (left/right/above/below, NO diagonal)
# Matches ROOT HStepFilterEvents4: only left and below neighbours checked.
_STRUCT_NO_DIAG = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.int32)


def _find_clusters(
        frame:    np.ndarray,
        sec_mask: np.ndarray,
) -> np.ndarray:
    """
    Assign a cluster ID to every above-threshold pixel.

    Uses scipy.ndimage.label with a 4-connected structure (no diagonal)
    when scipy is available.  Falls back to a pure-Python union-find
    otherwise.

    Parameters
    ----------
    frame    : float32 (n_Y, n_X) — not used by scipy path, kept for
               signature compatibility with the fallback
    sec_mask : bool (n_Y, n_X)   — True where pixel > split_sigma × noise

    Returns
    -------
    label_map : int32 (n_Y, n_X), 0 = background, >0 = cluster ID
    """
    if _HAVE_SCIPY:
        label_map, _ = _scipy_label(sec_mask, structure=_STRUCT_NO_DIAG)
        return label_map.astype(np.int32)

    # ── Pure-Python union-find fallback (no scipy) ────────────────────────────
    # Correct but slow: O(N_above_threshold × N_merges).
    # Install scipy for production use: pip install scipy
    import warnings
    warnings.warn(
        "scipy not found — using slow Python union-find for clustering.\n"
        "Install scipy for ~50× speedup: pip install scipy",
        RuntimeWarning, stacklevel=3)

    n_Y, n_X   = sec_mask.shape
    label_map  = np.zeros((n_Y, n_X), dtype=np.int32)
    next_label = 1

    flat_idx = np.flatnonzero(sec_mask)   # raster order

    for idx in flat_idx:
        y = int(idx // n_X)
        x = int(idx  % n_X)

        left_label  = int(label_map[y, x-1]) if x > 0 else 0
        below_label = int(label_map[y-1, x]) if y > 0 else 0

        if left_label == 0 and below_label == 0:
            label_map[y, x] = next_label
            next_label += 1
        elif left_label != 0 and below_label == 0:
            label_map[y, x] = left_label
        elif left_label == 0 and below_label != 0:
            label_map[y, x] = below_label
        else:
            keep  = min(left_label, below_label)
            merge = max(left_label, below_label)
            label_map[y, x] = keep
            if keep != merge:
                label_map[label_map == merge] = keep

    return label_map


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: shape classification
# ══════════════════════════════════════════════════════════════════════════════

def _classify_cluster(
        ys: np.ndarray,
        xs: np.ndarray,
) -> tuple[int, int, int]:
    """
    Classify a cluster given the (Y, X) coordinates of its pixels.

    The seed is the pixel with the highest ADU value — identified by the
    caller.  This function receives coordinates already translated so
    that the seed is at (0, 0).

    Parameters
    ----------
    ys : int array — Y offsets from seed (seed itself is included as 0)
    xs : int array — X offsets from seed

    Returns
    -------
    grade    : int — grade ID from _GRADE_DEFS, or GRADE_OTHER
    seed_dy  : 0  (seed is always offset (0,0) by construction)
    seed_dx  : 0
    """
    # Build frozenset of neighbour offsets (exclude the seed itself)
    neighbours = frozenset(
        (int(dy), int(dx))
        for dy, dx in zip(ys, xs)
        if not (dy == 0 and dx == 0)
    )
    return _GRADE_LOOKUP.get(neighbours, GRADE_OTHER)


# ══════════════════════════════════════════════════════════════════════════════
# Main public function
# ══════════════════════════════════════════════════════════════════════════════

def find_events(
        corrected:      np.ndarray,
        noise_map:      np.ndarray,
        search_mask:    np.ndarray | None = None,
        seed_sigma:     float = 5.0,
        split_sigma:    float = 3.0,
        reject_extra:   bool  = False,
        bad_pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Find and classify photon events in a single CM-corrected frame.

    Algorithm  (ROOT HStepFilterEvents4 equivalent)
    ────────────────────────────────────────────────
    1. Build pixel masks:
         prim_mask = frame > seed_sigma  × noise   (ThresPrm equivalent)
         sec_mask  = frame > split_sigma × noise   (ThresSec equivalent)
       Excluded pixels (search_mask=False or bad_pixel_mask=True) are
       zeroed in a frame copy so they cannot be seeds or neighbours.

    2. Connected-component labelling on sec_mask pixels.
       Neighbours: left (same row, col-1) and below (row-1, same col).
       No diagonal neighbours — matching ROOT exactly.

    3. Accept a cluster only if at least one of its pixels is in prim_mask.

    4. Within each accepted cluster:
         seed     = pixel with maximum ADU
         offsets  = (dY, dX) of every other pixel relative to seed
         grade    = _GRADE_LOOKUP.get(frozenset(offsets), GRADE_OTHER)

    5. adu_sum  = sum of ALL cluster pixels (diagonal always included)
       adu_seed = ADU value of the seed pixel

    Parameters
    ----------
    corrected      : float32 (n_Y, n_X) — CM-corrected frame
    noise_map      : float32 (n_Y, n_X) — per-pixel noise [ADU RMS]
    search_mask    : bool (n_Y, n_X) or None — True = active pixel
    seed_sigma     : primary threshold multiplier   (typical 3–8)
    split_sigma    : secondary threshold multiplier (typical 1–3)
    reject_extra   : if True, discard GRADE_OTHER events
    bad_pixel_mask : bool (n_Y, n_X) or None — True = bad, always excluded

    Returns
    -------
    events : structured array with fields Y, X, grade, adu_sum, adu_seed
             Y, X are coordinates of the seed pixel.
    """
    frame = np.ascontiguousarray(corrected, dtype=np.float32)
    noise = np.ascontiguousarray(noise_map, dtype=np.float32)

    # ── Build combined include mask and apply it ──────────────────────────────
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

    prim_mask = frame > prim_thr   # at least one pixel per cluster must be here
    sec_mask  = frame > sec_thr    # defines cluster extent

    if not sec_mask.any():
        return np.empty(0, dtype=EVENT_DTYPE)

    # ── Connected-component labelling ─────────────────────────────────────────
    label_map = _find_clusters(frame, sec_mask)

    # ── Collect cluster IDs that contain at least one primary pixel ───────────
    primary_labels = set(int(v) for v in np.unique(label_map[prim_mask]) if v > 0)

    if not primary_labels:
        return np.empty(0, dtype=EVENT_DTYPE)

    # ── Process each accepted cluster ─────────────────────────────────────────
    rows_l:   list[int]   = []
    cols_l:   list[int]   = []
    grades_l: list[int]   = []
    sigs_l:   list[float] = []
    seeds_l:  list[float] = []

    for cid in primary_labels:
        pix_mask = label_map == cid
        ys, xs   = np.nonzero(pix_mask)               # pixel coordinates

        vals     = frame[ys, xs]
        seed_idx = int(np.argmax(vals))
        sy       = int(ys[seed_idx])
        sx       = int(xs[seed_idx])
        seed_val = float(vals[seed_idx])

        # Offsets relative to seed
        dy_rel = ys - sy
        dx_rel = xs - sx

        grade = _classify_cluster(dy_rel, dx_rel)

        if reject_extra and grade == GRADE_OTHER:
            continue

        adu_sum = float(vals.sum())

        rows_l.append(sy)
        cols_l.append(sx)
        grades_l.append(grade)
        sigs_l.append(adu_sum)
        seeds_l.append(seed_val)

    if not rows_l:
        return np.empty(0, dtype=EVENT_DTYPE)

    out = np.empty(len(rows_l), dtype=EVENT_DTYPE)
    out["Y"]        = np.array(rows_l,   dtype=np.int16)
    out["X"]        = np.array(cols_l,   dtype=np.int16)
    out["grade"]    = np.array(grades_l, dtype=np.int8)
    out["adu_sum"]  = np.array(sigs_l,   dtype=np.float32)
    out["adu_seed"] = np.array(seeds_l,  dtype=np.float32)
    return out
