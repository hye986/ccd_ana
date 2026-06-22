"""
pnccd_ana.physics.event_filter
==============================
Photon-event detection using connected-component clustering.

Matches ROOT HStepFilterEvents4 — stores raw cluster data without
grade assignment.  Grade classification is deferred to the energy
calibration step where gain and CTI corrections stabilise the argmax
pixel and make pattern recognition reliable.

Algorithm
─────────
  1. THRESHOLD SCAN
       sec_mask  = frame > split_sigma × noise   (ROOT ThresSec)
       prim_mask = frame > seed_sigma  × noise   (ROOT ThresPrm)
     ThresSec is clamped to ThresPrm per pixel (ROOT behaviour).
     Excluded pixels (bad_pixel_mask, search_mask) are zeroed.

  2. 4-CONNECTED COMPONENT LABELLING on sec_mask.
     No diagonal neighbours — matches ROOT HStepFilterEvents4 exactly.
     scipy.ndimage.label used when available; pure-Python union-find
     fallback otherwise.

  3. ACCEPT cluster if ≥1 pixel > prim_mask (ROOT: HasExceededPrimThresh).

  4. STORE per cluster:
       Y, X      — seed pixel coordinates (argmax ADU)
       adu_sum   — sum of ALL cluster pixels (gain-independent)
       adu_seed  — ADU of seed pixel
       n_pixels  — cluster size (1=single, 2=double, 3=triple, 4=quad, ...)
       flag      — border / overflow / underflow bitmask

  Grade assignment is NOT done here.  After energy calibration:
       n_pixels == 1  →  single   (grade 0)
       n_pixels == 2  →  double   (grades 1–4)
       n_pixels == 3  →  triple   (grades 5–8)
       n_pixels == 4  →  quadruple (grades 9–12 or other)
       n_pixels >= 5  →  larger pattern

Coordinate convention
──────────────────────
  data[Y, X]   Y = row (axis 0),  X = col (axis 1)
  Y = 0 is the first readout row (rolling shutter bottom).

Grade table (kept here for energy_cal reuse)
─────────────────────────────────────────────
  Neighbour offsets are (dY, dX) relative to the seed pixel (0, 0).
  The seed is the maximum-ADU pixel in the gain-corrected cluster.

   0  single      no neighbours
   1  double      (0,+1)
   2  double      (+1, 0)
   3  double      (0,-1)
   4  double      (-1, 0)
   5  triple      (+1, 0)+(0,+1)
   6  triple      (0,-1)+(+1, 0)
   7  triple      (-1, 0)+(0,-1)
   8  triple      (-1, 0)+(0,+1)
   9  quadruple   (+1, 0)+(0,+1)+(+1,+1)   2×2 seed=bottom-left
  10  quadruple   (+1, 0)+(0,-1)+(+1,-1)   2×2 seed=bottom-right
  11  quadruple   (-1, 0)+(0,-1)+(-1,-1)   2×2 seed=top-right
  12  quadruple   (-1, 0)+(0,+1)+(-1,+1)   2×2 seed=top-left
  13  other       any unrecognised shape
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
# Grade table — kept here so energy_cal can import it for post-cal grading
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
N_GRADES       = GRADE_OTHER + 1   # 0..13 inclusive

GRADE_NAMES: dict[int, str] = {g: name for g, name, _ in _GRADE_DEFS}
GRADE_NAMES[GRADE_OTHER] = "other"

_GRADE_LOOKUP: dict[frozenset, int] = {
    offsets: gid for gid, _, offsets in _GRADE_DEFS
}

_GRADE_DEFS_BY_ID: dict[int, tuple[str, frozenset]] = {
    gid: (label, offsets) for gid, label, offsets in _GRADE_DEFS
}


# ══════════════════════════════════════════════════════════════════════════════
# Output dtype  (Option A: no grade field)
# ══════════════════════════════════════════════════════════════════════════════

EVENT_DTYPE = np.dtype([
    ("Y",        np.int16),    # seed pixel row
    ("X",        np.int16),    # seed pixel column
    ("adu_sum",  np.float32),  # sum of ALL cluster pixels
    ("adu_seed", np.float32),  # ADU of seed (argmax) pixel
    ("n_pixels", np.uint8),    # cluster size (1=single,2=double,3=triple,4=quad)
    ("flag",     np.uint8),    # bitmask: border/overflow/underflow
])

# Flag bits (matching ROOT HEventFlags kPix* values)
FLAG_BORDER    = np.uint8(1 << 3)   # kPixBorder
FLAG_OVERFLOW  = np.uint8(1 << 0)   # kPixOverflow
FLAG_UNDERFLOW = np.uint8(1 << 1)   # kPixUnderflow
FLAG_MISFIT    = np.uint8(1 << 2)   # kPixMisfit


# ══════════════════════════════════════════════════════════════════════════════
# Connected-component labelling (4-connected, no diagonal)
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
# Grade classification (used by energy_cal after gain+CTI correction)
# ══════════════════════════════════════════════════════════════════════════════

def classify_cluster(ys: np.ndarray,
                     xs: np.ndarray,
                     seed_idx: int) -> int:
    """
    Classify cluster shape relative to the seed (argmax) pixel.

    Should be called AFTER gain and CTI correction so that the argmax
    pixel is stable and the pattern is reliably recognised.

    Parameters
    ----------
    ys       : int array — Y coordinates of all cluster pixels
    xs       : int array — X coordinates of all cluster pixels
    seed_idx : int — index of the argmax (seed) pixel

    Returns
    -------
    grade : int — from _GRADE_LOOKUP or GRADE_OTHER
    """
    sy, sx = ys[seed_idx], xs[seed_idx]
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
        bad_pixel_mask:    np.ndarray | None = None,
        clamp_sec_to_prim: bool  = True,
        flag_border:       bool  = True,
) -> np.ndarray:
    """
    Find photon events in a single CM-corrected frame.

    Matches ROOT HStepFilterEvents4 — stores raw cluster data only.
    Grade assignment is deferred to energy_cal.

    Parameters
    ----------
    corrected          : float32 (n_Y, n_X) — CM-corrected frame
    noise_map          : float32 (n_Y, n_X) — per-pixel noise [ADU RMS]
    search_mask        : bool (n_Y, n_X) or None — True = active pixel
    seed_sigma         : primary threshold   (ROOT ThresPrm, typ. 5)
    split_sigma        : secondary threshold (ROOT ThresSec, typ. 3)
    bad_pixel_mask     : bool (n_Y, n_X) or None
    clamp_sec_to_prim  : clamp sec threshold to prim per pixel (ROOT)
    flag_border        : attach FLAG_BORDER to edge-touching events

    Returns
    -------
    events : structured array EVENT_DTYPE
             fields: Y, X, adu_sum, adu_seed, n_pixels, flag
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
        return np.empty(0, dtype=EVENT_DTYPE)

    # ── Clustering ────────────────────────────────────────────────────────────
    label_map = _find_clusters(sec_mask)

    primary_labels = set(
        int(v) for v in np.unique(label_map[prim_mask]) if v > 0
    )
    if not primary_labels:
        return np.empty(0, dtype=EVENT_DTYPE)

    # ── Build output arrays ───────────────────────────────────────────────────
    rows_l:    list[int]   = []
    cols_l:    list[int]   = []
    sigs_l:    list[float] = []
    seeds_l:   list[float] = []
    npix_l:    list[int]   = []
    flags_l:   list[int]   = []

    for cid in primary_labels:
        pix_mask = label_map == cid
        ys, xs   = np.nonzero(pix_mask)
        vals     = frame[ys, xs]

        seed_idx = int(np.argmax(vals))
        sy       = int(ys[seed_idx])
        sx       = int(xs[seed_idx])

        evt_flag = np.uint8(0)
        if flag_border and border_mask[ys, xs].any():
            evt_flag |= FLAG_BORDER

        rows_l.append(sy)
        cols_l.append(sx)
        sigs_l.append(float(vals.sum()))
        seeds_l.append(float(vals[seed_idx]))
        npix_l.append(int(len(ys)))
        flags_l.append(int(evt_flag))

    if not rows_l:
        return np.empty(0, dtype=EVENT_DTYPE)

    out = np.empty(len(rows_l), dtype=EVENT_DTYPE)
    out["Y"]        = np.array(rows_l,  dtype=np.int16)
    out["X"]        = np.array(cols_l,  dtype=np.int16)
    out["adu_sum"]  = np.array(sigs_l,  dtype=np.float32)
    out["adu_seed"] = np.array(seeds_l, dtype=np.float32)
    out["n_pixels"] = np.array(npix_l,  dtype=np.uint8)
    out["flag"]     = np.array(flags_l, dtype=np.uint8)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Helpers for energy_cal
# ══════════════════════════════════════════════════════════════════════════════

def _offsets_to_bitmask(offsets: frozenset) -> int:
    _OFFSET_TO_BIT = {
        ( 0,+1): 0, (+1, 0): 1, ( 0,-1): 2, (-1, 0): 3,
        (+1,+1): 4, (+1,-1): 5, (-1,-1): 6, (-1,+1): 7,
    }
    mask = 0
    for off in offsets:
        if off not in _OFFSET_TO_BIT:
            raise ValueError(f"Offset {off} outside 3×3 neighbourhood.")
        mask |= (1 << _OFFSET_TO_BIT[off])
    return mask


def build_c_grade_table() -> list[int]:
    """Build 256-entry lookup table for C/GPU grade classification."""
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
    """Write C header with grade lookup table for downstream C/GPU code."""
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
