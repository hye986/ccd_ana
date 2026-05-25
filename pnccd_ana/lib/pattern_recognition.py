"""
pnccd_ana.lib.pattern_recognition
==================================
Photon-event recognition with two-threshold design.

  seed_sigma  : higher threshold (e.g. 3–8 × noise) to find candidate centres
  split_sigma : lower threshold  (e.g. 1–3 × noise) to classify neighbours

A seed must:
  • exceed seed_sigma × noise,
  • equal the maximum of its 5×5 window, and
  • be the lexicographically smallest (smallest y; on tie, smallest x)
    among any pixels in that window that share the same value.
The last condition is a deterministic tie-breaker, so an evenly-split
charge cluster (or a hot 2×2 patch) yields exactly one event instead of
multiple overlapping duplicates.

The hot path (local_max_5x5 + inner per-pixel loop + grade lookup) runs in C
when the compiled extension is available, with a pure-NumPy fallback otherwise.

Grade definitions (central 3×3 of the 5×5 sub-matrix, centre at local index (2,2))
Offsets (dY, dX) from centre — the centre (0,0) is always included implicitly:
  0  single      –
  1  double      (0,+1)
  2  double      (+1,0)
  3  double      (0,-1)
  4  double      (-1,0)
  5  triple      (+1,0)+(0,+1)
  6  triple      (0,-1)+(+1,0)
  7  triple      (-1,0)+(0,-1)
  8  triple      (-1,0)+(0,+1)
  9  quadruple   (+1,0)+(0,+1)+(+1,+1)
 10  quadruple   (+1,0)+(0,-1)+(+1,-1)
 11  quadruple   (-1,0)+(0,-1)+(-1,-1)
 12  quadruple   (-1,0)+(0,+1)+(-1,+1)
 13  other       (centre is max, pattern not in 0-12)
 -1  rejected    (centre is not the 5×5 local maximum)

──────────────────────────────────────────────────────────────────────────────
SINGLE SOURCE OF TRUTH FOR GRADE DEFINITIONS
──────────────────────────────────────────────────────────────────────────────
All grade information lives in _GRADE_DEFS below.  The C extension's 256-entry
grade lookup table is generated from _GRADE_DEFS at import time and written to
a tiny header file (_grade_table_generated.h) that the C extension #includes.
If you add or modify grades, edit _GRADE_DEFS only — the C table is derived
automatically.

Neighbour bit-encoding (matches C extension):
  bit 0 (0x01) : right      (dY= 0, dX=+1)
  bit 1 (0x02) : up         (dY=+1, dX= 0)
  bit 2 (0x04) : left       (dY= 0, dX=-1)
  bit 3 (0x08) : down       (dY=-1, dX= 0)
  bit 4 (0x10) : up-right   (dY=+1, dX=+1)
  bit 5 (0x20) : up-left    (dY=+1, dX=-1)
  bit 6 (0x40) : down-left  (dY=-1, dX=-1)
  bit 7 (0x80) : down-right (dY=-1, dX=+1)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

# ── try to load the C extension ───────────────────────────────────────────────
try:
    from . import _pattern_recognition_c as _cext
    _HAVE_C = True
except ImportError:
    _HAVE_C = False

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE SOURCE OF TRUTH — edit only this table to add/remove grades
#
# Each entry: (grade_id, label, neighbour_offsets_frozenset)
# neighbour_offsets contains (dY, dX) pairs *excluding* the centre (0,0).
# The centre is always part of every pattern.
#
# Bit mapping for the 8 neighbours of the central 3×3:
#   (dY, dX) → bit
_OFFSET_TO_BIT: dict[tuple[int,int], int] = {
    ( 0,+1): 0,   # right
    (+1, 0): 1,   # up
    ( 0,-1): 2,   # left
    (-1, 0): 3,   # down
    (+1,+1): 4,   # up-right
    (+1,-1): 5,   # up-left
    (-1,-1): 6,   # down-left
    (-1,+1): 7,   # down-right
}

_GRADE_DEFS: list[tuple[int, str, frozenset]] = [
    # ── grade  label       neighbour offsets (dY, dX), centre excluded ──
    (  0, "single",    frozenset()),
    # doubles
    (  1, "double",    frozenset({( 0,+1)})),          # right
    (  2, "double",    frozenset({(+1, 0)})),          # up
    (  3, "double",    frozenset({( 0,-1)})),          # left
    (  4, "double",    frozenset({(-1, 0)})),          # down
    # triples
    (  5, "triple",    frozenset({(+1, 0), ( 0,+1)})),  # up+right
    (  6, "triple",    frozenset({( 0,-1), (+1, 0)})),  # left+up
    (  7, "triple",    frozenset({(-1, 0), ( 0,-1)})),  # down+left
    (  8, "triple",    frozenset({(-1, 0), ( 0,+1)})),  # down+right
    # quadruples (cardinal pair + diagonal corner)
    (  9, "quadruple", frozenset({(+1, 0), ( 0,+1), (+1,+1)})),  # up+right+ur
    ( 10, "quadruple", frozenset({(+1, 0), ( 0,-1), (+1,-1)})),  # up+left+ul
    ( 11, "quadruple", frozenset({(-1, 0), ( 0,-1), (-1,-1)})),  # down+left+dl
    ( 12, "quadruple", frozenset({(-1, 0), ( 0,+1), (-1,+1)})),  # down+right+dr
    # ── Add new grades here.  Example (uncomment to enable):
    # ( 14, "L-shape",  frozenset({(+1, 0), ( 0,+1), (+1,-1)})),
    # ─────────────────────────────────────────────────────────────────────────
]

# grade 13 ("other") is the catch-all — never listed in _GRADE_DEFS
GRADE_OTHER    = 13
GRADE_REJECTED = -1
N_GRADES       = GRADE_OTHER + 1   # 14 named grades including "other"

# Derived lookups — built once from _GRADE_DEFS
GRADE_NAMES: dict[int, str] = {g: name for g, name, _ in _GRADE_DEFS}
GRADE_NAMES[GRADE_OTHER] = "other"

_GRADE_LOOKUP: dict[frozenset, int] = {
    offsets: gid for gid, _, offsets in _GRADE_DEFS
}


def _offsets_to_bitmask(offsets: frozenset) -> int:
    """Convert a set of (dY, dX) neighbour offsets to the 8-bit mask used by C."""
    mask = 0
    for off in offsets:
        if off not in _OFFSET_TO_BIT:
            raise ValueError(
                f"Offset {off} is not one of the 8 central-3×3 neighbours. "
                "Only offsets with |dY|≤1 and |dX|≤1 (excluding (0,0)) are valid."
            )
        mask |= (1 << _OFFSET_TO_BIT[off])
    return mask


def build_c_grade_table() -> list[int]:
    """
    Generate the 256-entry grade lookup table for the C extension.
    Entry i = grade for neighbour bitmask i.  Default is GRADE_OTHER (13).
    """
    table = [GRADE_OTHER] * 256
    for gid, _, offsets in _GRADE_DEFS:
        mask = _offsets_to_bitmask(offsets)
        if table[mask] != GRADE_OTHER:
            raise RuntimeError(
                f"Grade {gid} bitmask 0x{mask:02X} collides with "
                f"grade {table[mask]} — check _GRADE_DEFS for duplicates."
            )
        table[mask] = gid
    return table


def write_grade_table_header(path: str | Path | None = None) -> Path:
    """
    Write _grade_table_generated.h next to this file (or to *path*).
    The C extension #includes this file instead of hardcoding the table.

    Call this whenever _GRADE_DEFS changes, then recompile the extension.
    """
    table = build_c_grade_table()
    if path is None:
        path = Path(__file__).with_name("_grade_table_generated.h")
    path = Path(path)

    lines = [
        "/* AUTO-GENERATED by pattern_recognition.py — DO NOT EDIT BY HAND. */",
        "/* Re-generate with: python -c \"from pattern_recognition import "
        "write_grade_table_header; write_grade_table_header()\" */",
        "",
        "/* Number of named grades (excluding catch-all 'other'). */",
        f"#define N_GRADE_DEFS {len(_GRADE_DEFS)}",
        f"#define GRADE_OTHER  {GRADE_OTHER}",
        "",
        "/* 256-entry table: bitmask → grade. */",
        "static const uint8_t GRADE_TABLE[256] = {",
    ]
    for i in range(0, 256, 16):
        row = ", ".join(f"{table[i+j]:3d}" for j in range(16))
        lines.append(f"    {row},  /* 0x{i:02X}–0x{i+15:02X} */")
    lines += ["};", ""]

    # Also emit the PATTERN_PIXELS array so the C file has zero grade knowledge
    # Pattern pixels: list of (dY, dX) including centre (0,0), max 4 pixels.
    all_offsets: list[list[tuple[int,int]]] = []
    for gid, _, offsets in _GRADE_DEFS:
        pixels = [(0, 0)] + sorted(offsets)   # centre first, then neighbours
        all_offsets.append(pixels)
    # grade 13 "other": centre only
    all_offsets.append([(0, 0)])

    max_pix = max(len(p) for p in all_offsets) + 1  # +1 for terminator
    lines += [
        f"/* PATTERN_PIXELS[grade][pixel] = {{dY, dX}}; {{99,99}} terminates. */",
        f"#define MAX_PATTERN_PIXELS {max_pix}",
        f"static const int8_t PATTERN_PIXELS[{len(all_offsets)}][{max_pix}][2] = {{",
    ]
    for gid_idx, pixels in enumerate(all_offsets):
        # Map to the actual grade id
        if gid_idx < len(_GRADE_DEFS):
            actual_gid = _GRADE_DEFS[gid_idx][0]
            label = _GRADE_DEFS[gid_idx][1]
        else:
            actual_gid = GRADE_OTHER
            label = "other"
        entries = ", ".join(f"{{{dy:+d},{dx:+d}}}" for dy, dx in pixels)
        entries += ", {99,99}" * (max_pix - len(pixels))
        lines.append(f"    /* grade {actual_gid:2d} {label:10s} */ {{{entries}}},")
    lines += ["};", ""]

    path.write_text("\n".join(lines))
    return path


# ── validate that C extension was built from the same grade table ─────────────

def _validate_c_extension() -> bool:
    """
    Check that the loaded C extension agrees with _GRADE_DEFS by probing
    a sample of bitmasks.  Logs a warning if they differ.
    """
    if not _HAVE_C:
        return False
    table = build_c_grade_table()
    try:
        mismatches = _cext.check_grade_table(table)
    except AttributeError:
        # Older C extension without check_grade_table() — skip silently
        return True
    if mismatches:
        import warnings
        warnings.warn(
            f"C extension grade table differs from _GRADE_DEFS at "
            f"{len(mismatches)} bitmask(s): {mismatches[:5]}... "
            "Recompile the extension after running write_grade_table_header().",
            RuntimeWarning, stacklevel=2,
        )
        return False
    return True


_HAVE_C = _HAVE_C and _validate_c_extension()

EVENT_DTYPE = np.dtype([
    ("Y",        np.int16),
    ("X",        np.int16),
    ("grade",    np.int8),
    ("adu_sum",  np.float32),   # summed charge of all cluster pixels
    ("adu_seed", np.float32),   # centre pixel value only
])


# ── 5×5 local maximum ─────────────────────────────────────────────────────────

def local_max_5x5(frame: np.ndarray) -> np.ndarray:
    """
    5×5 sliding-window maximum of *frame*.
    Uses C extension when available, then scipy, then pure numpy.
    """
    f32 = np.ascontiguousarray(frame, dtype=np.float32)
    if _HAVE_C:
        return _cext.local_max_5x5(f32)
    try:
        from scipy.ndimage import maximum_filter
        return maximum_filter(f32, size=5, mode="nearest").astype(np.float32)
    except ImportError:
        pass
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(f32, 2, mode="edge")
    windows = sliding_window_view(pad, (5, 5))
    return windows.max(axis=(-2, -1)).astype(np.float32)


# ── grade classifier (Python fallback) ────────────────────────────────────────

def _classify_patch(above: np.ndarray) -> int:
    """
    Map the central 3×3 above-threshold boolean patch to a grade (0-13).
    above: 5×5 bool array, centre at [2,2].
    Returns GRADE_REJECTED (-1) if centre is not above threshold.
    """
    if not above[2, 2]:
        return GRADE_REJECTED
    neighbours: frozenset = frozenset(
        (dY, dX)
        for dY in range(-1, 2)
        for dX in range(-1, 2)
        if not (dY == 0 and dX == 0) and above[2 + dY, 2 + dX]
    )
    return _GRADE_LOOKUP.get(neighbours, GRADE_OTHER)


# ── grade distribution diagnostic ─────────────────────────────────────────────

def grade_bitmask_histogram(
        corrected:   np.ndarray,
        noise_map:   np.ndarray,
        seed_sigma:  float = 5.0,
        split_sigma: float = 3.0,
) -> dict[int, int]:
    """
    Return a histogram of raw neighbour bitmasks for all seed candidates.
    Useful for diagnosing what patterns make up the 'other' (grade 13) fraction.

    Returns dict mapping bitmask (0–255) → count.
    """
    frame = np.ascontiguousarray(corrected, dtype=np.float32)
    noise = np.ascontiguousarray(noise_map, dtype=np.float32)
    lmax  = local_max_5x5(frame)

    n_Y, n_X   = frame.shape
    seed_thr   = seed_sigma  * noise
    split_thr  = split_sigma * noise

    hist: dict[int, int] = {}

    for y in range(2, n_Y - 2):
        for x in range(2, n_X - 2):
            cv = frame[y, x]
            if cv <= seed_thr[y, x]:
                continue
            if cv < lmax[y, x]:
                continue
            # Same lex-smallest tie-breaker as find_events(), so the histogram
            # reflects exactly the seed set that find_events() will produce.
            window = frame[y-2:y+3, x-2:x+3]
            if (window[:2, :] == cv).any() or (window[2, :2] == cv).any():
                continue
            mask = 0
            for (dY, dX), bit in _OFFSET_TO_BIT.items():
                if frame[y+dY, x+dX] > split_thr[y+dY, x+dX]:
                    mask |= (1 << bit)
            hist[mask] = hist.get(mask, 0) + 1
    return hist


def summarise_unknown_patterns(
        corrected:   np.ndarray,
        noise_map:   np.ndarray,
        seed_sigma:  float = 5.0,
        split_sigma: float = 3.0,
        top_n:       int   = 20,
) -> None:
    """
    Print the most common unrecognised bitmasks to help decide if new grades
    should be added to _GRADE_DEFS.
    """
    known_masks = {_offsets_to_bitmask(offsets) for _, _, offsets in _GRADE_DEFS}
    hist = grade_bitmask_histogram(corrected, noise_map, seed_sigma, split_sigma)
    unknown = {k: v for k, v in hist.items() if k not in known_masks}
    total = sum(hist.values())
    total_unknown = sum(unknown.values())

    print(f"Total candidates: {total}  Unknown: {total_unknown} "
          f"({100*total_unknown/max(total,1):.1f}%)")
    print(f"\nTop {top_n} unrecognised bitmasks:")
    print(f"  {'Mask':>6}  {'Count':>7}  {'%tot':>6}  Neighbours")
    for mask, count in sorted(unknown.items(), key=lambda kv: -kv[1])[:top_n]:
        offs = [name for name, bit_idx in {
            'R':0,'U':1,'L':2,'D':3,'UR':4,'UL':5,'DL':6,'DR':7}.items()
                if mask & (1 << bit_idx)]
        print(f"  0x{mask:02X}={mask:3d}  {count:7d}  {100*count/total:5.1f}%"
              f"  {'+'.join(offs) or '(none)'}")


# ── main recognition function ─────────────────────────────────────────────────

def find_events(
        corrected:       np.ndarray,
        noise_map:       np.ndarray,
        search_mask:     np.ndarray | None = None,
        seed_sigma:      float = 5.0,
        split_sigma:     float = 3.0,
        reject_extra:    bool  = False,
        bad_pixel_mask:  np.ndarray | None = None,
) -> np.ndarray:
    """
    Find and classify photon events in a single CM-corrected frame.

    Seed selection rule
    ───────────────────
    A pixel becomes a seed when it satisfies, in order:
      1. centre value  > seed_sigma × noise(centre)               (seed threshold)
      2. centre value == max(5×5 window centred on it)            (local max)
      3. centre is the lexicographically smallest pixel (smallest y;
         on tie, smallest x) among all 5×5-window pixels that share
         the same value                                            (tie-breaker)

    The third rule guarantees that an evenly-split charge cluster, a
    2×2 hot-pixel patch, or any other tied configuration produces exactly
    one event instead of N overlapping duplicates.

    Grade is then determined from the central 3×3 mask of pixels exceeding
    split_sigma × noise (the centre is implicitly included).

    Pixel masking
    ─────────────
    Two boolean masks may be supplied.  Pixels excluded by either are
    treated as zero in the frame copy used for recognition; consequently
    they cannot become a seed AND they never contribute to a neighbour's
    bitmask or ADU sum:

      search_mask    : True where the pixel is inside the active ASIC region
                       (geometric).  None → use the whole frame.
      bad_pixel_mask : True where the pixel is BAD (hot / cold / unstable —
                       typically from :func:`pnccd_ana.lib.noise.build_bad_pixel_mask`).
                       None → no bad-pixel filtering.

    Parameters
    ----------
    corrected      : float32 (n_Y, n_X) — CM-corrected frame
    noise_map      : float32 (n_Y, n_X) — per-pixel noise [ADU RMS]
    search_mask    : bool (n_Y, n_X) or None — geometric active mask
                     (True = inside the ASIC region).
    seed_sigma     : seed-detection threshold multiplier (typical 3–8)
    split_sigma    : neighbour-classification threshold multiplier (typical 1–3)
    reject_extra   : if True, discard grade-13 ("other") events
    bad_pixel_mask : bool (n_Y, n_X) or None — True where the pixel is bad
                     and must be excluded from event recognition.

    Returns
    -------
    events : structured array with fields Y, X, grade, adu_sum, adu_seed.
    """
    frame = np.ascontiguousarray(corrected, dtype=np.float32)
    noise = np.ascontiguousarray(noise_map, dtype=np.float32)

    # Combine the geometric search_mask (good=True) with the bad-pixel mask
    # (bad=True) into a single "include" mask.  Zero-filling everything
    # outside that mask in a frame copy is enough to suppress both seed
    # detection AND neighbour contribution in one step (a zero pixel cannot
    # exceed split_sigma × noise as long as noise > 0).
    include = None
    if search_mask is not None:
        include = search_mask.astype(bool, copy=False)
    if bad_pixel_mask is not None:
        bad_bool = bad_pixel_mask.astype(bool, copy=False)
        include  = (~bad_bool) if include is None else (include & ~bad_bool)
    if include is not None:
        frame = frame.copy()
        frame[~include] = 0.0

    lmax = local_max_5x5(frame)

    if _HAVE_C:
        rows, cols, grades, signals = _cext.find_events_c(
            frame, noise, lmax,
            float(seed_sigma), float(split_sigma),
            int(reject_extra),
        )
        n = len(rows)
        if n == 0:
            return np.empty(0, dtype=EVENT_DTYPE)
        out = np.empty(n, dtype=EVENT_DTYPE)
        out["Y"]        = rows.astype(np.int16)
        out["X"]        = cols.astype(np.int16)
        out["grade"]    = grades.astype(np.int8)
        out["adu_sum"]  = signals
        out["adu_seed"] = frame[rows.astype(int), cols.astype(int)]
        return out

    # ── pure Python / NumPy fallback ──────────────────────────────────────────
    n_Y, n_X  = frame.shape
    seed_thr  = seed_sigma  * noise
    split_thr = split_sigma * noise

    centres   = (
        (frame[2:-2, 2:-2] > seed_thr[2:-2, 2:-2]) &
        (frame[2:-2, 2:-2] == lmax[2:-2, 2:-2])
    )
    cand_Y, cand_X = np.nonzero(centres)
    cand_Y += 2
    cand_X += 2

    rows_l, cols_l, grades_l, sigs_l, seeds_l = [], [], [], [], []

    for cy, cx in zip(cand_Y, cand_X):
        cv = float(frame[cy, cx])

        # Tie-breaker: if any pixel in the 5×5 window with a strictly
        # smaller (y, x) lexicographic position has the same value as the
        # centre, this candidate loses the tie and is skipped.  The
        # lexicographically smallest tied pixel becomes the unique seed for
        # the cluster.  Matches the C extension exactly.
        window = frame[cy-2:cy+3, cx-2:cx+3]
        if (window[:2, :] == cv).any() or (window[2, :2] == cv).any():
            continue

        above = window > split_thr[cy-2:cy+3, cx-2:cx+3]
        grade = _classify_patch(above)
        if grade == GRADE_REJECTED or (reject_extra and grade == GRADE_OTHER):
            continue

        if grade == 0 or grade == GRADE_OTHER:
            sig = cv
        else:
            _, _, offsets = _GRADE_DEFS[grade]
            sig = cv + sum(float(frame[cy+dY, cx+dX]) for dY, dX in offsets)

        rows_l.append(int(cy));   cols_l.append(int(cx))
        grades_l.append(grade);   sigs_l.append(sig)
        seeds_l.append(cv)

    if not rows_l:
        return np.empty(0, dtype=EVENT_DTYPE)

    out = np.empty(len(rows_l), dtype=EVENT_DTYPE)
    out["Y"]        = np.array(rows_l,   dtype=np.int16)
    out["X"]        = np.array(cols_l,   dtype=np.int16)
    out["grade"]    = np.array(grades_l, dtype=np.int8)
    out["adu_sum"]  = np.array(sigs_l,   dtype=np.float32)
    out["adu_seed"] = np.array(seeds_l,  dtype=np.float32)
    return out
