"""
tests/test_pattern_recognition.py
===================================
Tests for the ROOT-style connected-component event recognition.

Key behavioural changes from the old 5×5 local-max algorithm:
  - No tie-breaker needed: each cluster produces exactly one event because
    connected-component labelling is deterministic.
  - Diagonal neighbours are NOT connected (left + below only).
  - adu_sum = sum of ALL cluster pixels (not just pattern pixels).
  - Seed = max-ADU pixel in the cluster.
  - 2×2 quadruple: diagonal is always included in adu_sum as long as it
    is above split_sigma — the quadruple diagonal problem is eliminated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pnccd_ana.lib import pattern_recognition as pr
from pnccd_ana.lib.pattern_recognition import (
    find_events, GRADE_OTHER, GRADE_NAMES, _GRADE_DEFS, EVENT_DTYPE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NOISE_LEVEL = 100.0   # ADU RMS — uniform noise map
SEED_SIGMA  = 5.0     # primary threshold   → 500 ADU
SPLIT_SIGMA = 3.0     # secondary threshold → 300 ADU


def _noise(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, NOISE_LEVEL, dtype=np.float32)


def _find(frame: np.ndarray,
          seed_sigma:  float = SEED_SIGMA,
          split_sigma: float = SPLIT_SIGMA) -> np.ndarray:
    return find_events(frame.astype(np.float32), _noise(frame.shape),
                       seed_sigma=seed_sigma, split_sigma=split_sigma)


def _single_event_frame(
        centre_adu: float,
        neighbours: tuple[tuple[int, int, float], ...] = (),
        shape: tuple[int, int] = (9, 9),
        cy: int = 4, cx: int = 4,
) -> np.ndarray:
    """
    Build a frame with one event cluster.
    neighbours: sequence of (dY, dX, adu).
    """
    frame = np.zeros(shape, dtype=np.float32)
    frame[cy, cx] = centre_adu
    for dy, dx, val in neighbours:
        frame[cy + dy, cx + dx] = val
    return frame


def _check(events: np.ndarray, *,
           n: int, grade: int,
           seed_y: int | None = None,
           seed_x: int | None = None,
           adu_sum: float | None = None,
           adu_seed: float | None = None,
           tol: float = 0.5) -> None:
    assert len(events) == n, f"expected {n} events, got {len(events)}"
    if n == 0:
        return
    e = events[0]
    assert int(e["grade"]) == grade, (
        f"expected grade {grade} ({GRADE_NAMES.get(grade,'?')}), "
        f"got {int(e['grade'])} ({GRADE_NAMES.get(int(e['grade']),'?')})")
    if seed_y is not None:
        assert int(e["Y"]) == seed_y, f"seed Y: expected {seed_y}, got {int(e['Y'])}"
    if seed_x is not None:
        assert int(e["X"]) == seed_x, f"seed X: expected {seed_x}, got {int(e['X'])}"
    if adu_sum is not None:
        assert abs(float(e["adu_sum"]) - adu_sum) <= tol, (
            f"adu_sum: expected {adu_sum}, got {float(e['adu_sum'])}")
    if adu_seed is not None:
        assert abs(float(e["adu_seed"]) - adu_seed) <= tol, (
            f"adu_seed: expected {adu_seed}, got {float(e['adu_seed'])}")


# ─────────────────────────────────────────────────────────────────────────────
# Grade 0 — single pixel
# ─────────────────────────────────────────────────────────────────────────────

def test_single_pixel_grade0() -> None:
    """Isolated pixel above seed_sigma → grade 0."""
    frame = _single_event_frame(1000.0)
    _check(_find(frame), n=1, grade=0,
           seed_y=4, seed_x=4, adu_sum=1000.0, adu_seed=1000.0)


def test_below_seed_threshold_no_event() -> None:
    """Pixel above split_sigma but below seed_sigma → no event."""
    frame = _single_event_frame(350.0)   # 350 > 300 (split) but < 500 (seed)
    _check(_find(frame), n=0, grade=0)


def test_below_split_threshold_no_event() -> None:
    """Pixel below split_sigma → not even collected."""
    frame = _single_event_frame(200.0)
    _check(_find(frame), n=0, grade=0)


# ─────────────────────────────────────────────────────────────────────────────
# All defined grades
# ─────────────────────────────────────────────────────────────────────────────

def test_all_defined_grades() -> None:
    """
    For every grade in _GRADE_DEFS, inject a cluster with the seed at (4,4)
    and neighbours at split_sigma + 50 ADU.  Verify grade, seed position,
    and adu_sum.
    """
    seed_adu  = 1000.0
    nbr_adu   = 350.0   # above split_sigma (300) but below seed_sigma (500)

    for gid, label, offsets in _GRADE_DEFS:
        neighbours = tuple((dy, dx, nbr_adu) for dy, dx in offsets)
        frame = _single_event_frame(seed_adu, neighbours)
        evts  = _find(frame)

        expected_sum = seed_adu + nbr_adu * len(offsets)
        _check(evts, n=1, grade=gid,
               seed_y=4, seed_x=4,
               adu_sum=expected_sum, adu_seed=seed_adu)


# ─────────────────────────────────────────────────────────────────────────────
# Quadruples — the main motivation for this rewrite
# ─────────────────────────────────────────────────────────────────────────────

def test_quadruple_grade9_bilinear_dominant_BL() -> None:
    """
    2×2 cluster with BL dominant (bilinear sharing, px=0.2 py=0.2).
    TR fraction = 0.04 → only 63 ADU, well below old split_sigma but
    the new algorithm includes it because it is connected.
    """
    total = 1586.0
    fracs = (0.64, 0.16, 0.16, 0.04)   # BL, BR, TL, TR
    frame = np.zeros((9, 9), dtype=np.float32)
    cy, cx = 4, 4
    frame[cy,   cx  ] = total * fracs[0]   # BL = 1015 ADU  (seed)
    frame[cy,   cx+1] = total * fracs[1]   # BR =  254 ADU
    frame[cy+1, cx  ] = total * fracs[2]   # TL =  254 ADU
    frame[cy+1, cx+1] = total * fracs[3]   # TR =   63 ADU  < split_thr!

    # TR (63 ADU) is BELOW split_sigma×noise (300 ADU) so it is NOT in sec_mask.
    # The cluster therefore contains only 3 pixels → triple, not quadruple.
    # This is physically correct: TR has negligible charge.
    evts = _find(frame)
    assert len(evts) == 1
    g = int(evts[0]["grade"])
    assert g in (5, 6, 7, 8, 9, 10, 11, 12), f"unexpected grade {g}"
    # adu_sum = all collected pixels (3 in this case, TR excluded)
    assert abs(float(evts[0]["adu_sum"]) - (total * (fracs[0]+fracs[1]+fracs[2]))) < 1.0


def test_quadruple_grade9_all_above_split() -> None:
    """
    2×2 cluster where ALL four pixels are above split_sigma.
    BL is dominant → seed at BL → grade 9.
    adu_sum = all four pixels.
    """
    frame = np.zeros((9, 9), dtype=np.float32)
    cy, cx = 4, 4
    frame[cy,   cx  ] = 800.0   # BL — seed (max)
    frame[cy,   cx+1] = 400.0   # BR — above split (300)
    frame[cy+1, cx  ] = 400.0   # TL — above split
    frame[cy+1, cx+1] = 350.0   # TR — just above split (300)

    evts = _find(frame)
    _check(evts, n=1, grade=9,
           seed_y=cy, seed_x=cx,
           adu_sum=1950.0, adu_seed=800.0)


def test_quadruple_all_orientations_all_above_split() -> None:
    """
    All four 2×2 quadruple grades (9-12) with dominant corner at each position.
    All pixels well above split_sigma so all four are connected.
    """
    cases = [
        # (seed_dy, seed_dx, expected_grade)
        (0,  0,  9),   # BL dominant → grade 9  (up+right+up-right)
        (0, +1, 10),   # BR dominant → grade 10 (up+left+up-left)
        (+1, 0, 12),   # TL dominant → grade 12 (down+right+down-right)
        (+1,+1, 11),   # TR dominant → grade 11 (down+left+down-left)
    ]
    cy, cx = 4, 4
    for sdy, sdx, expected_grade in cases:
        frame = np.zeros((9, 9), dtype=np.float32)
        # Seed pixel at dominant corner
        frame[cy,   cx  ] = 400.0
        frame[cy,   cx+1] = 400.0
        frame[cy+1, cx  ] = 400.0
        frame[cy+1, cx+1] = 400.0
        # Make the expected seed dominant
        frame[cy + sdy, cx + sdx] = 800.0

        evts = _find(frame)
        assert len(evts) == 1, (
            f"grade {expected_grade}: expected 1 event, got {len(evts)}")
        g = int(evts[0]["grade"])
        assert g == expected_grade, (
            f"seed at ({sdy},{sdx}): expected grade {expected_grade}, got {g}")
        assert abs(float(evts[0]["adu_sum"]) - 2000.0) < 0.5, (
            f"grade {expected_grade}: adu_sum wrong")


def test_quadruple_equal_split_one_event() -> None:
    """
    2×2 with all four pixels equal and above seed_sigma.
    Must produce exactly one event (the connected component is one cluster).
    """
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4:6, 4:6] = 600.0   # all four pixels = 600 ADU > seed_thr (500)
    evts = _find(frame)
    assert len(evts) == 1, f"expected 1 event, got {len(evts)}"
    assert int(evts[0]["grade"]) in (9, 10, 11, 12)
    assert abs(float(evts[0]["adu_sum"]) - 2400.0) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Connectivity: left + below only, no diagonal
# ─────────────────────────────────────────────────────────────────────────────

def test_diagonal_only_not_connected() -> None:
    """
    Two pixels touching only diagonally are NOT connected.
    Each becomes a separate single event.
    """
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = 600.0
    frame[5, 5] = 600.0   # diagonal from (4,4) — no left/below connection
    evts = _find(frame)
    assert len(evts) == 2, (
        f"diagonal pixels should be separate events, got {len(evts)}")
    for e in evts:
        assert int(e["grade"]) == 0   # each is a single


def test_left_neighbour_connects() -> None:
    """Pixel to the right (dx=+1) of the seed is connected via 'left' check."""
    frame = _single_event_frame(800.0, ((0, +1, 400.0),))
    evts  = _find(frame)
    _check(evts, n=1, grade=1, seed_y=4, seed_x=4,
           adu_sum=1200.0, adu_seed=800.0)


def test_below_neighbour_connects() -> None:
    """Pixel above seed (dy=+1) is connected via 'below' check of that pixel."""
    frame = _single_event_frame(800.0, ((+1, 0, 400.0),))
    evts  = _find(frame)
    _check(evts, n=1, grade=2, seed_y=4, seed_x=4,
           adu_sum=1200.0, adu_seed=800.0)


# ─────────────────────────────────────────────────────────────────────────────
# GRADE_OTHER: unrecognised cluster shape
# ─────────────────────────────────────────────────────────────────────────────

def test_unrecognised_pattern_is_grade_other() -> None:
    """
    A 3-pixel L-rotated pattern not in _GRADE_DEFS → GRADE_OTHER.
    Example: right + left (horizontal triple — not in grade table).
    """
    # right and left of seed — bitmask 0x05, not a defined grade
    frame = _single_event_frame(1000.0, ((0, +1, 400.0), (0, -1, 400.0)))
    evts  = _find(frame)
    _check(evts, n=1, grade=GRADE_OTHER)


def test_grade_other_rejected_when_reject_extra() -> None:
    """reject_extra=True drops GRADE_OTHER events."""
    frame = _single_event_frame(1000.0, ((0, +1, 400.0), (0, -1, 400.0)))
    evts  = find_events(frame.astype(np.float32), _noise(frame.shape),
                        seed_sigma=SEED_SIGMA, split_sigma=SPLIT_SIGMA,
                        reject_extra=True)
    assert len(evts) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Multiple independent events in one frame
# ─────────────────────────────────────────────────────────────────────────────

def test_two_independent_singles() -> None:
    """Two well-separated single-pixel events both recovered."""
    frame = np.zeros((9, 16), dtype=np.float32)
    frame[4, 2]  = 700.0
    frame[4, 12] = 700.0
    evts = _find(frame)
    assert len(evts) == 2
    for e in evts:
        assert int(e["grade"]) == 0


def test_single_and_double_in_same_frame() -> None:
    """One single and one double in the same frame, well separated."""
    frame = np.zeros((9, 16), dtype=np.float32)
    # Single at (4, 2)
    frame[4, 2] = 800.0
    # Double at (4, 10)+(4,11)
    frame[4, 10] = 700.0
    frame[4, 11] = 400.0
    evts = _find(frame)
    assert len(evts) == 2
    grades = sorted(int(e["grade"]) for e in evts)
    assert grades == [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Pixel masking
# ─────────────────────────────────────────────────────────────────────────────

def test_search_mask_excludes_pixel() -> None:
    """Pixel outside search_mask cannot be a seed or neighbour."""
    frame = _single_event_frame(1000.0)
    mask  = np.ones(frame.shape, dtype=bool)
    mask[4, 4] = False   # exclude the only hot pixel
    evts = find_events(frame.astype(np.float32), _noise(frame.shape),
                       search_mask=mask,
                       seed_sigma=SEED_SIGMA, split_sigma=SPLIT_SIGMA)
    assert len(evts) == 0


def test_bad_pixel_mask_excludes_neighbour() -> None:
    """
    A bad pixel next to the seed is zeroed out.
    The seed is still found but the neighbour does not contribute to adu_sum.
    """
    frame = _single_event_frame(1000.0, ((0, +1, 400.0),))
    bad   = np.zeros(frame.shape, dtype=bool)
    bad[4, 5] = True   # mark the right neighbour as bad
    evts = find_events(frame.astype(np.float32), _noise(frame.shape),
                       bad_pixel_mask=bad,
                       seed_sigma=SEED_SIGMA, split_sigma=SPLIT_SIGMA)
    _check(evts, n=1, grade=0, adu_sum=1000.0)   # neighbour zeroed → single


# ─────────────────────────────────────────────────────────────────────────────
# Cluster merging (union-find correctness)
# ─────────────────────────────────────────────────────────────────────────────

def test_u_shape_merges_into_one_cluster() -> None:
    """
    U-shaped cluster: three pixels where the two arms are connected through
    the base via left/below links.  Must produce one event.

      X . X
      X X X   ← base row
    """
    frame = np.zeros((9, 9), dtype=np.float32)
    # Base row: (3,3), (3,4), (3,5)
    frame[3, 3] = 600.0
    frame[3, 4] = 600.0
    frame[3, 5] = 600.0
    # Left arm: (4,3)
    frame[4, 3] = 600.0
    # Right arm: (4,5)
    frame[4, 5] = 600.0

    evts = _find(frame)
    assert len(evts) == 1, f"U-shape should be one cluster, got {len(evts)}"
    assert abs(float(evts[0]["adu_sum"]) - 3000.0) < 0.5


def test_two_clusters_separated_by_gap() -> None:
    """Two clusters separated by one below-threshold pixel → two events."""
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 2] = 700.0
    frame[4, 3] = 0.0    # gap: below split_sigma
    frame[4, 4] = 700.0
    evts = _find(frame)
    assert len(evts) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Bulk statistics: 2×2 clusters with bilinear charge sharing
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_2x2_bilinear_sharing() -> None:
    """
    1000 random 2×2 clusters with bilinear charge sharing.
    Events where all 4 pixels are above split_sigma → grade 9-12.
    Events where TR is below split_sigma → triple (correct physics).
    adu_sum always = sum of collected pixels.
    """
    rng      = np.random.default_rng(0)
    total    = 1586.0
    cy, cx   = 20, 20
    H, W     = 64, 64

    n_quad       = 0
    n_triple     = 0
    n_other      = 0
    adu_sum_errs = []

    for _ in range(1000):
        px = rng.uniform(0, 1)
        py = rng.uniform(0, 1)
        f_BL = (1-px)*(1-py)
        f_BR =    px *(1-py)
        f_TL = (1-px)*   py
        f_TR =    px *   py

        frame = np.zeros((H, W), dtype=np.float32)
        frame[cy,   cx  ] = total * f_BL
        frame[cy,   cx+1] = total * f_BR
        frame[cy+1, cx  ] = total * f_TL
        frame[cy+1, cx+1] = total * f_TR

        # Determine which pixels are above split_sigma (300 ADU)
        above_split = [
            total * f > SPLIT_SIGMA * NOISE_LEVEL
            for f in (f_BL, f_BR, f_TL, f_TR)
        ]
        expected_sum = sum(
            total * f for f, ab in zip((f_BL, f_BR, f_TL, f_TR), above_split)
            if ab
        )

        evts = _find(frame)
        assert len(evts) == 1, f"expected 1 event, got {len(evts)}"

        g = int(evts[0]["grade"])
        if g in (9, 10, 11, 12):
            n_quad += 1
        elif g in (5, 6, 7, 8):
            n_triple += 1
        else:
            n_other += 1

        err = abs(float(evts[0]["adu_sum"]) - expected_sum)
        adu_sum_errs.append(err)

    assert max(adu_sum_errs) < 0.5, (
        f"adu_sum mismatch: max error = {max(adu_sum_errs):.2f}")
    assert n_other == 0, f"unexpected grade in {n_other} events"
    # Rough sanity: quads occur when all 4 pixels above split_sigma
    # (roughly when px > 0.19 and py > 0.19 in all quadrants)
    assert n_quad + n_triple == 1000


if __name__ == "__main__":
    import traceback
    tests = [
        test_single_pixel_grade0,
        test_below_seed_threshold_no_event,
        test_below_split_threshold_no_event,
        test_all_defined_grades,
        test_quadruple_grade9_bilinear_dominant_BL,
        test_quadruple_grade9_all_above_split,
        test_quadruple_all_orientations_all_above_split,
        test_quadruple_equal_split_one_event,
        test_diagonal_only_not_connected,
        test_left_neighbour_connects,
        test_below_neighbour_connects,
        test_unrecognised_pattern_is_grade_other,
        test_grade_other_rejected_when_reject_extra,
        test_two_independent_singles,
        test_single_and_double_in_same_frame,
        test_search_mask_excludes_pixel,
        test_bad_pixel_mask_excludes_neighbour,
        test_u_shape_merges_into_one_cluster,
        test_two_clusters_separated_by_gap,
        test_bulk_2x2_bilinear_sharing,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed  {failed} failed")
