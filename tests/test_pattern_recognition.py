from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pnccd_ana.lib import pattern_recognition as pr


GRADE_OFFSETS = {
    0: (),
    1: ((0, +1),),
    2: ((+1, 0),),
    3: ((0, -1),),
    4: ((-1, 0),),
    5: ((+1, 0), (0, +1)),
    6: ((0, -1), (+1, 0)),
    7: ((-1, 0), (0, -1)),
    8: ((-1, 0), (0, +1)),
    9: ((+1, 0), (0, +1), (+1, +1)),
    10: ((+1, 0), (0, -1), (+1, -1)),
    11: ((-1, 0), (0, -1), (-1, -1)),
    12: ((-1, 0), (0, +1), (-1, +1)),
}


def _events(frame: np.ndarray, *, use_c: bool) -> np.ndarray:
    old = pr._HAVE_C
    pr._HAVE_C = use_c
    try:
        noise = np.full(frame.shape, 100.0, dtype=np.float32)
        return pr.find_events(frame, noise, seed_sigma=5.0, split_sigma=1.0)
    finally:
        pr._HAVE_C = old


def _frame_for_offsets(offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = 1000.0
    for dy, dx in offsets:
        frame[4 + dy, 4 + dx] = 300.0
    return frame


def _check_one_event(events: np.ndarray, grade: int, adu_sum: float) -> None:
    assert len(events) == 1
    assert int(events["Y"][0]) == 4
    assert int(events["X"][0]) == 4
    assert int(events["grade"][0]) == grade
    assert float(events["adu_seed"][0]) == 1000.0
    assert float(events["adu_sum"][0]) == adu_sum


def test_all_defined_grades_match_in_c_and_python() -> None:
    for grade, offsets in GRADE_OFFSETS.items():
        frame = _frame_for_offsets(offsets)
        expected_sum = 1000.0 + 300.0 * len(offsets)
        for use_c in (False, True):
            _check_one_event(_events(frame, use_c=use_c), grade, expected_sum)


def test_outer_5x5_pixel_does_not_change_central_3x3_grade() -> None:
    frame = _frame_for_offsets(((0, +1),))
    frame[6, 6] = 300.0
    for use_c in (False, True):
        _check_one_event(_events(frame, use_c=use_c), 1, 1300.0)


def test_unknown_central_3x3_pattern_is_grade_other() -> None:
    frame = _frame_for_offsets(((0, +1), (0, -1)))
    from pnccd_ana.lib.pattern_recognition import GRADE_OTHER
    for use_c in (False, True):
        _check_one_event(_events(frame, use_c=use_c), GRADE_OTHER, 1000.0)


# ── Tie-breaker regression tests ──────────────────────────────────────────────
# Two adjacent (or otherwise tied) pixels at exactly the same value must
# produce ONE event, not N overlapping ones.  The lexicographically smallest
# (smallest y; on tie, smallest x) pixel in the 5×5 window wins.

def _tied_pair(dy: int, dx: int, val: float = 600.0) -> np.ndarray:
    """Frame with two adjacent equal-value pixels at (4,4) and (4+dy, 4+dx)."""
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = val
    frame[4 + dy, 4 + dx] = val
    return frame


def test_tied_horizontal_pair_yields_one_grade1_event() -> None:
    # (4,4) and (4,5) tied at 600.  Lex-smallest is (4,4) → grade 1 (right).
    frame = _tied_pair(0, +1)
    for use_c in (False, True):
        events = _events(frame, use_c=use_c)
        assert len(events) == 1, f"use_c={use_c}: got {len(events)} events"
        assert int(events["Y"][0]) == 4
        assert int(events["X"][0]) == 4         # lex-smallest wins
        assert int(events["grade"][0]) == 1     # double-right
        assert float(events["adu_sum"][0]) == 1200.0


def test_tied_vertical_pair_yields_one_grade2_event() -> None:
    # (4,4) and (5,4) tied at 600.  Lex-smallest is (4,4) → grade 2 (down/up).
    frame = _tied_pair(+1, 0)
    for use_c in (False, True):
        events = _events(frame, use_c=use_c)
        assert len(events) == 1
        assert int(events["Y"][0]) == 4
        assert int(events["X"][0]) == 4
        assert int(events["grade"][0]) == 2
        assert float(events["adu_sum"][0]) == 1200.0


def test_tied_2x2_hot_patch_yields_one_quadruple() -> None:
    # 2×2 block of equal hot pixels at (4..5, 4..5).  Without the tie-breaker
    # the recogniser produced 4 grade 9/10/11/12 events for a single cluster.
    # With the fix only (4,4) survives and is a grade-9 quadruple.
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4:6, 4:6] = 600.0    # > seed_thr (5σ = 500)
    for use_c in (False, True):
        events = _events(frame, use_c=use_c)
        assert len(events) == 1, f"use_c={use_c}: got {len(events)} events"
        assert int(events["Y"][0]) == 4
        assert int(events["X"][0]) == 4
        assert int(events["grade"][0]) == 9     # centre + right + down + diag
        assert float(events["adu_sum"][0]) == 2400.0


def test_tied_pair_two_pixels_apart_in_5x5_window() -> None:
    # (4,4) and (4,6) both at 600: in each other's 5×5 (dx=±2) but NOT in
    # each other's central 3×3.  Tie-breaker keeps only (4,4); the other
    # candidate at (4,6) is silently dropped.  Conservative — losing one
    # event is better than double-counting overlapping ones.
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = 600.0
    frame[4, 6] = 600.0
    for use_c in (False, True):
        events = _events(frame, use_c=use_c)
        assert len(events) == 1
        assert int(events["Y"][0]) == 4
        assert int(events["X"][0]) == 4
        # (4,6) is at dx=+2, outside (4,4)'s central 3×3, so no split here.
        assert int(events["grade"][0]) == 0
        assert float(events["adu_sum"][0]) == 600.0


def test_independent_equal_events_outside_each_others_5x5() -> None:
    # Two equal pixels at (4,4) and (4,10): dx=6, completely independent
    # (outside each other's 5×5).  Both must survive — the tie-breaker is
    # *local* to the 5×5 window, not global.
    frame = np.zeros((9, 16), dtype=np.float32)
    frame[4, 4]  = 600.0
    frame[4, 10] = 600.0
    noise = np.full(frame.shape, 100.0, dtype=np.float32)
    for use_c in (False, True):
        old = pr._HAVE_C
        pr._HAVE_C = use_c
        try:
            events = pr.find_events(frame, noise,
                                    seed_sigma=5.0, split_sigma=1.0)
        finally:
            pr._HAVE_C = old
        assert len(events) == 2, f"use_c={use_c}: got {len(events)} events"
        ys = sorted(int(e["Y"]) for e in events)
        xs = sorted(int(e["X"]) for e in events)
        assert ys == [4, 4]
        assert xs == [4, 10]


if __name__ == "__main__":
    test_all_defined_grades_match_in_c_and_python()
    test_outer_5x5_pixel_does_not_change_central_3x3_grade()
    test_unknown_central_3x3_pattern_is_grade_13()
    test_tied_horizontal_pair_yields_one_grade1_event()
    test_tied_vertical_pair_yields_one_grade2_event()
    test_tied_2x2_hot_patch_yields_one_quadruple()
    test_tied_pair_two_pixels_apart_in_5x5_window()
    test_independent_equal_events_outside_each_others_5x5()
    print("pattern recognition tests passed")
