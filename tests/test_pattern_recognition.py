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


def test_unknown_central_3x3_pattern_is_grade_13() -> None:
    frame = _frame_for_offsets(((0, +1), (0, -1)))
    for use_c in (False, True):
        _check_one_event(_events(frame, use_c=use_c), 13, 1000.0)


if __name__ == "__main__":
    test_all_defined_grades_match_in_c_and_python()
    test_outer_5x5_pixel_does_not_change_central_3x3_grade()
    test_unknown_central_3x3_pattern_is_grade_13()
    print("pattern recognition tests passed")
