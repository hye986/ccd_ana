from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pnccd_ana.lib import pattern_recognition as pr
from pnccd_ana.lib.noise import build_bad_pixel_mask


# ── build_bad_pixel_mask ─────────────────────────────────────────────────────

def test_build_bad_pixel_mask_flags_hot_cold_stuck_nonfinite() -> None:
    noise = np.full((16, 16), 4.0, dtype=np.float32)   # baseline noise = 4 ADU
    noise[2, 2]  = 100.0      # very HOT
    noise[5, 5]  = 0.0        # STUCK / cold (= 0 ADU)
    noise[7, 7]  = 0.05        # COLD (< 0.1 × median)
    noise[9, 9]  = np.nan      # non-finite
    noise[11, 11] = np.inf

    mask = build_bad_pixel_mask(noise,
                                hot_rms_multiple=5.0,
                                cold_rms_fraction=0.1)
    bad_coords = {(int(y), int(x)) for y, x in zip(*np.where(mask))}
    assert (2, 2)   in bad_coords     # hot
    assert (5, 5)   in bad_coords     # stuck (rms = 0)
    assert (7, 7)   in bad_coords     # cold
    assert (9, 9)   in bad_coords     # NaN
    assert (11, 11) in bad_coords     # Inf
    # everything else is good
    assert len(bad_coords) == 5


def test_build_bad_pixel_mask_active_mask_excludes_outside_pixels() -> None:
    noise = np.full((16, 16), 4.0, dtype=np.float32)
    # An "ASIC" region: only the central 8×8 is active
    active = np.zeros_like(noise, dtype=bool)
    active[4:12, 4:12] = True

    mask = build_bad_pixel_mask(noise, active_mask=active,
                                hot_rms_multiple=5.0,
                                cold_rms_fraction=0.1)
    # Everything outside the active region is bad by construction
    assert mask[~active].all()
    # Everything inside is good (uniform noise)
    assert not mask[active].any()


def test_build_bad_pixel_mask_clip_fraction_threshold() -> None:
    noise = np.full((8, 8), 4.0, dtype=np.float32)
    n_clipped = np.zeros_like(noise)
    n_dark = 100
    n_clipped[1, 1] = 60      # 60% clipped → unstable
    n_clipped[2, 2] = 40      # 40% clipped → OK

    mask = build_bad_pixel_mask(noise, n_clipped_map=n_clipped,
                                n_dark_frames=n_dark,
                                hot_rms_multiple=10.0, cold_rms_fraction=0.0,
                                max_clip_fraction=0.5)
    assert mask[1, 1]
    assert not mask[2, 2]


# ── find_events with bad_pixel_mask ──────────────────────────────────────────

NOISE_VAL = 4.0          # ADU
SEED_SIGMA  = 5.0        # → seed_thr = 20
SPLIT_SIGMA = 3.0        # → split_thr = 12


def _noise_map(shape):
    return np.full(shape, NOISE_VAL, dtype=np.float32)


def test_bad_pixel_mask_suppresses_hot_seed() -> None:
    """A pixel flagged bad cannot become a seed even if it crosses seed_thr."""
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = 1000.0    # would normally seed
    noise = _noise_map(frame.shape)
    bad   = np.zeros_like(frame, dtype=bool)
    bad[4, 4] = True

    for use_c in (False, True):
        old = pr._HAVE_C
        pr._HAVE_C = use_c
        try:
            evts = pr.find_events(frame, noise,
                                  seed_sigma=SEED_SIGMA,
                                  split_sigma=SPLIT_SIGMA,
                                  bad_pixel_mask=bad)
        finally:
            pr._HAVE_C = old
        assert len(evts) == 0, f"use_c={use_c}: hot seed leaked through"


def test_bad_pixel_mask_excludes_neighbour_from_grade_and_sum() -> None:
    """A bad pixel adjacent to a real seed must not be counted as a split."""
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = 1000.0
    frame[4, 5] = 500.0     # would normally be a split-right neighbour
    noise = _noise_map(frame.shape)
    bad = np.zeros_like(frame, dtype=bool)
    bad[4, 5] = True        # flag the right neighbour as bad

    for use_c in (False, True):
        old = pr._HAVE_C
        pr._HAVE_C = use_c
        try:
            evts = pr.find_events(frame, noise,
                                  seed_sigma=SEED_SIGMA,
                                  split_sigma=SPLIT_SIGMA,
                                  bad_pixel_mask=bad)
        finally:
            pr._HAVE_C = old
        assert len(evts) == 1
        assert int(evts["grade"][0]) == 0           # grade 0 (single) — neighbour ignored
        assert float(evts["adu_sum"][0]) == 1000.0
        assert float(evts["adu_seed"][0]) == 1000.0


def test_bad_pixel_and_search_mask_combine_correctly() -> None:
    """When BOTH a search_mask AND a bad_pixel_mask are supplied,
    pixels excluded by either are zeroed out."""
    frame = np.zeros((9, 16), dtype=np.float32)
    frame[4, 4] = 1000.0     # inside search_mask, not bad → should seed
    frame[4, 10] = 1000.0    # outside search_mask → suppressed
    frame[4, 6]  = 500.0     # inside search_mask but BAD → suppressed
    noise = _noise_map(frame.shape)

    search = np.zeros_like(frame, dtype=bool)
    search[:, :8] = True      # only cols 0..7 active

    bad = np.zeros_like(frame, dtype=bool)
    bad[4, 6] = True

    for use_c in (False, True):
        old = pr._HAVE_C
        pr._HAVE_C = use_c
        try:
            evts = pr.find_events(frame, noise,
                                  search_mask=search,
                                  bad_pixel_mask=bad,
                                  seed_sigma=SEED_SIGMA,
                                  split_sigma=SPLIT_SIGMA)
        finally:
            pr._HAVE_C = old
        assert len(evts) == 1
        assert int(evts["Y"][0]) == 4 and int(evts["X"][0]) == 4
        assert int(evts["grade"][0]) == 0           # no split; bad neighbour suppressed
        assert float(evts["adu_sum"][0]) == 1000.0


def test_find_events_unchanged_when_bad_pixel_mask_is_none() -> None:
    """Calling without bad_pixel_mask must behave exactly as before."""
    frame = np.zeros((9, 9), dtype=np.float32)
    frame[4, 4] = 1000.0
    frame[4, 5] = 500.0
    noise = _noise_map(frame.shape)

    for use_c in (False, True):
        old = pr._HAVE_C
        pr._HAVE_C = use_c
        try:
            evts_a = pr.find_events(frame, noise,
                                    seed_sigma=SEED_SIGMA,
                                    split_sigma=SPLIT_SIGMA)
            evts_b = pr.find_events(frame, noise,
                                    seed_sigma=SEED_SIGMA,
                                    split_sigma=SPLIT_SIGMA,
                                    bad_pixel_mask=None)
        finally:
            pr._HAVE_C = old
        assert len(evts_a) == len(evts_b) == 1
        assert int(evts_a["grade"][0]) == int(evts_b["grade"][0]) == 1
        assert float(evts_a["adu_sum"][0]) == float(evts_b["adu_sum"][0]) == 1500.0


if __name__ == "__main__":
    test_build_bad_pixel_mask_flags_hot_cold_stuck_nonfinite()
    test_build_bad_pixel_mask_active_mask_excludes_outside_pixels()
    test_build_bad_pixel_mask_clip_fraction_threshold()
    test_bad_pixel_mask_suppresses_hot_seed()
    test_bad_pixel_mask_excludes_neighbour_from_grade_and_sum()
    test_bad_pixel_and_search_mask_combine_correctly()
    test_find_events_unchanged_when_bad_pixel_mask_is_none()
    print("bad-pixel mask tests passed")
