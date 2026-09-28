from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pnccd_ana.physics.noise import build_bad_pixel_mask


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


if __name__ == "__main__":
    test_build_bad_pixel_mask_flags_hot_cold_stuck_nonfinite()
    test_build_bad_pixel_mask_active_mask_excludes_outside_pixels()
    test_build_bad_pixel_mask_clip_fraction_threshold()
    print("bad-pixel mask tests passed")
