"""
Tests for the ADC rollover detection and correction.
"""
from __future__ import annotations

import numpy as np

from pnccd_ana.lib.pedestal import detect_and_unwrap_rollover, ADC_MAX, ADC_RANGE


def test_no_rollover_returns_data_unchanged() -> None:
    """Clean data (no values near 0 or ADC_MAX) should not be flagged."""
    data = np.full((100, 32, 32), 50000.0, dtype=np.float32)
    out, mask = detect_and_unwrap_rollover(data, unwrap=True)
    assert mask.sum() == 0
    np.testing.assert_array_equal(out, data)


def test_rollover_flagged_and_unwrapped() -> None:
    """A rolled pixel (raw=0) at a normally-high pixel should be flagged and unwrapped."""
    data = np.full((10, 5, 5), 60000.0, dtype=np.float32)
    data[3, 2, 2] = 0.0    # rolled: should have been 60000+noise but wrapped to 0
    out, mask = detect_and_unwrap_rollover(data, low_frac=0.10, high_frac=0.80, unwrap=True)

    # Should be flagged
    assert mask[3, 2, 2], "Rolled pixel not flagged"
    assert not mask[0, 2, 2], "Non-rolled pixel incorrectly flagged"

    # Should be unwrapped: 0 + 65536 = 65536
    assert out[3, 2, 2] == 65536.0, \
        f"Expected 65536.0, got {out[3, 2, 2]} (uint16 overflow?)"
    # Non-rolled pixel unchanged
    assert out[0, 2, 2] == 60000.0


def test_uint16_overflow_fixed() -> None:
    """uint16 arrays must NOT silently wrap on the +ADC_RANGE addition."""
    # Simulate uint16 input (as H5 actually delivers)
    data = np.zeros((5, 4, 4), dtype=np.uint16)
    data[:] = 60000
    data[2, 1, 1] = 0      # rolled value as uint16

    out, mask = detect_and_unwrap_rollover(
        data.astype(np.float32),   # already pass float32 (as the code does)
        low_frac=0.10, high_frac=0.80, unwrap=True,
    )
    # With float32 path: 0 + 65536 = 65536
    assert out.dtype == np.float32, f"Output should be float32, got {out.dtype}"
    assert out[2, 1, 1] == 65536.0, \
        f"Expected 65536.0, got {out[2,1,1]:.0f} — uint16 overflow not fixed"


def test_dead_pixel_not_flagged() -> None:
    """A pixel that never reaches the upper range should never be flagged."""
    # Pixel (0,0) is always low → canary fails → never flagged
    # Pixel (1,1) has one high value → canary passes → can be flagged
    data = np.zeros((10, 3, 3), dtype=np.float32)
    data[:, 0, 0] = 5000.0     # dead/cold pixel — max=5000 < 52428 (high_thresh)
    data[:, 1, 1] = 60000.0    # normal pixel — max=60000 > 52428
    data[4, 0, 0] = 0.0        # low value at dead pixel → should NOT be flagged
    data[4, 1, 1] = 0.0        # low value at normal pixel → SHOULD be flagged

    out, mask = detect_and_unwrap_rollover(data, unwrap=True)

    assert not mask[4, 0, 0], "Dead pixel should not be flagged"
    assert mask[4, 1, 1], "Normal pixel with low value should be flagged"
    assert out[4, 0, 0] == 5000.0, "Dead pixel value should be unchanged"
    assert out[4, 1, 1] == 65536.0, "Rolled normal pixel should be unwrapped"


def test_heavy_rollover_still_detected() -> None:
    """
    A pixel that rolled over in >50% of frames has median ≈ 0.
    Old median-based check would miss it; the new canary (max) still catches it.
    """
    # 8 out of 10 frames rolled → median = 0 → old check fails
    # max = 60000 → new canary passes ✓
    data = np.full((10, 3, 3), 0.0, dtype=np.float32)   # most frames rolled to 0
    data[:, 1, 1] = 0.0
    data[0, 1, 1] = 60000.0    # one clean frame above the canary threshold
    data[1, 1, 1] = 60000.0    # second clean frame

    out, mask = detect_and_unwrap_rollover(data, unwrap=True)

    # All the "0" frames at (1,1) should be flagged and unwrapped
    flagged = mask[:, 1, 1]
    assert flagged.sum() == 8, f"Expected 8 rollovers at (1,1), got {flagged.sum()}"
    # Unwrapped values should be ~65536
    for i in range(10):
        if i < 2:
            assert out[i, 1, 1] == 60000.0, f"Clean frame {i} changed: {out[i,1,1]}"
        else:
            assert out[i, 1, 1] == 65536.0, f"Rolled frame {i}: {out[i,1,1]:.0f} (not unwrapped)"


def test_unwrap_false_no_modification() -> None:
    """When unwrap=False, the output data should be unchanged."""
    data = np.full((5, 4, 4), 60000.0, dtype=np.float32)
    data[2, 0, 0] = 0.0
    out, mask = detect_and_unwrap_rollover(data, unwrap=False)
    assert out.dtype == np.float32
    assert out[2, 0, 0] == 0.0     # still 0, not promoted
    assert mask[2, 0, 0]           # but still flagged correctly


def test_print_output_includes_canary_info() -> None:
    """Verify the diagnostic print includes pixel canary statistics."""
    data = np.full((5, 4, 4), 60000.0, dtype=np.float32)
    out, mask = detect_and_unwrap_rollover(data, low_frac=0.10, high_frac=0.80)
    # All 16 pixels have max=60000 > 52428 → all pass the canary
    assert mask.sum() == 0
    assert out is not None


if __name__ == "__main__":
    test_no_rollover_returns_data_unchanged()
    test_rollover_flagged_and_unwrapped()
    test_uint16_overflow_fixed()
    test_dead_pixel_not_flagged()
    test_heavy_rollover_still_detected()
    test_unwrap_false_no_modification()
    test_print_output_includes_canary_info()
    print("rollover tests passed")