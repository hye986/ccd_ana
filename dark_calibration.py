#!/usr/bin/env python3
"""Standalone pnCCD dark-frame calibration.

This file is intentionally independent of the pnccd_ana package.  It reads
the native 512x512 RAW format, calculates a per-pixel median offset, noise,
and bad-pixel mask, then saves NumPy arrays and PNG plots.

Example:
    python dark_calibration.py dark.raw --maxframe 1000 -o calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


HEADER = b"16BU0000"
HEADER_BYTES = 8
HEIGHT = 512
WIDTH = 512
ROW_BYTES = 6 + WIDTH * 2
FRAME_BYTES = HEIGHT * ROW_BYTES


def read_raw(path: Path, maxframe: int | None, skip: int) -> np.ndarray:
    """Read complete frames from a native pnCCD RAW file as uint16."""
    if skip < 0:
        raise ValueError("--skip must be non-negative")
    size = path.stat().st_size
    with path.open("rb") as stream:
        if stream.read(len(HEADER)) != HEADER:
            raise ValueError(f"{path} does not start with the RAW header {HEADER!r}")

    available = max(0, (size - HEADER_BYTES) // FRAME_BYTES)
    if available == 0:
        raise ValueError("RAW file contains no complete 512x512 frames")
    if skip >= available:
        raise ValueError(f"--skip={skip} leaves no frames (file has {available})")
    count = available - skip if maxframe is None else min(maxframe, available - skip)
    if count <= 0:
        raise ValueError("--maxframe must be positive")

    # Each row record is marker (uint16), ADC counter (uint32), then 512 pixels.
    record_dtype = np.dtype([
        ("marker", "<u2"), ("counter", "<u4"), ("pixels", "<u2", (WIDTH,))
    ])
    records = np.memmap(path, dtype=record_dtype, mode="r", offset=HEADER_BYTES,
                        shape=((size - HEADER_BYTES) // ROW_BYTES,))
    start = skip * HEIGHT
    selected = records[start:start + count * HEIGHT]["pixels"]
    return np.asarray(selected, dtype=np.uint16).reshape(count, HEIGHT, WIDTH)


def median_offset(frames: np.ndarray) -> np.ndarray:
    """Calculate a per-pixel median, ignoring ADC under/overflow sentinels."""
    values = frames.astype(np.float32)
    values[(frames == 0) | (frames == 65535)] = np.nan
    return np.nanmedian(values, axis=0).astype(np.float32)


def pixel_noise(frames: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Calculate sample RMS noise around each pixel's median offset."""
    residual = frames.astype(np.float32) - offset[None, :, :]
    residual[(frames == 0) | (frames == 65535)] = np.nan
    if frames.shape[0] < 2:
        raise ValueError("At least two frames are required to calculate noise")
    return np.nanstd(residual, axis=0, ddof=1).astype(np.float32)


def bad_pixel_mask(noise: np.ndarray, hot_multiple: float,
                   cold_fraction: float) -> np.ndarray:
    """Flag non-finite, zero-noise, hot, and cold pixels."""
    finite_positive = np.isfinite(noise) & (noise > 0)
    if not finite_positive.any():
        raise ValueError("No finite, positive noise values available")
    reference = float(np.median(noise[finite_positive]))
    bad = ~finite_positive
    bad |= noise > hot_multiple * reference
    bad |= finite_positive & (noise < cold_fraction * reference)
    return bad.astype(bool)


def save_plots(output: Path, offset: np.ndarray, noise: np.ndarray,
               bad: np.ndarray) -> None:
    """Save compact offset, noise, and bad-pixel diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def map_plot(data: np.ndarray, name: str, cmap: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal")
        ax.set_title(name)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / f"{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    map_plot(offset, "Offset", "viridis")
    map_plot(noise, "Noise", "inferno")
    map_plot(bad.astype(np.uint8), "Bad Pixel Mask", "gray_r")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_file", type=Path, help="512x512 dark-frame RAW file")
    parser.add_argument("--maxframe", "--max-frames", type=int,
                        help="maximum number of frames to use; default: all")
    parser.add_argument("--skip", type=int, default=0,
                        help="number of initial frames to skip")
    parser.add_argument("-o", "--output", type=Path, default=Path("calibration"),
                        help="output directory (default: calibration)")
    parser.add_argument("--hot-multiple", type=float, default=5.0,
                        help="hot-pixel threshold in units of median noise")
    parser.add_argument("--cold-fraction", type=float, default=0.1,
                        help="cold-pixel threshold as fraction of median noise")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.maxframe is not None and args.maxframe <= 0:
        parser.error("--maxframe must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = read_raw(args.raw_file, args.maxframe, args.skip)
    offset = median_offset(frames)
    noise = pixel_noise(frames, offset)
    bad = bad_pixel_mask(noise, args.hot_multiple, args.cold_fraction)

    np.save(args.output / "offset_median.npy", offset)
    np.save(args.output / "noise.npy", noise)
    np.save(args.output / "bad_pixel_mask.npy", bad)
    if not args.no_plots:
        save_plots(args.output, offset, noise, bad)

    print(f"Frames used: {frames.shape[0]}")
    print(f"Offset: {args.output / 'offset_median.npy'}")
    print(f"Noise: {args.output / 'noise.npy'}")
    print(f"Bad pixels: {args.output / 'bad_pixel_mask.npy'} ({bad.sum()} pixels)")


if __name__ == "__main__":
    main()
