#!/usr/bin/env python3
"""Interactive RAW viewer with offset, bad-pixel, and common-mode correction.

This script is standalone: it imports only NumPy and Matplotlib.  It expects
the files written by ``dark_calibration.py``::

    python view_calibrated_raw.py dark.raw \
        --offset calibration/offset_median.npy \
        --mask calibration/bad_pixel_mask.npy \
        --cm-mode asic

``--cm-mode row`` subtracts one median from every detector row.  ``asic``
subtracts one median from each 64-column section of every row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


RAW_HEADER = b"16BU0000"
HEADER_BYTES = 8
HEIGHT = 512
WIDTH = 512
ROW_RECORD_BYTES = 6 + WIDTH * 2
FRAME_RECORDS = HEIGHT
FRAME_BYTES = FRAME_RECORDS * ROW_RECORD_BYTES
ROW_MARKER = 0xFFFE
FRAME_MARKER = 0xFFFF


class RawFile:
    """Memory-mapped reader for the native 512x512 HYB RAW format."""

    dtype = np.dtype([
        ("marker", "<u2"),
        ("adc", "<u4"),
        ("pixels", "<u2", (WIDTH,)),
    ])

    def __init__(self, path: Path):
        self.path = Path(path)
        size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            if stream.read(len(RAW_HEADER)) != RAW_HEADER:
                raise ValueError(f"Bad RAW header in {self.path}")
        complete_bytes = size - HEADER_BYTES
        self.n = complete_bytes // FRAME_BYTES
        if self.n < 1:
            raise ValueError(f"{self.path} has no complete RAW frames")
        self._records = np.memmap(
            self.path, dtype=self.dtype, mode="r", offset=HEADER_BYTES,
            shape=(self.n * FRAME_RECORDS,),
        )

    def read(self, index: int) -> np.ndarray:
        if not 0 <= index < self.n:
            raise IndexError(f"frame {index} outside 0..{self.n - 1}")
        start = index * FRAME_RECORDS
        # RAW storage is line-major; transpose to [Y, X] for display.
        return self._records[start:start + FRAME_RECORDS]["pixels"].T.copy()

    def marker_status(self, index: int) -> tuple[bool, int]:
        start = index * FRAME_RECORDS
        markers = self._records[start:start + FRAME_RECORDS]["marker"]
        expected = np.full(FRAME_RECORDS, ROW_MARKER, dtype=np.uint16)
        expected[0] = FRAME_MARKER
        bad = int(np.count_nonzero(markers != expected))
        return bad == 0, bad

    def close(self) -> None:
        del self._records


def load_calibration(offset_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate offset and bad-pixel arrays."""
    offset = np.asarray(np.load(offset_path), dtype=np.float32)
    bad = np.asarray(np.load(mask_path), dtype=bool)
    expected = (HEIGHT, WIDTH)
    if offset.shape != expected:
        raise ValueError(f"offset shape {offset.shape}, expected {expected}")
    if bad.shape != expected:
        raise ValueError(f"mask shape {bad.shape}, expected {expected}")
    return offset, bad


def correct_frame(raw_frame: np.ndarray, offset: np.ndarray,
                  bad_mask: np.ndarray, mode: str, asic_width: int = 64
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Subtract offset and common mode, returning corrected image and CM values.

    Bad pixels and ADC sentinel values are excluded from CM medians and are
    represented as NaN in the displayed corrected image.  The returned CM
    array has shape ``(512,)`` in row mode and ``(512, 8)`` in ASIC mode.
    """
    valid = (~bad_mask & (raw_frame != 0) & (raw_frame != 65535)
             & np.isfinite(offset))
    residual = raw_frame.astype(np.float32) - offset
    residual[~valid] = np.nan

    if mode == "row":
        with np.errstate(all="ignore"):
            cm = np.nanmedian(residual, axis=1)
        corrected = residual - cm[:, None]
    else:
        if WIDTH % asic_width:
            raise ValueError("ASIC width must divide the frame width")
        n_asic = WIDTH // asic_width
        cm = np.full((HEIGHT, n_asic), np.nan, dtype=np.float32)
        corrected = np.full_like(residual, np.nan, dtype=np.float32)
        for i in range(n_asic):
            lo, hi = i * asic_width, (i + 1) * asic_width
            with np.errstate(all="ignore"):
                cm[:, i] = np.nanmedian(residual[:, lo:hi], axis=1)
            corrected[:, lo:hi] = residual[:, lo:hi] - cm[:, i, None]

    # A row/ASIC with no valid pixels has NaN CM; keep all its pixels masked.
    corrected[~valid] = np.nan
    return corrected.astype(np.float32), cm.astype(np.float32)


class Viewer:
    def __init__(self, reader: RawFile, offset: np.ndarray, bad: np.ndarray,
                 mode: str, start: int, vmin: float | None,
                 vmax: float | None, cmap: str):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.reader, self.offset, self.bad = reader, offset, bad
        self.mode, self.index = mode, max(0, min(start, reader.n - 1))
        self.vmin, self.vmax, self.cmap = vmin, vmax, cmap
        self.autoscale = vmin is None and vmax is None
        self.fig = plt.figure(figsize=(17, 8))
        self.fig.patch.set_facecolor("#1a1a2e")
        self.ax_image = self.fig.add_axes((0.045, 0.18, 0.43, 0.73))
        self.ax_pixel_hist = self.fig.add_axes((0.52, 0.54, 0.20, 0.37))
        self.ax_cm_hist = self.fig.add_axes((0.52, 0.18, 0.20, 0.27))
        self.ax_info = self.fig.add_axes((0.76, 0.18, 0.21, 0.73))
        self.ax_info.axis("off")
        for ax in (self.ax_image, self.ax_pixel_hist, self.ax_cm_hist):
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#aaaacc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
        self.slider_ax = self.fig.add_axes((0.045, 0.075, 0.675, 0.035))
        self.slider_ax.set_facecolor("#2a2a4e")
        self.slider = Slider(self.slider_ax, "Frame", 0, reader.n - 1,
                             valinit=self.index, valstep=1)
        self.slider.on_changed(self._on_slider)
        self.image = None
        self._refresh()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _limits(self, image: np.ndarray) -> tuple[float, float]:
        values = image[np.isfinite(image)]
        if values.size == 0:
            return 0.0, 1.0
        if self.autoscale:
            lo, hi = np.percentile(values, (1, 99))
            if lo == hi:
                hi = lo + 1.0
            return float(lo), float(hi)
        lo = self.vmin if self.vmin is not None else float(values.min())
        hi = self.vmax if self.vmax is not None else float(values.max())
        return float(lo), float(hi if hi > lo else lo + 1.0)

    def _refresh(self) -> None:
        frame = self.reader.read(self.index)
        corrected, cm = correct_frame(frame, self.offset, self.bad, self.mode)
        lo, hi = self._limits(corrected)
        if self.image is None:
            self.image = self.ax_image.imshow(corrected, origin="lower",
                                              interpolation="nearest",
                                              cmap=self.cmap, vmin=lo, vmax=hi)
            self.fig.colorbar(self.image, ax=self.ax_image, label="ADU")
        else:
            self.image.set_data(corrected)
            self.image.set_clim(lo, hi)
        self.ax_image.set_title(f"Frame {self.index + 1}/{self.reader.n}   "
                                f"offset + {self.mode} CM",
                                color="#eeeeff", fontsize=11)
        self.ax_image.set_xlabel("Column", color="#aaaacc")
        self.ax_image.set_ylabel("Row", color="#aaaacc")

        self.ax_pixel_hist.clear()
        pixels = corrected[np.isfinite(corrected)]
        if pixels.size:
            self.ax_pixel_hist.hist(pixels, bins=100, color="steelblue")
        self.ax_pixel_hist.set_title("Corrected pixel content", color="#aaaacc")
        self.ax_pixel_hist.set_xlabel("ADU", color="#aaaacc")
        self.ax_pixel_hist.set_ylabel("Pixels", color="#aaaacc")

        self.ax_cm_hist.clear()
        cm_values = cm[np.isfinite(cm)]
        if cm_values.size:
            self.ax_cm_hist.hist(cm_values, bins=100, color="darkorange")
        self.ax_cm_hist.set_title(f"Common-mode values ({self.mode})", color="#aaaacc")
        self.ax_cm_hist.set_xlabel("CM (ADU)", color="#aaaacc")
        self.ax_cm_hist.set_ylabel("Rows" if self.mode == "row" else "ASIC rows",
                                   color="#aaaacc")
        complete, bad_markers = self.reader.marker_status(self.index)
        marker_text = "markers OK" if complete else f"bad markers: {bad_markers}"
        self.fig.suptitle(f"{self.reader.path.name} — {marker_text}",
                          color="#eeeeff", fontsize=12)
        self._update_info(frame, corrected, cm, lo, hi, marker_text)
        self.fig.canvas.draw_idle()

    def _update_info(self, raw_frame: np.ndarray, corrected: np.ndarray,
                     cm: np.ndarray, lo: float, hi: float,
                     marker_text: str) -> None:
        """Refresh the dark-theme frame information panel."""
        self.ax_info.clear()
        self.ax_info.axis("off")
        valid = corrected[np.isfinite(corrected)]
        cm_valid = cm[np.isfinite(cm)]
        lines = [
            ("── Frame ──", False),
            (f"file    : {self.reader.path.name}", False),
            (f"index   : {self.index}", False),
            (f"display : {self.index + 1}/{self.reader.n}", False),
            (f"size    : {WIDTH} × {HEIGHT}", False),
            (f"status  : {marker_text}", marker_text != "markers OK"),
            ("", False),
            ("── Calibration ──", False),
            (f"CM mode : {self.mode}", False),
            (f"bad px  : {int(self.bad.sum())}", False),
            (f"masked  : {raw_frame.size - valid.size}", False),
            (f"cmap    : {self.cmap}", False),
            (f"scale   : {lo:.1f} .. {hi:.1f}", False),
            ("", False),
            ("── Corrected pixels ──", False),
            (f"min     : {float(valid.min()):.2f}" if valid.size else "min     : n/a", False),
            (f"max     : {float(valid.max()):.2f}" if valid.size else "max     : n/a", False),
            (f"mean    : {float(valid.mean()):.2f}" if valid.size else "mean    : n/a", False),
            (f"std     : {float(valid.std()):.2f}" if valid.size else "std     : n/a", False),
            ("", False),
            ("── Common mode ──", False),
            (f"entries  : {cm_valid.size}", False),
            (f"mean     : {float(cm_valid.mean()):.2f}" if cm_valid.size else "mean     : n/a", False),
            (f"std      : {float(cm_valid.std()):.2f}" if cm_valid.size else "std      : n/a", False),
            ("", False),
            ("← → / n,p: frame   A: auto-scale", False),
            ("S: save PNG   Q/Esc: quit", False),
        ]
        height = 1.0 / (len(lines) + 1)
        for i, (text, warning) in enumerate(lines):
            self.ax_info.text(0.04, 1.0 - (i + 1) * height, text,
                              transform=self.ax_info.transAxes,
                              color="#ff5555" if warning else "#ccccee",
                              fontsize=8, va="top", ha="left",
                              fontfamily="monospace")

    def _on_slider(self, value: float) -> None:
        self.index = int(round(value))
        self._refresh()

    def _on_key(self, event) -> None:
        if event.key in ("right", "n", " "):
            self.index = min(self.reader.n - 1, self.index + 1)
        elif event.key in ("left", "p"):
            self.index = max(0, self.index - 1)
        elif event.key == "home":
            self.index = 0
        elif event.key == "end":
            self.index = self.reader.n - 1
        elif event.key == "a":
            self.autoscale = not self.autoscale
        elif event.key == "s":
            self.fig.savefig(f"frame_{self.index:06d}_calibrated.png", dpi=150)
            print(f"Saved frame_{self.index:06d}_calibrated.png")
            return
        elif event.key in ("q", "escape"):
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            return
        else:
            return
        self.slider.eventson = False
        self.slider.set_val(self.index)
        self.slider.eventson = True
        self._refresh()

    def show(self) -> None:
        import matplotlib.pyplot as plt
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_file", type=Path)
    parser.add_argument("--offset", type=Path, required=True,
                        help="offset_median.npy from dark_calibration.py")
    parser.add_argument("--mask", type=Path, required=True,
                        help="bad_pixel_mask.npy from dark_calibration.py")
    parser.add_argument("--cm-mode", choices=("row", "asic"), default="asic",
                        help="common-mode median geometry (default: asic)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    reader = RawFile(args.raw_file)
    offset, bad = load_calibration(args.offset, args.mask)
    print(f"Opened {reader.path}: {reader.n} complete frames")
    print(f"Bad pixels masked: {int(bad.sum())}")
    print(f"Common-mode mode: {args.cm_mode}")
    try:
        Viewer(reader, offset, bad, args.cm_mode, args.start,
               args.vmin, args.vmax, args.cmap).show()
    finally:
        reader.close()


if __name__ == "__main__":
    main()
