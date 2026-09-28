"""plotting sub-package – visualization utilities (Stage 1: offset calibration)."""

# ── Shared helpers ────────────────────────────────────────────────────────────
from .common import _cb, _stats_box, _adjust_bin_range, save_figure

# ── Stage 1: offset calibration ───────────────────────────────────────────────
from .offset_plots import plot_offsets, plot_noise, plot_cm_map, plot_bad_pixels
