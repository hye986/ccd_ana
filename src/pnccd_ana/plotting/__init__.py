"""plotting sub-package – visualization utilities."""

# ── Shared helpers ────────────────────────────────────────────────────────────
from .common import _cb, _stats_box, _adjust_bin_range, save_figure

# ── Stage 1: offset calibration ───────────────────────────────────────────────
from .offset_plots import plot_offsets, plot_noise, plot_bad_pixels

# ── Stage 2: event recognition ────────────────────────────────────────────────
from .event_plots import (plot_hitmap,
                           plot_spectrum,
                           plot_raw_spectrum,
                           plot_cm_map,
                           plot_cluster_size_distribution)

# ── Stage 3: energy calibration ───────────────────────────────────────────────
from .gain_plots    import (plot_gain_map,
                             plot_gain_histogram,
                             plot_column_peaks,
                             plot_grade_spectrum,
                             plot_gain_vs_row)
from .cti_plots     import (plot_cte_map,
                             plot_cti_summary,
                             plot_signal_vs_row)
from .spectrum_plots import plot_final_spectrum
