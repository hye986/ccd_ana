"""
pnccd_ana.analysis
==================
High-level public API for pnCCD Fe55 analysis.

Provides simple wrapper functions for running the three analysis stages
and loading results, suitable for use in notebooks and scripts without
needing to know about Config objects or internal module structure.

Pipeline stages
---------------
  Stage 1 -- offset calibration (dark frames):
      run_offset("analysis.yaml")
      -> output/offset.h5
      -> output/offset/offset_results.h5  (plot-backing)
      -> output/offset/*.png

  Stage 2 -- event recognition (source frames):
      run_event_rec("analysis.yaml")
      -> output/events.h5
      -> output/event_rec/event_rec_results.h5  (plot-backing)
      -> output/event_rec/*.png

  Stage 3 -- gain + CTI calibration:
      run_energy_cal("analysis.yaml")
      -> output/energy_cal.h5
      -> output/energy_cal/energy_cal_results.h5  (plot-backing)
      -> output/energy_cal/*.png

Loading results for replotting
-------------------------------
  dark   = load_offset_results("output/offset/offset_results.h5")
  source = load_event_rec_results("output/event_rec/event_rec_results.h5")
  cal    = load_energy_cal_results("output/energy_cal/energy_cal_results.h5")

Loading calibration data directly
---------------------------------
  offset_cal   = load_calibration("output/offset.h5")
  cluster_data = load_events("output/events.h5")
  gain_data    = load_energy_cal("output/energy_cal.h5")
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .cli.offset     import run as _run_offset
from .cli.event_rec  import run as _run_event_rec
from .cli.energy_cal import run as _run_energy_cal
from .io.hdf5 import (
    load_calibration_h5,
    load_events_h5,
    load_energy_cal_h5,
    load_offset_results_h5,
    load_event_rec_results_h5,
    load_energy_cal_results_h5,
)


# ----------------------------------------------------------------------
# Pipeline runners
# ----------------------------------------------------------------------

def run_offset(config_path: str | Path) -> dict:
    """
    Run offset calibration from a config file.

    Parameters
    ----------
    config_path : path to analysis.yaml

    Returns
    -------
    dict with key "global" -> offset/ noise/ bad_pixel_mask results.
    """
    cfg = Config.from_yaml(str(Path(config_path)))
    return _run_offset(cfg)


def run_event_rec(config_path: str | Path) -> dict:
    """
    Run event recognition from a config file.

    Parameters
    ----------
    config_path : path to analysis.yaml

    Returns
    -------
    dict with keys: cluster_data, spectra, bin_edges, hit_count, mean_adu.
    """
    cfg = Config.from_yaml(str(Path(config_path)))
    return _run_event_rec(cfg)


def run_energy_cal(config_path: str | Path) -> dict:
    """
    Run gain + CTI calibration from a config file.

    Parameters
    ----------
    config_path : path to analysis.yaml

    Returns
    -------
    dict with keys: gain_map, cte_map, bad_gain_map, grades, energy_sum,
    cog_row, cog_col, seed_energy, mean_gain.
    """
    cfg = Config.from_yaml(str(Path(config_path)))
    return _run_energy_cal(cfg)


# ----------------------------------------------------------------------
# Calibration data loaders
# ----------------------------------------------------------------------

def load_calibration(
        path:  str | Path,
        prefer: str = "sigclip",
        asics: list[str] | None = None,
) -> dict:
    """
    Load offset calibration (offset.h5) for use in notebooks.

    Parameters
    ----------
    path   : path to offset.h5
    prefer : which offset method to load as the primary "offset" key
             ("sigclip" | "median")
    asics  : ASIC names to load individually, or None for global only

    Returns
    -------
    dict with keys "global" (and optionally per-ASIC), each containing:
      offset        -- float32 array
      offset_type   -- "sigclip" or "median"
      noise         -- float32 array
      n_clipped_map -- float32 array (if sigma-clip was used)
      cm_noise      -- dict or array (per-ASIC or global)
    """
    return load_calibration_h5(path, asics=asics, prefer=prefer)


def load_events(path: str | Path) -> dict:
    """
    Load cluster data (events.h5).

    Returns the CSR cluster_data dict plus spectra, bin_edges, hit_count,
    and mean_adu -- everything needed to redraw the event-rec plots.

    Parameters
    ----------
    path : path to events.h5

    Returns
    -------
    dict with keys: cluster_data, spectra, bin_edges, hit_count, mean_adu,
    meta.
    """
    return load_events_h5(path)


def load_energy_cal(path: str | Path) -> dict:
    """
    Load gain map, CTE map, grades, energies from energy_cal.h5.

    Parameters
    ----------
    path : path to energy_cal.h5

    Returns
    -------
    dict with keys: gain_map, cte_map, bad_gain_map, mean_gain,
    grades, energy_sum, cog_row, cog_col, seed_energy, meta.
    """
    return load_energy_cal_h5(path)


# ----------------------------------------------------------------------
# Results (plot-backing) loaders
# ----------------------------------------------------------------------

def load_offset_results(path: str | Path) -> dict:
    """
    Load offset calibration plot-backing data from offset/offset_results.h5.

    Use to reload data for replotting or refitting without rerunning the
    offset calibration pipeline.

    Parameters
    ----------
    path : path to offset/offset_results.h5

    Returns
    -------
    dict with keys: offsets (method -> {map, hist_edges, hist_counts}),
    noise ({map, hist_edges, hist_counts, cm_noise, n_clipped}),
    bad_pixels ({mask, noise_hist_edges, noise_hist_counts,
                 thresholds, per_asic}), meta.
    """
    return load_offset_results_h5(path)


def load_event_rec_results(path: str | Path) -> dict:
    """
    Load event recognition plot-backing data from
    event_rec/event_rec_results.h5.

    Parameters
    ----------
    path : path to event_rec/event_rec_results.h5

    Returns
    -------
    dict with keys: raw_spectrum, cluster_size_distribution, meta.
    """
    return load_event_rec_results_h5(path)


def load_energy_cal_results(path: str | Path) -> dict:
    """
    Load energy calibration plot-backing data from
    energy_cal/energy_cal_results.h5.

    Parameters
    ----------
    path : path to energy_cal/energy_cal_results.h5

    Returns
    -------
    dict with keys: gain_map, cte_map, bad_gain_map,
    col_peaks ({ppos, sigma, used_fallback}),
    grades, energy_sum,
    filtered ({adu_sum, row, col}),
    resolution_fit (fit result scalars), meta.
    """
    return load_energy_cal_results_h5(path)
