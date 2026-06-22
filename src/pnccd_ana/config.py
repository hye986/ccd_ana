"""
pnccd_ana.config
================
YAML-based configuration system.

Config file format (YAML)
─────────────────────────

  general:
    output_dir: output            # output directory for results
    data_dir: .                  # base input directory
    data_format: raw             # raw | h5 (file format)
    frame_rows: 1024             # number of rows per frame (Y dimension)
    frame_cols: 512              # number of columns per frame (X dimension)
    n_workers: 8                 # parallel workers
    chunk_size: 64               # frames per chunk
    max_frames: null             # null = all frames
    complete_only: true          # skip incomplete last frame
    metadata:                    # arbitrary key-value pairs saved into HDF5
      operator: Alice
      sample: Fe55_source
      run_date: 2026-05-01

  offset:
    dark_run_file: dark_run.raw  # relative to data_dir
    pedestal_method: both        # median | sigclip | both
    sigma_clip_nsigma: 3.0
    compare_pedestal_methods: true
    split_even_odd: true         # separate even/odd column CM medians per ASIC
    save_npy: true
    save_h5: true

  event_rec:
    source_run_file: source_run.raw  # relative to data_dir
    calibration_file: offset.h5      # relative to data_dir
    seed_sigma: 5.0
    split_sigma: 3.0
    split_even_odd: true             # separate even/odd column CM medians per ASIC
    noise_scope: auto
    reject_extra: false
    adu_min: 0.0
    adu_max: 10000.0
    n_bins: 1000
    save_events: events.h5
    prefer_offset: sigclip

Path resolution:
  - Input paths (dark_run_file, source_run_file, calibration_file) are resolved
    relative to data_dir if not absolute
  - Output paths (save_events) are resolved relative to output_dir
  - calibration_file defaults to {output_dir}/offset.h5 if not set
  - save_events defaults to {output_dir}/events.h5 if not set
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "general": {
        "output_dir":       "output",
        "data_dir":         ".",
        "data_format":      "raw",
        "frame_rows":       None,
        "frame_cols":       None,
        "n_workers":        8,
        "chunk_size":       64,
        "max_frames":       None,
        "complete_only":    True,
        "save_frame_plots": True,
        "ASIC_num":         8,
        "ASIC_mask":        [],
        "metadata":         {},
    },
    "offset": {
        "dark_run_file":            None,
        "pedestal_method":          "both",
        "sigma_clip_nsigma":        3.0,
        "compare_pedestal_methods": True,
        "skip_frames":              0,
        "max_frames":               None,
        "save_npy":                 True,
        "save_h5":                  True,
        # ROOT Analysis.OffNoiMapHLL.SplitEvenOdd 1
        # Compute separate CM medians for even- and odd-indexed columns
        # within each ASIC segment.  Corrects alternating-channel correlated
        # noise seen in pnCCDs.  Matches ROOT HCommonModeMedianEvenOdd.
        "split_even_odd":           True,
    },
    "event_rec": {
        "source_run_file":    None,
        "calibration_file":   None,
        "skip_frames":        0,
        "max_frames":         None,
        "seed_sigma":         5.0,
        "split_sigma":        3.0,
        "noise_scope":        "auto",
        "reject_extra":       False,
        "adu_min":            0.0,
        "adu_max":            10000.0,
        "n_bins":             1000,
        "save_events":        None,
        "prefer_offset":      "sigclip",
        "split_even_odd":     True,
        "bad_pixel_mask": {
            "enabled":           True,
            "hot_rms_multiple":  5.0,
            "cold_rms_fraction": 0.1,
            "max_clip_fraction": 0.5,
            "n_dark_frames":     0,
        },
    },
    "energy_cal": {
        "events_file":       None,
        "output_file":       None,
        "target_ev":         5895.0,
        "kalpha_adu":        None,
        "kalpha_adu_window": 0.20,
        "fit_window_frac":   0.15,
        "rough_min_events":  100,
        "cti_row_bin_size":  64,
        "cti_min_events":    50,
        "cti_grade_filter":  None,
        "col_min_events":    30,
        "col_with_bg":       False,
        "col_grade_filter":  [0],
        "save_plots":        True,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    """
    Parsed and validated analysis configuration.

    Access sections as attributes::

        cfg = Config.from_yaml("analysis.yaml")
        print(cfg.general["output_dir"])
        print(cfg.offset["pedestal_method"])
        print(cfg.offset["split_even_odd"])    # True by default

    Sections not in the YAML are populated with defaults.
    Unknown top-level sections are preserved as-is (for future extensions).
    """

    def __init__(self, raw: dict) -> None:
        self._raw = raw
        # Merge each known section with defaults
        for section, defaults in _DEFAULTS.items():
            user = raw.get(section, {}) or {}
            # Deep-merge nested dicts (e.g. bad_pixel_mask sub-dict)
            merged: dict = {}
            for k, v in defaults.items():
                if isinstance(v, dict) and isinstance(user.get(k), dict):
                    merged[k] = {**v, **user[k]}
                else:
                    merged[k] = user.get(k, v)
            # Carry over any extra user keys not in defaults
            for k, v in user.items():
                if k not in merged:
                    merged[k] = v
            setattr(self, section, merged)
        # Preserve any extra sections verbatim
        for section in raw:
            if not hasattr(self, section):
                setattr(self, section, raw[section])

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def output_dir(self) -> Path:
        return Path(self.general["output_dir"])

    @property
    def data_dir(self) -> Path:
        return Path(self.general["data_dir"])

    def resolve_input_path(self, path: str | Path | None) -> Path | None:
        """Resolve an input file path relative to data_dir."""
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return p
        return self.data_dir / p

    def resolve_output_path(self, path: str | Path | None) -> Path | None:
        """Resolve an output file path relative to output_dir."""
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return p
        return self.output_dir / p

    def calibration_path(self) -> Path:
        """Default path for offset calibration file."""
        return self.resolve_output_path("offset.h5")

    def events_path(self) -> Path:
        """Default path for events file."""
        return self.resolve_output_path("events.h5")

    def asics_for(self, section: str) -> list[str] | None:
        """Return resolved ASIC list for *section*."""
        from .io.geometry import resolve_asics
        raw = self.general.get("asics", None)
        if isinstance(raw, str):
            raw = raw.split()
        return resolve_asics(raw)

    def methods_for_offset(self) -> list[str]:
        """Parse pedestal_method into ['median'], ['sigclip'], or both."""
        m = self.offset.get("pedestal_method", "both").lower().strip()
        if m == "both":
            return ["median", "sigclip"]
        if m in ("median", "sigclip"):
            return [m]
        raise ValueError(f"Unknown pedestal_method: {m!r}.  "
                         f"Use 'median', 'sigclip', or 'both'.")

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        if not _HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML config files.\n"
                "  pip install pyyaml")
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls(raw)
        print(f"Config loaded from: {path}")
        return cfg

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Build a Config directly from a nested dict (useful in notebooks)."""
        return cls(d)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_yaml(self, path: str | Path) -> None:
        """Write the merged (defaults + user overrides) config back to YAML."""
        if not _HAS_YAML:
            raise ImportError("PyYAML required:  pip install pyyaml")
        combined = {}
        for section in list(_DEFAULTS) + [s for s in vars(self)
                                           if not s.startswith("_")
                                           and s not in _DEFAULTS]:
            val = getattr(self, section, None)
            if val is not None:
                combined[section] = val
        with open(path, "w") as fh:
            yaml.dump(combined, fh, default_flow_style=False, sort_keys=False)
        print(f"Config written to: {path}")

    def __repr__(self) -> str:
        sections = [s for s in vars(self) if not s.startswith("_")]
        return f"Config(sections={sections})"


# ──────────────────────────────────────────────────────────────────────────────
# Template
# ──────────────────────────────────────────────────────────────────────────────

_TEMPLATE = """\
# pnccd_ana analysis configuration
# Generated by pnccd_ana.config — edit as needed.
# This template is for single hybrid RAW format (512x512, 1024x512, etc.).

general:
  output_dir: output                # output directory for results
  data_dir: .                       # base input directory (paths relative to this)
  data_format: raw                  # raw | h5 (file format)
  frame_rows: 1024                  # number of rows per frame (Y dimension)
  frame_cols: 512                   # number of columns per frame (X dimension)
  n_workers: 8                      # parallel workers
  chunk_size: 64                    # frames per chunk
  max_frames: null                  # null = all frames
  complete_only: true               # skip incomplete last frame
  save_frame_plots: true            # generate diagnostic plots
  ASIC_num: 8                       # number of ASICs (frame_cols / ASIC_num = cols per ASIC)
  ASIC_mask: []                     # masked ASIC indices e.g. [0, 2]
  metadata:
    operator: ""
    sample: ""
    run_date: ""

offset:
  dark_run_file: dark_run.raw       # relative to data_dir
  pedestal_method: both             # median | sigclip | both
  sigma_clip_nsigma: 3.0
  compare_pedestal_methods: true
  skip_frames: 0                    # skip N frames at start (warm-up / shutter transient)
  max_frames: null                  # overrides general.max_frames for this stage
  save_npy: true
  save_h5: true
  split_even_odd: true              # separate CM medians for even/odd columns per ASIC
                                    # matches ROOT Analysis.OffNoiMapHLL.SplitEvenOdd 1
                                    # corrects alternating-channel correlated noise in pnCCDs
                                    # set false for standard single-median CM correction

event_rec:
  source_run_file: source_run.raw   # relative to data_dir
  calibration_file: offset.h5       # relative to data_dir (or output_dir if not found)
  skip_frames: 0
  max_frames: null
  seed_sigma: 5.0                   # primary threshold — ROOT Analysis.Filter.ThresPrm 5 noise
  split_sigma: 3.0                  # secondary threshold — ROOT Analysis.Filter.ThresSec 3 noise
  split_even_odd: true              # separate CM medians for even/odd columns per ASIC
                                    # matches ROOT Analysis.Filter.SplitEvenOdd 1
                                    # MUST match the value used in the offset stage
  noise_scope: auto
  reject_extra: false               # discard grade-13 "other" events from output
  adu_min: 0.0
  adu_max: 10000.0
  n_bins: 1000
  save_events: events.h5            # relative to output_dir (null to skip)
  prefer_offset: sigclip            # sigclip | median
  bad_pixel_mask:
    enabled: true
    hot_rms_multiple: 5.0           # noise > N × median(active noise) → HOT
    cold_rms_fraction: 0.1          # noise < N × median(active noise) → COLD/stuck
    max_clip_fraction: 0.5          # clip rate above this → UNSTABLE
    n_dark_frames: 0                # 0 disables clip-fraction test

energy_cal:
  events_file: null                 # null = {output_dir}/events.h5
  output_file: null                 # null = {output_dir}/energy_cal.h5
  target_ev: 5895.0                 # Mn Kα reference energy [eV]
  kalpha_adu: 15000                 # REQUIRED: Kα peak position in ADU
                                    # read from event_rec spectrum (single-pixel peak)
  kalpha_adu_window: 0.20           # Phase 1 fit window ± fraction of kalpha_adu
  fit_window_frac: 0.15             # Phases 3+4 fit window ± fraction of target_ev [eV]
  rough_min_events: 100
  cti_row_bin_size: 64
  cti_min_events: 50
  cti_grade_filter: null            # null = all grades; or e.g. [0,1,2,3,4]
  col_min_events: 30
  col_with_bg: false
  col_grade_filter: [0]
  save_plots: true
"""


def write_template(path: str | Path = "analysis.yaml") -> None:
    """Write a fully-commented template config to *path*."""
    with open(path, "w") as fh:
        fh.write(_TEMPLATE)
    print(f"Template config written to: {path}")
