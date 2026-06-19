"""
pnccd_ana.config
================
YAML-based configuration system.

Config file format (YAML)
─────────────────────────

  general:
    output_dir: output            # output directory for results
    data_dir: .                  # base input directory
    # Shared settings (used by both dark_frames and source_spectrum):
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

  dark_frames:
    dark_run_file: dark_run.raw  # relative to data_dir
    pedestal_method: both        # median | sigclip | both
    sigma_clip_nsigma: 3.0
    compare_pedestal_methods: true
    save_npy: true
    save_h5: true

  source_spectrum:
    source_run_file: source_run.raw  # relative to data_dir
    calibration_file: dark_calibration.h5  # relative to data_dir
    seed_sigma: 5.0
    split_sigma: 3.0
    noise_scope: auto
    reject_extra: false
    adu_min: 0.0
    adu_max: 10000.0
    n_bins: 1000
    save_events: true
    save_events_to_file: events.h5  # relative to output_dir
    prefer_offset: sigclip

Path resolution:
  - Input paths (dark_run_file, source_run_file, calibration_file) are resolved
    relative to data_dir if not absolute
  - Output paths (save_events_to_file) are resolved relative to output_dir
  - calibration_file defaults to {output_dir}/dark_calibration.h5 if not set
  - save_events_to_file defaults to {output_dir}/events.h5 if not set

Sections not present in the YAML are simply skipped at runtime.
Each CLI script reads only the section(s) it needs, so you can reuse a
config file across multiple analysis stages.
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
# Defaults  (merged with user YAML, so only overrides are needed)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "general": {
        "output_dir":       "output",      # base output directory for results
        "data_dir":         ".",           # base input directory for data files
        # Shared settings (used by both dark_frames and source_spectrum):
        "data_format":      "raw",         # raw | h5 (file format)
        "frame_rows":       None,          # number of rows per frame (auto-detected if None)
        "frame_cols":      None,          # number of columns per frame (auto-detected if None)
        "n_workers":        8,             # parallel workers
        "chunk_size":       64,            # frames per chunk
        "max_frames":       None,          # null = all
        "complete_only":    True,          # skip incomplete last frame
        "save_frame_plots": True,          # generate diagnostic plots
        # ASIC configuration
        "ASIC_num":         8,             # number of ASICs horizontally (512 cols / ASIC_num = cols per ASIC)
        "ASIC_mask":        [],             # list of masked ASIC indices to exclude (e.g., [0, 2] to skip ASICs 0 and 2)
        "metadata":         {},
    },
    "dark_frames": {
        "dark_run_file":            None,
        "pedestal_method":          "both",
        "sigma_clip_nsigma":        3.0,
        "compare_pedestal_methods": True,
        "save_npy":                 True,
        "save_h5":                  True,
    },
    "source_spectrum": {
        "source_run_file":    None,
        "calibration_file":   None,   # defaults to {output_dir}/dark_calibration.h5
        "seed_sigma":         5.0,
        "split_sigma":        3.0,
        "noise_scope":        "auto",
        "reject_extra":       False,
        "adu_min":            0.0,
        "adu_max":            10000.0,
        "n_bins":             1000,
        "save_events":        True,
        "save_events_to_file": None,   # defaults to {output_dir}/events.h5
        "prefer_offset":      "sigclip",
        "bad_pixel_mask":     {           # set enabled: false to disable
            "enabled":           True,
            "hot_rms_multiple":  5.0,     # noise > this × median(noise) → HOT
            "cold_rms_fraction": 0.1,     # noise < this × median(noise) → COLD/stuck
            "max_clip_fraction": 0.5,     # if >50% of dark frames were clipped at the
                                          # pixel → UNSTABLE (needs n_dark_frames)
            "n_dark_frames":     0,       # 0 → skip clip-fraction test
        },
    },
    "gain_calibration": {
        "events_file":       None,    # input: relative to data_dir then output_dir
        "output_file":       None,    # output: relative to output_dir
        "target_ev":         5895.0,  # Mn Kα reference energy [eV]
        "fit_window_frac":   0.20,    # Gaussian fit window ± fraction of target
        "rough_min_events":  100,     # Phase 1: min single events per parity
        "cti_row_bin_size":  64,      # Phase 3: rows per CTI bin
        "cti_min_events":    50,      # Phase 3: min events per row bin
        "cti_grade_filter":  None,    # None = all grades
        "col_min_events":    30,      # Phase 4: min events per column
        "col_with_bg":       False,   # Phase 4: Gaussian + linear background
        "col_grade_filter":  [0],     # Phase 4: grades used (singles default)
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
        print(cfg.dark_frames["pedestal_method"])

    Sections not in the YAML are populated with defaults.
    Unknown top-level sections are preserved as-is (for future extensions).
    """

    def __init__(self, raw: dict) -> None:
        self._raw = raw
        # Merge each known section with defaults
        for section, defaults in _DEFAULTS.items():
            user   = raw.get(section, {}) or {}
            merged = {**defaults, **user}
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
        """
        Resolve an input file path relative to data_dir.
        
        If path is None, returns None.
        If path is absolute, returns as-is.
        If path is relative, joins with data_dir.
        """
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return p
        return self.data_dir / p

    def resolve_output_path(self, path: str | Path | None) -> Path | None:
        """
        Resolve an output file path relative to output_dir.
        
        If path is None, returns None.
        If path is absolute, returns as-is.
        If path is relative, joins with output_dir.
        """
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return p
        return self.output_dir / p

    def calibration_path(self) -> Path:
        """Default path for dark calibration file."""
        return self.resolve_output_path("dark_calibration.h5")

    def events_path(self) -> Path:
        """Default path for events file."""
        return self.resolve_output_path("events.h5")

    def asics_for(self, section: str) -> list[str] | None:
        """Return resolved ASIC list for *section*.

        Reads from general['asics'] for single-hybrid mode (ASICs are not
        stage-specific).
        """
        from .lib.geometry import resolve_asics
        raw = self.general.get("asics", None)
        if isinstance(raw, str):
            raw = raw.split()
        return resolve_asics(raw)

    def methods_for_dark(self) -> list[str]:
        """Parse pedestal_method into ['median'], ['sigclip'], or both."""
        m = self.dark_frames.get("pedestal_method", "both").lower().strip()
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
# Template generator
# ──────────────────────────────────────────────────────────────────────────────

_TEMPLATE = """\
# pnccd_ana analysis configuration
# Generated by pnccd_ana.config — edit as needed.
# This template is for single hybrid RAW format (512x512, 1024x512, etc.).

general:
  output_dir: output                # output directory for results
  data_dir: .                       # base input directory (paths are relative to this)
  # Shared settings (used by both dark_frames and source_spectrum):
  data_format: raw                  # raw | h5 (file format)
  frame_rows: 1024                  # number of rows per frame (Y dimension)
  frame_cols: 512                   # number of columns per frame (X dimension)
  n_workers: 8                      # parallel workers
  chunk_size: 64                    # frames per chunk
  max_frames: null                  # null = all frames
  complete_only: true               # skip incomplete last frame
  save_frame_plots: true            # generate diagnostic plots (hitmap, spectrum, etc.)
  # ASIC configuration
  ASIC_num: 8                       # number of ASICs horizontally (frame_cols / ASIC_num = cols per ASIC)
  ASIC_mask: []                     # list of masked ASIC indices to exclude (e.g., [0, 2] to skip ASICs 0 and 2)
  metadata:
    operator: ""
    sample: ""
    run_date: ""

dark_frames:
  dark_run_file: dark_run.raw       # relative to data_dir
  pedestal_method: both             # median | sigclip | both
  sigma_clip_nsigma: 3.0
  compare_pedestal_methods: true    # compare median vs sigma-clip offsets
  save_npy: true                    # save as .npy files
  save_h5: true                     # save as .h5 file

source_spectrum:
  source_run_file: source_run.raw   # relative to data_dir
  calibration_file: dark_calibration.h5  # relative to data_dir (or output_dir if not found)
  seed_sigma: 5.0                   # threshold for finding candidate centres (3-8 × noise)
  split_sigma: 3.0                  # threshold for classifying neighbours (1-3 × noise)
  noise_scope: auto                 # auto | global | asic
  reject_extra: false               # if true, discard grade-13 "other" events
  adu_min: 0.0
  adu_max: 10000.0
  n_bins: 1000
  save_events: true
  save_events_to_file: events.h5    # relative to output_dir
  prefer_offset: sigclip            # sigclip | median
  bad_pixel_mask:                   # exclude hot / cold / unstable pixels from event recognition
    enabled: true
    hot_rms_multiple: 5.0           # noise > this × median(active noise) → HOT
    cold_rms_fraction: 0.1          # noise < this × median(active noise) → COLD / stuck
    max_clip_fraction: 0.5          # frac of dark frames clipped per pixel above which it's UNSTABLE
    n_dark_frames: 0                # 0 disables the clip-fraction test (no info in the cal file)

gain_calibration:
  events_file: null             # null = use {output_dir}/events.h5
  output_file: null             # null = use {output_dir}/gain_calibration.h5
  target_ev: 5895.0             # Mn Kα reference energy [eV]
  fit_window_frac: 0.20         # Gaussian fit window ± fraction of target
  rough_min_events: 100         # Phase 1: min single events per parity pool
  cti_row_bin_size: 64          # Phase 3: rows per CTI bin
  cti_min_events: 50            # Phase 3: min events per row bin
  cti_grade_filter: null        # null = all grades; or e.g. [0,1,2,3,4]
  col_min_events: 30            # Phase 4: min events per column
  col_with_bg: false            # Phase 4: add linear background to Gaussian fit
  col_grade_filter: [0]         # Phase 4: grades used for per-column fit
  save_plots: true

# cti_calibration:                  # uncomment and fill in for CTI step
#   events_file: run0001/events.h5
"""


def write_template(path: str | Path = "analysis.yaml") -> None:
    """Write a fully-commented template config to *path*."""
    with open(path, "w") as fh:
        fh.write(_TEMPLATE)
    print(f"Template config written to: {path}")
