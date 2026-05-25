"""
pnccd_ana.config
================
YAML-based configuration system.

Config file format (YAML)
─────────────────────────

  general:
    output_dir: run0001
    save_frame_plots: true
    save_asic_plots: true
    metadata:                     # arbitrary key-value pairs saved into HDF5
      operator: Alice
      sample: Fe55_source
      run_date: 2026-05-01

  dark_frames:
    dark_run_file: dark_run.h5
    pedestal_method: both         # median | sigclip | both
    sigma_clip_nsigma: 3.0
    asics: [H0, H1, H2, H3]      # or 'all' or omit for full-frame
    rollover_check: true
    unwrap_rollover: false
    rollover_low_frac: 0.10
    rollover_high_frac: 0.80
    max_frames: null              # null = all
    save_npy: true
    save_h5: true
    compare_pedestal_methods: true

  source_spectrum:
    source_run_file: source_run.h5
    calibration_file: run0001/dark_calibration.h5   # or dir of .npy
    threshold_sigma: 3.0
    asics: [H0, H1, H2, H3]
    max_frames: null
    n_workers: 8
    chunk_size: 64
    adu_min: 0.0
    adu_max: 10000.0
    n_bins: 1000
    save_events: true
    save_events_to_file: run0001/events.h5

  gain_calibration:             # populated in a future step
    events_file: run0001/events.h5
    # ... (to be defined)

  cti_calibration:              # populated in a future step
    events_file: run0001/events.h5
    # ... (to be defined)

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
        "output_dir":       "output",
        "save_frame_plots": True,
        "save_asic_plots":  True,
        "metadata":         {},
    },
    "dark_frames": {
        "dark_run_file":            None,
        "data_format":              "h5",   # "h5" or "raw"
        "raw_height":               None,   # RAW only: frame height (auto-detected if None)
        "raw_width":                None,   # RAW only: frame width  (consistency check)
        "pedestal_method":          "both",
        "sigma_clip_nsigma":        3.0,
        "asics":                    None,
        "rollover_check":           True,
        "unwrap_rollover":          False,
        "rollover_low_frac":        0.10,
        "rollover_high_frac":       0.80,
        "max_frames":               None,
        "complete_only":            True,
        "save_npy":                 True,
        "save_h5":                  True,
        "compare_pedestal_methods": True,
        "n_workers":                8,
        "chunk_size":               64,
    },
    "source_spectrum": {
        "source_run_file":    None,
        "data_format":        "h5",   # "h5" or "raw"
        "raw_height":         None,   # RAW only: frame height (auto-detected if None)
        "raw_width":          None,   # RAW only: frame width  (consistency check)
        "calibration_file":   None,
        "seed_sigma":         5.0,
        "split_sigma":        3.0,
        "noise_scope":        "auto",
        "reject_extra":       False,
        "asics":              None,
        "max_frames":         None,
        "complete_only":      True,
        "n_workers":          8,
        "chunk_size":         64,
        "adu_min":            0.0,
        "adu_max":            10000.0,
        "n_bins":             1000,
        "save_events":        True,
        "save_events_to_file": None,
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
    "gain_calibration": {},
    "cti_calibration":  {},
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

    def asics_for(self, section: str) -> list[str] | None:
        """Return resolved ASIC list for *section* (e.g. 'dark_frames')."""
        from .lib.geometry import resolve_asics
        raw = getattr(self, section, {}).get("asics", None)
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

general:
  output_dir: run0001
  save_frame_plots: true
  save_asic_plots: true
  metadata:
    operator: ""
    sample: ""
    run_date: ""

dark_frames:
  dark_run_file: dark_run.h5
  data_format: h5             # h5 | raw
  # raw_height: 1024          # RAW only: frame height in pixels (auto-detected if omitted)
  # raw_width: 1024           # RAW only: frame width  in pixels (optional check)
  pedestal_method: both         # median | sigclip | both
  sigma_clip_nsigma: 3.0
  asics: [H0, H1, H2, H3]      # list of ASICs, 'all', or remove for full-frame
  rollover_check: true
  unwrap_rollover: false
  rollover_low_frac: 0.10
  rollover_high_frac: 0.80
  max_frames: null              # null = all frames
  complete_only: true
  save_npy: true
  save_h5: true
  compare_pedestal_methods: true
  n_workers: 8
  chunk_size: 64

source_spectrum:
  source_run_file: source_run.h5
  data_format: h5             # h5 | raw
  # raw_height: 1024          # RAW only: frame height in pixels (auto-detected if omitted)
  # raw_width: 1024           # RAW only: frame width  in pixels (optional check)
  calibration_file: run0001/dark_calibration.h5
  seed_sigma: 5.0           # threshold for finding candidate centres (3–8 × noise)
  split_sigma: 3.0          # threshold for classifying neighbours (1–3 × noise)
  noise_scope: auto          # auto | global | asic; auto avoids unrealistically low ASIC noise maps
  reject_extra: false        # if true, discard grade-13 "other" events
  asics: [H0, H1, H2, H3]
  max_frames: null
  complete_only: true
  n_workers: 8
  chunk_size: 64
  adu_min: 0.0
  adu_max: 10000.0
  n_bins: 1000
  save_events: true
  save_events_to_file: run0001/events.h5
  prefer_offset: sigclip        # sigclip | median
  bad_pixel_mask:               # exclude hot / cold / unstable pixels from event recognition
    enabled: true
    hot_rms_multiple: 5.0       # noise > this × median(active noise) → HOT
    cold_rms_fraction: 0.1      # noise < this × median(active noise) → COLD / stuck
    max_clip_fraction: 0.5      # frac of dark frames clipped per pixel above which it's UNSTABLE
    n_dark_frames: 0            # 0 disables the clip-fraction test (no info in the cal file)

# gain_calibration:             # uncomment and fill in for gain step
#   events_file: run0001/events.h5

# cti_calibration:              # uncomment and fill in for CTI step
#   events_file: run0001/events.h5
"""


def write_template(path: str | Path = "analysis.yaml") -> None:
    """Write a fully-commented template config to *path*."""
    with open(path, "w") as fh:
        fh.write(_TEMPLATE)
    print(f"Template config written to: {path}")
