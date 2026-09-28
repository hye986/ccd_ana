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

Path resolution:
  - Input paths (dark_run_file) are resolved relative to data_dir if not absolute
  - Output .npy files (offset, noise, CM noise, bad pixel mask) are written
    to output_dir
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
        "save_plots":               True,
        # ROOT Analysis.OffNoiMapHLL.SplitEvenOdd 1
        # Compute separate CM medians for even- and odd-indexed columns
        # within each ASIC segment.  Corrects alternating-channel correlated
        # noise seen in pnCCDs.  Matches ROOT HCommonModeMedianEvenOdd.
        "split_even_odd":           True,
        # Bad-pixel mask thresholds (used when saving the mask + plots)
        "bad_pixel_mask": {
            "enabled":           True,
            "hot_rms_multiple":  5.0,
            "cold_rms_fraction": 0.1,
            "max_clip_fraction": 0.5,
        },
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
        """Default output directory for offset calibration .npy products."""
        return self.output_dir

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
  save_npy: true                    # write offset/noise/CM/bad-pixel .npy files
  save_plots: true                  # generate diagnostic plots
  split_even_odd: true              # separate CM medians for even/odd columns per ASIC
                                    # matches ROOT Analysis.OffNoiMapHLL.SplitEvenOdd 1
                                    # corrects alternating-channel correlated noise in pnCCDs
                                    # set false for standard single-median CM correction
  bad_pixel_mask:
    enabled: true
    hot_rms_multiple: 5.0           # noise > N × median(active noise) → HOT
    cold_rms_fraction: 0.1          # noise < N × median(active noise) → COLD/stuck
    max_clip_fraction: 0.5          # clip rate above this → UNSTABLE
"""


def write_template(path: str | Path = "analysis.yaml") -> None:
    """Write a fully-commented template config to *path*."""
    with open(path, "w") as fh:
        fh.write(_TEMPLATE)
    print(f"Template config written to: {path}")
