"""
pnccd_ana.lib.geometry
======================
Detector geometry constants for pnCCD operation.

Array / axis convention
-----------------------
  data[frame, Y, X]
    Y = axis 0 = row index = vertical screen axis     (code: "Y" variable)
    X = axis 1 = column index = horizontal screen axis (code: "X" variable)

  imshow(arr, origin="lower") maps Y->y, X->x automatically.
  NO transpose needed.

  Plot axis labels used throughout:
    x-axis: "X [detector column]"  (horizontal direction)
    y-axis: "Y [detector row]"      (vertical direction)

Rolling shutter
---------------
  The sensor reads rows from bottom to top:
    - First line in RAW file → Y=0 (bottom, with origin="lower")
    - Last line in RAW file  → Y=H-1 (top)
  
  A "row" is a horizontal line (constant Y, all X pixels).

Multi-ASIC geometry
-------------------
  For 512 columns with 8 ASICs, each ASIC reads 64 columns:
    ASIC 0: X=0..63
    ASIC 1: X=64..127
    ...
    ASIC 7: X=448..511
  
  Common-mode correction is done per-ASIC (64 columns) for better accuracy.

Supported sizes
---------------
  512×512  — full single ASIC (H=512, W=512)
  1024×512 — 2 ASICs vertically stacked (H=1024, W=512)
  512×512 with 8 ASICs — 64 cols per ASIC (H=512, W=512)
  Other heights auto-detected via frame_rows config.

  Frame as displayed (origin=lower, Y=0 at bottom):
                      Y
                      ↑   H-1 (top)
                      │
                      └──X→  0 … W-1
"""

from __future__ import annotations

import numpy as np

ADC_MAX   = 65535   # 2^16 - 1
ADC_RANGE = 65536   # 2^16

# Default detector dimensions (updated at runtime from raw file)
DETECTOR_HEIGHT = 512   # Y axis  (axis 0 in 2-D array)
DETECTOR_WIDTH  = 512   # X axis  (axis 1 in 2-D array)

# ASIC configuration (updated at runtime via configure_asics)
_N_ASICS = 8                    # default number of ASICs horizontally
ASIC_WIDTH = 64                 # columns per ASIC (computed from width / N_ASICS)
ASIC_NAMES: list[str] = []     # populated by configure_asics()

# ASIC metadata (populated by configure_asics)
ASIC_LABEL: dict[str, str] = {}
ASIC_COLORS: dict[str, str] = {}
ASIC_GRID_POS: dict[str, tuple[int, int]] = {}

# ASIC slices (Y0, Y1, X0, X1) — updated at runtime
ASIC_SLICES: dict[str, tuple[int, int, int, int]] = {}

# ASIC mask: indices of ASICs to exclude from analysis
ASIC_MASK: set[int] = set()

# Default ASIC colors (8 colors for up to 8 ASICs)
_DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def configure_asics(n_asics: int, width: int, mask: list[int] | None = None) -> None:
    """
    Configure ASIC geometry based on detector dimensions.
    
    Parameters
    ----------
    n_asics : int — number of ASICs horizontally
    width   : int — total number of columns
    mask    : list of ASIC indices to exclude from analysis
    """
    global _N_ASICS, ASIC_WIDTH, ASIC_NAMES
    global ASIC_LABEL, ASIC_COLORS, ASIC_GRID_POS
    global ASIC_SLICES, ASIC_MASK
    
    _N_ASICS = n_asics
    ASIC_WIDTH = width // n_asics
    ASIC_NAMES = [f"C{i}" for i in range(n_asics)]
    ASIC_MASK = set(mask) if mask else set()
    
    # Populate metadata
    ASIC_LABEL.clear()
    ASIC_COLORS.clear()
    ASIC_GRID_POS.clear()
    ASIC_SLICES.clear()
    
    for i in range(n_asics):
        name = f"C{i}"
        ASIC_LABEL[name] = f"ASIC {i}" + (" (masked)" if i in ASIC_MASK else "")
        ASIC_COLORS[name] = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        ASIC_GRID_POS[name] = (0, i)
        ASIC_SLICES[name] = (0, DETECTOR_HEIGHT - 1, i * ASIC_WIDTH, (i + 1) * ASIC_WIDTH - 1)


def get_asic_slice(asic_name: str) -> tuple[slice, slice]:
    """Get (Y_slice, X_slice) for an ASIC."""
    if not ASIC_SLICES:
        configure_asics(_N_ASICS, DETECTOR_WIDTH)
    y0, y1, x0, x1 = ASIC_SLICES[asic_name]
    return (slice(y0, y1 + 1), slice(x0, x1 + 1))


def get_active_mask(height: int, width: int, n_asics: int, mask: list[int] | None = None) -> np.ndarray:
    """
    Create a boolean mask for active (non-masked) pixels.
    
    Returns a (height, width) boolean array where True = active pixel.
    """
    mask_set = set(mask) if mask else set()
    
    active = np.ones((height, width), dtype=bool)
    for i in range(n_asics):
        if i in mask_set:
            x0 = i * ASIC_WIDTH
            x1 = (i + 1) * ASIC_WIDTH
            active[:, x0:x1] = False
    
    return active


def get_frame_bounds(height: int, width: int) -> tuple[int, int, int, int]:
    """Return full frame bounds: (Y0, Y1, X0, X1)."""
    return (0, height - 1, 0, width - 1)


def resolve_asics(asics: list[str] | None, n_asics: int | None = None) -> list[str] | None:
    """
    Normalise a user-supplied ASIC list.

    Parameters
    ----------
    asics   : None / [] -> return all non-masked ASIC names
              list of ASIC names -> validate and return
              "single" -> return legacy single-hybrid ["C0"]
    n_asics : number of ASICs (for fallback)

    Returns None (full-frame mode) or list of ASIC names.
    """
    if asics is None or asics == []:
        # Return all non-masked ASICs
        return [f"C{i}" for i in range(n_asics or _N_ASICS) if i not in ASIC_MASK]
    
    if isinstance(asics, str):
        if asics.lower() in ("single", "single_hybrid", "legacy"):
            return ["C0"]
        asics = [asics]
    
    result = [a.upper() for a in asics]
    
    # Filter out masked ASICs
    valid_asics = [a for a in result if a in ASIC_SLICES and a not in _get_masked_names()]
    if not valid_asics:
        n = n_asics or _N_ASICS
        return [f"C{i}" for i in range(n) if i not in ASIC_MASK]
    
    return valid_asics


def _get_masked_names() -> set[str]:
    """Get set of masked ASIC names."""
    return {f"C{i}" for i in ASIC_MASK}


def split_asics(data, asic_names: list[str]) -> dict[str, object]:
    """
    Extract ASIC sub-arrays from a (n_frames, Y, X) or (Y, X) array.
    
    Parameters
    ----------
    data       : ndarray (n_frames, Y, X) or (Y, X)
    asic_names : list of ASIC names to extract
    
    Returns dict name -> sub-array (same ndim as input).
    """
    out: dict[str, object] = {}
    
    # Ensure ASIC_SLICES is initialized
    if not ASIC_SLICES:
        if data.ndim == 3:
            configure_asics(_N_ASICS, data.shape[2])
        else:
            configure_asics(_N_ASICS, data.shape[1])
    
    for name in asic_names:
        if name not in ASIC_SLICES:
            continue
        y_slice, x_slice = get_asic_slice(name)
        if data.ndim == 3:
            out[name] = data[:, y_slice, x_slice]
        else:
            out[name] = data[y_slice, x_slice]
    
    return out
