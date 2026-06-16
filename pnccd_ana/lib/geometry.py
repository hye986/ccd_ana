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

ADC_MAX   = 65535   # 2^16 - 1
ADC_RANGE = 65536   # 2^16

# Default detector dimensions (updated at runtime from raw file)
DETECTOR_HEIGHT = 512   # Y axis  (axis 0 in 2-D array)
DETECTOR_WIDTH  = 512   # X axis  (axis 1 in 2-D array)

# ASIC configuration
N_ASICS = 8                    # number of ASICs horizontally
ASIC_WIDTH = 64                # columns per ASIC
ASIC_NAMES = [f"H{i}" for i in range(N_ASICS)]

# ASIC metadata
ASIC_LABEL = {f"H{i}": f"ASIC {i}" for i in range(N_ASICS)}
ASIC_COLORS = {
    "H0": "#1f77b4", "H1": "#ff7f0e", "H2": "#2ca02c", "H3": "#d62728",
    "H4": "#9467bd", "H5": "#8c564b", "H6": "#e377c2", "H7": "#7f7f7f",
}
ASIC_GRID_POS = {f"H{i}": (0, i) for i in range(N_ASICS)}

# ASIC slices (Y0, Y1, X0, X1) — updated at runtime
# For 512x512 with 8 ASICs: each ASIC handles 64 columns
ASIC_SLICES: dict[str, tuple[int, int, int, int]] = {}

# Legacy single-hybrid support
ALL_ASICS_LEGACY = ["H0"]
ASIC_LABEL_LEGACY = {"H0": "single-hybrid"}
ASIC_COLORS_LEGACY = {"H0": "#4e9a9a"}
ASIC_GRID_POS_LEGACY = {"H0": (0, 0)}


def _update_asic_slices(height: int, width: int) -> None:
    """
    Update ASIC_SLICES to match frame dimensions.
    
    For 512 columns with 8 ASICs: 512 / 8 = 64 cols per ASIC
    """
    global ASIC_SLICES
    ASIC_SLICES.clear()
    
    if width == 512 and N_ASICS == 8:
        # Standard 8-ASIC configuration
        for i in range(N_ASICS):
            ASIC_SLICES[f"H{i}"] = (0, height - 1, i * ASIC_WIDTH, (i + 1) * ASIC_WIDTH - 1)
    else:
        # Fallback: treat as single ASIC
        ASIC_SLICES["H0"] = (0, height - 1, 0, width - 1)


def get_asic_slice(asic_name: str) -> tuple[int, slice]:
    """Get (Y_slice, X_slice) for an ASIC."""
    if asic_name not in ASIC_SLICES:
        _update_asic_slices(DETECTOR_HEIGHT, DETECTOR_WIDTH)
    y0, y1, x0, x1 = ASIC_SLICES[asic_name]
    return (slice(y0, y1 + 1), slice(x0, x1 + 1))


def get_frame_bounds(height: int, width: int) -> tuple[int, int, int, int]:
    """Return full frame bounds: (Y0, Y1, X0, X1)."""
    return (0, height - 1, 0, width - 1)


def resolve_asics(asics: list[str] | None, width: int = 512) -> list[str] | None:
    """
    Normalise a user-supplied ASIC list.

    Parameters
    ----------
    asics : None / [] -> return full list of ASIC names for the detector
            list of ASIC names -> validate and return
            "single" -> return legacy single-hybrid ["H0"]

    Returns None (full-frame mode) or list of ASIC names.
    """
    if asics is None or asics == []:
        # Default: return all ASICs for the detector width
        return [f"H{i}" for i in range(N_ASICS)]
    
    if isinstance(asics, str):
        if asics.lower() in ("single", "single_hybrid", "legacy"):
            return ["H0"]
        asics = [asics]
    
    result = [a.upper() for a in asics]
    
    # Validate ASIC names exist
    valid_asics = [a for a in result if a in ASIC_SLICES]
    if not valid_asics:
        return [f"H{i}" for i in range(N_ASICS)]
    
    return valid_asics


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
            _update_asic_slices(data.shape[1], data.shape[2])
        else:
            _update_asic_slices(data.shape[0], data.shape[1])
    
    for name in asic_names:
        if name not in ASIC_SLICES:
            continue
        y_slice, x_slice = get_asic_slice(name)
        if data.ndim == 3:
            out[name] = data[:, y_slice, x_slice]
        else:
            out[name] = data[y_slice, x_slice]
    
    return out
