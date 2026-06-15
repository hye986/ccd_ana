"""
pnccd_ana.lib.geometry
======================
Detector geometry constants for single-hybrid pnCCD operation.

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
  Common-mode correction: median over X for each Y row.

Single-hybrid supported sizes
-----------------------------
  512×512  — full single ASIC (H=512, W=512)
  1024×512 — 2 ASICs vertically stacked (H=1024, W=512)
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

# Single-hybrid: H0 covers the full frame
# These are kept for backward compatibility but for single-hybrid,
# the full frame is used (not sliced)
ALL_ASICS = ["H0"]
ASIC_LABEL = {"H0": "single-hybrid"}
ASIC_COLORS = {"H0": "#4e9a9a"}
ASIC_GRID_POS = {"H0": (0, 0)}

# For backward compatibility: H0 = full frame (dimensions set dynamically)
# This is a module-level variable that gets updated
ASIC_SLICES: dict[str, tuple[int, int, int, int]] = {}


def _update_asic_slices(height: int, width: int) -> None:
    """Update ASIC_SLICES to match frame dimensions."""
    global ASIC_SLICES
    ASIC_SLICES["H0"] = (0, height - 1, 0, width - 1)


def get_frame_bounds(height: int, width: int) -> tuple[int, int, int, int]:
    """Return full frame bounds for single-hybrid: (Y0, Y1, X0, X1)."""
    return (0, height - 1, 0, width - 1)


def resolve_asics(asics: list[str] | None) -> list[str] | None:
    """
    Normalise a user-supplied ASIC list.

    For single-hybrid mode, asics parameter is kept for backward compatibility
    but only H0 is used. Pass asics: [H0] or leave unset for full frame.

    Parameters
    ----------
    asics : None / [] -> return None (full-frame mode)
            ["H0"] or ["h0"] -> return ["H0"]

    Returns None (full-frame) or ["H0"] for single ASIC.
    """
    if not asics:
        return None
    if isinstance(asics, str):
        asics = [asics]
    result = [a.upper() for a in asics]
    if result == ["H0"]:
        return ["H0"]
    # For any other value, treat as full frame
    return None


def split_asics(data, asic_names: list[str]) -> dict[str, object]:
    """
    Extract ASIC sub-arrays from a (n_frames, Y, X) or (Y, X) array.
    
    For single-hybrid mode, returns the full array under key "H0".

    Returns dict  name -> sub-array  (same ndim as input).
    """
    out: dict[str, object] = {}
    for name in asic_names:
        if data.ndim == 3:
            out[name] = data
        else:
            out[name] = data
    return out
