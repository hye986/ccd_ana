"""
pnccd_ana.lib.geometry
======================
Shared detector geometry constants for the 1024x1024 pnCCD.

Array / axis convention
-----------------------
  data[frame, Y, X]
    Y = axis 0 = vertical   screen axis     (code: "Y" variable)
    X = axis 1 = horizontal screen axis     (code: "X" variable)

  imshow(arr) with origin="lower" maps Y->y, X->x automatically.
  NO transpose needed.

  Plot axis labels used throughout:
    x-axis: "X [detector column]"  (horizontal direction)
    y-axis: "Y [detector row]"      (vertical direction)

  Terminology note — two competing conventions live in this file:
    code's "X" variable = horizontal axis = your "detector column" direction
    code's "Y" variable = vertical axis   = your "detector row"    direction
  The physical CM algorithm (median per column) and every result are
  identical regardless of which word appears on the axis label.

ASIC layout  (your table format: Y0 X0 Y1 X1, exclusive end -> stored inclusive)
------------------------------------------------------------------------------
  H0: Y 512-1023, X 512-1023  (top-right)
  H1: Y   0- 511, X 512-1023  (bottom-right)
  H2: Y   0- 511, X   0- 511  (bottom-left)
  H3: Y 512-1023, X   0- 511  (top-left)

  Sensor as displayed (origin=lower-left  Y=0 at bottom):
  Y=1023 +-----------+-----------+
           |    H3    |    H0     |
           | (top-lft)|(top-right)|
  Y= 512 +-----------+-----------+
           |    H2    |    H1     |
           |(bot-left)|(bot-right)|
  Y=   0 +-----------+-----------+
        X=0         X=512       X=1023

  All plotting calls use origin="lower" explicitly so the sensor appears
  correct-side-up in saved PNGs.

CM correction:  median over Y  (axis=0 in 2-D, axis=1 in 3-D per-frame input)
               one CM value per X column; equivalently "median per vertical line".
"""

from __future__ import annotations

ADC_MAX   = 65535   # 2^16 - 1
ADC_RANGE = 65536   # 2^16

DETECTOR_HEIGHT = 1024   # Y axis  (axis 0 in 2-D array)
DETECTOR_WIDTH  = 1024   # X axis  (axis 1 in 2-D array)

# (Y0, Y1, X0, X1) inclusive
# Y0 = first row (smallest Y index), Y1 = last row (largest Y index)
# X0 = first column (smallest X index), X1 = last column (largest X index)
ASIC_SLICES: dict[str, tuple[int, int, int, int]] = {
    "H0": (512, 1023, 512, 1023),
    "H1": (  0,  511, 512, 1023),
    "H2": (  0,  511,   0,  511),
    "H3": (512, 1023,   0,  511),
}

ALL_ASICS = ["H0", "H1", "H2", "H3"]

ASIC_LABEL: dict[str, str] = {
    "H0": "top-right",
    "H1": "bottom-right",
    "H2": "bottom-left",
    "H3": "top-left",
}

ASIC_COLORS: dict[str, str] = {
    "H0": "#e07b39",
    "H1": "#4e9a9a",
    "H2": "#7b6fa0",
    "H3": "#6aaa64",
}

# matplotlib subplot (row, col) positions for the physical 2x2 ASIC layout.
# row=0 = top of figure = high Y  (H3 top-left, H0 top-right)
# col=0 = left of figure = low X  (H3 top-left, H2 bot-left)
ASIC_GRID_POS: dict[str, tuple[int, int]] = {
    "H3": (0, 0),
    "H0": (0, 1),
    "H2": (1, 0),
    "H1": (1, 1),
}


def resolve_asics(asics: list[str] | None) -> list[str] | None:
    """
    Normalise a user-supplied ASIC list.

    Parameters
    ----------
    asics : None / [] -> return None (full-frame mode)
            ["all"]   -> return ALL_ASICS
            ["H0", "h1", ...] -> upper-cased, validated

    Returns None (full-frame) or a non-empty validated list.
    """
    if not asics:
        return None
    if any(a.lower() == "all" for a in asics):
        return list(ALL_ASICS)
    result = [a.upper() for a in asics]
    bad = [a for a in result if a not in ASIC_SLICES]
    if bad:
        raise ValueError(f"Unknown ASIC name(s): {bad}.  Valid: {ALL_ASICS}")
    return result


def split_asics(data, asic_names: list[str]) -> dict[str, object]:
    """
    Extract ASIC sub-arrays from a (n_frames, Y, X) or (Y, X) array.

    Returns dict  name -> sub-array  (same ndim as input).
    """
    import numpy as np
    out: dict[str, object] = {}
    for name in asic_names:
        Y0, Y1, X0, X1 = ASIC_SLICES[name]
        if data.ndim == 3:
            out[name] = data[:, Y0:Y1+1, X0:X1+1]
        else:
            out[name] = data[Y0:Y1+1, X0:X1+1]
    return out
