"""
pnccd_ana.lib.geometry
======================
Shared detector geometry constants for the 1024×1024 pnCCD.

Array / axis convention
───────────────────────
  data[frame, Y, X]
    Y = axis=1 = vertical   screen axis = detector column direction
    X = axis=2 = horizontal screen axis = detector row    direction

  imshow(arr) maps axis=0(Y)→y-screen, axis=1(X)→x-screen automatically.
  NO transpose needed.
  x-axis label: "X (detector row)"
  y-axis label: "Y (detector column)"

ASIC layout  (HYB table: Y0 X0 Y1 X1, exclusive end → stored inclusive)
─────────────────────────────────────────────────────────────────────────
  H0: Y 512-1023, X 512-1023  (top-right)
  H1: Y   0- 511, X 512-1023  (bottom-right)
  H2: Y   0- 511, X   0- 511  (bottom-left)
  H3: Y 512-1023, X   0- 511  (top-left)

  Sensor as displayed (X on x-axis, Y on y-axis, origin=lower-left):
  Y=1023 ┌───────────┬───────────┐
         │    H3     │    H0     │
         │ (top-left)│(top-right)│
  Y= 512 ├───────────┼───────────┤
         │    H2     │    H1     │
         │(bot-left) │(bot-right)│
  Y=   0 └───────────┴───────────┘
        X=0        X=512       X=1023

CM correction: median over Y (axis=1) for each X position.
"""

from __future__ import annotations

ADC_MAX   = 65535   # 2^16 - 1
ADC_RANGE = 65536   # 2^16

DETECTOR_HEIGHT = 1024   # Y axis
DETECTOR_WIDTH  = 1024   # X axis

# (Y0, Y1, X0, X1) inclusive
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

# matplotlib subplot (row, col) positions for the physical 2×2 ASIC layout.
# row=0 = top of figure = high-Y  (H3, H0)
# col=0 = left of figure = low-X  (H3, H2)
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
    asics : None / [] → return None (full-frame mode)
            ["all"]   → return ALL_ASICS
            ["H0", "h1", ...] → upper-cased, validated

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

    Returns dict  name → sub-array  (same ndim as input).
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
