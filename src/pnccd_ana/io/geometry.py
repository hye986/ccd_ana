"""
pnccd_ana.io.geometry
======================
Detector geometry for the 512×512 single-hybrid pnCCD.

Array / axis convention
-----------------------
  data[frame, Y, X]
    Y = axis 0 = row index = vertical screen axis      (code: "Y" variable)
    X = axis 1 = column index = horizontal screen axis (code: "X" variable)

  imshow(arr, origin="lower") maps Y->y, X->x automatically.
  NO transpose needed.

Rolling shutter
---------------
  The sensor reads rows from bottom to top:
    - First line in RAW file → Y=0 (bottom, with origin="lower")
    - Last line in RAW file  → Y=H-1 (top)

  A "row" is a horizontal line (constant Y, all X pixels).

Multi-ASIC geometry
-------------------
  For 512 columns with 8 ASICs, each ASIC reads 64 columns.
  ASIC naming is reversed from hardware to match the physical layout:
    ASIC 7: X=0..63     (C7 in code)
    ASIC 6: X=64..127   (C6 in code)
    ...
    ASIC 0: X=448..511  (C0 in code)

  Common-mode correction is done per-ASIC (64 columns) for better accuracy.

  Frame as displayed (origin=lower, Y=0 at bottom):
                      Y
                      ↑   511 (top)
                      │
                      └──X→  0 … 511
"""

from __future__ import annotations

import numpy as np

ADC_MAX   = 65535   # 2^16 - 1
ADC_RANGE = 65536   # 2^16

# ASIC configuration (updated at runtime via configure_asics)
_N_ASICS = 8                    # default number of ASICs horizontally
ASIC_WIDTH = 64                 # columns per ASIC (computed from width / N_ASICS)
ASIC_NAMES: list[str] = []      # populated by configure_asics()

# ASIC slices (Y0, Y1, X0, X1) — updated at runtime
ASIC_SLICES: dict[str, tuple[int, int, int, int]] = {}

# ASIC mask: indices of ASICs to exclude from analysis
ASIC_MASK: set[int] = set()


def configure_asics(n_asics: int, width: int, height: int,
                    mask: list[int] | None = None) -> None:
    """
    Configure ASIC geometry based on detector dimensions.

    ASIC_SLICES stores (Y0, Y1, X0, X1) with **exclusive** upper bounds,
    matching Python slice semantics: data[Y0:Y1, X0:X1].

    Parameters
    ----------
    n_asics : int — number of ASICs horizontally
    width   : int — total number of columns
    height  : int — total number of rows
    mask    : list of ASIC indices to exclude from analysis
    """
    global _N_ASICS, ASIC_WIDTH, ASIC_NAMES
    global ASIC_SLICES, ASIC_MASK

    _N_ASICS = n_asics
    ASIC_WIDTH = width // n_asics
    # Hardware: col 0-63 = ASIC n_asics-1, col 64-127 = ASIC n_asics-2, ...
    # Code:     col 0-63 = C(n_asics-1),  col 64-127 = C(n_asics-2), ...
    ASIC_NAMES = [f"C{n_asics - 1 - i}" for i in range(n_asics)]
    ASIC_MASK = set(mask) if mask else set()

    ASIC_SLICES.clear()
    for i in range(n_asics):
        name = f"C{n_asics - 1 - i}"
        ASIC_SLICES[name] = (0, height, i * ASIC_WIDTH, (i + 1) * ASIC_WIDTH)


def get_active_mask(height: int, width: int, n_asics: int,
                    mask: list[int] | None = None) -> np.ndarray:
    """
    Create a boolean mask for active (non-masked) pixels.

    Computes the per-ASIC column width locally from (width // n_asics) so
    that the result is always consistent with the supplied dimensions,
    regardless of whether configure_asics() has been called.

    Returns a (height, width) boolean array where True = active pixel.
    """
    mask_set = set(mask) if mask else set()
    asic_width = width // n_asics          # local — never reads global ASIC_WIDTH

    active = np.ones((height, width), dtype=bool)
    for i in range(n_asics):
        if i in mask_set:
            active[:, i * asic_width:(i + 1) * asic_width] = False

    return active
