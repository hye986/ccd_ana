# Summary of ROOT Gain Calibration Code

## Overview

The code implements a two-step gain and CTI (Charge Transfer Inefficiency) calibration pipeline for a pnCCD detector using Fe55 source data. It consists of two main classes:

---

## Class 1: `HStepGainMapCCDHLL` — Gain Map Calculation

### Inputs
| Dependency | Type | Description |
|---|---|---|
| `EventTree` | `TTree` | Reconstructed events from source frames |
| `OffsetMap` | `TH2` | Offset map from dark frames (provides detector geometry) |

### Key Parameters
| Parameter | Default | Description |
|---|---|---|
| `Analysis.Calib.Line` | `"Mn-Ka"` | Calibration line (Fe55 → Mn-Kα at 5898.8 eV) |
| `Analysis.Calib.ROIlow/high` | 4000/12000 | ADU range of interest |
| `Analysis.Calib.UseUDSplit` | false | Include up/down split events |
| `Analysis.Calib.UseAllSplit` | false | Include all split events (enables iterations) |
| `Analysis.Calib.SplitFrame` | false | Sensor has two readout directions (top+bottom) |
| `Analysis.Calib.FullFrame` | false | Full-frame mode (frame store CTI also fitted) |
| `Analysis.Calib.SplitEvenOdd` | false | Separate gain histograms for even/odd columns |

---

### Step-by-Step Algorithm

#### **Phase 1: Event Filtering**

The tree branches read per event are:
```
NSignals, Signal[], Row[], Column[], Flags
```

**Iteration 0** — singles (and optional U/D splits):
- Selects events with `NSignals==1`, or `NSignals==2` with same column (vertical split) if `UseUDSplit=true`
- Rejects overflow/underflow via `EventFlag & 0x7 == 0`
- Computes signal sum; for splits, uses **signal-weighted row** (center of gravity, rounded)
- Applies ROI cut: `ROIlow < SignalSum < ROIhigh`

**Iterations 1–2** (only if `UseAllSplit=true`, up to 3 iterations total):
- Accepts **all** split patterns
- For each pixel in the cluster, retrieves the gain from the current `GainMap` (or falls back to `MeanGain` mean)
- Computes gain-corrected signal sum: `Signal[i] × gain[i]`
- Computes center-of-gravity column and row (gain-weighted, rounded)
- Divides total sum by gain at COG pixel → reconstructed ADU signal
- Applies same ROI cut

---

#### **Phase 2: Per-Column Peak Finding (Orientation Fit)**

Events are **sorted by column** for efficient per-column processing.

For each column (and each half if `SplitFrame`):

1. Fill a 1D histogram of `SignalSum` values
2. Fit a **Gaussian** (`HUtils::GaussFit`, with `NParamsGauss` parameters, typically 3)
3. Store peak position `PPositions[col][half]` and sigma `Sigmas[col][half]`

**Fallback mechanism** (Iteration 0 only):
- If fewer than 20 events per column → use **global peak** (fitted from all events across all columns, per parity/half)
- If fit fails or peak outside ROI → also use global peak
- Global peak requires ≥200 events to be trusted

---

#### **Phase 3: Per-Column CTE + Gain Fit (Iterative)**

For each column, up to **10 sub-iterations**:

**3a. Build CTE correction map** for this column:

Starting from row 0 (CTE = 1.0), propagate:
```
CTEMap[row] = CTEMap[row-1] / CTE_factor
```
- Frame store rows: use `CTEfs_b` (or `CTEfs_t`)
- Image area rows: use `CTEim_b` (or `CTEim_t`)
- For `SplitFrame`: top half reads out **downward** from `RowCount-1`

**3b. Apply CTE correction to signal**:
```
CTECorrectedSignal[i] = SignalSum[i] × CTEMap[Row[i]]
```

**3c. Determine fitting ROI**:
- Fit a Gaussian to corrected histogram → use `μ ± 2σ` as new ROI
- Fallback: use stored `PPositions ± 2×Sigmas` from Phase 2

**3d. Fit the CTE function** (`CTEFunction` via `TGraph::Fit`):

The fit function has up to 9 parameters:

| Index | Name | Description |
|---|---|---|
| 0 | `PPos_b` | Peak position, bottom half |
| 1 | `PPos_t` | Peak position, top half |
| 2 | `CTEfs_b` | CTE in frame store, bottom |
| 3 | `CTEim_b` | CTE in image area, bottom |
| 4 | `CTEfs_t` | CTE in frame store, top |
| 5 | `CTEim_t` | CTE in image area, top |
| 6–8 | geometry | `NRowFS`, `RowIndexROBorder`, `RowCount` (fixed) |

Fixed parameters depend on mode:
- Frame store mode (`FullFrame=false`): `CTEfs` fixed to 1.0
- Single readout direction (`SplitFrame=false`): top-half parameters fixed

**3e. Update CTE estimates** with **relaxed update** (damping factor 0.5):
```
ThisCTE *= pow(FittedCTE, 0.5)
```

**Convergence check**: stop when:
- Peak positions change < 1%
- All CTE parameters within 5×10⁻⁶ of 1.0

---

#### **Phase 4: Fill Output Maps**

For each column, per half:

**If fit is good** (`PPos` in ROI and all CTE > 0.995):
```
Gain[col, row=0] = CalibEnergy / PPos
Gain[col, row]   = Gain[col, row-1] / CTE_factor   (propagated downward)
CTE[col, row]    = CTE[col, row-1]  × CTE_factor
```

**If fit failed** (final iteration only):
- Use global peak position → uniform fallback gain
- Mark `BadGainMap`: `0`=good, `1`=fallback used, `2`=kept previous iteration values

---

### Outputs

| Result | Type | Description |
|---|---|---|
| `GainMap` | `TH2D` | Per-pixel gain in eV/ADU |
| `CTEMap` | `TH2D` | Per-pixel CTE correction factor |
| `BadGainMap` | `TH2C` | Quality flag per pixel |

Also stores mean CTI (= 1 − CTE) as parameter `Analysis.Calib.CTI`.

---

## Class 2: `HStepCalibEvents` — Apply Calibration to Events

### Purpose
Converts raw ADU signals to energy (eV) using the computed `GainMap`.

### Algorithm

For each event in `EventTree`:

```
Energy[i] = Signal[i]  (if Signal[i] > 0, else 0)
Energy[i] × Gain[Row[i], Col[i]]
```

Then computes **energy-weighted center of gravity**:
```
EnergySum = Σ Energy[i]
ColumnCOG = Σ (Col[i] × Energy[i]) / EnergySum
RowCOG    = Σ (Row[i] × Energy[i]) / EnergySum
```

Optional temperature drift correction (compile-time flag `TEMPERATURECOMPENSATION`):
```
Energy[i] × Θ,   Θ = 1 + Ag1·ΔT + Ag2·ΔT²
```

### Output: `CalibEvents` TTree

| Branch | Type | Description |
|---|---|---|
| `NSignals` | `Int_t` | Number of pixels in event |
| `Energy[]` | `Double_t[]` | Per-pixel energy in eV |
| `EnergySum` | `Float_t` | Total event energy in eV |
| `ColumnCOG` | `Float_t` | Energy-weighted column position |
| `RowCOG` | `Float_t` | Energy-weighted row position |

---

## Key Design Points

1. **CTI propagates gain with row**: pixels far from readout register have systematically lower apparent signal → gain map encodes this as higher gain value to compensate
2. **Iterative approach**: using split events in later iterations improves statistics but requires a bootstrapped gain map from singles first
3. **SplitFrame support**: handles sensors with bidirectional readout (e.g., top rows read upward, bottom rows read downward)
4. **Robust fallback chain**: global peak → fallback gain → bad pixel flagging ensures the gain map is always fully populated
