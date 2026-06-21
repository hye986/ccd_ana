ROOT 

# Summary of the ROOT Event Reconstruction (Filter) Code

## Overview

The code implements a two-stage pipeline:
1. **`HStepOffNoiMapHLL`** — Calibration step: computes per-pixel offset and noise maps from dark or low-rate photon frames
2. **`HStepFilterEvents4`** — Event reconstruction step: applies calibration and reconstructs split-pixel events (patterns/clusters) from source frames

---

## Stage 1: `HStepOffNoiMapHLL::Calculate` — Offset & Noise Map

### Input Parameters (from your config)
| Parameter | Value | Meaning |
|---|---|---|
| `FilterFactor` | 5 | σ-threshold for hit rejection |
| `ThresholdMIPS` | 2000 ADU | Frame rejection threshold for MIPs |
| `SplitEvenOdd` | 1 | Separate CM correction for even/odd columns |
| `PhotonFrames` | 1 | Use photon-frame mode (not dark-frame mode) |
| `NADCs` | 8 | Number of ADC channels per row |
| `CorrCmMd` | 1 | Apply common-mode correction |

### Processing Flow

```
Read all frames into memory
         │
         ▼
Preliminary offset = median per pixel (across all frames, skip over/underflows)
         │
         ▼
Preliminary residuals = raw - preliminary offset
         │
         ▼
Preliminary common-mode correction (median_evenodd per row-segment)
         │
         ▼
Per-pixel noise estimate via MAD (Median Absolute Deviation)
  sigma_MAD = 1.4826 × median(|residual - median(residuals)|)
  Fallback: median of all valid sigmas (split by even/odd col if SplitEvenOdd)
         │
         ▼
Hit finding: residual > FilterFactor × sigma_MAD  →  mark pixel as hit
Expand hits to 3×3 neighborhood  →  SkipPixels mask
         │
         ▼
Final offset = mean over non-hit (good) pixels per pixel
         │
         ▼
Final residuals = raw - final offset  (NaN for bad/hit pixels)
         │
         ▼
Final common-mode correction (median_evenodd, in-place)
         │
         ▼
Noise = sqrt(variance) over good pixels after CM correction
         │
         ▼
Output: OffsetMap, NoiseMap, ResidualOffsetMap, CMMDTree (even/odd)
```

### Common-Mode Calculation (`HCommonModeMedianEvenOdd`)
- The detector row is divided into **NADCs segments** (512 cols / 8 ADCs = 64 pixels/segment)
- For each segment (row × ADC), the median is computed **separately** for even-indexed and odd-indexed columns
- Both medians are subtracted from their respective columns in-place
- This accounts for the known even/odd channel pattern noise in pnCCDs

---

## Stage 2: `HStepFilterEvents4::Calculate` — Event Reconstruction

### Input Parameters (from your config)
| Parameter | Value | Meaning |
|---|---|---|
| `ThresPrm` | 5 × noise | Primary threshold (seed pixel) |
| `ThresSec` | 3 × noise | Secondary threshold (split pixel neighbor) |
| `CorrCmMd` | 1 | Apply common-mode correction |
| `SplitEvenOdd` | 1 | Even/odd CM correction |
| `CorrCrosstalk` | 0 | No crosstalk correction |

### Per-Frame Processing Flow

```
Read raw frame
      │
      ▼
Mark overflow/underflow pixels  →  BadPixels mask
      │
      ▼
Subtract offset map
      │
      ▼
Common-mode correction (median per row-ADC segment, excluding BadPixels)
      │
      ▼
Threshold scan over all pixels:
  pixel > ThresSec  →  candidate (secondary)
  pixel > ThresPrm  →  candidate (primary/seed)
  pixel < ThresNeg  →  misfit
      ▼
Build MaskIndices list (sorted by pixel index)
```

### Cluster (Event) Reconstruction Algorithm

This is a **connected-component labeling** algorithm that exploits the sorted order of `MaskIndices`:

```
For each candidate pixel i (in sorted order):
  ┌─ Check left neighbor (MaskIndices[i-1] == ThisIndex-1)?
  │    → assign same event ID
  │
  └─ Check bottom neighbor (ThisIndex - ColCount)?
       Search backward in already-processed hits for matching index
       If found AND pixel already has event ID:
         → merge two events (rename all occurrences of old ID to new ID)
       If found AND pixel has no ID yet:
         → assign neighbor's event ID
       If not found:
         → assign new event ID (++MaxEventID)
```

**Key constraint**: Only 4-connectivity (left + bottom neighbors) is checked. This works because `MaskIndices` is **sorted** — above-row and right-column neighbors are always processed later.

### Event Output (TTree "EventTree")

After grouping pixels into events:
- Events are sorted by event ID
- Only events with **at least one pixel above `ThresPrm`** are saved
- Each tree entry contains:

| Branch | Type | Content |
|---|---|---|
| `NSignals` | Int | Number of pixels in pattern |
| `Signal[NSignals]` | Double | Offset+CM-corrected ADU values |
| `Row[NSignals]` | Int | Pixel row indices |
| `Column[NSignals]` | Int | Pixel column indices |
| `Flags` | UInt | Bit flags: overflow/underflow/misfit/border |
| `FrameIndex` | UInt | Frame number |

---

## Likely Cause of Quadruple Spectrum Issues

Based on the algorithm, here are the most probable sources of the flat enhancement in the quadruple region:

### 1. **Cluster merging bug**
The event-merging step (renaming event IDs) only iterates backward over already-processed pixels (`k=i-1` to `0`). If a chain of pixels shares IDs through multiple bottom-neighbor connections, the renaming may be **incomplete** — leaving some pixels of one physical event with different IDs, causing it to be split into two events or merged incorrectly with another.

```cpp
for (Long_t k=i-1; k>=0; --k) {
  EventNumberArray[k] = 
    (EventNumberArray[k]==ChangeEventNumber ?
     EventNumberArray[i] :
     EventNumberArray[k]);
}
```
This renames all prior occurrences of `ChangeEventNumber` to `EventNumberArray[i]`, but does **not** recursively chase chains (A→B→C type merging).

### 2. **Diagonal split pixels not connected**
Only left and bottom neighbors are checked — **diagonal pixels** (e.g., bottom-left, bottom-right) are not explicitly connected. For quadruples in a 2×2 arrangement, if two pixels share only a corner, they will form **two separate events** rather than one, distorting the quadruple spectrum.

### 3. **Secondary threshold application**
Pixels between `ThresSec` and `ThresPrm` are included as cluster members but do not on their own "seed" a new event. If the secondary threshold pixels around a split event are not properly connected via primary pixels, the charge sum will be incorrect.

### 4. **Even/odd CM over-correction for split events**
With `SplitEvenOdd=1`, the common-mode is computed on the already-processed frame. If a photon event happens to deposit charge in multiple pixels of the same CM segment, this signal gets partially subtracted by the CM correction — introducing a low-side tail in the spectrum for any pattern type, and this effect compounds for larger patterns (quadruples).
