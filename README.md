```markdown
# pnccd_ana

Offline analysis pipeline for pnCCD detectors using Fe-55 X-ray sources.
Converts raw ADC readout frames into calibrated photon energy spectra.

---

## Overview

The pipeline runs in three sequential stages:

```
Raw dark frames  →  [1. offset]      →  offset.h5
Raw source frames →  [2. event_rec]  →  events.h5
events.h5        →  [3. energy_cal]  →  energy_cal.h5  +  calibrated spectra
```

| Stage | CLI command | Input | Output |
|-------|-------------|-------|--------|
| 1. Offset calibration | `pnccd-offset` | dark frames (RAW) | `offset.h5` |
| 2. Event recognition | `pnccd-event-rec` | source frames (RAW) | `events.h5` |
| 3. Energy calibration | `pnccd-energy-cal` | `events.h5` | `energy_cal.h5` |

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd ccd_ana

# Install (no compilation required — pure Python + scipy)
pip install -e .

# Verify
python -m pnccd_ana.cli.offset --help
```

**Dependencies** (installed automatically):

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `scipy` | Connected-component labelling (`scipy.ndimage.label`) |
| `h5py` | HDF5 file I/O |
| `matplotlib` | Diagnostic plots |
| `pyyaml` | Configuration file parsing |

---

## Quick Start

### 1. Generate a configuration template

```bash
python -m pnccd_ana.cli.template analysis.yaml
```

Edit `analysis.yaml` to point to your data files and set detector parameters.
Minimum required fields:

```yaml
general:
  output_dir: output
  data_dir: /path/to/your/data
  frame_rows: 1024       # sensor height in pixels
  frame_cols: 512        # sensor width in pixels
  ASIC_num: 8            # number of ASICs horizontally

offset:
  dark_run_file: dark_run.raw

event_rec:
  source_run_file: source_run.raw

energy_cal:
  kalpha_adu: 15000      # Kα peak position in ADU — read from event_rec spectrum
```

### 2. Run the three stages

```bash
python -m pnccd_ana.cli.offset      analysis.yaml
python -m pnccd_ana.cli.event_rec   analysis.yaml
python -m pnccd_ana.cli.energy_cal  analysis.yaml
```

All outputs go to `output_dir` defined in the config.

---

## Detector and Data Format

### Sensor geometry

```
pnCCD sensor:  1024 (Y) × 512 (X) pixels
Rolling shutter:  Y = 0 at bottom (readout side), reads upward
8 ASICs:  each reads 64 columns  (X = 0..63, 64..127, ..., 448..511)

Array layout:  data[Y, X]
  Y = row index  (axis 0, vertical)
  X = col index  (axis 1, horizontal)
```

### RAW file format

```
[8-byte header: b'16BU0000']
For each frame:
  [0xFFFF marker (2B)] [ADC counter (4B)] [W × uint16 pixels]  ← first row
  [0xFFFE marker (2B)] [ADC counter (4B)] [W × uint16 pixels]  ← rows 2..H
  ...
```

Frame geometry (H × W) is auto-detected from the marker pattern.
Pass `frame_rows` and `frame_cols` in config for non-standard sizes.

---

## Stage 1 — Offset Calibration (`offset`)

**Purpose:** Measure per-pixel electronic baseline (offset) and noise
from dark frames (no X-ray source).

**Algorithm:**

```
dark frames (N × H × W)
    │
    ├─ Offset estimation (per pixel, across N frames)
    │     · Median              — robust to rare hits (< 50% occupancy)
    │     · Sigma-clip          — iterative upper-tail clip, 3σ default
    │     · Both                — compute and compare both methods
    │
    ├─ Common-mode correction   — per-ASIC median per row, subtracted
    │
    ├─ Noise estimation         — per-pixel RMS after CM correction
    │
    └─ Bad-pixel mask
          HOT      : noise > 5 × median(noise)
          COLD     : noise < 0.1 × median(noise)
          NON-FINITE: NaN / Inf / ≤ 0
          UNSTABLE : >50% of dark frames clipped during sigma-clip
```

**Config options (`offset` section):**

```yaml
offset:
  dark_run_file: dark_run.raw       # path relative to data_dir
  pedestal_method: both             # median | sigclip | both
  sigma_clip_nsigma: 3.0            # sigma-clip threshold
  save_h5: true                     # save offset.h5
  save_npy: true                    # save .npy files for quick access
```

**Outputs:**

```
output/
├── offset.h5                       # primary: offset maps, noise, bad pixel mask
│   ├── global/offset/median        (H, W) float32
│   ├── global/offset/sigmaclip     (H, W) float32
│   ├── global/noise/pixel_rms      (H, W) float32
│   ├── global/noise/cm_rms_C{i}    (H,)   float32  — per ASIC
│   └── global/noise/n_clipped_per_pixel  (H, W) float32
├── offset_results.h5               # plot-backing data (histograms, maps)
├── bad_pixel_mask.npy              # (H, W) bool — quick numpy access
└── global/
    ├── offsets_global.png          # 2-D offset maps + histograms
    ├── noise_global.png            # noise map, CM noise, clip map
    ├── cm_map_C{i}.png             # CM correction per ASIC
    └── bad_pixels_global.png       # bad pixel categories + per-ASIC counts
```

---

## Stage 2 — Event Recognition (`event_rec`)

**Purpose:** Find and classify photon hit clusters in source frames
after offset subtraction and common-mode correction.

**Algorithm (matches ROOT HStepFilterEvents4):**

```
source frame (H × W)
    │
    ├─ Subtract offset map
    ├─ Common-mode correction (per-ASIC median per row)
    │
    ├─ Threshold scan
    │     prim_mask  =  frame > seed_sigma  × noise   (primary,   e.g. 5σ)
    │     sec_mask   =  frame > split_sigma × noise   (secondary, e.g. 3σ)
    │
    ├─ Connected-component labelling  (scipy.ndimage.label)
    │     Connectivity: LEFT + BELOW only  (no diagonal)
    │     Each connected group of sec_mask pixels → one cluster
    │
    ├─ Accept cluster if at least one pixel exceeds prim_mask
    │
    ├─ Shape classification
    │     seed  = pixel with maximum ADU in cluster
    │     offsets = (dY, dX) of all other cluster pixels relative to seed
    │     match against grade table → assign grade
    │
    └─ ADU sum = sum of ALL cluster pixels
       ADU seed = value of the seed pixel
```

**Grade table:**

| Grade | Name | Pattern (relative to seed) |
|-------|------|---------------------------|
| 0 | single | seed only |
| 1 | double | right (0,+1) |
| 2 | double | up (+1,0) |
| 3 | double | left (0,-1) |
| 4 | double | down (-1,0) |
| 5 | triple | up + right |
| 6 | triple | left + up |
| 7 | triple | down + left |
| 8 | triple | down + right |
| 9 | quadruple | up + right + up-right |
| 10 | quadruple | up + left + up-left |
| 11 | quadruple | down + left + down-left |
| 12 | quadruple | down + right + down-right |
| 13 | T-left | up + right + down |
| 14 | T-right | up + left + down |
| 15 | b-left | up + right + down + down-right |
| 16 | b-right | up + left + down + down-left |
| 17 | I-shape | up + down |
| 18 | other | any unrecognised pattern |

Grades are defined in `pnccd_ana/physics/pattern_recognition.py` (`_GRADE_DEFS`).
Adding a new grade requires only editing that one table.

**Config options (`event_rec` section):**

```yaml
event_rec:
  source_run_file: source_run.raw
  calibration_file: offset.h5       # defaults to {output_dir}/offset.h5
  seed_sigma: 5.0                   # primary threshold (3–8 × noise)
  split_sigma: 3.0                  # secondary threshold (1–3 × noise)
  reject_extra: false               # drop grade-18 events if true
  bad_pixel_mask:
    enabled: true
    hot_rms_multiple: 5.0
    cold_rms_fraction: 0.1
```

**Outputs:**

```
output/
├── events.h5                       # primary: all recognised events
│   ├── events/{Y, X, grade, adu_sum, adu_seed}
│   ├── spectra/grade{NN}           # per-grade ADU histograms
│   ├── maps/hit_count              (H, W) int32
│   └── maps/mean_adu               (H, W) float32
├── event_rec_results.h5            # plot-backing data
└── plots:
    ├── event_maps_global.png       # hit count + mean ADU maps
    ├── spectrum_full_detector.png  # per-grade + grouped spectra
    ├── grade_distribution.png      # event count per grade
    └── raw_spectrum_global.png     # pixel-level ADU before recognition
```

**Event structured array fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Y` | int16 | Row of seed pixel |
| `X` | int16 | Column of seed pixel |
| `grade` | int8 | Pattern grade (0–18) |
| `adu_sum` | float32 | Total charge: sum of all cluster pixels |
| `adu_seed` | float32 | Charge at seed pixel only |

---

## Stage 3 — Energy Calibration (`energy_cal`)

**Purpose:** Convert ADU to eV using the Fe-55 Mn Kα line (5895 eV)
as a reference. Accounts for even/odd column gain differences and
charge transfer inefficiency (CTI).

**Pipeline:**

```
events.h5
    │
    ├─ Phase 1: Rough global gain
    │     Single-pixel events (grade 0) only
    │     Split by column parity (even / odd X)
    │     Fit Mn Kα peak in ADU → G_even, G_odd  [eV/ADU]
    │
    ├─ Phase 2: Preliminary energy
    │     E_prelim = adu_sum × G_rough(parity)  [eV]
    │     Applied to all grades
    │
    ├─ Phase 3: CTI estimation and correction
    │     Bin events by row → fit Kα peak per bin
    │     Linear model:  E_meas(row) = E₀ × (1 − row × CTI)
    │     Apply correction:  E_cti = E_prelim / (1 − row × CTI)
    │
    ├─ Phase 4: Per-column fine-gain
    │     Fit Kα peak per column in CTI-corrected data
    │     f_col = 5895 / E_fit_col
    │
    └─ Final energy [eV]:
          E = adu_sum × G_rough(parity) × (1/(1 − row×CTI)) × f_col(X)
```

**Important:** You must supply the Kα peak position in ADU, read from
the `event_rec` spectrum plot (`spectrum_full_detector.png`):

```yaml
energy_cal:
  kalpha_adu: 15000      # ← read this from your spectrum plot
  kalpha_adu_window: 0.20
```

**Config options (`energy_cal` section):**

```yaml
energy_cal:
  events_file: null              # null = use {output_dir}/events.h5
  output_file: null              # null = use {output_dir}/energy_cal.h5
  target_ev: 5895.0              # Mn Kα reference energy [eV]
  kalpha_adu: 15000              # REQUIRED: Kα peak in ADU from spectrum plot
  kalpha_adu_window: 0.20        # fit window ± fraction of kalpha_adu
  fit_window_frac: 0.15          # Phases 3+4 fit window fraction (eV space)
  rough_min_events: 100          # Phase 1: min events per parity pool
  cti_row_bin_size: 64           # Phase 3: rows per CTI bin
  cti_min_events: 50             # Phase 3: min events per row bin
  cti_grade_filter: null         # null = all grades
  col_min_events: 30             # Phase 4: min events per column
  col_with_bg: false             # Phase 4: Gaussian + linear background
  col_grade_filter: [0]          # Phase 4: use singles only
```

**Outputs:**

```
output/
├── energy_cal.h5                   # primary: calibration constants + energies
│   ├── gain/{g_even, g_odd, peak_even_adu, peak_odd_adu}
│   ├── cti/{cti_coefficient, e0, row_bins, peak_per_bin, ...}
│   ├── column_gain/{f_col, peak_col, n_events_col, success_col}
│   └── calibrated_events/energy_ev (N,) float32
├── energy_cal_results.h5           # plot-backing data for redrawing
└── plots:
    ├── cal_phase1_rough_gain.png   # even/odd ADU spectra + Gaussian fits
    ├── cal_phase3_cti.png          # peak vs row + CTI linear fit + residuals
    ├── cal_cti_correction_check.png # before/after CTI comparison
    ├── cal_phase4_column_gain.png  # f_col map + distribution
    ├── cal_final_spectrum.png      # all-grades spectra + Kα resolution fit
    ├── cal_pixel_gain_map.png      # G_eff per column + gain histogram
    └── cal_cti_per_col.png         # local CTI per column bin + 2-D peak map
```

---

## Output File Summary

```
output/
├── offset.h5                   offset maps, noise, bad pixel mask
├── offset_results.h5           plot-backing data for offset stage
├── bad_pixel_mask.npy          (H, W) bool — fast numpy access
│
├── events.h5                   events, spectra, hit maps
├── event_rec_results.h5        plot-backing data for event_rec stage
│
├── energy_cal.h5               gains, CTI, f_col, calibrated energies
└── energy_cal_results.h5       plot-backing data for energy_cal stage
```

The `*_results.h5` files store all histogram bin edges and counts,
fit parameters, and 2-D maps used to generate the diagnostic plots.
Load them to redraw plots or refit peaks without rerunning the pipeline.

---

## Reloading Results for Redrawing

```python
from pnccd_ana.analysis import (
    load_dark_results,
    load_source_results,
    load_gain_results,
)

# Offset stage
dark = load_dark_results("output/offset_results.h5")
noise_map    = dark["noise"]["map"]
noise_edges  = dark["noise"]["hist_edges"]
noise_counts = dark["noise"]["hist_counts"]

# Event rec stage
src   = load_source_results("output/event_rec_results.h5")
grades = src["grade_distribution"]["grades"]
counts = src["grade_distribution"]["counts"]

# Energy calibration stage
gain = load_gain_results("output/energy_cal_results.h5")
kalpha = gain["final_spectrum"]["kalpha_fit"]
print(f"Kα peak:       {kalpha['peak_ev']:.1f} eV")
print(f"FWHM:          {kalpha['fwhm_ev']:.1f} eV")
print(f"Resolution:    {kalpha['resolution_pct']:.2f}%")

# Load calibrated energies directly
import h5py, numpy as np
with h5py.File("output/energy_cal.h5") as f:
    energy_ev = f["calibrated_events/energy_ev"][:]
print(f"{len(energy_ev):,} events calibrated")
```

---

## Programmatic API

```python
from pnccd_ana.analysis import (
    run_offset,
    load_calibration,
    build_noise_map,
    build_bad_pixel_mask,
    process_frames,
    run_energy_cal,
)
import numpy as np

# Stage 1 — offset calibration
run_offset("analysis.yaml")

# Load calibration
cal   = load_calibration("output/offset.h5")
noise = build_noise_map(cal)
bad   = build_bad_pixel_mask(noise)

# Stage 2 — event recognition (batch, no CLI)
from pnccd_ana.io.raw import read_chunk, get_frame_indices
indices = get_frame_indices("source_run.raw")
raw     = read_chunk("source_run.raw", indices[:1000])
events  = process_frames(raw, cal, noise, bad_pixel_mask=bad,
                          seed_sigma=5.0, split_sigma=3.0)
print(f"{len(events):,} events found")

# Grade distribution
grades, cnts = np.unique(events["grade"], return_counts=True)
for g, n in zip(grades, cnts):
    print(f"  G{g:2d}: {n:7,}  ({100*n/len(events):.1f}%)")
```

---

## Configuration Reference

Full annotated template — generate with:

```bash
python -m pnccd_ana.cli.template analysis.yaml
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `general.frame_rows` | auto | Sensor height in pixels (Y) |
| `general.frame_cols` | auto | Sensor width in pixels (X) |
| `general.ASIC_num` | 8 | ASICs horizontally |
| `general.ASIC_mask` | [] | ASIC indices to exclude |
| `offset.pedestal_method` | both | `median` / `sigclip` / `both` |
| `offset.sigma_clip_nsigma` | 3.0 | Sigma-clip threshold |
| `event_rec.seed_sigma` | 5.0 | Primary threshold multiplier |
| `event_rec.split_sigma` | 3.0 | Secondary threshold multiplier |
| `energy_cal.kalpha_adu` | — | **Required:** Kα peak in ADU |
| `energy_cal.cti_row_bin_size` | 64 | Rows per CTI measurement bin |
| `energy_cal.col_grade_filter` | [0] | Grades used for per-column fit |

---

## Adding New Event Grades

All grade definitions live in one place:

```
pnccd_ana/physics/pattern_recognition.py  →  _GRADE_DEFS
```

To add a grade, append one entry to `_GRADE_DEFS`:

```python
# Example: add a 5-pixel plus-shape
(19, "plus", frozenset({(+1,0), (-1,0), (0,+1), (0,-1)})),
```

No recompilation needed. Plots and spectra update automatically
because all grade colours and labels are derived from `_GRADE_DEFS`
at runtime.

---

## Physical Reference

| Line | Energy | Typical ADU |
|------|--------|-------------|
| Mn Kα | 5895.0 eV | ~15000 (detector-dependent) |
| Mn Kβ | 6490.0 eV | ~16500 (detector-dependent) |

Charge transfer inefficiency (CTI) convention:

```
E_measured(row) = E₀ × (1 − row × CTI)

Y = 0  at readout (bottom) → fewest transfers → least CTI loss
Y = H-1 at top            → most transfers  → most CTI loss
Positive CTI → peak energy decreases with Y
```

---

## File Structure

```
pnccd_ana/
├── physics/
│   ├── pedestal.py              Offset estimation
│   ├── common_mode.py           CM correction
│   ├── noise.py                 Noise + bad pixel mask
│   ├── pattern_recognition.py   Event finding + grade table
│   ├── gain.py                  Gain fitting (Phase 1-2)
│   └── cti.py                   CTI calibration (Phase 3-4)
├── io/
│   ├── geometry.py              ASIC layout
│   ├── hdf5.py                  HDF5 save / load
│   └── raw.py                   RAW file reader (memory-mapped)
├── plotting/
│   ├── common.py                Shared helpers
│   ├── offset_plots.py          Offset calibration plots
│   ├── event_plots.py           Event recognition plots
│   ├── gain_plots.py            Rough gain plots
│   ├── cti_plots.py             CTI + column gain plots
│   └── spectrum_plots.py        Final spectrum plots
├── cli/
│   ├── template.py              Config template generator
│   ├── offset.py                Stage 1: offset calibration
│   ├── event_rec.py             Stage 2: event recognition
│   └── energy_cal.py            Stage 3: energy calibration
├── analysis.py                  High-level public API
└── config.py                    YAML configuration system
```

