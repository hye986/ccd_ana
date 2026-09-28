# pnccd_ana

Offline offset-calibration pipeline for pnCCD detectors (Stage 1).

Measures the per-pixel electronic baseline (offset), per-pixel noise, the
common-mode correction, and a bad-pixel mask from dark frames (no X-ray
source). All calibration products are written as `.npy` arrays.

---

## Overview

```
Raw dark frames  →  [pnccd-offset]  →  offset / noise / CM / bad-pixel .npy
```

| Stage | CLI command | Input | Output |
|-------|-------------|-------|--------|
| Offset calibration | `pnccd-offset` | dark frames (RAW, 512×512) | `.npy` calibration arrays |

---

## Installation

```bash
git clone <repo_url>
cd ccd_ana

pip install -e .

# Verify
python -m pnccd_ana.cli.offset --help
```

**Dependencies:**

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `matplotlib` | Diagnostic plots |
| `pyyaml` | Configuration file parsing |

---

## Quick Start

### 1. Generate a configuration template

```bash
python -m pnccd_ana.cli.template analysis.yaml
```

Edit `analysis.yaml` to point to your dark-run file and set detector
parameters:

```yaml
general:
  output_dir: output
  data_dir: /path/to/your/data
  ASIC_num: 8            # number of ASICs horizontally

offset:
  dark_run_file: dark_run.raw
```

### 2. Run offset calibration

```bash
python -m pnccd_ana.cli.offset analysis.yaml
# or, after pip install -e .
pnccd-offset analysis.yaml
```

All outputs go to `output_dir` defined in the config.

---

## Detector and Data Format

### Sensor geometry

```
pnCCD sensor:  512 (Y) × 512 (X) pixels — single hybrid
Rolling shutter:  Y = 0 at bottom (readout side), reads upward
ASICs:  each reads an equal band of columns (512 / ASIC_num)
        e.g. with ASIC_num=8:  C7 = X[0:64], C6 = X[64:128], ..., C0 = X[448:512]

Array layout:  data[Y, X]
  Y = row index  (axis 0, vertical)
  X = col index  (axis 1, horizontal)
```

### RAW file format

```
[8-byte header: b'16BU0000']
For each frame:
  [0xFFFF marker (2B)] [ADC counter (4B)] [512 × uint16 pixels]  ← first row
  [0xFFFE marker (2B)] [ADC counter (4B)] [512 × uint16 pixels]  ← rows 2..512
  ...
```

A frame is 512 row records. The geometry is fixed; a trailing partial
frame (fewer than 512 rows) is ignored.

---

## Offset Calibration

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
          OUTSIDE  : pixels not inside an active ASIC
```

**Config options (`offset` section):**

```yaml
offset:
  dark_run_file: dark_run.raw       # path relative to data_dir
  pedestal_method: both             # median | sigclip | both
  sigma_clip_nsigma: 3.0            # sigma-clip threshold
  save_npy: true                    # save .npy calibration arrays
  save_plots: true                  # generate diagnostic plots
  split_even_odd: true              # separate even/odd column CM medians per ASIC
  bad_pixel_mask:
    enabled: true
    hot_rms_multiple: 5.0
    cold_rms_fraction: 0.1
    max_clip_fraction: 0.5
```

**Outputs:**

```
output/
├── offset_median.npy               (H, W) float32  — median offset
├── offset_sigclip.npy              (H, W) float32  — sigma-clip offset
├── noise.npy                       (H, W) float32  — per-pixel RMS
├── n_clipped_map.npy               (H, W) float32  — frames clipped per pixel
├── cm_map.npy                      (N, H, n_asics) float32 — CM correction
├── cm_noise_C{i}.npy               (H,) float32    — per-ASIC CM noise
├── bad_pixel_mask.npy              (H, W) bool     — True = bad pixel
└── offset/
    ├── offsets_offset.png          # 2-D offset maps + histograms
    ├── noise_offset.png            # noise map, CM noise, clip map
    ├── cm_map_C{i}.png             # CM correction per ASIC
    └── bad_pixels_offset.png       # bad pixel categories + per-ASIC counts
```

---

## Reloading Results

```python
from pnccd_ana.io import load_calibration_npy

cal = load_calibration_npy("output", prefer="sigclip")
offset        = cal["offset"]         # (H, W) float32
noise         = cal["noise"]          # (H, W) float32
n_clipped     = cal.get("n_clipped_map")
cm_map        = cal.get("cm_map")     # (N, H, n_asics)
cm_noise      = cal.get("cm_noise")   # dict: ASIC name -> (H,) float32
bad_pixel     = cal.get("bad_pixel_mask")   # (H, W) bool
```

To redraw the diagnostic plots from the `.npy` files without rerunning:

```bash
python tests/test_redraw_offset.py output
```

---

## Programmatic API

```python
import numpy as np
from pnccd_ana.config import Config
from pnccd_ana.cli.offset import run
from pnccd_ana.physics import (compute_offset_median,
                               compute_offset_sigma_clip,
                               apply_common_mode_correction,
                               compute_noise, compute_cm_noise,
                               build_bad_pixel_mask)

cfg = Config.from_yaml("analysis.yaml")
results = run(cfg)              # writes .npy + plots, returns result dict
r = results["global"]
print(r["noise"].shape)         # (H, W)
```

---

## Configuration Reference

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `general.n_workers` | 8 | Parallel frame-loading workers |
| `general.ASIC_num` | 8 | ASICs horizontally |
| `general.ASIC_mask` | [] | ASIC indices to exclude |
| `offset.pedestal_method` | both | `median` / `sigclip` / `both` |
| `offset.sigma_clip_nsigma` | 3.0 | Sigma-clip threshold |
| `offset.split_even_odd` | true | Separate even/odd column CM medians per ASIC |
| `offset.save_npy` | true | Write `.npy` calibration arrays |
| `offset.save_plots` | true | Generate diagnostic plots |

---

## File Structure

```
pnccd_ana/
├── physics/
│   ├── offsets.py               Offset estimation (median + sigma-clip)
│   ├── common_mode.py           CM correction
│   └── noise.py                 Noise + bad pixel mask
├── io/
│   ├── geometry.py              ASIC layout
│   ├── raw.py                   RAW reader (512×512, memory-mapped)
│   └── calibration.py           .npy calibration save / load
├── plotting/
│   ├── common.py                Shared helpers
│   └── offset_plots.py          Offset calibration plots
├── cli/
│   ├── template.py              Config template generator
│   └── offset.py                Offset calibration
└── config.py                    YAML configuration system
```
