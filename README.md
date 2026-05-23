# fsp_offAna


## Quickstart

### Step 1 — generate a template config

1. Install dependencies (once)
```bash
pip install numpy h5py hdf5plugin matplotlib pyyaml
```
2. Build the C extension (once, after any code change)
```bash
cd /path/containing/pnccd_ana/
python setup_ext.py build_ext --inplace
```
This compiles _pattern_recognition_c.so into pnccd_ana/lib/. If compilation fails, the code automatically falls back to pure Python/NumPy.

3. Generate a config template (once per run)
```bash
python -m pnccd_ana.cli.dark_frame_ana analysis.yaml --template
```
Then edit analysis.yaml.


### Step 2 — dark-frame calibration

```bash
python -m pnccd_ana.cli.dark_frame_ana analysis.yaml
```

### Step 3 — source analysis

```bash
python -m pnccd_ana.cli.source_ana analysis.yaml
```


### Use cases with different input format

Case 1 — HDF5, full frame, analyse H1 only

yaml

```
dark_frames:
  dark_run_file: dark_run.h5
  data_format: h5
  asics: [H1]
```

Case 2 — RAW, full frame, analyse H1 only

yaml

```
dark_frames:
  dark_run_file: dark_run.raw
  data_format: raw
  # raw_height and raw_width can be omitted — 1024 is in the auto-detect list
  asics: [H1]
```

Case 3 — RAW, 512×512 (H1 readout only)

yaml

```
dark_frames:
  dark_run_file: dark_run_H1.raw
  data_format: raw
  raw_height: 512        # tell the reader the frame height
  raw_width: 512         # optional but good as a consistency check
  asics: [H1]            # must match what was recorded — used for embeddin
```


### Starting from saved events (for gain/CTI calibration)

```python
from pnccd_ana.utils.io_h5 import load_events_h5

data = load_events_h5("run0001/events.h5")
events    = data["events"]       # structured array Y, X, grade, adu_sum
spectra   = data["spectra"]      # dict grade → histogram counts
bin_edges = data["bin_edges"]
hit_count = data["hit_count"]
meta      = data["meta"]

# Filter to singles only
singles = events[events["grade"] == 0]
```

When you add `gain_calibration` or `cti_calibration` steps, add their
section to the YAML and a new `cli/gain_cal.py` that calls
`load_events_h5` and reads `cfg.gain_calibration`.

---

## Using the library directly (notebooks / scripts)

Every algorithm is importable individually:

```python
import numpy as np
from pnccd_ana.lib.pedestal      import compute_offset_sigma_clip
from pnccd_ana.lib.common_mode   import cm_correct_frame
from pnccd_ana.lib.pattern_recognition import find_events
from pnccd_ana.utils.io_h5       import load_calibration_h5
from pnccd_ana.config            import Config

# Load config
cfg = Config.from_yaml("analysis.yaml")

# Load calibration
cal = load_calibration_h5("run0001/dark_calibration.h5",
                           asics=["H0", "H1"])

# Process one frame manually
raw = ...   # float32 (1024, 1024)
noise_map = cal["global"]["noise"]
corrected, _ = cm_correct_frame(raw - cal["global"]["offset"])
events = find_events(corrected, noise_map, threshold_sigma=3.0)
```

---

## Axis / ASIC convention

```
data[frame, Y, X]
  Y = axis=1 = vertical screen   = detector column direction  → y-axis in plots
  X = axis=2 = horizontal screen = detector row direction     → x-axis in plots

imshow(arr) — no transpose needed.

Sensor layout (X on x-axis, Y on y-axis, origin=lower-left):
Y=1023 ┌──────────┬──────────┐
       │    H3    │    H0    │  top
Y= 512 ├──────────┼──────────┤
       │    H2    │    H1    │  bottom
Y=   0 └──────────┴──────────┘
      X=0       X=512     X=1023
      left              right

CM correction: median over Y (axis=1) for each X position.
```


