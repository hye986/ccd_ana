# pnccd_ana

Single-hybrid pnCCD analysis for Fe-55 source characterization.


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


### Configuration

**Path resolution:**
- `general.data_dir` — base directory for input files (default: `.`)
- `general.output_dir` — base directory for output files (default: `output`)
- Input paths are resolved relative to `data_dir`
- Output paths are resolved relative to `output_dir`

**Example structure:**
```yaml
general:
  data_dir: /path/to/raw/data
  output_dir: ./results

dark_frames:
  dark_run_file: dark_run.raw   # resolves to /path/to/raw/data/dark_run.raw
  raw_height: 512               # or 1024 for 2-ASIC vertically stacked

source_spectrum:
  source_run_file: source.raw   # resolves to /path/to/raw/data/source.raw
  calibration_file: dark_calibration.h5  # resolves to /path/to/raw/data/dark_calibration.h5
  save_events_to_file: events.h5  # resolves to ./results/events.h5
```


### Supported frame sizes

Single-hybrid RAW format:
- **512×512** — full single ASIC
- **1024×512** — 2 ASICs vertically stacked
- Other heights auto-detected (128, 256, 512, 1024, 2048, 4096)


### Starting from saved events (for gain/CTI calibration)

```python
from pnccd_ana.utils.io_h5 import load_events_h5

data = load_events_h5("output/events.h5")
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
cal = load_calibration_h5(cfg.calibration_path(), asics=["H0"])

# Process one frame manually
raw = ...   # float32 (H, W)
noise_map = cal["global"]["noise"]
corrected, _ = cm_correct_frame(raw - cal["global"]["offset"])
events = find_events(corrected, noise_map, threshold_sigma=3.0)
```

---

## Axis / ASIC convention

```
data[frame, Y, X]
  Y = axis=0 = vertical screen   = detector row direction  → y-axis in plots
  X = axis=1 = horizontal screen = detector column direction → x-axis in plots

imshow(arr, origin="lower") — Y=0 at bottom

Single hybrid readout (e.g. H0 at top-right):
                    Y
                    ↑   0 … H-1
                    │
                    └──X→  0 … W-1

CM correction: median over Y (axis=0) for each X position.
```


