# AGENTS.md — pnccd_ana repository notes

Persistent notes for AI agents working in this repo.

## Build / test
```bash
pip install -q numpy scipy h5py hdf5plugin matplotlib pyyaml pytest
python setup_ext.py build_ext --inplace        # build the C extension
python -m pytest tests/ -v                     # run tests
```

The C extension must be rebuilt after any change to
`pnccd_ana/lib/_pattern_recognition_c.c`.

## Array / axis convention

```
data[frame, Y, X]        axis 0 = Y = vertical on screen  (detector "row")
                          axis 1 = X = horizontal on screen (detector "column")
imshow(arr, origin="lower")  Y=0 at bottom, X=0 at left
```

Your team's terminology:
- "row" = x-axis = vertical   = code's Y variable  ← same
- "column" = y-axis = horizontal = code's X variable ← same

ASIC layout (your table format Y0 X0 Y1 X1, exclusive → stored inclusive):
```
H0: Y=[512,1023] X=[512,1023]  top-right
H1: Y=[  0, 511] X=[512,1023]  bottom-right
H2: Y=[  0, 511] X=[  0, 511]  bottom-left
H3: Y=[512,1023] X=[  0, 511]  top-left
```
All plotting uses `origin="lower"` so sensor appears correct-side-up.

## What the package does (4 stages)

**Stage 1 — Dark frame calibration**: raw dark frames → offset maps + noise maps
  → `run_dark_calibration()` or `python -m pnccd_ana.cli.dark_frame_ana`

**Stage 2 — Noise + bad-pixel maps**: calibration HDF5 → noise_map + bad_pixel_mask
  → `build_noise_map()`, `build_bad_pixel_mask()` in `lib.noise`

**Stage 3 — Source processing**: raw frames → CM-corrected → events
  → `process_frames()` or `python -m pnccd_ana.cli.source_ana`
  - Algorithm: seed (local-max 5×5, above seed_sigma × noise) →
    pattern (central 3×3, above split_sigma × noise) → grade 0–12/13
  - Tie-breaker: lex-smallest (y,x) among 5×5 tied-max pixels → one event per cluster

**Stage 4 — Gain (future)**: events → photon energy calibration

## Threshold summary

| # | Stage | Knob | Default | What |
|---|-------|------|---------|-------|
| 1 | Pedestal | `dark_frames.rollover_low_frac`  | 0.10 | low value cutoff (× ADC_MAX) |
| 2 | Pedestal | `dark_frames.rollover_high_frac` | 0.80 | "typically-high" gate |
| 3 | Pedestal | `dark_frames.sigma_clip_nsigma` | 3.0 | sigma-clip multiplier |
| 4 | Event | `source_spectrum.seed_sigma` | 5.0 | seed threshold |
| 5 | Event | `source_spectrum.split_sigma` | 3.0 | neighbour threshold |
| 6 | Event (implicit) | lex-min tie-breaker | hard-coded | one event per tied pixel cluster |
| 7 | Bad-pixel | `bad_pixel_mask.hot_rms_multiple` | 5.0 | hot if noise > k × median(active) |
| 8 | Bad-pixel | `bad_pixel_mask.cold_rms_fraction` | 0.1 | cold/stuck if noise < k × median |
| 9 | Bad-pixel | `bad_pixel_mask.max_clip_fraction` | 0.5 | unstable if clip_rate > k |

## File layout (clean structure)

```
pnccd_ana/
  analysis.py          ← START HERE: explains all 4 stages with examples
  lib/
    geometry.py         ← ASIC bounds, grid positions, axis convention
    pedestal.py         ← median + sigma-clip offset computation
    common_mode.py      ← per-column median CM correction
    noise.py            ← pixel noise + build_bad_pixel_mask()
    pattern_recognition ← seed → pattern → grade (C + Python)
    process_frames.py   ← correct_frame, make_worker, process_frames_mt
  cli/
    dark_frame_ana.py  ← Stage 1 CLI
    source_ana.py       ← Stage 3 CLI
  utils/
    io_h5.py            ← HDF5 read/write
    io_raw.py           ← RAW read/write
    plotting.py         ← diagnostics + spectrum plots
  config.py             ← YAML config system
```

## Bugs fixed (do not regress)

1. **Tie-breaker** (2026-05): adjacent equal-value pixels double-counted as
   separate events.  Fixed by adding lex-smallest rule in both C extension
   and Python fallback.  Build C ext after changing.
2. **grade_bitmask_histogram** tie-check: mirror of the above fix.
3. **ASIC boundary exclusion**: C extension seeds only scanned at x=2..1021
   (Python fallback has no boundary exclusion, which is intentional — it
   handles the ±2 edge naturally).  Both paths give the same results for
   interior events.  X=0 and X=1 pixels are excluded by BOTH paths (by the
   ±2 border which is larger than the 5×5 window — a conservative design choice
   with no effect on real pnCCD data since ASIC seams are at X=512, not at X=0/1).

## Conventions

- Modify the original file rather than creating `*_fix.py` variants.
- Tests live in `tests/`; run with `pytest`.  C and Python paths always tested.
- Co-authored-by: openhands <openhands@all-hands.dev> on any commits.
