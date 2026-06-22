"""
pnccd_ana.cli.energy_cal
=========================
Stage 3: Gain and CTI calibration.

Matches ROOT HStepGainMapCCDHLL outer iteration loop.

Usage
-----
  pnccd-energy-cal analysis.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import Config
from ..io.hdf5 import (load_events_h5, save_energy_cal_h5,
                        save_energy_cal_results_h5)
from ..physics.gain import (fit_global_peak, fit_all_columns,
                              MN_KALPHA_EV, GlobalPeakResult)
from ..physics.cti  import fit_all_columns_cte
from ..physics.calibrate import (filter_events_for_iteration,
                                  update_mean_gain,
                                  assign_grades,
                                  compute_final_energies)
from ..plotting.gain_plots import (plot_gain_map, plot_gain_histogram,
                                    plot_column_peaks)
from ..plotting.cti_plots  import (plot_cte_map, plot_signal_vs_row,
                                    plot_cti_summary)


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: Config) -> dict:
    """Execute gain + CTI calibration from a Config object."""

    ec      = cfg.energy_cal
    gen     = cfg.general
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    events_file_cfg = ec.get("events_file")
    if events_file_cfg:
        events_path = cfg.resolve_output_path(events_file_cfg)
    else:
        events_path = cfg.events_path()   # already = output_dir/events.h5

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    output_file_cfg = ec.get("output_file")
    if output_file_cfg:
        output_path = cfg.resolve_output_path(output_file_cfg)
    else:
        output_path = cfg.output_dir / "energy_cal.h5"

    # ── Load events ───────────────────────────────────────────────────────────
    print(f"\nLoading events from: {events_path}")
    ev_data      = load_events_h5(events_path)
    cluster_data = ev_data["cluster_data"]
    n_events     = len(cluster_data["flag"])
    print(f"  {n_events:,} events loaded")

    if n_events == 0:
        raise RuntimeError("No events found — run event_rec first.")

    # ── Detector geometry from config ─────────────────────────────────────────
    n_rows  = int(gen.get("frame_rows") or
                  ev_data["meta"].get("frame_shape", "1024x512").split("x")[0])
    n_cols  = int(gen.get("frame_cols") or
                  ev_data["meta"].get("frame_shape", "1024x512").split("x")[1])
    mid_row = n_rows // 2

    # ── Calibration parameters ────────────────────────────────────────────────
    calib_energy    = float(ec.get("target_ev",       MN_KALPHA_EV))
    roi_low         = float(ec.get("roi_low",         4000.0))
    roi_high        = float(ec.get("roi_high",        12000.0))
    n_params_gauss  = int(ec.get("n_params_gauss",    3))
    use_ud_split    = bool(ec.get("use_ud_split",     False))
    use_all_split   = bool(ec.get("use_all_split",    False))
    split_even_odd  = bool(ec.get("split_even_odd",   True))
    split_frame     = bool(ec.get("split_frame",      False))
    full_frame      = bool(ec.get("full_frame",       False))
    relax           = float(ec.get("cte_relax",       0.5))
    max_cte_iter    = int(ec.get("max_cte_iter",      10))
    save_plots      = bool(ec.get("save_plots",       True))

    # Sensor geometry derived from split_frame / full_frame
    row_border = n_rows // 2 if split_frame else n_rows
    n_row_fs   = (row_border // 2) if full_frame else 0

    n_outer_iter = 3 if use_all_split else 1
    n_parity     = 2 if split_even_odd else 1

    print(f"\nGain+CTI calibration")
    print(f"  Detector:    {n_rows} rows × {n_cols} cols")
    print(f"  Calib line:  {calib_energy:.1f} eV")
    print(f"  ROI:         [{roi_low:.0f}, {roi_high:.0f}] ADU")
    print(f"  Outer iters: {n_outer_iter}  "
          f"({'all splits' if use_all_split else 'singles only'})")
    print(f"  CTE max sub-iter: {max_cte_iter}  relax={relax}")
    print(f"  SplitFrame: {split_frame}  FullFrame: {full_frame}")
    print(f"  SplitEvenOdd: {split_even_odd}")

    # ── Initialise maps ───────────────────────────────────────────────────────
    gain_map  = np.zeros((n_rows, n_cols), dtype=np.float64)
    # mean_gain fallback: initialise to calib_energy / ROI_center
    roi_center = 0.5 * (roi_low + roi_high)
    mean_gain  = np.full(n_parity, calib_energy / roi_center,
                         dtype=np.float64)

    cte_result   = None
    global_peak  = None

    # ══════════════════════════════════════════════════════════════════════════
    # Outer iteration loop
    # ══════════════════════════════════════════════════════════════════════════
    for iteration in range(n_outer_iter):
        print(f"\n{'─'*60}")
        print(f"Outer iteration {iteration}")
        print(f"{'─'*60}")

        # ── 1. Filter events for this iteration ───────────────────────────────
        print("  Filtering events …")
        filtered = filter_events_for_iteration(
            cluster_data   = cluster_data,
            gain_map       = gain_map,
            mean_gain      = mean_gain,
            iteration      = iteration,
            roi_low        = roi_low,
            roi_high       = roi_high,
            n_rows         = n_rows,
            n_cols         = n_cols,
            mid_row        = mid_row,
            split_even_odd = split_even_odd,
            use_ud_split   = use_ud_split,
        )
        print(f"  Accepted: {filtered.n_accepted:,} events")
        if filtered.n_accepted < 100:
            print("  ⚠  Too few events — check ROI and source data.")
            break

        # ── 2. Global peak fit (Iteration 0 only — ROOT fallback) ─────────────
        if iteration == 0:
            print("  Fitting global peak (fallback) …")
            global_peak = fit_global_peak(
                adu_values     = filtered.adu_sum,
                col_indices    = filtered.col.astype(np.int32),
                row_indices    = filtered.row.astype(np.int32),
                roi_low        = roi_low,
                roi_high       = roi_high,
                mid_row        = mid_row,
                n_params       = n_params_gauss,
                split_even_odd = split_even_odd,
                split_frame    = split_frame,
            )
            for p in range(n_parity):
                for h in range(2 if split_frame else 1):
                    pp = global_peak.ppos[p, h]
                    print(f"    global peak parity={p} half={h}: "
                          f"{pp:.1f} ADU  "
                          f"→ {calib_energy/pp:.4f} eV/ADU")
            # Initialise mean_gain from global peak
            for p in range(n_parity):
                mean_gain[p] = calib_energy / global_peak.ppos[p, 0]

        # ── 3. Per-column orientation peak fit ────────────────────────────────
        print("  Fitting per-column peaks …")
        col_peaks = fit_all_columns(
            adu_values     = filtered.adu_sum,
            col_indices    = filtered.col.astype(np.int32),
            row_indices    = filtered.row,
            roi_low        = roi_low,
            roi_high       = roi_high,
            n_cols         = n_cols,
            mid_row        = mid_row,
            global_peak    = global_peak,
            n_params       = n_params_gauss,
            split_even_odd = split_even_odd,
            split_frame    = split_frame,
        )
        n_fallback = int(col_peaks.used_fallback.sum())
        print(f"  Fallback used: {n_fallback}/{n_cols * (2 if split_frame else 1)} column-halves")

        # ── 4. Per-column CTE fit ─────────────────────────────────────────────
        print("  Fitting per-column CTE …")
        cte_result = fit_all_columns_cte(
            adu_values     = filtered.adu_sum,
            col_indices    = filtered.col.astype(np.int32),
            row_indices    = filtered.row,
            col_peaks      = col_peaks,
            global_peak    = global_peak,
            calib_energy   = calib_energy,
            prev_gain_map  = gain_map,
            n_rows         = n_rows,
            n_cols         = n_cols,
            n_row_fs       = n_row_fs,
            row_border     = row_border,
            split_frame    = split_frame,
            full_frame     = full_frame,
            split_even_odd = split_even_odd,
            roi_low        = roi_low,
            roi_high       = roi_high,
            relax          = relax,
            max_iter       = max_cte_iter,
        )

        gain_map = cte_result.gain_map
        n_good   = int((cte_result.bad_gain_map[0, :] == 0).sum())
        n_fb     = int((cte_result.bad_gain_map[0, :] == 1).sum())
        n_kept   = int((cte_result.bad_gain_map[0, :] == 2).sum())
        print(f"  Gain map: {n_good} good  {n_fb} fallback  {n_kept} kept")

        # ── 5. Update mean_gain from good columns ─────────────────────────────
        mean_gain = update_mean_gain(
            gain_map     = gain_map,
            bad_gain_map = cte_result.bad_gain_map,
            n_cols       = n_cols,
            n_rows       = n_rows,
            row_border   = row_border,
            calib_energy = calib_energy,
            split_even_odd = split_even_odd,
        )
        for p in range(n_parity):
            print(f"  mean_gain parity={p}: {mean_gain[p]:.4f} eV/ADU")

    # ══════════════════════════════════════════════════════════════════════════
    # Post-calibration: grade assignment + energy conversion
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("Post-calibration: grade assignment …")
    grades = assign_grades(cluster_data, gain_map, n_rows, n_cols)
    grade_vals, grade_cnts = np.unique(grades, return_counts=True)
    for g, c in zip(grade_vals, grade_cnts):
        from ..physics.calibrate import GRADE_NAMES
        print(f"  grade {g:2d} ({GRADE_NAMES.get(int(g), '?'):10s}): {c:8,}")

    print("\nComputing calibrated energies …")
    energy_sum, cog_row, cog_col, seed_energy = compute_final_energies(
        cluster_data, gain_map, n_rows, n_cols)

    # Mean CTI (ROOT: from CTEMap row=1 values)
    if cte_result is not None:
        cte_row1  = cte_result.cte_map[1, :]
        good_cols = cte_result.bad_gain_map[0, :] == 0
        if good_cols.any():
            mean_cti = float(np.mean(1.0 - cte_row1[good_cols]))
            print(f"  Mean CTI (row=1, good columns): {mean_cti:.3e}")

    # ── Save calibration ──────────────────────────────────────────────────────
    print(f"\nSaving calibration: {output_path}")
    save_energy_cal_h5(
        path         = output_path,
        gain_map     = gain_map,
        cte_map      = cte_result.cte_map if cte_result else np.zeros_like(gain_map),
        bad_gain_map = cte_result.bad_gain_map if cte_result else np.zeros(gain_map.shape, np.int8),
        grades       = grades,
        energy_sum   = energy_sum,
        cog_row      = cog_row,
        cog_col      = cog_col,
        seed_energy  = seed_energy,
        mean_gain    = mean_gain,
        metadata     = {
            **gen.get("metadata", {}),
            "calib_energy_ev": calib_energy,
            "roi_low":         roi_low,
            "roi_high":        roi_high,
            "n_outer_iter":    n_outer_iter,
            "n_rows":          n_rows,
            "n_cols":          n_cols,
        },
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    if save_plots and cte_result is not None:
        print("\nGenerating plots …")
        plot_gain_map(gain_map, cte_result.bad_gain_map, out_dir)
        plot_gain_histogram(gain_map, cte_result.bad_gain_map,
                            split_even_odd, out_dir)
        plot_column_peaks(col_peaks, n_cols, out_dir)
        plot_cte_map(cte_result.cte_map, out_dir)
        plot_cti_summary(cte_result.cte_map,
                         cte_result.bad_gain_map, out_dir)
        plot_signal_vs_row(filtered, out_dir)

    print(f"\n✓ Energy calibration complete.  Output: {out_dir}/")

    return dict(
        gain_map     = gain_map,
        cte_map      = cte_result.cte_map if cte_result else None,
        bad_gain_map = cte_result.bad_gain_map if cte_result else None,
        grades       = grades,
        energy_sum   = energy_sum,
        cog_row      = cog_row,
        cog_col      = cog_col,
        seed_energy  = seed_energy,
        mean_gain    = mean_gain,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="pnCCD gain + CTI calibration (stage 3 of 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("config", help="Path to analysis.yaml")
    args = parser.parse_args(argv)
    cfg  = Config.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
