"""physics sub-package – core algorithms."""
from .offsets      import compute_offset_median, compute_offset_sigma_clip
from .common_mode  import cm_correct_frame, apply_common_mode_correction, compute_cm_noise
from .noise        import compute_noise, build_bad_pixel_mask
from .event_filter import (find_events, get_cluster, n_pixels_per_event,
                            seed_pixels, adu_sums,
                            FLAG_BORDER, FLAG_OVERFLOW, FLAG_UNDERFLOW,
                            FLAG_MISFIT, _empty_result)
from .gain         import (fit_peak, fit_global_peak, fit_all_columns,
                            MN_KALPHA_EV, MN_KBETA_EV,
                            PeakFitResult, GlobalPeakResult, ColumnPeakResult)
from .cti          import (build_cte_map_column, build_cte_map,
                            cte_model, fit_cte_column, fit_all_columns_cte,
                            CTEParams, CTEFitResult, CTEMapResult)
from .calibrate    import (filter_events_for_iteration, update_mean_gain,
                            assign_grades, compute_final_energies,
                            FilteredEvents, GRADE_NAMES, N_GRADES,
                            GRADE_OTHER, _GRADE_LOOKUP)
from ..io.geometry import (ASIC_SLICES, configure_asics, get_active_mask,
                           ASIC_WIDTH, ASIC_MASK, ASIC_NAMES,
                           get_asic_slice, resolve_asics, split_asics,
                           get_frame_bounds, ADC_MAX, ADC_RANGE)
