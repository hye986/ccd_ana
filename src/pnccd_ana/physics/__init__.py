"""physics sub-package – core algorithms for physics calculations."""
from .offsets           import compute_offset_median, compute_offset_sigma_clip
from .common_mode        import cm_correct_frame, apply_common_mode_correction, compute_cm_noise
from .noise              import compute_noise, build_bad_pixel_mask
from .event_filter import (find_events, EVENT_DTYPE,
                            N_GRADES, GRADE_OTHER, GRADE_NAMES,
                            _GRADE_DEFS, _GRADE_LOOKUP,
                            classify_cluster, build_c_grade_table)
from .gain               import (RoughGainCalibrator, ColumnGainCalibrator,
                                  apply_rough_gain, apply_full_calibration,
                                  MN_KALPHA_EV, MN_KBETA_EV,
                                  fit_peak, PeakFitResult,
                                  SINGLE_GRADES, SPLIT_GRADES)
from .cti                import CtiCalibrator, CtiResult
from ..io.geometry import (ASIC_SLICES, configure_asics, get_active_mask,
                          ASIC_WIDTH, ASIC_MASK, ASIC_NAMES,
                          get_asic_slice, resolve_asics, split_asics,
                          get_frame_bounds, ADC_MAX, ADC_RANGE)
