"""physics sub-package – core algorithms for physics calculations."""
from .pedestal           import compute_offset_median, compute_offset_sigma_clip
from .common_mode        import cm_correct_frame, apply_common_mode_correction, compute_cm_noise
from .noise              import compute_noise, build_bad_pixel_mask
from .pattern_recognition import find_events, N_GRADES, GRADE_NAMES, EVENT_DTYPE, _GRADE_DEFS
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
