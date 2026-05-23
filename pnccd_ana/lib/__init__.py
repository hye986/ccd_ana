"""lib sub-package – core algorithms."""
from .geometry            import (ASIC_SLICES, ALL_ASICS, ASIC_LABEL,
                                   ASIC_COLORS, ASIC_GRID_POS,
                                   resolve_asics, split_asics,
                                   ADC_MAX, ADC_RANGE,
                                   DETECTOR_HEIGHT, DETECTOR_WIDTH)
from .pedestal            import (detect_and_unwrap_rollover,
                                   compute_offset_median,
                                   compute_offset_sigma_clip)
from .common_mode         import (cm_correct_frame,
                                   apply_common_mode_correction,
                                   compute_cm_noise)
from .noise               import compute_noise
from .pattern_recognition import (find_events, N_GRADES, GRADE_NAMES,
                                   EVENT_DTYPE, _GRADE_DEFS)
