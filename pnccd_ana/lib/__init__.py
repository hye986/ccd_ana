"""lib sub-package – core algorithms."""
from .geometry            import (ASIC_SLICES, ALL_ASICS, ASIC_LABEL,
                                   ASIC_COLORS, ASIC_GRID_POS,
                                   resolve_asics, split_asics,
                                   _update_asic_slices, get_frame_bounds,
                                   ADC_MAX, ADC_RANGE,
                                   DETECTOR_HEIGHT, DETECTOR_WIDTH)
from .pedestal            import (compute_offset_median,
                                   compute_offset_sigma_clip)
from .common_mode         import (cm_correct_frame,
                                   apply_common_mode_correction,
                                   compute_cm_noise)
from .noise               import compute_noise, build_bad_pixel_mask
from .pattern_recognition import (find_events, N_GRADES, GRADE_NAMES,
                                   EVENT_DTYPE, _GRADE_DEFS,
                                   local_max_5x5)
from .process_frames      import (correct_frame, make_worker, process_frames_mt)
