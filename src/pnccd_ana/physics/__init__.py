"""physics sub-package – core algorithms (Stage 1: offset calibration)."""
from .offsets      import compute_offset_median, compute_offset_sigma_clip
from .common_mode  import cm_correct_frame, apply_common_mode_correction, compute_cm_noise
from .noise        import compute_noise, build_bad_pixel_mask
from ..io.geometry import (ASIC_SLICES, configure_asics, get_active_mask,
                           ASIC_WIDTH, ASIC_MASK, ASIC_NAMES,
                           get_asic_slice, resolve_asics, split_asics,
                           get_frame_bounds, ADC_MAX, ADC_RANGE)
