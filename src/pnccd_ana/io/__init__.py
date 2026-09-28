"""io sub-package – RAW input and .npy calibration I/O."""
from .raw          import (get_frame_indices, _apply_frame_selection,
                           read_chunk, make_chunks, process_frames_mt,
                           FRAME_HEIGHT, FRAME_WIDTH)
from .calibration  import save_calibration_npy, load_calibration_npy
from .geometry     import (ASIC_SLICES, configure_asics, get_active_mask,
                           ASIC_WIDTH, ASIC_NAMES, ASIC_MASK,
                           ADC_MAX, ADC_RANGE)
