"""io sub-package – I/O for raw data, HDF5, and geometry."""
from .raw      import (get_frame_indices, _apply_frame_selection,
                        read_chunk, make_chunks, process_frames_mt,
                        detect_raw_geometry, get_io_module)
from .hdf5     import save_calibration_npy, load_calibration_npy
from .geometry import (ASIC_SLICES, ASIC_LABEL, ASIC_COLORS, ASIC_GRID_POS,
                        configure_asics, get_asic_slice, get_active_mask,
                        resolve_asics, split_asics, get_frame_bounds,
                        ADC_MAX, ADC_RANGE, ASIC_WIDTH, ASIC_NAMES, ASIC_MASK)
