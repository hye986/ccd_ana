"""io sub-package – I/O for raw data, HDF5, and geometry."""
from .raw      import (get_frame_indices, _apply_frame_selection,
                        read_chunk, make_chunks, process_frames_mt,
                        detect_raw_geometry, get_io_module)
from .hdf5     import (save_calibration_h5, load_calibration_h5,
                        save_calibration_npy, load_calibration_npy,
                        save_events_h5, load_events_h5,
                        save_energy_cal_h5, load_energy_cal_h5,
                        save_offset_results_h5,
                        save_energy_cal_results_h5,
                        load_offset_results_h5,
                        load_event_rec_results_h5,
                        load_energy_cal_results_h5,
                        save_event_rec_results_h5)
from .geometry import (ASIC_SLICES, ASIC_LABEL, ASIC_COLORS, ASIC_GRID_POS,
                        configure_asics, get_asic_slice, get_active_mask,
                        resolve_asics, split_asics, get_frame_bounds,
                        ADC_MAX, ADC_RANGE, ASIC_WIDTH, ASIC_NAMES, ASIC_MASK)
