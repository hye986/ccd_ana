"""utils sub-package – I/O and plotting."""
from .io_h5   import (get_frame_indices, make_chunks, read_chunk,
                       process_frames_mt,
                       save_calibration_h5, load_calibration_h5,
                       save_calibration_npy, load_calibration_npy,
                       save_events_h5, load_events_h5)

from .io_raw  import (get_frame_indices  as raw_get_frame_indices,
                       make_chunks        as raw_make_chunks,
                       read_chunk         as raw_read_chunk,
                       process_frames_mt  as raw_process_frames_mt,
                       detect_raw_geometry,
                       get_io_module)
                    
from .plotting import (plot_offsets, plot_noise, plot_cm_map,
                        plot_asic_overview, plot_summary_dashboard,
                        plot_hitmap, plot_asic_hitmaps,
                        plot_spectrum, plot_asic_spectra, 
                        plot_raw_spectrum, plot_raw_spectrum_per_asic,
                        plot_grade_distribution,plot_spectrum_comparison
                       )
