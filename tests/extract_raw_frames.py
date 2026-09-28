#!/usr/bin/env python3
"""
Extract a smaller contiguous frame range from a 512×512 RAW dark run.

Usage
-----
  python extract_raw_frames.py big.raw small_2000.raw \
      --start-frame 0 --n-frames 2000

  # take 2000 frames starting from frame 50000
  python extract_raw_frames.py big.raw small_from_50000.raw \
      --start-frame 50000 --n-frames 2000

The output keeps the same 8-byte header and complete 512×512 frame
structure as the input.
"""

import argparse
import os
import sys

RAW_HEADER   = b"16BU0000"
FRAME_RECS   = 512      # rows per frame
N_COLUMNS    = 512      # pixels per row
RECORD_BYTES = 2 + 4 + N_COLUMNS * 2   # marker + ADC counter + pixels
FRAME_BYTES  = FRAME_RECS * RECORD_BYTES


def copy_bytes(fin, fout, nbytes, chunk_bytes):
    remaining = nbytes
    while remaining > 0:
        n = min(chunk_bytes, remaining)
        buf = fin.read(n)
        if not buf:
            raise IOError("Unexpected end of file while copying")
        fout.write(buf)
        remaining -= len(buf)


def extract_frames(input_path, output_path, start_frame=0, n_frames=2000,
                   chunk_mb=256, overwrite=False):
    header_len = len(RAW_HEADER)

    size = os.path.getsize(input_path)
    if size < header_len:
        raise ValueError(f"{input_path}: file too small")

    data_bytes = size - header_len
    total_full_frames = data_bytes // FRAME_BYTES
    trailing = data_bytes % FRAME_BYTES

    if total_full_frames == 0:
        raise ValueError(f"{input_path}: no complete 512×512 frames found.")

    if trailing:
        print(f"Warning: input has {trailing} trailing bytes after "
              f"{total_full_frames} complete frames. Ignoring trailing bytes.",
              file=sys.stderr)

    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if n_frames <= 0:
        raise ValueError("--n-frames must be > 0")
    if start_frame >= total_full_frames:
        raise ValueError(
            f"start_frame {start_frame} is outside file. "
            f"Total full frames = {total_full_frames}")

    available = total_full_frames - start_frame
    if n_frames > available:
        print(f"Warning: requested {n_frames} frames, but only {available} "
              f"available from start_frame={start_frame}. Copying {available}.",
              file=sys.stderr)
        n_frames = available

    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Use --overwrite to replace it.")
        os.remove(output_path)

    copy_start_offset = header_len + start_frame * FRAME_BYTES
    copy_nbytes = n_frames * FRAME_BYTES
    chunk_bytes = max(1, chunk_mb) * 1024 * 1024

    print("Input file:")
    print(f"  {input_path}")
    print("Output file:")
    print(f"  {output_path}")
    print("Layout:")
    print(f"  records/frame     : {FRAME_RECS}")
    print(f"  columns/record    : {N_COLUMNS}")
    print(f"  record bytes      : {RECORD_BYTES}")
    print(f"  frame bytes       : {FRAME_BYTES}")
    print(f"  total full frames : {total_full_frames}")
    print("Extraction:")
    print(f"  start frame       : {start_frame}")
    print(f"  frames copied     : {n_frames}")
    print(f"  output data size  : {copy_nbytes / 1024 / 1024:.2f} MiB")

    with open(input_path, "rb", buffering=1024 * 1024) as fin, \
         open(output_path, "wb", buffering=1024 * 1024) as fout:
        hdr = fin.read(header_len)
        if hdr != RAW_HEADER:
            raise ValueError(
                f"{input_path}: bad header {hdr!r}, expected {RAW_HEADER!r}")
        fout.write(hdr)
        fin.seek(copy_start_offset)
        copy_bytes(fin, fout, copy_nbytes, chunk_bytes)

    out_size = os.path.getsize(output_path)
    expected_size = header_len + copy_nbytes
    if out_size != expected_size:
        raise IOError(
            f"Output size mismatch: got {out_size}, expected {expected_size}")

    print("Done.")
    print(f"Output size: {out_size / 1024 / 1024:.2f} MiB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a contiguous frame range from a 512×512 RAW file.")
    parser.add_argument("input", help="Input large .raw file")
    parser.add_argument("output", help="Output small .raw file")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First frame index to copy. Default: 0.")
    parser.add_argument("--n-frames", type=int, default=2000,
                        help="Number of frames to copy. Default: 2000.")
    parser.add_argument("--chunk-mb", type=int, default=256,
                        help="Copy buffer size in MiB. Default: 256.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output file if it already exists.")
    args = parser.parse_args()

    extract_frames(input_path=args.input, output_path=args.output,
                   start_frame=args.start_frame, n_frames=args.n_frames,
                   chunk_mb=args.chunk_mb, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
