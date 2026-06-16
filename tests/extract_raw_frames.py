#!/usr/bin/env python3

'''
Usage for your normal 1024-line frames:
python extract_raw_frames.py big.raw small_2000.raw \
    --start-frame 0 \
    --n-frames 2000 \
    --frame-recs 1024 \
    --n-columns 512
For 512-line frames:
python extract_raw_frames.py big.raw small_2000.raw \
    --start-frame 0 \
    --n-frames 2000 \
    --frame-recs 512 \
    --n-columns 512
To take 2000 frames starting from frame 50000:
python extract_raw_frames.py big.raw small_2000_from_50000.raw \
    --start-frame 50000 \
    --n-frames 2000 \
    --frame-recs 1024 \
    --n-columns 512
'''

import argparse
import os
import sys


RAW_HEADER = b"16BU0000"


def copy_bytes(fin, fout, nbytes, chunk_bytes):
    remaining = nbytes

    while remaining > 0:
        n = min(chunk_bytes, remaining)
        buf = fin.read(n)

        if not buf:
            raise IOError("Unexpected end of file while copying")

        fout.write(buf)
        remaining -= len(buf)


def extract_frames(
    input_path,
    output_path,
    start_frame=0,
    n_frames=2000,
    frame_recs=1024,
    n_columns=512,
    chunk_mb=256,
    overwrite=False,
):
    header_len = len(RAW_HEADER)

    record_bytes = 2 + 4 + n_columns * 2
    frame_bytes = frame_recs * record_bytes

    size = os.path.getsize(input_path)

    if size < header_len:
        raise ValueError(f"{input_path}: file too small")

    data_bytes = size - header_len
    total_full_frames = data_bytes // frame_bytes
    trailing = data_bytes % frame_bytes

    if total_full_frames == 0:
        raise ValueError(
            f"{input_path}: no complete frames found. "
            f"Check --frame-recs and --n-columns."
        )

    if trailing:
        print(
            f"Warning: input has {trailing} trailing bytes after "
            f"{total_full_frames} complete frames. Ignoring trailing bytes.",
            file=sys.stderr,
        )

    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")

    if n_frames <= 0:
        raise ValueError("--n-frames must be > 0")

    if start_frame >= total_full_frames:
        raise ValueError(
            f"start_frame {start_frame} is outside file. "
            f"Total full frames = {total_full_frames}"
        )

    available = total_full_frames - start_frame

    if n_frames > available:
        print(
            f"Warning: requested {n_frames} frames, but only {available} "
            f"available from start_frame={start_frame}. Copying {available}.",
            file=sys.stderr,
        )
        n_frames = available

    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Use --overwrite to replace it."
            )
        os.remove(output_path)

    copy_start_offset = header_len + start_frame * frame_bytes
    copy_nbytes = n_frames * frame_bytes
    chunk_bytes = max(1, chunk_mb) * 1024 * 1024

    print("Input file:")
    print(f"  {input_path}")
    print("Output file:")
    print(f"  {output_path}")
    print("Layout:")
    print(f"  header bytes      : {header_len}")
    print(f"  records/frame     : {frame_recs}")
    print(f"  columns/record    : {n_columns}")
    print(f"  record bytes      : {record_bytes}")
    print(f"  frame bytes       : {frame_bytes}")
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
                f"{input_path}: bad header {hdr!r}, expected {RAW_HEADER!r}"
            )

        # Write original header
        fout.write(hdr)

        # Seek to requested frame range
        fin.seek(copy_start_offset)

        # Copy complete frame bytes
        copy_bytes(fin, fout, copy_nbytes, chunk_bytes)

    out_size = os.path.getsize(output_path)

    expected_size = header_len + copy_nbytes
    if out_size != expected_size:
        raise IOError(
            f"Output size mismatch: got {out_size}, expected {expected_size}"
        )

    print("Done.")
    print(f"Output size: {out_size / 1024 / 1024:.2f} MiB")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a small contiguous frame range from a large .raw file. "
            "The output file keeps the same header and complete frame structure."
        )
    )

    parser.add_argument("input", help="Input large .raw file")
    parser.add_argument("output", help="Output small .raw file")

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to copy. Default: 0.",
    )

    parser.add_argument(
        "--n-frames",
        type=int,
        default=2000,
        help="Number of frames to copy. Default: 2000.",
    )

    parser.add_argument(
        "--frame-recs",
        type=int,
        default=1024,
        help=(
            "Number of records/lines per frame. "
            "Use 1024 or 512 depending on your file. Default: 1024."
        ),
    )

    parser.add_argument(
        "--n-columns",
        type=int,
        default=512,
        help="Number of uint16 pixels per record/line. Default: 512.",
    )

    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=256,
        help="Copy buffer size in MiB. Default: 256.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    args = parser.parse_args()

    extract_frames(
        input_path=args.input,
        output_path=args.output,
        start_frame=args.start_frame,
        n_frames=args.n_frames,
        frame_recs=args.frame_recs,
        n_columns=args.n_columns,
        chunk_mb=args.chunk_mb,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
