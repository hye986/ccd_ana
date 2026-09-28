"""
pnccd_ana.io.calibration
========================
Save / load offset-calibration arrays as .npy files.

Written by the offset stage into the output directory::

    offset_median.npy        (H, W) float32   — median offset
    offset_sigclip.npy       (H, W) float32   — sigma-clip offset
    noise.npy                (H, W) float32   — per-pixel RMS
    n_clipped_map.npy        (H, W) float32   — frames clipped per pixel
    cm_map.npy               (N, H, n_asics) float32 — CM correction values
    cm_noise_<ASIC>.npy      (H,) float32     — per-ASIC CM noise
    bad_pixel_mask.npy       (H, W) bool      — True = bad pixel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_calibration_npy(
        out_dir:        str | Path,
        results:        dict,
        cm_map:         np.ndarray | None = None,
        bad_pixel_mask: np.ndarray | None = None,
) -> None:
    """
    Save offset-calibration arrays as .npy files.

    Parameters
    ----------
    out_dir        : output directory
    results        : results dict ("global" → per-pixel maps)
    cm_map         : (N, H, n_asics) CM correction values, or None
    bad_pixel_mask : (H, W) bool bad pixel mask, or None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving .npy calibration to {out_dir}/")

    r = results.get("global", results)
    mapping = {
        "offset_median":  "offset_median.npy",
        "offset_sigclip": "offset_sigclip.npy",
        "noise":          "noise.npy",
        "n_clipped_map":  "n_clipped_map.npy",
    }
    for key, fname in mapping.items():
        if key in r and r[key] is not None:
            np.save(out_dir / fname, np.asarray(r[key], dtype=np.float32))
            print(f"  → {fname}")

    if cm_map is not None:
        np.save(out_dir / "cm_map.npy", np.asarray(cm_map, dtype=np.float32))
        print("  → cm_map.npy")

    cm_noise = r.get("cm_noise")
    if isinstance(cm_noise, dict):
        for name, arr in cm_noise.items():
            np.save(out_dir / f"cm_noise_{name}.npy",
                    np.asarray(arr, dtype=np.float32))
            print(f"  → cm_noise_{name}.npy")
    elif cm_noise is not None:
        np.save(out_dir / "cm_noise.npy", np.asarray(cm_noise, dtype=np.float32))
        print("  → cm_noise.npy")

    if bad_pixel_mask is not None:
        np.save(out_dir / "bad_pixel_mask.npy",
                np.asarray(bad_pixel_mask, dtype=bool))
        print("  → bad_pixel_mask.npy")


def load_calibration_npy(
        npy_dir: str | Path,
        prefer:  str = "sigclip",
) -> dict:
    """
    Load offset calibration from .npy files written by save_calibration_npy.

    Parameters
    ----------
    npy_dir : directory containing the .npy files
    prefer  : which offset method to load as the primary "offset" key
              ("sigclip" | "median")

    Returns
    -------
    dict with keys: offset, offset_type, noise, n_clipped_map (optional),
    cm_map (optional), cm_noise (optional; dict for per-ASIC),
    bad_pixel_mask (optional).
    """
    npy_dir = Path(npy_dir)
    alt = "median" if prefer == "sigclip" else "sigclip"
    d: dict = {}

    # Offset: preferred → alternative → any
    for name in (prefer, alt):
        fp = npy_dir / f"offset_{name}.npy"
        if fp.exists():
            d["offset"] = np.load(fp).astype(np.float32)
            d["offset_type"] = name
            print(f"  loaded {fp.name}  {d['offset'].shape}")
            break
    else:
        found = sorted(npy_dir.glob("offset_*.npy"))
        if found:
            d["offset"] = np.load(found[0]).astype(np.float32)
            d["offset_type"] = found[0].stem.split("offset_")[1]
            print(f"  Warning: preferred '{prefer}' not found, "
                  f"using '{found[0].name}'")
        else:
            raise RuntimeError(
                f"No offset_*.npy found in {npy_dir}.\n"
                f"Run  pnccd-offset  first.")

    # Required noise map
    fp = npy_dir / "noise.npy"
    if not fp.exists():
        raise RuntimeError(f"{fp.name} not found in {npy_dir}.")
    d["noise"] = np.load(fp).astype(np.float32)
    print(f"  loaded {fp.name}  {d['noise'].shape}")

    # Optional maps
    for key, fname in [("n_clipped_map",  "n_clipped_map.npy"),
                       ("cm_map",         "cm_map.npy"),
                       ("bad_pixel_mask", "bad_pixel_mask.npy")]:
        fp = npy_dir / fname
        if fp.exists():
            d[key] = np.load(fp)
            print(f"  loaded {fp.name}  {d[key].shape}")

    # CM noise: per-ASIC files, else a single file
    per_asic = {p.stem.split("cm_noise_")[1]: np.load(p).astype(np.float32)
                for p in sorted(npy_dir.glob("cm_noise_*.npy"))}
    if per_asic:
        d["cm_noise"] = per_asic
    elif (npy_dir / "cm_noise.npy").exists():
        d["cm_noise"] = np.load(npy_dir / "cm_noise.npy").astype(np.float32)

    return d
