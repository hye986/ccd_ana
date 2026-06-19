"""
test_quadruple_recognition.py
==============================
Inject synthetic 2×2 clusters into a clean frame and verify that
find_events() correctly classifies them as quadruple grades (9-12)
and recovers the correct total ADU sum.

Run with:
    python test_quadruple_recognition.py
"""

import numpy as np
import sys
from pathlib import Path

# ── make sure the package is importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnccd_ana.lib.pattern_recognition import (
    find_events, GRADE_OTHER, GRADE_NAMES, _GRADE_DEFS,
    local_max_5x5, _OFFSET_TO_BIT, build_c_grade_table,
)

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

FRAME_H, FRAME_W = 64, 64          # small synthetic frame
NOISE_LEVEL      = 8.0             # ADU RMS  (typical pnCCD)
SEED_SIGMA       = 5.0
SPLIT_SIGMA      = 3.0
TOTAL_ADU        = 1586.0          # Mn Kα in ADU

SEED_THR  = SEED_SIGMA  * NOISE_LEVEL   #  40 ADU
SPLIT_THR = SPLIT_SIGMA * NOISE_LEVEL   #  24 ADU


def make_clean_frame() -> tuple[np.ndarray, np.ndarray]:
    """Zero frame + uniform noise map."""
    frame = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    noise = np.full((FRAME_H, FRAME_W), NOISE_LEVEL, dtype=np.float32)
    return frame, noise


def inject_2x2(frame: np.ndarray,
               cy: int, cx: int,
               fractions: tuple[float, float, float, float],
               total: float = TOTAL_ADU) -> None:
    """
    Place a 2×2 cluster centred between (cy,cx),(cy,cx+1),(cy+1,cx),(cy+1,cx+1).

    fractions = (f_BL, f_BR, f_TL, f_TR)  must sum to 1.0
      BL = bottom-left  = (cy,   cx  )
      BR = bottom-right = (cy,   cx+1)
      TL = top-left     = (cy+1, cx  )
      TR = top-right    = (cy+1, cx+1)
    """
    f_BL, f_BR, f_TL, f_TR = fractions
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1"
    frame[cy,   cx  ] = total * f_BL
    frame[cy,   cx+1] = total * f_BR
    frame[cy+1, cx  ] = total * f_TL
    frame[cy+1, cx+1] = total * f_TR


def run_recognition(frame: np.ndarray,
                    noise: np.ndarray,
                    seed_sigma:  float = SEED_SIGMA,
                    split_sigma: float = SPLIT_SIGMA,
                    label: str = "") -> None:
    """Run find_events and print a one-line summary."""
    evts = find_events(frame, noise,
                       seed_sigma=seed_sigma,
                       split_sigma=split_sigma,
                       reject_extra=False)
    if len(evts) == 0:
        print(f"  {label:40s}  → NO EVENTS FOUND  *** BUG ***")
        return

    for ev in evts:
        g    = int(ev["grade"])
        name = GRADE_NAMES.get(g, "?")
        ok   = "✓" if g in (9, 10, 11, 12) else "✗ WRONG"
        adu_ok = "✓" if abs(ev["adu_sum"] - TOTAL_ADU) < 1.0 else f"✗ expected {TOTAL_ADU:.0f}"
        print(f"  {label:40s}  grade={g:2d}({name:10s}) {ok}  "
              f"adu_sum={ev['adu_sum']:7.1f} {adu_ok}  "
              f"seed=({int(ev['Y'])},{int(ev['X'])}) "
              f"adu_seed={ev['adu_seed']:7.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — perfectly equal split (all four pixels identical)
# ─────────────────────────────────────────────────────────────────────────────

def test_equal_split():
    print("\n── Test 1: equal split (each pixel = total/4) ──────────────────")
    frame, noise = make_clean_frame()
    cy, cx = 20, 20
    inject_2x2(frame, cy, cx, (0.25, 0.25, 0.25, 0.25))
    print(f"  Injected 2×2 at BL=({cy},{cx}):  all pixels = {TOTAL_ADU/4:.1f} ADU")
    print(f"  seed_thr={SEED_THR:.1f}  split_thr={SPLIT_THR:.1f}")
    run_recognition(frame, noise, label="equal split")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — asymmetric splits: sweep which pixel is largest
# ─────────────────────────────────────────────────────────────────────────────

def test_asymmetric_splits():
    print("\n── Test 2: asymmetric splits — seed at each corner ─────────────")
    # For each corner, make it dominant so it becomes the seed
    cases = [
        ("BL dominant → grade 9",  (0.55, 0.20, 0.15, 0.10)),
        ("BR dominant → grade 10", (0.20, 0.55, 0.10, 0.15)),
        ("TL dominant → grade 12", (0.15, 0.10, 0.55, 0.20)),
        ("TR dominant → grade 11", (0.10, 0.15, 0.20, 0.55)),
    ]
    cy, cx = 20, 20
    for label, fracs in cases:
        frame, noise = make_clean_frame()
        inject_2x2(frame, cy, cx, fracs)
        run_recognition(frame, noise, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — sweep split_sigma to find where diagonal drops below threshold
# ─────────────────────────────────────────────────────────────────────────────

def test_split_sigma_sweep():
    print("\n── Test 3: split_sigma sweep with diagonal-weak cluster ─────────")
    # Asymmetric: diagonal pixel (TR) gets only 5% of charge
    # At high split_sigma, TR drops below threshold → misclassified as triple
    fracs = (0.60, 0.20, 0.15, 0.05)   # BL=952, BR=317, TL=238, TR=79
    cy, cx = 20, 20

    print(f"  Fractions: BL={fracs[0]:.0%} BR={fracs[1]:.0%} "
          f"TL={fracs[2]:.0%} TR={fracs[3]:.0%}")
    print(f"  ADU:       BL={TOTAL_ADU*fracs[0]:.0f} BR={TOTAL_ADU*fracs[1]:.0f} "
          f"TL={TOTAL_ADU*fracs[2]:.0f} TR={TOTAL_ADU*fracs[3]:.0f}")
    print(f"  noise={NOISE_LEVEL:.1f} ADU")

    for ss in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        frame, noise = make_clean_frame()
        inject_2x2(frame, cy, cx, fracs)
        thr = ss * NOISE_LEVEL
        tr_adu = TOTAL_ADU * fracs[3]
        above = tr_adu > thr
        run_recognition(frame, noise,
                        split_sigma=ss,
                        label=f"split_sigma={ss:.1f} thr={thr:.1f} TR={'above' if above else 'BELOW'}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — manual bitmask inspection
# Directly print what bitmask the seed sees for each corner case
# ─────────────────────────────────────────────────────────────────────────────

def test_bitmask_inspection():
    print("\n── Test 4: manual bitmask inspection ───────────────────────────")
    grade_table = build_c_grade_table()

    cases = [
        ("BL seed → expect grade 9",
         (0.55, 0.20, 0.15, 0.10), (0, 0)),   # seed offset from BL=(cy,cx)
        ("BR seed → expect grade 10",
         (0.20, 0.55, 0.10, 0.15), (0, +1)),
        ("TL seed → expect grade 12",
         (0.15, 0.10, 0.55, 0.20), (+1, 0)),
        ("TR seed → expect grade 11",
         (0.10, 0.15, 0.20, 0.55), (+1, +1)),
    ]

    cy, cx = 20, 20
    split_sigma = SPLIT_SIGMA

    for label, fracs, (seed_dy, seed_dx) in cases:
        frame, noise = make_clean_frame()
        inject_2x2(frame, cy, cx, fracs)

        sy, sx = cy + seed_dy, cx + seed_dx   # expected seed position

        # Build bitmask manually
        nbr = 0
        neighbour_info = []
        for (dY, dX), bit in _OFFSET_TO_BIT.items():
            ny, nx = int(sy) + dY, int(sx) + dX
            if 0 <= ny < FRAME_H and 0 <= nx < FRAME_W:
                val  = float(frame[ny, nx])
                thr  = split_sigma * float(noise[ny, nx])
                above = val > thr
                if above:
                    nbr |= (1 << bit)
                neighbour_info.append((dY, dX, val, thr, above))

        grade = grade_table[nbr]
        name  = GRADE_NAMES.get(grade, "?")

        print(f"\n  {label}")
        print(f"    seed=({int(sy)},{int(sx)})  "
              f"seed_val={frame[int(sy),int(sx)]:.1f}  "
              f"bitmask=0x{nbr:02X}={nbr:08b}  → grade {grade} ({name})")
        print(f"    Neighbours above split_thr={SPLIT_THR:.1f}:")
        for dY, dX, val, thr, above in sorted(neighbour_info, key=lambda x: (x[0],x[1])):
            mark = "✓ ABOVE" if above else "  below"
            print(f"      ({dY:+d},{dX:+d})  val={val:7.1f}  thr={thr:.1f}  {mark}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — what happens when an OUTSIDE pixel (not in the 2×2)
#           accidentally exceeds split threshold due to noise
# ─────────────────────────────────────────────────────────────────────────────

def test_spurious_neighbour():
    print("\n── Test 5: spurious neighbour above split threshold ─────────────")
    cy, cx = 20, 20
    fracs = (0.55, 0.20, 0.15, 0.10)   # BL dominant → expect grade 9

    # Inject a noise spike at various positions around the 2×2
    # The spike is 3.5σ = just above split_threshold
    spike_adu = 3.5 * NOISE_LEVEL

    # Positions relative to seed (BL = cy,cx):
    spike_positions = [
        ("left of seed  (0,-1)",   0, -1),
        ("below seed   (-1, 0)",  -1,  0),
        ("below-left  (-1,-1)",   -1, -1),
        ("below-right (-1,+1)",   -1, +1),
        ("up-left     (+1,-1)",   +1, -1),
    ]

    for spike_label, sdy, sdx in spike_positions:
        frame, noise = make_clean_frame()
        inject_2x2(frame, cy, cx, fracs)
        # Add spike outside the 2×2
        frame[cy + sdy, cx + sdx] = spike_adu
        run_recognition(frame, noise,
                        label=f"spike at {spike_label}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — replicate real data statistics
# Many events, check grade distribution and adu_sum peaks
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_statistics():
    print("\n── Test 6: bulk statistics (1000 random 2×2 clusters) ──────────")
    rng = np.random.default_rng(42)

    n_events     = 1000
    grade_counts = {}
    adu_sums     = []

    for _ in range(n_events):
        frame, noise = make_clean_frame()

        # Random interaction point within one pixel → random charge fractions
        # Model: 2D Gaussian charge sharing
        # px, py = sub-pixel position within bottom-left pixel (0..1)
        px = rng.uniform(0, 1)
        py = rng.uniform(0, 1)
        # Simple bilinear charge sharing
        f_BL = (1-px) * (1-py)
        f_BR =    px  * (1-py)
        f_TL = (1-px) *    py
        f_TR =    px  *    py
        fracs = (f_BL, f_BR, f_TL, f_TR)

        cy, cx = 20, 20
        inject_2x2(frame, cy, cx, fracs)

        evts = find_events(frame, noise,
                           seed_sigma=SEED_SIGMA,
                           split_sigma=SPLIT_SIGMA,
                           reject_extra=False)
        if len(evts) == 0:
            grade_counts["NO_EVENT"] = grade_counts.get("NO_EVENT", 0) + 1
            continue

        for ev in evts:
            g = int(ev["grade"])
            grade_counts[g] = grade_counts.get(g, 0) + 1
            adu_sums.append(float(ev["adu_sum"]))

    print(f"  Grade distribution over {n_events} random 2×2 clusters:")
    total = sum(v for k,v in grade_counts.items() if k != "NO_EVENT")
    for g, cnt in sorted(grade_counts.items()):
        name = GRADE_NAMES.get(g, str(g)) if g != "NO_EVENT" else "NO_EVENT"
        bar  = "█" * (cnt * 40 // n_events)
        ok   = "✓ quad" if g in (9,10,11,12) else ("✗ WRONG" if g != "NO_EVENT" else "✗ MISSING")
        print(f"    grade {str(g):>8s} ({name:10s}): {cnt:5d}  {bar}  {ok}")

    if adu_sums:
        arr = np.array(adu_sums)
        print(f"\n  adu_sum statistics over recovered events:")
        print(f"    mean   = {arr.mean():.1f}  (expected {TOTAL_ADU:.1f})")
        print(f"    median = {np.median(arr):.1f}")
        print(f"    std    = {arr.std():.1f}")
        print(f"    min    = {arr.min():.1f}")
        print(f"    max    = {arr.max():.1f}")
        print(f"    within 5% of {TOTAL_ADU:.0f}: "
              f"{(np.abs(arr - TOTAL_ADU) < 0.05*TOTAL_ADU).sum()} / {len(arr)}")

    # Now add realistic noise and repeat
    print(f"\n  Repeating with added Gaussian noise (σ={NOISE_LEVEL:.1f} ADU):")
    grade_counts2 = {}
    adu_sums2     = []

    for _ in range(n_events):
        frame, noise = make_clean_frame()
        frame += rng.normal(0, NOISE_LEVEL, size=frame.shape).astype(np.float32)

        px = rng.uniform(0, 1)
        py = rng.uniform(0, 1)
        fracs = ((1-px)*(1-py), px*(1-py), (1-px)*py, px*py)

        cy, cx = 20, 20
        inject_2x2(frame, cy, cx, fracs)

        evts = find_events(frame, noise,
                           seed_sigma=SEED_SIGMA,
                           split_sigma=SPLIT_SIGMA,
                           reject_extra=False)
        if len(evts) == 0:
            grade_counts2["NO_EVENT"] = grade_counts2.get("NO_EVENT", 0) + 1
            continue
        for ev in evts:
            g = int(ev["grade"])
            grade_counts2[g] = grade_counts2.get(g, 0) + 1
            adu_sums2.append(float(ev["adu_sum"]))

    print(f"  Grade distribution with noise:")
    for g, cnt in sorted(grade_counts2.items()):
        name = GRADE_NAMES.get(g, str(g)) if g != "NO_EVENT" else "NO_EVENT"
        bar  = "█" * (cnt * 40 // n_events)
        ok   = "✓ quad" if g in (9,10,11,12) else ("✗ WRONG" if g != "NO_EVENT" else "✗ MISSING")
        print(f"    grade {str(g):>8s} ({name:10s}): {cnt:5d}  {bar}  {ok}")

    if adu_sums2:
        arr2 = np.array(adu_sums2)
        print(f"\n  adu_sum statistics (with noise):")
        print(f"    mean   = {arr2.mean():.1f}  (expected {TOTAL_ADU:.1f})")
        print(f"    std    = {arr2.std():.1f}")
        print(f"    within 5%: {(np.abs(arr2 - TOTAL_ADU) < 0.05*TOTAL_ADU).sum()} / {len(arr2)}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — probe the edge case: interaction exactly at corner (px=py=0.5)
# All 4 pixels equal → tie-breaker selects BL → must be grade 9
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_corner():
    print("\n── Test 7: exact corner (all 4 pixels equal) ───────────────────")
    frame, noise = make_clean_frame()
    cy, cx = 20, 20
    inject_2x2(frame, cy, cx, (0.25, 0.25, 0.25, 0.25))

    # Manual check: what does local_max_5x5 return?
    lmax = local_max_5x5(frame)
    adu4 = TOTAL_ADU / 4

    print(f"  Each pixel = {adu4:.1f} ADU")
    print(f"  local_max at BL ({cy},{cx})   = {lmax[cy,cx]:.1f}")
    print(f"  local_max at BR ({cy},{cx+1}) = {lmax[cy,cx+1]:.1f}")
    print(f"  local_max at TL ({cy+1},{cx}) = {lmax[cy+1,cx]:.1f}")
    print(f"  local_max at TR ({cy+1},{cx+1}) = {lmax[cy+1,cx+1]:.1f}")
    print(f"  (all should equal {adu4:.1f})")

    # Tie-breaker: BL=(cy,cx) should win (smallest y, smallest x)
    print(f"\n  Tie-breaker: BL=({cy},{cx}) should be the unique seed")
    print(f"  (it has no pixel with same value at smaller (y,x) in 5×5 window)")

    run_recognition(frame, noise, label="exact equal split")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — check corner cases near frame edges and ASIC boundaries
# ─────────────────────────────────────────────────────────────────────────────

def test_asic_boundary():
    print("\n── Test 8: cluster straddling column boundary (X=31/32) ─────────")
    # Simulate an ASIC boundary at X=32 (half of 64-wide frame)
    # Both pixels in the 2×2 are in different "ASICs"
    frame, noise = make_clean_frame()
    cy = 20
    cx = 31   # BL at col 31, BR at col 32 → straddles boundary

    inject_2x2(frame, cy, cx, (0.40, 0.35, 0.15, 0.10))
    print(f"  2×2 cluster: BL=({cy},{cx}) BR=({cy},{cx+1}) "
          f"TL=({cy+1},{cx}) TR=({cy+1},{cx+1})")
    print(f"  ADU: BL={TOTAL_ADU*0.40:.0f} BR={TOTAL_ADU*0.35:.0f} "
          f"TL={TOTAL_ADU*0.15:.0f} TR={TOTAL_ADU*0.10:.0f}")
    run_recognition(frame, noise, label="ASIC boundary straddle")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Quadruple event recognition diagnostic")
    print(f"  TOTAL_ADU={TOTAL_ADU:.0f}  noise={NOISE_LEVEL:.1f}  "
          f"seed_sigma={SEED_SIGMA}  split_sigma={SPLIT_SIGMA}")
    print("=" * 70)

    test_equal_split()
    test_asymmetric_splits()
    test_split_sigma_sweep()
    test_bitmask_inspection()
    test_spurious_neighbour()
    test_bulk_statistics()
    test_exact_corner()
    test_asic_boundary()

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)
