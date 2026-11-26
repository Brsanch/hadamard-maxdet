"""Generalized satellite-view sweep over (intra, cross) Gram-value pairs.

Goes beyond the canonical (+3, -1) Ehlich/Wojtas ansatz to find which
(a, b) Gram regime contains the proven H19 / H21 maxima — and identify
plausible regimes for Orrick's H23 record.

Outputs:

* ``results/generalized_satellite.json`` — full table:
  per-n top-10 triples, per-(a,b) top-prediction, calibration vs proven max.
* Stdout — human-readable summary.

Usage::

    python scripts/generalized_satellite.py
    python scripts/generalized_satellite.py --quick   # n=19, 21 only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.generalized_ehlich import (
    DEFAULT_AB_VALUES,
    calibrate_against_proven,
    enumerate_ab_grid,
    hadamard_ceiling,
    top_regimes_above,
)
from hadamard_maxdet.verifier import upper_bound


# Proven max ratios (from docs/findings.md / paper/orders.py)
PROVEN_RATIOS: dict[int, float] = {
    19: 0.9413,
    21: 0.9058,
}

ORRICK_RATIO = 0.7091
BTC_RATIO = 0.6702


def fmt_partition(p):
    return "(" + ",".join(str(x) for x in p) + ")"


def run_calibration_at_n(n: int, proven_ratio: float | None) -> dict:
    """Build the (a, b) calibration table at a given n."""
    grid = enumerate_ab_grid(n)
    rows = calibrate_against_proven(n, proven_ratio)
    barba = float(upper_bound(n))
    hadamard = hadamard_ceiling(n)

    plausible_rows = [r for r in rows if r.plausible]

    print(f"\n=== n={n} | Barba={barba:.4e} | Hadamard={hadamard:.4e} | "
          f"H/B={hadamard/barba:.4f} ===")
    if proven_ratio is not None:
        print(f"  Proven max ratio = {proven_ratio:.4f} (target)")
    print(f"  Total triples enumerated: {len(grid)}")
    print(f"  Distinct (a,b) pairs: {len(rows)}")
    if proven_ratio is not None:
        print(f"  Plausible (a,b) pairs (top_pred ≥ proven): {len(plausible_rows)}")

    print(f"\n  Top 10 (a, b) regimes by predicted ceiling:")
    print(f"  {'flag':>4}  {'a':>3} {'b':>3}  {'top_pred':>9}  "
          f"{'gap':>9}  partition")
    for r in rows[:10]:
        flag = " *  " if r.plausible else "    "
        gap_str = f"{r.gap_to_proven:+.4f}" if r.gap_to_proven is not None else "  n/a "
        print(f"  {flag} {r.intra:+3d} {r.cross:+3d}  {r.top_predicted_ratio:9.4f}  "
              f"{gap_str:>9}  {fmt_partition(r.top_partition)}")

    if proven_ratio is not None and plausible_rows:
        # Identify the (a, b) pair whose top prediction is CLOSEST (from above)
        # to the proven max — most likely to be the actual regime.
        closest = min(plausible_rows, key=lambda r: r.gap_to_proven)
        print(f"\n  Closest-from-above match: (a={closest.intra:+d}, b={closest.cross:+d}), "
              f"top_pred={closest.top_predicted_ratio:.4f}, "
              f"gap={closest.gap_to_proven:+.4f}, "
              f"partition={fmt_partition(closest.top_partition)}")

    return {
        "n": n,
        "barba": barba,
        "hadamard_ceiling": hadamard,
        "proven_ratio": proven_ratio,
        "n_triples_enumerated": len(grid),
        "n_ab_pairs": len(rows),
        "n_plausible_ab_pairs": len(plausible_rows),
        "ab_calibration_rows": [r.to_json() for r in rows],
        "top_10_triples": [g.to_json() for g in grid[:10]],
    }


def run_h23_targeting() -> dict:
    """Deliverable 3: identify (a, b, partition) triples for H23."""
    n = 23
    barba = float(upper_bound(n))
    hadamard = hadamard_ceiling(n)
    print(f"\n=== H23 targeting | Barba={barba:.4e} | Orrick={ORRICK_RATIO} ===")

    above_orrick = top_regimes_above(n, threshold_ratio=ORRICK_RATIO)
    print(f"  (a,b,partition) triples with predicted ratio > Orrick {ORRICK_RATIO}: "
          f"{len(above_orrick)}")

    # Group by (a, b).
    by_ab: dict[tuple[int, int], list] = {}
    for r in above_orrick:
        by_ab.setdefault((r.intra, r.cross), []).append(r)

    print(f"  Distinct (a,b) pairs above Orrick: {len(by_ab)}")
    print(f"\n  Top 10 (a, b, partition) triples by predicted ratio:")
    print(f"  {'a':>3} {'b':>3}  {'partition':<20}  {'pred_ratio':>10}  {'pred_det':>11}")
    for r in above_orrick[:10]:
        print(f"  {r.intra:+3d} {r.cross:+3d}  {fmt_partition(r.partition):<20}  "
              f"{r.predicted_ratio:10.4f}  {r.predicted_abs_det:11.4e}")

    print(f"\n  Top (a, b) pairs by ceiling at n=23 (best partition each):")
    pair_records = []
    for (a, b), triples in by_ab.items():
        triples.sort(key=lambda x: -x.predicted_ratio)
        top = triples[0]
        pair_records.append((top.predicted_ratio, a, b, top.partition))
    pair_records.sort(reverse=True)
    for ratio, a, b, p in pair_records[:10]:
        print(f"  (a={a:+d}, b={b:+d})  best partition {fmt_partition(p):<20}  "
              f"top ratio {ratio:.4f}")

    # Recommendation: the (a, b) pair whose top prediction sits CLOSEST to the
    # n=21-suggested target ratio (~0.91) — i.e. above Orrick by a healthy
    # margin but bounded.
    target_ratio = 0.91  # n=21 cross-n suggestion
    closest_to_target = min(
        pair_records,
        key=lambda r: abs(r[0] - target_ratio) if r[0] >= ORRICK_RATIO else 1e9
    )
    print(f"\n  Closest-to-{target_ratio} (n=21-suggested) (a,b): "
          f"(a={closest_to_target[1]:+d}, b={closest_to_target[2]:+d}), "
          f"partition {fmt_partition(closest_to_target[3])}, "
          f"ratio {closest_to_target[0]:.4f}")

    return {
        "n": 23,
        "barba": barba,
        "hadamard_ceiling": hadamard,
        "orrick_ratio": ORRICK_RATIO,
        "n_triples_above_orrick": len(above_orrick),
        "n_ab_pairs_above_orrick": len(by_ab),
        "top_10_triples_above_orrick": [r.to_json() for r in above_orrick[:10]],
        "ab_pairs_above_orrick": [
            {
                "intra": a,
                "cross": b,
                "top_partition": list(p),
                "top_predicted_ratio": ratio,
            }
            for ratio, a, b, p in pair_records
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip H23 sweep (n=19, 21 only).",
    )
    parser.add_argument(
        "--out-dir", default="results",
        help="Output directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Generalized satellite view: (intra, cross) Gram-value sweep")
    print("=" * 72)
    print(f"AB grid: {tuple(DEFAULT_AB_VALUES)}")
    print(f"Distinct (a,b) pairs (intra ≠ cross): "
          f"{len(DEFAULT_AB_VALUES) * (len(DEFAULT_AB_VALUES) - 1)}")

    payload: dict = {
        "ab_values": list(DEFAULT_AB_VALUES),
        "orrick_ratio": ORRICK_RATIO,
        "btc_ratio": BTC_RATIO,
        "calibration": {},
        "h23_targeting": None,
    }

    for n, ratio in PROVEN_RATIOS.items():
        payload["calibration"][str(n)] = run_calibration_at_n(n, ratio)

    if not args.quick:
        payload["h23_targeting"] = run_h23_targeting()

    out_path = out_dir / "generalized_satellite.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
