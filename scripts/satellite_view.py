"""Build the satellite view of the Hadamard maxdet landscape.

Outputs:

* ``results/satellite_view_n{19,21,23}_partitions.json`` — full Ehlich
  partition tables per n (predicted |det|, predicted ratio, complexity).
* ``results/satellite_view_h23_overlay.json`` — per-basin best-fit and
  per-partition reached/unreached status for H23, using all 29 in-repo
  basins.
* ``results/satellite_view_h23.png`` — 2D landscape scatter (complexity
  vs predicted ratio, reached vs unreached).
* Stdout: human-readable summary tables.

Usage::

    python scripts/satellite_view.py
    python scripts/satellite_view.py --quick       # skip overlay (enumeration only)
    python scripts/satellite_view.py --reach 0.85  # custom reach_threshold
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make `src/` importable when running from a fresh shell.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.basin_atlas import load_all_h23_basins
from hadamard_maxdet.orders import BENCHMARKS
from hadamard_maxdet.satellite import (
    cross_n_partition_summary,
    enumerate_all_ehlich_partitions,
    overlay_basins_on_partitions,
    partition_complexity,
    plot_satellite_view,
)
from hadamard_maxdet.verifier import upper_bound


ORRICK_RATIO = 0.7091
BTC_DET = 2_626_567_700_217_856
BTC_RATIO = 0.6702


def write_partition_tables(out_dir: Path) -> dict[int, list[dict]]:
    """Deliverable 1: full Ehlich partitions for n ∈ {19, 21, 23}."""
    tables: dict[int, list[dict]] = {}
    for n in (19, 21, 23):
        parts = enumerate_all_ehlich_partitions(n)
        barba = upper_bound(n)
        records = []
        for p in parts:
            records.append({
                "partition": list(p.sizes),
                "predicted_abs_det": p.predicted_abs_det,
                "predicted_ratio_to_barba": p.predicted_abs_det / barba,
                "complexity": partition_complexity(p.sizes),
                "intra_value": p.intra_value,
                "cross_value": p.cross_value,
            })
        tables[n] = records
        path = out_dir / f"satellite_view_n{n}_partitions.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "n": n,
                    "barba_upper_bound": barba,
                    "n_partitions": len(records),
                    "partitions": records,
                },
                f, indent=2,
            )
        print(f"  wrote {path.name}: {len(records)} partitions")
    return tables


def overlay_h23(
    out_dir: Path, *, reach_threshold: float, n_restarts: int
) -> tuple[list, list]:
    """Deliverable 2: overlay 29 H23 basins on every partition."""
    print("loading H23 basins...")
    basins = load_all_h23_basins()
    print(f"  loaded {len(basins)} distinct H23 basins")
    t0 = time.monotonic()
    per_basin, per_partition = overlay_basins_on_partitions(
        23, basins,
        reach_threshold=reach_threshold,
        n_restarts=n_restarts,
    )
    dt = time.monotonic() - t0
    print(f"  overlay completed in {dt:.1f} s "
          f"({len(basins)}×{len(per_partition)} fits @ {n_restarts} restarts)")
    out = {
        "n": 23,
        "n_basins": len(basins),
        "n_partitions": len(per_partition),
        "reach_threshold": reach_threshold,
        "n_restarts": n_restarts,
        "barba": upper_bound(23),
        "per_basin": [ov.to_json() for ov in per_basin],
        "per_partition": [lp.to_json() for lp in per_partition],
    }
    path = out_dir / "satellite_view_h23_overlay.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path.name}")
    return per_basin, per_partition


def print_h23_overlay_summary(per_basin, per_partition) -> dict:
    """Print headline counts and find the cluster of unreached high-prediction
    partitions. Returns a small summary dict for the writeup."""
    total = len(per_partition)
    above_orrick = [r for r in per_partition if r.predicted_ratio >= ORRICK_RATIO]
    in_band = [
        r for r in per_partition
        if 0.67 <= r.predicted_ratio < ORRICK_RATIO
    ]
    below_btc = [r for r in per_partition if r.predicted_ratio < BTC_RATIO]
    reached_above_orrick = [r for r in above_orrick if r.reached]
    reached_in_band = [r for r in in_band if r.reached]
    reached_below_btc = [r for r in below_btc if r.reached]

    print()
    print("=" * 72)
    print("H23 SATELLITE OVERLAY SUMMARY")
    print("=" * 72)
    print(f"Total partitions:                                   {total}")
    print(f"Partitions with predicted ratio ≥ Orrick (0.7091):  {len(above_orrick)}")
    print(f"  …of which REACHED (basin score ≥ threshold):      {len(reached_above_orrick)}")
    print(f"Partitions with ratio in [0.67, 0.7091):            {len(in_band)}")
    print(f"  …of which REACHED:                                {len(reached_in_band)}")
    print(f"Partitions with ratio < BTC (0.6702):               {len(below_btc)}")
    print(f"  …of which REACHED:                                {len(reached_below_btc)}")
    print()
    print("Top-10 highest-prediction partitions (predicted ratio descending):")
    print(f"  {'partition':<18} {'pred_ratio':>10} {'complexity':>10} "
          f"{'best_score':>10}  reached  best_basin")
    for r in per_partition[:10]:
        print(f"  {str(r.partition):<18} {r.predicted_ratio:>10.4f} "
              f"{r.complexity:>10.2f} {r.best_score:>10.4f}  "
              f"{'YES' if r.reached else ' no':<7}  {r.best_basin_label}")
    # Best-fit per basin.
    print()
    print("Per-basin best-fit partition (top 10 basins by |det|):")
    sorted_basins = sorted(per_basin, key=lambda b: -b.basin_abs_det)
    print(f"  {'basin':<28} {'|det|':>14} {'best_part':<18} "
          f"{'score':>7}  pred_gap")
    for ov in sorted_basins[:10]:
        gap = ov.gap
        print(f"  {ov.basin_label[:28]:<28} {ov.basin_abs_det:>14.3e} "
              f"{str(ov.best_partition):<18} {ov.best_score:>7.4f}  "
              f"{gap:+.3e}")

    # Top recommendation: highest-prediction unreached partition with
    # the *closest* best_score (i.e. the partition that is most "almost
    # reached" relative to its predicted height).
    above_orrick_unreached = [r for r in above_orrick if not r.reached]
    if above_orrick_unreached:
        # Recommend the one with highest predicted_ratio (greatest headroom).
        best_target = above_orrick_unreached[0]
        # Also report the closest-to-reach one.
        closest = max(above_orrick_unreached, key=lambda r: r.best_score)
        print()
        print("RECOMMENDED CONSTRUCTION TARGETS (above-Orrick, unreached):")
        print(f"  Highest prediction: {best_target.partition} "
              f"(ratio {best_target.predicted_ratio:.4f}, "
              f"|det| {best_target.predicted_abs_det:.3e}, "
              f"complexity {best_target.complexity:.2f}, "
              f"current best fit {best_target.best_score:.4f})")
        print(f"  Closest to reach:   {closest.partition} "
              f"(ratio {closest.predicted_ratio:.4f}, "
              f"|det| {closest.predicted_abs_det:.3e}, "
              f"current best fit {closest.best_score:.4f})")
    else:
        print()
        print("All above-Orrick partitions are reached.")
    return {
        "total": total,
        "above_orrick": len(above_orrick),
        "above_orrick_reached": len(reached_above_orrick),
        "in_band_67_71": len(in_band),
        "in_band_67_71_reached": len(reached_in_band),
        "below_btc": len(below_btc),
        "below_btc_reached": len(reached_below_btc),
    }


def print_cross_n_summary(rows) -> None:
    print()
    print("=" * 72)
    print("CROSS-n SATELLITE SUMMARY (n ∈ {19, 21, 23})")
    print("=" * 72)
    print(f"  {'n':>3} {'#parts':>7} {'Barba':>14} {'proven_max':>14} "
          f"{'>=proven':>9} {'>=in_repo':>10} {'>=0.95Barba':>12}")
    for r in rows:
        proven = f"{r.proven_max:.3e}" if r.proven_max else "  open"
        ap = str(r.n_above_proven) if r.n_above_proven is not None else "n/a"
        ai = str(r.n_above_in_repo) if r.n_above_in_repo is not None else "n/a"
        print(f"  {r.n:>3} {r.n_partitions:>7} {r.barba:>14.3e} {proven:>14} "
              f"{ap:>9} {ai:>10} {r.n_above_barba_95:>12}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Skip the H23 overlay (enumeration + cross-n only)")
    p.add_argument("--reach", type=float, default=0.9,
                   help="Reach threshold (default 0.9)")
    p.add_argument("--restarts", type=int, default=4,
                   help="Greedy assignment restarts per (basin, partition)")
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("Deliverable 1: Full Ehlich partition enumeration")
    print("=" * 72)
    tables = write_partition_tables(out_dir)
    for n, recs in tables.items():
        print(f"  n={n}: top-3 partitions by predicted ratio:")
        for rec in recs[:3]:
            print(f"    {rec['partition']}  "
                  f"ratio={rec['predicted_ratio_to_barba']:.4f}  "
                  f"|det|={rec['predicted_abs_det']:.3e}  "
                  f"complexity={rec['complexity']:.2f}")

    if not args.quick:
        print()
        print("=" * 72)
        print("Deliverable 2: Overlay H23 basins on partitions")
        print("=" * 72)
        per_basin, per_partition = overlay_h23(
            out_dir, reach_threshold=args.reach, n_restarts=args.restarts
        )
        summary = print_h23_overlay_summary(per_basin, per_partition)
        # Deliverable 3: plot.
        print()
        print("=" * 72)
        print("Deliverable 3: 2D landscape plot")
        print("=" * 72)
        plot_path = out_dir / "satellite_view_h23.png"
        ok = plot_satellite_view(
            23, per_partition, str(plot_path),
            orrick_ratio=ORRICK_RATIO, btc_ratio=BTC_RATIO,
        )
        if ok:
            print(f"  wrote {plot_path.name}")
        else:
            print("  matplotlib not available; skipped plot")

        # Persist headline summary alongside overlay.
        summary_path = out_dir / "satellite_view_h23_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "reach_threshold": args.reach,
                "n_restarts": args.restarts,
                "n_basins_used": len(per_basin),
                **summary,
            }, f, indent=2)
        print(f"  wrote {summary_path.name}")

    print()
    print("=" * 72)
    print("Deliverable 4: Cross-n partition summary")
    print("=" * 72)
    rows = cross_n_partition_summary(
        [19, 21, 23], in_repo_bests={23: BTC_DET},
    )
    cross_path = out_dir / "satellite_view_cross_n.json"
    with open(cross_path, "w") as f:
        json.dump([r.to_json() for r in rows], f, indent=2)
    print(f"  wrote {cross_path.name}")
    print_cross_n_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
