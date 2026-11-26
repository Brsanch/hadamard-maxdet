"""Run all five diagnostic deep-dive routines and write results.

Outputs (under ``results/``):
  diagnostic_basin_map.json
  diagnostic_energy_profile.json
  diagnostic_orbit_size.json
  diagnostic_move_profile.json
  diagnostic_cross_n.json
  diagnostic_report.md

Usage:
  python scripts/diagnostic_deepdive.py --quick
  python scripts/diagnostic_deepdive.py --full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.diagnostics import (  # noqa: E402
    basin_map,
    cross_n_validation,
    energy_profile,
    move_profile,
    orbit_size_estimate,
)
from hadamard_maxdet.orders import BENCHMARKS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="cheap budget for a smoke run")
    ap.add_argument("--full", action="store_true", help="full budget (default)")
    ap.add_argument(
        "--btc-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "H23_rich_ansatz_best.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Skip diagnostics by name: basin energy orbit move cross")
    args = ap.parse_args()

    quick = args.quick or not args.full
    args.out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)

    # Load BTC seed for diagnostic 2 + 3.
    btc = json.load(open(args.btc_path))
    btc_entries = btc["entries"]
    btc_abs_det = btc["abs_det"]
    print(f"Loaded BTC seed: n={btc['n']}, |det|={btc_abs_det:,d}")

    summary: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # -------- Diagnostic 1: basin map at small n --------
    basin_results: dict = {}
    if "basin" not in skip:
        if quick:
            basin_specs = [
                (7, 100, 5_000),
                (9, 100, 5_000),
                (11, 100, 5_000),
            ]
        else:
            basin_specs = [
                (9, 2000, 30_000),
                (11, 1000, 30_000),
                (13, 1000, 30_000),
            ]
        for n, n_seeds, iters in basin_specs:
            print(f"\n[Diagnostic 1] basin_map n={n} seeds={n_seeds} iters={iters}")
            t0 = time.monotonic()
            res = basin_map(n, n_seeds=n_seeds, iters_per_seed=iters)
            print(
                f"  -> {len(res.clusters)} clusters in {time.monotonic()-t0:.1f}s; "
                f"largest {res.clusters[0].cluster_size}/{n_seeds} "
                f"({100*res.clusters[0].cluster_size/n_seeds:.1f}%); "
                f"best |det| {max(c.abs_det for c in res.clusters):,d}"
            )
            basin_results[f"n{n}"] = res.to_json()
        (args.out_dir / "diagnostic_basin_map.json").write_text(
            json.dumps(basin_results, indent=2) + "\n"
        )
        summary["basin_map"] = {
            f"n{n}": {
                "n_clusters": basin_results[f"n{n}"]["n_clusters"],
                "largest_cluster_fraction": basin_results[f"n{n}"]["largest_cluster_fraction"],
                "best_abs_det": basin_results[f"n{n}"]["best_abs_det"],
                "proven_max": basin_results[f"n{n}"]["proven_max"],
                "proven_max_cluster_size": basin_results[f"n{n}"]["proven_max_cluster_size"],
            }
            for n, _, _ in basin_specs
        }

    # -------- Diagnostic 2: energy profile around BTC seed --------
    if "energy" not in skip:
        d2_samples = 500 if quick else 5_000
        print(f"\n[Diagnostic 2] energy_profile around BTC; samples/d={d2_samples}")
        t0 = time.monotonic()
        res2 = energy_profile(
            btc_entries,
            distances=(1, 2, 3),
            n_samples_per_distance=d2_samples,
            seed=1,
        )
        print(f"  -> done in {time.monotonic()-t0:.1f}s")
        for b in res2.bins:
            print(
                f"    d={b.distance}: above={b.n_above} equal={b.n_equal} "
                f"below={b.n_below} singular={b.n_singular} "
                f"max={b.max_abs_det:,d}"
            )
        (args.out_dir / "diagnostic_energy_profile.json").write_text(
            json.dumps(res2.to_json(), indent=2) + "\n"
        )
        summary["energy_profile"] = {
            "seed_abs_det": res2.seed_abs_det,
            "bins": [
                {
                    "d": b.distance,
                    "n_above": b.n_above,
                    "n_equal": b.n_equal,
                    "n_below": b.n_below,
                    "max": b.max_abs_det,
                }
                for b in res2.bins
            ],
        }

    # -------- Diagnostic 3: orbit size --------
    if "orbit" not in skip:
        n_samp = 2_000 if quick else 10_000
        print(f"\n[Diagnostic 3] orbit_size_estimate; n_samples={n_samp}")
        t0 = time.monotonic()
        res3 = orbit_size_estimate(btc_entries, n_samples=n_samp, seed=2)
        print(f"  -> {time.monotonic()-t0:.1f}s; {res3.interpretation}")
        (args.out_dir / "diagnostic_orbit_size.json").write_text(
            json.dumps(res3.to_json(), indent=2) + "\n"
        )
        summary["orbit_size"] = res3.to_json()

    # -------- Diagnostic 4: move profile --------
    if "move" not in skip:
        iters4 = 30_000 if quick else 200_000
        print(f"\n[Diagnostic 4] move_profile n=19 iters={iters4}")
        t0 = time.monotonic()
        res4 = move_profile(19, iterations=iters4, seed=3)
        print(
            f"  -> {time.monotonic()-t0:.1f}s; accepted={res4.accepted} "
            f"increase={res4.n_increase} decrease={res4.n_decrease} "
            f"first_best_step={res4.first_best_step} (T={res4.first_best_temperature:.4f}); "
            f"wasted_tail={res4.wasted_tail_steps}"
        )
        (args.out_dir / "diagnostic_move_profile.json").write_text(
            json.dumps(res4.to_json(), indent=2) + "\n"
        )
        summary["move_profile"] = {
            "best_abs_det": res4.best_abs_det,
            "n_increase": res4.n_increase,
            "n_decrease": res4.n_decrease,
            "first_best_step": res4.first_best_step,
            "first_best_temperature": res4.first_best_temperature,
            "wasted_tail_steps": res4.wasted_tail_steps,
        }

    # -------- Diagnostic 5: cross-n calibration --------
    if "cross" not in skip:
        if quick:
            orders = (11, 13, 15, 17)
            n_starts = 4
            iters5 = 30_000
            structured_seconds = 15.0
            caps: dict[int, int] = {}
        else:
            orders = (11, 13, 15, 17, 19, 21)
            n_starts = 8
            iters5 = 100_000
            structured_seconds = 60.0
            # Cap higher orders to remain within wall budget.
            caps = {19: 60_000, 21: 40_000}
        print(
            f"\n[Diagnostic 5] cross_n_validation orders={orders} "
            f"n_starts={n_starts} iters={iters5}"
        )
        t0 = time.monotonic()
        rows = cross_n_validation(
            orders_to_test=orders,
            n_starts=n_starts,
            iterations=iters5,
            structured_seconds=structured_seconds,
            base_seed=4,
            iteration_caps=caps,
        )
        print(f"  -> total {time.monotonic()-t0:.1f}s")
        for r in rows:
            ratio = f"{r.overall_ratio:.4f}" if r.overall_ratio else "N/A"
            print(
                f"    n={r.n}: best={r.overall_best:,d} proven={r.proven_max} "
                f"ratio={ratio} method={r.method}"
            )
        (args.out_dir / "diagnostic_cross_n.json").write_text(
            json.dumps([r.to_json() for r in rows], indent=2) + "\n"
        )
        summary["cross_n"] = [r.to_json() for r in rows]

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # ---- Human-readable report ----
    write_report(args.out_dir / "diagnostic_report.md", summary)
    print(f"\nWrote {args.out_dir/'diagnostic_report.md'}")
    return 0


def write_report(path: Path, summary: dict) -> None:
    lines: list[str] = []
    lines.append("# Hadamard-maxdet diagnostic deep-dive — report")
    lines.append("")
    lines.append(f"Started:  {summary.get('started')}")
    lines.append(f"Finished: {summary.get('finished')}")
    lines.append("")

    if "cross_n" in summary:
        lines.append("## Diagnostic 5 — cross-n validation (headline)")
        lines.append("")
        lines.append("| n | proven max | overall best | ratio | method |")
        lines.append("|---|---|---|---|---|")
        for r in summary["cross_n"]:
            ratio = f"{r['overall_ratio']:.4f}" if r["overall_ratio"] else "N/A"
            lines.append(
                f"| {r['n']} | {r['proven_max']:,} | {r['overall_best']:,} | "
                f"{ratio} | {r['method']} |"
            )
        lines.append("")

    if "basin_map" in summary:
        lines.append("## Diagnostic 1 — basin map at small n")
        lines.append("")
        lines.append("| n | clusters | largest fraction | best |det| | proven | proven cluster size |")
        lines.append("|---|---|---|---|---|---|")
        for k, v in summary["basin_map"].items():
            n = int(k.lstrip("n"))
            frac = f"{100*v['largest_cluster_fraction']:.1f}%"
            proven = v.get("proven_max")
            lines.append(
                f"| {n} | {v['n_clusters']} | {frac} | "
                f"{v['best_abs_det']:,} | {proven} | "
                f"{v['proven_max_cluster_size']} |"
            )
        lines.append("")

    if "energy_profile" in summary:
        lines.append("## Diagnostic 2 — energy profile around BTC seed (n=23)")
        lines.append("")
        lines.append(f"BTC seed |det| = {summary['energy_profile']['seed_abs_det']:,}")
        lines.append("")
        lines.append("| d | above | equal | below | max neighbour |det| |")
        lines.append("|---|---|---|---|---|")
        for b in summary["energy_profile"]["bins"]:
            lines.append(
                f"| {b['d']} | {b['n_above']} | {b['n_equal']} | "
                f"{b['n_below']} | {b['max']:,} |"
            )
        lines.append("")

    if "orbit_size" in summary:
        o = summary["orbit_size"]
        lines.append("## Diagnostic 3 — equivalence-class orbit size")
        lines.append("")
        lines.append(f"Samples drawn: {o['n_samples']}")
        lines.append(f"Unique canonical hashes: {o['n_unique']}")
        lines.append(f"Collisions: {o['n_collisions']}")
        lines.append(f"Estimated orbit size: {o['estimated_orbit_size']}")
        lines.append(f"95% lower bound: {o['lower_bound_95']}")
        lines.append("")
        lines.append(f"Interpretation: {o['interpretation']}")
        lines.append("")

    if "move_profile" in summary:
        m = summary["move_profile"]
        lines.append("## Diagnostic 4 — move profile (n=19)")
        lines.append("")
        lines.append(f"Best |det| this run: {m['best_abs_det']:,}")
        lines.append(f"Accepted moves that increased |det|: {m['n_increase']}")
        lines.append(f"Accepted moves that decreased |det|: {m['n_decrease']}")
        lines.append(f"First best-ever step: {m['first_best_step']}")
        lines.append(f"Temperature at that step: {m['first_best_temperature']:.6f}")
        lines.append(f"Wasted tail (steps after best-ever): {m['wasted_tail_steps']}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
