"""Class-tabu basin-exclusion campaign for n in {21, 23}.

Experiment 1 (basin enumeration): iteratively run class-tabu SA, locking each
discovered basin's signature, and report the distinct basins found.

Experiment 2 (bridge): from each non-BTC basin's seed at n=23, run unlocked
SA (long budget) to test whether the seed can drift toward a higher peak
(BTC's 2.626e15 ceiling or above).

Outputs (under ``results/``):
    H21_basin_<i>.json
    H23_basin_<i>.json
    H23_basin_exclusion_best.json

Run:
    python scripts/basin_exclusion_h23.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hadamard_maxdet.class_tabu import (
    Basin,
    bridge_from_seed,
    class_signature,
    iterative_basin_discovery,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# BTC reference at n=23 (from results/H23_attack_best.json).
BTC_H23_SIG: tuple[tuple[int, int], ...] = (
    (-3, 11), (-1, 209), (1, 11), (3, 22),
)
BTC_H23_ABSDET: int = 2_626_567_700_217_856

# BTC reference at n=21 (known BTC ceiling ~51.2T).
# Compute on the fly from any 51.2T witness if available; otherwise just
# report whichever class hits 51.2T as the BTC analogue.
BTC_H21_ABSDET: int = 40_960_000_000_000  # baseline reference (~51.2T ceiling)


def basin_to_dict(b: Basin, *, label: str | None = None) -> dict:
    return {
        "label": label,
        "n": b.matrix.n,
        "abs_det": b.abs_det,
        "abs_det_verified_bareiss": abs(integer_determinant(b.matrix.entries)),
        "signature": [list(p) for p in b.signature],
        "discovery_phase": b.discovery_phase,
        "walltime_seconds": b.walltime_seconds,
        "matrix_entries": b.matrix.entries,
    }


def write_basins(basins: list[Basin], n: int, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for i, b in enumerate(basins):
        fp = out_dir / f"H{n}_basin_{i}.json"
        fp.write_text(json.dumps(basin_to_dict(b, label=f"basin_{i}"), indent=2) + "\n")
        paths.append(fp)
    return paths


def print_basin_table(title: str, basins: list[Basin], btc_ref: int | None = None) -> None:
    print(f"\n=== {title} ({len(basins)} basins) ===")
    print(f"{'#':>2}  {'|det|':>20}  {'/ BTC':>9}  signature")
    for i, b in enumerate(basins):
        ratio = b.abs_det / btc_ref if btc_ref else float("nan")
        sig_str = ", ".join(f"({v:+d}: {c})" for v, c in b.signature)
        print(f"{i:>2}  {b.abs_det:>20,}  {ratio:>8.4f}   {sig_str}")


def run_experiment_1_n21(out_dir: Path, time_budget: float = 240.0) -> list[Basin]:
    print(f"\n>>> Experiment 1 (sanity) at n=21 — budget {time_budget:.0f} s")
    t0 = time.monotonic()
    result = iterative_basin_discovery(
        21,
        max_phases=20,
        iters_per_phase=40_000,
        base_seed=20210,
        patience=6,
        time_budget_seconds=time_budget,
    )
    walltime = time.monotonic() - t0
    print(f"  ran {result.phases_run} phases in {walltime:.1f} s; "
          f"{len(result.basins)} distinct basins")
    write_basins(result.basins, 21, out_dir)
    print_basin_table("n=21 basin table", result.basins, btc_ref=BTC_H21_ABSDET)
    # Pretty-print rejection history for diagnostic.
    print("\n  Phase-by-phase:")
    for h in result.rejection_history:
        marker = "NEW " if h["new_basin"] else "dup "
        sig_short = ", ".join(f"({v:+d}:{c})" for v, c in h["signature"])
        print(f"    phase {h['phase']:>2}: {marker} |det|={h['abs_det']:>20,}  "
              f"acc={h['accepted']:>5} cls_rej={h['rejected_class']:>5}  {sig_short}")
    return result.basins


def run_experiment_1_n23(out_dir: Path, time_budget: float = 360.0) -> list[Basin]:
    print(f"\n>>> Experiment 1 at n=23 — budget {time_budget:.0f} s")
    t0 = time.monotonic()
    result = iterative_basin_discovery(
        23,
        max_phases=24,
        iters_per_phase=60_000,
        base_seed=20230,
        patience=8,
        time_budget_seconds=time_budget,
    )
    walltime = time.monotonic() - t0
    print(f"  ran {result.phases_run} phases in {walltime:.1f} s; "
          f"{len(result.basins)} distinct basins")
    write_basins(result.basins, 23, out_dir)
    print_basin_table("n=23 basin table", result.basins, btc_ref=BTC_H23_ABSDET)
    print("\n  Phase-by-phase:")
    for h in result.rejection_history:
        marker = "NEW " if h["new_basin"] else "dup "
        sig_short = ", ".join(f"({v:+d}:{c})" for v, c in h["signature"])
        print(f"    phase {h['phase']:>2}: {marker} |det|={h['abs_det']:>20,}  "
              f"acc={h['accepted']:>5} cls_rej={h['rejected_class']:>5}  {sig_short}")
    return result.basins


def run_experiment_2(non_btc_basins: list[Basin], time_budget_per_seed: float = 90.0) -> dict:
    """For each non-BTC basin at n=23, run unlocked SA from its seed."""
    print(f"\n>>> Experiment 2: bridge from non-BTC seeds — "
          f"{time_budget_per_seed:.0f} s per seed")
    out: list[dict] = []
    overall_best_det = 0
    overall_best_matrix: Matrix | None = None
    overall_best_label: str | None = None

    for i, basin in enumerate(non_btc_basins):
        print(f"\n  Seed basin {i}: |det|={basin.abs_det:,}  sig={basin.signature}")
        # Approximate "long budget" by iteration count tuned for ~90 s.
        # n=23 incremental SA does ~30-40k iter/s; aim for ~3M iter for 90 s.
        iters = int(35_000 * time_budget_per_seed)
        bridge = bridge_from_seed(
            basin.matrix,
            iterations=iters,
            t_start=0.5,
            t_end=0.002,
            seed=99_991 + i,
        )
        end_sig = bridge.final_signature
        same = end_sig == basin.signature
        is_btc = end_sig == BTC_H23_SIG
        verdict = "stayed" if same else ("→ BTC" if is_btc else "→ third class")
        print(f"    -> |det|={bridge.final_abs_det:,}  walltime={bridge.walltime_seconds:.1f} s  "
              f"verdict: {verdict}")
        print(f"       end sig: {end_sig}")
        rec = {
            "seed_index": i,
            "seed_signature": [list(p) for p in basin.signature],
            "seed_abs_det": basin.abs_det,
            "final_signature": [list(p) for p in end_sig],
            "final_abs_det": bridge.final_abs_det,
            "drifted": bridge.drifted,
            "reached_btc_class": is_btc,
            "exceeded_btc_ceiling": bridge.final_abs_det > BTC_H23_ABSDET,
            "walltime_seconds": bridge.walltime_seconds,
            "verdict": verdict,
        }
        out.append(rec)
        if bridge.final_abs_det > overall_best_det:
            overall_best_det = bridge.final_abs_det
            overall_best_matrix = bridge.final_matrix
            overall_best_label = f"bridge_from_basin_{i}"

    return {
        "per_seed": out,
        "overall_best_det": overall_best_det,
        "overall_best_label": overall_best_label,
        "overall_best_matrix": (
            overall_best_matrix.entries if overall_best_matrix else None
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n21-budget", type=float, default=240.0)
    p.add_argument("--n23-budget", type=float, default=360.0)
    p.add_argument("--bridge-budget", type=float, default=90.0)
    p.add_argument("--skip-n21", action="store_true")
    p.add_argument("--skip-bridge", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.monotonic()

    if not args.skip_n21:
        run_experiment_1_n21(out_dir, time_budget=args.n21_budget)

    basins_23 = run_experiment_1_n23(out_dir, time_budget=args.n23_budget)

    # Identify non-BTC basins.
    non_btc = [b for b in basins_23 if b.signature != BTC_H23_SIG]
    print(f"\n  Of {len(basins_23)} basins at n=23, {len(non_btc)} are non-BTC.")

    bridge_summary: dict = {}
    if not args.skip_bridge and non_btc:
        bridge_summary = run_experiment_2(non_btc, time_budget_per_seed=args.bridge_budget)

    # Consolidated best.
    consolidated = {
        "experiment": "basin_exclusion_h23",
        "btc_reference": {
            "abs_det": BTC_H23_ABSDET,
            "signature": [list(p) for p in BTC_H23_SIG],
        },
        "n23_basins": [basin_to_dict(b) for b in basins_23],
        "non_btc_basin_count": len(non_btc),
        "bridge_results": bridge_summary,
        "best_basin_abs_det": (
            max(b.abs_det for b in basins_23) if basins_23 else 0
        ),
        "best_basin_index": (
            max(range(len(basins_23)), key=lambda i: basins_23[i].abs_det)
            if basins_23 else -1
        ),
        "exceeded_btc_ceiling": any(
            b.abs_det > BTC_H23_ABSDET for b in basins_23
        ) or (
            bridge_summary.get("overall_best_det", 0) > BTC_H23_ABSDET
            if bridge_summary else False
        ),
        "total_walltime_seconds": time.monotonic() - t_total,
    }
    fp = out_dir / "H23_basin_exclusion_best.json"
    fp.write_text(json.dumps(consolidated, indent=2) + "\n")
    print(f"\n  Wrote consolidated summary: {fp}")
    print(f"  Total walltime: {consolidated['total_walltime_seconds']:.1f} s")
    print(f"  Best basin |det|: {consolidated['best_basin_abs_det']:,}")
    print(f"  Exceeded BTC ceiling? {consolidated['exceeded_btc_ceiling']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
