"""M1-anchored SA seeded from the standing best (BTC at H23, near-best at H19).

The prior round (`scripts/m1_calibration.py`) established that M1 is a *causal*
SA bias from random init (1.6-2.6× lift over lambda=0) but cannot reach the
proven / BTC basins on its own.  This script tests the composition: start
*at* the standing best matrix and apply M1-anchored SA.

For each n in {19, 23}:

* Load the standing best matrix (H19: diagnostic-SA best, ratio 0.922; H23: BTC
  rich-ansatz best, |det| = 2.626e15).
* Verify with exact Bareiss.
* Run sa_m1_anchored(initial=M_seed_perturbed, ...) for lambda in {0, 0.01,
  0.1, 1.0, 10.0} and perturbation in {0, 1, 3, 5} random flips.
* 4 perturbations × 5 lambdas = 20 runs per n.  Each ~5 minutes wall-clock.
* Parallel via ProcessPoolExecutor across the runs.

Track best |det|, final T-card, basin signature (T-card multiset hash).

If any run beats the seed:
  1. Re-verify with exact integer_determinant.
  2. Re-verify a second time on a fresh load.
  3. Compare T-card / Gram fingerprint to the seed's.
  4. Save to results/H{n}_m1_seeded_BTC_best.json.

Usage:
    python scripts/m1_seeded_btc.py [--quick] [--ns 19 23] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadamard_maxdet.m1_anchor import (
    sa_m1_anchored,
    t_support_cardinality,
    t_support_counter,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# Reference values (per repo BENCHMARKS / docs/findings).
PROVEN = {
    19: 894_426_939_392,
}
H23_BTC_CEILING = 2_626_567_700_217_856
ORRICK_H23 = 2_778_181_500_731_392  # Orrick's best (per docs)

SEED_FILES = {
    19: REPO / "results" / "H19_diagnostic_sa_best.json",
    23: REPO / "results" / "H23_rich_ansatz_best.json",
}

# Iteration rate estimates (this M2 Ultra; from m1_calibration.py).
RATE_TABLE = {17: 10_500.0, 19: 6_700.0, 23: 5_000.0}


def load_seed(n: int) -> tuple[Matrix, int]:
    """Load and verify the standing best matrix for order n."""
    path = SEED_FILES[n]
    with path.open() as f:
        d = json.load(f)
    entries = d.get("matrix") or d.get("entries")
    if entries is None:
        raise ValueError(f"no matrix in {path}")
    if d["n"] != n:
        raise ValueError(f"n mismatch in {path}: {d['n']} != {n}")
    M = Matrix(n=n, entries=entries)
    actual = abs(integer_determinant(entries))
    expected = d.get("abs_det_verified") or d.get("abs_det")
    if expected is not None and actual != expected:
        raise ValueError(f"|det| mismatch in {path}: {actual} != {expected}")
    return M, actual


def perturb(M: Matrix, n_flips: int, seed: int) -> Matrix:
    """Return a copy of M with `n_flips` random distinct entries flipped."""
    if n_flips == 0:
        return M.copy()
    rng = random.Random(seed)
    n = M.n
    coords: set[tuple[int, int]] = set()
    while len(coords) < n_flips:
        coords.add((rng.randrange(n), rng.randrange(n)))
    new_entries = [row[:] for row in M.entries]
    for i, c in coords:
        new_entries[i][c] = -new_entries[i][c]
    return Matrix(n=n, entries=new_entries)


def t_signature_hash(entries) -> str:
    """A short content-hash of the T-multiset (basin fingerprint)."""
    cnt = t_support_counter(entries)
    items = sorted(cnt.items())
    h = hash(tuple(items))
    return f"{h & 0xFFFFFFFF:08x}"


def run_one(args: dict) -> dict:
    """One (n, lambda, perturbation, seed) restart.  Pickleable."""
    n = args["n"]
    lam = args["lambda_m1"]
    n_flips = args["n_flips"]
    run_seed = args["seed"]
    wall_budget = args["wall_budget"]
    seed_entries = args["seed_entries"]

    M_seed = Matrix(n=n, entries=[row[:] for row in seed_entries])
    M_init = perturb(M_seed, n_flips, seed=run_seed * 7919 + n_flips)
    init_det = abs(integer_determinant(M_init.entries))
    init_card = t_support_cardinality(M_init.entries)
    init_sig = t_signature_hash(M_init.entries)

    iters_per_sec = RATE_TABLE.get(n, 5_000.0)
    iterations = int(wall_budget * iters_per_sec)

    t0 = time.monotonic()
    res = sa_m1_anchored(
        n,
        iterations=iterations,
        seed=run_seed,
        lambda_m1=lam,
        initial=M_init,
        # Cooler schedule than calibration (we are starting near a deep basin,
        # we want exploration but not destruction).
        t_start=0.5,
        t_end=0.02,
        refresh_every=2000,
    )
    walltime = time.monotonic() - t0

    out_card = t_support_cardinality(res.matrix.entries)
    out_sig = t_signature_hash(res.matrix.entries)
    # Re-verify final det.
    final_det = abs(integer_determinant(res.matrix.entries))

    return {
        "n": n,
        "lambda": lam,
        "n_flips": n_flips,
        "seed": run_seed,
        "iterations": iterations,
        "walltime": walltime,
        "init_abs_det": int(init_det),
        "init_t_card": int(init_card),
        "init_t_sig": init_sig,
        "best_abs_det": int(final_det),
        "best_t_card": int(out_card),
        "best_t_sig": out_sig,
        "moved_from_seed": out_sig != init_sig,
        "matrix": res.matrix.entries,
        "accepted": res.accepted,
    }


def summarize(per_n: dict, seed_det: int, seed_card: int, ceiling_label: str,
              ceiling_value: int) -> dict:
    """Per-(lambda, n_flips) summary table."""
    by_lam: dict = {}
    for r in per_n["runs"]:
        lam = r["lambda"]
        nf = r["n_flips"]
        key = (lam, nf)
        if key not in by_lam:
            by_lam[key] = []
        by_lam[key].append(r)

    table_rows = []
    for (lam, nf), rs in sorted(by_lam.items()):
        # Single restart per (lam, nf) cell in this design, but be defensive.
        best = max(rs, key=lambda r: r["best_abs_det"])
        table_rows.append({
            "lambda": lam,
            "n_flips": nf,
            "best_abs_det": best["best_abs_det"],
            "best_t_card": best["best_t_card"],
            "ratio_to_seed": best["best_abs_det"] / seed_det,
            "ratio_to_ceiling": best["best_abs_det"] / ceiling_value,
            "moved_from_seed": best["moved_from_seed"],
            "matrix": best["matrix"],
        })

    overall_best = max(table_rows, key=lambda r: r["best_abs_det"])
    return {
        "seed_abs_det": seed_det,
        "seed_t_card": seed_card,
        "ceiling_label": ceiling_label,
        "ceiling_value": ceiling_value,
        "rows": [{k: v for k, v in r.items() if k != "matrix"}
                 for r in table_rows],
        "overall_best": {k: v for k, v in overall_best.items() if k != "matrix"},
        "overall_best_matrix": overall_best["matrix"],
        "beat_seed": overall_best["best_abs_det"] > seed_det,
        "beat_ceiling": overall_best["best_abs_det"] > ceiling_value,
    }


def print_table(n: int, summary: dict) -> None:
    print(f"\n=== n={n}: M1-seeded-BTC results ===")
    print(f"Seed |det|: {summary['seed_abs_det']:,d}  T-card: {summary['seed_t_card']}")
    print(f"Reference ceiling ({summary['ceiling_label']}): "
          f"{summary['ceiling_value']:,d}")
    print()
    print(f"{'lambda':>8}  {'flips':>5}  {'best |det|':>22}  "
          f"{'/seed':>8}  {'/ceiling':>8}  {'T-card':>6}  {'moved':>5}")
    for r in summary["rows"]:
        print(
            f"{r['lambda']:>8g}  {r['n_flips']:>5d}  "
            f"{r['best_abs_det']:>22,d}  "
            f"{r['ratio_to_seed']:>8.4f}  {r['ratio_to_ceiling']:>8.4f}  "
            f"{r['best_t_card']:>6d}  "
            f"{'Y' if r['moved_from_seed'] else 'N':>5}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Quick smoke (10s/run, fewer perturbations).")
    ap.add_argument("--ns", type=int, nargs="+", default=[19, 23])
    ap.add_argument("--workers", type=int, default=20,
                    help="ProcessPool worker count.")
    ap.add_argument("--out", type=str,
                    default=str(REPO / "results" / "m1_seeded_btc.json"))
    args = ap.parse_args()

    if args.quick:
        wall_per_run = 10.0
        lambdas = [0.0, 1.0]
        flip_counts = [0, 3]
    else:
        wall_per_run = 300.0  # 5 minutes per run (per spec)
        lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
        flip_counts = [0, 1, 3, 5]

    overall = {"meta": {
        "wall_per_run_seconds": wall_per_run,
        "lambdas": lambdas,
        "flip_counts": flip_counts,
        "ns": args.ns,
        "workers": args.workers,
    }, "per_n": {}}

    t_total = time.monotonic()

    for n in args.ns:
        seed_M, seed_det = load_seed(n)
        seed_card = t_support_cardinality(seed_M.entries)
        seed_sig = t_signature_hash(seed_M.entries)
        ceiling_label = "Orrick H23 record" if n == 23 else "Hadamard-class proven max"
        ceiling_value = ORRICK_H23 if n == 23 else PROVEN.get(n, seed_det)

        print(f"\n### n={n}: seed |det|={seed_det:,d}, T-card={seed_card}, "
              f"sig={seed_sig}")

        # Build the run grid.
        run_args = []
        for lam in lambdas:
            for nf in flip_counts:
                run_args.append({
                    "n": n,
                    "lambda_m1": lam,
                    "n_flips": nf,
                    "seed": 1000 + 13 * lambdas.index(lam) + nf,
                    "wall_budget": wall_per_run,
                    "seed_entries": seed_M.entries,
                })

        n_workers = min(args.workers, len(run_args))
        print(f"Running {len(run_args)} restarts on {n_workers} workers, "
              f"~{wall_per_run:.0f}s each ({wall_per_run * len(run_args) / n_workers / 60:.1f} min wall est)")

        results = []
        t_n = time.monotonic()
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(run_one, ra): ra for ra in run_args}
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    print(
                        f"  [n={n}] lam={r['lambda']:>5g} flips={r['n_flips']} "
                        f"|det|={r['best_abs_det']:>22,d} "
                        f"T-card={r['best_t_card']:>3d} "
                        f"moved={'Y' if r['moved_from_seed'] else 'N'} "
                        f"({r['walltime']:.1f}s)"
                    )
        else:
            for ra in run_args:
                r = run_one(ra)
                results.append(r)
                print(
                    f"  [n={n}] lam={r['lambda']:>5g} flips={r['n_flips']} "
                    f"|det|={r['best_abs_det']:>22,d} "
                    f"T-card={r['best_t_card']:>3d} "
                    f"({r['walltime']:.1f}s)"
                )
        print(f"  n={n} block elapsed: {time.monotonic() - t_n:.1f}s")

        per_n = {"runs": results,
                 "seed_abs_det": seed_det,
                 "seed_t_card": seed_card,
                 "seed_sig": seed_sig}
        summary = summarize(per_n, seed_det, seed_card, ceiling_label, ceiling_value)
        per_n["summary"] = summary
        overall["per_n"][n] = per_n

        print_table(n, summary)

        # Record-check for H23.
        if n == 23 and summary["overall_best"]["best_abs_det"] > H23_BTC_CEILING:
            best = summary["overall_best"]
            best_M = summary["overall_best_matrix"]
            # Re-verify TWICE.
            det_check_1 = abs(integer_determinant(best_M))
            det_check_2 = abs(integer_determinant(
                [row[:] for row in best_M]
            ))
            seed_full_sig = sorted(t_support_counter(seed_M.entries).items())
            best_full_sig = sorted(t_support_counter(best_M).items())
            same_basin = seed_full_sig == best_full_sig

            print(f"\n*** n=23: M1-seeded SA exceeded BTC ceiling! ***")
            print(f"    |det| = {best['best_abs_det']:,d}")
            print(f"    BTC   = {H23_BTC_CEILING:,d}")
            print(f"    re-verify 1: {det_check_1:,d} "
                  f"({'MATCH' if det_check_1 == best['best_abs_det'] else 'MISMATCH'})")
            print(f"    re-verify 2: {det_check_2:,d} "
                  f"({'MATCH' if det_check_2 == best['best_abs_det'] else 'MISMATCH'})")
            print(f"    same Gram-T basin as BTC seed: {same_basin}")

            payload = {
                "n": 23,
                "abs_det": int(det_check_1),
                "abs_det_verified_twice": int(det_check_2),
                "matrix": best_M,
                "method": "m1_anchored_sa_seeded_from_btc",
                "seed_source": "results/H23_rich_ansatz_best.json",
                "seed_abs_det": seed_det,
                "lambda_m1": best["lambda"],
                "n_flips_perturbation": best["n_flips"],
                "best_t_card": best["best_t_card"],
                "seed_t_card": seed_card,
                "moved_from_seed_basin": best["moved_from_seed"],
                "same_basin_as_btc_seed": same_basin,
                "t_signature_seed": dict(seed_full_sig),
                "t_signature_best": dict(best_full_sig),
                "ceiling_btc": H23_BTC_CEILING,
                "ceiling_orrick": ORRICK_H23,
                "beat_orrick": int(det_check_1) > ORRICK_H23,
            }
            out_record = REPO / "results" / "H23_m1_seeded_BTC_best.json"
            with out_record.open("w") as f:
                json.dump(payload, f, indent=2)
            print(f"    saved to {out_record}")

    overall["meta"]["total_walltime_seconds"] = time.monotonic() - t_total

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Final cross-n verdict.
    print("\n=== FINAL VERDICT ===")
    for n in args.ns:
        s = overall["per_n"][n]["summary"]
        if s["beat_ceiling"]:
            print(f"  n={n}: BEAT CEILING ({s['ceiling_label']})  "
                  f"best={s['overall_best']['best_abs_det']:,d}")
        elif s["beat_seed"]:
            print(f"  n={n}: improved on seed but not ceiling.  "
                  f"best={s['overall_best']['best_abs_det']:,d} "
                  f"vs seed {s['seed_abs_det']:,d} "
                  f"vs ceiling {s['ceiling_value']:,d}")
        else:
            # Did any run move basin?
            n_moved = sum(1 for r in overall["per_n"][n]["runs"]
                          if r["moved_from_seed"])
            n_total = len(overall["per_n"][n]["runs"])
            t_card_changed = any(
                r["best_t_card"] != s["seed_t_card"]
                for r in overall["per_n"][n]["runs"]
            )
            print(
                f"  n={n}: did NOT beat seed.  "
                f"best={s['overall_best']['best_abs_det']:,d} "
                f"(seed {s['seed_abs_det']:,d}, ratio "
                f"{s['overall_best']['best_abs_det'] / s['seed_abs_det']:.4f})  "
                f"runs that moved basin: {n_moved}/{n_total}, "
                f"T-card changed at any lam: {t_card_changed}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
