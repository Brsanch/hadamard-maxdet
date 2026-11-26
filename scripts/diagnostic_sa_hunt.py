"""Diagnostic SA hunt: 4-variant H19 calibration + best-variant H23 push.

Variants
--------
(a) baseline:        sa_incremental, 1 long run (200k iter, t_end=0.005)
(b) hot-only:        4 × sa_hot_only, 50k iter, t_end=0.05
(c) hot-only + balance bonus (best λ from sweep on smaller per-λ runs)
(d) hot-only + balance + weight-perturbation

All four variants get equal wall-clock at H19 (default ~60 s each).
The best variant is then applied at H23 from random seeds for ~5 min.

Outputs
-------
- results/diagnostic_sa_h19.json   (per-variant headline + fingerprint)
- results/diagnostic_sa_h23.json   (best matrix + fingerprint at H23)
- results/H19_diagnostic_sa_best.json  (best matrix at H19)
- results/H23_diagnostic_sa_best.json  (best matrix at H23, if it
  exceeds BTC; gitignored anyway)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from hadamard_maxdet.diagnostic_sa import (
    count_imbalance,
    sa_balanced,
    sa_hot_only,
)
from hadamard_maxdet.incremental import sa_incremental
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.orders import BENCHMARKS
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import fingerprint_matrix


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def gram_count_summary(M: list[list[int]]) -> dict[str, int]:
    """Returns the strict-upper-triangle count of -1, +1, 0, and other
    inner-product values."""
    n = len(M)
    counts: Counter = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            s = 0
            for k in range(n):
                s += M[i][k] * M[j][k]
            counts[s] += 1
    return {str(k): v for k, v in sorted(counts.items())}


def row_weight_diversity(M: list[list[int]]) -> int:
    weights = {sum(1 for x in row if x > 0) for row in M}
    return len(weights)


def variant_summary(label: str, abs_det: int, M: list[list[int]],
                    proven: int | None, walltime: float, n_starts: int,
                    extra: dict | None = None) -> dict:
    out = {
        "variant": label,
        "abs_det": abs_det,
        "proven": proven,
        "ratio_to_proven": (abs_det / proven) if proven else None,
        "walltime_seconds": walltime,
        "n_starts": n_starts,
        "gram_count_dist": gram_count_summary(M),
        "imbalance": count_imbalance(M),
        "row_weight_diversity": row_weight_diversity(M),
    }
    if extra:
        out.update(extra)
    return out


def run_baseline_h19(seed: int, iterations: int,
                     wall_budget: float | None = None
                     ) -> tuple[int, list[list[int]], float, int]:
    """Baseline ``sa_incremental`` long runs.

    If ``wall_budget`` is supplied, multi-restart fills the budget (so
    the baseline gets the same wall time as the other variants); else a
    single run.
    """
    t0 = time.monotonic()
    best_abs = 0
    best_M: list[list[int]] = []
    n_starts = 0
    s = seed
    while True:
        res = sa_incremental(
            19,
            iterations=iterations,
            t_start=1.0,
            t_end=0.005,
            seed=s,
            refresh_every=0,
        )
        n_starts += 1
        if int(res.abs_det) > best_abs:
            best_abs = int(res.abs_det)
            best_M = [row[:] for row in res.matrix.entries]
        s += 1009
        if wall_budget is None:
            break
        if time.monotonic() - t0 >= wall_budget:
            break
    walltime = time.monotonic() - t0
    return best_abs, best_M, walltime, n_starts


def run_hot_only_h19(seed: int, iterations_per: int, wall_budget: float
                     ) -> tuple[int, list[list[int]], float, int]:
    """Multi-restart hot-only, capped by wall budget."""
    t0 = time.monotonic()
    best_abs = 0
    best_M: list[list[int]] = []
    n_starts = 0
    s = seed
    while time.monotonic() - t0 < wall_budget:
        res = sa_hot_only(
            19,
            iterations=iterations_per,
            t_start=1.0,
            t_end=0.05,
            seed=s,
            refresh_every=0,
        )
        n_starts += 1
        if int(res.abs_det) > best_abs:
            best_abs = int(res.abs_det)
            best_M = [row[:] for row in res.matrix.entries]
        s += 1009
    walltime = time.monotonic() - t0
    return best_abs, best_M, walltime, n_starts


def run_balanced_h19(seed: int, iterations_per: int, wall_budget: float,
                     lambda_balance: float, weight_perturb_prob: float
                     ) -> tuple[int, list[list[int]], float, int]:
    """Multi-restart sa_balanced, capped by wall budget."""
    t0 = time.monotonic()
    best_abs = 0
    best_M: list[list[int]] = []
    n_starts = 0
    s = seed
    while time.monotonic() - t0 < wall_budget:
        res = sa_balanced(
            19,
            iterations=iterations_per,
            t_start=1.0,
            t_end=0.05,
            lambda_balance=lambda_balance,
            weight_perturb_prob=weight_perturb_prob,
            seed=s,
            refresh_every=0,
        )
        n_starts += 1
        if int(res.abs_det) > best_abs:
            best_abs = int(res.abs_det)
            best_M = [row[:] for row in res.matrix.entries]
        s += 1009
    walltime = time.monotonic() - t0
    return best_abs, best_M, walltime, n_starts


def lambda_sweep_h19(seed: int, iterations_per: int, per_lambda_budget: float,
                     lambdas: list[float]) -> tuple[float, dict]:
    """Quick sweep to pick the best λ. Returns (best_lambda, sweep_results)."""
    sweep = {}
    best_l = 0.0
    best_score = -1
    for lam in lambdas:
        abs_det, M, wt, ns = run_balanced_h19(
            seed, iterations_per, per_lambda_budget, lam, 0.0,
        )
        sweep[str(lam)] = {
            "abs_det": abs_det,
            "n_starts": ns,
            "walltime": wt,
            "imbalance": count_imbalance(M),
        }
        if abs_det > best_score:
            best_score = abs_det
            best_l = lam
    return best_l, sweep


def run_h23_best(seed: int, wall_budget: float, iterations_per: int,
                 lambda_balance: float, weight_perturb_prob: float
                 ) -> tuple[int, list[list[int]], float, int]:
    t0 = time.monotonic()
    best_abs = 0
    best_M: list[list[int]] = []
    n_starts = 0
    s = seed
    while time.monotonic() - t0 < wall_budget:
        res = sa_balanced(
            23,
            iterations=iterations_per,
            t_start=1.0,
            t_end=0.05,
            lambda_balance=lambda_balance,
            weight_perturb_prob=weight_perturb_prob,
            seed=s,
            refresh_every=0,
        )
        n_starts += 1
        if int(res.abs_det) > best_abs:
            best_abs = int(res.abs_det)
            best_M = [row[:] for row in res.matrix.entries]
        s += 1009
    walltime = time.monotonic() - t0
    return best_abs, best_M, walltime, n_starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h19-budget", type=float, default=60.0,
                        help="seconds per H19 variant")
    parser.add_argument("--h19-baseline-iters", type=int, default=200_000)
    parser.add_argument("--h19-hot-iters", type=int, default=50_000)
    parser.add_argument("--h23-budget", type=float, default=300.0)
    parser.add_argument("--h23-iters", type=int, default=50_000)
    parser.add_argument("--lambda-sweep-budget", type=float, default=20.0,
                        help="seconds per λ in the sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambdas", type=str,
                        default="0.0,0.001,0.01,0.1,0.5")
    parser.add_argument("--skip-h23", action="store_true")
    parser.add_argument("--out-h19", type=str,
                        default=str(RESULTS_DIR / "diagnostic_sa_h19.json"))
    parser.add_argument("--out-h23", type=str,
                        default=str(RESULTS_DIR / "diagnostic_sa_h23.json"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    proven_h19 = BENCHMARKS["H19"].best_known
    btc_h23 = 2_626_567_700_217_856  # Best Bordered Two-Circulant on H23

    print(f"=== H19 calibration (proven max: {proven_h19:,}) ===")
    print(f"Wall budget per variant: {args.h19_budget:.1f}s\n")

    # (a) baseline (multi-restart in same wall budget)
    print("(a) baseline sa_incremental multi-restart (same wall budget)...")
    a_det, a_M, a_wt, a_ns = run_baseline_h19(
        args.seed, args.h19_baseline_iters, args.h19_budget,
    )
    print(f"    abs_det = {a_det:,}  ({a_det/proven_h19:.4f} of proven), "
          f"{a_wt:.1f}s, {a_ns} starts")

    # (b) hot-only multi-restart
    print("(b) hot-only multi-restart...")
    b_det, b_M, b_wt, b_ns = run_hot_only_h19(
        args.seed, args.h19_hot_iters, args.h19_budget,
    )
    print(f"    abs_det = {b_det:,}  ({b_det/proven_h19:.4f} of proven), "
          f"{b_wt:.1f}s, {b_ns} starts")

    # (c) lambda sweep + balanced
    print("(c) lambda sweep ...")
    lambdas = [float(x) for x in args.lambdas.split(",")]
    best_lambda, sweep_results = lambda_sweep_h19(
        args.seed + 1, args.h19_hot_iters, args.lambda_sweep_budget, lambdas,
    )
    print(f"    best lambda = {best_lambda}")
    print(f"    sweep summary: {json.dumps(sweep_results, indent=2)}")
    print(f"(c) hot-only + balance (lambda={best_lambda})...")
    c_det, c_M, c_wt, c_ns = run_balanced_h19(
        args.seed + 100, args.h19_hot_iters, args.h19_budget,
        best_lambda, 0.0,
    )
    print(f"    abs_det = {c_det:,}  ({c_det/proven_h19:.4f} of proven), "
          f"{c_wt:.1f}s, {c_ns} starts")

    # (d) hot-only + balance + weight-perturbation
    print(f"(d) hot-only + balance (lambda={best_lambda}) + weight-perturb 0.2...")
    d_det, d_M, d_wt, d_ns = run_balanced_h19(
        args.seed + 200, args.h19_hot_iters, args.h19_budget,
        best_lambda, 0.2,
    )
    print(f"    abs_det = {d_det:,}  ({d_det/proven_h19:.4f} of proven), "
          f"{d_wt:.1f}s, {d_ns} starts")

    summaries = [
        variant_summary("baseline", a_det, a_M, proven_h19, a_wt, a_ns),
        variant_summary("hot_only", b_det, b_M, proven_h19, b_wt, b_ns),
        variant_summary("hot_only+balance", c_det, c_M, proven_h19, c_wt,
                        c_ns,
                        extra={"lambda": best_lambda}),
        variant_summary("hot_only+balance+weight", d_det, d_M, proven_h19,
                        d_wt, d_ns,
                        extra={"lambda": best_lambda,
                               "weight_perturb_prob": 0.2}),
    ]
    h19_out = {
        "n": 19,
        "proven": proven_h19,
        "lambda_sweep": sweep_results,
        "best_lambda": best_lambda,
        "variants": summaries,
    }
    Path(args.out_h19).write_text(json.dumps(h19_out, indent=2))
    print(f"\nSaved {args.out_h19}")

    # Determine winner of the four
    winner = max(summaries, key=lambda s: s["abs_det"])
    print(f"\nH19 winning variant: {winner['variant']} = {winner['abs_det']:,}"
          f" (ratio {winner['ratio_to_proven']:.4f})")

    # Save the best H19 matrix
    best_M_h19 = max(
        zip([a_det, b_det, c_det, d_det],
            [a_M, b_M, c_M, d_M],
            ["baseline", "hot_only", "hot_only+balance",
             "hot_only+balance+weight"]),
        key=lambda x: x[0],
    )
    h19_best_out = {
        "n": 19,
        "abs_det": best_M_h19[0],
        "abs_det_verified": abs(integer_determinant(best_M_h19[1])),
        "ratio_to_proven": best_M_h19[0] / proven_h19,
        "variant": best_M_h19[2],
        "matrix": best_M_h19[1],
    }
    Path(RESULTS_DIR / "H19_diagnostic_sa_best.json").write_text(
        json.dumps(h19_best_out, indent=2),
    )

    # === H23 push ===
    if args.skip_h23:
        print("Skipping H23 push (--skip-h23).")
        return

    print(f"\n=== H23 push (BTC: {btc_h23:,}) ===")
    print(f"Wall budget: {args.h23_budget:.1f}s, applying winning variant.\n")
    # Translate winner to (lambda, weight_perturb_prob)
    if winner["variant"] == "baseline":
        lam_h23 = 0.0
        wpp_h23 = 0.0
    elif winner["variant"] == "hot_only":
        lam_h23 = 0.0
        wpp_h23 = 0.0
    elif winner["variant"] == "hot_only+balance":
        lam_h23 = best_lambda
        wpp_h23 = 0.0
    else:
        lam_h23 = best_lambda
        wpp_h23 = 0.2
    h23_det, h23_M, h23_wt, h23_ns = run_h23_best(
        args.seed + 999, args.h23_budget, args.h23_iters,
        lam_h23, wpp_h23,
    )
    print(f"H23 best: {h23_det:,}  (ratio to BTC: {h23_det/btc_h23:.4f}), "
          f"{h23_wt:.1f}s, {h23_ns} starts")

    h23_summary = {
        "n": 23,
        "btc_reference": btc_h23,
        "abs_det": h23_det,
        "abs_det_verified": abs(integer_determinant(h23_M)) if h23_M else 0,
        "ratio_to_btc": h23_det / btc_h23,
        "lambda_balance": lam_h23,
        "weight_perturb_prob": wpp_h23,
        "walltime_seconds": h23_wt,
        "n_starts": h23_ns,
        "imbalance": count_imbalance(h23_M) if h23_M else None,
        "row_weight_diversity": (
            row_weight_diversity(h23_M) if h23_M else None
        ),
        "gram_count_dist": gram_count_summary(h23_M) if h23_M else None,
    }
    Path(args.out_h23).write_text(json.dumps(h23_summary, indent=2))
    h23_best_out = {
        "n": 23,
        "abs_det": h23_det,
        "abs_det_verified": (
            abs(integer_determinant(h23_M)) if h23_M else 0
        ),
        "ratio_to_btc": h23_det / btc_h23,
        "matrix": h23_M,
    }
    Path(RESULTS_DIR / "H23_diagnostic_sa_best.json").write_text(
        json.dumps(h23_best_out, indent=2),
    )
    print(f"Saved {args.out_h23}")


if __name__ == "__main__":
    main()
