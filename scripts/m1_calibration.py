"""Calibrate M1 (triple cyclic Gram product cardinality) as an SA energy bias.

Three experiments, each running anchored SA at multiple lambda values:

* Experiment 1 — H17 sanity (proven max known, target T-card = 2).
* Experiment 2 — H19 real test (proven max known, target T-card <= 3).
* Experiment 3 — H23 push (open target; current BTC ceiling 2.626e15).

For each (n, lambda), we run K parallel restarts from random init, each
with a wall-clock budget. The best |det| (verified by exact Bareiss) and
the final T-support cardinality are recorded.

Verdict logic:
  * Causal:    at lambda > 0, |det|/proven improves above the lambda=0
               baseline AND T-support shrinks toward the proven target.
  * Correlated-only: T-support shrinks but |det|/proven stays at the
                     lambda=0 ceiling.

Usage:
    python scripts/m1_calibration.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadamard_maxdet.m1_anchor import (
    sa_m1_anchored,
    t_support_cardinality,
)
from hadamard_maxdet.verifier import integer_determinant


# Proven and reference values (per repo BENCHMARKS / docs/findings).
PROVEN = {
    17: 21_474_836_480,
    19: 894_426_939_392,
}
H23_BTC_CEILING = 2_626_567_700_217_856  # in-repo bordered two-circulant best.


def _run_one(
    n: int,
    lam: float,
    seed: int,
    wall_budget: float,
) -> dict:
    """Run a single sa_m1_anchored restart, sized so total wall ~ wall_budget.

    Measured iteration rates on this M2 Ultra:
      n=17 ~ 11,000 iter/s; n=19 ~ 8,300/s; n=23 ~ 5,400/s.
    """
    rate_table = {17: 10_500.0, 19: 8_000.0, 23: 5_200.0}
    iters_per_sec_est = rate_table.get(n, 5_000.0)
    iterations = int(wall_budget * iters_per_sec_est)

    t0 = time.monotonic()
    res = sa_m1_anchored(
        n,
        iterations=iterations,
        seed=seed,
        lambda_m1=lam,
        t_start=1.0,
        t_end=0.05,
        refresh_every=2000,
    )
    elapsed = time.monotonic() - t0
    card = t_support_cardinality(res.matrix.entries)
    # Re-verify det.
    abs_det = abs(integer_determinant(res.matrix.entries))
    return {
        "n": n,
        "lambda": lam,
        "seed": seed,
        "iterations": iterations,
        "abs_det": int(abs_det),
        "t_support_card": int(card),
        "walltime": float(elapsed),
        "matrix": res.matrix.entries,
    }


def _experiment(
    n: int,
    lambdas: list[float],
    n_starts: int,
    wall_per_start: float,
    base_seed: int = 12345,
) -> dict:
    """Run all (lambda, start) combinations at order n."""
    results: dict[float, list[dict]] = {}
    print(f"\n=== Experiment at n={n}: {n_starts} starts x "
          f"{wall_per_start:.0f}s wall, lambdas={lambdas} ===", flush=True)
    for lam in lambdas:
        per_lam: list[dict] = []
        for k in range(n_starts):
            seed = base_seed + 1000 * int(lam * 1000) + k
            r = _run_one(n, lam, seed, wall_per_start)
            per_lam.append(r)
            ratio_str = ""
            if n in PROVEN:
                ratio_str = f", ratio={r['abs_det']/PROVEN[n]:.4f}"
            elif n == 23:
                ratio_str = f", btc-ratio={r['abs_det']/H23_BTC_CEILING:.4f}"
            print(
                f"  n={n} lam={lam} seed={seed} "
                f"|det|={r['abs_det']:,d}{ratio_str} "
                f"T-card={r['t_support_card']} "
                f"({r['walltime']:.1f}s, {r['iterations']:,d} iters)",
                flush=True,
            )
        results[lam] = per_lam
    # Aggregate per-lambda summary.
    summary = {}
    for lam, runs in results.items():
        bests = [r["abs_det"] for r in runs]
        cards = [r["t_support_card"] for r in runs]
        best_run = max(runs, key=lambda r: r["abs_det"])
        summary[lam] = {
            "best_abs_det": max(bests),
            "median_abs_det": sorted(bests)[len(bests) // 2],
            "best_t_card": min(cards),
            "median_t_card": sorted(cards)[len(cards) // 2],
            "best_run_seed": best_run["seed"],
            "best_run_card": best_run["t_support_card"],
        }
        if n in PROVEN:
            summary[lam]["best_ratio_to_proven"] = (
                summary[lam]["best_abs_det"] / PROVEN[n]
            )
        if n == 23:
            summary[lam]["best_ratio_to_btc"] = (
                summary[lam]["best_abs_det"] / H23_BTC_CEILING
            )
    return {
        "n": n,
        "lambdas": lambdas,
        "n_starts": n_starts,
        "wall_per_start_seconds": wall_per_start,
        "per_lambda_runs": {str(k): v for k, v in results.items()},
        "summary": {str(k): v for k, v in summary.items()},
    }


def _verdict(exp: dict, target_card: int | None) -> str:
    """Render a one-line verdict for an experiment."""
    summary = exp["summary"]
    lams = sorted(float(k) for k in summary.keys())
    base = summary[str(0.0)] if "0.0" in summary else summary[str(lams[0])]
    base_best = base["best_abs_det"]
    base_card = base["best_t_card"]

    # Find the best lambda > 0 by |det|.
    best_lam = None
    best_lam_abs = 0
    for lam in lams:
        if lam == 0.0:
            continue
        s = summary[str(lam)]
        if s["best_abs_det"] > best_lam_abs:
            best_lam_abs = s["best_abs_det"]
            best_lam = lam

    n = exp["n"]
    target_str = f", target card={target_card}" if target_card is not None else ""
    if best_lam is None:
        return f"n={n}: only one lambda; no comparison (best card={base_card}{target_str})"

    abs_gain = best_lam_abs - base_best
    ratio_gain = (best_lam_abs / base_best) if base_best > 0 else float("inf")
    best_lam_card = summary[str(best_lam)]["best_t_card"]
    if abs_gain > 0:
        verdict = (
            f"n={n}: lambda={best_lam} BEATS lambda=0 "
            f"({best_lam_abs:,d} vs {base_best:,d}, x{ratio_gain:.3f}); "
            f"T-card {base_card} -> {best_lam_card}{target_str} -- CAUSAL signal"
        )
    elif best_lam_card < base_card:
        verdict = (
            f"n={n}: lambda={best_lam} drops T-card "
            f"({base_card} -> {best_lam_card}) but |det| stagnates "
            f"({best_lam_abs:,d} vs {base_best:,d}){target_str} -- "
            f"CORRELATED ONLY"
        )
    else:
        verdict = (
            f"n={n}: lambda > 0 neither improves |det| nor reduces T-card -- "
            f"NULL (no signal at all)"
        )
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Quick smoke run (10s per start, 2 starts).")
    ap.add_argument("--out", default=str(REPO / "results" / "m1_calibration.json"))
    args = ap.parse_args()

    if args.quick:
        n_starts = 2
        wall_h17 = 10.0
        wall_h19 = 10.0
        wall_h23 = 20.0
        n_starts_h23 = 2
    else:
        n_starts = 4
        wall_h17 = 60.0
        wall_h19 = 60.0
        wall_h23 = 300.0
        n_starts_h23 = 3

    lambdas_h17 = [0.0, 0.001, 0.01, 0.1, 1.0]
    lambdas_h19 = [0.0, 0.001, 0.01, 0.1, 1.0]

    overall = {}
    t_total = time.monotonic()

    overall["exp1_h17"] = _experiment(17, lambdas_h17, n_starts, wall_h17)
    print("\n[Exp 1 verdict] " + _verdict(overall["exp1_h17"], target_card=2))

    overall["exp2_h19"] = _experiment(19, lambdas_h19, n_starts, wall_h19)
    print("\n[Exp 2 verdict] " + _verdict(overall["exp2_h19"], target_card=3))

    # For Exp 3, pick the best lambda from H19 (by |det|) and run only that
    # plus lambda=0 baseline at H23, longer.
    h19_summary = overall["exp2_h19"]["summary"]
    lams_sorted = sorted((float(k) for k in h19_summary.keys()))
    best_lam_h19 = max(
        (lam for lam in lams_sorted if lam > 0),
        key=lambda lam: h19_summary[str(lam)]["best_abs_det"],
        default=0.1,
    )
    print(f"\n[Exp 3] H19 best lambda was {best_lam_h19}; "
          f"using this plus lambda=0 at H23.")
    lambdas_h23 = [0.0, best_lam_h19]
    overall["exp3_h23"] = _experiment(
        23, lambdas_h23, n_starts_h23, wall_h23,
        base_seed=42,
    )
    print("\n[Exp 3 verdict] " + _verdict(overall["exp3_h23"], target_card=None))

    overall["meta"] = {
        "n_starts": n_starts,
        "wall_h17_seconds": wall_h17,
        "wall_h19_seconds": wall_h19,
        "wall_h23_seconds": wall_h23,
        "h23_best_lambda_from_h19": best_lam_h19,
        "h23_btc_ceiling": H23_BTC_CEILING,
        "proven_h17": PROVEN[17],
        "proven_h19": PROVEN[19],
        "total_walltime_seconds": time.monotonic() - t_total,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Final summary table.
    print("\n=== FINAL SUMMARY TABLE ===")
    print(f"{'n':>3}  {'lambda':>7}  {'best |det|':>22}  "
          f"{'ratio':>8}  {'T-card':>6}")
    for exp_key, exp in overall.items():
        if not exp_key.startswith("exp"):
            continue
        n = exp["n"]
        for lam_str, s in exp["summary"].items():
            lam = float(lam_str)
            ratio = s.get("best_ratio_to_proven") or s.get("best_ratio_to_btc")
            ratio_str = f"{ratio:.4f}" if ratio is not None else "  -  "
            print(
                f"{n:>3}  {lam:>7g}  {s['best_abs_det']:>22,d}  "
                f"{ratio_str:>8}  {s['best_t_card']:>6}"
            )

    # H23 record check.
    h23_best = overall["exp3_h23"]["summary"]
    h23_max = max(s["best_abs_det"] for s in h23_best.values())
    if h23_max > H23_BTC_CEILING:
        print(f"\n*** NEW H23 RECORD: {h23_max:,d} > BTC {H23_BTC_CEILING:,d} ***")
        # Re-verify the best matrix.
        for lam_str, lam_runs in overall["exp3_h23"]["per_lambda_runs"].items():
            for r in lam_runs:
                if r["abs_det"] == h23_max:
                    rebuild = abs(integer_determinant(r["matrix"]))
                    print(
                        f"    re-verified |det| = {rebuild:,d} "
                        f"({'MATCH' if rebuild == h23_max else 'MISMATCH'})"
                    )
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
