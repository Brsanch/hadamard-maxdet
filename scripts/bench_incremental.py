"""Benchmark: baseline SA vs rank-1 incremental SA at equal wall-clock time.

Runs both solvers for a fixed wall-clock budget on each target order and
reports iterations/sec and best |det| attained. Uses single-start for each
to keep the timing comparison clean — use ``scripts/run_baseline.py`` for
multistart production runs.

Targets by default:
  * H11 — proven max 327680
  * H15 — open (best-known on Orrick's site, not inlined here)
  * H19 — open

Usage:

    python scripts/bench_incremental.py
    python scripts/bench_incremental.py --budget 30 --orders H11 H15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.incremental import sa_incremental  # noqa: E402
from hadamard_maxdet.orders import BENCHMARKS  # noqa: E402
from hadamard_maxdet.search import simulated_annealing  # noqa: E402


def _run_for_budget(fn, n: int, budget_s: float, seed: int = 0, **kwargs):
    """Run ``fn(n, iterations=?, seed=seed, **kwargs)`` so wall clock ≈ budget.

    We first calibrate on a short run (≤ 1 s), then extrapolate iterations
    such that the real run lasts ~ ``budget_s``. Returns ``(result, rate)``.
    """
    # Calibration pass.
    cal_iters = 1000
    while True:
        t0 = time.monotonic()
        r = fn(n, iterations=cal_iters, seed=seed + 10_000, **kwargs)
        dt = time.monotonic() - t0
        if dt > 0.2 or cal_iters >= 200_000:
            break
        cal_iters *= 4
    rate = cal_iters / max(dt, 1e-9)
    target_iters = max(1000, int(rate * budget_s))

    t0 = time.monotonic()
    r = fn(n, iterations=target_iters, seed=seed, **kwargs)
    real_dt = time.monotonic() - t0
    real_rate = r.iterations / max(real_dt, 1e-9)
    return r, real_rate, real_dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=60.0, help="seconds per solver")
    ap.add_argument(
        "--orders",
        nargs="*",
        default=["H11", "H19"],
        help="order ids to bench (default: H11 H19)",
    )
    ap.add_argument("--t-start", type=float, default=1.5)
    ap.add_argument("--t-end", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(
        f"Benchmark: baseline vs incremental SA, {args.budget:.0f}s each, "
        f"t ∈ [{args.t_end}, {args.t_start}], seed={args.seed}"
    )
    print("-" * 78)

    for oid in args.orders:
        if oid not in BENCHMARKS:
            print(f"!! unknown order {oid}; skipping")
            continue
        order = BENCHMARKS[oid]
        n = order.n
        best_known = order.best_known
        proven = best_known if order.proven_optimal else None

        print(
            f"\n=== {oid} (n={n}) ===  "
            f"proven={proven}  best_known={best_known}"
        )

        base_r, base_rate, base_dt = _run_for_budget(
            simulated_annealing,
            n,
            args.budget,
            seed=args.seed,
            t_start=args.t_start,
            t_end=args.t_end,
        )
        print(
            f"  baseline     : |det|={base_r.abs_det:>15d}  "
            f"iters={base_r.iterations:>10d}  "
            f"{base_rate:>10.0f} it/s  ({base_dt:.1f}s)"
        )

        inc_r, inc_rate, inc_dt = _run_for_budget(
            sa_incremental,
            n,
            args.budget,
            seed=args.seed,
            t_start=args.t_start,
            t_end=args.t_end,
        )
        extra = (
            f"  refresh_hits={inc_r.residual_triggered}"
            if hasattr(inc_r, "residual_triggered")
            else ""
        )
        print(
            f"  incremental  : |det|={inc_r.abs_det:>15d}  "
            f"iters={inc_r.iterations:>10d}  "
            f"{inc_rate:>10.0f} it/s  ({inc_dt:.1f}s){extra}"
        )

        speedup = inc_rate / max(base_rate, 1e-9)
        abs_ratio = inc_r.abs_det / max(base_r.abs_det, 1)
        print(f"  speedup      : {speedup:.1f}×   |det|_inc/|det|_base: {abs_ratio:.3f}×")
        if proven is not None:
            print(
                f"  vs proven    : baseline {base_r.abs_det/proven:.3f}   "
                f"incremental {inc_r.abs_det/proven:.3f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
