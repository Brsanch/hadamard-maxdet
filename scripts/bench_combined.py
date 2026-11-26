"""Head-to-head on H19: baseline vs incremental vs bordered-circulant.

Equal wall-clock budget per method. Multi-start for all, re-seeded.
All scores are exact integer determinants.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.incremental import sa_incremental  # noqa: E402
from hadamard_maxdet.search import simulated_annealing  # noqa: E402
from hadamard_maxdet.structured import sa_bordered_circulant  # noqa: E402


def run_until_budget(label, factory, budget_s, n):
    t0 = time.monotonic()
    best = 0
    starts = 0
    iters_per_start = 20_000
    while time.monotonic() - t0 < budget_s:
        r = factory(seed=starts, iterations=iters_per_start)
        if r.abs_det > best:
            best = r.abs_det
        starts += 1
    dt = time.monotonic() - t0
    print(f"  {label:<28s} |det| = {best:>15}  starts={starts:>3d}  dt={dt:.1f}s")
    return best


def main():
    n = 19
    budget = 45.0
    print(f"H19 head-to-head, {budget}s per method, multi-start.\n")

    baseline = lambda seed, iterations: simulated_annealing(
        n, iterations=iterations, seed=seed,
    )
    incremental = lambda seed, iterations: sa_incremental(
        n, iterations=iterations, seed=seed,
    )
    bordered = lambda seed, iterations: sa_bordered_circulant(
        n, iterations=iterations, seed=seed,
    )

    results = {}
    results["baseline"]    = run_until_budget("baseline SA",           baseline,    budget, n)
    results["incremental"] = run_until_budget("incremental SA",        incremental, budget, n)
    results["bordered"]    = run_until_budget("bordered-circulant SA", bordered,    budget, n)
    print()
    best_method = max(results, key=results.get)
    print(f"Winner: {best_method} with |det|={results[best_method]}")


if __name__ == "__main__":
    main()
