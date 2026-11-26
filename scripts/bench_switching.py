"""Head-to-head on H19: plain SA vs switch-enhanced SA.

Equal wall-clock budget per method, multi-start, exact integer determinants.
The plain SA uses :func:`hadamard_maxdet.search.simulated_annealing`
(entry-flip-only proposals via the default `random_move` mix). The
switch-enhanced SA uses :func:`hadamard_maxdet.switching.sa_with_switching`
mixing 90% entry_flip + 5% row-pair block switch + 5% 2x2 block flip.

Both score via Bareiss (full O(n^3) recompute) so iteration rates are
comparable; the experiment isolates the *move set* effect, not the scoring
path. The previously-shipped `sa_incremental` is much faster but only
on entry_flip; comparing it to a non-local proposer would conflate moves
with scoring.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.search import simulated_annealing  # noqa: E402
from hadamard_maxdet.switching import sa_with_switching  # noqa: E402


def run_until_budget(label, factory, budget_s):
    t0 = time.monotonic()
    best = 0
    starts = 0
    total_iters = 0
    iters_per_start = 5_000
    while time.monotonic() - t0 < budget_s:
        remaining = budget_s - (time.monotonic() - t0)
        if remaining <= 0:
            break
        r = factory(seed=starts, iterations=iters_per_start, walltime_budget=remaining)
        total_iters += getattr(r, "iterations", iters_per_start)
        if r.abs_det > best:
            best = r.abs_det
        starts += 1
    dt = time.monotonic() - t0
    print(
        f"  {label:<32s} |det| = {best:>18,}  starts={starts:>3d}  iters={total_iters:>7d}  dt={dt:.1f}s"
    )
    return {
        "label": label,
        "best_abs_det": best,
        "starts": starts,
        "iterations": total_iters,
        "walltime_seconds": dt,
    }


def main():
    n = 19
    budget = 45.0
    print(f"H19 switching bench: {budget}s per method, multi-start, single thread.\n")

    def baseline(seed, iterations, walltime_budget=None):
        # The baseline simulated_annealing doesn't take walltime_budget; we
        # honour it by chopping iterations to roughly fit. At ~3.5k iter/s
        # for n=19, 5000 iters ≈ 1.4s, which is short enough that overshoot
        # of `walltime_budget` per start is bounded.
        if walltime_budget is not None and walltime_budget < 1.0:
            iterations = max(200, int(iterations * walltime_budget))
        return simulated_annealing(
            n,
            iterations=iterations,
            t_start=1.5,
            t_end=0.005,
            seed=seed,
        )

    def switched(seed, iterations, walltime_budget=None):
        return sa_with_switching(
            n,
            iterations=iterations,
            t_start=1.5,
            t_end=0.005,
            seed=seed,
            walltime_budget=walltime_budget,
            entry_weight=0.90,
            block_switch_weight=0.05,
            twobytwo_switch_weight=0.05,
        )

    results = {
        "baseline_SA":  run_until_budget("baseline SA (entry+xor+swap)", baseline, budget),
        "switching_SA": run_until_budget("switch-enhanced SA",            switched, budget),
    }
    print()
    if results["switching_SA"]["best_abs_det"] > results["baseline_SA"]["best_abs_det"]:
        delta = results["switching_SA"]["best_abs_det"] - results["baseline_SA"]["best_abs_det"]
        ratio = results["switching_SA"]["best_abs_det"] / max(1, results["baseline_SA"]["best_abs_det"])
        print(f"Switch-enhanced wins by {delta:,} (ratio {ratio:.4f}).")
    elif results["switching_SA"]["best_abs_det"] < results["baseline_SA"]["best_abs_det"]:
        delta = results["baseline_SA"]["best_abs_det"] - results["switching_SA"]["best_abs_det"]
        ratio = results["baseline_SA"]["best_abs_det"] / max(1, results["switching_SA"]["best_abs_det"])
        print(f"Baseline wins by {delta:,} (ratio {ratio:.4f}). Switching hurt SA at this budget.")
    else:
        print("Tie.")

    out_path = Path(__file__).resolve().parents[1] / "results" / "bench_switching.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "n": n,
                "budget_seconds": budget,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}.")


if __name__ == "__main__":
    main()
