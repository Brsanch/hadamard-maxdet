"""Top-level solver entry point.

This file is the intended locus of iteration: rewrite or extend ``solve``
below to try new strategies (different move operators, different starts,
rank-1 det updates, learned guidance, etc.). The library code under
``src/hadamard_maxdet/`` is deliberately kept stable: verifier, orders,
baseline moves, submission I/O.

The current implementation is a plain multi-start simulated-annealing
baseline. It is not tuned.

Run:
    python solver.py --order H7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hadamard_maxdet import BENCHMARKS
from hadamard_maxdet.io import write_submission
from hadamard_maxdet.search import AnnealResult, multi_start


def solve(
    order_id: str,
    *,
    iterations: int = 200_000,
    starts: int = 8,
    t_start: float = 1.0,
    t_end: float = 0.01,
    base_seed: int = 0,
) -> AnnealResult:
    """Baseline solver: multi-start simulated annealing.

    Rewrite or replace this function to try new search strategies.
    """
    order = BENCHMARKS[order_id]
    return multi_start(
        order.n,
        n_starts=starts,
        iterations=iterations,
        t_start=t_start,
        t_end=t_end,
        base_seed=base_seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", default="H7", choices=list(BENCHMARKS))
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    order = BENCHMARKS[args.order]
    print(f"Solving {order.id} (n={order.n}, class r{order.residue}).")
    if order.best_known is not None:
        label = "proven optimal" if order.proven_optimal else "best known"
        print(f"{label.capitalize()}: |det| = {order.best_known}.")
    else:
        print("Best known: (open — target is a new lower bound).")

    t0 = time.monotonic()
    result = solve(
        args.order,
        iterations=args.iterations,
        starts=args.starts,
        base_seed=args.seed,
    )
    wall = time.monotonic() - t0

    print(f"Best found: |det| = {result.abs_det} in {wall:.2f}s.")
    if order.best_known is not None:
        ratio = result.abs_det / order.best_known
        print(f"Ratio to best known: {ratio:.4f}.")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{order.id}_best.json"
    write_submission(result.matrix, order.id, path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
