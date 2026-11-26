"""Run the multi-start SA baseline on a benchmark order.

Thin wrapper around :mod:`solver.solve` that prints progress and writes a
submission JSON. For iteration, edit ``solver.py`` directly.

Usage:
    python scripts/run_baseline.py --order H7 --iterations 200000 --starts 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hadamard_maxdet import BENCHMARKS
from hadamard_maxdet.io import write_submission
from hadamard_maxdet.search import multi_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", default="H7", choices=list(BENCHMARKS))
    ap.add_argument("--iterations", type=int, default=100_000)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("results"))
    args = ap.parse_args()

    order = BENCHMARKS[args.order]
    print(f"Running SA baseline on {order.id} (n={order.n}).")
    t0 = time.monotonic()
    result = multi_start(
        order.n,
        n_starts=args.starts,
        iterations=args.iterations,
        base_seed=args.seed,
    )
    wall = time.monotonic() - t0
    print(f"Best |det|: {result.abs_det} ({result.iterations} iters, {wall:.2f}s, "
          f"{result.accepted} accepted)")

    if order.best_known is not None:
        print(f"Best known: {order.best_known}  (ratio {result.abs_det / order.best_known:.4f})")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{order.id}_sa_{args.iterations}x{args.starts}_seed{args.seed}.json"
    write_submission(result.matrix, order.id, path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
