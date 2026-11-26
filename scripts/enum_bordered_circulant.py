"""Exhaustive bordered-circulant enumeration across small n.

For each n in the configured list, enumerate every distinct
bordered-circulant core (up to cyclic-rotation equivalence) and report
the ansatz maximum |det|. The ratio column is best/proven (when known)
or best/SA (when only an SA value is known).

Cores larger than ~2^MAX_LOG2_TOTAL are skipped (current cap: 2^22 =
4,194,304). To run a specific n that's been skipped, lower or raise
the cap with ``--max-log2``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow running from repo root without install.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from hadamard_maxdet.enumerate import enumerate_bordered_circulant


# Reference values to print alongside the ansatz max. For proven n we use
# OEIS A003432 (matches src/hadamard_maxdet/orders.py). For open n we use
# the Round-1 SA values reported in docs/findings.md (2026-04-24 entry).
REFERENCE: dict[int, tuple[str, int | None]] = {
    13: ("proven (OEIS A003432)", 14_929_920),
    15: ("proven (Orrick 2005)", None),  # 25,389,000,576 if cross-checked
    17: ("proven (OEIS A003432)", None),
    19: ("SA round-1 (bordered)", 784_859_922_432),
    21: ("SA round-1 (parallel)", None),
    22: ("SA round-1", None),
    23: ("SA round-1", None),
}

DEFAULT_NS = (13, 15, 17, 19, 21, 22, 23)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-log2",
        type=int,
        default=22,
        help="skip n where 2^(n-1) exceeds 2^MAX_LOG2 (default 22 = 4M)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="multiprocessing pool size (default: min(8, cpu_count))",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=list(DEFAULT_NS),
        help=f"orders to enumerate (default {list(DEFAULT_NS)})",
    )
    parser.add_argument(
        "--single-thread",
        action="store_true",
        help="disable multiprocessing (for debugging)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/enum_bordered_circulant.json",
        help="path to write JSON summary",
    )
    args = parser.parse_args()

    cap = 1 << args.max_log2

    print(
        f"{'n':>3}  {'size':>10}  {'reference':>32}  {'ansatz_max':>20}  "
        f"{'ratio':>8}  {'walltime':>10}"
    )
    print("-" * 92)

    rows: list[dict] = []
    for n in args.ns:
        size = 1 << (n - 1)
        if size > cap:
            print(
                f"{n:>3}  {size:>10}  {'(skipped: > 2^' + str(args.max_log2) + ')':>32}"
            )
            rows.append(
                {
                    "n": n,
                    "size": size,
                    "skipped": True,
                    "reason": f"2^(n-1) > 2^{args.max_log2}",
                }
            )
            continue

        ref_label, ref_val = REFERENCE.get(n, ("(no reference)", None))
        t0 = time.monotonic()
        r = enumerate_bordered_circulant(
            n,
            use_multiprocessing=not args.single_thread,
            n_workers=args.workers,
        )
        wall = time.monotonic() - t0

        ratio_str = "n/a" if ref_val is None else f"{r.best_det / ref_val:.4f}"
        ref_show = (
            f"{ref_label}={ref_val}" if ref_val is not None else f"{ref_label}"
        )
        print(
            f"{n:>3}  {size:>10}  {ref_show[:32]:>32}  {r.best_det:>20}  "
            f"{ratio_str:>8}  {wall:>9.2f}s"
        )

        rows.append(
            {
                "n": n,
                "size": size,
                "visited": r.visited,
                "ansatz_max": r.best_det,
                "best_core": r.best_core,
                "walltime": wall,
                "reference_label": ref_label,
                "reference_value": ref_val,
                "ratio": None if ref_val is None else r.best_det / ref_val,
                "skipped": False,
            }
        )

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"rows": rows, "args": vars(args)}, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
