"""Verify a submission JSON file against the reference verifier.

Usage:
    python scripts/verify_submission.py results/H7_best.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hadamard_maxdet import BENCHMARKS, verify
from hadamard_maxdet.io import read_submission


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    order_id, n, entries = read_submission(args.path)
    verdict = verify(entries)
    print(f"File:     {args.path}")
    print(f"Order:    {order_id}  (n={n})")
    print(f"Valid:    {verdict.valid}")
    if not verdict.valid:
        print(f"Error:    {verdict.error}")
        return 1
    print(f"det:      {verdict.det}")
    print(f"|det|:    {verdict.abs_det}")

    if order_id in BENCHMARKS:
        order = BENCHMARKS[order_id]
        if order.best_known is not None:
            ratio = verdict.abs_det / order.best_known
            status = "matches" if verdict.abs_det == order.best_known else (
                "beats" if verdict.abs_det > order.best_known else "below"
            )
            print(f"Best known: {order.best_known}  (result {status}, ratio {ratio:.4f})")
        else:
            print("Best known: (open)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
