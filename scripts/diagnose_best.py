"""Diagnose the best ±1 matrix we have on disk and print the Gram fingerprint.

Fingerprint = Gram off-diagonal distribution + singular-value multiplicities +
eigenvalue product check. This tells us:

  * which inner-product values the matrix is stuck at (support of the
    off-diagonal distribution: if support is tight like {-1, +3}, the
    matrix is in an Ehlich-type equivalence class and local moves
    preserve that class),
  * the symmetry group acting on rows (from sv multiplicities),
  * the gap to Hadamard (sv max/min ratio; ideally 1).

Usage:
    python scripts/diagnose_best.py  results/bench_structured.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.telemetry import gram_offdiag_report  # noqa: E402
from hadamard_maxdet.verifier import integer_determinant  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    data = json.load(open(args.path))
    entries = data.get("best_entries") or data.get("entries")
    if entries is None:
        raise SystemExit(f"No matrix in {args.path}; expected 'best_entries' or 'entries'.")

    det = integer_determinant(entries)
    fp = gram_offdiag_report(entries)
    n = fp["n"]

    print(f"Matrix from {args.path}")
    print(f"  n = {n}, |det| = {abs(det):,d}")
    print()
    print("Gram off-diagonal distribution:")
    total = sum(fp["gram_offdiag_counts"].values())
    for v, c in fp["gram_offdiag_counts"].items():
        pct = 100 * c / total
        bar = "#" * int(60 * c / total)
        print(f"  <v_i, v_j> = {v:+4d}: {c:5d}  ({pct:5.2f}%) {bar}")
    print(f"  support = {fp['gram_offdiag_support']}")
    print()
    print(f"Singular-value spectrum: max={fp['sv_max']:.4f}, min={fp['sv_min']:.4f}")
    print(f"  ratio max/min = {fp['sv_ratio']:.4f}  "
          f"(Hadamard = 1.0; we are at {n**0.5 / fp['sv_min']:.2f}x of min)")
    print(f"Singular-value multiplicities (non-trivial clusters):")
    clusters = [(s, c) for s, c in fp["sv_multiplicities"] if c > 1]
    if clusters:
        for s, c in clusters:
            print(f"  sigma ≈ {s:.4f}  multiplicity {c}")
    else:
        print("  (all distinct)")
    print()
    # Heuristic: diagnose
    off = fp["gram_offdiag_support"]
    if len(off) == 2:
        print("DIAGNOSIS: two-valued Gram off-diagonal → matrix lies in a "
              "narrow Ehlich/Barba equivalence class. Single-entry flips "
              "preserve this class except at boundaries. Likely a local-"
              "ansatz ceiling, not a global minimum.")
    elif len(off) <= 4:
        print("DIAGNOSIS: narrow Gram support → strong symmetry structure. "
              "Moves that break the symmetry (row/col permutations, "
              "block switches) will explore outside this class.")
    else:
        print("DIAGNOSIS: broad Gram support → probably not a local optimum yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
