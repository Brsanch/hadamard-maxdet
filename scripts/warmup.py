"""Warm-up sanity check: Sylvester H_4 has |det| = 16 = 4^(4/2).

Run:  python scripts/warmup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hadamard_maxdet import Matrix, verify


def main() -> int:
    H4 = Matrix.sylvester(4)
    verdict = verify(H4.entries)
    print("Sylvester H_4:")
    for row in H4.entries:
        print("  " + " ".join(f"{v:+d}" for v in row))
    print(f"Valid:   {verdict.valid}")
    if not verdict.valid:
        print(f"Error:   {verdict.error}")
        return 1
    print(f"det:     {verdict.det}")
    print(f"|det|:   {verdict.abs_det}")
    assert verdict.abs_det == 16, "warm-up expects |det| = 16 (Hadamard bound)"
    print("\nOK: warm-up verifier agrees with Hadamard's bound at n=4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
