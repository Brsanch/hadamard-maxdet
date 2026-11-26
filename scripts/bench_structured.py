"""Benchmark structured-ansatz SA against unstructured SA on H19.

For a fixed wall-clock budget (default 120 s per method), run each
search and report the best |det| found.

Methods evaluated on H19 (n = 19, open problem, 2n-1 = 37 not a square
so Barba is not tight):

  unstructured
      ``multi_start`` from ``hadamard_maxdet.search`` over the full
      2^(19*19) = 2^361 ±1 lattice. This is the apples-to-apples
      baseline: same move kernel, same scoring, just no structural
      restriction.

  bordered_circulant
      SA over 2^18 = 262,144 cores. Block starts are the all-ones
      core and random cores.

  skew_circulant
      SA over 2^9 = 512 free-bit vectors (n=19 odd → (n-1)/2 = 9
      free bits). This is small enough to enumerate exhaustively in
      well under the budget; the SA call nonetheless burns its time
      budget so the comparison is time-bounded, not iteration-bounded.

  two_circulant: N/A (requires even n; n = 19 is odd).

All |det| values are exact integers via Bareiss; no floating-point
is used for scoring.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hadamard_maxdet import BENCHMARKS, verify
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.search import multi_start
from hadamard_maxdet.structured import (
    bordered_circulant,
    sa_bordered_circulant,
    sa_skew_circulant,
    skew_generator_from_free_bits,
)
from hadamard_maxdet.verifier import integer_determinant


def run_unstructured_budget(
    n: int, budget_s: float, base_seed: int = 0
) -> dict:
    """Repeatedly launch short SA runs until budget_s elapses; return best."""
    start = time.monotonic()
    best_abs = 0
    best_entries: list[list[int]] | None = None
    starts_done = 0
    iters_per_start = 5_000
    while time.monotonic() - start < budget_s:
        r = multi_start(
            n,
            n_starts=1,
            iterations=iters_per_start,
            base_seed=base_seed + starts_done,
        )
        starts_done += 1
        if r.abs_det > best_abs:
            best_abs = r.abs_det
            best_entries = [row[:] for row in r.matrix.entries]
    elapsed = time.monotonic() - start
    return {
        "method": "unstructured",
        "abs_det": best_abs,
        "elapsed_s": elapsed,
        "starts_done": starts_done,
        "iters_per_start": iters_per_start,
        "entries": best_entries,
    }


def run_bordered_budget(n: int, budget_s: float, base_seed: int = 0) -> dict:
    start = time.monotonic()
    best_abs = 0
    best_entries: list[list[int]] | None = None
    starts_done = 0
    iters_per_start = 20_000
    while time.monotonic() - start < budget_s:
        r = sa_bordered_circulant(
            n,
            iterations=iters_per_start,
            seed=base_seed + starts_done,
        )
        starts_done += 1
        if r.abs_det > best_abs:
            best_abs = r.abs_det
            best_entries = [row[:] for row in r.matrix.entries]
    elapsed = time.monotonic() - start
    return {
        "method": "bordered_circulant",
        "abs_det": best_abs,
        "elapsed_s": elapsed,
        "starts_done": starts_done,
        "iters_per_start": iters_per_start,
        "entries": best_entries,
    }


def run_skew_budget(n: int, budget_s: float, base_seed: int = 0) -> dict:
    if n % 2 == 0:
        return {
            "method": "skew_circulant",
            "abs_det": None,
            "elapsed_s": 0.0,
            "note": "N/A for even n",
        }
    start = time.monotonic()
    best_abs = 0
    best_entries: list[list[int]] | None = None
    starts_done = 0
    iters_per_start = 5_000
    while time.monotonic() - start < budget_s:
        r = sa_skew_circulant(
            n,
            iterations=iters_per_start,
            seed=base_seed + starts_done,
        )
        starts_done += 1
        if r.abs_det > best_abs:
            best_abs = r.abs_det
            best_entries = [row[:] for row in r.matrix.entries]
    elapsed = time.monotonic() - start
    return {
        "method": "skew_circulant",
        "abs_det": best_abs,
        "elapsed_s": elapsed,
        "starts_done": starts_done,
        "iters_per_start": iters_per_start,
        "entries": best_entries,
    }


def skew_exhaustive(n: int) -> dict:
    """For small (n-1)/2, enumerate every skew-circulant and return the max."""
    if n % 2 == 0:
        return {"method": "skew_exhaustive", "abs_det": None, "note": "N/A for even n"}
    half = (n - 1) // 2
    if half > 20:
        return {"method": "skew_exhaustive", "abs_det": None, "note": f"skip: 2^{half} too large"}
    start = time.monotonic()
    best = 0
    best_bits: tuple[int, ...] | None = None
    for bits in itertools.product([-1, 1], repeat=half):
        gen = skew_generator_from_free_bits(n, list(bits))
        d = abs(integer_determinant(Matrix.circulant(gen).entries))
        if d > best:
            best = d
            best_bits = bits
    elapsed = time.monotonic() - start
    return {
        "method": "skew_exhaustive",
        "abs_det": best,
        "elapsed_s": elapsed,
        "enumerated": 2 ** half,
        "best_bits": list(best_bits) if best_bits is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", default="H19", choices=list(BENCHMARKS))
    parser.add_argument("--budget", type=float, default=120.0,
                        help="wall-clock budget per method, in seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/bench_structured.json"))
    args = parser.parse_args()

    order = BENCHMARKS[args.order]
    n = order.n
    print(f"Benchmarking structured vs. unstructured SA on {order.id} (n={n}).")
    if order.best_known is not None:
        print(f"Known target: |det| = {order.best_known} ({'proven' if order.proven_optimal else 'best-known'}).")
    else:
        print("Open case — comparing local lower bounds.")
    print(f"Per-method wall-clock budget: {args.budget}s.\n")

    results = []

    print("=== unstructured multi-start SA ===")
    res_u = run_unstructured_budget(n, args.budget, base_seed=args.seed)
    print(f"  best |det| = {res_u['abs_det']}")
    print(f"  {res_u['starts_done']} starts, {res_u['iters_per_start']} iters each, {res_u['elapsed_s']:.1f}s")
    results.append(res_u)

    print("\n=== bordered-circulant SA ===")
    res_b = run_bordered_budget(n, args.budget, base_seed=args.seed)
    print(f"  best |det| = {res_b['abs_det']}")
    print(f"  {res_b['starts_done']} starts, {res_b['iters_per_start']} iters each, {res_b['elapsed_s']:.1f}s")
    # Verify.
    if res_b["entries"] is not None:
        v = verify(res_b["entries"])
        assert v.valid and v.abs_det == res_b["abs_det"]
    results.append(res_b)

    if n % 2 == 0:
        print("\n=== two-circulant SA ===")
        print("  N/A: two-circulant block form [[A,B],[B^T,-A^T]] requires even n "
              f"(n={n} is odd).")
    else:
        print("\n=== two-circulant SA ===")
        print(f"  N/A: two-circulant block form [[A,B],[B^T,-A^T]] requires even n "
              f"(n={n} is odd). No even-n two-circulant analog applies to H19.")

    print("\n=== skew-circulant SA ===")
    res_s = run_skew_budget(n, args.budget, base_seed=args.seed)
    if res_s["abs_det"] is not None:
        print(f"  best |det| = {res_s['abs_det']}")
        print(f"  {res_s['starts_done']} starts, {res_s['iters_per_start']} iters each, {res_s['elapsed_s']:.1f}s")
        if res_s.get("entries") is not None:
            v = verify(res_s["entries"])
            assert v.valid and v.abs_det == res_s["abs_det"]
    else:
        print(f"  {res_s['note']}")
    results.append(res_s)

    # For odd small n, also show the skew-ansatz ceiling by exhaustive enumeration.
    if n % 2 == 1 and (n - 1) // 2 <= 22:
        print("\n=== skew-circulant EXHAUSTIVE (ansatz ceiling) ===")
        res_e = skew_exhaustive(n)
        print(f"  max |det| over all 2^{(n-1)//2} = {res_e.get('enumerated')} skew-generators: {res_e['abs_det']}")
        print(f"  elapsed {res_e['elapsed_s']:.1f}s")
        results.append(res_e)

    print("\n=== Summary ===")
    labels = [("unstructured", res_u['abs_det']),
              ("bordered_circulant", res_b['abs_det']),
              ("skew_circulant", res_s.get('abs_det'))]
    for label, v in labels:
        print(f"  {label:24s} |det| = {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Drop heavy entries for JSON summary; keep the best ones.
    summary = {
        "order": order.id,
        "n": n,
        "budget_s": args.budget,
        "seed": args.seed,
        "results": [
            {k: v for k, v in r.items() if k != "entries"}
            for r in results
        ],
    }
    # Keep the best matrix (whichever method wins).
    best = max(
        (r for r in results if r.get("abs_det") is not None),
        key=lambda r: r["abs_det"],
    )
    if best.get("entries") is not None:
        summary["best_method"] = best["method"]
        summary["best_abs_det"] = best["abs_det"]
        summary["best_entries"] = best["entries"]
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
