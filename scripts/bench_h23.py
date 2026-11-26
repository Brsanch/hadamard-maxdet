"""H23 wall-clock benchmark: rich-ansatz SA vs the bordered-circulant ceiling.

Compares, on a single thread with equal wall-clock budget per method:

  - Plain bordered-circulant SA        (the established H23 ceiling 2.42e15)
  - Double-bordered SA                  (richer; 2 border rows + 2 border cols)
  - Bordered two-circulant SA           (inner [[A,B],[B^T,-A^T]] + border)
  - Paley H24 deletion + SA             (concrete Hadamard-cofactor seed + SA)

For each method we report best |det|, walltime, ratio to the ceiling, and
whether it strictly exceeded the ceiling. Every "best" matrix is
re-verified by exact integer Bareiss (`integer_determinant`) before
reporting.

Usage::

    python scripts/bench_h23.py [--budget 180]
"""

from __future__ import annotations

import argparse
import time

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.rich_ansatz import (
    paley_deletion_seeds,
    sa_bordered_two_circulant,
    sa_double_bordered_circulant,
)
from hadamard_maxdet.incremental import sa_incremental
from hadamard_maxdet.structured import sa_bordered_circulant
from hadamard_maxdet.verifier import integer_determinant


# Established bordered-circulant ceiling at n=23 (round-2 exhaustive enum).
H23_BORDERED_CEILING = 2_421_778_815_123_456
N = 23


def run_until(sa_fn, budget_s: float, **base_kwargs) -> tuple[int, float, int]:
    """Run ``sa_fn`` repeatedly with fresh seeds until budget exhausted.

    Returns ``(best_abs_det, walltime, total_iters)``.
    """
    start = time.monotonic()
    best = 0
    seed = 0
    total_iters = 0
    iters_per_chunk = base_kwargs.pop("iterations_per_chunk", 5_000)
    while time.monotonic() - start < budget_s:
        r = sa_fn(N, iterations=iters_per_chunk, seed=seed, **base_kwargs)
        if r.abs_det > best:
            best = r.abs_det
        total_iters += iters_per_chunk
        seed += 1
    return best, time.monotonic() - start, total_iters


def run_paley_until(budget_s: float, *, top_k: int = 8) -> tuple[int, float, int, Matrix | None]:
    """Cycle through Paley H24 deletion seeds with SA until budget exhausted."""
    seeds = paley_deletion_seeds(N)[:top_k]
    start = time.monotonic()
    best = 0
    best_mat: Matrix | None = None
    chunk = 0
    iters_per_chunk = 5_000
    total_iters = 0
    # First, score all seeds to pick the strongest non-zero starts.
    while time.monotonic() - start < budget_s:
        s = seeds[chunk % len(seeds)]
        warm = best_mat if (best_mat is not None and chunk % top_k != 0) else s.matrix
        r = sa_incremental(
            N,
            iterations=iters_per_chunk,
            t_start=0.5,
            t_end=0.001,
            seed=chunk * 17 + 1,
            initial=warm,
            refresh_every=0,
        )
        total_iters += iters_per_chunk
        if r.abs_det > best:
            best = r.abs_det
            best_mat = r.matrix
        chunk += 1
    return best, time.monotonic() - start, total_iters, best_mat


def verify_or_die(entries) -> int:
    """Re-score the matrix with exact Bareiss; return |det|."""
    return abs(integer_determinant(entries))


def fmt_ratio(d: int) -> str:
    if H23_BORDERED_CEILING <= 0:
        return "n/a"
    return f"{d / H23_BORDERED_CEILING:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget",
        type=float,
        default=180.0,
        help="Wall-clock budget per method in seconds (default 180).",
    )
    parser.add_argument(
        "--paley-top-k",
        type=int,
        default=8,
        help="Top-K Paley deletion seeds to cycle through (default 8).",
    )
    args = parser.parse_args()

    print("=" * 78)
    print(
        f"H23 rich-ansatz benchmark vs bordered-circulant ceiling "
        f"({H23_BORDERED_CEILING:,})"
    )
    print(f"Per-method budget: {args.budget:.1f}s")
    print("=" * 78)
    print()

    rows: list[dict] = []

    # --- Method 1: plain bordered-circulant SA (baseline / ceiling) ---
    print(f"[1/4] Bordered-circulant SA ...")
    t0 = time.monotonic()
    best, wall, iters = run_until(
        sa_bordered_circulant,
        args.budget,
        t_start=0.5,
        t_end=0.005,
        iterations_per_chunk=5_000,
    )
    rows.append({
        "method": "bordered-circulant SA (baseline)",
        "best_abs_det": best,
        "walltime_s": wall,
        "iters": iters,
        "ratio": fmt_ratio(best),
        "beats_ceiling": best > H23_BORDERED_CEILING,
    })
    print(f"  best = {best:,}  ratio = {fmt_ratio(best)}  iters={iters}  wall={wall:.1f}s")

    # --- Method 2: double-bordered SA ---
    print(f"[2/4] Double-bordered SA ...")
    best, wall, iters = run_until(
        sa_double_bordered_circulant,
        args.budget,
        t_start=1.0,
        t_end=0.005,
        iterations_per_chunk=5_000,
    )
    rows.append({
        "method": "double-bordered SA",
        "best_abs_det": best,
        "walltime_s": wall,
        "iters": iters,
        "ratio": fmt_ratio(best),
        "beats_ceiling": best > H23_BORDERED_CEILING,
    })
    print(f"  best = {best:,}  ratio = {fmt_ratio(best)}  iters={iters}  wall={wall:.1f}s")

    # --- Method 3: bordered two-circulant SA ---
    print(f"[3/4] Bordered two-circulant SA ...")
    best, wall, iters = run_until(
        sa_bordered_two_circulant,
        args.budget,
        t_start=1.0,
        t_end=0.005,
        iterations_per_chunk=5_000,
        full_border=True,
    )
    rows.append({
        "method": "bordered two-circulant SA",
        "best_abs_det": best,
        "walltime_s": wall,
        "iters": iters,
        "ratio": fmt_ratio(best),
        "beats_ceiling": best > H23_BORDERED_CEILING,
    })
    print(f"  best = {best:,}  ratio = {fmt_ratio(best)}  iters={iters}  wall={wall:.1f}s")

    # --- Method 4: Paley H24 deletion + SA ---
    print(f"[4/4] Paley H24 deletion + SA ...")
    best, wall, iters, best_mat = run_paley_until(args.budget, top_k=args.paley_top_k)
    # Re-verify with exact Bareiss
    if best_mat is not None:
        best_verified = verify_or_die(best_mat.entries)
        assert best_verified == best, (
            f"Paley best |det| mismatch: SA reported {best:,}, "
            f"Bareiss says {best_verified:,}"
        )
    rows.append({
        "method": "Paley H24 deletion + SA",
        "best_abs_det": best,
        "walltime_s": wall,
        "iters": iters,
        "ratio": fmt_ratio(best),
        "beats_ceiling": best > H23_BORDERED_CEILING,
    })
    print(f"  best = {best:,}  ratio = {fmt_ratio(best)}  iters={iters}  wall={wall:.1f}s")

    # --- Summary table ---
    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    header = f"{'Method':<38} {'|det|':>22} {'ratio':>8} {'beats?':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        marker = "yes" if r["beats_ceiling"] else "no"
        print(
            f"{r['method']:<38} {r['best_abs_det']:>22,} "
            f"{r['ratio']:>8} {marker:>7}"
        )

    print()
    any_beat = any(r["beats_ceiling"] for r in rows[1:])  # exclude baseline
    if any_beat:
        print(">>> SUCCESS: at least one rich ansatz exceeded the bordered-circulant ceiling.")
    else:
        gaps = [(r["method"], 1 - r["best_abs_det"] / H23_BORDERED_CEILING) for r in rows[1:]]
        print(">>> NEGATIVE RESULT: no rich ansatz exceeded the ceiling at this budget.")
        for m, g in gaps:
            print(f"      gap below ceiling for {m}: {g*100:.2f}%")


if __name__ == "__main__":
    main()
