"""Iteration 1: block-cyclic ansatz at n=21.

Hypothesis: 21 = 3*7 admits a block-cyclic structure
M = [[A,B,C],[C,A,B],[B,C,A]] with A,B,C of order 7. If each is itself a
7x7 circulant we have a 2^21 = 2M state space, exhaustively enumerable
in minutes by Bareiss (~1ms each → ~30 min single-threaded). We
parallelize via multiprocessing.

We also run SA over the free 147-bit space (no circulant constraint)
for 60s as a comparator.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

from hadamard_maxdet.n21_block_cyclic import (
    block_cyclic_circulant,
    block_cyclic_from_blocks,
    sa_block_cyclic_free,
)
from hadamard_maxdet.verifier import integer_determinant


def _bits7(idx: int) -> list[int]:
    return [1 if (idx >> k) & 1 else -1 for k in range(7)]


def _enum_chunk(args) -> tuple[int, tuple[int, int, int]]:
    """Enumerate a slice of (a,b,c) ∈ [a_lo,a_hi) × [0,128) × [0,128)."""
    a_lo, a_hi = args
    best_det = 0
    best_idx = (0, 0, 0)
    for ai in range(a_lo, a_hi):
        # Symmetry: fix a[0]=+1 (we can negate first row of A — but that
        # doesn't preserve the block-cyclic structure since A appears in
        # multiple positions. Skip the symmetry.) So we scan all 128.
        a = _bits7(ai)
        for bi in range(128):
            b = _bits7(bi)
            for ci in range(128):
                c = _bits7(ci)
                M = block_cyclic_circulant(a, b, c)
                d = abs(integer_determinant(M.entries))
                if d > best_det:
                    best_det = d
                    best_idx = (ai, bi, ci)
    return best_det, best_idx


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "refine_n21_iter_1.json"

    print("=== Iteration 1: block-cyclic ansatz, n=21 ===")
    print()

    # --- Part A: exhaustive enumeration of 2^21 circulant-block subfamily ---
    print("Part A: exhaustive enum of 2^21 = 2,097,152 circulant-block configs.")
    n_workers = max(1, mp.cpu_count() // 2)  # leave room for the OS
    print(f"  workers={n_workers}")

    t0 = time.monotonic()
    # Split a-axis into chunks
    a_chunks = []
    chunk_size = max(1, 128 // n_workers)
    for a_lo in range(0, 128, chunk_size):
        a_chunks.append((a_lo, min(a_lo + chunk_size, 128)))
    print(f"  chunks={len(a_chunks)} of size ~{chunk_size}")

    with mp.Pool(n_workers) as pool:
        results = pool.map(_enum_chunk, a_chunks)

    best_det_enum = 0
    best_idx = (0, 0, 0)
    for d, idx in results:
        if d > best_det_enum:
            best_det_enum = d
            best_idx = idx
    enum_wall = time.monotonic() - t0
    print(f"  enum best |det| = {best_det_enum:,}  walltime = {enum_wall:.1f}s")
    print(f"  best (a_idx, b_idx, c_idx) = {best_idx}")

    # Verify the best
    a, b, c = _bits7(best_idx[0]), _bits7(best_idx[1]), _bits7(best_idx[2])
    M_best = block_cyclic_circulant(a, b, c)
    verified = abs(integer_determinant(M_best.entries))
    assert verified == best_det_enum, f"verification mismatch: {verified} != {best_det_enum}"

    # --- Part B: SA over the 147-bit free block-cyclic space ---
    print()
    print("Part B: SA over 2^147 free-block space (60s budget, multi-start).")
    t0 = time.monotonic()
    sa_best_det = 0
    sa_best_blocks = None
    n_starts = 0
    seed_off = 0
    while time.monotonic() - t0 < 60.0:
        d, blocks, _tele = sa_block_cyclic_free(
            iterations=10_000,
            t_start=1.0,
            t_end=0.005,
            seed=seed_off,
        )
        if d > sa_best_det:
            sa_best_det = d
            sa_best_blocks = blocks
        n_starts += 1
        seed_off += 1
    sa_wall = time.monotonic() - t0
    print(f"  SA best |det| = {sa_best_det:,}  walltime = {sa_wall:.1f}s  starts = {n_starts}")

    if sa_best_blocks is not None:
        M_sa_best = block_cyclic_from_blocks(*sa_best_blocks)
        sa_verified = abs(integer_determinant(M_sa_best.entries))
        assert sa_verified == sa_best_det

    # --- Part C: SA seeded from circulant-enum best ---
    print()
    print("Part C: SA from enum-best as initial (10s).")
    t0 = time.monotonic()
    A0 = M_best.entries[:7][:]  # top-left 7x7 = A (circulant)
    A0 = [row[:7] for row in M_best.entries[:7]]
    B0 = [row[7:14] for row in M_best.entries[:7]]
    C0 = [row[14:21] for row in M_best.entries[:7]]
    seed_best_det = best_det_enum
    seed_best_blocks = (A0, B0, C0)
    seed_off = 1000
    while time.monotonic() - t0 < 10.0:
        d, blocks, _ = sa_block_cyclic_free(
            iterations=5_000,
            t_start=0.1,
            t_end=0.001,
            seed=seed_off,
            initial=seed_best_blocks,
        )
        if d > seed_best_det:
            seed_best_det = d
            seed_best_blocks = blocks
        seed_off += 1
    seed_wall = time.monotonic() - t0
    print(f"  Seeded SA best |det| = {seed_best_det:,}  walltime = {seed_wall:.1f}s")

    # --- Pick overall best ---
    overall_best = max(best_det_enum, sa_best_det, seed_best_det)
    if overall_best == best_det_enum:
        winner = "enum_circulant_blocks"
        winner_M = M_best
    elif overall_best == seed_best_det:
        winner = "sa_free_seeded_from_enum"
        winner_M = block_cyclic_from_blocks(*seed_best_blocks)
    else:
        winner = "sa_free_random_starts"
        winner_M = block_cyclic_from_blocks(*sa_best_blocks)

    overall_verified = abs(integer_determinant(winner_M.entries))
    assert overall_verified == overall_best

    proven = 59_392_000_000_000
    ratio = overall_best / proven
    print()
    print(f"=== Iteration 1 summary ===")
    print(f"  enum best          = {best_det_enum:,}")
    print(f"  SA random starts   = {sa_best_det:,}")
    print(f"  SA seeded from enum = {seed_best_det:,}")
    print(f"  overall best       = {overall_best:,} ({winner})")
    print(f"  proven n=21 max    = {proven:,}")
    print(f"  ratio              = {ratio:.4f}")
    print(f"  prior pipeline best = 51,200,000,000,000  (0.8621)")

    payload = {
        "iteration": 1,
        "hypothesis": "block-cyclic [[A,B,C],[C,A,B],[B,C,A]] with circulant blocks (2^21 enum) + free 147-bit SA",
        "n": 21,
        "proven_max": proven,
        "enum_best_det": best_det_enum,
        "enum_best_idx": list(best_idx),
        "enum_walltime_s": enum_wall,
        "sa_random_starts_best_det": sa_best_det,
        "sa_random_starts_walltime_s": sa_wall,
        "sa_random_starts_n_starts": n_starts,
        "sa_seeded_best_det": seed_best_det,
        "sa_seeded_walltime_s": seed_wall,
        "overall_best_det": overall_best,
        "overall_winner": winner,
        "overall_ratio": ratio,
        "winner_matrix": winner_M.entries,
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
