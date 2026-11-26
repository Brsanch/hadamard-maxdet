"""H41 cycle 51: (41, 16, 6) SBIBD-derived EW-tight matrix.

n=41 satisfies 2n − 1 = 81 = 9², so the Ehlich–Wojtas bound
    EW(41) = 9 · 40^20
is sharp, and an (n, k, λ) SBIBD with k − λ = (n−1)/4 = 10 wraps via
M = 2N − J to a ±1 matrix attaining EW(41) exactly.

Two candidate parameter sets:
    (41, 16, 6)   — k=16, λ=6, k−λ=10
    (41, 25, 15)  — complement of (41, 16, 6); same |det|

Both are BRC-feasible (z² = 10 x² + 6 y² has integer solution (1, 1, 4)).
We search (41, 16, 6) since smaller k means smaller search space.

Algorithm: point-wise DFS backtracking with pair-count pruning, parameterised
by (v, k, λ). Symmetry-broken canonical (B1, B2):
    B1 = {0, 1, ..., 15}
    B2 = {0, 1, ..., 5, 16, 17, ..., 25}        (|B1 ∩ B2| = λ = 6)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hadamard_maxdet.verifier import integer_determinant


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def ew_bound(n: int) -> int:
    """Ehlich–Wojtas bound for n ≡ 1 mod 4. EW(n) = sqrt((n−1)^(n−1) · (2n−1))."""
    s2 = 2 * n - 1
    s = int(round(s2 ** 0.5))
    assert s * s == s2, f"2n-1 = {s2} not a perfect square; EW not tight at n={n}"
    return s * (n - 1) ** ((n - 1) // 2)


def search_sbibd_backtrack(v: int, k: int, lam: int) -> list[list[int]] | None:
    """Point-wise backtracking SBIBD search with pair-count pruning.

    Generalised from cycle 50: parametric in (v, k, lam). Block 1 fixed
    to {0..k-1}; block 2 fixed to canonical lex-min lookalike with
    |B1 ∩ B2| = lam.
    """
    from itertools import combinations
    import time as _time

    B1 = list(range(k))
    B2 = list(range(lam)) + list(range(k, k + (k - lam)))
    blocks: list[list[int]] = [B1, B2]
    assert len(set(B1).intersection(B2)) == lam, "B2 canonical wrong"

    point_count = [0] * v
    for B in blocks:
        for p in B:
            point_count[p] += 1
    pair_count: dict[tuple[int, int], int] = {}
    for B in blocks:
        for p, q in combinations(sorted(B), 2):
            pair_count[(p, q)] = pair_count.get((p, q), 0) + 1

    t_start = _time.monotonic()
    nodes_visited = [0]
    last_print = [_time.monotonic()]
    deepest = [2]

    def feasibility_prune() -> bool:
        remaining_blocks = v - len(blocks)
        for p in range(v):
            need = k - point_count[p]
            if need < 0 or need > remaining_blocks:
                return False
        for p in range(v):
            for q in range(p + 1, v):
                need = lam - pair_count.get((p, q), 0)
                if need < 0:
                    return False
                cap = min(remaining_blocks, k - point_count[p], k - point_count[q])
                if need > cap:
                    return False
        return True

    def iter_candidates_pointwise(blocks_, lex_floor):
        n_blocks = len(blocks_)
        block_sets = [set(B) for B in blocks_]
        in_block = [[1 if p in block_sets[i] else 0 for p in range(v)] for i in range(n_blocks)]

        partial: list[int] = []
        inter = [0] * n_blocks

        def can_finish(slots_remaining: int) -> bool:
            for i in range(n_blocks):
                if inter[i] + slots_remaining < lam:
                    return False
            return True

        def dfs(start: int, remaining: int):
            if remaining == 0:
                for i in range(n_blocks):
                    if inter[i] != lam:
                        return
                if partial > lex_floor:
                    yield list(partial)
                return
            if not can_finish(remaining):
                return
            for p in range(start, v):
                if point_count[p] >= k:
                    continue
                ok = True
                for q in partial:
                    pp, qq = (q, p) if q < p else (p, q)
                    if pair_count.get((pp, qq), 0) >= lam:
                        ok = False; break
                if not ok:
                    continue
                bad = False
                for i in range(n_blocks):
                    if in_block[i][p]:
                        if inter[i] + 1 > lam:
                            bad = True; break
                if bad:
                    continue
                partial.append(p)
                for i in range(n_blocks):
                    if in_block[i][p]:
                        inter[i] += 1
                yield from dfs(p + 1, remaining - 1)
                partial.pop()
                for i in range(n_blocks):
                    if in_block[i][p]:
                        inter[i] -= 1

        yield from dfs(0, k)

    def add_block(B):
        blocks.append(B)
        for p in B:
            point_count[p] += 1
        for p, q in combinations(B, 2):
            pair_count[(p, q)] = pair_count.get((p, q), 0) + 1

    def remove_block():
        B = blocks.pop()
        for p in B:
            point_count[p] -= 1
        for p, q in combinations(B, 2):
            pair_count[(p, q)] -= 1
            if pair_count[(p, q)] == 0:
                del pair_count[(p, q)]

    def search():
        nodes_visited[0] += 1
        if len(blocks) > deepest[0]:
            deepest[0] = len(blocks)
        now = _time.monotonic()
        if now - last_print[0] > 30.0:
            last_print[0] = now
            elapsed = now - t_start
            print(
                f"  blocks={len(blocks)}/{v}  nodes={nodes_visited[0]:_}  "
                f"deepest={deepest[0]}  pair_max={max(pair_count.values()) if pair_count else 0}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        if len(blocks) == v:
            return [list(b) for b in blocks]
        if not feasibility_prune():
            return None
        last = blocks[-1]
        for c in iter_candidates_pointwise(blocks, last):
            add_block(c)
            r = search()
            if r is not None:
                return r
            remove_block()
        return None

    print(f"  starting backtrack on ({v}, {k}, {lam}) with B1={B1}", flush=True)
    print(f"                                              B2={B2}", flush=True)
    result = search()
    elapsed = _time.monotonic() - t_start
    if result is None:
        print(
            f"  EXHAUSTED: no SBIBD with this canonical (B1, B2). "
            f"{nodes_visited[0]:_} nodes, deepest={deepest[0]}, {elapsed:.1f}s",
            flush=True,
        )
        return None
    print(f"  FOUND in {nodes_visited[0]:_} nodes, {elapsed:.1f}s, deepest={deepest[0]}", flush=True)
    return result


def verify_sbibd(N, k: int, lam: int) -> bool:
    A = np.array(N, dtype=np.int64)
    v = A.shape[0]
    AAt = A @ A.T
    expected = (k - lam) * np.eye(v, dtype=np.int64) + lam * np.ones((v, v), dtype=np.int64)
    return np.array_equal(AAt, expected)


def main() -> None:
    n = 41
    EW = ew_bound(n)
    print(f"EW({n}) = sqrt({n-1}^{n-1} · {2*n-1}) = 9 · 40^20 = {EW} ({EW:.4e})")
    print(f"Searching for (41, 16, 6) SBIBD via point-wise backtracking...")
    print(f"(Cyclic Z_41 DS skipped: multiplier theorem suggests no abelian-regular form;")
    print(f" point-wise search is the same template that closed cycle 50 in 24 nodes.)")
    print()

    blocks = search_sbibd_backtrack(v=41, k=16, lam=6)
    if blocks is None:
        print("Backtracking exhausted; no (41, 16, 6) SBIBD with this canonical (B1, B2).")
        print("This is unexpected (EW is sharp at n=41). Would need alternative")
        print("canonical breakdowns or stronger isomorph rejection.")
        return

    # Build N (rows = points, cols = blocks)
    N = [[0] * 41 for _ in range(41)]
    for j, B in enumerate(blocks):
        for p in B:
            N[p][j] = 1
    if not verify_sbibd(N, 16, 6):
        print("  SBIBD verification FAILED — bug in search")
        return
    print(f"  N N^T = 10 I + 6 J ✓ (verified (41, 16, 6) SBIBD)")

    M = [[2 * N[i][j] - 1 for j in range(41)] for i in range(41)]
    det = abs(integer_determinant(M))
    ratio = det / EW
    print(f"  |det(2N − J)| = {det}")
    print(f"  EW(41)        = {EW}")
    print(f"  ratio         = {ratio}")
    print(f"  EW-tight      = {det == EW}")

    out = {
        "n": 41,
        "ew_41": EW,
        "det": det,
        "ratio": ratio,
        "ew_tight": det == EW,
        "blocks": blocks,
        "construction": "(41, 16, 6) SBIBD via point-wise backtracking; M = 2N - J",
    }
    (RESULTS / "H41_cycle51_sbibd.json").write_text(json.dumps(out, indent=2))
    np.save(RESULTS / "H41_cycle51_sbibd.npy", np.array(M, dtype=np.int8))
    print(f"\nSaved: results/H41_cycle51_sbibd.json + .npy")


if __name__ == "__main__":
    main()
