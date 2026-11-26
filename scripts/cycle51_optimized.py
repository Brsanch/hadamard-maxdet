"""H41 cycle 51 (optimized): numpy-vectorized inner-loop SBIBD search.

Same algorithm as cycle51_h41_sbibd.py — deterministic lex-min DFS from
canonical (B1, B2). Inner candidate-DFS uses numpy 2D arrays for
pair_count and in_block, plus vectorized constraint checks. Should produce
identical results, but much faster per node.

Hot-path differences vs cycle51:
  pair_count[v][v]  (numpy int8 2D, symmetric, diag = point_count)
  in_block[T][v]    (numpy int8 2D; T = #blocks so far)
  partial-loop pair check: (pair_count[partial_arr, p] >= lam).any()
  intersection check:      (inter + in_block[:, p] > lam).any()
"""
from __future__ import annotations

import json
import time as _time
from itertools import combinations
from pathlib import Path

import numpy as np

from hadamard_maxdet.verifier import integer_determinant


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def ew_bound(n: int) -> int:
    s2 = 2 * n - 1
    s = int(round(s2 ** 0.5))
    assert s * s == s2
    return s * (n - 1) ** ((n - 1) // 2)


def search_sbibd_backtrack_np(v: int, k: int, lam: int) -> list[list[int]] | None:
    """Numpy-vectorized point-wise backtracking. Identical lex-min order to
    the pure-Python version; deterministic.
    """
    B1 = list(range(k))
    B2 = list(range(lam)) + list(range(k, k + (k - lam)))
    blocks: list[list[int]] = [B1, B2]
    assert len(set(B1).intersection(B2)) == lam

    # Diag of pair_count = point_count by convention (single counter).
    pair_count = np.zeros((v, v), dtype=np.int32)
    for B in blocks:
        for p in B:
            pair_count[p, p] += 1
        for p, q in combinations(B, 2):
            pair_count[p, q] += 1
            pair_count[q, p] += 1

    # in_block stored as a list of length-v int8 vectors that we grow/shrink.
    in_block = [np.array([1 if p in set(B) else 0 for p in range(v)], dtype=np.int32) for B in blocks]

    t_start = _time.monotonic()
    nodes_visited = [0]
    last_print = [_time.monotonic()]
    deepest = [2]

    point_count_view = lambda: pair_count.diagonal()

    def feasibility_prune() -> bool:
        T = len(blocks)
        remaining = v - T
        pc = pair_count.diagonal()
        # Each point needs (k - pc[p]) more blocks; must fit in remaining.
        if (k - pc < 0).any() or (k - pc > remaining).any():
            return False
        # Pair coverage: lam - pair_count[i,j] must be ≤ min(remaining, k-pc[i], k-pc[j]).
        # Vectorize: cap[i,j] = min(remaining, k-pc[i], k-pc[j]).
        kpc = (k - pc).astype(np.int32)
        cap_ij = np.minimum(np.minimum(remaining, kpc[:, None]), kpc[None, :])
        # Exclude diagonal from this check (diagonals are point_count, not pair count).
        offdiag = pair_count.copy()
        np.fill_diagonal(offdiag, 0)
        need = lam - offdiag
        # need < 0 means we already exceeded lam → infeasible
        if (need < 0).any():
            return False
        # need > cap → can never reach lam from here
        # but only check off-diagonal entries
        np.fill_diagonal(need, 0)
        np.fill_diagonal(cap_ij, 0)
        if (need > cap_ij).any():
            return False
        return True

    def iter_candidates(lex_floor: list[int]):
        """Generator yielding lex-greater 9-subsets satisfying intersection constraints."""
        T = len(blocks)
        # Stack in_block into 2D array for fast vectorized access.
        in_block_arr = np.stack(in_block, axis=0)  # shape (T, v)
        partial: list[int] = []
        partial_arr = np.zeros(k, dtype=np.int64)  # preallocated
        n_partial = 0
        inter = np.zeros(T, dtype=np.int32)

        def can_finish(slots_remaining: int) -> bool:
            return bool((inter + slots_remaining >= lam).all())

        def dfs(start: int, remaining: int):
            nonlocal n_partial
            if remaining == 0:
                if (inter == lam).all():
                    if list(partial_arr[:n_partial]) > lex_floor:
                        yield list(partial_arr[:n_partial])
                return
            if not can_finish(remaining):
                return
            for p in range(start, v):
                # Point cap
                if pair_count[p, p] >= k:
                    continue
                # Pair cap with already-chosen points (vectorized)
                if n_partial > 0:
                    if (pair_count[partial_arr[:n_partial], p] >= lam).any():
                        continue
                # Intersection cap (vectorized)
                new_inter = inter + in_block_arr[:, p]
                if (new_inter > lam).any():
                    continue
                # Recurse
                partial_arr[n_partial] = p
                n_partial += 1
                inter[:] += in_block_arr[:, p]
                yield from dfs(p + 1, remaining - 1)
                inter[:] -= in_block_arr[:, p]
                n_partial -= 1

        yield from dfs(0, k)

    def add_block(B: list[int]) -> None:
        for p in B:
            pair_count[p, p] += 1
        for p, q in combinations(B, 2):
            pair_count[p, q] += 1
            pair_count[q, p] += 1
        in_block.append(np.array([1 if p in set(B) else 0 for p in range(v)], dtype=np.int32))
        blocks.append(B)

    def remove_block() -> None:
        B = blocks.pop()
        in_block.pop()
        for p in B:
            pair_count[p, p] -= 1
        for p, q in combinations(B, 2):
            pair_count[p, q] -= 1
            pair_count[q, p] -= 1

    def search() -> list[list[int]] | None:
        nodes_visited[0] += 1
        if len(blocks) > deepest[0]:
            deepest[0] = len(blocks)
        now = _time.monotonic()
        if now - last_print[0] > 30.0:
            last_print[0] = now
            elapsed = now - t_start
            offdiag = pair_count.copy()
            np.fill_diagonal(offdiag, 0)
            print(
                f"  [opt] blocks={len(blocks)}/{v}  nodes={nodes_visited[0]:_}  "
                f"deepest={deepest[0]}  pair_max={int(offdiag.max())}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        if len(blocks) == v:
            return [list(b) for b in blocks]
        if not feasibility_prune():
            return None
        last = blocks[-1]
        for c in iter_candidates(last):
            add_block(c)
            r = search()
            if r is not None:
                return r
            remove_block()
        return None

    print(f"  [opt] starting numpy backtrack on ({v}, {k}, {lam})", flush=True)
    print(f"        B1={B1}", flush=True)
    print(f"        B2={B2}", flush=True)
    result = search()
    elapsed = _time.monotonic() - t_start
    if result is None:
        print(f"  [opt] EXHAUSTED. nodes={nodes_visited[0]:_} deepest={deepest[0]} {elapsed:.1f}s", flush=True)
        return None
    print(f"  [opt] FOUND in {nodes_visited[0]:_} nodes, {elapsed:.1f}s, deepest={deepest[0]}", flush=True)
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
    print(f"[opt] EW({n}) = {EW} ({EW:.4e})")
    blocks = search_sbibd_backtrack_np(v=41, k=16, lam=6)
    if blocks is None:
        print("[opt] No SBIBD found.")
        return
    N = [[0] * 41 for _ in range(41)]
    for j, B in enumerate(blocks):
        for p in B:
            N[p][j] = 1
    if not verify_sbibd(N, 16, 6):
        print("[opt] SBIBD verify FAILED")
        return
    print(f"[opt] N N^T = 10 I + 6 J ✓")
    M = [[2 * N[i][j] - 1 for j in range(41)] for i in range(41)]
    det = abs(integer_determinant(M))
    ratio = det / EW
    print(f"[opt] |det| = {det}")
    print(f"[opt] EW    = {EW}")
    print(f"[opt] ratio = {ratio}")
    print(f"[opt] EW-tight = {det == EW}")
    out = {
        "n": 41, "ew_41": EW, "det": det, "ratio": ratio,
        "ew_tight": det == EW, "blocks": blocks,
        "construction": "(41, 16, 6) SBIBD via numpy point-wise backtracking",
    }
    (RESULTS / "H41_cycle51_sbibd.json").write_text(json.dumps(out, indent=2))
    np.save(RESULTS / "H41_cycle51_sbibd.npy", np.array(M, dtype=np.int8))
    print(f"\n[opt] Saved.")


if __name__ == "__main__":
    main()
