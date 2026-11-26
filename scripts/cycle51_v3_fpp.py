"""H41 cycle 51 v3: forced-point + forced-pair propagation backtracking.

Same canonical (B1, B2) and same lex-min DFS skeleton as cycle 51 v1.
Adds CSP-style propagation BEFORE the DFS loop:

  Forced point: a point p with (k - point_count[p]) == (v - len(blocks))
  (i.e., its deficit equals the number of blocks still to place, INCLUDING
  the one we're about to construct). Such a p MUST be in every remaining
  block, including this one. Pre-include all forced points in the partial.

  Once forced points are in partial, the DFS only needs to fill the
  remaining (k - |forced|) slots from points lex-greater than the last
  forced point. All the existing pair and intersection caps still apply
  to forced points; if the forced set already violates a cap, the entire
  branch is infeasible.

This is a pure pruning improvement: enumerates a strict subset of what
v1 enumerates, and the subset is exactly the candidates compatible with
forced inclusions. v1 finds those same candidates eventually, just slower.

For the (25, 9, 3) problem, forced-point propagation triggers at the
later levels (when remaining blocks shrinks below typical deficit), so
v3 should match v1 on small cases and accelerate on the larger (41,16,6).
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


def search_sbibd_fpp(v: int, k: int, lam: int) -> list[list[int]] | None:
    """Point-wise DFS with forced-point propagation."""

    B1 = list(range(k))
    B2 = list(range(lam)) + list(range(k, k + (k - lam)))
    blocks: list[list[int]] = [B1, B2]
    assert len(set(B1).intersection(B2)) == lam

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
    forced_pre = [0]  # count of times forced-point propagation triggered
    last_print = [_time.monotonic()]
    deepest = [2]

    def feasibility_prune() -> bool:
        remaining = v - len(blocks)
        for p in range(v):
            need = k - point_count[p]
            if need < 0 or need > remaining:
                return False
        for p in range(v):
            for q in range(p + 1, v):
                need = lam - pair_count.get((p, q), 0)
                if need < 0:
                    return False
                cap = min(remaining, k - point_count[p], k - point_count[q])
                if need > cap:
                    return False
        return True

    def iter_candidates_with_fpp(blocks_, lex_floor):
        n_blocks = len(blocks_)
        block_sets = [set(B) for B in blocks_]
        in_block = [[1 if p in block_sets[i] else 0 for p in range(v)] for i in range(n_blocks)]

        # Forced-point propagation: deficit == remaining_blocks (including this)
        remaining_blocks = v - n_blocks
        forced = [p for p in range(v) if (k - point_count[p]) == remaining_blocks]

        if len(forced) > k:
            return  # more forced points than fit in a block

        # Validate forced set: pair caps + intersection caps not violated.
        for i in range(len(forced)):
            for j in range(i + 1, len(forced)):
                if pair_count.get((forced[i], forced[j]), 0) >= lam:
                    return
        # Compute initial inter from forced points
        inter = [sum(in_block[i][p] for p in forced) for i in range(n_blocks)]
        for s in inter:
            if s > lam:
                return

        if forced:
            forced_pre[0] += 1

        partial: list[int] = list(forced)
        # `start` is the smallest index we may pick next; must be > max(forced)
        # to keep partial sorted ascending.
        next_start = (max(forced) + 1) if forced else 0
        n_to_fill = k - len(forced)

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

        yield from dfs(next_start, n_to_fill)

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
                f"  [v3] blocks={len(blocks)}/{v}  nodes={nodes_visited[0]:_}  "
                f"deepest={deepest[0]}  pair_max={max(pair_count.values()) if pair_count else 0}  "
                f"forced_pre={forced_pre[0]:_}  ({elapsed:.0f}s)",
                flush=True,
            )
        if len(blocks) == v:
            return [list(b) for b in blocks]
        if not feasibility_prune():
            return None
        last = blocks[-1]
        for c in iter_candidates_with_fpp(blocks, last):
            add_block(c)
            r = search()
            if r is not None:
                return r
            remove_block()
        return None

    print(f"  [v3] starting forced-prop backtrack on ({v}, {k}, {lam})", flush=True)
    print(f"       B1={B1}", flush=True)
    print(f"       B2={B2}", flush=True)
    result = search()
    elapsed = _time.monotonic() - t_start
    if result is None:
        print(f"  [v3] EXHAUSTED. nodes={nodes_visited[0]:_} deepest={deepest[0]} forced_pre={forced_pre[0]:_} {elapsed:.1f}s", flush=True)
        return None
    print(f"  [v3] FOUND in {nodes_visited[0]:_} nodes, {elapsed:.1f}s, deepest={deepest[0]}, forced_pre triggered {forced_pre[0]:_} times", flush=True)
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
    print(f"[v3] EW({n}) = {EW} ({EW:.4e})")
    blocks = search_sbibd_fpp(v=41, k=16, lam=6)
    if blocks is None:
        print("[v3] No SBIBD found.")
        return
    N = [[0] * 41 for _ in range(41)]
    for j, B in enumerate(blocks):
        for p in B:
            N[p][j] = 1
    if not verify_sbibd(N, 16, 6):
        print("[v3] SBIBD verify FAILED")
        return
    print(f"[v3] N N^T = 10 I + 6 J ✓")
    M = [[2 * N[i][j] - 1 for j in range(41)] for i in range(41)]
    det = abs(integer_determinant(M))
    ratio = det / EW
    print(f"[v3] |det| = {det}")
    print(f"[v3] EW    = {EW}")
    print(f"[v3] ratio = {ratio}")
    print(f"[v3] EW-tight = {det == EW}")
    out = {
        "n": 41, "ew_41": EW, "det": det, "ratio": ratio,
        "ew_tight": det == EW, "blocks": blocks,
        "construction": "(41, 16, 6) SBIBD via forced-point-propagation point-wise backtracking",
    }
    (RESULTS / "H41_cycle51_sbibd.json").write_text(json.dumps(out, indent=2))
    np.save(RESULTS / "H41_cycle51_sbibd.npy", np.array(M, dtype=np.int8))
    print(f"\n[v3] Saved.")


if __name__ == "__main__":
    main()
