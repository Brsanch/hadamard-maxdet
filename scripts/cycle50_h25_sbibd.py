"""H25 cycle 50: (25, 9, 3) SBIBD-derived EW-tight matrix.

For an SBIBD with parameters (v, k, λ), the {0,1}-incidence matrix N
satisfies N N^T = (k - λ) I + λ J. The wrap M = 2N - J then has

    M M^T = 4(k-λ) I + (4λ - 4k + v) J.

For (25, 9, 3): M M^T = 24 I + J. Eigenvalues of (24 I + J) are 49
(once) and 24 (24 times), so |det M|² = 49 · 24^24 and

    |det M| = 7 · 24^12 = 255,642,432,052,396,032 = EW(25),

hitting the Ehlich–Wojtas bound exactly. (25, 9, 3) admits a difference
set in Z_5 × Z_5 (it is not cyclic in Z_25). We search the difference-
set candidates by fixing (0, 0) and enumerating 8-subsets of the
remaining 24 group elements, checking the standard difference-multiset
condition: for every non-identity g, |{(d, d') ∈ D×D : d − d' = g}| = 3.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
EW_25 = 7 * 24**12


def find_diff_set_z5xz5() -> list[tuple[int, int]] | None:
    """Search Z_5 × Z_5 for a (25, 9, 3) difference set containing (0,0).

    By translation invariance, every difference set has a translate with
    (0, 0) ∈ D, so fixing (0, 0) loses no generality. We then enumerate
    the C(24, 8) ≈ 735K choices of the remaining 8 elements.

    A (25, 9, 3) difference set D ⊂ G = Z_5 × Z_5 satisfies, for every
    g ≠ 0 in G: |{(d₁, d₂) ∈ D × D : d₁ − d₂ = g}| = 3.
    """
    G = [(i, j) for i in range(5) for j in range(5)]
    G.remove((0, 0))  # fix (0,0) in D

    target = 3
    nz_mask = {g for g in G}  # 24 nonzero elements

    n_checked = 0
    t0 = time.monotonic()
    for combo in itertools.combinations(G, 8):
        n_checked += 1
        D = list(combo) + [(0, 0)]  # 9 elements
        # Compute difference multiset
        counts: dict[tuple[int, int], int] = {}
        ok = True
        for d1 in D:
            for d2 in D:
                if d1 == d2:
                    continue
                g = ((d1[0] - d2[0]) % 5, (d1[1] - d2[1]) % 5)
                counts[g] = counts.get(g, 0) + 1
                if counts[g] > target:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # Check that every nonzero g hit target times
        if all(counts.get(g, 0) == target for g in nz_mask):
            print(f"  found after {n_checked} candidates ({time.monotonic()-t0:.1f}s)")
            return D
        if n_checked % 50_000 == 0:
            print(f"  ... {n_checked} checked, {time.monotonic()-t0:.1f}s")
    print(f"  exhausted {n_checked} candidates, no DS found")
    return None


def sbibd_from_diff_set(D: list[tuple[int, int]]) -> list[list[int]]:
    """Build the v×v incidence matrix N from a difference set D ⊂ Z_5 × Z_5.

    Rows indexed by group elements g; row g lists indicator of (g + D).
    Equivalently N[g][h] = 1 iff h ∈ g + D.
    """
    G = [(i, j) for i in range(5) for j in range(5)]
    idx = {g: i for i, g in enumerate(G)}
    v = 25
    N = [[0] * v for _ in range(v)]
    for g in G:
        translate = {((g[0] + d[0]) % 5, (g[1] + d[1]) % 5) for d in D}
        for h in translate:
            N[idx[g]][idx[h]] = 1
    return N


def verify_sbibd(N: list[list[int]], k: int = 9, lam: int = 3) -> bool:
    A = np.array(N, dtype=np.int64)
    v = A.shape[0]
    AAt = A @ A.T
    expected = (k - lam) * np.eye(v, dtype=np.int64) + lam * np.ones((v, v), dtype=np.int64)
    return np.array_equal(AAt, expected)


def search_sbibd_backtrack() -> list[list[int]] | None:
    """Backtracking search for (25, 9, 3) SBIBD with cell-partition candidate
    generation.

    Key fact (dual constraint): in any symmetric (v, k, λ) design, every
    pair of blocks intersects in exactly λ points. So for (25, 9, 3): every
    pair of blocks shares exactly 3 elements.

    Symmetry breaking:
      Block 1 = {0, 1, ..., 8}                                (point-perm)
      Block 2 = {0, 1, 2, 9, 10, 11, 12, 13, 14}              (point-perm
        within {0..8} and within {9..24}, lex-min |∩B1|=3 + 6 outside)

    For block t ≥ 3, generate candidates by deciding how many points to
    take from each cell of the Venn-partition induced by all previous
    blocks. Each cell is constrained by:
      - sum of |B_t ∩ cell| over cells inside B_i = λ for each i.
      - cell sizes don't exceed cell capacity.
    Candidates are filtered by the global pair-count and point-count caps.
    """
    import time as _time
    from itertools import combinations

    v, k, lam = 25, 9, 3

    # Initial blocks (lex-min canonical form).
    B1 = list(range(9))                                      # {0..8}
    B2 = [0, 1, 2, 9, 10, 11, 12, 13, 14]                    # |B1∩B2|=3 ✓
    blocks: list[list[int]] = [B1, B2]

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
    deepest = [2]  # deepest level reached

    def cells_for(blocks: list[list[int]]) -> list[list[int]]:
        v = 25
        sigs: dict[int, list[int]] = {}
        for p in range(v):
            sig = 0
            for i, B in enumerate(blocks):
                if p in B:
                    sig |= 1 << i
            sigs.setdefault(sig, []).append(p)
        return list(sigs.values()), list(sigs.keys())

    def iter_candidates_pointwise(blocks: list[list[int]], lex_floor: list[int]):
        """Yield 9-subsets B with |B ∩ blocks[i]| = λ for all i, lex > lex_floor.

        Builds B point-by-point. Maintains:
          inter[i] = |partial ∩ blocks[i]|, must end at λ for all i.
          partial = current points (sorted ascending; lex ordering enforced).
        Pruning at each step:
          - point-count cap (point_count[p] < k).
          - pair-count cap (for q in partial: pair_count[(min(p,q),max(p,q))] < λ).
          - intersection cap (for each i: inter[i] ≤ λ).
          - intersection lower bound: for each i, can still reach λ with remaining slots.
          - lex ordering: only pick points > last in partial.
        """
        n_blocks = len(blocks)
        block_sets = [set(B) for B in blocks]
        # In-block-i indicator per point
        in_block = [[1 if p in block_sets[i] else 0 for p in range(v)] for i in range(n_blocks)]

        partial: list[int] = []
        inter = [0] * n_blocks

        def can_finish(slots_remaining: int) -> bool:
            for i in range(n_blocks):
                # max more points contributing to inter[i] is slots_remaining
                if inter[i] + slots_remaining < lam:
                    return False
            return True

        def dfs(start: int, remaining: int):
            # Generate next-block points from `start` upward.
            if remaining == 0:
                # Verify all inter == lam
                for i in range(n_blocks):
                    if inter[i] != lam:
                        return
                # Lex check vs lex_floor
                if partial > lex_floor:
                    yield list(partial)
                return
            if not can_finish(remaining):
                return
            for p in range(start, v):
                # Quick cap checks
                if point_count[p] >= k:
                    continue
                # Pair caps with already-picked points
                ok = True
                for q in partial:
                    pp, qq = (q, p) if q < p else (p, q)
                    if pair_count.get((pp, qq), 0) >= lam:
                        ok = False; break
                if not ok:
                    continue
                # Intersection cap
                bad = False
                for i in range(n_blocks):
                    if in_block[i][p]:
                        if inter[i] + 1 > lam:
                            bad = True; break
                if bad:
                    continue
                # Recurse
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

    def enumerate_candidates(blocks: list[list[int]]) -> list[list[int]]:
        """Enumerate all 9-subsets of {0..24} that intersect every block in
        exactly λ points. Use cell partition for efficiency."""
        cell_lists, cell_sigs = cells_for(blocks)
        n_cells = len(cell_lists)
        n_blocks = len(blocks)

        # For each cell c with signature sig_c (bitmask), and for each
        # block i, contribution of cell c to |B ∩ blocks[i]| is x_c if
        # bit i of sig_c is 1, else 0.
        # Constraints: for each i, sum_{c : sig_c has bit i} x_c = λ.
        #              sum_c x_c = k.
        #              0 ≤ x_c ≤ |cell_c|.

        cell_caps = [len(c) for c in cell_lists]

        # Enumerate (x_0, ..., x_{n_cells-1}) integer assignments via DFS.
        results: list[list[int]] = []
        x = [0] * n_cells

        def dfs_x(idx: int):
            if idx == n_cells:
                # Verify all block-intersection constraints satisfied
                if sum(x) != k:
                    return
                for i in range(n_blocks):
                    s = sum(x[c] for c in range(n_cells) if (cell_sigs[c] >> i) & 1)
                    if s != lam:
                        return
                # Materialise: pick x[c] from cell_lists[c], all combinations.
                # Use cartesian product of combinations.
                pools: list[list[tuple[int, ...]]] = []
                for c in range(n_cells):
                    if x[c] == 0:
                        pools.append([()])
                    else:
                        pools.append(list(combinations(cell_lists[c], x[c])))
                for combo in _cart(pools):
                    flat = [p for tup in combo for p in tup]
                    results.append(sorted(flat))
                return
            cap = cell_caps[idx]
            # Pruning: even at max remaining, can we reach target?
            remaining = k - sum(x[:idx])
            if remaining < 0 or remaining > sum(cell_caps[idx:]):
                return
            for v_choice in range(min(cap, remaining) + 1):
                x[idx] = v_choice
                dfs_x(idx + 1)
            x[idx] = 0

        def _cart(pools):
            if not pools:
                yield ()
                return
            head, *rest = pools
            for h in head:
                for r in _cart(rest):
                    yield (h,) + r

        dfs_x(0)
        return results

    def check_global(B: list[int]) -> bool:
        """B is a 9-subset; check it doesn't violate point/pair caps."""
        for p in B:
            if point_count[p] >= k:
                return False
        for p, q in combinations(B, 2):
            if pair_count.get((p, q), 0) >= lam:
                return False
        return True

    def add_block(B: list[int]) -> None:
        blocks.append(B)
        for p in B:
            point_count[p] += 1
        for p, q in combinations(B, 2):
            pair_count[(p, q)] = pair_count.get((p, q), 0) + 1

    def remove_block() -> None:
        B = blocks.pop()
        for p in B:
            point_count[p] -= 1
        for p, q in combinations(B, 2):
            pair_count[(p, q)] -= 1
            if pair_count[(p, q)] == 0:
                del pair_count[(p, q)]

    def feasibility_prune() -> bool:
        """Return True if current partial solution can possibly extend to a
        full SBIBD; False if proven infeasible.

        Tight checks:
          - Every point p needs (k − point_count[p]) more blocks containing
            it; remaining slots = v − len(blocks). If need > remaining,
            infeasible.
          - Every pair (p, q) needs (λ − pair_count) more co-blocks. The
            remaining blocks containing both p and q is bounded by the
            slots both have. If need > remaining-slots-with-both, infeasible.
        """
        remaining_blocks = v - len(blocks)
        for p in range(v):
            if k - point_count[p] > remaining_blocks:
                return False
            if k - point_count[p] < 0:
                return False
        for p in range(v):
            for q in range(p + 1, v):
                need = lam - pair_count.get((p, q), 0)
                if need < 0:
                    return False
                # Each block containing both p and q uses one remaining slot for both.
                # An upper bound on how many remaining blocks can contain both:
                #   min(remaining_blocks, k - point_count[p], k - point_count[q]).
                cap = min(remaining_blocks, k - point_count[p], k - point_count[q])
                if need > cap:
                    return False
        return True

    def search() -> list[list[int]] | None:
        nodes_visited[0] += 1
        if len(blocks) > deepest[0]:
            deepest[0] = len(blocks)
        now = _time.monotonic()
        if now - last_print[0] > 10.0:
            last_print[0] = now
            elapsed = now - t_start
            print(
                f"  blocks={len(blocks)}/25  nodes={nodes_visited[0]:_}  "
                f"deepest={deepest[0]}  pair_max={max(pair_count.values()) if pair_count else 0}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        if len(blocks) == v:
            return [list(b) for b in blocks]
        if not feasibility_prune():
            return None
        # Lazy point-wise candidate enumeration with pair-count pruning baked in.
        last = blocks[-1]
        for c in iter_candidates_pointwise(blocks, last):
            # Pair/point caps already enforced inside iter_candidates_pointwise.
            add_block(c)
            result = search()
            if result is not None:
                return result
            remove_block()
        return None

    print(f"  starting backtrack with B1={B1}, B2={B2}", flush=True)
    result = search()
    elapsed = _time.monotonic() - t_start
    if result is None:
        print(
            f"  search EXHAUSTED: no SBIBD with this canonical (B1, B2). "
            f"{nodes_visited[0]:_} nodes, deepest={deepest[0]}, {elapsed:.1f}s",
            flush=True,
        )
        return None
    print(f"  FOUND in {nodes_visited[0]:_} nodes, {elapsed:.1f}s", flush=True)
    return result


def search_sbibd_pt(
    *,
    chains: int = 4,
    iters_per_replica: int = 3_000_000,
    n_replicas: int = 6,
    swap_every: int = 1000,
    seed_base: int = 0,
) -> list[list[int]] | None:
    """Parallel-tempering search for (25, 9, 3) SBIBD using 2-row swaps.

    Init: circulant N with N[i, j] = 1 iff (j - i) mod 25 ∈ {0, ..., 8}.
    All row sums and col sums are 9 by construction. The 2-row swap move
    preserves both row and col sums, so we only minimise off-diagonal
    energy E = sum_{i<j} (NN^T[i,j] - 3)².

    Replicas at geometric temperatures; periodically attempt Metropolis
    swaps between adjacent-temperature replicas. The hot replicas explore;
    the cold replicas exploit; swaps shuttle good states down the ladder.
    """
    import math
    import random as _random

    target_off = 3
    n = 25

    def init_circulant() -> np.ndarray:
        N = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for k in range(9):
                N[i, (i + k) % n] = 1
        return N

    def offdiag_energy(N: np.ndarray) -> int:
        AAt = N @ N.T
        e = 0
        for i in range(n):
            for j in range(i + 1, n):
                e += (int(AAt[i, j]) - target_off) ** 2
        return e

    def two_row_swap_proposal(N: np.ndarray, rng: _random.Random) -> tuple[int, int, int, int] | None:
        """Pick two rows i ≠ j and find cols a, b with N[i,a]=1, N[i,b]=0,
        N[j,a]=0, N[j,b]=1. Return (i, j, a, b) or None if not found in
        a few tries.
        """
        for _ in range(20):
            i, j = rng.sample(range(n), 2)
            i_only = [c for c in range(n) if N[i, c] == 1 and N[j, c] == 0]
            j_only = [c for c in range(n) if N[i, c] == 0 and N[j, c] == 1]
            if not i_only or not j_only:
                continue
            a = rng.choice(i_only)
            b = rng.choice(j_only)
            return (i, j, a, b)
        return None

    def delta_after_swap(N: np.ndarray, AAt: np.ndarray, i: int, j: int, a: int, b: int) -> int:
        """Energy delta if we apply the (i, j, a, b) 2-row swap.

        After swap: N[i,a]=0, N[i,b]=1, N[j,a]=1, N[j,b]=0. Only rows i and
        j change; therefore AAt rows/cols i and j change.
        """
        de = 0
        for k in range(n):
            if k == i or k == j:
                continue
            old_ik = int(AAt[i, k])
            new_ik = old_ik - int(N[k, a]) + int(N[k, b])
            de += (new_ik - target_off) ** 2 - (old_ik - target_off) ** 2
            old_jk = int(AAt[j, k])
            new_jk = old_jk + int(N[k, a]) - int(N[k, b])
            de += (new_jk - target_off) ** 2 - (old_jk - target_off) ** 2
        # AAt[i, j] changes too: row i ⋅ row j after swap.
        # row_i and row_j each gain/lose one bit at cols a, b.
        # new dot = sum N[i,c]*N[j,c] over c with new values.
        # Differences: at col a: old N[i,a]N[j,a] = 1·0 = 0; new = 0·1 = 0. No change.
        # at col b: old = 0·1 = 0; new = 1·0 = 0. No change.
        # So AAt[i,j] is INVARIANT under this swap.
        return de

    best_overall: np.ndarray | None = None
    best_overall_e = float("inf")

    for chain in range(chains):
        rng = _random.Random(seed_base + chain * 1000)
        # n_replicas replicas, geometric temperatures
        t_min = 0.1
        t_max = 30.0
        temps = [t_min * (t_max / t_min) ** (k / (n_replicas - 1)) for k in range(n_replicas)]
        Ns = [init_circulant() for _ in range(n_replicas)]
        AAts = [N @ N.T for N in Ns]
        es = [offdiag_energy(N) for N in Ns]
        # Perturb each replica differently so they don't start identical
        for k in range(n_replicas):
            for _ in range(50 * (k + 1)):
                p = two_row_swap_proposal(Ns[k], rng)
                if p is None:
                    continue
                i, j, a, b = p
                Ns[k][i, a] = 0; Ns[k][i, b] = 1
                Ns[k][j, a] = 1; Ns[k][j, b] = 0
            AAts[k] = Ns[k] @ Ns[k].T
            es[k] = offdiag_energy(Ns[k])

        best_e = min(es)
        best_N = Ns[es.index(best_e)].copy()
        t0 = time.monotonic()

        for step in range(iters_per_replica):
            for k in range(n_replicas):
                p = two_row_swap_proposal(Ns[k], rng)
                if p is None:
                    continue
                i, j, a, b = p
                de = delta_after_swap(Ns[k], AAts[k], i, j, a, b)
                if de <= 0 or rng.random() < math.exp(-de / max(temps[k], 1e-9)):
                    # Apply
                    Ns[k][i, a] = 0; Ns[k][i, b] = 1
                    Ns[k][j, a] = 1; Ns[k][j, b] = 0
                    # Update AAt rows i and j (only these change)
                    AAts[k][i] = Ns[k] @ Ns[k][i]
                    AAts[k][:, i] = AAts[k][i]
                    AAts[k][j] = Ns[k] @ Ns[k][j]
                    AAts[k][:, j] = AAts[k][j]
                    es[k] += de
                    if es[k] < best_e:
                        best_e = es[k]
                        best_N = Ns[k].copy()
                        if best_e == 0:
                            break
            if best_e == 0:
                break
            if step % swap_every == swap_every - 1:
                # Try replica swap on a random adjacent pair
                k = rng.randrange(n_replicas - 1)
                de = (1.0 / temps[k] - 1.0 / temps[k + 1]) * (es[k + 1] - es[k])
                if de >= 0 or rng.random() < math.exp(de):
                    Ns[k], Ns[k + 1] = Ns[k + 1], Ns[k]
                    AAts[k], AAts[k + 1] = AAts[k + 1], AAts[k]
                    es[k], es[k + 1] = es[k + 1], es[k]
            if step % 50_000 == 0 and step > 0:
                hottest = es[-1]; coldest = es[0]
                print(f"  chain {chain} step {step}: best E={best_e}  cold={coldest} hot={hottest} ({time.monotonic()-t0:.1f}s)", flush=True)

        elapsed = time.monotonic() - t0
        print(f"  chain {chain} done: best E={best_e}  ({elapsed:.1f}s)", flush=True)
        if best_e < best_overall_e:
            best_overall_e = best_e
            best_overall = best_N.copy()
        if best_overall_e == 0:
            break

    if best_overall is not None and best_overall_e == 0:
        return [list(map(int, row)) for row in best_overall]
    return None


def search_sbibd_sa(
    *,
    chains: int = 8,
    iters_per_chain: int = 1_000_000,
    seed_base: int = 0,
) -> list[list[int]] | None:
    """SA search for a (25, 9, 3) SBIBD on a 25×25 {0,1} matrix.

    State: 25×25 {0,1} N with row sums = 9 (preserved by moves).
    Target: N N^T = 6 I + 3 J (so row dot products = 9 on diagonal,
    3 off-diagonal). We minimise the Frobenius energy
        E = sum_{i<j} (NN^T[i,j] - 3)² + sum_j (col_sum[j] - 9)².
    Moves: pick a row, swap a 1 and a 0 within that row (row-sum preserving).
    """
    import math
    import random as _random

    target_off = 3
    target_col = 9

    def energy(N: np.ndarray) -> int:
        n = N.shape[0]
        AAt = N @ N.T
        e = 0
        for i in range(n):
            for j in range(i + 1, n):
                e += (int(AAt[i, j]) - target_off) ** 2
        col_sums = N.sum(axis=0)
        for j in range(n):
            e += (int(col_sums[j]) - target_col) ** 2
        return e

    best_overall: np.ndarray | None = None
    best_overall_e = float("inf")
    for chain in range(chains):
        rng = _random.Random(seed_base + chain)
        # Init: each row has 9 random 1-positions.
        N = np.zeros((25, 25), dtype=np.int64)
        for i in range(25):
            cols = rng.sample(range(25), 9)
            for c in cols:
                N[i, c] = 1
        cur_e = energy(N)
        best_e = cur_e
        best_N = N.copy()
        t_start = max(cur_e * 0.05, 5.0)
        t_end = 0.05
        log_ratio = math.log(t_end / t_start)
        t0 = time.monotonic()
        for step in range(iters_per_chain):
            frac = step / max(1, iters_per_chain - 1)
            t = t_start * math.exp(log_ratio * frac)
            i = rng.randrange(25)
            ones = [c for c in range(25) if N[i, c] == 1]
            zeros = [c for c in range(25) if N[i, c] == 0]
            ca = rng.choice(ones)
            cb = rng.choice(zeros)
            # Compute delta E from this swap
            row_i = N[i].copy()
            row_i[ca] = 0
            row_i[cb] = 1
            # NN^T[i, k] changes by (row_i_new dot row_k) - (row_i_old dot row_k)
            #                     = N[k, cb] - N[k, ca].
            # Col sums: col_sum[ca] -= 1, col_sum[cb] += 1.
            de = 0
            for k in range(25):
                if k == i:
                    continue
                old_ik = int(N[i] @ N[k])
                new_ik = old_ik + int(N[k, cb]) - int(N[k, ca])
                de += (new_ik - target_off) ** 2 - (old_ik - target_off) ** 2
            col_sums = N.sum(axis=0)
            de += (int(col_sums[cb]) + 1 - target_col) ** 2 - (int(col_sums[cb]) - target_col) ** 2
            de += (int(col_sums[ca]) - 1 - target_col) ** 2 - (int(col_sums[ca]) - target_col) ** 2
            if de <= 0 or rng.random() < math.exp(-de / max(t, 1e-9)):
                N[i, ca] = 0
                N[i, cb] = 1
                cur_e += de
                if cur_e < best_e:
                    best_e = cur_e
                    best_N = N.copy()
                    if best_e == 0:
                        break
        elapsed = time.monotonic() - t0
        print(f"  chain {chain}: best E={best_e}  ({elapsed:.1f}s)")
        if best_e < best_overall_e:
            best_overall_e = best_e
            best_overall = best_N.copy()
        if best_overall_e == 0:
            break
    if best_overall is not None and best_overall_e == 0:
        return [list(map(int, row)) for row in best_overall]
    return None


def main() -> None:
    print(f"EW(25) = 7 · 24^12 = {EW_25} ({EW_25:.4e})")
    print("Searching Z_5 × Z_5 for a (25, 9, 3) difference set...")
    D = find_diff_set_z5xz5()
    if D is None:
        print("\nNo regular DS in Z_5 × Z_5. Falling back to backtracking SBIBD search.")
        result = search_sbibd_backtrack()
        if result is None:
            print("Backtracking search exhausted with no SBIBD found in this branch;")
            print("would need to vary the canonical (B1, B2) choice.")
            return
        print(f"\n  Backtracking found 25 blocks satisfying (25, 9, 3) constraints.")
        # Convert blocks → 25×25 incidence matrix N (rows = points, cols = blocks).
        N = [[0] * 25 for _ in range(25)]
        for j, B in enumerate(result):
            for p in B:
                N[p][j] = 1
        if not verify_sbibd(N):
            print("  SBIBD verification FAILED — this is a bug in the search")
            return
        print(f"  N N^T = 6 I + 3 J ✓ (verified (25, 9, 3) SBIBD)")
        M = [[2 * N[i][j] - 1 for j in range(25)] for i in range(25)]
        det = abs(integer_determinant(M))
        ratio = det / EW_25
        print(f"  |det(2N − J)| = {det}")
        print(f"  ratio to EW(25) = {ratio}")
        print(f"  matches EW exactly: {det == EW_25}")
        out = {
            "ew_25": EW_25,
            "det": det,
            "ratio": ratio,
            "ew_tight": det == EW_25,
            "blocks": result,
            "construction": "(25, 9, 3) SBIBD via backtracking from canonical (B1, B2); M = 2N - J",
        }
        (RESULTS / "H25_cycle50_sbibd.json").write_text(json.dumps(out, indent=2))
        np.save(RESULTS / "H25_cycle50_sbibd.npy", np.array(M, dtype=np.int8))
        print(f"\nSaved: results/H25_cycle50_sbibd.json + .npy")
        return

    print(f"  D = {D}")
    print("Building 25x25 incidence matrix N from D...")
    N = sbibd_from_diff_set(D)
    if not verify_sbibd(N):
        print("  SBIBD verification FAILED")
        return
    print(f"  N N^T = 6 I + 3 J ✓ (verified (25, 9, 3) SBIBD)")

    M = [[2 * N[i][j] - 1 for j in range(25)] for i in range(25)]
    det = abs(integer_determinant(M))
    ratio = det / EW_25
    print(f"  |det(2N − J)| = {det}")
    print(f"  ratio to EW(25) = {ratio}")
    print(f"  matches EW exactly: {det == EW_25}")

    out = {
        "ew_25": EW_25,
        "det": det,
        "ratio": ratio,
        "ew_tight": det == EW_25,
        "diff_set": D,
        "construction": "(25, 9, 3) SBIBD via Z_5 × Z_5 difference set, wrapped M = 2N - J",
    }
    (RESULTS / "H25_cycle50_sbibd.json").write_text(json.dumps(out, indent=2))
    np.save(RESULTS / "H25_cycle50_sbibd.npy", np.array(M, dtype=np.int8))
    print(f"\nSaved: results/H25_cycle50_sbibd.json + .npy")


if __name__ == "__main__":
    main()
