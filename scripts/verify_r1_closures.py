"""End-to-end exact-arithmetic verification of the n = 29 and n = 33 closures.

Executes and asserts every enumeration count and algebraic elimination in Section 6:
  1. Order n = 29:
     - Enumerates all 175 non-Barba {1, 3, 5}-supported Gram-class signatures
       (c1, c3, c5) with tr(G^2) < 25697.
     - Tests every canonical spread placement (p3, p5) with
       p3 + p5 + 2*((c3 - p3) + (c5 - p5)) <= 28 via exact Bareiss integer
       determinants and exact integer square roots.
     - Verifies the exact counts: 2 signatures with no spread placement,
       139 with no perfect-square spread placement, 34 yielding 44 perfect-square
       placements (26 above 329 * 7^12 * 2^28, 7 below 320 * 7^12 * 2^28, and
       11 in [320, 329) * 7^12 * 2^28).
     - Eliminates all 11 gap candidates in exact arithmetic: 4 by Hasse-Minkowski,
       2 by zero solutions to x^T G^{-1} x = 1, 3 by forced identical outside pairs,
       and 2 (at 322 * 7^12 * 2^28) by explicit exact integer Farkas certificates.
  2. Order n = 33:
     - Verifies in exact integer arithmetic that every shifted-block-diagonal
       (G - J_33 = 32I + 4E block-diagonal) configuration with a lambda_1 <= 0
       block lies strictly below Solomon's 441 * 2^74.
     - Exhaustively enumerates all A > 0 configurations with G - J_33
       block-diagonal and off-diagonal entries in {-7, -3, 1, 5, 9}
       (clique weights in {-2, -1, +1, +2}, plus isolated singletons) via exact
       arbitrary-precision integer branch-and-bound (zero float64 pruning):
       9,220 complete configurations attaining |det M| >= 441 * 2^74 across
       659,380 visited search-tree nodes (3,764 configurations across 54,991
       visited search-tree nodes for weights in {-1, +1}), yielding 24 exact
       perfect squares.
     - Eliminates all 24 perfect squares in exact arithmetic: 17 by Hasse-Minkowski,
       3 by zero column-norm solutions, 2 by forced identical or opposite rows,
       and 2 by exact column-moment obstructions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import sys
import time

import sympy as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hadamard_maxdet.feasibility import (
    check_r1_block_gram_columns,
    enumerate_r1_block_grams,
    hasse_minkowski_check,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Order n = 29: complete 175-signature spread-placement verification
# ---------------------------------------------------------------------------

def _forced_sign_n29(ra: int, rb: int, mag: int) -> int:
    for val in (mag, -mag):
        if (29 + ra + rb + val) % 4 == 0:
            return val
    raise AssertionError(f"No valid mod-4 sign for ra={ra}, rb={rb}, mag={mag}")


def build_n29_spread_gram(c3: int, c5: int, p3: int, p5: int) -> list[list[int]]:
    """Build the canonical spread-placement Gram matrix at n = 29.

    Row 0 is normalized to (+1)^29, rows 1..p3 have row sum 3, rows
    p3+1..p3+p5 have row sum 5, and the remaining 28 - p3 - p5 rows have row
    sum 1. All (c3 - p3) outside magnitude-3 pairs and (c5 - p5) outside
    magnitude-5 pairs are placed as mutually vertex-disjoint pairs inside the
    row-sum-1 vertex set (feasible iff p3 + p5 + 2*((c3 - p3) + (c5 - p5)) <= 28).
    """
    n = 29
    n_out3 = c3 - p3
    n_out5 = c5 - p5
    n_r1 = 28 - p3 - p5
    if n_out3 < 0 or n_out5 < 0 or 2 * (n_out3 + n_out5) > n_r1:
        raise ValueError("Invalid spread placement parameters")

    r = [29] + [3] * p3 + [5] * p5 + [1] * n_r1
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        G[i][i] = n
    for i in range(1, n):
        G[0][i] = G[i][0] = r[i]

    mag = [[1] * n for _ in range(n)]
    r1_vertices = list(range(1 + p3 + p5, n))
    idx = 0
    for _ in range(n_out3):
        a, b = r1_vertices[idx], r1_vertices[idx + 1]
        mag[a][b] = mag[b][a] = 3
        idx += 2
    for _ in range(n_out5):
        a, b = r1_vertices[idx], r1_vertices[idx + 1]
        mag[a][b] = mag[b][a] = 5
        idx += 2

    for i in range(1, n):
        for j in range(i + 1, n):
            G[i][j] = G[j][i] = _forced_sign_n29(r[i], r[j], mag[i][j])
    return G


def _eliminate_n29_column_eq(c3: int, c5: int, p3: int, p5: int, G: list[list[int]]) -> str:
    """Enumerate all orbit-sum solutions to x^T G^{-1} x = 1 for a spread placement."""
    Ginv = sp.Matrix(G).inv()
    n_out3 = c3 - p3
    n_out5 = c5 - p5
    r1_vertices = list(range(1 + p3 + p5, 29))
    out3_pairs = [(r1_vertices[2 * k], r1_vertices[2 * k + 1]) for k in range(n_out3)]
    out5_pairs = [(r1_vertices[2 * (n_out3 + k)], r1_vertices[2 * (n_out3 + k) + 1]) for k in range(n_out5)]
    used = {v for pair in (out3_pairs + out5_pairs) for v in pair}

    g3 = [i for i in range(1, 1 + p3) if i not in used]
    g5 = [i for i in range(1 + p3, 1 + p3 + p5) if i not in used]
    g1 = [i for i in range(1 + p3 + p5, 29) if i not in used]
    singles = [g for g in (g3, g5, g1) if g]

    pg_list: list[tuple[str, list[tuple[int, int]]]] = []
    if out3_pairs:
        pg_list.append(("-3", out3_pairs))
    if out5_pairs:
        pg_list.append(("+5", out5_pairs))

    sols_found: list[tuple[int, ...]] = []

    def _enum_pg(idx: int, x_assign: list[int], opp_flags: tuple[int, ...]) -> None:
        if idx == len(pg_list):
            def _enum_s(si: int, vec: list[int]) -> None:
                if si == len(singles):
                    v = sp.Matrix(vec)
                    if (v.T * Ginv * v)[0] == 1:
                        sols_found.append(opp_flags)
                    return
                orb = singles[si]
                m_o = len(orb)
                for k_pos in range(m_o + 1):
                    for t_i, row_idx in enumerate(orb):
                        vec[row_idx] = 1 if t_i < k_pos else -1
                    _enum_s(si + 1, vec)

            _enum_s(0, list(x_assign))
            return

        _, pairs = pg_list[idx]
        mp = len(pairs)
        for n_opp in range(mp + 1):
            for n_pos in range(mp - n_opp + 1):
                vec = list(x_assign)
                for j, (a, b) in enumerate(pairs):
                    if j < n_opp:
                        vec[a], vec[b] = 1, -1
                    elif j < n_opp + n_pos:
                        vec[a], vec[b] = 1, 1
                    else:
                        vec[a], vec[b] = -1, -1
                _enum_pg(idx + 1, vec, opp_flags + (n_opp,))

    _enum_pg(0, [1] * 29, ())
    if len(sols_found) == 0:
        return "Column eq: 0 solutions"
    forced_labels = [
        label for g_idx, (label, _) in enumerate(pg_list)
        if all(f[g_idx] == 0 for f in sols_found)
    ]
    assert forced_labels, "Expected zero solutions or forced identical outside pairs"
    return f"Column eq: {len(sols_found)} types, all force identical {'/'.join(forced_labels)} pairs"


def _verify_n29_farkas(p3: int, p5: int, y_int: list[int], G: list[list[int]]) -> str:
    """Verify exact integer Farkas certificate y^T A_j >= 0, y^T b < 0 for n = 29."""
    p1 = 28 - p3 - p5
    Ginv = sp.Matrix(G).inv()
    g3 = list(range(1, 1 + p3))
    g5 = list(range(1 + p3, 1 + p3 + p5))
    g1 = list(range(1 + p3 + p5, 29))

    d3, o3 = Ginv[g3[0], g3[0]], Ginv[g3[0], g3[1]]
    d5, o5 = Ginv[g5[0], g5[0]], Ginv[g5[0], g5[1]]
    d1, o1 = Ginv[g1[0], g1[0]], Ginv[g1[0], g1[1]]
    const = Ginv[0, 0] + p3 * (d3 - o3) + p5 * (d5 - o5) + p1 * (d1 - o1)
    l3, l5, l1 = 2 * Ginv[0, g3[0]], 2 * Ginv[0, g5[0]], 2 * Ginv[0, g1[0]]
    c35 = 2 * Ginv[g3[0], g5[0]]
    c31 = 2 * Ginv[g3[0], g1[0]]
    c51 = 2 * Ginv[g5[0], g1[0]]

    sols = [
        (s3, s5, s1)
        for s3 in range(-p3, p3 + 1, 2)
        for s5 in range(-p5, p5 + 1, 2)
        for s1 in range(-p1, p1 + 1, 2)
        if (
            const
            + l3 * s3 + l5 * s5 + l1 * s1
            + o3 * s3 * s3 + o5 * s5 * s5 + o1 * s1 * s1
            + c35 * s3 * s5 + c31 * s3 * s1 + c51 * s5 * s1
            == 1
        )
    ]
    min_lhs = min(
        sum(
            y_int[k] * [1, s3, s5, s1, s3 * s3, s5 * s5, s1 * s1, s3 * s5, s3 * s1, s5 * s1][k]
            for k in range(10)
        )
        for s3, s5, s1 in sols
    )
    assert min_lhs >= 0, f"Farkas violation: min y^T A_j = {min_lhs}"

    gsum = lambda A, B: sum(G[i][j] for i in A for j in B)
    rhs = [
        29,
        3 * p3,
        5 * p5,
        p1,
        gsum(g3, g3),
        gsum(g5, g5),
        gsum(g1, g1),
        gsum(g3, g5),
        gsum(g3, g1),
        gsum(g5, g1),
    ]
    yb = sum(y_int[k] * rhs[k] for k in range(10))
    assert yb < 0, f"Farkas RHS not strictly negative: {yb}"
    return f"Exact integer Farkas ({len(sols)} col types, y={y_int}, min y^T A_j={min_lhs}, y^T b={yb})"


def verify_n29() -> None:
    t0 = time.monotonic()
    unit = (7 ** 12) * (2 ** 28)
    lb = 320 * unit
    ub = 329 * unit

    # Enumerate all non-Barba (c3 + c5 >= 1) signatures with tr(G^2) < 25697
    sigs: list[tuple[int, int, int, int]] = []
    for c5 in range(0, 407):
        for c3 in range(0, 407 - c5):
            if c3 == 0 and c5 == 0:
                continue
            c1 = 406 - c3 - c5
            tr2 = 29 ** 3 + 2 * (c1 + 9 * c3 + 25 * c5)
            if tr2 < 25697:
                sigs.append((c1, c3, c5, tr2))

    assert len(sigs) == 175, f"Expected 175 signatures, got {len(sigs)}"

    no_spread_sigs: list[tuple[int, int, int]] = []
    no_square_count = 0
    with_square_count = 0
    sq_placements: list[tuple[int, int, int, int, int, int]] = []

    for c1, c3, c5, _ in sigs:
        valid_ps = [
            (p3, p5)
            for p3 in range(0, c3 + 1)
            for p5 in range(0, c5 + 1)
            if p3 + p5 + 2 * ((c3 - p3) + (c5 - p5)) <= 28
        ]
        if not valid_ps:
            no_spread_sigs.append((c1, c3, c5))
            continue

        found_sq = False
        for p3, p5 in valid_ps:
            G = build_n29_spread_gram(c3, c5, p3, p5)
            detG = integer_determinant(G)
            if detG > 0:
                root = math.isqrt(detG)
                if root * root == detG:
                    found_sq = True
                    sq_placements.append((c1, c3, c5, p3, p5, root))
        if found_sq:
            with_square_count += 1
        else:
            no_square_count += 1

    assert len(no_spread_sigs) == 2
    assert no_square_count == 139
    assert with_square_count == 34
    assert len(sq_placements) == 44

    above_ub = [x for x in sq_placements if x[5] >= ub]
    below_lb = [x for x in sq_placements if x[5] < lb]
    in_gap = sorted([x for x in sq_placements if lb <= x[5] < ub], key=lambda x: (-x[5], x[0], x[1], x[3]))

    assert len(above_ub) == 26
    assert len(below_lb) == 7
    assert len(in_gap) == 11
    assert sum(x[5] == ub for x in above_ub) == 2, "Expected exactly 2 placements at 329 * 7^12 * 2^28"

    print("=== Order n = 29: 175 low-trace signatures ===")
    print(f"Total signatures with tr(G^2) < 25697: {len(sigs)}")
    print(f"  No spread placement: {len(no_spread_sigs)} {no_spread_sigs}")
    print(f"  No perfect-square spread placement: {no_square_count}")
    print(f"  With >= 1 perfect-square spread placement: {with_square_count} ({len(sq_placements)} placements)")
    print(f"    Above UB (>= 329): {len(above_ub)}")
    print(f"    Below LB (< 320):  {len(below_lb)}")
    print(f"    In gap [320, 329): {len(in_gap)}")

    # Explicit exact integer Farkas witnesses for the two 322 * 7^12 * 2^28 candidates
    farkas_witnesses = {
        (389, 14, 3, 14, 3): [-18, 0, -1, -3, 1, 0, 0, 1, 0, 0],
        (393, 8, 5, 8, 5): [0, 0, 0, 7, 0, 0, -1, 0, 0, -2],
    }

    counts = Counter()
    for c1, c3, c5, p3, p5, root in in_gap:
        G = build_n29_spread_gram(c3, c5, p3, p5)
        hr = hasse_minkowski_check(G)
        assert hr.psd and hr.det_is_square and hr.abs_det == root
        mult = root / unit
        if not hr.hasse_ok:
            counts["hasse"] += 1
            reason = f"Hasse-Minkowski fails {hr.bad_primes}"
        elif (c1, c3, c5, p3, p5) in farkas_witnesses:
            counts["farkas"] += 1
            reason = _verify_n29_farkas(p3, p5, farkas_witnesses[(c1, c3, c5, p3, p5)], G)
        else:
            reason = _eliminate_n29_column_eq(c3, c5, p3, p5, G)
            if "0 solutions" in reason:
                counts["col_zero"] += 1
            else:
                counts["col_forced"] += 1
        print(f"  ({c1},{c3:2d},{c5}) p=({p3:2d},{p5})  mult={mult:7.3f}  -> {reason}")

    assert counts == Counter({"hasse": 4, "col_zero": 2, "col_forced": 3, "farkas": 2}), counts
    print(f"n = 29 VERIFIED in {time.monotonic() - t0:.2f}s: all 11 gap candidates eliminated.\n")


# ---------------------------------------------------------------------------
# Order n = 33: complete shifted-block-diagonal {-7, -3, 1, 5, 9} verification
# ---------------------------------------------------------------------------

def _verify_n33_non_pd_A_bounds(solomon: int) -> None:
    """Verify in exact integer arithmetic that every configuration with a block
    having lambda_1 = 8 + (s - 1)*w <= 0 satisfies det(G) < solomon^2.
    """
    sol_sq = solomon * solomon

    # Case 1: lambda_1 = 0 -> (9, -1) and (5, -2)
    for s, w in ((9, -1), (5, -2)):
        assert 8 + (s - 1) * w == 0
        det_ub = (4 ** 32) * s * ((8 - w) ** (s - 1)) * (8 ** (33 - s))
        assert det_ub < sol_sq
        print(f"  lambda_1 = 0 block ({s}, {w:+d}): sqrt(det_ub)/2^74 = {math.sqrt(det_ub)/(2**74):.4f} < 441")

    # Case 2: lambda_1 < 0 -> (s >= 10, w = -1) or (s >= 6, w = -2)
    max_neg_val = 0.0
    for w, s_min in ((-1, 10), (-2, 6)):
        for s in range(s_min, 34):
            lam1_abs = -(8 + (s - 1) * w)
            assert lam1_abs > 0
            if s - 4 * lam1_abs <= 0:
                continue  # Cannot even make G positive definite
            det_ub = (4 ** 32) * ((8 - w) ** (s - 1)) * (s - 4 * lam1_abs) * (8 ** (33 - s))
            assert det_ub < sol_sq
            val = math.sqrt(det_ub) / (2 ** 74)
            max_neg_val = max(max_neg_val, val)
    assert max_neg_val < 267.0
    print(f"  lambda_1 < 0 blocks: max sqrt(det_ub)/2^74 = {max_neg_val:.4f} < 267 < 441")


def _verify_n33_moment_survivors() -> None:
    """Verify exact column-moment obstructions for the final 2 Hasse-passing survivors."""
    # Survivor 1: 13*(2, -1) [A] + 2*(2, +1) [B] + 1*(3, +1) [C]
    a_inv_1 = sp.Rational(13 * 2, 7) + sp.Rational(2 * 2, 9) + sp.Rational(3, 10)
    denom1 = 4 * (4 + a_inv_1)
    types1 = set()
    for nA2 in range(14):
        for nB2 in range(3):
            for sC in (-3, -1, 1, 3):
                qA = sp.Rational(2, 7) * nA2 + sp.Rational(2, 9) * (13 - nA2)
                qB = sp.Rational(2, 9) * nB2 + sp.Rational(2, 7) * (2 - nB2)
                qC = sp.Rational(sC * sC, 30) + sp.Rational(9 - sC * sC, 21)
                q_tot = qA + qB + qC
                for pA in range(nA2 + 1):
                    sA = 2 * (2 * pA - nA2)
                    for pB in range(nB2 + 1):
                        sB = 2 * (2 * pB - nB2)
                        L = sp.Rational(sA, 7) + sp.Rational(sB, 9) + sp.Rational(sC, 10)
                        if q_tot / 4 - (L * L) / denom1 == 1:
                            types1.add((nA2, nB2, sC * sC))
    assert sorted(types1) == [(4, 0, 1), (6, 2, 1), (9, 1, 9)]
    n2 = (129 - 33) // 8
    n1 = (38 - n2) // 2
    n0 = 33 - n1 - n2
    assert (n0, n1, n2) == (8, 13, 12)
    assert 4 * n0 + 6 * n1 + 9 * n2 == 218 != 195
    print("  Survivor 1 [13*(2,-1) + 2*(2,+1) + 1*(3,+1)]: 3 types -> (n0,n1,n2)=(8,13,12), 218 != 195")

    # Survivor 2: 12*(2, -1) [A] + 3*(2, +1) [B] + 1*(3, -1) [C]
    a_inv_2 = sp.Rational(12 * 2, 7) + sp.Rational(3 * 2, 9) + sp.Rational(3, 6)
    denom2 = 4 * (4 + a_inv_2)
    types2 = set()
    for nA2 in range(13):
        for nB2 in range(4):
            for sC in (-3, -1, 1, 3):
                qA = sp.Rational(2, 7) * nA2 + sp.Rational(2, 9) * (12 - nA2)
                qB = sp.Rational(2, 9) * nB2 + sp.Rational(2, 7) * (3 - nB2)
                qC = sp.Rational(sC * sC, 18) + sp.Rational(9 - sC * sC, 27)
                q_tot = qA + qB + qC
                for pA in range(nA2 + 1):
                    sA = 2 * (2 * pA - nA2)
                    for pB in range(nB2 + 1):
                        sB = 2 * (2 * pB - nB2)
                        L = sp.Rational(sA, 7) + sp.Rational(sB, 9) + sp.Rational(sC, 6)
                        if q_tot / 4 - (L * L) / denom2 == 1:
                            types2.add((nA2, nB2, sC * sC, sA * sB))
    assert len(types2) == 9
    pos_ab = [t for t in sorted(types2) if t[3] > 0]
    assert pos_ab == [(3, 3, 9, 12)]
    max_cols_sC9 = (81 - 33) // 8
    assert max_cols_sC9 == 6 and 12 * max_cols_sC9 == 72 < 144
    print("  Survivor 2 [12*(2,-1) + 3*(2,+1) + 1*(3,-1)]: 9 types -> max sum(sA*sB) <= 12*6 = 72 < 144")


def verify_n33() -> None:
    t0 = time.monotonic()
    solomon = 441 * (2 ** 74)
    print("=== Order n = 33: shifted-block-diagonal (G - J_33 block-diagonal) {-7, -3, 1, 5, 9} enumeration ===")
    _verify_n33_non_pd_A_bounds(solomon)

    checked_w1, above_w1, cands_w1 = enumerate_r1_block_grams(
        33, solomon, weights=(-1, 1), return_above_count=True
    )
    assert checked_w1 == 54991 and above_w1 == 3764 and len(cands_w1) == 12
    print(
        f"  Exact integer DFS (weights in {{-1, +1}}):       "
        f"{above_w1:,} complete A>0 configs >= threshold "
        f"({checked_w1:,} visited search-tree nodes) -> {len(cands_w1)} squares"
    )

    checked_w2, above_w2, cands_w2 = enumerate_r1_block_grams(
        33, solomon, weights=(-2, -1, 1, 2), return_above_count=True
    )
    assert checked_w2 == 659380 and above_w2 == 9220 and len(cands_w2) == 24
    print(
        f"  Exact integer DFS (weights in {{-2, -1, +1, +2}}): "
        f"{above_w2:,} complete A>0 configs >= threshold "
        f"({checked_w2:,} visited search-tree nodes) -> {len(cands_w2)} squares"
    )

    counts = Counter()
    moment_blocks = []
    for idx, cand in enumerate(cands_w2, start=1):
        mult = cand.abs_det / (2 ** 74)
        cnt = dict(Counter(cand.blocks))
        if not cand.hasse_ok:
            counts["hasse"] += 1
            status = f"Hasse-Minkowski fails {cand.bad_primes}"
        else:
            cr = check_r1_block_gram_columns(33, cand.blocks)
            if cr.ruled_out_reason == "no_column_solutions":
                counts["col_zero"] += 1
                status = "Column eq: 0 solutions"
            elif cr.ruled_out_reason is not None:
                counts["col_forced"] += 1
                status = f"Column eq: {cr.ruled_out_reason} ({cr.num_column_types} types)"
            else:
                counts["moment"] += 1
                moment_blocks.append(cnt)
                status = f"Survives col eq ({cr.num_column_types} types) -> exact moment obstruction"
        print(f"  [{idx:2d}] mult={mult:8.4f}*2^74  blocks={cnt} -> {status}")

    assert counts == Counter({"hasse": 17, "col_zero": 3, "col_forced": 2, "moment": 2}), counts
    assert sorted(tuple(sorted(b.items())) for b in moment_blocks) == [
        (((2, -1), 12), ((2, 1), 3), ((3, -1), 1)),
        (((2, -1), 13), ((2, 1), 2), ((3, 1), 1)),
    ]
    _verify_n33_moment_survivors()
    print(
        f"n = 33 VERIFIED in {time.monotonic() - t0:.2f}s: all {above_w2:,} complete A>0 configurations >= 441*2^74 "
        f"eliminated ({checked_w2:,} visited search-tree nodes, 0 survivors).\n"
    )


def main() -> None:
    t_all = time.monotonic()
    verify_n29()
    verify_n33()
    print(f"ALL R1 CLOSURES VERIFIED in {time.monotonic() - t_all:.2f}s.")


if __name__ == "__main__":
    main()
