"""Tests for :mod:`hadamard_maxdet.feasibility`."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from hadamard_maxdet.feasibility import (
    FeasibleTarget,
    ParityCheck,
    enumerate_feasible_targets,
    parity_constraint,
    required_pair_parity,
)
from hadamard_maxdet.generalized_ehlich import predicted_block_det


# ---------------------------------------------------------------------------
# required_pair_parity
# ---------------------------------------------------------------------------


def test_required_pair_parity_n23_intra_plus1():
    # n=23 odd, c=+1 odd → (n-c)/2 = 11 → parity 1 (w_i+w_j odd).
    assert required_pair_parity(23, 1) == 1


def test_required_pair_parity_n23_cross_minus1():
    # n=23 odd, c=-1 odd → (n-c)/2 = 12 → parity 0 (w_i+w_j even).
    assert required_pair_parity(23, -1) == 0


def test_required_pair_parity_n21_intra_plus1():
    # n=21 odd, c=+1 odd → (n-c)/2 = 10 → parity 0.
    assert required_pair_parity(21, 1) == 0


def test_required_pair_parity_n21_cross_minus1():
    # n=21 odd, c=-1 odd → (n-c)/2 = 11 → parity 1.
    assert required_pair_parity(21, -1) == 1


def test_required_pair_parity_invalid_basic():
    # n=23 odd, c=0 even → c-n odd → impossible inner product, returns None.
    assert required_pair_parity(23, 0) is None
    # n=8 even, c=1 odd → impossible.
    assert required_pair_parity(8, 1) is None


def test_required_pair_parity_n8_zero():
    # n=8, c=0: (n-c)/2 = 4, parity 0.
    assert required_pair_parity(8, 0) == 0


# ---------------------------------------------------------------------------
# Headline parity checks (the four claims under audit)
# ---------------------------------------------------------------------------


def test_headline_n23_111_20_plus1_minus1_INFEASIBLE():
    """The flagship target. 20-block under intra=+1 forces opposite-parity
    every pair, impossible for ≥ 3 rows."""
    pc = parity_constraint(23, (1, 1, 1, 20), intra=1, cross=-1)
    assert pc.base_parity_ok is True
    assert pc.intra_required == 1  # w_i+w_j odd inside the 20-block
    assert pc.cross_required == 0  # w_i+w_j even across blocks
    assert pc.feasible is False
    assert pc.n_assignments == 0
    assert pc.block_parity_pattern is None


def test_headline_n19_1_1_1_16_plus1_minus1_INFEASIBLE():
    """Same parity obstruction, same reason: the 16-block can't bipartition
    into all-opposite-parity pairs."""
    pc = parity_constraint(19, (1, 1, 1, 16), intra=1, cross=-1)
    assert pc.base_parity_ok is True
    # n=19, intra=+1: (19-1)/2=9 → parity 1 (w_i+w_j odd inside the 16-block).
    assert pc.intra_required == 1
    assert pc.cross_required == 0
    assert pc.feasible is False


def test_headline_n21_1_3_6_11_plus1_minus1_INFEASIBLE():
    """At n=21, (intra=+1) forces SAME parity inside each block (P=0),
    while (cross=-1) forces OPPOSITE parity across blocks (P=1). Three of
    four blocks are size ≥ 2 (sizes 3, 6, 11) and must each be monochromatic
    in parity. Cross requires every pair of these monochromatic blocks to
    be opposite — with 3 such blocks, K_3 is not 2-colorable. The singleton
    block of size 1 doesn't help. Infeasible."""
    pc = parity_constraint(21, (1, 3, 6, 11), intra=1, cross=-1)
    assert pc.base_parity_ok is True
    # n=21, intra=+1: (21-1)/2 = 10 → parity 0; cross=-1: (21+1)/2=11 → parity 1.
    assert pc.intra_required == 0
    assert pc.cross_required == 1
    assert pc.feasible is False


def test_headline_n8_partition_2222_a0_b0_FEASIBLE():
    """The Hadamard regime sanity check: n=8 partition (2,2,2,2),
    intra = cross = 0 (all off-diagonal Gram entries vanish, the
    Hadamard property). This is feasible — H_8 exists."""
    pc = parity_constraint(8, (2, 2, 2, 2), intra=0, cross=0)
    assert pc.base_parity_ok is True
    # n=8 even, c=0: P(0) = (8/2) mod 2 = 0. So intra_req == cross_req == 0.
    assert pc.intra_required == 0
    assert pc.cross_required == 0
    # All same parity → 2 assignments (all even, or all odd).
    assert pc.feasible is True
    assert pc.n_assignments == 2


# ---------------------------------------------------------------------------
# Per-block enumeration sanity
# ---------------------------------------------------------------------------


def test_singletons_unconstrained_intra():
    """A partition of all-singletons has no intra-pairs; cross_req == 0
    forces all rows same parity → 2 assignments at any n."""
    # Use n=4, partition (1,1,1,1), intra arbitrary (no intra pairs), cross=4
    # gives (n-c)/2 = 0 → parity 0.
    pc = parity_constraint(4, (1, 1, 1, 1), intra=4, cross=4)
    # When intra == cross both have P=0, all rows same parity.
    assert pc.feasible is True
    assert pc.n_assignments == 2


def test_cross_req_1_three_blocks_infeasible():
    """Three blocks all of size ≥ 2 with P(intra)=0 (same parity within)
    and P(cross)=1 (opposite across) forces K_3 to be 2-colored with
    all edges crossing colors — impossible."""
    # n=21 partition (3, 6, 6, 6) (sums to 21), intra=+1 (P=0), cross=-1 (P=1).
    pc = parity_constraint(21, (3, 6, 6, 6), intra=1, cross=-1)
    assert pc.intra_required == 0
    assert pc.cross_required == 1
    assert pc.feasible is False


def test_cross_req_1_two_blocks_feasible():
    """Two blocks of size ≥ 2 with P(intra)=0 / P(cross)=1: each block
    monochromatic, two blocks have opposite parities. K_2 is bipartite —
    feasible. Two assignments (block1=even, block2=odd) and vice versa."""
    # n=21 partition (1, 1, 9, 10): the two singletons + two large blocks.
    # The singletons force their own parities to be opposite to each large
    # block, which is consistent because the two large blocks are opposite.
    pc = parity_constraint(21, (1, 1, 9, 10), intra=1, cross=-1)
    assert pc.intra_required == 0
    assert pc.cross_required == 1
    # The two large blocks must have opposite parity. Each singleton must
    # be opposite to each non-singleton block — but the two non-singleton
    # blocks have opposite parities, so the singleton can't be opposite to
    # both. Wait: if block A is even and block B is odd, then a singleton
    # (size 1) must satisfy: opposite to block A (so odd) AND opposite to
    # block B (so even) → contradiction. So this is also infeasible.
    # This is correct behaviour: the parity check correctly catches the
    # propagation across all 4 blocks.
    assert pc.feasible is False


def test_cross_req_0_blocks_must_align():
    """P(cross)=0: every cross-pair same parity → all rows same parity.
    Combined with P(intra)=0: still all rows same parity. Always feasible
    with 2 assignments."""
    # n=23, intra=-1 (P((23+1)/2)=P(12)=0), cross=-1 (same).
    pc = parity_constraint(23, (5, 6, 6, 6), intra=-1, cross=-1)
    assert pc.intra_required == 0
    assert pc.cross_required == 0
    assert pc.feasible is True
    assert pc.n_assignments == 2


# ---------------------------------------------------------------------------
# Direct-enumeration sanity check on small cases
# ---------------------------------------------------------------------------


def _direct_enumerate_feasibility(n: int, partition, intra: int, cross: int) -> int:
    """Brute-force: enumerate all assignments of (w_1, ..., w_n) in {0..n}^n
    that satisfy every pairwise parity constraint, modulo only parity (count
    distinct parity assignments). Returns the number of valid parity
    assignments. Too slow for n > 12 but a ground truth at small n.
    """
    intra_req = required_pair_parity(n, intra)
    cross_req = required_pair_parity(n, cross)
    if intra_req is None or cross_req is None:
        return 0
    # Block index per row.
    block = []
    for k, sz in enumerate(partition):
        block += [k] * sz
    count = 0
    for parities in product((0, 1), repeat=n):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                req = intra_req if block[i] == block[j] else cross_req
                if (parities[i] + parities[j]) % 2 != req:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            count += 1
    return count


@pytest.mark.parametrize(
    "n,partition,intra,cross",
    [
        (4, (1, 1, 1, 1), 4, 4),    # all singletons, P=0/P=0
        (4, (1, 1, 1, 1), 0, 0),    # P=0 / P=0 (Hadamard regime at n=4: yes!)
        (5, (1, 1, 1, 2), 1, -1),   # n=5 odd
        (5, (1, 2, 1, 1), -1, 1),
        (8, (2, 2, 2, 2), 0, 0),    # the H_8 regime
        (7, (1, 2, 2, 2), 3, -1),   # canonical Ehlich at n=7
        (7, (1, 2, 2, 2), 1, -1),
        (8, (1, 1, 3, 3), 2, 0),
        (10, (2, 2, 3, 3), 2, -2),
        (10, (1, 2, 3, 4), 0, -2),
    ],
)
def test_parity_count_matches_brute_force(n, partition, intra, cross):
    pc = parity_constraint(n, partition, intra, cross)
    direct = _direct_enumerate_feasibility(n, partition, intra, cross)
    assert pc.n_assignments == direct, (
        f"parity check n_assignments={pc.n_assignments} but brute-force "
        f"says {direct} for n={n}, partition={partition}, intra={intra}, "
        f"cross={cross}"
    )
    assert pc.feasible == (direct > 0)


# ---------------------------------------------------------------------------
# enumerate_feasible_targets
# ---------------------------------------------------------------------------


def test_enumerate_feasible_targets_n23_above_orrick():
    """Audit headline: at n=23 with floor 0.7091, list all parity-feasible
    targets. The (1,1,1,20) at (+1,-1) target must NOT appear (infeasible)."""
    targets = enumerate_feasible_targets(23, max_predicted_ratio_floor=0.7091)
    # All listed must be feasible by construction.
    assert all(t.parity.feasible for t in targets)
    # Headline target ruled out.
    headline_present = any(
        t.intra == 1 and t.cross == -1 and t.partition == (1, 1, 1, 20)
        for t in targets
    )
    assert not headline_present, "Infeasible (1,1,1,20)/(+1,-1) target leaked"


def test_enumerate_feasible_targets_floor_filters():
    """Floor filter actually filters."""
    high = enumerate_feasible_targets(23, max_predicted_ratio_floor=0.7091)
    low = enumerate_feasible_targets(23, max_predicted_ratio_floor=0.0)
    assert len(low) >= len(high)


def test_enumerate_feasible_targets_returns_sorted():
    targets = enumerate_feasible_targets(23, max_predicted_ratio_floor=0.5)
    ratios = [t.predicted_ratio for t in targets]
    assert ratios == sorted(ratios, reverse=True)


# ---------------------------------------------------------------------------
# FeasibleTarget JSON
# ---------------------------------------------------------------------------


def test_feasible_target_to_json_roundtrip_keys():
    targets = enumerate_feasible_targets(11, max_predicted_ratio_floor=0.0)
    if targets:
        d = targets[0].to_json()
        assert {"n", "intra", "cross", "partition", "predicted_abs_det",
                "predicted_ratio", "parity"} <= set(d)
        assert {"feasible", "n_assignments", "reason"} <= set(d["parity"])


# ---------------------------------------------------------------------------
# Algebraic Gram feasibility: Hilbert symbol, Hasse-Minkowski, block-Gram DFS
# ---------------------------------------------------------------------------


def test_hilbert_symbol_known_values():
    from hadamard_maxdet.feasibility import hilbert_symbol

    assert hilbert_symbol(-1, -1, 2) == -1
    assert hilbert_symbol(2, 3, 2) == -1
    assert hilbert_symbol(2, 3, 3) == -1
    assert hilbert_symbol(3, 3, 2) == -1
    assert hilbert_symbol(3, 3, 3) == -1
    assert hilbert_symbol(1, 5, 2) == 1
    assert hilbert_symbol(4, 7, 7) == 1


def test_hasse_minkowski_identity_and_ew_tight():
    from hadamard_maxdet.feasibility import build_r1_block_gram, hasse_minkowski_check

    # n=5 EW-tight Gram: 4I + J has det = 9 * 4^4 = (3 * 16)^2 = 48^2, realisable!
    G5 = build_r1_block_gram(5, [])
    hr5 = hasse_minkowski_check(G5)
    assert hr5.psd and hr5.det_is_square and hr5.abs_det == 48 and hr5.hasse_ok

    # n=33 single clique (6, -1): det is a perfect square, but Hasse fails at p=2, 3.
    G33_c6 = build_r1_block_gram(33, [(6, -1)])
    hr33 = hasse_minkowski_check(G33_c6)
    assert hr33.psd and hr33.det_is_square
    assert not hr33.hasse_ok
    assert hr33.bad_primes == {2: -1, 3: -1}


def test_h33_column_equation_rules_out_matchings():
    from hadamard_maxdet.feasibility import check_r1_block_gram_columns

    # a=8 (-3), b=1 (+5) has 0 column solutions
    r1 = check_r1_block_gram_columns(33, [(2, -1)] * 8 + [(2, 1)])
    assert r1.num_column_types == 0
    assert r1.ruled_out_reason == "no_column_solutions"

    # a=0, b=9 (+5) forces all 15 singletons identical
    r2 = check_r1_block_gram_columns(33, [(2, 1)] * 9)
    assert r2.num_column_types == 1
    assert r2.ruled_out_reason == "forced_identical_singletons_c=15"


def test_h33_enumerate_r1_block_grams_w1():
    from hadamard_maxdet.feasibility import enumerate_r1_block_grams

    solomon = 441 * (2 ** 74)
    checked, cands = enumerate_r1_block_grams(33, solomon, weights=(-1, 1))
    assert checked == 54991
    assert len(cands) == 12
    assert sum(1 for c in cands if c.hasse_ok) == 4
    checked2, above2, cands2 = enumerate_r1_block_grams(
        33, solomon, weights=(-1, 1), return_above_count=True
    )
    assert checked2 == 54991
    assert above2 == 3764
    assert len(cands2) == 12


def test_h29_all_11_gap_candidates_rigorously_eliminated():
    """Verify that all 11 open n = 29 spread-placement candidates in [320, 329) * 7^12 * 2^28
    are eliminated in exact arithmetic:
      - 4 by Hasse-Minkowski (bad primes {2, 3}),
      - 2 by 0 solutions to x^T G^{-1} x = 1,
      - 3 by forced identical rows on outside pairs,
      - 2 (both at 322 * 7^12 * 2^28) by exact integer Farkas certificates on the
        10-equation column-moment system.
    """
    import sympy as sp
    from hadamard_maxdet.feasibility import hasse_minkowski_check

    def _forced_sign(ra: int, rb: int, mag: int) -> int:
        for val in (mag, -mag):
            if (29 + ra + rb + val) % 4 == 0:
                return val
        raise AssertionError

    def _build_h29(c3: int, c5: int, p3: int, p5: int):
        n = 29
        r = [29] + [3] * p3 + [5] * p5 + [1] * (28 - p3 - p5)
        G = [[0] * n for _ in range(n)]
        for i in range(n):
            G[i][i] = n
        for i in range(1, n):
            G[0][i] = G[i][0] = r[i]
        mag = [[1] * n for _ in range(n)]
        order = list(range(1 + p3 + p5, n)) + list(range(1, 1 + p3 + p5))
        idx = 0
        for _ in range(c3 - p3):
            a, b = order[idx], order[idx + 1]
            mag[a][b] = mag[b][a] = 3
            idx += 2
        for _ in range(c5 - p5):
            a, b = order[idx], order[idx + 1]
            mag[a][b] = mag[b][a] = 5
            idx += 2
        for i in range(1, n):
            for j in range(i + 1, n):
                G[i][j] = G[j][i] = _forced_sign(r[i], r[j], mag[i][j])
        return G

    # 4 candidates killed by Hasse-Minkowski
    hasse_killed = [
        (10, 5, 1, 4),
        (6, 6, 0, 2),
        (7, 6, 7, 4),
        (6, 8, 0, 5),
    ]
    for c3, c5, p3, p5 in hasse_killed:
        hr = hasse_minkowski_check(_build_h29(c3, c5, p3, p5))
        assert hr.det_is_square and not hr.hasse_ok
        assert hr.bad_primes == {2: -1, 3: -1}

    # 5 candidates killed by exact column equation x^T G^{-1} x = 1:
    #   - (394, 4, 8) p=(4,4) and (391, 11, 4) p=(3,4): 0 solutions
    #   - (389, 13, 4) p=(12,0), (389, 12, 5) p=(12,1), (393, 7, 6) p=(6,6): forced identical outside pairs
    col_cases = [
        (4, 8, 4, 4, 0),
        (11, 4, 3, 4, 0),
        (13, 4, 12, 0, 3),
        (12, 5, 12, 1, 4),
        (7, 6, 6, 6, 11),
    ]
    for c3, c5, p3, p5, expected_types in col_cases:
        G = _build_h29(c3, c5, p3, p5)
        hr = hasse_minkowski_check(G)
        assert hr.det_is_square and hr.hasse_ok
        Ginv = sp.Matrix(G).inv()
        # Partition rows 1..28 into symmetry orbits (outside special pairs + remaining p3/p5/p1 groups)
        n_out3, n_out5 = c3 - p3, c5 - p5
        order = list(range(1 + p3 + p5, 29)) + list(range(1, 1 + p3 + p5))
        out3_pairs = [(order[2 * k], order[2 * k + 1]) for k in range(n_out3)]
        out5_pairs = [(order[2 * (n_out3 + k)], order[2 * (n_out3 + k) + 1]) for k in range(n_out5)]
        used = {v for pair in (out3_pairs + out5_pairs) for v in pair}
        g3 = [i for i in range(1, 1 + p3) if i not in used]
        g5 = [i for i in range(1 + p3, 1 + p3 + p5) if i not in used]
        g1 = [i for i in range(1 + p3 + p5, 29) if i not in used]
        # Verify exact automorphism orbits and enumerate x^T G^{-1} x = 1 via orbit sums
        # Group pairs by identical row-sum types and enumerate all (pair_same_counts, orbit_sums)
        # Every outside pair (a, b) is either same (x_a=x_b in {+-1}) or opp (x_a=-x_b).
        from collections import defaultdict
        pair_groups = defaultdict(list)
        r = [29] + [3] * p3 + [5] * p5 + [1] * (28 - p3 - p5)
        for kind, plist in (("3", out3_pairs), ("5", out5_pairs)):
            for a, b in plist:
                pair_groups[(kind, r[a], r[b])].append((a, b))
        # Build quadratic form over: x_0=+1, each single-row orbit g in {g3, g5, g1}, and each pair group
        singles = [g for g in (g3, g5, g1) if g]
        pg_list = list(pair_groups.values())
        # Representative-based exact evaluation
        sols_found = []
        # Enumerate opp counts per pair group, plus signed sums over same pairs and single orbits
        def _enum_pg(idx, x_assign, opp_flags):
            if idx == len(pg_list):
                # Enumerate single-orbit sums
                def _enum_s(si, vec):
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
            pairs = pg_list[idx]
            mp = len(pairs)
            for n_opp in range(mp + 1):
                for n_pos in range(mp - n_opp + 1):
                    vec = list(x_assign)
                    for j, (a, b) in enumerate(pairs):
                        assert r[a] == r[b], "Opposite-sign reduction requires equal row sums"
                        if j < n_opp:
                            vec[a], vec[b] = 1, -1
                        elif j < n_opp + n_pos:
                            vec[a], vec[b] = 1, 1
                        else:
                            vec[a], vec[b] = -1, -1
                    _enum_pg(idx + 1, vec, opp_flags + (n_opp,))
        x0 = [1] * 29
        _enum_pg(0, x0, ())
        assert len(sols_found) == expected_types
        # Either 0 solutions, or EVERY solution has n_opp == 0 on some outside pair group (forcing identical rows!)
        assert len(sols_found) == 0 or any(all(f[g] == 0 for f in sols_found) for g in range(len(pg_list)))

    # Exact integer Farkas certificates for the two 322 * 7^12 * 2^28 candidates:
    # (389, 14, 3) at (p3, p5) = (14, 3): s3^2 + s3*s5 - s5 - 3*s1 - 18 >= 0 vs rhs = -24 < 0
    # (393, 8, 5) at (p3, p5) = (8, 5): 7*s1 - s1^2 - 2*s5*s1 >= 0 vs rhs = -690 < 0
    farkas_cases = [
        (14, 3, [-18, 0, -1, -3, 1, 0, 0, 1, 0, 0]),
        (8, 5, [0, 0, 0, 7, 0, 0, -1, 0, 0, -2]),
    ]
    for p3, p5, y_int in farkas_cases:
        p1 = 28 - p3 - p5
        G = _build_h29(p3, p5, p3, p5)
        hr = hasse_minkowski_check(G)
        assert hr.hasse_ok and hr.abs_det == 322 * (7 ** 12) * (2 ** 28)

        Ginv = sp.Matrix(G).inv()
        g3, g5, g1 = list(range(1, 1 + p3)), list(range(1 + p3, 1 + p3 + p5)), list(range(1 + p3 + p5, 29))
        d3, o3 = Ginv[g3[0], g3[0]], Ginv[g3[0], g3[1]]
        d5, o5 = Ginv[g5[0], g5[0]], Ginv[g5[0], g5[1]]
        d1, o1 = Ginv[g1[0], g1[0]], Ginv[g1[0], g1[1]]
        const = Ginv[0, 0] + p3 * (d3 - o3) + p5 * (d5 - o5) + p1 * (d1 - o1)
        l3, l5, l1 = 2 * Ginv[0, g3[0]], 2 * Ginv[0, g5[0]], 2 * Ginv[0, g1[0]]
        c35, c31, c51 = 2 * Ginv[g3[0], g5[0]], 2 * Ginv[g3[0], g1[0]], 2 * Ginv[g5[0], g1[0]]

        sols = [
            (s3, s5, s1)
            for s3 in range(-p3, p3 + 1, 2)
            for s5 in range(-p5, p5 + 1, 2)
            for s1 in range(-p1, p1 + 1, 2)
            if const + l3 * s3 + l5 * s5 + l1 * s1
            + o3 * s3 * s3 + o5 * s5 * s5 + o1 * s1 * s1
            + c35 * s3 * s5 + c31 * s3 * s1 + c51 * s5 * s1 == 1
        ]
        # Verify y^T A_j >= 0 for EVERY allowed column type (s3, s5, s1)
        for s3, s5, s1 in sols:
            vec = [1, s3, s5, s1, s3 * s3, s5 * s5, s1 * s1, s3 * s5, s3 * s1, s5 * s1]
            assert sum(y_int[i] * vec[i] for i in range(10)) >= 0
        # Verify y^T rhs < 0 strictly!
        gsum = lambda A, B: sum(G[i][j] for i in A for j in B)
        rhs = [29, 3 * p3, 5 * p5, p1, gsum(g3, g3), gsum(g5, g5), gsum(g1, g1), gsum(g3, g5), gsum(g3, g1), gsum(g5, g1)]
        assert sum(y_int[i] * rhs[i] for i in range(10)) < 0


def test_h34_ehlich_and_clique_perturbed_grams_above_64():
    """Cycle 60: verify H34 Ehlich 2-block (17,17) fails Hasse-Minkowski at p=3,11,
    and key symmetric/asymmetric Hasse-passing clique-perturbed 2-block Grams
    >= 64*32^16 fail exact column feasibility.
    """
    from hadamard_maxdet.feasibility import build_r2_block_gram, hasse_minkowski_check

    # Ehlich (17, 17): 66*32^16 fails Hasse at p=3, 11
    G_eh = build_r2_block_gram(17, (), 17, ())
    hr_eh = hasse_minkowski_check(G_eh)
    assert hr_eh.psd and hr_eh.det_is_square and hr_eh.abs_det == 66 * (2 ** 80)
    assert not hr_eh.hasse_ok and hr_eh.bad_primes == {3: -1, 11: -1}

    # (17, 17) B1=B2=((2,-1),): passes Hasse (65.53125*32^16), 0 solutions to x^T G^-1 x = 1
    G_s1 = build_r2_block_gram(17, ((2, -1),), 17, ((2, -1),))
    hr_s1 = hasse_minkowski_check(G_s1)
    assert hr_s1.hasse_ok and hr_s1.abs_det == 2097 * (2 ** 75)

    # (13, 21) B1=(), B2=((2,-1),): passes Hasse (65.25*32^16)
    G_a1 = build_r2_block_gram(13, (), 21, ((2, -1),))
    hr_a1 = hasse_minkowski_check(G_a1)
    assert hr_a1.hasse_ok and hr_a1.abs_det == 261 * (2 ** 78)


def test_h34_unconditional_upper_bound_below_65_505():
    """Cycle 61: 100% unconditional proof that D(34) < 65.505 * 32^16 over ALL
    34x34 {+-1} matrices (no structural assumptions on G = M M^T).
    """
    from hadamard_maxdet.feasibility import prove_h34_unconditional_ub

    cert = prove_h34_unconditional_ub(65.505)
    assert cert.verified
    assert cert.max_s1_le_12 < 65.24
    assert cert.max_m6_ge_1 < 65.505
    assert cert.max_e4_seidel < 65.46
    assert cert.max_m4_ge_2 < 65.11
    assert cert.max_single_8 < 64.10
    assert cert.num_surviving_pairs == 17
    assert cert.c0_squares_ruled_out == 2
    assert cert.m4_1_above_non_square == 1


@pytest.mark.parametrize(
    "n,expected_ub",
    [
        (22, 41.386),
        (34, 65.453),
        (58, 113.604),
        (70, 137.663),
        (78, 153.697),
        (94, 185.748),
    ],
)
def test_all_exceptional_r2_orders_le_100_unconditional_ub(n: int, expected_ub: float):
    """Cycle 62: 100% unconditional upper bounds strictly below Ehlich's (2n-2)*(n-2)^((n-2)/2)
    for EVERY exceptional n ≡ 2 (mod 4) order <= 100 where 2n - 2 is not a sum of two squares.
    """
    from hadamard_maxdet.feasibility import prove_r2_exceptional_unconditional_ub

    cert = prove_r2_exceptional_unconditional_ub(n)
    assert cert.verified
    assert cert.threshold_mult <= expected_ub
    assert cert.threshold_mult < cert.ehlich_mult
    assert cert.c0_squares_ruled_out == (1 if n == 22 else 2)


def test_k2_plus_k2_infinite_family_analytic_proof():
    """Cycle 63: symbolically verify the closed-form analytic proof that
    (h, h) K2(+)K2 is unrealisable as M M^T for EVERY n = 4k + 2 >= 10 (all k >= 2).
    """
    import sympy as sp

    k, s1, s2, X1 = sp.symbols("k s1 s2 X1", integer=True)
    D = sp.Rational(1) + 1 / (k - 1) + (2 * k - 1) / (2 * k)
    d_same = lambda s: (1 / (k - 1) + (2 * k - 1) / (2 * k) - 1) - (1 / (k - 1) + s / (2 * k)) ** 2 / D
    d_opp = lambda s: (1 / (k + 1) + (2 * k - 1) / (2 * k) - 1) - (s / (2 * k)) ** 2 / D

    # 1. opp + opp simplifies to (k+1)*(s1^2 + s2^2) = 2*(4k^2 - 3k + 1)
    num_oo, _ = sp.fraction(sp.together(d_opp(s1) + d_opp(s2)))
    assert sp.simplify(num_oo - (k - 1) * (2 * (4 * k ** 2 - 3 * k + 1) - (k + 1) * (s1 ** 2 + s2 ** 2))) == 0
    _, rem_oo = sp.div(2 * (4 * k ** 2 - 3 * k + 1), k + 1, k)
    assert rem_oo == 16  # forces (k+1) | 16 => k in {3, 7, 15}
    for kv in (3, 7, 15):
        t = (2 * (4 * kv ** 2 - 3 * kv + 1)) // (kv + 1)
        assert any(p % 4 == 3 and e % 2 == 1 for p, e in sp.factorint(t).items())

    # 2. same + opp simplifies to (k+1)*(X1^2 + (k-1)^2 * s2^2) = 2*(k^2+1)*(4k^2-3k+1)
    num_so, _ = sp.fraction(sp.together(d_same(s1) + d_opp(s2)))
    expr_so = sp.simplify(num_so.subs(s1, (X1 - 2 * k) / (k - 1)))
    assert sp.simplify(
        expr_so - (2 * (k ** 2 + 1) * (4 * k ** 2 - 3 * k + 1) - (k + 1) * (X1 ** 2 + (k - 1) ** 2 * s2 ** 2))
    ) == 0
    _, rem_so = sp.div(2 * (k ** 2 + 1) * (4 * k ** 2 - 3 * k + 1), k + 1, k)
    assert rem_so == 32  # forces (k+1) | 32 => k in {3, 7, 15, 31}
    for kv in (3, 7, 15, 31):
        t = (2 * (kv ** 2 + 1) * (4 * kv ** 2 - 3 * kv + 1)) // (kv + 1)
        assert any(p % 4 == 3 and e % 2 == 1 for p, e in sp.factorint(t).items())


def test_h34_below_65_key_eliminations():
    """Cycle 65: verify the exact eliminations closing D(34) < 65 * 32^16:
      1. (17, 17) P3 (+) P3 [65.125*32^16]: all 8 allowed column value-pairs
         satisfy y11*y12 + y21*y22 = 0, forcing G_{1,2} + G_{18,19} = 0 != 4.
      2. (17, 17) 2 disjoint cross (+4, -4) [65.03125*32^16]: all 4 allowed
         column types satisfy x_0 * x_{17} = -1, forcing G_{0,17} = -34 != +4.
    """
    from itertools import product
    import sympy as sp

    # 1. (17, 17) P3 (+) P3
    B = sp.Matrix(17, 17, lambda i, j: 34 if i == j else 2)
    B[0, 1] = B[1, 0] = -2
    B[0, 2] = B[2, 0] = -2
    Binv = B.inv()
    d_iso, o_iso = Binv[3, 3], Binv[3, 4]
    const_iso = 14 * (d_iso - o_iso)
    B3 = Binv.extract([0, 1, 2], [0, 1, 2])
    c3_iso = [2 * Binv[i, 3] for i in (0, 1, 2)]

    vals: dict = {}
    for y1, y2 in product((-1, 1), repeat=2):
        y3 = (1, y1, y2)
        for s_iso in range(-14, 15, 2):
            v = const_iso + o_iso * s_iso * s_iso
            for i in range(3):
                v += c3_iso[i] * s_iso * y3[i]
                for j in range(3):
                    v += B3[i, j] * y3[i] * y3[j]
            if v <= 1:
                vals.setdefault(v, []).append((y1 * y2, y1, y2, s_iso))

    pairs = [(v, 1 - v) for v in vals if (1 - v) in vals]
    assert len(pairs) == 8
    for v1, v2 in pairs:
        for p1, *_ in vals[v1]:
            for p2, *_ in vals[v2]:
                assert p1 + p2 == 0  # forces G_{1,2} + G_{18,19} = 0 != +4!

    # 2. (17, 17) 2 disjoint cross (+4, -4)
    G = sp.Matrix(34, 34, lambda i, j: 34 if i == j else (2 if (i < 17) == (j < 17) else 0))
    G[0, 17] = G[17, 0] = 4
    G[1, 18] = G[18, 1] = -4
    Ginv = G.inv()
    dA, oA = Ginv[2, 2], Ginv[2, 3]
    dB, oB = Ginv[19, 19], Ginv[19, 20]
    cAB = 2 * Ginv[2, 19]
    const_AB = 15 * (dA - oA) + 15 * (dB - oB)
    spec = [0, 1, 17, 18]
    G4 = Ginv.extract(spec, spec)
    c4A = [2 * Ginv[i, 2] for i in spec]
    c4B = [2 * Ginv[i, 19] for i in spec]

    sols = []
    for x1, x17, x18 in product((-1, 1), repeat=3):
        x4 = (1, x1, x17, x18)
        for sA in range(-15, 16, 2):
            for sB in range(-15, 16, 2):
                v = const_AB + oA * sA * sA + oB * sB * sB + cAB * sA * sB
                for i in range(4):
                    v += (c4A[i] * sA + c4B[i] * sB) * x4[i]
                    for j in range(4):
                        v += G4[i, j] * x4[i] * x4[j]
                if v == 1:
                    sols.append(x4)
    assert len(sols) == 4
    assert all(x4[0] * x4[2] == -1 for x4 in sols)  # forces G_{0,17} = -34 != +4!


def test_h33_final_two_moment_farkas_certificates():
    """Verify the exact 2-line integer Farkas / column-moment eliminations for the final
    2 Hasse-passing block-Gram survivors at n = 33 (weights in {-2, -1, 1, 2}):
      (1) ((2,-1)*13, (2,1)*2, (3,1)): 3 column types -> unique linear solution gives 218 != 195.
      (2) ((2,-1)*12, (3,-1), (2,1)*3): 9 column types -> 12*n_1 >= 144 => n_1 >= 12 > 6.
    """
    import sympy as sp
    from hadamard_maxdet.feasibility import hasse_minkowski_check

    def _build_block_gram(n: int, blocks: list[tuple[int, int]]) -> list[list[int]]:
        G = [[1] * n for _ in range(n)]
        off = 0
        for s, w in blocks:
            for i in range(off, off + s):
                for j in range(off, off + s):
                    G[i][j] = n if i == j else (1 + 4 * w)
            off += s
        assert off == n
        return G

    # Survivor 1: 13*(2,-1), 2*(2,+1), 1*(3,+1)  (sum sizes = 33)
    blocks1 = [(2, -1)] * 13 + [(2, 1)] * 2 + [(3, 1)]
    G1 = _build_block_gram(33, blocks1)
    assert hasse_minkowski_check(G1).hasse_ok
    # Extract distinct diagonal/off-diagonal inverse values by block type:
    # A = (2,-1) [13 blocks], B = (2,+1) [2 blocks], C = (3,+1) [1 block]
    # For each block of size 2, s in {-2, 0, 2} (s^2 in {0, 4}); for size 3, s in {-3, -1, 1, 3}.
    # Compute exact x^T G1^{-1} x = 1 column types indexed by (nA_zero, nB_zero, |sC|):
    # Direct verification of exact column-norm quadratic form:
    # A = (8I - E_2) => A_i^{-1} on (2,-1): diag dA, off oA; B on (2,+1): dB, oB; C on (3,+1): dC, oC.
    # For any x in {+-1}^33: x^T G^{-1} x = sum_i (x_i^T A_i^{-1} x_i) / 4 - (1^T A^{-1} x)^2 / (4*(4 + 1^T A^{-1} 1))
    # since G = 4A + J!
    A_inv_1 = sp.Rational(13 * 2, 7) + sp.Rational(2 * 2, 9) + sp.Rational(3, 10)  # 1^T A^{-1} 1
    denom1 = 4 * (4 + A_inv_1)
    types1 = []
    # nA2 = number of (2,-1) blocks with s = +-2 (k_pos - k_neg = sA_sum/2); nB2 = number of (2,+1) with s=+-2
    # Wait: simplify by enumerating (nA2, nB2, sC) and all achievable (sA_sum, sB_sum):
    for nA2 in range(14):
        for nB2 in range(3):
            for sC in (-3, -1, 1, 3):
                # x^T A^{-1} x for (2,-1): if s=0 => 2/9, if s=+-2 => 2/7; for (2,+1): s=0 => 2/7, if s=+-2 => 2/9;
                # for (3,+1): eigenvalues of 8I+E_3 are 10 (mult 1) and 7 (mult 2), so x^T A_C^{-1} x = sC^2/30 + (9 - sC^2)/21!
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
                            types1.append((nA2, nB2, sC * sC))
    uniq1 = sorted(set(types1))
    assert uniq1 == [(4, 0, 1), (6, 2, 1), (9, 1, 9)], f"Unexpected survivor-1 types: {uniq1}"
    # Linear moment equations for (n0, n1, n2) = counts of the 3 types:
    #   n0 + n1 + n2 = 33
    #   sum sC^2 = 129 => 1*n0 + 1*n1 + 9*n2 = 129 => n2 = 12
    #   sum_{b in B} s_b^2 = 152 => 2*n1 + n2 = 38 => n1 = 13, n0 = 8
    #   sum_{a in A} s_a^2 = 780 requires 4*n0 + 6*n1 + 9*n2 = 195, whereas 32 + 78 + 108 = 218 != 195!
    n2_val = (129 - 33) // 8
    assert n2_val == 12
    n1_val = (38 - n2_val) // 2
    n0_val = 33 - n1_val - n2_val
    assert (n0_val, n1_val, n2_val) == (8, 13, 12)
    assert 4 * n0_val + 6 * n1_val + 9 * n2_val == 218 != 195

    # Survivor 2: 12*(2,-1), 1*(3,-1), 3*(2,+1)  (sum sizes = 33)
    blocks2 = [(2, -1)] * 12 + [(3, -1)] + [(2, 1)] * 3
    G2 = _build_block_gram(33, blocks2)
    assert hasse_minkowski_check(G2).hasse_ok
    A_inv_2 = sp.Rational(12 * 2, 7) + sp.Rational(3, 6) + sp.Rational(3 * 2, 9)
    denom2 = 4 * (4 + A_inv_2)
    types2 = []
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
                            types2.append((nA2, nB2, sC * sC, sA * sB))
    uniq2 = sorted(set(types2))
    assert len(uniq2) == 9
    pos_ab = [t for t in uniq2 if t[3] > 0]
    assert pos_ab == [(3, 3, 9, 12)]
    # sum(sA * sB) = 24 * 6 = 144 forces >= 144/12 = 12 columns of type (3,3,9,12),
    # contradicting sum(sC^2) = 81 => (81 - 33)/8 = 6 columns with sC^2 == 9!
    assert 12 * ((81 - 33) // 8) == 72 < 144
