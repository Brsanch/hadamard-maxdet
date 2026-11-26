"""Complete end-to-end verification of D(34) < 65 * 32^16 (full integer step below Ehlich's 66 * 32^16).

Every step of Theorem `thm:d34` in paper/hadamard_maxdet_algebraic_spectral.tex is executed and asserted:
  Step 1 (Coarse exclusions):
    - s1 <= 11: 2-moment spectral bound (< 64.900).
    - m6 >= 3 or |B_ij| >= 10 (delta_T >= 192): 2-moment spectral bound across s1 in {12..17} (< 64.554).
    - m6 = 2 in a single block: BOTH non-isomorphic placements P3 (adjacent) and 2K2 (disjoint)
      via exact KKT capped-spectrum bound (< 64.680 and < 64.654).
    - m6(B1) = 1 AND m6(B2) = 1 simultaneously: exact KKT capped-spectrum bound (< 64.630).
    - m6 = 1 for s in {12, 13, 21, 22}: exact KKT capped-spectrum bound (< 65.0).
  Step 2 (15-dual Cauchy-interlaced Seidel B&B with EXACT switching-equivalence deduplication):
    - Runs across ALL s in {12..22} (m6 = 0) and s in {14..20} (m6 = 1).
    - Uses (rounded spectrum, sorted |S^2|) ONLY as a hash bucket; an intermediate matrix S2 is
      discarded ONLY after exhibiting and verifying an explicit integer switching witness (P, D)
      satisfying P @ (D[:,None] * S1 * D[None,:]) @ P.T == S2 identically (zero false drops).
  Step 3 (Exact C = 0 square-det elimination):
    - Eliminates every C = 0 perfect-square pair >= 65 * 32^16 via Hasse-Minkowski or exact
      column-norm equation x^T G^{-1} x = 1 (all 5 survivors).
  Step 4 (Exact C != 0 elimination across all surviving (B1, B2) pairs >= 65 * 32^16):
    - |C_ij| >= 8 (F >= 64) and ALL k >= 3 (F >= 16k) via exact PSD trace Schur bound (< 65.0).
    - Exhaustive exact k = 1 (|C_ij| = 4): 0 perfect squares >= 65.0.
    - Exhaustive exact k = 2 (|C_ij| = 4): unique perfect square >= 65.0 is (17, 17) empty(+)empty
      with 2 disjoint (+4, -4) at 65.03125 * 32^16, whose 4 allowed column types all force
      x_0 * x_{17} = -1 => G_{0,17} = -34 != +4.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
import sys

import numpy as np
import sympy as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hadamard_maxdet.feasibility import hasse_minkowski_check

TARGET_MULT = 65.0
TARGET_SQ = (65 * (32 ** 16)) ** 2


# ---------------------------------------------------------------------------
# Exact spectral & capped-spectrum helpers (KKT over all r >= 0)
# ---------------------------------------------------------------------------

def _spectral_ub_block(s: int, delta_T: int) -> float:
    n, m = 34, 32
    if s == 0:
        return float(m)
    if s == 1:
        return float(n)
    V = 4 * s * (s - 1) + delta_T
    x = math.sqrt(V / (s * (s - 1)))
    return (((float(n) - x) / m) ** (s - 1)) * (float(n) + (s - 1) * x)


def _seidel_cap_exact_ub(s: int, mu_cap: float, delta_T_S: int) -> float:
    """Exact KKT global max of prod(34 + 2*mu_j)/32^(s-1) subject to sum mu=0,
    sum mu^2 = s(s-1) + delta_T_S, and -17 < mu_j <= mu_cap, over ALL r in {0..s-1}.
    """
    n, m = 34.0, 32.0
    T_S = float(s * (s - 1) + delta_T_S)
    best = 0.0
    for r in range(0, s):
        m_rem = s - r
        S_rem = -r * mu_cap
        T_rem = T_S - r * mu_cap * mu_cap
        if T_rem < -1e-12:
            break
        bar = S_rem / m_rem
        V_rem = T_rem - (S_rem * S_rem) / m_rem
        if V_rem < -1e-9:
            continue
        V_rem = max(0.0, V_rem)
        if m_rem == 1:
            if bar <= mu_cap + 1e-12 and n + 2.0 * bar > 0.0:
                lv = r * math.log(n + 2.0 * mu_cap) + math.log(n + 2.0 * bar) - (s - 1) * math.log(m)
                best = max(best, math.exp(lv))
            continue
        x = math.sqrt(V_rem / (m_rem * (m_rem - 1)))
        a = bar - x
        b = bar + (m_rem - 1) * x
        if b <= mu_cap + 1e-12 and n + 2.0 * a > 0.0:
            lv = (
                r * math.log(n + 2.0 * mu_cap)
                + (m_rem - 1) * math.log(n + 2.0 * a)
                + math.log(n + 2.0 * b)
                - (s - 1) * math.log(m)
            )
            best = max(best, math.exp(lv))
    assert best > 0.0
    return best


def _exact_int_weyl_enclosure(M: np.ndarray):
    """Exact arbitrary-precision integer polar-factor Weyl certification for symmetric integer M.

    Converts LAPACK's float64 eigendecomposition to an exact dyadic pair
    V_hat = V_q / 2^52, w_hat = w_q / 2^45 and computes ||M V_hat - V_hat Lambda_hat||_F
    and ||V_hat^T V_hat - I||_F over exact integers (zero floating-point residual error).
    """
    from fractions import Fraction

    M_int = np.rint(M).astype(np.int64)
    assert np.all(M_int == M), "Exact integer Weyl requires integer matrix M"
    k = M_int.shape[0]
    w_f, V_f = np.linalg.eigh(M_int.astype(np.float64))
    SV, SW = 52, 45
    V_q = np.rint(V_f * (1 << SV)).astype(np.int64).astype(object)
    w_q = np.rint(w_f * (1 << SW)).astype(np.int64).astype(object)
    M_obj = M_int.astype(object)

    def _ceil_sqrt(val: int) -> int:
        r = math.isqrt(val)
        return r if r * r == val else r + 1

    R_int = (M_obj @ V_q) * (1 << SW) - V_q * w_q[None, :]
    rho = Fraction(_ceil_sqrt(int(np.sum(R_int * R_int))), 1 << (SV + SW))
    E_int = (V_q.T @ V_q) - np.eye(k, dtype=object) * (1 << (2 * SV))
    eta = Fraction(_ceil_sqrt(int(np.sum(E_int * E_int))), 1 << (2 * SV))
    assert eta < Fraction(1, 10**8), f"Orthogonality defect too large: {float(eta)}"
    M_fro = _ceil_sqrt(int(np.sum(M_obj * M_obj)))
    lam_max_abs = Fraction(max(abs(int(x)) for x in w_q), 1 << SW)
    weyl = rho + (M_fro + lam_max_abs) * eta
    assert weyl < Fraction(1, 10**7), f"Weyl radius too large: {float(weyl)}"
    return M_obj, V_q, w_q, weyl, SV, SW


def _certified_eigval_enclosure(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Return exact-dyadic eigenvalues w and exact-integer certified Weyl radius delta."""
    _, _, w_q, weyl, _, SW = _exact_int_weyl_enclosure(M)
    w_exact = np.array([int(x) / float(1 << SW) for x in w_q], dtype=np.float64)
    return w_exact, float(weyl) * (1.0 + 1e-14)


def _certified_top2_and_v1_inf(M: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return exact-integer certified bounds (lam1_hi, lam2_lo, lam2_hi, v1_inf_hi, weyl)
    for symmetric integer M via polar-factor Weyl and Davis-Kahan sin(Theta) bounds.
    """
    from fractions import Fraction

    M_obj, V_q, w_q, weyl, _, SW = _exact_int_weyl_enclosure(M)
    l1 = Fraction(int(w_q[-1]), 1 << SW)
    l2 = Fraction(int(w_q[-2]), 1 << SW)
    assert l2 - weyl > 0
    gap_lo = l1 - l2 - 2 * weyl
    assert gap_lo > 1, f"Spectral gap too small for Davis-Kahan: {float(gap_lo)}"

    v1_q = V_q[:, -1]
    n1_sq = int(np.sum(v1_q * v1_q))
    n1_lo = math.isqrt(n1_sq)
    assert n1_lo > 0
    r1_int = (M_obj @ v1_q) * (1 << SW) - int(w_q[-1]) * v1_q
    r1_sq = int(np.sum(r1_int * r1_int))
    r1_hi = math.isqrt(r1_sq)
    if r1_hi * r1_hi < r1_sq:
        r1_hi += 1
    res_v1_hi = Fraction(r1_hi, (1 << SW) * n1_lo)
    sqrt2_hi = Fraction(1414213563, 10**9)
    v1_inf_hat_hi = Fraction(max(abs(int(x)) for x in v1_q), n1_lo)
    v1_inf_hi = min(Fraction(1), v1_inf_hat_hi + sqrt2_hi * res_v1_hi / gap_lo)
    eps = 1e-14
    return (
        float(l1 + weyl) * (1.0 + eps),
        float(l2 - weyl) * (1.0 - eps),
        float(l2 + weyl) * (1.0 + eps),
        min(1.0, float(v1_inf_hi) + eps),
        float(weyl),
    )


def _m6_ub(s: int, edges_6: tuple[tuple[int, int], ...]) -> float:
    """Exact capped-spectrum UB for an s x s block with |edges_6| off-diagonal +-6 pairs,
    using certified polar-factor Weyl upper enclosure of lambda_max(S).
    """
    S = np.ones((s, s), dtype=np.float64) - np.eye(s)
    for u, v in edges_6:
        S[u, v] = S[v, u] = 3.0
    w, weyl = _certified_eigval_enclosure(S)
    mu_cap = float(w[-1]) + weyl
    return _seidel_cap_exact_ub(s, mu_cap, delta_T_S=16 * len(edges_6))


def verify_step1_coarse_exclusions() -> dict[str, float]:
    """Step 1: verify all coarse exclusions strictly below 65 * 32^16."""
    max_s1_le_11 = max(
        math.sqrt(_spectral_ub_block(s1, 0) * _spectral_ub_block(34 - s1, 0))
        for s1 in range(0, 12)
    )
    assert max_s1_le_11 < TARGET_MULT

    # m6 >= 3 (delta_T >= 3 * 64 = 192) or any |B_ij| >= 10 (delta_T >= 2*(100-4) = 192)
    max_m6_ge_3 = max(
        max(
            math.sqrt(_spectral_ub_block(s1, 192) * _spectral_ub_block(34 - s1, 0)),
            math.sqrt(_spectral_ub_block(s1, 0) * _spectral_ub_block(34 - s1, 192)),
        )
        for s1 in range(12, 18)
    )
    assert max_m6_ge_3 < TARGET_MULT

    # m6 = 2 in a single block: BOTH non-isomorphic placements (P3 adjacent, 2K2 disjoint)
    max_m6_2_p3 = max(
        max(
            math.sqrt(_m6_ub(s1, ((0, 1), (0, 2))) * (32.0 + 2.0 * (34 - s1))),
            math.sqrt((32.0 + 2.0 * s1) * _m6_ub(34 - s1, ((0, 1), (0, 2)))),
        )
        for s1 in range(12, 18)
    )
    max_m6_2_2k2 = max(
        max(
            math.sqrt(_m6_ub(s1, ((0, 1), (2, 3))) * (32.0 + 2.0 * (34 - s1))),
            math.sqrt((32.0 + 2.0 * s1) * _m6_ub(34 - s1, ((0, 1), (2, 3)))),
        )
        for s1 in range(12, 18)
    )
    assert max_m6_2_p3 < TARGET_MULT and max_m6_2_2k2 < TARGET_MULT

    # m6(B1) = 1 AND m6(B2) = 1 simultaneously
    max_m6_1_1 = max(
        math.sqrt(_m6_ub(s1, ((0, 1),)) * _m6_ub(34 - s1, ((0, 1),)))
        for s1 in range(12, 18)
    )
    assert max_m6_1_1 < TARGET_MULT

    # m6 = 1 in one block for s1 in {12, 13} (s2 in {22, 21})
    max_m6_1_edge_s = max(
        max(
            math.sqrt(_m6_ub(s1, ((0, 1),)) * (32.0 + 2.0 * (34 - s1))),
            math.sqrt((32.0 + 2.0 * s1) * _m6_ub(34 - s1, ((0, 1),))),
        )
        for s1 in (12, 13)
    )
    assert max_m6_1_edge_s < TARGET_MULT

    stats = {
        "max_s1_le_11": max_s1_le_11,
        "max_m6_ge_3": max_m6_ge_3,
        "max_m6_2_p3": max_m6_2_p3,
        "max_m6_2_2k2": max_m6_2_2k2,
        "max_m6_1_1": max_m6_1_1,
        "max_m6_1_edge_s": max_m6_1_edge_s,
    }
    print("Step 1 (coarse exclusions) PASS:", {k: round(v, 4) for k, v in stats.items()}, flush=True)
    return stats


# ---------------------------------------------------------------------------
# Exact zero-false-positive Seidel switching equivalence tester
# ---------------------------------------------------------------------------

def _switch_to_vertex(S: np.ndarray, v0: int) -> tuple[np.ndarray, np.ndarray]:
    """Switch S (int array, zero diagonal, nonzero off-diagonals) so row v0 is strictly positive."""
    d = np.where(S[v0] < 0, -1, 1).astype(np.int8)
    d[v0] = 1
    S_sw = (d[:, None] * S * d[None, :]).astype(np.int8)
    return S_sw, d


def _graph_iso_exact(A: np.ndarray, B: np.ndarray) -> list[int] | None:
    """Exact isomorphism check between two m x m symmetric int8 matrices A, B.
    Returns permutation perm such that A[i, j] == B[perm[i], perm[j]] for all i, j, or None.
    """
    m = A.shape[0]
    # Compute exact 1-WL colors using raw value multisets (shared label space across A and B)
    colA = [tuple(sorted(A[i].tolist())) for i in range(m)]
    colB = [tuple(sorted(B[i].tolist())) for i in range(m)]
    if sorted(colA) != sorted(colB):
        return None
    for _ in range(3):
        nA = [(colA[i], tuple(sorted((int(A[i, j]), colA[j]) for j in range(m) if j != i))) for i in range(m)]
        nB = [(colB[i], tuple(sorted((int(B[i, j]), colB[j]) for j in range(m) if j != i))) for i in range(m)]
        if sorted(nA) != sorted(nB):
            return None
        uniq = {c: idx for idx, c in enumerate(sorted(set(nA)))}
        colA = [uniq[c] for c in nA]
        colB = [uniq[c] for c in nB]
        if len(uniq) == m:
            break

    # Order vertices of A by smallest color class size first
    b_by_col: dict[int, list[int]] = defaultdict(list)
    for j, c in enumerate(colB):
        b_by_col[c].append(j)
    order = sorted(range(m), key=lambda i: (len(b_by_col[colA[i]]), i))

    perm = [-1] * m
    used = [False] * m

    def _bt(pos: int) -> bool:
        if pos == m:
            return True
        u = order[pos]
        for v in b_by_col[colA[u]]:
            if used[v]:
                continue
            ok = True
            for prev_pos in range(pos):
                pu = order[prev_pos]
                pv = perm[pu]
                if A[u, pu] != B[v, pv]:
                    ok = False
                    break
            if ok:
                perm[u] = v
                used[v] = True
                if _bt(pos + 1):
                    return True
                used[v] = False
                perm[u] = -1
        return False

    return perm if _bt(0) else None


def are_switching_equivalent_verified(S1: np.ndarray, S2: np.ndarray) -> bool:
    """Return True ONLY IF we construct and verify an explicit (P, D) witness such that
    S1_switched_permuted == S2 identically in integer arithmetic. Zero false positives possible.
    """
    k = S1.shape[0]
    S1_sq = np.abs(S1.astype(np.int32) @ S1.astype(np.int32))
    S2_sq = np.abs(S2.astype(np.int32) @ S2.astype(np.int32))
    sig1 = [tuple(sorted(S1_sq[i].tolist())) for i in range(k)]
    sig2 = [tuple(sorted(S2_sq[i].tolist())) for i in range(k)]
    if sorted(sig1) != sorted(sig2):
        return False

    # Also match exact |S| row multisets (distinguishes m6=1 endpoints immediately)
    abs1 = [tuple(sorted(np.abs(S1[i]).tolist())) for i in range(k)]
    abs2 = [tuple(sorted(np.abs(S2[i]).tolist())) for i in range(k)]
    if sorted(abs1) != sorted(abs2):
        return False

    # Fix anchor vertex 0 in S1; try matching with each compatible vertex v0 in S2
    S1_sw, d1 = _switch_to_vertex(S1, 0)
    idx1 = list(range(1, k))
    A = S1_sw[np.ix_(idx1, idx1)]
    # Check row-0 compatibility (for m6=1, entry 3 must map to entry 3)
    row0_1 = S1_sw[0, idx1]

    for v0 in range(k):
        if sig2[v0] != sig1[0] or abs2[v0] != abs1[0]:
            continue
        S2_sw, d2 = _switch_to_vertex(S2, v0)
        idx2 = [j for j in range(k) if j != v0]
        B = S2_sw[np.ix_(idx2, idx2)]
        row0_2 = S2_sw[v0, idx2]
        # Augment diagonal temporarily with row0 entry (since diagonal is 0) so iso preserves row0!
        A_aug = A.copy()
        B_aug = B.copy()
        np.fill_diagonal(A_aug, row0_1)
        np.fill_diagonal(B_aug, row0_2)
        perm_sub = _graph_iso_exact(A_aug, B_aug)
        if perm_sub is not None:
            # Build full permutation P: i in S1 -> full_perm[i] in S2
            full_perm = [0] * k
            full_perm[0] = v0
            for a_pos, b_pos in enumerate(perm_sub):
                full_perm[idx1[a_pos]] = idx2[b_pos]
            # Verify explicit switching witness:
            # S1_sw[i, j] == S2_sw[full_perm[i], full_perm[j]]
            # i.e. d1[i]*S1[i,j]*d1[j] == d2[p[i]]*S2[p[i],p[j]]*d2[p[j]]
            # => D_eff[i] = d1[i]*d2[p[i]] satisfies D_eff[i]*S1[i,j]*D_eff[j] == S2[p[i],p[j]]
            p_arr = np.array(full_perm, dtype=int)
            d_eff = (d1 * d2[p_arr]).astype(np.int32)
            assert sorted(full_perm) == list(range(k)), "Permutation check failed!"
            assert np.all(np.abs(d_eff) == 1), "Switching signs check failed!"
            lhs = d_eff[:, None] * S1.astype(np.int32) * d_eff[None, :]
            rhs = S2.astype(np.int32)[np.ix_(p_arr, p_arr)]
            assert np.array_equal(lhs, rhs), "Internal witness verification failed!"
            return True
    return False


def run_seidel_bb(s: int, thresh_B: float, m6: int = 0) -> list[tuple[float, int, int, int, list[list[int]]]]:
    """15-dual Cauchy-interlaced Seidel branch-and-bound with exact verified switching deduplication."""
    S_tr = 34.0 * s
    T_tr = 34.0 ** 2 * s + 4.0 * s * (s - 1) + 64.0 * m6
    base_shift = (s - 1) * math.log(32.0)
    gmax = 32.0 + 2.0 * s + (4.0 if m6 == 1 else 0.0)
    dual_pairs = [
        (32.0, gmax), (32.0, gmax - 4), (32.0, gmax - 8), (32.0, gmax - 12), (32.0, gmax - 16),
        (31.0, gmax - 2), (31.0, gmax - 8), (31.0, gmax - 14),
        (30.0, gmax - 4), (30.0, gmax - 10), (30.0, gmax - 16),
        (29.0, gmax - 8), (28.0, gmax - 12), (33.0, gmax), (33.0, gmax - 6),
    ]

    def batch_ub(betas: np.ndarray, weyl_rad: np.ndarray) -> np.ndarray:
        """Evaluate 15-dual Cauchy-interlaced UB on Weyl-enlarged certified intervals
        [betas - weyl_rad, betas + weyl_rad], guaranteeing every exact eigenvalue is enclosed.
        """
        N, k = betas.shape
        lo = np.full((N, s), 1e-6)
        hi = np.full((N, s), gmax)
        hi[:, :k] = np.minimum(hi[:, :k], betas + weyl_rad[:, None])
        lo[:, s - k:] = np.maximum(lo[:, s - k:], betas - weyl_rad[:, None])
        non_pd = np.any(hi <= 1e-12, axis=1)
        hi_safe = np.maximum(hi, 1e-12)
        best = np.full(N, 1e9)
        for x1, x2 in dual_pairs:
            if x2 <= x1 + 1.0:
                continue
            alpha = -1.0 / (2.0 * x1 * x2)
            gamma = 1.0 / x1 + 1.0 / x2
            gl = np.log(lo) - alpha * lo * lo - gamma * lo
            gu = np.log(hi_safe) - alpha * hi_safe * hi_safe - gamma * hi_safe
            gx1 = math.log(x1) - alpha * x1 * x1 - gamma * x1
            val = np.maximum(gl, gu)
            val = np.where((lo < x1) & (x1 < hi_safe), np.maximum(val, gx1), val)
            ub = np.exp(alpha * T_tr + gamma * S_tr - base_shift + val.sum(axis=1))
            best = np.minimum(best, ub)
        return np.where(non_pd, 0.0, best)

    cur = [np.array([[34.0, 6.0 if m6 else 2.0], [6.0 if m6 else 2.0, 34.0]])]
    total_collisions_rescued = 0
    max_weyl_seen = 0.0
    for k in range(3, s + 1):
        nxt = []
        buckets: dict[tuple, list[np.ndarray]] = defaultdict(list)
        n_masks = 1 << (k - 2)
        chunk = 16384
        iu = np.triu_indices(k, 1)
        eye_k = np.eye(k, dtype=np.float64)
        for Bk in cur:
            for start in range(0, n_masks, chunk):
                end = min(n_masks, start + chunk)
                masks = np.arange(start, end, dtype=np.int32)
                bits = ((masks[:, None] >> np.arange(k - 2)) & 1)
                V = np.empty((len(masks), k - 1), dtype=np.float64)
                V[:, 0] = 2.0
                V[:, 1:] = np.where(bits, 2.0, -2.0)
                Bbatch = np.empty((len(masks), k, k), dtype=np.float64)
                Bbatch[:, :k - 1, :k - 1] = Bk
                Bbatch[:, k - 1, :k - 1] = V
                Bbatch[:, :k - 1, k - 1] = V
                Bbatch[:, k - 1, k - 1] = 34.0
                betas, Ubatch = np.linalg.eigh(Bbatch)
                # Vectorized polar-factor Weyl certificate with Higham (2002, §3.5) IEEE 754 backward error:
                u_mach = 2.0 ** -53
                gamma_kp1 = ((k + 1) * u_mach) / (1.0 - (k + 1) * u_mach)
                norm_fac = 1.0 + 2.0 * (k * k + 1) * u_mach
                R = Bbatch @ Ubatch - Ubatch * betas[:, None, :]
                E = np.swapaxes(Ubatch, 1, 2) @ Ubatch - eye_k
                B_fro = np.linalg.norm(Bbatch, axis=(1, 2))
                U_fro = np.linalg.norm(Ubatch, axis=(1, 2))
                w_abs = np.max(np.abs(betas), axis=1)
                rho = norm_fac * (np.linalg.norm(R, axis=(1, 2)) + gamma_kp1 * B_fro * U_fro + 2.0 * u_mach * U_fro * w_abs)
                eta = norm_fac * (np.linalg.norm(E, axis=(1, 2)) + gamma_kp1 * U_fro * U_fro + 2.0 * u_mach * math.sqrt(k))
                assert float(np.max(eta)) < 1e-8, f"Orthogonality defect too large: {np.max(eta)}"
                weyl = (1.0 + 4.0 * u_mach) * (rho + (B_fro + w_abs) * eta)
                wmax = float(np.max(weyl))
                assert wmax < 1e-7, f"Weyl radius too large: {wmax}"
                if wmax > max_weyl_seen:
                    max_weyl_seen = wmax
                ubs = batch_ub(betas, weyl)
                keep = np.where(ubs >= thresh_B - 1e-6)[0]
                for idx in keep:
                    M = Bbatch[idx]
                    S_int = np.round((M - 34.0 * np.eye(k)) / 2.0).astype(np.int8)
                    S2 = np.abs(S_int.astype(np.int32) @ S_int.astype(np.int32))
                    sp_key = tuple(np.round(betas[idx], 5))
                    key = (sp_key, tuple(sorted(S2[iu].tolist())))
                    bucket = buckets[key]
                    if not any(are_switching_equivalent_verified(rep, S_int) for rep in bucket):
                        if bucket:
                            total_collisions_rescued += 1
                        bucket.append(S_int)
                        nxt.append(M)
        cur = nxt
        if not cur:
            break

    out = []
    for M in cur:
        Mi = M.astype(int).tolist()
        d = int(sp.Matrix(Mi).det())
        mult = d / (32 ** (s - 1))
        if mult >= thresh_B - 1e-9:
            sqf = 1
            for p, e in sp.factorint(d).items():
                if e % 2 == 1:
                    sqf *= p
            out.append((mult, d, sqf, m6, Mi))
    out.sort(key=lambda x: -x[0])
    if total_collisions_rescued:
        print(f"    [exact iso rescued {total_collisions_rescued} non-switching-equivalent hash collisions at s={s}, m6={m6}; max_weyl={max_weyl_seen:.3e}]", flush=True)
    else:
        print(f"    [s={s}, m6={m6}: certified max_weyl={max_weyl_seen:.3e}]", flush=True)
    return out


# ---------------------------------------------------------------------------
# Step 3: Exact column-norm verification for every Hasse-passing C = 0 square
# ---------------------------------------------------------------------------

def _block_aut_orbits(M: list[list[int]]) -> list[list[int]]:
    """Partition indices of a symmetric block M into identical-row symmetry orbits
    (vertices with identical off-diagonal row profiles after switching to max +2).
    """
    s = len(M)
    S = np.array([[0 if i == j else (M[i][j] // 2) for j in range(s)] for i in range(s)], dtype=int)
    # Switch S so first vertex has all +1 (or keep raw M if already sparse in -2)
    neg_cnt = int(np.sum(S < 0))
    best_S = S
    for v in range(s):
        d = np.where(S[v] < 0, -1, 1)
        d[v] = 1
        cand = d[:, None] * S * d[None, :]
        if int(np.sum(cand < 0)) < neg_cnt:
            neg_cnt = int(np.sum(cand < 0))
            best_S = cand
    # Passive vertices have all off-diagonals == +1 in best_S
    passive = [i for i in range(s) if all(best_S[i, j] == 1 for j in range(s) if j != i)]
    active = [i for i in range(s) if i not in passive]
    return active, passive, (34 * np.eye(s, dtype=int) + 2 * best_S).tolist()


def verify_c0_block_pair_column_infeasible(M1: list[list[int]], M2: list[list[int]]) -> str:
    """Rigorously prove G = M1 (+) M2 has no realization M M^T = G via exact x^T G^{-1} x = 1."""
    act1, pas1, B1_list = _block_aut_orbits(M1)
    act2, pas2, B2_list = _block_aut_orbits(M2)

    def _block_vals(act: list[int], pas: list[int], B_list: list[list[int]]):
        Binv = sp.Matrix(B_list).inv()
        k = len(act)
        c = len(pas)
        B_act = Binv.extract(act, act) if k else sp.Matrix(0, 0, [])
        d_p = Binv[pas[0], pas[0]] if c >= 1 else sp.Integer(0)
        o_p = Binv[pas[0], pas[1]] if c >= 2 else sp.Integer(0)
        c_ap = [2 * Binv[a, pas[0]] for a in act] if c >= 1 else []
        const_p = c * (d_p - o_p)
        vals = defaultdict(list)
        for y in product((-1, 1), repeat=k):
            q_a = sum(y[i] * y[j] * B_act[i, j] for i in range(k) for j in range(k))
            lin_a = sum(y[i] * c_ap[i] for i in range(k))
            for sc in range(-c, c + 1, 2):
                val = q_a + const_p + o_p * sc * sc + lin_a * sc
                vals[val].append((y, sc))
        return vals, B_list

    v1, B1_sw = _block_vals(act1, pas1, B1_list)
    v2, B2_sw = _block_vals(act2, pas2, B2_list)

    sol_pairs = []
    for val1, lst1 in v1.items():
        val2 = sp.Integer(1) - val1
        if val2 in v2:
            for y1, sc1 in lst1:
                for y2, sc2 in v2[val2]:
                    sol_pairs.append((y1, sc1, y2, sc2))

    if not sol_pairs:
        return "0 solutions to x^T G^{-1} x = 1"

    # Check forced identical/opposite active pair inside B1 or B2
    for idx_b, (act, B_sw, pos_y) in enumerate(((act1, B1_sw, 0), (act2, B2_sw, 2))):
        k = len(act)
        for i in range(k):
            for j in range(i + 1, k):
                prods = {sp_t[pos_y][i] * sp_t[pos_y][j] for sp_t in sol_pairs}
                if len(prods) == 1:
                    val_p = next(iter(prods))
                    req = B_sw[act[i]][act[j]]
                    assert 34 * val_p != req
                    return f"forced y[{i}]*y[{j}]={val_p:+d} in block {idx_b+1} => G={34*val_p} != {req}"

    # Check linear obstruction y1[i]*y1[j] + y2[a]*y2[b] == 0 (e.g. P3 (+) P3)
    if len(act1) >= 2 and len(act2) >= 2:
        for i in range(len(act1)):
            for j in range(i + 1, len(act1)):
                for a in range(len(act2)):
                    for b in range(a + 1, len(act2)):
                        sums = {sp_t[0][i] * sp_t[0][j] + sp_t[2][a] * sp_t[2][b] for sp_t in sol_pairs}
                        if sums == {0}:
                            req = B1_sw[act1[i]][act1[j]] + B2_sw[act2[a]][act2[b]]
                            if req != 0:
                                return f"all {len(sol_pairs)} col types satisfy y1[{i}]y1[{j}]+y2[{a}]y2[{b}]=0 => sum G=0 != {req}"

    raise AssertionError(f"Could not eliminate Hasse-passing pair ({len(sol_pairs)} col types)")


# ---------------------------------------------------------------------------
# Step 4: Exhaustive C != 0 elimination for all surviving pairs >= 65 * 32^16
# ---------------------------------------------------------------------------

def verify_step4_cross_block(blocks: dict[int, list]) -> dict[str, float | int]:
    """Verify ALL C != 0 cases across every surviving (B1, B2) pair with sqrt(d1*d2) >= 65 * 32^16."""
    max_single_8 = 0.0
    max_k_ge_3 = 0.0
    k1_orbits_checked = 0
    k2_squares: list[tuple[float, int, int, str]] = []

    for s1 in range(12, 18):
        s2 = 34 - s1
        for i, (m1, d1, sq1, f1, M1) in enumerate(blocks[s1]):
            for j, (m2, d2, sq2, f2, M2) in enumerate(blocks[s2]):
                if s1 == s2 and j < i:
                    continue
                if f1 + f2 > 1:
                    continue
                prod_d = d1 * d2
                if prod_d < TARGET_SQ:
                    continue
                mult = math.exp(0.5 * math.log(prod_d) - 16.0 * math.log(32.0))

                # 4a. Certified Weyl + Davis-Kahan PSD trace Schur bound for:
                #     k = 1, |C_ij| >= 8 (F >= 64); k = 2, |C_ij| >= 8 (F >= 80); ALL k >= 3 (F >= 16k)
                B1f = np.array(M1, dtype=np.float64)
                B2f = np.array(M2, dtype=np.float64)
                lam1_hi, lam2_lo, lam2_hi, v1m_hi, _ = _certified_top2_and_v1_inf(B1f)
                mu1_hi, mu2_lo, mu2_hi, v2m_hi, _ = _certified_top2_and_v1_inf(B2f)

                def _ub_kF(k: int, F: float) -> float:
                    a1 = min(1.0, k * v1m_hi * v1m_hi)
                    a2 = min(1.0, k * v2m_hi * v2m_hi)
                    c0 = 1.0 - a1 - a2
                    tr_mode = F * (
                        (c0 / (lam2_hi * mu2_hi) if c0 >= 0.0 else c0 / (lam2_lo * mu2_lo))
                        + a1 / (lam1_hi * mu2_hi)
                        + a2 / (lam2_hi * mu1_hi)
                    )
                    tr_crude = F / (lam1_hi * mu1_hi)
                    return mult * math.exp(-0.5 * max(tr_mode, tr_crude))

                # Explicitly cover BOTH (k=1, |C_ij|>=8 => F>=64) AND (k=2, |C_ij|>=8 => F>=80)
                max_single_8 = max(max_single_8, _ub_kF(1, 64.0), _ub_kF(2, 80.0))
                k_cut = max(3, int(math.ceil(2.0 * math.log(mult / TARGET_MULT) * (lam1_hi * mu1_hi) / 16.0)) + 2)
                for k_val in range(3, k_cut + 1):
                    max_k_ge_3 = max(max_k_ge_3, _ub_kF(k_val, 16.0 * k_val))

                # 4b. Exhaustive exact k = 1 (|C_ij| = 4)
                B1inv = sp.Matrix(M1).inv()
                B2inv = sp.Matrix(M2).inv()
                diag1 = sorted({B1inv[u, u] for u in range(s1)})
                diag2 = sorted({B2inv[v, v] for v in range(s2)})
                for d1u in diag1:
                    for d2v in diag2:
                        k1_orbits_checked += 1
                        detG_rat = sp.Integer(prod_d) * (1 - 16 * d1u * d2v)
                        assert detG_rat.is_integer
                        detG = int(detG_rat)
                        if detG >= TARGET_SQ:
                            r = math.isqrt(detG)
                            assert r * r != detG, f"Unexpected k=1 square at ({s1},{s2})!"

                # 4c. Exhaustive exact k = 2 (|C_ij| = 4, two nonzero entries)
                # Fast certified PSD trace filter first: if even k=2 PSD trace bound < 65.0, no k=2 matrix can reach 65.0
                if _ub_kF(2, 32.0) < TARGET_MULT - 1e-6:
                    continue

                # Exact enumeration of ALL 2x2 Schur complements over ordered signed inverse-entry tuples
                # (preserving BOTH orderings (u1,u2) and (u2,u1) when diagonals differ, and exact sign of Binv[u1,u2])
                p1_types: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
                for u1 in range(s1):
                    p1_types[("same", B1inv[u1, u1], B1inv[u1, u1], B1inv[u1, u1])].append((u1, u1))
                    for u2 in range(s1):
                        if u1 != u2:
                            p1_types[("diff", B1inv[u1, u1], B1inv[u2, u2], B1inv[u1, u2])].append((u1, u2))
                p2_types: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
                for v1 in range(s2):
                    p2_types[("same", B2inv[v1, v1], B2inv[v1, v1], B2inv[v1, v1])].append((v1, v1))
                    for v2 in range(s2):
                        if v1 != v2:
                            p2_types[("diff", B2inv[v1, v1], B2inv[v2, v2], B2inv[v1, v2])].append((v1, v2))

                for t1_key, pairs1 in p1_types.items():
                    u1, u2 = pairs1[0]
                    for t2_key, pairs2 in p2_types.items():
                        v1, v2 = pairs2[0]
                        if t1_key[0] == "same" and t2_key[0] == "same":
                            continue  # single entry, not k=2
                        # Test BOTH relative signs sgn in (+1, -1) for ALL k=2 configurations (including same-row/col!)
                        for sgn in (1, -1):
                            P = sp.Matrix([[B1inv[u1, u1], B1inv[u1, u2]], [B1inv[u2, u1], B1inv[u2, u2]]])
                            Q = sp.Matrix([[B2inv[v1, v1], sgn * B2inv[v1, v2]], [sgn * B2inv[v2, v1], B2inv[v2, v2]]])
                            schur = (sp.eye(2) - 16 * P * Q).det()
                            detG_rat = sp.Integer(prod_d) * schur
                            assert detG_rat.is_integer
                            detG = int(detG_rat)
                            if detG >= TARGET_SQ:
                                r = math.isqrt(detG)
                                if r * r == detG:
                                    # Build full 34x34 G and verify elimination (and assert block symmetry if bucket > 1)
                                    G = [[0] * 34 for _ in range(34)]
                                    for a in range(s1):
                                        for b in range(s1):
                                            G[a][b] = M1[a][b]
                                    for a in range(s2):
                                        for b in range(s2):
                                            G[s1 + a][s1 + b] = M2[a][b]
                                    G[u1][s1 + v1] = G[s1 + v1][u1] = 4
                                    G[u2][s1 + v2] = G[s1 + v2][u2] = 4 * sgn
                                    hr = hasse_minkowski_check(G)
                                    if not hr.hasse_ok:
                                        reason = f"Hasse fails {hr.bad_primes}"
                                    else:
                                        # Verify exact column obstruction (unique survivor: (17,17) empty+empty 2 disjoint (+4,-4);
                                        # both blocks are empty => full S_17 x S_17 automorphism group, so all bucket reps are isomorphic!)
                                        assert s1 == 17 and s2 == 17 and r == 2081 * (2 ** 75)
                                        assert all(M1[a][b] == (34 if a == b else 2) for a in range(17) for b in range(17))
                                        assert all(M2[a][b] == (34 if a == b else 2) for a in range(17) for b in range(17))
                                        Ginv = sp.Matrix(G).inv()
                                        rem1 = [idx for idx in range(17) if idx not in (u1, u2)]
                                        rem2 = [17 + idx for idx in range(17) if idx not in (v1, v2)]
                                        d1r, o1r = Ginv[rem1[0], rem1[0]], Ginv[rem1[0], rem1[1]]
                                        d2r, o2r = Ginv[rem2[0], rem2[0]], Ginv[rem2[0], rem2[1]]
                                        c_r = 15 * (d1r - o1r) + 15 * (d2r - o2r)
                                        act4 = [u1, u2, s1 + v1, s1 + v2]
                                        G4 = Ginv.extract(act4, act4)
                                        c4_1 = [2 * Ginv[idx, rem1[0]] for idx in act4]
                                        c4_2 = [2 * Ginv[idx, rem2[0]] for idx in act4]
                                        c_12 = 2 * Ginv[rem1[0], rem2[0]]
                                        sols = []
                                        for y in product((-1, 1), repeat=4):
                                            if y[0] != 1:
                                                continue
                                            q4 = sum(y[a] * y[b] * G4[a, b] for a in range(4) for b in range(4))
                                            l1 = sum(y[a] * c4_1[a] for a in range(4))
                                            l2 = sum(y[a] * c4_2[a] for a in range(4))
                                            for sr1 in range(-15, 16, 2):
                                                for sr2 in range(-15, 16, 2):
                                                    val = q4 + c_r + o1r * sr1 * sr1 + o2r * sr2 * sr2 + c_12 * sr1 * sr2 + l1 * sr1 + l2 * sr2
                                                    if val == 1:
                                                        sols.append((y, sr1, sr2))
                                        assert len(sols) > 0 and all(y[0] * y[2] == -1 for y, _, _ in sols)
                                        reason = f"col eq ({len(sols)} types): all force x_u1*x_v1=-1 => G=-34 != +4"
                                    k2_squares.append((r / (32 ** 16), s1, s2, reason))

    assert max_single_8 < TARGET_MULT
    assert max_k_ge_3 < TARGET_MULT
    print(
        f"Step 4 (C != 0) PASS: max_|C|>=8={max_single_8:.4f}, max_k>=3={max_k_ge_3:.4f}, "
        f"k=1 orbits={k1_orbits_checked} (0 squares >= 65), k=2 squares >= 65 eliminated={len(k2_squares)}",
        flush=True,
    )
    for x in k2_squares:
        print(f"    k=2 square |det|={x[0]:.5f}*32^16 ({x[1]},{x[2]}): {x[3]}", flush=True)
    return {
        "max_single_8": max_single_8,
        "max_k_ge_3": max_k_ge_3,
        "k1_orbits_checked": k1_orbits_checked,
        "k2_squares_eliminated": len(k2_squares),
    }


def main() -> None:
    t0 = time.monotonic()
    verify_step1_coarse_exclusions()

    blocks: dict[int, list] = {}
    for s in range(12, 23):
        other = 34 - s
        thresh = (TARGET_MULT ** 2) / (32.0 + 2.0 * other)
        b0 = run_seidel_bb(s, thresh, m6=0)
        b1 = run_seidel_bb(s, thresh, m6=1) if 14 <= s <= 20 else []
        blocks[s] = b0 + b1
        print(f"s={s:2d} (thresh={thresh:6.3f}, {time.monotonic() - t0:5.1f}s): m6=0 -> {len(b0):2d}, m6=1 -> {len(b1):2d}", flush=True)

    c0_squares = []
    for s1 in range(12, 18):
        s2 = 34 - s1
        for i, (m1, d1, sq1, f1, M1) in enumerate(blocks[s1]):
            for j, (m2, d2, sq2, f2, M2) in enumerate(blocks[s2]):
                if s1 == s2 and j < i:
                    continue
                if f1 + f2 > 1 or sq1 != sq2:
                    continue
                if d1 * d2 >= TARGET_SQ:
                    r = math.isqrt(d1 * d2)
                    assert r * r == d1 * d2
                    G = [[0] * 34 for _ in range(34)]
                    for a in range(s1):
                        for b in range(s1):
                            G[a][b] = M1[a][b]
                    for a in range(s2):
                        for b in range(s2):
                            G[s1 + a][s1 + b] = M2[a][b]
                    hr = hasse_minkowski_check(G)
                    if not hr.hasse_ok:
                        elim = f"Hasse-Minkowski fails {hr.bad_primes}"
                    else:
                        elim = verify_c0_block_pair_column_infeasible(M1, M2)
                    c0_squares.append((r / (32 ** 16), s1, f1, m1, s2, f2, m2, elim))

    c0_squares.sort(reverse=True)
    print(f"\nStep 3 (C = 0) PASS — ALL {len(c0_squares)} square-det pairs >= 65.0 * 32^16 eliminated:")
    for x in c0_squares:
        print(f"  |det|={x[0]:7.4f}*32^16  ({x[1]:2d}[m6={x[2]},{x[3]:.3f}], {x[4]:2d}[m6={x[5]},{x[6]:.3f}])  -> {x[7]}")

    verify_step4_cross_block(blocks)
    print(f"\nALL STEPS VERIFIED in {time.monotonic() - t0:.1f}s: D(34) < 65 * 32^16.", flush=True)


if __name__ == "__main__":
    main()
