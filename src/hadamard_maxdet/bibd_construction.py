"""Algebraic-design construction at H23: the (23, 11, 5)-Paley biplane.

A symmetric (v, k, lambda)-design is a v-element point set with v blocks
of size k each, such that every pair of points lies in exactly lambda
blocks. The (v, k, lambda) = (23, 11, 5) design is the unique such object,
also called the **Paley biplane**: blocks are the cyclic translates of the
quadratic-residue set Q ⊂ Z/23.

Convert the {0, 1}-incidence matrix to ±1 (0 -> -1, 1 -> +1). The resulting
M satisfies:

  - row weight (sum of +1 entries per row) = k = 11; row sum = 2k - v = -1.
  - inner product <v_i, v_j> = 4*|B_i ∩ B_j| - 2*(|B_i|+|B_j|) + v
                              = 4*lambda - 4k + v
                              = 4*5 - 44 + 23 = -1
    for every i != j.
  - Gram G = M Mᵀ = v*I + (-1)*(J - I) = (v+1)*I - J = 24*I - J
    eigenvalues: 1 (mult 1) and 24 (mult 22).
  - |det M| = sqrt(|det G|) = sqrt(24^22) = 24^11.

So the biplane gives a 1-valued Gram off-diag at -1 (cardinality 1),
distinct from Orrick (whose off-diag multiset is (-1: 208, +3: 45),
cardinality 2).

This module:
  - Builds the (23, 11, 5)-design.
  - Returns the ±1 incidence matrix.
  - Enumerates k-flip perturbations whose perturbed Gram off-diagonal
    stays inside the 2-valued support {-1, +3}, restricted to
    single-row perturbations (combinatorially tractable: 2^11 < 2^22
    flip patterns to enumerate per starting row, with strong
    feasibility filters).

Why row-restricted?  A flip at position (r, c) only changes row r's
inner products with the other 22 rows; column c's effect on column-Gram
is irrelevant for the |det| formula via row Gram. The net change to
<v_r, v_i> for i != r equals sum over flipped columns c of
(-2 * M_old[r][c]) * M_old[i][c]  =  -2 * sum_c M_old[r][c] * M_old[i][c]
restricted to the flip set. So per-(r, i) the change is an even integer
in [-2k, +2k] for k flips, and feasibility (change in {0, +4}) gives a
strong sieve.

The biplane starts at Gram = -1 (all off-diag), so a feasible perturbation
sends each off-diag involving row r to either -1 or +3 (i.e., delta ∈ {0, +4}).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Optional

import itertools
import numpy as np

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Quadratic residues mod q
# ---------------------------------------------------------------------------


def quadratic_residues_mod_q(q: int) -> list[int]:
    """Return sorted list of nonzero quadratic residues mod q.

    For q = 23 prime this returns the 11-element set
    {1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18}, the well-known QR set.
    """
    if q < 2:
        raise ValueError(f"q must be >= 2, got {q}")
    return sorted({(x * x) % q for x in range(1, q)})


# ---------------------------------------------------------------------------
# Paley biplane (23, 11, 5)
# ---------------------------------------------------------------------------


def paley_biplane_23() -> list[list[int]]:
    """Build the ±1 incidence matrix of the (23, 11, 5)-Paley biplane.

    Construction:
      - Points: {0, 1, ..., 22}.
      - Block B_i = (i + Q) mod 23, where Q = quadratic_residues_mod_q(23).
        That is, B_i = {(i + q) mod 23 : q in Q}, a cyclic translate.
      - Incidence: matrix entry M[i][j] = +1 iff j ∈ B_i, else -1.

    Returns a 23x23 list-of-lists of ±1 ints.
    """
    Q = quadratic_residues_mod_q(23)
    n = 23
    M = [[-1] * n for _ in range(n)]
    for i in range(n):
        block = {(i + q) % n for q in Q}
        for j in block:
            M[i][j] = 1
    return M


def biplane_design_check(M: list[list[int]]) -> dict:
    """Verify (23, 11, 5)-design properties on a ±1 incidence matrix.

    Returns a dict with:
      - row_weights: list[int], should be all 11.
      - off_diag_inner_products: dict[int, int], multiset; should be {-1: 253}.
      - all_lambda_5: bool, every pair of rows shares exactly 5 +1 columns.
      - abs_det: int, exact via Bareiss.
    """
    A = np.asarray(M, dtype=int)
    n = A.shape[0]
    # Row weight = number of +1 entries per row
    row_weights = [(int((A[i] == 1).sum())) for i in range(n)]
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    off = G[iu].astype(int).tolist()
    from collections import Counter
    cd = dict(Counter(off))
    # Pairwise +1-coincidence count (the lambda)
    plus = ((A + 1) // 2)  # 0/1 incidence
    P = plus @ plus.T
    pairwise_lambda = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_lambda.append(int(P[i, j]))
    all_lambda_5 = all(x == 5 for x in pairwise_lambda)
    abs_det = abs(integer_determinant([row[:] for row in M]))
    return {
        "row_weights": row_weights,
        "row_weights_uniform_11": all(w == 11 for w in row_weights),
        "off_diag_multiset": cd,
        "off_diag_uniform_minus_1": cd == {-1: n * (n - 1) // 2},
        "all_lambda_5": all_lambda_5,
        "abs_det": abs_det,
        "abs_det_factored_24_pow_11": abs_det == 24 ** 11,
    }


# ---------------------------------------------------------------------------
# Single-row perturbation enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowPerturbation:
    """A single-row flip pattern of the biplane and its Gram footprint."""

    row: int
    flip_columns: tuple[int, ...]   # columns flipped in `row`
    abs_det: int
    count_minus1: int               # off-diag pairs at -1
    count_plus3: int                # off-diag pairs at +3
    deltas_to_other_rows: tuple[int, ...]  # length 22, each in {0, 4}


def _row_pert_inner_product_changes(
    M: np.ndarray, row: int, flip_cols: Sequence[int]
) -> np.ndarray:
    """Return the change Δ<v_r, v_i> for i = 0, ..., n-1 except i = row.

    Each flipped column c flips M[r, c] from ±1 to ∓1, so the new value is
    M[r, c]_new = -M[r, c]_old, i.e. delta = -2 * M[r, c]_old. The induced
    change in <v_r, v_i> equals  (-2 * M[r, c]_old) * M[i, c]  summed over
    flipped cols c.

    Returns an int array of length n-1 (skipping i = row).
    """
    n = M.shape[0]
    if not flip_cols:
        return np.zeros(n - 1, dtype=int)
    cols = np.asarray(flip_cols, dtype=int)
    # Per-column contribution at each i: -2 * M[row, c] * M[i, c].
    # Sum over c to get total delta per i.
    coeffs = -2 * M[row, cols]                       # shape (k,)
    contribs = M[:, cols] * coeffs[None, :]          # shape (n, k)
    delta_all = contribs.sum(axis=1)                 # shape (n,)
    # Drop the row itself.
    delta_others = np.delete(delta_all, row)
    return delta_others.astype(int)


def feasible_row_perturbations(
    M0: Sequence[Sequence[int]],
    row: int,
    max_flips: int,
    *,
    min_flips: int = 1,
    target_support: tuple[int, int] = (-1, 3),
    require_each_delta_in: tuple[int, ...] = (0, 4),
    base_offdiag_value: int = -1,
) -> Iterator[RowPerturbation]:
    """Enumerate all single-row flip subsets of size ≤ max_flips that produce
    a Gram whose off-diagonal stays in `target_support`.

    Restricted to flip patterns of one fixed `row`. Each flip set S of
    columns is characterised by:

      - For every other row i, the change to <v_row, v_i> must lie in
        `require_each_delta_in` (default {0, +4}, sending base off-diag
        value -1 to either -1 or +3).

    For each feasible S we compute |det| via exact Bareiss on the perturbed
    matrix, count the off-diagonal pairs in the new Gram at value -1 and at
    value +3, and yield a `RowPerturbation`.

    Note: brute enumeration over C(23, k) for k ≤ 11 totals
        sum_{k=0}^{11} C(23, k) = 4_194_303 + 1 ≈ 4.2e6
    pure subsets per row; with feasibility sieve the survivor set is far
    smaller. We iterate by k from min_flips up to max_flips.
    """
    M_arr = np.asarray(M0, dtype=int)
    n = M_arr.shape[0]
    if n != 23:
        raise ValueError(f"this enumerator targets n=23; got n={n}")
    if row < 0 or row >= n:
        raise ValueError(f"row out of range; got {row}")

    allowed_deltas = set(int(d) for d in require_each_delta_in)
    target_set = set(int(v) for v in target_support)

    # Precompute per-column "signed partner" vectors:
    #   delta_per_col[c, i] = -2 * M[row, c] * M[i, c]   (in {-2, +2})
    delta_per_col = (-2 * M_arr[row, :])[:, None] * M_arr.T   # shape (n, n)
    # delta_per_col[c, i] = change to <v_row, v_i> from flipping ONLY col c.
    # We want to drop the row index from i.

    cols = list(range(n))
    for k in range(max(min_flips, 0), max_flips + 1):
        if k == 0:
            # Trivial: zero flips. Gram unchanged from biplane (all -1).
            base_M = [r[:] for r in M0]
            d = abs(integer_determinant(base_M))
            yield RowPerturbation(
                row=row, flip_columns=(),
                abs_det=int(d),
                count_minus1=n * (n - 1) // 2, count_plus3=0,
                deltas_to_other_rows=tuple([0] * (n - 1)),
            )
            continue

        for subset in itertools.combinations(cols, k):
            # Compute total delta per i from this subset.
            sel = np.asarray(subset, dtype=int)
            tot = delta_per_col[sel, :].sum(axis=0)        # shape (n,)
            tot_others = np.delete(tot, row)
            if not all(int(d) in allowed_deltas for d in tot_others.tolist()):
                continue

            # Build perturbed matrix and verify exact |det|.
            M_new = [r[:] for r in M0]
            for c in subset:
                M_new[row][c] = -M_new[row][c]
            d = abs(integer_determinant(M_new))

            # Compute new off-diagonal Gram counts.
            arr_new = np.asarray(M_new, dtype=int)
            G_new = arr_new @ arr_new.T
            iu = np.triu_indices(n, k=1)
            off = G_new[iu].astype(int)
            # Quick feasibility check: every off-diag must be in target_set.
            if not set(off.tolist()).issubset(target_set):
                # Should be impossible given delta sieve, but guard anyway.
                continue
            cm1 = int((off == int(target_support[0])).sum())
            cp3 = int((off == int(target_support[1])).sum())
            yield RowPerturbation(
                row=row, flip_columns=tuple(subset),
                abs_det=int(d), count_minus1=cm1, count_plus3=cp3,
                deltas_to_other_rows=tuple(int(x) for x in tot_others.tolist()),
            )


# ---------------------------------------------------------------------------
# Multi-row block enumeration (cheap subset of full k-flip search)
# ---------------------------------------------------------------------------


def feasible_two_row_perturbations(
    M0: Sequence[Sequence[int]],
    row_a: int,
    row_b: int,
    max_flips_per_row: int,
    *,
    target_support: tuple[int, int] = (-1, 3),
    base_offdiag_value: int = -1,
) -> Iterator[RowPerturbation]:
    """Enumerate flip subsets supported on two distinct rows.

    For (row_a, row_b) and per-row flip subsets S_a, S_b each of size ≤
    max_flips_per_row, the Gram changes split into three parts:

      - <v_a, v_i>, i not in {a,b}: depends only on S_a.
      - <v_b, v_i>, i not in {a,b}: depends only on S_b.
      - <v_a, v_b>: depends on both S_a and S_b (with a cross-term at
        columns in S_a ∩ S_b where both row entries flip).

    All three deltas must put the resulting Gram entry in
    `target_support`. Note both rows touched ⇒ 2*(n-2) + 1 = 43 changed
    off-diag pairs out of C(23, 2) = 253.

    Because the Cartesian product blows up (4.2e6 × 4.2e6), we restrict
    to small max_flips_per_row (≤ 4 in practice).
    """
    if row_a == row_b:
        raise ValueError("two-row enumerator requires distinct rows")
    M_arr = np.asarray(M0, dtype=int)
    n = M_arr.shape[0]
    cols = list(range(n))
    # Precompute per-column partial deltas (for delta sieve)
    delta_a = (-2 * M_arr[row_a, :])[:, None] * M_arr.T   # (n, n)
    delta_b = (-2 * M_arr[row_b, :])[:, None] * M_arr.T   # (n, n)
    target_set = set(int(v) for v in target_support)
    iu = np.triu_indices(n, k=1)

    for ka in range(0, max_flips_per_row + 1):
        for kb in range(0, max_flips_per_row + 1):
            if ka + kb == 0:
                continue
            for Sa in itertools.combinations(cols, ka):
                sel_a = np.asarray(Sa, dtype=int) if Sa else np.empty(0, dtype=int)
                # Total delta to <v_a, v_i> from S_a alone (i != a, b yet)
                ta_all = delta_a[sel_a, :].sum(axis=0) if sel_a.size > 0 else np.zeros(n, dtype=int)
                # Pre-filter: deltas to rows != a, b must be in {0, +4}
                idx_other = [i for i in range(n) if i not in (row_a, row_b)]
                if not all(int(d) in (0, 4) for d in ta_all[idx_other].tolist()):
                    continue
                for Sb in itertools.combinations(cols, kb):
                    sel_b = np.asarray(Sb, dtype=int) if Sb else np.empty(0, dtype=int)
                    tb_all = delta_b[sel_b, :].sum(axis=0) if sel_b.size > 0 else np.zeros(n, dtype=int)
                    if not all(int(d) in (0, 4) for d in tb_all[idx_other].tolist()):
                        continue

                    # Cross delta to <v_a, v_b>: from S_a side
                    #   sum over c in S_a of (-2 * M[a,c]) * M[b,c]
                    # plus from S_b side
                    #   sum over c in S_b of M[a,c] * (-2 * M[b,c])
                    # plus cross-flip correction at c in S_a ∩ S_b:
                    #   we double-count (-2)*(-2)*M[a,c]*M[b,c]/(M[a,c]M[b,c])
                    # The clean way: compute full delta directly by building
                    # the perturbed matrix and recomputing <v_a, v_b>.
                    M_new = [r[:] for r in M0]
                    for c in Sa:
                        M_new[row_a][c] = -M_new[row_a][c]
                    for c in Sb:
                        M_new[row_b][c] = -M_new[row_b][c]
                    arr_new = np.asarray(M_new, dtype=int)
                    G_new = arr_new @ arr_new.T
                    if not set(G_new[iu].tolist()).issubset(target_set):
                        continue
                    off = G_new[iu]
                    cm1 = int((off == int(target_support[0])).sum())
                    cp3 = int((off == int(target_support[1])).sum())
                    d = abs(integer_determinant(M_new))
                    # Use the union of flips for the record.
                    flips = tuple(sorted(set([(row_a, c) for c in Sa] + [(row_b, c) for c in Sb])))
                    # Pack into RowPerturbation (re-using fields loosely).
                    yield RowPerturbation(
                        row=-1,
                        flip_columns=tuple([(r * n + c) for (r, c) in flips]),
                        abs_det=int(d),
                        count_minus1=cm1, count_plus3=cp3,
                        deltas_to_other_rows=tuple(),
                    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def multiset_grouped_results(
    pert_iter: Iterator[RowPerturbation],
) -> dict[tuple[int, int], dict]:
    """Group perturbations by (count_minus1, count_plus3) and report best |det|.

    Returns a dict mapping (cm1, cp3) -> {
        "n_perturbations": int,
        "best_abs_det": int,
        "best_flip_columns": tuple[int, ...],
        "best_row": int,
    }.
    """
    out: dict[tuple[int, int], dict] = {}
    for p in pert_iter:
        key = (int(p.count_minus1), int(p.count_plus3))
        rec = out.get(key)
        if rec is None:
            out[key] = {
                "n_perturbations": 1,
                "best_abs_det": int(p.abs_det),
                "best_flip_columns": tuple(int(c) for c in p.flip_columns),
                "best_row": int(p.row),
            }
        else:
            rec["n_perturbations"] += 1
            if int(p.abs_det) > rec["best_abs_det"]:
                rec["best_abs_det"] = int(p.abs_det)
                rec["best_flip_columns"] = tuple(int(c) for c in p.flip_columns)
                rec["best_row"] = int(p.row)
    return out


__all__ = [
    "quadratic_residues_mod_q",
    "paley_biplane_23",
    "biplane_design_check",
    "RowPerturbation",
    "feasible_row_perturbations",
    "feasible_two_row_perturbations",
    "multiset_grouped_results",
]
