"""Basin-bridging crossover for ±1 maxdet search.

Take two parent matrices with high |det| but distinct equivalence classes
(different Gram fingerprints, different basins). Produce children by:

  * row_recombination(P, Q, mask):
        child[i] = P[i] if mask[i] else Q[i]
        — replace a subset of rows of P with rows of Q.

  * row_negate_match(P, Q):
        align signs of Q's rows to maximize per-row Hamming match with P,
        then take the entrywise majority/random tie-break of P and Q.

Children are not, in general, valid local optima — they need to be
refined (SA + 2-flip polish). The hope is that they sit in a *third*
basin distinct from both parents, possibly with a higher peak.

Important: row-permutation and row-sign symmetries of |det| mean that
"P's row 5" can be aligned with any of Q's 19 rows up to sign. We use
:func:`align_rows_lex` to find the best matching permutation+sign that
maximizes total entrywise agreement, before recombining. Without
alignment, recombining typically produces low-|det| garbage.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from hadamard_maxdet.verifier import integer_determinant


def align_rows_to_match(
    target: Sequence[Sequence[int]],
    source: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int], list[int]]:
    """Permute and sign-flip rows of ``source`` to match ``target`` as
    closely as possible (maximize sum-of-row-Hamming-agreements).

    Returns (aligned_source, row_perm, row_signs) where:
      aligned_source[i] = row_signs[i] * source[row_perm[i]]
    and row_perm is a permutation of range(n).

    Greedy O(n^3) matching: for each target row, pick the unused source
    row + sign that maximizes the dot product with the target row.
    """
    n = len(target)
    T = np.array(target, dtype=np.int64)
    S = np.array(source, dtype=np.int64)
    used = [False] * n
    perm = [-1] * n
    signs = [0] * n
    for i in range(n):
        best_score = -10**9
        best_j = -1
        best_sign = 1
        for j in range(n):
            if used[j]:
                continue
            dot = int(np.dot(T[i], S[j]))
            if dot > best_score:
                best_score = dot
                best_j = j
                best_sign = 1
            if -dot > best_score:
                best_score = -dot
                best_j = j
                best_sign = -1
        perm[i] = best_j
        signs[i] = best_sign
        used[best_j] = True
    aligned = [(int(signs[i]) * S[perm[i]]).tolist() for i in range(n)]
    return aligned, perm, signs


def row_recombine(
    parent_a: Sequence[Sequence[int]],
    parent_b: Sequence[Sequence[int]],
    mask: Sequence[bool],
    *,
    align: bool = True,
) -> list[list[int]]:
    """Return child whose row i = a[i] if mask[i] else b'[i],
    with b' aligned to a (default) for sane recombination.
    """
    if align:
        b_aligned, _, _ = align_rows_to_match(parent_a, parent_b)
    else:
        b_aligned = [list(r) for r in parent_b]
    return [
        list(parent_a[i]) if mask[i] else list(b_aligned[i])
        for i in range(len(parent_a))
    ]


def random_mask(n: int, p_a: float, rng: random.Random) -> list[bool]:
    """Bernoulli(p_a) mask of length n."""
    return [rng.random() < p_a for _ in range(n)]


def crossover_children(
    parent_a: Sequence[Sequence[int]],
    parent_b: Sequence[Sequence[int]],
    *,
    n_children: int = 16,
    seed: int = 0,
    p_a_choices: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> list[tuple[list[list[int]], int]]:
    """Generate ``n_children`` row-recombination children of (a, b).

    Returns a list of (child_entries, abs_det) sorted by abs_det desc.
    Singular children are dropped.
    """
    rng = random.Random(seed)
    n = len(parent_a)
    # Pre-align b to a once
    b_aligned, _, _ = align_rows_to_match(parent_a, parent_b)
    children = []
    for _ in range(n_children):
        p_a = rng.choice(p_a_choices)
        mask = [rng.random() < p_a for _ in range(n)]
        child = [
            list(parent_a[i]) if mask[i] else list(b_aligned[i])
            for i in range(n)
        ]
        d = integer_determinant(child)
        if d == 0:
            continue
        children.append((child, abs(d)))
    children.sort(key=lambda x: -x[1])
    return children
