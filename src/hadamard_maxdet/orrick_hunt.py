"""Orrick H23 record hunt — skew-biased SA + Ehlich-block partition enumeration.

The class-tabu campaign (`docs/findings.md`, 2026-04-25) showed that all
in-repo H23 leaders sit in the 4-valued Gram support family
``{-3, -1, +1, +3}``. The current best (BTC, |det|=2.6266e15, 0.6702·Barba)
has multiset

    (-3:11, -1:209, +1:11, +3:22)

— extremely peaked at -1 (209 of 253 off-diagonal pairs). Orrick's published
record (|det|≈2.778e15, 0.7091·Barba) is hypothesised to live in the same
4-valued family, **even more skewed toward -1** than BTC.

This module provides three tools to test that hypothesis:

* :func:`skew_biased_energy` and :func:`sa_skew_biased` — SA whose energy
  rewards (count(-1)+count(-3))-(count(+1)+count(+3)) and heavily penalises
  any Gram off-diagonal outside ``{-3,-1,+1,+3}``. Steers toward
  more-skewed cousins of BTC.

* :func:`enumerate_ehlich_partitions` — for n ≡ 3 (mod 4), enumerate
  ordered partitions ``(s_1, s_2, s_3, s_4)`` of n with a *predicted*
  maxdet under the Ehlich block-structure conjecture: in the optimal
  4-valued Gram, rows partition into 4 disjoint blocks with intra-block
  inner product ``+3`` and cross-block inner product ``-1``. Every other
  off-diagonal entry must be ``-3`` or ``+1`` (the "perturbation" outside
  the strict block model). The closed-form predicted ``|det|`` is
  ``sqrt(det(G_predicted))``.

* :func:`construct_from_partition` — attempt to realise a partition's
  block structure as a ±1 matrix via constrained SA: target Gram is the
  predicted block-Gram, and SA minimises ``||G(M) - G_target||_F^2`` over
  ±1 matrices.

The Ehlich-Wojtas closed form for n ≡ 3 (mod 4) is:

    |det| ≤ sqrt((4n - 3)(n - 1)^(n-1))   (Barba)

attained iff there is a {-1,+1} matrix with G = MM^T equal to the special
block matrix above for some balanced partition. Existence has been
proven for some n (n=3, 7, 11, 15, 19) but **not** for n=23 — Orrick's
0.7091 number is below Barba's 1.0, indicating a strict gap and a
suboptimal block partition.

The partition enumeration here treats the predicted ``|det|`` as an
*upper bound on what that partition can achieve*; the construction
problem is hard and we do not promise to realise every partition.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from hadamard_maxdet.class_tabu import class_signature
from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Skew-biased energy (Strategy A)
# ---------------------------------------------------------------------------


ALLOWED_GRAM_VALUES_4VAL = frozenset({-3, -1, 1, 3})


def gram_offdiag_counts(matrix: Sequence[Sequence[int]]) -> Counter:
    """Return a ``Counter`` of strictly-upper-triangular Gram off-diagonals."""
    n = len(matrix)
    M = np.asarray(matrix, dtype=np.int64)
    G = M @ M.T
    counts: Counter = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            counts[int(G[i, j])] += 1
    return counts


def skew_biased_energy(
    matrix: Sequence[Sequence[int]],
    *,
    lambda_skew: float = 0.0,
    mu_support: float = 0.0,
    abs_det: int | None = None,
) -> float:
    """Energy ``E = -log|det| - lambda_skew * skew - mu_support * support_penalty``.

    * ``skew`` = (count(-1) + count(-3)) - (count(+1) + count(+3)).
      Larger when the matrix is more "−1-dominant" (BTC-like).
    * ``support_penalty`` = number of off-diagonal Gram entries OUTSIDE
      ``{-3, -1, +1, +3}``. Heavily penalised to keep the search inside
      the 4-valued cousin family. ``mu_support * support_penalty`` is
      *subtracted* from energy, so positive ``mu_support`` *raises*
      energy when the count is > 0 (the sign convention here: we actually
      want to *add* the penalty to energy, so the expression is
      equivalent to writing ``E = -log|det| - lambda_skew * skew + mu_support * support_penalty``;
      we name the parameter ``mu_support`` and treat it as an additive
      penalty per spec).

    For ``abs_det == 0`` we return ``+inf`` (singular matrices are never
    accepted).

    Notes
    -----
    The spec writes ``E(M) = -log|det| - lambda * skew - mu * support_penalty``
    and says ``mu = 100`` keeps us in the 4-valued class. A *negative*
    coefficient on a positive penalty would *reward* leaving the class,
    so we treat ``mu_support * support_penalty`` as a positive *additive*
    penalty (i.e. effectively ``+mu_support * support_penalty``). This
    matches the stated intent ("heavy penalty to stay in 4-val class").
    """
    if abs_det is None:
        det = integer_determinant(matrix)
        abs_det = abs(det)
    if abs_det == 0:
        return math.inf

    counts = gram_offdiag_counts(matrix)
    neg = counts.get(-1, 0) + counts.get(-3, 0)
    pos = counts.get(+1, 0) + counts.get(+3, 0)
    skew = neg - pos

    support_penalty = sum(
        c for v, c in counts.items() if v not in ALLOWED_GRAM_VALUES_4VAL
    )

    return -math.log(abs_det) - lambda_skew * skew + mu_support * support_penalty


def _compute_skew_and_penalty(M: Sequence[Sequence[int]]) -> tuple[int, int]:
    """Return ``(skew, support_penalty)`` for a ±1 matrix."""
    counts = gram_offdiag_counts(M)
    neg = counts.get(-1, 0) + counts.get(-3, 0)
    pos = counts.get(+1, 0) + counts.get(+3, 0)
    skew = neg - pos
    support_penalty = sum(
        c for v, c in counts.items() if v not in ALLOWED_GRAM_VALUES_4VAL
    )
    return skew, support_penalty


@dataclass
class SkewBiasedResult:
    matrix: Matrix
    abs_det: int
    final_signature: tuple
    skew: int
    support_penalty: int
    iterations: int
    accepted: int
    rejected_metropolis: int
    rejected_singular: int
    walltime_seconds: float
    lambda_skew: float
    mu_support: float
    history_top: list[tuple[int, int, int, int]] = field(default_factory=list)
    """Each entry: (step, abs_det, skew, support_penalty) at a new best."""


def sa_skew_biased(
    initial: Matrix,
    *,
    iterations: int = 200_000,
    lambda_skew: float = 0.01,
    mu_support: float = 100.0,
    t_start: float = 0.05,
    t_end: float = 0.0005,
    seed: int | None = None,
    refresh_every: int = 2000,
    wall_time_budget: float | None = None,
) -> SkewBiasedResult:
    """SA on ``E = -log|det| - lambda_skew*skew + mu_support*support_penalty``.

    Uses single-entry-flip moves with the rank-1 incremental adjugate
    machinery (so |det| updates are exact and O(n^2) per accepted flip).
    Recomputes ``(skew, support_penalty)`` from the modified Gram on each
    proposal — that is O(n) for a single entry flip (only one row of M
    changes, so 2(n-1) Gram off-diagonals change), but for code clarity
    we recompute the full Gram counts here (still O(n^2), ≈ 500 ops at
    n=23, dwarfed by the rank-1 adjugate update).

    The "best" matrix returned is the one minimising the *biased* energy,
    NOT |det| alone — by design: we want the most-skewed BTC cousin SA
    can find, even if its raw |det| is below BTC's 2.626e15. The final
    |det| is exact-Bareiss verified.

    Parameters
    ----------
    initial
        Starting ±1 matrix (e.g. BTC seed).
    iterations
        Total proposed flips.
    lambda_skew, mu_support
        Energy coefficients (see :func:`skew_biased_energy`).
    t_start, t_end
        Geometric temperature schedule on the biased energy.
    seed
        PRNG seed.
    refresh_every
        Periodic exact recompute of (det, adj). 0 to disable.
    wall_time_budget
        If set, terminate early when wall time exceeds this many seconds.
    """
    rng = random.Random(seed)
    n = initial.n
    state = IncrementalState.from_matrix(initial.entries)
    if state.det == 0:
        raise RuntimeError("orrick_hunt: refuse to anneal from singular seed")

    # Track best by biased energy (which is what we are minimising).
    cur_skew, cur_pen = _compute_skew_and_penalty(state.M)
    cur_abs_det = abs(state.det)
    cur_e = (
        -math.log(cur_abs_det)
        - lambda_skew * cur_skew
        + mu_support * cur_pen
    )

    best_entries = [row[:] for row in state.M]
    best_e = cur_e
    best_abs_det = cur_abs_det
    best_skew = cur_skew
    best_pen = cur_pen

    accepted = 0
    rejected_metro = 0
    rejected_singular = 0
    history_top: list[tuple[int, int, int, int]] = [
        (0, best_abs_det, best_skew, best_pen)
    ]

    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    t0 = time.monotonic()

    step_done = 0
    for step in range(iterations):
        step_done = step
        if wall_time_budget is not None and (step & 1023) == 0:
            if time.monotonic() - t0 >= wall_time_budget:
                break

        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        i = rng.randrange(n)
        j = rng.randrange(n)

        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            rejected_singular += 1
            continue

        # Tentatively flip M to compute proposed (skew, pen). We revert if
        # rejected; if accepted we let apply_entry_flip do its bookkeeping
        # from the original entry, so we revert M either way before deciding.
        state.M[i][j] = -state.M[i][j]
        prop_skew, prop_pen = _compute_skew_and_penalty(state.M)
        state.M[i][j] = -state.M[i][j]

        new_abs_det = abs(new_det)
        prop_e = (
            -math.log(new_abs_det)
            - lambda_skew * prop_skew
            + mu_support * prop_pen
        )
        delta = prop_e - cur_e

        if delta > 0 and rng.random() >= math.exp(-delta / max(t, 1e-12)):
            rejected_metro += 1
            continue

        apply_entry_flip(state, i, j)
        cur_e = prop_e
        cur_abs_det = new_abs_det
        cur_skew = prop_skew
        cur_pen = prop_pen
        accepted += 1

        if cur_e < best_e:
            best_e = cur_e
            best_abs_det = cur_abs_det
            best_skew = cur_skew
            best_pen = cur_pen
            best_entries = [row[:] for row in state.M]
            history_top.append((step, best_abs_det, best_skew, best_pen))

        if refresh_every > 0 and accepted % refresh_every == 0:
            refresh(state)
            if state.det != 0:
                cur_abs_det = abs(state.det)
                cur_skew, cur_pen = _compute_skew_and_penalty(state.M)
                cur_e = (
                    -math.log(cur_abs_det)
                    - lambda_skew * cur_skew
                    + mu_support * cur_pen
                )

    walltime = time.monotonic() - t0
    final_abs_det = abs(integer_determinant(best_entries))
    final_sig = class_signature(best_entries)
    final_skew, final_pen = _compute_skew_and_penalty(best_entries)
    return SkewBiasedResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=final_abs_det,
        final_signature=final_sig,
        skew=final_skew,
        support_penalty=final_pen,
        iterations=step_done + 1,
        accepted=accepted,
        rejected_metropolis=rejected_metro,
        rejected_singular=rejected_singular,
        walltime_seconds=walltime,
        lambda_skew=lambda_skew,
        mu_support=mu_support,
        history_top=history_top,
    )


# ---------------------------------------------------------------------------
# Ehlich block-decomposition enumeration (Strategy B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EhlichPartition:
    """Ordered partition of n into 4 nonneg parts with a predicted max-|det|.

    The model: rows of M partition into 4 blocks ``S_1, ..., S_4`` of sizes
    ``(s_1, s_2, s_3, s_4)``. Intra-block Gram entries (off-diagonal in G,
    i.e. distinct rows in the same block) are all ``intra_value``;
    cross-block Gram entries are all ``cross_value``. The diagonal of G
    is ``n``.

    The *predicted* ``|det|`` is ``sqrt(det(G))`` for that block-structured
    G (positive-definite when valid; otherwise the partition is ruled out).
    """

    sizes: tuple[int, int, int, int]
    intra_value: int
    cross_value: int
    n: int
    predicted_abs_det: float


def _block_gram_matrix(
    sizes: tuple[int, int, int, int],
    n_diag: int,
    intra: int,
    cross: int,
) -> np.ndarray:
    """Construct the block-structured Gram matrix for a given partition."""
    n = sum(sizes)
    G = np.full((n, n), cross, dtype=np.float64)
    block_starts: list[int] = []
    s = 0
    for sz in sizes:
        block_starts.append(s)
        s += sz
    for k, sz in enumerate(sizes):
        start = block_starts[k]
        G[start : start + sz, start : start + sz] = intra
    np.fill_diagonal(G, n_diag)
    return G


def enumerate_ehlich_partitions(
    n: int,
    *,
    intra_values: Sequence[int] = (3,),
    cross_values: Sequence[int] = (-1,),
    min_block_size: int = 1,
    top_k: int | None = None,
) -> list[EhlichPartition]:
    """Enumerate ordered 4-partitions of ``n`` with predicted block-Gram det.

    For ``n ≡ 3 (mod 4)`` the canonical 4-valued ansatz puts intra-block
    inner products at ``+3`` and cross-block at ``-1`` (cf. BTC's multiset
    weighting at H23). Pass alternate ``intra_values`` / ``cross_values``
    to scan related ansätze.

    Parameters
    ----------
    n
        Matrix order.
    intra_values, cross_values
        Candidate values to try for intra-block / cross-block Gram entries.
        Defaults ``(3,)`` and ``(-1,)`` per the Ehlich-Wojtas conjecture
        for ``n ≡ 3 (mod 4)``.
    min_block_size
        Drop partitions with any block smaller than this. ``1`` allows
        degenerate (one-row) blocks; set to ``2`` to require every block
        to contribute at least one intra-pair.
    top_k
        If set, return only the top-K partitions by predicted ``|det|``.
        Otherwise return all that yield positive-definite block-Gram.

    Notes
    -----
    Ordered partitions are returned (the labels ``S_1, S_2, S_3, S_4``
    matter for the construction step). Permutations of the same multiset
    of sizes give the same predicted determinant; we deduplicate by sorting
    the multiset.
    """
    out: list[EhlichPartition] = []
    seen_multisets: set[tuple] = set()

    for s1 in range(min_block_size, n - 3 * min_block_size + 1):
        for s2 in range(min_block_size, n - s1 - 2 * min_block_size + 1):
            for s3 in range(min_block_size, n - s1 - s2 - min_block_size + 1):
                s4 = n - s1 - s2 - s3
                if s4 < min_block_size:
                    continue
                key = tuple(sorted((s1, s2, s3, s4)))
                for intra in intra_values:
                    for cross in cross_values:
                        full_key = (key, intra, cross)
                        if full_key in seen_multisets:
                            continue
                        seen_multisets.add(full_key)
                        G = _block_gram_matrix((s1, s2, s3, s4), n, intra, cross)
                        # Compute determinant; require positivity.
                        # det(G) >= 0 is necessary for G = M M^T realisability.
                        sign, logdet = np.linalg.slogdet(G)
                        if sign <= 0:
                            continue
                        predicted = math.exp(0.5 * logdet)
                        out.append(
                            EhlichPartition(
                                sizes=key,
                                intra_value=intra,
                                cross_value=cross,
                                n=n,
                                predicted_abs_det=predicted,
                            )
                        )
    out.sort(key=lambda p: -p.predicted_abs_det)
    if top_k is not None:
        out = out[:top_k]
    return out


# ---------------------------------------------------------------------------
# Block construction (Strategy B continued / Strategy C)
# ---------------------------------------------------------------------------


def _row_block(row_index: int, block_starts: list[int], sizes: tuple[int, ...]) -> int:
    for k, (start, sz) in enumerate(zip(block_starts, sizes)):
        if start <= row_index < start + sz:
            return k
    raise ValueError(f"row {row_index} not in any block")


@dataclass
class ConstructionResult:
    matrix: Matrix
    abs_det: int
    target_intra: int
    target_cross: int
    final_gram_loss: int
    matched_block_structure: bool
    iterations: int
    walltime_seconds: float
    final_signature: tuple


def _gram_loss(M: np.ndarray, target_intra: int, target_cross: int,
               block_of_row: list[int]) -> int:
    """Sum of squared deviations between Gram off-diagonals and target."""
    n = M.shape[0]
    G = M @ M.T
    loss = 0
    for i in range(n):
        for j in range(i + 1, n):
            tgt = target_intra if block_of_row[i] == block_of_row[j] else target_cross
            d = int(G[i, j]) - tgt
            loss += d * d
    return loss


def construct_from_partition(
    partition: EhlichPartition,
    *,
    iterations: int = 50_000,
    seed: int | None = None,
    t_start: float = 4.0,
    t_end: float = 0.05,
    wall_time_budget: float | None = None,
) -> ConstructionResult:
    """Attempt to construct a ±1 matrix realising the partition's block Gram.

    SA on ``±1 matrix → ||G(M) - G_target||_F^2`` with single-entry-flip
    moves. Starts from a random ±1 matrix (no structured warm-start —
    structured warm-starts for arbitrary partitions are themselves a
    research problem). Returns the matrix with smallest Gram loss found,
    plus its exact |det|.

    A "successful" construction has ``final_gram_loss == 0`` and matches
    the predicted block structure exactly. Most partitions will *not*
    succeed under this attack (constraint satisfaction is hard); we report
    the best loss reached and the corresponding matrix's |det|.
    """
    rng = random.Random(seed)
    n = partition.n
    sizes = partition.sizes
    block_starts: list[int] = []
    s = 0
    for sz in sizes:
        block_starts.append(s)
        s += sz
    block_of_row = [_row_block(i, block_starts, sizes) for i in range(n)]

    # Random ±1 start.
    M = np.array(
        [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)],
        dtype=np.int64,
    )
    # Maintain Gram incrementally for O(n) per flip.
    G = M @ M.T

    def loss_from_G(G_local: np.ndarray) -> int:
        L = 0
        for i in range(n):
            for j in range(i + 1, n):
                tgt = (
                    partition.intra_value
                    if block_of_row[i] == block_of_row[j]
                    else partition.cross_value
                )
                d = int(G_local[i, j]) - tgt
                L += d * d
        return L

    cur_loss = loss_from_G(G)
    best_loss = cur_loss
    best_M = M.copy()

    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    t0 = time.monotonic()

    step_done = 0
    for step in range(iterations):
        step_done = step
        if wall_time_budget is not None and (step & 1023) == 0:
            if time.monotonic() - t0 >= wall_time_budget:
                break

        frac = step / max(1, iterations - 1)
        T = t_start * math.exp(log_ratio * frac)

        i = rng.randrange(n)
        j = rng.randrange(n)
        old = int(M[i, j])
        # Effect of flipping M[i,j] on G: only row i and column i of G change.
        # G_new[i,k] = G[i,k] - 2 * old * M[k,j], for k != i (and similarly G[k,i]).
        # G[i,i] is unchanged (diagonal: sum of squares).
        delta_loss = 0
        for k in range(n):
            if k == i:
                continue
            tgt = (
                partition.intra_value
                if block_of_row[i] == block_of_row[k]
                else partition.cross_value
            )
            old_val = int(G[i, k])
            new_val = old_val - 2 * old * int(M[k, j])
            old_d = old_val - tgt
            new_d = new_val - tgt
            delta_loss += new_d * new_d - old_d * old_d

        if delta_loss <= 0 or rng.random() < math.exp(-delta_loss / max(T, 1e-9)):
            # Accept: commit the flip and update G.
            M[i, j] = -old
            for k in range(n):
                if k == i:
                    continue
                G[i, k] -= 2 * old * int(M[k, j])
                G[k, i] = G[i, k]
            cur_loss += delta_loss
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_M = M.copy()
                if best_loss == 0:
                    break

    walltime = time.monotonic() - t0
    best_entries = [[int(x) for x in row] for row in best_M]
    abs_det = abs(integer_determinant(best_entries))
    matched = best_loss == 0
    return ConstructionResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=abs_det,
        target_intra=partition.intra_value,
        target_cross=partition.cross_value,
        final_gram_loss=int(best_loss),
        matched_block_structure=matched,
        iterations=step_done + 1,
        walltime_seconds=walltime,
        final_signature=class_signature(best_entries),
    )


__all__ = [
    "ALLOWED_GRAM_VALUES_4VAL",
    "gram_offdiag_counts",
    "skew_biased_energy",
    "SkewBiasedResult",
    "sa_skew_biased",
    "EhlichPartition",
    "enumerate_ehlich_partitions",
    "ConstructionResult",
    "construct_from_partition",
]
