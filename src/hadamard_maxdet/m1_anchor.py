"""M1-anchored simulated annealing.

Calibrates the M1 discriminator (cardinality of the triple cyclic Gram-product
multiset) as an SA energy bias:

    E(M) = -log|det(M)| + lambda * |T_support(M)|

where T_support(M) = number of distinct values among
T(i, j, k) = <v_i, v_j> * <v_j, v_k> * <v_k, v_i>  for i < j < k.

This module exposes:

* :func:`t_support_cardinality` — full O(n^3) recompute of |T-support|.
* :class:`TSupportState` — incremental tracker of the T-multiset under
  single-entry flips, with O(n^2) updates per flip.
* :func:`sa_m1_anchored` — SA over E(M) with incremental |det| (via
  :class:`IncrementalState`) and incremental T-support.

Per-flip cost: O(n^2) for T-support update + O(n^2) for adjugate update +
O(n) for Gram update. For n = 19 this is roughly 400 + 360 + 19 ≈ 800
operations per flip — well within the budget for 60-300 second SA runs.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hadamard_maxdet.incremental import (
    IncrementalAnnealResult,
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Full recompute (O(n^3)) — used for bootstrapping and tests
# ---------------------------------------------------------------------------


def t_support_cardinality(M: Sequence[Sequence[int]]) -> int:
    """Return |T-support(M)|, the number of distinct triple-product values.

    O(n^3) full recompute. Equivalent to
    ``triple_cyclic_product_stats(M)["cardinality"]`` from
    :mod:`structural_test` but avoids the dictionary/moment overhead.
    """
    A = np.asarray(M, dtype=int)
    n = A.shape[0]
    if n < 3:
        return 0
    G = (A @ A.T).astype(int)
    seen: set[int] = set()
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            gij = int(G[i, j])
            for k in range(j + 1, n):
                t = gij * int(G[j, k]) * int(G[k, i])
                seen.add(t)
    return len(seen)


def t_support_counter(M: Sequence[Sequence[int]]) -> Counter:
    """Return a Counter mapping every T value to its multiplicity.

    O(n^3) full recompute. Used by :class:`TSupportState` to seed itself.
    """
    A = np.asarray(M, dtype=int)
    n = A.shape[0]
    cnt: Counter[int] = Counter()
    if n < 3:
        return cnt
    G = (A @ A.T).astype(int)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            gij = int(G[i, j])
            for k in range(j + 1, n):
                t = gij * int(G[j, k]) * int(G[k, i])
                cnt[t] += 1
    return cnt


# ---------------------------------------------------------------------------
# Incremental T-support state
# ---------------------------------------------------------------------------


@dataclass
class TSupportState:
    """Carries the T-value multiset and the row Gram of M.

    Invariants:
      * ``G == M @ M.T`` (numpy int array of shape (n, n))
      * ``counter`` is the multiset of ``T(i, j, k)`` over all i < j < k.
      * ``cardinality`` equals the number of keys in ``counter`` with
        positive count.
    """

    n: int
    G: np.ndarray   # int row Gram (n, n)
    counter: Counter
    cardinality: int

    @classmethod
    def from_matrix(cls, M: Sequence[Sequence[int]]) -> "TSupportState":
        A = np.asarray(M, dtype=int)
        n = A.shape[0]
        G = (A @ A.T).astype(int)
        cnt = t_support_counter(M)
        # Drop zero-count keys defensively.
        cnt = Counter({k: v for k, v in cnt.items() if v > 0})
        return cls(n=n, G=G, counter=cnt, cardinality=len(cnt))

    # -- update -----------------------------------------------------------

    def update_after_flip(
        self, M: Sequence[Sequence[int]], i: int, c: int
    ) -> None:
        """Update G and T-multiset to reflect that M[i][c] has just been
        flipped (i.e. ``M[i][c]`` *is already the new value* on entry).

        Cost: O(n^2). The entry flip changes G[i, r] = G[r, i] for all
        r != i by ``delta_factor * M[r][c]`` where ``delta_factor =
        2 * M_new[i][c]`` (since M_new = -M_old, so old = -new and the
        delta to G[i, r] = sum_k M[i, k] * M[r, k] is
        +2 * M_new[i][c] * M[r][c]).

        Triples that change are exactly those involving row i (any
        ordered i' < j' < k' with i' = i, or j' = i, or k' = i). For each
        such triple, two of the three factors involve row i and change;
        the third does not. We rebuild T_old and T_new for each.
        """
        n = self.n
        if n < 3:
            # No triples; just refresh G.
            A = np.asarray(M, dtype=int)
            self.G = (A @ A.T).astype(int)
            return

        new_val = M[i][c]
        delta_factor = 2 * int(new_val)
        # Compute the per-r delta to G[i, r]; r != i.
        deltas = np.zeros(n, dtype=int)
        for r in range(n):
            if r == i:
                continue
            deltas[r] = delta_factor * int(M[r][c])

        # Gather T_old and T_new for every triple involving row i.
        # The other two rows (j, k) range over the n-1 non-i rows; pick
        # j < k with j, k != i. There are C(n-1, 2) such unordered pairs.
        # For each, the triple is the sorted version of (i, j, k).
        G = self.G
        cnt = self.counter

        # Capture the OLD G entries for row i before mutation. The middle
        # factor in a triple involves the (j, k) pair (j < k, both != i),
        # whose Gram entry G[j, k] does not change.
        Gi_old = G[i].copy()  # length n

        # Walk every unordered pair (a, b) with a < b, a != i, b != i.
        # For each, compute T_old and T_new of the sorted triple (i, a, b).
        # The triple's three factors are G[i', j'] * G[j', k'] * G[k', i']
        # where (i', j', k') is the sorted permutation. By construction of
        # T's row-symmetry (each row appears in exactly two factors), and
        # because flipping M[i, c] changes only the two factors that touch
        # row i, the OLD value uses Gi_old[a], Gi_old[b], G[a, b]; the NEW
        # value uses (Gi_old[a] + deltas[a]), (Gi_old[b] + deltas[b]),
        # G[a, b].
        non_i = [r for r in range(n) if r != i]
        for ia in range(len(non_i)):
            a = non_i[ia]
            ga_old = int(Gi_old[a])
            ga_new = ga_old + int(deltas[a])
            for ib in range(ia + 1, len(non_i)):
                b = non_i[ib]
                gb_old = int(Gi_old[b])
                gb_new = gb_old + int(deltas[b])
                gab = int(G[a, b])  # unchanged
                t_old = ga_old * gab * gb_old
                t_new = ga_new * gab * gb_new
                if t_old == t_new:
                    continue
                # Decrement old, increment new; maintain cardinality.
                old_count = cnt[t_old]
                if old_count == 1:
                    del cnt[t_old]
                else:
                    cnt[t_old] = old_count - 1
                if t_new in cnt:
                    cnt[t_new] += 1
                else:
                    cnt[t_new] = 1

        # Commit the Gram update for row/column i.
        for r in range(n):
            if r == i:
                continue
            G[i, r] += deltas[r]
            G[r, i] = G[i, r]

        self.cardinality = len(cnt)


# ---------------------------------------------------------------------------
# M1-anchored SA
# ---------------------------------------------------------------------------


def sa_m1_anchored(
    n: int,
    *,
    iterations: int = 50_000,
    t_start: float = 1.0,
    t_end: float = 0.05,
    lambda_m1: float = 0.0,
    seed: int | None = None,
    initial: Matrix | None = None,
    refresh_every: int = 2000,
    log_every: int = 0,
) -> IncrementalAnnealResult:
    """Simulated annealing over

        E(M) = -log|det(M)| + lambda_m1 * |T_support(M)|

    with rank-1 incremental |det| scoring (via :class:`IncrementalState`)
    and incremental T-support tracking (via :class:`TSupportState`).

    Parameters
    ----------
    n
        Matrix order.
    iterations
        Number of MC steps.
    t_start, t_end
        Geometric temperature schedule in energy units. The energy carries
        ``log|det|`` units plus a ``lambda_m1 * cardinality`` term, so when
        ``lambda_m1 = 0`` the schedule reduces exactly to that of
        :func:`sa_incremental`.
    lambda_m1
        Penalty coefficient on the T-support cardinality. ``0`` reduces to
        plain SA; positive biases toward small-cardinality basins.
    seed, initial, refresh_every, log_every
        As in :func:`sa_incremental`.
    """
    rng = random.Random(seed)

    if initial is None:
        mat = Matrix.random(n, rng)
    elif initial.n != n:
        raise ValueError(f"initial order {initial.n} != n={n}")
    else:
        mat = initial.copy()

    state = IncrementalState.from_matrix(mat.entries)
    tries = 0
    while state.det == 0 and tries < 16:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        tries += 1
    if state.det == 0:
        return IncrementalAnnealResult(
            matrix=Matrix(n=n, entries=[row[:] for row in state.M]),
            abs_det=0,
            iterations=0,
            accepted=0,
            walltime_seconds=0.0,
            refreshes=0,
            residual_triggered=0,
        )

    tsupp = TSupportState.from_matrix(state.M)

    def energy_of(det_abs: int, card: int) -> float:
        return -math.log(det_abs) + lambda_m1 * float(card)

    current_e = energy_of(abs(state.det), tsupp.cardinality)
    best_entries = [row[:] for row in state.M]
    best_abs = abs(state.det)
    best_card = tsupp.cardinality
    best_e = current_e

    accepted = 0
    refreshes = 0
    residual_triggered = 0
    history: list[int] = []
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    start = time.monotonic()

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        i = rng.randrange(n)
        c = rng.randrange(n)
        new_det = propose_entry_flip(state, i, c)
        if new_det == 0:
            continue

        # We need the new T-support cardinality WITHOUT mutating tsupp.
        # Cheapest correct path: try-it-then-roll-back. Cost is O(n^2)
        # either way (the update itself is the dominant work). We:
        #   1. apply the M flip (entry-level)
        #   2. update tsupp
        #   3. compute proposal energy
        #   4. if accepted: also commit the (det, adj) update
        #      else: revert the M flip and tsupp update
        # The (det, adj) propose is O(1) and apply is O(n^2). The Gram
        # half lives inside tsupp. Net per-iteration cost: ~3 * O(n^2).
        old_val = state.M[i][c]
        state.M[i][c] = -old_val
        tsupp.update_after_flip(state.M, i, c)
        proposal_card = tsupp.cardinality
        proposal_e = energy_of(abs(new_det), proposal_card)
        delta = proposal_e - current_e

        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            # Accept: also commit the (det, adj) update via apply_entry_flip.
            # apply_entry_flip flips state.M[i][c] internally, so we must
            # un-flip it first to keep the contract (it expects the OLD
            # value still in place).
            state.M[i][c] = old_val
            apply_entry_flip(state, i, c)
            current_e = proposal_e
            accepted += 1
            if abs(state.det) > best_abs:
                best_abs = abs(state.det)
                best_entries = [row[:] for row in state.M]
                best_card = proposal_card
            if current_e < best_e:
                best_e = current_e
            if refresh_every > 0 and accepted % refresh_every == 0:
                old_det = state.det
                refresh(state)
                refreshes += 1
                if state.det != old_det:
                    residual_triggered += 1
                if state.det != 0:
                    # Re-sync energy / re-derive T-support from scratch.
                    tsupp_new = TSupportState.from_matrix(state.M)
                    tsupp.G = tsupp_new.G
                    tsupp.counter = tsupp_new.counter
                    tsupp.cardinality = tsupp_new.cardinality
                    current_e = energy_of(abs(state.det), tsupp.cardinality)
        else:
            # Reject: roll back M and T-support (re-flip and update).
            state.M[i][c] = -state.M[i][c]  # back to old_val
            tsupp.update_after_flip(state.M, i, c)
            # The double update brings tsupp.G back to the pre-proposal
            # value (involutive); cardinality re-syncs naturally.

        if log_every and step % log_every == 0:
            history.append(best_abs)

    walltime = time.monotonic() - start

    # Re-verify with exact Bareiss; never trust the carried det for the
    # final report.
    best_abs_verified = abs(integer_determinant(best_entries))
    return IncrementalAnnealResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_verified,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        refreshes=refreshes,
        residual_triggered=residual_triggered,
        history=history,
    )


__all__ = [
    "t_support_cardinality",
    "t_support_counter",
    "TSupportState",
    "sa_m1_anchored",
]
