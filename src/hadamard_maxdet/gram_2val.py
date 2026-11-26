"""Gram 2-valued-support search for ±1 matrices.

Background
----------
At order n=23, Orrick et al. (2003) reach |det| = 2,779,447,296,000,000 with
a matrix whose Gram off-diagonal takes values in *exactly two* distinct
integers: -1 and +3. (Counts: 208 unordered pairs at -1, 45 at +3.) Every
search we have run prior to this module operates in 4-valued or richer Gram
support and so cannot reach Orrick's basin from inside it.

This module implements an SA driver that biases toward Gram off-diagonal
support {-1, +3} and toward a target count of +3 entries (`k_target`),
then sweeps `k_target` to test whether Orrick's k=45 split is the unique
optimum at this 2-valued support.

The energy is

    E(M) = -log|det(M)|
         + mu     * (count_outside_support({-1, +3}))^2
         + lambda * (count_offdiag_value(M, +3) - k_target)^2

with `count_outside_support` and `count_offdiag_value` measured on the
unordered off-diagonal pairs (i < j).

Each entry-flip update for these two counts is O(n): only row i and column i
change in the Gram, which touches n-1 unordered pairs.

API
---
    count_offdiag_value(M, v)            -> int
    count_outside_support(M, support)    -> int
    Gram2ValState                        -> dataclass with O(n)/flip updates
    sa_2valued(...)                      -> Sa2ValAnnealResult

The SA loop reuses :class:`hadamard_maxdet.incremental.IncrementalState` for
exact O(n^2) determinant tracking, and adds the constraint counts on top.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Full O(n^2) recompute helpers (used for tests, bootstrap, and audits)
# ---------------------------------------------------------------------------


def count_offdiag_value(M: Sequence[Sequence[int]], v: int) -> int:
    """Return the number of unordered off-diagonal Gram pairs equal to ``v``.

    The Gram is ``G = M @ M.T``; "unordered off-diagonal" means pairs
    (i, j) with i < j. For a matrix M of order n this returns an integer
    in [0, n(n-1)/2].
    """
    A = np.asarray(M, dtype=int)
    n = A.shape[0]
    if n < 2:
        return 0
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    return int(np.sum(G[iu] == v))


def count_outside_support(
    M: Sequence[Sequence[int]], support: Iterable[int]
) -> int:
    """Return the number of unordered off-diagonal Gram pairs NOT in ``support``.

    ``support`` is interpreted as a set of allowed off-diagonal values. Pairs
    with diagonal indices (i == j) are not counted. Returns an integer in
    [0, n(n-1)/2].
    """
    sup = set(int(s) for s in support)
    A = np.asarray(M, dtype=int)
    n = A.shape[0]
    if n < 2:
        return 0
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    vals = G[iu]
    bad = sum(1 for v in vals.tolist() if int(v) not in sup)
    return int(bad)


def gram_offdiag_counter(M: Sequence[Sequence[int]]) -> Counter:
    """Return a Counter of unordered off-diagonal Gram pair values."""
    A = np.asarray(M, dtype=int)
    n = A.shape[0]
    cnt: Counter[int] = Counter()
    if n < 2:
        return cnt
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    for v in G[iu].tolist():
        cnt[int(v)] += 1
    return cnt


# ---------------------------------------------------------------------------
# Incremental state with O(n) per-flip count updates
# ---------------------------------------------------------------------------


@dataclass
class Gram2ValState:
    """Tracks the row Gram and the two scalar counts under entry flips.

    Invariants (maintained by :meth:`update_after_flip`):

      * ``G == M @ M.T``   (numpy int (n, n))
      * ``count_pos == #{ (i, j) : i < j, G[i, j] == pos_value }``
      * ``count_neg == #{ (i, j) : i < j, G[i, j] == neg_value }``
      * ``count_out == n(n-1)/2 - count_pos - count_neg``

    The "support" we enforce is ``{neg_value, pos_value}`` (defaults
    ``(-1, +3)``, matching Orrick's H23 record).
    """

    n: int
    G: np.ndarray  # int Gram
    pos_value: int
    neg_value: int
    count_pos: int
    count_neg: int
    count_out: int

    @classmethod
    def from_matrix(
        cls,
        M: Sequence[Sequence[int]],
        *,
        pos_value: int = 3,
        neg_value: int = -1,
    ) -> "Gram2ValState":
        A = np.asarray(M, dtype=int)
        n = A.shape[0]
        if n < 2:
            return cls(
                n=n,
                G=A @ A.T if n else np.zeros((0, 0), dtype=int),
                pos_value=pos_value,
                neg_value=neg_value,
                count_pos=0,
                count_neg=0,
                count_out=0,
            )
        G = (A @ A.T).astype(int)
        iu = np.triu_indices(n, k=1)
        vals = G[iu]
        cp = int(np.sum(vals == pos_value))
        cn = int(np.sum(vals == neg_value))
        total = n * (n - 1) // 2
        co = total - cp - cn
        return cls(
            n=n, G=G,
            pos_value=pos_value, neg_value=neg_value,
            count_pos=cp, count_neg=cn, count_out=co,
        )

    # -- update -----------------------------------------------------------

    def deltas_for_flip(
        self, M_after: Sequence[Sequence[int]], i: int, c: int
    ) -> tuple[int, int, int]:
        """Return (d_pos, d_neg, d_out) for flipping entry (i, c) of M.

        ``M_after[i][c]`` is the *new* value; ``M_after[r][*]`` for r != i
        and ``M_after[i][k]`` for k != c are the unchanged old values.

        Costs O(n). Does NOT mutate the state.
        """
        n = self.n
        if n < 2:
            return 0, 0, 0
        new_val = int(M_after[i][c])
        delta_factor = 2 * new_val  # G_new[i, r] - G_old[i, r] for r != i
        Gi_old = self.G[i]
        d_pos = 0
        d_neg = 0
        for r in range(n):
            if r == i:
                continue
            old = int(Gi_old[r])
            new = old + delta_factor * int(M_after[r][c])
            old_pos = (old == self.pos_value)
            new_pos = (new == self.pos_value)
            old_neg = (old == self.neg_value)
            new_neg = (new == self.neg_value)
            d_pos += int(new_pos) - int(old_pos)
            d_neg += int(new_neg) - int(old_neg)
        # Each (r, i) with r != i is one unordered pair; the above loop visits
        # both (r > i) and (r < i) for r != i. The count over unordered pairs
        # changes by exactly the same amount (the pair appears once), so the
        # n-1 terms summed above ARE the change in count_pos / count_neg.
        d_out = -(d_pos + d_neg)
        return d_pos, d_neg, d_out

    def apply_after_flip(
        self, M_after: Sequence[Sequence[int]], i: int, c: int
    ) -> None:
        """Apply the deltas computed from a just-completed entry flip.

        Mutates ``self.G``, ``self.count_pos``, ``self.count_neg``,
        ``self.count_out``. Cost O(n).
        """
        n = self.n
        if n < 2:
            return
        new_val = int(M_after[i][c])
        delta_factor = 2 * new_val
        d_pos = 0
        d_neg = 0
        Gi = self.G[i]
        # Walk r != i, recomputing G[i, r] in-place.
        for r in range(n):
            if r == i:
                continue
            old = int(Gi[r])
            new = old + delta_factor * int(M_after[r][c])
            if new != old:
                old_pos = (old == self.pos_value)
                new_pos = (new == self.pos_value)
                old_neg = (old == self.neg_value)
                new_neg = (new == self.neg_value)
                d_pos += int(new_pos) - int(old_pos)
                d_neg += int(new_neg) - int(old_neg)
                Gi[r] = new
                self.G[r, i] = new
        self.count_pos += d_pos
        self.count_neg += d_neg
        self.count_out -= (d_pos + d_neg)


# ---------------------------------------------------------------------------
# SA driver
# ---------------------------------------------------------------------------


@dataclass
class Sa2ValAnnealResult:
    """Outcome of :func:`sa_2valued`."""

    matrix: Matrix
    abs_det: int
    iterations: int
    accepted: int
    walltime_seconds: float
    refreshes: int
    final_count_pos: int
    final_count_neg: int
    final_count_out: int
    feasible_ever: bool
    best_feasible_abs_det: int
    best_feasible_count_pos: int
    best_feasible_entries: list[list[int]] | None = None
    history: list[tuple[int, int, int, int]] = field(default_factory=list)
    # history entry: (step, abs_det, count_pos, count_out)


def sa_2valued(
    n: int,
    *,
    target_k_pos: int,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    mu_support: float = 1.0,
    lambda_count: float = 0.1,
    pos_value: int = 3,
    neg_value: int = -1,
    seed: int | None = None,
    initial: Matrix | None = None,
    refresh_every: int = 2000,
    log_every: int = 0,
    perturb_initial: int = 0,
    perturb_rng_seed: int | None = None,
) -> Sa2ValAnnealResult:
    """SA over

        E(M) = -log|det(M)|
             + mu_support * (count_out)^2
             + lambda_count * (count_pos - target_k_pos)^2

    where ``count_out`` is the number of unordered off-diagonal Gram pairs
    NOT in ``{neg_value, pos_value}``, and ``count_pos`` is the number equal
    to ``pos_value``.

    Per-flip cost: O(n) for the count updates + O(n^2) for the (det, adj)
    rank-1 update + O(1) for the energy. Periodic O(n^3) refreshes guard
    against logic drift.

    Parameters
    ----------
    n
        Matrix order.
    target_k_pos
        Target value for ``count_pos`` (number of unordered off-diagonal
        Gram pairs equal to ``pos_value``). At Orrick's H23: 45.
    iterations
        Number of MC steps.
    t_start, t_end
        Geometric temperature schedule on the energy.
    mu_support
        Penalty coefficient on ``count_out**2``. Large values strongly bias
        toward feasibility (``count_out == 0``).
    lambda_count
        Penalty coefficient on ``(count_pos - target_k_pos)**2``. Moderate
        values keep the search near the target split without dominating.
    pos_value, neg_value
        The two values defining the Gram off-diagonal support to enforce.
        Defaults: +3 and -1 (Orrick's H23 split).
    seed
        PRNG seed for SA acceptance and (if no initial provided) random init.
    initial
        Starting matrix; defaults to a random ±1 matrix.
    refresh_every
        Interval (in accepted flips) between full exact refreshes of
        ``(det, adj)``.
    log_every
        If positive, append ``(step, abs_det, count_pos, count_out)`` every
        that-many iterations.
    perturb_initial
        If > 0 AND ``initial`` is provided, flip this many uniformly random
        entries of ``initial`` before starting (escape from a fixed seed).
    perturb_rng_seed
        Seed for the perturbation RNG (independent of SA seed).
    """
    rng = random.Random(seed)

    if initial is None:
        mat = Matrix.random(n, rng)
    elif initial.n != n:
        raise ValueError(f"initial order {initial.n} != n={n}")
    else:
        mat = initial.copy()

    if perturb_initial > 0:
        prng = random.Random(perturb_rng_seed if perturb_rng_seed is not None else seed)
        for _ in range(perturb_initial):
            ii = prng.randrange(n)
            jj = prng.randrange(n)
            mat.entries[ii][jj] = -mat.entries[ii][jj]

    state = IncrementalState.from_matrix(mat.entries)
    tries = 0
    while state.det == 0 and tries < 16:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        tries += 1
    if state.det == 0:
        return Sa2ValAnnealResult(
            matrix=Matrix(n=n, entries=[row[:] for row in state.M]),
            abs_det=0, iterations=0, accepted=0, walltime_seconds=0.0,
            refreshes=0,
            final_count_pos=0, final_count_neg=0, final_count_out=0,
            feasible_ever=False,
            best_feasible_abs_det=0, best_feasible_count_pos=0,
            best_feasible_entries=None,
        )

    g2v = Gram2ValState.from_matrix(
        state.M, pos_value=pos_value, neg_value=neg_value
    )

    def energy_of(det_abs: int, c_pos: int, c_out: int) -> float:
        return (
            -math.log(det_abs)
            + mu_support * float(c_out) ** 2
            + lambda_count * float(c_pos - target_k_pos) ** 2
        )

    current_e = energy_of(abs(state.det), g2v.count_pos, g2v.count_out)
    best_entries = [row[:] for row in state.M]
    best_abs = abs(state.det)
    best_e = current_e
    feasible_ever = (g2v.count_out == 0)
    best_feasible_abs_det = abs(state.det) if feasible_ever else 0
    best_feasible_count_pos = g2v.count_pos if feasible_ever else 0
    best_feasible_entries = (
        [row[:] for row in state.M] if feasible_ever else None
    )

    accepted = 0
    refreshes = 0
    history: list[tuple[int, int, int, int]] = []
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

        # Compute (d_pos, d_neg, d_out) by simulating the flip on M (cheap O(n)).
        old_val = state.M[i][c]
        state.M[i][c] = -old_val
        d_pos, d_neg, d_out = g2v.deltas_for_flip(state.M, i, c)
        proposal_count_pos = g2v.count_pos + d_pos
        proposal_count_out = g2v.count_out + d_out
        proposal_e = energy_of(abs(new_det), proposal_count_pos, proposal_count_out)
        delta = proposal_e - current_e

        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            # Accept: commit (det, adj) AND (G, counts).
            state.M[i][c] = old_val  # apply_entry_flip expects OLD value in place
            apply_entry_flip(state, i, c)
            # state.M[i][c] is now the new value. Now update g2v with M_after.
            g2v.apply_after_flip(state.M, i, c)
            current_e = proposal_e
            accepted += 1
            if g2v.count_out == 0:
                feasible_ever = True
                if abs(state.det) > best_feasible_abs_det:
                    best_feasible_abs_det = abs(state.det)
                    best_feasible_count_pos = g2v.count_pos
                    best_feasible_entries = [row[:] for row in state.M]
            if abs(state.det) > best_abs:
                best_abs = abs(state.det)
                best_entries = [row[:] for row in state.M]
            if current_e < best_e:
                best_e = current_e
            if refresh_every > 0 and accepted % refresh_every == 0:
                refresh(state)
                refreshes += 1
                # Re-derive g2v from scratch in case of any drift.
                g2v_new = Gram2ValState.from_matrix(
                    state.M, pos_value=pos_value, neg_value=neg_value,
                )
                g2v.G = g2v_new.G
                g2v.count_pos = g2v_new.count_pos
                g2v.count_neg = g2v_new.count_neg
                g2v.count_out = g2v_new.count_out
                if state.det != 0:
                    current_e = energy_of(
                        abs(state.det), g2v.count_pos, g2v.count_out,
                    )
        else:
            # Reject: roll back M; counts/G untouched (we never wrote to g2v).
            state.M[i][c] = old_val

        if log_every and step % log_every == 0:
            history.append((step, abs(state.det), g2v.count_pos, g2v.count_out))

    walltime = time.monotonic() - start

    # Re-verify the bests with exact Bareiss (defensive).
    best_abs_det = abs(integer_determinant(best_entries))
    if best_feasible_entries is not None:
        bf = abs(integer_determinant(best_feasible_entries))
        # If the rank-1 path drifted, the recomputed exact value wins.
        best_feasible_abs_det = bf
    return Sa2ValAnnealResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        refreshes=refreshes,
        final_count_pos=g2v.count_pos,
        final_count_neg=g2v.count_neg,
        final_count_out=g2v.count_out,
        feasible_ever=feasible_ever,
        best_feasible_abs_det=best_feasible_abs_det,
        best_feasible_count_pos=best_feasible_count_pos,
        best_feasible_entries=best_feasible_entries,
        history=history,
    )
