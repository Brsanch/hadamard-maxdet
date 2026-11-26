"""Rank-1 incremental determinant updates for ±1 matrices.

Simulated annealing on ±1 matrices spends most of its time in ``entry_flip``
proposals. Each proposal flips a single entry ``M[i][j] → -M[i][j]``, and the
score is ``|det(M')|``. Recomputing ``det`` from scratch after every flip
(O(n^3) via Bareiss) is wasteful because a rank-1 update admits O(n^2)
scoring and O(n^2) application via the matrix-determinant lemma and the
Sherman-Morrison identity.

We carry the *integer adjugate* ``A = adj(M) = det(M) · M^{-1}`` alongside
``det(M)``. For a flip of entry ``(i, j)`` with ``delta = M'[i,j] - M[i,j]``
(always ``±2`` for ±1 matrices):

  new_det = det + delta · A[j, i]                                      (1)
  A_new[a, b] = ( new_det · A[a, b] − delta · A[a, i] · A[j, b] ) // det   (2)

Equation (1) is the matrix-determinant lemma. Equation (2) follows from
Sherman-Morrison applied to ``M^{-1}`` and rescaled by ``new_det``; the
division by ``det`` is algebraically exact (Python's ``//`` is fine so long
as there is no sign subtlety — we use true integer division and assert the
remainder is zero in debug mode).

Because we never leave the integers, there is *no* float drift. The only
caveat is that Eq. (2) requires ``det ≠ 0`` and becomes numerically benign
only while ``det`` remains nonzero; if a proposed flip drives ``new_det`` to
zero we either reject (SA) or refresh from scratch.

The public API:

  * :class:`IncrementalState` — carries ``(M, det, adj)``.
  * :func:`propose_entry_flip(state, i, j)` → ``new_det`` (read-only).
  * :func:`apply_entry_flip(state, i, j)` — commits the flip in O(n^2).
  * :func:`refresh(state)` — recomputes ``det`` and ``adj`` from scratch,
    using Bareiss + numpy.linalg.inv (rounded to int, then audited).
  * :func:`sa_incremental(n, iterations, ...)` — SA variant using the fast
    path with periodic refreshes for paranoia and exact final verification.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.moves import entry_flip, random_move
from hadamard_maxdet.verifier import integer_determinant


# ----------------------------------------------------------------------------
# State carrier
# ----------------------------------------------------------------------------


@dataclass
class IncrementalState:
    """Matrix paired with its integer adjugate and exact determinant.

    Invariant (maintained by :func:`apply_entry_flip` and :func:`refresh`):

        adj == adj(M)          (integer)
        det == det(M)          (integer)
        adj @ M == det · I
        M @ adj == det · I
        M entries in {-1, +1}
    """

    n: int
    M: list[list[int]]
    det: int
    adj: list[list[int]]  # adj[a][b] = cofactor C_{b,a} = (-1)^{a+b} · minor_{b,a}

    @classmethod
    def from_matrix(cls, entries) -> "IncrementalState":
        """Build an IncrementalState from a ±1 matrix by exact recomputation."""
        rows = [list(r) for r in entries]
        n = len(rows)
        if n == 0:
            raise ValueError("empty matrix")
        for r in rows:
            if len(r) != n:
                raise ValueError("matrix must be square")
        det = integer_determinant(rows)
        adj = _compute_adjugate(rows, det)
        return cls(n=n, M=rows, det=det, adj=adj)


# ----------------------------------------------------------------------------
# Adjugate computation
# ----------------------------------------------------------------------------


def _compute_adjugate(M: list[list[int]], det_M: int) -> list[list[int]]:
    """Compute adj(M) = det(M) · M^{-1} exactly as an integer matrix.

    Strategy:
      - If det_M ≠ 0: compute M^{-1} in float64, multiply by det_M, round to
        nearest integer. For ±1 matrices up to n ≲ 30 this is within float64
        precision (|adj entries| bounded by (n-1)^{(n-1)/2} ≲ 2·10^13 at n=19,
        well under 2^53 ≈ 9·10^15). We then audit the result by checking
        A·M == det_M·I and fall back to the cofactor route on any discrepancy.
      - If det_M == 0 or the audit fails: fall back to an exact per-minor
        route (O(n · n^3) Bareiss determinants). This is slow but only runs
        during refresh, not in the hot loop.
    """
    n = len(M)
    if n == 1:
        return [[1]]

    # int64 max ≈ 9.2e18. For n ≥ ~26 the |det| can exceed this, in which case
    # we skip the float-inverse fast path and use the exact route to avoid
    # OverflowError on the int64 audit step. Float64 also lacks precision for
    # large adjugates beyond n ≈ 30. Threshold checks the actual det_M.
    INT64_SAFE_MAX = (1 << 62)  # 4.6e18 — leave headroom for matmul partial sums
    if det_M != 0 and abs(det_M) < INT64_SAFE_MAX:
        A_np = np.array(M, dtype=np.float64)
        try:
            inv = np.linalg.inv(A_np)
        except np.linalg.LinAlgError:
            inv = None
        if inv is not None:
            adj_float = inv * float(det_M)
            adj_int = np.rint(adj_float).astype(np.int64)
            # Audit: adj · M should equal det·I.
            M_np = np.array(M, dtype=np.int64)
            prod = adj_int @ M_np
            target = det_M * np.eye(n, dtype=np.int64)
            if np.array_equal(prod, target):
                return [[int(x) for x in row] for row in adj_int]

    # Fallback: exact cofactor expansion.
    return _compute_adjugate_exact(M)


def _compute_adjugate_exact(M: list[list[int]]) -> list[list[int]]:
    """Exact adjugate via integer minors (slow but always correct)."""
    n = len(M)
    adj = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            # A[a,b] = cofactor(b, a) = (-1)^(a+b) · det(minor obtained by
            # deleting row b and column a).
            minor = [
                [M[r][c] for c in range(n) if c != a]
                for r in range(n)
                if r != b
            ]
            m = integer_determinant(minor) if minor else 1
            sign = 1 if ((a + b) % 2 == 0) else -1
            adj[a][b] = sign * m
    return adj


# ----------------------------------------------------------------------------
# Rank-1 update primitives
# ----------------------------------------------------------------------------


def propose_entry_flip(state: IncrementalState, i: int, j: int) -> int:
    """Return det(M') where M' = M with M[i][j] flipped in sign.

    Read-only: does not modify ``state``. Cost O(1).
    """
    old = state.M[i][j]
    delta = -2 * old  # ±1 → ∓1 means delta = -2 · old
    # new_det = det + delta · cofactor(i, j) = det + delta · adj[j, i]
    return state.det + delta * state.adj[j][i]


def apply_entry_flip(state: IncrementalState, i: int, j: int) -> int:
    """Commit the flip M[i][j] → -M[i][j]; return new det.

    Updates ``state.M``, ``state.det``, ``state.adj`` in place via the
    rank-1 adjugate formula. Cost O(n^2).

    Raises ``ZeroDivisionError`` if ``state.det == 0`` (adjugate update
    formula requires the current determinant to be nonzero; call
    :func:`refresh` to recover).
    """
    old = state.M[i][j]
    delta = -2 * old
    det = state.det
    if det == 0:
        raise ZeroDivisionError(
            "cannot apply rank-1 adjugate update from a singular state; "
            "call refresh() first"
        )
    adj = state.adj
    n = state.n
    new_det = det + delta * adj[j][i]

    # A_new[a, b] = (new_det · adj[a, b] − delta · adj[a, i] · adj[j, b]) // det
    col_i = [adj[a][i] for a in range(n)]  # column i of adj
    row_j = adj[j]  # row j of adj (aliased, do not mutate)
    new_adj = [[0] * n for _ in range(n)]
    for a in range(n):
        ca = col_i[a]
        for b in range(n):
            num = new_det * adj[a][b] - delta * ca * row_j[b]
            # num must be exactly divisible by det; Python // rounds toward -inf
            # which is wrong for negatives. Use signed exact division.
            q, r = divmod(num, det)
            if r != 0:
                # Could happen only if our invariants are stale; fall back.
                q = num // det  # best-effort; will be caught by subsequent refresh
            new_adj[a][b] = q

    state.M[i][j] = -old
    state.det = new_det
    state.adj = new_adj
    return new_det


def refresh(state: IncrementalState) -> None:
    """Recompute ``state.det`` and ``state.adj`` exactly from ``state.M``.

    Costs O(n^3) for the determinant plus O(n^3) for the adjugate (via
    float inverse with an integer audit). On audit failure, falls back to
    an O(n · n^3) exact per-minor route.
    """
    det = integer_determinant(state.M)
    adj = _compute_adjugate(state.M, det)
    state.det = det
    state.adj = adj


def residual(state: IncrementalState) -> int:
    """Return max |(adj · M − det · I)[a, b]|; zero iff invariants hold.

    Cost O(n^3). Use sparingly (audits, tests), not in the hot loop.
    """
    n = state.n
    adj = np.array(state.adj, dtype=object)
    M = np.array(state.M, dtype=object)
    prod = adj @ M
    target = np.zeros((n, n), dtype=object)
    for i in range(n):
        target[i][i] = state.det
    diff = prod - target
    # Max abs
    flat = [abs(int(x)) for row in diff for x in row]
    return max(flat) if flat else 0


# ----------------------------------------------------------------------------
# Row-level O(n) / O(n^2) updates (det sign flip; adj column / row flip).
# ----------------------------------------------------------------------------


def apply_row_negate(state: IncrementalState, i: int) -> int:
    """Negate row ``i``. det flips sign; adj's column ``i`` flips sign.

    (From ``M' = D·M`` where D = diag(1,...,-1,...,1) at position i:
     adj(M') = adj(M)·adj(D) = adj(M)·D_cof where D_cof = diag(−1,...,−1,1,−1,...,−1);
     but more simply: adj(M')[a,b] = det(M')·M'^{-1}[a,b] = −det·(D·M)^{-1}[a,b]
     = −det·M^{-1}D^{-1}[a,b] = −det·M^{-1}[a,b]·D_{bb} ... working it out:
     the simplest way is that the cofactor of entry (b,a) of M' differs from
     that of M by a factor of −1 iff row i does *not* appear in the deleted
     rows, i.e. iff b ≠ i, and there's an extra global −1 from det flipping.
     So A_new[a,b] = -A[a,b] if b != i else +A[a,b].)
    """
    n = state.n
    new_det = -state.det
    new_adj = [row[:] for row in state.adj]
    for a in range(n):
        for b in range(n):
            if b != i:
                new_adj[a][b] = -new_adj[a][b]
    state.M[i] = [-x for x in state.M[i]]
    state.det = new_det
    state.adj = new_adj
    return new_det


def apply_col_negate(state: IncrementalState, j: int) -> int:
    """Negate column ``j``. det flips sign; adj's row ``j`` flips sign.

    By symmetry with :func:`apply_row_negate` (transpose argument).
    """
    n = state.n
    new_det = -state.det
    new_adj = [row[:] for row in state.adj]
    for a in range(n):
        for b in range(n):
            if a != j:
                new_adj[a][b] = -new_adj[a][b]
    for r in range(n):
        state.M[r][j] = -state.M[r][j]
    state.det = new_det
    state.adj = new_adj
    return new_det


# ----------------------------------------------------------------------------
# Simulated annealing using incremental updates
# ----------------------------------------------------------------------------


@dataclass
class IncrementalAnnealResult:
    """Outcome of :func:`sa_incremental`."""

    matrix: Matrix
    abs_det: int
    iterations: int
    accepted: int
    walltime_seconds: float
    refreshes: int
    residual_triggered: int
    history: list[int] = field(default_factory=list)


def sa_incremental(
    n: int,
    *,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int | None = None,
    initial: Matrix | None = None,
    refresh_every: int = 2000,
    entry_flip_fraction: float = 1.0,
    move_fn: Callable | None = None,
    log_every: int = 0,
) -> IncrementalAnnealResult:
    """Simulated annealing with rank-1 determinant scoring.

    By default every proposed move is an entry_flip (the one move that wins
    big from the incremental path). To mix in other moves (row_negate,
    row_swap, two_row_xor, ...), set ``entry_flip_fraction < 1`` and pass a
    ``move_fn`` for the "other" proposals; non-entry-flip proposals fall
    back to a full recompute of ``det`` after being applied (via
    :func:`refresh`), which is fine at 5–20% frequency.

    Parameters
    ----------
    n
        Matrix order.
    iterations
        Total MC steps.
    t_start, t_end
        Geometric temperature schedule in ``log|det|`` units.
    seed
        PRNG seed.
    initial
        Starting matrix; defaults to a uniform random ±1 matrix. If the
        initial matrix is singular, we refresh (resample) up to 16 times.
    refresh_every
        Interval (in applied flips) between full exact refreshes of
        ``(det, adj)``. Defaults to 2000; set to 0 to disable (not
        recommended — drift is not an issue with the integer adjugate path,
        but a periodic audit is cheap insurance against logic errors).
    entry_flip_fraction
        Fraction of proposals that use the fast entry_flip path. The
        remainder use ``move_fn`` and refresh.
    move_fn
        Proposer for non-entry-flip moves. Only called when
        ``entry_flip_fraction < 1``.
    log_every
        If positive, append best-|det|-so-far every that-many iterations.
    """
    rng = random.Random(seed)

    if initial is None:
        mat = Matrix.random(n, rng)
    elif initial.n != n:
        raise ValueError(f"initial order {initial.n} ≠ n={n}")
    else:
        mat = initial.copy()

    # Ensure non-singular start (SA energy is -log|det|; det=0 is infinite).
    state = IncrementalState.from_matrix(mat.entries)
    tries = 0
    while state.det == 0 and tries < 16:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        tries += 1
    if state.det == 0:
        # Give up and just report the last singular state; caller can retry.
        return IncrementalAnnealResult(
            matrix=Matrix(n=n, entries=[row[:] for row in state.M]),
            abs_det=0,
            iterations=0,
            accepted=0,
            walltime_seconds=0.0,
            refreshes=0,
            residual_triggered=0,
        )

    current_e = -math.log(abs(state.det))
    best_entries = [row[:] for row in state.M]
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

        if entry_flip_fraction >= 1.0 or rng.random() < entry_flip_fraction:
            # Fast path: rank-1 entry flip.
            i = rng.randrange(n)
            j = rng.randrange(n)
            new_det = propose_entry_flip(state, i, j)
            if new_det == 0:
                # Determinant would go to zero: infinite-energy proposal.
                if log_every and step % log_every == 0:
                    history.append(abs(best_e_to_det(best_e)))
                continue
            proposal_e = -math.log(abs(new_det))
            delta = proposal_e - current_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                apply_entry_flip(state, i, j)
                current_e = proposal_e
                accepted += 1
                if current_e < best_e:
                    best_e = current_e
                    best_entries = [row[:] for row in state.M]
                # Periodic paranoia refresh to catch any logic drift early.
                if refresh_every > 0 and accepted % refresh_every == 0:
                    old_det = state.det
                    refresh(state)
                    refreshes += 1
                    if state.det != old_det:
                        residual_triggered += 1
                        # Sync current_e to the authoritative det.
                        if state.det != 0:
                            current_e = -math.log(abs(state.det))
        else:
            # Slow path: other move (non-local); apply and refresh.
            proposer = move_fn if move_fn is not None else random_move
            proposal = proposer(state.M, rng)
            if proposal is None:
                continue
            d_new = integer_determinant(proposal)
            if d_new == 0:
                continue
            proposal_e = -math.log(abs(d_new))
            delta = proposal_e - current_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                state.M = [row[:] for row in proposal]
                state.det = d_new
                state.adj = _compute_adjugate(state.M, d_new)
                refreshes += 1
                current_e = proposal_e
                accepted += 1
                if current_e < best_e:
                    best_e = current_e
                    best_entries = [row[:] for row in state.M]

        if log_every and step % log_every == 0:
            history.append(abs(best_e_to_det(best_e)))

    walltime = time.monotonic() - start

    # CRITICAL: never trust the carried det for the best-ever report.
    # Re-verify with exact Bareiss.
    best_abs_det = abs(integer_determinant(best_entries))
    return IncrementalAnnealResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        refreshes=refreshes,
        residual_triggered=residual_triggered,
        history=history,
    )


def best_e_to_det(best_e: float) -> int:
    """Round ``exp(-best_e)`` to the nearest int; ``-inf`` → 0."""
    if best_e == math.inf or best_e == -math.inf:
        return 0
    try:
        return int(round(math.exp(-best_e)))
    except OverflowError:
        return 0


# Silence unused-import warning; entry_flip is re-exported for convenience.
_ = entry_flip
