"""Row-weight parity feasibility for planted block-Gram targets.

The generalized satellite view (`generalized_ehlich.py`) enumerates closed-form
predicted ``|det|`` values over (intra, cross, partition) triples. A high
predicted value does not imply a ±1 realiser exists: the planted Gram must
be the Gram of *some* ±1 matrix, which imposes much stronger constraints
than positive-definiteness + Hadamard-bound + the n-mod-2 parity already
enforced in `enumerate_ab_grid`.

This module enforces the **row-weight parity constraint**.

Theory
------

For ±1 vectors v_i, v_j ∈ {±1}^n with weights w_i, w_j (count of +1's),

    <v_i, v_j> = 4 |A_i ∩ A_j| + n - 2 w_i - 2 w_j,

where A_i = {positions where v_i = +1}. So <v_i, v_j> = c requires

    2 |A_i ∩ A_j| = (c - n)/2 + w_i + w_j.

For an integer solution: (c - n) must be even, i.e. c ≡ n (mod 2).
Given that, |A_i ∩ A_j| is an integer iff (c - n)/2 + w_i + w_j is even, i.e.

    w_i + w_j ≡ (n - c)/2  (mod 2).

So the row-weight parity is forced by (n, c). Define

    P(c) := ((n - c) // 2) mod 2   ∈ {0, 1}.

Then for any pair (i, j) with required Gram entry c, we need
``(w_i + w_j) mod 2 = P(c)``.

* P(c) = 0: w_i and w_j have the **same** parity.
* P(c) = 1: w_i and w_j have **different** parities.

For a block partition (S_1, ..., S_4) with intra-block target ``a`` and
cross-block target ``b``, every row gets a parity ``p_i ∈ {0, 1}`` (parity
of its weight). The constraints on parity-assignment are:

* Intra-block (size ≥ 2 only): every pair (i, j) in the same block satisfies
  p_i ⊕ p_j = P(a).  [⊕ = XOR]
* Cross-block: every pair (i, j) in different blocks satisfies
  p_i ⊕ p_j = P(b).

Tractable cases by (P(a), P(b)):

* **(0, 0)**: all pairs same parity → all rows have the same parity. Always
  feasible (2 assignments: all even, or all odd).
* **(0, 1)**: same parity within each block, different parity across blocks.
  Each block is monochromatic in parity; across-block constraint forces a
  proper 2-coloring of K_k where k = number of blocks of size ≥ 1 with
  intra-pair constraints. With singletons (size 1), the singleton has no
  intra-constraint but still cross-pair constraints to all other blocks.
  K_k bipartite ⇔ k ≤ 2. So feasible only when ≤ 2 distinct
  parity-classes are needed. With 4 blocks all having cross constraints,
  we need K_4 to be 2-colorable with all edges crossing colors —
  impossible.
* **(1, 0)**: different parity within block (must split each block into
  bipartite halves), same parity across blocks. Within a block of size s,
  bipartite-graph-coloring K_s requires s ≤ 2.
* **(1, 1)**: every pair has p_i ⊕ p_j = 1 — every pair has different
  parity, which is impossible for 3 or more rows.

These rules are necessary; we also enumerate valid parity-assignments
explicitly to give a concrete count and an explicit witness when feasible.

Note
----
Parity feasibility is **necessary, not sufficient.** Even when a parity
assignment exists, the inner-product equations involving |A_i ∩ A_j| may
have no consistent set-system solution (Fisher-type or coding-bound
obstructions can rule out the full design). This module therefore returns
a parity *upper bound* on feasibility: ``infeasible`` here means provably
no ±1 matrix exists; ``feasible`` here means parity does not rule it out
and a construction attempt is warranted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from hadamard_maxdet.generalized_ehlich import (
    DEFAULT_AB_VALUES,
    enumerate_partitions,
    predicted_block_det,
    predicted_block_ratio,
)
from hadamard_maxdet.verifier import upper_bound


__all__ = [
    "required_pair_parity",
    "ParityCheck",
    "parity_constraint",
    "FeasibleTarget",
    "enumerate_feasible_targets",
]


def required_pair_parity(n: int, c: int) -> int | None:
    """Return required ``(w_i + w_j) mod 2`` for an inner product of ``c``.

    Returns ``None`` if ``c`` violates the basic parity ``c ≡ n (mod 2)``,
    which means no ±1 pair has that inner product at all.
    """
    if (c - n) % 2 != 0:
        return None
    return ((n - c) // 2) % 2


@dataclass(frozen=True)
class ParityCheck:
    """Result of a parity-feasibility check on a (partition, intra, cross) triple.

    Fields
    ------
    feasible
        True iff at least one parity-assignment ``(p_1, ..., p_n)`` exists
        satisfying every intra- and cross-block parity constraint.
    base_parity_ok
        True iff both ``intra ≡ n (mod 2)`` and ``cross ≡ n (mod 2)``. When
        False, no ±1 matrix can have these Gram entries irrespective of
        weights, so the target is doubly infeasible.
    intra_required
        ``required_pair_parity(n, intra)`` — the required ``(w_i+w_j) mod 2``
        for intra-block pairs (None if base_parity_ok is False).
    cross_required
        Same, for cross-block pairs.
    n_assignments
        Number of distinct ``(p_1, ..., p_n)`` ∈ {0,1}^n satisfying *all*
        constraints. ``0`` ⇔ infeasible.
    block_parity_pattern
        A representative valid assignment of *block-level* parities
        ``(P(S_1), P(S_2), P(S_3), P(S_4))`` if the intra-constraint forces
        each block to be monochromatic (intra_required == 0), else None.
        Useful for downstream construction.
    reason
        One-line explanation of feasibility / infeasibility.
    """

    feasible: bool
    base_parity_ok: bool
    intra_required: int | None
    cross_required: int | None
    n_assignments: int
    block_parity_pattern: tuple[int, ...] | None
    reason: str

    def to_json(self) -> dict:
        return {
            "feasible": self.feasible,
            "base_parity_ok": self.base_parity_ok,
            "intra_required": self.intra_required,
            "cross_required": self.cross_required,
            "n_assignments": self.n_assignments,
            "block_parity_pattern": (
                list(self.block_parity_pattern)
                if self.block_parity_pattern is not None
                else None
            ),
            "reason": self.reason,
        }


def _enumerate_within_block_parities(size: int, intra_req: int | None) -> list[tuple[int, ...]]:
    """Return all per-row parity assignments inside a single block of ``size``.

    With ``intra_req`` the required ``(p_i + p_j) mod 2`` for every intra-pair:
    * size ≤ 1: no intra-constraint; both ``(0,)`` and ``(1,)`` (or empty)
      are valid.
    * size ≥ 2 and intra_req == 0: all rows same parity → 2 assignments.
    * size ≥ 2 and intra_req == 1: every pair differs → bipartite K_size,
      possible iff size ≤ 2 → 2 assignments (01) and (10) when size == 2;
      0 assignments otherwise.
    * intra_req is None: no constraint admissible; treat as 2^size free
      (caller should set this only when base parity already failed).
    """
    if size == 0:
        return [()]
    if size == 1:
        return [(0,), (1,)]
    if intra_req is None:
        return [tuple(p) for p in product((0, 1), repeat=size)]
    if intra_req == 0:
        return [(0,) * size, (1,) * size]
    # intra_req == 1: bipartite K_size, only size 2 works.
    if size == 2:
        return [(0, 1), (1, 0)]
    return []


def parity_constraint(
    n: int,
    partition: Sequence[int],
    intra: int,
    cross: int,
) -> ParityCheck:
    """Check row-weight parity feasibility for a planted (intra, cross) Gram.

    Parameters
    ----------
    n
        Matrix order.
    partition
        Block sizes ``(s_1, ..., s_k)`` summing to ``n``. ``k`` may be any
        positive integer; the routine generalises beyond k=4.
    intra
        Required intra-block off-diagonal Gram value.
    cross
        Required cross-block off-diagonal Gram value.

    Returns
    -------
    ParityCheck
        See dataclass for field semantics.
    """
    if sum(partition) != n:
        raise ValueError(
            f"partition {tuple(partition)} sums to {sum(partition)}, expected {n}"
        )

    intra_req = required_pair_parity(n, intra)
    cross_req = required_pair_parity(n, cross)
    base_ok = intra_req is not None and cross_req is not None
    if not base_ok:
        return ParityCheck(
            feasible=False,
            base_parity_ok=False,
            intra_required=intra_req,
            cross_required=cross_req,
            n_assignments=0,
            block_parity_pattern=None,
            reason=(
                f"base parity n={n} mod 2 = {n%2} requires intra ≡ cross ≡ "
                f"{n%2} (mod 2); got intra={intra}, cross={cross}"
            ),
        )

    # Enumerate per-block parity assignments, then combine block-by-block,
    # filtering by the cross-block constraint at each step. The cross-block
    # constraint says that for any i in block A and j in block B (A ≠ B),
    # ``(p_i + p_j) mod 2 == cross_req``. Two cases:
    #
    # cross_req == 0: every cross-pair has same parity. So **every row in
    #   the matrix has the same parity** (the cross-constraint links all
    #   blocks of size ≥ 1, and parity is transitive). Only assignments
    #   that are constant across all rows survive.
    #
    # cross_req == 1: every cross-pair has different parity. So if row i in
    #   block A has parity p_i, every row in every other block must have
    #   parity 1 - p_i. This forces each block to be monochromatic in
    #   parity (a single block can't simultaneously be "opposite to row i"
    #   and "opposite to row j" if i, j are in different blocks unless those
    #   rows agree). Fix block A's parity to p; then every other block's
    #   parity is 1 - p. So at most 2 valid assignments at the block level
    #   (p ∈ {0, 1}), and each block must internally support a monochromatic
    #   parity (intra_req == 0 or size ≤ 1 or size == 2 with intra_req == 1).

    per_block_options: list[list[tuple[int, ...]]] = []
    for s in partition:
        per_block_options.append(_enumerate_within_block_parities(s, intra_req))
        if not per_block_options[-1]:
            return ParityCheck(
                feasible=False,
                base_parity_ok=True,
                intra_required=intra_req,
                cross_required=cross_req,
                n_assignments=0,
                block_parity_pattern=None,
                reason=(
                    f"intra constraint (P(intra)={intra_req}) requires bipartite "
                    f"parity within a size-{s} block; only sizes ≤ 2 admit this"
                ),
            )

    # Compose across blocks, filtering cross-pair parity.
    n_assignments = 0
    sample_block_pattern: tuple[int, ...] | None = None
    sizes = tuple(partition)

    # Helper: compatibility of block-i assignment a_i with block-j assignment a_j
    def cross_ok(a_i: tuple[int, ...], a_j: tuple[int, ...]) -> bool:
        for pi in a_i:
            for pj in a_j:
                if (pi + pj) % 2 != cross_req:
                    return False
        return True

    # Brute-force enumerate Cartesian product. With min_block_size = 1 in the
    # caller, blocks can be size 1 (2 options) or size 2 (≤ 2 options) or
    # size ≥ 2 with intra_req == 0 (2 options). For typical 4-block partitions
    # this is at most 2^4 = 16 combinations; very cheap.
    for combo in product(*per_block_options):
        ok = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if not cross_ok(combo[i], combo[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            n_assignments += 1
            if sample_block_pattern is None:
                # Record block-level parity pattern (parity of row 0 in each
                # block, which when intra_req == 0 fully determines the block).
                sample_block_pattern = tuple(c[0] for c in combo)

    feasible = n_assignments > 0
    if feasible:
        reason = (
            f"{n_assignments} parity assignment(s) survive intra "
            f"(P(intra)={intra_req}) and cross (P(cross)={cross_req}) constraints"
        )
    else:
        reason = (
            f"no parity assignment survives; intra/cross constraints "
            f"(P(intra)={intra_req}, P(cross)={cross_req}) jointly infeasible "
            f"on partition {tuple(partition)}"
        )

    return ParityCheck(
        feasible=feasible,
        base_parity_ok=True,
        intra_required=intra_req,
        cross_required=cross_req,
        n_assignments=n_assignments,
        block_parity_pattern=sample_block_pattern if feasible else None,
        reason=reason,
    )


@dataclass(frozen=True)
class FeasibleTarget:
    """A (n, intra, cross, partition) triple that survives parity feasibility."""

    n: int
    intra: int
    cross: int
    partition: tuple[int, int, int, int]
    predicted_abs_det: float
    predicted_ratio: float
    parity: ParityCheck

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "intra": self.intra,
            "cross": self.cross,
            "partition": list(self.partition),
            "predicted_abs_det": self.predicted_abs_det,
            "predicted_ratio": self.predicted_ratio,
            "parity": self.parity.to_json(),
        }


def enumerate_feasible_targets(
    n: int,
    *,
    max_predicted_ratio_floor: float = 0.71,
    ab_values: Sequence[int] = DEFAULT_AB_VALUES,
    require_distinct_ab: bool = True,
    min_block_size: int = 1,
) -> list[FeasibleTarget]:
    """Enumerate (a, b, partition) triples surviving parity at ``n``.

    Filters:
    * Predicted ratio (sqrt(det G_block) / upper_bound(n)) > floor.
    * (a, b) parity agrees with n mod 2 (basic parity at the entry level).
    * Row-weight parity has at least one valid assignment (this module's
      contribution).

    Returns
    -------
    list[FeasibleTarget]
        Sorted by predicted ratio descending.
    """
    base = float(upper_bound(n))
    n_par = n % 2
    partitions = enumerate_partitions(n, min_block_size=min_block_size)
    out: list[FeasibleTarget] = []
    for a in ab_values:
        if (a % 2) != n_par:
            continue
        for b in ab_values:
            if require_distinct_ab and a == b:
                continue
            if (b % 2) != n_par:
                continue
            for p in partitions:
                pred = predicted_block_det(n, p, a, b)
                if pred <= 0.0:
                    continue
                ratio = pred / base if base > 0 else 0.0
                if ratio <= max_predicted_ratio_floor:
                    continue
                check = parity_constraint(n, p, a, b)
                if not check.feasible:
                    continue
                out.append(
                    FeasibleTarget(
                        n=n,
                        intra=int(a),
                        cross=int(b),
                        partition=p,
                        predicted_abs_det=pred,
                        predicted_ratio=ratio,
                        parity=check,
                    )
                )
    out.sort(key=lambda t: -t.predicted_ratio)
    return out


def construct_from_partition_general(
    n: int,
    partition: Sequence[int],
    intra: int,
    cross: int,
    *,
    iterations: int = 200_000,
    seed: int | None = None,
    t_start: float = 4.0,
    t_end: float = 0.05,
    wall_time_budget: float | None = None,
):
    """Construction attempt for a generalised (intra, cross, partition) target.

    Generalises :func:`hadamard_maxdet.orrick_hunt.construct_from_partition`
    to arbitrary integer ``intra`` / ``cross`` Gram values (not just the
    ``(+3, -1)`` Ehlich/Wojtas regime). SA on ``±1 → ||G(M) - G_target||_F^2``.

    Returns the same :class:`hadamard_maxdet.orrick_hunt.ConstructionResult`
    type for downstream uniformity.
    """
    # Local import to avoid circular dependency.
    from hadamard_maxdet.orrick_hunt import (
        ConstructionResult,
        EhlichPartition,
        construct_from_partition,
    )

    sizes_sorted = tuple(sorted(partition))
    if len(sizes_sorted) != 4:
        raise ValueError(
            "construct_from_partition_general currently requires a 4-part partition"
        )
    pred = predicted_block_det(n, sizes_sorted, intra, cross)
    ep = EhlichPartition(
        sizes=sizes_sorted,
        intra_value=int(intra),
        cross_value=int(cross),
        n=n,
        predicted_abs_det=pred,
    )
    return construct_from_partition(
        ep,
        iterations=iterations,
        seed=seed,
        t_start=t_start,
        t_end=t_end,
        wall_time_budget=wall_time_budget,
    )


# ---------------------------------------------------------------------------
# Algebraic Gram feasibility: perfect-square det, Hasse-Minkowski, and
# rigorous block-Gram enumeration for n ≡ 1 (mod 4)
# ---------------------------------------------------------------------------


def hilbert_symbol(a: int, b: int, p: int) -> int:
    """Compute the Hilbert symbol ``(a, b)_p ∈ {-1, +1}`` for nonzero integers
    ``a, b`` and prime ``p``.

    Uses the standard p-adic valuation reduction and Legendre symbol for odd
    primes, and the mod-8 residue formula for ``p = 2``.
    """
    from sympy.functions.combinatorial.numbers import legendre_symbol

    if a == 0 or b == 0:
        raise ValueError("hilbert_symbol requires nonzero a and b")

    def _sqfree_padic(x: int, prime: int) -> tuple[int, int]:
        v = 0
        while x % prime == 0:
            x //= prime
            v += 1
        return v % 2, x

    va, ua = _sqfree_padic(a, p)
    vb, ub = _sqfree_padic(b, p)
    if p == 2:
        e = ((ua - 1) // 2) * ((ub - 1) // 2)
        if va:
            e += (ub * ub - 1) // 8
        if vb:
            e += (ua * ua - 1) // 8
        return -1 if (e % 2) else 1

    s = -1 if (va * vb * ((p - 1) // 2)) % 2 else 1
    if vb:
        s *= int(legendre_symbol(ua % p, p))
    if va:
        s *= int(legendre_symbol(ub % p, p))
    return s


@dataclass(frozen=True)
class HasseResult:
    """Result of exact rational congruence (Hasse-Minkowski) check for a Gram matrix."""

    psd: bool
    det_is_square: bool
    abs_det: int | None
    hasse_ok: bool
    bad_primes: dict[int, int]


def hasse_minkowski_check(G: Sequence[Sequence[int]]) -> HasseResult:
    """Check whether a symmetric integer matrix ``G`` is rationally congruent
    to the identity matrix ``I_n`` (necessary for ``G = M M^T`` with ``M ∈ Z^{n×n}``).

    Performs exact rational ``LDL^T`` diagonalization over ``Q``, verifies:
      1. Positive-definiteness (all pivots ``d_k > 0``).
      2. ``det(G) = ∏ d_k`` is an exact perfect square ``d^2``.
      3. For every prime ``p | 2 · det(G)``, the Hasse invariant
         ``c_p(G) = ∏_{i < j} (d_i, d_j)_p == +1``.
    """
    import math
    import sympy as sp

    n = len(G)
    M = [[sp.Rational(int(G[i][j])) for j in range(n)] for i in range(n)]
    D: list[sp.Rational] = []
    for k in range(n):
        piv = M[k][k]
        if piv <= 0:
            return HasseResult(False, False, None, False, {})
        D.append(piv)
        for i in range(k + 1, n):
            f = M[i][k] / piv
            if f != 0:
                for j in range(k, n):
                    M[i][j] -= f * M[k][j]

    detG = sp.Integer(1)
    for d in D:
        detG *= d
    detG_int = int(detG)
    root = math.isqrt(detG_int)
    if root * root != detG_int:
        return HasseResult(True, False, None, False, {})

    ints = [int(d.p * d.q) for d in D]
    primes = sorted({2} | set(sp.factorint(root).keys()))
    bad: dict[int, int] = {}
    for p in primes:
        prod = 1
        for i in range(n):
            for j in range(i + 1, n):
                prod *= hilbert_symbol(ints[i], ints[j], p)
        if prod == -1:
            bad[p] = -1

    return HasseResult(True, True, root, len(bad) == 0, bad)


def build_r1_block_gram(n: int, blocks: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Build the ``n × n`` Gram matrix ``G = (n - 1)I + J + 4E`` for ``n ≡ 1 (mod 4)``
    where ``E`` is a disjoint union of uniform-weight cliques ``(size, weight)``.
    """
    G = [[1] * n for _ in range(n)]
    for i in range(n):
        G[i][i] = n
    idx = 0
    for s, w in blocks:
        val = 1 + 4 * w
        for a in range(idx, idx + s):
            for b in range(idx, idx + s):
                if a != b:
                    G[a][b] = val
        idx += s
    if idx > n:
        raise ValueError(f"Block sizes sum to {idx} > n={n}")
    return G


@dataclass(frozen=True)
class BlockGramCandidate:
    """A shifted-block-diagonal ``n ≡ 1 (mod 4)`` Gram candidate surviving exact square-det check."""

    n: int
    blocks: tuple[tuple[int, int], ...]
    abs_det: int
    hasse_ok: bool
    bad_primes: dict[int, int]


def enumerate_r1_block_grams(
    n: int,
    min_abs_det: int,
    *,
    weights: tuple[int, ...] = (-1, 1),
    return_above_count: bool = False,
) -> tuple[int, list[BlockGramCandidate]] | tuple[int, int, list[BlockGramCandidate]]:
    """Exhaustively enumerate ALL Gram matrices ``G = (n-1)I + J + 4E`` for
    ``n ≡ 1 (mod 4)`` whose shifted matrix ``G - J = (n-1)I + 4E`` is
    block-diagonal with uniform-weight cliques ``(s, w)``, ``w ∈ weights``
    (and remaining isolated vertices) and ``A = ((n-1)/4)I + E ≻ 0``, whose
    determinant is an exact perfect square ``d^2 >= min_abs_det^2``.

    Uses exact arbitrary-precision integer branch-and-bound pruning (zero floating-point
    arithmetic): every valid block ``(s, w)`` satisfies ``det(B) <= m^s`` by AM-GM where
    ``m = (n-1)/4``, and ``q_B / s = 1/lam_1 - 1/m <= max_qv``. Scaling by
    ``Q = lcm(m, {lam_1})`` turns every determinant and upper-bound test into an exact
    integer comparison.

    Returns ``(visited_nodes, candidates)`` by default, or
    ``(visited_nodes, above_threshold_count, candidates)`` when
    ``return_above_count=True``, sorted by descending ``abs_det``.
    """
    import math

    if n % 4 != 1:
        raise ValueError(f"enumerate_r1_block_grams requires n ≡ 1 (mod 4); got {n}")
    m = (n - 1) // 4
    # G = 4A + J where A = m I + E. Base A_0 = m I has base_q = 4 + n/m = (4m + n)/m.
    raw_btypes: list[tuple[int, int, int, int]] = []
    lam1s: list[int] = [m]
    for w in weights:
        if w == 0:
            continue
        if w < 0:
            # A > 0 requires m + (s - 1)*w > 0 => s - 1 < m / |w|
            max_s = min(n, (m - 1) // (-w) + 1)
        else:
            max_s = n
        for s in range(2, max_s + 1):
            lam_perp = m - w
            lam_1 = m + (s - 1) * w
            if lam_perp <= 0 or lam_1 <= 0:
                continue
            detB = (lam_perp ** (s - 1)) * lam_1
            lam1s.append(lam_1)
            raw_btypes.append((s, w, detB, lam_1))

    q_den = math.lcm(*lam1s)
    base_q_scaled = q_den * (4 * m + n) // m
    max_qv_scaled = 0
    btypes: list[tuple[int, int, int, int]] = []
    for s, w, detB, lam_1 in raw_btypes:
        qv_scaled = q_den // lam_1 - q_den // m
        if qv_scaled > max_qv_scaled:
            max_qv_scaled = qv_scaled
        btypes.append((s, w, detB, s * qv_scaled))

    total_checked = 0
    total_above_threshold = 0
    square_hits: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    min_det_sq = min_abs_det * min_abs_det
    min_det_sq_scaled = min_det_sq * q_den
    four_pow = 4 ** (n - 1)
    m_pows = [m ** k for k in range(n + 1)]

    def _dfs(
        idx: int,
        used: int,
        cur_blks: list[tuple[int, int]],
        detA_num: int,
        q_delta_scaled: int,
    ) -> None:
        nonlocal total_checked, total_above_threshold
        total_checked += 1
        c = n - used
        detG_scaled = four_pow * detA_num * m_pows[c] * (base_q_scaled + q_delta_scaled)
        if detG_scaled >= min_det_sq_scaled:
            total_above_threshold += 1
            assert detG_scaled % q_den == 0
            detG_int = detG_scaled // q_den
            r = math.isqrt(detG_int)
            if r * r == detG_int:
                square_hits.append((r, tuple(cur_blks)))
        for i in range(idx, len(btypes)):
            s, w, detB, qB_scaled = btypes[i]
            if used + s > n:
                continue
            new_detA = detA_num * detB
            new_q = q_delta_scaled + qB_scaled
            rem = n - (used + s)
            ub_scaled = four_pow * new_detA * m_pows[rem] * (base_q_scaled + new_q + rem * max_qv_scaled)
            if ub_scaled < min_det_sq_scaled:
                continue
            cur_blks.append((s, w))
            _dfs(i, used + s, cur_blks, new_detA, new_q)
            cur_blks.pop()

    _dfs(0, 0, [], 1, 0)
    square_hits.sort(key=lambda x: -x[0])

    results: list[BlockGramCandidate] = []
    for r, blks in square_hits:
        G = build_r1_block_gram(n, blks)
        hr = hasse_minkowski_check(G)
        results.append(
            BlockGramCandidate(
                n=n,
                blocks=blks,
                abs_det=r,
                hasse_ok=hr.hasse_ok,
                bad_primes=hr.bad_primes,
            )
        )
    if return_above_count:
        return total_checked, total_above_threshold, results
    return total_checked, results


@dataclass(frozen=True)
class ColumnEquationResult:
    """Result of checking ``x^T G^{-1} x = 1`` over ``{±1}^n`` for an
    ``n ≡ 1 (mod 4)`` Gram matrix ``G = (n-1)I + J + 4E`` with block-diagonal ``E``.
    """

    n: int
    blocks: tuple[tuple[int, int], ...]
    num_column_types: int
    column_types: tuple[tuple, ...]
    ruled_out_reason: str | None


def check_r1_block_gram_columns(
    n: int,
    blocks: Sequence[tuple[int, int]],
) -> ColumnEquationResult:
    """Enumerate all column-type solutions to ``x^T G^{-1} x = 1`` for ``x ∈ {±1}^n``
    when ``G = (n-1)I + J + 4E`` has block-diagonal ``E`` with blocks of size ``s ∈ {2, 3}``.

    Also checks immediate obstructions:
      - Zero solutions to ``x^T G^{-1} x = 1``.
      - All solutions force ``|t| = s`` on some block (which forces identical rows,
        ``G_ij = n ≠ 1 + 4w``) or ``|dc| = c`` on ``c >= 2`` singletons (identical singleton rows).
    """
    from collections import Counter
    import sympy as sp

    if n % 4 != 1:
        raise ValueError(f"requires n ≡ 1 (mod 4); got {n}")
    m = (n - 1) // 4
    blks = tuple(blocks)
    for s, _ in blks:
        if s not in (2, 3):
            raise ValueError(f"check_r1_block_gram_columns currently supports s in {{2, 3}}; got {s}")
    counts = Counter(blks)
    used = sum(s for s, _ in blks)
    c = n - used

    denom = sp.Rational(4) + sp.Rational(c, m)
    for (s, w), cnt in counts.items():
        lam_1 = m + (s - 1) * w
        denom += cnt * sp.Rational(s, lam_1)

    btypes = sorted(counts.items())

    def _quad(s: int, w: int, abs_t: int) -> sp.Rational:
        lam_perp = m - w
        lam_1 = m + (s - 1) * w
        return (
            sp.Rational(s * s - abs_t * abs_t, s * lam_perp)
            + sp.Rational(abs_t * abs_t, s * lam_1)
        )

    sols: list[tuple] = []

    def _enum_k(idx: int, cur_k: list[int], cur_quad: sp.Rational) -> None:
        if idx == len(btypes):
            quad_A = cur_quad + sp.Rational(c, m)
            rhs = denom * (quad_A - 4)
            if rhs < 0:
                return
            sn = sp.integer_nthroot(rhs.p, 2)
            sd = sp.integer_nthroot(rhs.q, 2)
            if not (sn[1] and sd[1]):
                return
            L_target = sp.Rational(sn[0], sd[0])

            def _enum_d(j: int, cur_L: sp.Rational, cur_cs: int, d_vec: list[tuple[int, int]]) -> None:
                if j == len(btypes):
                    dc_rat = m * (L_target - cur_L)
                    if dc_rat.is_integer:
                        dc = int(dc_rat)
                        if abs(dc) <= c and (dc - c) % 2 == 0:
                            sols.append((tuple(cur_k), tuple(d_vec), dc, cur_cs + dc))
                    return
                (s, w), cnt = btypes[j]
                kh = cur_k[j]
                kl = cnt - kh
                lam_1 = m + (s - 1) * w
                for dh in range(-kh, kh + 1, 2):
                    if s == 2:
                        _enum_d(
                            j + 1,
                            cur_L + sp.Rational(dh * 2, lam_1),
                            cur_cs + dh * 2,
                            d_vec + [(dh, 0)],
                        )
                    else:
                        for dl in range(-kl, kl + 1, 2):
                            _enum_d(
                                j + 1,
                                cur_L + sp.Rational(dh * s + dl * (s - 2), lam_1),
                                cur_cs + dh * s + dl * (s - 2),
                                d_vec + [(dh, dl)],
                            )

            _enum_d(0, sp.Rational(0), 0, [])
            return

        (s, w), cnt = btypes[idx]
        qh = _quad(s, w, s)
        ql = _quad(s, w, s - 2)
        for kh in range(cnt + 1):
            _enum_k(idx + 1, cur_k + [kh], cur_quad + kh * qh + (cnt - kh) * ql)

    _enum_k(0, [], sp.Rational(0))

    reason: str | None = None
    if not sols:
        reason = "no_column_solutions"
    else:
        # Check if all solutions force |t|=s on some block type
        for j, ((s, w), cnt) in enumerate(btypes):
            if all(sol[0][j] == cnt for sol in sols):
                reason = f"forced_identical_rows_in_block_{(s, w)}"
                break
        if reason is None and c >= 2 and all(abs(sol[2]) == c for sol in sols):
            reason = f"forced_identical_singletons_c={c}"

    return ColumnEquationResult(
        n=n,
        blocks=blks,
        num_column_types=len(sols),
        column_types=tuple(sols),
        ruled_out_reason=reason,
    )


def build_r2_block_gram(
    s1: int,
    c1: Sequence[tuple[int, int]],
    s2: int,
    c2: Sequence[tuple[int, int]],
) -> list[list[int]]:
    """Build the ``(s1 + s2) × (s1 + s2)`` 2-block Gram matrix ``G = B1 ⊕ B2``
    for ``n = s1 + s2 ≡ 2 (mod 4)`` where each block has diagonal ``n``,
    base intra-block ``+2``, cross-block ``0``, and disjoint cliques ``(k, w)``
    with intra-clique value ``2 + 4*w``.
    """
    n = s1 + s2
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        G[i][i] = n
    for offset, sz, cfg in ((0, s1, c1), (s1, s2, c2)):
        for i in range(offset, offset + sz):
            for j in range(offset, offset + sz):
                if i != j:
                    G[i][j] = 2
        idx = offset
        for k, w in cfg:
            val = 2 + 4 * w
            for a in range(idx, idx + k):
                for b in range(idx, idx + k):
                    if a != b:
                        G[a][b] = val
            idx += k
        if idx > offset + sz:
            raise ValueError(f"Clique sizes exceed block size {sz}")
    return G


@dataclass(frozen=True)
class H34UnconditionalProof:
    """Certificate for the unconditional upper bound D(34) < threshold * 32^16."""

    threshold_mult: float
    max_s1_le_12: float
    max_m6_ge_1: float
    max_e4_seidel: float
    num_surviving_pairs: int
    max_m4_ge_2: float
    max_single_8: float
    c0_squares_ruled_out: int
    m4_1_above_non_square: int
    verified: bool


@dataclass(frozen=True)
class R2UnconditionalProof:
    """Certificate for an unconditional upper bound D(n) < threshold_mult * (n-2)^((n-2)/2)
    at an exceptional order n ≡ 2 (mod 4) where 2n - 2 is not a sum of two squares.
    """

    n: int
    ehlich_mult: int
    threshold_mult: float
    max_s1_below_min: float
    max_m6_ge_1: float
    max_e4_seidel: float
    num_surviving_pairs: int
    max_m4_ge_2: float
    max_single_8: float
    c0_squares_ruled_out: int
    m4_1_above_non_square: int
    verified: bool


def prove_r2_exceptional_unconditional_ub(
    n: int,
    threshold_mult: float | None = None,
) -> R2UnconditionalProof:
    """Prove a 100% unconditional upper bound ``D(n) < threshold_mult * (n-2)^((n-2)/2)``
    over ALL n×n {±1} matrices for an exceptional order ``n ≡ 2 (mod 4)`` where
    ``2n - 2`` is not a sum of two squares (e.g. ``n ∈ {34, 58, 70, 78, 94}``).

    If ``threshold_mult`` is omitted, automatically chooses the tightest 3-decimal
    ceiling above ``max(m6_ge_1_ub, e4_seidel_ub)``.
    """
    import math
    import numpy as np
    import sympy as sp

    if n % 4 != 2 or n < 10:
        raise ValueError(f"requires n ≡ 2 (mod 4), n >= 10; got {n}")
    ehlich = 2 * n - 2
    if not any(p % 4 == 3 and e % 2 == 1 for p, e in sp.factorint(ehlich).items()):
        raise ValueError(f"2n - 2 = {ehlich} IS a sum of two squares (Ehlich bound may be tight)")

    m = n - 2
    h = n // 2

    def _spectral_ub_block(s: int, delta_T: int) -> float:
        if s == 0:
            return float(m)
        if s == 1:
            return float(n)
        V = 4 * s * (s - 1) + delta_T
        x = math.sqrt(V / (s * (s - 1)))
        return (((float(n) - x) / m) ** (s - 1)) * (float(n) + (s - 1) * x)

    def _seidel_var_ub(s: int, mu_cap: float, delta_T_S: int = 0) -> float:
        """Exact KKT global maximum of prod_{j=1}^s (n + 2*mu_j) / m^(s-1) subject to
        sum mu_j = 0, sum mu_j^2 = s(s-1) + delta_T_S, and -n/2 < mu_j <= mu_cap.

        Checks ALL feasible r in {0, ..., s-1} eigenvalues pinned at mu_cap with
        (s - r - 1) copies of a and 1 copy of b <= mu_cap.
        """
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
        assert best > 0.0, f"No feasible capped spectrum for n={n}, s={s}, mu_cap={mu_cap}"
        return best

    def _exact_int_weyl_and_dk(M_int: np.ndarray, *, need_dk: bool = False):
        """Exact arbitrary-precision integer polar-factor Weyl (and optional Davis-Kahan)
        certification for symmetric integer matrix M_int.

        Converts LAPACK's float64 eigendecomposition to an exact dyadic pair
        V_hat = V_q / 2^52, w_hat = w_q / 2^45 and computes ||M V_hat - V_hat Lambda_hat||_F,
        ||V_hat^T V_hat - I||_F, and ||M u_1 - lam_1 u_1||_2 over exact integers (zero
        floating-point residual error).
        """
        from fractions import Fraction

        k_sz = M_int.shape[0]
        w_f, V_f = np.linalg.eigh(M_int.astype(np.float64))
        SV, SW = 52, 45
        V_q = np.rint(V_f * (1 << SV)).astype(np.int64).astype(object)
        w_q = np.rint(w_f * (1 << SW)).astype(np.int64).astype(object)
        M_obj = M_int.astype(np.int64).astype(object)

        def _ceil_sqrt(val: int) -> int:
            r = math.isqrt(val)
            return r if r * r == val else r + 1

        R_int = (M_obj @ V_q) * (1 << SW) - V_q * w_q[None, :]
        R_sq = int(np.sum(R_int * R_int))
        rho = Fraction(_ceil_sqrt(R_sq), 1 << (SV + SW))

        E_int = (V_q.T @ V_q) - np.eye(k_sz, dtype=object) * (1 << (2 * SV))
        E_sq = int(np.sum(E_int * E_int))
        eta = Fraction(_ceil_sqrt(E_sq), 1 << (2 * SV))
        assert eta < Fraction(1, 10**8), f"Orthogonality defect too large: {float(eta)}"

        M_fro = _ceil_sqrt(int(np.sum(M_obj * M_obj)))
        lam_max_abs = Fraction(max(abs(int(x)) for x in w_q), 1 << SW)
        weyl = rho + (M_fro + lam_max_abs) * eta
        assert weyl < Fraction(1, 10**7), f"Weyl radius too large: {float(weyl)}"

        l1 = Fraction(int(w_q[-1]), 1 << SW)
        if not need_dk:
            return float(l1 + weyl)

        l2 = Fraction(int(w_q[-2]), 1 << SW)
        gap_lo = l1 - l2 - 2 * weyl
        assert l2 - weyl > 0 and gap_lo > 1
        v1_q = V_q[:, -1]
        n1_sq = int(np.sum(v1_q * v1_q))
        n1_lo = math.isqrt(n1_sq)
        assert n1_lo > 0
        r1_int = (M_obj @ v1_q) * (1 << SW) - int(w_q[-1]) * v1_q
        r1_sq = int(np.sum(r1_int * r1_int))
        res_v1_hi = Fraction(_ceil_sqrt(r1_sq), (1 << SW) * n1_lo)
        # Rational upper bound on sqrt(2): 1414213563 / 10^9 (square = 2.0000000017896...)
        sqrt2_hi = Fraction(1414213563, 10**9)
        v1_inf_hat_hi = Fraction(max(abs(int(x)) for x in v1_q), n1_lo)
        v1_hi = min(Fraction(1), v1_inf_hat_hi + sqrt2_hi * res_v1_hi / gap_lo)
        # Convert exact rational bounds conservatively upward/downward to float64 (+2 ulps)
        eps = 1e-14
        return (
            float(l1 + weyl) * (1.0 + eps),
            float(l2 - weyl) * (1.0 - eps),
            float(l2 + weyl) * (1.0 + eps),
            min(1.0, float(v1_hi) + eps),
        )

    def _certified_top_eigval(S: np.ndarray) -> float:
        """Exact-integer certified polar-factor Weyl upper bound on lambda_max(S)."""
        return float(_exact_int_weyl_and_dk(np.rint(S).astype(np.int64), need_dk=False)) * (1.0 + 1e-14)

    def _m6_eq_1_seidel_ub(s: int) -> float:
        """Rigorous UB for a block of size s with exactly 1 pair |G_ij| = 6 (and
        all other off-diags +-2): in the non-negative top-eigenvector gauge of
        S_6, lambda_max(S_6) <= lambda_max(S_6^{(empty, +6)}), and tr(S_6^2) = s(s-1) + 16.
        """
        S = np.ones((s, s), dtype=np.float64) - np.eye(s)
        S[0, 1] = S[1, 0] = 3.0
        mu1 = _certified_top_eigval(S)
        return _seidel_var_ub(s, mu1, delta_T_S=16)

    def _fits(s: int, edges: Sequence[tuple[int, int]]) -> bool:
        return all(u < s and v < s for u, v in edges)

    graphs_by_e = {
        0: [("empty", ())],
        1: [("K2", ((0, 1),))],
        2: [("P3", ((0, 1), (0, 2))), ("2K2", ((0, 1), (2, 3)))],
        3: [
            ("K1,3", ((0, 1), (0, 2), (0, 3))),
            ("K3", ((0, 1), (0, 2), (1, 2))),
            ("P4", ((0, 1), (1, 2), (2, 3))),
            ("P3+K2", ((0, 1), (0, 2), (3, 4))),
            ("3K2", ((0, 1), (2, 3), (4, 5))),
        ],
        4: [
            ("K1,4", ((0, 1), (0, 2), (0, 3), (0, 4))),
            ("C4", ((0, 1), (1, 2), (2, 3), (3, 0))),
            ("paw", ((0, 1), (1, 2), (2, 0), (0, 3))),
            ("P5", ((0, 1), (1, 2), (2, 3), (3, 4))),
            ("fork", ((0, 1), (0, 2), (0, 3), (1, 4))),
            ("K3+K2", ((0, 1), (1, 2), (2, 0), (3, 4))),
            ("K1,3+K2", ((0, 1), (0, 2), (0, 3), (4, 5))),
            ("P4+K2", ((0, 1), (1, 2), (2, 3), (4, 5))),
            ("2P3", ((0, 1), (0, 2), (3, 4), (3, 5))),
            ("P3+2K2", ((0, 1), (0, 2), (3, 4), (5, 6))),
            ("4K2", ((0, 1), (2, 3), (4, 5), (6, 7))),
        ],
    }

    def _seidel_top_eig(s: int, edges: Sequence[tuple[int, int]]) -> float:
        S = np.ones((s, s), dtype=np.float64) - np.eye(s)
        for u, v in edges:
            S[u, v] = S[v, u] = -1.0
        return _certified_top_eigval(S)

    def _exact_det_and_diag_inv(s: int, edges: Sequence[tuple[int, int]]) -> tuple[int, dict]:
        active = sorted({u for e in edges for u in e})
        k = len(active)
        c = s - k
        idx_map = {v: i for i, v in enumerate(active)}
        if c == 0:
            M = sp.Matrix(k, k, lambda i, j: n if i == j else 2)
            for u, v in edges:
                M[idx_map[u], idx_map[v]] = M[idx_map[v], idx_map[u]] = -2
            detB = int(M.det())
            Minv = M.inv()
            return detB, {v: Minv[idx_map[v], idx_map[v]] for v in active}
        M = sp.Matrix(k + 1, k + 1, lambda i, j: 0)
        for i in range(k):
            for j in range(k):
                M[i, j] = n if i == j else 2
        for u, v in edges:
            M[idx_map[u], idx_map[v]] = M[idx_map[v], idx_map[u]] = -2
        for i in range(k):
            M[i, k] = M[k, i] = 2 * c
        M[k, k] = c * (m + 2 * c)
        detM = int(M.det())
        assert detM % c == 0
        detB = (m ** (c - 1)) * (detM // c)
        Minv = M.inv()
        diag_inv: dict = {v: Minv[idx_map[v], idx_map[v]] for v in active}
        diag_inv["passive"] = Minv[k, k] + sp.Rational(c - 1, c * m)
        return detB, diag_inv

    m6_h = max(
        math.sqrt(_m6_eq_1_seidel_ub(h) * (m + 2.0 * h)),
        math.sqrt(_spectral_ub_block(h, 128) * _spectral_ub_block(h, 0)),
    )
    if threshold_mult is None:
        e4_auto = max(
            max(
                math.sqrt(
                    max(_seidel_var_ub(s1, _seidel_top_eig(s1, ed)) for _, ed in graphs_by_e[4] if _fits(s1, ed))
                    * (m + 2.0 * (n - s1))
                ),
                math.sqrt(
                    (m + 2.0 * s1)
                    * max(_seidel_var_ub(n - s1, _seidel_top_eig(n - s1, ed)) for _, ed in graphs_by_e[4] if _fits(n - s1, ed))
                ),
            )
            for s1 in (h - 1, h)
        )
        threshold_mult = math.ceil(max(m6_h, e4_auto) * 1000.0) / 1000.0

    # 1. Partition bound
    s1_min = h
    while s1_min > 0 and math.sqrt(_spectral_ub_block(s1_min - 1, 0) * _spectral_ub_block(n - (s1_min - 1), 0)) >= threshold_mult:
        s1_min -= 1
    max_s1_below_min = (
        max(
            math.sqrt(_spectral_ub_block(s1, 0) * _spectral_ub_block(n - s1, 0))
            for s1 in range(0, s1_min)
        )
        if s1_min > 0
        else 0.0
    )

    # 2. m6 >= 1 across all s1 in [s1_min .. h]:
    #    m6 = 1 via Seidel-gauge S_6^{(empty,+6)} UB; m6 >= 2 (or any |G_ij|>=10) via delta_T >= 128 spectral UB.
    max_m6_ge_1 = max(
        max(
            math.sqrt(_m6_eq_1_seidel_ub(s1) * (m + 2.0 * (n - s1))),
            math.sqrt((m + 2.0 * s1) * _m6_eq_1_seidel_ub(n - s1)),
            math.sqrt(_spectral_ub_block(s1, 128) * _spectral_ub_block(n - s1, 0)),
            math.sqrt(_spectral_ub_block(s1, 0) * _spectral_ub_block(n - s1, 128)),
        )
        for s1 in range(s1_min, h + 1)
    )

    # 3. Seidel e(H) >= 4 across all s1 in [s1_min .. h]
    max_e4 = 0.0
    for s1 in range(s1_min, h + 1):
        s2 = n - s1
        u1 = max(_seidel_var_ub(s1, _seidel_top_eig(s1, ed)) for _, ed in graphs_by_e[4] if _fits(s1, ed))
        u2 = max(_seidel_var_ub(s2, _seidel_top_eig(s2, ed)) for _, ed in graphs_by_e[4] if _fits(s2, ed))
        max_e4 = max(max_e4, math.sqrt(u1 * (m + 2.0 * s2)), math.sqrt((m + 2.0 * s1) * u2))

    # 4. Enumerate surviving (B1, B2) with e <= 3
    det_cache: dict[int, list[tuple[int, dict, int, str, tuple[tuple[int, int], ...]]]] = {}
    for s in range(s1_min, n - s1_min + 1):
        det_cache[s] = []
        for e in range(4):
            for name, edges in graphs_by_e[e]:
                if not _fits(s, edges):
                    continue
                d, dinv = _exact_det_and_diag_inv(s, edges)
                det_cache[s].append((d, dinv, e, name, edges))

    surv_pairs = []
    for s1 in range(s1_min, h + 1):
        s2 = n - s1
        for d1, dinv1, e1, n1, ed1 in det_cache[s1]:
            for d2, dinv2, e2, n2, ed2 in det_cache[s2]:
                if s1 == s2 and (e1, n1) > (e2, n2):
                    continue
                prod_d = d1 * d2
                mult = math.exp(0.5 * math.log(prod_d) - (h - 1) * math.log(m))
                if mult >= threshold_mult - 1e-9:
                    surv_pairs.append((mult, prod_d, s1, n1, ed1, dinv1, s2, n2, ed2, dinv2))

    # 5. C = 0 squares & 6. C != 0 checks
    c0_squares_ruled_out = 0
    m4_1_above_non_square = 0
    max_m4_ge_2 = 0.0
    max_single_8 = 0.0
    half_m = m // 2

    for mult, prod_d, s1, n1, ed1, dinv1, s2, n2, ed2, dinv2 in surv_pairs:
        if math.isqrt(prod_d) ** 2 == prod_d:
            if (s1, n1, s2, n2) == (h, "empty", h, "empty"):
                # 2n - 2 already verified not a sum of two squares => Hasse-Minkowski fails
                c0_squares_ruled_out += 1
            elif (s1, n1, s2, n2) == (h, "K2", h, "K2"):
                # Exact column equation x^T G^{-1} x = 1: verify 0 solutions
                denom = sp.Rational(1) + sp.Rational(2, half_m - 2) + sp.Rational(h - 2, half_m)
                vals = set()
                for a_s in (0, 1):
                    qA = (
                        sp.Rational(2, half_m - 2) if a_s else sp.Rational(2, half_m + 2)
                    ) + sp.Rational(h - 2, half_m)
                    for da in ([-2, 2] if a_s else [0]):
                        for dc in range(-(h - 2), h - 1, 2):
                            L = sp.Rational(da, half_m - 2) + sp.Rational(dc, half_m)
                            vals.add(sp.Rational(1, 2) * (qA - (L * L) / denom))
                assert not any((1 - v) in vals for v in vals), f"Unexpected K2(+)K2 column sol at n={n}"
                c0_squares_ruled_out += 1
            else:
                raise AssertionError(f"Unexpected C=0 square at n={n}: {(s1, n1, s2, n2)}")

        # k = 1 exact (|C_ij| = 4)
        for b1u in dinv1.values():
            for b2v in dinv2.values():
                detG_rat = sp.Integer(prod_d) * (1 - 16 * b1u * b2v)
                assert detG_rat.is_integer
                detG = int(detG_rat)
                m1 = math.exp(0.5 * math.log(detG) - (h - 1) * math.log(m))
                if m1 >= threshold_mult - 1e-9:
                    assert math.isqrt(detG) ** 2 != detG, f"Unexpected k=1 square at n={n}"
                    m4_1_above_non_square += 1

        # k = 1 (|C_ij| >= 8) and ALL k >= 2 via exact-integer Weyl + Davis-Kahan PSD trace bound
        B1i = np.full((s1, s1), 2, dtype=np.int64)
        np.fill_diagonal(B1i, n)
        for u, v in ed1:
            B1i[u, v] = B1i[v, u] = -2
        B2i = np.full((s2, s2), 2, dtype=np.int64)
        np.fill_diagonal(B2i, n)
        for u, v in ed2:
            B2i[u, v] = B2i[v, u] = -2

        lam1_hi, lam2_lo, lam2_hi, v1m_hi = _exact_int_weyl_and_dk(B1i, need_dk=True)
        mu1_hi, mu2_lo, mu2_hi, v2m_hi = _exact_int_weyl_and_dk(B2i, need_dk=True)

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

        max_single_8 = max(max_single_8, _ub_kF(1, 64.0))
        # Check k = 2 .. k_crude_cutoff where crude bound alone is < threshold_mult (monotone in k!)
        k_cutoff = max(2, int(math.ceil(2.0 * math.log(mult / threshold_mult) * (lam1_hi * mu1_hi) / 16.0)) + 2)
        for k in range(2, k_cutoff + 1):
            max_m4_ge_2 = max(max_m4_ge_2, _ub_kF(k, 16.0 * k))

    verified = (
        max_s1_below_min < threshold_mult
        and max_m6_ge_1 < threshold_mult
        and max_e4 < threshold_mult
        and max_m4_ge_2 < threshold_mult
        and max_single_8 < threshold_mult
    )
    return R2UnconditionalProof(
        n=n,
        ehlich_mult=ehlich,
        threshold_mult=threshold_mult,
        max_s1_below_min=max_s1_below_min,
        max_m6_ge_1=max_m6_ge_1,
        max_e4_seidel=max_e4,
        num_surviving_pairs=len(surv_pairs),
        max_m4_ge_2=max_m4_ge_2,
        max_single_8=max_single_8,
        c0_squares_ruled_out=c0_squares_ruled_out,
        m4_1_above_non_square=m4_1_above_non_square,
        verified=verified,
    )


def prove_h34_unconditional_ub(threshold_mult: float = 65.505) -> H34UnconditionalProof:
    """Thin wrapper around :func:`prove_r2_exceptional_unconditional_ub` for n = 34."""
    r = prove_r2_exceptional_unconditional_ub(34, threshold_mult=threshold_mult)
    return H34UnconditionalProof(
        threshold_mult=r.threshold_mult,
        max_s1_le_12=r.max_s1_below_min,
        max_m6_ge_1=r.max_m6_ge_1,
        max_e4_seidel=r.max_e4_seidel,
        num_surviving_pairs=r.num_surviving_pairs,
        max_m4_ge_2=r.max_m4_ge_2,
        max_single_8=r.max_single_8,
        c0_squares_ruled_out=r.c0_squares_ruled_out,
        m4_1_above_non_square=r.m4_1_above_non_square,
        verified=r.verified,
    )
