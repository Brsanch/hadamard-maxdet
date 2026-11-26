"""Smart 2-valued (|G| ⊆ {1, 3}) submanifold search at H23.

Background
----------
At order n=23, three known structurally-distinct ±1 matrices have Gram off-diag
values in ``{-3, -1, +1, +3}`` — i.e. ``|G_off-diag| ∈ {1, 3}``:

* **Orrick (2003)** — ``|det| = 2,779,447,296,000,000`` (current world record).
  Multiset (count_|G|=1, count_|G|=3) = ``(208, 45)``. Gram support ``{-1, +3}``.
* **BTC (rich-ansatz)** — ``|det| = 2,626,567,700,217,856``. Multiset ``(220, 33)``.
  Gram support ``{-3, -1, +1, +3}``.
* **KMS-refined** — ``|det| = 2,363,844,814,962,688``. Multiset ``(201, 52)``.
  Gram support ``{-3, -1, +1, +3}``.

These three are "classes" in the multiset-of-|G_offdiag| sense. The hypothesis
of this module: a smarter search restricted to the ``|G| ⊆ {1, 3}`` submanifold
should find a 4th class (different multiset), ideally with ``|det| > Orrick``.

Three search techniques implemented:

1. **Manifold drift with class tabu**. Random ascent on
   ``|G| ⊆ {1, 3}``, starting from each known matrix, with the three known
   multisets locked out (any move into a locked class is rejected).
2. **Bridge excursion**. Take a known matrix, deliberately leave the manifold
   for ``n_bridge`` steps (random flips, allowing |G| ∉ {1, 3} entries), then
   apply a snap-back energy that drives ``count_outside`` back to 0. The
   intermediate excursion gives a chance to land in a different basin.
3. **Boundary restart**. Start from k-flip-neighbours of the known matrices
   that are themselves on the |G| ⊆ {1, 3} submanifold (such neighbours exist
   when the flip pattern preserves the manifold constraint), and run
   technique 1 from there.

Public API
----------

* :func:`gram_abs_class_signature` — multiset (count_|G|=1, count_|G|=3).
* :func:`is_on_manifold` — returns True iff every off-diag has |G| ∈ {1, 3}.
* :func:`enumerate_known_2val_classes` — load Orrick / BTC / KMS as
  ``KnownClass`` records.
* :func:`is_in_known_class` — check matrix's multiset against a tabu list.
* :class:`AbsClassState` — incremental tracker of (count_1, count_3) counts.
* :func:`manifold_walk` — Technique 1.
* :func:`bridge_excursion` — Technique 2.
* :func:`boundary_restart_seeds` — generate seeds for Technique 3 from a
  given starter matrix.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
# Class signature on |G_offdiag| (cardinality multiset)
# ---------------------------------------------------------------------------


AbsClassMultiset = tuple[tuple[int, int], ...]
"""Hashable tuple ``((|G|_value, count), ...)`` sorted by value."""


def gram_abs_class_signature(M: Sequence[Sequence[int]]) -> AbsClassMultiset:
    """Return the multiset of off-diag |G_ij| values (i < j) as a sorted tuple.

    For a ±1 matrix M of order n, this is the count distribution of
    ``|<v_i, v_j>|`` over the ``C(n, 2)`` unordered row pairs. Invariant under
    row/column permutations and row/column negations (both flip signs of off-
    diagonals symmetrically without changing magnitudes).
    """
    A = np.asarray(M, dtype=np.int64)
    n = A.shape[0]
    if n < 2:
        return ()
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    abs_vals = np.abs(G[iu])
    cnt: Counter[int] = Counter()
    for v in abs_vals.tolist():
        cnt[int(v)] += 1
    return tuple(sorted(cnt.items()))


def is_on_manifold(
    M: Sequence[Sequence[int]],
    *,
    allowed_abs: Iterable[int] = (1, 3),
) -> bool:
    """Return True iff every off-diagonal Gram entry has ``|G_ij| in allowed_abs``."""
    allowed = set(int(a) for a in allowed_abs)
    A = np.asarray(M, dtype=np.int64)
    n = A.shape[0]
    if n < 2:
        return True
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    for v in G[iu].tolist():
        if int(abs(int(v))) not in allowed:
            return False
    return True


def count_outside_abs_support(
    M: Sequence[Sequence[int]],
    *,
    allowed_abs: Iterable[int] = (1, 3),
) -> int:
    """Number of unordered off-diag pairs with ``|G_ij| not in allowed_abs``."""
    allowed = set(int(a) for a in allowed_abs)
    A = np.asarray(M, dtype=np.int64)
    n = A.shape[0]
    if n < 2:
        return 0
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    bad = sum(1 for v in G[iu].tolist() if int(abs(int(v))) not in allowed)
    return int(bad)


# ---------------------------------------------------------------------------
# Known 2-valued classes (loaded from results/*.json on demand)
# ---------------------------------------------------------------------------


@dataclass
class KnownClass:
    """Representative matrix for a known |G| ⊆ {1, 3} class at H23."""

    name: str
    matrix: Matrix
    abs_det: int
    abs_class_multiset: AbsClassMultiset


_RESULT_FILES = {
    "orrick": "H23_orrick2003_record.json",
    "btc": "H23_rich_ansatz_best.json",
    "kms": "H23_from_kms_best.json",
}


def enumerate_known_2val_classes(
    results_dir: str | Path = "results",
) -> list[KnownClass]:
    """Load Orrick, BTC, and KMS from the canonical result files.

    Each file must contain a top-level ``entries`` list-of-lists of ±1 ints
    and an ``abs_det`` field. The returned list is sorted by descending
    ``abs_det``.

    Raises ``FileNotFoundError`` if a file is missing, ``ValueError`` if any
    matrix is not on the |G| ⊆ {1, 3} submanifold (sanity check — this
    invariant is the whole point of the module).
    """
    base = Path(results_dir)
    out: list[KnownClass] = []
    for name, fname in _RESULT_FILES.items():
        path = base / fname
        with path.open() as f:
            data = json.load(f)
        entries = [list(row) for row in data["entries"]]
        n = len(entries)
        for r in entries:
            if len(r) != n:
                raise ValueError(f"{path}: matrix is not square")
            for v in r:
                if v not in (-1, 1):
                    raise ValueError(f"{path}: non ±1 entry {v!r}")
        if not is_on_manifold(entries, allowed_abs=(1, 3)):
            raise ValueError(
                f"{path}: matrix is not on the |G| ⊆ {{1, 3}} submanifold"
            )
        sig = gram_abs_class_signature(entries)
        # Defensive: recompute |det| from the file's entries.
        abs_det = abs(integer_determinant(entries))
        if "abs_det" in data and abs_det != int(data["abs_det"]):
            raise ValueError(
                f"{path}: abs_det field {data['abs_det']} disagrees with "
                f"recomputed {abs_det}"
            )
        out.append(
            KnownClass(
                name=name,
                matrix=Matrix(n=n, entries=entries),
                abs_det=abs_det,
                abs_class_multiset=sig,
            )
        )
    out.sort(key=lambda k: -k.abs_det)
    return out


def is_in_known_class(
    M: Sequence[Sequence[int]],
    locked_multisets: Iterable[AbsClassMultiset],
) -> bool:
    """Return True iff ``gram_abs_class_signature(M)`` is in ``locked_multisets``."""
    sig = gram_abs_class_signature(M)
    return sig in set(locked_multisets)


# ---------------------------------------------------------------------------
# Incremental |G| ⊆ {1, 3} class state with O(n) per-flip updates
# ---------------------------------------------------------------------------


@dataclass
class AbsClassState:
    """Tracks the row Gram and the (count_1, count_3, count_out) under flips.

    Invariants (maintained by :meth:`apply_after_flip`):

      * ``G == M @ M.T``  (numpy int (n, n))
      * ``count_abs1 == #{(i, j) : i < j, |G[i, j]| == 1}``
      * ``count_abs3 == #{(i, j) : i < j, |G[i, j]| == 3}``
      * ``count_out  == n(n-1)/2 - count_abs1 - count_abs3``

    The supported manifold is ``|G| in {1, 3}`` (off-diag).
    """

    n: int
    G: np.ndarray
    count_abs1: int
    count_abs3: int
    count_out: int

    @classmethod
    def from_matrix(cls, M: Sequence[Sequence[int]]) -> "AbsClassState":
        A = np.asarray(M, dtype=np.int64)
        n = A.shape[0]
        if n < 2:
            return cls(
                n=n,
                G=A @ A.T if n else np.zeros((0, 0), dtype=np.int64),
                count_abs1=0,
                count_abs3=0,
                count_out=0,
            )
        G = (A @ A.T).astype(np.int64)
        iu = np.triu_indices(n, k=1)
        vals = np.abs(G[iu])
        c1 = int(np.sum(vals == 1))
        c3 = int(np.sum(vals == 3))
        total = n * (n - 1) // 2
        co = total - c1 - c3
        return cls(n=n, G=G, count_abs1=c1, count_abs3=c3, count_out=co)

    def deltas_for_flip(
        self,
        M_after: Sequence[Sequence[int]],
        i: int,
        c: int,
    ) -> tuple[int, int, int]:
        """Return (d_abs1, d_abs3, d_out) for flipping entry (i, c) of M.

        ``M_after[i][c]`` is the *new* value; ``M_after[r][k]`` for other
        positions are the unchanged old values.

        Costs O(n). Does NOT mutate the state.
        """
        n = self.n
        if n < 2:
            return 0, 0, 0
        new_val = int(M_after[i][c])
        delta_factor = 2 * new_val
        Gi_old = self.G[i]
        d1 = 0
        d3 = 0
        for r in range(n):
            if r == i:
                continue
            old = int(Gi_old[r])
            new = old + delta_factor * int(M_after[r][c])
            ao = abs(old)
            an = abs(new)
            if ao == an:
                continue
            if ao == 1:
                d1 -= 1
            elif ao == 3:
                d3 -= 1
            if an == 1:
                d1 += 1
            elif an == 3:
                d3 += 1
        d_out = -(d1 + d3)
        return d1, d3, d_out

    def apply_after_flip(
        self,
        M_after: Sequence[Sequence[int]],
        i: int,
        c: int,
    ) -> None:
        """Apply the deltas from a just-completed entry flip. Cost O(n)."""
        n = self.n
        if n < 2:
            return
        new_val = int(M_after[i][c])
        delta_factor = 2 * new_val
        Gi = self.G[i]
        d1 = 0
        d3 = 0
        for r in range(n):
            if r == i:
                continue
            old = int(Gi[r])
            new = old + delta_factor * int(M_after[r][c])
            if new != old:
                ao = abs(old)
                an = abs(new)
                if ao != an:
                    if ao == 1:
                        d1 -= 1
                    elif ao == 3:
                        d3 -= 1
                    if an == 1:
                        d1 += 1
                    elif an == 3:
                        d3 += 1
                Gi[r] = new
                self.G[r, i] = new
        self.count_abs1 += d1
        self.count_abs3 += d3
        self.count_out -= (d1 + d3)


# ---------------------------------------------------------------------------
# Technique 1 — manifold drift with class tabu
# ---------------------------------------------------------------------------


@dataclass
class WalkResult:
    """Outcome of :func:`manifold_walk` (a single SA-like trajectory)."""

    technique: str
    seed_name: str
    final_matrix: list[list[int]]
    final_abs_det: int
    final_signature: AbsClassMultiset
    feasible_steps: int
    iterations: int
    accepted: int
    rejected_class: int
    rejected_metro: int
    rejected_singular: int
    rejected_off_manifold: int
    walltime_seconds: float
    best_feasible_abs_det: int
    best_feasible_signature: AbsClassMultiset | None
    best_feasible_entries: list[list[int]] | None
    visited_signatures: list[tuple[AbsClassMultiset, int]] = field(default_factory=list)
    """All distinct (signature, best |det| in this signature) pairs visited."""


def manifold_walk(
    seed: Matrix,
    *,
    iterations: int = 60_000,
    locked_multisets: Iterable[AbsClassMultiset] = (),
    t_start: float = 0.5,
    t_end: float = 0.005,
    rng_seed: int | None = None,
    refresh_every: int = 2000,
    technique_label: str = "manifold_drift",
    seed_name: str = "seed",
    abs1_target_bias: float = 0.0,
    target_abs3: int | None = None,
) -> WalkResult:
    """Single-entry-flip SA restricted to the |G| ⊆ {1, 3} submanifold.

    Energy: ``E = -log|det| + huge * count_out + lambda * (count_abs3 - target)^2``
    (the second term, with ``huge`` effectively, is implemented as a hard
    rejection of off-manifold proposals — both cheaper and stricter than a
    soft penalty). The third term is included only if ``target_abs3`` is
    given (Orrick's split has count_abs3 = 45; M1-cardinality-bias-style
    suggests biasing toward different counts to discover new classes).

    Class tabu: any proposal whose new ``gram_abs_class_signature`` matches
    one of ``locked_multisets`` is rejected.

    The walk starts from ``seed`` (which need not be on the manifold; it
    must have ``count_out == 0`` for the first-step feasibility check to
    pass — pass a known matrix or use the snap-back driver in
    :func:`bridge_excursion`).
    """
    rng = random.Random(rng_seed)
    n = seed.n
    locked = set(locked_multisets)
    state = IncrementalState.from_matrix(seed.entries)
    if state.det == 0:
        return WalkResult(
            technique=technique_label,
            seed_name=seed_name,
            final_matrix=[row[:] for row in state.M],
            final_abs_det=0,
            final_signature=gram_abs_class_signature(state.M),
            feasible_steps=0,
            iterations=0,
            accepted=0,
            rejected_class=0,
            rejected_metro=0,
            rejected_singular=0,
            rejected_off_manifold=0,
            walltime_seconds=0.0,
            best_feasible_abs_det=0,
            best_feasible_signature=None,
            best_feasible_entries=None,
        )

    abs_state = AbsClassState.from_matrix(state.M)
    on_manifold = abs_state.count_out == 0
    cur_e = -math.log(abs(state.det))
    if target_abs3 is not None:
        cur_e += abs1_target_bias * (abs_state.count_abs3 - target_abs3) ** 2
    feasible_steps = int(on_manifold)
    accepted = 0
    rejected_class = 0
    rejected_metro = 0
    rejected_singular = 0
    rejected_off_manifold = 0

    # Track best |det| per visited signature.
    visited: dict[AbsClassMultiset, int] = {}
    if on_manifold:
        visited[gram_abs_class_signature(state.M)] = abs(state.det)

    best_feasible_abs_det = 0
    best_feasible_sig: AbsClassMultiset | None = None
    best_feasible_entries: list[list[int]] | None = None
    if on_manifold:
        sig0 = gram_abs_class_signature(state.M)
        if sig0 not in locked:
            best_feasible_abs_det = abs(state.det)
            best_feasible_sig = sig0
            best_feasible_entries = [row[:] for row in state.M]

    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    t0 = time.monotonic()

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        i = rng.randrange(n)
        c = rng.randrange(n)
        new_det = propose_entry_flip(state, i, c)
        if new_det == 0:
            rejected_singular += 1
            continue

        # Manifold check first (cheap O(n)).
        old_val = state.M[i][c]
        state.M[i][c] = -old_val
        d1, d3, dout = abs_state.deltas_for_flip(state.M, i, c)
        new_count_out = abs_state.count_out + dout
        if new_count_out != 0:
            # Off-manifold proposal: hard reject.
            state.M[i][c] = old_val
            rejected_off_manifold += 1
            continue

        # Class-tabu check (signature is uniquely determined by counts when
        # count_out == 0, and the only nonzero entries in the multiset are
        # |G|=1 and |G|=3).
        new_count_abs1 = abs_state.count_abs1 + d1
        new_count_abs3 = abs_state.count_abs3 + d3
        new_sig = _abs_signature_from_counts(new_count_abs1, new_count_abs3)
        if new_sig in locked:
            state.M[i][c] = old_val
            rejected_class += 1
            continue

        # Energy check (Metropolis).
        prop_e = -math.log(abs(new_det))
        if target_abs3 is not None:
            prop_e += abs1_target_bias * (new_count_abs3 - target_abs3) ** 2
        delta = prop_e - cur_e
        if delta > 0 and rng.random() >= math.exp(-delta / max(t, 1e-12)):
            state.M[i][c] = old_val
            rejected_metro += 1
            continue

        # Accepted: commit (det, adj) AND counts/G.
        state.M[i][c] = old_val  # apply_entry_flip expects OLD value in place
        apply_entry_flip(state, i, c)
        abs_state.apply_after_flip(state.M, i, c)
        cur_e = prop_e
        accepted += 1
        feasible_steps += 1

        # Track visited signatures.
        ad = abs(state.det)
        prev = visited.get(new_sig, 0)
        if ad > prev:
            visited[new_sig] = ad

        if new_sig not in locked and ad > best_feasible_abs_det:
            best_feasible_abs_det = ad
            best_feasible_sig = new_sig
            best_feasible_entries = [row[:] for row in state.M]

        if refresh_every > 0 and accepted % refresh_every == 0:
            refresh(state)
            if state.det != 0:
                cur_e = -math.log(abs(state.det))
                if target_abs3 is not None:
                    cur_e += abs1_target_bias * (
                        abs_state.count_abs3 - target_abs3
                    ) ** 2
            # Re-derive abs_state in case of any drift.
            abs_state = AbsClassState.from_matrix(state.M)

    walltime = time.monotonic() - t0
    final_sig = gram_abs_class_signature(state.M)
    visited_pairs = sorted(visited.items(), key=lambda kv: -kv[1])
    return WalkResult(
        technique=technique_label,
        seed_name=seed_name,
        final_matrix=[row[:] for row in state.M],
        final_abs_det=abs(state.det),
        final_signature=final_sig,
        feasible_steps=feasible_steps,
        iterations=iterations,
        accepted=accepted,
        rejected_class=rejected_class,
        rejected_metro=rejected_metro,
        rejected_singular=rejected_singular,
        rejected_off_manifold=rejected_off_manifold,
        walltime_seconds=walltime,
        best_feasible_abs_det=best_feasible_abs_det,
        best_feasible_signature=best_feasible_sig,
        best_feasible_entries=best_feasible_entries,
        visited_signatures=visited_pairs,
    )


def _abs_signature_from_counts(c1: int, c3: int) -> AbsClassMultiset:
    """Build the multiset signature from (count_|G|=1, count_|G|=3) when
    ``count_out == 0``. Drops zero counts."""
    parts: list[tuple[int, int]] = []
    if c1 > 0:
        parts.append((1, c1))
    if c3 > 0:
        parts.append((3, c3))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Technique 2 — bridge excursion
# ---------------------------------------------------------------------------


@dataclass
class BridgeWalkResult:
    """Outcome of :func:`bridge_excursion`."""

    technique: str
    seed_name: str
    excursion_steps: int
    snap_back_iters: int
    snap_back_succeeded: bool
    final_count_out: int
    excursion_max_count_out: int
    walk: WalkResult | None
    walltime_seconds: float


def bridge_excursion(
    seed: Matrix,
    *,
    excursion_steps: int = 20,
    snap_back_iters: int = 30_000,
    walk_iters: int = 40_000,
    locked_multisets: Iterable[AbsClassMultiset] = (),
    rng_seed: int | None = None,
    snap_back_t_start: float = 0.05,
    snap_back_t_end: float = 0.001,
    snap_back_mu: float = 0.5,
    walk_t_start: float = 0.3,
    walk_t_end: float = 0.005,
    seed_name: str = "seed",
    technique_label: str = "bridge_excursion",
) -> BridgeWalkResult:
    """Excursion off the |G| ⊆ {1, 3} manifold, then snap back, then walk.

    Steps:
      1. Apply ``excursion_steps`` random entry flips with no constraint
         (the matrix may leave the manifold; ``count_out`` may grow).
      2. Run SA targeting ``E = -log|det| + mu * count_out`` for
         ``snap_back_iters`` iterations to drive ``count_out`` back to 0.
      3. If ``count_out == 0`` after step 2, run :func:`manifold_walk` from
         the snapped-back matrix for ``walk_iters`` iterations.
    """
    rng = random.Random(rng_seed)
    n = seed.n
    t0 = time.monotonic()

    # Step 1 — excursion.
    state = IncrementalState.from_matrix(seed.entries)
    abs_state = AbsClassState.from_matrix(state.M)
    excursion_max = abs_state.count_out
    eff_steps = 0
    for _ in range(excursion_steps):
        i = rng.randrange(n)
        c = rng.randrange(n)
        new_det = propose_entry_flip(state, i, c)
        if new_det == 0:
            continue
        apply_entry_flip(state, i, c)
        abs_state.apply_after_flip(state.M, i, c)
        if abs_state.count_out > excursion_max:
            excursion_max = abs_state.count_out
        eff_steps += 1

    # Step 2 — snap back.
    cur_e = -math.log(abs(state.det)) + snap_back_mu * float(abs_state.count_out)
    log_ratio = (
        math.log(snap_back_t_end / snap_back_t_start)
        if snap_back_t_start > 0
        else 0.0
    )
    snap_back_succeeded = abs_state.count_out == 0
    refresh_interval = 2000
    accepted_sb = 0
    for step in range(snap_back_iters):
        if abs_state.count_out == 0:
            snap_back_succeeded = True
            break
        frac = step / max(1, snap_back_iters - 1)
        t = snap_back_t_start * math.exp(log_ratio * frac)
        i = rng.randrange(n)
        c = rng.randrange(n)
        new_det = propose_entry_flip(state, i, c)
        if new_det == 0:
            continue
        old_val = state.M[i][c]
        state.M[i][c] = -old_val
        d1, d3, dout = abs_state.deltas_for_flip(state.M, i, c)
        new_count_out = abs_state.count_out + dout
        # Snap-back energy: drive count_out toward 0.
        prop_e = -math.log(abs(new_det)) + snap_back_mu * float(new_count_out)
        delta = prop_e - cur_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            state.M[i][c] = old_val
            apply_entry_flip(state, i, c)
            abs_state.apply_after_flip(state.M, i, c)
            cur_e = prop_e
            accepted_sb += 1
            if refresh_interval > 0 and accepted_sb % refresh_interval == 0:
                refresh(state)
                if state.det != 0:
                    cur_e = (
                        -math.log(abs(state.det))
                        + snap_back_mu * float(abs_state.count_out)
                    )
        else:
            state.M[i][c] = old_val
    snap_back_succeeded = abs_state.count_out == 0
    final_co = abs_state.count_out

    if not snap_back_succeeded:
        return BridgeWalkResult(
            technique=technique_label,
            seed_name=seed_name,
            excursion_steps=eff_steps,
            snap_back_iters=snap_back_iters,
            snap_back_succeeded=False,
            final_count_out=final_co,
            excursion_max_count_out=excursion_max,
            walk=None,
            walltime_seconds=time.monotonic() - t0,
        )

    # Step 3 — manifold walk from the snapped-back matrix.
    snapped = Matrix(n=n, entries=[row[:] for row in state.M])
    walk = manifold_walk(
        snapped,
        iterations=walk_iters,
        locked_multisets=locked_multisets,
        t_start=walk_t_start,
        t_end=walk_t_end,
        rng_seed=(None if rng_seed is None else rng_seed + 7),
        technique_label=technique_label,
        seed_name=seed_name + ".bridge",
    )
    return BridgeWalkResult(
        technique=technique_label,
        seed_name=seed_name,
        excursion_steps=eff_steps,
        snap_back_iters=snap_back_iters,
        snap_back_succeeded=True,
        final_count_out=final_co,
        excursion_max_count_out=excursion_max,
        walk=walk,
        walltime_seconds=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Technique 3 — boundary-restart seeds
# ---------------------------------------------------------------------------


def boundary_restart_seeds(
    seed: Matrix,
    *,
    k: int = 2,
    max_seeds: int = 20,
    rng_seed: int | None = None,
) -> list[Matrix]:
    """Generate up to ``max_seeds`` matrices that are k-flip neighbours of
    ``seed`` AND lie on the |G| ⊆ {1, 3} submanifold.

    For ``k == 2``, enumerate all C(n^2, 2) pairs of distinct flip positions,
    keep those that preserve the manifold. For ``k > 2``, sample randomly.

    The seed itself is excluded.
    """
    n = seed.n
    base = [row[:] for row in seed.entries]
    rng = random.Random(rng_seed)
    out: list[Matrix] = []

    if k == 2:
        positions = [(i, j) for i in range(n) for j in range(n)]
        # Shuffle to avoid biasing the chosen seeds spatially.
        rng.shuffle(positions)
        for ai in range(len(positions)):
            if len(out) >= max_seeds:
                break
            for bi in range(ai + 1, len(positions)):
                if len(out) >= max_seeds:
                    break
                p1 = positions[ai]
                p2 = positions[bi]
                cand = [row[:] for row in base]
                cand[p1[0]][p1[1]] = -cand[p1[0]][p1[1]]
                cand[p2[0]][p2[1]] = -cand[p2[0]][p2[1]]
                if is_on_manifold(cand, allowed_abs=(1, 3)):
                    out.append(Matrix(n=n, entries=cand))
        return out

    # k >= 3: random sampling.
    attempted = 0
    max_attempts = 50 * max_seeds
    while len(out) < max_seeds and attempted < max_attempts:
        attempted += 1
        flat_positions = rng.sample(range(n * n), k)
        cand = [row[:] for row in base]
        for p in flat_positions:
            ii, jj = divmod(p, n)
            cand[ii][jj] = -cand[ii][jj]
        if is_on_manifold(cand, allowed_abs=(1, 3)):
            out.append(Matrix(n=n, entries=cand))
    return out


# ---------------------------------------------------------------------------
# Triple-verification of any candidate above Orrick
# ---------------------------------------------------------------------------


def _triple_verify_abs_det(entries: Sequence[Sequence[int]]) -> int:
    """Run :func:`integer_determinant` twice on the matrix and once on its
    transpose, asserting all three agree, and return the absolute value."""
    M = [[int(v) for v in row] for row in entries]
    Mt = [[M[j][i] for j in range(len(M))] for i in range(len(M))]
    d1 = abs(integer_determinant(M))
    d2 = abs(integer_determinant(M))
    d3 = abs(integer_determinant(Mt))
    if not (d1 == d2 == d3):
        raise RuntimeError(
            f"triple-verification disagreement: d1={d1} d2={d2} d3(transpose)={d3}"
        )
    return d1


__all__ = [
    "AbsClassMultiset",
    "AbsClassState",
    "KnownClass",
    "WalkResult",
    "BridgeWalkResult",
    "gram_abs_class_signature",
    "is_on_manifold",
    "count_outside_abs_support",
    "enumerate_known_2val_classes",
    "is_in_known_class",
    "manifold_walk",
    "bridge_excursion",
    "boundary_restart_seeds",
    "_triple_verify_abs_det",
]
