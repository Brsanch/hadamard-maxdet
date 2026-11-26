"""Smart 3-valued (|G| ⊆ {1, 3, 5}) submanifold search at H23.

Background
----------
At order n=23, twenty-one of the 24 ``H23_basin_*.json`` matrices have Gram
off-diag absolute values in ``{1, 3, 5}`` exactly (the other three include a
single |G|=7 outlier). Treat the multiset

    (count_|G|=1, count_|G|=3, count_|G|=5)

as the "class signature" on this larger submanifold. The hypothesis here:
restricting search to ``|G| ⊆ {1, 3, 5}`` (rather than the tighter
``|G| ⊆ {1, 3}`` of the 2-valued module) gives more degrees of freedom —
maybe more new equivalence classes, possibly even one with |det| > Orrick's
H23 record at ``2,779,447,296,000,000``.

Three search techniques, all adapted from :mod:`smart_2val_search`:

1. **Manifold drift with class tabu** — single-entry-flip SA restricted to
   ``|G| ⊆ {1, 3, 5}``, with the known-class signatures locked out as a tabu
   set.
2. **Bridge excursion** — leave the manifold for a few flips, snap back via
   ``E = -log|det| + μ · count_outside``, then run technique 1.
3. **Boundary restart** — k-flip neighbours of a known matrix that
   themselves preserve the manifold; run technique 1 from each.

Public API
----------

* :func:`gram_abs_class_signature_3val` — multiset
  ``((1, c1), (3, c3), (5, c5))`` (zero counts dropped). Reuses the
  signature function from :mod:`smart_2val_search` but documents the
  3-valued reading.
* :func:`is_on_manifold_3val` — every off-diag has |G| ∈ {1, 3, 5}.
* :func:`enumerate_known_3val_classes` — load all 24 ``H23_basin_*.json``
  files, filter to those that are 3-valued, return as ``KnownClass3``.
* :class:`AbsClass3State` — incremental O(n)-per-flip tracker for
  ``(count_1, count_3, count_5, count_out)``.
* :func:`manifold_walk_3val` — Technique 1.
* :func:`bridge_excursion_3val` — Technique 2.
* :func:`boundary_restart_seeds_3val` — generate seeds for Technique 3.
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
from hadamard_maxdet.smart_2val_search import (
    AbsClassMultiset,
    gram_abs_class_signature,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# 3-valued signature, manifold predicate, count-outside
# ---------------------------------------------------------------------------


def gram_abs_class_signature_3val(M: Sequence[Sequence[int]]) -> AbsClassMultiset:
    """Multiset of off-diag |G_ij| values, sorted; same as the 2-valued
    version but read in the 3-valued context. For matrices on the
    ``|G| ⊆ {1, 3, 5}`` submanifold, returns at most three entries
    ``((1, c1), (3, c3), (5, c5))``."""
    return gram_abs_class_signature(M)


def is_on_manifold_3val(
    M: Sequence[Sequence[int]],
    *,
    allowed_abs: Iterable[int] = (1, 3, 5),
) -> bool:
    """Return True iff every off-diagonal Gram entry has ``|G_ij| ∈ allowed_abs``."""
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


def count_outside_abs_support_3val(
    M: Sequence[Sequence[int]],
    *,
    allowed_abs: Iterable[int] = (1, 3, 5),
) -> int:
    """Number of unordered off-diag pairs with ``|G_ij| not in allowed_abs``."""
    allowed = set(int(a) for a in allowed_abs)
    A = np.asarray(M, dtype=np.int64)
    n = A.shape[0]
    if n < 2:
        return 0
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    return int(sum(1 for v in G[iu].tolist() if int(abs(int(v))) not in allowed))


# ---------------------------------------------------------------------------
# Known 3-valued classes (loaded from results/H23_basin_*.json)
# ---------------------------------------------------------------------------


@dataclass
class KnownClass3:
    """Representative matrix for a known |G| ⊆ {1, 3, 5} class at H23."""

    name: str
    matrix: Matrix
    abs_det: int
    abs_class_multiset: AbsClassMultiset


def enumerate_known_3val_classes(
    results_dir: str | Path = "results",
    *,
    pattern: str = "H23_basin_*.json",
    require_3val: bool = True,
) -> list[KnownClass3]:
    """Load every ``H23_basin_*.json`` and return the 3-valued ones.

    Each file uses the key ``matrix_entries`` (per the basin atlas writer)
    rather than ``entries``. ``abs_det`` is recomputed from scratch and
    cross-checked against the file.

    With ``require_3val=True`` (default), files whose Gram support strays
    outside ``{1, 3, 5}`` (e.g. basins that include a |G|=7 outlier) are
    silently skipped — they are not on the submanifold of interest.

    The returned list is sorted by descending ``abs_det``.
    """
    base = Path(results_dir)
    out: list[KnownClass3] = []
    files = sorted(base.glob(pattern))
    for path in files:
        with path.open() as f:
            data = json.load(f)
        # Basin files use 'matrix_entries'; some legacy files might use 'entries'.
        if "matrix_entries" in data:
            entries = [list(row) for row in data["matrix_entries"]]
        elif "entries" in data:
            entries = [list(row) for row in data["entries"]]
        else:
            continue
        n = len(entries)
        if n != 23:
            continue
        for r in entries:
            if len(r) != n:
                raise ValueError(f"{path}: matrix is not square")
            for v in r:
                if v not in (-1, 1):
                    raise ValueError(f"{path}: non ±1 entry {v!r}")
        if require_3val and not is_on_manifold_3val(entries, allowed_abs=(1, 3, 5)):
            continue
        sig = gram_abs_class_signature(entries)
        abs_det = abs(integer_determinant(entries))
        if "abs_det" in data and abs_det != int(data["abs_det"]):
            raise ValueError(
                f"{path}: abs_det field {data['abs_det']} disagrees with "
                f"recomputed {abs_det}"
            )
        out.append(
            KnownClass3(
                name=path.stem,
                matrix=Matrix(n=n, entries=entries),
                abs_det=abs_det,
                abs_class_multiset=sig,
            )
        )
    out.sort(key=lambda k: -k.abs_det)
    return out


def is_in_known_class_3val(
    M: Sequence[Sequence[int]],
    locked_multisets: Iterable[AbsClassMultiset],
) -> bool:
    """Return True iff ``gram_abs_class_signature(M)`` is in ``locked_multisets``."""
    sig = gram_abs_class_signature(M)
    return sig in set(locked_multisets)


# ---------------------------------------------------------------------------
# Incremental |G| ⊆ {1, 3, 5} class state with O(n) per-flip updates
# ---------------------------------------------------------------------------


@dataclass
class AbsClass3State:
    """Tracks (count_abs1, count_abs3, count_abs5, count_out) under flips.

    Invariants (maintained by :meth:`apply_after_flip`):

      * ``G == M @ M.T``  (numpy int (n, n))
      * ``count_abs1 == #{(i, j) : i < j, |G[i, j]| == 1}``
      * ``count_abs3 == #{(i, j) : i < j, |G[i, j]| == 3}``
      * ``count_abs5 == #{(i, j) : i < j, |G[i, j]| == 5}``
      * ``count_out  == n(n-1)/2 - count_abs1 - count_abs3 - count_abs5``
    """

    n: int
    G: np.ndarray
    count_abs1: int
    count_abs3: int
    count_abs5: int
    count_out: int

    @classmethod
    def from_matrix(cls, M: Sequence[Sequence[int]]) -> "AbsClass3State":
        A = np.asarray(M, dtype=np.int64)
        n = A.shape[0]
        if n < 2:
            return cls(
                n=n,
                G=A @ A.T if n else np.zeros((0, 0), dtype=np.int64),
                count_abs1=0,
                count_abs3=0,
                count_abs5=0,
                count_out=0,
            )
        G = (A @ A.T).astype(np.int64)
        iu = np.triu_indices(n, k=1)
        vals = np.abs(G[iu])
        c1 = int(np.sum(vals == 1))
        c3 = int(np.sum(vals == 3))
        c5 = int(np.sum(vals == 5))
        total = n * (n - 1) // 2
        co = total - c1 - c3 - c5
        return cls(
            n=n,
            G=G,
            count_abs1=c1,
            count_abs3=c3,
            count_abs5=c5,
            count_out=co,
        )

    def deltas_for_flip(
        self,
        M_after: Sequence[Sequence[int]],
        i: int,
        c: int,
    ) -> tuple[int, int, int, int]:
        """Return (d_abs1, d_abs3, d_abs5, d_out) for flipping entry (i, c).

        ``M_after[i][c]`` is the *new* value; other ``M_after`` entries are
        unchanged. Cost O(n). Does NOT mutate the state.
        """
        n = self.n
        if n < 2:
            return 0, 0, 0, 0
        new_val = int(M_after[i][c])
        delta_factor = 2 * new_val
        Gi_old = self.G[i]
        d1 = 0
        d3 = 0
        d5 = 0
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
            elif ao == 5:
                d5 -= 1
            if an == 1:
                d1 += 1
            elif an == 3:
                d3 += 1
            elif an == 5:
                d5 += 1
        d_out = -(d1 + d3 + d5)
        return d1, d3, d5, d_out

    def apply_after_flip(
        self,
        M_after: Sequence[Sequence[int]],
        i: int,
        c: int,
    ) -> None:
        """Apply the deltas from a just-completed flip. Cost O(n)."""
        n = self.n
        if n < 2:
            return
        new_val = int(M_after[i][c])
        delta_factor = 2 * new_val
        Gi = self.G[i]
        d1 = 0
        d3 = 0
        d5 = 0
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
                    elif ao == 5:
                        d5 -= 1
                    if an == 1:
                        d1 += 1
                    elif an == 3:
                        d3 += 1
                    elif an == 5:
                        d5 += 1
                Gi[r] = new
                self.G[r, i] = new
        self.count_abs1 += d1
        self.count_abs3 += d3
        self.count_abs5 += d5
        self.count_out -= (d1 + d3 + d5)


# ---------------------------------------------------------------------------
# Helper — build a 3-valued signature multiset from counts.
# ---------------------------------------------------------------------------


def _abs_signature_from_counts_3val(c1: int, c3: int, c5: int) -> AbsClassMultiset:
    """Build the multiset signature from (c1, c3, c5) when ``count_out == 0``."""
    parts: list[tuple[int, int]] = []
    if c1 > 0:
        parts.append((1, c1))
    if c3 > 0:
        parts.append((3, c3))
    if c5 > 0:
        parts.append((5, c5))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Technique 1 — manifold drift with class tabu (3-valued)
# ---------------------------------------------------------------------------


@dataclass
class WalkResult3:
    """Outcome of :func:`manifold_walk_3val`."""

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


def manifold_walk_3val(
    seed: Matrix,
    *,
    iterations: int = 60_000,
    locked_multisets: Iterable[AbsClassMultiset] = (),
    t_start: float = 0.5,
    t_end: float = 0.005,
    rng_seed: int | None = None,
    refresh_every: int = 2000,
    technique_label: str = "manifold_drift_3val",
    seed_name: str = "seed",
    target_abs5: int | None = None,
    abs5_target_bias: float = 0.0,
    target_abs3: int | None = None,
    abs3_target_bias: float = 0.0,
) -> WalkResult3:
    """Single-entry-flip SA restricted to the |G| ⊆ {1, 3, 5} submanifold.

    Energy: ``E = -log|det| + λ₃·(count_3 − target_3)² + λ₅·(count_5 − target_5)²``
    where the off-manifold rejection is hard (not a soft penalty).

    Class tabu: any proposal whose new signature lies in ``locked_multisets``
    is rejected.
    """
    rng = random.Random(rng_seed)
    n = seed.n
    locked = set(locked_multisets)
    state = IncrementalState.from_matrix(seed.entries)
    if state.det == 0:
        return WalkResult3(
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

    abs_state = AbsClass3State.from_matrix(state.M)
    on_manifold = abs_state.count_out == 0
    cur_e = -math.log(abs(state.det))
    if target_abs3 is not None:
        cur_e += abs3_target_bias * (abs_state.count_abs3 - target_abs3) ** 2
    if target_abs5 is not None:
        cur_e += abs5_target_bias * (abs_state.count_abs5 - target_abs5) ** 2

    feasible_steps = int(on_manifold)
    accepted = 0
    rejected_class = 0
    rejected_metro = 0
    rejected_singular = 0
    rejected_off_manifold = 0

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
        d1, d3, d5, dout = abs_state.deltas_for_flip(state.M, i, c)
        new_count_out = abs_state.count_out + dout
        if new_count_out != 0:
            state.M[i][c] = old_val
            rejected_off_manifold += 1
            continue

        new_count_abs1 = abs_state.count_abs1 + d1
        new_count_abs3 = abs_state.count_abs3 + d3
        new_count_abs5 = abs_state.count_abs5 + d5
        new_sig = _abs_signature_from_counts_3val(
            new_count_abs1, new_count_abs3, new_count_abs5
        )
        if new_sig in locked:
            state.M[i][c] = old_val
            rejected_class += 1
            continue

        prop_e = -math.log(abs(new_det))
        if target_abs3 is not None:
            prop_e += abs3_target_bias * (new_count_abs3 - target_abs3) ** 2
        if target_abs5 is not None:
            prop_e += abs5_target_bias * (new_count_abs5 - target_abs5) ** 2
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
                    cur_e += abs3_target_bias * (
                        abs_state.count_abs3 - target_abs3
                    ) ** 2
                if target_abs5 is not None:
                    cur_e += abs5_target_bias * (
                        abs_state.count_abs5 - target_abs5
                    ) ** 2
            abs_state = AbsClass3State.from_matrix(state.M)

    walltime = time.monotonic() - t0
    final_sig = gram_abs_class_signature(state.M)
    visited_pairs = sorted(visited.items(), key=lambda kv: -kv[1])
    return WalkResult3(
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


# ---------------------------------------------------------------------------
# Technique 2 — bridge excursion (3-valued)
# ---------------------------------------------------------------------------


@dataclass
class BridgeWalkResult3:
    """Outcome of :func:`bridge_excursion_3val`."""

    technique: str
    seed_name: str
    excursion_steps: int
    snap_back_iters: int
    snap_back_succeeded: bool
    final_count_out: int
    excursion_max_count_out: int
    walk: WalkResult3 | None
    walltime_seconds: float


def bridge_excursion_3val(
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
    technique_label: str = "bridge_excursion_3val",
) -> BridgeWalkResult3:
    """Off-manifold excursion (|G| ⊆ {1, 3, 5}), then snap back, then walk."""
    rng = random.Random(rng_seed)
    n = seed.n
    t0 = time.monotonic()

    # Step 1 — excursion.
    state = IncrementalState.from_matrix(seed.entries)
    abs_state = AbsClass3State.from_matrix(state.M)
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
        _d1, _d3, _d5, dout = abs_state.deltas_for_flip(state.M, i, c)
        new_count_out = abs_state.count_out + dout
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
        return BridgeWalkResult3(
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

    snapped = Matrix(n=n, entries=[row[:] for row in state.M])
    walk = manifold_walk_3val(
        snapped,
        iterations=walk_iters,
        locked_multisets=locked_multisets,
        t_start=walk_t_start,
        t_end=walk_t_end,
        rng_seed=(None if rng_seed is None else rng_seed + 7),
        technique_label=technique_label,
        seed_name=seed_name + ".bridge",
    )
    return BridgeWalkResult3(
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
# Technique 3 — boundary-restart seeds (3-valued)
# ---------------------------------------------------------------------------


def boundary_restart_seeds_3val(
    seed: Matrix,
    *,
    k: int = 2,
    max_seeds: int = 20,
    rng_seed: int | None = None,
) -> list[Matrix]:
    """Generate up to ``max_seeds`` matrices that are k-flip neighbours of
    ``seed`` AND lie on the |G| ⊆ {1, 3, 5} submanifold.

    For ``k == 2``, enumerate pairs (after a shuffle for spatial mixing).
    For ``k > 2``, sample randomly. The seed itself is excluded.
    """
    n = seed.n
    base = [row[:] for row in seed.entries]
    rng = random.Random(rng_seed)
    out: list[Matrix] = []

    if k == 2:
        positions = [(i, j) for i in range(n) for j in range(n)]
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
                if is_on_manifold_3val(cand, allowed_abs=(1, 3, 5)):
                    out.append(Matrix(n=n, entries=cand))
        return out

    attempted = 0
    max_attempts = 50 * max_seeds
    while len(out) < max_seeds and attempted < max_attempts:
        attempted += 1
        flat_positions = rng.sample(range(n * n), k)
        cand = [row[:] for row in base]
        for p in flat_positions:
            ii, jj = divmod(p, n)
            cand[ii][jj] = -cand[ii][jj]
        if is_on_manifold_3val(cand, allowed_abs=(1, 3, 5)):
            out.append(Matrix(n=n, entries=cand))
    return out


# ---------------------------------------------------------------------------
# Triple verification
# ---------------------------------------------------------------------------


def _triple_verify_abs_det_3val(entries: Sequence[Sequence[int]]) -> int:
    """Twice on M, once on Mᵀ; assert all three agree; return |det|."""
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
    "AbsClass3State",
    "KnownClass3",
    "WalkResult3",
    "BridgeWalkResult3",
    "gram_abs_class_signature_3val",
    "is_on_manifold_3val",
    "count_outside_abs_support_3val",
    "enumerate_known_3val_classes",
    "is_in_known_class_3val",
    "manifold_walk_3val",
    "bridge_excursion_3val",
    "boundary_restart_seeds_3val",
    "_triple_verify_abs_det_3val",
    "_abs_signature_from_counts_3val",
]
