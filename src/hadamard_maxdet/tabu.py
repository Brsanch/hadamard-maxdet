"""Tabu-search wrapper around incremental SA.

Maintains a list of recently-visited canonical fingerprints (Gram
off-diagonal multiset only — fast O(n^2) to compute, identifies the
*equivalence-class signature* of the current state up to row/col sign
+ permutation). When the search proposes a move that would land back
in a tabu fingerprint, the move is rejected unconditionally even if
its delta is favourable.

This is a complementary mechanism to SA's Boltzmann acceptance: SA
accepts based on energy; tabu rejects based on *recently-visited
fingerprint*, breaking the symmetry that lets the chain return to a
recently-escaped basin.

Cost: O(n^2) per move to compute the off-diag fingerprint; small
constant compared to a Bareiss recompute. Tabu list bounded by ``capacity``;
LRU eviction.
"""
from __future__ import annotations

import math
import random
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
    _compute_adjugate,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


def fingerprint_offdiag(entries: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """O(n^2) sorted Gram off-diagonal multiset (canonical sign-and-perm
    signature of the matrix).
    """
    n = len(entries)
    M = np.array(entries, dtype=np.int64)
    G = M @ M.T
    off = []
    for i in range(n):
        for j in range(i + 1, n):
            off.append(int(G[i, j]))
    off.sort()
    return tuple(off)


@dataclass
class TabuList:
    """LRU tabu list of fingerprints with bounded capacity."""

    capacity: int = 1024
    items: OrderedDict[tuple, int] = field(default_factory=OrderedDict)

    def __contains__(self, fp: tuple) -> bool:
        return fp in self.items

    def add(self, fp: tuple, step: int) -> None:
        if fp in self.items:
            self.items.move_to_end(fp)
            self.items[fp] = step
            return
        self.items[fp] = step
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class TabuResult:
    matrix: Matrix
    abs_det: int
    iterations: int
    accepted: int
    rejected_by_tabu: int
    rejected_by_metropolis: int
    walltime_seconds: float
    tabu_size_final: int
    fingerprints_seen: int
    history_top: list[tuple[int, int]] = field(default_factory=list)


def sa_tabu(
    n: int,
    *,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int | None = None,
    initial: Matrix | None = None,
    tabu_capacity: int = 4096,
    fingerprint_every: int = 1,
    refresh_every: int = 2000,
    pre_seed_fingerprints: Sequence[tuple] = (),
) -> TabuResult:
    """SA with single-entry-flip moves + tabu list on Gram fingerprints.

    Parameters
    ----------
    pre_seed_fingerprints
        Fingerprints to populate the tabu list with at start (e.g., the
        BTC seed's fingerprint, to push the search away from it).
    fingerprint_every
        Compute fingerprint every k accepted moves; setting k>1 amortises
        the O(n^2) cost.
    """
    rng = random.Random(seed)
    if initial is None:
        mat = Matrix.random(n, rng)
    else:
        mat = initial.copy()

    state = IncrementalState.from_matrix(mat.entries)
    tries = 0
    while state.det == 0 and tries < 16:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        tries += 1
    if state.det == 0:
        return TabuResult(
            matrix=Matrix(n=n, entries=[row[:] for row in state.M]),
            abs_det=0, iterations=0, accepted=0,
            rejected_by_tabu=0, rejected_by_metropolis=0,
            walltime_seconds=0.0, tabu_size_final=0, fingerprints_seen=0,
        )

    tabu = TabuList(capacity=tabu_capacity)
    for fp in pre_seed_fingerprints:
        tabu.add(fp, -1)

    cur_e = -math.log(abs(state.det))
    best_entries = [row[:] for row in state.M]
    best_e = cur_e

    accepted = rejected_tabu = rejected_metro = 0
    fps_seen = 0
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    t0 = time.monotonic()
    history_top: list[tuple[int, int]] = []
    last_fp_check = 0

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)
        i = rng.randrange(n)
        j = rng.randrange(n)
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            continue
        prop_e = -math.log(abs(new_det))
        delta = prop_e - cur_e

        # Metropolis test
        if delta > 0 and rng.random() >= math.exp(-delta / max(t, 1e-12)):
            rejected_metro += 1
            continue

        # Apply (provisionally), check tabu, possibly revert
        old_val = state.M[i][j]
        apply_entry_flip(state, i, j)
        cur_e = prop_e
        accepted += 1
        if refresh_every > 0 and accepted % refresh_every == 0:
            refresh(state)

        if accepted - last_fp_check >= fingerprint_every:
            last_fp_check = accepted
            fp = fingerprint_offdiag(state.M)
            fps_seen += 1
            if fp in tabu:
                # Tabu hit: revert
                state.M[i][j] = old_val
                # det/adj revert: easier to just refresh
                refresh(state)
                cur_e = -math.log(abs(state.det)) if state.det else float("inf")
                rejected_tabu += 1
                accepted -= 1
                continue
            tabu.add(fp, step)

        if cur_e < best_e:
            best_e = cur_e
            best_entries = [row[:] for row in state.M]
            history_top.append((step, abs(integer_determinant(best_entries))))

    walltime = time.monotonic() - t0
    best_abs_det = abs(integer_determinant(best_entries))
    return TabuResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=iterations,
        accepted=accepted,
        rejected_by_tabu=rejected_tabu,
        rejected_by_metropolis=rejected_metro,
        walltime_seconds=walltime,
        tabu_size_final=len(tabu),
        fingerprints_seen=fps_seen,
        history_top=history_top,
    )
