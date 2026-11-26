"""Competitive Exclusion Niche SA for H23 maxdet.

Ecology analogy
---------------
Two ecological principles combine here:

1. **Competitive exclusion** (Gause 1934): two species occupying the same
   ecological niche cannot stably coexist — the fitter one drives the other
   to extinction.

2. **Island biogeography** (MacArthur & Wilson 1967): the equilibrium species
   richness of an island is set by the balance between immigration (new
   species arriving from the mainland) and local extinction. Island *area*
   determines carrying capacity: larger islands (more habitat resources)
   support more species at equilibrium, because local extinction rates fall
   with island size.  The classical species-area law S ~ A^z (z ≈ 0.25)
   captures this.

Mapping to H23
--------------
  Basin of attraction  ↔  Ecological niche
  SA chain             ↔  Colonising species attempting to establish
  Local maximum |det|  ↔  Island area / habitat quality
  Gram fingerprint     ↔  Niche identity (which basin)

Algorithm (CENS: Competitive Exclusion Niche Search)
-----------------------------------------------------
A *community* dict maps Gram fingerprint → NicheRecord.

At each step one of two strategies is sampled:

Exploration (probability 1 - exploit_prob):
  Start from a uniform random ±1 matrix.  Run SA.  Fingerprint the result.
  - New fingerprint → new niche colonised; add to community.
  - Known fingerprint, higher det → competitive displacement (newcomer wins).
  - Known fingerprint, lower det → competitive exclusion (newcomer lost).

Exploitation (probability exploit_prob):
  Select a niche champion weighted by |det|^island_exp (the island-size
  weighting, default island_exp=0.25 following the MacArthur-Wilson power
  law).  Perturb the champion by flipping k ∈ [k_lo, k_hi] random entries
  (escaping the local maximum while remaining in the basin neighbourhood).
  Run SA from the perturbed start; fingerprint; update community.

Key diagnostics:
  - niche_discovery_rate = distinct niches found / total chains
  - collision_rate       = chains landing in known niches / total chains
  - community size       = number of distinct basins in the community

Distinction from all prior iterations
--------------------------------------
  CEM (information_theory): averages entry-level Bernoullis over elites;
    no basin-identity tracking; no competitive exclusion.
  BRD (game_theory): single chain; no population; no niche structure.
  PT (thermodynamics): temperature-coupled replicas; no basin tracking.
  Hippocampal replay (neuroscience): buffer + row crossover; niches not
    tracked or used to guide starts.
  Quasispecies (evolutionary_biology): selection-mutation balance;
    population without basin-identity memory.
  CENS (ecology, this module): the community *explicitly tracks* which
    basins have been found and uses the island-area law to allocate
    exploitation budget proportionally to basin quality, while competitive
    exclusion ensures each niche is represented by its *best* known matrix
    rather than by any random incumbent.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant

N23 = 23
ORRICK_ABS_DET_H23: int = 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NicheRecord:
    """One entry in the community: best matrix found in a niche (basin)."""

    fingerprint: tuple[int, ...]
    abs_det: int
    matrix: list[list[int]]
    visits: int = 1


@dataclass
class NicheSAResult:
    """Outcome of :func:`niche_sa_search`."""

    best_abs_det: int
    best_matrix: list[list[int]]
    community_size: int
    total_chains: int
    exploration_chains: int
    exploitation_chains: int
    collisions: int
    new_niches: int
    elapsed_seconds: float
    ratio_to_orrick: float
    above_orrick: bool
    niche_det_histogram: list[int]


# ---------------------------------------------------------------------------
# Gram fingerprint
# ---------------------------------------------------------------------------


def gram_fingerprint(M: list[list[int]]) -> tuple[int, ...]:
    """Sorted tuple of all row-pair inner products — canonical niche ID.

    For a 23×23 ±1 matrix this yields 253 integers in {-23,-21,...,21,23}.
    Two matrices with the same fingerprint share the same multiset of
    pairwise row correlations — a reliable signature of basin identity after
    local-maximum convergence.
    """
    n = len(M)
    ips: list[int] = []
    for i in range(n):
        ri = M[i]
        for j in range(i + 1, n):
            rj = M[j]
            ips.append(sum(ri[k] * rj[k] for k in range(n)))
    ips.sort()
    return tuple(ips)


# ---------------------------------------------------------------------------
# Internal SA engine (single chain)
# ---------------------------------------------------------------------------


def _run_sa_chain(
    n: int,
    rng: random.Random,
    sa_iters: int,
    t_start: float,
    t_end: float,
    initial: list[list[int]] | None,
    greedy_iters: int,
    refresh_every: int = 2000,
) -> tuple[int, list[list[int]]]:
    """Run one SA chain; return (abs_det, best_matrix) as Python lists.

    Phase 1: SA with geometric cooling over sa_iters steps.
    Phase 2: greedy T=0 descent for greedy_iters steps to stabilise the
             local maximum so the Gram fingerprint is reliable.
    """
    # Initialise
    if initial is None:
        mat = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
    else:
        mat = [row[:] for row in initial]

    state = IncrementalState.from_matrix(mat)
    attempts = 0
    while state.det == 0 and attempts < 16:
        mat = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
        state = IncrementalState.from_matrix(mat)
        attempts += 1

    best_abs = abs(state.det)
    best_mat = [row[:] for row in state.M]
    current_log_det = math.log(max(best_abs, 1))

    log_ratio = math.log(t_end / t_start)
    accepted = 0

    # Phase 1: SA
    for step in range(sa_iters):
        # Guard: if state went singular (adjugate drift), restart fresh
        if state.det == 0:
            mat2 = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
            state = IncrementalState.from_matrix(mat2)
            current_log_det = math.log(max(abs(state.det), 1))
            if state.det == 0:
                continue

        frac = step / max(1, sa_iters - 1)
        t = t_start * math.exp(log_ratio * frac)

        i = rng.randrange(n)
        j = rng.randrange(n)
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            continue

        new_log = math.log(abs(new_det))
        delta = current_log_det - new_log  # positive = proposal worse
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            apply_entry_flip(state, i, j)
            current_log_det = new_log
            accepted += 1
            if abs(state.det) > best_abs:
                best_abs = abs(state.det)
                best_mat = [row[:] for row in state.M]
            if refresh_every > 0 and accepted % refresh_every == 0:
                refresh(state)
                if state.det == 0:
                    current_log_det = 0.0
                else:
                    current_log_det = math.log(abs(state.det))

    # Phase 2: greedy T=0 to converge to local maximum for stable fingerprint
    if state.det != 0:
        for _ in range(greedy_iters):
            improved = False
            for i in range(n):
                for j in range(n):
                    if state.det == 0:
                        break
                    new_det = propose_entry_flip(state, i, j)
                    if abs(new_det) > abs(state.det):
                        apply_entry_flip(state, i, j)
                        improved = True
            if not improved:
                break

    # Final best check
    final_abs = abs(state.det)
    if final_abs > best_abs:
        best_abs = final_abs
        best_mat = [row[:] for row in state.M]

    # Exact Bareiss verification
    exact = abs(integer_determinant(best_mat))
    return exact, best_mat


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------


def niche_sa_search(
    n: int = N23,
    budget_seconds: float = 600.0,
    sa_iters: int = 80_000,
    greedy_iters: int = 1,
    exploit_prob: float = 0.30,
    island_exp: float = 0.25,
    k_lo: int = 8,
    k_hi: int = 18,
    t_start: float = 2.0,
    t_end: float = 0.01,
    seed: int = 42,
) -> NicheSAResult:
    """Competitive Exclusion Niche Search.

    Parameters
    ----------
    n : matrix order (23 for H23).
    budget_seconds : wall-clock budget.
    sa_iters : SA steps per chain (phase 1).
    greedy_iters : greedy T=0 passes per chain (phase 2); 0 to disable.
    exploit_prob : probability of choosing exploitation over exploration.
    island_exp : MacArthur-Wilson island exponent for niche weighting.
    k_lo, k_hi : range for number of flips when perturbing a champion.
    t_start, t_end : SA temperature schedule endpoints.
    seed : random seed.
    """
    rng = random.Random(seed)
    community: dict[tuple[int, ...], NicheRecord] = {}
    global_best_det: int = 0
    global_best_mat: list[list[int]] = []

    total_chains = 0
    explore_chains = 0
    exploit_chains = 0
    collisions = 0
    new_niches = 0

    t0 = time.monotonic()

    while time.monotonic() - t0 < budget_seconds:
        # Choose exploration or exploitation
        if community and rng.random() < exploit_prob:
            # --- Exploitation: island-area weighted niche sampling ---
            fps = list(community.keys())
            weights = [community[fp].abs_det ** island_exp for fp in fps]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0.0
            chosen_fp = fps[-1]
            for fp, w in zip(fps, weights):
                cumulative += w
                if r < cumulative:
                    chosen_fp = fp
                    break

            champ = community[chosen_fp]
            # Perturb by flipping k random entries to escape the local max
            k = rng.randint(k_lo, k_hi)
            initial = [row[:] for row in champ.matrix]
            positions = [(i, j) for i in range(n) for j in range(n)]
            rng.shuffle(positions)
            for i, j in positions[:k]:
                initial[i][j] = -initial[i][j]

            abs_det, best_mat = _run_sa_chain(
                n, rng, sa_iters, t_start, t_end, initial, greedy_iters
            )
            community[chosen_fp].visits += 1
            exploit_chains += 1
        else:
            # --- Exploration: random start, discover new niches ---
            abs_det, best_mat = _run_sa_chain(
                n, rng, sa_iters, t_start, t_end, None, greedy_iters
            )
            explore_chains += 1

        total_chains += 1

        # Compute niche fingerprint of result
        fp = gram_fingerprint(best_mat)

        # Competitive exclusion update
        if fp in community:
            collisions += 1
            if abs_det > community[fp].abs_det:
                # Competitive displacement: newcomer beats resident
                community[fp].abs_det = abs_det
                community[fp].matrix = best_mat
        else:
            # New niche colonised
            new_niches += 1
            community[fp] = NicheRecord(
                fingerprint=fp,
                abs_det=abs_det,
                matrix=best_mat,
            )

        # Track global best
        if abs_det > global_best_det:
            global_best_det = abs_det
            global_best_mat = [row[:] for row in best_mat]

    elapsed = time.monotonic() - t0

    # Build det histogram across community (bucketed by ratio to Orrick)
    hist = [0] * 10
    for rec in community.values():
        bucket = min(9, int(rec.abs_det * 10 / ORRICK_ABS_DET_H23))
        hist[bucket] += 1

    return NicheSAResult(
        best_abs_det=global_best_det,
        best_matrix=global_best_mat,
        community_size=len(community),
        total_chains=total_chains,
        exploration_chains=explore_chains,
        exploitation_chains=exploit_chains,
        collisions=collisions,
        new_niches=new_niches,
        elapsed_seconds=elapsed,
        ratio_to_orrick=global_best_det / ORRICK_ABS_DET_H23,
        above_orrick=global_best_det > ORRICK_ABS_DET_H23,
        niche_det_histogram=hist,
    )


# ---------------------------------------------------------------------------
# Triple verification
# ---------------------------------------------------------------------------


def triple_verify(M: list[list[int]]) -> dict[str, int | bool]:
    """Verify |det(M)|, |det(M^T)|, and |det(copy)| are identical.

    Returns a dict with det, det_transpose, det_copy, consistent.
    """
    d1 = integer_determinant(M)
    Mt = [[M[j][i] for j in range(len(M))] for i in range(len(M))]
    d2 = integer_determinant(Mt)
    Mc = [row[:] for row in M]
    d3 = integer_determinant(Mc)
    return {
        "det": d1,
        "det_transpose": d2,
        "det_copy": d3,
        "consistent": (abs(d1) == abs(d2) == abs(d3)),
    }
