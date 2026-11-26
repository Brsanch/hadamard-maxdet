"""Evolutionary Biology: Quasispecies Error-Threshold Model for H23 maxdet.

Domain analogy — Eigen (1971) quasispecies model
-------------------------------------------------
Manfred Eigen's quasispecies theory (Eigen 1971; Eigen, McCaskill & Schuster 1988)
describes how populations of self-replicating molecules evolve on a fitness landscape
under selection and mutation. The central object is not a single optimal sequence, but
a *quasispecies*: a stationary distribution of mutant sequences around one or more
fitness peaks. Two regimes exist:

  • mu > mu_c  (above error threshold):
      Replication errors are too frequent; the quasispecies delocalises — the
      population drifts across the landscape without anchoring to any peak. This
      corresponds to the disordered paramagnetic phase in spin-glass language.

  • mu < mu_c  (below error threshold):
      The quasispecies localises near a fitness peak. The population cloud width
      is O(1 / ln(f_max/f_background)) entries from the peak. This is the
      condensed (spin-glass) phase for the H23 landscape.

  • mu ≈ mu_c  (at the error threshold):
      The quasispecies can diffuse across the landscape while maintaining weak
      anchoring near peaks — optimal for exploration.

For a sequence of length L (here L = n² = 529 ±1 entries) and peak-to-background
fitness ratio r = f_max/f_0, the error threshold is approximately
    mu_c ≈ ln(r) / L.

Algorithm
---------
We maintain a population of N_pop matrices. Each generation:

  1. Selection: sample N_pop offspring with replacement, proportional to
     fitness^alpha. This is the discrete analogue of Eigen's replication term.

  2. Mutation: each entry of each offspring flips independently with probability
     mu (copying error, analogous to the mutation matrix Q in the Eigen model).

  3. Fitness evaluation: compute |det(M)| exactly for each offspring using
     Bareiss integer arithmetic (no numpy overflow).

  4. Cooling: anneal mu from mu_0 (above error threshold) to mu_final (below),
     using a geometric schedule within each restart block.

  5. Restart: after restart_every generations, reinitialise the population from
     random ±1 matrices and reset mu = mu_0.  The global best is preserved
     across restarts.

Key distinctions from prior iterations
  - CEM (information_theory): learns a marginal entry-level distribution from
    elites; discards population dynamics entirely; no selection-mutation balance.
  - PT (thermodynamics): different-temperature replicas without explicit selection;
    population is not diverse (all replicas eventually lock to the same basin).
  - SQA (quantum_mechanics): imaginary-time coupling, no selection.
  - Hippocampal replay (neuroscience): buffer + crossover, not evolving population.
  - BRD (game_theory): greedy single-matrix ascent; no population.

The quasispecies approach is qualitatively different: selection pressure and
mutation rate jointly control the spread of the population cloud.  Near the error
threshold (intermediate mu), the cloud can straddle multiple basins — potentially
enabling inter-basin information transfer not accessible to any single-trajectory
method.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass

from hadamard_maxdet.verifier import integer_determinant

N23 = 23
ORRICK_ABS_DET_H23: int = 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# Result carrier
# ---------------------------------------------------------------------------


@dataclass
class QSResult:
    """Outcome of a quasispecies search run."""

    best_matrix: list[list[int]]
    best_abs_det: int
    n_generations: int
    n_restarts: int
    walltime_seconds: float
    best_ratio_to_orrick: float
    above_orrick: bool


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _random_pm1_matrix(n: int, rng: random.Random) -> list[list[int]]:
    return [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n)]


def _mutate(matrix: list[list[int]], mu: float, rng: random.Random) -> list[list[int]]:
    """Copy matrix and flip each entry with probability mu."""
    n = len(matrix)
    result = [row[:] for row in matrix]
    for i in range(n):
        for j in range(n):
            if rng.random() < mu:
                result[i][j] = -result[i][j]
    return result


def _weighted_sample_index(weights: list[float], total: float, rng: random.Random) -> int:
    """Sample one index proportional to weights (must sum to total)."""
    r = rng.random() * total
    cum = 0.0
    for k, w in enumerate(weights):
        cum += w
        if cum >= r:
            return k
    return len(weights) - 1


def triple_verify(matrix: list[list[int]]) -> tuple[int, int, int]:
    """Compute |det| three ways: M, M^T, deep copy. All must agree."""
    n = len(matrix)
    d1 = abs(integer_determinant(matrix))
    transp = [[matrix[j][i] for j in range(n)] for i in range(n)]
    d2 = abs(integer_determinant(transp))
    d3 = abs(integer_determinant(copy.deepcopy(matrix)))
    return d1, d2, d3


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def quasispecies_search(
    n: int = N23,
    pop_size: int = 20,
    mu_0: float = 0.25,
    mu_final: float = 0.001,
    alpha: float = 3.0,
    restart_every: int = 500,
    wall_time: float = 600.0,
    orrick: int = ORRICK_ABS_DET_H23,
    seed: int = 20260426,
    verbose: bool = True,
) -> QSResult:
    """Quasispecies error-threshold search for high-|det| n×n ±1 matrices.

    Parameters
    ----------
    n:
        Matrix size.
    pop_size:
        Number of individuals in the population.
    mu_0:
        Initial per-entry mutation probability (above error threshold — free exploration).
    mu_final:
        Final per-entry mutation probability (below error threshold — localised quasispecies).
    alpha:
        Selection pressure: fitness raised to alpha-th power before sampling.
        Higher alpha = stronger tournament pressure toward the best individual.
    restart_every:
        After this many generations, reinitialise the population from scratch
        (but preserve the global best).
    wall_time:
        Wall-clock budget in seconds.
    orrick:
        Reference |det| (Orrick 2003 record) for ratio computation.
    seed:
        RNG seed for reproducibility.
    verbose:
        Print progress messages.

    Returns
    -------
    QSResult
        Best matrix and diagnostics.
    """
    rng = random.Random(seed)
    t_start = time.time()

    best_matrix: list[list[int]] | None = None
    best_abs_det: int = 0
    n_generations: int = 0
    n_restarts: int = 0

    def _update_best(pop: list[list[list[int]]], fits: list[int]) -> None:
        nonlocal best_abs_det, best_matrix
        for m, f in zip(pop, fits):
            if f > best_abs_det:
                best_abs_det = f
                best_matrix = [row[:] for row in m]
                if verbose:
                    ratio = f / orrick
                    flag = " *** ABOVE ORRICK ***" if f > orrick else ""
                    print(
                        f"  [gen {n_generations}] New best: {f} = {ratio:.4f}×Orrick{flag}"
                    )

    # Theoretical error threshold (for diagnostic reporting only)
    # mu_c ≈ ln(r) / L where r = f_max/f_background, L = n²
    # Use r = orrick / (n/e)^(n/2) as crude estimate; just print mu_c range
    L = n * n

    def _compute_mu(gen_in_restart: int) -> float:
        frac = min(1.0, gen_in_restart / restart_every)
        return mu_0 * ((mu_final / mu_0) ** frac)

    # Initialise first population
    population = [_random_pm1_matrix(n, rng) for _ in range(pop_size)]
    fitnesses = [abs(integer_determinant(m)) for m in population]
    _update_best(population, fitnesses)
    gen_in_restart = 0

    if verbose:
        print(
            f"Quasispecies search: n={n}, pop_size={pop_size}, "
            f"mu_0={mu_0}, mu_final={mu_final}, alpha={alpha}, "
            f"restart_every={restart_every}, wall={wall_time}s"
        )
        print(f"Sequence length L={L}, mu_c estimate ≈ {math.log(10) / L:.5f} "
              f"(for r=10 peak/background)")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= wall_time:
            break

        mu = _compute_mu(gen_in_restart)

        # --- Selection ---
        # Compute weights = fitness^alpha; use integer exponentiation to stay exact,
        # but alpha may be float so convert to float weight.
        weights = [float(f) ** alpha for f in fitnesses]
        total_w = sum(weights)

        # --- Mutation + evaluation ---
        new_population: list[list[list[int]]] = []
        new_fitnesses: list[int] = []
        for _ in range(pop_size):
            parent_idx = _weighted_sample_index(weights, total_w, rng)
            offspring = _mutate(population[parent_idx], mu, rng)
            new_population.append(offspring)
            new_fitnesses.append(abs(integer_determinant(offspring)))

        population = new_population
        fitnesses = new_fitnesses
        n_generations += 1
        gen_in_restart += 1

        _update_best(population, fitnesses)

        # --- Restart check ---
        if gen_in_restart >= restart_every:
            n_restarts += 1
            if verbose:
                elapsed = time.time() - t_start
                mean_fit = sum(fitnesses) / pop_size
                std_fit = (
                    sum((f - mean_fit) ** 2 for f in fitnesses) / pop_size
                ) ** 0.5
                mu_now = _compute_mu(gen_in_restart)
                ratio = best_abs_det / orrick
                print(
                    f"Restart {n_restarts} after gen {n_generations} "
                    f"(t={elapsed:.1f}s): mu={mu_now:.5f}, "
                    f"best={best_abs_det} ({ratio:.4f}×Orrick), "
                    f"pop_mean={mean_fit:.0f} ± {std_fit:.0f}"
                )

            # Fresh population, reset mu schedule
            population = [_random_pm1_matrix(n, rng) for _ in range(pop_size)]
            fitnesses = [abs(integer_determinant(m)) for m in population]
            _update_best(population, fitnesses)
            gen_in_restart = 0

    walltime = time.time() - t_start

    if best_matrix is None:
        best_matrix = population[0]
        best_abs_det = fitnesses[0]

    above_orrick = best_abs_det > orrick
    ratio = best_abs_det / orrick

    return QSResult(
        best_matrix=best_matrix,
        best_abs_det=best_abs_det,
        n_generations=n_generations,
        n_restarts=n_restarts,
        walltime_seconds=walltime,
        best_ratio_to_orrick=ratio,
        above_orrick=above_orrick,
    )
