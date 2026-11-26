"""Parallel tempering (Replica Exchange Monte Carlo) for H23 maxdet.

Thermodynamic analogy: the ±1 matrix ensemble at inverse-temperature β has
Gibbs measure P(M) ∝ exp(−β·E(M)) where E(M) = −log|det(M)| is the energy.
The H23 landscape is a 1-RSB spin glass (confirmed by cavity analysis): it has
O(28) near-Orrick basins separated by barriers, inaccessible to single-chain SA
from random starts.

Parallel tempering (Geyer 1991; Hukushima & Nemoto 1996) runs R replicas at
temperatures T_1 > T_2 > ... > T_R simultaneously. After each epoch of local
SA steps, adjacent replicas attempt to SWAP their configurations under the
Metropolis criterion:

    P(swap i↔j) = min(1, exp((β_i − β_j)(E_i − E_j)))

where β_k = 1/T_k and E_k = −log|det(M_k)|.

Physical intuition: the high-T replicas (small β) are in the "paramagnetic"
phase, mixing freely across basin boundaries. When a hot replica visits a
low-energy basin, the swap mechanism propagates that configuration toward the
cold end where SA refines it. This is the algorithmic analog of simulated
quantum annealing tunneling: hot replicas tunnel through barriers; swaps
propagate discoveries downward.

Swap acceptance derivation:
  Before swap: Boltzmann factor ∝ exp(−β_i E_i − β_j E_j)
  After swap:  Boltzmann factor ∝ exp(−β_i E_j − β_j E_i)
  Detailed balance → accept ratio = exp((β_j−β_i)(E_j−E_i)) / same_reversed
    = exp((β_i−β_j)(E_i−E_j))

This satisfies detailed balance at each temperature simultaneously, so each
replica converges to the correct Gibbs measure at its temperature.
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

# Default temperature ladder: 5 replicas spanning [T_hot, T_cold]
# Chosen to span the SA schedule used by prior iterations (t_start=2.0, t_end≈0.003)
DEFAULT_TEMPS: list[float] = [2.0, 0.5, 0.1, 0.02, 0.003]


# ---------------------------------------------------------------------------
# Result carrier
# ---------------------------------------------------------------------------


@dataclass
class PTResult:
    """Outcome of a parallel tempering run."""

    best_matrix: list[list[int]]
    best_abs_det: int
    total_sa_steps: int
    total_swap_attempts: int
    swaps_accepted: int
    epochs_completed: int
    walltime_seconds: float
    swap_rates_by_pair: list[float]
    temps_used: list[float]
    best_ratio_to_orrick: float
    # Per-replica best |det| at end of run (indexed hot→cold)
    per_replica_best: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _random_nonsingular_state(n: int, rng: random.Random) -> IncrementalState:
    """Return a nonsingular random ±1 IncrementalState of order n."""
    for _ in range(200):
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        if state.det != 0:
            return state
    raise RuntimeError("Could not sample a nonsingular matrix in 200 tries")


def _sa_epoch(
    state: IncrementalState,
    n: int,
    n_steps: int,
    t: float,
    rng: random.Random,
    counter: list[int],
    refresh_interval: int = 2000,
) -> None:
    """Run n_steps SA steps in-place on state at temperature t.

    counter is a mutable single-element list tracking accepted flips since
    the last refresh; it follows the physical replica across temperature swaps.
    """
    inv_t = 1.0 / max(t, 1e-12)
    for _ in range(n_steps):
        i = rng.randrange(n)
        j = rng.randrange(n)
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            continue
        new_abs = abs(new_det)
        old_abs = abs(state.det)
        if new_abs >= old_abs:
            apply_entry_flip(state, i, j)
            counter[0] += 1
        else:
            # Metropolis: accept with prob exp(log(new/old) / T) = (new/old)^(1/T)
            log_ratio = math.log(new_abs) - math.log(old_abs)  # < 0
            if rng.random() < math.exp(log_ratio * inv_t):
                apply_entry_flip(state, i, j)
                counter[0] += 1
        if counter[0] >= refresh_interval:
            refresh(state)
            counter[0] = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parallel_tempering(
    n: int = N23,
    temps: list[float] | None = None,
    steps_per_epoch: int = 500,
    wall_cap_s: float = 600.0,
    seed: int | None = None,
    verbose: bool = False,
    verbose_interval_s: float = 30.0,
) -> PTResult:
    """Parallel tempering for the ±1 matrix maxdet problem.

    Maintains R = len(temps) replicas at fixed temperatures. Each epoch:
      1. Each replica executes steps_per_epoch SA steps at its temperature.
      2. Adjacent replica pairs attempt configuration swaps using the
         Metropolis criterion for detailed balance.
    Swap parity alternates (even/odd pairs) to reduce correlations.

    Parameters
    ----------
    n : matrix order (23 for H23)
    temps : temperature list, high-to-low; defaults to DEFAULT_TEMPS
    steps_per_epoch : SA steps per replica per epoch
    wall_cap_s : wall-clock time limit in seconds
    seed : PRNG seed for reproducibility
    verbose : if True, print epoch progress to stdout
    verbose_interval_s : seconds between verbose prints

    Returns
    -------
    PTResult with best matrix, diagnostics, and per-pair swap acceptance rates.
    """
    rng = random.Random(seed)

    if temps is None:
        temps = list(DEFAULT_TEMPS)
    n_replicas = len(temps)
    # Sorted high-to-low; temps[0] is hottest, temps[-1] coldest
    temps = sorted(temps, reverse=True)
    betas = [1.0 / t for t in temps]

    # Initialise one IncrementalState per temperature slot (hot to cold)
    replicas: list[IncrementalState] = [
        _random_nonsingular_state(n, rng) for _ in range(n_replicas)
    ]
    # Refresh counter per replica: follows the physical replica across swaps
    counters: list[list[int]] = [[0] for _ in range(n_replicas)]

    # Global best tracking
    best_abs_det = max(abs(r.det) for r in replicas)
    best_matrix: list[list[int]] = []
    for r in replicas:
        if abs(r.det) == best_abs_det:
            best_matrix = [row[:] for row in r.M]
            break

    swap_attempts = [0] * (n_replicas - 1)
    swap_accepted_count = [0] * (n_replicas - 1)
    total_sa_steps = 0
    epochs = 0
    last_verbose = 0.0

    start_t = time.monotonic()

    while time.monotonic() - start_t < wall_cap_s:
        # --- SA phase: each replica runs steps_per_epoch steps ---
        for k in range(n_replicas):
            _sa_epoch(
                replicas[k], n, steps_per_epoch, temps[k], rng, counters[k]
            )
            total_sa_steps += steps_per_epoch
            d = abs(replicas[k].det)
            if d > best_abs_det:
                best_abs_det = d
                best_matrix = [row[:] for row in replicas[k].M]

        # --- Swap phase: attempt adjacent pair swaps (alternating parity) ---
        parity = epochs % 2
        for offset in (parity, 1 - parity):
            for k in range(offset, n_replicas - 1, 2):
                ri, rj = replicas[k], replicas[k + 1]
                # temps[k] > temps[k+1], so betas[k] < betas[k+1]
                abs_i = abs(ri.det)
                abs_j = abs(rj.det)
                if abs_i == 0 or abs_j == 0:
                    continue
                E_i = -math.log(abs_i)   # hot replica energy
                E_j = -math.log(abs_j)   # cold replica energy
                beta_i, beta_j = betas[k], betas[k + 1]
                # Detailed-balance Metropolis for swap:
                # delta_S = (β_i − β_j)(E_i − E_j)
                # β_i < β_j → (β_i − β_j) < 0
                # Typical: E_i > E_j (hot worse) → (E_i − E_j) > 0 → product < 0
                delta_s = (beta_i - beta_j) * (E_i - E_j)
                swap_attempts[k] += 1
                if delta_s >= 0.0 or rng.random() < math.exp(delta_s):
                    # Swap configurations between temperature slots k and k+1
                    replicas[k], replicas[k + 1] = replicas[k + 1], replicas[k]
                    counters[k], counters[k + 1] = counters[k + 1], counters[k]
                    swap_accepted_count[k] += 1

        epochs += 1

        if verbose:
            now = time.monotonic() - start_t
            if now - last_verbose >= verbose_interval_s:
                ratio = best_abs_det / ORRICK_ABS_DET_H23
                rates = [
                    swap_accepted_count[k] / max(1, swap_attempts[k])
                    for k in range(n_replicas - 1)
                ]
                print(
                    f"  t={now:.0f}s  epoch={epochs}  steps={total_sa_steps:,}"
                    f"  best={best_abs_det} ({ratio:.4f}×Orrick)"
                    f"  swap_rates={[f'{r:.2f}' for r in rates]}"
                )
                last_verbose = now

    wall = time.monotonic() - start_t

    # Final refresh on all replicas; update best
    per_replica_best = []
    for r in replicas:
        refresh(r)
        d = abs(r.det)
        per_replica_best.append(d)
        if d > best_abs_det:
            best_abs_det = d
            best_matrix = [row[:] for row in r.M]

    # Final exact verification with integer_determinant
    if best_matrix:
        exact_abs = abs(integer_determinant(best_matrix))
        best_abs_det = exact_abs

    swap_rates = [
        swap_accepted_count[k] / max(1, swap_attempts[k])
        for k in range(n_replicas - 1)
    ]

    return PTResult(
        best_matrix=best_matrix,
        best_abs_det=best_abs_det,
        total_sa_steps=total_sa_steps,
        total_swap_attempts=sum(swap_attempts),
        swaps_accepted=sum(swap_accepted_count),
        epochs_completed=epochs,
        walltime_seconds=wall,
        swap_rates_by_pair=swap_rates,
        temps_used=temps,
        best_ratio_to_orrick=best_abs_det / ORRICK_ABS_DET_H23,
        per_replica_best=per_replica_best,
    )


def triple_verify(M: list[list[int]]) -> tuple[int, int, int]:
    """Return (det_M, det_MT, det_from_copy).

    All three should equal the same value (up to sign for det_M vs det_MT).
    Raises ValueError if M is not square or entries are not ±1.
    """
    n = len(M)
    if any(len(row) != n for row in M):
        raise ValueError("Matrix is not square")
    if any(v not in (-1, 1) for row in M for v in row):
        raise ValueError("Matrix has non-±1 entries")

    MT = [[M[j][i] for j in range(n)] for i in range(n)]
    M_copy = [row[:] for row in M]

    det_M = integer_determinant(M)
    det_MT = integer_determinant(MT)
    det_copy = integer_determinant(M_copy)

    return det_M, det_MT, det_copy
