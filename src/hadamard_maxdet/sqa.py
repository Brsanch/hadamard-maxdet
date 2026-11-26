"""Simulated Quantum Annealing (SQA) via Suzuki-Trotter path-integral MC.

Domain analogy: quantum mechanics.

In quantum annealing a transverse magnetic field Γ drives quantum
fluctuations (tunneling) that allow the system to cross energy barriers
classically unreachable via thermal fluctuations alone.  The
Suzuki-Trotter (1976) decomposition maps the d-dimensional quantum Ising
model to a (d+1)-dimensional classical Ising model by introducing P
imaginary-time "Trotter replicas" arranged in a ring.  Spins in adjacent
replicas are ferromagnetically coupled by an imaginary-time coupling
J_⊥(Γ, T, P).

For the ±1 matrix maxdet problem the mapping is:
  - P replicas, each a ±1 matrix M_0, …, M_{P-1} arranged in a ring.
  - Classical energy per replica: E_cl(M_p) = −log|det(M_p)|
  - Quantum coupling between adjacent replicas p and p+1 (mod P):
        E_q = −J_⊥ · ∑_{i,j} M_p[i,j] · M_{p+1}[i,j]
    J_⊥ = −(T/2) · log(tanh(Γ / (P·T)))       (Suzuki-Trotter formula)
    J_⊥ > 0 for Γ > 0 and T > 0  →  ferromagnetic imaginary-time coupling.
  - Total effective energy:
        E_eff = ∑_p [E_cl(M_p) + E_q(M_p, M_{p+1})]
  - Update: propose single-entry flip at position (i,j) in a random replica
    p; accept by Metropolis with the full classical + quantum ΔE.
  - Schedule: Γ decreases geometrically from Γ_max to Γ_min → 0.
    • Large Γ → J_⊥ ≈ 0: replicas explore the ±1 lattice independently.
    • Small Γ → J_⊥ → ∞: imaginary-time coupling forces all replicas to
      agree, recovering the classical fixed-temperature SA limit.

Key algorithmic difference from parallel tempering (the thermodynamics
domain analogy, domain-analogy iter 2):
  - PT runs replicas at *different temperatures* and swaps configurations
    between temperature slots.  Exploration comes from the hot replicas.
  - SQA runs *all replicas at the same temperature* and couples them
    through an imaginary-time ferromagnetic interaction.  Exploration
    comes from quantum tunneling: a spin can be "in different states
    simultaneously" across Trotter replicas, which provides a correlated
    multi-flip path through classically forbidden barriers.

References
----------
  Suzuki M (1976) Relationship between d-dimensional quantal spin systems
      and (d+1)-dimensional Ising systems.  Prog. Theor. Phys. 56 1454.
  Martoňák R, Santoro G, Tosatti E (2002) Quantum annealing by the
      path-integral Monte Carlo method.  Phys. Rev. B 66, 094203.
  Santoro G, Tosatti E (2006) Optimization using quantum mechanics.
      J. Phys. A 39 R393.
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
from hadamard_maxdet.verifier import integer_determinant

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N23: int = 23
ORRICK_ABS_DET_H23: int = 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SQAResult:
    """Outcome of a single SQA run."""

    best_matrix: list[list[int]]
    best_abs_det: int
    total_steps: int
    walltime_seconds: float
    best_ratio_to_orrick: float
    gamma_range: tuple[float, float]
    temperature: float
    n_trotter: int
    per_replica_best: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _j_perp(gamma: float, temperature: float, n_trotter: int) -> float:
    """Imaginary-time coupling between adjacent Trotter slices.

    Suzuki-Trotter formula:
        J_⊥ = −(T/2) · log(tanh(Γ / (P · T)))

    Because tanh(x) ∈ (0, 1) for x > 0 we have log(tanh(x)) < 0, so
    J_⊥ > 0 (ferromagnetic coupling).

    Asymptotic behaviour:
      Γ → ∞:  tanh(Γ/PT) → 1, J_⊥ → 0   (independent replicas)
      Γ → 0:  tanh(Γ/PT) → 0, J_⊥ → +∞  (replicas locked together)
    """
    if gamma <= 0.0:
        return float("inf")
    x = gamma / (n_trotter * temperature)
    if x > 50.0:
        # tanh(x) ≈ 1 for large x → log(tanh(x)) ≈ −2 exp(−2x) ≈ 0
        return 0.0
    th = math.tanh(x)
    if th <= 0.0:
        return float("inf")
    return -(temperature / 2.0) * math.log(th)


def _random_nonsingular_state(n: int, rng: random.Random) -> IncrementalState:
    """Return a nonsingular random ±1 IncrementalState of order n."""
    for _ in range(500):
        M = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(n)]
        state = IncrementalState.from_matrix(M)
        if state.det != 0:
            return state
    raise RuntimeError(f"Could not sample a nonsingular {n}×{n} matrix in 500 tries")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulated_quantum_annealing(
    n: int = N23,
    n_trotter: int = 8,
    gamma_start: float = 2.0,
    gamma_end: float = 0.005,
    temperature: float = 0.05,
    wall_cap_s: float = 600.0,
    seed: int | None = None,
    verbose: bool = False,
    verbose_interval_s: float = 30.0,
    initial_matrix: list[list[int]] | None = None,
    refresh_interval: int = 2000,
) -> SQAResult:
    """Simulated Quantum Annealing for the ±1 matrix maxdet problem.

    Maintains P = n_trotter Trotter replicas in a ring.  Each step:

      1. Choose a random replica p and a random entry position (i, j).
      2. Compute ΔE_cl = log(|det_old| / |det_new|) via rank-1 incremental
         adjugate (O(n²), never floating-point drift).
      3. Compute ΔE_q  = 2 · J_⊥ · M_p[i,j] · (M_{p-1}[i,j] + M_{p+1}[i,j])
         (O(1) lookup in neighbouring replicas).
      4. Accept with Metropolis: min(1, exp(−(ΔE_cl + ΔE_q) / T)).

    Γ decreases geometrically from gamma_start to gamma_end over the
    full wall_cap_s budget (frac-based schedule).

    Best |det| across all replicas at all steps is tracked.  The best
    matrix is re-verified by exact Bareiss ``integer_determinant`` at the end.

    Parameters
    ----------
    n              : Matrix order.
    n_trotter      : Number of Trotter replicas P.
    gamma_start    : Initial transverse field (large → independent replicas).
    gamma_end      : Final transverse field (small → classical limit).
    temperature    : Fixed temperature T for all replicas.
    wall_cap_s     : Wall-clock budget in seconds.
    seed           : RNG seed.
    verbose        : Print progress every verbose_interval_s seconds.
    initial_matrix : Seed matrix; all replicas start from independent copies
                     (with small random perturbations) of this matrix.
    refresh_interval: Force exact re-derivation of the adjugate every this
                     many accepted flips per replica.

    Returns
    -------
    SQAResult with best matrix, diagnostics, and per-replica final |det|.
    """
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Initialise replicas
    # ------------------------------------------------------------------
    replicas: list[IncrementalState] = []
    if initial_matrix is not None:
        for p in range(n_trotter):
            M_copy = [row[:] for row in initial_matrix]
            # Apply p small random perturbations so replicas are distinct
            for _ in range(p):
                ri = rng.randrange(n)
                rj = rng.randrange(n)
                M_copy[ri][rj] = -M_copy[ri][rj]
            state = IncrementalState.from_matrix(M_copy)
            if state.det == 0:
                # Recover with a fresh random
                state = _random_nonsingular_state(n, rng)
            replicas.append(state)
    else:
        for _ in range(n_trotter):
            replicas.append(_random_nonsingular_state(n, rng))

    # ------------------------------------------------------------------
    # Global best tracking
    # ------------------------------------------------------------------
    best_abs_det: int = max(abs(r.det) for r in replicas)
    best_matrix: list[list[int]] = []
    for r in replicas:
        if abs(r.det) == best_abs_det:
            best_matrix = [row[:] for row in r.M]
            break

    refresh_counters = [0] * n_trotter
    total_steps = 0
    inv_t = 1.0 / max(temperature, 1e-12)

    # Log-space Γ schedule: frac = steps / wall_cap_s time fraction
    log_gs = math.log(gamma_start)
    log_ge = math.log(gamma_end)

    start_t = time.monotonic()
    last_verbose = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        now = time.monotonic() - start_t
        if now >= wall_cap_s:
            break

        # Current Γ (time-fraction schedule)
        frac = min(1.0, now / wall_cap_s)
        gamma = math.exp(log_gs + frac * (log_ge - log_gs))
        j_p = _j_perp(gamma, temperature, n_trotter)

        # Run a block of inner steps at this Γ
        block = 500
        for _ in range(block):
            p = rng.randrange(n_trotter)
            i = rng.randrange(n)
            j = rng.randrange(n)

            # --- Classical energy change ---
            new_det = propose_entry_flip(replicas[p], i, j)
            if new_det == 0:
                total_steps += 1
                continue
            new_abs = abs(new_det)
            old_abs = abs(replicas[p].det)
            if old_abs == 0:
                total_steps += 1
                continue

            # ΔE_cl = E_new − E_old = −log|det_new| − (−log|det_old|)
            #       = log(|det_old| / |det_new|)
            de_cl = math.log(old_abs) - math.log(new_abs)

            # --- Quantum energy change ---
            # ΔE_q = 2 · J_⊥ · M_p[i,j] · (M_{p-1}[i,j] + M_{p+1}[i,j])
            p_left = (p - 1) % n_trotter
            p_right = (p + 1) % n_trotter
            current_val = replicas[p].M[i][j]
            val_left = replicas[p_left].M[i][j]
            val_right = replicas[p_right].M[i][j]
            de_q = 2.0 * j_p * current_val * (val_left + val_right)

            de_total = de_cl + de_q

            # --- Metropolis acceptance ---
            if de_total <= 0.0 or rng.random() < math.exp(-de_total * inv_t):
                apply_entry_flip(replicas[p], i, j)
                refresh_counters[p] += 1
                d = abs(replicas[p].det)
                if d > best_abs_det:
                    best_abs_det = d
                    best_matrix = [row[:] for row in replicas[p].M]

            total_steps += 1

        # Refresh adjugates to prevent integer drift
        for p in range(n_trotter):
            if refresh_counters[p] >= refresh_interval:
                refresh(replicas[p])
                refresh_counters[p] = 0

        if verbose:
            now = time.monotonic() - start_t
            if now - last_verbose >= verbose_interval_s:
                ratio = best_abs_det / ORRICK_ABS_DET_H23
                print(
                    f"  t={now:.0f}s  steps={total_steps:,}"
                    f"  gamma={gamma:.4f}  j_perp={j_p:.4f}"
                    f"  best={best_abs_det} ({ratio:.4f}×Orrick)"
                )
                last_verbose = now

    wall = time.monotonic() - start_t

    # ------------------------------------------------------------------
    # Final refresh and exact verification
    # ------------------------------------------------------------------
    per_replica_best: list[int] = []
    for p in range(n_trotter):
        refresh(replicas[p])
        d = abs(replicas[p].det)
        per_replica_best.append(d)
        if d > best_abs_det:
            best_abs_det = d
            best_matrix = [row[:] for row in replicas[p].M]

    if best_matrix:
        exact_abs = abs(integer_determinant(best_matrix))
        best_abs_det = exact_abs

    return SQAResult(
        best_matrix=best_matrix,
        best_abs_det=best_abs_det,
        total_steps=total_steps,
        walltime_seconds=wall,
        best_ratio_to_orrick=best_abs_det / ORRICK_ABS_DET_H23,
        gamma_range=(gamma_start, gamma_end),
        temperature=temperature,
        n_trotter=n_trotter,
        per_replica_best=per_replica_best,
    )


def triple_verify(M: list[list[int]]) -> tuple[int, int, int]:
    """Triple-verify |det(M)| via Bareiss on M, M^T, and a fresh copy.

    Returns (det_M, det_MT, det_copy).  All three absolute values should agree.
    Raises ValueError on malformed input.
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
