"""Parallel tempering (replica exchange) and population annealing for
the Hadamard maximal determinant problem.

Both methods target the regime where naive SA plateaus (n ≥ 11): they
share information between walkers at different temperatures to avoid
local-minimum traps in the {-1, +1}^{n×n} energy landscape with
E(M) = -log|det M|.

Replica exchange (Hukushima & Nemoto 1996; Swendsen & Wang 1986)
    Runs ``n_replicas`` independent Metropolis chains at a temperature
    ladder t_1 > t_2 > ... > t_R simultaneously. At periodic intervals
    it attempts to swap configurations between adjacent temperatures
    with standard Metropolis acceptance

        p_swap = min(1, exp((β_i - β_{i+1}) · (E_i - E_{i+1})))

    where β = 1/T. Hot replicas explore broadly; cold replicas exploit
    promising basins; swaps ratchet good configurations downward.

Population annealing (Hukushima & Iba 2003; Wang, Machta, Katzgraber 2015)
    Carries a population of walkers and *resamples* at each cooling
    step, weighting each walker by exp(-(β_{k+1} - β_k) · E) before
    multinomial copy. Between resamplings, each walker takes a few
    Metropolis steps at the new temperature. Resampling is what makes
    PA qualitatively different from ensemble SA: good walkers spawn
    descendants, bad walkers die, so computational effort concentrates
    on the basin the low-energy tail is leaving for.

Both methods use the exact integer determinant for scoring (never
float). The best-ever configuration across all replicas / walkers / all
steps is tracked and returned — not only the final cold-replica state.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.moves import random_move
from hadamard_maxdet.search import AnnealResult
from hadamard_maxdet.structured import (
    bordered_circulant,
    skew_generator_from_free_bits,
)
from hadamard_maxdet.verifier import integer_determinant


def _log_abs_det(entries: list[list[int]]) -> float:
    d = integer_determinant(entries)
    if d == 0:
        return -math.inf
    return math.log(abs(d))


def _default_ladder(n_replicas: int, t_cold: float, t_hot: float) -> list[float]:
    """Log-spaced temperature ladder from t_cold (index 0) to t_hot (index R-1)."""
    if n_replicas < 2:
        raise ValueError(f"need at least 2 replicas, got {n_replicas}")
    if t_cold <= 0 or t_hot <= 0:
        raise ValueError("temperatures must be positive")
    if t_hot <= t_cold:
        raise ValueError(f"t_hot ({t_hot}) must exceed t_cold ({t_cold})")
    log_lo, log_hi = math.log(t_cold), math.log(t_hot)
    return [
        math.exp(log_lo + (log_hi - log_lo) * k / (n_replicas - 1))
        for k in range(n_replicas)
    ]


@dataclass
class ReplicaExchangeResult(AnnealResult):
    """Outcome of a replica-exchange run.

    Extends :class:`AnnealResult` with per-pair acceptance-rate statistics
    for the swap attempts (diagnostic useful when tuning the ladder).
    """

    temperatures: list[float] = field(default_factory=list)
    swap_attempts: list[int] = field(default_factory=list)   # len = R - 1
    swap_accepts: list[int] = field(default_factory=list)    # len = R - 1
    move_accepts: list[int] = field(default_factory=list)    # per-replica
    move_attempts: list[int] = field(default_factory=list)

    def swap_rates(self) -> list[float]:
        """Swap acceptance rate per adjacent pair (0 = cold-next-to-cold)."""
        return [
            (a / t) if t > 0 else 0.0
            for a, t in zip(self.swap_accepts, self.swap_attempts)
        ]


def replica_exchange(
    n: int,
    *,
    n_replicas: int = 30,
    temperatures: Sequence[float] | None = None,
    iterations: int = 10_000,
    exchange_every: int = 1_000,
    move_fn: Callable = random_move,
    seed: int | None = None,
    initial: Matrix | None = None,
    t_cold: float = 0.005,
    t_hot: float = 5.0,
) -> ReplicaExchangeResult:
    """Parallel tempering on -log|det M|.

    Parameters
    ----------
    n
        Matrix order.
    n_replicas
        Number of replicas on the temperature ladder. Ignored if
        ``temperatures`` is provided.
    temperatures
        Explicit ladder, sorted cold-to-hot (temperatures[0] is the
        coldest replica). If ``None``, log-spaces from ``t_cold`` to
        ``t_hot`` across ``n_replicas`` levels.
    iterations
        Total Metropolis steps *per replica* (so wall work is
        ``iterations * n_replicas`` proposals).
    exchange_every
        Attempt an exchange sweep (adjacent-pair swaps across the ladder)
        every this-many Metropolis steps.
    move_fn
        Callable ``(entries, rng) -> new_entries | None`` proposing a move.
    seed
        PRNG seed; replica k uses ``seed + k`` for its per-replica RNG,
        and the swap RNG uses ``seed`` itself.
    initial
        Optional common starting matrix; all replicas start from a copy.
        If None, each replica starts from a fresh random matrix.
    t_cold, t_hot
        Used only when ``temperatures`` is None.

    Returns
    -------
    ReplicaExchangeResult
        ``.abs_det`` is the best |det| encountered at *any* replica at
        *any* step (not just the final cold-replica state).
    """
    if temperatures is None:
        temps = _default_ladder(n_replicas, t_cold, t_hot)
    else:
        temps = list(temperatures)
        if len(temps) < 2:
            raise ValueError("need at least 2 temperatures")
        n_replicas = len(temps)

    R = n_replicas
    master_rng = random.Random(seed)
    swap_rng = random.Random((seed or 0) ^ 0xABCDEF01)

    # Per-replica RNGs for independent move streams.
    rngs = [random.Random((seed or 0) + 1000 * (k + 1)) for k in range(R)]

    # Initialise all replicas.
    states: list[list[list[int]]] = []
    energies: list[float] = []
    for k in range(R):
        if initial is not None:
            if initial.n != n:
                raise ValueError(f"initial order {initial.n}, expected {n}")
            entries = [row[:] for row in initial.entries]
        else:
            entries = [
                [1 if rngs[k].random() < 0.5 else -1 for _ in range(n)]
                for _ in range(n)
            ]
        states.append(entries)
        energies.append(-_log_abs_det(entries))

    # Best-ever tracking.
    best_e = math.inf
    best_entries: list[list[int]] | None = None
    for k in range(R):
        if energies[k] < best_e:
            best_e = energies[k]
            best_entries = [row[:] for row in states[k]]

    swap_attempts = [0] * (R - 1)
    swap_accepts = [0] * (R - 1)
    move_attempts = [0] * R
    move_accepts = [0] * R

    start = time.monotonic()

    for step in range(iterations):
        # One Metropolis move per replica.
        for k in range(R):
            t = temps[k]
            proposal = move_fn(states[k], rngs[k])
            if proposal is None:
                continue
            move_attempts[k] += 1
            prop_e = -_log_abs_det(proposal)
            if prop_e == math.inf:
                continue  # singular; reject
            delta = prop_e - energies[k]
            if delta <= 0 or rngs[k].random() < math.exp(-delta / max(t, 1e-12)):
                states[k] = proposal
                energies[k] = prop_e
                move_accepts[k] += 1
                if prop_e < best_e:
                    best_e = prop_e
                    best_entries = [row[:] for row in proposal]

        # Periodic adjacent-pair exchange sweep.
        if exchange_every > 0 and (step + 1) % exchange_every == 0:
            # Alternate even/odd offsets to avoid bias.
            offset = 0 if ((step + 1) // exchange_every) % 2 == 0 else 1
            for i in range(offset, R - 1, 2):
                # Swap between replica i (cold side) and i+1 (hot side).
                beta_i = 1.0 / max(temps[i], 1e-12)
                beta_j = 1.0 / max(temps[i + 1], 1e-12)
                # Standard RE acceptance: exp((beta_i - beta_j)(E_i - E_j))
                # If energies are finite, this is well-defined.
                if energies[i] == math.inf or energies[i + 1] == math.inf:
                    continue
                log_p = (beta_i - beta_j) * (energies[i] - energies[i + 1])
                swap_attempts[i] += 1
                if log_p >= 0 or swap_rng.random() < math.exp(log_p):
                    states[i], states[i + 1] = states[i + 1], states[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    swap_accepts[i] += 1

    walltime = time.monotonic() - start

    if best_entries is None:
        # Should be impossible unless all random inits were singular.
        best_entries = states[0]
        best_e = energies[0]

    best_abs_det = abs(integer_determinant(best_entries))
    return ReplicaExchangeResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=iterations * R,
        accepted=sum(move_accepts),
        walltime_seconds=walltime,
        temperatures=temps,
        swap_attempts=swap_attempts,
        swap_accepts=swap_accepts,
        move_accepts=move_accepts,
        move_attempts=move_attempts,
    )


@dataclass
class PopulationAnnealingResult(AnnealResult):
    """Outcome of a population-annealing run.

    Adds diagnostics for family size and per-step effective sample size.
    """

    temperature_schedule: list[float] = field(default_factory=list)
    effective_sample_sizes: list[float] = field(default_factory=list)
    unique_families: list[int] = field(default_factory=list)


def population_annealing(
    n: int,
    *,
    population_size: int = 64,
    temperature_schedule: Sequence[float] | None = None,
    iterations_per_step: int = 200,
    move_fn: Callable = random_move,
    seed: int | None = None,
    t_start: float = 5.0,
    t_end: float = 0.005,
    n_temperatures: int = 40,
    initial: Matrix | None = None,
) -> PopulationAnnealingResult:
    """Population annealing to maximise |det M|.

    Walkers are evolved via Metropolis at a monotonically-decreasing
    temperature schedule; between consecutive temperatures the population
    is resampled with weights ``exp(-(β_{k+1} - β_k) · E)``.

    Parameters
    ----------
    n
        Matrix order.
    population_size
        Number of walkers carried through the anneal.
    temperature_schedule
        Monotonically decreasing list of temperatures (hot → cold). If
        ``None``, log-spaces from ``t_start`` down to ``t_end`` across
        ``n_temperatures`` levels.
    iterations_per_step
        Metropolis steps each walker takes at each temperature.
    move_fn
        Move proposer.
    seed
        PRNG seed.
    initial
        Optional starting matrix for every walker (else all random).

    Returns
    -------
    PopulationAnnealingResult
        ``.abs_det`` is the best |det| ever reached by any walker at any
        step.
    """
    if temperature_schedule is None:
        if n_temperatures < 2:
            raise ValueError("n_temperatures must be at least 2")
        # Log-spaced hot → cold.
        log_hi, log_lo = math.log(t_start), math.log(t_end)
        schedule = [
            math.exp(log_hi + (log_lo - log_hi) * k / (n_temperatures - 1))
            for k in range(n_temperatures)
        ]
    else:
        schedule = list(temperature_schedule)
        if len(schedule) < 2:
            raise ValueError("need at least 2 temperatures")
        # Sanity: should be non-increasing.
        for a, b in zip(schedule, schedule[1:]):
            if b > a + 1e-12:
                raise ValueError(
                    "temperature_schedule must be non-increasing (hot → cold)"
                )

    rng = random.Random(seed)
    # Per-walker RNGs so parallelisation is deterministic if needed.
    walker_rngs = [
        random.Random((seed or 0) + 7919 * (k + 1)) for k in range(population_size)
    ]

    # Initialise population.
    states: list[list[list[int]]] = []
    energies: list[float] = []
    families: list[int] = list(range(population_size))  # genealogy tracking

    for k in range(population_size):
        if initial is not None:
            if initial.n != n:
                raise ValueError(f"initial order {initial.n}, expected {n}")
            entries = [row[:] for row in initial.entries]
        else:
            entries = [
                [1 if walker_rngs[k].random() < 0.5 else -1 for _ in range(n)]
                for _ in range(n)
            ]
        states.append(entries)
        energies.append(-_log_abs_det(entries))

    best_e = math.inf
    best_entries: list[list[int]] | None = None
    for k in range(population_size):
        if energies[k] < best_e:
            best_e = energies[k]
            best_entries = [row[:] for row in states[k]]

    total_accepted = 0
    ess_history: list[float] = []
    unique_family_history: list[int] = []
    start = time.monotonic()

    # First temperature: no resampling (that's the initial population).
    # For k = 1 ... K-1: resample using β_{k-1} → β_k, then move at β_k.
    prev_beta = 1.0 / max(schedule[0], 1e-12)

    # Initial move phase at schedule[0] so walkers decorrelate from random init.
    t = schedule[0]
    for w in range(population_size):
        for _ in range(iterations_per_step):
            proposal = move_fn(states[w], walker_rngs[w])
            if proposal is None:
                continue
            prop_e = -_log_abs_det(proposal)
            if prop_e == math.inf:
                continue
            delta = prop_e - energies[w]
            if delta <= 0 or walker_rngs[w].random() < math.exp(
                -delta / max(t, 1e-12)
            ):
                states[w] = proposal
                energies[w] = prop_e
                total_accepted += 1
                if prop_e < best_e:
                    best_e = prop_e
                    best_entries = [row[:] for row in proposal]

    for step_idx in range(1, len(schedule)):
        t_new = schedule[step_idx]
        beta_new = 1.0 / max(t_new, 1e-12)
        dbeta = beta_new - prev_beta  # positive (beta increases as T decreases)

        # Resampling weights: walkers with LOW energy survive (E = -log|det|).
        # Weight ∝ exp(-dbeta · E). Normalize in log space for stability.
        log_w = [-dbeta * e for e in energies]
        m = max(log_w)
        w_unnorm = [math.exp(lw - m) for lw in log_w]
        total_w = sum(w_unnorm)
        if total_w == 0:
            # All walkers have effectively -inf weight; fall back to uniform.
            probs = [1.0 / population_size] * population_size
        else:
            probs = [w / total_w for w in w_unnorm]

        # Effective sample size.
        sum_w_sq = sum(p * p for p in probs)
        ess = 1.0 / sum_w_sq if sum_w_sq > 0 else 0.0
        ess_history.append(ess)

        # Multinomial resample: pick population_size indices with replacement.
        cum = []
        c = 0.0
        for p in probs:
            c += p
            cum.append(c)
        # Guard: float accumulation.
        cum[-1] = 1.0

        new_states: list[list[list[int]]] = []
        new_energies: list[float] = []
        new_families: list[int] = []
        for _ in range(population_size):
            u = rng.random()
            # Binary-search the cumulative distribution.
            lo, hi = 0, population_size - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if u <= cum[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            idx = lo
            new_states.append([row[:] for row in states[idx]])
            new_energies.append(energies[idx])
            new_families.append(families[idx])
        states = new_states
        energies = new_energies
        families = new_families
        unique_family_history.append(len(set(families)))

        # Move phase at the new temperature.
        for w in range(population_size):
            for _ in range(iterations_per_step):
                proposal = move_fn(states[w], walker_rngs[w])
                if proposal is None:
                    continue
                prop_e = -_log_abs_det(proposal)
                if prop_e == math.inf:
                    continue
                delta = prop_e - energies[w]
                if delta <= 0 or walker_rngs[w].random() < math.exp(
                    -delta / max(t_new, 1e-12)
                ):
                    states[w] = proposal
                    energies[w] = prop_e
                    total_accepted += 1
                    if prop_e < best_e:
                        best_e = prop_e
                        best_entries = [row[:] for row in proposal]

        prev_beta = beta_new

    walltime = time.monotonic() - start

    if best_entries is None:
        best_entries = states[0]

    total_proposals = population_size * iterations_per_step * len(schedule)
    best_abs_det = abs(integer_determinant(best_entries))
    return PopulationAnnealingResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=total_proposals,
        accepted=total_accepted,
        walltime_seconds=walltime,
        temperature_schedule=schedule,
        effective_sample_sizes=ess_history,
        unique_families=unique_family_history,
    )


# ---------------------------------------------------------------------------
# Replica exchange INSIDE structured ansatz cores.
# ---------------------------------------------------------------------------
#
# Round 1 found that bordered-circulant SA beats unstructured SA on H19
# by collapsing the search space from 2^(n^2) to 2^(n-1), and that 30-replica
# parallel tempering beats naive multistart on the unstructured space by
# +39%. The composition test: run replica exchange directly over the
# (n-1)-bit core. Each replica carries its own core, proposals flip a single
# bit, scoring rebuilds the n×n bordered matrix and uses integer_determinant.
# Adjacent-pair Metropolis swaps share configurations across the temperature
# ladder.


def _bits_to_bordered_entries(bits: Sequence[int]) -> list[list[int]]:
    return bordered_circulant(list(bits)).entries


def _bits_to_skew_entries(n: int, free_bits: Sequence[int]) -> list[list[int]]:
    gen = skew_generator_from_free_bits(n, list(free_bits))
    return Matrix.circulant(gen).entries


def _replica_exchange_over_bits(
    *,
    n_order: int,
    n_bits: int,
    build_entries: Callable[[Sequence[int]], list[list[int]]],
    n_replicas: int,
    temperatures: Sequence[float] | None,
    iterations: int,
    exchange_every: int,
    seed: int | None,
    initial_bits_per_replica: Sequence[Sequence[int]] | None,
    t_cold: float,
    t_hot: float,
) -> ReplicaExchangeResult:
    """Generic replica exchange over a ±1 bit vector.

    Each replica carries a length-``n_bits`` ±1 vector. A proposal flips one
    bit. Scoring rebuilds the full ``n_order × n_order`` matrix via
    ``build_entries`` and uses :func:`integer_determinant`. Adjacent-pair
    Metropolis swaps follow the standard
    ``p = min(1, exp((β_i − β_j)(E_i − E_j)))`` rule.
    """
    if temperatures is None:
        temps = _default_ladder(n_replicas, t_cold, t_hot)
    else:
        temps = list(temperatures)
        if len(temps) < 2:
            raise ValueError("need at least 2 temperatures")
        n_replicas = len(temps)

    R = n_replicas
    swap_rng = random.Random((seed or 0) ^ 0xABCDEF02)
    rngs = [random.Random((seed or 0) + 1000 * (k + 1)) for k in range(R)]

    # Initialise replicas.
    bits_states: list[list[int]] = []
    energies: list[float] = []
    if initial_bits_per_replica is not None:
        initial_list = [list(x) for x in initial_bits_per_replica]
        if len(initial_list) != R:
            raise ValueError(
                f"initial_bits_per_replica must have length {R}; got {len(initial_list)}"
            )
        for ib in initial_list:
            if len(ib) != n_bits:
                raise ValueError(
                    f"each initial bits vector must have length {n_bits}"
                )
            for v in ib:
                if v not in (-1, 1):
                    raise ValueError(f"bits must be ±1; got {v!r}")
    for k in range(R):
        if initial_bits_per_replica is not None:
            bits = list(initial_list[k])
        else:
            bits = [1 if rngs[k].random() < 0.5 else -1 for _ in range(n_bits)]
        # Avoid singular start by resampling a few times if needed.
        e = -_log_abs_det(build_entries(bits))
        tries = 0
        while e == math.inf and initial_bits_per_replica is None and tries < 8:
            bits = [1 if rngs[k].random() < 0.5 else -1 for _ in range(n_bits)]
            e = -_log_abs_det(build_entries(bits))
            tries += 1
        bits_states.append(bits)
        energies.append(e)

    best_e = math.inf
    best_bits: list[int] | None = None
    for k in range(R):
        if energies[k] < best_e:
            best_e = energies[k]
            best_bits = list(bits_states[k])

    swap_attempts = [0] * (R - 1)
    swap_accepts = [0] * (R - 1)
    move_attempts = [0] * R
    move_accepts = [0] * R

    start = time.monotonic()

    for step in range(iterations):
        # One Metropolis bit-flip per replica.
        for k in range(R):
            t = temps[k]
            bits = bits_states[k]
            idx = rngs[k].randrange(n_bits)
            bits[idx] = -bits[idx]
            move_attempts[k] += 1
            prop_e = -_log_abs_det(build_entries(bits))
            if prop_e == math.inf:
                # Singular proposal — revert.
                bits[idx] = -bits[idx]
                continue
            delta = prop_e - energies[k]
            if delta <= 0 or rngs[k].random() < math.exp(-delta / max(t, 1e-12)):
                energies[k] = prop_e
                move_accepts[k] += 1
                if prop_e < best_e:
                    best_e = prop_e
                    best_bits = list(bits)
            else:
                # Reject: undo flip.
                bits[idx] = -bits[idx]

        # Periodic adjacent-pair swap sweep.
        if exchange_every > 0 and (step + 1) % exchange_every == 0:
            offset = 0 if ((step + 1) // exchange_every) % 2 == 0 else 1
            for i in range(offset, R - 1, 2):
                if energies[i] == math.inf or energies[i + 1] == math.inf:
                    continue
                beta_i = 1.0 / max(temps[i], 1e-12)
                beta_j = 1.0 / max(temps[i + 1], 1e-12)
                log_p = (beta_i - beta_j) * (energies[i] - energies[i + 1])
                swap_attempts[i] += 1
                if log_p >= 0 or swap_rng.random() < math.exp(log_p):
                    bits_states[i], bits_states[i + 1] = (
                        bits_states[i + 1],
                        bits_states[i],
                    )
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    swap_accepts[i] += 1

    walltime = time.monotonic() - start

    if best_bits is None:
        best_bits = bits_states[0]
        best_e = energies[0]

    best_entries = build_entries(best_bits)
    best_abs_det = abs(integer_determinant(best_entries))
    return ReplicaExchangeResult(
        matrix=Matrix(n=n_order, entries=best_entries),
        abs_det=best_abs_det,
        iterations=iterations * R,
        accepted=sum(move_accepts),
        walltime_seconds=walltime,
        temperatures=temps,
        swap_attempts=swap_attempts,
        swap_accepts=swap_accepts,
        move_accepts=move_accepts,
        move_attempts=move_attempts,
    )


def replica_exchange_bordered_circulant(
    n: int,
    *,
    n_replicas: int = 30,
    temperatures: Sequence[float] | None = None,
    iterations: int = 1_000,
    exchange_every: int = 500,
    seed: int | None = None,
    initial_cores: Sequence[Sequence[int]] | None = None,
    t_cold: float = 0.005,
    t_hot: float = 5.0,
) -> ReplicaExchangeResult:
    """Replica exchange over the (n−1)-bit bordered-circulant core.

    Each of ``n_replicas`` replicas carries its own ±1 core of length
    ``n − 1``. Proposals flip a single core bit; scoring materialises the
    full ``n × n`` bordered matrix (border row/col fixed) and uses
    :func:`integer_determinant`. Standard adjacent-pair Metropolis swaps
    are attempted every ``exchange_every`` steps.

    The state space is 2^(n−1), so for n=19 that's 262,144 cores — small
    enough that a few hundred iterations per replica should sweep the
    entire ansatz when the ladder is well-tuned.

    Parameters
    ----------
    n
        Matrix order. ``n >= 2``.
    n_replicas
        Number of replicas on the ladder. Ignored if ``temperatures`` is
        provided.
    temperatures
        Explicit ladder, sorted cold-to-hot.
    iterations
        Metropolis steps per replica (so wall work is
        ``iterations × n_replicas`` proposals plus ~ ``iterations`` swap
        sweeps).
    exchange_every
        Attempt an adjacent-pair swap sweep every this-many bit-flip steps.
    seed
        Master PRNG seed.
    initial_cores
        Optional length-``n_replicas`` sequence of length-``(n−1)`` ±1
        cores; each replica starts from its own core. If None, each
        replica starts from a random non-singular core.

    Returns
    -------
    ReplicaExchangeResult
        ``.abs_det`` is the best |det| reached at any replica at any step.
    """
    if n < 2:
        raise ValueError(f"bordered-circulant ansatz requires n >= 2; got {n}")
    n_bits = n - 1

    if initial_cores is not None:
        initial_list = [list(x) for x in initial_cores]
        if len(initial_list) < 2 and temperatures is None:
            raise ValueError("need at least 2 initial cores (or 2 temperatures)")
    else:
        initial_list = None

    return _replica_exchange_over_bits(
        n_order=n,
        n_bits=n_bits,
        build_entries=_bits_to_bordered_entries,
        n_replicas=n_replicas,
        temperatures=temperatures,
        iterations=iterations,
        exchange_every=exchange_every,
        seed=seed,
        initial_bits_per_replica=initial_list,
        t_cold=t_cold,
        t_hot=t_hot,
    )


def replica_exchange_skew_circulant(
    n: int,
    *,
    n_replicas: int = 30,
    temperatures: Sequence[float] | None = None,
    iterations: int = 1_000,
    exchange_every: int = 500,
    seed: int | None = None,
    initial_free_bits: Sequence[Sequence[int]] | None = None,
    t_cold: float = 0.005,
    t_hot: float = 5.0,
) -> ReplicaExchangeResult:
    """Replica exchange over the (n−1)/2 free bits of a skew-type circulant.

    Requires odd ``n``. Each replica carries its own length-``(n−1)/2`` ±1
    free-bit vector; proposals flip one free bit and rebuild the full
    skew-circulant matrix. The state space 2^((n−1)/2) is *very* small —
    e.g. 2^9 = 512 at n=19 — so this is mostly useful as a uniformity
    check; for n where (n−1)/2 ≤ 22, exhaustive enumeration is cheaper.
    """
    if n % 2 == 0:
        raise ValueError(f"skew-circulant ansatz requires odd n; got n={n}")
    n_bits = (n - 1) // 2

    if initial_free_bits is not None:
        initial_list = [list(x) for x in initial_free_bits]
    else:
        initial_list = None

    def build(bits: Sequence[int]) -> list[list[int]]:
        return _bits_to_skew_entries(n, bits)

    return _replica_exchange_over_bits(
        n_order=n,
        n_bits=n_bits,
        build_entries=build,
        n_replicas=n_replicas,
        temperatures=temperatures,
        iterations=iterations,
        exchange_every=exchange_every,
        seed=seed,
        initial_bits_per_replica=initial_list,
        t_cold=t_cold,
        t_hot=t_hot,
    )
