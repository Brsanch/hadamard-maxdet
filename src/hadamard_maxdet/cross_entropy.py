"""Cross-Entropy Method (CEM) for Hadamard maxdet search.

Domain: Information Theory
Technique: Cross-Entropy Method — iteratively minimise the KL divergence
between a parametric sampling distribution and the (unknown) distribution
concentrated on high-|det| matrices.

The CEM models each of the n² matrix entries as an independent Bernoulli
variable with parameter p[i][j] = P(M[i][j] = +1).  Each iteration:

  1. Sample N candidate matrices from the current Bernoulli product.
  2. Refine each candidate via SA (warm start).
  3. Keep the top elite_k by |det| as the "elite set".
  4. Update each p[i][j] via maximum-likelihood on the elite set,
     smoothed by a learning-rate parameter alpha to prevent collapse.
  5. Enforce a minimum probability p_min to keep the distribution
     from pinning any entry, preserving entropy for exploration.

The entropy of the distribution H = Σ_ij H(p[i][j]) / n² is a summary
statistic: it starts at ln 2 ≈ 0.693 (uniform) and decreases as the
CEM learns entry-level structure from high-quality matrices.

Public API
----------
* :func:`cem_sample`   — draw one matrix from Bernoulli(p).
* :func:`update_probs` — MLE update of p from elite matrices.
* :func:`dist_entropy` — mean per-entry binary entropy.
* :func:`run_cem`      — full driver: SA + CEM iterations, returns
                         :class:`CemResult`.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.search import sa_incremental
from hadamard_maxdet.verifier import integer_determinant

ORRICK_DET_H23: int = 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# Core CEM primitives
# ---------------------------------------------------------------------------


def cem_sample(
    p: list[list[float]],
    rng: random.Random,
) -> list[list[int]]:
    """Sample a ±1 matrix from the Bernoulli product distribution p."""
    n = len(p)
    return [
        [1 if rng.random() < p[i][j] else -1 for j in range(n)]
        for i in range(n)
    ]


def update_probs(
    p: list[list[float]],
    elite: list[list[list[int]]],
    *,
    alpha: float = 0.7,
    p_min: float = 0.1,
) -> None:
    """In-place MLE update of p from elite matrices with smoothing.

    Uses the CEM cross-entropy update:
        p_new[i][j] = (1 - alpha) * p[i][j]
                    + alpha * (fraction of elite matrices with +1 at (i,j))

    then clips to [p_min, 1 - p_min].
    """
    n = len(p)
    k = len(elite)
    if k == 0:
        return
    for i in range(n):
        for j in range(n):
            frac_plus = sum(1 for m in elite if m[i][j] == 1) / k
            p[i][j] = (1.0 - alpha) * p[i][j] + alpha * frac_plus
            if p[i][j] < p_min:
                p[i][j] = p_min
            elif p[i][j] > 1.0 - p_min:
                p[i][j] = 1.0 - p_min


def dist_entropy(p: list[list[float]]) -> float:
    """Mean per-entry binary entropy of the distribution.

    Returns H in nats.  Maximum value is ln(2) ≈ 0.693 (uniform).
    Decreasing H indicates the CEM is learning structure.
    """
    n = len(p)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            pp = p[i][j]
            if 0.0 < pp < 1.0:
                total -= pp * math.log(pp) + (1.0 - pp) * math.log(1.0 - pp)
            count += 1
    return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CemResult:
    """Outcome of a CEM run on the Hadamard maxdet problem."""

    best_abs_det: int
    best_matrix: list[list[int]]
    ratio_to_orrick: float
    above_orrick: bool
    initial_entropy: float
    final_entropy: float
    n_sa_total: int
    walltime_seconds: float
    generation_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Full CEM driver
# ---------------------------------------------------------------------------


def run_cem(
    n: int,
    *,
    n_init: int = 6,
    n_samples: int = 5,
    n_elite: int = 3,
    n_generations: int = 4,
    sa_iterations: int = 500_000,
    alpha: float = 0.7,
    p_min: float = 0.1,
    t_sa_start: float = 2.0,
    t_sa_end: float = 0.002,
    wall_cap: float = 860.0,
    rng_seed: int = 0,
    verbose: bool = True,
) -> CemResult:
    """Run the Cross-Entropy Method on the Hadamard maxdet problem at order n.

    Phase 1 — warm-up:  ``n_init`` independent SA runs from random starts.
    Phase 2 — CEM loop: ``n_generations`` iterations of sample/refine/update.

    Parameters
    ----------
    n
        Matrix order.
    n_init
        Number of initial SA runs to seed the elite pool.
    n_samples
        New matrices sampled per CEM generation.
    n_elite
        Size of the elite set maintained across all generations.
    n_generations
        Number of CEM update iterations.
    sa_iterations
        SA steps per refinement call.
    alpha
        Learning rate for the CEM probability update.
    p_min
        Minimum per-entry probability (prevents distribution collapse).
    t_sa_start, t_sa_end
        SA temperature schedule endpoints.
    wall_cap
        Hard wall-clock cap (seconds); no new SA run starts after this.
    rng_seed
        Master PRNG seed.
    verbose
        Print progress lines.
    """
    t_wall = time.monotonic()
    rng = random.Random(rng_seed)

    def elapsed() -> float:
        return time.monotonic() - t_wall

    def time_left() -> float:
        return wall_cap - elapsed()

    # Minimum seconds to keep in reserve before starting a new SA run.
    # Scale with wall_cap so short test budgets still get runs.
    _time_guard = min(35.0, wall_cap * 0.1)

    # Initialise probability distribution at maximum entropy
    p: list[list[float]] = [[0.5] * n for _ in range(n)]
    initial_entropy = dist_entropy(p)

    # Track all (matrix, det) pairs; elite is always the top-n_elite subset
    pool: list[tuple[list[list[int]], int]] = []

    def current_elite() -> list[list[list[int]]]:
        top = sorted(pool, key=lambda x: x[1], reverse=True)[:n_elite]
        return [m for m, _ in top]

    best_det = 0
    best_matrix: list[list[int]] = []
    n_sa_total = 0
    generation_log: list[dict] = []

    def do_sa(initial_entries: list[list[int]] | None = None) -> tuple[list[list[int]], int]:
        seed = rng.randint(0, 2**31)
        initial = (
            Matrix(n=n, entries=[row[:] for row in initial_entries])
            if initial_entries is not None
            else None
        )
        res = sa_incremental(
            n,
            iterations=sa_iterations,
            seed=seed,
            t_start=t_sa_start,
            t_end=t_sa_end,
            initial=initial,
        )
        return res.matrix.entries, res.abs_det

    # -----------------------------------------------------------------------
    # Phase 1: Initial population
    # -----------------------------------------------------------------------
    if verbose:
        print(f"CEM Phase 1: {n_init} initial SA runs ({sa_iterations:,} iters each)")

    for k in range(n_init):
        if time_left() < _time_guard:
            break
        entries, det = do_sa()
        n_sa_total += 1
        pool.append((entries, det))
        if det > best_det:
            best_det = det
            best_matrix = entries
        if verbose:
            print(
                f"  init {k+1}/{n_init}: |det|={det:,d}  "
                f"ratio={det/ORRICK_DET_H23:.4f}  t={elapsed():.1f}s"
            )

    # Seed the probability distribution from the initial elite
    update_probs(p, current_elite(), alpha=alpha, p_min=p_min)
    h0 = dist_entropy(p)
    if verbose:
        print(f"  Post-init entropy: {h0:.4f}  (max={initial_entropy:.4f})")

    # -----------------------------------------------------------------------
    # Phase 2: CEM update loop
    # -----------------------------------------------------------------------
    for gen in range(n_generations):
        if time_left() < _time_guard:
            break

        if verbose:
            print(f"\nCEM generation {gen+1}/{n_generations}:")

        gen_best = 0
        gen_log: dict = {"generation": gen + 1, "samples": []}

        for k in range(n_samples):
            if time_left() < _time_guard:
                break
            # Sample from current distribution, then SA-refine
            candidate = cem_sample(p, rng)
            entries, det = do_sa(candidate)
            n_sa_total += 1
            pool.append((entries, det))

            if det > best_det:
                best_det = det
                best_matrix = entries
                if verbose:
                    print(f"  *** NEW BEST: {det:,d} ratio={det/ORRICK_DET_H23:.6f} ***")

            if det > gen_best:
                gen_best = det

            gen_log["samples"].append({"abs_det": det, "ratio": det / ORRICK_DET_H23})
            if verbose:
                print(
                    f"  sample {k+1}/{n_samples}: |det|={det:,d}  "
                    f"ratio={det/ORRICK_DET_H23:.4f}  t={elapsed():.1f}s"
                )

        # Update probability distribution from updated elite pool
        update_probs(p, current_elite(), alpha=alpha, p_min=p_min)
        h = dist_entropy(p)
        gen_log["entropy"] = h
        gen_log["best_abs_det"] = gen_best
        generation_log.append(gen_log)

        if verbose:
            print(
                f"  gen {gen+1} best={gen_best:,d}  "
                f"entropy={h:.4f} (↓={initial_entropy-h:.4f})  overall={best_det:,d}"
            )

    final_entropy = dist_entropy(p)
    walltime = elapsed()

    return CemResult(
        best_abs_det=best_det,
        best_matrix=best_matrix,
        ratio_to_orrick=best_det / ORRICK_DET_H23,
        above_orrick=best_det > ORRICK_DET_H23,
        initial_entropy=initial_entropy,
        final_entropy=final_entropy,
        n_sa_total=n_sa_total,
        walltime_seconds=walltime,
        generation_log=generation_log,
    )
