"""Benchmark replica exchange vs population annealing vs multi-start SA.

Targets:
  H11 — proven max 327680
  H13 — proven max 14929920
  H19 — open (no published-tight upper bound inlined in this repo)

For each (order, method), reports:
  - final best |det|
  - iterations (proposals) to first reach the proven max (if any)
  - walltime

Wall budget per run: 60 seconds.

Run:
    python scripts/bench_parallel.py
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make sure the local src/ package is importable when run from repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hadamard_maxdet.matrix import Matrix  # noqa: E402
from hadamard_maxdet.moves import random_move  # noqa: E402
from hadamard_maxdet.orders import BENCHMARKS  # noqa: E402
from hadamard_maxdet.parallel import (  # noqa: E402
    population_annealing,
    replica_exchange,
)
from hadamard_maxdet.search import simulated_annealing  # noqa: E402
from hadamard_maxdet.verifier import integer_determinant  # noqa: E402


TIME_BUDGET = 60.0  # seconds per run


@dataclass
class BenchResult:
    order_id: str
    method: str
    best_abs_det: int
    proposals: int
    walltime: float
    reached_target: bool
    proposals_to_target: int | None
    extra: str = ""


def _log_abs_det(entries):
    d = integer_determinant(entries)
    if d == 0:
        return -math.inf
    return math.log(abs(d))


def _time_budget_multi_start(
    n: int, target: int | None, *, seed: int, budget_s: float = TIME_BUDGET
) -> BenchResult:
    """Run independent SA restarts serially until the wall budget expires.

    Matches the `multi_start` API's spirit while enforcing a wall cap.
    Tracks total proposals and best |det|; records first proposal count
    at which ``target`` is reached (if ever).
    """
    start = time.monotonic()
    rng = random.Random(seed)
    best = 0
    total_proposals = 0
    reached_at: int | None = None

    # Each SA run: moderate iteration count so we finish many runs in 60s.
    per_run_iters = max(2_000, 20_000 // max(1, n // 4))

    run_seed = seed
    while time.monotonic() - start < budget_s:
        remaining = budget_s - (time.monotonic() - start)
        if remaining < 0.25:
            break
        iters = per_run_iters
        r = simulated_annealing(
            n, iterations=iters, seed=run_seed, t_start=1.0, t_end=0.01
        )
        total_proposals += iters
        if r.abs_det > best:
            best = r.abs_det
        if target is not None and reached_at is None and r.abs_det >= target:
            reached_at = total_proposals
        run_seed = rng.randrange(1 << 30)

    walltime = time.monotonic() - start
    return BenchResult(
        order_id="",
        method="multi_start",
        best_abs_det=best,
        proposals=total_proposals,
        walltime=walltime,
        reached_target=(target is not None and best >= target),
        proposals_to_target=reached_at,
    )


def _time_budget_replica_exchange(
    n: int, target: int | None, *, seed: int, budget_s: float = TIME_BUDGET
) -> BenchResult:
    """Replica exchange with 30 replicas, run in short sweeps so we can
    monitor the wall budget and the first-reach-target moment."""
    start = time.monotonic()
    rng = random.Random(seed)

    # We run RE in chunks of 500 steps/replica, iterating until budget expires
    # or target reached. This lets us report an approximate
    # proposals-to-target for the first crossing.
    from hadamard_maxdet.parallel import _default_ladder  # local import

    n_replicas = 30
    temps = _default_ladder(n_replicas, t_cold=0.005, t_hot=5.0)

    # Manual RE loop so we can break exactly on budget / target crossing
    # and report proposals-to-target precisely.

    master_rng = random.Random(seed)
    swap_rng = random.Random(seed ^ 0x5A5A5A5A)
    rngs = [random.Random(seed + 1000 * (k + 1)) for k in range(n_replicas)]

    states = []
    energies = []
    for k in range(n_replicas):
        entries = [
            [1 if rngs[k].random() < 0.5 else -1 for _ in range(n)]
            for _ in range(n)
        ]
        states.append(entries)
        energies.append(-_log_abs_det(entries))

    best_e = math.inf
    best_entries = None
    best_abs = 0
    for k in range(n_replicas):
        if energies[k] < best_e:
            best_e = energies[k]
            best_entries = [row[:] for row in states[k]]
            best_abs = int(round(math.exp(-best_e))) if best_e != math.inf else 0

    swap_attempts = [0] * (n_replicas - 1)
    swap_accepts = [0] * (n_replicas - 1)

    exchange_every = 500
    total_proposals = 0
    reached_at: int | None = None

    step = 0
    while time.monotonic() - start < budget_s:
        # One Metropolis sweep per replica.
        for k in range(n_replicas):
            t = temps[k]
            proposal = random_move(states[k], rngs[k])
            if proposal is None:
                continue
            total_proposals += 1
            prop_e = -_log_abs_det(proposal)
            if prop_e == math.inf:
                continue
            delta = prop_e - energies[k]
            if delta <= 0 or rngs[k].random() < math.exp(-delta / max(t, 1e-12)):
                states[k] = proposal
                energies[k] = prop_e
                if prop_e < best_e:
                    best_e = prop_e
                    best_entries = [row[:] for row in proposal]
                    best_abs = abs(integer_determinant(best_entries))
                    if (
                        target is not None
                        and reached_at is None
                        and best_abs >= target
                    ):
                        reached_at = total_proposals

        step += 1
        if step % exchange_every == 0:
            offset = 0 if (step // exchange_every) % 2 == 0 else 1
            for i in range(offset, n_replicas - 1, 2):
                beta_i = 1.0 / max(temps[i], 1e-12)
                beta_j = 1.0 / max(temps[i + 1], 1e-12)
                if energies[i] == math.inf or energies[i + 1] == math.inf:
                    continue
                log_p = (beta_i - beta_j) * (energies[i] - energies[i + 1])
                swap_attempts[i] += 1
                if log_p >= 0 or swap_rng.random() < math.exp(log_p):
                    states[i], states[i + 1] = states[i + 1], states[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    swap_accepts[i] += 1

    walltime = time.monotonic() - start
    cold_rate = (
        swap_accepts[0] / swap_attempts[0] if swap_attempts[0] > 0 else 0.0
    )
    rates = [
        (a / t) if t > 0 else 0.0 for a, t in zip(swap_accepts, swap_attempts)
    ]
    mean_rate = sum(rates) / len(rates) if rates else 0.0
    extra = (
        f"cold_pair_swap={cold_rate:.3f}, mean_swap={mean_rate:.3f}"
    )
    return BenchResult(
        order_id="",
        method="replica_exchange",
        best_abs_det=best_abs,
        proposals=total_proposals,
        walltime=walltime,
        reached_target=(target is not None and best_abs >= target),
        proposals_to_target=reached_at,
        extra=extra,
    )


def _time_budget_population_annealing(
    n: int, target: int | None, *, seed: int, budget_s: float = TIME_BUDGET
) -> BenchResult:
    """Population annealing; calibrate the schedule so one pass is
    comfortably within the budget, then repeat with fresh seeds."""
    start = time.monotonic()
    rng = random.Random(seed)
    best = 0
    total_proposals = 0
    reached_at: int | None = None

    pop = 64
    n_temps = 30
    iters_per_step = 100

    run_seed = seed
    while time.monotonic() - start < budget_s:
        remaining = budget_s - (time.monotonic() - start)
        if remaining < 0.5:
            break
        r = population_annealing(
            n,
            population_size=pop,
            n_temperatures=n_temps,
            iterations_per_step=iters_per_step,
            seed=run_seed,
        )
        total_proposals += r.iterations
        if r.abs_det > best:
            best = r.abs_det
        if target is not None and reached_at is None and r.abs_det >= target:
            reached_at = total_proposals
        run_seed = rng.randrange(1 << 30)

    walltime = time.monotonic() - start
    return BenchResult(
        order_id="",
        method="population_annealing",
        best_abs_det=best,
        proposals=total_proposals,
        walltime=walltime,
        reached_target=(target is not None and best >= target),
        proposals_to_target=reached_at,
    )


def run_bench(
    order_id: str, *, seed: int = 0, budget_s: float = TIME_BUDGET
) -> list[BenchResult]:
    order = BENCHMARKS[order_id]
    n = order.n
    target = order.best_known  # None for open orders
    print(
        f"\n=== {order_id} (n={n}, target={target}, budget={budget_s:.0f}s) ==="
    )
    results = []
    for fn, label in [
        (_time_budget_multi_start, "multi_start"),
        (_time_budget_replica_exchange, "replica_exchange"),
        (_time_budget_population_annealing, "population_annealing"),
    ]:
        r = fn(n, target, seed=seed, budget_s=budget_s)
        r.order_id = order_id
        r.method = label
        results.append(r)
        reached = "YES" if r.reached_target else ("n/a" if target is None else "NO")
        frac = (r.best_abs_det / target) if (target and target > 0) else float("nan")
        extra = f"  [{r.extra}]" if r.extra else ""
        print(
            f"  {label:<22}  best={r.best_abs_det:>14}  "
            f"ratio={frac:6.4f}  "
            f"props={r.proposals:>10}  "
            f"walltime={r.walltime:5.1f}s  "
            f"reached={reached}  "
            f"to_target={r.proposals_to_target}{extra}"
        )
    return results


def main():
    order_ids = ["H11", "H13", "H19"]
    all_results: list[BenchResult] = []
    for oid in order_ids:
        all_results.extend(run_bench(oid, seed=2026, budget_s=TIME_BUDGET))

    # Markdown table summary.
    print("\n\n### Summary table\n")
    print(
        "| order | method | best |det| | target | ratio | proposals | "
        "walltime (s) | reached | proposals_to_target | extra |"
    )
    print(
        "|------|--------|-----------:|-------:|------:|----------:|"
        "-------------:|:-------:|--------------------:|:------|"
    )
    for r in all_results:
        target = BENCHMARKS[r.order_id].best_known
        ratio = (
            f"{r.best_abs_det/target:.4f}" if (target and target > 0) else "n/a"
        )
        tgt = str(target) if target is not None else "open"
        reached = "yes" if r.reached_target else ("n/a" if target is None else "no")
        ptt = str(r.proposals_to_target) if r.proposals_to_target is not None else "-"
        print(
            f"| {r.order_id} | {r.method} | {r.best_abs_det} | {tgt} | "
            f"{ratio} | {r.proposals} | {r.walltime:.1f} | {reached} | "
            f"{ptt} | {r.extra} |"
        )


if __name__ == "__main__":
    main()
