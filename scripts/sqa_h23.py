"""SQA experiment at n=23 — domain-analogy iter 3 (quantum_mechanics).

Runs multiple independent SQA (Simulated Quantum Annealing via Suzuki-Trotter
path-integral MC) searches at n=23 within a 10-minute wall-clock budget.

Protocol
--------
* 4 Trotter replicas (P=4); single temperature T=0.05
* Γ annealed geometrically from 2.0 → 0.005 over the full wall-clock budget
* Each independent restart uses a different RNG seed
* Best result across all restarts is triple-verified via integer_determinant
* Print progress every 30 s; report final diagnostics
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

from hadamard_maxdet.sqa import (
    ORRICK_ABS_DET_H23,
    SQAResult,
    simulated_quantum_annealing,
    triple_verify,
)

WALL_CAP_S = 600.0      # 10 minutes total
N = 23
N_TROTTER = 4
TEMPERATURE = 0.05
GAMMA_START = 2.0
GAMMA_END = 0.005
# Wall budget per restart; multiple restarts fit within WALL_CAP_S
RESTART_BUDGET_S = 150.0  # 4 restarts × 150 s ≈ 600 s

seeds = [2026_04_26, 314159, 271828, 161803]


def run() -> None:
    overall_best: int = 0
    overall_best_matrix: list[list[int]] = []
    overall_start = time.monotonic()

    for run_idx, seed in enumerate(seeds):
        elapsed = time.monotonic() - overall_start
        remaining = WALL_CAP_S - elapsed
        if remaining < 5.0:
            print(f"  [budget exhausted before restart {run_idx}]")
            break

        budget = min(RESTART_BUDGET_S, remaining)
        print(
            f"\n=== SQA restart {run_idx + 1}/{len(seeds)}  seed={seed}"
            f"  budget={budget:.0f}s ==="
        )

        result: SQAResult = simulated_quantum_annealing(
            n=N,
            n_trotter=N_TROTTER,
            gamma_start=GAMMA_START,
            gamma_end=GAMMA_END,
            temperature=TEMPERATURE,
            wall_cap_s=budget,
            seed=seed,
            verbose=True,
            verbose_interval_s=30.0,
        )

        ratio = result.best_abs_det / ORRICK_ABS_DET_H23
        print(
            f"  restart {run_idx + 1} finished: best={result.best_abs_det:,}"
            f"  ({ratio:.6f}×Orrick)  steps={result.total_steps:,}"
            f"  wall={result.walltime_seconds:.1f}s"
        )
        print(f"  per-replica finals: {result.per_replica_best}")

        if result.best_abs_det > overall_best:
            overall_best = result.best_abs_det
            overall_best_matrix = [row[:] for row in result.best_matrix]
            print(f"  ** New overall best: {overall_best:,} **")

    # -------------------------------------------------------------------
    # Final summary and triple-verify
    # -------------------------------------------------------------------
    total_wall = time.monotonic() - overall_start
    ratio = overall_best / ORRICK_ABS_DET_H23

    print(f"\n{'='*60}")
    print(f"SQA H23 experiment complete.  Total wall time: {total_wall:.1f}s")
    print(f"Best |det| found:  {overall_best:,}")
    print(f"Orrick record:     {ORRICK_ABS_DET_H23:,}")
    print(f"Ratio to Orrick:   {ratio:.6f}")
    print(f"{'='*60}")

    if overall_best_matrix:
        print("\nTriple-verifying best matrix (M, M^T, fresh copy) …")
        d_M, d_MT, d_copy = triple_verify(overall_best_matrix)
        print(f"  det(M)      = {d_M}")
        print(f"  det(M^T)    = {d_MT}")
        print(f"  det(copy)   = {d_copy}")
        assert abs(d_M) == abs(d_MT) == abs(d_copy) == overall_best, (
            f"Triple-verify mismatch: {abs(d_M)}, {abs(d_MT)}, {abs(d_copy)}"
            f" vs reported {overall_best}"
        )
        print("  Triple-verify PASSED.")

        if overall_best > ORRICK_ABS_DET_H23:
            print("\n*** NEW RECORD — BEATS ORRICK ***")
            # Print matrix for manual inspection
            print("Best matrix (±1 rows):")
            for row in overall_best_matrix:
                print("  " + " ".join(f"{v:+d}" for v in row))
        else:
            deficit = ORRICK_ABS_DET_H23 - overall_best
            print(
                f"\nDid not beat Orrick.  Deficit: {deficit:,}"
                f"  ({deficit / ORRICK_ABS_DET_H23:.4%} below record)"
            )


if __name__ == "__main__":
    run()
