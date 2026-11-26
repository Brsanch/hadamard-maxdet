"""Sweep 2-valued-Gram-support SA across k_target ∈ {30,35,40,45,50,55,60,65}.

Energy:
    E(M) = -log|det(M)|
         + mu * (count_out)^2
         + lambda * (count_pos - k_target)^2

For each k_target, run two seed sources:
  (a) Random init × several seeds
  (b) Orrick seed perturbed by p ∈ {0, 5, 10, 25} flips × several seeds

Per run we record:
  - reached feasibility (count_out == 0 anywhere)?
  - best feasible |det| at any k
  - best feasible (k, |det|)
  - best |det| ignoring feasibility (sanity)

Any (k_target, |det|) > (45, 2,779,447,296,000,000) gets exact-Bareiss
verified 3× and saved as a record candidate.

Run: ``python scripts/gram_2val_sweep.py``

Outputs:
  - ``results/gram_2val_sweep.json`` — per-(k, source, seed) summary
  - ``results/gram_2val_sweep.log`` — line log
  - ``results/H23_above_orrick.json`` — only if a record candidate is found
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hadamard_maxdet.gram_2val import (
    Gram2ValState,
    count_offdiag_value,
    count_outside_support,
    sa_2valued,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


REPO_ROOT = Path(__file__).resolve().parents[1]
ORRICK_PATH = REPO_ROOT / "results" / "H23_orrick2003_record.json"
RESULTS_PATH = REPO_ROOT / "results" / "gram_2val_sweep.json"
LOG_PATH = REPO_ROOT / "results" / "gram_2val_sweep.log"
RECORD_PATH = REPO_ROOT / "results" / "H23_above_orrick.json"

ORRICK_DET = 2_779_447_296_000_000

K_TARGETS = [30, 35, 40, 45, 50, 55, 60, 65]
RANDOM_SEEDS = [101, 202, 303, 404, 505]
ORRICK_PERTURB_PLANS = [
    # (perturb_flips, seed)
    (0, 11),     # exact Orrick (only meaningful at k=45)
    (5, 17),
    (5, 19),
    (10, 23),
    (10, 29),
    (25, 31),
    (50, 37),
]
ITERATIONS = 3_000_000
T_START = 2.0
T_END = 0.001
MU_SUPPORT = 0.001
LAMBDA_COUNT = 0.05


def _load_orrick_entries() -> list[list[int]]:
    with open(ORRICK_PATH) as f:
        d = json.load(f)
    return [list(r) for r in d["entries"]]


def _verify_triple(entries: list[list[int]]) -> dict:
    """Re-verify abs_det three times in a row; return the agreed value or
    flag a discrepancy.
    """
    d1 = abs(integer_determinant(entries))
    d2 = abs(integer_determinant([row[:] for row in entries]))
    d3 = abs(integer_determinant([row[:] for row in entries]))
    agree = d1 == d2 == d3
    return {"abs_det": d1, "agree": agree, "d1": d1, "d2": d2, "d3": d3}


def _run_one(
    k_target: int,
    source: str,
    seed: int,
    initial_entries: list[list[int]] | None,
    perturb_initial: int,
    perturb_seed: int | None,
    iterations: int,
    t_start: float,
    t_end: float,
    mu_support: float,
    lambda_count: float,
) -> dict:
    """One SA run; called inside a worker process."""
    initial = (
        Matrix(n=23, entries=[row[:] for row in initial_entries])
        if initial_entries is not None
        else None
    )
    t0 = time.monotonic()
    res = sa_2valued(
        n=23,
        target_k_pos=k_target,
        iterations=iterations,
        t_start=t_start,
        t_end=t_end,
        mu_support=mu_support,
        lambda_count=lambda_count,
        pos_value=3,
        neg_value=-1,
        seed=seed,
        initial=initial,
        perturb_initial=perturb_initial,
        perturb_rng_seed=perturb_seed,
        refresh_every=2000,
    )
    dt = time.monotonic() - t0

    # Defensive: re-verify exactly the bests, including the feasible one.
    best_abs_det = abs(integer_determinant(res.matrix.entries))
    best_feasible_abs_det = (
        abs(integer_determinant(res.best_feasible_entries))
        if res.best_feasible_entries is not None
        else 0
    )

    # Also compute final count_pos & count_out from the BEST feasible matrix
    # (so reported k_pos is an exact recompute, not the carried state).
    if res.best_feasible_entries is not None:
        bfe = res.best_feasible_entries
        bf_pos = count_offdiag_value(bfe, 3)
        bf_out = count_outside_support(bfe, {-1, 3})
    else:
        bf_pos = 0
        bf_out = -1

    return {
        "k_target": k_target,
        "source": source,
        "seed": seed,
        "perturb_initial": perturb_initial,
        "perturb_seed": perturb_seed,
        "wall_seconds": dt,
        "iterations": res.iterations,
        "accepted": res.accepted,
        "refreshes": res.refreshes,
        "final_count_pos": res.final_count_pos,
        "final_count_out": res.final_count_out,
        "best_abs_det_any": best_abs_det,
        "feasible_ever": res.feasible_ever,
        "best_feasible_abs_det": best_feasible_abs_det,
        "best_feasible_count_pos_carried": res.best_feasible_count_pos,
        "best_feasible_count_pos_recomputed": bf_pos,
        "best_feasible_count_out_recomputed": bf_out,
        # Keep the best feasible matrix entries so the parent can verify and
        # potentially record. Caller decides what to do.
        "best_feasible_entries": res.best_feasible_entries,
    }


def _build_jobs(orrick_entries: list[list[int]]) -> list[dict]:
    """Build the full per-run job list."""
    jobs: list[dict] = []
    for k in K_TARGETS:
        # Random-init runs.
        for sd in RANDOM_SEEDS:
            jobs.append({
                "k_target": k,
                "source": "random",
                "seed": sd,
                "initial_entries": None,
                "perturb_initial": 0,
                "perturb_seed": None,
            })
        # Orrick-perturbed runs.
        for ptb, pseed in ORRICK_PERTURB_PLANS:
            # ptb=0 only meaningful at k=45 (it's literally Orrick itself).
            if ptb == 0 and k != 45:
                continue
            jobs.append({
                "k_target": k,
                "source": "orrick_perturb_{}".format(ptb),
                "seed": pseed,
                "initial_entries": orrick_entries,
                "perturb_initial": ptb,
                "perturb_seed": pseed + ptb,
            })
    return jobs


def _job_worker(job: dict) -> dict:
    return _run_one(
        k_target=job["k_target"],
        source=job["source"],
        seed=job["seed"],
        initial_entries=job["initial_entries"],
        perturb_initial=job["perturb_initial"],
        perturb_seed=job["perturb_seed"],
        iterations=ITERATIONS,
        t_start=T_START,
        t_end=T_END,
        mu_support=MU_SUPPORT,
        lambda_count=LAMBDA_COUNT,
    )


def main() -> None:
    log_lines: list[str] = []

    def log(s: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        log_lines.append(line)

    orrick_entries = _load_orrick_entries()
    orrick_det_check = abs(integer_determinant(orrick_entries))
    assert orrick_det_check == ORRICK_DET, (
        f"Orrick seed file does not match expected |det|: "
        f"got {orrick_det_check}, expected {ORRICK_DET}"
    )
    log(f"Orrick seed verified at |det|={ORRICK_DET}")

    jobs = _build_jobs(orrick_entries)
    log(f"Total runs: {len(jobs)} ({ITERATIONS} iterations each)")
    log(
        f"Energy: mu_support={MU_SUPPORT}, lambda_count={LAMBDA_COUNT}, "
        f"t_start={T_START}, t_end={T_END}"
    )

    # Parallelize. Be conservative on workers (n=23 SA is CPU-bound but light;
    # 4-6 workers fits the budget on M2 Ultra without saturating apfsd).
    max_workers = min(6, max(1, (os.cpu_count() or 4) - 2))
    log(f"Running {max_workers} workers in parallel.")

    t0 = time.monotonic()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(_job_worker, j): j for j in jobs}
        for fut in as_completed(future_to_job):
            try:
                r = fut.result()
            except Exception as e:
                j = future_to_job[fut]
                log(f"FAIL job k={j['k_target']} {j['source']} seed={j['seed']}: {e}")
                continue
            results.append(r)
            log(
                f"k={r['k_target']:>2d} {r['source']:<22s} seed={r['seed']:>4d}: "
                f"wall={r['wall_seconds']:5.1f}s acc={r['accepted']:>6d} "
                f"final(out={r['final_count_out']}, pos={r['final_count_pos']}) "
                f"any|det|={r['best_abs_det_any']:.3e} "
                f"fe={int(r['feasible_ever'])} "
                f"bf|det|={r['best_feasible_abs_det']:.3e} "
                f"bfk={r['best_feasible_count_pos_recomputed']}"
            )
    log(f"All runs complete: wall={time.monotonic()-t0:.1f}s")

    # Per-(k_target, source) best-feasible aggregation.
    agg: dict[tuple[int, str], dict] = {}
    for r in results:
        key = (r["k_target"], r["source"])
        cur = agg.get(key)
        if cur is None or r["best_feasible_abs_det"] > cur["best_feasible_abs_det"]:
            agg[key] = r

    # Per-k_target, by-seed-source table.
    per_k: dict[int, dict] = {}
    for k in K_TARGETS:
        rs_random = [r for r in results if r["k_target"] == k and r["source"] == "random"]
        rs_orrick = [r for r in results if r["k_target"] == k and r["source"].startswith("orrick_perturb")]
        per_k[k] = {
            "random_best_feasible_abs_det": max(
                (r["best_feasible_abs_det"] for r in rs_random), default=0
            ),
            "random_feasible_runs": sum(1 for r in rs_random if r["feasible_ever"]),
            "random_total_runs": len(rs_random),
            "orrick_best_feasible_abs_det": max(
                (r["best_feasible_abs_det"] for r in rs_orrick), default=0
            ),
            "orrick_feasible_runs": sum(1 for r in rs_orrick if r["feasible_ever"]),
            "orrick_total_runs": len(rs_orrick),
            "best_feasible_k_pos_recomputed": max(
                (r["best_feasible_count_pos_recomputed"] for r in (rs_random + rs_orrick) if r["feasible_ever"]),
                default=-1,
            ),
        }

    # Find the global best feasible across all runs.
    best_overall = max(
        (r for r in results if r["feasible_ever"]),
        default=None,
        key=lambda r: r["best_feasible_abs_det"],
    )

    record_candidate: dict | None = None
    if best_overall is not None and best_overall["best_feasible_abs_det"] > ORRICK_DET:
        log(
            f"!!! RECORD CANDIDATE: k_target={best_overall['k_target']}, "
            f"source={best_overall['source']}, seed={best_overall['seed']}, "
            f"|det|={best_overall['best_feasible_abs_det']} > Orrick {ORRICK_DET}"
        )
        # Triple-verify.
        ent = best_overall["best_feasible_entries"]
        triple = _verify_triple(ent)
        log(f"Triple verify: {triple}")
        if triple["agree"] and triple["abs_det"] > ORRICK_DET:
            record_candidate = {
                "abs_det": triple["abs_det"],
                "k_pos": best_overall["best_feasible_count_pos_recomputed"],
                "k_target": best_overall["k_target"],
                "source": best_overall["source"],
                "seed": best_overall["seed"],
                "entries": ent,
                "verified_triple": triple,
            }
            with open(RECORD_PATH, "w") as f:
                json.dump(record_candidate, f, indent=2)
            log(f"WROTE record candidate to {RECORD_PATH}")
    else:
        log(
            f"No record candidate. Best feasible across all runs: "
            f"|det|={best_overall['best_feasible_abs_det'] if best_overall else 0:.3e} "
            f"(Orrick: {ORRICK_DET:.3e})"
        )

    summary = {
        "n": 23,
        "orrick_det": ORRICK_DET,
        "k_targets": K_TARGETS,
        "iterations_per_run": ITERATIONS,
        "schedule": {"t_start": T_START, "t_end": T_END,
                     "mu_support": MU_SUPPORT, "lambda_count": LAMBDA_COUNT},
        "n_runs": len(results),
        "wall_total_seconds": time.monotonic() - t0,
        "per_k_target": per_k,
        # Per-(k, source) best (without entries to keep the JSON small).
        "per_k_source_best": {
            f"k={k}|{src}": {
                "best_feasible_abs_det": v["best_feasible_abs_det"],
                "feasible_ever": v["feasible_ever"],
                "best_feasible_count_pos_recomputed": v["best_feasible_count_pos_recomputed"],
                "seed": v["seed"],
                "perturb_initial": v["perturb_initial"],
                "wall_seconds": v["wall_seconds"],
                "accepted": v["accepted"],
                "best_abs_det_any": v["best_abs_det_any"],
            }
            for (k, src), v in agg.items()
        },
        "global_best_feasible_abs_det": (
            best_overall["best_feasible_abs_det"] if best_overall else 0
        ),
        "global_best_feasible_at_k_target": (
            best_overall["k_target"] if best_overall else None
        ),
        "global_best_feasible_count_pos": (
            best_overall["best_feasible_count_pos_recomputed"] if best_overall else None
        ),
        "any_run_beat_orrick": (
            best_overall is not None and best_overall["best_feasible_abs_det"] > ORRICK_DET
        ),
        "record_candidate_written": record_candidate is not None,
    }

    # Strip "best_feasible_entries" from the per-run records (they're large).
    slim_results = []
    for r in results:
        rr = dict(r)
        rr.pop("best_feasible_entries", None)
        slim_results.append(rr)
    summary["per_run"] = slim_results

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"WROTE summary to {RESULTS_PATH}")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"WROTE log to {LOG_PATH}")


if __name__ == "__main__":
    main()
