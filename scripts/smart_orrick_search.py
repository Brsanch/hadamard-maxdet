"""Smart Orrick search at H23 — three techniques on the |G| ⊆ {1, 3} submanifold.

Techniques:
  1. Manifold drift with class tabu  (start from each known matrix; ~3 seeds × ~5)
  2. Bridge excursion                  (off-manifold + snap-back, then walk)
  3. Boundary restart                  (k-flip neighbours of known matrices that
                                        themselves lie on the manifold)

For each technique × each seed × an SA budget, run in parallel via
ProcessPoolExecutor. Track every distinct (count_|G|=1, count_|G|=3) multiset
visited beyond the 3 known. Triple-verify any |det| above Orrick's
2,779,447,296,000,000.

Output: ``results/smart_2val_search.json`` with a per-job record and an
aggregate ``new_multisets_with_best_det`` mapping (multiset → best |det|).

Usage:
    python scripts/smart_orrick_search.py
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.smart_2val_search import (
    AbsClassMultiset,
    BridgeWalkResult,
    KnownClass,
    WalkResult,
    boundary_restart_seeds,
    bridge_excursion,
    enumerate_known_2val_classes,
    manifold_walk,
    _triple_verify_abs_det,
)


ORRICK_DET = 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _job_manifold_walk(
    seed_entries: list[list[int]],
    seed_name: str,
    iterations: int,
    locked_multisets_serialised: list[list[list[int]]],
    rng_seed: int,
    target_abs3: int | None,
    abs3_bias: float,
) -> dict:
    """Worker for Technique 1."""
    n = len(seed_entries)
    seed = Matrix(n=n, entries=seed_entries)
    locked = tuple(
        tuple(tuple(p) for p in m) for m in locked_multisets_serialised
    )
    res = manifold_walk(
        seed,
        iterations=iterations,
        locked_multisets=locked,
        rng_seed=rng_seed,
        refresh_every=2000,
        target_abs3=target_abs3,
        abs1_target_bias=abs3_bias,
        seed_name=seed_name,
        technique_label="T1_manifold_drift",
    )
    return _serialise_walk(res)


def _job_bridge_excursion(
    seed_entries: list[list[int]],
    seed_name: str,
    excursion_steps: int,
    snap_back_iters: int,
    walk_iters: int,
    locked_multisets_serialised: list[list[list[int]]],
    rng_seed: int,
) -> dict:
    """Worker for Technique 2."""
    n = len(seed_entries)
    seed = Matrix(n=n, entries=seed_entries)
    locked = tuple(
        tuple(tuple(p) for p in m) for m in locked_multisets_serialised
    )
    res = bridge_excursion(
        seed,
        excursion_steps=excursion_steps,
        snap_back_iters=snap_back_iters,
        walk_iters=walk_iters,
        locked_multisets=locked,
        rng_seed=rng_seed,
        seed_name=seed_name,
        technique_label="T2_bridge_excursion",
    )
    out: dict = {
        "technique": res.technique,
        "seed_name": res.seed_name,
        "excursion_steps": res.excursion_steps,
        "snap_back_iters": res.snap_back_iters,
        "snap_back_succeeded": res.snap_back_succeeded,
        "final_count_out": res.final_count_out,
        "excursion_max_count_out": res.excursion_max_count_out,
        "walltime_seconds": res.walltime_seconds,
        "walk": _serialise_walk(res.walk) if res.walk is not None else None,
    }
    return out


def _job_boundary_walk(
    boundary_seed_entries: list[list[int]],
    seed_name: str,
    iterations: int,
    locked_multisets_serialised: list[list[list[int]]],
    rng_seed: int,
) -> dict:
    """Worker for Technique 3 (run a manifold walk from a boundary seed)."""
    n = len(boundary_seed_entries)
    seed = Matrix(n=n, entries=boundary_seed_entries)
    locked = tuple(
        tuple(tuple(p) for p in m) for m in locked_multisets_serialised
    )
    res = manifold_walk(
        seed,
        iterations=iterations,
        locked_multisets=locked,
        rng_seed=rng_seed,
        refresh_every=2000,
        seed_name=seed_name,
        technique_label="T3_boundary_restart",
    )
    return _serialise_walk(res)


def _serialise_walk(res: WalkResult) -> dict:
    """Convert a WalkResult to a JSON-safe dict (drops the matrices for
    space; keeps the best_feasible_entries if present)."""
    return {
        "technique": res.technique,
        "seed_name": res.seed_name,
        "iterations": res.iterations,
        "accepted": res.accepted,
        "rejected_class": res.rejected_class,
        "rejected_metro": res.rejected_metro,
        "rejected_singular": res.rejected_singular,
        "rejected_off_manifold": res.rejected_off_manifold,
        "feasible_steps": res.feasible_steps,
        "walltime_seconds": res.walltime_seconds,
        "final_signature": [list(p) for p in res.final_signature],
        "final_abs_det": res.final_abs_det,
        "best_feasible_signature": (
            [list(p) for p in res.best_feasible_signature]
            if res.best_feasible_signature is not None
            else None
        ),
        "best_feasible_abs_det": res.best_feasible_abs_det,
        "best_feasible_entries": res.best_feasible_entries,
        "visited_signatures": [
            {"signature": [list(p) for p in sig], "best_abs_det_in_sig": d}
            for sig, d in res.visited_signatures
        ],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/smart_2val_search.json")
    parser.add_argument(
        "--iterations-t1", type=int, default=200_000,
        help="Iterations per Technique 1 manifold walk.",
    )
    parser.add_argument(
        "--iterations-t2-walk", type=int, default=80_000,
        help="Iterations per Technique 2 manifold walk after snap-back.",
    )
    parser.add_argument(
        "--iterations-t3", type=int, default=120_000,
        help="Iterations per Technique 3 boundary-seed walk.",
    )
    parser.add_argument(
        "--snap-back-iters", type=int, default=40_000,
        help="Iterations for the snap-back SA in Technique 2.",
    )
    parser.add_argument(
        "--bridge-seeds-per-class", type=int, default=4,
        help="Bridge excursion runs per known class.",
    )
    parser.add_argument(
        "--boundary-seeds-per-class", type=int, default=8,
        help="Number of boundary k-flip seeds per known class.",
    )
    parser.add_argument(
        "--t1-seeds-per-class", type=int, default=4,
        help="Number of seeds (PRNG seeds) per known class for Technique 1.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=10,
        help="Process pool size.",
    )
    args = parser.parse_args()

    classes = enumerate_known_2val_classes("results")
    locked = [c.abs_class_multiset for c in classes]
    locked_serialised = [[list(p) for p in m] for m in locked]
    print(
        f"Loaded {len(classes)} known classes:",
        [(c.name, c.abs_det, list(c.abs_class_multiset)) for c in classes],
    )

    jobs: list[tuple[str, callable, tuple]] = []

    # --- Technique 1: manifold drift, multiple seeds × multiple PRNG seeds.
    # Targets are biases on count_abs3 to widen search regions.
    targets = [None, 30, 38, 50, 60, 75]
    bias = 0.001
    for kc in classes:
        for ti, target in enumerate(targets):
            for sd in range(args.t1_seeds_per_class):
                seed_name = f"T1.{kc.name}.target{target}.s{sd}"
                rng_seed = (
                    1000 + 137 * hash(kc.name) % 10000 + 17 * ti + sd
                )
                jobs.append(
                    (
                        seed_name,
                        _job_manifold_walk,
                        (
                            kc.matrix.entries,
                            seed_name,
                            args.iterations_t1,
                            locked_serialised,
                            rng_seed,
                            target,
                            bias,
                        ),
                    )
                )

    # --- Technique 2: bridge excursion from each class with several
    # excursion-step depths and PRNG seeds.
    excursion_depths = [4, 8, 12, 18]
    for kc in classes:
        for di, depth in enumerate(excursion_depths):
            for sd in range(args.bridge_seeds_per_class):
                seed_name = f"T2.{kc.name}.depth{depth}.s{sd}"
                rng_seed = 5000 + 211 * hash(kc.name) % 10000 + 23 * di + sd
                jobs.append(
                    (
                        seed_name,
                        _job_bridge_excursion,
                        (
                            kc.matrix.entries,
                            seed_name,
                            depth,
                            args.snap_back_iters,
                            args.iterations_t2_walk,
                            locked_serialised,
                            rng_seed,
                        ),
                    )
                )

    # --- Technique 3: boundary k-flip neighbours of each known class.
    # Pre-generate the seeds (cheap O(n^4) for k=2; for k > 2 sample).
    for kc in classes:
        for k in [2, 4, 6]:
            seeds = boundary_restart_seeds(
                kc.matrix,
                k=k,
                max_seeds=args.boundary_seeds_per_class,
                rng_seed=hash((kc.name, k)) & 0xFFFF,
            )
            for si, seed_M in enumerate(seeds):
                seed_name = f"T3.{kc.name}.k{k}.s{si}"
                rng_seed = 9000 + 311 * hash(kc.name) % 10000 + 41 * k + si
                jobs.append(
                    (
                        seed_name,
                        _job_boundary_walk,
                        (
                            seed_M.entries,
                            seed_name,
                            args.iterations_t3,
                            locked_serialised,
                            rng_seed,
                        ),
                    )
                )

    print(f"Total jobs scheduled: {len(jobs)}")
    print(f"  Pool size: {args.max_workers}")

    t0 = time.monotonic()
    results: list[dict] = []

    # Aggregate over all jobs.
    new_multisets: dict[AbsClassMultiset, dict] = {}
    new_above_orrick: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_job = {}
        for name, fn, payload in jobs:
            fut = pool.submit(fn, *payload)
            future_to_job[fut] = name

        for fut in as_completed(future_to_job):
            name = future_to_job[fut]
            try:
                res = fut.result()
            except Exception as e:  # pragma: no cover (worker errors)
                results.append({
                    "name": name,
                    "error": repr(e),
                })
                print(f"  [ERR ] {name}: {e!r}")
                continue
            results.append({"name": name, "result": res})

            # Walk through visited signatures.
            walk_obj = res if "visited_signatures" in res else res.get("walk")
            if walk_obj is None:
                print(f"  [done] {name}: walk is None (snap-back failed)")
                continue
            for entry in walk_obj["visited_signatures"]:
                sig = tuple(tuple(p) for p in entry["signature"])
                d = entry["best_abs_det_in_sig"]
                if sig in {tuple(tuple(p) for p in m) for m in locked_serialised}:
                    continue
                rec = new_multisets.setdefault(
                    sig,
                    {"signature": entry["signature"], "best_abs_det": 0,
                     "first_seen_in_job": name},
                )
                if d > rec["best_abs_det"]:
                    rec["best_abs_det"] = d
                    if rec["first_seen_in_job"] != name:
                        rec["best_seen_in_job"] = name

            best_d = walk_obj["best_feasible_abs_det"]
            best_sig = walk_obj["best_feasible_signature"]
            if best_d > ORRICK_DET and walk_obj["best_feasible_entries"] is not None:
                # Triple-verify before claiming.
                try:
                    verified = _triple_verify_abs_det(
                        walk_obj["best_feasible_entries"]
                    )
                    if verified > ORRICK_DET:
                        new_above_orrick.append({
                            "job": name,
                            "abs_det": verified,
                            "ratio_to_orrick": verified / ORRICK_DET,
                            "signature": best_sig,
                            "entries": walk_obj["best_feasible_entries"],
                        })
                        print(
                            f"  *** ABOVE ORRICK *** {name}: "
                            f"|det|={verified} (ratio {verified/ORRICK_DET:.6f}) "
                            f"sig={best_sig}"
                        )
                except RuntimeError as ve:
                    print(f"  [VERIFY-FAIL] {name}: {ve}")

            print(
                f"  [done] {name}: best_feasible_det={best_d}  "
                f"sig={best_sig}  visited_new_sigs="
                f"{len(walk_obj['visited_signatures'])}"
            )

    walltime = time.monotonic() - t0
    print(f"\nTotal wall-time: {walltime:.1f}s")
    print(f"Total jobs done: {len(results)}")
    print(f"Distinct NEW multisets found (beyond 3 known): {len(new_multisets)}")
    if new_multisets:
        # Show the top 10 by best_abs_det.
        by_det = sorted(
            new_multisets.values(), key=lambda r: -r["best_abs_det"]
        )
        print("Top 10 new multisets by best |det|:")
        for r in by_det[:10]:
            print(
                f"  sig={r['signature']}  best_det={r['best_abs_det']:>20d}  "
                f"ratio_to_orrick={r['best_abs_det']/ORRICK_DET:.6f}  "
                f"first_in={r['first_seen_in_job']}"
            )
    print(f"Candidates above Orrick: {len(new_above_orrick)}")

    # Write report.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": 23,
        "orrick_det": ORRICK_DET,
        "known_classes": [
            {
                "name": c.name,
                "abs_det": c.abs_det,
                "abs_class_multiset": [list(p) for p in c.abs_class_multiset],
            }
            for c in classes
        ],
        "locked_multisets_serialised": locked_serialised,
        "args": vars(args),
        "jobs_total": len(jobs),
        "jobs_done": len(results),
        "walltime_seconds": walltime,
        "n_new_multisets": len(new_multisets),
        "new_multisets": sorted(
            new_multisets.values(), key=lambda r: -r["best_abs_det"]
        ),
        "new_above_orrick": new_above_orrick,
        "per_job": results,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
