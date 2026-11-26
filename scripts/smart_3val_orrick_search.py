"""Smart Orrick search at H23 — three techniques on the |G| ⊆ {1, 3, 5} submanifold.

This is the 3-valued analog of ``scripts/smart_orrick_search.py``. Seeds are
the 21 ``H23_basin_*.json`` matrices that lie on the |G| ⊆ {1, 3, 5}
submanifold (basins 1, 18, 22 are excluded because they have a |G|=7 outlier).

Output: ``results/smart_3val_search.json`` with per-job records and an
aggregate ``new_multisets`` mapping (multiset → best |det|).

Usage:
    python scripts/smart_3val_orrick_search.py
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.smart_3val_search import (
    AbsClassMultiset,
    KnownClass3,
    WalkResult3,
    boundary_restart_seeds_3val,
    bridge_excursion_3val,
    enumerate_known_3val_classes,
    manifold_walk_3val,
    _triple_verify_abs_det_3val,
)


ORRICK_DET = 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# Worker functions (top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _job_manifold_walk_3val(
    seed_entries: list[list[int]],
    seed_name: str,
    iterations: int,
    locked_multisets_serialised: list[list[list[int]]],
    rng_seed: int,
    target_abs3: int | None,
    abs3_bias: float,
    target_abs5: int | None,
    abs5_bias: float,
) -> dict:
    n = len(seed_entries)
    seed = Matrix(n=n, entries=seed_entries)
    locked = tuple(
        tuple(tuple(p) for p in m) for m in locked_multisets_serialised
    )
    res = manifold_walk_3val(
        seed,
        iterations=iterations,
        locked_multisets=locked,
        rng_seed=rng_seed,
        refresh_every=2000,
        target_abs3=target_abs3,
        abs3_target_bias=abs3_bias,
        target_abs5=target_abs5,
        abs5_target_bias=abs5_bias,
        seed_name=seed_name,
        technique_label="T1_manifold_drift_3val",
    )
    return _serialise_walk(res)


def _job_bridge_excursion_3val(
    seed_entries: list[list[int]],
    seed_name: str,
    excursion_steps: int,
    snap_back_iters: int,
    walk_iters: int,
    locked_multisets_serialised: list[list[list[int]]],
    rng_seed: int,
) -> dict:
    n = len(seed_entries)
    seed = Matrix(n=n, entries=seed_entries)
    locked = tuple(
        tuple(tuple(p) for p in m) for m in locked_multisets_serialised
    )
    res = bridge_excursion_3val(
        seed,
        excursion_steps=excursion_steps,
        snap_back_iters=snap_back_iters,
        walk_iters=walk_iters,
        locked_multisets=locked,
        rng_seed=rng_seed,
        seed_name=seed_name,
        technique_label="T2_bridge_excursion_3val",
    )
    return {
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


def _job_boundary_walk_3val(
    boundary_seed_entries: list[list[int]],
    seed_name: str,
    iterations: int,
    locked_multisets_serialised: list[list[list[int]]],
    rng_seed: int,
) -> dict:
    n = len(boundary_seed_entries)
    seed = Matrix(n=n, entries=boundary_seed_entries)
    locked = tuple(
        tuple(tuple(p) for p in m) for m in locked_multisets_serialised
    )
    res = manifold_walk_3val(
        seed,
        iterations=iterations,
        locked_multisets=locked,
        rng_seed=rng_seed,
        refresh_every=2000,
        seed_name=seed_name,
        technique_label="T3_boundary_restart_3val",
    )
    return _serialise_walk(res)


def _serialise_walk(res: WalkResult3) -> dict:
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
    parser.add_argument("--out", default="results/smart_3val_search.json")
    parser.add_argument("--iterations-t1", type=int, default=200_000)
    parser.add_argument("--iterations-t2-walk", type=int, default=80_000)
    parser.add_argument("--iterations-t3", type=int, default=120_000)
    parser.add_argument("--snap-back-iters", type=int, default=40_000)
    parser.add_argument("--bridge-seeds-per-class", type=int, default=2)
    parser.add_argument("--boundary-seeds-per-class", type=int, default=4)
    parser.add_argument("--t1-seeds-per-class", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument(
        "--max-seed-classes", type=int, default=0,
        help="If >0, limit to top-N seed classes by |det| (for budget).",
    )
    args = parser.parse_args()

    classes = enumerate_known_3val_classes("results")
    if args.max_seed_classes > 0:
        classes = classes[: args.max_seed_classes]
    locked = [c.abs_class_multiset for c in classes]
    locked_serialised = [[list(p) for p in m] for m in locked]
    print(
        f"Loaded {len(classes)} known 3-valued classes (top by |det|):",
    )
    for c in classes[:5]:
        print(f"  {c.name}: |det|={c.abs_det:.4e}  sig={list(c.abs_class_multiset)}")
    if len(classes) > 5:
        print(f"  ...and {len(classes) - 5} more")

    jobs: list[tuple[str, callable, tuple]] = []

    # --- Technique 1: manifold drift, multiple seeds × multiple targets.
    # Targets bias toward different (count_3, count_5) splits. None = no bias.
    # Use a small set of (target_3, target_5) pairs spanning the observed
    # ranges in the 21 known classes (count_3 ∈ [53, 73], count_5 ∈ [1, 8]).
    target_pairs = [
        (None, None),
        (50, 4),
        (60, 2),
        (45, 6),
        (70, 3),
    ]
    bias3 = 0.001
    bias5 = 0.005  # tighter on the rare |G|=5 count
    for kc in classes:
        for ti, (t3, t5) in enumerate(target_pairs):
            for sd in range(args.t1_seeds_per_class):
                seed_name = f"T1.{kc.name}.t3_{t3}.t5_{t5}.s{sd}"
                rng_seed = (
                    1000 + 137 * (hash(kc.name) % 10000) + 17 * ti + sd
                )
                jobs.append(
                    (
                        seed_name,
                        _job_manifold_walk_3val,
                        (
                            kc.matrix.entries,
                            seed_name,
                            args.iterations_t1,
                            locked_serialised,
                            rng_seed,
                            t3,
                            bias3,
                            t5,
                            bias5,
                        ),
                    )
                )

    # --- Technique 2: bridge excursion from each class.
    excursion_depths = [4, 8, 12]
    for kc in classes:
        for di, depth in enumerate(excursion_depths):
            for sd in range(args.bridge_seeds_per_class):
                seed_name = f"T2.{kc.name}.depth{depth}.s{sd}"
                rng_seed = 5000 + 211 * (hash(kc.name) % 10000) + 23 * di + sd
                jobs.append(
                    (
                        seed_name,
                        _job_bridge_excursion_3val,
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
    for kc in classes:
        for k in [2, 4]:
            seeds = boundary_restart_seeds_3val(
                kc.matrix,
                k=k,
                max_seeds=args.boundary_seeds_per_class,
                rng_seed=hash((kc.name, k)) & 0xFFFF,
            )
            for si, seed_M in enumerate(seeds):
                seed_name = f"T3.{kc.name}.k{k}.s{si}"
                rng_seed = 9000 + 311 * (hash(kc.name) % 10000) + 41 * k + si
                jobs.append(
                    (
                        seed_name,
                        _job_boundary_walk_3val,
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

    new_multisets: dict[AbsClassMultiset, dict] = {}
    new_above_orrick: list[dict] = []

    locked_set = {tuple(tuple(p) for p in m) for m in locked_serialised}

    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_job = {}
        for name, fn, payload in jobs:
            fut = pool.submit(fn, *payload)
            future_to_job[fut] = name

        done_count = 0
        for fut in as_completed(future_to_job):
            name = future_to_job[fut]
            done_count += 1
            try:
                res = fut.result()
            except Exception as e:  # pragma: no cover
                results.append({"name": name, "error": repr(e)})
                print(f"  [ERR ] {name}: {e!r}")
                continue
            results.append({"name": name, "result": res})

            walk_obj = res if "visited_signatures" in res else res.get("walk")
            if walk_obj is None:
                # Snap-back failed
                continue
            for entry in walk_obj["visited_signatures"]:
                sig = tuple(tuple(p) for p in entry["signature"])
                d = entry["best_abs_det_in_sig"]
                if sig in locked_set:
                    continue
                rec = new_multisets.setdefault(
                    sig,
                    {
                        "signature": entry["signature"],
                        "best_abs_det": 0,
                        "first_seen_in_job": name,
                    },
                )
                if d > rec["best_abs_det"]:
                    rec["best_abs_det"] = d
                    if rec["first_seen_in_job"] != name:
                        rec["best_seen_in_job"] = name

            best_d = walk_obj["best_feasible_abs_det"]
            best_sig = walk_obj["best_feasible_signature"]
            if best_d > ORRICK_DET and walk_obj["best_feasible_entries"] is not None:
                try:
                    verified = _triple_verify_abs_det_3val(
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
                            f"|det|={verified} ratio={verified/ORRICK_DET:.6f} "
                            f"sig={best_sig}"
                        )
                except RuntimeError as ve:
                    print(f"  [VERIFY-FAIL] {name}: {ve}")

            if done_count % 10 == 0 or done_count == len(jobs):
                print(
                    f"  [progress] {done_count}/{len(jobs)} done, "
                    f"distinct new multisets so far: {len(new_multisets)}"
                )

    walltime = time.monotonic() - t0
    print(f"\nTotal wall-time: {walltime:.1f}s")
    print(f"Total jobs done: {len(results)}")
    print(f"Distinct NEW multisets found (beyond {len(classes)} known): "
          f"{len(new_multisets)}")
    if new_multisets:
        by_det = sorted(new_multisets.values(), key=lambda r: -r["best_abs_det"])
        print("Top 10 new multisets by best |det|:")
        for r in by_det[:10]:
            print(
                f"  sig={r['signature']}  best_det={r['best_abs_det']:>20d}  "
                f"ratio_to_orrick={r['best_abs_det']/ORRICK_DET:.6f}  "
                f"first_in={r['first_seen_in_job']}"
            )
    print(f"Candidates above Orrick: {len(new_above_orrick)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": 23,
        "manifold": "|G_offdiag| in {1, 3, 5}",
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

    # If we found a new best class (largest |det|), save its matrix as a
    # standalone artifact (parallel to results/H23_smart_2val_best_new_class.json).
    if new_multisets:
        by_det = sorted(new_multisets.values(), key=lambda r: -r["best_abs_det"])
        top = by_det[0]
        # Find the per-job entry whose visited_signature contains this top sig
        # and whose best_feasible_entries are populated. We'll prefer the
        # job whose own best_feasible_signature == top sig.
        target_sig = tuple(tuple(p) for p in top["signature"])
        best_entries: list[list[int]] | None = None
        for r in results:
            walk = r.get("result")
            if walk is None:
                continue
            walk_obj = walk if "best_feasible_signature" in walk else walk.get("walk")
            if walk_obj is None:
                continue
            sig = walk_obj.get("best_feasible_signature")
            entries = walk_obj.get("best_feasible_entries")
            if entries is None or sig is None:
                continue
            if tuple(tuple(p) for p in sig) == target_sig and \
               walk_obj.get("best_feasible_abs_det") == top["best_abs_det"]:
                best_entries = entries
                break
        if best_entries is not None:
            verified_det = _triple_verify_abs_det_3val(best_entries)
            best_path = Path("results/H23_smart_3val_best_new_class.json")
            with best_path.open("w") as f:
                json.dump(
                    {
                        "n": 23,
                        "manifold": "|G_offdiag| in {1, 3, 5}",
                        "signature": top["signature"],
                        "abs_det": verified_det,
                        "ratio_to_orrick": verified_det / ORRICK_DET,
                        "first_seen_in_job": top["first_seen_in_job"],
                        "matrix_entries": best_entries,
                        "verification": "Bareiss x2 + transpose Bareiss agreed",
                    },
                    f,
                    indent=2,
                )
            print(f"Wrote {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
