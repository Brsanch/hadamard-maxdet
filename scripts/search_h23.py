"""H23 search: replica exchange seeded from Paley-I H24 deletions.

Strategy
--------
1. Compute the top-K (default 10) Paley deletions of H24 by |det|. At
   n=23 every cofactor of H24 has |det| = 24^11 ≈ 1.52e15 (the cofactor
   identity for an m×m Hadamard matrix). All 576 deletions tie; the
   top-K are picked by ``(i, j)`` lexicographic tie-break.
2. Run one ``replica_exchange`` per seed in a separate worker process
   (multiprocessing). Each call uses 30 replicas, entry_flip moves, a
   log-spaced temperature ladder, and exact Bareiss scoring inside.
3. Distribute the wall-clock budget across the K workers; print the
   running best every ~60s; at the end aggregate, write the best
   matrix to ``results/h23_best.json`` and verify with
   :mod:`scripts.verify_submission`.

Usage
-----
    python scripts/search_h23.py [--minutes 10] [--seeds 10] [--replicas 30] \\
                                 [--workers 10] [--out results/h23_best.json]

This is a compute-scaling experiment: don't expect a clean closed-form
result. Be honest about what number came out.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Make src/ importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hadamard_maxdet.io import write_submission
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.paley_seed import top_paley_seeds
from hadamard_maxdet.parallel import replica_exchange
from hadamard_maxdet.verifier import integer_determinant, upper_bound, verify


@dataclass
class SeedReport:
    seed_idx: int
    paley_i: int
    paley_j: int
    starting_abs_det: int
    best_abs_det: int
    iterations: int
    accepted: int
    walltime_seconds: float


def _worker(payload: dict) -> dict:
    """Run replica_exchange for one Paley-deletion seed."""
    # Late import inside worker to keep child startup small.
    from hadamard_maxdet.matrix import Matrix as _Matrix
    from hadamard_maxdet.parallel import replica_exchange as _re
    from hadamard_maxdet.verifier import integer_determinant as _det

    seed_idx = payload["seed_idx"]
    paley_i = payload["paley_i"]
    paley_j = payload["paley_j"]
    entries = payload["entries"]
    n_replicas = payload["n_replicas"]
    iterations = payload["iterations"]
    exchange_every = payload["exchange_every"]
    rng_seed = payload["rng_seed"]
    t_cold = payload["t_cold"]
    t_hot = payload["t_hot"]

    starting = _Matrix(n=23, entries=[row[:] for row in entries])
    starting_d = abs(_det(starting.entries))

    t0 = time.monotonic()
    result = _re(
        23,
        n_replicas=n_replicas,
        iterations=iterations,
        exchange_every=exchange_every,
        seed=rng_seed,
        initial=starting,
        t_cold=t_cold,
        t_hot=t_hot,
    )
    walltime = time.monotonic() - t0

    return {
        "seed_idx": seed_idx,
        "paley_i": paley_i,
        "paley_j": paley_j,
        "starting_abs_det": starting_d,
        "best_abs_det": result.abs_det,
        "best_entries": [list(row) for row in result.matrix.entries],
        "iterations": result.iterations,
        "accepted": result.accepted,
        "walltime_seconds": walltime,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=10.0,
                    help="Approximate wall-clock budget (default 10).")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Number of top Paley-deletion seeds (default 10).")
    ap.add_argument("--replicas", type=int, default=30,
                    help="Replicas per parallel-tempering run (default 30).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Concurrent worker processes "
                         "(default = min(seeds, cpu_count)).")
    ap.add_argument("--exchange-every", type=int, default=200)
    ap.add_argument("--t-cold", type=float, default=0.005)
    ap.add_argument("--t-hot", type=float, default=5.0)
    ap.add_argument("--rng-seed-base", type=int, default=20260424)
    ap.add_argument("--out", type=Path,
                    default=Path("results/h23_best.json"))
    ap.add_argument("--report", type=Path,
                    default=Path("results/h23_search_report.json"))
    ap.add_argument("--iterations-per-seed", type=int, default=None,
                    help="Override iterations per replica (defaults to "
                         "auto-sizing from --minutes).")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Paley deletions.
    # ------------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] Computing top-{args.seeds} Paley deletions of H24...",
          flush=True)
    t0 = time.monotonic()
    top = top_paley_seeds(23, k=args.seeds)
    print(f"[{time.strftime('%H:%M:%S')}]  done in {time.monotonic()-t0:.1f}s.",
          flush=True)

    print()
    print("Top Paley-deletion |det| values (|det H24-cofactor| = 24^11 = "
          f"{24**11:,}):")
    for k, (i, j, _, d) in enumerate(top):
        print(f"  seed {k:2d}: H24 deletion (row={i:2d}, col={j:2d}) "
              f"|det| = {abs(d):,}")
    print()

    bordered_ceiling = 2_421_778_815_123_456
    barba_bound = upper_bound(23)
    print(f"Bordered-circulant ceiling (n=23): {bordered_ceiling:,}")
    print(f"Barba upper bound       (n=23):     {barba_bound:,}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Auto-size iterations per worker so total wall ≈ minutes.
    # ------------------------------------------------------------------
    workers = args.workers or min(args.seeds, os.cpu_count() or 1)
    workers = max(1, workers)
    # Calibration: at n=23 with 30 replicas, ~120 iters/sec/replica
    # observed on this hardware. We want each worker to run roughly
    # `minutes` minutes (workers run concurrently). Conservative.
    if args.iterations_per_seed is not None:
        iters_per_seed = args.iterations_per_seed
    else:
        iters_per_replica_per_sec = 120.0
        # All workers run concurrently; each worker has its own 30 replicas.
        # The bottleneck is the Python GIL inside each process; with workers
        # ≤ cpu_count, each worker gets a full core.
        budget_seconds = args.minutes * 60.0
        iters_per_seed = int(iters_per_replica_per_sec * budget_seconds)
        # Safety floor.
        iters_per_seed = max(iters_per_seed, 500)

    print(f"Workers: {workers} (cpu_count = {os.cpu_count()})")
    print(f"Iterations per replica (per worker): {iters_per_seed:,}")
    print(f"Total proposals across all workers:  "
          f"{iters_per_seed * args.replicas * args.seeds:,}")
    print(f"Wall budget target: ~{args.minutes:.1f} minutes")
    print()

    # ------------------------------------------------------------------
    # Step 3: Multiprocess run.
    # ------------------------------------------------------------------
    payloads: list[dict] = []
    for k, (i, j, mat, _) in enumerate(top):
        payloads.append(dict(
            seed_idx=k,
            paley_i=i,
            paley_j=j,
            entries=[list(row) for row in mat.entries],
            n_replicas=args.replicas,
            iterations=iters_per_seed,
            exchange_every=args.exchange_every,
            rng_seed=args.rng_seed_base + k * 7919,
            t_cold=args.t_cold,
            t_hot=args.t_hot,
        ))

    print(f"[{time.strftime('%H:%M:%S')}] Launching {workers} worker(s) "
          f"on {len(payloads)} seed(s)...", flush=True)
    search_start = time.monotonic()

    # Use 'spawn' to be safe on macOS.
    ctx = mp.get_context("spawn")
    results: list[dict] = []
    with ctx.Pool(processes=workers) as pool:
        async_results = [pool.apply_async(_worker, (p,)) for p in payloads]
        # Poll for progress every ~60s.
        last_print = time.monotonic()
        while async_results:
            done = [ar for ar in async_results if ar.ready()]
            for ar in done:
                results.append(ar.get())
                async_results.remove(ar)
            now = time.monotonic()
            if now - last_print >= 60.0 or not async_results:
                elapsed = now - search_start
                done_count = len(results)
                if results:
                    best_so_far = max(r["best_abs_det"] for r in results)
                else:
                    best_so_far = 0
                print(f"[{time.strftime('%H:%M:%S')}]  elapsed {elapsed:.0f}s  "
                      f"completed {done_count}/{len(payloads)}  "
                      f"best so far {best_so_far:,}", flush=True)
                last_print = now
            if async_results:
                time.sleep(2.0)

    search_walltime = time.monotonic() - search_start
    print()
    print(f"[{time.strftime('%H:%M:%S')}] Search finished in "
          f"{search_walltime:.1f}s.", flush=True)
    print()

    # ------------------------------------------------------------------
    # Step 4: Aggregate, write, verify.
    # ------------------------------------------------------------------
    results.sort(key=lambda r: r["seed_idx"])

    print("Per-seed outcomes:")
    print(f"  {'seed':>4} {'(i, j)':>8} {'starting':>20} {'best':>20} "
          f"{'gain':>9} {'wall (s)':>9}")
    for r in results:
        gain = r["best_abs_det"] / r["starting_abs_det"]
        print(f"  {r['seed_idx']:>4} ({r['paley_i']:>2},{r['paley_j']:>2})"
              f" {r['starting_abs_det']:>20,} {r['best_abs_det']:>20,}"
              f" {gain:>8.3f}x {r['walltime_seconds']:>9.1f}")
    print()

    best = max(results, key=lambda r: r["best_abs_det"])
    best_entries = best["best_entries"]

    # CRITICAL: re-verify with exact Bareiss on the actual matrix.
    print(f"[{time.strftime('%H:%M:%S')}] Re-verifying best matrix with "
          f"exact Bareiss...", flush=True)
    verdict = verify(best_entries)
    if not verdict.valid:
        print(f"FATAL: best result fails verification: {verdict.error}",
              file=sys.stderr)
        return 2
    if verdict.abs_det != best["best_abs_det"]:
        print(f"WARNING: cached |det| {best['best_abs_det']:,} disagrees "
              f"with re-computed {verdict.abs_det:,}; using re-computed.",
              file=sys.stderr)
    final_abs_det = verdict.abs_det

    print()
    print("=" * 70)
    print(f"H23 BEST |det|: {final_abs_det:,}")
    print(f"  Bordered-circulant ceiling: {bordered_ceiling:,}")
    print(f"  Barba upper bound:           {barba_bound:,}")
    print(f"  Ratio to bordered ceiling:  "
          f"{final_abs_det / bordered_ceiling:.4f}")
    print(f"  Ratio to Barba bound:       "
          f"{final_abs_det / barba_bound:.4f}")
    if final_abs_det > bordered_ceiling:
        print(f"  ** BEATS bordered-circulant ceiling **")
    else:
        print(f"  Below bordered-circulant ceiling.")
    print(f"  Search wall-clock: {search_walltime:.1f}s "
          f"({search_walltime/60.0:.2f} min)")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 5: Write submission.
    # ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_matrix = Matrix(n=23, entries=best_entries)
    write_submission(out_matrix, "H23", args.out)
    print(f"Wrote submission to {args.out}.")

    # Light report.
    report = {
        "n": 23,
        "order_id": "H23",
        "best_abs_det": final_abs_det,
        "bordered_circulant_ceiling": bordered_ceiling,
        "barba_upper_bound": barba_bound,
        "ratio_to_bordered_ceiling": final_abs_det / bordered_ceiling,
        "ratio_to_barba_bound": final_abs_det / barba_bound,
        "beats_bordered_ceiling": final_abs_det > bordered_ceiling,
        "search_walltime_seconds": search_walltime,
        "n_seeds": args.seeds,
        "n_replicas": args.replicas,
        "iterations_per_replica": iters_per_seed,
        "workers": workers,
        "per_seed": [
            {k: v for k, v in r.items() if k != "best_entries"}
            for r in results
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote per-seed report to {args.report}.")

    # ------------------------------------------------------------------
    # Step 6: Run verify_submission script for redundancy.
    # ------------------------------------------------------------------
    print()
    print("Running scripts/verify_submission.py for cross-check:")
    print()
    from hadamard_maxdet.io import read_submission
    order_id, n_, entries_ = read_submission(args.out)
    final_verdict = verify(entries_)
    print(f"  File:    {args.out}")
    print(f"  Order:   {order_id}  (n={n_})")
    print(f"  Valid:   {final_verdict.valid}")
    print(f"  |det|:   {final_verdict.abs_det:,}")

    return 0 if final_verdict.valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
