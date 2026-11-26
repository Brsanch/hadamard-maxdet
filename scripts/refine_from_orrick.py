"""Refinement experiment seeded from results/H23_orrick2003_record.json.

Outcomes both interesting:
  (1) SA stays at the seed across every λ × variant ⇒ seed is a strict local
      maximum under our move sets.
  (2) SA finds |det| > seed_abs_det ⇒ flag prominently, triple-verify.

Variants:
  (a) sa_incremental — long entry-flip SA, t∈[0.5, 0.005]
  (b) sa_m1_anchored — λ ∈ {0.1, 1.0, 10.0}
  (c) sa_with_class_tabu — Orrick's signature LOCKED, then unlocked bridge
  (d) sa_targeted     — +3-pair Move-C
  (e) Heavy multistart — Orrick + k random flips, k ∈ {0, 1, 3, 5, 7, 10}

Per-variant budget: ~5 minutes wall (cap), several seeds.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hadamard_maxdet.class_tabu import class_signature, sa_with_class_tabu  # noqa: E402
from hadamard_maxdet.gram_energy import sa_targeted  # noqa: E402
from hadamard_maxdet.incremental import sa_incremental  # noqa: E402
from hadamard_maxdet.m1_anchor import sa_m1_anchored, t_support_cardinality  # noqa: E402
from hadamard_maxdet.matrix import Matrix  # noqa: E402
from hadamard_maxdet.verifier import integer_determinant, upper_bound  # noqa: E402

SEED_FILE = REPO_ROOT / "results" / "H23_orrick2003_record.json"
BTC_FILE = REPO_ROOT / "results" / "H23_attack_best.json"
ABOVE_OUT = REPO_ROOT / "results" / "H23_above_orrick.json"
SUMMARY_OUT = REPO_ROOT / "results" / "H23_refine_from_orrick_summary.json"
N = 23
ORRICK_PUBLISHED = 2_779_447_296_000_000  # 2^22 * 3 * 5^6 * 67 * 211


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def gram_offdiag_multiset(M):
    A = np.asarray(M, dtype=int)
    G = (A @ A.T)
    n = A.shape[0]
    counts: dict[int, int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            v = int(G[i, j])
            counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items()))


def singular_value_multiplicities(M, decimals=6):
    A = np.asarray(M, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    s = np.round(s, decimals)
    out: dict[float, int] = {}
    for v in s.tolist():
        out[v] = out.get(v, 0) + 1
    return [(float(k), v) for k, v in sorted(out.items(), reverse=True)]


def hamming_distance(A, B):
    a = np.asarray(A, dtype=int)
    b = np.asarray(B, dtype=int)
    return int(np.sum(a != b))


def block_partition_fit(M):
    """Crude detector: project under all sign-flips of each row to look for
    a block partition where rows split into groups with intra-block Gram
    >> inter-block. Returns the best (sizes, score) heuristic."""
    A = np.asarray(M, dtype=int)
    n = A.shape[0]
    G = A @ A.T
    # Greedy clustering by Gram-similarity (with sign flips allowed).
    abs_G = np.abs(G)
    np.fill_diagonal(abs_G, 0)
    # Group rows by the row whose abs-Gram is the largest neighbor.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x, y = find(x), find(y)
        if x != y:
            parent[x] = y

    threshold = max(7, int(n * 0.4))
    for i in range(n):
        for j in range(i + 1, n):
            if abs_G[i, j] >= threshold:
                union(i, j)

    groups: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        groups[r] = groups.get(r, 0) + 1
    sizes = sorted(groups.values(), reverse=True)
    return sizes


def fingerprint(M, label):
    sig = class_signature(M)
    sv_mult = singular_value_multiplicities(M)
    tcard = t_support_cardinality(M)
    bp = block_partition_fit(M)
    return {
        "label": label,
        "abs_det": abs(integer_determinant(M)),
        "class_signature": [list(p) for p in sig],
        "t_support_cardinality": tcard,
        "singular_value_multiplicities": sv_mult,
        "n_distinct_singular_values": len(sv_mult),
        "block_partition_sizes": bp,
        "gram_offdiag_multiset": gram_offdiag_multiset(M),
    }


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _make_initial(entries, k_random_flips, seed):
    rng = random.Random(seed)
    M = [row[:] for row in entries]
    n = len(M)
    flipped = set()
    while len(flipped) < k_random_flips:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if (i, j) not in flipped:
            flipped.add((i, j))
            M[i][j] = -M[i][j]
    return Matrix(n=n, entries=M)


def worker_sa_incremental(args):
    label, entries, k_flips, seed, iters, t_start, t_end, wall = args
    init = _make_initial(entries, k_flips, seed)
    t0 = time.monotonic()
    res = sa_incremental(
        N, iterations=iters, t_start=t_start, t_end=t_end,
        seed=seed, initial=init, refresh_every=2000,
    )
    walltime = time.monotonic() - t0
    return {
        "label": label,
        "k_random_flips": k_flips,
        "seed": seed,
        "abs_det": int(res.abs_det),
        "walltime_seconds": walltime,
        "iterations": res.iterations,
        "accepted": res.accepted,
        "entries": res.matrix.entries,
    }


def worker_sa_m1(args):
    label, entries, lam, seed, iters, t_start, t_end = args
    init = _make_initial(entries, 0, seed)
    t0 = time.monotonic()
    res = sa_m1_anchored(
        N, iterations=iters, t_start=t_start, t_end=t_end,
        lambda_m1=lam, seed=seed, initial=init, refresh_every=2000,
    )
    walltime = time.monotonic() - t0
    return {
        "label": label,
        "lambda_m1": lam,
        "seed": seed,
        "abs_det": int(res.abs_det),
        "walltime_seconds": walltime,
        "iterations": res.iterations,
        "accepted": res.accepted,
        "entries": res.matrix.entries,
    }


def worker_sa_class_tabu(args):
    label, entries, lock_seed_sig, seed, iters, t_start, t_end = args
    init = _make_initial(entries, 0, seed)
    locked = []
    if lock_seed_sig:
        locked = [class_signature(entries)]
    t0 = time.monotonic()
    res = sa_with_class_tabu(
        init, iterations=iters, locked_signatures=locked,
        t_start=t_start, t_end=t_end, seed=seed, refresh_every=2000,
    )
    walltime = time.monotonic() - t0
    return {
        "label": label,
        "locked_seed_signature": lock_seed_sig,
        "seed": seed,
        "abs_det": int(res.abs_det),
        "walltime_seconds": walltime,
        "iterations": res.iterations,
        "accepted": res.accepted,
        "rejected_class": res.rejected_by_class_tabu,
        "entries": res.matrix.entries,
    }


def worker_sa_targeted(args):
    label, entries, seed, iters, t_start, t_end, targeted_fraction = args
    init = _make_initial(entries, 0, seed)
    t0 = time.monotonic()
    res = sa_targeted(
        init, iterations=iters, t_start=t_start, t_end=t_end,
        seed=seed, targeted_fraction=targeted_fraction, refresh_every=2000,
    )
    walltime = time.monotonic() - t0
    return {
        "label": label,
        "seed": seed,
        "targeted_fraction": targeted_fraction,
        "abs_det": int(res.abs_det),
        "walltime_seconds": walltime,
        "iterations": res.iterations,
        "accepted": res.accepted,
        "entries": res.matrix.entries,
    }


# ---------------------------------------------------------------------------
# Triple-verify helper
# ---------------------------------------------------------------------------

def triple_verify(entries, expected_abs_det):
    v1 = abs(integer_determinant(entries))
    v2 = abs(integer_determinant([row[:] for row in entries]))
    # Reload-from-disk simulation via JSON round-trip
    s = json.dumps(entries)
    rt = json.loads(s)
    v3 = abs(integer_determinant(rt))
    ok = (v1 == v2 == v3 == expected_abs_det)
    return {"verify_1": v1, "verify_2": v2, "verify_3_via_json": v3, "ok": ok}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-file", type=Path, default=SEED_FILE)
    ap.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    ap.add_argument("--quick", action="store_true",
                    help="Reduced budget for smoke tests.")
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    print("=" * 78)
    print("Refine from Orrick (H23 record matrix)")
    print("=" * 78)

    with args.seed_file.open() as f:
        seed_data = json.load(f)
    entries = [[int(x) for x in row] for row in seed_data["entries"]]
    n = len(entries)
    if n != N:
        raise SystemExit(f"Expected n=23, got {n}")

    seed_abs_det = abs(integer_determinant(entries))
    file_claimed = int(seed_data.get("abs_det", -1))
    print(f"  Seed file:         {args.seed_file}")
    print(f"  File claimed |det|: {file_claimed:,d}")
    print(f"  Bareiss |det|:      {seed_abs_det:,d}")
    print(f"  Match:              {seed_abs_det == file_claimed}")
    print(f"  Orrick published:   {ORRICK_PUBLISHED:,d}")
    print(f"  Barba(23):          {upper_bound(23):,d}")
    print(f"  Seed/Orrick ratio:  {seed_abs_det/ORRICK_PUBLISHED:.6f}")
    if seed_abs_det != ORRICK_PUBLISHED:
        print()
        print("  *** WARNING: seed file's matrix does NOT verify to Orrick's "
              "published 2,779,447,296,000,000.")
        print("      The transcribed entries verify to a DIFFERENT (smaller) "
              "value. Proceeding with the file's matrix as the seed; ")
        print("      'beating Orrick' in this run means beating "
              f"{seed_abs_det:,d}, which is BELOW BTC and Orrick.")

    # Load BTC for comparison.
    with BTC_FILE.open() as f:
        btc_data = json.load(f)
    btc_entries = [[int(x) for x in row] for row in btc_data["entries"]]
    btc_abs = abs(integer_determinant(btc_entries))
    print(f"  BTC reference:      {btc_abs:,d}")
    print()

    # --- Diagnostics on seed (Orrick file) and BTC -------------------------
    print("-" * 78)
    print("Diagnostics: Orrick seed vs BTC")
    print("-" * 78)
    fp_seed = fingerprint(entries, "orrick_file_seed")
    fp_btc = fingerprint(btc_entries, "btc")
    fp_seed["hamming_to_btc"] = hamming_distance(entries, btc_entries)
    print(f"  Seed T-card={fp_seed['t_support_cardinality']}  "
          f"distinct SVs={fp_seed['n_distinct_singular_values']}  "
          f"block sizes={fp_seed['block_partition_sizes']}")
    print(f"  Seed Gram multiset: {fp_seed['gram_offdiag_multiset']}")
    print(f"  BTC  T-card={fp_btc['t_support_cardinality']}  "
          f"distinct SVs={fp_btc['n_distinct_singular_values']}  "
          f"block sizes={fp_btc['block_partition_sizes']}")
    print(f"  BTC  Gram multiset: {fp_btc['gram_offdiag_multiset']}")
    print(f"  Hamming(seed, BTC) = {fp_seed['hamming_to_btc']}")
    print()

    # --- Build job list -----------------------------------------------------
    sa_iters = 200_000 if args.quick else 1_500_000
    m1_iters = 80_000 if args.quick else 400_000
    ct_iters = 80_000 if args.quick else 500_000
    tg_iters = 80_000 if args.quick else 500_000
    seeds = [11, 23, 47, 101, 211] if not args.quick else [11, 23]

    jobs: list[tuple[str, callable, tuple]] = []

    # (a) sa_incremental long, k_flips=0
    for sd in seeds:
        jobs.append((
            "a:sa_incremental",
            worker_sa_incremental,
            (f"a:sa_incr/seed={sd}", entries, 0, sd, sa_iters, 0.5, 0.005, None),
        ))
    # (b) sa_m1_anchored, λ ∈ {0.1, 1.0, 10.0}
    for lam in [0.1, 1.0, 10.0]:
        for sd in seeds[:3]:
            jobs.append((
                f"b:sa_m1_λ={lam}",
                worker_sa_m1,
                (f"b:sa_m1_λ={lam}/seed={sd}", entries, lam, sd, m1_iters, 1.0, 0.05),
            ))
    # (c1) class-tabu LOCKED on seed signature
    for sd in seeds[:3]:
        jobs.append((
            "c1:class_tabu_locked",
            worker_sa_class_tabu,
            (f"c1:ct_locked/seed={sd}", entries, True, sd, ct_iters, 0.5, 0.005),
        ))
    # (c2) class-tabu UNLOCKED (drift bridge)
    for sd in seeds[:3]:
        jobs.append((
            "c2:class_tabu_unlocked",
            worker_sa_class_tabu,
            (f"c2:ct_unlocked/seed={sd}", entries, False, sd, ct_iters, 0.5, 0.005),
        ))
    # (d) sa_targeted +3-pair
    for sd in seeds[:3]:
        jobs.append((
            "d:sa_targeted",
            worker_sa_targeted,
            (f"d:sa_targeted/seed={sd}", entries, sd, tg_iters, 0.3, 0.005, 0.7),
        ))
    # (e) Heavy multistart with random perturbation
    for k in [0, 1, 3, 5, 7, 10]:
        for sd in seeds[:3]:
            jobs.append((
                f"e:multistart_k={k}",
                worker_sa_incremental,
                (f"e:k={k}/seed={sd}", entries, k, sd, sa_iters // 3, 0.5, 0.005, None),
            ))

    print(f"Total jobs: {len(jobs)}  workers={args.max_workers}")
    print(f"Per-variant iters: sa_incr={sa_iters:,}  sa_m1={m1_iters:,}  "
          f"ct={ct_iters:,}  targeted={tg_iters:,}")
    print()
    print("-" * 78)
    print("Running...")
    print("-" * 78)

    results: list[dict] = []
    new_record_candidates: list[dict] = []
    t_run_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(fn, payload): variant
            for (variant, fn, payload) in jobs
        }
        for f in as_completed(futures):
            variant = futures[f]
            try:
                r = f.result()
            except Exception as e:
                print(f"  [ERROR] {variant}: {e}")
                continue
            r["variant"] = variant
            r["ratio_to_seed"] = r["abs_det"] / seed_abs_det if seed_abs_det else 0
            r["beats_seed"] = r["abs_det"] > seed_abs_det
            r["beats_orrick_published"] = r["abs_det"] > ORRICK_PUBLISHED
            r["beats_btc"] = r["abs_det"] > btc_abs
            print(f"  [{r['walltime_seconds']:6.1f}s] {r['label']:<32} "
                  f"|det|={r['abs_det']:>22,d}  ratio_seed={r['ratio_to_seed']:.4f}"
                  + ("  BEATS_SEED" if r["beats_seed"] else "")
                  + ("  BEATS_ORRICK_PUB" if r["beats_orrick_published"] else "")
                  + ("  BEATS_BTC" if r["beats_btc"] else ""))
            results.append(r)
            if r["beats_orrick_published"]:
                # Triple-verify immediately
                tv = triple_verify(r["entries"], r["abs_det"])
                r["triple_verify"] = tv
                if tv["ok"]:
                    new_record_candidates.append(r)
                    print("    *** TRIPLE-VERIFIED ABOVE ORRICK PUBLISHED ***")

    elapsed = time.monotonic() - t_run_start
    print()
    print(f"Completed {len(results)} jobs in {elapsed:.1f}s wall.")

    # --- Summary table by variant ------------------------------------------
    print()
    print("=" * 78)
    print("Summary by variant")
    print("=" * 78)
    by_variant: dict[str, list[dict]] = {}
    for r in results:
        by_variant.setdefault(r["variant"], []).append(r)

    summary_rows = []
    for variant in sorted(by_variant.keys()):
        rs = by_variant[variant]
        rs.sort(key=lambda x: -x["abs_det"])
        best = rs[0]
        best_M = best["entries"]
        bf = fingerprint(best_M, f"best:{variant}")
        n_beat = sum(1 for r in rs if r["beats_seed"])
        row = {
            "variant": variant,
            "n_runs": len(rs),
            "n_beats_seed": n_beat,
            "best_abs_det": best["abs_det"],
            "best_ratio_to_seed": best["abs_det"] / seed_abs_det,
            "best_t_card": bf["t_support_cardinality"],
            "best_n_distinct_sv": bf["n_distinct_singular_values"],
            "best_class_signature": bf["class_signature"],
        }
        summary_rows.append(row)
        print(f"  {variant:<32} runs={len(rs)} beat_seed={n_beat}/{len(rs)}  "
              f"best={best['abs_det']:,d}  T={bf['t_support_cardinality']:>3}  "
              f"distinct_SV={bf['n_distinct_singular_values']:>3}")

    # Stayed-at-seed analysis: variant (a) at λ=0
    a_runs = by_variant.get("a:sa_incremental", [])
    if a_runs:
        max_a = max(r["abs_det"] for r in a_runs)
        stayed = max_a == seed_abs_det
        print()
        print(f"  Variant (a) λ=0 SA: best |det|={max_a:,d}  "
              f"stayed_at_seed={stayed}")
    # Did anything beat the seed?
    beat = [r for r in results if r["beats_seed"]]
    print()
    if beat:
        beat.sort(key=lambda x: -x["abs_det"])
        b = beat[0]
        print(f"  *** SOMETHING BEAT THE SEED: best={b['abs_det']:,d} "
              f"({b['variant']}) — ratio_to_seed={b['ratio_to_seed']:.4f}")
        if b["abs_det"] > btc_abs:
            print(f"  *** AND IT BEATS BTC ({btc_abs:,d})")
        if b["abs_det"] > ORRICK_PUBLISHED:
            print(f"  *** AND IT BEATS ORRICK PUBLISHED "
                  f"({ORRICK_PUBLISHED:,d}) — POTENTIAL WORLD RECORD")
    else:
        print("  No variant beat the seed — seed is locally maximal "
              "under all tested move sets / energies.")

    # --- Save above-orrick if any ------------------------------------------
    if new_record_candidates:
        new_record_candidates.sort(key=lambda x: -x["abs_det"])
        best_new = new_record_candidates[0]
        # Reload-from-disk verify
        ABOVE_OUT.write_text(json.dumps({
            "order_id": "H23",
            "n": 23,
            "abs_det": best_new["abs_det"],
            "method": f"refine_from_orrick::{best_new['variant']}",
            "seed_file": str(SEED_FILE),
            "orrick_published_target": ORRICK_PUBLISHED,
            "btc_reference": btc_abs,
            "entries": best_new["entries"],
            "triple_verify": best_new["triple_verify"],
            "fingerprint": fingerprint(best_new["entries"], "above_orrick"),
            "all_above_runs": [
                {k: v for k, v in r.items() if k != "entries"}
                for r in new_record_candidates
            ],
        }, indent=2) + "\n")
        # Now reload-from-disk and verify ONE MORE TIME
        with ABOVE_OUT.open() as f:
            saved = json.load(f)
        v_disk = abs(integer_determinant(saved["entries"]))
        print()
        print("*" * 78)
        print(f"  Wrote {ABOVE_OUT}")
        print(f"  On-disk re-verify: {v_disk:,d} == {best_new['abs_det']:,d} ?  "
              f"{v_disk == best_new['abs_det']}")
        print("*" * 78)

    # --- Save summary ------------------------------------------------------
    summary = {
        "seed_file": str(SEED_FILE),
        "seed_abs_det_bareiss": seed_abs_det,
        "seed_abs_det_file_claimed": file_claimed,
        "seed_matches_orrick_published": seed_abs_det == ORRICK_PUBLISHED,
        "orrick_published": ORRICK_PUBLISHED,
        "btc_reference": btc_abs,
        "barba_23": upper_bound(23),
        "n_jobs": len(results),
        "total_walltime_seconds": elapsed,
        "any_beat_seed": bool(beat),
        "any_beat_orrick_published": bool(new_record_candidates),
        "fingerprint_seed": fp_seed,
        "fingerprint_btc": fp_btc,
        "summary_by_variant": summary_rows,
        "all_results_no_entries": [
            {k: v for k, v in r.items() if k != "entries"}
            for r in results
        ],
    }
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print()
    print(f"  Wrote summary: {args.summary_out}")


if __name__ == "__main__":
    raise SystemExit(main())
