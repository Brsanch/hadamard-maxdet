"""Spectral-anchor search experiments.

Three experiments:
  1. Sanity at n=17 from random init (does anchored SA hit proven faster
     or more reliably than vanilla?).
  2. H19 from n=17-winner extension (target proven 894,426,939,392 from
     the n=17-saturating seed).
  3. H23 from H21 ladder seed (best high-mult H21 basin → extend by 2
     rows + 2 cols, target Orrick 2.778e15 / BTC 2.626e15).

Reports headlines + best matrices saved to ``results/H{19,23}_anchor_best.json``.

Usage::

    python scripts/spectral_anchor_hunt.py [--budget-h17 60]
                                            [--budget-h19 600]
                                            [--budget-h23 600]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo src/ on path when run from repo root.
HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.spectral_anchor import (
    bordered_extension_sa,
    distinct_sv_count,
    dominant_sv_multiplicity,
    sa_spectral_anchored,
)
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import find_winning_matrix
from hadamard_maxdet.incremental import sa_incremental
from hadamard_maxdet.orders import BENCHMARKS


PROVEN_H17 = 21_474_836_480
PROVEN_H19 = 894_426_939_392
BTC_H23 = 2_626_567_700_217_856
ORRICK_H23 = 2_778_127_000_000_000  # Orrick's reported best (~2.778e15)


def fmt(x: int) -> str:
    return f"{x:_}"


def fmt_e(x: int) -> str:
    return f"{x:.4e}"


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------


def experiment_1_n17_sanity(seconds: float = 60.0) -> dict:
    """Anchored vs vanilla SA at n=17 from random inits.

    Multi-start: run anchored SA from K random initial matrices for total
    ``seconds`` budget; do the same for vanilla SA at the same budget.
    Compare success rate (hit proven 21,474,836,480) and time-to-first-hit.
    """
    print("\n=== Experiment 1: n=17 sanity from random init ===\n")
    n = 17
    n_starts = 8
    seconds_per_start = seconds / n_starts
    iters_per_start = 200_000

    # Anchored.
    print(f"Anchored SA: {n_starts} starts, {iters_per_start} iters each "
          f"(~{seconds_per_start:.1f}s budget per start, λ=0.1, prop_T=2)")
    import random as _rand
    anchored_results: list[dict] = []
    t0 = time.monotonic()
    for s in range(n_starts):
        rng = _rand.Random(1000 + s)
        M0 = Matrix.random(n, rng)
        sub_t = time.monotonic()
        r = sa_spectral_anchored(
            M0,
            iterations=iters_per_start,
            lambda_anchor=0.1,
            tol=0.05,
            t_start=1.0,
            t_end=0.005,
            seed=2000 + s,
            refresh_eigenspace_every=200,
            rescore_anchor_every=20,
            proposal_temperature=2.0,
        )
        elapsed = time.monotonic() - sub_t
        hit = (r.abs_det == PROVEN_H17)
        anchored_results.append({
            "start": s, "abs_det": int(r.abs_det),
            "ratio": r.abs_det / PROVEN_H17, "hit_proven": hit,
            "elapsed_seconds": elapsed,
            "accepted": r.accepted,
        })
        print(f"  start {s}: |det|={fmt(r.abs_det)} ratio={r.abs_det/PROVEN_H17:.4f} "
              f"hit={hit} ({elapsed:.1f}s)")
    anchored_total = time.monotonic() - t0

    # Vanilla.
    print(f"\nVanilla SA: {n_starts} starts, {iters_per_start} iters each "
          f"(~{seconds_per_start:.1f}s budget per start)")
    vanilla_results: list[dict] = []
    t0 = time.monotonic()
    for s in range(n_starts):
        sub_t = time.monotonic()
        r = sa_incremental(
            n,
            iterations=iters_per_start,
            t_start=1.0,
            t_end=0.005,
            seed=2000 + s,
            refresh_every=0,
        )
        elapsed = time.monotonic() - sub_t
        hit = (r.abs_det == PROVEN_H17)
        vanilla_results.append({
            "start": s, "abs_det": int(r.abs_det),
            "ratio": r.abs_det / PROVEN_H17, "hit_proven": hit,
            "elapsed_seconds": elapsed,
            "accepted": r.accepted,
        })
        print(f"  start {s}: |det|={fmt(r.abs_det)} ratio={r.abs_det/PROVEN_H17:.4f} "
              f"hit={hit} ({elapsed:.1f}s)")
    vanilla_total = time.monotonic() - t0

    a_hits = sum(1 for x in anchored_results if x["hit_proven"])
    v_hits = sum(1 for x in vanilla_results if x["hit_proven"])
    a_best = max(x["abs_det"] for x in anchored_results)
    v_best = max(x["abs_det"] for x in vanilla_results)
    a_avg_acc = sum(x["accepted"] for x in anchored_results) / n_starts
    v_avg_acc = sum(x["accepted"] for x in vanilla_results) / n_starts

    print(f"\n--- Summary ---")
    print(f"Anchored: {a_hits}/{n_starts} hit proven, best={fmt(a_best)} "
          f"(ratio {a_best/PROVEN_H17:.4f}), {anchored_total:.1f}s total")
    print(f"Vanilla:  {v_hits}/{n_starts} hit proven, best={fmt(v_best)} "
          f"(ratio {v_best/PROVEN_H17:.4f}), {vanilla_total:.1f}s total")
    print(f"Acceptance: anchored avg {a_avg_acc:.0f}, vanilla avg {v_avg_acc:.0f}")

    return {
        "n": n,
        "iters_per_start": iters_per_start,
        "n_starts": n_starts,
        "anchored": {
            "results": anchored_results,
            "total_seconds": anchored_total,
            "hits": a_hits,
            "best_abs_det": a_best,
        },
        "vanilla": {
            "results": vanilla_results,
            "total_seconds": vanilla_total,
            "hits": v_hits,
            "best_abs_det": v_best,
        },
    }


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------


def get_n17_winner() -> list[list[int]]:
    """Get the proven n=17 winner via the structured SA path."""
    abs_det, entries, _ = find_winning_matrix(
        17, structured_seconds=20.0, n_starts=2, iterations=20_000,
    )
    if abs_det != PROVEN_H17:
        raise RuntimeError(
            f"Failed to obtain proven n=17 winner; got {abs_det}"
        )
    return entries


def experiment_2_h19_from_n17(wall_seconds: float = 600.0) -> dict:
    """H19 from n=17 winner extension.

    Take the proven n=17 winner; extend by 2 rows + 2 cols (72 free
    entries). Run spectral-anchored SA on free entries; report best
    |det|, ratio to proven 894,426,939,392, and final SV multiplicity.
    """
    print(f"\n=== Experiment 2: H19 from n=17 winner extension "
          f"({wall_seconds:.0f}s budget) ===\n")
    seed_n17 = get_n17_winner()
    print(f"Seed: n=17 winner |det| = {fmt(PROVEN_H17)} "
          f"(SV mult = {dominant_sv_multiplicity(seed_n17, tol=0.1)}/17)")

    # Run a few restarts of bordered_extension_sa, splitting the budget.
    n_restarts = 8
    iters_per_restart = 1_500_000

    print(f"Running {n_restarts} restarts × {iters_per_restart} iters, "
          f"λ=0.2, proposal_T=2 ...")
    t0 = time.monotonic()
    sub_budget = wall_seconds / n_restarts
    best_d = 0
    best_M = None
    best_history: list[dict] = []
    per_restart_results: list[dict] = []
    for r in range(n_restarts):
        if time.monotonic() - t0 >= wall_seconds:
            break
        sub_t = time.monotonic()
        result = bordered_extension_sa(
            seed_matrix=seed_n17,
            target_n=19,
            iterations=iters_per_restart,
            lambda_anchor=0.2,
            tol=0.05,
            t_start=1.0,
            t_end=0.005,
            seed=4242 + r * 7919,
            refresh_eigenspace_every=300,
            rescore_anchor_every=20,
            proposal_temperature=2.0,
            wall_seconds=sub_budget,
            n_restarts=1,
        )
        elapsed = time.monotonic() - sub_t
        d = result.abs_det
        mult = dominant_sv_multiplicity(result.matrix.entries, tol=0.1)
        distinct = distinct_sv_count(result.matrix.entries, tol=0.1)
        per_restart_results.append({
            "restart": r, "abs_det": int(d),
            "ratio_to_proven": d / PROVEN_H19,
            "sv_dominant_mult": mult,
            "sv_distinct": distinct,
            "elapsed_seconds": elapsed,
        })
        print(f"  restart {r}: |det|={fmt(d)} ratio={d/PROVEN_H19:.4f} "
              f"mult={mult}/19 distinct={distinct} ({elapsed:.1f}s)")
        if d > best_d:
            best_d = d
            best_M = [list(row) for row in result.matrix.entries]

    total = time.monotonic() - t0
    if best_M is None:
        print("WARNING: no successful run.")
        return {"error": "no successful runs", "wall_seconds": total}

    # Verify Bareiss.
    bareiss = abs(integer_determinant(best_M))
    assert bareiss == best_d, f"Bareiss mismatch: {bareiss} vs {best_d}"

    final_mult = dominant_sv_multiplicity(best_M, tol=0.1)
    final_distinct = distinct_sv_count(best_M, tol=0.1)
    sv_full = np.linalg.svd(np.array(best_M, dtype=float), compute_uv=False)

    print(f"\n--- Summary ---")
    print(f"Best |det| = {fmt(best_d)} ({fmt_e(best_d)})")
    print(f"Ratio to proven {fmt(PROVEN_H19)}: {best_d/PROVEN_H19:.4f}")
    print(f"Prior bench ceiling 0.9412 → did we beat? "
          f"{(best_d/PROVEN_H19) > 0.9412}")
    print(f"Final SV mult = {final_mult}/19, distinct = {final_distinct}")
    print(f"SVs: {[round(float(s),3) for s in sv_full]}")
    print(f"Wall: {total:.1f}s")

    return {
        "n": 19,
        "proven": PROVEN_H19,
        "best_abs_det": int(best_d),
        "ratio_to_proven": best_d / PROVEN_H19,
        "beats_prior_ceiling_0.9412": (best_d / PROVEN_H19) > 0.9412,
        "sv_dominant_mult": final_mult,
        "sv_distinct": final_distinct,
        "sv_values": [float(s) for s in sv_full],
        "best_matrix": best_M,
        "per_restart": per_restart_results,
        "wall_seconds": total,
    }


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------


def get_h21_ladder_seed() -> tuple[list[list[int]], int, int]:
    """Find best H21 basin matrix to use as ladder seed.

    Returns (matrix, abs_det, sv_mult).
    """
    import glob
    best_score = -1.0
    best: tuple = None  # type: ignore
    # Score: abs_det weighted by SV-multiplicity bonus.
    for f in sorted(glob.glob("results/H21_basin_*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:
            continue
        M = data.get("matrix_entries")
        ad = data.get("abs_det", 0)
        if not M or ad <= 0:
            continue
        mult = dominant_sv_multiplicity(M, tol=0.1)
        # Use abs_det × (1 + 0.05 * mult) as a soft score.
        score = ad * (1.0 + 0.05 * mult)
        if score > best_score:
            best_score = score
            best = (M, ad, mult, f)
    if best is None:
        raise RuntimeError("No H21 basin matrices found")
    M, ad, mult, fname = best
    print(f"Selected H21 ladder seed: {fname}, |det|={fmt(ad)} "
          f"(ratio {ad/59_392_000_000_000:.4f}, mult={mult}/21)")
    return M, ad, mult


def experiment_3_h23_from_h21(wall_seconds: float = 600.0) -> dict:
    """H23 from H21 ladder seed.

    Take best high-mult H21 basin matrix → extend by 2 rows + 2 cols
    (88 free entries) → run anchored SA. Report best |det|, ratio to
    BTC's 2.626e15, and ratio to Orrick's 2.778e15.
    """
    print(f"\n=== Experiment 3: H23 from H21 ladder seed "
          f"({wall_seconds:.0f}s budget) ===\n")
    seed_h21, h21_d, h21_mult = get_h21_ladder_seed()

    n_restarts = 8
    iters_per_restart = 1_500_000

    print(f"Running {n_restarts} restarts × {iters_per_restart} iters, "
          f"λ=0.2, proposal_T=2 ...")
    t0 = time.monotonic()
    sub_budget = wall_seconds / n_restarts
    best_d = 0
    best_M = None
    per_restart_results: list[dict] = []
    for r in range(n_restarts):
        if time.monotonic() - t0 >= wall_seconds:
            break
        sub_t = time.monotonic()
        result = bordered_extension_sa(
            seed_matrix=seed_h21,
            target_n=23,
            iterations=iters_per_restart,
            lambda_anchor=0.2,
            tol=0.05,
            t_start=1.0,
            t_end=0.005,
            seed=4242 + r * 7919,
            refresh_eigenspace_every=400,
            rescore_anchor_every=30,
            proposal_temperature=2.0,
            wall_seconds=sub_budget,
            n_restarts=1,
        )
        elapsed = time.monotonic() - sub_t
        d = result.abs_det
        mult = dominant_sv_multiplicity(result.matrix.entries, tol=0.1)
        distinct = distinct_sv_count(result.matrix.entries, tol=0.1)
        per_restart_results.append({
            "restart": r, "abs_det": int(d),
            "ratio_to_btc": d / BTC_H23,
            "ratio_to_orrick": d / ORRICK_H23,
            "sv_dominant_mult": mult,
            "sv_distinct": distinct,
            "elapsed_seconds": elapsed,
        })
        print(f"  restart {r}: |det|={fmt(d)} ratio_to_btc={d/BTC_H23:.4f} "
              f"ratio_to_orrick={d/ORRICK_H23:.4f} mult={mult}/23 "
              f"distinct={distinct} ({elapsed:.1f}s)")
        if d > best_d:
            best_d = d
            best_M = [list(row) for row in result.matrix.entries]

    total = time.monotonic() - t0
    if best_M is None:
        print("WARNING: no successful run.")
        return {"error": "no successful runs", "wall_seconds": total}

    # Bareiss-verify.
    bareiss = abs(integer_determinant(best_M))
    assert bareiss == best_d, f"Bareiss mismatch: {bareiss} vs {best_d}"

    final_mult = dominant_sv_multiplicity(best_M, tol=0.1)
    final_distinct = distinct_sv_count(best_M, tol=0.1)
    sv_full = np.linalg.svd(np.array(best_M, dtype=float), compute_uv=False)

    print(f"\n--- Summary ---")
    print(f"Best |det| = {fmt(best_d)} ({fmt_e(best_d)})")
    print(f"Ratio to BTC ({fmt_e(BTC_H23)}): {best_d/BTC_H23:.4f}")
    print(f"Ratio to Orrick ({fmt_e(ORRICK_H23)}): {best_d/ORRICK_H23:.4f}")
    print(f"Beats BTC? {best_d > BTC_H23}")
    print(f"Beats Orrick? {best_d > ORRICK_H23}")
    print(f"Final SV mult = {final_mult}/23, distinct = {final_distinct}")
    print(f"SVs: {[round(float(s),3) for s in sv_full]}")
    print(f"Wall: {total:.1f}s")

    return {
        "n": 23,
        "btc": BTC_H23,
        "orrick": ORRICK_H23,
        "best_abs_det": int(best_d),
        "ratio_to_btc": best_d / BTC_H23,
        "ratio_to_orrick": best_d / ORRICK_H23,
        "beats_btc": best_d > BTC_H23,
        "beats_orrick": best_d > ORRICK_H23,
        "sv_dominant_mult": final_mult,
        "sv_distinct": final_distinct,
        "sv_values": [float(s) for s in sv_full],
        "best_matrix": best_M,
        "h21_seed_abs_det": h21_d,
        "h21_seed_mult": h21_mult,
        "per_restart": per_restart_results,
        "wall_seconds": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-h17", type=float, default=60.0,
                        help="Experiment 1 wall seconds (default 60)")
    parser.add_argument("--budget-h19", type=float, default=600.0,
                        help="Experiment 2 wall seconds (default 600)")
    parser.add_argument("--budget-h23", type=float, default=600.0,
                        help="Experiment 3 wall seconds (default 600)")
    parser.add_argument("--out-h19", type=str,
                        default="results/H19_anchor_best.json")
    parser.add_argument("--out-h23", type=str,
                        default="results/H23_anchor_best.json")
    parser.add_argument("--out-summary", type=str,
                        default="results/spectral_anchor_hunt.json")
    parser.add_argument("--skip-1", action="store_true")
    parser.add_argument("--skip-2", action="store_true")
    parser.add_argument("--skip-3", action="store_true")
    args = parser.parse_args()

    summary: dict = {}

    if not args.skip_1:
        summary["experiment_1"] = experiment_1_n17_sanity(args.budget_h17)
    if not args.skip_2:
        e2 = experiment_2_h19_from_n17(args.budget_h19)
        summary["experiment_2"] = e2
        if "best_matrix" in e2:
            Path(args.out_h19).write_text(json.dumps({
                "n": 19,
                "abs_det": e2["best_abs_det"],
                "ratio_to_proven": e2["ratio_to_proven"],
                "sv_dominant_mult": e2["sv_dominant_mult"],
                "sv_distinct": e2["sv_distinct"],
                "matrix": e2["best_matrix"],
            }, indent=2) + "\n")
            print(f"\nSaved best H19 to {args.out_h19}")
    if not args.skip_3:
        e3 = experiment_3_h23_from_h21(args.budget_h23)
        summary["experiment_3"] = e3
        if "best_matrix" in e3:
            Path(args.out_h23).write_text(json.dumps({
                "n": 23,
                "abs_det": e3["best_abs_det"],
                "ratio_to_btc": e3["ratio_to_btc"],
                "ratio_to_orrick": e3["ratio_to_orrick"],
                "sv_dominant_mult": e3["sv_dominant_mult"],
                "sv_distinct": e3["sv_distinct"],
                "matrix": e3["best_matrix"],
            }, indent=2) + "\n")
            print(f"\nSaved best H23 to {args.out_h23}")

    # Strip matrices from summary to keep it compact.
    summary_compact = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            v2 = {kk: vv for kk, vv in v.items() if kk != "best_matrix"}
            summary_compact[k] = v2
        else:
            summary_compact[k] = v
    Path(args.out_summary).write_text(
        json.dumps(summary_compact, indent=2, default=str) + "\n",
    )
    print(f"\nSaved summary to {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
