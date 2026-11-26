"""Deep test of the column-Gram (Mᵀ M) discriminator.

The prior 2-vs-2 higher-order test noticed that proven winners (n=15, 17)
had row-Gram and column-Gram with identical off-diagonal *supports*
while two near-winners (our H19 best, BTC H23) did not. Sample sizes
2-vs-2 are not enough to claim a discriminator. This script pumps the
sample sizes and tests whether row-vs-column Gram symmetry actually
separates the two classes.

Three diagnostics per matrix:

  - ``transpose_symmetry_score`` ∈ [0, 1] : 1 ⇔ row & column Gram have
    identical off-diagonal multisets; lower ⇔ they diverge.

  - ``col_T_cardinality`` : triple-cyclic-product cardinality on the
    column Gram (the row-side analogue is the M1 cardinality already
    used in the higher-order test).

  - ``card_delta_col_minus_row`` : col_T_cardinality - row_T_cardinality.
    The hypothesis is that proven winners have ``delta == 0`` (row & col
    triangle invariants share a multiset cardinality) while near-winners
    don't.

Sample design:

  Proven class — 5 SA seeds at each of n ∈ {11, 13, 15, 17}: 20 matrices.
  All should saturate the proven max.

  Near class — 5 BTC SA seeds at each of n ∈ {19, 21, 23}: 15 matrices.
  These are reliably stuck in the 0.92-0.94 basin (or BTC fixed-ratio
  for H23).

The script prints class distributions for each diagnostic and reports
classification accuracies of two simple discriminators.

Usage:
    python scripts/column_gram_deep.py [--out PATH]

Wall-time: ~20-40 minutes on M2 Ultra (depends on SA iterations chosen).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadamard_maxdet.incremental import sa_incremental
from hadamard_maxdet.orders import BENCHMARKS
from hadamard_maxdet.rich_ansatz import sa_bordered_two_circulant
from hadamard_maxdet.structural_test import (
    col_gram_t_cardinality,
    compare_proven_vs_near,
    transpose_symmetry_score,
    triple_cyclic_product_stats,
)
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import winner_fingerprint


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------


def _sa_proven(n: int, seed: int, *, iterations: int = 80_000) -> tuple[
    list[list[int]], int
]:
    """Proven-saturating matrix at small n via the multi-strategy
    ``winner_fingerprint`` pipeline (Paley + bordered_two_circulant +
    SA multi-start). Reliably hits proven max for n ∈ {11, 13, 15, 17}.

    The ``seed`` parameter perturbs the SA base seed so different calls
    produce different matrices in the proven-saturating equivalence class
    (when the proven max admits multiple representatives modulo
    row/col permutations & sign flips).
    """
    fp = winner_fingerprint(
        n,
        n_starts=8,
        iterations=iterations,
        structured_seconds=12.0,
        base_seed=seed,
    )
    M = [list(row) for row in fp.matrix]
    d = abs(integer_determinant(M))
    return M, d


def _sa_btc(n: int, seed: int, *, iterations: int = 50_000) -> tuple[
    list[list[int]], int
]:
    """Bordered-two-circulant SA at odd n. Reliably hits the BTC basin."""
    r = sa_bordered_two_circulant(
        n,
        iterations=iterations,
        t_start=1.0,
        t_end=0.005,
        seed=seed,
    )
    M = [list(row) for row in r.matrix.entries]
    d = abs(integer_determinant(M))
    return M, d


def collect_proven_samples(
    ns: list[int], seeds_per_n: int, *, iterations: int = 80_000,
    base_seed: int = 1000,
) -> list[dict]:
    out: list[dict] = []
    for n in ns:
        bid = f"H{n}"
        proven = (
            BENCHMARKS[bid].best_known
            if (bid in BENCHMARKS and BENCHMARKS[bid].best_known is not None)
            else None
        )
        for s in range(seeds_per_n):
            t0 = time.monotonic()
            M, d = _sa_proven(n, base_seed + s * 17 + n * 23,
                              iterations=iterations)
            elapsed = time.monotonic() - t0
            ratio = (d / proven) if proven else None
            saturates = (proven is not None) and d >= proven
            out.append({
                "n": n, "seed": base_seed + s * 17 + n * 23,
                "abs_det": int(d), "proven": int(proven) if proven else None,
                "ratio_to_proven": ratio, "saturates_proven": bool(saturates),
                "matrix": M, "elapsed_seconds": elapsed,
                "source": "sa_incremental",
            })
            print(
                f"  proven n={n} seed={base_seed + s * 17 + n * 23} "
                f"|det|={d} ratio={ratio:.4f} saturates={saturates} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    return out


def collect_near_samples(
    ns: list[int], seeds_per_n: int, *, iterations: int = 50_000,
    base_seed: int = 7000,
) -> list[dict]:
    out: list[dict] = []
    for n in ns:
        bid = f"H{n}"
        proven = (
            BENCHMARKS[bid].best_known
            if (bid in BENCHMARKS and BENCHMARKS[bid].best_known is not None)
            else None
        )
        for s in range(seeds_per_n):
            t0 = time.monotonic()
            M, d = _sa_btc(n, base_seed + s * 19 + n * 29,
                           iterations=iterations)
            elapsed = time.monotonic() - t0
            ratio = (d / proven) if proven else None
            out.append({
                "n": n, "seed": base_seed + s * 19 + n * 29,
                "abs_det": int(d), "proven": int(proven) if proven else None,
                "ratio_to_proven": ratio,
                "matrix": M, "elapsed_seconds": elapsed,
                "source": "sa_bordered_two_circulant",
            })
            print(
                f"  near   n={n} seed={base_seed + s * 19 + n * 29} "
                f"|det|={d} ratio={ratio if ratio is None else f'{ratio:.4f}'} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    return out


# ---------------------------------------------------------------------------
# Per-matrix diagnostics
# ---------------------------------------------------------------------------


def diagnose(samples: list[dict]) -> list[dict]:
    """Compute all column-gram-deep diagnostics for each sample."""
    out: list[dict] = []
    for s in samples:
        M = s["matrix"]
        ts = transpose_symmetry_score(M)
        ct = col_gram_t_cardinality(M)
        rt = triple_cyclic_product_stats(M)
        rec = {
            "n": s["n"], "seed": s["seed"], "abs_det": s["abs_det"],
            "ratio_to_proven": s.get("ratio_to_proven"),
            "saturates_proven": s.get("saturates_proven"),
            "transpose_symmetry_score": ts["score"],
            "transpose_l1_distance": ts["l1_distance"],
            "row_gram_same_support": ts["same_support"],
            "row_gram_same_multiset": ts["same_multiset"],
            "col_T_cardinality": ct["cardinality"],
            "row_T_cardinality": rt["cardinality"],
            "card_delta_col_minus_row": int(
                ct["row_vs_col"]["card_delta_col_minus_row"]
            ),
            "row_gram_offdiag_support": sorted(set(int(v) for v in
                                              ts["row_counts"].keys())),
            "col_gram_offdiag_support": sorted(set(int(v) for v in
                                              ts["col_counts"].keys())),
            "source": s["source"],
        }
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _summ(arr: list[float]) -> dict:
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
        "min": float(a.min()), "max": float(a.max()),
        "median": float(np.median(a)),
    }


def _print_dist(label: str, vals: list[float]) -> None:
    s = _summ(vals)
    print(
        f"  {label:38s} n={s['n']:3d}  mean={s['mean']:.6f}  "
        f"std={s['std']:.6f}  min={s['min']:.6f}  max={s['max']:.6f}  "
        f"med={s['median']:.6f}"
    )


def report(diags_p: list[dict], diags_n: list[dict]) -> dict:
    print("\n" + "=" * 90)
    print("PROVEN-WINNER class distributions")
    print("=" * 90)
    _print_dist("transpose_symmetry_score",
                [d["transpose_symmetry_score"] for d in diags_p])
    _print_dist("transpose_l1_distance",
                [d["transpose_l1_distance"] for d in diags_p])
    _print_dist("row_T_cardinality",
                [d["row_T_cardinality"] for d in diags_p])
    _print_dist("col_T_cardinality",
                [d["col_T_cardinality"] for d in diags_p])
    _print_dist("card_delta_col_minus_row",
                [d["card_delta_col_minus_row"] for d in diags_p])
    n_same_support_p = sum(int(d["row_gram_same_support"]) for d in diags_p)
    n_same_multiset_p = sum(int(d["row_gram_same_multiset"]) for d in diags_p)
    print(f"  row==col SUPPORT (count): {n_same_support_p}/{len(diags_p)}")
    print(f"  row==col MULTISET (count): {n_same_multiset_p}/{len(diags_p)}")

    print("\n" + "=" * 90)
    print("NEAR-WINNER class distributions")
    print("=" * 90)
    _print_dist("transpose_symmetry_score",
                [d["transpose_symmetry_score"] for d in diags_n])
    _print_dist("transpose_l1_distance",
                [d["transpose_l1_distance"] for d in diags_n])
    _print_dist("row_T_cardinality",
                [d["row_T_cardinality"] for d in diags_n])
    _print_dist("col_T_cardinality",
                [d["col_T_cardinality"] for d in diags_n])
    _print_dist("card_delta_col_minus_row",
                [d["card_delta_col_minus_row"] for d in diags_n])
    n_same_support_n = sum(int(d["row_gram_same_support"]) for d in diags_n)
    n_same_multiset_n = sum(int(d["row_gram_same_multiset"]) for d in diags_n)
    print(f"  row==col SUPPORT (count): {n_same_support_n}/{len(diags_n)}")
    print(f"  row==col MULTISET (count): {n_same_multiset_n}/{len(diags_n)}")

    # Classifier accuracies via the helper.
    print("\n" + "=" * 90)
    print("Classifier evaluation")
    print("=" * 90)
    p_mats = [d for d in diags_p]
    n_mats = [d for d in diags_n]

    # Build proven vs near matrices' raw arrays for compare_proven_vs_near.
    # We can't easily re-run on matrices alone (the script used per-record
    # matrices); reconstruct from samples instead at call site.

    # Score-distribution-based classification: "proven if score == 1.0".
    p_perfect = sum(1 for d in diags_p if d["transpose_symmetry_score"] >= 1.0 - 1e-12)
    n_perfect = sum(1 for d in diags_n if d["transpose_symmetry_score"] >= 1.0 - 1e-12)
    print(
        f"  perfect-symmetry (score == 1.0): "
        f"proven {p_perfect}/{len(diags_p)} = {p_perfect/max(1,len(diags_p)):.3f},  "
        f"near   {n_perfect}/{len(diags_n)} = {n_perfect/max(1,len(diags_n)):.3f}"
    )

    # delta == 0 classifier
    p_delta0 = sum(1 for d in diags_p if d["card_delta_col_minus_row"] == 0)
    n_delta0 = sum(1 for d in diags_n if d["card_delta_col_minus_row"] == 0)
    print(
        f"  delta_col_minus_row == 0: "
        f"proven {p_delta0}/{len(diags_p)} = {p_delta0/max(1,len(diags_p)):.3f},  "
        f"near   {n_delta0}/{len(diags_n)} = {n_delta0/max(1,len(diags_n)):.3f}"
    )

    # Same SUPPORT classifier
    p_supp = sum(1 for d in diags_p if d["row_gram_same_support"])
    n_supp = sum(1 for d in diags_n if d["row_gram_same_support"])
    print(
        f"  row==col SUPPORT classifier (proven if same): "
        f"proven {p_supp}/{len(diags_p)} = {p_supp/max(1,len(diags_p)):.3f},  "
        f"near   {n_supp}/{len(diags_n)} = {n_supp/max(1,len(diags_n)):.3f}"
    )

    # Pooled accuracy of the same-support binary discriminator.
    total = len(diags_p) + len(diags_n)
    same_support_correct = p_supp + (len(diags_n) - n_supp)
    delta0_correct = p_delta0 + (len(diags_n) - n_delta0)
    perfect_correct = p_perfect + (len(diags_n) - n_perfect)
    print()
    print(f"  pooled accuracy (predict proven if same SUPPORT): "
          f"{same_support_correct}/{total} = {same_support_correct/total:.3f}")
    print(f"  pooled accuracy (predict proven if delta == 0): "
          f"{delta0_correct}/{total} = {delta0_correct/total:.3f}")
    print(f"  pooled accuracy (predict proven if score == 1.0): "
          f"{perfect_correct}/{total} = {perfect_correct/total:.3f}")

    # Verdict.
    p_score_min = min(d["transpose_symmetry_score"] for d in diags_p)
    n_score_max = max(d["transpose_symmetry_score"] for d in diags_n)
    clean = p_score_min > n_score_max
    print()
    print(f"  proven score min = {p_score_min:.6f}")
    print(f"  near   score max = {n_score_max:.6f}")
    print(f"  CLEAN SEPARATION (min(proven) > max(near))? {clean}")

    return {
        "proven_summary": {
            "n": len(diags_p),
            "transpose_symmetry_score": _summ(
                [d["transpose_symmetry_score"] for d in diags_p]
            ),
            "card_delta_col_minus_row": _summ(
                [d["card_delta_col_minus_row"] for d in diags_p]
            ),
            "n_same_support": int(n_same_support_p),
            "n_same_multiset": int(n_same_multiset_p),
            "n_perfect_score": int(p_perfect),
            "n_delta_zero": int(p_delta0),
            "score_min": float(p_score_min),
        },
        "near_summary": {
            "n": len(diags_n),
            "transpose_symmetry_score": _summ(
                [d["transpose_symmetry_score"] for d in diags_n]
            ),
            "card_delta_col_minus_row": _summ(
                [d["card_delta_col_minus_row"] for d in diags_n]
            ),
            "n_same_support": int(n_same_support_n),
            "n_same_multiset": int(n_same_multiset_n),
            "n_perfect_score": int(n_perfect),
            "n_delta_zero": int(n_delta0),
            "score_max": float(n_score_max),
        },
        "classifier_accuracy": {
            "same_support": float(same_support_correct / total),
            "delta_zero": float(delta0_correct / total),
            "perfect_score": float(perfect_correct / total),
        },
        "clean_separation": bool(clean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proven-ns", type=int, nargs="+", default=[11, 13, 15, 17],
        help="Orders for proven-class samples (SA reliably hits proven max).",
    )
    parser.add_argument(
        "--near-ns", type=int, nargs="+", default=[19, 21, 23],
        help="Orders for near-class samples (BTC SA stuck below proven).",
    )
    parser.add_argument(
        "--seeds-proven", type=int, default=5,
        help="SA seeds per proven order.",
    )
    parser.add_argument(
        "--seeds-near", type=int, default=5,
        help="SA seeds per near order.",
    )
    parser.add_argument(
        "--proven-iters", type=int, default=80_000,
        help="SA iterations per proven sample.",
    )
    parser.add_argument(
        "--near-iters", type=int, default=50_000,
        help="SA iterations per near sample.",
    )
    parser.add_argument(
        "--out", type=str,
        default=str(REPO / "results" / "column_gram_deep.json"),
    )
    args = parser.parse_args()

    print("=" * 90)
    print("Generating proven-class samples")
    print("=" * 90)
    t0 = time.monotonic()
    proven_samples = collect_proven_samples(
        args.proven_ns, args.seeds_proven, iterations=args.proven_iters,
    )
    print(
        f"\nProven generation done: {len(proven_samples)} samples "
        f"({time.monotonic() - t0:.1f}s)",
        flush=True,
    )

    print("\n" + "=" * 90)
    print("Generating near-class samples (BTC SA)")
    print("=" * 90)
    t1 = time.monotonic()
    near_samples = collect_near_samples(
        args.near_ns, args.seeds_near, iterations=args.near_iters,
    )
    print(
        f"\nNear generation done: {len(near_samples)} samples "
        f"({time.monotonic() - t1:.1f}s)",
        flush=True,
    )

    # Filter proven samples to only those that actually saturate proven max.
    proven_clean = [s for s in proven_samples if s["saturates_proven"]]
    print(
        f"\nProven samples that saturated proven max: "
        f"{len(proven_clean)}/{len(proven_samples)}"
    )
    if len(proven_clean) < 15:
        print("  WARNING: fewer than 15 proven-saturating samples; "
              "consider raising --seeds-proven or --proven-iters.")

    diags_p = diagnose(proven_clean)
    diags_n = diagnose(near_samples)

    summary = report(diags_p, diags_n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "n_proven_generated": len(proven_samples),
        "n_proven_saturating": len(proven_clean),
        "n_near_generated": len(near_samples),
        "diags_proven": diags_p,
        "diags_near": diags_n,
        "summary": summary,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
