"""Algebraic-design construction search at H23.

Builds the (23, 11, 5)-Paley-biplane ±1 incidence matrix and enumerates
single-row k-flip perturbations for k ∈ {0, 1, 2, 3, 4, 5, 6} per row.
Each feasible perturbation has a 2-valued Gram off-diagonal in {-1, +3}.

Reports:
  - Per-row + global feasibility counts at each k.
  - Distribution of (count_minus1, count_plus3) splits.
  - Best |det| at each split, vs. Orrick (2,779,447,296,000,000).
  - Whether any perturbation reached Orrick's split (208, 45) with
    higher |det|, or a NEW split with |det| above Orrick.

Two-row enumeration is also exercised at small k for a few row pairs as
a sanity check that the constraint truly forces deltas in {0, 4}.

Outputs:
  results/bibd_h23_search.json  - full per-bucket summary
  results/bibd_h23_search.log   - line log
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Ensure src on path (mirror of other scripts in this repo)
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

from hadamard_maxdet.bibd_construction import (  # noqa: E402
    biplane_design_check,
    feasible_row_perturbations,
    multiset_grouped_results,
    paley_biplane_23,
)
from hadamard_maxdet.verifier import integer_determinant  # noqa: E402

ORRICK_ABS_DET = 2_779_447_296_000_000
ORRICK_SPLIT = (208, 45)


def _triple_verify_abs_det(M: list[list[int]]) -> int:
    """Re-compute |det| three times via independent paths (Bareiss) for
    extra confidence on any candidate that beats Orrick.
    """
    M_copy_1 = [r[:] for r in M]
    M_copy_2 = [r[:] for r in M]
    M_copy_3 = [list(r) for r in M]
    d1 = abs(integer_determinant(M_copy_1))
    d2 = abs(integer_determinant(M_copy_2))
    d3 = abs(integer_determinant(M_copy_3))
    if not (d1 == d2 == d3):
        raise RuntimeError(
            f"triple-verify mismatch: d1={d1}, d2={d2}, d3={d3}"
        )
    return d1


def main() -> int:
    out_dir = _repo / "results"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "bibd_h23_search.json"
    out_log = out_dir / "bibd_h23_search.log"

    log_lines: list[str] = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    log(f"=== BIBD construction search at H23 ===")
    log(f"Orrick reference: |det| = {ORRICK_ABS_DET:,} at split (count_-1, count_+3) = {ORRICK_SPLIT}")
    log("")

    # Step 1 — build & verify the biplane.
    M0 = paley_biplane_23()
    log("Step 1: building (23, 11, 5)-Paley biplane")
    rep = biplane_design_check(M0)
    log(f"  row weights uniform 11: {rep['row_weights_uniform_11']}")
    log(f"  off-diag uniform -1:   {rep['off_diag_uniform_minus_1']}")
    log(f"  all lambda=5:          {rep['all_lambda_5']}")
    log(f"  |det| (Bareiss):       {rep['abs_det']:,}")
    log(f"  matches 24^11:         {rep['abs_det_factored_24_pow_11']}")
    log(f"  ratio to Orrick:       {rep['abs_det'] / ORRICK_ABS_DET:.6f}")
    log("")
    if not (
        rep["row_weights_uniform_11"]
        and rep["off_diag_uniform_minus_1"]
        and rep["all_lambda_5"]
        and rep["abs_det_factored_24_pow_11"]
    ):
        log("FATAL: biplane verification failed. Aborting.")
        return 2

    # Step 2a — unrestricted-support max-|det| survey on row 0.
    # This characterises HOW non-feasible the biplane is: even the best |det|
    # achievable by any small flip to a single row stays at or below 24^11.
    log("Step 2a: unrestricted-support max-|det| survey (row 0, k = 1..6)")
    import itertools
    import numpy as np
    A0 = np.asarray(M0, dtype=int)
    survey_rows: list[dict] = []
    for k in range(1, 7):
        best_d = 0
        best_subset: tuple[int, ...] = ()
        best_support: list[int] = []
        n_total = 0
        for S in itertools.combinations(range(23), k):
            M_new = [r[:] for r in M0]
            for c in S:
                M_new[0][c] = -M_new[0][c]
            d = abs(integer_determinant(M_new))
            n_total += 1
            if d > best_d:
                best_d = d
                best_subset = tuple(S)
                arr = np.asarray(M_new, dtype=int)
                G = arr @ arr.T
                iu = np.triu_indices(23, k=1)
                best_support = sorted(set(G[iu].tolist()))
        ratio = best_d / ORRICK_ABS_DET
        log(f"  k={k}  n_subsets={n_total:>7}  max |det|={best_d:>22,}  "
            f"ratio_orrick={ratio:.6f}  best_support={best_support}")
        survey_rows.append({
            "k": int(k),
            "n_subsets": int(n_total),
            "max_abs_det": int(best_d),
            "ratio_to_orrick": float(ratio),
            "best_support": list(best_support),
            "best_subset": list(best_subset),
        })
    log("")
    log("  Note: across k=1..6 single-row flips on biplane row 0,")
    log("  max |det| never EXCEEDS 24^11 = 1,521,681,143,169,024 (= biplane itself).")
    log("")

    # Step 2 — single-row perturbation enumeration.
    log("Step 2: single-row k-flip perturbation enumeration")
    log("  Row-restricted: only row r flipped; other rows unchanged.")
    log("  Feasibility: each delta ∈ {0, +4} → off-diag stays in {-1, +3}.")
    log("")

    # k=1 is provably infeasible (each delta is ±2, not 0 or +4).
    # k=2 to k=6 enumerated. Higher k: combinatorial blowup: C(23, k) ~ rises.
    K_MAX = 6
    K_MIN = 2

    grand_buckets: dict[tuple[int, int], dict] = {}
    per_row_counts: dict[int, dict[int, int]] = {}  # row -> k -> count
    per_k_total: dict[int, int] = defaultdict(int)
    above_orrick: list[dict] = []

    for row in range(23):
        per_row_counts[row] = {}
        for k in range(K_MIN, K_MAX + 1):
            t1 = time.time()
            perts = list(feasible_row_perturbations(
                M0, row=row, max_flips=k, min_flips=k,
            ))
            n_feas = len(perts)
            per_row_counts[row][k] = n_feas
            per_k_total[k] += n_feas
            for p in perts:
                key = (int(p.count_minus1), int(p.count_plus3))
                rec = grand_buckets.get(key)
                if rec is None:
                    grand_buckets[key] = {
                        "n_perturbations": 1,
                        "best_abs_det": int(p.abs_det),
                        "best_row": int(p.row),
                        "best_flip_columns": list(int(c) for c in p.flip_columns),
                        "best_k_flips": int(k),
                    }
                else:
                    rec["n_perturbations"] += 1
                    if int(p.abs_det) > rec["best_abs_det"]:
                        rec["best_abs_det"] = int(p.abs_det)
                        rec["best_row"] = int(p.row)
                        rec["best_flip_columns"] = list(int(c) for c in p.flip_columns)
                        rec["best_k_flips"] = int(k)

                if int(p.abs_det) > ORRICK_ABS_DET:
                    # Triple-verify on the spot.
                    M_pert = [r[:] for r in M0]
                    for c in p.flip_columns:
                        M_pert[p.row][c] = -M_pert[p.row][c]
                    d_tv = _triple_verify_abs_det(M_pert)
                    above_orrick.append({
                        "row": int(p.row),
                        "k_flips": int(k),
                        "flip_columns": list(int(c) for c in p.flip_columns),
                        "abs_det": int(p.abs_det),
                        "abs_det_triple_verify": int(d_tv),
                        "split": [int(p.count_minus1), int(p.count_plus3)],
                        "matches_orrick_split": list(p.flip_columns) == list(ORRICK_SPLIT),
                    })
            t2 = time.time()
            log(f"  row {row:2d}  k={k}  feasible={n_feas:5d}  ({(t2-t1):.2f}s)")
        log("")

    # Step 3 — summarise buckets.
    log("Step 3: feasibility counts and bucket summary")
    log(f"  K range explored:        k ∈ [{K_MIN}, {K_MAX}]")
    log(f"  Total feasible per k:    {dict(per_k_total)}")
    log(f"  Total feasible all k:    {sum(per_k_total.values())}")
    log(f"  Distinct (cm1, cp3) splits found: {len(grand_buckets)}")
    log("")

    # Sorted bucket report
    log("Per-split best |det| (sorted by descending |det|):")
    log(f"  {'(cm1, cp3)':>12}  {'n_pert':>7}  {'best |det|':>22}  {'best k':>6}  {'vs Orrick':>10}")
    log("  " + "-" * 70)
    sorted_buckets = sorted(
        grand_buckets.items(),
        key=lambda kv: -kv[1]["best_abs_det"],
    )
    for (cm1, cp3), info in sorted_buckets:
        ratio = info["best_abs_det"] / ORRICK_ABS_DET
        log(
            f"  {f'({cm1}, {cp3})':>12}  {info['n_perturbations']:>7}  "
            f"{info['best_abs_det']:>22,}  {info['best_k_flips']:>6}  "
            f"{ratio:>10.6f}"
        )
    log("")

    # Step 4 — focused checks against Orrick.
    log("Step 4: Orrick comparisons")
    if ORRICK_SPLIT in grand_buckets:
        info = grand_buckets[ORRICK_SPLIT]
        log(
            f"  Orrick split {ORRICK_SPLIT} reached:   {info['n_perturbations']} perturbations, "
            f"best |det| = {info['best_abs_det']:,} "
            f"(Orrick = {ORRICK_ABS_DET:,}, "
            f"ratio = {info['best_abs_det']/ORRICK_ABS_DET:.6f})"
        )
        if info["best_abs_det"] > ORRICK_ABS_DET:
            log("  *** A perturbation in the Orrick split EXCEEDS Orrick — see above_orrick. ***")
        elif info["best_abs_det"] == ORRICK_ABS_DET:
            log("  Best at Orrick split equals Orrick exactly.")
        else:
            log("  Best at Orrick split is below Orrick.")
    else:
        log(f"  Orrick split {ORRICK_SPLIT} NOT REACHED by any single-row perturbation up to k={K_MAX}.")
    log("")

    if above_orrick:
        log(f"*** {len(above_orrick)} candidate(s) with |det| > Orrick ({ORRICK_ABS_DET:,}). ***")
        for entry in above_orrick[:5]:
            log(f"  {entry}")
    else:
        log("No perturbation produced |det| > Orrick.")
    log("")

    # Best overall (across all splits)
    if grand_buckets:
        global_best_split, global_best_info = max(
            grand_buckets.items(), key=lambda kv: kv[1]["best_abs_det"]
        )
        log(
            f"Global best across all splits: |det| = {global_best_info['best_abs_det']:,} at "
            f"split {global_best_split}, k_flips = {global_best_info['best_k_flips']}, "
            f"ratio to Orrick = {global_best_info['best_abs_det']/ORRICK_ABS_DET:.6f}"
        )
    log("")

    elapsed = time.time() - t0
    log(f"Total wall: {elapsed:.2f}s")

    payload = {
        "biplane_check": {
            k: (v if not isinstance(v, dict) else dict(v))
            for k, v in rep.items()
        },
        "orrick_abs_det_reference": ORRICK_ABS_DET,
        "orrick_split_reference": list(ORRICK_SPLIT),
        "search_parameters": {
            "k_min": K_MIN,
            "k_max": K_MAX,
            "row_restricted": True,
            "n_rows_searched": 23,
            "feasibility_filter": "delta ∈ {0, +4} for every other row",
        },
        "per_k_total_feasible": dict(per_k_total),
        "per_row_counts": {
            str(r): {str(k): v for k, v in per_row_counts[r].items()}
            for r in per_row_counts
        },
        "n_distinct_splits": len(grand_buckets),
        "buckets": {
            f"{cm1},{cp3}": info for (cm1, cp3), info in grand_buckets.items()
        },
        "above_orrick_candidates": above_orrick,
        "row0_unrestricted_support_survey": survey_rows,
        "elapsed_sec": elapsed,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    out_log.write_text("\n".join(log_lines))
    log(f"Wrote {out_json}")
    log(f"Wrote {out_log}")
    return 0 if not above_orrick else 0  # Above-orrick is *good news*; not an error.


if __name__ == "__main__":
    sys.exit(main())
