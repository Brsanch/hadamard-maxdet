"""Build Orrick's H47 conference-method matrix and diagnose its structure.

Also build the H23 symmetric-conference variant and compare to
Orrick's H23 record (a known stronger matrix at n=23) on the same
diagnostics. Outputs JSON to ``results/conference_method_diagnostics.json``.

Run from repo root:
    PYTHONPATH=src python3 scripts/conference_method_h47.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hadamard_maxdet.conference_method import (
    build_orrick_h47,
    orrick_h47_target,
    conference_method_orrick,
    build_orrick_method_h23,
)
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.structural_test import (
    triple_cyclic_product_stats,
    column_gram_stats,
    singular_vector_ipr,
    transpose_symmetry_score,
    col_gram_t_cardinality,
    gram_clustering_score,
)


def diagnose(M, label: str) -> dict:
    M = np.asarray(M, dtype=int)
    n = int(M.shape[0])
    d = abs(integer_determinant(M.tolist()))
    G_row = M @ M.T
    iu = np.triu_indices(n, k=1)
    off = G_row[iu]
    vals, counts = np.unique(off, return_counts=True)
    gram_support = {int(v): int(c) for v, c in zip(vals, counts)}

    m1 = triple_cyclic_product_stats(M)
    m2 = column_gram_stats(M)
    m3 = singular_vector_ipr(M)
    ts = transpose_symmetry_score(M)
    cgt = col_gram_t_cardinality(M)
    h1 = gram_clustering_score(M, k_range=(2, 3, 4, 5))

    sv = m3["singular_values"]
    # Multiplicity of singular values (rounded to 4 decimals)
    sv_rounded = [round(float(s), 4) for s in sv]
    sv_unique, sv_counts = np.unique(sv_rounded, return_counts=True)
    sv_multiset = {float(v): int(c) for v, c in zip(sv_unique, sv_counts)}

    return {
        "label": label,
        "n": n,
        "abs_det": int(d),
        "gram_offdiag_support": gram_support,
        "gram_offdiag_cardinality": len(gram_support),
        "row_T_cardinality": int(m1["cardinality"]),
        "row_T_max_abs": int(m1["max_abs"]),
        "row_T_frac_zero": float(m1["frac_zero"]),
        "col_T_cardinality": int(cgt["cardinality"]),
        "col_minus_row_T_card": int(cgt["row_vs_col"]["card_delta_col_minus_row"]),
        "transpose_symmetry_score": float(ts["score"]),
        "transpose_symmetry_l1": int(ts["l1_distance"]),
        "row_col_same_multiset": bool(ts["same_multiset"]),
        "sv_multiplicity_classes": len(sv_multiset),
        "sv_unique_values_rounded": sorted(sv_multiset.keys()),
        "sv_multiplicities": sv_multiset,
        "sv_leading": float(sv[0]),
        "sv_trailing": float(sv[-1]),
        "ipr_leading": float(m3["leading_ipr"]),
        "ipr_mean": float(m3["mean_ipr"]),
        "h1_best_score": float(h1["best_score"]),
        "h1_best_k": h1["best_k"],
    }


def main() -> int:
    out: dict = {"runs": []}

    # --- 1. H47 via Orrick §2 (rowsum-2 H_4 variant) ---
    print("Building H47 via Orrick §2 (rowsum-2 H_4 variant)...")
    M_h47, d_h47 = build_orrick_h47()
    target = orrick_h47_target()
    print(f"  |det| = {d_h47}")
    print(f"  target = {target}")
    print(f"  ratio = {d_h47 / target:.6f}")
    print(f"  match exactly: {d_h47 == target}")

    h47_diag = diagnose(M_h47, "H47_orrick_method_rowsum2_h4")
    h47_diag["target_orrick_h47"] = int(target)
    h47_diag["ratio_to_target"] = float(d_h47 / target)
    h47_diag["matches_target_exactly"] = bool(d_h47 == target)
    out["h47_rowsum2"] = h47_diag

    # --- 2. H47 via Sylvester H_4 variant ---
    print("\nBuilding H47 via Sylvester H_4 variant...")
    M_h47_syl = conference_method_orrick(47, use_symmetric=False, h4_variant="sylvester")
    d_h47_syl = abs(integer_determinant(M_h47_syl.tolist()))
    print(f"  |det| = {d_h47_syl}")
    print(f"  ratio = {d_h47_syl / target:.6f}")

    h47_syl_diag = diagnose(M_h47_syl, "H47_orrick_method_sylvester_h4")
    h47_syl_diag["target_orrick_h47"] = int(target)
    h47_syl_diag["ratio_to_target"] = float(d_h47_syl / target)
    out["h47_sylvester"] = h47_syl_diag

    # --- 3. H23 via Orrick §2 SYMMETRIC variant ---
    print("\nBuilding H23 via Orrick §2 SYMMETRIC variant...")
    M_h23, d_h23 = build_orrick_method_h23(use_symmetric=True)
    print(f"  |det| = {d_h23}")
    orrick_h23 = 2_779_447_296_000_000
    print(f"  Orrick H23 record = {orrick_h23}")
    print(f"  ratio to record = {d_h23 / orrick_h23:.6f}")
    if d_h23 > orrick_h23:
        print("  *** EXCEEDS Orrick's record — triple-verifying ***")
        d_check_a = abs(integer_determinant(M_h23.tolist()))
        d_check_b = abs(integer_determinant(M_h23.T.tolist()))
        print(f"  re-verify (Bareiss): {d_check_a}")
        print(f"  re-verify (transpose): {d_check_b}")
        if d_check_a == d_h23 and d_check_b == d_h23:
            print("  TRIPLE-VERIFIED. Saving to H23_above_orrick.json")
            with open(os.path.join("results", "H23_above_orrick.json"), "w") as f:
                json.dump({
                    "abs_det": int(d_h23),
                    "construction": "Orrick conference-method §2 (symmetric variant)",
                    "matrix": M_h23.tolist(),
                    "orrick_h23_record": orrick_h23,
                    "ratio_above_record": float(d_h23 / orrick_h23),
                }, f, indent=2)

    h23_diag = diagnose(M_h23, "H23_orrick_method_symmetric")
    h23_diag["orrick_h23_record"] = int(orrick_h23)
    h23_diag["ratio_to_record"] = float(d_h23 / orrick_h23)
    h23_diag["above_record"] = bool(d_h23 > orrick_h23)
    out["h23_symmetric_method"] = h23_diag

    # --- 4. Compare against Orrick's H23 record matrix (already on disk) ---
    print("\nLoading Orrick H23 record matrix for diagnostic comparison...")
    with open(os.path.join("results", "H23_orrick2003_record.json")) as f:
        orrick_data = json.load(f)
    M_orrick23 = np.array(orrick_data["entries"], dtype=int)
    d_orrick23 = abs(integer_determinant(M_orrick23.tolist()))
    print(f"  Orrick H23 |det| = {d_orrick23} (verified)")
    h23_orrick_diag = diagnose(M_orrick23, "H23_orrick_published_record")
    out["h23_orrick_record"] = h23_orrick_diag

    # --- 5. Comparison summary ---
    print("\n=== Diagnostic comparison ===")
    print(f"{'metric':40s} | {'H47 method':14s} | {'H23 sym-method':14s} | {'H23 Orrick rec':14s}")
    print("-" * 100)
    for key in [
        "abs_det", "gram_offdiag_cardinality",
        "row_T_cardinality", "col_T_cardinality",
        "col_minus_row_T_card", "transpose_symmetry_score",
        "row_col_same_multiset", "sv_multiplicity_classes",
        "ipr_leading", "ipr_mean",
    ]:
        a = h47_diag.get(key, "—")
        b = h23_diag.get(key, "—")
        c = h23_orrick_diag.get(key, "—")
        if isinstance(a, float):
            print(f"{key:40s} | {a:14.4f} | {b:14.4f} | {c:14.4f}")
        else:
            print(f"{key:40s} | {str(a):14s} | {str(b):14s} | {str(c):14s}")

    out_path = os.path.join("results", "conference_method_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
