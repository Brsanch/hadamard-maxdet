"""Test three higher-order structural metrics on the same 5 matrices used
in the prior round (clustering / modularity / symmetry — all NULL).

Metrics:

* M1 — Triple cyclic Gram product distribution
       T(i, j, k) = <v_i, v_j> * <v_j, v_k> * <v_k, v_i>
* M2 — Column Gram (Mᵀ M) vs row Gram (M Mᵀ) off-diagonal multisets
* M3 — Inverse Participation Ratio (IPR) of singular vectors

Loads the same 5 matrices, computes all three diagnostics, prints
comparison tables, evaluates discrimination (proven > near > random or
the reverse), and saves a JSON dump.

Usage:
    python scripts/test_higher_order.py [--budget 60]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadamard_maxdet.structural_test import compare_matrices_v2
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import winner_fingerprint


def _load_proven(n: int, budget_seconds: float) -> tuple[list[list[int]], int]:
    print(f"  re-deriving proven n={n} via winner_fingerprint ...", flush=True)
    t0 = time.monotonic()
    fp = winner_fingerprint(
        n,
        n_starts=8,
        iterations=80_000,
        structured_seconds=min(budget_seconds, 30.0),
        base_seed=12345,
    )
    elapsed = time.monotonic() - t0
    abs_det = abs(integer_determinant(fp.matrix))
    print(
        f"  -> n={n} reached |det|={abs_det:,d} "
        f"(target {fp.proven_max:,d}, ratio {abs_det / fp.proven_max:.4f}) "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    return fp.matrix, abs_det


def _load_h19_best() -> tuple[list[list[int]], int]:
    path = REPO / "results" / "H19_diagnostic_sa_best.json"
    with path.open() as f:
        d = json.load(f)
    return d["matrix"], int(d["abs_det"])


def _load_h23_btc() -> tuple[list[list[int]], int]:
    path = REPO / "results" / "H23_rich_ansatz_best.json"
    with path.open() as f:
        d = json.load(f)
    return d["entries"], int(d["abs_det"])


def _random_h23(seed: int = 42) -> tuple[list[list[int]], int]:
    rng = random.Random(seed)
    n = 23
    M = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
    abs_det = abs(integer_determinant(M))
    return M, abs_det


def _evaluate_discrimination(rows: list[dict]) -> dict:
    """For each metric (single scalar reduction), check if there is a clean
    monotone ordering across the 5 matrices.

    Categories:
      proven_idx = [0, 1]  (proven n=17, proven n=15)
      near_idx   = [2, 3]  (our H19 best, BTC H23)
      rand_idx   = [4]     (random H23)
    """
    proven_idx = [0, 1]
    near_idx = [2, 3]
    rand_idx = [4]

    def _cleanly_ranks(values: list[float], lower_is_better: bool) -> tuple[bool, str]:
        vp_min = min(values[i] for i in proven_idx)
        vp_max = max(values[i] for i in proven_idx)
        vn_max = max(values[i] for i in near_idx)
        vn_min = min(values[i] for i in near_idx)
        vr = values[rand_idx[0]]
        if lower_is_better:
            sep_pn = vp_max < vn_min
            sep_nr = vn_max < vr
        else:
            sep_pn = vp_min > vn_max
            sep_nr = vn_min > vr
        ok = sep_pn and sep_nr
        return ok, f"proven_vs_near={'OK' if sep_pn else 'NO'}, near_vs_random={'OK' if sep_nr else 'NO'}"

    # Pull scalar reductions.
    m1_card = [r["m1"]["cardinality"] for r in rows]
    m1_std = [r["m1"]["std"] for r in rows]
    m1_frac0 = [r["m1"]["frac_zero"] for r in rows]
    m1_entropy = [r["m1"]["entropy"] for r in rows]
    m1_max = [r["m1"]["max_abs"] for r in rows]

    m2_row_card = [r["m2"]["row_gram"]["cardinality"] for r in rows]
    m2_col_card = [r["m2"]["column_gram"]["cardinality"] for r in rows]
    m2_card_delta = [r["m2"]["comparison"]["cardinality_delta_col_minus_row"] for r in rows]
    m2_same_set = [int(r["m2"]["comparison"]["same_support"]) for r in rows]

    m3_leading = [r["m3"]["leading_ipr"] for r in rows]
    m3_mean = [r["m3"]["mean_ipr"] for r in rows]
    m3_max = [r["m3"]["max_ipr"] for r in rows]
    m3_min = [r["m3"]["min_ipr"] for r in rows]
    # Distance of leading IPR to the uniform-delocalised lower bound 1/n
    m3_lead_excess = [
        r["m3"]["leading_ipr"] - r["m3"]["uniform_ipr"] for r in rows
    ]

    candidates: list[tuple[str, list[float], bool]] = [
        ("M1 cardinality (lower=tighter)", m1_card, True),
        ("M1 std (lower=tighter)", m1_std, True),
        ("M1 frac_zero (higher=cleaner)", m1_frac0, False),
        ("M1 entropy (lower=tighter)", m1_entropy, True),
        ("M1 max|T| (lower=tighter)", m1_max, True),
        ("M2 row Gram cardinality (lower=cleaner)", m2_row_card, True),
        ("M2 column Gram cardinality (lower=cleaner)", m2_col_card, True),
        ("M2 same_support (1=row==col)", m2_same_set, False),
        ("M3 leading IPR (lower=more delocalised)", m3_leading, True),
        ("M3 mean IPR (lower=more delocalised)", m3_mean, True),
        ("M3 max IPR (lower=more delocalised)", m3_max, True),
        ("M3 min IPR", m3_min, True),
        ("M3 leading IPR excess over 1/n (lower=more uniform)", m3_lead_excess, True),
    ]

    verdicts: dict[str, dict] = {}
    for name, vals, lower in candidates:
        ok, expl = _cleanly_ranks(vals, lower_is_better=lower)
        verdicts[name] = {
            "discriminates": bool(ok),
            "explanation": expl,
            "values": [float(v) for v in vals],
            "lower_is_better": bool(lower),
        }
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=60.0,
                        help="Per-proven-n re-derivation budget (seconds)")
    parser.add_argument("--out", type=str,
                        default=str(REPO / "results" / "higher_order_test.json"))
    args = parser.parse_args()

    print("[1/5] Loading proven n=17 ...", flush=True)
    M17, det17 = _load_proven(17, args.budget)

    print("[2/5] Loading proven n=15 ...", flush=True)
    M15, det15 = _load_proven(15, args.budget)

    print("[3/5] Loading H19 best (diagnostic SA) ...", flush=True)
    M19, det19 = _load_h19_best()
    print(f"  H19 |det|={det19:,d}", flush=True)

    print("[4/5] Loading H23 BTC ...", flush=True)
    M23btc, det23btc = _load_h23_btc()
    print(f"  H23 BTC |det|={det23btc:,d}", flush=True)

    print("[5/5] Generating random H23 control ...", flush=True)
    M23rnd, det23rnd = _random_h23(seed=42)
    print(f"  H23 random |det|={det23rnd:,d}", flush=True)

    matrices = [M17, M15, M19, M23btc, M23rnd]
    labels = [
        "proven n=17 (1.00)",
        "proven n=15 (1.00)",
        "our H19 (0.92)",
        "BTC H23 (1.00 BTC)",
        "random H23",
    ]
    abs_dets = [det17, det15, det19, det23btc, det23rnd]

    print("\n" + "=" * 110)
    print("Higher-order structural diagnostic comparison")
    print("=" * 110)
    rows = compare_matrices_v2(matrices, labels, print_table=True)

    for r, d in zip(rows, abs_dets):
        r["abs_det"] = int(d)

    print("\n" + "=" * 110)
    print("Discrimination scoring per scalar reduction")
    print("=" * 110)
    verdicts = _evaluate_discrimination(rows)
    any_clean = False
    for name, info in verdicts.items():
        flag = "DISCRIMINATES" if info["discriminates"] else "NULL"
        if info["discriminates"]:
            any_clean = True
        print(f"  {name:55s} -> {flag:14s}  {info['explanation']}")

    print()
    if any_clean:
        print(">>> At least one scalar reduction CLEANLY discriminates "
              "(proven vs near vs random). See list above for which.")
    else:
        print(">>> NULL result: no scalar reduction of M1/M2/M3 cleanly "
              "discriminates proven from near-winner from random.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": labels,
        "abs_dets": abs_dets,
        "rows": rows,
        "discrimination": verdicts,
        "any_clean_discriminator": bool(any_clean),
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
