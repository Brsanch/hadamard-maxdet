"""Decode the structural fingerprint of winning matrices.

Generates :class:`winner_decode.WinnerFingerprint` for n in {8, 11, 13,
15, 17}, attempts heavy reconstruction at H19, prints a comparison
table, and saves to ``results/winner_decode.json``.

Usage::

    python scripts/decode_winners.py [--budget-h19 600]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo src/ on path when run from repo root.
HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.winner_decode import (
    fingerprint_matrix,
    reconstruct_winner,
    winner_fingerprint,
)


def fmt_int(x: int | None) -> str:
    if x is None:
        return "?"
    return f"{x:_}"


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "?"
    return f"{x*100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-h19",
        type=float,
        default=600.0,
        help="Wall-clock seconds for the heavy H19 reconstruction (default 600).",
    )
    parser.add_argument(
        "--budget-small",
        type=float,
        default=25.0,
        help="Per-n structured-SA seconds for small-n fingerprints (default 25).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/winner_decode.json",
    )
    args = parser.parse_args()

    out: dict = {
        "small_n": [],
        "h19_reconstruction": None,
        "h19_matrix": None,
        "comparison_table": [],
    }

    # 1) Small-n fingerprints. Include n=8 (Sylvester Hadamard) for sanity.
    print("\n=== Small-n winner fingerprints ===\n")

    # n=8 Hadamard via Sylvester (instant).
    fp8 = fingerprint_matrix(Matrix.sylvester(8).entries, source="sylvester")
    out["small_n"].append(fp8.to_json(include_matrix=False))
    print(
        f"n= 8: |det|={fmt_int(fp8.abs_det_verified)} "
        f"ratio={fmt_pct(fp8.ratio_to_proven)} "
        f"support={fp8.gram_offdiag_support} "
        f"sv_dist={fp8.sv_distinct_count}"
    )

    # n=11..17: regenerate via SA + bordered_two_circulant.
    for n in (11, 13, 15, 17):
        t0 = time.monotonic()
        fp = winner_fingerprint(
            n,
            n_starts=8,
            iterations=120_000,
            structured_seconds=args.budget_small,
            base_seed=12345 + n * 31,
        )
        wall = time.monotonic() - t0
        out["small_n"].append(fp.to_json(include_matrix=False))
        print(
            f"n={n:>2}: |det|={fmt_int(fp.abs_det_verified)} "
            f"ratio={fmt_pct(fp.ratio_to_proven)} "
            f"support={fp.gram_offdiag_support} "
            f"sv_dist={fp.sv_distinct_count} "
            f"rw_dist={fp.distinct_row_weight_values} "
            f"src={fp.source} wall={wall:.1f}s"
        )

    # 2) H19 heavy reconstruction.
    print(f"\n=== H19 heavy reconstruction ({args.budget_h19:.0f} s budget) ===\n")
    t0 = time.monotonic()
    fp19 = reconstruct_winner(
        19, wall_seconds=args.budget_h19, base_seed=4242, sa_chunk_iters=50_000,
    )
    wall = time.monotonic() - t0
    print(
        f"H19 best |det|={fmt_int(fp19.abs_det_verified)} "
        f"proven={fmt_int(fp19.proven_max)} "
        f"ratio={fmt_pct(fp19.ratio_to_proven)} "
        f"src={fp19.source} wall={wall:.1f}s"
    )
    print(
        f"H19 fingerprint: support={fp19.gram_offdiag_support} "
        f"sv_dist={fp19.sv_distinct_count} "
        f"rw_dist={fp19.distinct_row_weight_values} "
        f"best_part={fp19.best_partition} "
        f"intra_cross={fp19.best_partition_intra_cross} "
        f"score={fp19.best_partition_score}"
    )
    out["h19_reconstruction"] = fp19.to_json(include_matrix=False)
    out["h19_matrix"] = fp19.matrix
    out["small_n"].append(fp19.to_json(include_matrix=False))

    # 3) Comparison table.
    table_rows = []
    for entry in out["small_n"]:
        table_rows.append(
            {
                "n": entry["n"],
                "abs_det": entry["abs_det"],
                "proven_max": entry["proven_max"],
                "ratio_to_proven": entry["ratio_to_proven"],
                "ratio_to_upper_bound": entry["ratio_to_upper_bound"],
                "gram_offdiag_support": entry["gram_offdiag_support"],
                "gram_offdiag_cardinality": entry["gram_offdiag_cardinality"],
                "sv_distinct_count": entry["sv_distinct_count"],
                "distinct_row_weight_values": entry["distinct_row_weight_values"],
                "best_partition": entry["best_partition"],
                "best_partition_intra_cross": entry["best_partition_intra_cross"],
                "best_partition_score": entry["best_partition_score"],
                "source": entry["source"],
            }
        )
    out["comparison_table"] = table_rows

    print("\n=== Cross-n comparison ===\n")
    print(
        f"{'n':>3}  {'ratio':>7}  {'gram_card':>9}  "
        f"{'support':<28}  {'sv':>3}  {'rw':>3}  best_part / (a,b) / score"
    )
    for r in table_rows:
        ratio = (
            f"{r['ratio_to_proven']*100:>6.2f}%"
            if r["ratio_to_proven"] is not None
            else "  ?   "
        )
        sup = ",".join(str(s) for s in r["gram_offdiag_support"])
        bp = r["best_partition"] or "—"
        ab = r["best_partition_intra_cross"] or "—"
        sc = (
            f"{r['best_partition_score']:.3f}"
            if r["best_partition_score"] is not None
            else "—"
        )
        print(
            f"{r['n']:>3}  {ratio}  {r['gram_offdiag_cardinality']:>9}  "
            f"{sup:<28}  {r['sv_distinct_count']:>3}  "
            f"{r['distinct_row_weight_values']:>3}  {bp} / {ab} / {sc}"
        )

    # 4) Save.
    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
