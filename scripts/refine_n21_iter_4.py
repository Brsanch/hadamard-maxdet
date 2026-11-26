"""Iteration 4: cross-equivalence-class crossover at n=21.

Take two distinct local optima at n=21:
  parent A: BTC seed (51.2T, ratio 0.8621)
  parent B: block-cyclic enum (41.2T, ratio 0.6945) - iter 1 result
  parent C: GS-array enum default border (40.96T, ratio 0.6896) - iter 2

Recombine row-wise / column-wise / 7x7-block-wise (3 schemes), then
refine each offspring with sa_incremental on the unconstrained ±1 lattice.

If the BTC and block-cyclic seeds are in the same Hadamard-equivalence
class (same SA basin), crossover should yield offspring that all
collapse back to BTC under SA. If they are in distinct classes, the
hybrid offspring may seed a new basin → potentially > 51.2T.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from hadamard_maxdet.incremental import sa_incremental
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.n21_block_cyclic import block_cyclic_circulant
from hadamard_maxdet.n21_crossover import (
    block_crossover,
    column_crossover,
    row_crossover,
)
from hadamard_maxdet.n21_gs_array import bordered_gs
from hadamard_maxdet.rich_ansatz import sa_bordered_two_circulant
from hadamard_maxdet.verifier import integer_determinant


def _bits_n(idx: int, n: int) -> list[int]:
    return [1 if (idx >> k) & 1 else -1 for k in range(n)]


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "refine_n21_iter_4.json"

    print("=== Iteration 4: cross-equivalence-class crossover, n=21 ===")
    print()

    # Parent A: BTC seed
    print("Parent A: BTC seed...")
    r_btc = sa_bordered_two_circulant(21, iterations=15_000, t_start=1.0, t_end=0.005, seed=0, full_border=True)
    parent_A = r_btc.matrix.entries
    det_A = abs(integer_determinant(parent_A))
    print(f"  |det(A)| = {det_A:,}")

    # Parent B: block-cyclic enum (load from iter 1)
    iter1 = json.loads((out_dir / "refine_n21_iter_1.json").read_text())
    a_idx, b_idx, c_idx = iter1["enum_best_idx"]
    a, b, c = _bits_n(a_idx, 7), _bits_n(b_idx, 7), _bits_n(c_idx, 7)
    M_B = block_cyclic_circulant(a, b, c)
    parent_B = M_B.entries
    det_B = abs(integer_determinant(parent_B))
    print(f"  |det(B)| = {det_B:,}  (block-cyclic enum)")

    # Parent C: GS-array enum (load from iter 2)
    iter2 = json.loads((out_dir / "refine_n21_iter_2.json").read_text())
    gs_idx = iter2["enum_best_idx"]
    aa, bb, cc, dd = (_bits_n(i, 5) for i in gs_idx)
    M_C = bordered_gs(aa, bb, cc, dd)
    parent_C = M_C.entries
    det_C = abs(integer_determinant(parent_C))
    print(f"  |det(C)| = {det_C:,}  (GS-array enum)")

    parents = {"A_btc": parent_A, "B_blockcyc": parent_B, "C_gs": parent_C}

    # Crossovers + SA refinement
    rng = random.Random(0)
    n_offspring_per_pair = 6
    sa_iters = 10_000

    pairs = [
        ("A_btc", "B_blockcyc"),
        ("A_btc", "C_gs"),
        ("B_blockcyc", "C_gs"),
    ]
    schemes = [("row", row_crossover), ("col", column_crossover), ("block7", block_crossover)]

    overall_best = max(det_A, det_B, det_C)
    overall_best_M = parent_A
    overall_best_label = "parent_A_btc"
    cross_results: list[dict] = []

    print()
    print(f"Producing {n_offspring_per_pair} offspring per pair × scheme; SA-refine each ({sa_iters} iters).")
    t0 = time.monotonic()
    for (lp, rp) in pairs:
        for (scheme_name, scheme_fn) in schemes:
            for k in range(n_offspring_per_pair):
                offspring = scheme_fn(parents[lp], parents[rp], rng)
                # Verify ±1
                # Could be singular; SA handles that
                init_det = abs(integer_determinant(offspring))
                # SA refine
                M_off = Matrix(n=21, entries=offspring)
                r = sa_incremental(
                    21,
                    iterations=sa_iters,
                    t_start=0.5,
                    t_end=0.005,
                    seed=rng.randrange(1 << 30),
                    initial=M_off,
                    refresh_every=2000,
                )
                cross_results.append({
                    "pair": f"{lp}+{rp}",
                    "scheme": scheme_name,
                    "k": k,
                    "init_det": init_det,
                    "refined_det": r.abs_det,
                })
                if r.abs_det > overall_best:
                    overall_best = r.abs_det
                    overall_best_M = r.matrix.entries
                    overall_best_label = f"{lp}+{rp}_{scheme_name}_{k}"
    cross_wall = time.monotonic() - t0
    print(f"  crossover sweep walltime = {cross_wall:.1f}s  ({len(cross_results)} runs)")

    # Verify
    overall_verified = abs(integer_determinant(overall_best_M))
    assert overall_verified == overall_best

    # Stats: how many offspring beat each parent's |det| post-refine?
    pa_count = sum(1 for r in cross_results if r["refined_det"] > det_A)
    pb_count = sum(1 for r in cross_results if r["refined_det"] > det_B)
    pc_count = sum(1 for r in cross_results if r["refined_det"] > det_C)
    eq_pa = sum(1 for r in cross_results if r["refined_det"] == det_A)
    print()
    print(f"Offspring statistics over {len(cross_results)} refined hybrids:")
    print(f"  beat parent_A (51.2T): {pa_count}")
    print(f"  ==  parent_A          : {eq_pa}")
    print(f"  beat parent_B (41.2T): {pb_count}")
    print(f"  beat parent_C (40.96T): {pc_count}")
    print(f"  refined-det range: {min(r['refined_det'] for r in cross_results):,} .. {max(r['refined_det'] for r in cross_results):,}")

    proven = 59_392_000_000_000
    ratio = overall_best / proven
    print()
    print(f"=== Iteration 4 summary ===")
    print(f"  parent_A (BTC)          = {det_A:,}")
    print(f"  parent_B (block-cyc)    = {det_B:,}")
    print(f"  parent_C (GS-array)     = {det_C:,}")
    print(f"  best refined offspring  = {overall_best:,}  ({overall_best_label})")
    print(f"  proven n=21 max         = {proven:,}")
    print(f"  ratio                   = {ratio:.4f}")

    payload = {
        "iteration": 4,
        "hypothesis": "row/col/7-block crossover of BTC × block-cyclic × GS parents + SA refine",
        "n": 21,
        "proven_max": proven,
        "parent_A_det": det_A,
        "parent_B_det": det_B,
        "parent_C_det": det_C,
        "n_offspring": len(cross_results),
        "n_beat_parent_A": pa_count,
        "n_eq_parent_A": eq_pa,
        "n_beat_parent_B": pb_count,
        "n_beat_parent_C": pc_count,
        "crossover_walltime_s": cross_wall,
        "overall_best_det": overall_best,
        "overall_best_label": overall_best_label,
        "overall_ratio": ratio,
        "winner_matrix": overall_best_M,
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
