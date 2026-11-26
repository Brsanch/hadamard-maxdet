"""Iteration 3 — basin-bridging crossover at n=19.

Hypothesis: the BTC seed (det=841,813,590,016, ratio 0.9412) is sealed
under d=2 and Paley-seed refinement lands in disjoint basins (max
0.8903). If we *combine* rows from both parents using row-aligned
crossover, the child sits in a third basin distinct from both, and SA
refinement may discover that basin's optimum is higher than either
parent's.

Procedure:
  1) Recover the iter 1 BTC champion matrix (det=841,813,590,016).
  2) Recover the iter 2 Paley-refined champion (det=796,344,975,360).
  3) Generate 24 row-recombined children with various p_a mixes.
  4) Refine each promising child (top 8 by initial det) with k-flip SA
     + 2-flip polish.
  5) Verify every reported det with Bareiss.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hadamard_maxdet.crossover import crossover_children
from hadamard_maxdet.k_flip import sa_with_k_flip, steepest_2_flip_polish
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant

N = 19
PROVEN = 894_426_939_392


def main(out: Path = Path("results/refine_iter_3.json")) -> dict:
    out.parent.mkdir(exist_ok=True, parents=True)
    iter1 = json.loads(Path("results/refine_iter_1.json").read_text())
    iter2 = json.loads(Path("results/refine_iter_2.json").read_text())
    parent_a = iter1["overall_best_matrix"]
    parent_b = iter2["overall_best_matrix"]
    det_a = abs(integer_determinant(parent_a))
    det_b = abs(integer_determinant(parent_b))
    assert det_a == iter1["overall_best"], "parent_a det mismatch"
    assert det_b == iter2["overall_best"], "parent_b det mismatch"
    print(f"parent A (BTC, iter 1): {det_a} ({det_a/PROVEN:.4f})")
    print(f"parent B (Paley, iter 2): {det_b} ({det_b/PROVEN:.4f})")

    t0 = time.monotonic()
    children = crossover_children(
        parent_a, parent_b,
        n_children=32, seed=12345,
        p_a_choices=(0.2, 0.35, 0.5, 0.65, 0.8),
    )
    t_cross = time.monotonic() - t0
    print(f"  generated {len(children)} non-singular children in {t_cross:.2f}s")
    print(f"  top 5 child dets: {[c[1] for c in children[:5]]}")

    # Refine top 8
    overall_best = max(det_a, det_b)
    overall_best_mat = parent_a if det_a >= det_b else parent_b
    per_child: list[dict] = []
    move_summary = {
        "total_proposed_single": 0, "total_accepted_single": 0,
        "total_proposed_2flip": 0, "total_accepted_2flip": 0,
        "total_proposed_kflip": 0, "total_accepted_kflip": 0,
        "total_polish_improvers": 0,
    }
    n_refine = min(8, len(children))
    for c_idx in range(n_refine):
        child_M, child_det = children[c_idx]

        t1 = time.monotonic()
        kf = sa_with_k_flip(
            N,
            iterations=200_000,
            t_start=0.3,
            t_end=0.005,
            seed=c_idx + 3000,
            initial=Matrix(n=N, entries=child_M),
            p_2flip=0.10,
            p_kflip=0.05,
            kflip_size=4,
            refresh_every=2000,
        )
        t_kflip = time.monotonic() - t1
        move_summary["total_proposed_single"] += kf.proposed_single
        move_summary["total_accepted_single"] += kf.accepted_single
        move_summary["total_proposed_2flip"] += kf.proposed_2flip
        move_summary["total_accepted_2flip"] += kf.accepted_2flip
        move_summary["total_proposed_kflip"] += kf.proposed_kflip
        move_summary["total_accepted_kflip"] += kf.accepted_kflip

        t2 = time.monotonic()
        final_M, final_det, n_polish = steepest_2_flip_polish(
            kf.matrix.entries, max_passes=30
        )
        t_polish = time.monotonic() - t2
        assert final_det == abs(integer_determinant(final_M))
        move_summary["total_polish_improvers"] += n_polish

        child_best = max(child_det, kf.abs_det, final_det)
        if child_best > overall_best:
            overall_best = child_best
            overall_best_mat = (
                final_M if final_det == child_best
                else kf.matrix.entries if kf.abs_det == child_best
                else child_M
            )

        per_child.append({
            "child_index": c_idx,
            "child_seed_det": child_det,
            "free_sa_kflip_det": kf.abs_det,
            "polished_final": final_det,
            "polish_passes": n_polish,
            "best": child_best,
            "ratio": child_best / PROVEN,
            "t_kflip_sa": t_kflip,
            "t_polish": t_polish,
        })
        print(f"child {c_idx}: seed={child_det} freeSA={kf.abs_det} "
              f"polish={final_det} best={child_best} ({child_best/PROVEN:.4f})")

    summary = {
        "iteration": 3,
        "hypothesis": "row-aligned crossover of iter1 BTC + iter2 Paley champions",
        "n": N,
        "proven_max": PROVEN,
        "parent_a_det": det_a,
        "parent_b_det": det_b,
        "overall_best": overall_best,
        "ratio": overall_best / PROVEN,
        "n_children_generated": len(children),
        "n_children_refined": n_refine,
        "best_initial_child_det": children[0][1] if children else 0,
        "walltime_total_seconds": time.monotonic() - t0,
        "per_child": per_child,
        "move_summary": move_summary,
        "overall_best_matrix": overall_best_mat,
    }
    if overall_best_mat is not None:
        v = abs(integer_determinant(overall_best_mat))
        assert v == overall_best
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nIter 3 best n={N}: {overall_best} ratio={overall_best/PROVEN:.4f}")
    return summary


if __name__ == "__main__":
    main()
