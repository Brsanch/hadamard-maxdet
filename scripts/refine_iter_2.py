"""Iteration 2 — paley_deletion seeds at n=19 + k-flip refinement.

Hypothesis: BTC seed sealed under d=2 (iter 1). Try a *different* seed
family (Paley I, n+1=20=q+1, q=19 prime, q≡3 mod 4 → Paley I exists)
which lives in a structurally distinct sub-lattice. Refine each Paley
seed with our k-flip SA, then a final 2-flip polish. If any Paley start
exceeds 841,813,590,016 we have evidence that the BTC basin is not
unique-best.

Procedure:
  1) Enumerate Paley seeds at n=19 (paley_deletion_seeds returns 400).
  2) Take top-K=8 by |det| (ties broken arbitrarily).
  3) For each, run free SA with k-flip moves (200k iterations) starting
     from the Paley seed; warm T=0.5 since seed is far from optimum.
  4) Final 2-flip polish.
  5) Verify with Bareiss.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hadamard_maxdet.k_flip import sa_with_k_flip, steepest_2_flip_polish
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.rich_ansatz import paley_deletion_seeds
from hadamard_maxdet.verifier import integer_determinant

N = 19
PROVEN = 894_426_939_392
N_STARTS = 8


def main(out: Path = Path("results/refine_iter_2.json")) -> dict:
    out.parent.mkdir(exist_ok=True, parents=True)
    overall_best = 0
    overall_best_mat: list[list[int]] | None = None
    per_start: list[dict] = []
    t_total = time.monotonic()
    move_summary = {
        "total_proposed_single": 0,
        "total_accepted_single": 0,
        "total_proposed_2flip": 0,
        "total_accepted_2flip": 0,
        "total_proposed_kflip": 0,
        "total_accepted_kflip": 0,
        "total_polish_improvers": 0,
    }

    print("Enumerating Paley deletion seeds at n=19...")
    t0 = time.monotonic()
    seeds = paley_deletion_seeds(N)
    print(f"  got {len(seeds)} seeds in {time.monotonic()-t0:.1f}s; "
          f"best seed |det| = {max(s.abs_det for s in seeds)} "
          f"(ratio {max(s.abs_det for s in seeds)/PROVEN:.4f})")
    # Sort by abs_det desc, take top-N_STARTS distinct (use entries hash to dedupe)
    seeds.sort(key=lambda s: -s.abs_det)
    seen_keys = set()
    top: list = []
    for s in seeds:
        key = tuple(tuple(r) for r in s.matrix.entries)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        top.append(s)
        if len(top) >= N_STARTS:
            break
    print(f"  refining top {len(top)} unique seeds")

    for s_idx, seed_obj in enumerate(top):
        seed_det = seed_obj.abs_det
        seed_mat = seed_obj.matrix.entries

        # Free SA with k-flip from Paley seed (warm because seed is far from optimum)
        t1 = time.monotonic()
        kf = sa_with_k_flip(
            N,
            iterations=200_000,
            t_start=0.5,
            t_end=0.005,
            seed=s_idx + 2000,
            initial=Matrix(n=N, entries=seed_mat),
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

        # Final 2-flip polish
        t2 = time.monotonic()
        final_M, final_det, n_polish = steepest_2_flip_polish(
            kf.matrix.entries, max_passes=30
        )
        t_polish = time.monotonic() - t2
        assert final_det == abs(integer_determinant(final_M))
        move_summary["total_polish_improvers"] += n_polish

        seed_best = max(seed_det, kf.abs_det, final_det)
        if seed_best > overall_best:
            overall_best = seed_best
            overall_best_mat = (
                final_M if final_det == seed_best
                else kf.matrix.entries if kf.abs_det == seed_best
                else seed_mat
            )

        per_start.append({
            "start_index": s_idx,
            "paley_seed_det": seed_det,
            "free_sa_kflip_det": kf.abs_det,
            "polished_final": final_det,
            "polish_passes": n_polish,
            "best": seed_best,
            "ratio": seed_best / PROVEN,
            "t_kflip_sa": t_kflip,
            "t_polish": t_polish,
            "kflip_acceptance_single": (kf.accepted_single / kf.proposed_single)
                if kf.proposed_single else 0,
            "kflip_acceptance_2flip": (kf.accepted_2flip / kf.proposed_2flip)
                if kf.proposed_2flip else 0,
            "kflip_acceptance_kflip": (kf.accepted_kflip / kf.proposed_kflip)
                if kf.proposed_kflip else 0,
        })
        print(f"start {s_idx}: paley={seed_det} freeSA={kf.abs_det} "
              f"polish={final_det} best={seed_best} "
              f"({seed_best/PROVEN:.4f}) t_sa={t_kflip:.1f}s")

    if overall_best_mat is not None:
        v = abs(integer_determinant(overall_best_mat))
        assert v == overall_best, f"verify mismatch {v} vs {overall_best}"

    summary = {
        "iteration": 2,
        "hypothesis": "Paley-deletion seed at n=19 + k-flip free SA + 2-flip polish",
        "n": N,
        "proven_max": PROVEN,
        "overall_best": overall_best,
        "ratio": overall_best / PROVEN,
        "n_starts": len(top),
        "best_paley_seed_det": top[0].abs_det,
        "best_paley_seed_ratio": top[0].abs_det / PROVEN,
        "walltime_total_seconds": time.monotonic() - t_total,
        "per_start": per_start,
        "move_summary": move_summary,
        "overall_best_matrix": overall_best_mat,
    }
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nIter 2 best n={N}: {overall_best} ratio={overall_best/PROVEN:.4f}")
    return summary


if __name__ == "__main__":
    main()
