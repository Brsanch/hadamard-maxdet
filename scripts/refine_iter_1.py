"""Iteration 1 — k-flip + 2-flip polish around the bordered_two_circulant
seed at n=19.

Hypothesis: SA reaches the BTC ansatz local-optimum at 841,813,590,016
(0.9412 of proven 894,426,939,392). Single-entry-flip SA is sealed at this
seed (per the n=23 diagnostic, expected to also bind at n=19). A 2-flip
polish (rank-2 exact O(n^4) enumeration) plus stochastic 4-flip moves
released onto the *full* ±1 lattice (no ansatz cage) may escape the
ansatz manifold to a higher basin.

Procedure: 8 multi-starts. For each start:
  1) SA on bordered_two_circulant ansatz (60s) -> seed matrix.
  2) Steepest 2-flip polish (until 2-flip local maximum).
  3) Free SA on full ±1 lattice with 10% 2-flip + 5% 4-flip moves
     (60s), seeded from the polished BTC matrix.
  4) Final steepest 2-flip polish.

Verify each |det| with exact Bareiss. Save best.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hadamard_maxdet import sa_bordered_two_circulant
from hadamard_maxdet.k_flip import sa_with_k_flip, steepest_2_flip_polish
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant

N = 19
PROVEN = 894_426_939_392
N_STARTS = 6
T_BTC_S = 12
T_FREE_S = 12


def main(out: Path = Path("results/refine_iter_1.json")) -> dict:
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

    for s in range(N_STARTS):
        t_seed_start = time.monotonic()
        # Step 1: BTC SA (uses Bareiss internally; budget by iters)
        btc = sa_bordered_two_circulant(N, iterations=70_000, seed=s)
        btc_best = btc.abs_det
        btc_mat = btc.matrix.entries
        t_btc = time.monotonic() - t_seed_start

        # Step 2: 2-flip polish
        t1 = time.monotonic()
        polished_M, polished_det, n_polish_1 = steepest_2_flip_polish(
            btc_mat, max_passes=30
        )
        t_polish_1 = time.monotonic() - t1
        # verify
        assert polished_det == abs(integer_determinant(polished_M))
        move_summary["total_polish_improvers"] += n_polish_1

        # Step 3: free SA with k-flip moves (no ansatz)
        t2 = time.monotonic()
        kf = sa_with_k_flip(
            N,
            iterations=200_000,
            t_start=0.05,    # warm restart at the cool tail (per move-profile finding)
            t_end=0.005,
            seed=s + 1000,
            initial=Matrix(n=N, entries=polished_M),
            p_2flip=0.10,
            p_kflip=0.05,
            kflip_size=4,
            refresh_every=2000,
        )
        t_kflip = time.monotonic() - t2
        move_summary["total_proposed_single"] += kf.proposed_single
        move_summary["total_accepted_single"] += kf.accepted_single
        move_summary["total_proposed_2flip"] += kf.proposed_2flip
        move_summary["total_accepted_2flip"] += kf.accepted_2flip
        move_summary["total_proposed_kflip"] += kf.proposed_kflip
        move_summary["total_accepted_kflip"] += kf.accepted_kflip

        # Step 4: final 2-flip polish
        t3 = time.monotonic()
        final_M, final_det, n_polish_2 = steepest_2_flip_polish(
            kf.matrix.entries, max_passes=30
        )
        t_polish_2 = time.monotonic() - t3
        assert final_det == abs(integer_determinant(final_M))
        move_summary["total_polish_improvers"] += n_polish_2

        seed_best = max(btc_best, polished_det, kf.abs_det, final_det)
        if seed_best > overall_best:
            overall_best = seed_best
            overall_best_mat = (
                final_M if final_det == seed_best
                else kf.matrix.entries if kf.abs_det == seed_best
                else polished_M if polished_det == seed_best
                else btc_mat
            )

        per_start.append({
            "seed": s,
            "btc_det": btc_best,
            "polished_after_btc": polished_det,
            "polish_passes_1": n_polish_1,
            "free_sa_kflip_det": kf.abs_det,
            "polished_final": final_det,
            "polish_passes_2": n_polish_2,
            "best": seed_best,
            "ratio": seed_best / PROVEN,
            "t_btc": t_btc,
            "t_polish_1": t_polish_1,
            "t_kflip_sa": t_kflip,
            "t_polish_2": t_polish_2,
            "kflip_acceptance_single": (kf.accepted_single / kf.proposed_single)
                if kf.proposed_single else 0,
            "kflip_acceptance_2flip": (kf.accepted_2flip / kf.proposed_2flip)
                if kf.proposed_2flip else 0,
            "kflip_acceptance_kflip": (kf.accepted_kflip / kf.proposed_kflip)
                if kf.proposed_kflip else 0,
        })
        print(f"start {s}: btc={btc_best} polish1={polished_det} "
              f"freeSA={kf.abs_det} polish2={final_det} best={seed_best} "
              f"({seed_best/PROVEN:.4f})")

    if overall_best_mat is not None:
        # Final exact verify
        v = abs(integer_determinant(overall_best_mat))
        assert v == overall_best, f"verify mismatch {v} vs {overall_best}"

    summary = {
        "iteration": 1,
        "hypothesis": "BTC seed + 2-flip polish + free-SA with k-flip moves",
        "n": N,
        "proven_max": PROVEN,
        "overall_best": overall_best,
        "ratio": overall_best / PROVEN,
        "n_starts": N_STARTS,
        "walltime_total_seconds": time.monotonic() - t_total,
        "per_start": per_start,
        "move_summary": move_summary,
        "overall_best_matrix": overall_best_mat,
    }
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nIter 1 best n={N}: {overall_best} ratio={overall_best/PROVEN:.4f}")
    return summary


if __name__ == "__main__":
    main()
