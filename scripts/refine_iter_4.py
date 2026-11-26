"""Iteration 4 — tabu search, push the chain away from the BTC fingerprint.

Hypothesis: SA + every refinement variant we've tried lands in the BTC
basin (det=841,813,590,016, ratio 0.9412). If we explicitly pre-seed
the tabu list with the BTC champion's Gram off-diagonal fingerprint,
the search MUST land somewhere else - and we test whether that
"somewhere else" can exceed the BTC basin.

Procedure:
  1) Compute BTC champion fingerprint (Gram off-diag multiset).
  2) Run sa_tabu with the BTC fingerprint pre-seeded as tabu, multi-start.
  3) For each result, run k-flip + 2-flip polish.
  4) Verify with Bareiss.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hadamard_maxdet.k_flip import sa_with_k_flip, steepest_2_flip_polish
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.tabu import fingerprint_offdiag, sa_tabu
from hadamard_maxdet.verifier import integer_determinant

N = 19
PROVEN = 894_426_939_392
N_STARTS = 6


def main(out: Path = Path("results/refine_iter_4.json")) -> dict:
    out.parent.mkdir(exist_ok=True, parents=True)
    iter1 = json.loads(Path("results/refine_iter_1.json").read_text())
    btc_mat = iter1["overall_best_matrix"]
    btc_fp = fingerprint_offdiag(btc_mat)
    print(f"BTC champion det: {iter1['overall_best']}")
    print(f"BTC fingerprint multiset support: {sorted(set(btc_fp))}")

    overall_best = iter1["overall_best"]
    overall_best_mat = btc_mat
    per_start: list[dict] = []
    move_summary = {
        "tabu_rejected_total": 0,
        "metro_rejected_total": 0,
        "fingerprints_seen_total": 0,
        "polish_improvers_total": 0,
    }
    t0 = time.monotonic()

    for s in range(N_STARTS):
        t1 = time.monotonic()
        # tabu SA with BTC fingerprint pre-seeded
        tres = sa_tabu(
            N,
            iterations=80_000,
            t_start=1.0,
            t_end=0.01,
            seed=s + 4000,
            tabu_capacity=8192,
            fingerprint_every=2,   # check every 2 accepted moves to amortise n^2
            pre_seed_fingerprints=[btc_fp],
        )
        t_tabu = time.monotonic() - t1
        move_summary["tabu_rejected_total"] += tres.rejected_by_tabu
        move_summary["metro_rejected_total"] += tres.rejected_by_metropolis
        move_summary["fingerprints_seen_total"] += tres.fingerprints_seen

        # Verify we DIDN'T land in BTC basin
        landed_fp = fingerprint_offdiag(tres.matrix.entries)
        landed_in_btc = (landed_fp == btc_fp)

        # Refine with k-flip SA
        t2 = time.monotonic()
        kf = sa_with_k_flip(
            N,
            iterations=120_000,
            t_start=0.05,
            t_end=0.005,
            seed=s + 5000,
            initial=tres.matrix,
            p_2flip=0.10,
            p_kflip=0.05,
            kflip_size=4,
        )
        t_kflip = time.monotonic() - t2

        # Polish
        t3 = time.monotonic()
        final_M, final_det, n_polish = steepest_2_flip_polish(
            kf.matrix.entries, max_passes=30
        )
        t_polish = time.monotonic() - t3
        assert final_det == abs(integer_determinant(final_M))
        move_summary["polish_improvers_total"] += n_polish

        seed_best = max(tres.abs_det, kf.abs_det, final_det)
        if seed_best > overall_best:
            overall_best = seed_best
            overall_best_mat = (
                final_M if final_det == seed_best
                else kf.matrix.entries if kf.abs_det == seed_best
                else tres.matrix.entries
            )

        per_start.append({
            "seed": s,
            "tabu_sa_det": tres.abs_det,
            "tabu_landed_in_btc": landed_in_btc,
            "tabu_rejected": tres.rejected_by_tabu,
            "metro_rejected": tres.rejected_by_metropolis,
            "fingerprints_seen": tres.fingerprints_seen,
            "kflip_sa_det": kf.abs_det,
            "polished_final": final_det,
            "polish_passes": n_polish,
            "best": seed_best,
            "ratio": seed_best / PROVEN,
            "t_tabu_sa": t_tabu,
            "t_kflip_sa": t_kflip,
            "t_polish": t_polish,
        })
        print(f"start {s}: tabu={tres.abs_det} (in_btc={landed_in_btc}, "
              f"tabu_rej={tres.rejected_by_tabu}) "
              f"kflip={kf.abs_det} polish={final_det} "
              f"best={seed_best} ({seed_best/PROVEN:.4f})")

    summary = {
        "iteration": 4,
        "hypothesis": "tabu search with BTC fingerprint pre-seeded as tabu",
        "n": N,
        "proven_max": PROVEN,
        "btc_seed_det": iter1["overall_best"],
        "overall_best": overall_best,
        "ratio": overall_best / PROVEN,
        "n_starts": N_STARTS,
        "walltime_total_seconds": time.monotonic() - t0,
        "per_start": per_start,
        "move_summary": move_summary,
        "overall_best_matrix": overall_best_mat,
    }
    if overall_best_mat is not None:
        v = abs(integer_determinant(overall_best_mat))
        assert v == overall_best
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nIter 4 best n={N}: {overall_best} ratio={overall_best/PROVEN:.4f}")
    return summary


if __name__ == "__main__":
    main()
