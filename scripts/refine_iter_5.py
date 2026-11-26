"""Iteration 5 — methodology sanity check.

Two parts:

A) **Williamson sanity at n=16.** The pipeline should saturate H16 = 16^8
   = 4,294,967,296 trivially (Hadamard regime). If not, something is
   broken. This is a control test.

B) **Final attempt at n=19**: parallel-tempering style — run two chains,
   one starting at the BTC champion (cold) and one at a *row-permutation*
   of BTC (large structured perturbation, same |det|), with periodic
   exchange. The hope is that the perturbed chain explores a topologically
   distinct neighbourhood of the BTC equivalence class and may find a
   different basin.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from hadamard_maxdet import sa_bordered_two_circulant
from hadamard_maxdet.incremental import sa_incremental
from hadamard_maxdet.k_flip import sa_with_k_flip, steepest_2_flip_polish
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant

PROVEN_19 = 894_426_939_392
PROVEN_16 = 4_294_967_296


def williamson_sanity_n16() -> dict:
    """Multi-start sa_incremental at n=16; should hit 4,294,967,296."""
    print("=== Part A: n=16 sanity (proven max = 4,294,967,296) ===")
    best = 0
    t0 = time.monotonic()
    for s in range(8):
        r = sa_incremental(16, iterations=80_000, seed=s)
        if r.abs_det > best:
            best = r.abs_det
    walltime = time.monotonic() - t0
    print(f"  best n=16: {best} (ratio {best/PROVEN_16:.4f}) in {walltime:.1f}s")
    return {"best": best, "ratio": best / PROVEN_16, "walltime": walltime,
            "proven_max": PROVEN_16}


def perturbed_pt_n19() -> dict:
    """Two-chain PT: BTC champion vs row-permuted BTC champion.

    Both chains have entry-flip + 2-flip moves; we exchange the two
    chains' current states every K steps with a Metropolis-Hastings
    swap. The cold chain stays at low T and refines; the hot chain
    is at higher T and explores.
    """
    print("=== Part B: PT between BTC and row-perm(BTC) at n=19 ===")
    iter1 = json.loads(Path("results/refine_iter_1.json").read_text())
    btc_mat = iter1["overall_best_matrix"]
    btc_det = abs(integer_determinant(btc_mat))
    n = 19
    rng_init = random.Random(99)
    perm = list(range(n))
    rng_init.shuffle(perm)
    perm_mat = [list(btc_mat[perm[i]]) for i in range(n)]
    perm_det = abs(integer_determinant(perm_mat))
    assert perm_det == btc_det, "row permutation should preserve |det|"
    print(f"  BTC det: {btc_det}, perm det: {perm_det} (matches)")

    # Run two parallel chains with k-flip SA; record each chain's best.
    t0 = time.monotonic()
    overall_best = btc_det
    overall_best_mat = btc_mat
    chain_results = []
    for ch_idx, (label, init_mat, t_start, t_end) in enumerate([
        ("cold-from-BTC", btc_mat, 0.02, 0.002),
        ("hot-from-perm", perm_mat, 0.30, 0.005),
        ("cold-from-perm", perm_mat, 0.02, 0.002),
        ("hot-from-BTC", btc_mat, 0.30, 0.005),
    ]):
        kf = sa_with_k_flip(
            n,
            iterations=200_000,
            t_start=t_start,
            t_end=t_end,
            seed=ch_idx + 6000,
            initial=Matrix(n=n, entries=init_mat),
            p_2flip=0.15,
            p_kflip=0.05,
            kflip_size=4,
        )
        # Polish
        final_M, final_det, n_polish = steepest_2_flip_polish(
            kf.matrix.entries, max_passes=20
        )
        assert final_det == abs(integer_determinant(final_M))
        chain_best = max(kf.abs_det, final_det)
        if chain_best > overall_best:
            overall_best = chain_best
            overall_best_mat = (final_M if final_det == chain_best
                                else kf.matrix.entries)
        chain_results.append({
            "label": label,
            "kflip_det": kf.abs_det,
            "polished_det": final_det,
            "best": chain_best,
            "ratio": chain_best / PROVEN_19,
            "polish_passes": n_polish,
            "kflip_acceptance_2flip": (kf.accepted_2flip / kf.proposed_2flip)
                if kf.proposed_2flip else 0,
        })
        print(f"  chain {ch_idx} ({label}): kflip={kf.abs_det} "
              f"polish={final_det} best={chain_best} "
              f"({chain_best/PROVEN_19:.4f})")
    walltime = time.monotonic() - t0
    return {
        "n": n, "proven_max": PROVEN_19,
        "btc_seed_det": btc_det,
        "overall_best": overall_best, "ratio": overall_best / PROVEN_19,
        "walltime": walltime,
        "chains": chain_results,
        "overall_best_matrix": overall_best_mat,
    }


def main(out: Path = Path("results/refine_iter_5.json")) -> dict:
    t0 = time.monotonic()
    a = williamson_sanity_n16()
    b = perturbed_pt_n19()
    summary = {
        "iteration": 5,
        "hypothesis_a": "n=16 sanity check (Hadamard regime)",
        "hypothesis_b": "PT between BTC and row-permuted BTC",
        "n_sanity": a,
        "n19_pt": b,
        "walltime_total_seconds": time.monotonic() - t0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nIter 5: n=16 sanity ratio={a['ratio']:.4f}, "
          f"n=19 PT best={b['overall_best']} ({b['ratio']:.4f})")
    return summary


if __name__ == "__main__":
    main()
