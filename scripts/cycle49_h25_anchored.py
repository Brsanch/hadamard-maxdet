"""H25 cycle 49: structurally-anchored search.

Cycle 48 (commit 27d2e5d) ran the cycle-41 recipe (SA → greedy → 2-flip polish
→ tabu) at H25 from a random initialisation and stalled at |det| = 1.435e17,
0.5612 of the Ehlich–Wojtas bound. The post-mortem identified the missing
ingredient: cycle 41's success at H21 came from a clean Williamson H21
*anchor* (0.69 EW), not from the algorithm. From random init the recipe
cannot find the high plateau.

This cycle re-runs the same recipe from two structural anchors:

  Anchor A — Paley I H24, bordered to 25×25, with greedy ±1 optimisation
             of the 49 border entries. Bordered Hadamard at n = 4k+1 has a
             theoretical |det| ceiling of 24^12 · (1 + sqrt(24))/7 ≈ 0.842
             of EW, attained when ||H24^T · c_col||_1 ≈ 24·sqrt(24).

  Anchor B — Paley II H28 (q = 13, prime ≡ 1 mod 4), greedy 3-stage row+col
             deletion to H25. Each deletion stage typically loses 5-10% of
             |det| / EW, so a clean H28 → H25 descent should land in the
             0.65-0.80 range.

Pipeline per anchor (cycle-41 recipe, verbatim):

  1. SA, 4 chains × 200K single-entry-flip iters, t = 1.0 → 0.01.
  2. Greedy single-flip until 1-local max.
  3. Steepest 2-flip polish via rank-2 adjugate (O(n^4) per pass).
  4. SA + tabu, 4 chains × 5K iters, tabu capacity 1024.

Results: per-anchor JSON + npy at each pipeline stage. Honest metric is
final |det| / EW; a "win" is anything ≥ 0.62 (10% over cycle 48 baseline).
EW(25) = 7 · 24^12 = 2.5564e17 since 2·25 − 1 = 49 = 7^2 makes EW tight.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.k_flip import sa_with_k_flip, steepest_2_flip_polish
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.tabu import sa_tabu
from hadamard_maxdet.verifier import integer_determinant


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
EW_25 = 7 * 24**12  # 2 * 25 - 1 = 49 = 7^2 ⇒ Ehlich–Wojtas tight at n = 25


# ---------------------------------------------------------------------------
# Anchor A: Paley I H24 + bordered
# ---------------------------------------------------------------------------


def make_anchor_a(*, sa_iters: int = 200_000, seed: int = 42) -> tuple[Matrix, int]:
    """Paley I H24 in the lower-right; optimise border for |det|.

    Closed-form: |det(H25)| = 24^11 · |24 c - r^T H24^T c_col|, with c, r, c_col
    ∈ ±1. For fixed c_col, optimal r = sign(−H24^T c_col) and optimal c = +1
    give |det| = 24^11 · (24 + ||H24^T c_col||_1). So we maximise
    ||H24^T c_col||_1 over c_col ∈ {±1}^24 by SA on bit flips with O(24)
    incremental update of v = H24^T c_col.

    Theoretical ceiling: ||v||_1 ≤ 24·sqrt(24) by Cauchy-Schwarz, giving
    ratio (1 + sqrt(24))/7 ≈ 0.842 of EW(25).
    """
    import random as _random
    H24_entries = Matrix.paley_type_i(24).entries
    H = np.array(H24_entries, dtype=np.int64)
    HT = H.T

    rng = _random.Random(seed)
    c_col = np.array([1 if rng.random() < 0.5 else -1 for _ in range(24)], dtype=np.int64)
    v = HT @ c_col  # length 24
    cur_l1 = int(np.abs(v).sum())
    best_l1 = cur_l1
    best_c_col = c_col.copy()

    import math
    t_start, t_end = 24.0, 0.1
    log_ratio = math.log(t_end / t_start)
    for step in range(sa_iters):
        frac = step / max(1, sa_iters - 1)
        t = t_start * math.exp(log_ratio * frac)
        i = rng.randrange(24)
        delta_c = -2 * c_col[i]  # ±2
        new_v = v + delta_c * HT[:, i]
        new_l1 = int(np.abs(new_v).sum())
        d = cur_l1 - new_l1
        if d <= 0 or rng.random() < math.exp(-d / max(t, 1e-9)):
            c_col[i] = -c_col[i]
            v = new_v
            cur_l1 = new_l1
            if cur_l1 > best_l1:
                best_l1 = cur_l1
                best_c_col = c_col.copy()

    # Build optimal r and c.
    v_best = HT @ best_c_col
    r = np.where(v_best >= 0, -1, 1).astype(np.int64)  # r maximises |24 - r^T v|; sign(-v)
    c_corner = 1

    # Actually the optimum: pick c_corner ∈ ±1 to align with sign of −r^T v.
    # |24 c − r^T v| is max when c · sign(−r^T v) = +1; flip if needed.
    inner = int(r @ v_best)
    if 24 * c_corner - inner < 0 and -(24 * c_corner - inner) < (24 + abs(inner)):
        # Try negating c
        if abs(24 * (-c_corner) - inner) > abs(24 * c_corner - inner):
            c_corner = -c_corner

    H25 = [[0] * 25 for _ in range(25)]
    H25[0][0] = c_corner
    for j in range(24):
        H25[0][j + 1] = int(r[j])
    for i in range(24):
        H25[i + 1][0] = int(best_c_col[i])
    for i in range(24):
        for j in range(24):
            H25[i + 1][j + 1] = H24_entries[i][j]

    det_val = abs(integer_determinant(H25))
    return Matrix(n=25, entries=H25), det_val


# ---------------------------------------------------------------------------
# Anchor B: Paley II H28 → 3-stage greedy row+col deletion to H25
# ---------------------------------------------------------------------------


def _delete_rows_cols(
    entries: list[list[int]],
    rows: tuple[int, ...],
    cols: tuple[int, ...],
) -> list[list[int]]:
    rs = set(rows)
    cs = set(cols)
    return [[entries[r][c] for c in range(len(entries)) if c not in cs]
            for r in range(len(entries)) if r not in rs]


def make_anchor_b(
    *,
    n_samples: int = 5_000,
    seed: int = 12345,
) -> tuple[Matrix, int, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Paley II H28 → H25 via Monte-Carlo over random (3-row, 3-col) deletions.

    Greedy 1-at-a-time tie-breaks identically (all Hadamard cofactors share
    |det| = 28^13), so we randomly sample (3-row, 3-col) pairs from H28
    and keep the best 25×25 submatrix by |det|.
    """
    import itertools, random as _random
    H28 = [row[:] for row in Matrix.paley_type_ii(28).entries]
    rng = _random.Random(seed)
    all_triples = list(itertools.combinations(range(28), 3))
    best_det = 0
    best_sub: list[list[int]] = []
    best_rc: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    for _ in range(n_samples):
        rows = rng.choice(all_triples)
        cols = rng.choice(all_triples)
        sub = _delete_rows_cols(H28, rows, cols)
        d = abs(integer_determinant(sub))
        if d > best_det:
            best_det = d
            best_sub = sub
            best_rc = (rows, cols)
    return Matrix(n=25, entries=best_sub), best_det, best_rc


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def stage_sa(initial: Matrix, *, chains: int, iterations: int) -> tuple[Matrix, int]:
    """Pure single-entry-flip SA, run 4 chains, return best."""
    best = initial
    best_det = abs(integer_determinant(initial.entries))
    for c in range(chains):
        res = sa_with_k_flip(
            n=initial.n,
            iterations=iterations,
            t_start=1.0,
            t_end=0.01,
            seed=10000 + c,
            initial=initial,
            p_2flip=0.0,
            p_kflip=0.0,
        )
        if res.abs_det > best_det:
            best_det = res.abs_det
            best = res.matrix
    return best, best_det


def stage_greedy_single_flip(initial: Matrix, *, max_passes: int = 100) -> tuple[Matrix, int]:
    """Take best improving single-flip until 1-local max."""
    state = IncrementalState.from_matrix([row[:] for row in initial.entries])
    n = state.n
    for _ in range(max_passes):
        cur = abs(state.det)
        best_new = cur
        best_ij: tuple[int, int] | None = None
        for i in range(n):
            for j in range(n):
                d_new = propose_entry_flip(state, i, j)
                if abs(d_new) > best_new:
                    best_new = abs(d_new)
                    best_ij = (i, j)
        if best_ij is None:
            break
        apply_entry_flip(state, best_ij[0], best_ij[1])
    return Matrix(n=n, entries=[row[:] for row in state.M]), abs(state.det)


def stage_polish_2flip(initial: Matrix, *, max_passes: int = 50) -> tuple[Matrix, int]:
    entries, det, _ = steepest_2_flip_polish(
        [row[:] for row in initial.entries], max_passes=max_passes
    )
    return Matrix(n=initial.n, entries=entries), det


def stage_tabu(initial: Matrix, *, chains: int, iterations: int) -> tuple[Matrix, int]:
    best = initial
    best_det = abs(integer_determinant(initial.entries))
    for c in range(chains):
        res = sa_tabu(
            n=initial.n,
            iterations=iterations,
            t_start=0.5,
            t_end=0.005,
            seed=20000 + c,
            initial=initial,
            tabu_capacity=1024,
        )
        if res.abs_det > best_det:
            best_det = res.abs_det
            best = res.matrix
    return best, best_det


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def ratio(d: int) -> float:
    return d / EW_25


def run_pipeline(name: str, anchor: Matrix, anchor_det: int) -> dict:
    print(f"\n=== {name}: anchor |det|={anchor_det:.4e} ratio={ratio(anchor_det):.4f}")
    t0 = time.monotonic()

    sa_mat, sa_det = stage_sa(anchor, chains=4, iterations=200_000)
    print(f"  SA           |det|={sa_det:.4e}  ratio={ratio(sa_det):.4f}  ({time.monotonic()-t0:.1f}s)")

    g_mat, g_det = stage_greedy_single_flip(sa_mat)
    print(f"  greedy       |det|={g_det:.4e}  ratio={ratio(g_det):.4f}  ({time.monotonic()-t0:.1f}s)")

    p_mat, p_det = stage_polish_2flip(g_mat)
    print(f"  2-flip polish|det|={p_det:.4e}  ratio={ratio(p_det):.4f}  ({time.monotonic()-t0:.1f}s)")

    t_mat, t_det = stage_tabu(p_mat, chains=4, iterations=5000)
    print(f"  tabu         |det|={t_det:.4e}  ratio={ratio(t_det):.4f}  ({time.monotonic()-t0:.1f}s)")

    return {
        "name": name,
        "anchor_det": anchor_det,
        "anchor_ratio": ratio(anchor_det),
        "sa_det": sa_det, "sa_ratio": ratio(sa_det),
        "greedy_det": g_det, "greedy_ratio": ratio(g_det),
        "polish_det": p_det, "polish_ratio": ratio(p_det),
        "tabu_det": t_det, "tabu_ratio": ratio(t_det),
        "final_matrix": [list(r) for r in t_mat.entries],
        "walltime_s": time.monotonic() - t0,
    }


def main() -> None:
    print(f"EW(25) = 7 · 24^12 = {EW_25} ({EW_25:.4e})")
    print(f"Cycle 48 baseline: 1.435e17 (0.5612 EW)")

    print("\nBuilding anchor A (Paley I H24 bordered, greedy)...")
    anchor_a, anchor_a_det = make_anchor_a()
    print(f"  anchor A |det|={anchor_a_det:.4e}  ratio={ratio(anchor_a_det):.4f}")

    print("\nBuilding anchor B (Paley II H28 → H25, random 200K (3,3)-deletions)...")
    anchor_b, anchor_b_det, dels = make_anchor_b()
    print(f"  anchor B |det|={anchor_b_det:.4e}  ratio={ratio(anchor_b_det):.4f}")
    print(f"  rows deleted: {dels[0]}  cols deleted: {dels[1]}")

    out_a = run_pipeline("A_paley24_bordered", anchor_a, anchor_a_det)
    out_b = run_pipeline("B_paley28_deletion", anchor_b, anchor_b_det)

    out = {
        "ew_25": EW_25,
        "cycle_48_baseline_det": 143469891687546880,
        "cycle_48_baseline_ratio": 0.5612131387411521,
        "anchor_A": out_a,
        "anchor_B": out_b,
    }
    out_a_path = RESULTS / "H25_anchor_A_paley24_bordered.json"
    out_b_path = RESULTS / "H25_anchor_B_paley28_deletion.json"
    summary_path = RESULTS / "H25_cycle49_summary.json"

    out_a_path.write_text(json.dumps(out_a, indent=2))
    out_b_path.write_text(json.dumps(out_b, indent=2))
    summary_path.write_text(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or k.startswith("anchor")}, default=lambda o: o, indent=2))
    np.save(RESULTS / "H25_anchor_A_final.npy", np.array(out_a["final_matrix"], dtype=np.int8))
    np.save(RESULTS / "H25_anchor_B_final.npy", np.array(out_b["final_matrix"], dtype=np.int8))

    best_ratio = max(out_a["tabu_ratio"], out_b["tabu_ratio"])
    print(f"\n=== CYCLE 49 SUMMARY ===")
    print(f"  Anchor A final ratio: {out_a['tabu_ratio']:.4f}")
    print(f"  Anchor B final ratio: {out_b['tabu_ratio']:.4f}")
    print(f"  Cycle 48 baseline:    0.5612")
    print(f"  Best improvement: {best_ratio - 0.5612:+.4f}")


if __name__ == "__main__":
    main()
