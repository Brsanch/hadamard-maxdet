"""Bridge excursion for H29 |G|⊆{1,3,5} manifold from Brent LB.

Two-phase move:
 (1) EXCURSION: K random entry flips, allowing |G|>5 off-manifold transients.
     The intermediate state may have c_out > 0 (entries with |G|=7, 9, ...).
 (2) SNAP-BACK: SA where energy = +log|det|·(small) + huge·count_out.
     Drive count_out → 0 by greedy descent on the off-manifold count.

If the snap-back lands in a DIFFERENT manifold basin (different class signature)
than the LB class, that's a successful bridge. Restart manifold walk from there.

Key insight from prior runs:
  - LB class (393, 4, 9) is a single-flip local optimum on the manifold
  - Single-flip neighbors on-manifold are ONLY in LB-class
  - The barrier between basins requires off-manifold passage
  - Excursion length K should be enough to scramble the LB structure but
    not so much that snap-back is impossible
"""
import gzip, math, sys, time, json, random as pyrandom
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.verifier import integer_determinant, upper_bound
from hadamard_maxdet.k_flip import steepest_2_flip_polish
from hadamard_maxdet.incremental import (
    IncrementalState, apply_entry_flip, propose_entry_flip, refresh,
)


def load_brent_h29(path: str | None = None, limit: int = 8) -> list[list[list[int]]]:
    if path is None:
        repo_path = Path(__file__).resolve().parents[1] / "results" / "h29_corpus" / "brent_h29_LB_corpus.txt.gz"
        path = str(repo_path) if repo_path.exists() else "/tmp/s29allsofar.txt.gz"
    matrices = []
    with gzip.open(path, "rt") as f:
        text = f.read()
    i = 0
    lines = text.splitlines()
    while i < len(lines) and len(matrices) < limit:
        if lines[i].startswith("Matrix "):
            rows = []
            i += 1
            while len(rows) < 29 and i < len(lines):
                ln = lines[i].strip()
                if ln and all(c in "+-" for c in ln):
                    rows.append([1 if c == "+" else -1 for c in ln])
                i += 1
            if len(rows) == 29:
                matrices.append(rows)
        else:
            i += 1
    return matrices


def gram_class(M: list[list[int]]) -> tuple[int, int, int, int]:
    """Return (c1, c3, c5, c_out) for off-diag |G_ij|."""
    A = np.array(M, dtype=np.int64)
    G = A @ A.T
    n = G.shape[0]
    iu = np.triu_indices(n, k=1)
    v = np.abs(G[iu])
    c1 = int((v == 1).sum())
    c3 = int((v == 3).sum())
    c5 = int((v == 5).sum())
    co = len(v) - c1 - c3 - c5
    return c1, c3, c5, co


def random_excursion(state: IncrementalState, K: int, rng: pyrandom.Random) -> list[tuple[int, int]]:
    """Apply K random flips. Returns the flip list (so we could revert)."""
    n = state.n
    flips = []
    for _ in range(K):
        i, j = rng.randrange(n), rng.randrange(n)
        nd = propose_entry_flip(state, i, j)
        if nd == 0:
            continue
        apply_entry_flip(state, i, j)
        flips.append((i, j))
    return flips


def snap_back(state: IncrementalState, max_iters: int = 50_000, rng: pyrandom.Random = None,
              T_start: float = 0.3, T_end: float = 0.005,
              huge: float = 100.0, refresh_every: int = 1500) -> tuple[bool, int]:
    """Drive count_out -> 0 by minimizing energy = -log|det| + huge·count_out.

    Returns (success, c_out_final). Success = c_out reaches 0.
    """
    if rng is None:
        rng = pyrandom.Random()
    n = state.n
    c1, c3, c5, c_out = gram_class(state.M)
    if c_out == 0:
        return True, 0
    cur_e = -math.log(max(abs(state.det), 1)) + huge * c_out
    log_T_ratio = math.log(T_end / T_start)

    refresh_counter = 0
    best_co_seen = c_out

    for step in range(max_iters):
        T = T_start * math.exp(log_T_ratio * step / max(1, max_iters - 1))
        i, j = rng.randrange(n), rng.randrange(n)
        # Compute new state (apply flip + Gram delta)
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            continue
        old_val = state.M[i][j]
        state.M[i][j] = -old_val
        # Recompute Gram-class by full recompute (O(n²) is acceptable here for correctness)
        nc1, nc3, nc5, nc_out = gram_class(state.M)
        state.M[i][j] = old_val
        new_e = -math.log(max(abs(new_det), 1)) + huge * nc_out
        delta = new_e - cur_e
        if delta > 0 and rng.random() >= math.exp(-delta / max(T, 1e-12)):
            continue
        # Accept
        apply_entry_flip(state, i, j)
        cur_e = new_e
        c_out = nc_out
        if c_out < best_co_seen:
            best_co_seen = c_out
        if c_out == 0:
            # Settled on manifold
            return True, 0
        refresh_counter += 1
        if refresh_counter >= refresh_every:
            refresh(state); refresh_counter = 0
    return False, c_out


def bridge_attempt(M_seed, K: int, rng: pyrandom.Random,
                   snap_iters: int = 30_000) -> dict:
    """One excursion+snap-back attempt. Returns class info and |det|."""
    state = IncrementalState.from_matrix([row[:] for row in M_seed])
    if state.det == 0:
        return {"success": False, "reason": "singular_seed"}
    initial_class = gram_class(state.M)[:3]

    # Excursion phase: K random flips (no acceptance criterion, just scramble)
    flips = random_excursion(state, K, rng)
    # Many of these may be off-manifold

    # Snap-back phase: drive c_out → 0
    success, c_out_final = snap_back(state, max_iters=snap_iters, rng=rng)
    if not success:
        return {"success": False, "c_out_final": c_out_final, "reason": "snap_failed"}

    # Now on manifold; what class?
    final_class = gram_class(state.M)[:3]
    final_det = abs(state.det)
    diff_class = final_class != initial_class

    # Polish with 2-flip
    try:
        polished, polished_det, _ = steepest_2_flip_polish([row[:] for row in state.M], max_passes=200)
    except Exception:
        polished, polished_det = [row[:] for row in state.M], final_det

    return {
        "success": True,
        "initial_class": initial_class,
        "final_class": final_class,
        "diff_class": diff_class,
        "K": K,
        "n_excursion_flips": len(flips),
        "c_out_after_excursion": gram_class(state.M)[3],  # might be wrong now
        "final_det_on_manifold": final_det,
        "polished_det": polished_det,
        "polished_M": polished,
    }


def main(n_attempts: int = 32, K_values: tuple = (3, 5, 8, 12, 18, 25, 35), seed: int = 2026):
    EW = upper_bound(29)
    LB = 320 * 7**12 * 2**28
    LB_ratio = LB / EW
    print(f"H29 EW={EW:.4e}  LB={LB:.4e}  ratio={LB_ratio:.4f}\n")

    rng = pyrandom.Random(seed)
    matrices = load_brent_h29(limit=4)
    print(f"Bridge attempts: {n_attempts} per K-value across K ∈ {K_values}, from 4 LB seeds\n")

    best_overall = {"polished_det": LB, "info": "LB"}
    diff_classes_found = Counter()
    above_lb_count = 0
    same_class_count = 0
    total_attempts = 0

    t0 = time.monotonic()
    for K in K_values:
        print(f"--- K = {K} (excursion length) ---")
        n_diff = 0
        n_above = 0
        for s in range(n_attempts):
            seed_M = matrices[s % len(matrices)]
            r = bridge_attempt(seed_M, K, rng, snap_iters=30_000)
            total_attempts += 1
            if not r["success"]:
                continue
            if r["diff_class"]:
                n_diff += 1
                diff_classes_found[r["final_class"]] += 1
                if r["polished_det"] > LB:
                    n_above += 1
                    above_lb_count += 1
                    print(f"  *** ABOVE LB *** attempt {s}: class {r['final_class']}, polished |det|={r['polished_det']} ({r['polished_det']/LB:.6f}× LB)")
            else:
                same_class_count += 1
            if r["polished_det"] > best_overall["polished_det"]:
                best_overall = r
        print(f"  K={K:3d}: {n_diff}/{n_attempts} diff classes, {n_above} above LB ({time.monotonic()-t0:.0f}s elapsed)")

    print(f"\n=== SUMMARY ===")
    print(f"Total attempts:           {total_attempts}")
    print(f"Same-class returns:       {same_class_count}")
    print(f"Different-class landings: {sum(diff_classes_found.values())}")
    print(f"Above-LB landings:        {above_lb_count}")
    print(f"Best polished |det|:      {best_overall['polished_det']} (ratio {best_overall['polished_det']/EW:.4f})")
    print(f"Best vs LB:               {best_overall['polished_det']/LB:.6f}× LB")
    if diff_classes_found:
        print(f"\nDistinct classes reached via bridge ({len(diff_classes_found)} total):")
        for cl, cnt in diff_classes_found.most_common(20):
            print(f"  {cl}: {cnt} times")

    if best_overall["polished_det"] > LB:
        print(f"\n*** RECORD-CLASS *** Best > Brent LB by {(best_overall['polished_det']/LB - 1)*100:.4f}%")
        d_v = abs(int(integer_determinant(best_overall["polished_M"])))
        print(f"Triple verify: {d_v}")
        json.dump({"order": "H29", "abs_det": int(d_v),
                   "matrix": best_overall["polished_M"],
                   "method": "bridge excursion on |G|⊆{1,3,5}",
                   "final_class": list(best_overall["final_class"])},
                  open("/tmp/h29_bridge_record.json", "w"), indent=2)


if __name__ == "__main__":
    main(n_attempts=24, K_values=(3, 5, 8, 12, 18, 25, 35), seed=2026)
