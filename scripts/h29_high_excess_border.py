"""H29 via high-excess H28 bordering — Proposition 13 of Browne-Egan-Hegarty-
Ó Catháin 2021 survey "A Survey of the Hadamard Maximal Determinant Problem"
(arxiv 2104.06756).

Theory:
  Let H be a Hadamard matrix of order n−1 with excess e(H) = sum of all
  entries. Then M = [[1, 1...1], [-1...-1, H]] (single row of 1s, single
  column of −1s, corner 1) satisfies det(M) = det(H) · (1 + e(H)/n).

  For n=29: take Hadamard H28, maximize its excess via row/col sign-flips
  (which preserve Hadamardness), border. Theoretical max excess of H_n
  is n·sqrt(n) = 28·sqrt(28) ≈ 148.16.

  Survey claims ≥ 0.7·Barba achievable for all n ≤ 100 via this technique
  (citing Orrick-Solomon).

This implementation uses Paley II H28 (q=13 prime ≡ 1 mod 4) as the
starting Hadamard matrix and SA on (D1, D2) ∈ {±1}^28 to maximize
excess(D1 · H28 · D2). 12 SA trials × 500K iters consistently find
excess = 140 (≈ 94.6% of theoretical max), giving |det(M29)| ≈ 1.092e21
= 0.7947·Barba — a substantial improvement over the prior in-repo best
of 0.6088 (h29_outlier).

Output: results/H29_high_excess_border_r0.7947.json with verified matrix.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant, upper_bound


def maximize_excess_sa(
    H: np.ndarray,
    *,
    iters: int = 500_000,
    n_trials: int = 12,
    t_start: float = 30.0,
    t_end: float = 0.1,
    seed_base: int = 0,
) -> tuple[int, np.ndarray, np.ndarray]:
    """SA on (D1, D2) ∈ {±1}^n to maximize excess of D1·H·D2.

    Returns (best_excess, best_D1, best_D2). Sign-flips preserve Hadamardness
    of H, so the resulting D1·H·D2 is still Hadamard.
    """
    n = H.shape[0]

    def excess(D1, D2):
        return int(D1 @ H @ D2)

    log_ratio = math.log(t_end / t_start)
    best_e_overall = -(10**9)
    best_D1 = None
    best_D2 = None

    for trial in range(n_trials):
        rng = random.Random(seed_base + trial)
        D1 = np.array([1 if rng.random() < 0.5 else -1 for _ in range(n)], dtype=np.int64)
        D2 = np.array([1 if rng.random() < 0.5 else -1 for _ in range(n)], dtype=np.int64)
        e_cur = excess(D1, D2)
        e_best = e_cur
        D1b, D2b = D1.copy(), D2.copy()
        for step in range(iters):
            frac = step / max(1, iters - 1)
            t = t_start * math.exp(log_ratio * frac)
            if rng.random() < 0.5:
                idx = rng.randrange(n)
                D1[idx] = -D1[idx]
                new_e = excess(D1, D2)
                de = -(new_e - e_cur)
                if de <= 0 or rng.random() < math.exp(-de / max(t, 1e-9)):
                    e_cur = new_e
                    if e_cur > e_best:
                        e_best = e_cur
                        D1b, D2b = D1.copy(), D2.copy()
                else:
                    D1[idx] = -D1[idx]
            else:
                jdx = rng.randrange(n)
                D2[jdx] = -D2[jdx]
                new_e = excess(D1, D2)
                de = -(new_e - e_cur)
                if de <= 0 or rng.random() < math.exp(-de / max(t, 1e-9)):
                    e_cur = new_e
                    if e_cur > e_best:
                        e_best = e_cur
                        D1b, D2b = D1.copy(), D2.copy()
                else:
                    D2[jdx] = -D2[jdx]
        if e_best > best_e_overall:
            best_e_overall = e_best
            best_D1 = D1b
            best_D2 = D2b

    assert best_D1 is not None and best_D2 is not None
    return best_e_overall, best_D1, best_D2


def border_to_n_plus_1(H: np.ndarray) -> np.ndarray:
    """Build M = [[1, 1..1], [-1..-1^T, H]] of order n+1."""
    n = H.shape[0]
    M = np.zeros((n + 1, n + 1), dtype=np.int64)
    M[0, 0] = 1
    for j in range(1, n + 1):
        M[0, j] = 1
    for i in range(1, n + 1):
        M[i, 0] = -1
    M[1:, 1:] = H
    return M


def main() -> None:
    H28 = np.array(Matrix.paley_type_ii(28).entries, dtype=np.int64)
    assert (H28 @ H28.T == 28 * np.eye(28, dtype=np.int64)).all(), "Paley II H28 should be Hadamard"
    print(f"Paley II H28 verified Hadamard (initial excess = {int(H28.sum())})")

    print("\nMaximizing excess via SA on (D1, D2) sign flips...")
    best_e, D1, D2 = maximize_excess_sa(H28, iters=500_000, n_trials=12)
    H28_opt = D1[:, None] * H28 * D2[None, :]
    e_check = int(H28_opt.sum())
    assert e_check == best_e
    assert (H28_opt @ H28_opt.T == 28 * np.eye(28, dtype=np.int64)).all()
    print(f"Best excess: {best_e} (theoretical max ≈ 148.16)")

    M29 = border_to_n_plus_1(H28_opt)
    det = abs(integer_determinant(M29.tolist()))
    ub = upper_bound(29)
    ratio = det / ub
    print(f"\n|det(M29)| = {det}")
    print(f"Ratio to upper bound = {ratio:.4f}")

    out = {
        "order_id": "H29",
        "n": 29,
        "source": "Paley II H28 + SA-optimized sign-flip excess + Proposition 13 bordering",
        "survey_reference": "Browne-Egan-Hegarty-Ó Catháin 2021 (arxiv 2104.06756)",
        "h28_excess": best_e,
        "h28_excess_max_possible": 148,
        "abs_det": det,
        "ratio_to_upper_bound": ratio,
        "prior_best_in_repo": 836834213845327151104,
        "prior_ratio": 0.6088,
        "improvement_ratio": ratio - 0.6088,
        "verified_by_bareiss": True,
        "D1": D1.tolist(),
        "D2": D2.tolist(),
        "entries": M29.tolist(),
    }
    out_path = REPO / "results" / "H29_high_excess_border_r0.7947.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
