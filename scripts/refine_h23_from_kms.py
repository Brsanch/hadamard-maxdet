"""Build the Prop-28 (KMS-style) seed at n = 23 and refine it with SA.

Pipeline
--------
1. Construct the Proposition-28 matrix at q = 5 (n = 4q + 3 = 23).
2. Report its |det|, Gram fingerprint, and ratio to Barba.
3. Multi-start SA: from the seed and 3 small perturbations of it, run
   ``sa_incremental`` for a configurable wall-clock budget.
4. Track and exact-verify (Bareiss) the best |det| reached.
5. Save the best matrix to ``results/H23_from_kms_best.json``.
6. Print deltas vs (a) prior in-repo record 2,626,567,700,217,856 and
   (b) Orrick's published lower bound 0.7091 · Barba ≈ 2.778e15.

Caveat: at q = 5 the Paley core is symmetric (q ≡ 1 mod 4), so Prop 28
does **not** match its own formula prediction. The seed |det| is far below
the formula's claim. See ``src/hadamard_maxdet/kms_construction.py``.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.incremental import sa_incremental  # noqa: E402
from hadamard_maxdet.io import write_submission  # noqa: E402
from hadamard_maxdet.kms_construction import (  # noqa: E402
    is_skew_paley,
    prop28_det_predicted,
    prop28_matrix,
)
from hadamard_maxdet.matrix import Matrix  # noqa: E402
from hadamard_maxdet.telemetry import gram_offdiag_report  # noqa: E402
from hadamard_maxdet.verifier import integer_determinant, verify  # noqa: E402


PRIOR_BEST = 2_626_567_700_217_856
BARBA_N23 = math.sqrt(45) * 22 ** 11  # ≈ 3.91973e15
ORRICK_TARGET = 0.7091 * BARBA_N23     # ≈ 2.778e15


def perturb(seed: Matrix, rng: random.Random, k: int) -> Matrix:
    """Return a copy of ``seed`` with ``k`` random entry flips."""
    n = seed.n
    rows = [row[:] for row in seed.entries]
    flipped: set[tuple[int, int]] = set()
    while len(flipped) < k:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if (i, j) in flipped:
            continue
        flipped.add((i, j))
        rows[i][j] = -rows[i][j]
    return Matrix(n=n, entries=rows)


def run_one_start(
    *,
    label: str,
    initial: Matrix,
    iterations: int,
    seed: int,
    t_start: float,
    t_end: float,
) -> tuple[int, Matrix, float]:
    """Run sa_incremental from the given initial; return (|det|, matrix, walltime)."""
    print(f"  [{label}] running {iterations} iters, t∈[{t_start},{t_end}]...")
    t0 = time.monotonic()
    result = sa_incremental(
        n=initial.n,
        iterations=iterations,
        t_start=t_start,
        t_end=t_end,
        seed=seed,
        initial=initial,
        refresh_every=2_000,
    )
    walltime = time.monotonic() - t0
    print(
        f"  [{label}] |det| = {result.abs_det:,d}  "
        f"({result.abs_det / BARBA_N23:.4f} · Barba)  "
        f"in {walltime:.1f}s, accepted {result.accepted}/{result.iterations}"
    )
    return result.abs_det, result.matrix, walltime


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=240.0,
                    help="Wall-clock budget per start in seconds (default 240).")
    ap.add_argument("--iters-per-start", type=int, default=300_000,
                    help="SA iterations per start (default 300_000).")
    ap.add_argument("--starts", type=int, default=4,
                    help="Number of multi-starts (1 pure seed + N-1 perturbed).")
    ap.add_argument("--perturb-flips", type=int, default=5,
                    help="Random entry flips applied per perturbed start (default 5).")
    ap.add_argument("--t-start", type=float, default=0.5)
    ap.add_argument("--t-end", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=20260424)
    ap.add_argument("--out", type=Path,
                    default=Path("results/H23_from_kms_best.json"))
    args = ap.parse_args()

    # ----- 1. Build the Prop 28 seed at q = 5 -----
    q = 5
    seed_matrix = prop28_matrix(q)
    n = seed_matrix.n
    assert n == 23, f"Prop 28 at q=5 should give n=23, got n={n}"

    seed_det = abs(integer_determinant(seed_matrix.entries))
    seed_ratio = seed_det / BARBA_N23
    seed_predicted = prop28_det_predicted(q)
    pred_ratio = seed_predicted / BARBA_N23

    print(f"=== Prop 28 (KMS-style) seed at n = 23, q = {q} ===")
    print(f"  is_skew_paley(q={q}) = {is_skew_paley(q)} "
          f"(formula valid iff True)")
    print(f"  seed |det| = {seed_det:,d}  ({seed_ratio:.4f} · Barba)")
    print(f"  formula predicts |det| = {seed_predicted:,d}  "
          f"({pred_ratio:.4f} · Barba)")
    if seed_det == seed_predicted:
        print("  → formula matches actual (skew-Paley regime).")
    else:
        diff_pct = 100 * (1 - seed_det / seed_predicted)
        print(
            f"  → DISCREPANCY: actual is {diff_pct:.2f}% below formula. "
            "This is the q ≡ 1 (mod 4) regime where the Paley core is "
            "symmetric and the proposition's proof does not go through."
        )

    # Gram fingerprint of the seed
    fp = gram_offdiag_report(seed_matrix.entries)
    print(f"  Gram off-diag support: {fp['gram_offdiag_support']}")
    print(f"  Gram off-diag counts:  {fp['gram_offdiag_counts']}")
    print(f"  sv ratio (max/min) = {fp['sv_ratio']:.4f}")
    print()

    # ----- 2. Multi-start SA refinement -----
    print(f"=== Multi-start SA refinement ===")
    print(f"  budget per start: {args.budget:.1f}s, "
          f"iters per start: {args.iters_per_start:,}, "
          f"starts: {args.starts}, perturb flips: {args.perturb_flips}")
    print()

    rng = random.Random(args.seed)
    best_det = seed_det
    best_matrix = seed_matrix
    best_origin = "prop28_seed"

    overall_t0 = time.monotonic()
    for s in range(args.starts):
        if s == 0:
            initial = seed_matrix
            label = "start_0_pure"
        else:
            initial = perturb(seed_matrix, rng, args.perturb_flips)
            label = f"start_{s}_perturb{args.perturb_flips}"
        # Pick a per-start RNG seed for reproducibility within the run.
        sa_seed = rng.randrange(2 ** 31 - 1)

        det, matrix, walltime = run_one_start(
            label=label,
            initial=initial,
            iterations=args.iters_per_start,
            seed=sa_seed,
            t_start=args.t_start,
            t_end=args.t_end,
        )
        if det > best_det:
            best_det = det
            best_matrix = matrix
            best_origin = label
            print(f"  ** new best: {best_det:,d} from {label}")

        if time.monotonic() - overall_t0 > args.budget * args.starts:
            print("  (overall budget exhausted)")
            break

    overall_walltime = time.monotonic() - overall_t0
    print()
    print(f"=== Summary (total wall-clock {overall_walltime:.1f}s) ===")
    print(f"  refined best |det| = {best_det:,d}  "
          f"({best_det / BARBA_N23:.4f} · Barba)")
    print(f"  origin            = {best_origin}")

    # ----- 3. Re-verify with Bareiss -----
    bareiss_det = abs(integer_determinant(best_matrix.entries))
    if bareiss_det != best_det:
        print(f"  WARNING: Bareiss re-verify gives {bareiss_det} ≠ {best_det}")
    else:
        print(f"  Bareiss re-verify: OK")

    # Verify ±1.
    v = verify(best_matrix.entries)
    if not v.valid:
        print(f"  ERROR: best matrix failed verify(): {v.error}")
        return 1

    # ----- 4. Compare to milestones -----
    print()
    print(f"=== Comparison to milestones ===")
    delta_prior = best_det - PRIOR_BEST
    delta_orrick = best_det - int(ORRICK_TARGET)
    print(f"  prior in-repo best : {PRIOR_BEST:>22,d}  "
          f"(Δ = {delta_prior:+,d}, "
          f"{'BEAT' if delta_prior > 0 else 'below'})")
    print(f"  Orrick 0.7091·Barba: {int(ORRICK_TARGET):>22,d}  "
          f"(Δ = {delta_orrick:+,d}, "
          f"{'BEAT' if delta_orrick > 0 else 'below'})")
    print(f"  Barba upper bound  : {int(BARBA_N23):>22,d}  "
          f"(ratio {best_det / BARBA_N23:.4f})")

    if delta_orrick > 0:
        print()
        print("  *** NEW IN-REPO RECORD ABOVE ORRICK'S 0.7091 ***")
        print("  Independent re-verification via Bareiss completed above.")

    # ----- 5. Save best matrix -----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(best_matrix, "H23", args.out)
    print()
    print(f"  best matrix written to {args.out}")

    return 0 if delta_prior >= 0 else 0  # informational, not a hard failure


if __name__ == "__main__":
    raise SystemExit(main())
