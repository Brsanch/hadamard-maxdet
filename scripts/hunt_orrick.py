"""Orrick H23 hunt — Strategy A (skew-biased SA) + Strategy B (Ehlich block).

Targets: BTC current best |det|=2,626,567,700,217,856 (0.6702·Barba). Orrick
record |det|≈2.778e15 (0.7091·Barba). Hypothesis: Orrick lives in the
4-valued Gram support family with a multiset more extreme than BTC's
(11,209,11,22).

This script:

  1. Strategy A — runs ``sa_skew_biased`` from the BTC seed for several
     λ values (default {0.001, 0.01, 0.1, 1.0}, μ=100).
  2. Strategy B — enumerates Ehlich block partitions of 23, picks the
     top-K by predicted |det|, attempts construction for each via
     ``construct_from_partition``.
  3. Verifies every candidate's |det| with exact Bareiss.
  4. Saves the best matrix to ``results/H23_orrick_hunt_best.json``.
  5. Prints a summary table; flags loudly if any candidate beats BTC,
     and re-verifies twice if it beats Orrick (2.778e15).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.matrix import Matrix  # noqa: E402
from hadamard_maxdet.orrick_hunt import (  # noqa: E402
    construct_from_partition,
    enumerate_ehlich_partitions,
    gram_offdiag_counts,
    sa_skew_biased,
)
from hadamard_maxdet.verifier import integer_determinant, upper_bound  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_FILE = REPO_ROOT / "results" / "H23_rich_ansatz_best.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "H23_orrick_hunt_best.json"

ORRICK_TARGET = 2_778_000_000_000_000  # 2.778e15, Orrick's published H23 record (approx)
BTC_BEST = 2_626_567_700_217_856  # current in-repo BTC best


def _fmt_multiset(c: dict[int, int]) -> str:
    return "(" + ", ".join(f"{v:+d}: {n}" for v, n in sorted(c.items())) + ")"


def _bareiss_verify(entries: list[list[int]]) -> int:
    return abs(integer_determinant(entries))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE,
                    help="JSON containing the BTC seed at H23.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--budget-per-lambda", type=float, default=300.0,
                    help="Wall budget (s) per λ for Strategy A. Default 300s.")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.001, 0.01, 0.1, 1.0],
                    help="λ values for Strategy A skew bias.")
    ap.add_argument("--mu-support", type=float, default=100.0,
                    help="μ penalty per off-support Gram entry.")
    ap.add_argument("--strategy-b-top-k", type=int, default=5,
                    help="Number of top Ehlich partitions to attempt to "
                         "construct in Strategy B.")
    ap.add_argument("--strategy-b-budget", type=float, default=60.0,
                    help="Wall budget (s) per partition construction in B.")
    ap.add_argument("--strategy-a-iters", type=int, default=2_000_000,
                    help="Iteration cap per λ (wall budget cuts earlier).")
    ap.add_argument("--strategy-b-iters", type=int, default=200_000,
                    help="Iteration cap per construction attempt.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-strategy-a", action="store_true")
    ap.add_argument("--skip-strategy-b", action="store_true")
    args = ap.parse_args()

    barba = upper_bound(23)
    print("=" * 78)
    print("Orrick H23 hunt")
    print("=" * 78)
    print(f"  Barba(23)  = {barba:,d}")
    print(f"  BTC best   = {BTC_BEST:,d}  ({BTC_BEST/barba:.4f} Barba)")
    print(f"  Orrick≈    = {ORRICK_TARGET:,d}  ({ORRICK_TARGET/barba:.4f} Barba)")
    print()

    # --- Load seed --------------------------------------------------------
    data = json.loads(args.seed_file.read_text())
    seed_entries = [[int(x) for x in row] for row in data["entries"]]
    seed_n = len(seed_entries)
    if seed_n != 23:
        raise SystemExit(f"seed n={seed_n}; expected 23")
    seed_abs = _bareiss_verify(seed_entries)
    if seed_abs != BTC_BEST:
        print(f"  WARNING: seed |det|={seed_abs:,d} ≠ BTC={BTC_BEST:,d}")
    seed_counts = gram_offdiag_counts(seed_entries)
    print(f"  seed multiset: {_fmt_multiset(seed_counts)}")
    print()

    seed_matrix = Matrix(n=23, entries=[row[:] for row in seed_entries])

    candidates: list[dict] = [
        {
            "label": "seed (BTC)",
            "abs_det": seed_abs,
            "barba_ratio": seed_abs / barba,
            "multiset": dict(sorted(seed_counts.items())),
            "skew": (seed_counts.get(-1, 0) + seed_counts.get(-3, 0))
                    - (seed_counts.get(+1, 0) + seed_counts.get(+3, 0)),
            "support_penalty": 0,
            "method": "BTC_seed",
            "entries": seed_entries,
        }
    ]

    # --- Strategy A -------------------------------------------------------
    if not args.skip_strategy_a:
        print("-" * 78)
        print("Strategy A — skew-biased SA from BTC seed")
        print("-" * 78)
        for k, lam in enumerate(args.lambdas):
            print(f"\n  λ={lam:g}, μ={args.mu_support:g}, "
                  f"budget={args.budget_per_lambda:.0f}s")
            t0 = time.monotonic()
            res = sa_skew_biased(
                seed_matrix.copy(),
                iterations=args.strategy_a_iters,
                lambda_skew=lam,
                mu_support=args.mu_support,
                t_start=0.05,
                t_end=0.0005,
                seed=args.seed + k,
                refresh_every=2000,
                wall_time_budget=args.budget_per_lambda,
            )
            walltime = time.monotonic() - t0
            counts = gram_offdiag_counts(res.matrix.entries)
            verified = _bareiss_verify(res.matrix.entries)
            assert verified == res.abs_det
            ratio = res.abs_det / barba
            print(f"    walltime={walltime:.1f}s  iters={res.iterations:,}  "
                  f"accepted={res.accepted:,}")
            print(f"    |det|={res.abs_det:,d}  ({ratio:.4f} Barba, "
                  f"{res.abs_det/seed_abs:.4f}·BTC)")
            print(f"    skew={res.skew}  support_penalty={res.support_penalty}")
            print(f"    multiset: {_fmt_multiset(counts)}")
            candidates.append({
                "label": f"strategy_A λ={lam:g}",
                "abs_det": res.abs_det,
                "barba_ratio": ratio,
                "multiset": dict(sorted(counts.items())),
                "skew": res.skew,
                "support_penalty": res.support_penalty,
                "method": f"sa_skew_biased(λ={lam:g}, μ={args.mu_support:g})",
                "iterations": res.iterations,
                "accepted": res.accepted,
                "walltime_seconds": walltime,
                "lambda_skew": lam,
                "mu_support": args.mu_support,
                "entries": res.matrix.entries,
                "history_top_size": len(res.history_top),
            })

    # --- Strategy B -------------------------------------------------------
    if not args.skip_strategy_b:
        print()
        print("-" * 78)
        print("Strategy B — Ehlich block-partition enumeration + construction")
        print("-" * 78)
        partitions = enumerate_ehlich_partitions(23, top_k=args.strategy_b_top_k)
        print(f"\n  Top {len(partitions)} (3,-1)-ansatz partitions of 23:")
        for p in partitions:
            print(f"    sizes={p.sizes}  predicted |det|={p.predicted_abs_det:.3e}  "
                  f"({p.predicted_abs_det/barba:.4f} Barba)")

        for k, p in enumerate(partitions):
            print(f"\n  [B/{k+1}] constructing for sizes={p.sizes} "
                  f"(predicted {p.predicted_abs_det/barba:.4f} Barba)")
            t0 = time.monotonic()
            cres = construct_from_partition(
                p,
                iterations=args.strategy_b_iters,
                seed=args.seed + 100 + k,
                t_start=4.0,
                t_end=0.05,
                wall_time_budget=args.strategy_b_budget,
            )
            walltime = time.monotonic() - t0
            counts = gram_offdiag_counts(cres.matrix.entries)
            verified = _bareiss_verify(cres.matrix.entries)
            assert verified == cres.abs_det
            ratio = cres.abs_det / barba
            success = cres.matched_block_structure
            tag = "MATCH" if success else "partial"
            print(f"    walltime={walltime:.1f}s  iters={cres.iterations:,}")
            print(f"    Gram loss = {cres.final_gram_loss}  ({tag})")
            print(f"    |det|={cres.abs_det:,d}  ({ratio:.4f} Barba, "
                  f"{cres.abs_det/seed_abs:.4f}·BTC)")
            print(f"    multiset: {_fmt_multiset(counts)}")
            cand_skew = (counts.get(-1, 0) + counts.get(-3, 0)) \
                        - (counts.get(+1, 0) + counts.get(+3, 0))
            cand_pen = sum(c for v, c in counts.items()
                           if v not in {-3, -1, 1, 3})
            candidates.append({
                "label": f"strategy_B sizes={p.sizes}",
                "abs_det": cres.abs_det,
                "barba_ratio": ratio,
                "multiset": dict(sorted(counts.items())),
                "skew": cand_skew,
                "support_penalty": cand_pen,
                "method": f"construct_from_partition(sizes={p.sizes}, intra=+3, cross=-1)",
                "predicted_abs_det": p.predicted_abs_det,
                "predicted_barba_ratio": p.predicted_abs_det / barba,
                "final_gram_loss": cres.final_gram_loss,
                "matched_block_structure": cres.matched_block_structure,
                "iterations": cres.iterations,
                "walltime_seconds": walltime,
                "entries": cres.matrix.entries,
            })

    # --- Summary ----------------------------------------------------------
    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    candidates.sort(key=lambda c: -c["abs_det"])
    print(f"  {'label':<36} {'|det|':>22} {'ratio':>8} {'vs BTC':>10}")
    print("  " + "-" * 80)
    for c in candidates:
        ratio = c["abs_det"] / barba
        rel = c["abs_det"] / seed_abs
        print(f"  {c['label']:<36} {c['abs_det']:>22,d} {ratio:>8.4f} {rel:>10.4f}")

    best = candidates[0]
    print()
    print(f"  BEST: {best['label']}  |det|={best['abs_det']:,d}  "
          f"({best['barba_ratio']:.4f} Barba)")
    if best["abs_det"] > BTC_BEST:
        print()
        print(("*" * 78))
        print(f"  ABOVE BTC by {best['abs_det'] - BTC_BEST:,d} "
              f"(+{(best['abs_det']/BTC_BEST - 1)*100:.2f}%)")
        # Re-verify twice.
        v1 = _bareiss_verify(best["entries"])
        v2 = _bareiss_verify([row[:] for row in best["entries"]])
        print(f"  re-verify(1): {v1:,d}  re-verify(2): {v2:,d}")
        assert v1 == v2 == best["abs_det"]
        if best["abs_det"] >= ORRICK_TARGET:
            print()
            print("  *** ABOVE ORRICK TARGET ***")
            print(f"      Orrick: {ORRICK_TARGET:,d}")
            print(f"      ours:   {best['abs_det']:,d} "
                  f"(+{(best['abs_det']/ORRICK_TARGET - 1)*100:.2f}%)")
        print(("*" * 78))

    # --- Save -------------------------------------------------------------
    payload = {
        "order_id": "H23",
        "n": 23,
        "abs_det": best["abs_det"],
        "barba_ratio": best["barba_ratio"],
        "method": f"H23_orrick_hunt::{best['label']}",
        "btc_baseline": BTC_BEST,
        "orrick_target": ORRICK_TARGET,
        "all_candidates_summary": [
            {k: v for k, v in c.items() if k != "entries"}
            for c in candidates
        ],
        "entries": best["entries"],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"  Wrote {args.output}")
    print(f"  Best vs Orrick: {best['abs_det']/ORRICK_TARGET:.4f}× target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
