"""Funnel search benchmark on H23 (15 minute budget).

Phase A — roll uniform random ±1 matrices, filter by Gram loudness, refine
the survivors with short rank-1 SA, accumulate distinct basins/fingerprints.
Switch to Phase B when refinement stops finding new basins.

Phase B — Gram-conserving pair flips on the running best matrix, mixed
50/50 with standard entry flips. Standard −log|det| Metropolis acceptance.

The Phase B initial seed is the BTC matrix from results/H23_rich_ansatz_best.json
(in-repo best at 2,626,567,700,217,856 = 0.6704·Barba) so Phase A only has
to *exceed* this to claim a new in-repo lower bound.

Outputs:
  - results/H23_funnel_best.json  (best matrix, re-verified)
  - results/funnel_h23_diagnostics.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.funnel import FunnelDiagnostics, funnel_search  # noqa: E402
from hadamard_maxdet.io import write_submission  # noqa: E402
from hadamard_maxdet.matrix import Matrix  # noqa: E402
from hadamard_maxdet.verifier import integer_determinant  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
BTC_SEED_PATH = REPO_ROOT / "results" / "H23_rich_ansatz_best.json"
OUT_BEST = REPO_ROOT / "results" / "H23_funnel_best.json"
OUT_DIAG = REPO_ROOT / "results" / "funnel_h23_diagnostics.json"


def load_btc_seed() -> Matrix:
    data = json.loads(BTC_SEED_PATH.read_text())
    entries = data["entries"]
    return Matrix(n=len(entries), entries=entries)


def _print_progress(phase: str, diag: FunnelDiagnostics, current_best_det: int) -> None:
    barba = 3.919e15
    ratio = current_best_det / barba
    if phase == "phase_a":
        print(
            f"  [phase_a +{diag.phase_a_seconds + (time.monotonic() - _print_progress._start - diag.phase_a_seconds):6.1f}s] "
            f"tried={diag.candidates_tried:6d}  "
            f"passed={diag.candidates_passed_filter:5d}  "
            f"basins={diag.distinct_basins:4d}  "
            f"fps={diag.distinct_fingerprints:4d}  "
            f"best={current_best_det:>20d}  "
            f"ratio={ratio:.4f}",
            flush=True,
        )
    else:
        b_ratio = diag.phase_b_pair_flip_accepted / max(1, diag.phase_b_pair_flip_proposed)
        e_ratio = diag.phase_b_entry_flip_accepted / max(1, diag.phase_b_entry_flip_proposed)
        print(
            f"  [phase_b iters={diag.phase_b_iterations:>9d}] "
            f"acc={diag.phase_b_accepted:>8d}  "
            f"pair_acc={b_ratio:.3f}  "
            f"entry_acc={e_ratio:.3f}  "
            f"best={current_best_det:>20d}  "
            f"ratio={ratio:.4f}",
            flush=True,
        )


_print_progress._start = 0.0


def main() -> int:
    n = 23
    budget = 900  # 15 minutes
    seed = 20260424
    barba = 3.919e15
    in_repo_best = 2_626_567_700_217_856  # BTC seed
    orrick = 2.778e15

    print("=" * 78)
    print("Funnel search on H23 (15-minute budget)")
    print("=" * 78)
    print(f"  n = {n}")
    print(f"  Barba bound  ≈ {barba:.3e}")
    print(f"  In-repo best = {in_repo_best:,d}  (0.6704 · Barba)")
    print(f"  Orrick bound = {orrick:.3e}  (0.7091 · Barba)")
    print()

    btc_seed = load_btc_seed()
    seed_det = abs(integer_determinant(btc_seed.entries))
    print(f"  BTC seed verified: |det| = {seed_det:,d}")
    assert seed_det == in_repo_best, "BTC seed |det| mismatch with documented value"
    print()

    print("Phase A starting...")
    print("-" * 78)
    _print_progress._start = time.monotonic()

    best, diag = funnel_search(
        n=n,
        total_budget_seconds=budget,
        seed=seed,
        initial_best=btc_seed,
        max_support_size=8,
        require_negative_bias=True,
        refine_iterations=50_000,
        refine_t_start=1.0,
        refine_t_end=0.01,
        phase_a_min_seconds=60.0,
        # Hard cap so Phase B still gets a budget even if the strict
        # K=8 loudness filter admits zero candidates at n=23 (empirically
        # the case: random ±1 matrices at n=23 have support 11-17).
        phase_a_max_seconds=300.0,
        no_new_window=50,
        phase_b_pair_fraction=0.5,
        phase_b_t_start=0.5,
        phase_b_t_end=0.005,
        phase_b_no_improve_iters=200_000,
        progress_callback=_print_progress,
    )

    print()
    print("=" * 78)
    print("Phase A summary")
    print("=" * 78)
    print(f"  candidates_tried             = {diag.candidates_tried}")
    print(f"  candidates_passed_filter     = {diag.candidates_passed_filter}")
    pass_rate = (
        diag.candidates_passed_filter / max(1, diag.candidates_tried) * 100
    )
    print(f"  pass rate                    = {pass_rate:.2f}%")
    print(f"  distinct basins (|det|)      = {diag.distinct_basins}")
    print(f"  distinct fingerprints        = {diag.distinct_fingerprints}")
    print(f"  phase A wallclock            = {diag.phase_a_seconds:.1f} s")
    print(f"  phase A best |det|           = {diag.phase_a_best_det:,d}")
    print(f"  phase A best Barba ratio     = {diag.phase_a_best_det / barba:.4f}")
    if diag.switch_fired_at_seconds is not None:
        print(f"  switch fired at              = {diag.switch_fired_at_seconds:.1f} s")
        print(f"  switch fired after candidates= {diag.switch_fired_at_candidates}")
        print(f"  switch reason                = {diag.switch_reason}")
    else:
        print(f"  switch did NOT fire          (reason: {diag.switch_reason})")

    print()
    print("=" * 78)
    print("Phase B summary")
    print("=" * 78)
    print(f"  phase B wallclock            = {diag.phase_b_seconds:.1f} s")
    print(f"  phase B iterations           = {diag.phase_b_iterations}")
    print(f"  phase B accepted             = {diag.phase_b_accepted}")
    print(
        f"  pair flips proposed/accepted = "
        f"{diag.phase_b_pair_flip_proposed} / {diag.phase_b_pair_flip_accepted}"
    )
    print(
        f"  entry flips proposed/accepted= "
        f"{diag.phase_b_entry_flip_proposed} / {diag.phase_b_entry_flip_accepted}"
    )
    print(f"  phase B best |det|           = {diag.phase_b_best_det:,d}")
    print(f"  phase B improved over A      = {diag.phase_b_improved_over_a}")

    print()
    print("=" * 78)
    print("Final")
    print("=" * 78)
    final_det = abs(integer_determinant(best.entries))
    assert final_det == diag.final_best_det, (
        f"final det mismatch: re-verify {final_det} vs reported {diag.final_best_det}"
    )
    print(f"  final |det|                  = {final_det:,d}")
    print(f"  Barba ratio                  = {final_det / barba:.4f}")
    print(f"  vs in-repo best              = {final_det / in_repo_best:.4f}× "
          f"({'NEW HIGH' if final_det > in_repo_best else 'no improvement'})")
    print(f"  vs Orrick lower bound        = {final_det / orrick:.4f}× "
          f"({'BEATS' if final_det >= orrick else 'short of'} Orrick)")

    # Re-verify and save (independent Bareiss as required by validation gate).
    print()
    print(f"Re-verifying with exact integer_determinant... ", end="")
    re_det = abs(integer_determinant(best.entries))
    assert re_det == final_det
    print(f"OK ({re_det:,d})")

    if final_det > in_repo_best:
        print("[!] New in-repo lower bound on H23 — write up in findings.md")

    write_submission(best, "H23", OUT_BEST)
    print(f"  best matrix saved to:        {OUT_BEST}")

    OUT_DIAG.write_text(json.dumps(diag.to_dict(), indent=2) + "\n")
    print(f"  diagnostics saved to:        {OUT_DIAG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
