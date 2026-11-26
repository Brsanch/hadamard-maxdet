"""H23 attack: three diagnostic-driven SA strategies seeded from BTC best.

Loads ``results/H23_rich_ansatz_best.json`` (the bordered two-circulant SA
result; |det| = 2,626,567,700,217,856 ≈ 0.6704·Barba) and runs three
180-second SA strategies starting from that matrix:

  (i)   Standard ``sa_incremental`` (rank-1 incremental |det| with -log|det|
        energy and uniform random entry flips). Control.
  (ii)  ``sa_gram_energy`` — SA over E_gram with O(n) delta. Move A.
  (iii) ``sa_targeted`` — biased proposer attacking max-Gram-pair, with
        standard -log|det| energy via incremental adjugate. Move C.

For each, logs best |det| reached, best E_gram, and the Gram fingerprint
of the final best matrix. Cross-verifies any beat with exact Bareiss
``integer_determinant``. Saves the overall best to
``results/H23_attack_best.json``.

Prints a final table comparing all three to the seed and to Orrick's
0.7091·Barba target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.gram_energy import (  # noqa: E402
    gram_energy,
    gram_offdiag_matrix,
    sa_gram_energy,
    sa_targeted,
)
from hadamard_maxdet.matrix import Matrix  # noqa: E402
from hadamard_maxdet.verifier import integer_determinant, upper_bound  # noqa: E402


def _gram_offdiag_counter(M: list[list[int]]) -> dict[int, int]:
    n = len(M)
    G = gram_offdiag_matrix(M)
    c: Counter = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            c[G[i][j]] += 1
    return dict(sorted(c.items()))


def _fmt_gram(fp: dict[int, int]) -> str:
    return ", ".join(f"{k:+d}: {v}" for k, v in sorted(fp.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "H23_rich_ansatz_best.json",
    )
    ap.add_argument("--budget-seconds", type=float, default=180.0,
                    help="Wall-clock budget per strategy (default 180s).")
    ap.add_argument("--output",
                    type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "H23_attack_best.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load the seed.
    data = json.loads(args.seed_file.read_text())
    n = int(data["n"])
    if n != 23:
        raise SystemExit(f"seed file has n={n}; expected 23")
    seed_entries = [[int(x) for x in row] for row in data["entries"]]
    seed_det = integer_determinant(seed_entries)
    seed_abs = abs(seed_det)
    expected_abs = 2_626_567_700_217_856
    if seed_abs != expected_abs:
        raise SystemExit(
            f"seed |det| mismatch: file says {seed_abs}, expected {expected_abs}"
        )
    seed_e_gram = gram_energy(seed_entries)
    seed_fp = _gram_offdiag_counter(seed_entries)

    barba = upper_bound(23)  # ≈ 3.919e15
    orrick_target = 0.7091 * barba
    seed_ratio = seed_abs / barba

    print("=" * 78)
    print(f"H23 attack — seed |det| = {seed_abs:,d}  ({seed_ratio:.4f} Barba)")
    print(f"          Barba(23)    = {barba:,d}")
    print(f"          Orrick 0.7091 = {orrick_target:,.0f}")
    print(f"  seed E_gram          = {seed_e_gram}")
    print(f"  seed Gram fingerprint: {{ {_fmt_gram(seed_fp)} }}")
    print("=" * 78)

    seed_matrix = Matrix(n=n, entries=[row[:] for row in seed_entries])
    results: list[dict] = []

    # --- Strategy (i): standard sa_incremental control. -------------------
    print()
    print(f"[i] sa_incremental (control, uniform entry flips, -log|det|)")
    print(f"  running with wall-clock budget ~{args.budget_seconds:.0f}s…")
    # Use a generous iteration cap (50_000_000); the wall-clock check inside
    # _sa_incremental_with_budget terminates well before that.
    t0 = time.monotonic()
    res_i = _sa_incremental_with_budget(
        seed_matrix.copy(),
        iterations=50_000_000,
        t_start=0.05,
        t_end=0.0005,
        seed=args.seed,
        wall_time_budget=args.budget_seconds,
    )
    walltime_i = time.monotonic() - t0
    best_i_det = res_i["abs_det"]
    best_i_entries = res_i["entries"]
    best_i_fp = _gram_offdiag_counter(best_i_entries)
    best_i_e_gram = gram_energy(best_i_entries)
    # Cross-verify with Bareiss.
    bareiss_i = abs(integer_determinant(best_i_entries))
    if bareiss_i != best_i_det:
        raise SystemExit(f"strategy (i) |det| audit mismatch: SA reported {best_i_det}, Bareiss says {bareiss_i}")
    print(f"  done: {walltime_i:.1f}s, {res_i['iterations']:,} iters, "
          f"{res_i['accepted']:,} accepted")
    print(f"  best |det| = {best_i_det:,d}  ({best_i_det/barba:.4f} Barba)")
    print(f"  best E_gram = {best_i_e_gram}")
    print(f"  Gram fp: {{ {_fmt_gram(best_i_fp)} }}")
    results.append({
        "strategy": "sa_incremental_control",
        "abs_det": best_i_det,
        "barba_ratio": best_i_det / barba,
        "e_gram": best_i_e_gram,
        "gram_fingerprint": best_i_fp,
        "walltime_seconds": walltime_i,
        "iterations": res_i["iterations"],
        "accepted": res_i["accepted"],
        "entries": best_i_entries,
    })

    # --- Strategy (ii): sa_gram_energy. -----------------------------------
    print()
    print("[ii] sa_gram_energy (E_gram energy, O(n) delta, uniform flips)")
    t0 = time.monotonic()
    res_ii = sa_gram_energy(
        initial=seed_matrix.copy(),
        iterations=10**9,
        t_start=20.0,
        t_end=0.5,
        seed=args.seed + 1,
        wall_time_budget=args.budget_seconds,
        correlation_every=200,
    )
    walltime_ii = time.monotonic() - t0
    bareiss_ii = abs(integer_determinant(res_ii.matrix.entries))
    assert bareiss_ii == res_ii.abs_det
    best_ii_det = res_ii.abs_det
    best_ii_e_gram = res_ii.best_e_gram
    print(f"  done: {walltime_ii:.1f}s, {res_ii.iterations:,} iters, "
          f"{res_ii.accepted:,} accepted")
    print(f"  best E_gram = {best_ii_e_gram}  (started {res_ii.starting_e_gram})")
    print(f"  |det| of best-E_gram state = {best_ii_det:,d}  "
          f"({best_ii_det/barba:.4f} Barba)")
    print(f"  Gram fp: {{ {_fmt_gram(res_ii.gram_fingerprint)} }}")
    print(f"  correlation samples collected: {len(res_ii.correlation_samples)}")
    # Compute correlation between E_gram and -log|det| over samples.
    corr_r = _correlation_e_gram_logdet(res_ii.correlation_samples)
    print(f"  Pearson correlation r(E_gram, -log|det|) = {corr_r:.4f}  "
          f"(over {len(res_ii.correlation_samples)} samples)")
    results.append({
        "strategy": "sa_gram_energy",
        "abs_det": best_ii_det,
        "barba_ratio": best_ii_det / barba,
        "e_gram": best_ii_e_gram,
        "gram_fingerprint": res_ii.gram_fingerprint,
        "walltime_seconds": walltime_ii,
        "iterations": res_ii.iterations,
        "accepted": res_ii.accepted,
        "correlation_r": corr_r,
        "n_samples": len(res_ii.correlation_samples),
        "entries": res_ii.matrix.entries,
    })

    # --- Strategy (iii): sa_targeted. -------------------------------------
    print()
    print("[iii] sa_targeted (biased +3-pair proposer, -log|det| energy)")
    t0 = time.monotonic()
    res_iii = sa_targeted(
        initial=seed_matrix.copy(),
        iterations=10**9,
        t_start=0.05,
        t_end=0.0005,
        seed=args.seed + 2,
        targeted_fraction=0.7,
        refresh_every=2000,
        wall_time_budget=args.budget_seconds,
    )
    walltime_iii = time.monotonic() - t0
    bareiss_iii = abs(integer_determinant(res_iii.matrix.entries))
    assert bareiss_iii == res_iii.abs_det
    best_iii_det = res_iii.abs_det
    best_iii_e_gram = gram_energy(res_iii.matrix.entries)
    print(f"  done: {walltime_iii:.1f}s, {res_iii.iterations:,} iters, "
          f"{res_iii.accepted:,} accepted "
          f"(targeted: {res_iii.targeted_accepted}/{res_iii.targeted_proposed}, "
          f"uniform: {res_iii.uniform_accepted}/{res_iii.uniform_proposed})")
    print(f"  best |det| = {best_iii_det:,d}  ({best_iii_det/barba:.4f} Barba)")
    print(f"  best E_gram (of best-|det| state) = {best_iii_e_gram}")
    print(f"  Gram fp: {{ {_fmt_gram(res_iii.gram_fingerprint)} }}")
    plus3_before = seed_fp.get(3, 0)
    plus3_after = res_iii.gram_fingerprint.get(3, 0)
    print(f"  +3 pair count: seed={plus3_before} → final={plus3_after}  "
          f"(change: {plus3_after - plus3_before:+d})")
    results.append({
        "strategy": "sa_targeted_plus3",
        "abs_det": best_iii_det,
        "barba_ratio": best_iii_det / barba,
        "e_gram": best_iii_e_gram,
        "gram_fingerprint": res_iii.gram_fingerprint,
        "starting_gram_fingerprint": res_iii.starting_gram_fingerprint,
        "walltime_seconds": walltime_iii,
        "iterations": res_iii.iterations,
        "accepted": res_iii.accepted,
        "targeted_proposed": res_iii.targeted_proposed,
        "targeted_accepted": res_iii.targeted_accepted,
        "uniform_proposed": res_iii.uniform_proposed,
        "uniform_accepted": res_iii.uniform_accepted,
        "entries": res_iii.matrix.entries,
    })

    # --- Final table. -----------------------------------------------------
    print()
    print("=" * 78)
    print("Summary — H23 attack")
    print("=" * 78)
    print(f"  Barba(23)               = {barba:,d}")
    print(f"  Orrick target (0.7091)  = {orrick_target:,.0f}")
    print(f"  Seed (BTC)              = {seed_abs:,d}  (0.6704 Barba)")
    print()
    header = f"{'strategy':<32} {'|det|':>22} {'ratio':>8} {'vs seed':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'seed (BTC)':<32} {seed_abs:>22,d} {seed_ratio:>8.4f} {'1.0000':>10}")
    for r in results:
        ratio = r["barba_ratio"]
        rel = r["abs_det"] / seed_abs
        print(f"{r['strategy']:<32} {r['abs_det']:>22,d} {ratio:>8.4f} {rel:>10.4f}")

    # --- Save best overall. ----------------------------------------------
    best = max(results, key=lambda r: r["abs_det"])
    if best["abs_det"] >= seed_abs:
        winner = best
    else:
        winner = {
            "strategy": "seed_held",
            "abs_det": seed_abs,
            "entries": seed_entries,
            "barba_ratio": seed_ratio,
        }
    payload = {
        "order_id": "H23",
        "n": 23,
        "abs_det": winner["abs_det"],
        "method": f"H23_attack_btc::{winner['strategy']}",
        "barba_ratio": winner["abs_det"] / barba,
        "all_results_summary": [
            {k: v for k, v in r.items() if k != "entries"} for r in results
        ],
        "entries": winner["entries"],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"Best overall ({winner['strategy']}): {winner['abs_det']:,d}")
    print(f"Wrote {args.output}")
    return 0


def _correlation_e_gram_logdet(samples: list[tuple[int, int, int]]) -> float:
    """Pearson r between E_gram and -log|det| across SA samples."""
    if len(samples) < 2:
        return float("nan")
    xs: list[float] = []
    ys: list[float] = []
    for _, e, d in samples:
        if d <= 0:
            continue
        xs.append(float(e))
        ys.append(-math.log(d))
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = (sxx * syy) ** 0.5
    if denom <= 0:
        return float("nan")
    return sxy / denom


def _sa_incremental_with_budget(
    initial: Matrix,
    iterations: int,
    *,
    t_start: float,
    t_end: float,
    seed: int,
    wall_time_budget: float,
) -> dict:
    """Wrapper around sa_incremental that polls a wall-clock budget.

    sa_incremental itself does not accept a wall-clock argument, so we
    inline a copy of the loop with the budget poll. Mirrors the same
    update rules; we just stop early when the budget is exceeded.
    """
    import random as _random

    from hadamard_maxdet.incremental import (
        IncrementalState,
        apply_entry_flip,
        propose_entry_flip,
        refresh,
    )

    n = initial.n
    rng = _random.Random(seed)
    state = IncrementalState.from_matrix(initial.entries)
    if state.det == 0:
        raise RuntimeError("seed is singular; refuse to anneal")

    current_e = -math.log(abs(state.det))
    best_entries = [row[:] for row in state.M]
    best_e = current_e

    accepted = 0
    refreshes = 0
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    start = time.monotonic()
    refresh_every = 2000

    step = 0
    for step in range(iterations):
        if (step & 1023) == 0:
            if time.monotonic() - start >= wall_time_budget:
                break
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)
        i = rng.randrange(n)
        j = rng.randrange(n)
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            continue
        proposal_e = -math.log(abs(new_det))
        delta = proposal_e - current_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            apply_entry_flip(state, i, j)
            current_e = proposal_e
            accepted += 1
            if current_e < best_e:
                best_e = current_e
                best_entries = [row[:] for row in state.M]
            if refresh_every > 0 and accepted % refresh_every == 0:
                refresh(state)
                refreshes += 1
                if state.det != 0:
                    current_e = -math.log(abs(state.det))

    best_abs_det = abs(integer_determinant(best_entries))
    return {
        "iterations": step + 1,
        "accepted": accepted,
        "abs_det": best_abs_det,
        "entries": best_entries,
        "refreshes": refreshes,
    }


if __name__ == "__main__":
    raise SystemExit(main())
